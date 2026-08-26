"""Assignment 6: Content-addressed ML pipeline controller (session-scoped state)."""

import copy

from app.core.errors import Conflict
from app.core.hashing import cj, sha256_hex
from app.core.persistence import get_store

NS_SESSIONS = "pipeline_sessions"

NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}
RECEIPT_NODES = {"register", "publish"}
STATUSES = {"started", "succeeded", "retryable_failed", "terminal_failed"}

INPUT_FIELDS = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

RECIPES = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig"],
    "train": ["@prepare", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["@train", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["@evaluate", "schemaDigest"],
    "publish": ["@register", "publishConfig"],
}

EVENT_FIELDS = {
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
}


def _new_session(revision: int, inputs: dict) -> dict:
    return {
        "revision": revision,
        "inputs": copy.deepcopy(inputs),
        "cache": {},        # node -> {key -> {"artifactDigest","eventId"}}
        "node_state": {},   # node -> state dict
        "events": {},       # eventId -> canonical json
    }


def _artifact_for(session: dict, node: str, keys: dict):
    k = keys[node]
    if k is None:
        return None
    binding = session["cache"].get(node, {}).get(k)
    return binding["artifactDigest"] if binding else None


def _compute_keys(session: dict) -> dict:
    keys = {}
    for node in NODES:
        parts = []
        ok = True
        for field in RECIPES[node]:
            if field.startswith("@"):
                parent = field[1:]
                art = _artifact_for(session, parent, keys)
                if art is None:
                    ok = False
                    break
                parts.append(art)
            else:
                parts.append(session["inputs"][field])
        keys[node] = sha256_hex(cj(parts).encode("utf-8")) if ok else None
    return keys


def _validate_request(body) -> dict:
    if not isinstance(body, dict):
        raise Conflict("INVALID_REQUEST")
    session_id = body.get("session")
    if not isinstance(session_id, str) or session_id == "":
        raise Conflict("INVALID_REQUEST")
    revision = body.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1 or revision > 9007199254740991:
        raise Conflict("INVALID_REQUEST")
    inputs = body.get("inputs")
    if not isinstance(inputs, dict):
        raise Conflict("INVALID_REQUEST")
    for f in INPUT_FIELDS:
        v = inputs.get(f)
        if not isinstance(v, str) or v == "":
            raise Conflict("INVALID_REQUEST")
    events = body.get("events")
    if not isinstance(events, list):
        raise Conflict("INVALID_REQUEST")
    return session_id


def _validate_event(e) -> None:
    """Structural validation; raises INVALID_EVENT (whole batch rolls back)."""
    if not isinstance(e, dict) or set(e.keys()) != EVENT_FIELDS:
        raise Conflict("INVALID_EVENT")
    if not isinstance(e["eventId"], str) or e["eventId"] == "":
        raise Conflict("INVALID_EVENT")
    rev = e["revision"]
    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 1 or rev > 9007199254740991:
        raise Conflict("INVALID_EVENT")
    if not isinstance(e["node"], str):
        raise Conflict("INVALID_EVENT")
    att = e["attempt"]
    if not isinstance(att, int) or isinstance(att, bool) or att < 1 or att > 9007199254740991:
        raise Conflict("INVALID_EVENT")
    if not isinstance(e["status"], str):
        raise Conflict("INVALID_EVENT")
    if not isinstance(e["key"], str) or e["key"] == "":
        raise Conflict("INVALID_EVENT")
    if e["artifactDigest"] is not None and not isinstance(e["artifactDigest"], str):
        raise Conflict("INVALID_EVENT")
    if e["receiptId"] is not None and not isinstance(e["receiptId"], str):
        raise Conflict("INVALID_EVENT")


def _apply_event(session: dict, e: dict, keys: dict, accepted: list, ignored: list) -> dict:
    """Apply one event to the working copy. Returns updated working keys."""
    eid = e["eventId"]
    canon_e = cj(e)

    if eid in session["events"]:
        if session["events"][eid] == canon_e:
            ignored.append(eid)
            return keys
        raise Conflict("EVENT_ID_CONFLICT")

    if e["revision"] != session["revision"]:
        ignored.append(eid)
        return keys

    node = e["node"]
    if node not in NODES:
        ignored.append(eid)
        return keys

    current_key = keys[node]
    if current_key is None or e["key"] != current_key:
        ignored.append(eid)
        return keys

    status = e["status"]
    if status not in STATUSES:
        ignored.append(eid)
        return keys

    artifact = e["artifactDigest"]
    receipt = e["receiptId"]

    if status == "succeeded":
        if not isinstance(artifact, str) or artifact == "":
            ignored.append(eid)
            return keys
        if node in RECEIPT_NODES:
            expected_receipt = f"receipt:{node}:{e['key']}"
            if receipt != expected_receipt:
                ignored.append(eid)
                return keys
        else:
            if receipt is not None:
                ignored.append(eid)
                return keys
    else:
        if artifact is not None or receipt is not None:
            ignored.append(eid)
            return keys

    st = session["node_state"].get(node)
    attempt = e["attempt"]

    def bind_success():
        node_cache = session["cache"].setdefault(node, {})
        existing = node_cache.get(current_key)
        if existing is not None:
            if existing["artifactDigest"] != artifact:
                raise Conflict("EVIDENCE_CONFLICT")
            # Original binding stays immutable.
        else:
            node_cache[current_key] = {"artifactDigest": artifact, "eventId": eid}
        session["node_state"][node] = {
            "status": "succeeded",
            "attempt": attempt,
            "key": current_key,
            "event_id": eid,
        }

    if st is None:
        if status == "started" and attempt == 1:
            session["node_state"][node] = {
                "status": "started",
                "attempt": 1,
                "key": current_key,
                "start_event_id": eid,
            }
        else:
            ignored.append(eid)
            return keys
    elif st["key"] != current_key:
        # Stale state for an older key; treat as no state.
        if status == "started" and attempt == 1:
            session["node_state"][node] = {
                "status": "started",
                "attempt": 1,
                "key": current_key,
                "start_event_id": eid,
            }
        else:
            ignored.append(eid)
            return keys
    else:
        s_status = st["status"]
        s_attempt = st["attempt"]
        if s_status == "terminal_failed":
            raise Conflict("STATUS_CONFLICT")
        if s_status == "succeeded":
            if status == "succeeded":
                binding = session["cache"][node][current_key]
                if binding["artifactDigest"] != artifact:
                    raise Conflict("EVIDENCE_CONFLICT")
            raise Conflict("STATUS_CONFLICT")
        if attempt < s_attempt:
            ignored.append(eid)
            return keys
        if s_status == "started":
            if attempt != s_attempt:
                raise Conflict("STATUS_CONFLICT")
            if status == "started":
                raise Conflict("STATUS_CONFLICT")
            if status == "succeeded":
                bind_success()
            elif status == "retryable_failed":
                session["node_state"][node] = {
                    "status": "retryable_failed",
                    "attempt": attempt,
                    "key": current_key,
                    "fail_event_id": eid,
                }
            else:  # terminal_failed
                session["node_state"][node] = {
                    "status": "terminal_failed",
                    "attempt": attempt,
                    "key": current_key,
                    "fail_event_id": eid,
                }
        elif s_status == "retryable_failed":
            if status == "started" and attempt == s_attempt + 1:
                session["node_state"][node] = {
                    "status": "started",
                    "attempt": attempt,
                    "key": current_key,
                    "start_event_id": eid,
                }
            else:
                raise Conflict("STATUS_CONFLICT")

    session["events"][eid] = canon_e
    accepted.append(eid)
    keys = _compute_keys(session)
    return keys


def handle(body) -> dict:
    session_id = _validate_request(body)
    store = get_store()
    sessions = store.bucket(NS_SESSIONS)
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    with store._lock:
        session = sessions.get(session_id)
        if session is None:
            session = _new_session(revision, inputs)
        else:
            if session["revision"] == revision:
                if session["inputs"] != inputs:
                    raise Conflict("REVISION_CONFLICT")
            else:
                session = {
                    "revision": revision,
                    "inputs": copy.deepcopy(inputs),
                    "cache": session["cache"],
                    "node_state": {},
                    "events": session["events"],
                }

        work = copy.deepcopy(session)
        keys = _compute_keys(work)
        accepted = []
        ignored = []

        for e in events:
            _validate_event(e)
        for e in events:
            keys = _apply_event(work, e, keys, accepted, ignored)

        sessions[session_id] = work

    final_keys = _compute_keys(work)
    nodes_out = []
    for idx, node in enumerate(NODES):
        dep = {}
        for field in RECIPES[node]:
            if field.startswith("@"):
                dep[field[1:] + "Artifact"] = _artifact_for(work, field[1:], final_keys)
            else:
                dep[field] = work["inputs"][field]
        dep["cacheKey"] = final_keys[node]

        # Determine upstream condition.
        upstream_terminal = False
        upstream_pending = False
        parent = PARENT[node]
        while parent is not None:
            pk = final_keys[parent]
            binding = work["cache"].get(parent, {}).get(pk) if pk is not None else None
            st = work["node_state"].get(parent)
            if binding is None:
                if st is not None and st.get("status") == "terminal_failed" and st.get("key") == pk:
                    upstream_terminal = True
                else:
                    upstream_pending = True
            parent = PARENT[parent]

        k = final_keys[node]
        cached = work["cache"].get(node, {}).get(k) if k is not None else None
        st = work["node_state"].get(node)
        if st is not None and st.get("key") != k:
            st = None

        if upstream_terminal:
            entry = ("block", ["UPSTREAM_TERMINAL"], [])
        elif upstream_pending or k is None:
            entry = ("block", ["UPSTREAM_PENDING"], [])
        elif cached is not None:
            entry = ("reuse", ["CACHE_HIT"], [cached["eventId"]])
        elif st is not None and st["status"] == "started":
            entry = ("block", ["RUNNING"], [st.get("start_event_id")])
        elif st is not None and st["status"] == "terminal_failed":
            entry = ("block", ["TERMINAL_FAILURE"], [st.get("fail_event_id")])
        elif st is not None and st["status"] == "retryable_failed":
            entry = ("rerun", ["RETRYABLE_FAILURE"], [st.get("fail_event_id")])
        else:
            entry = ("rerun", ["CACHE_MISS"], [])

        nodes_out.append(
            {
                "node": node,
                "action": entry[0],
                "reasonCodes": entry[1],
                "dependencyDigests": dep,
                "triggeringEventIds": entry[2],
            }
        )

    return {
        "revision": work["revision"],
        "acceptedEventIds": accepted,
        "ignoredEventIds": ignored,
        "nodes": nodes_out,
    }
