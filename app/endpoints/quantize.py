"""Assignment 5: Stateful two-phase quantized-candidate admission API."""

import math

from app.core.errors import Conflict, InvalidInput
from app.core.hashing import cj, is_non_negative_safe_int, round12_ratio, sha256_hex
from app.core.ordering import reason_codes, utf8_key
from app.core.persistence import get_store
from app.core.validation import valid_floor

NS_FREEZE = "quantize_freezes"


# ----------------------------- freeze ---------------------------------------


def _compute_inventory(files) -> "tuple[list[dict], int] | None":
    """Return (inventory sorted by filename, totalBytes) or None when invalid."""
    if not isinstance(files, dict) or len(files) == 0:
        return None
    entries = []
    for name, content in files.items():
        if not isinstance(name, str) or name == "" or not isinstance(content, str):
            return None
        data = content.encode("utf-8")
        entries.append(
            {"name": name, "bytes": len(data), "sha256": sha256_hex(data)}
        )
    entries.sort(key=lambda e: utf8_key(e["name"]))
    total = sum(e["bytes"] for e in entries)
    return entries, total


def _candidate_status(cand, request) -> "tuple[str, list[str]]":
    codes = set()
    if not isinstance(cand, dict):
        return "invalid", ["INVALID_INPUT"]

    reason = cand.get("unsupportedReason")
    has_reason = isinstance(reason, str) and reason != ""
    if isinstance(reason, str) and reason == "":
        has_reason = False
    if reason is not None and not isinstance(reason, str):
        return "invalid", ["INVALID_INPUT"]

    allowed = request.get("allowedUnsupportedReasons")
    allowed_set = set(allowed) if isinstance(allowed, list) else set()

    if has_reason:
        if reason in allowed_set:
            return "unsupported", []
        return "invalid", ["UNALLOWED_UNSUPPORTED_REASON"]

    loadable = cand.get("loadable")
    if loadable is not True:
        codes.add("NOT_LOADABLE")
    calib = cand.get("calibrationDigest")
    tok = cand.get("tokenizerDigest")
    if calib != request.get("calibrationDigest"):
        codes.add("CALIBRATION_MISMATCH")
    if tok != request.get("tokenizerDigest"):
        codes.add("TOKENIZER_MISMATCH")
    if calib is None or tok is None or not isinstance(calib, str) or not isinstance(tok, str) or calib == "" or tok == "":
        codes.add("INVALID_INPUT")
    status = "frozen" if not codes else "invalid"
    return status, sorted(codes)


def _do_freeze(body: dict) -> dict:
    freeze_id = body.get("freezeId")
    candidates = body.get("candidates")

    # Structural (request-level) validation -> HTTP 400.
    if not isinstance(freeze_id, str) or not (0 < len(freeze_id) <= 128):
        raise InvalidInput()
    for k in ("calibrationDigest", "tokenizerDigest"):
        v = body.get(k)
        if not isinstance(v, str) or v == "":
            raise InvalidInput()
    allowed = body.get("allowedUnsupportedReasons")
    if (
        not isinstance(allowed, list)
        or any(not isinstance(x, str) or x == "" for x in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise InvalidInput()
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise InvalidInput()

    out_candidates = []
    seen_names = {}
    for cand in candidates:
        name = cand.get("name") if isinstance(cand, dict) else None
        if not isinstance(name, str) or name == "":
            continue  # unnamed candidate cannot be represented; skipped
        inventory = _compute_inventory(cand.get("files"))
        status, codes = _candidate_status(cand, body)
        if name in seen_names:
            seen_names[name]["reasonCodes"] = reason_codes(
                seen_names[name]["reasonCodes"] + ["INVALID_INPUT"] + codes
            )
            seen_names[name]["status"] = (
                "invalid" if seen_names[name]["status"] != "invalid" else seen_names[name]["status"]
            )
            continue
        entry = {
            "name": name,
            "status": status,
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": reason_codes(codes),
        }
        if inventory is not None:
            entries, total = inventory
            entry["inventory"] = entries
            entry["totalBytes"] = total
            entry["packageDigest"] = sha256_hex(cj(entries).encode("utf-8"))
        else:
            entry["status"] = "invalid"
            entry["reasonCodes"] = reason_codes(entry["reasonCodes"] + ["INVALID_INPUT"])
        seen_names[name] = entry
        out_candidates.append(entry)

    out_candidates.sort(key=lambda e: utf8_key(e["name"]))
    response = {"freezeId": freeze_id, "candidates": out_candidates}
    return response


def _freeze_equal(a, b) -> bool:
    return a == b
# ----------------------------- select ---------------------------------------


def _predictions_binary(rows, names) -> bool:
    for row in rows:
        preds = row.get("predictions")
        if not isinstance(preds, dict):
            return False
        for n in names:
            v = preds.get(n)
            if not (isinstance(v, int) and not isinstance(v, bool) and v in (0, 1)):
                return False
    return True


def _do_select(body: dict, stored) -> dict:
    freeze_id = body.get("freezeId")
    supplied_candidates = body.get("candidates")
    policy = body.get("policy")
    latencies = body.get("latencies")
    rows = body.get("rows")

    # Request-level structural checks already done by caller (arrays + policy object).
    results = []

    frozen_response = stored["response"] if isinstance(stored, dict) and "response" in stored else stored
    stored_map = {c["name"]: c for c in frozen_response["candidates"]} if frozen_response else {}

    # ---- policy validation ----
    policy_codes = set()
    max_bytes = policy.get("maxBytes") if isinstance(policy, dict) else None
    aggregate_floor = policy.get("aggregateFloor") if isinstance(policy, dict) else None
    required_slices = policy.get("requiredSlices") if isinstance(policy, dict) else None
    max_latency = policy.get("maxLatencyMs") if isinstance(policy, dict) else None
    order = policy.get("candidateOrder") if isinstance(policy, dict) else None

    if not is_non_negative_safe_int(max_bytes):
        policy_codes.add("INVALID_POLICY")
    if not valid_floor(aggregate_floor):
        policy_codes.add("INVALID_POLICY")
    slices_ok = isinstance(required_slices, dict)
    if slices_ok:
        for k, v in required_slices.items():
            if not isinstance(k, str) or k == "" or not valid_floor(v):
                slices_ok = False
                break
    else:
        policy_codes.add("INVALID_POLICY")
    required_slices = required_slices if slices_ok else {}
    if not (
        isinstance(max_latency, (int, float))
        and not isinstance(max_latency, bool)
        and math.isfinite(max_latency)
        and max_latency >= 0
    ):
        policy_codes.add("INVALID_POLICY")
    lat_ok = isinstance(latencies, dict)
    if lat_ok:
        for k, v in latencies.items():
            if (
                not isinstance(k, str)
                or not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(v)
                or v < 0
            ):
                lat_ok = False
                break
    else:
        policy_codes.add("INVALID_POLICY")
    order_ok = (
        isinstance(order, list)
        and len(order) > 0
        and all(isinstance(x, str) and x != "" for x in order)
        and len(set(order)) == len(order)
    )
    if not order_ok:
        policy_codes.add("INVALID_POLICY")

    names = []
    for c in supplied_candidates:
        n = c.get("name") if isinstance(c, dict) else None
        if isinstance(n, str) and n not in names:
            names.append(n)

    if order_ok and set(order) != set(names):
        policy_codes.add("INVALID_POLICY")

    frozen_request = stored.get("request") if isinstance(stored, dict) else None

    # ---- per-candidate evaluation ----
    row_level_ok = (
        isinstance(rows, list)
        and len(rows) > 0
        and all(
            isinstance(r, dict)
            and isinstance(r.get("label"), int)
            and not isinstance(r.get("label"), bool)
            and r.get("label") in (0, 1)
            and isinstance(r.get("slice"), str)
            and r.get("slice") != ""
            for r in rows
        )
    )

    for name in names:
        codes = set()
        supplied = next((c for c in supplied_candidates if isinstance(c, dict) and c.get("name") == name), {})
        rec = stored_map.get(name)
        frozen_req_map = {}
        if isinstance(frozen_request, dict) and isinstance(frozen_request.get("candidates"), list):
            for c in frozen_request["candidates"]:
                if isinstance(c, dict) and isinstance(c.get("name"), str):
                    frozen_req_map[c["name"]] = c

        manifest_ok = True
        lineage_ok = True
        frozen_status = True

        if stored is None:
            codes.add("NOT_FROZEN")
            frozen_status = False
            manifest_ok = False
            lineage_ok = False
        else:
            if supplied != rec:
                codes.add("INVALID_MANIFEST")
                manifest_ok = False
            if rec is None or rec.get("status") != "frozen":
                codes.add("NOT_FROZEN")
                frozen_status = False
                manifest_ok = False
                lineage_ok = False
            else:
                stored_cand = frozen_req_map.get(name)
                if (
                    not isinstance(stored_cand, dict)
                    or stored_cand.get("calibrationDigest") != frozen_request.get("calibrationDigest")
                    or stored_cand.get("tokenizerDigest") != frozen_request.get("tokenizerDigest")
                ):
                    codes.add("INVALID_LINEAGE")
                    lineage_ok = False
                    manifest_ok = False

        # Recompute the manifest from the recorded freeze-request files.
        total_bytes = None
        package_digest = None
        files = frozen_req_map.get(name, {}).get("files") if stored is not None else None
        inv = _compute_inventory(files)
        if inv is None:
            if stored is not None:
                codes.add("INVALID_MANIFEST")
                manifest_ok = False
        else:
            entries, total_bytes = inv
            package_digest = sha256_hex(cj(entries).encode("utf-8"))
            if stored is not None and rec is not None:
                if (
                    rec.get("totalBytes") != total_bytes
                    or rec.get("packageDigest") != package_digest
                    or rec.get("inventory") != entries
                ):
                    codes.add("INVALID_MANIFEST")
                    manifest_ok = False

        latency_value = latencies.get(name) if isinstance(latencies, dict) else None
        latency_out = latency_value
        if latency_value is None:
            codes.add("LATENCY_LIMIT")
        elif isinstance(max_latency, (int, float)) and not isinstance(max_latency, bool):
            if float(latency_value) > float(max_latency):
                codes.add("LATENCY_LIMIT")
        else:
            codes.add("LATENCY_LIMIT")

        aggregate = None
        slices_acc = {}
        preds_ok = row_level_ok and _predictions_binary(rows, [name])
        if not row_level_ok:
            codes.add("INVALID_PREDICTIONS")
        if not preds_ok:
            if row_level_ok:
                codes.add("INVALID_PREDICTIONS")
        else:
            correct = sum(1 for r in rows if r["predictions"][name] == r["label"])
            aggregate = round12_ratio(correct, len(rows))
            by_slice = {}
            for r in rows:
                by_slice.setdefault(r["slice"], []).append(r)
            missing = False
            failed = False
            for sname, floor in required_slices.items():
                if sname not in by_slice:
                    missing = True
                    codes.add(f"MISSING_SLICE:{sname}")
                    continue
                srows = by_slice[sname]
                sacc = round12_ratio(
                    sum(1 for r in srows if r["predictions"][name] == r["label"]),
                    len(srows),
                )
                slices_acc[sname] = sacc
                if sacc < float(floor):
                    failed = True
                    codes.add(f"SLICE_FLOOR:{sname}")
            size_ok = False
            if total_bytes is not None and isinstance(max_bytes, int) and not isinstance(max_bytes, bool):
                if total_bytes <= max_bytes:
                    size_ok = True
                else:
                    codes.add("SIZE_LIMIT")
            else:
                codes.add("SIZE_LIMIT")
            if valid_floor(aggregate_floor) and aggregate < float(aggregate_floor):
                codes.add("AGGREGATE_FLOOR")
            admitted = bool(
                frozen_status
                and lineage_ok
                and manifest_ok
                and preds_ok
                and not missing
                and not failed
                and valid_floor(aggregate_floor)
                and aggregate >= float(aggregate_floor)
                and size_ok
                and latency_out is not None
                and isinstance(max_latency, (int, float))
                and not isinstance(max_latency, bool)
                and float(latency_out) <= float(max_latency)
                and "INVALID_POLICY" not in policy_codes
            )
            result_entry = {
                "name": name,
                "aggregate": aggregate,
                "slices": slices_acc,
                "totalBytes": total_bytes,
                "latencyMs": latency_out,
                "admitted": admitted,
                "reasonCodes": sorted(codes),
            }
            results.append((result_entry, order.index(name) if order_ok and name in order else None))
            continue

        # predictions-invalid path still emits a result row
        result_entry = {
            "name": name,
            "aggregate": aggregate,
            "slices": slices_acc,
            "totalBytes": total_bytes,
            "latencyMs": latency_out,
            "admitted": False,
            "reasonCodes": sorted(codes),
        }
        results.append((result_entry, order.index(name) if order_ok and name in order else None))

    # finalize ordering + codes
    def sort_key(item):
        entry, idx = item
        return (idx if idx is not None else 10 ** 9, utf8_key(entry["name"]))

    results.sort(key=sort_key)
    out_results = []
    for entry, _ in results:
        entry["reasonCodes"] = reason_codes(entry["reasonCodes"])
        out_results.append(entry)

    # attach global codes to every result
    global_codes = set(policy_codes)
    if stored is None:
        global_codes.add("NOT_FROZEN")
    for entry in out_results:
        entry["reasonCodes"] = reason_codes(entry["reasonCodes"] + sorted(global_codes))

    # selection among admitted
    admitted_entries = [e for e in out_results if e["admitted"]]

    selected = None
    package_manifest = None
    if admitted_entries:
        def admit_sort(e):
            tb = e["totalBytes"] if e["totalBytes"] is not None else math.inf
            lat = float(e["latencyMs"]) if e["latencyMs"] is not None else math.inf
            return (tb, lat, out_results.index(e))

        winner = min(admitted_entries, key=admit_sort)
        selected = winner["name"]
        if stored is not None:
            rec = stored_map.get(selected)
            package_manifest = rec if rec is not None else winner
        else:
            package_manifest = None

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": out_results,
        "packageManifest": package_manifest,
    }


def handle(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    phase = body.get("phase")
    store = get_store()
    if phase == "freeze":
        # Replay/conflict handling.
        freeze_id = body.get("freezeId")
        if isinstance(freeze_id, str) and 0 < len(freeze_id) <= 128:
            existing = store.get(NS_FREEZE, freeze_id)
            if existing is not None:
                if existing["request"] == body:
                    return existing["response"]
                raise Conflict("FREEZE_ID_CONFLICT")
        response = _do_freeze(body)  # raises InvalidInput on structural problems
        if isinstance(body.get("freezeId"), str):
            store.set(NS_FREEZE, body["freezeId"], {"request": body, "response": response})
        return response
    if phase == "select":
        candidates = body.get("candidates")
        rows = body.get("rows")
        policy = body.get("policy")
        if not isinstance(candidates, list) or not isinstance(rows, list) or not isinstance(policy, dict):
            raise InvalidInput()
        freeze_id = body.get("freezeId")
        stored = store.get(NS_FREEZE, freeze_id) if isinstance(freeze_id, str) else None
        return _do_select(body, stored)
    raise InvalidInput()


__all__ = ["handle", "_freeze_equal"]
