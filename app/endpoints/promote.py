"""Assignment 3: Deterministic model-registry promotion gate."""

import hashlib
import math
import re
from datetime import timedelta

from app.core.errors import Conflict, InvalidInput
from app.core.hashing import cj, round12
from app.core.ordering import reason_codes
from app.core.persistence import get_store
from app.core.timestamps import parse_timestamp
from app.core.validation import valid_floor

NS_MUTATIONS = "promote_mutations"

CANONICAL_VERSION_RE = re.compile(r"^[1-9][0-9]*$")
MAX_SAFE = 9007199254740991


def _canonical_version(v) -> bool:
    return (
        isinstance(v, str)
        and CANONICAL_VERSION_RE.match(v) is not None
        and int(v) <= MAX_SAFE
    )


def _policy_invalid(policy, as_of) -> bool:
    if parse_timestamp(as_of) is None:
        return True
    for k in ("datasetDigest", "schemaDigest"):
        v = policy.get(k)
        if not isinstance(v, str) or v == "":
            return True
    age = policy.get("maxAgeSeconds")
    if not isinstance(age, int) or isinstance(age, bool) or age < 0 or age > MAX_SAFE:
        return True
    if not valid_floor(policy.get("accuracyFloor")):
        return True
    if not valid_floor(policy.get("minImprovement")):
        return True
    lat = policy.get("maxLatencyMs")
    if not isinstance(lat, (int, float)) or isinstance(lat, bool):
        return True
    if not math.isfinite(lat) or lat < 0:
        return True
    size_limit = policy.get("maxSizeBytes")
    if not isinstance(size_limit, int) or isinstance(size_limit, bool) or size_limit < 0 or size_limit > MAX_SAFE:
        return True
    slices = policy.get("requiredSlices")
    if not isinstance(slices, dict):
        return True
    for k, v in slices.items():
        if not isinstance(k, str) or k == "" or not valid_floor(v):
            return True
    return False


def _finite_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _gate_version(entry, policy, as_of_dt) -> set:
    codes = set()
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, dict):
        return {"MISSING_EVALUATION"}

    created = parse_timestamp(evaluation.get("createdAt"))
    if created is None:
        codes.add("INVALID_TIMESTAMP")
    else:
        lower = as_of_dt - timedelta(seconds=int(policy["maxAgeSeconds"]))
        if created > as_of_dt:
            codes.add("FUTURE_EVALUATION")
        elif created < lower:
            codes.add("STALE_EVALUATION")

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    if entry.get("artifactDigest") != evaluation.get("artifactDigest"):
        codes.add("ARTIFACT_MISMATCH")
    if evaluation.get("datasetDigest") != policy["datasetDigest"]:
        codes.add("DATASET_MISMATCH")
    if evaluation.get("schemaDigest") != policy["schemaDigest"]:
        codes.add("SCHEMA_MISMATCH")

    values_finite = all(_finite_num(x) for x in (accuracy, latency, size))
    if not values_finite:
        codes.add("NON_FINITE")
    else:
        acc_f = float(accuracy)
        lat_f = float(latency)
        size_f = float(size)
        if not (0.0 <= acc_f <= 1.0):
            codes.add("METRIC_RANGE")
        if acc_f < float(policy["accuracyFloor"]):
            codes.add("ACCURACY_FLOOR")
        if lat_f > float(policy["maxLatencyMs"]):
            codes.add("LATENCY_LIMIT")
        if size_f > float(policy["maxSizeBytes"]):
            codes.add("SIZE_LIMIT")
        slices = evaluation.get("slices")
        slices_dict = slices if isinstance(slices, dict) else {}
        for name, floor in policy["requiredSlices"].items():
            if name not in slices_dict:
                codes.add(f"MISSING_SLICE:{name}")
                continue
            val = slices_dict[name]
            if not _finite_num(val) or not (0.0 <= float(val) <= 1.0):
                codes.add(f"SLICE_RANGE:{name}")
            elif float(val) < float(floor):
                codes.add(f"SLICE_FLOOR:{name}")
    return codes


def handle(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    policy = body.get("policy")
    versions = body.get("versions")
    champion_version = body.get("championVersion")
    as_of = body.get("asOf")

    if not isinstance(policy, dict):
        raise InvalidInput()
    if not isinstance(versions, list):
        raise InvalidInput()
    if not isinstance(champion_version, str):
        raise InvalidInput()

    store = get_store()
    fingerprint = hashlib.sha256(cj(body).encode("utf-8")).hexdigest()

    global_policy_invalid = _policy_invalid(policy, as_of)
    as_of_dt = parse_timestamp(as_of)

    parsed = []
    for entry in versions:
        v_raw = entry.get("version") if isinstance(entry, dict) else None
        parsed.append((entry, v_raw))

    pre_codes = {}  # version string -> structural codes
    valid_unique = []
    seen = set()
    for entry, v_raw in parsed:
        if not _canonical_version(v_raw):
            if isinstance(v_raw, str):
                pre_codes.setdefault(v_raw, set()).add("INVALID_VERSION")
            continue
        if v_raw in seen:
            pre_codes.setdefault(v_raw, set()).add("DUPLICATE_VERSION")
            continue
        seen.add(v_raw)
        valid_unique.append((entry, v_raw))

    failed_gates = {}
    eligible_entries = []
    for entry, v_raw in valid_unique:
        codes = set(pre_codes.get(v_raw, set()))
        if global_policy_invalid:
            codes.add("INVALID_POLICY")
        else:
            codes |= _gate_version(entry, policy, as_of_dt)
        failed_gates[v_raw] = codes
        if not codes:
            eligible_entries.append((entry, v_raw))

    def rank_key(item):
        entry, v_raw = item
        ev = entry["evaluation"]
        return (
            -float(ev["accuracy"]),
            float(ev["latencyMs"]),
            float(ev["sizeBytes"]),
            int(v_raw),
        )

    ranked = sorted(eligible_entries, key=rank_key)
    eligible_ids = [v for _, v in ranked]

    listed_versions = {v for _, v in valid_unique}
    champion_eligible = champion_version in listed_versions and champion_version in eligible_ids

    selected_id = None
    evidence = None
    alias_mutation = None

    if not champion_eligible:
        action = "block"
    else:
        champion_entry = next(e for e, v in valid_unique if v == champion_version)
        champion_eval = champion_entry["evaluation"]
        best_entry, best_id = ranked[0]
        improvement = round12(
            float(best_entry["evaluation"]["accuracy"]) - float(champion_eval["accuracy"])
        )
        prior = store.get(NS_MUTATIONS, fingerprint)
        if improvement >= float(policy["minImprovement"]):
            target_id = best_id if prior is None else prior
            if prior is None:
                store.set(NS_MUTATIONS, fingerprint, target_id)
                action = "promote"
                selected_id = target_id
                evidence = best_entry["evaluation"]
                alias_mutation = {"alias": "champion", "version": target_id}
            else:
                # Idempotent replay: alias already mutated; retain it.
                action = "retain"
                selected_id = target_id
                evidence = _evaluation_for(valid_unique, target_id)
        else:
            action = "retain"
            selected_id = champion_version
            evidence = champion_eval

    failed_out = {}
    for _, v_raw in parsed:
        if not isinstance(v_raw, str):
            continue
        if v_raw not in failed_out:
            codes = set(pre_codes.get(v_raw, set())) | set(failed_gates.get(v_raw, set()))
            failed_out[v_raw] = reason_codes(codes)

    return {
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected_id,
        "eligibleVersions": eligible_ids,
        "failedGates": failed_out,
        "aliasMutation": alias_mutation,
        "evidence": evidence,
    }


def _evaluation_for(valid_unique, version_id):
    for entry, v in valid_unique:
        if v == version_id:
            ev = entry.get("evaluation")
            return ev if isinstance(ev, dict) else None
    return None


__all__ = ["handle", "Conflict"]
