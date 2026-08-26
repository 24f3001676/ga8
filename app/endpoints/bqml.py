"""Assignment 2: Stateful two-phase BigQuery ML experiment gate."""

import math

from app.core.errors import InvalidInput
from app.core.hashing import (
    is_safe_int,
    round12_ratio,
    sha256_of_json,
)
from app.core.ordering import reason_codes, utf8_key
from app.core.persistence import get_store
from app.core.timestamps import parse_timestamp
from app.core.validation import is_hex64, valid_floor

NS = "bqml_selections"


def _valid_run_id(v) -> bool:
    return isinstance(v, str) and 0 < len(v) <= 128


def _feature_map_ok(features) -> bool:
    if not isinstance(features, dict):
        return False
    for name, spec in features.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            return False
        if "availableAt" not in spec or parse_timestamp(spec.get("availableAt")) is None:
            return False
    return True


def _row_ok(row) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("id"), str):
        return False
    if not isinstance(row.get("entity"), str):
        return False
    if parse_timestamp(row.get("eventTime")) is None:
        return False
    if parse_timestamp(row.get("predictionTime")) is None:
        return False
    v = row.get("version")
    if not is_safe_int(v) or v < 0:
        return False
    if row.get("split") not in ("TRAIN", "EVAL"):
        return False
    return _feature_map_ok(row.get("features"))


def _trial_metric_finite(trial) -> bool:
    m = trial.get("evalMetric")
    return isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m)


def _select_response(run_id, trial_id, train_ids, eval_ids, features, digest, codes):
    return {
        "runId": run_id,
        "selectedTrialId": trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
        "datasetDigest": digest,
        "reasonCodes": reason_codes(codes),
    }


def _do_select(body: dict) -> dict:
    run_id = body.get("runId")
    rows = body.get("rows")
    trials = body.get("trials")
    forbidden = body.get("forbiddenFeatures")
    limit = body.get("numTrialsLimit")

    malformed = False
    if not _valid_run_id(run_id):
        malformed = True
    if not isinstance(rows, list) or len(rows) == 0:
        malformed = True
        rows = []
    else:
        ids = [r.get("id") for r in rows if isinstance(r, dict)]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            malformed = True
        if any(not _row_ok(r) for r in rows):
            malformed = True
    if not isinstance(trials, list):
        malformed = True
        trials = []
    else:
        tids = [t.get("trialId") for t in trials if isinstance(t, dict)]
        if any(not is_safe_int(t) or t < 0 for t in tids) or len(set(tids)) != len(tids):
            malformed = True
        if any(
            not isinstance(t, dict) or t.get("status") not in ("SUCCEEDED", "FAILED")
            for t in trials
        ):
            malformed = True
    if not isinstance(forbidden, list) or any(not isinstance(f, str) for f in forbidden):
        malformed = True
    if not is_safe_int(limit) or limit < 1:
        malformed = True

    if malformed:
        return _select_response(run_id, None, [], [], [], None, ["INVALID_INPUT"])

    # Deduplicate rows by [entity, UTC(eventTime)].
    best = {}
    for r in rows:
        key = (r["entity"], parse_timestamp(r["eventTime"]))
        cur = best.get(key)
        if cur is None or (r["version"] > cur["version"]) or (
            r["version"] == cur["version"]
            and utf8_key(r["id"]) < utf8_key(cur["id"])
        ):
            best[key] = r

    train_rows = [r for r in best.values() if r["split"] == "TRAIN"]
    eval_rows = [r for r in best.values() if r["split"] == "EVAL"]

    feature_names = None
    if best:
        iters = [set(r["features"].keys()) for r in best.values()]
        common = set.intersection(*iters)
        forbidden_set = set(forbidden)
        eligible = []
        for name in common:
            if name in forbidden_set:
                continue
            ok = all(
                parse_timestamp(r["features"][name]["availableAt"])
                <= parse_timestamp(r["predictionTime"])
                for r in best.values()
            )
            if ok:
                eligible.append(name)
        feature_names = sorted(eligible, key=utf8_key)

    train_ids = sorted((r["id"] for r in train_rows), key=utf8_key)
    eval_ids = sorted((r["id"] for r in eval_rows), key=utf8_key)
    digest = sha256_of_json(
        {"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": feature_names}
    )

    codes = []
    selected = None
    if len(trials) > limit:
        codes.append("TRIAL_LIMIT_EXCEEDED")
    else:
        eligible_trials = [
            t for t in trials if t.get("status") == "SUCCEEDED" and _trial_metric_finite(t)
        ]
        if not eligible_trials:
            codes.append("NO_SUCCESSFUL_TRIAL")
        else:
            best_trial = min(
                eligible_trials,
                key=lambda t: (-float(t["evalMetric"]), t["trialId"]),
            )
            selected = best_trial["trialId"]

    return _select_response(
        run_id, selected, train_ids, eval_ids, feature_names, digest, codes
    )


def _do_evaluate(body: dict) -> dict:
    run_id = body.get("runId")
    selected_trial = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")
    metric_floor = body.get("metricFloor")
    required_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    from app.core.hashing import is_non_negative_safe_int

    codes = set()
    malformed = False
    if not _valid_run_id(run_id):
        malformed = True
    if not is_safe_int(selected_trial) or selected_trial < 0:
        malformed = True
    if not is_hex64(dataset_digest):
        malformed = True
    if not valid_floor(metric_floor):
        malformed = True
    slices_ok = isinstance(required_slices, dict)
    if slices_ok:
        for k, v in required_slices.items():
            if not isinstance(k, str) or k == "" or not valid_floor(v):
                slices_ok = False
                break
    if not slices_ok:
        malformed = True
        required_slices = {}
    if not isinstance(rows, list):
        malformed = True
        rows = []
    if not is_non_negative_safe_int(bytes_processed):
        malformed = True
    if not is_non_negative_safe_int(max_bytes):
        malformed = True
    if malformed:
        codes.add("INVALID_INPUT")

    # Lineage check against stored successful selection.
    store = get_store()
    record = store.get(NS, run_id) if isinstance(run_id, str) else None
    lineage_ok = bool(record) and (
        record["response"]["selectedTrialId"] == selected_trial
        and record["response"]["datasetDigest"] == dataset_digest
    )
    if not lineage_ok and "INVALID_INPUT" not in codes:
        codes.add("INVALID_LINEAGE")

    rows_valid = isinstance(body.get("rows"), list) and len(rows) > 0
    if rows_valid:
        for row in rows:
            if not isinstance(row, dict):
                rows_valid = False
                break
            label = row.get("label")
            pred = row.get("prediction")
            slice_name = row.get("slice")
            if not (
                isinstance(label, int)
                and not isinstance(label, bool)
                and label in (0, 1)
                and isinstance(pred, int)
                and not isinstance(pred, bool)
                and pred in (0, 1)
                and isinstance(slice_name, str)
                and slice_name != ""
            ):
                rows_valid = False
                break
    if not rows_valid:
        codes.add("INVALID_TEST_ROW")

    test_metric = None
    critical_pass = True

    skip_metrics = not rows_valid
    if "INVALID_INPUT" in codes or "INVALID_LINEAGE" in codes or skip_metrics:
        critical_pass = False
    if not skip_metrics and "INVALID_INPUT" not in codes:
        total = len(rows)
        correct = sum(1 for r in rows if r["prediction"] == r["label"])
        aggregate = round12_ratio(correct, total)
        test_metric = aggregate

        by_slice = {}
        for r in rows:
            by_slice.setdefault(r["slice"], []).append(r)

        failed_slice = False
        missing_slice = False
        for name, floor in sorted(required_slices.items(), key=lambda kv: utf8_key(kv[0])):
            if name not in by_slice:
                missing_slice = True
                codes.add(f"MISSING_SLICE:{name}")
                continue
            srows = by_slice[name]
            sacc = round12_ratio(
                sum(1 for r in srows if r["prediction"] == r["label"]), len(srows)
            )
            if sacc < float(floor):
                failed_slice = True
                codes.add(f"SLICE_FLOOR:{name}")
        if missing_slice or failed_slice:
            critical_pass = False
        if aggregate < float(metric_floor):
            codes.add("AGGREGATE_FLOOR")

    if bytes_processed is not None and max_bytes is not None:
        if isinstance(bytes_processed, int) and isinstance(max_bytes, int):
            if bytes_processed > max_bytes:
                codes.add("BYTE_LIMIT")

    decision = "admit" if not codes else "reject"

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes(codes),
    }


def handle(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    phase = body.get("phase")
    if phase == "select":
        return _handle_select(body)
    if phase == "evaluate":
        return _do_evaluate(body)
    raise InvalidInput()


def _handle_select(body: dict) -> dict:
    from app.core.errors import Conflict

    store = get_store()
    run_id = body.get("runId")
    if _valid_run_id(run_id):
        existing = store.get(NS, run_id)
        if existing is not None:
            if existing["request"] == body:
                return existing["response"]
            raise Conflict("RUN_ID_CONFLICT")
    result = _do_select(body)
    if result["reasonCodes"] == [] and _valid_run_id(result["runId"]):
        store.set(NS, result["runId"], {"request": body, "response": result})
    return result
