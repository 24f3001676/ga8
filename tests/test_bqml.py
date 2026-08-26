"""Tests for POST /bqml (stateful)."""

import pytest

from app.endpoints.bqml import handle
from app.core.errors import Conflict


def sel_body(run_id="run-1", **kw):
    body = {
        "phase": "select",
        "runId": run_id,
        "forbiddenFeatures": kw.get("forbiddenFeatures", ["secret"]),
        "numTrialsLimit": kw.get("numTrialsLimit", 10),
        "rows": kw.get("rows", [
            {
                "id": "t1",
                "entity": "e1",
                "eventTime": "2026-01-01T00:00:00Z",
                "predictionTime": "2026-01-02T00:00:00Z",
                "version": 1,
                "split": "TRAIN",
                "features": {"f1": {"value": "x", "availableAt": "2025-12-31T00:00:00Z"}},
            },
            {
                "id": "e1",
                "entity": "e1",
                "eventTime": "2026-01-01T07:00:00+06:00",
                "predictionTime": "2026-01-02T00:00:00Z",
                "version": 2,
                "split": "EVAL",
                "features": {"f1": {"value": "y", "availableAt": "2026-01-01T00:00:00Z"}},
            },
        ]),
        "trials": kw.get("trials", [
            {"trialId": 9, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 4, "status": "SUCCEEDED", "evalMetric": 0.9},
        ]),
    }
    body["runId"] = kw.get("runId", run_id)
    body.update({k: v for k, v in kw.items() if k in ("forbiddenFeatures", "numTrialsLimit", "rows", "trials")})
    return body


def test_select_tie_breaks_smallest_trial_id():
    res = handle(sel_body())
    assert res["reasonCodes"] == []
    assert res["selectedTrialId"] == 4
    assert res["trainRowIds"] == ["t1"]
    assert res["evalRowIds"] == ["e1"]
    assert res["featureNames"] == ["f1"]
    assert len(res["datasetDigest"]) == 64


def test_select_replay_and_conflict():
    b = sel_body()
    r1 = handle(b)
    r2 = handle(dict(b))
    assert r1 == r2
    with pytest.raises(Conflict) as err:
        handle(sel_body(numTrialsLimit=11))
    assert err.value.code == "RUN_ID_CONFLICT"


def test_select_point_in_time_filtering():
    # feature not available before prediction time in eval row -> ineligible
    rows = sel_body()["rows"]
    rows[0]["features"]["late"] = {"value": "v", "availableAt": "2026-01-03T00:00:00Z"}
    rows[0]["features"]["ok"] = {"value": "v", "availableAt": "2026-01-01T00:00:00Z"}
    rows[1]["features"]["ok"] = {"value": "v", "availableAt": "2025-06-01T00:00:00Z"}
    res = handle(sel_body(rows=rows))
    assert res["featureNames"] == ["f1", "ok"]


def test_forbidden_and_missing_feature():
    b = sel_body(forbiddenFeatures=["f1"])
    res = handle(b)
    assert res["featureNames"] == []
    assert res["selectedTrialId"] == 4  # still selects


def test_no_successful_trial():
    res = handle(sel_body(trials=[{"trialId": 1, "status": "FAILED"}]))
    assert res["selectedTrialId"] is None
    assert res["reasonCodes"] == ["NO_SUCCESSFUL_TRIAL"]
    assert res["datasetDigest"] is not None  # selection itself well-formed


def test_non_finite_metric_ineligible():
    trials = [{"trialId": 1, "status": "FAILED"}, {"trialId": 2, "status": "SUCCEEDED"}]
    res = handle(sel_body(trials=trials))
    assert res["reasonCodes"] == ["NO_SUCCESSFUL_TRIAL"]


def test_trial_limit():
    trials = [{"trialId": i, "status": "SUCCEEDED", "evalMetric": 0.5} for i in range(11)]
    res = handle(sel_body(numTrialsLimit=10, trials=trials))
    assert res["selectedTrialId"] is None
    assert res["reasonCodes"] == ["TRIAL_LIMIT_EXCEEDED"]


def test_malformed_selection_invalid_input():
    res = handle(sel_body(rows=[{"id": "dup", "entity": "e"}, {"id": "dup", "entity": "f"}]))
    assert res["selectedTrialId"] is None
    assert res["datasetDigest"] is None
    assert res["reasonCodes"] == ["INVALID_INPUT"]

    res2 = handle(sel_body(runId=""))
    assert res2["reasonCodes"] == ["INVALID_INPUT"]


def test_row_dedup_highest_version_smallest_id():
    rows = [
        {
            "id": "b", "entity": "e1", "eventTime": "2026-01-01T00:00:00Z",
            "predictionTime": "2026-01-02T00:00:00Z", "version": 3, "split": "TRAIN",
            "features": {"f1": {"value": "x", "availableAt": "2025-12-31T00:00:00Z"}},
        },
        {
            "id": "a", "entity": "e1", "eventTime": "2026-01-01T00:00:00Z",
            "predictionTime": "2026-01-02T00:00:00Z", "version": 3, "split": "TRAIN",
            "features": {"f1": {"value": "x", "availableAt": "2025-12-31T00:00:00Z"}},
        },
        {
            "id": "c", "entity": "e1", "eventTime": "2026-01-01T00:00:00Z",
            "predictionTime": "2026-01-02T00:00:00Z", "version": 1, "split": "TRAIN",
            "features": {"f1": {"value": "x", "availableAt": "2025-12-31T00:00:00Z"}},
        },
    ]
    res = handle(sel_body(rows=rows))
    assert res["trainRowIds"] == ["a"]


def _eval_body(**kw):
    from app.core.hashing import sha256_of_json

    digest = sha256_of_json(
        {"trainRowIds": ["t1"], "evalRowIds": ["e1"], "featureNames": ["f1"]}
    )
    body = {
        "phase": "evaluate",
        "runId": "run-1",
        "selectedTrialId": 4,
        "datasetDigest": digest,
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 0, "prediction": 0, "slice": "critical"},
            {"label": 1, "prediction": 1, "slice": "other"},
            {"label": 0, "prediction": 0, "slice": "other"},
        ],
        "bytesProcessed": 1000,
        "maxBytes": 2000,
    }
    body.update(kw)
    return body


@pytest.fixture(scope="function")
def ensure_run():
    handle(sel_body())  # persist run-1 for evaluate tests


def test_evaluate_admit(ensure_run):
    res = handle(_eval_body())
    assert res["decision"] == "admit"
    assert res["testMetric"] == 1.0
    assert res["criticalSlicePass"] is True
    assert res["reasonCodes"] == []


def test_evaluate_aggregate_floor(ensure_run):
    rows = [
        {"label": 1, "prediction": 1, "slice": "critical"},
        {"label": 1, "prediction": 0, "slice": "other"},
    ]
    res = handle(_eval_body(metricFloor=0.8, rows=rows))
    assert res["decision"] == "reject"
    assert "AGGREGATE_FLOOR" in res["reasonCodes"]
    assert res["testMetric"] == 0.5
    assert res["criticalSlicePass"] is True  # does not summarize aggregate gate


def test_evaluate_slice_floor_and_missing(ensure_run):
    rows = [
        {"label": 1, "prediction": 1, "slice": "critical"},
        {"label": 1, "prediction": 0, "slice": "critical"},
        {"label": 1, "prediction": 1, "slice": "other"},
        {"label": 1, "prediction": 1, "slice": "other"},
    ]
    res = handle(_eval_body(requiredSlices={"critical": 1.0}, rows=rows))
    assert "SLICE_FLOOR:critical" in res["reasonCodes"]
    assert res["criticalSlicePass"] is False

    res2 = handle(_eval_body(requiredSlices={"absent": 0.5}))
    assert f"MISSING_SLICE:{'absent'}" in res2["reasonCodes"]
    assert res2["criticalSlicePass"] is False


def test_evaluate_byte_limit(ensure_run):
    res = handle(_eval_body(bytesProcessed=3000))
    assert "BYTE_LIMIT" in res["reasonCodes"]
    assert res["decision"] == "reject"


def test_evaluate_lineage_mismatch(ensure_run):
    res = handle(_eval_body(selectedTrialId=99))
    assert "INVALID_LINEAGE" in res["reasonCodes"]
    assert res["criticalSlicePass"] is False

    res2 = handle(_eval_body(runId="unknown-run"))
    assert "INVALID_LINEAGE" in res2["reasonCodes"]


def test_evaluate_invalid_rows_skip_metrics(ensure_run):
    res = handle(_eval_body(rows=[{"label": 5, "prediction": 1, "slice": "critical"}]))
    assert "INVALID_TEST_ROW" in res["reasonCodes"]
    assert res["testMetric"] is None
    assert res["criticalSlicePass"] is False

    res2 = handle(_eval_body(rows=[]))
    assert "INVALID_TEST_ROW" in res2["reasonCodes"]
    assert res2["testMetric"] is None


def test_unknown_phase_400_shape():
    from app.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        handle({"phase": "nope"})
    with pytest.raises(InvalidInput):
        handle({})


# ---- regression: full final-test decision matrix per grader feedback ----


def _matrix_eval(ensure_run, **kw):
    return handle(_eval_body(**kw))


def test_matrix_aggregate_pass_slice_fail(ensure_run):
    rows = [
        {"label": 1, "prediction": 1, "slice": "critical"},
        {"label": 1, "prediction": 0, "slice": "critical"},
    ]
    res = _matrix_eval(ensure_run, rows=rows)
    assert res["decision"] == "reject"
    assert "SLICE_FLOOR:critical" in res["reasonCodes"]
    assert res["testMetric"] == 0.5
    assert res["criticalSlicePass"] is False


def test_matrix_aggregate_fail_slice_pass(ensure_run):
    rows = [
        {"label": 1, "prediction": 1, "slice": "critical"},
        {"label": 1, "prediction": 0, "slice": "other"},
        {"label": 1, "prediction": 0, "slice": "other"},
        {"label": 1, "prediction": 0, "slice": "other"},
    ]
    res = _matrix_eval(ensure_run, metricFloor=0.5, rows=rows)
    assert "AGGREGATE_FLOOR" in res["reasonCodes"]
    assert "SLICE_FLOOR:critical" not in res["reasonCodes"]
    assert res["criticalSlicePass"] is True  # not summarized by aggregate gate
    assert res["decision"] == "reject"


def test_matrix_bytes_exact_at_limit_and_exceeded(ensure_run):
    res = _matrix_eval(ensure_run, bytesProcessed=2000)
    assert res["decision"] == "admit"
    assert "BYTE_LIMIT" not in res["reasonCodes"]

    res2 = _matrix_eval(ensure_run, bytesProcessed=2001)
    assert res2["decision"] == "reject"
    assert "BYTE_LIMIT" in res2["reasonCodes"]
    assert res2["criticalSlicePass"] is True  # byte gate not part of slice pass


def test_matrix_invalid_row_still_checks_lineage_and_bytes(ensure_run):
    rows = [{"label": 2, "prediction": 1, "slice": "critical"}]
    res = _matrix_eval(ensure_run, rows=rows, bytesProcessed=9999)
    assert res["testMetric"] is None
    assert set(res["reasonCodes"]) == {"INVALID_TEST_ROW", "BYTE_LIMIT"}
    assert res["decision"] == "reject"

    res2 = _matrix_eval(ensure_run, runId="ghost", rows=rows)
    assert set(res2["reasonCodes"]) == {"INVALID_LINEAGE", "INVALID_TEST_ROW"}


def test_matrix_empty_rows_skip_slice_checks(ensure_run):
    res = _matrix_eval(ensure_run, rows=[], requiredSlices={"never": 0.5})
    assert "INVALID_TEST_ROW" in res["reasonCodes"]
    assert f"MISSING_SLICE:{'never'}" not in res["reasonCodes"]
    assert res["testMetric"] is None


def test_matrix_rounding_twelve_decimals(ensure_run):
    rows = [
        {"label": 1, "prediction": 1, "slice": "s"},
        {"label": 1, "prediction": 0, "slice": "s"},
        {"label": 1, "prediction": 1, "slice": "s"},
    ]
    res = _matrix_eval(ensure_run, rows=rows, requiredSlices={})
    assert res["testMetric"] == round(2 / 3, 12)


def test_evaluate_does_not_mutate_selection(ensure_run):
    before = handle(_eval_body())
    after = handle(_eval_body(metricFloor=0.1))
    assert before["datasetDigest"] == after["datasetDigest"]
    # original selection still replayable identically
    sel = sel_body()
    assert handle(sel)["selectedTrialId"] == 4
