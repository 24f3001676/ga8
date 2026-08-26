"""Tests for POST /promote."""

import pytest

from app.core.errors import InvalidInput
from app.endpoints.promote import handle


def evaluation(acc=0.9, lat=50, size=500000, artifact="sha:a", dataset="sha:d", schema="sha:s",
               created="2026-01-01T00:00:30Z", slices=None):
    return {
        "createdAt": created,
        "artifactDigest": artifact,
        "datasetDigest": dataset,
        "schemaDigest": schema,
        "accuracy": acc,
        "latencyMs": lat,
        "sizeBytes": size,
        "slices": {"critical": 0.85} if slices is None else slices,
    }


def body(versions, champion="1", as_of="2026-01-01T00:01:00Z", **policy_kw):
    policy = {
        "datasetDigest": "sha:d",
        "schemaDigest": "sha:s",
        "maxAgeSeconds": 3600,
        "accuracyFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "maxLatencyMs": 100,
        "maxSizeBytes": 1000000,
        "minImprovement": 0.01,
    }
    policy.update(policy_kw)
    return {"asOf": as_of, "championVersion": champion, "policy": policy, "versions": versions}


def v(version, eval_obj, artifact="sha:a", tags=None):
    return {"version": version, "artifactDigest": artifact, "tags": tags or {}, "evaluation": eval_obj}


def test_promote_best_challenger():
    b = body([
        v("1", evaluation(acc=0.80)),
        v("2", evaluation(acc=0.95, artifact="sha:b"), artifact="sha:b"),
    ])
    res = handle(b)
    assert res["action"] == "promote"
    assert res["selectedVersion"] == "2"
    # ranked by accuracy descending
    assert res["eligibleVersions"] == ["2", "1"]
    assert res["aliasMutation"] == {"alias": "champion", "version": "2"}
    assert res["evidence"]["accuracy"] == 0.95
    assert res["failedGates"] == {"1": [], "2": []}


def test_retain_when_improvement_too_small():
    b = body([
        v("1", evaluation(acc=0.89)),
        v("2", evaluation(acc=0.895, artifact="sha:b"), artifact="sha:b"),
    ])
    res = handle(b)
    assert res["action"] == "retain"
    assert res["selectedVersion"] == "1"
    assert res["aliasMutation"] is None
    assert res["evidence"]["accuracy"] == 0.89


def test_promote_idempotent_replay_becomes_retain():
    b = body([
        v("1", evaluation(acc=0.80)),
        v("2", evaluation(acc=0.95, artifact="sha:b"), artifact="sha:b"),
    ])
    r1 = handle(b)
    assert r1["action"] == "promote"
    r2 = handle(b)
    assert r2["action"] == "retain"
    assert r2["selectedVersion"] == "2"
    assert r2["aliasMutation"] is None


def test_ranking_order_accuracy_latency_size_version():
    b = body([
        v("3", evaluation(acc=0.9, lat=40)),
        v("2", evaluation(acc=0.9, lat=30)),
    ], champion="3")
    res = handle(b)
    assert res["eligibleVersions"] == ["2", "3"]

    # size breaks latency ties; version numeric ascending ("10" > "9" numerically)
    b2 = body([
        v("10", evaluation(acc=0.9, lat=30, size=100)),
        v("9", evaluation(acc=0.9, lat=30, size=100)),
    ], champion="10")
    res2 = handle(b2)
    assert res2["eligibleVersions"] == ["9", "10"]


def test_future_evaluation_blocked_champion():
    b = body([v("1", evaluation(created="2026-01-01T00:01:01Z"))])
    res = handle(b)
    assert res["action"] == "block"
    assert res["failedGates"]["1"] == ["FUTURE_EVALUATION"]
    assert res["selectedVersion"] is None
    assert res["evidence"] is None
    assert res["eligibleVersions"] == []


def test_stale_and_gate_codes():
    b = body([
        v("1", evaluation(created="2025-12-31T22:00:00Z")),  # older than asOf-3600s
    ])
    res = handle(b)
    codes = res["failedGates"]["1"]
    assert "STALE_EVALUATION" in codes

    b2 = body([v("1", evaluation(acc=0.5))])
    res2 = handle(b2)
    assert set(res2["failedGates"]["1"]) == {"ACCURACY_FLOOR"}

    b3 = body([v("1", evaluation(lat=200))])
    assert "LATENCY_LIMIT" in handle(b3)["failedGates"]["1"]

    b4 = body([v("1", evaluation(size=2000000))])
    assert "SIZE_LIMIT" in handle(b4)["failedGates"]["1"]

    b5 = body([v("1", evaluation(slices={"other": 0.9}))])
    assert f"MISSING_SLICE:{'critical'}" in handle(b5)["failedGates"]["1"]

    b6 = body([v("1", evaluation(slices={"critical": 0.5}))])
    assert "SLICE_FLOOR:critical" in handle(b6)["failedGates"]["1"]

    b7 = body([v("1", evaluation(slices={"critical": 1.5}))])
    assert "SLICE_RANGE:critical" in handle(b7)["failedGates"]["1"]

    b8 = body([v("1", evaluation(artifact="sha:OTHER"))], )
    assert "ARTIFACT_MISMATCH" in handle(b8)["failedGates"]["1"]

    b9 = body([v("1", evaluation(dataset="sha:X"))])
    assert "DATASET_MISMATCH" in handle(b9)["failedGates"]["1"]

    b10 = body([v("1", evaluation(schema="sha:X"))])
    assert "SCHEMA_MISMATCH" in handle(b10)["failedGates"]["1"]


def test_missing_evaluation_non_finite_metric_range_timestamp():
    e = evaluation()
    del e["accuracy"]
    b = body([v("1", e)])
    assert "MISSING_EVALUATION" in handle(body([{"version": "1", "artifactDigest": "sha:a"}]))["failedGates"]["1"]
    assert "NON_FINITE" in handle(b)["failedGates"]["1"]

    b2 = body([v("1", dict(evaluation(), accuracy=1.5))])
    assert "METRIC_RANGE" in handle(b2)["failedGates"]["1"]

    b3 = body([v("1", dict(evaluation(), createdAt="not-a-time"))])
    assert "INVALID_TIMESTAMP" in handle(b3)["failedGates"]["1"]


def test_invalid_and_duplicate_versions_rejected_before_maps():
    entries = [
        v("01", evaluation(acc=0.99)),
        v("2", evaluation(acc=0.99, artifact="sha:b"), artifact="sha:b"),
        v("2", evaluation(acc=0.5)),
        v("1", evaluation(acc=0.8)),
    ]
    res = handle(body(entries))
    assert res["failedGates"]["01"] == ["INVALID_VERSION"]
    assert set(res["failedGates"]["2"]) == {"DUPLICATE_VERSION"}
    assert "01" not in res["eligibleVersions"]
    # duplicates are excluded from eligibility entirely
    assert res["eligibleVersions"] == ["1"]


def test_champion_not_listed_blocks():
    b = body([v("2", evaluation())], champion="7")
    res = handle(b)
    assert res["action"] == "block"
    assert res["selectedVersion"] is None


def test_invalid_policy_attaches_code():
    res = handle(body([v("1", evaluation())], accuracyFloor=2.0))
    assert res["failedGates"]["1"] == ["INVALID_POLICY"]
    assert res["action"] == "block"


def test_400_invalid_input():
    with pytest.raises(InvalidInput):
        handle({"asOf": "2026-01-01T00:01:00Z", "versions": [], "championVersion": "1"})  # missing policy
    with pytest.raises(InvalidInput):
        handle({"asOf": "2026-01-01T00:01:00Z", "championVersion": "1",
                "policy": {"datasetDigest": "d", "schemaDigest": "s"}, "versions": "x"})
    with pytest.raises(InvalidInput):
        handle(body([v("1", evaluation())], champion=5))
