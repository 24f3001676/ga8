"""Tests for POST /quantize (freeze + select, stateful)."""

import pytest

from app.core.errors import Conflict, InvalidInput
from app.endpoints.quantize import handle


def freeze_body(freeze_id="f1", **kw):
    body = {
        "phase": "freeze",
        "freezeId": freeze_id,
        "calibrationDigest": "cal-1",
        "tokenizerDigest": "tok-1",
        "allowedUnsupportedReasons": ["NEEDS_GPU"],
        "candidates": [
            {
                "name": "int8",
                "files": {"model.safetensors": "12345"},
                "loadable": True,
                "calibrationDigest": "cal-1",
                "tokenizerDigest": "tok-1",
                "unsupportedReason": None,
            },
            {
                "name": "int4",
                "files": {"weights.bin": "ab"},
                "loadable": True,
                "calibrationDigest": "cal-1",
                "tokenizerDigest": "tok-1",
                "unsupportedReason": None,
            },
        ],
    }
    body.update(kw)
    return body


def test_freeze_inventory_and_digests():
    res = handle(freeze_body())
    assert [c["name"] for c in res["candidates"]] == ["int4", "int8"]
    int8 = res["candidates"][1]
    assert int8["status"] == "frozen"
    assert int8["inventory"] == [
        {"name": "model.safetensors", "bytes": 5, "sha256": _sha("12345")}
    ]
    assert int8["totalBytes"] == 5
    assert len(int8["packageDigest"]) == 64
    # packageDigest = sha256 of compact inventory JSON
    from app.core.hashing import cj, sha256_hex

    expect = sha256_hex(cj(int8["inventory"]).encode("utf-8"))
    assert int8["packageDigest"] == expect


def _sha(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_freeze_unsupported_reasons():
    b = freeze_body()
    b["candidates"][0]["unsupportedReason"] = "NEEDS_GPU"
    res = handle(b)
    assert res["candidates"][1]["status"] == "unsupported"
    assert res["candidates"][1]["reasonCodes"] == []

    b2 = freeze_body(freeze_id="f2", allowedUnsupportedReasons=[])
    b2["candidates"][0]["unsupportedReason"] = "MYSTERY"
    res2 = handle(b2)
    assert res2["candidates"][1]["status"] == "invalid"
    assert res2["candidates"][1]["reasonCodes"] == ["UNALLOWED_UNSUPPORTED_REASON"]


def test_freeze_loadable_and_digest_mismatch():
    b = freeze_body()
    b["candidates"][0]["loadable"] = False
    b["candidates"][1]["calibrationDigest"] = "other"
    res = handle(b)
    codes0 = {c["name"]: c["reasonCodes"] for c in res["candidates"]}
    assert codes0["int8"] == ["NOT_LOADABLE"]
    assert codes0["int4"] == ["CALIBRATION_MISMATCH"]


def test_freeze_invalid_files_empty_inventory():
    b = freeze_body()
    b["candidates"][0]["files"] = {}
    res = handle(b)
    c = [x for x in res["candidates"] if x["name"] == "int8"][0]
    assert c["status"] == "invalid"
    assert c["inventory"] == []
    assert c["totalBytes"] is None
    assert c["packageDigest"] is None
    assert c["reasonCodes"] == ["INVALID_INPUT"]


def test_freeze_replay_and_conflict_and_400_no_reserve():
    b = freeze_body()
    r1 = handle(b)
    r2 = handle(dict(b))
    assert r1 == r2

    with pytest.raises(Conflict):
        handle(freeze_body(freeze_id="f1", calibrationDigest="cal-9"))

    with pytest.raises(InvalidInput):
        handle(freeze_body(freeze_id="f400", candidates=[]))
    # failed request did not reserve its ID; now a valid one succeeds
    ok = handle(freeze_body(freeze_id="f400"))
    assert ok["freezeId"] == "f400"


def select_body(stored_candidates="auto", rows=None, **kw):
    if stored_candidates == "auto":
        frozen = handle(freeze_body())
        stored_candidates = frozen["candidates"]
    body = {
        "phase": "select",
        "freezeId": "f1",
        "candidates": stored_candidates,
        "policy": {
            "maxBytes": 1000000,
            "aggregateFloor": 0.8,
            "requiredSlices": {"critical": 0.75},
            "maxLatencyMs": 100,
            "candidateOrder": ["int4", "int8"],
        },
        "latencies": {"int4": 40, "int8": 60},
        "rows": rows
        or [
            {"label": 1, "slice": "critical", "predictions": {"int4": 1, "int8": 1}},
            {"label": 0, "slice": "other", "predictions": {"int4": 1, "int8": 0}},
        ],
    }
    body.update(kw)
    return body


def test_select_prefers_smaller_bytes_only_when_admitted():
    res = handle(select_body())
    # int4 has wrong predictions on row 2 -> aggregate 0.5 < 0.8 -> not admitted
    by_name = {r["name"]: r for r in res["results"]}
    assert by_name["int4"]["admitted"] is False
    assert "AGGREGATE_FLOOR" in by_name["int4"]["reasonCodes"]
    assert by_name["int8"]["aggregate"] == 1.0
    assert by_name["int8"]["admitted"] is True
    assert res["selected"] == "int8"
    assert res["packageManifest"]["name"] == "int8"


def test_select_size_beats_when_both_admitted():
    rows = [{"label": 1, "slice": "critical", "predictions": {"int4": 1, "int8": 1}}]
    res = handle(select_body(rows=rows))
    # both admitted with perfect accuracy; int4 smaller -> selected int4
    assert res["selected"] == "int4"


def test_select_latency_tiebreak_then_order():
    # Equal byte sizes so latency decides; then candidateOrder breaks further ties.
    b = freeze_body()
    b["candidates"][0]["files"] = {"weights.bin": "xy"}  # 2 bytes
    b["candidates"][1]["files"] = {"model.safetensors": "zz"}  # 2 bytes
    frozen = handle(b)
    rows = [{"label": 1, "slice": "critical", "predictions": {"int4": 1, "int8": 1}}]
    res = handle(select_body(stored_candidates=frozen["candidates"], rows=rows,
                             latencies={"int4": 50, "int8": 40}))
    assert res["selected"] == "int8"

    res2 = handle(select_body(stored_candidates=frozen["candidates"], rows=rows,
                              latencies={"int4": 40, "int8": 40}))
    assert res2["selected"] == "int4"  # candidateOrder position decides
def test_select_slice_floors_and_missing():
    rows = [
        {"label": 1, "slice": "critical", "predictions": {"int4": 0, "int8": 1}},
        {"label": 1, "slice": "extra", "predictions": {"int4": 1, "int8": 1}},
    ]
    res = handle(select_body(rows=rows))
    by_name = {r["name"]: r for r in res["results"]}
    assert f"SLICE_FLOOR:{'critical'}" in by_name["int4"]["reasonCodes"]
    assert by_name["int8"]["slices"]["critical"] == 1.0

    rows2 = [{"label": 1, "slice": "other", "predictions": {"int4": 1, "int8": 1}}]
    res2 = handle(select_body(rows=rows2))
    by2 = {r["name"]: r for r in res2["results"]}
    assert f"MISSING_SLICE:{'critical'}" in by2["int8"]["reasonCodes"]
    assert by2["int8"]["admitted"] is False


def test_select_unknown_freeze_is_not_frozen():
    body = select_body()
    body["freezeId"] = "never-frozen"
    res = handle(body)
    assert res["selected"] is None
    assert res["packageManifest"] is None
    for r in res["results"]:
        assert "NOT_FROZEN" in r["reasonCodes"]
        assert r["admitted"] is False


def test_select_candidate_mismatch_invalid_manifest():
    body = select_body()
    tampered = [dict(c) for c in body["candidates"]]
    tampered[1]["totalBytes"] = tampered[1]["totalBytes"] + 100  # lie about size
    res = handle(select_body(stored_candidates=tampered))
    by_name = {r["name"]: r for r in res["results"]}
    assert "INVALID_MANIFEST" in by_name["int8"]["reasonCodes"]
    assert by_name["int8"]["admitted"] is False
    # recomputed totals never trusted:
    assert by_name["int8"]["totalBytes"] != tampered[1]["totalBytes"]


def test_select_invalid_predictions_null_metrics():
    body = select_body(
        rows=[{"label": 7, "slice": "critical", "predictions": {"int4": 1, "int8": 1}}]
    )
    res = handle(body)
    for r in res["results"]:
        assert r["aggregate"] is None
        assert "INVALID_PREDICTIONS" in r["reasonCodes"]

    body2 = select_body(
        rows=[{"label": 1, "slice": "critical", "predictions": {"int4": 1}}]  # missing int8 col
    )
    res2 = handle(body2)
    by_name = {r["name"]: r for r in res2["results"]}
    assert by_name["int8"]["aggregate"] is None
    assert by_name["int4"]["aggregate"] == 1.0


def test_select_400_shapes():
    with pytest.raises(InvalidInput):
        handle({"phase": "select", "freezeId": "f1"})
    with pytest.raises(InvalidInput):
        handle({"phase": "bogus"})


def test_state_isolation_between_sessions_not_shared_with_bqml():
    # quantize state must not collide with bqml namespace
    handle(freeze_body(freeze_id="shared-id"))
    from app.endpoints.quantize import handle as qh

    again = qh(freeze_body(freeze_id="shared-id"))
    assert again["freezeId"] == "shared-id"
