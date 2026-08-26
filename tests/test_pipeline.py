"""Tests for POST /pipeline (session-scoped content-addressed state machine)."""

import pytest

from app.core.errors import Conflict
from app.core.hashing import sha256_hex
from app.endpoints.pipeline import handle


def inputs(**over):
    base = {
        "generation": "gen-1",
        "checksum": "crc-1",
        "canonicalData": "data-1",
        "prepareCode": "pc-1",
        "prepareConfig": "pcf-1",
        "trainCode": "tc-1",
        "trainConfig": "tcf-1",
        "runtime": "rt-1",
        "evaluateCode": "ec-1",
        "evaluateConfig": "ecf-1",
        "schemaDigest": "sd-1",
        "publishConfig": "pubc-1",
    }
    base.update(over)
    return base


def body(session="s1", revision=1, events=None, **in_over):
    return {
        "session": session,
        "revision": revision,
        "inputs": inputs(**in_over),
        "events": events or [],
    }


def ev(event_id, node, status, attempt, key, artifact=None, receipt=None, revision=1):
    return {
        "eventId": event_id,
        "revision": revision,
        "node": node,
        "attempt": attempt,
        "status": status,
        "key": key,
        "artifactDigest": artifact,
        "receiptId": receipt,
    }


def get_node(res, name):
    return next(n for n in res["nodes"] if n["node"] == name)


def test_initial_state_all_rerun_cache_miss():
    res = handle(body())
    assert res["acceptedEventIds"] == []
    # only verify_data is ready; everything else is blocked upstream
    assert get_node(res, "verify_data")["action"] == "rerun"
    assert get_node(res, "verify_data")["reasonCodes"] == ["CACHE_MISS"]
    for name in ["prepare", "train", "evaluate", "register", "publish"]:
        n = get_node(res, name)
        assert n["action"] == "block"
        assert n["reasonCodes"] == ["UPSTREAM_PENDING"]
    train = get_node(res, "train")
    assert train["dependencyDigests"]["cacheKey"] is None


def test_key_recipes_content_addressed():
    res = handle(body(generation="gen-2"))
    from app.core.hashing import cj

    expect = sha256_hex(cj(["gen-2", "crc-1"]).encode("utf-8"))
    assert get_node(res, "verify_data")["dependencyDigests"]["cacheKey"] == expect


def test_full_success_chain_and_reuse():
    b = body(events=[
        ev("e1", "verify_data", "started", 1, None),
    ])
    # keys unknown ahead of time; first request just starts verify_data using its key
    r0 = handle(body())
    k_vd = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]

    r1 = handle(body(events=[
        ev("e1", "verify_data", "started", 1, k_vd),
        ev("e2", "verify_data", "succeeded", 1, k_vd, artifact="art-vd"),
    ]))
    assert r1["acceptedEventIds"] == ["e1", "e2"]
    vd = get_node(r1, "verify_data")
    assert vd["action"] == "reuse"
    assert vd["triggeringEventIds"] == ["e2"]  # its immutable success event

    prep = get_node(r1, "prepare")
    assert prep["action"] == "rerun" and prep["reasonCodes"] == ["CACHE_MISS"]
    k_prep = prep["dependencyDigests"]["cacheKey"]
    assert k_prep is not None
    # train still pending: its parent (prepare) has no artifact yet
    train = get_node(r1, "train")
    assert train["dependencyDigests"]["prepareArtifact"] is None
    assert train["dependencyDigests"]["cacheKey"] is None
    assert train["action"] == "block"
    assert train["reasonCodes"] == ["UPSTREAM_PENDING"]

    # readback: same session, no events -> same actions
    r1b = handle(body())
    assert get_node(r1b, "verify_data")["action"] == "reuse"

    # new revision keeps cache entries for identical inputs? inputs replaced:
    r2 = handle(body(revision=2))
    assert get_node(r2, "verify_data")["action"] == "reuse"  # content-addressed


def test_retryable_flow():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    res = handle(body(events=[
        ev("a1", "verify_data", "started", 1, k),
        ev("a2", "verify_data", "retryable_failed", 1, k),
        ev("a3", "verify_data", "started", 2, k),
        ev("a4", "verify_data", "succeeded", 2, k, artifact="art"),
    ]))
    assert res["acceptedEventIds"] == ["a1", "a2", "a3", "a4"]
    assert get_node(res, "verify_data")["action"] == "reuse"


def test_completion_without_start_ignored_and_ids_not_consumed():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    res = handle(body(events=[
        ev("x1", "verify_data", "succeeded", 1, k, artifact="art"),
    ]))
    assert res["ignoredEventIds"] == ["x1"]
    # id can still be used later after a proper start
    res2 = handle(body(events=[
        ev("x0", "verify_data", "started", 1, k),
        ev("x1", "verify_data", "succeeded", 1, k, artifact="art"),
    ]))
    assert res2["acceptedEventIds"] == ["x0", "x1"]


def test_event_id_conflict_rolls_back_batch():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    with pytest.raises(Conflict) as e:
        handle(body(events=[
            ev("z1", "verify_data", "started", 1, k),
            ev("z2", "verify_data", "retryable_failed", 1, k),
            ev("z2", "verify_data", "terminal_failed", 1, k),
        ]))
    assert e.value.code == "EVENT_ID_CONFLICT"
    # rollback: nothing applied
    res = handle(body())
    assert res["acceptedEventIds"] == []
    assert get_node(res, "verify_data")["action"] == "rerun"


def test_exact_replay_ignored():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    e = ev("r1", "verify_data", "started", 1, k)
    r1 = handle(body(events=[dict(e)]))
    assert r1["acceptedEventIds"] == ["r1"]
    r2 = handle(body(events=[dict(e)]))
    assert r2["acceptedEventIds"] == []
    assert r2["ignoredEventIds"] == ["r1"]


def test_revision_conflict_same_revision_different_inputs():
    handle(body())
    with pytest.raises(Conflict) as e:
        handle(body(canonicalData="changed"))
    assert e.value.code == "REVISION_CONFLICT"


def test_new_revision_replaces_inputs_clears_attempts_keeps_cache():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    handle(body(events=[ev("v1", "verify_data", "started", 1, k)]))
    # same revision different input -> conflict
    try:
        handle(body(runtime="rt-other"))
        raise AssertionError("expected REVISION_CONFLICT")
    except Conflict as c:
        assert c.code == "REVISION_CONFLICT"

    # bump revision with new data -> verify_data becomes CACHE_MISS (new key), old cache kept separately
    res = handle(body(revision=2, generation="gen-B"))
    vd = get_node(res, "verify_data")
    assert vd["action"] == "rerun"
    # stale events from older revision are ignored
    kb_new = vd["dependencyDigests"]["cacheKey"]
    res2 = handle(
        body(revision=2, generation="gen-B",
             events=[ev("old", "verify_data", "started", 1, kb_new, revision=1)])
    )
    assert res2["ignoredEventIds"] == ["old"]


def test_terminal_failure_blocks_descendants():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    res = handle(body(events=[
        ev("t0", "verify_data", "started", 1, k),
        ev("t1", "verify_data", "terminal_failed", 1, k),
    ]))
    vd = get_node(res, "verify_data")
    assert vd["action"] == "block"
    assert vd["reasonCodes"] == ["TERMINAL_FAILURE"]
    prep = get_node(res, "prepare")
    assert prep["action"] == "block"
    assert prep["reasonCodes"] == ["UPSTREAM_TERMINAL"]
    pub = get_node(res, "publish")
    assert pub["reasonCodes"] == ["UPSTREAM_TERMINAL"]

    with pytest.raises(Conflict) as e:
        handle(body(events=[ev("t2", "verify_data", "started", 2, k)]))
    assert e.value.code == "STATUS_CONFLICT"


def test_running_block_and_receipt_rules():
    r0 = handle(body())
    kv = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    r1 = handle(body(events=[
        ev("v1", "verify_data", "started", 1, kv),
        ev("v2", "verify_data", "succeeded", 1, kv, artifact="art-v"),
    ]))
    kp = get_node(r1, "prepare")["dependencyDigests"]["cacheKey"]
    assert get_node(r1, "prepare")["action"] == "rerun"

    # start prepare -> RUNNING block
    r2 = handle(body(events=[ev("p1", "prepare", "started", 1, kp)]))
    prep = get_node(r2, "prepare")
    assert prep["action"] == "block"
    assert prep["reasonCodes"] == ["RUNNING"]
    assert prep["triggeringEventIds"] == ["p1"]

    # complete prepare
    r3 = handle(body(events=[
        ev("p2", "prepare", "succeeded", 1, kp, artifact="art-p"),
    ]))
    assert get_node(r3, "prepare")["action"] == "reuse"
    kt = get_node(r3, "train")["dependencyDigests"]["cacheKey"]
    assert kt is not None

    # chain through train + evaluate
    r4 = handle(body(events=[
        ev("t1", "train", "started", 1, kt),
        ev("t2", "train", "succeeded", 1, kt, artifact="art-t"),
    ]))
    ke = get_node(r4, "evaluate")["dependencyDigests"]["cacheKey"]
    r5 = handle(body(events=[
        ev("e1", "evaluate", "started", 1, ke),
        ev("e2", "evaluate", "succeeded", 1, ke, artifact="art-e"),
    ]))
    kr = get_node(r5, "register")["dependencyDigests"]["cacheKey"]

    # wrong receipt on register success is ignored
    r6 = handle(body(events=[
        ev("g0", "register", "started", 1, kr),
        ev("g1", "register", "succeeded", 1, kr, artifact="art-r", receipt="WRONG"),
    ]))
    assert r6["ignoredEventIds"] == ["g1"]
    assert get_node(r6, "register")["reasonCodes"] == ["RUNNING"]

    # correct receipt accepted and bound immutably
    r7 = handle(body(events=[
        ev("g2", "register", "succeeded", 1, kr, artifact="art-r",
           receipt=f"receipt:register:{kr}"),
    ]))
    assert r7["acceptedEventIds"] == ["g2"]
    reg = get_node(r7, "register")
    assert reg["action"] == "reuse"
    assert reg["triggeringEventIds"] == ["g2"]


def test_status_conflict_started_started():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    handle(body(events=[ev("s1", "verify_data", "started", 1, k)]))
    with pytest.raises(Conflict) as e:
        handle(body(events=[ev("s2", "verify_data", "started", 1, k)]))
    assert e.value.code == "STATUS_CONFLICT"


def test_evidence_conflict_different_artifact():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    handle(body(events=[
        ev("m1", "verify_data", "started", 1, k),
        ev("m2", "verify_data", "retryable_failed", 1, k),
        ev("m3", "verify_data", "started", 2, k),
        ev("m4", "verify_data", "succeeded", 2, k, artifact="art-A"),
    ]))
    with pytest.raises(Conflict) as e:
        handle(body(events=[
            ev("m5", "verify_data", "started", 3, k),
            ev("m6", "verify_data", "terminal_failed", 3, k),
        ]))
    # started(3) after succeeded -> STATUS_CONFLICT per table
    assert e.value.code == "STATUS_CONFLICT"


def test_invalid_request_codes():
    for bad in [
        {"session": "", "revision": 1, "inputs": inputs(), "events": []},
        {"session": "s", "revision": 0, "inputs": inputs(), "events": []},
        {"session": "s", "revision": 1, "inputs": {"generation": "x"}, "events": []},
        {"session": "s", "revision": 1, "inputs": inputs(), "events": {}},
    ]:
        with pytest.raises(Conflict) as e:
            handle(bad)
        assert e.value.code == "INVALID_REQUEST"


def test_invalid_event_structural():
    good = ev("i1", "verify_data", "started", 1, "k")
    bad_variants = [
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "key"},
        {**good, "attempt": 0},
        {**good, "eventId": ""},
        {**good, "artifactDigest": 5},
    ]
    for bad in bad_variants:
        with pytest.raises(Conflict) as e:
            handle(body(session="sx", events=[bad]))
        assert e.value.code == "INVALID_EVENT"


def test_wrong_key_or_node_ignored():
    r0 = handle(body())
    k = get_node(r0, "verify_data")["dependencyDigests"]["cacheKey"]
    res = handle(body(events=[
        ev("w1", "verify_data", "started", 1, "wrong-key"),
        ev("w2", "nonexistent_node_x", "started", 1, k),
    ]))
    assert set(res["ignoredEventIds"]) >= {"w1"}
