"""Tests for POST /build-corpus."""

import hashlib
import json

from app.core.hashing import crc32c_hex
from app.endpoints.build_corpus import handle


def make_row(id="r1", entity="  Héllo ", event_time="2026-01-02T05:30:00+05:30", revision=1, text="Hi\t THERE"):
    return {"id": id, "entity": entity, "eventTime": event_time, "revision": revision, "text": text}


def make_object(rows, uri="gs://b/o", generation="7", fetched="7", schema_id="training-v1", content=None):
    if content is None:
        content = "\n".join(json.dumps(r) for r in rows)
    return {
        "uri": uri,
        "generation": generation,
        "fetchedGeneration": fetched,
        "crc32c": crc32c_hex(content.encode("utf-8")),
        "schemaId": schema_id,
        "content": content,
    }


POLICY = {"minTime": "2026-01-01T00:00:00Z", "maxTime": "2026-12-31T23:59:59Z", "contaminationThreshold": 0.8}


def test_happy_path_single_row_train():
    row = make_row()
    body = {"policy": POLICY, "objects": [make_object([row])]}
    res = handle(body)
    # entity canonicalized to "héllo"
    assert len(res["splits"]["train"]) == 1
    line = res["splits"]["train"][0]
    assert line["entity"] == "héllo"
    assert line["eventTime"] == "2026-01-02T00:00:00.000Z"
    assert line["text"] == "hi there"
    # digest is over compact JSON (exact key order, non-ASCII direct) + newline
    expected = json.dumps(line, ensure_ascii=False, separators=(",", ":"),) + "\n"
    assert list(line.keys()) == ["id", "entity", "eventTime", "revision", "text"]
    assert res["digests"]["train"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert res["rejectedObjects"] == []
    assert res["rejectedRows"] == []
    assert len(res["lineage"]) == 1


def test_uri_invalid_and_generation_codes():
    obj = make_object([make_row()], uri="http://bad", generation="x", fetched="y")
    body = {"policy": POLICY, "objects": [obj]}
    res = handle(body)
    codes = res["rejectedObjects"][0]["reasonCodes"]
    assert "URI_INVALID" in codes
    # independently applicable: non-decimal fields AND unequal values
    assert "GENERATION_INVALID" in codes
    assert "GENERATION_MISMATCH" in codes


def test_generation_code_matrix():
    # invalid but equal values -> INVALID only
    obj = make_object([make_row()], generation="abc", fetched="abc")
    res = handle({"policy": POLICY, "objects": [obj]})
    assert res["rejectedObjects"][0]["reasonCodes"] == ["GENERATION_INVALID"]

    # valid but unequal -> MISMATCH only
    obj2 = make_object([make_row()], generation="5", fetched="6")
    res2 = handle({"policy": POLICY, "objects": [obj2]})
    assert res2["rejectedObjects"][0]["reasonCodes"] == ["GENERATION_MISMATCH"]

    # leading zeros are decimal strings; unequal strings -> MISMATCH
    obj3 = make_object([make_row()], generation="07", fetched="7")
    res3 = handle({"policy": POLICY, "objects": [obj3]})
    assert res3["rejectedObjects"][0]["reasonCodes"] == ["GENERATION_MISMATCH"]

    # missing fetchedGeneration -> INVALID + MISMATCH (None != "5")
    obj4 = make_object([make_row()], generation="5")
    obj4.pop("fetchedGeneration")
    res4 = handle({"policy": POLICY, "objects": [obj4]})
    assert set(res4["rejectedObjects"][0]["reasonCodes"]) == {
        "GENERATION_INVALID",
        "GENERATION_MISMATCH",
    }


def test_all_object_codes_emitted_independently():
    obj = {
        "uri": "s3://nope",
        "generation": "abc",
        "fetchedGeneration": "5",
        "crc32c": "ZZZZ",
        "schemaId": "wrong",
        "content": "{not json",
    }
    body = {"policy": POLICY, "objects": [obj]}
    res = handle(body)
    codes = set(res["rejectedObjects"][0]["reasonCodes"])
    assert codes == {
        "URI_INVALID",
        "GENERATION_INVALID",
        "GENERATION_MISMATCH",
        "CRC32C_INVALID",
        "SCHEMA_INVALID",
        "JSONL_INVALID",
    }
    assert res["rejectedObjects"][0]["uri"] == "s3://nope"  # supplied string kept


def test_crc_mismatch_only_with_valid_syntax_and_string_content():
    good_content = json.dumps(make_row())
    obj = make_object([make_row()])
    obj["crc32c"] = "deadbeef"
    body = {"policy": POLICY, "objects": [obj]}
    res = handle(body)
    codes = res["rejectedObjects"][0]["reasonCodes"]
    assert codes == ["CRC32C_MISMATCH"]

    # non-string content: no mismatch check, but SCHEMA_INVALID
    obj2 = make_object([])
    obj2["content"] = 12345
    res2 = handle({"policy": POLICY, "objects": [obj2]})
    codes2 = res2["rejectedObjects"][0]["reasonCodes"]
    assert "SCHEMA_INVALID" in codes2
    assert "CRC32C_MISMATCH" not in codes2


def test_empty_file_and_bad_row_shape():
    obj = make_object([])
    obj["content"] = ""
    obj["crc32c"] = crc32c_hex(b"")
    res = handle({"policy": POLICY, "objects": [obj]})
    assert res["rejectedObjects"][0]["reasonCodes"] == ["SCHEMA_INVALID"]

    bad = make_object([])
    bad["content"] = json.dumps({"id": "a", "entity": "e"}) + "\n"
    bad["crc32c"] = crc32c_hex(bad["content"].encode())
    res2 = handle({"policy": POLICY, "objects": [bad]})
    assert res2["rejectedObjects"][0]["reasonCodes"] == ["SCHEMA_INVALID"]


def test_jsonl_parse_failure():
    obj = make_object([])
    obj["content"] = '{"id":"a"}\nnot-json\n'
    obj["crc32c"] = crc32c_hex(obj["content"].encode())
    res = handle({"policy": POLICY, "objects": [obj]})
    codes = res["rejectedObjects"][0]["reasonCodes"]
    assert "JSONL_INVALID" in codes
    assert "SCHEMA_INVALID" in codes


def test_duplicate_revision_tiebreak():
    r_low = make_row(id="zzz", revision=1)
    r_high = make_row(id="aaa", revision=9)
    body = {"policy": POLICY, "objects": [make_object([r_low, r_high])]}
    res = handle(body)
    ids = {row["id"] for row in res["splits"]["train"]}
    rejected = [e["id"] for e in res["rejectedRows"]]
    assert rejected == ["zzz"]
    assert "aaa" in ids and "zzz" not in ids

    # equal revisions -> smallest UTF-8 id kept
    b2 = make_row(id="b", revision=5)
    a2 = make_row(id="a", revision=5)
    res2 = handle({"policy": POLICY, "objects": [make_object([b2, a2])]})
    rejected2 = [e["id"] for e in res2["rejectedRows"]]
    assert rejected2 == ["b"]


def test_policy_invalid_rejects_all_retained():
    row = make_row()
    policy = dict(POLICY)
    policy["contaminationThreshold"] = 1.5
    res = handle({"policy": policy, "objects": [make_object([row])]})
    assert res["rejectedRows"] == [{"id": "r1", "reasonCodes": ["POLICY_INVALID"]}]
    empty = hashlib.sha256(b"").hexdigest()
    assert res["digests"] == {"train": empty, "validation": empty, "test": empty}
    assert res["lineage"] != []  # object itself was accepted


def test_out_of_window():
    row = make_row(event_time="2025-01-01T00:00:00Z")
    res = handle({"policy": POLICY, "objects": [make_object([row])]})
    assert res["rejectedRows"] == [{"id": "r1", "reasonCodes": ["OUT_OF_WINDOW"]}]

    # inclusive boundaries
    edge = make_row(event_time="2026-01-01T00:00:00Z")
    res2 = handle({"policy": POLICY, "objects": [make_object([edge])]})
    assert res2["rejectedRows"] == []


def test_unicode_whitespace_collapse():
    row = make_row(entity="A\u00a0\u00a0B", text="x \n y")
    res = handle({"policy": POLICY, "objects": [make_object([row])]}
                 )
    line = res["splits"]["train"][0]
    assert line["entity"] == "a b"
    assert line["text"] == "x y"


def test_bucket_assignment_deterministic():
    rows = []
    for i in range(60):
        rows.append(make_row(id=f"id{i:03d}", entity=f"ent-{i}", revision=1, text=f"unique text {i}"))
    res = handle({"policy": POLICY, "objects": [make_object(rows)]})
    all_rows = (
        list(res["splits"]["train"])
        + list(res["splits"]["validation"])
        + list(res["splits"]["test"])
    )
    assert len(all_rows) == 60

    def bucket(entity):
        h = hashlib.sha256(entity.encode("utf-8")).digest()
        return h[0] % 10

    for row in res["splits"]["train"]:
        assert bucket(row["entity"]) <= 5
    for row in res["splits"]["validation"]:
        assert 6 <= bucket(row["entity"]) <= 7
    for row in res["splits"]["test"]:
        assert bucket(row["entity"]) >= 8


def test_contamination_rejects_val_test_but_not_identical_train_pairing():
    def bucket(entity):
        return hashlib.sha256(entity.encode()).digest()[0] % 10

    train_entity = next(e for i in range(1000) if (e := f"tr{i}") and bucket(e) <= 5)
    val_entity = next(e for i in range(1000) if (e := f"v{i}") and 6 <= bucket(e) <= 7)
    train_row = make_row(id="t1", entity=train_entity, text="the quick brown fox")
    val_row = make_row(id=val_entity, entity=val_entity, text="The quick! BROWN fox")
    res = handle(
        {
            "policy": dict(POLICY, contaminationThreshold=0.8),
            "objects": [make_object([train_row, val_row])],
        }
    )
    rejected = {e["id"]: e["reasonCodes"] for e in res["rejectedRows"]}
    assert rejected.get(val_entity) == ["TRAIN_CONTAMINATION"]
    assert "t1" not in rejected


def test_missing_policy_is_400_shape():
    import pytest

    from app.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        handle({"objects": []})
    with pytest.raises(InvalidInput):
        handle({"policy": POLICY, "objects": "nope"})


def test_lineage_sorted_and_supplied_values_kept():
    o1 = make_object([make_row(id="a")], uri="gs://b/zebra")
    o2 = make_object([make_row(id="b")], uri="gs://b/apple")
    res = handle({"policy": POLICY, "objects": [o1, o2]})
    uris = [e["uri"] for e in res["lineage"]]
    assert uris == sorted(uris)
    assert res["lineage"][0] == {
        "uri": "gs://b/apple",
        "generation": "7",
        "crc32c": o2["crc32c"],
        "schemaId": "training-v1",
    }


# ---- regression: exact JSONL_INVALID vs SCHEMA_INVALID partition ----


def test_garbage_only_file_is_jsonl_invalid_only():
    obj = make_object([])
    obj["content"] = "not json at all\nstill not json\n"
    obj["crc32c"] = crc32c_hex(obj["content"].encode())
    res = handle({"policy": POLICY, "objects": [obj]})
    assert res["rejectedObjects"][0]["reasonCodes"] == ["JSONL_INVALID"]


def test_blank_only_file_is_schema_invalid_only():
    obj = make_object([])
    obj["content"] = "   \n\t\n"
    obj["crc32c"] = crc32c_hex(obj["content"].encode())
    res = handle({"policy": POLICY, "objects": [obj]})
    assert res["rejectedObjects"][0]["reasonCodes"] == ["SCHEMA_INVALID"]


def test_mixed_parse_and_shape_failures_emit_both():
    good = json.dumps(make_row())
    obj = make_object([])
    obj["content"] = good + "\n{broken\n"
    obj["crc32c"] = crc32c_hex(obj["content"].encode())
    res = handle({"policy": POLICY, "objects": [obj]})
    assert set(res["rejectedObjects"][0]["reasonCodes"]) == {"JSONL_INVALID"}


def test_wrong_shape_rows_are_schema_invalid():
    obj = make_object([])
    obj["content"] = json.dumps({"id": "a", "entity": "e", "eventTime": "2026-01-01T00:00:00Z"}) + "\n"
    obj["crc32c"] = crc32c_hex(obj["content"].encode())
    res = handle({"policy": POLICY, "objects": [obj]})
    assert res["rejectedObjects"][0]["reasonCodes"] == ["SCHEMA_INVALID"]
