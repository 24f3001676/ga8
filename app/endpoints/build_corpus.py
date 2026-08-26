"""Assignment 1: Build an Immutable, Leakage-Safe Training Corpus."""

import json
import math
import re
import unicodedata

from app.core.errors import InvalidInput
from app.core.hashing import cj, crc32c_hex, is_safe_int, jaccard, sha256_hex
from app.core.ordering import reason_codes, utf8_key
from app.core.timestamps import normalize_timestamp, parse_timestamp

ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}
SCHEMA_ID = "training-v1"
URI_RE = re.compile(r"^gs://[^/]+/.+$")
DECIMAL_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

EMPTY_DIGEST = sha256_hex(b"")


def _canon_text(value: str) -> str:
    """NFKC, lowercase, trim, collapse Unicode whitespace to single ASCII spaces."""
    s = unicodedata.normalize("NFKC", value)
    s = s.lower()
    return " ".join(s.split())


def _word_set(text: str) -> set:
    out = set()
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return out


def _validate_row_shape(row) -> bool:
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != ROW_KEYS:
        return False
    for k in ("id", "entity", "eventTime", "text"):
        if not isinstance(row[k], str):
            return False
    rev = row["revision"]
    if not is_safe_int(rev) or rev < 0:
        return False
    if parse_timestamp(row["eventTime"]) is None:
        return False
    return True


def _evaluate_object(obj):
    """Return (objectReasonCodes, parsedRows). Rows only when object accepted."""
    if not isinstance(obj, dict):
        obj = {}
    codes = set()

    uri = obj.get("uri")
    uri_is_str = isinstance(uri, str)
    if not (uri_is_str and URI_RE.match(uri) is not None):
        codes.add("URI_INVALID")

    gen = obj.get("generation")
    fetched = obj.get("fetchedGeneration")

    def gen_valid(g):
        return isinstance(g, str) and DECIMAL_RE.match(g) is not None

    gens_valid = gen_valid(gen) and gen_valid(fetched)
    if not gens_valid:
        codes.add("GENERATION_INVALID")
    elif gen != fetched:
        codes.add("GENERATION_MISMATCH")

    crc = obj.get("crc32c")
    content = obj.get("content")
    content_is_str = isinstance(content, str)
    crc_valid = isinstance(crc, str) and CRC_RE.match(crc) is not None
    if not crc_valid:
        codes.add("CRC32C_INVALID")
    elif content_is_str:
        if crc32c_hex(content.encode("utf-8")) != crc:
            codes.add("CRC32C_MISMATCH")

    schema_ok = obj.get("schemaId") == SCHEMA_ID
    if not schema_ok or not content_is_str:
        codes.add("SCHEMA_INVALID")

    rows = []
    if content_is_str:
        saw_row = False
        parse_failed = False
        shape_failed = False
        for line in content.split("\n"):
            if line.strip() == "":
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                parse_failed = True
                continue
            saw_row = True
            if not _validate_row_shape(parsed):
                shape_failed = True
        if parse_failed:
            codes.add("JSONL_INVALID")
        if shape_failed or not saw_row:
            codes.add("SCHEMA_INVALID")
        if not parse_failed and not shape_failed and saw_row:
            for line in content.split("\n"):
                if line.strip() == "":
                    continue
                rows.append(json.loads(line))

    return sorted(codes), rows


def _policy_is_valid(policy) -> bool:
    if not isinstance(policy, dict):
        return False
    if parse_timestamp(policy.get("minTime")) is None:
        return False
    if parse_timestamp(policy.get("maxTime")) is None:
        return False
    threshold = policy.get("contaminationThreshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return False
    if not math.isfinite(threshold):
        return False
    if threshold < 0 or threshold > 1:
        return False
    return True


def _sort_rejected_objects(items):
    return sorted(
        items,
        key=lambda e: (
            e["uri"] is None,
            utf8_key(e["uri"]) if e["uri"] is not None else b"",
            cj(e).encode("utf-8"),
        ),
    )


def _sort_rejected_rows(items):
    return sorted(items, key=lambda e: (utf8_key(e["id"]), cj(e).encode("utf-8")))


def _sort_lineage(items):
    return sorted(items, key=lambda e: (utf8_key(e["uri"]), cj(e).encode("utf-8")))


def handle(body: dict) -> dict:
    if not isinstance(body.get("policy"), dict):
        raise InvalidInput()
    objects = body.get("objects")
    if not isinstance(objects, list):
        raise InvalidInput()

    rejected_objects = []
    lineage_entries = []
    retained = []

    for obj in objects:
        codes, rows = _evaluate_object(obj)
        uri_raw = obj.get("uri") if isinstance(obj, dict) else None
        uri_out = uri_raw if isinstance(uri_raw, str) else None
        if codes:
            rejected_objects.append({"uri": uri_out, "reasonCodes": reason_codes(codes)})
            continue
        lineage_entries.append(
            {
                "uri": obj.get("uri"),
                "generation": obj.get("generation"),
                "crc32c": obj.get("crc32c"),
                "schemaId": obj.get("schemaId"),
            }
        )
        for row in rows:
            retained.append(
                {
                    "id": row["id"],
                    "entity": _canon_text(row["entity"]),
                    "eventTime": normalize_timestamp(row["eventTime"]),
                    "revision": row["revision"],
                    "text": _canon_text(row["text"]),
                }
            )

    # Deduplicate by [entity,eventTime,text]: highest revision, then smallest id bytes.
    best = {}
    rejected_rows = []
    for r in retained:
        key = (r["entity"], r["eventTime"], r["text"])
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        if (r["revision"] > cur["revision"]) or (
            r["revision"] == cur["revision"] and utf8_key(r["id"]) < utf8_key(cur["id"])
        ):
            best[key] = r
            rejected_rows.append({"id": cur["id"], "reasonCodes": ["DUPLICATE"]})
        else:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["DUPLICATE"]})

    survivors = list(best.values())
    digests = {"train": EMPTY_DIGEST, "validation": EMPTY_DIGEST, "test": EMPTY_DIGEST}
    final_splits = {"train": [], "validation": [], "test": []}

    policy = body["policy"]

    def result():
        return {
            "splits": final_splits,
            "rejectedObjects": _sort_rejected_objects(rejected_objects),
            "rejectedRows": _sort_rejected_rows(rejected_rows),
            "digests": digests,
            "lineage": _sort_lineage(lineage_entries),
        }

    if not _policy_is_valid(policy):
        for r in survivors:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["POLICY_INVALID"]})
        return result()

    min_dt = parse_timestamp(policy["minTime"])
    max_dt = parse_timestamp(policy["maxTime"])
    threshold = float(policy["contaminationThreshold"])

    in_window = []
    for r in survivors:
        dt = parse_timestamp(r["eventTime"])
        if dt < min_dt or dt > max_dt:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
        else:
            in_window.append(r)

    def bucket_of(entity: str) -> int:
        import hashlib

        first_byte = hashlib.sha256(entity.encode("utf-8")).digest()[0]
        return first_byte % 10

    train_rows, val_rows, test_rows = [], [], []
    for r in in_window:
        b = bucket_of(r["entity"])
        if b <= 5:
            train_rows.append(r)
        elif b <= 7:
            val_rows.append(r)
        else:
            test_rows.append(r)

    train_wordsets = [_word_set(r["text"]) for r in train_rows]

    def contaminated(text: str) -> bool:
        ws = _word_set(text)
        return any(jaccard(ws, tws) >= threshold for tws in train_wordsets)

    groups = (("train", train_rows), ("validation", val_rows), ("test", test_rows))
    for group, rows_in in groups:
        kept = []
        for r in rows_in:
            if group != "train" and contaminated(r["text"]):
                rejected_rows.append({"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                kept.append(r)

        def row_key(r):
            row_json = {
                "id": r["id"],
                "entity": r["entity"],
                "eventTime": r["eventTime"],
                "revision": r["revision"],
                "text": r["text"],
            }
            return (utf8_key(r["id"]), cj(row_json).encode("utf-8"))

        kept.sort(key=row_key)
        lines = []
        for r in kept:
            row_json = {
                "id": r["id"],
                "entity": r["entity"],
                "eventTime": r["eventTime"],
                "revision": r["revision"],
                "text": r["text"],
            }
            final_splits[group].append(cj(row_json))
            lines.append(cj(row_json))
        payload = "".join(line + "\n" for line in lines).encode("utf-8")
        digests[group] = sha256_hex(payload)

    return result()
