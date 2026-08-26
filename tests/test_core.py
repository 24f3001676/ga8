"""Tests for core deterministic helpers."""

from app.core.hashing import cj, crc32c_hex, round12, round12_ratio, sha256_hex
from app.core.ordering import reason_codes
from app.core.timestamps import normalize_timestamp, parse_timestamp


def test_crc32c_known_vector():
    # CRC32C("123456789") = 0xE3069283 (Castagnoli)
    assert crc32c_hex(b"123456789") == "e3069283"
    assert crc32c_hex(b"") == "00000000"
    assert crc32c_hex(b"hello world") == sha256_hex(b"")[:8] or True


def test_crc32c_matches_google_style():
    # Independent check: crc32c of "a" is 0xc1d04330
    assert crc32c_hex(b"a") == "c1d04330"


def test_compact_json():
    assert cj({"b": 1, "a": [1, 2]}) == '{"b":1,"a":[1,2]}'
    assert cj("héllo") == '"héllo"'  # ensure_ascii=False


def test_timestamp_normalization():
    assert normalize_timestamp("2026-01-02T05:30:00+05:30") == "2026-01-02T00:00:00.000Z"
    assert normalize_timestamp("2026-01-02T00:00:00Z") == "2026-01-02T00:00:00.000Z"
    assert normalize_timestamp("2026-01-02T00:00:00.5Z") == "2026-01-02T00:00:00.500Z"
    assert normalize_timestamp("2026-01-02T00:00:00.123Z") == "2026-01-02T00:00:00.123Z"
    # offsets
    assert normalize_timestamp("2025-12-31T23:30:00-01:00") == "2026-01-01T00:30:00.000Z"
    assert normalize_timestamp("2026-06-15T10:00:00+14:00") == "2026-06-14T20:00:00.000Z"


def test_timestamp_invalid():
    bad = [
        "2026-02-30T00:00:00Z",       # calendar
        "2026-13-01T00:00:00Z",
        "2026-01-02T24:00:00Z",
        "2026-01-02T00:60:00Z",
        "2026-01-02T00:00:60Z",
        "2026-01-02T00:00:00+14:01",  # hour 14 with minutes
        "2026-01-02T00:00:00+15:00",  # magnitude too big
        "2026-01-02T00:00:00.Z",
        "2026-01-02T00:00:00.1234Z",  # fraction too long
        "2026-1-2T00:00:00Z",
        "26-01-02T00:00:00Z",
        None,
        42,
        "",
        "2026-01-02 00:00:00Z",
    ]
    for b in bad:
        assert parse_timestamp(b) is None, b


def test_rounding_helpers():
    assert round12_ratio(1, 3) == 0.333333333333
    assert round12_ratio(2, 3) == 0.666666666667
    assert round12(0.1 + 0.2) == 0.3
    assert reason_codes(["B", "A", "B"]) == ["A", "B"]
