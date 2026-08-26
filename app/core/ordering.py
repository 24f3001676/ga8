"""Deterministic ordering helpers (UTF-8 byte ordering, reason codes)."""

from app.core.hashing import cj


def utf8_key(s: str) -> bytes:
    return s.encode("utf-8")


def sort_by_utf8(items):
    return sorted(items, key=utf8_key)


def reason_codes(codes) -> "list[str]":
    """Sort and deduplicate reason codes by UTF-8 bytes."""
    seen = set()
    out = []
    for c in sorted(set(codes), key=utf8_key):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def entry_sort_key(entry: dict) -> bytes:
    """Sort entries by their compact JSON (tie-breaker)."""
    return cj(entry).encode("utf-8")
