"""Reusable request-validation helpers."""

from app.core.errors import InvalidInput
from app.core.hashing import (
    is_finite,
    is_non_negative_safe_int,
    is_positive_safe_int,
)
import re

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def require_object(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    return body


def require_member(obj: dict, key: str):
    if key not in obj:
        raise InvalidInput()
    return obj[key]


def is_hex64(v) -> bool:
    return isinstance(v, str) and HEX64_RE.match(v) is not None


def is_hex40(v) -> bool:
    return isinstance(v, str) and HEX40_RE.match(v) is not None


def valid_floor(v) -> bool:
    """Finite number in [0, 1]."""
    return is_finite(v) and 0.0 <= float(v) <= 1.0


def valid_non_negative_number(v) -> bool:
    return is_finite(v) and float(v) >= 0.0


__all__ = [
    "InvalidInput",
    "require_object",
    "require_member",
    "is_hex64",
    "is_hex40",
    "valid_floor",
    "valid_non_negative_number",
    "is_positive_safe_int",
    "is_non_negative_safe_int",
]
