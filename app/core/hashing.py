"""Deterministic hashing and canonical JSON serialization helpers."""

import hashlib
import json
import math
from fractions import Fraction

MAX_SAFE_INTEGER = 9007199254740991  # 2**53 - 1


def cj(obj) -> str:
    """Compact JSON: no spaces, non-ASCII emitted directly, NaN/Infinity rejected."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def cj_bytes(obj) -> bytes:
    return cj(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_json(obj) -> str:
    return sha256_hex(cj_bytes(obj))


def utf8(s: str) -> bytes:
    return s.encode("utf-8")


def crc32c(data: bytes) -> int:
    """Castagnoli CRC32C (polynomial 0x82F63B78, reflected)."""
    poly = 0x82F63B78
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = (crc >> 1) ^ (poly & mask)
    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return format(crc32c(data), "08x")


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_finite(v) -> bool:
    return is_number(v) and math.isfinite(v)


def is_safe_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and -(2 ** 53) < v < 2 ** 53


def is_non_negative_safe_int(v) -> bool:
    return is_safe_int(v) and v >= 0


def is_positive_safe_int(v) -> bool:
    return is_safe_int(v) and v >= 1


def is_binary_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v in (0, 1)


def round12(value: float) -> float:
    """Round to 12 decimal places, half-up, deterministically."""
    from decimal import Decimal, ROUND_HALF_UP

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("non-finite value")
    d = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    q = d.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    f = float(q)
    if f == 0.0:
        f = 0.0  # normalize -0.0
    return f


def round12_ratio(numerator: int, denominator: int) -> float:
    """Exactly round numerator/denominator to 12 decimal places (half-up)."""
    from decimal import Decimal, ROUND_HALF_UP, localcontext

    frac = Fraction(numerator, denominator)
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(frac.numerator) / Decimal(frac.denominator)
    q = d.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    f = float(q)
    if f == 0.0:
        f = 0.0
    return f


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)
