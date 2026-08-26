"""Strict timestamp parsing/canonicalization per the specification grammar.

Accepted: YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm), fraction 1-3 digits,
valid calendar values, offset magnitude at most 14:00 (hour 14 => minutes 00).
"""

import re
from datetime import datetime, timedelta, timezone

_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$"
)


def parse_timestamp(value) -> "datetime | None":
    """Return an aware UTC datetime, or None when invalid."""
    if not isinstance(value, str):
        return None
    m = _TS_RE.match(value)
    if m is None:
        return None
    year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(7)
    micro = int(frac.ljust(3, "0")) * 1000 if frac else 0
    off = m.group(8)
    if off == "Z":
        tz = timezone.utc
    else:
        sign = -1 if off[0] == "-" else 1
        oh = int(off[1:3])
        om = int(off[4:6])
        if om > 59:
            return None
        if oh > 14:
            return None
        if oh == 14 and om != 0:
            return None
        tz = timezone(sign * timedelta(hours=oh, minutes=om))
    try:
        dt = datetime(year, month, day, hour, minute, second, micro, tzinfo=tz)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc)


def is_valid_timestamp(value) -> bool:
    return parse_timestamp(value) is not None


def utc_norm(dt: "datetime") -> str:
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond // 1000:03d}Z"


def normalize_timestamp(value) -> "str | None":
    dt = parse_timestamp(value)
    if dt is None:
        return None
    return utc_norm(dt)
