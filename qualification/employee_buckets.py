"""LinkedIn employee-count bucket helpers shared by competition scoring."""

from __future__ import annotations

import re
from typing import Any, Sequence


LINKEDIN_EMPLOYEE_BUCKETS = (
    "0-1",
    "2-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001+",
)
GENERATED_EMPLOYEE_BUCKETS = LINKEDIN_EMPLOYEE_BUCKETS[1:]
DEFAULT_EMPLOYEE_BUCKET = "51-200"
DEFAULT_EMPLOYEE_BUCKET_RADIUS = 2

LEGACY_EMPLOYEE_BUCKET_MAP = {
    "1-10": "2-10",
    "10-50": "11-50",
    "50-200": "51-200",
    "200-500": "201-500",
    "500-1000": "501-1,000",
    "501-1000": "501-1,000",
    "1000-5000": "1,001-5,000",
    "1001-5000": "1,001-5,000",
    "5000-10000": "5,001-10,000",
    "5001-10000": "5,001-10,000",
    "5000+": "5,001-10,000",
    "10000+": "10,001+",
    "10001+": "10,001+",
}

_OBSERVED_INTERVALS = (
    (1, "0-1"),
    (10, "2-10"),
    (50, "11-50"),
    (200, "51-200"),
    (500, "201-500"),
    (1_000, "501-1,000"),
    (5_000, "1,001-5,000"),
    (10_000, "5,001-10,000"),
)


def normalize_employee_count_bucket(
    value: Any, *, default: str | None = DEFAULT_EMPLOYEE_BUCKET
) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            normalized = normalize_employee_count_bucket(item, default=None)
            if normalized:
                return normalized
        return str(default or "")
    raw = " ".join(str(value or "").strip().split())
    if raw in LINKEDIN_EMPLOYEE_BUCKETS:
        return raw
    cleaned = (
        raw.lower()
        .replace("employees", "")
        .replace("employee", "")
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )
    for bucket in LINKEDIN_EMPLOYEE_BUCKETS:
        if cleaned == bucket.lower().replace(",", "").replace(" ", ""):
            return bucket
    return str(
        LEGACY_EMPLOYEE_BUCKET_MAP.get(cleaned)
        or LEGACY_EMPLOYEE_BUCKET_MAP.get(raw)
        or default
        or ""
    )


def normalize_observed_employee_count_bucket(
    value: Any, *, default: str | None = None
) -> str:
    if isinstance(value, bool):
        return str(default or "")
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        if len(value) > 5:
            return "10,001+"
        count = int(value)
    else:
        return str(default or "")
    if count < 0:
        return str(default or "")
    for maximum, bucket in _OBSERVED_INTERVALS:
        if count <= maximum:
            return bucket
    return "10,001+"


def expand_employee_count_buckets(
    primary_bucket: Any,
    *,
    radius: int = DEFAULT_EMPLOYEE_BUCKET_RADIUS,
    all_buckets: bool = False,
) -> list[str]:
    primary = normalize_employee_count_bucket(primary_bucket)
    bucket_space: Sequence[str] = (
        LINKEDIN_EMPLOYEE_BUCKETS
        if primary == "0-1"
        else GENERATED_EMPLOYEE_BUCKETS
    )
    if all_buckets:
        return list(bucket_space)
    if primary not in bucket_space:
        primary = DEFAULT_EMPLOYEE_BUCKET
    index = list(bucket_space).index(primary)
    size = max(0, int(radius))
    return list(bucket_space[max(0, index - size) : index + size + 1])


def normalize_employee_count_buckets(
    value: Any,
    *,
    primary_bucket: Any,
    radius: int = DEFAULT_EMPLOYEE_BUCKET_RADIUS,
    all_buckets: bool = False,
    expand_single: bool = True,
) -> list[str]:
    primary = normalize_employee_count_bucket(primary_bucket)
    if isinstance(value, str):
        raw_items = [
            item.strip()
            for item in re.split(r"\s*(?:\||;|\bor\b)\s*", value, flags=re.I)
            if item.strip()
        ]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        raw_items = []
    explicit: list[str] = []
    for item in raw_items:
        bucket = normalize_employee_count_bucket(item, default=None)
        if bucket and bucket not in explicit:
            explicit.append(bucket)
    if not explicit or (len(explicit) == 1 and expand_single):
        explicit = expand_employee_count_buckets(
            primary,
            radius=radius,
            all_buckets=all_buckets,
        )
    if primary not in explicit:
        explicit.append(primary)
    order = {bucket: index for index, bucket in enumerate(LINKEDIN_EMPLOYEE_BUCKETS)}
    return sorted(dict.fromkeys(explicit), key=lambda bucket: order.get(bucket, 999))
