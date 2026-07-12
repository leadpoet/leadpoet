"""Canonical hashing helpers for Research Lab records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(data: Any) -> str:
    return sha256_text(canonical_json(data))


def normalize_snapshot_text(content: str) -> str:
    normalized = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_iso_z(value: str | datetime) -> str:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            match = re.fullmatch(
                r"(?P<prefix>.+T\d{2}:\d{2}:\d{2})\.(?P<fraction>\d+)(?P<suffix>[+-]\d{2}:\d{2})",
                text,
            )
            if match is None:
                raise
            fraction = match.group("fraction")[:6].ljust(6, "0")
            dt = datetime.fromisoformat(
                f"{match.group('prefix')}.{fraction}{match.group('suffix')}"
            )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
