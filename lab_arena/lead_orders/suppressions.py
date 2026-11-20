"""The suppression feed: hashed people that lead orders must never return.

The runner pushes hashes only; raw emails and LinkedIn URLs are never
accepted or stored. Miners and buyers hash their own data the same way to
check it against the feed:

* ``email_sha256``: sha256 hex of the trimmed, lowercased email.
* ``linkedin_sha256``: sha256 hex of the normalised profile URL. The URL is
  trimmed, ``http(s)://``, a two-letter country subdomain (``de.``) and
  ``www.`` are stripped, trailing slashes removed, and the rest lowercased:
  ``https://de.linkedin.com/in/Ada-One/`` becomes ``linkedin.com/in/ada-one``.
  This is leadpoet-site's ``normalizeContactLinkedin``.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

KINDS = ("email_sha256", "linkedin_sha256")
MAX_ENTRIES = 500  # two hashes each at most: within the store's 1,000 rows per call
PAGE_SIZE = 1000
FUTURE_SKEW_SECONDS = 300
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ADDED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


class SuppressionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def email_hash(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def normalise_linkedin_url(url: str) -> str:
    value = re.sub(r"^https?://", "", url.strip())
    value = re.sub(r"^[a-z]{2}\.(?=linkedin\.com)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^www\.", "", value)
    return re.sub(r"/+$", "", value).lower()


def linkedin_hash(url: str) -> str:
    return hashlib.sha256(normalise_linkedin_url(url).encode("utf-8")).hexdigest()


def _added_at(value: Any, now: float) -> str:
    if not isinstance(value, str) or not _ADDED_AT_RE.fullmatch(value):
        raise SuppressionError("added_at_invalid")
    parsed = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if parsed.timestamp() > now + FUTURE_SKEW_SECONDS or parsed.year < 2000:
        raise SuppressionError("added_at_invalid")
    return value


def rows_from_push(payload: Any, *, now: float) -> List[Dict[str, str]]:
    """Validate ``{"suppressions": [{"email_sha256"?, "linkedin_sha256"?, "added_at"}]}``.

    Fails closed on the whole push: any field other than the two hashes and
    ``added_at``, or any value that is not 64 lowercase hex characters (a raw
    email or URL included), refuses every row. Error codes never echo values.
    """

    if not isinstance(payload, dict) or set(payload) != {"suppressions"}:
        raise SuppressionError("suppressions_invalid")
    entries = payload["suppressions"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES:
        raise SuppressionError("suppressions_invalid")
    rows: List[Dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "added_at" not in entry:
            raise SuppressionError("suppression_invalid")
        extra = set(entry) - set(KINDS) - {"added_at"}
        if extra:
            raise SuppressionError("field_not_allowed")
        kinds = [kind for kind in KINDS if kind in entry]
        if not kinds:
            raise SuppressionError("hash_required")
        added_at = _added_at(entry["added_at"], now)
        for kind in kinds:
            value: Optional[Any] = entry[kind]
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise SuppressionError("hash_invalid")
            rows.append({"kind": kind, "hash": value, "added_at": added_at})
    return rows


__all__ = ["KINDS", "MAX_ENTRIES", "PAGE_SIZE", "SuppressionError", "email_hash", "linkedin_hash",
           "normalise_linkedin_url", "rows_from_push"]
