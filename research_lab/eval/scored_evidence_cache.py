"""Durable, global, day-scoped cache of company/intent scoring results.

The scoring judge is an LLM: asked to score the same companies against the
same ICP twice, it can drift. This cache pins the judgment so identical
inputs always yield identical scores. It is keyed on the full scoring input
(ICP + the exact companies + reference-vs-candidate flag), which captures the
same-company/same-intent-evidence requirement while remaining correct under
the scorer's per-call de-duplication (the whole company list is the key, so
de-dup outcomes are part of what is cached).

Durable: entries live in a JSON file per UTC day under a shared directory, so
they survive gateway pause/restart/resume. Global: every scoring worker
process reads and writes the same day file under a file lock, so a result
computed by one process is reused by all. Day-scoped: the file name carries
the UTC day and only today's file is consulted, so at 00:00 UTC scoring is
recomputed fresh alongside the day's new evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Mapping, Sequence

SCORING_CACHE_DIR_ENV = "RESEARCH_LAB_SCORING_CACHE_DIR"

# ICP fields that actually change scoring; volatile bookkeeping keys are
# excluded so the same substantive ICP hits the same cache entry.
_ICP_SCORING_FIELDS = (
    "industry",
    "sub_industry",
    "titles",
    "seniority",
    "employee_count",
    "location",
    "geography",
    "intent_signals",
    "required_attributes",
    "keywords",
    "description",
    "icp_text",
)


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def scoring_cache_key(
    icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]],
    is_reference_model: bool,
) -> str:
    """Stable key for one scoring call.

    Includes every ICP field that affects scoring, the exact companies (with
    their intent evidence, canonicalized and order-preserving because order
    drives de-dup), and the reference/candidate flag.
    """
    icp_part = {k: _canonical(icp.get(k)) for k in _ICP_SCORING_FIELDS if icp.get(k) is not None}
    payload = {
        "icp": icp_part,
        "companies": [_canonical(c) for c in companies],
        "ref": bool(is_reference_model),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ScoredEvidenceCache:
    """File-backed, lock-guarded, day-scoped scoring cache."""

    def __init__(self, cache_dir: str) -> None:
        self._dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _day_path(self, day: str) -> str:
        return os.path.join(self._dir, f"scores_{day}.json")

    def _lock_path(self, day: str) -> str:
        return os.path.join(self._dir, f"scores_{day}.lock")

    def _read(self, day: str) -> dict[str, Any]:
        path = self._day_path(day)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            return doc if isinstance(doc, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def get(self, key: str) -> list[dict[str, Any]] | None:
        record = self._read(_utc_day()).get(key)
        if isinstance(record, list):
            return record
        return None

    def put(self, key: str, breakdowns: Sequence[Mapping[str, Any]]) -> None:
        day = _utc_day()
        # Serialize concurrent writers across processes: hold the day lock for
        # a read-modify-write so no update is lost.
        import fcntl

        lock_path = self._lock_path(day)
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except Exception:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            doc = self._read(day)
            if key in doc:
                return
            doc[key] = [dict(b) for b in breakdowns]
            path = self._day_path(day)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            return
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


_INSTANCE: ScoredEvidenceCache | None = None
_INSTANCE_DIR = ""


def get_scored_evidence_cache() -> ScoredEvidenceCache | None:
    """Shared cache when RESEARCH_LAB_SCORING_CACHE_DIR is set, else None."""
    global _INSTANCE, _INSTANCE_DIR
    cache_dir = (os.getenv(SCORING_CACHE_DIR_ENV) or "").strip()
    if not cache_dir:
        return None
    if _INSTANCE is None or _INSTANCE_DIR != cache_dir:
        try:
            _INSTANCE = ScoredEvidenceCache(cache_dir)
            _INSTANCE_DIR = cache_dir
        except Exception:
            return None
    return _INSTANCE
