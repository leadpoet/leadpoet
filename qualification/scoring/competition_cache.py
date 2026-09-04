"""Small day-scoped cache for shared competition judgments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from typing import Any, Mapping, Optional, Sequence


CACHE_DIR_ENV = "RESEARCH_LAB_SCORING_CACHE_DIR"
logger = logging.getLogger(__name__)


def scoring_cache_key(
    icp: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]],
    scoring_adapter_version: str,
) -> str:
    payload = {
        "icp": dict(icp),
        "companies": [dict(company) for company in companies],
        "scoring_adapter_version": str(scoring_adapter_version),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CompetitionScoringCache:
    def __init__(self, directory: str) -> None:
        self._directory = directory
        os.makedirs(directory, mode=0o700, exist_ok=True)

    def _path(self, suffix: str) -> str:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return os.path.join(self._directory, f"competition_scores_{day}.{suffix}")

    def _read(self) -> dict[str, Any]:
        try:
            with open(self._path("json"), "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning(
                "competition_scoring_cache_read_failed type=%s",
                type(exc).__name__,
            )
            return {}

    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        value = self._read().get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, Mapping) for item in value
        ):
            return None
        return [dict(item) for item in value]

    def put(self, key: str, breakdowns: Sequence[Mapping[str, Any]]) -> None:
        lock_fd = os.open(self._path("lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            document = self._read()
            if key in document:
                return
            document[key] = [dict(item) for item in breakdowns]
            path = self._path("json")
            temporary = f"{path}.tmp.{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle, separators=(",", ":"))
            os.replace(temporary, path)
        except Exception as exc:
            logger.warning(
                "competition_scoring_cache_write_failed type=%s",
                type(exc).__name__,
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


_INSTANCE: Optional[CompetitionScoringCache] = None
_INSTANCE_DIRECTORY = ""


def get_competition_scoring_cache() -> Optional[CompetitionScoringCache]:
    global _INSTANCE, _INSTANCE_DIRECTORY
    directory = str(os.getenv(CACHE_DIR_ENV) or "").strip()
    if not directory:
        return None
    if _INSTANCE is None or _INSTANCE_DIRECTORY != directory:
        try:
            _INSTANCE = CompetitionScoringCache(directory)
            _INSTANCE_DIRECTORY = directory
        except Exception as exc:
            logger.warning(
                "competition_scoring_cache_init_failed type=%s",
                type(exc).__name__,
            )
            return None
    return _INSTANCE
