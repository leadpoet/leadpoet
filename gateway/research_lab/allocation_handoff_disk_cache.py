"""Restart-surviving disk cache for the assembled attested allocation handoff.

The in-memory handoff cache in ``gateway.research_lab.api`` is wiped by every
gateway process restart. A restart between the block-180 prewarm and the
block-300 weight submission therefore forced a full cold rebuild — receipt
ancestry reconstruction plus a fresh enclave attestation — inside the
validator's 90s fetch budget, which is exactly the window where a slow build
becomes a missed weight set.

The allocation authority is deterministic for an epoch (see the cache comment
in ``api.py``), so the fully-assembled handoff can be cached on local disk and
served after a process restart. This is a read-through cache only: it is never
an alternative authority (the validator re-validates the handoff and its
receipt bindings fail-closed), entries are keyed strictly by
``(epoch, persist_snapshot)``, expire on the same TTL the memory cache uses,
and every failure here falls open to the normal cold build.

Env knobs:
- ``RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE`` — set to ``0`` to disable.
- ``RESEARCH_LAB_ALLOCATION_HANDOFF_DIR`` — cache directory override.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FILE_PREFIX = "allocation_handoff_"


def _enabled() -> bool:
    return os.getenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE", "1") != "0"


def _cache_dir() -> str:
    configured = os.getenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DIR", "").strip()
    if configured:
        return configured
    return os.path.join(tempfile.gettempdir(), "leadpoet_allocation_handoff")


def _entry_path(epoch: int, persist_snapshot: bool) -> str:
    suffix = "persist" if persist_snapshot else "readonly"
    return os.path.join(
        _cache_dir(), f"{_FILE_PREFIX}{int(epoch)}_{suffix}.json"
    )


def store_handoff(
    epoch: int,
    persist_snapshot: bool,
    handoff: dict[str, Any],
    *,
    ttl_seconds: float,
) -> None:
    """Persist an assembled handoff; prune other-epoch entries. Fail-open."""

    if not _enabled():
        return
    try:
        directory = _cache_dir()
        os.makedirs(directory, exist_ok=True)
        document = {
            "schema": "leadpoet.allocation_handoff_disk_cache.v1",
            "epoch": int(epoch),
            "persist_snapshot": bool(persist_snapshot),
            "stored_at": time.time(),
            "ttl_seconds": float(ttl_seconds),
            "handoff": handoff,
        }
        path = _entry_path(epoch, persist_snapshot)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, separators=(",", ":"))
        os.replace(tmp_path, path)
        # The authority is only ever served for the current epoch; drop
        # entries from other epochs so the directory cannot grow unbounded.
        for name in os.listdir(directory):
            if not name.startswith(_FILE_PREFIX):
                continue
            if name.startswith(f"{_FILE_PREFIX}{int(epoch)}_"):
                continue
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass
    except Exception as exc:  # fail-open: disk cache is best-effort only
        logger.warning(
            "allocation_handoff_disk_cache_store_failed epoch=%s error=%s",
            epoch,
            str(exc)[:200],
        )


def load_handoff(
    epoch: int,
    persist_snapshot: bool,
) -> Optional[dict[str, Any]]:
    """Return a fresh same-epoch handoff from disk, or None. Fail-open."""

    if not _enabled():
        return None
    try:
        path = _entry_path(epoch, persist_snapshot)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
        if (
            not isinstance(document, dict)
            or document.get("schema")
            != "leadpoet.allocation_handoff_disk_cache.v1"
            or document.get("epoch") != int(epoch)
            or document.get("persist_snapshot") is not bool(persist_snapshot)
        ):
            return None
        stored_at = float(document.get("stored_at") or 0.0)
        ttl_seconds = float(document.get("ttl_seconds") or 0.0)
        if stored_at <= 0 or time.time() - stored_at >= ttl_seconds:
            return None
        handoff = document.get("handoff")
        if not isinstance(handoff, dict) or not handoff:
            return None
        logger.info(
            "allocation_handoff_disk_cache_hit epoch=%s persist_snapshot=%s",
            epoch,
            persist_snapshot,
        )
        return handoff
    except Exception as exc:  # fail-open: fall back to the cold build
        logger.warning(
            "allocation_handoff_disk_cache_load_failed epoch=%s error=%s",
            epoch,
            str(exc)[:200],
        )
        return None
