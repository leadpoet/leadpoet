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
``(netuid, epoch, exact release commit, persist_snapshot)``, expire on the
same TTL the memory cache uses, and every failure here falls open to the normal
cold build.

Env knobs:
- ``RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE`` — set to ``0`` to disable.
- ``RESEARCH_LAB_ALLOCATION_HANDOFF_DIR`` — cache directory override.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from typing import Any, Optional

from leadpoet_canonical.allocation_handoff_v2 import (
    validate_allocation_handoff_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json

logger = logging.getLogger(__name__)

_FILE_PREFIX = "allocation_handoff_"
_SCHEMA_VERSION = "leadpoet.allocation_handoff_disk_cache.v2"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_CACHE_DOCUMENT_BYTES = 256 * 1024 * 1024


def _enabled() -> bool:
    return os.getenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE", "1") != "0"


def _cache_dir() -> str:
    configured = os.getenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DIR", "").strip()
    if configured:
        return configured
    return os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "leadpoet",
        "allocation_handoff",
    )


def _normalized_release_commit(value: str) -> str:
    commit = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("allocation handoff cache release commit is invalid")
    return commit


def _entry_path(
    netuid: int,
    epoch: int,
    persist_snapshot: bool,
    release_commit: str,
) -> str:
    suffix = "persist" if persist_snapshot else "readonly"
    commit = _normalized_release_commit(release_commit)
    return os.path.join(
        _cache_dir(),
        f"{_FILE_PREFIX}{int(netuid)}_{int(epoch)}_{commit}_{suffix}.json",
    )


def _validated_handoff(
    handoff: dict[str, Any],
    *,
    netuid: int,
    epoch: int,
    release_commit: str,
) -> dict[str, Any]:
    commit = _normalized_release_commit(release_commit)
    normalized = validate_allocation_handoff_v2(
        handoff,
        expected_epoch_id=int(epoch),
        expected_netuid=int(netuid),
    )
    graph = normalized["receipt_graph"]
    root_hash = str(graph["root_receipt_hash"])
    root = next(
        (
            receipt
            for receipt in graph["receipts"]
            if str(receipt.get("receipt_hash") or "") == root_hash
        ),
        None,
    )
    if not isinstance(root, dict) or str(root.get("commit_sha") or "").lower() != commit:
        raise ValueError("allocation handoff cache release commit differs")
    return normalized


def store_handoff(
    netuid: int,
    epoch: int,
    persist_snapshot: bool,
    release_commit: str,
    handoff: dict[str, Any],
    *,
    ttl_seconds: float,
) -> None:
    """Persist an assembled handoff; prune other-epoch entries. Fail-open."""

    if not _enabled():
        return
    tmp_path = ""
    try:
        commit = _normalized_release_commit(release_commit)
        normalized = _validated_handoff(
            handoff,
            netuid=int(netuid),
            epoch=int(epoch),
            release_commit=commit,
        )
        directory = _cache_dir()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        document = {
            "schema_version": _SCHEMA_VERSION,
            "netuid": int(netuid),
            "epoch": int(epoch),
            "persist_snapshot": bool(persist_snapshot),
            "release_commit": commit,
            "stored_at": time.time(),
            "ttl_seconds": float(ttl_seconds),
            "handoff_hash": sha256_json(normalized),
            "handoff": normalized,
        }
        path = _entry_path(
            int(netuid),
            int(epoch),
            persist_snapshot,
            commit,
        )
        fd, tmp_path = tempfile.mkstemp(
            prefix=".allocation_handoff_",
            suffix=".tmp",
            dir=directory,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(document, fh, separators=(",", ":"))
            if fh.tell() > _MAX_CACHE_DOCUMENT_BYTES:
                raise ValueError("allocation handoff cache document is oversized")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        tmp_path = ""
        os.chmod(path, 0o600)
        # The authority is only ever served for the current epoch; drop
        # entries from other epochs so the directory cannot grow unbounded.
        current_prefix = (
            f"{_FILE_PREFIX}{int(netuid)}_{int(epoch)}_{commit}_"
        )
        for name in os.listdir(directory):
            if not name.startswith(_FILE_PREFIX):
                continue
            if name.startswith(current_prefix):
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
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_handoff(
    netuid: int,
    epoch: int,
    persist_snapshot: bool,
    release_commit: str,
) -> Optional[dict[str, Any]]:
    """Return a fresh same-epoch handoff from disk, or None. Fail-open."""

    if not _enabled():
        return None
    try:
        commit = _normalized_release_commit(release_commit)
        path = _entry_path(
            int(netuid),
            int(epoch),
            persist_snapshot,
            commit,
        )
        if not os.path.exists(path):
            return None
        if os.path.getsize(path) > _MAX_CACHE_DOCUMENT_BYTES:
            raise ValueError("allocation handoff cache document is oversized")
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
        if (
            not isinstance(document, dict)
            or set(document)
            != {
                "schema_version",
                "netuid",
                "epoch",
                "persist_snapshot",
                "release_commit",
                "stored_at",
                "ttl_seconds",
                "handoff_hash",
                "handoff",
            }
            or document.get("schema_version") != _SCHEMA_VERSION
            or document.get("netuid") != int(netuid)
            or document.get("epoch") != int(epoch)
            or document.get("persist_snapshot") != bool(persist_snapshot)
            or document.get("release_commit") != commit
        ):
            return None
        stored_at = float(document.get("stored_at") or 0.0)
        ttl_seconds = float(document.get("ttl_seconds") or 0.0)
        age_seconds = time.time() - stored_at
        if (
            stored_at <= 0
            or ttl_seconds <= 0
            or not math.isfinite(stored_at)
            or not math.isfinite(ttl_seconds)
            or age_seconds < -300
            or age_seconds >= ttl_seconds
        ):
            return None
        handoff = document.get("handoff")
        if not isinstance(handoff, dict) or not handoff:
            return None
        if document.get("handoff_hash") != sha256_json(handoff):
            raise ValueError("allocation handoff cache hash differs")
        normalized = _validated_handoff(
            handoff,
            netuid=int(netuid),
            epoch=int(epoch),
            release_commit=commit,
        )
        logger.info(
            "allocation_handoff_disk_cache_hit netuid=%s epoch=%s "
            "persist_snapshot=%s release_commit=%s",
            netuid,
            epoch,
            persist_snapshot,
            commit,
        )
        return normalized
    except Exception as exc:  # fail-open: fall back to the cold build
        logger.warning(
            "allocation_handoff_disk_cache_load_failed epoch=%s error=%s",
            epoch,
            str(exc)[:200],
        )
        return None
