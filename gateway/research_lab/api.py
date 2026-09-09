"""Research Lab production gateway API.

The namespace is production-facing but inert by default. All mutating routes
require explicit Research Lab flags and write only Research Lab tables/events.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Mapping, Optional

import gzip
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response

from gateway.build_info import get_build_info
from gateway.qualification.utils.chain import (
    BITTENSOR_NETUID,
    ChainRegistrationUnavailable,
    check_hotkey_registration as chain_is_hotkey_registered,
    verify_hotkey_signature,
)
from gateway.utils.bans import is_hotkey_banned

from .allocations import build_research_lab_allocation_bundle
from .config import ResearchLabGatewayConfig
from leadpoet_canonical.constants import EPOCH_LENGTH
from gateway.research_lab import allocation_handoff_disk_cache


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/research-lab", tags=["research-lab"])

@router.get("/status")
async def research_lab_status(request: Request) -> dict[str, object]:
    config = ResearchLabGatewayConfig.from_env()
    public_status = config.public_status()
    return {
        "service": "leadpoet-research-lab-gateway",
        "status": "configured" if config.api_enabled else "disabled",
        **public_status,
    }


_TERMINAL_CANDIDATE_STATUSES = {"scored", "rejected", "failed"}


async def _allocation_epoch_guard_and_persistence(
    config: ResearchLabGatewayConfig,
    epoch: int,
    internal_key: Optional[str],
) -> bool:
    """Reject future epochs; return whether this request may persist a snapshot.

    Anonymous GETs are read-only: an unauthenticated caller could otherwise
    mint active snapshots for arbitrary epochs (future rows for four epochs
    ahead were found persisted this way), which contaminates paid-to-date
    accounting. Only the authenticated validator path persists, and only for
    the current epoch it is about to submit.
    """
    from gateway.research_lab.allocations import allocation_snapshot_persistence_decision
    from gateway.utils.epoch import get_current_epoch_id_async

    try:
        current_epoch = await get_current_epoch_id_async()
    except Exception as exc:  # noqa: BLE001 - chain lookup outage must not break reads
        # Fail safe, not open: without the chain epoch we cannot prove the
        # requested epoch isn't in the future, so serve the computation but
        # never persist. The validator retries next cycle once the chain
        # lookup recovers, so persistence is delayed, not lost.
        logger.warning(
            "research_lab_allocation_epoch_guard_degraded epoch=%s error=%s",
            int(epoch),
            str(exc)[:120],
        )
        return False
    return _allocation_persistence_for_known_current_epoch(
        config=config,
        current_epoch=int(current_epoch),
        requested_epoch=int(epoch),
        internal_key=internal_key,
    )


def _allocation_persistence_for_known_current_epoch(
    *,
    config: ResearchLabGatewayConfig,
    current_epoch: int,
    requested_epoch: int,
    internal_key: Optional[str],
) -> bool:
    """Apply the normal persistence policy to an already-resolved epoch."""

    from gateway.research_lab.allocations import allocation_snapshot_persistence_decision

    normalized_key = internal_key if isinstance(internal_key, str) else None
    decision = allocation_snapshot_persistence_decision(
        current_epoch=int(current_epoch),
        requested_epoch=int(requested_epoch),
        provided_key=normalized_key,
        configured_key=str(getattr(config, "internal_api_key", "") or ""),
        live_allocation_enabled=bool(config.reimbursements_enabled or config.weight_mutation_enabled),
    )
    if decision == "future_epoch":
        raise HTTPException(
            status_code=422,
            detail=(
                f"allocation epoch {int(requested_epoch)} is in the future "
                f"(current {int(current_epoch)})"
            ),
        )
    if decision == "key_not_configured":
        raise HTTPException(status_code=403, detail="Research Lab internal API key is not configured")
    if decision == "invalid_key":
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")
    return decision == "persist"


@router.get("/allocations/live/{epoch}")
async def get_research_lab_live_allocation(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    persist_snapshot = await _allocation_epoch_guard_and_persistence(
        config, int(epoch), x_leadpoet_internal_key
    )
    try:
        return await build_research_lab_allocation_bundle(
            config=config,
            epoch=int(epoch),
            netuid=BITTENSOR_NETUID,
            persist_snapshot=persist_snapshot,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # This endpoint has no build task to report its failures, so the reason
        # for a 500 has to be recorded here or it is lost with the response.
        logger.error(
            "research_lab_live_allocation_build_failed epoch=%s "
            "persist_snapshot=%s error_type=%s error=%s",
            int(epoch),
            bool(persist_snapshot),
            type(exc).__name__,
            str(exc)[:240],
            exc_info=True,
        )
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Research Lab allocation unavailable: {str(exc)[:200]}") from exc


# Assembling one attested allocation bundle is expensive: it reconstructs the
# full ancestry receipt graph (hundreds of chunked reads over the large
# receipt tables) and takes tens of seconds. The validator polls this endpoint
# inside a fixed on-chain submission window and retries on slow responses, so
# without coordination every retry — and every concurrent poll — launches a
# fresh rebuild. The rebuilds then contend for the database pool and the
# enclave, each one slowing past the validator's fetch timeout, so the
# validator never receives an allocation and its fail-closed guard blocks the
# weight submission for the whole epoch. The assembled bundle is deterministic
# for a given epoch (only a cosmetic bundle_id and generated_at timestamp vary
# between builds), so it is safe to build it at most once per epoch and serve
# every other caller from that result.
_AllocationCacheKey = tuple[int, bool]
_ALLOCATION_HANDOFF_CACHE: "OrderedDict[_AllocationCacheKey, tuple[float, dict[str, Any]]]" = (
    OrderedDict()
)
_ALLOCATION_BUILD_TASKS: dict[
    _AllocationCacheKey,
    asyncio.Task[dict[str, Any]],
] = {}
# Finney targets roughly 12-second blocks. Retain a successful handoff for
# longer than one 360-block epoch so a preparation completed before the weight
# window survives until block 300, including normal finality/block-time drift.
_ALLOCATION_CACHE_TTL_SECONDS = float(EPOCH_LENGTH * 15)
_ALLOCATION_CACHE_MAX_EPOCHS = 16


def _allocation_cache_key(epoch: int, persist_snapshot: bool) -> _AllocationCacheKey:
    return (int(epoch), bool(persist_snapshot))


def _allocation_handoff_cache_get(
    epoch: int,
    persist_snapshot: bool,
) -> Optional[dict[str, Any]]:
    keys = [_allocation_cache_key(epoch, persist_snapshot)]
    if not persist_snapshot:
        # persist_snapshot only gates the authorized emission-snapshot write;
        # it never changes the returned document. A read-only caller may
        # therefore reuse a persisted handoff, but an authenticated caller must
        # never reuse a read-only result and skip its required persistence.
        keys.append(_allocation_cache_key(epoch, True))

    for key in keys:
        entry = _ALLOCATION_HANDOFF_CACHE.get(key)
        if entry is None:
            continue
        expires_at, handoff = entry
        if time.monotonic() >= expires_at:
            _ALLOCATION_HANDOFF_CACHE.pop(key, None)
            continue
        _ALLOCATION_HANDOFF_CACHE.move_to_end(key)
        return handoff
    return None


def _allocation_handoff_cache_put(
    epoch: int,
    persist_snapshot: bool,
    handoff: dict[str, Any],
) -> None:
    key = _allocation_cache_key(epoch, persist_snapshot)
    _ALLOCATION_HANDOFF_CACHE[key] = (
        time.monotonic() + _ALLOCATION_CACHE_TTL_SECONDS,
        handoff,
    )
    _ALLOCATION_HANDOFF_CACHE.move_to_end(key)
    while len(_ALLOCATION_HANDOFF_CACHE) > _ALLOCATION_CACHE_MAX_EPOCHS:
        evicted, _ = _ALLOCATION_HANDOFF_CACHE.popitem(last=False)
        completed = _ALLOCATION_BUILD_TASKS.get(evicted)
        if completed is not None and completed.done():
            _ALLOCATION_BUILD_TASKS.pop(evicted, None)


def _allocation_cache_release_commit() -> str:
    return str(get_build_info().get("git_commit") or "").strip().lower()


def _allocation_build_task(
    *,
    config: "ResearchLabGatewayConfig",
    epoch: int,
    persist_snapshot: bool,
) -> asyncio.Task[dict[str, Any]]:
    key = _allocation_cache_key(epoch, persist_snapshot)
    task = _ALLOCATION_BUILD_TASKS.get(key)
    if task is not None:
        return task
    if not persist_snapshot:
        # If an authenticated validator already owns the build, share that
        # server-owned task. Its persistence was authorized by that validator;
        # the anonymous waiter neither initiates nor upgrades persistence.
        persisted = _ALLOCATION_BUILD_TASKS.get(
            _allocation_cache_key(epoch, True)
        )
        if persisted is not None:
            return persisted
    task = asyncio.create_task(
        _build_and_cache_attested_allocation(
            config=config,
            epoch=int(epoch),
            persist_snapshot=bool(persist_snapshot),
        )
    )
    _ALLOCATION_BUILD_TASKS[key] = task

    def clear(completed: asyncio.Task[dict[str, Any]]) -> None:
        if _ALLOCATION_BUILD_TASKS.get(key) is completed:
            _ALLOCATION_BUILD_TASKS.pop(key, None)
        if completed.cancelled():
            logger.warning(
                "research_lab_allocation_build_cancelled epoch=%s persist_snapshot=%s",
                int(epoch),
                bool(persist_snapshot),
            )
            return
        error = completed.exception()
        if error is not None:
            logger.warning(
                "research_lab_allocation_build_failed epoch=%s "
                "persist_snapshot=%s error_type=%s error=%s",
                int(epoch),
                bool(persist_snapshot),
                type(error).__name__,
                str(error)[:240],
            )

    task.add_done_callback(clear)
    return task


async def _allocation_handoff_response(
    handoff: dict[str, Any],
    accept_encoding: Optional[str],
):
    """Serve the handoff gzip-compressed when the caller asks for it.

    The handoff is multi-MB, hash-dense JSON that compresses several-fold, and
    it is fetched inside the validator's bounded pre-submission budget — the
    dominant cost of a cold-epoch fetch should be the build, not the transfer.
    Callers that do not advertise gzip get the identity JSON exactly as
    before. Serialization + compression run in a worker thread so the response
    path never stalls the shared event loop.
    """

    if "gzip" not in str(accept_encoding or "").lower():
        return handoff

    def _encode() -> bytes:
        raw = json.dumps(handoff, separators=(",", ":")).encode("utf-8")
        return gzip.compress(raw, compresslevel=6)

    wire = await asyncio.to_thread(_encode)
    return Response(
        content=wire,
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
            "Vary": "Accept-Encoding",
        },
    )


def _research_lab_attested_allocation_config() -> ResearchLabGatewayConfig:
    config = ResearchLabGatewayConfig.from_env()
    _require_enabled(config.api_enabled, "Research Lab gateway API is disabled")
    _require_enabled(config.reports_enabled, "Research Lab reports are disabled")
    _require_enabled(config.shadow_bundles_enabled, "Research Lab report bundles are disabled")
    return config


async def _get_research_lab_attested_allocation_handoff(
    *,
    config: ResearchLabGatewayConfig,
    epoch: int,
    persist_snapshot: bool,
) -> dict[str, Any]:
    """Load or build one allocation handoff after persistence authorization."""

    cached_handoff = _allocation_handoff_cache_get(
        int(epoch),
        persist_snapshot,
    )
    if cached_handoff is not None:
        return cached_handoff
    # Warm-start after a process restart: the memory cache is wiped by every
    # gateway restart, and a restart between the block-180 prewarm and the
    # block-300 submission used to force a full cold rebuild (receipt-ancestry
    # reconstruction + fresh enclave attestation) inside the validator's 90s
    # fetch budget. The handoff is deterministic per epoch, the validator
    # re-validates it fail-closed, and any disk-cache failure falls open to
    # the normal build below.
    if _ALLOCATION_BUILD_TASKS.get(
        _allocation_cache_key(int(epoch), persist_snapshot)
    ) is None:
        disk_handoff = await asyncio.to_thread(
            allocation_handoff_disk_cache.load_handoff,
            int(BITTENSOR_NETUID),
            int(epoch),
            persist_snapshot,
            _allocation_cache_release_commit(),
        )
        if disk_handoff is not None:
            _allocation_handoff_cache_put(
                int(epoch),
                persist_snapshot,
                disk_handoff,
            )
            return disk_handoff
    # A concurrent persisted build can finish while this request yields to the
    # disk-cache lookup. Recheck memory before creating a second cold build.
    cached_handoff = _allocation_handoff_cache_get(
        int(epoch),
        persist_snapshot,
    )
    if cached_handoff is not None:
        return cached_handoff
    # Shield the complete build-and-cache task from client disconnects. The
    # previous lock only serialized live requests: when a validator timed out,
    # request cancellation released the lock before the finished authority
    # could be assembled and cached, so the next retry started over.
    built_handoff = await asyncio.shield(
        _allocation_build_task(
            config=config,
            epoch=int(epoch),
            persist_snapshot=persist_snapshot,
        )
    )
    return built_handoff


async def _get_research_lab_attested_allocation_for_resolved_current_epoch(
    *,
    epoch: int,
    current_epoch: int,
    internal_key: str,
) -> dict[str, Any]:
    """Build the pre-launch handoff using an exact epoch resolved by maintenance."""

    config = _research_lab_attested_allocation_config()
    persist_snapshot = _allocation_persistence_for_known_current_epoch(
        config=config,
        current_epoch=int(current_epoch),
        requested_epoch=int(epoch),
        internal_key=internal_key,
    )
    return await _get_research_lab_attested_allocation_handoff(
        config=config,
        epoch=int(epoch),
        persist_snapshot=persist_snapshot,
    )


@router.get("/allocations/attested/{epoch}")
async def get_research_lab_attested_allocation(
    epoch: int,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
    accept_encoding: Optional[str] = Header(default=None),
):
    """Return the unchanged live allocation plus its enclave-signed sidecar."""

    config = _research_lab_attested_allocation_config()
    # The guard rejects future epochs and decides snapshot persistence; it must
    # run on every request and is cheap relative to the bundle build.
    persist_snapshot = await _allocation_epoch_guard_and_persistence(
        config, int(epoch), x_leadpoet_internal_key
    )
    handoff = await _get_research_lab_attested_allocation_handoff(
        config=config,
        epoch=int(epoch),
        persist_snapshot=persist_snapshot,
    )
    return await _allocation_handoff_response(handoff, accept_encoding)


async def _build_and_cache_attested_allocation(
    *,
    config: "ResearchLabGatewayConfig",
    epoch: int,
    persist_snapshot: bool,
) -> dict[str, Any]:
    attestation: dict[str, Any] = {}
    try:
        bundle = await build_research_lab_allocation_bundle(
            config=config,
            epoch=int(epoch),
            netuid=BITTENSOR_NETUID,
            persist_snapshot=persist_snapshot,
            attestation_out=attestation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Research Lab attested allocation unavailable: {str(exc)[:200]}",
        ) from exc
    if attestation.get("status") != "matched":
        # The HTTPException raised below IS surfaced by the build task's
        # failure callback, but only generically (error_type=HTTPException).
        # The attestation status that caused it is not, so record it here:
        # a 503 inside the submission window should be greppable by the
        # status that produced it, not just by an access-log code.
        logger.error(
            "research_lab_attested_allocation_not_ready epoch=%s "
            "persist_snapshot=%s status=%s",
            int(epoch),
            bool(persist_snapshot),
            attestation.get("status", "unknown"),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Research Lab attested allocation is not ready: {attestation.get('status', 'unknown')}",
        )
    receipt = attestation.get("execution_receipt") or attestation.get("receipt")
    receipt_graph = attestation.get("receipt_graph")
    lineage_bindings = attestation.get("lineage_bindings")
    lineage_complete = attestation.get("lineage_complete")
    persistence = attestation.get("persistence")
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(receipt_graph, Mapping)
        or not isinstance(lineage_bindings, list)
        or lineage_complete is not True
        or not isinstance(persistence, Mapping)
    ):
        raise HTTPException(status_code=503, detail="Research Lab attested allocation receipt is incomplete")
    from leadpoet_canonical.allocation_handoff_v2 import (
        build_allocation_handoff_v2,
    )

    try:
        if receipt_graph.get("root_receipt_hash") != receipt.get("receipt_hash"):
            from gateway.research_lab.attested_v2_store import (
                load_receipt_graph_v2,
            )
            from leadpoet_canonical.attested_v2 import sha256_json

            receipt_graph = await load_receipt_graph_v2(
                str(receipt["receipt_hash"])
            )
            persistence = {
                "graph_hash": sha256_json(dict(receipt_graph)),
                "root_receipt_hash": str(receipt_graph["root_receipt_hash"]),
                "boot_count": len(receipt_graph["boot_identities"]),
                "receipt_count": len(receipt_graph["receipts"]),
                "transport_attempt_count": len(
                    receipt_graph["transport_attempts"]
                ),
                "host_operation_count": len(
                    receipt_graph["host_operations"]
                ),
            }
        handoff = build_allocation_handoff_v2(
            bundle=bundle,
            receipt_graph=receipt_graph,
            lineage_bindings=lineage_bindings,
            lineage_complete=lineage_complete,
            persistence=persistence,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Research Lab allocation V2 handoff is invalid",
        ) from exc
    # Only fully-assembled bundles are cached; failures above raise and are
    # retried by the next caller without poisoning the cache.
    _allocation_handoff_cache_put(
        int(epoch),
        persist_snapshot,
        handoff,
    )
    # Best-effort restart-surviving copy so a gateway restart mid-window can
    # warm-start instead of rebuilding cold (fail-open on any disk error).
    await asyncio.to_thread(
        allocation_handoff_disk_cache.store_handoff,
        int(BITTENSOR_NETUID),
        int(epoch),
        persist_snapshot,
        _allocation_cache_release_commit(),
        handoff,
        ttl_seconds=_ALLOCATION_CACHE_TTL_SECONDS,
    )
    return handoff


async def _verify_signed_miner(payload: object) -> None:
    signature_valid = verify_hotkey_signature(
        hotkey=payload.miner_hotkey,
        signature=payload.signature,
        message_data=payload.signed_payload(),
    )
    if not signature_valid:
        raise HTTPException(status_code=401, detail="invalid miner hotkey signature")

    is_banned, ban_reason = await is_hotkey_banned(payload.miner_hotkey)
    if is_banned:
        raise HTTPException(status_code=403, detail=f"hotkey is banned: {ban_reason}")

    try:
        is_registered, _role = await chain_is_hotkey_registered(
            payload.miner_hotkey
        )
    except ChainRegistrationUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="subnet registration check is temporarily unavailable; retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
    if not is_registered:
        raise HTTPException(status_code=403, detail="hotkey is not registered on this subnet")
def _require_enabled(enabled: bool, detail: str) -> None:
    if not enabled:
        raise HTTPException(status_code=403, detail=detail)
