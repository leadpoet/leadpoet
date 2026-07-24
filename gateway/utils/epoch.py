"""Exact-hash SN71 epoch authority for the LeadPoet gateway.

The Subtensor scheduler is read at one block hash and its official
``SubnetEpochIndex`` is mapped to the monotonic settlement ordinal established
by the immutable cutover manifest. There is no alternate epoch clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import os
import threading
import time

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.utils.subnet_epoch_archive import (
    validate_cutover_anchor_from_archive,
)

BITTENSOR_BLOCK_TIME_SECONDS = 12

# Network for epoch tracking (from environment variable)
_epoch_network = os.getenv("BITTENSOR_NETWORK", "finney")

# Async subtensor instance (injected at gateway startup)
_async_subtensor = None

# Sync subtensor instance (for quick block queries without subscription conflicts)
_sync_subtensor = None
_epoch_snapshot_lock = threading.Lock()
_validated_cutover_anchor_key = None
_validated_cutover_authority_hash = None
_validated_cutover_authority_at = 0.0
# A verified activation is re-checked on this cadence so an operator
# re-fencing or rolling the lifecycle back is honored by long-running
# processes instead of being masked by a never-expiring success cache.
_CUTOVER_AUTHORITY_REVALIDATE_SECONDS = 300.0
_cutover_state_cache = None
_cutover_state_cache_lock = threading.Lock()

_CUTOVER_STATE_TABLE = "research_lab_stateful_subnet_epoch_cutover_state_v1"
_CUTOVER_PUBLIC_STATE_RPC = (
    "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
)
_CUTOVER_LIFECYCLE_STATES = frozenset(
    {"legacy_open", "cutover_fenced", "stateful_staged", "stateful_active"}
)
_CUTOVER_STATE_CACHE_SECONDS = 2.0


def _load_cutover() -> SubnetEpochCutover:
    return load_subnet_epoch_cutover()


def _fixed_public_cutover_authority_enabled(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> bool:
    """Return whether this process is the production SN71 runtime.

    The repository's fixed public Supabase project is production authority for
    Finney subnet 71 only.
    """

    resolved_network = str(
        network if network is not None else os.getenv("BITTENSOR_NETWORK") or ""
    ).strip().lower()
    resolved_netuid = str(
        netuid if netuid is not None else os.getenv("BITTENSOR_NETUID") or ""
    ).strip()
    return resolved_network == "finney" and resolved_netuid == "71"


def _configured_cutover_service_authority_enabled() -> bool:
    """Return whether this process has an explicitly configured DB authority."""

    return bool(
        str(os.getenv("SUPABASE_URL") or "").strip()
        and str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    )


def _cutover_authority_cache_scope(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> tuple[str, ...]:
    """Return a non-secret key preventing cache reuse across authorities."""

    if _configured_cutover_service_authority_enabled():
        return ("configured_service", str(os.getenv("SUPABASE_URL") or ""))
    if _fixed_public_cutover_authority_enabled(
        network=network,
        netuid=netuid,
    ):
        return ("fixed_public", "finney", "71")
    return (
        "unconfigured",
        str(network if network is not None else os.getenv("BITTENSOR_NETWORK") or "")
        .strip()
        .lower(),
        str(netuid if netuid is not None else os.getenv("BITTENSOR_NETUID") or "")
        .strip(),
    )


def _read_cutover_state_from_db_sync(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Read and strictly normalize the singleton epoch namespace lifecycle."""
    supabase_url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role_key = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

    try:
        if supabase_url and service_role_key:
            from gateway.db.client import get_write_client

            client = get_write_client()
        elif _fixed_public_cutover_authority_enabled(
            network=network,
            netuid=netuid,
        ):
            # Validators and public auditors must not receive the service-role
            # secret. The cutover singleton contains no secrets and migration
            # 101 grants anon read-only access under RLS specifically for this
            # cross-runtime safety gate. Pin the public project authority rather
            # than trusting caller-provided endpoint/key overrides.
            from Leadpoet.utils.cloud_db import (
                SUPABASE_ANON_KEY as PUBLIC_SUPABASE_ANON_KEY,
                SUPABASE_URL as PUBLIC_SUPABASE_URL,
            )
            from gateway.db.client import create_http1_sync_client

            if not PUBLIC_SUPABASE_URL or not PUBLIC_SUPABASE_ANON_KEY:
                raise SubnetEpochError(
                    "durable epoch namespace public authority is unavailable"
                )
            client = create_http1_sync_client(
                PUBLIC_SUPABASE_URL,
                PUBLIC_SUPABASE_ANON_KEY,
            )
        else:
            raise SubnetEpochError(
                "durable epoch authority is not configured"
            )

        if supabase_url and service_role_key:
            result = (
                client
                .table(_CUTOVER_STATE_TABLE)
                .select(
                    "lifecycle_state,mapping_hash,last_legacy_epoch_id,"
                    "first_settlement_epoch_id"
                )
                .eq("singleton", True)
                .limit(2)
                .execute()
            )
        else:
            result = client.rpc(_CUTOVER_PUBLIC_STATE_RPC).execute()
    except Exception as exc:
        raise SubnetEpochError(
            "durable epoch namespace state database is unavailable"
        ) from exc

    rows = list(getattr(result, "data", None) or [])
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise SubnetEpochError(
            "durable epoch namespace singleton is missing or ambiguous"
        )
    row = dict(rows[0])
    lifecycle = row.get("lifecycle_state")
    if lifecycle not in _CUTOVER_LIFECYCLE_STATES:
        raise SubnetEpochError("durable epoch namespace lifecycle is invalid")
    first = row.get("first_settlement_epoch_id")
    last = row.get("last_legacy_epoch_id")
    if lifecycle == "legacy_open":
        if first is not None or last is not None or row.get("mapping_hash") is not None:
            raise SubnetEpochError("legacy-open epoch namespace state is invalid")
    elif (
        isinstance(first, bool)
        or not isinstance(first, int)
        or first < 1
        or isinstance(last, bool)
        or not isinstance(last, int)
        or last != first - 1
    ):
        raise SubnetEpochError("fenced epoch namespace ordinals are invalid")
    return row


def get_cutover_state(
    *,
    force_refresh: bool = False,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Return a short-lived exact copy of the durable namespace lifecycle."""

    global _cutover_state_cache
    now = time.monotonic()
    authority_scope = _cutover_authority_cache_scope(
        network=network,
        netuid=netuid,
    )
    with _cutover_state_cache_lock:
        if (
            not force_refresh
            and _cutover_state_cache is not None
            and _cutover_state_cache[0] == authority_scope
            and now - _cutover_state_cache[1] <= _CUTOVER_STATE_CACHE_SECONDS
        ):
            return dict(_cutover_state_cache[2])
        if network is None and netuid is None:
            state = _read_cutover_state_from_db_sync()
        else:
            state = _read_cutover_state_from_db_sync(
                network=network,
                netuid=netuid,
            )
        _cutover_state_cache = (authority_scope, now, dict(state))
        return dict(state)


async def get_cutover_state_async(
    *,
    force_refresh: bool = False,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    return await asyncio.to_thread(
        get_cutover_state,
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )

def validate_epoch_runtime_lifecycle(
    *,
    cutover: SubnetEpochCutover | None = None,
    force_refresh: bool = True,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Validate runtime authority against the durable cutover singleton."""

    resolved_cutover = cutover or _load_cutover()
    if not (
        _configured_cutover_service_authority_enabled()
        or _fixed_public_cutover_authority_enabled(
            network=network,
            netuid=netuid,
        )
    ):
        # Non-production runtimes remain bound to their locally validated
        # cutover manifest/archive anchor.
        return {
            "lifecycle_state": "stateful_manifest_only",
            "mapping_hash": resolved_cutover.mapping_hash,
            "last_legacy_epoch_id": resolved_cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": (
                resolved_cutover.first_settlement_epoch_id
            ),
        }
    state = get_cutover_state(
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )
    if (
        state.get("lifecycle_state") != "stateful_active"
        or state.get("mapping_hash") != resolved_cutover.mapping_hash
    ):
        raise SubnetEpochError(
            "stateful runtime does not match the active durable cutover"
        )
    return state


async def validate_epoch_runtime_lifecycle_async(
    *,
    cutover: SubnetEpochCutover | None = None,
    force_refresh: bool = True,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    return await asyncio.to_thread(
        validate_epoch_runtime_lifecycle,
        cutover=cutover,
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )


def _validate_cutover_authority_sync(
    cutover: SubnetEpochCutover,
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> None:
    """Require the configured mapping to exist in the receipt-backed ledger."""

    global _validated_cutover_authority_hash, _validated_cutover_authority_at
    authority_scope = _cutover_authority_cache_scope(
        network=network,
        netuid=netuid,
    )
    authority_key = (authority_scope, cutover.mapping_hash)
    if (
        _validated_cutover_authority_hash == authority_key
        and time.monotonic() - _validated_cutover_authority_at
        <= _CUTOVER_AUTHORITY_REVALIDATE_SECONDS
    ):
        return
    if not (
        _configured_cutover_service_authority_enabled()
        or _fixed_public_cutover_authority_enabled(
            network=network,
            netuid=netuid,
        )
    ):
        # Non-production stateful runtimes retain the manifest and archive
        # checks performed by the caller without consulting the fixed Finney
        # SN71 project.
        return
    state = get_cutover_state(
        force_refresh=True,
        network=network,
        netuid=netuid,
    )
    if (
        state.get("lifecycle_state") != "stateful_active"
        or state.get("mapping_hash") != cutover.mapping_hash
    ):
        raise SubnetEpochError(
            "configured cutover has not been explicitly activated after runtime verification"
        )

    supabase_url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role_key = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not supabase_url or not service_role_key:
        # The mapping hash is the canonical hash of the locally validated
        # manifest. SQL permits stateful_active only after the receipt-backed
        # cutover row and exact first initialization were staged atomically, so
        # the public singleton is sufficient for non-secret validator/auditor
        # startup binding. Gateway processes with service authority additionally
        # prove the exact immutable ledger row below.
        _validated_cutover_authority_hash = authority_key
        _validated_cutover_authority_at = time.monotonic()
        return

    from gateway.db.client import create_http1_sync_client

    client = create_http1_sync_client(supabase_url, service_role_key)
    result = (
        client.table("research_lab_stateful_subnet_epoch_cutovers_v1")
        .select("mapping_hash,manifest_doc")
        .eq("mapping_hash", cutover.mapping_hash)
        .limit(2)
        .execute()
    )
    rows = list(result.data or [])
    if len(rows) != 1 or rows[0].get("manifest_doc") != cutover.to_dict():
        raise SubnetEpochError(
            "configured cutover is absent from the receipt-backed authority ledger"
        )
    _validated_cutover_authority_hash = authority_key
    _validated_cutover_authority_at = time.monotonic()


def validate_stateful_cutover_authority(
    cutover: SubnetEpochCutover | None = None,
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> SubnetEpochCutover:
    """Validate and return the configured durable cutover authority."""

    resolved = cutover or _load_cutover()
    _validate_cutover_authority_sync(
        resolved,
        network=network,
        netuid=netuid,
    )
    return resolved


async def validate_stateful_cutover_authority_async(
    cutover: SubnetEpochCutover | None = None,
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> SubnetEpochCutover:
    return await asyncio.to_thread(
        validate_stateful_cutover_authority,
        cutover,
        network=network,
        netuid=netuid,
    )


def _read_subnet_epoch_snapshot_sync(*, finalized: bool = False) -> SubnetEpochSnapshot:
    global _validated_cutover_anchor_key
    if _sync_subtensor is None:
        raise SubnetEpochError(
            "Sync subtensor not initialized - call inject_async_subtensor() first"
        )
    netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
    with _epoch_snapshot_lock:
        snapshot = read_subnet_epoch_snapshot(
            _sync_subtensor,
            netuid=netuid,
            finalized=finalized,
        )
        cutover = _load_cutover()
        anchor_key = cutover.mapping_hash
        if _validated_cutover_anchor_key != anchor_key:
            validate_cutover_anchor_from_archive(cutover)
            _validated_cutover_anchor_key = anchor_key
        _validate_cutover_authority_sync(cutover)
        return snapshot


async def get_current_subnet_epoch_snapshot_async(
    *, finalized: bool = False
) -> SubnetEpochSnapshot:
    """Return one exact-hash official subnet epoch snapshot.

    This function never estimates state.  Callers that authorize admissions,
    rewards, settlement, or weights must fail closed if it raises.
    """

    return await asyncio.to_thread(
        _read_subnet_epoch_snapshot_sync,
        finalized=finalized,
    )


def get_current_subnet_epoch_snapshot(
    *, finalized: bool = False
) -> SubnetEpochSnapshot:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _read_subnet_epoch_snapshot_sync(finalized=finalized)
    raise RuntimeError(
        "Called get_current_subnet_epoch_snapshot() from async context; "
        "await get_current_subnet_epoch_snapshot_async() instead"
    )


def _settlement_epoch_id(snapshot: SubnetEpochSnapshot) -> int:
    return snapshot.settlement_epoch_id(_load_cutover())


def get_workflow_epoch_id(snapshot: SubnetEpochSnapshot) -> int:
    """Return the existing persistence/reward key for one chain snapshot.

    This is the monotonic settlement ordinal, not the official Bittensor epoch
    ID. Callers must use ``snapshot.subnet_epoch_index`` or
    ``snapshot.epoch_ref`` when they need the official chain identity.
    """

    return _settlement_epoch_id(snapshot)


def get_epoch_elapsed(snapshot: SubnetEpochSnapshot) -> int:
    return snapshot.epoch_block


def get_epoch_blocks_remaining(snapshot: SubnetEpochSnapshot) -> int:
    return snapshot.blocks_remaining


def _observed_datetime(snapshot: SubnetEpochSnapshot) -> datetime:
    observed = datetime.fromisoformat(snapshot.observed_at.replace("Z", "+00:00"))
    return observed.astimezone(timezone.utc).replace(tzinfo=None)


def get_current_epoch_times(
    snapshot: SubnetEpochSnapshot,
) -> tuple[datetime, datetime, datetime]:
    """Return current workflow start/end/close times from one observation.

    Times remain estimates because Subtensor schedules by blocks.  Identity and
    transition decisions use the on-chain index, never these wall-clock values.
    """

    observed = _observed_datetime(snapshot)
    elapsed = get_epoch_elapsed(snapshot)
    remaining = get_epoch_blocks_remaining(snapshot)
    start = observed - timedelta(
        seconds=elapsed * BITTENSOR_BLOCK_TIME_SECONDS
    )
    end = observed + timedelta(
        seconds=remaining * BITTENSOR_BLOCK_TIME_SECONDS
    )
    return start, end, end


def get_current_epoch_info_from_snapshot(
    snapshot: SubnetEpochSnapshot,
) -> dict:
    """Build current-epoch API information without performing another read."""

    workflow_epoch = get_workflow_epoch_id(snapshot)
    start, end, close = get_current_epoch_times(snapshot)
    remaining_blocks = get_epoch_blocks_remaining(snapshot)
    active = remaining_blocks > 0
    seconds_remaining = remaining_blocks * BITTENSOR_BLOCK_TIME_SECONDS
    result = {
        "epoch_id": workflow_epoch,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "close_time": close.isoformat(),
        "phase": "active" if active else "transition_pending",
        "is_active": active,
        "is_grace_period": False,
        "is_closed": False,
        "time_until_end": seconds_remaining,
        "time_until_close": seconds_remaining,
        "epoch_block": get_epoch_elapsed(snapshot),
        "blocks_remaining": remaining_blocks,
        "block": snapshot.current_block,
        "block_hash": snapshot.block_hash,
    }
    result.update(
        {
            "official_subnet_epoch_id": snapshot.subnet_epoch_index,
            "epoch_ref": snapshot.epoch_ref,
            "cutover_mapping_hash": _load_cutover().mapping_hash,
        }
    )
    return result


async def get_current_epoch_context_async(
    *, finalized: bool | None = None
) -> tuple[SubnetEpochSnapshot, int]:
    """Return one coherent chain snapshot and its workflow settlement key."""

    snapshot = await get_current_subnet_epoch_snapshot_async(
        finalized=True if finalized is None else finalized
    )
    return snapshot, get_workflow_epoch_id(snapshot)


async def get_current_epoch_admission_context_async():
    """Return finalized identity plus best-head timing for write admission.

    Finalized state remains the persistence/reward key authority.  Best state
    is only a fail-closed liveness veto so finality lag cannot admit work for
    epoch N after the live chain is due or has advanced to epoch N+1.
    """

    finalized = await get_current_subnet_epoch_snapshot_async(finalized=True)
    best = await get_current_subnet_epoch_snapshot_async(finalized=False)
    finalized_epoch = get_workflow_epoch_id(finalized)
    best_epoch = get_workflow_epoch_id(best)
    if (
        best_epoch != finalized_epoch
        or best.subnet_epoch_index != finalized.subnet_epoch_index
        or best.current_block < finalized.current_block
        or best.blocks_remaining <= 0
    ):
        raise SubnetEpochError(
            "best-head subnet epoch is no longer live for finalized admission"
        )
    return finalized, best, finalized_epoch


async def get_epoch_authority_status_async(
    *, finalized: bool = False
) -> dict:
    """Return one coherent public status document for shadow/cutover checks."""

    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=finalized)
    cutover = _load_cutover()
    status = snapshot.to_dict(cutover=cutover)
    status["official_subnet_epoch_id"] = snapshot.subnet_epoch_index
    await validate_epoch_runtime_lifecycle_async(
        cutover=cutover,
        force_refresh=True,
    )
    status.update(
        {
            "workflow_epoch_id": snapshot.settlement_epoch_id(cutover),
            "workflow_epoch_scheme": "leadpoet.settlement_ordinal.v1",
            "cutover_mapping_hash": cutover.mapping_hash,
        }
    )
    return status


def inject_async_subtensor(async_subtensor):
    """
    Inject async subtensor instance at gateway startup.
    
    Called from main.py lifespan to provide shared AsyncSubtensor instance.
    This eliminates memory leaks and HTTP 429 errors from repeated instance creation.
    
    Also creates a sync subtensor for quick block queries (avoids subscription conflicts).
    
    Args:
        async_subtensor: AsyncSubtensor instance from main.py lifespan
    
    Example:
        # In main.py lifespan:
        async with bt.AsyncSubtensor(network="finney") as async_sub:
            epoch_utils.inject_async_subtensor(async_sub)
    """
    global _async_subtensor, _sync_subtensor
    import bittensor as bt
    
    _async_subtensor = async_subtensor
    
    # Create sync subtensor for quick block queries (avoids WebSocket subscription conflicts)
    _sync_subtensor = bt.Subtensor(network=_async_subtensor.network)
    
    print(f"✅ AsyncSubtensor injected into epoch utils (network: {_async_subtensor.network})")
    print(f"✅ Sync subtensor created for block queries (avoids subscription conflicts)")


async def _get_current_block_async() -> int:
    """Return the block from one exact-hash official scheduler snapshot."""

    return (await get_current_subnet_epoch_snapshot_async()).current_block


def _get_current_block() -> int:
    """
    Get current block number (SYNC WRAPPER - prefer async version).
    
    DEPRECATED: This function creates a temporary event loop to run the async version.
    Only use this if you MUST call from synchronous context.
    
    For async contexts (FastAPI endpoints, background tasks), use _get_current_block_async() directly.
    
    Returns:
        Current block number
    
    Raises:
        RuntimeError: If called from within an async context (use _get_current_block_async instead)
        Exception: If async_subtensor not injected or query fails
    """
    import asyncio
    
    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        # We're in async context - this is an error
        raise RuntimeError(
            "Called _get_current_block() from async context. "
            "Use 'await _get_current_block_async()' instead. "
            "This is more efficient and doesn't create a new event loop."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # We're in sync context - create temp loop and run async version
            return asyncio.run(_get_current_block_async())
        else:
            # Error was from our check above - re-raise it
            raise


async def get_current_epoch_id_async() -> int:
    """Return the mapped settlement ID for the finalized official epoch."""

    return _settlement_epoch_id(
        await get_current_subnet_epoch_snapshot_async(finalized=True)
    )


async def get_current_subnet_epoch_id_async() -> int:
    """Return the official on-chain ``SubnetEpochIndex`` without translation."""

    return (await get_current_subnet_epoch_snapshot_async()).subnet_epoch_index


def get_current_epoch_id() -> int:
    """Return the mapped settlement ID for the finalized official epoch."""

    return _settlement_epoch_id(
        get_current_subnet_epoch_snapshot(finalized=True)
    )


def get_current_subnet_epoch_id() -> int:
    """Return the official on-chain ``SubnetEpochIndex`` without translation."""

    return get_current_subnet_epoch_snapshot().subnet_epoch_index


async def get_block_within_epoch_async() -> int:
    """Return the official position anchored by ``LastEpochBlock``."""

    return (
        await get_current_subnet_epoch_snapshot_async(finalized=True)
    ).epoch_block


def get_block_within_epoch() -> int:
    """Return the official position anchored by ``LastEpochBlock``."""

    return get_current_subnet_epoch_snapshot(finalized=True).epoch_block


async def get_epoch_start_time_async(epoch_id: int) -> datetime:
    """Return the estimated wall-clock start of the current official epoch."""

    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        raise SubnetEpochError(
            "historical epoch boundaries require the persisted boundary ledger"
        )
    return get_current_epoch_times(snapshot)[0]


def get_epoch_start_time(epoch_id: int) -> datetime:
    """Return the estimated wall-clock start of the current official epoch."""

    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        raise SubnetEpochError(
            "historical epoch boundaries require the persisted boundary ledger"
        )
    return get_current_epoch_times(snapshot)[0]


async def get_epoch_end_time_async(epoch_id: int) -> datetime:
    """
    Get UTC end time for given epoch ID (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_epoch_end_time() wrapper.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when the official epoch reaches its next boundary
    """
    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        raise SubnetEpochError(
            "historical epoch boundaries require the persisted boundary ledger"
        )
    return get_current_epoch_times(snapshot)[1]


def get_epoch_end_time(epoch_id: int) -> datetime:
    """
    Get UTC end time for given epoch ID (SYNC WRAPPER).
    
    DEPRECATED: Use get_epoch_end_time_async() from async contexts.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when the official epoch reaches its next boundary
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        raise SubnetEpochError(
            "historical epoch boundaries require the persisted boundary ledger"
        )
    return get_current_epoch_times(snapshot)[1]


async def get_epoch_close_time_async(epoch_id: int) -> datetime:
    """
    Get UTC close time for given epoch ID (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_epoch_close_time() wrapper.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when the official epoch closes
    """
    return await get_epoch_end_time_async(epoch_id)


def get_epoch_close_time(epoch_id: int) -> datetime:
    """
    Get UTC close time for given epoch ID (SYNC WRAPPER).
    
    DEPRECATED: Use get_epoch_close_time_async() from async contexts.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when the official epoch closes
    """
    return get_epoch_end_time(epoch_id)


async def is_epoch_active_async(epoch_id: int) -> bool:
    """
    Check if the official epoch is currently active (ASYNC VERSION).
    
    Use this from async contexts. For sync, use is_epoch_active() wrapper.
    
    During active phase:
    - Validators can fetch assigned leads
    - Validators can submit validation results (commit phase) anytime
    - New leads can be added to queue
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if validators can submit validation results
    
    Example:
        >>> current_epoch = await get_current_epoch_id_async()
        >>> await is_epoch_active_async(current_epoch)
        True
    """
    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    return (
        int(epoch_id) == _settlement_epoch_id(snapshot)
        and snapshot.blocks_remaining > 0
    )


def is_epoch_active(epoch_id: int) -> bool:
    """
    Check if the official epoch is currently active (SYNC WRAPPER).
    
    DEPRECATED: Use is_epoch_active_async() from async contexts.
    
    During active phase:
    - Validators can fetch assigned leads
    - Validators can submit validation results (commit phase) anytime
    - New leads can be added to queue
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if validators can submit validation results
    
    Example:
        >>> is_epoch_active(get_current_epoch_id())
        True
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    return (
        int(epoch_id) == _settlement_epoch_id(snapshot)
        and snapshot.blocks_remaining > 0
    )


def is_epoch_in_grace_period(epoch_id: int) -> bool:
    """
    Check if epoch is in grace period (DEPRECATED - no grace period).
    
    Grace period has been removed from the design. This function always returns False
    and is kept only for API compatibility with existing code.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Always returns False (no grace period in current design)
    
    Example:
        >>> is_epoch_in_grace_period(get_current_epoch_id())
        False  # No grace period exists
    """
    return False


async def is_epoch_closed_async(epoch_id: int) -> bool:
    """
    Check if an epoch precedes the current official mapped epoch.
    
    Use this from async contexts. For sync, use is_epoch_closed() wrapper.
    
    After epoch closes:
    - Consensus is computed
    - Reveals are required
    - No more submissions accepted
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if epoch is fully closed
    
    Example:
        >>> current = await get_current_epoch_id_async()
        >>> await is_epoch_closed_async(current - 1)
        True  # Previous epoch is closed
    """
    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    return int(epoch_id) < _settlement_epoch_id(snapshot)


def is_epoch_closed(epoch_id: int) -> bool:
    """
    Check if an epoch precedes the current official mapped epoch.
    
    DEPRECATED: Use is_epoch_closed_async() from async contexts.
    
    After epoch closes:
    - Consensus is computed
    - Reveals are required
    - No more submissions accepted
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if epoch is fully closed
    
    Example:
        >>> is_epoch_closed(get_current_epoch_id() - 1)
        True  # Previous epoch is closed
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    return int(epoch_id) < _settlement_epoch_id(snapshot)


def time_until_epoch_end(epoch_id: int) -> int:
    """
    Get seconds remaining until epoch validation phase ends.
    
    Useful for displaying countdown timers or scheduling tasks.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Seconds remaining (0 if already ended)
    
    Example:
        >>> time_until_epoch_end(get_current_epoch_id())
        3420  # 57 minutes remaining
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        return 0
    return snapshot.blocks_remaining * BITTENSOR_BLOCK_TIME_SECONDS


def time_until_epoch_close(epoch_id: int) -> int:
    """
    Get seconds remaining until epoch fully closes.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Seconds remaining (0 if already closed)
    
    Example:
        >>> time_until_epoch_close(get_current_epoch_id())
        3720  # 62 minutes remaining
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    if int(epoch_id) != _settlement_epoch_id(snapshot):
        return 0
    return snapshot.blocks_remaining * BITTENSOR_BLOCK_TIME_SECONDS


def get_epoch_phase(epoch_id: int) -> str:
    """
    Get current phase of the epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        "active" | "closed"
    
    Example:
        >>> get_epoch_phase(get_current_epoch_id())
        'active'
    """
    if is_epoch_active(epoch_id):
        return "active"
    else:
        return "closed"


def get_epoch_info(epoch_id: int) -> dict:
    """
    Get comprehensive information about an epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Dictionary with all epoch timing information
    
    Example:
        >>> info = get_epoch_info(100)
        >>> info
        {
            'epoch_id': 100,
            'start_time': '1970-01-06T00:00:00',
            'end_time': '1970-01-06T01:07:00',
            'close_time': '1970-01-06T01:12:00',
            'phase': 'closed',
            'is_active': False,
            'is_grace_period': False,
            'is_closed': True,
            'time_until_end': 0,
            'time_until_close': 0
        }
    """
    snapshot = get_current_subnet_epoch_snapshot(finalized=True)
    current_epoch = _settlement_epoch_id(snapshot)
    if int(epoch_id) != current_epoch:
        raise SubnetEpochError(
            "historical epoch details require the persisted boundary ledger"
        )
    return get_current_epoch_info_from_snapshot(snapshot)


async def get_epoch_info_async(epoch_id: int) -> dict:
    """Async epoch information built from one authoritative state read."""

    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    current_epoch = _settlement_epoch_id(snapshot)
    if int(epoch_id) != current_epoch:
        raise SubnetEpochError(
            "historical epoch details require the persisted boundary ledger"
        )
    return get_current_epoch_info_from_snapshot(snapshot)
