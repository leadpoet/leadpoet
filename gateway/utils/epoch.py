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
from Leadpoet.utils.public_supabase import call_public_rpc
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
_validated_terminal_cutover_state = None
_validated_terminal_cutover_state_lock = threading.Lock()
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

    rows = None
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
            public_value = call_public_rpc(
                _CUTOVER_PUBLIC_STATE_RPC,
                timeout_seconds=30.0,
            )
            if not isinstance(public_value, list):
                raise SubnetEpochError(
                    "durable epoch namespace public authority response is invalid"
                )
            rows = list(public_value)
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
    except Exception as exc:
        raise SubnetEpochError(
            "durable epoch namespace state database is unavailable"
        ) from exc

    if rows is None:
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


def validate_epoch_runtime_lifecycle(
    *,
    cutover: SubnetEpochCutover | None = None,
    force_refresh: bool = True,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Validate runtime authority against the durable cutover singleton."""

    global _validated_terminal_cutover_state
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
    authority_key = (
        _cutover_authority_cache_scope(
            network=network,
            netuid=netuid,
        ),
        resolved_cutover.mapping_hash,
    )
    with _validated_terminal_cutover_state_lock:
        if (
            _validated_terminal_cutover_state is not None
            and _validated_terminal_cutover_state[0] == authority_key
        ):
            return dict(_validated_terminal_cutover_state[1])
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
        # Migration 101 makes stateful_active terminal: service_role has
        # SELECT-only table access and every SECURITY DEFINER transition
        # accepts only a pre-active state. Cache only after the exact mapping
        # has passed, so a transient PostgREST outage cannot revoke a proven
        # namespace while a mismatched row never becomes trusted.
        _validated_terminal_cutover_state = (authority_key, dict(state))
        return dict(state)


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

    global _validated_cutover_authority_hash
    authority_scope = _cutover_authority_cache_scope(
        network=network,
        netuid=netuid,
    )
    authority_key = (authority_scope, cutover.mapping_hash)
    if _validated_cutover_authority_hash == authority_key:
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


async def get_epoch_info_async(epoch_id: int) -> dict:
    """Async epoch information built from one authoritative state read."""

    snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
    current_epoch = _settlement_epoch_id(snapshot)
    if int(epoch_id) != current_epoch:
        raise SubnetEpochError(
            "historical epoch details require the persisted boundary ledger"
        )
    return get_current_epoch_info_from_snapshot(snapshot)
