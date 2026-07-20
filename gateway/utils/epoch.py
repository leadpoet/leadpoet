"""Epoch authority for the LeadPoet gateway.

Legacy deployments used globally aligned ``block // 360`` buckets.  Current
Subtensor uses a stateful per-subnet scheduler.  Stateful mode reads the entire
scheduler at one exact block hash and maps its official ``SubnetEpochIndex`` to
the monotonic settlement ordinal established by the cutover manifest.

``LEADPOET_EPOCH_MODE`` defaults to ``legacy_global_360_v1`` so schema/code can
be deployed and shadow-verified before an operator performs the coordinated
cutover.  ``stateful_v1`` fails closed unless a valid cutover manifest is
provided; there is deliberately no time-estimated official epoch fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import asyncio
import os
import threading
import time

from Leadpoet.utils.subnet_epoch import (
    LEGACY_EPOCH_MODE,
    STATEFUL_EPOCH_MODE,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    get_epoch_mode,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.utils.subnet_epoch_archive import (
    validate_cutover_anchor_from_archive,
)

# Configuration constants (must match validator/reward.py)
EPOCH_DURATION_BLOCKS = 360  # 72 minutes = 360 blocks × 12 sec
BITTENSOR_BLOCK_TIME_SECONDS = 12

# Network for epoch tracking (from environment variable)
_epoch_network = os.getenv("BITTENSOR_NETWORK", "finney")

# Block caching for resilient estimation (must match validator/reward.py)
_last_known_block = None
_last_known_block_time = None
_block_cache_lock = threading.Lock()

# Async subtensor instance (injected at gateway startup)
_async_subtensor = None

# Sync subtensor instance (for quick block queries without subscription conflicts)
_sync_subtensor = None
_epoch_snapshot_lock = threading.Lock()
_validated_cutover_anchor_key = None
_validated_cutover_authority_hash = None
_cutover_state_cache = None
_cutover_state_cache_lock = threading.Lock()
_cutover_state_table_observed = False

_CUTOVER_STATE_TABLE = "research_lab_stateful_subnet_epoch_cutover_state_v1"
_CUTOVER_PUBLIC_STATE_RPC = (
    "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
)
_CUTOVER_LIFECYCLE_STATES = frozenset(
    {"legacy_open", "cutover_fenced", "stateful_staged", "stateful_active"}
)
_CUTOVER_STATE_CACHE_SECONDS = 2.0


class LegacyEpochNamespaceFencedError(SubnetEpochError):
    """The durable cutover fence reserves the requested legacy ordinal."""


@dataclass(frozen=True)
class LegacyEpochObservation:
    """Minimal block observation preserving the pre-cutover cache path."""

    current_block: int
    observed_at: str
    block_hash: str | None = None

    def to_dict(self) -> dict:
        """Return the minimal serializable legacy startup hint."""

        return {
            "mode": LEGACY_EPOCH_MODE,
            "current_block": self.current_block,
            "observed_at": self.observed_at,
            "block_hash": self.block_hash,
        }


def _load_cutover() -> SubnetEpochCutover:
    return load_subnet_epoch_cutover()


def _legacy_open_state() -> dict:
    return {
        "lifecycle_state": "legacy_open",
        "mapping_hash": None,
        "last_legacy_epoch_id": None,
        "first_settlement_epoch_id": None,
    }


def _fixed_public_cutover_authority_enabled(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> bool:
    """Return whether this process is the production SN71 runtime.

    The repository's fixed public Supabase project is production authority for
    Finney subnet 71 only.  Testnet and other-subnet processes must never bind
    themselves to that singleton, because a production fence would otherwise
    stop an unrelated legacy runtime.
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

    from gateway.config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def _cutover_authority_cache_scope(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> tuple[str, ...]:
    """Return a non-secret key preventing cache reuse across authorities."""

    if _configured_cutover_service_authority_enabled():
        from gateway.config import SUPABASE_URL

        return ("configured_service", str(SUPABASE_URL))
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


def _missing_cutover_state_table(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "pgrst205" in message
        or (
            "pgrst202" in message
            and _CUTOVER_PUBLIC_STATE_RPC in message
        )
        or (
            (
                _CUTOVER_STATE_TABLE in message
                or _CUTOVER_PUBLIC_STATE_RPC in message
            )
            and ("not found" in message or "does not exist" in message)
        )
    )


def _read_cutover_state_from_db_sync(
    *,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Read and strictly normalize the singleton epoch namespace lifecycle.

    A code-first legacy rollout remains possible before migration 100 exists.
    Once this process has observed the table, its disappearance or any database
    outage fails closed instead of silently restoring legacy writes.
    """

    global _cutover_state_table_observed
    from gateway.config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

    try:
        if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
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
            from supabase import create_client

            if not PUBLIC_SUPABASE_URL or not PUBLIC_SUPABASE_ANON_KEY:
                raise SubnetEpochError(
                    "durable epoch namespace public authority is unavailable"
                )
            client = create_client(
                PUBLIC_SUPABASE_URL,
                PUBLIC_SUPABASE_ANON_KEY,
            )
        else:
            # Preserve the pre-migration legacy namespace for testnet and
            # other subnets.  Only Finney SN71 is governed by the fixed public
            # project; callers with another authority must configure it.
            return _legacy_open_state()

        if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
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
        if not _cutover_state_table_observed and _missing_cutover_state_table(exc):
            return _legacy_open_state()
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
    _cutover_state_table_observed = True
    return row


def get_legacy_epoch_namespace_state(
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


async def get_legacy_epoch_namespace_state_async(
    *,
    force_refresh: bool = False,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    return await asyncio.to_thread(
        get_legacy_epoch_namespace_state,
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )


def assert_legacy_epoch_namespace_open(
    epoch_id: int,
    *,
    force_refresh: bool = False,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Allow legacy writes below the fence and reject its reserved ordinal.

    The database trigger remains the final authority.  This application gate
    prevents normal gateway and Research Lab callers from repeatedly attempting
    writes that the durable pre-boundary fence must reject.
    """

    if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 0:
        raise SubnetEpochError("legacy epoch namespace identity is invalid")
    state = get_legacy_epoch_namespace_state(
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )
    lifecycle = state["lifecycle_state"]
    if lifecycle == "legacy_open":
        return state
    if lifecycle == "stateful_active":
        raise LegacyEpochNamespaceFencedError(
            "legacy epoch runtime is disabled after stateful activation"
        )
    first = int(state["first_settlement_epoch_id"])
    if epoch_id >= first:
        raise LegacyEpochNamespaceFencedError(
            "legacy epoch namespace is fenced at reserved settlement ordinal %d"
            % first
        )
    return state


async def assert_legacy_epoch_namespace_open_async(
    epoch_id: int,
    *,
    force_refresh: bool = False,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    return await asyncio.to_thread(
        assert_legacy_epoch_namespace_open,
        epoch_id,
        force_refresh=force_refresh,
        network=network,
        netuid=netuid,
    )


def validate_epoch_runtime_lifecycle(
    *,
    mode: str | None = None,
    epoch_id: int | None = None,
    cutover: SubnetEpochCutover | None = None,
    force_refresh: bool = True,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    """Validate one runtime mode against the durable cutover singleton.

    Stateful processes still perform the full receipt-backed authority check at
    startup.  This narrow, fresh check is suitable before each externally
    visible operation so a stale process cannot continue after lifecycle state
    changes underneath it.
    """

    resolved_mode = get_epoch_mode() if mode is None else mode
    if resolved_mode == LEGACY_EPOCH_MODE:
        if epoch_id is None:
            raise SubnetEpochError(
                "legacy runtime lifecycle validation requires epoch_id"
            )
        return assert_legacy_epoch_namespace_open(
            int(epoch_id),
            force_refresh=force_refresh,
            network=network,
            netuid=netuid,
        )
    if resolved_mode != STATEFUL_EPOCH_MODE:
        raise SubnetEpochError("epoch runtime mode is invalid")
    resolved_cutover = cutover or _load_cutover()
    if not (
        _configured_cutover_service_authority_enabled()
        or _fixed_public_cutover_authority_enabled(
            network=network,
            netuid=netuid,
        )
    ):
        # Explicit non-production stateful mode remains bound to its locally
        # validated cutover manifest/archive anchor, but not to the production
        # SN71 Supabase singleton.
        return {
            "lifecycle_state": "stateful_manifest_only",
            "mapping_hash": resolved_cutover.mapping_hash,
            "last_legacy_epoch_id": resolved_cutover.last_legacy_epoch_id,
            "first_settlement_epoch_id": (
                resolved_cutover.first_settlement_epoch_id
            ),
        }
    state = get_legacy_epoch_namespace_state(
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
    mode: str | None = None,
    epoch_id: int | None = None,
    cutover: SubnetEpochCutover | None = None,
    force_refresh: bool = True,
    network: str | None = None,
    netuid: int | str | None = None,
) -> dict:
    return await asyncio.to_thread(
        validate_epoch_runtime_lifecycle,
        mode=mode,
        epoch_id=epoch_id,
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
    state = get_legacy_epoch_namespace_state(
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

    from gateway.config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        # The mapping hash is the canonical hash of the locally validated
        # manifest. SQL permits stateful_active only after the receipt-backed
        # cutover row and exact first initialization were staged atomically, so
        # the public singleton is sufficient for non-secret validator/auditor
        # startup binding. Gateway processes with service authority additionally
        # prove the exact immutable ledger row below.
        _validated_cutover_authority_hash = authority_key
        return

    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
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
        if get_epoch_mode() == STATEFUL_EPOCH_MODE:
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

    In stateful mode this is the monotonic settlement ordinal, not the official
    Bittensor epoch ID.  Callers must use ``snapshot.subnet_epoch_index`` or
    ``snapshot.epoch_ref`` when they need the official chain identity.
    """

    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return _settlement_epoch_id(snapshot)
    return snapshot.current_block // EPOCH_DURATION_BLOCKS


def get_epoch_elapsed(snapshot: SubnetEpochSnapshot) -> int:
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return snapshot.epoch_block
    return snapshot.current_block % EPOCH_DURATION_BLOCKS


def get_epoch_blocks_remaining(snapshot: SubnetEpochSnapshot) -> int:
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return snapshot.blocks_remaining
    return EPOCH_DURATION_BLOCKS - (
        snapshot.current_block % EPOCH_DURATION_BLOCKS
    )


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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
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
) -> tuple[SubnetEpochSnapshot | LegacyEpochObservation, int]:
    """Return one coherent chain snapshot and its workflow settlement key."""

    if get_epoch_mode() == LEGACY_EPOCH_MODE:
        current_block = await _get_current_block_async()
        workflow_epoch = current_block // EPOCH_DURATION_BLOCKS
        await assert_legacy_epoch_namespace_open_async(workflow_epoch)
        observation = LegacyEpochObservation(
            current_block=current_block,
            observed_at=datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        )
        return observation, workflow_epoch
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

    if get_epoch_mode() == LEGACY_EPOCH_MODE:
        observation, workflow_epoch = await get_current_epoch_context_async()
        return observation, observation, workflow_epoch
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
    mode = get_epoch_mode()
    cutover = _load_cutover() if mode == STATEFUL_EPOCH_MODE else None
    status = snapshot.to_dict(cutover=cutover)
    status["mode"] = mode
    status["official_subnet_epoch_id"] = snapshot.subnet_epoch_index
    if mode == STATEFUL_EPOCH_MODE:
        await validate_epoch_runtime_lifecycle_async(
            mode=mode,
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
    else:
        workflow_epoch = snapshot.current_block // EPOCH_DURATION_BLOCKS
        await validate_epoch_runtime_lifecycle_async(
            mode=mode,
            epoch_id=workflow_epoch,
            force_refresh=True,
        )
        status["workflow_epoch_id"] = workflow_epoch
        status["workflow_epoch_scheme"] = LEGACY_EPOCH_MODE
        status["legacy_epoch_block"] = (
            snapshot.current_block % EPOCH_DURATION_BLOCKS
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
    """
    Get current block number from injected async subtensor.
    
    ASYNC VERSION - Use this from async context (FastAPI endpoints, background tasks).
    Falls back to cached block + time-based estimation if subtensor unavailable.
    
    Uses retry logic to handle transient failures and rate limiting.
    This matches the logic in Leadpoet/validator/reward.py but uses async calls.
    
    Returns:
        Current block number
    
    Raises:
        Exception: If async_subtensor not injected or all retries fail and no cache
    """
    global _last_known_block, _last_known_block_time

    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return (await get_current_subnet_epoch_snapshot_async()).current_block
    
    if _async_subtensor is None:
        raise Exception(
            "AsyncSubtensor not injected - call inject_async_subtensor() first. "
            "This should be done in main.py lifespan."
        )
    
    # Use sync subtensor for block queries (avoids WebSocket subscription conflicts)
    # This is the same fix we used in the validator (neurons/validator.py line 556)
    if _sync_subtensor is None:
        raise Exception(
            "Sync subtensor not initialized - call inject_async_subtensor() first. "
            "This should be done in main.py lifespan."
        )
    
    try:
        # Use sync subtensor's .block property (fast, no subscription conflicts)
        current_block = _sync_subtensor.block
        
        # Cache the successful result
        with _block_cache_lock:
            _last_known_block = current_block
            _last_known_block_time = time.time()
        
        return current_block
        
    except Exception as e:
        # Fallback to cached estimation
        print(f"⚠️  Cannot get current block from sync subtensor: {e}")
        
        with _block_cache_lock:
            if _last_known_block is not None and _last_known_block_time is not None:
                # Calculate blocks elapsed since last known good block
                time_elapsed = time.time() - _last_known_block_time
                blocks_elapsed = int(time_elapsed / BITTENSOR_BLOCK_TIME_SECONDS)
                estimated_block = _last_known_block + blocks_elapsed
                
                print(f"   Using cached block estimation:")
                print(f"   Last known block: {_last_known_block} (cached {int(time_elapsed)}s ago)")
                print(f"   Estimated current: {estimated_block} (+{blocks_elapsed} blocks)")
                return estimated_block
            else:
                # No cache available - this should only happen on first run
                raise Exception(
                    "Cannot query subtensor and no cached block available. "
                    "Please ensure subtensor is accessible."
                )


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
    """
    Calculate current epoch ID based on Bittensor block number (ASYNC VERSION).
    
    Use this from async contexts (FastAPI endpoints, background tasks).
    For sync contexts, use get_current_epoch_id() wrapper.
    
    Returns:
        Epoch ID (integer, 0-indexed)
    
    Example:
        >>> epoch_id = await get_current_epoch_id_async()
        >>> print(epoch_id)
        5678  # Current epoch based on block number
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return _settlement_epoch_id(
            await get_current_subnet_epoch_snapshot_async(finalized=True)
        )
    current_block = await _get_current_block_async()
    epoch_id = current_block // EPOCH_DURATION_BLOCKS
    await assert_legacy_epoch_namespace_open_async(epoch_id)
    return epoch_id


async def get_current_subnet_epoch_id_async() -> int:
    """Return the official on-chain ``SubnetEpochIndex`` without translation."""

    return (await get_current_subnet_epoch_snapshot_async()).subnet_epoch_index


def get_current_epoch_id() -> int:
    """
    Calculate current epoch ID (SYNC WRAPPER - prefer async version).
    
    DEPRECATED: Use get_current_epoch_id_async() from async contexts.
    
    Returns:
        Epoch ID (integer, 0-indexed)
    
    Example:
        >>> get_current_epoch_id()
        5678  # Current epoch based on block number
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return _settlement_epoch_id(
            get_current_subnet_epoch_snapshot(finalized=True)
        )
    current_block = _get_current_block()
    epoch_id = current_block // EPOCH_DURATION_BLOCKS
    assert_legacy_epoch_namespace_open(epoch_id)
    return epoch_id


def get_current_subnet_epoch_id() -> int:
    """Return the official on-chain ``SubnetEpochIndex`` without translation."""

    return get_current_subnet_epoch_snapshot().subnet_epoch_index


async def get_block_within_epoch_async() -> int:
    """
    Get the current block number within the current epoch (0-359) (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_block_within_epoch() wrapper.
    
    Returns:
        Block number within current epoch (0-359)
    
    Example:
        >>> block_in_epoch = await get_block_within_epoch_async()
        >>> print(block_in_epoch)
        145  # Current block is 145 blocks into the epoch
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return (
            await get_current_subnet_epoch_snapshot_async(finalized=True)
        ).epoch_block
    current_block = await _get_current_block_async()
    await assert_legacy_epoch_namespace_open_async(
        current_block // EPOCH_DURATION_BLOCKS
    )
    return current_block % EPOCH_DURATION_BLOCKS


def get_block_within_epoch() -> int:
    """
    Get the current block number within the current epoch (SYNC WRAPPER).
    
    DEPRECATED: Use get_block_within_epoch_async() from async contexts.
    
    Returns:
        Block number within current epoch (0-359)
    
    Example:
        >>> get_block_within_epoch()
        145  # Current block is 145 blocks into the epoch
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        return get_current_subnet_epoch_snapshot(finalized=True).epoch_block
    current_block = _get_current_block()
    assert_legacy_epoch_namespace_open(
        current_block // EPOCH_DURATION_BLOCKS
    )
    return current_block % EPOCH_DURATION_BLOCKS


async def get_epoch_start_time_async(epoch_id: int) -> datetime:
    """
    Get UTC start time for given epoch ID (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_epoch_start_time() wrapper.
    
    Args:
        epoch_id: Epoch number (0-indexed)
    
    Returns:
        datetime (UTC) when epoch starts
    """
    try:
        if get_epoch_mode() == STATEFUL_EPOCH_MODE:
            snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
            current_epoch = _settlement_epoch_id(snapshot)
            if int(epoch_id) != current_epoch:
                raise SubnetEpochError(
                    "historical stateful epoch boundaries require the persisted boundary ledger"
                )
            return get_current_epoch_times(snapshot)[0]
        # Get current reference point using async call
        current_block = await _get_current_block_async()
        now = datetime.utcnow()
        
        # Calculate the start block of the target epoch
        target_start_block = epoch_id * EPOCH_DURATION_BLOCKS
        
        # Calculate blocks from target epoch start to current block
        block_offset = target_start_block - current_block
        
        # Convert block offset to time offset
        time_offset_seconds = block_offset * BITTENSOR_BLOCK_TIME_SECONDS
        
        # Apply offset to current time
        epoch_start = now + timedelta(seconds=time_offset_seconds)
        
        return epoch_start
        
    except SubnetEpochError:
        raise
    except Exception as e:
        # Legacy reporting fallback only. Stateful authority never estimates.
        print(f"⚠️  Error calculating legacy epoch start time: {e}")
        return datetime.utcnow()


def get_epoch_start_time(epoch_id: int) -> datetime:
    """
    Get UTC start time for given epoch ID (SYNC WRAPPER).
    
    DEPRECATED: Use get_epoch_start_time_async() from async contexts.
    
    Args:
        epoch_id: Epoch number (0-indexed)
    
    Returns:
        datetime (UTC) when epoch starts
    
    Example:
        >>> get_epoch_start_time(18895)
        datetime(2025, 11, 3, ...)  # Calculated from current block offset
    """
    try:
        if get_epoch_mode() == STATEFUL_EPOCH_MODE:
            snapshot = get_current_subnet_epoch_snapshot(finalized=True)
            current_epoch = _settlement_epoch_id(snapshot)
            if int(epoch_id) != current_epoch:
                raise SubnetEpochError(
                    "historical stateful epoch boundaries require the persisted boundary ledger"
                )
            return get_current_epoch_times(snapshot)[0]
        # Get current reference point
        current_block = _get_current_block()
        now = datetime.utcnow()
        
        # Calculate the start block of the target epoch
        target_start_block = epoch_id * EPOCH_DURATION_BLOCKS
        
        # Calculate blocks from target epoch start to current block
        # Negative offset = epoch started in the past
        # Positive offset = epoch starts in the future
        block_offset = target_start_block - current_block
        
        # Convert block offset to time offset
        time_offset_seconds = block_offset * BITTENSOR_BLOCK_TIME_SECONDS
        
        # Apply offset to current time
        epoch_start = now + timedelta(seconds=time_offset_seconds)
        
        return epoch_start
        
    except SubnetEpochError:
        raise
    except Exception as e:
        print(f"⚠️  Error calculating legacy epoch start time: {e}")
        return datetime.utcnow()


async def get_epoch_end_time_async(epoch_id: int) -> datetime:
    """
    Get UTC end time for given epoch ID (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_epoch_end_time() wrapper.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when validation phase ends (block 360)
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
        if int(epoch_id) != _settlement_epoch_id(snapshot):
            raise SubnetEpochError(
                "historical stateful epoch boundaries require the persisted boundary ledger"
            )
        return get_current_epoch_times(snapshot)[1]
    start = await get_epoch_start_time_async(epoch_id)
    return start + timedelta(
        seconds=EPOCH_DURATION_BLOCKS * BITTENSOR_BLOCK_TIME_SECONDS
    )


def get_epoch_end_time(epoch_id: int) -> datetime:
    """
    Get UTC end time for given epoch ID (SYNC WRAPPER).
    
    DEPRECATED: Use get_epoch_end_time_async() from async contexts.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when validation phase ends (block 360)
    
    Example:
        >>> start = get_epoch_start_time(100)
        >>> end = get_epoch_end_time(100)
        >>> (end - start).total_seconds()
        4320.0  # 360 blocks × 12 sec = 72 minutes
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        if int(epoch_id) != _settlement_epoch_id(snapshot):
            raise SubnetEpochError(
                "historical stateful epoch boundaries require the persisted boundary ledger"
            )
        return get_current_epoch_times(snapshot)[1]
    start = get_epoch_start_time(epoch_id)
    return start + timedelta(
        seconds=EPOCH_DURATION_BLOCKS * BITTENSOR_BLOCK_TIME_SECONDS
    )


async def get_epoch_close_time_async(epoch_id: int) -> datetime:
    """
    Get UTC close time for given epoch ID (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_epoch_close_time() wrapper.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when epoch closes (block 360)
    """
    # No grace period - epoch closes at block 360, same as end time
    return await get_epoch_end_time_async(epoch_id)


def get_epoch_close_time(epoch_id: int) -> datetime:
    """
    Get UTC close time for given epoch ID (SYNC WRAPPER).
    
    DEPRECATED: Use get_epoch_close_time_async() from async contexts.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when epoch closes (block 360)
    
    Example:
        >>> start = get_epoch_start_time(100)
        >>> close = get_epoch_close_time(100)
        >>> (close - start).total_seconds()
        4320.0  # 360 blocks × 12 sec = 72 minutes
    """
    # No grace period - epoch closes at block 360, same as end time
    return get_epoch_end_time(epoch_id)


async def is_epoch_active_async(epoch_id: int) -> bool:
    """
    Check if epoch is currently in validation phase (blocks 0-360) (ASYNC VERSION).
    
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
        True  # If current time is within validation window (blocks 0-360)
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
        return (
            int(epoch_id) == _settlement_epoch_id(snapshot)
            and snapshot.blocks_remaining > 0
        )
    now = datetime.utcnow()
    start = await get_epoch_start_time_async(epoch_id)
    end = await get_epoch_end_time_async(epoch_id)
    
    return start <= now <= end


def is_epoch_active(epoch_id: int) -> bool:
    """
    Check if epoch is currently in validation phase (blocks 0-360) (SYNC WRAPPER).
    
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
        True  # If current time is within validation window (blocks 0-360)
    """
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        return (
            int(epoch_id) == _settlement_epoch_id(snapshot)
            and snapshot.blocks_remaining > 0
        )
    now = datetime.utcnow()
    start = get_epoch_start_time(epoch_id)
    end = get_epoch_end_time(epoch_id)
    
    return start <= now <= end


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
    # No grace period - validators can submit anytime during blocks 0-360
    return False


async def is_epoch_closed_async(epoch_id: int) -> bool:
    """
    Check if epoch is closed (past block 360) (ASYNC VERSION).
    
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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
        return int(epoch_id) < _settlement_epoch_id(snapshot)
    now = datetime.utcnow()
    close = await get_epoch_close_time_async(epoch_id)
    
    return now > close


def is_epoch_closed(epoch_id: int) -> bool:
    """
    Check if epoch is closed (past block 360) (SYNC WRAPPER).
    
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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        return int(epoch_id) < _settlement_epoch_id(snapshot)
    now = datetime.utcnow()
    close = get_epoch_close_time(epoch_id)
    
    return now > close


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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        if int(epoch_id) != _settlement_epoch_id(snapshot):
            return 0
        return snapshot.blocks_remaining * BITTENSOR_BLOCK_TIME_SECONDS
    now = datetime.utcnow()
    end = get_epoch_end_time(epoch_id)
    
    if now > end:
        return 0
    
    return int((end - now).total_seconds())


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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        if int(epoch_id) != _settlement_epoch_id(snapshot):
            return 0
        return snapshot.blocks_remaining * BITTENSOR_BLOCK_TIME_SECONDS
    now = datetime.utcnow()
    close = get_epoch_close_time(epoch_id)
    
    if now > close:
        return 0
    
    return int((close - now).total_seconds())


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
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = get_current_subnet_epoch_snapshot(finalized=True)
        current_epoch = _settlement_epoch_id(snapshot)
        if int(epoch_id) != current_epoch:
            raise SubnetEpochError(
                "historical stateful epoch details require the persisted boundary ledger"
            )
        return get_current_epoch_info_from_snapshot(snapshot)
    return {
        'epoch_id': epoch_id,
        'start_time': get_epoch_start_time(epoch_id).isoformat(),
        'end_time': get_epoch_end_time(epoch_id).isoformat(),
        'close_time': get_epoch_close_time(epoch_id).isoformat(),
        'phase': get_epoch_phase(epoch_id),
        'is_active': is_epoch_active(epoch_id),
        'is_grace_period': is_epoch_in_grace_period(epoch_id),
        'is_closed': is_epoch_closed(epoch_id),
        'time_until_end': time_until_epoch_end(epoch_id),
        'time_until_close': time_until_epoch_close(epoch_id)
    }


async def get_epoch_info_async(epoch_id: int) -> dict:
    """Async epoch information built from one authoritative state read."""

    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        snapshot = await get_current_subnet_epoch_snapshot_async(finalized=True)
        current_epoch = _settlement_epoch_id(snapshot)
        if int(epoch_id) != current_epoch:
            raise SubnetEpochError(
                "historical stateful epoch details require the persisted boundary ledger"
            )
        return get_current_epoch_info_from_snapshot(snapshot)

    current_block = await _get_current_block_async()
    now = datetime.utcnow()
    target_start = int(epoch_id) * EPOCH_DURATION_BLOCKS
    start = now + timedelta(
        seconds=(target_start - current_block) * BITTENSOR_BLOCK_TIME_SECONDS
    )
    end = start + timedelta(
        seconds=EPOCH_DURATION_BLOCKS * BITTENSOR_BLOCK_TIME_SECONDS
    )
    active = start <= now <= end
    closed = now > end
    seconds_remaining = max(0, int((end - now).total_seconds()))
    return {
        "epoch_id": int(epoch_id),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "close_time": end.isoformat(),
        "phase": "active" if active else "closed",
        "is_active": active,
        "is_grace_period": False,
        "is_closed": closed,
        "time_until_end": seconds_remaining,
        "time_until_close": seconds_remaining,
    }
