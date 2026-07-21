"""Explicit operator activation for one receipt-backed subnet-epoch cutover.

The command is read-only by default.  ``--apply`` is required before it will
enter the coordinator enclave or insert the append-only cutover authority.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from Leadpoet.utils.restart_epoch_gate import (
    MAXIMUM_RESTART_EPOCH_BLOCK,
    load_restart_epoch_start,
)
from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_scoring_v2 import DEFAULT_RELEASE_MANIFEST_PATH
from gateway.research_lab.attested_v2_store import (
    load_receipt_graph_v2,
    persist_receipt_graph_v2,
)
from gateway.research_lab.stateful_epoch_authority_v1 import (
    CANDIDATE_TABLE,
    CUTOVER_TABLE,
    build_cutover_row_v1,
    validate_stored_pre_cutover_candidate_row_v1,
)
from gateway.research_lab.champion_settlement_v2 import (
    champion_v2_cutover_readiness,
)
from gateway.research_lab.store import call_rpc, select_all
from gateway.tee.coordinator_epoch_cutover_v2 import (
    CUTOVER_PURPOSE,
    CUTOVER_REQUEST_SCHEMA_VERSION,
    HISTORICAL_PREDECESSOR_KIND,
    NATIVE_PREDECESSOR_KIND,
    OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
    attest_subnet_epoch_cutover_v2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.release_lineage_v2 import (
    build_release_lineage_boot_verifier_v2,
    load_approved_release_lineage_v2,
)
from gateway.utils.subnet_epoch_archive import (
    read_exact_subnet_epoch_snapshot_from_archive,
    read_finalized_subnet_epoch_snapshot_from_archive,
    validate_cutover_anchor_from_archive,
)
from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    WEIGHT_ROLE,
    sha256_json,
    validate_receipt_graph,
    verify_boot_identity_nitro,
)


FINALIZED_ALLOCATION_VIEW = "research_lab_finalized_allocation_epochs_v2"
HISTORICAL_FINALIZATION_TABLE = (
    "research_lab_legacy_finalized_allocation_migrations_v2"
)
RECEIPT_TABLE = "research_lab_attested_execution_receipts_v2"
CUTOVER_PREFLIGHT_RPC = (
    "research_lab_stateful_subnet_epoch_cutover_preflight_v1"
)
CUTOVER_FENCE_RPC = "research_lab_stateful_subnet_epoch_cutover_fence_v1"
CUTOVER_BIND_RPC = "research_lab_stateful_subnet_epoch_cutover_bind_v1"
CUTOVER_STAGE_RPC = "research_lab_stateful_subnet_epoch_stage_v1"
CUTOVER_BOOTSTRAP_BIND_RPC = (
    "research_lab_stateful_subnet_epoch_cutover_bind_v2"
)
CUTOVER_BOOTSTRAP_STAGE_RPC = "research_lab_stateful_subnet_epoch_stage_v2"
CUTOVER_ACTIVATE_RPC = "research_lab_stateful_subnet_epoch_activate_v1"
CUTOVER_STATE_TABLE = "research_lab_stateful_subnet_epoch_cutover_state_v1"
ACTIVATION_REPORT_SCHEMA_VERSION = (
    "leadpoet.subnet_epoch_cutover_activation_report.v1"
)

_PREFLIGHT_FIELDS = frozenset(
    {
        "eligible",
        "legacy_high_water",
        "expected_last_legacy_epoch_id",
        "first_settlement_epoch_id",
        "first_settlement_occupied",
        "candidate_snapshot_hash",
        "candidate_receipt_hash",
    }
)
_FENCE_FIELDS = frozenset(
    {
        "lifecycle_state",
        "mapping_hash",
        "legacy_high_water",
        "last_legacy_epoch_id",
        "first_settlement_epoch_id",
        "candidate_snapshot_hash",
        "candidate_receipt_hash",
        "cutover_authority_hash",
        "cutover_receipt_hash",
        "initialization_nonce",
        "initialization_payload_hash",
    }
)
_EARLY_FENCE_FIELDS = frozenset(
    {
        "lifecycle_state",
        "network_genesis_hash",
        "netuid",
        "legacy_high_water",
        "last_legacy_epoch_id",
        "first_settlement_epoch_id",
        "first_settlement_occupied",
        "fenced_at",
    }
)
_STAGE_FIELDS = frozenset(
    {
        "lifecycle_state",
        "mapping_hash",
        "cutover_authority_hash",
        "cutover_receipt_hash",
        "initialization_nonce",
        "initialization_payload_hash",
    }
)


class StatefulEpochCutoverActivationError(RuntimeError):
    """A cutover activation input is absent, conflicting, or ineligible."""

    def __init__(self, message: str, *, report: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.report = dict(report) if isinstance(report, Mapping) else None


@contextmanager
def _stateful_epoch_environment(cutover: SubnetEpochCutover):
    """Temporarily install the cutover manifest inside this process."""

    previous = {
        name: os.environ.get(name)
        for name in (CUTOVER_JSON_ENV, CUTOVER_PATH_ENV)
    }
    os.environ[CUTOVER_JSON_ENV] = json.dumps(
        cutover.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    os.environ.pop(CUTOVER_PATH_ENV, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def _load_live_finalized_snapshot(
    cutover: SubnetEpochCutover,
) -> SubnetEpochSnapshot:
    def _read() -> SubnetEpochSnapshot:
        import bittensor as bt

        subtensor = bt.Subtensor(network=os.getenv("BITTENSOR_NETWORK", "finney"))
        return read_subnet_epoch_snapshot(
            subtensor,
            netuid=cutover.netuid,
            finalized=True,
        )

    return await asyncio.to_thread(_read)


async def _load_official_archive_boundary(
    netuid: int,
) -> SubnetEpochSnapshot:
    """Discover and re-read the latest finalized official epoch boundary."""

    current = await asyncio.to_thread(
        read_finalized_subnet_epoch_snapshot_from_archive,
        netuid=int(netuid),
    )
    if current.last_epoch_block <= 0:
        raise StatefulEpochCutoverActivationError(
            "official archive has no usable subnet epoch boundary"
        )
    return await asyncio.to_thread(
        read_exact_subnet_epoch_snapshot_from_archive,
        netuid=int(netuid),
        block_number=current.last_epoch_block,
    )


async def _validate_official_archive_anchor(
    cutover: SubnetEpochCutover,
) -> None:
    await asyncio.to_thread(validate_cutover_anchor_from_archive, cutover)


async def _load_epoch_initialization(epoch_id: int) -> Optional[Mapping[str, Any]]:
    from gateway.tasks.epoch_lifecycle import get_durable_epoch_event

    return await get_durable_epoch_event("EPOCH_INITIALIZATION", epoch_id)


async def _prepare_epoch_initialization(**kwargs: Any) -> Dict[str, Any]:
    from gateway.tasks.epoch_lifecycle import (
        build_epoch_event_row,
        compute_and_log_epoch_initialization,
    )

    captured: Dict[str, Any] = {}

    async def capture(event_type: str, epoch_id: int, payload: dict) -> str:
        captured.update(
            {
                "event_type": event_type,
                "epoch_id": epoch_id,
                "payload": dict(payload),
            }
        )
        return "prepared-not-persisted"

    snapshot = kwargs.get("epoch_snapshot")
    observed_at = getattr(snapshot, "observed_at", None)
    await compute_and_log_epoch_initialization(
        **kwargs,
        event_writer=capture,
        payload_timestamp=observed_at,
    )
    if set(captured) != {"event_type", "epoch_id", "payload"}:
        raise StatefulEpochCutoverActivationError(
            "first stateful EPOCH_INITIALIZATION was not prepared"
        )
    return build_epoch_event_row(
        str(captured["event_type"]),
        int(captured["epoch_id"]),
        dict(captured["payload"]),
        event_timestamp=observed_at,
    )


def _mixed_boot_verifier_from_release(
    release: Mapping[str, Any],
    *,
    nitro_verifier: Callable[..., Mapping[str, Any]] = verify_boot_identity_nitro,
    validator_pcr0_verifier: Optional[Callable[[str], Mapping[str, Any]]] = None,
    validator_boot_verifier: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Verify coordinator release boots and independently rebuilt validator boots."""

    if validator_pcr0_verifier is None:
        from gateway.utils.pcr0_builder import verify_pcr0

        validator_pcr0_verifier = verify_pcr0

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        physical_role = str(identity.get("physical_role") or "")
        service_role = str(identity.get("role") or "")
        if physical_role == "validator_weights":
            if service_role != WEIGHT_ROLE:
                raise ValueError("cutover validator boot role is invalid")
            if validator_boot_verifier is not None:
                return validator_boot_verifier(identity)
            rebuilt = validator_pcr0_verifier(str(identity.get("pcr0") or ""))
            if not isinstance(rebuilt, Mapping) or not rebuilt.get("valid"):
                raise ValueError(
                    "validator PCR0 is absent from the dynamic Git build cache"
                )
            if str(rebuilt.get("commit_hash") or "").lower() != str(
                identity.get("commit_sha") or ""
            ).lower():
                raise ValueError(
                    "validator PCR0 commit differs from boot identity"
                )
        elif physical_role == "gateway_coordinator":
            if service_role != COORDINATOR_ROLE:
                raise ValueError("cutover coordinator boot role is invalid")
            roles = release.get("roles")
            expected = (
                roles.get(physical_role)
                if isinstance(roles, Mapping)
                else None
            )
            if not isinstance(expected, Mapping):
                raise ValueError(
                    "gateway coordinator boot is absent from approved release"
                )
            for field in (
                "commit_sha",
                "pcr0",
                "build_manifest_hash",
                "dependency_lock_hash",
            ):
                release_field = (
                    "execution_manifest_hash"
                    if field == "build_manifest_hash"
                    else field
                )
                if str(identity.get(field) or "").lower() != str(
                    expected.get(release_field) or ""
                ).lower():
                    raise ValueError(
                        "gateway coordinator boot differs from approved release at %s"
                        % field
                    )
        else:
            raise ValueError("cutover receipt graph contains an unexpected boot role")
        return nitro_verifier(
            identity,
            expected_pcr0=str(identity.get("pcr0") or ""),
        )

    return verify


def build_cutover_mixed_boot_verifier_v1(
    release: Mapping[str, Any],
    *,
    validator_release_manifest: Optional[Mapping[str, Any]] = None,
    parent_graphs: Sequence[Mapping[str, Any]] = (),
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    current = validate_release_manifest(release)
    validator_boot_verifier = None
    if validator_release_manifest is not None:
        from gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1 import (
            build_validator_release_boot_verifier_v1,
        )

        validator_boot_verifier = build_validator_release_boot_verifier_v1(
            validator_release_manifest
        )
    current_verifier = _mixed_boot_verifier_from_release(
        current,
        validator_boot_verifier=validator_boot_verifier,
    )
    if not parent_graphs:
        return current_verifier
    lineage = load_approved_release_lineage_v2(
        current_release=current,
        parent_graphs=parent_graphs,
    )
    lineage_verifier = build_release_lineage_boot_verifier_v2(lineage)

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        if str(identity.get("physical_role") or "") == "gateway_coordinator":
            return lineage_verifier(identity)
        return current_verifier(identity)

    return verify


def _load_gateway_release(path: Optional[Path]) -> Dict[str, Any]:
    target = Path(path or DEFAULT_RELEASE_MANIFEST_PATH).expanduser()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatefulEpochCutoverActivationError(
            "approved gateway V2 release manifest is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise StatefulEpochCutoverActivationError(
            "approved gateway V2 release manifest is invalid"
        )
    try:
        return validate_release_manifest(value)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "approved gateway V2 release manifest is invalid"
        ) from exc


def _load_validator_release(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        raise StatefulEpochCutoverActivationError(
            "approved validator V2 release manifest is unavailable"
        )
    try:
        from gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1 import (
            load_validator_release_manifest_v2,
        )

        return load_validator_release_manifest_v2(Path(path))
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "approved validator V2 release manifest is invalid"
        ) from exc


def _validate_initialization(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
    snapshot_doc: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
        raise StatefulEpochCutoverActivationError(
            "first stateful EPOCH_INITIALIZATION readback is invalid"
        )
    row = dict(value)
    payload = dict(row["payload"])
    boundary = payload.get("epoch_boundaries")
    if (
        payload.get("epoch_id") != cutover.first_settlement_epoch_id
        or payload.get("epoch_key_semantics") != "settlement_ordinal"
        or payload.get("epoch_authority") != dict(snapshot_doc)
        or not isinstance(boundary, Mapping)
        or boundary.get("start_block") != snapshot_doc["last_epoch_block"]
        or boundary.get("end_block") != snapshot_doc["next_epoch_block"]
        or boundary.get("expected_end_block") != snapshot_doc["next_epoch_block"]
        or boundary.get("pending_epoch_at") != snapshot_doc["pending_epoch_at"]
        or boundary.get("tempo") != snapshot_doc["tempo"]
        or payload.get("timestamp") != snapshot_doc["observed_at"]
    ):
        raise StatefulEpochCutoverActivationError(
            "first stateful EPOCH_INITIALIZATION differs from candidate authority"
        )
    from gateway.tasks.epoch_lifecycle import build_epoch_event_row

    expected = build_epoch_event_row(
        "EPOCH_INITIALIZATION",
        cutover.first_settlement_epoch_id,
        payload,
        event_timestamp=snapshot_doc["observed_at"],
    )
    try:
        stored_ts = datetime.fromisoformat(str(row.get("ts")).replace("Z", "+00:00"))
        expected_ts = datetime.fromisoformat(
            str(expected["ts"]).replace("Z", "+00:00")
        )
        if stored_ts.tzinfo is None:
            stored_ts = stored_ts.replace(tzinfo=timezone.utc)
        if expected_ts.tzinfo is None:
            expected_ts = expected_ts.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise StatefulEpochCutoverActivationError(
            "first stateful EPOCH_INITIALIZATION timestamp is invalid"
        ) from exc
    if (
        row.get("event_type") != expected["event_type"]
        or row.get("actor_hotkey") != expected["actor_hotkey"]
        or str(row.get("nonce") or "") != expected["nonce"]
        or row.get("payload_hash") != expected["payload_hash"]
        or row.get("signature") != expected["signature"]
        or stored_ts.astimezone(timezone.utc) != expected_ts.astimezone(timezone.utc)
    ):
        raise StatefulEpochCutoverActivationError(
            "first stateful EPOCH_INITIALIZATION row binding is invalid"
        )
    return row


async def _initialization_status(
    *,
    cutover: SubnetEpochCutover,
    snapshot_doc: Mapping[str, Any],
    load_initialization: Callable[[int], Awaitable[Optional[Mapping[str, Any]]]],
    load_live_snapshot: Callable[[SubnetEpochCutover], Awaitable[SubnetEpochSnapshot]],
) -> Dict[str, Any]:
    with _stateful_epoch_environment(cutover):
        existing = await load_initialization(cutover.first_settlement_epoch_id)
        if existing is not None:
            durable = _validate_initialization(
                existing,
                cutover=cutover,
                snapshot_doc=snapshot_doc,
            )
            return {
                "exists": True,
                "status": "durable",
                "eligible": True,
                "event_hash": durable.get("event_hash"),
                "authority_hash": sha256_json(snapshot_doc),
            }

        live = await load_live_snapshot(cutover)
        restart_start = _configured_restart_start(cutover)
        return _live_initialization_window_status(
            cutover=cutover,
            snapshot_doc=snapshot_doc,
            live=live,
            restart_start=restart_start,
        )


def _configured_restart_start(
    cutover: SubnetEpochCutover,
) -> Optional[SubnetEpochSnapshot]:
    path = os.environ.get("LEADPOET_RESTART_START_PATH")
    if not path:
        return None
    try:
        return load_restart_epoch_start(Path(path), netuid=cutover.netuid)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "captured restart start is unavailable or invalid"
        ) from exc


def _live_initialization_window_status(
    *,
    cutover: SubnetEpochCutover,
    snapshot_doc: Mapping[str, Any],
    live: SubnetEpochSnapshot,
    restart_start: Optional[SubnetEpochSnapshot] = None,
) -> Dict[str, Any]:
    """Require the captured first epoch and one valid operator start."""

    try:
        boundary = SubnetEpochSnapshot.from_mapping(snapshot_doc)
        live_doc = live.to_dict(cutover=cutover)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "live finalized subnet epoch snapshot is invalid"
        ) from exc
    start = restart_start or live
    eligible = (
        boundary.network_genesis_hash == cutover.network_genesis_hash
        and boundary.netuid == cutover.netuid
        and boundary.current_block == cutover.cutover_block
        and boundary.block_hash == cutover.cutover_block_hash
        and boundary.last_epoch_block == cutover.cutover_block
        and boundary.subnet_epoch_index == cutover.first_subnet_epoch_index
        and boundary.settlement_epoch_id(cutover)
        == cutover.first_settlement_epoch_id
        and boundary.epoch_ref == live.epoch_ref
        and start.network_genesis_hash == cutover.network_genesis_hash
        and start.netuid == cutover.netuid
        and start.subnet_epoch_index == cutover.first_subnet_epoch_index
        and start.last_epoch_block == cutover.cutover_block
        and 0 <= start.epoch_block <= MAXIMUM_RESTART_EPOCH_BLOCK
        and live.network_genesis_hash == cutover.network_genesis_hash
        and live.netuid == cutover.netuid
        and live.subnet_epoch_index == cutover.first_subnet_epoch_index
        and live.settlement_epoch_id(cutover)
        == cutover.first_settlement_epoch_id
        and live.last_epoch_block == cutover.cutover_block
        and live.current_block >= start.current_block
        and live.blocks_remaining > 0
    )
    return {
        "exists": False,
        "status": "missing",
        "eligible": eligible,
        "live_subnet_epoch_index": live.subnet_epoch_index,
        "live_settlement_epoch_id": live_doc["settlement_epoch_id"],
        "live_epoch_block": live.epoch_block,
        "live_blocks_remaining": live.blocks_remaining,
        "live_block": live.current_block,
        "live_block_hash": live.block_hash,
        "authority_hash": sha256_json(snapshot_doc),
        "latest_safe_epoch_block": MAXIMUM_RESTART_EPOCH_BLOCK,
        "restart_start_epoch_block": start.epoch_block,
        "restart_start_captured": restart_start is not None,
        "deadline_reapplied": restart_start is None,
    }


def load_cutover_manifest_v1(path: Optional[Path] = None) -> SubnetEpochCutover:
    """Load one canonical manifest from an explicit file or configured env."""

    if path is None:
        return load_subnet_epoch_cutover()
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatefulEpochCutoverActivationError(
            "subnet epoch cutover manifest is unavailable or invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise StatefulEpochCutoverActivationError(
            "subnet epoch cutover manifest must be one object"
        )
    try:
        cutover = SubnetEpochCutover.from_mapping(value)
    except (TypeError, SubnetEpochError) as exc:
        raise StatefulEpochCutoverActivationError(
            "subnet epoch cutover manifest is invalid"
        ) from exc
    if dict(value) != cutover.to_dict():
        raise StatefulEpochCutoverActivationError(
            "subnet epoch cutover manifest is not canonical"
        )
    return cutover


async def fence_subnet_epoch_namespace_v1(
    *,
    network_genesis_hash: str,
    netuid: int,
    last_legacy_epoch_id: int,
    first_settlement_epoch_id: int,
    rpc: Callable[[str, Mapping[str, Any]], Awaitable[Any]] = call_rpc,
    select_rows: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = select_all,
) -> Dict[str, Any]:
    """Atomically measure and close the integer namespace before either boundary."""

    if (
        not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid <= 0
        or not isinstance(last_legacy_epoch_id, int)
        or isinstance(last_legacy_epoch_id, bool)
        or last_legacy_epoch_id < 0
        or first_settlement_epoch_id != last_legacy_epoch_id + 1
    ):
        raise StatefulEpochCutoverActivationError(
            "pre-boundary cutover fence inputs are invalid"
        )
    value = await rpc(
        CUTOVER_FENCE_RPC,
        {
            "p_network_genesis_hash": network_genesis_hash,
            "p_netuid": netuid,
            "p_last_legacy_epoch_id": last_legacy_epoch_id,
            "p_first_settlement_epoch_id": first_settlement_epoch_id,
        },
    )
    durable = _normalize_early_fence(
        value,
        network_genesis_hash=network_genesis_hash,
        netuid=netuid,
        last_legacy_epoch_id=last_legacy_epoch_id,
        first_settlement_epoch_id=first_settlement_epoch_id,
    )
    state_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    if len(state_rows) != 1 or not isinstance(state_rows[0], Mapping):
        raise StatefulEpochCutoverActivationError(
            "pre-boundary cutover fence durable readback is missing"
        )
    state = dict(state_rows[0])
    for field in (
        "lifecycle_state",
        "network_genesis_hash",
        "netuid",
        "last_legacy_epoch_id",
        "first_settlement_epoch_id",
    ):
        expected = (
            durable["legacy_high_water"]
            if field == "last_legacy_epoch_id"
            else durable[field]
        )
        if state.get(field) != expected:
            raise StatefulEpochCutoverActivationError(
                "pre-boundary cutover fence readback differs at %s" % field
            )
    return {
        "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
        "mode": "fence_before_boundary",
        "status": "cutover_fenced",
        "network_genesis_hash": network_genesis_hash,
        "netuid": netuid,
        "last_legacy_epoch_id": last_legacy_epoch_id,
        "first_settlement_epoch_id": first_settlement_epoch_id,
        "fenced_at": durable["fenced_at"],
        "next_step": (
            "wait for the official boundary, propose its archive-derived manifest, "
            "then capture and sign the validator candidate"
        ),
    }


async def propose_subnet_epoch_cutover_manifest_v1(
    *,
    network_genesis_hash: str,
    netuid: int,
    last_legacy_epoch_id: int,
    select_rows: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = select_all,
    load_boundary: Callable[[int], Awaitable[SubnetEpochSnapshot]] = (
        _load_official_archive_boundary
    ),
    validate_anchor: Callable[[SubnetEpochCutover], Awaitable[None]] = (
        _validate_official_archive_anchor
    ),
) -> Dict[str, Any]:
    """Derive a manifest from archive state before validator candidate capture.

    The candidate is cryptographically bound to this manifest, so proposal may
    not depend on a candidate row.  That would make the ceremony circular and
    impossible to execute from a clean database.
    """

    first_settlement_epoch_id = last_legacy_epoch_id + 1
    state_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    if len(state_rows) != 1 or not isinstance(state_rows[0], Mapping):
        raise StatefulEpochCutoverActivationError(
            "manifest proposal requires one durable pre-boundary fence"
        )
    state = dict(state_rows[0])
    if (
        state.get("lifecycle_state") != "cutover_fenced"
        or state.get("network_genesis_hash") != network_genesis_hash
        or state.get("netuid") != netuid
        or state.get("last_legacy_epoch_id") != last_legacy_epoch_id
        or state.get("first_settlement_epoch_id") != first_settlement_epoch_id
    ):
        raise StatefulEpochCutoverActivationError(
            "manifest proposal differs from the pre-boundary fence"
        )
    try:
        snapshot = await load_boundary(netuid)
    except StatefulEpochCutoverActivationError:
        raise
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "official archive boundary is unavailable"
        ) from exc
    cutover = SubnetEpochCutover(
        network_genesis_hash=network_genesis_hash,
        netuid=netuid,
        cutover_block=snapshot.current_block,
        cutover_block_hash=snapshot.block_hash,
        first_subnet_epoch_index=snapshot.subnet_epoch_index,
        first_settlement_epoch_id=first_settlement_epoch_id,
        last_legacy_epoch_id=last_legacy_epoch_id,
    )
    if (
        snapshot.head_kind != "exact"
        or snapshot.network_genesis_hash != cutover.network_genesis_hash
        or snapshot.netuid != cutover.netuid
        or snapshot.current_block != snapshot.last_epoch_block
        or snapshot.epoch_block != 0
    ):
        raise StatefulEpochCutoverActivationError(
            "archive boundary is not the exact reserved cutover transition"
        )
    try:
        await validate_anchor(cutover)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "official archive rejected the proposed cutover anchor"
        ) from exc
    snapshot_doc = snapshot.to_dict()
    return {
        "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
        "mode": "propose_manifest",
        "status": "manifest_proposed",
        "boundary_snapshot_hash": sha256_json(snapshot_doc),
        "manifest": cutover.to_dict(),
    }


async def _select_exactly_one(
    select_rows: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
    table: str,
    *,
    filters: Sequence[tuple[Any, ...]],
    label: str,
) -> Dict[str, Any]:
    rows = await select_rows(
        table,
        filters=tuple(filters),
        batch_size=2,
        max_rows=2,
    )
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise StatefulEpochCutoverActivationError(
            "%s must resolve to exactly one durable row" % label
        )
    return dict(rows[0])


def _normalize_preflight(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, Mapping) or set(value) != _PREFLIGHT_FIELDS:
        raise StatefulEpochCutoverActivationError(
            "cutover high-water preflight is unavailable or malformed"
        )
    result = dict(value)
    expected = {
        "candidate_snapshot_hash": candidate["snapshot_hash"],
        "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        "expected_last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "legacy_high_water": cutover.last_legacy_epoch_id,
        "first_settlement_occupied": False,
        "eligible": True,
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "cutover high-water preflight differs at %s" % field
            )
    return result


def _one_rpc_row(value: Any, *, fields: frozenset[str], label: str) -> Dict[str, Any]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StatefulEpochCutoverActivationError(
            "%s response is unavailable or malformed" % label
        )
    return dict(value)


def _normalize_early_fence(
    value: Any,
    *,
    network_genesis_hash: str,
    netuid: int,
    last_legacy_epoch_id: int,
    first_settlement_epoch_id: int,
) -> Dict[str, Any]:
    result = _one_rpc_row(
        value,
        fields=_EARLY_FENCE_FIELDS,
        label="pre-boundary cutover fence",
    )
    expected = {
        "lifecycle_state": "cutover_fenced",
        "network_genesis_hash": network_genesis_hash,
        "netuid": netuid,
        "legacy_high_water": last_legacy_epoch_id,
        "last_legacy_epoch_id": last_legacy_epoch_id,
        "first_settlement_epoch_id": first_settlement_epoch_id,
        "first_settlement_occupied": False,
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "pre-boundary cutover fence differs at %s" % field
            )
    if not result.get("fenced_at"):
        raise StatefulEpochCutoverActivationError(
            "pre-boundary cutover fence has no durable timestamp"
        )
    return result


def _normalize_binding(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
    candidate: Mapping[str, Any],
    authority_hash: str,
) -> Dict[str, Any]:
    result = _one_rpc_row(value, fields=_FENCE_FIELDS, label="cutover binding")
    expected = {
        "mapping_hash": cutover.mapping_hash,
        "legacy_high_water": cutover.last_legacy_epoch_id,
        "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        "candidate_snapshot_hash": candidate["snapshot_hash"],
        "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
        "cutover_authority_hash": authority_hash,
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "cutover binding differs at %s" % field
            )
    if result.get("lifecycle_state") not in {
        "cutover_fenced",
        "stateful_staged",
        "stateful_active",
    }:
        raise StatefulEpochCutoverActivationError(
            "cutover binding returned an invalid lifecycle state"
        )
    return result


def _normalize_stage(
    value: Any,
    *,
    cutover_row: Mapping[str, Any],
    initialization_event: Mapping[str, Any],
) -> Dict[str, Any]:
    result = _one_rpc_row(value, fields=_STAGE_FIELDS, label="atomic cutover stage")
    expected = {
        "lifecycle_state": "stateful_staged",
        "mapping_hash": cutover_row["mapping_hash"],
        "cutover_authority_hash": cutover_row["cutover_authority_hash"],
        "cutover_receipt_hash": cutover_row["cutover_receipt_hash"],
        "initialization_nonce": initialization_event["nonce"],
        "initialization_payload_hash": initialization_event["payload_hash"],
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "atomic cutover stage differs at %s" % field
            )
    return result


def _assert_cutover_state(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutover: SubnetEpochCutover,
    cutover_row: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise StatefulEpochCutoverActivationError(
            "stateful epoch cutover singleton readback is invalid"
        )
    row = dict(rows[0])
    if (
        row.get("lifecycle_state") not in {"stateful_staged", "stateful_active"}
        or row.get("mapping_hash") != cutover.mapping_hash
        or row.get("cutover_authority_hash")
        != cutover_row["cutover_authority_hash"]
        or row.get("cutover_receipt_hash") != cutover_row["cutover_receipt_hash"]
        or str(row.get("initialization_nonce") or "")
        != str(initialization.get("nonce") or "")
        or row.get("initialization_payload_hash")
        != initialization.get("payload_hash")
    ):
        raise StatefulEpochCutoverActivationError(
            "stateful epoch cutover singleton differs from atomic readback"
        )
    return row


def _require_preboundary_fence(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutover: SubnetEpochCutover,
) -> Dict[str, Any]:
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise StatefulEpochCutoverActivationError(
            "pre-boundary stateful epoch fence is missing"
        )
    row = dict(rows[0])
    if (
        row.get("lifecycle_state") != "cutover_fenced"
        or row.get("network_genesis_hash") != cutover.network_genesis_hash
        or row.get("netuid") != cutover.netuid
        or row.get("last_legacy_epoch_id") != cutover.last_legacy_epoch_id
        or row.get("first_settlement_epoch_id")
        != cutover.first_settlement_epoch_id
    ):
        raise StatefulEpochCutoverActivationError(
            "cutover manifest differs from the pre-boundary fence"
        )
    return row


def _assert_existing_cutover(
    row: Mapping[str, Any],
    *,
    cutover: SubnetEpochCutover,
    receipt_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    authority = row.get("authority_doc")
    snapshot = row.get("first_snapshot_doc")
    if not isinstance(authority, Mapping) or not isinstance(snapshot, Mapping):
        raise StatefulEpochCutoverActivationError(
            "existing cutover authority documents are invalid"
        )
    expected = build_cutover_row_v1(
        authority_doc=authority,
        first_snapshot_doc=snapshot,
        receipt_graph=receipt_graph,
    )
    if expected.get("mapping_hash") != cutover.mapping_hash:
        raise StatefulEpochCutoverActivationError(
            "existing cutover belongs to another manifest"
        )
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "existing cutover readback conflicts at %s" % field
            )
    return dict(row)


async def activate_subnet_epoch_cutover_v1(
    *,
    cutover: SubnetEpochCutover,
    apply: bool = False,
    select_rows: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = select_all,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    preflight_rpc: Callable[[str, Mapping[str, Any]], Awaitable[Any]] = call_rpc,
    bind_rpc: Callable[[str, Mapping[str, Any]], Awaitable[Any]] = call_rpc,
    stage_rpc: Callable[[str, Mapping[str, Any]], Awaitable[Any]] = call_rpc,
    execute: Callable[..., Awaitable[Mapping[str, Any]]] = execute_coordinator_v2,
    persist_graph: Callable[..., Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_initialization: Callable[
        [int], Awaitable[Optional[Mapping[str, Any]]]
    ] = _load_epoch_initialization,
    load_live_snapshot: Callable[
        [SubnetEpochCutover], Awaitable[SubnetEpochSnapshot]
    ] = _load_live_finalized_snapshot,
    prepare_initialization: Callable[..., Awaitable[Mapping[str, Any]]] = (
        _prepare_epoch_initialization
    ),
    boot_verifier: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
    release_manifest_path: Optional[Path] = None,
    validator_release_manifest_path: Optional[Path] = None,
    validate_anchor: Callable[[SubnetEpochCutover], Awaitable[None]] = (
        _validate_official_archive_anchor
    ),
    predecessor_kind: str = NATIVE_PREDECESSOR_KIND,
    load_cutover_readiness: Callable[..., Awaitable[Mapping[str, Any]]] = (
        champion_v2_cutover_readiness
    ),
) -> Dict[str, Any]:
    """Validate, and only when requested activate, one canonical cutover."""

    try:
        await validate_anchor(cutover)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "official archive rejected the cutover anchor"
        ) from exc
    manifest = cutover.to_dict()
    existing_rows = await select_rows(
        CUTOVER_TABLE,
        filters=(("mapping_hash", cutover.mapping_hash),),
        batch_size=2,
        max_rows=2,
    )
    if existing_rows:
        if len(existing_rows) != 1 or not isinstance(existing_rows[0], Mapping):
            raise StatefulEpochCutoverActivationError(
                "existing cutover mapping is ambiguous"
            )
        existing = dict(existing_rows[0])
        graph = await load_graph(str(existing.get("cutover_receipt_hash") or ""))
        durable = _assert_existing_cutover(
            existing,
            cutover=cutover,
            receipt_graph=graph,
        )
        resolved_boot_verifier = boot_verifier
        if resolved_boot_verifier is None:
            resolved_boot_verifier = build_cutover_mixed_boot_verifier_v1(
                _load_gateway_release(release_manifest_path),
                validator_release_manifest=_load_validator_release(
                    validator_release_manifest_path
                ),
            )
        validate_receipt_graph(
            graph,
            required_purposes={CUTOVER_PURPOSE},
            boot_attestation_verifier=resolved_boot_verifier,
            require_boot_attestation_verification=True,
        )
        with _stateful_epoch_environment(cutover):
            initialization_row = await load_initialization(
                cutover.first_settlement_epoch_id
            )
        initialization_row = _validate_initialization(
            initialization_row,
            cutover=cutover,
            snapshot_doc=durable["first_snapshot_doc"],
        )
        state_rows = await select_rows(
            CUTOVER_STATE_TABLE,
            filters=(("singleton", True),),
            batch_size=2,
            max_rows=2,
        )
        state = _assert_cutover_state(
            state_rows,
            cutover=cutover,
            cutover_row=durable,
            initialization=initialization_row,
        )
        return {
            "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
            "mode": "apply" if apply else "dry_run",
            "status": "already_%s" % state["lifecycle_state"],
            "mapping_hash": cutover.mapping_hash,
            "candidate_snapshot_hash": durable["first_snapshot_hash"],
            "last_legacy_bundle_hash": durable["last_legacy_bundle_hash"],
            "cutover_authority_hash": durable["cutover_authority_hash"],
            "cutover_receipt_hash": durable["cutover_receipt_hash"],
            "durable_readback_hash": sha256_json(durable),
            "fence_state": state["lifecycle_state"],
            "initialization": {
                "exists": True,
                "status": "durable",
                "eligible": True,
                "event_hash": initialization_row.get("event_hash"),
                "nonce": initialization_row.get("nonce"),
                "payload_hash": initialization_row.get("payload_hash"),
                "authority_hash": sha256_json(durable["first_snapshot_doc"]),
            },
        }

    preboundary_state_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    _require_preboundary_fence(preboundary_state_rows, cutover=cutover)

    candidate = await _select_exactly_one(
        select_rows,
        CANDIDATE_TABLE,
        filters=(
            ("mapping_hash", cutover.mapping_hash),
            ("network_genesis_hash", cutover.network_genesis_hash),
            ("netuid", cutover.netuid),
            ("current_block", cutover.cutover_block),
            ("block_hash", cutover.cutover_block_hash),
            ("subnet_epoch_index", cutover.first_subnet_epoch_index),
            (
                "proposed_settlement_epoch_id",
                cutover.first_settlement_epoch_id,
            ),
        ),
        label="stateful subnet epoch candidate",
    )
    snapshot_graph = await load_graph(
        str(candidate.get("chain_state_receipt_hash") or "")
    )
    candidate = validate_stored_pre_cutover_candidate_row_v1(
        candidate,
        cutover=manifest,
        receipt_graph=snapshot_graph,
    )

    if predecessor_kind == HISTORICAL_PREDECESSOR_KIND:
        readiness = dict(
            await load_cutover_readiness(
                epoch=cutover.first_settlement_epoch_id,
                netuid=cutover.netuid,
            )
        )
        if (
            readiness.get("ready") is not True
            or readiness.get("historical_classification_coverage") != 1.0
            or readiness.get("missing_historical_classifications")
        ):
            raise StatefulEpochCutoverActivationError(
                "historical cutover classification coverage is incomplete"
            )
        rows = await select_rows(
            HISTORICAL_FINALIZATION_TABLE,
            filters=(
                ("netuid", cutover.netuid),
                ("epoch_id", "lte", cutover.last_legacy_epoch_id),
            ),
            order_by=(("epoch_id", True),),
            batch_size=1,
            max_rows=1,
            allow_partial=True,
        )
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], Mapping)
        ):
            raise StatefulEpochCutoverActivationError(
                "attested historical cutover predecessor is unavailable"
            )
        finalization = dict(rows[0])
        finalization_doc = finalization.get("settlement_doc")
        if (
            not isinstance(finalization_doc, Mapping)
            or finalization.get("settlement_receipt_hash") is None
            or int(finalization.get("netuid", -1)) != cutover.netuid
            or int(finalization.get("epoch_id", -1))
            > cutover.last_legacy_epoch_id
        ):
            raise StatefulEpochCutoverActivationError(
                "attested historical cutover predecessor differs"
            )
        predecessor_receipt_hash = str(
            finalization["settlement_receipt_hash"]
        )
        finalization_graph = await load_graph(predecessor_receipt_hash)
        payload = {
            "schema_version": CUTOVER_REQUEST_SCHEMA_VERSION,
            "manifest": manifest,
            "first_snapshot": candidate["snapshot_doc"],
            "predecessor_kind": HISTORICAL_PREDECESSOR_KIND,
            "predecessor_finalization": dict(finalization_doc),
        }
        predecessor_artifacts = (
            str(finalization["allocation_hash"]),
            str(finalization["settlement_hash"]),
        )
    elif predecessor_kind == NATIVE_PREDECESSOR_KIND:
        finalization = await _select_exactly_one(
            select_rows,
            FINALIZED_ALLOCATION_VIEW,
            filters=(
                ("netuid", cutover.netuid),
                ("epoch_id", cutover.last_legacy_epoch_id),
                ("validator_hotkey", candidate["validator_hotkey"]),
            ),
            label="last legacy finalized allocation",
        )
        finalization_doc = finalization.get("finalization_doc")
        if (
            not isinstance(finalization_doc, Mapping)
            or finalization.get("bundle_hash") is None
            or finalization.get("weight_finalization_event_hash") is None
            or finalization.get("finalization_receipt_hash") is None
            or int(finalization.get("netuid", -1)) != cutover.netuid
            or int(finalization.get("epoch_id", -1))
            != cutover.last_legacy_epoch_id
            or finalization.get("validator_hotkey")
            != candidate["validator_hotkey"]
            or int(finalization.get("finalized_block", -1))
            >= cutover.cutover_block
        ):
            raise StatefulEpochCutoverActivationError(
                "last legacy finalized allocation differs from the manifest"
            )
        predecessor_receipt_hash = str(
            finalization["finalization_receipt_hash"]
        )
        finalization_graph = await load_graph(predecessor_receipt_hash)
        payload = {
            "schema_version": CUTOVER_REQUEST_SCHEMA_VERSION,
            "manifest": manifest,
            "first_snapshot": candidate["snapshot_doc"],
            "last_legacy_bundle_hash": str(finalization["bundle_hash"]),
            "last_legacy_finalization": dict(finalization_doc),
        }
        predecessor_artifacts = (
            str(finalization["bundle_hash"]),
            str(finalization["weight_finalization_event_hash"]),
        )
    else:
        raise StatefulEpochCutoverActivationError(
            "cutover predecessor kind is invalid"
        )
    parent_graphs = (snapshot_graph, finalization_graph)
    parent_roots = tuple(
        sorted(str(graph["root_receipt_hash"]) for graph in parent_graphs)
    )
    predicted_authority = attest_subnet_epoch_cutover_v2(
        payload,
        ExecutionContextV2(
            job_id="subnet-epoch-cutover-preflight:%s" % cutover.mapping_hash,
            purpose=CUTOVER_PURPOSE,
            epoch_id=cutover.first_settlement_epoch_id,
            parent_receipt_hashes=parent_roots,
            external_receipt_graphs=list(parent_graphs),
        ),
    )
    if (
        predecessor_kind == NATIVE_PREDECESSOR_KIND
        and predicted_authority[
            "last_legacy_weight_finalization_event_hash"
        ]
        != finalization["weight_finalization_event_hash"]
    ):
        raise StatefulEpochCutoverActivationError(
            "last legacy finalization event hash differs from durable state"
        )
    authority_hash = sha256_json(predicted_authority)
    coordinator_rows = await select_rows(
        RECEIPT_TABLE,
        filters=(
            ("role", "gateway_coordinator"),
            ("purpose", CUTOVER_PURPOSE),
            ("epoch_id", cutover.first_settlement_epoch_id),
            ("receipt_status", "succeeded"),
            ("output_root", authority_hash),
        ),
        batch_size=2,
        max_rows=2,
    )
    if not isinstance(coordinator_rows, list) or len(coordinator_rows) > 1:
        raise StatefulEpochCutoverActivationError(
            "existing coordinator cutover receipt is ambiguous"
        )
    resumed_graph = None
    resumed_receipt_hash = None
    if coordinator_rows:
        if not isinstance(coordinator_rows[0], Mapping):
            raise StatefulEpochCutoverActivationError(
                "existing coordinator cutover receipt is invalid"
            )
        resumed_receipt_hash = str(
            coordinator_rows[0].get("receipt_hash") or ""
        )
        resumed_graph = await load_graph(resumed_receipt_hash)
        resumed_boot_verifier = boot_verifier
        if resumed_boot_verifier is None:
            resumed_boot_verifier = build_cutover_mixed_boot_verifier_v1(
                _load_gateway_release(release_manifest_path),
                validator_release_manifest=_load_validator_release(
                    validator_release_manifest_path
                ),
                parent_graphs=parent_graphs,
            )
        validate_receipt_graph(
            resumed_graph,
            required_purposes={CUTOVER_PURPOSE},
            boot_attestation_verifier=resumed_boot_verifier,
            require_boot_attestation_verification=True,
        )
        resumed_row = build_cutover_row_v1(
            authority_doc=predicted_authority,
            first_snapshot_doc=candidate["snapshot_doc"],
            receipt_graph=resumed_graph,
        )
        if resumed_row["cutover_receipt_hash"] != resumed_receipt_hash:
            raise StatefulEpochCutoverActivationError(
                "existing coordinator cutover graph differs"
            )

    preflight_value = await preflight_rpc(
        CUTOVER_PREFLIGHT_RPC,
        {
            "p_mapping_hash": cutover.mapping_hash,
            "p_cutover_receipt_hash": resumed_receipt_hash,
        },
    )
    _normalize_preflight(
        preflight_value,
        cutover=cutover,
        candidate=candidate,
    )
    initialization_precheck = await _initialization_status(
        cutover=cutover,
        snapshot_doc=candidate["snapshot_doc"],
        load_initialization=load_initialization,
        load_live_snapshot=load_live_snapshot,
    )
    if not apply:
        return {
            "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
            "mode": "dry_run",
            "status": "eligible",
            "mapping_hash": cutover.mapping_hash,
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "predecessor_kind": predecessor_kind,
            "predecessor_epoch_id": int(finalization["epoch_id"]),
            "predecessor_receipt_hash": predecessor_receipt_hash,
            "predicted_cutover_authority_hash": authority_hash,
            "coordinator_receipt_exists": resumed_graph is not None,
            "would_write": False,
            "would_transition": [
                "legacy_open",
                "cutover_fenced",
                "stateful_staged",
            ],
            "requires_separate_activation": True,
            "initialization": initialization_precheck,
        }
    if (
        not initialization_precheck["exists"]
        and not initialization_precheck["eligible"]
    ):
        raise StatefulEpochCutoverActivationError(
            "first stateful epoch initialization window has closed before cutover",
            report={
                "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
                "status": "ineligible_before_cutover",
                "mapping_hash": cutover.mapping_hash,
                "initialization": initialization_precheck,
            },
        )

    binding_rpc_name = (
        CUTOVER_BOOTSTRAP_BIND_RPC
        if predecessor_kind == HISTORICAL_PREDECESSOR_KIND
        else CUTOVER_BIND_RPC
    )
    binding_value = await bind_rpc(
        binding_rpc_name,
        {
            "p_mapping_hash": cutover.mapping_hash,
            "p_cutover_authority_hash": authority_hash,
            "p_last_legacy_finalization_receipt_hash": (
                predecessor_receipt_hash
            ),
            "p_cutover_receipt_hash": resumed_receipt_hash,
        },
    )
    binding = _normalize_binding(
        binding_value,
        cutover=cutover,
        candidate=candidate,
        authority_hash=authority_hash,
    )
    if binding["lifecycle_state"] != "cutover_fenced":
        raise StatefulEpochCutoverActivationError(
            "cutover has an unexpected durable lifecycle state before staging"
        )

    if resumed_graph is None:
        release = None
        resolved_boot_verifier = boot_verifier
        if resolved_boot_verifier is None:
            release = _load_gateway_release(release_manifest_path)
            resolved_boot_verifier = build_cutover_mixed_boot_verifier_v1(
                release,
                validator_release_manifest=_load_validator_release(
                    validator_release_manifest_path
                ),
                parent_graphs=parent_graphs,
            )
        execute_kwargs: Dict[str, Any] = {
            "operation": OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
            "purpose": CUTOVER_PURPOSE,
            "epoch_id": cutover.first_settlement_epoch_id,
            "sequence": 0,
            "payload": payload,
            "parent_graphs": parent_graphs,
            "input_artifact_hashes": (
                cutover.mapping_hash,
                candidate["snapshot_hash"],
                *predecessor_artifacts,
            ),
            "persist_graph": persist_graph,
            "boot_verifier": resolved_boot_verifier,
        }
        if release is not None:
            execute_kwargs["release_manifest"] = release
        elif release_manifest_path is not None:
            execute_kwargs["release_manifest_path"] = Path(
                release_manifest_path
            )
        outcome = await execute(**execute_kwargs)
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("status") != "succeeded"
            or outcome.get("result") != predicted_authority
            or not isinstance(outcome.get("receipt_graph"), Mapping)
        ):
            raise StatefulEpochCutoverActivationError(
                "coordinator cutover attestation differs from preflight"
            )
        coordinator_graph = outcome["receipt_graph"]
    else:
        coordinator_graph = resumed_graph
    cutover_row = build_cutover_row_v1(
        authority_doc=predicted_authority,
        first_snapshot_doc=candidate["snapshot_doc"],
        receipt_graph=coordinator_graph,
    )
    if (
        cutover_row.get("cutover_authority_hash") != authority_hash
        or cutover_row.get("mapping_hash") != cutover.mapping_hash
    ):
        raise StatefulEpochCutoverActivationError(
            "cutover row differs from coordinator authority"
        )

    initialization_recheck = await _initialization_status(
        cutover=cutover,
        snapshot_doc=candidate["snapshot_doc"],
        load_initialization=load_initialization,
        load_live_snapshot=load_live_snapshot,
    )
    if initialization_recheck["exists"] or not initialization_recheck["eligible"]:
        raise StatefulEpochCutoverActivationError(
            "first stateful initialization became ineligible while fenced",
            report={
                "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
                "status": "cutover_fenced_not_staged",
                "mapping_hash": cutover.mapping_hash,
                "fence_state": "cutover_fenced",
                "initialization": initialization_recheck,
            },
        )

    from gateway.utils.epoch import get_current_epoch_times

    candidate_snapshot = SubnetEpochSnapshot.from_mapping(candidate["snapshot_doc"])
    epoch_start, epoch_end, epoch_close = get_current_epoch_times(
        candidate_snapshot
    )
    try:
        with _stateful_epoch_environment(cutover):
            initialization_event = dict(
                await prepare_initialization(
                    epoch_id=cutover.first_settlement_epoch_id,
                    epoch_start=epoch_start,
                    epoch_end=epoch_end,
                    epoch_close=epoch_close,
                    epoch_snapshot=candidate_snapshot,
                )
            )
        _validate_initialization(
            initialization_event,
            cutover=cutover,
            snapshot_doc=candidate["snapshot_doc"],
        )
        stage_rpc_name = (
            CUTOVER_BOOTSTRAP_STAGE_RPC
            if predecessor_kind == HISTORICAL_PREDECESSOR_KIND
            else CUTOVER_STAGE_RPC
        )
        stage_value = await stage_rpc(
            stage_rpc_name,
            {
                "p_cutover_row": cutover_row,
                "p_initialization_event": initialization_event,
            },
        )
        stage = _normalize_stage(
            stage_value,
            cutover_row=cutover_row,
            initialization_event=initialization_event,
        )
    except StatefulEpochCutoverActivationError:
        raise
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "atomic cutover staging failed while the write fence remains closed",
            report={
                "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
                "status": "cutover_fenced_not_staged",
                "mapping_hash": cutover.mapping_hash,
                "fence_state": "cutover_fenced",
            },
        ) from exc

    durable = await _select_exactly_one(
        select_rows,
        CUTOVER_TABLE,
        filters=(("mapping_hash", cutover.mapping_hash),),
        label="atomic stateful subnet epoch cutover readback",
    )
    durable = _assert_existing_cutover(
        durable,
        cutover=cutover,
        receipt_graph=coordinator_graph,
    )
    with _stateful_epoch_environment(cutover):
        initialization_row = await load_initialization(
            cutover.first_settlement_epoch_id
        )
    initialization_row = _validate_initialization(
        initialization_row,
        cutover=cutover,
        snapshot_doc=candidate["snapshot_doc"],
    )
    state_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    state = _assert_cutover_state(
        state_rows,
        cutover=cutover,
        cutover_row=durable,
        initialization=initialization_row,
    )
    if state["lifecycle_state"] != "stateful_staged":
        raise StatefulEpochCutoverActivationError(
            "cutover staging unexpectedly lifted the write fence"
        )
    return {
        "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
        "mode": "apply",
        "status": "stateful_staged",
        "mapping_hash": cutover.mapping_hash,
        "candidate_snapshot_hash": candidate["snapshot_hash"],
        "predecessor_kind": predecessor_kind,
        "predecessor_epoch_id": int(finalization["epoch_id"]),
        "predecessor_receipt_hash": predecessor_receipt_hash,
        "cutover_authority_hash": authority_hash,
        "cutover_receipt_hash": durable["cutover_receipt_hash"],
        "durable_readback_hash": sha256_json(dict(durable)),
        "fence_state": stage["lifecycle_state"],
        "requires_separate_activation": True,
        "activation_rpc": CUTOVER_ACTIVATE_RPC,
        "initialization": {
            "exists": True,
            "status": "created_atomically",
            "eligible": True,
            "event_hash": initialization_row.get("event_hash"),
            "nonce": initialization_row.get("nonce"),
            "payload_hash": initialization_row.get("payload_hash"),
            "authority_hash": sha256_json(candidate["snapshot_doc"]),
        },
    }


async def activate_staged_subnet_epoch_cutover_v1(
    *,
    cutover: SubnetEpochCutover,
    confirmed_cutover_authority_hash: str,
    select_rows: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = select_all,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    load_initialization: Callable[
        [int], Awaitable[Optional[Mapping[str, Any]]]
    ] = _load_epoch_initialization,
    load_live_snapshot: Callable[
        [SubnetEpochCutover], Awaitable[SubnetEpochSnapshot]
    ] = _load_live_finalized_snapshot,
    activate_rpc: Callable[[str, Mapping[str, Any]], Awaitable[Any]] = call_rpc,
    boot_verifier: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
    release_manifest_path: Optional[Path] = None,
    validator_release_manifest_path: Optional[Path] = None,
    validate_anchor: Callable[[SubnetEpochCutover], Awaitable[None]] = (
        _validate_official_archive_anchor
    ),
) -> Dict[str, Any]:
    """Open one exact staged namespace after its stateful release is prepared."""

    try:
        await validate_anchor(cutover)
    except Exception as exc:
        raise StatefulEpochCutoverActivationError(
            "official archive rejected the staged cutover anchor"
        ) from exc
    cutover_row = await _select_exactly_one(
        select_rows,
        CUTOVER_TABLE,
        filters=(("mapping_hash", cutover.mapping_hash),),
        label="staged stateful subnet epoch cutover",
    )
    if cutover_row.get("cutover_authority_hash") != confirmed_cutover_authority_hash:
        raise StatefulEpochCutoverActivationError(
            "cutover authority confirmation differs from durable state"
        )
    graph = await load_graph(str(cutover_row.get("cutover_receipt_hash") or ""))
    cutover_row = _assert_existing_cutover(
        cutover_row,
        cutover=cutover,
        receipt_graph=graph,
    )
    resolved_boot_verifier = boot_verifier
    if resolved_boot_verifier is None:
        resolved_boot_verifier = build_cutover_mixed_boot_verifier_v1(
            _load_gateway_release(release_manifest_path),
            validator_release_manifest=_load_validator_release(
                validator_release_manifest_path
            ),
            parent_graphs=(graph,),
        )
    validate_receipt_graph(
        graph,
        required_purposes={CUTOVER_PURPOSE},
        boot_attestation_verifier=resolved_boot_verifier,
        require_boot_attestation_verification=True,
    )
    with _stateful_epoch_environment(cutover):
        initialization = await load_initialization(
            cutover.first_settlement_epoch_id
        )
    initialization = _validate_initialization(
        initialization,
        cutover=cutover,
        snapshot_doc=cutover_row["first_snapshot_doc"],
    )
    state_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    state = _assert_cutover_state(
        state_rows,
        cutover=cutover,
        cutover_row=cutover_row,
        initialization=initialization,
    )
    if state["lifecycle_state"] not in {"stateful_staged", "stateful_active"}:
        raise StatefulEpochCutoverActivationError(
            "cutover is not staged for activation"
        )
    if state["lifecycle_state"] == "stateful_staged":
        live_status = _live_initialization_window_status(
            cutover=cutover,
            snapshot_doc=cutover_row["first_snapshot_doc"],
            live=await load_live_snapshot(cutover),
        )
        if not live_status["eligible"]:
            raise StatefulEpochCutoverActivationError(
                "stateful runtime activation missed the first-epoch restart window",
                report={
                    "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
                    "status": "stateful_staged_activation_window_closed",
                    "mapping_hash": cutover.mapping_hash,
                    "initialization": live_status,
                },
            )

    value = await activate_rpc(
        CUTOVER_ACTIVATE_RPC,
        {
            "p_mapping_hash": cutover.mapping_hash,
            "p_confirm_stateful_release_prepared": True,
        },
    )
    result = _one_rpc_row(
        value,
        fields=_STAGE_FIELDS,
        label="stateful cutover activation",
    )
    expected = {
        "lifecycle_state": "stateful_active",
        "mapping_hash": cutover.mapping_hash,
        "cutover_authority_hash": confirmed_cutover_authority_hash,
        "cutover_receipt_hash": cutover_row["cutover_receipt_hash"],
        "initialization_nonce": initialization["nonce"],
        "initialization_payload_hash": initialization["payload_hash"],
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise StatefulEpochCutoverActivationError(
                "stateful cutover activation differs at %s" % field
            )
    readback_rows = await select_rows(
        CUTOVER_STATE_TABLE,
        filters=(("singleton", True),),
        batch_size=2,
        max_rows=2,
    )
    readback = _assert_cutover_state(
        readback_rows,
        cutover=cutover,
        cutover_row=cutover_row,
        initialization=initialization,
    )
    if readback["lifecycle_state"] != "stateful_active":
        raise StatefulEpochCutoverActivationError(
            "stateful cutover activation exact readback failed"
        )
    return {
        "schema_version": ACTIVATION_REPORT_SCHEMA_VERSION,
        "mode": "activate_staged",
        "status": "stateful_active",
        "mapping_hash": cutover.mapping_hash,
        "cutover_authority_hash": confirmed_cutover_authority_hash,
        "cutover_receipt_hash": cutover_row["cutover_receipt_hash"],
        "initialization_nonce": initialization["nonce"],
        "initialization_payload_hash": initialization["payload_hash"],
        "next_step": "start only the prepared stateful runtimes and verify loaded production evidence",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--validator-release-manifest", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--fence-before-boundary", action="store_true")
    mode.add_argument("--propose-manifest", action="store_true")
    mode.add_argument("--activate-staged", action="store_true")
    parser.add_argument("--confirm-mapping-hash")
    parser.add_argument("--confirm-cutover-authority-hash")
    parser.add_argument("--network-genesis-hash")
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--last-legacy-epoch-id", type=int)
    parser.add_argument("--first-settlement-epoch-id", type=int)
    parser.add_argument("--confirm-first-settlement-epoch-id", type=int)
    parser.add_argument(
        "--acknowledge-reserved-first-settlement-ordinal",
        action="store_true",
    )
    parser.add_argument("--confirm-all-writers-stopped", action="store_true")
    parser.add_argument("--confirm-stateful-release-prepared", action="store_true")
    parser.add_argument(
        "--use-attested-historical-predecessor",
        action="store_true",
        help=(
            "bind the latest receipt-backed finalized historical allocation "
            "at or below the fenced namespace high-water"
        ),
    )
    args = parser.parse_args(argv)

    if (args.apply or args.activate_staged) and args.release_manifest is None:
        parser.error(
            "mutating cutover modes require an explicit --release-manifest"
        )
    if (
        args.apply or args.activate_staged
    ) and args.validator_release_manifest is None:
        parser.error(
            "mutating cutover modes require an explicit "
            "--validator-release-manifest"
        )
    if (args.apply or args.activate_staged) and not args.confirm_all_writers_stopped:
        parser.error(
            "mutating cutover modes require --confirm-all-writers-stopped"
        )

    if args.fence_before_boundary:
        required = (
            args.network_genesis_hash,
            args.netuid,
            args.last_legacy_epoch_id,
            args.first_settlement_epoch_id,
        )
        if any(value is None for value in required):
            parser.error(
                "--fence-before-boundary requires genesis, netuid, last legacy, and first settlement IDs"
            )
        if not args.acknowledge_reserved_first_settlement_ordinal:
            parser.error(
                "--fence-before-boundary requires "
                "--acknowledge-reserved-first-settlement-ordinal"
            )
        if args.confirm_first_settlement_epoch_id != args.first_settlement_epoch_id:
            parser.error(
                "--fence-before-boundary requires an exact first-settlement confirmation"
            )
        report = asyncio.run(
            fence_subnet_epoch_namespace_v1(
                network_genesis_hash=args.network_genesis_hash,
                netuid=args.netuid,
                last_legacy_epoch_id=args.last_legacy_epoch_id,
                first_settlement_epoch_id=args.first_settlement_epoch_id,
            )
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0

    if args.propose_manifest:
        if (
            args.network_genesis_hash is None
            or args.netuid is None
            or args.last_legacy_epoch_id is None
        ):
            parser.error(
                "--propose-manifest requires genesis, netuid, and last legacy ID"
            )
        report = asyncio.run(
            propose_subnet_epoch_cutover_manifest_v1(
                network_genesis_hash=args.network_genesis_hash,
                netuid=args.netuid,
                last_legacy_epoch_id=args.last_legacy_epoch_id,
            )
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0

    cutover = load_cutover_manifest_v1(args.manifest)
    if (args.apply or args.activate_staged) and args.confirm_mapping_hash != cutover.mapping_hash:
        parser.error(
            "mutating cutover modes require --confirm-mapping-hash equal to the canonical manifest"
        )
    if args.activate_staged:
        if not args.confirm_stateful_release_prepared:
            parser.error(
                "--activate-staged requires --confirm-stateful-release-prepared"
            )
        if not args.confirm_cutover_authority_hash:
            parser.error(
                "--activate-staged requires --confirm-cutover-authority-hash"
            )
        report = asyncio.run(
            activate_staged_subnet_epoch_cutover_v1(
                cutover=cutover,
                confirmed_cutover_authority_hash=args.confirm_cutover_authority_hash,
                release_manifest_path=args.release_manifest,
                validator_release_manifest_path=args.validator_release_manifest,
            )
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    try:
        report = asyncio.run(
            activate_subnet_epoch_cutover_v1(
                cutover=cutover,
                apply=bool(args.apply),
                release_manifest_path=args.release_manifest,
                validator_release_manifest_path=args.validator_release_manifest,
                predecessor_kind=(
                    HISTORICAL_PREDECESSOR_KIND
                    if args.use_attested_historical_predecessor
                    else NATIVE_PREDECESSOR_KIND
                ),
            )
        )
    except StatefulEpochCutoverActivationError as exc:
        if exc.report is None:
            raise
        report = {**exc.report, "error": str(exc)}
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
