"""Research Lab live allocation projection for validator consumption."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hmac
import logging
import os
from typing import Any, Mapping, Sequence

from gateway.research_lab.alpha_pricing import (
    inject_alpha_price_valuation,
    resolve_epoch_alpha_price_valuation,
)
from gateway.research_lab.tee_protocol import legacy_v1_enabled
from gateway.research_lab.v2_authority import build_allocation_v2
from gateway.research_lab.bundles import contains_secret_material, sha256_json
from gateway.research_lab.chain import resolve_hotkey_uids
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import (
    _is_transient_store_error,
    create_research_lab_emission_allocation_snapshot,
    select_all,
    select_many,
)
from leadpoet_verifier.economics import (
    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
    allocate_research_lab_epoch,
)


ACTIVE_REIMBURSEMENT_STATUSES = {"awarded"}
ACTIVE_SCHEDULE_STATUSES = {"scheduled"}
ACTIVE_CHAMPION_STATUSES = {"active", "queued", "partially_paid"}
SETTLEMENT_TRACKED_CHAMPION_STATUSES = ACTIVE_CHAMPION_STATUSES | {"paid"}
RATE_QUANT = Decimal("0.000001")
_SOURCE_ADD_CHAIN_QUANTIZATION_TOLERANCE_PERCENT = Decimal("100") / Decimal(
    "65535"
)
POSTGREST_IN_FILTER_CHUNK = 50
LATEST_NATIVE_COMPUTE_AUTHORITY_TABLE = (
    "research_lab_finalized_allocation_epochs_v2"
)
LATEST_LEGACY_COMPUTE_AUTHORITY_TABLE = (
    "research_lab_legacy_finalized_allocation_migrations_v2"
)
logger = logging.getLogger(__name__)
_ALLOCATION_V2_INFLIGHT: dict[
    tuple[int, int, int, str],
    asyncio.Task[dict[str, Any]],
] = {}
_ALLOCATION_V2_RETRY_GENERATIONS: dict[
    tuple[int, int, int, str],
    tuple[float, int],
] = {}
_ALLOCATION_V2_RETRY_GENERATION_TTL_SECONDS = 3600.0
_ALLOCATION_V2_RETRY_MAX_GENERATIONS = 8
_ALLOCATION_V2_RETRY_MIN_INTERVAL_SECONDS = 300.0
_ALLOCATION_V2_RECEIPT_TABLE = "research_lab_attested_execution_receipts_v2"
_ALLOCATION_V2_RECEIPT_ROLE = "gateway_coordinator"
_ALLOCATION_V2_RECEIPT_PURPOSE = "research_lab.allocation.v2"


def _active_gateway_commit() -> str:
    from gateway.build_info import get_build_info

    commit = str(get_build_info().get("git_commit") or "").strip().lower()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("active gateway commit is unavailable for allocation retry")
    return commit


def _receipt_issued_at(row: Mapping[str, Any]) -> datetime | None:
    """Read a receipt's issue time, treating anything unparsable as unknown."""

    raw = row.get("issued_at") or row.get("created_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def _load_durable_allocation_retry_generation(
    *,
    epoch_id: int,
) -> int:
    """Resume after failed allocation receipts from an earlier process.

    The retry allowance is spent over time, not over calls. Only failures
    inside the retry window count against it, so a transiently broken epoch
    becomes buildable again instead of staying refused for the life of the
    running gateway, and a fresh build is not started while the previous
    failure is still fresh, so an impatient caller cannot spend the whole
    allowance in a few minutes.
    """

    commit_sha = _active_gateway_commit()
    rows = await select_many(
        _ALLOCATION_V2_RECEIPT_TABLE,
        columns=(
            "sequence,receipt_status,failure_code,job_id,commit_sha,"
            "epoch_id,purpose,role,issued_at"
        ),
        filters=(
            ("role", _ALLOCATION_V2_RECEIPT_ROLE),
            ("purpose", _ALLOCATION_V2_RECEIPT_PURPOSE),
            ("epoch_id", int(epoch_id)),
            ("commit_sha", str(commit_sha)),
            ("receipt_status", "failed"),
        ),
        order_by=(("sequence", True),),
        limit=_ALLOCATION_V2_RETRY_MAX_GENERATIONS,
    )
    if not rows:
        return 0

    now = datetime.now(timezone.utc)
    highest_sequence = -1
    recent_failures = 0
    newest_failure_at: datetime | None = None
    for row in rows:
        sequence = row.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or row.get("role") != _ALLOCATION_V2_RECEIPT_ROLE
            or row.get("purpose") != _ALLOCATION_V2_RECEIPT_PURPOSE
            or int(row.get("epoch_id", -1)) != int(epoch_id)
            or str(row.get("commit_sha") or "").lower() != str(commit_sha)
            or row.get("receipt_status") != "failed"
            or not str(row.get("failure_code") or "").strip()
            or not str(row.get("job_id") or "").startswith(
                "scoring-v2:research-lab-allocation:"
            )
        ):
            raise RuntimeError("durable allocation retry receipt is invalid")
        highest_sequence = max(highest_sequence, sequence)

        issued_at = _receipt_issued_at(row)
        if issued_at is None:
            # An unreadable timestamp is treated as a failure that just
            # happened, so a malformed row can only ever be conservative.
            issued_at = now
        age_seconds = (now - issued_at).total_seconds()
        if age_seconds <= _ALLOCATION_V2_RETRY_GENERATION_TTL_SECONDS:
            recent_failures += 1
        if newest_failure_at is None or issued_at > newest_failure_at:
            newest_failure_at = issued_at

    if (
        newest_failure_at is not None
        and (now - newest_failure_at).total_seconds()
        < _ALLOCATION_V2_RETRY_MIN_INTERVAL_SECONDS
    ):
        raise RuntimeError("durable allocation retry is cooling down")
    if recent_failures >= _ALLOCATION_V2_RETRY_MAX_GENERATIONS:
        raise RuntimeError("durable allocation retry generations are exhausted")
    return highest_sequence + 1


def _allocation_v2_build_is_retryable(exc: BaseException) -> bool:
    """Retry transport/source failures without replaying policy failures."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "AttestedScoringV2Error":
            if "execution_supabasesourcev2error" in str(current).lower():
                return True
        if type(current).__name__ in {"SupabaseSourceV2Error", "TEEClientError"}:
            return True
        if _is_transient_store_error(current):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _build_allocation_v2_singleflight(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Share only overlapping identical authority builds in this event loop."""

    loop = asyncio.get_running_loop()
    key = (
        id(loop),
        int(epoch_id),
        int(netuid),
        sha256_json(dict(policy)),
    )
    task = _ALLOCATION_V2_INFLIGHT.get(key)
    if task is None:
        retry_state = _ALLOCATION_V2_RETRY_GENERATIONS.get(key)
        memory_generation = 0
        if retry_state is not None:
            retry_expires_at, memory_generation = retry_state
            if retry_expires_at <= loop.time():
                _ALLOCATION_V2_RETRY_GENERATIONS.pop(key, None)
                memory_generation = 0
        selected_generation: list[int] = []

        async def build_with_durable_generation() -> dict[str, Any]:
            durable_generation = await _load_durable_allocation_retry_generation(
                epoch_id=int(epoch_id),
            )
            if int(memory_generation) >= _ALLOCATION_V2_RETRY_MAX_GENERATIONS:
                raise RuntimeError("allocation retry generations are exhausted")
            generation = max(int(memory_generation), int(durable_generation))
            selected_generation.append(generation)
            return await build_allocation_v2(
                epoch_id=int(epoch_id),
                netuid=int(netuid),
                policy=dict(policy),
                allocation_sequence=generation,
            )

        task = loop.create_task(build_with_durable_generation())
        _ALLOCATION_V2_INFLIGHT[key] = task

        def clear(completed: asyncio.Task[dict[str, Any]]) -> None:
            if _ALLOCATION_V2_INFLIGHT.get(key) is completed:
                _ALLOCATION_V2_INFLIGHT.pop(key, None)
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException as exc:
                if selected_generation and _allocation_v2_build_is_retryable(exc):
                    _ALLOCATION_V2_RETRY_GENERATIONS[key] = (
                        loop.time()
                        + _ALLOCATION_V2_RETRY_GENERATION_TTL_SECONDS,
                        selected_generation[0] + 1,
                    )
                return
            _ALLOCATION_V2_RETRY_GENERATIONS.pop(key, None)

        task.add_done_callback(clear)
    return await asyncio.shield(task)


async def build_research_lab_allocation_bundle(
    *,
    config: ResearchLabGatewayConfig,
    epoch: int,
    netuid: int,
    persist_snapshot: bool = False,
    attestation_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sanitized Research Lab allocation bundle for one epoch."""
    if legacy_v1_enabled():
        policy = config.reimbursement_policy_doc(enabled=True)
        alpha_valuation = await resolve_epoch_alpha_price_valuation(
            network=_bittensor_network(),
            netuid=int(netuid),
            epoch=int(epoch),
            enabled=bool(config.reimbursement_dynamic_alpha_price_enabled),
            require_live=bool(config.reimbursement_require_live_alpha_price),
            miner_alpha_per_epoch=config.reimbursement_miner_alpha_per_epoch,
            static_usd_per_0_1_percent_epoch=(
                config.reimbursement_usd_per_0_1_percent_epoch
            ),
        )
        policy = inject_alpha_price_valuation(policy, alpha_valuation)
        reimbursement_obligations, reimbursement_skipped = (
            await _active_reimbursement_obligations(int(epoch), policy=policy)
        )
        champion_obligations, champion_skipped = (
            await _active_champion_obligations(
                int(epoch),
                netuid=int(netuid),
                enable_champ_cap=bool(config.enable_champ_cap),
            )
        )
        source_add_obligations, source_add_skipped = (
            await _active_source_add_obligations(int(epoch), netuid=int(netuid))
        )
        source_add_present = bool(source_add_obligations or source_add_skipped)
        fallback_reimbursement_obligations: list[dict[str, Any]] = []
        fallback_reimbursement_skipped: list[dict[str, Any]] = []
        fallback_source: dict[str, Any] = {}
        if not bool(config.enable_conservative):
            (
                fallback_reimbursement_obligations,
                fallback_reimbursement_skipped,
                fallback_source,
            ) = await _historical_compute_fallback_obligations(
                epoch=int(epoch),
                netuid=int(netuid),
                policy=policy,
            )
        allocation_inputs = {
            "epoch": int(epoch),
            "policy": policy,
            "active_reimbursement_obligations": reimbursement_obligations,
            "active_champion_obligations": champion_obligations,
        }
        if fallback_reimbursement_obligations:
            allocation_inputs["fallback_reimbursement_obligations"] = (
                fallback_reimbursement_obligations
            )
        if source_add_present:
            allocation_inputs["active_source_add_obligations"] = source_add_obligations
        allocation = allocate_research_lab_epoch(
            allocation_inputs["epoch"],
            allocation_inputs["policy"],
            allocation_inputs["active_reimbursement_obligations"],
            allocation_inputs["active_champion_obligations"],
            active_source_add_obligations=allocation_inputs.get(
                "active_source_add_obligations", []
            ),
            fallback_reimbursement_obligations=allocation_inputs.get(
                "fallback_reimbursement_obligations", []
            ),
        )
        source_state = {
            "epoch": int(epoch),
            "netuid": int(netuid),
            "policy_id": str(policy["policy_id"]),
            "policy": policy,
            "reimbursement_obligation_count": len(reimbursement_obligations),
            "champion_obligation_count": len(champion_obligations),
            "reimbursement_obligations": reimbursement_obligations,
            "champion_obligations": champion_obligations,
            "skipped": {
                "reimbursements": reimbursement_skipped,
                "champions": champion_skipped,
            },
        }
        if fallback_reimbursement_obligations or fallback_reimbursement_skipped:
            source_state.update(
                {
                    "fallback_reimbursement_obligation_count": len(
                        fallback_reimbursement_obligations
                    ),
                    "fallback_reimbursement_obligations": (
                        fallback_reimbursement_obligations
                    ),
                    "historical_compute_fallback_source": fallback_source,
                }
            )
            source_state["skipped"]["fallback_reimbursements"] = (
                fallback_reimbursement_skipped
            )
        if source_add_present:
            source_state["source_add_obligation_count"] = len(source_add_obligations)
            source_state["source_add_obligations"] = source_add_obligations
            source_state["skipped"]["source_add"] = source_add_skipped
        source_state_hash = sha256_json(source_state)
        attestation = {"status": "off", "protocol": "legacy_v1"}
    else:
        policy = config.reimbursement_policy_doc(enabled=True)
        attestation = await _build_allocation_v2_singleflight(
            epoch_id=int(epoch),
            netuid=int(netuid),
            policy=policy,
        )
        authority = attestation.get("result")
        if not isinstance(authority, Mapping):
            raise ValueError("Research Lab V2 allocation authority result is missing")
        allocation = authority.get("allocation")
        allocation_inputs = authority.get("allocation_inputs")
        source_state = authority.get("source_state")
        if (
            not isinstance(allocation, Mapping)
            or not isinstance(allocation_inputs, Mapping)
            or not isinstance(source_state, Mapping)
        ):
            raise ValueError("Research Lab V2 allocation authority result is invalid")
        allocation = dict(allocation)
        allocation_inputs = dict(allocation_inputs)
        source_state = dict(source_state)
        policy = dict(allocation_inputs.get("policy") or {})
        reimbursement_obligations = list(
            allocation_inputs.get("active_reimbursement_obligations") or []
        )
        champion_obligations = list(
            allocation_inputs.get("active_champion_obligations") or []
        )
        fallback_reimbursement_obligations = list(
            allocation_inputs.get("fallback_reimbursement_obligations") or []
        )
        source_add_present = "active_source_add_obligations" in allocation_inputs
        source_add_obligations = list(
            allocation_inputs.get("active_source_add_obligations") or []
        )
        skipped = source_state.get("skipped")
        if not isinstance(skipped, Mapping):
            raise ValueError("Research Lab V2 allocation skipped-state is invalid")
        reimbursement_skipped = list(skipped.get("reimbursements") or [])
        champion_skipped = list(skipped.get("champions") or [])
        source_add_skipped = list(skipped.get("source_add") or [])
        fallback_reimbursement_skipped = list(
            skipped.get("fallback_reimbursements") or []
        )
        source_state_hash = str(authority.get("source_state_hash") or "")
    if attestation_out is not None:
        attestation_out.clear()
        attestation_out.update(attestation)
    live_allocation_enabled = bool(config.reimbursements_enabled or config.weight_mutation_enabled)
    snapshot_status = "active" if live_allocation_enabled else "shadow"
    if persist_snapshot and config.production_writes_enabled:
        await create_research_lab_emission_allocation_snapshot(
            epoch=int(epoch),
            netuid=int(netuid),
            policy_id=str(policy["policy_id"]),
            snapshot_status=snapshot_status,
            allocation_doc=allocation,
        )
    if contains_secret_material(source_state) or contains_secret_material(allocation):
        raise ValueError("Research Lab allocation bundle contains private or secret material")
    if source_state_hash != sha256_json(source_state):
        raise ValueError("Research Lab allocation source-state hash differs")
    bundle_without_id = {
        "bundle_id": "",
        "schema_version": "1.0",
        "bundle_type": "research_lab_live_allocation_bundle",
        "epoch": int(epoch),
        "netuid": int(netuid),
        "generated_at": _utc_now_iso(),
        "shadow_only": not live_allocation_enabled,
        "read_only": not live_allocation_enabled,
        "submission_allowed": live_allocation_enabled,
        "on_chain_submission_allowed": live_allocation_enabled,
        "source_state_hash": source_state_hash,
        "source_state": source_state,
        "allocation_hash": allocation["allocation_hash"],
        "allocation_doc": allocation,
        "observability": {
            "lab_cap_alpha_percent": float(allocation.get("lab_cap_percent") or 0.0),
            "reimbursement_alpha_percent": float(allocation.get("reimbursement_alpha_percent") or 0.0),
            "champion_alpha_percent": float(allocation.get("champion_alpha_percent") or 0.0),
            "queued_champion_alpha_percent": float(allocation.get("queued_champion_alpha_percent") or 0.0),
            "unallocated_alpha_percent": float(allocation.get("unallocated_percent") or 0.0),
            "reimbursement_allocation_count": len(allocation.get("reimbursement_allocations") or []),
            "champion_allocation_count": len(allocation.get("champion_allocations") or []),
            "queued_champion_allocation_count": len(allocation.get("queued_champion_allocations") or []),
            "skipped_reimbursement_count": len(reimbursement_skipped),
            "skipped_fallback_reimbursement_count": len(
                fallback_reimbursement_skipped
            ),
            "skipped_champion_count": len(champion_skipped),
        },
        "verifier_contract": {
            "required_checks": [
                "secret_payload_absent",
                "source_state_hash_matches",
                "allocation_hash_matches",
                "allocation_recomputes_from_source_state",
                "validator_policy_lab_cap_ceiling",
                "gateway_allows_live_research_lab_weights",
                "validator_flags_allow_live_research_lab_weights",
            ],
        },
    }
    if source_add_present:
        bundle_without_id["observability"].update(
            {
                "source_add_alpha_percent": float(allocation.get("source_add_alpha_percent") or 0.0),
                "champion_reimbursement_cap_percent": float(
                    allocation.get("champion_reimbursement_cap_percent")
                    or allocation.get("lab_cap_percent")
                    or 0.0
                ),
                "source_add_allocation_count": len(allocation.get("source_add_allocations") or []),
                "skipped_source_add_count": len(source_add_skipped),
            }
        )
    bundle_id = "research_lab_allocation_bundle:" + sha256_json(bundle_without_id).split(":", 1)[1]
    return {**bundle_without_id, "bundle_id": bundle_id}


async def _active_reimbursement_obligations(
    epoch: int,
    *,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        epoch_span = max(1, int(policy.get("reimbursement_epochs") or 20))
    except (TypeError, ValueError):
        epoch_span = 20
    schedule_start_floor = max(0, int(epoch) - epoch_span)
    schedule_rows = await select_all(
        "research_reimbursement_schedules",
        filters=(
            ("schedule_status", "scheduled"),
            ("start_epoch", "lte", int(epoch)),
            ("start_epoch", "gte", schedule_start_floor),
        ),
        order_by=(("start_epoch", True),),
    )
    active_schedule_rows = [
        row
        for row in schedule_rows
        if str(row.get("schedule_status") or "") in ACTIVE_SCHEDULE_STATUSES and _epoch_active(row, epoch)
    ]
    award_ids = sorted(
        {
            str(schedule.get("award_id") or "")
            for schedule in active_schedule_rows
            if str(schedule.get("award_id") or "")
        }
    )
    awards_by_id: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(award_ids), POSTGREST_IN_FILTER_CHUNK):
        chunk = award_ids[offset : offset + POSTGREST_IN_FILTER_CHUNK]
        award_rows = await select_all(
            "research_reimbursement_award_current",
            filters=(
                ("award_id", "in", chunk),
                ("current_award_status", "awarded"),
            ),
            max_rows=len(chunk) + 1,
            allow_partial=False,
        )
        for award in award_rows:
            award_id = str(award.get("award_id") or "")
            status = str(
                award.get("current_award_status")
                or award.get("award_status")
                or ""
            )
            if (
                award_id not in chunk
                or status not in ACTIVE_REIMBURSEMENT_STATUSES
            ):
                raise ValueError(
                    "Research Lab reimbursement award batch differs"
                )
            if award_id in awards_by_id:
                raise ValueError(
                    "Research Lab reimbursement award is ambiguous"
                )
            awards_by_id[award_id] = award
    hotkeys = [str(row.get("miner_hotkey") or "") for row in awards_by_id.values()]
    hotkey_uids = await resolve_hotkey_uids(hotkeys)
    obligations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for schedule in active_schedule_rows:
        status = str(schedule.get("schedule_status") or "")
        if status not in ACTIVE_SCHEDULE_STATUSES:
            continue
        award = awards_by_id.get(str(schedule.get("award_id") or ""))
        if not award:
            continue
        miner_hotkey = str(award.get("miner_hotkey") or "")
        uid = hotkey_uids.get(miner_hotkey)
        if uid is None:
            skipped.append({"award_id": str(award.get("award_id") or ""), "reason": "miner_hotkey_not_registered"})
            continue
        obligations.append(
            {
                "uid": uid,
                "miner_uid": uid,
                "miner_hotkey": miner_hotkey,
                "source_id": str(schedule.get("schedule_id") or award.get("award_id") or ""),
                "schedule_id": str(schedule.get("schedule_id") or ""),
                "award_id": str(award.get("award_id") or ""),
                "run_id": str(award.get("run_id") or ""),
                "island": str(award.get("island") or "generalist"),
                "status": "active",
                "start_epoch": int(schedule.get("start_epoch") or 0),
                "epoch_count": int(schedule.get("epoch_count") or 0),
                "target_reimbursement_microusd": int(award.get("target_reimbursement_microusd") or 0),
                "total_microusd": int(schedule.get("total_microusd") or award.get("target_reimbursement_microusd") or 0),
                "eligible_compute_microusd": int(
                    award.get("eligible_cost_microusd")
                    or award.get("target_reimbursement_microusd")
                    or 0
                ),
                "participation_score": float(award.get("participation_score") or 0.0),
            }
        )
    return obligations, skipped


async def _historical_compute_fallback_obligations(
    *,
    epoch: int,
    netuid: int,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load the latest finalized compute allocation for no-burn replay."""

    resolved = await _load_latest_finalized_compute_snapshot_v2(
        epoch=epoch,
        netuid=netuid,
    )
    if resolved is None:
        return (
            [],
            [{"reason": "historical_compute_allocation_unavailable"}],
            {},
        )
    row, _authority = resolved
    allocation_doc = row.get("allocation_doc")
    hotkeys = []
    if isinstance(allocation_doc, Mapping):
        for item in allocation_doc.get("reimbursement_allocations") or ():
            if isinstance(item, Mapping) and item.get("miner_hotkey"):
                hotkeys.append(str(item["miner_hotkey"]))
    hotkey_uids = await resolve_hotkey_uids(hotkeys)
    return _historical_compute_fallback_from_snapshot(
        row,
        hotkey_uids=hotkey_uids,
        reward_epochs=max(
            1,
            int(
                policy.get("reimbursement_epochs")
                or policy.get("reward_epochs")
                or 20
            ),
        ),
        expected_netuid=int(netuid),
    )


async def _load_latest_finalized_compute_snapshot_v2(
    *,
    epoch: int,
    netuid: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Resolve the newest non-recursive compute snapshot with final authority."""

    from gateway.research_lab.champion_settlement_v2 import (
        load_finalized_allocation_history_v2,
    )

    native = await select_all(
        LATEST_NATIVE_COMPUTE_AUTHORITY_TABLE,
        columns="epoch_id,netuid",
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "lt", int(epoch)),
            (
                "bundle_doc->weight_snapshot->calculation_snapshot"
                "->research_lab_allocation_doc->reimbursement_allocations",
                "neq",
                [],
            ),
            (
                "bundle_doc->weight_snapshot->calculation_snapshot"
                "->research_lab_allocation_doc"
                "->>historical_compute_fallback_source_epoch",
                "is",
                "null",
            ),
        ),
        order_by=(("epoch_id", True),),
        batch_size=1,
        max_rows=1,
        allow_partial=True,
    )
    legacy = await select_all(
        LATEST_LEGACY_COMPUTE_AUTHORITY_TABLE,
        columns="netuid,epoch_id,allocation_hash,allocation_doc",
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "lt", int(epoch)),
            ("allocation_doc->reimbursement_allocations", "neq", []),
            (
                "allocation_doc->>historical_compute_fallback_source_epoch",
                "is",
                "null",
            ),
        ),
        order_by=(("epoch_id", True),),
        batch_size=1,
        max_rows=1,
        allow_partial=True,
    )
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    if native:
        candidates.append((int(native[0]["epoch_id"]), "native", native[0]))
    if legacy:
        candidates.append((int(legacy[0]["epoch_id"]), "legacy", legacy[0]))
    if not candidates:
        return None
    selected_epoch, selected_kind, selected_identity = max(
        candidates,
        key=lambda item: (item[0], item[1] == "native"),
    )
    history = await load_finalized_allocation_history_v2(
        netuid=int(netuid),
        start_epoch=selected_epoch,
        end_epoch=selected_epoch,
    )
    matches = []
    for authority in history:
        allocation_doc = authority.get("allocation_doc")
        authority_types = set(authority.get("authority_types") or ())
        if (
            int(authority.get("epoch", -1)) != selected_epoch
            or int(authority.get("netuid", -1)) != int(netuid)
            or not isinstance(allocation_doc, Mapping)
            or not allocation_doc.get("reimbursement_allocations")
            or allocation_doc.get(
                "historical_compute_fallback_source_epoch"
            )
            is not None
            or (
                selected_kind == "native"
                and "native_v2_finalization" not in authority_types
            )
            or (
                selected_kind == "legacy"
                and "legacy_finalized_chain_migration_v2"
                not in authority_types
            )
        ):
            continue
        if selected_kind == "legacy" and (
            str(authority.get("allocation_hash") or "")
            != str(selected_identity.get("allocation_hash") or "")
            or allocation_doc != selected_identity.get("allocation_doc")
        ):
            continue
        matches.append(dict(authority))
    if len(matches) != 1:
        raise ValueError(
            "historical compute fallback lacks finalized allocation authority"
        )
    authority = matches[0]
    return (
        {
            "epoch": int(authority["epoch"]),
            "netuid": int(authority["netuid"]),
            "allocation_hash": str(authority["allocation_hash"]),
            "allocation_doc": dict(authority["allocation_doc"]),
        },
        authority,
    )


def _historical_compute_fallback_from_snapshot(
    row: Mapping[str, Any],
    *,
    hotkey_uids: Mapping[str, int],
    reward_epochs: int,
    expected_netuid: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate one finalized compute-active allocation by current hotkey."""

    allocation_doc = row.get("allocation_doc")
    allocation_hash = str(row.get("allocation_hash") or "")
    if not isinstance(allocation_doc, Mapping):
        raise ValueError("historical compute allocation document is invalid")
    expected_hash = sha256_json(
        {
            key: value
            for key, value in allocation_doc.items()
            if key != "allocation_hash"
        }
    )
    try:
        source_epoch = int(row["epoch"])
        allocation_epoch = int(allocation_doc["epoch"])
        row_netuid = int(row.get("netuid", expected_netuid))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("historical compute allocation scope is invalid") from exc
    if (
        allocation_hash != expected_hash
        or allocation_doc.get("allocation_hash") != allocation_hash
        or allocation_epoch != source_epoch
        or (
            expected_netuid is not None
            and row_netuid != int(expected_netuid)
        )
        or allocation_doc.get("historical_compute_fallback_source_epoch")
        is not None
    ):
        raise ValueError("historical compute allocation authority differs")
    allocations = allocation_doc.get("reimbursement_allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("historical compute allocation has no reimbursements")

    grouped: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in allocations:
        if not isinstance(item, Mapping):
            raise ValueError("historical compute reimbursement row is invalid")
        source_id = str(item.get("source_id") or "")
        if (
            not source_id.startswith("reimbursement_schedule:")
            or item.get("reason") == "historical_compute_fallback_no_burn"
            or source_id in seen_sources
        ):
            raise ValueError(
                "historical compute reimbursement source is invalid or duplicated"
            )
        seen_sources.add(source_id)
        hotkey = str(item.get("miner_hotkey") or "")
        try:
            compute_microusd = int(
                item.get("eligible_compute_microusd")
                or item.get("spend_microusd")
                or 0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "historical compute reimbursement amount is invalid"
            ) from exc
        if compute_microusd <= 0 or not hotkey:
            raise ValueError(
                "historical compute reimbursement amount or hotkey is invalid"
            )
        uid = hotkey_uids.get(hotkey)
        if uid is None:
            skipped.append(
                {
                    "source_id": source_id,
                    "miner_hotkey": hotkey,
                    "reason": "miner_hotkey_not_registered",
                }
            )
            continue
        group = grouped.setdefault(
            hotkey,
            {
                "uid": int(uid),
                "miner_hotkey": hotkey,
                "spend_microusd": 0,
                "contributions": [],
            },
        )
        if int(group["uid"]) != int(uid):
            raise ValueError("historical compute hotkey UID mapping differs")
        group["spend_microusd"] += compute_microusd
        group["contributions"].append(
            {
                "source_id": source_id,
                "compute_microusd": compute_microusd,
            }
        )

    window_end = source_epoch
    window_start = max(0, source_epoch - max(1, int(reward_epochs)) + 1)
    obligations: list[dict[str, Any]] = []
    for hotkey in sorted(grouped):
        group = grouped[hotkey]
        contributions = sorted(
            group["contributions"],
            key=lambda item: str(item["source_id"]),
        )
        contribution_hash = sha256_json(contributions)
        source_hash = sha256_json(
            {
                "schema_version": (
                    "leadpoet.historical_compute_fallback_source.v1"
                ),
                "source_allocation_hash": allocation_hash,
                "miner_hotkey": hotkey,
                "contribution_hash": contribution_hash,
            }
        )
        obligations.append(
            {
                "uid": int(group["uid"]),
                "miner_uid": int(group["uid"]),
                "miner_hotkey": hotkey,
                "source_id": (
                    "historical_compute_fallback:"
                    + source_hash.split(":", 1)[1]
                ),
                "island": "historical_compute",
                "status": "active",
                "target_reimbursement_microusd": int(
                    group["spend_microusd"]
                ),
                "fallback_window_start_epoch": window_start,
                "fallback_window_end_epoch": window_end,
                "source_allocation_epoch": source_epoch,
                "source_allocation_hash": allocation_hash,
                "contribution_count": len(contributions),
                "contribution_hash": contribution_hash,
            }
        )
    source = {
        "schema_version": "leadpoet.historical_compute_fallback_authority.v1",
        "source_allocation_epoch": source_epoch,
        "source_allocation_hash": allocation_hash,
        "window_start_epoch": window_start,
        "window_end_epoch": window_end,
        "source_reimbursement_count": len(allocations),
        "eligible_miner_count": len(obligations),
    }
    return obligations, skipped, source


async def _active_champion_obligations(
    epoch: int,
    *,
    netuid: int,
    enable_champ_cap: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    champion_rows: list[dict[str, Any]] = []
    accepted_statuses = (
        ACTIVE_CHAMPION_STATUSES
        if enable_champ_cap
        else SETTLEMENT_TRACKED_CHAMPION_STATUSES
    )
    for status in sorted(accepted_statuses):
        champion_rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status), ("start_epoch", "lte", int(epoch))),
            )
        )
    paid_by_reward = await _champion_paid_alpha_to_date(epoch=int(epoch), netuid=int(netuid), champion_rows=champion_rows)
    hotkey_uids = await resolve_hotkey_uids(str(row.get("miner_hotkey") or "") for row in champion_rows)
    obligations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in champion_rows:
        status = str(row.get("current_reward_status") or row.get("reward_status") or "")
        if status not in accepted_statuses:
            continue
        miner_hotkey = str(row.get("miner_hotkey") or "")
        uid = hotkey_uids.get(miner_hotkey)
        if uid is None:
            skipped.append({"champion_reward_id": str(row.get("champion_reward_id") or ""), "reason": "miner_hotkey_not_registered"})
            continue
        replay_obligation = _champion_replay_obligation(
            row,
            paid_by_reward=paid_by_reward,
            epoch=int(epoch),
            enable_champ_cap=bool(enable_champ_cap),
        )
        if replay_obligation is None:
            continue
        obligations.append(
            {
                "uid": uid,
                "miner_uid": uid,
                "miner_hotkey": miner_hotkey,
                "source_id": str(row.get("champion_reward_id") or ""),
                "champion_reward_id": str(row.get("champion_reward_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "score_bundle_id": str(row.get("score_bundle_id") or ""),
                "run_id": str(row.get("run_id") or ""),
                "island": str(row.get("island") or "generalist"),
                "status": "active",
                "reward_kind": str(row.get("reward_kind") or "champion"),
                **replay_obligation,
            }
        )
    return obligations, skipped


async def _active_source_add_reward_rows(epoch: int) -> list[dict[str, Any]]:
    """Load active SOURCE_ADD rows without coupling them to champion rails."""

    rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
        try:
            source_rows = await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status), ("start_epoch", "lte", int(epoch))),
            )
        except Exception as exc:
            logger.warning(
                "research_lab_source_add_allocation_rows_unavailable epoch=%s error=%s",
                int(epoch),
                str(exc)[:300],
            )
            return []
        rows.extend(dict(row) for row in source_rows)
    return rows


async def _active_source_add_obligations(
    epoch: int,
    *,
    netuid: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = await _active_source_add_reward_rows(int(epoch))
    paid_by_reward = await _source_add_paid_alpha_to_date(
        epoch=int(epoch),
        netuid=int(netuid),
        source_rows=source_rows,
    )
    hotkey_uids = await resolve_hotkey_uids(str(row.get("miner_hotkey") or "") for row in source_rows)
    obligations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in source_rows:
        status = str(row.get("current_reward_status") or "")
        if status not in ACTIVE_CHAMPION_STATUSES:
            continue
        reward_ref = str(row.get("reward_ref") or "")
        miner_hotkey = str(row.get("miner_hotkey") or "")
        uid = hotkey_uids.get(miner_hotkey)
        if uid is None:
            skipped.append({"source_add_reward_id": reward_ref, "reason": "miner_hotkey_not_registered"})
            continue
        replay_obligation = _champion_replay_obligation(
            {
                "champion_reward_id": reward_ref,
                "start_epoch": int(row.get("start_epoch") or 0),
                "epoch_count": int(row.get("epoch_count") or row.get("reward_epochs") or 0),
                "desired_alpha_percent": float(
                    row.get("desired_alpha_percent") or row.get("alpha_percent") or 0.0
                ),
            },
            paid_by_reward=paid_by_reward,
            epoch=int(epoch),
        )
        if replay_obligation is None:
            continue
        obligations.append(
            {
                "uid": uid,
                "miner_uid": uid,
                "miner_hotkey": miner_hotkey,
                "source_id": reward_ref,
                "source_add_reward_id": reward_ref,
                "adapter_id": str(row.get("adapter_id") or ""),
                "leg": int(row.get("leg") or 0),
                "reward_kind": str(row.get("reward_kind") or ""),
                "created_at": str(row.get("created_at") or ""),
                "status": "active",
                **replay_obligation,
            }
        )
    return obligations, skipped


async def _champion_paid_alpha_to_date(
    *,
    epoch: int,
    netuid: int,
    champion_rows: list[dict[str, Any]],
) -> dict[str, float]:
    if not champion_rows:
        return {}
    start_epochs = [int(row.get("start_epoch") or 0) for row in champion_rows if int(row.get("start_epoch") or 0) <= int(epoch)]
    if not start_epochs:
        return {}
    start_floor = min(start_epochs)
    snapshot_rows = await select_all(
        "research_lab_emission_allocation_current",
        columns="epoch,allocation_doc",
        filters=(
            ("netuid", int(netuid)),
            ("epoch", "gte", int(start_floor)),
            ("epoch", "lt", int(epoch)),
        ),
        order_by=(("epoch", False),),
        max_rows=max(10000, int(epoch) - int(start_floor) + 100),
        allow_partial=True,
    )
    return _champion_paid_alpha_to_date_from_snapshots(
        snapshot_rows,
        obligation_caps=_champion_obligation_caps(champion_rows),
    )


async def _champion_finalized_paid_alpha_to_date(
    *,
    epoch: int,
    netuid: int,
    champion_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Return champion credit proven by finalized V2 chain evidence only."""

    if not champion_rows:
        return {}
    start_epochs = [
        int(row.get("start_epoch") or 0)
        for row in champion_rows
        if int(row.get("start_epoch") or 0) <= int(epoch)
    ]
    if not start_epochs:
        return {}
    start_floor = min(start_epochs)
    from gateway.research_lab.champion_settlement_v2 import (
        load_settled_allocation_history_v2,
    )

    finalized_rows = await load_settled_allocation_history_v2(
        netuid=int(netuid),
        start_epoch=int(start_floor),
        end_epoch=int(epoch) - 1,
    )
    return _champion_paid_alpha_to_date_from_snapshots(
        finalized_rows,
        obligation_caps=_champion_obligation_caps(champion_rows),
    )


async def _source_add_paid_alpha_to_date(
    *,
    epoch: int,
    netuid: int,
    source_rows: list[dict[str, Any]],
) -> dict[str, float]:
    if not source_rows:
        return {}
    start_epochs = [
        int(row.get("start_epoch") or 0)
        for row in source_rows
        if int(row.get("start_epoch") or 0) <= int(epoch)
    ]
    if not start_epochs:
        return {}
    start_floor = min(start_epochs)
    from gateway.research_lab.champion_settlement_v2 import (
        load_settled_allocation_history_v2,
    )

    snapshot_rows = await load_settled_allocation_history_v2(
        netuid=int(netuid),
        start_epoch=int(start_floor),
        end_epoch=int(epoch) - 1,
    )
    return _source_add_paid_alpha_to_date_from_snapshots(snapshot_rows)


def _source_add_paid_alpha_to_date_from_snapshots(
    snapshot_rows: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Count only first-class SOURCE_ADD allocation rows as settled."""

    paid_by_reward: dict[str, Decimal] = {}
    for row in snapshot_rows:
        allocation_doc = row.get("allocation_doc") or {}
        if not isinstance(allocation_doc, Mapping):
            continue
        allocations = allocation_doc.get("source_add_allocations") or []
        if not isinstance(allocations, list):
            continue
        for allocation in allocations:
            if not isinstance(allocation, Mapping):
                continue
            source_id = str(
                allocation.get("source_add_reward_id")
                or allocation.get("source_id")
                or ""
            )
            if not source_id.startswith("source_add_reward:"):
                continue
            paid = _decimal(allocation.get("paid_alpha_percent") or 0)
            # The exact chain-realized bundle remains payment authority. A
            # sub-u16 rounding deficit fulfills this scheduled epoch; any
            # larger shortfall remains due and enters normal replay.
            if allocation_doc.get("source") == "chain_realized_obligation_credits":
                if allocation_doc.get("authority_type") != "chain_realized_emission_v1":
                    raise ValueError(
                        "SOURCE_ADD chain settlement authority is invalid"
                    )
                scheduled = _decimal(
                    allocation.get("base_desired_alpha_percent") or 0
                )
                attributed = _decimal(
                    allocation.get("lab_attributed_alpha_percent") or 0
                )
                observed = _decimal(
                    allocation.get("observed_chain_alpha_percent") or 0
                )
                if (
                    not all(
                        amount.is_finite()
                        for amount in (paid, scheduled, attributed, observed)
                    )
                    or paid <= 0
                    or scheduled <= 0
                    or paid != min(attributed, scheduled)
                    or attributed > observed
                ):
                    raise ValueError(
                        "SOURCE_ADD chain settlement credit is invalid"
                    )
                if (
                    paid <= scheduled
                    and scheduled - paid
                    <= _SOURCE_ADD_CHAIN_QUANTIZATION_TOLERANCE_PERCENT
                ):
                    paid = scheduled
            paid_by_reward[source_id] = paid_by_reward.get(
                source_id, Decimal("0")
            ) + paid
    return {reward_id: _rate_float(paid) for reward_id, paid in paid_by_reward.items()}


def _champion_schedule_cap_start_epoch() -> int:
    """First epoch where surplus stops retiring the scheduled obligation.

    Epochs before the cutoff credit their full paid amount (legacy
    accounting): historical single-champion eras paid far above schedule and
    everyone treated those rewards as settled — recapping them retroactively
    revived long-finished champions, and their reopened shortfalls crowded
    the current champions out of the epoch pool. The default is the migration
    boundary after which the surplus-as-bonus accounting applies.
    """
    try:
        return int(os.getenv("RESEARCH_LAB_CHAMPION_SCHEDULE_CAP_START_EPOCH", "24100"))
    except ValueError:
        return 24100


def _champion_obligation_caps(
    champion_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Decimal]:
    caps: dict[str, Decimal] = {}
    for row in champion_rows:
        reward_id = str(
            row.get("champion_reward_id")
            or row.get("source_add_reward_id")
            or row.get("source_id")
            or ""
        )
        if not reward_id:
            continue
        desired = max(Decimal("0"), _decimal(row.get("desired_alpha_percent") or 0))
        try:
            epoch_count = max(0, int(row.get("epoch_count") or 0))
        except (TypeError, ValueError):
            epoch_count = 0
        caps[reward_id] = desired * Decimal(epoch_count)
    return caps


def _champion_paid_alpha_to_date_from_snapshots(
    snapshot_rows: list[Mapping[str, Any]],
    *,
    obligation_caps: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    ledger = _champion_lifetime_credit_ledger_from_snapshots(
        snapshot_rows,
        obligation_caps=obligation_caps,
    )
    for reward_id, excess in ledger["excess_by_reward"].items():
        if excess > 0:
            logger.warning(
                "champion_lifetime_excess reward_id=%s "
                "realized_excess_alpha_percent=%.6f",
                reward_id,
                excess,
            )
    return dict(ledger["applied_by_reward"])


def _champion_lifetime_credit_ledger_from_snapshots(
    snapshot_rows: list[Mapping[str, Any]],
    *,
    obligation_caps: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Return deterministic applied, raw-realized, and excess champion credit.

    Historical unmarked snapshots retain the existing scheduled-credit rule.
    Marked snapshots apply actual chain-realized champion attribution to the
    lifetime entitlement. Raw realized credit beyond that entitlement is
    retained as excess evidence but never reopens or overdraws the obligation.
    """
    cap_start_epoch = _champion_schedule_cap_start_epoch()
    applied_by_reward: dict[str, Decimal] = {}
    realized_by_reward: dict[str, Decimal] = {}
    excess_by_reward: dict[str, Decimal] = {}
    normalized_caps = {
        str(reward_id): max(Decimal("0"), _decimal(value))
        for reward_id, value in (obligation_caps or {}).items()
    }
    seen_lifetime_policy_epochs: set[int] = set()
    for row in sorted(snapshot_rows, key=lambda item: int(item.get("epoch") or 0)):
        allocation_doc = row.get("allocation_doc") or {}
        if not isinstance(allocation_doc, Mapping):
            continue
        policy_marker = allocation_doc.get("champion_credit_policy")
        if policy_marker not in (
            None,
            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
        ):
            raise ValueError("champion credit policy marker is invalid")
        lifetime_policy = (
            policy_marker
            == CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        )
        try:
            row_epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            if lifetime_policy:
                raise ValueError(
                    "champion lifetime credit epoch is invalid"
                ) from exc
            row_epoch = 0
        if lifetime_policy:
            try:
                allocation_epoch = int(allocation_doc["epoch"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "champion lifetime credit allocation epoch is invalid"
                ) from exc
            if allocation_epoch != row_epoch:
                raise ValueError(
                    "champion lifetime credit allocation epoch differs"
                )
            if row_epoch < 0 or row_epoch in seen_lifetime_policy_epochs:
                raise ValueError(
                    "champion lifetime credit epoch is duplicated or invalid"
                )
            seen_lifetime_policy_epochs.add(row_epoch)
        cap_applies = row_epoch >= cap_start_epoch
        seen_marked_rewards: set[str] = set()
        for section in ("champion_allocations", "queued_champion_allocations"):
            allocations = allocation_doc.get(section)
            if allocations is None and not lifetime_policy:
                allocations = []
            if not isinstance(allocations, list):
                if lifetime_policy:
                    raise ValueError(
                        "champion lifetime credit allocation section is invalid"
                    )
                continue
            for allocation in allocations:
                if not isinstance(allocation, Mapping):
                    if lifetime_policy:
                        raise ValueError(
                            "champion lifetime credit allocation is invalid"
                        )
                    continue
                source_id = str(allocation.get("source_id") or allocation.get("champion_reward_id") or "")
                if not source_id:
                    if lifetime_policy:
                        raise ValueError(
                            "champion lifetime credit source is invalid"
                        )
                    continue
                item_policy = allocation.get("champion_credit_policy")
                if item_policy not in (None, policy_marker):
                    raise ValueError(
                        "champion credit policy evidence is mixed"
                    )
                if lifetime_policy:
                    if source_id in seen_marked_rewards:
                        raise ValueError(
                            "champion lifetime credit is duplicated within epoch"
                        )
                    seen_marked_rewards.add(source_id)
                try:
                    paid = _decimal(
                        allocation.get("paid_alpha_percent") or 0
                    )
                except Exception as exc:
                    if lifetime_policy:
                        raise ValueError(
                            "champion lifetime credit amount is invalid"
                        ) from exc
                    raise
                if lifetime_policy and (not paid.is_finite() or paid < 0):
                    raise ValueError(
                        "champion lifetime credit amount is invalid"
                    )
                credit = paid
                if not lifetime_policy and cap_applies:
                    scheduled_raw = (
                        allocation.get("base_desired_alpha_percent")
                        if allocation.get("base_desired_alpha_percent") is not None
                        else allocation.get("intended_alpha_percent")
                    )
                    if scheduled_raw is not None:
                        scheduled = _decimal(scheduled_raw)
                        if scheduled > 0:
                            credit = min(paid, scheduled)
                realized_by_reward[source_id] = (
                    realized_by_reward.get(source_id, Decimal("0")) + credit
                )
                already_credited = applied_by_reward.get(
                    source_id,
                    Decimal("0"),
                )
                total_cap = normalized_caps.get(source_id)
                applied_credit = credit
                if total_cap is not None:
                    applied_credit = min(
                        credit,
                        max(Decimal("0"), total_cap - already_credited),
                    )
                applied_by_reward[source_id] = (
                    already_credited + applied_credit
                )
                excess_by_reward[source_id] = (
                    excess_by_reward.get(source_id, Decimal("0"))
                    + max(Decimal("0"), credit - applied_credit)
                )
    return {
        "applied_by_reward": {
            reward_id: _rate_float(value)
            for reward_id, value in applied_by_reward.items()
        },
        "realized_by_reward": {
            reward_id: _rate_float(value)
            for reward_id, value in realized_by_reward.items()
        },
        "excess_by_reward": {
            reward_id: _rate_float(value)
            for reward_id, value in excess_by_reward.items()
        },
    }


def _champion_replay_obligation(
    row: Mapping[str, Any],
    *,
    paid_by_reward: Mapping[str, float],
    epoch: int,
    enable_champ_cap: bool = True,
) -> dict[str, Any] | None:
    start_epoch = int(row.get("start_epoch") or 0)
    epoch_count = int(row.get("epoch_count") or 0)
    if epoch_count <= 0 or int(epoch) < start_epoch:
        return None
    champion_reward_id = str(row.get("champion_reward_id") or "")
    desired = _decimal(row.get("desired_alpha_percent") or 0)
    total_due = desired * Decimal(epoch_count)
    paid_to_date = min(total_due, _decimal(paid_by_reward.get(champion_reward_id, 0)))
    remaining = max(Decimal("0"), total_due - paid_to_date)
    nominal_end_epoch = start_epoch + epoch_count
    nominal_window_active = int(epoch) < nominal_end_epoch
    if (
        desired <= 0
        or (
            remaining <= 0
            and (bool(enable_champ_cap) or not nominal_window_active)
        )
    ):
        return None
    current_desired = (
        min(desired, remaining)
        if bool(enable_champ_cap)
        else desired
    )
    return {
        "start_epoch": start_epoch,
        "epoch_count": epoch_count,
        "nominal_end_epoch": nominal_end_epoch,
        "improvement_points": float(row.get("improvement_points") or 0.0),
        "threshold_points": float(row.get("threshold_points") or 0.0),
        "desired_alpha_percent": _rate_float(desired),
        "total_due_alpha_percent": _rate_float(total_due),
        "paid_alpha_percent_to_date": _rate_float(paid_to_date),
        "remaining_alpha_percent": _rate_float(remaining),
        "current_epoch_desired_alpha_percent": _rate_float(current_desired),
        "champ_cap_enabled": bool(enable_champ_cap),
        "replay_status": "extended_replay" if int(epoch) >= nominal_end_epoch else "nominal_window",
    }


def champion_reward_requires_allocation_history_v2(
    row: Mapping[str, Any],
    *,
    epoch: int,
    enable_champ_cap: bool,
) -> bool:
    """Return whether one reward can still affect the requested allocation."""

    status = str(
        row.get("current_reward_status") or row.get("reward_status") or ""
    )
    if status != "paid" or bool(enable_champ_cap):
        return True
    try:
        start_epoch = int(row.get("start_epoch") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("champion reward epoch fields are invalid") from exc
    if start_epoch < 0 or epoch_count <= 0:
        raise ValueError("champion reward epoch fields are invalid")
    return int(epoch) < start_epoch + epoch_count


def allocation_snapshot_persistence_decision(
    *,
    current_epoch: int,
    requested_epoch: int,
    provided_key: str | None,
    configured_key: str,
    live_allocation_enabled: bool,
) -> str:
    """Decide how an allocation GET may behave for one request.

    Returns one of:
      - "future_epoch": reject — snapshots must never exist ahead of time
        (anonymous GETs once pre-created active rows four epochs ahead,
        contaminating paid-to-date accounting).
      - "read_only": compute without persisting (anonymous callers).
      - "key_not_configured" / "invalid_key": authentication failures.
      - "persist": authenticated validator persisting the current epoch.
      - "authenticated_read_only": valid key but a past epoch — recomputing
        history with today's obligations must not overwrite the record.
    """
    if int(requested_epoch) > int(current_epoch):
        return "future_epoch"
    if provided_key is None:
        return "read_only"
    if not configured_key:
        return "key_not_configured"
    if not hmac.compare_digest(str(provided_key), str(configured_key)):
        return "invalid_key"
    if live_allocation_enabled and int(requested_epoch) == int(current_epoch):
        return "persist"
    return "authenticated_read_only"


def _epoch_active(row: Mapping[str, Any], epoch: int) -> bool:
    try:
        start_epoch = int(row.get("start_epoch") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
    except (TypeError, ValueError):
        return False
    return epoch_count > 0 and start_epoch <= int(epoch) < start_epoch + epoch_count


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _rate_float(value: Decimal) -> float:
    return float(value.quantize(RATE_QUANT, rounding=ROUND_HALF_UP))


def _bittensor_network() -> str:
    return (os.getenv("BITTENSOR_NETWORK") or os.getenv("SUBTENSOR_NETWORK") or "finney").strip() or "finney"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
