"""Research Lab live allocation projection for validator consumption."""

from __future__ import annotations

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
from gateway.research_lab.store import create_research_lab_emission_allocation_snapshot, select_all
from leadpoet_verifier.economics import allocate_research_lab_epoch


ACTIVE_REIMBURSEMENT_STATUSES = {"awarded"}
ACTIVE_SCHEDULE_STATUSES = {"scheduled"}
ACTIVE_CHAMPION_STATUSES = {"active", "queued", "partially_paid"}
SETTLEMENT_TRACKED_CHAMPION_STATUSES = ACTIVE_CHAMPION_STATUSES | {"paid"}
RATE_QUANT = Decimal("0.000001")
POSTGREST_IN_FILTER_CHUNK = 50
logger = logging.getLogger(__name__)


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
            await _active_champion_obligations(int(epoch), netuid=int(netuid))
        )
        source_add_obligations, source_add_skipped = (
            await _active_source_add_obligations(int(epoch), netuid=int(netuid))
        )
        source_add_present = bool(source_add_obligations or source_add_skipped)
        allocation_inputs = {
            "epoch": int(epoch),
            "policy": policy,
            "active_reimbursement_obligations": reimbursement_obligations,
            "active_champion_obligations": champion_obligations,
        }
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
        if source_add_present:
            source_state["source_add_obligation_count"] = len(source_add_obligations)
            source_state["source_add_obligations"] = source_add_obligations
            source_state["skipped"]["source_add"] = source_add_skipped
        source_state_hash = sha256_json(source_state)
        attestation = {"status": "off", "protocol": "legacy_v1"}
    else:
        attestation = await build_allocation_v2(
            epoch_id=int(epoch),
            netuid=int(netuid),
            policy=config.reimbursement_policy_doc(enabled=True),
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
                "participation_score": float(award.get("participation_score") or 0.0),
            }
        )
    return obligations, skipped


async def _active_champion_obligations(epoch: int, *, netuid: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    champion_rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
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
        if status not in ACTIVE_CHAMPION_STATUSES:
            continue
        miner_hotkey = str(row.get("miner_hotkey") or "")
        uid = hotkey_uids.get(miner_hotkey)
        if uid is None:
            skipped.append({"champion_reward_id": str(row.get("champion_reward_id") or ""), "reason": "miner_hotkey_not_registered"})
            continue
        replay_obligation = _champion_replay_obligation(row, paid_by_reward=paid_by_reward, epoch=int(epoch))
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
        load_finalized_allocation_history_v2,
    )

    finalized_rows = await load_finalized_allocation_history_v2(
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
            paid_by_reward[source_id] = paid_by_reward.get(source_id, Decimal("0")) + _decimal(
                allocation.get("paid_alpha_percent") or 0
            )
    return {reward_id: _rate_float(paid) for reward_id, paid in paid_by_reward.items()}


def _champion_schedule_cap_start_epoch() -> int:
    """First epoch where surplus stops retiring the scheduled obligation.

    Epochs before the cutoff credit their full paid amount (legacy
    accounting): historical single-champion eras paid far above schedule and
    everyone treated those rewards as settled — recapping them retroactively
    revived long-finished champions, and their reopened shortfalls crowded
    the current champions out of the epoch pool. The default is the start
    epoch of the first reward the surplus-as-bonus policy was written for.
    """
    try:
        return int(os.getenv("RESEARCH_LAB_CHAMPION_SCHEDULE_CAP_START_EPOCH", "23878"))
    except ValueError:
        return 23878


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
    """Sum per-epoch obligation credit from pre-validated settlement rows.

    Champions earn the scheduled rate for the full reward window; a
    single-champion epoch can pay far above schedule from the remaining Lab
    pool, and that surplus is a bonus — it must not retire the scheduled
    obligation early (a sole champion once hit its 20-epoch lifetime target
    in 8 epochs and lost the remaining 12). From the policy cutoff epoch
    onward, each epoch credits at most the entry's scheduled rate against
    the obligation; earlier epochs and entries without a recorded schedule
    credit their full paid amount.
    """
    cap_start_epoch = _champion_schedule_cap_start_epoch()
    paid_by_reward: dict[str, Decimal] = {}
    normalized_caps = {
        str(reward_id): max(Decimal("0"), _decimal(value))
        for reward_id, value in (obligation_caps or {}).items()
    }
    for row in sorted(snapshot_rows, key=lambda item: int(item.get("epoch") or 0)):
        allocation_doc = row.get("allocation_doc") or {}
        if not isinstance(allocation_doc, Mapping):
            continue
        try:
            row_epoch = int(row.get("epoch") or 0)
        except (TypeError, ValueError):
            row_epoch = 0
        cap_applies = row_epoch >= cap_start_epoch
        for section in ("champion_allocations", "queued_champion_allocations"):
            allocations = allocation_doc.get(section) or []
            if not isinstance(allocations, list):
                continue
            for allocation in allocations:
                if not isinstance(allocation, Mapping):
                    continue
                source_id = str(allocation.get("source_id") or allocation.get("champion_reward_id") or "")
                if not source_id:
                    continue
                paid = _decimal(allocation.get("paid_alpha_percent") or 0)
                credit = paid
                if cap_applies:
                    scheduled_raw = (
                        allocation.get("base_desired_alpha_percent")
                        if allocation.get("base_desired_alpha_percent") is not None
                        else allocation.get("intended_alpha_percent")
                    )
                    if scheduled_raw is not None:
                        scheduled = _decimal(scheduled_raw)
                        if scheduled > 0:
                            credit = min(paid, scheduled)
                already_credited = paid_by_reward.get(source_id, Decimal("0"))
                total_cap = normalized_caps.get(source_id)
                if total_cap is not None:
                    credit = min(credit, max(Decimal("0"), total_cap - already_credited))
                paid_by_reward[source_id] = already_credited + credit
    return {reward_id: _rate_float(paid) for reward_id, paid in paid_by_reward.items()}


def _champion_replay_obligation(
    row: Mapping[str, Any],
    *,
    paid_by_reward: Mapping[str, float],
    epoch: int,
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
    if desired <= 0 or remaining <= 0:
        return None
    nominal_end_epoch = start_epoch + epoch_count
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
        "current_epoch_desired_alpha_percent": _rate_float(min(desired, remaining)),
        "replay_status": "extended_replay" if int(epoch) >= nominal_end_epoch else "nominal_window",
    }


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
