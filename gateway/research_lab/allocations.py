"""Research Lab live allocation projection for validator consumption."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from gateway.research_lab.bundles import contains_secret_material, sha256_json
from gateway.research_lab.chain import resolve_hotkey_uids
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import create_research_lab_emission_allocation_snapshot, select_all, select_one
from leadpoet_verifier.economics import allocate_research_lab_epoch


ACTIVE_REIMBURSEMENT_STATUSES = {"awarded"}
ACTIVE_SCHEDULE_STATUSES = {"scheduled"}
ACTIVE_CHAMPION_STATUSES = {"active", "queued", "partially_paid"}


async def build_research_lab_allocation_bundle(
    *,
    config: ResearchLabGatewayConfig,
    epoch: int,
    netuid: int,
    persist_snapshot: bool = False,
) -> dict[str, Any]:
    """Build a sanitized Research Lab allocation bundle for one epoch."""
    policy = config.reimbursement_policy_doc(enabled=True)
    reimbursement_obligations, reimbursement_skipped = await _active_reimbursement_obligations(int(epoch), policy=policy)
    champion_obligations, champion_skipped = await _active_champion_obligations(int(epoch))
    allocation = allocate_research_lab_epoch(
        int(epoch),
        policy,
        reimbursement_obligations,
        champion_obligations,
    )
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
    if contains_secret_material(source_state) or contains_secret_material(allocation):
        raise ValueError("Research Lab allocation bundle contains private or secret material")
    source_state_hash = sha256_json(source_state)
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
    awards_by_id: dict[str, dict[str, Any]] = {}
    for schedule in active_schedule_rows:
        award_id = str(schedule.get("award_id") or "")
        if not award_id or award_id in awards_by_id:
            continue
        award = await select_one(
            "research_reimbursement_award_current",
            filters=(
                ("award_id", award_id),
                ("current_award_status", "awarded"),
            ),
        )
        if award and str(award.get("current_award_status") or award.get("award_status") or "") in ACTIVE_REIMBURSEMENT_STATUSES:
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


async def _active_champion_obligations(epoch: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    champion_rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
        champion_rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
            )
        )
    hotkey_uids = await resolve_hotkey_uids(str(row.get("miner_hotkey") or "") for row in champion_rows)
    obligations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in champion_rows:
        status = str(row.get("current_reward_status") or row.get("reward_status") or "")
        if status not in ACTIVE_CHAMPION_STATUSES:
            continue
        if not _epoch_active(row, epoch):
            continue
        miner_hotkey = str(row.get("miner_hotkey") or "")
        uid = hotkey_uids.get(miner_hotkey)
        if uid is None:
            skipped.append({"champion_reward_id": str(row.get("champion_reward_id") or ""), "reason": "miner_hotkey_not_registered"})
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
                "start_epoch": int(row.get("start_epoch") or 0),
                "epoch_count": int(row.get("epoch_count") or 0),
                "improvement_points": float(row.get("improvement_points") or 0.0),
                "threshold_points": float(row.get("threshold_points") or 0.0),
                "desired_alpha_percent": float(row.get("desired_alpha_percent") or 0.0),
            }
        )
    return obligations, skipped


def _epoch_active(row: Mapping[str, Any], epoch: int) -> bool:
    try:
        start_epoch = int(row.get("start_epoch") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
    except (TypeError, ValueError):
        return False
    return epoch_count > 0 and start_epoch <= int(epoch) < start_epoch + epoch_count


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
