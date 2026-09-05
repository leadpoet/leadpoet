"""Maintenance helpers for retained Research Lab reward settlement."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

from .config import ResearchLabGatewayConfig
from .store import select_all

logger = logging.getLogger(__name__)


def default_actor_ref() -> str:
    user = os.getenv("USER") or "operator"
    try:
        host = os.uname().nodename
    except AttributeError:
        host = "unknown-host"
    return f"{user}@{host}"


def dumps_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


async def reconcile_champion_reward_statuses(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 50,
    reason: str = "champion_reward_status_reconciler",
    actor_ref: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Mark champion rewards whose scheduled obligation is fully retired as paid.

    Paid-to-date is reconstructed only from allocation inputs whose exact
    validator weight extrinsic has finalized on chain.  Allocation snapshots
    that were merely produced or published cannot close an obligation.
    """
    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.allocations import (
        ACTIVE_CHAMPION_STATUSES,
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
        _champion_finalized_paid_alpha_to_date,
        _decimal,
        _rate_float,
    )
    from gateway.research_lab.config import ResearchLabGatewayConfig
    from .store import create_champion_reward_event

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    enable_champ_cap = bool(
        ResearchLabGatewayConfig.from_env().enable_champ_cap
    )
    reward_rows: list[dict[str, Any]] = []
    statuses = (
        ACTIVE_CHAMPION_STATUSES
        if enable_champ_cap
        else SETTLEMENT_TRACKED_CHAMPION_STATUSES
    )
    for status in sorted(statuses):
        reward_rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
                max_rows=max(1, int(limit or 50) * 5),
            )
        )
    paid_by_reward = await _champion_finalized_paid_alpha_to_date(
        epoch=effective_epoch,
        netuid=effective_netuid,
        champion_rows=reward_rows,
    )
    planned: list[dict[str, Any]] = []
    for row in reward_rows:
        reward_id = str(row.get("champion_reward_id") or "")
        if not reward_id:
            continue
        desired = _decimal(row.get("desired_alpha_percent") or 0)
        epoch_count = int(row.get("epoch_count") or 0)
        total_due = desired * epoch_count
        paid = min(total_due, _decimal(paid_by_reward.get(reward_id, 0)))
        remaining = total_due - paid
        start_epoch = int(row.get("start_epoch") or 0)
        nominal_end_epoch = start_epoch + epoch_count
        current_status = str(row.get("current_reward_status") or "")
        if desired <= 0 or epoch_count <= 0:
            continue
        if (
            not enable_champ_cap
            and current_status == "paid"
            and effective_epoch < nominal_end_epoch
        ):
            event_type = "active"
            status_target = "active"
        elif remaining <= 0 and (
            enable_champ_cap or effective_epoch >= nominal_end_epoch
        ):
            if current_status == "paid":
                continue
            event_type = "paid"
            status_target = "paid"
        else:
            continue
        planned.append(
            {
                "champion_reward_id": reward_id,
                "miner_uid": row.get("miner_uid"),
                "current_reward_status": current_status,
                "event_type": event_type,
                "reward_status_target": status_target,
                "nominal_end_epoch": nominal_end_epoch,
                "total_due_alpha_percent": _rate_float(total_due),
                "paid_alpha_percent_to_date": _rate_float(paid),
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "reconcile-champion-reward-statuses",
            "epoch": effective_epoch,
            "planned": planned,
        }

    repaired: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for plan in planned:
        try:
            event = await create_champion_reward_event(
                champion_reward_id=str(plan["champion_reward_id"]),
                event_type=str(plan["event_type"]),
                reward_status=str(plan["reward_status_target"]),
                reason=reason,
                event_doc={
                    "schema_version": "1.0",
                    "source": "finalized_chain_champion_status_reconciler",
                    "settlement_authority": "finalized_v2_weight_extrinsics",
                    "actor_ref": actor_ref or default_actor_ref(),
                    "epoch": effective_epoch,
                    "enable_champ_cap": enable_champ_cap,
                    "nominal_end_epoch": plan["nominal_end_epoch"],
                    "previous_reward_status": plan["current_reward_status"],
                    "total_due_alpha_percent": plan["total_due_alpha_percent"],
                    "paid_alpha_percent_to_date": plan["paid_alpha_percent_to_date"],
                },
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.warning(
                "research_lab_champion_status_reconcile_write_failed "
                "reward_id=%s error=%s",
                plan["champion_reward_id"],
                str(exc)[:240],
            )
            failed.append(
                {
                    "champion_reward_id": plan["champion_reward_id"],
                    "error": str(exc)[:240],
                }
            )
            continue
        repaired.append({**plan, "event_seq": event.get("seq"), "event_hash": event.get("anchored_hash")})
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "reconcile-champion-reward-statuses",
        "epoch": effective_epoch,
        "planned_count": len(planned),
        "repaired_count": len(repaired),
        "repaired": repaired,
        "failed": failed,
    }


async def reconcile_source_add_reward_statuses(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 50,
    reason: str = "source_add_reward_fully_delivered",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Stop forward payments on SOURCE_ADD rewards whose obligation is delivered.

    Paid-to-date is reconstructed from the per-epoch allocation snapshots —
    the same first-class SOURCE_ADD rows the allocator itself settles from.
    A reward whose snapshots already sum to its full promised alpha
    (``alpha_percent`` x ``reward_epochs``) gets a ``stopped_forward`` event
    appended, which removes it from the active-status set the allocator
    reads. This is the guard against the allocator's epoch counter freezing
    (as during the 2026-07-20 stateful-epoch switch) while weight setting
    keeps reusing the last computed payout pattern: without the stop event,
    a finished reward keeps collecting every epoch with nothing counting
    it down.
    """

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.allocations import (
        ACTIVE_CHAMPION_STATUSES,
        _decimal,
        _rate_float,
        _source_add_paid_alpha_to_date,
    )
    from .store import insert_row

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    reward_rows: list[dict[str, Any]] = []
    for status in sorted(ACTIVE_CHAMPION_STATUSES):
        reward_rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status),),
                max_rows=max(1, int(limit or 50) * 5),
            )
        )
    paid_by_reward = await _source_add_paid_alpha_to_date(
        epoch=effective_epoch,
        netuid=effective_netuid,
        source_rows=reward_rows,
    )
    planned: list[dict[str, Any]] = []
    planned_refs: set[str] = set()
    for row in reward_rows:
        reward_ref = str(row.get("reward_ref") or "")
        if not reward_ref or reward_ref in planned_refs:
            continue
        desired = _decimal(
            row.get("desired_alpha_percent") or row.get("alpha_percent") or 0
        )
        epoch_count = int(row.get("epoch_count") or row.get("reward_epochs") or 0)
        total_due = desired * epoch_count
        paid = _decimal(paid_by_reward.get(reward_ref, 0))
        if desired <= 0 or epoch_count <= 0 or paid < total_due:
            continue
        planned_refs.add(reward_ref)
        planned.append(
            {
                "reward_ref": reward_ref,
                "miner_hotkey": str(row.get("miner_hotkey") or ""),
                "current_reward_status": str(row.get("current_reward_status") or ""),
                "reward_status_target": "stopped_forward",
                "total_due_alpha_percent": _rate_float(total_due),
                "paid_alpha_percent_to_date": _rate_float(paid),
                "next_seq": int(row.get("current_event_seq") or 0) + 1,
            }
        )
        if len(planned) >= max(1, int(limit or 50)):
            break
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "reconcile-source-add-reward-statuses",
            "epoch": effective_epoch,
            "planned": planned,
        }

    stopped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for plan in planned:
        try:
            await insert_row(
                "research_lab_source_add_reward_events",
                {
                    "reward_ref": plan["reward_ref"],
                    "seq": plan["next_seq"],
                    "reward_status": "stopped_forward",
                    "reason": "%s paid=%.4f due=%.4f epoch=%s"
                    % (
                        reason,
                        plan["paid_alpha_percent_to_date"],
                        plan["total_due_alpha_percent"],
                        effective_epoch,
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.warning(
                "research_lab_source_add_status_reconcile_write_failed "
                "reward_ref=%s error=%s",
                plan["reward_ref"],
                str(exc)[:240],
            )
            failed.append(
                {"reward_ref": plan["reward_ref"], "error": str(exc)[:240]}
            )
            continue
        stopped.append(plan)
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "reconcile-source-add-reward-statuses",
        "epoch": effective_epoch,
        "planned_count": len(planned),
        "stopped_count": len(stopped),
        "stopped": stopped,
        "failed": failed,
    }


async def backfill_source_add_reward_v2_authority(
    *,
    epoch: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Idempotently attest measured pre-V2 SOURCE_ADD obligations."""

    from gateway.research_lab.allocations import (
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
    )
    from gateway.research_lab.v2_authority import (
        attest_historical_source_add_reward_v2,
    )
    from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
    from leadpoet_canonical.attested_v2 import sha256_json

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("created_at", False),),
                max_rows=max(1, int(limit)),
                allow_partial=False,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("start_epoch") or 0),
            str(row.get("reward_ref") or ""),
        ),
    )[: max(1, int(limit))]
    covered: list[str] = []
    planned: list[str] = []
    for row in rows:
        reward_ref = str(row.get("reward_ref") or "")
        expected_output = sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg%d" % int(row.get("leg") or 0),
                {**dict(row), "initial_reward_status": "active"},
            )
        )
        try:
            graph = await load_business_artifact_graph_by_ref_v2(
                artifact_kind="source_add_reward_decision",
                artifact_ref=reward_ref,
            )
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    receipt
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                ),
                None,
            )
            if (
                not isinstance(root, Mapping)
                or root.get("purpose") != "research_lab.reward_decision.v2"
                or root.get("output_root") != expected_output
            ):
                raise RuntimeError("stored SOURCE_ADD V2 receipt differs")
            covered.append(reward_ref)
        except Exception as exc:
            logger.info(
                "research_lab_source_add_v2_backfill_required "
                "reward_ref=%s reason=%s",
                reward_ref,
                str(exc)[:200],
            )
            planned.append(reward_ref)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "backfill-source-add-v2-authority",
            "epoch": effective_epoch,
            "inspected_count": len(rows),
            "already_covered_count": len(covered),
            "planned_count": len(planned),
            "planned_reward_refs": planned,
        }

    migrated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for reward_ref in planned:
        try:
            outcome = await attest_historical_source_add_reward_v2(
                epoch_id=effective_epoch,
                reward_ref=reward_ref,
            )
            receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
            migrated.append(
                {
                    "reward_ref": reward_ref,
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:
            logger.exception(
                "research_lab_source_add_v2_backfill_failed reward_ref=%s",
                reward_ref,
            )
            failed.append(
                {
                    "reward_ref": reward_ref,
                    "error": str(exc)[:300],
                }
            )
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "backfill-source-add-v2-authority",
        "epoch": effective_epoch,
        "inspected_count": len(rows),
        "already_covered_count": len(covered),
        "migrated_count": len(migrated),
        "migrated": migrated,
        "failed": failed,
    }


async def backfill_champion_reward_v2_authority(
    *,
    epoch: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Idempotently attest immutable pre-V2 champion obligations."""

    from gateway.research_lab.allocations import (
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
    )
    from gateway.research_lab.v2_authority import (
        attest_historical_champion_reward_v2,
    )
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2
    from leadpoet_canonical.attested_v2 import sha256_json

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("created_at", False),),
                max_rows=max(1, int(limit)),
                allow_partial=False,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("start_epoch") or 0),
            str(row.get("champion_reward_id") or ""),
        ),
    )[: max(1, int(limit))]
    covered: list[str] = []
    planned: list[str] = []
    for row in rows:
        reward_id = str(row.get("champion_reward_id") or "")
        expected_output = sha256_json(champion_reward_row_projection_v2(row))
        try:
            graph = await load_business_artifact_graph_by_ref_v2(
                artifact_kind="champion_reward_decision",
                artifact_ref=reward_id,
            )
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    receipt
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                ),
                None,
            )
            if (
                not isinstance(root, Mapping)
                or root.get("purpose") != "research_lab.reward_decision.v2"
                or root.get("output_root") != expected_output
            ):
                raise RuntimeError("stored champion V2 receipt differs")
            covered.append(reward_id)
        except Exception as exc:
            logger.info(
                "research_lab_champion_v2_backfill_required reward_id=%s reason=%s",
                reward_id,
                str(exc)[:200],
            )
            planned.append(reward_id)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "backfill-champion-v2-authority",
            "epoch": effective_epoch,
            "inspected_count": len(rows),
            "already_covered_count": len(covered),
            "planned_count": len(planned),
            "planned_champion_reward_ids": planned,
        }

    migrated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for reward_id in planned:
        try:
            outcome = await attest_historical_champion_reward_v2(
                epoch_id=effective_epoch,
                champion_reward_id=reward_id,
            )
            receipt = outcome.get("execution_receipt") or outcome.get("receipt") or {}
            migrated.append(
                {
                    "champion_reward_id": reward_id,
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:
            logger.exception(
                "research_lab_champion_v2_backfill_failed reward_id=%s",
                reward_id,
            )
            failed.append(
                {
                    "champion_reward_id": reward_id,
                    "error": str(exc)[:300],
                }
            )
    return {
        "ok": not failed,
        "dry_run": False,
        "action": "backfill-champion-v2-authority",
        "epoch": effective_epoch,
        "inspected_count": len(rows),
        "already_covered_count": len(covered),
        "migrated_count": len(migrated),
        "migrated": migrated,
        "failed": failed,
    }


async def backfill_historical_compute_fallback_v2_authority(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Classify the exact prior compute snapshot required by no-burn V2."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.allocations import (
        _historical_compute_fallback_from_snapshot,
        _load_latest_finalized_compute_snapshot_v2,
    )
    from gateway.research_lab.champion_settlement_v2 import (
        load_finalized_allocation_history_v2,
    )
    from gateway.research_lab.v2_authority import (
        classify_historical_champion_allocation_v2,
    )

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = (
        int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    )
    config = ResearchLabGatewayConfig.from_env()
    if bool(config.enable_conservative):
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": "conservative_mode_enabled",
            "classified_count": 0,
        }

    finalized = await _load_latest_finalized_compute_snapshot_v2(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    if finalized is not None:
        finalized_row, _authority = finalized
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": "already_classified",
            "source_allocation_epoch": int(finalized_row["epoch"]),
            "source_allocation_hash": str(
                finalized_row["allocation_hash"]
            ),
            "classified_count": 0,
        }

    rows = await select_all(
        "research_lab_emission_allocation_current",
        columns="epoch,netuid,allocation_hash,allocation_doc",
        filters=(
            ("netuid", effective_netuid),
            ("epoch", "lt", effective_epoch),
            ("allocation_doc->reimbursement_allocations", "neq", []),
            (
                "allocation_doc->>historical_compute_fallback_source_epoch",
                "is",
                "null",
            ),
        ),
        order_by=(("epoch", True),),
        batch_size=1,
        max_rows=1,
        allow_partial=True,
    )
    if not rows:
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": "no_prior_compute_snapshot",
            "classified_count": 0,
        }

    source_row = rows[0]
    allocation_doc = source_row.get("allocation_doc")
    snapshot_hotkey_uids: dict[str, int] = {}
    if isinstance(allocation_doc, Mapping):
        for item in allocation_doc.get("reimbursement_allocations") or ():
            if isinstance(item, Mapping) and item.get("miner_hotkey"):
                snapshot_hotkey_uids[str(item["miner_hotkey"])] = int(
                    item.get("uid") or 0
                )
    _, _, source = _historical_compute_fallback_from_snapshot(
        source_row,
        hotkey_uids=snapshot_hotkey_uids,
        reward_epochs=max(
            1,
            int(config.reimbursement_epochs or 20),
        ),
        expected_netuid=effective_netuid,
    )
    source_epoch = int(source["source_allocation_epoch"])
    source_hash = str(source["source_allocation_hash"])

    async def load_matching_authority() -> list[dict[str, Any]]:
        history = await load_finalized_allocation_history_v2(
            netuid=effective_netuid,
            start_epoch=source_epoch,
            end_epoch=source_epoch,
        )
        return [
            dict(item)
            for item in history
            if int(item.get("epoch") or -1) == source_epoch
            and int(item.get("netuid") or -1) == effective_netuid
            and str(item.get("allocation_hash") or "") == source_hash
            and item.get("allocation_doc") == allocation_doc
            and set(item.get("authority_types") or ()).intersection(
                {
                    "native_v2_finalization",
                    "legacy_finalized_chain_migration_v2",
                }
            )
        ]

    matching = await load_matching_authority()
    if len(matching) > 1:
        raise RuntimeError(
            "historical compute fallback finalized authority is ambiguous"
        )
    if matching:
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": "already_classified",
            "source_allocation_epoch": source_epoch,
            "source_allocation_hash": source_hash,
            "classified_count": 0,
        }
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": "classification_required",
            "source_allocation_epoch": source_epoch,
            "source_allocation_hash": source_hash,
            "classified_count": 0,
        }

    outcome = await classify_historical_champion_allocation_v2(
        epoch_id=effective_epoch,
        netuid=effective_netuid,
        settlement_epoch_id=source_epoch,
    )
    classification = str(outcome.get("status") or "")
    if classification != "finalized":
        return {
            "ok": False,
            "dry_run": False,
            "action": "backfill-historical-compute-fallback-v2-authority",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "status": classification or "classification_failed",
            "source_allocation_epoch": source_epoch,
            "source_allocation_hash": source_hash,
            "classified_count": 1,
        }
    matching = await load_matching_authority()
    if len(matching) != 1:
        raise RuntimeError(
            "historical compute fallback finalized authority readback differs"
        )
    return {
        "ok": True,
        "dry_run": False,
        "action": "backfill-historical-compute-fallback-v2-authority",
        "epoch": effective_epoch,
        "netuid": effective_netuid,
        "status": "finalized",
        "source_allocation_epoch": source_epoch,
        "source_allocation_hash": source_hash,
        "classified_count": 1,
    }


async def backfill_champion_settlement_v2_authority(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
    limit: int = 1000,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Classify missing pre-V2 champion allocation epochs.

    The cutover readiness report is the source of work. Invalid historical
    rows and allocation-hash conflicts remain blocking findings; they are
    never converted into payment authority by this command.
    """

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.v2_authority import (
        classify_historical_champion_allocation_v2,
    )
    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = (
        int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    )
    normalized_limit = max(1, int(limit or 1000))
    before = await champion_v2_cutover_readiness_report(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    missing = list(
        before.get("missing_historical_classifications")
        or before.get("missing_historical_settlements")
        or ()
    )
    planned = [
        {
            "epoch": int(item["epoch"]),
            "allocation_hash": str(item["allocation_hash"]),
        }
        for item in missing
        if isinstance(item, Mapping)
        and item.get("reason")
        == "missing_finalized_chain_classification_authority"
        and item.get("epoch") is not None
        and item.get("allocation_hash")
    ]
    planned = sorted(planned, key=lambda item: item["epoch"])[
        :normalized_limit
    ]
    blocked = [
        dict(item)
        for item in missing
        if not (
            isinstance(item, Mapping)
            and item.get("reason")
            == "missing_finalized_chain_classification_authority"
        )
    ]
    if dry_run:
        return {
            "ok": not blocked,
            "dry_run": True,
            "action": "backfill-champion-v2-settlements",
            "epoch": effective_epoch,
            "netuid": effective_netuid,
            "planned_count": len(planned),
            "planned": planned,
            "blocked": blocked,
            "readiness_before": before,
        }

    classified: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in planned:
        settlement_epoch = int(item["epoch"])
        try:
            outcome = await classify_historical_champion_allocation_v2(
                epoch_id=effective_epoch,
                netuid=effective_netuid,
                settlement_epoch_id=settlement_epoch,
            )
            receipt = (
                outcome.get("execution_receipt")
                or outcome.get("receipt")
                or {}
            )
            result = outcome.get("result") or {}
            classification = str(outcome.get("status") or "")
            classified.append(
                {
                    **item,
                    "classification": classification,
                    "settlement_hash": str(
                        result.get("settlement_hash") or ""
                    ),
                    "finding_hash": str(result.get("finding_hash") or ""),
                    "receipt_hash": str(receipt.get("receipt_hash") or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue the idempotent sweep
            logger.exception(
                "research_lab_champion_v2_settlement_backfill_failed "
                "netuid=%s settlement_epoch=%s",
                effective_netuid,
                settlement_epoch,
            )
            failed.append(
                {
                    **item,
                    "error": str(exc)[:300],
                }
            )
    after = await champion_v2_cutover_readiness_report(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    after_missing = list(
        after.get("missing_historical_classifications")
        or after.get("missing_historical_settlements")
        or ()
    )
    unresolved_identities = {
        (
            int(item["epoch"]),
            str(item.get("allocation_hash") or ""),
        )
        for item in after_missing
        if isinstance(item, Mapping) and item.get("epoch") is not None
    }
    unresolved_failed = [
        item
        for item in failed
        if (
            int(item["epoch"]),
            str(item.get("allocation_hash") or ""),
        )
        in unresolved_identities
    ]
    recovered_after_readback = [
        item
        for item in failed
        if (
            int(item["epoch"]),
            str(item.get("allocation_hash") or ""),
        )
        not in unresolved_identities
    ]
    classifications_complete = (
        not after_missing
        and float(
            after.get("historical_classification_coverage")
            or after.get("historical_settlement_coverage")
            or 0.0
        )
        == 1.0
    )
    return {
        # An enclave/database operation can commit durably before its response
        # is lost. The authoritative readback, not the transport outcome,
        # decides whether any classification remains missing.
        "ok": classifications_complete,
        "dry_run": False,
        "action": "backfill-champion-v2-settlements",
        "epoch": effective_epoch,
        "netuid": effective_netuid,
        "planned_count": len(planned),
        "classified_count": len(classified),
        "finalized_count": sum(
            1
            for item in classified
            if item.get("classification") == "finalized"
        ),
        "nonfinalized_count": sum(
            1
            for item in classified
            if item.get("classification") == "not_finalized"
        ),
        # Compatibility aliases for the existing operator command response.
        "migrated_count": len(classified),
        "migrated": classified,
        "blocked": [
            dict(item)
            for item in after_missing
            if not (
                isinstance(item, Mapping)
                and item.get("reason")
                == "missing_finalized_chain_classification_authority"
            )
        ],
        "failed": unresolved_failed,
        "recovered_after_readback_count": len(recovered_after_readback),
        "recovered_after_readback": recovered_after_readback,
        "readiness_before": before,
        "readiness_after": after,
    }


async def champion_v2_cutover_readiness_report(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
) -> dict[str, Any]:
    """Return the operator-visible 100% positive-balance receipt gate."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.champion_settlement_v2 import (
        champion_v2_cutover_readiness,
    )
    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = (
        int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    )
    return await champion_v2_cutover_readiness(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )


async def _resolve_maintenance_epoch(epoch: int | None) -> int:
    """Resolve an operator command epoch without requiring gateway lifespan state."""

    if epoch is not None:
        return int(epoch)
    from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch

    resolved_epoch, _block, _source = (
        await resolve_research_lab_evaluation_epoch()
    )
    return int(resolved_epoch)
