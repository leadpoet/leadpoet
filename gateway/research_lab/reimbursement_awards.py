"""Shared Research Lab reimbursement award writer.

This module keeps gateway reimbursement writes on the same deterministic kernel
used by local verification. It intentionally does not know whether a run
completed successfully or failed; callers provide a trusted cost snapshot and
metadata describing why the decision is being written.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
from typing import Any, Mapping, Sequence

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.logging_utils import compact_ref
from gateway.research_lab.store import (
    create_participation_snapshot as store_create_participation_snapshot,
    create_reimbursement_award,
    create_reimbursement_schedule,
    select_all,
    select_many,
)
from research_lab.reimbursements import (
    ReimbursementCapUsage,
    build_reimbursement_schedule,
    compute_participation_score,
    compute_reimbursement_award,
)


logger = logging.getLogger(__name__)

_MICRO_USD = Decimal("1000000")


def cost_evidence_from_loop_result(loop_result: Any) -> dict[str, Any]:
    """Build a neutral cost snapshot from a loop result object."""
    ledger: Mapping[str, Any] = {}
    if loop_result is not None and callable(getattr(loop_result, "cost_ledger", None)):
        try:
            maybe_ledger = loop_result.cost_ledger()
            if isinstance(maybe_ledger, Mapping):
                ledger = maybe_ledger
        except Exception:  # noqa: BLE001 - cost evidence is best effort
            ledger = {}
    provider_usage = list(getattr(loop_result, "provider_usage", ()) or ()) if loop_result is not None else []
    return normalize_cost_evidence(
        {
            "source": "loop_result",
            "trusted_cost_ledger": True,
            "provider_usage": provider_usage,
            "cost_ledger": dict(ledger),
            "actual_openrouter_cost_usd": getattr(loop_result, "actual_openrouter_cost_usd", None),
            "actual_openrouter_cost_microusd": ledger.get("actual_openrouter_cost_microusd"),
            "estimated_cost_usd": getattr(loop_result, "estimated_cost_usd", None),
            "openrouter_call_count": getattr(loop_result, "openrouter_call_count", None),
            "iterations_completed": getattr(loop_result, "iterations_completed", None),
            "stop_reason": getattr(loop_result, "stop_reason", None),
        }
    )


async def latest_reimbursable_loop_cost_evidence(run_id: str) -> dict[str, Any]:
    """Return the latest append-only loop event cost evidence for ``run_id``.

    Prefer a positive-cost ledger. If no positive-cost ledger exists, return the
    latest ledger-shaped event so callers can record ``no_reimbursable_compute``.
    """
    rows = await select_many(
        "research_lab_auto_research_loop_events",
        columns=(
            "run_id,seq,event_type,anchored_hash,provider_usage,cost_ledger,"
            "event_doc,elapsed_seconds"
        ),
        filters=(("run_id", run_id),),
        order_by=(("seq", True),),
        limit=50,
    )
    fallback: dict[str, Any] | None = None
    for row in rows:
        evidence = normalize_cost_evidence(
            {
                "source": "loop_event",
                "source_event_seq": row.get("seq"),
                "source_event_hash": row.get("anchored_hash"),
                "source_event_type": row.get("event_type"),
                "provider_usage": row.get("provider_usage"),
                "cost_ledger": row.get("cost_ledger"),
                "elapsed_seconds": row.get("elapsed_seconds"),
            }
        )
        if cost_evidence_actual_microusd(evidence) > 0:
            return evidence
        if fallback is None and isinstance(row.get("cost_ledger"), Mapping):
            fallback = evidence
    return fallback or normalize_cost_evidence({})


def normalize_cost_evidence(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    ledger = raw.get("cost_ledger") if isinstance(raw.get("cost_ledger"), Mapping) else {}
    provider_usage = raw.get("provider_usage")
    if not isinstance(provider_usage, Sequence) or isinstance(provider_usage, (str, bytes, bytearray)):
        provider_usage = ()

    actual_microusd = _cost_microusd(raw, ledger)
    actual_usd = _microusd_to_usd(actual_microusd)
    estimated_usd = _float_or_none(raw.get("estimated_cost_usd"))
    if estimated_usd is None:
        estimated_usd = _float_or_none(ledger.get("estimated_cost_usd") or ledger.get("estimated_total_usd"))

    openrouter_call_count = _int_or_none(raw.get("openrouter_call_count"))
    if openrouter_call_count is None:
        openrouter_call_count = _int_or_none(ledger.get("openrouter_call_count"))

    iterations_completed = _int_or_none(raw.get("iterations_completed"))
    if iterations_completed is None:
        iterations_completed = _int_or_none(ledger.get("iterations_completed"))

    return {
        "schema_version": "1.0",
        "source": str(raw.get("source") or "unknown"),
        "trusted_cost_ledger": bool(raw.get("trusted_cost_ledger")) or bool(ledger),
        "actual_openrouter_cost_microusd": actual_microusd,
        "actual_openrouter_cost_usd": actual_usd,
        "estimated_cost_usd": estimated_usd,
        "openrouter_call_count": openrouter_call_count,
        "iterations_completed": iterations_completed,
        "stop_reason": str(raw.get("stop_reason") or ledger.get("stop_reason") or "")[:160],
        "elapsed_seconds": _float_or_none(raw.get("elapsed_seconds") or ledger.get("elapsed_seconds")),
        "provider_usage": [dict(item) for item in provider_usage if isinstance(item, Mapping)],
        "cost_ledger": dict(ledger),
        "source_event_seq": raw.get("source_event_seq"),
        "source_event_hash": raw.get("source_event_hash"),
        "source_event_type": raw.get("source_event_type"),
    }


def cost_evidence_actual_microusd(evidence: Mapping[str, Any] | None) -> int:
    if not isinstance(evidence, Mapping):
        return 0
    try:
        return max(0, int(evidence.get("actual_openrouter_cost_microusd") or 0))
    except (TypeError, ValueError):
        return 0


def cost_evidence_cost_ledger(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return {}
    ledger = evidence.get("cost_ledger")
    return dict(ledger) if isinstance(ledger, Mapping) else {}


def cost_evidence_provider_usage(evidence: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(evidence, Mapping):
        return []
    usage = evidence.get("provider_usage")
    if not isinstance(usage, Sequence) or isinstance(usage, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in usage if isinstance(item, Mapping)]


async def create_reimbursement_decision(
    config: Any,
    *,
    run_id: str,
    ticket_id: str,
    ticket: Mapping[str, Any],
    payment: Mapping[str, Any] | None,
    receipt_id: str | None,
    budget_context: Mapping[str, Any],
    cost_evidence: Mapping[str, Any] | None,
    source: str,
    failed_run_reimbursement: bool = False,
    failure_reason: str | None = None,
    queue_terminal_status: str | None = None,
    actor_ref: str | None = None,
    run_day: str | None = None,
    miner_openrouter_key_ref: str | None = None,
    preserved_loop_start_credit: bool = False,
    require_positive_cost: bool = False,
    skip_ineligible_prereqs: bool = False,
    autoresearch_result: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create a reimbursement award/schedule using the canonical formula.

    ``require_positive_cost`` and ``skip_ineligible_prereqs`` are intended for
    terminal failures, where zero-cost or untrusted historical rows should be
    reported as skipped rather than creating a non-payable award row.
    """
    if not (config.reimbursements_enabled or config.shadow_reimbursements_enabled):
        return None

    evidence = normalize_cost_evidence(cost_evidence)
    actual_microusd = cost_evidence_actual_microusd(evidence)
    actual_usd = float(evidence.get("actual_openrouter_cost_usd") or 0.0)
    key_present = bool(str(miner_openrouter_key_ref or ticket.get("miner_openrouter_key_ref") or "").strip())
    has_payment_or_credit = bool(payment) or bool(preserved_loop_start_credit)
    trusted_cost = bool(evidence.get("trusted_cost_ledger"))

    if require_positive_cost and actual_microusd <= 0:
        return _skip_decision(
            "no_reimbursable_compute",
            run_id=run_id,
            actual_openrouter_cost_usd=actual_usd,
            failed_run_reimbursement=failed_run_reimbursement,
            source=source,
            reason="trusted cost evidence has no positive OpenRouter spend",
        )
    if skip_ineligible_prereqs:
        if not receipt_id:
            return _skip_decision(
                "missing_valid_receipt",
                run_id=run_id,
                actual_openrouter_cost_usd=actual_usd,
                failed_run_reimbursement=failed_run_reimbursement,
                source=source,
            )
        if not has_payment_or_credit:
            return _skip_decision(
                "missing_loop_start_payment_or_credit",
                run_id=run_id,
                actual_openrouter_cost_usd=actual_usd,
                failed_run_reimbursement=failed_run_reimbursement,
                source=source,
            )
        if not key_present:
            return _skip_decision(
                "missing_miner_openrouter_key",
                run_id=run_id,
                actual_openrouter_cost_usd=actual_usd,
                failed_run_reimbursement=failed_run_reimbursement,
                source=source,
            )
        if not trusted_cost:
            return _skip_decision(
                "missing_trusted_cost_ledger",
                run_id=run_id,
                actual_openrouter_cost_usd=actual_usd,
                failed_run_reimbursement=failed_run_reimbursement,
                source=source,
            )

    if not isinstance(autoresearch_result, Mapping):
        raise RuntimeError(
            "V2 reimbursement requires the exact signed autoresearch result"
        )
    evaluation_epoch, _block, _epoch_source = await resolve_research_lab_evaluation_epoch(
        config.evaluation_epoch
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
    )
    from gateway.research_lab.v2_authority import authorize_reward_decision_v2

    graph_lookup = {
        "artifact_kind": "autoresearch_run",
        "artifact_ref": str(run_id),
    }
    if autoresearch_result.get("status") == "failed":
        graph_lookup["allow_failed_root"] = True
    autoresearch_graph = await load_business_artifact_graph_by_ref_v2(
        **graph_lookup
    )
    authority = await authorize_reward_decision_v2(
        epoch_id=int(evaluation_epoch),
        decision_kind="reimbursement",
        decision_payload={
            "source_request": {
                "run_id": str(run_id),
                "ticket_id": str(ticket_id),
                "receipt_id": str(receipt_id or ""),
            },
            "autoresearch_result": dict(autoresearch_result),
        },
        expected_result=None,
        artifact_kind="reimbursement_decision",
        artifact_ref="",
        parent_graphs=(autoresearch_graph,),
    )
    result = authority.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("V2 reimbursement result is missing")
    award = result.get("award")
    schedule = result.get("schedule")
    source_state = result.get("source_state")
    if (
        not isinstance(award, Mapping)
        or not isinstance(schedule, Mapping)
        or not isinstance(source_state, Mapping)
    ):
        raise RuntimeError("V2 reimbursement result is incomplete")
    award = dict(award)
    schedule = dict(schedule)
    snapshot_doc = source_state.get("participation_snapshot")
    policy = source_state.get("policy")
    cap_usage = source_state.get("cap_usage")
    run_cost = source_state.get("run_cost")
    if any(
        not isinstance(value, Mapping)
        for value in (snapshot_doc, policy, cap_usage, run_cost)
    ):
        raise RuntimeError("V2 reimbursement source state is incomplete")
    snapshot_doc = dict(snapshot_doc)
    policy = dict(policy)
    cap_usage = dict(cap_usage)
    run_cost = dict(run_cost)
    evaluation_epoch = max(0, int(schedule.get("start_epoch") or 1) - 1)
    snapshot_row = await _persist_participation_snapshot(
        snapshot_doc=snapshot_doc,
        policy=policy,
    )
    shadow_only = not config.reimbursements_enabled
    award_doc = {
        "schema_version": "1.0",
        "award": award,
        "run_cost": _redacted_reimbursement_run_cost(run_cost),
        "policy": policy,
        "participation_snapshot": snapshot_doc,
        "cap_usage": cap_usage,
        "shadow_only": shadow_only,
        "submission_allowed": config.reimbursements_enabled,
        "source": source,
        "evaluation_epoch": int(evaluation_epoch),
        "failed_run_reimbursement": bool(failed_run_reimbursement),
        "failure_reason": str(failure_reason or "")[:500],
        "queue_terminal_status": queue_terminal_status,
        "cost_evidence_event_seq": evidence.get("source_event_seq"),
        "cost_evidence_event_hash": evidence.get("source_event_hash"),
        "cost_evidence_event_type": evidence.get("source_event_type"),
        "receipt_id": receipt_id,
        "actual_openrouter_cost_usd": actual_usd,
        "reimbursement_preserved": bool(failed_run_reimbursement),
        "actor_ref": actor_ref,
    }
    schedule_doc = {
        "schema_version": "1.0",
        "schedule": schedule,
        "shadow_only": shadow_only,
        "submission_allowed": config.reimbursements_enabled,
        "source": source,
        "evaluation_epoch": int(evaluation_epoch),
        "failed_run_reimbursement": bool(failed_run_reimbursement),
    }
    award_row, _award_event = await create_reimbursement_award(
        award=award,
        receipt_id=receipt_id,
        participation_snapshot_id=str(snapshot_row["participation_snapshot_id"]),
        policy_id=str(policy["policy_id"]),
        award_doc=award_doc,
    )
    if str(award_row["award_id"]) != str(schedule["award_id"]):
        raise RuntimeError(
            "persisted reimbursement award differs from the V2 signed decision"
        )
    schedule_row = await create_reimbursement_schedule(schedule=schedule, schedule_doc=schedule_doc)
    logger.info(
        "research_lab_reimbursement_decision run_id=%s status=%s target_usd=%.6f openrouter_usd=%.6f source=%s failed_run=%s shadow_only=%s",
        compact_ref(run_id),
        award["status"],
        float(award["target_reimbursement_usd"]),
        actual_usd,
        source,
        bool(failed_run_reimbursement),
        shadow_only,
    )
    return {
        "status": award["status"],
        "award_id": str(award_row["award_id"]),
        "schedule_id": str(schedule_row["schedule_id"]),
        "target_reimbursement_usd": award["target_reimbursement_usd"],
        "rebate_rate": award["rebate_rate"],
        "actual_openrouter_cost_usd": round(actual_usd, 6),
        "shadow_only": shadow_only,
        "failed_run_reimbursement": bool(failed_run_reimbursement),
        "source": source,
    }


async def _build_participation_snapshot(
    config: Any,
    ticket: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    island = str(ticket.get("island") or config.reimbursement_default_island)
    lookback_end = datetime.now(timezone.utc)
    lookback_start = lookback_end - timedelta(days=7)
    ticket_rows = await select_all(
        "research_loop_ticket_current",
        filters=(("island", island),),
        order_by=(("created_at", True),),
    )
    ticket_rows = [
        row
        for row in ticket_rows
        if _row_dt(row.get("created_at") or row.get("current_status_at")) >= lookback_start
    ]
    ticket_ids = {str(row.get("ticket_id")) for row in ticket_rows}
    queue_rows: list[dict[str, Any]] = []
    for ticket_id in ticket_ids:
        if not ticket_id:
            continue
        queue_rows.extend(
            await select_all(
                "research_loop_run_queue_current",
                filters=(("ticket_id", ticket_id),),
                order_by=(("current_status_at", True),),
                max_rows=100,
            )
        )
    funded_queue_rows = [
        row
        for row in queue_rows
        if str(row.get("ticket_id")) in ticket_ids
        and str(row.get("current_queue_status")) in {"queued", "started", "paused", "completed"}
        and _row_dt(row.get("current_status_at")) >= lookback_start
    ]
    distinct_hotkeys = {str(row.get("miner_hotkey")) for row in ticket_rows if row.get("miner_hotkey")}
    brief_refs = {str(row.get("brief_sanitized_ref")) for row in ticket_rows if row.get("brief_sanitized_ref")}
    snapshot_doc = {
        "snapshot_id": f"participation:{island}:{lookback_end.date().isoformat()}",
        "island": island,
        "lookback_start": lookback_start.isoformat(),
        "lookback_end": lookback_end.isoformat(),
        "distinct_funded_hotkeys": len(distinct_hotkeys),
        "paid_loop_count": len(funded_queue_rows),
        "unique_brief_count": len(brief_refs),
    }
    return snapshot_doc


async def _persist_participation_snapshot(
    *,
    snapshot_doc: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    participation_score = compute_participation_score(snapshot_doc, policy)
    return await store_create_participation_snapshot(
        island=str(snapshot_doc["island"]),
        lookback_start=snapshot_doc["lookback_start"],
        lookback_end=snapshot_doc["lookback_end"],
        distinct_funded_hotkeys=snapshot_doc["distinct_funded_hotkeys"],
        paid_loop_count=snapshot_doc["paid_loop_count"],
        unique_brief_count=snapshot_doc["unique_brief_count"],
        participation_score=float(participation_score),
        policy_id=str(policy["policy_id"]),
        snapshot_doc={
            **snapshot_doc,
            "participation_score": float(participation_score),
            "source": "research_loop_ticket_current_and_run_queue_current",
            "postgrest_limit": 1000,
        },
    )


async def _reimbursement_cap_usage(
    config: Any,
    ticket: Mapping[str, Any],
    *,
    run_day: str,
) -> dict[str, float]:
    rows = await select_all(
        "research_reimbursement_award_current",
        filters=(("current_award_status", "awarded"), ("run_day", run_day)),
    )
    miner_hotkey = str(ticket.get("miner_hotkey") or "")
    island = str(ticket.get("island") or config.reimbursement_default_island)
    eligible_rows = [
        row
        for row in rows
        if str(row.get("run_day")) == run_day
        and str(row.get("current_award_status") or row.get("award_status")) == "awarded"
    ]
    return {
        "hotkey_day_awarded_usd": _sum_award_usd(
            row for row in eligible_rows if str(row.get("miner_hotkey")) == miner_hotkey
        ),
        "island_day_awarded_usd": _sum_award_usd(
            row for row in eligible_rows if str(row.get("island")) == island
        ),
        "global_awarded_usd": _sum_award_usd(eligible_rows),
    }


def _skip_decision(
    status: str,
    *,
    run_id: str,
    actual_openrouter_cost_usd: float,
    failed_run_reimbursement: bool,
    source: str,
    reason: str | None = None,
) -> dict[str, Any]:
    logger.info(
        "research_lab_reimbursement_skipped run_id=%s status=%s source=%s reason=%s",
        compact_ref(run_id),
        status,
        source,
        reason or "",
    )
    return {
        "status": status,
        "skip_reason": reason or status,
        "actual_openrouter_cost_usd": round(float(actual_openrouter_cost_usd or 0.0), 6),
        "failed_run_reimbursement": bool(failed_run_reimbursement),
        "source": source,
    }


def _cost_microusd(raw: Mapping[str, Any], ledger: Mapping[str, Any]) -> int:
    for value in (
        raw.get("actual_openrouter_cost_microusd"),
        ledger.get("actual_openrouter_cost_microusd"),
        ledger.get("total_microusd"),
    ):
        parsed = _int_or_none(value)
        if parsed is not None:
            return max(0, parsed)
    for value in (
        raw.get("actual_openrouter_cost_usd"),
        ledger.get("actual_openrouter_cost_usd"),
        ledger.get("total_usd"),
    ):
        parsed = _float_or_none(value)
        if parsed is not None:
            return _usd_to_microusd(parsed)
    return 0


def _usd_to_microusd(value: Any) -> int:
    return int((Decimal(str(value or 0)) * _MICRO_USD).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _microusd_to_usd(value: int) -> float:
    return round(max(0, int(value)) / 1_000_000, 6)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _row_dt(value: Any) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sum_award_usd(rows: Any) -> float:
    total = Decimal("0")
    for row in rows:
        try:
            total += Decimal(str(row.get("target_reimbursement_microusd", 0))) / _MICRO_USD
        except Exception:
            continue
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _redacted_reimbursement_run_cost(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "run_id",
        "miner_hotkey",
        "island",
        "run_day",
        "funded_compute_budget_usd",
        "actual_openrouter_cost_usd",
        "loop_start_tao_fee_usd",
        "paid_research_loop",
        "valid_receipt",
        "verified_loop_start_payment",
        "preserved_loop_start_credit",
        "miner_openrouter_key_present",
        "trusted_cost_ledger",
        "passed_abuse_checks",
        "refunded",
        "voided",
        "duplicate",
        "novelty_rejected",
        "self_cancelled_before_minimum_work",
        "banned_hotkey",
    }
    return {key: value[key] for key in allowed if key in value}
