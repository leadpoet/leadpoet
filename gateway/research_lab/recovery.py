"""Research Lab lifecycle recovery operators.

Operator-safe functions to intentionally continue blocked Research Lab work:

- ``resume_failed_runs_from_checkpoint`` — re-queue terminally ``failed`` hosted
  auto-research runs that have a saved checkpoint, so the hosted worker resumes
  them from that checkpoint. OpenRouter credit-blocked failures are only resumed
  once the miner key reads as funded again.
- ``requeue_baseline_not_ready_candidates`` — wake candidates stuck ``queued``
  with reason ``baseline_not_ready`` once their private baseline is actually ready.
- ``rebase_stale_parent_candidates`` — rebase ``stale_parent_needs_rescore``
  candidates onto the current active parent (reusing the scoring worker's rebase
  path) and queue the derived candidate for scoring.

All operators are append-only (they never UPDATE rows — they append events), default
to ``dry_run=True``, are idempotent (they only act on rows still in the relevant
pre-state and skip work that was already done), and stamp actor/source provenance.
They reuse existing primitives and the scoring worker's tested logic rather than
re-implementing baseline/diff/build machinery.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Mapping
from uuid import uuid4

from .config import ResearchLabGatewayConfig
from .icp_window import RollingIcpWindowUnavailable, fetch_rolling_icp_window
from .key_vault import OpenRouterKeyVaultError, preflight_openrouter_key
from .maintenance import (
    _is_queue_capacity_conflict,
    autoresearch_queue_capacity_doc,
    default_actor_ref,
)
from .promotion import load_active_private_model
from .public_activity import safe_project_public_loop_activity
from .reimbursement_awards import (
    cost_evidence_actual_microusd,
    create_reimbursement_decision,
    latest_reimbursable_loop_cost_evidence,
)
from .scoring_worker import CandidateBaselineNotReady, ResearchLabGatewayScoringWorker
from .store import (
    create_candidate_evaluation_event,
    create_queue_event,
    latest_auto_research_checkpoint,
    select_all,
    select_many,
    select_one,
)
from .worker import OpenRouterKeyResolver
from research_lab.eval import PrivateModelArtifactManifest

logger = logging.getLogger(__name__)


# An OpenRouter HTTP 402 / insufficient-credit failure is resumable once the miner
# tops up; it should not be retried in a tight loop but can be re-queued after a
# successful credit re-check. These markers match the failure text the hosted worker
# records on the terminal `failed` queue event.
_OPENROUTER_CREDIT_MARKERS = (
    "insufficient credits",
    "insufficient balance",
    "payment required",
    "more credits",
    "http 402",
    "402:",
    "code\":402",
)


def _is_openrouter_credit_failure(error_text: str) -> bool:
    low = (error_text or "").lower()
    return any(marker in low for marker in _OPENROUTER_CREDIT_MARKERS)


async def _project_after_recovery(
    ticket_id: str,
    *,
    source_ref: str,
    reason: str,
    config: ResearchLabGatewayConfig,
) -> None:
    """Best-effort public-card projection after a state-changing recovery action.

    Recovery operators change lifecycle state (requeues, terminal marks, rebases)
    that the public dashboard must reflect; without this the card freezes at its
    pre-recovery status. Projection failure never fails or aborts the recovery —
    ``safe_project_public_loop_activity`` already swallows errors, and this wrapper
    guards against anything escaping it.
    """
    if not ticket_id:
        return
    try:
        await safe_project_public_loop_activity(
            ticket_id,
            source_ref=source_ref,
            reason=reason,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - projection must never fail recovery
        logger.warning(
            "research_lab_recovery_projection_failed ticket_id=%s reason=%s error=%s",
            ticket_id,
            reason,
            str(exc)[:200],
        )


async def _latest_queue_event_error(run_id: str) -> str:
    rows = await select_many(
        "research_loop_run_queue_events",
        columns="event_type,reason,event_doc,seq",
        filters=(("run_id", run_id),),
        order_by=(("seq", True),),
        limit=1,
    )
    if not rows:
        return ""
    doc = rows[0].get("event_doc") if isinstance(rows[0].get("event_doc"), Mapping) else {}
    return str(doc.get("error") or rows[0].get("reason") or "")


async def _openrouter_credit_ready(
    config: ResearchLabGatewayConfig, *, ticket_id: str
) -> tuple[bool, str]:
    """Best-effort credit/key readiness check for the miner that owns ``ticket_id``.

    Read-only (resolves the miner key and calls OpenRouter's key-info endpoint).
    Returns ``(ready, detail)``. ``limit_remaining is None`` is treated as ready
    (unmetered key); any resolution/preflight error is treated as not-ready so a
    credit-blocked run is not re-queued into the same wall.
    """
    ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not ticket:
        return False, "ticket_not_found"
    miner_hotkey = str(ticket.get("miner_hotkey") or "")
    key_ref = str(ticket.get("miner_openrouter_key_ref") or "")
    if not miner_hotkey or not key_ref:
        return False, "missing_miner_openrouter_key_ref"
    try:
        env = await OpenRouterKeyResolver(config).resolve(key_ref, miner_hotkey=miner_hotkey)
        raw_key = str(env.get("OPENROUTER_API_KEY") or "")
        info = await asyncio.to_thread(preflight_openrouter_key, raw_key)
    except OpenRouterKeyVaultError as exc:
        return False, f"credit_preflight_failed:{str(exc)[:120]}"
    except Exception as exc:  # noqa: BLE001 - best-effort; any failure => not ready
        return False, f"credit_check_error:{str(exc)[:120]}"
    remaining = info.get("limit_remaining")
    if remaining is None:
        return True, "limit_remaining_unmetered"
    try:
        return (float(remaining) > 0.0), f"limit_remaining={remaining}"
    except (TypeError, ValueError):
        return True, f"limit_remaining_unparsed={remaining}"


# --------------------------------------------------------------------------- #
# Operator 0: backfill failed-run reimbursement awards
# --------------------------------------------------------------------------- #
async def award_failed_run_reimbursements(
    *,
    run_id: str | None = None,
    limit: int = 50,
    dry_run: bool = False,
    reason: str = "operator_failed_run_reimbursement_backfill",
    actor_ref: str | None = None,
) -> dict[str, Any]:
    """Award reimbursement for terminal failed runs with trusted positive spend.

    The operator is idempotent and writes by default. It never resumes work or
    mutates queues; it only creates the standard reimbursement award/schedule via
    the same formula used for completed runs.
    """
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()
    if run_id:
        row = await select_one(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_reason,current_status_at,"
                "current_event_hash,current_event_seq,queue_priority"
            ),
            filters=(("run_id", run_id),),
        )
        rows = [row] if row else []
    else:
        rows = await select_many(
            "research_loop_run_queue_current",
            columns=(
                "run_id,ticket_id,current_queue_status,current_reason,current_status_at,"
                "current_event_hash,current_event_seq,queue_priority"
            ),
            filters=(("current_queue_status", "failed"),),
            order_by=(("current_status_at", False),),
            limit=max(1, int(limit or 50)),
        )

    result: dict[str, Any] = {
        "ok": True,
        "action": "award-failed-run-reimbursements",
        "dry_run": dry_run,
        "found": 0,
        "planned": [],
        "awarded": [],
        "skipped": [],
        "failed": [],
    }

    for row in rows:
        if not row:
            continue
        current_status = str(row.get("current_queue_status") or "")
        rid = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if not rid or not ticket_id:
            result["skipped"].append({"run_id": rid, "reason": "missing_run_or_ticket"})
            continue
        if current_status != "failed":
            result["skipped"].append({"run_id": rid, "reason": "not_failed", "status": current_status})
            continue
        result["found"] += 1

        existing_award = await select_one(
            "research_reimbursement_award_current",
            columns="award_id,run_id,current_award_status,target_reimbursement_microusd",
            filters=(("run_id", rid),),
        )
        if existing_award:
            result["skipped"].append(
                {
                    "run_id": rid,
                    "reason": "already_awarded",
                    "award_id": existing_award.get("award_id"),
                    "status": existing_award.get("current_award_status"),
                }
            )
            continue

        ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
        if not ticket:
            result["skipped"].append({"run_id": rid, "reason": "ticket_not_found"})
            continue

        receipt_id = await _latest_receipt_id_for_run(rid)
        if not receipt_id:
            result["skipped"].append({"run_id": rid, "reason": "missing_receipt"})
            continue

        queue_events = await select_many(
            "research_loop_run_queue_events",
            columns="seq,reason,event_doc,created_at",
            filters=(("run_id", rid),),
            order_by=(("seq", True),),
            limit=50,
        )
        payment = await _latest_payment_for_ticket(ticket_id)
        loop_start_credit_id = _loop_start_credit_id_from_queue_events(queue_events)
        if not payment and not loop_start_credit_id:
            result["skipped"].append({"run_id": rid, "reason": "missing_loop_start_payment_or_credit"})
            continue

        miner_key_ref = _openrouter_key_ref_from_ticket_or_events(ticket, queue_events)
        if not miner_key_ref:
            result["skipped"].append({"run_id": rid, "reason": "missing_miner_openrouter_key"})
            continue

        evidence = await latest_reimbursable_loop_cost_evidence(rid)
        actual_microusd = cost_evidence_actual_microusd(evidence)
        actual_usd = round(actual_microusd / 1_000_000, 6)
        if actual_microusd <= 0:
            result["skipped"].append(
                {"run_id": rid, "reason": "no_reimbursable_compute", "actual_openrouter_cost_usd": actual_usd}
            )
            continue

        failure_reason = await _latest_queue_event_error(rid)
        if not failure_reason:
            failure_reason = str(row.get("current_reason") or "failed")
        budget_context = _run_budget_context_for_backfill(config, ticket, payment, queue_events)
        plan = {
            "run_id": rid,
            "ticket_id": ticket_id,
            "receipt_id": receipt_id,
            "actual_openrouter_cost_usd": actual_usd,
            "failure_reason": failure_reason[:200],
            "run_day": _day_from_status_at(row.get("current_status_at")),
        }
        if dry_run:
            result["planned"].append({**plan, "dry_run": True})
            continue

        try:
            decision = await create_reimbursement_decision(
                config,
                run_id=rid,
                ticket_id=ticket_id,
                ticket=ticket,
                payment=payment,
                receipt_id=receipt_id,
                budget_context=budget_context,
                cost_evidence=evidence,
                source="operator_failed_run_reimbursement_backfill",
                failed_run_reimbursement=True,
                failure_reason=failure_reason,
                queue_terminal_status="failed",
                actor_ref=actor,
                run_day=plan["run_day"],
                miner_openrouter_key_ref=miner_key_ref,
                preserved_loop_start_credit=bool(loop_start_credit_id),
                require_positive_cost=True,
                skip_ineligible_prereqs=True,
            )
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({"run_id": rid, "error": str(exc)[:300]})
            continue

        if not decision or "award_id" not in decision:
            result["skipped"].append({**plan, "reason": (decision or {}).get("status", "not_awarded")})
            continue
        result["awarded"].append({**plan, **decision, "reason": reason})
        await _project_after_recovery(
            ticket_id,
            source_ref=f"recovery_reimbursement_awarded:{rid}",
            reason="award_failed_run_reimbursements",
            config=config,
        )

    result["ok"] = not result["failed"]
    return result


# --------------------------------------------------------------------------- #
# Operator 1: resume failed hosted runs from checkpoint
# --------------------------------------------------------------------------- #
async def resume_failed_runs_from_checkpoint(
    *,
    run_ids: list[str] | None = None,
    dry_run: bool = True,
    require_openrouter_credit: bool = True,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()
    capacity_doc = autoresearch_queue_capacity_doc(config)
    columns = (
        "run_id,ticket_id,current_queue_status,queue_priority,"
        "current_event_hash,current_status_at"
    )

    if run_ids:
        rows: list[Mapping[str, Any]] = []
        for run_id in run_ids:
            row = await select_one(
                "research_loop_run_queue_current", columns=columns, filters=(("run_id", run_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_loop_run_queue_current",
            columns=columns,
            filters=(("current_queue_status", "failed"),),
            max_rows=10000,
        )

    result: dict[str, Any] = {
        "ok": True,
        "action": "resume-failed-runs-from-checkpoint",
        "dry_run": dry_run,
        "found": 0,
        "resumed": 0,
        "not_failed": [],
        "not_resumable_no_checkpoint": [],
        "blocked_for_credit": [],
        "blocked": [],
        "runs": [],
    }

    for row in rows:
        run_id = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if str(row.get("current_queue_status") or "") != "failed":
            result["not_failed"].append(run_id)
            continue
        result["found"] += 1

        checkpoint = await latest_auto_research_checkpoint(run_id)
        if not checkpoint:
            result["not_resumable_no_checkpoint"].append(run_id)
            result["runs"].append({"run_id": run_id, "status": "not_resumable_no_checkpoint"})
            continue
        checkpoint_hash = str(checkpoint.get("checkpoint_hash") or "")

        error_text = await _latest_queue_event_error(run_id)
        credit_failure = _is_openrouter_credit_failure(error_text)
        credit_detail: str | None = None
        if credit_failure and require_openrouter_credit:
            ready, credit_detail = await _openrouter_credit_ready(config, ticket_id=ticket_id)
            if not ready:
                result["blocked_for_credit"].append(run_id)
                result["runs"].append(
                    {"run_id": run_id, "status": "blocked_for_credit", "detail": credit_detail}
                )
                continue

        plan = {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "status": "resumable",
            "checkpoint_hash": checkpoint_hash,
            "credit_failure": credit_failure,
            "credit_detail": credit_detail,
        }
        if dry_run:
            result["runs"].append({**plan, "dry_run": True})
            continue

        try:
            event = await create_queue_event(
                run_id=run_id,
                ticket_id=ticket_id,
                event_type="queued",
                queue_priority=int(row.get("queue_priority") or 0),
                worker_ref=actor,
                reason="operator_resume_from_checkpoint",
                event_doc={
                    "schema_version": "1.0",
                    **capacity_doc,
                    "resume_source": "resume_failed_runs_from_checkpoint",
                    "previous_queue_status": "failed",
                    "previous_event_hash": row.get("current_event_hash"),
                    "previous_status_at": row.get("current_status_at"),
                    "checkpoint_hash": checkpoint_hash,
                    "previous_error": error_text[:300],
                    "credit_failure": credit_failure,
                    "actor_ref": actor,
                },
            )
        except Exception as exc:  # noqa: BLE001
            if _is_queue_capacity_conflict(exc):
                result["blocked"].append(
                    {"run_id": run_id, "error": "queue_capacity_or_hotkey_conflict", "detail": str(exc)[:200]}
                )
                result["runs"].append({**plan, "status": "blocked", "error": str(exc)[:200]})
                continue
            raise

        result["resumed"] += 1
        result["runs"].append(
            {
                **plan,
                "status": "resumed",
                "requeued_event_id": event.get("event_id"),
                "requeued_event_seq": event.get("seq"),
                "requeued_event_hash": event.get("anchored_hash"),
            }
        )
        await _project_after_recovery(
            ticket_id,
            source_ref=f"recovery_resume_from_checkpoint:{run_id}",
            reason="resume_failed_runs_from_checkpoint",
            config=config,
        )

    return result


async def resume_credit_blocked_runs_for_miner(
    miner_hotkey: str,
    *,
    run_ids: list[str] | None = None,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    """Re-queue a miner's credit-blocked (paused/blocked_for_credit) runs after a top-up.

    Re-checks the miner's OpenRouter key; re-queues funded runs with reason
    `credit_topup_resume` (the hosted worker's preflight is the final gate, so a
    still-unfunded key just re-pauses). Never consumes another loop-start payment.
    Used by the miner-facing gateway endpoint and available to operators. Fastapi-free
    so it is unit-testable without the web stack.
    """
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or f"miner:{str(miner_hotkey)[:12]}"
    capacity_doc = autoresearch_queue_capacity_doc(config)
    requested = set(run_ids or [])

    blocked = await select_all(
        "research_loop_run_queue_current",
        columns="run_id,ticket_id,current_queue_status,current_reason,queue_priority",
        filters=(("current_queue_status", "paused"), ("current_reason", "blocked_for_credit")),
        max_rows=10000,
    )
    requeued = 0
    still_blocked = 0
    results: list[dict[str, Any]] = []

    for row in blocked:
        run_id = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if requested and run_id not in requested:
            continue
        ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
        if not ticket or str(ticket.get("miner_hotkey") or "") != str(miner_hotkey):
            continue  # only this miner's runs

        ready, detail = await _openrouter_credit_ready(config, ticket_id=ticket_id)
        if not ready:
            still_blocked += 1
            results.append({"run_id": run_id, "status": "still_blocked_for_credit", "detail": detail})
            continue
        try:
            await create_queue_event(
                run_id=run_id,
                ticket_id=ticket_id,
                event_type="queued",
                queue_priority=int(row.get("queue_priority") or 0),
                worker_ref=actor,
                reason="credit_topup_resume",
                event_doc={
                    "schema_version": "1.0",
                    **capacity_doc,
                    "resume_source": "miner_credit_topup_resume",
                    "previous_queue_status": "paused",
                    "previous_reason": "blocked_for_credit",
                    "credit_check": detail,
                },
            )
            requeued += 1
            results.append({"run_id": run_id, "status": "requeued", "credit_check": detail})
            await _project_after_recovery(
                ticket_id,
                source_ref=f"recovery_credit_topup_resume:{run_id}",
                reason="resume_credit_blocked_runs_for_miner",
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            status = "requeue_capacity_conflict" if _is_queue_capacity_conflict(exc) else "requeue_failed"
            results.append({"run_id": run_id, "status": status, "detail": str(exc)[:160]})

    return {"requeued": requeued, "still_blocked": still_blocked, "results": results}


# --------------------------------------------------------------------------- #
# Operator 2: requeue baseline_not_ready candidates once the baseline is ready
# --------------------------------------------------------------------------- #
async def requeue_baseline_not_ready_candidates(
    *,
    candidate_ids: list[str] | None = None,
    dry_run: bool = True,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()

    if candidate_ids:
        rows: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            row = await select_one(
                "research_lab_candidate_evaluation_current", filters=(("candidate_id", candidate_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            filters=(
                ("current_candidate_status", "queued"),
                ("current_reason", "baseline_not_ready"),
            ),
            max_rows=10000,
        )

    result: dict[str, Any] = {
        "ok": True,
        "action": "requeue-baseline-not-ready",
        "dry_run": dry_run,
        "found": 0,
        "requeued": 0,
        "skipped": [],
        "still_waiting_for_baseline": [],
        "candidates": [],
    }

    worker: ResearchLabGatewayScoringWorker | None = None
    window = None

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        status = str(row.get("current_candidate_status") or "")
        reason = str(row.get("current_reason") or "")
        if status != "queued" or reason != "baseline_not_ready":
            result["skipped"].append({"candidate_id": candidate_id, "status": status, "reason": reason})
            continue
        manifest_doc = row.get("private_model_manifest_doc")
        if not isinstance(manifest_doc, Mapping):
            result["skipped"].append({"candidate_id": candidate_id, "reason": "missing_manifest_doc"})
            continue
        result["found"] += 1

        if worker is None:
            worker = ResearchLabGatewayScoringWorker(config, worker_ref=actor)
        if window is None:
            try:
                window = await fetch_rolling_icp_window(
                    days=config.lab_champion_eval_days,
                    icps_per_day=config.lab_champion_icps_per_day,
                    window_mode=config.lab_champion_window_mode,
                    fresh_icp_count=config.lab_champion_fresh_icp_count,
                    retained_icp_count=config.lab_champion_retained_icp_count,
                    min_new_icp_count=config.lab_champion_fresh_icp_count,
                    allow_partial=config.scoring_worker_allow_partial_icp_window,
                )
            except RollingIcpWindowUnavailable as exc:
                result["ok"] = False
                result["error"] = f"rolling_window_unavailable:{str(exc)[:160]}"
                return result

        artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
        try:
            await worker._candidate_private_holdout_gate(artifact=artifact, window_hash=window.window_hash)
        except CandidateBaselineNotReady:
            result["still_waiting_for_baseline"].append(candidate_id)
            result["candidates"].append(
                {"candidate_id": candidate_id, "status": "still_waiting_for_baseline"}
            )
            continue

        plan = {"candidate_id": candidate_id, "status": "baseline_ready"}
        if dry_run:
            result["candidates"].append({**plan, "dry_run": True})
            continue

        event = await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(row["run_id"]),
            ticket_id=str(row["ticket_id"]),
            event_type="queued",
            candidate_status="queued",
            reason="baseline_ready_requeue",
            evaluator_ref=actor,
            event_doc={
                "operator_action": "requeue-baseline-not-ready",
                "resume_source": "requeue_baseline_not_ready_candidates",
                "previous_reason": "baseline_not_ready",
                "rolling_window_hash": window.window_hash,
                "actor_ref": actor,
            },
        )
        result["requeued"] += 1
        result["candidates"].append(
            {
                **plan,
                "status": "requeued",
                "requeued_event_id": event.get("event_id"),
                "requeued_event_seq": event.get("seq"),
            }
        )
        await _project_after_recovery(
            str(row.get("ticket_id") or ""),
            source_ref=f"recovery_baseline_ready_requeue:{candidate_id}",
            reason="requeue_baseline_not_ready_candidates",
            config=config,
        )

    return result


# --------------------------------------------------------------------------- #
# Operator 3: rebase stale-parent candidates onto the current active parent
# --------------------------------------------------------------------------- #
async def rebase_stale_parent_candidates(
    *,
    candidate_ids: list[str] | None = None,
    dry_run: bool = True,
    max_batch_size: int = 25,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()

    if candidate_ids:
        rows: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            row = await select_one(
                "research_lab_candidate_evaluation_current", filters=(("candidate_id", candidate_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            filters=(
                ("current_candidate_status", "rejected"),
                ("current_reason", "stale_parent_needs_rescore"),
            ),
            max_rows=10000,
        )

    if max_batch_size and len(rows) > int(max_batch_size):
        rows = rows[: int(max_batch_size)]

    active = await load_active_private_model(config, register_bootstrap=False)
    active_parent = active.artifact.model_artifact_hash

    result: dict[str, Any] = {
        "ok": True,
        "action": "rebase-stale-parents",
        "dry_run": dry_run,
        "active_parent_artifact_hash": active_parent,
        "found": 0,
        "rebased": 0,
        "skipped": [],
        "already_rebased": [],
        "already_on_current_parent": [],
        "failed": [],
        "rebases": [],
    }

    worker: ResearchLabGatewayScoringWorker | None = None
    evaluation_epoch: int | None = None

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        status = str(row.get("current_candidate_status") or "")
        reason = str(row.get("current_reason") or "")
        if status != "rejected" or reason != "stale_parent_needs_rescore":
            result["skipped"].append({"candidate_id": candidate_id, "status": status, "reason": reason})
            continue
        result["found"] += 1

        existing = await select_many(
            "research_lab_candidate_promotion_events",
            columns="promotion_event_id,event_type,derived_candidate_id",
            filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
            limit=1,
        )
        if existing:
            result["already_rebased"].append(
                {"candidate_id": candidate_id, "derived_candidate_id": existing[0].get("derived_candidate_id")}
            )
            continue

        candidate_parent = str(row.get("parent_artifact_hash") or "")
        if candidate_parent == active_parent:
            result["already_on_current_parent"].append(candidate_id)
            continue

        plan = {
            "candidate_id": candidate_id,
            "candidate_parent_artifact_hash": candidate_parent,
            "active_parent_artifact_hash": active_parent,
            "status": "rebase_eligible",
        }
        if dry_run:
            result["rebases"].append({**plan, "dry_run": True})
            continue

        if worker is None:
            worker = ResearchLabGatewayScoringWorker(config, worker_ref=actor)
            evaluation_epoch = await worker._resolve_evaluation_epoch()

        try:
            rebase = await worker._maybe_rebase_stale_candidate_before_scoring(
                row,
                evaluation_epoch=int(evaluation_epoch or 0),
                elapsed_seconds=lambda: 0.0,
            )
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({"candidate_id": candidate_id, "error": str(exc)[:200]})
            result["rebases"].append({**plan, "status": "failed", "error": str(exc)[:200]})
            continue

        rebase_status = str(rebase.get("status") or "")
        if rebase_status == "stale_parent_rebased_to_current":
            result["rebased"] += 1
            result["rebases"].append(
                {
                    **plan,
                    "status": rebase_status,
                    "derived_candidate_id": rebase.get("derived_candidate_id"),
                    "repair_used": rebase.get("repair_used"),
                }
            )
            await _project_after_recovery(
                str(row.get("ticket_id") or ""),
                source_ref=f"recovery_stale_parent_rebased:{candidate_id}",
                reason="rebase_stale_parent_candidates",
                config=config,
            )
        else:
            result["rebases"].append({**plan, "status": rebase_status or "no_rebase", "detail": rebase})
            if rebase_status != "current_parent":
                result["failed"].append({"candidate_id": candidate_id, "status": rebase_status})

    if result["failed"]:
        result["ok"] = result["rebased"] > 0 or dry_run
    return result


# Fields the hosted worker needs on a run's queue event to do real work: the miner's
# OpenRouter key reference (so the loop can call the model) and the compute budget. They
# live on the run's first `queued` event, written at loop-start. A regeneration run copies
# them so the fresh loop runs identically, while preserving the original loop-start payment
# (no second charge) the same way the stale-parent rebase path does.
_REGENERATION_COPY_KEYS = (
    "payment_id",
    "payment_ref",
    "payment_kind",
    "loop_start_credit_id",
    "miner_openrouter_key_ref",
    "miner_openrouter_key_handling",
    "research_model_tier",
    "requested_compute_budget_usd",
    "max_compute_budget_usd",
)


async def _first_queued_event_doc(run_id: str) -> dict[str, Any]:
    rows = await select_many(
        "research_loop_run_queue_events",
        columns="run_id,seq,event_type,reason,event_doc",
        filters=(("run_id", run_id), ("event_type", "queued")),
        order_by=(("seq", False),),
        limit=1,
    )
    if not rows:
        return {}
    doc = rows[0].get("event_doc")
    return dict(doc) if isinstance(doc, Mapping) else {}


async def _existing_regeneration_run(ticket_id: str, source_candidate_id: str) -> str | None:
    rows = await select_many(
        "research_loop_run_queue_events",
        columns="run_id,event_doc",
        filters=(("ticket_id", ticket_id), ("reason", "regenerate_after_rebase_unavailable")),
        limit=200,
    )
    for row in rows:
        doc = row.get("event_doc")
        if isinstance(doc, Mapping) and str(doc.get("regenerated_from_candidate_id") or "") == source_candidate_id:
            return str(row.get("run_id") or "")
    return None


async def recover_rebase_failed_candidates(
    *,
    candidate_ids: list[str] | None = None,
    dry_run: bool = True,
    max_batch_size: int = 25,
    regenerate: bool = True,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    """Recover candidates terminally stuck at ``stale_parent_rebase_failed``.

    These are loops whose candidate could not be re-fit to the advanced live model
    (e.g. a legacy ``image_build`` whose stored build doc has no source-diff artifact to
    re-apply). The stale-parent rebase path cannot help them, so they sit terminally
    rejected while still looking unfinished on the public dashboard.

    For each such candidate this:
      1. Marks the old candidate explicitly terminal with reason
         ``stale_parent_rebase_unavailable`` (so the public projection reads it as a
         finished, failed loop rather than an in-progress one), and
      2. when ``regenerate`` is set, spawns a fresh run under the same ticket so the loop
         re-drafts a new candidate against the current parent. The fresh run copies the
         original loop-start payment and OpenRouter key reference, so the miner is not
         charged a second time (reimbursement preserved, exactly as the rebase path does).

    Append-only and idempotent: a candidate that already carries
    ``stale_parent_rebase_unavailable`` or already has a regeneration run is skipped.
    ``dry_run`` (default) reports the plan without writing.
    """
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()

    if candidate_ids:
        rows: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            row = await select_one(
                "research_lab_candidate_evaluation_current", filters=(("candidate_id", candidate_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            filters=(
                ("current_candidate_status", "rejected"),
                ("current_reason", "stale_parent_rebase_failed"),
            ),
            max_rows=10000,
        )

    if max_batch_size and len(rows) > int(max_batch_size):
        rows = rows[: int(max_batch_size)]

    result: dict[str, Any] = {
        "ok": True,
        "action": "recover-rebase-failed",
        "dry_run": dry_run,
        "regenerate": regenerate,
        "found": 0,
        "recovered": 0,
        "regenerated": 0,
        "skipped": [],
        "already_recovered": [],
        "failed": [],
        "plans": [],
    }

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        status = str(row.get("current_candidate_status") or "")
        reason = str(row.get("current_reason") or "")
        run_id = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if reason == "stale_parent_rebase_unavailable":
            result["already_recovered"].append({"candidate_id": candidate_id})
            continue
        if status != "rejected" or reason != "stale_parent_rebase_failed":
            result["skipped"].append({"candidate_id": candidate_id, "status": status, "reason": reason})
            continue
        if not run_id or not ticket_id:
            result["skipped"].append({"candidate_id": candidate_id, "reason": "missing_run_or_ticket"})
            continue
        result["found"] += 1

        existing_regen = await _existing_regeneration_run(ticket_id, candidate_id)
        if existing_regen:
            result["already_recovered"].append(
                {"candidate_id": candidate_id, "regenerated_run_id": existing_regen}
            )
            continue

        plan: dict[str, Any] = {
            "candidate_id": candidate_id,
            "ticket_id": ticket_id,
            "source_run_id": run_id,
            "candidate_kind": str(row.get("candidate_kind") or ""),
            "regenerate": regenerate,
        }

        source_doc = await _first_queued_event_doc(run_id) if regenerate else {}
        if regenerate and not source_doc.get("miner_openrouter_key_ref"):
            # Without the miner's key reference the fresh loop cannot call the model;
            # do not spawn a run that is guaranteed to fail. Mark terminal only.
            plan["regenerate"] = False
            plan["regenerate_skipped_reason"] = "source_run_missing_openrouter_key_ref"

        if dry_run:
            result["plans"].append({**plan, "dry_run": True})
            continue

        new_run_id = str(uuid4()) if plan["regenerate"] else None
        try:
            if plan["regenerate"]:
                regen_doc: dict[str, Any] = {
                    key: source_doc[key] for key in _REGENERATION_COPY_KEYS if key in source_doc
                }
                regen_doc.update(autoresearch_queue_capacity_doc(config))
                regen_doc.update(
                    {
                        "requested_loop_count": 1,
                        "regenerated_from_candidate_id": candidate_id,
                        "regenerated_from_run_id": run_id,
                        "resume_source": "recover_rebase_failed_candidates",
                        "recovered_from": "stale_parent_rebase_unavailable",
                        "reimbursement_preserved": True,
                        "reimbursement_source": "original_loop_start_payment",
                        "operator_ref": actor,
                    }
                )
                await create_queue_event(
                    run_id=new_run_id,
                    ticket_id=ticket_id,
                    event_type="queued",
                    queue_priority=0,
                    reason="regenerate_after_rebase_unavailable",
                    worker_ref=actor,
                    event_doc=regen_doc,
                )

            await create_candidate_evaluation_event(
                candidate_id=candidate_id,
                run_id=run_id,
                ticket_id=ticket_id,
                event_type="rejected",
                candidate_status="rejected",
                reason="stale_parent_rebase_unavailable",
                evaluator_ref=actor,
                event_doc={
                    "action": "rebase_failed_candidate_recovered",
                    "prior_reason": "stale_parent_rebase_failed",
                    "regenerated_run_id": new_run_id,
                    "regenerated": bool(plan["regenerate"]),
                    "operator_ref": actor,
                },
            )
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({"candidate_id": candidate_id, "error": str(exc)[:200]})
            continue

        result["recovered"] += 1
        if plan["regenerate"]:
            result["regenerated"] += 1
        result["plans"].append({**plan, "regenerated_run_id": new_run_id})
        await _project_after_recovery(
            ticket_id,
            source_ref=f"recovery_rebase_failed_recovered:{candidate_id}",
            reason="recover_rebase_failed_candidates",
            config=config,
        )

    if result["failed"]:
        result["ok"] = result["recovered"] > 0 or dry_run
    return result


async def _latest_receipt_id_for_run(run_id: str) -> str | None:
    rows = await select_many(
        "research_loop_receipt_current",
        columns="receipt_id,current_receipt_status,current_status_at",
        filters=(("run_id", run_id),),
        order_by=(("current_status_at", True),),
        limit=1,
    )
    return str(rows[0].get("receipt_id") or "") if rows else None


async def _latest_payment_for_ticket(ticket_id: str) -> Mapping[str, Any] | None:
    rows = await select_many(
        "research_loop_start_payments",
        filters=(("ticket_id", ticket_id),),
        order_by=(("verified_at", True),),
        limit=1,
    )
    return rows[0] if rows else None


def _loop_start_credit_id_from_queue_events(events: list[Mapping[str, Any]]) -> str | None:
    for event in events:
        doc = event.get("event_doc")
        if isinstance(doc, Mapping) and doc.get("loop_start_credit_id"):
            return str(doc["loop_start_credit_id"])
    return None


def _openrouter_key_ref_from_ticket_or_events(
    ticket: Mapping[str, Any],
    events: list[Mapping[str, Any]],
) -> str:
    direct = str(ticket.get("miner_openrouter_key_ref") or "").strip()
    if direct:
        return direct
    for event in events:
        doc = event.get("event_doc")
        if isinstance(doc, Mapping) and doc.get("miner_openrouter_key_ref"):
            return str(doc["miner_openrouter_key_ref"]).strip()
    return ""


def _run_budget_context_for_backfill(
    config: ResearchLabGatewayConfig,
    ticket: Mapping[str, Any],
    payment: Mapping[str, Any] | None,
    queue_events: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ticket_doc = ticket.get("ticket_doc") if isinstance(ticket.get("ticket_doc"), Mapping) else {}
    queue_doc = _latest_event_doc(queue_events)
    payment_doc = (
        payment.get("verification_doc")
        if payment and isinstance(payment.get("verification_doc"), Mapping)
        else {}
    )
    tier = (
        queue_doc.get("research_model_tier")
        or payment_doc.get("research_model_tier")
        or ticket_doc.get("research_model_tier")
        or config.default_auto_research_model_tier
    )
    requested_budget = (
        queue_doc.get("requested_compute_budget_usd")
        or payment_doc.get("compute_budget_usd")
        or payment_doc.get("requested_compute_budget_usd")
        or ticket_doc.get("requested_compute_budget_usd")
        or config.default_compute_budget_usd
    )
    max_budget = (
        queue_doc.get("max_compute_budget_usd")
        or payment_doc.get("max_compute_budget_usd")
        or ticket_doc.get("max_compute_budget_usd")
        or config.max_compute_budget_usd
    )
    return {
        "schema_version": "1.0",
        "research_model_tier": str(tier),
        "requested_compute_budget_usd": config.clamp_compute_budget_usd(requested_budget),
        "max_compute_budget_usd": config.clamp_compute_budget_usd(max_budget),
        "payment_kind": str(queue_doc.get("payment_kind") or payment_doc.get("payment_kind") or "loop_start"),
        "budget_policy_version": "research-lab-budget:v1",
    }


def _latest_event_doc(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    for event in events:
        doc = event.get("event_doc")
        if isinstance(doc, Mapping):
            return dict(doc)
    return {}


def _day_from_status_at(value: Any) -> str:
    if not value:
        return datetime.now(timezone.utc).date().isoformat()
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()
