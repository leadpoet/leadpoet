"""Operator CLI for Research Lab maintenance controls."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Mapping

from .config import ResearchLabGatewayConfig
from .maintenance import (
    autoresearch_queue_status_counts,
    default_actor_ref,
    dumps_status,
    get_autoresearch_maintenance_state,
    pause_pending_autoresearch_runs,
    rebase_stale_parent_candidates as maintenance_rebase_stale_parent_candidates,
    reconcile_terminal_loop_projections,
    repair_public_loop_cards,
    requeue_failed_candidate,
    requeue_failed_loop,
    requeue_paused_autoresearch_runs,
    resume_credit_blocked_run,
    set_autoresearch_maintenance_paused,
    wait_until_autoresearch_drained,
)
from .promotion import ResearchLabPromotionController, promotion_improvement_metric
from .recovery import (
    award_failed_run_reimbursements,
    rebase_stale_parent_candidates as recovery_rebase_stale_parent_candidates,
    recover_rebase_failed_candidates,
    requeue_baseline_not_ready_candidates,
    resume_failed_runs_from_checkpoint,
)
from .store import select_many, select_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Research Lab admin controls")
    sub = parser.add_subparsers(dest="command", required=True)

    pause = sub.add_parser("pause-autoresearch", help="Pause hosted auto-research admission and workers")
    pause.add_argument("--reason", required=True)
    pause.add_argument("--actor-ref", default=default_actor_ref())

    resume = sub.add_parser("resume-autoresearch", help="Resume hosted auto-research and requeue paused runs")
    resume.add_argument("--reason", default="maintenance complete")
    resume.add_argument("--actor-ref", default=default_actor_ref())
    resume.add_argument("--no-requeue", action="store_true")

    requeue_candidate = sub.add_parser(
        "requeue-candidate",
        help="Requeue a candidate terminally failed by the baseline-readiness race",
    )
    requeue_candidate.add_argument("--candidate-id", required=True)
    requeue_candidate.add_argument("--reason", default="baseline_ready")
    requeue_candidate.add_argument("--actor-ref", default=default_actor_ref())
    requeue_candidate.add_argument(
        "--dry-run", action="store_true", help="Report the planned requeue without writing"
    )
    requeue_candidate.add_argument(
        "--force",
        action="store_true",
        help="Skip the baseline-race and baseline-existence safety checks",
    )

    requeue_loop = sub.add_parser(
        "requeue-loop",
        help="Requeue an auto-research loop run that is terminally failed (resumes from "
        "checkpoint if present, else restarts); the reaper never recovers failed runs",
    )
    requeue_loop.add_argument("--run-id", help="exact run_id to requeue")
    requeue_loop.add_argument(
        "--ticket-id", help="ticket_id to requeue (resolves to that ticket's latest run)"
    )
    requeue_loop.add_argument("--reason", default="operator_resume")
    requeue_loop.add_argument("--actor-ref", default=default_actor_ref())
    requeue_loop.add_argument(
        "--dry-run", action="store_true", help="Report the planned requeue without writing"
    )
    requeue_loop.add_argument(
        "--force",
        action="store_true",
        help="Skip the failed-state and not-completed safety checks",
    )

    # --- Lifecycle recovery operators (default dry-run; pass --apply to write) ---
    resume_runs = sub.add_parser(
        "resume-failed-runs",
        help="Resume terminally failed hosted runs from their latest checkpoint",
    )
    resume_runs.add_argument(
        "--run-id", action="append", dest="run_ids", help="run_id to resume (repeatable; omit for all failed)"
    )
    resume_runs.add_argument("--actor-ref", default=default_actor_ref())
    resume_runs.add_argument(
        "--apply", action="store_true", help="Apply the requeue (default is dry-run)"
    )
    resume_runs.add_argument(
        "--no-require-openrouter-credit",
        dest="require_openrouter_credit",
        action="store_false",
        help="Skip the OpenRouter credit re-check for credit-blocked failures",
    )
    resume_runs.set_defaults(require_openrouter_credit=True)

    requeue_baseline = sub.add_parser(
        "requeue-baseline-not-ready",
        help="Requeue baseline_not_ready candidates whose baseline is now ready",
    )
    requeue_baseline.add_argument(
        "--candidate-id", action="append", dest="candidate_ids", help="candidate_id (repeatable; omit for all)"
    )
    requeue_baseline.add_argument("--actor-ref", default=default_actor_ref())
    requeue_baseline.add_argument(
        "--apply", action="store_true", help="Apply the requeue (default is dry-run)"
    )

    rebase_stale = sub.add_parser(
        "rebase-stale-parents",
        help="Rebase stale_parent_needs_rescore candidates onto the current active parent",
    )
    rebase_stale.add_argument(
        "--candidate-id", action="append", dest="candidate_ids", help="candidate_id (repeatable; omit for all)"
    )
    rebase_stale.add_argument("--max-batch-size", type=int, default=25)
    rebase_stale.add_argument("--actor-ref", default=default_actor_ref())
    rebase_stale.add_argument(
        "--apply", action="store_true", help="Apply the rebase build+queue (default is dry-run)"
    )

    recover_rebase_failed = sub.add_parser(
        "recover-rebase-failed",
        help="Recover candidates terminally stuck at stale_parent_rebase_failed "
        "(mark terminal + spawn a no-charge regeneration run under the same ticket)",
    )
    recover_rebase_failed.add_argument(
        "--candidate-id", action="append", dest="candidate_ids", help="candidate_id (repeatable; omit for all)"
    )
    recover_rebase_failed.add_argument("--max-batch-size", type=int, default=25)
    recover_rebase_failed.add_argument("--actor-ref", default=default_actor_ref())
    recover_rebase_failed.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Only mark candidates terminal; do not spawn a regeneration run",
    )
    recover_rebase_failed.add_argument(
        "--apply", action="store_true", help="Apply the recovery (default is dry-run)"
    )

    sub.add_parser("status", help="Print maintenance state and queue counts")

    wait = sub.add_parser("wait-drained", help="Wait until no queued/started hosted runs remain")
    wait.add_argument("--timeout-seconds", type=int, default=1800)
    wait.add_argument("--poll-seconds", type=float, default=5.0)

    reconcile = sub.add_parser(
        "reconcile-loop-projections",
        help="Repair terminal queue rows whose loop projection is still nonterminal",
    )
    reconcile.add_argument("--run-id")
    reconcile.add_argument("--limit", type=int, default=50)
    reconcile.add_argument("--reason", default="terminal_queue_reconciler")
    reconcile.add_argument("--actor-ref", default=default_actor_ref())
    reconcile.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    reconcile.add_argument("--write", dest="dry_run", action="store_false")

    repair_cards = sub.add_parser(
        "repair-public-cards",
        help="Re-project sanitized public loop cards from current append-only state",
    )
    repair_cards.add_argument("--ticket-id")
    repair_cards.add_argument("--limit", type=int, default=50)
    repair_cards.add_argument("--reason", default="operator_public_card_repair")
    repair_cards.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    repair_cards.add_argument("--write", dest="dry_run", action="store_false")

    rebase = sub.add_parser(
        "rebase-stale-candidates",
        help="Queue derived candidates for historical stale-parent candidate rebase",
    )
    rebase.add_argument("--candidate-id")
    rebase.add_argument("--limit", type=int, default=25)
    rebase.add_argument("--max-batch-size", type=int, default=5)
    rebase.add_argument("--actor-ref", default=default_actor_ref())
    rebase.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    rebase.add_argument("--write", dest="dry_run", action="store_false")

    resume_credit = sub.add_parser(
        "resume-credit-blocked-run",
        help="Explicitly resume a credit-blocked run only after OpenRouter key preflight passes",
    )
    resume_credit.add_argument("--run-id", required=True)
    resume_credit.add_argument("--reason", default="credit_preflight_passed_resume")
    resume_credit.add_argument("--actor-ref", default=default_actor_ref())
    resume_credit.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    resume_credit.add_argument("--write", dest="dry_run", action="store_false")

    failed_reimbursements = sub.add_parser(
        "award-failed-run-reimbursements",
        help="Backfill reimbursement awards for terminal failed auto-research runs with verified spend",
    )
    failed_reimbursements.add_argument("--run-id")
    failed_reimbursements.add_argument("--limit", type=int, default=50)
    failed_reimbursements.add_argument("--reason", default="operator_failed_run_reimbursement_backfill")
    failed_reimbursements.add_argument("--actor-ref", default=default_actor_ref())
    failed_reimbursements.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Report planned awards without writing; default writes awards/schedules",
    )
    failed_reimbursements.add_argument(
        "--write",
        dest="dry_run",
        action="store_false",
        help="Write awards/schedules; retained for compatibility and is the default",
    )

    promote_scored = sub.add_parser(
        "promote-scored-candidate",
        help="Replay promotion for an already-scored candidate without re-running scoring",
    )
    promote_scored.add_argument("--candidate-id", required=True)
    promote_scored.add_argument("--score-bundle-id")
    promote_scored.add_argument("--reason", default="operator_scored_candidate_promotion_replay")
    promote_scored.add_argument("--actor-ref", default=default_actor_ref())
    promote_scored.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Report the planned promotion without writing; default is dry-run",
    )
    promote_scored.add_argument("--write", dest="dry_run", action="store_false")
    promote_scored.add_argument(
        "--force",
        action="store_true",
        help="Replay even if this candidate already has an active_version_created event",
    )

    return parser


async def _promote_scored_candidate(
    *,
    candidate_id: str,
    score_bundle_id: str | None,
    dry_run: bool,
    force: bool,
    actor_ref: str,
    reason: str,
) -> dict[str, Any]:
    candidate = await select_one(
        "research_lab_candidate_evaluation_current",
        filters=(("candidate_id", candidate_id),),
    )
    if not candidate:
        return {
            "ok": False,
            "action": "promote-scored-candidate",
            "candidate_id": candidate_id,
            "error": "candidate_not_found",
        }

    existing_promotions = await select_many(
        "research_lab_candidate_promotion_events",
        columns="promotion_event_id,event_type,promotion_status,private_model_version_id,created_at",
        filters=(("candidate_id", candidate_id), ("event_type", "active_version_created")),
        order_by=(("created_at", True),),
        limit=1,
    )
    if existing_promotions and not force:
        return {
            "ok": True,
            "action": "promote-scored-candidate",
            "dry_run": dry_run,
            "candidate_id": candidate_id,
            "status": "already_promoted",
            "existing_promotion": existing_promotions[0],
        }

    bundle_row = await _resolve_score_bundle_for_candidate(
        candidate,
        score_bundle_id=score_bundle_id,
    )
    if not bundle_row:
        return {
            "ok": False,
            "action": "promote-scored-candidate",
            "candidate_id": candidate_id,
            "score_bundle_id": score_bundle_id,
            "error": "matching_score_bundle_not_found",
        }
    score_bundle = bundle_row.get("score_bundle_doc")
    if not isinstance(score_bundle, Mapping):
        return {
            "ok": False,
            "action": "promote-scored-candidate",
            "candidate_id": candidate_id,
            "score_bundle_id": bundle_row.get("score_bundle_id"),
            "error": "score_bundle_doc_missing",
        }
    bundle_status = str(bundle_row.get("current_event_status") or bundle_row.get("bundle_status") or "")
    if bundle_status != "scored":
        return {
            "ok": False,
            "action": "promote-scored-candidate",
            "candidate_id": candidate_id,
            "score_bundle_id": bundle_row.get("score_bundle_id"),
            "error": "score_bundle_not_currently_scored",
            "score_bundle_status": bundle_status,
        }

    expected_artifact = str(candidate.get("candidate_artifact_hash") or "")
    if expected_artifact and str(bundle_row.get("candidate_artifact_hash") or "") != expected_artifact:
        return {
            "ok": False,
            "action": "promote-scored-candidate",
            "candidate_id": candidate_id,
            "score_bundle_id": bundle_row.get("score_bundle_id"),
            "error": "score_bundle_candidate_artifact_mismatch",
            "candidate_artifact_hash": expected_artifact,
            "score_bundle_candidate_artifact_hash": bundle_row.get("candidate_artifact_hash"),
        }

    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    metric = promotion_improvement_metric(score_bundle)
    planned = {
        "candidate_id": candidate_id,
        "score_bundle_id": bundle_row.get("score_bundle_id"),
        "candidate_status": candidate.get("current_candidate_status"),
        "candidate_reason": candidate.get("current_reason"),
        "candidate_kind": candidate.get("candidate_kind"),
        "parent_artifact_hash": candidate.get("parent_artifact_hash"),
        "candidate_artifact_hash": bundle_row.get("candidate_artifact_hash"),
        "mean_delta": aggregates.get("mean_delta"),
        "promotion_improvement_points": metric.improvement_points,
        "promotion_metric": metric.event_doc(),
        "threshold_points": ResearchLabGatewayConfig.from_env().improvement_threshold_points,
        "reason": reason,
    }
    if dry_run:
        return {
            "ok": True,
            "action": "promote-scored-candidate",
            "dry_run": True,
            "planned": planned,
        }

    config = ResearchLabGatewayConfig.from_env()
    result = await ResearchLabPromotionController(
        config,
        worker_ref=actor_ref,
    ).process_scored_candidate(
        candidate=candidate,
        score_bundle_row=bundle_row,
        score_bundle=dict(score_bundle),
    )
    return {
        "ok": result.get("status") == "merged",
        "action": "promote-scored-candidate",
        "dry_run": False,
        "planned": planned,
        "promotion_result": result,
    }


async def _resolve_score_bundle_for_candidate(
    candidate: Mapping[str, Any],
    *,
    score_bundle_id: str | None,
) -> dict[str, Any] | None:
    if score_bundle_id:
        return await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", score_bundle_id),),
        )
    current_score_bundle_id = str(candidate.get("current_score_bundle_id") or "")
    if current_score_bundle_id:
        row = await select_one(
            "research_evaluation_score_bundle_current",
            filters=(("score_bundle_id", current_score_bundle_id),),
        )
        if row:
            return row
    artifact_hash = str(candidate.get("candidate_artifact_hash") or "")
    if not artifact_hash:
        manifest_doc = candidate.get("candidate_model_manifest_doc")
        if isinstance(manifest_doc, Mapping):
            artifact_hash = str(manifest_doc.get("model_artifact_hash") or "")
    if not artifact_hash:
        return None
    rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("candidate_artifact_hash", artifact_hash), ("current_event_status", "scored")),
        order_by=(("current_status_at", True),),
        limit=1,
    )
    return rows[0] if rows else None


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "pause-autoresearch":
        event = await set_autoresearch_maintenance_paused(
            paused=True,
            reason=args.reason,
            actor_ref=args.actor_ref,
            event_doc={"operator_action": "pause-autoresearch"},
        )
        pause_drain = await pause_pending_autoresearch_runs(
            actor_ref=args.actor_ref,
            reason=args.reason,
        )
        return {
            "ok": True,
            "action": "pause-autoresearch",
            "event_id": event.get("event_id"),
            "event_seq": event.get("seq"),
            "event_hash": event.get("anchored_hash"),
            "pause_drain": pause_drain,
            "state": await get_autoresearch_maintenance_state(),
            "queue_counts": await autoresearch_queue_status_counts(),
        }
    if args.command == "resume-autoresearch":
        event = await set_autoresearch_maintenance_paused(
            paused=False,
            reason=args.reason,
            actor_ref=args.actor_ref,
            event_doc={"operator_action": "resume-autoresearch"},
        )
        resume_requeue: dict[str, Any] = {
            "found_paused": 0,
            "requeued": 0,
            "capacity_limited": 0,
            "failed": 0,
            "blocked": [],
        }
        if not args.no_requeue:
            resume_requeue = await requeue_paused_autoresearch_runs(actor_ref=args.actor_ref)
        queue_counts = await autoresearch_queue_status_counts()
        remaining_paused = int(queue_counts.get("paused") or 0)
        ok = bool(args.no_requeue or remaining_paused == 0)
        return {
            "ok": ok,
            "action": "resume-autoresearch",
            "event_id": event.get("event_id"),
            "event_seq": event.get("seq"),
            "event_hash": event.get("anchored_hash"),
            "requeued_paused_runs": int(resume_requeue.get("requeued") or 0),
            "remaining_paused_runs": remaining_paused,
            "resume_requeue": resume_requeue,
            "state": await get_autoresearch_maintenance_state(),
            "queue_counts": queue_counts,
        }
    if args.command == "requeue-candidate":
        return await requeue_failed_candidate(
            candidate_id=args.candidate_id,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
            force=args.force,
        )
    if args.command == "requeue-loop":
        return await requeue_failed_loop(
            run_id=args.run_id,
            ticket_id=args.ticket_id,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
            force=args.force,
        )
    if args.command == "resume-failed-runs":
        return await resume_failed_runs_from_checkpoint(
            run_ids=args.run_ids,
            dry_run=not args.apply,
            require_openrouter_credit=args.require_openrouter_credit,
            actor_ref=args.actor_ref,
        )
    if args.command == "requeue-baseline-not-ready":
        return await requeue_baseline_not_ready_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            actor_ref=args.actor_ref,
        )
    if args.command == "rebase-stale-parents":
        return await recovery_rebase_stale_parent_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            max_batch_size=args.max_batch_size,
            actor_ref=args.actor_ref,
        )
    if args.command == "recover-rebase-failed":
        return await recover_rebase_failed_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            max_batch_size=args.max_batch_size,
            regenerate=not args.no_regenerate,
            actor_ref=args.actor_ref,
        )
    if args.command == "status":
        return {
            "ok": True,
            "state": await get_autoresearch_maintenance_state(),
            "queue_counts": await autoresearch_queue_status_counts(),
        }
    if args.command == "wait-drained":
        result = await wait_until_autoresearch_drained(
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        return {"ok": bool(result.get("drained")), **result}
    if args.command == "reconcile-loop-projections":
        return await reconcile_terminal_loop_projections(
            run_id=args.run_id,
            limit=args.limit,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "repair-public-cards":
        return await repair_public_loop_cards(
            ticket_id=args.ticket_id,
            limit=args.limit,
            reason=args.reason,
            dry_run=args.dry_run,
        )
    if args.command == "rebase-stale-candidates":
        return await maintenance_rebase_stale_parent_candidates(
            candidate_id=args.candidate_id,
            limit=args.limit,
            max_batch_size=args.max_batch_size,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "resume-credit-blocked-run":
        return await resume_credit_blocked_run(
            run_id=args.run_id,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "award-failed-run-reimbursements":
        return await award_failed_run_reimbursements(
            run_id=args.run_id,
            limit=args.limit,
            dry_run=args.dry_run,
            reason=args.reason,
            actor_ref=args.actor_ref,
        )
    if args.command == "promote-scored-candidate":
        return await _promote_scored_candidate(
            candidate_id=args.candidate_id,
            score_bundle_id=args.score_bundle_id,
            dry_run=args.dry_run,
            force=args.force,
            actor_ref=args.actor_ref,
            reason=args.reason,
        )
    raise ValueError(f"unknown command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
