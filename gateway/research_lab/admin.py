"""Operator CLI for Research Lab maintenance controls."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from gateway.deploy_readiness import (
    assert_resume_allowed,
    build_deploy_readiness,
    default_manifest_path,
    write_deploy_readiness_manifest,
)

from .config import ResearchLabGatewayConfig
from .maintenance import (
    autoresearch_queue_status_counts,
    candidate_scoring_status_counts,
    default_actor_ref,
    dumps_status,
    expire_unpaid_tickets,
    get_autoresearch_maintenance_state,
    get_scoring_maintenance_state,
    pause_pending_autoresearch_runs,
    rebase_stale_parent_candidates as maintenance_rebase_stale_parent_candidates,
    reconcile_terminal_loop_projections,
    reconcile_terminal_ticket_statuses,
    repair_public_loop_cards,
    requeue_failed_candidate,
    requeue_failed_loop,
    requeue_stale_started_autoresearch_runs,
    requeue_paused_autoresearch_runs,
    resume_credit_blocked_run,
    set_autoresearch_maintenance_paused,
    set_scoring_maintenance_paused,
    ticket_lifecycle_health,
    wait_until_autoresearch_drained,
)
from .promotion import (
    CONFIRMATION_REJECTED_REASON,
    ResearchLabPromotionController,
    confirmation_doc_from_event,
    load_baseline_summary_doc_for_gate,
    load_confirmation_state,
    promotion_confirmation_rerun_enabled,
    promotion_improvement_metric,
    reconcile_active_private_model_lineage,
    reconcile_failed_private_source_pushes,
    reconcile_pending_champion_rewards,
    reregister_active_manifest,
    sync_active_model_to_repo_head,
)
from .store import select_many, select_one


logger = logging.getLogger(__name__)


def _add_deploy_readiness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gateway-commit", help="current gateway source commit")
    parser.add_argument("--validator-commit", help="current validator source commit")
    parser.add_argument("--gateway-pcr0", help="current gateway enclave PCR0")
    parser.add_argument("--validator-pcr0", help="current validator enclave PCR0")
    parser.add_argument("--expected-gateway-commit", help="expected gateway source commit")
    parser.add_argument("--expected-validator-commit", help="expected validator source commit")
    parser.add_argument("--expected-gateway-pcr0", help="expected gateway enclave PCR0")
    parser.add_argument("--expected-validator-pcr0", help="expected validator enclave PCR0")
    parser.add_argument(
        "--require-same-commit",
        action="store_true",
        help="fail unless gateway and validator commits match",
    )
    parser.add_argument(
        "--require-pcr0",
        action="store_true",
        help="fail unless both supplied gateway and validator PCR0s are present",
    )
    parser.add_argument(
        "--require-pcr0-commit-match",
        action="store_true",
        help="fail unless matched static PCR0 allowlist metadata points at the running commit",
    )
    parser.add_argument(
        "--include-docker-health",
        action="store_true",
        help="include Docker daemon and disk headroom health as a warning check",
    )
    parser.add_argument(
        "--require-docker-build-health",
        action="store_true",
        help="run a tiny Docker smoke build and fail readiness if Docker/build storage is unhealthy",
    )


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

    pause_scoring = sub.add_parser("pause-scoring", help="Pause new candidate scoring claims")
    pause_scoring.add_argument("--reason", required=True)
    pause_scoring.add_argument("--actor-ref", default=default_actor_ref())

    resume_scoring = sub.add_parser("resume-scoring", help="Resume candidate scoring claims")
    resume_scoring.add_argument("--reason", default="maintenance complete")
    resume_scoring.add_argument("--actor-ref", default=default_actor_ref())

    deploy_readiness = sub.add_parser(
        "check-deploy-readiness",
        help="Check gateway/validator commit and PCR0 alignment before resuming production work",
    )
    _add_deploy_readiness_args(deploy_readiness)
    deploy_readiness.add_argument(
        "--write-manifest",
        nargs="?",
        const="",
        help=(
            "write the check result to a resume-guard manifest; omit the value to use "
            f"{default_manifest_path()}"
        ),
    )
    deploy_readiness.add_argument(
        "--no-enforce-resume-block",
        action="store_true",
        help="write the manifest for visibility only; do not let it block resume commands",
    )

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

    requeue_stale_started = sub.add_parser(
        "requeue-stale-started-runs",
        help="Discover stale hosted auto-research runs stuck in started and requeue them append-only",
    )
    requeue_stale_started.add_argument("--reason", default="operator_requeue_stale_started")
    requeue_stale_started.add_argument("--actor-ref", default=default_actor_ref())
    requeue_stale_started.add_argument("--max-batch-size", type=int, default=25)
    requeue_stale_started.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    requeue_stale_started.add_argument("--write", dest="dry_run", action="store_false")

    recover_stale_candidate_claims = sub.add_parser(
        "recover-stale-candidate-claims",
        help="Discover stale candidate scoring claims and recover them append-only across all scoring shards",
    )
    recover_stale_candidate_claims.add_argument("--actor-ref", default=default_actor_ref())
    recover_stale_candidate_claims.add_argument("--max-batch-size", type=int, default=200)
    recover_stale_candidate_claims.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    recover_stale_candidate_claims.add_argument("--write", dest="dry_run", action="store_false")

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

    reconcile_tickets = sub.add_parser(
        "reconcile-terminal-tickets",
        help="Repair tickets still open after all expected queue runs are terminal",
    )
    reconcile_tickets.add_argument("--ticket-id")
    reconcile_tickets.add_argument("--limit", type=int, default=50)
    reconcile_tickets.add_argument("--reason", default="terminal_ticket_status_reconciler")
    reconcile_tickets.add_argument("--actor-ref", default=default_actor_ref())
    reconcile_tickets.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    reconcile_tickets.add_argument("--write", dest="dry_run", action="store_false")

    expire_tickets = sub.add_parser(
        "expire-unpaid-tickets",
        help="Expire miner tickets that passed the fixed 24-hour unpaid deadline",
    )
    expire_tickets.add_argument(
        "--ticket-id",
        action="append",
        dest="ticket_ids",
        help="exact ticket_id to inspect or expire (repeatable; omit for oldest eligible tickets)",
    )
    expire_tickets.add_argument("--limit", type=int, default=100)
    expire_tickets.add_argument("--reason", default="operator_unpaid_ticket_expiry")
    expire_tickets.add_argument("--actor-ref", default=default_actor_ref())
    expire_tickets.add_argument(
        "--apply",
        action="store_true",
        help="Append expired events; default is a read-only dry run",
    )

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
        help="Replay even when this candidate already has a prior promoted event",
    )

    sub.add_parser(
        "check-duplicate-active",
        help="Read-only: list private model versions whose latest event is 'active' "
        "(more than one means a duplicate-active state; supersede strays before "
        "relying on promotions)",
    )

    reregister_manifest = sub.add_parser(
        "reregister-active-manifest",
        help="Deliberately re-register the active lineage row after an operator "
        "manifest update (mutable_manifest_hash_mismatch): verifies the manifest "
        "at the row's URI and, on --write, supersedes the mismatched row and "
        "registers the loaded manifest as the new active version",
    )
    reregister_manifest.add_argument("--actor-ref", default=default_actor_ref())
    reregister_manifest.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    reregister_manifest.add_argument("--write", dest="dry_run", action="store_false")

    reconcile_lineage = sub.add_parser(
        "reconcile-active-lineage",
        help="Repair the zero-active-versions crash window: when the lineage is "
        "non-empty but nothing is active, re-activate the newest superseded version",
    )
    reconcile_lineage.add_argument("--actor-ref", default=default_actor_ref())
    reconcile_lineage.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    reconcile_lineage.add_argument("--write", dest="dry_run", action="store_false")

    sync_repo_head = sub.add_parser(
        "sync-active-model-to-repo-head",
        help="Sync active private model lineage to the signed current.json for private repo main; "
        "dry-run reports repo/head/current active SHA state",
    )
    sync_repo_head.add_argument("--actor-ref", default=default_actor_ref())
    sync_repo_head.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    sync_repo_head.add_argument("--write", dest="dry_run", action="store_false")
    sync_repo_head.add_argument(
        "--wait-for-current-json",
        action="store_true",
        help="Poll current.json until it matches repo main before writing",
    )
    sync_repo_head.add_argument("--timeout-seconds", type=int, default=None)
    sync_repo_head.add_argument("--poll-seconds", type=int, default=None)

    reconcile_private_source = sub.add_parser(
        "reconcile-failed-private-source-pushes",
        help="Retry candidates that passed promotion but failed while pushing private source",
    )
    reconcile_private_source.add_argument(
        "--candidate-id", action="append", dest="candidate_ids", help="candidate_id (repeatable; omit for all failed)"
    )
    reconcile_private_source.add_argument("--limit", type=int, default=None)
    reconcile_private_source.add_argument("--retry-after-seconds", type=int, default=None)
    reconcile_private_source.add_argument("--actor-ref", default=default_actor_ref())
    reconcile_private_source.add_argument(
        "--apply", action="store_true", help="Apply retries/finalization (default is dry-run)"
    )

    reconcile_rewards = sub.add_parser(
        "reconcile-champion-rewards",
        help="Finalize champion rewards stuck at reward_pending_uid by re-resolving "
        "the miner UID and creating the obligation",
    )
    reconcile_rewards.add_argument(
        "--candidate-id", action="append", dest="candidate_ids", help="candidate_id (repeatable; omit for all pending)"
    )
    reconcile_rewards.add_argument("--limit", type=int, default=25)
    reconcile_rewards.add_argument("--actor-ref", default=default_actor_ref())
    reconcile_rewards.add_argument(
        "--apply", action="store_true", help="Apply the reward creation (default is dry-run)"
    )

    house_comparison = sub.add_parser(
        "house-arm-comparison",
        help="Read-only §9.3 honesty report: matched-budget house-vs-miner arm "
        "yield per dollar (candidates scored, deltas, keeps) over a date range",
    )
    house_comparison.add_argument(
        "--start-date", help="Window start (YYYY-MM-DD, inclusive; default 30 days ago)"
    )
    house_comparison.add_argument(
        "--end-date", help="Window end (YYYY-MM-DD, inclusive; default today)"
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

    force_bypassed_gates: list[str] = []
    bypass_gates: frozenset[str] = frozenset()
    if force:
        if existing_promotions:
            force_bypassed_gates.append("already_promoted")
        logger.warning(
            "promote-scored-candidate --force candidate=%s bypassing safeguards: %s",
            candidate_id,
            ", ".join(force_bypassed_gates) or "-",
        )

    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    holdout_gate = (
        score_bundle.get("private_holdout_gate")
        if isinstance(score_bundle.get("private_holdout_gate"), Mapping)
        else None
    )
    baseline_summary_doc, baseline_doc_status = await load_baseline_summary_doc_for_gate(holdout_gate)
    metric = promotion_improvement_metric(
        score_bundle,
        baseline_score_summary_doc=baseline_summary_doc,
    )
    confirmation_state = await _confirmation_state_summary(
        candidate_id=candidate_id,
        score_bundle_id=str(bundle_row.get("score_bundle_id") or ""),
    )
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
        "promotion_rejection_status": metric.rejection_status,
        "baseline_summary_doc_status": baseline_doc_status,
        "confirmation": confirmation_state,
        "force_bypassed_gates": force_bypassed_gates,
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
        bypass_gates=bypass_gates,
    )
    ok = result.get("status") == "merged"
    if result.get("status") == "already_promoted":
        private_source = result.get("private_source_status")
        private_status = private_source.get("status") if isinstance(private_source, Mapping) else ""
        reward_status = str(result.get("champion_reward_status") or "")
        ok = private_status != "failed" and reward_status in {"created", "already_created"}
    return {
        "ok": ok,
        "action": "promote-scored-candidate",
        "dry_run": False,
        "planned": planned,
        "promotion_result": result,
    }


async def _confirmation_state_summary(
    *,
    candidate_id: str,
    score_bundle_id: str,
) -> dict[str, Any]:
    """Operator-visible §5.2-2 confirmation state for one candidate+bundle.

    Derived from ``promotion.load_confirmation_state`` (hold / recorded
    measurement / attempts) plus the terminal ``below_threshold`` rejection
    event, so replay and dry-run output show whether a replay would hold for
    confirmation, decide from a recorded delta, or has already been rejected —
    with both deltas — before anything is written.
    """
    state = await load_confirmation_state(
        candidate_id=candidate_id,
        score_bundle_id=score_bundle_id,
    )
    held_event = state.get("held_event")
    result_event = state.get("result_event")
    summary: dict[str, Any] = {
        "confirmation_rerun_enabled": promotion_confirmation_rerun_enabled(),
        "latest_reason": str(state.get("latest_reason") or ""),
        "attempts": int(state.get("attempts") or 0),
        "held_pending_confirmation": held_event is not None and result_event is None,
    }
    if held_event is not None:
        held_doc = held_event.get("event_doc") if isinstance(held_event.get("event_doc"), Mapping) else {}
        summary["held_event"] = {
            "promotion_event_id": str(held_event.get("promotion_event_id") or ""),
            "created_at": held_event.get("created_at"),
            "first_pass_improvement_points": held_doc.get("first_pass_improvement_points"),
            "confirmation_min_delta": held_doc.get("confirmation_min_delta"),
            "baseline_benchmark_bundle_id": held_doc.get("baseline_benchmark_bundle_id"),
        }
    if result_event is not None:
        confirmation_doc = confirmation_doc_from_event(result_event)
        summary["recorded_confirmation"] = {
            "promotion_event_id": str(result_event.get("promotion_event_id") or ""),
            "created_at": result_event.get("created_at"),
            "confirmation_delta": confirmation_doc.get("confirmation_delta"),
            "window_match": confirmation_doc.get("window_match"),
            "rolling_window_hash": confirmation_doc.get("rolling_window_hash"),
        }
    rejection = await _confirmation_rejection_event(
        candidate_id=candidate_id,
        score_bundle_id=score_bundle_id,
    )
    if rejection is not None:
        rejection_doc = rejection.get("event_doc") if isinstance(rejection.get("event_doc"), Mapping) else {}
        summary["rejected_confirmation_failed"] = {
            "promotion_event_id": str(rejection.get("promotion_event_id") or ""),
            "created_at": rejection.get("created_at"),
            "failure_mode": rejection_doc.get("failure_mode"),
            "first_pass_improvement_points": rejection_doc.get("first_pass_improvement_points"),
            "confirmation_delta": rejection_doc.get("confirmation_delta"),
            "confirmation_min_delta": rejection_doc.get("confirmation_min_delta"),
        }
    return summary


async def _confirmation_rejection_event(
    *,
    candidate_id: str,
    score_bundle_id: str,
) -> dict[str, Any] | None:
    """The terminal ``rejected_confirmation_failed`` promotion event, if any.

    The rejection is written as a ``below_threshold`` event (not
    ``promotion_checked``), so it lives outside ``load_confirmation_state``'s
    event set; the reason match happens client-side to stay portable across
    stores without JSON-path filters.
    """
    rows = await select_many(
        "research_lab_candidate_promotion_events",
        columns=(
            "promotion_event_id,candidate_id,event_type,promotion_status,"
            "source_score_bundle_id,event_doc,created_at"
        ),
        filters=(
            ("candidate_id", candidate_id),
            ("source_score_bundle_id", score_bundle_id),
            ("event_type", "below_threshold"),
        ),
        order_by=(("created_at", True),),
        limit=25,
    )
    for row in rows:
        doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
        if str(doc.get("reason") or "") == CONFIRMATION_REJECTED_REASON:
            return dict(row)
    return None


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


async def _recover_stale_candidate_claims_operator(
    *,
    dry_run: bool,
    actor_ref: str,
    max_batch_size: int,
) -> dict[str, Any]:
    from dataclasses import replace

    from .scoring_worker import (
        ResearchLabGatewayScoringWorker,
        _completed_icp_count_from_progress_doc,
        _count_claim_attempts,
        _stale_claim_recovery_owner_index,
        _status_is_stale,
    )

    config = ResearchLabGatewayConfig.from_env()
    total_workers = max(1, int(config.scoring_worker_total_workers or 1))
    stale_after_seconds = max(120, int(config.scoring_worker_model_timeout_seconds or 900) + 60)
    max_attempts = int(config.scoring_worker_max_claim_requeues)
    limit = max(1, int(max_batch_size or 1))
    inspector = ResearchLabGatewayScoringWorker(
        config,
        worker_ref=f"{actor_ref}:candidate-claim-recovery-inspect",
    )

    rows: list[dict[str, Any]] = []
    for status in ("assigned", "evaluating"):
        rows.extend(
            await select_many(
                "research_lab_candidate_evaluation_current",
                columns=(
                    "candidate_id,run_id,ticket_id,receipt_id,current_candidate_status,current_status_at,"
                    "current_evaluator_ref,current_event_hash,private_model_manifest_doc,"
                    "candidate_model_manifest_doc,candidate_artifact_hash"
                ),
                filters=(("current_candidate_status", status),),
                order_by=(("current_status_at", False),),
                limit=limit,
            )
        )

    planned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item.get("current_status_at") or "")):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        if not _status_is_stale(row.get("current_status_at"), stale_after_seconds):
            continue
        event_rows = await select_many(
            "research_lab_candidate_evaluation_events",
            columns="candidate_id,event_type,candidate_status,reason,seq",
            filters=(
                ("candidate_id", candidate_id),
                ("event_type", "in", ("assigned", "queued")),
            ),
            order_by=(("seq", False),),
            limit=200,
        )
        claim_attempts = _count_claim_attempts(event_rows)
        owner_index = _stale_claim_recovery_owner_index(candidate_id, total_workers)
        progress_summary: dict[str, Any] = {}
        if claim_attempts >= max_attempts:
            progress_summary = await inspector._candidate_scoring_progress_summary(row)
        completed_icps = _completed_icp_count_from_progress_doc(progress_summary)
        action = (
            "requeue_progress_preserved"
            if claim_attempts >= max_attempts and completed_icps > 0
            else "fail_retry_limit"
            if claim_attempts >= max_attempts
            else "requeue"
        )
        planned.append(
            {
                "candidate_id": candidate_id,
                "current_candidate_status": row.get("current_candidate_status"),
                "current_evaluator_ref": row.get("current_evaluator_ref"),
                "current_status_at": row.get("current_status_at"),
                "recovery_reason": "stale_claim",
                "recovery_owner_worker_index": owner_index + 1,
                "claim_attempts": claim_attempts,
                "max_claim_attempts": max_attempts,
                "completed_icp_count": completed_icps,
                "scoring_progress": progress_summary if completed_icps > 0 else {},
                "action": action,
            }
        )
        if len(planned) >= limit:
            break

    summary = {
        "found": len(planned),
        "would_requeue": sum(1 for item in planned if item.get("action") == "requeue"),
        "would_requeue_progress_preserved": sum(
            1 for item in planned if item.get("action") == "requeue_progress_preserved"
        ),
        "would_fail_retry_limit": sum(
            1 for item in planned if item.get("action") == "fail_retry_limit"
        ),
    }
    if dry_run:
        return {
            "ok": True,
            "action": "recover-stale-candidate-claims",
            "dry_run": True,
            "stale_after_seconds": stale_after_seconds,
            "total_workers": total_workers,
            "summary": summary,
            "planned": planned,
        }

    results: list[dict[str, int]] = []
    for index in range(total_workers):
        worker_config = replace(
            config,
            scoring_worker_index=index,
            scoring_worker_total_workers=total_workers,
        )
        worker = ResearchLabGatewayScoringWorker(
            worker_config,
            worker_ref=f"{actor_ref}:candidate-claim-recovery-{index + 1}",
        )
        recovered = await worker._recover_stale_candidate_claims()
        if recovered:
            results.append({"worker_index": index + 1, "recovered": recovered})

    return {
        "ok": True,
        "action": "recover-stale-candidate-claims",
        "dry_run": False,
        "stale_after_seconds": stale_after_seconds,
        "total_workers": total_workers,
        "planned_before_write": planned,
        "summary_before_write": summary,
        "recovered": sum(item["recovered"] for item in results),
        "by_worker": results,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "check-deploy-readiness":
        result = build_deploy_readiness(
            gateway_commit=args.gateway_commit,
            validator_commit=args.validator_commit,
            gateway_pcr0=args.gateway_pcr0,
            validator_pcr0=args.validator_pcr0,
            expected_gateway_commit=args.expected_gateway_commit,
            expected_validator_commit=args.expected_validator_commit,
            expected_gateway_pcr0=args.expected_gateway_pcr0,
            expected_validator_pcr0=args.expected_validator_pcr0,
            require_same_commit=args.require_same_commit,
            require_pcr0=args.require_pcr0,
            require_pcr0_commit_match=args.require_pcr0_commit_match,
            include_docker_health=args.include_docker_health,
            require_docker_build_health=args.require_docker_build_health,
        )
        result["action"] = "check-deploy-readiness"
        if args.write_manifest is not None:
            manifest_path = write_deploy_readiness_manifest(
                result,
                args.write_manifest or None,
                enforce_resume_block=not args.no_enforce_resume_block,
            )
            result["manifest_path"] = str(manifest_path)
            result["enforce_resume_block"] = not args.no_enforce_resume_block
        return result

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
        try:
            deploy_guard = assert_resume_allowed()
        except RuntimeError as exc:
            return {
                "ok": False,
                "action": "resume-autoresearch",
                "blocked_reason": "deploy_readiness_guard_failed",
                "error": str(exc),
            }
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
            "deploy_readiness_guard": deploy_guard,
            "state": await get_autoresearch_maintenance_state(),
            "queue_counts": queue_counts,
        }
    if args.command == "pause-scoring":
        event = await set_scoring_maintenance_paused(
            paused=True,
            reason=args.reason,
            actor_ref=args.actor_ref,
            event_doc={"operator_action": "pause-scoring"},
        )
        return {
            "ok": True,
            "action": "pause-scoring",
            "event_id": event.get("event_id"),
            "event_seq": event.get("seq"),
            "event_hash": event.get("anchored_hash"),
            "state": await get_scoring_maintenance_state(),
            "candidate_counts": await candidate_scoring_status_counts(),
        }
    if args.command == "resume-scoring":
        try:
            deploy_guard = assert_resume_allowed()
        except RuntimeError as exc:
            return {
                "ok": False,
                "action": "resume-scoring",
                "blocked_reason": "deploy_readiness_guard_failed",
                "error": str(exc),
            }
        event = await set_scoring_maintenance_paused(
            paused=False,
            reason=args.reason,
            actor_ref=args.actor_ref,
            event_doc={"operator_action": "resume-scoring"},
        )
        return {
            "ok": True,
            "action": "resume-scoring",
            "event_id": event.get("event_id"),
            "event_seq": event.get("seq"),
            "event_hash": event.get("anchored_hash"),
            "deploy_readiness_guard": deploy_guard,
            "state": await get_scoring_maintenance_state(),
            "candidate_counts": await candidate_scoring_status_counts(),
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
    if args.command == "requeue-stale-started-runs":
        return await requeue_stale_started_autoresearch_runs(
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
            max_batch_size=args.max_batch_size,
        )
    if args.command == "recover-stale-candidate-claims":
        return await _recover_stale_candidate_claims_operator(
            dry_run=args.dry_run,
            actor_ref=args.actor_ref,
            max_batch_size=args.max_batch_size,
        )
    if args.command == "resume-failed-runs":
        from .recovery import resume_failed_runs_from_checkpoint

        return await resume_failed_runs_from_checkpoint(
            run_ids=args.run_ids,
            dry_run=not args.apply,
            require_openrouter_credit=args.require_openrouter_credit,
            actor_ref=args.actor_ref,
        )
    if args.command == "requeue-baseline-not-ready":
        from .recovery import requeue_baseline_not_ready_candidates

        return await requeue_baseline_not_ready_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            actor_ref=args.actor_ref,
        )
    if args.command == "rebase-stale-parents":
        from .recovery import rebase_stale_parent_candidates as recovery_rebase_stale_parent_candidates

        return await recovery_rebase_stale_parent_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            max_batch_size=args.max_batch_size,
            actor_ref=args.actor_ref,
        )
    if args.command == "recover-rebase-failed":
        from .recovery import recover_rebase_failed_candidates

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
            "scoring_state": await get_scoring_maintenance_state(),
            "queue_counts": await autoresearch_queue_status_counts(),
            "candidate_counts": await candidate_scoring_status_counts(),
            "ticket_lifecycle_health": await ticket_lifecycle_health(sample_limit=25),
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
    if args.command == "reconcile-terminal-tickets":
        return await reconcile_terminal_ticket_statuses(
            ticket_id=args.ticket_id,
            limit=args.limit,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "expire-unpaid-tickets":
        return await expire_unpaid_tickets(
            ticket_ids=args.ticket_ids,
            limit=args.limit,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=not args.apply,
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
        from .recovery import award_failed_run_reimbursements

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
    if args.command == "check-duplicate-active":
        return await _check_duplicate_active_versions()
    if args.command == "house-arm-comparison":
        from .house_arm import build_house_arm_comparison_report

        today = datetime.now(timezone.utc).date()
        end_date = args.end_date or today.isoformat()
        start_date = args.start_date or (today - timedelta(days=30)).isoformat()
        return await build_house_arm_comparison_report(
            start_date=start_date,
            end_date=end_date,
        )
    if args.command == "reregister-active-manifest":
        return await reregister_active_manifest(
            ResearchLabGatewayConfig.from_env(),
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "reconcile-active-lineage":
        return await reconcile_active_private_model_lineage(
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "sync-active-model-to-repo-head":
        return await sync_active_model_to_repo_head(
            ResearchLabGatewayConfig.from_env(),
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
            wait_for_repo_head=args.wait_for_current_json,
            wait_timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    if args.command == "reconcile-failed-private-source-pushes":
        return await reconcile_failed_private_source_pushes(
            ResearchLabGatewayConfig.from_env(),
            worker_ref=args.actor_ref,
            candidate_ids=args.candidate_ids,
            limit=args.limit,
            retry_after_seconds=args.retry_after_seconds,
            dry_run=not args.apply,
        )
    if args.command == "reconcile-champion-rewards":
        return await reconcile_pending_champion_rewards(
            ResearchLabGatewayConfig.from_env(),
            worker_ref=args.actor_ref,
            candidate_ids=args.candidate_ids,
            limit=args.limit,
            dry_run=not args.apply,
        )
    raise ValueError(f"unknown command: {args.command}")


async def _check_duplicate_active_versions() -> dict[str, Any]:
    """Read-only duplicate-active check (bug #4 / §0.3 Week-1 step 0a)."""

    rows = await select_many(
        "research_lab_private_model_version_current",
        columns=(
            "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
            "current_event_type,current_version_status,current_status_at,created_at"
        ),
        filters=(("current_version_status", "active"),),
        order_by=(("current_status_at", True),),
        limit=50,
    )
    return {
        "ok": len(rows) <= 1,
        "action": "check-duplicate-active",
        "active_version_count": len(rows),
        "duplicate_active": len(rows) > 1,
        "active_versions": rows,
        "guidance": (
            "at most one version may have a latest 'active' event; supersede stray "
            "versions before applying scripts/60 promotions"
            if len(rows) > 1
            else "ok"
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
