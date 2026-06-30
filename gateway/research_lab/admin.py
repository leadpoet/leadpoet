"""Operator CLI for Research Lab maintenance controls."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from .maintenance import (
    autoresearch_queue_status_counts,
    default_actor_ref,
    dumps_status,
    get_autoresearch_maintenance_state,
    pause_pending_autoresearch_runs,
    requeue_failed_candidate,
    requeue_failed_loop,
    requeue_paused_autoresearch_runs,
    set_autoresearch_maintenance_paused,
    wait_until_autoresearch_drained,
)
from .recovery import (
    rebase_stale_parent_candidates,
    requeue_baseline_not_ready_candidates,
    resume_failed_runs_from_checkpoint,
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

    sub.add_parser("status", help="Print maintenance state and queue counts")

    wait = sub.add_parser("wait-drained", help="Wait until no queued/started hosted runs remain")
    wait.add_argument("--timeout-seconds", type=int, default=1800)
    wait.add_argument("--poll-seconds", type=float, default=5.0)

    return parser


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
        return await rebase_stale_parent_candidates(
            candidate_ids=args.candidate_ids,
            dry_run=not args.apply,
            max_batch_size=args.max_batch_size,
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
    raise ValueError(f"unknown command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
