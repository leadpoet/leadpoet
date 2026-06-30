"""Verify derive_public_loop_outcome emits the canonical completed-label statuses.

Pure-function table test (no DB). Run:
  python3 scripts/verify_research_lab_public_status_derivation.py   (exit 0 == pass)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.research_lab.public_activity import derive_public_loop_outcome  # noqa: E402

TS = "2026-06-30T00:00:00+00:00"


def _ticket(status="running"):
    return {"current_ticket_status": status, "ticket_doc": {}, "island": "generalist", "miner_hotkey": "5H"}


def _q(status, reason=""):
    return [{"current_queue_status": status, "current_reason": reason, "current_status_at": TS, "run_id": "r1", "ticket_id": "t1"}]


def _cand(status, reason=""):
    return [{
        "current_candidate_status": status, "current_reason": reason, "current_status_at": TS,
        "run_id": "r1", "ticket_id": "t1", "candidate_id": "candidate:x", "redacted_public_summary": "",
    }]


def derive(ticket, queue_rows, candidate_rows):
    return derive_public_loop_outcome(
        ticket=ticket, queue_rows=queue_rows, receipt_rows=[], candidate_rows=candidate_rows,
        score_bundle_rows=[], promotion_event_rows=[],
    )


def check(errors, label, expected, ticket, queue_rows, candidate_rows):
    out = derive(ticket, queue_rows, candidate_rows)
    if out.outcome_label != expected:
        errors.append(f"[{label}] expected {expected}, got {out.outcome_label} (event_type={out.event_type})")


def main() -> int:
    errors: list[str] = []

    # --- canonical completed-label matrix ---
    check(errors, "completed+baseline_not_ready", "waiting_for_baseline",
          _ticket(), _q("completed"), _cand("queued", "baseline_not_ready"))
    check(errors, "completed+evaluating", "scoring",
          _ticket(), _q("completed"), _cand("evaluating"))
    check(errors, "completed+stale_parent_rejected", "needs_rescore",
          _ticket(), _q("completed"), _cand("rejected", "stale_parent_needs_rescore"))
    check(errors, "completed+no_candidate", "completed_no_candidate",
          _ticket(), _q("completed"), [])
    # --- credit-block ---
    check(errors, "paused+blocked_for_credit", "waiting_for_credits",
          _ticket(), _q("paused", "blocked_for_credit"), [])
    # --- regressions: existing behavior preserved ---
    check(errors, "started+no_candidate", "running",
          _ticket("running"), _q("started"), [])
    check(errors, "queued+no_candidate", "queued",
          _ticket("queued"), _q("queued"), [])
    check(errors, "evaluating (not completed)", "scoring",
          _ticket(), _q("started"), _cand("evaluating"))

    # --- migration marker check ---
    sql_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "55-research-lab-public-loop-canonical-status.sql")
    try:
        sql = open(sql_path).read()
        for label in ("waiting_for_credits", "waiting_for_baseline", "needs_rescore", "completed_no_candidate"):
            if sql.count(f"'{label}'") < 2:  # present in both event_type + outcome_label CHECKs
                errors.append(f"[migration] label {label} not widened in both CHECK constraints")
    except FileNotFoundError:
        errors.append("[migration] scripts/55 not found")

    if errors:
        print("FAIL — public status derivation verification")
        for e in errors:
            print("  -", e)
        return 1
    print("PASS — public status derivation (completed-label matrix: waiting_for_baseline / scoring / "
          "needs_rescore / completed_no_candidate; waiting_for_credits; running/queued regressions; "
          "migration widens both CHECKs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
