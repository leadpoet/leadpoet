#!/usr/bin/env python3
"""Verify Research Lab atomic claim guard migration markers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "scripts" / "42-research-lab-atomic-claim-guards.sql"


def main() -> int:
    sql = SQL_PATH.read_text(encoding="utf-8")
    required = {
        "guard_research_lab_run_claim",
        "guard_research_lab_candidate_claim",
        "guard_research_lab_credit_consume",
        "guard_research_loop_run_claim_insert",
        "guard_research_lab_candidate_claim_insert",
        "guard_research_loop_start_credit_consume_insert",
        "research_lab_run_claim_conflict",
        "research_lab_candidate_claim_conflict",
        "research_lab_credit_consume_conflict",
        "pg_advisory_xact_lock",
        "latest_status IS DISTINCT FROM 'queued'",
        "latest_status IS DISTINCT FROM 'available'",
    }
    missing = sorted(marker for marker in required if marker not in sql)
    if missing:
        for marker in missing:
            print(f"missing atomic claim guard marker: {marker}")
        return 1
    print("Research Lab atomic claim guard SQL verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
