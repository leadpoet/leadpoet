#!/usr/bin/env python3
"""Verify Research Lab local alpha reimbursement kernel."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.reimbursements import verify_research_lab_reimbursements


def main() -> int:
    summary = verify_research_lab_reimbursements()
    print(
        "Research Lab alpha reimbursement kernel verified: "
        f"low rate {summary['low_rebate_rate']:.6f}, "
        f"high rate {summary['high_rebate_rate']:.6f}, "
        f"low award {summary['low_award_microusd']} microusd, "
        f"high award {summary['high_award_microusd']} microusd, "
        f"schedule {summary['schedule_epochs']} epochs / "
        f"{summary['schedule_total_microusd']} microusd, "
        f"{summary['fixture_cases']} fixture runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
