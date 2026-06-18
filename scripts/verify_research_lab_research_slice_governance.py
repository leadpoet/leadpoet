#!/usr/bin/env python3
"""Verify Phase 4.7 research-slice governance contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.research_slice_governance import verify_research_lab_research_slice_governance


def main() -> int:
    summary = verify_research_lab_research_slice_governance()
    print(
        "Research Lab Phase 4.7 research-slice governance verified: "
        f"proposal={summary['proposal_id']}, "
        f"proposal_ready={summary['proposal_ready']}, "
        f"decision={summary['decision_id']}, "
        f"raise_approved={summary['raise_approved']}, "
        f"current_bps={summary['current_research_slice_bps']}, "
        f"approved_bps={summary['approved_research_slice_bps']}, "
        f"budget_mutation_enabled={summary['budget_mutation_enabled']}, "
        f"emission_schedule_changed={summary['emission_schedule_changed']}, "
        f"ready_control_validates={summary['ready_control_validates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
