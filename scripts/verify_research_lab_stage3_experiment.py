#!/usr/bin/env python3
"""Verify Phase 4.5 Stage-3 end-to-end experiment contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.stage3_experiment import verify_research_lab_stage3_experiment


def main() -> int:
    summary = verify_research_lab_stage3_experiment()
    print(
        "Research Lab Phase 4.5 Stage-3 experiment verified: "
        f"plan={summary['experiment_plan_id']}, "
        f"plan_approved={summary['plan_approved']}, "
        f"run={summary['experiment_run_id']}, "
        f"production_experiment_run_valid={summary['production_experiment_run_valid']}, "
        f"heldout_eval={summary['heldout_eval_id']}, "
        f"measured_heldout_pass={summary['measured_heldout_pass']}, "
        f"success={summary['success_claim_id']}, "
        f"success_claimed={summary['success_claimed']}, "
        f"ready_control_validates={summary['ready_control_validates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
