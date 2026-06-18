#!/usr/bin/env python3
"""Verify Research Lab Phase 3.4 Engine v-next A/B evaluator contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.engine_ab_evaluator import verify_research_lab_engine_ab_evaluator


def main() -> int:
    summary = verify_research_lab_engine_ab_evaluator()
    print(
        "Research Lab Phase 3.4 Engine A/B evaluator verified: "
        f"dataset={summary['fine_tune_dataset_id']}, "
        f"training={summary['fine_tune_training_id']}, "
        f"control={summary['control_result_id']}, "
        f"candidate={summary['candidate_result_id']}, "
        f"comparison={summary['comparison_id']}, "
        f"matched_budget_cents={summary['matched_budget_cents']}, "
        f"yield_delta_pct={summary['yield_delta_pct']}, "
        f"improvement_claimed={summary['improvement_claimed']}, "
        f"local_only={summary['local_only']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
