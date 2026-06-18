#!/usr/bin/env python3
"""Verify Research Lab Phase 3.5 reranker distillation contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.reranker_distillation import verify_research_lab_reranker_distillation


def main() -> int:
    summary = verify_research_lab_reranker_distillation()
    print(
        "Research Lab Phase 3.5 reranker distillation verified: "
        f"corpus={summary['corpus_manifest_id']}, "
        f"dataset={summary['dataset_id']}, "
        f"dataset_ready={summary['dataset_ready_claimed']}, "
        f"training={summary['training_run_id']}, "
        f"training_started={summary['training_started']}, "
        f"teacher_eval={summary['teacher_eval_id']}, "
        f"student_eval={summary['student_eval_id']}, "
        f"comparison={summary['comparison_id']}, "
        f"quality_retention_pct={summary['quality_retention_pct']}, "
        f"cost_ratio_to_teacher_pct={summary['cost_ratio_to_teacher_pct']}, "
        f"parity_claimed={summary['parity_claimed']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
