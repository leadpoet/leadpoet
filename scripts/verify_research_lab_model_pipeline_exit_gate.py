#!/usr/bin/env python3
"""Verify Research Lab Phase 3.7 exit-gate contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.model_pipeline_exit_gate import verify_model_pipeline_exit_gate


def main() -> int:
    summary = verify_model_pipeline_exit_gate()
    print(
        "Research Lab Phase 3.7 exit gate verified: "
        f"exit_gate={summary['exit_gate_id']}, "
        f"local_code_complete={summary['local_code_complete']}, "
        f"production_operational_ready={summary['production_operational_ready']}, "
        f"missing_checks={','.join(summary['missing_checks'])}, "
        f"measured_control_validates={summary['measured_control_validates']}, "
        f"engine_threshold={summary['engine_yield_threshold_pct']}, "
        f"reranker_quality_threshold={summary['reranker_quality_threshold_pct']}, "
        f"reranker_cost_max={summary['reranker_cost_ratio_max_pct']}, "
        f"judge_auc_threshold={summary['judge_auc_threshold']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
