#!/usr/bin/env python3
"""Verify Research Lab Phase 3.0 foundation contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.model_pipeline_foundation import verify_model_pipeline_foundation


def main() -> int:
    summary = verify_model_pipeline_foundation()
    print(
        "Research Lab Phase 3 foundation verified: "
        f"{summary['data_gates']} data gates, "
        f"{summary['workflow_guard_fields']} workflow guards, "
        f"local_code_complete={summary['local_code_complete']}, "
        f"model_pipeline_operation_claimed_ready={summary['model_pipeline_operation_claimed_ready']}, "
        f"missing={summary['missing_data_gates']}, "
        f"market_ready={summary['market_production_operational_ready']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
