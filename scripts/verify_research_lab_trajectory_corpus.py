#!/usr/bin/env python3
"""Verify Research Lab Phase 3.1 trajectory corpus contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.trajectory_corpus import verify_research_lab_trajectory_corpus


def main() -> int:
    summary = verify_research_lab_trajectory_corpus()
    print(
        "Research Lab Phase 3.1 trajectory corpus verified: "
        f"sources={summary['source_records']}, "
        f"trajectory_count={summary['trajectory_count']}, "
        f"splits={summary['train_count']}/{summary['validation_count']}/{summary['holdout_count']}, "
        f"training_ready={summary['training_ready_claimed']}, "
        f"distillation_ready={summary['distillation_ready_claimed']}, "
        f"calibration_ready={summary['calibration_ready_claimed']}, "
        f"builder={summary['builder_manifest_id']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
