#!/usr/bin/env python3
"""Verify Research Lab Phase 1.9 baseline-arm operations contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.baseline_arm_ops import verify_research_lab_baseline_arm_ops


def main() -> int:
    summary = verify_research_lab_baseline_arm_ops()
    print(
        "Research Lab Phase 1.9 baseline-arm ops verified: "
        f"{summary['policy_id']} budget "
        f"${summary['daily_budget_min_cents'] / 100:.0f}-"
        f"${summary['daily_budget_max_cents'] / 100:.0f}/day, "
        f"comparison={summary['comparison_state']}, "
        f"monitor={summary['monitor_state']}, "
        f"publication={summary['publication_state']}, "
        f"{summary['map_cells']} map cell inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
