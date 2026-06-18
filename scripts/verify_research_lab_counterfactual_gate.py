#!/usr/bin/env python3
"""Verify Research Lab Phase 2.9 counterfactual-gate local contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.counterfactual_gate import verify_research_lab_counterfactual_gate


def main() -> int:
    summary = verify_research_lab_counterfactual_gate()
    print(
        "Research Lab Phase 2.9 counterfactual gate verified: "
        f"comparison={summary['comparison_id']} ({summary['comparison_state']}), "
        f"miner_yield={summary['miner_yield']}, "
        f"allocator_yield={summary['allocator_yield']}, "
        f"matched_budget_cents={summary['matched_budget_cents']}, "
        f"consequence={summary['consequence_state']}, "
        f"allocator={summary['allocator_selection']}, "
        f"baseline_policy={summary['baseline_policy']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
