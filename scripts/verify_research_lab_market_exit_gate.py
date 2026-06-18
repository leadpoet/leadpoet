#!/usr/bin/env python3
"""Verify Research Lab Phase 2.10 exit-gate contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.market_exit_gate import verify_market_exit_gate


def main() -> int:
    summary = verify_market_exit_gate()
    print(
        "Research Lab Phase 2.10 exit gate verified: "
        f"local_code_complete={summary['local_code_complete']}, "
        f"production_operational_ready={summary['production_operational_ready']}, "
        f"exit_gate={summary['exit_gate_id']}, "
        f"missing={summary['missing_exit_gates']}, "
        f"components_seen={summary['component_types_seen']}, "
        f"counterfactual={summary['counterfactual_comparison']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
