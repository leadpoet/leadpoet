#!/usr/bin/env python3
"""Verify Research Lab Phase 2.4 meta-allocator contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.meta_allocator import verify_research_lab_meta_allocator


def main() -> int:
    summary = verify_research_lab_meta_allocator()
    print(
        "Research Lab Phase 2.4 meta-allocator verified: "
        f"{summary['prior_count']} priors, "
        f"selection={summary['selection_id']}, "
        f"selected={summary['selected_cell_refs']}, "
        f"prediction={summary['prediction_id']}, "
        f"pricing={summary['pricing_id']}, "
        f"pool_spend_cents={summary['pool_spend_cents']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
