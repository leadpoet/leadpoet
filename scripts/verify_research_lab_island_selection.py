#!/usr/bin/env python3
"""Verify Research Lab Phase 2.1 island-selection contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.island_selection import verify_research_lab_island_selection


def main() -> int:
    summary = verify_research_lab_island_selection()
    print(
        "Research Lab Phase 2.1 island selection verified: "
        f"{summary['candidate_count']} candidates, "
        f"selected={','.join(summary['selected_islands'])}, "
        f"tie-break={','.join(summary['tie_break_selected_islands'])}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
