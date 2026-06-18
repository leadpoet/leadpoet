#!/usr/bin/env python3
"""Verify Research Lab Phase 1.7 Research Map v0 contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.research_map import verify_research_lab_research_map


def main() -> int:
    summary = verify_research_lab_research_map()
    print(
        "Research Lab Phase 1.7 research map verified: "
        f"projection {summary['projection_id']}, "
        f"artifact {summary['artifact_ref']}, "
        f"{summary['cell_count']} cells, "
        f"{summary['allocator_predictions']} allocator predictions, "
        f"{summary['engine_components']} engine components, "
        f"{summary['loop_failure_board_items']} loop-game board items."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
