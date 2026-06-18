#!/usr/bin/env python3
"""Verify Research Lab Phase 1.3 Engine v1 typed-patch contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.engine_v1 import verify_research_lab_engine_v1


def main() -> int:
    summary = verify_research_lab_engine_v1()
    print(
        "Research Lab Phase 1.3 Engine v1 verified: "
        f"{summary['components']} components, "
        f"{summary['valid_patches']} valid patches, "
        f"{summary['invalid_patches']} invalid patches rejected, "
        f"{summary['bandit_cells']} uniform bandit cells, "
        f"selected {summary['selected_cell']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
