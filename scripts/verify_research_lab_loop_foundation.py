#!/usr/bin/env python3
"""Verify Research Lab Phase 1 foundation contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.loop_foundation import verify_loop_foundation


def main() -> int:
    summary = verify_loop_foundation()
    print(
        "Research Lab Phase 1 foundation verified: "
        f"{summary['valid_release_records']} valid release records, "
        f"{summary['invalid_release_records']} invalid release records, "
        f"{summary['workflow_guard_fields']} workflow guards, "
        f"{summary['baseline_dependency_gates']} Phase 0 dependency gates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
