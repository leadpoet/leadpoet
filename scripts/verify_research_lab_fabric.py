#!/usr/bin/env python3
"""Verify Research Lab Phase 1.2 tiered run-fabric contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.fabric import verify_research_lab_fabric


def main() -> int:
    summary = verify_research_lab_fabric()
    print(
        "Research Lab Phase 1.2 fabric verified: "
        f"evidence {summary['evidence_bundle_ref']}, "
        f"snapshot {summary['snapshot_ref']}, "
        f"{summary['invalid_records']} invalid records rejected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
