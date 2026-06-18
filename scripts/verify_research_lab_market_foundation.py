#!/usr/bin/env python3
"""Verify Research Lab Phase 2.0 foundation contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.market_foundation import verify_market_foundation


def main() -> int:
    summary = verify_market_foundation()
    print(
        "Research Lab Phase 2 foundation verified: "
        f"{summary['dependency_gates']} dependency gates, "
        f"{summary['workflow_guard_fields']} workflow guards, "
        f"readiness={summary['local_readiness_id']}, "
        f"sealing={summary['local_sealing_state']}, "
        f"loop release fixtures={summary['loop_release_records']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
