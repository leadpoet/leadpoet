#!/usr/bin/env python3
"""Verify Research Lab Phase 2.2 component-market contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.component_market import verify_research_lab_component_market


def main() -> int:
    summary = verify_research_lab_component_market()
    print(
        "Research Lab Phase 2.2 component market verified: "
        f"{len(summary['component_types'])} component types, "
        f"submission={summary['submission_id']}, "
        f"yield={summary['measured_trial_yield']}, "
        f"bounty_band={summary['bounty_band_ref']}, "
        f"bounty_cents={summary['bounty_cents']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
