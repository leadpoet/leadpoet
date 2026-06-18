#!/usr/bin/env python3
"""Verify Research Lab Phase 1.5 SOURCE_ADD coverage-track contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.source_add import verify_research_lab_source_add


def main() -> int:
    summary = verify_research_lab_source_add()
    print(
        "Research Lab Phase 1.5 SOURCE_ADD verified: "
        f"adapter {summary['adapter_id']}, "
        f"{summary['trial_outputs']} trial outputs, "
        f"yield {summary['measured_trial_yield']}, "
        f"band {summary['bounty_band_ref']} / {summary['bounty_cents']} cents inert, "
        f"bounty {summary['bounty_ref']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
