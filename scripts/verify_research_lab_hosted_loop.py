#!/usr/bin/env python3
"""Verify Research Lab Hosted Loop MVP local/staged contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.hosted_loop import verify_research_lab_hosted_loop


def main() -> int:
    summary = verify_research_lab_hosted_loop()
    print(
        "Research Lab Hosted Loop MVP verified: "
        f"winning receipt {summary['winning_receipt_ref']}, "
        f"losing receipt {summary['losing_receipt_ref']}, "
        f"{summary['trajectory_events']} trajectory events, "
        f"{summary['results_rows']} result rows, "
        f"map {summary['map_projection_id']}, "
        f"spent {summary['actual_spend_cents']} / released {summary['released_cents']} cents, "
        f"{summary['provider_usage_count']} provider usage rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
