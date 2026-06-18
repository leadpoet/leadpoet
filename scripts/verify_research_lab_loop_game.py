#!/usr/bin/env python3
"""Verify Research Lab Phase 1.4 loop-game and autopilot contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.loop_game import verify_research_lab_loop_game


def main() -> int:
    summary = verify_research_lab_loop_game()
    print(
        "Research Lab Phase 1.4 loop game verified: "
        f"{summary['failure_board_items']} failure-board items, "
        f"{summary['private_probe_receipts']} private probe receipts, "
        f"receipt {summary['receipt_ref']}, "
        f"autopilot target {summary['autopilot_item_id']}, "
        f"spent {summary['committed_spend_cents']} / released {summary['released_cents']} cents."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
