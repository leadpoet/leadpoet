#!/usr/bin/env python3
"""Verify Research Lab Phase 2.3 private-probe upgrade contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.private_probe_upgrade import verify_research_lab_private_probe_upgrade


def main() -> int:
    summary = verify_research_lab_private_probe_upgrade()
    print(
        "Research Lab Phase 2.3 private-probe upgrade verified: "
        f"probe={summary['probe_receipt_ref']}, "
        f"target_component={summary['target_component_type']}, "
        f"graduation_ready={summary['graduation_ready']}, "
        f"handoff={summary['handoff_id']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
