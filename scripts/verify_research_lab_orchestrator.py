#!/usr/bin/env python3
"""Verify Research Lab Phase 1.1 local orchestrator scaffold."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.orchestrator import verify_research_lab_orchestrator


def main() -> int:
    summary = verify_research_lab_orchestrator()
    print(
        "Research Lab Phase 1.1 orchestrator verified: "
        f"{summary['queue_entries']} queue entries, "
        f"{summary['prior_receipts']} prior receipts, "
        f"{summary['novelty_reject_receipts']} reject receipt links, "
        f"probe {summary['probe_id']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
