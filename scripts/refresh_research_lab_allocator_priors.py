"""Manually run the allocator-priors selection refresh (activation Phase 2).

The hosted worker runs this nightly on its designated index
(``RESEARCH_LAB_ALLOCATOR_PRIORS_REFRESH_*``); this CLI covers cron-based
deployments and operator backfills. Idempotent per (day, ledger window).

Usage:
    python scripts/refresh_research_lab_allocator_priors.py [--day YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day",
        default=None,
        help="UTC day to persist the selection under (default: today)",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        default=None,
        help="Ledger window size (default: allocator_priors.DEFAULT_WINDOW_ROWS)",
    )
    args = parser.parse_args()

    from gateway.research_lab.allocator_priors import (
        DEFAULT_WINDOW_ROWS,
        refresh_allocator_priors,
    )

    result = asyncio.run(
        refresh_allocator_priors(
            day=args.day,
            window_rows=args.window_rows or DEFAULT_WINDOW_ROWS,
            created_by="cli:refresh_research_lab_allocator_priors",
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("persisted", "already_persisted") else 1


if __name__ == "__main__":
    sys.exit(main())
