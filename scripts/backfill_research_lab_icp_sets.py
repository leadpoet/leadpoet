#!/usr/bin/env python3
"""Dry-run-first Research Lab ICP set backfill.

This uses the current gateway ICP generator prompt/model. Writes require:

  --write --confirm-project qplwoislplkcegvdmbim

Backfilled sets are stored inactive and do not disturb the active daily set.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ID = "qplwoislplkcegvdmbim"


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Generate/backfill Research Lab ICP sets for prior UTC dates")
    parser.add_argument("--days", type=int, default=10, help="Number of previous UTC dates to generate")
    parser.add_argument("--write", action="store_true", help="Write inactive sets to Supabase")
    parser.add_argument("--confirm-project", default="", help=f"Required for writes: {PROJECT_ID}")
    args = parser.parse_args()

    if args.days <= 0:
        print("ERROR: --days must be positive")
        return 1
    if args.write:
        if args.confirm_project != PROJECT_ID:
            print(f"ERROR: writes require --confirm-project {PROJECT_ID}")
            return 1
        if PROJECT_ID not in os.getenv("SUPABASE_URL", ""):
            print("ERROR: SUPABASE_URL does not match confirmed project")
            return 1

    if not os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_KEY"]
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: set OPENROUTER_API_KEY or OPENROUTER_KEY")
        return 1

    from gateway.qualification.models import ICPPrompt
    from gateway.tasks.icp_generator import (
        OPENROUTER_MODEL,
        compute_icp_set_hash,
        generate_icps_with_openrouter,
        store_icp_set,
    )
    from gateway.research_lab.icp_window import select_rolling_icp_window_from_sets

    today = datetime.now(timezone.utc).date()
    generated_rows = []
    print(f"Research Lab ICP backfill dry_run={not args.write}, model={OPENROUTER_MODEL}")
    for offset in range(args.days, 0, -1):
        day = today - timedelta(days=offset)
        set_id = int(day.strftime("%Y%m%d"))
        result = await generate_icps_with_openrouter(set_id, total_icps=20)
        if not result:
            print(f"ERROR: generator returned no ICPs for set_id={set_id}")
            return 1
        icps, industry_distribution, icp_set_hash = result
        parsed = [ICPPrompt(**dict(icp)) for icp in icps]
        if len(parsed) != 20:
            print(f"ERROR: set_id={set_id} expected 20 ICPs, got {len(parsed)}")
            return 1
        row = {"set_id": set_id, "icps": icps, "icp_set_hash": icp_set_hash}
        generated_rows.append(row)
        one_day_window = select_rolling_icp_window_from_sets(
            [row],
            days=1,
            icps_per_day=5,
            allow_partial=False,
        )
        print(
            f"set_id={set_id} icps={len(icps)} hash=sha256:{icp_set_hash} "
            f"selected_5_hash={one_day_window.window_hash}"
        )
        if args.write:
            active_from = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            active_until = active_from + timedelta(days=1)
            ok = await store_icp_set(
                set_id=set_id,
                icps=icps,
                icp_set_hash=icp_set_hash,
                industry_distribution=industry_distribution,
                active_from=active_from,
                active_until=active_until,
                generation_seed=str(set_id),
            )
            if not ok:
                print(f"ERROR: failed to store inactive ICP set {set_id}")
                return 1

    window = select_rolling_icp_window_from_sets(
        generated_rows,
        days=min(args.days, 10),
        icps_per_day=5,
        allow_partial=False,
    )
    print(
        f"rolling_window_hash={window.window_hash} "
        f"sets={len(window.set_ids)} icps={len(window.benchmark_items)}"
    )
    print("No active set was changed." if args.write else "DRY RUN: no Supabase writes performed.")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
