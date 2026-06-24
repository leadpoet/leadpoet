#!/usr/bin/env python3
"""Read-only Supabase dry run for the Research Lab 60-ICP rolling window."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.icp_window import intent_signal_signature, select_rolling_icp_window_from_sets


def main() -> int:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        print("SKIP: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY for read-only dry run")
        return 0

    try:
        from supabase import create_client
    except Exception as exc:
        print(f"SKIP: supabase client unavailable: {exc.__class__.__name__}")
        return 0

    response = (
        create_client(url, key)
        .table("qualification_private_icp_sets")
        .select("set_id,icps,icp_set_hash,active_from,active_until,is_active")
        .order("set_id", desc=True)
        .limit(10)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    window = select_rolling_icp_window_from_sets(rows, days=10, icps_per_day=6, allow_partial=False)

    industries = set()
    sub_industries = set()
    signatures = set()
    for item in window.benchmark_items:
        icp = item.get("icp") if isinstance(item.get("icp"), dict) else {}
        industries.add(str(icp.get("industry") or "").strip().lower())
        sub_industries.add(str(icp.get("sub_industry") or "").strip().lower())
        signatures.add(intent_signal_signature(icp))

    print(
        "Research Lab Supabase ICP window dry run: "
        f"sets={len(window.set_ids)} icps={len(window.benchmark_items)} "
        f"industries={len(industries)} sub_industries={len(sub_industries)} "
        f"intent_signatures={len(signatures)} hash={window.window_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
