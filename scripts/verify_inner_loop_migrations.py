#!/usr/bin/env python3
"""Live read-only check that the inner-loop activation migrations applied.

Confirms scripts/70 (allocator records), scripts/71 (score calibration), and
scripts/93 (automatic activation events/current state) exist and are readable
by service_role. Function existence is checked through PostgREST's schema
metadata without invoking the mutating transition RPC.

Read-only by design: both tables are append-only with a delete/update-blocking
trigger, so this never inserts a probe row. A `select <pk-column> limit 1`
round-trips through PostgREST — a missing table or column surfaces as an error.

Usage:
    python scripts/verify_inner_loop_migrations.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import httpx

# Run-from-anywhere: put the repo root on the path before importing gateway.*
# (matches scripts/run_research_lab_scoring_worker.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import order matters: gateway.config runs load_dotenv() so SUPABASE_* resolve.
import gateway.config  # noqa: F401,E402
from gateway.config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL  # noqa: E402
from gateway.db.client import get_write_client  # noqa: E402

# (table/view, a representative column that must exist per the migration)
CHECKS = (
    ("research_lab_allocator_selection_records", "selection_record_id"),
    ("research_lab_allocator_selection_records", "selection_doc"),
    ("research_lab_score_calibration", "calibration_id"),
    ("research_lab_score_calibration", "realized_mean_delta"),
    ("research_lab_inner_loop_activation_events", "event_hash"),
    ("research_lab_inner_loop_activation_events", "evidence_doc"),
    ("research_lab_inner_loop_activation_current", "phase"),
)
ACTIVATION_RPC_PATH = "/rpc/append_research_lab_inner_loop_activation_event"


def _activation_rpc_is_exposed() -> bool:
    """Inspect PostgREST OpenAPI metadata without invoking the write RPC."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False
    response = httpx.get(
        f"{SUPABASE_URL.rstrip('/')}/rest/v1/",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "accept": "application/openapi+json",
        },
        timeout=15.0,
    )
    response.raise_for_status()
    document = response.json()
    paths = document.get("paths") if isinstance(document, dict) else None
    return isinstance(paths, dict) and ACTIVATION_RPC_PATH in paths


def main() -> int:
    client = get_write_client()
    failures: list[str] = []
    checked_tables: set[str] = set()
    for table, column in CHECKS:
        try:
            client.table(table).select(column).limit(1).execute()
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            failures.append(f"{table}.{column}: {str(exc)[:200]}")
            continue
        checked_tables.add(table)

    for table in sorted(checked_tables):
        print(f"OK   {table}")
    try:
        if _activation_rpc_is_exposed():
            print("OK   append_research_lab_inner_loop_activation_event RPC")
        else:
            failures.append(
                "append_research_lab_inner_loop_activation_event: not exposed by PostgREST"
            )
    except Exception as exc:  # noqa: BLE001 - report metadata failure
        failures.append(
            "append_research_lab_inner_loop_activation_event: " + str(exc)[:200]
        )
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(
            "\nOne or more objects are missing. Re-run the migration for the "
            "failing object (scripts/70, scripts/71, or scripts/93) and check "
            "for SQL errors."
        )
        return 1
    print(
        "\nAll inner-loop migrations verified: tables/views exist, columns "
        "are present, and service_role can read them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
