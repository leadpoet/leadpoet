#!/usr/bin/env python3
"""Live read-only check that the inner-loop activation migrations applied.

Confirms scripts/70 (research_lab_allocator_selection_records) and scripts/71
(research_lab_score_calibration) exist and are readable by service_role.

Read-only by design: both tables are append-only with a delete/update-blocking
trigger, so this never inserts a probe row. A `select <pk-column> limit 1`
round-trips through PostgREST — a missing table or column surfaces as an error.

Usage:
    python scripts/verify_inner_loop_migrations.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Run-from-anywhere: put the repo root on the path before importing gateway.*
# (matches scripts/run_research_lab_scoring_worker.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import order matters: gateway.config runs load_dotenv() so SUPABASE_* resolve.
import gateway.config  # noqa: F401,E402
from gateway.db.client import get_write_client  # noqa: E402

# (table, a representative column that must exist per the migration)
CHECKS = (
    ("research_lab_allocator_selection_records", "selection_record_id"),
    ("research_lab_allocator_selection_records", "selection_doc"),
    ("research_lab_score_calibration", "calibration_id"),
    ("research_lab_score_calibration", "realized_mean_delta"),
)


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
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(
            "\nOne or more objects are missing. Re-run the migration for the "
            "failing table (scripts/70 or scripts/71) and check for SQL errors."
        )
        return 1
    print("\nBoth inner-loop migrations verified: tables exist, columns present, "
          "service_role can read them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
