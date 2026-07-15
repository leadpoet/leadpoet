#!/usr/bin/env python3
"""Live read-only check that the Git-tree inner-loop migration applied.

Confirms scripts/70 (allocator records), scripts/71 (score calibration), and
scripts/95 (tree state and atomic transitions) exist and are readable by
service_role. Function existence is checked through PostgREST's schema metadata
without invoking a mutating RPC.

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
    ("research_lab_autoresearch_trees", "tree_id"),
    ("research_lab_autoresearch_tree_nodes", "parent_node_id"),
    ("research_lab_autoresearch_tree_events", "event_hash"),
    ("research_lab_autoresearch_operation_current", "logical_operation_id"),
    ("research_lab_autoresearch_frontier_commitments", "frontier_hash"),
    ("research_lab_autoresearch_tree_current", "current_frontier_doc"),
    ("research_lab_autoresearch_tree_handoffs", "candidate_id"),
)
REQUIRED_RPC_PATHS = {
    "/rpc/create_research_lab_autoresearch_tree",
    "/rpc/plan_research_lab_autoresearch_tree_node",
    "/rpc/transition_research_lab_autoresearch_operation",
    "/rpc/append_research_lab_autoresearch_tree_event",
    "/rpc/commit_research_lab_autoresearch_frontier",
    "/rpc/select_research_lab_autoresearch_tree_final",
    "/rpc/fail_research_lab_autoresearch_tree",
    "/rpc/record_research_lab_autoresearch_tree_handoff",
}


def _missing_rpc_paths() -> set[str]:
    """Inspect PostgREST metadata without invoking any write RPC."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return set(REQUIRED_RPC_PATHS)
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
    if not isinstance(paths, dict):
        return set(REQUIRED_RPC_PATHS)
    return set(REQUIRED_RPC_PATHS).difference(paths)


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
        missing_rpcs = _missing_rpc_paths()
        for path in sorted(REQUIRED_RPC_PATHS.difference(missing_rpcs)):
            print(f"OK   {path.removeprefix('/rpc/')} RPC")
        failures.extend(
            f"{path.removeprefix('/rpc/')}: not exposed by PostgREST"
            for path in sorted(missing_rpcs)
        )
    except Exception as exc:  # noqa: BLE001 - report metadata failure
        failures.append("Git-tree RPC metadata: " + str(exc)[:200])
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(
            "\nOne or more objects are missing. Re-run the migration for the "
            "failing object (scripts/70, scripts/71, or scripts/95) and check "
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
