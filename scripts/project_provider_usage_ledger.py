#!/usr/bin/env python3
"""Project the evidence-proxy usage-ledger JSONL into Supabase (W3 §5.4).

The proxy appends one JSON row per provider call to a local JSONL file
(``RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH``). This script projects those rows
into ``public.research_lab_provider_usage_ledger`` (scripts/72) so spend can
be reconciled against provider dashboards.

Idempotent: each row's primary key is a deterministic UUID over its content,
and rows are upserted with ignore-duplicates — re-running over the same file
(or an ever-growing file) never double-counts. Run from cron, e.g.:

    */15 * * * *  python3 scripts/project_provider_usage_ledger.py

Options:
    --ledger PATH   override the JSONL path (default: env)
    --batch N       upsert batch size (default 500)
    --dry-run       parse + count, no writes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TABLE = "research_lab_provider_usage_ledger"
_VALID_EVIDENCE = {
    "hit", "recorded", "error", "blocked", "quota_exhausted", "credential_missing", "replay_miss",
}


def _projected_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    from gateway.research_lab.store import deterministic_uuid

    utc_day = str(raw.get("utc_day") or "")
    evidence = str(raw.get("evidence") or "")
    if len(utc_day) != 10 or evidence not in _VALID_EVIDENCE:
        return None
    row = {
        "schema_version": "1.0",
        "utc_day": utc_day,
        "recorded_at": str(raw.get("recorded_at") or "") or None,
        "provider_id": str(raw.get("provider_id") or "unknown")[:80],
        "endpoint_class": str(raw.get("endpoint_class") or "")[:200],
        "request_fingerprint": str(raw.get("request_fingerprint") or "")[:200],
        "evidence": evidence,
        "status": int(raw.get("status") or 0),
        "est_cost_microusd": max(0, int(raw.get("est_cost_microusd") or 0)),
        "caller_doc": raw.get("caller") if isinstance(raw.get("caller"), dict) else {},
    }
    if row["recorded_at"] is None:
        row.pop("recorded_at")
    # Deterministic PK over the full source row: same JSONL line → same UUID →
    # duplicate upserts are no-ops.
    row["usage_row_id"] = deterministic_uuid("provider_usage_ledger", json.dumps(raw, sort_keys=True))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Project provider usage JSONL into Supabase")
    parser.add_argument(
        "--ledger",
        default=os.getenv("RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH") or "",
        help="usage-ledger JSONL path (default: RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH)",
    )
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.ledger:
        print("usage ledger path not configured (set RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH)", file=sys.stderr)
        return 2
    ledger_path = Path(args.ledger)
    if not ledger_path.is_file():
        print(json.dumps({"ledger": str(ledger_path), "rows_read": 0, "rows_upserted": 0, "note": "no file yet"}))
        return 0

    rows: list[dict[str, Any]] = []
    skipped = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            projected = _projected_row(raw) if isinstance(raw, dict) else None
            if projected is None:
                skipped += 1
                continue
            rows.append(projected)

    if args.dry_run:
        print(json.dumps({"ledger": str(ledger_path), "rows_read": len(rows), "skipped": skipped, "dry_run": True}))
        return 0

    from gateway.db.client import get_write_client

    client = get_write_client()
    upserted = 0
    batch_size = max(1, int(args.batch))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # ignore_duplicates: existing usage_row_id rows are left untouched —
        # the table is append-only (UPDATE is trigger-blocked), so a plain
        # upsert-with-merge would error on re-runs.
        client.table(TABLE).upsert(batch, on_conflict="usage_row_id", ignore_duplicates=True).execute()
        upserted += len(batch)

    print(
        json.dumps(
            {
                "ledger": str(ledger_path),
                "rows_read": len(rows),
                "rows_upserted": upserted,
                "skipped": skipped,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
