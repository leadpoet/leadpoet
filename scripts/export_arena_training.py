#!/usr/bin/env python3
"""Export one Arena execution with explicit proof gaps, using read-only access."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ledger(store, run_id):
    rows, cursor = [], None
    while True:
        page = store.list_ledger(run_id=run_id, after_entry_id=cursor, limit=1000)
        rows.extend(page)
        if len(page) < 1000:
            return rows
        next_cursor = int(page[-1]["entry_id"])
        if cursor is not None and next_cursor <= cursor:
            raise ValueError("ledger_pagination_failed")
        cursor = next_cursor


def export_run(store, objects, run_id):
    """Read only the requested round and its referenced accepted authorities."""
    from lab_arena.training_export import ExportError, assemble_example

    execution = store.get_run(run_id)
    if execution is None or execution.get("kind") != "execute":
        raise ExportError("execution_run_missing")
    round_row = store.get_round(execution["round_id"])
    submission = store.get_submission(execution["submission_id"])
    if not round_row or not submission:
        raise ExportError("context_missing")
    runs = store.list_runs(execution["round_id"])
    related = [row for row in runs if row.get("assignment_id") == execution["assignment_id"]
               or row.get("kind") == "score" and row.get("scored_run_id") == run_id]
    ledger = [entry for row in related for entry in _ledger(store, row["run_id"])]
    caches, company_rows = {}, {}
    for row in related:
        if row.get("kind") != "score":
            continue
        if row.get("judgment_cache_key"):
            key = row["judgment_cache_key"]
            cached = store.get_judgment_cache(key)
            if cached is not None:
                caches[key] = cached
        lease = (row.get("claim_response") or {}).get("company_judgment_cache") or {}
        for item in list(lease.get("hits") or []) + list(lease.get("misses") or []):
            identity = (item["cache_key"], item["authority_slot"])
            cached = store.get_company_judgment(*identity)
            if cached is not None:
                company_rows[identity] = cached
    return assemble_example(run_id=run_id, round_row=round_row, submission_row=submission,
                            runs=runs, ledger_rows=ledger, objects=objects,
                            cache_rows=list(caches.values()), company_judgment_rows=list(company_rows.values()))


def write_private(path: Path, document: dict) -> None:
    """Create a new mode-0600 file; do not overwrite prior artifacts or links."""
    from lab_arena.contracts import canonical_json

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(document) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", args.run_id):
        parser.error("invalid run ID")
    store = None
    try:
        if args.environment_file:
            from scripts.run_lab_arena_service import load_scoped_environment
            load_scoped_environment(args.environment_file)
        from lab_arena.service import S3ObjectStore
        from lab_arena.store import ArenaStore, PostgrestTransport
        # No service construction: it can connect chain clients and start work.
        store = ArenaStore(PostgrestTransport(os.environ["LAB_ARENA_SUPABASE_URL"],
                                               service_key=os.environ["LAB_ARENA_SERVICE_KEY"]))
        objects = S3ObjectStore(os.environ["LAB_ARENA_BUCKET"], region_name=os.environ.get("AWS_REGION"),
                                prefix=os.environ.get("LAB_ARENA_OBJECT_PREFIX", ""))
        result = export_run(store, objects, args.run_id)
        write_private(args.output, result)
        print(json.dumps({"complete": result["completeness"]["complete"],
                          "dimensions": result["completeness"]["dimensions"],
                          "incomplete_reasons": result["completeness"]["incomplete_reasons"],
                          "artifact": str(args.output)}, sort_keys=True))
        return 0 if result["completeness"]["complete"] else 2
    except Exception as exc:
        # Never print a database/object-store exception: it can contain endpoints
        # or request data. Fixed error type is sufficient for this CLI boundary.
        print("Arena training export failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
