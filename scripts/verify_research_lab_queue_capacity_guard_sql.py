#!/usr/bin/env python3
"""Verify Research Lab queue capacity guard migration markers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "scripts" / "43-research-lab-queue-capacity-guard.sql"
API_PATH = ROOT / "gateway" / "research_lab" / "api.py"


def main() -> int:
    sql = SQL_PATH.read_text(encoding="utf-8")
    api = API_PATH.read_text(encoding="utf-8")
    required_sql = {
        "guard_research_lab_queue_capacity",
        "guard_research_loop_queue_capacity_insert",
        "research_lab_queue_capacity_conflict",
        "research_lab_queue_hotkey_conflict",
        "pg_advisory_xact_lock",
        "current_queue_status IN ('queued', 'started')",
        "NEW.event_doc->>'autoresearch_capacity'",
    }
    required_api = {
        "_queue_capacity_doc(config)",
        '"autoresearch_capacity_policy": "proxy_worker_capacity:v1"',
        '"autoresearch_capacity": int(_autoresearch_loop_capacity(config))',
        '"active_loop_stale_after_seconds": max(60, int(config.active_loop_stale_after_seconds or 7200))',
        "research_lab_queue_capacity_conflict",
        "research_lab_queue_hotkey_conflict",
    }
    missing = [f"sql:{marker}" for marker in sorted(required_sql) if marker not in sql]
    missing.extend(f"api:{marker}" for marker in sorted(required_api) if marker not in api)
    if missing:
        for marker in missing:
            print(f"missing queue capacity guard marker: {marker}")
        return 1
    print("Research Lab queue capacity guard verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
