#!/usr/bin/env python3
"""Verify Research Lab queue capacity guard migration markers."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "scripts" / "43-research-lab-queue-capacity-guard.sql"
RESUME_SQL_PATH = ROOT / "scripts" / "54-research-lab-resume-requeue-hotkey-guard.sql"
API_PATH = ROOT / "gateway" / "research_lab" / "api.py"


def main() -> int:
    sql = SQL_PATH.read_text(encoding="utf-8")
    resume_sql = RESUME_SQL_PATH.read_text(encoding="utf-8")
    api = API_PATH.read_text(encoding="utf-8")
    required_sql = {
        "guard_research_lab_queue_capacity",
        "guard_research_loop_queue_capacity_insert",
        "research_lab_queue_capacity_conflict",
        "research_lab_queue_hotkey_conflict",
        "pg_advisory_xact_lock",
        "current_queue_status IN ('queued', 'started', 'paused')",
        "is_existing_run_requeue BOOLEAN",
        "q.run_id <> NEW.run_id",
        "NEW.event_doc ? 'resume_source'",
        "q.current_queue_status IN ('queued', 'started')",
        "(q.current_queue_status = 'paused' OR q.current_status_at >= cutoff)",
        "NEW.event_doc->>'autoresearch_capacity'",
        "v_miner_hotkey TEXT",
        "btrim(v_miner_hotkey)",
    }
    required_api = {
        "_queue_capacity_doc(config)",
        '"autoresearch_capacity_policy": "proxy_worker_capacity:v1"',
        '"autoresearch_capacity": int(_autoresearch_loop_capacity(config))',
        "DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS",
        "research_lab_queue_capacity_conflict",
        "research_lab_queue_hotkey_conflict",
    }
    missing = [f"sql:{marker}" for marker in sorted(required_sql) if marker not in sql]
    required_resume_sql = {
        "maintenance requeues ignore paused rows",
        "same_hotkey_count",
        "is_existing_run_requeue",
        "q.current_queue_status IN ('queued', 'started')",
        "NOT is_existing_run_requeue",
        "q.current_queue_status IN ('queued', 'started', 'paused')",
        "research_lab_queue_hotkey_conflict",
        "research_lab_queue_capacity_conflict",
    }
    missing.extend(
        f"resume_sql:{marker}" for marker in sorted(required_resume_sql) if marker not in resume_sql
    )
    same_hotkey_match = re.search(
        r"SELECT COUNT\(\*\)\s+INTO same_hotkey_count(?P<body>.*?)IF same_hotkey_count > 0",
        resume_sql,
        flags=re.S,
    )
    if not same_hotkey_match:
        missing.append("resume_sql:same_hotkey_count query block")
    else:
        same_hotkey_body = same_hotkey_match.group("body")
        for marker in (
            "is_existing_run_requeue",
            "q.current_queue_status IN ('queued', 'started')",
            "NOT is_existing_run_requeue",
            "q.current_queue_status IN ('queued', 'started', 'paused')",
        ):
            if marker not in same_hotkey_body:
                missing.append(f"resume_sql:same_hotkey:{marker}")
    missing.extend(f"api:{marker}" for marker in sorted(required_api) if marker not in api)
    if missing:
        for marker in missing:
            print(f"missing queue capacity guard marker: {marker}")
        return 1
    if re.search(r"(?m)^\s+miner_hotkey\s+TEXT;", sql):
        print("ambiguous PL/pgSQL variable name remains: miner_hotkey TEXT")
        return 1
    print("Research Lab queue capacity guard verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
