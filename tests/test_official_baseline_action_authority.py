from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "163-research-lab-official-baseline-action-authority.sql"
)
BEHAVIOR = (
    ROOT / "tests" / "sql" / "test_official_baseline_action_authority_v1.sql"
)


def test_official_baseline_action_authority_is_closed_and_append_only():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "action_sequence             INTEGER NOT NULL" in sql
    assert "UNIQUE (run_sha256, unit_ref, action_sequence)" in sql
    assert "official_baseline_provider_frontier.v1" in sql
    assert "ORDER BY attempt.action_sequence" in sql
    assert "terminal_uncertain" in sql
    assert "research_lab_official_baseline_unit_actions_incomplete" in sql
    assert "BEFORE UPDATE OR DELETE" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "TO service_role;" in sql
    assert "ON CONFLICT DO UPDATE" not in sql


def test_official_baseline_behavior_covers_replay_zero_call_and_uncertainty():
    sql = BEHAVIOR.read_text(encoding="utf-8")
    assert "reserved_existing" in sql
    assert "'action_type', 'verify_company'" in sql
    assert "'call_count', 0" in sql
    assert "'provider_request_ref', NULL" in sql
    assert "unit closure replay was not idempotent" in sql
    assert "uncertain provider call was redispatched" in sql
    assert "append-only attempt update unexpectedly succeeded" in sql


@pytest.mark.skipif(
    not os.getenv("OFFICIAL_BASELINE_TEST_PG_DSN"),
    reason="set OFFICIAL_BASELINE_TEST_PG_DSN for disposable PostgreSQL behavior test",
)
def test_official_baseline_disposable_postgres_behavior():
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is unavailable")
    dsn = os.environ["OFFICIAL_BASELINE_TEST_PG_DSN"]
    migration = subprocess.run(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert migration.returncode == 0, migration.stderr
    behavior = subprocess.run(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-f", str(BEHAVIOR)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert behavior.returncode == 0, behavior.stderr
