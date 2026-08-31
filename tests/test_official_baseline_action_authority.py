from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "164-research-lab-official-baseline-action-authority.sql"
)
TIMEOUT_MIGRATION = (
    ROOT / "scripts" / "166-research-lab-zero-call-verifier-timeout.sql"
)
REQUEST_SCOPE_MIGRATION = (
    ROOT / "scripts" / "167-research-lab-provider-request-attempt-scope.sql"
)
BEHAVIOR = (
    ROOT / "tests" / "sql" / "test_official_baseline_action_authority_v1.sql"
)
SUPABASE_SETUP_SQL = """
DO $$ BEGIN CREATE ROLE anon NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE authenticated NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE service_role NOLOGIN BYPASSRLS; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
CREATE OR REPLACE FUNCTION public.research_lab_routing_jsonb_hash_v2(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $jsonb_hash$
    SELECT 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_value::TEXT, 'UTF8'),
            'sha256'
        ),
        'hex'
    )
$jsonb_hash$;
"""


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
    assert "zero-timeout provider reservation unexpectedly succeeded" in sql
    assert "positive-timeout verifier reservation unexpectedly succeeded" in sql
    assert "'provider_request_ref', NULL" in sql
    assert "failed verifier replay custody invalid" in sql
    assert "unit closure replay was not idempotent" in sql
    assert "uncertain provider call was redispatched" in sql
    assert "append-only attempt update unexpectedly succeeded" in sql


def test_zero_call_verifier_timeout_patch_preserves_paid_provider_bounds():
    sql = TIMEOUT_MIGRATION.read_text(encoding="utf-8")
    assert "research_lab_official_baseline_action_attempts_timeout_ms_check" in sql
    assert "timeout_ms = 0" in sql
    assert "timeout_ms BETWEEN 1 AND 900000" in sql
    assert "research_lab_official_baseline_reserve_action_v1" in sql


def test_provider_request_scope_patch_is_retry_safe_and_conflict_closed():
    sql = REQUEST_SCOPE_MIGRATION.read_text(encoding="utf-8")
    assert "pg_advisory_xact_lock" in sql
    assert "^provider_request:[0-9a-f]{64}$" in sql
    assert "prior_attempt.run_sha256 IS DISTINCT FROM current_attempt.run_sha256" in sql
    assert "prior_attempt.unit_ref IS NOT DISTINCT FROM current_attempt.unit_ref" in sql
    assert "model_provider_response_sha256 IS DISTINCT FROM" in sql
    assert "research_lab_official_baseline_provider_request_replay_conflict" in sql
    assert "idx_rl_official_baseline_provider_request_v2" in sql
    assert "research_lab_official_baseline_request_scope_v2" in sql

    behavior = BEHAVIOR.read_text(encoding="utf-8")
    assert "legacy provider retry replay was not accepted" in behavior
    assert "conflicting legacy provider replay unexpectedly succeeded" in behavior


@pytest.mark.skipif(
    not os.getenv("OFFICIAL_BASELINE_TEST_PG_DSN"),
    reason="set OFFICIAL_BASELINE_TEST_PG_DSN for disposable PostgreSQL behavior test",
)
def test_official_baseline_disposable_postgres_behavior():
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is unavailable")
    dsn = os.environ["OFFICIAL_BASELINE_TEST_PG_DSN"]
    setup = subprocess.run(
        [psql, dsn, "-v", "ON_ERROR_STOP=1"],
        input=SUPABASE_SETUP_SQL,
        text=True,
        capture_output=True,
        check=False,
    )
    assert setup.returncode == 0, setup.stderr
    for path in (
        MIGRATION,
        TIMEOUT_MIGRATION,
        REQUEST_SCOPE_MIGRATION,
        TIMEOUT_MIGRATION,
        REQUEST_SCOPE_MIGRATION,
    ):
        migration = subprocess.run(
            [psql, dsn, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
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
