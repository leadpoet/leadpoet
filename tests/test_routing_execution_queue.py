from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "159-research-lab-routing-execution-queue.sql"
BEHAVIOR = ROOT / "tests" / "sql" / "test_routing_execution_queue_v2.sql"


def test_execution_queue_migration_is_additive_and_uses_skip_locked():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.research_lab_routing_execution_request_leases_v2" in sql
    assert "FOR UPDATE OF request SKIP LOCKED" in sql
    assert "research_lab_routing_claim_execution_requests_v2" in sql
    assert "research_lab_routing_renew_execution_request_lease_v2" in sql
    assert "research_lab_routing_close_execution_request_lease_v2" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "REVOKE ALL ON TABLE public.research_lab_routing_execution_request_leases_v2" in sql
    assert "GRANT SELECT ON TABLE public.research_lab_routing_execution_request_leases_v2" in sql
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_execution_requests_v2" in sql
    assert "p_claim_nonce" not in sql
    assert "p_provider" not in sql


def test_execution_queue_migration_has_generation_and_terminal_fence():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "lease_generation BIGINT NOT NULL CHECK (lease_generation > 0)" in sql
    assert "lease_state IN ('claimed', 'completed', 'failed', 'recovered')" in sql
    assert "lease_generation = p_lease_generation" in sql
    assert "lease_state = 'claimed'" in sql
    assert "lease_expires_at > pg_catalog.clock_timestamp()" in sql
    assert "'stale', TRUE" in sql


def test_execution_queue_binds_the_product_claim_once_to_the_active_lease():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "execution_claim_key TEXT" in sql
    assert "execution_claim_generation BIGINT" in sql
    assert "research_lab_routing_claim_execution_v3" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "lease_row.lease_state IS DISTINCT FROM 'claimed'" in sql
    assert "execution_claim_key IS NULL" in sql
    assert "research_lab_routing_claim_experiment_v3" in sql
    assert "GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_execution_v3" in sql


def test_execution_queue_terminally_recovers_an_expired_bound_claim():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "lease.execution_claim_key IS NOT NULL" in sql
    assert "research_lab_routing_recover_claim_v3" in sql
    assert "leadpoet.research_lab.routing_queue_recovery.v3" in sql
    assert "queue_lease_expired" in sql
    assert "lease.execution_claim_key IS NULL" in sql

    behavior = BEHAVIOR.read_text(encoding="utf-8")
    assert "live claim with stale queue lease passed the fence" in behavior
    assert "bound queue request was reclaimed after expiry" in behavior
    assert "open reservation was not retained at full uncertain ceiling" in behavior
    assert "exact terminal recovery replay was not idempotent" in behavior
    assert "leadpoet.research_lab.routing_budget_reservation_result.v3" in behavior


@pytest.mark.skipif(
    not os.getenv("ROUTING_EXPERIMENT_TEST_PG_DSN"),
    reason="set ROUTING_EXPERIMENT_TEST_PG_DSN for disposable PostgreSQL behavior test",
)
def test_execution_queue_disposable_postgres_behavior():
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is unavailable")
    dsn = os.environ["ROUTING_EXPERIMENT_TEST_PG_DSN"]
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
