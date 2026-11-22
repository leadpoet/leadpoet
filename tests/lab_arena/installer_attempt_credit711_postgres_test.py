"""A zero-cost host setup failure must preserve the miner's normal attempts."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import installer_capacity710_postgres_test as capacity
from tests.lab_arena import stage_cutoff_drain420_postgres_test as jobs
from tests.lab_arena.host_interruption_retry474_postgres_test import event

MIGRATION = Path(__file__).resolve().parents[2] / "scripts/711-lab-arena-installer-attempt-credit.sql"


@pytest.fixture
def database():
    yield from capacity.database.__wrapped__()


def setup_failure(conn, lease, token, *, message=capacity.MESSAGE):
    detail = dict(status="provider_error", failure_stage="setup",
                  error_class="DependencyInstallInfrastructureError", failure_kind="installer_error",
                  diagnostic_detail=message)
    assert event(conn, lease, "runtime.error", detail, token=token)["status"] == "accepted"
    result = {"terminal_status": "provider_error", "resource_summary": {"provider_call_count": 0, "wall_seconds": 0}}
    return jobs._rpc(conn, "SELECT lab_arena_complete_attempt(%s,%s,%s::jsonb,'provider_error',NULL)",
                     (lease["run_id"], "sha256:" + token * 64, json.dumps(result)))


def seed(conn, date="2026-11-21", policy="trusted_infrastructure_v1"):
    jobs._seed(conn, positions=1, current_schema=True)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("""UPDATE lab_arena_rounds SET evaluation_date=%s,
            configuration_doc=jsonb_set(configuration_doc,'{infrastructure_retry_cost_policy}',to_jsonb(%s::text))
            WHERE round_id=%s""", (date, policy, jobs.ROUND))
    conn.commit()


def count(conn, lease):
    return jobs._rpc(conn, "SELECT lab_arena_execution_attempt_count(r) FROM lab_arena_runs r WHERE run_id=%s",
                     (lease["run_id"],))


def migrate(conn):
    with conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
    conn.commit()


def test_upgrade_recovers_previously_exhausted_assignment(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        seed(conn)
        first = capacity.claim(conn, jobs.RUNNER, "a")
        assert setup_failure(conn, first, "a")["status"] == "failed"
        second = capacity.claim(conn, capacity.HEALTHY, "b")
        assert second["attempt"] == 2
        assert jobs._complete(conn, second, "provider_error", "b")["status"] == "failed"
        assert count(conn, second) == 2
        assert capacity.claim(conn, capacity.HEALTHY, "c")["status"] == "no_pending"
        migrate(conn)
        assert count(conn, second) == 1
        for _ in range(2):
            jobs._rpc(conn, "SELECT lab_arena_expire_leases(%s)", (jobs.ROUND,))
        third = capacity.claim(conn, capacity.HEALTHY, "c")
        assert third["attempt"] == 3
        assert jobs._complete(conn, third, token="c")["status"] == "accepted"
        assert count(conn, third) == 2
        assert capacity.claim(conn, capacity.HEALTHY, "d")["status"] == "no_pending"


@pytest.mark.parametrize("date,policy,credit", [
    ("2026-11-21", "trusted_infrastructure_v1", 1),
    ("2026-11-20", "trusted_infrastructure_v1", 0),
    ("2026-11-21", "untrusted", 0),
])
def test_credit_is_bounded_and_policy_scoped(database, date, policy, credit):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        seed(conn, date, policy)
        migrate(conn)
        for index, token in enumerate("ab", 1):
            lease = capacity.claim(conn, jobs.RUNNER, token)
            assert lease["attempt"] == index
            setup_failure(conn, lease, token)
            assert count(conn, lease) == index - credit


@pytest.mark.parametrize("evidence", ["ordinary_package_error", "ledger", "model_activity"])
def test_ambiguous_or_paid_setup_is_still_counted(database, evidence):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        seed(conn)
        migrate(conn)
        lease = capacity.claim(conn, jobs.RUNNER, "a")
        message = capacity.MESSAGE
        if evidence == "ordinary_package_error":
            message = "a submitted dependency could not be installed"
        elif evidence == "ledger":
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,
                    run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
                    SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                    'sha256:'||repeat('9',64),'openrouter','openrouter.chat','host',100,'{}'
                    FROM lab_arena_runs WHERE run_id=%s""", (lease["run_id"],))
            conn.commit()
        else:
            assert event(conn, lease, "provider.response", {"status": "returned"}, token="a")["status"] == "accepted"
        setup_failure(conn, lease, "a", message=message)
        assert count(conn, lease) == 1
