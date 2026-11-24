"""A proved installer fault must survive the same-runner fallback on a retry."""

import json
from pathlib import Path
import pytest

from tests.lab_arena import installer_capacity710_postgres_test as setup
from tests.lab_arena import paid_expiry_capacity713_postgres_test as previous
from tests.lab_arena import stage_cutoff_drain420_postgres_test as jobs

SQL = Path(__file__).parents[2] / "scripts/738-lab-arena-installer-retry-assignment.sql"


@pytest.fixture(scope="module")
def database():
    for pg, dsn in previous.database.__wrapped__():
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute(SQL.read_text())
            q.execute(SQL.read_text())
        yield pg, dsn


@pytest.mark.parametrize("proved", [True, False])
def test_installer_failure_on_single_assignment(database, proved):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        jobs._seed(conn, positions=2, current_schema=True)
        first = setup.claim(conn, jobs.RUNNER, "a")
        event = dict(
            status="provider_error",
            failure_stage="setup",
            error_class="DependencyInstallInfrastructureError",
            failure_kind="installer_error",
            diagnostic_detail=setup.MESSAGE if proved else "different cause",
        )
        assert (
            setup.event(conn, first, "runtime.error", event, token="a")["status"]
            == "accepted"
        )
        # Extra legitimate recorder fields make the old whole-object zero test fail.
        result = dict(
            terminal_status="provider_error",
            failure_diagnostic=dict(
                stage="provider_call",
                error_class="provider_unavailable",
                reason="provider_error",
            ),
            resource_summary=dict(
                wall_seconds=0,
                cpu_seconds=0,
                max_rss_bytes=0,
                stdout_bytes=0,
                stderr_bytes=0,
                provider_call_count=0,
                worker_observations={"status": "stored"},
            ),
        )
        assert (
            jobs._rpc(
                conn,
                "SELECT lab_arena_complete_attempt(%s,%s,%s::jsonb,'provider_error',NULL)",
                (first["run_id"], "sha256:" + "a" * 64, json.dumps(result)),
            )["status"]
            == "failed"
        )
        # This runner is the only recently active host. A genuine fallback can
        # claim its own retry for other errors, but never for a proved installer miss.
        second = setup.claim(conn, jobs.RUNNER, "b")
        assert second["status"] == "leased"
        if proved:
            assert second["assignment_id"] != first["assignment_id"]
            assert setup.claim(conn, jobs.RUNNER, "c")["status"] == "no_pending"
            alternate = setup.claim(conn, setup.HEALTHY, "d")
            assert (
                alternate["assignment_id"] == first["assignment_id"]
                and alternate["attempt"] == 2
            )
            # Replay the surviving lease even after matching a prior failed attempt.
            assert setup.claim(conn, setup.HEALTHY, "d") == alternate
            assert jobs._complete(conn, alternate, token="d")["status"] == "accepted"
        else:
            assert (
                second["assignment_id"] == first["assignment_id"]
                and second["attempt"] == 2
            )


def test_round_wide_quarantine_and_existing_lease_still_work(database):
    setup.test_native_claim_after_setup_failures(database, "missing", True)


def test_paid_expiry_cooldown_remains_round_wide(database):
    previous.test_native_claim_after_paid_batch_loss(database, "paid", True)
