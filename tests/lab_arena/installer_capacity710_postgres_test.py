"""A validator missing the sandbox installer can still finish leases and judge."""

from pathlib import Path
import json

import pytest

from tests.lab_arena import accepted_execution_correction703_postgres_test as schema
from tests.lab_arena import stage_cutoff_drain420_postgres_test as jobs
from tests.lab_arena.host_interruption_retry474_postgres_test import event
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/710-lab-arena-incompatible-installer-capacity.sql"
)
MESSAGE = "compatible dependency installer unavailable: CPython 3.11 required"
HEALTHY = hotkey("installer-capacity-healthy")


@pytest.fixture(scope="module")
def database():
    for pg, dsn in schema.database.__wrapped__():
        with pg.connect(**dsn) as conn, conn.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
        yield pg, dsn


def claim(conn, runner, suffix):
    return jobs._rpc(
        conn,
        "SELECT lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)",
        (
            jobs.ROUND,
            runner,
            suffix * 32,
            "sha256:" + suffix * 64,
            "sha256:" + suffix * 64,
        ),
    )


@pytest.mark.parametrize(
    "signal,blocked",
    [("missing", True), ("two", False), ("network", False), ("paid", False)],
)
def test_native_claim_after_setup_failures(database, signal, blocked):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        jobs._seed(conn, positions=7, current_schema=True)
        leases = [claim(conn, jobs.RUNNER, token) for token in "abcd"]
        assert all(row["status"] == "leased" for row in leases)
        # Leave the fourth lease active to verify exact replay and completion.
        for i, (lease, token) in enumerate(zip(leases[:3], "abc")):
            diagnostic = dict(
                status="provider_error",
                failure_stage="setup",
                error_class="DependencyInstallInfrastructureError",
                failure_kind="installer_error",
                diagnostic_detail=MESSAGE,
            )
            if i == 2 and signal in ("two", "network"):
                diagnostic["diagnostic_detail"] = "temporary installation error"
                if signal == "network":
                    diagnostic["failure_kind"] = "network_error"
            assert (
                event(conn, lease, "runtime.error", diagnostic, token=token)["status"]
                == "accepted"
            )
            if i == 2 and signal == "paid":
                with conn.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,
                        run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
                        SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                        'sha256:'||repeat('9',64),'openrouter','openrouter.chat','host',100,'{}'
                        FROM lab_arena_runs WHERE run_id=%s""",
                        (lease["run_id"],),
                    )
                conn.commit()
            result = {
                "terminal_status": "provider_error",
                "resource_summary": {"provider_call_count": 0, "wall_seconds": 0},
            }
            done = jobs._rpc(
                conn,
                "SELECT lab_arena_complete_attempt(%s,%s,%s::jsonb,'provider_error',NULL)",
                (lease["run_id"], "sha256:" + token * 64, json.dumps(result)),
            )
            assert done["status"] == "failed"
        # A proven missing installer affects only fresh execution claims.
        assert claim(conn, jobs.RUNNER, "d") == leases[3]
        assert claim(conn, jobs.RUNNER, "e")["status"] == (
            "no_pending" if blocked else "leased"
        )
        assert claim(conn, HEALTHY, "f")["status"] == "leased"
        assert jobs._complete(conn, leases[3], token="d")["status"] == "accepted"

        if blocked:
            # In a disposable snapshot, move to judging the accepted source.
            # Pending execution demand remains present, so the execution-only
            # quarantine really is active when this runner claims the judge.
            with conn.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    """UPDATE lab_arena_rounds SET status='stage2_scoring',
                    configuration_doc=configuration_doc || jsonb_build_object('schedule',
                    configuration_doc->'schedule' || jsonb_build_object('final_scoring_close',
                    (now()+interval '2 hours')::text,'publication_deadline',(now()+interval '3 hours')::text))
                    WHERE round_id=%s""",
                    (jobs.ROUND,),
                )
                cursor.execute(
                    """INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
                    stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,scored_run_id,
                    judgment_group_leader,judgment_group_miner_hotkeys)
                    SELECT assignment_id||':score:1',assignment_id||':score',round_id,submission_id,miner_hotkey,
                    stage,icp_position,1,'score','pending',stage_generation,0,run_id,true,ARRAY[miner_hotkey]
                    FROM lab_arena_runs WHERE run_id=%s""",
                    (leases[3]["run_id"],),
                )
                cursor.execute("SET LOCAL session_replication_role=origin")
            conn.commit()
            judge = claim(conn, jobs.RUNNER, "7")
            assert (
                judge["status"] == "leased"
                and judge["kind"] == "score"
                and judge["scored_run_id"] == leases[3]["run_id"]
            )
