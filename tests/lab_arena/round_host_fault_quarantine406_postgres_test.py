"""An authenticated zero-call host fault quarantines only its round."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.lab_arena import execute_host_cooldown396_postgres_test as prior396
from tests.lab_arena import active_host_fault_guard397_postgres_test as prior397
from tests.lab_arena import untransitioned_host_fault_guard399_postgres_test as prior399
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior


SQL406 = Path(__file__).parents[2] / "scripts/406-lab-arena-round-host-fault-quarantine.sql"
LIVE_PREIMAGE = "f1f82a1dc510e9339ad76e0b08ad29d8dad3d8194d99700603d359b5d85ac7f7"
database = prior399.database


@pytest.fixture(scope="module")
def migrated(database):
    prior399.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            before = prior396._security(cursor)
            local_hash = prior396._hash(cursor)
            sql = SQL406.read_text()
            assert LIVE_PREIMAGE in sql
            sql = sql.replace(LIVE_PREIMAGE, local_hash)
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(local_hash, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert prior396._hash(cursor) == local_hash
            cursor.execute(sql)
            applied_hash = prior396._hash(cursor)
            cursor.execute(sql)
            assert prior396._hash(cursor) == applied_hash
            assert prior396._security(cursor) == before
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)",
                           (prior396.SIGNATURE,))
            definition = cursor.fetchone()[0]
            for marker in (
                "lab_arena_score_submission_serialization",
                "judgment_group_leader",
                "lab_arena_active_host_fault_claim_guard_v1",
                "lab_arena_untransitioned_host_fault_guard_v1",
                "lab_arena_round_host_fault_quarantine_v1",
            ):
                assert marker in definition
            assert definition.count("lab_arena_round_host_fault_quarantine_v1") == 1
    return True


def _old_faults(conn, faults, *, age_seconds=7200, transition=True):
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET lease_expires_at="
            "now()-(%s*interval '1 second')"
            + (",status='failed',terminal_cause='lease_expired'" if transition else "")
            + " WHERE run_id=ANY(%s)",
            (age_seconds, faults),
        )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("transition", [False, True])
def test_old_host_faults_block_same_round_but_not_healthy_runner(
    database, migrated, kind, transition,
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind=kind)
        _old_faults(conn, faults, transition=transition)
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh


def test_first_stage_execute_faults_block_later_score_claim(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn, kind="score")
        _old_faults(conn, faults)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET kind='execute',"
                           "stage_generation=0 WHERE run_id=ANY(%s)", (faults,))
            cursor.execute("UPDATE public.lab_arena_trajectory_events SET "
                           "run_kind='execute' WHERE run_id=ANY(%s)", (faults,))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh


@pytest.mark.parametrize("change", [
    "two_faults", "wrong_class", "no_event", "ledger", "result", "output",
    "other_runner", "other_round",
])
def test_only_exact_current_round_host_faults_quarantine(database, migrated, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn)
        _old_faults(conn, faults)
        run = faults[0]
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            if change == "two_faults":
                cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (run,))
            elif change == "wrong_class":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET "
                               "content=jsonb_set(content,'{error_class}',"
                               "'\"OSError\"'::jsonb) WHERE run_id=%s", (run,))
            elif change == "no_event":
                cursor.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=%s", (run,))
            elif change == "ledger":
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
                    "submission_id,run_id,stage,call_identity,provider,operation_id,"
                    "funding_source,amount_microusd) VALUES "
                    "('reservation',%s,%s,%s,%s,1,%s,'deepline','deepline.execute','host',1)",
                    (prior.MINER, prior.ROUND, prior.SUBMISSION, run, "sha256:" + "d" * 64),
                )
            elif change == "result":
                cursor.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                               "WHERE run_id=%s", (run,))
            elif change == "output":
                cursor.execute("UPDATE public.lab_arena_runs SET output_ref='present' "
                               "WHERE run_id=%s", (run,))
            elif change == "other_runner":
                cursor.execute("UPDATE public.lab_arena_runs SET runner_hotkey=%s "
                               "WHERE run_id=%s", (prior.RUNNER_B, run))
            elif change == "other_round":
                cursor.execute("UPDATE public.lab_arena_runs SET round_id=%s "
                               "WHERE run_id=ANY(%s)", ("earlier-round", faults))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "a")["run_id"] == fresh


def test_large_pending_set_keeps_quarantine_claim_bounded(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = prior397._seed(conn)
        _old_faults(conn, faults)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "lease_generation,stage_generation) "
                "SELECT 'round-load-'||i,'round-load-'||i,%s,%s,%s,1,i%%10,1,"
                "'execute','pending',0,1 FROM generate_series(1,2560) AS i",
                (prior.ROUND, prior.SUBMISSION, prior.MINER),
            )
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        blocked_seconds = time.monotonic() - start
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_B, "b")["status"] == "leased"
        healthy_seconds = time.monotonic() - start
        # The unchanged claim scan takes about 10 seconds for a healthy
        # runner at 2,560 synthetic jobs in this local Postgres harness.
        assert blocked_seconds < 5.0 and healthy_seconds < 12.0
