"""A lease-bound host failure cools execute claims only after normal expiry."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.lab_arena import expiry_runner_handoff390_postgres_test as score390
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).parents[2]
SQL390 = ROOT / "scripts/390-lab-arena-expired-retry-runner-handoff.sql"
SQL396 = ROOT / "scripts/396-lab-arena-execute-host-cooldown.sql"
LIVE_390_PREIMAGE = "0252fa3efafe895a7cade9169a36493dc8be70a9148a5ca037d0d0dde870a7c0"
LIVE_396_PREIMAGE = "f81a3cdb122caad413d38e4d24573b5d67fb07f480b79bf7ce3ec052419eb488"
SIGNATURE = "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)"


def _hash(cursor):
    cursor.execute(
        "SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
        "'sha256'),'hex')", (SIGNATURE,)
    )
    return cursor.fetchone()[0]


def _security(cursor):
    cursor.execute(
        "SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment' "
        "AND p.pronargs=9"
    )
    return cursor.fetchone()


@pytest.fixture(scope="module")
def database():
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "359-lab-arena-setup-failure-model-retry.sql",
        "360-lab-arena-zero-setup-runner-handoff.sql",
        "365-lab-arena-trajectories.sql",
        "366-lab-arena-trajectory-capacity.sql",
    )
    yield from database_with_lab_arena_migration(migrations)


@pytest.fixture(scope="module")
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql390 = SQL390.read_text().replace(LIVE_390_PREIMAGE, _hash(cursor))
            cursor.execute(sql390)
            before = _security(cursor)
            old_hash = _hash(cursor)
            sql396 = SQL396.read_text()
            assert LIVE_396_PREIMAGE in sql396
            sql396 = sql396.replace(LIVE_396_PREIMAGE, old_hash)
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql396.replace(old_hash, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert _hash(cursor) == old_hash
            cursor.execute(sql396)
            applied_hash = _hash(cursor)
            cursor.execute(sql396)
            assert _hash(cursor) == applied_hash
            assert _security(cursor) == before
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            definition = cursor.fetchone()[0]
            assert definition.count("lab_arena_execute_host_cooldown_v1") == 1
            assert definition.count("lab_arena_zero_call_score_runner_cooldown_v1") == 1
    return True


def _event(cursor, run_id, *, content=None):
    content = content or {
        "status": "abandoned",
        "failure_stage": "runtime",
        "error_class": "RuntimeHostError",
    }
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events("
        "run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
        "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,"
        "model_role,event_kind,occurred_at,content) "
        "SELECT r.run_id,gen_random_uuid(),r.round_id,r.submission_id,"
        "r.miner_hotkey,r.runner_hotkey,r.assignment_id,r.round_id||':icp:'||r.icp_position,"
        "r.stage,r.icp_position,r.attempt,r.kind,'baseline','runtime.error',"
        "now(),%s::jsonb FROM public.lab_arena_runs r WHERE r.run_id=%s",
        (json.dumps(content), run_id),
    )


def _seed(conn, *, positions=(0, 1, 2), age=1, generation=1, stage=1,
          round_status="stage1", error_content=None):
    fresh = score390._cooldown_seed(
        conn, failure_positions=positions, fresh_kind="execute",
        age_seconds=age, generation=generation, stage=stage,
        round_status=round_status,
    )
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_runs SET kind='execute' WHERE status='failed'")
        cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE status='failed' ORDER BY run_id")
        failures = [row[0] for row in cursor.fetchall()]
        for run_id in failures:
            _event(cursor, run_id, content=error_content)
        cursor.execute("SET session_replication_role=origin")
    conn.commit()
    return fresh, failures


@pytest.mark.parametrize("positions,blocked", [
    ((0,), False), ((0, 1), False), ((0, 1, 2), True),
    ((0, 0, 1), False),
])
def test_three_distinct_expired_host_errors_only(database, migrated, positions, blocked):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, _ = _seed(conn, positions=positions)
        answer = prior._claim(conn, prior.RUNNER_A, "a")
        assert (answer == {"status": "no_pending"}) is blocked
        if blocked:
            assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh
        else:
            assert answer["run_id"] == fresh


def test_score_cooldown_is_unchanged(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh = score390._cooldown_seed(conn)
        assert prior._claim(conn, prior.RUNNER_A, "2") == {"status": "no_pending"}
        assert prior._claim(conn, prior.RUNNER_B, "3")["run_id"] == fresh


def test_single_execute_expiry_keeps_lone_runner_retry(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        score390._leased_prior(conn, kind="execute")
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            _event(cursor, prior.ASSIGNMENT + ":1")
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_expire_leases(%s)", (prior.ROUND,))
            assert cursor.fetchone()[0]["retried"] == 1
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "4")["run_id"] == prior.ASSIGNMENT + ":2"


@pytest.mark.parametrize("change", [
    "no_event", "wrong_class", "wrong_stage", "wrong_status", "ledger",
    "result", "output", "not_expired", "old_ttl", "old_generation",
    "old_stage", "other_runner", "score_only", "wrong_cause",
])
def test_other_failures_leave_execute_claims_open(database, migrated, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        opts = {}
        if change == "old_ttl":
            opts["age"] = 421  # the fixture freezes lease_ttl_seconds=420
        elif change == "old_generation":
            opts["generation"] = 0
        elif change == "old_stage":
            opts["round_status"] = "stage2"
        fresh, failures = _seed(conn, **opts)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            run = failures[0]
            if change == "no_event":
                cursor.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=%s", (run,))
            elif change in ("wrong_class", "wrong_stage", "wrong_status"):
                key, value = {
                    "wrong_class": ("error_class", "OSError"),
                    "wrong_stage": ("failure_stage", "setup"),
                    "wrong_status": ("status", "failed"),
                }[change]
                cursor.execute(
                    "UPDATE public.lab_arena_trajectory_events SET content="
                    "jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE run_id=%s",
                    ([key], value, run),
                )
            elif change == "ledger":
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
                    "submission_id,run_id,stage,call_identity,provider,operation_id,"
                    "funding_source,amount_microusd) VALUES "
                    "('reservation',%s,%s,%s,%s,1,%s,'deepline','deepline.execute','host',1)",
                    (prior.MINER, prior.ROUND, prior.SUBMISSION, run, "sha256:" + "d" * 64),
                )
            elif change == "result":
                cursor.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb WHERE run_id=%s", (run,))
            elif change == "output":
                cursor.execute("UPDATE public.lab_arena_runs SET output_ref='present' WHERE run_id=%s", (run,))
            elif change == "not_expired":
                cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()+interval '1 minute' WHERE run_id=%s", (run,))
            elif change == "other_runner":
                cursor.execute("UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s", (prior.RUNNER_B, run))
            elif change == "score_only":
                cursor.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s", (run,))
            elif change == "wrong_cause":
                cursor.execute("UPDATE public.lab_arena_runs SET terminal_cause='model_error' WHERE run_id=%s", (run,))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "c")["run_id"] == fresh


def test_normal_expiry_and_lone_runner_recover_after_ttl(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, failures = _seed(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=ANY(%s)", (failures,))
            cursor.execute("UPDATE public.lab_arena_runs SET status='leased',"
                           "terminal_cause=NULL,lease_token_hash=%s,"
                           "lease_expires_at=now()+interval '1 minute' "
                           "WHERE run_id=ANY(%s)", ("sha256:" + "a" * 64, failures))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT has_table_privilege('lab_arena_service',"
                "'public.lab_arena_trajectory_events','INSERT')"
            )
            assert cursor.fetchone()[0] is False
            cursor.execute("SET ROLE lab_arena_service")
            for run in failures:
                event = {
                    "event_id": str(uuid.uuid4()),
                    "kind": "runtime.error",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "content": {
                        "status": "abandoned", "failure_stage": "runtime",
                        "error_class": "RuntimeHostError",
                    },
                }
                cursor.execute(
                    "SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",
                    (run, "sha256:" + "b" * 64, json.dumps([event])),
                )
                assert cursor.fetchone()[0]["status"] == "stale"
                cursor.execute(
                    "SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",
                    (run, "sha256:" + "a" * 64, json.dumps([event])),
                )
                assert cursor.fetchone()[0]["status"] == "accepted"
            cursor.execute("RESET ROLE")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' "
                           "WHERE run_id=ANY(%s)", (failures,))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_expire_leases(%s)", (prior.ROUND,))
            assert cursor.fetchone()[0]["retried"] == 3
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "d") == {"status": "no_pending"}
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '421 seconds' "
                           "WHERE run_id=ANY(%s)", (failures,))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "e")["status"] == "leased"


def test_2560_pending_jobs_keep_claim_bounded(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        _seed(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "lease_generation,stage_generation) "
                "SELECT 'load-'||i,'load-'||i,%s,%s,%s,1,i%%10,1,'execute',"
                "'pending',0,1 FROM generate_series(1,2560) AS i",
                (prior.ROUND, prior.SUBMISSION, prior.MINER),
            )
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_A, "f") == {"status": "no_pending"}
        blocked_seconds = time.monotonic() - start
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_B, "1")["status"] == "leased"
        healthy_seconds = time.monotonic() - start
        assert blocked_seconds < 5.0
        assert healthy_seconds < 5.0
        print(f"2560-job claim seconds: cooled={blocked_seconds:.3f}, healthy={healthy_seconds:.3f}")
