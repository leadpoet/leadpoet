"""Three authenticated host faults on active leases stop further claims."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.lab_arena import execute_host_cooldown396_postgres_test as expired396
from tests.lab_arena import expiry_runner_handoff390_postgres_test as score390
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior


ROOT = Path(__file__).parents[2]
SQL397 = ROOT / "scripts/397-lab-arena-active-host-fault-claim-guard.sql"
LIVE_PREIMAGE = "755819d63ea1ae5f330552fd90fbf6905cbd85b6611e1b26e092ba80fb85ebcc"
database = expired396.database


@pytest.fixture(scope="module")
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for path, expected in (
                (expired396.SQL390, expired396.LIVE_390_PREIMAGE),
                (expired396.SQL396, expired396.LIVE_396_PREIMAGE),
                (SQL397, LIVE_PREIMAGE),
            ):
                before = expired396._security(cursor)
                old_hash = expired396._hash(cursor)
                sql = path.read_text()
                assert expected in sql
                sql = sql.replace(expected, old_hash)
                cursor.execute("BEGIN")
                with pytest.raises(psycopg.Error, match="preimage differs"):
                    cursor.execute(sql.replace(old_hash, "0" * 64, 1))
                cursor.execute("ROLLBACK")
                assert expired396._hash(cursor) == old_hash
                cursor.execute(sql)
                applied_hash = expired396._hash(cursor)
                cursor.execute(sql)
                assert expired396._hash(cursor) == applied_hash
                assert expired396._security(cursor) == before
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (expired396.SIGNATURE,))
            definition = cursor.fetchone()[0]
            for marker in (
                "lab_arena_zero_call_score_runner_cooldown_v1",
                "lab_arena_execute_host_cooldown_v1",
                "lab_arena_active_host_fault_claim_guard_v1",
            ):
                assert definition.count(marker) == 1
    return True


def _seed(conn, *, positions=(0, 1, 2), kind="execute", generation=1,
          round_status=None, with_events=True):
    round_status = round_status or ("stage1" if kind == "execute" else "stage1_scoring")
    fresh = score390._cooldown_seed(
        conn, failure_positions=positions, fresh_kind=kind,
        round_status=round_status,
    )
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='leased',kind=%s,"
            "terminal_cause=NULL,stage_generation=%s,lease_token_hash=%s,"
            "lease_expires_at=now()+interval '10 minutes' "
            "WHERE status='failed'",
            (kind, generation, "sha256:" + "a" * 64),
        )
        cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE status='leased' ORDER BY run_id")
        faults = [row[0] for row in cursor.fetchall()]
        if kind == "score":
            # Score claims serialize per submission. Use distinct frozen miners
            # so these leases model three concurrent validator assignments.
            for index, run in enumerate(faults):
                submission = f"fault-score-sub-{index}"
                miner = "5" + chr(ord("D") + index) * 47
                source = f"arena/{prior.ROUND}/sources/{submission}.tar.gz"
                cursor.execute(
                    "INSERT INTO public.lab_arena_submissions("
                    "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                    "source_ref,source_size_bytes,consent,frozen_at) "
                    "SELECT %s,round_id,%s,status,FALSE,"
                    "jsonb_set(submission_doc,'{source_ref}',to_jsonb(%s::text)),"
                    "%s,source_size_bytes,consent,frozen_at "
                    "FROM public.lab_arena_submissions WHERE submission_id=%s",
                    (submission, miner, source, source, prior.SUBMISSION),
                )
                cursor.execute("UPDATE public.lab_arena_runs SET submission_id=%s,"
                               "miner_hotkey=%s WHERE run_id=%s", (submission, miner, run))
        if with_events:
            for run in faults:
                expired396._event(cursor, run)
            if kind == "score":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET model_role='miner' "
                               "WHERE run_id=ANY(%s)", (faults,))
        cursor.execute("SET session_replication_role=origin")
    conn.commit()
    return fresh, faults


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("positions,blocked", [
    ((0,), False), ((0, 1), False), ((0, 1, 2), True),
])
def test_active_threshold_distinct_and_healthy_handoff(database, migrated, kind, positions, blocked):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = _seed(conn, positions=positions, kind=kind)
        answer = prior._claim(conn, prior.RUNNER_A, "a")
        assert (answer == {"status": "no_pending"}) is blocked
        if blocked:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM public.lab_arena_runs "
                               "WHERE run_id=ANY(%s) AND status='leased' "
                               "AND lease_expires_at>now()", (faults,))
                assert cursor.fetchone()[0] == len(faults)
            assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh
        else:
            assert answer["run_id"] == fresh


@pytest.mark.parametrize("change", [
    "no_event", "wrong_class", "wrong_stage_code", "wrong_status",
    "wrong_event_run", "wrong_event_runner", "wrong_event_kind",
    "ledger", "result", "output",
    "expired", "old_generation", "old_stage", "other_round", "other_kind",
    "other_runner",
])
def test_unqualified_active_faults_do_not_block(database, migrated, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = _seed(conn)
        run = faults[0]
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            if change == "no_event":
                cursor.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=%s", (run,))
            elif change in ("wrong_class", "wrong_stage_code", "wrong_status"):
                key, value = {
                    "wrong_class": ("error_class", "OSError"),
                    "wrong_stage_code": ("failure_stage", "setup"),
                    "wrong_status": ("status", "failed"),
                }[change]
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET content="
                               "jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE run_id=%s",
                               ([key], value, run))
            elif change == "wrong_event_runner":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s "
                               "WHERE run_id=%s", (prior.RUNNER_B, run))
            elif change == "wrong_event_run":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET run_id=%s "
                               "WHERE run_id=%s", (fresh, run))
            elif change == "wrong_event_kind":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET run_kind='score' "
                               "WHERE run_id=%s", (run,))
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
            elif change == "expired":
                cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' "
                               "WHERE run_id=%s", (run,))
            elif change == "old_generation":
                cursor.execute("UPDATE public.lab_arena_runs SET stage_generation=0 WHERE run_id=%s", (run,))
            elif change == "old_stage":
                cursor.execute("UPDATE public.lab_arena_runs SET stage=2 WHERE run_id=%s", (run,))
            elif change == "other_round":
                cursor.execute("UPDATE public.lab_arena_runs SET round_id='other-round' WHERE run_id=%s", (run,))
            elif change == "other_kind":
                cursor.execute("UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s", (run,))
            elif change == "other_runner":
                cursor.execute("UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s",
                               (prior.RUNNER_B, run))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "c")["run_id"] == fresh


def test_lease_authenticates_fault_event_and_expiry_is_unchanged(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        fresh, faults = _seed(conn, with_events=False)
        with conn.cursor() as cursor:
            cursor.execute("SELECT has_table_privilege('lab_arena_service',"
                           "'public.lab_arena_trajectory_events','INSERT')")
            assert cursor.fetchone()[0] is False
            cursor.execute("SET ROLE lab_arena_service")
            for run in faults:
                event = {
                    "event_id": str(uuid.uuid4()), "kind": "runtime.error",
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "content": {"status": "abandoned", "failure_stage": "runtime",
                                "error_class": "RuntimeHostError"},
                }
                cursor.execute("SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",
                               (run, "sha256:" + "b" * 64, json.dumps([event])))
                assert cursor.fetchone()[0]["status"] == "stale"
                cursor.execute("SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",
                               (run, "sha256:" + "a" * 64, json.dumps([event])))
                assert cursor.fetchone()[0]["status"] == "accepted"
            cursor.execute("RESET ROLE")
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_A, "d") == {"status": "no_pending"}
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_expire_leases(%s)", (prior.ROUND,))
            assert cursor.fetchone()[0]["retried"] == 0
            cursor.execute("SELECT COUNT(*) FROM public.lab_arena_runs WHERE run_id=ANY(%s) "
                           "AND status='leased' AND lease_expires_at>now()", (faults,))
            assert cursor.fetchone()[0] == 3
        conn.commit()
        assert prior._claim(conn, prior.RUNNER_B, "e")["run_id"] == fresh


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
                "SELECT 'active-load-'||i,'active-load-'||i,%s,%s,%s,1,i%%10,1,"
                "'execute','pending',0,1 FROM generate_series(1,2560) AS i",
                (prior.ROUND, prior.SUBMISSION, prior.MINER),
            )
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_A, "f") == {"status": "no_pending"}
        cooled = time.monotonic() - start
        start = time.monotonic()
        assert prior._claim(conn, prior.RUNNER_B, "e")["status"] == "leased"
        healthy = time.monotonic() - start
        assert cooled < 5.0
        assert healthy < 5.0
        print(f"2560-job active guard claim seconds: cooled={cooled:.3f}, healthy={healthy:.3f}")
