"""The ordinary judge queue survives score close without changing source authority."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from tests.lab_arena import bounded_baseline_recovery470_postgres_test as previous
from tests.lab_arena import execute_host_cooldown396_postgres_test as host_faults


SQL = Path(__file__).parents[2] / "scripts/472-lab-arena-general-judge-continuation.sql"


@pytest.fixture(scope="module")
def database():
    generator = previous.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cursor:
            cursor.execute(SQL.read_text())
            cursor.execute(SQL.read_text())
        yield db
    finally:
        generator.close()


def prepare(conn, *, cause="judge_error"):
    assignment, cfg = previous.seed(conn, kind="score", cause=cause)
    source_id = assignment.removesuffix(":score") + ":1"
    item = previous.scope_item(conn, source_id)
    with conn.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("""UPDATE public.lab_arena_rounds SET status='scored',stage_generation=8,
          stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s""", (
            json.dumps({"work_items": [{"scored_run_id": source_id,
              "submission_id": previous.SUB, "icp_position": 0,
              "output_ref": "source-output"}]}), previous.ROUND))
        cursor.execute("""UPDATE public.lab_arena_runs SET judgment_cache_key=%s,
          judgment_input_hash=%s,judgment_scope_doc=%s::jsonb
          WHERE round_id=%s""", (
            item["judgment_cache_key"], item["judgment_input_hash"],
            json.dumps(item["judgment_scope_doc"]),
            previous.ROUND))
        cursor.execute("SET LOCAL session_replication_role=origin")
    conn.commit()
    return assignment, source_id, cfg


def test_old_failure_rejoins_existing_queue_after_close(database):
    with database[0].connect(**database[1]) as conn:
        assignment, source_id, _ = prepare(conn)
        before = conn.cursor()
        before.execute("SELECT result_doc,output_ref,status FROM public.lab_arena_runs WHERE run_id=%s",
                       (assignment + ":2",))
        old = before.fetchone()
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 1
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT result_doc,output_ref,status FROM public.lab_arena_runs WHERE run_id=%s",
                           (assignment + ":2",))
            assert cursor.fetchone() == old
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                           (assignment + ":3",))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        lease = previous.claim(conn)
        assert (lease["status"], lease["attempt"], lease["scored_run_id"]) == (
            "leased", 3, source_id), lease
        result = previous.rpc(conn, "lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)", (
            lease["run_id"], previous.TOKEN,
            json.dumps({"terminal_status": "judge_error"}), "judge_error", ""))
        assert result["confirmation_attempt"] == 4


def test_retry_waits_for_backoff_after_close(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 1
        assert previous.claim(conn) == {"status": "no_pending"}
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                           (assignment + ":3",))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        assert previous.claim(conn)["run_id"] == assignment + ":3"


def test_expired_score_lease_keeps_partial_refs_in_successor(database):
    with database[0].connect(**database[1]) as conn:
        assignment, source_id, _ = prepare(conn)
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 1
        refs = [{"company_index": 0, "cache_key": "sha256:" + "c" * 64,
                 "company_input_hash": "sha256:" + "d" * 64}]
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("""UPDATE public.lab_arena_runs SET
              created_at=clock_timestamp()-interval '20 minutes',
              company_judgment_refs=%s::jsonb WHERE run_id=%s""",
              (json.dumps(refs), assignment + ":3"))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        lease = previous.claim(conn)
        assert (lease["run_id"], lease["scored_run_id"], lease["miner_hotkey"]) == (
            assignment + ":3", source_id, previous.MINER)
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",
                           (lease["run_id"],))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        expired = previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))
        assert (expired["expired"], expired["retried"]) == (1, 1), expired
        with conn.cursor() as cursor:
            cursor.execute("""SELECT attempt,status,terminal_cause,company_judgment_refs,
              scored_run_id,miner_hotkey FROM public.lab_arena_runs
              WHERE assignment_id=%s ORDER BY attempt""", (assignment,))
            rows = cursor.fetchall()
        assert rows[-2][1:4] == ("failed", "lease_expired", refs)
        assert rows[-1][:2] == (4, "pending")
        assert rows[-1][3:] == (refs, source_id, previous.MINER)
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 0


def test_accepted_judge_completion_is_idempotent_and_stops_retry(database):
    with database[0].connect(**database[1]) as conn:
        assignment, source_id, _ = prepare(conn, cause="credential_error")
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 1
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                           (assignment + ":3",))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        lease = previous.claim(conn)
        assert (lease["run_id"], lease["scored_run_id"], lease["miner_hotkey"],
                lease["submission_id"]) == (
            assignment + ":3", source_id, previous.MINER, previous.SUB)
        args = (lease["run_id"], previous.TOKEN,
                json.dumps({"terminal_status": "accepted"}), "accepted",
                "accepted-judge-output")
        first = previous.rpc(conn,
            "lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)", args)
        again = previous.rpc(conn,
            "lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)", args)
        assert first["status"] == "accepted"
        assert again["status"] == "accepted" and again["idempotent"] is True
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*),max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s",
                           (assignment,))
            assert cursor.fetchone() == (3, 3)


def test_ambiguous_call_blocks_retry_and_final_authority(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn, cause="credential_error")
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES ('dispatch',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','miner_key',100,'{}'::jsonb)", (
                previous.MINER, previous.ROUND, previous.SUB, assignment + ":2",
                "sha256:" + "f" * 64))
        conn.commit()
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (previous.ROUND,))
            assert cursor.fetchone()[0] is False


def test_host_fault_quarantine_covers_scored_recovery(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)",
                            (previous.ROUND,))["retried"] == 1
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                           (assignment + ":3",))
            for position in (1, 2, 3):
                run_id = f"host-fault-{position}"
                cursor.execute("""INSERT INTO public.lab_arena_runs
                  (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
                   icp_position,attempt,kind,status,stage_generation,terminal_cause,
                   runner_hotkey,lease_expires_at)
                  VALUES (%s,%s,%s,%s,%s,1,%s,1,'execute','failed',1,
                    'lease_expired',%s,clock_timestamp()-interval '10 minutes')""",
                  (run_id, run_id, previous.ROUND, previous.SUB, previous.MINER,
                   position, previous.RUNNER))
                host_faults._event(cursor, run_id)
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        assert previous.claim(conn) == {"status": "no_pending"}
        healthy = "5" + "C" * 47
        result = previous.rpc(conn,
          "lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)",
          (previous.ROUND, healthy, "b" * 32, "sha256:" + "b" * 64,
           previous.TOKEN))
        assert result["run_id"] == assignment + ":3", result


def test_credential_failure_waits_for_valid_zero_without_prior_score(database):
    with database[0].connect(**database[1]) as conn:
        assignment, source_id, _ = prepare(conn, cause="credential_error")
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 1
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                           (assignment + ":3",))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        lease = previous.claim(conn)
        assert lease["status"] == "leased", lease
        assert previous.rpc(conn, "lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)", (
            lease["run_id"], previous.TOKEN,
            json.dumps({"terminal_status": "accepted"}), "accepted",
            "accepted-judge-output"))["status"] == "accepted"
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (previous.ROUND,))
            assert cursor.fetchone()[0] is False
        result = previous.rpc(conn, "lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)", (
            previous.ROUND, json.dumps([{"run_id": source_id,
              "per_icp_score": 0, "qualification_doc": {"companies": []}}])))
        assert result["status"] == "ok" and result["recorded"] == 1, result
        with conn.cursor() as cursor:
            cursor.execute("SELECT per_icp_score,public.lab_arena__all_judgments_ready_v1(%s) FROM public.lab_arena_runs WHERE run_id=%s",
                           (previous.ROUND, source_id))
            assert cursor.fetchone() == (0, True)


def test_pending_first_judge_is_admitted_only_for_frozen_work_item(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id IN (%s,%s)",
                           (assignment + ":1", assignment + ":2"))
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scored_run_id,judgment_cache_key,judgment_input_hash,judgment_scope_doc) SELECT %s,%s,%s,%s,%s,1,0,1,'score','pending',7,%s,%s,%s,%s::jsonb", (
                assignment + ":1", assignment, previous.ROUND, previous.SUB,
                previous.MINER, assignment.removesuffix(":score") + ":1",
                previous.scope_item(conn, assignment.removesuffix(":score") + ":1")["judgment_cache_key"],
                previous.scope_item(conn, assignment.removesuffix(":score") + ":1")["judgment_input_hash"],
                json.dumps(previous.scope_item(conn, assignment.removesuffix(":score") + ":1")["judgment_scope_doc"])))
        conn.commit()
        lease = previous.claim(conn)
        assert (lease["status"], lease["attempt"]) == ("leased", 1), lease
        completed = previous.rpc(conn, "lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)", (
            lease["run_id"], previous.TOKEN,
            json.dumps({"terminal_status": "credential_error"}),
            "credential_error", ""))
        assert completed["confirmation_attempt"] == 2
        with conn.cursor() as cursor:
            cursor.execute("SELECT per_icp_score FROM public.lab_arena_runs WHERE run_id=%s",
                           (assignment.removesuffix(":score") + ":1",))
            assert cursor.fetchone()[0] is None


def test_prior_evaluation_day_is_not_enrolled_or_backfilled(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13' WHERE round_id=%s",
                           (previous.ROUND,))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_judge_continuation_enabled_v1(%s)",
                           (previous.ROUND,))
            assert cursor.fetchone()[0] is False
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3",
                           (assignment,))
            assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize("rewards_enabled,expected", [(False, True), (True, False)])
def test_prior_day_shadow_enrollment_requires_disabled_rewards(database,
                                                               rewards_enabled, expected):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("""UPDATE public.lab_arena_rounds
              SET evaluation_date='2026-10-13', rewards_enabled=%s,
                created_at='2026-10-13T16:31:00Z',
                configuration_doc=jsonb_set(configuration_doc,'{mode}','"shadow"'::jsonb)
              WHERE round_id=%s""", (rewards_enabled, previous.ROUND))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_judge_continuation_enabled_v1(%s)",
                           (previous.ROUND,))
            assert cursor.fetchone()[0] is expected
        result = previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))
        assert result["retried"] == int(expected)


def test_old_reward_disabled_shadow_is_not_enrolled(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _, _ = prepare(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("""UPDATE public.lab_arena_rounds
              SET evaluation_date='2026-10-13', rewards_enabled=false,
                created_at='2026-10-13T16:29:59Z',
                configuration_doc=jsonb_set(configuration_doc,'{mode}','"shadow"'::jsonb)
              WHERE round_id=%s""", (previous.ROUND,))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        assert previous.rpc(conn, "lab_arena_expire_leases(%s)", (previous.ROUND,))["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3",
                           (assignment,))
            assert cursor.fetchone()[0] == 0


def test_prior_day_legacy_baseline_recovery_is_fenced(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _ = previous.seed(conn, kind="execute", cause="provider_error")
        with conn.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13' WHERE round_id=%s",
                           (previous.ROUND,))
            cursor.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
        assert previous.recover(conn) == {"status": "stale", "queued": 0}
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3",
                           (assignment,))
            assert cursor.fetchone()[0] == 0
