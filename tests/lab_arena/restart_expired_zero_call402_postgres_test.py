"""Guarded restart preserves naturally expired zero-call execute attempts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import restart_expired_zero_call394_postgres_test as prior


database = prior.database
migration_sql = prior.migration_sql
installed = prior.installed
source = prior.source
MIGRATION = Path(__file__).parents[2] / "scripts/402-lab-arena-restart-expired-zero-call-execute-drain.sql"
PREIMAGE = prior.POSTIMAGE
POSTIMAGE = (
    "9515a96a25174d1bb5daba23f9920f7c665edf11aa6f9d73570ecd87f3c47707",
    "3a4f36e387c478abbd09805639a56e1e71b0c1d927605439399b5e133ebe5d45",
)


@pytest.fixture(scope="module")
def upgraded(installed):
    psycopg, dsn = installed
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            before = prior._function_state(cursor)
            assert tuple(row[0] for row in before) == PREIMAGE
            sql = MIGRATION.read_text()
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(PREIMAGE[0], "f" * 64, 1))
            cursor.execute("ROLLBACK")
            assert prior._function_state(cursor) == before
            cursor.execute(sql)
            after = prior._function_state(cursor)
            assert tuple(row[0] for row in after) == POSTIMAGE
            cursor.execute(sql)
            assert prior._function_state(cursor) == after
            assert all(row[1:] == old[1:] for row, old in zip(after, before))
    yield psycopg, dsn


def _execute_lease(connection, *, stage=1):
    prior.prior._leased_prior(connection, kind="execute")
    if stage == 2:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s", (source.ROUND,))
            cursor.execute("UPDATE public.lab_arena_runs SET stage=2,icp_position=10 WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            cursor.execute("SET session_replication_role=origin")
        connection.commit()


def test_original_guard_rejects_naturally_captured_execute(installed):
    psycopg, dsn = installed
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _execute_lease(connection)
        acquired = prior._acquire(connection, 0)
        generation = acquired["guard_generation"]
        assert acquired["drain"]["captured_count"] == 1
        connection.commit()
        with pytest.raises(psycopg.Error, match="captured expiry write set differs"):
            prior._quiescence(connection, generation=generation)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT status,result_doc,output_ref FROM public.lab_arena_runs WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            assert cursor.fetchone() == ("leased", None, None)
        prior._rpc(connection, "lab_arena_abort_restart_guard_v1", prior.GUARD, prior.OWNER, generation, "test-abort")


@pytest.mark.parametrize("stage", [1, 2])
def test_captured_zero_call_execute_expires_with_retry(upgraded, stage):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _execute_lease(connection, stage=stage)
        with connection.cursor() as cursor:
            cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
            generation = cursor.fetchone()[0]
        acquired = prior._acquire(connection, generation)
        generation = acquired["guard_generation"]
        assert acquired["drain"]["captured_count"] == 1
        connection.commit()
        drained = prior._quiescence(connection, generation=generation)
        assert drained["preserved"] is True
        assert (drained["expired_receipt_count"], drained["lost_or_mutated_count"], drained["pending_retry_count"]) == (1, 0, 1)
        assert prior._quiescence(connection, generation=generation)["outcome_commitment"] == drained["outcome_commitment"]
        with connection.cursor() as cursor:
            cursor.execute("SELECT status,kind,stage,terminal_cause,result_doc,output_ref,terminal_doc->>'expired_at' FROM public.lab_arena_runs WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            status, kind, actual_stage, cause, result, output, expired_at = cursor.fetchone()
            assert (status,kind,actual_stage,cause,result,output) == ("failed","execute",stage,"lease_expired",None,None)
            assert expired_at is not None
            cursor.execute("SELECT status,kind,attempt,result_doc,output_ref FROM public.lab_arena_runs WHERE run_id=%s", (source.ASSIGNMENT + ":2",))
            assert cursor.fetchone() == ("pending","execute",2,None,None)
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            assert cursor.fetchone()[0] == 0
        prior._rpc(connection, "lab_arena_abort_restart_guard_v1", prior.GUARD, prior.OWNER, generation, "test-abort")


def test_cost_bearing_execute_stays_leased(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _execute_lease(connection)
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) VALUES ('reservation',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',1)", (source.MINER, source.ROUND, source.SUBMISSION, source.ASSIGNMENT + ":1", "sha256:" + "e" * 64))
            cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
            generation = cursor.fetchone()[0]
        generation = prior._acquire(connection, generation)["guard_generation"]
        connection.commit()
        waiting = prior._quiescence(connection, generation=generation)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            assert cursor.fetchone()[0] == "leased"
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            assert cursor.fetchone()[0] == 1
        prior._rpc(connection, "lab_arena_abort_restart_guard_v1", prior.GUARD, prior.OWNER, generation, "test-abort")


def test_original_score_expiry_remains_admitted(upgraded):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        prior._lease(connection, overdue=True)
        with connection.cursor() as cursor:
            cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
            generation = cursor.fetchone()[0]
        generation = prior._acquire(connection, generation)["guard_generation"]
        connection.commit()
        drained = prior._quiescence(connection, generation=generation)
        assert drained["preserved"] is True
        assert (drained["expired_receipt_count"], drained["pending_retry_count"]) == (1, 1)
        prior._rpc(connection, "lab_arena_abort_restart_guard_v1", prior.GUARD, prior.OWNER, generation, "test-abort")


@pytest.mark.parametrize("change", ["phase", "generation", "output"])
def test_execute_with_changed_write_set_stays_leased(upgraded, change):
    psycopg, dsn = upgraded
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _execute_lease(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
            generation = cursor.fetchone()[0]
        generation = prior._acquire(connection, generation)["guard_generation"]
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            if change == "phase":
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring' WHERE round_id=%s", (source.ROUND,))
            elif change == "generation":
                cursor.execute("UPDATE public.lab_arena_rounds SET stage_generation=2 WHERE round_id=%s", (source.ROUND,))
            else:
                cursor.execute("UPDATE public.lab_arena_runs SET output_ref='unexpected-output' WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with pytest.raises(psycopg.Error, match="captured expiry write set differs"):
            prior._quiescence(connection, generation=generation)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM public.lab_arena_runs WHERE run_id=%s", (source.ASSIGNMENT + ":1",))
            assert cursor.fetchone()[0] == "leased"
        prior._rpc(connection, "lab_arena_abort_restart_guard_v1", prior.GUARD, prior.OWNER, generation, "test-abort")
