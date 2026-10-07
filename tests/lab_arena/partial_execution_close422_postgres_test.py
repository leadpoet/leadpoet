"""Disposable PostgreSQL proof for closing execution with unfinished work."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena.oct07_cutoff_recovery419_postgres_test import database as current_database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _seed_deadline_close_guard
from tests.lab_arena.stage_cutoff_drain420_postgres_test import (
    ROUND, SUB, _boundary, _claim, _close, _complete, _seed, _state,
    base_database, database,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "scripts/422-lab-arena-partial-execution-close.sql"
SIGNATURES = (
    "public.lab_arena_close_parallel_execution_v1(text)",
    "public.lab_arena_close_stage(text,smallint)",
)


def _functions(cur):
    cur.execute("""
        SELECT p.oid::regprocedure::TEXT, p.proowner, p.proacl, p.prosecdef,
          p.provolatile, p.proconfig,
          encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')
        FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[])
        ORDER BY p.oid::regprocedure::TEXT
    """, (list(SIGNATURES),))
    return cur.fetchall()


@pytest.fixture(scope="module")
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        before = _functions(cur)
        cur.execute(SQL.read_text())
        after = _functions(cur)
        assert [row[:-1] for row in after] == [row[:-1] for row in before]
        assert [row[-1] for row in after] != [row[-1] for row in before]
        cur.execute(SQL.read_text())
        assert _functions(cur) == after
    return database


def _make_six_ninety(conn):
    """Ten frozen stage-two positions for each of 69 submissions."""
    _seed(conn, positions=10)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        for group in range(1, 69):
            sub = f"{SUB}-{group}"
            cur.execute("""
                INSERT INTO public.lab_arena_submissions
                SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,
                  to_jsonb(s)||jsonb_build_object(
                    'submission_id',%s,'miner_hotkey',%s))).*
                FROM public.lab_arena_submissions s WHERE submission_id=%s
            """, (sub, hotkey(sub), SUB))
            for position in range(10, 20):
                assignment = f"{ROUND}:{sub}:2:{position}"
                cur.execute("""
                    INSERT INTO public.lab_arena_runs(
                      run_id,assignment_id,round_id,submission_id,miner_hotkey,
                      stage,icp_position,attempt,kind,status,stage_generation)
                    VALUES (%s,%s,%s,%s,%s,2,%s,1,'execute','pending',7)
                """, (assignment + ":1", assignment, ROUND, sub,
                      hotkey(sub), position))
        cur.execute("""
            UPDATE public.lab_arena_rounds r SET participants=(
              SELECT jsonb_agg(jsonb_build_object(
                'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
                'is_king',false,'source_ref',s.source_ref)
                ORDER BY s.submission_id)
              FROM public.lab_arena_submissions s WHERE s.round_id=r.round_id)
            WHERE r.round_id=%s
        """, (ROUND,))
        cur.execute("SET LOCAL session_replication_role=origin")
    conn.commit()
    lease = _claim(conn)
    assert lease["status"] == "leased"
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("""
            WITH chosen AS (
              SELECT run_id FROM public.lab_arena_runs
              WHERE round_id=%s AND status='pending'
              ORDER BY run_id LIMIT 675)
            UPDATE public.lab_arena_runs r
            SET status='accepted', terminal_cause='accepted',
              result_doc='{"terminal_status":"accepted"}'::jsonb,
              output_ref='arena/partial-close/accepted.json'
            FROM chosen WHERE r.run_id=chosen.run_id
        """, (ROUND,))
        cur.execute("""
            WITH chosen AS (
              SELECT run_id FROM public.lab_arena_runs
              WHERE round_id=%s AND status='pending'
              ORDER BY run_id LIMIT 5)
            UPDATE public.lab_arena_runs r
            SET status='failed', terminal_cause='model_timeout',
              terminal_doc='{"terminal_status":"model_timeout"}'::jsonb,
              result_doc='{"terminal_status":"model_timeout"}'::jsonb,
              per_icp_score=0
            FROM chosen WHERE r.run_id=chosen.run_id
        """, (ROUND,))
        cur.execute("""
            SELECT run_id,submission_id,miner_hotkey
            FROM public.lab_arena_runs
            WHERE round_id=%s AND status='accepted' ORDER BY run_id LIMIT 1
        """, (ROUND,))
        paid_run, paid_sub, paid_hotkey = cur.fetchone()
        cur.execute("""
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,
              amount_microusd,entry_doc)
            VALUES ('settlement',%s,%s,%s,%s,2,%s,'openrouter',
              'openrouter.responses','host',1234,'{}'::jsonb)
        """, (paid_hotkey, ROUND, paid_sub, paid_run,
              "sha256:" + "b" * 64))
        cur.execute("SET LOCAL session_replication_role=origin")
    conn.commit()
    return lease


def test_six_ninety_assignment_close_preserves_results_and_costs(migrated):
    psycopg, dsn = migrated
    with psycopg.connect(**dsn) as conn:
        lease = _make_six_ninety(conn)
        before = _state(conn)
        assert {status: sum(r["status"] == status for r in before["lab_arena_runs"])
                for status in ("accepted", "failed", "leased", "pending")} == {
                    "accepted": 675, "failed": 5, "leased": 1, "pending": 9}
        accepted = [r for r in before["lab_arena_runs"] if r["status"] == "accepted"]
        genuine_failures = [r for r in before["lab_arena_runs"] if r["status"] == "failed"]
        paid = before["lab_arena_ledger"]
        assert len(paid) == 1 and paid[0]["amount_microusd"] == 1234
        _boundary(conn)
        at_cutoff = _state(conn)
        assert _close(conn)["status"] == "draining"
        assert _state(conn) == at_cutoff
        _boundary(conn, -4501)
        assert _close(conn) == {
            "status": "closed", "round_status": "stage2_closed",
            "incomplete_assignments": 10, "stage_generation": 8,
        }
        after = _state(conn)
        assert after["lab_arena_rounds"][0]["status"] == "stage2_closed"
        assert after["lab_arena_rounds"][0]["cancel_reason"] is None
        assert [r for r in after["lab_arena_runs"] if r["status"] == "accepted"] == accepted
        assert [r for r in after["lab_arena_runs"] if r["terminal_cause"] == "model_timeout"] == genuine_failures
        unfinished = [r for r in after["lab_arena_runs"]
                      if r["terminal_doc"] and r["terminal_doc"].get("infrastructure_incomplete")]
        assert len(unfinished) == 10
        assert all(r["terminal_cause"] == "stage_closed" and
                   r["terminal_doc"]["infrastructure_incomplete"] is True and
                   r["per_icp_score"] is None and r["result_doc"] is None
                   for r in unfinished)
        assert after["lab_arena_ledger"] == paid
        assert _close(conn)["status"] == "existing"
        assert _state(conn) == after
        assert _complete(conn, lease)["status"] != "accepted"
        assert _state(conn) == after


def test_parallel_close_keeps_accepted_and_incomplete_separate(migrated):
    psycopg, dsn = migrated
    round_id = "arena-2099-01-02-p422"
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        _seed_deadline_close_guard(
            conn, round_id, retry_status="pending",
            prior_cause="provider_error", ledger_kinds=(),
        )
        cur.execute("""
            SELECT to_jsonb(r) FROM public.lab_arena_runs r
            WHERE round_id=%s AND status='accepted'
        """, (round_id,))
        accepted = cur.fetchone()[0]
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("""
            INSERT INTO public.lab_arena_submissions(
              submission_id,round_id,miner_hotkey,status,is_king,source_ref,
              source_size_bytes,submission_doc,code_review_status,
              code_review_doc,code_review_claim,code_review_started_at,
              code_review_attempts)
            VALUES (%s,%s,%s,'frozen',false,%s,123,'{}','passed',
              '{"decision":"pass"}','sha256:'||repeat('1',64),now(),1)
        """, (accepted["submission_id"], round_id, accepted["miner_hotkey"],
              f"arena/{round_id}/sources/{accepted['submission_id']}.tar.gz"))
        cur.execute("SET LOCAL session_replication_role=origin")
        cur.execute("SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                    (round_id,))
        assert cur.fetchone()[0] == {
            "status": "closed", "round_status": "stage1_closed",
            "incomplete_assignments": 1, "stage_generation": 2,
        }
        cur.execute("""
            SELECT to_jsonb(r) FROM public.lab_arena_runs r
            WHERE round_id=%s AND status='accepted'
        """, (round_id,))
        assert cur.fetchone()[0] == accepted
        cur.execute("""
            SELECT status,terminal_cause,terminal_doc,per_icp_score
            FROM public.lab_arena_runs WHERE round_id=%s AND attempt=2
        """, (round_id,))
        status, cause, doc, score = cur.fetchone()
        assert (status, cause, score) == ("failed", "stage_closed", None)
        assert doc["infrastructure_incomplete"] is True
        cur.execute("SELECT status,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s",
                    (round_id,))
        assert cur.fetchone() == ("stage1_closed", None)
        conn.commit()


def test_provider_failure_without_retry_is_infrastructure_incomplete(migrated):
    psycopg, dsn = migrated
    round_id = "arena-2099-01-02-q422"
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        _seed_deadline_close_guard(
            conn, round_id, retry_status="pending",
            prior_cause="provider_error", ledger_kinds=(),
        )
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND attempt=2",
                    (round_id,))
        cur.execute("""
            SELECT submission_id,miner_hotkey FROM public.lab_arena_runs
            WHERE round_id=%s LIMIT 1
        """, (round_id,))
        submission_id, miner_hotkey = cur.fetchone()
        cur.execute("""
            INSERT INTO public.lab_arena_submissions(
              submission_id,round_id,miner_hotkey,status,is_king,source_ref,
              source_size_bytes,submission_doc,code_review_status,
              code_review_doc,code_review_claim,code_review_started_at,
              code_review_attempts)
            VALUES (%s,%s,%s,'frozen',false,%s,123,'{}','passed',
              '{"decision":"pass"}','sha256:'||repeat('1',64),now(),1)
        """, (submission_id, round_id, miner_hotkey,
              f"arena/{round_id}/sources/{submission_id}.tar.gz"))
        cur.execute("SET LOCAL session_replication_role=origin")
        cur.execute("SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                    (round_id,))
        result = cur.fetchone()[0]
        assert result["status"] == "closed"
        assert result["incomplete_assignments"] == 1
        cur.execute("""
            SELECT status,terminal_cause,terminal_doc,per_icp_score
            FROM public.lab_arena_runs WHERE round_id=%s
              AND terminal_cause='provider_error'
        """, (round_id,))
        status, cause, doc, score = cur.fetchone()
        assert (status, cause, score) == ("failed", "provider_error", None)
        assert doc["infrastructure_incomplete"] is True
        conn.commit()


def test_dispatched_exhausted_provider_retry_remains_eligible(migrated):
    psycopg, dsn = migrated
    round_id = "arena-2099-01-02-r422"
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        _seed_deadline_close_guard(
            conn, round_id, retry_status="leased",
            prior_cause="provider_error",
            ledger_kinds=("reservation", "dispatch"),
        )
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("""
            SELECT submission_id,miner_hotkey FROM public.lab_arena_runs
            WHERE round_id=%s LIMIT 1
        """, (round_id,))
        submission_id, miner_hotkey = cur.fetchone()
        cur.execute("""
            INSERT INTO public.lab_arena_submissions(
              submission_id,round_id,miner_hotkey,status,is_king,source_ref,
              source_size_bytes,submission_doc,code_review_status,
              code_review_doc,code_review_claim,code_review_started_at,
              code_review_attempts)
            VALUES (%s,%s,%s,'frozen',false,%s,123,'{}','passed',
              '{"decision":"pass"}','sha256:'||repeat('1',64),now(),1)
        """, (submission_id, round_id, miner_hotkey,
              f"arena/{round_id}/sources/{submission_id}.tar.gz"))
        cur.execute("SET LOCAL session_replication_role=origin")
        cur.execute("SELECT public.lab_arena_close_parallel_execution_v1(%s)",
                    (round_id,))
        result = cur.fetchone()[0]
        assert result["status"] == "closed"
        assert result["incomplete_assignments"] == 0
        cur.execute("""
            SELECT terminal_cause,terminal_doc FROM public.lab_arena_runs
            WHERE round_id=%s AND attempt=2
        """, (round_id,))
        cause, doc = cur.fetchone()
        assert cause == "stage_closed"
        assert doc["deadline_provider_retry_exhausted"] is True
        assert "infrastructure_incomplete" not in doc
        conn.commit()


def test_migration_rejects_function_drift_without_partial_write(migrated):
    psycopg, dsn = migrated
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        before = _functions(cur)
        cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURES[1],))
        drifted = cur.fetchone()[0].replace("BEGIN", "BEGIN\n  -- drift", 1)
        cur.execute(drifted)
        with pytest.raises(Exception, match="preimage differs"):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert _functions(cur) == before
