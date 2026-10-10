"""The scoring-close RPC and claims take claim-control before the round lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


SQL = Path(__file__).resolve().parents[2] / "scripts/459-lab-arena-close-scoring-lock-order.sql"
ROUND = "arena-2026-10-13"


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _seed(psycopg, dsn):
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,status_generation,stage_generation,configuration_doc,"
            "participants,benchmark_ref,rewards_enabled) "
            "VALUES (%s,'stage2_scoring',1,1,'{}'::jsonb,'[]'::jsonb,%s,false)",
            (ROUND, f"arena/{ROUND}/benchmark.json"),
        )


def _metadata(cursor):
    cursor.execute(
        "SELECT proowner,proacl,prosecdef,provolatile,proconfig "
        "FROM pg_catalog.pg_proc WHERE oid="
        "'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure"
    )
    return cursor.fetchone()


def _round_status(cursor):
    cursor.execute("SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
    return cursor.fetchone()[0]


def _reset(psycopg, dsn):
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage2_scoring',"
            "status_generation=1,stage_generation=1 WHERE round_id=%s", (ROUND,)
        )


def _wait_for_round_lock(psycopg, dsn, pid):
    until = time.monotonic() + 5
    with psycopg.connect(**dsn) as observer:
        observer.autocommit = True
        with observer.cursor() as cursor:
            while time.monotonic() < until:
                cursor.execute(
                    "SELECT wait_event_type FROM pg_catalog.pg_stat_activity WHERE pid=%s", (pid,)
                )
                if cursor.fetchone()[0] == "Lock":
                    return
                time.sleep(0.01)
    raise AssertionError("claim-side session did not wait for the round row")


def test_old_cycle_deadlocks_and_new_order_progresses_with_hold_intact(database):
    psycopg, dsn = database
    _seed(psycopg, dsn)

    # The old RPC enters with the round row already held, as its first SELECT
    # does. A concurrent claimant owns claim-control and waits for that row.
    # The status trigger then waits for claim-control: a real two-session cycle.
    with psycopg.connect(**dsn) as closer, psycopg.connect(**dsn) as claimant:
        closer.autocommit = False
        claimant.autocommit = False
        with closer.cursor() as a, claimant.cursor() as b:
            a.execute("SET deadlock_timeout='100ms'")
            b.execute("SET deadlock_timeout='100ms'")
            a.execute("SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE", (ROUND,))
            b.execute(
                "SELECT pg_catalog.pg_advisory_xact_lock("
                "pg_catalog.hashtextextended('lab-arena-claim-control',0))"
            )
            b.execute("SELECT pg_catalog.pg_backend_pid()")
            pid = b.fetchone()[0]
            claimant_errors = []

            def claim_round():
                try:
                    with claimant.cursor() as claim_cursor:
                        claim_cursor.execute(
                            "SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE",
                            (ROUND,),
                        )
                except psycopg.Error as error:
                    claimant_errors.append(error)

            worker = threading.Thread(target=claim_round, daemon=True)
            worker.start()
            _wait_for_round_lock(psycopg, dsn, pid)
            close_error = None
            try:
                a.execute("SELECT public.lab_arena_close_scoring(%s,2::smallint)", (ROUND,))
            except psycopg.Error as error:
                close_error = error
            worker.join(timeout=5)
            assert not worker.is_alive()
            assert any(error.pgcode == "40P01" for error in [close_error, *claimant_errors] if error)
            closer.rollback()
            claimant.rollback()

    _reset(psycopg, dsn)
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _metadata(cursor)
        cursor.execute(SQL.read_text())
        cursor.execute(SQL.read_text())  # exact idempotent replay
        assert _metadata(cursor) == before

    # In the new order, the close waits for claim-control before it can take
    # the round row. A claimant that already owns claim-control can complete.
    with psycopg.connect(**dsn) as closer, psycopg.connect(**dsn) as claimant:
        closer.autocommit = False
        claimant.autocommit = False
        with closer.cursor() as a, claimant.cursor() as b:
            a.execute("SET deadlock_timeout='100ms'")
            b.execute("SET deadlock_timeout='100ms'")
            b.execute(
                "SELECT pg_catalog.pg_advisory_xact_lock("
                "pg_catalog.hashtextextended('lab-arena-claim-control',0))"
            )
            a.execute("SELECT pg_catalog.pg_backend_pid()")
            pid = a.fetchone()[0]
            outcomes = []

            def close_round():
                with closer.cursor() as close_cursor:
                    close_cursor.execute(
                        "SELECT public.lab_arena_close_scoring(%s,2::smallint)", (ROUND,)
                    )
                    outcomes.append(close_cursor.fetchone()[0])
                closer.commit()

            worker = threading.Thread(target=close_round, daemon=True)
            worker.start()
            _wait_for_round_lock(psycopg, dsn, pid)
            b.execute("SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE", (ROUND,))
            assert b.fetchone() == (1,)
            claimant.commit()
            worker.join(timeout=5)
            assert not worker.is_alive()
            assert outcomes[0]["status"] == "closed"

    _reset(psycopg, dsn)
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton"
        )
        with pytest.raises(psycopg.Error) as error:
            cursor.execute("SELECT public.lab_arena_close_scoring(%s,2::smallint)", (ROUND,))
        assert error.value.pgcode == "55000"
        connection.rollback()
        assert _round_status(cursor) == "stage2_scoring"
