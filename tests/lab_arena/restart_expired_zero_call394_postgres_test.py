"""Captured zero-ledger score expiry can finish a guarded restart safely."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena import expiry_runner_handoff390_postgres_test as prior
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import (
    CANDIDATE,
    GUARD,
    OWNER,
    _rpc,
)


migration_sql = prior.migration_sql
source = prior.prior
MIGRATION = Path(__file__).parents[2] / "scripts/394-lab-arena-restart-expired-zero-call-drain.sql"


@pytest.fixture(scope="module")
def database():
    preimage = tuple(name for name in CURRENT_SERVICE_MIGRATIONS
                     if name != MIGRATION.name)
    assert len(preimage) + 1 == len(CURRENT_SERVICE_MIGRATIONS)
    yield from database_with_lab_arena_migration(
        preimage + (source.MIGRATION_359.name, source.MIGRATION.name)
    )
FUNCTIONS = (
    "public.lab_arena__restart_drain_state_v1()",
    "public.lab_arena_restart_quiescence_v1(text,text,bigint)",
)
PREIMAGE = (
    "dbc37aedf894a8e26a7a6eb4fe3a72c6d1d6cdb1a71ee8557183670320494b9a",
    "d33ef97bf5fd0a95b61cd807d4e1102449024a3e0f9b02a044aea0740b5c1b46",
)
POSTIMAGE = (
    "183fbb7525dd7561451bb386960ef9e602efd9c205dce754683c9e60ead2ee47",
    "c3a87110def2cb47833ec4bc353c5c6a965c655f0f03c490019fd34d5de4dcfb",
)


def _function_state(cursor):
    states = []
    for name in FUNCTIONS:
        cursor.execute(
            "SELECT encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex'),"
            "owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner "
            "ON owner.oid=p.proowner WHERE p.oid=%s::regprocedure",
            (name,),
        )
        states.append(cursor.fetchone())
    return states


@pytest.fixture(scope="module")
def installed(database, migration_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            before = _function_state(cursor)
            assert tuple(row[0] for row in before) == PREIMAGE
            sql = MIGRATION.read_text()
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql.replace(PREIMAGE[0], "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert _function_state(cursor) == before
            cursor.execute(sql)
            after = _function_state(cursor)
            assert tuple(row[0] for row in after) == POSTIMAGE
            cursor.execute(sql)  # Replay cannot change an active guard or receipt.
            assert _function_state(cursor) == after
            assert all(row[1:] == prior_row[1:] for row, prior_row in zip(after, before))
    yield psycopg, dsn


def _acquire(connection, generation: int):
    return _rpc(
        connection, "lab_arena_acquire_restart_guard_v1",
        GUARD, OWNER, generation, 600, CANDIDATE, "gateway", "test-restart-expiry",
    )


def _quiescence(connection, *, owner: str = OWNER, generation: int):
    return _rpc(
        connection, "lab_arena_restart_quiescence_v1", GUARD, owner, generation,
    )


def _lease(connection, *, overdue: bool):
    prior._leased_prior(connection)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET scored_run_id='source-execution', "
            "lease_expires_at = "
            "clock_timestamp() + %s * interval '1 second' WHERE run_id=%s",
            (-1 if overdue else 300, source.ASSIGNMENT + ":1"),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def test_owner_expiry_preserves_audit_and_retry_through_release(installed):
    psycopg, dsn = installed
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _lease(connection, overdue=False)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,stage_generation,configuration_doc,participants "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (source.ROUND,),
            )
            frozen_round = cursor.fetchone()
        state = _acquire(connection, 0)
        assert state["guard_generation"] == 1
        assert state["drain"]["captured_count"] == 1
        connection.commit()
        with pytest.raises(psycopg.Error, match="owner_or_generation_differs"):
            _quiescence(
                connection, owner="lab_arena_restart_owner:" + "f" * 64,
                generation=1,
            )
        connection.rollback()
        waiting = _quiescence(connection, generation=1)
        assert waiting["still_leased_count"] == 1
        assert waiting["expired_receipt_count"] == 0
        assert waiting["preserved"] is False
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at = "
                "clock_timestamp() - interval '1 second' WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
        drained = _quiescence(connection, generation=1)
        assert drained["captured_count"] == 1
        assert drained["expired_receipt_count"] == 1
        assert drained["accepted_receipt_count"] == 0
        assert drained["reported_terminal_receipt_count"] == 0
        assert drained["lost_or_mutated_count"] == 0
        assert drained["current_leased_count"] == 0
        assert drained["pending_retry_count"] == 1
        assert drained["preserved"] is True
        assert _quiescence(connection, generation=1)["outcome_commitment"] == drained["outcome_commitment"]
        with connection.cursor() as cursor:
            for bad_doc in (None, {"expired_at": "9999-01-01T00:00:00+00:00"}):
                cursor.execute("SET session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET terminal_doc=%s "
                    "WHERE run_id=%s",
                    (json.dumps(bad_doc) if bad_doc is not None else None,
                     source.ASSIGNMENT + ":1"),
                )
                cursor.execute("SET session_replication_role=origin")
                assert _quiescence(connection, generation=1)["preserved"] is False
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_doc="
                "jsonb_build_object('expired_at',lease_expires_at) "
                "WHERE run_id=%s", (source.ASSIGNMENT + ":1",),
            )
            cursor.execute("SET session_replication_role=origin")
        assert _quiescence(connection, generation=1)["preserved"] is True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,terminal_cause,result_doc,output_ref,"
                "terminal_doc->>'expired_at',lease_expires_at "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
            old_status, cause, result, output, expired_at, lease_expiry = cursor.fetchone()
            assert (old_status, cause, result, output) == (
                "failed", "lease_expired", None, None,
            )
            assert expired_at is not None and lease_expiry is not None
            cursor.execute(
                "SELECT status,attempt,kind,scored_run_id,previous_runner_hotkey,"
                "result_doc,output_ref FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":2",),
            )
            retry = cursor.fetchone()
            assert retry == (
                "pending", 2, "score", "source-execution",
                source.RUNNER_A, None, None,
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT status,stage_generation,configuration_doc,participants "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (source.ROUND,),
            )
            assert cursor.fetchone() == frozen_round
        assert source._claim(connection, source.RUNNER_B, "a") == {"status": "paused"}
        _rpc(connection, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "gateway_destructive")
        _rpc(connection, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "gateway_ready")
        released = _rpc(connection, "lab_arena_release_restart_guard_v1", GUARD, OWNER, 1, "test-release")
        assert released["guard_present"] is False
        assert source._claim(connection, source.RUNNER_A, "b") == {"status": "no_pending"}
        retry_claim = source._claim(connection, source.RUNNER_B, "c")
        assert retry_claim["status"] == "leased"
        assert retry_claim["run_id"] == source.ASSIGNMENT + ":2"


def test_cost_bearing_captured_lease_is_not_expired(installed):
    psycopg, dsn = installed
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _lease(connection, overdue=True)
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd) "
                "VALUES ('reservation',%s,%s,%s,%s,1,%s,'openrouter',"
                "'openrouter.chat','host',1)",
                (source.MINER, source.ROUND, source.SUBMISSION,
                 source.ASSIGNMENT + ":1", "sha256:" + "d" * 64),
            )
        state = _acquire(connection, 1)
        assert state["guard_generation"] == 2
        waiting = _quiescence(connection, generation=2)
        assert waiting["preserved"] is False
        assert waiting["still_leased_count"] == 1
        assert waiting["expired_receipt_count"] == 0
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
            assert cursor.fetchone()[0] == "leased"
            cursor.execute(
                "SELECT entry_kind FROM public.lab_arena_ledger WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
            assert [row[0] for row in cursor.fetchall()] == ["reservation"]
        _rpc(connection, "lab_arena_abort_restart_guard_v1", GUARD, OWNER, 2, "test-abort")


@pytest.mark.parametrize("field,value", (
    ("lease_generation", 999),
    ("output_ref", "unexpected-output"),
    ("kind", "execute"),
))
def test_changed_captured_lease_fails_closed(installed, field, value):
    psycopg, dsn = installed
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        _lease(connection, overdue=True)
        with connection.cursor() as cursor:
            cursor.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control")
            prior_generation = cursor.fetchone()[0]
        generation = _acquire(connection, prior_generation)["guard_generation"]
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                f"UPDATE public.lab_arena_runs SET {field}=%s WHERE run_id=%s",
                (value, source.ASSIGNMENT + ":1"),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        if field == "lease_generation":
            assert _quiescence(connection, generation=generation)["preserved"] is False
        else:
            with pytest.raises(psycopg.Error, match="write set differs"):
                _quiescence(connection, generation=generation)
            connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM public.lab_arena_runs WHERE run_id=%s",
                (source.ASSIGNMENT + ":1",),
            )
            assert cursor.fetchone()[0] == "leased"
        _rpc(connection, "lab_arena_abort_restart_guard_v1", GUARD, OWNER,
             generation, "test-abort")
