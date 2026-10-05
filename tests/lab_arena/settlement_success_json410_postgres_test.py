"""410 decodes settlement success once without changing cost decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.successful_call_cost_postgres_test import _seed
from tests.lab_arena.test_lab_arena_migration_postgres import sha


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MIGRATIONS = tuple(
    (SCRIPTS / name).read_text(encoding="utf-8")
    for name in (
        "407-lab-arena-cost-run-lookup.sql",
        "408-lab-arena-cost-run-index.sql",
        "410-lab-arena-settlement-success-json-once.sql",
    )
)


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _entry(cursor, participant, round_id, label, entry_kind, amount, *,
           terminal=None, entry_doc=None, provider="openrouter", run_id=None):
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response) "
        "VALUES (%s,%s,%s,%s,%s,1,%s,%s,'openrouter.chat','miner_key',"
        "%s,%s::jsonb,%s::jsonb) RETURNING entry_id",
        (entry_kind, participant["miner_hotkey"], round_id,
         participant["submission_id"], run_id or participant["run_id"],
         sha("410-" + label), provider, amount, json.dumps(entry_doc or {}),
         json.dumps(terminal) if terminal is not None else None),
    )
    return cursor.fetchone()[0]


def _states(cursor, round_id, submission_id):
    cursor.execute(
        "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL), "
        "public.lab_arena__successful_call_cost_state(%s,'execute','openrouter'), "
        "public.lab_arena__successful_call_cost_state(%s,'score',NULL), "
        "public.lab_arena__successful_icp_cost_state(%s,%s,0), "
        "public.lab_arena__successful_icp_cost_state(%s,%s,1), "
        "public.lab_arena_submission_costs(%s)",
        (submission_id, submission_id, submission_id,
         round_id, submission_id, round_id, submission_id, submission_id),
    )
    return cursor.fetchone()


def _function_identity(cursor):
    cursor.execute(
        "SELECT p.proname, owner.rolname, p.proacl::text, p.prosecdef, "
        "p.provolatile, p.proconfig "
        "FROM pg_catalog.pg_proc AS p "
        "JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner "
        "WHERE p.oid IN ("
        "'public.lab_arena__successful_call_cost_state(text,text,text)'"
        "::pg_catalog.regprocedure,"
        "'public.lab_arena__successful_icp_cost_state(text,text,integer)'"
        "::pg_catalog.regprocedure) ORDER BY p.proname"
    )
    return cursor.fetchall()


def test_settlement_success_json_once_matches_all_cost_states(database):
    connection, round_id, participants = _seed(database)
    participant = participants[0]
    try:
        with connection.cursor() as cursor:
            _entry(cursor, participant, round_id, "true", "settlement", 100,
                   terminal={"call_succeeded": True})
            _entry(cursor, participant, round_id, "false", "settlement", 200,
                   terminal={"call_succeeded": False})
            _entry(cursor, participant, round_id, "string", "settlement", 300,
                   terminal={"call_succeeded": "true"})
            _entry(cursor, participant, round_id, "object", "settlement", 30,
                   terminal={"call_succeeded": {"wrong": True}})
            _entry(cursor, participant, round_id, "missing", "settlement", 50,
                   terminal={})
            _entry(cursor, participant, round_id, "null", "settlement", 60)
            _entry(cursor, participant, round_id, "json-null", "settlement", 70,
                   terminal={"call_succeeded": None})
            _entry(cursor, participant, round_id, "cancelled", "settlement", 400,
                   terminal={"call_succeeded": True}, entry_doc={
                       "late_reconciliation": "true",
                       "reconciled_uncertainty_reason": "round_cancelled",
                   })
            _entry(cursor, participant, round_id, "short-circuit-false",
                   "settlement", 40, terminal={"call_succeeded": False},
                   entry_doc={
                       "openrouter_delayed_reconciliation": "true",
                       "reconciled_uncertainty_entry_id": "not-an-integer",
                   })
            prior = _entry(cursor, participant, round_id, "delayed", "uncertain", 0,
                           entry_doc={"call": {"call_succeeded": True}})
            _entry(cursor, participant, round_id, "delayed", "settlement", 150,
                   terminal={"call_succeeded": "wrong-type"}, entry_doc={
                       "openrouter_delayed_reconciliation": "true",
                       "reconciled_uncertainty_entry_id": str(prior),
                   })
            _entry(cursor, participant, round_id, "uncertain-true", "uncertain", 80,
                   entry_doc={"call": {"call_succeeded": True}})
            _entry(cursor, participant, round_id, "uncertain-false", "uncertain", 90,
                   entry_doc={"call": {"call_succeeded": False}})
            _entry(cursor, participant, "arena-other-round", "foreign-round",
                   "settlement", 700, terminal={"call_succeeded": True},
                   provider="deepline")
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs "
                "WHERE submission_id=%s AND kind='execute' AND icp_position=1 "
                "ORDER BY run_id LIMIT 1", (participant["submission_id"],),
            )
            other_run = cursor.fetchone()[0]
            _entry(cursor, participant, round_id, "other-icp", "settlement", 35,
                   terminal={"call_succeeded": True}, run_id=other_run)
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,status,stage_generation,kind,scored_run_id) "
                "VALUES (%s,%s,%s,%s,%s,1,0,1,'pending',1,'score',%s)",
                (round_id + ':score:410', round_id + ':assignment:410', round_id,
                 participant["submission_id"], participant["miner_hotkey"],
                 participant["run_id"]),
            )
            _entry(cursor, participant, round_id, "score", "settlement", 20,
                   terminal={"call_succeeded": True},
                   run_id=round_id + ':score:410')
            cursor.execute(MIGRATIONS[0])
            cursor.execute(MIGRATIONS[1])
            before = _states(cursor, round_id, participant["submission_id"])
            before_identity = _function_identity(cursor)
            assert before[1]["settled_microusd"] == 1435
            assert before[1]["successful_microusd"] == 285
            assert before[1]["success_unresolved_microusd"] == 590
            assert before[3]["settled_microusd"] == 1400
            assert before[4]["settled_microusd"] == 35
            cursor.execute(MIGRATIONS[2])
            assert _states(cursor, round_id, participant["submission_id"]) == before
            assert _function_identity(cursor) == before_identity
            cursor.execute(MIGRATIONS[2])
            assert _states(cursor, round_id, participant["submission_id"]) == before
            assert _function_identity(cursor) == before_identity
    finally:
        connection.close()


def test_settlement_success_json_once_rejects_changed_preimage(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATIONS[0])
            cursor.execute(MIGRATIONS[1])
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena__successful_call_cost_state(text,text,text)'"
                "::pg_catalog.regprocedure)"
            )
            definition = cursor.fetchone()[0]
            changed = definition.replace("  FROM heads\n", "  FROM heads /* changed */\n")
            assert changed != definition
            cursor.execute(changed)
            with pytest.raises(psycopg2.Error, match="cost_410_call_preimage_changed"):
                cursor.execute(MIGRATIONS[2])
            connection.rollback()
    finally:
        connection.close()
