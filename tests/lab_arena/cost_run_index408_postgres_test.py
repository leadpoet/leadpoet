"""408 keeps cost filters exact while indexing distinct ledger run IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.successful_call_cost_postgres_test import _head, _seed
from tests.lab_arena.test_lab_arena_migration_postgres import sha


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MIGRATION_407 = (SCRIPTS / "407-lab-arena-cost-run-lookup.sql").read_text()
MIGRATION_408 = (SCRIPTS / "408-lab-arena-cost-run-index.sql").read_text()


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _states(cursor, round_id, submission_id):
    cursor.execute(
        "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL), "
        "public.lab_arena__successful_call_cost_state(%s,'execute','openrouter'), "
        "public.lab_arena__successful_call_cost_state(%s,'execute','deepline'), "
        "public.lab_arena__successful_icp_cost_state(%s,%s,0), "
        "public.lab_arena_submission_costs(%s)",
        (submission_id, submission_id, submission_id,
         round_id, submission_id, submission_id),
    )
    return cursor.fetchone()


def _insert_outside(cursor, *, participant, round_id, label, provider, amount):
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response) "
        "VALUES ('settlement',%s,%s,%s,%s,1,%s,%s,'openrouter.chat',"
        "'miner_key',%s,%s::jsonb,%s::jsonb)",
        (participant["miner_hotkey"], round_id, participant["submission_id"],
         participant["run_id"], sha(label), provider, amount,
         json.dumps({}), json.dumps({"call_succeeded": True})),
    )


def test_cost_run_index_keeps_final_round_and_provider_filters(database):
    connection, round_id, participants = _seed(database)
    participant = participants[0]
    try:
        with connection.cursor() as cursor:
            _head(cursor, round_id=round_id, participant=participant,
                  label="in-scope", amount=250, succeeded=True)
            _insert_outside(cursor, participant=participant, round_id=round_id,
                            label="other-provider", provider="deepline", amount=999)
            _insert_outside(cursor, participant=participant,
                            round_id="arena-other-round", label="other-round",
                            provider="openrouter", amount=777)
            cursor.execute(MIGRATION_407)
            before = _states(cursor, round_id, participant["submission_id"])
            assert before[0]["settled_microusd"] == 2026
            assert before[1]["settled_microusd"] == 1027
            assert before[2]["settled_microusd"] == 999
            assert before[3]["settled_microusd"] == 1249
            cursor.execute(MIGRATION_408)
            assert _states(cursor, round_id, participant["submission_id"]) == before
            cursor.execute(MIGRATION_408)
            assert _states(cursor, round_id, participant["submission_id"]) == before
            cursor.execute(
                "SELECT pg_catalog.pg_get_indexdef(i.indexrelid), "
                "i.indisvalid, i.indisready "
                "FROM pg_catalog.pg_index AS i "
                "WHERE i.indexrelid='public.lab_arena_ledger_cost_run_idx'"
                "::pg_catalog.regclass"
            )
            indexdef, valid, ready = cursor.fetchone()
            assert indexdef == (
                "CREATE INDEX lab_arena_ledger_cost_run_idx ON "
                "public.lab_arena_ledger USING btree (submission_id, run_id) "
                "WHERE (call_identity IS NOT NULL)"
            )
            assert (valid, ready) == (True, True)
    finally:
        connection.close()


def test_cost_run_index_rejects_wrong_existing_index(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION_407)
            cursor.execute(
                "CREATE INDEX lab_arena_ledger_cost_run_idx "
                "ON public.lab_arena_ledger (submission_id)"
            )
            with pytest.raises(psycopg2.Error, match="cost_408_index_shape_invalid"):
                cursor.execute(MIGRATION_408)
            connection.rollback()
    finally:
        connection.close()
