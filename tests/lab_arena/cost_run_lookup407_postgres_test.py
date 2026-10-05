"""407 keeps every cost result while bounding run metadata lookups."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.successful_call_cost_postgres_test import _head, _seed


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/407-lab-arena-cost-run-lookup.sql"


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _costs(cursor, round_id, submission_id):
    cursor.execute(
        "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL), "
        "public.lab_arena__successful_call_cost_state(%s,'execute','openrouter'), "
        "public.lab_arena__successful_call_cost_state(%s,'score',NULL), "
        "public.lab_arena__successful_icp_cost_state(%s,%s,0), "
        "public.lab_arena__successful_icp_cost_state(%s,%s,1), "
        "public.lab_arena_submission_costs(%s)",
        (
            submission_id, submission_id, submission_id,
            round_id, submission_id, round_id, submission_id, submission_id,
        ),
    )
    return cursor.fetchone()


def test_cost_run_lookup_preserves_results_and_replays(database):
    connection, round_id, participants = _seed(database)
    participant = participants[0]
    try:
        with connection.cursor() as cursor:
            _head(cursor, round_id=round_id, participant=participant,
                  label="success", amount=250, succeeded=True)
            _head(cursor, round_id=round_id, participant=participant,
                  label="failed-charge", amount=450, succeeded=False)
            _head(cursor, round_id=round_id, participant=participant,
                  label="open", entry_kind="uncertain", amount=300)
            _head(cursor, round_id=round_id, participant=participant,
                  label="judge", kind="score", amount=125, succeeded=True)
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs "
                "WHERE submission_id=%s AND kind='execute' "
                "AND icp_position=1 ORDER BY run_id LIMIT 1",
                (participant["submission_id"],),
            )
            second_run = cursor.fetchone()[0]
            _head(cursor, round_id=round_id,
                  participant={**participant, "run_id": second_run},
                  label="next-icp", amount=750, succeeded=True)
            before = _costs(cursor, round_id, participant["submission_id"])
            assert before[3]["settled_microusd"] == 700
            assert before[4]["settled_microusd"] == 750
            migration = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration)
            assert _costs(cursor, round_id, participant["submission_id"]) == before
            cursor.execute(migration)
            assert _costs(cursor, round_id, participant["submission_id"]) == before
            for signature in (
                "lab_arena__successful_call_cost_state(text,text,text)",
                "lab_arena__successful_icp_cost_state(text,text,integer)",
                "lab_arena_submission_costs(text)",
            ):
                cursor.execute(
                    "SELECT p.prosecdef, p.provolatile, p.proowner::regrole::text "
                    "FROM pg_catalog.pg_proc AS p "
                    "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                    ("public." + signature,),
                )
                assert cursor.fetchone() == (True, "s", "lab_arena_owner")
    finally:
        connection.close()


def test_cost_run_lookup_refuses_changed_function_preimage(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena__successful_call_cost_state(text,text,text)'"
                "::pg_catalog.regprocedure)"
            )
            definition = cursor.fetchone()[0]
            changed = definition.replace("  FROM heads\n", "  FROM heads /* concurrent edit */\n")
            assert changed != definition
            cursor.execute(changed)
            with pytest.raises(psycopg2.Error, match="cost_407_call_preimage_changed"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            connection.rollback()
    finally:
        connection.close()
