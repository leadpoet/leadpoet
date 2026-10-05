"""The existing transition RPC alone gets the bounded publication window."""

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/410-lab-arena-publication-transition-timeout.sql"
).read_text()
SIGNATURE = "public.lab_arena_transition_round(text,text,text,jsonb)"


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _identity(cursor):
    cursor.execute(
        "SELECT md5(pg_get_functiondef(oid)), pg_get_userbyid(proowner), "
        "proacl::text, prosecdef, provolatile, proconfig "
        "FROM pg_proc WHERE oid=%s::regprocedure",
        (SIGNATURE,),
    )
    return cursor.fetchone()


def test_transition_timeout_is_exact_and_idempotent(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _identity(cursor)
        assert before[0] == "8cd26a7f0b9737160cce98438013dc91"
        assert before[-1] == ["search_path=pg_catalog, public"]
        cursor.execute(MIGRATION)
        after = _identity(cursor)
        assert after[0] == "02ed5004801fb47066f06623ae5abc7d"
        assert after[1:5] == before[1:5]
        assert after[-1] == ["search_path=pg_catalog, public", "statement_timeout=60s"]
        cursor.execute(MIGRATION)
        assert _identity(cursor) == after


def test_transition_timeout_refuses_unexpected_function_config(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "ALTER FUNCTION " + SIGNATURE + " SET statement_timeout = '61s'"
        )
        with pytest.raises(psycopg.Error, match="transition_timeout_410_preimage_changed"):
            cursor.execute(MIGRATION)


def test_transition_timeout_refuses_permission_drift(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("REVOKE EXECUTE ON FUNCTION " + SIGNATURE + " FROM lab_arena_service")
        with pytest.raises(psycopg.Error, match="transition_timeout_410_security_shape_changed"):
            cursor.execute(MIGRATION)
