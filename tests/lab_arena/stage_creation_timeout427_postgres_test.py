"""Only execution and scoring queue creation get the bounded RPC timeout."""

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/427-lab-arena-stage-creation-timeout.sql"
).read_text()
TARGETS = {
    "public.lab_arena_open_stage(text,smallint,jsonb,integer[])": (
        "3ca058cd605fdd4f26d185a420b5b791", "6e62319fbc2f3b40eb9902f51463018a",
        "af0b332658d8b2bf2d1a8500cc662264",
    ),
    "public.lab_arena_open_scoring_v3(text,smallint,jsonb)": (
        "299f54f7517e4ca694071a7e76e7008a", "53c211e88cb553e1421a84fe86c3c13c",
        "62c9e563ea5da26ed3a1e2ab9f564f6e",
    ),
}


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _identity(cursor, signature):
    cursor.execute(
        "SELECT md5(pg_get_functiondef(oid)), md5(prosrc), "
        "pg_get_userbyid(proowner), proacl::text, prosecdef, provolatile, proconfig "
        "FROM pg_proc WHERE oid=%s::regprocedure",
        (signature,),
    )
    return cursor.fetchone()


def _identities(cursor):
    return {signature: _identity(cursor, signature) for signature in TARGETS}


def test_stage_creation_timeout_is_exact_and_idempotent(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _identities(cursor)
        for signature, (preimage, _postimage, body) in TARGETS.items():
            assert before[signature] == (
                preimage, body, "lab_arena_owner",
                "{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",
                True, "v", ["search_path=pg_catalog, public"],
            )
        cursor.execute("SHOW statement_timeout")
        session_timeout = cursor.fetchone()
        # Capture all other Arena function definitions to prove the scope.
        cursor.execute(
            "SELECT oid, md5(pg_get_functiondef(oid)) FROM pg_proc "
            "WHERE pronamespace='public'::regnamespace "
            "AND proname LIKE 'lab_arena_%%' AND oid NOT IN (%s::regprocedure, %s::regprocedure) "
            "ORDER BY oid", tuple(TARGETS),
        )
        other_functions = cursor.fetchall()
        cursor.execute(MIGRATION)
        after = _identities(cursor)
        for signature, (_preimage, postimage, _body) in TARGETS.items():
            assert after[signature] == (
                postimage, *before[signature][1:-1],
                ["search_path=pg_catalog, public", "statement_timeout=60s"],
            )
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE'), "
                "has_function_privilege('anon', %s, 'EXECUTE')", (signature, signature),
            )
            assert cursor.fetchone() == (True, False)
        cursor.execute(
            "SELECT oid, md5(pg_get_functiondef(oid)) FROM pg_proc "
            "WHERE pronamespace='public'::regnamespace "
            "AND proname LIKE 'lab_arena_%%' AND oid NOT IN (%s::regprocedure, %s::regprocedure) "
            "ORDER BY oid", tuple(TARGETS),
        )
        assert cursor.fetchall() == other_functions
        cursor.execute("SHOW statement_timeout")
        assert cursor.fetchone() == session_timeout
        cursor.execute(MIGRATION)
        assert _identities(cursor) == after


@pytest.mark.parametrize("signature", TARGETS)
@pytest.mark.parametrize("setting", [
    "statement_timeout = '61s'", "lock_timeout = '1s'",
    "search_path = public, pg_catalog",
])
def test_unexpected_config_aborts_without_changing_function(database, signature, setting):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("ALTER FUNCTION " + signature + " SET " + setting)
        connection.commit()
        before = _identities(cursor)
        with pytest.raises(psycopg.Error, match="stage_creation_timeout_427_configuration_changed"):
            cursor.execute(MIGRATION)
        connection.rollback()
        assert _identities(cursor) == before


@pytest.mark.parametrize("signature", TARGETS)
@pytest.mark.parametrize("permission_sql", [
    "REVOKE EXECUTE ON FUNCTION {signature} FROM lab_arena_service",
    "GRANT EXECUTE ON FUNCTION {signature} TO PUBLIC",
])
def test_permission_drift_aborts_without_changing_function(database, signature, permission_sql):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(permission_sql.format(signature=signature))
        connection.commit()
        before = _identities(cursor)
        with pytest.raises(psycopg.Error, match="stage_creation_timeout_427_security_shape_changed"):
            cursor.execute(MIGRATION)
        connection.rollback()
        assert _identities(cursor) == before


@pytest.mark.parametrize("signature", TARGETS)
def test_existing_body_is_preserved(database, signature):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
        definition = cursor.fetchone()[0]
        assert "BEGIN\n" in definition
        cursor.execute(definition.replace("BEGIN\n", "BEGIN\n  -- unexpected body drift\n", 1))
        connection.commit()
        before = _identities(cursor)
        assert before[signature][1] != TARGETS[signature][2]
        cursor.execute(MIGRATION)
        after = _identities(cursor)
        # The migration changes the timeout and retains the existing body,
        # including branches added by historical round recovery migrations.
        for target in TARGETS:
            assert after[target][1:-1] == before[target][1:-1]
            assert after[target][-1] == [
                "search_path=pg_catalog, public", "statement_timeout=60s",
            ]
        cursor.execute(MIGRATION)
        assert _identities(cursor) == after
