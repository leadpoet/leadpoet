"""Publication cost summaries share one authoritative cost document."""

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/409-lab-arena-publication-cost-document-reuse.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _definition(cursor, signature):
    cursor.execute("SELECT pg_catalog.pg_get_functiondef(%s::regprocedure)", (signature,))
    return cursor.fetchone()[0]


def _identity(cursor, signature):
    cursor.execute(
        "SELECT pg_get_userbyid(proowner),proacl,prosecdef,provolatile,proconfig "
        "FROM pg_proc WHERE oid=%s::regprocedure",
        (signature,),
    )
    return cursor.fetchone()


def test_migration_is_exact_idempotent_and_preserves_security(database):
    psycopg2, dsn = database
    wrapper = "public.lab_arena__cost_kind_summary_v1(text,text)"
    guard = "public.lab_arena__per_icp_publication_valid(text,jsonb)"
    helper = "public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        before = {name: (_definition(cursor, name), _identity(cursor, name))
                  for name in (wrapper, guard)}
        cursor.execute(MIGRATION.read_text())
        after = {name: (_definition(cursor, name), _identity(cursor, name))
                 for name in (wrapper, guard)}
        assert after[wrapper][1] == before[wrapper][1]
        assert after[guard][1] == before[guard][1]
        assert after[wrapper][0] != before[wrapper][0]
        assert after[guard][0] != before[guard][0]
        assert _identity(cursor, helper)[0] == "lab_arena_owner"
        assert _identity(cursor, helper) == before[wrapper][1]
        cursor.execute(
            "SELECT has_function_privilege('anon',%s,'EXECUTE'),"
            "has_function_privilege('service_role',%s,'EXECUTE')",
            (helper, helper),
        )
        assert cursor.fetchone() == (False, False)
        cursor.execute(MIGRATION.read_text())
        assert {name: (_definition(cursor, name), _identity(cursor, name))
                for name in (wrapper, guard)} == after


def test_non_superuser_migration_restores_public_schema_acl():
    # Supabase's migration role can own the schema but cannot bypass the
    # target function owner's missing CREATE permission.
    database = database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)
    psycopg2, dsn = next(database)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE ROLE arena_migration_test NOLOGIN")
            cursor.execute("GRANT lab_arena_owner TO arena_migration_test")
            cursor.execute("ALTER SCHEMA public OWNER TO arena_migration_test")
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
            cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
            original_acl = cursor.fetchone()[0]
            cursor.execute("SET ROLE arena_migration_test")
            cursor.execute("BEGIN")
            cursor.execute(
                "CREATE FUNCTION public.arena_409_permission_probe() "
                "RETURNS integer LANGUAGE sql AS 'SELECT 1'"
            )
            with pytest.raises(psycopg2.Error) as denied:
                cursor.execute(
                    "ALTER FUNCTION public.arena_409_permission_probe() "
                    "OWNER TO lab_arena_owner"
                )
            assert denied.value.pgcode == "42501"
            cursor.execute("ROLLBACK")

            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == original_acl
            cursor.execute(
                "SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),"
                "pg_get_userbyid(proowner) FROM pg_proc WHERE oid=%s::regprocedure",
                ("public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)",),
            )
            assert cursor.fetchone() == (False, "lab_arena_owner")
    finally:
        connection.close()
        database.close()


@pytest.mark.parametrize("kind", ["execute", "score", "unknown", None])
def test_helper_matches_original_aggregation_and_rejects_tampered_costs(database, kind):
    psycopg2, dsn = database
    rows = [
        dict(kind="execute", provider="source", settled_microusd=17,
             reserved_or_uncertain_microusd=3, inflight_calls=1,
             uncertain_calls=0, refused_calls=0, call_count=2,
             successful_microusd=17, successful_calls=1,
             success_unresolved_microusd=0, success_unresolved_calls=0),
        dict(kind="score", provider="judge", settled_microusd=31,
             reserved_or_uncertain_microusd=0, inflight_calls=0,
             uncertain_calls=0, refused_calls=1, call_count=3,
             successful_microusd=31, successful_calls=2,
             success_unresolved_microusd=0, success_unresolved_calls=0),
    ]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(
            "SELECT public.lab_arena__cost_kind_summary_from_doc_v1(%s::jsonb,%s)",
            (json.dumps({"providers": rows}), kind),
        )
        actual = cursor.fetchone()[0]
        if kind == "execute":
            assert actual["settled_microusd"] == 17
            assert actual["conservative_microusd"] == 20
            assert actual["providers"][0]["provider"] == "source"
        elif kind == "score":
            assert actual["settled_microusd"] == 31
            assert actual["refused_calls"] == 1
            assert actual["providers"][0]["provider"] == "judge"
        else:
            assert actual["settled_microusd"] == 0
            assert actual["providers"] == []
        if kind in ("execute", "score"):
            forged = dict(actual)
            forged["settled_microusd"] += 1
            cursor.execute("SELECT %s::jsonb IS DISTINCT FROM %s::jsonb",
                           (json.dumps(forged), json.dumps(actual)))
            assert cursor.fetchone()[0] is True


def test_missing_provider_array_keeps_original_null_behavior(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(
            "SELECT public.lab_arena__cost_kind_summary_from_doc_v1(%s::jsonb,'score')",
            (json.dumps({}),),
        )
        result = cursor.fetchone()[0]
        assert result["settled_microusd"] == 0
        assert result["providers"] == []
