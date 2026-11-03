"""Modern readers on historical fixtures retain the original write authority."""

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
    prepare_current_service_reads,
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _functions(cursor):
    cursor.execute(
        "SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),p.proacl::text "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND p.proname LIKE 'lab_arena_%' ORDER BY 1"
    )
    return {row[0]: row[1:] for row in cursor.fetchall()}


def test_read_dependencies_enable_current_queries_without_replacing_historical_writes(database):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    try:
        with pytest.raises(ArenaStoreError, match='lab_arena_execution_attempt_count'):
            store.list_runs('arena-2099-01-01', columns='*,execution_attempt:lab_arena_execution_attempt_count')
        with pytest.raises(ArenaStoreError, match='scoring_generation_floor'):
            store.get_published_results_round('arena-2099-01-01')
        with connect() as connection, connection.cursor() as cursor:
            before = _functions(cursor)
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            schema_acl = cursor.fetchone()
            cursor.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname")
            indexes = cursor.fetchall()

        prepare_current_service_reads(connect)
        assert store.list_runs('arena-2099-01-01', columns='*,execution_attempt:lab_arena_execution_attempt_count') == []
        assert store.get_published_results_round('arena-2099-01-01') is None
        assert store.get_public_submissions_round('arena-2099-01-01') is None
        with connect() as connection, connection.cursor() as cursor:
            after = _functions(cursor)
            assert {signature: after[signature] for signature in before} == before
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone() == schema_acl
            cursor.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname")
            assert [row for row in cursor.fetchall() if row in indexes] == indexes
            cursor.execute("SELECT to_regprocedure('public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text)')")
            assert cursor.fetchone()[0] is None
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='public.lab_arena_runs'::regclass "
                           "AND attname='scoring_generation_floor' AND NOT attisdropped)")
            assert cursor.fetchone() == (False,)
        # Never replace an installed reader with the older introduction body.
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("COMMENT ON FUNCTION public.lab_arena_execution_attempt_count(public.lab_arena_runs) "
                           "IS 'retained later deployment'")
        prepare_current_service_reads(connect)
        with connect() as connection, connection.cursor() as cursor:
            assert _functions(cursor) == after
            cursor.execute("SELECT obj_description('public.lab_arena_execution_attempt_count(public.lab_arena_runs)'::regprocedure)")
            assert cursor.fetchone() == ('retained later deployment',)
    finally:
        store.close()
