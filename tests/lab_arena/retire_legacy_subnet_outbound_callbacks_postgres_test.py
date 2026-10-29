"""The exact callback retirement fails closed and preserves unrelated state."""
import hashlib
from pathlib import Path
import re

import pytest

from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


SQL = (Path(__file__).resolve().parents[2] /
       "scripts/retire_legacy_subnet_outbound_callbacks.sql").read_text()
EXPECTED = (
    ("public.process_consensus_async(uuid)", "p_prospect_id uuid",
     "cecf1696058702e956c6ec1b31c49235d49232fa68eea12c3771b43098729ddf"),
    ("public.process_rep_score_async(uuid)", "p_rep_score_pending_id uuid",
     "8af2094e7ac4fe8895067f4c6dd3b219d231364a684d072aebbbf0a02bbc508c"),
    ("public.cron_process_rep_scores()", "",
     "eaaaeac7c5d4d25056c2d5fb1f9c9d3a87de1eaf4b8efc0c135bb1c0101c0ca5"),
    ("public.process_pending_rep_scores()", "",
     "1671b69bf59c070b9ddcc2887d013bb84ca5dbbbb86292565205b2ed3938cb79"),
    ("public.process_pending_rep_scores_continuous()", "",
     "c7d8f01c38355266f3a489b3fe750fab4c066a1476fb5af9c9c5b1daae31226f"),
)
NAMES = [signature.split(".", 1)[1].split("(", 1)[0]
         for signature, _, _ in EXPECTED]


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(())


@pytest.fixture()
def connection(database):
    pg, dsn = database
    conn = pg.connect(**dsn)
    conn.autocommit = True
    with conn.cursor() as c:
        c.execute("DROP VIEW IF EXISTS public.callback_dependency")
        c.execute("DROP FUNCTION IF EXISTS public.callback_text_caller()")
        c.execute("DROP FUNCTION IF EXISTS " + ",".join(x[0] for x in EXPECTED))
        c.execute("DROP FUNCTION IF EXISTS public.process_consensus_async(text)")
        c.execute("DROP PUBLICATION IF EXISTS callback_test_publication")
        c.execute("DROP TABLE IF EXISTS public.callback_history")
        c.execute("CREATE TABLE public.callback_history(id int PRIMARY KEY, value text)")
        c.execute("INSERT INTO public.callback_history VALUES (1,'preserved')")
        c.execute("GRANT SELECT ON public.callback_history TO anon")
        c.execute("CREATE PUBLICATION callback_test_publication FOR TABLE public.callback_history")
        c.execute("CREATE OR REPLACE FUNCTION public.callback_current_api() RETURNS int "
                  "LANGUAGE sql STABLE AS 'SELECT 71'")
        c.execute("REVOKE ALL ON FUNCTION public.callback_current_api() FROM PUBLIC")
        c.execute("GRANT EXECUTE ON FUNCTION public.callback_current_api() TO service_role")
    try:
        yield conn, pg
    finally:
        with conn.cursor() as c:
            c.execute("ROLLBACK")
        conn.close()


def snapshot(conn):
    with conn.cursor() as c:
        c.execute("SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),"
                  "pg_get_userbyid(p.proowner),p.proacl::text,p.proconfig "
                  "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                  "WHERE n.nspname='public' ORDER BY 1")
        functions = c.fetchall()
        c.execute("SELECT * FROM public.callback_history ORDER BY id")
        rows = c.fetchall()
        c.execute("SELECT relacl::text FROM pg_class WHERE oid='public.callback_history'::regclass")
        acl = c.fetchall()
        c.execute("SELECT * FROM pg_publication_tables ORDER BY pubname,schemaname,tablename")
        publication = c.fetchall()
        c.execute("SELECT public.callback_current_api()")
        api_result = c.fetchone()
    return functions, rows, acl, publication, api_result


def install_synthetic_beforeimage(conn):
    """Only hashes change for safe no-op fixtures; production guards stay exact."""
    fixture_sql = SQL
    with conn.cursor() as c:
        for signature, arguments, expected_hash in EXPECTED:
            name = signature.split("(", 1)[0]
            c.execute(f"CREATE FUNCTION {name}({arguments}) RETURNS void "
                      "LANGUAGE plpgsql SECURITY DEFINER AS 'BEGIN RETURN; END'")
            c.execute(f"GRANT EXECUTE ON FUNCTION {signature} TO anon,authenticated,service_role")
            c.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            fixture_hash = hashlib.sha256(c.fetchone()[0].encode()).hexdigest()
            assert fixture_sql.count(expected_hash) == 1
            fixture_sql = fixture_sql.replace(expected_hash, fixture_hash)
    return fixture_sql


def rejected_unchanged(conn, pg, sql, match):
    before = snapshot(conn)
    with pytest.raises(pg.Error, match=match):
        with conn.cursor() as c:
            c.execute(sql)
    with conn.cursor() as c:
        c.execute("ROLLBACK")
    assert snapshot(conn) == before


def test_committed_manifest_and_drop_scope_are_exact():
    assert re.findall(r"\('([^']+)', '([^']*)',\s*'([0-9a-f]{64})'\)", SQL) == list(EXPECTED)
    drops = re.findall(r"DROP FUNCTION (.*?);", SQL, re.S)
    assert len(drops) == 1
    assert " ".join(drops[0].split()) == ", ".join(x[0] for x in EXPECTED) + " RESTRICT"
    assert "CASCADE" not in SQL


def test_all_absent_is_idempotent_and_preserves_unrelated_state(connection):
    conn, _ = connection
    before = snapshot(conn)
    for _ in range(2):
        with conn.cursor() as c:
            c.execute(SQL)
        assert snapshot(conn) == before


def test_all_five_removed_together_and_rerun_preserves_data_acl_api_publication(connection):
    conn, _ = connection
    sql = install_synthetic_beforeimage(conn)
    before = snapshot(conn)
    with conn.cursor() as c:
        c.execute(sql)
    after = snapshot(conn)
    assert len(before[0]) - len(after[0]) == 5
    assert [x for x in before[0] if x[0].split("(", 1)[0] not in NAMES] == after[0]
    assert before[1:] == after[1:]
    with conn.cursor() as c:
        c.execute(SQL)
    assert snapshot(conn) == after


def test_real_production_hashes_reject_noop_replacements(connection):
    conn, pg = connection
    install_synthetic_beforeimage(conn)
    rejected_unchanged(conn, pg, SQL, "identity differs")


@pytest.mark.parametrize("change,reason", [
    ("DROP FUNCTION public.process_rep_score_async(uuid)", "set differs"),
    ("CREATE FUNCTION public.process_consensus_async(text) RETURNS void "
     "LANGUAGE plpgsql AS 'BEGIN RETURN; END'", "set differs"),
    ("ALTER FUNCTION public.process_pending_rep_scores_continuous() SECURITY INVOKER", "identity differs"),
    ("ALTER FUNCTION public.process_pending_rep_scores_continuous() OWNER TO service_role", "identity differs"),
    ("REVOKE EXECUTE ON FUNCTION public.process_pending_rep_scores_continuous() FROM anon", "identity differs"),
    ("ALTER FUNCTION public.process_pending_rep_scores_continuous() SET search_path=public", "identity differs"),
    ("CREATE OR REPLACE FUNCTION public.process_pending_rep_scores_continuous() RETURNS void "
     "LANGUAGE plpgsql SECURITY DEFINER AS 'BEGIN RAISE EXCEPTION ''changed''; END'", "identity differs"),
    ("CREATE VIEW public.callback_dependency AS SELECT "
     "public.process_pending_rep_scores_continuous() IS NULL AS completed", "has a dependent"),
    ("CREATE FUNCTION public.callback_text_caller() RETURNS void LANGUAGE plpgsql "
     "AS 'BEGIN PERFORM public.process_pending_rep_scores_continuous(); END'", "stored routine caller"),
])
def test_unexpected_state_aborts_all_without_changing_history(connection, change, reason):
    conn, pg = connection
    sql = install_synthetic_beforeimage(conn)
    with conn.cursor() as c:
        c.execute(change)
    rejected_unchanged(conn, pg, sql, reason)
