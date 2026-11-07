"""Sparse startup lookup, unchanged metadata, and concurrent native index build."""
from pathlib import Path
import re
import threading
import time

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/661-lab-arena-trajectory-kind-run-index.sql"
DDL = MIGRATION.read_text()
INDEX = "lab_arena_trajectory_kind_run_idx"
EXPECTED = ("CREATE INDEX " + INDEX + " ON public.lab_arena_trajectory_events USING btree "
            "(event_kind, run_id)")

# The production PostgREST statement, including bound JSON keys and event kind.
QUERY = """WITH pgrst_source AS (
 SELECT trajectory_id,run_id,submission_id,runner_hotkey,assignment_id,attempt,
 content->>$1 AS source_commit,content->>$2 AS source_dirty,
 content->>$3 AS start_lease_generation
 FROM public.lab_arena_trajectory_events
 WHERE round_id=$4 AND event_kind=$5 AND run_id=ANY($6)
 ORDER BY trajectory_id ASC LIMIT $7 OFFSET $8
) SELECT $9::bigint AS total_result_set,count(_postgrest_t) AS page_total,
 coalesce(json_agg(_postgrest_t),$10) AS body,
 nullif(current_setting($11,$12),$13) AS response_headers,
 nullif(current_setting($14,$15),$16) AS response_status,
 $17 AS response_inserted FROM pgrst_source _postgrest_t"""
TYPES = "text,text,text,text,text,text[],bigint,bigint,bigint,json,text,boolean,text,text,boolean,text,bigint"


def parameters(ids, *, kind="runtime.started", offset=0, round_id="arena-2026-11-03"):
    return ("validator_source_commit", "validator_source_dirty", "lease_generation",
            round_id, kind, ids, 500, offset, None, "[]", "response.headers", True,
            "", "response.status", True, "", None)


def nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from nodes(child)


def assert_index(cursor):
    cursor.execute("SELECT i.indisvalid,i.indisready,i.indisunique,"
                   "pg_get_userbyid(c.relowner),pg_get_indexdef(c.oid) "
                   "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                   "WHERE c.oid=to_regclass(%s)", ("public." + INDEX,))
    assert cursor.fetchone() == (True, True, False, "lab_arena_owner", EXPECTED)


def security(cursor):
    cursor.execute("SELECT relowner,relacl,relrowsecurity,relforcerowsecurity "
                   "FROM pg_class WHERE oid='public.lab_arena_trajectory_events'::regclass")
    relation = cursor.fetchone()
    cursor.execute("SELECT policyname,roles,cmd,qual,with_check FROM pg_policies "
                   "WHERE schemaname='public' AND tablename='lab_arena_trajectory_events'")
    return relation, cursor.fetchall()


def fingerprint(cursor):
    cursor.execute("SELECT count(*),max(trajectory_id),sum(trajectory_id) "
                   "FROM public.lab_arena_trajectory_events")
    return cursor.fetchone()


def measure(cursor, mode, ids):
    cursor.execute("SET plan_cache_mode=" + mode)
    cursor.execute("SET ROLE lab_arena_service")
    cursor.execute("PREPARE startup_lookup(" + TYPES + ") AS " + QUERY)
    execute = "EXECUTE startup_lookup(" + ",".join(["%s"] * 17) + ")"
    results = []
    for args in (parameters(ids), parameters(ids, offset=500),
                 parameters(ids, kind="provider.received"),
                 parameters(ids, round_id="absent"), parameters([])):
        cursor.execute(execute, args)
        results.append(cursor.fetchone())
    if mode == "auto":
        for _ in range(8):
            cursor.execute(execute, parameters(ids))
            cursor.fetchone()
    cursor.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) " + execute,
                   parameters(ids))
    plan = cursor.fetchone()[0][0]
    cursor.execute("SELECT generic_plans,custom_plans FROM pg_prepared_statements "
                   "WHERE name='startup_lookup'")
    cached_plans = cursor.fetchone()
    cursor.execute("DEALLOCATE startup_lookup")
    cursor.execute("RESET ROLE")
    return results, plan, cached_plans


def append_provider(cursor, run_id, label):
    cursor.execute("INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
        "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,"
        "event_kind,occurred_at,content) SELECT run_id,md5(%s)::uuid,round_id,"
        "submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,"
        "icp_position,attempt,run_kind,model_role,'provider.concurrent_test',"
        "occurred_at,'{}'::jsonb FROM public.lab_arena_trajectory_events "
        "WHERE run_id=%s AND event_kind='runtime.started' LIMIT 1 RETURNING trajectory_id",
        (label, run_id))
    return cursor.fetchone()[0]


@pytest.fixture(scope="module")
def proof():
    generator = database_with_lab_arena_migration(())
    pg, dsn = next(generator)
    connection = pg.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE ROLE lab_arena_owner NOLOGIN;"
                           "CREATE ROLE lab_arena_service NOLOGIN;"
                           "CREATE TABLE public.lab_arena_runs(run_id text PRIMARY KEY)")
            # Use the real trajectory schema, four indexes, owner, ACL and RLS.
            # The parent identity table is sufficient for this index-only test.
            original = (ROOT / "scripts/365-lab-arena-trajectories.sql").read_text()
            schema = original[original.index("CREATE TABLE IF NOT EXISTS"):
                original.index("CREATE OR REPLACE FUNCTION public.lab_arena_append_trajectory_events_v1")]
            cursor.execute("GRANT CREATE ON SCHEMA public TO lab_arena_owner")
            cursor.execute(schema)
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
            cursor.execute("INSERT INTO public.lab_arena_runs SELECT 'arena-2026-11-'||"
                "CASE WHEN s<=250 THEN '03' ELSE '02' END||':sub-'||lpad(s::text,32,'0')"
                "||':2:0:1' FROM generate_series(1,500) s")
            fields = ("run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
                      "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,"
                      "model_role,event_kind,occurred_at,content")
            cursor.execute("INSERT INTO public.lab_arena_trajectory_events(" + fields + ") "
                "SELECT r.run_id,md5(r.run_id||':'||s::text)::uuid,split_part(r.run_id,':',1),"
                "split_part(r.run_id,':',2),'miner','runner',r.run_id,'icp0',2,0,1,'execute',"
                "'miner','provider.received','2026-11-03Z'::timestamptz,"
                "jsonb_build_object('payload',repeat('p',700),'validator_source_commit',"
                "repeat('b',40),'validator_source_dirty','dirty','lease_generation',99) "
                "FROM public.lab_arena_runs r CROSS JOIN generate_series(1,400) s")
            cursor.execute("INSERT INTO public.lab_arena_trajectory_events(" + fields + ") "
                "SELECT r.run_id,md5(r.run_id||':start')::uuid,split_part(r.run_id,':',1),"
                "split_part(r.run_id,':',2),'miner','runner',r.run_id,'icp0',2,0,1,'execute',"
                "'miner','runtime.started','2026-11-03Z'::timestamptz,"
                "jsonb_build_object('validator_source_commit',repeat('a',40),"
                "'validator_source_dirty','clean','lease_generation',1) "
                "FROM public.lab_arena_runs r")
            # Same run ID with a wrong round or event kind must not supply the
            # startup identity, even when its index keys otherwise match.
            for label, round_id, kind in (("wrong-round", "wrong-round", "runtime.started"),
                                           ("wrong-kind", "arena-2026-11-03", "runtime.finished")):
                cursor.execute("INSERT INTO public.lab_arena_trajectory_events(" + fields + ") "
                    "SELECT run_id,md5(run_id||%s)::uuid,%s,submission_id,miner_hotkey,"
                    "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,"
                    "run_kind,model_role,%s,occurred_at,jsonb_build_object("
                    "'validator_source_commit',repeat('c',40),'validator_source_dirty','dirty',"
                    "'lease_generation',99) FROM public.lab_arena_trajectory_events "
                    "WHERE round_id='arena-2026-11-03' AND event_kind='runtime.started' "
                    "ORDER BY trajectory_id LIMIT 1", (label, round_id, kind))
            cursor.execute("VACUUM(ANALYZE) public.lab_arena_trajectory_events")
            cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE run_id LIKE "
                           "'arena-2026-11-03:%' ORDER BY run_id LIMIT 25")
            ids = [row[0] for row in cursor.fetchall()]
            before_security, before_rows = security(cursor), fingerprint(cursor)
            modes = ("force_custom_plan", "force_generic_plan", "auto")
            before = {mode: measure(cursor, mode, ids) for mode in modes}
            cursor.execute("SET plan_cache_mode=auto;SET ROLE lab_arena_service")
            cursor.execute("PREPARE startup_retained(" + TYPES + ") AS " + QUERY)
            retained_execute = "EXECUTE startup_retained(" + ",".join(["%s"] * 17) + ")"
            for _ in range(12):
                cursor.execute(retained_execute, parameters(ids))
                retained_before = cursor.fetchone()
            cursor.execute("SELECT generic_plans,custom_plans FROM pg_prepared_statements "
                           "WHERE name='startup_retained'")
            retained_before_stats = cursor.fetchone()
            assert retained_before_stats[0] > 0
            cursor.execute("RESET ROLE")

        # A prior writer makes the concurrent build wait. A second writer and
        # existing readers must still finish before that prior writer commits.
        held = pg.connect(**dsn)
        thread_result = {}
        try:
            with held.cursor() as writer:
                first = append_provider(writer, ids[0], "held-runtime-index661")
            def build():
                build_connection = pg.connect(**dsn)
                build_connection.autocommit = True
                try:
                    with build_connection.cursor() as builder:
                        builder.execute("SET statement_timeout='15s'")
                        builder.execute(DDL)
                    thread_result["complete"] = True
                except Exception as exc:
                    thread_result["error"] = type(exc).__name__
                finally:
                    build_connection.close()
            worker = threading.Thread(target=build)
            worker.start()
            deadline = time.monotonic() + 10
            waiting = False
            with connection.cursor() as cursor:
                while time.monotonic() < deadline:
                    cursor.execute("SELECT phase FROM pg_stat_progress_create_index "
                        "WHERE relid='public.lab_arena_trajectory_events'::regclass")
                    waiting = any(row[0].startswith("waiting for writers") for row in cursor.fetchall())
                    if waiting or not worker.is_alive():
                        break
                    time.sleep(0.01)
                assert waiting, thread_result
            with pg.connect(**dsn) as other, other.cursor() as writer:
                writer.execute("SET LOCAL lock_timeout='1s';SET LOCAL statement_timeout='2s'")
                second = append_provider(writer, ids[1], "other-runtime-index661")
                writer.execute("SET LOCAL ROLE lab_arena_service")
                writer.execute("SELECT count(*) FROM public.lab_arena_trajectory_events "
                               "WHERE round_id='arena-2026-11-03' "
                               "AND event_kind='runtime.started' AND run_id=ANY(%s)", (ids,))
                assert writer.fetchone()[0] == 25
            assert worker.is_alive()  # Reader and second writer did not wait for the build.
            held.commit()
            worker.join(timeout=10)
            assert not worker.is_alive() and thread_result == {"complete": True}
        finally:
            held.rollback()
            held.close()
        with connection.cursor() as cursor:
            assert_index(cursor)
            assert security(cursor) == before_security
            count, maximum, total = before_rows
            assert fingerprint(cursor) == (count + 2, max(maximum, first, second), total + first + second)
            # Do not deallocate or re-prepare: existing PostgREST connections
            # can retain a statement that already chose generic plans.
            cursor.execute("SET ROLE lab_arena_service")
            cursor.execute(retained_execute, parameters(ids))
            retained_after = cursor.fetchone()
            cursor.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) "
                           + retained_execute, parameters(ids))
            retained_plan = cursor.fetchone()[0][0]
            cursor.execute("SELECT generic_plans,custom_plans FROM pg_prepared_statements "
                           "WHERE name='startup_retained'")
            retained_after_stats = cursor.fetchone()
            cursor.execute("DEALLOCATE startup_retained;RESET ROLE")
            after = {mode: measure(cursor, mode, ids) for mode in modes}
            cursor.execute(DDL)  # Exact replay keeps the valid index.
            assert_index(cursor)
            assert security(cursor) == before_security
        yield {"pg": pg, "dsn": dsn, "ids": ids, "before": before,
               "after": after, "writer_and_reader_survived": True,
               "retained_before": retained_before, "retained_after": retained_after,
               "retained_before_stats": retained_before_stats,
               "retained_after_stats": retained_after_stats, "retained_plan": retained_plan}
    finally:
        connection.close()
        generator.close()


def test_migration_is_one_concurrent_statement():
    sql = re.sub(r"--[^\n]*", "", DDL).strip()
    assert sql.count(";") == 1
    assert sql.startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS " + INDEX)
    assert "INCLUDE" not in sql and "content" not in sql


@pytest.mark.parametrize("mode", ["force_custom_plan", "force_generic_plan", "auto"])
def test_exact_query_results_and_metadata_are_unchanged(proof, mode):
    before, after = proof["before"][mode], proof["after"][mode]
    assert before[0] == after[0]
    assert after[0][0][1] == 25 and after[0][1][1] == 0
    assert after[0][2][1] == 500 and after[0][3][1] == after[0][4][1] == 0
    assert all(row["source_commit"] == "a" * 40 and row["source_dirty"] == "clean"
               and row["start_lease_generation"] == "1" for row in after[0][0][2])


@pytest.mark.parametrize("mode", ["force_custom_plan", "force_generic_plan", "auto"])
def test_startup_lookup_reads_fewer_provider_pages(proof, mode):
    before, after = proof["before"][mode][1], proof["after"][mode][1]
    assert any(node.get("Index Name") == INDEX for node in nodes(after["Plan"]))
    def buffers(plan):
        return plan["Plan"].get("Shared Hit Blocks", 0) + plan["Plan"].get("Shared Read Blocks", 0)
    assert buffers(after) < buffers(before) / 20


def test_forced_generic_plan_uses_the_narrow_index(proof):
    plan = proof["after"]["force_generic_plan"][1]
    assert any(node.get("Index Name") == INDEX for node in nodes(plan["Plan"]))
    assert proof["writer_and_reader_survived"]


def test_retained_auto_statement_uses_index_without_reset(proof):
    assert proof["retained_before"] == proof["retained_after"]
    assert proof["retained_before_stats"][0] > 0
    assert proof["retained_after_stats"][0] > proof["retained_before_stats"][0]
    assert any(node.get("Index Name") == INDEX for node in nodes(proof["retained_plan"]["Plan"])), {
        "before_plans": proof["retained_before_stats"],
        "after_plans": proof["retained_after_stats"],
        "milliseconds": proof["retained_plan"]["Execution Time"],
        "buffer_blocks": proof["retained_plan"]["Plan"].get("Shared Hit Blocks", 0)
                         + proof["retained_plan"]["Plan"].get("Shared Read Blocks", 0),
    }


def test_native_store_returns_exact_startup_identity(proof):
    pg, dsn = proof["pg"], proof["dsn"]
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    try:
        rows = ArenaStore(transport).list_runtime_starts("arena-2026-11-03", run_ids=proof["ids"])
        assert len(rows) == 25
        assert all(row["source_commit"] == "a" * 40 and row["source_dirty"] == "clean"
                   and row["start_lease_generation"] == "1" for row in rows)
    finally:
        transport.close()


@pytest.mark.parametrize("drift", ["invalid", "unready", "owner", "definition", "predicate"])
def test_required_readback_refuses_wrong_existing_index(proof, drift):
    with proof["pg"].connect(**proof["dsn"]) as connection, connection.cursor() as cursor:
        if drift in {"invalid", "unready"}:
            column = "indisvalid" if drift == "invalid" else "indisready"
            cursor.execute("UPDATE pg_index SET " + column + "=false WHERE indexrelid=%s::regclass", (INDEX,))
        elif drift == "owner":
            cursor.execute("ALTER TABLE public.lab_arena_trajectory_events OWNER TO postgres")
        else:
            cursor.execute("DROP INDEX public." + INDEX)
            cursor.execute("CREATE INDEX " + INDEX + " ON public.lab_arena_trajectory_events"
                + ("(run_id)" if drift == "definition" else "(run_id,trajectory_id) WHERE event_kind='runtime.finished'"))
        with pytest.raises(AssertionError):
            assert_index(cursor)
        connection.rollback()


def test_anonymous_trajectory_access_remains_denied(proof):
    with proof["pg"].connect(**proof["dsn"]) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE anon")
        with pytest.raises(proof["pg"].Error) as failure:
            cursor.execute("SELECT * FROM public.lab_arena_trajectory_events LIMIT 1")
        assert failure.value.pgcode == "42501"
        connection.rollback()
