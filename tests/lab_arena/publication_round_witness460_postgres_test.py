"""Publication permits only proof-backed unfinished Arena assignments."""

from pathlib import Path
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import copy
import json

import pytest

from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge,
    _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
    current_service_database,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/460-lab-arena-publication-round-witness.sql"
HELPER = "public.lab_arena__publication_execution_incomplete_v1(text,text)"
OUTER = "public.lab_arena_integrity_publication_guard_v1()"
BASELINE = "public.lab_arena_publication_baseline_guard_v1()"


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
        "410-lab-arena-publication-transition-timeout.sql",
        "423-lab-arena-partial-publication.sql",
        "424-lab-arena-partial-baseline-stage-transition.sql",
        "430-lab-arena-unscored-judge-infrastructure.sql",
    ))


MODERN_ZERO_COST_FIELDS = frozenset({
    'competition_settled_microusd', 'competition_successful_microusd',
    'excluded_infrastructure_settled_microusd',
    'excluded_infrastructure_successful_microusd',
})


def _historical_cost_bucket(actual, expected):
    if actual == expected:
        return copy.deepcopy(actual)
    assert set(actual) - set(expected) == MODERN_ZERO_COST_FIELDS
    assert set(expected) <= set(actual)
    assert all(type(actual[key]) is int and actual[key] == 0
               for key in MODERN_ZERO_COST_FIELDS)
    projected = {key: value for key, value in actual.items()
                 if key not in MODERN_ZERO_COST_FIELDS}
    assert projected == expected
    return projected


@pytest.mark.parametrize('field', sorted(MODERN_ZERO_COST_FIELDS))
def test_historical_cost_projection_rejects_nonzero_modern_cost(field):
    expected = {'settled_microusd': 0}
    actual = {**expected, **dict.fromkeys(MODERN_ZERO_COST_FIELDS, 0)}
    assert _historical_cost_bucket(actual, expected) == expected
    actual[field] = 1
    with pytest.raises(AssertionError):
        _historical_cost_bucket(actual, expected)


def _historical_publication_costs(connect, round_id, publication):
    projected = copy.deepcopy(publication)
    with connect() as connection, connection.cursor() as cursor:
        for ranking in projected['final_ranking']:
            cost = ranking.get('cost_summary')
            if cost is None:
                continue
            submission_id = ranking['submission_id']
            for row in cost['per_icp']:
                cursor.execute('SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',
                               (round_id, submission_id, row['icp_position'], row['qualified_company_count']))
                expected = cursor.fetchone()[0]
                row['execution'] = _historical_cost_bucket(row['execution'], expected['execution'])
                assert all(key in row and row[key] == value for key, value in expected.items())
            for kind in ('execute', 'score'):
                cursor.execute('SELECT public.lab_arena__cost_kind_summary_v1(%s,%s)',
                               (submission_id, kind))
                key = 'execution' if kind == 'execute' else 'judge'
                cost[key] = _historical_cost_bucket(cost[key], cursor.fetchone()[0])
    return projected


@pytest.fixture()
def service_database():
    yield from current_service_database()


def _copy_historical_round(source_connect, target_connect, round_id, artifact_path):
    # A real publication payload binds participants, plans, costs and cache
    # receipts. Fragment-only SQL seeds cannot prove its complete transition.
    # Copy bounded data only; every function and trigger remains historical.
    tables = (
        ('lab_arena_rounds', 1), ('lab_arena_submissions', 2),
        ('lab_arena_runs', 80), ('lab_arena_ledger', 2000),
        ('lab_arena_judgment_cache', 40),
    )
    from psycopg2 import sql
    source_artifact = {}
    with source_connect() as source, target_connect() as target:
        with source.cursor() as src, target.cursor() as dst:
            dst.execute("SELECT tgrelid::regclass::text,tgname,tgenabled FROM pg_trigger "
                        "WHERE tgrelid=ANY(%s::regclass[]) ORDER BY 1,2",
                        ([table for table, _ in tables],))
            triggers_before = dst.fetchall()
            dst.execute('SET LOCAL session_replication_role=replica')
            for table, limit in tables:
                dst.execute("SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema='public' AND table_name=%s AND is_generated='NEVER' ORDER BY ordinal_position", (table,))
                historical_columns = [row[0] for row in dst.fetchall()]
                src.execute("SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema='public' AND table_name=%s", (table,))
                source_columns = {row[0] for row in src.fetchall()}
                columns = [name for name in historical_columns if name in source_columns]
                assert columns == historical_columns, (table, 'historical column absent')
                predicate = (sql.SQL('cache_key IN (SELECT judgment_cache_key FROM public.lab_arena_runs WHERE round_id=%s)')
                             if table == 'lab_arena_judgment_cache' else sql.SQL('round_id=%s'))
                src.execute(sql.SQL('SELECT row_to_json(t) FROM public.{} t WHERE {} LIMIT %s').format(
                    sql.Identifier(table), predicate), (round_id, limit + 1))
                rows = [row[0] for row in src.fetchall()]
                assert 0 < len(rows) <= limit, (table, len(rows), limit)
                source_artifact[table] = rows
                names = sql.SQL(',').join(map(sql.Identifier, columns))
                for row in rows:
                    payload = {name: row[name] for name in columns}
                    dst.execute(sql.SQL('INSERT INTO public.{} ({}) SELECT {} FROM '
                                        'jsonb_populate_record(NULL::public.{},%s::jsonb)').format(
                        sql.Identifier(table), names, names, sql.Identifier(table)), (json.dumps(payload),))
            dst.execute('SET LOCAL session_replication_role=origin')
            dst.execute("SELECT current_setting('session_replication_role')")
            assert dst.fetchone()[0] == 'origin'
            dst.execute("SELECT tgrelid::regclass::text,tgname,tgenabled FROM pg_trigger "
                        "WHERE tgrelid=ANY(%s::regclass[]) ORDER BY 1,2",
                        ([table for table, _ in tables],))
            assert dst.fetchall() == triggers_before
    artifact_path.write_text(json.dumps(source_artifact, sort_keys=True))


@pytest.mark.parametrize("failed_execution", [False, True])
def test_archived_accepted_judge_blocks_exact_current_incomplete_transition(
    database, service_database, tmp_path, monkeypatch, failed_execution,
):
    psycopg2, dsn = database
    historical_connect = lambda: psycopg2.connect(**dsn)
    service_psycopg, service_dsn = service_database
    connect = lambda: service_psycopg.connect(**service_dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["PartialMiner"], runners=["alpha"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    assert fixtures._start_round(harness, day=27 + int(failed_execution), epoch=62_427) == 2
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("scored", runners=1)
    round_row = harness.service.store.get_round(harness.round_id)
    ids = {
        bool(item["is_king"]): item["submission_id"]
        for item in round_row["participants"]
    }
    assert set(ids) == {False, True}

    captured = {}
    transition = harness.service.store.transition_round
    def capture(round_id, expected, target, patch=None):
        assert (round_id, expected, target) == (harness.round_id, "scored", "published")
        captured.update(patch or {})
        return {"status": "captured"}
    harness.service.store.transition_round = capture
    try:
        assert harness.service.publish(harness.round_id)["status"] == "captured"
    finally:
        harness.service.store.transition_round = transition
    full = captured["publication_doc"]
    assert full["king_decision"]["outcome"] == "no_king"
    _copy_historical_round(connect, historical_connect, harness.round_id, tmp_path / "source-round.json")
    # Separate current-schema smoke: its real transition must also succeed.
    assert harness.service.publish(harness.round_id)['status'] == 'ok'
    connect = historical_connect
    source_publication = copy.deepcopy(full)
    (tmp_path / "source-publication.json").write_text(json.dumps(source_publication, sort_keys=True))
    full = _historical_publication_costs(connect, harness.round_id, source_publication)
    assert captured["publication_doc"] == source_publication

    publication=copy.deepcopy(full)
    submission_id=ids[False]
    ranking=next(item for item in publication['final_ranking'] if item['submission_id']==submission_id)
    ranking.update(final_score=None,cost_summary=None,eligible=False,eligibility_reason='execution_incomplete')
    with connect() as connection,connection.cursor() as cursor:

        cursor.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER')
        cursor.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER')
        cursor.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND status='accepted' ORDER BY icp_position LIMIT 1",(harness.round_id,submission_id))
        execution_id=cursor.fetchone()[0]
        archive_id=harness.round_id+'-r452archive'
        cursor.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,participants) SELECT %s,'cancelled',configuration_doc,participants FROM public.lab_arena_rounds WHERE round_id=%s",(archive_id,harness.round_id))
        cursor.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(j)||jsonb_build_object('run_id',j.run_id||':archived448','round_id',%s,'assignment_id',j.assignment_id||':archived448'))).* FROM public.lab_arena_runs j WHERE scored_run_id=%s AND kind='score' AND status='accepted'",(archive_id,execution_id))
        cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",(execution_id,))
        cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{}'::jsonb WHERE scored_run_id=%s AND kind='score' AND round_id=%s",(execution_id,harness.round_id))
        if failed_execution:
            cursor.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{\"infrastructure_incomplete\":true}'::jsonb WHERE run_id=%s",(execution_id,))
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((now()-interval '1 minute')::text)),'{schedule,stage_2_close}',to_jsonb((now()-interval '1 minute')::text)) WHERE round_id=%s",(harness.round_id,))
        cursor.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER')
        cursor.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER')
        cursor.execute('SET LOCAL ROLE lab_arena_owner')
        cursor.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        assert cursor.fetchone()[0] is False
        cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)',(harness.round_id,submission_id))
        assert cursor.fetchone()[0] is False
        cursor.execute('SAVEPOINT actual_transition')
        with pytest.raises(psycopg2.Error) as error:
            cursor.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",(harness.round_id,json.dumps({'publication_doc':publication,'published_at':publication['published_at']})))
        print('EXACT_TRANSITION_FAILURE',error.value.pgcode,error.value.diag.message_primary)
        cursor.execute('ROLLBACK TO SAVEPOINT actual_transition')
        cursor.execute('RESET ROLE')
        cursor.execute(MIGRATION.read_text().replace("BEGIN;", "").replace("COMMIT;", ""))
        cursor.execute('SET LOCAL ROLE lab_arena_owner')
        cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)',(harness.round_id,submission_id))
        assert cursor.fetchone()[0] is True
        cursor.execute('RESET ROLE')
        if not failed_execution:
            cursor.execute('SAVEPOINT missing_current_judge')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND scored_run_id=%s", (harness.round_id,execution_id))
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)', (harness.round_id,submission_id))
            assert cursor.fetchone()[0] is False
            cursor.execute('ROLLBACK TO SAVEPOINT missing_current_judge')
            cursor.execute('SAVEPOINT malformed_deadline')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','null'::jsonb) WHERE round_id=%s",(harness.round_id,))
            cursor.execute('SET LOCAL session_replication_role=origin')
            cursor.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s)', (harness.round_id,submission_id))
            assert cursor.fetchone()[0] is False
            cursor.execute('ROLLBACK TO SAVEPOINT malformed_deadline')
        cursor.execute('SET LOCAL ROLE lab_arena_owner')
        cursor.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",(harness.round_id,json.dumps({'publication_doc':publication,'published_at':publication['published_at']})))
        assert cursor.fetchone()[0]['status']=='ok'
        cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status='accepted'",(archive_id,))
        assert cursor.fetchone()[0]>=1
        connection.rollback()


def test_legacy_current_authority_and_deadline_controls(database, service_database, tmp_path, monkeypatch):
    from tests.lab_arena import unscored_judge_infrastructure430_postgres_test as legacy
    monkeypatch.setattr(legacy, "MIGRATION", MIGRATION)
    legacy.test_exhausted_judge_proof_preserves_null_ranking_and_blocks_false_proof(
        database, tmp_path, monkeypatch, service_database
    )


def test_migration_replay_and_security(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,))
        identity=cur.fetchone()
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute("SELECT proowner,proacl,prosecdef,provolatile,proconfig,encode(extensions.digest(pg_get_functiondef(oid),'sha256'),'hex') FROM pg_proc WHERE oid=%s::regprocedure", (HELPER,))
        after=cur.fetchone()
        assert after[:-1]==identity
        assert after[-1]=='5fb513506ad7ecd78de3009bce3d004cbc8e9883af163fafb1934b7cebe904d7'


def test_migration_refuses_unknown_definition(database):
    psycopg, dsn = database
    conn=psycopg.connect(**dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (HELPER,))
            definition=cur.fetchone()[0]
            altered=definition.replace("RETURN v_missing > 0;", "RETURN v_missing > 0 OR FALSE;")
            assert altered!=definition
            cur.execute(altered)
            cur.execute('SAVEPOINT unknown_preimage')
            with pytest.raises(psycopg.Error,match='preimage differs'):
                cur.execute(MIGRATION.read_text().replace('BEGIN;','').replace('COMMIT;',''))
            cur.execute('ROLLBACK TO SAVEPOINT unknown_preimage')
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)",(HELPER,))
            assert cur.fetchone()[0]==altered
    finally:
        conn.rollback()
        conn.close()
