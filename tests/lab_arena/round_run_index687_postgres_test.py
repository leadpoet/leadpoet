"""Round-bounded reads preserve exact rows while avoiding unrelated run pages."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time
import uuid

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / 'scripts/687-lab-arena-round-run-index.sql'
DDL = MIGRATION.read_text()
STATEMENTS = re.findall(r'DO \$guard687\$.*?\$guard687\$;|CREATE INDEX CONCURRENTLY.*?;|DO \$verify687\$.*?\$verify687\$;', DDL, re.S)
INDEX = 'lab_arena_runs_round_run_idx'
TARGET = 'arena-2026-11-10-index687102'
BULK = 'arena-2026-11-10-index687101'
COLUMNS = 'run_id,submission_id,icp_position,attempt,per_icp_score'


def apply(cursor):
    assert cursor.connection.autocommit
    assert len(STATEMENTS) == 3
    for statement in STATEMENTS:
        cursor.execute(statement)


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


def security(cursor):
    cursor.execute("SELECT relowner,relacl,relrowsecurity,relforcerowsecurity FROM pg_class "
                   "WHERE oid='public.lab_arena_runs'::regclass")
    table = cursor.fetchone()
    cursor.execute("SELECT policyname,roles,cmd,qual,with_check FROM pg_policies "
                   "WHERE schemaname='public' AND tablename='lab_arena_runs' ORDER BY policyname")
    policies = cursor.fetchall()
    cursor.execute("SELECT oid::regprocedure::text,md5(pg_get_functiondef(oid)),proowner,proacl::text "
                   "FROM pg_proc WHERE pronamespace='public'::regnamespace ORDER BY oid")
    return table, policies, cursor.fetchall()


def fingerprint(cursor):
    cursor.execute("SELECT count(*),md5(string_agg(md5(to_jsonb(r)::text),'' ORDER BY run_id)) "
                   "FROM public.lab_arena_runs r")
    return cursor.fetchone()


def measure(cursor, *, mode='force_custom_plan', rid=TARGET, stage=1, kind='execute', status=None, after='', limit=500):
    cursor.execute('SET plan_cache_mode=' + mode)
    cursor.execute('SET ROLE lab_arena_service')
    predicates = ['round_id=$1', 'run_id>$2']
    args = [rid, after]
    types = ['text', 'text']
    for key, value in [('stage', stage), ('kind', kind), ('status', status)]:
        if value is not None:
            args.append(value)
            types.append('smallint' if key == 'stage' else 'text')
            predicates.append(key + '=$' + str(len(args)))
    query = 'SELECT ' + COLUMNS + ' FROM public.lab_arena_runs WHERE ' + ' AND '.join(predicates) + ' ORDER BY run_id' + ('' if limit is None else ' LIMIT '+str(limit))
    cursor.execute('PREPARE round687(' + ','.join(types) + ') AS ' + query)
    execute = 'EXECUTE round687(' + ','.join(['%s'] * len(args)) + ')'
    elapsed = []
    for _ in range(7):
        start = time.perf_counter()
        cursor.execute(execute, args)
        rows = cursor.fetchall()
        elapsed.append((time.perf_counter() - start) * 1000)
    cursor.execute('EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) ' + execute, args)
    plan = cursor.fetchone()[0][0]
    cursor.execute('DEALLOCATE round687')
    cursor.execute('RESET ROLE')
    return {'rows': rows, 'plan': plan, 'median_ms': statistics.median(elapsed)}



def measure_write(cursor):
    # Read-plan experiments must not force replanning each row's unchanged
    # trigger/FK statements during the write comparison.
    cursor.execute('SET plan_cache_mode=auto')
    plans = []
    for _ in range(5):
        cursor.execute('BEGIN')
        try:
            cursor.execute("EXPLAIN(ANALYZE,BUFFERS,WAL,FORMAT JSON,TIMING FALSE) "
                "INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                "miner_hotkey,stage,icp_position,attempt) SELECT %s||s,"
                "%s||s,%s,'sub687-102',%s,1,0,1 FROM generate_series(1,500) s",
                ('bench687-'+uuid.uuid4().hex+'-', 'bench687-'+uuid.uuid4().hex+'-',
                 TARGET, hotkey('index687-miner')))
            plans.append(cursor.fetchone()[0][0])
        finally:
            cursor.execute('ROLLBACK')
    return {'median_ms':statistics.median(plan['Execution Time'] for plan in plans),
            'wal_bytes':statistics.median(plan['Plan'].get('WAL Bytes',0) for plan in plans),
            'plans':plans}


def snapshot(store):
    result = {}
    for rid in (TARGET, BULK, 'arena-2026-11-10-index6872', 'absent687'):
        for stage, kind, status in [(None,None,None),(1,'execute',None),(2,'score',None),
                                    (None,None,'accepted'),(None,None,'failed'),
                                    (None,None,'pending'),(None,None,'leased')]:
            rows = store.list_runs(rid, stage=stage, kind=kind, status=status)
            result[(rid,stage,kind,status)] = hashlib.sha256(
                json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return result


@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
    generator = current_service_database()
    pg, dsn = next(generator)
    connection = pg.connect(**dsn)
    connection.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout='30s'")
            # Include the existing specialized closed-score index. The new
            # general index must still help execute and all-kind round reads.
            cursor.execute((ROOT / 'scripts/615-lab-arena-closed-score-run-covering-index.sql').read_text())
        for index in range(1, 103):
            rid = 'arena-2026-11-10-index687' + str(index)
            store.create_round(rid, round_config(rid, [hotkey('index687-runner')]))
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) "
                           "SELECT 'sub687-'||s, 'arena-2026-11-10-index687'||s,%s,'accepted' "
                           "FROM generate_series(1,102) s", (hotkey('index687-miner'),))
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("""WITH payload AS (SELECT string_agg(md5(s::text),'') AS text
              FROM generate_series(1,35) s), seed AS (
              SELECT s,CASE WHEN s>50000 THEN 102 WHEN s%%32=0 THEN 101 ELSE 1+(s%%100) END AS r
              FROM generate_series(1,50010) s)
              INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,
                miner_hotkey,stage,icp_position,attempt,kind,status,per_icp_score,
                claim_response,result_doc,terminal_doc,reviewer_runtime_doc,scoring_generation_floor)
              SELECT 'run687-'||lpad(s::text,6,'0'),'assignment687-'||s,
                'arena-2026-11-10-index687'||r,'sub687-'||r,%s,
                CASE WHEN r=102 OR s%%2=0 THEN 1 ELSE 2 END,(s%%10)::smallint,1,
                CASE WHEN r=102 OR s%%3<>0 THEN 'execute' ELSE 'score' END,
                CASE WHEN r=102 THEN 'accepted'
                     WHEN (CASE WHEN r=101 THEN s/32 ELSE s END)%%4=0 THEN 'accepted'
                     WHEN (CASE WHEN r=101 THEN s/32 ELSE s END)%%4=1 THEN 'failed'
                     WHEN (CASE WHEN r=101 THEN s/32 ELSE s END)%%4=2 THEN 'pending' ELSE 'leased' END,
                CASE WHEN r=102 OR s%%4=0 THEN 42.0 ELSE NULL END,
                jsonb_build_object('private_claim',left(text,180)),
                jsonb_build_object('private_output',text),
                jsonb_build_object('private_terminal',left(text,180)),
                jsonb_build_object('private_runtime',left(text,180)),0 FROM seed CROSS JOIN payload""",
                (hotkey('index687-miner'),))
            cursor.execute('RESET session_replication_role')
            cursor.execute('VACUUM(ANALYZE) public.lab_arena_runs')
            cursor.execute('SELECT DISTINCT status FROM lab_arena_runs WHERE round_id=%s',(BULK,))
            assert {row[0] for row in cursor.fetchall()} == {'accepted','failed','pending','leased'}
            before_writes = measure_write(cursor)
            before_security = security(cursor)
            before_rows = fingerprint(cursor)
            before = {mode: measure(cursor, mode=mode) for mode in ('force_custom_plan','force_generic_plan')}
            before['bulk'] = measure(cursor, rid=BULK, stage=None, kind=None)
            before['cursor'] = measure(cursor, rid=BULK, stage=None, kind=None, after='run687-000499')
            before['full_round'] = measure(cursor, rid=BULK, stage=None, kind=None, limit=None)
            full_before = snapshot(store)
            cursor.execute('SET plan_cache_mode=force_generic_plan;SET ROLE lab_arena_service')
            cursor.execute('PREPARE retained687(text) AS SELECT '+COLUMNS+
                           " FROM public.lab_arena_runs WHERE round_id=$1 AND stage=1 AND kind='execute' ORDER BY run_id LIMIT 500")
            cursor.execute('EXECUTE retained687(%s)',(TARGET,))
            retained_rows = cursor.fetchall()
            cursor.execute('RESET ROLE')
            # A transaction that already wrote must delay the concurrent
            # build, while independent writes and service reads still finish.
            held = pg.connect(**dsn)
            with held.cursor() as writer:
                writer.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                               "miner_hotkey,stage,icp_position,attempt) VALUES "
                               "('writer687-held','writer687-held',%s,'sub687-102',%s,1,0,1)",
                               (TARGET,hotkey('index687-miner')))
            built = {}
            def build():
                with closing(pg.connect(**dsn)) as session:
                    session.autocommit = True
                    try:
                        with session.cursor() as builder:
                            builder.execute("SET statement_timeout='20s';SET lock_timeout='10s'")
                            apply(builder)
                        built['ok'] = True
                    except Exception as exc:
                        built['error'] = str(exc)
            worker = threading.Thread(target=build)
            worker.start()
            try:
                deadline = time.monotonic() + 8
                waiting = False
                while time.monotonic() < deadline:
                    cursor.execute("SELECT phase FROM pg_stat_progress_create_index WHERE relid='public.lab_arena_runs'::regclass")
                    waiting = any(row[0].startswith('waiting for writers') for row in cursor.fetchall())
                    if waiting or not worker.is_alive():
                        break
                    time.sleep(0.01)
                assert waiting, built
                with closing(pg.connect(**dsn)) as other, other.cursor() as writer:
                    writer.execute("SET lock_timeout='1s';SET statement_timeout='2s'")
                    writer.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                                   "miner_hotkey,stage,icp_position,attempt) VALUES "
                                   "('writer687-other','writer687-other',%s,'sub687-102',%s,1,1,1)",
                                   (TARGET,hotkey('index687-miner')))
                    writer.execute('SET ROLE lab_arena_service')
                    writer.execute('SELECT count(*) FROM lab_arena_runs WHERE round_id=%s',(TARGET,))
                    assert writer.fetchone()[0] == 11
                    other.rollback()  # Probe write is not part of the equality dataset.
                assert worker.is_alive()
                held.rollback()
                worker.join(timeout=15)
                assert not worker.is_alive() and built == {'ok':True}, built
            finally:
                held.rollback()
                held.close()
                worker.join(timeout=25)
            apply(cursor)
            assert security(cursor) == before_security
            assert fingerprint(cursor) == before_rows
            assert snapshot(store) == full_before
            after = {mode: measure(cursor, mode=mode) for mode in ('force_custom_plan','force_generic_plan')}
            after['bulk'] = measure(cursor, rid=BULK, stage=None, kind=None)
            after['cursor'] = measure(cursor, rid=BULK, stage=None, kind=None, after='run687-000499')
            after['full_round'] = measure(cursor, rid=BULK, stage=None, kind=None, limit=None)
            cursor.execute('SELECT pg_relation_size(%s::regclass)',('public.'+INDEX,))
            index_bytes = cursor.fetchone()[0]
            cursor.execute('SET ROLE lab_arena_service')
            cursor.execute('EXECUTE retained687(%s)',(TARGET,))
            assert cursor.fetchall() == retained_rows
            cursor.execute('EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE retained687(%s)',(TARGET,))
            retained_plan = cursor.fetchone()[0][0]
            assert any(node.get('Index Name') == INDEX for node in nodes(retained_plan['Plan']))
            cursor.execute('DEALLOCATE retained687;RESET ROLE')
            after_writes = measure_write(cursor)
            # Compare a wider scoped alternative on the same disposable data.
            # It orders one exact stage/kind, but cannot directly serve the
            # existing all-kind ORDER BY run_id without a separate sort.
            cursor.execute('DROP INDEX public.'+INDEX)
            cursor.execute('CREATE INDEX CONCURRENTLY lab_arena_round_kind_stage687_test_idx '
                           'ON public.lab_arena_runs(round_id,kind,stage,run_id)')
            alternative = {mode:measure(cursor,mode=mode) for mode in ('force_custom_plan','force_generic_plan')}
            alternative['bulk'] = measure(cursor,rid=BULK,stage=None,kind=None)
            alternative['cursor'] = measure(cursor,rid=BULK,stage=None,kind=None,after='run687-000499')
            alternative['full_round'] = measure(cursor,rid=BULK,stage=None,kind=None,limit=None)
            cursor.execute("SELECT pg_relation_size('public.lab_arena_round_kind_stage687_test_idx')")
            alternative_bytes = cursor.fetchone()[0]
            cursor.execute('DROP INDEX public.lab_arena_round_kind_stage687_test_idx')
            apply(cursor)
            report = {'before':before,'after':after,'alternative':alternative,
                      'index_bytes':index_bytes,'alternative_bytes':alternative_bytes,'retained_plan':retained_plan,
                      'write_before':before_writes,'write_after':after_writes,'rows':before_rows[0]}
            artifact = tmp_path_factory.mktemp('round687') / 'plans.json'
            artifact.write_text(json.dumps(report, default=str, indent=2))
            print('INDEX687_REPORT', artifact)
        yield pg, dsn, report
    finally:
        store.close()
        connection.close()
        generator.close()


def test_round_bounded_access_and_output_equality(indexed):
    _,_,report = indexed
    for key,before in report['before'].items():
        after = report['after'][key]
        assert before['rows'] == after['rows']
        assert any(node.get('Index Name') == INDEX for node in nodes(after['plan']['Plan']))
        assert after['plan']['Plan']['Shared Hit Blocks'] < before['plan']['Plan']['Shared Hit Blocks']/5
    assert report['rows'] == 50010


@pytest.mark.parametrize('drift',['invalid','unready','owner','definition','predicate'])
def test_replay_rejects_wrong_or_unusable_existing_index(indexed,drift):
    pg,dsn,_ = indexed
    with closing(pg.connect(**dsn)) as connection, connection.cursor() as cursor:
        if drift in ('invalid','unready'):
            cursor.execute('UPDATE pg_index SET '+('indisvalid' if drift=='invalid' else 'indisready')+
                           '=false WHERE indexrelid=%s::regclass',('public.'+INDEX,))
        elif drift=='owner':
            cursor.execute('ALTER TABLE public.lab_arena_runs OWNER TO postgres')
        else:
            cursor.execute('DROP INDEX public.'+INDEX)
            cursor.execute('CREATE INDEX '+INDEX+' ON public.lab_arena_runs '+
                           ('(run_id)' if drift=='definition' else "(round_id,run_id) WHERE kind='execute'"))
        with pytest.raises(pg.Error,match='687 .*differs'):
            cursor.execute(STATEMENTS[0])
        connection.rollback()


def test_migration_has_only_explicit_autocommit_preflight_build_and_readback():
    normalized = lambda value: re.sub(r'\s+', '', re.sub(r'--[^\n]*', '', value))
    assert normalized(DDL) == normalized(''.join(STATEMENTS))
    assert len(STATEMENTS) == 3
    assert STATEMENTS[1].startswith('CREATE INDEX CONCURRENTLY IF NOT EXISTS '+INDEX)
    assert 'INCLUDE' not in STATEMENTS[1] and 'WHERE' not in STATEMENTS[1]
