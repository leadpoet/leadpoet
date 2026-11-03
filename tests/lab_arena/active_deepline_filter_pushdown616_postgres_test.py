"""Reuse559 without changing native cost authority, NULLs or run/cursor scope."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import execute_reconciliation_scope505_postgres_test as prior
from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as heads
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/616-lab-arena-active-deepline-filter-pushdown.sql'
SQL = MIGRATION.read_text()
NAMES = ('lab_arena_list_deepline_cost_reconciliations_v1',
         'lab_arena_list_deepline_cost_reconciliations_v2')
PATCHES = re.findall(r"\('([^']+)',\s*'([0-9a-f]{64})',\s*'([0-9a-f]{64})'", SQL)
assert len(PATCHES) == 2
ORIGINAL = {}
INDEX = 'lab_arena_ledger_deepline_uncertain_round_entry_idx'


def signature(name):
    return '(text,text,bigint,integer' + (',boolean' if name.endswith('_v2') else '') + ')'


def definition(cur, name):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.' + name + signature(name),))
    return cur.fetchone()[0]


def snapshot(cur):
    cur.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY indexname")
    indexes = cur.fetchall()
    return catalog(cur), rows(cur), indexes


def guard_body(sql=SQL):
    return sql.split('DO $active_deepline_filter_pushdown616$', 1)[1].split('$active_deepline_filter_pushdown616$;', 1)[0]


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            for name, (_, pre, _) in zip(NAMES, PATCHES):
                source = definition(cur, name)
                assert hashlib.sha256(source.encode()).hexdigest() == pre
                ORIGINAL[name] = source
                oracle = 'test_before616_' + name
                cur.execute(source.replace('FUNCTION public.' + name + '(', 'FUNCTION public.' + oracle + '(', 1))
                cur.execute('ALTER FUNCTION public.' + oracle + signature(name) + ' OWNER TO lab_arena_owner')
                cur.execute('REVOKE ALL ON FUNCTION public.' + oracle + signature(name) + ' FROM PUBLIC')
                cur.execute('GRANT EXECUTE ON FUNCTION public.' + oracle + signature(name) + ' TO lab_arena_service')
            cur.execute((ROOT / 'scripts/559-lab-arena-deepline-uncertain-round-entry-index.sql').read_text())
            before = snapshot(cur)
            cur.execute(SQL)
            after = snapshot(cur)
            assert before[1:] == after[1:]
            changed = []
            for old, new in zip(before[0], after[0]):
                assert old[0] == new[0] and old[2:] == new[2:]
                if old[1] != new[1]:
                    changed.append(old[0])
            assert changed == [row[0].removeprefix('public.') for row in PATCHES]
            for name, (_, _, post) in zip(NAMES, PATCHES):
                assert hashlib.sha256(definition(cur, name).encode()).hexdigest() == post
            cur.execute(SQL)
            assert snapshot(cur) == after
        conn.close()
        yield db
    finally:
        gen.close()


def pair(cur, rid, run_id, cursor, limit, flag, name):
    args = (rid, run_id, cursor, limit) + ((flag,) if name.endswith('_v2') else ())
    slots = ','.join(['%s'] * len(args))
    cur.execute('SET LOCAL ROLE lab_arena_service')
    cur.execute('SELECT public.test_before616_' + name + '(' + slots + '),public.' + name + '(' + slots + ')', args + args)
    before, after = cur.fetchone()
    cur.execute('RESET ROLE')
    assert before == after
    return after


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'volatility', 'path', 'security',
                                  'index_missing', 'index_invalid', 'index_unready',
                                  'index_owner', 'index_definition', 'postimage', 'second_function'])
def test_exact_source_security_index_and_replay_guards_are_atomic(database, installed, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if not installed:
            for source in ORIGINAL.values():
                cur.execute(source)
        first, second = NAMES
        target = second if drift == 'second_function' else first
        qualified = 'public.' + target + signature(target)
        if drift in {'source', 'second_function'}:
            cur.execute(definition(cur, target).replace('AS $function$\n', 'AS $function$\n-- drift\n', 1))
        elif drift in {'owner', 'acl', 'volatility', 'path', 'security'}:
            cur.execute({
                'owner': 'ALTER FUNCTION ' + qualified + ' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION ' + qualified + ' TO anon',
                'volatility': 'ALTER FUNCTION ' + qualified + ' VOLATILE',
                'path': 'ALTER FUNCTION ' + qualified + ' SET search_path=public,pg_catalog',
                'security': 'ALTER FUNCTION ' + qualified + ' SECURITY INVOKER',
            }[drift])
        elif drift == 'index_missing':
            cur.execute('DROP INDEX public.' + INDEX)
        elif drift in {'index_invalid', 'index_unready'}:
            column = 'indisvalid' if drift == 'index_invalid' else 'indisready'
            cur.execute('UPDATE pg_index SET ' + column + '=false WHERE indexrelid=%s::regclass', ('public.' + INDEX,))
        elif drift == 'index_owner':
            cur.execute('ALTER TABLE public.lab_arena_ledger OWNER TO postgres')
        elif drift == 'index_definition':
            cur.execute('DROP INDEX public.' + INDEX)
            cur.execute('CREATE INDEX ' + INDEX + ' ON public.lab_arena_ledger(run_id,entry_id)')
        text = SQL.replace(PATCHES[0][2], '0' * 64) if drift == 'postimage' else SQL
        before = snapshot(cur)
        cur.execute('SAVEPOINT rejected')
        with pytest.raises(pg.Error, match='Active Deepline'):
            cur.execute('DO $test$' + guard_body(text) + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT rejected')
        assert snapshot(cur) == before  # V1 must roll back if V2 fails.
        conn.rollback()


def test_exact_180_results_and_added_predicates_preserve_null_scope_and_bindings(database):
    rid, candidate = prior.seed(database, 'filter616')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('SET LOCAL jit=off')
        heads._duplicate_calls(cur, candidate, 48, start=616000)
        cur.execute("SELECT entry_id FROM lab_arena_ledger WHERE round_id=%s AND entry_kind='uncertain' AND provider='deepline' ORDER BY entry_id", (rid,))
        ids = [row[0] for row in cur.fetchall()]
        assert len(ids) == 49
        checks = 0
        for case in ('populated', 'superseded', 'settled', 'failed_billable', 'no_candidate'):
            cur.execute('SAVEPOINT case_state')
            if case == 'superseded':
                cur.execute("UPDATE lab_arena_ledger SET entry_id=-entry_id WHERE entry_kind='uncertain' AND round_id=%s", (rid,))
            elif case == 'settled':
                cur.execute("INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,0,'{}'::jsonb FROM lab_arena_ledger WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s", (rid,))
            elif case == 'failed_billable':
                cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,call_succeeded}','false'::jsonb) WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s", (rid,))
            elif case == 'no_candidate':
                cur.execute("UPDATE lab_arena_ledger SET provider='openrouter' WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s", (rid,))
            for name in NAMES:
                for flag in ((None,) if name.endswith('_v1') else (True, False, None)):
                    for cursor in (0, ids[24], ids[-1]):
                        for run_id in ('', candidate['run_id'], 'missing-run'):
                            pair(cur, rid, run_id, cursor, 1, flag, name)
                            checks += 1
            cur.execute('ROLLBACK TO SAVEPOINT case_state')
        assert checks == 180
        extra = 0
        for name in NAMES:
            for flag in ((None,) if name.endswith('_v1') else (True, False, None)):
                for rr, run_id, cursor, limit in ((None, None, None, None), (rid, None, None, 1),
                    (rid, '', -1, 0), (rid, '', 2**63-1, 21), ('absent', None, 0, 20)):
                    pair(cur, rr, run_id, cursor, limit, flag, name)
                    extra += 1
        for damage in ('null_call', 'round_mismatch', 'reservation_payer', 'missing_reservation'):
            cur.execute('SAVEPOINT binding')
            if damage == 'null_call':
                cur.execute('UPDATE lab_arena_ledger SET call_identity=NULL WHERE entry_id=%s', (ids[0],))
            elif damage == 'round_mismatch':
                cur.execute("UPDATE lab_arena_ledger SET round_id='absent-round' WHERE entry_id=ANY(%s)", (ids,))
            elif damage == 'reservation_payer':
                cur.execute("UPDATE lab_arena_ledger SET funding_source='host' WHERE entry_kind='reservation' AND round_id=%s", (rid,))
            else:
                cur.execute("DELETE FROM lab_arena_ledger WHERE entry_kind='reservation' AND round_id=%s", (rid,))
            for name in NAMES:
                for flag in ((None,) if name.endswith('_v1') else (True, False, None)):
                    pair(cur, rid, '', 0, 20, flag, name)
                    extra += 1
            cur.execute('ROLLBACK TO SAVEPOINT binding')
        assert extra == 36


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


def body_plan(cur, source, name, rid, mode):
    body = source.split('AS $function$', 1)[1].rsplit('$function$', 1)[0].strip().rstrip(';')
    for i, key in enumerate(('p_round_id', 'p_run_id', 'p_after_entry_id', 'p_limit', 'p_successful_execute_only'), 1):
        body = re.sub(r'\b' + key + r'\b', '$' + str(i), body)
    cur.execute('SET plan_cache_mode=' + mode)
    cur.execute('SET jit=off')
    cur.execute("SET statement_timeout='20s'")
    cur.execute('SET ROLE lab_arena_owner')
    cur.execute('PREPARE filter616' + signature(name) + ' AS ' + body)
    args = (rid, '', 0, 1) + ((True,) if name.endswith('_v2') else ())
    cur.execute('EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE filter616(' + ','.join(['%s']*len(args)) + ')', args)
    result = cur.fetchone()[0][0]
    cur.execute('DEALLOCATE filter616')
    cur.execute('RESET ROLE')
    return result


def test_existing_round_index_bounds_ordinary_history_without_new_index(database, tmp_path):
    rid, candidate = prior.seed(database, 'plan616')
    pg, dsn = database
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        cur = conn.cursor()
        cur.execute('SET session_replication_role=replica')
        cur.execute("INSERT INTO lab_arena_runs SELECT(jsonb_populate_record(NULL::lab_arena_runs,to_jsonb(base)||jsonb_build_object('run_id','test616-run-'||g.n,'assignment_id','test616-assignment-'||g.n,'claim_request_id',NULL))).* FROM lab_arena_runs base CROSS JOIN generate_series(1,100)g(n)WHERE base.run_id=%s", (candidate['run_id'],))
        cur.execute("INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)SELECT'uncertain',base.miner_hotkey,base.round_id,base.submission_id,'test616-run-'||g.n,base.stage,'sha256:'||lpad(to_hex(g.n*1000+e.n+6000000),64,'0'),'openrouter',base.operation_id,base.funding_source,0,base.entry_doc FROM(SELECT*FROM lab_arena_ledger WHERE entry_id=%s)base CROSS JOIN generate_series(1,100)g(n)CROSS JOIN generate_series(1,300)e(n)", (candidate['uncertain_entry_id'],))
        cur.execute('RESET session_replication_role')
        for table in ('lab_arena_runs', 'lab_arena_ledger'):
            cur.execute('VACUUM(ANALYZE) ' + table)
        before = snapshot(cur)
        reports = {}
        for case in ('populated', 'empty'):
            if case == 'empty':
                cur.execute('SET session_replication_role=replica')
                cur.execute("UPDATE lab_arena_ledger SET provider='openrouter' WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s", (rid,))
                cur.execute('RESET session_replication_role')
                cur.execute('ANALYZE lab_arena_ledger')
            for mode in ('force_generic_plan', 'force_custom_plan'):
                for name in NAMES:
                    old = body_plan(cur, ORIGINAL[name], name, rid, mode)
                    new = body_plan(cur, definition(cur, name), name, rid, mode)
                    indexes = [node.get('Index Name') for node in nodes(new['Plan'])]
                    assert INDEX in indexes
                    assert new['Plan']['Shared Hit Blocks'] * 5 < old['Plan']['Shared Hit Blocks']
                    assert not any('uncertain_run_entry' in str(value) for value in indexes)
                    reports['|'.join((case, mode, name))] = {'before': old, 'after': new}
        assert catalog(cur) == before[0]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname='lab_arena_ledger_deepline_uncertain_run_entry_idx'")
        assert cur.fetchone()[0] == 0
        (tmp_path / 'filter616-plans.json').write_text(json.dumps(reports))
    finally:
        conn.close()
