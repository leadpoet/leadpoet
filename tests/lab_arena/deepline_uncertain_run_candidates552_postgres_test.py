"""Exact active Deepline query parity and physical run-index regression."""
import re
from pathlib import Path

import pytest

from tests.lab_arena import execute_reconciliation_scope505_postgres_test as scope
from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as heads

ROOT = Path(__file__).parents[2]
INDEX = ROOT / 'scripts/551-lab-arena-deepline-uncertain-run-index.sql'
QUERY = ROOT / 'scripts/552-lab-arena-deepline-uncertain-run-candidates.sql'
NAMES = ('lab_arena_list_deepline_cost_reconciliations_v1',
         'lab_arena_list_deepline_cost_reconciliations_v2')
POST = ('bedaa8c553065504baec0f365c79963981e10130fc23a829e5ad42a6c45e793f',
        '0f0a7e3635094e22140b627d1c352f81867f20a51ba5d24b74576e34ab9bc3a0')
PARTIAL_INDEX = 'lab_arena_ledger_deepline_uncertain_run_entry_idx'


@pytest.fixture(scope='module')
def database():
    gen = scope.database.__wrapped__()
    db = next(gen)
    try:
        yield db
    finally:
        gen.close()


def _clone_before(cur):
    for name in NAMES:
        signature = '(text,text,bigint,integer' + (',boolean' if name.endswith('_v2') else '') + ')'
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.'+name+signature,))
        source = cur.fetchone()[0]
        cur.execute(source.replace('FUNCTION public.'+name+'(',
                                   'FUNCTION public.test_before552_'+name+'(', 1))


def _pair(cur, round_id, run_id, cursor, limit, flag, *, name):
    args = (round_id,run_id,cursor,limit) + ((flag,) if name.endswith('_v2') else ())
    slots = ','.join(['%s']*len(args))
    cur.execute('SELECT public.test_before552_'+name+'('+slots+'), '
                'public.'+name+'('+slots+')',args+args)
    before,after = cur.fetchone()
    assert before == after
    return after


def _body_plan(cur, name, round_id, *, before=False):
    signature = '(text,text,bigint,integer' + (',boolean' if name.endswith('_v2') else '') + ')'
    qualified='public.'+('test_before552_' if before else '')+name+signature
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (qualified,))
    source = cur.fetchone()[0]
    body = source.split('AS $function$\n',1)[1].rsplit('\n$function$',1)[0].strip().rstrip(';')
    for key,value in {'p_round_id':"'"+round_id+"'::text",'p_run_id':"''::text",
                      'p_after_entry_id':'0::bigint','p_limit':'1::integer',
                      'p_successful_execute_only':'TRUE::boolean'}.items():
        body = re.sub(r'\b'+key+r'\b',value,body)
    cur.execute('SET LOCAL jit=off')
    cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) '+body)
    return cur.fetchone()[0][0]['Plan']


def _indexes(node):
    result=[]
    if node.get('Index Name'):
        result.append(node['Index Name'])
    for child in node.get('Plans',[]):
        result += _indexes(child)
    return result


def test_exact_query_parity_and_partial_index_eliminate_ordinary_run_reads(database):
    db=database
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        _clone_before(cur)
    round_id,candidate=scope.seed(db,'partial552')
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        heads._duplicate_calls(cur,candidate,48,start=552000)

    # A function switch cannot run until the concurrent index is valid.
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        with pytest.raises(db[0].Error,match='index absent, invalid, or wrong shape'):
            cur.execute(QUERY.read_text())
        conn.rollback()
    conn=db[0].connect(**db[1])
    try:
        conn.autocommit=True
        with conn.cursor() as cur:
            cur.execute(INDEX.read_text())
            cur.execute(INDEX.read_text())  # Safe partial-deployment replay.
    finally:
        conn.close()
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(QUERY.read_text())
        cur.execute(QUERY.read_text())
        cur.execute('BEGIN')
        cur.execute("""
            SELECT proname,encode(extensions.digest(pg_get_functiondef(oid),'sha256'),'hex')
            FROM pg_proc WHERE proname=ANY(%s) ORDER BY proname
        """,(list(NAMES),))
        assert [row[1] for row in cur.fetchall()] == list(POST)
        cur.execute('SELECT entry_id FROM public.lab_arena_ledger WHERE round_id=%s '
                    "AND entry_kind='uncertain' AND provider='deepline' ORDER BY entry_id",(round_id,))
        ids=[r[0] for r in cur.fetchall()]
        assert len(ids)==49
        checks=0
        cur.execute('SET LOCAL session_replication_role=replica')
        for case in ('populated','superseded','settled','failed_billable','no_candidate'):
            cur.execute('SAVEPOINT case_state')
            if case=='superseded':
                cur.execute('UPDATE public.lab_arena_ledger SET entry_id=-entry_id '
                            "WHERE entry_kind='uncertain' AND round_id=%s",(round_id,))
            elif case=='settled':
                cur.execute("""
                    INSERT INTO public.lab_arena_ledger
                      (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                       call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
                    SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,
                           call_identity,provider,operation_id,funding_source,0,'{}'::jsonb
                    FROM public.lab_arena_ledger WHERE entry_kind='uncertain'
                      AND provider='deepline' AND round_id=%s
                """,(round_id,))
            elif case=='failed_billable':
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc="
                            "jsonb_set(entry_doc,'{call,call_succeeded}','false'::jsonb) "
                            "WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s",
                            (round_id,))
            elif case=='no_candidate':
                cur.execute("UPDATE public.lab_arena_ledger SET provider='openrouter' "
                            "WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s",
                            (round_id,))
            for name in NAMES:
                for flag in ((None,) if name.endswith('_v1') else (True,False,None)):
                    for cursor in (0,ids[24],ids[-1]):
                        for run_id in ('',candidate['run_id'],'missing-run'):
                            _pair(cur,round_id,run_id,cursor,1,flag,name=name)
                            checks += 1
            cur.execute('ROLLBACK TO SAVEPOINT case_state')
        assert checks==180
        # Keep the full semantic matrix small. The physical regression then
        # adds production-shaped ordinary history and profiles the same SQL.
        cur.execute("""
            INSERT INTO public.lab_arena_runs
            SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,
              to_jsonb(base) || jsonb_build_object(
                'run_id','test552-run-'||g.n,
                'assignment_id','test552-assignment-'||g.n,
                'claim_request_id',NULL))).*
            FROM public.lab_arena_runs base CROSS JOIN generate_series(1,100) g(n)
            WHERE base.run_id=%s
        """,(candidate['run_id'],))
        cur.execute("""
            INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)
            SELECT 'uncertain',base.miner_hotkey,base.round_id,base.submission_id,
                   'test552-run-'||g.n,base.stage,
                   'sha256:'||lpad(to_hex(g.n*1000+e.n+4000000),64,'0'),
                   'openrouter',base.operation_id,base.funding_source,0,base.entry_doc
            FROM (SELECT * FROM public.lab_arena_ledger WHERE entry_id=%s) base
            CROSS JOIN generate_series(1,100) g(n)
            CROSS JOIN generate_series(1,300) e(n)
        """,(candidate['uncertain_entry_id'],))
        cur.execute('ANALYZE public.lab_arena_runs')
        cur.execute('ANALYZE public.lab_arena_ledger')
        original={name:_body_plan(cur,name,round_id,before=True) for name in NAMES}
        optimized={name:_body_plan(cur,name,round_id) for name in NAMES}
        for name in NAMES:
            assert original[name]['Shared Hit Blocks'] > 10000
            assert PARTIAL_INDEX not in _indexes(original[name])
            assert PARTIAL_INDEX in _indexes(optimized[name])
            assert optimized[name]['Shared Hit Blocks'] < 1000
            assert optimized[name]['Shared Hit Blocks'] * 20 < original[name]['Shared Hit Blocks']
        cur.execute('SAVEPOINT no_candidates')
        cur.execute("UPDATE public.lab_arena_ledger SET provider='openrouter' "
                    "WHERE entry_kind='uncertain' AND provider='deepline' AND round_id=%s",
                    (round_id,))
        cur.execute('ANALYZE public.lab_arena_ledger')
        for name in NAMES:
            old=_body_plan(cur,name,round_id,before=True)
            new=_body_plan(cur,name,round_id)
            assert old['Shared Hit Blocks'] > 10000
            assert PARTIAL_INDEX in _indexes(new)
            assert new['Shared Hit Blocks'] < 500
            assert new['Shared Hit Blocks'] * 20 < old['Shared Hit Blocks']
        cur.execute('ROLLBACK TO SAVEPOINT no_candidates')
