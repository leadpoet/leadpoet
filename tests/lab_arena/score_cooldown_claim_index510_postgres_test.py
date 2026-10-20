"""The exact existing cooldown predicate uses a narrow index without new rules."""
from pathlib import Path
import json
import re

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import legacy_company_cache501_postgres_test as company
from tests.lab_arena import company_judgments_postgres_test as original
from tests.lab_arena import signal_judgments503_postgres_test as schema
from tests.lab_arena import claim_eligibility_fast_paths509_postgres_test as admission

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/510-lab-arena-score-cooldown-claim-index.sql'
INDEX='public.lab_arena_runs_score_cooldown_claim_idx'
RID='arena-2099-10-20-index510'
RUNNER=original.hotkey('index510-runner')
OTHER=original.hotkey('index510-other')


@pytest.fixture(scope='module')
def database():
    gen=schema.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute((ROOT/'scripts/504-lab-arena-zero-call-host-attempt-credit.sql').read_text())
            cur.execute(admission.SQL.read_text())
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(admission.SIG,))
            import hashlib
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest()==admission.POST
        yield db
    finally:gen.close()


def test_one_concurrent_statement_no_clock_or_function_change():
    text=re.sub(r'--[^\n]*','',SQL.read_text()).strip()
    assert text.startswith('CREATE INDEX CONCURRENTLY ')
    assert text.count(';')==1 and text.endswith(';')
    assert not re.search(r'\b(BEGIN|COMMIT|IF\s+NOT\s+EXISTS|FUNCTION|clock_timestamp)\b',text)


def _guard(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(admission.SIG,))
    source=cur.fetchone()[0]
    start='       SELECT 1 FROM public.lab_arena_runs AS expired_score\n'
    end='      ))\n\n    AND ('
    assert source.count(start)==source.count(end)==1
    body=start+source.split(start,1)[1].split(end,1)[0]
    for field in ('stage_generation','configuration_doc'):
        body=body.replace('v_round.'+field,'(context.v_round).'+field)
    for name in ('p_round_id','p_runner_hotkey','v_stage'):
        body=re.sub(r'(?<![.\w])'+name+r'\b','context.'+name,body)
    # The production candidate filter correlates this unchanged subquery and
    # executes it once per remaining candidate. Varying the stage expression
    # keeps that correlation visible for a reproducible 150-candidate test.
    return """SELECT candidate.i,NOT EXISTS (
"""+body+"""
) AS guard_passes
FROM generate_series(1,150) candidate(i)
CROSS JOIN LATERAL (
 SELECT r AS v_round,r.round_id AS p_round_id,%s::text AS p_runner_hotkey,
   CASE WHEN candidate.i BETWEEN 1 AND 150 THEN 1 ELSE 2 END AS v_stage
 FROM public.lab_arena_rounds r WHERE r.round_id=%s
) context ORDER BY candidate.i"""


def _insert(cur, source, label, count, runner, **changes):
    row={'runner_hotkey':runner,'stage':1,'stage_generation':2,'kind':'score',
         'status':'failed','terminal_cause':'lease_expired','lease_generation':1,
         'lease_expires_at':'2000-01-01T00:00:00Z','result_doc':None,'output_ref':None,
         'company_judgment_refs':None,'claim_response':None}
    row.update(changes)
    cur.execute('SELECT attname FROM pg_attribute WHERE attrelid=\'public.lab_arena_runs\'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=\'\' ORDER BY attnum')
    names=','.join(x[0] for x in cur.fetchall())
    cur.execute('SET session_replication_role=replica')
    cur.execute('INSERT INTO public.lab_arena_runs('+names+') SELECT '+names+" FROM generate_series(1,%s) n CROSS JOIN LATERAL jsonb_populate_record(NULL::public.lab_arena_runs, %s::jsonb || jsonb_build_object('assignment_id',%s||n::text,'run_id',%s||n::text||':1')) clone",
                (count,json.dumps({**source,**row}),label,label))
    cur.execute('SET session_replication_role=origin')


def _nodes(value):
    if isinstance(value,dict):
        if value.get('Node Type'):yield value
        for v in value.values():yield from _nodes(v)
    elif isinstance(value,list):
        for v in value:yield from _nodes(v)


def test_exact_correlated_guard_parity_scope_cost_time_and_index_plan(database):
    store=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
    store._test_database=database
    try:
        company.open_new(store,round_id=RID,markers_by_submission=[['a']])
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score'",(RID,));source=cur.fetchone()[0]
        conn=database[0].connect(**database[1]);conn.autocommit=True
        try:
            with conn.cursor() as cur:
                # Many globally eligible-looking expiries belong to another
                # runner. The new exact equality prefix must avoid those rows.
                _insert(cur,source,'noise510:',27169,OTHER)
                cur.execute('ANALYZE public.lab_arena_runs')
                query=_guard(cur);args=(RUNNER,RID)
                cur.execute(query,args);before=cur.fetchall();assert before==[(i,True) for i in range(1,151)]
                old_catalog=company.latest.catalog(cur);old_rows=company.latest.rows(cur)
                cur.execute(SQL.read_text())
                assert company.latest.catalog(cur)==old_catalog
                assert company.latest.rows(cur)==old_rows
                cur.execute(query,args);assert cur.fetchall()==before
                cur.execute('EXPLAIN (ANALYZE,BUFFERS,TIMING OFF,FORMAT JSON) '+query,args)
                plan=cur.fetchone()[0];nodes=list(_nodes(plan))
                hits=[n for n in nodes if n.get('Index Name')==INDEX.split('.')[1]]
                assert hits and sum(n['Actual Loops'] for n in hits)==150
                assert all(n['Actual Rows']==0 for n in hits)
                cur.execute('SELECT pg_get_indexdef(i.indexrelid),i.indisvalid,i.indisready,i.indisunique,am.amname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam WHERE i.indexrelid=%s::regclass',(INDEX,))
                definition,valid,ready,unique,method=cur.fetchone()
                assert definition==("CREATE INDEX lab_arena_runs_score_cooldown_claim_idx ON public.lab_arena_runs USING btree (round_id, runner_hotkey, stage, stage_generation, lease_expires_at) WHERE ((kind = 'score'::text) AND (status = 'failed'::text) AND (terminal_cause = 'lease_expired'::text) AND (result_doc IS NULL) AND (output_ref IS NULL))")
                assert (valid,ready,unique,method)==(True,True,False,'btree')
                with pytest.raises(database[0].errors.DuplicateTable):cur.execute(SQL.read_text())
                # Real zero-call expiries need three DISTINCT assignments in
                # this exact runner/stage/generation/TTL scope to stop claims.
                from datetime import datetime,timedelta,timezone
                recent=(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat()
                _insert(cur,source,'bound510:',2,RUNNER,lease_expires_at=recent)
                cur.execute(query,args);assert all(x[1] for x in cur.fetchall())
                _insert(cur,source,'third510:',1,RUNNER,lease_expires_at=recent)
                cur.execute(query,args);assert not any(x[1] for x in cur.fetchall())
                for column,value in [('stage',2),('stage_generation',999),('runner_hotkey',OTHER),('lease_expires_at','2000-01-01T00:00:00Z'),('lease_expires_at','2099-12-31T00:00:00Z'),('kind','execute'),('terminal_cause','provider_error'),('result_doc',{'terminal_status':'provider_error'}),('output_ref','fixture/retained-output')]:
                    cur.execute('BEGIN');cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute('UPDATE public.lab_arena_runs SET '+column+'=%s WHERE run_id=%s',(json.dumps(value) if column=='result_doc' else value,'third510:1:1'))
                    cur.execute('SET LOCAL session_replication_role=origin')
                    cur.execute(query,args);assert all(x[1] for x in cur.fetchall()),column
                    cur.execute('ROLLBACK')
                # Any actual prior ledger activity excludes that expiry from
                # this specifically zero-call guard; no amount is discarded.
                cur.execute('INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES(\'reservation\',%s,%s,%s,%s,1,%s,\'openrouter\',\'openrouter.chat_completions\',\'host\',7,\'{}\')',(source['miner_hotkey'],RID,source['submission_id'],'third510:1:1','sha256:'+'a'*64))
                cur.execute(query,args);assert all(x[1] for x in cur.fetchall())
        finally:conn.close()
    finally:store.close()
