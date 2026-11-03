"""Exact three counted execution retries retain original cost and call limits."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid

import pytest

from lab_arena import contracts,scoring
from lab_arena.store import ArenaStore,ArenaStoreError,PsycopgTransport,hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import current_service_database,SCRIPTS
from tests.lab_arena.stage_cutoff_drain420_postgres_test import fixture_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL=SCRIPTS/'609-arena-2026-11-03-three-unfinished-executions.sql'
ROUND='arena-2026-11-03'
SUB='sub-f013f522a7c764e18df792e094e7b820'
MINER=hotkey('operator609-miner')
RUNNER=hotkey('operator609-runner')
TOKEN=hash_lease_token('operator609-token')
ORIGINALS={}


def assignment(pos):return f'{ROUND}:{SUB}:2:{pos}'
def scalar(cur,sql,args=()):cur.execute(sql,args);return cur.fetchone()[0]
def pg_hash(cur,doc):return scalar(cur,"SELECT encode(extensions.digest(%s::jsonb::text,'sha256'),'hex')",(json.dumps(doc),))


@pytest.fixture(scope='module')
def database():
    g=current_service_database();pg,dsn=next(g)
    try:
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute((SCRIPTS/'596-lab-arena-bounded-judge-attempts.sql').read_text())
            cur.execute((SCRIPTS/'597-lab-arena-zero-call-judge-attempt-credit.sql').read_text())
            cur.execute((SCRIPTS/'604-arena-2026-11-03-forward-only-verifier.sql').read_text().split('-- Exact one-time live preimage604')[0]);conn.commit()
            cur.execute((SCRIPTS/'607-arena-2026-11-03-baseline8-corrected-judge.sql').read_text().split('-- Exact one-time corrected attempt607')[0]);conn.commit()
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            for signature in ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
                    'public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
                    'public.lab_arena_expire_leases(text)',
                    'public.lab_arena_execution_attempt_count(public.lab_arena_runs)'):
                ORIGINALS[signature]=scalar(cur,'SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
        yield pg,dsn
    finally:g.close()


def store_for(pg,dsn):return ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))


def payment(store,run,amount,*,provider='openrouter'):
    identity='sha256:'+uuid.uuid4().hex*2
    args=dict(run_id=run,lease_token_hash=TOKEN,call_identity=identity)
    result=store.reserve_call(**args,operation_id='openrouter.chat' if provider=='openrouter' else 'deepline.execute',provider=provider,funding_source='miner_key',amount_microusd=amount,call_doc={},lease_ttl_seconds=4500)
    if result['status']!='reserved':return result
    assert store.mark_dispatched(**args)['status']=='dispatched'
    return store.settle_call(**args,actual_microusd=amount,terminal_response={'status':200,'call_succeeded':True},lease_ttl_seconds=4500)


def seed(pg,dsn,*,quota=2,old_calls=3,old_total=300000):
    cfg=fixture_configuration(ROUND)
    cfg.pop('contact_policy',None)
    cfg.update(execution_sequence_policy='baseline_scored_first_v1',sourcing_cost_eligibility_policy='successful_calls_flat_per_icp_v1',
        reasoning_capture_policy='full_generated_thinking_v1',infrastructure_retry_cost_policy='trusted_infrastructure_v1',
        execution_icp_cap_microusd=6000000,cost_per_company_microusd=6000000,max_attempts_per_assignment=2,
        stage_1_icp_count=5,stage_2_icp_count=5,lease_ttl_seconds=4500,runner_slot_ceiling=20,
        checkpoint_deadline_policy='atomic_checkpoint_60m_v1',icp_wall_clock_seconds=3600,
        call_quotas={'openrouter':quota,'scrapingdog':quota,'deepline':0},
        deepline_catalog={'schema_version':'leadpoet.lab_arena.deepline_catalog.v1'},
        company_judgment_policy='company_judgment_v1',
        scorer_policy=scoring.build_scorer_policy(company_judgment_cache=True,scoring_adapter_version='qualification_integrity_v2'))
    future=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat().replace('+00:00','Z')
    cfg['schedule']['stage_2_close']=future
    cfg['schedule']['final_scoring_close']=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat().replace('+00:00','Z')
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events CASCADE')
        cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]',candidate_commit='',restart_scope='',restart_phase=''")
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,evaluation_date,icp_set_date,benchmark_ref,confirmation_bank_ref,confirmation_bank_hash) VALUES (%s,'stage2',5,4,%s::jsonb,%s::jsonb,'2026-11-03','2026-11-02','arena/arena-2026-11-03/benchmark.json','arena/confirmation.json','sha256:'||repeat('a',64))",(ROUND,json.dumps(cfg),json.dumps([dict(submission_id=SUB,miner_hotkey=MINER,is_king=False)])))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,source_ref,source_size_bytes,source_archive_sha256,source_normalized_sha256,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES (%s,%s,%s,'frozen',%s,123,%s,'sha256:'||repeat('a',64),'passed','{\"decision\":\"pass\"}',%s,now(),1)",(SUB,ROUND,MINER,f'arena/{ROUND}/sources/{SUB}.tar.gz','sha256:'+'a'*64,'sha256:'+'b'*64))
        for pos in (4,5,6):
            for n in (1,2):
                claim={'status':'leased','lease_generation':n,'lease_expires_at':future}
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,runner_hotkey,stage,icp_position,attempt,status,kind,stage_generation,scoring_generation_floor,lease_generation,lease_token_hash,lease_expires_at,claim_request_id,claim_request_hash,claim_response,terminal_cause) VALUES (%s,%s,%s,%s,%s,%s,2,%s,%s,'failed','execute',4,0,%s,%s,now()-interval '1 hour',%s,%s,%s::jsonb,'lease_expired')",(assignment(pos)+':'+str(n),assignment(pos),ROUND,SUB,MINER,RUNNER,pos,n,n,TOKEN,uuid.uuid4().hex,'sha256:'+'c'*64,json.dumps(claim)))
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,kind,stage_generation,scoring_generation_floor,terminal_cause,output_ref,per_icp_score,qualification_doc) VALUES (%s,%s,%s,%s,%s,2,7,1,'accepted','execute',4,0,'accepted','retained-completed.json',0,'{\"companies\":[]}')",(assignment(7)+':1',assignment(7),ROUND,SUB,MINER))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,1,'nov03-forward-only-verifier604-v1',%s,'fixture609',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}','{\"operation\":\"nov03_forward_only_verifier_v1\"}' FROM public.lab_arena_rounds r WHERE round_id=%s",('sha256:'+'d'*64,ROUND))
    store=store_for(pg,dsn)
    try:
        for pos in (4,5,6):
            for i in range(old_calls):
                n=1 if i<min(quota,old_calls) else 2;run=assignment(pos)+':'+str(n)
                with pg.connect(**dsn) as conn,conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour' WHERE run_id=%s",(run,))
                amount=old_total//old_calls+(old_total%old_calls if i==old_calls-1 else 0)
                assert payment(store,run,amount)['status']=='settled'
                with pg.connect(**dsn) as conn,conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',lease_expires_at=now()-interval '1 hour' WHERE run_id=%s",(run,))
    finally:store._transport.close()
    return cfg


def render(conn,cfg):
    with conn.cursor() as cur:
        rows=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(r)) FROM public.lab_arena_runs r WHERE round_id=%s',(ROUND,))
        source=scalar(cur,"SELECT jsonb_build_object('round_id',round_id,'submission_id',submission_id,'miner_hotkey',miner_hotkey,'source_ref',source_ref,'source_size_bytes',source_size_bytes,'source_archive_sha256',source_archive_sha256) FROM public.lab_arena_submissions WHERE submission_id=%s",(SUB,))
        manifest={'original_run_hashes':{rid:pg_hash(cur,row) for rid,row in rows.items() if row['icp_position'] in (4,5,6)},'source_identity':source}
        sql=SQL.read_text();literal=re.search(r'\$manifest609\$(.*?)\$manifest609\$',sql,re.S)[1]
        return sql.replace(literal,json.dumps(manifest,separators=(',',':'),sort_keys=True)).replace('2026-11-03T14:00:02Z',cfg['schedule']['stage_2_close']),rows


def apply(conn,sql):
    with conn.cursor() as cur:cur.execute(sql)
    conn.commit()


def claim(conn):
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',(ROUND,hotkey('healthy609'),uuid.uuid4().hex,'sha256:'+uuid.uuid4().hex*2,TOKEN))
        lease=cur.fetchone()[0];cur.execute('RESET ROLE')
    conn.commit();return lease


def snapshot(cur):
    return {table:scalar(cur,f'SELECT jsonb_agg(to_jsonb(t) ORDER BY {key}) FROM public.{table} t') for table,key in [('lab_arena_rounds','round_id'),('lab_arena_submissions','submission_id'),('lab_arena_runs','run_id'),('lab_arena_ledger','entry_id'),('lab_arena_restart_claim_control','singleton')]}


def test_exact_three_append_replay_and_preserved_completed_work(database):
    pg,dsn=database;cfg=seed(pg,dsn)
    with pg.connect(**dsn) as conn:
        sql,rows=render(conn,cfg)
        with conn.cursor() as cur:before=snapshot(cur)
        assert claim(conn)['status']=='no_pending'
        apply(conn,sql)
        with conn.cursor() as cur:
            after=snapshot(cur)
            for table in before:
                if table!='lab_arena_runs':assert after[table]==before[table]
            assert len(after['lab_arena_runs'])==len(rows)+3
            for signature,definition in ORIGINALS.items():
                if 'lab_arena_reserve_call(' not in signature:
                    assert scalar(cur,'SELECT pg_get_functiondef(%s::regprocedure)',(signature,))==definition
            history=scalar(cur,'SELECT public.lab_arena_scoring_policy_history_v1(%s)',(ROUND,))
            assert history['revisions'][-1]['generation_floor']==0 and history['revisions'][-1]['receipt_generation']==2
            assert {r['run_id']:r for r in after['lab_arena_runs'] if r['attempt']!=3}==rows
            assert [scalar(cur,'SELECT public.lab_arena_execution_attempt_count(r) FROM public.lab_arena_runs r WHERE run_id=%s',(assignment(pos)+':3',)) for pos in (4,5,6)]==[3,3,3]
        apply(conn,sql)
        with conn.cursor() as cur:assert snapshot(cur)==after
        store=store_for(pg,dsn)
        try:
            for _ in range(3):
                lease=claim(conn);assert lease['status']=='leased' and lease['attempt']==3,lease
                assert lease['icp_position'] in (4,5,6)
                result=store.complete_attempt(run_id=lease['run_id'],lease_token_hash=TOKEN,result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref='fresh-unfinished-output.json')
                assert result['status']=='accepted' and 'confirmation_attempt' not in result
            assert claim(conn)['status']=='no_pending'
            apply(conn,sql)
            assert store.get_run(assignment(7)+':1')==rows[assignment(7)+':1']
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s",(ROUND,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            apply(conn,sql)  # A completed-stage replay never requeues outputs.
        finally:store._transport.close()


def test_original_assignment_quota_includes_prior_two_and_survives_invalid_proof(database):
    pg,dsn=database;cfg=seed(pg,dsn,quota=2,old_calls=3)
    with pg.connect(**dsn) as conn:
        sql,_=render(conn,cfg);apply(conn,sql);lease=claim(conn);assert lease['icp_position']==4
        store=store_for(pg,dsn)
        try:
            assert payment(store,lease['run_id'],100)['status']=='settled'
            # Failing-before witness: the unchanged legacy condition sees
            # count==physical3 and admits a fifth call beyond original2*2.
            with conn.cursor() as cur:
                cur.execute('SAVEPOINT legacy_quota609')
                cur.execute(ORIGINALS['public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'])
                cur.execute('SET ROLE lab_arena_service')
                cur.execute("SELECT public.lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',1,'{}',4500)",(lease['run_id'],TOKEN,'sha256:'+uuid.uuid4().hex*2))
                assert cur.fetchone()[0]['status']=='reserved'
                cur.execute('RESET ROLE');cur.execute('ROLLBACK TO SAVEPOINT legacy_quota609')
            conn.commit()
            denied=payment(store,lease['run_id'],1)
            assert (denied['status'],denied['reason'])==('refused','per_icp_quota')
            with conn.cursor() as cur:
                assert scalar(cur,"SELECT public.lab_arena__execute_assignment_calls_v1(%s,'openrouter')",(assignment(4),))==4
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_scoring_supersessions SET recovery_doc=jsonb_set(recovery_doc,ARRAY['original_run_hashes',%s],to_jsonb(repeat('0',64))) WHERE request_id='nov03-three-unfinished-executions609-v1'",(assignment(4)+':1',))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            with conn.cursor() as cur:
                assert scalar(cur,'SELECT public.lab_arena__nov03_unfinished_execute609_v1(r,a) FROM public.lab_arena_rounds r,public.lab_arena_runs a WHERE r.round_id=%s AND a.run_id=%s',(ROUND,lease['run_id'])) is False
            denied=payment(store,lease['run_id'],1)
            assert (denied['status'],denied['reason'])==('refused','per_icp_quota')
        finally:store._transport.close()


def test_original_six_dollar_cost_is_not_reset(database):
    pg,dsn=database;cfg=seed(pg,dsn,quota=100,old_total=5900000)
    with pg.connect(**dsn) as conn:
        sql,_=render(conn,cfg);apply(conn,sql);lease=claim(conn)
        store=store_for(pg,dsn)
        try:
            with conn.cursor() as cur:
                cost=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,4)',(ROUND,SUB))
                assert cost['competition_successful_microusd']==5900000
                assert cost['excluded_infrastructure_successful_microusd']==0
            assert payment(store,lease['run_id'],100000)['status']=='settled'
            denied=payment(store,lease['run_id'],1)
            assert (denied['status'],denied['reason'])==('refused','money_cap')
            with conn.cursor() as cur:
                cost=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,4)',(ROUND,SUB))
                assert cost['competition_successful_microusd']==6000000
            apply(conn,sql)
        finally:store._transport.close()


@pytest.mark.parametrize('variant',['source','old_row','accepted','pending','deadline','spent'])
def test_changed_or_completed_preimage_never_creates_a_retry(database,variant):
    pg,dsn=database;cfg=seed(pg,dsn,old_total=6000000 if variant=='spent' else 300000)
    with pg.connect(**dsn) as conn:
        sql,_=render(conn,cfg)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if variant=='source':
                cur.execute("UPDATE public.lab_arena_submissions SET source_archive_sha256='sha256:'||repeat('f',64) WHERE submission_id=%s",(SUB,))
            elif variant=='old_row':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id=%s",(assignment(4)+':2',))
            elif variant in ('accepted','pending'):
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,terminal_cause,output_ref) VALUES (%s,%s,%s,%s,%s,2,4,3,'execute',%s,4,0,%s,%s)",(assignment(4)+':3',assignment(4),ROUND,SUB,MINER,variant,'accepted' if variant=='accepted' else None,'completed.json' if variant=='accepted' else None))
            elif variant=='deadline':
                past=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb(%s::text)) WHERE round_id=%s",(past,ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with conn.cursor() as cur:before=snapshot(cur)
        with pytest.raises(pg.Error,match='Nov03 unfinished execution'):
            apply(conn,sql)
        conn.rollback()
        with conn.cursor() as cur:
            assert snapshot(cur)==before
            assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov03-three-unfinished-executions609-v1'")==0


def event(store,run,kind,content):
    assert store.append_trajectory_events(run,TOKEN,[dict(event_id=str(uuid.uuid4()),kind=kind,content=content,occurred_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat())])['status']=='accepted'


@pytest.mark.parametrize('outcome',['provider_error','model_error','credential_error','budget_exhausted','expired','host','champion','host_champion'])
def test_third_terminal_paths_never_create_fourth(database,outcome):
    pg,dsn=database;cfg=seed(pg,dsn)
    with pg.connect(**dsn) as conn:
        sql,old_rows=render(conn,cfg);apply(conn,sql);lease=claim(conn)
        run=lease['run_id'];store=store_for(pg,dsn)
        try:
            if outcome in ('champion','host_champion'):
                with pytest.raises(ArenaStoreError,match='fallback_not_champion'):
                    store.mark_champion_provider_fallback(run,TOKEN,'openrouter',{})
            if outcome in ('host','host_champion'):
                event(store,run,'runtime.started',dict(status='starting',lease_generation=lease['lease_generation']))
                event(store,run,'runtime.error',dict(status='abandoned',failure_stage='runtime',error_class='RuntimeHostError',runtime_host_reason='sandbox_launcher_signaled',launch_exit_code=-2,launch_timed_out=False))
            if outcome in ('champion','host_champion','expired'):
                with conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    if outcome in ('champion','host_champion'):
                        # A malformed-data control. The ordinary public RPC
                        # above rejects this flag for the exact miner identity.
                        cur.execute('UPDATE public.lab_arena_runs SET champion_restart_required=true WHERE run_id=%s',(run,))
                    cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(run,))
                    cur.execute('SET LOCAL session_replication_role=origin')
                conn.commit()
            if outcome in ('expired','host','champion','host_champion'):
                store.expire_leases(ROUND)
                assert store.get_run(run)['status']=='failed'
            else:
                result=store.complete_attempt(run_id=run,lease_token_hash=TOKEN,result={'terminal_status':outcome},terminal_cause=outcome,output_ref=None)
                assert result['status']=='failed' and 'confirmation_attempt' not in result
            store.expire_leases(ROUND)
            credit=store.retry_credit_failures(ROUND,SUB,MINER,'sha256:'+uuid.uuid4().hex*2)
            assert credit['requeued_count']==0
            with conn.cursor() as cur:
                expected=2 if outcome=='host' else 3
                assert scalar(cur,'SELECT public.lab_arena_execution_attempt_count(a) FROM public.lab_arena_runs a WHERE run_id=%s',(run,))==expected
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>3',(assignment(4),))==0
                for rid,old in old_rows.items():assert scalar(cur,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(rid,))==old
        finally:store._transport.close()


def test_simultaneous_application_creates_one_receipt_and_three_rows(database):
    pg,dsn=database;cfg=seed(pg,dsn)
    with pg.connect(**dsn) as conn:sql,rows=render(conn,cfg)
    barrier=Barrier(2)
    def worker(_):
        with pg.connect(**dsn) as conn:
            barrier.wait(timeout=10)
            apply(conn,sql)
        return True
    with ThreadPoolExecutor(max_workers=2) as pool:assert list(pool.map(worker,range(2)))==[True,True]
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov03-three-unfinished-executions609-v1'")==1
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND attempt=3",(ROUND,))==3
        assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(r)) FROM public.lab_arena_runs r WHERE attempt<>3')==rows


def test_parallel_reservations_share_the_last_original_assignment_slot(database):
    pg,dsn=database;cfg=seed(pg,dsn,quota=2,old_calls=3)
    with pg.connect(**dsn) as conn:
        sql,_=render(conn,cfg);apply(conn,sql);lease=claim(conn)
    barrier=Barrier(2)
    def worker(_):
        store=store_for(pg,dsn)
        try:
            barrier.wait(timeout=10)
            return store.reserve_call(run_id=lease['run_id'],lease_token_hash=TOKEN,call_identity='sha256:'+uuid.uuid4().hex*2,
                operation_id='openrouter.chat',provider='openrouter',funding_source='miner_key',amount_microusd=1,call_doc={},lease_ttl_seconds=4500)
        finally:store._transport.close()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,range(2)))
    assert sorted(r['status'] for r in results)==['refused','reserved']
    assert [r['reason'] for r in results if r['status']=='refused']==['per_icp_quota']
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        assert scalar(cur,"SELECT public.lab_arena__execute_assignment_calls_v1(%s,'openrouter')",(assignment(4),))==4
