"""Append eleven operator attempts only after exact natural expiry."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json
import re
import uuid

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena import nov03_three_executions609_postgres_test as prior
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL=prior.SCRIPTS/'611-arena-2026-11-03-g3-natural-expiry-executions.sql'
ROUND=prior.ROUND;A4='sub-a4fe920d6204d92824d54d9cd0149bc6';F0=prior.SUB
scalar=prior.scalar;database=prior.database
TARGETS=[(A4,p,2) for p in range(2,10)]+[(F0,p,3) for p in (4,5,6)]


def aid(s,p):return f'{ROUND}:{s}:2:{p}'

def seed(pg,dsn,*,old_total=300000):
    # Restore the exact pre-609 entry points between cases.
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        for sig,d in prior.ORIGINALS.items():cur.execute(d)
        history=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)")
        history=history.replace(",'nov03_three_unfinished_executions609_v1'",'').replace(",'nov03_baseline8_checkpoint_continuation610_v1'",'').replace(",'nov03_g3_natural_expiry_executions611_v1'",'')
        cur.execute(history)
    cfg=prior.seed(pg,dsn,quota=4,old_calls=3,old_total=old_total)
    cfg['schedule']['publication_deadline']=cfg['schedule']['final_scoring_close']
    with pg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
            cur.execute("UPDATE public.lab_arena_scoring_supersessions SET new_round_doc=jsonb_set(new_round_doc,'{configuration_doc}',%s::jsonb) WHERE round_id=%s",(json.dumps(cfg),ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        sql,_=prior.render(conn,cfg);prior.apply(conn,sql)
        with conn.cursor() as cur:
            cur.execute((prior.SCRIPTS/'610-arena-2026-11-03-baseline8-checkpoint-continuation.sql').read_text().split('-- Exact one-time append.')[0]);conn.commit()
            cur.execute('SET LOCAL session_replication_role=replica')
            source=scalar(cur,'SELECT to_jsonb(s) FROM public.lab_arena_submissions s WHERE submission_id=%s',(F0,))
            source.update(submission_id=A4,miner_hotkey=hotkey('operator611-a4'),source_ref=f'arena/{ROUND}/sources/{A4}.tar.gz')
            cur.execute('INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,%s::jsonb)).*',(json.dumps(source),))
            cur.execute("UPDATE public.lab_arena_rounds SET participants=participants||%s::jsonb WHERE round_id=%s",(json.dumps([{'submission_id':A4,'miner_hotkey':source['miner_hotkey'],'is_king':False}]),ROUND))
            cur.execute("UPDATE public.lab_arena_rounds r SET participants=(SELECT jsonb_agg(p||jsonb_build_object('source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes)) FROM jsonb_array_elements(r.participants)p JOIN public.lab_arena_submissions s ON s.submission_id=p->>'submission_id') WHERE r.round_id=%s",(ROUND,))
            template=scalar(cur,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(prior.assignment(4)+':1',))
            for sid,pos,n in TARGETS:
                if sid==A4:
                    for attempt in (1,2):
                        row={**template,'run_id':aid(sid,pos)+':'+str(attempt),'assignment_id':aid(sid,pos),'submission_id':sid,'miner_hotkey':source['miner_hotkey'],'icp_position':pos,'attempt':attempt,'lease_generation':attempt,'claim_request_id':uuid.uuid4().hex}
                        if attempt==2:row.update(terminal_doc=None)
                        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(row),))
                else:
                    cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_expires_at=now()-interval '1 minute',claim_request_id=%s,claim_request_hash=%s,claim_response=jsonb_build_object('status','leased','lease_generation',lease_generation,'lease_expires_at',now()-interval '1 minute') WHERE run_id=%s",(prior.RUNNER,prior.TOKEN,uuid.uuid4().hex,'sha256:'+'c'*64,aid(sid,pos)+':3'))
            cur.execute('SET LOCAL session_replication_role=origin')
    store=prior.store_for(pg,dsn)
    try:
        for sid,pos,n in TARGETS:
            if sid!=A4:continue
            old=aid(sid,pos)+':1'
            with pg.connect(**dsn) as conn,conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica');cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_expires_at=now()+interval '1 hour' WHERE run_id=%s",(old,))
            assert prior.payment(store,old,old_total)['status']=='settled'
            with pg.connect(**dsn) as conn,conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica');cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',lease_expires_at=now()-interval '1 hour' WHERE run_id=%s",(old,))
    finally:store._transport.close()
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL WHERE submission_id=%s AND attempt=2",(A4,))
        cur.execute('SET LOCAL session_replication_role=origin')
        rows=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        manifest=dict(configuration_sha256=prior.pg_hash(cur,cfg),source_identities={},original_failed_hashes={},targets=[])
        for sid in (A4,F0):manifest['source_identities'][sid]=scalar(cur,"SELECT jsonb_build_object('round_id',round_id,'submission_id',submission_id,'miner_hotkey',miner_hotkey,'source_ref',source_ref,'source_size_bytes',source_size_bytes,'source_archive_sha256',source_archive_sha256) FROM public.lab_arena_submissions WHERE submission_id=%s",(sid,))
        for sid,pos,n in TARGETS:
            assignment=aid(sid,pos);parent=rows[assignment+':'+str(n)]
            for attempt in range(1,n):manifest['original_failed_hashes'][assignment+':'+str(attempt)]=prior.pg_hash(cur,rows[assignment+':'+str(attempt)])
            manifest['targets'].append(dict(assignment_id=assignment,submission_id=sid,icp_position=pos,parent_attempt=n,parent_run_id=parent['run_id'],new_run_id=assignment+':'+str(n+1),parent_full_sha256=prior.pg_hash(cur,parent),parent_immutable_sha256=prior.pg_hash(cur,{k:v for k,v in parent.items() if k not in ('status','terminal_cause','terminal_doc','updated_at')}),original_terminal_doc=parent['terminal_doc'],lease_expires_at=parent['lease_expires_at']))
        sql=SQL.read_text();literal=re.search(r'\$manifest611\$(.*?)\$manifest611\$',sql,re.S)[1]
        sql=sql.replace(literal,json.dumps(manifest,separators=(',',':'),sort_keys=True)).replace('2026-11-03T14:00:02Z',cfg['schedule']['stage_2_close'])
    return cfg,sql,manifest


def expire(pg,dsn):
    store=prior.store_for(pg,dsn)
    try:store.expire_leases(ROUND)
    finally:store._transport.close()


def apply(pg,dsn,sql):
    with pg.connect(**dsn) as conn:prior.apply(conn,sql)


def mature(pg,dsn):
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE status='pending'")


def claim_all(pg,dsn):
    leases=[]
    with pg.connect(**dsn) as conn:
        for _ in TARGETS:
            result=prior.claim(conn)
            assert result['status']=='leased',result
            leases.append(result)
    return leases


def test_natural_expiry_then_exact_append_replay_and_preservation(database):
    pg,dsn=database;cfg,sql,manifest=seed(pg,dsn)
    with pytest.raises(Exception):apply(pg,dsn,sql)
    expire(pg,dsn)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a')
        ledger=scalar(cur,'SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a')
    apply(pg,dsn,sql);apply(pg,dsn,sql)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        after=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a')
        assert all(after[k]==v for k,v in before.items())
        assert len(after)==len(before)+11
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a')==ledger
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov03-g3-natural-expiry611-v1'")==1
    mature(pg,dsn);leases=claim_all(pg,dsn)
    assert {r['run_id'] for r in leases}=={t['new_run_id'] for t in manifest['targets']}


@pytest.mark.parametrize('outcome',['failed','expired','accepted','credential_error','host','champion','host_champion'])
def test_no_further_automatic_attempt(database,outcome):
    pg,dsn=database;_,sql,_=seed(pg,dsn);expire(pg,dsn);apply(pg,dsn,sql);mature(pg,dsn);leases=claim_all(pg,dsn)
    store=prior.store_for(pg,dsn)
    try:
        if outcome in ('host','host_champion'):
            for lease in leases:
                prior.event(store,lease['run_id'],'runtime.started',dict(status='starting',lease_generation=lease['lease_generation']))
                prior.event(store,lease['run_id'],'runtime.error',dict(status='abandoned',failure_stage='runtime',error_class='RuntimeHostError',runtime_host_reason='sandbox_launcher_signaled',launch_exit_code=-2,launch_timed_out=False))
        if outcome in ('expired','host','champion','host_champion'):
            with pg.connect(**dsn) as conn,conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                if outcome in ('champion','host_champion'):
                    for lease in leases:
                        with pytest.raises(ArenaStoreError,match='fallback_not_champion'):
                            store.mark_champion_provider_fallback(lease['run_id'],prior.TOKEN,'openrouter',{})
                    cur.execute("UPDATE public.lab_arena_runs SET champion_restart_required=true WHERE status='leased'")
                if outcome!='host':cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE status='leased'")
            store.expire_leases(ROUND)
        else:
            for lease in leases:
                cause='accepted' if outcome=='accepted' else 'credential_error' if outcome=='credential_error' else 'provider_error'
                result=store.complete_attempt(run_id=lease['run_id'],lease_token_hash=prior.TOKEN,result={'terminal_status':cause},terminal_cause=cause,output_ref='normal-complete.json' if cause=='accepted' else None)
                assert result['status']==('accepted' if cause=='accepted' else 'failed')
        store.expire_leases(ROUND)
        for lease in leases:
            terminal=store.get_run(lease['run_id'])
            assert terminal['status']==('accepted' if outcome=='accepted' else 'failed')
            if outcome=='host':
                assert terminal['terminal_cause']=='lease_expired'
                with pg.connect(**dsn) as conn,conn.cursor() as cur:
                    assert scalar(cur,'SELECT public.lab_arena_execution_attempt_count(a) FROM public.lab_arena_runs a WHERE run_id=%s',(lease['run_id'],))==lease['attempt']-1
        for sid in (A4,F0):
            miner=hotkey('operator611-a4') if sid==A4 else prior.MINER
            assert store.retry_credit_failures(ROUND,sid,miner,'sha256:'+uuid.uuid4().hex*2)['requeued_count']==0
        apply(pg,dsn,sql)
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            for sid,p,n in TARGETS:assert scalar(cur,'SELECT max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s',(aid(sid,p),))==n+1
    finally:store._transport.close()


@pytest.mark.parametrize('sid,pos,n',[(A4,2,2),(F0,4,3)])
def test_original_aggregate_quota_and_cost_cap_stay_bound(database,sid,pos,n):
    pg,dsn=database;_,sql,_=seed(pg,dsn,old_total=5900000);expire(pg,dsn);apply(pg,dsn,sql);mature(pg,dsn);claim_all(pg,dsn)
    store=prior.store_for(pg,dsn);run=aid(sid,pos)+':'+str(n+1)
    try:
        assert prior.payment(store,run,100000)['status']=='settled'
        assert prior.payment(store,run,1)['status']=='refused'
    finally:store._transport.close()


def test_concurrent_exact_application_is_idempotent(database):
    pg,dsn=database;_,sql,_=seed(pg,dsn);expire(pg,dsn);barrier=Barrier(2)
    def worker(_):barrier.wait(timeout=5);apply(pg,dsn,sql)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov03-g3-natural-expiry611-v1'")==1


@pytest.mark.parametrize('mutation',["UPDATE public.lab_arena_rounds SET participants=jsonb_set(participants,'{0,source_ref}','\"tampered\"'::jsonb)", "UPDATE public.lab_arena_rounds SET status='stage2_closed'", "UPDATE public.lab_arena_rounds SET published_at=now()", "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb((now()-interval '1 second')::text))", "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true", "UPDATE public.lab_arena_runs SET terminal_doc=terminal_doc||'{\"recovery_reason\":\"fake\"}' WHERE run_id LIKE '%:3'", "UPDATE public.lab_arena_runs SET terminal_doc='{\"expired_at\":null}' WHERE run_id LIKE '%:3'", "UPDATE public.lab_arena_runs SET lease_expires_at=lease_expires_at+interval '1 second' WHERE run_id LIKE '%:3'"])
def test_stage_deadline_pause_or_non_natural_transition_rolls_back(database,mutation):
    pg,dsn=database;_,sql,_=seed(pg,dsn);expire(pg,dsn)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica');cur.execute(mutation)
    with pytest.raises(Exception):apply(pg,dsn,sql)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov03-g3-natural-expiry611-v1'")==0


@pytest.mark.parametrize('sid,pos,n',[(A4,2,2),(F0,4,3)])
@pytest.mark.parametrize('parallel',[False,True])
def test_last_original_aggregate_quota_slot_and_receipt_independence(database,sid,pos,n,parallel):
    pg,dsn=database;_,sql,_=seed(pg,dsn);expire(pg,dsn);apply(pg,dsn,sql);mature(pg,dsn);claim_all(pg,dsn)
    assignment=aid(sid,pos);run=assignment+':'+str(n+1)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        calls=scalar(cur,"SELECT public.lab_arena__execute_assignment_calls_v1(%s,'openrouter')",(assignment,))
        # Independent immutable historical-call fixtures fill seven of the
        # original two*four slots without consuming a new attempt's quota.
        for _ in range(7-calls):
            identity='sha256:'+uuid.uuid4().hex*2
            for kind in ('reservation','settlement'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0,'{\"response\":{\"status\":200,\"call_succeeded\":true}}' FROM public.lab_arena_runs WHERE run_id=%s",(kind,identity,assignment+':1'))
    store=prior.store_for(pg,dsn)
    try:
        if parallel:
            barrier=Barrier(2)
            def compete(_):
                local=prior.store_for(pg,dsn)
                try:barrier.wait(timeout=5);return prior.payment(local,run,1)
                finally:local._transport.close()
            with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(compete,range(2)))
            assert sorted(r['status'] for r in results)==['refused','settled']
        else:
            assert prior.payment(store,run,1)['status']=='settled'
        # Before this exact restriction, a counted new attempt admits a ninth
        # assignment call even though it has room in its per-run quota.
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute('SAVEPOINT before611')
            current=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure)")
            old=re.search(r'\$old611\$(.*?)\$old611\$',SQL.read_text(),re.S)[1]
            new=re.search(r'\$new611\$(.*?)\$new611\$',SQL.read_text(),re.S)[1]
            assert current.count(new)==1
            cur.execute(current.replace(new,old));cur.execute('SET ROLE lab_arena_service')
            cur.execute("SELECT public.lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',1,'{}',4500)",(run,prior.TOKEN,'sha256:'+uuid.uuid4().hex*2))
            assert cur.fetchone()[0]['status']=='reserved'
            cur.execute('RESET ROLE');cur.execute('ROLLBACK TO SAVEPOINT before611')
        denied=prior.payment(store,run,1)
        assert (denied['status'],denied['reason'])==('refused','per_icp_quota')
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_scoring_supersessions SET recovery_doc='{}' WHERE request_id='nov03-g3-natural-expiry611-v1'")
        assert prior.payment(store,run,1)['reason']=='per_icp_quota'
    finally:store._transport.close()
