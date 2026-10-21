"""A newer paid same-key dispatch permits a normal retry, never a verdict."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import hashlib,json,re
import pytest
from tests.lab_arena import recovery_lock_order523_postgres_test as prior
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import reset

SQL=Path(__file__).parents[2]/'scripts/524-lab-arena-proven-funding-recovery.sql'
SIG='public.lab_arena_miner_credit_failure_v1(text)'
FINGERPRINT='sha256:'+'c'*64

@pytest.fixture(scope='module')
def database():
    gen=prior.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as c,c.cursor() as q:q.execute(SQL.read_text());q.execute(SQL.read_text())
        yield db
    finally:gen.close()


def call(c,runid,label,*,outcome='success',fingerprint=FINGERPRINT,funding='miner_key',provider='deepline',at=None,dispatch_at=None,settled_at=None,amount=2000,dispatches=1,native_id=None,cache=False):
    at=at or datetime.now(timezone.utc)-timedelta(minutes=5)
    dispatch_at=dispatch_at or at+timedelta(seconds=1)
    settled_at=settled_at or dispatch_at+timedelta(seconds=1)
    cid='sha256:'+hashlib.sha256((runid+label).encode()).hexdigest()
    accounting={'base_call_identity':cid,'provider_attempt':1,'action_sequence':0}
    if fingerprint is not None:accounting['credential_fingerprint']=fingerprint
    if cache:accounting['judgment_cache_source_call_identity']='sha256:'+'e'*64
    if outcome=='credit':
        evidence={'error_class':'account_credential_failure','reason':'out_of_credit','provider_status':402,**{k:accounting[k]for k in ('base_call_identity','provider_attempt','action_sequence')}}
        response={'status':402,'call_succeeded':False,'account_failure_evidence':evidence,'credit_failure_proof':{'schema_version':'leadpoet.lab_arena.credit_failure_proof.v1','provider':provider,'reason':'out_of_credit','provider_status':402,'actual_microusd':0}}
        amount=0
    elif outcome=='auth':
        response={'status':401,'call_succeeded':False,'account_failure_evidence':{'error_class':'account_credential_failure','provider_status':401,**{k:accounting[k]for k in ('base_call_identity','provider_attempt','action_sequence')}}};amount=0
    elif outcome=='transport':response={'status':503,'call_succeeded':False};amount=0
    else:response={'status':200,'call_succeeded':True,'provider_cost':{'request_id':native_id if native_id is not None else 'native-'+label}}
    with c.cursor() as q:
        q.execute('SELECT miner_hotkey,round_id,submission_id,stage FROM public.lab_arena_runs WHERE run_id=%s',(runid,));owner,rid,sub,stage=q.fetchone()
        rows=[('reservation',accounting,None,at,0)]+[('dispatch',accounting,None,dispatch_at,0)]*dispatches+[('settlement',{},response,settled_at,amount)]
        for kind,doc,terminal,created,value in rows:
            q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response,created_at)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'deepline.execute',%s,%s,%s::jsonb,%s::jsonb,%s)",(kind,owner,rid,sub,runid,stage,cid,provider,funding,value,json.dumps(doc),json.dumps(terminal)if terminal is not None else None,created))
    return cid


def seed(database):
    baseline,_=prior.gate.current_lease(database,error=None,accounting=None)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
        q.execute('SELECT round_id FROM public.lab_arena_runs WHERE run_id=%s',(baseline['run_id'],));rid=q.fetchone()[0]
        q.execute('SET LOCAL ROLE lab_arena_service')
        q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',(rid,prior.gate.queue.revision.company.original.hotkey('ordered520'),[],'b'*32,'sha256:'+'d'*64,prior.gate.TOKEN));lease=q.fetchone()[0]
        assert lease['status']=='leased' and lease['submission_id']!=baseline['submission_id']
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        denial=datetime.now(timezone.utc)-timedelta(minutes=10)
        call(c,lease['run_id'],'exhausted',outcome='credit',at=denial)
        assert prior.complete(q,lease,cause='credential_error')['status']=='failed'
        q.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(lease['run_id'],));run=q.fetchone()[0]
    return run,denial


def proof(q,runid):
    q.execute('SELECT public.lab_arena_miner_credit_failure_v1(%s)',(runid,));return q.fetchone()[0]


def clone(c,run,tag='funded',*,floor=None,submission=None,owner=None):
    with c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("SELECT string_agg(quote_ident(attname),',' ORDER BY attnum) FROM pg_attribute WHERE attrelid='public.lab_arena_runs'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''");columns=q.fetchone()[0]
        patch={'run_id':run['round_id']+':proof:'+tag,'assignment_id':run['round_id']+':proof-assignment:'+tag,'status':'failed','terminal_cause':'judge_error','scoring_generation_floor':run['scoring_generation_floor']if floor is None else floor}
        patch['claim_request_id']=hashlib.sha256(tag.encode()).hexdigest()[:32]
        patch['claim_request_hash']='sha256:'+hashlib.sha256(('claim-'+tag).encode()).hexdigest()
        if submission is not None:patch['submission_id']=submission
        if owner is not None:patch['miner_hotkey']=owner
        q.execute('INSERT INTO public.lab_arena_runs('+columns+')SELECT '+columns+' FROM jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)',(json.dumps({**run,**patch}),))
        q.execute('SET LOCAL session_replication_role=origin')
    return patch['run_id']


def snapshot(q):
    out=[]
    for table in ('lab_arena_runs','lab_arena_ledger','lab_arena_company_judgments'):
        q.execute('SELECT coalesce(jsonb_agg(to_jsonb(x)),\'[]\') FROM public.'+table+' x');out.append(q.fetchone()[0])
    return out


def test_paid_new_dispatch_retires_stop_without_mutating_history(database):
    run,denial=seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        assert proof(q,run['run_id']) is True
        funded=clone(c,run);call(c,funded,'fresh')
        before=snapshot(q)
        assert proof(q,run['run_id']) is False
        assert snapshot(q)==before
        q.execute('SELECT public.lab_arena__judge_continuation_v1(r,s) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs s USING(round_id) WHERE s.run_id=%s',(run['run_id'],));assert q.fetchone()[0] is True


@pytest.mark.parametrize('fault',['other_key','missing_key','host','other_provider','old_floor','future_floor','other_submission','other_owner','free','unpriced','failed','no_dispatch','old_dispatch','no_native','blank_native','tab_native','newline_native','cache','terminal_cache','duplicate_native','auth_after_denial','new_credit_after_success'])
def test_untrusted_or_stale_paid_proof_never_retires_credit(database,fault):
    run,denial=seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        funded=clone(c,run,floor=run['scoring_generation_floor']-1 if fault=='old_floor' else run['scoring_generation_floor']+1 if fault=='future_floor' else None,submission='different-submission'if fault=='other_submission'else None,owner='different-owner'if fault=='other_owner'else None)
        kw={}
        if fault=='other_key':kw['fingerprint']='sha256:'+'d'*64
        if fault=='missing_key':kw['fingerprint']=None
        if fault=='host':kw['funding']='host'
        if fault=='other_provider':kw['provider']='openrouter'
        if fault=='free':kw['amount']=0
        if fault=='failed':kw['outcome']='transport'
        if fault=='no_dispatch':kw['dispatches']=0
        if fault=='duplicate_dispatch':kw['dispatches']=2
        if fault=='old_dispatch':kw.update(at=denial-timedelta(minutes=2),dispatch_at=denial-timedelta(minutes=1),settled_at=denial+timedelta(minutes=2))
        if fault=='no_native':kw['native_id']=''
        if fault=='blank_native':kw['native_id']='   '
        if fault=='tab_native':kw['native_id']='\t'
        if fault=='newline_native':kw['native_id']='\n'
        if fault=='cache':kw['cache']=True
        cid=call(c,funded,'fresh',**kw)
        if fault=='terminal_cache':
            q.execute('SET LOCAL session_replication_role=replica');q.execute("UPDATE public.lab_arena_ledger SET terminal_response=terminal_response||jsonb_build_object('judgment_cache_source_run_id','cached-run')WHERE run_id=%s AND call_identity=%s AND entry_kind='settlement'",(funded,cid));q.execute('SET LOCAL session_replication_role=origin')
        if fault=='unpriced':
            q.execute('SET LOCAL session_replication_role=replica');q.execute("UPDATE public.lab_arena_ledger SET entry_kind='uncertain',terminal_response=NULL,entry_doc=jsonb_build_object('call',jsonb_build_object('call_succeeded',true))WHERE run_id=%s AND call_identity=%s AND entry_kind='settlement'",(funded,cid));q.execute('SET LOCAL session_replication_role=origin')
        if fault=='duplicate_native':call(c,clone(c,run,'duplicate'),'other',native_id='native-fresh')
        if fault=='auth_after_denial':call(c,funded,'invalid',outcome='auth',at=denial+timedelta(minutes=1))
        if fault=='new_credit_after_success':call(c,funded,'new-denial',outcome='credit',at=datetime.now(timezone.utc)-timedelta(minutes=1))
        assert proof(q,run['run_id']) is True


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_security_and_source_guard_precedes_replay(database,installed,drift):
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));post=q.fetchone()[0]
        text=SQL.read_text();old=dict(re.findall(r'\$old_(funded524_\d+)\$(.*?)\$old_\1\$',text,re.S));new=dict(re.findall(r'\$new_(funded524_\d+)\$(.*?)\$new_\1\$',text,re.S));before=post
        if not installed:
            for tag in reversed(old):assert before.count(new[tag])==1;before=before.replace(new[tag],old[tag])
            q.execute(before)
        q.execute('SAVEPOINT hostile')
        if drift=='source':q.execute(before.replace('BEGIN\n','BEGIN\n  -- unknown drift\n',1))
        else:q.execute({'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres','acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon','secdef':'ALTER FUNCTION '+SIG+' SECURITY INVOKER','volatility':'ALTER FUNCTION '+SIG+' VOLATILE','path':'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog'}[drift])
        body=text.split("SET LOCAL statement_timeout='30s';",1)[1].removesuffix('COMMIT;\n')
        with pytest.raises(database[0].Error):q.execute(body)
        q.execute('ROLLBACK TO SAVEPOINT hostile');q.execute(post)


def test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity(database):
    run,_=seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        funded=clone(c,run);cid=call(c,funded,'fresh')
        q.execute('SAVEPOINT duplicate')
        with pytest.raises(database[0].Error) as error:
            q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc)SELECT entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc FROM public.lab_arena_ledger WHERE run_id=%s AND call_identity=%s AND entry_kind='dispatch'",(funded,cid))
        assert error.value.pgcode=='23505';q.execute('ROLLBACK TO SAVEPOINT duplicate')
        assert proof(q,run['run_id']) is False


def test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof(database):
    revision=prior.gate.queue.revision
    store,row,old,new=revision.seed(database)
    try:
        source_id=old[1]['scored_run_id'];old_id=source_id[:-2]+':score:1'
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}'WHERE run_id=%s",(old_id,))
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"')WHERE round_id=%s",(row['round_id'],));q.execute('SET LOCAL session_replication_role=origin')
            call(c,old_id,'old-credit',outcome='credit',at=datetime.now(timezone.utc)-timedelta(minutes=10))
            assert proof(q,old_id) is True
            q.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(row['round_id'],));fresh=q.fetchone()[0]
            assert revision.invoke(q,fresh,new)['status']=='superseded'
            q.execute('SELECT public.lab_arena_expire_leases(%s)',(row['round_id'],))
            q.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE scored_run_id=%s AND scoring_generation_floor>0 ORDER BY attempt DESC LIMIT 1',(source_id,));current=q.fetchone()[0]
            assert current['terminal_cause']=='stage_closed' and proof(q,current['run_id']) is True
            call(c,clone(c,current),'current-funded')
            before=snapshot(q)
            assert proof(q,current['run_id']) is False
            assert proof(q,old_id) is True
            assert snapshot(q)==before
            q.execute('SELECT public.lab_arena_expire_leases(%s)',(row['round_id'],));assert q.fetchone()[0]['retried']>=1
            q.execute("SELECT count(*) FROM public.lab_arena_runs WHERE scored_run_id=%s AND scoring_generation_floor=%s AND status='pending'",(source_id,current['scoring_generation_floor']));assert q.fetchone()[0]==1
    finally:store.close()


def test_new_denial_consumes_old_watermark_and_stops_normal_retry_again(database):
    run,_=seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        call(c,clone(c,run),'funded')
        assert proof(q,run['run_id']) is False
        q.execute('SELECT public.lab_arena_expire_leases(%s)',(run['round_id'],));assert q.fetchone()[0]['retried']>=1
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE assignment_id=%s AND status='pending'",(run['assignment_id'],));q.execute('SET LOCAL session_replication_role=origin')
        q.execute('SET LOCAL ROLE lab_arena_service')
        q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',(run['round_id'],prior.gate.queue.revision.company.original.hotkey('ordered520'),[],'c'*32,'sha256:'+'e'*64,prior.gate.TOKEN));lease=q.fetchone()[0]
        assert lease['status']=='leased' and lease['run_id']==run['assignment_id']+':'+str(run['attempt']+1)
        q.execute('RESET ROLE')
        call(c,lease['run_id'],'denied-again',outcome='credit',at=datetime.now(timezone.utc)-timedelta(seconds=30))
        assert prior.complete(q,lease,cause='credential_error')['status']=='failed'
        assert proof(q,lease['run_id']) is True
        before=snapshot(q)
        for _ in range(2):
            q.execute('SELECT public.lab_arena_expire_leases(%s)',(run['round_id'],));assert q.fetchone()[0]['retried']==0
        assert snapshot(q)==before


def test_every_exhausted_key_requires_its_own_new_paid_dispatch(database):
    run,denial=seed(database)
    second='sha256:'+'d'*64
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        call(c,run['run_id'],'second-denial',outcome='credit',fingerprint=second,at=denial)
        funded=clone(c,run)
        call(c,funded,'first-funded')
        assert proof(q,run['run_id']) is True
        call(c,funded,'second-funded',fingerprint=second)
        assert proof(q,run['run_id']) is False


def test_large_same_key_ledger_keeps_recovery_bounded(database):
    import time
    run,denial=seed(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        funded=clone(c,run)
        # Historical real lifecycle shapes precede the denial and cannot retire it.
        for index in range(500):
            call(c,funded,'historical-'+str(index),at=denial-timedelta(minutes=5))
        assert proof(q,run['run_id']) is True
        call(c,funded,'new-funded')
        started=time.monotonic()
        assert proof(q,run['run_id']) is False
        elapsed=time.monotonic()-started
        print('524 same-key reservations=502 predicate_seconds=%.6f'%elapsed)
        assert elapsed<1.0
