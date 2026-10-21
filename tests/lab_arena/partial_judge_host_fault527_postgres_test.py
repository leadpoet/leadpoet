"""A retained partial cannot hide an unresolved authenticated host fault."""
from pathlib import Path
import hashlib,re,json
from datetime import datetime,timezone,timedelta
import pytest
from lab_arena import contracts
from tests.lab_arena import recovery_lock_order523_postgres_test as recovery
from tests.lab_arena import score_expiry_start_proof511_postgres_test as started

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/527-lab-arena-partial-judge-host-fault.sql'
SIG=started.fast.SIG
PRE='b645001e6ff490dacc0bf5255b1fa9d2b1c2def53a4980056b29b9ce35edd427'
POST='17c0b8d87e6662d01c56d9c3c31d01f833b229e2e5f825d848efa2aca00d4106'
EXPIRY='public.lab_arena_expire_leases(text)'

def definition(q,sig=SIG):
    q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(sig,));return q.fetchone()[0]

def fragments():
    text=SQL.read_text()
    return {label:re.search(r'\$'+label+r'\$(.*?)\$'+label+r'\$',text,re.S)[1]
            for label in ('old_shared','new_shared','old_handoff','new_handoff')}

def old_source(post):
    f=fragments()
    for label in ('shared','handoff'):
        assert post.count(f['new_'+label])==1
        post=post.replace(f['new_'+label],f['old_'+label])
    assert hashlib.sha256(post.encode()).hexdigest()==PRE
    return post

def apply(c):
    with c.cursor()as q:q.execute(SQL.read_text())
    c.commit()

@pytest.fixture(scope='module')
def database():
    g=recovery.database.__wrapped__();db=next(g)
    try:
        with db[0].connect(**db[1])as c:apply(c)
        yield db
    finally:g.close()

def score_seed(c, *, positions=3):
    # The pre-revision helper inserts with triggers disabled and omits the new
    # required run floor. Seed that same normal floor-zero shape explicitly.
    seed=started.fast.seed
    cfg=seed.fixture_configuration(seed.ROUND)
    cfg.update(checkpoint_deadline_policy="atomic_checkpoint_60m_v1",
        icp_wall_clock_seconds=3600,lease_ttl_seconds=4500,
        runner_hotkeys=[seed.RUNNER],parallel_twenty_icp_execution=False,
        sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        execution_icp_cap_microusd=4_000_000,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,max_attempts_per_assignment=2)
    deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    cfg['schedule']['stage_2_close']=deadline
    cfg['schedule']['final_scoring_close']=deadline
    source=f"arena/{seed.ROUND}/sources/{seed.SUB}.tar.gz"
    with c.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        q.execute('TRUNCATE public.lab_arena_rounds RESTART IDENTITY CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]'")
        q.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,benchmark_ref,rewards_enabled,scoring_generation_floor) VALUES (%s,'stage2_scoring',9,7,%s::jsonb,%s::jsonb,%s,false,0)",(seed.ROUND,json.dumps(cfg),json.dumps([dict(submission_id=seed.SUB,miner_hotkey=seed.hotkey(seed.SUB),is_king=False,source_ref=source)]),f"arena/{seed.ROUND}/benchmark.json"))
        q.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES (%s,%s,%s,'frozen',false,%s,123,'{}','passed','{\"decision\":\"pass\"}','sha256:'||repeat('1',64),now(),1)",(seed.SUB,seed.ROUND,seed.hotkey(seed.SUB),source))
        for position in range(10,10+positions):
            assignment=f"{seed.ROUND}:{seed.SUB}:2:{position}:score"
            q.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,created_at) VALUES (%s,%s,%s,%s,%s,2,%s,1,'score','pending',7,0,clock_timestamp()-interval '20 minutes')",(assignment+':1',assignment,seed.ROUND,seed.SUB,seed.hotkey(seed.SUB),position))
        q.execute('SET LOCAL session_replication_role=origin')
    c.commit()
    return cfg


@pytest.fixture(autouse=True)
def restore(database):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        post=definition(q)
        q.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        q.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
    yield
    with database[0].connect(**database[1])as c,c.cursor()as q:q.execute(post)

fast=started.fast
event=started.event

def cohort(conn, *, count=3, extra=1, started=None, explicit_errors=False):
    cfg = score_seed(conn, positions=count+extra)
    # Ordinary retry is authorized by the materialized scoring plan and its
    # accepted source, as in production; a bare pending score fixture is not.
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("SELECT run_id,assignment_id,icp_position FROM public.lab_arena_runs WHERE kind='score' ORDER BY icp_position")
        items = []
        for run_id, assignment, position in cur.fetchall():
            source_assignment = assignment.removesuffix(':score')
            source_id = source_assignment+':1'
            output_ref = 'fixture/'+source_id
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,scoring_generation_floor) SELECT %s,%s,round_id,submission_id,miner_hotkey,stage,icp_position,1,'execute','accepted',stage_generation,'accepted',%s,0 FROM public.lab_arena_runs WHERE run_id=%s", (source_id, source_assignment, output_ref, run_id))
            input_hash = 'sha256:'+hashlib.sha256(source_id.encode()).hexdigest()
            cache_key = 'sha256:'+hashlib.sha256(assignment.encode()).hexdigest()
            scope = {'scorer_image_digest': cfg['scorer_image_digest'],
                     'scorer_image_reference': cfg['scorer_image_reference'],
                     'round_id': fast.seed.ROUND, 'scoring_input_hash': input_hash,
                     'cache_key': cache_key}
            cur.execute('UPDATE public.lab_arena_runs SET scored_run_id=%s,judgment_input_hash=%s,judgment_cache_key=%s,judgment_scope_doc=%s::jsonb WHERE run_id=%s', (source_id, input_hash, cache_key, json.dumps(scope), run_id))
            items.append({'scored_run_id': source_id, 'submission_id': fast.seed.SUB,
                          'icp_position': position, 'output_ref': output_ref})
        cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-20',stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s", (json.dumps({'work_items': items}), fast.seed.ROUND))
    conn.commit()
    leases = []
    for n, suffix in enumerate('abcdef0'[:count]):
        lease = fast.rpc(conn, suffix=suffix)
        assert lease['status'] == 'leased'
        leases.append(lease)
        if started is not None and n < started:
            assert event(conn, lease, suffix)['status'] == 'accepted'
        if explicit_errors:
            assert event(conn, lease, suffix, kind='runtime.error', content={
                'status': 'abandoned', 'failure_stage': 'runtime',
                'error_class': 'RuntimeHostError'})['status'] == 'accepted'
    return leases



def fault(c,lease,suffix,status='judge_error'):
    assert started.event(c,lease,suffix,kind='runtime.error',content={
        'status':status,'failure_stage':'runtime','error_class':'RuntimeHostError',
        'runtime_host_reason':'sandbox_launcher_signaled','launch_exit_code':-2,
        'launch_timed_out':False})['status']=='accepted'


def test_unresolved_partial_fault_status_old_gap_and_new_quarantine(database):
    for version,expected in [('old','leased'),('new','no_pending')]:
        with database[0].connect(**database[1])as c,c.cursor()as q:
            if version=='old':q.execute(old_source(definition(q)))
            else:apply(c)
            leases=cohort(c,count=3,extra=1,started=3)
            for suffix,lease in zip('abc',leases):fault(c,lease,suffix)
            assert started.expire(c)=={'status':'ok','expired':0,'retried':0}
            assert started.fast.rpc(c,suffix='d')['status']==expected


@pytest.mark.parametrize('status,paid,result,expected',[
 ('abandoned',False,False,'no_pending'),('judge_error',False,False,'no_pending'),
 ('model_error',False,False,'leased'),('judge_error',True,False,'leased'),
 ('abandoned',True,False,'leased'),('judge_error',False,True,'leased')])
def test_existing_cost_result_and_status_guards_remain(database,status,paid,result,expected):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        leases=cohort(c,count=3,extra=1,started=3)
        for suffix,lease in zip('abc',leases):fault(c,lease,suffix,status)
        if paid or result:
            q.execute('SET LOCAL session_replication_role=replica')
            target=leases[0]['run_id']
            if paid:q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,%s,'deepline','exa.contents','host',1 FROM public.lab_arena_runs WHERE run_id=%s",('sha256:'+'f'*64,target))
            if result:q.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb WHERE run_id=%s",(target,))
            q.execute('SET LOCAL session_replication_role=origin');c.commit()
        assert started.fast.rpc(c,suffix='d')['status']==expected


@pytest.mark.parametrize('fault_name',[
 'none','run_id','runner_hotkey','round_id','submission_id','miner_hotkey','assignment_id',
 'stage','icp_position','attempt','run_kind','event_kind','status','failure_stage',
 'error_class','created_early','created_late','occurred_early','occurred_late','wrong_start_generation'])
def test_expired_partial_handoff_retains_all_existing_binding_and_time_gates(database,fault_name):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        lease=cohort(c,count=1,extra=0,started=1)[0]
        fault(c,lease,'a')
        assert started.expire(c)=={'status':'ok','expired':0,'retried':0}
        started.expire_clock(c)
        assert started.expire(c)['retried']==1
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        # Move the whole authenticated allocation, not only its actual expiry.
        # Otherwise a recent event must correctly fail the actual-expiry bound.
        q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=lease_expires_at-interval '76 minutes',claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb((claim_response->>'lease_expires_at')::timestamptz-interval '76 minutes')) WHERE run_id=%s",(lease['run_id'],))
        q.execute("UPDATE public.lab_arena_trajectory_events SET created_at=created_at-interval '76 minutes',occurred_at=occurred_at-interval '76 minutes' WHERE run_id=%s",(lease['run_id'],))
        text_fields={'run_id':'unrelated-run','runner_hotkey':started.fast.hotkey('wrong-host-error'),
            'round_id':'wrong-round','submission_id':'wrong-submission','miner_hotkey':started.fast.hotkey('wrong-miner'),
            'assignment_id':'wrong-assignment','run_kind':'execute','event_kind':'runtime.other'}
        if fault_name in text_fields:q.execute('UPDATE public.lab_arena_trajectory_events SET '+fault_name+"=%s WHERE event_kind='runtime.error'",(text_fields[fault_name],))
        elif fault_name in ('stage','icp_position','attempt'):q.execute('UPDATE public.lab_arena_trajectory_events SET '+fault_name+'='+fault_name+"+1 WHERE event_kind='runtime.error'")
        elif fault_name in ('status','failure_stage','error_class'):
            value={'status':'accepted','failure_stage':'setup','error_class':'LeasedImageError'}[fault_name]
            q.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE event_kind='runtime.error'",([fault_name],value))
        elif fault_name=='wrong_start_generation':q.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','999'::jsonb) WHERE event_kind='runtime.started'")
        elif fault_name!='none':
            column='created_at'if fault_name.startswith('created')else'occurred_at';offset=-4501 if fault_name.endswith('early') else 1
            q.execute('UPDATE public.lab_arena_trajectory_events e SET '+column+"=(r.claim_response->>'lease_expires_at')::timestamptz+make_interval(secs=>%s) FROM public.lab_arena_runs r WHERE e.run_id=r.run_id AND e.event_kind='runtime.error'",(offset,))
        q.execute('SET LOCAL session_replication_role=origin');c.commit()
        assert started.fast.rpc(c,suffix='b')['status']==('no_pending'if fault_name=='none'else'leased')


def test_exact_replay_security_and_unchanged_expiry_and_rows(database):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        cohort(c,count=1,extra=1,started=1)
        post=definition(q);before=old_source(post);q.execute(before)
        q.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",(SIG,));metadata=q.fetchone()[0]
        expiry=definition(q,EXPIRY)
        assert hashlib.sha256(expiry.encode()).hexdigest()=='61e1dd5c5c472ce8060d356794bd9f621a8fc0f579ce091941826e19bf261468'
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');rows=q.fetchone()[0]
        apply(c);apply(c)
        assert definition(q)==post and hashlib.sha256(post.encode()).hexdigest()==POST
        assert definition(q,EXPIRY)==expiry
        q.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",(SIG,));assert q.fetchone()[0]==metadata
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert q.fetchone()[0]==rows
        # Exactly two status gates change; the execute-only predicate is intact.
        assert post.count("host_error.content->>'status'='abandoned'")==before.count("host_error.content->>'status'='abandoned'")==3


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['source','owner','acl','path','security','volatility'])
def test_preimage_and_replay_reject_source_or_security_drift(database,installed,drift):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        saved=definition(q)
        if not installed:q.execute(old_source(saved))
        q.execute('SAVEPOINT hostile')
        if drift=='source':q.execute(definition(q).replace('BEGIN\n','BEGIN\n  NULL;\n',1))
        else:q.execute({
          'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres',
          'acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
          'path':'ALTER FUNCTION '+SIG+' SET search_path TO public',
          'security':'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
          'volatility':'ALTER FUNCTION '+SIG+' STABLE',}[drift])
        body=SQL.read_text().split('DO $partial_judge_host_fault$')[1].split('$partial_judge_host_fault$;')[0]
        with pytest.raises(Exception):q.execute('DO $partial_judge_host_fault$'+body+'$partial_judge_host_fault$;')
        q.execute('ROLLBACK TO SAVEPOINT hostile');q.execute(saved)


def test_non_superuser_owner_member_apply_preserves_exact_metadata(database):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        saved=definition(q);q.execute(old_source(saved))
        q.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",(SIG,));metadata=q.fetchone()[0]
        q.execute('CREATE ROLE capture527_migrator NOLOGIN NOSUPERUSER INHERIT')
        q.execute('GRANT lab_arena_owner TO capture527_migrator')
        q.execute('GRANT USAGE,CREATE ON SCHEMA public TO capture527_migrator')
        q.execute('ALTER DEFAULT PRIVILEGES FOR ROLE capture527_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        c.commit();q.execute('SET ROLE capture527_migrator')
        try:
            apply(c);apply(c)
            assert definition(q)==saved
            q.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",(SIG,));assert q.fetchone()[0]==metadata
        finally:q.execute('RESET ROLE')
