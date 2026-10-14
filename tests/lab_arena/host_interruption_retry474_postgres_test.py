"""Authenticated host losses retain normal attempts and cumulative authority."""
from pathlib import Path
import pytest
from tests.lab_arena import general_judge_continuation472_postgres_test as previous

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/474-lab-arena-host-interruption-execution-retries.sql'

@pytest.fixture(scope='module')
def database():
    gen=previous.database.__wrapped__(); db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute((ROOT/'scripts/473-lab-arena-recovered-execution-judge-continuation.sql').read_text())
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield db
    finally: gen.close()

def test_exact_upgrade_is_idempotent(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT has_function_privilege('lab_arena_service','public.lab_arena_execution_attempt_count(public.lab_arena_runs)','EXECUTE'),has_function_privilege('anon','public.lab_arena_execution_attempt_count(public.lab_arena_runs)','EXECUTE')")
        assert cur.fetchone()==(True,False)

import json
import uuid
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from lab_arena import contracts,scoring
from lab_arena.store import ArenaStore,PsycopgTransport,PostgrestTransport
from tests.lab_arena import stage_cutoff_drain420_postgres_test as cutoff

ROUND='arena-2099-10-14-host474'
RUNNER=cutoff.RUNNER

def rpc(conn,expression,args=()):
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.'+expression,args); value=cur.fetchone()[0]
        cur.execute('RESET ROLE')
    conn.commit(); return value

def seed(conn,*,day='2099-10-14',quota=10):
    with conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
    conn.commit()
    with patch.object(cutoff,'ROUND',ROUND): cfg=cutoff._seed(conn,positions=1)
    cfg['call_quotas']={p:quota for p in ('openrouter','scrapingdog','deepline')}
    cfg['cost_per_company_microusd']=800000
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET evaluation_date=%s,configuration_doc=%s::jsonb WHERE round_id=%s',(day,json.dumps(cfg),ROUND))
        cur.execute("UPDATE public.lab_arena_runs SET run_id=replace(run_id,':2:10:1',':2:8:1'),assignment_id=replace(assignment_id,':2:10',':2:8'),icp_position=8 WHERE round_id=%s",(ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit(); return cfg

def claim(conn,suffix='a'):
    return rpc(conn,'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::TEXT[],%s,%s,%s,4500)',(ROUND,RUNNER,suffix*32,'sha256:'+suffix*64,'sha256:'+suffix*64))

def event(conn,lease,kind,content,*,token='a'):
    value=[dict(event_id=str(uuid.uuid4()),kind=kind,content=content,occurred_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat())]
    return rpc(conn,'lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',(lease['run_id'],'sha256:'+token*64,json.dumps(value)))

def reserve(conn,lease,*,token='a',call='e'):
    return rpc(conn,'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,0,%s::jsonb,4500)',(lease['run_id'],'sha256:'+token*64,'sha256:'+call*64,'deepline.execute','deepline','miner_key',json.dumps({'reserve_remaining_budget':True,'tool':'exa_search'})))

def pay(conn,lease,*,token='a',call='e',amount=100,unsettled=False):
    assert reserve(conn,lease,token=token,call=call)['status']=='reserved'
    assert rpc(conn,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],'sha256:'+token*64,'sha256:'+call*64))['status']=='dispatched'
    if unsettled:
        return rpc(conn,'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',(lease['run_id'],'sha256:'+token*64,'sha256:'+call*64,json.dumps({'reason':'missing_provider_cost','call_succeeded':True,'provider_status':200})))
    return rpc(conn,'lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,4500)',(lease['run_id'],'sha256:'+token*64,'sha256:'+call*64,amount,json.dumps({'status':200,'call_succeeded':True})))

def finish(conn,lease,cause='provider_error',*,token='b'):
    return rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],'sha256:'+token*64,json.dumps({'terminal_status':cause}),cause,'retained-output' if cause=='accepted' else None))

def expire(conn):return rpc(conn,'lab_arena_expire_leases(%s)',(ROUND,))

def host_loss(conn,*,proof=True,unsettled=False,amount=100):
    lease=claim(conn); assert lease['status']=='leased',lease
    if proof:
        assert event(conn,lease,'runtime.started',{'status':'starting','lease_generation':lease['lease_generation']})['status']=='accepted'
    assert pay(conn,lease,amount=amount,unsettled=unsettled)['status'] in ('settled','uncertain')
    assert event(conn,lease,'runtime.error',{'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError','runtime_host_reason':'sandbox_launcher_signaled','launch_exit_code':-2,'launch_timed_out':False})['status']=='accepted'
    if unsettled or not proof:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    assert expire(conn)['retried']==1
    return lease

def projected(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r)||jsonb_build_object(\'execution_attempt\',public.lab_arena_execution_attempt_count(r)) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY attempt',(ROUND,))
        return [r[0] for r in cur.fetchall()]

def plan(conn,cfg):
    return scoring.build_scoring_plan(round_id=ROUND,stage=2,runs=projected(conn),execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,configuration=cfg)

def close(conn):return rpc(conn,'lab_arena_close_stage(%s,2::SMALLINT)',(ROUND,))

def test_host_then_provider_then_accepted_uses_original_queue_and_preserves_history(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); first=host_loss(conn)
        second=claim(conn,'b'); assert second['attempt']==2,second
        assert pay(conn,second,token='b',call='f')['status']=='settled'
        before=projected(conn)[0]
        result=finish(conn,second); assert result['confirmation_attempt']==3,result
        assert projected(conn)[0]==before
        assert expire(conn)['retried']==0
        third=claim(conn,'c'); assert third['attempt']==3,third
        assert finish(conn,third,'accepted',token='c')['status']=='accepted'
        assert expire(conn)['retried']==0
        assert close(conn)['status']=='closed'
        built=plan(conn,cfg)
        assert len(built['work_items'])==1 and built['zero_rows']==[]
        assert [r['execution_attempt'] for r in projected(conn)]==[0,1,2]

def test_host_then_two_provider_failures_exhaust_two_normal_attempts(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn)
        second=claim(conn,'b'); assert finish(conn,second)['confirmation_attempt']==3
        third=claim(conn,'c'); assert 'confirmation_attempt' not in finish(conn,third,token='c')
        assert expire(conn)['retried']==0
        assert close(conn)['status']=='closed'
        assert plan(conn,cfg)['zero_rows']==[{'submission_id':cutoff.SUB,'icp_position':8,'cause':'provider_error'}]

def test_two_genuine_provider_failures_still_exhaust(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); first=claim(conn)
        assert finish(conn,first,token='a')['confirmation_attempt']==2
        second=claim(conn,'b'); assert 'confirmation_attempt' not in finish(conn,second)
        assert expire(conn)['retried']==0
        assert close(conn)['status']=='closed'
        assert len(plan(conn,cfg)['zero_rows'])==1

@pytest.mark.parametrize('cause',['model_error','invalid_output','credential_error','budget_exhausted'])
def test_genuine_terminal_failure_is_not_replayed_because_of_host_credit(database,cause):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn); second=claim(conn,'b')
        assert 'confirmation_attempt' not in finish(conn,second,cause)
        assert expire(conn)['retried']==0
        assert len(projected(conn))==2
        assert close(conn)['status']=='closed'
        assert plan(conn,cfg)['zero_rows']==[{'submission_id':cutoff.SUB,'icp_position':8,'cause':cause}]

def test_missing_authenticated_start_does_not_create_host_credit(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn,proof=False); second=claim(conn,'b')
        assert 'confirmation_attempt' not in finish(conn,second)
        assert [r['execution_attempt'] for r in projected(conn)]==[1,2]
        assert expire(conn)['retried']==0

def test_unsettled_host_fee_does_not_turn_missing_evaluation_into_zero(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn,unsettled=True); second=claim(conn,'b')
        assert 'confirmation_attempt' not in finish(conn,second)
        assert expire(conn)['retried']==0
        assert close(conn)['status']=='closed'
        built=plan(conn,cfg)
        assert built['zero_rows']==[] and built['work_items']==[]
        assert built['incomplete_rows']==[{'submission_id':cutoff.SUB,'icp_position':8,'cause':'provider_error'}]
        assert [r['execution_attempt'] for r in projected(conn)]==[0,1]

def test_extra_attempt_does_not_reset_cumulative_provider_quota(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn,quota=1); host_loss(conn); second=claim(conn,'b')
        assert pay(conn,second,token='b',call='f')['status']=='settled'
        assert finish(conn,second)['confirmation_attempt']==3
        third=claim(conn,'c'); denied=reserve(conn,third,token='c',call='d')
        assert denied['status']=='refused' and denied['reason']=='per_icp_quota',denied
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='dispatch'",(third['run_id'],))
            assert cur.fetchone()[0]==0

def test_exhausted_cumulative_cost_blocks_host_credit_dispatch_and_keeps_incomplete(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn,amount=3999900); second=claim(conn,'b')
        assert pay(conn,second,token='b',call='f',amount=100)['status']=='settled'
        assert 'confirmation_attempt' not in finish(conn,second)
        assert expire(conn)['retried']==0
        assert close(conn)['status']=='closed'
        assert plan(conn,cfg)['zero_rows']==[]

@pytest.mark.parametrize('status',['stage2_closed','published'])
def test_closed_round_does_not_materialize_successor(database,status):
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn); second=claim(conn,'b')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s',(status,ROUND))
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='provider_error',result_doc='{\"terminal_status\":\"provider_error\"}' WHERE run_id=%s",(second['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit(); assert expire(conn)['retried']==0
        assert len(projected(conn))==2

def test_old_day_keeps_frozen_retry_policy(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn,day='2026-10-13'); host_loss(conn); second=claim(conn,'b')
        assert 'confirmation_attempt' not in finish(conn,second)
        assert [r['execution_attempt'] for r in projected(conn)]==[1,2]


def test_current_stopped_lease_with_unresolved_billing_closes_incomplete(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); first=claim(conn)
        assert finish(conn,first,token='a')['confirmation_attempt']==2
        second=claim(conn,'b')
        event(conn,second,'runtime.started',{'lease_generation':second['lease_generation']},token='b')
        pay(conn,second,token='b',unsettled=True)
        event(conn,second,'runtime.error',{'status':'abandoned','failure_stage':'runtime',
            'error_class':'RuntimeHostError','runtime_host_reason':'sandbox_launcher_signaled',
            'launch_exit_code':-2,'launch_timed_out':False},token='b')
        # Billing holds this stopped lease until the absolute frozen drain bound.
        assert expire(conn)['expired']==0
        assert projected(conn)[-1]['execution_attempt']==1
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb(%s::text)) WHERE round_id=%s",
                ((datetime.now(timezone.utc)-timedelta(seconds=4501)).isoformat(),ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert close(conn)['status']=='closed'
        row=projected(conn)[-1]
        assert row['terminal_cause']=='stage_closed'
        assert row['terminal_doc'].get('deadline_provider_retry_exhausted') is not True
        built=plan(conn,cfg)
        assert built['zero_rows']==[] and built['work_items']==[]
        assert len(built['incomplete_rows'])==1


@pytest.mark.parametrize('mutation', ['wrong_runner','wrong_generation','finished','cleanup','positive_exit','string_exit','timed_out'])
def test_untrusted_or_contradictory_host_evidence_does_not_discount(database,mutation):
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if mutation=='wrong_runner':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s WHERE event_kind='runtime.error'",('5'+'A'*47,))
            elif mutation=='wrong_generation':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','999') WHERE event_kind='runtime.started'")
            elif mutation in ('finished','cleanup'):
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE event_kind='runtime.started'",('runtime.finished' if mutation=='finished' else 'runtime.cleanup_error',))
            else:
                field='launch_timed_out' if mutation=='timed_out' else 'launch_exit_code'
                value={'positive_exit':'2','string_exit':'"-2"','timed_out':'true'}[mutation]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,ARRAY[%s],%s::jsonb) WHERE event_kind='runtime.error'",(field,value))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        second=claim(conn,'b')
        assert 'confirmation_attempt' not in finish(conn,second)
        assert expire(conn)['retried']==0
        assert [r['execution_attempt'] for r in projected(conn)]==[1,2]


def test_existing_failed_assignment_backfill_is_idempotent_and_keeps_terminal_rows(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn); second=claim(conn,'b')
        assert finish(conn,second)['confirmation_attempt']==3
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=3',(second['assignment_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before=projected(conn)
        assert expire(conn)['retried']==1
        assert expire(conn)['retried']==0
        assert projected(conn)[:2]==before


def test_projection_preserves_default_select_and_service_role_read(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn)
    transport=PsycopgTransport(lambda:database[0].connect(**database[1]),role='lab_arena_service')
    try:
        store=ArenaStore(transport)
        default=store.list_runs(ROUND)
        projected_rows=store.list_runs(ROUND,columns='*,execution_attempt:lab_arena_execution_attempt_count')
        assert all('execution_attempt' not in r for r in default)
        assert [r['execution_attempt'] for r in projected_rows]==[0,1]
        assert [{k:v for k,v in r.items() if k!='execution_attempt'} for r in projected_rows]==default
    finally:
        transport.close()
    import httpx
    requests=[]
    def respond(request):
        requests.append(dict(request.url.params))
        return httpx.Response(200,json=[])
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        store=ArenaStore(PostgrestTransport('http://127.0.0.1:3000',service_key='sb_secret_test',http_client=client))
        store.list_runs(ROUND)
        store.list_runs(ROUND,columns='*,execution_attempt:lab_arena_execution_attempt_count')
    assert requests[0]['select']=='*'
    assert requests[1]['select']=='*,execution_attempt:lab_arena_execution_attempt_count'


def test_production_sized_projection_and_transition_remain_bounded(database,record_property):
    import time
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn); second=claim(conn,'b')
        assert finish(conn,second)['confirmation_attempt']==3
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("""INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
                stage,icp_position,attempt,kind,status,stage_generation)
                SELECT 'timing474:'||n,'timing474:'||n,round_id,submission_id,miner_hotkey,stage,8,1,kind,'pending',stage_generation
                FROM public.lab_arena_runs CROSS JOIN generate_series(20,2019) n WHERE run_id=%s""",(second['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        started=time.monotonic(); rows=projected(conn); projected_seconds=time.monotonic()-started
        started=time.monotonic(); assert expire(conn)['retried']==0; transition_seconds=time.monotonic()-started
        assert len(rows)==2003
        assert projected_seconds<5 and transition_seconds<5
        record_property('projection_rows',len(rows))
        record_property('projection_seconds',round(projected_seconds,4))
        record_property('transition_seconds',round(transition_seconds,4))


def test_budget_only_deepline_keeps_original_zero_quota_semantics(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("""UPDATE public.lab_arena_rounds SET configuration_doc=
              jsonb_set(jsonb_set(configuration_doc,'{call_quotas,deepline}','0'),
              '{deepline_catalog}','{"schema_version":"leadpoet.lab_arena.deepline_catalog.v1"}')
              WHERE round_id=%s""",(ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        host_loss(conn); second=claim(conn,'b')
        assert finish(conn,second)['confirmation_attempt']==3
        third=claim(conn,'c')
        assert pay(conn,third,token='c',call='f')['status']=='settled'


def test_extra_attempt_enforces_exact_original_cost_boundary_before_dispatch(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn); host_loss(conn,amount=3999998); second=claim(conn,'b')
        assert pay(conn,second,token='b',call='f',amount=1)['status']=='settled'
        assert finish(conn,second)['confirmation_attempt']==3
        third=claim(conn,'c')
        assert pay(conn,third,token='c',call='d',amount=1)['status']=='settled'
        denied=reserve(conn,third,token='c',call='c')
        assert denied['status']=='refused' and denied['reason']=='money_cap',denied
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE entry_kind='settlement'")
            assert cur.fetchone()==(3,4000000)
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'",('sha256:'+'c'*64,))
            assert cur.fetchone()[0]==0


def test_host_credit_cannot_extend_frozen_execution_deadline(database):
    with database[0].connect(**database[1]) as conn:
        cfg=seed(conn); host_loss(conn); second=claim(conn,'b')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb(%s::text)) WHERE round_id=%s",
                ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert 'confirmation_attempt' not in finish(conn,second)
        assert expire(conn)['retried']==0
        assert len(projected(conn))==2
        assert close(conn)['status']=='closed'
        assert plan(conn,cfg)['zero_rows']==[]


@pytest.mark.parametrize('mutation',['source','security'])
def test_migration_refuses_unreviewed_rpc_changes_transactionally(database,mutation):
    signature='public.lab_arena_close_stage(text,smallint)'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
        before=cur.fetchone()[0]
        if mutation=='source':
            cur.execute(before.replace('BEGIN','BEGIN\n  -- unrelated concurrent edit',1))
        else:
            cur.execute('ALTER FUNCTION '+signature+' SET search_path=public')
        with pytest.raises(database[0].Error,match='Host retry474 (preimage|security shape) differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
        assert cur.fetchone()[0]==before


def test_current_schema_model_zero_preserves_baseline_scores_and_publication(database,tmp_path):
    import copy
    from dataclasses import replace
    from lab_arena import runtime
    connect=lambda:database[0].connect(**database[1])
    with connect() as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
    h=cutoff.QualityHarness(connect,tmp_path,challengers=['PublicBaselineAlternate'],runners=['alpha','beta'])
    h.service.config.defaults=replace(h.service.config.defaults,benchmark_icp_count=10,
        checkpoint_deadline_enabled=True,runner_slot_ceiling=10,company_quality_from=None,
        intent_details_from='2026-01-01T00:00:00Z',contacts_from=None,
        execution_sequence_from='2026-01-01T00:00:00Z',per_icp_cost_policy=True,rewards_enabled=True)
    h.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':int(kwargs['set_id']),'icps':copy.deepcopy(cutoff.daily_icps()[:10])}
    observed=cutoff._install_company_only_sandbox(h)
    h.round_id='arena-2099-10-14-full474'
    h.service.create_round(datetime.now(timezone.utc)+timedelta(minutes=30),round_id=h.round_id)
    challenger=h.submit('PublicBaselineAlternate',h.round_id)
    h.clock.advance_to(h.schedule()['submission_cutoff'])
    assert h.service.advance_round(h.round_id)['status']=='ok'
    participants=h.service.store.get_round(h.round_id)['participants']
    for participant in participants: h.flavors.setdefault(participant['submission_id'],'PublicBaseline')
    original=h.sandbox.run_icp
    failures=[]
    def with_model_failure(spec,**kwargs):
        doc=json.loads((spec.input_dir/runtime.INPUT_FILE_NAME).read_text())
        if (doc.get('schema_version')!=scoring.SCORING_INPUT_SCHEMA_VERSION
            and spec.source_dir.parent.name=='submission-'+challenger
            and int(str(doc['icp']['icp_id']).rsplit('_',1)[-1])==1):
            failures.append(1)
            return runtime.fake_result(exit_code=1,output_bytes=None,stderr=b'provider error')
        return original(spec,**kwargs)
    h.sandbox.run_icp=with_model_failure
    original_runner=h.runner
    def verified_runner(index,parallel=10):
        runner=original_runner(index,parallel)
        runner._config.proxy_worker_pool=cutoff._verified_test_pool(parallel)
        return runner
    h.runner=verified_runner
    h.clock.advance_to(h.schedule()['stage_1_start'])
    assert h.service.advance_round(h.round_id)['status']=='ok'
    h.advance_until('published',runners=2,max_steps=60)
    final=h.service.store.get_round(h.round_id)
    assert failures==[1,1]
    assert final['stage2_scoring_plan_doc']['zero_rows']==[
        {'submission_id':challenger,'icp_position':0,'cause':'model_error'}]
    runs=h.service.store.list_runs(h.round_id)
    executions=[r for r in runs if r['kind']=='execute']
    judges=[r for r in runs if r['kind']=='score']
    assert len(executions)==21 and len(judges)==19
    assert sum(r['status']=='accepted' for r in executions)==19
    assert all(r['status']=='accepted' for r in judges)
    failed=[r for r in executions if r['status']=='failed']
    assert len(failed)==2 and all(r['terminal_cause']=='model_error' for r in failed)
    assert sum(r['per_icp_score']==0 for r in failed)==1
    assert final['publication_doc']['final_ranking']
    assert h.service.activate_reward(h.round_id)['status']=='activated'
    proof={'execute_rows':21,'accepted_execute_rows':19,'accepted_judge_rows':19,
        'model_zero_rows':1,'publication':final['status'],'reward_activated':True,
        'sandbox_completed':observed,'limitation':'Disposable PostgreSQL; deterministic sandbox, provider transport, and judge.'}
    (tmp_path/'host-retry474-model-zero-publication-proof.json').write_text(json.dumps(proof,indent=2)+'\n')


def test_current_schema_accepted_scores_publish_and_activate_reward(database,tmp_path):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
    cutoff.test_real_signed_late_execution_preserves_costs_and_publishes(database,tmp_path)
