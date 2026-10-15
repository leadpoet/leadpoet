"""Original-assignment recovery uses verbatim deployed SQL and ordinary RPCs."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from tests.lab_arena import zero_call_setup_recovery467_postgres_test as deployed
from tests.lab_arena import stage_cutoff_drain420_postgres_test as cutoff
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/470-lab-arena-bounded-baseline-recovery.sql'
ROUND='arena-2099-10-13-recovery470'
SUB='baseline-2099-10-13-recovery470'
MINER=hotkey(SUB)
RUNNER=cutoff.RUNNER
TOKEN='sha256:'+'a'*64

@pytest.fixture(scope='module')
def database():
    gen=deployed.database.__wrapped__(); db=next(gen)
    try:
        deployed.migrated.__wrapped__(db)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for number in (455,458,459,468,469):
                paths=list((ROOT/'scripts').glob(f'{number}-*.sql')); assert len(paths)==1
                cur.execute(paths[0].read_text())
            cur.execute(SQL.read_text()); cur.execute(SQL.read_text())
        yield db
    finally: gen.close()

def rpc(conn, fn, args=()):
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.'+fn,args); result=cur.fetchone()[0]
        cur.execute('RESET ROLE')
    conn.commit(); return result

def recover(conn,items=()):
    return rpc(conn,'lab_arena_recover_baseline_v1(%s,%s::jsonb)',(ROUND,json.dumps(list(items))))

def seed(conn,*,kind='execute',cause=None):
    cfg=cutoff.fixture_configuration(ROUND)
    now=datetime.now(timezone.utc)
    cfg.update(baseline_hotkey=MINER,runner_hotkeys=[RUNNER],parallel_twenty_icp_execution=False,
      execution_sequence_policy='baseline_scored_first_v1',
      sourcing_cost_eligibility_policy='successful_calls_per_icp_v1',
      checkpoint_deadline_policy='atomic_checkpoint_60m_v1',icp_wall_clock_seconds=3600,
      lease_ttl_seconds=4500,execution_icp_cap_microusd=4000000,max_attempts_per_assignment=2)
    cfg['schedule']['stage_1_close']=cutoff._stamp(now-timedelta(hours=3))
    cfg['schedule']['stage_1_scoring_close']=cutoff._stamp(now-timedelta(hours=2))
    cfg['schedule']['stage_2_close']=cutoff._stamp(now+timedelta(hours=1))
    cfg['schedule']['final_scoring_close']=cutoff._stamp(now+timedelta(hours=5))
    assignment=f'{ROUND}:{SUB}:1:0'+(':score' if kind=='score' else '')
    source=f'{ROUND}:{SUB}:1:0:1'
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]'")
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,stage_generation,configuration_doc,participants,benchmark_ref,evaluation_date,rewards_enabled,stage1_scoring_plan_doc) VALUES (%s,'stage2',5,%s::jsonb,%s::jsonb,'benchmark','2099-10-13',false,%s::jsonb)",
          (ROUND,json.dumps(cfg),json.dumps([dict(submission_id=SUB,miner_hotkey=MINER,is_king=True)]),json.dumps({'unchanged':'frozen-plan'})))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES (%s,%s,%s,'frozen',true,'arena/'||%s||'/sources/'||%s||'.tar.gz',100,'passed','{}','sha256:'||repeat('a',64),now(),1)",(SUB,ROUND,MINER,ROUND,SUB))
        if kind=='score':
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref) VALUES (%s,%s,%s,%s,%s,1,0,1,'execute','accepted',1,'accepted','source-output')",(source,source[:-2],ROUND,SUB,MINER))
        for attempt in (1,2):
            terminal=cause or ('provider_error' if kind=='execute' else 'judge_error')
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc,scored_run_id,judgment_group_leader,judgment_group_miner_hotkeys) VALUES (%s,%s,%s,%s,%s,1,0,%s,%s,'failed',3,%s,%s::jsonb,%s,true,ARRAY[%s])",(assignment+f':{attempt}',assignment,ROUND,SUB,MINER,attempt,kind,terminal,json.dumps({'terminal_status':terminal}),source if kind=='score' else None,MINER))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit(); return assignment,cfg

def claim(conn,*,suffix='a'):
    return rpc(conn,'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',(ROUND,RUNNER,suffix*32,'sha256:'+suffix*64,TOKEN))

def test_original_execute_lease_provider_completion_and_restart_idempotency(database):
    with database[0].connect(**database[1]) as conn:
        assignment,cfg=seed(conn)
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r ORDER BY run_id'); old=cur.fetchall()
        assert recover(conn)=={'status':'ok','queued':1}
        assert recover(conn)=={'status':'ok','queued':0}
        lease=claim(conn); assert (lease['status'],lease['stage'],lease['attempt'])==('leased',1,3),lease
        identity='sha256:'+'b'*64
        assert rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],TOKEN,identity))['status']=='reserved'
        assert rpc(conn,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],TOKEN,identity))['status']=='dispatched'
        assert rpc(conn,'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',(lease['run_id'],TOKEN,identity,json.dumps({'call_succeeded':True,'status':200})))['status']=='settled'
        assert rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','new-output'))['status']=='accepted'
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE attempt<3 ORDER BY run_id'); assert cur.fetchall()==old
            cur.execute('SELECT configuration_doc,stage1_scoring_plan_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,)); assert cur.fetchone()==(cfg,{'unchanged':'frozen-plan'})
        assert recover(conn)=={'status':'ok','queued':0}

@pytest.mark.parametrize('fault',['model_error','credential_error','invalid_output','accepted','live','retired','alias','foreign','stale_source','deadline','published','paused','unknown_success','inflight'])
def test_no_grant_for_other_failures_live_work_or_unknown_success(database,fault):
    with database[0].connect(**database[1]) as conn:
        assignment,cfg=seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault in ('model_error','credential_error','invalid_output'):
                cur.execute('UPDATE public.lab_arena_runs SET terminal_cause=%s WHERE attempt=2',(fault,))
            elif fault in ('accepted','live'):cur.execute('UPDATE public.lab_arena_runs SET status=%s WHERE attempt=1',('accepted' if fault=='accepted' else 'leased',))
            elif fault=='retired':cur.execute('UPDATE public.lab_arena_runs SET champion_restart_required=true WHERE attempt=1')
            elif fault=='alias':cur.execute("UPDATE public.lab_arena_runs SET assignment_id=assignment_id||':alias'")
            elif fault=='foreign':cur.execute("UPDATE public.lab_arena_submissions SET is_king=false")
            elif fault=='stale_source':cur.execute("UPDATE public.lab_arena_submissions SET status='accepted'")
            elif fault=='deadline':cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
            elif fault=='published':cur.execute("UPDATE public.lab_arena_rounds SET status='published'")
            elif fault=='paused':cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
            elif fault in ('unknown_success','inflight'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES (%s,%s,%s,%s,%s,1,'sha256:'||repeat('c',64),'deepline','exa_search','host',100,%s::jsonb)",('uncertain' if fault=='unknown_success' else 'dispatch',MINER,ROUND,SUB,assignment+':2',json.dumps({'call':{'call_succeeded':True}})))
        conn.commit()
        assert recover(conn)['queued']==0

@pytest.mark.parametrize('kind',['execute','score'])
def test_historical470_extra_attempt_failure_does_not_grant_four(database,kind):
    # Historical470 fixture stops here;485 proves the authorized general chain.
    with database[0].connect(**database[1]) as conn:
        assignment,cfg=seed(conn,kind=kind)
        assert recover(conn)['queued']==1
        if kind=='score':
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring'")
            conn.commit()
        lease=claim(conn); assert lease['status']=='leased',lease
        cause='provider_error' if kind=='execute' else 'judge_error'
        done=rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],TOKEN,json.dumps({'terminal_status':cause}),cause,''))
        assert done['status']=='failed' and 'confirmation_attempt' not in done
        assert recover(conn)['queued']==0
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt),count(*) FROM public.lab_arena_runs WHERE assignment_id=%s',(assignment,)); assert cur.fetchone()==(3,3)

def scope_item(conn,source):
    with conn.cursor() as cur:
        cur.execute('SELECT configuration_doc,evaluation_date,arena_network_name,arena_netuid FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,)); cfg,date,network,netuid=cur.fetchone()
    key='sha256:'+'d'*64; ih='sha256:'+'e'*64
    return dict(scored_run_id=source,submission_id=SUB,icp_position=0,output_ref='new-output',
      judgment_cache_key=key,judgment_input_hash=ih,judgment_scope_doc=dict(cache_key=key,scoring_input_hash=ih,
        round_id=ROUND,network_name=network,netuid=netuid,evaluation_date=date,integrity_policy=cfg['integrity_policy'],
        scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference']))

def accepted_recovery(conn):
    assignment,cfg=seed(conn); assert recover(conn)['queued']==1
    lease=claim(conn); assert lease['status']=='leased'
    assert rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','new-output'))['status']=='accepted'
    return assignment,cfg,lease

@pytest.mark.parametrize('field',['judgment_cache_key','judgment_input_hash'])
@pytest.mark.parametrize('missing',[True,False])
def test_missing_cache_identity_is_not_admitted(database,field,missing):
    with database[0].connect(**database[1]) as conn:
        _,_,lease=accepted_recovery(conn); item=scope_item(conn,lease['run_id'])
        if missing:item.pop(field)
        else:item[field]=None
        with pytest.raises(database[0].Error,match='recovery_scoring_item_invalid'):recover(conn,[item])

@pytest.mark.parametrize('fault',['foreign','wrong_output','wrong_position','wrong_network','wrong_image','wrong_evaluation','duplicate','oversized','quality'])
def test_scoring_input_frozen_binding_and_bounds(database,fault):
    with database[0].connect(**database[1]) as conn:
        _,_,lease=accepted_recovery(conn); item=scope_item(conn,lease['run_id']); items=[item]
        if fault=='foreign':item['scored_run_id']='foreign'
        elif fault=='wrong_output':item['output_ref']='foreign-output'
        elif fault=='wrong_position':item['icp_position']=1
        elif fault=='wrong_network':item['judgment_scope_doc']['network_name']='other'
        elif fault=='wrong_image':item['judgment_scope_doc']['scorer_image_digest']='sha256:'+'f'*64
        elif fault=='wrong_evaluation':item['judgment_scope_doc']['evaluation_date']='2099-10-14'
        elif fault=='duplicate':items=[item,item]
        elif fault=='oversized':items=[item]*101
        elif fault=='quality':
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{company_quality_policy}','\"company_quality_v1\"')")
            conn.commit()
        with pytest.raises(database[0].Error,match='recovery_(scoring_)?input_invalid|recovery_scoring_item_invalid'):recover(conn,items)


def test_original_source_judging_retries_and_pending_generation_refresh(database):
    with database[0].connect(**database[1]) as conn:
        assignment,cfg,source=accepted_recovery(conn)
        item=scope_item(conn,source['run_id']); assert recover(conn,[item])['queued']==1
        assert recover(conn,[item])['queued']==0
        assert claim(conn,suffix='b')['status']=='no_pending'  # Score held through miner execution.
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=7")
        conn.commit(); assert recover(conn)['queued']==0
        judge=claim(conn,suffix='c'); assert (judge['status'],judge['stage'],judge['attempt'],judge['scored_run_id'])==('leased',1,1,source['run_id'])
        result=rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(judge['run_id'],TOKEN,json.dumps({'terminal_status':'judge_error'}),'judge_error',''))
        assert result['confirmation_attempt']==2
        second=claim(conn,suffix='d'); assert second['attempt']==2
        result=rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(second['run_id'],TOKEN,json.dumps({'terminal_status':'judge_error'}),'judge_error',''))
        assert 'confirmation_attempt' not in result
        assert recover(conn)['queued']==0
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt) FROM public.lab_arena_runs WHERE kind=\'score\''); assert cur.fetchone()[0]==2


def test_failed_fees_preserved_and_successful_costs_remain_cumulative(database):
    with database[0].connect(**database[1]) as conn:
        assignment,_=seed(conn)
        with conn.cursor() as cur:
            for suffix,kind,doc,terminal,amount in [('f','uncertain',{'call':{'call_succeeded':False}},None,900),('b','settlement',{}, {'call_succeeded':True},4000000)]:
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) VALUES (%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',%s,%s::jsonb,%s::jsonb)",(kind,MINER,ROUND,SUB,assignment+':2','sha256:'+suffix*64,amount,json.dumps(doc),json.dumps(terminal)))
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id'); before=cur.fetchall()
        conn.commit(); assert recover(conn)['queued']==1
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id'); assert cur.fetchall()==before
            cur.execute('SELECT public.lab_arena__successful_icp_cost_state(%s,%s,0)',(ROUND,SUB)); state=cur.fetchone()[0]
            assert state['successful_microusd']==4000000 and state['uncertain_calls']==1 and state['success_unresolved_calls']==0
        lease=claim(conn); assert lease['status']=='leased'
        reservation=rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],TOKEN,'sha256:'+'c'*64))
        assert reservation['status']!='reserved',reservation


def test_natural_expiry_consumes_extra_attempt_and_never_regrants(database):
    with database[0].connect(**database[1]) as conn:
        assignment,_=seed(conn); assert recover(conn)['queued']==1
        lease=claim(conn); assert lease['status']=='leased'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE attempt=3")
        conn.commit()
        result=rpc(conn,'lab_arena_expire_leases(%s)',(ROUND,)); assert (result['expired'],result['retried'])==(1,0)
        assert recover(conn)['queued']==0
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s',(assignment,)); assert cur.fetchone()[0]==3


def test_stage_close_preserves_source_and_refreshes_only_pending_original_judges(database):
    with database[0].connect(**database[1]) as conn:
        assignment,cfg=seed(conn,kind='score'); assert recover(conn)['queued']==1
        assert rpc(conn,'lab_arena_close_stage(%s,2::smallint)',(ROUND,))['status']=='closed'
        assert recover(conn)['queued']==0
        assert rpc(conn,"lab_arena_open_scoring_v2(%s,2::smallint,'[]'::jsonb)",(ROUND,))['status']=='ok'
        assert recover(conn)['queued']==0
        lease=claim(conn); assert (lease['status'],lease['stage'],lease['attempt'])==('leased',1,3)
        assert lease['scored_run_id']==f'{ROUND}:{SUB}:1:0:1'
        assert rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(ROUND,))['status']=='closed'
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],)); assert cur.fetchone()==('failed','stage_closed')
            cur.execute("SELECT status,output_ref FROM public.lab_arena_runs WHERE kind='execute'"); assert cur.fetchone()==('accepted','source-output')


def test_pending_recovery_execution_is_closed_by_existing_deadline_path(database):
    with database[0].connect(**database[1]) as conn:
        assignment,_=seed(conn); assert recover(conn)['queued']==1
        assert rpc(conn,'lab_arena_close_stage(%s,2::smallint)',(ROUND,))['status']=='closed'
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s',(assignment+':3',)); assert cur.fetchone()==('failed','stage_closed')


def test_partial_late_score_preserves_write_once_and_cannot_create_final_authority(database):
    with database[0].connect(**database[1]) as conn:
        assignment,_=seed(conn,kind='score'); assert recover(conn)['queued']==1
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring'")
            cur.execute("UPDATE public.lab_arena_runs SET result_doc='{\"terminal_status\":\"accepted\"}' WHERE kind='execute'")
        conn.commit(); lease=claim(conn); assert lease['status']=='leased'
        assert rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','judge-output'))['status']=='accepted'
        assert rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(ROUND,))['status']=='closed'
        scores=[{'run_id':lease['scored_run_id'],'per_icp_score':0,'qualification_doc':{'companies':[]}}]
        first=rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',(ROUND,json.dumps(scores))); assert first['recorded']==1
        again=rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',(ROUND,json.dumps(scores))); assert again['existing']==1
        with conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)',(ROUND,)); assert not cur.fetchone()[0]
        scores[0]['per_icp_score']=1
        with pytest.raises(database[0].Error,match='score_write_once'):rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',(ROUND,json.dumps(scores)))
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s",(lease['scored_run_id'],)); assert cur.fetchone()[0]['per_icp_score']==0

@pytest.fixture(scope='module')
def current_runner_database():
    # Full latest-service proofs use the complete committed recovery chain.
    # Keep the original 470 database for its historical bounded-policy tests.
    from tests.lab_arena import paid_host_quarantine480_postgres_test as current
    generator = current.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(current.SQL.read_text())
        yield db
    finally:
        generator.close()


@pytest.mark.parametrize('recover_successfully',[True,False])
def test_normal_signed_runner_recovers_only_missing_baseline_work_and_preserves_miners(current_runner_database,tmp_path,monkeypatch,recover_successfully):
    exercise_signed_baseline_recovery(current_runner_database, tmp_path, monkeypatch,
                                     recover_successfully)


def exercise_signed_baseline_recovery(database, tmp_path, monkeypatch, recover_successfully):
    """Real signed-runner proof using the caller's complete current migration chain."""
    from dataclasses import replace
    from lab_arena import contracts
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    pg,dsn=database

    class PaidRecoveryTransport(flows.FakeProviderTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            # Post one real fixture charge per successful fake provider job.
            # The ordinary exact-billing bridge reads and settles this record.
            with self._deepline_lock:
                for entry in self._deepline_jobs:
                    entry.update(credits=1, delta=-1)
            return response

    monkeypatch.setattr(flows, 'FakeProviderTransport', PaidRecoveryTransport)
    with pg.connect(**dsn) as conn:
        seed(conn)  # Remove prior controlled fixture state and accounting.
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
    class RecoveryHarness(IntegrityHarness):
        def objects_key(self):
            # These disposable databases repeat round IDs. Their immutable
            # artifacts must not share another database's fixture bucket.
            return 'baseline-recovery-%s-%s' % (dsn['port'], tmp_path.name)

    harness=RecoveryHarness(lambda:pg.connect(**dsn),tmp_path,challengers=['RecoveryMiner'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',per_icp_cost_policy=True,
        rewards_enabled=False,benchmark_icp_count=10)
    harness.service.config.daily_icp_source=lambda **kw:dict(status='ready',set_id=int(kw['set_id']),icps=daily_icps()[:10])
    _enable_verified_proxy_runtime(harness)
    normal_run = harness.run_stage_with_runners

    def run_after_retry_backoff(count=1):
        # Advance only the disposable retry timestamps, as the 472 tests do.
        # The real claim RPC still checks backoff, signatures and every guard.
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' "
                        "WHERE round_id=%s AND kind='score' AND status='pending' AND attempt>1",
                        (harness.round_id,))
        return normal_run(count)

    monkeypatch.setattr(harness, 'run_stage_with_runners', run_after_retry_backoff)
    monkeypatch.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    assert flows._start_round(harness,day=23 if recover_successfully else 24,epoch=62523 if recover_successfully else 62524)==2
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('stage1_scored',runners=1)
    row=harness.service.store.get_round(harness.round_id)
    baseline=next(p for p in row['participants'] if p['is_king'])['submission_id']
    execution=harness.service.store.list_runs(harness.round_id,stage=1,kind='execute')
    failed_source=next(r for r in execution if r['icp_position']==0)
    retained_source=next(r for r in execution if r['icp_position']==3)
    complete=[r for r in execution if r['icp_position'] not in (0,3)]
    before_plan=json.loads(json.dumps(row['stage1_scoring_plan_doc']))
    # Seed the actual resume boundary: ICP0 has two source-provider failures;
    # ICP3 retains accepted source but has exhausted review infrastructure.
    # Only disposable fixture rows are altered. Subsequent work uses normal
    # signed runner, provider bridge, lease, completion and scoring paths.
    frozen_plan={**before_plan,'work_items':[i for i in before_plan['work_items'] if i['icp_position']!=0],
        'zero_rows':[dict(submission_id=baseline,icp_position=0,cause='provider_error')]}
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_judgment_cache WHERE source_scored_run_id IN (%s,%s)',(failed_source['run_id'],retained_source['run_id']))
        cur.execute("DELETE FROM public.lab_arena_runs WHERE kind='score' AND scored_run_id=%s",(failed_source['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='provider_error',result_doc='{\"terminal_status\":\"provider_error\"}',output_ref=NULL,per_icp_score=0,qualification_doc='{\"companies\":[]}' WHERE run_id=%s",(failed_source['run_id'],))
        cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id',assignment_id||':2','attempt',2,'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL,'lease_token_hash',NULL))).* FROM public.lab_arena_runs r WHERE run_id=%s",(failed_source['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",(retained_source['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',result_doc='{\"terminal_status\":\"judge_error\"}',output_ref=NULL WHERE kind='score' AND scored_run_id=%s",(retained_source['run_id'],))
        cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id',assignment_id||':2','attempt',2,'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL,'lease_token_hash',NULL))).* FROM public.lab_arena_runs r WHERE kind='score' AND scored_run_id=%s",(retained_source['run_id'],))
        cur.execute('UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(frozen_plan),harness.round_id))
    # This source-failure fixture has no original ICP0 judge. Remove only the
    # task-created fake judge artifact that preceded the seeded boundary.
    harness.objects._path('arena/%s/scores/items/%s:score:1.json' % (harness.round_id,failed_source['assignment_id'])).unlink()
    if not recover_successfully:
        harness.sandbox.judge_failures.add(('PublicBaseline',3))
    harness.clock.advance_to(harness.schedule()['stage_2_start'])
    harness.service.advance_round(harness.round_id)
    harness.advance_until('scored',runners=1)
    final=harness.service.store.get_round(harness.round_id)
    all_exec=harness.service.store.list_runs(harness.round_id,kind='execute')
    all_judges=harness.service.store.list_runs(harness.round_id,kind='score')
    new_source=next(r for r in all_exec if r['submission_id']==baseline and r['icp_position']==0 and r['attempt']==3)
    assert new_source['status']=='accepted' and new_source['stage']==1
    assert new_source['lease_token_hash'] and new_source['claim_request_id']
    assert new_source['claim_response']['status']=='leased'
    execution_ledger = harness.service.store.list_ledger(run_id=new_source['run_id'])
    assert {'reservation','dispatch','settlement'} <= {item['entry_kind'] for item in execution_ledger}
    assert any(item['entry_kind']=='settlement' and item['amount_microusd']>0
               for item in execution_ledger)
    assert sum(r['submission_id']==baseline and r['icp_position']==3 for r in all_exec)==1
    assert next(r for r in all_exec if r['run_id']==failed_source['assignment_id']+':2')['per_icp_score']==0
    assert final['stage1_scoring_plan_doc']==frozen_plan
    assert [r for r in all_exec if r['run_id'] in {c['run_id'] for c in complete}]==complete
    miners=[r for r in all_exec if r['submission_id']!=baseline]
    assert len(miners)==10 and all(r['status']=='accepted' and r['per_icp_score'] is not None for r in miners)
    if recover_successfully:
        assert new_source['per_icp_score']==80
        assert harness.service._baseline_final_ready(final)
        assert harness.service.publish(harness.round_id)['status']=='ok'
    else:
        assert new_source['per_icp_score'] is None
        assert not harness.service._baseline_final_ready(final)
        assert harness.service.publish(harness.round_id)['reason']=='baseline_incomplete'
        assert final['published_at'] is None and final['reward_activated_at'] is None
        retained_judges = [r for r in all_judges if r['scored_run_id']==retained_source['run_id']]
        assert any(r['status']=='failed' and r['attempt']==3 for r in retained_judges)
        assert any(r['status']=='pending' and r['attempt']>=4 for r in retained_judges)
    harness.service.store.close()

@pytest.mark.parametrize('state',[None,{}, {'inflight_calls':0}, {'inflight_calls':None,'success_unresolved_calls':0}])
def test_missing_cost_counters_fail_closed(database,state):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE OR REPLACE FUNCTION public.lab_arena__successful_icp_cost_state(p_round_id text,p_submission_id text,p_icp_position integer) RETURNS JSONB LANGUAGE SQL STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS %s",('SELECT '+("NULL::JSONB" if state is None else "'"+json.dumps(state)+"'::JSONB"),))
            cur.execute('SET ROLE lab_arena_service')
            cur.execute("SELECT public.lab_arena_recover_baseline_v1(%s,'[]'::jsonb)",(ROUND,)); assert cur.fetchone()[0]['queued']==0
        conn.rollback()  # Restore the real private policy helper, no new artifact.

@pytest.mark.parametrize('fault',['null_cause','null_result','mismatch_result'])
def test_malformed_failed_parent_is_not_a_recovery_grant(database,fault):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault=='null_cause':cur.execute('UPDATE public.lab_arena_runs SET terminal_cause=NULL WHERE attempt=1')
            elif fault=='null_result':cur.execute('UPDATE public.lab_arena_runs SET result_doc=NULL WHERE attempt=1')
            else:cur.execute("UPDATE public.lab_arena_runs SET result_doc='{\"terminal_status\":\"accepted\"}' WHERE attempt=1")
        conn.commit(); assert recover(conn)['queued']==0
