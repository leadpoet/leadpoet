"""Failed baseline chains reuse the authenticated queue after execution closes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import bounded_baseline_recovery470_postgres_test as base
from tests.lab_arena import miner_credit_terminal484_postgres_test as previous
from tests.lab_arena import baseline_recovery_priority481_postgres_test as priority

ROOT = Path(__file__).parents[2]
SQL = ROOT/'scripts/485-lab-arena-baseline-infrastructure-continuation.sql'


def functions(cur):
    cur.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),proowner,proacl::text,prosecdef,provolatile,proconfig FROM pg_proc WHERE pronamespace='public'::regnamespace AND proname LIKE 'lab_arena_%%' AND prokind='f' ORDER BY oid")
    return cur.fetchall()


@pytest.fixture(scope='module')
def database():
    gen = previous.database.__wrapped__(); db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute((ROOT/'scripts/478-lab-arena-baseline-closed-settlement-recovery.sql').read_text())
            conn.commit()
            before = functions(cur)
            assignment = seed(conn)
            # New service on the old schema must hold this phase until485
            # activates the existing queue; otherwise a deploy-first race can
            # strand failed attempt3 in stage2_judged/scored.
            from lab_arena.service import ArenaService
            from lab_arena.store import ArenaStore,PsycopgTransport
            import threading
            service=object.__new__(ArenaService)
            service._store=ArenaStore(PsycopgTransport(lambda:db[0].connect(**db[1])),lease_ttl_seconds=4500)
            service._lock=threading.RLock()
            service._round=lambda rid:service._store.get_round(rid)
            service.now=lambda:datetime.now(timezone.utc)
            original=priority.preserved(conn)
            assert service._advance_round_locked(base.ROUND)=={
                'status':'waiting','round_status':'stage2_scoring',
                'reason':'baseline_infrastructure_recovery'}
            assert priority.preserved(conn)==original
            service._store.close()
            text = SQL.read_text()
            first_hash = text.split("v_preimage CONSTANT TEXT := '")[1].split("'")[0]
            with pytest.raises(db[0].Error, match='preimage differs'):
                cur.execute(text.replace(first_hash, '0'*64, 1))
            conn.rollback()
            assert functions(cur) == before
            post_hash = text.split("v_postimage CONSTANT TEXT := '")[1].split("'")[0]
            with pytest.raises(db[0].Error, match='postimage differs'):
                cur.execute(text.replace(post_hash, '0'*64, 1))
            conn.rollback(); assert functions(cur) == before
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs) TO lab_arena_service')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback(); assert functions(cur) == before
            cur.execute('CREATE ROLE baseline485_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO baseline485_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO baseline485_migrator WITH GRANT OPTION')
            cur.execute('SET ROLE baseline485_migrator')
            cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
            assert cur.fetchone()==(False,)
            cur.execute(text); after = functions(cur)
            cur.execute(text); assert functions(cur) == after
            cur.execute('RESET ROLE')
            assert [(r[0],r[2:]) for r in before] == [(r[0],r[2:]) for r in after]
            assert sum(a[1] != b[1] for a,b in zip(before,after)) == 7
            cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
            assert cur.fetchone()==(False,False)
        yield db
    finally: gen.close()


def seed(conn, status='stage2_scoring'):
    # Exactly nine accepted execution/judge pairs plus 1 lease loss and 2
    # signed provider failures. No miner credit hold can mask close behavior.
    run = priority.seed(conn)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='provider_error',result_doc='{\"terminal_status\":\"provider_error\"}' WHERE run_id=%s", (run,))
        cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='lease_expired',result_doc=NULL WHERE run_id=%s", (run[:-1]+'1',))
        cur.execute("UPDATE public.lab_arena_runs SET result_doc='{\"terminal_status\":\"accepted\"}' WHERE status='accepted'")
        cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb((clock_timestamp()-interval '1 hour')::text))", (status,))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit(); return run[:-1]


def cool(conn, seconds=1200):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-(%s*interval '1 second') WHERE status='pending'", (seconds,))
    conn.commit()


def complete(conn, run, cause='accepted', output='new-output'):
    return base.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
        (run['run_id'],base.TOKEN,json.dumps({'terminal_status':cause}),cause,output if cause=='accepted' else ''))


def final_ready(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)', (base.ROUND,))
        return cur.fetchone()[0]


@pytest.mark.parametrize('status', ['stage2','stage2_closed','stage2_scoring'])
def test_failed_third_attempt_has_one_successor_through_provider_judge_and_final_guard(database,status):
    with database[0].connect(**database[1]) as conn:
        assignment = seed(conn,status)
        before = priority.preserved(conn)
        assert not final_ready(conn)
        assert base.recover(conn) == {'status':'ok','queued':1}
        assert base.recover(conn) == {'status':'ok','queued':0}
        # Fresh pending successor cannot bypass the bounded exponential backoff.
        assert priority.claim(conn)['status']=='no_pending'
        cool(conn)
        if status=='stage2':
            assert base.rpc(conn,'lab_arena_close_stage(%s,2::smallint)',(base.ROUND,))['status']=='closed'
        if status!='stage2_scoring':
            assert base.rpc(conn,"lab_arena_open_scoring_v2(%s,2::smallint,'[]'::jsonb)",(base.ROUND,))['status']=='ok'
            assert base.recover(conn)['queued']==0
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='draining'
        lease=priority.claim(conn)
        assert (lease['status'],lease['run_id'],lease['attempt'])==('leased',assignment+'4',4),lease
        call='sha256:'+'f'*64
        assert base.rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],base.TOKEN,call))['status']=='reserved'
        assert base.rpc(conn,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],base.TOKEN,call))['status']=='dispatched'
        assert base.rpc(conn,'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',(lease['run_id'],base.TOKEN,call,json.dumps({'status':200,'call_succeeded':True})))['status']=='settled'
        assert complete(conn,lease)['status']=='accepted'
        # An accepted execute with no judge holds close independently of miners.
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='draining'
        assert base.recover(conn,[base.scope_item(conn,lease['run_id'])])['queued']==1
        assert base.recover(conn,[base.scope_item(conn,lease['run_id'])])['queued']==0
        judge=priority.claim(conn); assert judge['scored_run_id']==lease['run_id']
        # The existing general judge chain, including attempt 3+, owns retries.
        for attempt in (1,2,3):
            assert judge['attempt']==attempt
            assert complete(conn,judge,'judge_error')['confirmation_attempt']==attempt+1
            cool(conn); judge=priority.claim(conn)
        assert judge['attempt']==4
        assert complete(conn,judge,output='judge-output')['status']=='accepted'
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='closed'
        assert not final_ready(conn)
        scores=[dict(run_id=lease['run_id'],per_icp_score=0,qualification_doc={'companies':[]})]
        assert base.rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',(base.ROUND,json.dumps(scores)))['recorded']==1
        assert final_ready(conn)
        after=priority.preserved(conn)
        # Only new execution/judgments/costs and expected round phase changed.
        assert after[0]['configuration_doc']==before[0]['configuration_doc']
        assert after[1]==before[1]
        assert all(row in after[2] for row in before[2])
        assert base.recover(conn)['status']=='stale'


@pytest.mark.parametrize('fault', ['model_error','credential_error','invalid_output','budget_exhausted',
    'unsigned','output','accepted','live','missing_parent','retired','paused','drain','deadline','published','cancelled','inflight','unknown_success','prior_day'])
def test_failed_chain_controls_remain_closed(database,fault):
    with database[0].connect(**database[1]) as conn:
        assignment=seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault in ('model_error','credential_error','invalid_output','budget_exhausted'):
                cur.execute('UPDATE public.lab_arena_runs SET terminal_cause=%s,result_doc=jsonb_build_object(\'terminal_status\',%s) WHERE run_id=%s',(fault,fault,assignment+'3'))
            elif fault=='unsigned':cur.execute('UPDATE public.lab_arena_runs SET result_doc=NULL WHERE run_id=%s',(assignment+'3',))
            elif fault=='output':cur.execute("UPDATE public.lab_arena_runs SET output_ref='saved-output' WHERE run_id=%s",(assignment+'3',))
            elif fault in ('accepted','live'):cur.execute('UPDATE public.lab_arena_runs SET status=%s WHERE run_id=%s',('accepted' if fault=='accepted' else 'leased',assignment+'3'))
            elif fault=='missing_parent':cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',(assignment+'2',))
            elif fault=='retired':cur.execute('UPDATE public.lab_arena_runs SET champion_restart_required=true WHERE run_id=%s',(assignment+'2',))
            elif fault=='paused':cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
            elif fault=='drain':cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining'")
            elif fault=='deadline':cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
            elif fault in ('published','cancelled'):cur.execute('UPDATE public.lab_arena_rounds SET status=%s',(fault,))
            elif fault=='prior_day':cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-13'")
            elif fault in ('inflight','unknown_success'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES (%s,%s,%s,%s,%s,1,'sha256:'||repeat('b',64),'deepline','generic_http_request','host',100,%s::jsonb)",('dispatch' if fault=='inflight' else 'uncertain',base.MINER,base.ROUND,base.SUB,assignment+'3',json.dumps({'call':{'call_succeeded':True}})))
        conn.commit()
        result=base.recover(conn)
        # A restart drain can queue work but can never claim it.
        assert result['queued']==(1 if fault=='drain' else 0)
        if fault=='drain':
            cool(conn);assert priority.claim(conn)['status']=='paused'
        if fault in ('model_error','credential_error','invalid_output','budget_exhausted','unsigned','missing_parent','retired','deadline','prior_day'):
            assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='closed'


@pytest.mark.parametrize('cause,provider_status', [('provider_error',429),('provider_error',502),('lease_expired',None),('credential_error',None)])
def test_further_failed_attempts_continue_only_infrastructure_with_backoff(database,cause,provider_status):
    with database[0].connect(**database[1]) as conn:
        assignment=seed(conn);assert base.recover(conn)['queued']==1
        cool(conn);lease=priority.claim(conn);assert lease['attempt']==4
        if provider_status is not None:
            call='sha256:'+'f'*64
            assert base.rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],base.TOKEN,call))['status']=='reserved'
            assert base.rpc(conn,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],base.TOKEN,call))['status']=='dispatched'
            assert base.rpc(conn,'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)',(lease['run_id'],base.TOKEN,call,json.dumps({'status':provider_status,'call_succeeded':False})))['status']=='settled'
        if cause=='lease_expired':
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
            conn.commit();assert base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))['expired']==1
        else:assert complete(conn,lease,cause)['status']=='failed'
        assert base.recover(conn)['queued']==(0 if cause=='credential_error' else 1)
        if cause!='credential_error':
            assert priority.claim(conn)['status']=='no_pending'
            cool(conn);assert priority.claim(conn)['attempt']==5


def test_concurrent_ticks_create_one_successor(database):
    with database[0].connect(**database[1]) as conn:seed(conn)
    def tick(_):
        with database[0].connect(**database[1]) as conn:return base.recover(conn)['queued']
    with ThreadPoolExecutor(max_workers=4) as pool:assert sorted(pool.map(tick,range(4)))==[0,0,0,1]


def test_signed_runner_completes_missing_tenth_pair_and_publishes(database,tmp_path,monkeypatch):
    """Real signed gateway/runner/provider/scorer path from the failed boundary."""
    from dataclasses import replace
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    pg,dsn=database
    with pg.connect(**dsn) as conn:base.seed(conn)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')

    class PaidTransport(flows.FakeProviderTransport):
        def send(self, **kwargs):
            response=super().send(**kwargs)
            with self._deepline_lock:
                for entry in self._deepline_jobs:entry.update(credits=1,delta=-1)
            return response

    class Harness(IntegrityHarness):
        def objects_key(self):return 'baseline485-%s-%s'%(dsn['port'],tmp_path.name)

    monkeypatch.setattr(flows,'FakeProviderTransport',PaidTransport)
    monkeypatch.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    harness=Harness(lambda:pg.connect(**dsn),tmp_path,challengers=['RecoveryMiner'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',per_icp_cost_policy=True,
        rewards_enabled=False,benchmark_icp_count=10)
    harness.service.config.daily_icp_source=lambda **kw:dict(status='ready',set_id=int(kw['set_id']),icps=daily_icps()[:10])
    _enable_verified_proxy_runtime(harness)
    original_run=harness.run_stage_with_runners
    def run_after_backoff(count=1):
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending' AND attempt>1",(harness.round_id,))
        return original_run(count)
    monkeypatch.setattr(harness,'run_stage_with_runners',run_after_backoff)
    assert flows._start_round(harness,day=25,epoch=62525)==2
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('scored',runners=1)
    round_id=harness.round_id
    row=harness.service.store.get_round(round_id)
    baseline=next(p['submission_id'] for p in row['participants'] if p['is_king'])
    executions=harness.service.store.list_runs(round_id,kind='execute')
    failed=next(r for r in executions if r['submission_id']==baseline and r['icp_position']==0)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_judgment_cache WHERE source_scored_run_id=%s',(failed['run_id'],))
        cur.execute("DELETE FROM public.lab_arena_runs WHERE kind='score' AND scored_run_id=%s",(failed['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',result_doc=NULL,output_ref=NULL,per_icp_score=0,qualification_doc='{\"companies\":[]}' WHERE run_id=%s",(failed['run_id'],))
        for attempt in (2,3):
            cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id',assignment_id||':'||%s::text,'attempt',%s,'terminal_cause','provider_error','result_doc',jsonb_build_object('terminal_status','provider_error'),'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL,'lease_token_hash',NULL))).* FROM public.lab_arena_runs r WHERE run_id=%s",(attempt,attempt,failed['run_id']))
        # Fixture represents the frozen remaining-day boundary. No production
        # schedule changes occur; the configuration is fixed before proof starts.
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb((clock_timestamp()-interval '1 hour')::text)) WHERE round_id=%s",(round_id,))
    harness.objects._path('arena/%s/scores/items/%s:score:1.json'%(round_id,failed['assignment_id'])).unlink()
    frozen=harness.service.store.get_round(round_id)
    old_runs=harness.service.store.list_runs(round_id)
    preserved=[r for r in old_runs if r['status']=='accepted']
    assert len([r for r in preserved if r['submission_id']==baseline])==18
    assert not harness.service._baseline_final_ready(frozen)
    before_ledger=harness.service.store.list_ledger()
    assert harness.service.publish(round_id)=={'status':'stale','round_status':'stage2_scoring'}
    assert harness.service.store.get_round(round_id)==frozen
    assert harness.service.store.list_runs(round_id)==old_runs
    assert harness.service.store.list_ledger()==before_ledger
    harness.service.advance_round(round_id)
    pending=harness.service.store.list_runs(round_id,status='pending')
    assert len(pending)==1 and pending[0]['attempt']==4 and pending[0]['icp_position']==0
    assert harness.service.advance_round(round_id)['status']=='waiting'
    harness.advance_until('scored',runners=1)
    final=harness.service.store.get_round(round_id)
    rows=harness.service.store.list_runs(round_id)
    new=next(r for r in rows if r['run_id']==failed['assignment_id']+':4')
    assert new['status']=='accepted'
    # Continuation judgments remain unmaterialized until publication's guarded
    # score writer validates the accepted output and its bound judgment.
    assert new['per_icp_score'] is None and new['qualification_doc'] is None
    assert new['lease_token_hash'] and new['claim_request_id'] and new['claim_response']['status']=='leased'
    judges=[r for r in rows if r['kind']=='score' and r['scored_run_id']==new['run_id']]
    assert len(judges)==1 and judges[0]['status']=='accepted'
    assert [r for r in rows if r['run_id'] in {r['run_id'] for r in preserved}]==preserved
    assert final['configuration_doc']==frozen['configuration_doc']
    assert final['stage1_scoring_plan_doc']==frozen['stage1_scoring_plan_doc']
    ledger=harness.service.store.list_ledger(run_id=new['run_id'])
    assert {'reservation','dispatch','settlement'}<={r['entry_kind'] for r in ledger}
    assert any(r['entry_kind']=='settlement' and r['amount_microusd']>0 for r in ledger)
    assert not harness.service._baseline_final_ready(final)
    assert final['publication_doc'] is None and final['reward_activated_at'] is None
    before_publication_ledger=harness.service.store.list_ledger()
    assert harness.service.publish(round_id)['status']=='ok'
    published=harness.service.store.get_round(round_id)
    assert published['status']=='published'
    scored=harness.service.store.get_run(new['run_id'])
    assert scored['per_icp_score']==80 and scored['qualification_doc'] is not None
    assert {k:v for k,v in scored.items() if k not in ('per_icp_score','qualification_doc','updated_at')}=={
        k:v for k,v in new.items() if k not in ('per_icp_score','qualification_doc','updated_at')}
    assert datetime.fromisoformat(scored['updated_at'])>=datetime.fromisoformat(new['updated_at'])
    assert [r for r in harness.service.store.list_runs(round_id) if r['run_id']!=new['run_id']]==[
        r for r in rows if r['run_id']!=new['run_id']]
    assert harness.service.store.list_ledger()==before_publication_ledger
    assert harness.service._baseline_final_ready(published)
    harness.service.store.close()


def test_unresolved_cost_holds_phase_then_settlement_releases_one_successor(database):
    with database[0].connect(**database[1]) as conn:
        assignment=seed(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) VALUES ('uncertain',%s,%s,%s,%s,1,'sha256:'||repeat('e',64),'deepline','generic_http_request','host',100,'{\"call\":{\"call_succeeded\":true}}')",(base.MINER,base.ROUND,base.SUB,assignment+'3'))
        conn.commit()
        assert base.recover(conn)['queued']==0
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='draining'
        with conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) VALUES ('settlement',%s,%s,%s,%s,1,'sha256:'||repeat('e',64),'deepline','generic_http_request','host',100,'{}','{\"status\":502,\"call_succeeded\":false}')",(base.MINER,base.ROUND,base.SUB,assignment+'3'))
        conn.commit();assert base.recover(conn)['queued']==1
        assert base.recover(conn)['queued']==0
        cool(conn);lease=priority.claim(conn);assert lease['attempt']==4
        assert complete(conn,lease)['status']=='accepted'
        assert base.recover(conn,[base.scope_item(conn,lease['run_id'])])['queued']==1
        judge=priority.claim(conn);assert complete(conn,judge)['status']=='accepted'
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='closed'
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE run_id=%s",(assignment+'3',));assert cur.fetchone()==(2,200)


def test_final_close_blocks_new_claims_but_drains_leased_execute_and_judge(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn);assert base.recover(conn)['queued']==1
        cool(conn);lease=priority.claim(conn);assert lease['attempt']==4
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
        conn.commit()
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='draining'
        assert complete(conn,lease)['status']=='accepted'
        assert base.recover(conn,[base.scope_item(conn,lease['run_id'])])['queued']==1
        judge=priority.claim(conn);assert judge['kind']=='score'
        assert complete(conn,judge)['status']=='accepted'
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='closed'


def test_pending_execute_cannot_claim_after_final_close(database):
    with database[0].connect(**database[1]) as conn:
        assignment=seed(conn);assert base.recover(conn)['queued']==1
        cool(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
        conn.commit();assert priority.claim(conn)['status']=='no_pending'
        assert base.rpc(conn,'lab_arena_close_scoring(%s,2::smallint)',(base.ROUND,))['status']=='closed'
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s',(assignment+'4',));assert cur.fetchone()==('failed','stage_closed')


def test_paid_host_quarantine_still_blocks_late_recovery_claim(database):
    from tests.lab_arena import paid_host_quarantine480_postgres_test as quarantine
    with database[0].connect(**database[1]) as conn:
        assignment=seed(conn)
        parents=[f'{base.ROUND}:{sub}:2:{position}:1'
                 for sub,position in [(priority.MINERS[0],0),(priority.MINERS[0],1),(priority.MINERS[1],0)]]
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_runs SET runner_hotkey=%s,lease_generation=1,result_doc=NULL WHERE run_id=ANY(%s)',(base.RUNNER,parents))
        conn.commit()
        quarantine._paid_faults(conn,parents,expired=True)
        assert quarantine._proofs(conn,parents)==[True]*3
        assert base.recover(conn)['queued']==1
        cool(conn);assert priority.claim(conn)['status']=='no_pending'


@pytest.mark.parametrize('limit',['budget','calls'])
def test_existing_provider_budget_and_call_cap_still_limit_successors(database,limit):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if limit=='calls':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas}',%s::jsonb)",(json.dumps({provider:1 for provider in ('openrouter','deepline','scrapingdog')}),))
        conn.commit();assert base.recover(conn)['queued']==1
        cool(conn);lease=priority.claim(conn)
        call='sha256:'+'f'*64
        assert base.rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],base.TOKEN,call))['status']=='reserved'
        assert base.rpc(conn,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],base.TOKEN,call))['status']=='dispatched'
        assert base.rpc(conn,'lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,4500)',(lease['run_id'],base.TOKEN,call,4000000 if limit=='budget' else 100,json.dumps({'status':200,'call_succeeded':True})))['status']=='settled'
        assert base.rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)",(lease['run_id'],base.TOKEN,'sha256:'+'e'*64))['status']!='reserved'
