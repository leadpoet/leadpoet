"""Healthy captured work does not need a lock-taking recovery scan per poll."""
from pathlib import Path
import json
import time
from contextlib import contextmanager
import pytest
from tests.lab_arena import ordered_claim_admission520_postgres_test as queue
from tests.lab_arena import test_lab_arena_restart_claim_drain_postgres as guard
from tests.lab_arena import owner_stopped_score_drain454_postgres_test as stopped
from tests.lab_arena import owner_overdue_lease_expiry463_postgres_test as overdue
from tests.lab_arena import settled_score_host_recovery441_postgres_test as paid

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/522-lab-arena-quiescence-recovery-gate.sql'
SIG='public.lab_arena_restart_quiescence_v1(text,text,bigint)'
TOKEN='sha256:'+'b'*64

@pytest.fixture(scope='module')
def database():
    g=queue.database.__wrapped__();db=next(g)
    try:
        with db[0].connect(**db[1]) as c,c.cursor() as q:
            q.execute((ROOT/'scripts/521-lab-arena-proven-credit-closure-label.sql').read_text())
            q.execute(SQL.read_text());q.execute(SQL.read_text())
        yield db
    finally:g.close()

@pytest.fixture(autouse=True)
def reset(database):
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")


def acquire(c):
    with c.cursor() as q:
        q.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=q.fetchone()[0]
    result=guard._acquire(c,generation=generation);c.commit();return result


def test_seven_healthy_captured_leases_skip_expiry_and_keep_provider_rows_unlocked(database):
    queue.queue.test_three_concurrent_real_claims_from_large_post_deadline_queue(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute("SELECT round_id FROM public.lab_arena_rounds WHERE round_id LIKE '%supersession' LIMIT 1");rid=q.fetchone()[0]
        for i in range(4):
            q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,8,%s,%s,%s,%s,4500)',(rid,queue.revision.company.original.hotkey('credit519-concurrent'),[],format(i+1,'032x'),'sha256:'+'a'*64,'sha256:'+'b'*64));assert q.fetchone()[0]['status']=='leased'
    with database[0].connect(**database[1]) as c:captured=acquire(c)
    assert captured['drain']['captured_count']==7
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));post=q.fetchone()[0]
        old=SQL.read_text().split('$old$')[1];new=SQL.read_text().split('$new$')[1]
        assert post.count(new)==1;pre=post.replace(new,old)
        q.execute("SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)");expiry=q.fetchone()[0]
        q.execute('CREATE TABLE gate522_expiry_calls(n integer)');q.execute('GRANT ALL ON gate522_expiry_calls TO lab_arena_owner')
        q.execute(expiry.replace('BEGIN\n',"BEGIN\n  INSERT INTO public.gate522_expiry_calls VALUES(1);\n",1))
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');before=q.fetchone()[0]
        q.execute("SELECT run_id FROM public.lab_arena_runs WHERE status='leased' ORDER BY run_id LIMIT 1");target=q.fetchone()[0]
        q.execute(pre)
    try:
        times={}
        for version in ('before','after'):
            if version=='after':
                with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(SQL.read_text())
            with database[0].connect(**database[1]) as c:
                started=time.perf_counter();state=guard._quiescence(c,generation=captured['guard_generation']);times[version]=time.perf_counter()-started
                assert state['still_leased_count']==7 and state['lost_or_mutated_count']==0
                with c.cursor() as q:
                    q.execute('SELECT count(*) FROM gate522_expiry_calls');assert q.fetchone()==((1 if version=='before' else 0),)
                    q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert q.fetchone()[0]==before
                with database[0].connect(**database[1]) as other,other.cursor() as check:
                    if version=='before':
                        with pytest.raises(database[0].Error) as error:check.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE NOWAIT',(target,))
                        assert error.value.pgcode=='55P03'
                    else:check.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE NOWAIT',(target,));assert check.fetchone()==(1,)
                c.rollback()
        print('HEALTHY_CAPTURE_QUIESCENCE_SECONDS',times,flush=True)
    finally:
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(post);q.execute(expiry);q.execute('DROP TABLE gate522_expiry_calls')


@contextmanager
def legacy_floor_zero(database):
    # Historical replica-mode fixture inserts predate the immutable run-floor
    # column. They represent floor zero explicitly; normal insert auth stays on.
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor SET DEFAULT 0')
    try:yield
    finally:
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute('ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT')


def current_lease(database, *, started=True, error=None, accounting='settled'):
    store,row,old,new=queue.revision.seed(database)
    try:
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{lease_ttl_seconds}','4500'::jsonb),'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"') WHERE round_id=%s",(row['round_id'],))
            q.execute('SET LOCAL session_replication_role=origin')
            q.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(row['round_id'],));fresh=q.fetchone()[0]
            queue.revision.invoke(q,fresh,new)
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            q.execute('SET LOCAL session_replication_role=origin')
            q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
        rid=row['round_id']
    finally:store.close()
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        lease=queue.claim(q,rid);assert lease['status']=='leased'
    with database[0].connect(**database[1]) as c:
        if started:paid._event(c,lease,'runtime.started',{'status':'starting','runtime':'runsc','lease_generation':lease['lease_generation']},token_hash=TOKEN)
        if accounting:
            assert paid._rpc(c,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',100,'{}'::jsonb,4500)",(lease['run_id'],TOKEN,paid.CALL))['status']=='reserved'
            if accounting!='reservation':assert paid._dispatch(c,lease,token_hash=TOKEN)['status']=='dispatched'
            if accounting=='settled':assert paid._settle(c,lease,token_hash=TOKEN)['status']=='settled'
        if error:
            paid._event(c,lease,'runtime.error',error,token_hash=TOKEN)
        captured=acquire(c)
    return lease,captured

HOST_ERROR={'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError',
            'runtime_host_reason':'sandbox_launcher_signaled','launch_exit_code':-2,'launch_timed_out':False}

@pytest.mark.parametrize('fault',['none','no_start','open_dispatch','unbound_error','setup_image','setup_runtime'])
def test_signal_keeps_full_stopped_host_authorization(database,fault):
    error=dict(HOST_ERROR)
    if fault=='unbound_error':error['error_class']='UntrustedHostError'
    if fault.startswith('setup_'):
        error={'status':'abandoned','failure_stage':'setup','error_class':'LeasedImageError' if fault=='setup_image' else 'RuntimeHostError'}
    lease,captured=current_lease(database,started=fault!='no_start',error=error,
                                accounting=None if fault.startswith('setup_') else ('dispatch' if fault=='open_dispatch' else 'settled'))
    with database[0].connect(**database[1]) as c:
        before=paid._audit(c,lease)
        state=guard._quiescence(c,generation=captured['guard_generation']);c.commit()
        after=paid._audit(c,lease)
        assert after['lab_arena_ledger']==before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events']==before['lab_arena_trajectory_events']
        if fault=='none':
            assert state['expired_receipt_count']==1 and state['pending_retry_count']==1
            assert after['run'][:2]==('failed','lease_expired')
        else:
            assert state['still_leased_count']==1 and after==before

@pytest.mark.parametrize('kind',['execute','score'])
@pytest.mark.parametrize('head',['reservation','dispatch'])
def test_overdue_without_runtime_error_keeps_exact_liability_and_retry(database,kind,head):
    if kind=='execute':
        with legacy_floor_zero(database):
            overdue.test_expired_captured_lease_preserves_evidence_liability_and_dormant_retry(database,None,head,kind)
        return
    lease,captured=current_lease(database,started=False,accounting=head)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
        q.execute('SET LOCAL session_replication_role=origin')
    with database[0].connect(**database[1]) as c:
        before=paid._audit(c,lease);state=guard._quiescence(c,generation=captured['guard_generation']);c.commit();after=paid._audit(c,lease)
        assert state['expired_receipt_count']==1 and state['lost_or_mutated_count']==0
        assert after['run'][:2]==('failed','lease_expired')
        assert after['lab_arena_trajectory_events']==before['lab_arena_trajectory_events']
        assert after['lab_arena_ledger'][:-1]==before['lab_arena_ledger']
        assert after['lab_arena_ledger'][-1]['entry_kind']==('uncertain' if head=='dispatch' else 'recovery')


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_source_and_security_drift_rejected_before_replay(database,installed,drift):
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));saved=q.fetchone()[0]
        if not installed:q.execute(saved.replace(SQL.read_text().split('$new$')[1],SQL.read_text().split('$old$')[1]))
        q.execute('SAVEPOINT hostile')
        if drift=='source':
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));q.execute(q.fetchone()[0].replace('BEGIN\n','BEGIN\n  -- unknown source drift\n',1))
        else:
            q.execute({'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres','acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon','secdef':'ALTER FUNCTION '+SIG+' SECURITY INVOKER','volatility':'ALTER FUNCTION '+SIG+' STABLE','path':'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog'}[drift])
        body=SQL.read_text().split('DO $quiescence_recovery_gate$')[1].split('$quiescence_recovery_gate$;')[0]
        with pytest.raises(database[0].Error):q.execute('DO $gate$'+body+'$gate$')
        q.execute('ROLLBACK TO SAVEPOINT hostile');q.execute(saved)

@pytest.mark.parametrize('version',['before','after'])
def test_actual_v3_completion_and_healthy_quiescence_lock_cycle(database,version):
    from concurrent.futures import ThreadPoolExecutor
    import queue as messages
    lease,captured=current_lease(database,error=None)
    qgate,cgate=7522001,7522002
    blockers=[]
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute("SELECT pg_get_functiondef('public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)'::regprocedure)");complete=q.fetchone()[0]
        q.execute("SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)");expiry=q.fetchone()[0]
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));quiescence=q.fetchone()[0]
        if version=='before':q.execute(quiescence.replace(SQL.read_text().split('$new$')[1],SQL.read_text().split('$old$')[1]))
        marker='  WHERE run_id = p_run_id FOR UPDATE;'
        assert complete.count(marker)==1
        q.execute(complete.replace(marker,marker+'\n  PERFORM pg_advisory_xact_lock('+str(cgate)+');',1))
        marker='SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;'
        assert expiry.count(marker)==1
        q.execute(expiry.replace(marker,marker+'\n  PERFORM pg_advisory_xact_lock('+str(qgate)+');',1))
    pool=ThreadPoolExecutor(max_workers=2);ids=messages.Queue()
    def operation(kind):
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute("SET LOCAL statement_timeout='8s'");q.execute("SET LOCAL deadlock_timeout='100ms'")
            ids.put((kind,c.get_backend_pid()))
            try:
                if kind=='complete':
                    q.execute("SELECT public.lab_arena_complete_attempt_v3(%s,%s,'{\"terminal_status\":\"judge_error\"}'::jsonb,'judge_error',NULL,NULL,'[]'::jsonb,%s)",(lease['run_id'],TOKEN,'sha256:'+'c'*64));result=q.fetchone()[0]
                else:result=guard._quiescence(c,generation=captured['guard_generation'])
                return {'result':result}
            except database[0].Error as exc:
                c.rollback();return {'sqlstate':exc.pgcode}
    def wait_for(predicate):
        deadline=time.monotonic()+5
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            while True:
                q.execute(predicate)
                if q.fetchone()[0]:return
                assert time.monotonic()<deadline,'controlled lock boundary not reached'
                time.sleep(.01)
    try:
        for gate in (qgate,cgate):
            c=database[0].connect(**database[1]);blockers.append(c)
            with c.cursor() as q:q.execute('SELECT pg_advisory_xact_lock(%s)',(gate,))
        completed=pool.submit(operation,'complete');kind,cpid=ids.get(timeout=5);assert kind=='complete'
        wait_for('SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype=\'advisory\' AND objid='+str(cgate)+' AND NOT granted)')
        quiet=pool.submit(operation,'quiescence');kind,qpid=ids.get(timeout=5);assert kind=='quiescence'
        if version=='before':
            wait_for('SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype=\'advisory\' AND objid='+str(qgate)+' AND NOT granted)')
            blockers[0].commit()
            wait_for('SELECT '+str(cpid)+'=ANY(pg_blocking_pids('+str(qpid)+'))')
        else:
            state=quiet.result(timeout=5);assert state['result']['still_leased_count']==1
        blockers[1].commit();blockers[0].commit()
        results=[completed.result(timeout=8),quiet.result(timeout=8)]
        if version=='before':assert any(r.get('sqlstate')=='40P01' for r in results),results
        else:
            assert all('sqlstate' not in r for r in results),results
            assert results[0]['result']['status']=='failed'
        print('COMPLETION_QUIESCENCE_LOCK_CONTROL',version,results,flush=True)
    finally:
        for c in blockers:c.rollback();c.close()
        pool.shutdown(wait=True)
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(complete);q.execute(expiry);q.execute(quiescence)

@pytest.mark.parametrize('timing',['due','after_gate'])
def test_natural_expiry_retains_fresh_clock_without_runtime_start(database,timing):
    lease,captured=current_lease(database,started=False,accounting=None)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));original=q.fetchone()[0]
        q.execute("SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)");expiry=q.fetchone()[0]
        q.execute('CREATE TABLE gate522_clock_calls(phase text)');q.execute('GRANT ALL ON gate522_clock_calls TO lab_arena_owner')
        q.execute(expiry.replace('BEGIN\n',"BEGIN\n  INSERT INTO public.gate522_clock_calls VALUES(coalesce(current_setting('test.gate522_phase',TRUE),'early'));\n",1))
        if timing=='after_gate':
            marker='  -- lab_arena_guard_owned_natural_expiry_v1'
            assert original.count(marker)==1
            q.execute(original.replace(marker,"  PERFORM set_config('test.gate522_phase','late',TRUE);\n  PERFORM pg_sleep(2.2);\n"+marker,1))
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+make_interval(secs=>%s) WHERE run_id=%s",(2 if timing=='after_gate' else 0,lease['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
    try:
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            state=guard._quiescence(c,generation=captured['guard_generation'])
            assert state['expired_receipt_count']==1 and state['lost_or_mutated_count']==0
            q.execute('SELECT phase FROM gate522_clock_calls');phases=[r[0] for r in q.fetchall()]
            assert phases==(['late'] if timing=='after_gate' else ['early'])
    finally:
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(original);q.execute(expiry);q.execute('DROP TABLE gate522_clock_calls')
