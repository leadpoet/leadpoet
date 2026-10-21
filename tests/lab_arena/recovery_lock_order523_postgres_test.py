"""Canonical recovery and completion lock the round before its run."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue as messages
import json,re,time
import pytest
from tests.lab_arena import quiescence_recovery_gate522_postgres_test as gate
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import reset

ROOT=Path(__file__).parents[2]
SQL=ROOT/'scripts/523-lab-arena-recovery-lock-order.sql'
SIGNATURES={
 'v3_lock_order':'public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)',
 'healthy_expiry':'public.lab_arena_expire_leases(text)',
}

@pytest.fixture(scope='module')
def database():
    g=gate.database.__wrapped__();db=next(g)
    try:
        with db[0].connect(**db[1]) as c,c.cursor() as q:
            q.execute(SQL.read_text());q.execute(SQL.read_text())
        yield db
    finally:g.close()


def source(q,sig):
    q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(sig,));return q.fetchone()[0]


def preimage(label,post):
    text=SQL.read_text()
    old=dict(re.findall(r'\$old_('+label+r'_\d+)\$(.*?)\$old_\1\$',text,re.S))
    new=dict(re.findall(r'\$new_('+label+r'_\d+)\$(.*?)\$new_\1\$',text,re.S))
    assert old.keys()==new.keys()
    for tag in reversed(old):
        assert post.count(new[tag])==1;post=post.replace(new[tag],old[tag])
    return post


def complete(q,lease,*,token=gate.TOKEN,result=None,cause='judge_error',output=None,evidence=None):
    q.execute('SELECT public.lab_arena_complete_attempt_v3(%s,%s,%s::jsonb,%s,%s,NULL,%s::jsonb,%s)',
      (lease['run_id'],token,json.dumps(result or {'terminal_status':cause}),cause,output,json.dumps(evidence or []),'sha256:'+'c'*64))
    return q.fetchone()[0]


def test_ordinary_healthy_expiry_skips_recovery_locks_without_changing_work(database):
    gate.test_seven_healthy_captured_leases_skip_expiry_and_keep_provider_rows_unlocked(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
        post=source(q,SIGNATURES['healthy_expiry']);q.execute(preimage('healthy_expiry',post))
        q.execute("SELECT round_id,run_id FROM public.lab_arena_runs WHERE status='leased' LIMIT 1");rid,target=q.fetchone()
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');before=q.fetchone()[0]
    try:
        times={}
        for version in ('before','after'):
            if version=='after':
                with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(post)
            with database[0].connect(**database[1]) as c,c.cursor() as q:
                started=time.perf_counter();q.execute('SELECT public.lab_arena_expire_leases(%s)',(rid,));result=q.fetchone()[0];times[version]=time.perf_counter()-started
                assert result=={'status':'ok','expired':0,'retried':0}
                q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert q.fetchone()[0]==before
                with database[0].connect(**database[1]) as other,other.cursor() as check:
                    if version=='before':
                        with pytest.raises(database[0].Error) as error:check.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE NOWAIT',(target,))
                        assert error.value.pgcode=='55P03'
                    else:check.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE NOWAIT',(target,));assert check.fetchone()==(1,)
                c.rollback()
        print('ORDINARY_HEALTHY_EXPIRY_SECONDS',times,flush=True)
    finally:
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(post)


@pytest.mark.parametrize('case',['overdue','stopped'])
@pytest.mark.parametrize('version',['before','after'])
def test_v3_and_real_recovery_have_consistent_lock_order(database,case,version):
    lease,captured=gate.current_lease(database,error=gate.HOST_ERROR if case=='stopped' else None)
    qgate,cgate=7523001,7523002
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        if case=='overdue':
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
            q.execute('SET LOCAL session_replication_role=origin')
        post=source(q,SIGNATURES['v3_lock_order']);chosen=preimage('v3_lock_order',post) if version=='before' else post
        expiry=source(q,SIGNATURES['healthy_expiry'])
        marker='  WHERE run_id = p_run_id FOR UPDATE;';assert chosen.count(marker)==1
        q.execute(chosen.replace(marker,marker+'\n  PERFORM pg_advisory_xact_lock('+str(cgate)+');',1))
        marker='SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;';assert expiry.count(marker)==1
        q.execute(expiry.replace(marker,marker+'\n  PERFORM pg_advisory_xact_lock('+str(qgate)+');',1))
    ids=messages.Queue();blockers=[];pool=ThreadPoolExecutor(max_workers=2)
    def operation(kind):
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute("SET LOCAL statement_timeout='8s'");q.execute("SET LOCAL deadlock_timeout='100ms'")
            ids.put((kind,c.get_backend_pid()))
            try:return {'result':complete(q,lease) if kind=='complete' else gate.guard._quiescence(c,generation=captured['guard_generation'])}
            except database[0].Error as exc:c.rollback();return {'sqlstate':exc.pgcode}
    def wait_for(predicate):
        deadline=time.monotonic()+5
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            while True:
                q.execute(predicate)
                if q.fetchone()[0]:return
                assert time.monotonic()<deadline,'controlled lock boundary not reached'
                time.sleep(.01)
    try:
        for lock in (qgate,cgate):
            c=database[0].connect(**database[1]);blockers.append(c)
            with c.cursor() as q:q.execute('SELECT pg_advisory_xact_lock(%s)',(lock,))
        completed=pool.submit(operation,'complete');kind,cpid=ids.get(timeout=5);assert kind=='complete'
        wait_for("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid="+str(cgate)+' AND NOT granted)')
        quiet=pool.submit(operation,'quiescence');kind,qpid=ids.get(timeout=5);assert kind=='quiescence'
        if version=='before':
            wait_for("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid="+str(qgate)+' AND NOT granted)')
            blockers[0].commit();wait_for('SELECT '+str(cpid)+'=ANY(pg_blocking_pids('+str(qpid)+'))')
        else:
            # The completion already holds round SHARE. Recovery waits there,
            # before acquiring its run lock, so the completion can finish.
            wait_for('SELECT '+str(cpid)+'=ANY(pg_blocking_pids('+str(qpid)+'))')
        blockers[1].commit();blockers[0].commit()
        results=[completed.result(timeout=8),quiet.result(timeout=8)]
        if version=='before':assert any(r.get('sqlstate')=='40P01' for r in results),results
        else:
            assert all('sqlstate' not in r for r in results),results
            assert results[0]['result']['status']==('stale' if case=='overdue' else 'failed')
            assert results[1]['result']['lost_or_mutated_count']==0
        print('V3_RECOVERY_LOCK_CONTROL',case,version,results,flush=True)
    finally:
        for c in blockers:c.rollback();c.close()
        pool.shutdown(wait=True)
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(post);q.execute(expiry)


@pytest.mark.parametrize('fault',['missing','stale_token','invalid_result','invalid_output','invalid_evidence'])
def test_completion_input_and_authority_fences_unchanged(database,fault):
    lease,captured=gate.current_lease(database,error=None)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');before=q.fetchone()[0]
        q.execute('SAVEPOINT invalid')
        if fault=='stale_token':assert complete(q,lease,token='sha256:'+'f'*64)['status']=='stale'
        else:
            bad=dict(lease);kwargs={}
            if fault=='missing':bad['run_id']='missing-run'
            if fault=='invalid_result':kwargs['result']={'terminal_status':'accepted'}
            if fault=='invalid_output':kwargs['output']='invalid-not-empty'
            if fault=='invalid_evidence':kwargs['evidence']=[{'not':'authority'}]
            with pytest.raises(database[0].Error):complete(q,bad,**kwargs)
        q.execute('ROLLBACK TO SAVEPOINT invalid')
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert q.fetchone()[0]==before


@pytest.mark.parametrize('state',['leased','failed'])
def test_source_round_move_during_lock_acquisition_fails_closed(database,state):
    lease,captured=gate.current_lease(database,error=None)
    lock=7523003
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        if state=='failed':assert complete(q,lease)['status']=='failed'
        q.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(lease['run_id'],));before=q.fetchone()[0]
        if state=='failed':assert complete(q,lease,token='sha256:'+'f'*64)['idempotent'] is True
        # Use a real target round: direct owner moves are possible, so the
        # completion must fence the lookup/locked-reread race itself.
        peer='arena-2099-10-20-peer'
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("SELECT string_agg(quote_ident(attname),',' ORDER BY attnum) FROM pg_attribute WHERE attrelid='public.lab_arena_rounds'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''")
        columns=q.fetchone()[0]
        q.execute("INSERT INTO public.lab_arena_rounds ("+columns+") SELECT "+columns+" FROM jsonb_populate_record(NULL::public.lab_arena_rounds,(SELECT to_jsonb(r)||jsonb_build_object('round_id',%s) FROM public.lab_arena_rounds r WHERE round_id=%s))",(peer,before['round_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        post=source(q,SIGNATURES['v3_lock_order'])
        marker='    WHERE source.run_id=p_run_id;';assert post.count(marker)==1
        q.execute(post.replace(marker,marker+'\n  PERFORM pg_advisory_xact_lock('+str(lock)+');',1))
    blocker=database[0].connect(**database[1]);pool=ThreadPoolExecutor(max_workers=1)
    def operation():
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute("SET LOCAL statement_timeout='8s'")
            try:complete(q,lease)
            except database[0].Error as exc:c.rollback();return exc.pgcode
    try:
        with blocker.cursor() as q:q.execute('SELECT pg_advisory_xact_lock(%s)',(lock,))
        pending=pool.submit(operation)
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            deadline=time.monotonic()+5
            while True:
                q.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND objid=%s AND NOT granted)",(lock,))
                if q.fetchone()[0]:break
                assert time.monotonic()<deadline
                time.sleep(.01)
            q.execute('UPDATE public.lab_arena_runs SET round_id=%s WHERE run_id=%s',(peer,lease['run_id']))
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');runs=q.fetchone()[0]
            q.execute("SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),'[]') FROM public.lab_arena_ledger l");charges=q.fetchone()[0]
            q.execute("SELECT coalesce(jsonb_agg(to_jsonb(j) ORDER BY cache_key),'[]') FROM public.lab_arena_company_judgments j");judgments=q.fetchone()[0]
        blocker.commit();assert pending.result(timeout=8)=='40001'
        with database[0].connect(**database[1]) as c,c.cursor() as q:
            q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert q.fetchone()[0]==runs
            q.execute("SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),'[]') FROM public.lab_arena_ledger l");assert q.fetchone()[0]==charges
            q.execute("SELECT coalesce(jsonb_agg(to_jsonb(j) ORDER BY cache_key),'[]') FROM public.lab_arena_company_judgments j");assert q.fetchone()[0]==judgments
    finally:
        blocker.rollback();blocker.close();pool.shutdown(wait=True)
        with database[0].connect(**database[1]) as c,c.cursor() as q:q.execute(post)


def test_accepted_terminal_replay_preserves_exact_judgment_and_charges(database):
    gate.queue.revision.test_old_accepted_parent_new_roots_complete_and_survive_later_close(database)
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        q.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='score' AND status='accepted' AND scoring_generation_floor>0 ORDER BY run_id LIMIT 1");row=q.fetchone()[0]
        q.execute('SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),\'[]\') FROM public.lab_arena_ledger l');charges=q.fetchone()[0]
        replay=complete(q,row,token='sha256:'+'f'*64)
        assert replay['status']=='accepted' and replay['idempotent'] is True
        q.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(row['run_id'],));assert q.fetchone()[0]==row
        q.execute('SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),\'[]\') FROM public.lab_arena_ledger l');assert q.fetchone()[0]==charges


def test_saved_outputs_still_accept_persist_and_publish(database,tmp_path,monkeypatch):
    gate.queue.revision.test_saved_outputs_regrade_through_service_persistence_and_publication(database,tmp_path,monkeypatch)


@pytest.mark.parametrize('label',list(SIGNATURES))
@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('drift',['owner','acl','secdef','volatility','path','source'])
def test_atomic_source_and_security_preflight_reject_drift(database,label,installed,drift):
    with database[0].connect(**database[1]) as c,c.cursor() as q:
        saved={k:source(q,sig) for k,sig in SIGNATURES.items()}
        if not installed:
            for k,d in saved.items():q.execute(preimage(k,d))
        before={k:source(q,sig) for k,sig in SIGNATURES.items()}
        q.execute('SAVEPOINT hostile');sig=SIGNATURES[label]
        if drift=='source':q.execute(source(q,sig).replace('BEGIN\n','BEGIN\n  -- unknown source drift\n',1))
        else:q.execute({'owner':'ALTER FUNCTION '+sig+' OWNER TO postgres','acl':'GRANT EXECUTE ON FUNCTION '+sig+' TO anon','secdef':'ALTER FUNCTION '+sig+' SECURITY INVOKER','volatility':'ALTER FUNCTION '+sig+' STABLE','path':'ALTER FUNCTION '+sig+' SET search_path=public,pg_catalog'}[drift])
        text=SQL.read_text().split("SET LOCAL statement_timeout='30s';",1)[1].removesuffix('COMMIT;\n')
        with pytest.raises(database[0].Error):q.execute(text)
        q.execute('ROLLBACK TO SAVEPOINT hostile')
        assert {k:source(q,s) for k,s in SIGNATURES.items()}==before
        for d in saved.values():q.execute(d)


@pytest.mark.parametrize('fault',['none','no_start','open_dispatch','unbound_error','setup_image','setup_runtime'])
def test_expiry_signal_preserves_full_recovery_authorization(database,fault):
    gate.test_signal_keeps_full_stopped_host_authorization(database,fault)


@pytest.mark.parametrize('kind',['execute','score'])
@pytest.mark.parametrize('head',['reservation','dispatch'])
def test_expiry_keeps_overdue_liability_and_retry(database,kind,head):
    gate.test_overdue_without_runtime_error_keeps_exact_liability_and_retry(database,kind,head)


@pytest.mark.parametrize('timing',['due','after_gate'])
def test_expiry_keeps_fresh_clock_boundary(database,timing):
    gate.test_natural_expiry_retains_fresh_clock_without_runtime_start(database,timing)
