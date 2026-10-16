"""Reservation-only closeout preserves dispatch ambiguity and company evidence."""
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from lab_arena import broker as br, contracts, scoring
from lab_arena.store import ArenaStore, ArenaStoreUnavailable, PsycopgTransport, hash_lease_token
from lab_arena.service import ServiceError
from tests.lab_arena import progression_lock_order490_postgres_test as prior
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import reserve
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import setup, frame
from tests.lab_arena.deepline_completed_response_recovery_test import catalog
from tests.lab_arena.partial_company_judgments_postgres_test import _setup, _request, _sign
from tests.lab_arena.company_judgments_postgres_test import _complete

ROOT = Path(__file__).parents[2]
SQL = ROOT/'scripts/493-lab-arena-completion-reservation-closeout.sql'
PREIMAGE = ROOT/'tests/fixtures/arena_completion493_historical_function.sql'
SIGNATURE = 'public.lab_arena_complete_attempt(text,text,jsonb,text,text)'
PRE = 'dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b'
POST = '084e1afa3a4c277427d05c5ca509f235ff5814c7ecdae1b9dd29c39ec6f56073'

@pytest.fixture(scope='module')
def database():
    generator=prior.database.__wrapped__();db=next(generator)
    try:
        assert db[1]['host']=='127.0.0.1'
        yield db
    finally:generator.close()

@pytest.fixture()
def resources(database):
    connect=lambda:database[0].connect(**database[1])
    with connect() as c,c.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute(PREIMAGE.read_text())
    store=ArenaStore(PsycopgTransport(connect))
    yield store,connect
    store.close()


def apply(connect):
    with connect() as c,c.cursor() as cur:cur.execute(SQL.read_text())


def snapshot(connect):
    with connect() as c,c.cursor() as cur:
        cur.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
            "FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        return {name:(source,metadata) for name,source,metadata in cur.fetchall()}


def history(connect,identity):
    with connect() as c,c.cursor() as cur:
        cur.execute('SELECT entry_id,entry_kind,amount_microusd,entry_doc,terminal_response '
            'FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id',(identity,))
        return cur.fetchall()


def complete(cur,lease,token,*,cause='accepted',key=None):
    cur.execute('SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
        (lease['run_id'],key or hash_lease_token(token),json.dumps({'terminal_status':cause}),
         cause,'hermetic/output.json' if cause=='accepted' else ''))
    return cur.fetchone()[0]


class NoNetwork:
    requests=0
    def send(self,**kwargs):
        self.requests+=1
        pytest.fail('store failure must prevent provider dispatch')


def execution(database,tmp_path,label='30'):
    transport=NoNetwork()
    h,lease,token,connect,broker=setup(database,tmp_path,label,transport)
    context=br.RunContext(run_id=lease['run_id'],assignment_id=lease['assignment_id'],
        icp_position=lease['icp_position'],lease_token_hash=hash_lease_token(token),
        miner_hotkey=lease['miner_hotkey'],submission_id=lease['submission_id'],
        stage=lease['stage'],kind='execute',attempt=lease['attempt'],round_id=h.round_id,
        deepline_catalog=catalog())
    return h,lease,token,connect,broker,context,transport


def test_exact_migration_preserves_metadata_and_only_changes_completion(resources,database):
    _,connect=resources
    before=snapshot(connect)
    assert hashlib.sha256(before[SIGNATURE.removeprefix('public.')][0].encode()).hexdigest()==PRE
    with connect() as c,c.cursor() as cur:
        cur.execute('CREATE ROLE completion493_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO completion493_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO completion493_migrator')
        cur.execute('SET ROLE completion493_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone()==(False,)
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
    after=snapshot(connect)
    assert after.keys()==before.keys()
    for name in before:
        assert before[name][1]==after[name][1]
        if name!=SIGNATURE.removeprefix('public.'):assert before[name][0]==after[name][0]
    assert hashlib.sha256(after[SIGNATURE.removeprefix('public.')][0].encode()).hexdigest()==POST


@pytest.mark.parametrize('bad',['preimage','postimage','security'])
def test_migration_fails_closed(resources,database,bad):
    _,connect=resources;before=snapshot(connect)
    with connect() as c,c.cursor() as cur:
        if bad=='security':
            cur.execute('GRANT EXECUTE ON FUNCTION '+SIGNATURE+' TO PUBLIC')
            sql=SQL.read_text()
        else:sql=SQL.read_text().replace(PRE if bad=='preimage' else POST,'0'*64,1)
        with pytest.raises(database[0].Error,match='completion493 .* differs'):cur.execute(sql)
        c.rollback()
    assert snapshot(connect)==before


@pytest.mark.parametrize('marker_committed',[False,True])
def test_actual_broker_marker_failure_and_completion(resources,database,tmp_path,monkeypatch,marker_committed):
    _,connect=resources
    h,lease,token,_,broker,context,net=execution(database,tmp_path)
    original=h.service.store.mark_dispatched
    def unavailable(**kwargs):
        if marker_committed:original(**kwargs)
        raise ArenaStoreUnavailable('hermetic mark_dispatched acknowledgment failure')
    monkeypatch.setattr(h.service.store,'mark_dispatched',unavailable)
    with pytest.raises(ArenaStoreUnavailable):broker.execute(context,**frame())
    assert net.requests==0
    ledger=h.service.store.list_ledger(run_id=lease['run_id'])
    identity=ledger[0]['call_identity'];before=history(connect,identity)
    assert [r[1] for r in before]==(['reservation','dispatch'] if marker_committed else ['reservation'])
    with connect() as c,c.cursor() as cur:assert complete(cur,lease,token)=={'status':'accounting_open','open_calls':1}
    apply(connect)
    with connect() as c,c.cursor() as cur:result=complete(cur,lease,token)
    after=history(connect,identity)
    assert after[:len(before)]==before
    if marker_committed:
        assert result=={'status':'accounting_open','open_calls':1} and after==before
    else:
        assert result['status']=='accepted'
        assert [r[1] for r in after]==['reservation','recovery'] and after[-1][2]==0
        with connect() as c,c.cursor() as cur:assert complete(cur,lease,token)['idempotent'] is True
        assert history(connect,identity)==after
        # A delayed same-run marker cannot authorize a provider request.
        assert original(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=identity)['status']=='stale'
    assert net.requests==0


@pytest.mark.parametrize('invalid',['token','stale','cause','result'])
def test_invalid_completion_cannot_release_reservation(resources,database,tmp_path,invalid):
    _,connect=resources;h,lease,token,_,_,_,_=execution(database,tmp_path)
    identity,_=reserve(h,lease,token,'invalid');before=history(connect,identity);apply(connect)
    with connect() as c,c.cursor() as cur:
        if invalid=='stale':
            cur.execute('UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval \'1 second\' WHERE run_id=%s',(lease['run_id'],))
            c.commit()
        if invalid in ('cause','result'):
            cause='judge_error' if invalid=='cause' else 'accepted'
            result={'terminal_status':'accepted' if invalid=='result' else cause}
            if invalid=='result':result={}
            with pytest.raises(database[0].Error):
                cur.execute('SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],hash_lease_token(token),json.dumps(result),cause,'hermetic/output.json' if cause=='accepted' else ''))
            c.rollback()
        else:assert complete(cur,lease,token,key=hash_lease_token('wrong') if invalid=='token' else None)['status']=='stale'
    assert history(connect,identity)==before


@pytest.mark.parametrize('marker_first',[False,True])
def test_concurrent_completion_and_marker_use_existing_run_lock(resources,database,tmp_path,marker_first):
    _,connect=resources;h,lease,token,_,_,_,_=execution(database,tmp_path);apply(connect)
    identity,_=reserve(h,lease,token,'race');c1,c2=connect(),connect()
    ready=threading.Event();result={}
    def marker(cur):
        cur.execute('SELECT public.lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],hash_lease_token(token),identity))
        return cur.fetchone()[0]
    def second():
        try:
            with c2.cursor() as cur:
                cur.execute('SELECT pg_backend_pid()');result['pid']=cur.fetchone()[0];ready.set()
                result['response']=complete(cur,lease,token) if marker_first else marker(cur)
            c2.commit()
        except BaseException as exc:result['error']=repr(exc)
        finally:c2.close()
    with c1.cursor() as cur:first=marker(cur) if marker_first else complete(cur,lease,token)
    thread=threading.Thread(target=second);thread.start();assert ready.wait(2)
    blocked=False
    for _ in range(100):
        with connect() as c,c.cursor() as cur:
            cur.execute('SELECT wait_event_type FROM pg_stat_activity WHERE pid=%s',(result['pid'],));row=cur.fetchone()
        if row and row[0]=='Lock':blocked=True;break
        time.sleep(.01)
    c1.commit();c1.close();thread.join(3)
    assert blocked and not thread.is_alive() and 'error' not in result,result
    kinds=[r[1] for r in history(connect,identity)]
    if marker_first:
        assert first['status']=='dispatched' and result['response']=={'status':'accounting_open','open_calls':1}
        assert kinds==['reservation','dispatch']
    else:
        assert first['status']=='accepted' and result['response']['status']=='stale'
        assert kinds==['reservation','recovery']


@pytest.mark.parametrize('terminal',['accepted','judge_error','invalid_evidence'])
def test_v3_completion_retains_company_evidence_and_paid_sibling(resources,database,terminal,monkeypatch):
    store,connect=resources;apply(connect)
    # The shared company fixture predates251. Seed its immutable legacy bank
    # columns locally without restoring the retired RPC on the modern schema.
    from tests.lab_arena import company_judgments_postgres_test as company_fixture
    def historical_bank(_store,rid,ref,digest):
        with connect() as c,c.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET confirmation_bank_ref=%s,confirmation_bank_hash=%s WHERE round_id=%s',(ref,digest,rid))
        return {'status':'ok'}
    monkeypatch.setattr(company_fixture,'prepare_historical_confirmation_bank',historical_bank)
    from datetime import datetime,timedelta,timezone
    original_config=company_fixture.round_config
    def current_schedule(*args,**kwargs):
        config=original_config(*args,**kwargs)
        for hour,name in enumerate(('stage_1_close','stage_1_scoring_close','stage_2_close','final_scoring_close'),1):
            config['schedule'][name]=(datetime.now(timezone.utc)+timedelta(hours=hour)).isoformat()
        return config
    monkeypatch.setattr(company_fixture,'round_config',current_schedule)
    service,rid,key,leased,token,refs=_setup(store,'judge_error')
    # Two exact provider calls: one saved positive fee and one never dispatched.
    def reserve_call(seq):
        identity=contracts.document_hash({'493':seq})
        result=store.reserve_call(run_id=leased['run_id'],lease_token_hash=hash_lease_token(token),
            call_identity=identity,operation_id='openrouter.chat',provider='openrouter',funding_source='miner_key',
            amount_microusd=0,call_doc={'action_sequence':seq})
        assert result['status']=='reserved';return identity
    paid=reserve_call(1)
    store.mark_dispatched(run_id=leased['run_id'],lease_token_hash=hash_lease_token(token),call_identity=paid)
    store.settle_call(run_id=leased['run_id'],lease_token_hash=hash_lease_token(token),call_identity=paid,
        actual_microusd=1234,terminal_response={'status':200,'call_succeeded':True})
    paid_before=history(connect,paid);undispatched=reserve_call(2)
    request=_request(rid,key,leased,token,terminal='judge_error')
    if terminal=='invalid_evidence':
        before=history(connect,undispatched)
        body=json.loads(json.dumps(request['body']))
        body['output']['company_judgments'][0]['company_input_hash']='sha256:'+'b'*64
        with pytest.raises(ServiceError):service.handle_complete(_sign(rid,key,body))
        assert history(connect,undispatched)==before and history(connect,paid)==paid_before
        assert store.get_run(leased['run_id'])['status']=='leased'
        assert all(store.get_company_judgment(ref['cache_key'],0) is None for ref in refs)
        return
    if terminal=='accepted':
        # Actual complete_attempt_v3 SQL path with all bound company evidence.
        response=_complete(store,leased,token,refs=refs)
    else:
        # Failed partial evidence also traverses signed API/service validation.
        response=service.handle_complete(request)
    assert response['status']==('accepted' if terminal=='accepted' else 'failed')
    expected=3 if terminal=='accepted' else 2
    assert response['company_judgments_stored']==expected
    persisted=store.get_run(leased['run_id']);assert persisted['result_doc']['terminal_status']==terminal
    assert persisted['output_ref']
    for ref in refs[:expected]:assert store.get_company_judgment(ref['cache_key'],0) is not None
    if terminal!='accepted':assert store.get_company_judgment(refs[-1]['cache_key'],0) is None
    assert history(connect,paid)==paid_before
    recovered=history(connect,undispatched);assert [r[1] for r in recovered]==['reservation','recovery']
    replay=(_complete(store,leased,token,refs=refs) if terminal=='accepted' else service.handle_complete(request))
    assert replay['idempotent'] is True
    assert history(connect,paid)==paid_before and history(connect,undispatched)==recovered
    assert store.get_run(leased['run_id'])['result_doc']==persisted['result_doc']


@pytest.mark.parametrize('other_head',['dispatch','uncertain'])
def test_closeout_preserves_other_open_or_uncertain_heads(resources,database,tmp_path,other_head):
    _,connect=resources;h,lease,token,_,_,_,_=execution(database,tmp_path);apply(connect)
    first,_=reserve(h,lease,token,'other-head')
    h.service.store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=first)
    if other_head=='uncertain':
        h.service.store.mark_uncertain(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),
            call_identity=first,call_doc={'call_succeeded':False,'reason':'hermetic_transport_failure'})
    other_before=history(connect,first)
    second,_=reserve(h,lease,token,'undispatched')
    with connect() as c,c.cursor() as cur:response=complete(cur,lease,token)
    assert history(connect,first)==other_before
    assert [r[1] for r in history(connect,second)]==['reservation','recovery']
    late=h.service.store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=second)
    if other_head=='dispatch':
        assert response=={'status':'accounting_open','open_calls':1}
        assert late['status']=='recovered'
    else:
        # Existing completion semantics permit an explicitly unsuccessful
        # uncertainty; its unknown fee/document is retained without repricing.
        assert response['status']=='accepted' and late['status']=='stale'
    assert history(connect,first)==other_before
