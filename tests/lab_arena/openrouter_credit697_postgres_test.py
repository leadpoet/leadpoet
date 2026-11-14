"""Native account refusals stop unfunded retries without changing paid authority."""
from pathlib import Path
import hashlib
import pytest
from tests.lab_arena import funding_proof_keyed558_postgres_test as base
from tests.lab_arena import proven_funding_recovery524_postgres_test as prior
from tests.lab_arena import miner_credit_terminal484_postgres_test as terminal
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import reset  # noqa: F401

SQL=Path(__file__).parents[2]/'scripts/697-lab-arena-openrouter-credit-account-proof.sql'
POST='9800912044b2aba92b6e66f203e348a412f4c53b3753612b43573f1989370aec'
OLD="IF h.provider='deepline' AND evidence->>'reason'='out_of_credit'"
NEW="IF h.provider IN ('deepline','openrouter') AND evidence->>'reason'='out_of_credit'"

@pytest.fixture(scope='module')
def database():
    gen=base.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1])as c,c.cursor()as q:
            q.execute(SQL.read_text());q.execute(SQL.read_text())
            assert hashlib.sha256(base.base.definition(q).encode()).hexdigest()==POST
        yield db
    finally:gen.close()


def native_unknown(c,runid):
    with c.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_ledger SET provider='openrouter',operation_id='openrouter.chat',entry_doc=CASE WHEN entry_kind='settlement' THEN jsonb_build_object('call',jsonb_build_object('call_succeeded',false,'provider_status',402,'account_failure_evidence',terminal_response->'account_failure_evidence')) ELSE entry_doc END,terminal_response=CASE WHEN entry_kind='settlement' THEN NULL ELSE terminal_response END,entry_kind=CASE WHEN entry_kind='settlement' THEN 'uncertain' ELSE entry_kind END WHERE run_id=%s AND call_identity IN (SELECT call_identity FROM public.lab_arena_ledger WHERE run_id=%s AND coalesce(terminal_response#>>'{account_failure_evidence,reason}',entry_doc#>>'{call,account_failure_evidence,reason}')='out_of_credit')",(runid,runid))
        q.execute('SET LOCAL session_replication_role=origin')


def test_native_refusal_red_before_green_and_no_history_mutation(database):
    run,_=prior.seed(database)
    with database[0].connect(**database[1])as c,c.cursor()as q:
        native_unknown(c,run['run_id']);before=prior.snapshot(q)
        post=base.base.definition(q);q.execute(post.replace(NEW,OLD))
        assert not prior.proof(q,run['run_id'])
        q.execute(SQL.read_text())
        assert prior.proof(q,run['run_id'])
        assert prior.snapshot(q)==before
        q.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='settlement'",(run['run_id'],))
        assert q.fetchone()==(0,)


@pytest.mark.parametrize('fault',['host','binding','owner','untyped','mixed_provider_failure'])
def test_ambiguous_or_unbound_failures_remain_recoverable(database,fault):
    run,_=prior.seed(database)
    with database[0].connect(**database[1])as c,c.cursor()as q:
        native_unknown(c,run['run_id']);q.execute('SET LOCAL session_replication_role=replica')
        if fault=='host':q.execute("UPDATE public.lab_arena_ledger SET funding_source='host' WHERE run_id=%s",(run['run_id'],))
        if fault=='owner':q.execute("UPDATE public.lab_arena_ledger SET miner_hotkey='wrong-owner' WHERE run_id=%s",(run['run_id'],))
        if fault=='binding':q.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,account_failure_evidence,action_sequence}','9') WHERE run_id=%s AND entry_kind='uncertain'",(run['run_id'],))
        if fault=='untyped':q.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc#-'{call,account_failure_evidence,reason}' WHERE run_id=%s AND entry_kind='uncertain'",(run['run_id'],))
        q.execute('SET LOCAL session_replication_role=origin')
        if fault=='mixed_provider_failure':prior.call(c,run['run_id'],'other',provider='openrouter',outcome='transport')
        assert not prior.proof(q,run['run_id'])


def test_new_paid_same_key_recovers_but_different_key_does_not(database):
    run,_=prior.seed(database)
    with database[0].connect(**database[1])as c,c.cursor()as q:
        native_unknown(c,run['run_id']);assert prior.proof(q,run['run_id'])
        funded=prior.clone(c,run)
        prior.call(c,funded,'wrong-key',provider='openrouter',fingerprint='sha256:'+'d'*64)
        assert prior.proof(q,run['run_id'])
        prior.call(c,funded,'same-key',provider='openrouter')
        before=prior.snapshot(q)
        assert not prior.proof(q,run['run_id'])
        assert prior.snapshot(q)==before


def test_full_baseline_miner_scoring_and_publication_with_native_unknown_fee(database,tmp_path,monkeypatch):
    original=terminal.credit
    def native(c,run_id,**kw):
        original(c,run_id,**kw);native_unknown(c,run_id)
    monkeypatch.setattr(terminal,'credit',native)
    terminal.test_full_partial_publication_preserves_baseline_and_peer_authority(database,tmp_path,monkeypatch)


def test_native_http_refusal_persists_bound_reason_without_duplicate_spend(database,tmp_path):
    from dataclasses import replace
    from fastapi.testclient import TestClient
    from lab_arena import broker as br,runner
    from lab_arena.api import create_app
    from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round
    from tests.lab_arena.test_lab_arena_migration_postgres import claim
    from tests.lab_arena.test_lab_arena_service_round import Harness,keypair
    from tests.lab_arena.test_lab_arena_broker import CHAT,FakeTransport,HOST_KEYS,price_table
    from tests.lab_arena.openrouter_account_credit_test import REFUSAL
    connect=lambda:database[0].connect(**database[1])
    h=Harness(connect,tmp_path,challengers=['NativeCredit'],runners=['native-credit'])
    miner=keypair('svc-miner-NativeCredit').ss58_address
    h.chain.runners.append(miner);h.chain.permits[miner]=False
    h.service.config.defaults=replace(h.service.config.defaults,per_icp_cost_policy=True,integrity_from='2000-01-01T00:00:00Z')
    _start_parallel_round(h,'arena-2099-09-14',slot_ceiling=2)
    lease,token=claim(h.service.store,h.round_id,h.runner_keys[0],parallelism=2,ceiling=2,excluded=(h.baseline_hotkey,))[:2]
    transport=FakeTransport([(402,{**REFUSAL,'id':'gen-native-credit'})])
    broker=br.Broker(store=h.service.store,key_for=lambda p:HOST_KEYS[p],credential_for=lambda c,p:HOST_KEYS[p],funding_source_for=lambda c:'miner_key',price_table=price_table(),transport=transport,clock=h.clock)
    h.service._broker_for=lambda _:broker
    h.service._hot_rounds.clear()
    client=TestClient(create_app(h.service))
    try:
        api=runner.HttpArenaApiClient('http://127.0.0.1',client=client)
        frame={'operation_id':'openrouter.chat','parameters':CHAT,'action_sequence':0,'timeout_ms':5000}
        first=api.provider(lease['run_id'],token,frame)
        assert first['status']==402 and first['call']['outcome']=='uncertain'
        assert br.provider_credit_failure_reason(first['call'])=='provider_credit_exhausted'
        assert 'actual_microusd' not in first['call'] and 'credit_failure_proof' not in first['call']
        with connect()as c,c.cursor()as q:
            q.execute("SELECT entry_kind,entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id",(first['call']['call_identity'],))
            entries=q.fetchall()
            assert [x[0]for x in entries]==['reservation','dispatch','uncertain']
            assert entries[-1][1]['call']['account_failure_evidence']==first['call']['account_failure_evidence']
            q.execute("SELECT content#>>'{call,account_failure_evidence,reason}' FROM public.lab_arena_trajectory_events WHERE run_id=%s AND event_kind='provider.response'",(lease['run_id'],))
            assert q.fetchall()==[('out_of_credit',)]
        sent=len(transport.sent)
        replay=api.provider(lease['run_id'],token,frame)
        assert replay['status'] in (402,409)
        assert br.provider_credit_failure_reason(replay['call'])=='provider_credit_exhausted'
        assert len(transport.sent)==sent and sum(r['method']=='POST'for r in transport.sent)==1
    finally:client.close()


def test_migration_is_idempotent_preserves_acl_and_all_rows(database):
    prior.seed(database)
    with database[0].connect(**database[1])as c,c.cursor()as q:
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        functions=q.fetchall();rows=prior.snapshot(q)
        q.execute(SQL.read_text());q.execute(SQL.read_text())
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        assert q.fetchall()==functions and prior.snapshot(q)==rows

@pytest.mark.parametrize('fault',['source','owner','acl','secdef','path','volatility'])
def test_migration_rejects_changed_security_or_source_even_after_apply(database,fault):
    with database[0].connect(**database[1])as c,c.cursor()as q:
        post=base.base.definition(q);q.execute('SAVEPOINT guard')
        if fault=='source':q.execute(post.replace('BEGIN\n','BEGIN\n-- unexpected drift\n',1))
        else:q.execute({'owner':'ALTER FUNCTION '+prior.SIG+' OWNER TO postgres','acl':'GRANT EXECUTE ON FUNCTION '+prior.SIG+' TO anon','secdef':'ALTER FUNCTION '+prior.SIG+' SECURITY INVOKER','path':'ALTER FUNCTION '+prior.SIG+' SET search_path=public,pg_catalog','volatility':'ALTER FUNCTION '+prior.SIG+' VOLATILE'}[fault])
        body=SQL.read_text().split("SET LOCAL statement_timeout='30s';",1)[1].removesuffix('COMMIT;\n')
        with pytest.raises(database[0].Error):q.execute(body)
        q.execute('ROLLBACK TO SAVEPOINT guard')
