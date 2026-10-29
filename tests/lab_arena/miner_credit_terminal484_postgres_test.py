"""Proved miner funding failures end without invented scores or relaxed baselines."""
from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path

import pytest

from tests.lab_arena import general_judge_continuation472_postgres_test as prior
from tests.lab_arena import bounded_baseline_recovery470_postgres_test as base
from tests.lab_arena import infrastructure_retry_cost478_postgres_test as deployed

SQL=Path(__file__).parents[2]/'scripts/484-lab-arena-miner-credit-terminal-judgment.sql'

@pytest.fixture(scope='module')
def database():
    gen=deployed.database.__wrapped__(); db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            for number in (479,480,481,482):
                paths=list(SQL.parent.glob(f'{number}-*.sql')); assert len(paths)==1
                cur.execute(paths[0].read_text())
            cur.execute(SQL.read_text()); cur.execute(SQL.read_text())
        yield db
    finally: gen.close()


def seed(conn, *, future=False, baseline=False, funding='miner_key', proof=True, uncertain=False):
    assignment,source,cfg=prior.prepare(conn,cause='credential_error')
    cutoff=datetime.now(timezone.utc)+timedelta(hours=1 if future else -1)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring',configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}',%s::jsonb),participants=%s::jsonb WHERE round_id=%s",(json.dumps(cutoff.isoformat().replace('+00:00','Z')),json.dumps([{'submission_id':base.SUB,'miner_hotkey':base.MINER,'is_king':baseline}]+([] if baseline else [{'submission_id':'other-baseline','miner_hotkey':'other-owner','is_king':True}])),base.ROUND))
        cur.execute('UPDATE public.lab_arena_submissions SET is_king=%s WHERE submission_id=%s',(baseline,base.SUB))
        if not baseline:
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{baseline_hotkey}','\"other-owner\"') WHERE round_id=%s",(base.ROUND,))
        cur.execute('UPDATE public.lab_arena_runs SET stage_generation=8 WHERE round_id=%s',(base.ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    for attempt in (1,2): credit(conn,assignment+f':{attempt}',funding=funding,proof=proof,uncertain=uncertain)
    conn.commit()
    return assignment,source


def credit(conn,run_id,*,funding='miner_key',proof=True,uncertain=False,suffix='b',reason='out_of_credit'):
    identity='sha256:'+hashlib.sha256((run_id+suffix).encode()).hexdigest()
    call={'base_call_identity':'sha256:'+'a'*64,'provider_attempt':1,'action_sequence':0}
    evidence={'error_class':'account_credential_failure','reason':reason,'provider_status':402,**call}
    terminal={'status':402,'call_succeeded':False}
    if proof:
        terminal['credit_failure_proof']={'schema_version':'leadpoet.lab_arena.credit_failure_proof.v1','provider':'deepline','reason':'out_of_credit','provider_status':402,'actual_microusd':0}
        terminal['account_failure_evidence']=evidence
    doc={'call':{'call_succeeded':False,'provider_status':402,**({'account_failure_evidence':evidence} if proof else {})}}
    with conn.cursor() as cur:
        cur.execute('SELECT miner_hotkey,round_id,submission_id,stage FROM public.lab_arena_runs WHERE run_id=%s',(run_id,))
        owner,round_id,submission_id,stage=cur.fetchone()
        for kind,entry,response in [('reservation',call,None),('dispatch',call,None),('uncertain' if uncertain else 'settlement',doc if uncertain else {},None if uncertain else terminal)]:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) VALUES(%s,%s,%s,%s,%s,%s,%s,'deepline','deepline.execute',%s,0,%s::jsonb,%s::jsonb)",(kind,owner,round_id,submission_id,run_id,stage,identity,funding,json.dumps(entry),json.dumps(response) if response is not None else None))


def pending(conn,assignment,attempt):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scored_run_id,judgment_cache_key,judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,created_at) SELECT assignment_id||':'||%s,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,%s,kind,'pending',8,scored_run_id,judgment_cache_key,judgment_input_hash,judgment_scope_doc,true,judgment_group_miner_hotkeys,clock_timestamp()-interval '20 minutes' FROM public.lab_arena_runs WHERE run_id=%s",(attempt,attempt,assignment+':2'))
    conn.commit()


def state(conn,run_id):
    with conn.cursor() as cur:
        cur.execute('SELECT status,terminal_cause,per_icp_score FROM public.lab_arena_runs WHERE run_id=%s',(run_id,));return cur.fetchone()


def proved(conn,run_id):
    return base.rpc(conn,'lab_arena_miner_credit_failure_v1(%s)',(run_id,))


def terminal(conn,source):
    return base.rpc(conn,'lab_arena__miner_credit_judgment_terminal_v1(%s)',(source,))


@pytest.mark.parametrize('uncertain',[False,True])
def test_proved_credit_is_terminal_without_score_and_keeps_billing(database,uncertain):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,uncertain=uncertain)
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id');before=cur.fetchall()
        assert proved(conn,assignment+':2')
        assert terminal(conn,source)
        assert base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))['retried']==0
        with conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s),public.lab_arena__publication_execution_incomplete_v1(%s,%s)',(base.ROUND,base.ROUND,base.SUB))
            # Other missing ICPs make the full participant proof fail closed.
            assert cur.fetchone()==(True,False)
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id');assert cur.fetchall()==before
            cur.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)',(base.ROUND,));assert cur.fetchone()[0] is False
        assert state(conn,source)==('accepted','accepted',None)


@pytest.mark.parametrize('change',['baseline','host','missing_proof','wrong_owner','wrong_binding','wrong_reason','wrong_scope','mixed_infrastructure'])
def test_negative_proof_controls_remain_recoverable(database,change):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,baseline=change=='baseline',funding='host' if change=='host' else 'miner_key',proof=change!='missing_proof',uncertain=True)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if change=='wrong_owner':
                cur.execute('UPDATE public.lab_arena_ledger SET miner_hotkey=%s WHERE run_id=%s',('wrong-owner',assignment+':2'))
            elif change=='wrong_binding':
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,account_failure_evidence,action_sequence}','9') WHERE run_id=%s AND entry_kind='uncertain'",(assignment+':2',))
            elif change=='wrong_reason':
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,account_failure_evidence,reason}','\"managed_provider_failure\"') WHERE run_id=%s AND entry_kind='uncertain'",(assignment+':2',))
            elif change=='wrong_scope':
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc=jsonb_set(judgment_scope_doc,'{scorer_image_digest}','\"wrong\"') WHERE run_id=%s",(assignment+':2',))
                cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SET LOCAL session_replication_role=origin')
        if change=='mixed_infrastructure': credit(conn,assignment+':2',proof=False,uncertain=True,suffix='d')
        conn.commit()
        assert not proved(conn,assignment+':2')
        assert not terminal(conn,source)
        assert base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))['retried']==(0 if change=='wrong_scope' else 1)


@pytest.mark.parametrize('attempt',[3,4,9])
def test_existing_above_cap_successor_never_claims_or_recontinues(database,attempt):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,future=True)
        before=state(conn,assignment+':2')
        pending(conn,assignment,attempt)
        assert proved(conn,assignment+f':{attempt}')
        assert base.claim(conn)=={'status':'no_pending'}
        result=base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))
        assert result['retried']==0
        assert state(conn,assignment+f':{attempt}')[:2]==('failed','stage_closed')
        assert state(conn,assignment+':2')==before
        assert not terminal(conn,source) # frozen owner window stays open
        assert base.rpc(conn,'lab_arena_close_scoring(%s,1::smallint)',(base.ROUND,))['status']=='draining'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s",(base.ROUND,))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert terminal(conn,source)
        assert base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))['retried']==0


def test_live_successor_lease_is_preserved_and_new_infra_failure_recovers(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,future=True)
        pending(conn,assignment,3)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s",(base.RUNNER,base.TOKEN,assignment+':3'))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert not proved(conn,assignment+':3')
        assert not terminal(conn,source)
        assert base.rpc(conn,'lab_arena_close_scoring(%s,1::smallint)',(base.ROUND,))['status']=='draining'
        assert state(conn,assignment+':3')[0]=='leased'
        result=base.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(assignment+':3',base.TOKEN,json.dumps({'terminal_status':'judge_error'}),'judge_error',''))
        assert result['status']=='failed' and result['confirmation_attempt']==4
        assert not proved(conn,assignment+':3')
        assert state(conn,assignment+':4')[0]=='pending'
        assert not terminal(conn,source)


def test_new_credit_completion_stops_continuation(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,future=True)
        pending(conn,assignment,3)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s",(base.RUNNER,base.TOKEN,assignment+':3'))
            cur.execute('SET LOCAL session_replication_role=origin')
        credit(conn,assignment+':3',uncertain=True);conn.commit()
        result=base.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(assignment+':3',base.TOKEN,json.dumps({'terminal_status':'credential_error'}),'credential_error',''))
        assert result['status']=='failed' and 'confirmation_attempt' not in result
        assert proved(conn,assignment+':3')
        assert base.rpc(conn,'lab_arena_expire_leases(%s)',(base.ROUND,))['retried']==0


def test_owner_signed_retry_keeps_original_settled_zero_and_deadline_guards(database):
    from tests.lab_arena.credit_failure_retry403_postgres_test import REQUEST
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,future=True)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_ledger WHERE run_id=%s',(assignment+':2',))
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',(assignment+':2',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert base.rpc(conn,'lab_arena_close_scoring(%s,1::smallint)',(base.ROUND,))['status']=='draining'
        result=base.rpc(conn,'lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)',(base.ROUND,base.SUB,base.MINER,REQUEST))
        assert result=={'status':'queued','requeued_count':1}
        assert base.rpc(conn,'lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)',(base.ROUND,base.SUB,base.MINER,REQUEST))=={'status':'replayed','requeued_count':1}
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 minute' WHERE run_id=%s",(assignment+':2',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        lease=base.claim(conn)
        assert lease['status']=='leased' and lease['attempt']==2
        assert base.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],base.TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','accepted-proof'))['status']=='accepted'
        assert not terminal(conn,source)
        assert not proved(conn,lease['run_id'])
        with conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(base.ROUND,));assert cur.fetchone()[0] is False
        assert base.rpc(conn,'lab_arena_close_scoring(%s,1::smallint)',(base.ROUND,))['status']=='closed'
        # Accepted evidence still needs a real measured score, including zero.
        assert base.rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',(base.ROUND,json.dumps([{'run_id':source,'per_icp_score':0,'qualification_doc':{'companies':[]}}])))['status']=='ok'
        with conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(base.ROUND,));assert cur.fetchone()[0] is True


def test_full_partial_publication_preserves_baseline_and_peer_authority(database,tmp_path,monkeypatch):
    """Real service output, SQL publication triggers, then final baseline guard."""
    from dataclasses import replace
    from tests.lab_arena import test_lab_arena_service_round as fixtures
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    pg,dsn=database
    harness=IntegrityHarness(lambda:pg.connect(**dsn),tmp_path,challengers=['CreditMiner'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,execution_sequence_from='2000-01-01T00:00:00Z',per_icp_cost_policy=True)
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(fixtures,'deterministic_scorer',_baseline_first_judge)
    assert fixtures._start_round(harness,day=27,epoch=62427)==2
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('stage1_scored',runners=1)
    harness.service.advance_round(harness.round_id)
    harness.advance_until('scored',runners=1)
    row=harness.service.store.get_round(harness.round_id)
    miner=next(p for p in row['participants'] if not p['is_king'])
    baseline=next(p for p in row['participants'] if p['is_king'])
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        # The current round already enrolls in continuation. Keep its original
        # evaluation date, which is bound into the accepted judge-cache scopes.
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 minute')::text)) WHERE round_id=%s",(harness.round_id,))
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE submission_id=%s AND kind='execute' AND status='accepted' ORDER BY icp_position LIMIT 1",(miner['submission_id'],));source=cur.fetchone()[0]
        cur.execute("SELECT run_id,judgment_cache_key FROM public.lab_arena_runs WHERE scored_run_id=%s AND kind='score' AND status='accepted'",(source,));judge,cache=cur.fetchone()
        cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s',(source,))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}',output_ref=NULL WHERE run_id=%s",(judge,))
        cur.execute('DELETE FROM public.lab_arena_judgment_cache WHERE cache_key=%s',(cache,))
        cur.execute('SET LOCAL session_replication_role=origin')
        credit(conn,judge,uncertain=True)
    harness.service._invalidate_hot_round()
    assert harness.service.store.miner_credit_judgment_terminal(source)
    row=harness.service.store.get_round(harness.round_id)
    assert not harness.service._unfinished_judgments(row)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__publication_execution_incomplete_v1(%s,%s),public.lab_arena__baseline_final_ready_v1(%s)',(harness.round_id,miner['submission_id'],harness.round_id));assert cur.fetchone()==(True,True)
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",(harness.round_id,))
        cur.execute('SET LOCAL session_replication_role=origin')
    assert harness.service.score_stage(harness.round_id,2)['status']=='ok'
    # Current publication waits for the frozen publication deadline before
    # omitting an incomplete miner; the complete baseline remains required.
    assert harness.service.publish(harness.round_id) == {
        'status': 'retry', 'round_status': 'scored',
        'reason': 'execution_infrastructure_incomplete',
    }
    harness.clock.advance_to(harness.schedule()['publication_deadline'])
    publication = harness.service.publish(harness.round_id)
    assert publication['status']=='ok', publication
    published=harness.service.store.get_round(harness.round_id)
    ranking=next(r for r in published['publication_doc']['final_ranking'] if r['submission_id']==miner['submission_id'])
    assert ranking['final_score'] is None and ranking['eligible'] is False and ranking['eligibility_reason']=='execution_incomplete'
    assert harness.service.store.get_run(source)['per_icp_score'] is None
    # A missing baseline score must still reject exactly the same publication.
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='scored',publication_doc=NULL,published_at=NULL WHERE round_id=%s",(harness.round_id,))
        cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE submission_id=%s AND kind='execute' AND status='accepted' ORDER BY icp_position LIMIT 1)",(baseline['submission_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        with pytest.raises(pg.Error,match='lab_arena_final_baseline_incomplete'):
            cur.execute("UPDATE public.lab_arena_rounds SET status='published',publication_doc=%s::jsonb,published_at=%s WHERE round_id=%s",(json.dumps(published['publication_doc']),published['published_at'],harness.round_id))


def extra_call(conn,run_id,tag,*,status=503,succeeded=False,attempt=1,base_identity='d',uncertain=False):
    identity='sha256:'+hashlib.sha256((run_id+tag).encode()).hexdigest()
    call={'base_call_identity':'sha256:'+base_identity*64,'provider_attempt':attempt,'action_sequence':0}
    with conn.cursor() as cur:
        cur.execute('SELECT miner_hotkey,round_id,submission_id,stage FROM public.lab_arena_runs WHERE run_id=%s',(run_id,))
        owner,round_id,submission_id,stage=cur.fetchone()
        for kind,doc,response in [('reservation',call,None),('dispatch',call,None),('uncertain' if uncertain else 'settlement',{'call':{'call_succeeded':succeeded,'provider_status':status}} if uncertain else {},None if uncertain else {'status':status,'call_succeeded':succeeded})]:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) VALUES(%s,%s,%s,%s,%s,%s,%s,'deepline','deepline.execute','miner_key',0,%s::jsonb,%s::jsonb)",(kind,owner,round_id,submission_id,run_id,stage,identity,json.dumps(doc),json.dumps(response) if response else None))
    conn.commit()


@pytest.mark.parametrize('status',[404,422])
def test_ordinary_terminal_request_failure_does_not_require_replay_credit_proof(database,status):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,uncertain=True)
        extra_call(conn,assignment+':2','ordinary',status=status)
        assert proved(conn,assignment+':2') and terminal(conn,source)


def test_exact_request_recovery_does_not_collapse_different_targets(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,uncertain=True)
        extra_call(conn,assignment+':2','transport',uncertain=True)
        assert not proved(conn,assignment+':2')
        extra_call(conn,assignment+':2','different-target',status=200,succeeded=True,attempt=2,base_identity='e')
        assert not proved(conn,assignment+':2')
        extra_call(conn,assignment+':2','same-request-success',status=200,succeeded=True,attempt=2)
        assert proved(conn,assignment+':2')
        # A later authenticated success for the exact exhausted-credit request
        # clears that credit proof. Old receipt history cannot reject recovery.
        extra_call(conn,assignment+':2','restored-credit',status=200,succeeded=True,attempt=2,base_identity='a')
        assert not proved(conn,assignment+':2') and not terminal(conn,source)


def test_intervening_dispatched_evidence_cannot_inherit_prior_credit(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn)
        pending(conn,assignment,3)
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc='{\"previous_status\":\"pending\"}' WHERE run_id=%s",(assignment+':3',))
        extra_call(conn,assignment+':3','intervening',uncertain=True)
        pending(conn,assignment,4)
        assert not proved(conn,assignment+':4')
        assert not terminal(conn,source)


def test_live_24_head_shape_keeps_uncertain_fees_and_actual_spend(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,uncertain=True)
        for n in range(5): credit(conn,assignment+':2',uncertain=True,suffix=f'uncertain-{n}')
        for n in range(12): credit(conn,assignment+':2',suffix=f'zero-credit-{n}')
        for n in range(6):
            extra_call(conn,assignment+':2',f'openrouter-{n}',status=200,succeeded=True,base_identity='e')
            identity='sha256:'+hashlib.sha256((assignment+':2'+f'openrouter-{n}').encode()).hexdigest()
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_ledger SET provider='openrouter',operation_id='openrouter.chat',amount_microusd=CASE WHEN entry_kind='settlement' THEN %s ELSE 0 END WHERE run_id=%s AND call_identity=%s",(1404 if n<5 else 1406,assignment+':2',identity))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),count(*) FILTER(WHERE entry_kind='uncertain'),sum(amount_microusd) FILTER(WHERE entry_kind='settlement') FROM (SELECT DISTINCT ON(call_identity) * FROM public.lab_arena_ledger WHERE run_id=%s ORDER BY call_identity,entry_id DESC) h",(assignment+':2',));assert cur.fetchone()==(24,6,8426)
        assert proved(conn,assignment+':2') and terminal(conn,source)


@pytest.mark.parametrize('change',['uncertain_cost','charged_refusal','deadline','wrong_owner'])
def test_owner_retry_does_not_gain_billing_or_stage_authority(database,change):
    from tests.lab_arena.credit_failure_retry403_postgres_test import REQUEST
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn,future=change!='deadline',uncertain=change=='uncertain_cost')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_ledger WHERE run_id=%s',(assignment+':2',))
            cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',(assignment+':2',))
            if change=='charged_refusal':
                cur.execute("UPDATE public.lab_arena_ledger SET amount_microusd=7 WHERE run_id=%s AND entry_kind='settlement'",(assignment+':1',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        owner='wrong-owner' if change=='wrong_owner' else base.MINER
        result=base.rpc(conn,'lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)',(base.ROUND,base.SUB,owner,REQUEST))
        assert result['status']=='no_eligible'


def test_existing_accepted_cache_retains_normal_recovery_authority(database):
    with database[0].connect(**database[1]) as conn:
        assignment,source=seed(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_judgment_cache(cache_key,scope_doc,scoring_input_hash,evidence_hash,evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey) SELECT judgment_cache_key,judgment_scope_doc,judgment_input_hash,'sha256:'||repeat('c',64),jsonb_build_object('cache_key',judgment_cache_key,'scoring_input_hash',judgment_input_hash,'source_score_run_id',run_id,'source_scored_run_id',scored_run_id,'source_runner_hotkey',%s,'runner_authority_exclusions',jsonb_build_array(%s)),run_id,scored_run_id,%s FROM public.lab_arena_runs WHERE run_id=%s",(base.RUNNER,base.RUNNER,base.RUNNER,assignment+':2'))
        conn.commit()
        assert not proved(conn,assignment+':2') and not terminal(conn,source)


def test_new_proof_helpers_remain_service_only(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        for signature in ('public.lab_arena_miner_credit_failure_v1(text)',
                          'public.lab_arena__miner_credit_judgment_terminal_v1(text)'):
            cur.execute("SELECT pg_get_userbyid(proowner),prosecdef,has_function_privilege('lab_arena_service',oid,'EXECUTE'),has_function_privilege('anon',oid,'EXECUTE'),has_function_privilege('authenticated',oid,'EXECUTE'),has_function_privilege('service_role',oid,'EXECUTE') FROM pg_proc WHERE oid=%s::regprocedure",(signature,))
            assert cur.fetchone()==('lab_arena_owner',True,True,False,False,False)
