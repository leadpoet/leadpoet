"""Owner-only finite external-capacity recovery uses666's existing pipeline."""
from pathlib import Path
import base64
import hashlib
import json
import pytest
from tests.lab_arena import operator_scorer_bug_recovery666_postgres_test as base

SQL = Path(__file__).parents[2]/'scripts/671-lab-arena-owner-provider-fault-recovery.sql'
database=base.database
prepared=base.prepared
prepared_company=base.prepared_company
case=base.case
SIG='lab_arena_grant_provider_fault_recovery_v1(%s,%s,%s,%s,%s,%s,%s,%s)'
HELPERS=[f'public.lab_arena__operator_scorer_bug_{n}_v1('+('public.lab_arena_rounds,public.lab_arena_runs'if n=='claim'else'public.lab_arena_runs')+')' for n in ['child','host','claim']]
OLDGRANT='public.lab_arena_grant_scorer_bug_recovery_v1(text,text,text,text,text,text,boolean)'

@pytest.fixture
def extended(case):
    p=case
    with p.c.cursor()as q:
        original=[base.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))for sig in HELPERS+[OLDGRANT]]
        q.execute(SQL.read_text())
    p.c.commit()
    try:yield p
    finally:
        p.c.rollback()
        with p.c.cursor()as q:
            for d in original:q.execute(d)
            q.execute('DROP FUNCTION public.lab_arena_grant_provider_fault_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,BOOLEAN)')
            q.execute('DROP FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN,TEXT,BIGINT)')
        p.c.commit()

@pytest.fixture
def provider_case(extended):
    p=extended
    receipt={'error':{'code':429,'message':'Provider returned error','metadata':{'is_byok':False,'limit_source':'upstream_provider_shared_pool'}}}
    p.receipt=receipt
    with p.c.cursor()as q:
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{failure_diagnostic,reason}','\"provider_error\"'),reviewer_runtime_doc=%s::jsonb WHERE run_id=%s",(json.dumps({'runtime_image_digest':base.RUNTIME}),p.parent['run_id']))
        cid='sha256:'+'9'*64
        request_hash='sha256:'+'4'*64
        raw=json.dumps(receipt).encode()
        for kind in ['reservation','dispatch','uncertain']:
            entry={'call':{'call_succeeded':False,'upstream_response_receipt':{'status':429,'body_b64':base64.b64encode(raw).decode(),'body_sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),'call_identity':cid,'operation_id':'openrouter.chat','request_hash':request_hash}}}if kind=='uncertain'else{'request_hash':request_hash}
            base.old.insert(q,'lab_arena_ledger',dict(entry_kind=kind,run_id=p.parent['run_id'],round_id=base.old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],stage=2,call_identity=cid,provider='openrouter',operation_id='openrouter.chat',funding_source='host'if p.organizer else'miner_key',amount_microusd=0,entry_doc=entry))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        p.proof=base.old.scalar(q,"SELECT max(entry_id)FROM lab_arena_ledger WHERE run_id=%s",(p.parent['run_id'],))
        p.parent=base.old.scalar(q,'SELECT to_jsonb(r)FROM lab_arena_runs r WHERE run_id=%s',(p.parent['run_id'],))
        p.parent_hash='sha256:'+base.old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s",(p.parent['run_id'],))
    p.c.commit()
    yield p

def grant(p,**changes):
    args=dict(parent=p.parent['run_id'],sha=p.parent_hash,runtime=base.RUNTIME,request='provider-fault-test-grant',actor='fixture-owner',proof_ref='evidence:retained-provider-fault',entry=p.proof,dry=False)
    args.update(changes)
    with p.c.cursor()as q:result=base.old.scalar(q,'SELECT public.'+SIG,tuple(args.values()))
    p.c.commit();return result


def test_missing_entrypoint_before_migration(case):
    with case.c.cursor()as q:assert base.old.scalar(q,"SELECT to_regprocedure('public.lab_arena_grant_provider_fault_recovery_v1(text,text,text,text,text,text,bigint,boolean)')")is None


def test_provider_dry_run_single_append_replay_preserves_state(provider_case):
    p=provider_case;base.old.release_guard(p);before=base.snapshot(p)
    preview=grant(p,dry=True);assert preview['status']=='dry_run';assert base.snapshot(p)==before
    result=grant(p);assert result['child_identity']==preview['child_identity'];assert len(result['checkpoints'])==2
    after=base.snapshot(p);assert len(after['lab_arena_runs'])==len(before['lab_arena_runs'])+1
    originals={r['run_id']:r for r in before['lab_arena_runs']};assert {r['run_id']:r for r in after['lab_arena_runs']if r['run_id']in originals}==originals
    for t in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:assert before[t]==after[t]
    assert grant(p)['idempotent']is True;assert base.snapshot(p)==after
    child=result['child_identity']['run_id'];assert base.old.funding(p,child)['funding_source']==('host'if p.organizer else'miner_key')
    journal=after['lab_arena_scoring_supersessions'][-1]['recovery_doc'];assert journal['recovery_kind']=='operator_provider_fault_recovery_v1';assert journal['provider_entry_id']==p.proof

@pytest.mark.parametrize('fault',['proof_missing','runtime_changed','stale_hash','accepted','later','calls_pending','expired','published','wrong_diagnostic'])
def test_provider_invalid_grant_fails_closed(provider_case,fault):
    p=provider_case;changes={}
    if fault=='proof_missing':changes['entry']=p.proof+999999
    elif fault=='runtime_changed':changes['runtime']=base.OLD_RUNTIME
    elif fault=='wrong_diagnostic':
        with p.c.cursor()as q:
            q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal');q.execute('SET LOCAL session_replication_role=replica')
            q.execute("UPDATE lab_arena_runs SET result_doc=jsonb_set(result_doc,'{failure_diagnostic,reason}','\"malformed_response\"')WHERE run_id=%s",(p.parent['run_id'],))
            q.execute('SET LOCAL session_replication_role=origin');q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
            changes['sha']='sha256:'+base.old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s",(p.parent['run_id'],))
        p.c.commit()
    else:
        with p.c.cursor()as q:
            q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal');q.execute('SET LOCAL session_replication_role=replica')
            if fault=='stale_hash':changes['sha']='sha256:'+'0'*64
            elif fault=='accepted':q.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='accepted'WHERE run_id=%s",(p.parent['assignment_id']+':4',))
            elif fault=='later':
                child=dict(p.parent);child.update(run_id=child['assignment_id']+':99',attempt=99,status='failed')
                for key in ['created_at','updated_at','reviewer_runtime_doc']:child.pop(key,None)
                base.old.insert(q,'lab_arena_runs',child)
            elif fault=='calls_pending':base.old.insert(q,'lab_arena_ledger',dict(entry_kind='reservation',run_id=p.parent['run_id'],round_id=base.old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],stage=2,call_identity='sha256:'+'7'*64,provider='openrouter',operation_id='openrouter.chat',funding_source='host'if p.organizer else'miner_key',amount_microusd=500,entry_doc={}))
            elif fault=='expired':q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb('2000-01-01T00:00:00Z'::text))WHERE round_id=%s",(base.old.ROUND,))
            elif fault=='published':q.execute("UPDATE lab_arena_rounds SET published_at=clock_timestamp(),status='published',publication_doc='{}'WHERE round_id=%s",(base.old.ROUND,))
            q.execute('SET LOCAL session_replication_role=origin');q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        p.c.commit()
    before=base.snapshot(p)
    with pytest.raises(p.pg.Error):grant(p,**changes)
    p.c.rollback();assert base.snapshot(p)==before

@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_provider_normal_claim_terminal_never_retries(provider_case,outcome):
    p=provider_case;child=grant(p)['child_identity']['run_id']
    with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE run_id=%s",(child,))
    p.c.commit();base.old.release_guard(p);lease=base.old.claim(p);assert lease['run_id']==child
    assert len(lease['company_judgment_cache']['hits'])==2
    if outcome=='expired':
        with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second'WHERE run_id=%s",(child,))
        p.c.commit();base.old.rpc(p,'lab_arena_expire_leases(%s)',(base.old.ROUND,))
    else:base.old.rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(child,base.old.TOKEN,json.dumps({'terminal_status':outcome}),outcome,child+'-output'if outcome=='accepted'else None))
    base.old.rpc(p,'lab_arena_expire_leases(%s)',(base.old.ROUND,))
    with p.c.cursor()as q:assert base.old.scalar(q,'SELECT count(*)FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',(lease['assignment_id'],lease['attempt']))==0
    p.c.commit();assert grant(p)['idempotent']is True
    with pytest.raises(p.pg.Error):grant(p,parent=child,request='recursive-provider-retry')
    p.c.rollback()


def test_service_cannot_grant(provider_case):
    p=provider_case
    with pytest.raises(p.pg.errors.InsufficientPrivilege):base.old.rpc(p,SIG,(p.parent['run_id'],p.parent_hash,base.RUNTIME,'unauthorized-provider-grant','service','evidence:provider',p.proof,False))


def test_legacy_scorer_grant_behavior_preserved(extended):
    base.test_dry_run_then_single_grant_is_idempotent_and_preserves_all_state(extended)


def test_migration_is_idempotent(provider_case):
    p=provider_case;before=base.snapshot(p)
    with p.c.cursor()as q:q.execute(SQL.read_text())
    p.c.commit();assert base.snapshot(p)==before

@pytest.mark.parametrize('fault',['stale_hash','accepted','later','calls_pending','runtime','same_runtime','expired','published','generation','provider_failure'])
def test_legacy_refusals_preserved(extended,fault):
    base.test_invalid_grant_rolls_back_without_any_work_change(extended,fault)

@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_legacy_lifecycle_preserved(extended,outcome):
    base.test_real_claim_and_terminal_paths_never_create_another_attempt(extended,outcome)

@pytest.mark.parametrize('fault',['status','error_code','byok','account_limit','wrong_call','wrong_body_hash','wrong_request','other_run','payer'])
def test_retained_provider_proof_is_exact_and_shared_pool_only(provider_case,fault):
    p=provider_case
    with p.c.cursor()as q:
        rows=base.old.scalar(q,"SELECT jsonb_agg(to_jsonb(l)ORDER BY entry_id)FROM lab_arena_ledger l WHERE run_id=%s AND call_identity=%s",(p.parent['run_id'],'sha256:'+'9'*64))
        cid='sha256:'+'a'*64
        for row in rows:
            row.pop('entry_id');row.pop('created_at');row['call_identity']=cid
            if row['entry_kind']=='uncertain':
                u=row['entry_doc']['call']['upstream_response_receipt'];u['call_identity']=cid
                body=json.loads(base64.b64decode(u['body_b64']))
                if fault=='status':u['status']=401
                elif fault=='error_code':body['error']['code']=401
                elif fault=='byok':body['error']['metadata']['is_byok']=True
                elif fault=='account_limit':body['error']['metadata']['limit_source']='account'
                elif fault=='wrong_call':u['call_identity']='sha256:'+'b'*64
                elif fault=='wrong_request':u['request_hash']='sha256:'+'b'*64
                elif fault=='other_run':row['run_id']=p.parent['assignment_id']+':4'
                elif fault=='payer':row['funding_source']='miner_key'if p.organizer else'host'
                raw=json.dumps(body).encode();u['body_b64']=base64.b64encode(raw).decode();u['body_sha256']='sha256:'+hashlib.sha256(raw).hexdigest()
                if fault=='wrong_body_hash':u['body_sha256']='sha256:'+'0'*64
            q.execute('SET LOCAL session_replication_role=replica');base.old.insert(q,'lab_arena_ledger',row);q.execute('SET LOCAL session_replication_role=origin')
        proof=base.old.scalar(q,"SELECT max(entry_id)FROM lab_arena_ledger WHERE call_identity=%s",(cid,))
    p.c.commit();before=base.snapshot(p)
    with pytest.raises(p.pg.Error):grant(p,entry=proof)
    p.c.rollback();assert base.snapshot(p)==before

@pytest.mark.parametrize('check',[base.test_prior_costs_still_exhaust_the_same_cumulative_budget,base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_provider_recovery_preserves_existing_cost_and_quota_caps(provider_case,monkeypatch,check):
    monkeypatch.setattr(base,'grant',grant)
    check(provider_case)


def test_provider_same_frozen_runtime_without_historical_pin(provider_case):
    p=provider_case
    with p.c.cursor()as q:
        cfg=base.old.scalar(q,'SELECT configuration_doc FROM lab_arena_rounds WHERE round_id=%s',(base.old.ROUND,))
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica');q.execute('UPDATE lab_arena_runs SET reviewer_runtime_doc=NULL WHERE run_id=%s',(p.parent['run_id'],));q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)',(cfg['scorer_image_digest'],cfg['scorer_image_reference'],cfg['scorer_image_digest'],cfg['scorer_image_reference']))
        p.parent_hash='sha256:'+base.old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s",(p.parent['run_id'],))
    p.c.commit()
    child=grant(p,runtime=cfg['scorer_image_digest'])['child_identity']['run_id']
    with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE run_id=%s",(child,))
    p.c.commit();base.old.release_guard(p);lease=base.old.claim(p);assert lease['run_id']==child
    with p.c.cursor()as q:
        pinned=base.old.scalar(q,"SELECT reviewer_runtime_doc->>'runtime_image_digest'FROM lab_arena_runs WHERE run_id=%s",(child,))
    p.c.commit();assert pinned==cfg['scorer_image_digest']


def test_provider_changed_mapping_after_grant_cannot_claim(provider_case,monkeypatch):
    monkeypatch.setattr(base,'grant',grant)
    base.test_changed_runtime_selection_after_grant_cannot_lease_child(provider_case)


def test_provider_replay_must_use_same_durable_proof(provider_case):
    p=provider_case;grant(p);before=base.snapshot(p)
    with pytest.raises(p.pg.Error):grant(p,entry=p.proof+1)
    p.c.rollback();assert base.snapshot(p)==before

@pytest.mark.parametrize('mutation',['source','volatility','language','strict','leakproof','parallel','acl'])
def test_migration_replay_rejects_tampered_function_source_or_security(provider_case,mutation):
    p=provider_case;sig=HELPERS[0]
    with p.c.cursor()as q:
        original=base.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))
        if mutation=='source':q.execute(original.replace(' SELECT coalesce',' SELECT /* tamper */ coalesce',1))
        elif mutation=='volatility':q.execute('ALTER FUNCTION '+sig+' VOLATILE')
        elif mutation=='language':
            q.execute('SET check_function_bodies=off');q.execute(original.replace('LANGUAGE sql','LANGUAGE plpgsql',1));q.execute('SET check_function_bodies=on')
        elif mutation=='strict':q.execute('ALTER FUNCTION '+sig+' STRICT')
        elif mutation=='leakproof':q.execute('ALTER FUNCTION '+sig+' LEAKPROOF')
        elif mutation=='parallel':q.execute('ALTER FUNCTION '+sig+' PARALLEL SAFE')
        elif mutation=='acl':q.execute('GRANT EXECUTE ON FUNCTION '+sig+' TO lab_arena_service')
    p.c.commit();before=base.snapshot(p)
    with pytest.raises(p.pg.Error):
        with p.c.cursor()as q:q.execute(SQL.read_text())
    p.c.rollback();assert base.snapshot(p)==before
    with p.c.cursor()as q:
        q.execute(original)
        q.execute('REVOKE ALL ON FUNCTION '+sig+' FROM lab_arena_service')
    p.c.commit()
