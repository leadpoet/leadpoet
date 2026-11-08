"""An explicit finite same-policy owner exception preserves native authority."""
import json
from pathlib import Path
import pytest
from tests.lab_arena import provider_child_scorer_correction675_postgres_test as prior

old=prior.old
base=prior.base
SQL=Path(__file__).parents[2]/'scripts/676-lab-arena-owner-judge-budget-exception.sql'
SIG='lab_arena_grant_judge_budget_exception_v1(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)'
database=prior.database
prepared=prior.prepared
prepared_company=prior.prepared_company
case=prior.case
base_extended=prior.base_extended
extended=prior.extended
provider_case=prior.provider_case
correctable=prior.correctable
corrected=prior.corrected

def install_finite(p):
 with p.c.cursor()as q:
  originals=[old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(s,))for s in [prior.COMMON]+prior.provider.HELPERS]
  q.execute(SQL.read_text())
  p.checkpoints=old.scalar(q,"SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')FROM lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(%s::jsonb)ref WHERE ref->>'cache_key'=j.cache_key)",(json.dumps(p.parent['company_judgment_refs']),))
  p.signal_checkpoints=old.scalar(q,"SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')FROM lab_arena_signal_judgments j WHERE j.source_scored_run_id=%s",(p.parent['scored_run_id'],))
 p.c.commit()
 try:yield p
 finally:
  p.c.rollback()
  with p.c.cursor()as q:
   for d in originals:q.execute(d)
   q.execute('DROP FUNCTION lab_arena_grant_judge_budget_exception_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,JSONB,JSONB,BOOLEAN)')
  p.c.commit()

@pytest.fixture
def finite(corrected):
 p=corrected
 child=prior.grant(p)['child_identity']['run_id']
 prior.claim_child(p,child)
 prior.complete_failed(p,child)
 prior.saved_parent(p,child)
 yield from install_finite(p)

@pytest.fixture(params=['ordinary','provider'])
def plain_finite(provider_case,request):
 p=provider_case
 p.parent_category=request.param
 if request.param=='provider':
  child=prior.provider.grant(p)['child_identity']['run_id']
  prior.claim_child(p,child);prior.complete_failed(p,child);prior.saved_parent(p,child)
 with p.c.cursor()as q:
  originals=[old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))for sig in [prior.COMMON,prior.provider.HELPERS[1]]]
  q.execute(prior.SQL.read_text())
 p.c.commit();p.exception_runtime=base.RUNTIME
 try:yield from install_finite(p)
 finally:
  p.c.rollback()
  with p.c.cursor()as q:
   for d in originals:q.execute(d)
  p.c.commit()

def grant(p,**changes):
 args=dict(parent=p.parent['run_id'],parent_hash=p.parent_hash,runtime=getattr(p,'exception_runtime',prior.RUNTIME),request_id='finite-owner-exception',actor='fixture-owner',approval='approval:explicit-one-invocation',checkpoints=p.checkpoints,signal_checkpoints=p.signal_checkpoints,dry=False)
 args.update(changes)
 args['checkpoints']=json.dumps(args['checkpoints']);args['signal_checkpoints']=json.dumps(args['signal_checkpoints'])
 with p.c.cursor()as q:result=old.scalar(q,'SELECT public.'+SIG,tuple(args.values()))
 p.c.commit();return result

def test_finite_preview_append_replay_preserves_native_state(finite):
 p=finite
 before=base.snapshot(p)
 preview=grant(p,dry=True)
 assert preview['status']=='dry_run' and base.snapshot(p)==before
 result=grant(p)
 assert result['child_identity']==preview['child_identity']
 after=base.snapshot(p)
 assert len(after['lab_arena_runs'])==len(before['lab_arena_runs'])+1
 assert len(after['lab_arena_scoring_supersessions'])==len(before['lab_arena_scoring_supersessions'])+1
 originals={r['run_id']:r for r in before['lab_arena_runs']}
 assert {r['run_id']:r for r in after['lab_arena_runs']if r['run_id']in originals}==originals
 for t in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:assert before[t]==after[t]
 journal=after['lab_arena_scoring_supersessions'][-1]['recovery_doc']
 assert journal['recovery_kind']=='operator_judge_budget_exception_v1'
 assert journal['approval_ref']=='approval:explicit-one-invocation' and 'correction_ref'not in journal
 assert grant(p)['idempotent']is True and base.snapshot(p)==after
 with pytest.raises(p.pg.Error):grant(p,request_id='another-request-same-parent')
 p.c.rollback()
 child=result['child_identity']['run_id']
 assert old.funding(p,child)['funding_source']==('host'if p.organizer else'miner_key')
 assert old.funding(p,child,'deepline')['funding_source']=='miner_key'
 prior.claim_child(p,child)

@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_native_finality_and_no_second_exception(finite,outcome):
 p=finite
 child=grant(p)['child_identity']['run_id']
 lease=prior.claim_child(p,child)
 with p.c.cursor()as q:assert old.scalar(q,"SELECT reviewer_runtime_doc->>'runtime_image_digest'FROM lab_arena_runs WHERE run_id=%s",(child,))==prior.RUNTIME
 p.c.commit()
 if outcome=='expired':
  with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second'WHERE run_id=%s",(child,))
  p.c.commit();old.rpc(p,'lab_arena_expire_leases(%s)',(old.ROUND,))
 else:
  old.rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(child,old.TOKEN,json.dumps({'terminal_status':outcome,'failure_diagnostic':{'stage':'scorer','error_class':'judge_error','reason':'malformed_response'}}),outcome,child+'-output'if outcome=='accepted'else None))
 old.rpc(p,'lab_arena_expire_leases(%s)',(old.ROUND,))
 with p.c.cursor()as q:
  assert old.scalar(q,'SELECT count(*)FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s',(lease['assignment_id'],lease['attempt']))==0
  assert old.scalar(q,'SELECT status FROM lab_arena_runs WHERE run_id=%s',(child,))==('accepted'if outcome=='accepted'else'failed')
 p.c.commit()
 assert grant(p)['idempotent']is True
 prior.saved_parent(p,child)
 before=base.snapshot(p)
 with pytest.raises(p.pg.Error):grant(p,request_id='no-recursive-exception')
 p.c.rollback();assert base.snapshot(p)==before

@pytest.mark.parametrize('fault',['stale_hash','runtime','approval','checkpoints','signal_checkpoints','accepted','later','live','calls_pending','expired','published','generation','frozen_source'])
def test_refusals_preserve_every_row(finite,fault):
 p=finite;changes={}
 with p.c.cursor()as q:
  q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal');q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime');q.execute('SET LOCAL session_replication_role=replica')
  if fault=='stale_hash':changes['parent_hash']='sha256:'+'0'*64
  elif fault=='runtime':changes['runtime']=base.RUNTIME
  elif fault=='approval':changes['approval']='  '
  elif fault=='checkpoints':changes['checkpoints']={}
  elif fault=='signal_checkpoints':changes['signal_checkpoints']={'wrong':'hash'}
  elif fault=='accepted':q.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='fixture'WHERE run_id=%s",(p.original_parent['run_id'],))
  elif fault in ('later','live'):
   row=dict(p.parent);row.update(run_id=row['assignment_id']+':99',attempt=99,status='pending'if fault=='live'else'failed',claim_request_id=None)
   for key in ['created_at','updated_at','reviewer_runtime_doc']:row.pop(key,None)
   old.insert(q,'lab_arena_runs',row)
  elif fault=='calls_pending':old.insert(q,'lab_arena_ledger',dict(entry_kind='reservation',run_id=p.parent['run_id'],round_id=old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],stage=2,call_identity='sha256:'+'7'*64,provider='openrouter',operation_id='openrouter.chat',funding_source='host'if p.organizer else'miner_key',amount_microusd=500,entry_doc={}))
  elif fault=='expired':q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb('2000-01-01T00:00:00Z'::text))WHERE round_id=%s",(old.ROUND,))
  elif fault=='published':q.execute("UPDATE lab_arena_rounds SET status='published',published_at=clock_timestamp(),publication_doc='{}'WHERE round_id=%s",(old.ROUND,))
  elif fault=='generation':q.execute('UPDATE lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s',(old.ROUND,))
  elif fault=='frozen_source':
   q.execute("UPDATE lab_arena_runs SET judgment_input_hash='sha256:'||repeat('0',64)WHERE run_id=%s",(p.parent['run_id'],))
   changes['parent_hash']='sha256:'+old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')FROM lab_arena_runs r WHERE run_id=%s",(p.parent['run_id'],))
  q.execute('SET LOCAL session_replication_role=origin');q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal');q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
 p.c.commit();before=base.snapshot(p)
 with pytest.raises(p.pg.Error):grant(p,**changes)
 p.c.rollback();assert base.snapshot(p)==before

@pytest.mark.parametrize('role',['lab_arena_service','anon','authenticated','service_role'])
def test_owner_only(finite,role):
 p=finite
 c=p.pg.connect(**p.dsn)
 try:
  with c,c.cursor()as q:
   q.execute('SET ROLE '+role)
   with pytest.raises(p.pg.errors.InsufficientPrivilege):
    old.scalar(q,'SELECT public.'+SIG,(p.parent['run_id'],p.parent_hash,prior.RUNTIME,'unauthorized','actor','approval',json.dumps(p.checkpoints),json.dumps(p.signal_checkpoints),False))
 finally:c.close()

@pytest.mark.parametrize('check',[base.test_prior_costs_still_exhaust_the_same_cumulative_budget,base.test_existing_stage_call_quota_is_not_reset_by_the_grant])
def test_existing_money_and_provider_quotas(finite,monkeypatch,check):
 monkeypatch.setattr(base,'grant',grant)
 check(finite)

@pytest.mark.parametrize('field',['approval','checkpoints','signal_checkpoints','parent_hash','runtime','dry'])
def test_null_inputs_are_refused(finite,field):
 p=finite;before=base.snapshot(p)
 with pytest.raises(p.pg.Error):grant(p,**{field:None})
 p.c.rollback();assert base.snapshot(p)==before

def test_replay_requires_same_approval_and_checkpoint_maps(finite):
 p=finite;grant(p);before=base.snapshot(p)
 for changes in [dict(approval='different-approval'),dict(checkpoints={}),dict(signal_checkpoints={'fake':'hash'})]:
  with pytest.raises(p.pg.Error):grant(p,**changes)
  p.c.rollback();assert base.snapshot(p)==before

def test_changed_mapping_prevents_claim(finite):
 p=finite;child=grant(p)['child_identity']['run_id'];prior.mapping(p,'sha256:'+'4'*64)
 with p.c.cursor()as q:q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour'WHERE run_id=%s",(child,))
 p.c.commit();old.release_guard(p)
 assert old.claim(p)['status']!='leased'

def test_migration_replay_and_unknown_source_refusal(finite):
 p=finite;before=base.snapshot(p)
 with p.c.cursor()as q:q.execute(SQL.read_text())
 p.c.commit();assert base.snapshot(p)==before
 with p.c.cursor()as q:
  q.execute("CREATE OR REPLACE FUNCTION lab_arena__operator_scorer_bug_child_v1(p_run public.lab_arena_runs)RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS 'SELECT FALSE'")
  with pytest.raises(p.pg.Error,match='preimage differs'):q.execute(SQL.read_text())
 p.c.rollback();assert base.snapshot(p)==before


def test_ordinary_and_provider_parents_keep_exact_payer_and_cache(plain_finite):
 p=plain_finite;before=base.snapshot(p);preview=grant(p,dry=True)
 assert base.snapshot(p)==before
 child=grant(p)['child_identity']['run_id']
 assert preview['child_identity']['run_id']==child
 assert old.funding(p,child)['funding_source']==('host'if p.organizer else'miner_key')
 assert old.funding(p,child,'deepline')['funding_source']=='miner_key'
 prior.claim_child(p,child)


def test_existing_grants_cannot_supply_this_same_runtime_exception(corrected):
 p=corrected
 child=prior.grant(p)['child_identity']['run_id']
 prior.claim_child(p,child);prior.complete_failed(p,child);prior.saved_parent(p,child)
 before=base.snapshot(p)
 with p.c.cursor()as q:assert old.scalar(q,"SELECT to_regprocedure('public.lab_arena_grant_judge_budget_exception_v1(text,text,text,text,text,text,jsonb,jsonb,boolean)')")is None
 p.c.commit()
 with pytest.raises(p.pg.Error):prior.grant(p,request_id='same-runtime-without-exception')
 p.c.rollback();assert base.snapshot(p)==before


def test_legacy_corrected_runtime_grant_still_works_after_extension(plain_finite):
 p=plain_finite;prior.mapping(p,prior.RUNTIME)
 result=prior.grant(p,request_id='legacy-scorer-correction-after-extension')
 assert result['runtime_image_digest']==prior.RUNTIME
 assert prior.grant(p,request_id='legacy-scorer-correction-after-extension')['idempotent']is True
 prior.claim_child(p,result['child_identity']['run_id'])


def test_legacy_provider_grant_and_nonrecursion_stay_exact(plain_finite):
 p=plain_finite;before=base.snapshot(p)
 if p.parent_category=='provider':
  with pytest.raises(p.pg.Error):prior.provider.grant(p,request='legacy-provider-recursion-still-forbidden')
  p.c.rollback();assert base.snapshot(p)==before
 else:
  result=prior.provider.grant(p,request='legacy-provider-after-extension')
  assert result['runtime_image_digest']==base.RUNTIME
  prior.claim_child(p,result['child_identity']['run_id'])
