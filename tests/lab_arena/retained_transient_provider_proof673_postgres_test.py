"""Finite retry proof distinguishes API transport from failed source crawls."""
import base64
import hashlib
import json
from pathlib import Path
import pytest
from tests.lab_arena import operator_provider_fault_recovery671_postgres_test as old

SQL=Path(__file__).parents[2]/'scripts/673-lab-arena-retained-transient-provider-proof.sql'
COMMON='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'
database=old.database
prepared=old.prepared
prepared_company=old.prepared_company
case=old.case
base_extended=old.extended

@pytest.fixture
def extended(base_extended):
 p=base_extended
 with p.c.cursor()as q:
  original=old.base.old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(COMMON,));q.execute(SQL.read_text())
 p.c.commit()
 try:yield p
 finally:
  p.c.rollback()
  with p.c.cursor()as q:
   q.execute(original);q.execute('DROP FUNCTION public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)')
  p.c.commit()

provider_case=old.provider_case

def receipt(p,*,status=504,operation='openrouter.chat',provider='openrouter',body=None,success=False,mutate=None):
 if body is None:body={'error':{'code':status,'message':'Temporary upstream failure'}}
 raw=json.dumps(body).encode();cid='sha256:'+hashlib.sha256(raw+operation.encode()).hexdigest();request='sha256:'+'4'*64
 u={'status':status,'body_b64':base64.b64encode(raw).decode(),'body_sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),'call_identity':cid,'operation_id':operation,'request_hash':request,'call_succeeded':success}
 call={'status':status if success else 502,'call_succeeded':success,'upstream_response_receipt':u}
 if mutate:mutate(call)
 with p.c.cursor()as q:
  q.execute('SET LOCAL session_replication_role=replica')
  for kind in ['reservation','dispatch','settlement']:
   old.base.old.insert(q,'lab_arena_ledger',dict(entry_kind=kind,run_id=p.parent['run_id'],round_id=old.base.old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],stage=2,call_identity=cid,provider=provider,operation_id=operation,funding_source='host'if p.organizer else'miner_key',amount_microusd=0,entry_doc={'request_hash':request}if kind=='reservation'else{},terminal_response=call if kind=='settlement'else None))
  q.execute('SET LOCAL session_replication_role=origin');entry=old.base.old.scalar(q,'SELECT max(entry_id)FROM lab_arena_ledger WHERE call_identity=%s',(cid,))
 p.c.commit();return entry

def crawl_body(code=504,tag='CRAWL_LIVECRAWL_TIMEOUT',results=None):
 return {'job_id':'fixture','status':'completed','result':{'data':{'requestId':'fixture','results':[]if results is None else results,'statuses':[{'id':'https://source.example/about','status':'error','error':{'httpStatusCode':code,'tag':tag}}]}}}

@pytest.mark.parametrize('status',[500,502,503,504])
def test_upstream_failed_http_is_one_finite_retry(provider_case,status):
 p=provider_case;entry=receipt(p,status=status);before=old.base.snapshot(p)
 result=old.grant(p,entry=entry);assert result['status']=='granted';assert result['runtime_image_digest']==old.base.RUNTIME
 after=old.base.snapshot(p)
 for table in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:assert before[table]==after[table]
 assert old.grant(p,entry=entry)['idempotent']is True


def test_actual_completed_api_with_empty_livecrawl_timeout_is_retryable(provider_case):
 p=provider_case;entry=receipt(p,status=200,operation='exa.contents',provider='deepline',body=crawl_body(),success=True)
 result=old.grant(p,entry=entry);assert result['status']=='granted'
 assert old.grant(p,entry=entry)['idempotent']is True

@pytest.mark.parametrize('fault',['auth','capability','unsupported','success','wrong_operation','wrong_code','wrong_tag','positive_data','wrong_hash','wrong_call','wrong_request','ambiguous_timeout','wrapped_auth','caller_invalid'])
def test_unproven_or_successful_source_is_ineligible(provider_case,fault):
 p=provider_case;args=dict(status=200,operation='exa.contents',provider='deepline',body=crawl_body(),success=True)
 if fault=='auth':args.update(status=401,success=False,body={'error':{'code':401}})
 elif fault=='capability':args.update(status=403,success=False,body={'code':'PROVIDER_AUTHORIZATION_FAILED','credential_owner':'deepline_managed','upstream_error_code':'THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED'})
 elif fault=='unsupported':args.update(status=501,success=False,body={'error':{'code':501}})
 elif fault=='success':args.update(status=200,body={'results':[{'text':'valid source'}]})
 elif fault=='wrong_operation':args['operation']='exa.search'
 elif fault=='wrong_code':args['body']=crawl_body(403,'CRAWL_HTTP_403')
 elif fault=='wrong_tag':args['body']=crawl_body(504,'OTHER')
 elif fault=='positive_data':args['body']=crawl_body(results=[{'text':'positive data'}])
 elif fault in ('wrong_hash','wrong_call','wrong_request'):
  field={'wrong_hash':'body_sha256','wrong_call':'call_identity','wrong_request':'request_hash'}[fault]
  args['mutate']=lambda call:call['upstream_response_receipt'].update({field:'sha256:'+'0'*64})
 elif fault=='ambiguous_timeout':args.update(status=0,success=False,body={'transport_error_class':'ReadTimeout'})
 elif fault=='wrapped_auth':args.update(status=503,success=False,body={'error':{'code':503,'metadata':{'provider_code':403}}})
 elif fault=='caller_invalid':args.update(status=500,success=False,body={'code':'UPSTREAM_BAD_INPUT','error_category':'validation','failure_origin':'caller'})
 entry=receipt(p,**args);before=old.base.snapshot(p)
 with pytest.raises(p.pg.Error):old.grant(p,entry=entry)
 p.c.rollback();assert old.base.snapshot(p)==before


def test_new_migration_replay_preserves_every_row(provider_case):
 p=provider_case;before=old.base.snapshot(p)
 with p.c.cursor()as q:q.execute(SQL.read_text())
 p.c.commit();assert old.base.snapshot(p)==before

# Existing671 callers retain their successful proof/claim/finality/cap behavior.
test_existing_shared_pool_dry_run_replay=old.test_provider_dry_run_single_append_replay_preserves_state
test_existing_provider_finality=old.test_provider_normal_claim_terminal_never_retries
test_existing_owner_only=old.test_service_cannot_grant
test_existing_exact_proof=old.test_retained_provider_proof_is_exact_and_shared_pool_only
test_existing_cost_quota=old.test_provider_recovery_preserves_existing_cost_and_quota_caps
test_existing_same_runtime_without_pin=old.test_provider_same_frozen_runtime_without_historical_pin
test_existing_mapping_pin=old.test_provider_changed_mapping_after_grant_cannot_claim
test_existing_idempotent_proof=old.test_provider_replay_must_use_same_durable_proof
