"""Native router affordability failures are account diagnostics, never free-call proof."""
from copy import deepcopy
from dataclasses import replace
import json
import pytest
from lab_arena import broker as br, trajectory
from tests.lab_arena.test_lab_arena_broker import CHAT,CONTEXT,FakeTransport,HOST_KEYS,make_broker
from tests.lab_arena.provider_credit_diagnostic_test import DiagnosticReplayStore

# Redacted structure of Nov14 upstream receipt e72b97bf. No message matching.
REFUSAL={'error':{'code':402,'message':'This request requires more credits, or fewer max_tokens.','metadata':{'provider_name':None}},'openrouter_metadata':{'is_byok':False,'endpoints':{'available':[{'provider_name':'Nebius','selected':False}]}}}

def response(doc=None,status=402,provenance=None):
    return br.ProviderResponse(status,{},json.dumps(REFUSAL if doc is None else doc).encode(),provenance)

def invoke(*,kind='score',funding='miner_key',doc=None,status=402,operation='openrouter.chat'):
    transport=FakeTransport([(status,dict(deepcopy(REFUSAL if doc is None else doc),id='gen-credit-diagnostic'))])
    instance,store,_=make_broker(store=DiagnosticReplayStore(),transport=transport,judge_models=(CHAT["model"],),
        funding_source_for=lambda _:funding,credential_for=lambda _,provider:HOST_KEYS[provider])
    context=replace(CONTEXT,kind=kind)
    parameters = CHAT if operation == 'openrouter.chat' else {'model': CHAT['model'], 'input': 'research'}
    args=dict(operation_id=operation,parameters=parameters,action_sequence=0,timeout_ms=5000)
    return instance.execute(context,**args),instance,store,transport,context,args

@pytest.mark.parametrize('kind',['execute','score'])
@pytest.mark.parametrize('operation',['openrouter.chat','openrouter.responses'])
def test_native_affordability_reason_is_bound_and_replay_never_dispatches(kind, operation):
    first,instance,store,transport,context,args=invoke(kind=kind,operation=operation)
    assert first.status==402 and first.call['outcome']=='uncertain'
    assert first.call['error_code']=='miner_credentials_unavailable'
    assert br.provider_credit_failure_reason(first.call)=='provider_credit_exhausted'
    evidence=first.call['account_failure_evidence'];assert evidence['reason']=='out_of_credit'
    assert store.calls[first.call['call_identity']]['uncertain_doc']['account_failure_evidence']==evidence
    assert 'actual_microusd'not in first.call and 'credit_failure_proof'not in first.call
    projected=trajectory.provider_response_content(first.to_document(),elapsed_ms=10)
    assert projected['call']['account_failure_evidence']==evidence
    sent=len(transport.sent);again=instance.execute(context,**args)
    assert again.status==402 and again.call['outcome']=='uncertain'
    assert br.provider_credit_failure_reason(again.call)=='provider_credit_exhausted'
    assert len(transport.sent)==sent and sum(r['method']=='POST'for r in transport.sent)==1

@pytest.mark.parametrize('actual',[0,2000])
def test_later_billing_does_not_erase_credit_or_invent_retry_proof(actual):
    first,instance,store,transport,context,args=invoke()
    call=store.calls[first.call['call_identity']]
    call.update(kind='settlement',actual=actual,terminal=br._terminal_response_document(502,{},br.operations.GENERIC_UNAVAILABLE_BODY,call_succeeded=False,provider_cost={'basis':'openrouter_generation_cost','units':str(actual/1000000),'unit_name':'usd','operation':'openrouter.chat','request_id':'gen-credit-diagnostic'}))
    again=instance.execute(context,**args)
    assert again.call['actual_microusd']==actual
    assert br.provider_credit_failure_reason(again.call)=='provider_credit_exhausted'
    assert 'credit_failure_proof'not in again.call
    assert sum(r['method']=='POST'for r in transport.sent)==1

@pytest.mark.parametrize('fault',['inflight','key_limit','other_limit','provider','byok','selected','unknown_metadata','error_type','missing_message','code_string','code_bool','output','choices','usage','embedded','wrong_http','synthetic','invalid_json','not_object'])
def test_ambiguous_or_non_account_failures_do_not_gain_credit_authority(fault):
    doc=deepcopy(REFUSAL);status=402;original=402;provenance=None
    if fault in ('inflight','key_limit','other_limit'):doc['error']['metadata']['limit_source']={'inflight':'openrouter_in_flight_budget','key_limit':'openrouter_key_limit','other_limit':'unknown'}[fault]
    if fault=='provider':doc['error']['metadata']['provider_name']='Nebius'
    if fault=='byok':doc['openrouter_metadata']['is_byok']=True
    if fault=='selected':doc['openrouter_metadata']['endpoints']['available'][0]['selected']=True
    if fault=='unknown_metadata':doc['error']['metadata']['provider_error']={}
    if fault=='error_type':doc['error']['metadata']['error_type']='upstream_payment_required'
    if fault=='missing_message':doc['error'].pop('message')
    if fault=='code_string':doc['error']['code']='402'
    if fault=='code_bool':doc['error']['code']=True
    if fault in ('output','choices','usage'):doc[fault]=[]if fault!='usage'else{'cost':0.01}
    if fault=='embedded':original=200
    if fault=='wrong_http':status=403
    if fault=='synthetic':provenance='credential_echo'
    if fault=='not_object':doc=[]
    result=response(doc,status,provenance)
    if fault=='invalid_json':result=br.ProviderResponse(402,{},b'not JSON')
    assert not br._openrouter_account_credit_response(result,original_http_status=original)

@pytest.mark.parametrize('kind',['execute','score'])
def test_host_funding_never_becomes_miner_credit_diagnostic(kind):
    first,*_=invoke(kind=kind,funding='host')
    assert br.provider_credit_failure_reason(first.call)is None
    assert 'reason'not in first.call.get('account_failure_evidence',{})

@pytest.mark.parametrize('field,value',[('base_call_identity','sha256:'+'f'*64),('action_sequence',9),('provider_attempt',2)])
def test_binding_mutation_does_not_supply_account_authority(field,value):
    first,*_=invoke();call=deepcopy(first.call);call['account_failure_evidence'][field]=value
    assert br.provider_credit_failure_reason(call)is None

def test_typed_credits_and_native_minimal_envelope_are_recognized():
    doc=deepcopy(REFUSAL);doc['error']['metadata']['limit_source']='openrouter_credits'
    assert br._openrouter_account_credit_response(response(doc))
    assert br._openrouter_account_credit_response(response({'error':{'code':402,'message':'Insufficient account credits'}}))
