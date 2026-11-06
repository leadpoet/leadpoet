"""Only response opaque-state omission can supply a marked internal SFT projection."""
import json

import pytest

from lab_arena import contracts, internal_capture, training_export
from lab_arena.service import LocalObjectStore
from scripts.export_arena_internal_training import main
from tests.lab_arena import internal_capture_core_test as core


def saved(tmp_path, *, api='openrouter.chat', request_secret=False, context_secret=False,
          outcome_secret=False, response_secret=False, finish=True, tools=None, called=None,
          other_opaque=None, status=200, provider_error=False):
    request=({'model':'test/model','messages':[{'role':'user','content':'Inspect the synthetic company.'}], 'max_tokens':8192}
             if api=='openrouter.chat' else
             {'model':'test/model','input':[{'role':'user','content':'Inspect the synthetic company.'}],'max_output_tokens':8192})
    if request_secret:
        request['messages' if api.endswith('chat') else 'input'][0]['content']='Source contains synthetic-private-credential'
    if tools is not None:request['tools']=tools
    if other_opaque=='request':request['encrypted_content']='request opaque state'
    response=({'model':'test/model','choices':[{'index':0,'finish_reason':'stop','message':{
        'role':'assistant','content':'Supported synthetic answer.','reasoning':'Read the synthetic evidence.',
        'reasoning_details':[{'type':'reasoning.encrypted','data':'synthetic-opaque-state'}]}}]}
        if api=='openrouter.chat' else
        {'object':'response','model':'test/model','status':'completed','output':[
            {'type':'reasoning','id':'r1','encrypted_content':'synthetic-opaque-state',
             'content':[{'type':'reasoning_text','text':'Read the synthetic evidence.'}]},
            {'type':'message','role':'assistant','id':'answer','status':'completed',
             'content':[{'type':'output_text','text':'Supported synthetic answer.'}]}]})
    if called:
        response['choices'][0].update(finish_reason='tool_calls')
        response['choices'][0]['message'].update(content=None,tool_calls=[{'type':'function','id':'tool-1',
            'function':{'name':called,'arguments':'{"query":"synthetic evidence"}'}}])
    if response_secret:response['api_key']='synthetic-private-credential'
    if provider_error:response['error']={'message':'Synthetic error.'}
    objects=LocalObjectStore(tmp_path)
    call=internal_capture.InternalCapture(objects).start(purpose='icp_generation',api=api,
        context={'api_key':'synthetic-private-credential'} if context_secret else {'set_id':20261106},
        request=request,secrets=('synthetic-private-credential',))
    call.response(body=json.dumps(response).encode(),status=status)
    outcome={'api_key':'synthetic-private-credential'} if outcome_secret else {'business_outcome':'generated'}
    if other_opaque=='outcome':outcome['encrypted_content']='outcome opaque state'
    if finish:call.finish(outcome=outcome,cost_microusd=10)
    return objects,call,request


@pytest.mark.parametrize('api',['openrouter.chat','openrouter.responses'])
def test_only_opaque_response_is_a_marked_projection_with_raw_privacy_unchanged(tmp_path,api):
    objects,call,request=saved(tmp_path,api=api)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    assert candidate is not None
    assert candidate['input']==request
    assert candidate['source']['response_projection']=={'opaque_reasoning_omitted':True,'exact_provider_replay':False}
    assert candidate['source']['capture_completeness']==raw['completeness']
    assert raw['completeness']['complete'] is False and raw['completeness']['gaps']==['privacy_redactions']
    assert raw['completeness']['dimensions']['privacy'] is False
    assert raw['events']['response']['capture']['redactions']==['hidden_reasoning']
    assert raw['sft_exclusions']==[]
    assert raw['reasoning']['contains_opaque'] is True
    assert candidate['generated_thinking']==raw['reasoning']['full_generated_thinking']==[]
    assert candidate['available_reasoning']==raw['reasoning']['available_segments']
    projected,_=training_export.without_hidden_reasoning(raw['response'])
    target=projected['choices'][0]['message'] if api.endswith('chat') else [item for item in projected['output'] if item.get('type')!='reasoning']
    if api.endswith('chat'):target={k:target[k] for k in ('role','content')}
    assert candidate['target']==target
    assert 'synthetic-opaque-state' not in json.dumps(raw)
    assert candidate['quality']['status']=='unreviewed' and candidate['training_permission']['status']=='unconfirmed'


def test_projection_does_not_promote_approved_raw_reasoning_to_candidate_full_thinking(tmp_path):
    registry=core.reviewed_registry()
    payload,_=internal_capture.reasoning_capture.apply('openrouter.chat',core.request(),
        policy=internal_capture.reasoning_capture.POLICY,registry=registry)
    provider=core.response()
    provider.update(provider='Route A',openrouter_metadata={'endpoints':{'available':[
        {'selected':True,'model':'test/model','provider':'Route A'}]}})
    provider['choices'][0]['message']['reasoning_details']=[{'type':'reasoning.encrypted','data':'opaque'}]
    objects,call=core.captured(tmp_path,provider=provider,payload=payload)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json',registry=registry)
    assert raw['reasoning']['full_generated_thinking'][0]['text']=='Check the original source.'
    assert candidate['available_reasoning']==raw['reasoning']['available_segments']
    assert candidate['generated_thinking']==[]
    assert candidate['source']['response_projection']['exact_provider_replay'] is False


@pytest.mark.parametrize('where',['request','context','outcome','response'])
def test_credentials_in_any_stage_still_exclude_opaque_projection(tmp_path,where):
    kwargs={where+'_secret':True}
    objects,call,_=saved(tmp_path,**kwargs)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    assert candidate is None and raw['sft_exclusions']==['privacy_nonexact']
    assert 'synthetic-private-credential' not in json.dumps(raw)


def test_missing_outcome_does_not_acquire_projection(tmp_path):
    objects,call,_=saved(tmp_path,finish=False)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    assert candidate is None and raw['sft_exclusions']==['privacy_nonexact']


@pytest.mark.parametrize('where',['request','outcome'])
def test_opaque_omission_outside_response_is_not_a_projection(tmp_path,where):
    objects,call,_=saved(tmp_path,other_opaque=where)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    assert candidate is None and raw['sft_exclusions']==['privacy_nonexact']


@pytest.mark.parametrize('status,error,reason',[(503,False,'provider_http_error'),
                                             (200,True,'target_shape_or_completion_unsupported')])
def test_projection_does_not_turn_provider_error_into_a_target(tmp_path,status,error,reason):
    objects,call,_=saved(tmp_path,status=status,provider_error=error)
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    assert candidate is None and raw['sft_exclusions']==[reason]


@pytest.mark.parametrize('stage',['request','response','outcome'])
@pytest.mark.parametrize('field',['context_privacy','event_privacy'])
def test_projection_requires_unredacted_context_and_events(tmp_path,stage,field):
    objects,call,_=saved(tmp_path)
    ref=call.prefix+stage+'.json'
    event=json.loads(objects.get_bounded(ref,internal_capture.MAX_EVENT_BYTES))
    event[field]={'redacted':True,'redactions':['hidden_reasoning']}
    event['event_sha256']=contracts.document_hash({k:v for k,v in event.items() if k!='event_sha256'})
    changed=contracts.canonical_json(event).encode()
    class View:
        def get_optional_bounded(self,key,limit):
            return changed if key==ref else objects.get_optional_bounded(key,limit)
        def get_bounded(self,key,limit):return objects.get_bounded(key,limit)
    raw,candidate=internal_capture.read_call(View(),call.prefix+'request.json')
    assert candidate is None and raw['sft_exclusions']==['privacy_nonexact']


@pytest.mark.parametrize('declared',['yes','none','different','hosted_only'])
def test_chat_tool_validation_is_unchanged_for_opaque_projection(tmp_path,declared):
    function={'type':'function','function':{'name':'search_source' if declared=='yes' else 'other',
                                         'parameters':{'type':'object'}}}
    tools=([function] if declared in {'yes','different'} else [] if declared=='none'
           else [{'type':'openrouter:web_search','parameters':{'engine':'native'}}])
    objects,call,_=saved(tmp_path,tools=tools,called='search_source')
    raw,candidate=internal_capture.read_call(objects,call.prefix+'request.json')
    if declared=='yes':
        assert candidate is not None and candidate['target']['tool_calls'][0]['function']['name']=='search_source'
    else:
        assert candidate is None and raw['sft_exclusions']==['target_shape_or_completion_unsupported']


def test_export_manifest_keeps_partial_privacy_separate_from_projected_candidate(tmp_path,capsys):
    objects,call,_=saved(tmp_path/'objects')
    out=tmp_path/'export'
    assert main(['--refs',call.prefix+'request.json','--object-root',str(tmp_path/'objects'),'--output',str(out)])==2
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['counts']['calls']==manifest['counts']['sft_candidates']==1
    assert manifest['counts'].get('complete',0)==0
    assert manifest['gap_counts']=={'privacy_redactions':1}
    candidate=json.loads((out/'sft.jsonl').read_text())
    assert candidate['source']['response_projection']['exact_provider_replay'] is False
