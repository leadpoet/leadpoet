"""Real managed-retry shape: settled free429 has no client write of its own."""
from copy import deepcopy
import pytest
from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_eligibility_test import refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


def recovered_example(*, truncated=False, api='openrouter.responses'):
    value = cap_example(api)
    success = value['provider_calls'][0]
    body=success['request']['outbound']['body']
    body['reasoning'].pop('exclude',None)
    success['request']['outbound']['body_sha256']=contracts.document_hash(body)
    if not truncated:
        if api == 'openrouter.responses':
            success['response']['output'][1]['arguments'] = '{}'
            success['response']['usage']['output_tokens'] = 50
        else:
            success['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{}'
            success['response']['usage']['completion_tokens'] = 50
    refresh_response_links(value, 0)
    value = without_client_callbacks(value)
    for call in value['provider_calls']:
        call['action_sequence'] += 1
    for row in value['worker_observation_artifacts']['run']['payload']['response_receipts']:
        row['action_sequence'] += 1
    failed = deepcopy(value['provider_calls'][0])
    failed.update(call_identity='hidden-free-retry', action_sequence=0, funding_source='miner_key',
                  accounting_state='settlement', actual_microusd=0, terminal_entry_id=1,
                  response_status=502, response={'error': {'code': 'provider_unavailable'}})
    failed['response_sha256'] = contracts.document_hash(failed['response'])
    failed['response_provenance'] = {'source':'stored_provider_ledger_receipt','delivery_to_worker':'not_proved'}
    failed['response_receipts'] = [{'entry_id':1, 'entry_kind':'settlement',
        'response_ref':{'entry_id':1,'field':'response'}, 'response_sha256':failed['response_sha256'],
        'wire_body_sha256':failed['response_sha256'], 'wire_body_bytes':41,
        'metadata':{'status':502,'call_succeeded':False}}]
    raw = {'error': {'message':'Provider returned error','code':429,'metadata':{'provider_name':None}}}
    digest = contracts.document_hash(raw)
    failed['upstream_response_receipts'] = [{'entry_id':1,'status':429,'response':raw,
        'response_sha256':digest,'privacy_redactions':[], 'body_sha256':digest,'wire_body_sha256':digest,
        'body_bytes':len(contracts.canonical_json(raw).encode()),'wire_body_bytes':len(contracts.canonical_json(raw).encode()),
        'provenance':'private_upstream_response_receipt','delivery_to_worker':'not_proved'}]
    failed['lifecycle'] = [{'entry_id':1,'entry_kind':'settlement','amount_microusd':0,'accounting':{}}]
    value['provider_calls'].insert(0,failed)
    artifact=value['worker_observation_artifacts']['run']
    artifact['payload'].update(complete=True, failures=[])
    artifact['payload_sha256']=contracts.document_hash(artifact['payload'])
    for call in value['provider_calls'][1:]:
        call['response_provenance']['worker_receipt_ref']['payload_sha256']=artifact['payload_sha256']
    return seal(value)


def test_hidden_free_retry_keeps_eligibility_and_successful_targets_without_a_fake_write():
    value=recovered_example();before=deepcopy(value);failed=value['provider_calls'][0]
    assert training_thinking.verified_failed_attempt(value,failed)
    result=training_eligibility.assess_capture(submission(),{'run':value},['run'],enabled=True)
    assert result['eligible'],result
    linked=deepcopy(value)
    errors=training_export._link_worker_receipts({'run_id':'run','kind':'execute'},linked['provider_calls'],
        linked['worker_observation_artifacts']['run'],require_client_ack=False,configuration=value['configuration'])
    assert not errors
    splits,manifest=training_thinking.build([value],training_steps.Rules(production_rounds_only=False))
    assert len(splits['train'])+len(splits['eval'])==3,manifest
    assert value==before


def test_hidden_free_retry_and_recovered_cap_eof_keep_their_separate_proofs():
    value=recovered_example(truncated=True)
    failed,recovered=value['provider_calls'][:2]
    assert training_thinking.verified_recovered_provider_retry(value,failed)
    assert training_thinking.verified_cap_truncated_target(value,recovered)
    result=training_eligibility.assess_capture(submission(),{'run':value},['run'],enabled=True)
    assert result['eligible'],result
    splits,manifest=training_thinking.build([value],training_steps.Rules(production_rounds_only=False))
    assert len(splits['train'])+len(splits['eval'])==2,manifest


@pytest.mark.parametrize('damage',['request','order','ledger_order','cost','unknown_cost','body','wire','write',
    'route','body_route','privacy','receipt_hash','missing_wire','ack','claimed_upstream_write','claimed_received',
    'legacy','generated'])
def test_unproved_or_changed_hidden_retry_remains_hard(damage):
    value=recovered_example();failed,success=value['provider_calls'][:2]
    if damage=='request': success['request_hash']=contracts.hash_bytes(b'different')
    elif damage=='order': failed['action_sequence']=success['action_sequence']
    elif damage=='ledger_order': failed['terminal_entry_id']=success['terminal_entry_id']
    elif damage=='cost': failed['actual_microusd']=1
    elif damage=='unknown_cost': failed['actual_microusd']=None
    elif damage=='body': failed['upstream_response_receipts'][0]['response']['error']['code']=500
    elif damage=='wire': failed['upstream_response_receipts'][0]['wire_body_bytes']+=1
    elif damage=='write': success['response_provenance']['delivery_to_worker']='write_failed'
    elif damage=='route': success['reasoning_capture']['provider_route']='other'
    elif damage=='body_route':
        for call in (failed,success):
            call['request']['outbound']['body']['provider']['only']=['other']
            call['request']['outbound']['body_sha256']=contracts.document_hash(call['request']['outbound']['body'])
    elif damage=='privacy': failed['upstream_response_receipts'][0]['privacy_redactions']=['credential']
    elif damage=='receipt_hash': failed['upstream_response_receipts'][0]['response_sha256']='forged'
    elif damage=='missing_wire': failed['upstream_response_receipts'][0].pop('wire_body_sha256')
    elif damage=='ack': failed['response_provenance']['agent_receipt']='reported_exact_body_received'
    elif damage=='claimed_upstream_write': failed['upstream_response_receipts'][0]['delivery_to_worker']='socket_written'
    elif damage=='claimed_received': failed['response_provenance']['worker_received_receipt_entry_ids']=[failed['terminal_entry_id']]
    elif damage=='legacy': value['configuration'].pop('training_maximum_missing_reasoning_percent')
    else: failed['upstream_response_receipts'][0]['response']['output']=[{'type':'message'}]
    seal(value)
    assert not training_thinking.verified_failed_attempt(value,failed)
