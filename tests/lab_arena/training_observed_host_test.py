"""Historical provider-visible pairs need host evidence, never invented reads."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import fixture, response_fixture, seal


def historical_host_example(api='openrouter.chat'):
    example = fixture() if api == 'openrouter.chat' else response_fixture()
    example['configuration'] = {}
    example = without_client_callbacks(example)
    example.pop('configuration', None)
    return seal(example)


def rows(example):
    splits, manifest = training_steps.build([example])
    return splits['train'] + splits['eval'], manifest


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('role', ['miner', 'baseline'])
def test_historical_host_pair_exports_without_changing_policy_or_claiming_consumption(api, role):
    example = historical_host_example(api)
    example['role'] = role
    seal(example)
    before = deepcopy(example)
    candidates, manifest = rows(example)
    assert len(candidates) == 1, manifest['counts']
    candidate = candidates[0]
    assert candidate['input'] == example['provider_calls'][0]['request']['outbound']['body']
    assert candidate['source_refs'][0]['capture_scope'] == 'provider_visible_request_result'
    assert candidate['source_refs'][0]['client_consumption'] == 'not_proved'
    assert manifest['training_permission']['status'] == 'unconfirmed'
    assert example == before
    with pytest.raises(ValueError, match='agent_read_unverified'):
        training_steps._candidate(example, example['provider_calls'][0])
    strict, _ = training_thinking.build([example])
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('damage', ['missing_snapshot', 'payload_hash', 'write_state', 'write_status',
    'write_length', 'write_hash', 'write_identity', 'receipt_hash', 'request_hash', 'outbound_hash',
    'request_privacy', 'response_privacy', 'ack_claim', 'ack_ref'])
def test_historical_host_pair_keeps_exactness_and_existing_ack_checks(damage):
    example = historical_host_example()
    call = example['provider_calls'][0]
    artifact = example['worker_observation_artifacts']['run']
    write = artifact['payload']['response_receipts'][0]
    if damage == 'missing_snapshot': example['worker_observation_artifacts'] = {}
    elif damage == 'payload_hash': artifact['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'write_state': write['write_state'] = 'socket_write_failed'
    elif damage == 'write_status': write['status'] = 201
    elif damage == 'write_length': write['body_bytes'] += 1
    elif damage == 'write_hash': write['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'write_identity': write['call_identity'] = 'other'
    elif damage == 'receipt_hash': call['response_receipts'][0]['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'request_privacy': call['request_capture'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'response_privacy': call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'ack_claim': call['response_provenance']['agent_receipt'] = 'reported_exact_body_received'
    elif damage == 'ack_ref': call['response_provenance']['agent_receipt_refs'] = [{'id': 'fake'}]
    if damage in {'write_state', 'write_status', 'write_length', 'write_hash', 'write_identity'}:
        artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    assert rows(seal(example))[0] == []


def test_generic_host_pair_does_not_relax_legacy_full_thinking_capture_or_champion_gate():
    # Keep the exact approved historical route, but remove client callbacks.
    # Generic delivery evidence must not change the frozen strict rules.
    example = without_client_callbacks(captured_example())
    example['configuration'].pop('training_maximum_missing_reasoning_percent')
    seal(example)
    before = deepcopy(example)
    candidates, _ = training_steps.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(candidates['train']) + len(candidates['eval']) > 0
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    assert not training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']
    assert example == before


def test_generic_host_delivery_keeps_candidate_identity_and_split_groups():
    original = fixture()
    candidate = training_steps._candidate(original, original['provider_calls'][0])
    expected_id = contracts.document_hash({key: candidate[key] for key in ('format', 'input', 'target')})
    historical = historical_host_example()
    split, _ = training_steps.build([original, historical])
    assert len(split['train']) + len(split['eval']) == 1
    result = (split['train'] + split['eval'])[0]
    assert result['candidate_id'] == expected_id
    assert len(result['source_refs']) == 2
    assert {ref['example_hash'] for ref in result['source_refs']} == {
        original['example_hash'], historical['example_hash']}


def historical_readable_history(api):
    example = historical_host_example(api)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    if api == 'openrouter.chat':
        body['messages'] += [
            {'role': 'assistant', 'content': None, 'reasoning': 'Read the retained source.',
             'reasoning_details': [{'type': 'reasoning.text', 'text': 'Compare the evidence.'}],
             'tool_calls': [{'id': 'lookup-1', 'type': 'function',
                             'function': {'name': 'lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'lookup-1', 'content': 'Retained source evidence.'}]
        body['tools'] = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    else:
        body['input'] = [
            {'type': 'message', 'id': 'user-1', 'role': 'user', 'content': 'Review the company.'},
            {'type': 'reasoning', 'id': 'reasoning-1', 'encrypted_content': None, 'summary': [],
             'content': [{'type': 'reasoning_text', 'text': 'Read the retained source.'}]},
            {'type': 'function_call', 'id': 'function-1', 'call_id': 'lookup-1',
             'name': 'lookup', 'namespace': None, 'arguments': '{}'},
            {'type': 'function_call_output', 'id': 'result-1', 'call_id': 'lookup-1',
             'output': 'Retained source evidence.'}]
        body['tools'] = [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}]
        # The real TYCHE response includes this empty native transport field.
        call['response']['output'][-1]['content'][0]['logprobs'] = []
    return refresh_history(example)


def refresh_history(example):
    call = example['provider_calls'][0]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    wire = contracts.canonical_json(call['response']).encode()
    wire_hash = contracts.hash_bytes(wire)
    receipt = call['response_receipts'][0]
    receipt.update(response_sha256=contracts.document_hash(call['response']),
                   wire_body_sha256=wire_hash, wire_body_bytes=len(wire))
    artifact = example['worker_observation_artifacts']['run']
    artifact['payload']['response_receipts'][0].update(body_sha256=wire_hash, body_bytes=len(wire))
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    call['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    return seal(example)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_generic_historical_readable_history_preserves_exact_native_context(api):
    example = historical_readable_history(api)
    before = deepcopy(example)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    with pytest.raises(ValueError):
        training_steps._target(body, call['response'], api, allow_reasoning_history=False)
    candidates, manifest = rows(example)
    assert len(candidates) == 1, manifest['counts']
    assert candidates[0]['input'] == body
    assert candidates[0]['source_refs'][0]['client_consumption'] == 'not_proved'
    strict, _ = training_thinking.build([example])
    assert not strict['train'] and not strict['eval']
    assert example == before


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['opaque', 'reasoning_type', 'malformed_arguments',
                                   'wrong_result_id', 'missing_result', 'unknown_field'])
def test_generic_readable_history_keeps_closed_schema_privacy_and_tool_links(api, damage):
    example = historical_readable_history(api)
    body = example['provider_calls'][0]['request']['outbound']['body']
    if api == 'openrouter.chat':
        assistant, result = body['messages'][1:]
        if damage == 'opaque': assistant['reasoning_details'] = [{'type': 'reasoning.encrypted', 'data': 'opaque-state'}]
        elif damage == 'reasoning_type': assistant['reasoning'] = {'text': 'wrong type'}
        elif damage == 'malformed_arguments': assistant['tool_calls'][0]['function']['arguments'] = '{'
        elif damage == 'wrong_result_id': result['tool_call_id'] = 'different'
        elif damage == 'missing_result': body['messages'].pop()
        elif damage == 'unknown_field': assistant['unrecognized'] = 'not admitted'
    else:
        reasoning, function, result = body['input'][1:]
        if damage == 'opaque': reasoning['encrypted_content'] = 'opaque-state'
        elif damage == 'reasoning_type': reasoning['content'] = [{'type': 'reasoning_text', 'text': 123}]
        elif damage == 'malformed_arguments': function['arguments'] = '{'
        elif damage == 'wrong_result_id': result['call_id'] = 'different'
        elif damage == 'missing_result': body['input'].pop()
        elif damage == 'unknown_field': reasoning['unrecognized'] = 'not admitted'
    assert not rows(refresh_history(example))[0]


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['malformed_target_arguments', 'empty_message_scaffold'])
def test_generic_readable_history_does_not_relax_generated_targets(api, damage):
    example = historical_readable_history(api)
    response = example['provider_calls'][0]['response']
    if api == 'openrouter.chat':
        message = response['choices'][0]['message']
        if damage == 'malformed_target_arguments':
            message.update(content=None, tool_calls=[{'id': 'new-call', 'type': 'function',
                'function': {'name': 'lookup', 'arguments': '{'}}])
            response['choices'][0]['finish_reason'] = 'tool_calls'
        else:
            message['content'] = None
    elif damage == 'malformed_target_arguments':
        response['output'] = [{'type': 'function_call', 'id': 'new-function', 'call_id': 'new-call',
            'name': 'lookup', 'arguments': '{', 'status': 'completed'}]
    else:
        response['output'][-1]['content'] = []
    assert not rows(refresh_history(example))[0]
