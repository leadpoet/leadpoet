"""Exact tool-error history is provider input, never a valid generated target."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


def history_example(api='openrouter.responses', arguments='{"company":'):
    example = cap_example(api)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    if api.endswith('responses'):
        body['input'] += [
            {'type': 'function_call', 'id': 'fc-prior', 'call_id': 'prior',
             'name': 'lookup_evidence', 'arguments': arguments},
            {'type': 'function_call_output', 'call_id': 'prior', 'output': [
                {'type': 'input_text', 'text': 'err: EOF while parsing an object'}]},
        ]
        call['response']['output'][1]['arguments'] = '{"company":"valid next action"}'
    else:
        name = body['tools'][0]['function']['name']
        body['messages'] += [
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'prior', 'type': 'function',
                'function': {'name': name, 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': 'prior', 'content': 'err: EOF while parsing an object'},
        ]
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"company":"valid next action"}'
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    return seal(example)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('arguments', ['{"company":', 'not JSON', '{"wrong": invalid}', '{"company":"' + 'x' * 51_300], ids=['eof', 'text', 'syntax', 'long_eof'])
def test_exact_paired_tool_error_history_keeps_valid_following_target(api, arguments):
    example = history_example(api, arguments)
    before = deepcopy(example)
    call = example['provider_calls'][0]
    candidate = training_steps._candidate(example, call)
    assert candidate['input'] == call['request']['outbound']['body']
    gate = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert gate['eligible'], gate
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits['train']) + len(splits['eval']) == 3
    assert example == before


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['missing_output', 'duplicate_call', 'duplicate_output', 'reused_id',
                                   'unregistered', 'wrong_kind', 'argument_type', 'result_type',
                                   'request_hash', 'host_hash', 'target_json', 'legacy', 'off'])
def test_malformed_history_does_not_waive_pairing_declaration_types_or_proof(api, damage):
    example = history_example(api)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    values = body['input'] if api.endswith('responses') else body['messages']
    prior, output = values[-2:]
    if damage == 'missing_output': values.pop()
    elif damage == 'duplicate_call': values.insert(-1, deepcopy(prior))
    elif damage == 'duplicate_output': values.append(deepcopy(output))
    elif damage == 'reused_id': values.extend([deepcopy(prior), deepcopy(output)])
    elif damage == 'unregistered':
        (prior if api.endswith('responses') else prior['tool_calls'][0]['function'])['name'] = 'not_declared'
    elif damage == 'wrong_kind':
        if api.endswith('responses'):
            output['type'] = 'custom_tool_call_output'
        else: prior['tool_calls'][0]['type'] = 'custom'
    elif damage == 'argument_type':
        (prior if api.endswith('responses') else prior['tool_calls'][0]['function'])['arguments'] = {'bad': 'object'}
    elif damage == 'result_type':
        if api.endswith('responses'): output['output'] = {'bad': 'object'}
        else: output['content'] = {'bad': 'object'}
    elif damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'bad')
    elif damage == 'host_hash': call['response_provenance']['worker_receipt_ref']['ordinal'] = 999
    elif damage == 'target_json':
        if api.endswith('responses'): call['response']['output'][1]['arguments'] = '{"bad":'
        else: call['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"bad":'
        usage_key = 'output_tokens' if api.endswith('responses') else 'completion_tokens'
        call['response']['usage'][usage_key] -= 1  # No cap omission allowance.
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent')
    else: example['configuration']['reasoning_capture_policy'] = None
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    seal(example)
    with pytest.raises(ValueError): training_steps._candidate(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_valid_history_keeps_previous_acceptance_when_old_tool_is_no_longer_declared(api):
    example = history_example(api, '{}')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    if api.endswith('responses'):
        body['input'][-2]['name'] = 'previous_valid_tool'
    else:
        body['messages'][-2]['tool_calls'][0]['function']['name'] = 'previous_valid_tool'
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    seal(example)
    assert training_steps._candidate(example, call)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_valid_history_keeps_previous_acceptance_for_reused_completed_ids(api):
    example = history_example(api, '{}')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    items = body['input'] if api.endswith('responses') else body['messages']
    items.extend(deepcopy(items[-2:]))
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    seal(example)
    assert training_steps._candidate(example, call)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_prior_malformed_history_does_not_make_a_new_malformed_target_valid(api):
    example = history_example(api)
    call = example['provider_calls'][0]
    if api.endswith('responses'):
        call['response']['output'][1]['arguments'] = '{"new":'
    else:
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"new":'
    refresh_response_links(example, 0)
    with pytest.raises(ValueError, match='tool_call_shape_unsupported'):
        training_steps._candidate(example, call)
    assert training_thinking.verified_cap_truncated_target(example, call)
