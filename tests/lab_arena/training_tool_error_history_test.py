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


def changed_tools_example(api, current_tools):
    example = history_example(api)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    response = call['response']
    if current_tools == 'removed':
        body['tools'] = []
    elif api.endswith('responses'):
        body['tools'][0]['name'] = 'next_step_tool'
    else:
        body['tools'][0]['function']['name'] = 'next_step_tool'
    # This turn ends with text. The prior failed tool call is only input.
    if api.endswith('responses'):
        response['output'][1] = {'type': 'message', 'role': 'assistant', 'status': 'completed',
                                 'content': [{'type': 'output_text', 'text': 'Final answer.'}]}
    else:
        choice = response['choices'][0]
        choice['finish_reason'] = 'stop'
        choice['message']['content'] = 'Final answer.'
        choice['message'].pop('tool_calls')
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    seal(example)
    return example


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('current_tools', ['removed', 'renamed'])
def test_closed_malformed_history_survives_changed_current_tools(api, current_tools):
    example = changed_tools_example(api, current_tools)
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    assert training_steps._candidate(example, call)['input'] == body
    assert training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']


def test_namespaced_closed_malformed_history_survives_removed_current_tools():
    example = changed_tools_example('openrouter.responses', 'removed')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['input'][-2]['namespace'] = 'mcp__review'
    body['input'][-2]['name'] = 'review'
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    seal(example)
    assert training_steps._candidate(example, call)['input'] == body


def test_responses_current_declaration_conflicts_still_fail():
    example = history_example()
    body = example['provider_calls'][0]['request']['outbound']['body']
    conflicting = deepcopy(body['tools'][0])
    conflicting['description'] = 'Different declaration.'
    body['input'].insert(0, {'type': 'additional_tools', 'role': 'developer',
                              'tools': [conflicting]})
    with pytest.raises(ValueError, match='^tool_definition_unsupported$'):
        training_steps._responses_input(body['input'], tools=body['tools'],
                                        allow_tool_error_history=True)


def test_chat_current_declaration_shape_still_fails_before_cap_omission():
    example = history_example('openrouter.chat')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'][0]['function'] = 'invalid'
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    call['response']['choices'][0]['finish_reason'] = 'length'
    refresh_response_links(example, 0)
    seal(example)
    with pytest.raises(AttributeError):
        training_steps._candidate(example, call)
    assert not training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['missing_output', 'wrong_output', 'duplicate_output',
                                   'new_undeclared_target', 'new_malformed_target'])
def test_removed_current_tools_do_not_waive_historical_pairing_or_new_target_checks(api, damage):
    example = changed_tools_example(api, 'removed')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    values = body['input'] if api.endswith('responses') else body['messages']
    if damage == 'missing_output':
        values.pop()
    elif damage == 'wrong_output':
        if api.endswith('responses'):
            values[-1]['type'] = 'custom_tool_call_output'
        else:
            values[-1]['tool_call_id'] = 'different'
    elif damage == 'duplicate_output':
        values.append(deepcopy(values[-1]))
    else:
        if api.endswith('responses'):
            call['response']['output'][1] = {
                'type': 'function_call', 'id': 'fc-new', 'call_id': 'new',
                'name': 'not_declared', 'status': 'completed',
                'arguments': '{}' if damage == 'new_undeclared_target' else '{'}
        else:
            choice = call['response']['choices'][0]
            choice['finish_reason'] = 'tool_calls'
            choice['message']['content'] = None
            choice['message']['tool_calls'] = [{'id': 'new', 'type': 'function', 'function': {
                'name': 'not_declared', 'arguments': '{}' if damage == 'new_undeclared_target' else '{'}}]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    seal(example)
    expected = ('tool_context_unresolved' if damage in {'missing_output', 'wrong_output', 'duplicate_output'}
                else 'undeclared_tool_call' if damage == 'new_undeclared_target'
                else 'tool_call_shape_unsupported')
    with pytest.raises(ValueError, match=f'^{expected}$'):
        training_steps._candidate(example, call)


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
                                   'new_undeclared_target', 'wrong_kind', 'argument_type', 'result_type',
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
    elif damage == 'new_undeclared_target':
        if api.endswith('responses'):
            call['response']['output'][1]['name'] = 'not_declared'
        else:
            call['response']['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'not_declared'
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
