"""Strict next-response parsing follows the native stateless Responses protocol."""
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, operations, training_steps as steps, training_thinking
from tests.lab_arena.training_steps_test import bind_selected_output, response_fixture, records, seal
from tests.lab_arena.training_thinking_test import full_example


def example(*, native=False):
    result = response_fixture()
    call = result['provider_calls'][0]
    template = full_example()
    route = deepcopy(template['configuration']['reasoning_route_registry']['approved_routes'][0])
    route.update(api='openrouter.responses', full_text_fields=['output.content.reasoning_text'])
    registry = {'schema_version': 1, 'approved_routes': [route],
                'baseline_route': {'model_id': route['model_id'], 'api': route['api']}}
    call['reasoning_capture'] = dict(template['provider_calls'][0]['reasoning_capture'],
        api=route['api'], full_text_fields=route['full_text_fields'],
        registry_sha256=contracts.document_hash(registry).split(':')[1])
    call['response_receipts'][0]['metadata']['reasoning_route_observation'] = deepcopy(
        template['provider_calls'][0]['response_receipts'][0]['metadata']['reasoning_route_observation'])
    result['role'] = 'miner'
    result['configuration'] = {'reasoning_capture_policy': training_thinking.POLICY,
                               'reasoning_route_registry': registry}
    result['completeness'].update(complete=True, incomplete_reasons=[])
    body = call['request']['outbound']['body']
    body['reasoning'] = {'effort': 'high'}
    body['tools'] = [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}]
    body['input'] = [{'role': 'user', 'content': 'Inspect the retained source.'}]
    call['response']['output'][0] = {'type': 'reasoning', 'id': 'r1', 'status': 'completed',
        'content': [{'type': 'reasoning_text', 'text': 'Inspect source evidence before deciding.'}]}
    if native:
        body['input'] = [
            {'type': 'additional_tools', 'id': 'at1', 'role': 'developer', 'tools': [{
                'type': 'namespace', 'name': 'functions', 'description': 'Native local tools',
                'tools': [{'type': 'custom', 'name': 'exec', 'format': {'type': 'text'}}]}]},
            {'type': 'message', 'id': 'u1', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Inspect source.'}]},
            {'type': 'reasoning', 'id': 'r0', 'status': 'completed', 'summary': [],
             'content': [{'type': 'reasoning_text', 'text': 'Read the source.'}]},
            {'type': 'custom_tool_call', 'id': 'ct1', 'status': 'completed', 'namespace': 'functions',
             'call_id': 'c1', 'name': 'exec', 'input': 'text("source")'},
            {'type': 'custom_tool_call_output', 'id': 'co1', 'call_id': 'c1',
             'output': [{'type': 'input_text', 'text': 'Retained source text'}]},
        ]
    return refresh(result)


def refresh(result):
    call = result['provider_calls'][0]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    call['response_sha256'] = contracts.document_hash(call['response'])
    call['response_receipts'][0]['response_sha256'] = call['response_sha256']
    return seal(result)


def test_native_history_and_typed_tool_output_become_exact_input():
    value = example(native=True)
    body = value['provider_calls'][0]['request']['outbound']['body']
    operations.validate_operation_request('openrouter.responses', body, allow_full_thinking=True)
    original = deepcopy(value)
    output, _ = records(value)
    assert len(output) == 1
    assert output[0]['input'] == body
    assert value == original
    strict, _ = training_thinking.build([value])
    assert len(strict['train']) + len(strict['eval']) == 1


def test_provider_empty_logprobs_and_phase_are_preserved_in_target():
    value = example()
    message = value['provider_calls'][0]['response']['output'][1]
    message['phase'] = 'final_answer'
    message['content'][0]['logprobs'] = []
    refresh(value)
    output, _ = records(value)
    assert len(output) == 1
    assert output[0]['target'] == [message]
    assert output[0]['source_refs'][0]['target_source_paths'] == [['output', 1]]


@pytest.mark.parametrize('kind', ['function_call', 'custom_tool_call'])
def test_namespaced_native_tool_target_preserves_exact_fields(kind):
    value = example(native=True)
    body = value['provider_calls'][0]['request']['outbound']['body']
    declaration = body['input'][0]['tools'][0]['tools'][0]
    declaration.update(type='function', parameters={'type': 'object'}) if kind == 'function_call' else None
    if kind == 'function_call':
        declaration.pop('format')
    target = {'id': 'next-call', 'type': kind, 'status': 'completed', 'namespace': 'functions',
              'call_id': 'next-1', 'name': 'exec',
              **({'arguments': '{}'} if kind == 'function_call' else {'input': 'text("next")'})}
    value['provider_calls'][0]['response']['output'][1] = target
    refresh(value)
    output, _ = records(value)
    assert len(output) == 1
    assert output[0]['target'] == [target]


def test_completed_json_schema_answer_retains_declared_schema_and_exact_json():
    value = example(native=True)
    body = value['provider_calls'][0]['request']['outbound']['body']
    body['text'] = {'format': {'type': 'json_schema', 'name': 'verdict', 'strict': True,
        'schema': {'type': 'object', 'properties': {'qualified': {'type': 'boolean'}},
                   'required': ['qualified'], 'additionalProperties': False}}}
    value['provider_calls'][0]['response']['output'][1]['content'][0]['text'] = '{"qualified":true}'
    refresh(value)
    output, _ = records(value)
    assert len(output) == 1
    assert output[0]['input']['text'] == body['text']
    assert json.loads(output[0]['target'][0]['content'][0]['text']) == {'qualified': True}


@pytest.mark.parametrize('damage', ['request_hash', 'response_hash', 'ack', 'truncated', 'input_field', 'output_field', 'encrypted'])
def test_native_parser_keeps_integrity_privacy_and_completion_failures_hard(damage):
    value = example(native=True)
    call = value['provider_calls'][0]
    if damage == 'request_hash':
        call['request']['outbound']['body_sha256'] = 'forged'
    elif damage == 'response_hash':
        call['response_receipts'][0]['response_sha256'] = 'forged'
    elif damage == 'ack':
        call['response_provenance']['agent_receipt'] = 'not_reported'
    elif damage == 'truncated':
        call['response']['status'] = 'incomplete'
        refresh(value)
    elif damage == 'input_field':
        call['request']['outbound']['body']['input'][1]['untrusted'] = 'not in the schema'
        refresh(value)
    elif damage == 'output_field':
        call['response']['output'][1]['untrusted'] = 'not in the schema'
        refresh(value)
    else:
        call['request']['outbound']['body']['input'][2]['encrypted_content'] = 'opaque-token'
        refresh(value)
    seal(value)
    assert not records(value)[0]


def test_actual_three_turn_responses_wire_shapes_keep_tools_thinking_and_json():
    from pathlib import Path
    path = Path(__file__).parents[1] / 'fixtures/research_review/responses_native_three_turn_protocol.json'
    fixture = json.loads(path.read_text())
    for index, turn in enumerate(fixture['turns']):
        value = example()
        call = value['provider_calls'][0]
        body = deepcopy(turn['request'])
        call['request']['outbound']['body'] = body
        call['request']['effective_parameters']['model'] = body['model']
        call['request']['hashed_parameters'] = {'model': body['model']}
        call['request_hash'] = call['request']['request_hash'] = contracts.document_hash(call['request']['hashed_parameters'])
        call['response'] = deepcopy(turn['response'])
        refresh(value)
        original = deepcopy(value)
        got, _ = records(value)
        assert len(got) == 1, index
        assert got[0]['input'] == turn['request']
        assert got[0]['target'] == [item for item in turn['response']['output'] if item['type'] != 'reasoning']
        assert value == original
        if index == 2:
            assert json.loads(got[0]['target'][0]['content'][0]['text'])['qualified'] is False
            assert got[0]['input']['text']['format']['type'] == 'json_schema'


def test_two_native_turns_preserve_reasoning_tool_pair_and_second_answer():
    first = example(native=True)
    call = first['provider_calls'][0]
    call['response']['output'][1] = {'id': 'fc-next', 'type': 'function_call', 'status': 'completed',
        'call_id': 'next-lookup', 'name': 'lookup', 'arguments': '{}'}
    refresh(first)
    second = example(native=True)
    second_body = second['provider_calls'][0]['request']['outbound']['body']
    second_body['input'] = deepcopy(call['request']['outbound']['body']['input']) + deepcopy(call['response']['output']) + [
        {'type': 'function_call_output', 'call_id': 'next-lookup',
         'output': [{'type': 'input_text', 'text': 'The retained result.'}]}]
    refresh(second)
    for value in (first, second):
        assert len(records(value)[0]) == 1
    assert records(second)[0][0]['input']['input'][-1]['output'][0]['text'] == 'The retained result.'


@pytest.mark.parametrize('kind', ['unknown', 'mismatched_type', 'duplicate_definition', 'unpaired_output', 'retrieval'])
def test_native_tools_cannot_hide_undeclared_calls_or_unresolved_observations(kind):
    value = example(native=True)
    body = value['provider_calls'][0]['request']['outbound']['body']
    if kind in {'unknown', 'mismatched_type'}:
        value['provider_calls'][0]['response']['output'][1] = {'type': 'function_call', 'id': 'f2',
            'status': 'completed', 'call_id': 'c2', 'namespace': 'functions',
            'name': 'missing' if kind == 'unknown' else 'exec', 'arguments': '{}'}
    elif kind == 'duplicate_definition':
        body['input'][0]['tools'][0]['tools'].append({'type': 'function', 'name': 'exec', 'parameters': {}})
    elif kind == 'unpaired_output':
        body['input'][-1]['call_id'] = 'not-a-pending-call'
    else:
        body['input'][1]['content'][0]['annotations'] = [{'type': 'url_citation', 'url': 'https://example.com', 'title': 'Unretained'}]
    refresh(value)
    assert not records(value)[0]


def _eligible_example():
    from tests.lab_arena.training_eligibility_test import SOURCE_HASH
    value = example()
    value.update(source={'sha256': SOURCE_HASH}, output={'companies': []},
                 input_snapshots={'run': {'input': {'icp': 'retained input'}}})
    value['configuration']['training_maximum_missing_reasoning_percent'] = 1
    value['provider_calls'][0].update(accounting_state='settlement', actual_microusd=1,
                                      response_status=200)
    return refresh(bind_selected_output(value))


def _note_and_tool():
    return [{'type': 'message', 'id': 'note', 'role': 'assistant', 'status': 'completed',
             'phase': 'commentary', 'content': [{'type': 'output_text', 'text': 'I will inspect the source.',
                                               'annotations': [], 'logprobs': []}]},
            {'type': 'function_call', 'id': 'next', 'status': 'completed', 'call_id': 'next-lookup',
             'name': 'lookup', 'arguments': '{}'}]


@pytest.mark.parametrize('reported_reasoning', [0, 10])
def test_native_note_plus_tool_supports_full_and_explicit_zero_without_fabricating_thinking(reported_reasoning):
    from tests.lab_arena.training_eligibility_test import submission, refresh_response_links
    from lab_arena import training_eligibility
    value = _eligible_example()
    if reported_reasoning == 0:
        first = value['provider_calls'][0]
        call = deepcopy(first)
        call.update(call_identity='call-2', action_sequence=1, terminal_entry_id=6)
        call['response_receipts'][0].update(entry_id=6, response_ref={'field': 'response', 'entry_id': 6})
        payload = value['worker_observation_artifacts']['run']['payload']
        write = deepcopy(payload['response_receipts'][0])
        write.update(call_identity='call-2', action_sequence=1, ordinal=2)
        read = deepcopy(payload['agent_observations'][0])
        read.update(id='ack-2')
        read['data']['call_identity'] = 'call-2'
        payload['response_receipts'].append(write)
        payload['agent_observations'].append(read)
        call['response_provenance']['worker_receipt_ref']['ordinal'] = 2
        call['response_provenance']['worker_received_receipt_entry_ids'] = [6]
        call['response_provenance']['agent_receipt_refs'][0]['id'] = 'ack-2'
        call['response']['output'] = _note_and_tool()
        value['provider_calls'].append(call)
    else:
        call = value['provider_calls'][0]
        call['response']['output'] = call['response']['output'][:1] + _note_and_tool()
    call['response']['usage'] = {'output_tokens_details': {'reasoning_tokens': reported_reasoning}}
    for index in range(len(value['provider_calls'])):
        refresh_response_links(value, index)
        item = value['provider_calls'][index]
        item['request']['outbound']['body_sha256'] = contracts.document_hash(item['request']['outbound']['body'])
    seal(value)
    step_rows, _ = records(value)
    assert len(step_rows) == (2 if reported_reasoning == 0 else 1)
    mixed = next(row for row in step_rows if row['target'] == _note_and_tool())
    assert mixed['source_refs'][0]['target_source_paths'] == (
        [['output', 0], ['output', 1]] if reported_reasoning == 0 else [['output', 1], ['output', 2]])
    verdict = training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    assert verdict['eligible'] is True, verdict
    strict, _ = training_thinking.build([value])
    strict_rows = strict['train'] + strict['eval']
    assert len(strict_rows) == 1
    assert all(row['generated_thinking'] for row in strict_rows)
    if reported_reasoning == 0:
        assert strict_rows[0]['source_refs'][0]['call_identity'] != 'call-2'
        assert verdict['counts']['tool_only_empty_calls'] == 1
    else:
        assert strict_rows[0]['target'] == _note_and_tool()


@pytest.mark.parametrize('ambiguity', ['two_messages', 'final_with_tool', 'invalid_phase'])
def test_native_mixed_turn_rejects_ambiguous_final_answers_and_invalid_phase(ambiguity):
    value = example()
    output = _note_and_tool()
    if ambiguity == 'two_messages':
        output.insert(1, deepcopy(output[0]))
    elif ambiguity == 'final_with_tool':
        output[0]['phase'] = 'final_answer'
    else:
        output[0]['phase'] = 'unknown_phase'
    value['provider_calls'][0]['response']['output'] = value['provider_calls'][0]['response']['output'][:1] + output
    refresh(value)
    assert not records(value)[0]


def test_native_null_encrypted_placeholder_is_exact_without_accepting_opaque_bytes():
    value = example(native=True)
    history = value['provider_calls'][0]['request']['outbound']['body']['input'][2]
    history['encrypted_content'] = None
    refresh(value)
    original = deepcopy(value)
    got, _ = records(value)
    assert len(got) == 1
    assert got[0]['input']['input'][2]['encrypted_content'] is None
    assert value == original
    history['encrypted_content'] = 'real-opaque-content'
    refresh(value)
    assert not records(value)[0]
