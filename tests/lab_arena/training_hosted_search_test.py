"""Retain the exposed hosted trajectory without inventing local tool results."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_steps, training_thinking
from tests.lab_arena.training_native_responses_test import example, refresh
from tests.lab_arena.training_steps_test import records, seal


def hosted_example():
    value = example()
    call = value['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'] = [{'type': 'openrouter:web_search', 'parameters': {
        'engine': 'exa', 'max_uses': 1, 'max_total_results': 5}}]
    body['max_tool_calls'] = 1
    first = deepcopy(call['response']['output'][0])
    second = deepcopy(first)
    second.update(id='r2')
    second['content'][0]['text'] = 'Read the returned source before answering.'
    message = deepcopy(call['response']['output'][1])
    message['id'] = 'm-final'
    message['content'][0]['annotations'] = [{'type': 'url_citation',
        'url': 'https://example.com/evidence', 'title': 'Original source',
        'content': 'The complete retained excerpt.', 'start_index': 0, 'end_index': 6}]
    call['response']['output'] = [first, {'type': 'message', 'id': 'm-progress',
        'role': 'assistant', 'status': 'completed',
        'content': [{'type': 'output_text', 'text': 'Search the original source.'}]},
        {'type': 'openrouter:web_search', 'id': 's1', 'status': 'completed',
         'action': {'type': 'search', 'query': 'original evidence',
                    'sources': [{'type': 'url', 'url': 'https://example.com/evidence'}]}},
        {'type': 'openrouter:web_search', 'id': 's2', 'status': 'completed',
         'action': {'type': 'search', 'query': 'original evidence follow up'}}, second, message]
    return refresh(value)


def test_hosted_target_is_the_exact_ordered_exposed_trajectory():
    value = hosted_example()
    original = deepcopy(value)
    rows, manifest = records(value)
    assert len(rows) == 1, manifest
    row = rows[0]
    output = value['provider_calls'][0]['response']['output']
    assert row['target'] == output
    assert row['source_refs'][0]['target_source_paths'] == [['output', i] for i in range(len(output))]
    assert row['source_refs'][0]['target_semantics'] == 'provider_executed_trajectory'
    assert value == original
    strict, manifest = training_thinking.build([value])
    strict_rows = strict['train'] + strict['eval']
    assert len(strict_rows) == 1, manifest
    assert strict_rows[0]['target'] == output
    assert [s['text'] for s in strict_rows[0]['generated_thinking']] == [
        output[0]['content'][0]['text'], output[4]['content'][0]['text']]


def test_readable_hosted_response_preserves_sequence_and_all_exposed_items():
    value = hosted_example()
    report = training_thinking.readable_trace(value)
    events = [event for event in report['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal'
              and event.get('source_path', [])[:4] == ['provider_calls', 0, 'response', 'output']]
    indexes = [event['source_path'][4] for event in events]
    assert indexes == sorted(indexes)
    whole = {event['source_path'][4]: event for event in events if len(event['source_path']) == 5}
    output = value['provider_calls'][0]['response']['output']
    assert [whole[i]['value'] for i in range(len(output))] == output
    assert whole[2]['kind'] == whole[3]['kind'] == 'provider_hosted_tool'
    assert whole[2]['execution'] == 'provider'
    assert 'The complete retained excerpt.' in training_thinking.readable_trace_markdown(report)


def test_hosted_history_needs_no_invented_local_function_result():
    value = hosted_example()
    call = value['provider_calls'][0]
    call['request']['outbound']['body']['input'] += deepcopy(call['response']['output'])
    refresh(value)
    assert len(records(value)[0]) == 1


def test_prior_hosted_context_survives_search_tool_removal_without_data_injection():
    value = hosted_example()
    call = value['provider_calls'][0]
    body = call['request']['outbound']['body']
    previous = deepcopy(call['response']['output'])
    body['input'] += previous
    body.pop('tools')
    body.pop('max_tool_calls')
    call['response']['output'] = [deepcopy(previous[-1])]
    refresh(value)
    rows, manifest = records(value)
    assert len(rows) == 1, manifest
    assert rows[0]['input']['input'][-len(previous):] == previous
    assert rows[0]['target'] == [previous[-1]]


@pytest.mark.parametrize('paired', [True, False])
def test_hosted_history_does_not_close_or_block_a_pending_local_call(paired):
    value = hosted_example()
    call = value['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'].append({'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}})
    body['input'] += [
        {'type': 'function_call', 'call_id': 'local', 'name': 'lookup',
         'arguments': '{}', 'status': 'completed'},
        deepcopy(call['response']['output'][2]),
    ]
    if paired:
        body['input'].append({'type': 'function_call_output', 'call_id': 'local',
                              'output': 'Retained local source evidence.'})
    original = deepcopy(body['input'])
    refresh(value)
    rows, manifest = records(value)
    if paired:
        assert len(rows) == 1, manifest
        assert rows[0]['input']['input'] == original
    else:
        assert rows == []
        assert manifest['counts']['exclusion_reasons'] == {'tool_context_unresolved': 1}


def test_hosted_capture_keeps_host_result_policy_without_client_receipts():
    from tests.lab_arena.training_host_result_test import without_client_callbacks

    value = without_client_callbacks(hosted_example())
    rows, manifest = records(value)
    assert len(rows) == 1, manifest
    assert rows[0]['source_refs'][0]['client_consumption'] == 'not_proved'
    strict, manifest = training_thinking.build([value])
    assert len(strict['train']) + len(strict['eval']) == 1, manifest


@pytest.mark.parametrize('declared', [True, False])
def test_hosted_and_local_actions_keep_separate_execution_and_declarations(declared):
    value = hosted_example()
    call = value['provider_calls'][0]
    if declared:
        call['request']['outbound']['body']['tools'].append({
            'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}})
    local_call = {'type': 'function_call', 'id': 'local-id', 'call_id': 'local-call',
                  'status': 'completed', 'name': 'lookup', 'arguments': '{}'}
    call['response']['output'].append(local_call)
    refresh(value)
    rows, _ = records(value)
    if not declared:
        assert rows == []
        return
    assert len(rows) == 1
    assert rows[0]['target'][-1] == local_call
    assert not any(item['type'].endswith('_output') for item in rows[0]['target'])
    trace = training_thinking.readable_trace(value)
    event = next(event for event in trace['calls'][0]['events'] if event.get('value') == local_call)
    assert event['kind'] == 'tool_call'
    assert 'execution' not in event


@pytest.mark.parametrize('mutation', ['missing_declaration', 'unfinished', 'unknown_field',
    'bad_citation', 'unpaired_local', 'privacy', 'provenance'])
def test_hosted_trajectory_keeps_closed_schema_privacy_and_provenance(mutation):
    value = hosted_example()
    call = value['provider_calls'][0]
    body, output = call['request']['outbound']['body'], call['response']['output']
    if mutation == 'missing_declaration':
        body.pop('tools'); body.pop('max_tool_calls')
    elif mutation == 'unfinished':
        output[2]['status'] = 'in_progress'
    elif mutation == 'unknown_field':
        output[2]['action']['invented'] = 'unsupported'
    elif mutation == 'bad_citation':
        output[-1]['content'][0]['annotations'][0]['url'] = 'file:///private'
    elif mutation == 'unpaired_local':
        body['tools'].append({'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}})
        body['input'].append({'type': 'function_call', 'call_id': 'unclosed',
                             'name': 'lookup', 'arguments': '{}', 'status': 'completed'})
    elif mutation == 'privacy':
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
    else:
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = 'bad'
    refresh(value)
    assert records(value)[0] == []
