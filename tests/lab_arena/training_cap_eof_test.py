"""A proven output-cap EOF is omitted without repairing malformed tool JSON."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


def cap_example(api='openrouter.chat'):
    example = captured_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    cap = call['reasoning_capture']['max_output_tokens']
    if api == 'openrouter.chat':
        message = call['response']['choices'][0]['message']
        message['reasoning'] = 'Inspect the saved source before reviewing the company.'
        message['tool_calls'][0]['function']['arguments'] = '{"company":'
        call['response']['usage']['completion_tokens'] = cap
        call['response']['usage']['completion_tokens_details'] = {'reasoning_tokens': 10}
        body['max_tokens'] = cap
    else:
        registry = example['configuration']['reasoning_route_registry']
        route = deepcopy(next(r for r in registry['approved_routes'] if r['model_id'] == body['model']))
        route.update(api=api, full_text_fields=['output.content.reasoning_text'])
        registry['approved_routes'].append(route)
        for item in example['provider_calls']:
            item['reasoning_capture']['registry_sha256'] = contracts.document_hash(registry).split(':')[1]
        call['operation_id'] = call['request']['operation_id'] = api
        call['request']['outbound']['path'] = '/api/v1/responses'
        call['reasoning_capture'].update(api=api, full_text_fields=['output.content.reasoning_text'])
        body.pop('messages')
        body.pop('max_tokens', None)
        # Convert the saved Chat fixture to a valid Responses request. These
        # Chat controls must not be admitted by the evidence reader either.
        body['reasoning'].pop('exclude', None)
        body.pop('tool_choice', None)
        body.update(input=[{'role': 'user', 'content': 'Review the saved company.'}], max_output_tokens=cap,
                    tools=[{'type': 'function', 'name': 'lookup_evidence', 'parameters': {'type': 'object'}}])
        call['response'] = {'status': 'completed', 'error': None, 'output': [
            {'type': 'reasoning', 'id': 'r', 'status': 'completed', 'content': [
                {'type': 'reasoning_text', 'text': 'Inspect the saved source before reviewing the company.'}]},
            {'type': 'function_call', 'id': 'f', 'call_id': 'call-eof', 'name': 'lookup_evidence',
             'arguments': '{"company":', 'status': 'completed'}],
            'usage': {'output_tokens': cap, 'output_tokens_details': {'reasoning_tokens': 10}}}
        artifact = example['worker_observation_artifacts']['run']
        artifact['payload']['response_receipts'][0]['operation_id'] = api
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    return seal(example)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_cap_eof_omits_only_this_generation_and_keeps_other_full_targets(api):
    example = cap_example(api)
    before = deepcopy(example)
    assert training_steps._host_result(example, example['provider_calls'][0])
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    assert result['counts']['reasoning_by_run'][0]['truncated_model_calls'] == 1
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(r['source_refs'][0]['call_identity'] != example['provider_calls'][0]['call_identity'] for r in records)
    assert example == before


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('arguments', ['{"company":"unfinished', '{"company":"escaped\\',
    '{"company":"unicode\\u12', '{"company":1e', '{"company":-1.2e+',
    '{"company":1.', '{"company":tru', '{"company":nul', '{"company":-'])
def test_cap_cut_inside_a_valid_json_token_omits_only_that_call(api, arguments):
    example = cap_example(api)
    call = example['provider_calls'][0]
    target = (call['response']['choices'][0]['message']['tool_calls'][0]['function']
              if api.endswith('chat') else call['response']['output'][1])
    target['arguments'] = arguments
    refresh_response_links(example, 0)
    before = deepcopy(example)
    assert training_thinking.verified_cap_truncated_target(example, call)
    assert training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(row['source_refs'][0]['call_identity'] != call['call_identity'] for row in records)
    assert example == before


@pytest.mark.parametrize('document', [
    '{}', '{"value":[true,false,null,-1,0,1.25e-4,{"escaped":"a\\\\b\\\"c\\u1234"}]}',
    ' { "unicode": "source café", "nested": { "values": [[],{}] } }',
])
def test_every_incomplete_object_prefix_is_recognized_without_repair(document):
    for length in range(1, len(document)):
        prefix = document[:length]
        assert training_thinking._incomplete_json_object_prefix(prefix) is bool(prefix.strip()), prefix
    assert not training_thinking._incomplete_json_object_prefix(document)


@pytest.mark.parametrize('arguments', [
    '{"x":"\\q', '{"x":"\\uZZ', '{"x":"raw\n', '{"x":01', '{"x":1.e',
    '{"x":1.2.', '{"x":0\n.', '{"x":0\ne', '{"x":0\r.', '{"x":0\t.',
    '{"x":0 .', '{"x":0 e+', '{"x":1e2e', '{"x":"word"e', '{"x":[]e', '{"x":t ', '{"x":truex',
    '{"x":NaN,', '{"x":Infinity,', '{"x":1,}', '{"x":1}\u00a0', '{}{"x":"',
    '{"x" 1', '[1,', '', '  ', '{}', '{"x":1} garbage',
])
def test_invalid_or_complete_json_cannot_be_a_cap_prefix(arguments):
    assert not training_thinking._incomplete_json_object_prefix(arguments)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('arguments', ['{"company":1.2.', '{"company":0\n.'])
def test_invalid_number_suffix_at_cap_keeps_the_entrant_ineligible(api, arguments):
    example = cap_example(api)
    call = example['provider_calls'][0]
    target = (call['response']['choices'][0]['message']['tool_calls'][0]['function']
              if api.endswith('chat') else call['response']['output'][1])
    target['arguments'] = arguments
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    assert not training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['below_cap', 'above_cap', 'missing_usage', 'boolean_usage', 'wrong_request_cap',
                                   'generic_json', 'array_eof', 'complete_json', 'schema', 'host', 'request', 'route', 'legacy'])
def test_cap_eof_never_waives_other_output_or_evidence_failures(api, damage):
    example = cap_example(api)
    call = example['provider_calls'][0]
    response = call['response']
    body = call['request']['outbound']['body']
    usage_key = 'completion_tokens' if api.endswith('chat') else 'output_tokens'
    cap_key = 'max_tokens' if api.endswith('chat') else 'max_output_tokens'
    if api.endswith('chat'):
        target = response['choices'][0]['message']['tool_calls'][0]['function']
    else:
        target = response['output'][1]
    if damage == 'below_cap': response['usage'][usage_key] -= 1
    elif damage == 'above_cap': response['usage'][usage_key] += 1
    elif damage == 'missing_usage': response['usage'].pop(usage_key)
    elif damage == 'boolean_usage': response['usage'][usage_key] = True
    elif damage == 'wrong_request_cap':
        body[cap_key] -= 1
        response['usage'][usage_key] = body[cap_key]
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    elif damage == 'generic_json': target['arguments'] = '{"company": invalid}'
    elif damage == 'array_eof': target['arguments'] = '[1,'
    elif damage == 'complete_json': target['arguments'] = '[]'
    elif damage == 'schema': target['unexpected'] = True
    elif damage == 'host': call['response_provenance']['worker_receipt_ref']['ordinal'] = 999
    elif damage == 'request': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'route': call['reasoning_capture']['provider_route'] = 'wrong-route'
    else: example['configuration'].pop('training_maximum_missing_reasoning_percent')
    refresh_response_links(example, 0)
    seal(example)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert not result['eligible']
    if damage != 'legacy':
        coverage = training_thinking.coverage([example])
        assert coverage['trajectories']['ineligible'] == 1
        assert coverage['trajectories']['complete_full_thinking'] == 0
        assert coverage['trajectories']['selected_research_full_thinking'] == 0


@pytest.mark.parametrize('arguments', ['{"x":"\\q', '{}{"x":"', '{"x" 1', '{"x":1} garbage', '{"x":NaN,', '{"x":1}\u00a0'])
def test_non_eof_json_errors_are_not_truncation(arguments):
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    call['response']['output'][1]['arguments'] = arguments
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)


@pytest.mark.parametrize('damage', ['undeclared_name', 'wrong_kind', 'extra_field', 'bad_phase', 'bad_declaration'])
def test_eof_omission_still_validates_the_entire_declared_target_shape(damage):
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    if damage == 'undeclared_name': call['response']['output'][1]['name'] = 'undeclared'
    elif damage == 'wrong_kind': body['tools'][0] = {'type': 'custom', 'name': 'lookup_evidence', 'format': {'type': 'text'}}
    elif damage == 'extra_field': call['response']['output'][1]['unexpected'] = 'not supported'
    elif damage == 'bad_phase': call['response']['output'].append({'type': 'message', 'role': 'assistant', 'status': 'completed', 'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': 'Final'}]})
    else: body['tools'][0]['name'] = ''
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)


def test_cap_truncation_tokens_are_excluded_and_coverage_matches_target_admission():
    example = cap_example('openrouter.responses')
    report = training_thinking.coverage([example])
    assert report['attempts'][0]['returned'] == 'full_thinking'
    assert report['attempts'][0]['target_omission_reason'] == 'inferred_output_cap_json_eof'
    trace = training_thinking.readable_trace(example)
    assert trace['calls'][0]['target_omission_reason'] == 'inferred_output_cap_json_eof'
    assert trace['calls'][0]['reasoning_coverage']['finish_reason'] == 'completed'
    assert example['provider_calls'][0]['response']['status'] == 'completed'
    assert report['trajectories']['ineligible'] == 0
    assert report['trajectories']['selected_research_full_thinking'] == 1
    expected = sum(training_thinking.attempt_coverage(c, 'miner', example['configuration']['reasoning_route_registry'])['tokens_reported'] for c in example['provider_calls'][1:])
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['counts']['completed_untruncated_verified_reasoning_tokens'] == expected


def test_all_cap_truncated_targets_count_one_incomplete_icp_at_the_existing_floor():
    from tests.lab_arena.training_eligibility_test import ten_selected_examples
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    example['provider_calls'] = [call]
    artifact = example['worker_observation_artifacts']['run']
    payload = artifact['payload']
    payload['response_receipts'] = [row for row in payload['response_receipts'] if row['call_identity'] == call['call_identity']]
    payload['agent_observations'] = [row for row in payload['agent_observations'] if row.get('data', {}).get('call_identity') == call['call_identity']]
    artifact['payload_sha256'] = contracts.document_hash(payload)
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    examples = ten_selected_examples()
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 0
        seal(other)
    examples.pop('run-9')
    examples['run'] = example
    result = training_eligibility.assess_capture(submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    coverage = training_thinking.coverage([example])
    assert coverage['generated_thinking_texts'] == 1  # Original exposed thinking stays inventoried.
    for field in ('with_full_thinking', 'complete_full_thinking', 'selected_research_full_thinking'):
        assert coverage['trajectories'][field] == 0


def test_eof_after_json_whitespace_is_still_the_actual_end_of_argument_bytes():
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    call['response']['output'][1]['arguments'] = '{"company":   '
    refresh_response_links(example, 0)
    assert training_thinking.verified_cap_truncated_target(example, call)


@pytest.mark.parametrize('tail', ['reasoning', 'empty', 'whitespace', 'empty_parts', 'mixed'])
def test_inferred_cap_cut_can_precede_only_reasoning_and_empty_scaffolding(tail):
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    reasoning = deepcopy(call['response']['output'][0])
    reasoning['id'] = 'trailing-reasoning'
    message = {'type':'message', 'id':'trailing-empty', 'status':'completed', 'role':'assistant',
               'content':[] if tail == 'empty_parts' else [{'type':'output_text', 'text':' \n\t' if tail == 'whitespace' else ''}]}
    call['response']['output'].extend([reasoning, message] if tail == 'mixed' else [reasoning if tail == 'reasoning' else message])
    refresh_response_links(example, 0)
    before = deepcopy(example)
    assert training_thinking.verified_cap_truncated_target(example, call)
    result = training_eligibility.assess_capture(submission(), {'run':example}, ['run'], enabled=True)
    assert result['eligible'], result
    assert result['counts']['truncated_model_calls'] == 1
    assert result['counts']['completed_untruncated_verified_reasoning_tokens'] == 17
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(row['source_refs'][0]['call_identity'] != call['call_identity'] for row in records)
    assert example == before


@pytest.mark.parametrize('cut_index,complete_index,expected', [(1,0,True),(0,1,False),(1,1,False),(1,None,False)])
def test_inferred_chat_cap_cut_uses_complete_unique_provider_indexes(cut_index, complete_index, expected):
    example = cap_example('openrouter.chat')
    call = example['provider_calls'][0]
    tools = call['response']['choices'][0]['message']['tool_calls']
    cut = tools[0]
    cut['index'] = cut_index
    complete = deepcopy(cut)
    complete.update(id='complete-other')
    complete['function']['arguments'] = '{}'
    if complete_index is None:
        complete.pop('index')
    else:
        complete['index'] = complete_index
    tools.append(complete)
    refresh_response_links(example, 0)
    assert training_thinking.verified_cap_truncated_target(example, call) is expected


@pytest.mark.parametrize('tail', [
    {'type':'reasoning', 'status':'incomplete', 'content':[]},
    {'type':'reasoning', 'status':'completed', 'content':[], 'unknown':'data'},
    {'type':'message', 'status':'completed', 'role':'assistant', 'content':[], 'unknown':'data'},
    {'type':'message', 'status':'incomplete', 'role':'assistant', 'content':[]},
    {'type':'message', 'status':'completed', 'role':'user', 'content':[]},
    {'type':'message', 'status':'completed', 'role':'assistant',
     'content':[{'type':'output_text', 'text':'', 'annotations':[{'type':'url_citation', 'url':'https://example.com', 'title':'Source'}]}]},
])
def test_cap_cut_never_excuses_invalid_or_nonempty_metadata_in_trailing_frames(tail):
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    call['response']['output'].append(deepcopy(tail))
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    result = training_eligibility.assess_capture(submission(), {'run':example}, ['run'], enabled=True, minimum_coverage_percent=80)
    assert not result['eligible']


def test_citation_after_cut_is_not_empty_even_when_hosted_search_is_declared():
    from tests.lab_arena.hosted_search_continuation_test import hosted_request
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'].extend(hosted_request()['tools'])
    body['max_tool_calls'] = hosted_request()['max_tool_calls']
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    call['response']['output'].append({'type':'message', 'status':'completed', 'role':'assistant',
        'content':[{'type':'output_text', 'text':'', 'annotations':[
            {'type':'url_citation', 'url':'https://example.com', 'title':'Source'}]}]})
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_valid_object_json_at_the_exact_cap_remains_usable(api):
    example = cap_example(api)
    call = example['provider_calls'][0]
    if api.endswith('chat'):
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"company":"saved"}'
    else:
        call['response']['output'][1]['arguments'] = '{"company":"saved"}'
    refresh_response_links(example, 0)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    assert training_steps._candidate(example, call)
    assert training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('new_policy', [False, True])
def test_excessive_argument_json_nesting_is_rejected_without_crashing(api, new_policy):
    import sys
    example = cap_example(api)
    call = example['provider_calls'][0]
    depth = max(1500, sys.getrecursionlimit() + 100)
    arguments = '{"company":' + '[' * depth + '0' + ']' * depth + '}'
    if api.endswith('chat'):
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
    else:
        call['response']['output'][1]['arguments'] = arguments
    if not new_policy:
        example['configuration'].pop('training_maximum_missing_reasoning_percent')
    refresh_response_links(example, 0)
    assert not training_steps._arguments(arguments)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    with pytest.raises(ValueError, match='tool_call_shape_unsupported'):
        training_steps._candidate(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits['train']) + len(splits['eval']) == 2
    if new_policy:
        assert training_thinking.coverage([example])['trajectories']['ineligible'] == 1


@pytest.mark.parametrize('api,trailing', [
    ('openrouter.chat', 'tool_call'),
    ('openrouter.responses', 'tool_call'),
    ('openrouter.responses', 'message'),
])
def test_inferred_cap_cut_cannot_precede_retained_complete_output(api, trailing):
    example = cap_example(api)
    call = example['provider_calls'][0]
    if api == 'openrouter.chat':
        tools = call['response']['choices'][0]['message']['tool_calls']
        following = deepcopy(tools[0])
        following['id'] = 'following-call'
        following['function']['arguments'] = '{}'
        tools.append(following)
    else:
        output = call['response']['output']
        if trailing == 'tool_call':
            following = deepcopy(output[-1])
            following.update(id='following-item', call_id='following-call', arguments='{}')
        elif trailing == 'message':
            following = {'type': 'message', 'role': 'assistant', 'status': 'completed',
                         'content': [{'type': 'output_text', 'text': 'Later retained output.'}]}
        output.append(following)
    refresh_response_links(example, 0)
    original = deepcopy(example)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    with pytest.raises(ValueError, match='^tool_call_shape_unsupported$'):
        training_steps._candidate(example, call)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True, minimum_coverage_percent=80)
    assert not result['eligible'] and 'selected_run_evidence_invalid' in result['counts']['rejected_runs'][0]['reasons']
    report = training_thinking.coverage([example])
    assert report['trajectories']['ineligible'] == 1
    assert 'target_omission_reason' not in report['attempts'][0]
    assert 'target_omission_reason' not in training_thinking.readable_trace(example)['calls'][0]
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == 2
    assert example == original


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('earlier_unfinished', [False, True])
def test_only_the_final_argument_can_be_an_inferred_cap_cut(api, earlier_unfinished):
    example = cap_example(api)
    call = example['provider_calls'][0]
    if api == 'openrouter.chat':
        calls = call['response']['choices'][0]['message']['tool_calls']
        earlier = deepcopy(calls[-1])
        earlier['id'] = 'earlier-call'
        earlier['index'] = 0
        calls[-1]['index'] = 1
        if not earlier_unfinished:
            earlier['function']['arguments'] = '{}'
        calls.insert(0, earlier)
    else:
        output = call['response']['output']
        earlier = deepcopy(output[-1])
        earlier.update(id='earlier-item', call_id='earlier-call')
        if not earlier_unfinished:
            earlier['arguments'] = '{}'
        output.insert(-1, earlier)
    refresh_response_links(example, 0)
    original = deepcopy(example)
    assert training_thinking.verified_cap_truncated_target(example, call) is (not earlier_unfinished)
    result = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'] is (not earlier_unfinished)
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    rows = strict['train'] + strict['eval']
    assert len(rows) == 2
    assert all(row['source_refs'][0]['call_identity'] != call['call_identity'] for row in rows)
    assert example == original


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_explicit_provider_length_stop_does_not_depend_on_inferred_cut_position(api):
    example = cap_example(api)
    call = example['provider_calls'][0]
    if api == 'openrouter.chat':
        choice = call['response']['choices'][0]
        following = deepcopy(choice['message']['tool_calls'][0])
        following['id'] = 'following-call'
        following['function']['arguments'] = '{}'
        choice['message']['tool_calls'].append(following)
        choice['finish_reason'] = 'length'
    else:
        call['response']['output'].append({'type': 'message', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'Retained output before explicit length stop.'}]})
        call['response'].update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    refresh_response_links(example, 0)
    original = deepcopy(example)
    assert not training_thinking.verified_cap_truncated_target(example, call)
    with pytest.raises(ValueError, match='^response_not_complete$'):
        training_steps._candidate(example, call)
    assert training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    rows = strict['train'] + strict['eval']
    assert len(rows) == 2
    assert all(row['source_refs'][0]['call_identity'] != call['call_identity'] for row in rows)
    assert example == original
