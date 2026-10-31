"""Proven unusable provider outputs consume ICP coverage, never become targets."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import assess_ten, refresh_response_links, ten_selected_examples
from tests.lab_arena.training_steps_test import seal


def output_fault(kind='undeclared'):
    if kind == 'empty_chat':
        example = cap_example('openrouter.chat')
        call = example['provider_calls'][0]
        call['response']['choices'][0] = {'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': None, 'reasoning': None,
            'reasoning_content': None, 'refusal': None}}
        call['response']['usage'].update(completion_tokens=33,
                                        completion_tokens_details={'reasoning_tokens': 0})
        refresh_response_links(example, 0)
        return seal(example)
    example = cap_example('openrouter.responses')
    call = example['provider_calls'][0]
    call['response']['usage']['output_tokens'] = 50
    output = call['response']['output']
    output[1]['arguments'] = '{}'
    if kind == 'reasoning_only':
        output.pop()
    else:
        output[1]['name'] = 'not_declared_by_request'
        if kind == 'message_and_undeclared':
            output.insert(1, {'type': 'message', 'role': 'assistant', 'status': 'completed',
                              'content': [{'type': 'output_text', 'text': 'I will look this up.'}]})
    refresh_response_links(example, 0)
    return seal(example)


def bank(example):
    examples = ten_selected_examples()
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 0
        seal(other)
    examples.pop('run-9')
    examples['run'] = example
    return examples


@pytest.mark.parametrize('kind', ['undeclared', 'reasoning_only', 'message_and_undeclared', 'empty_chat'])
def test_proven_provider_output_fault_is_one_incomplete_icp(kind):
    example = output_fault(kind)
    before = deepcopy(example)
    with pytest.raises(ValueError, match='^(undeclared_tool_call|response_choice_ambiguous|assistant_shape_unsupported)$'):
        training_steps._candidate(example, example['provider_calls'][0])
    examples = bank(example)
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    runs = {ref['run_id'] for split in ('train', 'eval') for row in strict[split] for ref in row['source_refs']}
    assert runs == set(examples)
    assert not assess_ten(examples, percent=100)['eligible']
    examples['run-8'] = examples['run-7'] = None
    assert 'training_coverage_incomplete' in assess_ten(examples)['reasons']
    assert example == before


@pytest.mark.parametrize('damage', ['request_hash', 'outbound_hash', 'history', 'host',
    'receipt', 'response_hash', 'route', 'privacy', 'receipt_privacy', 'hidden_reasoning',
    'unknown_cost', 'unknown_tokens', 'billed_reasoning', 'legacy', 'summary', 'unknown_shape',
    'missing_content', 'empty_string', 'nonempty_reasoning', 'tool_calls', 'refusal', 'length',
    'later_bad_call'])
def test_empty_chat_fault_requires_exact_zero_reasoning_output_and_all_existing_proofs(damage):
    example = output_fault('empty_chat')
    call = example['provider_calls'][0]
    message = call['response']['choices'][0]['message']
    if damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'history':
        call['request']['outbound']['body']['messages'].append({'role': 'tool', 'tool_call_id': 'missing', 'content': 'unpaired'})
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
    elif damage == 'host': call['response_provenance']['worker_receipt_ref']['ordinal'] = 999
    elif damage == 'receipt': call['response_receipts'] = []
    elif damage == 'response_hash': call['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'route': call['response_receipts'][0]['metadata']['reasoning_route_observation']['status'] = 'mismatched'
    elif damage == 'privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'receipt_privacy': call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'hidden_reasoning': call['response_privacy_redactions'] = ['hidden_reasoning']
    elif damage == 'unknown_cost': call['actual_microusd'] = None
    elif damage == 'unknown_tokens': call['response']['usage'].pop('completion_tokens_details')
    elif damage == 'billed_reasoning': call['response']['usage']['reasoning_tokens'] = 1
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent')
    elif damage == 'summary': message['reasoning_details'] = [{'type': 'reasoning.summary', 'summary': 'Only a summary'}]
    elif damage == 'unknown_shape': message['unknown'] = None
    elif damage == 'missing_content': message.pop('content')
    elif damage == 'empty_string': message['content'] = ''
    elif damage == 'nonempty_reasoning': message['reasoning'] = 'Retained reasoning'
    elif damage == 'tool_calls': message['tool_calls'] = [{'unexpected': True}]
    elif damage == 'refusal': message['refusal'] = 'Refused'
    elif damage == 'length': call['response']['choices'][0]['finish_reason'] = 'length'
    else:
        example['provider_calls'][1]['response']['choices'] = []
        refresh_response_links(example, 1)
    if damage in {'route', 'unknown_tokens', 'billed_reasoning', 'summary', 'unknown_shape',
                  'missing_content', 'empty_string', 'nonempty_reasoning', 'tool_calls', 'refusal', 'length'}:
        refresh_response_links(example, 0)
    seal(example)
    assert training_eligibility._verified_provider_output_fault(example, call) is (damage == 'later_bad_call')
    # Explicit provider length stops keep their existing incomplete-ICP path.
    if damage != 'length':
        assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('location', ['response', 'choice'])
@pytest.mark.parametrize('field,value', [
    ('output', [{'type': 'message', 'content': 'Other generated output'}]),
    ('error', {'code': 500, 'message': 'Failed'}),
    ('reasoning', 'Other reasoning'), ('reasoning_content', 'Other reasoning'),
    ('reasoning_details', [{'type': 'reasoning.summary', 'summary': 'Summary'}]),
    ('content', 'Other output'), ('text', 'Other output'), ('output_text', 'Other output'),
    ('message', {'content': 'Other output'}), ('delta', {'content': 'Other output'}),
])
def test_empty_chat_fault_rejects_output_or_reasoning_outside_the_message(location, field, value):
    example = output_fault('empty_chat')
    call = example['provider_calls'][0]
    target = call['response'] if location == 'response' else call['response']['choices'][0]
    target[field] = value
    refresh_response_links(example, 0)
    assert not training_eligibility._verified_provider_output_fault(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('location', [None, 'completion_tokens_details', 'output_tokens_details'])
@pytest.mark.parametrize('tokens', ['5', -1, True, False, 1.0, 0.0, None, 1])
def test_empty_chat_fault_requires_every_present_reasoning_counter_to_be_exact_zero(location, tokens):
    example = output_fault('empty_chat')
    call = example['provider_calls'][0]
    usage = call['response']['usage']
    target = usage if location is None else usage.setdefault(location, {})
    target['reasoning_tokens'] = tokens
    refresh_response_links(example, 0)
    assert not training_eligibility._verified_provider_output_fault(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('variant', ['absent_reasoning_aliases', 'standard_choice_metadata', 'all_zero_counters', 'top_level_counter'])
def test_empty_chat_fault_keeps_supported_empty_shapes(variant):
    example = output_fault('empty_chat')
    call = example['provider_calls'][0]
    choice = call['response']['choices'][0]
    if variant == 'absent_reasoning_aliases':
        for field in ('reasoning', 'reasoning_content', 'refusal'):
            choice['message'].pop(field)
    elif variant == 'standard_choice_metadata':
        choice.update(index=0, native_finish_reason='stop', logprobs=None)
    else:
        usage = call['response']['usage']
        usage['reasoning_tokens'] = 0
        if variant == 'all_zero_counters':
            usage['output_tokens_details'] = {'reasoning_tokens': 0}
        else:
            usage.pop('completion_tokens_details')
    refresh_response_links(example, 0)
    assert training_eligibility._verified_provider_output_fault(example, call)
    assert assess_ten(bank(example))['eligible']


@pytest.mark.parametrize('missing', [1, 2])
@pytest.mark.parametrize('empty_output', [False, True])
def test_empty_chat_omission_preserves_valid_zero_reasoning_answers_and_one_percent_budget(missing, empty_output):
    from tests.lab_arena.training_eligibility_test import submission
    example = output_fault('empty_chat')
    if not empty_output:
        example['provider_calls'][0]['response']['choices'][0]['message']['content'] = 'A valid answer.'
        refresh_response_links(example, 0)
        assert training_steps._candidate(example, example['provider_calls'][0])
        assert not training_eligibility._verified_provider_output_fault(example, example['provider_calls'][0])
    for index, tokens in ((1, missing), (2, 100 - missing)):
        call = example['provider_calls'][index]
        call['response']['usage']['completion_tokens_details']['reasoning_tokens'] = tokens
        if index == 1:
            for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
                call['response']['choices'][0]['message'].pop(field, None)
        refresh_response_links(example, index)
    examples = bank(example)
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 1
        seal(other)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 + int(not empty_output and missing == 1)
    row = next(row for row in result['counts']['reasoning_by_run'] if row['run_id'] == 'run')
    assert row['missing_reasoning_tokens'] == missing
    assert row['completed_untruncated_verified_reasoning_tokens'] == 100


@pytest.mark.parametrize('policy', ['eighty', 'historical_hundred', 'off'])
def test_empty_chat_publication_preserves_measured_scores_and_policy_switch(monkeypatch, policy):
    from tests.lab_arena.training_eligibility_publication_test import _export, _published, _ranking
    from tests.lab_arena.training_mixed_thinking_coverage_test import _ten_icp_publication
    harness = _ten_icp_publication(monkeypatch)
    configuration = harness[1]['configuration_doc']
    configuration['training_maximum_missing_reasoning_percent'] = 0
    examples = harness[4]
    examples['miner-9'] = _export('miner-9', 'miner')
    example = examples['miner-9']
    example['provider_calls'][0]['response'] = deepcopy(output_fault('empty_chat')['provider_calls'][0]['response'])
    refresh_response_links(example, 0)
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 0
        seal(other)
    if policy == 'historical_hundred':
        configuration.pop('training_minimum_icp_coverage_percent')
    elif policy == 'off':
        configuration.pop('reasoning_capture_policy')
    publication = _published(harness)
    miner, peer = (_ranking(publication, key) for key in ('miner', 'peer'))
    assert miner['final_score'] == 3.0 and peer['final_score'] == 2.0
    assert peer['eligible'] is True
    assert miner['eligible'] is (policy != 'historical_hundred')
    assert publication['king_decision']['winner_submission_id'] == ('peer' if policy == 'historical_hundred' else 'miner')
    if policy == 'eighty':
        assert miner['training_data_eligibility']['capture']['counts']['complete_runs'] == 9
    elif policy == 'off':
        assert 'training_data_eligibility' not in miner


@pytest.mark.parametrize('damage', ['request_hash', 'outbound_hash', 'host_hash', 'write',
    'response_hash', 'response_mismatch', 'route', 'privacy', 'receipt_privacy',
    'unknown_cost', 'negative_cost', 'gap', 'legacy', 'input_pair', 'reasoning_shape',
    'ambiguous_messages', 'malformed_arguments', 'later_bad_call'])
def test_output_fault_cannot_hide_bad_evidence_or_other_target_failures(damage):
    example = output_fault()
    call = example['provider_calls'][0]
    if damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_hash': call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'write':
        example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['write_state'] = 'write_failed'
    elif damage == 'response_hash': call['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response_mismatch':
        call['response'] = deepcopy(call['response'])
        call['response']['output'][0]['content'][0]['text'] = 'Different unreceived reasoning'
        call['response_sha256'] = contracts.document_hash(call['response'])
    elif damage == 'route': call['response_receipts'][0]['metadata']['reasoning_route_observation']['status'] = 'mismatched'
    elif damage == 'privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'receipt_privacy': call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'unknown_cost': call['actual_microusd'] = None
    elif damage == 'negative_cost': call['actual_microusd'] = -1
    elif damage == 'gap': call['evidence_gaps'] = ['full_response_unavailable']
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent')
    elif damage == 'input_pair':
        call['request']['outbound']['body']['input'].append({'type': 'function_call_output', 'call_id': 'missing', 'output': 'unpaired'})
    elif damage == 'reasoning_shape': call['response']['output'][0]['content'][0]['type'] = 'unrecognized'
    elif damage == 'ambiguous_messages':
        call['response']['output'] = [{'type': 'message', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': text}]} for text in ('One', 'Two')]
    elif damage == 'malformed_arguments': call['response']['output'][1]['arguments'] = '{garbage'
    else:
        other = example['provider_calls'][-1]
        other['response']['choices'] = []
        refresh_response_links(example, len(example['provider_calls']) - 1)
    if damage in {'route', 'reasoning_shape', 'ambiguous_messages', 'malformed_arguments'}:
        refresh_response_links(example, 0)
    if damage == 'input_pair':
        body = call['request']['outbound']['body']
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    seal(example)
    assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('gap', ['cost', 'privacy', 'truncated', 'empty'])
@pytest.mark.parametrize('tamper', [False, True])
@pytest.mark.parametrize('kind', ['undeclared', 'empty_chat'])
def test_provider_output_fault_composes_only_with_independently_proven_gaps(gap, tamper, kind):
    from tests.lab_arena.training_eligibility_test import _unsettled_selected_call, selected_privacy_cascade_example
    if gap == 'cost':
        example = cap_example('openrouter.responses')
        example['provider_calls'][0]['response']['output'][1]['arguments'] = '{}'
        refresh_response_links(example, 0)
        example = _unsettled_selected_call(example)
        index = 1
        function = example['provider_calls'][index]['response']['choices'][0]['message']['tool_calls'][0]['function']
        function['name'] = 'undeclared'
    elif gap == 'privacy':
        example = selected_privacy_cascade_example()
        example['configuration']['training_maximum_missing_reasoning_percent'] = 0
        index = 2
        call = example['provider_calls'][index]
        call['request']['outbound']['body']['tools'] = [{'type': 'function', 'function': {'name': 'declared', 'parameters': {'type': 'object'}}}]
        call['request']['outbound']['body_sha256'] = contracts.document_hash(call['request']['outbound']['body'])
        choice = call['response']['choices'][0]
        choice['finish_reason'] = 'tool_calls'
        choice['message']['tool_calls'] = [{'id': 'undeclared', 'type': 'function', 'function': {'name': 'undeclared', 'arguments': '{}'}}]
    else:
        example = output_fault(kind)
        index = 0
        other = example['provider_calls'][1]
        if gap == 'truncated':
            other['response']['choices'][0]['finish_reason'] = 'length'
        else:
            message = other['response']['choices'][0]['message']
            for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
                message.pop(field, None)
        refresh_response_links(example, 1)
    if kind == 'empty_chat' and gap in {'cost', 'privacy'}:
        example['provider_calls'][index]['response'] = deepcopy(output_fault(kind)['provider_calls'][0]['response'])
    refresh_response_links(example, index)
    if tamper:
        example['provider_calls'][2]['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    seal(example)
    result = assess_ten(bank(example))
    assert result['eligible'] and result['counts']['complete_runs'] == 9, result
    assert result['counts']['incomplete_runs'] == 1


@pytest.mark.parametrize('damage', [None, 'attribution', 'input', 'output', 'source', 'route', 'cost', 'receipt'])
@pytest.mark.parametrize('kind', ['undeclared', 'empty_chat'])
def test_output_fault_with_attributed_judge_failure_keeps_every_other_proof(damage, kind):
    from tests.lab_arena.training_eligibility_test import failed_score_gap_example
    example = failed_score_gap_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][0]
    if kind == 'empty_chat':
        call['response'] = deepcopy(output_fault(kind)['provider_calls'][0]['response'])
    else:
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'undeclared'
    refresh_response_links(example, 0)
    if damage == 'attribution': example['related_run_gap_attribution']['run']['research_process_gaps'] = ['full_response_unavailable']
    elif damage == 'input': example['input_snapshots']['score']['input_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'output': example['output_identity']['document_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'source': example['source']['sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'route': call['reasoning_capture']['provider_route'] = 'unapproved'
    elif damage == 'cost': call['actual_microusd'] = None
    elif damage == 'receipt': call['response_receipts'][0]['wire_body_sha256'] = contracts.hash_bytes(b'wrong')
    seal(example)
    assert (training_thinking.selected_research_evidence(example) is None) is (damage in {'input', 'output'})
    result = assess_ten(bank(example))
    assert result['eligible'] and result['counts']['complete_runs'] == 9, result
    assert not assess_ten(bank(example), percent=100)['eligible']
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == (0 if damage in {'input', 'output'} else 2)


@pytest.mark.parametrize('damage', ['amount', 'uncertain', 'duplicate', 'missing', 'identity', 'bool_amount'])
@pytest.mark.parametrize('kind', ['undeclared', 'empty_chat'])
def test_output_fault_requires_matching_settled_terminal(damage, kind):
    example = output_fault(kind)
    call = example['provider_calls'][0]
    row = call['lifecycle'][-1]
    if damage == 'amount': call['actual_microusd'] += 100
    elif damage == 'uncertain': row['entry_kind'] = 'uncertain'
    elif damage == 'duplicate': call['lifecycle'].append(deepcopy(row))
    elif damage == 'missing': call['lifecycle'].pop()
    elif damage == 'identity': row['entry_id'] += 1
    else: row['amount_microusd'] = True
    seal(example)
    assert not training_eligibility._verified_provider_output_fault(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('tamper', [False, True])
@pytest.mark.parametrize('kind', ['undeclared', 'empty_chat'])
def test_output_fault_with_source_timeout_and_unknown_cost_keeps_retained_proof(tamper, kind):
    from tests.lab_arena.training_call_semantics_test import _source_timeout_and_cost_example
    from tests.lab_arena.training_eligibility_test import submission
    example = _source_timeout_and_cost_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][1]
    if kind == 'empty_chat':
        call['response'] = deepcopy(output_fault(kind)['provider_calls'][0]['response'])
    else:
        call['response']['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'undeclared'
    refresh_response_links(example, 1)
    if tamper:
        call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    seal(example)
    examples = bank(example)
    source_ref = f"arena/{example['round_id']}/sources/{example['submission_id']}.tar.gz"
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
        unavailable_artifacts={'run': {'source': source_ref}})
    assert result['eligible'] and result['counts']['complete_runs'] == 9, result


@pytest.mark.parametrize('kind', ['undeclared', 'reasoning_only', 'empty_chat'])
def test_only_unusable_output_consumes_one_icp_without_counting_its_reasoning_tokens(kind):
    example = output_fault(kind)
    example['provider_calls'] = example['provider_calls'][:1]
    seal(example)
    result = assess_ten(bank(example))
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    row = next(row for row in result['counts']['reasoning_by_run'] if row['run_id'] == 'run')
    assert row['completed_untruncated_verified_reasoning_tokens'] == 0
    assert row['missing_reasoning_tokens'] == 0


@pytest.mark.parametrize('tamper', [False, True])
def test_chat_undeclared_tool_is_incomplete_coverage_with_exact_proof(tamper):
    example = cap_example('openrouter.chat')
    call = example['provider_calls'][0]
    function = call['response']['choices'][0]['message']['tool_calls'][0]['function']
    function.update(name='undeclared', arguments='{}')
    call['response']['usage']['completion_tokens'] = 50
    refresh_response_links(example, 0)
    with pytest.raises(ValueError, match='^undeclared_tool_call$'):
        training_steps._candidate(example, call)
    if tamper:
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    seal(example)
    result = assess_ten(bank(example))
    assert result['eligible'] and result['counts']['complete_runs'] == 9, result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == 2
