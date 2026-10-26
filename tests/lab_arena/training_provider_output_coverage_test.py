"""Proven unusable provider outputs consume ICP coverage, never become targets."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import assess_ten, refresh_response_links, ten_selected_examples
from tests.lab_arena.training_steps_test import seal


def output_fault(kind='undeclared'):
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


@pytest.mark.parametrize('kind', ['undeclared', 'reasoning_only', 'message_and_undeclared'])
def test_proven_provider_output_fault_is_one_incomplete_icp(kind):
    example = output_fault(kind)
    before = deepcopy(example)
    with pytest.raises(ValueError, match='^(undeclared_tool_call|response_choice_ambiguous)$'):
        training_steps._candidate(example, example['provider_calls'][0])
    examples = bank(example)
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    runs = {ref['run_id'] for split in ('train', 'eval') for row in strict[split] for ref in row['source_refs']}
    assert runs == set(examples) - {'run'}
    assert not assess_ten(examples, percent=100)['eligible']
    examples['run-8'] = examples['run-7'] = None
    assert 'training_coverage_incomplete' in assess_ten(examples)['reasons']
    assert example == before


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
    assert not assess_ten(bank(example))['eligible']


@pytest.mark.parametrize('gap', ['cost', 'privacy', 'truncated', 'empty'])
@pytest.mark.parametrize('tamper', [False, True])
def test_provider_output_fault_composes_only_with_independently_proven_gaps(gap, tamper):
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
        example = output_fault()
        index = 0
        other = example['provider_calls'][1]
        if gap == 'truncated':
            other['response']['choices'][0]['finish_reason'] = 'length'
        else:
            message = other['response']['choices'][0]['message']
            for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
                message.pop(field, None)
        refresh_response_links(example, 1)
    refresh_response_links(example, index)
    if tamper:
        example['provider_calls'][2]['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    seal(example)
    result = assess_ten(bank(example))
    assert result['eligible'] is (not tamper), result
    assert result['counts']['incomplete_runs'] == 1


@pytest.mark.parametrize('damage', [None, 'attribution', 'input', 'output', 'source', 'route', 'cost', 'receipt'])
def test_output_fault_with_attributed_judge_failure_keeps_every_other_proof(damage):
    from tests.lab_arena.training_eligibility_test import failed_score_gap_example
    example = failed_score_gap_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][0]
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
    # The shared export path must keep its strict default.
    assert training_thinking.selected_research_evidence(example) is None
    result = assess_ten(bank(example))
    assert result['eligible'] is (damage is None), result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('damage', ['amount', 'uncertain', 'duplicate', 'missing', 'identity', 'bool_amount'])
def test_output_fault_requires_matching_settled_terminal(damage):
    example = output_fault()
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
    assert not assess_ten(bank(example))['eligible']


@pytest.mark.parametrize('tamper', [False, True])
def test_output_fault_with_source_timeout_and_unknown_cost_keeps_retained_proof(tamper):
    from tests.lab_arena.training_call_semantics_test import _source_timeout_and_cost_example
    from tests.lab_arena.training_eligibility_test import submission
    example = _source_timeout_and_cost_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    call = example['provider_calls'][1]
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
    assert result['eligible'] is (not tamper), result


@pytest.mark.parametrize('kind', ['undeclared', 'reasoning_only'])
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
    assert result['eligible'] is (not tamper), result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
