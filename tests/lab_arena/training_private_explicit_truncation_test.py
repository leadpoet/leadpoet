"""Private discarded source excerpts cannot conceal a proved output-cap stop."""
from copy import deepcopy

import pytest

from lab_arena import contracts, operations, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


def private_truncated_example():
    example = cap_example('openrouter.responses')
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    call = example['provider_calls'][0]
    body = call['request']['outbound']['body']
    body['tools'] = [{'type': 'openrouter:web_search', 'parameters': {
        'engine': 'native', 'max_uses': 1, 'max_total_results': 1}}]
    body['max_tool_calls'] = operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS
    body['input'] = [{'role': 'user', 'content': 'Review the retained company.'}]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    call['response'].update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    call['response']['output'][1] = {
        'type': 'message', 'id': 'partial', 'role': 'assistant', 'status': 'incomplete',
        'content': [{'type': 'output_text', 'text': 'Partial retained answer.', 'annotations': [{
            'type': 'url_citation', 'url': 'https://example.com/public', 'title': 'Public source',
            'content': 'An unrelated source example: api_key=[REDACTED]'}]}]}
    call['response_privacy_redactions'] = ['credential']
    call['response_receipts'][0]['privacy_redactions'] = ['credential']
    call['evidence_gaps'] = ['response_privacy_nonexact']
    example['completeness'].update(complete=False, incomplete_reasons=['response_privacy_nonexact'])
    example['completeness']['dimensions']['research_process'] = False
    example['completeness']['reasons_by_dimension']['research_process'] = ['response_privacy_nonexact']
    refresh_response_links(example, 0)
    return seal(example)


def assess(example):
    return training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)


def test_private_explicit_truncation_is_discarded_without_losing_other_pairs():
    example = private_truncated_example()
    original = deepcopy(example)
    result = assess(example)
    assert result['eligible'], result
    counts = result['counts']['reasoning_by_run'][0]
    assert counts['truncated_model_calls'] == 1
    assert counts['missing_reasoning_tokens'] == 0
    registry = example['configuration']['reasoning_route_registry']
    expected_tokens = sum(training_thinking.attempt_coverage(c, 'miner', registry)['tokens_reported']
                          for c in example['provider_calls'][1:])
    assert counts['completed_untruncated_verified_reasoning_tokens'] == expected_tokens
    splits, manifest = training_thinking.build(
        [example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    omitted = example['provider_calls'][0]['call_identity']
    assert all(ref['call_identity'] != omitted for r in records for ref in r['source_refs'])
    assert manifest['coverage']['trajectories']['ineligible'] == 0
    assert example == original


def test_discarded_private_turn_cannot_complete_an_icp_without_full_thinking():
    example = private_truncated_example()
    example['provider_calls'] = example['provider_calls'][:1]
    seal(example)
    assert not assess(example)['eligible']
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not splits['train'] + splits['eval']


@pytest.mark.parametrize('damage', [
    'completed', 'unknown_reason', 'missing_reason', 'redacted_request', 'request_hash',
    'host_hash', 'host_status', 'response_hash', 'route', 'input_history', 'later_private_input', 'legacy',
])
def test_private_truncation_never_excuses_bad_proof_or_later_input(damage):
    example = private_truncated_example()
    call = example['provider_calls'][0]
    if damage == 'completed':
        call['response'].update(status='completed', incomplete_details=None)
        call['response']['output'][1]['status'] = 'completed'
        refresh_response_links(example, 0)
    elif damage == 'unknown_reason':
        call['response']['incomplete_details']['reason'] = 'unknown'
        refresh_response_links(example, 0)
    elif damage == 'missing_reason':
        call['response'].pop('incomplete_details')
        refresh_response_links(example, 0)
    elif damage == 'redacted_request':
        call['request_capture'].update(redacted=True, redactions=['credential'])
    elif damage == 'request_hash':
        call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_hash':
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'host_status':
        example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['status'] = 201
    elif damage == 'response_hash':
        call['response_receipts'][0]['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'route':
        call['reasoning_capture']['provider_route'] = 'wrong-route'
    elif damage == 'input_history':
        body = call['request']['outbound']['body']
        body['input'] = [{'type': 'function_call_output', 'call_id': 'missing', 'output': 'unbound'}]
        call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    elif damage == 'later_private_input':
        later = example['provider_calls'][1]
        later['request_capture'].update(redacted=True, redactions=['credential'])
        later['evidence_gaps'] = ['request_privacy_nonexact']
    else:
        example['configuration'].pop('training_maximum_missing_reasoning_percent')
    seal(example)
    assert not assess(example)['eligible']
