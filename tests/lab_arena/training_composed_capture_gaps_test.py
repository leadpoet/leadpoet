"""Independently proven gaps share the frozen ICP allowance, never targets."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_capture, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    _unsettled_selected_call, captured_example, refresh_response_links,
    selected_privacy_cascade_example, submission)
from tests.lab_arena.training_failed_delivery_coverage_test import failed_delivery, refresh
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_steps_test import seal


def composed(kind, *, length_stop=False):
    if kind == 'privacy':
        example = without_client_callbacks(selected_privacy_cascade_example())
        example['provider_calls'] = [call for call in example['provider_calls'] if call['run_id'] == 'run']
        privacy = set(example['completeness']['incomplete_reasons'])
        _unsettled_selected_call(example, privacy=True)
        proof = example['completeness']
        proof['incomplete_reasons'] = sorted(privacy | set(proof['incomplete_reasons']))
        proof['reasons_by_dimension']['research_process'] = sorted(
            privacy | set(proof['reasons_by_dimension']['research_process']))
    else:
        example = failed_delivery()
        authority = _unsettled_selected_call(captured_example())
        example['configuration']['sourcing_cost_eligibility_policy'] = authority['configuration']['sourcing_cost_eligibility_policy']
        example['competition_cost_eligibility'] = deepcopy(authority['competition_cost_eligibility'])
        proof = example['completeness']
        proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | {
            'provider_cost_unresolved', 'competition_cost_eligibility_unavailable'})
        proof['reasons_by_dimension']['research_process'].append('provider_cost_unresolved')
        proof['reasons_by_dimension']['costs'] = ['competition_cost_eligibility_unavailable', 'provider_cost_unresolved']
        proof['dimensions']['costs'] = False
        attribution = example['related_run_gap_attribution']['run']
        attribution['research_process_gaps'].append('provider_cost_unresolved')
        attribution['cost_gaps'] = ['provider_cost_unresolved']
    call = example['provider_calls'][0]
    call.update(accounting_state='uncertain', actual_microusd=None)
    call['evidence_gaps'] = sorted(set(call.get('evidence_gaps') or []) | {'provider_cost_unresolved'})
    call['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
    if kind == 'triple':
        privacy = {'context_privacy_nonexact', 'request_privacy_nonexact', 'response_privacy_nonexact'}
        example['privacy']['context_redactions'] = ['credential']
        call['response_privacy_redactions'] = ['credential']
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
        call['evidence_gaps'].append('response_privacy_nonexact')
        second = example['provider_calls'][1]
        second['request_capture'].update(redacted=True, redactions=['credential'])
        second['request']['privacy'] = {'redacted': True, 'redactions': ['credential']}
        second['request']['hashed_parameters'] = {'message': '[redacted:credential]'}
        second['evidence_gaps'] = ['request_privacy_nonexact']
        proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | privacy)
        proof['reasons_by_dimension']['research_process'] = sorted(set(proof['reasons_by_dimension']['research_process']) | privacy)
        attribution['research_process_gaps'] += ['request_privacy_nonexact', 'response_privacy_nonexact']
    if length_stop:
        for item in example['provider_calls']:
            item['response_provenance'].setdefault('agent_receipt_refs', [])
        # A separate delivered call still needs the exact target/route proof.
        example['provider_calls'][2]['response']['choices'][0]['finish_reason'] = 'length'
        refresh_response_links(example, 2)
    return refresh(example)


def frozen_bank(example):
    examples = bank(example)
    for item in examples.values():
        item['configuration']['training_maximum_missing_reasoning_percent'] = 1
        seal(item)
    return examples


def assess(examples, *, percent=80):
    return training_eligibility.assess_capture(submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=percent, maximum_missing_reasoning_percent=1)






def actual_redacted_request(example):
    from tests.lab_arena.training_export_test import Objects
    call = example['provider_calls'][1]
    objects = Objects()
    secret = call['request']['outbound']['body']['messages'][0]['content']
    original_body_hash = call['request']['outbound']['body_sha256']
    pointer = training_capture.RequestCapture(objects)(call['request'], secrets=(secret,))
    call['request_capture'] = pointer
    call['request'] = training_capture.read_request(objects, pointer)
    assert pointer['redacted'] and pointer['redactions'] == ['credential']
    assert call['request']['outbound']['body_sha256'] == original_body_hash
    assert contracts.document_hash(call['request']['outbound']['body']) != original_body_hash
    return seal(example)


@pytest.mark.parametrize('kind', ['privacy_only', 'privacy', 'triple'])
def test_real_request_capture_redaction_retains_original_outbound_digest(kind):
    example = composed(kind) if kind != 'privacy_only' else without_client_callbacks(selected_privacy_cascade_example())
    if kind == 'privacy_only':
        example['provider_calls'] = [call for call in example['provider_calls'] if call['run_id'] == 'run']
    actual_redacted_request(example)
    examples = frozen_bank(example)
    before = deepcopy(example)
    result = assess(examples)
    assert result['eligible'] and result['counts']['complete_runs'] == 9, result
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = strict['train'] + strict['eval']
    assert len(records) == 1
    assert all(ref['call_identity'] != example['provider_calls'][1]['call_identity']
               for row in records for ref in row['source_refs'])
    assert example == before


@pytest.mark.parametrize('damage', ['provider', 'method', 'host', 'path', 'digest_shape', 'unredacted_body'])
def test_real_request_redaction_keeps_route_and_nonredacted_hash_checks(damage):
    example = actual_redacted_request(composed('triple'))
    call = example['provider_calls'][1]
    outbound = call['request']['outbound']
    if damage == 'provider': call['provider'] = 'wrong'
    elif damage in ('method', 'host', 'path'): outbound[damage] = 'wrong'
    elif damage == 'digest_shape': outbound['body_sha256'] = 'invalid'
    else: example['provider_calls'][2]['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    assert not assess(frozen_bank(example), percent=100)['eligible']
