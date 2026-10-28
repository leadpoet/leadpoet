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


@pytest.mark.parametrize('kind', ['privacy', 'delivery', 'triple'])
@pytest.mark.parametrize('length_stop', [False, True])
def test_composed_cost_gaps_use_one_icp_and_never_become_strict_targets(kind, length_stop):
    example = composed(kind, length_stop=length_stop)
    examples = frozen_bank(example)
    result = assess(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    before = deepcopy(example)
    assert not assess(examples, percent=100)['eligible']
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    targets = {ref['run_id'] for split in ('train', 'eval') for row in strict[split] for ref in row['source_refs']}
    assert targets == set(examples) - {'run'}
    assert example['provider_calls'][0]['actual_microusd'] is None
    assert example == before
    examples['run-8'] = None
    assert assess(examples)['eligible']
    examples['run-7'] = None
    assert 'training_coverage_incomplete' in assess(examples)['reasons']


@pytest.mark.parametrize('kind', ['privacy', 'delivery', 'triple'])
@pytest.mark.parametrize('damage', ['response_hash', 'request_hash', 'outbound_hash', 'missing_host',
    'host_hash', 'route', 'source', 'input', 'output', 'unknown_privacy', 'cost_authority',
    'cost_policy', 'cost_reason', 'terminal_kind', 'terminal_missing', 'terminal_duplicate',
    'known_amount', 'unknown_gap', 'legacy', 'failed_body', 'attribution', 'prior_gap',
    'provider', 'host', 'unknown_context_privacy', 'complete_flag', 'cost_dimension',
    'research_dimension', 'mixed_privacy_reason', 'unknown_cost_policy',
    'attempt_status', 'attempt_kind', 'unattributed_call'])
def test_composed_gaps_cannot_hide_unproven_evidence(kind, damage):
    example = composed(kind, length_stop=True)
    examples = frozen_bank(example)
    call = example['provider_calls'][0]
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'response_hash': call['response_sha256'] = wrong
    elif damage == 'request_hash': call['request']['request_hash'] = wrong
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = wrong
    elif damage == 'missing_host':
        payload = example['worker_observation_artifacts']['run']['payload']
        payload['response_receipts'].pop(0)
        if 'response_receipt_count' in payload:
            payload['response_receipt_count'] -= 1
    elif damage == 'host_hash': example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['body_sha256'] = wrong
    elif damage == 'route': example['provider_calls'][2]['reasoning_capture']['provider_route'] = 'unapproved'
    elif damage == 'source': example['source']['sha256'] = wrong
    elif damage == 'input': example['input_snapshots'][example['score_run_id']]['input_sha256'] = wrong
    elif damage == 'output': example['output_identity']['document_hash'] = wrong
    elif damage == 'unknown_privacy':
        if kind == 'privacy': example['provider_calls'][1]['request_capture']['redactions'] = ['unknown']
        else: call['response_privacy_redactions'] = ['unknown']
    elif damage == 'cost_authority': example['competition_cost_eligibility']['source_rpc'] = 'untrusted'
    elif damage == 'cost_policy': example['competition_cost_eligibility']['policy'] = 'untrusted'
    elif damage == 'unknown_cost_policy':
        example['competition_cost_eligibility']['policy'] = 'unknown'
        example['configuration']['sourcing_cost_eligibility_policy'] = 'unknown'
    elif damage == 'cost_reason': example['competition_cost_eligibility']['reason'] = 'unknown'
    elif damage == 'terminal_kind': call['lifecycle'][-1]['entry_kind'] = 'settlement'
    elif damage == 'terminal_missing': call['lifecycle'].pop()
    elif damage == 'terminal_duplicate': call['lifecycle'].append(deepcopy(call['lifecycle'][-1]))
    elif damage == 'known_amount': call['actual_microusd'] = 0
    elif damage == 'provider': call['provider'] = 'wrong'
    elif damage == 'host': call['request']['outbound']['host'] = 'wrong.invalid'
    elif damage == 'unknown_context_privacy': example['privacy'].setdefault('context_redactions', []).append('unknown')
    elif damage == 'complete_flag': example['completeness']['complete'] = True
    elif damage == 'cost_dimension': example['completeness']['dimensions']['costs'] = True
    elif damage == 'research_dimension': example['completeness']['dimensions']['research_process'] = True
    elif damage == 'mixed_privacy_reason':
        if kind in ('privacy', 'triple'): example['provider_calls'][1]['request_capture']['redactions'].append('unknown')
        else: call['response_privacy_redactions'] = ['credential', 'unknown']
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent', None)
    elif damage == 'failed_body':
        if kind != 'privacy': example['provider_calls'][-1]['response_sha256'] = wrong
        else: call['response_receipts'][0]['wire_body_sha256'] = wrong
    elif damage == 'attribution':
        if kind != 'privacy': example['related_run_gap_attribution']['run']['cost_gaps'] = []
        else: example['completeness']['reasons_by_dimension']['costs'] = []
    elif damage in ('attempt_status', 'attempt_kind'):
        if kind != 'privacy':
            field, value = ('status', 'failed') if damage == 'attempt_status' else ('kind', 'score')
            example['attempts'][0][field] = value
            example['related_run_gap_attribution']['run'][field] = value
        else: example['completeness']['dimensions']['exact_inputs'] = False
    elif damage == 'unattributed_call':
        prior = deepcopy(call)
        prior.update(run_id='unattributed', evidence_gaps=[])
        example['provider_calls'].append(prior)
        if kind == 'privacy': example['completeness']['incomplete_reasons'].append('unknown_input')
    elif damage == 'prior_gap':
        prior = deepcopy(call)
        prior['run_id'] = 'earlier'
        example['provider_calls'].append(prior)
        example['attempts'].append({'run_id': 'earlier', 'kind': 'execute', 'status': 'failed'})
    else: example['completeness']['incomplete_reasons'].append('unknown_integrity_gap')
    refresh(example)
    assert not assess(examples)['eligible'], damage


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
    assert not strict['train'] and not strict['eval']
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
    assert not assess(frozen_bank(example))['eligible']


@pytest.mark.parametrize('composed_cost', [False, True])
@pytest.mark.parametrize('length_stop', [False, True])
def test_legacy_zero_percent_arguments_preserve_privacy_only_allowance(composed_cost, length_stop):
    example = selected_privacy_cascade_example()
    example['provider_calls'] = [call for call in example['provider_calls'] if call['run_id'] == 'run']
    if composed_cost:
        privacy = set(example['completeness']['incomplete_reasons'])
        _unsettled_selected_call(example, privacy=True)
        example['provider_calls'][0]['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
        proof = example['completeness']
        proof['incomplete_reasons'] = sorted(privacy | set(proof['incomplete_reasons']))
        proof['reasons_by_dimension']['research_process'] = sorted(
            privacy | set(proof['reasons_by_dimension']['research_process']))
    example['configuration'].pop('training_maximum_missing_reasoning_percent', None)
    if length_stop:
        example['provider_calls'][2]['response']['choices'][0]['finish_reason'] = 'length'
        refresh_response_links(example, 2)
    actual_redacted_request(example)
    examples = bank(example)
    result = training_eligibility.assess_capture(submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=0)
    assert result['eligible'] is (not composed_cost and not length_stop), result
