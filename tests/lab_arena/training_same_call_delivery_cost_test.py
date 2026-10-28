"""Exact failed writes and unknown bills on one call share one ICP allowance."""
import base64
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_composed_capture_gaps_test import assess, composed, frozen_bank
from tests.lab_arena.training_failed_delivery_coverage_test import failed_delivery, refresh
from tests.lab_arena.training_observations_test import observation


def same_call_gap(*, bodyless=False, model=False):
    example = failed_delivery(model=model, status=502)
    authority = composed('delivery')
    example['configuration']['sourcing_cost_eligibility_policy'] = authority['configuration']['sourcing_cost_eligibility_policy']
    example['competition_cost_eligibility'] = deepcopy(authority['competition_cost_eligibility'])
    proof = example['completeness']
    proof['incomplete_reasons'] += ['provider_cost_unresolved', 'competition_cost_eligibility_unavailable']
    proof['reasons_by_dimension']['research_process'].append('provider_cost_unresolved')
    proof['reasons_by_dimension']['costs'] = ['provider_cost_unresolved', 'competition_cost_eligibility_unavailable']
    proof['dimensions']['costs'] = False
    attr = example['related_run_gap_attribution']['run']
    attr['research_process_gaps'].append('provider_cost_unresolved')
    attr['cost_gaps'] = ['provider_cost_unresolved']
    call = next(call for call in example['provider_calls']
                if call['response_provenance']['delivery_to_worker'] == 'socket_write_failed')
    call.update(accounting_state='uncertain', actual_microusd=None, evidence_gaps=['provider_cost_unresolved'])
    call['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
    if bodyless:
        payload = example['worker_observation_artifacts']['run']['payload']
        write = next(row for row in payload['response_receipts'] if row['call_identity'] == call['call_identity'])
        raw = contracts.canonical_json(call['response']).encode()
        assert len(raw) == write['body_bytes'] and contracts.hash_bytes(raw) == write['body_sha256']
        row = observation(kind='nonstandard_response', ordinal=write['ordinal'],
            action=call['action_sequence'], identity=call['call_identity'], operation=call['operation_id'])
        row['request'] = {'admitted_parameters': deepcopy(call['request']['admitted_parameters']), 'timeout_ms': 1000}
        row['outcome'] = {'protocol': write['protocol'], 'response': {
            'status': 502, 'headers': {}, 'body_b64': base64.b64encode(raw).decode()}}
        payload['observations'] = [row]
        payload['observation_count'] = 1
        call['response_receipts'] = [{'entry_id': call['terminal_entry_id'], 'entry_kind': 'uncertain', 'created_at': None,
                                     'metadata': {}, 'response_unavailable': True}]
        for key in ('response', 'response_sha256', 'response_status'):
            call.pop(key)
    refresh(example)
    snapshot = example['worker_observation_artifacts']['run']
    assert training_observations._validate_payload(snapshot['payload']) == snapshot['payload']
    assert not training_export._link_worker_receipts({'run_id': 'run', 'kind': 'execute'},
        example['provider_calls'], snapshot, require_client_ack=False, configuration=example['configuration'])
    refresh(example)
    # These independent production proofs already pass before the eligibility fix.
    assert training_steps._host_response(example, call, write_state='socket_write_failed', allow_worker_error=True)
    assert training_thinking._terminal_accounting_matches(call, 'uncertain', 0)
    return example


@pytest.mark.parametrize('bodyless', [False, True])
@pytest.mark.parametrize('model', [False, True])
def test_exact_uncertain_failed_write_consumes_one_icp_without_training(bodyless, model):
    example = same_call_gap(bodyless=bodyless, model=model)
    examples = frozen_bank(example)
    before = deepcopy(example)
    result = assess(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert not assess(examples, percent=100)['eligible']
    strict, _ = training_thinking.build(list(examples.values()), training_steps.Rules(production_rounds_only=False))
    targets = {ref['run_id'] for split in ('train', 'eval') for row in strict[split] for ref in row['source_refs']}
    assert targets == set(examples) - {'run'}
    failed = next(call for call in example['provider_calls'] if call['accounting_state'] == 'uncertain')
    assert failed['actual_microusd'] is None
    assert example == before
    examples['run-8'] = None
    assert assess(examples)['eligible']
    examples['run-7'] = None
    assert 'training_coverage_incomplete' in assess(examples)['reasons']


@pytest.mark.parametrize('bodyless', [False, True])
@pytest.mark.parametrize('damage', ['known_cost', 'terminal_kind', 'terminal_missing', 'terminal_duplicate',
    'terminal_amount', 'host_hash', 'host_missing', 'request_hash', 'outbound_hash', 'provider',
    'request_privacy', 'response_privacy', 'upstream_hash', 'upstream_privacy', 'upstream_missing',
    'ack_claim', 'source', 'cost_authority', 'unknown_gap', 'legacy', 'legacy_zero'])
def test_uncertain_failed_write_does_not_hide_unproven_constituents(bodyless, damage):
    example = same_call_gap(bodyless=bodyless)
    examples = frozen_bank(example)
    call = next(call for call in example['provider_calls'] if call['accounting_state'] == 'uncertain')
    payload = example['worker_observation_artifacts']['run']['payload']
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'known_cost': call['actual_microusd'] = 0
    elif damage == 'terminal_kind': call['lifecycle'][-1]['entry_kind'] = 'settlement'
    elif damage == 'terminal_missing': call['lifecycle'].pop()
    elif damage == 'terminal_duplicate': call['lifecycle'].append(deepcopy(call['lifecycle'][-1]))
    elif damage == 'terminal_amount': call['lifecycle'][-1]['amount_microusd'] = 1
    elif damage == 'host_hash': payload['response_receipts'][-1]['body_sha256'] = wrong
    elif damage == 'host_missing': payload['response_receipts'].pop(); payload['response_receipt_count'] -= 1
    elif damage == 'request_hash': call['request']['request_hash'] = wrong
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = wrong
    elif damage == 'provider': call['provider'] = 'wrong'
    elif damage == 'request_privacy': call['request_capture']['redacted'] = True
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'upstream_hash': call['upstream_response_receipts'][0]['response_sha256'] = wrong
    elif damage == 'upstream_privacy': call['upstream_response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'upstream_missing': call.pop('upstream_response_receipts')
    elif damage == 'ack_claim': call['response_provenance']['selected_response_acknowledged'] = True
    elif damage == 'source': example['source']['sha256'] = wrong
    elif damage == 'cost_authority': example['competition_cost_eligibility']['source_rpc'] = 'untrusted'
    elif damage == 'unknown_gap': call['evidence_gaps'].append('unknown_gap')
    else: example['configuration'].pop('training_maximum_missing_reasoning_percent')
    refresh(example)
    if damage == 'legacy_zero':
        from tests.lab_arena.training_eligibility_test import assess_ten
        from tests.lab_arena.training_steps_test import seal
        for control in examples.values():
            if control is not example:
                control['configuration']['training_maximum_missing_reasoning_percent'] = 0
                seal(control)
        assert not training_eligibility._failed_delivery_call_verified(example, call)
        assert not assess_ten(examples)['eligible']
    else:
        assert not assess(examples)['eligible'], damage


@pytest.mark.parametrize('damage', ['observation_missing', 'observation_body', 'observation_privacy',
    'admitted_request', 'placeholder_missing', 'placeholder_extra', 'placeholder_kind',
    'unexpected_primary', 'unexpected_error', 'body_source', 'observation_ref'])
def test_bodyless_uncertain_write_requires_the_exact_existing_host_error_proof(damage):
    example = same_call_gap(bodyless=True)
    call = next(call for call in example['provider_calls'] if call['accounting_state'] == 'uncertain')
    payload = example['worker_observation_artifacts']['run']['payload']
    row = payload['observations'][0]
    if damage == 'observation_missing': payload['observations'] = []; payload['observation_count'] = 0
    elif damage == 'observation_body': row['outcome']['response']['body_b64'] = base64.b64encode(b'{}').decode()
    elif damage == 'observation_privacy': row['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'admitted_request': row['request']['admitted_parameters'] = {'changed': True}
    elif damage == 'placeholder_missing': call['response_receipts'] = []
    elif damage == 'placeholder_extra': call['response_receipts'][0]['wire_body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'placeholder_kind': call['response_receipts'][0]['entry_kind'] = 'settlement'
    elif damage == 'unexpected_primary': call['response'] = {'error': {'code': 'provider_unavailable'}}
    elif damage == 'unexpected_error':
        raw = contracts.canonical_json({'error': {'code': 'other_error'}}).encode()
        row['outcome']['response']['body_b64'] = base64.b64encode(raw).decode()
        payload['response_receipts'][-1].update(body_sha256=contracts.hash_bytes(raw), body_bytes=len(raw))
    elif damage == 'body_source': call['response_provenance']['worker_body_source'] = 'ledger_terminal'
    else: call['response_provenance']['worker_error_observation_ref']['ordinal'] += 1
    refresh(example)
    assert not assess(frozen_bank(example))['eligible'], damage
