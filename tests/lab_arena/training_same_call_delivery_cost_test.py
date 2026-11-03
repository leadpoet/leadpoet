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
def test_unknown_bill_and_failed_write_cannot_create_a_target_or_erase_other_pairs(bodyless, model):
    example = same_call_gap(bodyless=bodyless, model=model)
    examples = frozen_bank(example)
    original = deepcopy(example)
    failed = next(call for call in example['provider_calls'] if call['accounting_state'] == 'uncertain')
    result = assess(examples)
    assert result['eligible']
    assert result['counts']['complete_runs'] == (9 if model else 10)
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = strict['train'] + strict['eval']
    assert len(records) == 2
    assert all(ref['call_identity'] != failed['call_identity'] for row in records for ref in row['source_refs'])
    assert example == original
