"""A failed judge's absent delivery must not erase exact execution targets."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import failed_score_gap_example, submission
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import seal


def failed_judge_delivery(*, mismatch=False):
    example = without_client_callbacks(failed_score_gap_example())
    call = example['provider_calls'][-1]
    call.update(call_identity=contracts.hash_bytes(b'failed judge call'), action_sequence=0,
                response_receipts=[], response_provenance={})
    payload = {'observations': [], 'response_receipts': [], 'agent_observations': []}
    if mismatch:
        payload['response_receipts'].append({
            'ordinal': 0, 'call_identity': contracts.hash_bytes(b'different call'),
            'action_sequence': 0, 'operation_id': call['operation_id'],
            'status': 200, 'body_sha256': contracts.hash_bytes(b'{}'),
            'body_bytes': 2, 'write_state': 'socket_written',
        })
    snapshot = {'payload': payload, 'payload_sha256': contracts.document_hash(payload)}
    errors = training_export._link_worker_receipts(
        {'run_id': 'failed-score', 'kind': 'score', 'status': 'failed'}, [call], snapshot,
        require_completed_phases=False, require_client_ack=False)
    proof = example['completeness']
    proof['incomplete_reasons'] = sorted(set(proof['incomplete_reasons']) | errors)
    proof['reasons_by_dimension']['research_process'] = sorted(
        set(proof['reasons_by_dimension']['research_process']) | errors)
    row = example['related_run_gap_attribution']['failed-score']
    row['research_process_gaps'] = sorted(set(row['research_process_gaps']) | errors)
    return seal(example), errors


def test_failed_judge_absent_delivery_preserves_exact_execution_targets():
    example, errors = failed_judge_delivery()
    assert not errors
    before = deepcopy(example)
    gate = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True, minimum_coverage_percent=80)
    assert gate['eligible'], gate
    assert gate['counts']['complete_runs'] == 1
    splits, _ = training_thinking.build([example])
    assert len(splits['train']) + len(splits['eval']) == 2
    assert example == before and not example['completeness']['complete']


def test_failed_judge_mismatched_delivery_does_not_erase_exact_accepted_data():
    example, errors = failed_judge_delivery(mismatch=True)
    assert errors == {'worker_response_receipt_link_invalid'}
    gate = training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True, minimum_coverage_percent=80)
    assert gate['eligible']
    splits, _ = training_thinking.build([example])
    assert len(splits['train']) + len(splits['eval']) == 2


@pytest.mark.parametrize('kind,status', [('score', 'failed'), ('score', 'accepted'), ('execute', 'failed')])
@pytest.mark.parametrize('mismatch', [False, True])
def test_related_attempt_absence_and_invalid_link_are_distinct(kind, status, mismatch):
    call = {'call_identity': 'call', 'operation_id': 'exa.contents', 'action_sequence': 0}
    payload = {'observations': [], 'agent_observations': [], 'response_receipts': []}
    if mismatch:
        payload['response_receipts'] = [{'call_identity': 'other', 'action_sequence': 0,
                                         'operation_id': 'exa.contents'}]
    errors = training_export._link_worker_receipts(
        {'run_id': 'related', 'kind': kind, 'status': status}, [call], {'payload': payload},
        require_completed_phases=False, require_client_ack=False)
    assert errors == ({'worker_response_receipt_link_invalid'} if mismatch else set())


@pytest.mark.parametrize('require_client_ack', [False, True])
def test_selected_execution_missing_host_remains_incomplete(require_client_ack):
    example = without_client_callbacks()
    payload = example['worker_observation_artifacts']['run']['payload']
    payload['response_receipts'] = []
    errors = training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute', 'status': 'accepted'}, example['provider_calls'],
        {'payload': payload}, require_completed_phases=True, require_client_ack=require_client_ack)
    expected = ('agent_response_receipts_unverified' if require_client_ack
                else 'worker_response_receipt_link_invalid')
    assert expected in errors
    if require_client_ack:
        assert 'worker_response_receipt_link_invalid' not in errors
    else:
        artifact = example['worker_observation_artifacts']['run']
        artifact['payload_sha256'] = contracts.document_hash(payload)
        seal(example)
        with pytest.raises(ValueError):
            training_steps._candidate(example, example['provider_calls'][0])


def test_historical_related_absence_keeps_existing_behavior():
    errors = training_export._link_worker_receipts(
        {'run_id': 'old-judge', 'kind': 'score', 'status': 'failed'},
        [{'call_identity': 'call'}],
        {'payload': {'observations': [], 'response_receipts': [], 'agent_observations': []}},
        require_completed_phases=False, require_client_ack=True)
    assert not errors
