"""Repeated identical terminal bodies keep one consistent route observation."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena import full_thinking_saved_probe_test as saved_probe
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_steps_test import seal


def reconciled_example(*, callbacks=False):
    captured = {}
    decode = training_export._calls

    def record(rows, *args):
        captured.update(rows=deepcopy(rows), args=args)
        return decode(rows, *args)

    # Start with real saved synthetic-provider bytes, normal request capture,
    # the production decoder and its actual host-delivery fixture.
    with patch.object(saved_probe, '_calls', record):
        example = captured_example()
    rows = captured['rows']
    settled = deepcopy(rows[2])
    rows[2].update(entry_kind='uncertain', amount_microusd=100)
    settled.update(entry_id=4, entry_doc={'late_reconciliation': True,
        'openrouter_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 3})
    rows.append(settled)
    calls, gaps = decode(rows, *captured['args'])
    assert not gaps
    example['provider_calls'] = calls
    example['configuration']['training_maximum_missing_reasoning_percent'] = 0
    snapshot = example['worker_observation_artifacts']['run']
    if not callbacks:
        snapshot['payload']['agent_observations'] = []
        snapshot['payload_sha256'] = contracts.document_hash(snapshot['payload'])
    else:
        for row in snapshot['payload']['agent_observations']:
            row.setdefault('depends_on', [])
        snapshot['payload_sha256'] = contracts.document_hash(snapshot['payload'])
    errors = training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, calls, snapshot,
        require_completed_phases=False, require_client_ack=callbacks)
    assert not errors
    return seal(example)


@pytest.mark.parametrize('callbacks', [False, True])
def test_real_decoder_identical_reconciled_body_keeps_eligibility_and_strict_targets(callbacks):
    example = reconciled_example(callbacks=callbacks)
    before = deepcopy(example)
    call = example['provider_calls'][0]
    assert [row['entry_kind'] for row in call['lifecycle']] == ['reservation', 'dispatch', 'uncertain', 'settlement']
    assert len(call['response_receipts']) == 2
    assert len({row['response_sha256'] for row in call['response_receipts']}) == 1
    assert training_steps._host_result(example, call)
    assert training_steps._candidate(example, call)
    state = training_thinking.attempt_coverage(call, 'miner', example['configuration']['reasoning_route_registry'])
    assert state['provider_route_verified']
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert result['eligible'], result
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits['train']) + len(splits['eval']) == 2
    assert example == before


@pytest.mark.parametrize('damage', ['contradiction', 'missing', 'extra_field', 'bad_type', 'wrong_hash',
                                   'override_conflict', 'host_duplicate'])
def test_repeated_receipts_do_not_hide_conflicting_or_unbound_proof(damage):
    example = reconciled_example()
    call = example['provider_calls'][0]
    receipt = call['response_receipts'][1]
    observation = receipt['metadata']['reasoning_route_observation']
    if damage == 'contradiction': observation['selected_endpoint_provider_name'] = 'Other'
    elif damage == 'missing': receipt['metadata'].pop('reasoning_route_observation')
    elif damage == 'extra_field': observation['unexpected'] = True
    elif damage == 'bad_type': observation['selected_endpoint_provider_name'] = 7
    elif damage == 'wrong_hash':
        for row in call['response_receipts']:
            row['response_sha256'] = contracts.hash_bytes(b'foreign body')
    elif damage == 'override_conflict':
        call['route_observation'] = deepcopy(observation)
        call['route_observation']['selected_endpoint_provider_name'] = 'Other'
    else:
        artifact = example['worker_observation_artifacts']['run']
        artifact['payload']['response_receipts'].append(deepcopy(artifact['payload']['response_receipts'][0]))
        artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
        for row in example['provider_calls']:
            row['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    result = training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)
    assert not result['eligible']
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 2
    assert all(ref['call_identity'] != call['call_identity'] for row in records for ref in row['source_refs'])


def test_different_body_receipt_cannot_supply_the_selected_response_route():
    example = reconciled_example()
    call = example['provider_calls'][0]
    for receipt in call['response_receipts']:
        receipt['metadata'].pop('reasoning_route_observation')
    unrelated = deepcopy(call['response_receipts'][0])
    unrelated['entry_id'] = 999
    unrelated['response_sha256'] = contracts.hash_bytes(b'unrelated')
    unrelated['metadata']['reasoning_route_observation'] = deepcopy(
        captured_example()['provider_calls'][0]['response_receipts'][0]['metadata']['reasoning_route_observation'])
    call['response_receipts'].append(unrelated)
    assert not training_thinking.attempt_coverage(call, 'miner', example['configuration']['reasoning_route_registry'])['provider_route_verified']


def test_repeated_route_proof_does_not_disable_old_round_client_requirement():
    example = reconciled_example()
    example['configuration'].pop('training_maximum_missing_reasoning_percent')
    call = example['provider_calls'][0]
    assert training_thinking.attempt_coverage(call, 'miner', example['configuration']['reasoning_route_registry'])['provider_route_verified']
    with pytest.raises(ValueError, match='agent_read_unverified'):
        training_steps._candidate(seal(example), call)


def test_same_body_proof_does_not_borrow_a_conflicting_unrelated_body_route():
    example = reconciled_example()
    call = example['provider_calls'][0]
    unrelated = deepcopy(call['response_receipts'][0])
    unrelated['entry_id'] = 999
    unrelated['response_sha256'] = contracts.hash_bytes(b'unrelated body')
    unrelated['metadata']['reasoning_route_observation']['selected_endpoint_provider_name'] = 'Other'
    call['response_receipts'].append(unrelated)
    assert training_thinking.attempt_coverage(call, 'miner', example['configuration']['reasoning_route_registry'])['provider_route_verified']
