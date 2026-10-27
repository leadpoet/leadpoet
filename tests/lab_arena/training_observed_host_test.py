"""Historical provider-visible pairs need host evidence, never invented reads."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, submission
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import fixture, response_fixture, seal


def historical_host_example(api='openrouter.chat'):
    example = fixture() if api == 'openrouter.chat' else response_fixture()
    example['configuration'] = {}
    example = without_client_callbacks(example)
    example.pop('configuration', None)
    return seal(example)


def rows(example):
    splits, manifest = training_steps.build([example])
    return splits['train'] + splits['eval'], manifest


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('role', ['miner', 'baseline'])
def test_historical_host_pair_exports_without_changing_policy_or_claiming_consumption(api, role):
    example = historical_host_example(api)
    example['role'] = role
    seal(example)
    before = deepcopy(example)
    candidates, manifest = rows(example)
    assert len(candidates) == 1, manifest['counts']
    candidate = candidates[0]
    assert candidate['input'] == example['provider_calls'][0]['request']['outbound']['body']
    assert candidate['source_refs'][0]['capture_scope'] == 'provider_visible_request_result'
    assert candidate['source_refs'][0]['client_consumption'] == 'not_proved'
    assert manifest['training_permission']['status'] == 'unconfirmed'
    assert example == before
    with pytest.raises(ValueError, match='agent_read_unverified'):
        training_steps._candidate(example, example['provider_calls'][0])
    strict, _ = training_thinking.build([example])
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('damage', ['missing_snapshot', 'payload_hash', 'write_state', 'write_status',
    'write_length', 'write_hash', 'write_identity', 'receipt_hash', 'request_hash', 'outbound_hash',
    'request_privacy', 'response_privacy', 'ack_claim', 'ack_ref'])
def test_historical_host_pair_keeps_exactness_and_existing_ack_checks(damage):
    example = historical_host_example()
    call = example['provider_calls'][0]
    artifact = example['worker_observation_artifacts']['run']
    write = artifact['payload']['response_receipts'][0]
    if damage == 'missing_snapshot': example['worker_observation_artifacts'] = {}
    elif damage == 'payload_hash': artifact['payload_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'write_state': write['write_state'] = 'socket_write_failed'
    elif damage == 'write_status': write['status'] = 201
    elif damage == 'write_length': write['body_bytes'] += 1
    elif damage == 'write_hash': write['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'write_identity': write['call_identity'] = 'other'
    elif damage == 'receipt_hash': call['response_receipts'][0]['response_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'request_hash': call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'request_privacy': call['request_capture'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'response_privacy': call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'ack_claim': call['response_provenance']['agent_receipt'] = 'reported_exact_body_received'
    elif damage == 'ack_ref': call['response_provenance']['agent_receipt_refs'] = [{'id': 'fake'}]
    if damage in {'write_state', 'write_status', 'write_length', 'write_hash', 'write_identity'}:
        artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    assert rows(seal(example))[0] == []


def test_generic_host_pair_does_not_relax_legacy_full_thinking_capture_or_champion_gate():
    # Keep the exact approved historical route, but remove client callbacks.
    # Generic delivery evidence must not change the frozen strict rules.
    example = without_client_callbacks(captured_example())
    example['configuration'].pop('training_maximum_missing_reasoning_percent')
    seal(example)
    before = deepcopy(example)
    candidates, _ = training_steps.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(candidates['train']) + len(candidates['eval']) > 0
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    assert not training_eligibility.assess_capture(
        submission(), {'run': example}, ['run'], enabled=True)['eligible']
    assert example == before


def test_generic_host_delivery_keeps_candidate_identity_and_split_groups():
    original = fixture()
    candidate = training_steps._candidate(original, original['provider_calls'][0])
    expected_id = contracts.document_hash({key: candidate[key] for key in ('format', 'input', 'target')})
    historical = historical_host_example()
    split, _ = training_steps.build([original, historical])
    assert len(split['train']) + len(split['eval']) == 1
    result = (split['train'] + split['eval'])[0]
    assert result['candidate_id'] == expected_id
    assert len(result['source_refs']) == 2
    assert {ref['example_hash'] for ref in result['source_refs']} == {
        original['example_hash'], historical['example_hash']}
