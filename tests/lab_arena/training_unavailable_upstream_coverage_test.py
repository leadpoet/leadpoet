"""An exact host error can prove incomplete coverage without an upstream reply."""
import base64
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_call_semantics_test import _unsettled_worker_failure
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_eligibility_test import assess_ten
from tests.lab_arena.training_steps_test import seal


def unavailable_upstream():
    example = _unsettled_worker_failure()
    call = example['provider_calls'][0]
    call['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
    call.pop('response_status', None)
    call['evidence_gaps'].append('full_response_unavailable')
    call['response_receipts'] = [{'entry_id': call['terminal_entry_id'], 'entry_kind': 'uncertain',
                                  'metadata': {}, 'response_unavailable': True}]
    body = contracts.canonical_json({'error': {'code': 'provider_unavailable'}}).encode()
    payload = example['worker_observation_artifacts']['run']['payload']
    payload['observations'][0]['outcome']['response']['body_b64'] = base64.b64encode(body).decode()
    for row in payload['response_receipts'] + payload['agent_observations']:
        data = row.get('data', row)
        if data.get('call_identity') == call['call_identity']:
            data.update(body_sha256=contracts.hash_bytes(body), body_bytes=len(body))
    proof = example['completeness']
    proof['incomplete_reasons'].append('full_response_unavailable')
    proof['reasons_by_dimension']['research_process'].append('full_response_unavailable')
    rebind_payload(example)
    return seal(example)


def rebind_payload(example):
    artifact = example['worker_observation_artifacts']['run']
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for item in example['provider_calls']:
        provenance = item['response_provenance']
        refs = provenance.get('agent_receipt_refs', []) + [provenance[k] for k in
                ('worker_receipt_ref', 'worker_error_observation_ref') if k in provenance]
        for ref in refs:
            ref['payload_sha256'] = artifact['payload_sha256']


def test_exact_missing_upstream_response_keeps_unknown_charge_and_consumes_one_icp():
    example = unavailable_upstream()
    original = deepcopy(example)
    call = example['provider_calls'][0]
    assert training_steps._host_result(example, call, allow_worker_error=True)['write']['status'] == 502
    assert training_thinking.verified_failed_attempt(example, call)
    result = assess_ten(bank(example))
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 10 and result['counts']['incomplete_runs'] == 0
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(strict['train']) + len(strict['eval']) == 2
    assert call['accounting_state'] == 'uncertain' and call['actual_microusd'] is None
    assert example == original


@pytest.mark.parametrize('damage', ['primary_response', 'primary_hash', 'primary_status',
    'upstream_success', 'terminal_success', 'terminal_wire', 'terminal_ref', 'terminal_missing',
    'terminal_duplicate', 'terminal_identity', 'terminal_kind', 'settled', 'known_cost',
    'host_missing', 'host_failed', 'host_body', 'request', 'legacy', 'privacy', 'extra_gap'])
def test_no_upstream_waiver_requires_exact_failure_and_unchanged_unknown_bill(damage):
    example = unavailable_upstream()
    call = example['provider_calls'][0]
    receipt = call['response_receipts'][0]
    payload = example['worker_observation_artifacts']['run']['payload']
    if damage == 'primary_response': call['response'] = {'output': []}
    elif damage == 'primary_hash': call['response_sha256'] = contracts.hash_bytes(b'unknown')
    elif damage == 'primary_status': call['response_status'] = 200
    elif damage == 'upstream_success': call['upstream_response_receipts'] = [{'metadata': {'status': 200}, 'response_unavailable': True}]
    elif damage == 'terminal_success': receipt['metadata'] = {'status': 200, 'call_succeeded': True}
    elif damage == 'terminal_wire': receipt['wire_body_sha256'] = contracts.hash_bytes(b'unretained')
    elif damage == 'terminal_ref': receipt['response_ref'] = {'entry_id': 1, 'field': 'response'}
    elif damage == 'terminal_missing': call['response_receipts'] = []
    elif damage == 'terminal_duplicate': call['response_receipts'].append(deepcopy(receipt))
    elif damage == 'terminal_identity': receipt['entry_id'] += 1
    elif damage == 'terminal_kind': receipt['entry_kind'] = 'settlement'
    elif damage == 'settled': call['accounting_state'] = 'settlement'
    elif damage == 'known_cost': call['actual_microusd'] = 0
    elif damage == 'host_missing': payload['observations'] = []
    elif damage == 'host_failed': payload['response_receipts'][0]['write_state'] = 'write_failed'
    elif damage == 'host_body':
        raw = contracts.canonical_json({'error': {'code': 'other_failure'}}).encode()
        payload['observations'][0]['outcome']['response']['body_b64'] = base64.b64encode(raw).decode()
        for row in payload['response_receipts'] + payload['agent_observations']:
            data = row.get('data', row)
            if data.get('call_identity') == call['call_identity']:
                data.update(body_sha256=contracts.hash_bytes(raw), body_bytes=len(raw))
    elif damage == 'request': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent')
    elif damage == 'privacy': call['response_privacy_redactions'] = ['credential']
    else: call['evidence_gaps'].append('worker_observation_link_invalid')
    rebind_payload(example)
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call), damage
    assert not assess_ten(bank(example), percent=100)['eligible'], damage


@pytest.mark.parametrize('field', ['upstream_response_receipts', 'response_privacy_redactions'])
@pytest.mark.parametrize('value', [None, False, {}, ''])
def test_missing_upstream_proof_rejects_malformed_empty_metadata(field, value):
    example = unavailable_upstream()
    call = example['provider_calls'][0]
    call[field] = value
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']


def test_missing_actual_cost_field_is_not_a_recorded_unknown_charge():
    example = unavailable_upstream()
    call = example['provider_calls'][0]
    call.pop('actual_microusd')
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']


@pytest.mark.parametrize('damage', ['settled', 'charged', 'duplicate', 'missing', 'identity', 'bool_amount'])
def test_missing_upstream_exception_requires_matching_uncertain_terminal(damage):
    example = unavailable_upstream()
    call = example['provider_calls'][0]
    row = call['lifecycle'][-1]
    if damage == 'settled': row['entry_kind'] = 'settlement'
    elif damage == 'charged': row['amount_microusd'] = 100
    elif damage == 'duplicate': call['lifecycle'].append(deepcopy(row))
    elif damage == 'missing': call['lifecycle'].pop()
    elif damage == 'identity': row['entry_id'] += 1
    else: row['amount_microusd'] = False
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call)
    assert not assess_ten(bank(example), percent=100)['eligible']
