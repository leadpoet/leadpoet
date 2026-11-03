"""Successful host writes cannot be failed attempts or skip target validation."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_admission_refusal_test import refusal_example
from tests.lab_arena.training_call_semantics_test import _failed_example
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_hidden_retry_test import recovered_example
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_unavailable_upstream_coverage_test import unavailable_upstream


def success_example(operation, *, legacy=False):
    example = cap_example(operation)
    if legacy:
        example['configuration'].pop('training_maximum_missing_reasoning_percent')
    else:
        example = without_client_callbacks(example)
    return seal(example)


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('legacy', [False, True])
def test_referenced_success_does_not_enter_expensive_failed_attempt_proof(operation, legacy):
    example = success_example(operation, legacy=legacy)
    before = deepcopy(example)
    with patch.object(training_steps, '_exact', wraps=training_steps._exact) as exact:
        assert not training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    assert exact.call_count == 0
    assert example == before


@pytest.mark.parametrize('damage', ['missing_snapshot', 'missing_write', 'duplicate_write',
    'ref_ordinal', 'ref_run', 'ref_hash', 'write_state', 'write_identity', 'write_operation',
    'write_sequence', 'status_missing', 'status_string', 'status_bool', 'body_hash', 'body_bytes'])
def test_ambiguous_or_malformed_write_keeps_existing_failure_checks(damage):
    example = success_example('openrouter.chat')
    call = example['provider_calls'][0]
    artifact = example['worker_observation_artifacts']['run']
    writes = artifact['payload']['response_receipts']
    write = writes[0]
    ref = call['response_provenance']['worker_receipt_ref']
    if damage == 'missing_snapshot': example['worker_observation_artifacts'].pop('run')
    elif damage == 'missing_write': writes.pop(0)
    elif damage == 'duplicate_write': writes.append(deepcopy(write))
    elif damage == 'ref_ordinal': ref['ordinal'] = None
    elif damage == 'ref_run': ref['run_id'] = 'other'
    elif damage == 'ref_hash': ref['payload_sha256'] = contracts.hash_bytes(b'other')
    elif damage == 'write_state': write['write_state'] = 'socket_write_failed'
    elif damage == 'write_identity': write['call_identity'] = 'other'
    elif damage == 'write_operation': write['operation_id'] = 'openrouter.responses'
    elif damage == 'write_sequence': write['action_sequence'] += 1
    elif damage == 'status_missing': write.pop('status')
    elif damage == 'status_string': write['status'] = '200'
    elif damage == 'status_bool': write['status'] = True
    elif damage == 'body_hash': write['body_sha256'] = 'invalid'
    elif damage == 'body_bytes': write['body_bytes'] = None
    before = deepcopy(example)
    with patch.object(training_steps, '_exact', wraps=training_steps._exact) as exact:
        assert not training_thinking.verified_failed_attempt(example, call)
    assert exact.call_count > 0
    assert example == before


@pytest.mark.parametrize('status', [402, 404, 429, 500, 502, 503, 504])
@pytest.mark.parametrize('legacy', [False, True])
def test_non_success_write_keeps_existing_failure_checks(status, legacy):
    example = _failed_example(status)
    if not legacy:
        example = without_client_callbacks(example)
    before = deepcopy(example)
    with patch.object(training_steps, '_exact', wraps=training_steps._exact) as exact:
        result = training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    assert result is (status not in {402, 404})
    assert exact.call_count > 0
    assert example == before


@pytest.mark.parametrize('legacy', [False, True])
def test_host_written_error_survives_later_primary_success_status(legacy):
    example = _failed_example(503)
    if not legacy:
        example = without_client_callbacks(example)
    call = example['provider_calls'][0]
    call['response_status'] = 200
    seal(example)
    assert training_thinking.verified_failed_attempt(example, call)


@pytest.mark.parametrize('legacy', [False, True])
def test_later_success_receipt_does_not_replace_the_referenced_failed_body(legacy):
    example = _failed_example(503)
    if not legacy:
        example = without_client_callbacks(example)
    call = example['provider_calls'][0]
    earlier = call['response_receipts'][0]
    earlier['response'] = deepcopy(call['response'])
    earlier.pop('response_ref', None)
    later = {'choices': [{'message': {'role': 'assistant', 'content': 'A later response.'}}]}
    digest = contracts.document_hash(later)
    call.update(response=later, response_sha256=digest, response_status=200, terminal_entry_id=999)
    call['response_receipts'].append({'entry_id': 999, 'metadata': {'status': 200},
        'response_sha256': digest, 'response_ref': {'field': 'response', 'entry_id': 999},
        'wire_body_sha256': digest, 'wire_body_bytes': len(contracts.canonical_json(later).encode())})
    call['response_provenance']['selected_response_acknowledged'] = False
    seal(example)
    before = deepcopy(example)
    assert training_thinking.verified_failed_attempt(example, call)
    assert example == before


def test_hidden_retry_keeps_its_special_proof_before_any_status_filter():
    example = recovered_example()
    call = example['provider_calls'][0]
    with patch.object(training_thinking, 'verified_recovered_provider_retry', return_value=True) as recovered:
        with patch.object(training_steps, '_exact', wraps=training_steps._exact) as exact:
            assert training_thinking.verified_failed_attempt(example, call)
    recovered.assert_called_once_with(example, call, worker_proofs=None)
    assert exact.call_count == 0
    assert training_thinking.verified_failed_attempt(example, call)


@pytest.mark.parametrize('kind', ['admission', 'missing_upstream'])
def test_worker_error_without_primary_response_still_gets_full_failure_proof(kind):
    example = refusal_example()[0] if kind == 'admission' else unavailable_upstream()
    assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0])


@pytest.mark.parametrize('damage', ['request', 'response', 'host_hash', 'route'])
def test_fast_negative_does_not_admit_a_tampered_successful_target(damage):
    example = without_client_callbacks()
    call = example['provider_calls'][0]
    if damage == 'request': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'response': call['response']['changed'] = True
    elif damage == 'host_hash':
        example['worker_observation_artifacts']['run']['payload']['response_receipts'][0]['body_sha256'] = contracts.hash_bytes(b'wrong')
    else: call['reasoning_capture']['provider_route'] = 'wrong'
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']
