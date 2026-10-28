"""Optional diagnostics and exact nondispatch proof consume one incomplete ICP."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_failed_delivery_coverage_test import failed_delivery
from tests.lab_arena.training_optional_capture_test import summary
from tests.lab_arena.training_retry_snapshot_test import save_snapshot
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_undispatched_coverage_test import assess_bank, undispatched_values


def optional_undispatched(category):
    values = undispatched_values(operation='openrouter.responses', complete=True)
    run = values['runs'][0]
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    row = summary()
    row['id'] = max([item['id'] for item in payload['agent_observations']] + [-1]) + 1
    row['data']['value']['counts'] = {
        'responses_started': 1, 'responses_complete': 1, 'client_httpx': 1,
        'checkpoint_failure': 1, 'checkpoint_failure_' + category + '_transport_error': 1}
    buffer = training_observations.Buffer()
    assert buffer.append_agent_observation(row) == row['id']
    payload['agent_observations'].append(row)
    payload['agent_observation_count'] += 1
    payload.update(complete=False, failures=buffer.document()['failures'])
    assert payload['failures'] == [training_observations.OPTIONAL_CAPTURE_INCOMPLETE]
    assert training_observations._validate_payload(payload) == payload
    save_snapshot(values, run, payload)
    return values, training_export.assemble_example(**values)


def relink(example):
    artifact = example['worker_observation_artifacts'][example['run_id']]
    digest = contracts.document_hash(artifact['payload'])
    artifact['payload_sha256'] = digest
    for call in example['provider_calls']:
        provenance = call.get('response_provenance') or {}
        if provenance.get('worker_receipt_ref'):
            provenance['worker_receipt_ref']['payload_sha256'] = digest
        for ref in provenance.get('agent_receipt_refs') or []:
            ref['payload_sha256'] = digest
        if call.get('worker_observation_ref'):
            call['worker_observation_ref']['payload_sha256'] = digest
    return seal(example)


def assert_no_targets(example):
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('category', ['web_fetch', 'response_received'])
def test_optional_diagnostic_and_exact_nondispatch_consume_one_icp(category):
    values, example = optional_undispatched(category)
    call = example['provider_calls'][-1]
    assert training_thinking.verified_undispatched_transport(example, call)
    assert call['actual_microusd'] == 0 and 'response' not in call
    assert training_thinking._selected_input_output_binding(example) is not None
    for delivered in example['provider_calls'][:-1]:
        if delivered['run_id'] == example['run_id']:
            assert training_steps._host_result(example, delivered)
    before = deepcopy(example)
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert not assess_bank(values, example, percent=100)['eligible']
    assert_no_targets(example)
    assert example == before


@pytest.mark.parametrize('category', ['web_fetch', 'response_received'])
@pytest.mark.parametrize('damage', ['dispatch', 'missing_host_receipt', 'unknown_bill',
                                  'unproved_nondispatch', 'unknown_diagnostic'])
def test_optional_nondispatch_composition_keeps_required_proofs_hard(category, damage):
    values, example = optional_undispatched(category)
    call = example['provider_calls'][-1]
    payload = example['worker_observation_artifacts'][example['run_id']]['payload']
    if damage == 'dispatch':
        call['lifecycle'].insert(1, {**deepcopy(call['lifecycle'][0]), 'entry_kind': 'dispatch'})
    elif damage == 'missing_host_receipt':
        payload['response_receipts'].pop()
        payload['response_receipt_count'] -= 1
    elif damage == 'unknown_bill':
        call['actual_microusd'] = None
    elif damage == 'unproved_nondispatch':
        call['lifecycle'][-1]['accounting']['reason'] = 'unknown'
    elif damage == 'unknown_diagnostic':
        payload['agent_observations'][-1]['data']['value']['counts']['adapter_failure'] = 1
    relink(example)
    before = deepcopy(example)
    assert not assess_bank(values, example)['eligible']
    assert_no_targets(example)
    assert example == before


def test_optional_only_failed_write_cannot_use_nondispatch_exception():
    example = failed_delivery()
    payload = example['worker_observation_artifacts'][example['run_id']]['payload']
    payload['agent_observations'].append(summary())
    payload['agent_observation_count'] += 1
    # A failed host write must retain its generic failure; an optional-only
    # document cannot acquire the failed-write allowance without nondispatch.
    payload['failures'] = [training_observations.OPTIONAL_CAPTURE_INCOMPLETE]
    relink(example)
    assert not any(training_thinking.verified_undispatched_transport(example, call)
                   for call in example['provider_calls'])
    assert training_eligibility._failed_delivery_gap_verified(example) is None
    assert_no_targets(example)
