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
