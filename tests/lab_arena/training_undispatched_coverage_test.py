"""An authenticated request that never dispatched consumes one incomplete ICP."""
from copy import deepcopy
from types import SimpleNamespace
import base64

import pytest

from lab_arena import contracts, runner, training_capture, training_eligibility, training_export, training_observations, training_reader, training_steps, training_thinking
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_retry_snapshot_test import retry_values, save_snapshot
from tests.lab_arena.training_steps_test import seal


def undispatched_values(*, local_error=True, http_error=False, operation='deepline.execute', complete=False):
    values, predecessor, _ = retry_values()
    values['runs'].remove(predecessor)
    run = values['runs'][0]
    parameters = {'tool': 'exa_search', 'payload': {'query': 'synthetic request'}}
    body = {'provider': 'exa', 'operation': 'exa_search', 'payload': parameters['payload']}
    annotation = None
    if operation.startswith('openrouter.'):
        from tests.lab_arena.training_cap_eof_test import cap_example
        model = cap_example(operation)
        body = deepcopy(model['provider_calls'][0]['request']['outbound']['body'])
        parameters = {key: value for key, value in body.items() if key != 'provider'}
        annotation = model['provider_calls'][0]['reasoning_capture']
        values['round_row']['configuration_doc']['reasoning_route_registry'] = model['configuration']['reasoning_route_registry']
        for row in values['ledger_rows']:
            if 'reasoning_capture' in row.get('entry_doc', {}):
                row['entry_doc']['reasoning_capture']['registry_sha256'] = annotation['registry_sha256']
    request_hash = contracts.document_hash(parameters)
    pointer = training_capture.RequestCapture(values['objects'])({
        'operation_id': operation, 'request_hash': request_hash,
        'hashed_parameters': parameters, 'admitted_parameters': parameters,
        'effective_parameters': parameters, 'outbound': {'method': 'POST',
            'host': 'openrouter.ai' if annotation else 'code.deepline.com',
            'path': ('/api/v1/responses' if operation.endswith('responses') else '/api/v1/chat/completions')
                if annotation else '/api/v2/integrations/exa_search/execute', 'query': {}, 'body': body,
            'body_sha256': contracts.document_hash(body), 'content_type': 'application/json'}})
    identity = contracts.hash_bytes(b'never-dispatched-call')
    context = {key: run[key] for key in ('run_id', 'round_id', 'submission_id', 'miner_hotkey')}
    context.update(provider='openrouter' if annotation else 'deepline', operation_id=operation,
        call_identity=identity, funding_source='host')
    values['ledger_rows'].extend([
        dict(context, entry_id=10001, entry_kind='reservation', amount_microusd=0,
            entry_doc={'action_sequence': 3, 'base_call_identity': identity, 'provider_attempt': 1,
                'request_hash': request_hash, 'request_capture': pointer}),
        dict(context, entry_id=10002, entry_kind='recovery', amount_microusd=0,
            entry_doc={'reason': 'attempt_completed_before_dispatch', 'reserved_microusd': 0})])
    if annotation:
        values['ledger_rows'][-2]['entry_doc']['reasoning_capture'] = annotation
    snapshot = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])
    payload = snapshot['payload']
    payload['observations'].append({'ordinal': 3, 'action_sequence': 3, 'operation_id': operation,
        'kind': 'transport_failure', 'request': {'admitted_parameters': parameters, 'timeout_ms': 1000},
        'outcome': {'protocol': 'frame', 'exception_class': 'ReadTimeout'}, 'call_identity': None,
        'privacy': {'redacted': False, 'redactions': []}, 'contributes_to_provider_call_count': True})
    if local_error:
        payload['observations'][0]['outcome'].update(error_code='worker_unavailable', response={'error': 'worker_unavailable'})
        if http_error:
            payload['observations'][0]['outcome'].update(protocol='http', response={
                'status': 503, 'headers': {}, 'body_b64': base64.b64encode(runner._http_error_body('worker_unavailable')).decode()})
    payload.update(observation_count=1, complete=complete, failures=[] if complete else ['capture_incomplete'])
    save_snapshot(values, run, payload)
    run['result_doc']['resource_summary']['provider_call_count'] += 1
    return values


def capture_bank(example):
    examples = bank(example)
    examples.pop('run')
    examples[example['run_id']] = example
    for control in examples.values():
        control['source'] = deepcopy(example['source'])
        control['submission_id'] = example['submission_id']
        control['role'] = example['role']
        seal(control)
    return examples


def assess_bank(values, example, *, percent=80):
    examples = capture_bank(example)
    return training_eligibility.assess_capture(values['submission_row'], examples, list(examples),
        enabled=True, minimum_coverage_percent=percent)








@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('complete', [False, True])
def test_reader_keeps_zero_recovery_proof_and_no_response_claim(operation, complete):
    values = undispatched_values(operation=operation, complete=complete)
    store = SimpleNamespace(
        get_run=lambda identity: next(row for row in values['runs'] if row['run_id'] == identity),
        get_round=lambda _: values['round_row'], get_submission=lambda _: values['submission_row'],
        list_runs=lambda _: values['runs'],
        list_ledger=lambda *, run_id, after_entry_id, limit: [row for row in values['ledger_rows']
            if row['run_id'] == run_id and (after_entry_id is None or row['entry_id'] > after_entry_id)][:limit],
        icp_cost_eligibility=lambda **identity: values['cost_eligibility_reader'](**identity)['eligibility'])
    example = training_reader.export_run(store, values['objects'], values['run_id'])
    call = example['provider_calls'][-1]
    assert training_thinking.verified_undispatched_transport(example, call)
    assert assess_bank(values, example)['eligible']
    assert 'response' not in call and call['actual_microusd'] == 0


@pytest.mark.parametrize('protocol', ['frame', 'http'])
@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('complete', [False, True])
def test_real_runner_transport_observation_is_not_a_provider_response(tmp_path, protocol, operation, complete):
    from tests.lab_arena.test_lab_arena_runner import lease
    class Api:
        def provider(self, *_):
            raise runner.RunnerError('synthetic transport failure')
    values = undispatched_values(operation=operation, complete=complete)
    state = runner.RunState(lease=lease(), lease_token='private-lease-token')
    server = runner.WorkerSocketServer(tmp_path / 'worker.sock', Api(), state)
    parameters = values['ledger_rows'][-2]['entry_doc']
    request = training_capture.read_request(values['objects'], parameters['request_capture'])
    observation = server._begin_observation(operation, request['admitted_parameters'], 1000, protocol)
    error, document = server._dispatch_observed(operation, request['admitted_parameters'], 1000,
        observation=observation)
    assert error == 'worker_unavailable' and document is None
    returned = ({'error': error} if protocol == 'frame' else {
        'status': 503, 'headers': {}, 'body_b64': base64.b64encode(runner._http_error_body(error)).decode()})
    server._finish_observation(observation, returned, error=error)
    observed = state.worker_observations.document()['observations'][0]
    assert observed['kind'] == 'transport_failure' and observed['outcome'] == {
        'protocol': protocol, 'response': returned,
        'exception_class': 'RunnerError', 'error_code': 'worker_unavailable'}
    assert state.worker_observations.document()['response_receipts'] == []
    observed.update(action_sequence=3, ordinal=3)
    run = values['runs'][0]
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    payload['observations'] = [observed]
    save_snapshot(values, run, payload)
    example = training_export.assemble_example(**values)
    assert training_thinking.verified_undispatched_transport(example, example['provider_calls'][-1])
    assert assess_bank(values, example)['eligible']
