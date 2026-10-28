"""An authenticated request that never dispatched consumes one incomplete ICP."""
from copy import deepcopy
from types import SimpleNamespace
import base64

import pytest

from lab_arena import contracts, runner, training_capture, training_eligibility, training_export, training_observations, training_reader, training_steps, training_thinking
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_retry_snapshot_test import retry_values, save_snapshot
from tests.lab_arena.training_steps_test import seal


def undispatched_values(*, local_error=True, http_error=False):
    values, predecessor, _ = retry_values()
    values['runs'].remove(predecessor)
    run = values['runs'][0]
    parameters = {'tool': 'synthetic_tool', 'arguments': {'query': 'synthetic request'}}
    request_hash = contracts.document_hash(parameters)
    pointer = training_capture.RequestCapture(values['objects'])({
        'operation_id': 'deepline.execute', 'request_hash': request_hash,
        'hashed_parameters': parameters, 'admitted_parameters': parameters,
        'effective_parameters': parameters, 'outbound': {'body': parameters}})
    identity = contracts.hash_bytes(b'never-dispatched-call')
    context = {key: run[key] for key in ('run_id', 'round_id', 'submission_id', 'miner_hotkey')}
    context.update(provider='deepline', operation_id='deepline.execute', call_identity=identity, funding_source='miner')
    values['ledger_rows'].extend([
        dict(context, entry_id=10001, entry_kind='reservation', amount_microusd=0,
            entry_doc={'action_sequence': 3, 'base_call_identity': identity, 'provider_attempt': 1,
                'request_hash': request_hash, 'request_capture': pointer}),
        dict(context, entry_id=10002, entry_kind='recovery', amount_microusd=0,
            entry_doc={'reason': 'attempt_completed_before_dispatch', 'reserved_microusd': 0})])
    snapshot = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])
    payload = snapshot['payload']
    payload['observations'].append({'ordinal': 3, 'action_sequence': 3, 'operation_id': 'deepline.execute',
        'kind': 'transport_failure', 'request': {'admitted_parameters': parameters, 'timeout_ms': 1000},
        'outcome': {'protocol': 'frame', 'exception_class': 'ReadTimeout'}, 'call_identity': None,
        'privacy': {'redacted': False, 'redactions': []}, 'contributes_to_provider_call_count': True})
    if local_error:
        payload['observations'][0]['outcome'].update(error_code='worker_unavailable', response={'error': 'worker_unavailable'})
        if http_error:
            payload['observations'][0]['outcome'].update(protocol='http', response={
                'status': 503, 'headers': {}, 'body_b64': base64.b64encode(runner._http_error_body('worker_unavailable')).decode()})
    payload.update(observation_count=1, complete=False, failures=['capture_incomplete'])
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


@pytest.mark.parametrize('role', ['baseline', 'miner'])
@pytest.mark.parametrize('optional', [False, True])
@pytest.mark.parametrize('style', ['exception', 'frame', 'http'])
def test_actual_export_keeps_undispatched_failure_without_forging_a_response(role, optional, style):
    values = undispatched_values(local_error=style != 'exception', http_error=style == 'http')
    if role == 'miner':
        submission = values['submission_row']
        submission['is_king'] = False
        submission['source_archive_sha256'] = contracts.hash_bytes(values['objects'].values[submission['source_ref']])
    if optional:
        from tests.lab_arena.training_optional_capture_test import summary
        run = values['runs'][0]
        payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
            run['lease_generation'], run['claim_request_hash'])['payload']
        payload['agent_observations'].append(summary())
        payload['agent_observation_count'] += 1
        payload['failures'].append(training_observations.OPTIONAL_CAPTURE_INCOMPLETE)
        save_snapshot(values, run, payload)
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    assert call['accounting_state'] == 'recovery' and call['actual_microusd'] == 0
    assert example['completeness']['incomplete_reasons'] == (
        ['worker_observations_incomplete'] if style != 'exception'
        else ['full_response_unavailable', 'worker_observations_incomplete'])
    assert 'response' not in call and call['response_receipts'] == [
        {'created_at': None, 'entry_id': 10002, 'entry_kind': 'recovery', 'metadata': {}, 'response_unavailable': True}]
    assert call['lifecycle'][-1]['accounting']['reason'] == 'attempt_completed_before_dispatch'
    assert call['response_provenance'] == {'source': 'stored_provider_ledger_receipt', 'delivery_to_worker': 'not_proved'}
    before = deepcopy(example)
    # Keep the generated controls bound to the actual frozen source identity.
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert not assess_bank(values, example, percent=100)['eligible']
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    assert example == before


def test_reader_keeps_zero_recovery_proof_and_no_response_claim():
    values = undispatched_values()
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
def test_real_runner_transport_observation_is_not_a_provider_response(tmp_path, protocol):
    from tests.lab_arena.test_lab_arena_runner import lease
    class Api:
        def provider(self, *_):
            raise runner.RunnerError('synthetic transport failure')
    values = undispatched_values()
    state = runner.RunState(lease=lease(), lease_token='private-lease-token')
    server = runner.WorkerSocketServer(tmp_path / 'worker.sock', Api(), state)
    parameters = values['ledger_rows'][-2]['entry_doc']
    request = training_capture.read_request(values['objects'], parameters['request_capture'])
    observation = server._begin_observation('deepline.execute', request['admitted_parameters'], 1000, protocol)
    error, document = server._dispatch_observed('deepline.execute', request['admitted_parameters'], 1000,
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


def test_undispatched_gap_uses_the_same_eighty_percent_floor():
    values = undispatched_values()
    example = training_export.assemble_example(**values)
    examples = capture_bank(example)
    examples['run-8'] = None
    def assess():
        return training_eligibility.assess_capture(values['submission_row'], examples, list(examples),
            enabled=True, minimum_coverage_percent=80)
    result = assess()
    assert result['eligible'] and result['counts']['complete_runs'] == 8
    examples['run-7'] = None
    assert assess()['reasons'] == ['training_coverage_incomplete']


@pytest.mark.parametrize('damage', ['dispatch', 'reason', 'missing_reason', 'recovery_amount', 'actual_cost',
    'bool_cost', 'reserved_mismatch', 'bool_reserved', 'reservation_id', 'terminal_id', 'duplicate_recovery',
    'response', 'response_hash', 'upstream', 'delivery_claim', 'host_receipt', 'missing_observation',
    'observation_kind', 'observation_identity', 'observation_operation', 'observation_action',
    'observation_request', 'observation_response', 'privacy', 'snapshot_hash', 'snapshot_run', 'snapshot_claim',
    'request_hash', 'request_privacy', 'worker_ref', 'legacy', 'policy_off', 'other_operation'])
def test_undispatched_proof_cannot_excuse_dispatched_or_unproved_failure(damage):
    values = undispatched_values()
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    artifact = example['worker_observation_artifacts'][example['run_id']]
    payload = artifact['payload']
    observation = payload['observations'][0]
    wrong = contracts.hash_bytes(b'wrong')
    if damage == 'dispatch': call['lifecycle'].insert(1, {'entry_id': 10000, 'entry_kind': 'dispatch', 'amount_microusd': 0})
    elif damage == 'reason': call['lifecycle'][-1]['accounting']['reason'] = 'call_expired'
    elif damage == 'missing_reason': call['lifecycle'][-1]['accounting'].pop('reason')
    elif damage == 'recovery_amount': call['lifecycle'][-1]['amount_microusd'] = 1
    elif damage == 'actual_cost': call['actual_microusd'] = None
    elif damage == 'bool_cost': call['actual_microusd'] = False
    elif damage == 'reserved_mismatch': call['lifecycle'][-1]['accounting']['reserved_microusd'] = 1
    elif damage == 'bool_reserved': call['lifecycle'][-1]['accounting']['reserved_microusd'] = False
    elif damage == 'reservation_id': call['reservation_entry_id'] += 1
    elif damage == 'terminal_id': call['terminal_entry_id'] += 1
    elif damage == 'duplicate_recovery': call['lifecycle'].append(deepcopy(call['lifecycle'][-1]))
    elif damage == 'response': call['response'] = {'result': 'invented'}
    elif damage == 'response_hash': call['response_sha256'] = wrong
    elif damage == 'upstream': call['upstream_response_receipts'] = [{'entry_id': 10002}]
    elif damage == 'delivery_claim': call['response_provenance']['delivery_to_worker'] = 'socket_written'
    elif damage == 'host_receipt':
        receipt = deepcopy(payload['response_receipts'][0])
        receipt.update(ordinal=3, action_sequence=3, call_identity=call['call_identity'], operation_id=call['operation_id'])
        payload['response_receipts'].append(receipt)
        payload['response_receipt_count'] += 1
    elif damage == 'missing_observation': payload.update(observations=[], observation_count=0)
    elif damage == 'observation_kind': observation['kind'] = 'nonstandard_response'
    elif damage == 'observation_identity': observation['call_identity'] = wrong
    elif damage == 'observation_operation': observation['operation_id'] = 'openrouter.chat'
    elif damage == 'observation_action': observation['action_sequence'] = 4
    elif damage == 'observation_request': observation['request']['admitted_parameters']['tool'] = 'different'
    elif damage == 'observation_response': observation['outcome'] = {'protocol': 'frame', 'response': {'error': 'transport'}}
    elif damage == 'privacy': observation['privacy'] = {'redacted': True, 'redactions': ['credential']}
    elif damage == 'snapshot_hash': artifact['payload_sha256'] = wrong
    elif damage == 'snapshot_run': artifact['run_id'] = 'other'
    elif damage == 'snapshot_claim': artifact['claim_request_hash'] = 'invalid'
    elif damage == 'request_hash': call['request']['request_hash'] = wrong
    elif damage == 'request_privacy': call['request_capture']['redacted'] = True
    elif damage == 'worker_ref': call['worker_observation_ref']['ordinal'] += 1
    elif damage == 'legacy': example['configuration'].pop('training_maximum_missing_reasoning_percent')
    elif damage == 'policy_off': example['configuration']['reasoning_capture_policy'] = None
    elif damage == 'other_operation': call['operation_id'] = 'exa.contents'
    if damage != 'snapshot_hash':
        artifact['payload_sha256'] = contracts.document_hash(payload)
        if damage != 'worker_ref': call['worker_observation_ref']['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    assert not training_thinking.verified_undispatched_transport(example, call), damage
    assert not assess_bank(values, example)['eligible'], damage
    assert training_thinking.selected_research_evidence(example) is None


@pytest.mark.parametrize('damage', ['other_request', 'other_response', 'other_receipt', 'other_route', 'other_cost'])
def test_undispatched_allowance_still_checks_each_delivered_call(damage):
    values = undispatched_values()
    example = training_export.assemble_example(**values)
    other = example['provider_calls'][0]
    if damage == 'other_request': other['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'other_response': other['response']['changed'] = True
    elif damage == 'other_receipt': other['response_provenance']['worker_receipt_ref']['ordinal'] = 1000
    elif damage == 'other_route': other['reasoning_capture']['provider_route'] = 'wrong'
    elif damage == 'other_cost': other['actual_microusd'] = None
    seal(example)
    assert training_thinking.verified_undispatched_transport(example, example['provider_calls'][-1])
    assert not assess_bank(values, example)['eligible'], damage
