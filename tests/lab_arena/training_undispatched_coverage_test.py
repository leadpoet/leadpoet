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
@pytest.mark.parametrize('style', ['exception', 'frame', 'http'])
def test_undispatched_model_and_complete_capture_consume_only_one_icp(operation, complete, style):
    values = undispatched_values(operation=operation, complete=complete,
        local_error=style != 'exception', http_error=style == 'http')
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    assert training_thinking.verified_undispatched_transport(example, call)
    assert 'worker_response_receipt_link_invalid' not in example['completeness']['incomplete_reasons']
    payload = example['worker_observation_artifacts'][example['run_id']]['payload']
    assert payload['complete'] is complete
    assert payload['failures'] == ([] if complete else ['capture_incomplete'])
    if complete and style != 'exception':
        assert example['completeness']['complete'] is True
        assert example['completeness']['incomplete_reasons'] == []
    before = deepcopy(example)
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert not assess_bank(values, example, percent=100)['eligible']
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']
    assert 'response' not in call and call['actual_microusd'] == 0
    assert example == before


def test_five_undispatched_requests_with_mixed_provider_operations_consume_one_icp():
    values = undispatched_values(operation='openrouter.responses', complete=True)
    deepline = undispatched_values(complete=True)
    run = values['runs'][0]
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    for sequence in range(4, 8):
        source = deepline if sequence == 7 else values
        rows = deepcopy(source['ledger_rows'][-2:])
        if source is deepline:
            ref = rows[0]['entry_doc']['request_capture']['ref']
            values['objects'].values[ref] = deepline['objects'].values[ref]
        identity = contracts.hash_bytes(f'undispatched-{sequence}'.encode())
        for offset, row in enumerate(rows):
            row.update(call_identity=identity, entry_id=10001 + 2 * (sequence - 3) + offset)
        rows[0]['entry_doc'].update(action_sequence=sequence, base_call_identity=identity)
        values['ledger_rows'].extend(rows)
        observation = deepcopy(payload['observations'][0])
        observation.update(ordinal=sequence, action_sequence=sequence, operation_id=rows[0]['operation_id'])
        observation['request']['admitted_parameters'] = training_capture.read_request(
            values['objects'], rows[0]['entry_doc']['request_capture'])['admitted_parameters']
        payload['observations'].append(observation)
    payload['observation_count'] = 5
    run['result_doc']['resource_summary']['provider_call_count'] += 4
    save_snapshot(values, run, payload)
    example = training_export.assemble_example(**values)
    calls = [call for call in example['provider_calls'] if call['accounting_state'] == 'recovery']
    assert len(calls) == 5
    assert all(training_thinking.verified_undispatched_transport(example, call) for call in calls)
    assert example['completeness']['complete'] is True
    assert training_eligibility._failed_delivery_gap_verified(example) == frozenset()
    result = assess_bank(values, example)
    assert result['eligible'] and result['counts']['incomplete_runs'] == 1
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


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


@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('complete', [False, True])
def test_undispatched_gap_uses_the_same_eighty_percent_floor(operation, complete):
    values = undispatched_values(operation=operation, complete=complete)
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


def test_complete_capture_without_undispatched_call_cannot_use_delivery_exception():
    values = undispatched_values(complete=True)
    values['ledger_rows'] = values['ledger_rows'][:-2]
    run = values['runs'][0]
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    payload.update(observations=[], observation_count=0)
    run['result_doc']['resource_summary']['provider_call_count'] -= 1
    save_snapshot(values, run, payload)
    example = training_export.assemble_example(**values)
    assert example['completeness']['complete'] is True
    assert example['completeness']['incomplete_reasons'] == []
    assert training_eligibility._failed_delivery_gap_verified(example) is None
    assert training_thinking._attributed_score_only_gaps(example, set(),
        undispatched_proof=training_thinking.verified_undispatched_transport) is None
    assert assess_bank(values, example)['counts']['complete_runs'] == 10


@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('failure', ['socket_write', 'provider_error'])
def test_undispatched_proof_composes_with_existing_delivered_failure_proof(operation, failure):
    from lab_arena import broker
    values = undispatched_values(operation=operation, complete=failure == 'provider_error')
    run = values['runs'][0]
    payload = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])['payload']
    write = payload['response_receipts'][0]
    if failure == 'socket_write':
        write.update(write_state='socket_write_failed', socket_written_at=None)
    else:
        terminal = values['ledger_rows'][2]
        raw = contracts.canonical_json({'error': {'code': 503, 'message': 'Saved upstream failure'}}).encode()
        terminal.update(amount_microusd=0, terminal_response=broker._terminal_response_document(
            503, {}, raw, call_succeeded=False))
        write.update(status=503, body_sha256=contracts.hash_bytes(raw), body_bytes=len(raw))
    save_snapshot(values, run, payload)
    example = training_export.assemble_example(**values)
    if failure == 'provider_error':
        assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0])
    else:
        assert training_eligibility._failed_delivery_call_verified(example, example['provider_calls'][0])
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9 and result['counts']['incomplete_runs'] == 1
    assert training_thinking.selected_research_evidence(example) is None
    strict, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert not strict['train'] and not strict['eval']


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('damage', ['duplicate_action', 'duplicate_identity', 'agent_read',
    'unexpected_observation', 'unexpected_failure', 'missing_snapshot'])
def test_complete_model_capture_requires_exact_unambiguous_host_proof(operation, damage):
    values = undispatched_values(operation=operation, complete=True)
    run = values['runs'][0]
    snapshot = training_observations.read_snapshot(values['objects'], run['run_id'], 'execute',
        run['lease_generation'], run['claim_request_hash'])
    if damage == 'unexpected_observation':
        row = deepcopy(snapshot['payload']['observations'][0])
        row.update(ordinal=4, action_sequence=4)
        snapshot['payload']['observations'].append(row)
        snapshot['payload']['observation_count'] += 1
        run['result_doc']['resource_summary']['provider_call_count'] += 1
        save_snapshot(values, run, snapshot['payload'])
    example = training_export.assemble_example(**values)
    artifact = example['worker_observation_artifacts'][example['run_id']]
    payload = artifact['payload']
    call = example['provider_calls'][-1]
    if damage == 'duplicate_action':
        row = deepcopy(payload['observations'][0])
        row['ordinal'] += 1
        payload['observations'].append(row)
        payload['observation_count'] += 1
    elif damage == 'duplicate_identity':
        example['provider_calls'][0]['call_identity'] = call['call_identity']
    elif damage == 'agent_read':
        write = payload['response_receipts'][0]
        payload['agent_observations'].append({'id': 0, 'kind': 'response_received',
            'observed_at': '2026-10-28T00:00:00Z', 'after_action_sequence': 3, 'depends_on': [],
            'data': {key: call['call_identity'] if key == 'call_identity' else write[key]
                for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')},
            'privacy': {'redacted': False, 'redactions': []}})
        payload['agent_observation_count'] += 1
        assert training_observations._validate_payload(payload) == payload
    elif damage == 'unexpected_failure': payload['failures'] = ['capture_incomplete']
    elif damage == 'missing_snapshot': example['worker_observation_artifacts'].pop(example['run_id'])
    artifact['payload_sha256'] = contracts.document_hash(payload)
    call['worker_observation_ref']['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    if damage not in {'duplicate_identity', 'unexpected_observation'}:
        assert not training_thinking.verified_undispatched_transport(example, call)
    assert not assess_bank(values, example)['eligible'], damage


@pytest.mark.parametrize('damage', ['dispatch', 'reason', 'missing_reason', 'recovery_amount', 'actual_cost',
    'bool_cost', 'reserved_mismatch', 'bool_reserved', 'reservation_id', 'terminal_id', 'duplicate_recovery',
    'response', 'response_hash', 'upstream', 'delivery_claim', 'host_receipt', 'missing_observation',
    'observation_kind', 'observation_identity', 'observation_operation', 'observation_action',
    'observation_request', 'observation_response', 'privacy', 'snapshot_hash', 'snapshot_run', 'snapshot_claim',
    'request_hash', 'request_privacy', 'worker_ref', 'legacy', 'policy_off', 'other_operation', 'provider'])
@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('complete', [False, True])
def test_undispatched_proof_cannot_excuse_dispatched_or_unproved_failure(damage, operation, complete):
    values = undispatched_values(operation=operation, complete=complete)
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
    elif damage == 'observation_operation': observation['operation_id'] = 'exa.contents'
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
    elif damage == 'provider': call['provider'] = 'exa'
    if damage != 'snapshot_hash':
        artifact['payload_sha256'] = contracts.document_hash(payload)
        if damage != 'worker_ref': call['worker_observation_ref']['payload_sha256'] = artifact['payload_sha256']
    seal(example)
    assert not training_thinking.verified_undispatched_transport(example, call), damage
    assert not assess_bank(values, example)['eligible'], damage
    assert training_thinking.selected_research_evidence(example) is None


@pytest.mark.parametrize('damage', ['other_request', 'other_response', 'other_receipt', 'other_route', 'other_cost'])
@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('complete', [False, True])
def test_undispatched_allowance_still_checks_each_delivered_call(damage, operation, complete):
    values = undispatched_values(operation=operation, complete=complete)
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


@pytest.mark.parametrize('damage', ['capture_missing', 'capture_empty', 'capture_hash', 'capture_status',
    'capture_ref', 'capture_size', 'capture_compression', 'capture_redactions', 'request_privacy',
    'request_redactions', 'outbound_missing', 'outbound_body', 'outbound_hash', 'outbound_host',
    'outbound_path', 'outbound_method', 'outbound_query', 'outbound_content_type', 'worker_ref_missing',
    'funding_missing', 'funding_unknown', 'funding_miner'])
def test_missing_request_evidence_or_unknown_funding_cannot_use_recovery_allowance(damage):
    values = undispatched_values()
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    if damage == 'capture_missing': call.pop('request_capture')
    elif damage == 'capture_empty': call['request_capture'] = {}
    elif damage == 'capture_hash': call['request_capture']['sha256'] = 'invalid'
    elif damage == 'capture_status': call['request_capture'].pop('status')
    elif damage == 'capture_ref': call['request_capture']['ref'] = 'unrelated.json.gz'
    elif damage == 'capture_size': call['request_capture']['size_bytes'] += 1
    elif damage == 'capture_compression': call['request_capture']['compressed_bytes'] = -1
    elif damage == 'capture_redactions': call['request_capture']['redactions'] = ['credential']
    elif damage == 'request_privacy': call['request']['privacy']['redacted'] = True
    elif damage == 'request_redactions': call['request']['privacy']['redactions'] = ['credential']
    elif damage == 'outbound_missing': call['request'].pop('outbound')
    elif damage == 'outbound_body': call['request']['outbound']['body'] = {'unrelated': 'body'}
    elif damage == 'outbound_hash': call['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'outbound_host': call['request']['outbound']['host'] = 'unrelated.example'
    elif damage == 'outbound_path': call['request']['outbound']['path'] = '/unrelated'
    elif damage == 'outbound_method': call['request']['outbound']['method'] = 'GET'
    elif damage == 'outbound_query': call['request']['outbound'].pop('query')
    elif damage == 'outbound_content_type': call['request']['outbound'].pop('content_type')
    elif damage == 'worker_ref_missing': call.pop('worker_observation_ref')
    else:
        for record in [call, *call['lifecycle']]:
            if damage == 'funding_missing': record.pop('funding_source')
            else: record['funding_source'] = 'miner' if damage == 'funding_miner' else 'unknown'
    seal(example)
    assert not training_thinking.verified_undispatched_transport(example, call)
    assert not assess_bank(values, example)['eligible']


@pytest.mark.parametrize('funding_source', ['host', 'miner_key'])
def test_valid_funding_and_observation_attachment_phases(funding_source):
    values = undispatched_values()
    for row in values['ledger_rows'][-2:]:
        row['funding_source'] = funding_source
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    assert training_thinking.verified_undispatched_transport(example, call)
    assert assess_bank(values, example)['eligible']
    call.pop('worker_observation_ref')
    assert not training_thinking.verified_undispatched_transport(example, call)
    assert training_thinking.verified_undispatched_transport(example, call, require_observation_ref=False)


@pytest.mark.parametrize('field', ['method', 'host', 'path', 'body_sha256', 'query', 'content_type'])
def test_outbound_validation_is_independent_of_the_capture_document_hash(field):
    values = undispatched_values()
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][-1]
    call['request']['outbound'].pop(field)
    raw = contracts.canonical_json(call['request']).encode()
    digest = contracts.hash_bytes(raw)
    call['request_capture'].update(sha256=digest,
        ref=training_capture.OBJECT_PREFIX + digest[7:] + '.json.gz', size_bytes=len(raw))
    seal(example)
    assert not training_thinking.verified_undispatched_transport(example, call)
    assert not assess_bank(values, example)['eligible']
