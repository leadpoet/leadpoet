"""Miners choose one tested provider without changing host privacy or fallbacks."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc, shim, training_capture, training_thinking
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.training_export_test import Objects


def registry(api='openrouter.chat'):
    value = _reviewed_registry(api=api)
    other = deepcopy(value['approved_routes'][0])
    other.update(provider='example/alternative', provider_name='Alternative Provider',
                 price_ceiling={'prompt': '0.0000004', 'completion': '0.000002',
                                'request': '0', 'internal_reasoning': '0'})
    other['evidence']['probe_id'] = 'probe-alternative-001'
    value['approved_routes'].append(other)
    return value


def request(api='openrouter.chat', provider=None):
    value = {'model': bt.CHAT['model'], **({'input': 'Find evidence.'} if api.endswith('responses')
                 else {'messages': [{'role': 'user', 'content': 'Find evidence.'}]})}
    if provider is not None:
        value['provider'] = {'only': [provider]}
    return value


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_registry_and_exact_provider_choice_are_supported(api):
    frozen = rc.validate_registry(registry(api))
    params = operations.validate_operation_request(api, request(api, 'example/alternative'), allow_full_thinking=True)
    applied, annotation = br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=frozen)
    assert annotation['provider_route'] == 'example/alternative'
    assert applied['provider'] == {'only': ['example/alternative']}
    assert annotation['probe_id'] == 'probe-alternative-001'


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_omitted_provider_keeps_existing_default_and_request_hash(api):
    single = _reviewed_registry(api=api)
    default = br.normalized_request(api, request(api), reasoning_capture_policy=rc.POLICY, reasoning_route_registry=single)
    expanded = br.normalized_request(api, request(api), reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry(api))
    assert default == expanded and contracts.document_hash(default) == contracts.document_hash(expanded)
    assert rc.apply(api, request(api), policy=rc.POLICY, registry=registry(api))[1]['provider_route'] == 'example/verified'
    assert rc.validate_registry(single) == single


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('provider', [None, 'example/alternative'])
def test_broker_pins_selected_route_cost_capture_and_replay_without_another_send(api, provider):
    frozen = registry(api)
    selected = frozen['approved_routes'][int(provider is not None)]
    response = {'model': bt.CHAT['model'], 'provider': selected['provider_name'],
        'openrouter_metadata': {'endpoints': {'available': [{'selected': True,
            'model': selected['model_version'], 'provider': selected['provider_name']}]}},
        'usage': {'cost': '0.00001'}, **({'status': 'completed', 'output': [
            {'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': 'Done.'}]}]}
            if api.endswith('responses') else {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Done.'}}]})}
    objects = Objects()
    transport = bt.FakeTransport([(200, response)])
    broker, store, _ = bt.make_broker(transport=transport, capture_request=training_capture.RequestCapture(objects).prepare)
    context = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=frozen)
    arguments = dict(operation_id=api, parameters=request(api, provider), action_sequence=0, timeout_ms=30000)
    first = broker.execute(context, **arguments)
    assert first.status == 200
    sent = json.loads(transport.sent[0]['body'])
    assert sent['provider']['only'] == [selected['provider']]
    assert sent['provider']['zdr'] is True and sent['provider']['data_collection'] == 'deny'
    assert sent['provider']['allow_fallbacks'] is False and sent['provider']['require_parameters'] is True
    assert sent['provider']['max_price']['completion'] == float(selected['price_ceiling']['completion']) * 1_000_000
    saved = store.calls[first.call['call_identity']]
    captured = training_capture.read_request(objects, saved['call_doc']['request_capture'])
    assert captured['reasoning_capture']['provider_route'] == selected['provider']
    assert captured['outbound']['body']['provider'] == sent['provider']
    assert contracts.document_hash(captured['hashed_parameters']) == first.call['request_hash']
    observation = br.broker_observations(saved['terminal'])['reasoning_route_observation']
    assert observation['status'] == 'matched'
    call = {'provider': 'openrouter', 'operation_id': api, 'reasoning_capture': captured['reasoning_capture'],
            'request': captured, 'route_observation': observation, 'response': response}
    assert training_thinking._policy(call, frozen)
    assert training_thinking.attempt_coverage(call, 'miner', frozen)['provider_route_verified']
    replay = broker.execute(context, **arguments)
    assert replay.body == first.body and len(transport.sent) == 1


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('bad', [
    {'only': []}, {'only': ['example/verified', 'example/alternative']}, {'only': 'example/alternative'},
    {'only': ['example/alternative'], 'allow_fallbacks': True},
    {'only': ['example/alternative'], 'zdr': False},
    {'only': ['example/alternative'], 'data_collection': 'allow'},
    {'only': ['example/alternative'], 'order': ['example/alternative']},
    {'only': ['example/alternative'], 'require_parameters': False},
])
def test_provider_selector_cannot_change_host_privacy_or_fallback_policy(api, bad):
    with pytest.raises(operations.OperationError):
        operations.validate_operation_request(api, {**request(api), 'provider': bad}, allow_full_thinking=True)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_unapproved_provider_fails_before_reservation(api):
    broker, store, transport = bt.make_broker()
    context = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry(api))
    result = broker.execute(context, operation_id=api, parameters=request(api, 'not/approved'), action_sequence=0, timeout_ms=1000)
    assert result.status == 400 and not store.log and not transport.sent


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_policy_off_still_rejects_provider_choice(api):
    with pytest.raises(operations.OperationError):
        operations.validate_operation_request(api, request(api, 'example/alternative'))


def test_exact_duplicate_route_still_fails_registry_validation():
    frozen = registry()
    frozen['approved_routes'].append(deepcopy(frozen['approved_routes'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        rc.validate_registry(frozen)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_shim_worker_and_http_matching_preserve_the_selector(api):
    parameters = request(api, 'example/alternative')
    path = '/api/v1/responses' if api.endswith('responses') else '/api/v1/chat/completions'
    matched, normalized = operations.match_request('POST', 'https://openrouter.ai' + path,
        json.dumps(parameters).encode(), {'content-type': 'application/json'}, allow_full_thinking=True)
    assert matched == api and normalized['provider'] == parameters['provider']
    frame = shim.build_operation_frame(api, normalized, 1000)
    decoded_api, decoded, timeout = shim.decode_operation_frame(frame, allow_full_thinking=True)
    assert decoded_api == api and decoded == normalized and timeout == 1000
    with pytest.raises(operations.OperationError):
        shim.decode_operation_frame(frame)


def test_preexisting_chat_privacy_shape_keeps_its_hash_and_safe_controls():
    parameters = request()
    compatible = {**parameters, 'provider': {'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False}}
    for full in (False, True):
        assert operations.validate_operation_request('openrouter.chat', compatible, allow_full_thinking=full) == operations.validate_operation_request('openrouter.chat', parameters, allow_full_thinking=full)
    chosen = {**compatible, 'provider': {**compatible['provider'], 'only': ['example/alternative']}}
    assert operations.validate_operation_request('openrouter.chat', chosen, allow_full_thinking=True)['provider'] == {'only': ['example/alternative']}
    with pytest.raises(operations.OperationError):
        operations.validate_operation_request('openrouter.responses', {
            **request('openrouter.responses', 'example/alternative'),
            'provider': {'only': ['example/alternative'], 'require_parameters': True}}, allow_full_thinking=True)


def test_actual_returned_provider_must_still_match_the_selected_frozen_row():
    from tests.lab_arena.training_eligibility_test import captured_example
    from tests.lab_arena.training_steps_test import seal
    example = captured_example()
    call = example['provider_calls'][0]
    frozen = example['configuration']['reasoning_route_registry']
    original = next(row for row in frozen['approved_routes'] if row['model_id'] == call['reasoning_capture']['model_id'] and row['api'] == call['operation_id'])
    alternative = deepcopy(original)
    alternative.update(provider='example/alternative', provider_name='Alternative Provider')
    alternative['evidence']['probe_id'] = 'alternative-probe-saved'
    frozen['approved_routes'].append(alternative)
    annotation = call['reasoning_capture']
    annotation.update(provider_route=alternative['provider'], provider_name=alternative['provider_name'],
        probe_id=alternative['evidence']['probe_id'], registry_sha256=contracts.document_hash(frozen).split(':')[1])
    # Bind the explicit request to the alternate row. The unchanged saved
    # provider observation still identifies the old provider and must fail.
    request_doc = call['request']
    request_doc['admitted_parameters']['provider'] = {'only': [alternative['provider']]}
    request_doc['hashed_parameters']['provider'] = {'only': [alternative['provider']]}
    request_doc['outbound']['body']['provider']['only'] = [alternative['provider']]
    request_doc['request_hash'] = call['request_hash'] = contracts.document_hash(request_doc['hashed_parameters'])
    request_doc['outbound']['body_sha256'] = contracts.document_hash(request_doc['outbound']['body'])
    assert training_thinking._policy(call, frozen)
    assert not training_thinking.attempt_coverage(call, 'miner', frozen)['provider_route_verified']
    annotation['provider_route'] = 'not/approved'
    assert not training_thinking._policy(call, frozen)


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_alternative_provider_survives_actual_worker_storage_and_strict_training(api, tmp_path):
    import base64
    import socket
    import tempfile
    from pathlib import Path
    from lab_arena import runner, training_export, training_steps
    from tests.lab_arena.test_lab_arena_runner import lease
    from tests.lab_arena.training_steps_test import bind_selected_output, fixture, seal

    frozen = registry(api)
    route = frozen['approved_routes'][1]
    response = {'model': bt.CHAT['model'], 'provider': route['provider_name'],
        'openrouter_metadata': {'endpoints': {'available': [{'selected': True,
            'model': route['model_version'], 'provider': route['provider_name']}]}},
        'usage': {'cost': '0.00001', 'completion_tokens_details': {'reasoning_tokens': 10}},
        **({'status': 'completed', 'output': [
            {'type': 'reasoning', 'id': 'rs', 'status': 'completed', 'content': [
                {'type': 'reasoning_text', 'text': 'Check the retained evidence before answering.'}]},
            {'type': 'message', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': '{"complete":true}'}]}]}
            if api.endswith('responses') else {'choices': [{'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': '{"complete":true}', 'reasoning_details': [
                    {'type': 'reasoning.text', 'text': 'Check the retained evidence before answering.'}]}}]})}
    objects = Objects()
    transport = bt.FakeTransport([(200, response)])
    broker, store, _ = bt.make_broker(transport=transport, capture_request=training_capture.RequestCapture(objects).prepare)
    context = replace(bt.CONTEXT, run_id='r1', reasoning_capture_policy=rc.POLICY, reasoning_route_registry=frozen)

    class Api:
        def provider(self, run_id, token, frame):
            return broker.execute(context, **frame).to_document()

    state = runner.RunState(lease={**lease('r1'), 'kind': 'execute', 'reasoning_capture_policy': rc.POLICY}, lease_token='tok-r1')
    directory = tempfile.TemporaryDirectory(prefix='choice-', dir='/tmp')
    path = Path(directory.name) / 'w.sock'
    server = runner.WorkerSocketServer(path, Api(), state)
    server.start()
    try:
        frame = shim.build_operation_frame(api, request(api, 'example/alternative'), 1000)
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
            client.sendall(len(frame).to_bytes(4, 'big') + frame)
            size = int.from_bytes(runner._recv_exact(client, 4), 'big')
            wire = json.loads(runner._recv_exact(client, size))
        assert wire['status'] == 200 and json.loads(base64.b64decode(wire['body_b64']))['model'] == bt.CHAT['model']
    finally:
        server.stop()
        directory.cleanup()
    assert len(transport.sent) == 1
    identity, saved = next(iter(store.calls.items()))
    membership = {'run_id': 'r1', 'round_id': context.round_id, 'submission_id': context.submission_id,
        'provider': 'openrouter', 'operation_id': api, 'funding_source': 'host', 'call_identity': identity}
    rows = [{**membership, 'entry_id': 1, 'entry_kind': 'reservation', 'amount_microusd': saved['amount'], 'entry_doc': saved['call_doc']},
            {**membership, 'entry_id': 2, 'entry_kind': 'dispatch', 'amount_microusd': 0, 'entry_doc': {}},
            {**membership, 'entry_id': 3, 'entry_kind': 'settlement', 'amount_microusd': saved['actual'],
             'entry_doc': {}, 'terminal_response': saved['terminal']}]
    calls, gaps = training_export._calls(rows, {'r1'}, objects, {'r1': 0})
    assert not gaps
    payload = state.worker_observations.document()
    assert payload['agent_observations'] == []
    snapshot = {'payload': payload, 'payload_sha256': contracts.document_hash(payload)}
    assert not training_export._link_worker_receipts({'run_id': 'r1', 'kind': 'execute'}, calls, snapshot, require_client_ack=False)
    example = fixture()
    example['attempts'][0]['run_id'] = 'r1'
    example['input_snapshots']['r1'] = example['input_snapshots'].pop('run')
    example['input_snapshots'][example['score_run_id']]['input']['scored_run_id'] = 'r1'
    example.update(run_id='r1', provider_calls=calls, worker_observation_artifacts={'r1': snapshot},
        configuration={'reasoning_capture_policy': rc.POLICY, 'reasoning_route_registry': frozen,
                       'training_maximum_missing_reasoning_percent': 0},
        completeness={'complete': True, 'incomplete_reasons': []})
    bind_selected_output(example)
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 1
    assert records[0]['source_refs'][0]['route']['provider_route'] == 'example/alternative'
    assert records[0]['generated_thinking'][0]['text'] == 'Check the retained evidence before answering.'


@pytest.mark.parametrize('api', ['openrouter.chat', 'openrouter.responses'])
def test_provider_selection_is_per_request_without_implicit_sticky_state_or_fallback(api):
    transport = bt.FakeTransport([(503, {'error': {'code': 503}}), (200, {
        'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Done.'}}],
        'usage': {'cost': '0.00001'}})])
    broker, store, _ = bt.make_broker(transport=transport)
    context = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry(api))
    first = broker.execute(context, operation_id=api, parameters=request(api, 'example/alternative'), action_sequence=0, timeout_ms=1000)
    assert first.status == 502 and len(transport.sent) == 1
    assert json.loads(transport.sent[0]['body'])['provider']['only'] == ['example/alternative']
    second = broker.execute(context, operation_id=api, parameters=request(api), action_sequence=1, timeout_ms=1000)
    assert len(transport.sent) == 2
    assert json.loads(transport.sent[1]['body'])['provider']['only'] == ['example/verified']
    assert first.call['call_identity'] != second.call['call_identity']
    assert all(json.loads(row['body'])['provider']['allow_fallbacks'] is False for row in transport.sent)


@pytest.mark.parametrize('damage', ['admitted_only', 'hashed_only', 'admitted_hash_disagree',
                                   'both_disagree_annotation', 'outbound_disagree', 'outbound_missing'])
def test_saved_explicit_selector_is_bound_to_request_annotation_and_wire(damage):
    from tests.lab_arena.training_host_result_test import without_client_callbacks
    from tests.lab_arena.training_eligibility_test import submission
    from tests.lab_arena.training_steps_test import seal
    from lab_arena import training_eligibility, training_steps
    example = without_client_callbacks()
    call = example['provider_calls'][0]
    request_doc = call['request']
    expected = call['reasoning_capture']['provider_route']
    request_doc['admitted_parameters']['provider'] = {'only': [expected]}
    request_doc['hashed_parameters']['provider'] = {'only': [expected]}
    request_doc['outbound']['body']['provider']['only'] = [expected]
    if damage == 'admitted_only': request_doc['hashed_parameters'].pop('provider')
    elif damage == 'hashed_only': request_doc['admitted_parameters'].pop('provider')
    elif damage == 'admitted_hash_disagree': request_doc['admitted_parameters']['provider']['only'] = ['not/the/annotation']
    elif damage == 'both_disagree_annotation':
        request_doc['admitted_parameters']['provider']['only'] = ['not/the/annotation']
        request_doc['hashed_parameters']['provider']['only'] = ['not/the/annotation']
    elif damage == 'outbound_disagree': request_doc['outbound']['body']['provider']['only'] = ['not/the/annotation']
    else: request_doc['outbound']['body']['provider'].pop('only')
    request_doc['request_hash'] = call['request_hash'] = contracts.document_hash(request_doc['hashed_parameters'])
    request_doc['outbound']['body_sha256'] = contracts.document_hash(request_doc['outbound']['body'])
    seal(example)
    assert not training_thinking._policy(call, example['configuration']['reasoning_route_registry'])
    with pytest.raises(ValueError): training_steps._candidate(example, call)
    assert not training_eligibility.assess_capture(submission(), {'run': example}, ['run'], enabled=True)['eligible']


def test_direct_builder_rejects_a_selected_provider_different_from_its_host_pin():
    with pytest.raises(operations.OperationError):
        operations.build_outbound_request('openrouter.chat', request(provider='example/alternative'),
            allow_full_thinking=True, openrouter_provider_policy={
                'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False,
                'only': ['example/verified'], 'require_parameters': True,
                'max_price': {'prompt': 1, 'completion': 1, 'request': 0}})


@pytest.mark.parametrize('annotation', [True, ['malformed'], 'malformed', 7])
def test_generic_legacy_candidate_does_not_crash_on_non_mapping_optional_annotation(annotation):
    from lab_arena import training_steps
    from tests.lab_arena.training_steps_test import fixture, seal
    example = fixture()
    example['provider_calls'][0]['reasoning_capture'] = annotation
    assert training_steps._candidate(seal(example), example['provider_calls'][0])
