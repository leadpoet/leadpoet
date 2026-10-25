"""Synthetic regressions for native05's exact rejected string lengths."""
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, operations, shim, training_steps


def continuation(kind, size, *, glyph='x'):
    value = glyph * size
    if kind == 'arguments':
        value = json.dumps({'text': glyph * (size - 11)}, separators=(',', ':'), ensure_ascii=False)
        assert len(value) == size
        items = [
            {'type': 'function_call', 'call_id': 'call-1', 'name': 'review', 'arguments': value},
            {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'Tool rejected the supplied evidence.'},
        ]
    elif kind == 'custom':
        items = [
            {'type': 'custom_tool_call', 'call_id': 'call-1', 'name': 'review', 'input': value},
            {'type': 'custom_tool_call_output', 'call_id': 'call-1', 'output': 'Finished.'},
        ]
    elif kind == 'assistant':
        items = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': value}]}]
    else:
        items = [{'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': value}], 'summary': []}]
    return {'model': 'openai/gpt-4o-mini', 'input': [{'role': 'user', 'content': 'Continue from the saved evidence.'}, *items],
            'reasoning': {'effort': 'high'}, 'max_output_tokens': 32768}


@pytest.mark.parametrize('kind,size', [('reasoning', 32414), ('arguments', 51314), ('arguments', 82130)])
def test_observed_native_continuations_preserve_exact_json(kind, size):
    parameters = continuation(kind, size)
    original = deepcopy(parameters)
    raw = shim.build_operation_frame('openrouter.responses', parameters, 1000)
    operation, normalized, _ = shim.decode_operation_frame(raw, allow_full_thinking=True)
    assert operation == 'openrouter.responses'
    assert normalized == parameters == original
    training_steps._responses_input(normalized['input'], allow_reasoning_history=True)


@pytest.mark.parametrize('kind', ['reasoning', 'arguments', 'custom', 'assistant'])
def test_full_thinking_history_is_not_undercut_by_structural_string_limit(kind):
    parameters = continuation(kind, 160_000, glyph='é')
    assert operations.validate_operation_request('openrouter.responses', parameters, allow_full_thinking=True) == parameters


@pytest.mark.parametrize('kind,size', [('reasoning', 32414), ('arguments', 82130)])
def test_policy_off_preserves_previous_rejection(kind, size):
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request('openrouter.responses', continuation(kind, size))


@pytest.mark.parametrize('field', ['reasoning', 'reasoning_content', 'details', 'arguments', 'content'])
def test_chat_full_thinking_history_keeps_long_provider_values(field):
    value = 'é' * 160_000
    assistant = {'role': 'assistant', 'content': 'Previous answer.'}
    messages = [{'role': 'user', 'content': 'Continue.'}, assistant]
    if field == 'details':
        assistant['reasoning_details'] = [{'type': 'reasoning.text', 'text': value, 'signature': None}]
    elif field == 'arguments':
        assistant['content'] = None
        assistant['tool_calls'] = [{'id': 'call-1', 'type': 'function', 'index': 0,
            'function': {'name': 'review', 'arguments': json.dumps({'text': value}, ensure_ascii=False)}}]
        messages.append({'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Done.'})
    else:
        assistant[field] = value
    params = {'model': 'openai/gpt-4o-mini', 'messages': messages, 'max_tokens': 32768}
    original = deepcopy(params)
    assert operations.validate_operation_request('openrouter.chat', params, allow_full_thinking=True) == original
    training_steps._chat_messages(messages, allow_reasoning_history=True)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request('openrouter.chat', params)


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('role', ['user', 'system', 'developer'])
def test_full_thinking_does_not_expand_other_prompt_strings(operation, role):
    message = {'role': role, 'content': 'x' * 32_001}
    params = {'model': 'openai/gpt-4o-mini', 'messages' if operation.endswith('chat') else 'input': [message]}
    with pytest.raises(operations.OperationRequestError, match='invalid_field'):
        operations.validate_operation_request(operation, params, allow_full_thinking=True)


def test_catalog_and_previously_accepted_hashes_are_unchanged():
    before = deepcopy(operations.operation_table_document())
    old = {'model': 'openai/gpt-4o-mini', 'input': 'Continue.', 'max_output_tokens': 8192}
    old_normalized = operations.validate_operation_request('openrouter.responses', old)
    operations.validate_operation_request('openrouter.responses', continuation('arguments', 160_000), allow_full_thinking=True)
    assert operations.operation_table_document() == before
    assert operations.validate_operation_request('openrouter.responses', old) == old_normalized
    assert operations.validate_operation_request('openrouter.responses', old, allow_full_thinking=True) == old_normalized
    assert operations.OPERATION_LIMITS.max_string_bytes == 128_000
    assert contracts.PROVIDER_FRAME_LIMITS.max_string_bytes == 131_072
    assert contracts.RESPONSES_PROVIDER_FRAME_LIMITS.max_string_bytes == 131_072
    assert operations.OPERATIONS['openrouter.chat'].request_fields['messages'].fields['tool_calls'].fields['function'].fields['arguments'].max_length == 32_000


@pytest.mark.parametrize('glyph,count', [('x', 1_000_001), ('é', 500_001)])
def test_overlimit_history_and_request_bytes_remain_rejected(glyph, count):
    params = continuation('reasoning', count, glyph=glyph)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request('openrouter.responses', params, allow_full_thinking=True)
    raw = shim.build_operation_frame('openrouter.responses', params, 1000)
    with pytest.raises(operations.OperationRequestError):
        shim.decode_operation_frame(raw, allow_full_thinking=True)


def test_aggregate_request_and_frame_caps_stay_unchanged():
    params = continuation('reasoning', 500_000)
    params['input'].append(deepcopy(params['input'][-1]))
    with pytest.raises(operations.OperationRequestError, match='request_too_large'):
        operations.validate_operation_request('openrouter.responses', params, allow_full_thinking=True)
    raw = shim.build_operation_frame('openrouter.responses', params, 1000)
    with pytest.raises(operations.OperationRequestError, match='request_too_large'):
        shim.decode_operation_frame(raw, allow_full_thinking=True)
    with pytest.raises(shim.OperationFrameError, match='frame_too_large'):
        shim.decode_operation_frame(b' ' * (shim.MAX_FRAME_BYTES + 1), allow_full_thinking=True)


def broker_fixture(*, full=True, kind='execute'):
    from dataclasses import replace
    from lab_arena import reasoning_capture as rc, training_capture
    from tests.lab_arena import test_lab_arena_broker as bt
    from tests.lab_arena.reasoning_capture_test import _reviewed_registry
    from tests.lab_arena.training_export_test import Objects
    registry = _reviewed_registry(api='openrouter.responses')
    registry['approved_routes'][0]['context_tokens'] = 1_000_000
    registry['approved_routes'][0]['max_output_tokens'] = 32768
    response = {'id': 'resp-test', 'object': 'response', 'status': 'completed',
        'model': 'openai/gpt-4o-mini', 'provider': 'Example Provider',
        'output': [{'type': 'message', 'id': 'msg-1', 'status': 'completed', 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Continuation received.'}]}],
        'usage': {'input_tokens': 10, 'output_tokens': 10, 'cost': 0.000001}}
    transport = bt.FakeTransport([(200, response)])
    objects = Objects()
    broker, store, transport = bt.make_broker(transport=transport,
        capture_request=training_capture.RequestCapture(objects).prepare)
    context = replace(bt.CONTEXT, kind=kind, reasoning_capture_policy=rc.POLICY if full else None,
        reasoning_route_registry=registry if full else None)
    return broker, store, transport, context, objects


@pytest.mark.parametrize('kind,size,glyph', [('reasoning', 32414, 'x'), ('arguments', 51314, 'x'),
    ('arguments', 82130, 'x'), ('arguments', 160_000, 'é')])
def test_actual_bridge_worker_broker_and_saved_request_keep_every_continuation_byte(kind, size, glyph):
    from pathlib import Path
    import tempfile
    import httpx
    from lab_arena import lab_arena_codex as codex, reasoning_capture as rc, runner, training_capture
    from scripts.verify_arena_live_capture import BrokerSocketApi
    from tests.lab_arena.test_lab_arena_runner import lease
    broker, store, transport, context, objects = broker_fixture()
    api = BrokerSocketApi(broker, context)
    state = runner.RunState(lease=dict(lease(), kind='execute', reasoning_capture_policy=rc.POLICY), lease_token='host-token')
    params = continuation(kind, size, glyph=glyph)
    with tempfile.TemporaryDirectory(prefix='full-cont-', dir='/tmp') as directory:
        path = Path(directory)/'worker.sock'
        worker = runner.WorkerSocketServer(path, api, state)
        worker.start()
        try:
            with codex.ResponsesBridge(str(path), max_output_tokens=32768) as bridge, httpx.Client(trust_env=False) as client:
                reply = client.post(bridge.base_url+'/responses', headers={'Authorization': 'Bearer '+bridge.token},
                    json=dict(params, stream=True, store=False))
            assert reply.status_code == 200, reply.text
            assert 'event: response.completed' in reply.text
        finally:
            worker.stop()
    assert len(transport.sent) == len(api.results) == 1
    outbound = json.loads(transport.sent[0]['body'])
    assert outbound['input'] == params['input']
    result = api.results[0]
    assert result.call['outcome'] == 'settled'
    pointer = next(iter(store.calls.values()))['call_doc']['request_capture']
    saved = training_capture.read_request(objects, pointer)
    assert saved['admitted_parameters']['input'] == saved['outbound']['body']['input'] == params['input']
    assert saved['request_hash'] == contracts.document_hash(saved['hashed_parameters'])
    assert saved['outbound']['body_sha256'] == contracts.document_hash(outbound)
    training_steps._responses_input(saved['outbound']['body']['input'], allow_reasoning_history=True)
    assert len(state.worker_observations.response_receipts) == 1
    assert state.worker_observations.agent_observations == []


def api_service(*, full=True, kind='execute'):
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    broker, store, transport, context, objects = broker_fixture(full=full, kind=kind)
    service = object.__new__(ArenaService)
    reads = []
    def run_context(run_id, token):
        reads.append((run_id, token))
        return {'round_id': 'round-test', 'run_id': run_id}, context
    service._run_context = run_context
    service._broker_for = lambda _round: broker
    service._store = SimpleNamespace(append_trajectory_events=lambda *_args: {'status': 'accepted', 'accepted': 1})
    return service, transport, reads


@pytest.mark.parametrize('full,kind,allowed', [(True, 'execute', True), (False, 'execute', False), (True, 'score', False)])
def test_api_and_service_gate_larger_frames_by_actual_frozen_execute_policy(full, kind, allowed):
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    service, transport, reads = api_service(full=full, kind=kind)
    frame = {'operation_id': 'openrouter.responses', 'parameters': continuation('arguments', 160_000, glyph='é'),
             'timeout_ms': 1000, 'action_sequence': 0}
    with TestClient(create_app(service)) as client:
        answer = client.post('/arena/v1/runs/r1/provider', content=contracts.canonical_json(frame),
            headers={'x-lab-arena-lease': 'a'*64})
    assert len(reads) == 1  # Reuse the one existing lease/context lookup.
    if allowed:
        assert answer.status_code == 200 and answer.json()['status'] == 200
        assert len(transport.sent) == 1
        assert json.loads(transport.sent[0]['body'])['input'] == frame['parameters']['input']
    else:
        # The old outer decoder/service rejected this exact same field/path.
        with pytest.raises(contracts.ArenaContractError) as old:
            contracts.check_strict_document(frame, contracts.RESPONSES_PROVIDER_FRAME_LIMITS)
        assert answer.status_code == 400
        assert answer.json() == {'status': 'rejected', 'code': 'contract:'+str(old.value)[:100]}
        assert transport.sent == []


def test_api_policy_off_keeps_operation_rejection_before_provider_dispatch():
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    service, transport, reads = api_service(full=False)
    frame = {'operation_id': 'openrouter.responses', 'parameters': continuation('arguments', 82130),
             'timeout_ms': 1000, 'action_sequence': 0}
    with TestClient(create_app(service)) as client:
        answer = client.post('/arena/v1/runs/r1/provider', content=contracts.canonical_json(frame),
            headers={'x-lab-arena-lease': 'a'*64})
    assert answer.status_code == 200 and answer.json()['status'] == 400
    assert answer.json()['call']['error_code'] == 'invalid_request'
    assert transport.sent == [] and len(reads) == 1


def test_api_total_envelope_cap_rejects_before_context_or_provider_dispatch():
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    service, transport, reads = api_service()
    frame = {'operation_id': 'openrouter.responses', 'parameters': continuation('arguments', 1_100_001),
             'timeout_ms': 1000, 'action_sequence': 0}
    with TestClient(create_app(service)) as client:
        answer = client.post('/arena/v1/runs/r1/provider', content=contracts.canonical_json(frame),
            headers={'x-lab-arena-lease': 'a'*64})
    assert answer.status_code == 413
    assert transport.sent == [] and reads == []


@pytest.mark.parametrize('kind,size', [('reasoning', 32414), ('arguments', 51314), ('arguments', 82130)])
def test_saved_training_replay_retains_exact_long_history(kind, size):
    from tests.lab_arena.training_native_responses_test import example, refresh
    from tests.lab_arena.training_steps_test import records
    value = example()
    body = value['provider_calls'][0]['request']['outbound']['body']
    body['input'] = continuation(kind, size)['input']
    body['tools'] = [{'type': 'function', 'name': 'review', 'parameters': {'type': 'object'}}]
    refresh(value)
    original = deepcopy(value)
    output, _ = records(value)
    assert len(output) == 1
    assert output[0]['input']['input'] == body['input']
    assert value == original
