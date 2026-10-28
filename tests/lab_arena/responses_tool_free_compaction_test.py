"""Native tool-free compaction must not acquire tools or require their controls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from lab_arena import broker, contracts, lab_arena_codex as codex, operations, reasoning_capture, training_capture
from lab_arena.service import LocalObjectStore
from tests.lab_arena import codex_runtime_test as runtime_fixture
from tests.lab_arena import test_lab_arena_broker as broker_fixture
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.responses_parallel_default_test import PIN, native_request

FIXTURE = Path(__file__).parents[1] / 'fixtures/research_review/native_tool_free_compaction_parameters.json'


def compaction_request():
    return json.loads(FIXTURE.read_text())['request']


def full_wire(request):
    return json.loads(operations.build_outbound_request('openrouter.responses', request,
        allow_full_thinking=True, openrouter_provider_policy=PIN).body)


@pytest.mark.parametrize('choice', ['auto', 'none'])
def test_tool_free_serial_default_is_wire_only_and_history_is_unchanged(choice):
    request = compaction_request()
    request['tool_choice'] = choice
    original = deepcopy(request)
    wire = full_wire(request)
    assert 'parallel_tool_calls' not in wire
    assert wire['tools'] == []
    assert wire['input'] == request['input']
    assert request == original
    assert wire['provider'] == PIN
    # Historical routes keep their old wire contract.
    historical = json.loads(operations.build_outbound_request('openrouter.responses', request).body)
    assert historical['parallel_tool_calls'] is False


@pytest.mark.parametrize('tool', [
    {'type': 'function', 'name': 'inspect', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'type': 'custom', 'name': 'exec', 'format': {'type': 'text'}},
    {'type': 'namespace', 'name': 'local', 'description': 'Local tools',
     'tools': [{'type': 'function', 'name': 'inspect', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}]},
    {'type': 'openrouter:web_search', 'parameters': {'engine': 'native', 'max_uses': 1, 'max_total_results': 5}},
])
def test_real_tools_keep_serial_control(tool):
    request = compaction_request()
    request['tools'] = [tool]
    if tool['type'] == 'openrouter:web_search':
        request['max_tool_calls'] = 1
    wire = full_wire(request)
    assert wire['parallel_tool_calls'] is False
    assert wire['tools'] == request['tools']


@pytest.mark.parametrize('tools', [[], [{'type': 'function', 'name': 'inspect',
                                      'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}]])
def test_additional_declarations_keep_serial_control_even_when_empty(tools):
    request = compaction_request()
    request['input'].insert(0, {'type': 'additional_tools', 'role': 'developer', 'tools': tools})
    assert full_wire(request)['parallel_tool_calls'] is False


@pytest.mark.parametrize('change', [
    {'tools': [{'type': 'future_tool'}]},
    {'tools': {}},
    {'input': [{'type': 'future_tool_declaration', 'tools': []}]},
    {'input': [{'type': 'additional_tools', 'role': 'developer', 'tools': [{'type': 'future_tool'}]}]},
    {'parallel_tool_calls': 0},
])
def test_unknown_or_malformed_controls_still_fail_admission(change):
    request = compaction_request()
    request.update(change)
    assert not codex._tool_free_serial_request(request)
    with pytest.raises(operations.OperationRequestError):
        full_wire(request)


@pytest.mark.parametrize('variant,expect_search,expect_serial', [
    ('compaction', False, False),
    ('research', True, False),
    ('serial_function', True, True),
    ('parallel_empty', True, False),
    ('additional_tools', True, True),
])
def test_native_bridge_broker_capture_and_wire_preserve_tool_intent(monkeypatch, tmp_path,
                                                                    variant, expect_search, expect_serial):
    request = compaction_request()
    if variant in ('research', 'serial_function'):
        request['tools'] = native_request()['tools']
    if variant in ('research', 'parallel_empty'):
        request['parallel_tool_calls'] = True
    if variant == 'additional_tools':
        request['input'].insert(0, {'type': 'additional_tools', 'role': 'developer',
                                   'tools': [native_request()['tools'][0]]})
    original = deepcopy(request)
    objects = LocalObjectStore(tmp_path / 'objects')
    registry = _reviewed_registry(model=request['model'], api='openrouter.responses')
    context = replace(broker_fixture.CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
                      reasoning_route_registry=registry)

    class Transport:
        def __init__(self):
            self.sent = []

        def send(self, **kwargs):
            self.sent.append(kwargs)
            body = json.loads(kwargs['body'])
            # Reproduce the saved upstream parameter rejection for the compact
            # shape, without any network call or paid provider request.
            if variant == 'compaction' and ('parallel_tool_calls' in body or body.get('tools')):
                return broker.ProviderResponse(404, {}, b'{"error":{"code":404,"message":"No endpoints found that can handle the requested parameters"}}')
            reply = runtime_fixture.response(model=request['model'], output=[
                {'type': 'reasoning', 'id': 'rs-summary', 'summary': [], 'content': [
                    {'type': 'reasoning_text', 'text': 'Retain the saved facts without taking actions.'}]},
                {'type': 'message', 'id': 'msg-summary', 'role': 'assistant', 'status': 'completed',
                 'content': [{'type': 'output_text', 'text': 'Synthetic task summary.'}]}])
            return broker.ProviderResponse(200, {}, json.dumps(reply).encode())

    transport = Transport()
    make_broker = runtime_fixture.make_broker

    def captured_broker(**kwargs):
        service, store, selected_transport = make_broker(**kwargs)
        service._capture_request = training_capture.RequestCapture(objects).prepare
        return service, store, selected_transport

    monkeypatch.setattr(runtime_fixture, 'make_broker', captured_broker)
    monkeypatch.setattr(runtime_fixture, 'CONTEXT', context)
    with runtime_fixture.broker_socket(monkeypatch, transport, priced_models=(request['model'],)) as (store, _, path):
        with codex.ResponsesBridge(str(path), web_search='live') as bridge:
            with httpx.Client(trust_env=False) as client:
                result = client.post(bridge.base_url + '/responses',
                    headers={'Authorization': 'Bearer ' + bridge.token},
                    json=dict(request, stream=True, store=False, client_metadata={'source': 'synthetic'}))
    assert result.status_code == 200, result.text
    assert 'Synthetic task summary.' in result.text
    assert request == original
    assert len(transport.sent) == 1
    wire = json.loads(transport.sent[0]['body'])
    assert ('parallel_tool_calls' in wire) is expect_serial
    if expect_serial:
        assert wire['parallel_tool_calls'] is False
    assert any(tool['type'] == 'openrouter:web_search' for tool in wire['tools']) is expect_search
    assert wire['input'] == request['input']
    assert 'client_metadata' not in wire
    assert wire['provider']['only'] == ['example/verified']
    assert wire['provider']['allow_fallbacks'] is False
    assert wire['provider']['require_parameters'] is True
    assert wire['provider']['data_collection'] == 'deny' and wire['provider']['zdr'] is True
    assert store.log == ['reserve', 'dispatch', 'settle']
    saved = next(iter(store.calls.values()))
    captured = training_capture.read_request(objects, saved['call_doc']['request_capture'])
    for field in ('admitted_parameters', 'normalized_parameters', 'hashed_parameters', 'effective_parameters'):
        assert captured[field]['parallel_tool_calls'] is request['parallel_tool_calls']
        assert captured[field]['input'] == request['input']
    assert captured['request_hash'] == contracts.document_hash(captured['hashed_parameters'])
    assert captured['outbound']['body'] == wire
