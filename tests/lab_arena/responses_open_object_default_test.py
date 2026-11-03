"""Make native non-strict empty maps explicit only on full Responses wires."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture, training_capture
from lab_arena.service import LocalObjectStore
from tests.lab_arena import test_lab_arena_broker as broker_fixture
from tests.lab_arena.reasoning_capture_test import _reviewed_registry

PIN = {'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False,
       'only': ['example/verified'], 'require_parameters': True,
       'max_price': {'prompt': 1, 'completion': 1, 'request': 0}}


def native_tool():
    # Exact schema shape from native-schema-native_open_object's saved request.
    return {'type': 'function', 'name': 'tyche_lookup', 'strict': False,
        'parameters': {'type': 'object', 'additionalProperties': False,
            'properties': {'checks': {'type': 'array', 'minItems': 1,
                'items': {'type': 'object', 'additionalProperties': False,
                    'properties': {
                        'approach': {'type': 'string'},
                        'inputs': {'type': 'object', 'properties': {}},
                        'phase': {'type': 'string', 'enum': ['account_discovery', 'account_verification']},
                        'provider': {'type': 'string', 'enum': ['deepline', 'scrapingdog']},
                        'purpose': {'type': 'string'}, 'status_read': {'type': 'boolean'},
                        'target': {'type': 'string'}, 'tool': {'type': 'string'}},
                    'required': ['target', 'purpose', 'inputs']}}},
            'required': ['checks']}}


def parameters(tool=None):
    return {'model': 'example/model', 'input': 'Read https://www.microsoft.com/en-us/about',
            'tools': [native_tool() if tool is None else tool]}


def wire(request, *, full=True, operation='openrouter.responses'):
    return json.loads(operations.build_outbound_request(
        operation, request, allow_full_thinking=full,
        openrouter_provider_policy=PIN if full else None).body)


def inputs(schema):
    return schema['properties']['checks']['items']['properties']['inputs']


def test_real_native_schema_has_only_open_map_default_made_explicit():
    request = parameters()
    original = deepcopy(request)
    normalized = operations.validate_operation_request('openrouter.responses', request, allow_full_thinking=True)
    normalized_before = deepcopy(normalized)
    body = wire(normalized)
    expected = deepcopy(original['tools'][0]['parameters'])
    inputs(expected)['additionalProperties'] = True
    assert body['tools'][0]['parameters'] == expected
    assert normalized == normalized_before and request == original


@pytest.mark.parametrize('additional', [False, True, {}, {'type': 'string'},
                                       {'type': 'object', 'properties': {}}])
def test_explicit_additional_properties_and_limits_are_preserved(additional):
    tool = native_tool()
    target = inputs(tool['parameters'])
    target.update(additionalProperties=deepcopy(additional), minProperties=1,
                  maxProperties=3, required=['url'])
    original = deepcopy(tool)
    assert wire(parameters(tool))['tools'][0] == original


@pytest.mark.parametrize('strict', [False, None, True])
def test_strict_true_is_never_modified(strict):
    tool = native_tool()
    if strict is None:
        tool.pop('strict')
    else:
        tool['strict'] = strict
    result = wire(parameters(tool))['tools'][0]
    assert ('additionalProperties' in inputs(result['parameters'])) is (strict is not True)
    if strict is True:
        assert result == tool


def test_policy_off_and_chat_do_not_receive_schema_projection():
    tool = native_tool()
    assert wire(parameters(tool), full=False)['tools'][0] == tool
    chat = {'model': 'example/model', 'messages': [{'role': 'user', 'content': 'Read the source.'}],
            'tools': [{'type': 'function', 'function': {key: deepcopy(value)
                       for key, value in tool.items() if key != 'type'}}]}
    assert wire(chat, operation='openrouter.chat')['tools'] == chat['tools']


@pytest.mark.parametrize('keyword,value', [
    ('$ref', '#/$defs/entry'), ('$dynamicRef', '#entry'), ('$defs', {'entry': {'type': 'object'}}),
    ('allOf', [{'required': ['url']}]), ('anyOf', [{'type': 'object'}]),
    ('oneOf', [{'type': 'object'}]), ('not', {'required': ['forbidden']}),
    ('if', {'required': ['url']}), ('unevaluatedProperties', False),
    ('patternProperties', {'.*': {'type': 'string'}}),
    ('dependentSchemas', {'url': {'required': ['token']}}),
    ('dependencies', {'url': ['token']}),
])
def test_nonordinary_schema_applicators_leave_whole_parameter_schema_unchanged(keyword, value):
    tool = native_tool()
    tool['parameters'][keyword] = deepcopy(value)
    assert wire(parameters(tool))['tools'][0] == tool


def test_nested_unevaluated_rules_and_refs_are_not_relaxed():
    tool = native_tool()
    inputs(tool['parameters'])['unevaluatedProperties'] = False
    assert wire(parameters(tool))['tools'][0] == tool
    target = inputs(tool['parameters'])
    target.pop('unevaluatedProperties')
    target['additionalProperties'] = {'$ref': '#/closed'}
    assert wire(parameters(tool))['tools'][0] == tool


def test_only_schema_locations_are_walked_not_examples_defaults_or_output_data():
    tool = native_tool()
    data = {'type': 'object', 'properties': {}, '$ref': 'literal data',
            'allOf': [], 'unevaluatedProperties': False}
    target = inputs(tool['parameters'])
    target.update(default=deepcopy(data), examples=[deepcopy(data)], enum=[deepcopy(data)])
    tool['parameters']['properties']['output'] = {'type': 'string', 'default': deepcopy(data)}
    tool['output_schema'] = {'type': 'object', 'properties': {}}
    original = deepcopy(tool)
    result = wire(parameters(tool))['tools'][0]
    expected = deepcopy(original)
    inputs(expected['parameters'])['additionalProperties'] = True
    assert result == expected
    assert tool == original


@pytest.mark.parametrize('location', ['namespace', 'additional_tools', 'additional_namespace'])
def test_declared_namespaces_and_additional_tools_are_supported(location):
    tool = native_tool()
    namespace = {'type': 'namespace', 'name': 'lookup', 'description': 'Local tools', 'tools': [tool]}
    request = parameters(tool)
    if location == 'namespace':
        request['tools'] = [namespace]
    else:
        request['tools'] = []
        request['input'] = [{'type': 'additional_tools', 'role': 'developer',
                             'tools': [namespace if location == 'additional_namespace' else tool]},
                            {'role': 'user', 'content': 'Read the source.'}]
    original = deepcopy(request)
    result = wire(request)
    declared = (result['tools'][0]['tools'][0] if location == 'namespace' else
                result['input'][0]['tools'][0]['tools'][0] if location == 'additional_namespace' else
                result['input'][0]['tools'][0])
    assert inputs(declared['parameters'])['additionalProperties'] is True
    assert request == original


def test_nested_array_object_defaults_are_idempotent_and_preserve_required_limits():
    tool = {'type': 'function', 'name': 'generic', 'strict': False,
            'parameters': {'type': 'object', 'properties': {
                'rows': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {
                    'type': 'object', 'properties': {}, 'minProperties': 1,
                    'maxProperties': 4, 'required': ['url']}}}}}
    first = wire(parameters(tool))['tools'][0]
    second = wire(parameters(first))['tools'][0]
    assert first == second
    target = first['parameters']['properties']['rows']['items']
    assert target == {**tool['parameters']['properties']['rows']['items'], 'additionalProperties': True}


def test_schema_expansion_cannot_exceed_existing_wire_byte_limit(monkeypatch):
    request = parameters()
    expanded = wire(request)
    maximum = len(contracts.canonical_json(expanded).encode()) - 1
    real_operation = operations._operation
    monkeypatch.setattr(operations, '_operation', lambda name: replace(
        real_operation(name), max_request_bytes=maximum) if name == 'openrouter.responses' else real_operation(name))
    with pytest.raises(operations.OperationRequestError) as caught:
        wire(request)
    assert caught.value.code == 'request_too_large'


def test_broker_keeps_original_schema_hash_capture_and_replays_exact_projected_wire(tmp_path):
    request = parameters()
    original = deepcopy(request)

    class Transport:
        calls = 0

        def send(self, **kwargs):
            self.calls += 1
            body = json.loads(kwargs['body'])
            allowed = inputs(body['tools'][0]['parameters']).get('additionalProperties') is True
            arguments = {'checks': [{'inputs': {'url': 'https://www.microsoft.com/en-us/about'} if allowed else {}}]}
            return br.ProviderResponse(200, {}, json.dumps({'object': 'response', 'model': request['model'],
                'status': 'completed', 'output': [{'type': 'function_call', 'status': 'completed',
                    'id': 'fc-1', 'call_id': 'call-1', 'name': 'tyche_lookup', 'arguments': json.dumps(arguments)}],
                'usage': {'cost': '0.00001'}}).encode())

    transport = Transport()
    objects = LocalObjectStore(tmp_path / 'objects')
    table = broker_fixture.price_table()
    table['models'][request['model']] = dict(next(iter(table['models'].values())))
    broker = br.Broker(store=broker_fixture.FakeLedgerStore(), key_for=lambda _: 'fixture-key',
        price_table=br.validate_price_table(table), transport=transport,
        capture_request=training_capture.RequestCapture(objects).prepare)
    context = replace(broker_fixture.CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=_reviewed_registry(model=request['model'], api='openrouter.responses'))
    first = broker.execute(context, operation_id='openrouter.responses', parameters=request,
                           action_sequence=0, timeout_ms=30000)
    assert first.status == 200
    assert json.loads(json.loads(first.body)['output'][0]['arguments'])['checks'][0]['inputs'] == {
        'url': 'https://www.microsoft.com/en-us/about'}
    saved = broker._store.calls[first.call['call_identity']]
    capture = training_capture.read_request(objects, saved['call_doc']['request_capture'])
    for field in ('admitted_parameters', 'normalized_parameters', 'hashed_parameters', 'effective_parameters'):
        assert capture[field]['tools'] == original['tools']
    assert capture['request_hash'] == contracts.document_hash(capture['hashed_parameters'])
    outbound = capture['outbound']['body']
    assert inputs(outbound['tools'][0]['parameters'])['additionalProperties'] is True
    assert capture['outbound']['body_sha256'] == contracts.document_hash(outbound)
    assert request == original
    replay = broker.execute(context, operation_id='openrouter.responses', parameters=request,
                            action_sequence=0, timeout_ms=30000)
    assert replay.body == first.body and transport.calls == 1
