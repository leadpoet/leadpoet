"""Only redundant true is omitted from new full-thinking Responses wires."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from lab_arena import broker as br, operations, reasoning_capture as rc, training_capture
from lab_arena.service import LocalObjectStore
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.reasoning_capture_test import _reviewed_registry

FIXTURE = Path(__file__).parents[1] / 'fixtures/research_review/native_responses_parameters.json'
PIN = {'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False,
       'only': ['example/verified'], 'require_parameters': True,
       'max_price': {'prompt': 1, 'completion': 1, 'request': 0}}


def native_request():
    return json.loads(FIXTURE.read_text())['request']


@pytest.mark.parametrize('operation,full,value,retained', [
    ('openrouter.responses', True, True, False),
    ('openrouter.responses', True, False, True),
    ('openrouter.responses', False, True, True),
    ('openrouter.responses', False, False, True),
    ('openrouter.chat', True, True, True),
    ('openrouter.chat', True, False, True),
])
def test_only_full_responses_redundant_true_changes_wire(operation, full, value, retained):
    parameters = {'model': 'example/model', 'parallel_tool_calls': value,
        **({'input': 'Inspect retained evidence.'} if operation.endswith('responses')
           else {'messages': [{'role': 'user', 'content': 'Inspect retained evidence.'}]})}
    original = deepcopy(parameters)
    wire = operations.build_outbound_request(operation, parameters,
        allow_full_thinking=full, openrouter_provider_policy=PIN if full else None)
    body = json.loads(wire.body)
    assert ('parallel_tool_calls' in body) is retained
    if retained:
        assert body['parallel_tool_calls'] is value
    assert parameters == original


def test_real_native_parameter_shape_preserves_hash_capture_two_calls_and_both_tool_results(tmp_path):
    request = native_request()
    original = deepcopy(request)
    model = request['model']
    name = request['tools'][0]['name']
    calls = [{'type': 'function_call', 'id': 'fn-' + suffix, 'call_id': 'call-' + suffix,
        'name': name, 'arguments': '{"cmd":"printf retained"}', 'status': 'completed'}
        for suffix in ('a', 'b')]
    class Transport:
        def __init__(self): self.sent = []
        def send(self, **kwargs):
            body = json.loads(kwargs['body'])
            self.sent.append(body)
            if body.get('parallel_tool_calls') is True:
                return br.ProviderResponse(404, {}, b'{"error":{"code":404,"message":"No endpoints found that can handle the requested parameters"}}')
            output = ([{'type': 'reasoning', 'id': 'rs1', 'content': [
                {'type': 'reasoning_text', 'text': 'Read the two retained observations.'}]}] + calls
                if len(self.sent) == 1 else [{'type': 'message', 'id': 'msg2', 'role': 'assistant',
                    'status': 'completed', 'content': [{'type': 'output_text', 'text': '{"complete":true}'}]}])
            return br.ProviderResponse(200, {}, json.dumps({'object': 'response', 'model': model,
                'status': 'completed', 'output': output, 'usage': {'cost': '0.00001'}}).encode())
    transport = Transport()
    objects = LocalObjectStore(tmp_path / 'objects')
    table = bt.price_table()
    table['models'][model] = dict(next(iter(table['models'].values())))
    broker = br.Broker(store=bt.FakeLedgerStore(), key_for=lambda _: 'fixture-key',
        price_table=br.validate_price_table(table), transport=transport,
        capture_request=training_capture.RequestCapture(objects).prepare)
    context = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=_reviewed_registry(model=model, api='openrouter.responses'))
    first = broker.execute(context, operation_id='openrouter.responses', parameters=request,
                           action_sequence=0, timeout_ms=30000)
    assert first.status == 200
    assert json.loads(first.body)['output'][1:] == calls
    saved = broker._store.calls[first.call['call_identity']]
    captured = training_capture.read_request(objects, saved['call_doc']['request_capture'])
    for field in ('admitted_parameters', 'normalized_parameters', 'hashed_parameters', 'effective_parameters'):
        assert captured[field]['parallel_tool_calls'] is True
    assert captured['request_hash'] == br.contracts.document_hash(captured['hashed_parameters'])
    assert 'parallel_tool_calls' not in captured['outbound']['body']
    assert captured['outbound']['body']['provider']['require_parameters'] is True
    outputs = [{'type': 'function_call_output', 'call_id': call['call_id'],
                'output': [{'type': 'input_text', 'text': 'Result ' + call['call_id']}]} for call in calls]
    second_request = deepcopy(request)
    second_request['input'] += json.loads(first.body)['output'] + outputs
    second = broker.execute(context, operation_id='openrouter.responses', parameters=second_request,
                            action_sequence=1, timeout_ms=30000)
    assert second.status == 200
    assert transport.sent[1]['input'][-2:] == outputs
    assert request == original
    before = len(transport.sent)
    replay = broker.execute(context, operation_id='openrouter.responses', parameters=request,
                            action_sequence=0, timeout_ms=30000)
    assert replay.body == first.body and len(transport.sent) == before == 2
