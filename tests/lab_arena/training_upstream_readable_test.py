"""Failed upstream output remains visible without becoming an agent target."""
import base64
import copy
import json

import pytest

from lab_arena import broker, contracts, operations, training_export, training_thinking
from tests.lab_arena.training_export_test import add_call, fixture
from tests.lab_arena.training_thinking_test import full_example, rows, seal


def upstream_example(operation, *, redacted=False):
    values = fixture()
    add_call(values, operation=operation)
    reserve, _, terminal = values['ledger_rows']
    # November 15 Responses failures returned a reasoning item and usage before
    # the provider failed. The broker correctly returned a generic 502 instead.
    thinking = 'Check the dated source before qualifying this company.'
    response = {'status': 'failed', 'error': {'message': 'provider interrupted'},
                'usage': {'output_tokens_details': {'reasoning_tokens': 425}}}
    if operation == 'openrouter.responses':
        response['output'] = [
            {'type': 'reasoning', 'status': 'completed', 'content': [
                {'type': 'reasoning_text', 'text': thinking}]},
            {'type': 'function_call', 'call_id': 'lookup-1', 'name': 'lookup',
             'arguments': '{"url":"https://example.com/source"}'},
        ]
    else:
        response['choices'] = [{'finish_reason': 'error', 'message': {
            'role': 'assistant', 'content': None, 'reasoning_content': thinking,
            'tool_calls': [{'id': 'lookup-1', 'type': 'function', 'function': {
                'name': 'lookup', 'arguments': '{"url":"https://example.com/source"}'}}],
        }}]
    if redacted:
        response['api_key'] = 'private-test-secret'
        response['encrypted_content'] = 'private-opaque-state'
    captured = broker._capture_upstream_response(
        broker.ProviderResponse(200, {}, json.dumps(response).encode()),
        operation_id=operation, call_identity=reserve['call_identity'],
        request_hash=reserve['entry_doc']['request_hash'])
    terminal['terminal_response'] = {
        'status': 502, 'body_b64': base64.b64encode(operations.GENERIC_UNAVAILABLE_BODY).decode(),
        'upstream_response_receipt': captured}
    return training_export.assemble_example(**values), thinking


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('redacted', [False, True])
def test_saved_upstream_reasoning_and_tools_have_exact_readable_paths(operation, redacted):
    example, thinking = upstream_example(operation, redacted=redacted)
    original = copy.deepcopy(example)
    report = training_thinking.readable_trace(example)
    call = report['calls'][0]
    root = ['provider_calls', 0, 'upstream_response_receipts', 0, 'response']
    events = [e for e in call['events'] if e.get('source_path', [])[:len(root)] == root]
    assert events
    assert {e['delivery'] for e in events} == {'not_proved'}
    reasoning = [e for e in events if e['kind'] == 'unverified_reasoning_text']
    assert [e['value'] for e in reasoning] == [thinking]
    assert not any(e['kind'] == 'full_generated_thinking' for e in events)
    assert len([e for e in events if e['kind'] == 'tool_call']) == 1
    assert len([e for e in events if e['kind'] == 'usage']) == 1
    if operation == 'openrouter.responses':
        outputs = [e for e in events if e['kind'] in {'reasoning_item', 'tool_call'}]
        assert [e['source_path'][-1] for e in outputs] == [0, 1]
    for event in events:
        if event.get('derived'):
            continue
        value = example
        for key in event['source_path']:
            value = value[key]
        assert event['value'] == value
        assert event['value_sha256'] == contracts.document_hash(value)
    rendered = training_thinking.readable_trace_markdown(report)
    assert '#### unverified_reasoning_text' in rendered
    assert 'Receipt source: `not_proved`' in rendered
    assert 'private-test-secret' not in rendered and 'private-opaque-state' not in rendered
    assert example == original
    assert example['provider_calls'][0]['response'] == json.loads(operations.GENERIC_UNAVAILABLE_BODY)
    assert report['projection_truncated'] is False
    assert not rows(example)[0]


def test_upstream_reasoning_does_not_change_successful_targets_or_coverage():
    example = full_example()
    before = rows(copy.deepcopy(example))
    before_coverage = training_thinking.readable_trace(seal(example))['calls'][0]['reasoning_coverage']
    failed, _ = upstream_example('openrouter.chat')
    example['provider_calls'][0]['upstream_response_receipts'] = failed['provider_calls'][0]['upstream_response_receipts']
    seal(example)
    report = training_thinking.readable_trace(example)
    assert report['calls'][0]['reasoning_coverage'] == before_coverage
    after = rows(example)
    assert len(after[0]) == len(before[0]) == 1
    for key in ('candidate_id', 'input', 'target', 'generated_thinking', 'quality'):
        assert after[0][0][key] == before[0][0][key]
    assert after[1]['coverage'] == before[1]['coverage']
    assert sum(e['kind'] == 'full_generated_thinking' for e in report['calls'][0]['events']) == 1


def test_unavailable_upstream_does_not_invent_events():
    example = full_example()
    before = training_thinking.readable_trace(seal(example))['calls'][0]['events']
    example['provider_calls'][0]['upstream_response_receipts'] = [
        {'response_unavailable': True, 'delivery_to_worker': 'not_proved'}]
    assert training_thinking.readable_trace(seal(example))['calls'][0]['events'] == before
