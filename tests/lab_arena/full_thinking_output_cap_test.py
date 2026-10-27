"""Current output caps bind new live evidence; old captures keep frozen policy."""

import base64
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc
from lab_arena import training_capture, training_steps, training_thinking
from tests.lab_arena import test_lab_arena_broker as broker_tests
from tests.lab_arena.full_thinking_saved_probe_test import historical_registry


MANIFEST_PATH = Path(__file__).parents[2] / 'docs/evidence/full-thinking/chat-output-cap-32768-20261025.json'
MANIFEST = json.loads(MANIFEST_PATH.read_text())
PASSED_MODELS = {row['model_id'] for row in MANIFEST['captures']}


def test_current_caps_and_source_bounds_are_evidence_bound_without_rewriting_old_routes():
    registry = rc.load_registry()
    original = historical_registry()
    assert PASSED_MODELS == {
        'mistralai/mistral-large-4-0', 'deepseek/deepseek-v4.1-flash-1017',
        'reflection/beam-1.0-501b-a23b', 'google/gemma-4-31b-it',
    }
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    for old, route in zip(original['approved_routes'], registry['approved_routes']):
        assert route['model_id'] == old['model_id']
        assert old['max_output_tokens'] == 8192
        if route['model_id'] in PASSED_MODELS:
            assert route['max_output_tokens'] == operations.OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS == 32768
            assert route['evidence']['probe_manifest_sha256'] == digest
            assert {k: v for k, v in route.items() if k not in {'max_output_tokens', 'evidence'}} == {
                k: v for k, v in old.items() if k not in {'max_output_tokens', 'evidence'}}
        else:
            assert route == old
            assert route['model_id'] in {'openai/gpt-oss-120b', 'openai/gpt-oss-20b'}
    bad = json.loads(json.dumps(registry))
    bad['approved_routes'][1]['max_output_tokens'] = 32769
    with pytest.raises(ValueError):
        rc.validate_registry(bad)
    assert all(route['api'] == 'openrouter.chat' for route in registry['approved_routes'][:6])
    assert [(route['model_id'], route['api']) for route in registry['approved_routes'][6:7]] == [
        ('deepseek/deepseek-v4.1-flash-1017', 'openrouter.responses')]


@pytest.mark.parametrize('capture', MANIFEST['captures'], ids=lambda row: row['model_id'])
def test_real_cap_wire_and_worker_readback_roundtrip_through_current_broker(capture):
    assert all(capture['checks'].values())
    example = capture['example']
    assert json.loads(contracts.canonical_json(example)) == example
    assert example['example_hash'] == contracts.document_hash({k: v for k, v in example.items() if k != 'example_hash'})
    assert training_capture.sanitize_training_value(example) == (example, [])
    trace = training_thinking.readable_trace(example)
    # The immutable digest predates indexing Chat choice/message metadata.
    # Removing only the new exact metadata events must recover that view,
    # including the original reasoning/tool order and every source hash.
    legacy_trace = deepcopy(trace)
    for call in legacy_trace['calls']:
        call['events'] = [event for event in call['events']
                          if not (event['kind'] == 'response_metadata'
                                  and 'choices' in event['source_path'])]
    assert contracts.document_hash(legacy_trace) == capture['readable_trace_sha256']
    for call in trace['calls']:
        for event in call['events']:
            if 'source_path' not in event:
                continue
            value = example
            for key in event['source_path']:
                value = value[key]
            assert event['value'] == value
            assert event['value_sha256'] == contracts.document_hash(value)
    assert trace['projection_truncated'] is False
    markdown = training_thinking.readable_trace_markdown(trace)
    registry = rc.load_registry()
    route = next(row for row in registry['approved_routes'] if row['model_id'] == capture['model_id'])
    table = broker_tests.price_table()
    table['models'][capture['model_id']] = {
        component: route['price_ceiling'].get(component, '0') for component in br.PRICED_COMPONENTS}
    transport = broker_tests.FakeTransport([(200, call['response']) for call in example['provider_calls']])
    broker = br.Broker(store=broker_tests.FakeLedgerStore(), key_for=lambda p: broker_tests.HOST_KEYS[p],
                       price_table=br.validate_price_table(table), transport=transport,
                       clock=lambda: datetime(2026, 9, 2, 1, tzinfo=timezone.utc))
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
    retained_full = 0
    for index, (call, wire) in enumerate(zip(example['provider_calls'], capture['wire_responses'])):
        raw = base64.b64decode(wire['body_b64'], validate=True)
        assert contracts.hash_bytes(raw) == wire['wire_body_sha256']
        assert len(raw) == wire['wire_body_bytes']
        assert json.loads(raw) == call['response']
        receipt = call['response_receipts'][0]
        assert receipt['wire_body_sha256'] == wire['wire_body_sha256']
        assert receipt['wire_body_bytes'] == wire['wire_body_bytes']
        assert call['accounting_state'] == 'settlement' and not call['evidence_gaps']
        ledger = [row for row in capture['ledger_readback']['rows'] if row['call_identity'] == call['call_identity']]
        assert {row['entry_kind'] for row in ledger} == {'reservation', 'dispatch', 'settlement'}
        reserved = next(row for row in ledger if row['entry_kind'] == 'reservation')
        settled = next(row for row in ledger if row['entry_kind'] == 'settlement')
        assert reserved['request_hash'] == call['request_hash']
        assert settled['entry_id'] == receipt['entry_id']
        assert settled['amount_microusd'] == call['actual_microusd']
        assert base64.b64decode(settled['terminal_response']['body_b64']) == raw
        # This is the actual exact request/receipt/agent-read validator used
        # by strict export, not a report flag or a hash-only projection.
        candidate = training_steps._candidate(example, call)
        assert candidate['source_refs'][0]['delivered_response_sha256'] == call['response_sha256']
        full, status = training_thinking.full_text_segments(
            call, 'miner', example['configuration']['reasoning_route_registry'])
        if full:
            retained_full += 1
            assert all(segment['text'] in markdown for segment in full)
        else:
            assert training_thinking.verified_no_thinking(call, call['response'], status)
        request = call['request']['outbound']['body']
        assert request['max_tokens'] == 32768
        parameters = {k: v for k, v in request.items() if k not in {'provider', 'stream'}}
        result = broker.execute(context, operation_id='openrouter.chat', parameters=parameters,
                                action_sequence=index, timeout_ms=30000)
        assert result.status == 200
        assert json.loads(transport.sent[-1]['body']) == request
        assert json.loads(result.body) == call['response']
    assert retained_full > 0
    final = json.loads(example['provider_calls'][-1]['response']['choices'][0]['message']['content'])
    assert capture['scenario'] == 'conflicting_evidence'
    assert final['company'] == 'Acme Test' and final['qualified'] is False
    assert set(final['evidence_ids']) == {'profile', 'headcount'}


def test_supplement_has_no_credentials_or_local_paths_and_keeps_limits_explicit():
    raw = MANIFEST_PATH.read_text()
    for marker in ('/Users/', '/tmp/', 'sk-or-v1-', 'Authorization', 'api_key', 'password'):
        assert marker not in raw
    assert 'do not exhaust the envelope' in MANIFEST['proof_limit']
    assert 'production' in MANIFEST['proof_limit']
