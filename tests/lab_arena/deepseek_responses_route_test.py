"""Saved synthetic Responses proof binds the seventh reviewed exact route."""

import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from lab_arena import broker as br, contracts, reasoning_capture as rc
from lab_arena import training_capture, training_export, training_steps, training_thinking
from tests.lab_arena import test_lab_arena_broker as broker_tests


PATH = Path(__file__).parents[2] / 'docs/evidence/full-thinking/deepseek-fireworks-responses-20261025.json'
PROOF = json.loads(PATH.read_text())


def test_seventh_route_is_evidence_bound_and_does_not_select_a_baseline():
    registry = rc.load_registry()
    assert len(registry['approved_routes']) >= 7
    assert registry['baseline_route'] is None
    routes = [route for route in registry['approved_routes']
              if route['api'] == 'openrouter.responses' and route['provider'] == 'fireworks']
    assert len(routes) == 1
    route = routes[0]
    assert registry['approved_routes'][6] == route
    assert route['model_id'] == PROOF['model_id'] == 'deepseek/deepseek-v4.1-flash-1017'
    assert route['provider'] == PROOF['provider'] == 'fireworks'
    assert route['mode'] == {'effort': 'high'}
    assert route['full_text_fields'] == PROOF['full_text_fields'] == ['output.content.reasoning_text']
    assert route['max_output_tokens'] == PROOF['max_output_tokens'] == 32768
    assert route['evidence']['probe_manifest_sha256'] == hashlib.sha256(PATH.read_bytes()).hexdigest()
    assert route['evidence']['probe_id'] == PROOF['probe_id']
    assert route['evidence']['documentation_url'] == PROOF['semantic_source']
    assert all(PROOF['checks'].values())


def test_actual_responses_wire_readable_and_strict_targets_replay_through_current_broker():
    example = PROOF['example']
    assert example['example_hash'] == contracts.document_hash({k: v for k, v in example.items() if k != 'example_hash'})
    assert training_capture.sanitize_training_value(example) == (example, [])
    assert json.loads(contracts.canonical_json(example)) == example
    trace = training_thinking.readable_trace(example)
    assert trace['projection_truncated'] is False
    assert contracts.document_hash(trace) == PROOF['readable_trace_sha256']
    markdown = training_thinking.readable_trace_markdown(trace)
    registry = rc.load_registry()
    route = next(r for r in registry['approved_routes'] if r['api'] == 'openrouter.responses')
    table = broker_tests.price_table()
    table['models'][route['model_id']] = {
        component: route['price_ceiling'].get(component, '0') for component in br.PRICED_COMPONENTS}
    raw_values = [json.loads(base64.b64decode(row['body_b64'], validate=True)) for row in PROOF['wire_responses']]
    transport = broker_tests.FakeTransport([(200, response) for response in raw_values])
    broker = br.Broker(store=broker_tests.FakeLedgerStore(), key_for=lambda p: broker_tests.HOST_KEYS[p],
                       price_table=br.validate_price_table(table), transport=transport,
                       clock=lambda: datetime(2026, 9, 2, 1, tzinfo=timezone.utc))
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
    candidates = []
    assert len(example['provider_calls']) == len(PROOF['wire_responses']) == 3
    for index, (call, wire, raw_value) in enumerate(zip(example['provider_calls'], PROOF['wire_responses'], raw_values)):
        raw = base64.b64decode(wire['body_b64'], validate=True)
        assert contracts.hash_bytes(raw) == wire['wire_body_sha256']
        assert len(raw) == wire['wire_body_bytes']
        assert training_export.without_hidden_reasoning(raw_value) == (call['response'], False)
        receipt = call['response_receipts'][0]
        assert (receipt['wire_body_sha256'], receipt['wire_body_bytes']) == (wire['wire_body_sha256'], wire['wire_body_bytes'])
        assert call['response_status'] == 200 and call['accounting_state'] == 'settlement' and not call['evidence_gaps']
        ledger = [row for row in PROOF['ledger_readback']['rows'] if row['call_identity'] == call['call_identity']]
        assert {row['entry_kind'] for row in ledger} == {'reservation', 'dispatch', 'settlement'}
        reserved = next(row for row in ledger if row['entry_kind'] == 'reservation')
        settled = next(row for row in ledger if row['entry_kind'] == 'settlement')
        assert reserved['request_hash'] == call['request_hash']
        assert settled['amount_microusd'] == call['actual_microusd']
        assert settled['entry_id'] == receipt['entry_id']
        assert base64.b64decode(settled['terminal_response']['body_b64'], validate=True) == raw
        candidates.append(training_steps._candidate(example, call))
        assert candidates[-1]['source_refs'][0]['delivered_response_sha256'] == call['response_sha256']
        full, status = training_thinking.full_text_segments(call, 'miner', example['configuration']['reasoning_route_registry'])
        assert len(full) == 1 and full[0]['field'] == 'output.content.reasoning_text'
        assert status['route_reviewed'] and status['provider_route_verified']
        assert full[0]['text'] in markdown
        request = call['request']['outbound']['body']
        assert request['max_output_tokens'] == 32768
        assert 'parallel_tool_calls' not in request
        parameters = {k: v for k, v in request.items() if k not in {'provider', 'stream', 'store'}}
        result = broker.execute(context, operation_id='openrouter.responses', parameters=parameters,
                                action_sequence=index, timeout_ms=30000)
        assert result.status == 200
        assert json.loads(transport.sent[-1]['body']) == request
        assert json.loads(result.body) == raw_value
    assert [candidate['target'][0]['type'] for candidate in candidates] == ['function_call', 'function_call', 'message']
    final = json.loads(candidates[-1]['target'][0]['content'][0]['text'])
    assert final == {'company': 'Acme Test', 'qualified': False, 'evidence_ids': ['profile', 'headcount']}
    # A strict next-response target is not a complete labeled native workflow.
    assert example['completeness']['complete'] is False
    assert 'scoring_label_unavailable' in example['completeness']['incomplete_reasons']


def test_public_manifest_has_only_synthetic_content_and_decoded_wires_have_no_secrets():
    def check(value):
        if isinstance(value, dict):
            assert not any(re.search(r'authorization|api.?key|password|secret|credential|cookie', key, re.I) for key in value)
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str):
            assert not any(marker in value for marker in ('/Users/', '/tmp/', 'sk-or-v1-', 'BEGIN PRIVATE KEY'))
            assert not re.search(r'Bearer\s+\S+', value)
    check(PROOF)
    for wire in PROOF['wire_responses']:
        check(json.loads(base64.b64decode(wire['body_b64'], validate=True)))
    for row in PROOF['ledger_readback']['rows']:
        if row.get('terminal_response'):
            check(json.loads(base64.b64decode(row['terminal_response']['body_b64'], validate=True)))
    assert 'not complete native TYCHE workflow' in PROOF['proof_limit']
