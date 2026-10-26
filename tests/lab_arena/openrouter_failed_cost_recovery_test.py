"""A returned error is not an exact bill; preserve its reply and billing identity."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, reasoning_capture
from tests.lab_arena.hosted_search_continuation_test import hosted_request
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, CONTEXT, FakeTransport, HOST_KEYS, ZeroReservationLedgerStore, make_broker,
)


class HistoryStore(ZeroReservationLedgerStore):
    """Retain the existing SQL uncertainty/history shape, including after billing."""
    def list_ledger(self, *, call_identity=None, **kwargs):
        rows = super().list_ledger(call_identity=call_identity, **kwargs)
        if not rows:
            return rows
        reservation = rows[0]
        call = self.calls[call_identity]
        result = [reservation]
        if call['kind'] != 'reservation':
            result.append(dict(reservation, entry_id=2, entry_kind='dispatch'))
        if 'uncertain_doc' in call:
            result.append(dict(reservation, entry_id=3, entry_kind='uncertain',
                entry_doc={'reason': 'worker_reported', 'call': deepcopy(call['uncertain_doc'])}))
        if call['kind'] == 'settlement':
            result.append(dict(reservation, entry_id=4, entry_kind='settlement',
                amount_microusd=call['actual'], terminal_response=call.get('terminal')))
        return result


def request(variant):
    if variant == 'chat':
        return CONTEXT, 'openrouter.chat', CHAT
    if variant == 'responses':
        return CONTEXT, 'openrouter.responses', {
            'model': CHAT['model'], 'input': 'hello', 'max_output_tokens': 200}
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=_reviewed_registry(api='openrouter.responses'))
    parameters = hosted_request()
    parameters['input'] = parameters['input'][:1]
    return context, 'openrouter.responses', parameters


def failed_call(variant='chat', status=404, *, envelope=False, generation=True, **options):
    context, operation, parameters = request(variant)
    payload = {'error': {'code': status, 'message': 'provider rejected this request'}}
    headers = {'X-Generation-Id': 'gen-failed-cost'} if generation else {}
    transport = FakeTransport([(200 if envelope else status, payload, headers),
        (404, {'error': {'code': 404, 'message': 'generation not found'}})])
    instance, store, _ = make_broker(store=HistoryStore(), transport=transport, **options)
    def execute():
        return instance.execute(context, operation_id=operation, parameters=parameters,
            action_sequence=0, timeout_ms=30000)
    return execute, instance, store, transport, payload


@pytest.mark.parametrize('variant', ['chat', 'responses', 'hosted'])
@pytest.mark.parametrize('status', [404, 429])
@pytest.mark.parametrize('envelope', [False, True])
def test_generation_error_stays_unknown_and_replays_same_worker_reply(variant, status, envelope):
    execute, _, store, transport, payload = failed_call(variant, status, envelope=envelope)
    first = execute()
    assert first.call['outcome'] == 'uncertain'
    assert first.call.get('actual_microusd') is None
    state = store.calls[first.call['call_identity']]
    assert state['uncertain_doc']['openrouter_generation_id'] == 'gen-failed-cost'
    assert state['uncertain_doc']['credential_fingerprint'] == br._credential_fingerprint(HOST_KEYS['openrouter'])
    assert store.openrouter_capacity == 10_000_000  # Existing confirmed-only admission: no hold.
    assert first.status == (404 if status == 404 else 502)
    assert json.loads(first.body) == (payload if status == 404 else {'error': {'code': 'provider_unavailable'}})
    replay = execute()
    assert (replay.status, replay.headers, replay.body) == (first.status, first.headers, first.body)
    assert replay.call['provider_status'] == status
    assert replay.call['outcome'] == 'uncertain' and replay.call.get('actual_microusd') is None
    assert [sent['method'] for sent in transport.sent] == ['POST', 'GET']


@pytest.mark.parametrize('status', [400, 401, 402, 403, 404, 422, 429])
def test_no_generation_pre_router_error_keeps_existing_zero_behavior(status):
    execute, _, store, transport, _ = failed_call(status=status, generation=False)
    result = execute()
    assert result.call['outcome'] == 'settled' and result.call['actual_microusd'] == 0
    assert [sent['method'] for sent in transport.sent] == ['POST']


@pytest.mark.parametrize('status', [401, 402, 403])
def test_generation_credential_error_keeps_worker_classification_on_replay(status):
    execute, _, store, transport, _ = failed_call(status=status,
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: 'miner_key')
    first, replay = execute(), execute()
    assert first.call['outcome'] == replay.call['outcome'] == 'uncertain'
    assert first.status == replay.status == 402
    assert first.body == replay.body == b'{"error":{"code":"miner_credentials_unavailable"}}'
    assert first.call['error_code'] == replay.call['error_code'] == 'miner_credentials_unavailable'
    assert first.call['provider_status'] == replay.call['provider_status'] == status
    assert [sent['method'] for sent in transport.sent] == ['POST', 'GET']


@pytest.mark.parametrize('change', ['credential', 'generation', 'succeeded', 'reservation', 'credential_echo'])
def test_saved_error_replay_requires_original_binding_and_sanitized_failure(change):
    execute, _, store, _, _ = failed_call()
    first = execute()
    assert first.call['outcome'] == 'uncertain'
    call = store.calls[first.call['call_identity']]
    saved = call['uncertain_doc']
    if change == 'credential':
        saved['credential_fingerprint'] = 'sha256:' + 'a' * 64
    elif change == 'generation':
        saved['openrouter_generation_id'] = 'invalid/id'
    elif change == 'succeeded':
        saved['openrouter_terminal_response']['call_succeeded'] = True
    elif change == 'reservation':
        call['call_doc']['request_hash'] = 'sha256:' + 'a' * 64
    else:
        import base64
        saved['openrouter_terminal_response']['body_b64'] = base64.b64encode(HOST_KEYS['openrouter'].encode()).decode()
    result = execute()
    assert result.call['error_code'] == 'broker_unavailable'


def test_billing_settles_between_reservation_and_history_read_without_changing_error():
    execute, _, store, transport, _ = failed_call()
    first = execute()
    read = store.list_ledger
    def settle_during_read(**kwargs):
        call = store.calls[first.call['call_identity']]
        call.update(kind='settlement', actual=19,
            terminal=br._terminal_response_document(502, {}, br.operations.GENERIC_UNAVAILABLE_BODY,
                call_succeeded=False, provider_cost={'basis': 'openrouter_generation_cost',
                    'units': '0.000019', 'unit_name': 'usd', 'operation': 'openrouter.chat',
                    'request_id': 'gen-failed-cost'}))
        return read(**kwargs)
    store.list_ledger = settle_during_read
    replay = execute()
    assert (replay.status, replay.headers, replay.body) == (first.status, first.headers, first.body)
    assert replay.call['outcome'] == 'settled' and replay.call['actual_microusd'] == 19
    assert [sent['method'] for sent in transport.sent] == ['POST', 'GET']


def test_expired_or_replaced_lease_does_not_replay_private_error():
    execute, _, store, transport, _ = failed_call()
    execute()
    store.stale = True
    replay = execute()
    assert replay.call['error_code'] == 'lease_stale'
    assert [sent['method'] for sent in transport.sent] == ['POST', 'GET']


@pytest.mark.parametrize('headers,body_id', [
    ({'X-Generation-Id': 'invalid/id'}, None),
    ({'X-Generation-Id': 'gen-one'}, 'gen-two'),
    ({'X-Generation-Id': ''}, None),
])
def test_malformed_or_conflicting_generation_keeps_unknown_error_without_unsafe_get(headers, body_id):
    payload = {'error': {'code': 404, 'message': 'routing failed'}}
    if body_id is not None:
        payload['id'] = body_id
    transport = FakeTransport([(404, payload, headers)])
    instance, store, _ = make_broker(store=HistoryStore(), transport=transport)
    first = instance.execute(CONTEXT, operation_id='openrouter.chat', parameters=CHAT,
        action_sequence=0, timeout_ms=30000)
    assert first.call['outcome'] == 'uncertain' and first.call.get('actual_microusd') is None
    saved = store.calls[first.call['call_identity']]['uncertain_doc']
    assert 'openrouter_generation_id' not in saved
    assert saved['openrouter_generation_present'] is True
    replay = instance.execute(CONTEXT, operation_id='openrouter.chat', parameters=CHAT,
        action_sequence=0, timeout_ms=30000)
    assert (first.status, first.headers, first.body) == (replay.status, replay.headers, replay.body)
    assert first.status == 404 and json.loads(first.body) == payload
    assert [sent['method'] for sent in transport.sent] == ['POST']


@pytest.mark.parametrize('settled', [False, True])
def test_export_and_host_delivery_keep_original_error_after_billing(settled):
    import base64
    from lab_arena import contracts, training_export, training_steps, training_thinking
    from tests.lab_arena.training_export_test import fixture, add_call
    values = fixture()
    add_call(values, status='uncertain')
    reserve, _, uncertain = values['ledger_rows']
    run = values['runs'][0]
    body = b'{"error":{"code":404,"message":"provider routing failed"}}'
    terminal = br._terminal_response_document(404,
        {'content-type': 'application/json', 'content-length': str(len(body))}, body, call_succeeded=False)
    uncertain.pop('terminal_response', None)
    uncertain['entry_doc'] = {'reason': 'worker_reported', 'call': {
        'reason': 'missing_provider_cost', 'call_succeeded': False,
        'openrouter_generation_present': True, 'provider_status': 404,
        'openrouter_generation_id': 'gen-retained-export',
        'credential_fingerprint': br._credential_fingerprint(HOST_KEYS['openrouter']),
        'openrouter_terminal_response': terminal}}
    if settled:
        values['ledger_rows'].append(dict(reserve, entry_id=4, entry_kind='settlement', amount_microusd=19,
            entry_doc={'openrouter_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 3},
            terminal_response=br._terminal_response_document(502, {}, br.operations.GENERIC_UNAVAILABLE_BODY,
                call_succeeded=False, provider_cost={'basis': 'openrouter_generation_cost',
                    'units': '0.000019', 'unit_name': 'usd', 'operation': 'openrouter.chat',
                    'request_id': 'gen-retained-export'})))
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][0]
    assert call['actual_microusd'] == (19 if settled else None)
    assert call['terminal_entry_id'] == (4 if settled else 3)
    assert training_export.response_entry_id(call) == 3
    assert call['response_status'] == 404 and call['response'] == json.loads(body)
    receipt = {'call_identity': call['call_identity'], 'operation_id': call['operation_id'],
        'action_sequence': 0, 'ordinal': 0, 'status': 404, 'body_sha256': contracts.hash_bytes(body),
        'body_bytes': len(body), 'write_state': 'socket_written'}
    read = {'id': 'agent:0', 'kind': 'response_received', 'privacy': {'redacted': False},
        'depends_on': [], 'after_action_sequence': 0,
        'data': {key: receipt[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}}
    payload = {'response_receipts': [receipt], 'observations': [], 'agent_observations': [read]}
    snapshot = {'payload': payload, 'payload_sha256': contracts.document_hash(payload)}
    assert not training_export._link_worker_receipts(run, [call], snapshot,
        require_completed_phases=False, require_client_ack=False)
    example['worker_observation_artifacts'] = {run['run_id']: snapshot}
    evidence = training_steps._host_result(example, call)
    assert evidence['response'] == json.loads(body)
    example['example_hash'] = contracts.document_hash({k: v for k, v in example.items() if k != 'example_hash'})
    readable = training_thinking.readable_trace(example)
    assert 'provider routing failed' in json.dumps(readable)
    assert call['response_provenance']['selected_response_acknowledged'] is True


@pytest.mark.parametrize('field,value', [('response_entry_id', 4), ('response_status', 200)])
def test_retained_error_response_pointer_cannot_be_forged(field, value):
    from lab_arena import training_export
    call = {'provider': 'openrouter', 'terminal_entry_id': 4, 'response_entry_id': 3,
        'response_status': 404, 'response_sha256': 'sha256:' + 'a' * 64,
        'response_receipts': [{'entry_id': 3, 'entry_kind': 'uncertain',
            'terminal_source': 'entry_doc.call.openrouter_terminal_response',
            'metadata': {'call_succeeded': False, 'status': 404},
            'response_sha256': 'sha256:' + 'a' * 64,
            'response_ref': {'field': 'response', 'entry_id': 3}}],
        'lifecycle': [{'entry_id': 4, 'entry_kind': 'settlement',
            'accounting': {'openrouter_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 3}}]}
    assert training_export.response_entry_id(call) == 3
    call[field] = value
    with pytest.raises(training_export.ExportError):
        training_export.response_entry_id(call)
