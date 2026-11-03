"""Saved provider error replies remain separate from later billing evidence."""

import copy
import json

import pytest

from lab_arena import broker, contracts, operations, training_export, training_steps, training_thinking
from tests.lab_arena.training_export_test import add_call, fixture


def retained_deepline(*, status=403, settled=False, private=False, provider='deepline'):
    values = fixture()
    add_call(values, status='uncertain', provider=provider,
             operation='openrouter.chat' if provider == 'openrouter' else 'scrapingdog.scrape')
    reserve, _, uncertain = values['ledger_rows']
    document = {'error': {'code': 'provider_request_rejected', 'message': 'Retained failure.'}}
    if private:
        document['authorization'] = 'Bearer synthetic-private-credential'
    body = contracts.canonical_json(document).encode()
    if status == 402:
        body = operations.GENERIC_UNAVAILABLE_BODY
        reserve['funding_source'] = 'host'
    terminal = broker._terminal_response_document(status,
        {'content-type': 'application/json'}, body, call_succeeded=False)
    uncertain.pop('terminal_response')
    uncertain['entry_doc'] = {'reason': 'worker_reported', 'call': {
        'reason': 'missing_provider_cost', 'call_succeeded': False,
        'provider_status': status, f'{provider}_terminal_response': terminal,
        **({'openrouter_generation_present': True} if provider == 'openrouter' else {}),
        **({'deepline_host_payment_refusal': True} if status == 402 else {})}}
    if settled:
        values['ledger_rows'].append(dict(reserve, entry_id=4, entry_kind='settlement',
            amount_microusd=19,
            entry_doc={f'{provider}_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 3},
            terminal_response=broker._terminal_response_document(502, {},
                operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False)))
    return values, body


@pytest.mark.parametrize('status', [400, 403, 404, 422, 402])
@pytest.mark.parametrize('settled', [False, True])
def test_saved_deepline_error_is_exported_without_replacing_accounting(status, settled):
    values, body = retained_deepline(status=status, settled=settled)
    original = copy.deepcopy(values['ledger_rows'])
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][0]
    assert call['response'] == json.loads(body)
    assert call['response_status'] == status
    assert call['response_sha256'] == contracts.document_hash(json.loads(body))
    assert call['terminal_entry_id'] == (4 if settled else 3)
    assert training_export.response_entry_id(call) == 3
    assert call['actual_microusd'] == (19 if settled else None)
    assert ('provider_cost_unresolved' in call['evidence_gaps']) is not settled
    assert 'full_response_unavailable' not in call['evidence_gaps']
    assert call['response_provenance']['delivery_to_worker'] == 'not_proved'
    assert len(call['response_receipts']) == 1
    receipt = call['response_receipts'][0]
    assert receipt['terminal_source'] == 'entry_doc.call.deepline_terminal_response'
    assert receipt['response_ref'] == {'field': 'response', 'entry_id': 3}
    assert receipt['wire_body_sha256'] == contracts.hash_bytes(body)
    assert receipt['wire_body_bytes'] == len(body)
    assert receipt['privacy_redactions'] == call['response_privacy_redactions'] == []
    assert values['ledger_rows'] == original
    trace = training_thinking.readable_trace(example)
    assert call['response_sha256'] == trace['calls'][0]['response_sha256']
    assert 'provider_request_rejected' in json.dumps(trace) or status == 402


def test_saved_deepline_error_uses_existing_privacy_filter():
    values, body = retained_deepline(private=True)
    call = training_export.assemble_example(**values)['provider_calls'][0]
    assert call['response']['authorization'] == '[REDACTED]'
    assert call['response_privacy_redactions'] == ['credential']
    assert 'response_privacy_nonexact' in call['evidence_gaps']
    assert call['response_receipts'][0]['wire_body_sha256'] == contracts.hash_bytes(body)
    assert call['response_sha256'] == contracts.document_hash(call['response'])


@pytest.mark.parametrize('settled', [False, True])
def test_saved_deepline_error_retains_worker_delivery_links(settled):
    values, body = retained_deepline(settled=settled)
    example = training_export.assemble_example(**values)
    call = example['provider_calls'][0]
    receipt = {'call_identity': call['call_identity'], 'operation_id': call['operation_id'],
        'action_sequence': 0, 'ordinal': 0, 'status': 403,
        'body_sha256': contracts.hash_bytes(body), 'body_bytes': len(body),
        'write_state': 'socket_written'}
    read = {'id': 'agent:0', 'kind': 'response_received', 'privacy': {'redacted': False},
        'depends_on': [], 'after_action_sequence': 0,
        'data': {key: receipt[key] for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')}}
    payload = {'response_receipts': [receipt], 'observations': [], 'agent_observations': [read]}
    snapshot = {'payload': payload, 'payload_sha256': contracts.document_hash(payload)}
    run = values['runs'][0]
    assert not training_export._link_worker_receipts(run, [call], snapshot,
        require_completed_phases=False, require_client_ack=False)
    example['worker_observation_artifacts'] = {run['run_id']: snapshot}
    assert training_steps._host_result(example, call)['response'] == json.loads(body)
    assert call['response_provenance']['selected_response_acknowledged'] is True


@pytest.mark.parametrize('provider', ['openrouter', 'deepline'])
def test_later_actual_response_keeps_normal_identity_and_retained_error_history(provider):
    values, original_body = retained_deepline(provider=provider)
    reserve = values['ledger_rows'][0]
    response = {'result': 'Actual later response.'}
    body = contracts.canonical_json(response).encode()
    values['ledger_rows'].append(dict(reserve, entry_id=4, entry_kind='recovery',
        amount_microusd=19, entry_doc={},
        terminal_response=broker._terminal_response_document(200, {}, body, call_succeeded=True)))
    call = training_export.assemble_example(**values)['provider_calls'][0]
    assert call['response'] == response
    assert call['response_status'] == 200
    assert call['actual_microusd'] == 19
    assert call['terminal_entry_id'] == training_export.response_entry_id(call) == 4
    assert 'response_entry_id' not in call
    assert len(call['response_receipts']) == 2
    earlier, later = call['response_receipts']
    assert earlier['entry_id'] == 3
    assert earlier['response'] == json.loads(original_body)
    assert earlier['wire_body_sha256'] == contracts.hash_bytes(original_body)
    assert earlier['terminal_source'] == f'entry_doc.call.{provider}_terminal_response'
    assert later['entry_id'] == 4
    assert later['response_ref'] == {'field': 'response', 'entry_id': 4}
    assert later['wire_body_sha256'] == contracts.hash_bytes(body)


@pytest.mark.parametrize('provider', ['openrouter', 'deepline'])
def test_saved_export_identity_rejects_conflicting_reconciliation_provider(provider):
    values, _ = retained_deepline(provider=provider, settled=True)
    call = training_export.assemble_example(**values)['provider_calls'][0]
    assert training_export.response_entry_id(call) == 3
    other = 'deepline' if provider == 'openrouter' else 'openrouter'
    head = next(row for row in call['lifecycle'] if row['entry_id'] == 4)
    head['accounting'][f'{other}_delayed_reconciliation'] = True
    with pytest.raises(training_export.ExportError, match='retained_provider_error_identity_invalid'):
        training_export.response_entry_id(call)


def test_upstream_502_without_saved_bridge_terminal_does_not_create_a_response():
    values, _ = retained_deepline()
    uncertain = values['ledger_rows'][-1]
    saved = uncertain['entry_doc']['call']
    saved.pop('deepline_terminal_response')
    saved['provider_status'] = 502
    reserve = values['ledger_rows'][0]
    saved['upstream_response_receipt'] = broker._capture_upstream_response(
        broker.ProviderResponse(502, {}, b'{"error":{"code":"upstream_failed"}}'),
        operation_id=reserve['operation_id'], request_hash=reserve['entry_doc']['request_hash'],
        call_identity=reserve['call_identity'])
    call = training_export.assemble_example(**values)['provider_calls'][0]
    assert 'response' not in call
    assert 'response_entry_id' not in call
    assert 'full_response_unavailable' in call['evidence_gaps']
    assert call['upstream_response_receipts'][0]['status'] == 502


@pytest.mark.parametrize('damage', ['success', 'bad_status', 'payment_proof',
                                  'bad_reconciliation', 'false_reconciliation',
                                  'wrong_provider_reconciliation', 'both_provider_reconciliation'])
def test_saved_deepline_error_rejects_invalid_proof(damage):
    values, _ = retained_deepline(status=402 if damage == 'payment_proof' else 403,
                                  settled=damage.endswith('reconciliation'))
    saved = values['ledger_rows'][2]['entry_doc']['call']
    if damage == 'success':
        saved['deepline_terminal_response']['call_succeeded'] = True
    elif damage == 'bad_status':
        saved['deepline_terminal_response']['status'] = 200
    elif damage == 'payment_proof':
        saved.pop('deepline_host_payment_refusal')
    elif damage == 'bad_reconciliation':
        values['ledger_rows'][-1]['entry_doc']['reconciled_uncertainty_entry_id'] = 2
    elif damage == 'false_reconciliation':
        values['ledger_rows'][-1]['entry_doc']['deepline_delayed_reconciliation'] = False
    elif damage == 'wrong_provider_reconciliation':
        accounting = values['ledger_rows'][-1]['entry_doc']
        accounting['openrouter_delayed_reconciliation'] = accounting.pop('deepline_delayed_reconciliation')
    else:
        values['ledger_rows'][-1]['entry_doc']['openrouter_delayed_reconciliation'] = True
    with pytest.raises(training_export.ExportError, match='retained_provider_error'):
        example = training_export.assemble_example(**values)
        training_export.response_entry_id(example['provider_calls'][0])
