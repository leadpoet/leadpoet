"""Accepted async starts preserve their job handles without inventing final costs."""
import json
from dataclasses import replace

import pytest

from lab_arena import broker as br, deepline_catalog
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeLedgerStore, make_broker

START = 'firecrawl_batch_scrape'
POLL = 'firecrawl_get_batch_scrape_status'
NATIVE = 'iad1::fixture-1791328443493-4050b75c4488'
JOB = '11111111-2222-4333-8444-555555555555'


def snapshot():
    start = row(START, provider='firecrawl', categories=['automation'],
        inputSchema={'type': 'object', 'properties': {'urls': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['urls']},
        asyncFlow={'startAction': START, 'pollActions': [POLL], 'finishAction': None},
        asyncOperation={'job': {'idPaths': ['id', 'data.id']}})
    poll = row(POLL, provider='firecrawl', categories=['admin'],
        inputSchema={'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']},
        billingSource='free', pricing={'unit': 'call', 'usdPerUnit': 0, 'creditsPerUnit': 0})
    return deepline_catalog.freeze_catalog({'tools': [start, poll]})


def receipt():
    # Structural shape observed through the live production provider on Oct 6.
    return {'job_id': NATIVE, 'status': 'running',
        'toolResponse': {'view': 'data', 'rawV2': {'data': {
            'success': True, 'id': JOB, 'status': 'queued', 'data': []}},
            'responseMeta': {'async_lifecycle': {'status': 'running', 'provider_job_id': JOB,
                                               'poll_actions': [POLL], 'finish_action': None}}},
        'billing': {'pricing_status': 'pending', 'estimated_credits': .02, 'estimated_cost_usd': .002}}


class AsyncTransport:
    def __init__(self, document=None, final=False, poll_status="completed"):
        self.document = document if document is not None else receipt()
        self.final = final
        self.poll_status = poll_status
        self.requests = []

    def send(self, **request):
        self.requests.append(request)
        if request['method'] == 'POST':
            wire = json.loads(request['body'])
            if wire['operation'] == START:
                assert 'idempotency-key' not in request['headers']
                document = self.document
            else:
                assert wire['operation'] == POLL and wire['payload']['id'] == JOB
                document = {'job_id': 'poll-native-1', 'status': self.poll_status,
                    'toolResponse': {'rawV2': {'data': {'success': True, 'status': 'processing' if self.poll_status == 'running' else 'completed', 'data': []}}},
                    'billing': {'credits_charged': 0, 'pricing_status': 'final'}}
            return br.ProviderResponse(200, {}, json.dumps(document).encode())
        assert request['method'] == 'GET' and request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE, safe='')
        return br.ProviderResponse(200, {}, json.dumps({'recent': {'request_id': NATIVE, 'entries': [{
            'id': 'usage-async-1', 'request_id': NATIVE, 'provider': 'firecrawl', 'operation': START,
            'credits': .02 if self.final else 0, 'delta': -.02 if self.final else 0,
            'charge_state': 'posted', 'charge_finality': 'final',
            'billing_mode': 'deduct_on_settle' if self.final else 'async_hold', 'metadata': {},
        }]}}).encode())


class ZeroHoldStore(FakeLedgerStore):
    def reserve_call(self, **kwargs):
        kwargs['amount_microusd'] = 0
        kwargs['call_doc'] = {k: v for k, v in kwargs['call_doc'].items() if k != 'reserve_remaining_budget'}
        return super().reserve_call(**kwargs)

    def list_ledger(self, *, call_identity=None, after_entry_id=0, **kwargs):
        if call_identity is not None:
            return super().list_ledger(call_identity=call_identity, **kwargs)
        rows = []
        for call in self.calls.values():
            for source in super().list_ledger(call_identity=call['identity']):
                source['entry_id'] = len(rows) + 1
                if source['entry_kind'] == 'uncertain':
                    source['entry_doc'] = {'call': call['uncertain_doc']}
                if source['entry_kind'] == 'settlement':
                    source['terminal_response'] = call['terminal']
                rows.append(source)
        return [r for r in rows if r['run_id'] == kwargs['run_id'] and r['entry_id'] > after_entry_id]


def execute(broker, context, tool=START, sequence=0, payload=None):
    return broker.execute(context, operation_id='deepline.execute',
        parameters={'tool': tool, 'payload': payload or {'urls': ['https://example.com']}},
        action_sequence=sequence, timeout_ms=1000)


@pytest.mark.parametrize('final,poll_status', [(False, 'completed'), (True, 'completed'), (False, 'running')])
def test_running_start_preserves_owned_job_pending_cost_and_no_paid_replay(monkeypatch, final, poll_status):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    context = replace(CONTEXT, deepline_catalog=snapshot())
    transport = AsyncTransport(final=final, poll_status=poll_status)
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    result = execute(broker, context)
    assert result.status == 200 and json.loads(result.body)['status'] == 'running'
    saved = store.calls[result.call['call_identity']]
    assert saved['amount'] == 0
    if final:
        assert result.call['actual_microusd'] == 2000
        assert saved['terminal']['call_succeeded'] is True
        assert saved['terminal']['deepline_async_job_ids'] == [JOB]
    else:
        assert result.call['outcome'] == 'uncertain' and 'actual_microusd' not in result.call
        assert saved['uncertain_doc']['call_succeeded'] is True
        assert saved['uncertain_doc']['deepline_async_job_ids'] == [JOB]
    before = len(transport.requests)
    execute(broker, context)
    assert len(transport.requests) == before  # Never execute the same paid start again.
    assert execute(broker, context, POLL, 1, {'id': 'foreign-job'}).status >= 400
    assert execute(broker, replace(context, run_id='different-run'), POLL, 1, {'id': JOB}).status >= 400
    assert len(transport.requests) == before
    polled = execute(broker, context, POLL, 2, {'id': JOB})
    assert polled.status == 200 and polled.call['actual_microusd'] == 0


@pytest.mark.parametrize('change', ['no_job', 'envelope_error', 'provider_error', 'failed_job', 'success_false', 'bad_native', 'conflicting_native', 'bad_status'])
def test_invalid_async_receipts_do_not_fabricate_success(monkeypatch, change):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    document = receipt()
    data = document['toolResponse']['rawV2']['data']
    if change == 'no_job': del data['id']
    elif change == 'envelope_error': document['error'] = {'code': 'FAILED'}
    elif change == 'provider_error': data['error'] = 'failed'
    elif change == 'failed_job': data['status'] = 'failed'
    elif change == 'success_false': data['success'] = False
    elif change == 'bad_native': document['job_id'] = ''
    elif change == 'conflicting_native': document['requestId'] = 'different'
    else: document['status'] = 'failed'
    transport = AsyncTransport(document)
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    result = execute(broker, replace(CONTEXT, deepline_catalog=snapshot()))
    assert result.status == 502
    saved = store.calls[result.call['call_identity']]['uncertain_doc']
    assert saved['call_succeeded'] is False and 'deepline_async_job_ids' not in saved


@pytest.mark.parametrize('location,field,value', [
    ('top', 'status', {}), ('top', 'status', []),
    ('data', 'status', {}), ('data', 'status', []), ('data', 'status', 200),
    ('data', 'error', []), ('data', 'error', {}),
    ('data', 'isError', 'true'), ('data', 'isError', 0), ('data', 'isError', None),
    ('data', 'success', 'false'), ('data', 'success', 0), ('data', 'success', None),
])
def test_malformed_async_protocol_fields_fail_closed_without_type_errors(location, field, value):
    document = receipt()
    node = document if location == 'top' else document['toolResponse']['rawV2']['data']
    node[field] = value
    assert br._deepline_async_response_accepted(br.ProviderResponse(200, {}, b''), document) is False


def test_valid_optional_async_flags_and_untrusted_response_controls():
    document = receipt()
    data = document['toolResponse']['rawV2']['data']
    data['isError'] = False
    assert br._deepline_async_response_accepted(br.ProviderResponse(200, {}, b''), document) is True
    assert br._deepline_async_response_accepted(br.ProviderResponse(502, {}, b''), document) is False
    assert br._deepline_async_response_accepted(br.ProviderResponse(200, {}, b'', 'credential_echo'), document) is False
    del data['success']
    assert br._deepline_async_response_accepted(br.ProviderResponse(200, {}, b''), document) is True


@pytest.mark.parametrize('credits', [0, .02])
def test_nonterminal_inline_bill_requires_explicit_finality(monkeypatch, credits):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    document = receipt()
    document['billing'] = {'credits_charged': credits}
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=AsyncTransport(document))
    result = execute(broker, replace(CONTEXT, deepline_catalog=snapshot()))
    assert result.status == 200 and result.call['outcome'] == 'uncertain'
    assert 'actual_microusd' not in result.call
    assert store.calls[result.call['call_identity']]['kind'] == 'uncertain'


@pytest.mark.parametrize('credits', [0, .02])
def test_explicit_final_nonterminal_inline_bill_settles_once(monkeypatch, credits):
    document = receipt()
    document['billing'] = {'credits_charged': credits, 'pricing_status': 'final'}
    transport = AsyncTransport(document)
    broker, store, _ = make_broker(store=ZeroHoldStore(), transport=transport)
    result = execute(broker, replace(CONTEXT, deepline_catalog=snapshot()))
    assert result.status == 200 and result.call['outcome'] == 'settled'
    assert result.call['actual_microusd'] == round(credits * 100000)
    assert len(transport.requests) == 1
    assert store.calls[result.call['call_identity']]['terminal']['call_succeeded'] is True
