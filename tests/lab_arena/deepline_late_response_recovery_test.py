"""Same action recovers late keyed responses without another paid execution."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_completed_response_recovery_test import (
    NATIVE, catalog, execute, record,
)
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, FakeLedgerStore, make_broker


class ResponseStore(FakeLedgerStore):
    def __init__(self):
        super().__init__()
        self.responses = {}

    def reserve_call(self, **kwargs):
        kwargs['amount_microusd'] = 0
        return super().reserve_call(**kwargs)

    def _view(self, call):
        view = super()._view(call)
        view['deepline_response_missing'] = call.get('response_missing') is True
        if call['identity'] in self.responses:
            view['terminal_response'] = self.responses[call['identity']]
        return view

    def recover_deepline_response(self, **kwargs):
        with self.lock:
            if self.stale:
                return {'status': 'stale'}
            identity = kwargs['call_identity']
            call = self.calls[identity]
            doc = call['call_doc']
            for key, field in [('request_hash', 'request_hash'),
                ('execution_key', 'deepline_execution_key'),
                ('credential_fingerprint', 'credential_fingerprint'), ('operation', 'tool')]:
                if kwargs[key] != doc[field]:
                    return {'status': 'conflict'}
            if call['kind'] == 'settlement' and kwargs['actual_microusd'] not in (None, call['actual']):
                return {'status': 'conflict'}
            if kwargs['actual_microusd'] is not None and call['kind'] != 'settlement':
                call.update(kind='settlement', actual=kwargs['actual_microusd'], terminal=kwargs['terminal_response'])
            self.responses.setdefault(identity, kwargs['terminal_response'])
            return self._view(call)


class LateTransport:
    def __init__(self, *, state='running', complete=False, bill=False):
        self.state, self.complete, self.bill = state, complete, bill
        self.requests = []
        self.key = None
        self.paid = 0
        self.supported = True
        self.patch = {}

    def send(self, **request):
        self.requests.append(request)
        if request['method'] == 'POST':
            key = request['headers']['idempotency-key']
            if self.key is None:
                self.key = key
                self.original = (request['url'], request['body'])
                self.paid += 1
                raise br.ProviderTransportError('ReadTimeout')
            assert key == self.key
            assert (request['url'], request['body']) == self.original
            self.complete = True
            return br.ProviderResponse(200, {}, json.dumps(record()['response']).encode())
        if '/executions/by-key/' in request['url']:
            value = record()
            value['executionRecovery'].update(idempotencyKey=self.key,
                state='completed' if self.complete else self.state)
            if not self.complete:
                value.pop('response'); value.pop('responseStatus')
            elif not self.bill:
                value['response'].pop('billing')
            value.update(self.patch)
            return br.ProviderResponse(200, {'x-deepline-idempotency-supported': 'true'} if self.supported else {}, json.dumps(value).encode())
        assert '/billing/usage?request_id=' in request['url']
        return br.ProviderResponse(200, {}, json.dumps({'recent': {'request_id': NATIVE, 'entries': [{
            'id': 'exact', 'request_id': NATIVE, 'provider': 'parallel', 'operation': 'parallel_search',
            'credits': .02 if self.bill else 0, 'delta': -.02 if self.bill else 0,
            'charge_state': 'posted', 'charge_finality': 'final' if self.bill else 'pending',
            'billing_mode': 'deduct_on_settle' if self.bill else 'async_hold', 'metadata': {},
        }]}}).encode())


def setup(monkeypatch, **options):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport(**options)
    broker, store, _ = make_broker(store=ResponseStore(), transport=transport)
    context = replace(CONTEXT, deepline_catalog=catalog())
    return broker, store, transport, context


def test_timeout_running_then_late_completion_replays_saved_sanitized_result(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    first = execute(broker, context)
    assert first.status == 502 and first.call['outcome'] == 'uncertain'
    transport.complete = True; transport.bill = True
    recovered = execute(broker, context)
    assert recovered.status == 200 and recovered.call['actual_microusd'] == 2000
    assert recovered.call['call_identity'] == first.call['call_identity']
    assert json.loads(recovered.body)['result'] == record()['response']['result']
    assert transport.paid == 1 and sum(r['method'] == 'POST' for r in transport.requests) == 1
    saved = store.responses[first.call['call_identity']]
    assert br._decode_terminal(saved)[2] == recovered.body
    assert transport.key not in json.dumps(saved)
    before = len(transport.requests)
    replay = execute(broker, context)
    assert replay.body == recovered.body and len(transport.requests) == before


def test_running_key_resume_uses_original_input_and_key_once(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    first = execute(broker, context)
    assert first.status == 502
    second = execute(broker, context)
    assert second.status == 200 and second.call['actual_microusd'] == 2000
    assert transport.paid == 1 and store.log.count('dispatch') == 1
    assert sum(r['method'] == 'POST' for r in transport.requests) == 2


def test_routed_score_recovers_running_key_with_original_paid_request(monkeypatch):
    from tests.lab_arena.test_lab_arena_broker import DL_KEY

    class RoutedTransport(LateTransport):
        def send(self, **request):
            response = super().send(**request)
            if request['method'] == 'POST':
                return br.ProviderResponse(200, {}, json.dumps({
                    'job_id': NATIVE, 'status': 'completed',
                    'billing': {'credits_charged': 0.02},
                    'result': {'data': {
                        'rawHtml': '<html>Recovered page</html>',
                        'metadata': {'url': 'https://example.com/',
                                     'sourceURL': 'https://example.com/',
                                     'statusCode': 200},
                    }},
                }).encode())
            payload = json.loads(response.body)
            if '/executions/by-key/' in request['url']:
                payload['toolId'] = 'firecrawl_scrape'
            else:
                payload['recent']['entries'][0].update(
                    provider='firecrawl', operation='firecrawl_scrape'
                )
            return br.ProviderResponse(response.status, response.headers,
                                       json.dumps(payload).encode())

    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = RoutedTransport()
    broker, store, _ = make_broker(
        store=ResponseStore(), transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: 'miner_key',
    )
    context = replace(CONTEXT, kind='score', round_id='arena-2026-10-06')
    request = dict(operation_id='scrapingdog.scrape',
                   parameters={'url': 'https://example.com/'},
                   action_sequence=0, timeout_ms=60_000)

    first = broker.execute(context, **request)
    assert first.status == 502 and first.call['outcome'] == 'uncertain'
    second = broker.execute(context, **request)

    assert second.status == 200 and b'Recovered page' in second.body
    assert second.call['call_identity'] == first.call['call_identity']
    assert second.call['actual_microusd'] == 2000
    assert transport.paid == 1 and store.log.count('dispatch') == 1
    posts = [item for item in transport.requests if item['method'] == 'POST']
    assert len(posts) == 2
    assert posts[0]['headers']['idempotency-key'] == posts[1]['headers']['idempotency-key']
    assert posts[0]['headers']['authorization'] == posts[1]['headers']['authorization']
    assert (posts[0]['url'], posts[0]['body']) == (posts[1]['url'], posts[1]['body'])


@pytest.mark.parametrize('change', ['unknown', 'unsupported', 'tool', 'key', 'native', 'credential', 'owner', 'refused'])
def test_unbound_recovery_cannot_dispatch_or_deliver(monkeypatch, change):
    broker, store, transport, context = setup(monkeypatch)
    first = execute(broker, context)
    call = store.calls[first.call['call_identity']]
    if change == 'unknown': transport.state = 'outcome_unknown'
    elif change == 'unsupported': transport.supported = False
    elif change == 'tool': transport.patch['toolId'] = 'other_tool'
    elif change == 'key': transport.patch['executionRecovery'] = {'idempotencyKey': 'arena:' + 'f'*64, 'state': 'running'}
    elif change == 'native': transport.patch['requestId'] = 'invalid/id'
    elif change == 'credential': call['call_doc']['credential_fingerprint'] = 'sha256:' + 'f'*64
    elif change == 'owner': call['run_id'] = 'foreign-run'
    else: call['kind'] = 'refusal'; call['reason'] = 'money_cap'
    before = sum(r['method'] == 'POST' for r in transport.requests)
    result = execute(broker, context)
    assert result.status >= 400
    assert sum(r['method'] == 'POST' for r in transport.requests) == before
    assert not store.responses


def test_pending_billing_result_is_saved_without_claiming_free_cost(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    execute(broker, context)
    transport.complete = True
    result = execute(broker, context)
    assert result.status == 200 and result.call['outcome'] == 'uncertain'
    assert 'actual_microusd' not in result.call
    before = len(transport.requests)
    assert execute(broker, context).body == result.body
    assert len(transport.requests) == before


def test_simultaneous_completed_recovery_has_one_charge_and_saved_response(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    first = execute(broker, context)
    transport.complete = True; transport.bill = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: execute(broker, context), range(2)))
    assert all(r.status == 200 for r in results)
    assert results[0].body == results[1].body
    assert len(store.responses) == 1 and transport.paid == 1
    assert len(store.calls) == 1


@pytest.mark.parametrize('status', [402, 403, 404, 429, 500])
def test_known_settled_provider_failure_does_not_lookup_or_resume(monkeypatch, status):
    broker, store, transport, context = setup(monkeypatch)
    first = execute(broker, context)
    call = store.calls[first.call['call_identity']]
    body = json.dumps({'error': {'code': 'provider_unavailable'}}).encode()
    terminal = br._terminal_response_document(status, {'content-type': 'application/json'},
                                              body, call_succeeded=False)
    call.update(kind='settlement', actual=0, terminal=terminal)
    before = len(transport.requests)
    result = execute(broker, context)
    assert result.status == status and result.body == body
    assert len(transport.requests) == before


def test_running_resume_that_still_runs_is_not_saved_as_final_result(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    execute(broker, context)
    original = transport.send
    def running(**request):
        if request['method'] == 'POST':
            transport.requests.append(request)
            return br.ProviderResponse(200, {}, json.dumps({'job_id': NATIVE,
                'status': 'running', 'executionRecovery': {'idempotencyKey': transport.key,
                'state': 'running'}}).encode())
        return original(**request)
    transport.send = running
    result = execute(broker, context)
    assert result.status == 409 and not store.responses and transport.paid == 1


def test_resume_lookup_and_post_share_one_deadline(monkeypatch):
    broker, store, transport, context = setup(monkeypatch)
    execute(broker, context)
    times = [10.0, 10.0, 14.0, 14.0]
    monkeypatch.setattr(br.time, 'monotonic', lambda: times.pop(0) if times else 14.0)
    recovered = br._deepline_execution_response_readback(transport=transport,
        secret=DL_KEY,
        execution_key=transport.key, operation='parallel_search', operation_aliases=(),
        max_response_bytes=4096, resume_request={**transport.requests[0], 'timeout_seconds': 5.0})
    assert recovered.status == 200
    assert transport.requests[-1]['timeout_seconds'] == 1.0
