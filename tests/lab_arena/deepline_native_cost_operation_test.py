"""Native receipt operation survives legacy aliases and saved-response recovery."""
import json
from dataclasses import replace

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE, catalog
from tests.lab_arena.deepline_late_response_recovery_test import ResponseStore
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, make_broker


CASES = [
    ('exa.search', {'query': 'synthetic example'}, 'exa_search', 'exa'),
    ('exa.contents', {'urls': ['https://example.com/']}, 'exa_contents', 'exa'),
    ('deepline.execute', {'tool': 'parallel_search', 'payload': {'objective': 'synthetic example'}}, 'parallel_search', 'parallel'),
    ('scrapingdog.scrape', {'url': 'https://example.com/'}, 'firecrawl_scrape', 'firecrawl'),
]


class NativeTransport:
    """One lost paid call, final bill, then a completed GET-only saved reply."""
    def __init__(self, tool, provider):
        self.tool, self.provider = tool, provider
        self.complete = False
        self.requests = []
        self.key = None
        self.lookup_patch = {}
        self.credits = 0.1

    def send(self, **request):
        self.requests.append(request)
        if request['method'] == 'POST':
            assert self.key is None, 'recovery must not repeat the paid execution'
            self.key = request['headers']['idempotency-key']
            raise br.ProviderTransportError('ReadTimeout')
        assert request['method'] == 'GET' and request['body'] == b''
        if '/executions/by-key/' in request['url']:
            doc = {'requestId': NATIVE, 'toolId': self.tool,
                   'executionRecovery': {'idempotencyKey': self.key,
                                         # This billing-only fixture has no resumable running proof.
                                         'state': 'completed' if self.complete else 'outcome_unknown'}}
            if self.complete:
                data = {'results': [{'url': 'https://example.com/', 'title': 'Synthetic', 'text': 'Synthetic page'}]}
                if self.tool == 'firecrawl_scrape':
                    data = {'rawHtml': '<html>Synthetic page</html>', 'metadata': {
                        'url': 'https://example.com/', 'sourceURL': 'https://example.com/', 'statusCode': 200}}
                doc.update(responseStatus=200, response={
                    'job_id': NATIVE, 'status': 'completed',
                    'billing': {'credits_charged': self.credits, 'pricing_status': 'final'},
                    'result': {'data': data}})
            doc.update(self.lookup_patch)
            return br.ProviderResponse(200, {'x-deepline-idempotency-supported': 'true'}, json.dumps(doc).encode())
        assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE, safe='')
        return br.ProviderResponse(200, {}, json.dumps({'recent': {'request_id': NATIVE, 'entries': [{
            'id': 'synthetic-charge', 'request_id': NATIVE, 'provider': self.provider, 'operation': self.tool,
            'credits': 0.1, 'delta': -0.1, 'charge_state': 'posted', 'charge_finality': 'final',
            'billing_mode': 'deduct_on_settle', 'metadata': {}}]}}).encode())


class NativeReceiptStore(ResponseStore):
    def recover_deepline_response(self, **kwargs):
        # Match the production SQL guard that the older fake omitted.
        cost = kwargs['terminal_response'].get('provider_cost') or {}
        if kwargs['actual_microusd'] is not None and (
            cost.get('operation') != kwargs['operation']
            or cost.get('request_id') != kwargs['request_id']
        ):
            return {'status': 'conflict'}
        return super().recover_deepline_response(**kwargs)


@pytest.mark.parametrize('operation,parameters,tool,provider', CASES)
def test_native_operation_recovers_lost_paid_reply_without_another_execution(monkeypatch, operation, parameters, tool, provider):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = NativeTransport(tool, provider)
    broker, store, _ = make_broker(store=NativeReceiptStore(), transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: 'miner_key')
    context = replace(CONTEXT, kind='score', round_id='arena-2026-10-16',
                      deepline_catalog=catalog() if operation == 'deepline.execute' else None)
    request = dict(operation_id=operation, parameters=parameters, action_sequence=32, timeout_ms=60_000)
    first = broker.execute(context, **request)
    assert first.status == 502 and first.call['actual_microusd'] == 10000
    call = store.calls[first.call['call_identity']]
    original_terminal = json.loads(json.dumps(call['terminal']))
    assert original_terminal['deepline_response_missing_reason'] == 'transport_failure'
    assert original_terminal['provider_cost']['operation'] == tool
    transport.complete = True
    before = len(transport.requests)
    recovered = broker.execute(context, **request)
    assert recovered.status == 200 and recovered.call['actual_microusd'] == 10000
    assert recovered.call['call_identity'] == first.call['call_identity']
    assert call['terminal'] == original_terminal
    assert store.responses[first.call['call_identity']]['provider_cost']['operation'] == tool
    assert all(r['method'] == 'GET' for r in transport.requests[before:])
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
    assert store.log.count('dispatch') == 1


@pytest.mark.parametrize('change', ['cost', 'tool', 'owner', 'key', 'credential'])
def test_native_operation_does_not_bypass_recovery_bindings(monkeypatch, change):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = NativeTransport('exa_search', 'exa')
    broker, store, _ = make_broker(store=NativeReceiptStore(), transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: 'miner_key')
    context = replace(CONTEXT, kind='score', round_id='arena-2026-10-16')
    request = dict(operation_id='exa.search', parameters={'query': 'synthetic example'},
                   action_sequence=32, timeout_ms=60_000)
    first = broker.execute(context, **request)
    assert first.status == 502 and first.call['actual_microusd'] == 10000
    transport.complete = True
    if change == 'cost':
        transport.credits = 0.2
    elif change == 'tool':
        transport.lookup_patch['toolId'] = 'other_tool'
    elif change == 'owner':
        store.calls[first.call['call_identity']]['run_id'] = 'foreign-run'
    elif change == 'key':
        transport.lookup_patch['executionRecovery'] = {
            'state': 'completed', 'idempotencyKey': 'arena:' + 'f' * 64}
    else:
        store.calls[first.call['call_identity']]['call_doc']['credential_fingerprint'] = 'sha256:' + 'f' * 64
    before = len(transport.requests)
    refused = broker.execute(context, **request)
    assert refused.status >= 400 and not store.responses
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
    assert all(r['method'] == 'GET' for r in transport.requests[before:])
