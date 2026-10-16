"""Keyed legacy edge locators recover only their authenticated native bill."""
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    CALL_IDENTITY, FINGERPRINT, NATIVE_REQUEST_ID as EDGE, ReconciliationStore,
    SECRET, _broker, _candidate,
)
from tests.lab_arena.deepline_delayed_single_read_test import ExactTransport as NativeTransport, NATIVE, clock
from tests.lab_arena.deepline_execution_recovery_test import KEY, exact_bill


class ExactTransport(NativeTransport):
    def send(self, **request):
        if request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(EDGE, safe=''):
            self.requests.append(request)
            assert request['method'] == 'GET' and 0 < request['timeout_seconds'] <= 5
            return br.ProviderResponse(404, {}, b'{}')
        return super().send(**request)


def candidate(**patch):
    return _candidate(**dict(request_id=EDGE, execution_key=KEY,
                             billing_provider='firecrawl', **patch))


@pytest.mark.parametrize('fault', ['final', '404', 'pending', 'nonfinal', 'identity'])
def test_keyed_edge_reads_exact_recovered_bill_without_changing_candidate(clock, fault):
    store = ReconciliationStore(actual_microusd=2000, cost_units='0.02')
    transport = ExactTransport(fault)
    result = _broker(store, transport).reconcile_deepline_cost(candidate())
    assert result['status'] == ('settled' if fault == 'final' else 'pending')
    assert len(transport.requests) == 3 and '/executions/by-key/' in transport.requests[1]['url']
    assert transport.requests[2]['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE, safe='')
    assert transport.requests[0]['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(EDGE, safe='')
    assert clock.sleeps == []
    if fault == 'final':
        assert store.calls[0]['request_id'] == EDGE
        assert store.calls[0]['recovered_request_id'] == NATIVE
        assert store.calls[0]['execution_key'] == KEY
        assert store.calls[0]['credential_fingerprint'] == FINGERPRINT
    else:
        assert not store.calls


@pytest.mark.parametrize('fault', ['foreign_key', 'tool', 'malformed_native', 'unsupported', 'key404', 'bill_native', 'bill_provider', 'bill_tool'])
def test_unbound_key_or_bill_cannot_settle(clock, fault):
    class Transport(ExactTransport):
        def send(self, **request):
            response = super().send(**request)
            doc = json.loads(response.body)
            headers = response.headers
            if '/executions/by-key/' in request['url']:
                if fault == 'foreign_key':
                    doc['executionRecovery']['idempotencyKey'] = 'arena:' + 'a' * 64
                elif fault == 'tool':
                    doc['toolId'] = 'firecrawl_search'
                elif fault == 'malformed_native':
                    doc['requestId'] = 'bad/native'
                elif fault == 'unsupported':
                    headers = {}
                elif fault == 'key404':
                    return br.ProviderResponse(404, headers, b'{}')
            elif fault.startswith('bill_') and response.status == 200:
                doc['recent']['entries'][0][{'bill_native':'request_id', 'bill_provider':'provider', 'bill_tool':'operation'}[fault]] = 'other'
            return br.ProviderResponse(response.status, headers, json.dumps(doc).encode())
    store, transport = ReconciliationStore(), Transport('final')
    assert _broker(store, transport).reconcile_deepline_cost(candidate()) == {'status':'pending'}
    assert not store.calls and clock.sleeps == []


@pytest.mark.parametrize('patch', [
    {'execution_key': 'arena:' + 'a' * 64},
    {'execution_key': 'malformed'},
    {'credential_fingerprint': 'sha256:' + 'c' * 64},
])
def test_invalid_original_binding_performs_no_reads(clock, patch):
    store, transport = ReconciliationStore(), ExactTransport('final')
    value = candidate()
    value.update(patch)
    assert _broker(store, transport).reconcile_deepline_cost(value)['status'] in ('invalid','credential_mismatch')
    assert not transport.requests and not store.calls


@pytest.mark.parametrize('request_id, key', [(EDGE, None), (NATIVE, KEY), ('opaque_real_native_id', KEY)])
def test_unkeyed_edge_and_opaque_native_keep_exact_read_behavior(clock, request_id, key):
    class Transport:
        requests = []
        def send(self, **request):
            self.requests.append(request)
            assert request['method'] == 'GET'
            assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(request_id, safe='')
            return br.ProviderResponse(404, {}, b'{}')
    transport = Transport()
    result = br._deepline_exact_readback(transport=transport, secret=SECRET,
        request_id=request_id, execution_key=key, operation='firecrawl_scrape',
        reconciliation_deadline=5, poll=False)
    assert result == (request_id, None) and len(transport.requests) == 1


def test_key_and_bill_share_original_five_second_deadline(clock):
    class Transport(ExactTransport):
        def send(self, **request):
            response = super().send(**request)
            clock.now += 4 if '/executions/by-key/' in request['url'] else (1 if br.quote(NATIVE,safe='') in request['url'] else 0)
            return response
    store, transport = ReconciliationStore(actual_microusd=2000,cost_units='0.02'), Transport('final')
    assert _broker(store, transport).reconcile_deepline_cost(candidate())['status'] == 'settled'
    assert [r['timeout_seconds'] for r in transport.requests] == [5,5,1]
    assert clock.sleeps == []


def test_exhausted_key_read_cannot_start_bill(clock):
    class Transport(ExactTransport):
        def send(self, **request):
            response = super().send(**request)
            clock.now += 5 if '/executions/by-key/' in request['url'] else 0
            return response
    store, transport = ReconciliationStore(), Transport('final')
    assert _broker(store, transport).reconcile_deepline_cost(candidate()) == {'status':'pending'}
    assert len(transport.requests) == 2 and not store.calls


@pytest.mark.parametrize('credits', ['0.02', '0'])
def test_valid_legacy_edge_bill_settles_without_unavailable_key_lookup(clock, credits):
    class Transport:
        requests = []
        def send(self, **request):
            self.requests.append(request)
            assert request['method'] == 'GET'
            assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(EDGE,safe='')
            return br.ProviderResponse(200,{},json.dumps(exact_bill(request_id=EDGE,
                credits=credits,delta='-' + credits,
                charge_state='free' if credits=='0' else 'posted')).encode())
    amount = 0 if credits=='0' else 2000
    store,transport = ReconciliationStore(actual_microusd=amount,cost_units=credits),Transport()
    assert _broker(store,transport).reconcile_deepline_cost(candidate())['status']=='settled'
    assert len(transport.requests)==1 and store.calls[0]['recovered_request_id']==EDGE


@pytest.mark.parametrize('status', [200, 402, 503])
def test_edge_lookup_requires_404_before_key_fallback(clock, status):
    class Transport:
        requests = []
        def send(self, **request):
            self.requests.append(request)
            assert '/executions/by-key/' not in request['url']
            return br.ProviderResponse(status,{},b'{}')
    store,transport=ReconciliationStore(),Transport()
    assert _broker(store,transport).reconcile_deepline_cost(candidate())=={'status':'pending'}
    assert len(transport.requests)==1 and not store.calls
