"""Retained bills retry across passes, without sleeping or skipping the tail."""

import json
from urllib.parse import parse_qs, urlparse

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_cost_priority_test import service_for
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    CALL_IDENTITY, ROUND_ID, ReconciliationStore, _broker, _candidate, _ledger_entry,
)
from tests.lab_arena.deepline_execution_recovery_test import exact_bill
from tests.lab_arena.deepline_score_billing_latency_test import Clock

NATIVE = 'public-execute:11111111-2222-4333-8444-555555555555'


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(br.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(br.time, 'sleep', clock.sleep)
    return clock


def native_candidate(**patch):
    return _candidate(request_id=NATIVE, execution_key='arena:' + CALL_IDENTITY[7:],
                      billing_provider='firecrawl', **patch)


class ExactTransport:
    def __init__(self, fault='pending'):
        self.requests = []
        self.fault = fault
        self.amount = '0.02'

    def send(self, **request):
        self.requests.append(request)
        assert request['method'] == 'GET'
        assert 0 < request['timeout_seconds'] <= 5
        if '/executions/by-key/' in request['url']:
            return br.ProviderResponse(200, {'x-deepline-idempotency-supported': 'true'},
                json.dumps({'toolId': 'firecrawl_scrape', 'requestId': NATIVE,
                            'executionRecovery': {'idempotencyKey': 'arena:' + CALL_IDENTITY[7:]}}).encode())
        assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE, safe='')
        if self.fault == 'transport':
            raise br.ProviderTransportError('ReadTimeout')
        if self.fault in ('404', '402', '503'):
            return br.ProviderResponse(int(self.fault), {}, b'{}')
        doc = exact_bill(request_id=NATIVE, credits=self.amount, delta='-' + self.amount,
                         charge_state='free' if self.amount == '0' else 'posted')
        if self.fault == 'pending':
            doc['recent']['entries'] = []
        elif self.fault == 'nonfinal':
            doc['recent']['entries'][0]['charge_finality'] = 'pending'
        elif self.fault == 'identity':
            doc['recent']['entries'][0]['request_id'] = 'another-request'
        return br.ProviderResponse(200, {}, json.dumps(doc).encode())


@pytest.mark.parametrize('kind', ['execute', 'score'])
@pytest.mark.parametrize('fault', ['pending', 'nonfinal', '404', '402', '503', 'transport', 'identity'])
def test_unknown_bill_is_one_read_and_later_exact_bill_settles(clock, kind, fault):
    store = ReconciliationStore(actual_microusd=2000, cost_units='0.02')
    transport = ExactTransport(fault)
    broker = _broker(store, transport)
    candidate = native_candidate(kind=kind)
    assert broker.reconcile_deepline_cost(candidate) == {'status': 'pending'}
    assert len(transport.requests) == 1 and not store.calls
    assert clock.sleeps == [] and clock.now == 0
    transport.fault = 'final'
    assert broker.reconcile_deepline_cost(candidate)['actual_microusd'] == 2000
    assert len(transport.requests) == 2 and len(store.calls) == 1
    assert store.calls[0]['credential_fingerprint'] == candidate['credential_fingerprint']
    assert store.calls[0]['recovered_request_id'] == NATIVE


def test_missing_native_receipt_uses_one_key_and_bill_chain(clock):
    store = ReconciliationStore(actual_microusd=2000, cost_units='0.02')
    transport = ExactTransport('final')
    candidate = _candidate(execution_key='arena:' + CALL_IDENTITY[7:], billing_provider='firecrawl')
    assert _broker(store, transport).reconcile_deepline_cost(candidate)['status'] == 'settled'
    assert len(transport.requests) == 2 and '/executions/by-key/' in transport.requests[0]['url']
    assert clock.sleeps == []


def test_only_explicit_final_zero_releases_unknown_bill(clock):
    store = ReconciliationStore(actual_microusd=0, cost_units='0')
    transport = ExactTransport('nonfinal')
    transport.amount = '0'
    broker = _broker(store, transport)
    assert broker.reconcile_deepline_cost(native_candidate()) == {'status': 'pending'}
    assert not store.calls
    transport.fault = 'final'
    assert broker.reconcile_deepline_cost(native_candidate())['actual_microusd'] == 0
    assert len(store.calls) == 1 and clock.sleeps == []


def test_original_credential_mismatch_performs_no_read(clock):
    store, transport = ReconciliationStore(), ExactTransport()
    assert _broker(store, transport).reconcile_deepline_cost(native_candidate(
        credential_fingerprint='sha256:' + 'c' * 64)) == {'status': 'credential_mismatch'}
    assert not transport.requests and not store.calls and clock.sleeps == []


def test_historical_pages_complete_once_then_retry_next_pass(clock):
    class Ledger:
        final = False
        requests = []

        def send(self, **request):
            self.requests.append(request)
            assert request['method'] == 'GET' and request['timeout_seconds'] <= 5
            cursor = parse_qs(urlparse(request['url']).query).get('cursor')
            doc = ({'entries': [], 'has_more': True, 'next_cursor': 'page-two'}
                   if cursor is None else
                   {'entries': [_ledger_entry()] if self.final else [], 'has_more': False})
            return br.ProviderResponse(200, {}, json.dumps(doc).encode())

    store, transport = ReconciliationStore(), Ledger()
    broker = _broker(store, transport)
    assert broker.reconcile_deepline_cost(_candidate()) == {'status': 'pending'}
    assert len(transport.requests) == 2 and not store.calls
    assert clock.sleeps == [] and clock.now == 0
    transport.final = True
    assert broker.reconcile_deepline_cost(_candidate())['actual_microusd'] == 3000
    assert len(transport.requests) == 4 and len(store.calls) == 1
    assert clock.sleeps == []


def test_idle_full_batch_checks_score_tail_and_wraps_without_skips():
    class Store:
        items = [_candidate(uncertain_entry_id=n, kind='execute' if n < 1000 else 'score',
                            call_identity='sha256:' + '%064x' % n)
                 for n in list(range(1, 301)) + list(range(1001, 1094))]

        def list_abandoned_billing_runs(self, _round):
            return []

        def list_deepline_cost_reconciliations(self, _round, *, run_id, after_entry_id,
                                             limit, successful_execute_only=False, score_only=False):
            if successful_execute_only:
                return []
            rows = [row for row in self.items if not score_only or row['kind'] == 'score']
            return sorted(rows, key=lambda row: (row['uncertain_entry_id'] <= after_entry_id,
                                                row['uncertain_entry_id']))[:limit]

    service, broker = service_for(Store())

    def pending(candidate):
        broker.seen.append(candidate['uncertain_entry_id'])
        return {'status': 'pending'}

    broker.reconcile_deepline_cost = pending
    for _ in range(3):
        before = len(broker.seen)
        assert service._reconcile_active_deepline_cost(ROUND_ID) == {
            'status': 'pending', 'checked': 96, 'settled': 0}
        checked = broker.seen[before:]
        assert len(set(checked)) == 96
        assert set(range(1001, 1094)) <= set(checked)
