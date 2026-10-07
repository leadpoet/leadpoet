"""Bounded, fair active-round billing checks with independent wrapping cursors."""
import json
import threading

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    CALL_IDENTITY, CursorBroker, ROUND_ID, ReconciliationStore, SECRET,
    _bare_service, _broker, _candidate,
)
from tests.lab_arena.deepline_execution_recovery_test import exact_bill


class PriorityStore:
    def __init__(self, *, priority=True):
        self.items = [_candidate(uncertain_entry_id=n) for n in (10, 20, 100, 200)]
        self.priority = {100, 200} if priority else set()
        self.reads = []

    def list_deepline_cost_reconciliations(
        self, round_id, *, run_id, after_entry_id, limit,
        successful_execute_only=False,
    ):
        self.reads.append((successful_execute_only, run_id, after_entry_id))
        assert round_id == ROUND_ID and limit == 1
        rows = [item for item in self.items
                if (not successful_execute_only or item['uncertain_entry_id'] in self.priority)
                and (not run_id or item['run_id'] == run_id)]
        return sorted(rows, key=lambda item: (
            item['uncertain_entry_id'] <= after_entry_id, item['uncertain_entry_id']
        ))[:1]


def service_for(store):
    lock = threading.RLock()
    broker = CursorBroker(lock)
    service = _bare_service(store, broker)
    service._lock = lock
    service._deepline_priority_reconciliation_after = {}
    service._deepline_priority_next = {}
    return service, broker


def test_successes_are_checked_first_without_starving_older_failures_and_wrap():
    store = PriorityStore()
    service, broker = service_for(store)
    for _ in range(10):
        service._reconcile_active_deepline_cost(ROUND_ID)
    assert broker.seen == [100, 10, 200, 20, 100, 100, 200, 200, 100, 10]
    assert store.reads[:6] == [
        (True, '', 0), (False, '', 0), (True, '', 100),
        (False, '', 10), (True, '', 200), (False, '', 20),
    ]
    # Restart discards both in-memory cursors but still checks priority first.
    restarted, broker = service_for(store)
    restarted._reconcile_active_deepline_cost(ROUND_ID)
    restarted._reconcile_active_deepline_cost(ROUND_ID)
    assert broker.seen == [100, 10]
    assert store.reads[-2:] == [(True, '', 0), (False, '', 0)]


def test_empty_priority_falls_back_without_consuming_a_provider_check():
    store = PriorityStore(priority=False)
    service, broker = service_for(store)
    for _ in range(3):
        service._reconcile_active_deepline_cost(ROUND_ID)
    assert broker.seen == [10, 20, 100]
    assert store.reads == [
        (True, '', 0), (False, '', 0), (False, '', 10),
        (True, '', 0), (False, '', 20),
    ]


def test_ordinary_empty_and_targeted_recovery_keep_existing_semantics():
    store = PriorityStore()
    service, broker = service_for(store)
    run_id = store.items[0]['run_id']
    service._reconcile_deepline_cost(ROUND_ID, run_id=run_id)
    service._reconcile_deepline_cost(ROUND_ID, run_id=run_id)
    assert broker.seen == [10, 20]
    assert store.reads == [(False, run_id, 0), (False, run_id, 10)]
    assert service._deepline_priority_next == {}
    assert service._deepline_priority_reconciliation_after == {}
    store.items = []
    assert service._reconcile_active_deepline_cost(ROUND_ID) == {'status': 'none'}
    assert broker.seen == [10, 20]


# Preserve the two incident amounts through the new scheduling lane.
@pytest.mark.parametrize('amount,units', [(10000, '0.1'), (0, '0')])
def test_prioritized_execute_keeps_exact_cost_credentials_and_get_only(amount, units):
    native = 'public-execute:11111111-2222-4333-8444-555555555555'
    candidate = _candidate(kind='execute', run_status='accepted',
        funding_source='miner_key', request_id=native, operation='exa_search',
        execution_key='arena:' + CALL_IDENTITY[7:], billing_provider='exa',
        operation_aliases=['exa_search', 'search'])

    class Store(ReconciliationStore):
        def list_deepline_cost_reconciliations(self, _round, **options):
            assert options['successful_execute_only'] is True
            return [] if self.calls else [candidate]

    class Transport:
        def __init__(self):
            self.requests = []

        def send(self, **request):
            self.requests.append(request)
            assert request['method'] == 'GET'
            assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(native, safe='')
            assert request['headers']['authorization'] == 'Bearer ' + SECRET
            return br.ProviderResponse(200, {}, json.dumps(exact_bill(
                request_id=native, operation='search', provider='exa',
                credits=units, delta='-' + units, status='success',
            )).encode())

    store, transport = Store(actual_microusd=amount, cost_units=units), Transport()
    broker = _broker(store, transport)
    broker._provider_funding_source_for = lambda _context, provider: 'miner_key'
    service = _bare_service(store, broker)
    service._deepline_priority_reconciliation_after = {}
    service._deepline_priority_next = {}
    assert service._reconcile_active_deepline_cost(ROUND_ID)['status'] == 'settled'
    assert len(transport.requests) == len(store.calls) == 1
    assert store.calls[0]['actual_microusd'] == amount
    assert store.calls[0]['request_id'] == store.calls[0]['recovered_request_id'] == native
    assert store.calls[0]['execution_key'] == candidate['execution_key']


@pytest.mark.parametrize('patch,status', [
    ({'execution_key': 'arena:' + '0' * 64}, 'invalid'),
    ({'credential_fingerprint': br._credential_fingerprint('other-miner-key')}, 'credential_mismatch'),
    ({'funding_source': 'host'}, 'credential_mismatch'),
])
def test_priority_does_not_bypass_original_credential_and_key_binding(patch, status):
    candidate = _candidate(kind='execute', run_status='accepted',
        funding_source='miner_key', execution_key='arena:' + CALL_IDENTITY[7:])
    candidate.update(patch)
    store = ReconciliationStore()

    class NoRequests:
        def send(self, **request):
            pytest.fail('unbound candidate must not reach provider')

    broker = _broker(store, NoRequests())
    broker._provider_funding_source_for = lambda _context, _provider: 'miner_key'
    service = _bare_service(type('Store', (), {
        'list_deepline_cost_reconciliations': lambda _self, _round, **_options: [candidate]
    })(), broker)
    service._deepline_priority_reconciliation_after = {}
    service._deepline_priority_next = {}
    assert service._reconcile_active_deepline_cost(ROUND_ID)['status'] == status
    assert store.calls == []
