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
        self.items = [_candidate(uncertain_entry_id=n, call_identity='sha256:' + ('%064x' % n)) for n in (10, 20, 100, 200)]
        self.priority = {100, 200} if priority else set()
        self.reads = []

    def list_abandoned_billing_runs(self, round_id):
        return []

    def list_deepline_cost_reconciliations(
        self, round_id, *, run_id, after_entry_id, limit,
        successful_execute_only=False, score_only=False,
    ):
        self.reads.append((successful_execute_only, score_only, run_id, after_entry_id))
        assert round_id == ROUND_ID and 1 <= limit <= 20
        rows = [item for item in self.items
                if (not successful_execute_only or item['uncertain_entry_id'] in self.priority)
                and (not score_only or item['kind'] == 'score')
                and (not run_id or item['run_id'] == run_id)]
        return sorted(rows, key=lambda item: (
            item['uncertain_entry_id'] <= after_entry_id, item['uncertain_entry_id']
        ))[:limit]


def service_for(store):
    lock = threading.RLock()
    broker = CursorBroker(lock)
    service = _bare_service(store, broker)
    service._lock = lock
    service._deepline_priority_reconciliation_after = {}
    service._deepline_score_reconciliation_after = {}
    return service, broker


def test_successes_and_older_failures_have_independent_wrapping_cursors():
    store = PriorityStore()
    service, broker = service_for(store)
    first = service._reconcile_active_deepline_cost(ROUND_ID)
    assert first['checked'] == 4
    assert sorted(broker.seen) == [10, 20, 100, 200]
    service._reconcile_active_deepline_cost(ROUND_ID)
    assert sorted(broker.seen[4:]) == [10, 20, 100, 200]
    assert any(row[0] for row in store.reads)
    assert any(row[1] for row in store.reads)
    assert any(not row[0] and not row[1] for row in store.reads)
    restarted, broker = service_for(store)
    restarted._reconcile_active_deepline_cost(ROUND_ID)
    assert sorted(broker.seen) == [10, 20, 100, 200]


def test_empty_priority_does_not_remove_general_capacity():
    store = PriorityStore(priority=False)
    service, broker = service_for(store)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 4
    assert sorted(broker.seen) == [10, 20, 100, 200]
    assert any(row[0] for row in store.reads)
    assert any(not row[0] and not row[1] for row in store.reads)


def test_targeted_recovery_keeps_existing_semantics_and_empty_batch_is_idle():
    store = PriorityStore()
    service, broker = service_for(store)
    run_id = store.items[0]['run_id']
    service._reconcile_deepline_cost(ROUND_ID, run_id=run_id)
    service._reconcile_deepline_cost(ROUND_ID, run_id=run_id)
    assert broker.seen == [10, 20]
    assert store.reads == [(False, False, run_id, 0),
                           (False, False, run_id, 10)]
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
        def list_abandoned_billing_runs(self, _round):
            return []

        def list_deepline_cost_reconciliations(self, _round, **options):
            return [candidate] if options.get('successful_execute_only') else []

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
        'list_abandoned_billing_runs': lambda _self, _round: [],
        'list_deepline_cost_reconciliations': lambda _self, _round, **_options: [candidate]
    })(), broker)
    service._deepline_priority_reconciliation_after = {}
    assert service._reconcile_active_deepline_cost(ROUND_ID)['status'] == status
    assert store.calls == []
