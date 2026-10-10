"""Recovery bills make progress beside old unknowns without replay or expiry."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStore
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    ROUND_ID, _bare_service, _candidate,
)


def candidate(number, *, run_id=None, kind='score'):
    return _candidate(
        uncertain_entry_id=number, call_identity='sha256:' + ('%064x' % number),
        run_id=run_id or 'old-%04d' % number, kind=kind,
    )


class QueueStore:
    def __init__(self):
        self.items = [candidate(n) for n in range(1, 607)]
        self.urgent = ['abandoned-%02d' % n for n in range(12)]
        self.items += [candidate(1000 + n, run_id=run) for n, run in enumerate(self.urgent)]
        self.items += [candidate(2000, kind='execute')]
        self.reads = []

    def list_abandoned_billing_runs(self, round_id):
        assert round_id == ROUND_ID
        return self.urgent

    def list_deepline_cost_reconciliations(
        self, round_id, *, run_id, after_entry_id, limit, successful_execute_only=False,
    ):
        self.reads.append((run_id, after_entry_id, limit, successful_execute_only))
        rows = [item for item in self.items
                if (not run_id or item['run_id'] == run_id)
                and (not successful_execute_only or item['kind'] == 'execute')]
        page = sorted(rows, key=lambda row: (
            row['uncertain_entry_id'] <= after_entry_id, row['uncertain_entry_id'],
        ))[:limit]
        # The real SQL JSON aggregate returns entry-ID order after selecting
        # a cursor-ordered page. A wrapped batch must restore cursor order.
        return sorted(page, key=lambda row: row['uncertain_entry_id'])


class RecordingBroker:
    def __init__(self, store, *, statuses=None):
        self.store = store
        self.statuses = statuses or {}
        self.seen = []
        self.lock = threading.Lock()
        self.service = None

    def reconcile_deepline_cost(self, item):
        # Another thread can take the service lock during every billing GET.
        assert self.service._lock.acquire(blocking=False)
        self.service._lock.release()
        with self.lock:
            self.seen.append(item['uncertain_entry_id'])
        status = self.statuses.get(item['uncertain_entry_id'], 'pending')
        if isinstance(status, Exception):
            raise status
        return {'status': status}


def service_for(store, broker=None):
    broker = broker or RecordingBroker(store)
    service = _bare_service(store, broker)
    broker.service = service
    return service, broker


def test_twelve_abandoned_runs_get_checks_before_606_old_unknowns_drain():
    store = QueueStore()
    service, broker = service_for(store)
    for step in range(3):
        result = service._reconcile_active_deepline_cost(ROUND_ID)
        assert result['checked'] == 8
        assert len(set(broker.seen[step * 8:(step + 1) * 8])) == 8
    assert set(range(1000, 1012)) <= set(broker.seen)
    assert set(range(1, 10)) <= set(broker.seen)
    assert broker.seen.count(2000) == 3
    assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 9
    # Pending bills stay unknown; priority only selects, never expires or frees.
    assert len(store.items) == 619
    service._reconcile_active_deepline_cost(ROUND_ID)
    assert broker.seen.count(1000) == 2  # Urgent run cursor wraps.
    assert {10, 11, 12} <= set(broker.seen)


def test_failure_and_pending_bill_do_not_cancel_other_reads_or_leak_error(caplog):
    store = QueueStore()
    service, broker = service_for(store, RecordingBroker(store, statuses={
        1000: RuntimeError('private provider payload must not be logged'),
        1001: 'pending', 1002: 'settled', 1003: 'stale',
    }))
    result = service._reconcile_active_deepline_cost(ROUND_ID)
    assert result == {'status': 'settled', 'checked': 8, 'settled': 1}
    assert len(broker.seen) == 8
    assert 'RuntimeError' in caplog.text
    assert 'private provider payload' not in caplog.text
    assert service._active_deepline_reconciliations == set()
    assert len(store.items) == 619


def test_priority_and_general_overlap_checks_each_call_once():
    store = QueueStore()
    store.items = [candidate(1, run_id=store.urgent[0], kind='execute')]
    service, broker = service_for(store)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 1
    assert broker.seen == [1]
    assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 1
    assert service._deepline_priority_reconciliation_after[(ROUND_ID, '')] == 1


def test_overlapping_driver_cannot_select_or_dispatch_same_batch_twice():
    store = QueueStore()
    entered, release = threading.Event(), threading.Event()

    class BlockingBroker(RecordingBroker):
        def reconcile_deepline_cost(self, item):
            entered.set()
            assert release.wait(3)
            return super().reconcile_deepline_cost(item)

    service, broker = service_for(store, BlockingBroker(store))
    with ThreadPoolExecutor(max_workers=1) as runner:
        pending = runner.submit(service._reconcile_active_deepline_cost, ROUND_ID)
        assert entered.wait(3)
        reads = list(store.reads)
        assert service._reconcile_active_deepline_cost(ROUND_ID) == {'status': 'none'}
        assert store.reads == reads
        # Cursors are reserved before any provider read completes.
        assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 3
        release.set()
        assert pending.result(timeout=3)['checked'] == 8
    assert len(broker.seen) == len(set(broker.seen)) == 8
    assert service._active_deepline_reconciliations == set()


def test_four_worker_bound_finishes_eight_slow_reads_in_two_waves():
    store = QueueStore()

    class SlowBroker(RecordingBroker):
        active = maximum = 0

        def reconcile_deepline_cost(self, item):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.05)
                return super().reconcile_deepline_cost(item)
            finally:
                with self.lock:
                    self.active -= 1

    service, broker = service_for(store, SlowBroker(store))
    started = time.monotonic()
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 8
    elapsed = time.monotonic() - started
    assert broker.maximum == 4
    assert 0.09 <= elapsed < 2
    assert br.DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS == 5


@pytest.mark.parametrize('failure', ['abandoned', 'urgent', 'priority', 'general'])
def test_selection_failure_keeps_other_lanes_and_clears_inflight(failure):
    class FailingStore(QueueStore):
        def list_abandoned_billing_runs(self, round_id):
            if failure == 'abandoned':
                raise TimeoutError('metadata')
            return super().list_abandoned_billing_runs(round_id)

        def list_deepline_cost_reconciliations(self, round_id, **options):
            if ((failure == 'urgent' and options['run_id'])
                or (failure == 'priority' and options.get('successful_execute_only'))
                or (failure == 'general' and not options['run_id']
                    and not options.get('successful_execute_only'))):
                raise TimeoutError('candidate')
            return super().list_deepline_cost_reconciliations(round_id, **options)

    store = FailingStore()
    service, broker = service_for(store)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] >= 4
    assert service._active_deepline_reconciliations == set()
    if failure != 'general':
        assert {1, 2, 3} <= set(broker.seen)
    if failure not in ('abandoned', 'urgent'):
        assert {1000, 1001, 1002, 1003} <= set(broker.seen)


def test_broker_construction_failure_releases_inflight_marker():
    store = QueueStore()
    service, _ = service_for(store)
    service._broker_for = lambda _round: (_ for _ in ()).throw(RuntimeError('construction'))
    with pytest.raises(RuntimeError, match='construction'):
        service._reconcile_active_deepline_cost(ROUND_ID)
    assert service._active_deepline_reconciliations == set()


def test_abandoned_selector_reads_only_leased_run_error_metadata_in_small_batches():
    requests = []
    ids = ['leased-%02d' % n for n in range(26)]

    class Transport:
        def select(self, table, **options):
            requests.append((table, options))
            if table == 'lab_arena_runs':
                assert options['filters'] == {'round_id': ROUND_ID, 'status': 'leased'}
                assert options['columns'] == 'run_id'
                return [{'run_id': run} for run in ids]
            assert table == 'lab_arena_trajectory_events'
            assert options['filters'] == {'round_id': ROUND_ID, 'event_kind': 'runtime.error'}
            assert len(options['run_ids']) <= 25
            assert options['limit'] == 500 and options['descending'] is True
            assert 'content,' not in options['columns']
            return [
                {'run_id': options['run_ids'][0], 'status': 'abandoned',
                 'error_class': 'RuntimeHostError', 'failure_stage': 'runtime'},
                {'run_id': 'not-leased', 'status': 'abandoned',
                 'error_class': 'RuntimeHostError', 'failure_stage': 'runtime'},
                {'run_id': 'leased-01', 'status': 'abandoned',
                 'error_class': 'RuntimeHostError', 'failure_stage': 'cleanup'},
            ]

    assert ArenaStore(Transport()).list_abandoned_billing_runs(ROUND_ID) == [ids[0], ids[25]]
    assert len(requests) == 3


def test_abandoned_projection_matches_postgrest_and_parameterized_sql():
    import httpx
    from types import SimpleNamespace
    from lab_arena.store import PostgrestTransport, PsycopgTransport

    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json=(
            [{'run_id': 'leased:1'}] if len(requests) == 1 else []
        ))

    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        store = ArenaStore(PostgrestTransport(
            'https://example.test', service_key='sb_secret_test', http_client=http,
        ))
        assert store.list_abandoned_billing_runs(ROUND_ID) == []
    assert requests[1].url.params['event_kind'] == 'eq.runtime.error'
    assert requests[1].url.params['run_id'] == 'in.(leased:1)'
    assert requests[1].url.params['order'] == 'trajectory_id.desc'
    assert requests[1].url.params['limit'] == '500'
    assert requests[1].url.params['select'].split(',') == [
        'run_id', 'status:content->>status', 'error_class:content->>error_class',
        'failure_stage:content->>failure_stage',
    ]
    queries = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self):
            return [({'run_id': 'leased:1'},)] if len(queries) == 1 else []

    transport = object.__new__(PsycopgTransport)
    transport._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    transport._release = lambda _: None
    assert ArenaStore(transport).list_abandoned_billing_runs(ROUND_ID) == []
    sql, values = queries[1]
    assert "content ->> 'status' AS status" in sql
    assert "content ->> 'error_class' AS error_class" in sql
    assert "content ->> 'failure_stage' AS failure_stage" in sql
    assert 'run_id = ANY(%s)' in sql
    assert 'ORDER BY trajectory_id DESC LIMIT 500' in sql
    assert values == [ROUND_ID, 'runtime.error', ['leased:1']]


def test_settlement_telemetry_counts_all_completed_bills(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from lab_arena import telemetry

    observations = []

    @contextmanager
    def stage(name):
        observed = SimpleNamespace(count=0, idle=False)
        observations.append((name, observed))
        yield observed

    store = QueueStore()
    broker = RecordingBroker(store, statuses={n: 'settled' for n in (
        1, 2, 3, 1000, 1001, 1002, 1003, 2000,
    )})
    service, _ = service_for(store, broker)
    monkeypatch.setattr(telemetry, 'stage', stage)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['settled'] == 8
    assert observations[0][0] == 'reconcile_active_deepline_costs'
    assert observations[0][1].count == 8
    assert observations[0][1].idle is False


def test_wrapped_rpc_page_reserves_last_cursor_item_not_largest_entry_id():
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(n) for n in range(1, 11)]
    service, broker = service_for(store)
    service._deepline_reconciliation_after[(ROUND_ID, '')] = 9
    service._reconcile_active_deepline_cost(ROUND_ID)
    assert set(broker.seen) == {10, 1, 2}
    assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 2
    service._reconcile_active_deepline_cost(ROUND_ID)
    assert set(broker.seen[3:]) == {3, 4, 5}
