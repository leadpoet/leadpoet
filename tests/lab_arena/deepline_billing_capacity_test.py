"""Recovery bills make progress beside old unknowns without replay or expiry."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel, STORE_ERROR_SPAN_FIELDS
from lab_arena import telemetry
from lab_arena import broker as br
from lab_arena.store import ArenaStore, ArenaStoreError, ArenaStoreUnavailable
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
        self, round_id, *, run_id, after_entry_id, limit,
        successful_execute_only=False, score_only=False,
    ):
        assert 1 <= limit <= 20
        self.reads.append((run_id, after_entry_id, limit,
                           successful_execute_only, score_only))
        rows = [item for item in self.items
                if (not run_id or item['run_id'] == run_id)
                and (not successful_execute_only or item['kind'] == 'execute')
                and (not score_only or item['kind'] == 'score')]
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


def test_store_routes_three_read_only_candidate_lanes_without_false_v2():
    class Transport:
        def __init__(self):
            self.calls = []

        def rpc(self, name, params):
            self.calls.append((name, params))
            return {'status': 'ok', 'items': []}

    transport = Transport()
    store = ArenaStore(transport)
    store.list_deepline_cost_reconciliations(ROUND_ID)
    store.list_deepline_cost_reconciliations(ROUND_ID, successful_execute_only=True)
    store.list_deepline_cost_reconciliations(ROUND_ID, score_only=True)
    assert [name[-2:] for name, _ in transport.calls] == ['v1', 'v2', 'v3']
    assert transport.calls[1][1]['p_successful_execute_only'] is True
    assert 'p_successful_execute_only' not in transport.calls[2][1]
    with pytest.raises(ValueError, match='mutually exclusive'):
        store.list_deepline_cost_reconciliations(
            ROUND_ID, successful_execute_only=True, score_only=True)


def test_twelve_abandoned_runs_get_checks_before_606_old_unknowns_drain():
    store = QueueStore()
    service, broker = service_for(store)
    for step in range(3):
        result = service._reconcile_active_deepline_cost(ROUND_ID)
        assert result['checked'] == 96
        assert len(set(broker.seen[step * 96:(step + 1) * 96])) == 96
    assert set(range(1000, 1012)) <= set(broker.seen)
    assert set(range(1, 25)) <= set(broker.seen)
    assert broker.seen.count(2000) == 3
    assert service._deepline_score_reconciliation_after[(ROUND_ID, '')] >= 88
    # Pending bills stay unknown; priority only selects, never expires or frees.
    assert len(store.items) == 619
    service._reconcile_active_deepline_cost(ROUND_ID)
    assert broker.seen.count(1000) == 2  # Urgent run cursor wraps.
    assert {30, 31, 32} <= set(broker.seen)


def test_592_older_failed_execute_heads_do_not_starve_new_score_bills():
    store = QueueStore()
    store.urgent = []
    store.items = ([candidate(n, kind='execute') for n in range(1, 593)]
                   + [candidate(n, kind='score') for n in range(593, 793)])
    original = store.list_deepline_cost_reconciliations

    def list_candidates(round_id, **options):
        # The old execute heads have call_succeeded=false and are correctly
        # absent from the separate successful-execute RPC.
        if options.get('successful_execute_only'):
            return []
        return original(round_id, **options)

    store.list_deepline_cost_reconciliations = list_candidates
    service, broker = service_for(store)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 96
    assert len([number for number in broker.seen if number > 592]) == 93
    assert broker.seen.count(593) == 1
    assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 3
    assert service._deepline_score_reconciliation_after[(ROUND_ID, '')] >= 680
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 96
    assert len([number for number in broker.seen[96:] if number > 592]) == 93
    assert all(read[2] <= 20 for read in store.reads)


def test_pending_score_pages_wrap_without_skipping_unconsumed_calls():
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(n) for n in range(1, 607)]
    service, broker = service_for(store)
    for _ in range(7):
        assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 96
    assert set(broker.seen) == set(range(1, 607))
    assert all(read[2] <= 20 for read in store.reads)


def test_56_new_judge_bills_each_tick_still_reduces_old_backlog():
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(n) for n in range(1, 1701)]
    service, broker = service_for(store)
    for tick in range(3):
        before = len(store.items)
        assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 96
        settled = set(broker.seen[-96:])
        store.items = [item for item in store.items
                       if item['uncertain_entry_id'] not in settled]
        store.items += [candidate(1701 + tick * 56 + n) for n in range(56)]
        assert len(store.items) == before - 40


def test_closed_deepline_selector_batches_published_scores_only():
    class ClosedStore(QueueStore):
        def __init__(self):
            super().__init__()
            self.urgent = []
            self.items = [candidate(n) for n in range(1, 151)]
            self.items += [candidate(1000, kind='execute')]
            self.selected = []

        def next_closed_provider_reconciliation(self, **scope):
            self.selected.append(scope)
            return {'status': 'ok', 'provider': 'deepline', 'round_id': ROUND_ID,
                    'run_id': 'old-0001',
                    'uncertain_entry_id': min(item['uncertain_entry_id']
                                              for item in self.items if item['kind'] == 'score')}

        def list_deepline_cost_reconciliations(self, round_id, **options):
            if options.get('successful_execute_only'):
                return []
            original = self.items
            try:
                # Published V1/V3 admit only the strict closed score helpers;
                # the execute liability remains in the ledger, not this list.
                self.items = [item for item in original if item['kind'] == 'score']
                return super().list_deepline_cost_reconciliations(round_id, **options)
            finally:
                self.items = original

    store = ClosedStore()
    service, broker = service_for(store)
    service._config = SimpleNamespace(mode='live', network_name='finney',
                                      netuid=71, pinned_round_id=None)
    service._closed_provider_reconciliation_after = 0
    assert service.reconcile_closed_provider_costs()['checked'] == 96
    assert 1000 not in broker.seen
    assert service._closed_provider_reconciliation_after == 1
    first = set(broker.seen)
    store.items = [item for item in store.items
                   if item['uncertain_entry_id'] not in first]
    assert service.reconcile_closed_provider_costs()['checked'] == 54
    assert not first.intersection(broker.seen[96:])
    assert store.selected[1]['after_entry_id'] == 1


def test_failure_and_pending_bill_do_not_cancel_other_reads_or_leak_error(caplog):
    store = QueueStore()
    service, broker = service_for(store, RecordingBroker(store, statuses={
        1000: RuntimeError('private provider payload must not be logged'),
        1001: 'pending', 1002: 'settled', 1003: 'stale',
    }))
    result = service._reconcile_active_deepline_cost(ROUND_ID)
    assert result == {'status': 'settled', 'checked': 96, 'settled': 1}
    assert len(broker.seen) == 96
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
        assert service._deepline_score_reconciliation_after[(ROUND_ID, '')] >= 88
        release.set()
        assert pending.result(timeout=3)['checked'] == 96
    assert len(broker.seen) == len(set(broker.seen)) == 96
    assert service._active_deepline_reconciliations == set()


def test_eight_worker_bound_finishes_96_slow_reads_in_twelve_waves():
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
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] == 96
    elapsed = time.monotonic() - started
    assert broker.maximum == 8
    assert 0.59 <= elapsed < 2
    assert br.DEEPLINE_DELAYED_RECONCILIATION_TIMEOUT_SECONDS == 5


@pytest.mark.parametrize('failure', ['abandoned', 'urgent', 'priority', 'score', 'general'])
def test_selection_failure_keeps_other_lanes_and_clears_inflight(failure):
    class FailingStore(QueueStore):
        def list_abandoned_billing_runs(self, round_id):
            if failure == 'abandoned':
                raise TimeoutError('metadata')
            return super().list_abandoned_billing_runs(round_id)

        def list_deepline_cost_reconciliations(self, round_id, **options):
            if ((failure == 'urgent' and options['run_id'])
                or (failure == 'priority' and options.get('successful_execute_only'))
                or (failure == 'score' and options.get('score_only'))
                or (failure == 'general' and not options['run_id']
                    and not options.get('successful_execute_only')
                    and not options.get('score_only'))):
                raise TimeoutError('candidate')
            return super().list_deepline_cost_reconciliations(round_id, **options)

    store = FailingStore()
    service, broker = service_for(store)
    assert service._reconcile_active_deepline_cost(ROUND_ID)['checked'] >= 4
    assert service._active_deepline_reconciliations == set()
    if failure not in ('general', 'score'):
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
    def stage(name, **_kwargs):
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
    settlement = next(observed for name, observed in observations if name == 'reconcile_active_deepline_costs')
    assert settlement.count == 8
    assert settlement.idle is False
    assert any(name == 'reconcile_deepline_billing_lookup' for name, _ in observations)


@pytest.mark.parametrize("operation,resource,diagnostics", [
    ("select", "lab_arena_submissions", {"http_status": 502, "code": "PGRST000"}),
    ("rpc", "lab_arena_get_submission_credential", {"code": "XX000"}),
    ("rpc", "lab_arena_reconcile_deepline_cost_v2", {"code": "22023"}),
    ("rpc", "lab_arena_reconcile_deepline_cost_v1", {"transport_error": "ReadTimeout"}),
])
def test_candidate_settlement_failure_retains_safe_diagnostics_and_other_progress(
    monkeypatch, caplog, operation, resource, diagnostics,
):
    canary = "secret-prompt-SQL-https://private.invalid/token"
    error = ArenaStoreError(canary, operation=operation, resource=resource, **diagnostics)
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(7), candidate(8)]
    before = [dict(item) for item in store.items]
    broker = RecordingBroker(store, statuses={7: error, 8: "settled"})
    service, _ = service_for(store, broker)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, "_recorder", configure_arena_otel(None, span_exporter=exporter))

    assert service._reconcile_active_deepline_cost(ROUND_ID) == {
        "status": "settled", "checked": 2, "settled": 1,
    }
    assert sorted(broker.seen) == [7, 8]
    assert store.items == before  # An unavailable check cannot erase the liability.
    spans = exporter.get_finished_spans()
    failed = [span for span in spans if span.attributes.get("arena.outcome") == "failed"]
    assert len(failed) == 1 and failed[0].name == "arena.reconcile_deepline_cost"
    metadata = {key.removeprefix("arena.store_"): value for key, value in failed[0].attributes.items()
                if key in STORE_ERROR_SPAN_FIELDS}
    assert metadata == {"operation": operation, "resource": resource, **diagnostics}
    completed = [span for span in spans if span.name == "arena.reconcile_deepline_cost"
                 and span.attributes.get("arena.outcome") == "ok"]
    assert len(completed) == 1
    outer = next(span for span in spans if span.name == "arena.reconcile_active_deepline_costs")
    assert outer.attributes["arena.count"] == 1 and outer.attributes["arena.outcome"] == "ok"
    assert canary not in "".join(span.to_json() for span in spans) + caplog.text


def test_candidate_settlement_plain_error_has_no_untrusted_diagnostics(monkeypatch, caplog):
    canary = "secret-prompt-SQL-https://private.invalid/token"
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(7)]
    error = RuntimeError(canary)
    error.store_error = {"operation": "rpc", "resource": "lab_arena_reconcile_deepline_cost_v2",
                         "code": "22023", "message": canary}
    service, _ = service_for(store, RecordingBroker(store, statuses={7: error}))
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, "_recorder", configure_arena_otel(None, span_exporter=exporter))

    assert service._reconcile_active_deepline_cost(ROUND_ID) == {
        "status": "unavailable", "checked": 1, "settled": 0,
    }
    failed = [span for span in exporter.get_finished_spans()
              if span.attributes.get("arena.outcome") == "failed"]
    assert len(failed) == 1
    assert not STORE_ERROR_SPAN_FIELDS.intersection(failed[0].attributes)
    assert canary not in "".join(span.to_json() for span in exporter.get_finished_spans()) + caplog.text


@pytest.mark.parametrize("fault", ["ReadTimeout", "55P03"])
def test_candidate_lookup_failure_reports_safe_resource_without_stopping_other_lanes(monkeypatch, caplog, fault):
    canary = "secret-prompt-SQL-https://private.invalid/token"
    resource = "lab_arena_list_deepline_cost_reconciliations_v2"

    class FailedPriorityStore(QueueStore):
        def __init__(self):
            super().__init__()
            self.urgent = []
            self.items = [candidate(7)]
            self.priority_reads = 0

        def list_deepline_cost_reconciliations(self, round_id, **options):
            if options.get("successful_execute_only"):
                self.priority_reads += 1
                diagnostics = ({"transport_error": fault} if fault == "ReadTimeout"
                               else {"code": fault, "http_status": 500})
                raise ArenaStoreUnavailable(canary, operation="rpc", resource=resource,
                                            **diagnostics)
            return super().list_deepline_cost_reconciliations(round_id, **options)

    store = FailedPriorityStore()
    service, broker = service_for(store)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, "_recorder", configure_arena_otel(None, span_exporter=exporter))
    result = service._reconcile_active_deepline_cost(ROUND_ID)

    assert result == {"status": "pending", "checked": 1, "settled": 0}
    assert broker.seen == [7] and store.priority_reads == 1
    assert service._deepline_reconciliation_after[(ROUND_ID, "")] == 7
    assert (ROUND_ID, "") not in service._deepline_priority_reconciliation_after
    failed = [span for span in exporter.get_finished_spans()
              if span.attributes.get("arena.outcome") == "failed"]
    assert len(failed) == 1 and failed[0].name == "arena.reconcile_deepline_billing_lookup"
    metadata = {key.removeprefix("arena.store_"): value for key, value in failed[0].attributes.items()
                if key in STORE_ERROR_SPAN_FIELDS}
    expected = {"operation": "rpc", "resource": resource}
    expected.update({"transport_error": fault} if fault == "ReadTimeout"
                    else {"code": fault, "http_status": 500})
    assert metadata == expected
    assert {key: failed[0].attributes[key] for key in (
        'arena.round_id', 'arena.billing_lane', 'arena.cursor_entry_id',
        'arena.run_scoped')} == {
            'arena.round_id': ROUND_ID, 'arena.billing_lane': 'execute',
            'arena.cursor_entry_id': 0, 'arena.run_scoped': False}
    assert canary not in str(failed[0]) + caplog.text


def test_abandoned_lookup_failure_reports_safe_select_and_preserves_candidate_checks(monkeypatch, caplog):
    canary = "secret-prompt-SQL-https://private.invalid/token"

    class FailedAbandonedStore(QueueStore):
        def __init__(self):
            super().__init__()
            self.items = [candidate(7)]

        def list_abandoned_billing_runs(self, round_id):
            raise ArenaStoreUnavailable(canary, operation="select",
                                        resource="lab_arena_runs", transport_error="ReadTimeout")

    store = FailedAbandonedStore()
    service, broker = service_for(store)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, "_recorder", configure_arena_otel(None, span_exporter=exporter))
    result = service._reconcile_active_deepline_cost(ROUND_ID)

    assert result == {"status": "pending", "checked": 1, "settled": 0}
    assert broker.seen == [7]
    failed = [span for span in exporter.get_finished_spans()
              if span.attributes.get("arena.outcome") == "failed"]
    assert len(failed) == 1 and failed[0].name == "arena.reconcile_deepline_billing_lookup"
    metadata = {key.removeprefix("arena.store_"): value for key, value in failed[0].attributes.items()
                if key in STORE_ERROR_SPAN_FIELDS}
    assert metadata == {"operation": "select", "resource": "lab_arena_runs",
                        "transport_error": "ReadTimeout"}
    assert canary not in str(failed[0]) + caplog.text


def test_wrapped_rpc_page_reserves_last_cursor_item_not_largest_entry_id():
    store = QueueStore()
    store.urgent = []
    store.items = [candidate(n) for n in range(1, 11)]
    service, broker = service_for(store)
    service._deepline_reconciliation_after[(ROUND_ID, '')] = 9
    lookup_context = {}
    first = service._deepline_cost_candidates(
        ROUND_ID, limit=3, lookup_context=lookup_context)
    assert lookup_context['cursor_entry_id'] == store.reads[-1][1] == 9
    assert {item['uncertain_entry_id'] for item in first} == {10, 1, 2}
    assert service._deepline_reconciliation_after[(ROUND_ID, '')] == 2
    second = service._deepline_cost_candidates(ROUND_ID, limit=3)
    assert {item['uncertain_entry_id'] for item in second} == {3, 4, 5}
