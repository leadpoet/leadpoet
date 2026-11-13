"""Active exact-generation billing is bounded, fair, and lease-aware."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import threading

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import broker as br, service as svc, telemetry
from tests.lab_arena.openrouter_delayed_cost_reconciliation_unit_test import _candidate
from tests.lab_arena.test_lab_arena_broker import HOST_KEYS, make_broker


ROUND = "arena-2026-09-12"
NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def candidate(number, *, status="accepted", run_id=None):
    item = _candidate(number)
    item.update(run_status=status, run_id=run_id or "run-%04d" % number,
                lease_expires_at=(NOW + timedelta(hours=1)).isoformat())
    return item


class QueueStore:
    def __init__(self, items):
        self.items = items
        self.reads = []
        self.run_reads = []
        self.expired = []

    def list_runs(self, round_id, **options):
        assert round_id == ROUND
        assert options == {"status": "leased", "kind": "score", "columns": "run_id"}
        self.run_reads.append(options)
        return [{"run_id": run} for run in sorted({item["run_id"] for item in self.items
                if item["run_status"] == "leased" and item["kind"] == "score"})]

    def list_openrouter_cost_reconciliations(self, round_id, *, run_id, after_entry_id, limit):
        assert round_id == ROUND and 1 <= limit <= 20
        self.reads.append((run_id, after_entry_id, limit))
        rows = [item for item in self.items if not run_id or item["run_id"] == run_id]
        page = sorted(rows, key=lambda item: (
            item["uncertain_entry_id"] <= after_entry_id, item["uncertain_entry_id"],
        ))[:limit]
        # Production JSON aggregates in entry-ID order, even on cursor wrap.
        return sorted(page, key=lambda item: item["uncertain_entry_id"])

    def operator_hold_active(self):
        return False

    def expire_leases(self, round_id):
        self.expired.append(round_id)


class RecordingBroker:
    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.seen = []
        self.lock = threading.Lock()

    def reconcile_openrouter_cost(self, item):
        assert self.service._lock.acquire(blocking=False)
        self.service._lock.release()
        with self.lock:
            self.seen.append(item["uncertain_entry_id"])
        status = self.statuses.get(item["uncertain_entry_id"], "pending")
        if isinstance(status, Exception):
            raise status
        return {"status": status}


def service_for(items, broker=None):
    service = object.__new__(svc.ArenaService)
    service._store = QueueStore(items)
    service._lock = threading.RLock()
    service._clock = lambda: NOW
    service._openrouter_reconciliation_after = {}
    service._openrouter_billing_run_after = {}
    service._active_openrouter_reconciliations = set()
    broker = broker or RecordingBroker()
    broker.service = service
    service._broker_for = lambda _: broker
    service._round = lambda _: {"status": "stage1_scoring"}
    service._invalidate_hot_round = lambda: None
    service._reconcile_active_deepline_cost = lambda _: {"status": "none"}
    service._advance_round_locked = lambda _: {"status": "advanced"}
    return service, broker


def production_backlog():
    return [candidate(n, status="leased" if n == 956 else
                      "accepted" if n <= 790 else "failed") for n in range(1, 961)]


def test_driver_reaches_rank956_live_bill_on_first_tick_beside_959_unknowns():
    service, broker = service_for(production_backlog())
    assert service.advance_round(ROUND)["reason"] == "provider_billing_pending"
    assert len(broker.seen) == 20
    assert 956 in broker.seen
    assert set(range(1, 20)) <= set(broker.seen)
    assert service._openrouter_reconciliation_after[(ROUND, "")] == 19
    assert service._store.expired == [ROUND]


def test_all960_pending_bills_are_visited_within51_ticks_without_priority_jumping_cursor():
    service, broker = service_for(production_backlog())
    for tick in range(51):
        result = service._reconcile_active_openrouter_cost(ROUND)
        assert result["checked"] == 20
        assert len(set(broker.seen[tick * 20:(tick + 1) * 20])) == 20
    assert set(broker.seen) == set(range(1, 961))
    assert broker.seen.count(956) == 51
    assert len(service._store.items) == 960
    assert all(limit == 1 for run, _, limit in service._store.reads if run)
    assert all(limit == 20 for run, _, limit in service._store.reads if not run)


def test_driver_batch_uses_original_broker_exact_get_and_settlement_binding_only():
    service, _ = service_for(production_backlog())
    fingerprint = br._credential_fingerprint(HOST_KEYS["openrouter"])
    for item in service._store.items:
        item["credential_fingerprint"] = fingerprint
    reads, settlements = [], []
    class ExactReadTransport:
        def send(self, *, method, url, headers, body, timeout_seconds):
            assert method == "GET" and body == b""
            assert headers["authorization"] == "Bearer " + HOST_KEYS["openrouter"]
            assert timeout_seconds == 5.0
            generation_id = url.removeprefix(br.OPENROUTER_GENERATION_URL)
            reads.append(generation_id)
            available = generation_id == "gen-956"
            payload = ({"data": {"id": generation_id, "total_cost": "0.003494"}}
                       if available else {"error": {"code": 404}})
            return br.ProviderResponse(200 if available else 404,
                                       {"content-type": "application/json"}, json.dumps(payload).encode())
    def settle(**binding):
        settlements.append(binding)
        return {"status": "settled"}
    service._store.reconcile_openrouter_cost = settle
    broker, _, _ = make_broker(store=service._store, transport=ExactReadTransport())
    service._broker_for = lambda _: broker
    assert service.advance_round(ROUND) == {"status": "advanced"}
    assert len(reads) == len(set(reads)) == 20
    assert set(reads) == {"gen-956"} | {"gen-%d" % n for n in range(1, 20)}
    assert settlements == [{"round_id": ROUND, "run_id": "run-0956",
                            "call_identity": _candidate(956)["call_identity"],
                            "uncertain_entry_id": 956, "generation_id": "gen-956",
                            "credential_fingerprint": fingerprint,
                            "actual_microusd": 3494, "cost_units": "0.003494"}]


@pytest.mark.parametrize("priority", (1, 2, 21))
def test_overlap_does_not_duplicate_bills_or_skip_unconsumed_global_rows(priority):
    service, broker = service_for([candidate(n, status="leased" if n == priority
                                            else "accepted") for n in range(1, 42)])
    service._reconcile_active_openrouter_cost(ROUND)
    expected_cursor = 20 if priority <= 20 else 19
    assert service._openrouter_reconciliation_after[(ROUND, "")] == expected_cursor
    assert len(broker.seen) == len(set(broker.seen)) == 20
    service._reconcile_active_openrouter_cost(ROUND)
    assert expected_cursor + 1 in broker.seen[20:]
    assert len(broker.seen[20:]) == len(set(broker.seen[20:])) == 20


def test_wrapped_global_json_page_reserves_last_consumed_circular_entry():
    service, broker = service_for([candidate(n) for n in range(1, 26)])
    service._openrouter_reconciliation_after[(ROUND, "")] = 20
    assert service._reconcile_active_openrouter_cost(ROUND)["checked"] == 20
    assert set(broker.seen) == set(range(21, 26)) | set(range(1, 16))
    assert service._openrouter_reconciliation_after[(ROUND, "")] == 15
    service._reconcile_active_openrouter_cost(ROUND)
    assert set(range(16, 21)) <= set(broker.seen[20:])
    assert service._openrouter_reconciliation_after[(ROUND, "")] == 10


def test_leased_run_probes_are_bounded_rotate_and_return_unused_slot_to_global():
    service, broker = service_for([candidate(n) for n in range(1, 61)])
    service._store.list_runs = lambda *_args, **_kwargs: [
        {"run_id": "leased-%02d" % n} for n in range(41)]
    for _ in range(3):
        assert service._reconcile_active_openrouter_cost(ROUND)["checked"] == 20
    probes = [run for run, _, _ in service._store.reads if run]
    assert probes[:41] == ["leased-%02d" % n for n in range(41)]
    assert len(probes) == 60
    assert set(broker.seen) == set(range(1, 61))


def test_priority_run_and_entry_cursors_rotate_independently():
    items = [candidate(n) for n in range(1, 61)]
    items += [candidate(100 + n, status="leased", run_id="urgent-%d" % (n % 3))
              for n in range(6)]
    service, broker = service_for(items)
    for tick in range(6):
        service._reconcile_active_openrouter_cost(ROUND)
        assert 100 + tick in broker.seen[tick * 20:(tick + 1) * 20]
        assert service._openrouter_reconciliation_after[(ROUND, "urgent-%d" % (tick % 3))] == 100 + tick


def test_pending_live_lease_is_not_masked_by_settlement_expired_or_invalid_lease():
    items = [candidate(n, status="leased") for n in range(1, 5)]
    items[1]["lease_expires_at"] = (NOW - timedelta(minutes=1)).isoformat()
    items[2]["lease_expires_at"] = "malformed"
    items[3]["lease_expires_at"] = (NOW + timedelta(hours=2)).isoformat()
    service, broker = service_for(items, RecordingBroker({1: "settled"}))
    result = service._reconcile_active_openrouter_cost(ROUND)
    assert result["settled"] == 1 and result["status"] == "pending"
    assert result["lease_expires_at"] == items[3]["lease_expires_at"]
    assert service.advance_round(ROUND)["reason"] == "provider_billing_pending"


@pytest.mark.parametrize("expiry", (None, 10, "malformed", "", (NOW-timedelta(seconds=1)).isoformat()))
def test_invalid_or_expired_pending_lease_keeps_normal_advance(expiry):
    item = candidate(1, status="leased")
    item["lease_expires_at"] = expiry
    service, _ = service_for([item])
    assert service.advance_round(ROUND) == {"status": "advanced"}


@pytest.mark.parametrize("status", ("accepted", "failed"))
def test_unresolved_historical_bills_do_not_block_round_progress(status):
    service, _ = service_for([candidate(n, status=status) for n in range(1, 21)])
    assert service.advance_round(ROUND) == {"status": "advanced"}


def test_expired_run_bill_remains_in_global_rotation_after_leased_priority_disappears():
    items = production_backlog()
    items[955].update(run_status="failed", lease_expires_at=(NOW-timedelta(seconds=1)).isoformat())
    service, broker = service_for(items)
    for _ in range(48):
        assert service._reconcile_active_openrouter_cost(ROUND)["checked"] == 20
    assert set(broker.seen) == set(range(1, 961))
    assert len(service._store.reads) == 48


def test_provider_exception_keeps_liability_blocks_live_lease_and_redacts_details(caplog):
    service, broker = service_for(production_backlog(), RecordingBroker({
        956: RuntimeError("private-generation-and-key"), 1: "settled"}))
    result = service._reconcile_active_openrouter_cost(ROUND)
    assert result["checked"] == 20 and result["settled"] == 1
    assert result["status"] == "unavailable" and result["run_status"] == "leased"
    assert service.advance_round(ROUND)["reason"] == "provider_billing_pending"
    assert "RuntimeError" in caplog.text and "private-generation-and-key" not in caplog.text
    assert service._active_openrouter_reconciliations == set()


@pytest.mark.parametrize("failure", ("runs", "priority", "global"))
def test_selector_failure_preserves_other_lane_progress_and_releases_guard(failure, caplog):
    service, broker = service_for(production_backlog())
    original = service._store.list_openrouter_cost_reconciliations
    error = RuntimeError("private-query")
    def fail():
        raise error
    if failure == "runs":
        service._store.list_runs = lambda *_args, **_kwargs: fail()
    else:
        def lookup(round_id, **options):
            if bool(options["run_id"]) == (failure == "priority"):
                fail()
            return original(round_id, **options)
        service._store.list_openrouter_cost_reconciliations = lookup
    if failure == "global":
        with pytest.raises(RuntimeError) as caught:
            service.advance_round(ROUND)
        assert caught.value is error
        assert broker.seen == [956]
        assert (ROUND, "") not in service._openrouter_reconciliation_after
        assert service._store.expired == []
    else:
        result = service._reconcile_active_openrouter_cost(ROUND)
        assert result["checked"] == 20
    assert (956 in broker.seen) == (failure == "global")
    assert service._active_openrouter_reconciliations == set()
    assert "private-query" not in caplog.text


def test_global_lookup_failure_without_selected_priority_keeps_cursor_and_blocks_advance():
    service, _ = service_for([candidate(1)])
    service._openrouter_reconciliation_after[(ROUND, "")] = 500
    def fail(*_args, **_kwargs):
        raise RuntimeError("global unavailable")
    service._store.list_openrouter_cost_reconciliations = fail
    with pytest.raises(RuntimeError):
        service.advance_round(ROUND)
    assert service._openrouter_reconciliation_after[(ROUND, "")] == 500
    assert service._store.expired == []
    assert service._active_openrouter_reconciliations == set()


def test_concurrent_batch_is_guarded_and_eight_reads_can_progress_without_service_lock():
    entered, release = threading.Event(), threading.Event()
    class BlockingBroker(RecordingBroker):
        active = 0
        maximum = 0
        def reconcile_openrouter_cost(self, item):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                if self.active == 8:
                    entered.set()
            assert release.wait(5)
            try:
                return super().reconcile_openrouter_cost(item)
            finally:
                with self.lock:
                    self.active -= 1
    service, broker = service_for(production_backlog(), BlockingBroker())
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(service._reconcile_active_openrouter_cost, ROUND)
        try:
            assert entered.wait(5)
            before = list(service._store.reads)
            assert service._reconcile_active_openrouter_cost(ROUND) == {"status": "in_progress"}
            assert service.advance_round(ROUND) == {
                "status": "retry", "round_status": "stage1_scoring",
                "reason": "provider_billing_pending",
            }
            assert service._store.expired == [ROUND]
            assert service._store.reads == before
            assert service._lock.acquire(blocking=False)
            service._lock.release()
        finally:
            release.set()
        assert future.result()["checked"] == 20
    assert broker.maximum == 8 and len(broker.seen) == 20
    assert service._active_openrouter_reconciliations == set()


@pytest.mark.parametrize("held,status,expiry_expected", (
    (False, "stage1", True), (False, "stage2_scoring", True),
    (True, "stage1_scoring", False), (False, "scored", False),
))
def test_concurrent_batch_wait_uses_existing_hold_and_expiry_scope(held, status, expiry_expected):
    service, _ = service_for([])
    service._active_openrouter_reconciliations.add(ROUND)
    service._store.operator_hold_active = lambda: held
    service._round = lambda _: {"status": status}
    assert service.advance_round(ROUND) == {
        "status": "retry", "round_status": status, "reason": "provider_billing_pending"}
    assert service._store.expired == ([ROUND] if expiry_expected else [])


def test_batch_telemetry_counts_checked_and_settled_without_private_billing_fields():
    exporter = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exporter))
    service, _ = service_for(production_backlog(), RecordingBroker({956: "settled", 1: "settled"}))
    try:
        assert service.advance_round(ROUND) == {"status": "advanced"}
    finally:
        telemetry.install_recorder(None)
    spans = {span.attributes["arena.stage"]: span.attributes for span in exporter.get_finished_spans()}
    assert spans["advance_billing_read"]["arena.count"] == 20
    assert spans["reconcile_active_openrouter_costs"]["arena.count"] == 2
    assert "gen-" not in repr(spans) and "sha256:" not in repr(spans)


def test_empty_batch_and_broker_factory_exception_release_guard():
    service, _ = service_for([])
    assert service._reconcile_active_openrouter_cost(ROUND) == {"status": "none"}
    assert service._active_openrouter_reconciliations == set()
    service._store.items = [candidate(1)]
    def fail(_):
        raise RuntimeError("broker unavailable")
    service._broker_for = fail
    with pytest.raises(RuntimeError):
        service._reconcile_active_openrouter_cost(ROUND)
    assert service._active_openrouter_reconciliations == set()
