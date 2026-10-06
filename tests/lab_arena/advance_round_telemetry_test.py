"""Round diagnostics preserve failures, locks, and existing recovery behavior."""

from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import telemetry
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStoreError


@pytest.fixture
def observed_service():
    exporter = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exporter))
    now = datetime(2026, 10, 6, 5, tzinfo=timezone.utc)
    row = {"status": "stage2", "configuration_doc": {"schedule": {
        "submission_cutoff": (now - timedelta(hours=5)).isoformat().replace("+00:00", "Z"),
        "stage_2_close": (now + timedelta(hours=5)).isoformat().replace("+00:00", "Z"),
    }}}
    service = object.__new__(ArenaService)
    service._clock = lambda: now
    service._lock = Lock()
    service._round = lambda _: row
    service._invalidate_hot_round = lambda: None
    service._reconcile_deepline_cost = lambda _: {"status": "none"}
    service._reconcile_openrouter_cost = lambda _: {"status": "none"}
    service._store = SimpleNamespace(
        operator_hold_active=lambda: False,
        expire_leases=lambda _: None,
    )
    service.stage_is_complete = lambda *_: False
    try:
        yield service, exporter, row
    finally:
        telemetry.install_recorder(None)


@pytest.mark.parametrize("billing_pending", [False, True])
def test_expiry_failure_names_the_step_without_swallowing_or_leaking(observed_service, billing_pending):
    service, exporter, _ = observed_service
    failure = ArenaStoreError("private-round-and-database-details")

    def fail(_):
        raise failure

    service._store.expire_leases = fail
    if billing_pending:
        service._reconcile_openrouter_cost = lambda _: {
            "status": "pending", "run_status": "leased",
            "lease_expires_at": (service.now() + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        }
    with pytest.raises(ArenaStoreError) as caught:
        service.advance_round("private-round-id")
    assert caught.value is failure
    failed = [s for s in exporter.get_finished_spans()
              if s.attributes["arena.outcome"] == "failed"]
    assert len(failed) == 1
    assert failed[0].attributes["arena.stage"] == "advance_expire_leases"
    assert failed[0].attributes["arena.error_type"] == "ArenaStoreError"
    assert "private-" not in repr([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert service._lock.acquire(blocking=False)
    service._lock.release()


def test_scoring_run_count_and_wait_are_observed_without_closing_active_work(observed_service):
    service, exporter, row = observed_service
    row["status"] = "stage2_scoring"
    row["configuration_doc"]["schedule"]["final_scoring_close"] = (
        service.now() + timedelta(hours=2)
    ).isoformat().replace("+00:00", "Z")
    service._store.list_runs = lambda *_args, **_kwargs: [
        {"status": "accepted"}, {"status": "leased"},
    ]
    assert service.advance_round("round") == {"status": "waiting", "round_status": "stage2_scoring"}
    spans = {s.attributes["arena.stage"]: s for s in exporter.get_finished_spans()}
    assert spans["advance_list_scoring_runs"].attributes["arena.count"] == 2
    assert spans["advance_lock_wait"].attributes["duration_ms"] >= 0


def test_exporter_failure_does_not_stop_real_round_transition(observed_service):
    service, _, row = observed_service
    row["status"] = "scored"
    publication = {"status": "published"}
    service.publish = lambda _: publication

    class BrokenRecorder:
        def record(self, *_args, **_kwargs):
            raise RuntimeError("export unavailable")

    telemetry.install_recorder(BrokenRecorder())
    assert service.advance_round("round") is publication
    assert service._lock.acquire(blocking=False)
    service._lock.release()
