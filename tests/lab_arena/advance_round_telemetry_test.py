"""Round diagnostics preserve failures, locks, and existing recovery behavior."""

from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import telemetry
from lab_arena.driver import _advance_active
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
    service._reconcile_active_deepline_cost = lambda _: {"status": "none"}
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


def test_publish_store_error_names_safe_resource_on_nested_advance_spans(observed_service):
    service, exporter, row = observed_service
    row["status"] = "scored"
    service.active_rounds = lambda: [{"round_id": "private-round"}]
    canary = "secret-prompt-SQL-https://private.invalid/token"
    failure = ArenaStoreError(
        canary, operation="rpc", resource="lab_arena_transition_round",
        http_status=503, code="PGRST301",
    )
    calls = []

    def fail(round_id):
        calls.append(round_id)
        raise failure

    service.publish = fail
    assert _advance_active(service) == "failed advance_round private-round: ArenaStoreError"
    assert calls == ["private-round"]
    spans = {span.attributes["arena.stage"]: span
             for span in exporter.get_finished_spans()}
    assert spans["active_rounds"].attributes["arena.outcome"] == "ok"
    expected = {
        "arena.store_operation": "rpc",
        "arena.store_resource": "lab_arena_transition_round",
        "arena.store_http_status": 503,
        "arena.store_code": "PGRST301",
    }
    for name in ("advance_publish", "advance_round"):
        attributes = spans[name].attributes
        assert attributes["arena.outcome"] == "failed"
        assert attributes["arena.error_type"] == "ArenaStoreError"
        assert {key: attributes[key] for key in expected} == expected
    assert canary not in repr([dict(span.attributes) for span in spans.values()])


def test_publish_retry_is_recorded_idle_so_ok_means_the_round_published(observed_service):
    service, exporter, row = observed_service
    row["status"] = "scored"
    service.publish = lambda _: {
        "status": "retry", "round_status": "scored", "reason": "judge_incomplete",
    }
    assert service.advance_round("round")["status"] == "retry"
    spans = {s.attributes["arena.stage"]: s for s in exporter.get_finished_spans()}
    assert spans["advance_publish"].attributes["arena.outcome"] == "idle"


def test_a_round_that_actually_publishes_is_recorded_ok(observed_service):
    service, exporter, row = observed_service
    row["status"] = "scored"
    service.publish = lambda _: {
        "status": "published", "king_outcome": "crowned", "king_hotkey": "hotkey",
    }
    assert service.advance_round("round")["king_outcome"] == "crowned"
    spans = {s.attributes["arena.stage"]: s for s in exporter.get_finished_spans()}
    assert spans["advance_publish"].attributes["arena.outcome"] == "ok"
