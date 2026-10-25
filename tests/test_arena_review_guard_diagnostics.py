"""Guard failures must be visible without opening the paid-review gate."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import telemetry
from lab_arena.store import ArenaStoreUnavailable
import lab_arena.wiring
from scripts import run_lab_arena_service as entrypoint


def run_two_review_cycles(monkeypatch, *, fail_at=None, error=None, guard_present=False,
                          malformed=False, nested=False):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, "_recorder", configure_arena_otel(None, span_exporter=exporter))
    monkeypatch.setattr(entrypoint, "_install_arena_telemetry", lambda _: None)
    monkeypatch.setattr(entrypoint, "_record_service_lifecycle", lambda *a, **k: None)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LEADPOET_SENTRY_ENABLED", "0")
    reads, reviews, waits, callbacks = [], [], [], []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, seconds):
            waits.append(seconds)
            if len(waits) == 2:
                self.set()
            return self.stopped

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self):
            pass

    stop = Stop()
    monkeypatch.setattr(entrypoint, "threading", SimpleNamespace(Event=lambda: stop, Thread=InlineThread))
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *a: None)

    def rpc(function, params):
        assert function == "lab_arena_restart_guard_state_v1" and params == {}
        reads.append(function)
        if len(reads) == fail_at:
            raise error
        return {} if malformed else {
            "schema_version": "leadpoet.lab_arena.restart_guard_state.v1",
            "guard_present": guard_present,
        }

    def review(*, should_continue):
        reviews.append(True)
        if nested:
            callbacks.append(should_continue())

    service = SimpleNamespace(
        startup_checks=lambda: {"database_identity": {"current_user": "service_role"}},
        _store=SimpleNamespace(_transport=SimpleNamespace(rpc=rpc)),
        review_pending_submissions=review,
    )
    monkeypatch.setattr(lab_arena.wiring, "build_service_from_environment", lambda _: (service, FastAPI()))
    assert entrypoint.main(["--driver-only"]) == 0
    assert waits == [5, 5]
    return exporter.get_finished_spans(), reads, reviews, callbacks


@pytest.mark.parametrize("nested", [False, True], ids=["pre-review", "in-review"])
@pytest.mark.parametrize("diagnostics", [
    {"http_status": 500, "code": "55P03"},
    {"transport_error": "ReadTimeout"},
])
def test_guard_failure_exports_safe_diagnostics_and_worker_recovers(monkeypatch, capsys, nested, diagnostics):
    error = ArenaStoreUnavailable(
        "private-response credential=never-export", operation="rpc",
        resource="lab_arena_restart_guard_state_v1", **diagnostics,
    )
    spans, reads, reviews, callbacks = run_two_review_cycles(
        monkeypatch, fail_at=2 if nested else 1, error=error, nested=nested,
    )
    guard_spans = [s for s in spans if s.name == "arena.review_guard"]
    assert len(guard_spans) == len(reads) == (4 if nested else 2)
    failed = [s for s in guard_spans if s.attributes["arena.outcome"] == "failed"]
    assert len(failed) == 1
    span = failed[0]
    assert span.start_time <= span.end_time
    assert span.attributes["arena.error_type"] == "ArenaStoreUnavailable"
    assert span.attributes["arena.store_operation"] == "rpc"
    assert span.attributes["arena.store_resource"] == "lab_arena_restart_guard_state_v1"
    for key, value in diagnostics.items():
        assert span.attributes["arena.store_" + key] == value
    assert len(reviews) == (2 if nested else 1)
    assert callbacks == ([False, True] if nested else [])
    assert guard_spans[-1].attributes["arena.outcome"] == "ok"
    assert "never-export" not in repr([dict(s.attributes) for s in spans]) + str(capsys.readouterr())


@pytest.mark.parametrize("guard_present", [False, True])
def test_valid_guard_states_do_not_report_failures_or_add_reads(monkeypatch, guard_present):
    spans, reads, reviews, _ = run_two_review_cycles(monkeypatch, guard_present=guard_present)
    guards = [s for s in spans if s.name == "arena.review_guard"]
    assert len(guards) == len(reads) == 2
    assert all(s.attributes["arena.outcome"] == "ok" for s in guards)
    assert not any(k.startswith("arena.store_") for s in guards for k in s.attributes)
    assert len(reviews) == (0 if guard_present else 2)


def test_malformed_guard_remains_closed_without_invented_store_diagnostics(monkeypatch):
    spans, reads, reviews, _ = run_two_review_cycles(monkeypatch, malformed=True)
    guards = [s for s in spans if s.name == "arena.review_guard"]
    assert len(guards) == len(reads) == 2
    assert not reviews
    assert all(s.attributes["arena.error_type"] == "ValueError" for s in guards)
    assert not any(k.startswith("arena.store_") for s in guards for k in s.attributes)
