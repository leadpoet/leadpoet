"""Provider spans must cover the broker call, including a raised failure."""

from types import SimpleNamespace

import opentelemetry.sdk.trace as sdk_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from gateway.observability import otel_bootstrap
from lab_arena import service as service_module, telemetry
from lab_arena.broker import BrokerResult
from tests.lab_arena.provider_trajectory_audit_test import FRAME
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory


@pytest.fixture
def provider_timing(monkeypatch):
    exporter = InMemorySpanExporter()
    telemetry.install_recorder(otel_bootstrap.configure_arena_otel(None, span_exporter=exporter))
    clock = {"ns": 1_800_000_000_000_000_000, "mono": 100.0}

    def advance(milliseconds):
        clock["ns"] += milliseconds * 1_000_000
        clock["mono"] += milliseconds / 1000.0

    monkeypatch.setattr(service_module, "time", SimpleNamespace(
        monotonic=lambda: clock["mono"], time_ns=lambda: clock["ns"]))
    monkeypatch.setattr(otel_bootstrap, "time", SimpleNamespace(time_ns=lambda: clock["ns"]))
    monkeypatch.setattr(sdk_trace, "time_ns", lambda: clock["ns"])

    class Store:
        events = []
        get_run = staticmethod(lambda _: _run())

        def append_trajectory_events(self, _run_id, _lease_hash, events):
            self.events.extend(events)
            if events[0]["kind"] == "provider.request":
                advance(100)  # Request-audit work precedes the broker interval.
            return {"status": "accepted", "accepted": len(events)}

    store = Store()
    yield SimpleNamespace(clock=clock, advance=advance, exporter=exporter, store=store)
    telemetry.install_recorder(None)


@pytest.mark.parametrize("outcome", ["success", "http_failure", "raised_failure"])
def test_provider_span_uses_the_start_before_broker_execution(provider_timing, outcome):
    case = provider_timing
    started = []
    failure = TimeoutError("private provider failure")
    result = BrokerResult(504 if outcome == "http_failure" else 200, {}, b'{"answer":"ok"}', {
        "operation_id": "openrouter.chat", "provider": "openrouter",
        "provider_status": 504 if outcome == "http_failure" else 200,
        "actual_microusd": 1375,
    })

    def execute(*_args, **_kwargs):
        started.append(case.clock["ns"])
        case.advance(311)
        if outcome == "raised_failure":
            raise failure
        return result

    service = _service_for_trajectory(case.store, SimpleNamespace(execute=execute))
    if outcome == "raised_failure":
        with pytest.raises(TimeoutError) as caught:
            service.handle_provider("run-1", "lease-token", FRAME)
        assert caught.value is failure
    else:
        assert service.handle_provider("run-1", "lease-token", FRAME) == result.to_document()

    (span,) = case.exporter.get_finished_spans()
    assert span.start_time == started[0] == 1_800_000_000_100_000_000
    assert span.end_time == started[0] + 311_000_000
    assert span.end_time - span.start_time == pytest.approx(span.attributes["duration_ms"] * 1_000_000)
    assert span.attributes["duration_ms"] == pytest.approx(311.0)
    assert span.attributes["arena.outcome"] == ("ok" if outcome == "success" else "failed")
    assert span.attributes["arena.cost_microusd"] == (0 if outcome == "raised_failure" else 1375)
    assert "private provider failure" not in str(span.attributes)
    assert [event["kind"] for event in case.store.events] == [
        "provider.request", "provider.error" if outcome == "raised_failure" else "provider.response"]


@pytest.mark.parametrize("replay_flag", ["cached", "idempotent"])
def test_provider_replay_keeps_span_suppression(provider_timing, replay_flag):
    case = provider_timing
    result = BrokerResult(200, {}, b'{"answer":"ok"}', {
        "operation_id": "openrouter.chat", "provider": "openrouter", replay_flag: True})

    def execute(*_args, **_kwargs):
        case.advance(311)
        return result

    service = _service_for_trajectory(case.store, SimpleNamespace(execute=execute))
    assert service.handle_provider("run-1", "lease-token", FRAME) == result.to_document()
    assert case.exporter.get_finished_spans() == ()
