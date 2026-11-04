"""Provider spans must cover the broker call, including a raised failure."""

from types import SimpleNamespace

import opentelemetry.sdk.trace as sdk_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from gateway.observability import otel_bootstrap
from lab_arena import reasoning_capture, service as service_module, telemetry
from lab_arena.broker import BrokerResult
from tests.lab_arena.provider_trajectory_audit_test import FRAME
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory
from tests.lab_arena.test_lab_arena_broker import make_broker


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

    service = _service_for_trajectory(case.store, SimpleNamespace(
        execute=execute, _price_table={"models": {FRAME["parameters"]["model"]: {}}}))
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
    assert span.attributes["arena.model"] == FRAME["parameters"]["model"]
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

    service = _service_for_trajectory(case.store, SimpleNamespace(
        execute=execute, _price_table={"models": {FRAME["parameters"]["model"]: {}}}))
    assert service.handle_provider("run-1", "lease-token", FRAME) == result.to_document()
    assert case.exporter.get_finished_spans() == ()


@pytest.mark.parametrize("source", ["catalog", "frozen_registry"])
@pytest.mark.parametrize("operation_id", ["openrouter.chat", "openrouter.responses"])
def test_requested_model_uses_host_owned_ids_for_each_api(provider_timing, source, operation_id):
    case = provider_timing
    model = "deepseek/deepseek-v4.1-flash-1017"
    parameters = ({"messages": [{"role": "user", "content": "hello"}]}
                  if operation_id == "openrouter.chat" else {"input": "hello"})
    frame = {**FRAME, "operation_id": operation_id, "parameters": {"model": model, **parameters}}
    result = BrokerResult(404, {}, b'{"error":"unavailable"}', {
        "operation_id": operation_id, "provider": "openrouter",
        "provider_status": 404, "error_code": "provider_unavailable"})

    def execute(_context, **kwargs):
        assert kwargs["parameters"] == frame["parameters"]
        case.advance(311)
        return result

    broker = SimpleNamespace(execute=execute, _price_table={"models": {model: {}}} if source == "catalog" else {})
    service = _service_for_trajectory(case.store, broker)
    if source == "frozen_registry":
        service._hot_round = lambda round_id: {"round_id": round_id, "configuration_doc": {
            "reasoning_capture_policy": reasoning_capture.POLICY,
            "reasoning_route_registry": reasoning_capture.load_registry(),
        }}
    assert service.handle_provider("run-1", "lease-token", frame) == result.to_document()
    (span,) = case.exporter.get_finished_spans()
    assert span.attributes["arena.model"] == model
    assert span.attributes["arena.provider_status"] == 404
    assert span.start_time == 1_800_000_000_100_000_000
    assert span.end_time - span.start_time == 311_000_000


@pytest.mark.parametrize("model", [
    "private/customer-sensitive-label", "ada@example.com", "no-slash", None, 7,
])
def test_untrusted_model_input_stays_out_of_refusal_spans(provider_timing, model):
    case = provider_timing
    frame = {**FRAME, "parameters": {**FRAME["parameters"], "model": model}}
    result = BrokerResult(400, {}, b'{"error":"invalid"}', {
        "operation_id": "openrouter.chat", "provider": "openrouter",
        "error_code": "model_not_allowed"})
    service = _service_for_trajectory(case.store, SimpleNamespace(
        execute=lambda *_a, **_kw: result,
        _price_table={"models": {"openai/gpt-4o-mini": {}}}))
    assert service.handle_provider("run-1", "lease-token", frame) == result.to_document()
    (span,) = case.exporter.get_finished_spans()
    assert span.attributes["arena.model"] == telemetry.NO_MODEL
    if isinstance(model, str):
        assert model not in repr(dict(span.attributes))


def test_deepline_call_has_no_model_attribution(provider_timing):
    case = provider_timing
    frame = {**FRAME, "operation_id": "exa.search", "parameters": {"query": "public company"}}
    result = BrokerResult(200, {}, b'{"results":[]}', {
        "operation_id": "exa.search", "provider": "deepline"})
    service = _service_for_trajectory(case.store, SimpleNamespace(execute=lambda *_a, **_kw: result))
    assert service.handle_provider("run-1", "lease-token", frame) == result.to_document()
    (span,) = case.exporter.get_finished_spans()
    assert span.attributes["arena.model"] == telemetry.NO_MODEL


@pytest.mark.parametrize("parameters", [None, [], 7, "private malformed parameters"])
@pytest.mark.parametrize("operation_id", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("telemetry_enabled", [False, True])
@pytest.mark.parametrize("raised_failure", [False, True])
def test_malformed_parameters_preserve_real_broker_outcomes_without_sends(
    provider_timing, parameters, operation_id, telemetry_enabled, raised_failure,
):
    case = provider_timing
    if not telemetry_enabled:
        telemetry.install_recorder(None)
    failure = RuntimeError("private setup failure")

    def fail_setup(_context):
        raise failure

    broker, ledger, transport = make_broker(
        retry_miner_credential_for=fail_setup if raised_failure else None)
    service = _service_for_trajectory(case.store, broker)
    frame = {**FRAME, "operation_id": operation_id, "parameters": parameters}
    if raised_failure:
        with pytest.raises(RuntimeError) as caught:
            service.handle_provider("run-1", "lease-token", frame)
        assert caught.value is failure
    else:
        _, context = service._run_context("run-1", "lease-token")
        expected = broker.execute(context, operation_id=operation_id, parameters=parameters,
                                  action_sequence=frame["action_sequence"],
                                  timeout_ms=frame["timeout_ms"])
        assert expected.status == 400
        assert expected.call["error_code"] == "invalid_request"
        assert service.handle_provider("run-1", "lease-token", frame) == expected.to_document()
    assert transport.sent == []
    assert ledger.calls == {}
    spans = case.exporter.get_finished_spans()
    if telemetry_enabled:
        (span,) = spans
        assert span.attributes["arena.model"] == telemetry.NO_MODEL
        assert "private" not in repr(dict(span.attributes))
    else:
        assert spans == ()
