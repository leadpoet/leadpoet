"""The Arena sidecar's telemetry must name what the gateway cannot, and must
still be unable to export anything beyond operational metadata.

At the gateway every Arena call collapses into the ``/arena/{arena_path:path}``
catch-all, so the operation is unrecoverable. These tests pin the two things
that fixes it — per-route request spans and per-stage pipeline spans — to the
same fail-closed envelope: a round id, submission id, hotkey, source path,
prompt, score, or model output must never reach a span, and a stage name or
outcome outside the frozen vocabulary must drop the span whole.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gateway.observability.otel_bootstrap import (
    ARENA_HTTP_SCOPE,
    ARENA_SERVICE_NAME,
    ARENA_TASK_SCOPE,
    ARENA_TASK_STAGES,
    TASK_SPAN_ATTRIBUTE_ALLOWLIST,
    configure_arena_otel,
)
from lab_arena import telemetry


def _app(exporter=None):
    app = FastAPI()

    @app.post("/arena/v1/runs/{run_id}/complete")
    def _complete(run_id: str):
        return {"status": "ok"}

    @app.get("/arena/v1/current")
    def _current():
        return {"status": "ok"}

    recorder = configure_arena_otel(app, span_exporter=exporter)
    return app, recorder


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GATEWAY_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("GATEWAY_OTEL_ENDPOINT", raising=False)
    assert configure_arena_otel(FastAPI()) is None


def test_request_spans_name_the_operation_without_leaking_the_run_id():
    exp = InMemorySpanExporter()
    app, recorder = _app(exp)
    assert recorder is not None
    with TestClient(app) as client:
        assert client.post("/arena/v1/runs/run-abc-123/complete?token=secret").status_code == 200

    (span,) = exp.get_finished_spans()
    assert span.name == "POST /arena/v1/runs/{run_id}/complete"
    assert span.instrumentation_scope.name == ARENA_HTTP_SCOPE
    assert span.resource.attributes["service.name"] == ARENA_SERVICE_NAME
    assert dict(span.attributes) == {
        "http.request.method": "POST",
        "http.route": "/arena/v1/runs/{run_id}/complete",
        "http.response.status_code": 200,
        "duration_ms": pytest.approx(span.attributes["duration_ms"]),
    }
    haystack = repr(dict(span.attributes)) + span.name
    assert "run-abc-123" not in haystack
    assert "secret" not in haystack


def test_arena_identity_is_never_the_gateway_identity():
    """INC-238's shape: a second emitter must not inherit the first's name."""
    exp = InMemorySpanExporter()
    _app(exp)
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record("driver_tick", "ok")
    for span in exp.get_finished_spans():
        assert span.resource.attributes["service.name"] == ARENA_SERVICE_NAME


def test_stage_span_carries_only_the_approved_attributes():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record("promote_baselines", "ok", count=3, duration_ms=12.5)

    (span,) = exp.get_finished_spans()
    assert span.name == "arena.promote_baselines"
    assert span.instrumentation_scope.name == ARENA_TASK_SCOPE
    assert set(span.attributes) == TASK_SPAN_ATTRIBUTE_ALLOWLIST
    assert span.attributes["arena.stage"] == "promote_baselines"
    assert span.attributes["arena.outcome"] == "ok"
    assert span.attributes["arena.error_type"] == "-"
    assert span.attributes["arena.count"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "exfiltrate", "outcome": "ok"},
        {"stage": "driver_tick", "outcome": "round-7-scored"},
        {"stage": "driver_tick", "outcome": "failed", "error_type": "psycopg2 ERROR: relation x"},
        {"stage": "driver_tick", "outcome": "ok", "count": -1},
    ],
)
def test_stage_spans_outside_the_frozen_vocabulary_are_dropped_whole(kwargs):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record(
        kwargs["stage"],
        kwargs["outcome"],
        error_type=kwargs.get("error_type", "-"),
        count=kwargs.get("count", 0),
    )
    assert exp.get_finished_spans() == ()


def test_recorded_stage_names_match_the_pipeline_seam():
    """A stage the driver reports must be one the validator will admit."""
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    for stage in sorted(ARENA_TASK_STAGES):
        recorder.record(stage, "ok")
    assert len(exp.get_finished_spans()) == len(ARENA_TASK_STAGES)


def test_pipeline_seam_is_a_no_op_until_a_recorder_is_installed():
    telemetry.install_recorder(None)
    with telemetry.stage("driver_tick") as observed:
        observed.count = 1
    telemetry.record("driver_tick", "ok")  # must not raise


def test_stage_context_records_the_failure_and_re_raises():
    exp = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exp))
    try:
        with pytest.raises(ValueError):
            with telemetry.stage("advance_round"):
                raise ValueError("round-9 scoring rejected by database")

        (span,) = exp.get_finished_spans()
        assert span.attributes["arena.stage"] == "advance_round"
        assert span.attributes["arena.outcome"] == "failed"
        assert span.attributes["arena.error_type"] == "ValueError"
        assert "round-9" not in repr(dict(span.attributes))
    finally:
        telemetry.install_recorder(None)


def test_a_broken_recorder_never_breaks_the_pipeline():
    class _Exploding:
        def record(self, *args, **kwargs):
            raise RuntimeError("exporter down")

    telemetry.install_recorder(_Exploding())
    try:
        with telemetry.stage("driver_tick"):
            pass
    finally:
        telemetry.install_recorder(None)
