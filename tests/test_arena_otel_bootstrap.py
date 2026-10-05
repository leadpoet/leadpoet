"""The Arena sidecar's telemetry must name what the gateway cannot, and must
still be unable to export anything beyond operational metadata.

At the gateway every Arena call collapses into the ``/arena/{arena_path:path}``
catch-all, so the operation is unrecoverable. These tests pin per-route,
per-stage, provider, and terminal spans to exact fail-closed envelopes. Only
provider and committed terminal spans may carry bounded run-row identifiers;
source paths, prompts, scores, and model outputs must never reach a span.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gateway.observability.otel_bootstrap import (
    ARENA_GATE_SCOPE,
    ARENA_HTTP_SCOPE,
    ARENA_OPERATION_PROVIDERS,
    ARENA_PROVIDER_OUTCOMES,
    ARENA_PROVIDER_SCOPE,
    ARENA_PROVIDERS,
    ARENA_RUN_SCOPE,
    ARENA_SERVICE_NAME,
    ARENA_TASK_SCOPE,
    ARENA_TASK_STAGES,
    PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST,
    RUN_IDENTITY_ATTRIBUTE_ALLOWLIST,
    TASK_SPAN_ATTRIBUTE_ALLOWLIST,
    configure_arena_otel,
)
from lab_arena import telemetry


TRUSTED_RUN_IDENTITY = {
    "arena.run_id": "run-2026-10-04-1",
    "arena.round_id": "arena-2026-10-04",
    "arena.submission_id": "baseline-2026-10-04",
    "arena.runner_hotkey": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
    "arena.stage": 1,
    "arena.icp_position": 3,
    "arena.attempt": 1,
}


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
        "arena.denial": "-",
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


# -- provider calls -----------------------------------------------------------


def test_provider_span_carries_only_the_approved_attributes():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        "openrouter",
        "openrouter.responses",
        "ok",
        http_status=200,
        provider_status=200,
        attempts=2,
        cost_microusd=1375,
        duration_ms=842.0,
    )

    (span,) = exp.get_finished_spans()
    assert span.name == "arena.provider.openrouter"
    assert span.instrumentation_scope.name == ARENA_PROVIDER_SCOPE
    assert span.resource.attributes["service.name"] == ARENA_SERVICE_NAME
    assert set(span.attributes) == PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST
    assert span.attributes["arena.provider"] == "openrouter"
    assert span.attributes["arena.operation"] == "openrouter.responses"
    assert span.attributes["arena.outcome"] == "ok"
    assert span.attributes["arena.error_code"] == "-"
    assert span.attributes["arena.cost_microusd"] == 1375
    assert span.attributes["arena.attempts"] == 2


def test_provider_and_terminal_spans_share_only_bounded_run_identity():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        "deepline", "exa.search", "ok", run_identity=TRUSTED_RUN_IDENTITY,
    )
    recorder.record_run(
        "execute", "accepted", run_identity=TRUSTED_RUN_IDENTITY,
    )

    provider, terminal = exp.get_finished_spans()
    assert set(TRUSTED_RUN_IDENTITY) == RUN_IDENTITY_ATTRIBUTE_ALLOWLIST
    for span in (provider, terminal):
        assert {key: span.attributes[key] for key in TRUSTED_RUN_IDENTITY} == TRUSTED_RUN_IDENTITY
        assert "arena.miner_hotkey" not in span.attributes
        assert "arena.icp_identifier" not in span.attributes
    assert set(provider.attributes) == PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST | RUN_IDENTITY_ATTRIBUTE_ALLOWLIST
    assert terminal.attributes["arena.terminal_cause"] == "accepted"


@pytest.mark.parametrize(
    "patch",
    [
        {"arena.attempt": True},
        {"arena.attempt": 3},
        {"arena.icp_position": 100},
        {"arena.runner_hotkey": "forged-validator"},
        {"arena.round_id": "round with prompt"},
        {"arena.submission_id": "sub-" + "x" * 100},
        {"arena.run_id": "run-id\nsecret"},
        {"arena.lease_token": "secret"},
        {"arena.provider": "openrouter"},
    ],
)
def test_extra_or_invalid_run_identity_drops_provider_and_terminal_spans(patch):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    identity = {**TRUSTED_RUN_IDENTITY, **patch}
    recorder.record_provider("deepline", "exa.search", "ok", run_identity=identity)
    recorder.record_run("execute", "accepted", run_identity=identity)
    assert exp.get_finished_spans() == ()


def test_partial_run_identity_drops_provider_and_terminal_spans():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    identity = {"arena.run_id": TRUSTED_RUN_IDENTITY["arena.run_id"]}
    recorder.record_provider("deepline", "exa.search", "ok", run_identity=identity)
    recorder.record_run("execute", "accepted", run_identity=identity)
    assert exp.get_finished_spans() == ()


def test_retry_terminal_spans_keep_distinct_run_ids_and_attempts():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    retry = dict(TRUSTED_RUN_IDENTITY)
    retry["arena.run_id"] = "run-2026-10-04-2"
    retry["arena.attempt"] = 2
    retry["arena.runner_hotkey"] = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo8"
    recorder.record_run("execute", "model_error", run_identity=TRUSTED_RUN_IDENTITY)
    recorder.record_run("execute", "accepted", run_identity=retry)

    spans = exp.get_finished_spans()
    assert len(spans) == 2
    assert [(span.attributes["arena.run_id"], span.attributes["arena.attempt"])
            for span in spans] == [("run-2026-10-04-1", 1), ("run-2026-10-04-2", 2)]
    assert spans[0].attributes["arena.runner_hotkey"] != spans[1].attributes["arena.runner_hotkey"]


def test_every_provider_and_outcome_in_the_vocabulary_is_admitted():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    pairs = [
        (provider, outcome)
        for provider in sorted(ARENA_PROVIDERS)
        for outcome in sorted(ARENA_PROVIDER_OUTCOMES)
    ]
    for provider, outcome in pairs:
        recorder.record_provider(provider, "unknown", outcome)
    assert len(exp.get_finished_spans()) == len(pairs)


def test_exact_operation_provider_pairs_are_required():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider("deepline", "exa.search", "ok")
    assert exp.get_finished_spans()[0].attributes["arena.operation"] == "exa.search"
    exp.clear()
    recorder.record_provider("openrouter", "openrouter.private_token", "refused")
    recorder.record_provider("openrouter", "exa.search", "ok")
    assert exp.get_finished_spans() == ()
    assert ARENA_OPERATION_PROVIDERS["exa.search"] == "deepline"


@pytest.mark.parametrize(
    "kwargs",
    [
        # a provider name outside the contract
        {"provider": "exfiltrate", "operation": "unknown", "outcome": "ok"},
        # an operation whose provider half disagrees with the provider
        {"provider": "deepline", "operation": "openrouter.responses", "outcome": "ok"},
        # a model id smuggled in as an operation
        {"provider": "openrouter", "operation": "anthropic/claude-sonnet-4", "outcome": "ok"},
        # an outcome outside the four
        {"provider": "openrouter", "operation": "unknown", "outcome": "settled"},
        # an error code outside the vocabulary
        {"provider": "openrouter", "operation": "unknown", "outcome": "failed",
         "error_code": "quota for hotkey 5F3s exhausted"},
        # an exception message in place of a class name
        {"provider": "openrouter", "operation": "unknown", "outcome": "failed",
         "error_type": "HTTPError: 402 for key sk-or-v1-abc"},
        # out-of-range magnitudes
        {"provider": "openrouter", "operation": "unknown", "outcome": "ok", "http_status": 1000},
        {"provider": "openrouter", "operation": "unknown", "outcome": "ok", "provider_status": -1},
        {"provider": "openrouter", "operation": "unknown", "outcome": "ok", "attempts": 5},
        {"provider": "openrouter", "operation": "unknown", "outcome": "ok",
         "cost_microusd": 1_000_000_001},
    ],
)
def test_provider_spans_outside_the_frozen_envelope_are_dropped_whole(kwargs):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        kwargs.pop("provider"), kwargs.pop("operation"), kwargs.pop("outcome"), **kwargs
    )
    assert exp.get_finished_spans() == ()


def test_the_service_projection_only_produces_admissible_provider_spans():
    """The sidecar's own projection must agree with the exporter's envelope.

    ``_provider_telemetry_fields`` reduces a broker call summary — which also
    carries a call identity, a response hash, and a credential fingerprint —
    to the exported fields. Anything it fails to reduce would be dropped at
    the exporter, losing the observation entirely.
    """
    from lab_arena.service import _provider_telemetry_fields

    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    summaries = [
        {"operation_id": "openrouter.responses", "provider": "openrouter",
         "outcome": "settled", "actual_microusd": 1375, "provider_status": 200,
         "call_identity": "sha256:" + "a" * 64, "response_hash": "b" * 64},
        {"operation_id": "deepline.execute", "provider": "deepline",
         "error_code": "budget_refused", "outcome": "settled"},
        {"operation_id": "openrouter.responses", "provider": "openrouter",
         "error_code": "provider_unavailable", "outcome": "uncertain"},
        {"operation_id": "scrapingdog.google", "provider": "scrapingdog",
         "error_code": "call_uncertain"},
        # an operation the table does not know must degrade, not leak
        {"operation_id": "anthropic/claude-sonnet-4", "provider": "anthropic"},
        {},
    ]
    for summary in summaries:
        fields = _provider_telemetry_fields(summary, 200, 1)
        recorder.record_provider(duration_ms=5.0, **fields)

    spans = exp.get_finished_spans()
    assert len(spans) == len(summaries), "every projected call must be exportable"
    haystack = repr([dict(span.attributes) for span in spans])
    assert "sha256:" not in haystack and "claude-sonnet-4" not in haystack


# -- run outcomes -------------------------------------------------------------


def test_run_span_names_the_kind_and_the_terminal_cause():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_run("execute", "budget_exhausted")

    (span,) = exp.get_finished_spans()
    assert span.name == "arena.run.execute"
    assert span.instrumentation_scope.name == ARENA_RUN_SCOPE
    assert dict(span.attributes) == {
        "arena.run_kind": "execute",
        "arena.terminal_cause": "budget_exhausted",
        "arena.outcome": "failed",
    }


def test_every_contract_terminal_cause_is_admitted():
    """A cause the contract can produce must be one the validator will admit."""
    from lab_arena import contracts

    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    for cause in sorted(contracts.TERMINAL_CAUSES):
        recorder.record_run("execute", cause)
    assert len(exp.get_finished_spans()) == len(contracts.TERMINAL_CAUSES)


@pytest.mark.parametrize(
    "kind,cause",
    [
        ("execute", "round-7 rejected"),
        ("promote", "accepted"),
        ("execute", "accepted:run-abc-123"),
    ],
)
def test_run_spans_outside_the_frozen_envelope_are_dropped_whole(kind, cause):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_run(kind, cause)
    assert exp.get_finished_spans() == ()


# -- provider concurrency gate ------------------------------------------------


def test_gate_span_records_contention_only():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_gate("admitted_after_wait", duration_ms=2400.0)

    (span,) = exp.get_finished_spans()
    assert span.name == "arena.gate"
    assert span.instrumentation_scope.name == ARENA_GATE_SCOPE
    assert set(span.attributes) == {"arena.gate_outcome", "duration_ms"}
    assert span.attributes["arena.gate_outcome"] == "admitted_after_wait"


def test_gate_outcomes_outside_the_vocabulary_are_dropped_whole():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_gate("admitted")
    recorder.record_gate("waiting behind 5F3sKeyFingerprint")
    assert exp.get_finished_spans() == ()


# -- naming the refusal on a request span -------------------------------------


def test_a_refusal_code_names_the_4xx_without_its_message():
    """``ServiceError.code`` can interpolate an exception message.

    At the gateway a refused Arena call is an anonymous 4xx. The request span
    names it — but only the literal prefix, never the interpolated tail.
    """
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/claim")
    def _claim():
        telemetry.note_denial("run_result_invalid:round-7 hotkey 5F3s rejected")
        return JSONResponse(status_code=409, content={"status": "rejected"})

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            assert client.get("/arena/v1/claim").status_code == 409
    finally:
        telemetry.install_recorder(None)

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.denial"] == "run_result_invalid"
    assert "5F3s" not in repr(dict(span.attributes))


def test_a_denial_code_does_not_leak_into_the_next_request():
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/claim")
    def _claim():
        telemetry.note_denial("lease_token_invalid")
        return JSONResponse(status_code=403, content={"status": "rejected"})

    @app.get("/arena/v1/current")
    def _current():
        return {"status": "ok"}

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            client.get("/arena/v1/claim")
            client.get("/arena/v1/current")
    finally:
        telemetry.install_recorder(None)

    refused, served = exp.get_finished_spans()
    assert refused.attributes["arena.denial"] == "lease_token_invalid"
    assert served.attributes["arena.denial"] == "-"


def test_a_shape_valid_unknown_denial_code_is_not_exported():
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/current")
    def _current():
        telemetry.note_denial("private_token")
        return JSONResponse(status_code=403, content={"status": "rejected"})

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            assert client.get("/arena/v1/current").status_code == 403
    finally:
        telemetry.install_recorder(None)

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.denial"] == "-"


def test_the_deep_capture_seam_is_a_no_op_until_a_recorder_is_installed():
    telemetry.install_recorder(None)
    telemetry.record_provider("openrouter", "openrouter.responses", "ok")
    telemetry.record_run("execute", "accepted")
    telemetry.record_gate("timed_out")
    telemetry.note_denial("lease_token_invalid")  # must not raise


def test_a_broken_recorder_never_breaks_a_provider_call():
    class _Exploding:
        def __getattr__(self, name):
            def _boom(*args, **kwargs):
                raise RuntimeError("exporter down")

            return _boom

    telemetry.install_recorder(_Exploding())
    try:
        telemetry.record_provider("openrouter", "openrouter.responses", "ok")
        telemetry.record_run("execute", "accepted")
        telemetry.record_gate("timed_out")
        telemetry.note_denial("lease_token_invalid")
    finally:
        telemetry.install_recorder(None)
