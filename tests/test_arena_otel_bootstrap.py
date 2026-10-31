"""The Arena sidecar's telemetry must name what the gateway cannot, and must
still be unable to export anything beyond operational metadata.

At the gateway every Arena call collapses into the ``/arena/{arena_path:path}``
catch-all, so the operation is unrecoverable. These tests pin per-route,
per-stage, provider, and terminal spans to exact fail-closed envelopes. Only
provider and committed terminal spans may carry bounded run-row identifiers;
source paths, prompts, scores, and model outputs must never reach a span.
"""

import pytest
from fastapi import APIRouter, FastAPI
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
    ARENA_TOOL_NONE,
    ARENA_MODEL_NONE,
    ARENA_RUN_SCOPE,
    ARENA_REVIEW_SCOPE,
    ARENA_RUNTIME_SCOPE,
    ARENA_SERVICE_NAME,
    ARENA_TASK_SCOPE,
    ARENA_TASK_STAGES,
    ARENA_LIFECYCLE_STAGES,
    LIFECYCLE_TASK_SPAN_ATTRIBUTE_TYPES,
    PROVIDER_SPAN_ATTRIBUTE_ALLOWLIST,
    RUN_IDENTITY_ATTRIBUTE_ALLOWLIST,
    REVIEW_SPAN_ATTRIBUTE_ALLOWLIST,
    RUNTIME_SPAN_ATTRIBUTE_ALLOWLIST,
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


def test_included_arena_route_exports_full_template_and_canonical_round():
    exp = InMemorySpanExporter()
    app = FastAPI()
    child = APIRouter()

    @child.get("/rounds/{round_id}")
    def _round(round_id: str):
        return {"round_id": round_id}

    parent = APIRouter()
    parent.include_router(child, prefix="/v1")
    app.include_router(parent, prefix="/arena")
    assert configure_arena_otel(app, span_exporter=exp) is not None

    with TestClient(app) as client:
        assert client.get("/arena/v1/rounds/arena-2026-10-22?token=private").status_code == 200
        assert client.get("/arena/v1/rounds/secret-extra/path").status_code == 404

    matched, unmatched = exp.get_finished_spans()
    assert matched.name == "GET /arena/v1/rounds/{round_id}"
    assert matched.attributes["arena.round_id"] == "arena-2026-10-22"
    assert unmatched.attributes["http.route"] == "/_unmatched"
    assert "private" not in repr((matched.name, dict(matched.attributes), dict(unmatched.attributes)))


def test_public_round_spans_identify_only_canonical_matched_rounds():
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/rounds/{round_id}")
    def _round(round_id: str):
        if round_id == "arena-2026-10-17":
            return {"status": "ok"}
        telemetry.note_denial("round_missing")
        return JSONResponse(status_code=404, content={"status": "missing"})

    @app.get("/arena/v1/rounds/{round_id}/submissions")
    def _submissions(round_id: str):
        telemetry.note_denial("round_missing")
        return JSONResponse(status_code=404, content={"status": "missing"})

    @app.get("/arena/v1/runs/{run_id}/complete")
    def _complete(run_id: str):
        return {"status": "ok"}

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            assert client.get("/arena/v1/rounds/arena-2026-10-17").status_code == 200
            assert client.get("/arena/v1/rounds/arena-2026-10-19").status_code == 404
            assert client.get("/arena/v1/rounds/arena-2026-10-19/submissions").status_code == 404
            assert client.get("/arena/v1/rounds/arena-2020-02-29").status_code == 404
            assert client.get("/arena/v1/runs/arena-2026-10-19/complete").status_code == 200
            assert client.get("/arena/v1/rounds/arena-2026-02-30").status_code == 404
            assert client.get("/arena/v1/rounds/arena-2026-10-19-private").status_code == 404
            assert client.get("/arena/v1/rounds/arena-2026-10-19?token=secret").status_code == 404
            assert client.get("/arena/v1/rounds/arena-2026-10-19/missing").status_code == 404
    finally:
        telemetry.install_recorder(None)

    spans = exp.get_finished_spans()
    assert len(spans) == 9, "invalid IDs and unmatched paths must keep the base span"
    assert [span.attributes.get("arena.round_id") for span in spans] == [
        "arena-2026-10-17", "arena-2026-10-19", "arena-2026-10-19",
        "arena-2020-02-29", None, None, None, "arena-2026-10-19", None,
    ]
    assert [span.attributes["arena.denial"] for span in spans[:3]] == [
        "-", "round_missing", "round_missing",
    ]
    assert spans[-1].attributes["http.route"] == "/_unmatched"
    assert "secret" not in repr([dict(span.attributes) for span in spans])


def test_public_round_http_exporter_rejects_forged_round_attributes():
    from gateway.observability.otel_bootstrap import _validating_exporter
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import SpanKind

    route = "/arena/v1/rounds/{round_id}"
    allowed = {route, "/arena/v1/runs/{run_id}/complete"}
    exp = InMemorySpanExporter()
    resource = {"service.name": ARENA_SERVICE_NAME}
    validator = _validating_exporter(
        exp, allowed_routes=lambda: allowed, expected_resource=resource,
        scopes={ARENA_HTTP_SCOPE: "arena_http"},
    )
    provider = TracerProvider(resource=Resource(resource))
    provider.add_span_processor(SimpleSpanProcessor(validator))
    tracer = provider.get_tracer(ARENA_HTTP_SCOPE)

    def emit(route, round_id):
        span = tracer.start_span("GET " + route, kind=SpanKind.SERVER, context=Context())
        for key, value in {
            "http.request.method": "GET", "http.route": route,
            "http.response.status_code": 404, "arena.denial": "round_missing",
            "duration_ms": 1.0,
        }.items():
            span.set_attribute(key, value)
        if round_id is not None:
            span.set_attribute("arena.round_id", round_id)
        span.end()

    emit(route, "arena-2026-10-19")
    for invalid in ("arena-2026-02-30", "arena-2026-10-19-private", "secret"):
        emit(route, invalid)
    emit("/arena/v1/runs/{run_id}/complete", "arena-2026-10-19")
    emit(route, None)
    spans = exp.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].attributes["arena.round_id"] == "arena-2026-10-19"
    assert "arena.round_id" not in spans[1].attributes


def test_public_round_store_503_retains_bounded_diagnostics():
    exp = InMemorySpanExporter()
    app = FastAPI()
    recorder = configure_arena_otel(app, span_exporter=exp)

    @app.get("/arena/v1/rounds/{round_id}")
    def _round(round_id: str):
        recorder.note_denial("arena_store_unavailable", store_error={
            "operation": "select", "resource": "lab_arena_rounds",
            "transport_error": "ReadTimeout",
        })
        return JSONResponse(status_code=503, content={"status": "unavailable"})

    with TestClient(app) as client:
        assert client.get("/arena/v1/rounds/arena-2026-10-19").status_code == 503

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.round_id"] == "arena-2026-10-19"
    assert span.attributes["arena.denial"] == "arena_store_unavailable"
    assert span.attributes["arena.store_operation"] == "select"
    assert span.attributes["arena.store_resource"] == "lab_arena_rounds"
    assert span.attributes["arena.store_transport_error"] == "ReadTimeout"


def test_arena_identity_is_never_the_gateway_identity():
    """INC-238's shape: a second emitter must not inherit the first's name."""
    exp = InMemorySpanExporter()
    _app(exp)
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record("driver_tick", "ok")
    for span in exp.get_finished_spans():
        assert span.resource.attributes["service.name"] == ARENA_SERVICE_NAME


def test_review_span_uses_its_exact_fail_closed_envelope():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    values = dict(
        error_code="code_review_provider_credit", categories="-", provider_http_status=402,
        retryable=False, funding_route="miner_key",
        round_id="arena-2026-10-15", submission_id="miner-submission",
        miner_hotkey="5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        attempt=1,
    )
    recorder.record_review("error", **values)
    (span,) = exp.get_finished_spans()
    assert span.instrumentation_scope.name == ARENA_REVIEW_SCOPE
    assert set(span.attributes) == REVIEW_SPAN_ATTRIBUTE_ALLOWLIST
    assert span.attributes["arena.error_code"] == "code_review_provider_credit"
    assert span.attributes["arena.review_categories"] == "-"
    assert span.attributes["arena.provider_http_status"] == 402

    exp.clear()
    for changed in (
        {"error_code": "secret provider body"},
        {"categories": "private model finding text"},
        {"categories": "duplicate_submission;duplicate_submission"},
        {"categories": "fabricated_evidence;duplicate_submission"},
        {"funding_route": "private-key-value"},
        {"submission_id": "../../../private/source"},
        {"miner_hotkey": "invalid"},
        {"attempt": 0},
        {"provider_http_status": 900},
    ):
        recorder.record_review("error", **{**values, **changed})
    assert exp.get_finished_spans() == ()

    recorder.record_review("rejected", **{
        **values, "error_code": "-", "categories": "duplicate_submission;fabricated_evidence",
    })
    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.review_categories"] == "duplicate_submission;fabricated_evidence"


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


def test_billing_lookup_stage_exports_only_bounded_context_on_failure():
    exp = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exp))
    try:
        context = {'round_id': 'arena-2026-10-18', 'lane': 'execute',
                   'run_scoped': False}
        with pytest.raises(ValueError):
            with telemetry.stage('reconcile_deepline_billing_lookup',
                                 billing_lookup=context):
                context['cursor_entry_id'] = 5121031
                raise ValueError('private provider body')
        (span,) = exp.get_finished_spans()
        assert span.attributes['arena.round_id'] == 'arena-2026-10-18'
        assert span.attributes['arena.billing_lane'] == 'execute'
        assert span.attributes['arena.cursor_entry_id'] == 5121031
        assert span.attributes['arena.run_scoped'] is False
        assert span.attributes['arena.outcome'] == 'failed'
        assert 'private provider body' not in str(span.attributes)
    finally:
        telemetry.install_recorder(None)


def test_billing_lookup_context_rejects_unapproved_values():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record('reconcile_deepline_billing_lookup', 'idle', billing_lookup={
        'round_id': 'arena-2026-10-18/private-secret', 'lane': 'execute',
        'cursor_entry_id': 4, 'run_scoped': False})
    recorder.record('driver_tick', 'ok', billing_lookup={
        'round_id': 'arena-2026-10-18', 'lane': 'execute',
        'cursor_entry_id': 4, 'run_scoped': False})
    spans = exp.get_finished_spans()
    assert len(spans) == 2
    assert all('arena.round_id' not in span.attributes for span in spans)


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
        recorder.record(
            stage, "ok", source_sha="a" * 40 if stage in ARENA_LIFECYCLE_STAGES else None
        )
    assert len(exp.get_finished_spans()) == len(ARENA_TASK_STAGES)


def test_lifecycle_spans_export_only_validated_source_version():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record("service_startup", "ok", source_sha="a" * 40)
    recorder.record("service_shutdown", "failed", source_sha="a" * 40)
    recorder.record("service_preflight", "failed", source_sha="a" * 40)
    spans = exp.get_finished_spans()
    assert [span.name for span in spans] == [
        "arena.service_startup", "arena.service_shutdown", "arena.service_preflight"
    ]
    for span in spans:
        assert set(span.attributes) == set(LIFECYCLE_TASK_SPAN_ATTRIBUTE_TYPES)
        assert span.attributes["arena.source_sha"] == "a" * 40
        assert span.resource.attributes == {"service.name": ARENA_SERVICE_NAME}


def test_lifecycle_version_cannot_escape_on_other_stages_or_malformed_sha():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record("driver_tick", "ok", source_sha="a" * 40)
    recorder.record("service_startup", "ok", source_sha="secret")
    recorder.record("service_startup", "ok")
    assert exp.get_finished_spans() == ()


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
    assert span.attributes["arena.tool"] == ARENA_TOOL_NONE


def test_provider_span_names_the_deepline_tool():
    """``deepline.execute`` is one operation over many upstream tools.

    Without the tool, a Deepline failure can only be attributed to "Deepline",
    which is never the answer an operator needs.
    """
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        "deepline",
        "deepline.execute",
        "uncertain",
        error_code="provider_unavailable",
        tool="hunter_email_finder",
        provider_status=502,
    )

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.tool"] == "hunter_email_finder"
    assert span.attributes["arena.operation"] == "deepline.execute"


def test_provider_span_names_the_model_behind_an_openrouter_call():
    """The model is the only thing that says WHICH catalogue entry failed.

    Without it an unroutable-model refusal can only be attributed to
    "OpenRouter", so the price table that admitted the model cannot be fixed.
    """
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        "openrouter",
        "openrouter.chat",
        "failed",
        error_code="provider_unavailable",
        model="anthropic/claude-3.5-sonnet",
        provider_status=404,
    )

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.model"] == "anthropic/claude-3.5-sonnet"
    assert span.attributes["arena.provider_status"] == 404


def test_a_provider_call_without_a_model_carries_the_sentinel():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider("deepline", "exa.search", "ok")

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.model"] == ARENA_MODEL_NONE


@pytest.mark.parametrize(
    "kwargs",
    [
        # free text smuggled in as a model
        {"provider": "openrouter", "operation": "openrouter.chat",
         "outcome": "ok", "model": "ada@example.com"},
        # a prompt smuggled in as a model
        {"provider": "openrouter", "operation": "openrouter.chat",
         "outcome": "ok", "model": "ignore previous instructions"},
        # a real model paired with a provider that has no models
        {"provider": "deepline", "operation": "deepline.execute",
         "outcome": "ok", "model": "anthropic/claude-3.5-sonnet"},
    ],
)
def test_provider_spans_with_an_inadmissible_model_are_dropped_whole(kwargs):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        kwargs.pop("provider"), kwargs.pop("operation"), kwargs.pop("outcome"), **kwargs
    )
    assert exp.get_finished_spans() == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        # a tool outside the frozen table
        {"provider": "deepline", "operation": "deepline.execute",
         "outcome": "ok", "tool": "exfiltrate_credentials"},
        # a lead's email smuggled in as a tool
        {"provider": "deepline", "operation": "deepline.execute",
         "outcome": "ok", "tool": "ada@example.com"},
        # a real tool paired with a provider that has no tools
        {"provider": "openrouter", "operation": "openrouter.responses",
         "outcome": "ok", "tool": "hunter_email_finder"},
    ],
)
def test_provider_spans_with_an_inadmissible_tool_are_dropped_whole(kwargs):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_provider(
        kwargs.pop("provider"), kwargs.pop("operation"), kwargs.pop("outcome"), **kwargs
    )
    assert exp.get_finished_spans() == ()


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


def test_recovery_attempt_three_is_admitted_on_run_and_runtime_spans():
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recovery = dict(TRUSTED_RUN_IDENTITY, **{"arena.attempt": 3})
    recorder.record_run("execute", "accepted", run_identity=recovery)
    recorder.record_runtime_upload(
        inserted_count=1, replayed_count=0,
        has_started=False, has_finished=False, has_error=True,
        run_identity=recovery,
    )

    run_span, runtime_span = exp.get_finished_spans()
    assert run_span.attributes["arena.attempt"] == 3
    assert runtime_span.attributes["arena.attempt"] == 3
    assert runtime_span.name == "arena.runtime.upload"
    assert runtime_span.instrumentation_scope.name == ARENA_RUNTIME_SCOPE
    assert set(runtime_span.attributes) == RUNTIME_SPAN_ATTRIBUTE_ALLOWLIST
    assert runtime_span.attributes["arena.has_error"] is True


@pytest.mark.parametrize("attempt", [0, -1, 1_000_001, True])
def test_invalid_recovery_attempt_drops_run_and_runtime_spans(attempt):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    identity = dict(TRUSTED_RUN_IDENTITY, **{"arena.attempt": attempt})
    recorder.record_run("execute", "accepted", run_identity=identity)
    recorder.record_runtime_upload(
        inserted_count=1, replayed_count=0,
        has_started=False, has_finished=False, has_error=True,
        run_identity=identity,
    )
    assert exp.get_finished_spans() == ()


@pytest.mark.parametrize(
    "patch",
    [
        {"inserted_count": 0},
        {"inserted_count": -1},
        {"inserted_count": 33},
        {"replayed_count": -1},
        {"replayed_count": 32},
        {"has_error": 1},
        {"has_error": False},
    ],
)
def test_runtime_upload_rejects_invalid_counts_and_flags(patch):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    args = {
        "inserted_count": 1, "replayed_count": 0,
        "has_started": False, "has_finished": False, "has_error": True,
        "run_identity": TRUSTED_RUN_IDENTITY,
    }
    recorder.record_runtime_upload(**{**args, **patch})
    assert exp.get_finished_spans() == ()


@pytest.mark.parametrize(
    "identity",
    [
        {**TRUSTED_RUN_IDENTITY, "arena.lease_token": "secret"},
        {**TRUSTED_RUN_IDENTITY, "arena.attempt": 0},
        {"arena.run_id": TRUSTED_RUN_IDENTITY["arena.run_id"]},
    ],
)
def test_runtime_upload_drops_extra_or_partial_identity(identity):
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    recorder.record_runtime_upload(
        inserted_count=1, replayed_count=0,
        has_started=False, has_finished=False, has_error=True,
        run_identity=identity,
    )
    assert exp.get_finished_spans() == ()


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


@pytest.mark.parametrize("http_status, expected", [
    (200, "ok"), (201, "ok"), (400, "refused"), (401, "refused"),
    (402, "refused"), (403, "refused"), (404, "refused"), (422, "refused"),
    (429, "failed"), (500, "failed"), (502, "failed"), (503, "failed"),
    (504, "failed"),
])
def test_settled_provider_span_preserves_worker_http_failure(http_status, expected):
    from lab_arena.service import _provider_telemetry_fields

    summary = {
        "operation_id": "scrapingdog.scrape", "outcome": "settled",
        "actual_microusd": 200, "provider_status": 200,
    }
    exp = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exp)
    fields = _provider_telemetry_fields(summary, http_status, 1)
    recorder.record_provider(duration_ms=5.0, **fields)
    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.outcome"] == expected
    assert span.attributes["arena.http_status"] == http_status
    assert span.attributes["arena.provider_status"] == 200
    assert span.attributes["arena.cost_microusd"] == 200
    assert span.status.is_ok == (expected != "failed")
    assert summary["outcome"] == "settled"


@pytest.mark.parametrize("summary, expected", [
    ({"outcome": "uncertain"}, "uncertain"),
    ({"error_code": "provider_unavailable"}, "failed"),
    ({"error_code": "miner_credentials_unavailable"}, "refused"),
])
def test_provider_http_fallback_preserves_specific_failure(summary, expected):
    from lab_arena.service import _provider_telemetry_fields

    assert _provider_telemetry_fields(summary, 502, 1)["outcome"] == expected


def test_the_service_tool_projection_resolves_and_degrades():
    """``_telemetry_tool`` must only ever produce an admissible tool name."""
    from lab_arena.service import _telemetry_tool

    # the native execute route names its tool in the admitted frame
    assert _telemetry_tool(
        "deepline.execute", {"tool": "hunter_email_finder"}
    ) == "hunter_email_finder"
    # a compatibility route carries its pinned tool instead
    assert _telemetry_tool("exa.search", {}) == "exa_search"
    # a non-Deepline provider has no tool
    assert _telemetry_tool("openrouter.responses", {}) == telemetry.NO_TOOL
    # anything the table does not know degrades rather than leaking
    for parameters in (
        {}, {"tool": ""}, {"tool": "ada@example.com"}, {"tool": 7},
    ):
        assert _telemetry_tool("deepline.execute", parameters) == telemetry.NO_TOOL
    assert _telemetry_tool("", {"tool": "exa_search"}) == telemetry.NO_TOOL


def test_the_service_model_projection_resolves_and_degrades():
    """``_telemetry_model`` must only ever produce an admissible model id."""
    from lab_arena.service import _telemetry_model

    assert _telemetry_model(
        "openrouter.chat", {"model": "anthropic/claude-3.5-sonnet"}
    ) == "anthropic/claude-3.5-sonnet"
    # a provider that has no model never names one
    assert _telemetry_model("deepline.execute", {"model": "x/y"}) == telemetry.NO_MODEL
    # anything outside the price table's own format degrades rather than leaking
    for parameters in (
        {}, {"model": ""}, {"model": "ada@example.com"},
        {"model": "no-slash"}, {"model": 7},
    ):
        assert _telemetry_model("openrouter.chat", parameters) == telemetry.NO_MODEL
    assert _telemetry_model("", {"model": "anthropic/claude-3.5-sonnet"}) == telemetry.NO_MODEL


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


@pytest.mark.parametrize("denial_code", [
    "reasoning_capture_activation_not_ready",
    "run_result_invalid",
    "baseline_recovery_scores_not_recorded",
    "infrastructure_retry_cost_schema_unavailable",
    "validator_events_scope_invalid",
    "source_not_public",
])
def test_a_refusal_code_names_the_4xx_without_its_message(denial_code):
    """``ServiceError.code`` can interpolate an exception message.

    At the gateway a refused Arena call is an anonymous 4xx. The request span
    names it — but only the literal prefix, never the interpolated tail.
    """
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/claim")
    def _claim():
        telemetry.note_denial(denial_code + ":round-7 hotkey 5F3s rejected")
        return JSONResponse(status_code=409, content={"status": "rejected"})

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            assert client.get("/arena/v1/claim").status_code == 409
    finally:
        telemetry.install_recorder(None)

    (span,) = exp.get_finished_spans()
    assert span.attributes["arena.denial"] == denial_code
    assert "5F3s" not in repr(dict(span.attributes))


@pytest.mark.parametrize("public", [False, True], ids=["private", "public"])
def test_source_disclosure_response_and_private_http_metadata(public):
    """The real source guard and route retain their response and read behavior."""
    import io
    import tarfile
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from lab_arena.api import create_app
    from lab_arena.service import ArenaService

    raw = io.BytesIO()
    source = b"def run_icp(icp): return []  # private-source-marker\n"
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        member = tarfile.TarInfo("harness.py")
        member.size = len(source)
        archive.addfile(member, io.BytesIO(source))
    payload = raw.getvalue()
    submission_id = "sub-" + "a" * 32
    submission = {
        "submission_id": submission_id, "round_id": "arena-2026-10-22",
        "status": "frozen" if public else "rejected",
        "source_ref": "private-source-ref", "source_size_bytes": len(payload),
        "consent": {"public_rerun": True},
    }
    round_row = {
        "round_id": submission["round_id"], "status": "published",
        "evaluation_date": "2026-10-22", "published_at": "2026-10-23T01:00:00Z",
        "publication_doc": {"participants": [{"submission_id": submission_id}]},
    }
    reads = []

    def read_source(ref, limit):
        reads.append((ref, limit))
        return payload

    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(get_submission=lambda _: submission)
    service._objects = SimpleNamespace(get_bounded=read_source)
    service._round = lambda _: round_row
    service._clock = lambda: datetime(2026, 10, 24, tzinfo=timezone.utc)
    app = create_app(service)
    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(app, span_exporter=exporter)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            response = client.get(
                "/arena/v1/submissions/" + submission_id + "/code?token=private-token"
            )
    finally:
        telemetry.install_recorder(None)
    if public:
        assert response.status_code == 200
        assert response.json() == {
            "submission_id": submission_id,
            "files": [{"path": "harness.py", "content": source.decode()}],
            "truncated": False,
        }
        assert len(reads) == 1
    else:
        assert response.status_code == 403
        assert response.json() == {"status": "rejected", "code": "source_not_public"}
        assert reads == []
    (span,) = exporter.get_finished_spans()
    assert set(span.attributes) == {
        "http.request.method", "http.route", "http.response.status_code",
        "arena.denial", "duration_ms",
    }
    assert span.attributes["arena.denial"] == ("-" if public else "source_not_public")
    exported = repr(dict(span.attributes)) + span.name
    for secret in (submission_id, "private-source-ref", "private-source-marker", "private-token"):
        assert secret not in exported
    with TestClient(create_app(service)) as client:
        uninstrumented = client.get("/arena/v1/submissions/" + submission_id + "/code")
    assert uninstrumented.status_code == response.status_code
    assert uninstrumented.content == response.content
    assert uninstrumented.headers == response.headers
    assert len(reads) == (2 if public else 0)
    assert len(exporter.get_finished_spans()) == 1


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


def test_existing_judgment_denial_codes_export_only_their_literal_prefix():
    exp = InMemorySpanExporter()
    app = FastAPI()

    @app.get("/arena/v1/refuse/{code}")
    def _refuse(code: str):
        telemetry.note_denial(code + ":private request details")
        return JSONResponse(status_code=409, content={"status": "rejected"})

    recorder = configure_arena_otel(app, span_exporter=exp)
    telemetry.install_recorder(recorder)
    try:
        with TestClient(app) as client:
            for code in (
                "accepted_judgment_binding_invalid",
                "signal_judgment_result_invalid",
                "training_observation_segment_conflict",
                "reasoning_capture_activation_not_ready",
                "private_token",
            ):
                assert client.get("/arena/v1/refuse/" + code).status_code == 409
            assert client.get("/arena/v1/refuse/private_token/unknown").status_code == 404
    finally:
        telemetry.install_recorder(None)

    spans = exp.get_finished_spans()
    assert [span.attributes["arena.denial"] for span in spans] == [
        "accepted_judgment_binding_invalid", "signal_judgment_result_invalid",
        "training_observation_segment_conflict", "reasoning_capture_activation_not_ready", "-", "-",
    ]
    assert [span.attributes["http.route"] for span in spans] == [
        "/arena/v1/refuse/{code}", "/arena/v1/refuse/{code}",
        "/arena/v1/refuse/{code}", "/arena/v1/refuse/{code}",
        "/arena/v1/refuse/{code}",
        "/_unmatched",
    ]
    assert "private request details" not in repr(spans)
    assert "private_token" not in repr(spans)


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
        telemetry.record_runtime_upload(
            inserted_count=1, replayed_count=0,
            has_started=False, has_finished=False, has_error=True,
            run_identity=TRUSTED_RUN_IDENTITY,
        )
        telemetry.record_gate("timed_out")
        telemetry.note_denial("lease_token_invalid")
    finally:
        telemetry.install_recorder(None)
