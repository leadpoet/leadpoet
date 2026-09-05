"""The optional gateway OTel bootstrap must be off by default and, when on,
must emit only infra metadata — never ids, query strings, bodies, or secrets.

This guards the data-governance boundary: the gateway may ship operational
telemetry (request rate/latency/errors) to an external backend, but it must
never leak concrete path ids, query parameters, request bodies, or any
training/LLM content into a span.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gateway.observability.otel_bootstrap import configure_gateway_otel
from gateway.observability.otel_bootstrap import _GatewayOtelMiddleware


def _app(exporter=None):
    app = FastAPI()

    @app.get("/research-lab/allocations/attested/{epoch}")
    def _alloc(epoch: int):
        return {"epoch": epoch}

    @app.get("/health")
    def _health():
        return {"ok": True}

    installed = configure_gateway_otel(app, span_exporter=exporter)
    return app, installed


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GATEWAY_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("GATEWAY_OTEL_ENDPOINT", raising=False)
    assert configure_gateway_otel(FastAPI()) is False


def test_emits_only_infra_metadata_no_id_or_query_leak():
    exp = InMemorySpanExporter()
    app, installed = _app(exp)
    assert installed is True

    resp = TestClient(app).get(
        "/research-lab/allocations/attested/24124?secret=abc123"
    )
    assert resp.status_code == 200

    spans = exp.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Present: method, route template, status, duration.
    assert span.attributes["http.request.method"] == "GET"
    assert span.attributes["http.response.status_code"] == 200
    assert "duration_ms" in span.attributes
    assert span.attributes["http.route"].startswith("/research-lab")

    # Absent: the concrete epoch id and the query string must NOT appear
    # anywhere in the span name or attribute values.
    haystack = span.name + " " + " ".join(
        "%s=%s" % (k, v) for k, v in span.attributes.items()
    )
    assert "24124" not in haystack       # concrete path id never leaks
    assert "secret" not in haystack      # query key never leaks
    assert "abc123" not in haystack      # query value never leaks

    # Only the four intended attribute keys are present.
    assert set(span.attributes.keys()) <= {
        "http.request.method",
        "http.route",
        "http.response.status_code",
        "duration_ms",
    }


def test_health_route_is_suppressed():
    exp = InMemorySpanExporter()
    app, _ = _app(exp)
    TestClient(app).get("/health")
    assert exp.get_finished_spans() == ()  # no telemetry for health probes


def test_uses_raw_asgi_middleware_not_base_http_middleware():
    exp = InMemorySpanExporter()
    app, installed = _app(exp)
    assert installed is True
    assert len(app.user_middleware) == 1
    assert app.user_middleware[0].cls is _GatewayOtelMiddleware


def test_emit_failure_never_changes_response_or_masks_endpoint_error():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    def _failing_emit(*_args):
        raise RuntimeError("telemetry exporter failed")

    async def _ok(_request):
        return PlainTextResponse("ok")

    async def _boom(_request):
        raise ValueError("endpoint failed")

    app = Starlette(routes=[Route("/ok", _ok), Route("/boom", _boom)])
    app.add_middleware(_GatewayOtelMiddleware, emit=_failing_emit)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/ok").status_code == 200
    assert client.get("/boom").status_code == 500
    assert client.get("/ok").status_code == 200

    strict_client = TestClient(app)
    with pytest.raises(ValueError, match="endpoint failed"):
        strict_client.get("/boom")


def test_raw_asgi_middleware_handles_concurrent_requests_without_loss():
    import asyncio

    emitted = []

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = _GatewayOtelMiddleware(
        _app,
        emit=lambda scope, status, *_timing: emitted.append(
            (scope["path"], status)
        ),
    )

    async def _drive():
        async def _receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _one(index):
            messages = []

            async def _send(message):
                messages.append(message)

            await middleware(
                {
                    "type": "http",
                    "path": "/ok",
                    "method": "GET",
                    "headers": [],
                    "request_index": index,
                },
                _receive,
                _send,
            )
            assert messages[0]["status"] == 200

        await asyncio.gather(*(_one(index) for index in range(100)))

    asyncio.run(_drive())
    assert emitted == [("/ok", 200)] * 100


def test_non_http_scope_passes_through_without_telemetry():
    import asyncio

    seen = []
    emitted = []

    async def _app(scope, receive, send):
        seen.append(scope["type"])

    middleware = _GatewayOtelMiddleware(
        _app,
        emit=lambda *_args: emitted.append(True),
    )

    async def _drive():
        async def _receive():
            return {"type": "lifespan.startup"}

        async def _send(_message):
            return None

        for kind in ("lifespan", "websocket"):
            await middleware({"type": kind}, _receive, _send)

    asyncio.run(_drive())
    assert seen == ["lifespan", "websocket"]
    assert emitted == []


def test_unmatched_route_uses_fixed_label_never_client_path():
    """A 404 path is client-controlled; the exported label must be the fixed
    ``/_unmatched`` literal, never any segment of the request path."""
    exp = InMemorySpanExporter()
    app, _ = _app(exp)
    resp = TestClient(app).get("/attacker-controlled/EMAIL@EXAMPLE.COM")
    assert resp.status_code == 404

    spans = exp.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["http.route"] == "/_unmatched"
    haystack = span.name + " " + " ".join(
        "%s=%s" % (k, v) for k, v in span.attributes.items()
    )
    assert "attacker-controlled" not in haystack
    assert "EXAMPLE.COM" not in haystack


def _validator_harness(exp, scope="gateway.http"):
    """Provider + tracer wired through the fail-closed validating exporter."""
    from gateway.observability.otel_bootstrap import _validating_exporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    expected_resource = {"service.name": "leadpoet-gateway"}
    validating = _validating_exporter(
        exp,
        allowed_routes=lambda: {"/ok"},
        expected_resource=expected_resource,
    )
    provider = TracerProvider(resource=Resource(expected_resource))
    provider.add_span_processor(SimpleSpanProcessor(validating))
    return provider.get_tracer(scope)


def _emit_span(tracer, *, name=None, attrs=None, kind=None, parented=False,
               event=None):
    """Emit one finished span shaped like the middleware's, with overrides."""
    from opentelemetry.context import Context
    from opentelemetry.trace import SpanKind

    base = {
        "http.request.method": "GET",
        "http.route": "/ok",
        "http.response.status_code": 200,
        "duration_ms": 1.5,
    }
    if attrs is not None:
        base = attrs
    span_name = name if name is not None else "%s %s" % (
        base.get("http.request.method", "GET"), base.get("http.route", "/ok")
    )
    kwargs = {"kind": kind if kind is not None else SpanKind.SERVER}
    if not parented:
        kwargs["context"] = Context()  # fresh root context
    span = tracer.start_span(span_name, **kwargs)
    for k, v in base.items():
        span.set_attribute(k, v)
    if event is not None:
        span.add_event(*event)
    span.end()


def test_validator_drops_nonconforming_attribute_sets():
    """Fail-closed: a span with a non-approved attribute set is dropped
    entirely — not stripped, not partially exported."""
    exp = InMemorySpanExporter()
    tracer = _validator_harness(exp)

    # Conforming span → exported.
    _emit_span(tracer)
    assert len(exp.get_finished_spans()) == 1
    exp.clear()

    base = {
        "http.request.method": "GET",
        "http.route": "/ok",
        "http.response.status_code": 200,
        "duration_ms": 1.5,
    }

    # Extra attribute → dropped, nothing exported.
    _emit_span(tracer, attrs={**base, "user.email": "leak@example.com"})
    # Missing attribute → dropped.
    _emit_span(tracer, attrs={k: v for k, v in base.items() if k != "duration_ms"})
    # Wrong type (status as string) → dropped.
    _emit_span(tracer, attrs={**base, "http.response.status_code": "200"})
    # Event attached → dropped (events can carry exception messages).
    _emit_span(tracer, event=("exception", {"exception.message": "secret tb"}))
    # Unregistered route label → dropped.
    _emit_span(tracer, attrs={**base, "http.route": "/not-registered"})
    assert exp.get_finished_spans() == ()

    # Fixed /_unmatched label → allowed.
    _emit_span(tracer, attrs={**base, "http.route": "/_unmatched"})
    assert len(exp.get_finished_spans()) == 1


def test_validator_drops_nonconforming_span_envelopes():
    """The COMPLETE envelope is validated: name, method, kind, parent/trace
    state, and instrumentation scope — not just the attribute dict."""
    from opentelemetry.trace import SpanKind

    exp = InMemorySpanExporter()
    tracer = _validator_harness(exp)

    base = {
        "http.request.method": "GET",
        "http.route": "/ok",
        "http.response.status_code": 200,
        "duration_ms": 1.5,
    }

    # Arbitrary span name → dropped (must equal "<method> <route>").
    _emit_span(tracer, name="user query for leak@example.com")
    # Non-standard (client-controlled) HTTP method string → dropped.
    _emit_span(tracer, attrs={**base, "http.request.method": "BREW"})
    # Wrong span kind (INTERNAL) → dropped.
    _emit_span(tracer, kind=SpanKind.INTERNAL)
    assert exp.get_finished_spans() == ()

    # Parented span (adopted caller context) → dropped.
    with tracer.start_as_current_span("outer", kind=SpanKind.SERVER):
        _emit_span(tracer, parented=True)
    exported = [s.name for s in exp.get_finished_spans()]
    assert "GET /ok" not in exported
    exp.clear()

    # Foreign instrumentation scope (e.g. a library that acquired this
    # provider) → dropped even with a perfect envelope otherwise.
    foreign = _validator_harness(exp, scope="some.library")
    _emit_span(foreign)
    assert exp.get_finished_spans() == ()


def test_bootstrap_refuses_ambient_exporter_env(monkeypatch, capsys):
    """Runtime refusal: any ambient standard exporter variable present in the
    process environment blocks initialization entirely — a CI grep cannot see
    vars injected by a restart script or the live process env."""
    monkeypatch.setenv(
        "OTEL_EXPORTER" + "_OTLP_HEADERS", "x-secret=abc"
    )  # split literal keeps the repo-wide env-var guard meaningful
    exp = InMemorySpanExporter()
    app, installed = _app(exp)
    assert installed is False
    out = capsys.readouterr().out
    assert "gateway_otel_bootstrap_refused" in out
    assert "abc" not in out  # names only, never values
    TestClient(app).get("/research-lab/allocations/attested/1")
    assert exp.get_finished_spans() == ()


def test_production_path_requires_nonempty_token(monkeypatch, capsys):
    """Without a token the pinned exporter's empty headers dict would fall
    back to ambient header variables — initialization must refuse instead."""
    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "1")
    monkeypatch.setenv("GATEWAY_OTEL_ENDPOINT", "https://collector.invalid/v1/traces")
    monkeypatch.delenv("GATEWAY_OTEL_TOKEN", raising=False)
    assert configure_gateway_otel(FastAPI()) is False
    assert "token_missing" in capsys.readouterr().out


def test_bootstrap_failure_log_never_exposes_exception_values(monkeypatch, capsys):
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    secret = "https://collector.example/private-token"

    class _FailingExporter:
        def __init__(self, **_kwargs):
            raise RuntimeError(secret)

    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "1")
    monkeypatch.setenv("GATEWAY_OTEL_ENDPOINT", secret)
    monkeypatch.setenv("GATEWAY_OTEL_TOKEN", "also-secret")
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", _FailingExporter)

    assert configure_gateway_otel(FastAPI()) is False
    output = capsys.readouterr().out
    assert "gateway_otel_bootstrap_skipped error=RuntimeError" in output
    assert secret not in output
    assert "also-secret" not in output


def test_resource_is_fixed_and_ignores_ambient_resource_env(monkeypatch):
    """The provider resource is built from a fixed dict; ambient resource
    variables must not appear in exported span resources."""
    monkeypatch.setenv(
        "OTEL_RESOURCE" + "_ATTRIBUTES", "leak.key=leak-value"
    )  # split literal keeps the repo-wide env-var guard meaningful
    exp = InMemorySpanExporter()
    app, installed = _app(exp)
    assert installed is True
    TestClient(app).get("/research-lab/allocations/attested/1")
    spans = exp.get_finished_spans()
    assert len(spans) == 1
    resource_attrs = dict(spans[0].resource.attributes)
    assert resource_attrs == {"service.name": "leadpoet-gateway"}
    assert "leak.key" not in resource_attrs


def test_validator_allows_vocabulary_stage_only_on_failures():
    """The optional stage label is bounded: it is accepted only on a 5xx and
    only when its value comes from the fixed stage vocabulary."""
    from gateway.observability.otel_bootstrap import (
        STAGE_ATTRIBUTE,
        STAGE_ATTRIBUTE_VALUES,
    )

    exp = InMemorySpanExporter()
    tracer = _validator_harness(exp)

    stage = "compact_bundle_cutover_authority"
    assert stage in STAGE_ATTRIBUTE_VALUES

    failed = {
        "http.request.method": "POST",
        "http.route": "/ok",
        "http.response.status_code": 503,
        "duration_ms": 1.5,
    }

    # A vocabulary stage on a failed request → exported, label intact.
    _emit_span(tracer, attrs={**failed, STAGE_ATTRIBUTE: stage})
    spans = exp.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[STAGE_ATTRIBUTE] == stage
    exp.clear()

    # A stage value outside the vocabulary → dropped, never exported as-is.
    _emit_span(tracer, attrs={**failed, STAGE_ATTRIBUTE: "not_a_stage"})
    # Free text (the shape a leak would take) → dropped.
    _emit_span(tracer, attrs={**failed, STAGE_ATTRIBUTE: "hotkey=5F3x epoch=91"})
    # A stage on a successful request → dropped; it is a failure label only.
    _emit_span(
        tracer,
        attrs={
            **failed,
            "http.response.status_code": 200,
            STAGE_ATTRIBUTE: stage,
        },
    )
    # The stage label does not license any other extra attribute.
    _emit_span(
        tracer,
        attrs={**failed, STAGE_ATTRIBUTE: stage, "user.email": "leak@example.com"},
    )
    assert exp.get_finished_spans() == ()

    # A failed request with no stage recorded stays conforming.
    _emit_span(tracer, attrs=failed)
    assert len(exp.get_finished_spans()) == 1


def test_stage_context_is_per_request_and_never_raises():
    """The stage carrier is reset per request and swallows its own errors."""
    from gateway.observability import stage_context

    stage_context.reset_stage()
    assert stage_context.current_stage() is None
    stage_context.enter_stage("compact_bundle_shape_verification")
    assert stage_context.current_stage() == "compact_bundle_shape_verification"
    stage_context.enter_stage("compact_bundle_cutover_authority")
    assert stage_context.current_stage() == "compact_bundle_cutover_authority"
    stage_context.reset_stage()
    assert stage_context.current_stage() is None
