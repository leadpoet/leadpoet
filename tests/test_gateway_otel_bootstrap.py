"""The optional gateway OTel bootstrap must be off by default and, when on,
must emit only infra metadata — never ids, query strings, bodies, or secrets.

This guards the data-governance boundary: the gateway may ship operational
telemetry (request rate/latency/errors) to an external backend, but it must
never leak concrete path ids, query parameters, request bodies, or any
training/LLM content into a span.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gateway.observability.otel_bootstrap import configure_gateway_otel


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
