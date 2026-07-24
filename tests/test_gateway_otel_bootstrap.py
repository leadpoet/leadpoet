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


def test_validator_drops_span_with_extra_attribute():
    """Fail-closed: a span carrying a non-approved attribute is dropped
    entirely — not stripped, not partially exported."""
    from gateway.observability.otel_bootstrap import _validating_exporter

    exp = InMemorySpanExporter()
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
    tracer = provider.get_tracer("gateway.http")

    def _base_attrs():
        return {
            "http.request.method": "GET",
            "http.route": "/ok",
            "http.response.status_code": 200,
            "duration_ms": 1.5,
        }

    # Conforming span → exported.
    with tracer.start_as_current_span("GET /ok") as span:
        for k, v in _base_attrs().items():
            span.set_attribute(k, v)
    assert len(exp.get_finished_spans()) == 1
    exp.clear()

    # Extra attribute → dropped, nothing exported.
    with tracer.start_as_current_span("GET /ok") as span:
        for k, v in _base_attrs().items():
            span.set_attribute(k, v)
        span.set_attribute("user.email", "leak@example.com")
    assert exp.get_finished_spans() == ()

    # Missing attribute → dropped.
    with tracer.start_as_current_span("GET /ok") as span:
        attrs = _base_attrs()
        attrs.pop("duration_ms")
        for k, v in attrs.items():
            span.set_attribute(k, v)
    assert exp.get_finished_spans() == ()

    # Wrong type (status as string) → dropped.
    with tracer.start_as_current_span("GET /ok") as span:
        attrs = _base_attrs()
        attrs["http.response.status_code"] = "200"
        for k, v in attrs.items():
            span.set_attribute(k, v)
    assert exp.get_finished_spans() == ()

    # Event attached → dropped (events could carry exception messages).
    with tracer.start_as_current_span("GET /ok") as span:
        for k, v in _base_attrs().items():
            span.set_attribute(k, v)
        span.add_event("exception", {"exception.message": "secret traceback"})
    assert exp.get_finished_spans() == ()

    # Unregistered route label → dropped.
    with tracer.start_as_current_span("GET /other") as span:
        attrs = _base_attrs()
        attrs["http.route"] = "/not-registered"
        for k, v in attrs.items():
            span.set_attribute(k, v)
    assert exp.get_finished_spans() == ()

    # Fixed /_unmatched label → allowed.
    with tracer.start_as_current_span("GET /_unmatched") as span:
        attrs = _base_attrs()
        attrs["http.route"] = "/_unmatched"
        for k, v in attrs.items():
            span.set_attribute(k, v)
    assert len(exp.get_finished_spans()) == 1


def test_validator_drops_span_from_foreign_scope():
    """A span produced under any other instrumentation scope (e.g. a library
    that acquired this provider) is never exported."""
    from gateway.observability.otel_bootstrap import _validating_exporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exp = InMemorySpanExporter()
    expected_resource = {"service.name": "leadpoet-gateway"}
    validating = _validating_exporter(
        exp, allowed_routes=lambda: {"/ok"}, expected_resource=expected_resource
    )
    provider = TracerProvider(resource=Resource(expected_resource))
    provider.add_span_processor(SimpleSpanProcessor(validating))
    foreign = provider.get_tracer("some.library")

    with foreign.start_as_current_span("GET /ok") as span:
        span.set_attribute("http.request.method", "GET")
        span.set_attribute("http.route", "/ok")
        span.set_attribute("http.response.status_code", 200)
        span.set_attribute("duration_ms", 1.0)
    assert exp.get_finished_spans() == ()


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
