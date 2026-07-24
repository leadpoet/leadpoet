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
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
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
