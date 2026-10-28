"""A body refused before the service is reached still names why on its span."""
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import ARENA_DENIAL_CODES, configure_arena_otel
from lab_arena import telemetry
from lab_arena.api import MAX_JSON_BODY_BYTES, create_app
from tests.lab_arena.test_lab_arena_api import StubService

LEASE = "a" * 64
ROUTE = "/arena/v1/runs/run-1/signal-judgment"


@pytest.fixture
def observed_client():
    exporter = InMemorySpanExporter()
    app = create_app(StubService())
    telemetry.install_recorder(configure_arena_otel(app, span_exporter=exporter))
    try:
        with TestClient(app) as client:
            yield client, exporter
    finally:
        telemetry.install_recorder(None)


def _denial(exporter, route):
    span, = [s for s in exporter.get_finished_spans() if s.attributes["http.route"] == route]
    return span.attributes["http.response.status_code"], span.attributes["arena.denial"]


@pytest.mark.parametrize("body,status,denial", [
    (b"{not json", 400, "request_body_not_json"),
    (b'{"a": ' + b'[' * 64 + b']' * 64 + b"}", 400, "contract"),
])
def test_a_refused_body_names_its_refusal_on_the_span(observed_client, body, status, denial):
    client, exporter = observed_client
    response = client.post(ROUTE, content=body, headers={"x-lab-arena-lease": LEASE})
    assert response.status_code == status
    assert _denial(exporter, "/arena/v1/runs/{run_id}/signal-judgment") == (status, denial)


def test_a_declared_oversize_body_names_its_refusal_on_the_span(observed_client):
    client, exporter = observed_client
    response = client.post(
        "/arena/v1/runs/run-1/trajectory", content=b"{}",
        headers={"x-lab-arena-lease": LEASE,
                 "content-length": str(MAX_JSON_BODY_BYTES * 4)},
    )
    assert response.status_code == 413
    assert _denial(exporter, "/arena/v1/runs/{run_id}/trajectory")[1] == "request_body_too_large"


@pytest.mark.parametrize("code", ["request_body_not_json", "request_body_too_large",
                                  "content_length_invalid", "contract"])
def test_every_body_refusal_code_may_leave_the_sidecar(code):
    assert code in ARENA_DENIAL_CODES
