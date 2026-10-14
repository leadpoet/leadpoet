"""Only the gateway's actual pre-router capacity rejection gets admission tags."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.middleware.priority import PriorityMiddleware, RouteClass, _Pool
from gateway.observability.otel_bootstrap import configure_gateway_otel
from tests.test_gateway_otel_bootstrap import _emit_span, _validator_harness

BASE_FIELDS = {
    "http.request.method", "http.route", "http.response.status_code", "duration_ms",
}
CAPACITY_FIELDS = {"gateway.admission", "gateway.priority_class"}
CANARY = "PRIVATE-CONTENT@example.invalid"


def _app(*, status=200):
    app = FastAPI()
    pools = []
    called = []

    @app.api_route("/arena/{arena_path:path}", methods=["POST"])
    async def arena(arena_path: str):
        called.append(True)
        return JSONResponse({"status": "unchanged"}, status_code=status)

    class CapturedPriority(PriorityMiddleware):
        def __init__(self, app):
            super().__init__(app)
            pools.append(self)

    app.add_middleware(CapturedPriority)
    exporter = InMemorySpanExporter()
    assert configure_gateway_otel(app, span_exporter=exporter)
    app.middleware_stack = app.build_middleware_stack()
    return app, pools[0], exporter, called


@pytest.mark.parametrize("mode", ["queue_full", "wait_timeout"])
@pytest.mark.parametrize(("path", "pool_class"), [
    ("/arena/v1/submissions/" + CANARY + "/finalize", "miner"),
    ("/testnet/arena/v1/submissions/" + CANARY + "/finalize", "miner"),
    ("/arena/v1/runs/claim", "other"),
    ("/unknown/" + CANARY, "other"),
    ("/epoch/" + CANARY, "validator"),
])
def test_capacity_rejection_exports_only_fixed_pair(mode, path, pool_class):
    app, priority, exporter, called = _app()
    limits = RouteClass(pool_class, 1, 0, 1) if mode == "queue_full" else RouteClass(pool_class, 0, 2, .001)
    priority.pools[pool_class] = _Pool(limits)
    with TestClient(app) as client:
        response = client.post(path + "?private=" + CANARY, content=CANARY,
                               headers={"authorization": CANARY})
    assert response.status_code == 503
    assert response.json() == {"detail": "Gateway at capacity - retry shortly"}
    assert response.headers["retry-after"] == "15"
    assert called == []
    span, = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert set(attrs) == BASE_FIELDS | CAPACITY_FIELDS
    assert attrs["http.route"] == "/_unmatched"
    assert attrs["http.request.method"] == "POST"
    assert attrs["http.response.status_code"] == 503
    assert attrs["gateway.admission"] == "capacity_shed"
    assert attrs["gateway.priority_class"] == pool_class
    assert CANARY not in span.name + str(attrs) + str(span.resource.attributes)
    assert priority.pools[pool_class].shed == 1
    assert priority.pools[pool_class].waiting == 0
    assert priority.pools[pool_class].in_flight == 0
    assert priority.pools[pool_class]._active_calls == 0


@pytest.mark.parametrize(("path", "status", "label"), [
    ("/unknown/" + CANARY, 404, "/_unmatched"),
    ("/arena/v1/runs/claim", 503, "/arena/{arena_path:path}"),
    ("/arena/v1/submissions/presign", 409, "/arena/{arena_path:path}"),
    ("/arena/v1/runs/claim", 200, "/arena/{arena_path:path}"),
])
def test_ordinary_responses_cannot_claim_capacity_shedding(path, status, label):
    app, _, exporter, _ = _app(status=status)
    with TestClient(app) as client:
        response = client.post(path, headers={
            "gateway.admission": "capacity_shed", "gateway.priority_class": "miner",
            "_gateway_capacity_shed_class": "miner", "authorization": CANARY,
        })
    assert response.status_code == status
    span, = exporter.get_finished_spans()
    assert set(span.attributes) == BASE_FIELDS
    assert span.attributes["http.route"] == label
    assert CANARY not in span.name + str(dict(span.attributes))


@pytest.mark.parametrize("pool_class", ["miner", "validator", "other"])
def test_exporter_accepts_exact_capacity_pair(pool_class):
    exporter = InMemorySpanExporter()
    tracer = _validator_harness(exporter)
    _emit_span(tracer, attrs={
        "http.request.method": "POST", "http.route": "/_unmatched",
        "http.response.status_code": 503, "duration_ms": 1.0,
        "gateway.admission": "capacity_shed", "gateway.priority_class": pool_class,
    })
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("change", [
    {"gateway.admission": None}, {"gateway.priority_class": None},
    {"gateway.admission": "unknown"}, {"gateway.priority_class": "unknown"},
    {"gateway.priority_class": True}, {"http.response.status_code": 200},
    {"private.path": CANARY}, {"duration_ms": None},
])
def test_exporter_drops_invalid_or_partial_capacity_pair(change):
    exporter = InMemorySpanExporter()
    tracer = _validator_harness(exporter)
    attrs = {
        "http.request.method": "POST", "http.route": "/_unmatched",
        "http.response.status_code": 503, "duration_ms": 1.0,
        "gateway.admission": "capacity_shed", "gateway.priority_class": "miner",
    }
    for key, value in change.items():
        if value is None:
            attrs.pop(key)
        else:
            attrs[key] = value
    _emit_span(tracer, attrs=attrs)
    assert exporter.get_finished_spans() == ()


def _scope(path):
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path, "root_path": "",
            "query_string": b"", "headers": [], "server": ("localhost", 80),
            "client": ("127.0.0.1", 1234)}


def test_cancelled_waiter_and_other_request_do_not_share_admission_marker():
    app, priority, exporter, called = _app()
    pool = priority.pools["miner"] = _Pool(RouteClass("miner", 0, 2, 10))
    blocked = _scope("/arena/v1/submissions/presign")
    normal = _scope("/arena/v1/runs/claim")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        pass

    async def drive():
        task = asyncio.create_task(app(blocked, receive, send))
        async def waiting():
            while not pool.waiting:
                await asyncio.sleep(0)
        await asyncio.wait_for(waiting(), 1)
        priority.pools["other"] = _Pool(RouteClass("other", 1, 0, 1))
        await app(_scope("/unknown/" + CANARY), receive, send)
        priority.pools["other"] = _Pool(RouteClass("other", 1, 2, 1))
        await app(normal, receive, send)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "_gateway_capacity_shed_class" not in blocked
        assert pool.waiting == pool.in_flight == pool._active_calls == 0
        # Reused scope cannot inherit a prior internal marker either.
        blocked["_gateway_capacity_shed_class"] = "miner"
        priority.pools["miner"] = _Pool(RouteClass("miner", 1, 2, 1))
        await app(blocked, receive, send)
        assert "_gateway_capacity_shed_class" not in blocked

    asyncio.run(drive())
    assert called == [True, True]
    spans = exporter.get_finished_spans()
    assert sorted(s.attributes["http.response.status_code"] for s in spans) == [200, 200, 500, 503]
    for span in spans:
        if span.attributes["http.response.status_code"] == 503:
            assert set(span.attributes) == BASE_FIELDS | CAPACITY_FIELDS
            assert span.attributes["gateway.priority_class"] == "other"
        else:
            assert set(span.attributes) == BASE_FIELDS
        assert CANARY not in span.name + str(dict(span.attributes))
