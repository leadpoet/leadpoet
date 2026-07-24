"""PriorityMiddleware is pure ASGI and cannot hit the anyio task-group race.

Starlette ``BaseHTTPMiddleware`` bridges the app through an anyio task group;
under concurrency, or when the endpoint raises / the client disconnects, that
group's ``__aexit__`` mutates its task deque during iteration and raises
``RuntimeError: deque mutated during iteration``, turning a valid response
into an intermittent 5xx. On the weight path that dropped whole epochs. The
pure-ASGI implementation awaits the app directly, so the wrapper — and the
race — are gone, while the concurrency-pool accounting is unchanged.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.middleware.priority import (
    PriorityMiddleware,
    RouteClass,
    _Pool,
    classify_path,
)
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _http_scope(path: str = "/other", method: str = "GET") -> dict:
    return {"type": "http", "path": path, "method": method, "headers": []}


async def _receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


class _Collector:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m["status"]
        return None


def test_classify_path_unchanged():
    assert classify_path("/weights/submit/v2") == "validator"
    assert classify_path("/epoch/24123") == "validator"
    assert classify_path("/fulfillment/requests/active") == "miner"
    assert classify_path("/anything-else") == "other"


def test_http_request_passes_through_and_releases_slot():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = PriorityMiddleware(app)
    send = _Collector()
    asyncio.run(mw(_http_scope("/weights/submit/v2"), _receive, send))

    assert send.status == 200
    # The validator slot must be fully released after a normal response.
    assert mw.pools["validator"].in_flight == 0


def test_downstream_exception_releases_slot_and_propagates():
    # This is the exact scenario the old BaseHTTPMiddleware masked as
    # "deque mutated during iteration": the endpoint raises. Pure ASGI must
    # propagate the real error AND release the slot in finally.
    async def app(scope, receive, send):
        raise RuntimeError("endpoint failed")

    mw = PriorityMiddleware(app)
    send = _Collector()
    with pytest.raises(RuntimeError, match="endpoint failed"):
        asyncio.run(mw(_http_scope("/weights/submit/v2"), _receive, send))

    assert mw.pools["validator"].in_flight == 0  # released despite the raise


def test_non_http_scope_passes_through_untouched():
    seen = {}

    async def app(scope, receive, send):
        seen["type"] = scope["type"]

    mw = PriorityMiddleware(app)
    send = _Collector()
    for kind in ("websocket", "lifespan"):
        asyncio.run(mw({"type": kind}, _receive, send))
        assert seen["type"] == kind
    # No HTTP pooling occurred for non-http scopes.
    assert all(p.requests == 0 for p in mw.pools.values())


def test_shed_returns_503_without_calling_app():
    async def app(scope, receive, send):
        raise AssertionError("app must not run when the request is shed")

    mw = PriorityMiddleware(app)
    # max_waiting=0 makes acquire() shed every request deterministically.
    mw.pools["other"] = _Pool(RouteClass("other", 1, 0, 1.0))
    send = _Collector()
    asyncio.run(mw(_http_scope("/random-path"), _receive, send))

    assert send.status == 503
    assert mw.pools["other"].shed == 1
    assert mw.pools["other"].in_flight == 0


def test_no_slot_leak_under_repeated_exceptions():
    async def app(scope, receive, send):
        raise RuntimeError("boom")

    mw = PriorityMiddleware(app)

    async def drive():
        for _ in range(50):
            with pytest.raises(RuntimeError):
                await mw(_http_scope("/weights/submit/v2"), _receive, _Collector())

    asyncio.run(drive())
    # 50 failing requests must leave zero in-flight and full capacity.
    assert mw.pools["validator"].in_flight == 0
    assert (
        mw.pools["validator"].semaphore._value
        == mw.pools["validator"].route_class.max_concurrent
    )


def test_integration_in_real_starlette_stack():
    async def ok(request):
        return PlainTextResponse("ok")

    async def boom(request):
        raise RuntimeError("kaboom")

    app = Starlette(routes=[Route("/validate", ok), Route("/boom", boom)])
    app.add_middleware(PriorityMiddleware, max_concurrent_miners=75)

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/validate").status_code == 200
    # An unhandled endpoint error becomes a clean 500 via ServerErrorMiddleware,
    # never a masked deque error, and the slot is released.
    assert client.get("/boom").status_code == 500
    # Subsequent requests still succeed (no slot leak from the error above).
    assert client.get("/validate").status_code == 200
