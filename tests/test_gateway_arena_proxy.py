from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from gateway.api import arena_proxy
from gateway.middleware.body_size import BodySizeLimitMiddleware


def test_gateway_mounts_the_public_arena_proxy() -> None:
    source = (Path(__file__).resolve().parents[1] / "gateway/main.py").read_text(
        encoding="utf-8"
    )
    assert "app.include_router(arena_proxy_router)" in source
    assert "app.include_router(arena_testnet_proxy_router)" in source


def _app() -> TestClient:
    app = FastAPI()
    app.include_router(arena_proxy.router)
    app.include_router(arena_proxy.testnet_router)
    return TestClient(app)


def test_disabled_arena_is_not_public(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_MODE", "off")
    assert _app().get("/arena/v1/current").status_code == 404


def test_proxy_forwards_only_the_arena_request(monkeypatch):
    observed = {}

    async def forward(method, path, *, query, body, headers):
        observed.update(
            method=method,
            path=path,
            query=query,
            body=body,
            headers=dict(headers),
        )
        return httpx.Response(
            202,
            content=b'{"status":"accepted"}',
            headers={
                "content-type": "application/json",
                "x-content-type-options": "nosniff",
                "x-private": "no",
            },
        )

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    response = _app().post(
        "/arena/v1/submissions?one=1",
        content=b'{"submission":1}',
        headers={
            "content-type": "application/json",
            "x-lab-arena-lease": "a" * 64,
            "authorization": "must-not-cross",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    assert observed == {
        "method": "POST",
        "path": "v1/submissions",
        "query": "one=1",
        "body": b'{"submission":1}',
        "headers": {
            "content-type": "application/json",
            "x-lab-arena-lease": "a" * 64,
        },
    }
    assert "x-private" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


def test_proxy_refuses_oversized_and_invalid_paths(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_MODE", "shadow")
    client = _app()
    oversized = client.post(
        "/arena/v1/submissions",
        content=b"{}",
        headers={"content-length": str(arena_proxy._MAX_REQUEST_BYTES + 1)},
    )
    assert oversized.status_code == 413
    assert client.get("/arena/v1//current").status_code == 404


@pytest.fixture
def real_httpx_sidecar(monkeypatch):
    """Keep HTTPX URL validation and replace only its network transport."""
    forwarded = []
    real_client = httpx.AsyncClient

    def handle(request):
        forwarded.append(request)
        if request.method == "HEAD":
            return httpx.Response(204)
        return httpx.Response(200, json={"status": "ok"})

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy.httpx, "AsyncClient", client_factory)
    return forwarded


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("encoded", ["%00", "%09", "%0D", "%1F", "%7F"],
    ids=["nul", "tab", "cr", "unit_separator", "del"])
def test_decoded_control_paths_return_404_before_httpx(prefix, encoded, real_httpx_sidecar):
    response = _app().get(prefix + "/v1/current" + encoded)
    assert response.status_code == 404
    assert response.json() == {"detail": "arena path invalid"}
    assert real_httpx_sidecar == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
async def test_control_path_rejection_precedes_body_and_lease_preflight(prefix, real_httpx_sidecar):
    app = FastAPI()
    app.include_router(arena_proxy.router)
    app.include_router(arena_proxy.testnet_router)
    messages = []

    async def unread():
        pytest.fail("invalid path must be rejected before reading the body")

    async def send(message):
        messages.append(message)

    await app({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": prefix + "/v1/runs/r\x00/training-observations", "query_string": b"", "headers": [],
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, unread, send)
    assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 404
    assert real_httpx_sidecar == []


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("query", ["", "?value=%00%09%0D%1F%7F"])
def test_valid_paths_and_encoded_query_controls_still_forward(prefix, query, real_httpx_sidecar):
    response = _app().get(prefix + "/v1/current" + query)
    assert response.status_code == 200
    assert len(real_httpx_sidecar) == 1
    request = real_httpx_sidecar[0]
    assert request.url.path == "/arena/v1/current"
    assert request.url.port == (8793 if prefix == "/testnet/arena" else int(arena_proxy._MAIN_SIDECAR_PORT))
    assert request.url.query == query.removeprefix("?").encode()


async def _asgi_post(app, path, chunks, *, headers=()):
    incoming = iter(chunks)
    outgoing = []

    async def receive():
        return next(incoming)

    async def send(message):
        outgoing.append(message)

    await app({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "query_string": b"", "headers": list(headers),
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, receive, send)
    return outgoing


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("partial", [False, True], ids=["before_body", "midbody"])
@pytest.mark.parametrize("observation", [False, True])
async def test_disconnected_body_returns_400_without_post_forward(
    prefix, partial, observation, real_httpx_sidecar,
):
    chunks = ([{"type": "http.request", "body": b"partial", "more_body": True}] if partial else [])
    chunks.append({"type": "http.disconnect"})
    path = "/v1/runs/r1/training-observations" if observation else "/v1/submissions/presign"
    outgoing = await _asgi_post(_app().app, prefix + path, chunks,
        headers=[(b"x-lab-arena-lease", b"a" * 64)] if observation else [])
    assert [message["status"] for message in outgoing if message["type"] == "http.response.start"] == [400]
    assert b"".join(message.get("body", b"") for message in outgoing) == b'{"detail":"request body incomplete"}'
    assert [request.method for request in real_httpx_sidecar] == (["HEAD"] if observation else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
async def test_outer_stream_size_limit_keeps_single_413_without_exception(prefix, real_httpx_sidecar):
    app = BodySizeLimitMiddleware(_app().app, max_body_bytes=8)
    outgoing = await _asgi_post(app, prefix + "/v1/submissions/presign", [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"56789", "more_body": False},
    ])
    assert [message["status"] for message in outgoing if message["type"] == "http.response.start"] == [413]
    assert b"".join(message.get("body", b"") for message in outgoing) == b'{"detail":"Request body too large"}'
    assert real_httpx_sidecar == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
async def test_unexpected_body_error_still_surfaces(prefix, real_httpx_sidecar):
    async def receive():
        raise RuntimeError("unexpected receive failure")

    async def send(_message):
        pass

    with pytest.raises(RuntimeError, match="unexpected receive failure"):
        await _app().app({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": prefix + "/v1/submissions/presign", "query_string": b"", "headers": [],
            "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, receive, send)
    assert real_httpx_sidecar == []


def test_lease_image_access_preserves_authorization_and_no_store(monkeypatch):
    observed = {}

    async def forward(method, path, *, query, body, headers):
        observed.update(method=method, path=path, headers=dict(headers))
        assert query == "" and body == b""
        return httpx.Response(
            200, json={"schema_version": "leadpoet.arena.scorer_image_access.v1"},
            headers={"cache-control": "no-store", "x-content-type-options": "nosniff"},
        )

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    response = _app().get(
        "/arena/v1/runs/run-1/image-access",
        headers={"x-lab-arena-lease": "a" * 64, "authorization": "not-forwarded"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert observed == {
        "method": "GET", "path": "v1/runs/run-1/image-access",
        "headers": {"x-lab-arena-lease": "a" * 64},
    }


def test_proxy_uses_the_shared_larger_limit_only_for_completions(monkeypatch):
    observed = {}

    async def forward(method, path, *, query, body, headers):
        observed.update(method=method, path=path, body=body)
        return httpx.Response(200, content=b'{}')

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    body = b"x" * (arena_proxy._MAX_REQUEST_BYTES + 1)
    assert len(body) < arena_proxy._MAX_COMPLETION_REQUEST_BYTES

    complete = client.post("/arena/v1/runs/r1/complete", content=body)
    assert complete.status_code == 200
    assert observed == {
        "method": "POST",
        "path": "v1/runs/r1/complete",
        "body": body,
    }
    assert client.post("/arena/v1/submissions", content=body).status_code == 413
    assert client.post(
        "/arena/v1/runs/r1/complete",
        content=b"{}",
        headers={
            "content-length": str(
                arena_proxy._MAX_COMPLETION_REQUEST_BYTES + 1
            )
        },
    ).status_code == 413


@pytest.mark.parametrize("route", ["training-input", "training-observations"])
def test_proxy_allows_full_training_input_only_on_exact_route(monkeypatch, route):
    observed = {}
    async def forward(method, path, *, query, body, headers):
        if method == "HEAD":
            return httpx.Response(204)
        observed.update(path=path, body=body, headers=dict(headers))
        return httpx.Response(200, json={"status": "stored"})
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    body = b"x" * (arena_proxy._MAX_REQUEST_BYTES + 1)
    response = client.post(f"/arena/v1/runs/r1/{route}", content=body, headers={"x-lab-arena-lease": "a" * 64})
    assert response.status_code == 200
    assert observed == {"path": f"v1/runs/r1/{route}", "body": body, "headers": {"x-lab-arena-lease": "a" * 64}}
    assert client.post(f"/arena/v1/runs/r1/{route}/extra", content=body).status_code == 413
    limit = (arena_proxy.training_observations.MAX_OBJECT_BYTES if route == "training-observations"
             else arena_proxy.training_input.MAX_INPUT_BYTES)
    assert client.post(f"/arena/v1/runs/r1/{route}", content=b"{}", headers={"content-length": str(limit + 1)}).status_code == 413


def _observation_request(headers, receive):
    return Request({
        "type": "http", "method": "POST", "scheme": "http",
        "path": "/arena/v1/runs/r1/training-observations", "query_string": b"private=not-preflight",
        "server": ("testserver", 80), "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
    }, receive)


@pytest.mark.asyncio
@pytest.mark.parametrize("testnet", [False, True])
async def test_observation_upload_authorizes_before_reading_more_than_eight_mib(monkeypatch, testnet):
    events = []
    body = b"x" * (arena_proxy.training_input.MAX_INPUT_BYTES + 1)

    async def receive():
        events.append("body")
        return {"type": "http.request", "body": body, "more_body": False}

    async def forward(method, path, *, query, body, headers, **routing):
        events.append(method)
        assert routing == ({"testnet": True} if testnet else {})
        if method == "HEAD":
            assert path == "v1/runs/r1/training-observations" and query == "" and body == b""
            assert headers == {"x-lab-arena-lease": "a" * 64}
            return httpx.Response(204)
        assert path == "v1/runs/r1/training-observations"
        assert query == "private=not-preflight" and len(body) > 8 * 1024 * 1024
        assert headers == {"x-lab-arena-lease": "a" * 64, "content-type": "application/json"}
        return httpx.Response(200, json={"status": "stored"})

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    request = _observation_request({"x-lab-arena-lease": "a" * 64,
        "content-type": "application/json", "authorization": "never-forward"}, receive)
    response = await arena_proxy._proxy_request("v1/runs/r1/training-observations", request, testnet=testnet)
    assert response.status_code == 200 and events == ["HEAD", "body", "POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,status", [
    ({}, 401), ({"x-lab-arena-lease": "wrong"}, 401),
    ({"x-lab-arena-lease": "A" * 64}, 401),
    ({"content-length": str(256 * 1024 * 1024 + 1)}, 413),
    ({"content-length": "invalid"}, 400),
])
async def test_observation_upload_rejects_local_failures_before_preflight_or_body(monkeypatch, headers, status):
    async def forbidden(*_args, **_kwargs):
        pytest.fail("local rejection must not read the body or contact the sidecar")

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forbidden)
    with pytest.raises(arena_proxy.HTTPException) as error:
        await arena_proxy._proxy_request("v1/runs/r1/training-observations",
            _observation_request(headers, forbidden))
    assert error.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 409, 429, 503, 302])
async def test_observation_upload_preserves_preflight_denials_without_reading_body(monkeypatch, status):
    calls = []

    async def unread():
        pytest.fail("denied or expired leases must not consume the upload")

    async def forward(method, path, **kwargs):
        calls.append((method, path))
        return httpx.Response(status, json={"code": "quota_unavailable"}, headers={
            "cache-control": "no-store", "retry-after": "5", "x-private": "hidden", "location": "https://invalid.test"})

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    response = await arena_proxy._proxy_request("v1/runs/r1/training-observations",
        _observation_request({"x-lab-arena-lease": "a" * 64}, unread))
    assert response.status_code == status
    assert response.body == b'{"code":"quota_unavailable"}'
    assert response.headers["cache-control"] == "no-store" and response.headers["retry-after"] == "5"
    assert "x-private" not in response.headers and "location" not in response.headers
    assert calls == [("HEAD", "v1/runs/r1/training-observations")]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "unexpected_success"])
async def test_observation_upload_authority_check_fails_closed(monkeypatch, failure):
    async def unread():
        pytest.fail("unproved authority must not consume the upload")

    async def forward(*_args, **_kwargs):
        if failure == "transport":
            raise httpx.ConnectError("private exception must not reach response")
        return httpx.Response(200)

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    with pytest.raises(arena_proxy.HTTPException) as error:
        await arena_proxy._proxy_request("v1/runs/r1/training-observations",
            _observation_request({"x-lab-arena-lease": "a" * 64}, unread))
    assert error.value.status_code == 503 and error.value.detail == "agent competition is unavailable"


def test_proxy_applies_compact_trajectory_limit(monkeypatch):
    observed = {}

    async def forward(method, path, *, query, body, headers):
        observed.update(method=method, path=path, body=body, headers=dict(headers))
        return httpx.Response(200, json={"status": "accepted"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    accepted = client.post(
        "/arena/v1/runs/r1/trajectory",
        content=b"{}",
        headers={"x-lab-arena-lease": "a" * 64},
    )
    assert accepted.status_code == 200
    assert observed["path"] == "v1/runs/r1/trajectory"
    assert observed["headers"] == {"x-lab-arena-lease": "a" * 64}
    refused = client.post(
        "/arena/v1/runs/r1/trajectory",
        content=b"x" * (arena_proxy._MAX_TRAJECTORY_REQUEST_BYTES + 1),
    )
    assert refused.status_code == 413


def test_proxy_contains_sidecar_failure(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", fail)
    response = _app().get("/arena/v1/current")
    assert response.status_code == 503
    assert response.json() == {"detail": "agent competition is unavailable"}


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
def test_provider_proxy_allows_admission_execution_and_billing(monkeypatch, prefix):
    from lab_arena import operations
    from lab_arena.runner import MAX_PROVIDER_API_TIMEOUT_SECONDS

    required_seconds = (
        operations.BUDGET_ADMISSION_MAX_SECONDS
        + max(operation.timeout_seconds for operation in operations.OPERATIONS.values())
        + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
    )
    observed = []

    class SidecarClient:
        def __init__(self, *, timeout, **_kwargs):
            self.timeout = timeout
            observed.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def request(self, *_args, **_kwargs):
            # Model a valid slow response without sleeping or calling a provider.
            if self.timeout.read < required_seconds:
                raise httpx.ReadTimeout("billing response exceeded proxy deadline")
            return httpx.Response(200, json={"status": "settled"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy.httpx, "AsyncClient", SidecarClient)
    response = _app().post(
        prefix + "/v1/runs/run-1/provider",
        json={"timeout_ms": 120000},
        headers={"x-lab-arena-lease": "a" * 64},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "settled"}
    assert observed[0].read == MAX_PROVIDER_API_TIMEOUT_SECONDS
    assert observed[0].connect == 3.0
    assert observed[0].write == 30.0
    assert observed[0].pool == 3.0


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/arena/v1/current"),
        ("POST", "/arena/v1/submissions/presign"),
        ("POST", "/arena/v1/runs/run-1/complete"),
        ("GET", "/arena/v1/runs/run-1/provider"),
        ("POST", "/arena/v1/runs/run-1/provider/extra"),
    ],
)
def test_non_provider_proxy_timeout_is_unchanged(monkeypatch, method, path):
    observed = []

    class SidecarClient:
        def __init__(self, *, timeout, **_kwargs):
            observed.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def request(self, *_args, **_kwargs):
            return httpx.Response(200, json={})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy.httpx, "AsyncClient", SidecarClient)
    assert _app().request(method, path).status_code == 200
    assert observed[0].read == 150.0


def test_testnet_route_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_TESTNET_ENABLED", raising=False)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    assert _app().get("/testnet/arena/v1/current").status_code == 404


def test_testnet_proxy_preserves_boundaries_and_never_falls_back(monkeypatch):
    observed = []

    async def forward(method, path, *, query, body, headers, testnet=False):
        observed.append((method, path, headers, testnet))
        raise httpx.ConnectError("testnet unavailable")

    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setenv("LAB_ARENA_MODE", "off")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    response = client.get("/testnet/arena/v1/current", headers={"authorization": "not-forwarded"})
    assert response.status_code == 503
    assert observed == [("GET", "v1/current", {}, True)]
    assert client.get("/arena/v1/current").status_code == 404
    assert client.get("/testnet/arena/v1//current").status_code == 404
    assert client.post("/testnet/arena/v1/submissions", content=b"x" * (arena_proxy._MAX_REQUEST_BYTES + 1)).status_code == 413


def test_testnet_benchmark_is_private_and_never_reaches_sidecar(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise AssertionError("private testnet benchmark reached the sidecar")

    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", fail)
    client = _app()

    blocked = client.get(
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/benchmark?ignored=1"
    )
    assert blocked.status_code == 403
    assert blocked.json() == {"detail": "testnet benchmark is private"}
    assert client.get(
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/benchmark/"
    ).status_code == 404
    assert client.get(
        "/testnet/arena/v1/rounds//benchmark"
    ).status_code == 404
    # TestClient performs an additional decode: the double-encoded spelling
    # becomes the exact private path, while the next level exercises the
    # residual-percent rejection used by a one-decode ASGI server.
    assert client.get(
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/%2562enchmark"
    ).status_code == 403
    assert client.get(
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/%252562enchmark"
    ).status_code == 404
    assert client.get(
        "/testnet/arena/v%2531/rounds/arena-2026-09-05-testnet2/benchmark"
    ).status_code == 403
    assert client.get(
        "/testnet/arena/v%252531/rounds/arena-2026-09-05-testnet2/benchmark"
    ).status_code == 404


def test_testnet_privacy_guards_preserve_mainnet_and_private_worker_routes(monkeypatch):
    observed = []

    async def forward(method, path, *, query, body, headers, testnet=False):
        observed.append((method, path, query, body, testnet))
        return httpx.Response(200, content=b'{}')

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()

    assert client.get(
        "/arena/v1/rounds/arena-2026-09-05/benchmark"
    ).status_code == 200
    assert client.post(
        "/testnet/arena/v1/runs/claim", content=b'{"signed":"claim"}'
    ).status_code == 200
    blocked_results = client.get(
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/results/submission-1"
    )
    assert blocked_results.status_code == 403
    assert blocked_results.json() == {"detail": "testnet results are private"}
    for path in (
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2",
        "/testnet/arena/v1/rounds/arena-2026-09-05-testnet2/submissions",
        "/testnet/arena/v1/submissions/submission-1/code",
    ):
        blocked = client.get(path)
        assert blocked.status_code == 403
        assert blocked.json() == {"detail": "testnet evaluation is private"}
    assert observed == [
        (
            "GET",
            "v1/rounds/arena-2026-09-05/benchmark",
            "",
            b"",
            False,
        ),
        (
            "POST",
            "v1/runs/claim",
            "",
            b'{"signed":"claim"}',
            True,
        ),
    ]


def test_testnet_disclosure_guards_do_not_block_mainnet_reads(monkeypatch):
    observed = []

    async def forward(method, path, *, query, body, headers, testnet=False):
        observed.append((method, path, testnet))
        return httpx.Response(200, content=b'{}')

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    paths = (
        "v1/rounds/arena-2026-09-05",
        "v1/rounds/arena-2026-09-05/submissions",
        "v1/submissions/submission-1/code",
    )

    for path in paths:
        assert client.get("/arena/" + path).status_code == 200

    assert observed == [("GET", path, False) for path in paths]


def test_sidecar_destination_is_fixed_by_network(monkeypatch):
    import asyncio

    observed = []
    clients = []
    options = []

    class Client:
        def __init__(self, **kwargs):
            clients.append(self)
            options.append(kwargs)
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def request(self, method, url, **kwargs):
            observed.append(url)
            return httpx.Response(200)

    monkeypatch.setattr(arena_proxy.httpx, "AsyncClient", Client)
    for testnet in [False, True]:
        asyncio.run(arena_proxy._request_sidecar("GET", "v1/current", query="", body=b"", headers={}, testnet=testnet))
    asyncio.run(arena_proxy._request_sidecar(
        "POST", "v1/runs/r1/provider", query="", body=b"", headers={}
    ))
    assert observed == [
        "http://127.0.0.1:8792/arena/v1/current",
        "http://127.0.0.1:8793/arena/v1/current",
        "http://127.0.0.1:8792/arena/v1/runs/r1/provider",
    ]
    assert len({id(client) for client in clients}) == 3
    assert options[0]["verify"] is options[1]["verify"] is options[2]["verify"]
    assert [option["timeout"].read for option in options] == [
        150.0, 150.0, arena_proxy._PROVIDER_READ_TIMEOUT_SECONDS,
    ]
    assert all(option["timeout"].connect == 3.0 for option in options)
    assert all(option["timeout"].pool == 3.0 for option in options)


def test_sidecar_ssl_context_uses_httpx_default_trust_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time

    original = httpx.create_ssl_context
    calls = []

    def create_context(**kwargs):
        calls.append(kwargs)
        time.sleep(0.002)  # Make simultaneous first use overlap.
        return original(**kwargs)

    monkeypatch.setattr(arena_proxy, "_SIDECAR_SSL_CONTEXT", None)
    monkeypatch.setattr(httpx, "create_ssl_context", create_context)
    with ThreadPoolExecutor(max_workers=8) as pool:
        contexts = list(pool.map(lambda _: arena_proxy._sidecar_ssl_context(), range(8)))
    context = contexts[0]
    assert all(other is context for other in contexts)
    assert calls == [{"trust_env": False}]
    default = original(trust_env=False)
    assert context.verify_mode == default.verify_mode
    assert context.check_hostname == default.check_hostname
    assert context.cert_store_stats() == default.cert_store_stats()


def test_sidecar_ssl_context_keeps_cookies_and_connections_per_request(monkeypatch):
    import asyncio

    async def run():
        connections = 0
        cookies = []

        async def sidecar(reader, writer):
            nonlocal connections
            connections += 1
            try:
                while True:
                    try:
                        request = await reader.readuntil(b"\r\n\r\n")
                    except asyncio.IncompleteReadError:
                        break
                    cookies.append(b"\r\ncookie:" in request.lower())
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                        b"Set-Cookie: probe=1; Path=/\r\nConnection: keep-alive\r\n\r\nok"
                    )
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(sidecar, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(arena_proxy, "_SIDECAR_URL", f"http://127.0.0.1:{port}")
        try:
            for _ in range(2):
                response = await arena_proxy._request_sidecar(
                    "GET", "v1/current", query="", body=b"", headers={}
                )
                assert response.status_code == 200
                assert response.content == b"ok"
        finally:
            server.close()
            await server.wait_closed()
        assert connections == 2
        assert cookies == [False, False]

    asyncio.run(run())
