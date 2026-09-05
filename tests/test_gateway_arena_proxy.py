from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import arena_proxy


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
            headers={"content-type": "application/json", "x-private": "no"},
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


def test_proxy_contains_sidecar_failure(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", fail)
    response = _app().get("/arena/v1/current")
    assert response.status_code == 503
    assert response.json() == {"detail": "agent competition is unavailable"}


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


def test_sidecar_destination_is_fixed_by_network(monkeypatch):
    import asyncio

    observed = []

    class Client:
        def __init__(self, **kwargs):
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
    assert observed == ["http://127.0.0.1:8792/arena/v1/current", "http://127.0.0.1:8793/arena/v1/current"]
