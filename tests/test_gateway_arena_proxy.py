from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import arena_proxy


def _app() -> TestClient:
    app = FastAPI()
    app.include_router(arena_proxy.router)
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
