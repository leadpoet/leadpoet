"""The stable Arena edge keeps the gateway proxy's public boundaries."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.api import arena_proxy
from scripts.run_arena_proxy_edge import create_app, load_edge_environment


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
def test_edge_observation_upload_crosses_both_old_limits_after_lease_preflight(monkeypatch, prefix):
    calls = []
    body = b"x" * (11 * 1024 * 1024)

    async def forward(method, path, *, query, body, headers, **routing):
        calls.append(method)
        assert routing == ({"testnet": True} if prefix.startswith("/testnet") else {})
        assert headers.get("x-lab-arena-lease") == "a" * 64
        assert "authorization" not in headers
        if method == "HEAD":
            assert path == "v1/runs/r1/training-observations" and query == "" and body == b""
            assert set(headers) == {"x-lab-arena-lease"}
            return httpx.Response(204)
        assert path == "v1/runs/r1/training-observations" and len(body) == 11 * 1024 * 1024
        return httpx.Response(200, json={"status": "stored"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = TestClient(create_app())
    result = client.post(prefix + "/v1/runs/r1/training-observations", content=body,
        headers={"x-lab-arena-lease": "a" * 64, "authorization": "never-forward"})
    assert result.status_code == 200 and calls == ["HEAD", "POST"]
    assert client.post(prefix + "/v1/runs/r1/training-input", content=body).status_code == 413
    assert client.post(prefix + "/v1/runs/r1/training-observations/extra", content=body).status_code == 413
    assert client.post(prefix + "/v1/runs/r1/training-observations", content=b"{}",
        headers={"content-encoding": "gzip"}).status_code == 415
    assert calls == ["HEAD", "POST"]


@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("streamed", [False, True])
def test_edge_segment_upload_uses_exact_segment_limit(monkeypatch, prefix, streamed):
    calls = []
    body = b"x" * arena_proxy.training_observations.MAX_SEGMENT_BYTES

    async def forward(method, path, *, query, body, headers, **routing):
        calls.append(method)
        assert method == "POST" and path == "v1/runs/r1/training-observation-segments"
        assert routing == ({"testnet": True} if prefix.startswith("/testnet") else {})
        assert headers == {"x-lab-arena-lease": "a" * 64}
        assert len(body) == arena_proxy.training_observations.MAX_SEGMENT_BYTES
        assert body == b"x" * len(body)
        return httpx.Response(200, json={"status": "stored"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = TestClient(create_app())
    route = prefix + "/v1/runs/r1/training-observation-segments"
    content = iter((body[:8 * 1024 * 1024], body[8 * 1024 * 1024:])) if streamed else body
    result = client.post(route, content=content,
        headers={"x-lab-arena-lease": "a" * 64, "authorization": "never-forward"})
    assert result.status_code == 200 and calls == ["POST"]
    overflow = iter((body, b"x")) if streamed else body + b"x"
    assert client.post(route, content=overflow).status_code == 413
    # The route exception must not expand generic or nearby route limits.
    generic_overflow = b"x" * (arena_proxy._MAX_REQUEST_BYTES + 1)
    assert client.post(prefix + "/v1/submissions", content=generic_overflow).status_code == 413
    assert client.post(route + "/extra", content=generic_overflow).status_code == 413
    assert client.post(route, content=b"{}", headers={"content-encoding": "gzip"}).status_code == 415
    assert calls == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
async def test_edge_segment_declared_overflow_rejected_before_body_read(monkeypatch, prefix):
    async def unread():
        pytest.fail("declared segment overflow must precede body consumption")

    async def no_forward(*args, **kwargs):
        pytest.fail("declared segment overflow must not contact the sidecar")

    messages = []
    async def send(message):
        messages.append(message)

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", no_forward)
    await create_app()({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": prefix + "/v1/runs/r1/training-observation-segments", "query_string": b"",
        "headers": [(b"content-length", str(arena_proxy.training_observations.MAX_SEGMENT_BYTES + 1).encode())],
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, unread, send)
    assert [message["status"] for message in messages if message["type"] == "http.response.start"] == [413]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/arena", "/testnet/arena"])
@pytest.mark.parametrize("case,status", [("missing", 401), ("malformed", 401),
    ("oversize", 413), ("expired", 409), ("transport", 503), ("unexpected_success", 503)])
async def test_edge_observation_auth_and_declared_size_fail_before_body_read(monkeypatch, prefix, case, status):
    calls = []
    headers = [] if case == "missing" else [(b"x-lab-arena-lease", b"a" * 64)]
    if case == "malformed":
        headers = [(b"x-lab-arena-lease", b"wrong")]
    if case == "oversize":
        headers.append((b"content-length", str(256 * 1024 * 1024 + 1).encode()))

    async def unread():
        pytest.fail("rejection must precede body consumption")

    messages = []
    async def send(message):
        messages.append(message)

    async def forward(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if case == "transport":
            raise httpx.ConnectError("private error")
        return httpx.Response(200 if case == "unexpected_success" else 409, json={"code": "lease_expired"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    await create_app()({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": prefix + "/v1/runs/r1/training-observations", "query_string": b"", "headers": headers,
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, unread, send)
    assert next(message["status"] for message in messages if message["type"] == "http.response.start") == status
    assert len(calls) == (0 if case in {"missing", "malformed", "oversize"} else 1)
    if calls:
        assert calls[0][0:2] == ("HEAD", "v1/runs/r1/training-observations")
        assert calls[0][2].get("testnet", False) == prefix.startswith("/testnet")


def test_edge_observation_stream_limit_still_rejects_overflow(monkeypatch):
    calls = []
    async def forward(method, path, **kwargs):
        calls.append(method)
        return httpx.Response(204)

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    monkeypatch.setattr(arena_proxy.training_observations, "MAX_OBJECT_BYTES", 16)
    client = TestClient(create_app(), raise_server_exceptions=False)
    result = client.post("/arena/v1/runs/r1/training-observations", content=iter((b"x" * 8, b"x" * 9)),
        headers={"x-lab-arena-lease": "a" * 64})
    assert result.status_code == 413 and calls == ["HEAD"]


def test_edge_forwards_arena_with_gateway_header_and_body_rules(monkeypatch):
    observed = {}

    async def forward(method, path, *, query, body, headers):
        observed.update(method=method, path=path, query=query, body=body, headers=dict(headers))
        return httpx.Response(
            202,
            json={"status": "accepted"},
            headers={"cache-control": "no-store", "x-private": "hidden"},
        )

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = TestClient(create_app())
    response = client.post(
        "/arena/v1/submissions/presign?item=1",
        content=b'{"body":1}',
        headers={
            "content-type": "application/json",
            "x-lab-arena-lease": "a" * 64,
            "authorization": "never-forward",
            "x-forwarded-for": "spoofed",
        },
    )
    assert response.status_code == 202
    assert response.headers["cache-control"] == "no-store"
    assert "x-private" not in response.headers
    assert observed == {
        "method": "POST",
        "path": "v1/submissions/presign",
        "query": "item=1",
        "body": b'{"body":1}',
        "headers": {"content-type": "application/json", "x-lab-arena-lease": "a" * 64},
    }
    assert client.post(
        "/arena/v1/submissions/presign",
        content=b"{}",
        headers={"content-encoding": "gzip"},
    ).status_code == 415
    assert client.post(
        "/arena/v1/submissions/presign",
        content=b"x" * (arena_proxy._MAX_REQUEST_BYTES + 1),
    ).status_code == 413


def test_edge_does_not_expose_gateway_or_testnet_benchmark(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")
    client = TestClient(create_app())
    assert client.get("/attestation/document").status_code == 404
    assert client.get("/health/v2-authority").status_code == 404
    assert client.get("/testnet/arena/v1/rounds/r/benchmark").status_code == 403


def test_edge_main_port_is_restricted_to_release_slots(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_PROXY_MAIN_PORT", "8794")
    # Production loads the module once per process. Test the same import-time
    # validation in a fresh interpreter so no test changes a live module.
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    ok = subprocess.run(
        [sys.executable, "-c", "from gateway.api.arena_proxy import _SIDECAR_URL; print(_SIDECAR_URL)"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert ok.stdout.strip() == "http://127.0.0.1:8794"
    env["LAB_ARENA_PROXY_MAIN_PORT"] = "8000"
    bad = subprocess.run(
        [sys.executable, "-c", "import gateway.api.arena_proxy"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "approved loopback port" in bad.stderr


def test_edge_environment_reads_only_routing_and_ingest_values(tmp_path, monkeypatch):
    environment = tmp_path / "gateway.env"
    environment.write_text(
        "LAB_ARENA_MODE=live\n"
        "LAB_ARENA_TESTNET_ENABLED=true\n"
        "GATEWAY_OTEL_ENABLED=true\n"
        "MINER_MAX_CONCURRENT=12\n"
        "LAB_ARENA_SERVICE_KEY=must-not-load\n"
        "LAB_ARENA_OPENROUTER_API_KEY=must-not-load\n",
        encoding="utf-8",
    )
    for key in (
        "LAB_ARENA_MODE", "LAB_ARENA_TESTNET_ENABLED", "GATEWAY_OTEL_ENABLED",
        "MINER_MAX_CONCURRENT",
        "LAB_ARENA_SERVICE_KEY", "LAB_ARENA_OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    load_edge_environment(environment)
    import os

    assert os.environ["LAB_ARENA_MODE"] == "live"
    assert os.environ["LAB_ARENA_TESTNET_ENABLED"] == "true"
    assert os.environ["GATEWAY_OTEL_ENABLED"] == "true"
    assert os.environ["MINER_MAX_CONCURRENT"] == "12"
    assert "LAB_ARENA_SERVICE_KEY" not in os.environ
    assert "LAB_ARENA_OPENROUTER_API_KEY" not in os.environ


@pytest.mark.parametrize("edge_port,api_port", [(8890, 8792), (8891, 8794)])
def test_edge_cli_binds_dedicated_slot_and_keeps_its_api_target(
    tmp_path, monkeypatch, edge_port, api_port,
):
    import sys
    from types import SimpleNamespace
    import uvicorn
    from scripts import run_arena_proxy_edge as edge

    environment = tmp_path / "gateway.env"
    environment.write_text("")
    sha = "a" * 40
    monkeypatch.setenv("LAB_ARENA_RELEASE_COMMIT", sha)
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setenv("LAB_ARENA_RELEASE_ROOT", str(edge.ROOT))
    monkeypatch.setenv("LAB_ARENA_PROXY_MAIN_PORT", "8792")
    monkeypatch.setattr(sys, "argv", [
        "arena-edge", "--port", str(edge_port), "--sidecar-port", str(api_port),
        "--environment-file", str(environment),
    ])
    monkeypatch.setitem(sys.modules, "leadpoet_observability", SimpleNamespace(
        init_sentry=lambda **_kwargs: None, configure_sentry_context=lambda **_kwargs: None,
    ))
    application = object()
    observed = {}
    monkeypatch.setattr(edge, "create_app", lambda: application)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: observed.update(app=app, **kwargs))

    assert edge.main() == 0
    assert observed["app"] is application
    assert observed["host"] == "127.0.0.1" and observed["port"] == edge_port
    assert edge.os.environ["LAB_ARENA_PROXY_MAIN_PORT"] == str(api_port)
    assert observed["forwarded_allow_ips"] == "127.0.0.1"


@pytest.mark.parametrize("old_port", [8800, 8801])
def test_edge_cli_rejects_ports_outside_dedicated_allocation(monkeypatch, old_port):
    import sys
    from scripts import run_arena_proxy_edge as edge

    monkeypatch.setattr(sys, "argv", [
        "arena-edge", "--port", str(old_port), "--sidecar-port", "8794",
        "--environment-file", "/unused/gateway.env",
    ])
    with pytest.raises(SystemExit) as error:
        edge.main()
    assert error.value.code == 2
