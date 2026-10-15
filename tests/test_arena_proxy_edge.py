"""The stable Arena edge keeps the gateway proxy's public boundaries."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.api import arena_proxy
from scripts.run_arena_proxy_edge import create_app, load_edge_environment


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
