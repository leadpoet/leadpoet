from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from gateway.tee import proxy_transport_preflight_v2 as module
from gateway.tee.provider_broker_v2 import ProviderBrokerV2Error
from gateway.tee.proxy_transport_preflight_v2 import (
    WorkerProxyTransportPreflightV2Error,
    verify_tls_proxy_connect_v2,
    verify_worker_proxy_fleets_v2,
)


class _Tunnel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_tls_connect_probe_uses_measured_handshake_and_closes_tunnel(monkeypatch):
    observed = []
    tunnel = _Tunnel()

    class Probe:
        def __init__(self, *, connector, timeout_seconds):
            observed.append(("init", connector, timeout_seconds))

        def _open_upstream_proxy_tunnel(self, **kwargs):
            observed.append(("connect", kwargs))
            return tunnel

    monkeypatch.setattr(module, "_HostProxyProbe", Probe)
    connector = object()
    verify_tls_proxy_connect_v2(
        "https://worker:secret@proxy.example.com:443",
        destination_host="openrouter.ai",
        connector=connector,
        attempts=2,
        timeout_seconds=3.5,
    )

    assert observed == [
        ("init", connector, 3.5),
        (
            "connect",
            {
                "proxy_url": "https://worker:secret@proxy.example.com:443",
                "destination_host": "openrouter.ai",
                "destination_port": 443,
            },
        ),
    ]
    assert tunnel.closed


def test_tls_connect_probe_retries_then_succeeds_without_exposing_url(monkeypatch):
    attempts = []
    sleeps = []

    class Probe:
        def __init__(self, **_kwargs):
            pass

        def _open_upstream_proxy_tunnel(self, **_kwargs):
            attempts.append("attempt")
            if len(attempts) == 1:
                raise OSError("connection reset")
            return _Tunnel()

    monkeypatch.setattr(module, "_HostProxyProbe", Probe)
    verify_tls_proxy_connect_v2(
        "https://worker:secret@proxy.example.com",
        destination_host="api.exa.ai",
        attempts=2,
        sleep=sleeps.append,
    )

    assert attempts == ["attempt", "attempt"]
    assert sleeps == [0.2]


def test_tls_connect_probe_rejects_plaintext_before_opening_socket(monkeypatch):
    monkeypatch.setattr(
        module,
        "_HostProxyProbe",
        lambda **_kwargs: pytest.fail("plaintext proxy reached transport"),
    )
    with pytest.raises(
        ProviderBrokerV2Error,
        match="HTTPS proxy on port 443",
    ):
        verify_tls_proxy_connect_v2(
            "http://worker:secret@proxy.example.com:6162",
            destination_host="openrouter.ai",
        )


def test_fleet_probe_checks_role_specific_destinations_and_reports_index():
    observed = []

    def verify(proxy_url, *, destination_host, destination_port):
        observed.append((proxy_url, destination_host, destination_port))
        if proxy_url.endswith("bad.example.com"):
            raise WorkerProxyTransportPreflightV2Error("failed")

    with pytest.raises(
        WorkerProxyTransportPreflightV2Error,
        match=(
            "gateway_scoring worker proxy 2 failed V2 TLS CONNECT preflight "
            r"\(1/3 failed\)"
        ),
    ):
        verify_worker_proxy_fleets_v2(
            {
                "gateway_autoresearch": ("https://auto.example.com",),
                "gateway_scoring": (
                    "https://score.example.com",
                    "https://bad.example.com",
                ),
            },
            max_workers=3,
            verify_proxy=verify,
        )

    assert sorted(observed) == sorted(
        [
            ("https://auto.example.com", "openrouter.ai", 443),
            ("https://score.example.com", "api.exa.ai", 443),
            ("https://bad.example.com", "api.exa.ai", 443),
        ]
    )


def test_rehearsal_tls_service_exercises_verified_authenticated_connect(
    tmp_path,
    monkeypatch,
):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    environment = {
        **os.environ,
        "REHEARSAL_STATE_ROOT": str(tmp_path),
        "REHEARSAL_TLS_CONNECT_PROXY_PORT": str(port),
    }
    service = subprocess.Popen(
        [
            sys.executable,
            "tests/restart_rehearsal/tls_connect_proxy_service.py",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        ready = tmp_path / "tls-connect-proxy.ready"
        for _attempt in range(100):
            if ready.is_file():
                break
            if service.poll() is not None:
                stdout, stderr = service.communicate(timeout=1)
                pytest.fail(
                    "TLS CONNECT rehearsal service exited early: "
                    f"{stdout!r} {stderr!r}"
                )
            time.sleep(0.02)
        assert ready.is_file()

        import certifi

        monkeypatch.setattr(
            certifi,
            "where",
            lambda: str(tmp_path / "tls-connect-proxy-ca.pem"),
        )

        def connector(_host, _port):
            return socket.create_connection(("127.0.0.1", port), timeout=2)

        verify_tls_proxy_connect_v2(
            "https://rehearsal-auto:rehearsal-auto-password@"
            "autoresearch-proxy.example.com:443",
            destination_host="openrouter.ai",
            attempts=1,
            timeout_seconds=2,
            connector=connector,
        )
        with pytest.raises(
            WorkerProxyTransportPreflightV2Error,
            match="authenticated CONNECT preflight",
        ):
            verify_tls_proxy_connect_v2(
                "https://rehearsal-auto:wrong@"
                "autoresearch-proxy.example.com:443",
                destination_host="openrouter.ai",
                attempts=1,
                timeout_seconds=2,
                connector=connector,
            )

        rows = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["operation"] == "proxy_connect"
        assert rows[0]["target"] == "openrouter.ai:443"
        assert rows[0]["authenticated"] is True
        assert "password" not in json.dumps(rows).lower()
    finally:
        service.terminate()
        service.wait(timeout=5)
