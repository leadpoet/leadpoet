from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from gateway.tee import proxy_transport_preflight_v2 as module
from gateway.tee.proxy_transport_preflight_v2 import (
    WorkerProxyTransportCleanupV2Error,
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

        def _retry_retired_cleanup(self):
            return True

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

        def _retry_retired_cleanup(self):
            return True

    monkeypatch.setattr(module, "_HostProxyProbe", Probe)
    verify_tls_proxy_connect_v2(
        "https://worker:secret@proxy.example.com",
        destination_host="api.exa.ai",
        attempts=2,
        sleep=sleeps.append,
    )

    assert attempts == ["attempt", "attempt"]
    assert sleeps == [0.2]


def test_connect_probe_accepts_authenticated_http_connect_proxy(monkeypatch):
    observed = []

    class Probe:
        def __init__(self, **_kwargs):
            pass

        def _open_upstream_proxy_tunnel(self, **kwargs):
            observed.append(kwargs)
            return _Tunnel()

        def _retry_retired_cleanup(self):
            return True

    monkeypatch.setattr(module, "_HostProxyProbe", Probe)
    verify_tls_proxy_connect_v2(
        "http://worker:secret@proxy.example.com:6162",
        destination_host="openrouter.ai",
    )

    assert observed == [
        {
            "proxy_url": "http://worker:secret@proxy.example.com:6162",
            "destination_host": "openrouter.ai",
            "destination_port": 443,
        }
    ]


def test_connect_probe_fails_closed_and_retains_unclosed_tunnel(monkeypatch):
    class Tunnel(_Tunnel):
        def __init__(self):
            super().__init__()
            self.allow_close = False

        def close(self):
            self.closed = self.allow_close
            return self.allow_close

    tunnel = Tunnel()

    class Probe:
        def __init__(self, **_kwargs):
            pass

        def _open_upstream_proxy_tunnel(self, **_kwargs):
            return tunnel

        def _retry_retired_cleanup(self):
            return True

    monkeypatch.setattr(module, "_HostProxyProbe", Probe)
    with pytest.raises(
        WorkerProxyTransportCleanupV2Error,
        match="transport cleanup failed",
    ) as raised:
        verify_tls_proxy_connect_v2(
            "https://worker:secret@proxy.example.com",
            destination_host="openrouter.ai",
            attempts=2,
        )

    assert raised.value.stage == "connect_transport_cleanup"
    assert module._RETIRED_CLEANUP_RESOURCES[id(tunnel)][1] is tunnel
    tunnel.allow_close = True
    verify_tls_proxy_connect_v2(
        "https://worker:secret@proxy.example.com",
        destination_host="openrouter.ai",
        attempts=1,
    )
    assert module._RETIRED_CLEANUP_RESOURCES == {}


def test_retired_preflight_cleanup_allows_concurrent_owner_transfer(
    monkeypatch,
):
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingTunnel:
        def shutdown(self, _how):
            return None

        def close(self):
            close_started.set()
            assert release_close.wait(timeout=1)
            return None

    class ConcurrentTunnel:
        def shutdown(self, _how):
            return None

        def close(self):
            return None

    first = BlockingTunnel()
    concurrent = ConcurrentTunnel()
    concurrent_primary = OSError("concurrent cleanup")
    monkeypatch.setattr(module, "_RETIRED_CLEANUP_RESOURCES", {})
    module._retain_cleanup_resource(
        first,
        kind="transport",
        primary_error=OSError("first cleanup"),
    )
    recovery_results = []
    recovery = threading.Thread(
        target=lambda: recovery_results.append(module._retry_retired_cleanup())
    )
    recovery.start()
    assert close_started.wait(timeout=1)
    retained = threading.Event()

    def retain():
        module._retain_cleanup_resource(
            concurrent,
            kind="transport",
            primary_error=concurrent_primary,
        )
        retained.set()

    retention = threading.Thread(target=retain)
    retention.start()
    assert retained.wait(timeout=1)
    assert module._RETIRED_CLEANUP_RESOURCES[id(concurrent)][1] is concurrent
    release_close.set()
    recovery.join(timeout=1)
    retention.join(timeout=1)

    assert recovery_results == [("transport", concurrent_primary)]
    assert list(module._RETIRED_CLEANUP_RESOURCES.values())[0][1] is concurrent
    assert module._retry_retired_cleanup() is None
    assert module._RETIRED_CLEANUP_RESOURCES == {}


def test_parent_timeout_failure_preserves_primary_when_close_is_unproven():
    primary = ValueError("timeout setup failed")

    class Connection:
        def __init__(self):
            self.allow_close = False

        def settimeout(self, _timeout):
            raise primary

        def close(self):
            return self.allow_close

    connection = Connection()
    probe = module._HostProxyProbe(
        connector=lambda _host, _port: connection,
        timeout_seconds=1,
    )
    with pytest.raises(module.EnclaveEgressProxyCleanupError) as raised:
        probe._open_parent_tunnel("proxy.example.com", 443, purpose="upstream_proxy")

    assert raised.value.primary_error is primary
    assert raised.value.__cause__ is primary
    assert raised.value._resources == (connection,)
    connection.allow_close = True
    assert probe._retry_retired_cleanup() is True


def test_default_connector_cleanup_failure_transfers_candidate_ownership():
    primary = OSError("connect failed")

    class Candidate:
        def __init__(self):
            self.allow_close = False

        def close(self):
            return self.allow_close

    candidate = Candidate()

    def connector(_host, _port):
        raise module.TEEEgressForwarderCleanupError(
            primary_error=primary,
            resource=candidate,
        ) from primary

    with pytest.raises(WorkerProxyTransportCleanupV2Error) as raised:
        verify_tls_proxy_connect_v2(
            "https://worker:secret@proxy.example.com",
            destination_host="openrouter.ai",
            attempts=1,
            connector=connector,
        )

    assert raised.value.primary_error.__cause__ is primary
    retained_probes = [
        resource
        for kind, resource, _primary_error in (
            module._RETIRED_CLEANUP_RESOURCES.values()
        )
        if kind == "probe"
    ]
    assert len(retained_probes) == 1
    assert retained_probes[0]._retired_cleanup_resources[id(candidate)][1] is candidate
    candidate.allow_close = True
    assert module._retry_retired_cleanup() is None


def test_fleet_probe_quarantines_failed_profile_after_all_destination_checks():
    observed = []

    def verify(proxy_url, *, destination_host, destination_port):
        observed.append((proxy_url, destination_host, destination_port))
        if proxy_url.endswith("bad.example.com"):
            raise WorkerProxyTransportPreflightV2Error("failed")

    verified = verify_worker_proxy_fleets_v2(
        {
            "gateway_scoring": (
                "https://score.example.com",
                "https://bad.example.com",
            ),
        },
        max_workers=3,
        verify_proxy=verify,
    )

    expected_destinations = {
        ("openrouter.ai", 443),
        ("api.exa.ai", 443),
        ("api.scrapingdog.com", 443),
        ("code.deepline.com", 443),
    }
    for proxy in (
        "https://score.example.com",
        "https://bad.example.com",
    ):
        assert {
            (host, port)
            for observed_proxy, host, port in observed
            if observed_proxy == proxy
        } == expected_destinations
    assert verified == {
        "gateway_scoring": ("https://score.example.com",),
    }


def test_fleet_probe_fails_closed_when_role_has_no_verified_profile():
    def verify(proxy_url, *, destination_host, destination_port):
        if proxy_url.endswith("bad.example.com"):
            raise WorkerProxyTransportPreflightV2Error("failed")

    with pytest.raises(
        WorkerProxyTransportPreflightV2Error,
        match=(
            "gateway_scoring worker proxy fleet has no verified profiles; "
            r"proxy 1 failed V2 TLS CONNECT preflight to api\.exa\.ai:443 "
            r"\(4/4 role probes failed\)"
        ),
    ):
            verify_worker_proxy_fleets_v2(
                {
                    "gateway_scoring": ("https://bad.example.com",),
                },
            max_workers=2,
            verify_proxy=verify,
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

        destinations = (
            "openrouter.ai",
            "api.exa.ai",
            "api.scrapingdog.com",
            "code.deepline.com",
        )
        for destination in destinations:
            verify_tls_proxy_connect_v2(
                "https://rehearsal-auto:rehearsal-auto-password@"
                "autoresearch-proxy.example.com:443",
                destination_host=destination,
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
        assert len(rows) == len(destinations)
        assert all(row["operation"] == "proxy_connect" for row in rows)
        assert {row["target"] for row in rows} == {
            destination + ":443"
            for destination in destinations
        }
        assert all(row["authenticated"] is True for row in rows)
        assert "password" not in json.dumps(rows).lower()
    finally:
        service.terminate()
        service.wait(timeout=5)
