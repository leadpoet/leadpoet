from __future__ import annotations

import ast
import asyncio
import base64
import json
from pathlib import Path
import socket
import threading

import pytest

from gateway.tee.egress_policy import (
    EgressPolicyError,
    destination_policy_hash,
    normalize_destination,
)
from gateway.tee import egress_proxy
from gateway.tee.egress_proxy import EnclaveEgressProxy, _parse_proxy_request
from gateway.utils.tee_client import _recv_exact
from gateway.utils.tee_egress_forwarder import (
    TEEEgressForwarderError,
    _global_address_infos,
    _handle_connection,
    _relay_bidirectional,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame(value: dict) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return len(encoded).to_bytes(4, "big") + encoded


def _read_frame(connection: socket.socket) -> dict:
    prefix = _recv_exact(connection, 4)
    assert len(prefix) == 4
    body = _recv_exact(connection, int.from_bytes(prefix, "big"))
    return json.loads(body.decode("ascii"))


def test_destination_policy_allows_public_dns_https_only():
    assert normalize_destination("API.OpenRouter.AI.", 443) == ("api.openrouter.ai", 443)
    assert destination_policy_hash().startswith("sha256:")


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("127.0.0.1", 443),
        ("169.254.169.254", 80),
        ("localhost", 443),
        ("service.internal", 443),
        ("example.com", 22),
        ("example.com", 80),
        ("user@example.com", 443),
    ],
)
def test_destination_policy_rejects_local_literal_and_non_http_destinations(host, port):
    with pytest.raises(EgressPolicyError):
        normalize_destination(host, port)


def test_parent_dns_gate_rejects_any_non_global_answer():
    def resolver(_host, port, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    with pytest.raises(TEEEgressForwarderError, match="non-global"):
        _global_address_infos("example.com", 443, resolver=resolver)


def test_parent_forwarder_uses_bounded_handshake_then_relays_opaque_bytes():
    client, parent = socket.socketpair()
    upstream, origin = socket.socketpair()
    called = []

    def connector(host, port):
        called.append((host, port))
        return upstream

    thread = threading.Thread(
        target=_handle_connection,
        kwargs={"connection": parent, "connector": connector, "idle_timeout_seconds": 2.0},
        daemon=True,
    )
    thread.start()
    try:
        client.sendall(
            _frame(
                {
                    "method": "connect",
                    "params": {
                        "host": "api.openrouter.ai",
                        "port": 443,
                        "policy_hash": destination_policy_hash(),
                    },
                }
            )
        )
        response = _read_frame(client)
        assert response == {
            "result": {
                "policy_hash": destination_policy_hash(),
                "status": "connected",
            }
        }
        assert called == [("api.openrouter.ai", 443)]

        client.sendall(b"opaque-tls-request")
        assert origin.recv(64) == b"opaque-tls-request"
        origin.sendall(b"opaque-tls-response")
        assert client.recv(64) == b"opaque-tls-response"
    finally:
        client.close()
        origin.close()
        thread.join(timeout=2)


def test_parent_forwarder_rejects_policy_mismatch_before_connecting():
    client, parent = socket.socketpair()
    called = []

    def connector(host, port):
        called.append((host, port))
        raise AssertionError("must not connect")

    thread = threading.Thread(
        target=_handle_connection,
        kwargs={"connection": parent, "connector": connector},
        daemon=True,
    )
    thread.start()
    try:
        client.sendall(
            _frame(
                {
                    "method": "connect",
                    "params": {
                        "host": "api.openrouter.ai",
                        "port": 443,
                        "policy_hash": "sha256:" + "0" * 64,
                    },
                }
            )
        )
        response = _read_frame(client)
        assert response["status"] == "error"
        assert called == []
    finally:
        client.close()
        thread.join(timeout=2)


def test_parent_relay_reports_directional_bytes_and_first_close():
    enclave, parent = socket.socketpair()
    upstream, provider = socket.socketpair()
    observed = {}

    def run():
        observed.update(
            _relay_bidirectional(
                parent,
                upstream,
                idle_timeout_seconds=2,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    enclave.sendall(b"request")
    assert provider.recv(64) == b"request"
    provider.sendall(b"response")
    provider.shutdown(socket.SHUT_WR)
    assert enclave.recv(64) == b"response"
    enclave.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert observed == {
        "enclave_to_provider_bytes": 7,
        "provider_to_enclave_bytes": 8,
        "first_closed": "provider",
    }
    enclave.close()
    parent.close()
    upstream.close()
    provider.close()


def test_enclave_proxy_parses_connect_without_exposing_http_payload():
    parsed = _parse_proxy_request(
        b"CONNECT api.exa.ai:443 HTTP/1.1\r\nHost: api.exa.ai:443\r\n\r\n"
    )
    assert parsed == {
        "method": "CONNECT",
        "host": "api.exa.ai",
        "port": 443,
        "forward_headers": b"",
        "tls_protected": True,
    }


def test_enclave_proxy_accepts_upstream_proxy_only_as_loopback_control_metadata():
    proxy_url = "https://worker-7:password@proxy.example.com:443"
    encoded = base64.b64encode(proxy_url.encode("utf-8"))
    parsed = _parse_proxy_request(
        b"CONNECT api.exa.ai:443 HTTP/1.1\r\n"
        b"Host: api.exa.ai:443\r\n"
        b"X-Leadpoet-Upstream-Proxy-B64: " + encoded + b"\r\n\r\n"
    )

    assert parsed["host"] == "api.exa.ai"
    assert parsed["upstream_proxy_url"] == proxy_url
    assert parsed["forward_headers"] == b""


def test_enclave_proxy_rejects_external_plaintext_http():
    with pytest.raises(egress_proxy.EnclaveEgressProxyError, match="forbidden"):
        _parse_proxy_request(
            b"GET http://archive.org/wayback/available?url=x HTTP/1.1\r\n"
            b"Host: archive.org\r\nProxy-Authorization: Basic secret\r\n\r\n"
        )


def test_enclave_parent_handshake_uses_same_length_prefixed_json_contract():
    enclave, parent = socket.socketpair()

    class _VsockAdapter:
        def __init__(self, inner):
            self._inner = inner

        def connect(self, _address):
            return None

        def sendall(self, data):
            return self._inner.sendall(data)

        def recv(self, size):
            return self._inner.recv(size)

        def close(self):
            return self._inner.close()

    adapted_enclave = _VsockAdapter(enclave)

    def socket_factory(*_args):
        return adapted_enclave

    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        socket_factory=socket_factory,
        idle_timeout_seconds=2,
    )

    def parent_side():
        request = _read_frame(parent)
        assert request["method"] == "connect"
        assert request["params"] == {
            "host": "api.scrapingdog.com",
            "port": 443,
            "policy_hash": destination_policy_hash(),
        }
        parent.sendall(
            _frame(
                {
                    "result": {
                        "status": "connected",
                        "policy_hash": destination_policy_hash(),
                    }
                }
            )
        )

    thread = threading.Thread(target=parent_side, daemon=True)
    thread.start()
    tunnel = proxy._open_parent_tunnel("api.scrapingdog.com", 443)
    assert tunnel is adapted_enclave
    thread.join(timeout=2)
    enclave.close()
    parent.close()


def test_enclave_proxy_health_exposes_bounded_last_failure_stage():
    client, proxy_side = socket.socketpair()
    proxy = EnclaveEgressProxy(recv_exact=_recv_exact)

    def fail_parent(_host, _port):
        error = OSError(111, "connection refused")
        raise error

    proxy._open_parent_tunnel = fail_parent
    client.sendall(
        b"CONNECT qplwoislplkcegvdmbim.supabase.co:443 HTTP/1.1\r\n"
        b"Host: qplwoislplkcegvdmbim.supabase.co:443\r\n\r\n"
    )
    proxy._handle_client(proxy_side)
    response = client.recv(4096)
    status = proxy.status()

    assert response.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert status["last_failure"] == {
        "stage": "open_parent_tunnel",
        "error_type": "OSError",
        "errno": 111,
        "destination_ref": "4877532cd3300944",
    }
    assert "supabase" not in str(status)
    client.close()


def test_aiohttp_requests_are_forced_through_enclave_local_proxy(monkeypatch):
    import aiohttp

    observed = {}

    async def original_request(_session, method, url, *args, **kwargs):
        observed.update({"method": method, "url": url, "kwargs": kwargs})
        return "response"

    monkeypatch.setattr(aiohttp.ClientSession, "_request", original_request)
    monkeypatch.setattr(egress_proxy, "_AIOHTTP_ORIGINAL_REQUEST", None)
    monkeypatch.setattr(egress_proxy, "_AIOHTTP_PROXY_URL", "")

    proxy_url = "http://127.0.0.1:18080"
    egress_proxy._install_aiohttp_proxy(proxy_url)
    result = asyncio.run(
        aiohttp.ClientSession._request(object(), "GET", "https://api.exa.ai/search")
    )

    assert result == "response"
    assert observed == {
        "method": "GET",
        "url": "https://api.exa.ai/search",
        "kwargs": {"proxy": proxy_url},
    }


def test_enclave_egress_modules_parse_under_gateway_python39_image():
    for relative in (
        "gateway/tee/egress_policy.py",
        "gateway/tee/egress_proxy.py",
    ):
        path = ROOT / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
