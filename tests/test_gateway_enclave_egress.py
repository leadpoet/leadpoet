from __future__ import annotations

import ast
import asyncio
import base64
from datetime import datetime, timedelta, timezone
import errno
import http.client
import json
from pathlib import Path
import socket
import ssl
import threading
import time

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gateway.tee.egress_framing import (
    EgressTunnelFramingError,
    TUNNEL_FRAME_BYTES,
    TUNNEL_FRAMING_HEADER,
    TUNNEL_FRAMING_MODE,
    relay_raw_and_framed,
    send_tunnel_frame,
)
from gateway.tee.egress_policy import (
    EgressPolicyError,
    destination_policy_hash,
    normalize_destination,
    normalize_proxy_destination,
    policy_document,
)
from gateway.tee.provider_broker_v2 import HTTPXProviderTransport
from gateway.tee import egress_proxy
from gateway.tee.egress_proxy import (
    IFF_UP,
    SIOCGIFFLAGS,
    SIOCSIFFLAGS,
    EnclaveEgressProxy,
    EnclaveEgressProxyError,
    _ensure_loopback_interface,
    _parse_proxy_request,
    _relay_bidirectional as _relay_enclave_bidirectional,
)
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


def _write_test_server_identity(tmp_path: Path) -> tuple[Path, Path]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.com")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("example.com")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_path = tmp_path / "server-cert.pem"
    private_key_path = tmp_path / "server-key.pem"
    certificate_path.write_bytes(
        certificate.public_bytes(serialization.Encoding.PEM)
    )
    private_key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, private_key_path


def _unused_loopback_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
    finally:
        listener.close()


def test_destination_policy_allows_public_dns_https_only():
    assert normalize_destination("API.OpenRouter.AI.", 443) == ("api.openrouter.ai", 443)
    assert destination_policy_hash().startswith("sha256:")
    assert policy_document()["framed_tunnel_modes"] == [TUNNEL_FRAMING_MODE]
    assert policy_document()["framed_tunnel_requires_enclave_opt_in"] is True


def test_destination_policy_allows_proxy_port_without_relaxing_provider_port():
    assert normalize_proxy_destination("Proxy.Example.Com.", 6162) == (
        "proxy.example.com",
        6162,
    )
    assert normalize_proxy_destination("8.8.8.8", 6162) == (
        "8.8.8.8",
        6162,
    )
    with pytest.raises(EgressPolicyError, match="port is blocked"):
        normalize_destination("proxy.example.com", 6162)
    with pytest.raises(EgressPolicyError, match="IP literal"):
        normalize_destination("8.8.8.8", 443)


@pytest.mark.parametrize(
    "host",
    (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
    ),
)
def test_destination_policy_rejects_non_global_proxy_ip_literals(host):
    with pytest.raises(EgressPolicyError, match="not globally routable"):
        normalize_proxy_destination(host, 6162)


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


def test_parent_forwarder_accepts_nonstandard_port_only_for_proxy_purpose():
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
                        "host": "proxy.example.com",
                        "port": 6162,
                        "policy_hash": destination_policy_hash(),
                        "purpose": "upstream_proxy",
                    },
                }
            )
        )
        response = _read_frame(client)
        assert response["result"]["status"] == "connected"
        assert called == [("proxy.example.com", 6162)]
    finally:
        client.close()
        origin.close()
        thread.join(timeout=2)


def test_parent_forwarder_accepts_global_ip_only_for_proxy_purpose():
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
                        "host": "8.8.8.8",
                        "port": 6162,
                        "policy_hash": destination_policy_hash(),
                        "purpose": "upstream_proxy",
                    },
                }
            )
        )
        response = _read_frame(client)
        assert response["result"]["status"] == "connected"
        assert called == [("8.8.8.8", 6162)]
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


@pytest.mark.parametrize(
    "extra_params",
    (
        {"tunnel_framing": ""},
        {"tunnel_framing": "length-v2"},
        {"tunnel_framing": TUNNEL_FRAMING_MODE, "purpose": "upstream_proxy"},
    ),
)
def test_parent_forwarder_rejects_noncanonical_tunnel_framing(extra_params):
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
                        "policy_hash": destination_policy_hash(),
                        **extra_params,
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
    assert enclave.recv(1) == b""
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


def test_framed_relay_preserves_large_bidirectional_payload_and_explicit_eof():
    client, enclave_raw = socket.socketpair()
    enclave_framed, parent_framed = socket.socketpair()
    parent_raw, provider = socket.socketpair()
    request = b"r" * (TUNNEL_FRAME_BYTES * 2 + 137)
    response = b"s" * (TUNNEL_FRAME_BYTES + 271)
    observed = {}
    errors = []

    def run(name, *args, **kwargs):
        try:
            observed[name] = relay_raw_and_framed(*args, **kwargs)
        except Exception as exc:
            errors.append(exc)

    enclave_thread = threading.Thread(
        target=run,
        args=("enclave", enclave_raw, enclave_framed),
        kwargs={
            "idle_timeout_seconds": 2,
            "max_bytes_per_direction": len(request) + len(response),
            "raw_label": "client",
            "framed_label": "parent",
        },
        daemon=True,
    )
    parent_thread = threading.Thread(
        target=run,
        args=("parent", parent_raw, parent_framed),
        kwargs={
            "idle_timeout_seconds": 2,
            "max_bytes_per_direction": len(request) + len(response),
            "raw_label": "provider",
            "framed_label": "enclave",
        },
        daemon=True,
    )
    enclave_thread.start()
    parent_thread.start()
    try:
        request_thread = threading.Thread(
            target=client.sendall,
            args=(request,),
            daemon=True,
        )
        request_thread.start()
        assert _recv_exact(provider, len(request)) == request
        request_thread.join(timeout=2)
        assert not request_thread.is_alive()
        response_thread = threading.Thread(
            target=provider.sendall,
            args=(response,),
            daemon=True,
        )
        response_thread.start()
        assert _recv_exact(client, len(response)) == response
        response_thread.join(timeout=2)
        assert not response_thread.is_alive()
        provider.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
        client.shutdown(socket.SHUT_WR)
        enclave_thread.join(timeout=2)
        parent_thread.join(timeout=2)
    finally:
        for connection in (
            client,
            enclave_raw,
            enclave_framed,
            parent_framed,
            parent_raw,
            provider,
        ):
            connection.close()

    assert not enclave_thread.is_alive()
    assert not parent_thread.is_alive()
    assert errors == []
    assert observed["enclave"]["client_to_parent_bytes"] == len(request)
    assert observed["enclave"]["parent_to_client_bytes"] == len(response)
    assert observed["parent"]["enclave_to_provider_bytes"] == len(request)
    assert observed["parent"]["provider_to_enclave_bytes"] == len(response)


def test_framed_relay_fails_closed_without_forwarding_truncated_frame():
    client, raw = socket.socketpair()
    framed, peer = socket.socketpair()
    errors = []

    def run():
        try:
            relay_raw_and_framed(
                raw,
                framed,
                idle_timeout_seconds=2,
                max_bytes_per_direction=1024,
                raw_label="client",
                framed_label="parent",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    peer.sendall((10).to_bytes(4, "big") + b"short")
    peer.close()
    thread.join(timeout=2)
    client.settimeout(0.1)
    try:
        with pytest.raises(socket.timeout):
            client.recv(1)
    finally:
        client.close()
        raw.close()
        framed.close()

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], EgressTunnelFramingError)
    assert "explicit EOF frame" in str(errors[0])


def test_parent_relay_tolerates_late_tls_write_after_provider_full_close():
    enclave, parent = socket.socketpair()
    upstream, provider = socket.socketpair()
    observed = {}
    errors = []

    def run():
        try:
            observed.update(
                _relay_bidirectional(
                    parent,
                    upstream,
                    idle_timeout_seconds=2,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    enclave.sendall(b"request")
    assert provider.recv(64) == b"request"
    provider.sendall(b"complete-response")
    provider.close()
    assert enclave.recv(64) == b"complete-response"
    assert enclave.recv(1) == b""
    enclave.sendall(b"late-tls-close")
    enclave.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert observed == {
        "enclave_to_provider_bytes": 7,
        "provider_to_enclave_bytes": 17,
        "first_closed": "provider",
        "write_closed": "provider",
    }
    enclave.close()
    parent.close()
    upstream.close()


@pytest.mark.parametrize(
    ("relay", "expected_first"),
    (
        (_relay_bidirectional, "provider"),
        (_relay_enclave_bidirectional, "parent"),
    ),
)
def test_relays_treat_peer_close_recv_error_as_directional_eof(
    relay,
    expected_first,
):
    left_peer, left = socket.socketpair()
    right, right_peer = socket.socketpair()

    class _PeerCloseRecv:
        def fileno(self):
            return right.fileno()

        def recv(self, _size):
            raise OSError(errno.ENOTCONN, "peer is no longer connected")

        def sendall(self, data):
            return right.sendall(data)

        def shutdown(self, how):
            return right.shutdown(how)

    wrapped_right = _PeerCloseRecv()
    observed = {}
    errors = []

    def run():
        try:
            observed.update(
                relay(
                    left,
                    wrapped_right,
                    idle_timeout_seconds=2,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    right_peer.close()
    assert left_peer.recv(1) == b""
    left_peer.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert observed["first_closed"] == expected_first
    left_peer.close()
    left.close()
    right.close()


def test_parent_relay_does_not_turn_incomplete_provider_body_into_success():
    enclave, parent = socket.socketpair()
    upstream, provider = socket.socketpair()
    errors = []

    def run():
        try:
            _relay_bidirectional(
                parent,
                upstream,
                idle_timeout_seconds=2,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    enclave.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    assert provider.recv(128).startswith(b"GET / HTTP/1.1")
    provider.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort"
    )
    provider.close()

    response = http.client.HTTPResponse(enclave)
    response.begin()
    with pytest.raises(http.client.IncompleteRead):
        response.read()
    enclave.sendall(b"late-tls-close")
    enclave.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    enclave.close()
    parent.close()
    upstream.close()


def test_enclave_relay_tolerates_late_client_write_after_parent_full_close():
    client, proxy_side = socket.socketpair()
    parent_side, upstream = socket.socketpair()
    observed = {}
    errors = []

    def run():
        try:
            observed.update(
                _relay_enclave_bidirectional(
                    proxy_side,
                    parent_side,
                    idle_timeout_seconds=2,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client.sendall(b"request")
    assert upstream.recv(64) == b"request"
    upstream.sendall(b"complete-response")
    upstream.close()
    assert client.recv(64) == b"complete-response"
    assert client.recv(1) == b""
    client.sendall(b"late-tls-close")
    client.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert observed == {
        "client_to_parent_bytes": 7,
        "parent_to_client_bytes": 17,
        "first_closed": "parent",
        "write_closed": "parent",
    }
    client.close()
    proxy_side.close()
    parent_side.close()


def test_httpx_tls_transport_uses_explicit_framing_through_both_relays(
    tmp_path: Path,
):
    certificate_path, private_key_path = _write_test_server_identity(tmp_path)
    tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls_context.load_cert_chain(
        certfile=str(certificate_path),
        keyfile=str(private_key_path),
    )
    origin_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    origin_listener.bind(("127.0.0.1", 0))
    origin_listener.listen(1)
    origin_address = origin_listener.getsockname()
    origin_errors = []

    def serve_origin():
        try:
            connection, _address = origin_listener.accept()
            protected = tls_context.wrap_socket(connection, server_side=True)
            request = bytearray()
            while b"\r\n\r\n" not in request:
                request.extend(protected.recv(4096))
            body = b'{"ok":true}'
            protected.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 11\r\n"
                b"Connection: close\r\n\r\n"
                + body
            )
            # Do not perform TLS unwrap: reproduce a provider that closes its
            # complete response before the client emits its final TLS bytes.
            protected.close()
        except Exception as exc:
            origin_errors.append(exc)
        finally:
            origin_listener.close()

    origin_thread = threading.Thread(target=serve_origin, daemon=True)
    origin_thread.start()
    host_threads = []

    def open_parent_tunnel(_host, _port, *, tunnel_framing=""):
        assert tunnel_framing == TUNNEL_FRAMING_MODE
        enclave_side, host_side = socket.socketpair()
        host_thread = threading.Thread(
            target=_handle_connection,
            kwargs={
                "connection": host_side,
                "connector": lambda _name, _number: socket.create_connection(
                    origin_address,
                    timeout=2,
                ),
                "idle_timeout_seconds": 2,
            },
            daemon=True,
        )
        host_thread.start()
        host_threads.append(host_thread)
        request = _frame(
            {
                "method": "connect",
                "params": {
                    "host": "example.com",
                    "port": 443,
                    "policy_hash": destination_policy_hash(),
                    "tunnel_framing": TUNNEL_FRAMING_MODE,
                },
            }
        )
        enclave_side.sendall(request)
        response = _read_frame(enclave_side)
        assert response["result"]["status"] == "connected"
        assert response["result"]["tunnel_framing"] == TUNNEL_FRAMING_MODE
        return enclave_side

    proxy_port = _unused_loopback_port()
    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        local_port=proxy_port,
        loopback_initializer=lambda: None,
        idle_timeout_seconds=2,
    )
    proxy._open_parent_tunnel = open_parent_tunnel
    proxy._configure_environment = lambda: None
    proxy.start()
    try:
        result = HTTPXProviderTransport(
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ca_bundle=str(certificate_path),
            parent_tunnel_framing=TUNNEL_FRAMING_MODE,
        )(
            method="GET",
            url="https://example.com/artifact",
            headers={"accept": "application/json"},
            body=b"",
            timeout_ms=3000,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not proxy.status().get("last_tunnel"):
            time.sleep(0.01)
    finally:
        proxy.stop()
        origin_thread.join(timeout=2)
        for thread in host_threads:
            thread.join(timeout=2)

    assert origin_errors == []
    assert result["http_status"] == 200
    assert result["body"] == b'{"ok":true}'
    assert result["tls_protocol"].startswith("TLSv1.")
    assert proxy.status().get("last_failure") is None
    assert proxy.status()["last_tunnel"]["parent_to_client_bytes"] > 0


def test_httpx_framed_transport_reuses_one_tunnel_under_sustained_load(
    tmp_path: Path,
):
    request_count = 64
    certificate_path, private_key_path = _write_test_server_identity(tmp_path)
    tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls_context.load_cert_chain(
        certfile=str(certificate_path),
        keyfile=str(private_key_path),
    )
    origin_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    origin_listener.bind(("127.0.0.1", 0))
    origin_listener.listen(1)
    origin_address = origin_listener.getsockname()
    origin_connections = []
    requests_seen = []
    origin_errors = []

    def serve_origin():
        try:
            connection, _address = origin_listener.accept()
            origin_connections.append(connection)
            protected = tls_context.wrap_socket(connection, server_side=True)
            try:
                for index in range(request_count):
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        chunk = protected.recv(4096)
                        if not chunk:
                            raise AssertionError("artifact client closed early")
                        request.extend(chunk)
                    requests_seen.append(bytes(request))
                    connection_header = (
                        b"close" if index == request_count - 1 else b"keep-alive"
                    )
                    body = ('{"sequence":%d}' % index).encode("ascii")
                    protected.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: "
                        + str(len(body)).encode("ascii")
                        + b"\r\nConnection: "
                        + connection_header
                        + b"\r\n\r\n"
                        + body
                    )
            finally:
                protected.close()
        except Exception as exc:
            origin_errors.append(exc)
        finally:
            origin_listener.close()

    origin_thread = threading.Thread(target=serve_origin, daemon=True)
    origin_thread.start()
    host_threads = []
    opened_parent_tunnels = []

    def open_parent_tunnel(_host, _port, *, tunnel_framing=""):
        assert tunnel_framing == TUNNEL_FRAMING_MODE
        opened_parent_tunnels.append((_host, _port))
        enclave_side, host_side = socket.socketpair()
        host_thread = threading.Thread(
            target=_handle_connection,
            kwargs={
                "connection": host_side,
                "connector": lambda _name, _number: socket.create_connection(
                    origin_address,
                    timeout=2,
                ),
                "idle_timeout_seconds": 3,
            },
            daemon=True,
        )
        host_thread.start()
        host_threads.append(host_thread)
        enclave_side.sendall(
            _frame(
                {
                    "method": "connect",
                    "params": {
                        "host": "example.com",
                        "port": 443,
                        "policy_hash": destination_policy_hash(),
                        "tunnel_framing": TUNNEL_FRAMING_MODE,
                    },
                }
            )
        )
        response = _read_frame(enclave_side)
        assert response["result"]["tunnel_framing"] == TUNNEL_FRAMING_MODE
        return enclave_side

    proxy_port = _unused_loopback_port()
    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        local_port=proxy_port,
        loopback_initializer=lambda: None,
        idle_timeout_seconds=3,
    )
    proxy._open_parent_tunnel = open_parent_tunnel
    proxy._configure_environment = lambda: None
    proxy.start()
    transport = HTTPXProviderTransport(
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        ca_bundle=str(certificate_path),
        response_body_ceiling_bytes=1024,
        allow_authenticated_complete_body_eof=True,
        parent_tunnel_framing=TUNNEL_FRAMING_MODE,
    )
    try:
        results = [
            transport(
                method="GET",
                url="https://example.com/artifact",
                headers={"accept": "application/json"},
                body=b"",
                timeout_ms=3000,
                max_response_bytes=1024,
            )
            for _ in range(request_count)
        ]
    finally:
        transport.close()
        proxy.stop()
        origin_thread.join(timeout=3)
        for thread in host_threads:
            thread.join(timeout=3)

    assert origin_errors == []
    assert len(origin_connections) == 1
    assert len(opened_parent_tunnels) == 1
    assert len(requests_seen) == request_count
    assert all(b"Connection: close" not in request for request in requests_seen)
    assert [result["body"] for result in results] == [
        ('{"sequence":%d}' % index).encode("ascii")
        for index in range(request_count)
    ]
    assert proxy.status().get("last_failure") is None


def test_httpx_direct_transport_reuses_and_recovers_enclave_tunnel(tmp_path: Path):
    certificate_path, private_key_path = _write_test_server_identity(tmp_path)
    tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls_context.load_cert_chain(
        certfile=str(certificate_path),
        keyfile=str(private_key_path),
    )
    origin_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    origin_listener.settimeout(3)
    origin_listener.bind(("127.0.0.1", 0))
    origin_listener.listen(2)
    origin_address = origin_listener.getsockname()
    origin_errors = []
    origin_connections = []
    requests_seen = []

    def serve_origin():
        try:
            while len(requests_seen) < 3:
                connection, _address = origin_listener.accept()
                origin_connections.append(connection)
                protected = tls_context.wrap_socket(connection, server_side=True)
                try:
                    while len(requests_seen) < 3:
                        request = bytearray()
                        while b"\r\n\r\n" not in request:
                            chunk = protected.recv(4096)
                            if not chunk:
                                break
                            request.extend(chunk)
                        if not request:
                            break
                        requests_seen.append(bytes(request))
                        body = b'{"ok":true}'
                        connection_header = (
                            b"close" if len(requests_seen) in {2, 3} else b"keep-alive"
                        )
                        protected.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: application/json\r\n"
                            b"Content-Length: 11\r\n"
                            b"Set-Cookie: must-not-persist=secret; Secure\r\n"
                            b"Connection: "
                            + connection_header
                            + b"\r\n\r\n"
                            + body
                        )
                        if len(requests_seen) in {2, 3}:
                            break
                finally:
                    protected.close()
        except Exception as exc:
            origin_errors.append(exc)
        finally:
            origin_listener.close()

    origin_thread = threading.Thread(target=serve_origin, daemon=True)
    origin_thread.start()
    host_threads = []
    opened_parent_tunnels = []

    def open_parent_tunnel(_host, _port):
        opened_parent_tunnels.append((_host, _port))
        enclave_side, host_side = socket.socketpair()
        host_thread = threading.Thread(
            target=_handle_connection,
            kwargs={
                "connection": host_side,
                "connector": lambda _name, _number: socket.create_connection(
                    origin_address,
                    timeout=2,
                ),
                "idle_timeout_seconds": 2,
            },
            daemon=True,
        )
        host_thread.start()
        host_threads.append(host_thread)
        enclave_side.sendall(
            _frame(
                {
                    "method": "connect",
                    "params": {
                        "host": "example.com",
                        "port": 443,
                        "policy_hash": destination_policy_hash(),
                    },
                }
            )
        )
        response = _read_frame(enclave_side)
        assert response["result"]["status"] == "connected"
        return enclave_side

    proxy_port = _unused_loopback_port()
    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        local_port=proxy_port,
        loopback_initializer=lambda: None,
        idle_timeout_seconds=2,
    )
    proxy._open_parent_tunnel = open_parent_tunnel
    proxy._configure_environment = lambda: None
    proxy.start()
    transport = HTTPXProviderTransport(
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        ca_bundle=str(certificate_path),
    )
    try:
        results = [
            transport(
                method="GET",
                url="https://example.com/artifact",
                headers={"accept": "application/json"},
                body=b"",
                timeout_ms=3000,
            )
            for _ in range(2)
        ]
        assert opened_parent_tunnels == [("example.com", 443)]
        results.append(
            transport(
                method="GET",
                url="https://example.com/artifact",
                headers={"accept": "application/json"},
                body=b"",
                timeout_ms=3000,
            )
        )
    finally:
        transport.close()
        proxy.stop()
        origin_thread.join(timeout=3)
        for thread in host_threads:
            thread.join(timeout=3)

    assert origin_errors == []
    assert len(requests_seen) == 3
    assert all(b"must-not-persist" not in request for request in requests_seen)
    assert len(origin_connections) == 2
    assert opened_parent_tunnels == [
        ("example.com", 443),
        ("example.com", 443),
    ]
    assert [result["body"] for result in results] == [
        b'{"ok":true}',
        b'{"ok":true}',
        b'{"ok":true}',
    ]
    assert all(result["tls_protocol"].startswith("TLSv1.") for result in results)


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


def test_enclave_proxy_accepts_exact_tunnel_framing_opt_in():
    parsed = _parse_proxy_request(
        b"CONNECT immutable.example:443 HTTP/1.1\r\n"
        b"Host: immutable.example:443\r\n"
        + TUNNEL_FRAMING_HEADER.encode("ascii")
        + b": "
        + TUNNEL_FRAMING_MODE.encode("ascii")
        + b"\r\n\r\n"
    )

    assert parsed == {
        "method": "CONNECT",
        "host": "immutable.example",
        "port": 443,
        "forward_headers": b"",
        "tls_protected": True,
        "tunnel_framing": TUNNEL_FRAMING_MODE,
    }


def test_enclave_proxy_rejects_duplicate_tunnel_framing_opt_in():
    framing_header = TUNNEL_FRAMING_HEADER.encode("ascii")
    with pytest.raises(EnclaveEgressProxyError, match="duplicated"):
        _parse_proxy_request(
            b"CONNECT immutable.example:443 HTTP/1.1\r\n"
            b"Host: immutable.example:443\r\n"
            + framing_header
            + b": length-v1\r\n"
            + framing_header
            + b": length-v1\r\n\r\n"
        )


@pytest.mark.parametrize("value", (b"", b"length-v2", b"length-v1, length-v1"))
def test_enclave_proxy_rejects_invalid_tunnel_framing_opt_in(value):
    with pytest.raises(EnclaveEgressProxyError, match="tunnel framing header"):
        _parse_proxy_request(
            b"CONNECT immutable.example:443 HTTP/1.1\r\n"
            b"Host: immutable.example:443\r\n"
            + TUNNEL_FRAMING_HEADER.encode("ascii")
            + b": "
            + value
            + b"\r\n\r\n"
        )


def test_enclave_proxy_rejects_tunnel_framing_with_upstream_proxy():
    encoded = base64.b64encode(b"https://worker:secret@proxy.example.com:443")
    with pytest.raises(EnclaveEgressProxyError, match="cannot use framed"):
        _parse_proxy_request(
            b"CONNECT immutable.example:443 HTTP/1.1\r\n"
            b"Host: immutable.example:443\r\n"
            b"X-Leadpoet-Upstream-Proxy-B64: "
            + encoded
            + b"\r\n"
            + TUNNEL_FRAMING_HEADER.encode("ascii")
            + b": "
            + TUNNEL_FRAMING_MODE.encode("ascii")
            + b"\r\n\r\n"
        )


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


def test_enclave_proxy_uses_authenticated_http_connect_on_configured_proxy_port():
    enclave_side, proxy_side = socket.socketpair()
    observed = {}
    enclave = EnclaveEgressProxy(recv_exact=_recv_exact)

    def open_parent(host, port, *, purpose="provider"):
        observed["parent"] = (host, port, purpose)
        return enclave_side

    enclave._open_parent_tunnel = open_parent

    def serve_proxy():
        headers = b""
        while b"\r\n\r\n" not in headers:
            headers += proxy_side.recv(4096)
        observed["headers"] = headers
        proxy_side.sendall(
            b"HTTP/1.1 200 Connection Established\r\n"
            b"Proxy-Agent: fixture\r\n\r\n"
        )

    thread = threading.Thread(target=serve_proxy, daemon=True)
    thread.start()
    tunnel = enclave._open_upstream_proxy_tunnel(
        proxy_url="http://worker:secret@proxy.example.com:6162",
        destination_host="openrouter.ai",
        destination_port=443,
    )
    thread.join(timeout=2)

    assert tunnel is enclave_side
    assert observed["parent"] == (
        "proxy.example.com",
        6162,
        "upstream_proxy",
    )
    assert observed["headers"].startswith(
        b"CONNECT openrouter.ai:443 HTTP/1.1\r\n"
    )
    assert b"Proxy-Authorization: Basic " in observed["headers"]
    enclave_side.close()
    proxy_side.close()


def test_enclave_proxy_uses_authenticated_http_connect_to_global_ip_proxy():
    enclave_side, proxy_side = socket.socketpair()
    observed = {}
    enclave = EnclaveEgressProxy(recv_exact=_recv_exact)

    def open_parent(host, port, *, purpose="provider"):
        observed["parent"] = (host, port, purpose)
        return enclave_side

    enclave._open_parent_tunnel = open_parent

    def serve_proxy():
        headers = b""
        while b"\r\n\r\n" not in headers:
            headers += proxy_side.recv(4096)
        observed["headers"] = headers
        proxy_side.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

    thread = threading.Thread(target=serve_proxy, daemon=True)
    thread.start()
    tunnel = enclave._open_upstream_proxy_tunnel(
        proxy_url="http://worker:secret@8.8.8.8:6162",
        destination_host="openrouter.ai",
        destination_port=443,
    )
    thread.join(timeout=2)

    assert tunnel is enclave_side
    assert observed["parent"] == ("8.8.8.8", 6162, "upstream_proxy")
    assert b"Proxy-Authorization: Basic " in observed["headers"]
    enclave_side.close()
    proxy_side.close()


def test_enclave_proxy_reports_407_before_rejecting_response_body():
    enclave_side, proxy_side = socket.socketpair()
    enclave = EnclaveEgressProxy(recv_exact=_recv_exact)
    enclave._open_parent_tunnel = (
        lambda _host, _port, *, purpose="provider": enclave_side
    )

    def serve_proxy():
        headers = b""
        while b"\r\n\r\n" not in headers:
            headers += proxy_side.recv(4096)
        proxy_side.sendall(
            b"HTTP/1.1 407 Proxy Authentication Required\r\n"
            b"Content-Length: 17\r\n\r\n"
            b"not authenticated"
        )

    thread = threading.Thread(target=serve_proxy, daemon=True)
    thread.start()
    with pytest.raises(
        egress_proxy.EnclaveEgressProxyError,
        match="CONNECT failed with HTTP status 407",
    ):
        enclave._open_upstream_proxy_tunnel(
            proxy_url="http://worker:invalid@proxy.example.com:6162",
            destination_host="openrouter.ai",
            destination_port=443,
        )
    thread.join(timeout=2)
    proxy_side.close()


def test_enclave_proxy_rejects_external_plaintext_http():
    with pytest.raises(egress_proxy.EnclaveEgressProxyError, match="forbidden"):
        _parse_proxy_request(
            b"GET http://archive.org/wayback/available?url=x HTTP/1.1\r\n"
            b"Host: archive.org\r\nProxy-Authorization: Basic secret\r\n\r\n"
        )


def test_enclave_proxy_brings_loopback_interface_up_before_use():
    observed = []
    flags = {"value": 0}

    class _ControlSocket:
        def fileno(self):
            return 7

        def close(self):
            observed.append("close")

    def ioctl(file_descriptor, operation, request):
        observed.append((file_descriptor, operation))
        name, requested_flags, padding = egress_proxy.struct.unpack(
            "16sH22s",
            request,
        )
        assert name.rstrip(b"\0") == b"lo"
        if operation == SIOCGIFFLAGS:
            return egress_proxy.struct.pack(
                "16sH22s",
                name,
                flags["value"],
                padding,
            )
        assert operation == SIOCSIFFLAGS
        flags["value"] = requested_flags
        return request

    _ensure_loopback_interface(
        ioctl=ioctl,
        socket_factory=lambda *_args: _ControlSocket(),
    )

    assert flags["value"] & IFF_UP
    assert observed == [
        (7, SIOCGIFFLAGS),
        (7, SIOCSIFFLAGS),
        (7, SIOCGIFFLAGS),
        "close",
    ]


def test_enclave_proxy_start_proves_loopback_listener_before_ready(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        local_port = int(reservation.getsockname()[1])

    initialized = []
    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        local_port=local_port,
        loopback_initializer=lambda: initialized.append(True),
    )
    monkeypatch.setattr(proxy, "_configure_environment", lambda: None)

    status = proxy.start()
    try:
        assert initialized == [True]
        assert status["status"] == "running"
        assert status["loopback_listener_verified"] is True
    finally:
        proxy.stop()


def test_enclave_proxy_start_fails_closed_when_loopback_initialization_fails():
    def fail_loopback():
        raise EnclaveEgressProxyError("loopback unavailable")

    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        loopback_initializer=fail_loopback,
    )

    with pytest.raises(EnclaveEgressProxyError, match="loopback unavailable"):
        proxy.start()

    assert proxy.status()["status"] == "stopped"
    assert proxy.status()["loopback_listener_verified"] is False


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


def test_enclave_parent_handshake_binds_exact_framing_mode():
    enclave, parent = socket.socketpair()

    class _VsockAdapter:
        def connect(self, _address):
            return None

        def sendall(self, data):
            return enclave.sendall(data)

        def recv(self, size):
            return enclave.recv(size)

        def close(self):
            return enclave.close()

    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        socket_factory=lambda *_args: _VsockAdapter(),
        idle_timeout_seconds=2,
    )

    def parent_side():
        request = _read_frame(parent)
        assert request["params"] == {
            "host": "immutable.example",
            "port": 443,
            "policy_hash": destination_policy_hash(),
            "tunnel_framing": TUNNEL_FRAMING_MODE,
        }
        parent.sendall(
            _frame(
                {
                    "result": {
                        "status": "connected",
                        "policy_hash": destination_policy_hash(),
                        "tunnel_framing": TUNNEL_FRAMING_MODE,
                    }
                }
            )
        )

    thread = threading.Thread(target=parent_side, daemon=True)
    thread.start()
    tunnel = proxy._open_parent_tunnel(
        "immutable.example",
        443,
        tunnel_framing=TUNNEL_FRAMING_MODE,
    )
    assert isinstance(tunnel, _VsockAdapter)
    thread.join(timeout=2)
    enclave.close()
    parent.close()


def test_enclave_parent_handshake_rejects_framing_downgrade():
    enclave, parent = socket.socketpair()

    class _VsockAdapter:
        def connect(self, _address):
            return None

        def sendall(self, data):
            return enclave.sendall(data)

        def recv(self, size):
            return enclave.recv(size)

        def close(self):
            return enclave.close()

    proxy = EnclaveEgressProxy(
        recv_exact=_recv_exact,
        socket_factory=lambda *_args: _VsockAdapter(),
        idle_timeout_seconds=2,
    )

    def parent_side():
        _read_frame(parent)
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
    with pytest.raises(EnclaveEgressProxyError, match="framing mismatch"):
        proxy._open_parent_tunnel(
            "immutable.example",
            443,
            tunnel_framing=TUNNEL_FRAMING_MODE,
        )
    thread.join(timeout=2)
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
        "error_type": type(OSError(111, "connection refused")).__name__,
        "errno": 111,
        "destination_ref": "4877532cd3300944",
    }
    assert "supabase" not in str(status)
    client.close()


def test_enclave_proxy_health_exposes_bounded_clean_tunnel_summary():
    client, proxy_side = socket.socketpair()
    parent_side, upstream = socket.socketpair()
    proxy = EnclaveEgressProxy(recv_exact=_recv_exact)

    def open_parent(_host, _port):
        return parent_side

    proxy._open_parent_tunnel = open_parent

    thread = threading.Thread(
        target=proxy._handle_client,
        args=(proxy_side,),
        daemon=True,
    )
    thread.start()
    client.sendall(
        b"CONNECT qplwoislplkcegvdmbim.supabase.co:443 HTTP/1.1\r\n"
        b"Host: qplwoislplkcegvdmbim.supabase.co:443\r\n\r\n"
    )
    assert client.recv(4096).startswith(b"HTTP/1.1 200 Connection Established")
    client.sendall(b"client hello")
    assert upstream.recv(64) == b"client hello"
    upstream.sendall(b"server hello")
    upstream.shutdown(socket.SHUT_WR)
    assert client.recv(64) == b"server hello"
    client.shutdown(socket.SHUT_WR)
    thread.join(timeout=2)

    assert proxy.status()["last_tunnel"] == {
        "stage": "relay_tls_tunnel",
        "destination_ref": "4877532cd3300944",
        "client_to_parent_bytes": 12,
        "parent_to_client_bytes": 12,
        "first_closed": "parent",
    }
    client.close()
    upstream.close()


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
