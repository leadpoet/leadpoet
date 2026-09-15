from __future__ import annotations

from datetime import datetime, timedelta, timezone
import http.client
import importlib.util
import shutil
import socket
import ssl
import tempfile
import threading
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from lab_arena import web_egress
from lab_arena.web_egress_bridge import LoopbackWebEgressBridge


PUBLIC_IP = "93.184.216.34"


def _read_headers(connection: socket.socket) -> bytes:
    buffered = bytearray()
    while b"\r\n\r\n" not in buffered:
        chunk = connection.recv(4096)
        if not chunk:
            break
        buffered.extend(chunk)
    return bytes(buffered)


def _certificate(tmp_path: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "public.example")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("public.example")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "origin-cert.pem"
    key_path = tmp_path / "origin-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


class FakeUpstreamProxy:
    def __init__(self, cert_path: Path, key_path: Path) -> None:
        self.requests: list[bytes] = []
        self.inner_requests: list[bytes] = []
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self.address = self._listener.getsockname()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._tls.load_cert_chain(str(cert_path), str(key_path))
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def connect(self, _host: str, _port: int) -> socket.socket:
        return socket.create_connection(self.address, timeout=2)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle, args=(connection,), daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def _handle(self, connection: socket.socket) -> None:
        try:
            request = _read_headers(connection)
            self.requests.append(request)
            if request.startswith(b"CONNECT "):
                connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                with self._tls.wrap_socket(connection, server_side=True) as protected:
                    inner = _read_headers(protected)
                    self.inner_requests.append(inner)
                    protected.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nsecure hello"
                    )
                return
            if b"/chunked " in request.split(b"\r\n", 1)[0]:
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
                )
                return
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nplain hello"
            )
        except (OSError, ssl.SSLError):
            pass
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=1)
        for thread in self._threads:
            thread.join(timeout=1)


def _resolver(host: str, port: int, **_kwargs):
    assert host in {
        "public.example",
        "allowed.example",
        "other.example",
    }
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port))]


@pytest.fixture
def proxy(tmp_path):
    cert_path, key_path = _certificate(tmp_path)
    service = FakeUpstreamProxy(cert_path, key_path)
    try:
        yield service, cert_path
    finally:
        service.close()


@pytest.fixture
def socket_root():
    path = Path(tempfile.mkdtemp(prefix="arena-egress-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _stack(tmp_path: Path, proxy: FakeUpstreamProxy, name: str = "attempt"):
    server = web_egress.WebEgressServer(
        tmp_path / (name + ".sock"),
        "http://worker:secret@proxy.example:6162",
        resolver=_resolver,
        upstream_connector=proxy.connect,
        idle_timeout_seconds=2,
    ).start()
    bridge = LoopbackWebEgressBridge(server.path, idle_timeout_seconds=2).start()
    return server, bridge


def test_real_plain_http_client_is_pinned_through_attempt_proxy(socket_root, proxy):
    service, _cert_path = proxy
    server, bridge = _stack(socket_root, service)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=2)
        connection.request(
            "GET",
            "http://public.example/news?q=1",
            headers={"Host": "public.example"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"plain hello"
        connection.close()
    finally:
        bridge.stop()
        server.stop()

    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.startswith(
        b"GET http://%s/news?q=1 HTTP/1.1\r\n" % PUBLIC_IP.encode()
    )
    assert request.lower().count(b"\r\nhost:") == 1
    assert b"Host: public.example\r\n" in request
    assert b"Proxy-Authorization: Basic d29ya2VyOnNlY3JldA==\r\n" in request
    assert b"Connection: close\r\n" in request
    assert not server.path.exists()


def test_plain_http_chunked_response_is_relayed_without_following_requests(
    socket_root, proxy
):
    service, _cert_path = proxy
    server, bridge = _stack(socket_root, service)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=2)
        connection.request(
            "GET",
            "http://public.example/chunked",
            headers={"Host": "public.example"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"hello world"
        connection.close()
    finally:
        bridge.stop()
        server.stop()


def test_real_https_client_keeps_hostname_sni_while_proxy_uses_pinned_ip(
    socket_root, proxy
):
    service, cert_path = proxy
    server, bridge = _stack(socket_root, service)
    context = ssl.create_default_context(cafile=str(cert_path))
    try:
        connection = http.client.HTTPSConnection(
            "127.0.0.1", bridge.port, timeout=3, context=context
        )
        connection.set_tunnel("public.example", 443)
        connection.request("GET", "/secure", headers={"Host": "public.example"})
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"secure hello"
        connection.close()
    finally:
        bridge.stop()
        server.stop()

    assert service.requests[0].startswith(
        b"CONNECT %s:443 HTTP/1.1\r\n" % PUBLIC_IP.encode()
    )
    assert b"Proxy-Authorization: Basic d29ya2VyOnNlY3JldA==\r\n" in service.requests[0]
    assert service.inner_requests[0].startswith(b"GET /secure HTTP/1.1\r\n")


@pytest.mark.parametrize(
    "raw_request",
    [
        b"GET http://openrouter.ai/api/v1/models HTTP/1.1\r\nHost: openrouter.ai\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: other.example\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nProxy-Authorization: Basic eA==\r\n\r\n",
        b"CONNECT allowed.example:80 HTTP/1.1\r\nHost: allowed.example:80\r\n\r\n",
        b"GET http://127.0.0.1/ HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
        b"GET http://docs.openrouter.ai/ HTTP/1.1\r\nHost: docs.openrouter.ai\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nTransfer-Encoding: chunked\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\n\r\nGET http://other.example/ HTTP/1.1\r\nHost: other.example\r\n\r\n",
        b"GET http://allowed.example/evil\tpath HTTP/1.1\r\nHost: allowed.example\r\n\r\n",
        b"TRACE http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\n\r\n",
        b"CONNECT allowed.example:443/path HTTP/1.1\r\nHost: allowed.example:443\r\n\r\n",
        b"CONNECT allowed.example:443 HTTP/1.1\r\nHost: allowed.example:443\r\nContent-Length: 1\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example/path\r\n\r\n",
    ],
)
def test_paid_private_mismatched_and_authenticated_bypasses_are_denied(
    socket_root, proxy, raw_request
):
    service, _cert_path = proxy
    server, bridge = _stack(socket_root, service)
    try:
        client = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
        client.sendall(raw_request)
        assert _read_headers(client).startswith(b"HTTP/1.1 403 Forbidden")
        client.close()
    finally:
        bridge.stop()
        server.stop()
    assert service.requests == []


def test_dns_answer_with_private_address_fails_closed_before_proxy(socket_root, proxy):
    service, _cert_path = proxy

    def mixed_resolver(_host, port, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    server = web_egress.WebEgressServer(
        socket_root / "mixed.sock",
        "http://proxy.example:6162",
        resolver=mixed_resolver,
        upstream_connector=service.connect,
    ).start()
    bridge = LoopbackWebEgressBridge(server.path).start()
    try:
        client = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
        client.sendall(
            b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n\r\n"
        )
        assert _read_headers(client).startswith(b"HTTP/1.1 403 Forbidden")
        client.close()
    finally:
        bridge.stop()
        server.stop()
    assert service.requests == []


def test_connect_sni_must_match_before_any_upstream_connection(socket_root, proxy):
    service, _cert_path = proxy
    server, bridge = _stack(socket_root, service)
    try:
        raw = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
        raw.sendall(
            b"CONNECT allowed.example:443 HTTP/1.1\r\nHost: allowed.example:443\r\n\r\n"
        )
        assert _read_headers(raw).startswith(b"HTTP/1.1 200 Connection Established")
        context = ssl.create_default_context()
        with pytest.raises((OSError, ssl.SSLError)):
            context.wrap_socket(raw, server_hostname="other.example")
    finally:
        bridge.stop()
        server.stop()
    assert service.requests == []


def test_ech_grease_extension_preserves_visible_sni_policy():
    encoded_host = b"public.example"
    server_name = b"\x00" + len(encoded_host).to_bytes(2, "big") + encoded_host
    sni_value = len(server_name).to_bytes(2, "big") + server_name
    prefix = b"\x03\x03" + b"x" * 32 + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00"
    extensions = (
        b"\x00\x00"
        + len(sni_value).to_bytes(2, "big")
        + sni_value
        + b"\xfe\x0d\x00\x04\x00\x00\x00\x00"
    )
    body = prefix + len(extensions).to_bytes(2, "big") + extensions
    assert web_egress._parse_client_hello(body) == "public.example"


def test_truncated_client_hello_is_rejected_as_policy_error():
    with pytest.raises(web_egress.WebEgressError, match="malformed"):
        web_egress._parse_client_hello(b"x" * 34)


def test_attempt_shutdown_is_independent_and_removes_only_its_socket(
    socket_root, proxy
):
    service, _cert_path = proxy
    first_server, first_bridge = _stack(socket_root, service, "first")
    second_server, second_bridge = _stack(socket_root, service, "second")
    first_path = first_server.path
    second_path = second_server.path
    try:
        first_bridge.stop()
        first_server.stop()
        assert not first_path.exists()
        assert second_path.is_socket()
        connection = http.client.HTTPConnection(
            "127.0.0.1", second_bridge.port, timeout=2
        )
        connection.request(
            "GET", "http://public.example/", headers={"Host": "public.example"}
        )
        assert connection.getresponse().read() == b"plain hello"
        connection.close()
    finally:
        second_bridge.stop()
        second_server.stop()
    assert not second_path.exists()


def test_attempt_connection_limit_cannot_be_reset_by_reopening(socket_root, proxy):
    service, _cert_path = proxy
    server = web_egress.WebEgressServer(
        socket_root / "connection-limit.sock",
        "http://proxy.example:6162",
        max_total_connections=1,
        resolver=_resolver,
        upstream_connector=service.connect,
    ).start()
    try:
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        first.settimeout(2)
        first.connect(str(server.path))
        first.sendall(
            b"GET http://openrouter.ai/ HTTP/1.1\r\nHost: openrouter.ai\r\n\r\n"
        )
        assert _read_headers(first).startswith(b"HTTP/1.1 403 Forbidden")
        first.close()

        second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        second.settimeout(2)
        second.connect(str(server.path))
        second.sendall(
            b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n\r\n"
        )
        assert _read_headers(second).startswith(b"HTTP/1.1 503 Service Unavailable")
        second.close()
    finally:
        server.stop()

    assert server.summary["accepted_connection_count"] == 1
    assert server.summary["total_limit_rejection_count"] == 1
    assert service.requests == []


def test_attempt_byte_limit_is_cumulative_and_summary_is_secret_free(
    socket_root, proxy
):
    service, _cert_path = proxy
    server = web_egress.WebEgressServer(
        socket_root / "byte-limit.sock",
        "http://worker:secret@proxy.example:6162",
        max_total_bytes=180,
        resolver=_resolver,
        upstream_connector=service.connect,
    ).start()
    bridge = LoopbackWebEgressBridge(server.path).start()
    raw_request = b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n\r\n"
    try:
        first = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
        first.sendall(raw_request)
        assert b"plain hello" in _read_headers(first)
        first.close()

        second = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
        second.sendall(raw_request)
        assert _read_headers(second).startswith(b"HTTP/1.1 403 Forbidden")
        second.close()
    finally:
        bridge.stop()
        server.stop()

    summary = server.summary
    assert summary["schema_version"] == "lab_arena.web_egress_summary.v1"
    assert summary["accepted_connection_count"] == 2
    assert summary["byte_limit_rejection_count"] == 1
    assert summary["client_to_web_bytes"] > 0
    assert summary["web_to_client_bytes"] > 0
    assert summary["client_to_web_bytes"] + summary["web_to_client_bytes"] <= 180
    serialized = repr(summary).lower()
    assert all(
        forbidden not in serialized
        for forbidden in ("worker", "secret", "proxy.example", "public.example")
    )
    assert len(service.requests) == 1


def test_unproven_shared_proxy_cleanup_blocks_new_connections_until_released(
    socket_root,
):
    class UnclosedProxyStream:
        def __init__(self):
            self.allow_close = False

        def settimeout(self, _timeout):
            raise OSError("proxy setup failed")

        def shutdown(self, _how):
            return None

        def close(self):
            return None if self.allow_close else False

    stream = UnclosedProxyStream()
    upstream_connections = []

    def connect_upstream(_host, _port):
        upstream_connections.append(1)
        return stream

    server = web_egress.WebEgressServer(
        socket_root / "cleanup.sock",
        "http://proxy.example:6162",
        resolver=_resolver,
        upstream_connector=connect_upstream,
    ).start()
    raw_request = b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n\r\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as first:
            first.settimeout(2)
            first.connect(str(server.path))
            first.sendall(raw_request)
            assert _read_headers(first).startswith(b"HTTP/1.1 403 Forbidden")
        assert server.summary["pending_cleanup_count"] == 1
        assert len(upstream_connections) == 1

        # Admission is blocked before the server reads a request. Using the
        # host socket directly avoids a bridge relay racing the immediate close.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as second:
            second.settimeout(2)
            second.connect(str(server.path))
            assert _read_headers(second).startswith(b"HTTP/1.1 503 Service Unavailable")
        assert server.summary["pending_cleanup_count"] == 1
        assert len(upstream_connections) == 1

        stream.allow_close = True
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as third:
            third.settimeout(2)
            third.connect(str(server.path))
            third.sendall(raw_request)
            assert _read_headers(third).startswith(b"HTTP/1.1 403 Forbidden")
        assert len(upstream_connections) == 2
    finally:
        stream.allow_close = True
        server.stop()

    assert server.summary["pending_cleanup_count"] == 0
    assert server.summary["cleanup_block_rejection_count"] == 1


def test_stop_fails_closed_while_transport_cleanup_is_unproven(socket_root):
    class UnclosedProxyStream:
        allow_close = False

        def settimeout(self, _timeout):
            raise OSError("proxy setup failed")

        def shutdown(self, _how):
            return None

        def close(self):
            return None if self.allow_close else False

    stream = UnclosedProxyStream()
    server = web_egress.WebEgressServer(
        socket_root / "stop-cleanup.sock",
        "http://proxy.example:6162",
        resolver=_resolver,
        upstream_connector=lambda _host, _port: stream,
    ).start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(str(server.path))
    client.sendall(
        b"GET http://public.example/ HTTP/1.1\r\nHost: public.example\r\n\r\n"
    )
    assert _read_headers(client).startswith(b"HTTP/1.1 403 Forbidden")
    client.close()
    with pytest.raises(web_egress.WebEgressError, match="cleanup is incomplete"):
        server.stop()
    assert server.summary["pending_cleanup_count"] == 1

    stream.allow_close = True
    server.stop()
    assert server.summary["pending_cleanup_count"] == 0


def test_stop_fails_closed_while_accept_loop_is_alive(socket_root):
    entered = threading.Event()
    release = threading.Event()

    class StuckAcceptListener:
        def __init__(self, *args):
            self.socket = socket.socket(*args)

        def __getattr__(self, name):
            return getattr(self.socket, name)

        def accept(self):
            entered.set()
            release.wait(timeout=10)
            raise OSError("accept released")

    server = web_egress.WebEgressServer(
        socket_root / "stuck-accept.sock",
        socket_factory=StuckAcceptListener,
    ).start()
    assert entered.wait(timeout=2)
    with pytest.raises(web_egress.WebEgressError, match="cleanup is incomplete"):
        server.stop()
    assert server.last_failure["stage"] == "transport_cleanup"

    release.set()
    server.stop()
    assert server._thread is None


def test_bridge_module_loads_standalone_with_only_standard_library():
    path = Path(__file__).resolve().parents[2] / "lab_arena/web_egress_bridge.py"
    spec = importlib.util.spec_from_file_location("isolated_arena_bridge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LoopbackWebEgressBridge.__module__ == "isolated_arena_bridge"
