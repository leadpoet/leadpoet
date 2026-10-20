"""Real socket controls for passive Python HTTP response capture."""
import asyncio
import gzip
import hashlib
import http.client
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from lab_arena.automatic_capture import install


IDENTITY = "sha256:" + "a" * 64
BODY = b'hello world\n{"value": 42}\n'


class Checkpoint:
    def __init__(self, delay=0):
        self.web = []
        self.acks = []
        self.states = []
        self.delay = delay

    def record_web_fetch(self, **kwargs):
        time.sleep(self.delay)
        self.web.append(kwargs)
        return len(self.web)

    def record_response_received(self, **kwargs):
        self.acks.append(kwargs)
        return len(self.acks)

    def record_state_snapshot(self, name, value):
        self.states.append((name, value))
        return len(self.states)


class QuietHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if sys.exc_info()[0] in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.server.paths.append(self.path)
            data = BODY
            code = 200
            media = "text/plain; charset=utf-8"
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/text")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.endswith("gzip"):
                data = gzip.compress(data)
            if self.path == "/latin":
                data = "caf\xe9".encode("latin-1")
                media = "text/plain; charset=iso-8859-1"
            if self.path == "/empty":
                data = b""
            if self.path == "/longtype":
                media = "text/plain; " + "a" * 256
            if self.path == "/binary":
                data = b"\x00\xff\x10"
                media = "application/octet-stream"
            if self.path == "/invalid":
                data = b"abc\xff"
            if self.path == "/unicode":
                data = "\u20acabc".encode()
            if self.path in {"/error", "/provider-error"}:
                code = 503
            self.send_response(code)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(data) + (10 if self.path == "/broken" else 0)))
            if self.path.endswith("gzip"):
                self.send_header("Content-Encoding", "gzip")
            if self.path.startswith("/provider"):
                self.send_header("x-leadpoet-call-identity", "invalid" if self.path == "/provider-invalid" else IDENTITY)
            self.end_headers()
            self.server.last_data = data
            try:
                if self.path == "/slow":
                    self.wfile.write(data[:3])
                    self.wfile.flush()
                    time.sleep(0.2)
                    data = data[3:]
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if self.path == "/broken":
                self.close_connection = True

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:" + str(server.server_port), server
    server.shutdown()
    server.server_close()
    thread.join(1)


def finish(recorder, checkpoint):
    summary = recorder.close()
    assert not recorder.worker.is_alive(), summary
    assert recorder.retained == 0
    return summary


def assert_web(event, data, complete=True):
    capture = event["capture"]
    assert event["body"].encode(capture["encoding"]) == data
    assert capture["received_body_bytes"] == len(data)
    assert capture["received_body_sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
    assert capture["complete"] is complete


@pytest.mark.parametrize("client", ["stdlib", "urllib", "httpx", "requests", "urllib3"])
def test_real_http_bytes_once(endpoint, client):
    url, server = endpoint
    httpx = pytest.importorskip("httpx")
    requests = pytest.importorskip("requests")
    urllib3 = pytest.importorskip("urllib3")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        if client == "stdlib":
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", "/text")
            response = connection.getresponse()
            assert type(response) is http.client.HTTPResponse
            result = response.read()
            connection.close()
        elif client == "urllib":
            with urlopen(url + "/text") as response:
                result = response.read()
        elif client == "httpx":
            response = httpx.get(url + "/text")
            assert type(response) is httpx.Response
            result = response.content
        elif client == "requests":
            response = requests.get(url + "/text")
            assert type(response) is requests.Response
            result = response.content
        else:
            response = urllib3.PoolManager().request("GET", url + "/text")
            result = response.data
        assert result == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1, (checkpoint.web, summary)
    assert_web(checkpoint.web[0], BODY)
    assert server.paths == ["/text"]


@pytest.mark.parametrize("client", ["httpx", "requests", "stdlib"])
def test_lazy_partial_close_and_unread(endpoint, client):
    url, server = endpoint
    httpx = pytest.importorskip("httpx")
    requests = pytest.importorskip("requests")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        if client == "httpx":
            with httpx.Client() as session:
                with session.stream("GET", url + "/text") as response:
                    assert checkpoint.web == []
                    iterator = response.iter_bytes(chunk_size=3)
                    assert next(iterator) == BODY[:3]
                with session.stream("GET", url + "/text"):
                    pass
        elif client == "requests":
            with requests.get(url + "/text", stream=True) as response:
                assert checkpoint.web == []
                iterator = response.iter_content(chunk_size=3)
                assert next(iterator) == BODY[:3]
            with requests.get(url + "/text", stream=True):
                pass
        else:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", "/text")
            response = connection.getresponse()
            assert checkpoint.web == []
            assert response.read(3) == BODY[:3]
            response.close()
            connection.close()
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1, summary
    assert_web(checkpoint.web[0], BODY[:3], False)
    assert summary["complete"] is False


@pytest.mark.parametrize("client", ["httpx", "requests", "urllib3"])
def test_gzip_decoded_and_redirect_semantics(endpoint, client):
    url, server = endpoint
    module = pytest.importorskip(client)
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        if client == "httpx":
            assert module.get(url + "/gzip").content == BODY
            response = module.get(url + "/redirect", follow_redirects=True)
            assert len(response.history) == 1
            assert response.content == BODY
        elif client == "requests":
            assert module.get(url + "/gzip").content == BODY
            response = module.get(url + "/redirect")
            assert len(response.history) == 1
            assert response.content == BODY
        else:
            session = module.PoolManager()
            assert session.request("GET", url + "/gzip").data == BODY
            assert session.request("GET", url + "/redirect").data == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 3, (checkpoint.web, summary)
    for event in checkpoint.web:
        assert_web(event, b"" if event["status"] == 302 else BODY)
    assert server.paths == ["/gzip", "/redirect", "/text"]


def test_readinto_iteration_and_zero_reads(endpoint):
    url, server = endpoint
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/text")
        response = connection.getresponse()
        assert response.read(0) == b""
        buffer = bytearray(4)
        assert response.readinto(buffer) == 4
        assert bytes(buffer) == BODY[:4]
        assert b"".join(response) == BODY[4:]
        connection.close()
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1, summary
    assert_web(checkpoint.web[0], BODY)


@pytest.mark.parametrize("path,expected", [("/latin", "caf\xe9".encode("latin-1")), ("/error", BODY)])
def test_non_utf8_and_http_error_body(endpoint, path, expected):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        assert httpx.get(endpoint[0] + path).content == expected
    finally:
        finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1
    assert_web(checkpoint.web[0], expected)
    assert checkpoint.web[0]["status"] == (503 if path == "/error" else 200)


@pytest.mark.parametrize("path,counter", [("/binary", "binary_body_unsupported"), ("/invalid", "text_encoding_unsupported")])
def test_binary_invalid_utf8_is_explicit(endpoint, path, counter):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        httpx.get(endpoint[0] + path).content
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.web == []
    assert summary["counts"][counter] == 1
    assert summary["complete"] is False


def test_partial_split_unicode_is_explicit(endpoint):
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        with urlopen(endpoint[0] + "/unicode") as response:
            assert response.read(1) == b"\xe2"
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.web == []
    assert summary["counts"]["text_encoding_unsupported"] == 1


@pytest.mark.parametrize("client", ["httpx", "requests", "stdlib"])
def test_provider_exact_ack_no_web_copy(endpoint, client, monkeypatch):
    monkeypatch.setattr("lab_arena.automatic_capture._PROVIDER_HOSTS", {"127.0.0.1"})
    httpx = pytest.importorskip("httpx")
    requests = pytest.importorskip("requests")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        if client == "httpx":
            assert httpx.get(endpoint[0] + "/provider").content == BODY
        elif client == "requests":
            assert requests.get(endpoint[0] + "/provider").content == BODY
        else:
            with urlopen(endpoint[0] + "/provider") as response:
                assert response.read() == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.web == []
    assert checkpoint.acks == [{"call_identity": IDENTITY, "status": 200,
        "body_sha256": "sha256:" + hashlib.sha256(BODY).hexdigest(), "body_bytes": len(BODY)}], summary


def test_async_httpx_and_aiohttp(endpoint):
    httpx = pytest.importorskip("httpx")
    aiohttp = pytest.importorskip("aiohttp")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)

    async def run():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", endpoint[0] + "/text") as response:
                assert b"".join([data async for data in response.aiter_bytes(chunk_size=3)]) == BODY
        async with aiohttp.ClientSession() as client:
            async with client.get(endpoint[0] + "/gzip") as response:
                assert await response.read() == BODY
            async with client.get(endpoint[0] + "/text") as response:
                assert b"".join([data async for data in response.content.iter_chunked(3)]) == BODY
    try:
        asyncio.run(run())
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 3, (checkpoint.web, summary)
    for event in checkpoint.web:
        assert_web(event, BODY)


def test_original_transport_exception(endpoint):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        with pytest.raises(httpx.RemoteProtocolError):
            httpx.get(endpoint[0] + "/broken")
    finally:
        summary = finish(recorder, checkpoint)
    assert summary["counts"]["responses_partial"] >= 1


def test_multi_thread_and_backpressure_bound(endpoint):
    requests = pytest.importorskip("requests")
    checkpoint = Checkpoint(delay=0.3)
    recorder = install(checkpoint, memory_limit=64, queue_limit=1, close_timeout=0.01)
    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: requests.get(endpoint[0] + "/text").content, range(12)))
        elapsed = time.monotonic() - start
        assert results == [BODY] * 12
        assert recorder.retained <= 64
        assert elapsed < 0.8
    finally:
        summary = recorder.close()
    assert summary["complete"] is False
    assert summary["counts"].get("queue_overflow", 0) + summary["counts"].get("body_buffer_overflow", 0) > 0
    recorder.worker.join(2)
    assert recorder.retained == 0


def test_imports_are_lazy_and_restored():
    import builtins
    import importlib
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        assert importlib.import_module("hashlib") is hashlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("_arena_nonexistent_capture_control")
    finally:
        finish(recorder, checkpoint)
    assert builtins.__import__ is original_import
    assert importlib.import_module is original_import_module


def test_direct_httpx_raw_and_cached_replay(endpoint):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        with httpx.Client() as client:
            with client.stream("GET", endpoint[0] + "/text") as response:
                assert b"".join(response.iter_raw(chunk_size=3)) == BODY
            response = client.get(endpoint[0] + "/text")
            assert response.content == BODY
            assert b"".join(response.iter_bytes(2)) == BODY
            assert response.text == BODY.decode()
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 2, (checkpoint.web, summary)
    for event in checkpoint.web:
        assert_web(event, BODY)


def test_no_observation_failure_changes_bytes_or_original_error(endpoint, monkeypatch):
    import lab_arena.automatic_capture as capture
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    sentinel = RuntimeError("observer failed")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(capture._Body, "add", fail)
    try:
        with urlopen(endpoint[0] + "/text") as response:
            assert response.read() == BODY
        with urlopen(endpoint[0] + "/broken") as response:
            with pytest.raises(http.client.IncompleteRead) as caught:
                response.read()
            assert caught.value.partial == BODY
            assert caught.value is not sentinel
    finally:
        summary = finish(recorder, checkpoint)
    assert summary["counts"]["adapter_failure"] >= 2


def test_overflow_never_emits_prefix_with_full_body_hash(endpoint):
    checkpoint = Checkpoint()
    recorder = install(checkpoint, memory_limit=4)
    try:
        with urlopen(endpoint[0] + "/text") as response:
            assert response.read() == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.web == []
    assert summary["counts"]["body_buffer_overflow"] == 1
    assert summary["complete"] is False


def test_async_provider_ack_and_aiohttp_partial_cancel(endpoint, monkeypatch):
    monkeypatch.setattr("lab_arena.automatic_capture._PROVIDER_HOSTS", {"127.0.0.1"})
    httpx = pytest.importorskip("httpx")
    aiohttp = pytest.importorskip("aiohttp")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)

    async def run():
        async with httpx.AsyncClient() as client:
            assert (await client.get(endpoint[0] + "/provider")).content == BODY
        async with aiohttp.ClientSession() as client:
            async with client.get(endpoint[0] + "/text") as response:
                assert await response.content.read(3) == BODY[:3]
            async with client.get(endpoint[0] + "/text"):
                pass
    try:
        asyncio.run(run())
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.acks) == 1
    assert checkpoint.acks[0]["body_sha256"] == "sha256:" + hashlib.sha256(BODY).hexdigest()
    assert len(checkpoint.web) == 1, (checkpoint.web, summary)
    assert_web(checkpoint.web[0], BODY[:3], False)


def test_import_before_client_dependencies_in_fresh_process():
    import subprocess
    import sys
    script = '''
import sys
from lab_arena.automatic_capture import install
class Checkpoint:
    def record_state_snapshot(self, *args): return 1
recorder = install(Checkpoint())
assert all(name not in sys.modules for name in ("httpx", "urllib3", "requests", "aiohttp"))
import importlib
httpx = importlib.import_module("httpx")
assert "httpx_sync_async" in recorder.adapters
import requests, aiohttp
assert "urllib3_requests" in recorder.adapters
assert "aiohttp_stream_reader" in recorder.adapters
assert recorder.close()["complete"]
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_spoof_provider_header_is_normal_web_body(endpoint):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        assert httpx.get(endpoint[0] + "/provider").content == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.acks == []
    assert len(checkpoint.web) == 1, summary
    assert_web(checkpoint.web[0], BODY)


def test_provider_partial_error_invalid_header_and_compressed_raw(endpoint, monkeypatch):
    monkeypatch.setattr("lab_arena.automatic_capture._PROVIDER_HOSTS", {"127.0.0.1"})
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        with httpx.Client() as client:
            with client.stream("GET", endpoint[0] + "/provider") as response:
                iterator = response.iter_bytes(chunk_size=3)
                assert next(iterator) == BODY[:3]
            assert client.get(endpoint[0] + "/provider-error").status_code == 503
            assert client.get(endpoint[0] + "/provider-invalid").content == BODY
            response = client.get(endpoint[0] + "/provider-gzip")
            assert response.content == BODY
            wire_body = endpoint[1].last_data
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.acks) == 1, (checkpoint.acks, summary)
    assert checkpoint.acks[0]["body_sha256"] == "sha256:" + hashlib.sha256(wire_body).hexdigest()
    assert len(checkpoint.web) == 1
    assert_web(checkpoint.web[0], BODY)


def test_iterator_early_close_preserves_original_stream_cleanup():
    httpx = pytest.importorskip("httpx")

    def probe(enabled):
        checkpoint = Checkpoint()
        recorder = install(checkpoint) if enabled else None
        effects = []

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                try:
                    yield BODY[:4]
                    yield BODY[4:]
                finally:
                    effects.append("iterator_finally")

            def close(self):
                effects.append("stream_close")

        response = httpx.Response(200, stream=Stream(), request=httpx.Request("GET", "https://example.com/text"),
                                  headers={"Content-Type": "text/plain"})
        iterator = response.iter_bytes(chunk_size=3)
        assert next(iterator) == BODY[:3]
        iterator.close()
        response.close()
        if recorder:
            finish(recorder, checkpoint)
        return effects

    assert probe(True) == probe(False) == ["iterator_finally", "stream_close"]


def test_async_cancellation_preserves_original_result_and_partial_bytes(endpoint):
    aiohttp = pytest.importorskip("aiohttp")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)

    async def run():
        async with aiohttp.ClientSession() as client:
            async with client.get(endpoint[0] + "/slow") as response:
                assert await response.content.readexactly(3) == BODY[:3]
                task = asyncio.create_task(response.content.readexactly(len(BODY)))
                await asyncio.sleep(0.02)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
    try:
        asyncio.run(run())
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1, summary
    assert_web(checkpoint.web[0], BODY[:3], False)


@pytest.mark.parametrize("via_connect", [False, True])
def test_real_local_tls_identical_requests_on_and_off(endpoint, tmp_path, via_connect):
    import shutil
    import ssl
    import subprocess
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("OpenSSL executable is unavailable")
    cert = tmp_path / "test-cert.pem"
    key = tmp_path / "test-key.pem"
    result = subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                             "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1",
                             "-keyout", str(key), "-out", str(cert)], capture_output=True, timeout=10)
    if result.returncode:
        pytest.skip("Local OpenSSL cannot create a short-lived test certificate")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = QuietHTTPServer(("127.0.0.1", 0), endpoint[1].RequestHandlerClass)
    server.paths = []
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trusted = ssl.create_default_context(cafile=str(cert))
    url = "https://127.0.0.1:" + str(server.server_port) + "/text"
    proxy = None
    proxy_thread = None
    if via_connect:
        import select
        import socket
        import socketserver

        class Tunnel(socketserver.BaseRequestHandler):
            def handle(self):
                header = bytearray()
                while not header.endswith(b"\r\n\r\n"):
                    data = self.request.recv(1)
                    if not data:
                        return
                    header.extend(data)
                request_line = header.decode("ascii").split("\r\n", 1)[0]
                method, authority, _ = request_line.split(" ")
                assert method == "CONNECT"
                host, port = authority.rsplit(":", 1)
                self.server.targets.append((host, int(port)))
                with socket.create_connection((host, int(port))) as remote:
                    self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    while True:
                        ready, _, _ = select.select([self.request, remote], [], [], 2)
                        if not ready:
                            return
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (remote if source is self.request else self.request).sendall(data)

        class Proxy(socketserver.ThreadingTCPServer):
            daemon_threads = True

        proxy = Proxy(("127.0.0.1", 0), Tunnel)
        proxy.targets = []
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()

    def fetch():
        if not via_connect:
            with urlopen(url, context=trusted) as response:
                return response.read()
        connection = http.client.HTTPSConnection("127.0.0.1", proxy.server_address[1], context=trusted)
        try:
            connection.set_tunnel("127.0.0.1", server.server_port)
            connection.request("GET", "/text")
            return connection.getresponse().read()
        finally:
            connection.close()

    checkpoint = Checkpoint()
    recorder = None
    try:
        baseline = fetch()
        recorder = install(checkpoint)
        observed = fetch()
        summary = finish(recorder, checkpoint)
    finally:
        if recorder is not None:
            recorder.close()
        server.shutdown()
        server.server_close()
        thread.join(1)
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
            proxy_thread.join(1)
    assert baseline == observed == BODY
    assert server.paths == ["/text", "/text"]
    assert len(checkpoint.web) == 1, summary
    assert_web(checkpoint.web[0], BODY)
    assert checkpoint.web[0]["url"] == url
    if via_connect:
        assert proxy.targets == [("127.0.0.1", server.server_port)] * 2
        assert proxy.server_address[1] != server.server_port


def test_complete_empty_web_body_preserves_url_and_status(endpoint):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        response = httpx.get(endpoint[0] + "/empty")
        assert response.status_code == 200 and response.content == b""
    finally:
        summary = finish(recorder, checkpoint)
    assert len(checkpoint.web) == 1, summary
    assert checkpoint.web[0]["url"] == endpoint[0] + "/empty"
    assert_web(checkpoint.web[0], b"")


@pytest.mark.parametrize("path", ["/longtype", "/" + "a" * 2000])
def test_oversize_metadata_is_not_silently_truncated(endpoint, path):
    httpx = pytest.importorskip("httpx")
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    try:
        assert httpx.get(endpoint[0] + path).content == BODY
    finally:
        summary = finish(recorder, checkpoint)
    assert checkpoint.web == []
    assert summary["counts"]["metadata_unsupported"] >= 1
    assert summary["complete"] is False


def test_close_deadline_and_pending_summary_are_truthful(endpoint):
    checkpoint = Checkpoint(delay=0.5)
    recorder = install(checkpoint)
    try:
        start = time.monotonic()
        with urlopen(endpoint[0] + "/text") as response:
            assert response.read() == BODY
        assert time.monotonic() - start < 0.3
    finally:
        start = time.monotonic()
        summary = recorder.close()
        assert time.monotonic() - start < 0.35
    assert summary["complete"] is False
    assert summary["inflight"] + summary["pending_queue"] > 0
    assert summary["counts"]["close_flush_timeout"] == 1
    assert recorder.close() == summary
    recorder.worker.join(2)
    assert recorder.retained == 0


@pytest.mark.parametrize("outcome", ["error", "exhausted", "explicit_close"])
def test_sync_iterator_close_only_on_explicit_consumer_close(outcome):
    from types import SimpleNamespace
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    original_error = ValueError("original read failure")
    close_error = RuntimeError("original explicit close failure")
    closes = []

    class Iterator:
        first = True

        def __next__(self):
            if self.first:
                self.first = False
                return BODY[:3]
            if outcome == "exhausted":
                raise StopIteration
            raise original_error

        def close(self):
            closes.append(1)
            raise close_error

    iterator = recorder._iterator("test", lambda response: (
        "https://example.com/text", 200, {"content-type": "text/plain"}))(
            lambda response: Iterator())(SimpleNamespace())
    try:
        assert next(iterator) == BODY[:3]
        if outcome == "explicit_close":
            with pytest.raises(RuntimeError) as caught:
                iterator.close()
            assert caught.value is close_error
            assert closes == [1]
        elif outcome == "exhausted":
            with pytest.raises(StopIteration):
                next(iterator)
            assert closes == []
        else:
            with pytest.raises(ValueError) as caught:
                next(iterator)
            assert caught.value is original_error
            assert closes == []
    finally:
        finish(recorder, checkpoint)


@pytest.mark.parametrize("outcome", ["error", "cancelled", "exhausted", "explicit_close"])
def test_async_iterator_close_does_not_replace_error_or_cancellation(outcome):
    from types import SimpleNamespace
    checkpoint = Checkpoint()
    recorder = install(checkpoint)
    original_error = asyncio.CancelledError("original cancellation") if outcome == "cancelled" else ValueError("original read failure")
    close_error = RuntimeError("original explicit close failure")
    closes = []

    class Iterator:
        first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.first:
                self.first = False
                return BODY[:3]
            if outcome == "exhausted":
                raise StopAsyncIteration
            raise original_error

        async def aclose(self):
            closes.append(1)
            raise close_error

    async def run():
        iterator = recorder._async_iterator("test", lambda response: (
            "https://example.com/text", 200, {"content-type": "text/plain"}))(
                lambda response: Iterator())(SimpleNamespace())
        assert await iterator.__anext__() == BODY[:3]
        if outcome == "explicit_close":
            with pytest.raises(RuntimeError) as caught:
                await iterator.aclose()
            assert caught.value is close_error
            assert closes == [1]
        elif outcome == "exhausted":
            with pytest.raises(StopAsyncIteration):
                await iterator.__anext__()
            assert closes == []
        else:
            with pytest.raises(type(original_error)) as caught:
                await iterator.__anext__()
            assert caught.value is original_error
            assert closes == []
    try:
        asyncio.run(run())
    finally:
        finish(recorder, checkpoint)
