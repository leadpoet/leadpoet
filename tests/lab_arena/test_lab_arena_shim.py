"""Sandbox shim tests (labarena.md sections 3.3, 6.1, 18.4).

A threaded Unix-socket fake worker records every frame. The shim is installed
in-process for each test and uninstalled afterwards, restoring the original
client methods so the rest of the suite is unaffected.

Raw TCP or UDP to a provider host is not something the shim can block; runsc
``--network=none`` does that on the runner (section 9.2) and is covered by the
Linux-only sandbox checks, not here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import tempfile
import threading
import urllib.error
import urllib.request

import aiohttp
import httpx
import pytest
import requests

from lab_arena import contracts, operations, shim

ORIGINALS = {
    "urlopen": urllib.request.urlopen,
    "httpx_send": httpx.Client.send,
    "httpx_async_send": httpx.AsyncClient.send,
    "requests_send": requests.Session.send,
    "aiohttp_request": aiohttp.ClientSession._request,
}


class FakeWorker:
    """Records frames and answers with a configurable framed document."""

    def __init__(self) -> None:
        # macOS limits Unix socket paths to 104 bytes, so the socket lives in
        # a short temporary directory rather than pytest's tmp_path.
        self.directory = tempfile.mkdtemp(prefix="la-shim-", dir="/tmp")
        self.path = os.path.join(self.directory, "w.sock")
        self.frames: list = []
        self.raw_frames: list = []
        self.reply = self.success_reply
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(16)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @staticmethod
    def success_reply(frame: dict) -> bytes:
        body = json.dumps({"ok": True, "operation_id": frame["operation_id"]}).encode("utf-8")
        return shim.encode_worker_response(200, {"content-type": "application/json", "x-should-not-leak": "1"}, body)

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            with connection:
                connection.settimeout(5)
                try:
                    size = int.from_bytes(self._recv(connection, 4), "big")
                    data = self._recv(connection, size)
                except OSError:
                    continue
                self.raw_frames.append(data)
                frame = json.loads(data.decode("utf-8"))
                self.frames.append(frame)
                reply = self.reply(frame)
                if reply is not None:
                    connection.sendall(len(reply).to_bytes(4, "big") + reply)

    @staticmethod
    def _recv(connection: socket.socket, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                raise OSError("closed")
            data += chunk
        return data

    def close(self) -> None:
        self._server.close()
        shutil.rmtree(self.directory, ignore_errors=True)


@pytest.fixture
def worker(monkeypatch):
    fake = FakeWorker()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, fake.path)
    shim.install()
    try:
        yield fake
    finally:
        shim.uninstall()
        fake.close()
    assert urllib.request.urlopen is ORIGINALS["urlopen"]
    assert httpx.Client.send is ORIGINALS["httpx_send"]
    assert httpx.AsyncClient.send is ORIGINALS["httpx_async_send"]
    assert requests.Session.send is ORIGINALS["requests_send"]
    assert aiohttp.ClientSession._request is ORIGINALS["aiohttp_request"]


def assert_frame_is_minimal(frame: dict, raw: bytes, operation_id: str) -> None:
    assert set(frame) == set(shim.FRAME_FIELDS)
    assert frame["schema_version"] == shim.OPERATION_FRAME_SCHEMA_VERSION
    assert frame["operation_id"] == operation_id
    assert set(frame["parameters"]) <= set(operations.OPERATIONS[operation_id].request_fields)
    assert 1 <= frame["timeout_ms"] <= operations.OPERATIONS[operation_id].timeout_seconds * 1000
    text = raw.decode("utf-8").lower()
    operation = operations.OPERATIONS[operation_id]
    # No raw request URL, provider host, headers, credential, or identity
    # field; target URLs inside declared parameters (``url``/``urls``) are
    # validated parameters, not the request URL.
    forbidden = (
        operation.host,
        operation.path,
        '"authorization"',
        "bearer",
        "user-agent",
        '"headers"',
        '"host"',
        '"method"',
        "round_id",
        "lease",
        "miner",
        "sk-",
    )
    for marker in forbidden:
        assert marker not in text, marker
    # Frames are canonical JSON so the worker can hash them for call identity.
    assert raw == contracts.canonical_json(frame).encode("utf-8")


# ---------------------------------------------------------------------------
# The four clients
# ---------------------------------------------------------------------------


def test_urllib_post_sends_one_minimal_frame(worker):
    request = urllib.request.Request(
        "https://code.deepline.com/api/v2/integrations/exa_search/execute",
        data=json.dumps({"payload": {"query": "fintech"}}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "model/1.0"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"ok": True, "operation_id": "deepline.execute"}
        assert response.headers["content-type"] == "application/json"
        assert response.headers.get("x-should-not-leak") is None
    assert len(worker.frames) == 1
    assert_frame_is_minimal(worker.frames[0], worker.raw_frames[0], "deepline.execute")
    assert worker.frames[0]["parameters"] == {"tool": "exa_search", "payload": {"query": "fintech"}}
    assert worker.frames[0]["timeout_ms"] == 5000


def test_urllib_error_status_raises_http_error(worker):
    worker.reply = lambda frame: shim.encode_worker_response(429, {"content-type": "application/json"}, b'{"error": "slow down"}')
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen("https://api.scrapingdog.com/google?query=acme")
    assert excinfo.value.code == 429
    assert excinfo.value.read() == b'{"error": "slow down"}'
    assert len(worker.frames) == 1


def test_requests_post_and_get(worker):
    response = requests.post("https://code.deepline.com/api/v2/integrations/exa_contents/execute", json={"payload": {"urls": ["https://example.com/"], "text": True}}, timeout=3)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "operation_id": "deepline.execute"}
    assert "x-should-not-leak" not in response.headers
    response = requests.get("https://api.scrapingdog.com/scrape", params={"url": "https://example.com/jobs?page=2", "dynamic": "true"}, timeout=(2, 4))
    assert response.ok
    assert len(worker.frames) == 2
    assert_frame_is_minimal(worker.frames[0], worker.raw_frames[0], "deepline.execute")
    assert worker.frames[0]["parameters"] == {"tool": "exa_contents", "payload": {"urls": ["https://example.com/"], "text": True}}
    assert_frame_is_minimal(worker.frames[1], worker.raw_frames[1], "scrapingdog.scrape")
    assert worker.frames[1]["parameters"] == {"url": "https://example.com/jobs?page=2", "dynamic": True}
    assert worker.frames[1]["timeout_ms"] == 4000


def test_httpx_sync_and_async(worker):
    with httpx.Client(timeout=7) as client:
        response = client.get("https://api.scrapingdog.com/google", params={"query": "acme corp"})
        assert response.status_code == 200
        assert response.json()["operation_id"] == "scrapingdog.google"

    async def run_async():
        async with httpx.AsyncClient() as client:
            return await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    response = asyncio.run(run_async())
    assert response.status_code == 200
    assert response.json()["operation_id"] == "openrouter.chat"
    assert len(worker.frames) == 2
    assert_frame_is_minimal(worker.frames[0], worker.raw_frames[0], "scrapingdog.google")
    assert worker.frames[0]["timeout_ms"] == 7000
    assert_frame_is_minimal(worker.frames[1], worker.raw_frames[1], "openrouter.chat")
    assert worker.frames[1]["parameters"]["max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS


def test_httpx_accepts_baseline_unix_socket_urls_for_closed_provider_operations(worker):
    transport = httpx.HTTPTransport(uds=worker.path)
    with httpx.Client(transport=transport) as client:
        response = client.post(
            "http://code.deepline.com/api/v2/integrations/hunter_discover/execute",
            json={"payload": {"domain": "example.com"}},
        )
        assert response.status_code == 200
        response = client.get(
            "http://api.scrapingdog.com/google_news",
            params={"query": "example launch"},
        )
        assert response.status_code == 200

    async def run_async():
        transport = httpx.AsyncHTTPTransport(uds=worker.path)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post(
                "http://openrouter.ai/api/v1/chat/completions",
                json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    response = asyncio.run(run_async())
    assert response.status_code == 200
    assert [frame["operation_id"] for frame in worker.frames] == [
        "deepline.execute",
        "scrapingdog.google_news",
        "openrouter.chat",
    ]
    assert worker.frames[0]["parameters"] == {"tool": "hunter_discover", "payload": {"domain": "example.com"}}
    assert worker.frames[1]["parameters"] == {"query": "example launch", "country": "us"}


def test_aiohttp_get_and_post(worker):
    async def run():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=9)) as session:
            async with session.get("https://api.scrapingdog.com/scrape", params={"url": "https://example.com/"}) as response:
                assert response.status == 200
                assert response.ok
                assert (await response.json())["operation_id"] == "scrapingdog.scrape"
                assert "x-should-not-leak" not in response.headers
            async with session.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "x", "includeDomains": ["Example.com"]}}) as response:
                assert (await response.text()).startswith("{")
                response.raise_for_status()

    asyncio.run(run())
    assert len(worker.frames) == 2
    assert_frame_is_minimal(worker.frames[0], worker.raw_frames[0], "scrapingdog.scrape")
    assert worker.frames[0]["timeout_ms"] == 9000
    assert_frame_is_minimal(worker.frames[1], worker.raw_frames[1], "deepline.execute")
    # Tool payloads are opaque: passed through byte-for-byte, never normalized.
    assert worker.frames[1]["parameters"] == {"tool": "exa_search", "payload": {"query": "x", "includeDomains": ["Example.com"]}}


def test_aiohttp_error_status_raise_for_status(worker):
    worker.reply = lambda frame: shim.encode_worker_response(500, {"content-type": "application/json"}, b"{}")

    async def run():
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.scrapingdog.com/google", params={"query": "x"}) as response:
                assert response.status == 500
                with pytest.raises(aiohttp.ClientResponseError):
                    response.raise_for_status()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Nothing else is reachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com/",
        "http://127.0.0.1:8080/",
        "http://localhost/",
        "http://api.scrapingdog.com:81/google?query=acme",
        "https://code.deepline.com/api/v2/integrations/exa_search/execute/extra",
        "https://code.deepline.com:8443/api/v2/integrations/exa_search/execute",
        "https://user@code.deepline.com/api/v2/integrations/exa_search/execute",
        "https://code.deepline.com/api/v2/integrations/execute",
        "https://openrouter.ai/api/v1/models",
    ],
)
def test_unknown_targets_fail_inside_every_client_without_a_frame(worker, url):
    with pytest.raises(urllib.error.URLError) as urllib_error:
        urllib.request.urlopen(url)
    assert "no_matching_operation" in str(urllib_error.value)
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get(url, timeout=2)
    with pytest.raises(httpx.ConnectError):
        httpx.get(url)

    async def run():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientConnectionError):
                await session.get(url)

    asyncio.run(run())
    assert worker.frames == []


def test_credential_headers_and_unknown_fields_never_reach_the_worker(worker):
    with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
        requests.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "x"}}, headers={"Authorization": "Bearer sk-live-secret"})
    assert "forbidden_header" in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value)
    with pytest.raises(httpx.ConnectError) as httpx_error:
        httpx.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "x"}, "numResults": 100})
    assert "unknown_field" in str(httpx_error.value)
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(urllib.request.Request("https://code.deepline.com/api/v2/integrations/exa_search/execute", data=b"not json", method="POST"))
    assert worker.frames == []


def test_worker_error_frame_surfaces_as_generic_client_error(worker):
    worker.reply = lambda frame: shim.encode_worker_error("budget_exhausted")
    with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
        requests.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "x"}})
    assert str(excinfo.value) == "lab arena: budget_exhausted"
    worker.reply = lambda frame: shim.encode_worker_error("bad code with details sk-live")
    with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
        requests.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "x"}})
    assert str(excinfo.value) == "lab arena: invalid_response"


def test_missing_socket_fails_closed_without_network(monkeypatch):
    monkeypatch.delenv(shim.WORKER_SOCKET_ENV, raising=False)
    shim.install()
    try:
        with pytest.raises(urllib.error.URLError) as excinfo:
            urllib.request.urlopen("https://api.scrapingdog.com/google?query=x")
        assert "socket_unavailable" in str(excinfo.value)
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, "/nonexistent/lab-arena/worker.sock")
        with pytest.raises(httpx.ConnectError) as httpx_error:
            httpx.get("https://api.scrapingdog.com/google", params={"query": "x"})
        assert "worker_unavailable" in str(httpx_error.value)
    finally:
        shim.uninstall()


def test_malformed_worker_responses_are_rejected(worker):
    for reply in (
        b"{",
        json.dumps({"status": 200, "headers": {}, "body_b64": "!!"}).encode(),
        json.dumps({"status": 200, "headers": {}, "body_b64": "", "extra": 1}).encode(),
        json.dumps({"status": 999, "headers": {}, "body_b64": ""}).encode(),
        json.dumps({"status": 200, "headers": [], "body_b64": ""}).encode(),
        json.dumps({"status": 200, "body_b64": ""}).encode(),
    ):
        worker.reply = lambda frame, reply=reply: reply
        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            requests.get("https://api.scrapingdog.com/google", params={"query": "x"})
        assert "invalid_response" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Frame validator (worker side)
# ---------------------------------------------------------------------------


def _frame(**overrides):
    frame = {
        "schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION,
        "operation_id": "deepline.execute",
        "parameters": {"tool": "exa_search", "payload": {"query": "x"}},
        "timeout_ms": 1000,
    }
    frame.update(overrides)
    return frame


def test_validate_operation_frame_accepts_the_canonical_frame():
    assert shim.validate_operation_frame(_frame()) == ("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}}, 1000)
    encoded = shim.build_operation_frame("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}}, 999_999)
    assert shim.decode_operation_frame(encoded)[2] == operations.OPERATIONS["deepline.execute"].timeout_seconds * 1000


@pytest.mark.parametrize(
    "frame",
    [
        {**_frame(), "round_id": "arena-2026-09-02"},
        {**_frame(), "lease_token": "abc"},
        {**_frame(), "miner": "5F3sa2TJAWMqDhXG6jhV4N8ko9SKwHbcjkMLsg1JZbAf"},
        {**_frame(), "headers": {"Authorization": "x"}},
        {**_frame(), "url": "https://code.deepline.com/api/v2/integrations/exa_search/execute"},
        {k: v for k, v in _frame().items() if k != "timeout_ms"},
        _frame(schema_version="leadpoet.lab_arena.operation_frame.v0"),
        _frame(timeout_ms=0),
        _frame(timeout_ms=True),
        _frame(timeout_ms="1000"),
        _frame(timeout_ms=operations.OPERATIONS["deepline.execute"].timeout_seconds * 1000 + 1),
        "not a mapping",
        [],
    ],
)
def test_validate_operation_frame_rejects_extra_missing_or_bad_envelope(frame):
    with pytest.raises(shim.OperationFrameError) as excinfo:
        shim.validate_operation_frame(frame)
    assert excinfo.value.code == "invalid_frame"


def test_validate_operation_frame_rejects_unknown_operation_and_bad_parameters():
    with pytest.raises(shim.OperationFrameError) as excinfo:
        shim.validate_operation_frame(_frame(operation_id="deepline.play"))
    assert excinfo.value.code == "no_matching_operation"
    with pytest.raises(operations.OperationRequestError) as request_error:
        shim.validate_operation_frame(_frame(parameters={"tool": "exa_search", "payload": {"query": "x"}, "url": "https://evil.example/"}))
    assert request_error.value.code == "forbidden_field"
    with pytest.raises(operations.OperationRequestError):
        shim.validate_operation_frame(_frame(parameters={"tool": "exa_search", "payload": {"query": "x"}, "numResults": 100}))
    with pytest.raises(operations.OperationRequestError) as nested:
        shim.validate_operation_frame(_frame(parameters={"tool": "exa_search", "payload": {"query": "x", "headers": {"x-api-key": "k"}}}))
    assert nested.value.code == "forbidden_field"
    with pytest.raises(shim.OperationFrameError) as too_large:
        shim.decode_operation_frame(b"{" + b" " * shim.MAX_FRAME_BYTES + b"}")
    assert too_large.value.code == "frame_too_large"
    with pytest.raises(shim.OperationFrameError):
        shim.decode_operation_frame(b"\xff")


# ---------------------------------------------------------------------------
# Install contract
# ---------------------------------------------------------------------------


def test_install_is_idempotent_and_uninstall_restores_originals():
    assert not shim.installed()
    shim.install()
    try:
        patched = urllib.request.urlopen
        shim.install()
        assert urllib.request.urlopen is patched
        assert shim.installed()
        assert httpx.Client.send is not ORIGINALS["httpx_send"]
        assert requests.Session.send is not ORIGINALS["requests_send"]
        assert aiohttp.ClientSession._request is not ORIGINALS["aiohttp_request"]
    finally:
        shim.uninstall()
        shim.uninstall()
    assert not shim.installed()
    assert urllib.request.urlopen is ORIGINALS["urlopen"]
    assert httpx.Client.send is ORIGINALS["httpx_send"]
    assert httpx.AsyncClient.send is ORIGINALS["httpx_async_send"]
    assert requests.Session.send is ORIGINALS["requests_send"]
    assert aiohttp.ClientSession._request is ORIGINALS["aiohttp_request"]


def test_sitecustomize_source_and_shim_identity():
    compile(shim.SITECUSTOMIZE_SOURCE, "sitecustomize.py", "exec")
    assert "lab_arena.shim" in shim.SITECUSTOMIZE_SOURCE and "install()" in shim.SITECUSTOMIZE_SOURCE
    for module_path in shim.SHIM_IMAGE_MODULES:
        assert (contracts.__file__ and os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(contracts.__file__)), module_path)))


def test_shim_import_closure_is_pure(tmp_path):
    import subprocess
    import sys

    script = (
        "import sys; import lab_arena.shim; "
        "bad = sorted(m for m in sys.modules if m.startswith(('gateway', 'research_lab', 'lab_arena.signing', 'lab_arena.runtime'))); "
        "print(bad)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(contracts.__file__))),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.strip() == "[]"


def test_encode_helpers_round_trip():
    encoded = shim.encode_worker_response(201, {"content-type": "text/plain"}, b"hello")
    status, headers, body = shim.parse_worker_response(json.loads(encoded))
    assert (status, body) == (201, b"hello")
    assert headers == {"content-type": "text/plain", "content-length": "5"}
    assert base64.b64decode(json.loads(encoded)["body_b64"]) == b"hello"
    with pytest.raises(shim.ShimProviderError) as excinfo:
        shim.parse_worker_response(json.loads(shim.encode_worker_error("stage_closed")))
    assert excinfo.value.code == "stage_closed"
