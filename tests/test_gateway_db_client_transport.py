from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

from gateway.db import client as db_client


def test_db_client_import_does_not_construct_async_lock():
    """Import must work after Python 3.9 has cleared its current event loop."""
    repo_root = Path(__file__).resolve().parent.parent
    probe = """
import asyncio
import inspect

original_lock = asyncio.Lock

def guarded_lock(*args, **kwargs):
    caller = inspect.currentframe().f_back
    if caller.f_globals.get("__name__") == "gateway.db.client":
        raise AssertionError("gateway.db.client constructed an asyncio.Lock at import time")
    return original_lock(*args, **kwargs)

asyncio.Lock = guarded_lock
from gateway.db import client
assert client._async_lock is None
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_async_client_lock_is_created_lazily_and_serializes_initialization(
    monkeypatch,
):
    expected = SimpleNamespace()
    create_calls = 0

    async def create_async_client(_url, _key):
        nonlocal create_calls
        create_calls += 1
        await __import__("asyncio").sleep(0)
        return expected

    monkeypatch.setattr(db_client, "_async_read_client", None)
    monkeypatch.setattr(db_client, "_async_write_client", None)
    monkeypatch.setattr(db_client, "_async_lock", None)
    monkeypatch.setattr(db_client, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(db_client, "SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setattr(db_client, "create_async_client", create_async_client)
    monkeypatch.setattr(db_client, "_install_async_send_retry", lambda client: client)

    assert db_client._async_lock is None
    clients = await __import__("asyncio").gather(
        *(db_client.get_async_read_client() for _ in range(10))
    )

    assert clients == [expected] * 10
    assert create_calls == 1
    assert db_client._async_lock is not None


class _FakeHttpClient:
    instances: list["_FakeHttpClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def test_sync_supabase_client_disables_http2(monkeypatch):
    captured = {}
    expected = SimpleNamespace()

    def create_client(url, key, *, options):
        captured.update(url=url, key=key, options=options)
        return expected

    _FakeHttpClient.instances.clear()
    monkeypatch.setattr(db_client.httpx, "Client", _FakeHttpClient)
    monkeypatch.setattr(db_client, "create_client", create_client)

    result = db_client._create_sync_client("https://example.supabase.co", "key")

    assert result is expected
    assert captured["url"] == "https://example.supabase.co"
    assert captured["key"] == "key"
    assert len(_FakeHttpClient.instances) == 1
    transport = _FakeHttpClient.instances[0]
    assert transport.kwargs["http1"] is True
    assert transport.kwargs["http2"] is False
    assert transport.kwargs["follow_redirects"] is True
    assert captured["options"].httpx_client is transport


def test_sync_http_client_closes_when_supabase_creation_fails(monkeypatch):
    def create_client(_url, _key, *, options):
        assert options.httpx_client is _FakeHttpClient.instances[0]
        raise RuntimeError("client construction failed")

    _FakeHttpClient.instances.clear()
    monkeypatch.setattr(db_client.httpx, "Client", _FakeHttpClient)
    monkeypatch.setattr(db_client, "create_client", create_client)

    with pytest.raises(RuntimeError, match="client construction failed"):
        db_client._create_sync_client("https://example.supabase.co", "key")

    assert len(_FakeHttpClient.instances) == 1
    assert _FakeHttpClient.instances[0].closed is True


def test_http1_migration_factory_preserves_legacy_timeout(monkeypatch):
    expected = SimpleNamespace()

    _FakeHttpClient.instances.clear()
    monkeypatch.setattr(db_client.httpx, "Client", _FakeHttpClient)
    monkeypatch.setattr(
        db_client,
        "create_client",
        lambda _url, _key, *, options: expected,
    )

    assert (
        db_client.create_http1_sync_client(
            "https://example.supabase.co",
            "key",
        )
        is expected
    )
    timeout = _FakeHttpClient.instances[0].kwargs["timeout"]
    assert timeout.read == 120.0
    assert timeout.write == 120.0
    assert timeout.connect == 120.0
    assert timeout.pool == 120.0


def test_shared_sync_client_handles_100_concurrent_postgrest_reads():
    class ConcurrentTestServer(ThreadingHTTPServer):
        request_queue_size = 128

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            body = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", "*/0")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args) -> None:
            return

    server = ConcurrentTestServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = db_client._create_sync_client(
        f"http://127.0.0.1:{server.server_port}",
        "test-key",
    )

    try:
        def read_row(_index: int):
            return client.table("probe").select("*").execute().data

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(read_row, range(100)))
        assert results == [[]] * 100
    finally:
        client.postgrest.session.close()
        server.shutdown()
        server.server_close()
        server_thread.join()
