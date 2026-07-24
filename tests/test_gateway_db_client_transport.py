from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from gateway.db import client as db_client


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


def test_shared_sync_client_handles_100_concurrent_postgrest_reads():
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
