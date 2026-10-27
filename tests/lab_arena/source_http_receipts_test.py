"""The reviewed HTML fallback keeps receipts on both worker transports."""
import base64
import hashlib
import http.client
import shutil
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import pytest

from lab_arena import runner, scoring_provider_compat as compat, shim
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.typed_refusal_transport_test import IDENTITY, _compatibility_call


@pytest.mark.parametrize("version", [compat.COMPATIBILITY_VERSION, compat.SOURCE_HTTP_COMPATIBILITY_VERSION])
@pytest.mark.parametrize("protocol", ["http", "frame"])
def test_source_html_response_keeps_exact_delivery_receipt(monkeypatch, version, protocol):
    body = b"<html><body>Verified source text.</body></html>"
    call = {**_compatibility_call(), "compatibility_version": version,
            "outcome": "settled", "error_code": None}
    document = {"status": 200, "headers": {"content-type": "text/html"},
                "body_b64": base64.b64encode(body).decode(), "call": call}
    state = runner.RunState(lease={**lease(), "kind": "score"}, lease_token="test-token")
    api = FakeApi([], broker_documents=[document])
    directory = Path(tempfile.mkdtemp(prefix="ar-receipt-", dir="/tmp"))
    socket_path = directory / "w.sock"
    server = runner.WorkerSocketServer(socket_path, api, state)
    server.start()
    try:
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
        if protocol == "frame":
            status, headers, returned = shim.dispatch(
                "scrapingdog.scrape", {"url": "https://example.com/"}, 1000)
        else:
            connection = http.client.HTTPConnection("api.scrapingdog.com", timeout=3)
            connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.sock.settimeout(3)
            connection.sock.connect(str(socket_path))
            try:
                connection.request("GET", "/scrape?" + urlencode({"url": "https://example.com/"}))
                response = connection.getresponse()
                status, headers, returned = response.status, dict(response.getheaders()), response.read()
            finally:
                connection.close()
        assert (status, returned) == (200, body)
        assert {key.lower(): value for key, value in headers.items()}[shim.CALL_IDENTITY_HEADER] == IDENTITY
    finally:
        server.stop()
        shutil.rmtree(directory)
    payload = state.worker_observations.document()
    assert payload["complete"] and payload["failures"] == []
    assert payload["observations"] == []  # The retained ledger body is not duplicated.
    assert len(api.provider_frames) == len(state.calls) == payload["response_receipt_count"] == 1
    receipt = payload["response_receipts"][0]
    assert receipt["write_state"] == "socket_written" and receipt["socket_written_at"]
    assert receipt["call_identity"] == IDENTITY and receipt["protocol"] == protocol
    assert receipt["body_sha256"] == "sha256:" + hashlib.sha256(body).hexdigest()
    assert receipt["body_bytes"] == len(body)


@pytest.mark.parametrize("changes", [
    {"adapter": "firecrawl_raw_html", "tool": compat.FIRECRAWL_TOOL},
    {"adapter": "generic_ats_json:ashby"},
    {"adapter": "unknown"}, {"tool": "unknown"},
    {"effective_operation_id": "openrouter.chat"},
    {"provider": "openrouter"}, {"compatibility_version": "unknown"},
])
def test_http_v2_does_not_authorize_other_provider_routes(changes):
    call = {**_compatibility_call(), "compatibility_version": compat.SOURCE_HTTP_COMPATIBILITY_VERSION,
            **changes}
    assert not runner._call_provider_matches_operation(call, "scrapingdog.scrape")
