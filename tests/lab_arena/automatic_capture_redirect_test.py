"""Consumed HTTP responses retain private-safe redirect links without extra reads."""
import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lab_arena import automatic_capture, lab_arena_checkpoint as checkpoint, runner
from lab_arena import training_observations as observations
from tests.lab_arena.automatic_capture_test import Checkpoint, endpoint, finish
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.training_input_test import CLAIM_HASH


LOCATION = "https://name:password@other.example/next?token=private#fragment"
TARGET = "https://other.example/next"


@pytest.fixture
def redirect_endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            binary = self.path == "/binary"
            data = b"\x00\xff\x10remaining" if binary else b"moved to another page"
            self.send_response(302)
            self.send_header("Location", LOCATION)
            self.send_header("Content-Type", "application/octet-stream" if binary else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


@pytest.mark.parametrize("client", ["requests", "httpx", "aiohttp"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_consumed_redirect_retains_link_without_changing_response(redirect_endpoint, tmp_path, client, binary, partial):
    source = redirect_endpoint + ("/binary" if binary else "/text")
    expected = b"\x00\xff\x10remaining" if binary else b"moved to another page"
    cp = Checkpoint()
    recorder = automatic_capture.install(cp)

    async def asynchronous():
        aiohttp = pytest.importorskip("aiohttp")
        async with aiohttp.ClientSession() as session:
            async with session.get(source, allow_redirects=False) as response:
                assert response.status == 302
                assert response.headers["Location"] == LOCATION
                return await response.content.readexactly(3) if partial else await response.read()

    try:
        if client == "requests":
            requests = pytest.importorskip("requests")
            # Session.send consumes redirect bodies while preparing response.next,
            # even with redirects disabled. The adapter preserves a partial read.
            adapter = requests.adapters.HTTPAdapter()
            with adapter.send(requests.Request("GET", source).prepare(), stream=True, timeout=3) as response:
                assert response.status_code == 302
                assert response.headers["Location"] == LOCATION
                returned = response.raw.read(3) if partial else response.content
            adapter.close()
        elif client == "httpx":
            httpx = pytest.importorskip("httpx")
            with httpx.Client() as session:
                with session.stream("GET", source) as response:
                    assert response.status_code == 302
                    assert response.headers["Location"] == LOCATION
                    returned = next(response.iter_bytes(chunk_size=3)) if partial else response.read()
        else:
            returned = asyncio.run(asynchronous())
    finally:
        summary = finish(recorder, cp)
    assert returned == (expected[:3] if partial else expected)
    events = cp.binary if binary else cp.web
    assert len(events) == 1, summary
    event = events[0]
    assert event["url"] == event["final_url"] == source
    assert event["capture"]["redirect_url"] == TARGET
    assert event["capture"]["complete"] is not partial
    assert event["body"] == (returned if binary else returned.decode())
    assert cp.acks == cp.metadata == []
    # The real frame and host snapshot validators must accept the new metadata.
    data = {key: value for key, value in event.items() if not key.startswith("_")}
    if binary:
        import base64
        data["body_b64"] = base64.b64encode(data.pop("body")).decode("ascii")
    frame = {"schema_version": checkpoint.TRAINING_SCHEMA_VERSION,
             "control": checkpoint.TRAINING_CONTROL, "kind": "web_fetch_binary" if binary else "web_fetch",
             "depends_on": [], "data": data}
    checkpoint.validate_training_frame(frame)
    state = runner.RunState(lease=lease(), lease_token="private-lease-token")
    server = runner.WorkerSocketServer(tmp_path / "w.sock", FakeApi([]), state)
    assert json.loads(server._handle_training_control(json.dumps(frame).encode()))["recorded"] is True
    payload = state.worker_observations.document()
    saved = payload["agent_observations"][0]
    assert saved["data"]["capture"]["redirect_url"] == TARGET
    serialized = json.dumps(saved)
    assert "password" not in serialized and "private" not in serialized and "name:" not in serialized
    snapshot, raw = observations.build_snapshot("r1", "execute", payload,
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    objects = SimpleNamespace(get_bounded=lambda ref, limit: raw)
    assert observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH) == snapshot
    segment_payload = {key: value for key, value in payload.items() if key != "complete"}
    segment_payload.update(observation_from=0, receipt_from=0, agent_from=0, receipt_updates=[])
    segment, raw_segment = observations.build_segment("r1", "execute", segment_payload,
        lease_generation=1, claim_request_hash=CLAIM_HASH, number=0,
        previous_sha256=None, cumulative_bytes=len(json.dumps(segment_payload).encode()))
    objects = SimpleNamespace(get_optional_bounded=lambda ref, limit: raw_segment)
    assert observations.read_segment(objects, "r1", "execute", 1, CLAIM_HASH, 0) == segment
    assert state.action_sequence == len(state.calls) == 0
    assert payload["response_receipts"] == []


@pytest.mark.parametrize("client", ["requests", "httpx", "aiohttp"])
def test_followed_redirect_retains_each_response_and_link(endpoint, client):
    url, server = endpoint
    cp = Checkpoint()
    recorder = automatic_capture.install(cp)

    async def asynchronous():
        aiohttp = pytest.importorskip("aiohttp")
        async with aiohttp.ClientSession() as session:
            async with session.get(url + "/redirect") as response:
                return await response.read()

    try:
        if client == "requests":
            requests = pytest.importorskip("requests")
            with requests.get(url + "/redirect", timeout=3) as response:
                returned = response.content
        elif client == "httpx":
            httpx = pytest.importorskip("httpx")
            with httpx.Client(follow_redirects=True) as session:
                returned = session.get(url + "/redirect").content
        else:
            returned = asyncio.run(asynchronous())
    finally:
        summary = finish(recorder, cp)
    assert returned == b'hello world\n{"value": 42}\n'
    assert server.paths == ["/redirect", "/text"]
    # aiohttp releases an unread redirect body; its existing header-only
    # observation remains explicit instead of inventing consumed bytes.
    first = next(event for event in cp.web + cp.metadata if event["status"] == 302)
    assert first.get("capture", first)["redirect_url"] == url + "/text"
    assert any(event["url"] == url + "/text" for event in cp.web)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("target", [
    "https://name:password@example.com/next", "https://example.com/next?token=private",
    "https://example.com/next#private", "https://example.com/next\nprivate",
    "https://example.com/" + "a" * 2000, "https://[broken", "/next", "file:///private", True,
])
def test_consumed_capture_rejects_private_invalid_or_unbounded_redirects(binary, target):
    capture = {"source": "runtime_http_v1", "complete": True,
               "received_body_sha256": "sha256:" + hashlib.sha256(b"body").hexdigest(),
               "received_body_bytes": 4, "redirect_url": target}
    if binary:
        validate = checkpoint.validate_binary_capture_metadata
    else:
        capture["encoding"] = "utf-8"
        validate = checkpoint.validate_web_capture_metadata
    with pytest.raises(ValueError):
        validate(capture)


@pytest.mark.parametrize("binary", [False, True])
def test_capture_metadata_without_redirect_keeps_its_original_shape(binary):
    capture = {"source": "runtime_http_v1", "complete": True,
               "received_body_sha256": "sha256:" + hashlib.sha256(b"body").hexdigest(),
               "received_body_bytes": 4}
    if binary:
        assert checkpoint.validate_binary_capture_metadata(capture, b"body") == capture
    else:
        capture["encoding"] = "utf-8"
        assert checkpoint.validate_web_capture_metadata(capture, "body") == capture
