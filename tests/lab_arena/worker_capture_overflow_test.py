"""Capture makes bounded progress without a ninth provider dispatch."""

import base64
import json
import socket
import socketserver
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, lab_arena_checkpoint as checkpoint, runner, shim
from tests.lab_arena.test_lab_arena_runner import lease
from tests.lab_arena.worker_final_capture_drain_test import _request_socket


class PausedApi:
    def __init__(self):
        self.condition = threading.Condition()
        self.calls = 0
        self.release = threading.Event()

    def provider(self, _run_id, _token, frame):
        with self.condition:
            self.calls += 1
            self.condition.notify_all()
        assert self.release.wait(8)
        return {
            "status": 200, "headers": {},
            "body_b64": base64.b64encode(b'{"choices":[]}').decode(),
            "call": {"operation_id": "openrouter.chat", "provider": "openrouter",
                     "action_sequence": frame["action_sequence"],
                     "call_identity": contracts.hash_bytes(str(frame["action_sequence"]).encode()),
                     "outcome": "settled", "status": 200, "provider_status": 200},
        }

    def wait_for_calls(self, count):
        with self.condition:
            assert self.condition.wait_for(lambda: self.calls == count, timeout=2)


def wait_for_handlers(state, count):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with state.lock:
            if state.worker_handlers_inflight == count:
                return
        time.sleep(0.005)
    assert state.worker_handlers_inflight == count


@pytest.fixture
def busy(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="ar-", dir="/tmp") as directory:
        api = PausedApi()
        state = runner.RunState(lease=lease(), lease_token="test-token")
        server = runner.WorkerSocketServer(Path(directory) / "w.sock", api, state)
        server.start()
        monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(server._path))
        connections = []

        def fill():
            before = api.calls
            api.release.clear()
            for number in range(8):
                connections.append(_request_socket(server._path))
                api.wait_for_calls(before + number + 1)
            wait_for_handlers(state, 8)

        def finish():
            api.release.set()
            for connection in connections:
                response = connection.getresponse()
                assert response.status == 200
                assert response.read() == b'{"choices":[]}'
                connection.close()
            connections.clear()
            wait_for_handlers(state, 0)

        try:
            fill()
            yield SimpleNamespace(api=api, state=state, server=server, fill=fill, finish=finish)
        finally:
            try:
                finish()
            finally:
                server.stop()
                assert server.wait_for_handlers()


def capture_web(*, retry=True):
    return checkpoint.record_web_fetch(
        url="https://example.test/page", final_url="https://example.test/page",
        status=200, content_type="text/html", body="retained web evidence",
        _retry_ambiguous=retry,
    )


def test_capture_progresses_with_eight_busy_provider_handlers(busy):
    assert busy.api.calls == busy.state.action_sequence == 8
    assert capture_web() == 0
    wait_for_handlers(busy.state, 8)
    assert busy.api.calls == busy.state.action_sequence == 8
    assert busy.state.worker_observations.agent_observations[0]["data"]["body"] == "retained web evidence"
    busy.finish()
    busy.fill()
    assert busy.api.calls == busy.state.action_sequence == 16
    assert capture_web() == 1
    busy.finish()
    assert len(busy.state.worker_observations.response_receipts) == 16
    assert all(row["write_state"] == "socket_written"
               for row in busy.state.worker_observations.response_receipts)


def connect(busy, payload=b""):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(2)
    connection.connect(str(busy.server._path))
    if payload:
        connection.sendall(payload)
    return connection


def assert_closed(connection):
    try:
        assert connection.recv(4) == b""
    except ConnectionResetError:
        pass


def training_frame(raw):
    return (len(raw) | 0x80000000).to_bytes(4, "big") + raw


def read_reply(connection):
    size = int.from_bytes(runner._recv_exact(connection, 4), "big")
    return json.loads(runner._recv_exact(connection, size))


@pytest.mark.parametrize("protocol", ["http", "frame"])
def test_overflow_never_dispatches_a_ninth_provider(busy, protocol):
    raw = (b"POST /api/v1/chat/completions HTTP/1.1\r\nHost: openrouter.ai\r\n\r\n"
           if protocol == "http" else shim.build_operation_frame(
               "openrouter.chat", {"model": "openai/gpt-4o-mini",
                                   "messages": [{"role": "user", "content": "must not dispatch"}],
                                   "max_tokens": 8}, 1000))
    payload = raw if protocol == "http" else len(raw).to_bytes(4, "big") + raw
    with connect(busy, payload) as connection:
        assert_closed(connection)
    wait_for_handlers(busy.state, 8)
    assert busy.api.calls == busy.state.action_sequence == 8
    assert capture_web() == 0


def test_overflow_cannot_become_provider_admission_after_capacity_returns(busy):
    with connect(busy) as connection:
        wait_for_handlers(busy.state, 9)
        busy.api.release.set()
        wait_for_handlers(busy.state, 1)
        raw = shim.build_operation_frame(
            "openrouter.chat", {"model": "openai/gpt-4o-mini",
                                "messages": [{"role": "user", "content": "still refused"}],
                                "max_tokens": 8}, 1000)
        connection.sendall(len(raw).to_bytes(4, "big") + raw)
        assert_closed(connection)
    wait_for_handlers(busy.state, 0)
    assert busy.api.calls == busy.state.action_sequence == 8


@pytest.mark.parametrize("high_bit", [True, False])
def test_overflow_cannot_reach_quota_or_other_control_dispatch(busy, monkeypatch, high_bit):
    def forbidden(*args, **kwargs):
        pytest.fail("capture-only admission entered normal control dispatch")

    monkeypatch.setattr(busy.server, "handle_frame", forbidden)
    monkeypatch.setattr(busy.server, "_handle_quota_control", forbidden)
    raw = json.dumps(checkpoint.QUOTA_CONTROL_FRAME).encode()
    payload = training_frame(raw) if high_bit else len(raw).to_bytes(4, "big") + raw
    with connect(busy, payload) as connection:
        if high_bit:
            assert read_reply(connection) == {"recorded": False}
        else:
            assert_closed(connection)
    wait_for_handlers(busy.state, 8)
    assert busy.api.calls == busy.state.action_sequence == 8


@pytest.mark.parametrize("raw", [b"[]", b"not-json", json.dumps({
    "schema_version": checkpoint.TRAINING_SCHEMA_VERSION,
    "control": checkpoint.TRAINING_CONTROL, "kind": "web_fetch",
    "depends_on": [], "data": {"body": "missing required metadata"},
}).encode()])
def test_malformed_capture_keeps_validation_and_releases_slot(busy, raw):
    with connect(busy, training_frame(raw)) as connection:
        assert read_reply(connection) == {"recorded": False}
    wait_for_handlers(busy.state, 8)
    assert busy.api.calls == busy.state.action_sequence == 8
    assert not busy.state.worker_observations.agent_observations
    assert capture_web() == 0


def test_oversized_capture_is_rejected_before_body_read(busy):
    size = checkpoint.MAX_TRAINING_FRAME_BYTES + 1
    with connect(busy, (size | 0x80000000).to_bytes(4, "big")) as connection:
        assert read_reply(connection) == {"error": "frame_too_large"}
    wait_for_handlers(busy.state, 8)
    assert busy.state.training_control_frames == busy.state.training_control_bytes == 0
    assert busy.api.calls == busy.state.action_sequence == 8
    assert capture_web() == 0


@pytest.mark.parametrize("limit", ["frames", "bytes"])
def test_overflow_preserves_aggregate_capture_caps(busy, limit):
    name, maximum = (("training_control_frames", runner.MAX_TRAINING_CONTROL_FRAMES)
                     if limit == "frames" else
                     ("training_control_bytes", runner.MAX_TRAINING_CONTROL_BYTES))
    setattr(busy.state, name, maximum)
    assert capture_web() is None
    assert not busy.state.worker_observations.agent_observations
    assert getattr(busy.state, name) == maximum
    assert busy.api.calls == busy.state.action_sequence == 8


@pytest.mark.parametrize("payload", [b"", b"\x80", training_frame(b"{}")[0:4] + b"{"])
def test_idle_or_partial_overflow_has_one_second_read_bound(busy, payload):
    before = time.monotonic()
    with connect(busy, payload) as connection:
        assert_closed(connection)
    elapsed = time.monotonic() - before
    assert 0.8 <= elapsed < 1.8
    wait_for_handlers(busy.state, 8)
    assert capture_web() == 0
    assert busy.api.calls == busy.state.action_sequence == 8


def test_slow_bytes_cannot_extend_overflow_read_deadline(busy):
    with connect(busy, (32 | 0x80000000).to_bytes(4, "big")) as connection:
        before = time.monotonic()
        for _ in range(4):
            connection.sendall(b" ")
            time.sleep(0.2)
        assert_closed(connection)
        assert time.monotonic() - before < 1.6
    wait_for_handlers(busy.state, 8)
    assert capture_web() == 0


def test_only_one_overflow_handler_is_admitted(busy):
    with connect(busy, (2 | 0x80000000).to_bytes(4, "big")):
        wait_for_handlers(busy.state, 9)
        assert capture_web() is None
        assert checkpoint._training_failure_reason() == "transport_error"
        assert busy.state.worker_handlers_inflight == 9
        assert busy.api.calls == busy.state.action_sequence == 8
    wait_for_handlers(busy.state, 8)
    assert capture_web() == 0


def test_score_lease_cannot_use_capture_overflow(busy):
    busy.state.lease["kind"] = "score"
    assert capture_web() is None
    assert not busy.state.worker_observations.agent_observations
    assert busy.api.calls == busy.state.action_sequence == 8


@pytest.mark.parametrize("saturated", [True, False])
def test_thread_start_failure_releases_the_owned_permit(busy, monkeypatch, saturated):
    if not saturated:
        busy.finish()
    original = socketserver.ThreadingMixIn.process_request
    fail = [True]
    errors = []

    def fail_once(self, request, address):
        if fail[0]:
            fail[0] = False
            raise RuntimeError("test thread start failure")
        return original(self, request, address)

    monkeypatch.setattr(socketserver.ThreadingMixIn, "process_request", fail_once)
    monkeypatch.setattr(busy.server._server, "handle_error", lambda *args: errors.append(True))
    assert capture_web(retry=False) is None
    wait_for_handlers(busy.state, 8 if saturated else 0)
    assert errors == [True]
    if not saturated:
        busy.fill()  # All eight regular permits must still be available.
    assert capture_web() == 0


def test_capture_handler_exception_releases_overflow(busy, monkeypatch):
    original = busy.server._handle_training_control
    errors = []

    def fail(_raw):
        raise RuntimeError("test capture handler failure")

    monkeypatch.setattr(busy.server._server, "handle_error", lambda *args: errors.append(True))
    monkeypatch.setattr(busy.server, "_handle_training_control", fail)
    assert capture_web(retry=False) is None
    wait_for_handlers(busy.state, 8)
    assert errors == [True]
    monkeypatch.setattr(busy.server, "_handle_training_control", original)
    assert capture_web() == 0


def test_disconnected_capture_writer_releases_overflow(busy, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = busy.server._handle_training_control

    def pause(raw):
        entered.set()
        assert release.wait(2)
        return original(raw)

    monkeypatch.setattr(busy.server, "_handle_training_control", pause)
    raw = json.dumps({"schema_version": checkpoint.TRAINING_SCHEMA_VERSION,
                      "control": checkpoint.TRAINING_CONTROL, "kind": "state_snapshot",
                      "depends_on": [], "data": {"name": "test", "value": 1}}).encode()
    connection = connect(busy, training_frame(raw))
    try:
        assert entered.wait(1)
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        release.set()
        wait_for_handlers(busy.state, 8)
        assert capture_web() == 1
        assert busy.api.calls == busy.state.action_sequence == 8
    finally:
        release.set()
        connection.close()


def test_shutdown_drains_started_overflow_capture(busy, monkeypatch):
    original = busy.server._handle_training_control
    entered, release = threading.Event(), threading.Event()
    result, drained = [], []

    def pause(raw):
        entered.set()
        assert release.wait(3)
        return original(raw)

    monkeypatch.setattr(busy.server, "_handle_training_control", pause)
    client = threading.Thread(target=lambda: result.append(capture_web()))
    client.start()
    assert entered.wait(2)
    assert busy.state.worker_handlers_inflight == 9
    try:
        busy.server.stop()
        busy.api.release.set()
        wait_for_handlers(busy.state, 1)
        waiter = threading.Thread(target=lambda: drained.append(busy.server.wait_for_handlers()))
        waiter.start()
        release.set()
        client.join(2)
        waiter.join(2)
        assert not client.is_alive() and not waiter.is_alive()
        assert result == [0] and drained == [True]
        assert busy.state.worker_observations.document()["complete"]
    finally:
        release.set()
        client.join(2)
