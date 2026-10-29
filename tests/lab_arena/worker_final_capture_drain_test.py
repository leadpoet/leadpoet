"""Final worker capture retains each admitted handler's original deadline."""

import base64
import http.client
import json
import socket
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, operations, runner, runtime, shim
from tests.lab_arena.test_lab_arena_runner import lease, valid_company
from tests.lab_arena.trajectory_runner_test import BridgingRuntime, LoggingApi, run_model


IDENTITY = "sha256:" + "1" * 64
BODY = b'{"id":"local-proof","choices":[]}'


class PausedApi:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.uploaded = []
        self.frames = []

    def provider(self, _run_id, _token, frame):
        self.frames.append(dict(frame))
        self.entered.set()
        assert self.release.wait(20)
        return {
            "status": 200, "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(BODY).decode("ascii"),
            "call": {"operation_id": "openrouter.chat", "provider": "openrouter",
                     "action_sequence": frame["action_sequence"],
                     "call_identity": IDENTITY, "outcome": "settled",
                     "status": 200, "provider_status": 200},
        }

    def training_observations(self, _run_id, _token, document):
        self.uploaded.append(document)
        return {"status": "stored", "payload_sha256": contracts.document_hash(document)}


@pytest.fixture
def short_dir():
    with tempfile.TemporaryDirectory(prefix="ar-", dir="/tmp") as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def small_fake_image_reserve(monkeypatch):
    # run_model uses an empty fake image exporter. Keep the real disk-space
    # guard active, with a test-only reserve that matches that exporter.
    monkeypatch.setattr(runner.images, "DEFAULT_MAX_ROOTFS_BYTES", 1024)
    monkeypatch.setattr(runner.images, "DEFAULT_MAX_IMAGE_BYTES", 1024)


def _start(short_dir, api, *, read_timeout=0.5):
    state = runner.RunState(lease=lease(), lease_token="local-test-token")
    server = runner.WorkerSocketServer(short_dir / "worker.sock", api, state,
                                       read_timeout_seconds=read_timeout)
    server.start()
    return server, state


def _request_socket(path):
    connection = http.client.HTTPConnection("openrouter.ai", timeout=3)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.settimeout(3)
    connection.sock.connect(str(path))
    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "local test"}],
               "max_tokens": 8}
    connection.request("POST", "/api/v1/chat/completions", body=json.dumps(payload),
                       headers={"content-type": "application/json"})
    return connection


def _capture(api, state):
    return runner._record_training_observations(
        SimpleNamespace(api=api), state.lease, state.lease_token, state,
    )


def _receive_exact(connection, size):
    body = bytearray()
    while len(body) < size:
        part = connection.recv(size - len(body))
        assert part
        body.extend(part)
    return bytes(body)


def _start_frame(short_dir, *, read_timeout, timeout_ms=1000):
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=read_timeout)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(3)
    connection.connect(str(server._path))
    raw = shim.build_operation_frame("openrouter.chat", {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "local frame test"}],
        "max_tokens": 8,
    }, timeout_ms)
    connection.sendall(len(raw).to_bytes(4, "big") + raw)
    assert api.entered.wait(3)
    server.stop()
    return api, server, state, connection


def _read_frame(connection):
    size = int.from_bytes(_receive_exact(connection, 4), "big")
    response = json.loads(_receive_exact(connection, size))
    assert response["status"] == 200
    assert base64.b64decode(response["body_b64"], validate=True) == BODY


def test_late_success_is_in_final_snapshot(short_dir):
    api = PausedApi()
    server, state = _start(short_dir, api)
    connection = _request_socket(server._path)
    assert api.entered.wait(3)
    try:
        server.stop()
        with state.lock:
            assert state.worker_handlers_inflight == 1
        timer = threading.Timer(0.1, api.release.set)
        timer.start()
        assert server.wait_for_handlers()
        timer.join(1)
        summary = _capture(api, state)
        response = connection.getresponse()
        assert response.status == 200 and response.read() == BODY
        assert summary["complete"] is True
        assert api.uploaded[0]["response_receipt_count"] == 1
        assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
    finally:
        api.release.set()
        connection.close()


@pytest.mark.parametrize("protocol,read_timeout,delay", [
    ("http", 0.05, 0.15),
    ("frame", 0.05, 0.15),
    ("http", 10.0, 10.1),
    ("http", 10.0, 3.0),
])
def test_provider_completion_uses_original_window_after_read_bound(
    short_dir, protocol, read_timeout, delay,
):
    if protocol == "frame":
        api, server, state, connection = _start_frame(short_dir, read_timeout=read_timeout)
    else:
        api = PausedApi()
        server, state = _start(short_dir, api, read_timeout=read_timeout)
        connection = _request_socket(server._path)
        assert api.entered.wait(3)
        server.stop()
    with state.lock:
        original_deadline = next(iter(server._admissions.values())).completion_deadline
    timer = threading.Timer(delay, api.release.set)
    timer.start()
    try:
        assert server.wait_for_handlers()
        summary = _capture(api, state)
        if protocol == "frame":
            _read_frame(connection)
        else:
            response = connection.getresponse()
            assert response.status == 200 and response.read() == BODY
        assert time.monotonic() < original_deadline
        assert summary["complete"] is True
        assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
        expected_timeout = (1000 if protocol == "frame"
                            else operations.get_operation("openrouter.chat").timeout_seconds * 1000)
        assert [(row["operation_id"], row["timeout_ms"], row["action_sequence"])
                for row in api.frames] == [("openrouter.chat", expected_timeout, 0)]
        assert api.frames[0]["parameters"]["model"] == "openai/gpt-4o-mini"
        assert len(state.calls) == state.action_sequence == 1
    finally:
        api.release.set()
        timer.join(1)
        connection.close()


def test_drain_does_not_reset_partly_elapsed_provider_window(short_dir, monkeypatch):
    monkeypatch.setattr(runner, "MAX_PROVIDER_API_TIMEOUT_SECONDS",
                        runner.MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS)
    api, server, state, connection = _start_frame(
        short_dir, read_timeout=0.05, timeout_ms=2000,
    )
    with state.lock:
        deadline = next(iter(server._admissions.values())).completion_deadline
    try:
        time.sleep(0.15)
        start = time.monotonic()
        remaining = deadline - start
        assert 0 < remaining < 1.9  # stop already used part of the original two-second window.
        assert server.wait_for_handlers() is False
        elapsed = time.monotonic() - start
        assert remaining - 0.02 <= elapsed <= remaining + 0.1
        assert server.wait_for_handlers() is False
        assert time.monotonic() - start <= remaining + 0.1
        assert _capture(api, state)["complete"] is False
        assert len(api.frames) == 1
    finally:
        api.release.set()
        connection.close()


def test_stop_waits_for_admitted_request_to_start_before_final_drain(short_dir, monkeypatch):
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=2.0)
    bounded_server = server._server
    admitted, start_handler = threading.Event(), threading.Event()
    handler_entered, finish_handler = threading.Event(), threading.Event()
    process = bounded_server.process_request
    handle_http = server.handle_http

    def pause_start(request, address):
        admitted.set()
        assert start_handler.wait(3)
        return process(request, address)

    def pause_handler(*args, **kwargs):
        handler_entered.set()
        assert finish_handler.wait(3)
        return handle_http(*args, **kwargs)

    monkeypatch.setattr(bounded_server, "process_request", pause_start)
    monkeypatch.setattr(server, "handle_http", pause_handler)
    connection = _request_socket(server._path)
    assert admitted.wait(2)
    stopper = threading.Thread(target=server.stop)
    stopper.start()
    try:
        assert server._stopping.wait(2)
        # stop's synchronous server shutdown joins the accept loop. It cannot
        # return before an already-admitted request reaches process_request.
        assert stopper.is_alive()
        start_handler.set()
        assert handler_entered.wait(2)
        stopper.join(2)
        assert not stopper.is_alive()
        with state.lock:
            assert state.worker_handlers_inflight == len(server._admissions) == 1
        finish_handler.set()
        assert server.wait_for_handlers()
        # Decoding may finish after sandbox exit. The existing stop behavior
        # blocks the first provider dispatch and keeps its original refusal.
        response = connection.getresponse()
        assert response.status == 503
        assert json.loads(response.read()) == {"error": {"code": "worker_unavailable"}}
        assert api.frames == [] and state.calls == []
        assert state.action_sequence == 1
        assert _capture(api, state)["complete"] is True
    finally:
        start_handler.set()
        finish_handler.set()
        stopper.join(2)
        connection.close()


@pytest.mark.parametrize("protocol", ["http", "frame"])
def test_handler_still_reading_cannot_gain_fresh_window_at_stop(short_dir, protocol):
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=0.2)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(server._path))
    # Incomplete headers or frame body must never reach provider dispatch.
    connection.sendall(b"POST /api/v1/chat/completions HTTP/1.1\r\n" if protocol == "http"
                       else (100).to_bytes(4, "big") + b"{")
    try:
        deadline = time.monotonic() + 2
        while state.worker_handlers_inflight != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert state.worker_handlers_inflight == 1
        with state.lock:
            original_deadline = next(iter(server._admissions.values())).completion_deadline
        server.stop()
        start = time.monotonic()
        remaining = max(0.0, original_deadline - start)
        # Cleanup may finish before or after its read deadline. Neither case
        # gives finalization a new read window or permits a provider dispatch.
        server.wait_for_handlers()
        assert time.monotonic() - start < remaining + 0.1
        assert api.frames == [] and state.action_sequence == 0
    finally:
        server.stop()
        connection.close()


def test_responses_hidden_retry_retains_original_absolute_lifetime(short_dir, monkeypatch):
    from tests.lab_arena.codex_rate_limit_retry_test import identified_document

    class RetryApi(PausedApi):
        def provider(self, _run_id, _token, frame):
            self.frames.append(dict(frame))
            if len(self.frames) == 1:
                return identified_document(frame["action_sequence"])
            self.entered.set()
            assert self.release.wait(3)
            return identified_document(frame["action_sequence"], status=200,
                                       provider_status=200, actual=7, error_code=None)

    monkeypatch.setattr(runner, "MAX_PROVIDER_API_TIMEOUT_SECONDS",
                        runner.MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS)
    monkeypatch.setattr(runner, "RESPONSES_RATE_LIMIT_MIN_PROVIDER_SECONDS", 0.001)
    monkeypatch.setattr(runner.WorkerSocketServer, "_responses_rate_limit_delay",
                        staticmethod(lambda _call, _retry: 0.1))
    retained_deadlines = []
    retain = runner.WorkerSocketServer._retain_handler_deadline

    def record_deadline(server, deadline, clock):
        retain(server, deadline, clock)
        retained_deadlines.append(server._handler_context.admission.provider_deadline)

    monkeypatch.setattr(runner.WorkerSocketServer, "_retain_handler_deadline", record_deadline)
    api = RetryApi()
    server, state = _start(short_dir, api, read_timeout=0.05)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(3)
    connection.connect(str(server._path))
    frame = shim.build_operation_frame("openrouter.responses", {
        "model": "openai/gpt-4o-mini", "input": "research",
    }, 1000)
    connection.sendall(len(frame).to_bytes(4, "big") + frame)
    assert api.entered.wait(2)
    with state.lock:
        admission = next(iter(server._admissions.values()))
        assert len(retained_deadlines) == 3  # Outer operation, first attempt, retry.
        assert all(deadline <= retained_deadlines[0] for deadline in retained_deadlines)
        original_deadline = admission.provider_deadline
    server.stop()
    timer = threading.Timer(0.15, api.release.set)
    timer.start()
    try:
        assert server.wait_for_handlers()
        assert time.monotonic() < original_deadline
        response = json.loads(_receive_exact(connection, int.from_bytes(_receive_exact(connection, 4), "big")))
        assert response["status"] == 200
        assert [frame["action_sequence"] for frame in api.frames] == [0, 1]
        assert api.frames[0]["timeout_ms"] == 1000
        assert 1 <= api.frames[1]["timeout_ms"] < 900
        assert len(state.calls) == 2
        assert state.recovered_responses_retry_call_identities == {state.calls[0]["call_identity"]}
        assert _capture(api, state)["complete"] is True
        assert len(api.uploaded[0]["response_receipts"]) == 1
        assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
    finally:
        api.release.set()
        timer.join(1)
        server.stop()
        connection.close()


@pytest.mark.parametrize("write_phase", ["frame", "http_headers", "http_body"])
@pytest.mark.parametrize("expire_write", [False, True])
def test_existing_write_window_survives_provider_deadline_or_freezes_incomplete(
    short_dir, monkeypatch, write_phase, expire_write,
):
    monkeypatch.setattr(runner, "MAX_PROVIDER_API_TIMEOUT_SECONDS",
                        runner.MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS)
    operation = operations.get_operation("openrouter.chat")
    monkeypatch.setattr(operations, "OPERATIONS", {
        **operations.OPERATIONS, "openrouter.chat": replace(operation, timeout_seconds=1),
    })
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=0.05 if expire_write else 3.0)
    write_entered, release_write = threading.Event(), threading.Event()
    sendall = socket.socket.sendall

    def paused_write(connection, payload, *args, **kwargs):
        if getattr(server._handler_context, "admission", None) is not None:
            headers = payload.startswith(b"HTTP/")
            selected = (write_phase == "frame" or
                        (write_phase == "http_headers" and headers) or
                        (write_phase == "http_body" and not headers))
            if selected and not write_entered.is_set():
                write_entered.set()
                assert release_write.wait(5)
        return sendall(connection, payload, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "sendall", paused_write)
    if write_phase == "frame":
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(3)
        connection.connect(str(server._path))
        raw = shim.build_operation_frame("openrouter.chat", {
            "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "write proof"}],
            "max_tokens": 8,
        }, 1000)
        connection.sendall(len(raw).to_bytes(4, "big") + raw)
    else:
        connection = _request_socket(server._path)
    timer = None
    try:
        assert api.entered.wait(2)
        api.release.set()
        assert write_entered.wait(2)
        with state.lock:
            admission = next(iter(server._admissions.values()))
            provider_deadline = admission.provider_deadline
            write_deadline = admission.completion_deadline
        server.stop()
        if expire_write:
            assert server.wait_for_handlers() is False
            assert time.monotonic() >= write_deadline
            assert _capture(api, state)["complete"] is False
            document = api.uploaded[0]
            assert document["failures"] == ["capture_incomplete"]
            assert document["response_receipts"][0]["write_state"] == "prepared"
            frozen_hash = contracts.document_hash(document)
            release_write.set()
        else:
            # Finish after the provider's original window, inside the actual
            # socket write timeout. No provider timeout or request is changed.
            timer = threading.Timer(1.05, release_write.set)
            timer.start()
            assert server.wait_for_handlers()
            assert provider_deadline < time.monotonic() < write_deadline
            assert _capture(api, state)["complete"] is True
            assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
        if write_phase == "frame":
            _read_frame(connection)
        else:
            response = connection.getresponse()
            assert response.status == 200 and response.read() == BODY
        deadline = time.monotonic() + 1
        while state.worker_handlers_inflight and time.monotonic() < deadline:
            time.sleep(0.005)
        assert state.worker_handlers_inflight == 0
        assert len(api.frames) == state.action_sequence == 1
        if expire_write:
            assert contracts.document_hash(api.uploaded[0]) == frozen_hash
            assert contracts.document_hash(state.worker_observations.document()) == frozen_hash
    finally:
        api.release.set()
        release_write.set()
        if timer is not None:
            timer.join(2)
        server.stop()
        connection.close()


@pytest.mark.parametrize("protocol", ["http", "frame"])
def test_failed_socket_write_is_retained_after_drain(short_dir, protocol):
    if protocol == "frame":
        api, server, state, connection = _start_frame(short_dir, read_timeout=0.05)
        peer = connection
    else:
        api = PausedApi()
        server, state = _start(short_dir, api)
        connection = _request_socket(server._path)
        assert api.entered.wait(3)
        peer = connection.sock
        server.stop()
    try:
        peer.shutdown(socket.SHUT_RDWR)
        connection.close()
        api.release.set()
        assert server.wait_for_handlers()
        summary = _capture(api, state)
        assert summary["complete"] is False
        assert api.uploaded[0]["response_receipt_count"] == 1
        assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_write_failed"
    finally:
        api.release.set()
        connection.close()


def test_never_finishing_handler_times_out_and_capture_stays_incomplete(short_dir, monkeypatch):
    # Use a 20ms existing operation deadline, rather than waiting its real host
    # overhead. Socket ingress shutdown consumes this original window.
    monkeypatch.setattr(runner, "MAX_PROVIDER_API_TIMEOUT_SECONDS",
                        runner.MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS)
    api, server, state, connection = _start_frame(
        short_dir, read_timeout=0.05, timeout_ms=20,
    )
    try:
        start = time.monotonic()
        assert server.wait_for_handlers() is False
        assert time.monotonic() - start < 0.1
        summary = _capture(api, state)
        assert summary["complete"] is False
        assert api.uploaded[0]["failures"] == ["capture_incomplete"]
        assert api.uploaded[0]["response_receipts"] == []
        assert len(api.frames) == state.action_sequence == 1
    finally:
        api.release.set()
        connection.close()


def test_late_native_frame_is_in_final_snapshot(short_dir):
    api, server, state, connection = _start_frame(short_dir, read_timeout=0.5)
    timer = threading.Timer(0.05, api.release.set)
    timer.start()
    try:
        assert server.wait_for_handlers()
        summary = _capture(api, state)
        _read_frame(connection)
        assert summary["complete"] is True
        assert api.uploaded[0]["response_receipt_count"] == 1
        assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
    finally:
        api.release.set()
        timer.join(1)
        connection.close()


def test_native_frame_timeout_cannot_mutate_frozen_snapshot(short_dir, monkeypatch):
    monkeypatch.setattr(runner, "MAX_PROVIDER_API_TIMEOUT_SECONDS",
                        runner.MAX_PROVIDER_OPERATION_TIMEOUT_SECONDS)
    api, server, state, connection = _start_frame(short_dir, read_timeout=0.02, timeout_ms=20)
    try:
        start = time.monotonic()
        assert server.wait_for_handlers() is False
        assert time.monotonic() - start < 0.3
        _capture(api, state)
        document = api.uploaded[0]
        assert document["failures"] == ["capture_incomplete"]
        assert document["response_receipts"] == []
        frozen_hash = contracts.document_hash(document)
        api.release.set()
        _read_frame(connection)
        deadline = time.monotonic() + 1
        while state.worker_handlers_inflight and time.monotonic() < deadline:
            time.sleep(0.005)
        assert state.worker_handlers_inflight == 0
        assert contracts.document_hash(api.uploaded[0]) == frozen_hash
        assert contracts.document_hash(state.worker_observations.document()) == frozen_hash
    finally:
        api.release.set()
        connection.close()


def test_no_handler_path_is_immediate_in_runner(tmp_path, monkeypatch):
    original = runner.WorkerSocketServer.wait_for_handlers
    calls = []

    def observed(server):
        start = time.monotonic()
        drained = original(server)
        calls.append((drained, time.monotonic() - start))
        return drained

    monkeypatch.setattr(runner.WorkerSocketServer, "wait_for_handlers", observed)
    api = LoggingApi()
    run_model(tmp_path, api, BridgingRuntime(output={"companies": []}, calls=0))
    assert len(calls) == 1 and calls[0][0] is True and calls[0][1] < 0.1
    assert len(api.completions) == 1


def test_optional_drain_exception_keeps_valid_output_and_marks_capture(tmp_path, monkeypatch):
    def broken_drain(_server):
        raise RuntimeError("synthetic optional drain failure")

    monkeypatch.setattr(runner.WorkerSocketServer, "wait_for_handlers", broken_drain)
    api = LoggingApi()
    worker = run_model(tmp_path, api, BridgingRuntime(
        output={"companies": [valid_company(1)]}, calls=0,
    ))
    assert worker.abandoned == 0 and len(api.completions) == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    assert completion["result"]["resource_summary"]["worker_observations"]["complete"] is False
    assert [row["company_name"] for row in completion["output"]["companies"]] == ["Co 1"]


@pytest.mark.parametrize("provider_status,output,exit_code,timed_out,expected", [
    (200, {"companies": [valid_company(1)]}, 0, False, "accepted"),
    (502, {"companies": [valid_company(1)]}, 0, False, "accepted"),
    (502, {"companies": []}, 0, False, "provider_error"),
    (502, {"companies": [valid_company(1)]}, 1, False, "accepted"),
    (502, None, 1, False, "provider_error"),
    (200, {"companies": [valid_company(1)]}, 1, True, "model_timeout"),
    (502, {"companies": [valid_company(1)]}, 1, True, "provider_error"),
    (200, None, 1, True, "model_timeout"),
])
def test_late_call_uses_existing_output_classification(
    tmp_path, monkeypatch, provider_status, output, exit_code, timed_out, expected,
):
    class Api(LoggingApi):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.uploaded = []

        def provider(self, _run_id, _token, frame):
            self.entered.set()
            assert self.release.wait(5)
            body = (b'{"results":[{"url":"https://co1.example.com"}]}'
                    if provider_status == 200 else b'{"error":{"code":"provider_unavailable"}}')
            return {
                "status": provider_status,
                "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(body).decode("ascii"),
                "call": {
                    "operation_id": "deepline.execute", "provider": "deepline",
                    "funding_source": "host", "action_sequence": frame["action_sequence"],
                    "call_identity": contracts.document_hash(["late", frame["action_sequence"]]),
                    "outcome": "settled", "status": provider_status,
                    "provider_status": provider_status,
                    "error_code": None if provider_status == 200 else "provider_unavailable",
                    "actual_microusd": 0,
                },
            }

        def training_observations(self, _run_id, _token, document):
            self.uploaded.append(document)
            return {"status": "stored", "payload_sha256": contracts.document_hash(document)}

    class EarlyExitRuntime:
        def __init__(self, api):
            self.api = api
            self.thread = None
            self.response_status = None

        def run_icp(self, spec, **_):
            def request():
                connection = http.client.HTTPConnection("code.deepline.com", timeout=3)
                connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.sock.settimeout(3)
                connection.sock.connect(str(spec.socket_path))
                try:
                    connection.request("POST", "/api/v2/integrations/exa_search/execute", body=json.dumps({
                        "payload": {"query": "fintech"},
                    }), headers={"content-type": "application/json"})
                    response = connection.getresponse()
                    self.response_status = response.status
                    response.read()
                finally:
                    connection.close()

            self.thread = threading.Thread(target=request, daemon=True)
            self.thread.start()
            assert self.api.entered.wait(3)
            if output is not None:
                spec.output_path.write_bytes(json.dumps(output).encode())
            return runtime.fake_result(exit_code=exit_code, timed_out=timed_out,
                                       output_bytes=runtime.read_output(spec))

    api = Api()
    sandbox = EarlyExitRuntime(api)
    original_init = runner.WorkerSocketServer.__init__

    def short_socket_read(server, *args, **kwargs):
        kwargs["read_timeout_seconds"] = 0.05
        original_init(server, *args, **kwargs)

    monkeypatch.setattr(runner.WorkerSocketServer, "__init__", short_socket_read)
    original = runner.WorkerSocketServer.wait_for_handlers
    timers = []

    def release_during_drain(server):
        timer = threading.Timer(0.15, api.release.set)
        timers.append(timer)
        timer.start()
        return original(server)

    monkeypatch.setattr(runner.WorkerSocketServer, "wait_for_handlers", release_during_drain)
    try:
        run_model(tmp_path, api, sandbox)
    finally:
        api.release.set()
        for timer in timers:
            timer.cancel()
            timer.join(1)
        if sandbox.thread is not None:
            sandbox.thread.join(3)
    assert sandbox.thread is not None and not sandbox.thread.is_alive()
    assert sandbox.response_status == provider_status
    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert api.uploaded[0]["response_receipt_count"] == 1
    assert result["terminal_status"] == expected
    assert result["resource_summary"]["provider_call_count"] == 1
    assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
    if expected == "accepted":
        assert [row["company_name"] for row in api.completions[0]["body"]["output"]["companies"]] == [
            row["company_name"] for row in output["companies"]
        ]
    else:
        assert api.completions[0]["body"]["output"] is None
