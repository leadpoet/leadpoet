"""Final worker capture waits briefly for already-started socket handlers."""

import base64
import http.client
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, runner, runtime, shim
from tests.lab_arena.test_lab_arena_runner import lease, valid_company
from tests.lab_arena.trajectory_runner_test import BridgingRuntime, LoggingApi, run_model


IDENTITY = "sha256:" + "1" * 64
BODY = b'{"id":"local-proof","choices":[]}'


class PausedApi:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.uploaded = []

    def provider(self, _run_id, _token, frame):
        self.entered.set()
        assert self.release.wait(5)
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


def _start_frame(short_dir, *, read_timeout):
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=read_timeout)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(3)
    connection.connect(str(server._path))
    raw = shim.build_operation_frame("openrouter.chat", {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "local frame test"}],
        "max_tokens": 8,
    }, 1000)
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


def test_failed_socket_write_is_retained_after_drain(short_dir):
    api = PausedApi()
    server, state = _start(short_dir, api)
    connection = _request_socket(server._path)
    assert api.entered.wait(3)
    try:
        server.stop()
        connection.sock.shutdown(socket.SHUT_RDWR)
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


def test_never_finishing_handler_times_out_and_capture_stays_incomplete(short_dir):
    api = PausedApi()
    server, state = _start(short_dir, api, read_timeout=0.05)
    connection = _request_socket(server._path)
    assert api.entered.wait(3)
    try:
        server.stop()
        start = time.monotonic()
        assert server.wait_for_handlers() is False
        assert time.monotonic() - start < 0.5
        summary = _capture(api, state)
        assert summary["complete"] is False
        assert api.uploaded[0]["failures"] == ["capture_incomplete"]
        assert api.uploaded[0]["response_receipts"] == []
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


def test_native_frame_timeout_cannot_mutate_frozen_snapshot(short_dir):
    api, server, state, connection = _start_frame(short_dir, read_timeout=0.02)
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


@pytest.mark.parametrize("provider_status,output,exit_code,expected", [
    (200, {"companies": [valid_company(1)]}, 0, "accepted"),
    (502, {"companies": [valid_company(1)]}, 0, "accepted"),
    (502, {"companies": []}, 0, "provider_error"),
    (502, {"companies": [valid_company(1)]}, 1, "accepted"),
    (502, None, 1, "provider_error"),
])
def test_late_call_uses_existing_output_classification(
    tmp_path, monkeypatch, provider_status, output, exit_code, expected,
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
            return runtime.fake_result(exit_code=exit_code, output_bytes=runtime.read_output(spec))

    api = Api()
    sandbox = EarlyExitRuntime(api)
    original = runner.WorkerSocketServer.wait_for_handlers

    def release_during_drain(server):
        timer = threading.Timer(0.05, api.release.set)
        timer.start()
        try:
            return original(server)
        finally:
            timer.join(1)

    monkeypatch.setattr(runner.WorkerSocketServer, "wait_for_handlers", release_during_drain)
    try:
        run_model(tmp_path, api, sandbox)
    finally:
        api.release.set()
        if sandbox.thread is not None:
            sandbox.thread.join(3)
    assert sandbox.thread is not None and not sandbox.thread.is_alive()
    assert sandbox.response_status == provider_status
    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == expected
    assert result["resource_summary"]["provider_call_count"] == 1
    assert api.uploaded[0]["response_receipt_count"] == 1
    assert api.uploaded[0]["response_receipts"][0]["write_state"] == "socket_written"
    if expected == "accepted":
        assert [row["company_name"] for row in api.completions[0]["body"]["output"]["companies"]] == [
            row["company_name"] for row in output["companies"]
        ]
    else:
        assert api.completions[0]["body"]["output"] is None
