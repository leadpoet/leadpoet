"""Bounded model-reported decisions cross the existing worker trajectory path."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from lab_arena import contracts, lab_arena_checkpoint, runner, runtime, shim
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime,
    lease,
    make_config,
    valid_company,
)
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model


DECISION = {
    "objective": "Select the strongest candidate",
    "evidence": ["https://example.com/source"],
    "rationale": "The source directly supports the required signal.",
    "next_action": "Save the supported candidate.",
    "decision": "accept",
    "candidate": "Example Co",
}


class DecisionRuntime(BridgingRuntime):
    def __init__(self, *, fail: bool = False):
        super().__init__(output={"companies": [valid_company(1)]}, calls=1)
        self.fail = fail
        self.recorded = None

    def run_icp(self, spec, **kwargs):
        os.environ[lab_arena_checkpoint.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            status, _headers, body = shim.dispatch(
                "deepline.execute",
                {"tool": "exa_search", "payload": {"query": "fintech"}},
                5000,
            )
            assert status == 200 and json.loads(body)["results"]
            self.recorded = lab_arena_checkpoint.log_decision(**DECISION)
        finally:
            os.environ.pop(lab_arena_checkpoint.WORKER_SOCKET_ENV, None)
        if self.fail:
            raise ValueError("private runtime failure")
        spec.output_path.write_text(
            json.dumps(self.output), encoding="utf-8"
        )
        return runtime.fake_result(output_bytes=runtime.read_output(spec))


def _events(api):
    return [event for batch in api.batches for event in batch["events"]]


def test_real_helper_reaches_runner_trajectory_without_provider_dispatch(tmp_path):
    api = LoggingApi()
    sandbox = DecisionRuntime()
    run_model(tmp_path, api, sandbox)

    events = _events(api)
    decision = next(event for event in events if event["kind"] == "runtime.decision")
    summary = next(
        event for event in events if event["kind"] == "runtime.decision_capture"
    )
    assert sandbox.recorded is True
    assert decision["content"] == {
        "source": "model_reported",
        "sequence": 0,
        "after_action_sequence": 0,
        **DECISION,
    }
    assert summary["content"] == {
        "status": "provided",
        "recorded": 1,
        "omitted": 0,
    }
    assert len(api.provider_frames) == 1
    assert [event["kind"] for event in events].count("runtime.decision") == 1
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"


def test_runtime_and_secondary_cleanup_failure_upload_decision_once(
    tmp_path, monkeypatch,
):
    original_stop = runner.WorkerSocketServer.stop

    def fail_after_stop(server):
        original_stop(server)
        raise OSError("private cleanup failure")

    monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_after_stop)
    api = LoggingApi()
    sandbox = DecisionRuntime(fail=True)
    worker = run_model(tmp_path, api, sandbox)

    events = _events(api)
    assert sandbox.recorded is True
    assert worker.abandoned == 1
    assert [event["kind"] for event in events].count("runtime.decision") == 1
    assert [event["kind"] for event in events].count(
        "runtime.decision_capture"
    ) == 1
    assert [event["kind"] for event in events][-2:] == [
        "runtime.error",
        "runtime.cleanup_error",
    ]
    assert "private runtime failure" not in json.dumps(events)
    assert "private cleanup failure" not in json.dumps(events)


def test_concurrent_capture_is_capped_and_never_calls_provider(tmp_path, monkeypatch):
    class NoProviderApi:
        def __init__(self):
            self.provider_calls = 0

        def provider(self, *_args, **_kwargs):
            self.provider_calls += 1
            raise AssertionError("decision control dispatched to provider")

    api = NoProviderApi()
    state = runner.RunState(lease={"run_id": "run-1"}, lease_token="lease-secret")
    with tempfile.TemporaryDirectory(prefix="decision-", dir="/tmp") as directory:
        socket_path = os.path.join(directory, "worker.sock")
        server = runner.WorkerSocketServer(
            socket_path, api, state, max_connections=80
        )
        server.start()
        monkeypatch.setenv(lab_arena_checkpoint.WORKER_SOCKET_ENV, socket_path)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda index: lab_arena_checkpoint.log_decision(
                            **{**DECISION, "candidate": "Candidate %d" % index}
                        ),
                        range(80),
                    )
                )
            assert sum(results) == runner.MAX_BUFFERED_DECISION_EVENTS - 1
            assert lab_arena_checkpoint.log_decision(
                **{
                    **DECISION,
                    "decision": "finish",
                    "next_action": "Return output.",
                }
            ) is True
            assert lab_arena_checkpoint.log_decision(
                **{
                    **DECISION,
                    "decision": "finish",
                    "next_action": "Return again.",
                }
            ) is False
        finally:
            server.stop()

    assert state.decision_events_omitted == 18
    assert len(state.decision_events) == runner.MAX_BUFFERED_DECISION_EVENTS
    assert state.decision_events[-1]["content"]["decision"] == "finish"
    assert sorted(
        event["content"]["sequence"] for event in state.decision_events
    ) == list(range(runner.MAX_BUFFERED_DECISION_EVENTS))
    assert state.action_sequence == 0
    assert state.calls == []
    assert api.provider_calls == 0


def test_frame_limits_redaction_and_forged_identity_are_fail_closed(
    tmp_path, monkeypatch,
):
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token="lease-secret-value"
    )
    with tempfile.TemporaryDirectory(prefix="decision-", dir="/tmp") as directory:
        socket_path = os.path.join(directory, "worker.sock")
        server = runner.WorkerSocketServer(socket_path, object(), state)
        server.start()
        monkeypatch.setenv(lab_arena_checkpoint.WORKER_SOCKET_ENV, socket_path)
        try:
            assert lab_arena_checkpoint.log_decision(
                **{
                    **DECISION,
                    "rationale": "Authorization: Bearer abcdefghijklmnop",
                    "next_action": "Do not expose lease-secret-value",
                }
            ) is True
            assert lab_arena_checkpoint.log_decision(
                **{**DECISION, "objective": "x" * 501}
            ) is False
            assert lab_arena_checkpoint.log_decision(
                **{
                    **DECISION,
                    "objective": "\U0001f600" * 500,
                    "rationale": "\U0001f600" * 500,
                    "next_action": "\U0001f600" * 500,
                    "evidence": ["\U0001f600" * 500],
                }
            ) is False
            forged = {
                "schema_version": lab_arena_checkpoint.DECISION_SCHEMA_VERSION,
                "control": lab_arena_checkpoint.DECISION_CONTROL,
                **DECISION,
                "run_id": "forged-run",
            }
            raw = contracts.canonical_json(forged).encode("utf-8")
            assert json.loads(server.handle_frame(raw)) == {"recorded": False}
        finally:
            server.stop()

    encoded = json.dumps(state.decision_events)
    assert "abcdefghijklmnop" not in encoded
    assert "lease-secret-value" not in encoded
    assert "[REDACTED]" in encoded
    assert len(state.decision_events) == 1
    assert state.action_sequence == 0 and state.calls == []


def test_helper_returns_false_against_older_worker(tmp_path, monkeypatch):
    directory = tempfile.TemporaryDirectory(prefix="decision-", dir="/tmp")
    path = os.path.join(directory.name, "legacy.sock")
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(path)
    listening.listen(1)

    def legacy_worker():
        connection, _ = listening.accept()
        try:
            size = int.from_bytes(connection.recv(4), "big")
            received = bytearray()
            while len(received) < size:
                received.extend(connection.recv(size - len(received)))
            payload = b'{"error":"invalid_frame"}'
            connection.sendall(len(payload).to_bytes(4, "big") + payload)
        finally:
            connection.close()
            listening.close()

    thread = threading.Thread(target=legacy_worker, daemon=True)
    thread.start()
    monkeypatch.setenv(lab_arena_checkpoint.WORKER_SOCKET_ENV, path)
    assert lab_arena_checkpoint.log_decision(**DECISION) is False
    thread.join(timeout=2)
    assert not thread.is_alive()
    directory.cleanup()


def test_execute_without_model_report_emits_only_not_provided_summary(tmp_path):
    api = LoggingApi()
    run_model(
        tmp_path,
        api,
        BridgingRuntime(output={"companies": [valid_company(1)]}),
    )
    events = _events(api)
    assert all(event["kind"] != "runtime.decision" for event in events)
    summary = next(
        event for event in events if event["kind"] == "runtime.decision_capture"
    )
    assert summary["content"] == {
        "status": "not_provided",
        "recorded": 0,
        "omitted": 0,
    }
