"""The Codex bridge leaves capture to the existing organizer worker."""
import tempfile
from pathlib import Path

import pytest

from lab_arena import lab_arena_codex as codex, lab_arena_checkpoint as checkpoint, runner
from tests.lab_arena.codex_runtime_test import response
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
from tests.lab_arena.test_lab_arena_runner import lease


@pytest.mark.parametrize("active_capture", [False, True])
def test_real_worker_codex_response_keeps_host_capture_without_client_ack(monkeypatch, active_capture):
    broker, store, transport = make_broker(transport=FakeTransport([(200, response())]))

    class Api:
        def provider(self, run_id, token, frame):
            return broker.execute(CONTEXT, **frame).to_document()

    state = runner.RunState(lease=lease("r1"), lease_token="tok-r1")
    with tempfile.TemporaryDirectory(prefix="codex-ack-", dir="/tmp") as directory:
        path = Path(directory) / "w.sock"
        server = runner.WorkerSocketServer(path, Api(), state)
        server.start()
        if active_capture:
            monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", str(path))
        else:
            monkeypatch.delenv("LAB_ARENA_WORKER_SOCKET", raising=False)
        try:
            status, body = codex._dispatch(str(path), {"model": "openai/gpt-4o-mini", "input": "hi"})
        finally:
            server.stop()
    assert status == 200 and body
    receipts = state.worker_observations.document()["response_receipts"]
    observations = state.worker_observations.document()["agent_observations"]
    assert len(receipts) == 1
    assert observations == []
    assert receipts[0]["write_state"] == "socket_written"
    assert len(transport.sent) == store.log.count("dispatch") == store.log.count("settle") == 1


def test_codex_dispatch_does_not_invoke_client_capture(monkeypatch):
    broker, store, transport = make_broker(transport=FakeTransport([(200, response())]))

    class Api:
        def provider(self, run_id, token, frame):
            return broker.execute(CONTEXT, **frame).to_document()

    with tempfile.TemporaryDirectory(prefix="codex-ack-", dir="/tmp") as directory:
        path = Path(directory) / "w.sock"
        server = runner.WorkerSocketServer(path, Api(), runner.RunState(lease=lease("r1"), lease_token="tok-r1"))
        server.start()
        monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", str(path))
        monkeypatch.setattr(checkpoint, "record_response_received", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("capture unavailable")))
        try:
            assert codex._dispatch(str(path), {"model": "openai/gpt-4o-mini", "input": "hi"})[0] == 200
        finally:
            server.stop()
    assert len(transport.sent) == 1
