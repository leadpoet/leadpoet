"""The Codex bridge acknowledges only complete, host-bound response bytes."""
import tempfile
from pathlib import Path

import pytest

from lab_arena import lab_arena_codex as codex, lab_arena_checkpoint as checkpoint, runner
from tests.lab_arena.codex_runtime_test import response
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
from tests.lab_arena.test_lab_arena_runner import lease


@pytest.mark.parametrize("active_capture", [False, True])
def test_real_worker_codex_response_ack_is_exact_and_does_not_send_provider_again(monkeypatch, active_capture):
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
    assert len(observations) == int(active_capture)
    if active_capture:
        ack = observations[0]
        assert ack["kind"] == "response_received"
        assert all(ack["data"][key] == receipts[0][key]
                   for key in ("call_identity", "status", "body_sha256", "body_bytes"))
    assert len(transport.sent) == store.log.count("dispatch") == store.log.count("settle") == 1


def test_ack_capture_failure_cannot_change_the_paid_response(monkeypatch):
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
