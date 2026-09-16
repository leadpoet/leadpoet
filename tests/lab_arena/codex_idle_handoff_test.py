"""Passive Codex bridge handoff after an interrupted client process."""

from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from lab_arena import broker as br
from lab_arena import lab_arena_codex as codex
from lab_arena import lab_arena_checkpoint
from lab_arena import output as arena_output
from lab_arena import runtime as arena_runtime
from tests.lab_arena.codex_runtime_test import broker_socket, response
from tests.lab_arena.test_lab_arena_broker import FakeTransport


class BlockingFakeTransport(FakeTransport):
    """Hold only the first strict fake-provider response until released."""

    def __init__(self, responses):
        super().__init__(responses)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.attempts = 0

    def send(self, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            self.entered.set()
            if not self.release.wait(5):
                raise br.ProviderTransportError("test release timeout")
        return super().send(**kwargs)


def _abandoned_post(base_url: str, token: str) -> socket.socket:
    """Send one complete request, then let the caller close the client."""

    target = urlsplit(base_url)
    body = json.dumps({
        "model": "openai/gpt-4o-mini",
        "input": "continue slow research",
        "stream": True,
        "store": False,
    }, separators=(",", ":")).encode()
    client = socket.create_connection((target.hostname, target.port), timeout=2)
    request = (
        "POST /v1/responses HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n" % (token, len(body))
    ).encode() + body
    client.sendall(request)
    return client


def test_interrupted_request_settles_before_finalization_checkpoint(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_path = output_dir / "companies.json"
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))
    final_text = "[]"
    final_output = [{
        "id": "msg-final", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": final_text, "annotations": []}],
    }]
    transport = BlockingFakeTransport([
        (200, response(id="research-old")),
        (200, response(id="finalizer", output=final_output)),
    ])

    with broker_socket(monkeypatch, transport) as (store, _, _), codex.session(
            model="openai/gpt-4o-mini") as environment:
        abandoned = _abandoned_post(
            environment["LAB_ARENA_CODEX_BASE_URL"],
            environment["LAB_ARENA_CODEX_TOKEN"],
        )
        assert transport.entered.wait(2)
        abandoned.shutdown(socket.SHUT_RDWR)
        abandoned.close()

        # A zero-time wait is passive: it neither admits a finalizer nor adds
        # a provider call while the old research request still owns the gate.
        assert environment.wait_idle(0) is False
        assert transport.attempts == 1

        idle_result = []
        wait_finished = threading.Event()

        def wait_for_idle():
            idle_result.append(environment.wait_idle(2.0))
            wait_finished.set()

        waiter = threading.Thread(target=wait_for_idle)
        waiter.start()
        assert not wait_finished.wait(0.05)
        transport.release.set()
        assert wait_finished.wait(2)
        waiter.join(timeout=2)
        assert idle_result == [True]

        with httpx.Client(trust_env=False) as client:
            finalizer = client.post(
                environment["LAB_ARENA_CODEX_BASE_URL"] + "/responses",
                headers={"Authorization": "Bearer " + environment["LAB_ARENA_CODEX_TOKEN"]},
                json={
                    "model": "openai/gpt-4o-mini",
                    "input": "finalize and checkpoint",
                    "stream": True,
                    "store": False,
                },
            )

        assert finalizer.status_code == 200, finalizer.text
        events = [
            json.loads(line.removeprefix("data: "))
            for line in finalizer.text.splitlines()
            if line.startswith("data: ")
        ]
        completed = next(event["response"] for event in events
                         if event["type"] == "response.completed")
        returned_text = completed["output"][0]["content"][0]["text"]
        assert returned_text == final_text

        lab_arena_checkpoint.write(
            json.loads(returned_text),
            output_path=Path(environment["LAB_ARENA_OUTPUT_PATH"]),
        )
        host_bytes = arena_runtime.read_output(SimpleNamespace(output_path=output_path))
        assert host_bytes == b'{"companies":[]}'
        assert arena_output.output_document_from_bytes(host_bytes)["companies"] == []
        assert list(output_dir.iterdir()) == [output_path]

    assert transport.attempts == len(transport.sent) == len(store.calls) == 2
    assert all(call["kind"] == "settlement" for call in store.calls.values())
    assert all(call["terminal"]["call_succeeded"] is True for call in store.calls.values())
    assert [call["actual"] for call in store.calls.values()] == [12, 12]


@pytest.mark.parametrize("timeout", [
    True, False, -0.01, math.inf, -math.inf, math.nan, 2700.01, 10 ** 1000, "1",
])
def test_wait_idle_rejects_invalid_timeouts_without_dispatch(timeout):
    with codex.ResponsesBridge("/unused.sock") as bridge:
        with pytest.raises(codex.CodexRuntimeError, match="idle timeout"):
            bridge.wait_idle(timeout)


def test_session_environment_remains_subprocess_compatible_and_scoped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-cross")
    first_environment = None
    first_wait = None
    second_environment = None
    second_wait = None

    with broker_socket(monkeypatch) as (store, transport, _):
        with codex.session(model="openai/gpt-4o-mini") as first_environment:
            first_wait = first_environment.wait_idle
            assert isinstance(first_environment, dict)
            assert "wait_idle" not in first_environment
            assert "must-not-cross" not in first_environment.values()
            assert first_environment.wait_idle(0) is True
            assert first_environment.copy() == dict(first_environment)
            child = subprocess.run(
                [sys.executable, "-c", (
                    "import os; "
                    "assert os.environ['CODEX_HOME']; "
                    "assert os.environ['LAB_ARENA_CODEX_BASE_URL'].startswith('http://127.0.0.1:'); "
                    "assert 'OPENAI_API_KEY' not in os.environ; "
                    "print('SUBPROCESS_ENV_OK')"
                )],
                env=first_environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert child.stdout.strip() == "SUBPROCESS_ENV_OK"
            assert first_environment["LAB_ARENA_CODEX_TOKEN"] not in repr(first_wait)

            with codex.session(model="openai/gpt-4o-mini") as second_environment:
                second_wait = second_environment.wait_idle
                assert second_wait is not first_wait
                assert second_environment.wait_idle(0) is True
            with pytest.raises(codex.CodexRuntimeError, match="session is closed"):
                second_wait(0)
            assert first_environment.wait_idle(0) is True

    assert first_environment is not None and second_environment is not None
    assert first_wait is not None and second_wait is not None
    with pytest.raises(codex.CodexRuntimeError, match="session is closed"):
        first_environment.wait_idle(0)
    with pytest.raises(codex.CodexRuntimeError, match="session is closed"):
        second_environment.wait_idle(0)
    assert not transport.sent and not store.calls
    assert not Path(first_environment["CODEX_HOME"]).exists()
    assert not Path(second_environment["CODEX_HOME"]).exists()
