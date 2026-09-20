"""Passive Codex bridge handoff after an interrupted client process."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from lab_arena import broker as br
from lab_arena import lab_arena_codex as codex
from lab_arena import lab_arena_checkpoint
from lab_arena import output as arena_output
from lab_arena import runner as arena_runner
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


def _post(environment, **changes):
    document = {
        "model": "openai/gpt-4o-mini",
        "input": "unchanged input",
        "stream": True,
        "store": False,
        **changes,
    }
    with httpx.Client(trust_env=False) as client:
        return client.post(
            environment["LAB_ARENA_CODEX_BASE_URL"] + "/responses",
            headers={"Authorization": "Bearer " + environment["LAB_ARENA_CODEX_TOKEN"]},
            json=document,
        )


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
    cancellation_observed = threading.Event()
    original_dispatch = codex._dispatch

    def observed_dispatch(socket_path, document, *, cancel_requested, **kwargs):
        def observed_cancel_requested():
            cancelled = cancel_requested()
            if cancelled:
                cancellation_observed.set()
            return cancelled

        return original_dispatch(
            socket_path,
            document,
            cancel_requested=observed_cancel_requested,
            **kwargs,
        )

    monkeypatch.setattr(codex, "_dispatch", observed_dispatch)

    with broker_socket(monkeypatch, transport) as (store, _, _), codex.session(
            model="openai/gpt-4o-mini") as environment:
        abandoned = _abandoned_post(
            environment["LAB_ARENA_CODEX_BASE_URL"],
            environment["LAB_ARENA_CODEX_TOKEN"],
        )
        assert transport.entered.wait(2)
        abandoned.shutdown(socket.SHUT_RDWR)
        abandoned.close()
        assert cancellation_observed.wait(1)

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
    True, False, -0.01, math.inf, -math.inf, math.nan, 5400.01, 10 ** 1000, "1",
])
def test_wait_idle_rejects_invalid_timeouts_without_dispatch(timeout):
    with codex.ResponsesBridge("/unused.sock") as bridge:
        with pytest.raises(codex.CodexRuntimeError, match="idle timeout"):
            bridge.wait_idle(timeout)


def test_explicit_90m_window_crosses_real_session_and_bridge(monkeypatch):
    monkeypatch.setattr(codex.time, "monotonic", lambda: 100.0)
    transport = FakeTransport([(200, response(id="within-90m-window"))])
    with broker_socket(monkeypatch, transport) as (store, transport, _), codex.session(
        model="openai/gpt-4o-mini",
        response_deadline=100.0 + 5370,
    ) as environment:
        config = (Path(environment["CODEX_HOME"]) / "config.toml").read_text()
        assert "stream_idle_timeout_ms = 5370000" in config
        assert environment.wait_idle(5370) is True
        reply = _post(environment)
        assert environment.wait_idle(5370) is True

    assert reply.status_code == 200, reply.text
    assert len(store.calls) == len(transport.sent) == 1


def test_omitted_deadline_keeps_45m_default(monkeypatch):
    monkeypatch.setattr(codex.time, "monotonic", lambda: 100.0)
    with codex.ResponsesBridge("/unused.sock") as bridge:
        assert codex.DEFAULT_IDLE_WAIT_SECONDS == 2700
        assert codex.MAX_IDLE_WAIT_SECONDS == 5400
        assert bridge.response_deadline == 2800.0
        assert bridge.wait_idle(2700) is True
    assert codex.run.__kwdefaults__["timeout_seconds"] == 2700


@pytest.mark.parametrize(
    "deadline",
    [True, 99.0, 100.0, math.inf, -math.inf, math.nan, 5500.01],
)
def test_session_and_bridge_reject_invalid_explicit_deadlines(monkeypatch, deadline):
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/egress.sock")
    monkeypatch.setattr(codex.time, "monotonic", lambda: 100.0)
    with pytest.raises(codex.CodexRuntimeError, match="response deadline"):
        codex.ResponsesBridge("/unused.sock", response_deadline=deadline)
    with pytest.raises(codex.CodexRuntimeError, match="response deadline"):
        with codex.session(
            model="openai/gpt-4o-mini", response_deadline=deadline,
        ):
            pass


@pytest.mark.parametrize("timeout", [True, 0, -1, math.inf, math.nan, 5400.01])
def test_run_rejects_invalid_90m_timeout_before_launch(timeout):
    with pytest.raises(codex.CodexRuntimeError, match="invalid Codex timeout"):
        codex.run(
            "do local work", model="openai/gpt-4o-mini", cwd="/tmp",
            timeout_seconds=timeout,
        )


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


@pytest.mark.parametrize("guard", [False, 0, "allow", object()])
def test_session_rejects_noncallable_guard_before_listener(monkeypatch, guard):
    listeners = []
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/unused-worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/unused-egress.sock")
    monkeypatch.setattr(
        codex, "ThreadingHTTPServer",
        lambda *_args, **_kwargs: listeners.append(True),
    )
    with pytest.raises(codex.CodexRuntimeError, match="request guard"):
        with codex.session(model="openai/gpt-4o-mini", request_guard=guard):
            pass
    assert listeners == []


@pytest.mark.parametrize("gate", [False, object(), SimpleNamespace(acquire=lambda: True)])
def test_session_rejects_invalid_shared_request_gate_before_listener(
    monkeypatch, gate,
):
    listeners = []
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/unused-worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/unused-egress.sock")
    monkeypatch.setattr(
        codex, "ThreadingHTTPServer",
        lambda *_args, **_kwargs: listeners.append(True),
    )
    with pytest.raises(codex.CodexRuntimeError, match="request gate"):
        with codex.session(
                model="openai/gpt-4o-mini", request_gate=gate):
            pass
    assert listeners == []


@pytest.mark.parametrize(
    "gate_factory",
    [threading.Lock, lambda: threading.BoundedSemaphore(1)],
    ids=("lock", "semaphore"),
)
def test_two_sessions_share_request_gate_and_recheck_guards_after_wait(
    monkeypatch, gate_factory,
):
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/fixture-worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/fixture-egress.sock")
    gate = gate_factory()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    state_lock = threading.Lock()
    dispatches = []
    guard_calls = []

    def dispatch(_socket_path, parameters, **kwargs):
        assert callable(kwargs["cancel_requested"])
        assert kwargs["response_deadline"] > time.monotonic()
        with state_lock:
            dispatches.append(parameters["input"])
            call_number = len(dispatches)
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(2)
        else:
            second_entered.set()
        return 200, json.dumps(response(), separators=(",", ":")).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.session(
            model="openai/gpt-4o-mini", request_gate=gate,
            request_guard=lambda: guard_calls.append("first") or True,
    ) as first, codex.session(
            model="openai/gpt-4o-mini", request_gate=gate,
            request_guard=lambda: guard_calls.append("second") or True,
    ) as second:
        replies = {}

        def send(name, environment):
            replies[name] = _post(environment, input=name)

        first_thread = threading.Thread(target=send, args=("first", first))
        second_thread = threading.Thread(target=send, args=("second", second))
        first_thread.start()
        assert first_entered.wait(2)
        second_thread.start()
        assert not second_entered.wait(0.1)
        assert guard_calls == ["first"]
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert [replies[name].status_code for name in ("first", "second")] == [200, 200]
    assert dispatches == ["first", "second"]
    assert guard_calls == ["first", "second"]


def test_shared_request_gate_timeout_refuses_without_dispatch(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    assert gate.acquire(blocking=False)
    dispatches = []
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: dispatches.append(True),
    )

    try:
        with codex.ResponsesBridge(
                "/fixture-worker.sock", request_gate=gate,
                response_deadline=time.monotonic() + 0.05) as bridge:
            reply = _post({
                "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
                "LAB_ARENA_CODEX_TOKEN": bridge.token,
            })
    finally:
        gate.release()

    assert reply.status_code == 429
    assert reply.json() == {"error": {"message": "request unavailable"}}
    assert dispatches == []


def test_bridge_close_cancels_shared_gate_wait_without_dispatch(monkeypatch):
    wait_started = threading.Event()

    class ObservedSemaphore(threading.Semaphore):
        def acquire(self, *args, **kwargs):
            wait_started.set()
            return super().acquire(*args, **kwargs)

    gate = ObservedSemaphore(0)
    dispatches = []
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: dispatches.append(True),
    )
    bridge = codex.ResponsesBridge(
        "/fixture-worker.sock", request_gate=gate,
    )
    bridge.__enter__()
    replies = []

    def send():
        replies.append(_post({
            "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
            "LAB_ARENA_CODEX_TOKEN": bridge.token,
        }))

    request = threading.Thread(target=send)
    request.start()
    assert wait_started.wait(2)
    bridge.__exit__(None, None, None)
    request.join(timeout=2)

    assert not request.is_alive()
    assert replies[0].status_code == 429
    assert dispatches == []


def test_client_disconnect_cancels_shared_gate_wait_without_releasing_owner(monkeypatch):
    wait_started = threading.Event()

    class ObservedSemaphore(threading.BoundedSemaphore):
        def acquire(self, *args, **kwargs):
            wait_started.set()
            return super().acquire(*args, **kwargs)

    gate = ObservedSemaphore(1)
    assert gate.acquire(blocking=False)
    wait_started.clear()
    dispatches = []
    monkeypatch.setattr(
        codex, "_dispatch", lambda *_args, **_kwargs: dispatches.append(True),
    )
    try:
        with codex.ResponsesBridge(
                "/fixture-worker.sock", request_gate=gate,
                response_deadline=time.monotonic() + 5) as bridge:
            client = _abandoned_post(bridge.base_url, bridge.token)
            assert wait_started.wait(2)
            client.close()
            assert bridge.wait_idle(1)
            assert not gate.acquire(blocking=False)
    finally:
        gate.release()
    assert dispatches == []


def test_shared_request_gate_releases_after_dispatch_failure(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/fixture-worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/fixture-egress.sock")
    gate = threading.BoundedSemaphore(1)
    dispatches = []

    def dispatch(_socket_path, _parameters, **kwargs):
        assert callable(kwargs["cancel_requested"])
        dispatches.append("called")
        if len(dispatches) == 1:
            raise codex.CodexRuntimeError("fixture failure")
        return 200, json.dumps(response(), separators=(",", ":")).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.session(
            model="openai/gpt-4o-mini", request_gate=gate) as first, codex.session(
            model="openai/gpt-4o-mini", request_gate=gate) as second:
        first_reply = _post(first)
        second_reply = _post(second)

    assert first_reply.status_code == 502
    assert second_reply.status_code == 200
    assert dispatches == ["called", "called"]
    assert gate.acquire(blocking=False)
    gate.release()


def test_shared_request_gate_releases_after_guard_refusal(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    dispatches = []
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: dispatches.append(True),
    )

    with codex.ResponsesBridge(
            "/fixture-worker.sock", request_gate=gate,
            request_guard=lambda: False) as bridge:
        reply = _post({
            "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
            "LAB_ARENA_CODEX_TOKEN": bridge.token,
        })

    assert reply.status_code == 429
    assert dispatches == []
    assert gate.acquire(blocking=False)
    gate.release()


def test_client_disconnect_during_guard_releases_gate_without_dispatch(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    guard_started = threading.Event()
    release_guard = threading.Event()
    dispatches = []

    def guard():
        guard_started.set()
        assert release_guard.wait(2)
        return True

    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: dispatches.append(True),
    )
    with codex.ResponsesBridge(
            "/fixture-worker.sock", request_gate=gate,
            request_guard=guard) as bridge:
        abandoned = _abandoned_post(bridge.base_url, bridge.token)
        assert guard_started.wait(2)
        abandoned.shutdown(socket.SHUT_RDWR)
        abandoned.close()
        release_guard.set()
        assert bridge.wait_idle(2) is True

    assert dispatches == []
    assert gate.acquire(blocking=False)
    gate.release()


def test_bridge_close_during_guard_releases_gate_without_dispatch(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    guard_started = threading.Event()
    release_guard = threading.Event()
    dispatches = []
    replies = []

    def guard():
        guard_started.set()
        assert release_guard.wait(2)
        return True

    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: dispatches.append(True),
    )
    bridge = codex.ResponsesBridge(
        "/fixture-worker.sock", request_gate=gate,
        request_guard=guard,
    )
    bridge.__enter__()

    request = threading.Thread(target=lambda: replies.append(_post({
        "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
        "LAB_ARENA_CODEX_TOKEN": bridge.token,
    })))
    request.start()
    assert guard_started.wait(2)
    close = threading.Thread(target=bridge.__exit__, args=(None, None, None))
    close.start()
    assert bridge._closed.wait(2)
    release_guard.set()
    request.join(timeout=2)
    close.join(timeout=2)

    assert not request.is_alive() and not close.is_alive()
    assert replies[0].status_code == 429
    assert dispatches == []
    assert gate.acquire(blocking=False)
    gate.release()


@pytest.mark.parametrize("outcome", [False, None, 0, "allow", "exception", "base-exception"])
def test_request_guard_refusal_is_generic_and_never_dispatches(monkeypatch, outcome):
    calls = []

    def guard():
        calls.append("called")
        if outcome == "exception":
            raise RuntimeError("guard-secret-must-not-cross")
        if outcome == "base-exception":
            raise KeyboardInterrupt("guard-secret-must-not-cross")
        return outcome

    with broker_socket(monkeypatch) as (store, transport, path), codex.ResponsesBridge(
            str(path), request_guard=guard) as bridge:
        environment = {
            "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
            "LAB_ARENA_CODEX_TOKEN": bridge.token,
        }
        reply = _post(environment)
        # The client can receive the refusal before the handler's finally
        # releases its active slot. Wait for that cleanup, not scheduler timing.
        assert bridge.wait_idle(1) is True

    assert reply.status_code == 429
    assert reply.json() == {"error": {"message": "request unavailable"}}
    assert "guard-secret-must-not-cross" not in reply.text
    assert calls == ["called"]
    assert not transport.sent and not store.calls


def test_guard_runs_only_after_valid_request_while_active_and_preserves_body(monkeypatch):
    observations = []

    def guard():
        observations.append(bridge.wait_idle(0))
        return True

    with broker_socket(monkeypatch, FakeTransport([(200, response())])) as (
            store, transport, path), codex.ResponsesBridge(
                str(path), request_guard=guard) as bridge:
        with httpx.Client(trust_env=False) as client:
            assert client.post(bridge.base_url + "/responses", json={}).status_code == 401
            assert client.post(
                bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json={"store": True},
            ).status_code == 400
            assert client.post(bridge.base_url + "/other", json={}).status_code == 404
        assert observations == []
        reply = _post({
            "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
            "LAB_ARENA_CODEX_TOKEN": bridge.token,
        })

    assert reply.status_code == 200
    assert observations == [False]
    assert len(store.calls) == len(transport.sent) == 1
    outbound = json.loads(transport.sent[0]["body"])
    assert outbound["input"] == "unchanged input"


def test_guard_is_once_per_bridge_post_across_existing_hidden_retries(monkeypatch):
    calls = []
    throttle = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    transport = FakeTransport([
        (200, throttle, {"retry-after": "0"}),
        (200, throttle, {"retry-after": "0"}),
        (200, response(id="after-hidden-retries")),
    ])
    monkeypatch.setattr(arena_runner.secrets, "randbelow", lambda _bound: 0)

    with broker_socket(monkeypatch, transport) as (store, transport, path), codex.ResponsesBridge(
            str(path), request_guard=lambda: calls.append("called") or True) as bridge:
        reply = _post({
            "LAB_ARENA_CODEX_BASE_URL": bridge.base_url,
            "LAB_ARENA_CODEX_TOKEN": bridge.token,
        })

    assert reply.status_code == 200, reply.text
    assert calls == ["called"]
    assert len(transport.sent) == len(store.calls) == arena_runner.RESPONSES_RATE_LIMIT_RETRIES + 1 == 3
    assert br.CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS == 4
    # These independent existing layers can compose. The guard is an
    # admission check for one HTTP post, never a quota reservation.
    assert br.CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS * (
        arena_runner.RESPONSES_RATE_LIMIT_RETRIES + 1
    ) == 12


def test_request_guards_are_session_scoped_and_stale_sessions_do_not_call_them(monkeypatch):
    first_calls = []
    second_calls = []
    transport = FakeTransport([(200, response(id="second-session"))])

    with broker_socket(monkeypatch, transport) as (store, transport, _):
        with codex.session(
                model="openai/gpt-4o-mini",
                request_guard=lambda: first_calls.append("first") or False,
        ) as first:
            assert _post(first).status_code == 429
            with codex.session(
                    model="openai/gpt-4o-mini",
                    request_guard=lambda: second_calls.append("second") or True,
            ) as second:
                second_url = second["LAB_ARENA_CODEX_BASE_URL"]
                second_token = second["LAB_ARENA_CODEX_TOKEN"]
                assert _post(second).status_code == 200
            assert _post(first).status_code == 429
        first_url = first["LAB_ARENA_CODEX_BASE_URL"]
        first_token = first["LAB_ARENA_CODEX_TOKEN"]

        with httpx.Client(trust_env=False) as client:
            for url, token in ((first_url, first_token), (second_url, second_token)):
                with pytest.raises(httpx.ConnectError):
                    client.post(
                        url + "/responses",
                        headers={"Authorization": "Bearer " + token},
                        json={"model": "openai/gpt-4o-mini", "input": "stale"},
                    )

    assert first_calls == ["first", "first"]
    assert second_calls == ["second"]
    assert len(transport.sent) == len(store.calls) == 1


@pytest.mark.skipif(
    not os.environ.get("ARENA_TEST_CODEX_BINARY"),
    reason="set ARENA_TEST_CODEX_BINARY to the exact Codex 0.154.0 package",
)
def test_native_codex_guard_refusal_exits_and_next_phase_proceeds(monkeypatch, tmp_path):
    binary = os.environ["ARENA_TEST_CODEX_BINARY"]
    assert subprocess.check_output([binary, "--version"], text=True).strip() == "codex-cli 0.154.0"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    phase = {"allow": False}
    guard_calls = []

    def guard():
        guard_calls.append(phase["allow"])
        return phase["allow"]

    transport = FakeTransport([(200, response(output=[{
        "id": "msg-guard", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "ARENA_GUARD_PHASE_OK", "annotations": []}],
    }]))])

    with broker_socket(monkeypatch, transport) as (store, transport, _), codex.session(
            model="openai/gpt-4o-mini", request_guard=guard) as environment:
        def invoke(name):
            final_path = tmp_path / (name + ".txt")
            completed = subprocess.run(
                [binary, "exec", "--skip-git-repo-check", "--ephemeral", "--color", "never",
                 "-C", str(tmp_path), "-o", str(final_path), "-"],
                input="Reply with the supplied final answer.",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
                timeout=30,
            )
            return completed, final_path

        refused, refused_path = invoke("refused")
        assert refused.returncode != 0
        assert not refused_path.exists()
        assert guard_calls == [False]
        assert not transport.sent and not store.calls

        phase["allow"] = True
        accepted, accepted_path = invoke("accepted")

    assert accepted.returncode == 0, accepted.stdout[-8000:]
    assert accepted_path.read_text() == "ARENA_GUARD_PHASE_OK"
    assert guard_calls == [False, True]
    assert len(transport.sent) == len(store.calls) == 1
