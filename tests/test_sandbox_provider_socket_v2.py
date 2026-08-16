from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import socket
import threading
import time

import pytest

from gateway.tee import sandbox_provider_socket_v2
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.sandbox_http_shim_v2 import execute
from gateway.tee.sandbox_provider_socket_v2 import (
    SandboxProviderSocketServerV2,
    SandboxProviderSocketV2Error,
    _shutdown_and_close_socket,
)
from leadpoet_canonical.attested_v2 import build_transport_attempt


def _hash(character):
    return "sha256:" + character * 64


def _bare_server(tmp_path, *, drain_timeout_seconds=0.1):
    transport = BrokeredProviderTransportV2(lambda _request: {})
    scope = transport.create_scope(
        job_id="model-job-lifecycle",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-job-lifecycle",
        retry_policy_hashes={},
    )
    return (
        SandboxProviderSocketServerV2(
            socket_path=tmp_path / "sandbox-provider.sock",
            transport=transport,
            execution_scope=scope,
            drain_timeout_seconds=drain_timeout_seconds,
        ),
        transport,
    )


def test_sandbox_socket_preserves_shared_attempt_scope_and_strips_credentials(
    tmp_path, monkeypatch
):
    requests = []
    terminals = []

    def broker(request):
        requests.append(dict(request))
        attempt = build_transport_attempt(
            request_id=("a" if len(requests) == 1 else "b") * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=_hash("1"),
            nonsecret_headers_hash=_hash("2"),
            body_hash=_hash("3"),
            credential_ref_hash=_hash("4"),
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T00:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=_hash("5"),
            request_artifact_hash=_hash("6"),
            response_artifact_hash=_hash("5"),
            tls_peer_chain_hash=_hash("7"),
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T00:00:01Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(b'{"ok":true}').decode(),
            "encrypted_request_artifact_id": _hash("6"),
            "encrypted_artifact_id": _hash("5"),
            "transport_attempt": attempt,
        }

    transport = BrokeredProviderTransportV2(broker)
    scope = transport.create_scope(
        job_id="model-job-1",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-job-1",
        retry_policy_hashes={"openrouter": _hash("8")},
        terminal_sink=lambda attempt: terminals.append(dict(attempt)),
    )
    socket_path = Path("/tmp") / (
        "lp-sandbox-%s.sock"
        % hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    server = SandboxProviderSocketServerV2(
        socket_path=socket_path,
        transport=transport,
        execution_scope=scope,
    )
    server.start()
    monkeypatch.setenv("LEADPOET_SANDBOX_PROVIDER_SOCKET", str(socket_path))
    try:
        first = execute(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer sandbox-value", "x-title": "Leadpoet"},
            body=b"{}",
            timeout_ms=30000,
        )
        second = execute(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer another-value", "x-title": "Leadpoet"},
            body=b"{}",
            timeout_ms=30000,
        )
    finally:
        server.close()
        transport.restore()

    assert base64.b64decode(first["body_b64"]) == b'{"ok":true}'
    assert base64.b64decode(second["body_b64"]) == b'{"ok":true}'
    assert [request["attempt_number"] for request in requests] == [0, 1]
    assert all("Authorization" not in request["headers"] for request in requests)
    assert len(terminals) == 2


def test_sandbox_listener_cleanup_retains_failed_ownership(tmp_path):
    class Listener:
        def __init__(self):
            self.close_result = False

        def shutdown(self, _how):
            return None

        def close(self):
            return self.close_result

    listener = Listener()
    assert _shutdown_and_close_socket(listener) is False
    server, transport = _bare_server(tmp_path)
    server._listener = listener

    with pytest.raises(
        SandboxProviderSocketV2Error,
        match="listener cleanup failed",
    ):
        server.close()

    assert server._listener is listener
    assert server.status()["status"] == "cleanup_failed"
    assert server.status()["socket_cleanup_failure_count"] == 1

    listener.close_result = None
    server.close()
    assert server._listener is None
    assert server.status()["status"] == "stopped"
    transport.restore()


def test_sandbox_start_preserves_primary_and_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    class Listener:
        def bind(self, _address):
            raise RuntimeError("primary bind failure")

        def shutdown(self, _how):
            return None

        def close(self):
            return False

    listener = Listener()
    server, transport = _bare_server(tmp_path)
    monkeypatch.setattr(
        sandbox_provider_socket_v2.socket,
        "socket",
        lambda *_args: listener,
    )

    with pytest.raises(
        SandboxProviderSocketV2Error,
        match="listener cleanup failed after startup",
    ) as captured:
        server.start()

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "primary bind failure"
    assert server._listener is listener
    assert server.status()["last_failure"]["primary_error_type"] == "RuntimeError"
    transport.restore()


def test_sandbox_close_retains_live_accept_thread(tmp_path):
    class Listener:
        def shutdown(self, _how):
            return None

        def close(self):
            return None

    class Thread:
        def __init__(self):
            self.joined = False

        def join(self, timeout=None):
            assert timeout is not None
            self.joined = True

        def is_alive(self):
            return True

    server, transport = _bare_server(tmp_path)
    listener = Listener()
    thread = Thread()
    server._listener = listener
    server._thread = thread

    with pytest.raises(
        SandboxProviderSocketV2Error,
        match="accept loop did not terminate",
    ):
        server.close()

    assert thread.joined is True
    assert server._listener is listener
    assert server._thread is thread
    assert server.status()["status"] == "cleanup_failed"
    transport.restore()


def test_sandbox_accept_loop_death_is_observable(tmp_path):
    class Listener:
        def accept(self):
            raise OSError(5, "redacted")

    class DeadThread:
        def is_alive(self):
            return False

    server, transport = _bare_server(tmp_path)
    stop_event = threading.Event()
    server._stop = stop_event
    server._accept_loop(Listener(), stop_event)
    server._listener = Listener()
    server._thread = DeadThread()

    status = server.status()
    assert status["status"] == "failed"
    assert status["last_failure"]["stage"] == "accept_loop"
    assert status["last_failure"]["errno"] == 5
    transport.restore()


def test_sandbox_handler_cleanup_retains_primary_diagnostic(
    tmp_path,
    monkeypatch,
):
    class Connection:
        def __init__(self):
            self.close_result = False
            self.responses = []

        def sendall(self, payload):
            self.responses.append(payload)

        def shutdown(self, _how):
            return None

        def close(self):
            return self.close_result

    server, transport = _bare_server(tmp_path)
    connection = Connection()
    monkeypatch.setattr(
        sandbox_provider_socket_v2,
        "_read_frame",
        lambda _connection: (_ for _ in ()).throw(
            RuntimeError("primary request failure")
        ),
    )

    server._handle(connection)

    status = server.status()
    assert status["socket_cleanup_failure_count"] == 1
    assert status["pending_endpoint_cleanup_count"] == 1
    assert status["last_failure"] == {
        "stage": "handler_endpoint_cleanup",
        "error_type": "SandboxProviderSocketV2Error",
        "errno": 0,
        "endpoint": "sandbox",
        "primary_error_type": "RuntimeError",
    }
    assert connection.responses
    connection.close_result = None
    server.close()
    assert server.status()["pending_endpoint_cleanup_count"] == 0
    transport.restore()


def _blocking_handler_server(tmp_path, *, drain_timeout_seconds):
    transport = BrokeredProviderTransportV2(lambda _request: {})
    scope = transport.create_scope(
        job_id="model-job-drain",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-job-drain",
        retry_policy_hashes={},
    )
    socket_path = Path("/tmp") / (
        "lp-sandbox-drain-%s.sock"
        % hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    server = SandboxProviderSocketServerV2(
        socket_path=socket_path,
        transport=transport,
        execution_scope=scope,
        drain_timeout_seconds=drain_timeout_seconds,
    )
    started = threading.Event()
    release = threading.Event()

    def block(connection):
        started.set()
        release.wait(timeout=2)
        connection.close()

    server._handle = block
    server.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    assert started.wait(timeout=1)
    return server, transport, client, release


def test_sandbox_socket_close_drains_active_request_handlers(tmp_path):
    server, transport, client, release = _blocking_handler_server(
        tmp_path,
        drain_timeout_seconds=1,
    )
    closed = threading.Event()
    errors = []

    def close_server():
        try:
            server.close()
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            closed.set()

    closer = threading.Thread(target=close_server)
    closer.start()
    assert not closed.wait(timeout=0.05)
    release.set()
    assert closed.wait(timeout=1)
    closer.join(timeout=1)
    client.close()
    transport.restore()
    assert errors == []


def test_sandbox_socket_close_fails_closed_when_handlers_do_not_drain(tmp_path):
    server, transport, client, release = _blocking_handler_server(
        tmp_path,
        drain_timeout_seconds=0.05,
    )
    started = time.monotonic()
    with pytest.raises(
        SandboxProviderSocketV2Error,
        match="provider request handlers did not drain",
    ):
        server.close()
    assert time.monotonic() - started < 0.5
    release.set()
    deadline = time.monotonic() + 1
    while server._handlers and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server._handlers == set()
    client.close()
    transport.restore()


def test_timed_out_handler_cannot_mutate_frozen_execution_artifacts(tmp_path):
    transport = BrokeredProviderTransportV2(lambda _request: {})
    scope = transport.create_scope(
        job_id="model-job-drain",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-job-drain",
        retry_policy_hashes={},
    )
    context = ExecutionContextV2(
        job_id=scope.job_id,
        purpose=scope.purpose,
        epoch_id=1,
    )
    socket_path = Path("/tmp") / (
        "lp-sandbox-freeze-%s.sock"
        % hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    server = SandboxProviderSocketServerV2(
        socket_path=socket_path,
        transport=transport,
        execution_scope=scope,
        drain_timeout_seconds=0.05,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    late_errors = []

    def append_late(connection):
        started.set()
        release.wait(timeout=2)
        try:
            context.record_artifact(_hash("9"))
        except Exception as exc:
            late_errors.append(exc)
        finally:
            connection.close()
            finished.set()

    server._handle = append_late
    server.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(socket_path))
    assert started.wait(timeout=1)

    with pytest.raises(
        SandboxProviderSocketV2Error,
        match="provider request handlers did not drain",
    ):
        server.close()
    assert context.freeze_artifact_hashes() == ()
    release.set()
    assert finished.wait(timeout=1)

    assert len(late_errors) == 1
    assert "after execution was finalized" in str(late_errors[0])
    assert context.freeze_artifact_hashes() == ()
    client.close()
    transport.restore()
