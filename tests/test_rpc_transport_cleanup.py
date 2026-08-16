from __future__ import annotations

import errno
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway" / "tee"))

from gateway.tee import inter_enclave_tls
from gateway.tee import tee_service as gateway_service
from gateway.utils import tee_client
from gateway.utils import tee_inter_enclave_relay as relay
from validator_tee.enclave import tee_service as validator_service


HASH = "sha256:" + "a" * 64


class _ControlledSocket:
    def __init__(
        self,
        payload: bytes = b"",
        *,
        close_result=None,
        close_error: BaseException | None = None,
        close_sequence=None,
    ) -> None:
        self.payload = bytearray(payload)
        self.close_result = close_result
        self.close_error = close_error
        self.close_sequence = list(close_sequence or [])
        self.close_calls = 0
        self.shutdown_calls = 0
        self.sent = []

    def recv(self, size):
        if not self.payload:
            return b""
        chunk = bytes(self.payload[:size])
        del self.payload[:size]
        return chunk

    def sendall(self, value):
        self.sent.append(bytes(value))

    def settimeout(self, value):
        self.timeout = value

    def connect(self, address):
        self.address = address

    def shutdown(self, _direction):
        self.shutdown_calls += 1

    def close(self):
        self.close_calls += 1
        if self.close_sequence:
            outcome = self.close_sequence.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        if self.close_error is not None:
            raise self.close_error
        return self.close_result


def _framed(value) -> bytes:
    body = json.dumps(value).encode("utf-8")
    return len(body).to_bytes(4, "big") + body


def _reset_relay_cleanup_state(monkeypatch) -> None:
    monkeypatch.setattr(relay, "_RELAY_TERMINAL_FAILURE_EVENT", threading.Event())
    monkeypatch.setattr(relay, "_relay_pending_cleanup_failures", [])
    monkeypatch.setattr(relay, "_relay_cleanup_recovery_count", 0)
    monkeypatch.setattr(relay, "_relay_cleanup_attempt_count", 0)
    monkeypatch.setattr(relay, "_relay_cleanup_failure_count", 0)
    monkeypatch.setattr(relay, "_relay_last_primary_error_type", "")
    monkeypatch.setattr(relay, "_relay_last_cleanup_error_type", "")


def _reset_gateway_cleanup_state(monkeypatch) -> None:
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_attempt_count", 0)
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_failure_count", 0)
    monkeypatch.setattr(gateway_service, "vsock_rpc_last_primary_error_type", "")
    monkeypatch.setattr(gateway_service, "vsock_rpc_last_cleanup_error_type", "")
    monkeypatch.setattr(gateway_service, "vsock_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_recovery_count", 0)
    monkeypatch.setattr(
        gateway_service,
        "vsock_rpc_terminal_failure_event",
        threading.Event(),
    )


def _bare_tls_server():
    server = object.__new__(inter_enclave_tls.AttestedTLSRPCServer)
    server._transport_health_lock = threading.Lock()
    server._transport_recovery_lock = threading.Lock()
    server._transport_cleanup_attempt_count = 0
    server._transport_cleanup_failure_count = 0
    server._last_cleanup_primary_error_type = ""
    server._last_cleanup_error_type = ""
    server._pending_cleanup_failures = []
    server._cleanup_recovery_count = 0
    server._terminal_cleanup_failure_event = threading.Event()
    return server


def test_relay_cleanup_failure_is_retained_and_supervisor_terminal(
    monkeypatch,
):
    primary = ValueError("secret primary transport text")
    connection = _ControlledSocket(
        close_sequence=(
            OSError("secret cleanup transport text"),
            OSError("second cleanup cycle failed"),
            None,
        )
    )
    _reset_relay_cleanup_state(monkeypatch)
    terminal_event = relay._RELAY_TERMINAL_FAILURE_EVENT
    monkeypatch.setattr(
        relay,
        "_read_control",
        lambda _connection: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(relay.InterEnclaveRelayCleanupError) as captured:
        relay._handle_connection(
            connection,
            source_cid=17,
            cleanup_failure_callback=relay._retain_relay_cleanup_failure,
        )

    assert captured.value.__cause__ is primary
    assert captured.value.primary_error_type == "ValueError"
    assert captured.value.cleanup_error_type == "OSError"
    assert "secret" not in str(captured.value)
    assert terminal_event.is_set()
    health = relay.relay_transport_health()
    assert health == {
        "schema_version": relay.RELAY_TRANSPORT_HEALTH_SCHEMA_VERSION,
        "status": "error",
        "cleanup_attempt_count": 1,
        "cleanup_failure_count": 1,
        "last_primary_error_type": "ValueError",
        "last_cleanup_error_type": "OSError",
        "terminal_failure_latched": True,
        "retained_resource_count": 1,
        "cleanup_recovery_count": 0,
    }

    listener = _ControlledSocket()
    listener.bind = lambda address: setattr(listener, "address", address)
    listener.listen = lambda backlog: setattr(listener, "backlog", backlog)
    listener.accept = lambda: (_ for _ in ()).throw(
        RuntimeError("test listener stopped")
    )
    monkeypatch.setattr(relay.socket, "socket", lambda *_args: listener)
    with pytest.raises(RuntimeError, match="test listener stopped"):
        relay.serve_forever(port=5002)
    assert listener.close_calls == 1
    recovered = relay.relay_transport_health()
    assert recovered["status"] == "healthy"
    assert recovered["retained_resource_count"] == 0
    assert recovered["cleanup_recovery_count"] == 1


def test_relay_connect_failure_requires_target_socket_close(monkeypatch):
    target = _ControlledSocket(close_result=False)

    def fail_connect(_address):
        raise OSError("secret target connect text")

    target.connect = fail_connect
    monkeypatch.setattr(relay.socket, "socket", lambda *_args: target)

    with pytest.raises(relay.InterEnclaveRelayCleanupError) as captured:
        relay._connect_target(16, 5003)

    assert captured.value.primary_error_type == "OSError"
    assert captured.value.cleanup_error_type == "_ExplicitCloseFailure"
    assert target.close_calls == 1


def test_relay_merges_nested_target_and_source_cleanup_ownership(monkeypatch):
    target = _ControlledSocket(close_result=False)
    source = _ControlledSocket(close_result=False)
    nested = relay.InterEnclaveRelayCleanupError(
        primary_error=OSError("target connect failed"),
        cleanup_error=OSError("target cleanup failed"),
        resources=(target,),
    )
    _reset_relay_cleanup_state(monkeypatch)
    monkeypatch.setattr(relay, "_read_control", lambda _connection: {})
    monkeypatch.setattr(
        relay,
        "_validated_target",
        lambda _request, **_kwargs: (16, 5003, "a" * 32),
    )

    def fail_connector(_cid, _port):
        raise nested

    with pytest.raises(relay.InterEnclaveRelayCleanupError) as captured:
        relay._handle_connection(
            source,
            source_cid=17,
            connector=fail_connector,
            cleanup_failure_callback=relay._retain_relay_cleanup_failure,
        )

    assert {id(resource) for resource in captured.value._resources} == {
        id(target),
        id(source),
    }
    assert relay.relay_transport_health()["retained_resource_count"] == 2


@pytest.mark.parametrize("failure_stage", ("address", "construct", "start"))
def test_relay_thread_failure_transfers_accepted_owner(
    monkeypatch,
    failure_stage,
):
    accepted = _ControlledSocket(close_sequence=(False, None))
    listener = _ControlledSocket()
    listener.bind = lambda _address: None
    listener.listen = lambda _backlog: None
    class Address:
        def __getitem__(self, _index):
            if failure_stage == "address":
                raise RuntimeError("relay address parsing failed")
            return 17

    listener.accept = lambda: (accepted, Address())

    class FailedThread:
        def __init__(self, **_kwargs):
            if failure_stage == "construct":
                raise RuntimeError("thread construction failed")

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(relay.socket, "socket", lambda *_args: listener)
    monkeypatch.setattr(relay.threading, "Thread", FailedThread)
    _reset_relay_cleanup_state(monkeypatch)

    with pytest.raises(RuntimeError, match="thread|address"):
        relay.serve_forever(port=5002)

    assert accepted.close_calls == (
        1 + relay.RELAY_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
    )
    assert relay.relay_transport_health()["status"] == "healthy"


def test_relay_listener_setup_failure_retains_unconfirmed_owner(monkeypatch):
    listener = _ControlledSocket(close_sequence=(False, False, None))
    listener.bind = lambda _address: None
    listener.listen = lambda _backlog: None

    def fail_settimeout(_timeout):
        raise OSError("listener setup failed")

    listener.settimeout = fail_settimeout
    monkeypatch.setattr(relay.socket, "socket", lambda *_args: listener)
    _reset_relay_cleanup_state(monkeypatch)

    with pytest.raises(relay.InterEnclaveRelayCleanupError) as captured:
        relay.serve_forever(port=5002)

    assert captured.value.primary_error_type == "OSError"
    assert captured.value.__cause__.__class__ is OSError
    assert listener.close_calls == 2
    assert relay.relay_transport_health()["retained_resource_count"] == 1
    assert relay._recover_relay_cleanup_failures() is True
    assert relay.relay_transport_health()["status"] == "healthy"


def test_relay_retries_only_transient_accept_errors(monkeypatch):
    listener = _ControlledSocket()
    listener.bind = lambda _address: None
    listener.listen = lambda _backlog: None
    calls = 0

    def accept():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EINTR, "interrupted")
        raise RuntimeError("non-transient accept failed")

    listener.accept = accept
    monkeypatch.setattr(relay.socket, "socket", lambda *_args: listener)
    _reset_relay_cleanup_state(monkeypatch)

    with pytest.raises(RuntimeError, match="non-transient accept failed"):
        relay.serve_forever(port=5002)
    assert calls == 2


def test_gateway_one_shot_client_rejects_unconfirmed_close_after_result(
    monkeypatch,
):
    failed_socket = _ControlledSocket(
        _framed({"result": {"status": "healthy"}}),
        close_sequence=(False, False, None),
    )
    recovered_socket = _ControlledSocket(
        _framed({"result": {"status": "recovered"}}),
    )
    sockets = iter((failed_socket, recovered_socket))
    monkeypatch.setattr(
        tee_client.socket,
        "socket",
        lambda *_args, **_kwargs: next(sockets),
    )
    monkeypatch.setattr(tee_client, "_tee_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(tee_client, "_tee_rpc_cleanup_recovery_count", 0)

    with pytest.raises(tee_client.TEETransportCleanupError) as captured:
        tee_client.TEEClient(cid=16)._send_rpc_blocking(
            cid=16,
            request_bytes=b"{}",
        )

    assert captured.value.primary_error_type == "_ExplicitCloseFailure"
    assert captured.value.cleanup_error_type == "_ExplicitCloseFailure"
    assert captured.value.__cause__.__class__.__name__ == "_ExplicitCloseFailure"
    assert failed_socket.shutdown_calls == 1
    assert failed_socket.close_calls == 1
    assert tee_client.tee_rpc_transport_health()["status"] == "error"
    with pytest.raises(tee_client.TEETransportUnavailableError):
        tee_client.TEEClient(cid=16)._send_rpc_blocking(
            cid=16,
            request_bytes=b"{}",
        )
    assert tee_client.tee_rpc_transport_health()["status"] == "error"
    assert tee_client.TEEClient(cid=16)._send_rpc_blocking(
        cid=16,
        request_bytes=b"{}",
    ) == {"status": "recovered"}
    assert tee_client.tee_rpc_transport_health()["status"] == "healthy"
    assert failed_socket.close_calls == 3


def test_gateway_client_serializes_concurrent_pending_cleanup_recovery(
    monkeypatch,
):
    monkeypatch.setattr(tee_client, "_tee_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(tee_client, "_tee_rpc_cleanup_recovery_count", 0)
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingCloseSocket(_ControlledSocket):
        def close(self):
            self.close_calls += 1
            close_started.set()
            assert release_close.wait(1)
            return None

    first = tee_client.TEETransportCleanupError(
        primary_error=ValueError("primary"),
        cleanup_error=OSError("cleanup"),
        resource=BlockingCloseSocket(),
    )
    second = tee_client.TEETransportCleanupError(
        primary_error=ValueError("primary"),
        cleanup_error=OSError("cleanup"),
        resource=_ControlledSocket(),
    )
    tee_client._retain_tee_rpc_cleanup_failure(first)
    recovery_errors = []

    def recover():
        try:
            tee_client._recover_tee_rpc_cleanup_failures()
        except Exception as exc:
            recovery_errors.append(exc)

    recovery = threading.Thread(target=recover)
    recovery.start()
    assert close_started.wait(1)
    tee_client._retain_tee_rpc_cleanup_failure(second)
    assert tee_client.tee_rpc_transport_health()["retained_resource_count"] == 2
    release_close.set()
    recovery.join(timeout=1)

    assert len(recovery_errors) == 1
    assert tee_client.tee_rpc_transport_health()["retained_resource_count"] == 1
    tee_client._recover_tee_rpc_cleanup_failures()
    health = tee_client.tee_rpc_transport_health()
    assert health["status"] == "healthy"
    assert health["cleanup_recovery_count"] == 2


def test_inter_enclave_client_closes_only_active_tls_owner(monkeypatch):
    raw_socket = _ControlledSocket()
    tls_socket = _ControlledSocket(close_sequence=(False, False, None))
    tls_socket.getpeercert = lambda binary_form: b"peer"

    class Registry:
        def peer(self, _role):
            return {
                "certificate_pem": b"peer",
                "boot_identity": {"boot_identity_hash": HASH},
            }

        def peer_for_certificate(self, _certificate):
            return {"physical_role": "gateway_coordinator"}

    class Context:
        def wrap_socket(self, connection, **_kwargs):
            assert connection is raw_socket
            return tls_socket

    client = object.__new__(inter_enclave_tls.AttestedTLSRPCClient)
    client.local_physical_role = "gateway_scoring"
    client.local_boot_identity = {"boot_identity_hash": HASH}
    client.peer_registry = Registry()
    client.identity_paths = {}
    client._transport_health_lock = threading.Lock()
    client._transport_recovery_lock = threading.Lock()
    client._pending_cleanup_failures = []
    client._cleanup_recovery_count = 0
    client._connect_relay = lambda: raw_socket
    frames = iter(
        (
            {"result": {"status": "connected"}},
            {"result": {"status": "healthy"}, "channel_id": "f" * 32},
        )
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "create_mutual_tls_context",
        lambda **_kwargs: Context(),
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "_read_frame",
        lambda _connection: next(frames),
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "_send_frame",
        lambda _connection, _value: None,
    )

    with pytest.raises(
        inter_enclave_tls.InterEnclaveTransportCleanupError
    ) as captured:
        client._call_once(
            target_physical_role="gateway_coordinator",
            method="channel_health",
            params={},
            channel_id="f" * 32,
        )

    assert captured.value.stage == "client_rpc_cleanup"
    assert tls_socket.close_calls == 1
    assert raw_socket.close_calls == 0
    assert client.transport_health()["status"] == "error"
    with pytest.raises(inter_enclave_tls.InterEnclaveTLSError):
        client._require_transport_healthy()
    assert client.transport_health()["status"] == "error"
    client._require_transport_healthy()
    assert client.transport_health()["status"] == "healthy"
    assert tls_socket.close_calls == 3


def test_inter_enclave_server_latches_safe_cleanup_health(monkeypatch):
    tls_socket = _ControlledSocket(
        close_sequence=(
            OSError("secret TLS cleanup text"),
            OSError("second TLS cleanup cycle failed"),
            None,
        )
    )
    tls_socket.getpeercert = lambda binary_form: b"peer"

    class Registry:
        def trusted_certificates(self):
            return (b"peer",)

        def peer_for_certificate(self, _certificate):
            return {"boot_identity": {"boot_identity_hash": HASH}}

    class Context:
        def wrap_socket(self, _connection, **_kwargs):
            return tls_socket

    server = _bare_tls_server()
    server.identity_paths = {}
    server.peer_registry = Registry()
    server.local_boot_identity = {"boot_identity_hash": HASH}
    server._cached_response = lambda **_kwargs: {
        "result": {"status": "healthy"},
        "channel_id": "a" * 32,
    }
    monkeypatch.setattr(
        inter_enclave_tls,
        "create_mutual_tls_context",
        lambda **_kwargs: Context(),
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "_read_frame",
        lambda _connection: {"channel_id": "a" * 32},
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "validate_rpc_request",
        lambda request, **_kwargs: request,
    )
    monkeypatch.setattr(
        inter_enclave_tls,
        "_send_frame",
        lambda _connection, _value: None,
    )

    with pytest.raises(inter_enclave_tls.InterEnclaveTransportCleanupError):
        server.handle_connection(_ControlledSocket())

    health = server.transport_health()
    assert health["status"] == "error"
    assert health["cleanup_attempt_count"] == 1
    assert health["cleanup_failure_count"] == 1
    assert health["last_cleanup_error_type"] == "OSError"
    assert health["terminal_failure_latched"] is True
    assert "secret" not in json.dumps(health)

    listener = _ControlledSocket()
    listener.accept = lambda: (_ for _ in ()).throw(
        RuntimeError("test TLS listener stopped")
    )
    monkeypatch.setattr(inter_enclave_tls.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="test TLS listener stopped"):
        server.serve_forever(listener=listener)
    assert server.transport_health()["status"] == "healthy"
    assert tls_socket.close_calls == 3


@pytest.mark.parametrize("failure_stage", ("construct", "start"))
def test_tls_server_thread_failure_transfers_accepted_owner(
    monkeypatch,
    failure_stage,
):
    accepted = _ControlledSocket(close_sequence=(False, None))
    listener = _ControlledSocket()
    listener.accept = lambda: (accepted, (3, 5003))
    server = _bare_tls_server()

    class FailedThread:
        def __init__(self, **_kwargs):
            if failure_stage == "construct":
                raise RuntimeError("TLS thread construction failed")

        def start(self):
            raise RuntimeError("TLS thread start failed")

    monkeypatch.setattr(inter_enclave_tls.threading, "Thread", FailedThread)

    with pytest.raises(RuntimeError, match="TLS thread"):
        server.serve_forever(listener=listener)

    assert accepted.close_calls == (
        1 + inter_enclave_tls.TRANSPORT_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
    )
    assert server.transport_health()["status"] == "healthy"


def test_tls_listener_setup_failure_retains_unconfirmed_owner(monkeypatch):
    listener = _ControlledSocket(close_sequence=(False, False, None))
    listener.bind = lambda _address: None
    listener.listen = lambda _backlog: None

    def fail_settimeout(_timeout):
        raise OSError("TLS listener setup failed")

    listener.settimeout = fail_settimeout
    server = _bare_tls_server()
    monkeypatch.setattr(
        inter_enclave_tls.socket,
        "socket",
        lambda *_args: listener,
    )

    with pytest.raises(
        inter_enclave_tls.InterEnclaveTransportCleanupError
    ) as captured:
        server.serve_forever()

    assert captured.value.stage == "server_listener_cleanup"
    assert captured.value.primary_error_type == "OSError"
    assert captured.value.__cause__.__class__ is OSError
    assert listener.close_calls == 2
    assert server.transport_health()["retained_resource_count"] == 1
    assert server._recover_transport_cleanup_failures() is True
    assert server.transport_health()["status"] == "healthy"


@pytest.mark.parametrize("registry", ("relay", "tls_server", "gateway"))
def test_cleanup_recovery_allows_concurrent_owner_transfer(
    monkeypatch,
    registry,
):
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingCloseSocket(_ControlledSocket):
        def close(self):
            self.close_calls += 1
            close_started.set()
            assert release_close.wait(1)
            return None

    first_resource = BlockingCloseSocket()
    second_resource = _ControlledSocket()
    if registry == "relay":
        _reset_relay_cleanup_state(monkeypatch)
        first = relay.InterEnclaveRelayCleanupError(
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resources=(first_resource,),
        )
        second = relay.InterEnclaveRelayCleanupError(
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resources=(second_resource,),
        )
        retain = relay._retain_relay_cleanup_failure
        recover = relay._recover_relay_cleanup_failures
        health = relay.relay_transport_health
    elif registry == "tls_server":
        server = _bare_tls_server()
        first = inter_enclave_tls.InterEnclaveTransportCleanupError(
            stage="server_rpc_cleanup",
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resource=first_resource,
        )
        second = inter_enclave_tls.InterEnclaveTransportCleanupError(
            stage="server_rpc_cleanup",
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resource=second_resource,
        )
        retain = server._retain_transport_cleanup_failure
        recover = server._recover_transport_cleanup_failures
        health = server.transport_health
    else:
        _reset_gateway_cleanup_state(monkeypatch)
        first = gateway_service.VSOCKRPCCleanupError(
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resource=first_resource,
        )
        second = gateway_service.VSOCKRPCCleanupError(
            primary_error=ValueError("primary"),
            cleanup_error=OSError("cleanup"),
            resource=second_resource,
        )
        retain = gateway_service._retain_vsock_rpc_cleanup_failure
        recover = gateway_service._recover_vsock_rpc_cleanup_failures
        health = gateway_service.vsock_rpc_transport_health

    retain(first)
    recovery_results = []
    recovery = threading.Thread(target=lambda: recovery_results.append(recover()))
    recovery.start()
    assert close_started.wait(1)
    retain(second)
    assert health()["retained_resource_count"] == 2
    release_close.set()
    recovery.join(timeout=1)

    assert not recovery.is_alive()
    assert recovery_results == [False]
    assert health()["retained_resource_count"] == 1
    assert recover() is True
    assert health()["status"] == "healthy"


def test_dead_tls_server_with_unresolved_cleanup_cannot_be_replaced(monkeypatch):
    class DeadThread:
        @staticmethod
        def is_alive():
            return False

    class PriorServer:
        @staticmethod
        def _recover_transport_cleanup_failures():
            return False

        @staticmethod
        def transport_health():
            return {"status": "error"}

    prior = PriorServer()
    monkeypatch.setattr(gateway_service, "v2_tls_server", prior)
    monkeypatch.setattr(gateway_service, "v2_tls_server_thread", DeadThread())

    with pytest.raises(RuntimeError, match="retains unresolved"):
        gateway_service.start_v2_tls_service()

    assert gateway_service.v2_tls_server is prior


@pytest.mark.parametrize(
    ("server_health", "client_health", "expected_status"),
    (
        (
            {
                "schema_version": (
                    "leadpoet.inter_enclave_transport_health.v2"
                ),
                "status": "healthy",
            },
            {
                "schema_version": (
                    "leadpoet.inter_enclave_transport_health.v2"
                ),
                "status": "healthy",
            },
            "healthy",
        ),
        ({"status": "healthy"}, {"status": "healthy"}, "error"),
        (
            {
                "schema_version": (
                    "leadpoet.inter_enclave_transport_health.v2"
                ),
                "status": "healthy",
            },
            {"status": "unavailable"},
            "error",
        ),
        (
            {
                "schema_version": (
                    "leadpoet.inter_enclave_transport_health.v2"
                ),
                "status": "unknown",
            },
            {
                "schema_version": (
                    "leadpoet.inter_enclave_transport_health.v2"
                ),
                "status": "healthy",
            },
            "error",
        ),
    ),
)
def test_composite_inter_enclave_health_requires_exact_healthy_children(
    monkeypatch,
    server_health,
    client_health,
    expected_status,
):
    class Transport:
        def __init__(self, health):
            self.health = health

        def transport_health(self):
            return self.health

    monkeypatch.setattr(
        gateway_service,
        "v2_tls_server",
        Transport(server_health),
    )
    monkeypatch.setattr(
        gateway_service,
        "v2_inter_enclave_client",
        Transport(client_health),
    )

    health = gateway_service._inter_enclave_transport_health()

    assert health["status"] == expected_status


def test_gateway_executor_submit_failure_transfers_accepted_owner(monkeypatch):
    accepted = _ControlledSocket(close_sequence=(False, None))
    listener = _ControlledSocket()
    listener.accept = lambda: (accepted, (3, 5000))
    executor_state = {"shutdown": False}

    class FailedExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args):
            raise RuntimeError("executor submission failed")

        def shutdown(self, *, wait):
            assert wait is True
            executor_state["shutdown"] = True

    _reset_gateway_cleanup_state(monkeypatch)
    monkeypatch.setattr(gateway_service, "ThreadPoolExecutor", FailedExecutor)

    with pytest.raises(RuntimeError, match="executor submission"):
        gateway_service._serve_vsock_connections(listener)

    assert accepted.close_calls == (
        1 + gateway_service.VSOCK_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
    )
    assert gateway_service.vsock_rpc_transport_health()["status"] == "healthy"
    assert executor_state["shutdown"] is True


def test_gateway_accepted_rpc_close_failure_is_health_visible(monkeypatch):
    rpc_socket = _ControlledSocket(
        _framed({"method": "role_health", "params": {}}),
        close_sequence=(False, False, None),
    )
    monkeypatch.setattr(
        gateway_service,
        "handle_rpc",
        lambda _method, _params: {"result": {"status": "healthy"}},
    )
    _reset_gateway_cleanup_state(monkeypatch)

    with pytest.raises(gateway_service.VSOCKRPCCleanupError):
        gateway_service._handle_vsock_connection(rpc_socket, (3, 5000))

    health = gateway_service.vsock_rpc_transport_health()
    assert health["status"] == "error"
    assert health["cleanup_attempt_count"] == 1
    assert health["cleanup_failure_count"] == 1
    assert health["last_cleanup_error_type"] == "_ExplicitVSOCKCloseFailure"
    assert health["terminal_failure_latched"] is True

    assert gateway_service._recover_vsock_rpc_cleanup_failures() is False
    assert gateway_service.vsock_rpc_transport_health()["status"] == "error"
    assert gateway_service._recover_vsock_rpc_cleanup_failures() is True
    recovered = gateway_service.vsock_rpc_transport_health()
    assert recovered["status"] == "healthy"
    assert recovered["cleanup_recovery_count"] == 1
    assert rpc_socket.close_calls == 3


def test_validator_accepted_rpc_preserves_primary_when_close_raises(
    monkeypatch,
):
    primary = ValueError("secret validator primary text")
    rpc_socket = _ControlledSocket(
        _framed({"command": "health"}),
        close_error=OSError("secret validator cleanup text"),
    )
    monkeypatch.setattr(
        validator_service,
        "handle_request",
        lambda _request: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(
        validator_service.ValidatorVSOCKRPCCleanupError
    ) as captured:
        validator_service._handle_vsock_client(rpc_socket, (3, 5001))

    assert captured.value.__cause__ is primary
    assert captured.value.primary_error_type == "ValueError"
    assert captured.value.cleanup_error_type == "OSError"
    assert "secret" not in str(captured.value)
