import json
import threading

import pytest

from gateway.tee import tee_service as gateway_service
from gateway.utils import tee_client


class _ControlledSocket:
    def __init__(self, payload=b"", *, close_sequence=()):
        self.payload = bytearray(payload)
        self.close_sequence = list(close_sequence)
        self.close_calls = 0
        self.shutdown_calls = 0
        self.sent = []

    def recv(self, size):
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
        if not self.close_sequence:
            return None
        outcome = self.close_sequence.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _framed(value):
    body = json.dumps(value).encode("utf-8")
    return len(body).to_bytes(4, "big") + body


def _reset_gateway_cleanup_state(monkeypatch):
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_attempt_count", 0)
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_failure_count", 0)
    monkeypatch.setattr(gateway_service, "vsock_rpc_last_primary_error_type", "")
    monkeypatch.setattr(gateway_service, "vsock_rpc_last_cleanup_error_type", "")
    monkeypatch.setattr(gateway_service, "vsock_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(gateway_service, "vsock_rpc_cleanup_recovery_count", 0)
    monkeypatch.setattr(gateway_service, "vsock_rpc_terminal_failure_event", threading.Event())


def test_host_client_retains_unconfirmed_socket_until_cleanup_recovers(monkeypatch):
    failed = _ControlledSocket(
        _framed({"result": {"status": "healthy"}}),
        close_sequence=(False, False, None),
    )
    recovered = _ControlledSocket(_framed({"result": {"status": "recovered"}}))
    sockets = iter((failed, recovered))
    monkeypatch.setattr(tee_client.socket, "socket", lambda *_args, **_kwargs: next(sockets))
    monkeypatch.setattr(tee_client, "_tee_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(tee_client, "_tee_rpc_cleanup_recovery_count", 0)

    with pytest.raises(tee_client.TEETransportCleanupError):
        tee_client.TEEClient(cid=16)._send_rpc_blocking(cid=16, request_bytes=b"{}")
    with pytest.raises(tee_client.TEETransportUnavailableError):
        tee_client.TEEClient(cid=16)._send_rpc_blocking(cid=16, request_bytes=b"{}")
    assert tee_client.TEEClient(cid=16)._send_rpc_blocking(
        cid=16, request_bytes=b"{}"
    ) == {"status": "recovered"}
    assert tee_client.tee_rpc_transport_health()["status"] == "healthy"
    assert failed.close_calls == 3


def test_host_client_serializes_concurrent_cleanup_recovery(monkeypatch):
    monkeypatch.setattr(tee_client, "_tee_rpc_pending_cleanup_failures", [])
    monkeypatch.setattr(tee_client, "_tee_rpc_cleanup_recovery_count", 0)
    failure = tee_client.TEETransportCleanupError(
        primary_error=ValueError("primary"),
        cleanup_error=OSError("cleanup"),
        resource=_ControlledSocket(),
    )
    tee_client._retain_tee_rpc_cleanup_failure(failure)
    tee_client._recover_tee_rpc_cleanup_failures()
    assert tee_client.tee_rpc_transport_health()["cleanup_recovery_count"] == 1


def test_enclave_accepted_rpc_close_failure_is_health_visible(monkeypatch):
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
    assert gateway_service.vsock_rpc_transport_health()["status"] == "error"
    assert gateway_service._recover_vsock_rpc_cleanup_failures() is False
    assert gateway_service._recover_vsock_rpc_cleanup_failures() is True
    assert gateway_service.vsock_rpc_transport_health()["status"] == "healthy"


def test_enclave_executor_submit_failure_transfers_accepted_owner(monkeypatch):
    accepted = _ControlledSocket(close_sequence=(False, None))
    listener = _ControlledSocket()
    listener.accept = lambda: (accepted, (3, 5000))

    class FailedExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args):
            raise RuntimeError("executor submission failed")

        def shutdown(self, *, wait):
            assert wait is True

    _reset_gateway_cleanup_state(monkeypatch)
    monkeypatch.setattr(gateway_service, "ThreadPoolExecutor", FailedExecutor)
    with pytest.raises(RuntimeError, match="executor submission"):
        gateway_service._serve_vsock_connections(listener)
    assert gateway_service.vsock_rpc_transport_health()["status"] == "healthy"
