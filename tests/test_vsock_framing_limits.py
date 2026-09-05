import asyncio
import ast
import inspect
import json
import os
import socket
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import textwrap

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway" / "tee"))

from gateway.tee import tee_service as gateway_service
from gateway.utils import tee_client as gateway_client
from validator_tee.enclave import tee_service as validator_service
from validator_tee.host import vsock_client as validator_client


class _FragmentSocket:
    def __init__(self, data, fragment_size=2):
        self.data = bytearray(data)
        self.fragment_size = fragment_size

    def recv(self, size):
        if not self.data:
            return b""
        count = min(size, self.fragment_size, len(self.data))
        value = bytes(self.data[:count])
        del self.data[:count]
        return value


class _RPCSocket(_FragmentSocket):
    def __init__(self, data, fragment_size=2):
        super().__init__(data, fragment_size)
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def connect(self, address):
        self.address = address

    def sendall(self, data):
        self.request = data

    def close(self):
        self.closed = True


def test_gateway_exact_reader_handles_fragmented_prefix_and_body():
    assert gateway_service._recv_exact(_FragmentSocket(b"abcdefgh", 1), 8) == b"abcdefgh"
    assert gateway_client._recv_exact(_FragmentSocket(b"abcdefgh", 3), 8) == b"abcdefgh"


def test_validator_receives_length_prefixed_request():
    body = json.dumps({"command": "health"}).encode()
    request, framed = validator_service._receive_request(
        _FragmentSocket(len(body).to_bytes(4, "big") + body, 1)
    )
    assert request == {"command": "health"}
    assert framed is True


def test_validator_envelope_matches_gateway_receipt_graph_transport():
    # Gateway jobs upload their large inputs in bounded chunks. The validator
    # authoritative-weight RPC transports one complete authenticated graph.
    assert gateway_client.MAX_RPC_REQUEST_BYTES == 64 * 1024 * 1024
    assert gateway_service.MAX_RPC_REQUEST_BYTES == 64 * 1024 * 1024
    assert (
        validator_client.MAX_RPC_REQUEST_BYTES
        == validator_service.MAX_RPC_REQUEST_BYTES
        == 128 * 1024 * 1024
    )
    assert (
        validator_client.MAX_RPC_REQUEST_FRAME_BYTES
        == validator_service.MAX_RPC_REQUEST_FRAME_BYTES
        == 16 * 1024 * 1024
    )
    assert (
        validator_client.MAX_RPC_RESPONSE_FRAME_BYTES
        == validator_service.MAX_RPC_RESPONSE_FRAME_BYTES
        == 16 * 1024 * 1024
    )
    assert (
        validator_client.MAX_RPC_RESPONSE_BYTES
        == validator_service.MAX_RPC_RESPONSE_BYTES
        == 128 * 1024 * 1024
    )


def test_validator_receives_production_sized_receipt_graph_request():
    frame_limit = 16 * 1024 * 1024
    request = {
        "command": "compute_authoritative_weights_v2",
        "weight_request": {
            "upstream_receipt_set": {
                "transport_attempts": ["x" * (64 * 1024 * 1024 + 1)],
            },
        },
    }
    body = json.dumps(request).encode()
    assert frame_limit < len(body) < validator_service.MAX_RPC_REQUEST_BYTES
    frame = validator_client._encode_rpc_payload(
        body,
        logical_limit=validator_client.MAX_RPC_REQUEST_BYTES,
        frame_limit=validator_client.MAX_RPC_REQUEST_FRAME_BYTES,
    )
    assert frame.startswith(b"LPZ2")
    assert len(frame) < validator_service.MAX_RPC_REQUEST_FRAME_BYTES

    observed, framed = validator_service._receive_request(
        _FragmentSocket(len(frame).to_bytes(4, "big") + frame, 64 * 1024)
    )

    assert observed == request
    assert framed is True


def test_restart_rehearsal_validator_relay_uses_candidate_frame_limits():
    source_path = (
        Path(__file__).resolve().parent
        / "restart_rehearsal"
        / "sitecustomize.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_LocalVsock"
    ]
    assert len(classes) == 1
    timeout_methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "settimeout"
    ]
    assert len(timeout_methods) == 1
    timeout_sets = [
        node
        for node in ast.walk(timeout_methods[0])
        if isinstance(node, ast.Set)
    ]
    assert len(timeout_sets) == 1
    allowed_timeouts = {
        float(element.value)
        for element in timeout_sets[0].elts
        if isinstance(element, ast.Constant)
    }
    assert allowed_timeouts == {30.0, 120.0, 180.0}
    compute_source = ast.parse(
        textwrap.dedent(
            inspect.getsource(
                validator_client.ValidatorEnclaveClient.compute_authoritative_weights_v2
            )
        )
    )
    requested_timeouts = {
        float(keyword.value.value)
        for node in ast.walk(compute_source)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "timeout_seconds"
        and isinstance(keyword.value, ast.Constant)
    }
    assert requested_timeouts == {180.0}
    assert requested_timeouts <= allowed_timeouts
    sendall = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "sendall"
    ]
    assert len(sendall) == 1
    decode_calls = [
        node
        for node in ast.walk(sendall[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_decode_rpc_payload"
    ]
    assert len(decode_calls) == 1
    decode_keywords = {
        keyword.arg: keyword.value for keyword in decode_calls[0].keywords
    }
    logical_limit = decode_keywords.get("logical_limit")
    assert isinstance(logical_limit, ast.Attribute)
    assert isinstance(logical_limit.value, ast.Name)
    assert logical_limit.value.id == "tee_service"
    assert logical_limit.attr == "MAX_RPC_REQUEST_BYTES"
    assert "tee_service.MAX_RPC_RESPONSE_FRAME_BYTES" in ast.unparse(sendall[0])


def test_validator_large_response_uses_same_bounded_compressed_frame():
    body = json.dumps(
        {"status": "ok", "receipt_graph": "x" * (64 * 1024 * 1024 + 1)}
    ).encode()
    frame = validator_service._encode_rpc_payload(
        body,
        logical_limit=validator_service.MAX_RPC_RESPONSE_BYTES,
        frame_limit=validator_service.MAX_RPC_RESPONSE_FRAME_BYTES,
    )

    assert frame.startswith(b"LPZ2")
    assert len(frame) < validator_client.MAX_RPC_RESPONSE_FRAME_BYTES
    assert validator_client._decode_rpc_payload(
        frame,
        logical_limit=validator_client.MAX_RPC_RESPONSE_BYTES,
    ) == body


def test_validator_client_sends_large_request_as_compressed_frame(monkeypatch):
    response = json.dumps({"status": "ok", "accepted": True}).encode()
    rpc_socket = _RPCSocket(
        len(response).to_bytes(4, "big") + response,
        fragment_size=3,
    )
    monkeypatch.setattr(
        validator_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )
    request = {
        "command": "compute_authoritative_weights_v2",
        "weight_request": {"receipt_graph": "x" * (17 * 1024 * 1024)},
    }

    observed = validator_client.ValidatorEnclaveClient(
        enclave_cid=16
    )._send_request(request)

    frame_size = int.from_bytes(rpc_socket.request[:4], "big")
    frame = rpc_socket.request[4:]
    assert rpc_socket.address == (16, validator_client.RPC_PORT)
    assert len(frame) == frame_size
    assert frame.startswith(b"LPZ2")
    assert json.loads(
        validator_service._decode_rpc_payload(
            frame,
            logical_limit=validator_service.MAX_RPC_REQUEST_BYTES,
        )
    ) == request
    assert observed["accepted"] is True
    assert rpc_socket.closed is True


def test_validator_client_decodes_large_compressed_response(monkeypatch):
    response = json.dumps(
        {"status": "ok", "receipt_graph": "x" * (17 * 1024 * 1024)}
    ).encode()
    response_frame = validator_service._encode_rpc_payload(
        response,
        logical_limit=validator_service.MAX_RPC_RESPONSE_BYTES,
        frame_limit=validator_service.MAX_RPC_RESPONSE_FRAME_BYTES,
    )
    rpc_socket = _RPCSocket(
        len(response_frame).to_bytes(4, "big") + response_frame,
        fragment_size=64 * 1024,
    )
    monkeypatch.setattr(
        validator_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )

    observed = validator_client.ValidatorEnclaveClient(
        enclave_cid=16
    )._send_request({"command": "health"})

    assert len(observed["receipt_graph"]) == 17 * 1024 * 1024
    assert rpc_socket.closed is True


def test_validator_client_closes_socket_when_timeout_setup_fails(monkeypatch):
    class _TimeoutFailureSocket(_RPCSocket):
        def settimeout(self, _timeout):
            raise ValueError("invalid timeout")

    rpc_socket = _TimeoutFailureSocket(b"")
    monkeypatch.setattr(
        validator_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )

    with pytest.raises(ValueError, match="invalid timeout"):
        validator_client.ValidatorEnclaveClient(
            enclave_cid=16
        )._send_request({"command": "health"})

    assert rpc_socket.closed is True


def test_validator_client_retains_valid_response_until_close_succeeds(monkeypatch):
    response = json.dumps({"status": "ok", "accepted": True}).encode()

    class CloseFailureSocket(_RPCSocket):
        def __init__(self, data):
            super().__init__(data)
            self.allow_close = False

        def close(self):
            self.closed = self.allow_close
            return self.allow_close

    rpc_socket = CloseFailureSocket(len(response).to_bytes(4, "big") + response)
    recovered_socket = _RPCSocket(
        len(response).to_bytes(4, "big") + response
    )
    sockets = iter((rpc_socket, recovered_socket))
    monkeypatch.setattr(
        validator_client.socket,
        "socket",
        lambda *_args, **_kwargs: next(sockets),
    )
    client = validator_client.ValidatorEnclaveClient(enclave_cid=16)

    with pytest.raises(
        validator_client.ValidatorEnclaveTransportCleanupError
    ) as raised:
        client._send_request({"command": "health"})

    assert raised.value._resource is rpc_socket
    assert raised.value._response == {"status": "ok", "accepted": True}
    assert raised.value.primary_error is raised.value.__cause__
    assert validator_client._RETIRED_CLEANUP[id(rpc_socket)][0] is rpc_socket
    rpc_socket.allow_close = True
    observed = validator_client.ValidatorEnclaveClient(
        enclave_cid=16
    )._send_request({"command": "health"})
    assert observed == {"status": "ok", "accepted": True}
    assert recovered_socket.closed is True
    assert validator_client._RETIRED_CLEANUP == {}


def test_validator_client_cleanup_preserves_original_rpc_error(monkeypatch):
    primary = ValueError("invalid timeout")

    class TimeoutAndCloseFailureSocket(_RPCSocket):
        def __init__(self):
            super().__init__(b"")
            self.allow_close = False

        def settimeout(self, _timeout):
            raise primary

        def close(self):
            self.closed = self.allow_close
            return self.allow_close

    rpc_socket = TimeoutAndCloseFailureSocket()
    monkeypatch.setattr(
        validator_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )
    client = validator_client.ValidatorEnclaveClient(enclave_cid=16)

    with pytest.raises(
        validator_client.ValidatorEnclaveTransportCleanupError
    ) as raised:
        client._send_request({"command": "health"})

    assert raised.value.primary_error is primary
    assert raised.value.__cause__ is primary
    assert raised.value._response is None
    rpc_socket.allow_close = True
    validator_client.ValidatorEnclaveClient(
        enclave_cid=16
    )._require_retired_cleanup()


def test_validator_client_cleanup_retry_preserves_concurrent_owner(monkeypatch):
    close_started = threading.Event()
    release_close = threading.Event()
    retain_done = threading.Event()

    class Resource:
        def __init__(self, *, blocking, allow_close):
            self.blocking = blocking
            self.allow_close = allow_close

        def shutdown(self, _how):
            return None

        def close(self):
            if self.blocking:
                close_started.set()
                assert release_close.wait(timeout=2)
            return self.allow_close

    monkeypatch.setattr(validator_client, "_RETIRED_CLEANUP", {})
    monkeypatch.setattr(
        validator_client,
        "_RETIRED_CLEANUP_RECOVERY_LOCK",
        threading.Lock(),
    )
    client = validator_client.ValidatorEnclaveClient(enclave_cid=16)
    first = Resource(blocking=True, allow_close=True)
    concurrent = Resource(blocking=False, allow_close=False)
    primary = OSError("cleanup failed")
    client._retain_cleanup_failure(
        first,
        primary_error=primary,
        response={"status": "first"},
    )
    recovery_errors = []

    def recover():
        try:
            client._require_retired_cleanup()
        except validator_client.ValidatorEnclaveTransportCleanupError as exc:
            recovery_errors.append(exc)

    def retain():
        client._retain_cleanup_failure(
            concurrent,
            primary_error=primary,
            response={"status": "concurrent"},
        )
        retain_done.set()

    recovery = threading.Thread(target=recover)
    recovery.start()
    assert close_started.wait(timeout=2)
    retention = threading.Thread(target=retain)
    retention.start()
    assert retain_done.wait(timeout=1)
    with validator_client._RETIRED_CLEANUP_LOCK:
        assert len(validator_client._RETIRED_CLEANUP) == 2
    release_close.set()
    recovery.join(timeout=2)
    retention.join(timeout=2)

    assert not recovery.is_alive()
    assert not retention.is_alive()
    assert len(recovery_errors) == 1
    assert recovery_errors[0]._resource is concurrent
    assert id(first) not in validator_client._RETIRED_CLEANUP
    assert validator_client._RETIRED_CLEANUP[id(concurrent)][0] is concurrent
    concurrent.allow_close = True
    client._require_retired_cleanup()
    assert validator_client._RETIRED_CLEANUP == {}


def test_validator_rejects_incompressible_frame_above_wire_limit():
    payload = os.urandom(validator_client.MAX_RPC_REQUEST_FRAME_BYTES + 1)
    with pytest.raises(RuntimeError, match="compressed frame exceeds"):
        validator_client._encode_rpc_payload(
            payload,
            logical_limit=validator_client.MAX_RPC_REQUEST_BYTES,
            frame_limit=validator_client.MAX_RPC_REQUEST_FRAME_BYTES,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda frame: frame[:-1],
        lambda frame: frame + b"trailing",
        lambda frame: frame + zlib.compress(b"{}"),
        lambda frame: frame[:4]
        + (int.from_bytes(frame[4:8], "big") + 1).to_bytes(4, "big")
        + frame[8:],
    ),
)
def test_validator_rejects_invalid_compressed_frames(mutate):
    body = json.dumps({"payload": "x" * (17 * 1024 * 1024)}).encode()
    frame = validator_client._encode_rpc_payload(
        body,
        logical_limit=validator_client.MAX_RPC_REQUEST_BYTES,
        frame_limit=validator_client.MAX_RPC_REQUEST_FRAME_BYTES,
    )

    with pytest.raises(ValueError, match="compressed frame"):
        validator_service._decode_rpc_payload(
            mutate(frame),
            logical_limit=validator_service.MAX_RPC_REQUEST_BYTES,
        )


def test_validator_rejects_compressed_frame_claiming_oversized_output():
    frame = (
        b"LPZ2"
        + (validator_service.MAX_RPC_REQUEST_BYTES + 1).to_bytes(4, "big")
        + zlib.compress(b"{}")
    )
    with pytest.raises(ValueError, match="decoded message size"):
        validator_service._decode_rpc_payload(
            frame,
            logical_limit=validator_service.MAX_RPC_REQUEST_BYTES,
        )


def test_validator_still_accepts_legacy_eof_request():
    body = json.dumps({"command": "health"}).encode()
    request, framed = validator_service._receive_request(_FragmentSocket(body, 3))
    assert request == {"command": "health"}
    assert framed is False


def test_validator_rejects_oversized_frame_before_reading_body():
    prefix = (validator_service.MAX_RPC_REQUEST_FRAME_BYTES + 1).to_bytes(4, "big")
    with pytest.raises(ValueError, match="outside"):
        validator_service._receive_request(_FragmentSocket(prefix, 4))


def test_validator_client_exact_reader_handles_fragments():
    assert validator_client._recv_exact(_FragmentSocket(b"response", 1), 8) == b"response"


def test_gateway_client_can_be_constructed_in_rpc_worker_without_event_loop():
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gateway-vsock-rpc",
    ) as executor:
        client = executor.submit(gateway_client.TEEClient, 16).result(timeout=1.0)

    assert client.cid == 16


def test_fixed_cid_gateway_client_is_reusable_across_consecutive_event_loops():
    client = gateway_client.TEEClient(cid=16)

    first = asyncio.run(client._resolved_cid())
    second = asyncio.run(client._resolved_cid())

    assert (first, second) == (16, 16)


@pytest.mark.asyncio
async def test_gateway_client_surfaces_enclave_error_without_status_field(
    monkeypatch,
):
    body = json.dumps({"error": "credential hash mismatch"}).encode()
    client = gateway_client.TEEClient(cid=16)
    rpc_socket = _RPCSocket(len(body).to_bytes(4, "big") + body)
    monkeypatch.setattr(
        gateway_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )

    with pytest.raises(RuntimeError, match="credential hash mismatch"):
        await client._send_rpc("v2_provision_encrypted_secret", {})
    assert rpc_socket.closed is True


@pytest.mark.asyncio
async def test_gateway_client_preserves_structured_enclave_error_type(
    monkeypatch,
):
    body = json.dumps(
        {
            "status": "error",
            "error": "recipient request was already used",
            "error_type": "KMSRecipientV2Error",
        }
    ).encode()
    client = gateway_client.TEEClient(cid=16)
    rpc_socket = _RPCSocket(len(body).to_bytes(4, "big") + body)
    monkeypatch.setattr(
        gateway_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )

    with pytest.raises(gateway_client.TEEEnclaveRPCError) as captured:
        await client._send_rpc("v2_seal_source_add_ingress_credential", {})
    assert captured.value.error_type == "KMSRecipientV2Error"
    assert rpc_socket.closed is True


@pytest.mark.asyncio
async def test_gateway_client_does_not_log_each_successful_vsock_connection(
    monkeypatch,
    capsys,
):
    body = json.dumps({"result": {"status": "healthy"}}).encode()
    client = gateway_client.TEEClient(cid=16)
    rpc_socket = _RPCSocket(len(body).to_bytes(4, "big") + body)
    monkeypatch.setattr(
        gateway_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )

    assert await client._send_rpc("scoring_v2_health", {}) == {
        "status": "healthy"
    }
    assert "Connected to enclave via vsock" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_gateway_client_uses_call_scoped_sockets_for_concurrent_rpc(
    monkeypatch,
):
    sockets = [
        _RPCSocket(
            len(body).to_bytes(4, "big") + body,
            fragment_size=1,
        )
        for body in (
            json.dumps({"result": {"request": 1}}).encode(),
            json.dumps({"result": {"request": 2}}).encode(),
        )
    ]
    sockets_lock = threading.Lock()

    def socket_factory(*_args, **_kwargs):
        with sockets_lock:
            return sockets.pop(0)

    created = list(sockets)
    monkeypatch.setattr(gateway_client.socket, "socket", socket_factory)
    client = gateway_client.TEEClient(cid=17)

    results = await asyncio.gather(
        client._send_rpc("scoring_v2_health", {}),
        client._send_rpc("scoring_v2_health", {}),
    )

    assert sorted(result["request"] for result in results) == [1, 2]
    assert all(rpc_socket.closed for rpc_socket in created)
    assert len({id(rpc_socket) for rpc_socket in created}) == 2


@pytest.mark.asyncio
async def test_gateway_client_socket_io_does_not_block_event_loop(monkeypatch):
    body = json.dumps({"result": {"status": "healthy"}}).encode()
    started = threading.Event()
    release = threading.Event()

    class BlockingSocket(_RPCSocket):
        def recv(self, size):
            started.set()
            if not release.wait(1.0):
                raise TimeoutError("test socket was not released")
            return super().recv(size)

    rpc_socket = BlockingSocket(len(body).to_bytes(4, "big") + body)
    monkeypatch.setattr(
        gateway_client.socket,
        "socket",
        lambda *_args, **_kwargs: rpc_socket,
    )
    client = gateway_client.TEEClient(cid=17)
    request = asyncio.create_task(client._send_rpc("scoring_v2_health", {}))

    assert await asyncio.to_thread(started.wait, 0.5)
    await asyncio.sleep(0)
    release.set()

    assert await request == {"status": "healthy"}


def _send_gateway_rpc(connection, method):
    body = json.dumps({"method": method, "params": {}}).encode()
    connection.sendall(len(body).to_bytes(4, "big") + body)
    response_length = int.from_bytes(
        gateway_client._recv_exact(connection, 4), "big"
    )
    return json.loads(
        gateway_client._recv_exact(connection, response_length).decode()
    )


def test_gateway_vsock_server_does_not_serialize_behind_blocked_rpc(
    monkeypatch,
):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(gateway_service.VSOCK_RPC_LISTEN_BACKLOG)
    listener.settimeout(0.05)
    address = listener.getsockname()
    stop_event = threading.Event()
    blocked = threading.Event()
    release = threading.Event()

    def handle_rpc(method, _params):
        if method == "blocked":
            blocked.set()
            assert release.wait(2.0)
        return {"result": {"method": method}}

    monkeypatch.setattr(gateway_service, "handle_rpc", handle_rpc)
    server = threading.Thread(
        target=gateway_service._serve_vsock_connections,
        args=(listener,),
        kwargs={"stop_event": stop_event, "max_connections": 2},
        daemon=True,
    )
    server.start()
    first = socket.create_connection(address, timeout=1.0)
    second = None
    try:
        first_body = json.dumps({"method": "blocked", "params": {}}).encode()
        first.sendall(len(first_body).to_bytes(4, "big") + first_body)
        assert blocked.wait(0.5)

        second = socket.create_connection(address, timeout=1.0)
        second.settimeout(0.5)
        response = _send_gateway_rpc(second, "health")
        assert response == {"result": {"method": "health"}}
        assert release.is_set() is False
    finally:
        release.set()
        first.close()
        if second is not None:
            second.close()
        stop_event.set()
        listener.close()
        server.join(timeout=2.0)
    assert server.is_alive() is False


def test_gateway_vsock_incomplete_frame_expires(monkeypatch):
    server_socket, client_socket = socket.socketpair()
    monkeypatch.setattr(
        gateway_service,
        "VSOCK_RPC_CONNECTION_TIMEOUT_SECONDS",
        0.05,
    )
    handler = threading.Thread(
        target=gateway_service._handle_vsock_connection,
        args=(server_socket, (3, 1234)),
        daemon=True,
    )
    handler.start()
    client_socket.sendall(b"\x00")
    handler.join(timeout=0.5)
    client_socket.close()

    assert handler.is_alive() is False


def test_gateway_vsock_survives_abandoned_peer_across_worker_rounds(
    monkeypatch,
):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(gateway_service.VSOCK_RPC_LISTEN_BACKLOG)
    listener.settimeout(0.05)
    address = listener.getsockname()
    stop_event = threading.Event()
    monkeypatch.setattr(
        gateway_service,
        "VSOCK_RPC_CONNECTION_TIMEOUT_SECONDS",
        0.25,
    )
    monkeypatch.setattr(
        gateway_service,
        "handle_rpc",
        lambda method, _params: {"result": {"method": method}},
    )
    server = threading.Thread(
        target=gateway_service._serve_vsock_connections,
        args=(listener,),
        kwargs={"stop_event": stop_event, "max_connections": 32},
        daemon=True,
    )
    server.start()
    abandoned = socket.create_connection(address, timeout=1.0)
    abandoned.sendall(b"\x00")

    def run_worker(worker_index):
        connection = socket.create_connection(address, timeout=1.0)
        connection.settimeout(1.0)
        try:
            response = _send_gateway_rpc(
                connection,
                "worker-%s" % worker_index,
            )
            return response["result"]["method"]
        finally:
            connection.close()

    try:
        for round_index in range(3):
            with ThreadPoolExecutor(max_workers=25) as executor:
                observed = list(
                    executor.map(
                        run_worker,
                        range(round_index * 25, (round_index + 1) * 25),
                    )
                )
            assert observed == [
                "worker-%s" % worker_index
                for worker_index in range(
                    round_index * 25,
                    (round_index + 1) * 25,
                )
            ]
    finally:
        abandoned.close()
        stop_event.set()
        listener.close()
        server.join(timeout=2.0)
    assert server.is_alive() is False


def test_validator_vsock_connection_sets_anti_wedge_deadline(monkeypatch):
    body = json.dumps({"command": "health"}).encode()
    rpc_socket = _RPCSocket(len(body).to_bytes(4, "big") + body)
    observed = []

    def settimeout(value):
        observed.append(value)

    rpc_socket.settimeout = settimeout
    monkeypatch.setattr(
        validator_service,
        "handle_request",
        lambda request: {"status": "ok", "command": request["command"]},
    )

    validator_service._handle_vsock_client(rpc_socket, (3, 1234))

    assert observed == [
        validator_service.VSOCK_RPC_RECEIVE_TIMEOUT_SECONDS,
        validator_service.VSOCK_RPC_RESPONSE_TIMEOUT_SECONDS,
    ]
    assert rpc_socket.closed is True
    response_length = int.from_bytes(rpc_socket.request[:4], "big")
    response = json.loads(rpc_socket.request[4:].decode())
    assert response_length == len(rpc_socket.request[4:])
    assert response == {"status": "ok", "command": "health"}
