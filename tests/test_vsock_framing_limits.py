import asyncio
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway" / "tee"))

from gateway.tee import tee_service as gateway_service
from gateway.utils import tee_client as gateway_client


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
        await client._send_rpc("v2_provision_encrypted_secret", {})
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
