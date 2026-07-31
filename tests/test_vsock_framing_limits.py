import json
import os
import zlib
from pathlib import Path
import sys

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
    assert validator_client.MAX_RPC_REQUEST_BYTES == gateway_client.MAX_RPC_REQUEST_BYTES
    assert validator_service.MAX_RPC_REQUEST_BYTES == gateway_service.MAX_RPC_REQUEST_BYTES
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
        == 64 * 1024 * 1024
    )


def test_validator_receives_production_sized_receipt_graph_request():
    frame_limit = 16 * 1024 * 1024
    request = {
        "command": "compute_authoritative_weights_v2",
        "weight_request": {
            "upstream_receipt_set": {
                "transport_attempts": ["x" * frame_limit],
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


def test_validator_large_response_uses_same_bounded_compressed_frame():
    body = json.dumps({"status": "ok", "receipt_graph": "x" * (17 * 1024 * 1024)}).encode()
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


@pytest.mark.asyncio
async def test_gateway_client_surfaces_enclave_error_without_status_field():
    body = json.dumps({"error": "credential hash mismatch"}).encode()
    client = gateway_client.TEEClient(cid=16)
    client._socket = _RPCSocket(len(body).to_bytes(4, "big") + body)

    async def already_connected():
        return None

    client._ensure_connected = already_connected

    with pytest.raises(RuntimeError, match="credential hash mismatch"):
        await client._send_rpc("v2_provision_encrypted_secret", {})
