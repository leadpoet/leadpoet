import json
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


def test_validator_still_accepts_legacy_eof_request():
    body = json.dumps({"command": "health"}).encode()
    request, framed = validator_service._receive_request(_FragmentSocket(body, 3))
    assert request == {"command": "health"}
    assert framed is False


def test_validator_rejects_oversized_frame_before_reading_body():
    prefix = (validator_service.MAX_RPC_REQUEST_BYTES + 1).to_bytes(4, "big")
    with pytest.raises(ValueError, match="outside"):
        validator_service._receive_request(_FragmentSocket(prefix, 4))


def test_validator_client_exact_reader_handles_fragments():
    assert validator_client._recv_exact(_FragmentSocket(b"response", 1), 8) == b"response"
