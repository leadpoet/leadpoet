from __future__ import annotations

import base64
import json

import pytest

from gateway.tee.qualification_epoch_guard_v2 import (
    QualificationEpochGuardV2,
    QualificationEpochGuardV2Error,
)


class Transport:
    def __init__(self, *, block=36000, failure=None, status=200):
        self.block = block
        self.failure = failure
        self.status = status
        self.calls = []

    def execute_http(self, **request):
        self.calls.append(request)
        if self.failure:
            return {
                "terminal_status": "transport_failure",
                "failure_code": self.failure,
            }
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "number": hex(self.block),
                    "stateRoot": "0x" + "a" * 64,
                    "parentHash": "0x" + "b" * 64,
                    "extrinsicsRoot": "0x" + "c" * 64,
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "terminal_status": "authenticated_response",
            "http_status": self.status,
            "body_b64": base64.b64encode(body).decode("ascii"),
        }


def test_guard_matches_legacy_epoch_greater_than_semantics():
    same = Transport(block=100 * 360 + 359)
    guard = QualificationEpochGuardV2(same)
    assert guard(100, 2) is False
    request = json.loads(same.calls[0]["body"])
    assert request == {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "chain_getHeader",
        "params": [],
    }
    assert same.calls[0]["url"] == "https://entrypoint-finney.opentensor.ai/"

    advanced = QualificationEpochGuardV2(Transport(block=101 * 360))
    assert advanced(100, 2) is True


@pytest.mark.parametrize(
    "transport,match",
    [
        (Transport(failure="timeout"), "transport failed"),
        (Transport(status=503), "authenticated HTTP 503"),
    ],
)
def test_guard_fails_closed_without_authenticated_chain_header(transport, match):
    with pytest.raises(QualificationEpochGuardV2Error, match=match):
        QualificationEpochGuardV2(transport)(100, 0)
