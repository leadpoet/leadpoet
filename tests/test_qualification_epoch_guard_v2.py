from __future__ import annotations

import base64
import json

import pytest

from Leadpoet.utils.subnet_epoch import STATEFUL_EPOCH_MODE, SubnetEpochCutover
from gateway.tee.qualification_epoch_guard_v2 import (
    QualificationEpochGuardV2,
    QualificationEpochGuardV2Error,
)
from leadpoet_canonical.chain_source_v2 import subnet_epoch_storage_key


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


class StatefulTransport:
    GENESIS = "0x" + "1" * 64
    FINALIZED = "0x" + "2" * 64
    CUTOVER = "0x" + "3" * 64
    PREDECESSOR = "0x" + "4" * 64

    def __init__(self, *, current_index=52, cutover_index=50):
        self.current_index = current_index
        self.cutover_index = cutover_index
        self.calls = []
        self.names = {
            subnet_epoch_storage_key(storage_name=name, netuid=71): name
            for name in (
                "Tempo",
                "LastEpochBlock",
                "PendingEpochAt",
                "SubnetEpochIndex",
                "BlocksSinceLastStep",
            )
        }

    @staticmethod
    def _header(block, parent):
        return {
            "number": hex(block),
            "stateRoot": "0x" + "a" * 64,
            "parentHash": parent,
            "extrinsicsRoot": "0x" + "c" * 64,
        }

    @staticmethod
    def _encode(name, value):
        width = 2 if name == "Tempo" else 8
        return "0x" + int(value).to_bytes(width, "little").hex()

    def execute_http(self, **request):
        self.calls.append(request)
        payload = json.loads(request["body"])
        method = payload["method"]
        params = payload["params"]
        request_id = payload["id"]
        if method == "chain_getFinalizedHead":
            result = self.FINALIZED
        elif method == "chain_getBlockHash":
            result = self.GENESIS if params == [0] else (
                self.CUTOVER if params == [8000] else self.PREDECESSOR
            )
        elif method == "chain_getHeader":
            if params == [self.FINALIZED]:
                result = self._header(9000, "0x" + "9" * 64)
            else:
                result = self._header(8000, self.PREDECESSOR)
        elif method == "state_getStorage":
            name = self.names[params[0]]
            at_hash = params[1]
            if at_hash == self.CUTOVER:
                values = {"SubnetEpochIndex": self.cutover_index, "LastEpochBlock": 8000}
            elif at_hash == self.PREDECESSOR:
                values = {"SubnetEpochIndex": 49}
            else:
                values = {
                    "Tempo": 360,
                    "LastEpochBlock": 8800,
                    "PendingEpochAt": 0,
                    "SubnetEpochIndex": self.current_index,
                    "BlocksSinceLastStep": 200,
                }
            result = self._encode(name, values[name])
        else:  # pragma: no cover - makes unexpected RPC expansion obvious
            raise AssertionError(method)
        body = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "result": result},
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "body_b64": base64.b64encode(body).decode("ascii"),
        }


def _stateful_authority():
    cutover = SubnetEpochCutover(
        network_genesis_hash=StatefulTransport.GENESIS,
        netuid=71,
        cutover_block=8000,
        cutover_block_hash=StatefulTransport.CUTOVER,
        first_subnet_epoch_index=50,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )
    return {"mode": STATEFUL_EPOCH_MODE, "cutover": cutover.to_dict()}


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


def test_stateful_guard_maps_exact_finalized_official_epoch():
    transport = StatefulTransport()
    guard = QualificationEpochGuardV2(
        transport,
        epoch_authority=_stateful_authority(),
        netuid=71,
    )

    assert guard(103, 2) is False
    assert guard(102, 2) is True
    first_observation = transport.calls[:14]
    requests = [json.loads(call["body"]) for call in first_observation]
    archive_calls = [
        call
        for call in first_observation
        if call["url"] == "https://archive.chain.opentensor.ai/"
    ]
    live_calls = [
        call
        for call in first_observation
        if call["url"] == "https://entrypoint-finney.opentensor.ai/"
    ]
    assert len(archive_calls) == 7
    assert len(live_calls) == 7
    assert all(
        json.loads(call["body"])["method"] != "chain_getFinalizedHead"
        for call in archive_calls
    )
    assert all(
        not (
            json.loads(call["body"])["method"] == "state_getStorage"
            and json.loads(call["body"])["params"][1]
            in {StatefulTransport.CUTOVER, StatefulTransport.PREDECESSOR}
        )
        for call in live_calls
    )
    current_storage = [
        request for request in requests
        if request["method"] == "state_getStorage"
        and request["params"][1] == StatefulTransport.FINALIZED
    ]
    assert len(current_storage) == 5
    assert all(request["params"][1] == StatefulTransport.FINALIZED for request in current_storage)


def test_stateful_guard_rejects_invented_cutover_mapping():
    guard = QualificationEpochGuardV2(
        StatefulTransport(cutover_index=51),
        epoch_authority=_stateful_authority(),
        netuid=71,
    )
    with pytest.raises(QualificationEpochGuardV2Error, match="official transition"):
        guard(103, 0)
