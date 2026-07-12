from __future__ import annotations

import pytest

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
)
from validator_tee.host import gateway_weight_inputs_v2 as client_module
from validator_tee.host.gateway_weight_inputs_v2 import (
    GatewayWeightInputsV2Error,
    fetch_gateway_weight_inputs_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


class FakeClient:
    def __init__(self):
        self.messages = []

    def sign_application_message_v2(self, message):
        self.messages.append(bytes(message))
        return {
            "purpose": "validator.gateway_weight_inputs.v2",
            "validator_hotkey": HOTKEY,
            "signature": "1" * 128,
            "receipt": {"receipt_hash": "sha256:" + "2" * 64},
        }


def _calculation():
    return {
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "research_lab_allocation_doc": {
            "allocation_hash": "sha256:" + "3" * 64,
        },
    }


def _response(request):
    authority_hash = "sha256:" + "4" * 64
    allocation_parent = {
        "receipt_hash": authority_hash,
        "parent_receipt_hashes": [],
    }
    receipts = [allocation_parent]
    hashes = {}
    for index, category in enumerate(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES), start=10):
        receipt_hash = "sha256:" + ("%064x" % index)
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        receipts.append(
            {
                "receipt_hash": receipt_hash,
                "parent_receipt_hashes": (
                    [authority_hash]
                    if category
                    in {
                        "research_lab_allocation",
                        "champions",
                        "reimbursements",
                        "source_add_rewards",
                    }
                    else []
                ),
                "role": role,
                "purpose": purpose,
                "epoch_id": 100,
            }
        )
        hashes[category] = receipt_hash
    return {
        "request_hash": request["request_hash"],
        "calculation_snapshot_hash": request["calculation_snapshot_hash"],
        "input_receipt_hashes": hashes,
        "gateway_authority_event_hash": authority_hash,
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": receipts,
            "transport_attempts": [],
            "host_operations": [],
        },
    }


@pytest.mark.asyncio
async def test_client_signs_exact_request_and_preserves_complete_receipt_set(monkeypatch):
    monkeypatch.setattr(client_module, "validate_boot_identity", lambda _value: None)
    monkeypatch.setattr(
        client_module, "validate_signed_execution_receipt", lambda _value: None
    )
    monkeypatch.setattr(client_module, "validate_transport_attempt", lambda _value: None)
    monkeypatch.setattr(
        client_module, "validate_host_operation_record", lambda _value: None
    )
    observed = {}

    async def post_json(url, payload, timeout):
        observed.update(url=url, payload=dict(payload), timeout=timeout)
        return _response(payload["request"])

    client = FakeClient()
    result = await fetch_gateway_weight_inputs_v2(
        gateway_url="https://gateway.example",
        calculation_snapshot=_calculation(),
        validator_hotkey=HOTKEY,
        allocation_hash="sha256:" + "3" * 64,
        leaderboard_window_start="2026-07-03T20:00:00Z",
        leaderboard_window_end="2026-07-10T20:00:00Z",
        client=client,
        post_json=post_json,
    )
    assert observed["url"] == "https://gateway.example/weights/inputs/v2"
    assert observed["payload"]["validator_hotkey_signature"] == "1" * 128
    assert observed["payload"]["request"]["calculation_snapshot_hash"] == sha256_json(
        _calculation()
    )
    assert len(client.messages) == 1
    assert set(result["input_receipt_hashes"]) == set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    assert result["gateway_authority_event_hash"] == "sha256:" + "4" * 64


@pytest.mark.asyncio
async def test_client_rejects_public_plaintext_gateway_before_request():
    client = FakeClient()

    async def unused(*_args):
        raise AssertionError("network must not be reached")

    with pytest.raises(GatewayWeightInputsV2Error, match="requires HTTPS"):
        await fetch_gateway_weight_inputs_v2(
            gateway_url="http://52.91.135.79:8000",
            calculation_snapshot=_calculation(),
            validator_hotkey=HOTKEY,
            allocation_hash="sha256:" + "3" * 64,
            leaderboard_window_start="2026-07-03T20:00:00Z",
            leaderboard_window_end="2026-07-10T20:00:00Z",
            client=client,
            post_json=unused,
        )
    assert client.messages == []


@pytest.mark.asyncio
async def test_client_rejects_response_bound_to_another_request(monkeypatch):
    monkeypatch.setattr(client_module, "validate_boot_identity", lambda _value: None)
    monkeypatch.setattr(
        client_module, "validate_signed_execution_receipt", lambda _value: None
    )
    monkeypatch.setattr(client_module, "validate_transport_attempt", lambda _value: None)
    monkeypatch.setattr(
        client_module, "validate_host_operation_record", lambda _value: None
    )

    async def post_json(_url, payload, _timeout):
        response = _response(payload["request"])
        response["request_hash"] = "sha256:" + "f" * 64
        return response

    with pytest.raises(GatewayWeightInputsV2Error, match="does not bind"):
        await fetch_gateway_weight_inputs_v2(
            gateway_url="https://gateway.example",
            calculation_snapshot=_calculation(),
            validator_hotkey=HOTKEY,
            allocation_hash="sha256:" + "3" * 64,
            leaderboard_window_start="2026-07-03T20:00:00Z",
            leaderboard_window_end="2026-07-10T20:00:00Z",
            client=FakeClient(),
            post_json=post_json,
        )
