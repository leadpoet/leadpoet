from __future__ import annotations

import pytest
import struct

from validator_tee.host import authoritative_weight_flow_v2 as flow_module
from validator_tee.host.authoritative_weight_flow_v2 import (
    AuthoritativeWeightFlowV2Error,
    finalize_authoritative_weight_publication_v2,
    prepare_authoritative_weight_publication_v2,
    resume_prepared_weight_publication_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
COMPUTED_RECEIPT = "sha256:" + "1" * 64
ROOT = "sha256:" + "2" * 64
EVENT = "sha256:" + "3" * 64


class Client:
    def __init__(self):
        self.compute_requests = []
        self.binding_requests = []

    def compute_authoritative_weights_v2(self, request):
        self.compute_requests.append(dict(request))
        return {
            "weight_snapshot": {"snapshot": True},
            "weight_result": {
                "netuid": 71,
                "epoch_id": 100,
                "block": 36099,
                "uids": [0, 1],
                "weights": [0.8, 0.2],
                "weight_float_bits": [
                    struct.pack("!d", value).hex() for value in (0.8, 0.2)
                ],
                "sparse_uids": [0, 1],
                "sparse_weights_u16": [65535, 16384],
                "weights_hash": "4" * 64,
            },
            "weights_signature": "5" * 128,
            "receipt_graph": {"root_receipt_hash": ROOT, "receipts": []},
            "boot_identity": {
                "signing_pubkey": "6" * 64,
                "build_manifest_hash": "sha256:" + "7" * 64,
                "commit_sha": "8" * 40,
            },
            "weight_authorization_id": "sha256:" + "9" * 64,
            "source_artifacts": [],
        }

    def sign_application_message_v2(self, message, *, parent_receipt_hash=None):
        self.binding_requests.append((bytes(message), parent_receipt_hash))
        return {
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": HOTKEY,
            "signature": "a" * 128,
            "receipt": {"receipt_hash": ROOT},
        }

    def confirm_weight_publication_v2(self, authorization_id):
        assert authorization_id == "sha256:" + "9" * 64
        return {
            "finalization": {
                "epoch_id": 100,
                "weights_hash": "4" * 64,
                "extrinsic_hash": "0x" + "e" * 64,
                "finalized_block": 36105,
            },
            "receipt_graph": {"root_receipt_hash": "sha256:" + "f" * 64},
            "source_artifacts": [],
        }


async def _inputs(**kwargs):
    assert kwargs["validator_hotkey"] == HOTKEY
    return {
        "input_receipt_hashes": {"research_lab_allocation": "sha256:" + "b" * 64},
        "gateway_authority_event_hash": "sha256:" + "c" * 64,
        "upstream_receipt_set": {"receipts": []},
    }


def _bundle(**kwargs):
    assert kwargs["binding_signature_result"]["signature"] == "a" * 128
    response = kwargs["enclave_response"]
    return {
        "schema_version": "leadpoet.published_weight_bundle.v2",
        "receipt_graph": {
            "receipts": [
                {
                    "receipt_hash": COMPUTED_RECEIPT,
                    "purpose": "validator.weights.computed.v2",
                }
            ]
        },
        "weight_result": response["weight_result"],
    }


def _ack(**overrides):
    value = {
        "success": True,
        "epoch_id": 100,
        "weights_count": 2,
        "weights_hash": "4" * 64,
        "weight_receipt_hash": COMPUTED_RECEIPT,
        "weight_submission_event_hash": EVENT,
        "message": "published",
    }
    value.update(overrides)
    return value


@pytest.mark.asyncio
async def test_flow_orders_inputs_compute_parent_binding_and_durable_publication(monkeypatch):
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)
    observed = {}

    order = []

    async def post(url, payload, timeout):
        order.append("post")
        observed.update(url=url, payload=payload, timeout=timeout)
        return _ack()

    def before_publish(prepared):
        order.append("journal")
        assert prepared["weight_authorization_id"] == "sha256:" + "9" * 64
        assert prepared["published_bundle"]["schema_version"].endswith(".v2")

    client = Client()
    result = await prepare_authoritative_weight_publication_v2(
        calculation_snapshot={"epoch_id": 100},
        host_uids=[0, 1],
        host_weights=[0.8, 0.2],
        validator_hotkey=HOTKEY,
        allocation_hash="sha256:" + "d" * 64,
        leaderboard_window_start="2026-07-03T20:00:00Z",
        leaderboard_window_end="2026-07-10T20:00:00Z",
        gateway_url="https://gateway.example",
        expected_chain="wss://entrypoint-finney.opentensor.ai:443",
        client=client,
        fetch_inputs=_inputs,
        post_json=post,
        before_publish=before_publish,
    )
    assert client.compute_requests[0]["gateway_authority_event_hash"] == (
        "sha256:" + "c" * 64
    )
    assert client.binding_requests[0][1] == ROOT
    assert b"version=" + ("8" * 40).encode() in client.binding_requests[0][0]
    assert observed["url"] == "https://gateway.example/weights/submit/v2"
    assert result["uids"] == [0, 1]
    assert result["weight_submission_event_hash"] == EVENT
    assert order == ["journal", "post"]


@pytest.mark.asyncio
async def test_prepared_publication_replays_exact_bundle_and_validates_ack():
    bundle = _bundle(
        enclave_response=Client().compute_authoritative_weights_v2({}),
        validator_hotkey=HOTKEY,
        binding_message="binding",
        binding_signature_result={"signature": "a" * 128},
    )
    observed = {}

    async def post(url, payload, timeout):
        observed.update(url=url, payload=payload, timeout=timeout)
        return _ack()

    acknowledgment = await resume_prepared_weight_publication_v2(
        journal_record={"published_bundle": bundle},
        gateway_url="https://gateway.example",
        post_json=post,
    )
    assert observed["payload"] is bundle
    assert acknowledgment["weight_submission_event_hash"] == EVENT


@pytest.mark.asyncio
async def test_flow_rejects_acknowledgment_for_another_vector(monkeypatch):
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)

    async def post(*_args):
        return _ack(weights_hash="f" * 64)

    with pytest.raises(AuthoritativeWeightFlowV2Error, match="acknowledgment differs"):
        await prepare_authoritative_weight_publication_v2(
            calculation_snapshot={"epoch_id": 100},
            host_uids=[0, 1],
            host_weights=[0.8, 0.2],
            validator_hotkey=HOTKEY,
            allocation_hash="sha256:" + "d" * 64,
            leaderboard_window_start="2026-07-03T20:00:00Z",
            leaderboard_window_end="2026-07-10T20:00:00Z",
            gateway_url="https://gateway.example",
            expected_chain="wss://entrypoint-finney.opentensor.ai:443",
            client=Client(),
            fetch_inputs=_inputs,
            post_json=post,
        )


@pytest.mark.asyncio
async def test_flow_rejects_plaintext_gateway_before_enclave_work(monkeypatch):
    client = Client()
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)
    with pytest.raises(AuthoritativeWeightFlowV2Error, match="requires HTTPS"):
        await prepare_authoritative_weight_publication_v2(
            calculation_snapshot={"epoch_id": 100},
            host_uids=[0, 1],
            host_weights=[0.8, 0.2],
            validator_hotkey=HOTKEY,
            allocation_hash="sha256:" + "d" * 64,
            leaderboard_window_start="2026-07-03T20:00:00Z",
            leaderboard_window_end="2026-07-10T20:00:00Z",
            gateway_url="http://52.91.135.79:8000",
            expected_chain="wss://entrypoint-finney.opentensor.ai:443",
            client=client,
            fetch_inputs=_inputs,
        )
    assert client.compute_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_uids", "host_weights", "error"),
    [
        ([1, 0], [0.2, 0.8], "UID order differ"),
        ([0, 1], [0.8000000000000002, 0.2], "float weights differ"),
    ],
)
async def test_flow_rejects_any_host_enclave_vector_difference(
    monkeypatch, host_uids, host_weights, error
):
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)

    async def post(*_args):
        raise AssertionError("mismatched weights must not be published")

    with pytest.raises(AuthoritativeWeightFlowV2Error, match=error):
        await prepare_authoritative_weight_publication_v2(
            calculation_snapshot={"epoch_id": 100},
            host_uids=host_uids,
            host_weights=host_weights,
            validator_hotkey=HOTKEY,
            allocation_hash="sha256:" + "d" * 64,
            leaderboard_window_start="2026-07-03T20:00:00Z",
            leaderboard_window_end="2026-07-10T20:00:00Z",
            gateway_url="https://gateway.example",
            expected_chain="wss://entrypoint-finney.opentensor.ai:443",
            client=Client(),
            fetch_inputs=_inputs,
            post_json=post,
        )


@pytest.mark.asyncio
async def test_finalization_requires_exact_enclave_and_gateway_ack(monkeypatch):
    monkeypatch.setattr(
        flow_module,
        "build_weight_finalization_submission_v2",
        lambda **kwargs: {
            "schema_version": "leadpoet.weight_finalization_submission.v2",
            **kwargs,
        },
    )
    observed = {}

    async def post(url, payload, timeout):
        observed.update(url=url, payload=payload, timeout=timeout)
        return {
            "success": True,
            "epoch_id": 100,
            "weights_hash": "4" * 64,
            "extrinsic_hash": "0x" + "e" * 64,
            "finalized_block": 36105,
            "weight_submission_event_hash": EVENT,
            "weight_finalization_event_hash": "sha256:" + "a" * 64,
            "message": "finalized",
        }

    result = await finalize_authoritative_weight_publication_v2(
        prepared_publication={
            "weight_authorization_id": "sha256:" + "9" * 64,
            "weight_submission_event_hash": EVENT,
        },
        validator_hotkey=HOTKEY,
        gateway_url="https://gateway.example",
        client=Client(),
        post_json=post,
    )
    assert observed["url"] == "https://gateway.example/weights/finalize/v2"
    assert result["acknowledgment"]["weight_submission_event_hash"] == EVENT


@pytest.mark.asyncio
async def test_finalization_rejects_gateway_ack_for_another_extrinsic(monkeypatch):
    monkeypatch.setattr(
        flow_module,
        "build_weight_finalization_submission_v2",
        lambda **kwargs: kwargs,
    )

    async def post(*_args):
        return {
            "success": True,
            "epoch_id": 100,
            "weights_hash": "4" * 64,
            "extrinsic_hash": "0x" + "0" * 64,
            "finalized_block": 36105,
            "weight_submission_event_hash": EVENT,
            "weight_finalization_event_hash": "sha256:" + "a" * 64,
            "message": "finalized",
        }

    with pytest.raises(AuthoritativeWeightFlowV2Error, match="acknowledgment differs"):
        await finalize_authoritative_weight_publication_v2(
            prepared_publication={
                "weight_authorization_id": "sha256:" + "9" * 64,
                "weight_submission_event_hash": EVENT,
            },
            validator_hotkey=HOTKEY,
            gateway_url="https://gateway.example",
            client=Client(),
            post_json=post,
        )
