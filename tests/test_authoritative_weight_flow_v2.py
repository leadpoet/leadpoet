from __future__ import annotations

import pytest
import struct

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host import authoritative_weight_flow_v2 as flow_module
from validator_tee.host.authoritative_weight_flow_v2 import (
    AuthoritativeWeightFlowV2Error,
    finalize_authoritative_weight_publication_v2,
    prepare_authoritative_weight_publication_v2,
    publish_stateful_epoch_evidence_v1,
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


def _epoch_evidence():
    graph = {"root_receipt_hash": "sha256:" + "5" * 64, "receipts": []}
    boundary = {
        "subnet_epoch_index": 35,
        "settlement_epoch_id": 100,
        "current_block": 36_000,
    }
    return {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": HOTKEY,
        "bundle_hash": "sha256:" + "6" * 64,
        "cutover_mapping_hash": "sha256:" + "7" * 64,
        "epoch_authority": {**boundary, "current_block": 36_099},
        "epoch_authority_hash": "sha256:" + "8" * 64,
        "epoch_authority_receipt_hash": "sha256:" + "9" * 64,
        "epoch_boundary": boundary,
        "epoch_boundary_hash": "sha256:" + "a" * 64,
        "epoch_boundary_receipt_hash": "sha256:" + "b" * 64,
        "receipt_graph": graph,
    }


def _epoch_ack(evidence):
    return {
        "schema_version": "leadpoet.subnet_epoch_boundary_ack.v1",
        "bundle_hash": evidence["bundle_hash"],
        "mapping_hash": evidence["cutover_mapping_hash"],
        "subnet_epoch_index": evidence["epoch_boundary"]["subnet_epoch_index"],
        "settlement_epoch_id": evidence["epoch_boundary"]["settlement_epoch_id"],
        "boundary_block": evidence["epoch_boundary"]["current_block"],
        "epoch_authority_hash": evidence["epoch_authority_hash"],
        "epoch_authority_receipt_hash": evidence[
            "epoch_authority_receipt_hash"
        ],
        "boundary_hash": evidence["epoch_boundary_hash"],
        "boundary_receipt_hash": evidence["epoch_boundary_receipt_hash"],
        "receipt_graph_hash": sha256_json(evidence["receipt_graph"]),
        "durable_readback_hash": "sha256:" + "c" * 64,
    }


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
async def test_stateful_recovery_replays_bundle_then_epoch_evidence_before_return():
    bundle = _bundle(
        enclave_response=Client().compute_authoritative_weights_v2({}),
        validator_hotkey=HOTKEY,
        binding_message="binding",
        binding_signature_result={"signature": "a" * 128},
    )
    evidence = _epoch_evidence()
    calls = []

    async def post(url, payload, timeout):
        calls.append((url, payload, timeout))
        if url.endswith("/weights/submit/v2"):
            return _ack()
        assert url.endswith("/weights/subnet-epoch/boundary/v1")
        return _epoch_ack(evidence)

    acknowledgment = await resume_prepared_weight_publication_v2(
        journal_record={
            "published_bundle": bundle,
            "epoch_evidence": evidence,
        },
        gateway_url="https://gateway.example",
        post_json=post,
    )
    assert acknowledgment["weight_submission_event_hash"] == EVENT
    assert [item[0].rsplit("/", 3)[-3:] for item in calls] == [
        ["weights", "submit", "v2"],
        ["subnet-epoch", "boundary", "v1"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_field",
    (
        "epoch_authority_hash",
        "epoch_authority_receipt_hash",
        "epoch_boundary_hash",
    ),
)
async def test_stateful_epoch_evidence_ack_mismatch_fails_closed(changed_field):
    evidence = _epoch_evidence()

    async def post(_url, _payload, _timeout):
        return _epoch_ack(
            {**evidence, changed_field: "sha256:" + "d" * 64}
        )

    with pytest.raises(AuthoritativeWeightFlowV2Error, match="evidence acknowledgment"):
        await publish_stateful_epoch_evidence_v1(
            epoch_evidence=evidence,
            gateway_url="https://gateway.example",
            post_json=post,
        )


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
