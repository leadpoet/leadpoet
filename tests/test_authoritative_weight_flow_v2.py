from __future__ import annotations

import base64
import gzip
import json
import struct

import pytest

from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    validate_weight_transport_authorization_v2,
    weight_transport_authorization_message_v2,
)
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
FINALIZATION_SCAN = "sha256:" + "d" * 64


def test_small_gateway_authority_request_uses_compact_uncompressed_json():
    payload = {
        "receipt_graph": [
            {"receipt_hash": "sha256:" + str(index).zfill(64)}
            for index in range(10)
        ]
    }

    body, headers, logical_body = flow_module._encode_json_request(payload)

    assert headers == {"Content-Type": "application/json"}
    assert body == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert logical_body == body
    assert len(body) < len(json.dumps(payload).encode("utf-8"))
    assert json.loads(body) == payload


def test_large_gateway_authority_request_uses_bounded_gzip_transport():
    limit = 10 * 1024 * 1024
    payload = {"receipt_graph": ["x" * 30] * 315_000}

    body, headers, logical_body = flow_module._encode_json_request(payload)

    assert len(body) < limit
    assert len(logical_body) < limit
    assert len(json.dumps(payload).encode("utf-8")) > limit
    assert headers == {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
    }
    assert gzip.decompress(body) == logical_body


def test_gateway_authority_request_fails_closed_at_both_size_limits(
    monkeypatch,
):
    monkeypatch.setattr(
        flow_module,
        "MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES",
        64,
    )
    with pytest.raises(
        AuthoritativeWeightFlowV2Error,
        match="logical size limit",
    ):
        flow_module._encode_json_request({"value": "x" * 64})

    monkeypatch.setattr(
        flow_module,
        "MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES",
        1024,
    )
    monkeypatch.setattr(flow_module, "MAX_WEIGHT_TRANSPORT_WIRE_BYTES", 8)
    monkeypatch.setattr(flow_module, "_COMPRESS_REQUEST_MIN_BYTES", 1)
    with pytest.raises(
        AuthoritativeWeightFlowV2Error,
        match="compressed wire size limit",
    ):
        flow_module._encode_json_request({"value": "not-compressible-enough"})


@pytest.mark.asyncio
async def test_post_json_sends_preencoded_compact_body(monkeypatch):
    import aiohttp

    observed = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def text(self):
            return '{"success":true}'

        async def json(self):
            return {"success": True}

    class Session:
        def __init__(self, **kwargs):
            observed["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            observed["url"] = url
            observed["request"] = kwargs
            return Response()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    payload = {"z": [1, 2], "a": {"value": True}}

    result = await flow_module._post_json(
        "https://gateway.example/weights/submit/v2",
        payload,
        30.0,
    )

    assert result == {"success": True}
    assert observed["request"] == {
        "data": b'{"a":{"value":true},"z":[1,2]}',
        "headers": {"Content-Type": "application/json"},
    }
    assert observed["session"]["trust_env"] is False


@pytest.mark.asyncio
async def test_post_json_authorizes_exact_compressed_transport(monkeypatch):
    import aiohttp

    observed = {}

    class Enclave:
        def sign_application_message_v2(self, message):
            observed["signed_message"] = bytes(message)
            return {
                "purpose": "validator.gateway_weight_transport.v2",
                "validator_hotkey": HOTKEY,
                "signature": "a" * 128,
            }

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def text(self):
            return '{"success":true}'

        async def json(self):
            return {"success": True}

    class Session:
        def __init__(self, **kwargs):
            observed["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            observed["url"] = url
            observed["request"] = kwargs
            return Response()

    monkeypatch.setattr(flow_module, "ValidatorEnclaveClient", Enclave)
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    payload = {
        "validator_hotkey": HOTKEY,
        "receipt_graph": ["x" * (1024 * 1024)],
    }

    result = await flow_module._post_json(
        "https://gateway.example/weights/submit/v2",
        payload,
        30.0,
    )

    request = observed["request"]
    logical_body = gzip.decompress(request["data"])
    authorization = validate_weight_transport_authorization_v2(
        json.loads(
            base64.b64decode(
                request["headers"]["X-Leadpoet-Weight-Transport"],
                validate=True,
            )
        )
    )
    assert result == {"success": True}
    assert json.loads(logical_body) == payload
    assert authorization["path"] == "/weights/submit/v2"
    assert authorization["wire_body_hash"] == sha256_bytes(request["data"])
    assert authorization["wire_body_bytes"] == len(request["data"])
    assert authorization["logical_body_hash"] == sha256_bytes(logical_body)
    assert authorization["logical_body_bytes"] == len(logical_body)
    assert observed["signed_message"] == (
        weight_transport_authorization_message_v2(
            authorization
        ).encode("utf-8")
    )
    assert request["headers"]["X-Leadpoet-Weight-Transport-Signature"] == (
        "a" * 128
    )
    assert request["headers"]["Content-Encoding"] == "gzip"


@pytest.mark.asyncio
async def test_compressed_transport_rejects_invalid_enclave_authorization(
    monkeypatch,
):
    class Enclave:
        def sign_application_message_v2(self, _message):
            return {
                "purpose": "validator.gateway_weight_transport.v2",
                "validator_hotkey": HOTKEY,
                "signature": "short",
            }

    monkeypatch.setattr(flow_module, "ValidatorEnclaveClient", Enclave)
    with pytest.raises(
        AuthoritativeWeightFlowV2Error,
        match="did not authorize",
    ):
        await flow_module._post_json(
            "https://gateway.example/weights/submit/v2",
            {
                "validator_hotkey": HOTKEY,
                "receipt_graph": ["x" * (1024 * 1024)],
            },
            30.0,
        )


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

    def confirm_weight_publication_v2(
        self,
        authorization_id,
        *,
        finalization_scan_id,
        compact_ancestry_context=None,
    ):
        assert authorization_id == "sha256:" + "9" * 64
        assert finalization_scan_id == FINALIZATION_SCAN
        assert compact_ancestry_context is None
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


def _verified_bundle(_bundle):
    return {"weight_receipt_hash": COMPUTED_RECEIPT}


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
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )
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

    def inputs_verified(inputs):
        order.append("inputs")
        assert inputs["gateway_authority_event_hash"] == "sha256:" + "c" * 64

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
        on_inputs_verified=inputs_verified,
    )
    assert client.compute_requests[0]["gateway_authority_event_hash"] == (
        "sha256:" + "c" * 64
    )
    assert client.binding_requests[0][1] == ROOT
    assert b"version=" + ("8" * 40).encode() in client.binding_requests[0][0]
    assert observed["url"] == "https://gateway.example/weights/submit/v2"
    assert observed["timeout"] == 600.0
    assert result["uids"] == [0, 1]
    assert result["weight_submission_event_hash"] == EVENT
    assert order == ["inputs", "journal", "post"]


@pytest.mark.asyncio
async def test_flow_replays_verified_gateway_inputs_without_network_fetch(
    monkeypatch,
):
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )
    prepared_inputs = await _inputs(validator_hotkey=HOTKEY)

    async def fetch_must_not_run(**_kwargs):
        raise AssertionError("verified gateway inputs must not be fetched again")

    async def post(_url, _payload, _timeout):
        return _ack()

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
        client=Client(),
        fetch_inputs=fetch_must_not_run,
        post_json=post,
        prepared_gateway_inputs=prepared_inputs,
        on_inputs_verified=lambda _inputs: (_ for _ in ()).throw(
            AssertionError("replayed inputs must not be recorded as newly fetched")
        ),
    )
    assert result["uids"] == [0, 1]


@pytest.mark.asyncio
async def test_flow_refetches_after_enclave_rejects_shape_valid_gateway_inputs(
    monkeypatch,
):
    monkeypatch.setattr(flow_module, "build_authoritative_weight_bundle_v2", _bundle)
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )
    invalid_inputs = await _inputs(validator_hotkey=HOTKEY)
    invalid_inputs["gateway_authority_event_hash"] = "sha256:" + "d" * 64
    valid_inputs = await _inputs(validator_hotkey=HOTKEY)
    responses = [invalid_inputs, valid_inputs]
    fetched = []
    journaled = []

    async def fetch(**_kwargs):
        response = responses[len(fetched)]
        fetched.append(response)
        return response

    class RejectFirstInputs(Client):
        def compute_authoritative_weights_v2(self, request):
            if request["gateway_authority_event_hash"] == (
                invalid_inputs["gateway_authority_event_hash"]
            ):
                raise RuntimeError("validator enclave rejected gateway ancestry")
            return super().compute_authoritative_weights_v2(request)

    async def post(_url, _payload, _timeout):
        return _ack()

    def record_gateway_inputs(inputs):
        journaled.append(dict(inputs))

    client = RejectFirstInputs()
    kwargs = {
        "calculation_snapshot": {"epoch_id": 100},
        "host_uids": [0, 1],
        "host_weights": [0.8, 0.2],
        "validator_hotkey": HOTKEY,
        "allocation_hash": "sha256:" + "d" * 64,
        "leaderboard_window_start": "2026-07-03T20:00:00Z",
        "leaderboard_window_end": "2026-07-10T20:00:00Z",
        "gateway_url": "https://gateway.example",
        "expected_chain": "wss://entrypoint-finney.opentensor.ai:443",
        "client": client,
        "fetch_inputs": fetch,
        "post_json": post,
        "on_inputs_verified": record_gateway_inputs,
    }

    with pytest.raises(
        AuthoritativeWeightFlowV2Error,
        match="preparation failed closed",
    ) as rejected:
        await prepare_authoritative_weight_publication_v2(**kwargs)

    assert isinstance(rejected.value.__cause__, RuntimeError)
    assert str(rejected.value.__cause__) == (
        "validator enclave rejected gateway ancestry"
    )
    assert journaled == []
    result = await prepare_authoritative_weight_publication_v2(**kwargs)

    assert result["uids"] == [0, 1]
    assert fetched == [invalid_inputs, valid_inputs]
    assert journaled == [valid_inputs]


@pytest.mark.asyncio
async def test_prepared_publication_replays_exact_bundle_and_validates_ack(
    monkeypatch,
):
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )
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
async def test_stateful_recovery_replays_bundle_then_epoch_evidence_before_return(
    monkeypatch,
):
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )
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
    monkeypatch.setattr(
        flow_module,
        "validate_published_weight_bundle_v2",
        _verified_bundle,
    )

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


def test_host_vector_accepts_unsorted_full_vector_with_canonical_sparse_order():
    host_uids = [7, 0, 3]
    host_weights = [0.2, 0.7, 0.1]
    sparse_uids, sparse_weights = flow_module.normalize_to_u16_with_uids_pure(
        [0, 3, 7],
        [0.7, 0.1, 0.2],
    )

    flow_module._verify_host_vector(
        host_uids=host_uids,
        host_weights=host_weights,
        enclave_result={
            "uids": host_uids,
            "weight_float_bits": [
                struct.pack("!d", value).hex() for value in host_weights
            ],
            "sparse_uids": sparse_uids,
            "sparse_weights_u16": sparse_weights,
        },
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
        finalization_scan_id=FINALIZATION_SCAN,
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
            finalization_scan_id=FINALIZATION_SCAN,
            validator_hotkey=HOTKEY,
            gateway_url="https://gateway.example",
            client=Client(),
            post_json=post,
        )
