from __future__ import annotations

import asyncio
from types import SimpleNamespace

import bittensor as bt
import pytest
from bittensor_wallet import Wallet

if not hasattr(bt, "wallet"):
    bt.wallet = Wallet

import neurons.validator as validator_module
from leadpoet_canonical.weights import normalize_to_u16_with_uids
from validator_tee.host.legacy_v1_compat import (
    AUTHORITATIVE_V2_PROTOCOL,
    LEGACY_V1_COMPAT_PROTOCOL,
    LegacyV1EnclaveClient,
    build_legacy_v1_submission,
    normalize_weight_protocol,
    verify_existing_legacy_v1_bundle,
)


class _RawClient:
    def __init__(self):
        self.calls = []

    def _send_request(self, request):
        self.calls.append(dict(request))
        command = request["command"]
        if command == "health":
            return {"status": "ok"}
        if command == "get_public_key":
            return {"public_key": "1" * 64, "code_hash": "2" * 64}
        if command == "sign_weights":
            return {"signature": "3" * 128}
        if command == "get_attestation":
            return {"attestation_b64": "attestation"}
        raise AssertionError(command)


def test_protocol_selection_is_explicit_and_defaults_to_v2():
    assert normalize_weight_protocol(None) == AUTHORITATIVE_V2_PROTOCOL
    assert (
        normalize_weight_protocol("LEGACY_V1_COMPAT")
        == LEGACY_V1_COMPAT_PROTOCOL
    )
    with pytest.raises(RuntimeError, match="VALIDATOR_WEIGHT_PROTOCOL"):
        normalize_weight_protocol("automatic")


def test_legacy_bundle_uses_one_sparse_vector_for_gateway_and_chain():
    raw_client = _RawClient()
    client = LegacyV1EnclaveClient(raw_client)
    prepared = build_legacy_v1_submission(
        client=client,
        netuid=71,
        epoch_id=10,
        block=3945,
        uids=[9, 2, 7],
        weights=[0.75, 0.25, 1e-30],
        validator_hotkey="validator-hotkey",
        sign_binding_message=lambda message: b"signed:" + message[:8],
        expected_chain="wss://entrypoint-finney.opentensor.ai:443",
        validator_version="abcdef0",
    )

    expected_uids, expected_u16 = normalize_to_u16_with_uids(
        [2, 7, 9], [0.25, 1e-30, 0.75]
    )
    assert prepared["payload"]["uids"] == expected_uids
    assert prepared["payload"]["weights_u16"] == expected_u16
    assert prepared["uids"] == expected_uids
    assert len(prepared["chain_weights"]) == len(expected_uids)
    assert all(weight > 0 for weight in prepared["chain_weights"])
    assert [call["command"] for call in raw_client.calls] == [
        "sign_weights",
        "get_public_key",
        "get_attestation",
    ]


def test_duplicate_recovery_requires_the_exact_signed_bundle():
    payload = {
        "netuid": 71,
        "epoch_id": 10,
        "block": 3945,
        "uids": [1, 2],
        "weights_u16": [100, 200],
        "weights_hash": "a" * 64,
        "validator_hotkey": "validator",
        "validator_enclave_pubkey": "b" * 64,
        "validator_signature": "c" * 128,
        "validator_code_hash": "d" * 64,
    }
    existing = {
        **payload,
        "weight_submission_event_hash": "sha256:" + "e" * 64,
    }
    assert verify_existing_legacy_v1_bundle(existing, payload) == existing[
        "weight_submission_event_hash"
    ]
    with pytest.raises(RuntimeError, match="weights_u16"):
        verify_existing_legacy_v1_bundle(
            {**existing, "weights_u16": [100, 201]}, payload
        )


def _compat_validator():
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator._weight_protocol = LEGACY_V1_COMPAT_PROTOCOL
    validator._legacy_v1_client = object()
    validator.config = SimpleNamespace(netuid=71)
    validator.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda message: b"signature",
        )
    )
    return validator


class _Response:
    def __init__(self, status, *, json_body=None, text_body=""):
        self.status = status
        self._json_body = json_body
        self._text_body = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self._json_body

    async def text(self):
        return self._text_body


class _Session:
    def __init__(self, post_response, get_response=None):
        self.post_response = post_response
        self.get_response = get_response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url, *, json):
        self.calls.append(("post", url, json))
        return self.post_response

    def get(self, url):
        self.calls.append(("get", url, None))
        return self.get_response


def test_legacy_gateway_uses_production_v1_endpoint(monkeypatch):
    validator = _compat_validator()
    event_hash = "sha256:" + "1" * 64
    session = _Session(
        _Response(
            200,
            json_body={"weight_submission_event_hash": event_hash},
        )
    )
    monkeypatch.setenv("GATEWAY_URL", "http://52.91.135.79:8000/")
    monkeypatch.setattr(
        validator_module.aiohttp,
        "ClientSession",
        lambda **_kwargs: session,
    )

    payload = {"netuid": 71, "epoch_id": 23915}
    assert asyncio.run(validator._publish_legacy_v1_bundle(payload)) == event_hash
    assert session.calls == [
        (
            "post",
            "http://52.91.135.79:8000/weights/submit",
            payload,
        )
    ]


def test_legacy_gateway_duplicate_requires_exact_recovery(monkeypatch):
    validator = _compat_validator()
    event_hash = "sha256:" + "1" * 64
    payload = {
        "netuid": 71,
        "epoch_id": 23915,
        "block": 8609745,
        "uids": [1],
        "weights_u16": [65535],
        "weights_hash": "a" * 64,
        "validator_hotkey": "validator",
        "validator_enclave_pubkey": "b" * 64,
        "validator_signature": "c" * 128,
        "validator_code_hash": "d" * 64,
    }
    session = _Session(
        _Response(409, text_body="duplicate"),
        _Response(
            200,
            json_body={
                **payload,
                "weight_submission_event_hash": event_hash,
            },
        ),
    )
    monkeypatch.setenv("GATEWAY_URL", "http://52.91.135.79:8000")
    monkeypatch.setattr(
        validator_module.aiohttp,
        "ClientSession",
        lambda **_kwargs: session,
    )

    assert asyncio.run(validator._publish_legacy_v1_bundle(payload)) == event_hash
    assert [call[:2] for call in session.calls] == [
        ("post", "http://52.91.135.79:8000/weights/submit"),
        (
            "get",
            "http://52.91.135.79:8000/weights/latest/71/23915",
        ),
    ]


def test_compat_publishes_before_chain_and_uses_prepared_vector(monkeypatch):
    validator = _compat_validator()
    calls = []
    prepared = {
        "payload": {"epoch_id": 10},
        "uids": [2, 9],
        "weights_u16": [16384, 49151],
        "chain_weights": [0.25, 0.75],
    }

    monkeypatch.setattr(
        validator_module,
        "build_legacy_v1_submission",
        lambda **_kwargs: dict(prepared),
    )
    monkeypatch.setattr(
        validator_module,
        "_current_validator_commit_sha",
        lambda: "f" * 40,
    )

    async def publish(payload):
        calls.append(("publish", payload))
        return "sha256:" + "1" * 64

    async def set_weights(**kwargs):
        calls.append(("chain", kwargs))
        return True

    validator._publish_legacy_v1_bundle = publish
    validator._set_legacy_weights_until_epoch_end = set_weights
    result = asyncio.run(
        validator._publish_and_set_weights(
            snapshot={"epoch_id": 10, "block": 3945},
            host_uids=[2, 7, 9],
            host_weights=[0.25, 1e-30, 0.75],
            allocation_hash="sha256:" + "2" * 64,
            leaderboard_window_start="start",
            leaderboard_window_end="end",
        )
    )
    assert result is True
    assert [call[0] for call in calls] == ["publish", "chain"]
    assert calls[1][1]["uids"] == [2, 9]
    assert calls[1][1]["weights"] == [0.25, 0.75]


def test_compat_gateway_failure_blocks_chain(monkeypatch):
    validator = _compat_validator()
    monkeypatch.setattr(
        validator_module,
        "build_legacy_v1_submission",
        lambda **_kwargs: {
            "payload": {"epoch_id": 10},
            "uids": [2],
            "weights_u16": [65535],
            "chain_weights": [1.0],
        },
    )
    monkeypatch.setattr(
        validator_module,
        "_current_validator_commit_sha",
        lambda: "f" * 40,
    )

    async def publish(_payload):
        return None

    async def unexpected_chain(**_kwargs):
        raise AssertionError("chain submission must remain blocked")

    validator._publish_legacy_v1_bundle = publish
    validator._set_legacy_weights_until_epoch_end = unexpected_chain
    result = asyncio.run(
        validator._publish_and_set_weights(
            snapshot={"epoch_id": 10, "block": 3945},
            host_uids=[2],
            host_weights=[1.0],
            allocation_hash="sha256:" + "2" * 64,
            leaderboard_window_start="start",
            leaderboard_window_end="end",
        )
    )
    assert result is False
