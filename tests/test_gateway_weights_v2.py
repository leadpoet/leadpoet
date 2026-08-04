import asyncio
import gzip
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api import weights as weights_api
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    DISABLED_LEADERBOARD_WINDOW_V1,
    build_weight_inputs_request_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
)


COMMIT = "a" * 40
BUILD_MANIFEST = "sha256:" + "b" * 64
ROOT_RECEIPT = "sha256:" + "c" * 64
WEIGHT_RECEIPT = "sha256:" + "8" * 64
BUNDLE_HASH = "sha256:" + "d" * 64
DURABLE_HASH = "sha256:" + "e" * 64
EVENT_HASH = "sha256:" + "f" * 64
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
VALIDATOR_BOOT_HASH = "sha256:" + "0" * 64
COMPACT_SUBMISSION_HASH = "sha256:" + "1" * 64
LINEAGE_ID = "sha256:" + "2" * 64


def _request(*, accept_encoding: str = "", query_string: bytes = b"") -> Request:
    headers = []
    if accept_encoding:
        headers.append((b"accept-encoding", accept_encoding.encode("ascii")))
    return Request(
        {
            "type": "http",
            "headers": headers,
            "query_string": query_string,
        }
    )


def _legacy_submission():
    return weights_api.WeightSubmission(
        netuid=71,
        epoch_id=300,
        block=108350,
        uids=[],
        weights_u16=[],
        weights_hash="5" * 64,
        validator_hotkey="validator-hotkey",
        validator_enclave_pubkey="1" * 64,
        validator_signature="2" * 128,
        validator_attestation_b64="attestation",
        validator_code_hash="3" * 64,
        binding_message="binding",
        validator_hotkey_signature="4" * 128,
    )


@pytest.mark.asyncio
async def test_v1_weight_endpoint_is_retired_by_default(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_TEE_PROTOCOL", raising=False)
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights(_legacy_submission())
    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_v1_weight_endpoint_remains_retired_for_legacy_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1_compat")
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights(_legacy_submission())
    assert exc.value.status_code == 410



def _submission():
    return weights_api.WeightSubmissionV2(
        schema_version=PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        validator_hotkey="validator-hotkey",
        binding_message=create_binding_message(
            netuid=71,
            chain=weights_api.EXPECTED_CHAIN,
            enclave_pubkey="1" * 64,
            validator_code_hash=BUILD_MANIFEST,
            version=COMMIT,
        ),
        validator_hotkey_signature="2" * 128,
        weight_snapshot={},
        weight_result={},
        weights_signature="3" * 128,
        receipt_graph={
            "boot_identities": [
                {
                    "physical_role": "validator_weights",
                    "commit_sha": COMMIT,
                    "build_manifest_hash": BUILD_MANIFEST,
                    "boot_identity_hash": VALIDATOR_BOOT_HASH,
                }
            ]
        },
    )


def _verified():
    return {
        "bundle_hash": BUNDLE_HASH,
        "root_receipt_hash": ROOT_RECEIPT,
        "weight_receipt_hash": WEIGHT_RECEIPT,
        "snapshot_hash": "sha256:" + "4" * 64,
        "validator_hotkey": "validator-hotkey",
        "validator_enclave_pubkey": "1" * 64,
        "netuid": 71,
        "epoch_id": 300,
        "block": 108350,
        "uids": [0],
        "weights_u16": [65535],
        "weights_hash": "5" * 64,
        "validator_boot_identity_hash": VALIDATOR_BOOT_HASH,
    }


def _validator_boot():
    return {
        "physical_role": "validator_weights",
        "commit_sha": COMMIT,
        "build_manifest_hash": BUILD_MANIFEST,
        "boot_identity_hash": VALIDATOR_BOOT_HASH,
    }


def test_submission_selects_computing_boot_from_historical_validator_ancestry(
    monkeypatch,
):
    submission = _submission()
    historical_boot = {
        "physical_role": "validator_weights",
        "commit_sha": "9" * 40,
        "build_manifest_hash": "sha256:" + "8" * 64,
        "boot_identity_hash": "sha256:" + "7" * 64,
    }
    submission.receipt_graph["boot_identities"].insert(0, historical_boot)
    monkeypatch.setattr(
        weights_api,
        "validate_published_weight_bundle_v2",
        lambda *_args, **_kwargs: _verified(),
    )
    lineage_verified = []
    dynamic_verified = []
    monkeypatch.setattr(
        weights_api,
        "_build_authoritative_v2_receipt_boot_verifier",
        lambda _graph: lineage_verified.append(_graph)
        or (lambda identity: identity),
    )
    monkeypatch.setattr(
        weights_api,
        "_verify_authoritative_v2_boot",
        lambda identity: dynamic_verified.append(identity) or identity,
    )

    verified, selected = weights_api._validate_authoritative_v2_submission(
        submission
    )

    assert verified["validator_boot_identity_hash"] == VALIDATOR_BOOT_HASH
    assert selected == _validator_boot()
    assert lineage_verified == [submission.receipt_graph]
    assert dynamic_verified == [_validator_boot()]


def test_submission_fails_when_computing_boot_is_not_dynamically_rebuilt(
    monkeypatch,
):
    submission = _submission()
    monkeypatch.setattr(
        weights_api,
        "validate_published_weight_bundle_v2",
        lambda *_args, **_kwargs: _verified(),
    )
    monkeypatch.setattr(
        weights_api,
        "_build_authoritative_v2_receipt_boot_verifier",
        lambda _graph: lambda identity: identity,
    )
    monkeypatch.setattr(
        weights_api,
        "_verify_authoritative_v2_boot",
        lambda _identity: (_ for _ in ()).throw(
            ValueError("validator PCR0 is absent from the dynamic Git build cache")
        ),
    )

    with pytest.raises(ValueError, match="dynamic Git build cache"):
        weights_api._validate_authoritative_v2_submission(submission)


class _Subtensor:
    def get_current_block(self):
        return 108350


def _weight_inputs_authorization(
    *,
    leaderboard_window_start: str = "2026-07-03T20:00:00Z",
    leaderboard_window_end: str = "2026-07-10T20:00:00Z",
):
    calculation = {
        "netuid": 71,
        "epoch_id": 300,
        "block": 108350,
        "research_lab_allocation_doc": {
            "allocation_hash": "sha256:" + "9" * 64,
        },
    }
    request = build_weight_inputs_request_v2(
        validator_hotkey=VALIDATOR_HOTKEY,
        netuid=71,
        epoch_id=300,
        block=108350,
        calculation_snapshot_hash=sha256_json(calculation),
        allocation_hash=calculation["research_lab_allocation_doc"]["allocation_hash"],
        leaderboard_window_start=leaderboard_window_start,
        leaderboard_window_end=leaderboard_window_end,
    )
    return weights_api.WeightInputsV2Authorization(
        request=request,
        calculation_snapshot=calculation,
        validator_hotkey_signature="1" * 128,
    )


def _patch_epoch_authority(monkeypatch):
    async def _official_epoch_authority(**_kwargs):
        return {
            "schema_version": "leadpoet.weight_epoch_authority.v1",
            "epoch_scheme": "bittensor.subnet_epoch_index.v1",
            "settlement_epoch_id": 300,
            "subnet_epoch_index": 235,
            "epoch_block": 350,
        }

    monkeypatch.setattr(
        weights_api,
        "_verify_epoch_block_authority",
        _official_epoch_authority,
    )


def _patch_common(monkeypatch):
    _patch_epoch_authority(monkeypatch)
    monkeypatch.setattr(
        weights_api,
        "_validate_authoritative_v2_submission",
        lambda _submission: (_verified(), _validator_boot()),
    )
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_binding_message", lambda *args, **kwargs: True)


def _compact_submission():
    return weights_api.CompactWeightSubmissionV2(
        schema_version="leadpoet.compact_weight_submission.v2",
        validator_hotkey=VALIDATOR_HOTKEY,
        binding_message="binding",
        validator_hotkey_signature="2" * 128,
        weight_snapshot={},
        weight_result={},
        weights_signature="3" * 128,
        ancestry_commitment="sha256:" + "3" * 64,
        upstream_ancestry_proofs={},
        upstream_transport_attempts=[],
        validator_receipt_delta={},
        binding_receipt={},
        validator_ancestry_proof={},
        epoch_authority=None,
        epoch_boundary=None,
        compact_submission_hash=COMPACT_SUBMISSION_HASH,
    )


def _compact_normalized():
    return {
        "schema_version": "leadpoet.compact_weight_submission.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_result": {
            "netuid": 71,
            "epoch_id": 300,
            "block": 108350,
            "uids": [0],
            "weights_u16": [65535],
            "weights_hash": "5" * 64,
        },
        "validator_ancestry_proof": {"proof": True},
        "compact_submission_hash": COMPACT_SUBMISSION_HASH,
    }


def _compact_verified():
    return {
        "bundle_hash": BUNDLE_HASH,
        "compact_submission_hash": COMPACT_SUBMISSION_HASH,
        "root_receipt_hash": ROOT_RECEIPT,
        "weight_receipt_hash": WEIGHT_RECEIPT,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "netuid": 71,
        "epoch_id": 300,
        "block": 108350,
        "uids": [0],
        "weights_hash": "5" * 64,
    }


def _patch_compact_submission_route(monkeypatch, *, calls, coordinator):
    from gateway.research_lab import attested_coordinator_v2, attested_v2_store
    from gateway.utils import epoch as gateway_epoch

    normalized = _compact_normalized()
    verified = _compact_verified()
    cutover = SimpleNamespace(
        mapping_hash="sha256:" + "4" * 64,
        network_genesis_hash="0x" + "5" * 64,
        netuid=71,
    )

    async def validate_cutover(_cutover):
        calls.append("cutover")

    async def persist_graph(_graph):
        calls.append("graph")
        return {"root_receipt_hash": ROOT_RECEIPT}

    async def persist_checkpoint(*_args, **_kwargs):
        calls.append("checkpoint")
        return {"root_receipt_hash": ROOT_RECEIPT}

    async def persist_submission(_submission):
        calls.append("submission")
        return {
            "bundle_hash": BUNDLE_HASH,
            "binding_receipt_hash": ROOT_RECEIPT,
            "durable_readback_hash": DURABLE_HASH,
        }

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(
        weights_api,
        "validate_compact_weight_submission_shape_v2",
        lambda _payload: normalized,
    )
    monkeypatch.setattr(weights_api, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(
        gateway_epoch,
        "validate_stateful_cutover_authority_async",
        validate_cutover,
    )
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda _cutover: None,
    )
    monkeypatch.setattr(
        weights_api,
        "derive_ancestry_lineage_id_v2",
        lambda **_kwargs: LINEAGE_ID,
    )
    monkeypatch.setattr(
        weights_api,
        "verify_compact_weight_submission_v2",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        weights_api,
        "build_checkpointed_weight_graph_from_compact_v2",
        lambda *_args, **_kwargs: {"root_receipt_hash": ROOT_RECEIPT},
    )
    monkeypatch.setattr(
        weights_api,
        "_verify_authoritative_v2_boot",
        lambda identity: identity,
    )
    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", persist_graph)
    monkeypatch.setattr(
        attested_v2_store,
        "persist_ancestry_checkpoint_v2",
        persist_checkpoint,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_submission_v2",
        persist_submission,
    )
    monkeypatch.setattr(attested_coordinator_v2, "execute_coordinator_v2", coordinator)
    monkeypatch.setattr(
        weights_api,
        "build_compact_published_weight_authority_v2",
        lambda **kwargs: {
            "authority_stage": "published",
            "compact_submission": normalized,
            "publication": kwargs["publication"],
        },
    )
    monkeypatch.setattr(
        weights_api,
        "verify_compact_published_weight_authority_v2",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        weights_api,
        "build_stateful_epoch_evidence_from_compact_v2",
        lambda *_args, **_kwargs: None,
    )
    return normalized, verified


@pytest.mark.asyncio
async def test_compact_publication_intent_allows_exact_retry_after_window(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store
    from gateway.utils import logger as gateway_logger

    calls = []
    intent = None
    coordinator_attempts = 0

    async def coordinator(**kwargs):
        nonlocal coordinator_attempts
        coordinator_attempts += 1
        calls.append("coordinator")
        if coordinator_attempts == 1:
            raise TimeoutError("publication transport interrupted")
        return {
            "result": {
                "schema_version": "leadpoet.weight_publication.v2",
                **kwargs["payload"],
            },
            "receipt": {"receipt_hash": "sha256:" + "6" * 64},
            "ancestry_compact_proof": {"proof_hash": "sha256:" + "7" * 64},
        }

    _patch_compact_submission_route(
        monkeypatch,
        calls=calls,
        coordinator=coordinator,
    )

    async def load_authority(**_kwargs):
        calls.append("authority_load")
        return None

    async def load_intent(**_kwargs):
        calls.append("intent_load")
        return intent

    async def verify_epoch(**_kwargs):
        calls.append("time_gate")
        return {"epoch_id": 300, "submission_window": "open"}

    async def log_event(_event_type, _payload):
        calls.append("transparency")
        return {"event_hash": EVENT_HASH}

    async def persist_intent(**_kwargs):
        nonlocal intent
        calls.append("intent_persist")
        intent = {
            "bundle_hash": BUNDLE_HASH,
            "compact_submission_hash": COMPACT_SUBMISSION_HASH,
            "root_receipt_hash": ROOT_RECEIPT,
            "durable_readback_hash": DURABLE_HASH,
            "transparency_event_hash": EVENT_HASH,
        }
        return intent

    async def persist_authority(_authority):
        calls.append("authority_persist")
        return {"bundle_hash": BUNDLE_HASH}

    monkeypatch.setattr(
        attested_v2_store,
        "load_compact_weight_authority_v2",
        load_authority,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_compact_weight_publication_intent_v2",
        load_intent,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_publication_intent_v2",
        persist_intent,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_authority_v2",
        persist_authority,
    )
    monkeypatch.setattr(weights_api, "_verify_epoch_block_authority", verify_epoch)
    monkeypatch.setattr(gateway_logger, "log_event", log_event)

    with pytest.raises(HTTPException) as first:
        await weights_api.submit_compact_weights_v2(_compact_submission())
    assert first.value.status_code == 503
    assert calls.count("time_gate") == 1
    assert calls.count("intent_persist") == 1

    async def stale_gate(**_kwargs):
        raise AssertionError("durable exact retry must not reapply the time gate")

    monkeypatch.setattr(weights_api, "_verify_epoch_block_authority", stale_gate)
    response = await weights_api.submit_compact_weights_v2(_compact_submission())

    assert response["bundle_hash"] == BUNDLE_HASH
    assert response["compact_submission_hash"] == COMPACT_SUBMISSION_HASH
    assert calls.count("time_gate") == 1
    assert calls.count("transparency") == 1
    assert calls.count("intent_persist") == 1
    assert calls.count("coordinator") == 2
    assert calls[-1] == "authority_persist"


@pytest.mark.asyncio
async def test_compact_publication_uses_concurrent_immutable_intent_winner(
    monkeypatch,
):
    from gateway.research_lab import attested_coordinator_v2, attested_v2_store
    from gateway.utils import logger as gateway_logger

    calls = []
    winner = {
        "bundle_hash": BUNDLE_HASH,
        "compact_submission_hash": COMPACT_SUBMISSION_HASH,
        "root_receipt_hash": ROOT_RECEIPT,
        "durable_readback_hash": DURABLE_HASH,
        "transparency_event_hash": "sha256:" + "9" * 64,
    }
    intent_loads = 0

    async def coordinator(**kwargs):
        calls.append("coordinator")
        assert kwargs["payload"]["transparency_event_hash"] == winner[
            "transparency_event_hash"
        ]
        return {
            "result": {
                "schema_version": "leadpoet.weight_publication.v2",
                **kwargs["payload"],
            },
            "receipt": {"receipt_hash": "sha256:" + "6" * 64},
            "ancestry_compact_proof": {"proof_hash": "sha256:" + "7" * 64},
        }

    _patch_compact_submission_route(
        monkeypatch,
        calls=calls,
        coordinator=coordinator,
    )

    async def load_authority(**_kwargs):
        return None

    async def load_intent(**_kwargs):
        nonlocal intent_loads
        intent_loads += 1
        return None if intent_loads == 1 else winner

    async def reject_loser(**_kwargs):
        calls.append("intent_conflict")
        raise attested_v2_store.AttestedV2StoreError("immutable winner exists")

    async def verify_epoch(**_kwargs):
        calls.append("time_gate")
        return {"epoch_id": 300, "submission_window": "open"}

    async def log_event(_event_type, _payload):
        return {"event_hash": EVENT_HASH}

    async def persist_authority(_authority):
        calls.append("authority_persist")
        return {"bundle_hash": BUNDLE_HASH}

    monkeypatch.setattr(
        attested_v2_store,
        "load_compact_weight_authority_v2",
        load_authority,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_compact_weight_publication_intent_v2",
        load_intent,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_publication_intent_v2",
        reject_loser,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_authority_v2",
        persist_authority,
    )
    monkeypatch.setattr(weights_api, "_verify_epoch_block_authority", verify_epoch)
    monkeypatch.setattr(gateway_logger, "log_event", log_event)
    monkeypatch.setattr(attested_coordinator_v2, "execute_coordinator_v2", coordinator)

    response = await weights_api.submit_compact_weights_v2(_compact_submission())
    assert response["bundle_hash"] == BUNDLE_HASH
    assert intent_loads == 2
    assert calls.count("time_gate") == 1
    assert "intent_conflict" in calls
    assert calls[-1] == "authority_persist"


@pytest.mark.asyncio
async def test_authoritative_v2_persists_and_publishes_before_ack(monkeypatch):
    from gateway.research_lab import attested_coordinator_v2, attested_v2_store
    from gateway.utils import logger as gateway_logger

    _patch_common(monkeypatch)
    calls = []

    async def _load(**_kwargs):
        calls.append("duplicate_check")
        return None

    async def _persist_bundle(_bundle):
        calls.append("bundle")
        return {
            **_verified(),
            "durable_readback_hash": DURABLE_HASH,
        }

    async def _persist_graph(_graph):
        calls.append("graph")
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def _log_event(_event_type, _payload):
        calls.append("transparency")
        return {"event_hash": "7" * 64}

    async def _coordinator(**kwargs):
        calls.append("coordinator")
        assert kwargs["purpose"] == "gateway.weights.publication.v2"
        assert kwargs["boot_verifier"]("identity") == "lineage:identity"
        return {
            "result": {
                "schema_version": "leadpoet.weight_publication.v2",
                **kwargs["payload"],
            },
            "receipt_graph": {"root_receipt_hash": "sha256:" + "6" * 64},
        }

    async def _persist_publication(**_kwargs):
        calls.append("publication")
        return {
            "weight_submission_event_hash": EVENT_HASH,
            "publication_receipt_hash": "sha256:" + "6" * 64,
        }

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", _load)
    monkeypatch.setattr(attested_v2_store, "persist_weight_bundle_v2", _persist_bundle)
    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", _persist_graph)
    monkeypatch.setattr(
        attested_v2_store,
        "persist_weight_publication_v2",
        _persist_publication,
    )
    monkeypatch.setattr(
        weights_api,
        "_build_authoritative_v2_receipt_boot_verifier",
        lambda _graph: lambda identity: "lineage:" + identity,
    )
    monkeypatch.setattr(gateway_logger, "log_event", _log_event)
    monkeypatch.setattr(attested_coordinator_v2, "execute_coordinator_v2", _coordinator)

    response = await weights_api.submit_weights_v2(_submission())

    assert response.success is True
    assert response.weight_submission_event_hash == EVENT_HASH
    assert response.weight_receipt_hash == WEIGHT_RECEIPT
    assert calls == [
        "duplicate_check",
        "bundle",
        "transparency",
        "coordinator",
        "publication",
    ]


@pytest.mark.asyncio
async def test_authoritative_v2_exact_replay_returns_original_ack_without_time_gate(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    _patch_common(monkeypatch)
    submission = _submission()

    async def existing(**_kwargs):
        return submission.model_dump(mode="python")

    async def publication(**_kwargs):
        return {
            "weight_submission_event_hash": EVENT_HASH,
            "bundle_hash": BUNDLE_HASH,
        }

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", existing)
    monkeypatch.setattr(
        attested_v2_store, "load_weight_publication_v2", publication
    )
    monkeypatch.setattr(
        weights_api,
        "get_subtensor",
        lambda: (_ for _ in ()).throw(
            AssertionError("idempotent replay must not apply a new time gate")
        ),
    )

    response = await weights_api.submit_weights_v2(submission)
    assert response.success is True
    assert response.weight_submission_event_hash == EVENT_HASH
    assert "already" in response.message


@pytest.mark.asyncio
async def test_authoritative_v2_rejects_conflicting_epoch_replay(monkeypatch):
    from gateway.research_lab import attested_v2_store

    _patch_common(monkeypatch)

    async def conflicting(**_kwargs):
        return {"different": True}

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", conflicting)
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights_v2(_submission())
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_weight_inputs_v2_authenticates_and_returns_complete_measured_set(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization()
    allocation_graph = {"root_receipt_hash": "sha256:" + "6" * 64}
    expected = {
        "input_receipt_hashes": {"research_lab_allocation": "sha256:" + "7" * 64},
        "gateway_authority_event_hash": "sha256:" + "6" * 64,
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
    }
    calls = []
    _patch_epoch_authority(monkeypatch)

    async def load_graph(**kwargs):
        calls.append(("load", kwargs))
        return allocation_graph

    async def build_inputs(**kwargs):
        calls.append(("build", kwargs))
        return expected

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(
        attested_v2_store, "load_business_artifact_graph_v2", load_graph
    )
    monkeypatch.setattr(
        attested_weight_inputs_v2, "build_gateway_weight_inputs_v2", build_inputs
    )

    request = Request(
        {
            "type": "http",
            "headers": [(b"accept-encoding", b"identity")],
        }
    )
    response = await weights_api.get_weight_inputs_v2(authorization, request)
    response_doc = json.loads(response.body)
    assert response_doc["request_hash"] == authorization.request["request_hash"]
    assert response_doc["input_receipt_hashes"] == expected["input_receipt_hashes"]
    assert calls[0] == (
        "load",
        {
            "artifact_kind": "allocation",
            "artifact_ref": "epoch:300",
            "artifact_hash": "sha256:" + "9" * 64,
        },
    )
    assert calls[1][1]["leaderboard_window_start"] == "2026-07-03T20:00:00Z"
    assert calls[1][1]["leaderboard_window_end"] == "2026-07-10T20:00:00Z"


@pytest.mark.asyncio
async def test_weight_inputs_v2_compact_request_never_falls_back_to_full_graph(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization()
    expected = {
        "input_receipt_hashes": {
            "research_lab_allocation": "sha256:" + "7" * 64
        },
        "gateway_authority_event_hash": "sha256:" + "6" * 64,
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
        "compact_ancestry": None,
    }
    _patch_epoch_authority(monkeypatch)

    async def load_graph(**_kwargs):
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def build_inputs(**_kwargs):
        return expected

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(
        attested_v2_store, "load_business_artifact_graph_v2", load_graph
    )
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "build_gateway_weight_inputs_v2",
        build_inputs,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(
            authorization,
            _request(query_string=b"ancestry=compact-v2"),
        )

    assert exc.value.status_code == 503
    assert "compact V2 weight ancestry" in exc.value.detail


@pytest.mark.asyncio
async def test_weight_inputs_v2_preserves_disabled_leaderboard_commitment(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization(
        leaderboard_window_start=DISABLED_LEADERBOARD_WINDOW_V1,
        leaderboard_window_end=DISABLED_LEADERBOARD_WINDOW_V1,
    )
    expected = {
        "input_receipt_hashes": {
            "research_lab_allocation": "sha256:" + "7" * 64
        },
        "gateway_authority_event_hash": "sha256:" + "6" * 64,
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
    }
    observed = {}
    _patch_epoch_authority(monkeypatch)

    async def load_graph(**_kwargs):
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def build_inputs(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(
        attested_v2_store, "load_business_artifact_graph_v2", load_graph
    )
    monkeypatch.setattr(
        attested_weight_inputs_v2, "build_gateway_weight_inputs_v2", build_inputs
    )

    request = Request(
        {
            "type": "http",
            "headers": [(b"accept-encoding", b"identity")],
        }
    )
    response = await weights_api.get_weight_inputs_v2(authorization, request)

    assert response.status_code == 200
    assert observed["leaderboard_window_start"] == DISABLED_LEADERBOARD_WINDOW_V1
    assert observed["leaderboard_window_end"] == DISABLED_LEADERBOARD_WINDOW_V1


@pytest.mark.asyncio
async def test_weight_inputs_v2_coalesces_retries_and_caches_exact_success(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization()
    expected = {
        "input_receipt_hashes": {
            "research_lab_allocation": "sha256:" + "7" * 64
        },
        "gateway_authority_event_hash": "sha256:" + "6" * 64,
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
    }
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    _patch_epoch_authority(monkeypatch)

    async def load_graph(**_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def build_inputs(**_kwargs):
        return expected

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(attested_v2_store, "load_business_artifact_graph_v2", load_graph)
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "build_gateway_weight_inputs_v2",
        build_inputs,
    )

    first = asyncio.create_task(
        weights_api.get_weight_inputs_v2(authorization, _request())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(
        weights_api.get_weight_inputs_v2(authorization, _request())
    )
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    response = await asyncio.wait_for(second, timeout=1)
    assert json.loads(response.body)["input_receipt_hashes"] == expected[
        "input_receipt_hashes"
    ]

    cached = await weights_api.get_weight_inputs_v2(authorization, _request())
    assert json.loads(cached.body) == json.loads(response.body)
    assert calls == 1


@pytest.mark.asyncio
async def test_weight_inputs_v2_exact_retry_reuses_authorized_work_after_block_drift(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization()
    authority_calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def epoch_authority(**_kwargs):
        nonlocal authority_calls
        authority_calls += 1
        if authority_calls > 1:
            raise HTTPException(status_code=400, detail="block drift is too large")
        return {"settlement_epoch_id": 300}

    async def load_graph(**_kwargs):
        started.set()
        await release.wait()
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def build_inputs(**_kwargs):
        return {
            "input_receipt_hashes": {
                "research_lab_allocation": "sha256:" + "7" * 64
            },
            "gateway_authority_event_hash": "sha256:" + "6" * 64,
            "upstream_receipt_set": {
                "boot_identities": [],
                "receipts": [],
                "transport_attempts": [],
                "host_operations": [],
            },
        }

    monkeypatch.setattr(weights_api, "_verify_epoch_block_authority", epoch_authority)
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(attested_v2_store, "load_business_artifact_graph_v2", load_graph)
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "build_gateway_weight_inputs_v2",
        build_inputs,
    )

    first = asyncio.create_task(
        weights_api.get_weight_inputs_v2(authorization, _request())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    retry = asyncio.create_task(
        weights_api.get_weight_inputs_v2(authorization, _request())
    )
    release.set()
    first_response, retry_response = await asyncio.gather(first, retry)

    assert json.loads(first_response.body) == json.loads(retry_response.body)
    assert authority_calls == 1


@pytest.mark.asyncio
async def test_weight_inputs_v2_failed_shared_load_does_not_poison_retry(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store, attested_weight_inputs_v2

    authorization = _weight_inputs_authorization()
    calls = 0
    _patch_epoch_authority(monkeypatch)

    async def load_graph(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary graph read timeout")
        return {"root_receipt_hash": "sha256:" + "6" * 64}

    async def build_inputs(**_kwargs):
        return {
            "input_receipt_hashes": {
                "research_lab_allocation": "sha256:" + "7" * 64
            },
            "gateway_authority_event_hash": "sha256:" + "6" * 64,
            "upstream_receipt_set": {
                "boot_identities": [],
                "receipts": [],
                "transport_attempts": [],
                "host_operations": [],
            },
        }

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(attested_v2_store, "load_business_artifact_graph_v2", load_graph)
    monkeypatch.setattr(
        attested_weight_inputs_v2,
        "build_gateway_weight_inputs_v2",
        build_inputs,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(authorization, _request())
    assert exc.value.status_code == 503
    await asyncio.sleep(0)

    response = await weights_api.get_weight_inputs_v2(authorization, _request())
    assert json.loads(response.body)["request_hash"] == authorization.request[
        "request_hash"
    ]
    assert calls == 2


@pytest.mark.asyncio
async def test_weight_inputs_v2_returns_conflict_for_stale_signed_snapshot(
    monkeypatch,
):
    from gateway.research_lab.attested_weight_inputs_v2 import (
        WeightInputSnapshotDriftV2Error,
    )

    authorization = _weight_inputs_authorization()
    _patch_epoch_authority(monkeypatch)

    async def stale(**_kwargs):
        raise WeightInputSnapshotDriftV2Error(
            "bans measured input differs from calculation"
        )

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(weights_api, "_build_weight_inputs_v2_singleflight", stale)
    monkeypatch.setattr(weights_api, "capture_failure", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(authorization, _request())

    assert exc.value.status_code == 409
    assert exc.value.detail == "Authorized V2 weight input snapshot is stale"


@pytest.mark.asyncio
async def test_weight_inputs_v2_keeps_receipt_corruption_as_hard_failure(monkeypatch):
    from gateway.research_lab.attested_weight_inputs_v2 import (
        AttestedWeightInputsV2Error,
    )

    authorization = _weight_inputs_authorization()
    _patch_epoch_authority(monkeypatch)

    async def corrupt(**_kwargs):
        raise AttestedWeightInputsV2Error(
            "bans measured input receipt is invalid"
        )

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_LOADS_INFLIGHT", {})
    monkeypatch.setattr(weights_api, "_WEIGHT_INPUT_RESULTS", {})
    monkeypatch.setattr(weights_api, "_build_weight_inputs_v2_singleflight", corrupt)
    monkeypatch.setattr(weights_api, "capture_failure", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(authorization, _request())

    assert exc.value.status_code == 503
    assert exc.value.detail == (
        "Authoritative V2 weight input reconstruction failed closed"
    )


@pytest.mark.asyncio
async def test_weight_inputs_v2_rejects_tampered_snapshot_before_measured_reads(
    monkeypatch,
):
    authorization = _weight_inputs_authorization()
    authorization.calculation_snapshot["block"] += 1
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(authorization, _request())
    assert exc.value.status_code == 400
    assert "bind" in exc.value.detail


@pytest.mark.asyncio
async def test_weight_inputs_v2_fails_closed_on_missing_allocation_lineage(monkeypatch):
    from gateway.research_lab import attested_v2_store

    authorization = _weight_inputs_authorization()
    _patch_epoch_authority(monkeypatch)

    async def missing(**_kwargs):
        raise RuntimeError("lineage missing")

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *args: True)
    monkeypatch.setattr(
        attested_v2_store, "load_business_artifact_graph_v2", missing
    )
    with pytest.raises(HTTPException) as exc:
        await weights_api.get_weight_inputs_v2(authorization, _request())
    assert exc.value.status_code == 503
    assert "failed closed" in exc.value.detail


@pytest.mark.asyncio
async def test_compact_finalization_reports_initial_validation_failure(monkeypatch):
    submission = SimpleNamespace(
        model_dump=lambda **_kwargs: {"invalid": True},
        validator_hotkey=VALIDATOR_HOTKEY,
    )

    def reject(_payload):
        raise ValueError("invalid finalization shape")

    monkeypatch.setattr(
        weights_api,
        "validate_compact_weight_finalization_shape_v2",
        reject,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.finalize_compact_weights_v2(submission)

    assert exc.value.status_code == 503
    assert "invalid finalization shape" in exc.value.detail


@pytest.mark.asyncio
async def test_compact_finalization_exact_replay_is_idempotent_without_writes(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store
    from gateway.utils import epoch as gateway_epoch

    finalization = {
        "netuid": 71,
        "epoch_id": 300,
        "extrinsic_hash": "0x" + "a" * 64,
        "finalized_block": 108355,
    }
    normalized = {
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_submission_event_hash": EVENT_HASH,
        "finalization": finalization,
    }
    submission = SimpleNamespace(
        model_dump=lambda **_kwargs: normalized,
        validator_hotkey=VALIDATOR_HOTKEY,
    )
    authority = {
        "authority_stage": "finalized",
        "finalization": {"compact_submission": normalized},
    }
    verified = {
        "epoch_id": 300,
        "weights_hash": "5" * 64,
        "extrinsic_hash": finalization["extrinsic_hash"],
        "finalized_block": finalization["finalized_block"],
        "weight_submission_event_hash": EVENT_HASH,
        "weight_finalization_event_hash": "sha256:" + "b" * 64,
    }
    cutover = SimpleNamespace(
        mapping_hash="sha256:" + "4" * 64,
        network_genesis_hash="0x" + "5" * 64,
        netuid=71,
    )

    async def load_authority(**_kwargs):
        return authority

    async def no_write(*_args, **_kwargs):
        raise AssertionError("exact finalized replay must not append evidence")

    monkeypatch.setattr(
        weights_api,
        "validate_compact_weight_finalization_shape_v2",
        lambda _payload: normalized,
    )
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {VALIDATOR_HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(
        gateway_epoch,
        "validate_stateful_cutover_authority_async",
        lambda _cutover: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda _cutover: None,
    )
    monkeypatch.setattr(
        weights_api,
        "derive_ancestry_lineage_id_v2",
        lambda **_kwargs: LINEAGE_ID,
    )
    monkeypatch.setattr(
        weights_api,
        "verify_compact_published_weight_authority_v2",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_compact_weight_authority_for_identity_v2",
        load_authority,
    )
    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", no_write)
    monkeypatch.setattr(
        attested_v2_store,
        "persist_ancestry_checkpoint_v2",
        no_write,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_compact_weight_authority_v2",
        no_write,
    )

    response = await weights_api.finalize_compact_weights_v2(submission)
    assert response.success is True
    assert "already" in response.message
    assert response.weight_finalization_event_hash == verified[
        "weight_finalization_event_hash"
    ]


@pytest.mark.asyncio
async def test_authoritative_v2_replay_is_rejected_before_any_write(monkeypatch):
    from gateway.research_lab import attested_v2_store

    _patch_common(monkeypatch)

    async def _existing(**_kwargs):
        return {"schema_version": PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION}

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", _existing)
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights_v2(_submission())
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_authoritative_v2_publication_failure_is_fail_closed(monkeypatch):
    from gateway.research_lab import attested_coordinator_v2, attested_v2_store
    from gateway.utils import logger as gateway_logger

    _patch_common(monkeypatch)
    monkeypatch.setattr(
        attested_v2_store,
        "load_weight_bundle_v2",
        lambda **_kwargs: None,
    )

    async def _load(**_kwargs):
        return None

    async def _persist_bundle(_bundle):
        return {**_verified(), "durable_readback_hash": DURABLE_HASH}

    async def _log_event(_event_type, _payload):
        return {"event_hash": "7" * 64}

    async def _failed_coordinator(**_kwargs):
        raise RuntimeError("coordinator unavailable")

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", _load)
    monkeypatch.setattr(attested_v2_store, "persist_weight_bundle_v2", _persist_bundle)
    monkeypatch.setattr(gateway_logger, "log_event", _log_event)
    monkeypatch.setattr(
        attested_coordinator_v2,
        "execute_coordinator_v2",
        _failed_coordinator,
    )
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights_v2(_submission())
    assert exc.value.status_code == 503
    assert "failed closed" in exc.value.detail


@pytest.mark.asyncio
async def test_v1_submission_is_gone(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_TEE_PROTOCOL", raising=False)
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights(None)
    assert exc.value.status_code == 410


@pytest.mark.asyncio
async def test_v2_latest_returns_only_finalized_authority(monkeypatch):
    from gateway.research_lab import attested_v2_store

    expected = {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": _submission().model_dump(mode="python"),
        "publication": {"published": True},
        "finalization": {"finalized": True},
    }

    async def _load(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "epoch_id": 300,
            "validator_hotkey": "validator-hotkey",
        }
        return expected

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"})
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)
    response = await weights_api.get_attested_weights_v2(
        71,
        300,
        _request(),
    )
    assert json.loads(response.body) == expected
    assert response.headers["vary"] == "Accept-Encoding"


@pytest.mark.asyncio
async def test_v2_published_returns_not_found_while_publication_is_pending(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    async def _load(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "epoch_id": 300,
            "validator_hotkey": "validator-hotkey",
            "require_finalization": False,
        }
        return None

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_published_weights_v2(
            71,
            300,
            _request(),
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == "published v2 weight authority not found"


@pytest.mark.asyncio
async def test_v2_published_coalesces_only_overlapping_identical_loads(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def _load(**kwargs):
        nonlocal calls
        assert kwargs == {
            "netuid": 71,
            "epoch_id": 300,
            "validator_hotkey": "validator-hotkey",
            "require_finalization": False,
        }
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            return {
                "authority_stage": "published",
                "epoch_id": 300,
            }
        return {
            "authority_stage": "finalized",
            "epoch_id": 300,
        }

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(
        weights_api,
        "_WEIGHT_AUTHORITY_LOADS_INFLIGHT",
        {},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    requests = [
        asyncio.create_task(
            weights_api.get_published_weights_v2(71, 300, _request())
        )
        for _ in range(3)
    ]
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0)
    overlapping_calls = calls
    release.set()
    responses = await asyncio.wait_for(
        asyncio.gather(*requests),
        timeout=1,
    )
    assert overlapping_calls == 1
    assert {
        json.loads(response.body)["authority_stage"]
        for response in responses
    } == {"published"}

    await asyncio.sleep(0)
    refreshed = await weights_api.get_published_weights_v2(
        71,
        300,
        _request(),
    )
    assert calls == 2
    assert json.loads(refreshed.body)["authority_stage"] == "finalized"


@pytest.mark.asyncio
async def test_v2_published_singleflight_is_scoped_by_epoch(monkeypatch):
    from gateway.research_lab import attested_v2_store

    calls = []
    both_started = asyncio.Event()

    async def _load(**kwargs):
        calls.append(kwargs["epoch_id"])
        if len(calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return {
            "authority_stage": "published",
            "epoch_id": kwargs["epoch_id"],
        }

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(
        weights_api,
        "_WEIGHT_AUTHORITY_LOADS_INFLIGHT",
        {},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    responses = await asyncio.wait_for(
        asyncio.gather(
            weights_api.get_published_weights_v2(71, 300, _request()),
            weights_api.get_published_weights_v2(71, 301, _request()),
        ),
        timeout=2,
    )

    assert sorted(calls) == [300, 301]
    assert [json.loads(response.body)["epoch_id"] for response in responses] == [
        300,
        301,
    ]


@pytest.mark.asyncio
async def test_v2_published_waiter_cancellation_keeps_shared_load_alive(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    loader_cancelled = False

    async def _load(**_kwargs):
        nonlocal calls, loader_cancelled
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            loader_cancelled = True
            raise
        return {"authority_stage": "published", "epoch_id": 300}

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(
        weights_api,
        "_WEIGHT_AUTHORITY_LOADS_INFLIGHT",
        {},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    first = asyncio.create_task(
        weights_api.get_published_weights_v2(71, 300, _request())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(
        weights_api.get_published_weights_v2(71, 300, _request())
    )
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert loader_cancelled is False

    release.set()
    response = await asyncio.wait_for(second, timeout=1)
    assert json.loads(response.body)["epoch_id"] == 300
    assert calls == 1
    await asyncio.sleep(0)
    assert weights_api._WEIGHT_AUTHORITY_LOADS_INFLIGHT == {}


@pytest.mark.asyncio
async def test_v2_published_failed_load_does_not_poison_retry(monkeypatch):
    from gateway.research_lab import attested_v2_store

    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def _load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise TimeoutError("temporary Supabase read timeout")
        return {"authority_stage": "published", "epoch_id": 300}

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(
        weights_api,
        "_WEIGHT_AUTHORITY_LOADS_INFLIGHT",
        {},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    requests = [
        asyncio.create_task(
            weights_api.get_published_weights_v2(71, 300, _request())
        )
        for _ in range(2)
    ]
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0)
    overlapping_calls = calls
    release.set()
    failures = await asyncio.wait_for(
        asyncio.gather(*requests, return_exceptions=True),
        timeout=1,
    )
    assert overlapping_calls == 1
    assert all(
        isinstance(exc, HTTPException)
        and exc.status_code == 503
        and exc.headers == {"Retry-After": "5"}
        for exc in failures
    )

    await asyncio.sleep(0)
    response = await weights_api.get_published_weights_v2(
        71,
        300,
        _request(),
    )
    assert calls == 2
    assert json.loads(response.body)["epoch_id"] == 300


@pytest.mark.asyncio
async def test_v2_published_pending_result_does_not_hide_later_publication(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    calls = 0

    async def _load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return {"authority_stage": "published", "epoch_id": 300}

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(
        weights_api,
        "_WEIGHT_AUTHORITY_LOADS_INFLIGHT",
        {},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_published_weights_v2(71, 300, _request())
    assert exc.value.status_code == 404

    await asyncio.sleep(0)
    response = await weights_api.get_published_weights_v2(
        71,
        300,
        _request(),
    )
    assert calls == 2
    assert json.loads(response.body)["epoch_id"] == 300


@pytest.mark.asyncio
async def test_v2_published_returns_retryable_unavailable_for_store_timeout(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    async def _load(**_kwargs):
        raise TimeoutError("temporary Supabase read timeout")

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_published_weights_v2(71, 300, _request())

    assert exc.value.status_code == 503
    assert exc.value.headers == {"Retry-After": "5"}


@pytest.mark.asyncio
async def test_v2_published_keeps_integrity_failures_as_internal_errors(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    async def _load(**_kwargs):
        raise RuntimeError("stored authority receipt differs")

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_published_weights_v2(71, 300, _request())

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_v2_latest_returns_retryable_unavailable_for_store_timeout(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    async def _load(**_kwargs):
        raise TimeoutError("temporary Supabase read timeout")

    monkeypatch.setattr(
        weights_api,
        "PRIMARY_VALIDATOR_HOTKEYS",
        {"validator-hotkey"},
    )
    monkeypatch.setattr(attested_v2_store, "load_weight_authority_v2", _load)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_attested_weights_v2(71, 300, _request())

    assert exc.value.status_code == 503
    assert exc.value.headers == {"Retry-After": "5"}


def test_v2_authority_response_gzips_without_changing_json():
    authority = {
        "schema_version": "leadpoet.published_weight_authority_stage.v2",
        "authority_stage": "published",
        "bundle": {"receipt_graph": {"receipts": ["a" * 4096]}},
    }

    response = weights_api._weight_authority_http_response(
        authority,
        accept_encoding="gzip, deflate",
    )

    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["vary"] == "Accept-Encoding"
    assert json.loads(gzip.decompress(response.body)) == authority


def test_v2_authority_response_honors_gzip_q_zero():
    authority = {"payload": "a" * 4096}

    response = weights_api._weight_authority_http_response(
        authority,
        accept_encoding="gzip;q=0, deflate",
    )

    assert "content-encoding" not in response.headers
    assert json.loads(response.body) == authority


@pytest.mark.asyncio
async def test_v2_finalization_is_acknowledged_only_after_durable_append(monkeypatch):
    from gateway.research_lab import attested_v2_store
    from leadpoet_canonical import attested_v2

    event_hash = "sha256:" + "1" * 64
    final_event_hash = "sha256:" + "2" * 64
    payload = weights_api.WeightFinalizationV2(
        schema_version="leadpoet.weight_finalization_submission.v2",
        validator_hotkey="validator-hotkey",
        weight_submission_event_hash=event_hash,
        finalization={"committed": True},
        receipt_graph={"root_receipt_hash": "sha256:" + "3" * 64},
    )
    verified = {
        "validator_hotkey": "validator-hotkey",
        "netuid": 71,
        "epoch_id": 300,
        "weights_hash": "4" * 64,
        "extrinsic_hash": "0x" + "5" * 64,
        "finalized_block": 108345,
        "weight_submission_event_hash": event_hash,
    }
    monkeypatch.setattr(
        weights_api,
        "validate_weight_finalization_submission_v2",
        lambda _value: dict(verified),
    )
    monkeypatch.setattr(attested_v2, "validate_receipt_graph", lambda *_a, **_k: [])
    monkeypatch.setattr(
        weights_api,
        "_build_authoritative_v2_receipt_boot_verifier",
        lambda _graph: lambda identity: identity,
    )
    monkeypatch.setattr(
        weights_api,
        "_receipt_root_boot_identity",
        lambda _graph: _validator_boot(),
    )
    monkeypatch.setattr(
        weights_api,
        "_verify_authoritative_v2_boot",
        lambda identity: identity,
    )
    monkeypatch.setattr(
        weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"}
    )
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    calls = []

    async def persist(**kwargs):
        calls.append(kwargs)
        return {"weight_finalization_event_hash": final_event_hash}

    monkeypatch.setattr(attested_v2_store, "persist_weight_finalization_v2", persist)
    response = await weights_api.finalize_weights_v2(payload)
    assert calls and calls[0]["submission"] == payload.model_dump(mode="python")
    assert response.success is True
    assert response.weight_finalization_event_hash == final_event_hash
