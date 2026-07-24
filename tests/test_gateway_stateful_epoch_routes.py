from __future__ import annotations

import pytest
from fastapi import HTTPException

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochSnapshot,
)
from gateway.api import weights as weights_api
from leadpoet_canonical.attested_v2 import sha256_json


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
SIGNATURE = "0x" + "1" * 128
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=1_000,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )


def _snapshot(*, block: int, block_hash: str) -> SubnetEpochSnapshot:
    return SubnetEpochSnapshot(
        network_genesis_hash=_cutover().network_genesis_hash,
        netuid=71,
        head_kind="finalized",
        block_hash=block_hash,
        current_block=block,
        last_epoch_block=1_000,
        pending_epoch_at=0,
        subnet_epoch_index=10,
        tempo=360,
        blocks_since_last_step=block - 1_000,
        observed_at="2026-07-16T12:00:00Z",
    )


def test_gateway_boot_verifier_maps_receipt_manifest_to_release_execution_manifest(
    monkeypatch,
):
    identity = {
        "physical_role": "gateway_scoring",
        "commit_sha": "d" * 40,
        "pcr0": "e" * 96,
        "build_manifest_hash": HASH_A,
        "dependency_lock_hash": HASH_B,
    }
    monkeypatch.setattr(
        weights_api,
        "_gateway_v2_release_manifest",
        lambda: {
            "commit_sha": identity["commit_sha"],
            "roles": {
                "gateway_scoring": {
                    "commit_sha": identity["commit_sha"],
                    "pcr0": identity["pcr0"],
                    "execution_manifest_hash": HASH_A,
                    "dependency_lock_hash": HASH_B,
                }
            }
        },
    )
    monkeypatch.setattr(
        weights_api,
        "verify_boot_identity_nitro",
        lambda observed, **kwargs: {
            "identity": observed,
            "expected_pcr0": kwargs["expected_pcr0"],
        },
    )
    result = weights_api._verify_authoritative_v2_boot(identity)
    assert result["identity"] == identity
    assert result["expected_pcr0"] == identity["pcr0"]


@pytest.mark.asyncio
async def test_candidate_route_verifies_hotkey_dynamic_boot_and_chain_before_ack(
    monkeypatch,
):
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from leadpoet_canonical import attested_v2

    cutover = _cutover()
    boundary = _snapshot(block=1_000, block_hash=cutover.cutover_block_hash)
    graph = {"root_receipt_hash": HASH_A}
    capture = {
        "schema_version": authority.CAPTURE_SCHEMA_VERSION,
        "epoch_authority": {},
        "epoch_boundary": {},
        "epoch_authority_receipt_hash": HASH_B,
        "epoch_boundary_receipt_hash": HASH_B,
        "receipt_graph": graph,
        "boot_identity": {},
        "source_artifacts": [],
    }
    request = weights_api.SubnetEpochCandidateSubmissionV1(
        schema_version=authority.CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        validator_hotkey=HOTKEY,
        validator_hotkey_signature=SIGNATURE,
        cutover_manifest=cutover.to_dict(),
        capture=capture,
    )
    calls = []

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: object())
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda observed: calls.append(("anchor", observed.mapping_hash)),
    )

    def verify_signature(message, signature, hotkey):
        calls.append(("signature", message, signature, hotkey))
        return True

    monkeypatch.setattr(weights_api, "verify_wallet_signature", verify_signature)

    def validate_graph(value, **kwargs):
        calls.append(("graph", value, kwargs))

    monkeypatch.setattr(attested_v2, "validate_receipt_graph", validate_graph)
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_args, **_kwargs: {
            "boundary": boundary,
            "boundary_hash": HASH_C,
            "boundary_receipt": {"receipt_hash": HASH_B},
        },
    )

    async def persist(*_args, **kwargs):
        calls.append(("persist", kwargs))
        return {"snapshot_hash": HASH_C, "created_at": "now"}

    monkeypatch.setattr(authority, "persist_pre_cutover_candidate_v1", persist)

    response = await weights_api.stage_subnet_epoch_candidate_v1(request)

    payload_hash = sha256_json(
        {
            "schema_version": request.schema_version,
            "cutover_manifest": cutover.to_dict(),
            "capture": capture,
        }
    )
    assert response["candidate_hash"] == payload_hash
    assert response["mapping_hash"] == cutover.mapping_hash
    assert response["boundary_block"] == cutover.cutover_block
    graph_call = next(item for item in calls if item[0] == "graph")
    assert graph_call[2]["boot_attestation_verifier"] is (
        weights_api._verify_authoritative_v2_boot
    )
    assert graph_call[2]["require_boot_attestation_verification"] is True
    assert [item[0] for item in calls].index("signature") < [
        item[0] for item in calls
    ].index("persist")
    assert [item[0] for item in calls].index("graph") < [
        item[0] for item in calls
    ].index("persist")
    assert [item[0] for item in calls].index("anchor") < [
        item[0] for item in calls
    ].index("persist")


@pytest.mark.asyncio
async def test_candidate_route_rejects_unauthorized_hotkey_before_attestation(
    monkeypatch,
):
    from gateway.research_lab import stateful_epoch_authority_v1 as authority

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    request = weights_api.SubnetEpochCandidateSubmissionV1(
        schema_version=authority.CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        validator_hotkey="5" + "A" * 47,
        validator_hotkey_signature=SIGNATURE,
        cutover_manifest=_cutover().to_dict(),
        capture={},
    )
    with pytest.raises(HTTPException) as exc:
        await weights_api.stage_subnet_epoch_candidate_v1(request)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_candidate_route_reports_archive_outage_without_persisting(
    monkeypatch,
):
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from leadpoet_canonical import attested_v2

    cutover = _cutover()
    boundary = _snapshot(block=1_000, block_hash=cutover.cutover_block_hash)
    capture = {
        "schema_version": authority.CAPTURE_SCHEMA_VERSION,
        "epoch_authority": {},
        "epoch_boundary": {},
        "epoch_authority_receipt_hash": HASH_B,
        "epoch_boundary_receipt_hash": HASH_B,
        "receipt_graph": {"root_receipt_hash": HASH_A},
        "boot_identity": {},
        "source_artifacts": [],
    }
    request = weights_api.SubnetEpochCandidateSubmissionV1(
        schema_version=authority.CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        validator_hotkey=HOTKEY,
        validator_hotkey_signature=SIGNATURE,
        cutover_manifest=cutover.to_dict(),
        capture=capture,
    )

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *_a: True)
    monkeypatch.setattr(attested_v2, "validate_receipt_graph", lambda *_a, **_k: None)
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_a, **_k: {
            "boundary": boundary,
            "boundary_hash": HASH_C,
            "boundary_receipt": {"receipt_hash": HASH_B},
        },
    )

    def unavailable_archive(*_args, **_kwargs):
        raise RuntimeError("archive transport unavailable")

    async def forbidden_persist(*_args, **_kwargs):
        raise AssertionError("archive outage must fail before persistence")

    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        unavailable_archive,
    )
    monkeypatch.setattr(
        authority,
        "persist_pre_cutover_candidate_v1",
        forbidden_persist,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.stage_subnet_epoch_candidate_v1(request)
    assert exc.value.status_code == 503
    assert exc.value.detail == "Official archive cutover authority is unavailable"


def _stored_bundle(graph):
    return {
        "schema_version": "leadpoet.published_weight_bundle.v2",
        "validator_hotkey": HOTKEY,
        "binding_message": "binding",
        "validator_hotkey_signature": "2" * 128,
        "weight_snapshot": {},
        "weight_result": {},
        "weights_signature": "3" * 128,
        "receipt_graph": graph,
    }


def _stub_normal_epoch_boot_verification(monkeypatch):
    """Keep route tests focused after immutable release-lineage verification."""

    def verify_boot(identity):
        return {"identity": identity}

    monkeypatch.setattr(
        weights_api,
        "_verify_authoritative_v2_boot",
        verify_boot,
    )
    monkeypatch.setattr(
        weights_api,
        "_build_authoritative_v2_receipt_boot_verifier",
        lambda _graph: verify_boot,
    )
    monkeypatch.setattr(
        weights_api,
        "_receipt_root_boot_identity",
        lambda _graph: {},
    )
    return verify_boot


@pytest.mark.asyncio
async def test_normal_epoch_route_binds_published_bundle_and_acks_both_snapshots(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from gateway.utils import epoch as gateway_epoch
    from leadpoet_canonical import attested_v2

    cutover = _cutover()
    boundary = _snapshot(block=1_000, block_hash=cutover.cutover_block_hash)
    current = _snapshot(block=1_310, block_hash="0x" + "3" * 64)
    graph = {"root_receipt_hash": HASH_A}
    request = weights_api.SubnetEpochEvidenceSubmissionV1(
        schema_version="leadpoet.validator_subnet_epoch_evidence.v1",
        validator_hotkey=HOTKEY,
        bundle_hash=HASH_C,
        cutover_mapping_hash=cutover.mapping_hash,
        epoch_authority={},
        epoch_authority_hash=HASH_A,
        epoch_authority_receipt_hash=HASH_A,
        epoch_boundary={},
        epoch_boundary_hash=HASH_B,
        epoch_boundary_receipt_hash=HASH_B,
        receipt_graph=graph,
    )
    calls = []

    boot_verifier = _stub_normal_epoch_boot_verification(monkeypatch)
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: object())
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda *_args: calls.append("anchor"),
    )

    async def validate_cutover(observed):
        assert observed == cutover
        calls.append("cutover")

    monkeypatch.setattr(
        gateway_epoch,
        "validate_stateful_cutover_authority_async",
        validate_cutover,
    )

    def validate_graph(_value, **kwargs):
        assert kwargs["boot_attestation_verifier"] is boot_verifier
        assert kwargs["require_boot_attestation_verification"] is True
        calls.append("graph")

    monkeypatch.setattr(attested_v2, "validate_receipt_graph", validate_graph)
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_args, **_kwargs: {
            "current": current,
            "current_hash": HASH_A,
            "current_receipt": {"receipt_hash": HASH_A},
            "boundary": boundary,
            "boundary_hash": HASH_B,
            "boundary_receipt": {"receipt_hash": HASH_B},
        },
    )

    stored = _stored_bundle(graph)

    async def load_bundle(**kwargs):
        assert kwargs == {
            "netuid": 71,
            "epoch_id": 101,
            "validator_hotkey": HOTKEY,
        }
        calls.append("bundle")
        return stored

    async def load_publication(**kwargs):
        assert kwargs == {"bundle_hash": HASH_C}
        calls.append("publication")
        return {"bundle_hash": HASH_C}

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", load_bundle)
    monkeypatch.setattr(
        attested_v2_store,
        "load_weight_publication_v2",
        load_publication,
    )
    monkeypatch.setattr(
        weights_api,
        "_validate_authoritative_v2_submission",
        lambda _bundle: (
            {
                "bundle_hash": HASH_C,
                "validator_hotkey": HOTKEY,
                "netuid": 71,
                "epoch_id": 101,
                "block": 1_310,
            },
            {},
        ),
    )

    async def persist(*_args, **_kwargs):
        calls.append("persist")
        return {
            "receipt_graph_hash": HASH_A,
            "durable_readback_hash": HASH_C,
        }

    monkeypatch.setattr(authority, "persist_post_cutover_evidence_v1", persist)

    response = await weights_api.persist_subnet_epoch_evidence_v1(request)

    assert response["epoch_authority_hash"] == HASH_A
    assert response["epoch_authority_receipt_hash"] == HASH_A
    assert response["boundary_hash"] == HASH_B
    assert response["boundary_receipt_hash"] == HASH_B
    assert response["durable_readback_hash"] == HASH_C
    assert calls.index("bundle") < calls.index("publication") < calls.index("persist")


@pytest.mark.asyncio
async def test_normal_epoch_route_fails_closed_without_publication(monkeypatch):
    from gateway.research_lab import attested_v2_store
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from gateway.utils import epoch as gateway_epoch
    from leadpoet_canonical import attested_v2

    cutover = _cutover()
    current = _snapshot(block=1_310, block_hash="0x" + "3" * 64)
    boundary = _snapshot(block=1_000, block_hash=cutover.cutover_block_hash)
    graph = {"root_receipt_hash": HASH_A}
    request = weights_api.SubnetEpochEvidenceSubmissionV1(
        schema_version="leadpoet.validator_subnet_epoch_evidence.v1",
        validator_hotkey=HOTKEY,
        bundle_hash=HASH_C,
        cutover_mapping_hash=cutover.mapping_hash,
        epoch_authority={},
        epoch_authority_hash=HASH_A,
        epoch_authority_receipt_hash=HASH_A,
        epoch_boundary={},
        epoch_boundary_hash=HASH_B,
        epoch_boundary_receipt_hash=HASH_B,
        receipt_graph=graph,
    )

    _stub_normal_epoch_boot_verification(monkeypatch)
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: object())
    monkeypatch.setattr(weights_api, "validate_cutover_anchor_from_archive", lambda *_a: None)
    monkeypatch.setattr(
        gateway_epoch,
        "validate_stateful_cutover_authority_async",
        lambda *_a: _async_none(),
    )
    monkeypatch.setattr(attested_v2, "validate_receipt_graph", lambda *_a, **_k: None)
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_a, **_k: {
            "current": current,
            "current_hash": HASH_A,
            "current_receipt": {"receipt_hash": HASH_A},
            "boundary": boundary,
            "boundary_hash": HASH_B,
            "boundary_receipt": {"receipt_hash": HASH_B},
        },
    )
    monkeypatch.setattr(
        weights_api,
        "_validate_authoritative_v2_submission",
        lambda _bundle: (
            {
                "bundle_hash": HASH_C,
                "validator_hotkey": HOTKEY,
                "netuid": 71,
                "epoch_id": 101,
                "block": 1_310,
            },
            {},
        ),
    )

    async def load_bundle(**_kwargs):
        return _stored_bundle(graph)

    async def missing_publication(**_kwargs):
        return None

    async def forbidden_persist(*_args, **_kwargs):
        raise AssertionError("missing publication must not persist epoch evidence")

    monkeypatch.setattr(attested_v2_store, "load_weight_bundle_v2", load_bundle)
    monkeypatch.setattr(
        attested_v2_store,
        "load_weight_publication_v2",
        missing_publication,
    )
    monkeypatch.setattr(
        authority,
        "persist_post_cutover_evidence_v1",
        forbidden_persist,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.persist_subnet_epoch_evidence_v1(request)
    assert exc.value.status_code == 409
    assert "publication is not durable yet" in exc.value.detail


@pytest.mark.asyncio
async def test_normal_epoch_route_reports_bundle_store_outage_as_unavailable(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from gateway.utils import epoch as gateway_epoch
    from leadpoet_canonical import attested_v2

    cutover = _cutover()
    current = _snapshot(block=1_310, block_hash="0x" + "3" * 64)
    boundary = _snapshot(block=1_000, block_hash=cutover.cutover_block_hash)
    graph = {"root_receipt_hash": HASH_A}
    request = weights_api.SubnetEpochEvidenceSubmissionV1(
        schema_version="leadpoet.validator_subnet_epoch_evidence.v1",
        validator_hotkey=HOTKEY,
        bundle_hash=HASH_C,
        cutover_mapping_hash=cutover.mapping_hash,
        epoch_authority={},
        epoch_authority_hash=HASH_A,
        epoch_authority_receipt_hash=HASH_A,
        epoch_boundary={},
        epoch_boundary_hash=HASH_B,
        epoch_boundary_receipt_hash=HASH_B,
        receipt_graph=graph,
    )

    _stub_normal_epoch_boot_verification(monkeypatch)
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(weights_api, "validate_cutover_anchor_from_archive", lambda *_a: None)
    monkeypatch.setattr(
        gateway_epoch,
        "validate_stateful_cutover_authority_async",
        lambda *_a: _async_none(),
    )
    monkeypatch.setattr(attested_v2, "validate_receipt_graph", lambda *_a, **_k: None)
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_a, **_k: {
            "current": current,
            "current_hash": HASH_A,
            "current_receipt": {"receipt_hash": HASH_A},
            "boundary": boundary,
            "boundary_hash": HASH_B,
            "boundary_receipt": {"receipt_hash": HASH_B},
        },
    )

    async def unavailable_bundle_store(**_kwargs):
        raise RuntimeError("database transport unavailable")

    async def forbidden_publication(*_args, **_kwargs):
        raise AssertionError("bundle outage must fail before publication lookup")

    async def forbidden_persist(*_args, **_kwargs):
        raise AssertionError("bundle outage must fail before evidence persistence")

    monkeypatch.setattr(
        attested_v2_store,
        "load_weight_bundle_v2",
        unavailable_bundle_store,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_weight_publication_v2",
        forbidden_publication,
    )
    monkeypatch.setattr(
        authority,
        "persist_post_cutover_evidence_v1",
        forbidden_persist,
    )

    with pytest.raises(HTTPException) as exc:
        await weights_api.persist_subnet_epoch_evidence_v1(request)
    assert exc.value.status_code == 503
    assert exc.value.detail == "Authoritative V2 weight bundle store is unavailable"


async def _async_none():
    return None
