from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochSnapshot,
)
from gateway.api import weights as weights_api
from gateway.research_lab import stateful_epoch_candidate_ingest_cli_v1 as cli
from leadpoet_canonical.attested_v2 import (
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
SIGNATURE = "0x" + "1" * 128
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
COMMIT = "d" * 40
PCR0 = "e" * 96
NOW = "2026-07-16T12:00:00Z"


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


def _candidate_document() -> dict:
    from gateway.research_lab.stateful_epoch_authority_v1 import (
        CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        CAPTURE_SCHEMA_VERSION,
    )

    return {
        "schema_version": CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "validator_hotkey_signature": SIGNATURE,
        "cutover_manifest": _cutover().to_dict(),
        "capture": {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "epoch_authority": {},
            "epoch_boundary": {},
            "epoch_authority_receipt_hash": HASH_B,
            "epoch_boundary_receipt_hash": HASH_B,
            "receipt_graph": {"root_receipt_hash": HASH_A},
            "boot_identity": {},
            "source_artifacts": [],
        },
    }


def _validator_release_manifest(*, commit: str = COMMIT, pcr0: str = PCR0) -> dict:
    release = build_validator_release(
        commit_sha=commit,
        pcr0=pcr0,
        app_manifest_hash=HASH_A,
        dependency_lock_hash=HASH_B,
        normalized_image_hash=HASH_C,
        eif_hash="sha256:" + "d" * 64,
        dockerfile_hash="sha256:" + "e" * 64,
        base_dockerfile_hash="sha256:" + "f" * 64,
    )
    return build_validator_release_manifest(
        [
            build_validator_build_evidence(
                release,
                builder_domain=domain,
                builder_id=domain + "-parent",
                build_ordinal=ordinal,
            )
            for domain in ("gateway", "validator")
            for ordinal in (1, 2, 3)
        ]
    )


def _real_candidate_document() -> dict:
    from gateway.research_lab.stateful_epoch_authority_v1 import (
        CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        CAPTURE_SCHEMA_VERSION,
    )
    from gateway.tee.coordinator_epoch_cutover_v2 import SNAPSHOT_PURPOSE

    cutover = _cutover()
    snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash=cutover.cutover_block_hash,
        current_block=cutover.cutover_block,
        last_epoch_block=cutover.cutover_block,
        pending_epoch_at=0,
        subnet_epoch_index=cutover.first_subnet_epoch_index,
        tempo=360,
        blocks_since_last_step=0,
        observed_at=NOW,
    ).to_dict(cutover=cutover)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="validator_weights",
            physical_role="validator_weights",
            commit_sha=COMMIT,
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="4" * 32,
            signing_pubkey=public_key,
            transport_pubkey="5" * 64,
            transport_certificate_hash=HASH_A,
            attestation_user_data_hash=HASH_B,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(
            b"synthetic-test-attestation"
        ).decode("ascii"),
    )
    snapshot_hash = sha256_json(snapshot)
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="validator_weights",
            purpose=SNAPSHOT_PURPOSE,
            job_id="subnet-epoch-boundary:101",
            epoch_id=cutover.first_settlement_epoch_id,
            sequence=0,
            commit_sha=COMMIT,
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=HASH_A,
            output_root=snapshot_hash,
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=merkle_root(
                [snapshot_hash],
                domain="leadpoet-artifact-v2",
            ),
            parent_receipt_hashes=[],
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[receipt],
        transport_attempts=[],
        host_operations=[],
    )
    capture = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "epoch_authority": snapshot,
        "epoch_boundary": snapshot,
        "epoch_authority_receipt_hash": receipt["receipt_hash"],
        "epoch_boundary_receipt_hash": receipt["receipt_hash"],
        "receipt_graph": graph,
        "boot_identity": boot,
        "source_artifacts": [],
    }
    return {
        "schema_version": CANDIDATE_SUBMISSION_SCHEMA_VERSION,
        "validator_hotkey": HOTKEY,
        "validator_hotkey_signature": SIGNATURE,
        "cutover_manifest": cutover.to_dict(),
        "capture": capture,
    }


def _ack(document: dict, *, durable_hash: str = HASH_A) -> dict:
    cutover = _cutover()
    candidate_hash = cli.candidate_payload_hash_v1(document)
    return {
        "schema_version": "leadpoet.subnet_epoch_boundary_candidate_ack.v1",
        "candidate_hash": candidate_hash,
        "validator_hotkey": HOTKEY,
        "candidate_authorization_hash": sha256_json(
            {
                "validator_hotkey": HOTKEY,
                "candidate_payload_hash": candidate_hash,
                "validator_hotkey_signature": SIGNATURE,
            }
        ),
        "mapping_hash": cutover.mapping_hash,
        "subnet_epoch_index": cutover.first_subnet_epoch_index,
        "settlement_epoch_id": cutover.first_settlement_epoch_id,
        "boundary_block": cutover.cutover_block,
        "boundary_hash": sha256_json(document["capture"]["epoch_boundary"]),
        "boundary_receipt_hash": document["capture"][
            "epoch_boundary_receipt_hash"
        ],
        "receipt_graph_hash": sha256_json(document["capture"]["receipt_graph"]),
        "durable_readback_hash": durable_hash,
    }


@pytest.mark.asyncio
async def test_dry_run_reuses_route_checks_with_zero_persistence_calls(monkeypatch):
    from gateway.research_lab import stateful_epoch_authority_v1 as authority
    from leadpoet_canonical import attested_v2

    document = _candidate_document()
    cutover = _cutover()
    boundary = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash=cutover.cutover_block_hash,
        current_block=cutover.cutover_block,
        last_epoch_block=cutover.cutover_block,
        pending_epoch_at=0,
        subnet_epoch_index=cutover.first_subnet_epoch_index,
        tempo=360,
        blocks_since_last_step=0,
        observed_at="2026-07-16T12:00:00Z",
    )
    calls: list[str] = []

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: object())
    monkeypatch.setattr(
        weights_api,
        "verify_wallet_signature",
        lambda *_args: calls.append("signature") or True,
    )
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda *_args: calls.append("anchor"),
    )
    monkeypatch.setattr(
        attested_v2,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: calls.append("graph"),
    )
    monkeypatch.setattr(
        authority,
        "validate_epoch_evidence_envelope_v1",
        lambda *_args, **_kwargs: {
            "boundary": boundary,
            "boundary_hash": sha256_json(document["capture"]["epoch_boundary"]),
            "boundary_receipt": {
                "receipt_hash": document["capture"][
                    "epoch_boundary_receipt_hash"
                ]
            },
        },
    )

    async def forbidden_persistence(*_args, **_kwargs):
        raise AssertionError("dry-run called durable candidate persistence")

    monkeypatch.setattr(
        authority,
        "persist_pre_cutover_candidate_v1",
        forbidden_persistence,
    )

    async def preview_only(*_args, **_kwargs):
        calls.append("preview")
        return {"candidate_payload_hash": cli.candidate_payload_hash_v1(document)}

    monkeypatch.setattr(cli, "_preview_candidate_row", preview_only)

    report = await cli.ingest_subnet_epoch_candidate_v1(
        document,
        apply=False,
        boot_attestation_verifier=lambda _identity: {},
    )

    assert report["status"] == "validated_no_writes"
    assert report["writes_applied"] is False
    assert report["candidate_payload_hash"] == cli.candidate_payload_hash_v1(
        document
    )
    assert calls == ["signature", "graph", "anchor", "preview"]


@pytest.mark.asyncio
async def test_apply_uses_exact_route_acknowledgment():
    document = _candidate_document()
    observed = []

    async def apply_stage(submission, **_kwargs):
        observed.append(submission.model_dump(mode="python"))
        return _ack(document, durable_hash=HASH_C)

    report = await cli.ingest_subnet_epoch_candidate_v1(
        document,
        apply=True,
        boot_attestation_verifier=lambda _identity: {},
        apply_stage=apply_stage,
    )

    assert observed == [document]
    assert report["status"] == "durably_staged"
    assert report["write_mode"] == "insert_or_exact_replay"
    assert report["durable_readback_hash"] == HASH_C


@pytest.mark.asyncio
async def test_cli_rejects_route_ack_for_another_boundary():
    document = _candidate_document()
    acknowledgment = _ack(document)
    acknowledgment["boundary_block"] += 1

    async def wrong_stage(_submission, **_kwargs):
        return acknowledgment

    with pytest.raises(cli.StatefulEpochCandidateIngestError, match="differs"):
        await cli.ingest_subnet_epoch_candidate_v1(
            document,
            apply=True,
            boot_attestation_verifier=lambda _identity: {},
            apply_stage=wrong_stage,
        )


def test_apply_requires_exact_payload_hash_before_ingest(
    monkeypatch,
    tmp_path,
):
    document = _candidate_document()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    release_path = tmp_path / "validator-release.json"
    release_path.write_text(
        json.dumps(_validator_release_manifest()),
        encoding="utf-8",
    )

    async def forbidden_ingest(*_args, **_kwargs):
        raise AssertionError("wrong confirmation reached candidate ingest")

    monkeypatch.setattr(cli, "ingest_subnet_epoch_candidate_v1", forbidden_ingest)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--candidate",
                str(path),
                "--validator-release-manifest",
                str(release_path),
                "--apply",
                "--confirm-candidate-payload-hash",
                HASH_A,
            ]
        )
    assert exc.value.code == 2


def test_payload_hash_matches_validator_signed_domain():
    document = _candidate_document()
    assert cli.candidate_payload_hash_v1(document) == sha256_json(
        {
            "schema_version": document["schema_version"],
            "cutover_manifest": document["cutover_manifest"],
            "capture": document["capture"],
        }
    )


@pytest.mark.asyncio
async def test_fresh_process_uses_explicit_release_not_empty_dynamic_cache(
    monkeypatch,
):
    from gateway.utils import pcr0_builder

    document = _real_candidate_document()
    pcr0_builder._pcr0_cache.clear()

    def forbidden_dynamic_cache(_pcr0):
        raise AssertionError("one-shot ingest consulted process-local PCR0 cache")

    monkeypatch.setattr(pcr0_builder, "verify_pcr0", forbidden_dynamic_cache)
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: object())
    monkeypatch.setattr(
        weights_api,
        "verify_wallet_signature",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        weights_api,
        "validate_cutover_anchor_from_archive",
        lambda *_args: None,
    )
    nitro_calls = []

    def verify_nitro(identity, *, expected_pcr0):
        nitro_calls.append((identity["boot_identity_hash"], expected_pcr0))
        return {"pcr0": expected_pcr0}

    boot_verifier = cli.build_validator_release_boot_verifier_v1(
        _validator_release_manifest(),
        nitro_verifier=verify_nitro,
    )
    report = await cli.ingest_subnet_epoch_candidate_v1(
        document,
        apply=False,
        boot_attestation_verifier=boot_verifier,
    )

    assert report["status"] == "validated_no_writes"
    assert report["writes_applied"] is False
    assert nitro_calls == [
        (
            document["capture"]["boot_identity"]["boot_identity_hash"],
            PCR0,
        )
    ]
    assert pcr0_builder._pcr0_cache == {}


@pytest.mark.asyncio
async def test_valid_but_mismatched_release_fails_before_preview_persistence(
    monkeypatch,
):
    document = _real_candidate_document()
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {HOTKEY})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "verify_wallet_signature", lambda *_args: True)

    async def forbidden_preview(*_args, **_kwargs):
        raise AssertionError("mismatched release reached preview persistence")

    monkeypatch.setattr(cli, "_preview_candidate_row", forbidden_preview)

    boot_verifier = cli.build_validator_release_boot_verifier_v1(
        _validator_release_manifest(commit="f" * 40),
        nitro_verifier=lambda *_args, **_kwargs: {},
    )
    with pytest.raises(cli.StatefulEpochCandidateIngestError) as exc:
        await cli.ingest_subnet_epoch_candidate_v1(
            document,
            apply=False,
            boot_attestation_verifier=boot_verifier,
        )
    assert "HTTP 400" in str(exc.value)


def test_invalid_release_manifest_file_fails_closed(tmp_path):
    manifest = _validator_release_manifest()
    manifest["release_manifest_hash"] = HASH_A
    path = tmp_path / "validator-release.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(cli.StatefulEpochCandidateIngestError) as exc:
        cli.load_validator_release_manifest_v2(path)
    assert str(exc.value) == "approved validator release manifest is invalid"
