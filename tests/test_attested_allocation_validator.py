import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_receipts import (
    build_receipt_body,
    create_signed_receipt,
    sha256_json as attested_sha256_json,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch
from research_lab.canonical import sha256_json
from research_lab import validator_integration


COMMIT = "1" * 40
PCR0 = "a" * 96


def _signed_receipt(
    *, purpose, job_id, epoch, input_root, output_root, parents=(), evidence_roots=None
):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_receipt_body(
        role="gateway_scoring",
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch,
        commit_sha=COMMIT,
        build_manifest_hash="sha256:" + "2" * 64,
        config_hash="sha256:" + "3" * 64,
        input_root=input_root,
        output_root=output_root,
        evidence_roots=dict(evidence_roots or {}),
        parent_receipt_hashes=list(parents),
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    return create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(b"attestation").decode(),
        sign_digest=private_key.sign,
    )


def _document():
    epoch = 17
    policy = {
        "policy_id": "policy-1",
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
    }
    allocation = allocate_research_lab_epoch(epoch, policy, [], [])
    source_state = {
        "epoch": epoch,
        "netuid": 71,
        "policy_id": "policy-1",
        "policy": policy,
        "reimbursement_obligation_count": 0,
        "champion_obligation_count": 0,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "skipped": {"reimbursements": [], "champions": []},
    }
    bundle = {
        "bundle_id": "research_lab_allocation_bundle:test",
        "bundle_type": "research_lab_live_allocation_bundle",
        "epoch": epoch,
        "netuid": 71,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "source_state": source_state,
        "source_state_hash": sha256_json(source_state),
        "allocation_doc": allocation,
        "allocation_hash": allocation["allocation_hash"],
    }
    receipt_input = {
        "epoch": epoch,
        "policy": policy,
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
        "receipt_lineage_bindings": [],
    }
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_receipt_body(
        role="gateway_scoring",
        purpose="research_lab.allocation.v1",
        job_id="attested-scoring:allocation:test",
        epoch_id=epoch,
        commit_sha=COMMIT,
        build_manifest_hash="sha256:" + "2" * 64,
        config_hash="sha256:" + "3" * 64,
        input_root=attested_sha256_json(receipt_input),
        output_root=attested_sha256_json({"allocation": allocation}),
        evidence_roots={"allocation": allocation["allocation_hash"]},
        parent_receipt_hashes=[],
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    receipt = create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(b"attestation").decode(),
        sign_digest=private_key.sign,
    )
    return {
        "schema_version": validator_integration.ATTESTED_ALLOCATION_SCHEMA_VERSION,
        "bundle": bundle,
        "receipt": receipt,
        "parent_receipts": [],
        "lineage_bindings": [],
        "lineage_complete": True,
        "gateway_pcr0": PCR0,
        "persistence_status": "persisted",
    }


def _fake_nitro(*, receipt, expected_epoch_id, expected_pcr0, expected_purpose):
    if expected_pcr0 and expected_pcr0 != PCR0:
        return False, {"error": "PCR0 mismatch", "pcr0": PCR0}
    return True, {
        "pcr0": PCR0,
        "purpose": expected_purpose,
        "epoch_id": expected_epoch_id,
        "enclave_pubkey": receipt["enclave_pubkey"],
    }


def test_attested_allocation_verifies_but_is_not_required_ready_without_independent_build(monkeypatch):
    monkeypatch.setattr(validator_integration, "verify_gateway_allocation_attestation", _fake_nitro)
    result = validator_integration.verify_research_lab_attested_allocation_bundle(
        _document(),
        flags=validator_integration.ResearchLabValidatorFlags(fetch_enabled=True),
    )

    assert result["passed"] is True
    assert result["required_ready"] is False
    assert result["pcr0_verification_mode"] == "aws_signature_only"


def test_attested_allocation_is_required_ready_with_independent_pcr0_and_commit(monkeypatch):
    monkeypatch.setattr(validator_integration, "verify_gateway_allocation_attestation", _fake_nitro)
    result = validator_integration.verify_research_lab_attested_allocation_bundle(
        _document(),
        flags=validator_integration.ResearchLabValidatorFlags(fetch_enabled=True),
        expected_gateway_pcr0=PCR0,
        expected_gateway_commit=COMMIT,
    )

    assert result["passed"] is True
    assert result["required_ready"] is True
    assert result["pcr0_verification_mode"] == "independent_expected_pcr0"


def test_attested_allocation_rejects_tampered_output(monkeypatch):
    monkeypatch.setattr(validator_integration, "verify_gateway_allocation_attestation", _fake_nitro)
    document = _document()
    document["bundle"]["allocation_doc"]["unallocated_percent"] = 19.0
    result = validator_integration.verify_research_lab_attested_allocation_bundle(
        document,
        flags=validator_integration.ResearchLabValidatorFlags(fetch_enabled=True),
    )

    assert result["passed"] is False
    assert any("allocation" in error for error in result["errors"])


def test_independent_gateway_identity_loader_uses_configured_cache(tmp_path, monkeypatch):
    cache = tmp_path / "gateway-cache.json"
    cache.write_text(
        __import__("json").dumps(
            {
                "schema_version": "leadpoet.gateway_pcr0_cache.v2",
                "entries": [
                    {
                        "commit_sha": COMMIT,
                        "role": "gateway_coordinator",
                        "pcr0": PCR0,
                        "verified_build_count": 3,
                    }
                ],
                "pinned_deployments": [],
            }
        )
    )
    monkeypatch.setenv("VALIDATOR_GATEWAY_PCR0_CACHE_FILE", str(cache))

    assert validator_integration.load_independent_gateway_identity(COMMIT)["pcr0"] == PCR0
    assert validator_integration.load_independent_gateway_identity("f" * 40) is None


def _document_with_lineage():
    epoch = 17
    score_bundle_hash = "sha256:" + "9" * 64
    score_bundle_id = "score_bundle:" + "9" * 64
    policy = {
        "policy_id": "policy-lineage",
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
    }
    champion = {
        "uid": 14,
        "miner_hotkey": "5Fchampion14",
        "source_id": "champion_reward:14",
        "champion_reward_id": "champion_reward:14",
        "candidate_id": "candidate:14",
        "score_bundle_id": score_bundle_id,
        "run_id": "run:14",
        "island": "generalist",
        "start_epoch": 10,
        "epoch_count": 20,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 5.0,
    }
    allocation = allocate_research_lab_epoch(epoch, policy, [], [champion])
    source_state = {
        "epoch": epoch,
        "netuid": 71,
        "policy_id": "policy-lineage",
        "policy": policy,
        "reimbursement_obligation_count": 0,
        "champion_obligation_count": 1,
        "reimbursement_obligations": [],
        "champion_obligations": [champion],
        "skipped": {"reimbursements": [], "champions": []},
    }
    bundle = {
        "bundle_id": "research_lab_allocation_bundle:lineage",
        "bundle_type": "research_lab_live_allocation_bundle",
        "epoch": epoch,
        "netuid": 71,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "source_state": source_state,
        "source_state_hash": sha256_json(source_state),
        "allocation_doc": allocation,
        "allocation_hash": allocation["allocation_hash"],
    }
    baseline_summary_hash = "sha256:" + "8" * 64
    baseline_receipt = _signed_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline:14",
        epoch=16,
        input_root="sha256:" + "9" * 64,
        output_root="sha256:" + "a" * 64,
        evidence_roots={"baseline_score_summary": baseline_summary_hash},
    )
    score_receipt = _signed_receipt(
        purpose="research_lab.candidate_score.v1",
        job_id="score:14",
        epoch=16,
        input_root="sha256:" + "4" * 64,
        output_root="sha256:" + "5" * 64,
        parents=[baseline_receipt["receipt_hash"]],
        evidence_roots={
            "score_bundle": score_bundle_hash,
            "baseline_score_summary": baseline_summary_hash,
        },
    )
    promotion_receipt = _signed_receipt(
        purpose="research_lab.promotion_decision.v1",
        job_id="promotion:14",
        epoch=16,
        input_root="sha256:" + "6" * 64,
        output_root="sha256:" + "7" * 64,
        parents=[score_receipt["receipt_hash"]],
        evidence_roots={
            "score_bundle": score_bundle_hash,
            "promotion_decision_status": attested_sha256_json(
                {"status": "promotion_passed"}
            ),
        },
    )
    bindings = [
        {
            "score_bundle_id": score_bundle_id,
            "score_bundle_hash": score_bundle_hash,
            "receipt_hash": promotion_receipt["receipt_hash"],
            "receipt_purpose": promotion_receipt["purpose"],
        }
    ]
    receipt_input = {
        "epoch": epoch,
        "policy": policy,
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [champion],
        "receipt_lineage_bindings": bindings,
    }
    allocation_receipt = _signed_receipt(
        purpose="research_lab.allocation.v1",
        job_id="allocation:lineage",
        epoch=epoch,
        input_root=attested_sha256_json(receipt_input),
        output_root=attested_sha256_json({"allocation": allocation}),
        parents=[promotion_receipt["receipt_hash"]],
        evidence_roots={"allocation": allocation["allocation_hash"]},
    )
    return {
        "schema_version": validator_integration.ATTESTED_ALLOCATION_SCHEMA_VERSION,
        "bundle": bundle,
        "receipt": allocation_receipt,
        "parent_receipts": [baseline_receipt, score_receipt, promotion_receipt],
        "lineage_bindings": bindings,
        "lineage_complete": True,
        "gateway_pcr0": PCR0,
        "persistence_status": "persisted",
    }


def test_attested_allocation_verifies_complete_score_to_promotion_lineage(monkeypatch):
    monkeypatch.setattr(validator_integration, "verify_gateway_allocation_attestation", _fake_nitro)
    document = _document_with_lineage()
    result = validator_integration.verify_research_lab_attested_allocation_bundle(
        document,
        flags=validator_integration.ResearchLabValidatorFlags(fetch_enabled=True),
        expected_gateway_identities={COMMIT: {"commit_sha": COMMIT, "pcr0": PCR0}},
    )

    assert result["passed"] is True
    assert result["required_ready"] is True
    assert result["lineage_receipt_count"] == 3


def test_attested_allocation_rejects_missing_transitive_parent(monkeypatch):
    monkeypatch.setattr(validator_integration, "verify_gateway_allocation_attestation", _fake_nitro)
    document = _document_with_lineage()
    document["parent_receipts"] = document["parent_receipts"][1:]
    result = validator_integration.verify_research_lab_attested_allocation_bundle(
        document,
        flags=validator_integration.ResearchLabValidatorFlags(fetch_enabled=True),
        expected_gateway_identities={COMMIT: {"commit_sha": COMMIT, "pcr0": PCR0}},
    )

    assert result["passed"] is False
    assert any("missing parent" in error for error in result["errors"])
