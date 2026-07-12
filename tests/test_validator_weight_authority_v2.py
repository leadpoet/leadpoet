from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    WEIGHT_ROLE,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    merkle_root,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    weight_input_output_roots_v2,
    weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)
from validator_tee.enclave.weight_authority_v2 import (
    ValidatorWeightAuthorityV2,
    ValidatorWeightAuthorityV2Error,
)


NOW = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
COMMIT = "a" * 40
GATEWAY_COMMIT = "b" * 40
VALIDATOR_PCR0 = "c" * 96
GATEWAY_PCR0 = "d" * 96
VALIDATOR_MANIFEST = "sha256:" + "1" * 64
GATEWAY_MANIFEST = "sha256:" + "2" * 64
VALIDATOR_LOCK = "sha256:" + "3" * 64
GATEWAY_LOCK = "sha256:" + "4" * 64
VALIDATOR_BOOT_CONFIG = "sha256:" + "5" * 64
GATEWAY_BOOT_CONFIG = "sha256:" + "6" * 64


def _keypair():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return private_key, public_key


def _boot(
    *,
    role,
    physical_role,
    commit,
    pcr0,
    manifest,
    dependency_lock,
    config_hash,
    private_key,
    public_key,
    nonce,
):
    del private_key
    body = build_boot_identity_body(
        role=role,
        physical_role=physical_role,
        commit_sha=commit,
        pcr0=pcr0,
        build_manifest_hash=manifest,
        dependency_lock_hash=dependency_lock,
        config_hash=config_hash,
        boot_nonce=nonce * 32,
        signing_pubkey=public_key,
        transport_pubkey=public_key,
        transport_certificate_hash=sha256_json(
            {"physical_role": physical_role, "kind": "transport"}
        ),
        attestation_user_data_hash=sha256_json(
            {"physical_role": physical_role, "kind": "claim"}
        ),
        issued_at="2026-07-10T20:00:00Z",
    )
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(
            ("nitro:" + physical_role).encode("ascii")
        ).decode("ascii"),
    )


def _receipt(
    *,
    boot,
    private_key,
    purpose,
    job_id,
    sequence,
    output_root,
    transport_root=EMPTY_TRANSPORT_ROOT,
    artifact_root=EMPTY_ARTIFACT_ROOT,
    artifact_domain="leadpoet-artifact-v2",
):
    body = build_execution_receipt_body(
        role=boot["role"],
        purpose=purpose,
        job_id=job_id,
        epoch_id=100,
        sequence=sequence,
        commit_sha=boot["commit_sha"],
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=boot["config_hash"],
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=sha256_json({"job_id": job_id, "side": "input"}),
        output_root=output_root,
        transport_root_hash=transport_root,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-07-10T20:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=private_key.sign,
    )


def _source_attempt(*, category, job_id, purpose, sequence, provider_id, host, method):
    return build_transport_attempt(
        request_id="%032x" % (sequence + 1),
        logical_operation_id="weight-source:%s" % category,
        job_id=job_id,
        purpose=purpose,
        provider_id=provider_id,
        attempt_number=0,
        method=method,
        destination_host=host,
        destination_port=443,
        path_hash=sha256_json({"category": category, "path": "source"}),
        nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
        body_hash=sha256_json({"body": ""}),
        credential_ref_hash=sha256_json({"credential": provider_id}),
        retry_policy_hash=sha256_json({"retry": provider_id}),
        timeout_ms=30000,
        started_at="2026-07-10T20:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=sha256_json({"category": category, "response": "body"}),
        request_artifact_hash=sha256_json(
            {"category": category, "artifact": "request"}
        ),
        response_artifact_hash=sha256_json(
            {"category": category, "artifact": "response"}
        ),
        tls_peer_chain_hash=sha256_json({"tls": host}),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-07-10T20:00:00Z",
    )


def _calculation_snapshot(parent_hashes, allocation_hash):
    snapshot = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "commit_sha": COMMIT,
        "config_hash": "",
        "parent_receipt_hashes": sorted(parent_hashes),
        "research_lab_allocation_receipt_hash": allocation_hash,
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn-hotkey",
        "metagraph_hotkeys": [
            "burn-hotkey",
            "fulfillment-hotkey",
            "lab-hotkey",
            "source-hotkey",
        ],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {
            "lab_cap_percent": 20.0,
            "unallocated_percent": 15.0,
            "reimbursement_allocations": [],
            "champion_allocations": [
                {
                    "uid": 2,
                    "miner_hotkey": "lab-hotkey",
                    "paid_alpha_percent": 5.0,
                }
            ],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [
            {"miner_hotkey": "fulfillment-hotkey", "wins": 9}
        ],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [
            {"hotkey": "fulfillment-hotkey", "share": 0.705}
        ],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125_000,
        "min_total_rep_for_distribution": 100,
    }
    snapshot["config_hash"] = weight_config_hash(snapshot)
    return snapshot


def _fixture(*, category_output_override=None):
    validator_key, validator_pub = _keypair()
    gateway_key, gateway_pub = _keypair()
    validator_boot = _boot(
        role=WEIGHT_ROLE,
        physical_role="validator_weights",
        commit=COMMIT,
        pcr0=VALIDATOR_PCR0,
        manifest=VALIDATOR_MANIFEST,
        dependency_lock=VALIDATOR_LOCK,
        config_hash=VALIDATOR_BOOT_CONFIG,
        private_key=validator_key,
        public_key=validator_pub,
        nonce="7",
    )
    gateway_boot = _boot(
        role=COORDINATOR_ROLE,
        physical_role="gateway_coordinator",
        commit=GATEWAY_COMMIT,
        pcr0=GATEWAY_PCR0,
        manifest=GATEWAY_MANIFEST,
        dependency_lock=GATEWAY_LOCK,
        config_hash=GATEWAY_BOOT_CONFIG,
        private_key=gateway_key,
        public_key=gateway_pub,
        nonce="8",
    )
    preliminary = _calculation_snapshot([], "")
    finalized_chain_state_root = sha256_json({"block": 36099})
    gateway_authority_event_hash = sha256_json({"epoch": 100})
    expected_output_roots = weight_input_output_roots_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    input_documents = weight_input_value_documents_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    receipts = []
    attempts = []
    input_hashes = {}
    for sequence, category in enumerate(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)):
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        assert role == COORDINATOR_ROLE
        boot = gateway_boot
        key = gateway_key
        job_id = "weight-input-%s" % category
        attempt = None
        if role == COORDINATOR_ROLE and category != "anomaly_adjustments":
            attempt = _source_attempt(
                category=category,
                job_id=job_id,
                purpose=purpose,
                sequence=sequence,
                provider_id="supabase",
                host="qplwoislplkcegvdmbim.supabase.co",
                method="GET",
            )
        artifact_hashes = [sha256_json(input_documents[category]["value"])]
        if attempt is not None:
            attempts.append(attempt)
            artifact_hashes.extend(
                [attempt["request_artifact_hash"], attempt["response_artifact_hash"]]
            )
        receipt = _receipt(
            boot=boot,
            private_key=key,
            purpose=purpose,
            job_id=job_id,
            sequence=sequence,
            output_root=(
                category_output_override[category]
                if category_output_override
                and category in category_output_override
                else expected_output_roots[category]
            ),
            transport_root=(
                merkle_root(
                    [attempt["attempt_hash"]],
                    domain="leadpoet-transport-v2",
                )
                if attempt is not None
                else EMPTY_TRANSPORT_ROOT
            ),
            artifact_root=merkle_root(
                artifact_hashes,
                domain="leadpoet-artifact-v2",
            ),
        )
        receipts.append(receipt)
        input_hashes[category] = receipt["receipt_hash"]
    calculation = _calculation_snapshot(
        input_hashes.values(),
        input_hashes["research_lab_allocation"],
    )
    request = {
        "validator_hotkey": "5ValidatorHotkey",
        "calculation_snapshot": calculation,
        "input_receipt_hashes": input_hashes,
        "gateway_authority_event_hash": gateway_authority_event_hash,
        "upstream_receipt_set": {
            "boot_identities": [gateway_boot],
            "receipts": receipts,
            "transport_attempts": attempts,
            "host_operations": [],
        },
    }
    expectations = {
        "gateway_coordinator": {
            "commit_sha": GATEWAY_COMMIT,
            "pcr0": GATEWAY_PCR0,
            "build_manifest_hash": GATEWAY_MANIFEST,
        }
    }
    verified_boots = []

    def verify_boot(identity, *, expected_pcr0=None):
        if identity["pcr0"] != expected_pcr0:
            raise ValueError("PCR0 mismatch")
        verified_boots.append(identity["boot_identity_hash"])
        return {"verified": True}

    chain_artifacts = []
    chain_attempts = []
    for offset, (job_id, purpose, operation) in enumerate(
        (
            ("chain-state:100", "validator.chain_state.v2", "head"),
            ("chain-state:100", "validator.chain_state.v2", "header"),
            (
                "metagraph-state:100",
                "validator.metagraph_state.v2",
                "metagraph",
            ),
        )
    ):
        request_body = ("request:" + operation).encode("ascii")
        response_body = ("response:" + operation).encode("ascii")
        request_hash = sha256_bytes(request_body)
        response_hash = sha256_bytes(response_body)
        attempt = build_transport_attempt(
            request_id="%032x" % (100 + offset),
            logical_operation_id=job_id + ":" + operation,
            job_id=job_id,
            purpose=purpose,
            provider_id="bittensor_chain",
            attempt_number=offset if job_id == "chain-state:100" else 0,
            method="POST",
            destination_host="entrypoint-finney.opentensor.ai",
            destination_port=443,
            path_hash=sha256_json({"path": "/"}),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=request_hash,
            credential_ref_hash=sha256_json({"credential": "none"}),
            retry_policy_hash=sha256_json({"retry": "chain"}),
            timeout_ms=30000,
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=response_hash,
            request_artifact_hash=request_hash,
            response_artifact_hash=response_hash,
            tls_peer_chain_hash=sha256_json({"tls": "chain"}),
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        chain_attempts.append(attempt)
        chain_artifacts.extend(
            [
                {
                    "artifact_hash": request_hash,
                    "kind": "chain_rpc_request",
                    "body_b64": base64.b64encode(request_body).decode("ascii"),
                },
                {
                    "artifact_hash": response_hash,
                    "kind": "chain_rpc_response",
                    "body_b64": base64.b64encode(response_body).decode("ascii"),
                },
            ]
        )

    class FakeChainSource:
        def read_finalized_snapshot(self, *, netuid, epoch_id):
            assert netuid == 71
            assert epoch_id == 100
            return {
                "finalized_block_hash": "ab" * 32,
                "header": {
                    "block": 36099,
                    "state_root": "12" * 32,
                    "state_root_commitment": finalized_chain_state_root,
                    "parent_hash": "34" * 32,
                    "extrinsics_root": "56" * 32,
                },
                "metagraph": {
                    "netuid": 71,
                    "block": 36099,
                    "owner_hotkey": "burn-hotkey",
                    "hotkeys": list(preliminary["metagraph_hotkeys"]),
                },
                "attempts": list(chain_attempts),
                "artifacts": list(chain_artifacts),
                "jobs": {
                    "chain_state": "chain-state:100",
                    "metagraph_state": "metagraph-state:100",
                },
            }

    chain_source = FakeChainSource()
    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: validator_boot,
        gateway_expectations_supplier=lambda: expectations,
        sign_digest=validator_key.sign,
        chain_source=chain_source,
        boot_verifier=verify_boot,
        clock=lambda: NOW,
    )
    return {
        "authority": authority,
        "request": request,
        "validator_key": validator_key,
        "validator_boot": validator_boot,
        "gateway_boot": gateway_boot,
        "expectations": expectations,
        "verified_boots": verified_boots,
        "finalized_chain_state_root": finalized_chain_state_root,
        "chain_source": chain_source,
    }


def test_validator_authority_computes_and_signs_exact_canonical_weights():
    fixture = _fixture()
    value = fixture["authority"].compute(fixture["request"])
    expected = compute_final_weights(value["weight_snapshot"]["calculation_snapshot"])
    assert value["weight_result"] == expected
    assert value["weight_snapshot"]["calculation_snapshot"]["block"] == 36099
    assert value["weight_snapshot"]["calculation_snapshot"]["metagraph_hotkeys"] == fixture["request"]["calculation_snapshot"]["metagraph_hotkeys"]
    assert set(value["weight_snapshot"]["input_receipt_hashes"]) == set(WEIGHT_INPUT_PURPOSES)
    assert value["source_artifacts"]
    assert (
        fixture["validator_boot"]["config_hash"]
        != value["weight_snapshot"]["calculation_snapshot"]["config_hash"]
    )
    Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(fixture["validator_boot"]["signing_pubkey"])
    ).verify(
        bytes.fromhex(value["weights_signature"]),
        bytes.fromhex(expected["weights_hash"]),
    )
    purposes = {item["purpose"] for item in value["receipt_graph"]["receipts"]}
    assert "validator.weight_snapshot.v2" in purposes
    assert "validator.weights.computed.v2" in purposes
    assert set(fixture["verified_boots"]) == {
        fixture["validator_boot"]["boot_identity_hash"],
        fixture["gateway_boot"]["boot_identity_hash"],
    }


def test_validator_authority_rejects_missing_semantic_input_category():
    fixture = _fixture()
    del fixture["request"]["input_receipt_hashes"]["bans"]
    with pytest.raises(ValidatorWeightAuthorityV2Error, match="categories"):
        fixture["authority"].compute(fixture["request"])


def test_validator_authority_rejects_category_bound_to_wrong_purpose():
    fixture = _fixture()
    fixture["request"]["input_receipt_hashes"]["champions"] = fixture[
        "request"
    ]["input_receipt_hashes"]["reimbursements"]
    with pytest.raises(ValidatorWeightAuthorityV2Error, match="role/purpose"):
        fixture["authority"].compute(fixture["request"])


def test_validator_authority_rejects_category_receipt_for_unrelated_value():
    fixture = _fixture(category_output_override={"bans": "sha256:" + "9" * 64})
    with pytest.raises(
        ValidatorWeightAuthorityV2Error,
        match="weight input receipt output differs: bans",
    ):
        fixture["authority"].compute(fixture["request"])


def test_validator_authority_rejects_forged_upstream_receipt():
    fixture = _fixture()
    forged = copy.deepcopy(
        fixture["request"]["upstream_receipt_set"]["receipts"][0]
    )
    forged["enclave_signature"] = "0" * 128
    fixture["request"]["upstream_receipt_set"]["receipts"][0] = forged
    with pytest.raises(Exception, match="signature"):
        fixture["authority"].compute(fixture["request"])


def test_validator_authority_rejects_gateway_release_mismatch():
    fixture = _fixture()
    bad_expectations = copy.deepcopy(fixture["expectations"])
    bad_expectations["gateway_coordinator"]["build_manifest_hash"] = (
        "sha256:" + "9" * 64
    )
    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: fixture["validator_boot"],
        gateway_expectations_supplier=lambda: bad_expectations,
        sign_digest=fixture["validator_key"].sign,
        chain_source=fixture["chain_source"],
        boot_verifier=lambda identity, expected_pcr0=None: {"verified": True},
        clock=lambda: NOW,
    )
    with pytest.raises(ValidatorWeightAuthorityV2Error, match="gateway boot"):
        authority.compute(fixture["request"])


def test_validator_authority_fails_when_nitro_verifier_rejects_boot():
    fixture = _fixture()

    def reject_boot(identity, *, expected_pcr0=None):
        del identity, expected_pcr0
        raise ValueError("Nitro signature invalid")

    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: fixture["validator_boot"],
        gateway_expectations_supplier=lambda: fixture["expectations"],
        sign_digest=fixture["validator_key"].sign,
        chain_source=fixture["chain_source"],
        boot_verifier=reject_boot,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="Nitro signature"):
        authority.compute(fixture["request"])


def test_validator_authority_rejects_disconnected_or_missing_boot_identity():
    fixture = _fixture()
    fixture["request"]["upstream_receipt_set"]["boot_identities"] = [
        fixture["validator_boot"]
    ]
    with pytest.raises(Exception, match="boot identity is missing"):
        fixture["authority"].compute(fixture["request"])
