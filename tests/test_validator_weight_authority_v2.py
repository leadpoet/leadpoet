from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import json

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
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    validate_published_weight_bundle_v2,
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
from validator_tee.host import subnet_epoch_boundary_capture_v2 as capture_module
from validator_tee.host.subnet_epoch_boundary_capture_v2 import (
    SubnetEpochBoundaryCaptureV2Error,
    build_subnet_epoch_candidate_authorization_message_v1,
    capture_subnet_epoch_boundary_v2,
    publish_subnet_epoch_boundary_candidate_v1,
)
from validator_tee.host.publication_journal_v2 import (
    LEGACY_JOURNAL_SCHEMA_VERSION,
    WeightPublicationJournalV2Error,
    validate_publication_journal_v2,
)
from validator_tee.host.weight_authority_v2 import (
    HostWeightAuthorityV2Error,
    build_authoritative_weight_bundle_v2,
    build_stateful_epoch_evidence_v1,
    validate_stateful_epoch_evidence_v1,
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
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


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
    parent_receipt_hashes=(),
    epoch_id=100,
):
    body = build_execution_receipt_body(
        role=boot["role"],
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
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
        parent_receipt_hashes=parent_receipt_hashes,
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


def _fixture(
    *,
    category_output_override=None,
    stateful=False,
    historical_validator_ancestry=False,
):
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
    historical_boot = None
    historical_receipts = []
    historical_commit = "f" * 40
    if historical_validator_ancestry:
        historical_key, historical_pub = _keypair()
        historical_boot = _boot(
            role=WEIGHT_ROLE,
            physical_role="validator_weights",
            commit=historical_commit,
            pcr0="a" * 96,
            manifest="sha256:" + "b" * 64,
            dependency_lock="sha256:" + "c" * 64,
            config_hash="sha256:" + "d" * 64,
            private_key=historical_key,
            public_key=historical_pub,
            nonce="9",
        )
        historical_snapshot = _receipt(
            boot=historical_boot,
            private_key=historical_key,
            purpose="validator.weight_snapshot.v2",
            job_id="historical-weight-snapshot:99",
            sequence=0,
            output_root=sha256_json({"epoch_id": 99, "kind": "snapshot"}),
            epoch_id=99,
        )
        historical_computed = _receipt(
            boot=historical_boot,
            private_key=historical_key,
            purpose="validator.weights.computed.v2",
            job_id="historical-weight-computation:99",
            sequence=1,
            output_root=sha256_json({"epoch_id": 99, "kind": "computed"}),
            parent_receipt_hashes=(historical_snapshot["receipt_hash"],),
            epoch_id=99,
        )
        historical_finalized = _receipt(
            boot=historical_boot,
            private_key=historical_key,
            purpose="validator.weights.finalized.v2",
            job_id="historical-weight-finalization:99",
            sequence=2,
            output_root=sha256_json({"epoch_id": 99, "kind": "finalized"}),
            parent_receipt_hashes=(historical_computed["receipt_hash"],),
            epoch_id=99,
        )
        historical_receipts = [
            historical_snapshot,
            historical_computed,
            historical_finalized,
        ]
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
    receipts = list(historical_receipts)
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
            parent_receipt_hashes=(
                (historical_receipts[-1]["receipt_hash"],)
                if historical_receipts
                and category == "research_lab_allocation"
                else ()
            ),
        )
        receipts.append(receipt)
        input_hashes[category] = receipt["receipt_hash"]
    calculation = _calculation_snapshot(
        input_hashes.values(),
        input_hashes["research_lab_allocation"],
    )
    request = {
        "validator_hotkey": VALIDATOR_HOTKEY,
        "calculation_snapshot": calculation,
        "input_receipt_hashes": input_hashes,
        "gateway_authority_event_hash": gateway_authority_event_hash,
        "upstream_receipt_set": {
            "boot_identities": [gateway_boot]
            + ([historical_boot] if historical_boot is not None else []),
            "receipts": receipts,
            "transport_attempts": attempts,
            "host_operations": [],
        },
    }
    expectations = {
        GATEWAY_COMMIT: {
            "roles": {
                "gateway_coordinator": {
                    "commit_sha": GATEWAY_COMMIT,
                    "pcr0": GATEWAY_PCR0,
                    "build_manifest_hash": GATEWAY_MANIFEST,
                    "dependency_lock_hash": GATEWAY_LOCK,
                }
            }
        }
    }
    if historical_boot is not None:
        expectations[historical_commit] = {
            "roles": {
                "validator_weights": {
                    "commit_sha": historical_commit,
                    "pcr0": historical_boot["pcr0"],
                    "build_manifest_hash": historical_boot[
                        "build_manifest_hash"
                    ],
                    "dependency_lock_hash": historical_boot[
                        "dependency_lock_hash"
                    ],
                }
            }
        }
    verified_boots = []
    boot_verification_modes = {}

    def verify_boot(
        identity,
        *,
        expected_pcr0=None,
        certificate_validity_at_attestation_time=False,
    ):
        if identity["pcr0"] != expected_pcr0:
            raise ValueError("PCR0 mismatch")
        verified_boots.append(identity["boot_identity_hash"])
        boot_verification_modes[identity["boot_identity_hash"]] = (
            certificate_validity_at_attestation_time
        )
        return {"verified": True}

    chain_artifacts = []
    chain_attempts = []
    chain_operations = [
            ("chain-state:100", "validator.chain_state.v2", "head"),
            ("chain-state:100", "validator.chain_state.v2", "header"),
            (
                "metagraph-state:100",
                "validator.metagraph_state.v2",
                "metagraph",
            ),
    ]
    if stateful:
        chain_operations.extend(
            [
                (
                    "subnet-epoch-snapshot:100:36099",
                    "validator.subnet_epoch_snapshot.v2",
                    "current",
                ),
                (
                    "subnet-epoch-boundary:100",
                    "validator.subnet_epoch_snapshot.v2",
                    "boundary",
                ),
            ]
        )
    for offset, (job_id, purpose, operation) in enumerate(chain_operations):
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
            value = {
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
            if stateful:
                cutover_body = {
                    "schema_version": "leadpoet.subnet_epoch_cutover.v1",
                    "epoch_scheme": "bittensor.subnet_epoch_index.v1",
                    "network_genesis_hash": "0x" + "1" * 64,
                    "netuid": 71,
                    "cutover_block": 36_000,
                    "cutover_block_hash": "0x" + "2" * 64,
                    "first_subnet_epoch_index": 35,
                    "first_settlement_epoch_id": 100,
                    "last_legacy_epoch_id": 99,
                }
                cutover_mapping_hash = sha256_json(cutover_body)
                value["epoch_boundary"] = {
                    "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
                    "epoch_scheme": "bittensor.subnet_epoch_index.v1",
                    "network_genesis_hash": "0x" + "1" * 64,
                    "netuid": 71,
                    "head_kind": "finalized",
                    "block_hash": "0x" + "2" * 64,
                    "current_block": 36_000,
                    "last_epoch_block": 36_000,
                    "pending_epoch_at": 0,
                    "subnet_epoch_index": 35,
                    "tempo": 360,
                    "blocks_since_last_step": 0,
                    "observed_at": "2026-07-10T19:59:00Z",
                    "epoch_id": 35,
                    "epoch_ref": "sha256:" + "8" * 64,
                    "epoch_block": 0,
                    "next_epoch_block": 36_360,
                    "blocks_remaining": 360,
                    "settlement_epoch_id": 100,
                    "cutover_mapping_hash": cutover_mapping_hash,
                }
                value["epoch_authority"] = {
                    **value["epoch_boundary"],
                    "block_hash": "0x" + "3" * 64,
                    "current_block": 36_099,
                    "last_epoch_block": 36_000,
                    "observed_at": "2026-07-10T20:08:54Z",
                    "epoch_block": 99,
                    "blocks_remaining": 261,
                }
                value["jobs"]["subnet_epoch_snapshot"] = (
                    "subnet-epoch-snapshot:100:36099"
                )
                value["jobs"]["subnet_epoch_boundary"] = (
                    "subnet-epoch-boundary:100"
                )
            return value

        def capture_stateful_epoch_boundary(
            self, *, cutover_manifest, settlement_epoch_id, capture_scope
        ):
            assert stateful is True
            assert settlement_epoch_id == 100
            assert cutover_manifest["first_settlement_epoch_id"] == 100
            assert capture_scope == validator_boot["boot_identity_hash"]
            value = self.read_finalized_snapshot(netuid=71, epoch_id=100)
            capture_attempts = [
                item
                for item in value["attempts"]
                if item["job_id"] == value["jobs"]["subnet_epoch_boundary"]
            ]
            capture_artifact_hashes = {
                artifact_hash
                for item in capture_attempts
                for artifact_hash in (
                    item["request_artifact_hash"],
                    item["response_artifact_hash"],
                )
            }
            return {
                "finalized_block_hash": value["finalized_block_hash"],
                "header": value["header"],
                "epoch_authority": value["epoch_boundary"],
                "epoch_boundary": value["epoch_boundary"],
                "attempts": capture_attempts,
                "artifacts": [
                    item
                    for item in value["artifacts"]
                    if item["artifact_hash"] in capture_artifact_hashes
                ],
                "jobs": {
                    "subnet_epoch_snapshot": value["jobs"][
                        "subnet_epoch_boundary"
                    ],
                    "subnet_epoch_boundary": value["jobs"][
                        "subnet_epoch_boundary"
                    ],
                },
            }

    chain_source = FakeChainSource()
    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: validator_boot,
        gateway_release_lineage_supplier=lambda: expectations,
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
        "gateway_key": gateway_key,
        "historical_boot": historical_boot,
        "historical_receipts": historical_receipts,
        "expectations": expectations,
        "verified_boots": verified_boots,
        "boot_verification_modes": boot_verification_modes,
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
    assert fixture["boot_verification_modes"] == {
        fixture["validator_boot"]["boot_identity_hash"]: True,
        fixture["gateway_boot"]["boot_identity_hash"]: True,
    }


def test_validator_authority_preserves_approved_historical_validator_ancestry():
    fixture = _fixture(historical_validator_ancestry=True)
    enclave_response = fixture["authority"].compute(fixture["request"])
    enclave_response["weight_authorization_id"] = "sha256:" + "a" * 64
    graph = enclave_response["receipt_graph"]
    historical_boot = fixture["historical_boot"]
    historical_hashes = {
        receipt["receipt_hash"] for receipt in fixture["historical_receipts"]
    }

    assert historical_boot is not None
    assert historical_boot["boot_identity_hash"] in {
        boot["boot_identity_hash"] for boot in graph["boot_identities"]
    }
    assert historical_hashes.issubset(
        {receipt["receipt_hash"] for receipt in graph["receipts"]}
    )
    assert fixture["boot_verification_modes"][
        historical_boot["boot_identity_hash"]
    ] is True

    boot = fixture["validator_boot"]
    validator_hotkey = fixture["request"]["validator_hotkey"]
    binding_message = create_binding_message(
        netuid=71,
        chain="wss://entrypoint-finney.opentensor.ai:443",
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    application_request = build_application_signature_request_v2(
        message=binding_message.encode("utf-8"),
        validator_hotkey=validator_hotkey,
        boot_identity_hash=boot["boot_identity_hash"],
    )
    hotkey_signature = "f" * 128
    output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": application_request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": validator_hotkey,
        "signature": hotkey_signature,
    }
    binding_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=WEIGHT_ROLE,
            purpose="validator.hotkey_signature.v2",
            job_id="application-signature:%s"
            % application_request["request_hash"].split(":", 1)[1][:32],
            epoch_id=100,
            sequence=0,
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=application_request["request_hash"],
            output_root=sha256_json(output),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(graph["root_receipt_hash"],),
            status="succeeded",
            failure_code=None,
            issued_at="2026-07-10T20:00:00Z",
        ),
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=fixture["validator_key"].sign,
    )
    bundle = build_authoritative_weight_bundle_v2(
        enclave_response=enclave_response,
        validator_hotkey=validator_hotkey,
        binding_message=binding_message,
        binding_signature_result={**output, "receipt": binding_receipt},
    )
    verified = validate_published_weight_bundle_v2(bundle)

    assert verified["validator_boot_identity_hash"] == boot["boot_identity_hash"]
    assert verified["weight_receipt_hash"] == graph["root_receipt_hash"]


def test_validator_authority_connects_gateway_persistence_receipts():
    fixture = _fixture()
    upstream = fixture["request"]["upstream_receipt_set"]
    persistence_hashes = set()
    for sequence, input_receipt in enumerate(list(upstream["receipts"])):
        persistence_receipt = _receipt(
            boot=fixture["gateway_boot"],
            private_key=fixture["gateway_key"],
            purpose="leadpoet.artifact_persistence.v2",
            job_id="weight-input-persistence-%d" % sequence,
            sequence=sequence,
            output_root=sha256_json(
                {"persisted_receipt_hash": input_receipt["receipt_hash"]}
            ),
            parent_receipt_hashes=(input_receipt["receipt_hash"],),
        )
        upstream["receipts"].append(persistence_receipt)
        persistence_hashes.add(persistence_receipt["receipt_hash"])

    value = fixture["authority"].compute(fixture["request"])
    snapshot_receipt = next(
        receipt
        for receipt in value["receipt_graph"]["receipts"]
        if receipt["purpose"] == "validator.weight_snapshot.v2"
    )

    assert persistence_hashes.issubset(snapshot_receipt["parent_receipt_hashes"])


def test_stateful_authority_emits_dedicated_current_and_boundary_receipts():
    fixture = _fixture(stateful=True)
    value = fixture["authority"].compute(fixture["request"])
    receipts = value["receipt_graph"]["receipts"]
    epoch_receipts = [
        item
        for item in receipts
        if item["purpose"] == "validator.subnet_epoch_snapshot.v2"
    ]
    assert len(epoch_receipts) == 2
    source = fixture["chain_source"].read_finalized_snapshot(
        netuid=71,
        epoch_id=100,
    )
    boundary = source["epoch_boundary"]
    current = source["epoch_authority"]
    boundary_receipt = next(
        item for item in epoch_receipts if item["output_root"] == sha256_json(boundary)
    )
    current_receipt = next(
        item for item in epoch_receipts if item["output_root"] == sha256_json(current)
    )
    assert boundary_receipt["output_root"] == sha256_json(boundary)
    assert boundary_receipt["issued_at"] == boundary["observed_at"]
    assert current_receipt["issued_at"] == current["observed_at"]
    assert current_receipt["parent_receipt_hashes"] == [
        boundary_receipt["receipt_hash"]
    ]
    assert value["epoch_authority"] == current
    assert value["epoch_boundary"] == boundary
    chain_receipt = next(
        item for item in receipts if item["purpose"] == "validator.chain_state.v2"
    )
    assert chain_receipt["parent_receipt_hashes"] == [
        current_receipt["receipt_hash"]
    ]
    boundary_attempts = [
        item
        for item in value["receipt_graph"]["transport_attempts"]
        if item["purpose"] == "validator.subnet_epoch_snapshot.v2"
    ]
    assert len(boundary_attempts) == 2


def test_host_preserves_stateful_epoch_evidence_outside_unchanged_v2_bundle():
    fixture = _fixture(stateful=True)
    enclave_response = fixture["authority"].compute(fixture["request"])
    enclave_response["weight_authorization_id"] = "sha256:" + "a" * 64
    boot = fixture["validator_boot"]
    validator_hotkey = fixture["request"]["validator_hotkey"]
    binding_message = create_binding_message(
        netuid=71,
        chain="wss://entrypoint-finney.opentensor.ai:443",
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    application_request = build_application_signature_request_v2(
        message=binding_message.encode("utf-8"),
        validator_hotkey=validator_hotkey,
        boot_identity_hash=boot["boot_identity_hash"],
    )
    hotkey_signature = "f" * 128
    output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": application_request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": validator_hotkey,
        "signature": hotkey_signature,
    }
    binding_body = build_execution_receipt_body(
        role="validator_weights",
        purpose="validator.hotkey_signature.v2",
        job_id="application-signature:%s"
        % application_request["request_hash"].split(":", 1)[1][:32],
        epoch_id=100,
        sequence=0,
        commit_sha=boot["commit_sha"],
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=boot["config_hash"],
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=application_request["request_hash"],
        output_root=sha256_json(output),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(
            enclave_response["receipt_graph"]["root_receipt_hash"],
        ),
        status="succeeded",
        failure_code=None,
        issued_at="2026-07-10T20:00:00Z",
    )
    binding_receipt = create_signed_execution_receipt(
        body=binding_body,
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=fixture["validator_key"].sign,
    )
    bundle = build_authoritative_weight_bundle_v2(
        enclave_response=enclave_response,
        validator_hotkey=validator_hotkey,
        binding_message=binding_message,
        binding_signature_result={
            **output,
            "receipt": binding_receipt,
        },
    )
    evidence = build_stateful_epoch_evidence_v1(
        enclave_response=enclave_response,
        published_bundle=bundle,
    )

    assert set(bundle) == {
        "schema_version",
        "validator_hotkey",
        "binding_message",
        "validator_hotkey_signature",
        "weight_snapshot",
        "weight_result",
        "weights_signature",
        "receipt_graph",
    }
    assert evidence["bundle_hash"].startswith("sha256:")
    assert evidence["epoch_authority"] == enclave_response["epoch_authority"]
    assert evidence["epoch_boundary"] == enclave_response["epoch_boundary"]
    assert evidence["receipt_graph"] == bundle["receipt_graph"]
    with pytest.raises(HostWeightAuthorityV2Error, match="missing its epoch evidence"):
        validate_stateful_epoch_evidence_v1(
            None,
            published_bundle=bundle,
        )
    legacy_journal_body = {
        "schema_version": LEGACY_JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "revision": 0,
        "weight_authorization_id": "sha256:" + "a" * 64,
        "published_bundle": bundle,
        "publication": None,
        "extrinsic_signature_results": [],
        "updated_at": "2026-07-10T20:00:00Z",
    }
    legacy_journal = {
        **legacy_journal_body,
        "journal_hash": sha256_json(legacy_journal_body),
    }
    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="epoch evidence is invalid",
    ):
        validate_publication_journal_v2(legacy_journal)


@pytest.mark.asyncio
async def test_explicit_boundary_capture_deduplicates_identical_snapshot_receipt():
    fixture = _fixture(stateful=True)
    cutover_body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": "0x" + "1" * 64,
        "netuid": 71,
        "cutover_block": 36_000,
        "cutover_block_hash": "0x" + "2" * 64,
        "first_subnet_epoch_index": 35,
        "first_settlement_epoch_id": 100,
        "last_legacy_epoch_id": 99,
    }
    cutover_manifest = {
        **cutover_body,
        "mapping_hash": sha256_json(cutover_body),
    }
    result = fixture["authority"].capture_epoch_boundary(
        {
            "cutover_manifest": cutover_manifest,
            "settlement_epoch_id": 100,
        }
    )
    assert result["schema_version"] == "leadpoet.subnet_epoch_boundary_capture.v1"
    assert result["epoch_boundary"]["current_block"] == 36_000
    assert result["epoch_authority"] == result["epoch_boundary"]
    assert result["epoch_authority_receipt_hash"] == result[
        "epoch_boundary_receipt_hash"
    ]
    receipts = {
        item["receipt_hash"]: item for item in result["receipt_graph"]["receipts"]
    }
    assert len(receipts) == 1
    boundary_receipt = receipts[result["epoch_boundary_receipt_hash"]]
    assert boundary_receipt["output_root"] == sha256_json(
        result["epoch_boundary"]
    )
    assert boundary_receipt["parent_receipt_hashes"] == []
    assert result["receipt_graph"]["root_receipt_hash"] == boundary_receipt[
        "receipt_hash"
    ]
    assert result["source_artifacts"]

    class FakeClient:
        def capture_subnet_epoch_boundary_v2(self, **kwargs):
            assert kwargs == {
                "cutover_manifest": cutover_manifest,
                "settlement_epoch_id": 100,
            }
            return result

    observed_boot_verification = {}

    def verify_capture_boot(
        _identity,
        *,
        expected_pcr0,
        certificate_validity_at_attestation_time=False,
    ):
        observed_boot_verification["expected_pcr0"] = expected_pcr0
        observed_boot_verification["certificate_validity_at_attestation_time"] = (
            certificate_validity_at_attestation_time
        )
        return {"verified": True}

    host_result = capture_subnet_epoch_boundary_v2(
        cutover_manifest=cutover_manifest,
        expected_pcr0=VALIDATOR_PCR0,
        client=FakeClient(),
        boot_verifier=verify_capture_boot,
    )
    assert host_result == result
    assert observed_boot_verification == {
        "expected_pcr0": VALIDATOR_PCR0,
        "certificate_validity_at_attestation_time": True,
    }

    with pytest.raises(
        SubnetEpochBoundaryCaptureV2Error,
        match="approved validator PCR0",
    ):
        capture_subnet_epoch_boundary_v2(
            cutover_manifest=cutover_manifest,
            expected_pcr0=VALIDATOR_PCR0[:-1] + "g",
            client=FakeClient(),
            boot_verifier=verify_capture_boot,
        )

    observed = {}
    wallet = type(
        "Wallet",
        (),
        {
            "hotkey": type(
                "Hotkey",
                (),
                {
                    "ss58_address": VALIDATOR_HOTKEY,
                    "sign": lambda _self, message: (
                        observed.update(signed_message=message.decode("utf-8"))
                        or bytes.fromhex("c" * 128)
                    ),
                },
            )()
        },
    )()

    async def post(url, payload, timeout):
        observed.update(url=url, payload=payload, timeout=timeout)
        unsigned_payload = {
            "schema_version": payload["schema_version"],
            "cutover_manifest": payload["cutover_manifest"],
            "capture": payload["capture"],
        }
        payload_hash = sha256_json(unsigned_payload)
        authorization_hash = sha256_json(
            {
                "validator_hotkey": VALIDATOR_HOTKEY,
                "candidate_payload_hash": payload_hash,
                "validator_hotkey_signature": "0x" + "c" * 128,
            }
        )
        return {
            "schema_version": "leadpoet.subnet_epoch_boundary_candidate_ack.v1",
            "candidate_hash": payload_hash,
            "validator_hotkey": VALIDATOR_HOTKEY,
            "candidate_authorization_hash": authorization_hash,
            "mapping_hash": cutover_manifest["mapping_hash"],
            "subnet_epoch_index": 35,
            "settlement_epoch_id": 100,
            "boundary_block": 36_000,
            "boundary_hash": sha256_json(result["epoch_boundary"]),
            "boundary_receipt_hash": result["epoch_boundary_receipt_hash"],
            "receipt_graph_hash": sha256_json(result["receipt_graph"]),
            "durable_readback_hash": "sha256:" + "b" * 64,
        }

    acknowledgment = await publish_subnet_epoch_boundary_candidate_v1(
        cutover_manifest=cutover_manifest,
        capture=result,
        gateway_url="https://gateway.example",
        wallet=wallet,
        post_json=post,
    )
    assert observed["url"].endswith("/weights/subnet-epoch/candidate/v1")
    assert observed["payload"] == {
        "schema_version": "leadpoet.subnet_epoch_boundary_candidate_submission.v1",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "validator_hotkey_signature": "0x" + "c" * 128,
        "cutover_manifest": cutover_manifest,
        "capture": result,
    }
    unsigned_payload = {
        "schema_version": observed["payload"]["schema_version"],
        "cutover_manifest": cutover_manifest,
        "capture": result,
    }
    assert observed["signed_message"] == (
        build_subnet_epoch_candidate_authorization_message_v1(
            validator_hotkey=VALIDATOR_HOTKEY,
            candidate_payload=unsigned_payload,
        )
    )
    assert acknowledgment["candidate_hash"] == sha256_json(unsigned_payload)

    async def tampered_ack(url, payload, timeout):
        value = await post(url, payload, timeout)
        return {
            **value,
            "candidate_authorization_hash": "sha256:" + "f" * 64,
        }

    with pytest.raises(
        SubnetEpochBoundaryCaptureV2Error,
        match="acknowledgment differs",
    ):
        await publish_subnet_epoch_boundary_candidate_v1(
            cutover_manifest=cutover_manifest,
            capture=result,
            gateway_url="https://gateway.example",
            wallet=wallet,
            post_json=tampered_ack,
        )


def test_candidate_cli_writes_exact_signed_body_without_a_gateway(
    monkeypatch,
    tmp_path,
    capsys,
):
    cutover_body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": "0x" + "1" * 64,
        "netuid": 71,
        "cutover_block": 36_000,
        "cutover_block_hash": "0x" + "2" * 64,
        "first_subnet_epoch_index": 35,
        "first_settlement_epoch_id": 100,
        "last_legacy_epoch_id": 99,
    }
    manifest = {**cutover_body, "mapping_hash": sha256_json(cutover_body)}
    manifest_path = tmp_path / "cutover.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_path = tmp_path / "handoff" / "candidate.json"
    capture = {
        "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1",
        "offline_fixture": True,
    }
    wallet = type(
        "Wallet",
        (),
        {
            "hotkey": type(
                "Hotkey",
                (),
                {
                    "ss58_address": VALIDATOR_HOTKEY,
                    "sign": lambda _self, _message: bytes.fromhex("c" * 128),
                },
            )()
        },
    )()
    observed = {}

    def fake_capture(**kwargs):
        observed.update(kwargs)
        return capture

    monkeypatch.setattr(
        capture_module,
        "_approved_validator_pcr0",
        lambda _path: VALIDATOR_PCR0,
    )
    monkeypatch.setattr(
        capture_module,
        "capture_subnet_epoch_boundary_v2",
        fake_capture,
    )
    from validator_tee.host import enclave_hotkey_v2

    monkeypatch.setattr(
        enclave_hotkey_v2,
        "build_enclave_backed_wallet_v2",
        lambda **_kwargs: wallet,
    )
    argv = [
        "--cutover-manifest",
        str(manifest_path),
        "--validator-release-manifest",
        str(tmp_path / "release.json"),
        "--wallet-name",
        "validator_72",
        "--wallet-hotkey",
        "default",
        "--candidate-output",
        str(candidate_path),
    ]
    assert capture_module.main(argv) == 0
    capsys.readouterr()
    signed = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert signed == capture_module.build_signed_subnet_epoch_candidate_submission_v1(
        cutover_manifest=manifest,
        capture=capture,
        wallet=wallet,
    )
    assert candidate_path.stat().st_mode & 0o777 == 0o600
    assert observed["expected_pcr0"] == VALIDATOR_PCR0

    with pytest.raises(
        SubnetEpochBoundaryCaptureV2Error,
        match="explicit overwrite",
    ):
        capture_module.main(argv)
    assert capture_module.main(
        [*argv, "--overwrite-candidate-output"]
    ) == 0
    capsys.readouterr()
    assert candidate_path.stat().st_mode & 0o777 == 0o600


def test_candidate_cli_requires_enclave_wallet_for_offline_output(tmp_path):
    with pytest.raises(SystemExit):
        capture_module.main(
            [
                "--cutover-manifest",
                str(tmp_path / "cutover.json"),
                "--candidate-output",
                str(tmp_path / "candidate.json"),
            ]
        )


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
    bad_expectations[GATEWAY_COMMIT]["roles"]["gateway_coordinator"][
        "build_manifest_hash"
    ] = (
        "sha256:" + "9" * 64
    )
    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: fixture["validator_boot"],
        gateway_release_lineage_supplier=lambda: bad_expectations,
        sign_digest=fixture["validator_key"].sign,
        chain_source=fixture["chain_source"],
        boot_verifier=lambda identity, **_kwargs: {"verified": True},
        clock=lambda: NOW,
    )
    with pytest.raises(ValidatorWeightAuthorityV2Error, match="boot differs"):
        authority.compute(fixture["request"])


def test_validator_authority_accepts_only_exact_historical_release_lineage_boot():
    fixture = _fixture()
    historical_commit = "f" * 40
    historical_key = Ed25519PrivateKey.generate()
    historical_pub = historical_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    historical_boot = _boot(
        role=COORDINATOR_ROLE,
        physical_role="gateway_coordinator",
        commit=historical_commit,
        pcr0="a" * 96,
        manifest="sha256:" + "b" * 64,
        dependency_lock="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        private_key=historical_key,
        public_key=historical_pub,
        nonce="9",
    )
    fixture["expectations"][historical_commit] = {
        "roles": {
            "gateway_coordinator": {
                "commit_sha": historical_commit,
                "pcr0": historical_boot["pcr0"],
                "build_manifest_hash": historical_boot["build_manifest_hash"],
                "dependency_lock_hash": historical_boot["dependency_lock_hash"],
            }
        }
    }

    verified = fixture["authority"]._verify_one_boot(
        historical_boot,
        fixture["validator_boot"],
    )
    assert verified == {"verified": True}
    assert fixture["boot_verification_modes"][
        historical_boot["boot_identity_hash"]
    ] is True

    changed = copy.deepcopy(historical_boot)
    changed["dependency_lock_hash"] = "sha256:" + "0" * 64
    with pytest.raises(
        ValidatorWeightAuthorityV2Error,
        match="approved release lineage",
    ):
        fixture["authority"]._verify_one_boot(
            changed,
            fixture["validator_boot"],
        )

    unknown = copy.deepcopy(historical_boot)
    unknown["commit_sha"] = "0" * 40
    with pytest.raises(
        ValidatorWeightAuthorityV2Error,
        match="commit is not in approved release lineage",
    ):
        fixture["authority"]._verify_one_boot(
            unknown,
            fixture["validator_boot"],
        )


def test_validator_authority_accepts_exact_historical_validator_boot():
    fixture = _fixture()
    historical_commit = "f" * 40
    historical_key = Ed25519PrivateKey.generate()
    historical_pub = historical_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    historical_boot = _boot(
        role=WEIGHT_ROLE,
        physical_role="validator_weights",
        commit=historical_commit,
        pcr0="a" * 96,
        manifest="sha256:" + "b" * 64,
        dependency_lock="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        private_key=historical_key,
        public_key=historical_pub,
        nonce="9",
    )
    fixture["expectations"][historical_commit] = {
        "roles": {
            "validator_weights": {
                "commit_sha": historical_commit,
                "pcr0": historical_boot["pcr0"],
                "build_manifest_hash": historical_boot["build_manifest_hash"],
                "dependency_lock_hash": historical_boot[
                    "dependency_lock_hash"
                ],
            }
        }
    }

    verified = fixture["authority"]._verify_one_boot(
        historical_boot,
        fixture["validator_boot"],
    )
    assert verified == {"verified": True}
    assert fixture["boot_verification_modes"][
        historical_boot["boot_identity_hash"]
    ] is True

    changed = copy.deepcopy(historical_boot)
    changed["pcr0"] = "0" * 96
    with pytest.raises(
        ValidatorWeightAuthorityV2Error,
        match="boot differs from approved release lineage",
    ):
        fixture["authority"]._verify_one_boot(
            changed,
            fixture["validator_boot"],
        )


def test_validator_authority_fails_when_nitro_verifier_rejects_boot():
    fixture = _fixture()

    def reject_boot(identity, **kwargs):
        del identity, kwargs
        raise ValueError("Nitro signature invalid")

    authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: fixture["validator_boot"],
        gateway_release_lineage_supplier=lambda: fixture["expectations"],
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
