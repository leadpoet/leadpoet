from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    WEIGHT_ROLE,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.auditor_v2 import (
    AuditorV2Error,
    IDENTITY_CACHE_SCHEMA_VERSION,
    verify_attested_weight_authority_v2,
    verify_attested_weight_bundle_v2,
)
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    VALIDATOR_WEIGHT_INPUT_CATEGORIES,
    PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    WEIGHT_INPUT_PURPOSES,
    WeightAuthorityV2Error,
    build_weight_snapshot_v2,
    validate_weight_input_source_evidence_v2,
    validate_weight_finalization_submission_v2,
    validate_published_weight_bundle_v2,
    weight_input_output_roots_v2,
    weight_input_value_documents_v2,
    gateway_weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)
from validator_tee.host.weight_authority_v2 import (
    HostWeightAuthorityV2Error,
    build_authoritative_weight_bundle_v2,
)


HASH = "sha256:" + "1" * 64
HASH_B = "sha256:" + "2" * 64
COMMIT = "3" * 40
PCR0 = "4" * 96
NOW = "2026-07-10T20:00:00Z"
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


def _keypair():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return private_key, public_key


def _boot(role, private_key, public_key, config_hash):
    body = build_boot_identity_body(
        role=role,
        physical_role=(
            "gateway_coordinator"
            if role == COORDINATOR_ROLE
            else "validator_weights"
        ),
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH,
        dependency_lock_hash=HASH_B,
        config_hash=config_hash,
        boot_nonce=("5" if role == COORDINATOR_ROLE else "6") * 32,
        signing_pubkey=public_key,
        transport_pubkey=("7" if role == COORDINATOR_ROLE else "8") * 64,
        transport_certificate_hash=HASH_B,
        attestation_user_data_hash=HASH,
        issued_at=NOW,
    )
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(
            ("attestation:" + role).encode("ascii")
        ).decode("ascii"),
    )


def _receipt(
    *,
    role,
    purpose,
    job_id,
    private_key,
    public_key,
    boot,
    config_hash,
    input_root=HASH,
    output_root=HASH_B,
    parents=(),
    sequence=1,
    transport_root=EMPTY_TRANSPORT_ROOT,
    artifact_root=EMPTY_ARTIFACT_ROOT,
):
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=100,
        sequence=sequence,
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH,
        dependency_lock_hash=HASH_B,
        config_hash=config_hash,
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=transport_root,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=parents,
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _source_attempt(*, category, job_id, purpose, sequence, provider_id, host, method):
    return build_transport_attempt(
        request_id=("%032x" % (sequence + 1)),
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
        started_at=NOW,
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
        completed_at=NOW,
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
        "leaderboard_entries": [{"miner_hotkey": "fulfillment-hotkey", "wins": 9}],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [{"hotkey": "fulfillment-hotkey", "share": 0.705}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125_000,
        "min_total_rep_for_distribution": 100,
    }
    snapshot["config_hash"] = weight_config_hash(snapshot)
    return snapshot


def test_gateway_weight_inputs_are_epoch_scoped_and_finalized_block_invariant():
    first = _calculation_snapshot([], "")
    second = copy.deepcopy(first)
    second["block"] += 1
    event_hash = sha256_json({"epoch": first["epoch_id"]})

    first_gateway = gateway_weight_input_value_documents_v2(
        calculation_snapshot=first,
        gateway_authority_event_hash=event_hash,
    )
    second_gateway = gateway_weight_input_value_documents_v2(
        calculation_snapshot=second,
        gateway_authority_event_hash=event_hash,
    )

    assert set(first_gateway) == set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    assert first_gateway == second_gateway
    assert all("block" not in document for document in first_gateway.values())

    first_all = weight_input_value_documents_v2(
        calculation_snapshot=first,
        finalized_chain_state_root=sha256_json({"block": first["block"]}),
        gateway_authority_event_hash=event_hash,
    )
    second_all = weight_input_value_documents_v2(
        calculation_snapshot=second,
        finalized_chain_state_root=sha256_json({"block": second["block"]}),
        gateway_authority_event_hash=event_hash,
    )
    assert {
        category: first_all[category]
        for category in GATEWAY_WEIGHT_INPUT_CATEGORIES
    } == first_gateway
    assert all(
        first_all[category] != second_all[category]
        and first_all[category]["block"] == first["block"]
        and second_all[category]["block"] == second["block"]
        for category in VALIDATOR_WEIGHT_INPUT_CATEGORIES
    )


def _bundle(*, category_purpose_override=None, category_output_override=None):
    coordinator_key, coordinator_pub = _keypair()
    weight_key, weight_pub = _keypair()
    preliminary = _calculation_snapshot([], "")
    weight_config = preliminary["config_hash"]
    coordinator_boot = _boot(COORDINATOR_ROLE, coordinator_key, coordinator_pub, HASH)
    weight_boot = _boot(WEIGHT_ROLE, weight_key, weight_pub, weight_config)
    finalized_chain_state_root = sha256_json({"block": preliminary["block"]})
    gateway_authority_event_hash = sha256_json(
        {"epoch": preliminary["epoch_id"]}
    )
    expected_output_roots = weight_input_output_roots_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )

    source_receipts = []
    source_attempts = []
    input_hashes = {}
    ordered_categories = ["chain_state", "metagraph_state", "burn_ownership"] + sorted(
        set(WEIGHT_INPUT_PURPOSES) - {"chain_state", "metagraph_state", "burn_ownership"}
    )
    for index, category in enumerate(ordered_categories):
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        if category_purpose_override and category in category_purpose_override:
            role, purpose = category_purpose_override[category]
        if role == COORDINATOR_ROLE:
            key, pub, boot, config = (
                coordinator_key,
                coordinator_pub,
                coordinator_boot,
                HASH,
            )
        else:
            key, pub, boot, config = weight_key, weight_pub, weight_boot, weight_config
        job_id = "weight-input-%s" % category
        attempt = None
        if role == COORDINATOR_ROLE and category != "anomaly_adjustments":
            attempt = _source_attempt(
                category=category,
                job_id=job_id,
                purpose=purpose,
                sequence=index,
                provider_id="supabase",
                host="qplwoislplkcegvdmbim.supabase.co",
                method="GET",
            )
        elif category in {"chain_state", "metagraph_state"}:
            attempt = _source_attempt(
                category=category,
                job_id=job_id,
                purpose=purpose,
                sequence=index,
                provider_id="bittensor_chain",
                host="entrypoint-finney.opentensor.ai",
                method="WSS",
            )
        artifact_hashes = [
            sha256_json(
                weight_input_value_documents_v2(
                    calculation_snapshot=preliminary,
                    finalized_chain_state_root=finalized_chain_state_root,
                    gateway_authority_event_hash=gateway_authority_event_hash,
                )[category]["value"]
            )
        ]
        if attempt is not None:
            source_attempts.append(attempt)
            artifact_hashes.extend(
                [
                    attempt["request_artifact_hash"],
                    attempt["response_artifact_hash"],
                ]
            )
        receipt = _receipt(
            role=role,
            purpose=purpose,
            job_id=job_id,
            private_key=key,
            public_key=pub,
            boot=boot,
            config_hash=config,
            input_root=sha256_json({"category": category, "kind": "input"}),
            output_root=(
                category_output_override[category]
                if category_output_override
                and category in category_output_override
                else expected_output_roots[category]
            ),
            sequence=index,
            parents=(
                [input_hashes["chain_state"]]
                if category == "metagraph_state"
                else [input_hashes["metagraph_state"]]
                if category == "burn_ownership"
                else []
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
        source_receipts.append(receipt)
        input_hashes[category] = receipt["receipt_hash"]

    calculation = _calculation_snapshot(
        list(input_hashes.values()),
        input_hashes["research_lab_allocation"],
    )
    snapshot = build_weight_snapshot_v2(
        validator_hotkey=VALIDATOR_HOTKEY,
        calculation_snapshot=calculation,
        input_receipt_hashes=input_hashes,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=gateway_authority_event_hash,
    )
    snapshot_receipt = _receipt(
        role=WEIGHT_ROLE,
        purpose="validator.weight_snapshot.v2",
        job_id="weight-snapshot-100",
        private_key=weight_key,
        public_key=weight_pub,
        boot=weight_boot,
        config_hash=weight_config,
        input_root=snapshot["source_input_root"],
        output_root=snapshot["snapshot_hash"],
        parents=sorted(input_hashes.values()),
        sequence=100,
    )
    result = compute_final_weights(calculation)
    weight_receipt = _receipt(
        role=WEIGHT_ROLE,
        purpose="validator.weights.computed.v2",
        job_id="weight-computation-100",
        private_key=weight_key,
        public_key=weight_pub,
        boot=weight_boot,
        config_hash=weight_config,
        input_root=snapshot["snapshot_hash"],
        output_root=sha256_json(result),
        parents=[snapshot_receipt["receipt_hash"]],
        sequence=101,
    )
    binding_message = create_binding_message(
        netuid=71,
        chain="wss://entrypoint-finney.opentensor.ai:443",
        enclave_pubkey=weight_pub,
        validator_code_hash=weight_boot["build_manifest_hash"],
        version=COMMIT,
    )
    hotkey_signature = "9" * 128
    application_request = build_application_signature_request_v2(
        message=binding_message.encode(),
        validator_hotkey=VALIDATOR_HOTKEY,
        boot_identity_hash=weight_boot["boot_identity_hash"],
    )
    binding_output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": application_request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "signature": hotkey_signature,
    }
    hotkey_receipt = _receipt(
        role=WEIGHT_ROLE,
        purpose="validator.hotkey_signature.v2",
        job_id="hotkey-signature-100",
        private_key=weight_key,
        public_key=weight_pub,
        boot=weight_boot,
        config_hash=weight_config,
        input_root=application_request["request_hash"],
        output_root=sha256_json(binding_output),
        parents=[weight_receipt["receipt_hash"]],
        sequence=102,
    )
    graph = build_receipt_graph(
        root_receipt_hash=hotkey_receipt["receipt_hash"],
        boot_identities=[coordinator_boot, weight_boot],
        receipts=[
            *source_receipts,
            snapshot_receipt,
            weight_receipt,
            hotkey_receipt,
        ],
        transport_attempts=source_attempts,
    )
    return {
        "schema_version": PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "binding_message": binding_message,
        "validator_hotkey_signature": hotkey_signature,
        "weight_snapshot": snapshot,
        "weight_result": result,
        "weights_signature": weight_key.sign(bytes.fromhex(result["weights_hash"])).hex(),
        "receipt_graph": graph,
    }


def test_complete_v2_weight_authority_graph_validates():
    verified = validate_published_weight_bundle_v2(_bundle())
    assert verified["netuid"] == 71
    assert verified["epoch_id"] == 100
    assert verified["weights_hash"]
    assert verified["root_receipt_hash"]
    assert verified["snapshot_receipt_hash"]


def test_host_composes_binding_receipt_as_authoritative_graph_root():
    expected = _bundle()
    bound_graph = expected["receipt_graph"]
    binding_receipt = next(
        receipt
        for receipt in bound_graph["receipts"]
        if receipt["purpose"] == "validator.hotkey_signature.v2"
    )
    computed_root = binding_receipt["parent_receipt_hashes"][0]
    unbound_graph = build_receipt_graph(
        root_receipt_hash=computed_root,
        boot_identities=bound_graph["boot_identities"],
        receipts=[
            receipt
            for receipt in bound_graph["receipts"]
            if receipt["receipt_hash"] != binding_receipt["receipt_hash"]
        ],
        transport_attempts=bound_graph["transport_attempts"],
        host_operations=bound_graph["host_operations"],
    )
    response = {
        "weight_snapshot": expected["weight_snapshot"],
        "weight_result": expected["weight_result"],
        "weights_signature": expected["weights_signature"],
        "receipt_graph": unbound_graph,
        "boot_identity": next(
            boot
            for boot in bound_graph["boot_identities"]
            if boot["role"] == WEIGHT_ROLE
        ),
        "weight_authorization_id": "sha256:" + "a" * 64,
        "source_artifacts": [],
    }
    built = build_authoritative_weight_bundle_v2(
        enclave_response=response,
        validator_hotkey=expected["validator_hotkey"],
        binding_message=expected["binding_message"],
        binding_signature_result={
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": expected["validator_hotkey"],
            "signature": expected["validator_hotkey_signature"],
            "receipt": binding_receipt,
        },
    )
    assert built == expected
    with pytest.raises(HostWeightAuthorityV2Error, match="receipt is missing"):
        build_authoritative_weight_bundle_v2(
            enclave_response=response,
            validator_hotkey=expected["validator_hotkey"],
            binding_message=expected["binding_message"],
            binding_signature_result={
                "purpose": "validator.gateway_binding.v2",
                "validator_hotkey": expected["validator_hotkey"],
                "signature": expected["validator_hotkey_signature"],
            },
        )


def test_v2_weight_snapshot_requires_every_input_category():
    bundle = _bundle()
    snapshot = bundle["weight_snapshot"]
    inputs = dict(snapshot["input_receipt_hashes"])
    del inputs["bans"]
    with pytest.raises(WeightAuthorityV2Error, match="categories are incomplete"):
        build_weight_snapshot_v2(
            validator_hotkey=snapshot["validator_hotkey"],
            calculation_snapshot=snapshot["calculation_snapshot"],
            input_receipt_hashes=inputs,
            finalized_chain_state_root=snapshot["finalized_chain_state_root"],
            gateway_authority_event_hash=snapshot["gateway_authority_event_hash"],
        )


def test_v2_weight_bundle_recomputes_instead_of_trusting_claimed_result():
    bundle = _bundle()
    bundle["weight_result"] = copy.deepcopy(bundle["weight_result"])
    bundle["weight_result"]["sparse_weights_u16"][0] -= 1
    with pytest.raises(WeightAuthorityV2Error, match="canonical computation"):
        validate_published_weight_bundle_v2(bundle)


def test_v2_weight_bundle_rejects_blind_or_wrong_signature():
    bundle = _bundle()
    bundle["weights_signature"] = "a" * 128
    with pytest.raises(WeightAuthorityV2Error, match="weight signature"):
        validate_published_weight_bundle_v2(bundle)


def test_v2_weight_bundle_requires_exact_semantic_input_purpose():
    with pytest.raises(WeightAuthorityV2Error, match="missing required purpose"):
        validate_published_weight_bundle_v2(
            _bundle(
                category_purpose_override={
                    "champions": (
                        COORDINATOR_ROLE,
                        "research_lab.reimbursement_input.v2",
                    )
                }
            )
        )


def test_v2_weight_bundle_rejects_semantic_receipt_with_unrelated_output():
    with pytest.raises(
        WeightAuthorityV2Error,
        match="bans receipt output does not bind its weight input",
    ):
        validate_published_weight_bundle_v2(
            _bundle(category_output_override={"bans": HASH})
        )


def _source_evidence_fixture(category):
    bundle = _bundle()
    snapshot = bundle["weight_snapshot"]
    graph = bundle["receipt_graph"]
    receipt_hash = snapshot["input_receipt_hashes"][category]
    receipt = next(
        item for item in graph["receipts"] if item["receipt_hash"] == receipt_hash
    )
    documents = weight_input_value_documents_v2(
        calculation_snapshot=snapshot["calculation_snapshot"],
        finalized_chain_state_root=snapshot["finalized_chain_state_root"],
        gateway_authority_event_hash=snapshot["gateway_authority_event_hash"],
    )
    return receipt, documents[category], graph["transport_attempts"]


def test_v2_weight_input_rejects_missing_authenticated_source_attempt():
    receipt, document, _attempts = _source_evidence_fixture("bans")
    with pytest.raises(WeightAuthorityV2Error, match="no authenticated database read"):
        validate_weight_input_source_evidence_v2(
            category="bans",
            receipt=receipt,
            document=document,
            transport_attempts=[],
        )


def test_v2_weight_input_rejects_wrong_supabase_project():
    receipt, document, attempts = _source_evidence_fixture("bans")
    source = next(
        item
        for item in attempts
        if item["job_id"] == receipt["job_id"]
        and item["purpose"] == receipt["purpose"]
    )
    wrong = build_transport_attempt(
        request_id=source["request_id"],
        logical_operation_id=source["logical_operation_id"],
        job_id=source["job_id"],
        purpose=source["purpose"],
        provider_id=source["provider_id"],
        attempt_number=source["attempt_number"],
        method=source["method"],
        destination_host="attacker.example.com",
        destination_port=source["destination_port"],
        path_hash=source["path_hash"],
        nonsecret_headers_hash=source["nonsecret_headers_hash"],
        body_hash=source["body_hash"],
        credential_ref_hash=source["credential_ref_hash"],
        retry_policy_hash=source["retry_policy_hash"],
        timeout_ms=source["timeout_ms"],
        started_at=source["started_at"],
        terminal_status=source["terminal_status"],
        http_status=source["http_status"],
        response_hash=source["response_hash"],
        request_artifact_hash=source["request_artifact_hash"],
        response_artifact_hash=source["response_artifact_hash"],
        tls_peer_chain_hash=source["tls_peer_chain_hash"],
        tls_protocol=source["tls_protocol"],
        failure_code=source["failure_code"],
        completed_at=source["completed_at"],
    )
    with pytest.raises(WeightAuthorityV2Error, match="wrong Supabase project"):
        validate_weight_input_source_evidence_v2(
            category="bans",
            receipt=receipt,
            document=document,
            transport_attempts=[wrong],
        )


def test_v2_weight_input_rejects_transport_failure_as_provider_response():
    receipt, document, attempts = _source_evidence_fixture("bans")
    source = next(
        item
        for item in attempts
        if item["job_id"] == receipt["job_id"]
        and item["purpose"] == receipt["purpose"]
    )
    failed = build_transport_attempt(
        request_id=source["request_id"],
        logical_operation_id=source["logical_operation_id"],
        job_id=source["job_id"],
        purpose=source["purpose"],
        provider_id=source["provider_id"],
        attempt_number=source["attempt_number"],
        method=source["method"],
        destination_host=source["destination_host"],
        destination_port=source["destination_port"],
        path_hash=source["path_hash"],
        nonsecret_headers_hash=source["nonsecret_headers_hash"],
        body_hash=source["body_hash"],
        credential_ref_hash=source["credential_ref_hash"],
        retry_policy_hash=source["retry_policy_hash"],
        timeout_ms=source["timeout_ms"],
        started_at=source["started_at"],
        terminal_status="transport_failure",
        http_status=None,
        response_hash=None,
        request_artifact_hash=source["request_artifact_hash"],
        response_artifact_hash=None,
        tls_peer_chain_hash=None,
        tls_protocol=None,
        failure_code="host_dropped",
        completed_at=source["completed_at"],
    )
    with pytest.raises(
        WeightAuthorityV2Error,
        match="no successful authenticated database response",
    ):
        validate_weight_input_source_evidence_v2(
            category="bans",
            receipt=receipt,
            document=document,
            transport_attempts=[failed],
        )


def test_v2_weight_input_accepts_authenticated_error_followed_by_success():
    receipt, document, attempts = _source_evidence_fixture("bans")
    success = next(
        item
        for item in attempts
        if item["job_id"] == receipt["job_id"]
        and item["purpose"] == receipt["purpose"]
    )
    provider_error = build_transport_attempt(
        request_id="f" * 32,
        logical_operation_id=success["logical_operation_id"] + "-prior",
        job_id=success["job_id"],
        purpose=success["purpose"],
        provider_id=success["provider_id"],
        attempt_number=0,
        method=success["method"],
        destination_host=success["destination_host"],
        destination_port=success["destination_port"],
        path_hash=success["path_hash"],
        nonsecret_headers_hash=success["nonsecret_headers_hash"],
        body_hash=success["body_hash"],
        credential_ref_hash=success["credential_ref_hash"],
        retry_policy_hash=success["retry_policy_hash"],
        timeout_ms=success["timeout_ms"],
        started_at=success["started_at"],
        terminal_status="authenticated_response",
        http_status=503,
        response_hash=HASH_B,
        request_artifact_hash=HASH_B,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=success["tls_peer_chain_hash"],
        tls_protocol=success["tls_protocol"],
        failure_code=None,
        completed_at=success["completed_at"],
    )
    scoped = [provider_error, success]
    retried_receipt = dict(receipt)
    retried_receipt["transport_root"] = merkle_root(
        [item["attempt_hash"] for item in scoped],
        domain="leadpoet-transport-v2",
    )
    retried_receipt["artifact_root"] = merkle_root(
        [
            sha256_json(document["value"]),
            provider_error["request_artifact_hash"],
            provider_error["response_artifact_hash"],
            success["request_artifact_hash"],
            success["response_artifact_hash"],
        ],
        domain="leadpoet-artifact-v2",
    )
    validate_weight_input_source_evidence_v2(
        category="bans",
        receipt=retried_receipt,
        document=document,
        transport_attempts=scoped,
    )


def test_v2_weight_input_rejects_incomplete_source_artifact_commitment():
    receipt, document, attempts = _source_evidence_fixture("bans")
    incomplete = dict(receipt)
    incomplete["artifact_root"] = merkle_root(
        [sha256_json(document["value"])],
        domain="leadpoet-artifact-v2",
    )
    with pytest.raises(WeightAuthorityV2Error, match="artifact evidence is incomplete"):
        validate_weight_input_source_evidence_v2(
            category="bans",
            receipt=incomplete,
            document=document,
            transport_attempts=attempts,
        )


def test_v2_weight_bundle_rejects_snapshot_receipt_that_omits_an_input():
    bundle = _bundle()
    graph = copy.deepcopy(bundle["receipt_graph"])
    snapshot_receipt = next(
        receipt
        for receipt in graph["receipts"]
        if receipt["purpose"] == "validator.weight_snapshot.v2"
    )
    snapshot_receipt["parent_receipt_hashes"] = snapshot_receipt["parent_receipt_hashes"][:-1]
    bundle["receipt_graph"] = graph
    with pytest.raises(WeightAuthorityV2Error, match="invalid V2 receipt graph"):
        validate_published_weight_bundle_v2(bundle)


def test_v2_weight_bundle_hash_commits_binding_and_complete_graph():
    bundle = _bundle()
    bundle["binding_message"] += "|tampered=true"
    with pytest.raises(WeightAuthorityV2Error, match="hotkey binding receipt"):
        validate_published_weight_bundle_v2(bundle)


def _identity_cache(bundle):
    return {
        "schema_version": IDENTITY_CACHE_SCHEMA_VERSION,
        "entries": [
            {
                "physical_role": boot["physical_role"],
                "role": boot["role"],
                "commit_sha": boot["commit_sha"],
                "pcr0": boot["pcr0"],
                "build_manifest_hash": boot["build_manifest_hash"],
                "dependency_lock_hash": boot["dependency_lock_hash"],
                "verified_build_count": 3,
            }
            for boot in bundle["receipt_graph"]["boot_identities"]
        ],
    }


def test_auditor_v2_verifies_every_boot_against_independent_builds():
    bundle = _bundle()
    observed = []

    def verify_boot(boot, *, expected_pcr0=None):
        assert boot["pcr0"] == expected_pcr0
        observed.append(boot["boot_identity_hash"])
        return {"verified": True}

    verified = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=_identity_cache(bundle),
        boot_verifier=verify_boot,
    )
    assert set(observed) == {
        boot["boot_identity_hash"]
        for boot in bundle["receipt_graph"]["boot_identities"]
    }
    assert len(verified["independent_receipt_identities"]) == 2


def test_auditor_v2_rejects_independent_manifest_mismatch():
    bundle = _bundle()
    cache = _identity_cache(bundle)
    cache["entries"][0]["build_manifest_hash"] = "sha256:" + "9" * 64
    with pytest.raises(AuditorV2Error, match="manifest differs"):
        verify_attested_weight_bundle_v2(
            bundle,
            identity_cache=cache,
            boot_verifier=lambda boot, expected_pcr0=None: {"verified": True},
        )


def _chain_profile():
    return {
        "schema_version": "leadpoet.chain_signing_profile.v2",
        "network": "finney",
        "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443",
        "genesis_hash": "0" * 64,
        "spec_version": 432,
        "transaction_version": 1,
        "version_key": 10005000,
        "commit_call_index": "0776",
        "serve_axon_call_index": "0704",
        "commit_reveal_version": 4,
        "mechid": 0,
        "tempo": 360,
        "subnet_reveal_period_epochs": 1,
        "block_time_millis": 12000,
        "max_snapshot_block_drift": 64,
        "extrinsic_period": 8,
        "signed_extensions": [
            "CheckMortality",
            "CheckNonce",
            "ChargeTransactionPayment",
            "CheckMetadataHash",
            "CheckSpecVersion",
            "CheckTxVersion",
            "CheckGenesis",
            "CheckMortalityAdditionalSigned",
            "CheckMetadataHashAdditionalSigned",
        ],
    }


def test_auditor_v2_requires_publication_and_finalized_state_transition():
    bundle = _bundle()
    bundle_verified = validate_published_weight_bundle_v2(bundle)
    coordinator_key, coordinator_pub = _keypair()
    coordinator_boot = _boot(
        COORDINATOR_ROLE, coordinator_key, coordinator_pub, HASH
    )
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": bundle_verified["bundle_hash"],
        "root_receipt_hash": bundle_verified["root_receipt_hash"],
        "durable_readback_hash": sha256_json({"durable": True}),
        "transparency_event_hash": sha256_json({"transparency": True}),
    }
    publication_receipt = _receipt(
        role=COORDINATOR_ROLE,
        purpose="gateway.weights.publication.v2",
        job_id="weight-publication-100",
        private_key=coordinator_key,
        public_key=coordinator_pub,
        boot=coordinator_boot,
        config_hash=HASH,
        input_root=sha256_json({"publication": "input"}),
        output_root=sha256_json(publication_doc),
        parents=[bundle_verified["root_receipt_hash"]],
        sequence=200,
    )
    publication_graph = build_receipt_graph(
        root_receipt_hash=publication_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"]
        + [coordinator_boot],
        receipts=bundle["receipt_graph"]["receipts"] + [publication_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": bundle_verified["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc[
                "transparency_event_hash"
            ],
            "durable_readback_hash": publication_doc["durable_readback_hash"],
        }
    )

    weight_key, weight_pub = _keypair()
    weight_config = next(
        boot["config_hash"]
        for boot in bundle["receipt_graph"]["boot_identities"]
        if boot["role"] == WEIGHT_ROLE
    )
    finalization_boot = _boot(
        WEIGHT_ROLE, weight_key, weight_pub, weight_config
    )
    commitment = b"measured-commitment"
    authorization = build_weight_extrinsic_authorization_v2(
        profile=_chain_profile(),
        validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex="aa" * 32,
        epoch_id=bundle_verified["epoch_id"],
        netuid=bundle_verified["netuid"],
        weight_receipt_hash=bundle_verified["weight_receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash=bundle_verified["weights_hash"],
        sparse_uids=bundle_verified["uids"],
        sparse_weights_u16=bundle_verified["weights_u16"],
        commitment=commitment,
        reveal_round=998877,
        era_current=36099,
        nonce=7,
        block_hash="b" * 64,
    )
    extrinsic_signature = "cc" * 64
    extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex="aa" * 32,
        signature_hex=extrinsic_signature,
        era_period=authorization["era_period"],
        era_current=authorization["era_current"],
        nonce=authorization["nonce"],
        call_data_hex=authorization["call_data_hex"],
    )
    extrinsic_hash = signed_extrinsic_hash_v2(extrinsic)
    extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": VALIDATOR_HOTKEY,
        "signature": extrinsic_signature,
        "extrinsic_hash": extrinsic_hash,
    }
    extrinsic_receipt = _receipt(
        role=WEIGHT_ROLE,
        purpose="validator.set_weights_extrinsic.v2",
        job_id="set-weights-100",
        private_key=weight_key,
        public_key=weight_pub,
        boot=finalization_boot,
        config_hash=weight_config,
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[bundle_verified["weight_receipt_hash"]],
        sequence=201,
    )
    finalization_job = "weight-finalization-100"
    finalization_attempt = _source_attempt(
        category="weight-finalization",
        job_id=finalization_job,
        purpose="validator.weights.finalized.v2",
        sequence=300,
        provider_id="bittensor_chain",
        host="entrypoint-finney.opentensor.ai",
        method="POST",
    )
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "netuid": bundle_verified["netuid"],
        "epoch_id": bundle_verified["epoch_id"],
        "weights_hash": bundle_verified["weights_hash"],
        "weight_receipt_hash": bundle_verified["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": extrinsic_signature,
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": 36105,
        "finalized_block_hash": "d" * 64,
        "state_transition_hash": sha256_json({"state": "committed"}),
    }
    finalization_receipt = _receipt(
        role=WEIGHT_ROLE,
        purpose="validator.weights.finalized.v2",
        job_id=finalization_job,
        private_key=weight_key,
        public_key=weight_pub,
        boot=finalization_boot,
        config_hash=weight_config,
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [
                    extrinsic_receipt["receipt_hash"]
                ],
            }
        ),
        output_root=sha256_json(finalization_doc),
        parents=[extrinsic_receipt["receipt_hash"]],
        sequence=202,
        transport_root=merkle_root(
            [finalization_attempt["attempt_hash"]],
            domain="leadpoet-transport-v2",
        ),
        artifact_root=merkle_root(
            [
                finalization_attempt["request_artifact_hash"],
                finalization_attempt["response_artifact_hash"],
            ],
            domain="leadpoet-artifact-v2",
        ),
    )
    finalization_receipts = [
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt["purpose"] != "validator.hotkey_signature.v2"
    ] + [extrinsic_receipt, finalization_receipt]
    finalization_graph = build_receipt_graph(
        root_receipt_hash=finalization_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"]
        + [finalization_boot],
        receipts=finalization_receipts,
        transport_attempts=bundle["receipt_graph"]["transport_attempts"]
        + [finalization_attempt],
    )
    finalization_submission = {
        "schema_version": "leadpoet.weight_finalization_submission.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_submission_event_hash": submission_event_hash,
        "finalization": finalization_doc,
        "receipt_graph": finalization_graph,
    }
    verified_finalization = validate_weight_finalization_submission_v2(
        finalization_submission,
        chain_signing_profile=_chain_profile(),
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": bundle_verified["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": authorization[
                "authorization_hash"
            ],
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc[
                "finalized_block_hash"
            ],
            "state_transition_hash": finalization_doc[
                "state_transition_hash"
            ],
        }
    )
    authority = {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": bundle,
        "publication": {
            "weight_submission_event_hash": submission_event_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "publication_doc": publication_doc,
            "receipt_graph": publication_graph,
        },
        "finalization": {
            "weight_finalization_event_hash": finalization_event_hash,
            "submission": finalization_submission,
        },
    }
    verified = verify_attested_weight_authority_v2(
        authority,
        identity_cache=_identity_cache(bundle),
        chain_signing_profile=_chain_profile(),
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "pcr0": expected_pcr0
        },
    )
    assert verified["extrinsic_hash"] == extrinsic_hash
    assert verified["weight_finalization_event_hash"] == finalization_event_hash

    tampered = copy.deepcopy(authority)
    tampered["finalization"]["submission"]["finalization"][
        "state_transition_hash"
    ] = sha256_json({"state": "tampered"})
    with pytest.raises((AuditorV2Error, WeightAuthorityV2Error)):
        verify_attested_weight_authority_v2(
            tampered,
            identity_cache=_identity_cache(bundle),
            chain_signing_profile=_chain_profile(),
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "pcr0": expected_pcr0
            },
        )
