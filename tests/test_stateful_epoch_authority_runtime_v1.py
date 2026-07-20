from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot
from gateway.research_lab.stateful_epoch_authority_v1 import (
    BOUNDARY_TABLE,
    CANDIDATE_SUBMISSION_SCHEMA_VERSION,
    CANDIDATE_TABLE,
    CUTOVER_TABLE,
    SNAPSHOT_TABLE,
    StatefulEpochAuthorityStoreError,
    build_boundary_row_v1,
    build_cutover_row_v1,
    build_pre_cutover_candidate_row_v1,
    build_snapshot_row_v1,
    persist_boundary_v1,
    persist_cutover_v1,
    persist_post_cutover_evidence_v1,
    persist_pre_cutover_candidate_v1,
    persist_snapshot_v1,
    validate_stored_pre_cutover_candidate_row_v1,
)
from gateway.research_lab.stateful_epoch_cutover_cli_v1 import (
    ACTIVATION_REPORT_SCHEMA_VERSION,
    CUTOVER_BIND_RPC,
    CUTOVER_BOOTSTRAP_BIND_RPC,
    CUTOVER_BOOTSTRAP_STAGE_RPC,
    CUTOVER_PREFLIGHT_RPC,
    CUTOVER_STAGE_RPC,
    CUTOVER_STATE_TABLE,
    FINALIZED_ALLOCATION_VIEW,
    HISTORICAL_FINALIZATION_TABLE,
    RECEIPT_TABLE,
    StatefulEpochCutoverActivationError,
    _mixed_boot_verifier_from_release,
    activate_subnet_epoch_cutover_v1,
)
from gateway.tee.coordinator_epoch_cutover_v2 import (
    CUTOVER_AUTHORITY_SCHEMA_VERSION,
    CUTOVER_BOOTSTRAP_AUTHORITY_SCHEMA_VERSION,
    CUTOVER_PURPOSE,
    CUTOVER_REQUEST_SCHEMA_VERSION,
    FINALIZATION_PURPOSE,
    HISTORICAL_FINALIZATION_PURPOSE,
    HISTORICAL_PREDECESSOR_KIND,
    OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
    SNAPSHOT_PURPOSE,
)
from gateway.tee.coordinator_executor_v2 import (
    COORDINATOR_OPERATIONS_V2,
    CoordinatorExecutorV2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
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
    transport_root,
    validate_receipt_graph,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)


NOW = "2026-07-16T12:00:00Z"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
COMMIT = "d" * 40
PCR0 = "e" * 96
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


def _cutover(**updates):
    values = {
        "network_genesis_hash": "0x" + "1" * 64,
        "netuid": 71,
        "cutover_block": 1_000,
        "cutover_block_hash": "0x" + "2" * 64,
        "first_subnet_epoch_index": 10,
        "first_settlement_epoch_id": 101,
        "last_legacy_epoch_id": 100,
    }
    values.update(updates)
    return SubnetEpochCutover(**values)


def _snapshot(
    *,
    index=10,
    block=1_000,
    block_hash="0x" + "2" * 64,
    cutover=None,
):
    mapping = cutover or _cutover()
    snapshot = SubnetEpochSnapshot(
        network_genesis_hash=mapping.network_genesis_hash,
        netuid=mapping.netuid,
        head_kind="finalized",
        block_hash=block_hash,
        current_block=block,
        last_epoch_block=block,
        pending_epoch_at=0,
        subnet_epoch_index=index,
        tempo=360,
        blocks_since_last_step=0,
        observed_at=NOW,
    )
    return snapshot.to_dict(cutover=mapping)


def _keypair():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return private_key, public_key


def _boot(role, private_key, public_key):
    body = build_boot_identity_body(
        role=role,
        physical_role=(
            "gateway_coordinator"
            if role == COORDINATOR_ROLE
            else "validator_weights"
        ),
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH_A,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_nonce=("3" if role == COORDINATOR_ROLE else "4") * 32,
        signing_pubkey=public_key,
        transport_pubkey=("5" if role == COORDINATOR_ROLE else "6") * 64,
        transport_certificate_hash=HASH_A,
        attestation_user_data_hash=HASH_B,
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
    private_key,
    public_key,
    boot,
    role,
    purpose,
    job_id,
    epoch_id,
    output_root,
    input_root=HASH_A,
    parents=(),
    attempts=(),
    sequence=0,
    artifact_root=EMPTY_ARTIFACT_ROOT,
):
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=sequence,
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH_A,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=(
            transport_root(attempts) if attempts else EMPTY_TRANSPORT_ROOT
        ),
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=list(parents),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _attempt(*, job_id, purpose):
    return build_transport_attempt(
        request_id="7" * 32,
        logical_operation_id=job_id + ":chain-read",
        job_id=job_id,
        purpose=purpose,
        provider_id="bittensor_chain",
        attempt_number=0,
        method="POST",
        destination_host="entrypoint-finney.opentensor.ai",
        destination_port=443,
        path_hash=HASH_A,
        nonsecret_headers_hash=HASH_B,
        body_hash=HASH_C,
        credential_ref_hash=HASH_A,
        retry_policy_hash=HASH_B,
        timeout_ms=30_000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH_C,
        request_artifact_hash=HASH_A,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH_C,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
    )


def _snapshot_graph(snapshot_doc, *, output_root=None):
    private_key, public_key = _keypair()
    boot = _boot(WEIGHT_ROLE, private_key, public_key)
    receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id="subnet-epoch-boundary:%d" % snapshot_doc["settlement_epoch_id"],
        epoch_id=snapshot_doc["settlement_epoch_id"],
        output_root=output_root or sha256_json(snapshot_doc),
        artifact_root=merkle_root(
            [sha256_json(snapshot_doc)],
            domain="leadpoet-artifact-v2",
        ),
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[receipt],
        transport_attempts=[],
    )
    return graph, receipt, boot, private_key, public_key


def _capture_evidence(snapshot_doc, *, output_root=None):
    graph, receipt, boot, private_key, public_key = _snapshot_graph(
        snapshot_doc,
        output_root=output_root,
    )
    evidence = {
        "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1",
        "epoch_authority": dict(snapshot_doc),
        "epoch_boundary": dict(snapshot_doc),
        "epoch_authority_receipt_hash": receipt["receipt_hash"],
        "epoch_boundary_receipt_hash": receipt["receipt_hash"],
        "receipt_graph": graph,
        "boot_identity": boot,
        "source_artifacts": [],
    }
    return evidence, receipt, boot, private_key, public_key


def _candidate_auth(envelope, cutover):
    signature = "0x" + "1" * 128
    payload_hash = sha256_json(
        {
            "schema_version": CANDIDATE_SUBMISSION_SCHEMA_VERSION,
            "cutover_manifest": cutover.to_dict(),
            "capture": envelope,
        }
    )
    return {
        "validator_hotkey": VALIDATOR_HOTKEY,
        "candidate_payload_hash": payload_hash,
        "validator_hotkey_signature": signature,
        "candidate_authorization_hash": sha256_json(
            {
                "validator_hotkey": VALIDATOR_HOTKEY,
                "candidate_payload_hash": payload_hash,
                "validator_hotkey_signature": signature,
            }
        ),
    }


def _normal_evidence(
    *,
    cutover,
    index=11,
    boundary_block=1_360,
    current_block=1_700,
):
    boundary_doc = _snapshot(
        index=index,
        block=boundary_block,
        block_hash=(
            cutover.cutover_block_hash
            if index == cutover.first_subnet_epoch_index
            and boundary_block == cutover.cutover_block
            else "0x" + "3" * 64
        ),
        cutover=cutover,
    )
    current = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash="0x" + "4" * 64,
        current_block=current_block,
        last_epoch_block=boundary_block,
        pending_epoch_at=0,
        subnet_epoch_index=index,
        tempo=360,
        blocks_since_last_step=current_block - boundary_block,
        observed_at="2026-07-16T12:01:00Z",
    )
    current_doc = current.to_dict(cutover=cutover)
    private_key, public_key = _keypair()
    boot = _boot(WEIGHT_ROLE, private_key, public_key)
    boundary_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id="subnet-epoch-boundary:%d" % current_doc["settlement_epoch_id"],
        epoch_id=current_doc["settlement_epoch_id"],
        output_root=sha256_json(boundary_doc),
    )
    current_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id="subnet-epoch-current:%d" % current_doc["settlement_epoch_id"],
        epoch_id=current_doc["settlement_epoch_id"],
        output_root=sha256_json(current_doc),
        parents=[boundary_receipt["receipt_hash"]],
        sequence=1,
    )
    graph = build_receipt_graph(
        root_receipt_hash=current_receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[boundary_receipt, current_receipt],
        transport_attempts=[],
    )
    evidence = {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "bundle_hash": sha256_json(
            {"bundle": current_doc["settlement_epoch_id"]}
        ),
        "cutover_mapping_hash": cutover.mapping_hash,
        "epoch_authority": current_doc,
        "epoch_authority_hash": sha256_json(current_doc),
        "epoch_authority_receipt_hash": current_receipt["receipt_hash"],
        "epoch_boundary": boundary_doc,
        "epoch_boundary_hash": sha256_json(boundary_doc),
        "epoch_boundary_receipt_hash": boundary_receipt["receipt_hash"],
        "receipt_graph": graph,
    }
    return evidence


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


def _finalization_graph(*, epoch_id=100, finalized_block=999):
    private_key, public_key = _keypair()
    boot = _boot(WEIGHT_ROLE, private_key, public_key)
    computed = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose="validator.weights.computed.v2",
        job_id="weights-computed:%d" % epoch_id,
        epoch_id=epoch_id,
        output_root=HASH_B,
    )
    submission_event_hash = sha256_json({"submission": epoch_id})
    authorization = build_weight_extrinsic_authorization_v2(
        profile=_chain_profile(),
        validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex="8" * 64,
        epoch_id=epoch_id,
        netuid=71,
        weight_receipt_hash=computed["receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash="9" * 64,
        sparse_uids=[0],
        sparse_weights_u16=[65535],
        commitment=b"stateful-epoch-test",
        reveal_round=1234,
        era_current=990,
        nonce=1,
        block_hash="a" * 64,
    )
    extrinsic_signature = "b" * 128
    extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex="8" * 64,
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
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose="validator.set_weights_extrinsic.v2",
        job_id="set-weights:%d" % epoch_id,
        epoch_id=epoch_id,
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[computed["receipt_hash"]],
        sequence=1,
    )
    finalization = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "netuid": 71,
        "epoch_id": epoch_id,
        "weights_hash": "9" * 64,
        "weight_receipt_hash": computed["receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": extrinsic_signature,
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": finalized_block,
        "finalized_block_hash": "c" * 64,
        "state_transition_hash": sha256_json({"transition": epoch_id}),
    }
    job_id = "weight-finalization:%d" % epoch_id
    attempt = _attempt(job_id=job_id, purpose=FINALIZATION_PURPOSE)
    finalization_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose=FINALIZATION_PURPOSE,
        job_id=job_id,
        epoch_id=epoch_id,
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [extrinsic_receipt["receipt_hash"]],
            }
        ),
        output_root=sha256_json(finalization),
        parents=[extrinsic_receipt["receipt_hash"]],
        attempts=[attempt],
        sequence=2,
    )
    graph = build_receipt_graph(
        root_receipt_hash=finalization_receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[computed, extrinsic_receipt, finalization_receipt],
        transport_attempts=[attempt],
    )
    return graph, finalization_receipt, finalization


def _coordinator_request(**updates):
    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    snapshot_graph, snapshot_receipt, *_ = _snapshot_graph(snapshot_doc)
    finalization_graph, finalization_receipt, finalization = _finalization_graph()
    payload = {
        "schema_version": CUTOVER_REQUEST_SCHEMA_VERSION,
        "manifest": cutover.to_dict(),
        "first_snapshot": snapshot_doc,
        "last_legacy_bundle_hash": sha256_json({"bundle": 100}),
        "last_legacy_finalization": finalization,
    }
    payload.update(updates)
    context = ExecutionContextV2(
        job_id="subnet-epoch-cutover:101",
        purpose=CUTOVER_PURPOSE,
        epoch_id=101,
        parent_receipt_hashes=tuple(
            sorted(
                [
                    snapshot_receipt["receipt_hash"],
                    finalization_receipt["receipt_hash"],
                ]
            )
        ),
        external_receipt_graphs=[snapshot_graph, finalization_graph],
    )
    return payload, context


@pytest.mark.asyncio
async def test_measured_cutover_requires_exact_snapshot_and_finalization_parents():
    payload, context = _coordinator_request()
    assert COORDINATOR_OPERATIONS_V2[OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2] == {
        CUTOVER_PURPOSE
    }
    result = await CoordinatorExecutorV2()(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    authority = result.output
    assert authority["schema_version"] == CUTOVER_AUTHORITY_SCHEMA_VERSION
    assert authority["mapping_hash"] == payload["manifest"]["mapping_hash"]
    assert authority["first_snapshot_hash"] == sha256_json(
        payload["first_snapshot"]
    )
    assert authority["first_snapshot_receipt_hash"] in context.parent_receipt_hashes
    assert authority["last_legacy_finalization_receipt_hash"] in context.parent_receipt_hashes
    assert result.receipt_output is None
    assert set(result.artifact_hashes) == {
        authority["mapping_hash"],
        authority["first_snapshot_hash"],
        authority["last_legacy_bundle_hash"],
        authority["last_legacy_weight_finalization_event_hash"],
    }
    # ExecutionJobManagerV2 hashes result.output for the signed receipt root.
    assert sha256_json(authority).startswith("sha256:")


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation,match", [
    ("wrong_parent", "parent roots differ"),
    ("duplicate_purpose", "parent purpose is duplicated"),
    ("tampered_snapshot", "derivations differ"),
    ("late_finalization", "differs from cutover"),
])
async def test_measured_cutover_fails_closed_on_parent_and_payload_tampering(
    mutation,
    match,
):
    if mutation == "late_finalization":
        late_graph, late_receipt, late_doc = _finalization_graph(
            finalized_block=1_000
        )
        payload, context = _coordinator_request(
            last_legacy_finalization=late_doc
        )
        context.external_receipt_graphs[1] = late_graph
        context.parent_receipt_hashes = tuple(
            sorted(
                [
                    context.external_receipt_graphs[0]["root_receipt_hash"],
                    late_receipt["receipt_hash"],
                ]
            )
        )
    else:
        payload, context = _coordinator_request()
    if mutation == "wrong_parent":
        context.parent_receipt_hashes = (context.parent_receipt_hashes[0], HASH_A)
    elif mutation == "duplicate_purpose":
        duplicate_graph, duplicate_receipt, *_ = _snapshot_graph(
            payload["first_snapshot"]
        )
        context.external_receipt_graphs[1] = duplicate_graph
        context.parent_receipt_hashes = tuple(
            sorted(
                [
                    context.external_receipt_graphs[0]["root_receipt_hash"],
                    duplicate_receipt["receipt_hash"],
                ]
            )
        )
    elif mutation == "tampered_snapshot":
        payload["first_snapshot"] = dict(payload["first_snapshot"])
        payload["first_snapshot"]["epoch_ref"] = HASH_A
    with pytest.raises(ValueError, match=match):
        await CoordinatorExecutorV2()(
            OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
            payload,
            context,
        )


@pytest.mark.asyncio
async def test_measured_cutover_rejects_wrong_legacy_epoch_and_snapshot_root():
    wrong_graph, wrong_receipt, wrong_doc = _finalization_graph(epoch_id=99)
    payload, context = _coordinator_request(last_legacy_finalization=wrong_doc)
    context.external_receipt_graphs[1] = wrong_graph
    context.parent_receipt_hashes = tuple(
        sorted(
            [
                context.external_receipt_graphs[0]["root_receipt_hash"],
                wrong_receipt["receipt_hash"],
            ]
        )
    )
    with pytest.raises(ValueError, match="differs from cutover"):
        await CoordinatorExecutorV2()(
            OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
            payload,
            context,
        )


@pytest.mark.asyncio
async def test_measured_cutover_accepts_attested_historical_predecessor(
    monkeypatch,
):
    from gateway.tee import coordinator_epoch_cutover_v2 as cutover_authority

    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    snapshot_graph, snapshot_receipt, *_ = _snapshot_graph(snapshot_doc)
    settlement = {
        "schema_version": "leadpoet.legacy_finalized_allocation.v2",
        "netuid": 71,
        "epoch_id": 92,
        "allocation_hash": HASH_B,
        "settlement_hash": HASH_C,
        "chain_target_block": 900,
    }
    monkeypatch.setattr(
        cutover_authority,
        "validate_legacy_settlement_document_v2",
        lambda value: dict(value),
    )
    private_key, public_key = _keypair()
    boot = _boot(COORDINATOR_ROLE, private_key, public_key)
    predecessor_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=COORDINATOR_ROLE,
        purpose=HISTORICAL_FINALIZATION_PURPOSE,
        job_id="legacy-settlement:92",
        epoch_id=101,
        output_root=sha256_json(settlement),
    )
    predecessor_graph = build_receipt_graph(
        root_receipt_hash=predecessor_receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[predecessor_receipt],
        transport_attempts=[],
    )
    payload = {
        "schema_version": CUTOVER_REQUEST_SCHEMA_VERSION,
        "manifest": cutover.to_dict(),
        "first_snapshot": snapshot_doc,
        "predecessor_kind": HISTORICAL_PREDECESSOR_KIND,
        "predecessor_finalization": settlement,
    }
    context = ExecutionContextV2(
        job_id="subnet-epoch-cutover:101",
        purpose=CUTOVER_PURPOSE,
        epoch_id=101,
        parent_receipt_hashes=tuple(
            sorted(
                (
                    snapshot_receipt["receipt_hash"],
                    predecessor_receipt["receipt_hash"],
                )
            )
        ),
        external_receipt_graphs=[snapshot_graph, predecessor_graph],
    )

    measured = await CoordinatorExecutorV2()(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )

    assert (
        measured.output["schema_version"]
        == CUTOVER_BOOTSTRAP_AUTHORITY_SCHEMA_VERSION
    )
    assert measured.output["predecessor_epoch_id"] == 92
    assert measured.output["predecessor_kind"] == HISTORICAL_PREDECESSOR_KIND
    assert measured.output["predecessor_receipt_hash"] == predecessor_receipt[
        "receipt_hash"
    ]
    coordinator_graph = _cutover_receipt_graph(
        measured.output,
        snapshot_graph,
        predecessor_graph,
    )
    row = build_cutover_row_v1(
        authority_doc=measured.output,
        first_snapshot_doc=snapshot_doc,
        receipt_graph=coordinator_graph,
    )
    assert row["last_legacy_epoch_id"] == 100
    assert row["predecessor_epoch_id"] == 92
    assert row["last_legacy_finalization_receipt_hash"] is None

    payload, context = _coordinator_request()
    bad_graph, bad_receipt, *_ = _snapshot_graph(
        payload["first_snapshot"],
        output_root=HASH_A,
    )
    context.external_receipt_graphs[0] = bad_graph
    context.parent_receipt_hashes = tuple(
        sorted(
            [
                bad_receipt["receipt_hash"],
                context.external_receipt_graphs[1]["root_receipt_hash"],
            ]
        )
    )
    with pytest.raises(ValueError, match="snapshot receipt is invalid"):
        await CoordinatorExecutorV2()(
            OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
            payload,
            context,
        )


class _MemoryStore:
    def __init__(self):
        self.tables = {
            CANDIDATE_TABLE: {},
            BOUNDARY_TABLE: {},
            SNAPSHOT_TABLE: {},
            CUTOVER_TABLE: {},
        }
        self.graphs = {}

    async def insert(self, table, row):
        key_field = {
            CANDIDATE_TABLE: "snapshot_hash",
            BOUNDARY_TABLE: "boundary_hash",
            SNAPSHOT_TABLE: "snapshot_hash",
            CUTOVER_TABLE: "cutover_authority_hash",
        }[table]
        key = row[key_field]
        if key in self.tables[table]:
            raise RuntimeError("23505 duplicate key unique constraint")
        self.tables[table][key] = copy.deepcopy(row)
        return copy.deepcopy(row)

    async def select(self, table, *, filters):
        field, value = filters[0]
        for row in self.tables[table].values():
            if row.get(field) == value:
                return copy.deepcopy(row)
        return None

    async def persist_graph(self, graph):
        root = graph["root_receipt_hash"]
        self.graphs[root] = copy.deepcopy(graph)
        return {
            "root_receipt_hash": root,
            "graph_hash": sha256_json(graph),
        }

    async def load_graph(self, root):
        return copy.deepcopy(self.graphs[root])


@pytest.mark.asyncio
async def test_candidate_persistence_matches_sql_row_and_is_exact_idempotent():
    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    envelope, *_ = _capture_evidence(snapshot_doc)
    auth = _candidate_auth(envelope, cutover)
    graph = envelope["receipt_graph"]
    row = build_pre_cutover_candidate_row_v1(
        envelope,
        cutover=cutover.to_dict(),
        **auth,
    )
    assert row["snapshot_hash"] == sha256_json(snapshot_doc)
    assert row == {
        "snapshot_hash": sha256_json(snapshot_doc),
        "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
        "mapping_hash": cutover.mapping_hash,
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": cutover.network_genesis_hash,
        "netuid": 71,
        "head_kind": "finalized",
        "block_hash": cutover.cutover_block_hash,
        "current_block": 1_000,
        "last_epoch_block": 1_000,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 10,
        "epoch_ref": snapshot_doc["epoch_ref"],
        "proposed_settlement_epoch_id": 101,
        "tempo": 360,
        "blocks_since_last_step": 0,
        "next_epoch_block": 1_360,
        "blocks_remaining": 360,
        "chain_state_receipt_hash": graph["root_receipt_hash"],
        **auth,
        "snapshot_doc": snapshot_doc,
        "observed_at": NOW,
    }

    store = _MemoryStore()
    first = await persist_pre_cutover_candidate_v1(
        envelope,
        cutover=cutover.to_dict(),
        **auth,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    second = await persist_pre_cutover_candidate_v1(
        envelope,
        cutover=cutover.to_dict(),
        **auth,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert first == second == row


def test_candidate_capture_rejects_extra_receipts_even_with_same_output_root():
    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    graph, boundary_receipt, boot, private_key, public_key = _snapshot_graph(
        snapshot_doc
    )
    current_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id="subnet-epoch-current:101",
        epoch_id=101,
        output_root=sha256_json(snapshot_doc),
        sequence=1,
    )
    chain_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=WEIGHT_ROLE,
        purpose="validator.chain_state.v2",
        job_id="chain-state:101",
        epoch_id=101,
        output_root=HASH_C,
        parents=[
            boundary_receipt["receipt_hash"],
            current_receipt["receipt_hash"],
        ],
        sequence=2,
    )
    graph = build_receipt_graph(
        root_receipt_hash=chain_receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[boundary_receipt, current_receipt, chain_receipt],
        transport_attempts=[],
    )
    envelope = {
        "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1",
        "epoch_authority": snapshot_doc,
        "epoch_boundary": snapshot_doc,
        "epoch_authority_receipt_hash": boundary_receipt["receipt_hash"],
        "epoch_boundary_receipt_hash": boundary_receipt["receipt_hash"],
        "receipt_graph": graph,
        "boot_identity": boot,
        "source_artifacts": [],
    }
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="one exact boundary receipt",
    ):
        build_pre_cutover_candidate_row_v1(
            envelope,
            cutover=cutover.to_dict(),
            **_candidate_auth(envelope, cutover),
        )


def test_candidate_authorization_commitments_fail_closed_on_tampering():
    cutover = _cutover()
    envelope, *_ = _capture_evidence(_snapshot(cutover=cutover))
    auth = _candidate_auth(envelope, cutover)
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="candidate payload hash differs",
    ):
        build_pre_cutover_candidate_row_v1(
            envelope,
            cutover=cutover.to_dict(),
            **{**auth, "candidate_payload_hash": HASH_A},
        )
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="candidate authorization hash differs",
    ):
        build_pre_cutover_candidate_row_v1(
            envelope,
            cutover=cutover.to_dict(),
            **{**auth, "candidate_authorization_hash": HASH_A},
        )


@pytest.mark.asyncio
async def test_candidate_duplicate_conflict_and_graph_readback_tamper_fail_closed():
    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    envelope, *_ = _capture_evidence(snapshot_doc)
    auth = _candidate_auth(envelope, cutover)
    store = _MemoryStore()
    stored = await persist_pre_cutover_candidate_v1(
        envelope,
        cutover=cutover.to_dict(),
        **auth,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    store.tables[CANDIDATE_TABLE][stored["snapshot_hash"]]["netuid"] = 99
    with pytest.raises(StatefulEpochAuthorityStoreError, match="conflicts at netuid"):
        await persist_pre_cutover_candidate_v1(
            envelope,
            cutover=cutover.to_dict(),
            **auth,
            persist_graph=store.persist_graph,
            load_graph=store.load_graph,
            insert=store.insert,
            select=store.select,
        )

    async def tampered_load(root):
        loaded = await store.load_graph(root)
        loaded["root_receipt_hash"] = HASH_A
        return loaded

    with pytest.raises(StatefulEpochAuthorityStoreError, match="readback differs"):
        await persist_pre_cutover_candidate_v1(
            envelope,
            cutover=cutover.to_dict(),
            **auth,
            persist_graph=store.persist_graph,
            load_graph=tampered_load,
            insert=store.insert,
            select=store.select,
        )


@pytest.mark.asyncio
async def test_candidate_readback_accepts_equivalent_postgres_timestamptz_format():
    cutover = _cutover()
    snapshot_doc = _snapshot(cutover=cutover)
    envelope, *_ = _capture_evidence(snapshot_doc)
    auth = _candidate_auth(envelope, cutover)
    store = _MemoryStore()

    async def normalized_insert(table, row):
        normalized = copy.deepcopy(row)
        normalized["observed_at"] = "2026-07-16T12:00:00+00:00"
        return await store.insert(table, normalized)

    stored = await persist_pre_cutover_candidate_v1(
        envelope,
        cutover=cutover.to_dict(),
        **auth,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=normalized_insert,
        select=store.select,
    )
    assert stored["observed_at"] == "2026-07-16T12:00:00+00:00"


@pytest.mark.asyncio
async def test_post_cutover_boundary_persistence_matches_sql_row_and_is_idempotent():
    cutover = _cutover()
    envelope = _normal_evidence(cutover=cutover)
    snapshot_doc = envelope["epoch_boundary"]
    row = build_boundary_row_v1(envelope, cutover=cutover.to_dict())
    assert row["boundary_hash"] == sha256_json(snapshot_doc)
    assert row["boundary_doc"]["snapshot"] == snapshot_doc
    assert row["settlement_epoch_id"] == 102

    store = _MemoryStore()
    first = await persist_boundary_v1(
        envelope,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    second = await persist_boundary_v1(
        envelope,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert first == second == row

    bad = copy.deepcopy(envelope)
    bad["epoch_boundary"]["epoch_ref"] = HASH_A
    with pytest.raises(StatefulEpochAuthorityStoreError, match="derivations differ"):
        build_boundary_row_v1(bad, cutover=cutover.to_dict())


@pytest.mark.asyncio
async def test_normal_evidence_persists_explicit_current_snapshot_and_boundary_once():
    cutover = _cutover()
    evidence = _normal_evidence(cutover=cutover)
    snapshot = evidence["epoch_authority"]
    snapshot_row = build_snapshot_row_v1(
        evidence,
        cutover=cutover.to_dict(),
    )
    assert snapshot_row == {
        "snapshot_hash": evidence["epoch_authority_hash"],
        "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
        "mapping_hash": cutover.mapping_hash,
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": cutover.network_genesis_hash,
        "netuid": 71,
        "head_kind": "finalized",
        "block_hash": snapshot["block_hash"],
        "current_block": 1_700,
        "last_epoch_block": 1_360,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 11,
        "epoch_ref": snapshot["epoch_ref"],
        "settlement_epoch_id": 102,
        "tempo": 360,
        "blocks_since_last_step": 340,
        "epoch_block": 340,
        "next_epoch_block": 1_720,
        "blocks_remaining": 20,
        "chain_state_receipt_hash": evidence[
            "epoch_authority_receipt_hash"
        ],
        "snapshot_doc": snapshot,
        "observed_at": "2026-07-16T12:01:00Z",
    }

    store = _MemoryStore()
    standalone = await persist_snapshot_v1(
        evidence,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert standalone == snapshot_row

    store = _MemoryStore()
    first = await persist_post_cutover_evidence_v1(
        evidence,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    second = await persist_post_cutover_evidence_v1(
        evidence,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert first == second
    assert first["snapshot"] == snapshot_row
    assert first["boundary"]["boundary_hash"] == evidence[
        "epoch_boundary_hash"
    ]
    assert first["receipt_graph_hash"] == sha256_json(evidence["receipt_graph"])
    assert first["durable_readback_hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_first_normal_epoch_persists_snapshot_without_duplicate_boundary_row():
    cutover = _cutover()
    evidence = _normal_evidence(
        cutover=cutover,
        index=10,
        boundary_block=1_000,
        current_block=1_340,
    )
    store = _MemoryStore()
    result = await persist_post_cutover_evidence_v1(
        evidence,
        cutover=cutover.to_dict(),
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert result["boundary"] is None
    assert result["snapshot"]["subnet_epoch_index"] == 10
    assert store.tables[BOUNDARY_TABLE] == {}


def test_normal_evidence_rejects_explicit_hash_tampering_and_shadow_conflation():
    cutover = _cutover()
    evidence = _normal_evidence(cutover=cutover)
    evidence["epoch_authority_hash"] = HASH_A
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="commitments differ",
    ):
        build_snapshot_row_v1(evidence, cutover=cutover.to_dict())

    capture, *_ = _capture_evidence(_snapshot(cutover=cutover))
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="normal validator evidence",
    ):
        build_snapshot_row_v1(capture, cutover=cutover.to_dict())
    normal = _normal_evidence(cutover=cutover)
    wrong_receipt = copy.deepcopy(normal)
    wrong_receipt["epoch_authority_receipt_hash"] = wrong_receipt[
        "epoch_boundary_receipt_hash"
    ]
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="declared snapshot receipt",
    ):
        build_snapshot_row_v1(
            wrong_receipt,
            cutover=cutover.to_dict(),
        )
    with pytest.raises(
        StatefulEpochAuthorityStoreError,
        match="shadow capture evidence",
    ):
        build_pre_cutover_candidate_row_v1(
            normal,
            cutover=cutover.to_dict(),
            **_candidate_auth(normal, cutover),
        )


def _cutover_receipt_graph(authority, snapshot_graph, finalization_graph):
    private_key, public_key = _keypair()
    coordinator_boot = _boot(COORDINATOR_ROLE, private_key, public_key)
    parents = sorted(
        [
            snapshot_graph["root_receipt_hash"],
            finalization_graph["root_receipt_hash"],
        ]
    )
    receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=coordinator_boot,
        role=COORDINATOR_ROLE,
        purpose=CUTOVER_PURPOSE,
        job_id="subnet-epoch-cutover:101",
        epoch_id=101,
        output_root=sha256_json(authority),
        parents=parents,
    )
    boots = {
        boot["boot_identity_hash"]: boot
        for boot in (
            snapshot_graph["boot_identities"]
            + finalization_graph["boot_identities"]
            + [coordinator_boot]
        )
    }
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=list(boots.values()),
        receipts=(
            snapshot_graph["receipts"]
            + finalization_graph["receipts"]
            + [receipt]
        ),
        transport_attempts=(
            snapshot_graph["transport_attempts"]
            + finalization_graph["transport_attempts"]
        ),
    )


@pytest.mark.asyncio
async def test_cutover_row_and_persistence_bind_coordinator_output_and_both_parents():
    payload, context = _coordinator_request()
    result = await CoordinatorExecutorV2()(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    graph = _cutover_receipt_graph(
        result.output,
        context.external_receipt_graphs[0],
        context.external_receipt_graphs[1],
    )
    row = build_cutover_row_v1(
        authority_doc=result.output,
        first_snapshot_doc=payload["first_snapshot"],
        receipt_graph=graph,
    )
    assert row["cutover_authority_hash"] == sha256_json(result.output)
    assert row["cutover_receipt_hash"] == graph["root_receipt_hash"]

    store = _MemoryStore()
    first = await persist_cutover_v1(
        authority_doc=result.output,
        first_snapshot_doc=payload["first_snapshot"],
        receipt_graph=graph,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    second = await persist_cutover_v1(
        authority_doc=result.output,
        first_snapshot_doc=payload["first_snapshot"],
        receipt_graph=graph,
        persist_graph=store.persist_graph,
        load_graph=store.load_graph,
        insert=store.insert,
        select=store.select,
    )
    assert first == second == row

    tampered = copy.deepcopy(result.output)
    tampered["last_legacy_bundle_hash"] = HASH_A
    with pytest.raises(StatefulEpochAuthorityStoreError, match="coordinator receipt"):
        build_cutover_row_v1(
            authority_doc=tampered,
            first_snapshot_doc=payload["first_snapshot"],
            receipt_graph=graph,
        )


def _cutover_activation_dependencies():
    payload, context = _coordinator_request()
    cutover = SubnetEpochCutover.from_mapping(payload["manifest"])
    snapshot_graph = context.external_receipt_graphs[0]
    capture = {
        "schema_version": "leadpoet.subnet_epoch_boundary_capture.v1",
        "epoch_authority": payload["first_snapshot"],
        "epoch_boundary": payload["first_snapshot"],
        "epoch_authority_receipt_hash": snapshot_graph["root_receipt_hash"],
        "epoch_boundary_receipt_hash": snapshot_graph["root_receipt_hash"],
        "receipt_graph": snapshot_graph,
        "boot_identity": snapshot_graph["boot_identities"][0],
        "source_artifacts": [],
    }
    auth = _candidate_auth(capture, cutover)
    candidate = build_pre_cutover_candidate_row_v1(
        capture,
        cutover=cutover.to_dict(),
        **auth,
    )
    candidate["created_at"] = NOW
    authority = CoordinatorExecutorV2()
    return payload, context, cutover, candidate, authority


def _fenced_cutover_state(cutover):
    return {
        "singleton": True,
        "lifecycle_state": "cutover_fenced",
        "network_genesis_hash": cutover.network_genesis_hash,
        "netuid": cutover.netuid,
        "mapping_hash": None,
        "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        "cutover_authority_hash": None,
        "cutover_receipt_hash": None,
        "initialization_nonce": None,
        "initialization_payload_hash": None,
    }


def _cutover_binding_response(
    cutover,
    candidate,
    authority_hash,
    *,
    receipt_hash=None,
):
    return {
        "lifecycle_state": "cutover_fenced",
        "mapping_hash": cutover.mapping_hash,
        "legacy_high_water": cutover.last_legacy_epoch_id,
        "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
        "candidate_snapshot_hash": candidate["snapshot_hash"],
        "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
        "cutover_authority_hash": authority_hash,
        "cutover_receipt_hash": receipt_hash,
        "initialization_nonce": None,
        "initialization_payload_hash": None,
    }


def _prepared_initialization(cutover, snapshot_doc):
    from gateway.tasks.epoch_lifecycle import build_epoch_event_row

    payload = {
        "epoch_id": cutover.first_settlement_epoch_id,
        "epoch_key_semantics": "settlement_ordinal",
        "epoch_authority": copy.deepcopy(snapshot_doc),
        "epoch_boundaries": {
            "start_block": snapshot_doc["last_epoch_block"],
            "end_block": snapshot_doc["next_epoch_block"],
            "expected_end_block": snapshot_doc["next_epoch_block"],
            "pending_epoch_at": snapshot_doc["pending_epoch_at"],
            "tempo": snapshot_doc["tempo"],
        },
        "timestamp": snapshot_doc["observed_at"],
    }
    return build_epoch_event_row(
        "EPOCH_INITIALIZATION",
        cutover.first_settlement_epoch_id,
        payload,
        event_timestamp=snapshot_doc["observed_at"],
    )


async def _async_none():
    return None


@pytest.mark.asyncio
async def test_cutover_mixed_boot_verifier_accepts_real_validator_coordinator_ancestry():
    payload, context = _coordinator_request()
    measured = await CoordinatorExecutorV2()(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    graph = _cutover_receipt_graph(
        measured.output,
        context.external_receipt_graphs[0],
        context.external_receipt_graphs[1],
    )
    observed = []

    def nitro_verifier(identity, *, expected_pcr0):
        assert identity["pcr0"] == expected_pcr0
        observed.append(identity["physical_role"])
        return {"verified": True}

    def validator_pcr0_verifier(pcr0):
        assert pcr0 == PCR0
        return {"valid": True, "commit_hash": COMMIT}

    verifier = _mixed_boot_verifier_from_release(
        {
            "roles": {
                "gateway_coordinator": {
                    "commit_sha": COMMIT,
                    "pcr0": PCR0,
                    "execution_manifest_hash": HASH_A,
                    "dependency_lock_hash": HASH_B,
                }
            }
        },
        nitro_verifier=nitro_verifier,
        validator_pcr0_verifier=validator_pcr0_verifier,
    )
    validate_receipt_graph(
        graph,
        required_purposes={CUTOVER_PURPOSE},
        boot_attestation_verifier=verifier,
        require_boot_attestation_verification=True,
    )
    assert observed.count("gateway_coordinator") == 1
    assert observed.count("validator_weights") == 2

    validator_boot = next(
        boot
        for boot in graph["boot_identities"]
        if boot["physical_role"] == "validator_weights"
    )
    with pytest.raises(ValueError, match="commit differs"):
        verifier({**validator_boot, "commit_sha": "0" * 40})


@pytest.mark.asyncio
async def test_cutover_operator_dry_run_is_read_only_and_checks_high_water():
    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    result = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    finalization_graph = context.external_receipt_graphs[1]
    finalization = {
        "bundle_hash": payload["last_legacy_bundle_hash"],
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_finalization_event_hash": result.output[
            "last_legacy_weight_finalization_event_hash"
        ],
        "finalization_receipt_hash": finalization_graph["root_receipt_hash"],
        "finalized_block": payload["last_legacy_finalization"]["finalized_block"],
        "finalization_doc": payload["last_legacy_finalization"],
    }
    selected = []

    async def select_rows(table, **_kwargs):
        selected.append(table)
        if table == CUTOVER_TABLE:
            return []
        if table == CUTOVER_STATE_TABLE:
            return [_fenced_cutover_state(cutover)]
        if table == CANDIDATE_TABLE:
            return [copy.deepcopy(candidate)]
        if table == FINALIZED_ALLOCATION_VIEW:
            return [copy.deepcopy(finalization)]
        if table == RECEIPT_TABLE:
            return []
        raise AssertionError(table)

    graphs = {
        context.external_receipt_graphs[0]["root_receipt_hash"]: context.external_receipt_graphs[0],
        context.external_receipt_graphs[1]["root_receipt_hash"]: context.external_receipt_graphs[1],
    }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    observed_preflight = {}

    async def preflight(name, params):
        observed_preflight.update({"name": name, "params": dict(params)})
        return {
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 100,
            "first_settlement_occupied": False,
            "eligible": True,
        }

    async def forbidden(**_kwargs):
        raise AssertionError("dry-run entered a mutating path")

    async def no_initialization(_epoch_id):
        return None

    async def live_snapshot(_cutover):
        return SubnetEpochSnapshot.from_mapping(payload["first_snapshot"])

    report = await activate_subnet_epoch_cutover_v1(
        cutover=cutover,
        apply=False,
        select_rows=select_rows,
        load_graph=load_graph,
        preflight_rpc=preflight,
        execute=forbidden,
        load_initialization=no_initialization,
        load_live_snapshot=live_snapshot,
        validate_anchor=lambda _cutover: _async_none(),
    )
    assert report["schema_version"] == ACTIVATION_REPORT_SCHEMA_VERSION
    assert report["mode"] == "dry_run"
    assert report["status"] == "eligible"
    assert report["mapping_hash"] == cutover.mapping_hash
    assert report["candidate_snapshot_hash"] == candidate["snapshot_hash"]
    assert report["predicted_cutover_authority_hash"] == sha256_json(result.output)
    assert report["would_write"] is False
    assert report["would_transition"] == [
        "legacy_open",
        "cutover_fenced",
        "stateful_staged",
    ]
    assert report["requires_separate_activation"] is True
    assert report["initialization"]["eligible"] is True
    assert report["initialization"]["latest_safe_epoch_block"] == 300
    assert selected == [
        CUTOVER_TABLE,
        CUTOVER_STATE_TABLE,
        CANDIDATE_TABLE,
        FINALIZED_ALLOCATION_VIEW,
        RECEIPT_TABLE,
    ]
    assert observed_preflight == {
        "name": CUTOVER_PREFLIGHT_RPC,
        "params": {
            "p_mapping_hash": cutover.mapping_hash,
            "p_cutover_receipt_hash": None,
        },
    }


@pytest.mark.asyncio
async def test_cutover_operator_selects_latest_attested_historical_predecessor(
    monkeypatch,
):
    from gateway.tee import coordinator_epoch_cutover_v2 as cutover_authority

    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    settlement = {
        "schema_version": "leadpoet.legacy_finalized_allocation.v2",
        "netuid": 71,
        "epoch_id": 92,
        "allocation_hash": HASH_B,
        "settlement_hash": HASH_C,
        "chain_target_block": 900,
    }
    monkeypatch.setattr(
        cutover_authority,
        "validate_legacy_settlement_document_v2",
        lambda value: dict(value),
    )
    private_key, public_key = _keypair()
    boot = _boot(COORDINATOR_ROLE, private_key, public_key)
    predecessor_receipt = _receipt(
        private_key=private_key,
        public_key=public_key,
        boot=boot,
        role=COORDINATOR_ROLE,
        purpose=HISTORICAL_FINALIZATION_PURPOSE,
        job_id="legacy-settlement:92",
        epoch_id=101,
        output_root=sha256_json(settlement),
    )
    predecessor_graph = build_receipt_graph(
        root_receipt_hash=predecessor_receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[predecessor_receipt],
        transport_attempts=[],
    )
    snapshot_graph = context.external_receipt_graphs[0]
    selected = []
    durable_cutover = None
    durable_initialization = None
    cutover_state = _fenced_cutover_state(cutover)

    async def select_rows(table, **kwargs):
        selected.append((table, kwargs))
        if table == CUTOVER_TABLE:
            return (
                []
                if durable_cutover is None
                else [copy.deepcopy(durable_cutover)]
            )
        if table == CUTOVER_STATE_TABLE:
            return [copy.deepcopy(cutover_state)]
        if table == CANDIDATE_TABLE:
            return [copy.deepcopy(candidate)]
        if table == HISTORICAL_FINALIZATION_TABLE:
            return [{
                "netuid": 71,
                "epoch_id": 92,
                "allocation_hash": HASH_B,
                "settlement_hash": HASH_C,
                "settlement_receipt_hash": predecessor_receipt[
                    "receipt_hash"
                ],
                "settlement_doc": settlement,
            }]
        if table == RECEIPT_TABLE:
            return []
        raise AssertionError(table)

    graphs = {
        snapshot_graph["root_receipt_hash"]: snapshot_graph,
        predecessor_graph["root_receipt_hash"]: predecessor_graph,
    }

    async def preflight(_name, _params):
        return {
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 100,
            "first_settlement_occupied": False,
            "eligible": True,
        }

    async def readiness(**_kwargs):
        return {
            "ready": True,
            "historical_classification_coverage": 1.0,
            "missing_historical_classifications": [],
        }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    async def no_initialization(_epoch):
        return None

    async def live_snapshot(_cutover):
        return SubnetEpochSnapshot.from_mapping(payload["first_snapshot"])

    report = await activate_subnet_epoch_cutover_v1(
        cutover=cutover,
        apply=False,
        predecessor_kind=HISTORICAL_PREDECESSOR_KIND,
        select_rows=select_rows,
        load_graph=load_graph,
        preflight_rpc=preflight,
        load_cutover_readiness=readiness,
        load_initialization=no_initialization,
        load_live_snapshot=live_snapshot,
        validate_anchor=lambda _cutover: _async_none(),
    )

    assert report["status"] == "eligible"
    assert report["predecessor_kind"] == HISTORICAL_PREDECESSOR_KIND
    assert report["predecessor_epoch_id"] == 92
    historical_call = next(
        kwargs
        for table, kwargs in selected
        if table == HISTORICAL_FINALIZATION_TABLE
    )
    assert historical_call["filters"] == (
        ("netuid", 71),
        ("epoch_id", "lte", 100),
    )
    assert historical_call["order_by"] == (("epoch_id", True),)
    assert all(table != FINALIZED_ALLOCATION_VIEW for table, _ in selected)

    observed = {}

    async def execute(**kwargs):
        execution_context = ExecutionContextV2(
            job_id=f"subnet-epoch-cutover:{kwargs['epoch_id']}",
            purpose=kwargs["purpose"],
            epoch_id=kwargs["epoch_id"],
            parent_receipt_hashes=tuple(
                sorted(
                    graph["root_receipt_hash"]
                    for graph in kwargs["parent_graphs"]
                )
            ),
            external_receipt_graphs=list(kwargs["parent_graphs"]),
        )
        measured = await executor(
            kwargs["operation"],
            kwargs["payload"],
            execution_context,
        )
        graph = _cutover_receipt_graph(
            measured.output,
            snapshot_graph,
            predecessor_graph,
        )
        observed["authority"] = measured.output
        return {
            "status": "succeeded",
            "result": measured.output,
            "receipt_graph": graph,
        }

    async def persist_graph(graph):
        return {
            "root_receipt_hash": graph["root_receipt_hash"],
            "graph_hash": sha256_json(graph),
        }

    async def bind(name, params):
        assert name == CUTOVER_BOOTSTRAP_BIND_RPC
        return _cutover_binding_response(
            cutover,
            candidate,
            params["p_cutover_authority_hash"],
            receipt_hash=params["p_cutover_receipt_hash"],
        )

    async def stage(name, params):
        nonlocal durable_cutover, durable_initialization, cutover_state
        assert name == CUTOVER_BOOTSTRAP_STAGE_RPC
        durable_cutover = copy.deepcopy(params["p_cutover_row"])
        durable_cutover["created_at"] = NOW
        durable_initialization = copy.deepcopy(params["p_initialization_event"])
        cutover_state = {
            **cutover_state,
            "lifecycle_state": "stateful_staged",
            "mapping_hash": cutover.mapping_hash,
            "cutover_authority_hash": durable_cutover[
                "cutover_authority_hash"
            ],
            "cutover_receipt_hash": durable_cutover["cutover_receipt_hash"],
            "initialization_nonce": durable_initialization["nonce"],
            "initialization_payload_hash": durable_initialization[
                "payload_hash"
            ],
        }
        return {
            key: cutover_state[key]
            for key in (
                "lifecycle_state",
                "mapping_hash",
                "cutover_authority_hash",
                "cutover_receipt_hash",
                "initialization_nonce",
                "initialization_payload_hash",
            )
        }

    async def load_initialization(_epoch):
        return copy.deepcopy(durable_initialization)

    async def prepare_initialization(**_kwargs):
        return _prepared_initialization(cutover, payload["first_snapshot"])

    apply_report = await activate_subnet_epoch_cutover_v1(
        cutover=cutover,
        apply=True,
        predecessor_kind=HISTORICAL_PREDECESSOR_KIND,
        select_rows=select_rows,
        load_graph=load_graph,
        preflight_rpc=preflight,
        bind_rpc=bind,
        stage_rpc=stage,
        execute=execute,
        persist_graph=persist_graph,
        load_cutover_readiness=readiness,
        load_initialization=load_initialization,
        load_live_snapshot=live_snapshot,
        prepare_initialization=prepare_initialization,
        boot_verifier=lambda _identity: {"verified": True},
        validate_anchor=lambda _cutover: _async_none(),
    )

    assert apply_report["status"] == "stateful_staged"
    assert apply_report["predecessor_kind"] == HISTORICAL_PREDECESSOR_KIND
    assert apply_report["predecessor_epoch_id"] == 92
    assert apply_report["predecessor_receipt_hash"] == predecessor_receipt[
        "receipt_hash"
    ]
    assert apply_report["cutover_authority_hash"] == sha256_json(
        observed["authority"]
    )


@pytest.mark.asyncio
async def test_cutover_operator_apply_enters_coordinator_and_persists_exact_graph():
    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    measured = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    finalization_graph = context.external_receipt_graphs[1]
    finalization = {
        "bundle_hash": payload["last_legacy_bundle_hash"],
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_finalization_event_hash": measured.output[
            "last_legacy_weight_finalization_event_hash"
        ],
        "finalization_receipt_hash": finalization_graph["root_receipt_hash"],
        "finalized_block": payload["last_legacy_finalization"]["finalized_block"],
        "finalization_doc": payload["last_legacy_finalization"],
    }

    durable_cutover = None
    durable_initialization = None
    cutover_state = _fenced_cutover_state(cutover)

    async def select_rows(table, **_kwargs):
        if table == CUTOVER_TABLE:
            return (
                []
                if durable_cutover is None
                else [copy.deepcopy(durable_cutover)]
            )
        if table == CUTOVER_STATE_TABLE:
            return [copy.deepcopy(cutover_state)]
        return {
            CANDIDATE_TABLE: [copy.deepcopy(candidate)],
            FINALIZED_ALLOCATION_VIEW: [copy.deepcopy(finalization)],
            RECEIPT_TABLE: [],
        }[table]

    graphs = {
        context.external_receipt_graphs[0]["root_receipt_hash"]: context.external_receipt_graphs[0],
        context.external_receipt_graphs[1]["root_receipt_hash"]: context.external_receipt_graphs[1],
    }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    async def preflight(_name, _params):
        return [{
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 100,
            "first_settlement_occupied": False,
            "eligible": True,
        }]

    observed = {}
    cutover_graph = _cutover_receipt_graph(
        measured.output,
        context.external_receipt_graphs[0],
        context.external_receipt_graphs[1],
    )

    async def execute(**kwargs):
        observed["execute"] = kwargs
        return {
            "status": "succeeded",
            "result": measured.output,
            "receipt_graph": cutover_graph,
        }

    async def graph_persist(graph):
        observed["graph_persist"] = graph
        return {
            "root_receipt_hash": graph["root_receipt_hash"],
            "graph_hash": sha256_json(graph),
        }

    def mixed_boot_verifier(_identity):
        return {"verified": True}

    async def load_initialization(_epoch_id):
        return copy.deepcopy(durable_initialization)

    async def live_snapshot(_cutover):
        return SubnetEpochSnapshot.from_mapping(payload["first_snapshot"])

    async def prepare_initialization(**_kwargs):
        return _prepared_initialization(cutover, payload["first_snapshot"])

    async def bind(name, params):
        assert name == CUTOVER_BIND_RPC
        observed["bind"] = dict(params)
        return _cutover_binding_response(
            cutover,
            candidate,
            sha256_json(measured.output),
            receipt_hash=params["p_cutover_receipt_hash"],
        )

    async def stage(name, params):
        nonlocal durable_cutover, durable_initialization, cutover_state
        assert name == CUTOVER_STAGE_RPC
        observed["stage"] = copy.deepcopy(params)
        durable_cutover = copy.deepcopy(params["p_cutover_row"])
        durable_cutover["created_at"] = NOW
        durable_initialization = copy.deepcopy(params["p_initialization_event"])
        cutover_state = {
            **cutover_state,
            "lifecycle_state": "stateful_staged",
            "mapping_hash": cutover.mapping_hash,
            "cutover_authority_hash": durable_cutover[
                "cutover_authority_hash"
            ],
            "cutover_receipt_hash": durable_cutover["cutover_receipt_hash"],
            "initialization_nonce": durable_initialization["nonce"],
            "initialization_payload_hash": durable_initialization[
                "payload_hash"
            ],
        }
        return {
            key: cutover_state[key]
            for key in (
                "lifecycle_state",
                "mapping_hash",
                "cutover_authority_hash",
                "cutover_receipt_hash",
                "initialization_nonce",
                "initialization_payload_hash",
            )
        }

    report = await activate_subnet_epoch_cutover_v1(
        cutover=cutover,
        apply=True,
        select_rows=select_rows,
        load_graph=load_graph,
        preflight_rpc=preflight,
        bind_rpc=bind,
        stage_rpc=stage,
        execute=execute,
        persist_graph=graph_persist,
        load_initialization=load_initialization,
        load_live_snapshot=live_snapshot,
        prepare_initialization=prepare_initialization,
        boot_verifier=mixed_boot_verifier,
        validate_anchor=lambda _cutover: _async_none(),
    )
    assert report["status"] == "stateful_staged"
    assert report["cutover_authority_hash"] == sha256_json(measured.output)
    assert observed["execute"]["operation"] == OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2
    assert observed["execute"]["parent_graphs"] == tuple(context.external_receipt_graphs)
    assert observed["execute"]["persist_graph"] is graph_persist
    assert observed["execute"]["boot_verifier"] is mixed_boot_verifier
    assert observed["stage"]["p_cutover_row"] == {
        key: durable_cutover[key]
        for key in durable_cutover
        if key != "created_at"
    }
    assert report["initialization"] == {
        "exists": True,
        "status": "created_atomically",
        "eligible": True,
        "event_hash": durable_initialization.get("event_hash"),
        "nonce": durable_initialization["nonce"],
        "payload_hash": durable_initialization["payload_hash"],
        "authority_hash": sha256_json(payload["first_snapshot"]),
    }


@pytest.mark.asyncio
async def test_cutover_operator_resumes_persisted_coordinator_graph_without_reexecution():
    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    measured = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    coordinator_graph = _cutover_receipt_graph(
        measured.output,
        context.external_receipt_graphs[0],
        context.external_receipt_graphs[1],
    )
    finalization = {
        "bundle_hash": payload["last_legacy_bundle_hash"],
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_finalization_event_hash": measured.output[
            "last_legacy_weight_finalization_event_hash"
        ],
        "finalization_receipt_hash": context.external_receipt_graphs[1]["root_receipt_hash"],
        "finalized_block": payload["last_legacy_finalization"]["finalized_block"],
        "finalization_doc": payload["last_legacy_finalization"],
    }

    durable_cutover = None
    durable_initialization = None
    cutover_state = _fenced_cutover_state(cutover)

    async def select_rows(table, **_kwargs):
        if table == CUTOVER_TABLE:
            return (
                []
                if durable_cutover is None
                else [copy.deepcopy(durable_cutover)]
            )
        if table == CUTOVER_STATE_TABLE:
            return [copy.deepcopy(cutover_state)]
        return {
            CANDIDATE_TABLE: [copy.deepcopy(candidate)],
            FINALIZED_ALLOCATION_VIEW: [copy.deepcopy(finalization)],
            RECEIPT_TABLE: [{
                "receipt_hash": coordinator_graph["root_receipt_hash"],
            }],
        }[table]

    graphs = {
        graph["root_receipt_hash"]: graph
        for graph in (
            *context.external_receipt_graphs,
            coordinator_graph,
        )
    }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    observed = {}

    async def preflight(name, params):
        observed["preflight"] = (name, dict(params))
        return {
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 100,
            "first_settlement_occupied": False,
            "eligible": True,
        }

    async def load_initialization(_epoch_id):
        return copy.deepcopy(durable_initialization)

    async def live_snapshot(_cutover):
        return SubnetEpochSnapshot.from_mapping(payload["first_snapshot"])

    async def prepare_initialization(**_kwargs):
        return _prepared_initialization(cutover, payload["first_snapshot"])

    async def bind(name, params):
        assert name == CUTOVER_BIND_RPC
        return _cutover_binding_response(
            cutover,
            candidate,
            sha256_json(measured.output),
            receipt_hash=params["p_cutover_receipt_hash"],
        )

    async def stage(name, params):
        nonlocal durable_cutover, durable_initialization, cutover_state
        assert name == CUTOVER_STAGE_RPC
        durable_cutover = copy.deepcopy(params["p_cutover_row"])
        durable_cutover["created_at"] = NOW
        durable_initialization = copy.deepcopy(params["p_initialization_event"])
        cutover_state = {
            **cutover_state,
            "lifecycle_state": "stateful_staged",
            "mapping_hash": cutover.mapping_hash,
            "cutover_authority_hash": durable_cutover[
                "cutover_authority_hash"
            ],
            "cutover_receipt_hash": durable_cutover["cutover_receipt_hash"],
            "initialization_nonce": durable_initialization["nonce"],
            "initialization_payload_hash": durable_initialization[
                "payload_hash"
            ],
        }
        return {
            key: cutover_state[key]
            for key in (
                "lifecycle_state",
                "mapping_hash",
                "cutover_authority_hash",
                "cutover_receipt_hash",
                "initialization_nonce",
                "initialization_payload_hash",
            )
        }

    async def forbidden(**_kwargs):
        raise AssertionError("persisted coordinator graph was re-executed")

    report = await activate_subnet_epoch_cutover_v1(
        cutover=cutover,
        apply=True,
        select_rows=select_rows,
        load_graph=load_graph,
        preflight_rpc=preflight,
        bind_rpc=bind,
        stage_rpc=stage,
        execute=forbidden,
        load_initialization=load_initialization,
        load_live_snapshot=live_snapshot,
        prepare_initialization=prepare_initialization,
        boot_verifier=lambda _identity: {"verified": True},
        validate_anchor=lambda _cutover: _async_none(),
    )
    assert report["status"] == "stateful_staged"
    assert observed["preflight"] == (
        CUTOVER_PREFLIGHT_RPC,
        {
            "p_mapping_hash": cutover.mapping_hash,
            "p_cutover_receipt_hash": coordinator_graph[
                "root_receipt_hash"
            ],
        },
    )
    assert durable_cutover["cutover_receipt_hash"] == coordinator_graph[
        "root_receipt_hash"
    ]


@pytest.mark.asyncio
async def test_cutover_operator_fails_closed_before_enclave_when_high_water_differs():
    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    measured = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    finalization = {
        "bundle_hash": payload["last_legacy_bundle_hash"],
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_finalization_event_hash": measured.output[
            "last_legacy_weight_finalization_event_hash"
        ],
        "finalization_receipt_hash": context.external_receipt_graphs[1]["root_receipt_hash"],
        "finalized_block": payload["last_legacy_finalization"]["finalized_block"],
        "finalization_doc": payload["last_legacy_finalization"],
    }

    async def select_rows(table, **_kwargs):
        return {
            CUTOVER_TABLE: [],
            CUTOVER_STATE_TABLE: [_fenced_cutover_state(cutover)],
            CANDIDATE_TABLE: [copy.deepcopy(candidate)],
            FINALIZED_ALLOCATION_VIEW: [copy.deepcopy(finalization)],
            RECEIPT_TABLE: [],
        }[table]

    graphs = {
        graph["root_receipt_hash"]: graph for graph in context.external_receipt_graphs
    }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    async def preflight(_name, _params):
        return {
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 99,
            "first_settlement_occupied": False,
            "eligible": False,
        }

    async def forbidden(**_kwargs):
        raise AssertionError("ineligible cutover entered the enclave")

    with pytest.raises(
        StatefulEpochCutoverActivationError,
        match="preflight differs at legacy_high_water",
    ):
        await activate_subnet_epoch_cutover_v1(
            cutover=cutover,
            apply=True,
            select_rows=select_rows,
            load_graph=load_graph,
            preflight_rpc=preflight,
            execute=forbidden,
            validate_anchor=lambda _cutover: _async_none(),
        )


@pytest.mark.asyncio
async def test_cutover_operator_rejects_nonatomic_missing_initialization():
    payload, context, cutover, _candidate, executor = (
        _cutover_activation_dependencies()
    )
    measured = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    graph = _cutover_receipt_graph(
        measured.output,
        context.external_receipt_graphs[0],
        context.external_receipt_graphs[1],
    )
    durable_cutover = build_cutover_row_v1(
        authority_doc=measured.output,
        first_snapshot_doc=payload["first_snapshot"],
        receipt_graph=graph,
    )
    durable_cutover["created_at"] = NOW

    async def select_rows(table, **_kwargs):
        assert table == CUTOVER_TABLE
        return [copy.deepcopy(durable_cutover)]

    async def load_graph(root):
        assert root == graph["root_receipt_hash"]
        return copy.deepcopy(graph)

    initialization = None

    async def load_initialization(_epoch_id):
        return copy.deepcopy(initialization)

    async def forbidden(**_kwargs):
        raise AssertionError("existing cutover re-entered coordinator persistence")

    with pytest.raises(
        StatefulEpochCutoverActivationError,
        match="EPOCH_INITIALIZATION readback is invalid",
    ):
        await activate_subnet_epoch_cutover_v1(
            cutover=cutover,
            apply=True,
            select_rows=select_rows,
            load_graph=load_graph,
            execute=forbidden,
            load_initialization=load_initialization,
            boot_verifier=lambda _identity: {"verified": True},
            validate_anchor=lambda _cutover: _async_none(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "live_index,live_block,last_epoch_block,elapsed",
    [
        (10, 1_301, 1_000, 301),
        (11, 1_360, 1_360, 0),
    ],
)
async def test_cutover_operator_rejects_late_or_next_finalized_epoch_before_enclave(
    live_index,
    live_block,
    last_epoch_block,
    elapsed,
):
    payload, context, cutover, candidate, executor = (
        _cutover_activation_dependencies()
    )
    measured = await executor(
        OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2,
        payload,
        context,
    )
    finalization = {
        "bundle_hash": payload["last_legacy_bundle_hash"],
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "weight_finalization_event_hash": measured.output[
            "last_legacy_weight_finalization_event_hash"
        ],
        "finalization_receipt_hash": context.external_receipt_graphs[1]["root_receipt_hash"],
        "finalized_block": payload["last_legacy_finalization"]["finalized_block"],
        "finalization_doc": payload["last_legacy_finalization"],
    }

    async def select_rows(table, **_kwargs):
        return {
            CUTOVER_TABLE: [],
            CUTOVER_STATE_TABLE: [_fenced_cutover_state(cutover)],
            CANDIDATE_TABLE: [copy.deepcopy(candidate)],
            FINALIZED_ALLOCATION_VIEW: [copy.deepcopy(finalization)],
            RECEIPT_TABLE: [],
        }[table]

    graphs = {
        graph["root_receipt_hash"]: graph for graph in context.external_receipt_graphs
    }

    async def load_graph(root):
        return copy.deepcopy(graphs[root])

    async def preflight(_name, _params):
        return {
            "candidate_snapshot_hash": candidate["snapshot_hash"],
            "candidate_receipt_hash": candidate["chain_state_receipt_hash"],
            "first_settlement_epoch_id": 101,
            "expected_last_legacy_epoch_id": 100,
            "legacy_high_water": 100,
            "first_settlement_occupied": False,
            "eligible": True,
        }

    async def load_initialization(_epoch_id):
        return None

    async def live_snapshot(_cutover):
        return SubnetEpochSnapshot(
            network_genesis_hash=cutover.network_genesis_hash,
            netuid=71,
            head_kind="finalized",
            block_hash="0x" + "8" * 64,
            current_block=live_block,
            last_epoch_block=last_epoch_block,
            pending_epoch_at=0,
            subnet_epoch_index=live_index,
            tempo=360,
            blocks_since_last_step=elapsed,
            observed_at=NOW,
        )

    async def forbidden(**_kwargs):
        raise AssertionError("late cutover entered coordinator or persistence")

    with pytest.raises(StatefulEpochCutoverActivationError) as failure:
        await activate_subnet_epoch_cutover_v1(
            cutover=cutover,
            apply=True,
            select_rows=select_rows,
            load_graph=load_graph,
            preflight_rpc=preflight,
            execute=forbidden,
            load_initialization=load_initialization,
            load_live_snapshot=live_snapshot,
            validate_anchor=lambda _cutover: _async_none(),
        )
    assert failure.value.report["status"] == "ineligible_before_cutover"
