from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import attested_v2_store
from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    SCORING_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    build_transition_command_body,
    create_signed_transition_command,
)
from leadpoet_canonical.sourcing_history_v2 import build_sourcing_epoch_v2


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
NOW = "2026-07-10T20:00:00Z"
LATER = "2026-07-10T20:01:00Z"


def _graph(with_transport=False, with_parent=False):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=COORDINATOR_ROLE,
            physical_role="gateway_coordinator",
            commit_sha="d" * 40,
            pcr0="e" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="1" * 32,
            signing_pubkey=public_key,
            transport_pubkey="2" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )
    attempts = []
    transport_root = EMPTY_TRANSPORT_ROOT
    if with_transport:
        attempt = build_transport_attempt(
            request_id="3" * 32,
            logical_operation_id="provider-operation-1",
            job_id="provider-job-1",
            purpose="research_lab.provider_evidence.v2",
            provider_id="openrouter",
            attempt_number=0,
            method="POST",
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH_B,
            body_hash=HASH_C,
            credential_ref_hash=HASH,
            retry_policy_hash=HASH_B,
            timeout_ms=30000,
            started_at=NOW,
            terminal_status="authenticated_response",
            http_status=503,
            response_hash=HASH_C,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH_B,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at=LATER,
        )
        attempts = [attempt]
        from leadpoet_canonical.attested_v2 import transport_root as calculate_root

        transport_root = calculate_root(attempts)
    purpose = (
        "research_lab.provider_evidence.v2"
        if with_transport
        else "research_lab.admission.v2"
    )
    job_id = "provider-job-1" if with_transport else "admission-job-1"
    parent_receipts = []
    parent_hashes = []
    if with_parent:
        parent = create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role=COORDINATOR_ROLE,
                purpose="research_lab.admission.v2",
                job_id="admission-parent-1",
                epoch_id=10,
                sequence=0,
                commit_sha="d" * 40,
                pcr0="e" * 96,
                build_manifest_hash=HASH,
                dependency_lock_hash=HASH_B,
                config_hash=HASH_C,
                boot_identity_hash=boot["boot_identity_hash"],
                input_root=HASH_C,
                output_root=HASH,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=EMPTY_ARTIFACT_ROOT,
                parent_receipt_hashes=[],
                status="succeeded",
                failure_code=None,
                issued_at=NOW,
            ),
            enclave_pubkey=public_key,
            sign_digest=private_key.sign,
        )
        parent_receipts.append(parent)
        parent_hashes.append(parent["receipt_hash"])
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=COORDINATOR_ROLE,
            purpose=purpose,
            job_id=job_id,
            epoch_id=10,
            sequence=0,
            commit_sha="d" * 40,
            pcr0="e" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=HASH,
            output_root=HASH_B,
            transport_root_hash=transport_root,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=parent_hashes,
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=parent_receipts + [receipt],
        transport_attempts=attempts,
    )


def _sourcing_graph():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    source_doc = build_sourcing_epoch_v2(epoch_id=10, decisions=[])
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=SCORING_ROLE,
            physical_role="gateway_scoring_a",
            commit_sha="d" * 40,
            pcr0="e" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="4" * 32,
            signing_pubkey=public_key,
            transport_pubkey="5" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro-scoring").decode("ascii"),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=SCORING_ROLE,
            purpose="qualification.sourcing_epoch.v2",
            job_id="qualification-sourcing-epoch:10",
            epoch_id=10,
            sequence=0,
            commit_sha="d" * 40,
            pcr0="e" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=HASH,
            output_root=attested_v2_store.sha256_json(source_doc),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=[],
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return source_doc, build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[receipt],
        transport_attempts=[],
    )


def test_v2_storage_rows_preserve_canonical_documents():
    graph = _graph(with_transport=True)
    boot = attested_v2_store.boot_storage_row(graph["boot_identities"][0])
    attempt = attested_v2_store.transport_storage_row(graph["transport_attempts"][0])
    receipt = attested_v2_store.receipt_storage_row(graph["receipts"][0])

    assert boot["identity_doc"] == graph["boot_identities"][0]
    assert boot["attestation_document_hash"].startswith("sha256:")
    assert attempt["http_status"] == 503
    assert attempt["terminal_status"] == "authenticated_response"
    assert attempt["destination_hash"].startswith("sha256:")
    assert receipt["receipt_doc"] == graph["receipts"][0]


@pytest.mark.asyncio
async def test_v2_graph_persists_identity_transport_receipt_then_links(monkeypatch):
    graph = _graph(with_transport=True)
    writes = []
    rows = {}

    async def _insert(table, row):
        writes.append(table)
        key = next(
            row[field]
            for field in (
                "boot_identity_hash",
                "attempt_hash",
                "receipt_hash",
            )
            if field in row
        )
        rows[(table, key)] = dict(row)
        return dict(row)

    async def _select(table, *, filters):
        return rows.get((table, filters[0][1]))

    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _select)

    stored = await attested_v2_store.persist_receipt_graph_v2(graph)

    assert writes == [
        attested_v2_store.BOOT_TABLE,
        attested_v2_store.TRANSPORT_TABLE,
        attested_v2_store.RECEIPT_TABLE,
        attested_v2_store.RECEIPT_TRANSPORT_TABLE,
    ]
    assert stored["boot_count"] == 1
    assert stored["receipt_count"] == 1
    assert stored["transport_attempt_count"] == 1


def _persisted_rows(graph):
    rows = {
        attested_v2_store.BOOT_TABLE: [
            attested_v2_store.boot_storage_row(identity)
            for identity in graph["boot_identities"]
        ],
        attested_v2_store.RECEIPT_TABLE: [
            attested_v2_store.receipt_storage_row(receipt)
            for receipt in graph["receipts"]
        ],
        attested_v2_store.TRANSPORT_TABLE: [
            attested_v2_store.transport_storage_row(attempt)
            for attempt in graph["transport_attempts"]
        ],
        attested_v2_store.EDGE_TABLE: [],
        attested_v2_store.RECEIPT_TRANSPORT_TABLE: [],
        attested_v2_store.HOST_OPERATION_TABLE: [],
    }
    attempts_by_scope = {}
    for attempt in graph["transport_attempts"]:
        attempts_by_scope.setdefault(
            (attempt["job_id"], attempt["purpose"]), []
        ).append(attempt)
    for receipt in graph["receipts"]:
        for parent_hash in receipt["parent_receipt_hashes"]:
            rows[attested_v2_store.EDGE_TABLE].append(
                {
                    "child_receipt_hash": receipt["receipt_hash"],
                    "parent_receipt_hash": parent_hash,
                }
            )
        for attempt in attempts_by_scope.get(
            (receipt["job_id"], receipt["purpose"]), []
        ):
            rows[attested_v2_store.RECEIPT_TRANSPORT_TABLE].append(
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "attempt_hash": attempt["attempt_hash"],
                }
            )
    return rows


@pytest.mark.asyncio
async def test_v2_graph_loader_reconstructs_complete_persisted_ancestry(monkeypatch):
    graph = _graph(with_transport=True, with_parent=True)
    rows = _persisted_rows(graph)

    async def _select_all(table, *, filters, **_kwargs):
        field, operator, values = filters[0]
        assert operator == "in"
        return [
            dict(row)
            for row in rows.get(table, [])
            if row.get(field) in set(values)
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    loaded = await attested_v2_store.load_receipt_graph_v2(
        graph["root_receipt_hash"]
    )

    assert loaded["root_receipt_hash"] == graph["root_receipt_hash"]
    assert {
        receipt["receipt_hash"]: receipt for receipt in loaded["receipts"]
    } == {receipt["receipt_hash"]: receipt for receipt in graph["receipts"]}
    assert loaded["transport_attempts"] == graph["transport_attempts"]


@pytest.mark.asyncio
async def test_v2_graph_loader_rejects_missing_persisted_parent_edge(monkeypatch):
    graph = _graph(with_parent=True)
    rows = _persisted_rows(graph)
    rows[attested_v2_store.EDGE_TABLE] = []

    async def _select_all(table, *, filters, **_kwargs):
        field, _operator, values = filters[0]
        return [
            dict(row)
            for row in rows.get(table, [])
            if row.get(field) in set(values)
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="edges"):
        await attested_v2_store.load_receipt_graph_v2(graph["root_receipt_hash"])


@pytest.mark.asyncio
async def test_sourcing_epoch_persists_graph_before_durable_epoch_row(monkeypatch):
    source_doc, graph = _sourcing_graph()
    writes = []
    rows = {}

    async def _persist_graph(value):
        writes.append("graph")
        assert value == graph
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def _insert(table, row):
        writes.append(table)
        rows[(table, row["epoch_id"])] = dict(row)
        return dict(row)

    async def _select(table, *, filters):
        return rows.get((table, filters[0][1]))

    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", _persist_graph)
    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _select)

    result = await attested_v2_store.persist_sourcing_epoch_v2(
        source_doc=source_doc,
        graph=graph,
    )

    assert writes == ["graph", attested_v2_store.SOURCING_EPOCH_TABLE]
    assert result["epoch_hash"] == source_doc["epoch_hash"]
    assert result["receipt_hash"] == graph["root_receipt_hash"]


@pytest.mark.asyncio
async def test_sourcing_epoch_rejects_receipt_for_different_output(monkeypatch):
    source_doc, graph = _sourcing_graph()
    tampered = dict(source_doc)
    tampered["decision_root"] = HASH_C
    body = {key: value for key, value in tampered.items() if key != "epoch_hash"}
    tampered["epoch_hash"] = attested_v2_store.sha256_json(body)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="does not bind",
    ):
        await attested_v2_store.persist_sourcing_epoch_v2(
            source_doc=tampered,
            graph=graph,
        )


@pytest.mark.asyncio
async def test_duplicate_v2_row_must_match_exactly(monkeypatch):
    row = {"receipt_hash": HASH, "value": "expected"}

    async def _duplicate(_table, _row):
        raise RuntimeError("duplicate key 23505")

    async def _conflicting(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return {"receipt_hash": HASH, "value": "different"}

    monkeypatch.setattr(attested_v2_store, "insert_row", _duplicate)
    monkeypatch.setattr(attested_v2_store, "select_one", _conflicting)

    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="conflicts"):
        await attested_v2_store._insert_exact(
            "example",
            row,
            key_filters=(("receipt_hash", HASH),),
        )


@pytest.mark.asyncio
async def test_weight_bundle_is_acknowledged_only_after_durable_readback(monkeypatch):
    bundle = {"schema_version": "leadpoet.published_weight_bundle.v2"}
    verified = {
        "bundle_hash": HASH,
        "netuid": 71,
        "epoch_id": 10,
        "block": 3600,
        "validator_hotkey": "validator",
        "root_receipt_hash": HASH_B,
        "weights_hash": "c" * 64,
        "snapshot_hash": HASH_C,
    }
    rows = {}

    monkeypatch.setattr(
        attested_v2_store,
        "validate_published_weight_bundle_v2",
        lambda _bundle: dict(verified),
    )

    async def _persist_graph(_graph):
        return {
            "graph_hash": HASH,
            "root_receipt_hash": HASH_B,
            "boot_count": 1,
            "receipt_count": 2,
            "transport_attempt_count": 0,
        }

    async def _insert(table, row):
        rows[(table, row["bundle_hash"])] = dict(row)
        return dict(row)

    async def _select(table, *, filters):
        return rows.get((table, filters[0][1]))

    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", _persist_graph)
    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _select)

    stored = await attested_v2_store.persist_weight_bundle_v2(
        {**bundle, "receipt_graph": {}}
    )

    assert stored["durable_readback_hash"].startswith("sha256:")
    assert stored["bundle_hash"] == HASH


@pytest.mark.asyncio
async def test_missing_durable_readback_fails_closed(monkeypatch):
    monkeypatch.setattr(
        attested_v2_store,
        "validate_published_weight_bundle_v2",
        lambda _bundle: {
            "bundle_hash": HASH,
            "netuid": 71,
            "epoch_id": 10,
            "block": 3600,
            "validator_hotkey": "validator",
            "root_receipt_hash": HASH_B,
            "weights_hash": "c" * 64,
            "snapshot_hash": HASH_C,
        },
    )

    async def _persist_graph(_graph):
        return {}

    async def _insert(_table, row):
        return dict(row)

    async def _missing(_table, *, filters):
        return None

    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", _persist_graph)
    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _missing)

    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="readback"):
        await attested_v2_store.persist_weight_bundle_v2(
            {"schema_version": "leadpoet.published_weight_bundle.v2", "receipt_graph": {}}
        )


@pytest.mark.asyncio
async def test_v2_publication_is_acknowledged_only_after_receipt_and_readback(
    monkeypatch,
):
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": HASH,
        "root_receipt_hash": HASH_B,
        "durable_readback_hash": HASH_C,
        "transparency_event_hash": "sha256:" + "d" * 64,
    }
    publication_receipt = {
        "receipt_hash": "sha256:" + "e" * 64,
        "role": "gateway_coordinator",
        "purpose": "gateway.weights.publication.v2",
        "parent_receipt_hashes": [HASH_B],
        "output_root": attested_v2_store.sha256_json(publication_doc),
    }
    graph = {
        "root_receipt_hash": publication_receipt["receipt_hash"],
        "receipts": [publication_receipt],
    }
    rows = {}

    monkeypatch.setattr(
        attested_v2_store,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: [publication_receipt["receipt_hash"]],
    )

    async def _persist_graph(_graph):
        return {
            "graph_hash": "sha256:" + "f" * 64,
            "root_receipt_hash": publication_receipt["receipt_hash"],
        }

    async def _insert(table, row):
        rows[(table, row["bundle_hash"])] = dict(row)
        return dict(row)

    async def _select(table, *, filters):
        return rows.get((table, filters[0][1]))

    monkeypatch.setattr(attested_v2_store, "persist_receipt_graph_v2", _persist_graph)
    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _select)

    result = await attested_v2_store.persist_weight_publication_v2(
        bundle_result={
            "bundle_hash": HASH,
            "root_receipt_hash": HASH_B,
            "durable_readback_hash": HASH_C,
        },
        publication_graph=graph,
        publication_doc=publication_doc,
    )
    assert result["weight_submission_event_hash"].startswith("sha256:")
    assert result["publication_receipt_hash"] == publication_receipt["receipt_hash"]


@pytest.mark.asyncio
async def test_v2_finalization_requires_publication_bundle_and_durable_readback(
    monkeypatch,
):
    event_hash = "sha256:" + "1" * 64
    bundle_hash = "sha256:" + "2" * 64
    receipt_hash = "sha256:" + "3" * 64
    verified = {
        "validator_hotkey": "validator-hotkey",
        "netuid": 71,
        "epoch_id": 100,
        "weights_hash": "4" * 64,
        "weight_receipt_hash": "sha256:" + "5" * 64,
        "weight_submission_event_hash": event_hash,
        "extrinsic_authorization_hash": "sha256:" + "6" * 64,
        "extrinsic_receipt_hash": "sha256:" + "7" * 64,
        "extrinsic_hash": "0x" + "8" * 64,
        "finalized_block": 36105,
        "finalized_block_hash": "9" * 64,
        "state_transition_hash": "sha256:" + "a" * 64,
        "finalization_receipt_hash": receipt_hash,
    }
    monkeypatch.setattr(
        attested_v2_store,
        "validate_weight_finalization_submission_v2",
        lambda *_args, **_kwargs: dict(verified),
    )
    monkeypatch.setattr(
        attested_v2_store,
        "validate_published_weight_bundle_v2",
        lambda _bundle: {
            "bundle_hash": bundle_hash,
            "validator_hotkey": "validator-hotkey",
            "netuid": 71,
            "epoch_id": 100,
            "weights_hash": "4" * 64,
            "weight_receipt_hash": "sha256:" + "5" * 64,
        },
    )
    async def persist_graph(_graph):
        return {
            "graph_hash": "sha256:" + "b" * 64,
            "root_receipt_hash": receipt_hash,
        }

    monkeypatch.setattr(
        attested_v2_store,
        "persist_receipt_graph_v2",
        persist_graph,
    )
    inserted = {}

    async def insert_exact(table, row, **_kwargs):
        assert table == attested_v2_store.FINALIZATION_TABLE
        inserted.update(row)
        return dict(row)

    async def select_one(table, **_kwargs):
        if table == attested_v2_store.PUBLICATION_TABLE:
            return {
                "weight_submission_event_hash": event_hash,
                "bundle_hash": bundle_hash,
            }
        if table == attested_v2_store.BUNDLE_TABLE:
            return {"bundle_hash": bundle_hash, "bundle_doc": {"bundle": True}}
        if table == attested_v2_store.FINALIZATION_TABLE:
            return dict(inserted)
        raise AssertionError(table)

    monkeypatch.setattr(attested_v2_store, "_insert_exact", insert_exact)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    submission = {
        "receipt_graph": {"root_receipt_hash": receipt_hash},
        "finalization": {"schema_version": "leadpoet.weight_finalization.v2"},
    }
    result = await attested_v2_store.persist_weight_finalization_v2(
        submission=submission
    )
    assert result["bundle_hash"] == bundle_hash
    assert result["weight_finalization_event_hash"].startswith("sha256:")
    assert inserted["state_transition_hash"] == verified["state_transition_hash"]


@pytest.mark.asyncio
async def test_v2_artifact_links_require_compliance_and_exact_readback(monkeypatch):
    writes = []

    async def _insert(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    result = await attested_v2_store.persist_artifact_links_v2(
        receipt_hash=HASH,
        artifacts=[
            {
                "status": "persisted",
                "artifact_kind": "provider_response",
                "artifact_ref": "s3://immutable/artifact.json",
                "artifact_hash": HASH_B,
                "encryption_context_hash": HASH_C,
                "object_lock_mode": "COMPLIANCE",
                "retain_until": "2027-07-10T20:00:00Z",
            }
        ],
    )
    assert writes[0][0] == attested_v2_store.ARTIFACT_TABLE
    assert result["artifact_link_count"] == 1

    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="fields"):
        await attested_v2_store.persist_artifact_links_v2(
            receipt_hash=HASH,
            artifacts=[
                {
                    "status": "persisted",
                    "artifact_kind": "provider_response",
                    "artifact_ref": "s3://immutable/artifact.json",
                    "artifact_hash": HASH_B,
                    "encryption_context_hash": HASH_C,
                    "object_lock_mode": "GOVERNANCE",
                    "retain_until": "2027-07-10T20:00:00Z",
                }
            ],
        )


@pytest.mark.asyncio
async def test_v2_transition_commands_are_signature_checked_and_persisted(monkeypatch):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    command = create_signed_transition_command(
        body=build_transition_command_body(
            operation="apply_result",
            target="research_lab_candidate",
            idempotency_key="candidate-1",
            expected_state_hash=HASH,
            payload_hash=HASH_B,
            receipt_hash=HASH_C,
            issued_at=NOW,
            expires_at=LATER,
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    writes = []

    async def _insert(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    result = await attested_v2_store.persist_transition_commands_v2([command])
    assert writes[0][0] == attested_v2_store.TRANSITION_TABLE
    assert result["transition_count"] == 1

    tampered = {**command, "payload_hash": HASH_C}
    with pytest.raises(Exception):
        await attested_v2_store.persist_transition_commands_v2([tampered])


@pytest.mark.asyncio
async def test_business_artifact_link_is_unique_and_graph_backed(monkeypatch):
    graph = _graph()
    root = graph["root_receipt_hash"]
    inserted = []

    async def load(value):
        assert value == root
        return graph

    async def insert(table, row, *, key_filters):
        inserted.append((table, row, key_filters))
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "load_receipt_graph_v2", load)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", insert)
    result = await attested_v2_store.persist_business_artifact_links_v2(
        receipt_hash=root,
        artifacts=[
            {
                "artifact_kind": "score_bundle",
                "artifact_ref": "score_bundle:" + "a" * 64,
                "artifact_hash": HASH,
            }
        ],
    )
    assert result["business_artifact_link_count"] == 1
    assert inserted[0][0] == attested_v2_store.BUSINESS_ARTIFACT_TABLE
    assert inserted[0][2] == (
        ("artifact_kind", "score_bundle"),
        ("artifact_ref", "score_bundle:" + "a" * 64),
        ("artifact_hash", HASH),
    )


@pytest.mark.asyncio
async def test_business_artifact_lookup_rejects_ambiguous_rows(monkeypatch):
    async def select(*_args, **_kwargs):
        return [
            {"receipt_hash": HASH, "artifact_kind": "score_bundle"},
            {"receipt_hash": HASH_B, "artifact_kind": "score_bundle"},
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", select)
    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="missing or ambiguous",
    ):
        await attested_v2_store.load_business_artifact_graph_v2(
            artifact_kind="score_bundle",
            artifact_ref="score_bundle:" + "a" * 64,
            artifact_hash=HASH,
        )
