from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_exact_duplicate_retries_readback_without_reinserting(monkeypatch):
    inserts = []
    reads = []
    sleeps = []
    expected = {"receipt_hash": HASH, "purpose": "research_lab.allocation.v2"}

    async def insert(_table, row):
        inserts.append(dict(row))
        raise RuntimeError("duplicate key value violates unique constraint 23505")

    async def select(_table, *, filters):
        reads.append(tuple(filters))
        return dict(expected) if len(reads) == 3 else None

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(attested_v2_store, "insert_row", insert)
    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", sleep)

    stored = await attested_v2_store._insert_exact(
        attested_v2_store.RECEIPT_TABLE,
        expected,
        key_filters=(("receipt_hash", HASH),),
    )

    assert stored == expected
    assert inserts == [expected]
    assert len(reads) == 3
    assert sleeps == [0.1, 0.25]


@pytest.mark.asyncio
async def test_exact_duplicate_readback_conflict_still_fails_closed(monkeypatch):
    expected = {"receipt_hash": HASH, "purpose": "research_lab.allocation.v2"}

    async def insert(_table, _row):
        raise RuntimeError("duplicate key value violates unique constraint 23505")

    async def select(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return {**expected, "purpose": "research_lab.reward_decision.v2"}

    monkeypatch.setattr(attested_v2_store, "insert_row", insert)
    monkeypatch.setattr(attested_v2_store, "select_one", select)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="stored row conflicts at purpose",
    ):
        await attested_v2_store._insert_exact(
            attested_v2_store.RECEIPT_TABLE,
            expected,
            key_filters=(("receipt_hash", HASH),),
        )


@pytest.mark.asyncio
async def test_exact_duplicate_readback_retry_is_bounded(monkeypatch):
    inserts = 0
    reads = 0

    async def insert(_table, _row):
        nonlocal inserts
        inserts += 1
        raise RuntimeError("duplicate key value violates unique constraint 23505")

    async def select(_table, *, filters):
        nonlocal reads
        reads += 1
        assert filters == (("receipt_hash", HASH),)
        return None

    async def sleep(_seconds):
        return None

    monkeypatch.setattr(attested_v2_store, "insert_row", insert)
    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", sleep)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="duplicate could not be reloaded after bounded retry",
    ):
        await attested_v2_store._insert_exact(
            attested_v2_store.RECEIPT_TABLE,
            {"receipt_hash": HASH},
            key_filters=(("receipt_hash", HASH),),
        )

    assert inserts == 1
    assert reads == attested_v2_store._DUPLICATE_READBACK_ATTEMPTS


@pytest.mark.asyncio
async def test_exact_rows_use_bounded_postgrest_batches(monkeypatch):
    rows = [
        {"row_id": "row-%03d" % index, "value": index}
        for index in range(250)
    ]
    batches = []

    async def insert_batch(table, values):
        assert table == "example"
        batch = [dict(row) for row in values]
        batches.append(batch)
        return list(reversed(batch))

    async def unexpected_read(*_args, **_kwargs):
        pytest.fail("successful exact batches must not require reconciliation")

    async def unexpected_single_insert(*_args, **_kwargs):
        pytest.fail("multi-row chunks must not fall back to row-at-a-time inserts")

    monkeypatch.setattr(attested_v2_store, "insert_rows", insert_batch)
    monkeypatch.setattr(attested_v2_store, "select_one", unexpected_read)
    monkeypatch.setattr(
        attested_v2_store,
        "_insert_exact",
        unexpected_single_insert,
    )

    await attested_v2_store._insert_exact_rows(
        "example",
        rows,
        key_fields=("row_id",),
    )

    assert [len(batch) for batch in batches] == [100, 100, 50]
    assert [row for batch in batches for row in batch] == rows


@pytest.mark.asyncio
async def test_exact_batch_recovers_unknown_committed_response(monkeypatch):
    rows = [
        {"row_id": "row-a", "value": 1},
        {"row_id": "row-b", "value": 2},
    ]
    durable = {}
    insert_calls = 0

    async def lost_response(_table, values):
        nonlocal insert_calls
        insert_calls += 1
        for row in values:
            durable[row["row_id"]] = dict(row)
        raise ConnectionError("connection reset after batch commit")

    async def select(_table, *, filters):
        return durable.get(dict(filters)["row_id"])

    monkeypatch.setattr(attested_v2_store, "insert_rows", lost_response)
    monkeypatch.setattr(attested_v2_store, "select_one", select)

    await attested_v2_store._insert_exact_batch(
        "example",
        rows,
        key_fields=("row_id",),
    )

    assert insert_calls == 1
    assert durable == {row["row_id"]: row for row in rows}


@pytest.mark.asyncio
async def test_exact_batch_retries_only_missing_concurrent_rows(monkeypatch):
    rows = [
        {"row_id": "row-a", "value": 1},
        {"row_id": "row-b", "value": 2},
        {"row_id": "row-c", "value": 3},
    ]
    durable = {}
    insert_calls = []
    sleeps = []

    async def insert_batch(_table, values):
        batch = [dict(row) for row in values]
        insert_calls.append([row["row_id"] for row in batch])
        if len(insert_calls) == 1:
            durable[batch[0]["row_id"]] = batch[0]
            raise RuntimeError("duplicate key value violates unique constraint 23505")
        for row in batch:
            durable[row["row_id"]] = row
        return list(reversed(batch))

    async def select(_table, *, filters):
        return durable.get(dict(filters)["row_id"])

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(attested_v2_store, "insert_rows", insert_batch)
    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", sleep)

    await attested_v2_store._insert_exact_batch(
        "example",
        rows,
        key_fields=("row_id",),
    )

    assert insert_calls == [["row-a", "row-b", "row-c"], ["row-b", "row-c"]]
    assert sleeps == [0.1]
    assert durable == {row["row_id"]: row for row in rows}


@pytest.mark.asyncio
async def test_exact_batch_conflicting_readback_fails_closed(monkeypatch):
    rows = [
        {"row_id": "row-a", "value": 1},
        {"row_id": "row-b", "value": 2},
    ]

    async def lost_response(_table, _values):
        raise ConnectionError("connection reset after batch commit")

    async def conflicting(_table, *, filters):
        row_id = dict(filters)["row_id"]
        return {"row_id": row_id, "value": 99}

    monkeypatch.setattr(attested_v2_store, "insert_rows", lost_response)
    monkeypatch.setattr(attested_v2_store, "select_one", conflicting)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="stored row conflicts at value",
    ):
        await attested_v2_store._insert_exact_batch(
            "example",
            rows,
            key_fields=("row_id",),
        )
HASH_C = "sha256:" + "c" * 64
NOW = "2026-07-10T20:00:00Z"
LATER = "2026-07-10T20:01:00Z"


def test_chain_settlement_replay_uses_the_live_signed_projection():
    from gateway.tee.coordinator_executor_v2 import (
        coordinator_receipt_output_v2,
    )

    settlement_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": 71,
        "epoch_id": 100,
        "credit_hashes": [],
        "observation_summary": {},
    }
    result = {
        "settlement_doc": settlement_doc,
        "settlement_hash": attested_v2_store.sha256_json(settlement_doc),
        "credits": [],
    }

    assert attested_v2_store._execution_result_projection_v2(
        operation="attest_chain_realized_settlement_v1",
        result=result,
    ) == coordinator_receipt_output_v2(
        "attest_chain_realized_settlement_v1",
        result,
    )


def _chain_settlement_package(credit_count):
    settlement_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": 71,
        "epoch_id": 100,
        "credit_hashes": [],
    }
    credits = []
    for index in range(credit_count):
        credit_doc = {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_obligation_credit.v1"
            ),
            "netuid": 71,
            "epoch_id": 100,
            "obligation_kind": "champion_reward",
            "obligation_source_id": f"reward-{index}",
            "miner_hotkey": f"miner-{index}",
            "miner_uid": index + 1,
            "observed_chain_alpha_percent": "1",
            "lab_attributed_alpha_percent": "1",
            "scheduled_alpha_percent": "1",
            "credited_alpha_percent": "1",
            "attribution_doc": {},
            "observation_doc": {},
        }
        credit_hash = attested_v2_store.sha256_json(credit_doc)
        credits.append(
            {"credit_hash": credit_hash, "credit_doc": credit_doc}
        )
        settlement_doc["credit_hashes"].append(credit_hash)
    settlement_doc["credit_hashes"].sort()
    return {
        "settlement_doc": settlement_doc,
        "settlement_hash": attested_v2_store.sha256_json(settlement_doc),
        "credits": credits,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("credit_count", [0, 2])
async def test_chain_settlement_atomic_retry_and_complete_readback(
    monkeypatch,
    credit_count,
):
    from gateway.research_lab import champion_settlement_v2

    package = _chain_settlement_package(credit_count)
    settlement_hash = package["settlement_hash"]
    calls = []

    async def load_graphs(receipt_hashes):
        assert receipt_hashes == {HASH}
        return {HASH: {"root_receipt_hash": HASH}}

    def validate_settlements(rows, **_kwargs):
        row = rows[0]
        return [
            {
                "netuid": row["netuid"],
                "epoch": row["epoch_id"],
                "settlement_hash": row["settlement_hash"],
                "settlement_doc": row["settlement_doc"],
                "settlement_receipt_hash": row[
                    "settlement_receipt_hash"
                ],
                "credit_hashes": list(
                    row["settlement_doc"]["credit_hashes"]
                ),
            }
        ]

    def validate_credits(rows, **_kwargs):
        return [
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_hash": settlement_hash,
                "chain_realized_settlement_hash": settlement_hash,
                "chain_realized_settlement_receipt_hash": HASH,
                "chain_realized_credit_hashes": sorted(
                    row["credit_hash"] for row in rows
                ),
            }
        ]

    async def call_rpc(_name, parameters):
        calls.append(parameters)
        if len(calls) == 1:
            raise TimeoutError("transient edge timeout")
        return {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_persistence.v1"
            ),
            "netuid": 71,
            "epoch_id": 100,
            "settlement_hash": settlement_hash,
            "settlement_receipt_hash": HASH,
            "credit_count": credit_count,
            "credit_hashes": sorted(
                item["credit_hash"] for item in package["credits"]
            ),
        }

    async def select_one(_table, **_kwargs):
        return {
            "netuid": 71,
            "epoch_id": 100,
            "schema_version": package["settlement_doc"]["schema_version"],
            "settlement_hash": settlement_hash,
            "settlement_receipt_hash": HASH,
            "settlement_doc": package["settlement_doc"],
        }

    async def select_all(_table, **_kwargs):
        return list(calls[-1]["requested_credits"])

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(
        attested_v2_store, "load_receipt_graphs_v2", load_graphs
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(attested_v2_store, "select_all", select_all)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_epoch_settlements_v1",
        validate_settlements,
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_obligation_credits_v1",
        validate_credits,
    )

    result = await attested_v2_store.persist_chain_realized_settlement_v1(
        package=package,
        receipt_hash=HASH,
    )

    assert len(calls) == 2
    assert result["credit_count"] == credit_count
    assert result["durable_readback_hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_lifetime_settlement_uses_versioned_persistence_rpc(monkeypatch):
    from gateway.research_lab import champion_settlement_v2

    settlement_doc = {
        "schema_version": (
            champion_settlement_v2
            .CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
        ),
        "netuid": 71,
        "epoch_id": 100,
        "champion_credit_policy": (
            champion_settlement_v2
            .CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        ),
        "credit_hashes": [],
    }
    package = {
        "settlement_doc": settlement_doc,
        "settlement_hash": attested_v2_store.sha256_json(settlement_doc),
        "credits": [],
    }
    called: list[str] = []

    async def load_graphs(_receipt_hashes):
        return {HASH: {"root_receipt_hash": HASH}}

    def validate_settlements(rows, **_kwargs):
        row = rows[0]
        return [
            {
                "netuid": row["netuid"],
                "epoch": row["epoch_id"],
                "settlement_hash": row["settlement_hash"],
                "settlement_doc": row["settlement_doc"],
                "settlement_receipt_hash": row[
                    "settlement_receipt_hash"
                ],
                "credit_hashes": [],
            }
        ]

    def validate_credits(rows, **_kwargs):
        return [
            {
                "epoch": 100,
                "netuid": 71,
                "allocation_hash": package["settlement_hash"],
                "chain_realized_settlement_hash": package[
                    "settlement_hash"
                ],
                "chain_realized_settlement_receipt_hash": HASH,
                "chain_realized_credit_hashes": sorted(
                    row["credit_hash"] for row in rows
                ),
            }
        ]

    async def call_rpc(name, parameters):
        called.append(name)
        return {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_persistence.v1"
            ),
            "netuid": 71,
            "epoch_id": 100,
            "settlement_hash": package["settlement_hash"],
            "settlement_receipt_hash": HASH,
            "credit_count": 0,
            "credit_hashes": [],
        }

    async def select_one(_table, **_kwargs):
        return {
            "netuid": 71,
            "epoch_id": 100,
            "schema_version": settlement_doc["schema_version"],
            "settlement_hash": package["settlement_hash"],
            "settlement_receipt_hash": HASH,
            "settlement_doc": settlement_doc,
        }

    async def select_all(_table, **_kwargs):
        return []

    monkeypatch.setattr(
        attested_v2_store, "load_receipt_graphs_v2", load_graphs
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(attested_v2_store, "select_all", select_all)
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_epoch_settlements_v1",
        validate_settlements,
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_obligation_credits_v1",
        validate_credits,
    )

    await attested_v2_store.persist_chain_realized_settlement_v1(
        package=package,
        receipt_hash=HASH,
    )

    assert called == [
        attested_v2_store.CHAIN_REALIZED_LIFETIME_SETTLEMENT_RPC
    ]


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


def _replayable_weight_result():
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
            boot_nonce="4" * 32,
            signing_pubkey=public_key,
            transport_pubkey="5" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )
    result = {
        "category": "champions",
        "epoch_id": 10,
        "value": {"rows": []},
    }
    artifacts = [HASH_C]
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=COORDINATOR_ROLE,
            purpose="research_lab.champion_input.v2",
            job_id="scoring-v2:attest-weight-input:" + "1" * 32,
            epoch_id=10,
            sequence=1,
            commit_sha="d" * 40,
            pcr0="e" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=HASH,
            output_root=attested_v2_store.sha256_json(result),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=attested_v2_store.merkle_root(
                artifacts,
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
    )
    return result, artifacts, receipt, graph


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


def test_v2_persistence_derives_parent_first_order_from_validated_membership():
    graph = _graph(with_parent=True)
    parent_hash = graph["receipts"][0]["receipt_hash"]
    child_hash = graph["receipts"][1]["receipt_hash"]

    # Checkpoint certificates expose a canonical membership projection, not a
    # database insertion order. Reproduce the production child-before-parent
    # projection that caused the stateful epoch authority trigger to fail.
    ordered = attested_v2_store._parent_first_receipt_hashes_v2(
        graph,
        validated_receipts=(child_hash, parent_hash),
    )

    assert ordered == (parent_hash, child_hash)


@pytest.mark.asyncio
async def test_v2_graph_persistence_does_not_use_checkpoint_projection_order(
    monkeypatch,
):
    graph = _graph(with_parent=True)
    parent_hash = graph["receipts"][0]["receipt_hash"]
    child_hash = graph["receipts"][1]["receipt_hash"]
    inserted_receipts = []

    monkeypatch.setattr(
        attested_v2_store,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: (child_hash, parent_hash),
    )

    async def _select_all(*_args, **_kwargs):
        return []

    async def _insert(table, row, *, key_filters):
        del key_filters
        if table == attested_v2_store.RECEIPT_TABLE:
            inserted_receipts.append(row["receipt_hash"])
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", _insert)

    await attested_v2_store.persist_receipt_graph_v2(graph)

    assert inserted_receipts == [parent_hash, child_hash]


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

    async def _select_all(_table, *, filters, **_kwargs):
        field, operator, values = filters[0]
        assert field
        assert operator == "in"
        assert isinstance(values, list)
        return []

    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _select)
    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)

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
async def test_v2_edge_value_query_orders_multi_page_results_by_primary_key(
    monkeypatch,
):
    children = [f"child-{index:02d}" for index in range(42)]
    edge_rows = [
        {
            "child_receipt_hash": child,
            "parent_receipt_hash": f"parent-{child}-{parent:02d}",
        }
        for child in children
        for parent in range(28)
    ]
    edge_rows.extend(
        {
            "child_receipt_hash": child,
            "parent_receipt_hash": f"parent-{child}-28",
        }
        for child in children[:16]
    )
    assert len(edge_rows) == 1192

    async def _select_all(
        table,
        *,
        filters,
        order_by,
        max_rows,
        **_kwargs,
    ):
        assert table == attested_v2_store.EDGE_TABLE
        assert filters == (("child_receipt_hash", "in", children),)
        assert order_by == (
            ("child_receipt_hash", False),
            ("parent_receipt_hash", False),
        )
        assert max_rows == 10000
        return list(reversed(edge_rows))

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)

    selected = await attested_v2_store._select_by_values(
        attested_v2_store.EDGE_TABLE,
        field="child_receipt_hash",
        values=children,
        key_fields=("child_receipt_hash", "parent_receipt_hash"),
    )

    assert len(selected) == 1192
    assert len(
        {
            (row["child_receipt_hash"], row["parent_receipt_hash"])
            for row in selected
        }
    ) == 1192


@pytest.mark.asyncio
async def test_v2_existing_exact_rows_pages_large_aggregate_history(
    monkeypatch,
):
    expected_rows = [
        {"attempt_hash": f"attempt-{index:05d}"}
        for index in range(attested_v2_store._MAX_GRAPH_ROWS + 1)
    ]
    expected_by_key = {
        row["attempt_hash"]: dict(row) for row in expected_rows
    }
    queried_chunks = []

    async def _select_all(
        table,
        *,
        filters,
        order_by,
        max_rows,
        **_kwargs,
    ):
        assert table == attested_v2_store.TRANSPORT_TABLE
        field, operator, values = filters[0]
        assert field == "attempt_hash"
        assert operator == "in"
        assert order_by == (("attempt_hash", False),)
        assert max_rows == attested_v2_store._MAX_GRAPH_ROWS
        queried_chunks.append(tuple(values))
        return [expected_by_key[value] for value in values]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)

    existing = await attested_v2_store._existing_exact_rows(
        attested_v2_store.TRANSPORT_TABLE,
        key_field="attempt_hash",
        expected_rows=expected_rows,
    )

    assert existing == set(expected_by_key)
    assert len(queried_chunks) > 1
    assert max(map(len, queried_chunks)) <= attested_v2_store._GRAPH_QUERY_CHUNK


@pytest.mark.asyncio
async def test_v2_value_query_splits_oversized_owner_batch(monkeypatch):
    values = [f"receipt-{index:02d}" for index in range(9)]
    queried_chunks = []

    async def _select_all(_table, *, filters, **_kwargs):
        chunk = tuple(filters[0][2])
        queried_chunks.append(chunk)
        if len(chunk) > 4:
            raise RuntimeError(
                "relation: paginated select exceeded max_rows=10000"
            )
        return [{"receipt_hash": value} for value in chunk]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)

    rows = await attested_v2_store._select_by_values(
        attested_v2_store.RECEIPT_TRANSPORT_TABLE,
        field="receipt_hash",
        values=values,
        key_fields=("receipt_hash",),
        max_total_rows=None,
    )

    assert [row["receipt_hash"] for row in rows] == values
    assert [len(chunk) for chunk in queried_chunks] == [9, 4, 5, 2, 3]


@pytest.mark.asyncio
async def test_v2_value_query_keeps_receipt_ancestry_limit(monkeypatch):
    async def _unexpected_select(*_args, **_kwargs):
        raise AssertionError("oversized ancestry must fail before querying")

    monkeypatch.setattr(
        attested_v2_store,
        "select_all",
        _unexpected_select,
    )

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="V2 receipt graph exceeds row limit",
    ):
        await attested_v2_store._select_by_values(
            attested_v2_store.RECEIPT_TABLE,
            field="receipt_hash",
            values=(
                f"receipt-{index:05d}"
                for index in range(attested_v2_store._MAX_GRAPH_ROWS + 1)
            ),
            key_fields=("receipt_hash",),
        )


@pytest.mark.asyncio
async def test_v2_graph_persistence_batch_verifies_existing_ancestry(monkeypatch):
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

    async def _unexpected_insert(*_args, **_kwargs):
        raise AssertionError("exact existing ancestry must not be reinserted")

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", _unexpected_insert)

    stored = await attested_v2_store.persist_receipt_graph_v2(graph)

    assert stored["root_receipt_hash"] == graph["root_receipt_hash"]
    assert stored["receipt_count"] == 2
    assert stored["transport_attempt_count"] == 1


@pytest.mark.asyncio
async def test_v2_graph_persistence_inserts_only_missing_descendants(monkeypatch):
    graph = _graph(with_transport=True, with_parent=True)
    rows = _persisted_rows(graph)
    parent_hash = graph["receipts"][0]["receipt_hash"]
    rows[attested_v2_store.RECEIPT_TABLE] = [
        row
        for row in rows[attested_v2_store.RECEIPT_TABLE]
        if row["receipt_hash"] == parent_hash
    ]
    rows[attested_v2_store.EDGE_TABLE] = []
    rows[attested_v2_store.RECEIPT_TRANSPORT_TABLE] = []
    inserted = []

    async def _select_all(table, *, filters, **_kwargs):
        field, operator, values = filters[0]
        assert operator == "in"
        return [
            dict(row)
            for row in rows.get(table, [])
            if row.get(field) in set(values)
        ]

    async def _insert(table, row, *, key_filters):
        inserted.append((table, dict(row), tuple(key_filters)))
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", _insert)

    await attested_v2_store.persist_receipt_graph_v2(graph)

    assert [table for table, _row, _filters in inserted] == [
        attested_v2_store.RECEIPT_TABLE,
        attested_v2_store.EDGE_TABLE,
        attested_v2_store.RECEIPT_TRANSPORT_TABLE,
    ]
    assert inserted[0][1]["receipt_hash"] == graph["root_receipt_hash"]


@pytest.mark.asyncio
async def test_v2_graph_persistence_rejects_conflicting_existing_ancestry(
    monkeypatch,
):
    graph = _graph(with_transport=True)
    rows = _persisted_rows(graph)
    rows[attested_v2_store.TRANSPORT_TABLE][0]["response_hash"] = HASH_B

    async def _select_all(table, *, filters, **_kwargs):
        field, operator, values = filters[0]
        assert operator == "in"
        return [
            dict(row)
            for row in rows.get(table, [])
            if row.get(field) in set(values)
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="stored row conflicts at response_hash",
    ):
        await attested_v2_store.persist_receipt_graph_v2(graph)


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
async def test_v2_batch_graph_loader_reuses_shared_ancestry(monkeypatch):
    graph = _graph(with_transport=True, with_parent=True)
    rows = _persisted_rows(graph)
    root = graph["root_receipt_hash"]
    parent = next(
        receipt["receipt_hash"]
        for receipt in graph["receipts"]
        if receipt["receipt_hash"] != root
    )
    receipt_queries = []

    async def _select_all(table, *, filters, **_kwargs):
        field, operator, values = filters[0]
        assert operator == "in"
        if table == attested_v2_store.RECEIPT_TABLE:
            receipt_queries.append(set(values))
        return [
            dict(row)
            for row in rows.get(table, [])
            if row.get(field) in set(values)
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", _select_all)
    loaded = await attested_v2_store.load_receipt_graphs_v2(
        (root, parent)
    )

    assert set(loaded) == {root, parent}
    assert {
        receipt["receipt_hash"] for receipt in loaded[root]["receipts"]
    } == {root, parent}
    assert [
        receipt["receipt_hash"] for receipt in loaded[parent]["receipts"]
    ] == [parent]
    shared_root_receipt = next(
        receipt
        for receipt in loaded[root]["receipts"]
        if receipt["receipt_hash"] == parent
    )
    assert shared_root_receipt is loaded[parent]["receipts"][0]
    assert receipt_queries == [{root, parent}]


@pytest.mark.asyncio
async def test_v2_batch_graph_loader_splits_only_aggregate_row_limit(
    monkeypatch,
):
    roots = tuple("sha256:" + f"{index:064x}" for index in range(1, 5))
    batches = []

    async def _load_batch(root_hashes, *, allowed_failed_receipt_hashes=()):
        normalized = tuple(root_hashes)
        batches.append(normalized)
        assert not tuple(allowed_failed_receipt_hashes)
        if len(normalized) > 1:
            raise attested_v2_store.AttestedV2StoreError(
                "V2 receipt graph exceeds row limit"
            )
        return {
            normalized[0]: {
                "root_receipt_hash": normalized[0],
            }
        }

    async def _no_checkpoints(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(
        attested_v2_store,
        "_load_receipt_graph_batch_v2",
        _load_batch,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_checkpointed_receipt_graphs_v2",
        _no_checkpoints,
    )

    loaded = await attested_v2_store.load_receipt_graphs_v2(roots)

    assert set(loaded) == set(roots)
    assert batches == [
        roots,
        roots[:2],
        roots[:1],
        roots[1:2],
        roots[2:],
        roots[2:3],
        roots[3:],
    ]


@pytest.mark.asyncio
async def test_v2_batch_graph_loader_keeps_single_graph_row_limit_fail_closed(
    monkeypatch,
):
    root = "sha256:" + "1" * 64

    async def _load_batch(_root_hashes, *, allowed_failed_receipt_hashes=()):
        assert not tuple(allowed_failed_receipt_hashes)
        raise attested_v2_store.AttestedV2StoreError(
            "V2 receipt graph exceeds row limit"
        )

    async def _no_checkpoints(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(
        attested_v2_store,
        "_load_receipt_graph_batch_v2",
        _load_batch,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_checkpointed_receipt_graphs_v2",
        _no_checkpoints,
    )

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="V2 receipt graph exceeds row limit",
    ):
        await attested_v2_store.load_receipt_graphs_v2((root,))


@pytest.mark.asyncio
async def test_v2_batch_graph_loader_rejects_shared_failed_allowance():
    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="failed receipt allowance requires one graph root",
    ):
        await attested_v2_store.load_receipt_graphs_v2(
            ("sha256:" + "1" * 64, "sha256:" + "2" * 64),
            allowed_failed_receipt_hashes=("sha256:" + "1" * 64,),
        )


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
async def test_duplicate_v2_row_accepts_equivalent_database_timestamp(
    monkeypatch,
):
    row = {
        "boot_identity_hash": HASH,
        "issued_at": "2026-07-10T20:00:00Z",
        "identity_doc": {"issued_at": "2026-07-10T20:00:00Z"},
    }

    async def _duplicate(_table, _row):
        raise RuntimeError("duplicate key 23505")

    async def _stored(_table, *, filters):
        assert filters == (("boot_identity_hash", HASH),)
        return {
            **row,
            "issued_at": "2026-07-10T20:00:00+00:00",
        }

    monkeypatch.setattr(attested_v2_store, "insert_row", _duplicate)
    monkeypatch.setattr(attested_v2_store, "select_one", _stored)

    stored = await attested_v2_store._insert_exact(
        "example",
        row,
        key_filters=(("boot_identity_hash", HASH),),
    )

    assert stored["identity_doc"] == row["identity_doc"]


@pytest.mark.asyncio
async def test_transient_exact_insert_retries_only_after_absent_readback(
    monkeypatch,
):
    row = {"receipt_hash": HASH, "value": "expected"}
    attempts = 0
    sleeps = []

    class CloudflareEdgeError(RuntimeError):
        code = "400"
        message = "cloudflare: JSON could not be generated"

    async def _insert(_table, _row):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise CloudflareEdgeError("cloudflare: JSON could not be generated")
        return dict(row)

    async def _absent(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return None

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(attested_v2_store, "insert_row", _insert)
    monkeypatch.setattr(attested_v2_store, "select_one", _absent)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", _sleep)

    stored = await attested_v2_store._insert_exact(
        "example",
        row,
        key_filters=(("receipt_hash", HASH),),
    )

    assert stored == row
    assert attempts == 2
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_transient_exact_insert_accepts_only_exact_committed_readback(
    monkeypatch,
):
    row = {"receipt_hash": HASH, "value": "expected"}

    class LostResponseError(ConnectionError):
        pass

    async def _lost_response(_table, _row):
        raise LostResponseError("connection reset after commit")

    async def _stored(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "insert_row", _lost_response)
    monkeypatch.setattr(attested_v2_store, "select_one", _stored)

    stored = await attested_v2_store._insert_exact(
        "example",
        row,
        key_filters=(("receipt_hash", HASH),),
    )

    assert stored == row


@pytest.mark.asyncio
async def test_transient_exact_insert_rejects_conflicting_readback(monkeypatch):
    row = {"receipt_hash": HASH, "value": "expected"}

    async def _lost_response(_table, _row):
        raise ConnectionError("connection reset after commit")

    async def _conflicting(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return {"receipt_hash": HASH, "value": "different"}

    monkeypatch.setattr(attested_v2_store, "insert_row", _lost_response)
    monkeypatch.setattr(attested_v2_store, "select_one", _conflicting)

    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="conflicts"):
        await attested_v2_store._insert_exact(
            "example",
            row,
            key_filters=(("receipt_hash", HASH),),
        )


@pytest.mark.asyncio
async def test_nontransient_exact_insert_is_never_retried(monkeypatch):
    calls = 0

    async def _invalid(_table, _row):
        nonlocal calls
        calls += 1
        raise ValueError("invalid row")

    async def _unexpected_read(*_args, **_kwargs):
        pytest.fail("non-transient insertion must not be reconciled")

    monkeypatch.setattr(attested_v2_store, "insert_row", _invalid)
    monkeypatch.setattr(attested_v2_store, "select_one", _unexpected_read)

    with pytest.raises(ValueError, match="invalid row"):
        await attested_v2_store._insert_exact(
            "example",
            {"receipt_hash": HASH},
            key_filters=(("receipt_hash", HASH),),
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_transient_exact_insert_exhaustion_still_fails_closed(monkeypatch):
    attempts = 0
    sleeps = []

    async def _unavailable(_table, _row):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("connection reset")

    async def _absent(_table, *, filters):
        assert filters == (("receipt_hash", HASH),)
        return None

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(attested_v2_store, "insert_row", _unavailable)
    monkeypatch.setattr(attested_v2_store, "select_one", _absent)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", _sleep)

    with pytest.raises(ConnectionError, match="connection reset"):
        await attested_v2_store._insert_exact(
            "example",
            {"receipt_hash": HASH},
            key_filters=(("receipt_hash", HASH),),
        )

    assert attempts == 4
    assert sleeps == [0.25, 0.75, 1.5]


def test_stored_retention_comparison_accepts_equivalent_database_timestamp():
    attested_v2_store._assert_stored_row(
        "example",
        {"retain_until": "2027-07-10T20:00:00+00:00"},
        {"retain_until": "2027-07-10T20:00:00Z"},
    )


def test_stored_timestamp_comparison_rejects_different_instant():
    with pytest.raises(attested_v2_store.AttestedV2StoreError, match="issued_at"):
        attested_v2_store._assert_stored_row(
            "example",
            {"issued_at": "2026-07-10T20:00:01+00:00"},
            {"issued_at": "2026-07-10T20:00:00Z"},
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
        "status": "succeeded",
        "epoch_id": 10,
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
            "epoch_id": 10,
        },
        publication_graph=graph,
        publication_doc=publication_doc,
    )
    assert result["weight_submission_event_hash"].startswith("sha256:")
    assert result["publication_receipt_hash"] == publication_receipt["receipt_hash"]


@pytest.mark.asyncio
async def test_load_v2_publication_reproves_exact_bundle_parent(monkeypatch):
    bundle_doc = {"schema_version": "leadpoet.published_weight_bundle.v2"}
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
    bundle_row = {
        "bundle_hash": HASH,
        "schema_version": bundle_doc["schema_version"],
        "netuid": 71,
        "epoch_id": 10,
        "block": 3600,
        "validator_hotkey": "validator",
        "root_receipt_hash": HASH_B,
        "weights_hash": "c" * 64,
        "snapshot_hash": HASH_C,
        "bundle_doc": bundle_doc,
    }
    bundle_readback_hash = attested_v2_store.sha256_json(
        {field: bundle_row[field] for field in sorted(bundle_row)}
    )
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": HASH,
        "root_receipt_hash": HASH_B,
        "durable_readback_hash": bundle_readback_hash,
        "transparency_event_hash": "sha256:" + "d" * 64,
    }
    receipt_hash = "sha256:" + "e" * 64
    event_hash = attested_v2_store.sha256_json(
        {
            "bundle_hash": HASH,
            "publication_receipt_hash": receipt_hash,
            "transparency_event_hash": publication_doc[
                "transparency_event_hash"
            ],
            "durable_readback_hash": bundle_readback_hash,
        }
    )
    publication_row = {
        "weight_submission_event_hash": event_hash,
        "bundle_hash": HASH,
        "publication_receipt_hash": receipt_hash,
        "transparency_event_hash": publication_doc[
            "transparency_event_hash"
        ],
        "durable_readback_hash": bundle_readback_hash,
        "publication_doc": publication_doc,
    }
    graph = {
        "root_receipt_hash": receipt_hash,
        "receipts": [
            {
                "receipt_hash": receipt_hash,
                "role": "gateway_coordinator",
                "purpose": "gateway.weights.publication.v2",
                "status": "succeeded",
                "epoch_id": 10,
                "parent_receipt_hashes": [HASH_B],
                "output_root": attested_v2_store.sha256_json(
                    publication_doc
                ),
            }
        ],
    }

    monkeypatch.setattr(
        attested_v2_store,
        "validate_published_weight_bundle_v2",
        lambda _bundle: dict(verified),
    )

    async def select(table, *, filters):
        assert filters == (("bundle_hash", HASH),)
        if table == attested_v2_store.PUBLICATION_TABLE:
            return dict(publication_row)
        if table == attested_v2_store.BUNDLE_TABLE:
            return dict(bundle_row)
        raise AssertionError(table)

    async def load_graph(value):
        assert value == receipt_hash
        return graph

    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store, "load_receipt_graph_v2", load_graph)

    loaded = await attested_v2_store.load_weight_publication_v2(
        bundle_hash=HASH
    )
    assert loaded == publication_row

    graph["receipts"][0]["parent_receipt_hashes"] = [HASH_C]
    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="does not bind its bundle",
    ):
        await attested_v2_store.load_weight_publication_v2(bundle_hash=HASH)


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


@pytest.mark.asyncio
async def test_business_artifact_batch_lookup_loads_all_roots_once(monkeypatch):
    first = ("champion_reward_decision", "champion_reward:1")
    second = ("champion_reward_decision", "champion_reward:2")
    first_root = "sha256:" + "1" * 64
    second_root = "sha256:" + "2" * 64
    rows = [
        {
            "artifact_kind": first[0],
            "artifact_ref": first[1],
            "artifact_hash": "sha256:" + "3" * 64,
            "receipt_hash": first_root,
        },
        {
            "artifact_kind": second[0],
            "artifact_ref": second[1],
            "artifact_hash": "sha256:" + "4" * 64,
            "receipt_hash": second_root,
        },
    ]
    loaded_roots = []

    async def select(_table, *, filters, **_kwargs):
        refs = set(filters[1][2])
        return [dict(row) for row in rows if row["artifact_ref"] in refs]

    async def load_graphs(roots, **_kwargs):
        loaded_roots.append(set(roots))
        return {
            root: {"root_receipt_hash": root}
            for root in roots
        }

    monkeypatch.setattr(attested_v2_store, "select_all", select)
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_graphs,
    )

    result = await attested_v2_store.load_business_artifact_graphs_by_ref_v2(
        (first, second)
    )

    assert set(result) == {first, second}
    assert loaded_roots == [{first_root, second_root}]


@pytest.mark.asyncio
async def test_exact_business_artifact_batch_loads_shared_ancestry_once(
    monkeypatch,
):
    first = (
        "allocation",
        "epoch:100",
        "sha256:" + "3" * 64,
    )
    second = (
        "allocation",
        "epoch:101",
        "sha256:" + "4" * 64,
    )
    first_root = "sha256:" + "1" * 64
    second_root = "sha256:" + "2" * 64
    rows = [
        {
            "artifact_kind": first[0],
            "artifact_ref": first[1],
            "artifact_hash": first[2],
            "receipt_hash": first_root,
        },
        {
            "artifact_kind": second[0],
            "artifact_ref": second[1],
            "artifact_hash": second[2],
            "receipt_hash": second_root,
        },
        {
            "artifact_kind": first[0],
            "artifact_ref": first[1],
            "artifact_hash": "sha256:" + "9" * 64,
            "receipt_hash": "sha256:" + "8" * 64,
        },
    ]
    loaded_roots = []

    async def select(_table, *, filters, **_kwargs):
        refs = set(filters[1][2])
        return [dict(row) for row in rows if row["artifact_ref"] in refs]

    async def load_graphs(roots, **_kwargs):
        loaded_roots.append(set(roots))
        return {
            root: {"root_receipt_hash": root}
            for root in roots
        }

    monkeypatch.setattr(attested_v2_store, "select_all", select)
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graphs_v2",
        load_graphs,
    )

    result = await attested_v2_store.load_business_artifact_graphs_v2(
        (first, second)
    )

    assert set(result) == {first, second}
    assert loaded_roots == [{first_root, second_root}]


@pytest.mark.asyncio
async def test_exact_business_artifact_batch_rejects_noncanonical_stored_hash(
    monkeypatch,
):
    requested = (
        "allocation",
        "epoch:100",
        "sha256:" + "3" * 64,
    )

    async def select(*_args, **_kwargs):
        return [
            {
                "artifact_kind": requested[0],
                "artifact_ref": requested[1],
                "artifact_hash": requested[2].upper(),
                "receipt_hash": "sha256:" + "1" * 64,
            }
        ]

    monkeypatch.setattr(attested_v2_store, "select_all", select)

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="row conflicts",
    ):
        await attested_v2_store.load_business_artifact_graphs_v2((requested,))


@pytest.mark.asyncio
async def test_legacy_settlement_concurrent_retries_persist_one_exact_row(
    monkeypatch,
):
    from leadpoet_canonical import legacy_settlement_v2

    document = {
        "schema_version": "leadpoet.legacy_finalized_allocation.v2",
        "netuid": 71,
        "epoch_id": 100,
        "allocation_hash": HASH_B,
        "settlement_hash": HASH_C,
        "allocation_doc": {"allocation_hash": HASH_B},
    }
    receipt_doc = {
        "receipt_hash": HASH,
        "role": "gateway_coordinator",
        "purpose": "research_lab.legacy_finalized_allocation.v2",
        "status": "succeeded",
        "output_root": attested_v2_store.sha256_json(document),
    }
    stored_row = None
    lock = asyncio.Lock()

    async def insert(table, row):
        nonlocal stored_row
        assert table == attested_v2_store.LEGACY_SETTLEMENT_TABLE
        async with lock:
            if stored_row is not None:
                raise RuntimeError("duplicate key 23505")
            stored_row = dict(row)
            return dict(stored_row)

    async def select(table, *, filters):
        if table == attested_v2_store.RECEIPT_TABLE:
            return {"receipt_doc": receipt_doc}
        assert table == attested_v2_store.LEGACY_SETTLEMENT_TABLE
        assert filters == (("netuid", 71), ("epoch_id", 100))
        return dict(stored_row) if stored_row is not None else None

    monkeypatch.setattr(
        legacy_settlement_v2,
        "validate_legacy_settlement_document_v2",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        attested_v2_store,
        "validate_signed_execution_receipt",
        lambda _value: None,
    )
    monkeypatch.setattr(attested_v2_store, "insert_row", insert)
    monkeypatch.setattr(attested_v2_store, "select_one", select)

    results = await asyncio.gather(
        *(
            attested_v2_store.persist_legacy_finalized_allocation_migration_v2(
                settlement=document,
                receipt_hash=HASH,
            )
            for _index in range(10)
        )
    )
    assert stored_row is not None
    assert len(results) == 10
    assert {result["durable_readback_hash"] for result in results} == {
        results[0]["durable_readback_hash"]
    }


@pytest.mark.asyncio
async def test_legacy_nonfinalization_persists_without_payment_credit(
    monkeypatch,
):
    from leadpoet_canonical import legacy_settlement_v2

    document = {
        "schema_version": "leadpoet.legacy_allocation_nonfinalization.v2",
        "netuid": 71,
        "epoch_id": 100,
        "allocation_hash": HASH_B,
        "finding_hash": HASH_C,
        "allocation_doc": {"allocation_hash": HASH_B},
    }
    receipt_doc = {
        "receipt_hash": HASH,
        "role": "gateway_coordinator",
        "purpose": "research_lab.legacy_finalized_allocation.v2",
        "status": "succeeded",
        "output_root": attested_v2_store.sha256_json(document),
    }
    inserted = []

    async def select(table, *, filters):
        assert table == attested_v2_store.RECEIPT_TABLE
        assert filters == (("receipt_hash", HASH),)
        return {"receipt_doc": receipt_doc}

    async def insert(table, row, *, key_filters):
        inserted.append((table, dict(row), key_filters))
        return dict(row)

    monkeypatch.setattr(
        legacy_settlement_v2,
        "validate_legacy_nonfinalization_document_v2",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        attested_v2_store,
        "validate_signed_execution_receipt",
        lambda _value: None,
    )
    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", insert)

    result = (
        await attested_v2_store.persist_legacy_allocation_nonfinalization_v2(
            finding=document,
            receipt_hash=HASH,
        )
    )

    assert inserted[0][0] == attested_v2_store.LEGACY_NONFINALIZATION_TABLE
    assert inserted[0][2] == (("netuid", 71), ("epoch_id", 100))
    assert result["finding_hash"] == HASH_C


@pytest.mark.asyncio
async def test_replayable_result_requires_durable_receipt_and_exact_readback(
    monkeypatch,
):
    result, artifacts, receipt, graph = _replayable_weight_result()
    release_hash = "sha256:" + "f" * 64
    expected = attested_v2_store._execution_result_storage_row_v2(
        operation="attest_weight_input",
        result=result,
        receipt=receipt,
        artifact_hashes=artifacts,
        release_hash=release_hash,
    )
    events = []

    async def select(table, *, filters):
        events.append(("select", table, filters))
        assert table == attested_v2_store.RECEIPT_TABLE
        return {"receipt_doc": dict(receipt)}

    async def insert(table, row, *, key_filters):
        events.append(("insert", table, key_filters))
        assert row == expected
        return dict(row)

    monkeypatch.setattr(attested_v2_store, "select_one", select)
    monkeypatch.setattr(attested_v2_store, "_insert_exact", insert)
    stored = await attested_v2_store.persist_execution_result_v2(
        operation="attest_weight_input",
        result=result,
        receipt=receipt,
        artifact_hashes=artifacts,
        release_hash=release_hash,
    )
    assert stored == expected
    assert events[0][0:2] == ("select", attested_v2_store.RECEIPT_TABLE)
    assert events[1][0:2] == (
        "insert",
        attested_v2_store.EXECUTION_RESULT_TABLE,
    )

    async def load_row(table, *, filters):
        assert table == attested_v2_store.EXECUTION_RESULT_TABLE
        assert filters == (
            ("role", "gateway_coordinator"),
            ("operation", "attest_weight_input"),
            ("purpose", "research_lab.champion_input.v2"),
            ("job_id", receipt["job_id"]),
        )
        return dict(expected)

    async def load_graph(root_hash):
        assert root_hash == receipt["receipt_hash"]
        return dict(graph)

    rehydrated = []
    expected_receipt = receipt

    async def rehydrate_graph(value, *, receipt):
        assert value == graph
        assert receipt == expected_receipt
        rehydrated.append(receipt["receipt_hash"])
        return dict(value)

    monkeypatch.setattr(attested_v2_store, "select_one", load_row)
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graph_v2",
        load_graph,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "_rehydrate_compact_execution_graph_v2",
        rehydrate_graph,
    )
    loaded = await attested_v2_store.load_execution_result_v2(
        role="gateway_coordinator",
        operation="attest_weight_input",
        purpose="research_lab.champion_input.v2",
        job_id=receipt["job_id"],
    )
    assert loaded["result"] == result
    assert loaded["receipt"] == receipt
    assert loaded["receipt_graph"] == graph
    assert rehydrated == [receipt["receipt_hash"]]


@pytest.mark.asyncio
async def test_allocation_frontier_activation_binds_exact_bootstrap(monkeypatch):
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
        validate_allocation_settlement_frontier_v2,
    )

    bootstrap = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    latest = build_allocation_settlement_frontier_v2(
        mode="bounded_delta_v1",
        netuid=71,
        allocation_epoch=101,
        predecessor_frontier_hash=bootstrap["frontier_hash"],
        reward_checkpoints=(),
    )
    first_receipt = "sha256:" + "1" * 64
    latest_receipt = "sha256:" + "2" * 64
    activation = {
        "schema_version": (
            "leadpoet.research_lab_allocation_settlement_frontier_activation.v2"
        ),
        "netuid": 71,
        "first_allocation_epoch": 100,
        "first_frontier_hash": bootstrap["frontier_hash"],
        "source_receipt_hash": first_receipt,
    }

    def row(frontier, receipt_hash):
        return {
            "netuid": 71,
            "allocation_epoch": frontier["allocation_epoch"],
            "settled_through_epoch": frontier["settled_through_epoch"],
            "schema_version": frontier["schema_version"],
            "frontier_hash": frontier["frontier_hash"],
            "predecessor_frontier_hash": frontier["predecessor_frontier_hash"],
            "source_receipt_hash": receipt_hash,
            "source_state_hash": "sha256:" + "3" * 64,
            "frontier_doc": frontier,
        }

    first_row = row(bootstrap, first_receipt)
    latest_row = row(latest, latest_receipt)

    async def select_one(table, *, filters):
        if table == attested_v2_store.ALLOCATION_SETTLEMENT_FRONTIER_ACTIVATION_TABLE:
            return dict(activation)
        assert table == attested_v2_store.ALLOCATION_SETTLEMENT_FRONTIER_TABLE
        assert filters == (("netuid", 71), ("allocation_epoch", 100))
        return dict(first_row)

    async def select_many(*_args, **_kwargs):
        return [dict(latest_row)]

    loaded_receipts = []

    async def load_source(receipt_hash, **_kwargs):
        loaded_receipts.append(receipt_hash)
        return {"receipt_hash": receipt_hash}

    def validate_row(stored, *, source):
        assert source["receipt_hash"] == stored["source_receipt_hash"]
        return {
            "frontier": validate_allocation_settlement_frontier_v2(
                stored["frontier_doc"]
            ),
            "source": source,
            "row": dict(stored),
        }

    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(attested_v2_store, "select_many", select_many)
    monkeypatch.setattr(
        attested_v2_store,
        "_load_allocation_settlement_frontier_source_v2",
        load_source,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "_validate_allocation_settlement_frontier_storage_v2",
        validate_row,
    )

    loaded = await attested_v2_store.load_allocation_settlement_frontier_context_v2(
        netuid=71,
        before_epoch=102,
    )
    assert loaded["frontier"] == latest
    assert loaded["activation"] == activation
    assert loaded["activation_source"] == {"receipt_hash": first_receipt}
    assert loaded_receipts == [first_receipt, latest_receipt]

    activation["first_frontier_hash"] = "sha256:" + "9" * 64
    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="activation source differs",
    ):
        await attested_v2_store.load_allocation_settlement_frontier_context_v2(
            netuid=71,
            before_epoch=102,
        )


@pytest.mark.asyncio
async def test_allocation_frontier_recovers_committed_rpc_timeout(monkeypatch):
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
    )

    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    receipt_hash = "sha256:" + "1" * 64
    source_state_hash = "sha256:" + "2" * 64
    stored = {
        "netuid": 71,
        "allocation_epoch": 100,
        "settled_through_epoch": 99,
        "schema_version": frontier["schema_version"],
        "frontier_hash": frontier["frontier_hash"],
        "predecessor_frontier_hash": None,
        "source_receipt_hash": receipt_hash,
        "source_state_hash": source_state_hash,
        "frontier_doc": frontier,
    }
    source = {
        "source": "validated",
        "row": {"operation": "research_lab_allocation"},
    }
    rpc_calls = []
    source_loads = []

    async def load_source(value, **_kwargs):
        source_loads.append(value)
        return source

    async def call_rpc(name, payload):
        rpc_calls.append((name, payload))
        raise TimeoutError("response timed out after commit")

    async def select_one(table, *, filters):
        assert table == attested_v2_store.ALLOCATION_SETTLEMENT_FRONTIER_TABLE
        assert filters == (("netuid", 71), ("allocation_epoch", 100))
        return dict(stored)

    def validate_row(row, *, source):
        assert row == stored
        assert source["source"] == "validated"
        return {"frontier": frontier, "source": source, "row": row}

    monkeypatch.setattr(
        attested_v2_store,
        "_load_allocation_settlement_frontier_source_v2",
        load_source,
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(
        attested_v2_store,
        "_validate_allocation_settlement_frontier_storage_v2",
        validate_row,
    )

    result = await attested_v2_store.persist_allocation_settlement_frontier_v2(
        frontier=frontier,
        source_receipt_hash=receipt_hash,
        source_state_hash=source_state_hash,
    )

    assert result == {
        "status": "already_persisted",
        "netuid": 71,
        "allocation_epoch": 100,
        "frontier_hash": frontier["frontier_hash"],
        "source_receipt_hash": receipt_hash,
        "source_state_hash": source_state_hash,
    }
    assert len(rpc_calls) == 1
    assert source_loads == [receipt_hash, receipt_hash]


@pytest.mark.asyncio
async def test_allocation_frontier_retries_uncommitted_rpc_timeout(monkeypatch):
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
    )

    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    receipt_hash = "sha256:" + "1" * 64
    source_state_hash = "sha256:" + "2" * 64
    stored = {
        "netuid": 71,
        "allocation_epoch": 100,
        "settled_through_epoch": 99,
        "schema_version": frontier["schema_version"],
        "frontier_hash": frontier["frontier_hash"],
        "predecessor_frontier_hash": None,
        "source_receipt_hash": receipt_hash,
        "source_state_hash": source_state_hash,
        "frontier_doc": frontier,
    }
    source = {
        "source": "validated",
        "row": {"operation": "research_lab_allocation"},
    }
    rpc_calls = []
    reads = []
    sleeps = []

    async def load_source(*_args, **_kwargs):
        return source

    async def call_rpc(_name, _payload):
        rpc_calls.append(1)
        if len(rpc_calls) == 1:
            raise TimeoutError("request timed out before commit")
        return {
            "status": "persisted",
            "netuid": 71,
            "allocation_epoch": 100,
            "frontier_hash": frontier["frontier_hash"],
            "source_receipt_hash": receipt_hash,
            "source_state_hash": source_state_hash,
        }

    async def select_one(table, *, filters):
        assert table == attested_v2_store.ALLOCATION_SETTLEMENT_FRONTIER_TABLE
        assert filters == (("netuid", 71), ("allocation_epoch", 100))
        reads.append(1)
        return None if len(reads) == 1 else dict(stored)

    async def sleep(seconds):
        sleeps.append(seconds)

    def validate_row(row, *, source):
        assert row == stored
        assert source["source"] == "validated"
        return {"frontier": frontier, "source": source, "row": row}

    monkeypatch.setattr(
        attested_v2_store,
        "_load_allocation_settlement_frontier_source_v2",
        load_source,
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", sleep)
    monkeypatch.setattr(
        attested_v2_store,
        "_validate_allocation_settlement_frontier_storage_v2",
        validate_row,
    )

    result = await attested_v2_store.persist_allocation_settlement_frontier_v2(
        frontier=frontier,
        source_receipt_hash=receipt_hash,
        source_state_hash=source_state_hash,
    )

    assert result["status"] == "persisted"
    assert len(rpc_calls) == 2
    assert len(reads) == 2
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_allocation_frontier_retry_exhaustion_fails_closed(monkeypatch):
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
    )

    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    calls = []
    sleeps = []

    async def load_source(*_args, **_kwargs):
        return {"row": {"operation": "research_lab_allocation"}}

    def validate_row(*_args, **_kwargs):
        return {}

    async def call_rpc(*_args, **_kwargs):
        calls.append(1)
        raise TimeoutError("persistent edge timeout")

    async def select_one(*_args, **_kwargs):
        return None

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        attested_v2_store,
        "_load_allocation_settlement_frontier_source_v2",
        load_source,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "_validate_allocation_settlement_frontier_storage_v2",
        validate_row,
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)
    monkeypatch.setattr(attested_v2_store, "select_one", select_one)
    monkeypatch.setattr(attested_v2_store.asyncio, "sleep", sleep)

    with pytest.raises(TimeoutError, match="persistent edge timeout"):
        await attested_v2_store.persist_allocation_settlement_frontier_v2(
            frontier=frontier,
            source_receipt_hash="sha256:" + "1" * 64,
            source_state_hash="sha256:" + "2" * 64,
        )

    assert len(calls) == attested_v2_store._EXACT_INSERT_ATTEMPTS
    assert sleeps == list(
        attested_v2_store._EXACT_INSERT_BACKOFF_SECONDS
    )


@pytest.mark.asyncio
async def test_allocation_frontier_does_not_retry_contract_failure(monkeypatch):
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
    )

    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    calls = []

    async def load_source(*_args, **_kwargs):
        return {"row": {"operation": "research_lab_allocation"}}

    def validate_row(*_args, **_kwargs):
        return {}

    async def call_rpc(*_args, **_kwargs):
        calls.append(1)
        error = RuntimeError("allocation_settlement_frontier_conflict")
        error.code = "23505"
        raise error

    monkeypatch.setattr(
        attested_v2_store,
        "_load_allocation_settlement_frontier_source_v2",
        load_source,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "_validate_allocation_settlement_frontier_storage_v2",
        validate_row,
    )
    monkeypatch.setattr(attested_v2_store, "call_rpc", call_rpc)

    with pytest.raises(RuntimeError, match="frontier_conflict"):
        await attested_v2_store.persist_allocation_settlement_frontier_v2(
            frontier=frontier,
            source_receipt_hash="sha256:" + "1" * 64,
            source_state_hash="sha256:" + "2" * 64,
        )

    assert calls == [1]
