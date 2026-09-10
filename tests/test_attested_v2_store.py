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
