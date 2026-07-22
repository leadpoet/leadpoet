"""Durable append-only persistence for authoritative V2 attestation records."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Iterable, Mapping, Optional

from gateway.research_lab.store import insert_row, select_all, select_one
from leadpoet_canonical.attested_v2 import (
    build_receipt_graph,
    merkle_root,
    sha256_json,
    validate_boot_identity,
    validate_host_operation_record,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    validate_signed_transition_command,
    validate_transport_attempt,
)
from leadpoet_canonical.sourcing_history_v2 import validate_sourcing_epoch_v2
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
    validate_weight_finalization_submission_v2,
    validate_published_weight_bundle_v2,
)


BOOT_TABLE = "research_lab_attested_boot_identities_v2"
TRANSPORT_TABLE = "research_lab_attested_transport_attempts_v2"
RECEIPT_TABLE = "research_lab_attested_execution_receipts_v2"
EDGE_TABLE = "research_lab_attested_receipt_edges_v2"
RECEIPT_TRANSPORT_TABLE = "research_lab_attested_receipt_transport_v2"
HOST_OPERATION_TABLE = "research_lab_attested_host_operations_v2"
BUNDLE_TABLE = "research_lab_attested_weight_bundles_v2"
PUBLICATION_TABLE = "research_lab_attested_publication_events_v2"
FINALIZATION_TABLE = "research_lab_attested_weight_finalizations_v2"
ARTIFACT_TABLE = "research_lab_attested_artifact_links_v2"
BUSINESS_ARTIFACT_TABLE = "research_lab_attested_business_artifact_links_v2"
EXECUTION_RESULT_TABLE = "research_lab_attested_execution_results_v2"
TRANSITION_TABLE = "research_lab_signed_transition_commands_v2"
SOURCING_EPOCH_TABLE = "validator_sourcing_epoch_inputs_v2"
LEGACY_SETTLEMENT_TABLE = "research_lab_legacy_finalized_allocation_migrations_v2"
LEGACY_NONFINALIZATION_TABLE = (
    "research_lab_legacy_allocation_nonfinalizations_v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GRAPH_QUERY_CHUNK = 50
_MAX_GRAPH_ROWS = 10000
_REPLAYABLE_EXECUTION_PAIRS = frozenset(
    {("research_lab_allocation", "research_lab.allocation.v2")}
    | {
        ("attest_weight_input", purpose)
        for role, purpose in WEIGHT_INPUT_PURPOSES.values()
        if role == "gateway_coordinator"
    }
)


class AttestedV2StoreError(RuntimeError):
    """A V2 append or durable readback failed or conflicted."""


def replayable_execution_result_v2(*, operation: str, purpose: str) -> bool:
    return (str(operation or ""), str(purpose or "")) in _REPLAYABLE_EXECUTION_PAIRS


def _is_duplicate_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "duplicate" in message or "unique" in message or "23505" in message


def _timestamp_instant(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _stored_value_matches(field: str, stored: Any, expected: Any) -> bool:
    if stored == expected:
        return True
    if field.endswith(("_at", "_until")):
        stored_instant = _timestamp_instant(stored)
        expected_instant = _timestamp_instant(expected)
        return (
            stored_instant is not None
            and expected_instant is not None
            and stored_instant == expected_instant
        )
    return False


async def _insert_exact(
    table: str,
    row: Mapping[str, Any],
    *,
    key_filters: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    expected = dict(row)
    try:
        stored = await insert_row(table, expected)
    except Exception as exc:
        if not _is_duplicate_error(exc):
            raise
        stored = await select_one(table, filters=key_filters)
        if not isinstance(stored, Mapping):
            raise AttestedV2StoreError(
                "%s duplicate could not be reloaded" % table
            ) from exc
    for field, value in expected.items():
        if not _stored_value_matches(field, stored.get(field), value):
            raise AttestedV2StoreError(
                "%s stored row conflicts at %s" % (table, field)
            )
    return dict(stored)


def _attestation_document(identity: Mapping[str, Any]) -> tuple[str, str]:
    try:
        document = base64.b64decode(
            str(identity.get("attestation_document_b64") or ""),
            validate=True,
        )
    except Exception as exc:
        raise AttestedV2StoreError("boot attestation is not valid base64") from exc
    if not document:
        raise AttestedV2StoreError("boot attestation is empty")
    digest = "sha256:" + hashlib.sha256(document).hexdigest()
    return "inline:%s" % digest, digest


def boot_storage_row(identity: Mapping[str, Any]) -> dict[str, Any]:
    validate_boot_identity(identity)
    document_ref, document_hash = _attestation_document(identity)
    return {
        "boot_identity_hash": identity["boot_identity_hash"],
        "schema_version": identity["schema_version"],
        "role": identity["role"],
        "physical_role": identity["physical_role"],
        "commit_sha": identity["commit_sha"],
        "pcr0": identity["pcr0"],
        "build_manifest_hash": identity["build_manifest_hash"],
        "dependency_lock_hash": identity["dependency_lock_hash"],
        "config_hash": identity["config_hash"],
        "signing_pubkey": identity["signing_pubkey"],
        "transport_pubkey": identity["transport_pubkey"],
        "transport_certificate_hash": identity["transport_certificate_hash"],
        "boot_nonce": identity["boot_nonce"],
        "attestation_user_data_hash": identity["attestation_user_data_hash"],
        "attestation_document_ref": document_ref,
        "attestation_document_hash": document_hash,
        "identity_doc": dict(identity),
        "issued_at": identity["issued_at"],
    }


def transport_storage_row(attempt: Mapping[str, Any]) -> dict[str, Any]:
    validate_transport_attempt(attempt)
    destination_hash = sha256_json(
        {
            "method": attempt["method"],
            "destination_host": attempt["destination_host"],
            "destination_port": attempt["destination_port"],
            "path_hash": attempt["path_hash"],
        }
    )
    return {
        "attempt_hash": attempt["attempt_hash"],
        "schema_version": attempt["schema_version"],
        "request_id": attempt["request_id"],
        "logical_operation_id": attempt["logical_operation_id"],
        "job_id": attempt["job_id"],
        "purpose": attempt["purpose"],
        "provider_id": attempt["provider_id"],
        "attempt_number": attempt["attempt_number"],
        "request_hash": attempt["request_hash"],
        "destination_hash": destination_hash,
        "terminal_status": attempt["terminal_status"],
        "http_status": attempt["http_status"],
        "response_hash": attempt["response_hash"],
        "request_artifact_hash": attempt["request_artifact_hash"],
        "response_artifact_hash": attempt["response_artifact_hash"],
        "tls_peer_chain_hash": attempt["tls_peer_chain_hash"],
        "failure_code": attempt["failure_code"],
        "attempt_doc": dict(attempt),
        "started_at": attempt["started_at"],
        "completed_at": attempt["completed_at"],
    }


def receipt_storage_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    validate_signed_execution_receipt(receipt)
    return {
        "receipt_hash": receipt["receipt_hash"],
        "schema_version": receipt["schema_version"],
        "role": receipt["role"],
        "purpose": receipt["purpose"],
        "job_id": receipt["job_id"],
        "epoch_id": receipt["epoch_id"],
        "sequence": receipt["sequence"],
        "commit_sha": receipt["commit_sha"],
        "pcr0": receipt["pcr0"],
        "build_manifest_hash": receipt["build_manifest_hash"],
        "dependency_lock_hash": receipt["dependency_lock_hash"],
        "config_hash": receipt["config_hash"],
        "boot_identity_hash": receipt["boot_identity_hash"],
        "input_root": receipt["input_root"],
        "output_root": receipt["output_root"],
        "transport_root": receipt["transport_root"],
        "host_operation_root": receipt["host_operation_root"],
        "artifact_root": receipt["artifact_root"],
        "receipt_status": receipt["status"],
        "failure_code": receipt["failure_code"],
        "enclave_pubkey": receipt["enclave_pubkey"],
        "enclave_signature": receipt["enclave_signature"],
        "receipt_doc": dict(receipt),
        "issued_at": receipt["issued_at"],
    }


def host_operation_storage_row(
    record: Mapping[str, Any], *, receipt_hash: str
) -> dict[str, Any]:
    validate_host_operation_record(record)
    request = record["request"]
    terminal = record["terminal"]
    return {
        "request_hash": request["request_hash"],
        "terminal_hash": terminal["terminal_hash"],
        "receipt_hash": receipt_hash,
        "job_id": request["job_id"],
        "purpose": request["purpose"],
        "operation": request["operation"],
        "sequence": request["sequence"],
        "terminal_status": terminal["terminal_status"],
        "failure_code": terminal["failure_code"],
        "request_doc": dict(request),
        "terminal_doc": dict(terminal),
    }


def _assert_stored_row(
    table: str, stored: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for field, value in expected.items():
        if not _stored_value_matches(field, stored.get(field), value):
            raise AttestedV2StoreError(
                "%s stored row conflicts at %s" % (table, field)
            )


async def _select_by_values(
    table: str,
    *,
    field: str,
    values: Iterable[str],
) -> list[dict[str, Any]]:
    normalized = sorted({str(value) for value in values})
    if len(normalized) > _MAX_GRAPH_ROWS:
        raise AttestedV2StoreError("V2 receipt graph exceeds row limit")
    rows = []
    for offset in range(0, len(normalized), _GRAPH_QUERY_CHUNK):
        chunk = normalized[offset : offset + _GRAPH_QUERY_CHUNK]
        rows.extend(
            await select_all(
                table,
                filters=((field, "in", chunk),),
                max_rows=_MAX_GRAPH_ROWS,
            )
        )
        if len(rows) > _MAX_GRAPH_ROWS:
            raise AttestedV2StoreError("V2 receipt graph exceeds row limit")
    return rows


async def _existing_exact_rows(
    table: str,
    *,
    key_field: str,
    expected_rows: Iterable[Mapping[str, Any]],
) -> set[str]:
    """Return existing keys only after exact durable-row verification."""

    expected_by_key: dict[str, dict[str, Any]] = {}
    for value in expected_rows:
        row = dict(value)
        key = str(row.get(key_field) or "")
        if not key or key in expected_by_key:
            raise AttestedV2StoreError(
                "%s expected row key is missing or duplicated" % table
            )
        expected_by_key[key] = row
    if not expected_by_key:
        return set()

    stored_rows = await _select_by_values(
        table,
        field=key_field,
        values=expected_by_key,
    )
    existing: set[str] = set()
    for stored in stored_rows:
        key = str(stored.get(key_field) or "")
        expected = expected_by_key.get(key)
        if expected is None or key in existing:
            raise AttestedV2StoreError(
                "%s durable row key is unexpected or duplicated" % table
            )
        _assert_stored_row(table, stored, expected)
        existing.add(key)
    return existing


async def _existing_exact_relations(
    table: str,
    *,
    owner_field: str,
    owner_values: Iterable[str],
    key_fields: tuple[str, ...],
    expected_rows: Iterable[Mapping[str, Any]],
) -> set[tuple[str, ...]]:
    """Verify all durable relations for each graph-owned parent key."""

    owners = {str(value) for value in owner_values}
    expected_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for value in expected_rows:
        row = dict(value)
        key = tuple(str(row.get(field) or "") for field in key_fields)
        if (
            not all(key)
            or str(row.get(owner_field) or "") not in owners
            or key in expected_by_key
        ):
            raise AttestedV2StoreError(
                "%s expected relation is invalid or duplicated" % table
            )
        expected_by_key[key] = row
    if not owners:
        if expected_by_key:
            raise AttestedV2StoreError(
                "%s expected relation has no graph owner" % table
            )
        return set()

    stored_rows = await _select_by_values(
        table,
        field=owner_field,
        values=owners,
    )
    existing: set[tuple[str, ...]] = set()
    for stored in stored_rows:
        key = tuple(str(stored.get(field) or "") for field in key_fields)
        expected = expected_by_key.get(key)
        if expected is None or key in existing:
            raise AttestedV2StoreError(
                "%s durable relation is unexpected or duplicated" % table
            )
        _assert_stored_row(table, stored, expected)
        existing.add(key)
    return existing


async def persist_receipt_graph_v2(
    graph: Mapping[str, Any],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Persist a complete graph parent-first; partial retries are idempotent."""

    ordered_receipts = validate_receipt_graph(
        graph,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    boot_by_hash = {
        str(identity["boot_identity_hash"]): identity
        for identity in graph["boot_identities"]
    }
    receipt_by_hash = {
        str(receipt["receipt_hash"]): receipt for receipt in graph["receipts"]
    }
    attempts_by_scope: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    host_operations_by_scope: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for attempt in graph["transport_attempts"]:
        scope = (str(attempt["job_id"]), str(attempt["purpose"]))
        attempts_by_scope.setdefault(scope, []).append(attempt)
    for record in graph["host_operations"]:
        validate_host_operation_record(record)
        request = record["request"]
        scope = (str(request["job_id"]), str(request["purpose"]))
        host_operations_by_scope.setdefault(scope, []).append(record)
    for receipt in graph["receipts"]:
        if str(receipt["boot_identity_hash"]) not in boot_by_hash:
            raise AttestedV2StoreError("receipt boot identity is absent")

    boot_rows = [
        boot_storage_row(identity) for identity in graph["boot_identities"]
    ]
    transport_rows = [
        transport_storage_row(attempt)
        for attempt in graph["transport_attempts"]
    ]
    receipt_rows = [
        receipt_storage_row(receipt_by_hash[receipt_hash])
        for receipt_hash in ordered_receipts
    ]
    edge_rows = [
        {
            "child_receipt_hash": receipt_hash,
            "parent_receipt_hash": parent_hash,
        }
        for receipt_hash in ordered_receipts
        for parent_hash in receipt_by_hash[receipt_hash]["parent_receipt_hashes"]
    ]
    receipt_transport_rows = [
        {
            "receipt_hash": receipt_hash,
            "attempt_hash": attempt["attempt_hash"],
        }
        for receipt_hash in ordered_receipts
        for attempt in attempts_by_scope.get(
            (
                str(receipt_by_hash[receipt_hash]["job_id"]),
                str(receipt_by_hash[receipt_hash]["purpose"]),
            ),
            [],
        )
    ]
    host_operation_rows = []
    for receipt_hash in ordered_receipts:
        receipt = receipt_by_hash[receipt_hash]
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        for record in host_operations_by_scope.pop(scope, []):
            host_operation_rows.append(
                host_operation_storage_row(record, receipt_hash=receipt_hash)
            )
    if host_operations_by_scope:
        raise AttestedV2StoreError(
            "V2 graph contains host operations without a receipt"
        )

    existing_boots = await _existing_exact_rows(
        BOOT_TABLE,
        key_field="boot_identity_hash",
        expected_rows=boot_rows,
    )
    existing_attempts = await _existing_exact_rows(
        TRANSPORT_TABLE,
        key_field="attempt_hash",
        expected_rows=transport_rows,
    )
    existing_receipts = await _existing_exact_rows(
        RECEIPT_TABLE,
        key_field="receipt_hash",
        expected_rows=receipt_rows,
    )
    receipt_hashes = tuple(receipt_by_hash)
    existing_edges = await _existing_exact_relations(
        EDGE_TABLE,
        owner_field="child_receipt_hash",
        owner_values=receipt_hashes,
        key_fields=("child_receipt_hash", "parent_receipt_hash"),
        expected_rows=edge_rows,
    )
    existing_receipt_transports = await _existing_exact_relations(
        RECEIPT_TRANSPORT_TABLE,
        owner_field="receipt_hash",
        owner_values=receipt_hashes,
        key_fields=("receipt_hash", "attempt_hash"),
        expected_rows=receipt_transport_rows,
    )
    existing_host_operations = await _existing_exact_relations(
        HOST_OPERATION_TABLE,
        owner_field="receipt_hash",
        owner_values=receipt_hashes,
        key_fields=("request_hash",),
        expected_rows=host_operation_rows,
    )

    for row in boot_rows:
        if row["boot_identity_hash"] in existing_boots:
            continue
        await _insert_exact(
            BOOT_TABLE,
            row,
            key_filters=(("boot_identity_hash", row["boot_identity_hash"]),),
        )
    for row in transport_rows:
        if row["attempt_hash"] in existing_attempts:
            continue
        await _insert_exact(
            TRANSPORT_TABLE,
            row,
            key_filters=(("attempt_hash", row["attempt_hash"]),),
        )
    for row in receipt_rows:
        if row["receipt_hash"] in existing_receipts:
            continue
        await _insert_exact(
            RECEIPT_TABLE,
            row,
            key_filters=(("receipt_hash", row["receipt_hash"]),),
        )
    for row in edge_rows:
        key = (row["child_receipt_hash"], row["parent_receipt_hash"])
        if key in existing_edges:
            continue
        await _insert_exact(
            EDGE_TABLE,
            row,
            key_filters=(
                ("child_receipt_hash", row["child_receipt_hash"]),
                ("parent_receipt_hash", row["parent_receipt_hash"]),
            ),
        )
    for row in receipt_transport_rows:
        key = (row["receipt_hash"], row["attempt_hash"])
        if key in existing_receipt_transports:
            continue
        await _insert_exact(
            RECEIPT_TRANSPORT_TABLE,
            row,
            key_filters=(
                ("receipt_hash", row["receipt_hash"]),
                ("attempt_hash", row["attempt_hash"]),
            ),
        )
    for row in host_operation_rows:
        key = (row["request_hash"],)
        if key in existing_host_operations:
            continue
        await _insert_exact(
            HOST_OPERATION_TABLE,
            row,
            key_filters=(("request_hash", row["request_hash"]),),
        )
    return {
        "graph_hash": sha256_json(dict(graph)),
        "root_receipt_hash": graph["root_receipt_hash"],
        "boot_count": len(graph["boot_identities"]),
        "receipt_count": len(graph["receipts"]),
        "transport_attempt_count": len(graph["transport_attempts"]),
        "host_operation_count": len(graph["host_operations"]),
    }


async def load_receipt_graphs_v2(
    root_receipt_hashes: Iterable[str],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Reconstruct complete persisted ancestry for many roots in set queries."""

    root_hashes = sorted({str(value or "") for value in root_receipt_hashes})
    if not root_hashes:
        return {}
    if (
        len(root_hashes) > _MAX_GRAPH_ROWS
        or any(not _HASH_RE.fullmatch(value) for value in root_hashes)
    ):
        raise AttestedV2StoreError("V2 graph root receipt hash is invalid")
    allowed_failed = {
        str(value or "") for value in allowed_failed_receipt_hashes
    }
    if any(not _HASH_RE.fullmatch(value) for value in allowed_failed):
        raise AttestedV2StoreError("V2 allowed failed receipt hash is invalid")
    if allowed_failed and len(root_hashes) != 1:
        raise AttestedV2StoreError(
            "V2 failed receipt allowance requires one graph root"
        )

    receipt_docs: dict[str, dict[str, Any]] = {}
    parents_by_child: dict[str, list[str]] = {}
    pending = set(root_hashes)
    while pending:
        requested = set(pending)
        pending.clear()
        rows = await _select_by_values(
            RECEIPT_TABLE,
            field="receipt_hash",
            values=requested,
        )
        by_hash: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            receipt_hash = str(row.get("receipt_hash") or "")
            if receipt_hash in by_hash:
                raise AttestedV2StoreError("V2 receipt row is duplicated")
            by_hash[receipt_hash] = row
        if set(by_hash) != requested:
            raise AttestedV2StoreError("V2 receipt graph is missing a receipt row")

        edge_rows = await _select_by_values(
            EDGE_TABLE,
            field="child_receipt_hash",
            values=requested,
        )
        edge_pairs = set()
        for row in edge_rows:
            child_hash = str(row.get("child_receipt_hash") or "")
            parent_hash = str(row.get("parent_receipt_hash") or "")
            pair = (child_hash, parent_hash)
            if pair in edge_pairs or child_hash not in requested:
                raise AttestedV2StoreError("V2 receipt edge is duplicated or invalid")
            edge_pairs.add(pair)
            parents_by_child[child_hash] = (
                parents_by_child.get(child_hash, []) + [parent_hash]
            )

        for receipt_hash in sorted(requested):
            row = by_hash[receipt_hash]
            document = row.get("receipt_doc")
            if not isinstance(document, Mapping):
                raise AttestedV2StoreError("V2 receipt document is missing")
            normalized = dict(document)
            expected_row = receipt_storage_row(normalized)
            _assert_stored_row(RECEIPT_TABLE, row, expected_row)
            observed_parents = sorted(parents_by_child.get(receipt_hash, []))
            expected_parents = sorted(normalized["parent_receipt_hashes"])
            if observed_parents != expected_parents:
                raise AttestedV2StoreError("V2 persisted receipt edges are incomplete")
            receipt_docs[receipt_hash] = normalized
            pending.update(
                parent_hash
                for parent_hash in expected_parents
                if parent_hash not in receipt_docs
            )
        pending.difference_update(receipt_docs)
        if len(receipt_docs) + len(pending) > _MAX_GRAPH_ROWS:
            raise AttestedV2StoreError("V2 receipt graph exceeds row limit")

    boot_hashes = {
        str(receipt["boot_identity_hash"]) for receipt in receipt_docs.values()
    }
    boot_rows = await _select_by_values(
        BOOT_TABLE,
        field="boot_identity_hash",
        values=boot_hashes,
    )
    boots = {}
    for row in boot_rows:
        identity = row.get("identity_doc")
        if not isinstance(identity, Mapping):
            raise AttestedV2StoreError("V2 boot identity document is missing")
        normalized = dict(identity)
        boot_hash = str(normalized.get("boot_identity_hash") or "")
        if boot_hash in boots:
            raise AttestedV2StoreError("V2 boot identity row is duplicated")
        _assert_stored_row(BOOT_TABLE, row, boot_storage_row(normalized))
        boots[boot_hash] = normalized
    if set(boots) != boot_hashes:
        raise AttestedV2StoreError("V2 receipt graph is missing a boot identity")

    receipt_hashes = set(receipt_docs)
    if not allowed_failed.issubset(receipt_hashes):
        raise AttestedV2StoreError(
            "V2 allowed failed receipt is absent from loaded graphs"
        )
    link_rows = await _select_by_values(
        RECEIPT_TRANSPORT_TABLE,
        field="receipt_hash",
        values=receipt_hashes,
    )
    link_pairs = set()
    attempt_hashes = set()
    attempt_hashes_by_receipt: dict[str, set[str]] = {}
    for row in link_rows:
        pair = (str(row.get("receipt_hash") or ""), str(row.get("attempt_hash") or ""))
        if pair in link_pairs or pair[0] not in receipt_hashes:
            raise AttestedV2StoreError("V2 receipt transport link is duplicated or invalid")
        link_pairs.add(pair)
        attempt_hashes.add(pair[1])
        attempt_hashes_by_receipt.setdefault(pair[0], set()).add(pair[1])
    attempt_rows = await _select_by_values(
        TRANSPORT_TABLE,
        field="attempt_hash",
        values=attempt_hashes,
    )
    attempts = {}
    for row in attempt_rows:
        document = row.get("attempt_doc")
        if not isinstance(document, Mapping):
            raise AttestedV2StoreError("V2 transport attempt document is missing")
        normalized = dict(document)
        attempt_hash = str(normalized.get("attempt_hash") or "")
        if attempt_hash in attempts:
            raise AttestedV2StoreError("V2 transport attempt row is duplicated")
        _assert_stored_row(TRANSPORT_TABLE, row, transport_storage_row(normalized))
        attempts[attempt_hash] = normalized
    if set(attempts) != attempt_hashes:
        raise AttestedV2StoreError("V2 receipt graph is missing a transport attempt")

    host_rows = await _select_by_values(
        HOST_OPERATION_TABLE,
        field="receipt_hash",
        values=receipt_hashes,
    )
    host_operations_by_receipt: dict[str, list[dict[str, Any]]] = {}
    seen_requests = set()
    for row in host_rows:
        request = row.get("request_doc")
        terminal = row.get("terminal_doc")
        if not isinstance(request, Mapping) or not isinstance(terminal, Mapping):
            raise AttestedV2StoreError("V2 host operation document is missing")
        record = {"request": dict(request), "terminal": dict(terminal)}
        request_hash = str(request.get("request_hash") or "")
        if request_hash in seen_requests:
            raise AttestedV2StoreError("V2 host operation row is duplicated")
        seen_requests.add(request_hash)
        expected_row = host_operation_storage_row(
            record,
            receipt_hash=str(row.get("receipt_hash") or ""),
        )
        _assert_stored_row(HOST_OPERATION_TABLE, row, expected_row)
        receipt_hash = str(row.get("receipt_hash") or "")
        if receipt_hash not in receipt_hashes:
            raise AttestedV2StoreError(
                "V2 host operation receipt link is invalid"
            )
        host_operations_by_receipt.setdefault(receipt_hash, []).append(record)

    graphs: dict[str, dict[str, Any]] = {}
    for root_hash in root_hashes:
        closure: set[str] = set()
        graph_pending = {root_hash}
        while graph_pending:
            receipt_hash = graph_pending.pop()
            if receipt_hash in closure:
                continue
            if receipt_hash not in receipt_docs:
                raise AttestedV2StoreError(
                    "V2 receipt graph is missing a receipt row"
                )
            closure.add(receipt_hash)
            graph_pending.update(parents_by_child.get(receipt_hash, ()))
            if len(closure) + len(graph_pending) > _MAX_GRAPH_ROWS:
                raise AttestedV2StoreError("V2 receipt graph exceeds row limit")

        graph_boot_hashes = {
            str(receipt_docs[receipt_hash]["boot_identity_hash"])
            for receipt_hash in closure
        }
        graph_attempt_hashes: set[str] = set()
        graph_host_operations: list[dict[str, Any]] = []
        for receipt_hash in closure:
            graph_attempt_hashes.update(
                attempt_hashes_by_receipt.get(receipt_hash, ())
            )
            graph_host_operations.extend(
                host_operations_by_receipt.get(receipt_hash, ())
            )
        graph_allowed_failed = allowed_failed.intersection(closure)
        graphs[root_hash] = build_receipt_graph(
            root_receipt_hash=root_hash,
            boot_identities=[
                boots[key] for key in sorted(graph_boot_hashes)
            ],
            receipts=[receipt_docs[key] for key in sorted(closure)],
            transport_attempts=[
                attempts[key] for key in sorted(graph_attempt_hashes)
            ],
            host_operations=sorted(
                graph_host_operations,
                key=lambda record: record["request"]["request_hash"],
            ),
            allowed_failed_receipt_hashes=graph_allowed_failed,
        )
    return graphs


async def load_receipt_graph_v2(
    root_receipt_hash: str,
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Reconstruct one complete persisted ancestry graph and reject partial rows."""

    root_hash = str(root_receipt_hash or "")
    graphs = await load_receipt_graphs_v2(
        (root_hash,),
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    return graphs[root_hash]


def _execution_result_projection_v2(
    *,
    operation: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if str(operation) == "research_lab_allocation":
        allocation = result.get("allocation")
        source_state = result.get("source_state")
        source_state_hash = str(result.get("source_state_hash") or "")
        if (
            set(result)
            != {
                "allocation",
                "allocation_inputs",
                "source_state",
                "source_state_hash",
            }
            or not isinstance(allocation, Mapping)
            or not isinstance(result.get("allocation_inputs"), Mapping)
            or not isinstance(source_state, Mapping)
            or source_state_hash != sha256_json(dict(source_state))
        ):
            raise AttestedV2StoreError(
                "replayable allocation result is invalid"
            )
        return {"allocation": dict(allocation)}
    return dict(result)


def _execution_result_storage_row_v2(
    *,
    operation: str,
    result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    artifact_hashes: Iterable[str],
    release_hash: str,
) -> dict[str, Any]:
    validate_signed_execution_receipt(receipt)
    normalized_operation = str(operation or "")
    purpose = str(receipt.get("purpose") or "")
    if not replayable_execution_result_v2(
        operation=normalized_operation,
        purpose=purpose,
    ):
        raise AttestedV2StoreError("execution result purpose is not replayable")
    if receipt.get("role") != "gateway_coordinator" or receipt.get(
        "status"
    ) != "succeeded":
        raise AttestedV2StoreError(
            "replayable execution receipt is not successful coordinator authority"
        )
    normalized_release_hash = str(release_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_release_hash):
        raise AttestedV2StoreError("execution result release hash is invalid")
    normalized_artifacts = sorted(
        {str(item or "").lower() for item in artifact_hashes}
    )
    if any(not _HASH_RE.fullmatch(item) for item in normalized_artifacts):
        raise AttestedV2StoreError("execution result artifact hash is invalid")
    expected_artifact_root = merkle_root(
        normalized_artifacts,
        domain="leadpoet-artifact-v2",
    )
    if receipt.get("artifact_root") != expected_artifact_root:
        raise AttestedV2StoreError(
            "execution result artifacts differ from receipt"
        )
    normalized_result = dict(result)
    from gateway.research_lab.bundles import contains_secret_material

    if contains_secret_material(normalized_result):
        raise AttestedV2StoreError(
            "replayable execution result contains secret material"
        )
    projection = _execution_result_projection_v2(
        operation=normalized_operation,
        result=normalized_result,
    )
    if receipt.get("output_root") != sha256_json(projection):
        raise AttestedV2StoreError(
            "execution result output differs from receipt"
        )
    return {
        "receipt_hash": str(receipt["receipt_hash"]),
        "schema_version": "leadpoet.attested_execution_result.v2",
        "role": str(receipt["role"]),
        "operation": normalized_operation,
        "purpose": purpose,
        "job_id": str(receipt["job_id"]),
        "epoch_id": int(receipt["epoch_id"]),
        "sequence": int(receipt["sequence"]),
        "release_hash": normalized_release_hash,
        "input_root": str(receipt["input_root"]),
        "output_root": str(receipt["output_root"]),
        "artifact_root": str(receipt["artifact_root"]),
        "result_hash": sha256_json(normalized_result),
        "artifact_hashes": normalized_artifacts,
        "result_doc": normalized_result,
    }


async def persist_execution_result_v2(
    *,
    operation: str,
    result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    artifact_hashes: Iterable[str],
    release_hash: str,
) -> dict[str, Any]:
    """Persist one sanitized result only after its execution receipt is durable."""

    row = _execution_result_storage_row_v2(
        operation=operation,
        result=result,
        receipt=receipt,
        artifact_hashes=artifact_hashes,
        release_hash=release_hash,
    )
    stored_receipt = await select_one(
        RECEIPT_TABLE,
        filters=(("receipt_hash", row["receipt_hash"]),),
    )
    receipt_doc = (
        stored_receipt.get("receipt_doc")
        if isinstance(stored_receipt, Mapping)
        else None
    )
    if not isinstance(receipt_doc, Mapping) or dict(receipt_doc) != dict(receipt):
        raise AttestedV2StoreError(
            "execution result receipt is not durably persisted"
        )
    stored = await _insert_exact(
        EXECUTION_RESULT_TABLE,
        row,
        key_filters=(("receipt_hash", row["receipt_hash"]),),
    )
    return {key: stored[key] for key in row}


async def load_execution_result_v2(
    *,
    role: str,
    operation: str,
    purpose: str,
    job_id: str,
) -> Optional[dict[str, Any]]:
    """Load and fully validate one exact same-job result replay."""

    normalized_role = str(role or "")
    normalized_operation = str(operation or "")
    normalized_purpose = str(purpose or "")
    normalized_job_id = str(job_id or "")
    if (
        normalized_role != "gateway_coordinator"
        or not replayable_execution_result_v2(
            operation=normalized_operation,
            purpose=normalized_purpose,
        )
    ):
        return None
    stored = await select_one(
        EXECUTION_RESULT_TABLE,
        filters=(
            ("role", normalized_role),
            ("operation", normalized_operation),
            ("purpose", normalized_purpose),
            ("job_id", normalized_job_id),
        ),
    )
    if not isinstance(stored, Mapping):
        return None
    receipt_hash = str(stored.get("receipt_hash") or "")
    graph = await load_receipt_graph_v2(receipt_hash)
    receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    receipt = receipts.get(receipt_hash)
    result = stored.get("result_doc")
    artifacts = stored.get("artifact_hashes")
    if (
        graph.get("root_receipt_hash") != receipt_hash
        or not isinstance(receipt, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(artifacts, list)
    ):
        raise AttestedV2StoreError(
            "replayable execution result is incomplete"
        )
    expected = _execution_result_storage_row_v2(
        operation=normalized_operation,
        result=result,
        receipt=receipt,
        artifact_hashes=artifacts,
        release_hash=str(stored.get("release_hash") or ""),
    )
    _assert_stored_row(EXECUTION_RESULT_TABLE, stored, expected)
    return {
        "row": expected,
        "result": dict(result),
        "receipt": dict(receipt),
        "receipt_graph": dict(graph),
        "artifact_hashes": list(expected["artifact_hashes"]),
    }


async def persist_sourcing_epoch_v2(
    *, source_doc: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist one measured sourcing aggregate and its complete receipt ancestry."""

    source = validate_sourcing_epoch_v2(source_doc)
    validate_receipt_graph(
        graph,
        required_purposes={"qualification.sourcing_epoch.v2"},
    )
    receipt_by_hash = {
        str(receipt["receipt_hash"]): receipt for receipt in graph["receipts"]
    }
    root_hash = str(graph["root_receipt_hash"])
    root = receipt_by_hash.get(root_hash)
    if (
        not isinstance(root, Mapping)
        or root.get("role") != "gateway_scoring"
        or root.get("purpose") != "qualification.sourcing_epoch.v2"
        or root.get("status") != "succeeded"
        or int(root.get("epoch_id", -1)) != source["epoch_id"]
        or root.get("output_root") != sha256_json(source)
    ):
        raise AttestedV2StoreError(
            "V2 sourcing receipt does not bind the canonical epoch aggregate"
        )

    graph_result = await persist_receipt_graph_v2(graph)
    row = {
        "epoch_id": source["epoch_id"],
        "schema_version": source["schema_version"],
        "epoch_hash": source["epoch_hash"],
        "decision_root": source["decision_root"],
        "receipt_hash": root_hash,
        "source_doc": source,
        "receipt_doc": dict(root),
    }
    await _insert_exact(
        SOURCING_EPOCH_TABLE,
        row,
        key_filters=(("epoch_id", row["epoch_id"]),),
    )
    durable = await select_one(
        SOURCING_EPOCH_TABLE,
        filters=(("epoch_id", row["epoch_id"]),),
    )
    if not isinstance(durable, Mapping):
        raise AttestedV2StoreError("V2 sourcing epoch durable readback returned no row")
    _assert_stored_row(SOURCING_EPOCH_TABLE, durable, row)
    return {
        **graph_result,
        "epoch_id": source["epoch_id"],
        "epoch_hash": source["epoch_hash"],
        "receipt_hash": root_hash,
        "durable_readback_hash": sha256_json(
            {field: durable[field] for field in sorted(row)}
        ),
    }


async def load_sourcing_epoch_graphs_v2(
    *, current_epoch: int, window: int = 30
) -> list[dict[str, Any]]:
    """Load every persisted sourcing graph in the unchanged rolling window."""

    if (
        not isinstance(current_epoch, int)
        or isinstance(current_epoch, bool)
        or current_epoch < 0
        or not isinstance(window, int)
        or isinstance(window, bool)
        or window <= 0
    ):
        raise AttestedV2StoreError("V2 sourcing graph window is invalid")
    if current_epoch == 0:
        return []
    rows = await select_all(
        SOURCING_EPOCH_TABLE,
        filters=(
            ("epoch_id", "gte", max(0, current_epoch - window)),
            ("epoch_id", "lt", current_epoch),
        ),
        order_by=(("epoch_id", False),),
        max_rows=window,
    )
    observed_epochs = set()
    graphs = []
    for row in rows:
        source = validate_sourcing_epoch_v2(row.get("source_doc"))
        epoch_id = source["epoch_id"]
        receipt_hash = str(row.get("receipt_hash") or "")
        receipt = row.get("receipt_doc")
        if (
            epoch_id in observed_epochs
            or int(row.get("epoch_id", -1)) != epoch_id
            or row.get("epoch_hash") != source["epoch_hash"]
            or row.get("decision_root") != source["decision_root"]
            or not isinstance(receipt, Mapping)
            or receipt.get("receipt_hash") != receipt_hash
            or receipt.get("output_root") != sha256_json(source)
        ):
            raise AttestedV2StoreError("V2 sourcing epoch row is inconsistent")
        observed_epochs.add(epoch_id)
        graph = await load_receipt_graph_v2(receipt_hash)
        if graph.get("root_receipt_hash") != receipt_hash:
            raise AttestedV2StoreError("V2 sourcing epoch graph root differs")
        graphs.append(graph)
    return graphs


async def persist_artifact_links_v2(
    *,
    receipt_hash: str,
    artifacts: Any,
) -> dict[str, Any]:
    if not _HASH_RE.fullmatch(str(receipt_hash or "")):
        raise AttestedV2StoreError("V2 artifact receipt hash is invalid")
    normalized = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise AttestedV2StoreError("V2 artifact link is not an object")
        if artifact.get("status") != "persisted":
            raise AttestedV2StoreError("V2 artifact link is not persisted")
        row = {
            "receipt_hash": str(receipt_hash),
            "artifact_kind": str(artifact.get("artifact_kind") or ""),
            "artifact_ref": str(artifact.get("artifact_ref") or ""),
            "artifact_hash": str(artifact.get("artifact_hash") or ""),
            "encryption_context_hash": str(
                artifact.get("encryption_context_hash") or ""
            ),
            "object_lock_mode": str(artifact.get("object_lock_mode") or ""),
            "retain_until": str(artifact.get("retain_until") or ""),
        }
        if (
            not row["artifact_kind"]
            or not row["artifact_ref"].startswith("s3://")
            or not _HASH_RE.fullmatch(row["artifact_hash"])
            or not _HASH_RE.fullmatch(row["encryption_context_hash"])
            or row["object_lock_mode"] != "COMPLIANCE"
            or not row["retain_until"].endswith("Z")
        ):
            raise AttestedV2StoreError("V2 artifact link fields are invalid")
        await _insert_exact(
            ARTIFACT_TABLE,
            row,
            key_filters=(
                ("receipt_hash", row["receipt_hash"]),
                ("artifact_kind", row["artifact_kind"]),
                ("artifact_ref", row["artifact_ref"]),
                ("artifact_hash", row["artifact_hash"]),
            ),
        )
        normalized.append(row)
    return {
        "artifact_link_count": len(normalized),
        "artifact_link_set_hash": sha256_json(normalized),
    }


async def persist_business_artifact_links_v2(
    *,
    receipt_hash: str,
    artifacts: Iterable[Mapping[str, Any]],
    allow_failed_root: bool = False,
) -> dict[str, Any]:
    """Bind existing immutable business artifacts to one verified V2 root."""

    root_hash = str(receipt_hash or "").lower()
    if not _HASH_RE.fullmatch(root_hash):
        raise AttestedV2StoreError("V2 business artifact receipt hash is invalid")
    if allow_failed_root:
        await load_receipt_graph_v2(
            root_hash,
            allowed_failed_receipt_hashes=(root_hash,),
        )
    else:
        await load_receipt_graph_v2(root_hash)
    normalized = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "artifact_kind",
            "artifact_ref",
            "artifact_hash",
        }:
            raise AttestedV2StoreError("V2 business artifact link fields are invalid")
        row = {
            "receipt_hash": root_hash,
            "artifact_kind": str(artifact.get("artifact_kind") or "").strip(),
            "artifact_ref": str(artifact.get("artifact_ref") or "").strip(),
            "artifact_hash": str(artifact.get("artifact_hash") or "").lower(),
        }
        if (
            not row["artifact_kind"]
            or not row["artifact_ref"]
            or not _HASH_RE.fullmatch(row["artifact_hash"])
        ):
            raise AttestedV2StoreError("V2 business artifact link is invalid")
        await _insert_exact(
            BUSINESS_ARTIFACT_TABLE,
            row,
            key_filters=(
                ("artifact_kind", row["artifact_kind"]),
                ("artifact_ref", row["artifact_ref"]),
                ("artifact_hash", row["artifact_hash"]),
            ),
        )
        normalized.append(row)
    normalized.sort(
        key=lambda item: (
            item["artifact_kind"],
            item["artifact_ref"],
            item["artifact_hash"],
        )
    )
    return {
        "business_artifact_link_count": len(normalized),
        "business_artifact_link_set_hash": sha256_json(normalized),
    }


async def load_business_artifact_graph_v2(
    *,
    artifact_kind: str,
    artifact_ref: str,
    artifact_hash: str,
    allow_failed_root: bool = False,
) -> dict[str, Any]:
    """Resolve exactly one V2 receipt graph for an immutable business artifact."""

    kind = str(artifact_kind or "").strip()
    ref = str(artifact_ref or "").strip()
    digest = str(artifact_hash or "").lower()
    if not kind or not ref or not _HASH_RE.fullmatch(digest):
        raise AttestedV2StoreError("V2 business artifact lookup is invalid")
    rows = await select_all(
        BUSINESS_ARTIFACT_TABLE,
        filters=(
            ("artifact_kind", kind),
            ("artifact_ref", ref),
            ("artifact_hash", digest),
        ),
    )
    if len(rows) != 1:
        raise AttestedV2StoreError(
            "V2 business artifact lineage is missing or ambiguous"
        )
    row = rows[0]
    for field, expected in (
        ("artifact_kind", kind),
        ("artifact_ref", ref),
        ("artifact_hash", digest),
    ):
        if row.get(field) != expected:
            raise AttestedV2StoreError("V2 business artifact row conflicts")
    receipt_hash = str(row.get("receipt_hash") or "")
    if allow_failed_root:
        graph = await load_receipt_graph_v2(
            receipt_hash,
            allowed_failed_receipt_hashes=(receipt_hash,),
        )
    else:
        graph = await load_receipt_graph_v2(receipt_hash)
    if graph.get("root_receipt_hash") != row.get("receipt_hash"):
        raise AttestedV2StoreError("V2 business artifact graph root differs")
    return graph


async def load_business_artifact_graph_by_ref_v2(
    *,
    artifact_kind: str,
    artifact_ref: str,
    allow_failed_root: bool = False,
) -> dict[str, Any]:
    """Resolve a business artifact only when its kind/ref mapping is immutable."""

    kind = str(artifact_kind or "").strip()
    ref = str(artifact_ref or "").strip()
    if not kind or not ref:
        raise AttestedV2StoreError("V2 business artifact reference is invalid")
    graphs = await load_business_artifact_graphs_by_ref_v2(
        ((kind, ref),),
        allow_failed_root=allow_failed_root,
    )
    return graphs[(kind, ref)]


async def load_business_artifact_graphs_by_ref_v2(
    artifacts: Iterable[tuple[str, str]],
    *,
    allow_failed_root: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Resolve immutable business-artifact graphs with bounded set queries."""

    requested = sorted(
        {
            (str(kind or "").strip(), str(ref or "").strip())
            for kind, ref in artifacts
        }
    )
    if not requested:
        return {}
    if (
        len(requested) > _MAX_GRAPH_ROWS
        or any(not kind or not ref for kind, ref in requested)
    ):
        raise AttestedV2StoreError("V2 business artifact reference is invalid")

    requested_set = set(requested)
    refs_by_kind: dict[str, list[str]] = {}
    for kind, ref in requested:
        refs_by_kind.setdefault(kind, []).append(ref)

    rows_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for kind in sorted(refs_by_kind):
        refs = sorted(set(refs_by_kind[kind]))
        for offset in range(0, len(refs), _GRAPH_QUERY_CHUNK):
            chunk = refs[offset : offset + _GRAPH_QUERY_CHUNK]
            rows = await select_all(
                BUSINESS_ARTIFACT_TABLE,
                filters=(
                    ("artifact_kind", kind),
                    ("artifact_ref", "in", chunk),
                ),
                max_rows=_MAX_GRAPH_ROWS,
                allow_partial=False,
            )
            for row in rows:
                key = (
                    str(row.get("artifact_kind") or ""),
                    str(row.get("artifact_ref") or ""),
                )
                if key not in requested_set or key in rows_by_key:
                    raise AttestedV2StoreError(
                        "V2 business artifact reference is missing or ambiguous"
                    )
                digest = str(row.get("artifact_hash") or "").lower()
                receipt_hash = str(row.get("receipt_hash") or "").lower()
                if (
                    not _HASH_RE.fullmatch(digest)
                    or not _HASH_RE.fullmatch(receipt_hash)
                ):
                    raise AttestedV2StoreError(
                        "V2 business artifact reference hash is invalid"
                    )
                rows_by_key[key] = row

    if set(rows_by_key) != requested_set:
        raise AttestedV2StoreError(
            "V2 business artifact reference is missing or ambiguous"
        )
    receipt_hashes = {
        str(row["receipt_hash"]).lower() for row in rows_by_key.values()
    }
    if allow_failed_root:
        graphs = {
            receipt_hash: await load_receipt_graph_v2(
                receipt_hash,
                allowed_failed_receipt_hashes=(receipt_hash,),
            )
            for receipt_hash in sorted(receipt_hashes)
        }
    else:
        graphs = await load_receipt_graphs_v2(receipt_hashes)
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in rows_by_key.items():
        receipt_hash = str(row["receipt_hash"]).lower()
        graph = graphs.get(receipt_hash)
        if (
            not isinstance(graph, Mapping)
            or graph.get("root_receipt_hash") != receipt_hash
        ):
            raise AttestedV2StoreError(
                "V2 business artifact graph root differs"
            )
        resolved[key] = dict(graph)
    return resolved


async def persist_transition_commands_v2(commands: Any) -> dict[str, Any]:
    normalized = []
    for command in commands:
        validate_signed_transition_command(command)
        row = {
            "command_hash": command["command_hash"],
            "schema_version": command["schema_version"],
            "operation": command["operation"],
            "target": command["target"],
            "idempotency_key": command["idempotency_key"],
            "expected_state_hash": command["expected_state_hash"],
            "payload_hash": command["payload_hash"],
            "receipt_hash": command["receipt_hash"],
            "enclave_pubkey": command["enclave_pubkey"],
            "enclave_signature": command["enclave_signature"],
            "command_doc": dict(command),
            "issued_at": command["issued_at"],
            "expires_at": command["expires_at"],
        }
        await _insert_exact(
            TRANSITION_TABLE,
            row,
            key_filters=(("command_hash", row["command_hash"]),),
        )
        normalized.append(dict(command))
    return {
        "transition_count": len(normalized),
        "transition_set_hash": sha256_json(normalized),
    }


async def persist_execution_sidecars_v2(
    *,
    artifact_receipt_hash: str,
    artifacts: Any,
    transitions: Any,
) -> dict[str, Any]:
    artifact_result = await persist_artifact_links_v2(
        receipt_hash=artifact_receipt_hash,
        artifacts=artifacts,
    )
    transition_result = await persist_transition_commands_v2(transitions)
    return {**artifact_result, **transition_result}


async def persist_legacy_finalized_allocation_migration_v2(
    *,
    settlement: Mapping[str, Any],
    receipt_hash: str,
) -> dict[str, Any]:
    """Persist one measured pre-V2 settlement with exact duplicate recovery."""

    from leadpoet_canonical.legacy_settlement_v2 import (
        validate_legacy_settlement_document_v2,
    )

    document = validate_legacy_settlement_document_v2(settlement)
    normalized_receipt_hash = str(receipt_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_receipt_hash):
        raise AttestedV2StoreError("legacy settlement receipt hash is invalid")
    stored_receipt = await select_one(
        RECEIPT_TABLE,
        filters=(("receipt_hash", normalized_receipt_hash),),
    )
    receipt_doc = (
        stored_receipt.get("receipt_doc")
        if isinstance(stored_receipt, Mapping)
        else None
    )
    if not isinstance(receipt_doc, Mapping):
        raise AttestedV2StoreError("legacy settlement receipt is not durable")
    validate_signed_execution_receipt(receipt_doc)
    if (
        receipt_doc.get("receipt_hash") != normalized_receipt_hash
        or receipt_doc.get("role") != "gateway_coordinator"
        or receipt_doc.get("purpose")
        != "research_lab.legacy_finalized_allocation.v2"
        or receipt_doc.get("status") != "succeeded"
        or receipt_doc.get("output_root") != sha256_json(document)
    ):
        raise AttestedV2StoreError("legacy settlement receipt differs")
    row = {
        "netuid": int(document["netuid"]),
        "epoch_id": int(document["epoch_id"]),
        "schema_version": str(document["schema_version"]),
        "allocation_hash": str(document["allocation_hash"]),
        "settlement_hash": str(document["settlement_hash"]),
        "settlement_receipt_hash": normalized_receipt_hash,
        "allocation_doc": dict(document["allocation_doc"]),
        "settlement_doc": dict(document),
    }
    stored = await _insert_exact(
        LEGACY_SETTLEMENT_TABLE,
        row,
        key_filters=(
            ("netuid", row["netuid"]),
            ("epoch_id", row["epoch_id"]),
        ),
    )
    return {
        "schema_version": document["schema_version"],
        "netuid": row["netuid"],
        "epoch_id": row["epoch_id"],
        "allocation_hash": row["allocation_hash"],
        "settlement_hash": row["settlement_hash"],
        "settlement_receipt_hash": normalized_receipt_hash,
        "durable_readback_hash": sha256_json(
            {key: stored[key] for key in row}
        ),
    }


async def persist_legacy_allocation_nonfinalization_v2(
    *,
    finding: Mapping[str, Any],
    receipt_hash: str,
) -> dict[str, Any]:
    """Persist proof that one signed legacy allocation was not paid on chain."""

    from leadpoet_canonical.legacy_settlement_v2 import (
        validate_legacy_nonfinalization_document_v2,
    )

    document = validate_legacy_nonfinalization_document_v2(finding)
    normalized_receipt_hash = str(receipt_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_receipt_hash):
        raise AttestedV2StoreError(
            "legacy nonfinalization receipt hash is invalid"
        )
    stored_receipt = await select_one(
        RECEIPT_TABLE,
        filters=(("receipt_hash", normalized_receipt_hash),),
    )
    receipt_doc = (
        stored_receipt.get("receipt_doc")
        if isinstance(stored_receipt, Mapping)
        else None
    )
    if not isinstance(receipt_doc, Mapping):
        raise AttestedV2StoreError(
            "legacy nonfinalization receipt is not durable"
        )
    validate_signed_execution_receipt(receipt_doc)
    if (
        receipt_doc.get("receipt_hash") != normalized_receipt_hash
        or receipt_doc.get("role") != "gateway_coordinator"
        or receipt_doc.get("purpose")
        != "research_lab.legacy_finalized_allocation.v2"
        or receipt_doc.get("status") != "succeeded"
        or receipt_doc.get("output_root") != sha256_json(document)
    ):
        raise AttestedV2StoreError(
            "legacy nonfinalization receipt differs"
        )
    row = {
        "netuid": int(document["netuid"]),
        "epoch_id": int(document["epoch_id"]),
        "schema_version": str(document["schema_version"]),
        "allocation_hash": str(document["allocation_hash"]),
        "finding_hash": str(document["finding_hash"]),
        "finding_receipt_hash": normalized_receipt_hash,
        "allocation_doc": dict(document["allocation_doc"]),
        "finding_doc": dict(document),
    }
    stored = await _insert_exact(
        LEGACY_NONFINALIZATION_TABLE,
        row,
        key_filters=(
            ("netuid", row["netuid"]),
            ("epoch_id", row["epoch_id"]),
        ),
    )
    return {
        "schema_version": document["schema_version"],
        "netuid": row["netuid"],
        "epoch_id": row["epoch_id"],
        "allocation_hash": row["allocation_hash"],
        "finding_hash": row["finding_hash"],
        "finding_receipt_hash": normalized_receipt_hash,
        "durable_readback_hash": sha256_json(
            {key: stored[key] for key in row}
        ),
    }


async def persist_weight_bundle_v2(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Persist and read back an authoritative bundle before it can be acknowledged."""

    verified = validate_published_weight_bundle_v2(bundle)
    graph_result = await persist_receipt_graph_v2(bundle["receipt_graph"])
    row = {
        "bundle_hash": verified["bundle_hash"],
        "schema_version": bundle["schema_version"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "block": verified["block"],
        "validator_hotkey": verified["validator_hotkey"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "weights_hash": verified["weights_hash"],
        "snapshot_hash": verified["snapshot_hash"],
        "bundle_doc": dict(bundle),
    }
    await _insert_exact(
        BUNDLE_TABLE,
        row,
        key_filters=(("bundle_hash", row["bundle_hash"]),),
    )
    durable = await select_one(
        BUNDLE_TABLE,
        filters=(("bundle_hash", row["bundle_hash"]),),
    )
    if not isinstance(durable, Mapping):
        raise AttestedV2StoreError("V2 bundle durable readback returned no row")
    for field, value in row.items():
        if durable.get(field) != value:
            raise AttestedV2StoreError(
                "V2 bundle durable readback conflicts at %s" % field
            )
    durable_readback_hash = sha256_json(
        {field: durable[field] for field in sorted(row)}
    )
    return {
        **verified,
        **graph_result,
        "durable_readback_hash": durable_readback_hash,
    }


async def load_weight_bundle_v2(
    *, netuid: int, epoch_id: int, validator_hotkey: str
) -> dict[str, Any] | None:
    row = await select_one(
        BUNDLE_TABLE,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", int(epoch_id)),
            ("validator_hotkey", str(validator_hotkey)),
        ),
    )
    if not isinstance(row, Mapping):
        return None
    bundle = row.get("bundle_doc")
    if not isinstance(bundle, Mapping):
        raise AttestedV2StoreError("stored V2 bundle document is missing")
    verified = validate_published_weight_bundle_v2(bundle)
    expected = {
        "bundle_hash": verified["bundle_hash"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "block": verified["block"],
        "validator_hotkey": verified["validator_hotkey"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "weights_hash": verified["weights_hash"],
        "snapshot_hash": verified["snapshot_hash"],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise AttestedV2StoreError("stored V2 bundle conflicts at %s" % field)
    return dict(bundle)


async def persist_weight_publication_v2(
    *,
    bundle_result: Mapping[str, Any],
    publication_graph: Mapping[str, Any],
    publication_doc: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the coordinator publication receipt and final authority event."""

    required_bundle_fields = {
        "bundle_hash",
        "root_receipt_hash",
        "durable_readback_hash",
        "epoch_id",
    }
    if not isinstance(bundle_result, Mapping) or not required_bundle_fields <= set(
        bundle_result
    ):
        raise AttestedV2StoreError("V2 bundle persistence result is incomplete")
    expected_publication = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": str(bundle_result["bundle_hash"]),
        "root_receipt_hash": str(bundle_result["root_receipt_hash"]),
        "durable_readback_hash": str(bundle_result["durable_readback_hash"]),
        "transparency_event_hash": str(
            publication_doc.get("transparency_event_hash") or ""
        ),
    }
    if dict(publication_doc) != expected_publication or any(
        not _HASH_RE.fullmatch(str(expected_publication[field] or ""))
        for field in (
            "bundle_hash",
            "root_receipt_hash",
            "durable_readback_hash",
            "transparency_event_hash",
        )
    ):
        raise AttestedV2StoreError("V2 publication document is invalid")
    validate_receipt_graph(
        publication_graph,
        required_purposes={"gateway.weights.publication.v2"},
    )
    receipt_by_hash = {
        str(receipt["receipt_hash"]): receipt
        for receipt in publication_graph["receipts"]
    }
    root_hash = str(publication_graph["root_receipt_hash"])
    root_receipt = receipt_by_hash.get(root_hash)
    if (
        not isinstance(root_receipt, Mapping)
        or root_receipt.get("role") != "gateway_coordinator"
        or root_receipt.get("purpose") != "gateway.weights.publication.v2"
        or root_receipt.get("status") != "succeeded"
        or int(root_receipt.get("epoch_id", -1))
        != int(bundle_result["epoch_id"])
        or root_receipt.get("parent_receipt_hashes")
        != [bundle_result["root_receipt_hash"]]
        or root_receipt.get("output_root") != sha256_json(expected_publication)
    ):
        raise AttestedV2StoreError(
            "V2 publication receipt does not bind the durable bundle"
        )
    graph_result = await persist_receipt_graph_v2(publication_graph)
    event_hash = sha256_json(
        {
            "bundle_hash": bundle_result["bundle_hash"],
            "publication_receipt_hash": root_hash,
            "transparency_event_hash": expected_publication[
                "transparency_event_hash"
            ],
            "durable_readback_hash": bundle_result["durable_readback_hash"],
        }
    )
    row = {
        "weight_submission_event_hash": event_hash,
        "bundle_hash": bundle_result["bundle_hash"],
        "publication_receipt_hash": root_hash,
        "transparency_event_hash": expected_publication[
            "transparency_event_hash"
        ],
        "durable_readback_hash": bundle_result["durable_readback_hash"],
        "publication_doc": expected_publication,
    }
    await _insert_exact(
        PUBLICATION_TABLE,
        row,
        key_filters=(("bundle_hash", row["bundle_hash"]),),
    )
    durable = await select_one(
        PUBLICATION_TABLE,
        filters=(("bundle_hash", row["bundle_hash"]),),
    )
    if not isinstance(durable, Mapping):
        raise AttestedV2StoreError(
            "V2 publication durable readback returned no row"
        )
    for field, value in row.items():
        if durable.get(field) != value:
            raise AttestedV2StoreError(
                "V2 publication durable readback conflicts at %s" % field
            )
    return {
        **graph_result,
        "weight_submission_event_hash": event_hash,
        "publication_receipt_hash": root_hash,
    }


async def load_weight_publication_v2(
    *, bundle_hash: str
) -> dict[str, Any] | None:
    """Read back and re-prove one durable publication and its exact bundle."""

    normalized_bundle_hash = str(bundle_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_bundle_hash):
        raise AttestedV2StoreError("V2 publication bundle hash is invalid")
    row = await select_one(
        PUBLICATION_TABLE,
        filters=(("bundle_hash", normalized_bundle_hash),),
    )
    if not isinstance(row, Mapping):
        return None
    bundle_row = await select_one(
        BUNDLE_TABLE,
        filters=(("bundle_hash", normalized_bundle_hash),),
    )
    bundle_doc = (
        bundle_row.get("bundle_doc")
        if isinstance(bundle_row, Mapping)
        else None
    )
    if not isinstance(bundle_doc, Mapping):
        raise AttestedV2StoreError("stored V2 publication bundle is missing")
    bundle = validate_published_weight_bundle_v2(bundle_doc)
    expected_bundle_row = {
        "bundle_hash": bundle["bundle_hash"],
        "schema_version": bundle_doc["schema_version"],
        "netuid": bundle["netuid"],
        "epoch_id": bundle["epoch_id"],
        "block": bundle["block"],
        "validator_hotkey": bundle["validator_hotkey"],
        "root_receipt_hash": bundle["root_receipt_hash"],
        "weights_hash": bundle["weights_hash"],
        "snapshot_hash": bundle["snapshot_hash"],
        "bundle_doc": dict(bundle_doc),
    }
    _assert_stored_row(BUNDLE_TABLE, bundle_row, expected_bundle_row)
    bundle_readback_hash = sha256_json(
        {
            field: expected_bundle_row[field]
            for field in sorted(expected_bundle_row)
        }
    )
    publication_doc = row.get("publication_doc")
    if not isinstance(publication_doc, Mapping):
        raise AttestedV2StoreError("stored V2 publication document is missing")
    expected_fields = {
        "schema_version",
        "bundle_hash",
        "root_receipt_hash",
        "durable_readback_hash",
        "transparency_event_hash",
    }
    if (
        set(publication_doc) != expected_fields
        or publication_doc.get("schema_version")
        != "leadpoet.weight_publication.v2"
        or publication_doc.get("bundle_hash") != normalized_bundle_hash
        or publication_doc.get("root_receipt_hash")
        != bundle["root_receipt_hash"]
        or publication_doc.get("durable_readback_hash")
        != bundle_readback_hash
        or any(
            not _HASH_RE.fullmatch(str(publication_doc.get(field) or ""))
            for field in (
                "bundle_hash",
                "root_receipt_hash",
                "durable_readback_hash",
                "transparency_event_hash",
            )
        )
    ):
        raise AttestedV2StoreError("stored V2 publication document is invalid")
    event_hash = sha256_json(
        {
            "bundle_hash": normalized_bundle_hash,
            "publication_receipt_hash": row.get("publication_receipt_hash"),
            "transparency_event_hash": publication_doc.get(
                "transparency_event_hash"
            ),
            "durable_readback_hash": publication_doc.get(
                "durable_readback_hash"
            ),
        }
    )
    expected = {
        "weight_submission_event_hash": event_hash,
        "bundle_hash": normalized_bundle_hash,
        "publication_receipt_hash": row.get("publication_receipt_hash"),
        "transparency_event_hash": publication_doc.get(
            "transparency_event_hash"
        ),
        "durable_readback_hash": publication_doc.get("durable_readback_hash"),
        "publication_doc": dict(publication_doc),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise AttestedV2StoreError(
                "stored V2 publication conflicts at %s" % field
            )
    graph = await load_receipt_graph_v2(
        str(row.get("publication_receipt_hash") or "")
    )
    receipt_by_hash = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    root_hash = str(graph.get("root_receipt_hash") or "")
    root = receipt_by_hash.get(root_hash)
    if (
        root_hash != row.get("publication_receipt_hash")
        or not isinstance(root, Mapping)
        or root.get("role") != "gateway_coordinator"
        or root.get("purpose") != "gateway.weights.publication.v2"
        or root.get("status") != "succeeded"
        or int(root.get("epoch_id", -1)) != int(bundle["epoch_id"])
        or root.get("parent_receipt_hashes")
        != [bundle["root_receipt_hash"]]
        or root.get("output_root") != sha256_json(dict(publication_doc))
    ):
        raise AttestedV2StoreError(
            "stored V2 publication receipt does not bind its bundle"
        )
    return expected


async def persist_weight_finalization_v2(
    *,
    submission: Mapping[str, Any],
    chain_signing_profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Persist finalized inclusion and state-transition proof for one V2 bundle."""

    verified = validate_weight_finalization_submission_v2(
        submission,
        chain_signing_profile=chain_signing_profile,
    )
    publication = await select_one(
        PUBLICATION_TABLE,
        filters=(
            (
                "weight_submission_event_hash",
                verified["weight_submission_event_hash"],
            ),
        ),
    )
    if not isinstance(publication, Mapping):
        raise AttestedV2StoreError(
            "V2 finalization has no durable publication parent"
        )
    bundle_hash = str(publication.get("bundle_hash") or "")
    bundle_row = await select_one(
        BUNDLE_TABLE,
        filters=(("bundle_hash", bundle_hash),),
    )
    if not isinstance(bundle_row, Mapping) or not isinstance(
        bundle_row.get("bundle_doc"), Mapping
    ):
        raise AttestedV2StoreError("V2 finalization bundle is unavailable")
    bundle = validate_published_weight_bundle_v2(bundle_row["bundle_doc"])
    for field, expected in (
        ("validator_hotkey", bundle["validator_hotkey"]),
        ("netuid", bundle["netuid"]),
        ("epoch_id", bundle["epoch_id"]),
        ("weights_hash", bundle["weights_hash"]),
        ("weight_receipt_hash", bundle["weight_receipt_hash"]),
    ):
        if verified[field] != expected:
            raise AttestedV2StoreError(
                "V2 finalization differs from bundle at %s" % field
            )
    graph_result = await persist_receipt_graph_v2(submission["receipt_graph"])
    event_hash = sha256_json(
        {
            "weight_submission_event_hash": verified[
                "weight_submission_event_hash"
            ],
            "bundle_hash": bundle_hash,
            "finalization_receipt_hash": verified[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": verified[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": verified["extrinsic_hash"],
            "finalized_block": verified["finalized_block"],
            "finalized_block_hash": verified["finalized_block_hash"],
            "state_transition_hash": verified["state_transition_hash"],
        }
    )
    row = {
        "weight_finalization_event_hash": event_hash,
        "weight_submission_event_hash": verified[
            "weight_submission_event_hash"
        ],
        "bundle_hash": bundle_hash,
        "finalization_receipt_hash": verified["finalization_receipt_hash"],
        "extrinsic_authorization_hash": verified[
            "extrinsic_authorization_hash"
        ],
        "extrinsic_hash": verified["extrinsic_hash"],
        "finalized_block": verified["finalized_block"],
        "finalized_block_hash": verified["finalized_block_hash"],
        "state_transition_hash": verified["state_transition_hash"],
        "finalization_doc": dict(submission["finalization"]),
    }
    await _insert_exact(
        FINALIZATION_TABLE,
        row,
        key_filters=(
            (
                "weight_submission_event_hash",
                row["weight_submission_event_hash"],
            ),
        ),
    )
    durable = await select_one(
        FINALIZATION_TABLE,
        filters=(
            (
                "weight_submission_event_hash",
                row["weight_submission_event_hash"],
            ),
        ),
    )
    if not isinstance(durable, Mapping):
        raise AttestedV2StoreError(
            "V2 finalization durable readback returned no row"
        )
    for field, expected in row.items():
        if durable.get(field) != expected:
            raise AttestedV2StoreError(
                "V2 finalization durable readback conflicts at %s" % field
            )
    return {
        **verified,
        **graph_result,
        "bundle_hash": bundle_hash,
        "weight_finalization_event_hash": event_hash,
    }


async def load_weight_authority_v2(
    *,
    netuid: int,
    epoch_id: int,
    validator_hotkey: str,
    require_finalization: bool = True,
) -> dict[str, Any] | None:
    """Load the bundle, gateway publication, and finalized-chain proof.

    With ``require_finalization=True`` (the default) the historical payload
    shape is returned unchanged and only fully finalized authority exists.
    With ``require_finalization=False`` a staged payload is returned as soon
    as the durable gateway publication exists: ``authority_stage`` is
    ``"published"`` until the finalized-chain proof lands, after which the
    same request returns ``"finalized"`` with the full proof attached. The
    staged shape lets auditors mirror the enclave-signed publication within
    the live epoch instead of one epoch behind.
    """

    bundle = await load_weight_bundle_v2(
        netuid=int(netuid),
        epoch_id=int(epoch_id),
        validator_hotkey=str(validator_hotkey),
    )
    if bundle is None:
        return None
    bundle_verified = validate_published_weight_bundle_v2(bundle)
    publication = await select_one(
        PUBLICATION_TABLE,
        filters=(("bundle_hash", bundle_verified["bundle_hash"]),),
    )
    if not isinstance(publication, Mapping):
        raise AttestedV2StoreError("V2 bundle publication is missing")
    finalization = await select_one(
        FINALIZATION_TABLE,
        filters=(("bundle_hash", bundle_verified["bundle_hash"]),),
    )
    if not isinstance(finalization, Mapping):
        if require_finalization:
            return None
        staged_publication_graph = await load_receipt_graph_v2(
            str(publication.get("publication_receipt_hash") or "")
        )
        return {
            "schema_version": (
                "leadpoet.published_weight_authority_stage.v2"
            ),
            "authority_stage": "published",
            "bundle": bundle,
            "publication": {
                "weight_submission_event_hash": publication[
                    "weight_submission_event_hash"
                ],
                "publication_receipt_hash": publication[
                    "publication_receipt_hash"
                ],
                "publication_doc": dict(publication["publication_doc"]),
                "receipt_graph": staged_publication_graph,
            },
            "finalization": None,
        }
    publication_graph = await load_receipt_graph_v2(
        str(publication.get("publication_receipt_hash") or "")
    )
    finalization_graph = await load_receipt_graph_v2(
        str(finalization.get("finalization_receipt_hash") or "")
    )
    finalization_submission = {
        "schema_version": "leadpoet.weight_finalization_submission.v2",
        "validator_hotkey": str(validator_hotkey),
        "weight_submission_event_hash": str(
            publication.get("weight_submission_event_hash") or ""
        ),
        "finalization": dict(finalization.get("finalization_doc") or {}),
        "receipt_graph": finalization_graph,
    }
    publication_section = {
        "weight_submission_event_hash": publication[
            "weight_submission_event_hash"
        ],
        "publication_receipt_hash": publication[
            "publication_receipt_hash"
        ],
        "publication_doc": dict(publication["publication_doc"]),
        "receipt_graph": publication_graph,
    }
    finalization_section = {
        "weight_finalization_event_hash": finalization[
            "weight_finalization_event_hash"
        ],
        "submission": finalization_submission,
    }
    if not require_finalization:
        return {
            "schema_version": (
                "leadpoet.published_weight_authority_stage.v2"
            ),
            "authority_stage": "finalized",
            "bundle": bundle,
            "publication": publication_section,
            "finalization": finalization_section,
        }
    return {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": bundle,
        "publication": publication_section,
        "finalization": finalization_section,
    }
