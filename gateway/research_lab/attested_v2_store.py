"""Durable append-only persistence for authoritative V2 attestation records."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import heapq
import json
import logging
import re
from typing import Any, Iterable, Mapping, Optional

from gateway.research_lab.store import (
    _is_transient_store_error,
    call_rpc,
    insert_row,
    insert_rows,
    select_all,
    select_many,
    select_one,
)
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    RECEIPT_GRAPH_SCHEMA_VERSION,
    build_checkpointed_receipt_graph,
    build_receipt_graph,
    compact_checkpointed_receipt_graph,
    merkle_root,
    sha256_json,
    validate_boot_identity,
    validate_host_operation_record,
    validate_receipt_graph,
    validate_receipt_graphs,
    validate_signed_execution_receipt,
    validate_signed_transition_command,
    validate_transport_attempt,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.compact_auditor_authority_v2 import (
    validate_compact_published_weight_authority_shape_v2,
)
from leadpoet_canonical.compact_weight_authority_v2 import (
    compact_weight_bundle_hash_v2,
    validate_compact_weight_submission_shape_v2,
)
from leadpoet_canonical.sourcing_history_v2 import validate_sourcing_epoch_v2
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
    validate_weight_finalization_submission_v2,
    validate_published_weight_bundle_v2,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    validate_source_add_credential_envelope_v2,
    validate_source_add_runtime_catalog_v2,
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
ANCESTRY_CHECKPOINT_TABLE = "research_lab_attested_ancestry_checkpoints_v2"
ANCESTRY_ACTIVATION_TABLE = "research_lab_attested_ancestry_activations_v2"
ANCESTRY_CHECKPOINT_RPC = "persist_research_lab_ancestry_checkpoint_v2"
COMPACT_WEIGHT_SUBMISSION_TABLE = "research_lab_compact_weight_submissions_v2"
COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE = (
    "research_lab_compact_weight_publication_intents_v2"
)
COMPACT_WEIGHT_AUTHORITY_TABLE = "research_lab_compact_weight_authorities_v2"
COMPACT_WEIGHT_AUTHORITY_MAX_BYTES_V2 = 8_388_608
LEGACY_NONFINALIZATION_TABLE = (
    "research_lab_legacy_allocation_nonfinalizations_v2"
)
CHAIN_REALIZED_SETTLEMENT_TABLE = (
    "research_lab_chain_realized_epoch_settlements_v1"
)
CHAIN_REALIZED_CREDIT_TABLE = (
    "research_lab_chain_realized_obligation_credits_v1"
)
CHAIN_REALIZED_SETTLEMENT_RPC = (
    "persist_research_lab_chain_realized_settlement_v1"
)
CHAIN_REALIZED_LIFETIME_SETTLEMENT_RPC = (
    "persist_research_lab_chain_realized_lifetime_settlement_v2"
)
CHAIN_REALIZED_UNATTRIBUTED_SETTLEMENT_RPC = (
    "persist_research_lab_chain_realized_unattributed_v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_TABLE = (
    "research_lab_allocation_settlement_frontiers_v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_ACTIVATION_TABLE = (
    "research_lab_allocation_settlement_frontier_activation_v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_RPC = (
    "persist_research_lab_allocation_settlement_frontier_v2"
)
ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_RPC = (
    "persist_research_lab_allocation_frontier_bootstrap_v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_ADD_ENV_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_GRAPH_QUERY_CHUNK = 50
_MAX_GRAPH_ROWS = 10000
_EXACT_INSERT_ATTEMPTS = 4
_EXACT_INSERT_BACKOFF_SECONDS = (0.25, 0.75, 1.5)
_EXACT_INSERT_BATCH_ROWS = 100
_DUPLICATE_READBACK_ATTEMPTS = 4
_DUPLICATE_READBACK_BACKOFF_SECONDS = (0.1, 0.25, 0.5)
_ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
_REPLAYABLE_EXECUTION_PAIRS = frozenset(
    {
        ("research_lab_allocation", "research_lab.allocation.v2"),
        (
            "allocation_settlement_frontier_bootstrap_v2",
            "research_lab.allocation_settlement_frontier_bootstrap.v2",
        ),
        (
            "source_add_catalog_snapshot_v2",
            "research_lab.source_add_catalog_snapshot.v2",
        ),
        (
            "observe_chain_realized_weights_v1",
            "research_lab.chain_weight_observation.v1",
        ),
        (
            "attest_chain_realized_settlement_v1",
            "research_lab.chain_realized_epoch_settlement.v1",
        ),
    }
    | {
        ("attest_weight_input", purpose)
        for role, purpose in WEIGHT_INPUT_PURPOSES.values()
        if role == "gateway_coordinator"
    }
)

logger = logging.getLogger(__name__)


class AttestedV2StoreError(RuntimeError):
    """A V2 append or durable readback failed or conflicted."""


async def _ancestry_checkpoint_unknown_commit_sleep(seconds: float) -> None:
    """Sleep between read-only checks for one unknown RPC commit outcome."""

    await asyncio.sleep(seconds)


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
    for attempt in range(_EXACT_INSERT_ATTEMPTS):
        try:
            stored = await insert_row(table, expected)
        except Exception as exc:
            duplicate = _is_duplicate_error(exc)
            transient = _is_transient_store_error(exc)
            if not duplicate and not transient:
                raise

            if duplicate:
                for readback_attempt in range(_DUPLICATE_READBACK_ATTEMPTS):
                    stored = await select_one(table, filters=key_filters)
                    if isinstance(stored, Mapping):
                        _assert_stored_row(table, stored, expected)
                        return dict(stored)
                    if readback_attempt == _DUPLICATE_READBACK_ATTEMPTS - 1:
                        raise AttestedV2StoreError(
                            "%s duplicate could not be reloaded after bounded retry"
                            % table
                        ) from exc
                    backoff = _DUPLICATE_READBACK_BACKOFF_SECONDS[
                        min(
                            readback_attempt,
                            len(_DUPLICATE_READBACK_BACKOFF_SECONDS) - 1,
                        )
                    ]
                    logger.warning(
                        "duplicate_exact_insert_readback_retry "
                        "table=%s attempt=%s/%s",
                        table,
                        readback_attempt + 1,
                        _DUPLICATE_READBACK_ATTEMPTS,
                    )
                    await asyncio.sleep(backoff)

            stored = await select_one(table, filters=key_filters)
            if isinstance(stored, Mapping):
                _assert_stored_row(table, stored, expected)
                return dict(stored)
            if attempt == _EXACT_INSERT_ATTEMPTS - 1:
                raise

            backoff = _EXACT_INSERT_BACKOFF_SECONDS[
                min(attempt, len(_EXACT_INSERT_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "transient_exact_insert_retry table=%s attempt=%s/%s "
                "type=%s error=%s",
                table,
                attempt + 1,
                _EXACT_INSERT_ATTEMPTS,
                type(exc).__name__,
                str(exc)[:160],
            )
            await asyncio.sleep(backoff)
            continue

        _assert_stored_row(table, stored, expected)
        return dict(stored)

    raise AssertionError("exact insert retry loop exhausted unexpectedly")


def _exact_batch_row_key(
    table: str,
    row: Mapping[str, Any],
    *,
    key_fields: tuple[str, ...],
) -> tuple[str, ...]:
    key = tuple(str(row.get(field) or "") for field in key_fields)
    if not key_fields or len(set(key_fields)) != len(key_fields) or not all(key):
        raise AttestedV2StoreError(
            "%s exact batch row key is invalid" % table
        )
    return key


async def _read_exact_batch_rows(
    table: str,
    *,
    expected_by_key: Mapping[tuple[str, ...], Mapping[str, Any]],
    key_fields: tuple[str, ...],
) -> set[tuple[str, ...]]:
    """Reconcile one ambiguous batch through exact per-key durable reads."""

    existing: set[tuple[str, ...]] = set()
    for key, expected in expected_by_key.items():
        stored = await select_one(
            table,
            filters=tuple(zip(key_fields, key)),
        )
        if stored is None:
            continue
        _assert_stored_row(table, stored, expected)
        existing.add(key)
    return existing


async def _insert_exact_batch(
    table: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> None:
    """Insert one independent row batch with exact ambiguous-outcome recovery."""

    expected_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        key = _exact_batch_row_key(table, row, key_fields=key_fields)
        if key in expected_by_key:
            raise AttestedV2StoreError(
                "%s exact batch row key is duplicated" % table
            )
        expected_by_key[key] = row
    if not expected_by_key:
        return

    pending = dict(expected_by_key)
    last_error: BaseException | None = None
    last_error_kind = ""
    for attempt in range(_EXACT_INSERT_ATTEMPTS):
        try:
            stored_rows = await insert_rows(table, pending.values())
        except Exception as exc:
            duplicate = _is_duplicate_error(exc)
            transient = _is_transient_store_error(exc)
            if not duplicate and not transient:
                raise
            last_error = exc
            last_error_kind = "duplicate" if duplicate else "transient"
        else:
            observed: set[tuple[str, ...]] = set()
            for stored in stored_rows:
                if not isinstance(stored, Mapping):
                    raise AttestedV2StoreError(
                        "%s exact batch insert returned an invalid row" % table
                    )
                key = _exact_batch_row_key(
                    table,
                    stored,
                    key_fields=key_fields,
                )
                expected = pending.get(key)
                if expected is None or key in observed:
                    raise AttestedV2StoreError(
                        "%s exact batch insert returned an unexpected row" % table
                    )
                _assert_stored_row(table, stored, expected)
                observed.add(key)
            if observed == set(pending):
                return
            last_error = AttestedV2StoreError(
                "%s exact batch insert response is incomplete" % table
            )
            last_error_kind = "ambiguous_response"

        existing = await _read_exact_batch_rows(
            table,
            expected_by_key=pending,
            key_fields=key_fields,
        )
        pending = {
            key: row for key, row in pending.items() if key not in existing
        }
        if not pending:
            return
        if attempt == _EXACT_INSERT_ATTEMPTS - 1:
            if last_error_kind == "duplicate":
                raise AttestedV2StoreError(
                    "%s batch duplicate could not be reloaded after bounded retry"
                    % table
                ) from last_error
            assert last_error is not None
            raise last_error

        if last_error_kind == "duplicate":
            backoff = _DUPLICATE_READBACK_BACKOFF_SECONDS[
                min(attempt, len(_DUPLICATE_READBACK_BACKOFF_SECONDS) - 1)
            ]
        else:
            backoff = _EXACT_INSERT_BACKOFF_SECONDS[
                min(attempt, len(_EXACT_INSERT_BACKOFF_SECONDS) - 1)
            ]
        logger.warning(
            "exact_batch_insert_retry table=%s attempt=%s/%s kind=%s "
            "remaining_rows=%s",
            table,
            attempt + 1,
            _EXACT_INSERT_ATTEMPTS,
            last_error_kind,
            len(pending),
        )
        await asyncio.sleep(backoff)

    raise AssertionError("exact batch insert retry loop exhausted unexpectedly")


async def _insert_exact_rows(
    table: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> None:
    """Persist independent rows in bounded batches without weakening readback."""

    normalized = [dict(row) for row in rows]
    seen: set[tuple[str, ...]] = set()
    for row in normalized:
        key = _exact_batch_row_key(table, row, key_fields=key_fields)
        if key in seen:
            raise AttestedV2StoreError(
                "%s exact row key is duplicated" % table
            )
        seen.add(key)
    for offset in range(0, len(normalized), _EXACT_INSERT_BATCH_ROWS):
        batch = normalized[offset : offset + _EXACT_INSERT_BATCH_ROWS]
        if len(batch) == 1:
            row = batch[0]
            await _insert_exact(
                table,
                row,
                key_filters=tuple(
                    (field, row[field]) for field in key_fields
                ),
            )
            continue
        await _insert_exact_batch(
            table,
            batch,
            key_fields=key_fields,
        )


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
    key_fields: tuple[str, ...],
    max_total_rows: int | None = _MAX_GRAPH_ROWS,
) -> list[dict[str, Any]]:
    """Select bounded value chunks in one stable, unique row order.

    Receipt ancestry is bounded by ``_MAX_GRAPH_ROWS``. Evidence attached to
    that ancestry can legitimately exceed the same aggregate count, so callers
    may disable only the aggregate bound while every individual query remains
    bounded. Oversized owner batches are split until the offending owner is
    isolated; one owner can never return more than ``_MAX_GRAPH_ROWS`` rows.
    """

    normalized = sorted({str(value) for value in values})
    if max_total_rows is not None and len(normalized) > max_total_rows:
        raise AttestedV2StoreError("V2 receipt graph exceeds row limit")
    if not key_fields or len(set(key_fields)) != len(key_fields):
        raise AttestedV2StoreError("V2 durable row key fields are invalid")

    async def select_chunk(chunk: list[str]) -> list[dict[str, Any]]:
        try:
            return await select_all(
                table,
                filters=((field, "in", chunk),),
                order_by=tuple((key_field, False) for key_field in key_fields),
                max_rows=_MAX_GRAPH_ROWS,
            )
        except RuntimeError as exc:
            if "paginated select exceeded max_rows=" not in str(exc):
                raise
            if len(chunk) == 1:
                raise AttestedV2StoreError(
                    "V2 receipt graph exceeds row limit"
                ) from exc
            midpoint = len(chunk) // 2
            return (
                await select_chunk(chunk[:midpoint])
                + await select_chunk(chunk[midpoint:])
            )

    rows = []
    for offset in range(0, len(normalized), _GRAPH_QUERY_CHUNK):
        chunk = normalized[offset : offset + _GRAPH_QUERY_CHUNK]
        rows.extend(await select_chunk(chunk))
        if max_total_rows is not None and len(rows) > max_total_rows:
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
        key_fields=(key_field,),
        max_total_rows=None,
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
        key_fields=key_fields,
        max_total_rows=None,
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


def _parent_first_receipt_hashes_v2(
    graph: Mapping[str, Any],
    *,
    validated_receipts: Iterable[str],
) -> tuple[str, ...]:
    """Derive a deterministic persistence order from local parent edges.

    Checkpoint validation returns the certificate's canonical receipt-hash
    projection. That projection proves membership but is not a topological
    insertion order. External checkpoint parents are intentionally omitted and
    remain subject to the durable edge foreign keys.
    """

    receipt_by_hash = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    validated = tuple(str(receipt_hash) for receipt_hash in validated_receipts)
    if (
        len(receipt_by_hash) != len(graph.get("receipts") or ())
        or len(validated) != len(set(validated))
        or set(validated) != set(receipt_by_hash)
    ):
        raise AttestedV2StoreError(
            "validated V2 receipt membership differs from graph"
        )

    local_parent_count = {receipt_hash: 0 for receipt_hash in receipt_by_hash}
    local_children = {receipt_hash: [] for receipt_hash in receipt_by_hash}
    for child_hash, receipt in receipt_by_hash.items():
        for parent_hash in receipt.get("parent_receipt_hashes") or ():
            parent_hash = str(parent_hash)
            if parent_hash not in receipt_by_hash:
                continue
            local_parent_count[child_hash] += 1
            local_children[parent_hash].append(child_hash)

    ready = [
        receipt_hash
        for receipt_hash, parent_count in local_parent_count.items()
        if parent_count == 0
    ]
    heapq.heapify(ready)
    ordered = []
    while ready:
        receipt_hash = heapq.heappop(ready)
        ordered.append(receipt_hash)
        for child_hash in sorted(local_children[receipt_hash]):
            local_parent_count[child_hash] -= 1
            if local_parent_count[child_hash] == 0:
                heapq.heappush(ready, child_hash)

    if len(ordered) != len(receipt_by_hash):
        raise AttestedV2StoreError(
            "validated V2 receipt graph has no parent-first persistence order"
        )
    return tuple(ordered)


async def persist_receipt_graph_v2(
    graph: Mapping[str, Any],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Persist one legacy graph or bounded checkpoint delta idempotently.

    A checkpointed graph contains only current local evidence. Its external
    parent rows must already be durable; the edge foreign keys enforce that
    ordering without loading or reconstructing historical bodies.
    """

    validated_receipts = validate_receipt_graph(
        graph,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    ordered_receipts = _parent_first_receipt_hashes_v2(
        graph,
        validated_receipts=validated_receipts,
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
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        if (
            len(attempts_by_scope.get(scope, ())) > _MAX_GRAPH_ROWS
            or len(host_operations_by_scope.get(scope, ())) > _MAX_GRAPH_ROWS
        ):
            raise AttestedV2StoreError("V2 receipt graph exceeds row limit")

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
    if (
        len(boot_rows) > _MAX_GRAPH_ROWS
        or len(receipt_rows) > _MAX_GRAPH_ROWS
    ):
        raise AttestedV2StoreError("V2 receipt graph exceeds row limit")
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
    await _insert_exact_rows(
        TRANSPORT_TABLE,
        (
            row
            for row in transport_rows
            if row["attempt_hash"] not in existing_attempts
        ),
        key_fields=("attempt_hash",),
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
    await _insert_exact_rows(
        RECEIPT_TRANSPORT_TABLE,
        (
            row
            for row in receipt_transport_rows
            if (row["receipt_hash"], row["attempt_hash"])
            not in existing_receipt_transports
        ),
        key_fields=("receipt_hash", "attempt_hash"),
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


def ancestry_checkpoint_storage_row_v2(
    proof: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Any,
    allowed_issuer_roles: Iterable[str],
) -> dict[str, Any]:
    """Validate one detached proof and project its exact append-only row."""

    normalized = validate_compact_ancestry_proof_v2(
        proof,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
    )
    certificate = normalized["certificate"]
    claim = certificate["claim"]
    return {
        "root_receipt_hash": str(claim["output_root_receipt_hash"]),
        "schema_version": str(certificate["schema_version"]),
        "lineage_id": str(claim["lineage_id"]),
        "certificate_hash": str(certificate["certificate_hash"]),
        "certificate_sequence": int(claim["certificate_sequence"]),
        "issuer_boot_identity_hash": str(claim["issuer_boot_identity_hash"]),
        "proof_hash": str(normalized["proof_hash"]),
        "certificate_doc": dict(certificate),
        "proof_doc": dict(normalized),
    }


async def persist_ancestry_checkpoint_v2(
    proof: Mapping[str, Any],
    *,
    checkpointed_graph: Mapping[str, Any],
    expected_lineage_id: str,
    boot_attestation_verifier: Any,
    allowed_issuer_roles: Iterable[str],
) -> dict[str, Any]:
    """Atomically append one bounded certificate and activate its lineage.

    Runtime code never reconstructs a complete graph here. The local graph is
    cryptographically bound to the enclave certificate, while the database RPC
    verifies that every certificate parent is already durable and permanently
    rejects a legacy full projection after that exact root is compacted.
    """

    row = ancestry_checkpoint_storage_row_v2(
        proof,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
    )
    root_hash = str(row["root_receipt_hash"])
    if (
        not isinstance(checkpointed_graph, Mapping)
        or checkpointed_graph.get("schema_version")
        not in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS
        or checkpointed_graph.get("root_receipt_hash") != root_hash
        or checkpointed_graph.get("ancestry_lineage_id")
        != expected_lineage_id
        or checkpointed_graph.get("ancestry_proof") != row["proof_doc"]
    ):
        raise AttestedV2StoreError(
            "ancestry checkpoint bounded graph identity differs"
        )
    failed_receipts = tuple(
        sorted(
            str(item.get("receipt_hash") or "")
            for item in checkpointed_graph.get("receipts") or ()
            if isinstance(item, Mapping) and item.get("status") != "succeeded"
        )
    )
    validate_receipt_graph(
        checkpointed_graph,
        allowed_failed_receipt_hashes=failed_receipts,
        boot_attestation_verifier=boot_attestation_verifier,
        require_boot_attestation_verification=True,
    )
    existing = await select_one(
        ANCESTRY_CHECKPOINT_TABLE,
        filters=(("root_receipt_hash", root_hash),),
    )
    if (
        checkpointed_graph.get("schema_version")
        == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
    ):
        existing_graph = (
            existing.get("checkpoint_graph_doc")
            if isinstance(existing, Mapping)
            else None
        )
        existing_schema = (
            existing_graph.get("schema_version")
            if isinstance(existing_graph, Mapping)
            else None
        )
        if existing_schema not in {
            None,
            CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
            COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        }:
            raise AttestedV2StoreError(
                "ancestry checkpoint durable graph schema is invalid"
            )
        if (
            existing_schema
            != CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
        ):
            checkpointed_graph = compact_checkpointed_receipt_graph(
                checkpointed_graph,
                allowed_failed_receipt_hashes=failed_receipts,
                boot_attestation_verifier=boot_attestation_verifier,
                require_boot_attestation_verification=True,
            )
    if checkpointed_graph.get("schema_version") not in {
        CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    }:
        raise AttestedV2StoreError(
            "ancestry checkpoint compact graph schema differs"
        )
    row = {
        **row,
        "checkpoint_graph_hash": sha256_json(dict(checkpointed_graph)),
        "checkpoint_graph_doc": dict(checkpointed_graph),
    }
    expected_ack = {
        "root_receipt_hash": root_hash,
        "certificate_hash": row["certificate_hash"],
        "proof_hash": row["proof_hash"],
        "lineage_id": row["lineage_id"],
        "certificate_sequence": row["certificate_sequence"],
        "checkpoint_graph_hash": row["checkpoint_graph_hash"],
    }

    async def exact_durable_ack(
        stored_checkpoint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        stored = stored_checkpoint
        if not isinstance(stored, Mapping):
            stored = await select_one(
                ANCESTRY_CHECKPOINT_TABLE,
                filters=(("root_receipt_hash", root_hash),),
            )
        if not isinstance(stored, Mapping):
            return None
        _assert_stored_row(ANCESTRY_CHECKPOINT_TABLE, stored, row)
        activation = await select_one(
            ANCESTRY_ACTIVATION_TABLE,
            filters=(("activation_root_receipt_hash", root_hash),),
        )
        if not isinstance(activation, Mapping):
            return None
        _assert_stored_row(
            ANCESTRY_ACTIVATION_TABLE,
            activation,
            {
                "lineage_id": row["lineage_id"],
                "activation_root_receipt_hash": root_hash,
                "activation_certificate_hash": row["certificate_hash"],
            },
        )
        return {**expected_ack, "root_activated": True}

    replay_ack = (
        await exact_durable_ack(existing)
        if isinstance(existing, Mapping)
        else None
    )
    if replay_ack is not None:
        return replay_ack

    try:
        result = await call_rpc(ANCESTRY_CHECKPOINT_RPC, {"checkpoint": row})
    except Exception as exc:
        if not _is_transient_store_error(exc):
            raise
        readback_attempts = (
            len(_ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS) + 1
        )
        for readback_attempt in range(readback_attempts):
            if readback_attempt:
                delay = _ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS[
                    readback_attempt - 1
                ]
                logger.warning(
                    "ancestry_checkpoint_rpc_unknown_commit_pending "
                    "root=%s attempt=%s/%s delay_seconds=%s type=%s",
                    root_hash,
                    readback_attempt + 1,
                    readback_attempts,
                    delay,
                    type(exc).__name__,
                )
                await _ancestry_checkpoint_unknown_commit_sleep(delay)
            try:
                durable_ack = await exact_durable_ack()
            except Exception as readback_exc:
                if not _is_transient_store_error(readback_exc):
                    raise
                durable_ack = None
            if durable_ack is None:
                continue
            logger.warning(
                "ancestry_checkpoint_rpc_transient_recovered "
                "root=%s attempt=%s/%s type=%s",
                root_hash,
                readback_attempt + 1,
                readback_attempts,
                type(exc).__name__,
            )
            return durable_ack
        logger.warning(
            "ancestry_checkpoint_rpc_unknown_commit_exhausted "
            "root=%s attempts=%s type=%s",
            root_hash,
            readback_attempts,
            type(exc).__name__,
        )
        raise
    if not isinstance(result, Mapping):
        raise AttestedV2StoreError(
            "ancestry checkpoint RPC returned no durable acknowledgment"
        )
    if any(result.get(field) != value for field, value in expected_ack.items()):
        raise AttestedV2StoreError(
            "ancestry checkpoint RPC acknowledgment conflicts"
        )
    durable_ack = await exact_durable_ack()
    if durable_ack is None:
        raise AttestedV2StoreError(
            "ancestry checkpoint durable activation readback is missing"
        )
    if result.get("root_activated") is not True:
        raise AttestedV2StoreError(
            "ancestry checkpoint RPC activation acknowledgment conflicts"
        )
    return durable_ack


async def load_ancestry_checkpoint_proofs_v2(
    root_receipt_hashes: Iterable[str],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Any,
    allowed_issuer_roles: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Load selected immutable compact proofs; absent roots remain absent."""

    roots = sorted({str(value or "").lower() for value in root_receipt_hashes})
    if any(not _HASH_RE.fullmatch(root) for root in roots):
        raise AttestedV2StoreError("ancestry checkpoint root hash is invalid")
    if not roots:
        return {}
    rows = await _select_by_values(
        ANCESTRY_CHECKPOINT_TABLE,
        field="root_receipt_hash",
        values=roots,
        key_fields=("root_receipt_hash",),
        max_total_rows=len(roots),
    )
    row_by_root: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        root_hash = str(row.get("root_receipt_hash") or "")
        if root_hash not in roots or root_hash in row_by_root:
            raise AttestedV2StoreError(
                "ancestry checkpoint readback is unexpected or duplicated"
            )
        row_by_root[root_hash] = row
    known = {
        root_hash: str(row.get("certificate_hash") or "")
        for root_hash, row in row_by_root.items()
    }
    loaded: dict[str, dict[str, Any]] = {}
    for root_hash, row in row_by_root.items():
        proof = row.get("proof_doc")
        if not isinstance(proof, Mapping):
            raise AttestedV2StoreError(
                "ancestry checkpoint proof document is unavailable"
            )
        normalized = validate_compact_ancestry_proof_v2(
            proof,
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles=allowed_issuer_roles,
            required_receipt_hashes=(root_hash,),
            known_certificate_hashes_by_root=known,
        )
        expected = ancestry_checkpoint_storage_row_v2(
            normalized,
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles=allowed_issuer_roles,
        )
        _assert_stored_row(ANCESTRY_CHECKPOINT_TABLE, row, expected)
        loaded[root_hash] = normalized
    return loaded


async def load_ancestry_checkpoint_proof_v2(
    root_receipt_hash: str,
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Any,
    allowed_issuer_roles: Iterable[str],
) -> Optional[dict[str, Any]]:
    loaded = await load_ancestry_checkpoint_proofs_v2(
        (root_receipt_hash,),
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
    )
    return loaded.get(str(root_receipt_hash or "").lower())


async def load_checkpointed_receipt_graphs_v2(
    root_receipt_hashes: Iterable[str],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Load exact bounded graphs without traversing historical receipt edges."""

    roots = sorted({str(value or "").lower() for value in root_receipt_hashes})
    allowed_failed = {
        str(value or "").lower() for value in allowed_failed_receipt_hashes
    }
    if any(not _HASH_RE.fullmatch(value) for value in [*roots, *allowed_failed]):
        raise AttestedV2StoreError("checkpointed receipt graph hash is invalid")
    if allowed_failed and len(roots) != 1:
        raise AttestedV2StoreError(
            "V2 failed receipt allowance requires one graph root"
        )
    if not roots:
        return {}
    rows = await _select_by_values(
        ANCESTRY_CHECKPOINT_TABLE,
        field="root_receipt_hash",
        values=roots,
        key_fields=("root_receipt_hash",),
        max_total_rows=len(roots),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for row in rows:
        root_hash = str(row.get("root_receipt_hash") or "")
        graph = row.get("checkpoint_graph_doc")
        proof = row.get("proof_doc")
        if (
            root_hash not in roots
            or root_hash in loaded
            or not isinstance(graph, Mapping)
            or not isinstance(proof, Mapping)
            or graph.get("root_receipt_hash") != root_hash
            or graph.get("ancestry_proof") != proof
            or row.get("checkpoint_graph_hash") != sha256_json(dict(graph))
        ):
            raise AttestedV2StoreError(
                "checkpointed receipt graph durable row is inconsistent"
            )
        graph_failed = allowed_failed.intersection(
            str(item.get("receipt_hash") or "")
            for item in graph.get("receipts") or ()
            if isinstance(item, Mapping)
        )
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=graph_failed,
        )
        expected = ancestry_checkpoint_storage_row_v2(
            proof,
            expected_lineage_id=str(graph.get("ancestry_lineage_id") or ""),
            boot_attestation_verifier=lambda identity: identity,
            allowed_issuer_roles={
                "gateway_coordinator",
                "gateway_scoring",
                "gateway_autoresearch",
                "validator_weights",
            },
        )
        expected.update(
            {
                "checkpoint_graph_hash": sha256_json(dict(graph)),
                "checkpoint_graph_doc": dict(graph),
            }
        )
        _assert_stored_row(ANCESTRY_CHECKPOINT_TABLE, row, expected)
        loaded[root_hash] = dict(graph)
    return loaded


async def load_checkpointed_receipt_graph_v2(
    root_receipt_hash: str,
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> Optional[dict[str, Any]]:
    loaded = await load_checkpointed_receipt_graphs_v2(
        (root_receipt_hash,),
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    return loaded.get(str(root_receipt_hash or "").lower())


def compact_weight_submission_storage_row_v2(
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one independently hash-bound compact weight submission."""

    compact = validate_compact_weight_submission_shape_v2(submission)
    result = compact["weight_result"]
    proof = compact["validator_ancestry_proof"]
    claim = proof.get("certificate", {}).get("claim", {})
    lineage_id = str(claim.get("lineage_id") or "")
    binding_hash = str(compact["binding_receipt"].get("receipt_hash") or "")
    if (
        not _HASH_RE.fullmatch(lineage_id)
        or claim.get("output_root_receipt_hash") != binding_hash
    ):
        raise AttestedV2StoreError(
            "compact weight submission ancestry identity differs"
        )
    return {
        "compact_submission_hash": str(compact["compact_submission_hash"]),
        "bundle_hash": compact_weight_bundle_hash_v2(compact),
        "netuid": int(result["netuid"]),
        "epoch_id": int(result["epoch_id"]),
        "validator_hotkey": str(compact["validator_hotkey"]),
        "lineage_id": lineage_id,
        "binding_receipt_hash": binding_hash,
        "submission_doc": dict(compact),
    }


async def persist_compact_weight_submission_v2(
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    """Append and exact-read one first-class compact weight submission."""

    row = compact_weight_submission_storage_row_v2(submission)
    await _insert_exact(
        COMPACT_WEIGHT_SUBMISSION_TABLE,
        row,
        key_filters=(("compact_submission_hash", row["compact_submission_hash"]),),
    )
    stored = await select_one(
        COMPACT_WEIGHT_SUBMISSION_TABLE,
        filters=(("compact_submission_hash", row["compact_submission_hash"]),),
    )
    if not isinstance(stored, Mapping):
        raise AttestedV2StoreError(
            "compact weight submission durable readback is missing"
        )
    _assert_stored_row(COMPACT_WEIGHT_SUBMISSION_TABLE, stored, row)
    return {
        "compact_submission_hash": row["compact_submission_hash"],
        "bundle_hash": row["bundle_hash"],
        "binding_receipt_hash": row["binding_receipt_hash"],
        "lineage_id": row["lineage_id"],
        "durable_readback_hash": sha256_json(
            {field: stored[field] for field in sorted(row)}
        ),
    }


def compact_weight_publication_intent_storage_row_v2(
    *,
    submission: Mapping[str, Any],
    durable_readback_hash: str,
    epoch_authority: Mapping[str, Any],
    transparency_event_hash: str,
) -> dict[str, Any]:
    """Project one immutable retry boundary for compact publication."""

    compact_row = compact_weight_submission_storage_row_v2(submission)
    durable_hash = str(durable_readback_hash or "").lower()
    transparency_hash = str(transparency_event_hash or "").lower()
    if (
        not _HASH_RE.fullmatch(durable_hash)
        or not _HASH_RE.fullmatch(transparency_hash)
        or not isinstance(epoch_authority, Mapping)
    ):
        raise AttestedV2StoreError(
            "compact weight publication intent inputs are invalid"
        )
    authority_doc = dict(epoch_authority)
    epoch_authority_hash = sha256_json(authority_doc)
    body = {
        "schema_version": "leadpoet.compact_weight_publication_intent.v2",
        "bundle_hash": compact_row["bundle_hash"],
        "compact_submission_hash": compact_row["compact_submission_hash"],
        "netuid": compact_row["netuid"],
        "epoch_id": compact_row["epoch_id"],
        "validator_hotkey": compact_row["validator_hotkey"],
        "root_receipt_hash": compact_row["binding_receipt_hash"],
        "durable_readback_hash": durable_hash,
        "transparency_event_hash": transparency_hash,
        "epoch_authority_hash": epoch_authority_hash,
        "epoch_authority": authority_doc,
    }
    intent_doc = {**body, "intent_hash": sha256_json(body)}
    return {
        "bundle_hash": body["bundle_hash"],
        "compact_submission_hash": body["compact_submission_hash"],
        "netuid": body["netuid"],
        "epoch_id": body["epoch_id"],
        "validator_hotkey": body["validator_hotkey"],
        "root_receipt_hash": body["root_receipt_hash"],
        "durable_readback_hash": durable_hash,
        "transparency_event_hash": transparency_hash,
        "epoch_authority_hash": epoch_authority_hash,
        "intent_hash": intent_doc["intent_hash"],
        "intent_doc": intent_doc,
    }


async def persist_compact_weight_publication_intent_v2(
    *,
    submission: Mapping[str, Any],
    durable_readback_hash: str,
    epoch_authority: Mapping[str, Any],
    transparency_event_hash: str,
) -> dict[str, Any]:
    """Append and exact-read one retryable compact publication intent."""

    row = compact_weight_publication_intent_storage_row_v2(
        submission=submission,
        durable_readback_hash=durable_readback_hash,
        epoch_authority=epoch_authority,
        transparency_event_hash=transparency_event_hash,
    )
    filters = (("bundle_hash", row["bundle_hash"]),)
    await _insert_exact(
        COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE,
        row,
        key_filters=filters,
    )
    stored = await select_one(
        COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE,
        filters=filters,
    )
    if not isinstance(stored, Mapping):
        raise AttestedV2StoreError(
            "compact weight publication intent durable readback is missing"
        )
    _assert_stored_row(COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE, stored, row)
    return dict(row["intent_doc"])


async def load_compact_weight_publication_intent_v2(
    *, bundle_hash: str
) -> Optional[dict[str, Any]]:
    """Load and revalidate one immutable compact publication intent."""

    normalized_hash = str(bundle_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_hash):
        raise AttestedV2StoreError(
            "compact weight publication intent bundle hash is invalid"
        )
    row = await select_one(
        COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE,
        filters=(("bundle_hash", normalized_hash),),
    )
    if not isinstance(row, Mapping):
        return None
    submission_row = await select_one(
        COMPACT_WEIGHT_SUBMISSION_TABLE,
        filters=(("bundle_hash", normalized_hash),),
    )
    submission = (
        submission_row.get("submission_doc")
        if isinstance(submission_row, Mapping)
        else None
    )
    intent = row.get("intent_doc")
    if not isinstance(submission, Mapping) or not isinstance(intent, Mapping):
        raise AttestedV2StoreError(
            "compact weight publication intent authority is incomplete"
        )
    expected = compact_weight_publication_intent_storage_row_v2(
        submission=submission,
        durable_readback_hash=str(intent.get("durable_readback_hash") or ""),
        epoch_authority=(
            intent.get("epoch_authority")
            if isinstance(intent.get("epoch_authority"), Mapping)
            else {}
        ),
        transparency_event_hash=str(
            intent.get("transparency_event_hash") or ""
        ),
    )
    _assert_stored_row(COMPACT_WEIGHT_PUBLICATION_INTENT_TABLE, row, expected)
    return dict(expected["intent_doc"])


def compact_weight_authority_storage_row_v2(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one first-class compact public authority."""

    normalized = validate_compact_published_weight_authority_shape_v2(
        authority
    )
    compact = validate_compact_weight_submission_shape_v2(
        normalized["compact_submission"]
    )
    if (
        normalized["bundle_hash"] != compact_weight_bundle_hash_v2(compact)
        or normalized["publication"].get("publication_receipt_hash")
        != normalized["publication"]["ancestry_proof"]
        .get("certificate", {})
        .get("claim", {})
        .get("output_root_receipt_hash")
    ):
        raise AttestedV2StoreError(
            "compact weight authority identity differs"
        )
    return _compact_weight_authority_row_from_normalized_v2(normalized)


def _compact_weight_authority_row_from_normalized_v2(
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    authority_bytes = len(
        json.dumps(
            dict(normalized),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    )
    if authority_bytes > COMPACT_WEIGHT_AUTHORITY_MAX_BYTES_V2:
        raise AttestedV2StoreError(
            "compact weight authority exceeds the 8 MiB transport bound"
        )
    compact = validate_compact_weight_submission_shape_v2(
        normalized["compact_submission"]
    )
    finalization = normalized.get("finalization")
    compact_finalization_hash = (
        str(finalization["compact_submission"]["compact_finalization_hash"])
        if isinstance(finalization, Mapping)
        else None
    )
    finalization_receipt_hash = (
        str(
            finalization["compact_submission"]["validator_receipt_delta"][
                "root_receipt_hash"
            ]
        )
        if isinstance(finalization, Mapping)
        else None
    )
    return {
        "bundle_hash": str(normalized["bundle_hash"]),
        "netuid": int(compact["weight_result"]["netuid"]),
        "epoch_id": int(compact["weight_result"]["epoch_id"]),
        "validator_hotkey": str(compact["validator_hotkey"]),
        "authority_stage": str(normalized["authority_stage"]),
        "schema_version": str(normalized["schema_version"]),
        "lineage_id": str(normalized["lineage_id"]),
        "authority_hash": str(normalized["authority_hash"]),
        "compact_submission_hash": str(compact["compact_submission_hash"]),
        "publication_receipt_hash": str(
            normalized["publication"]["publication_receipt_hash"]
        ),
        "compact_finalization_hash": compact_finalization_hash,
        "finalization_receipt_hash": finalization_receipt_hash,
        "authority_doc": dict(normalized),
    }


async def persist_compact_weight_authority_v2(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Append and exact-read one bounded public audit authority."""

    row = compact_weight_authority_storage_row_v2(authority)
    key_filters = (
        ("bundle_hash", row["bundle_hash"]),
        ("authority_stage", row["authority_stage"]),
    )
    await _insert_exact(
        COMPACT_WEIGHT_AUTHORITY_TABLE,
        row,
        key_filters=key_filters,
    )
    stored = await select_one(
        COMPACT_WEIGHT_AUTHORITY_TABLE,
        filters=key_filters,
    )
    if not isinstance(stored, Mapping):
        raise AttestedV2StoreError(
            "compact weight authority durable readback is missing"
        )
    _assert_stored_row(COMPACT_WEIGHT_AUTHORITY_TABLE, stored, row)
    return {
        "bundle_hash": row["bundle_hash"],
        "authority_stage": row["authority_stage"],
        "authority_hash": row["authority_hash"],
    }


async def load_compact_weight_authority_v2(
    *,
    bundle_hash: str,
    prefer_finalized: bool = True,
) -> Optional[dict[str, Any]]:
    """Load an exact sidecar without reconstructing any historical graph."""

    normalized_bundle_hash = str(bundle_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_bundle_hash):
        raise AttestedV2StoreError("compact weight authority bundle hash is invalid")
    stages = ("finalized", "published") if prefer_finalized else ("published",)
    for stage in stages:
        row = await select_one(
            COMPACT_WEIGHT_AUTHORITY_TABLE,
            filters=(
                ("bundle_hash", normalized_bundle_hash),
                ("authority_stage", stage),
            ),
        )
        if not isinstance(row, Mapping):
            continue
        authority = row.get("authority_doc")
        if not isinstance(authority, Mapping):
            raise AttestedV2StoreError(
                "compact weight authority document is unavailable"
            )
        normalized = validate_compact_published_weight_authority_shape_v2(
            authority
        )
        expected = _compact_weight_authority_row_from_normalized_v2(normalized)
        _assert_stored_row(COMPACT_WEIGHT_AUTHORITY_TABLE, row, expected)
        return normalized
    return None


async def load_compact_weight_authority_for_identity_v2(
    *,
    netuid: int,
    epoch_id: int,
    validator_hotkey: str,
) -> Optional[dict[str, Any]]:
    """Resolve the strongest bounded sidecar without loading the full graph."""

    for stage in ("finalized", "published"):
        row = await select_one(
            COMPACT_WEIGHT_AUTHORITY_TABLE,
            filters=(
                ("netuid", int(netuid)),
                ("epoch_id", int(epoch_id)),
                ("validator_hotkey", str(validator_hotkey)),
                ("authority_stage", stage),
            ),
        )
        if not isinstance(row, Mapping):
            continue
        authority = row.get("authority_doc")
        if not isinstance(authority, Mapping):
            raise AttestedV2StoreError(
                "compact weight authority document is unavailable"
            )
        normalized = validate_compact_published_weight_authority_shape_v2(
            authority
        )
        expected = _compact_weight_authority_row_from_normalized_v2(normalized)
        _assert_stored_row(COMPACT_WEIGHT_AUTHORITY_TABLE, row, expected)
        return normalized
    return None


async def _load_receipt_graph_batch_v2(
    root_receipt_hashes: Iterable[str],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Reconstruct one bounded batch of persisted receipt graphs."""

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
            key_fields=("receipt_hash",),
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
            key_fields=("child_receipt_hash", "parent_receipt_hash"),
            max_total_rows=None,
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
        key_fields=("boot_identity_hash",),
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
        key_fields=("receipt_hash", "attempt_hash"),
        max_total_rows=None,
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
        key_fields=("attempt_hash",),
        max_total_rows=None,
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
        key_fields=("request_hash",),
        max_total_rows=None,
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
    allowed_failed_by_graph: list[set[str]] = []
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
        allowed_failed_by_graph.append(allowed_failed.intersection(closure))
        graphs[root_hash] = {
            "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
            "root_receipt_hash": root_hash,
            "boot_identities": [
                boots[key] for key in sorted(graph_boot_hashes)
            ],
            "receipts": [receipt_docs[key] for key in sorted(closure)],
            "transport_attempts": [
                attempts[key] for key in sorted(graph_attempt_hashes)
            ],
            "host_operations": sorted(
                graph_host_operations,
                key=lambda record: record["request"]["request_hash"],
            ),
        }
    validate_receipt_graphs(
        list(graphs.values()),
        allowed_failed_receipt_hashes_by_graph=allowed_failed_by_graph,
    )
    return graphs


async def _load_legacy_receipt_graphs_v2(
    root_receipt_hashes: Iterable[str],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Legacy-only recursive expansion used during finite checkpoint bootstrap."""

    root_hashes = sorted({str(value or "") for value in root_receipt_hashes})
    allowed_failed = sorted(
        {str(value or "") for value in allowed_failed_receipt_hashes}
    )
    try:
        return await _load_receipt_graph_batch_v2(
            root_hashes,
            allowed_failed_receipt_hashes=allowed_failed,
        )
    except AttestedV2StoreError as exc:
        if (
            str(exc) != "V2 receipt graph exceeds row limit"
            or len(root_hashes) < 2
            or allowed_failed
        ):
            raise

    midpoint = len(root_hashes) // 2
    left = await _load_legacy_receipt_graphs_v2(root_hashes[:midpoint])
    right = await _load_legacy_receipt_graphs_v2(root_hashes[midpoint:])
    overlap = set(left).intersection(right)
    if overlap:
        raise AttestedV2StoreError("V2 receipt graph root is duplicated")
    return {**left, **right}


async def load_receipt_graphs_v2(
    root_receipt_hashes: Iterable[str],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Prefer constant-size checkpoint documents; bootstrap legacy roots once."""

    roots = sorted({str(value or "").lower() for value in root_receipt_hashes})
    if any(not _HASH_RE.fullmatch(value) for value in roots):
        raise AttestedV2StoreError("V2 graph root receipt hash is invalid")
    bounded = await load_checkpointed_receipt_graphs_v2(
        roots,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    missing = [root for root in roots if root not in bounded]
    if not missing:
        return bounded
    if bounded and allowed_failed_receipt_hashes:
        raise AttestedV2StoreError(
            "V2 failed receipt allowance cannot span graph authority kinds"
        )
    legacy = await _load_legacy_receipt_graphs_v2(
        missing,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    return {**bounded, **legacy}


async def load_receipt_graph_v2(
    root_receipt_hash: str,
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Load one bounded checkpoint graph or bootstrap one legacy full graph."""

    root_hash = str(root_receipt_hash or "")
    graphs = await load_receipt_graphs_v2(
        (root_hash,),
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    return graphs[root_hash]


async def _rehydrate_compact_execution_graph_v2(
    graph: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore only one compact replay root's signed local sidecars.

    Durable checkpoint documents intentionally omit transport and host-operation
    sidecars. Exact execution-result replay still needs those direct records for
    semantic source-evidence validation. Load only rows linked to this receipt,
    then validate the reconstructed local delta against the existing signed
    checkpoint certificate. Historical ancestry remains compact and is never
    recursively expanded here.
    """

    if (
        graph.get("schema_version")
        != COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
    ):
        return dict(graph)
    receipt_hash = str(receipt.get("receipt_hash") or "").lower()
    if (
        not _HASH_RE.fullmatch(receipt_hash)
        or graph.get("root_receipt_hash") != receipt_hash
    ):
        raise AttestedV2StoreError(
            "compact execution replay root receipt differs"
        )

    link_rows = await _select_by_values(
        RECEIPT_TRANSPORT_TABLE,
        field="receipt_hash",
        values=(receipt_hash,),
        key_fields=("receipt_hash", "attempt_hash"),
    )
    attempt_hashes: set[str] = set()
    for row in link_rows:
        attempt_hash = str(row.get("attempt_hash") or "").lower()
        expected = {
            "receipt_hash": receipt_hash,
            "attempt_hash": attempt_hash,
        }
        if (
            row.get("receipt_hash") != receipt_hash
            or not _HASH_RE.fullmatch(attempt_hash)
            or attempt_hash in attempt_hashes
        ):
            raise AttestedV2StoreError(
                "compact execution replay transport link is invalid"
            )
        _assert_stored_row(RECEIPT_TRANSPORT_TABLE, row, expected)
        attempt_hashes.add(attempt_hash)

    attempt_rows = await _select_by_values(
        TRANSPORT_TABLE,
        field="attempt_hash",
        values=attempt_hashes,
        key_fields=("attempt_hash",),
    )
    attempts: dict[str, dict[str, Any]] = {}
    for row in attempt_rows:
        document = row.get("attempt_doc")
        if not isinstance(document, Mapping):
            raise AttestedV2StoreError(
                "compact execution replay transport document is missing"
            )
        attempt = dict(document)
        attempt_hash = str(attempt.get("attempt_hash") or "").lower()
        if (
            attempt_hash not in attempt_hashes
            or attempt_hash in attempts
            or attempt.get("job_id") != receipt.get("job_id")
            or attempt.get("purpose") != receipt.get("purpose")
        ):
            raise AttestedV2StoreError(
                "compact execution replay transport scope differs"
            )
        validate_transport_attempt(attempt)
        _assert_stored_row(
            TRANSPORT_TABLE,
            row,
            transport_storage_row(attempt),
        )
        attempts[attempt_hash] = attempt
    if set(attempts) != attempt_hashes:
        raise AttestedV2StoreError(
            "compact execution replay is missing a transport attempt"
        )

    host_rows = await _select_by_values(
        HOST_OPERATION_TABLE,
        field="receipt_hash",
        values=(receipt_hash,),
        key_fields=("request_hash",),
    )
    host_operations: dict[str, dict[str, Any]] = {}
    for row in host_rows:
        request = row.get("request_doc")
        terminal = row.get("terminal_doc")
        if not isinstance(request, Mapping) or not isinstance(terminal, Mapping):
            raise AttestedV2StoreError(
                "compact execution replay host operation is missing"
            )
        record = {"request": dict(request), "terminal": dict(terminal)}
        request_hash = str(request.get("request_hash") or "").lower()
        if (
            not _HASH_RE.fullmatch(request_hash)
            or request_hash in host_operations
            or request.get("job_id") != receipt.get("job_id")
            or request.get("purpose") != receipt.get("purpose")
        ):
            raise AttestedV2StoreError(
                "compact execution replay host-operation scope differs"
            )
        validate_host_operation_record(record)
        _assert_stored_row(
            HOST_OPERATION_TABLE,
            row,
            host_operation_storage_row(record, receipt_hash=receipt_hash),
        )
        host_operations[request_hash] = record

    try:
        hydrated = build_checkpointed_receipt_graph(
            root_receipt_hash=receipt_hash,
            boot_identities=graph["boot_identities"],
            receipts=graph["receipts"],
            transport_attempts=[attempts[key] for key in sorted(attempts)],
            host_operations=[
                host_operations[key] for key in sorted(host_operations)
            ],
            ancestry_lineage_id=str(graph["ancestry_lineage_id"]),
            ancestry_proof=graph["ancestry_proof"],
        )
    except Exception as exc:
        raise AttestedV2StoreError(
            "compact execution replay local evidence differs"
        ) from exc
    return dict(hydrated)


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
    if str(operation) == "attest_chain_realized_settlement_v1":
        settlement = result.get("settlement_doc")
        if (
            set(result) != {"settlement_doc", "settlement_hash", "credits"}
            or not isinstance(settlement, Mapping)
            or result.get("settlement_hash")
            != sha256_json(dict(settlement))
            or not isinstance(result.get("credits"), list)
        ):
            raise AttestedV2StoreError(
                "replayable chain-realized settlement result is invalid"
            )
        return dict(settlement)
    if str(operation) == "source_add_catalog_snapshot_v2":
        expected_fields = {
            "schema_version",
            "provisioned_sources",
            "provisioned_sources_hash",
            "private_registry_rows",
            "private_registry_rows_hash",
            "runtime_catalog",
            "runtime_catalog_hash",
        }
        provisioned_sources = result.get("provisioned_sources")
        private_registry_rows = result.get("private_registry_rows")
        runtime_catalog = result.get("runtime_catalog")
        if (
            set(result) != expected_fields
            or result.get("schema_version")
            != "leadpoet.source_add_catalog_snapshot.v2"
            or not isinstance(provisioned_sources, list)
            or any(not isinstance(item, Mapping) for item in provisioned_sources)
            or not isinstance(private_registry_rows, list)
            or any(not isinstance(item, Mapping) for item in private_registry_rows)
            or not isinstance(runtime_catalog, Mapping)
        ):
            raise AttestedV2StoreError(
                "replayable SOURCE_ADD catalog result is invalid"
            )
        normalized_sources = [dict(item) for item in provisioned_sources]
        normalized_private_rows = [dict(item) for item in private_registry_rows]
        try:
            normalized_catalog = validate_source_add_runtime_catalog_v2(
                runtime_catalog
            )
            independently_derived_catalog = build_source_add_runtime_catalog_v2(
                normalized_sources
            )
        except Exception as exc:
            raise AttestedV2StoreError(
                "replayable SOURCE_ADD runtime catalog is invalid"
            ) from exc
        if (
            result.get("provisioned_sources_hash")
            != sha256_json(normalized_sources)
            or result.get("private_registry_rows_hash")
            != sha256_json(normalized_private_rows)
            or normalized_catalog != independently_derived_catalog
            or result.get("runtime_catalog_hash")
            != normalized_catalog["catalog_hash"]
        ):
            raise AttestedV2StoreError(
                "replayable SOURCE_ADD catalog commitment differs"
            )
        return dict(result)
    return dict(result)


def _source_add_catalog_secret_scan_projection_v2(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only validated encrypted credential metadata before scanning."""

    scan_result = dict(result)
    scan_sources: list[dict[str, Any]] = []
    for raw_source in result.get("provisioned_sources") or ():
        source = dict(raw_source)
        raw_envelope = source.pop("credential_envelope", {})
        provision = source.get("provision_doc")
        provider = (
            provision.get("provider_registry_entry")
            if isinstance(provision, Mapping)
            else None
        )
        if not isinstance(provider, Mapping):
            raise AttestedV2StoreError(
                "replayable SOURCE_ADD credential projection is invalid"
            )
        provider_scan = dict(provider)
        auth_kind = str(provider_scan.get("auth_kind") or "none").lower()
        envelope_ref = ""
        if auth_kind != "none":
            try:
                normalized_envelope = validate_source_add_credential_envelope_v2(
                    raw_envelope
                )
            except Exception as exc:
                raise AttestedV2StoreError(
                    "replayable SOURCE_ADD credential projection is invalid"
                ) from exc
            envelope_ref = str(normalized_envelope["credential_ref"])
        elif raw_envelope:
            raise AttestedV2StoreError(
                "replayable SOURCE_ADD credential projection is invalid"
            )

        if "credential_ref" in provider_scan:
            refs = provider_scan.pop("credential_ref")
            if not isinstance(refs, list) or any(
                not isinstance(item, str)
                or (
                    item != envelope_ref
                    and not _SOURCE_ADD_ENV_REF_RE.fullmatch(item)
                )
                for item in refs
            ):
                raise AttestedV2StoreError(
                    "replayable SOURCE_ADD credential projection is invalid"
                )
        if "credential_ready" in provider_scan:
            ready = provider_scan.pop("credential_ready")
            if (
                ready is not None
                and not isinstance(ready, bool)
                and ready != "[redacted]"
            ):
                raise AttestedV2StoreError(
                    "replayable SOURCE_ADD credential projection is invalid"
                )

        provision_scan = dict(provision)
        provision_scan["provider_registry_entry"] = provider_scan
        source["provision_doc"] = provision_scan
        scan_sources.append(source)
    scan_result["provisioned_sources"] = scan_sources

    runtime_catalog = dict(result.get("runtime_catalog") or {})
    scan_routes: list[dict[str, Any]] = []
    for raw_route in runtime_catalog.get("routes") or ():
        route = dict(raw_route)
        for field in (
            "credential_slot",
            "credential_value_hash",
            "credential_env_refs",
            "credential_envelope_hash",
        ):
            if field not in route:
                raise AttestedV2StoreError(
                    "replayable SOURCE_ADD credential projection is invalid"
                )
            route.pop(field)
        scan_routes.append(route)
    runtime_catalog["routes"] = scan_routes
    scan_result["runtime_catalog"] = runtime_catalog
    return scan_result


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
    projection = _execution_result_projection_v2(
        operation=normalized_operation,
        result=normalized_result,
    )
    from gateway.research_lab.bundles import contains_secret_material

    secret_scan_projection = normalized_result
    if normalized_operation == "source_add_catalog_snapshot_v2":
        secret_scan_projection = _source_add_catalog_secret_scan_projection_v2(
            normalized_result
        )
    if contains_secret_material(secret_scan_projection):
        raise AttestedV2StoreError(
            "replayable execution result contains secret material"
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
    graph = await _rehydrate_compact_execution_graph_v2(
        graph,
        receipt=receipt,
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


async def load_execution_result_by_receipt_v2(
    receipt_hash: str,
    *,
    expected_operation: str,
    expected_purpose: str,
    require_checkpointed_graph: bool = False,
) -> dict[str, Any]:
    """Load and validate one exact replayable result by signed receipt."""

    normalized_receipt_hash = str(receipt_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized_receipt_hash):
        raise AttestedV2StoreError("execution result receipt hash is invalid")
    stored = await select_one(
        EXECUTION_RESULT_TABLE,
        filters=(("receipt_hash", normalized_receipt_hash),),
    )
    if not isinstance(stored, Mapping):
        raise AttestedV2StoreError("execution result is unavailable")
    operation = str(expected_operation or "")
    purpose = str(expected_purpose or "")
    if (
        stored.get("operation") != operation
        or stored.get("purpose") != purpose
        or stored.get("role") != "gateway_coordinator"
    ):
        raise AttestedV2StoreError("execution result scope differs")
    if not isinstance(require_checkpointed_graph, bool):
        raise AttestedV2StoreError(
            "execution result graph requirement is invalid"
        )
    if require_checkpointed_graph:
        graph = await load_checkpointed_receipt_graph_v2(
            normalized_receipt_hash
        )
        if not isinstance(graph, Mapping):
            raise AttestedV2StoreError(
                "execution result checkpointed authority is unavailable"
            )
    else:
        graph = await load_receipt_graph_v2(normalized_receipt_hash)
    receipts = {
        str(item.get("receipt_hash") or ""): item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    receipt = receipts.get(normalized_receipt_hash)
    result = stored.get("result_doc")
    artifacts = stored.get("artifact_hashes")
    if (
        graph.get("root_receipt_hash") != normalized_receipt_hash
        or not isinstance(receipt, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(artifacts, list)
    ):
        raise AttestedV2StoreError("execution result authority is incomplete")
    expected = _execution_result_storage_row_v2(
        operation=operation,
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


def _validate_allocation_settlement_frontier_storage_v2(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        frontier_bootstrap_artifact_hashes_v2,
        validate_allocation_settlement_frontier_bootstrap_v2,
    )
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        frontier_artifact_hashes_v2,
        validate_allocation_settlement_frontier_v2,
    )

    frontier = validate_allocation_settlement_frontier_v2(
        row.get("frontier_doc")
    )
    source_result = source.get("result")
    source_row = source.get("row")
    source_receipt = source.get("receipt")
    source_artifacts = source.get("artifact_hashes")
    if (
        not isinstance(source_result, Mapping)
        or not isinstance(source_row, Mapping)
        or not isinstance(source_receipt, Mapping)
        or not isinstance(source_artifacts, list)
    ):
        raise AttestedV2StoreError("allocation frontier source is incomplete")
    source_operation = str(source_row.get("operation") or "")
    source_purpose = str(source_row.get("purpose") or "")
    if source_operation == "research_lab_allocation":
        if source_purpose != "research_lab.allocation.v2":
            raise AttestedV2StoreError(
                "allocation settlement frontier source purpose differs"
            )
        source_state = source_result.get("source_state")
        source_state_hash = str(source_result.get("source_state_hash") or "")
        required_artifacts = set(frontier_artifact_hashes_v2(frontier)) | {
            source_state_hash
        }
        source_matches = (
            isinstance(source_state, Mapping)
            and source_state_hash == sha256_json(dict(source_state))
            and source_state.get("settlement_frontier") == frontier
            and int(source_state.get("netuid", -1)) == int(frontier["netuid"])
            and int(source_state.get("epoch", -1))
            == int(frontier["allocation_epoch"])
            and int(source_row.get("epoch_id", -1))
            == int(frontier["allocation_epoch"])
            and required_artifacts.issubset(set(source_artifacts))
        )
    elif (
        source_operation == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION
        and source_purpose == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
    ):
        bootstrap = validate_allocation_settlement_frontier_bootstrap_v2(
            source_result
        )
        allocation_source = source.get("allocation_source")
        allocation_row = (
            allocation_source.get("row")
            if isinstance(allocation_source, Mapping)
            else None
        )
        allocation_result = (
            allocation_source.get("result")
            if isinstance(allocation_source, Mapping)
            else None
        )
        allocation_receipt = (
            allocation_source.get("receipt")
            if isinstance(allocation_source, Mapping)
            else None
        )
        allocation_state = (
            allocation_result.get("source_state")
            if isinstance(allocation_result, Mapping)
            else None
        )
        parent_hashes = source_receipt.get("parent_receipt_hashes")
        source_state_hash = str(bootstrap["source_state_hash"])
        required_artifacts = set(
            frontier_bootstrap_artifact_hashes_v2(bootstrap)
        )
        source_matches = (
            bootstrap["frontier"] == frontier
            and int(bootstrap["netuid"]) == int(frontier["netuid"])
            and int(bootstrap["allocation_epoch"])
            == int(frontier["allocation_epoch"])
            and int(bootstrap["bootstrap_epoch"])
            == int(source_row.get("epoch_id", -1))
            and isinstance(allocation_row, Mapping)
            and isinstance(allocation_result, Mapping)
            and isinstance(allocation_receipt, Mapping)
            and isinstance(allocation_state, Mapping)
            and allocation_row.get("operation") == "research_lab_allocation"
            and allocation_row.get("purpose") == "research_lab.allocation.v2"
            and int(allocation_row.get("epoch_id", -1))
            == int(frontier["allocation_epoch"])
            and allocation_receipt.get("receipt_hash")
            == bootstrap["allocation_source_receipt_hash"]
            and sha256_json(dict(allocation_state)) == source_state_hash
            and allocation_result.get("source_state_hash") == source_state_hash
            and int(allocation_state.get("netuid", -1))
            == int(frontier["netuid"])
            and int(allocation_state.get("epoch", -1))
            == int(frontier["allocation_epoch"])
            and allocation_state.get("settlement_frontier") is None
            and isinstance(parent_hashes, list)
            and bootstrap["allocation_source_receipt_hash"] in parent_hashes
            and required_artifacts.issubset(set(source_artifacts))
        )
    else:
        raise AttestedV2StoreError(
            "allocation settlement frontier source operation differs"
        )
    if (
        not source_matches
        or source_receipt.get("receipt_hash")
        != str(row.get("source_receipt_hash") or "")
        or str(row.get("source_state_hash") or "") != source_state_hash
        or str(row.get("frontier_hash") or "")
        != str(frontier["frontier_hash"])
        or int(row.get("netuid", -1)) != int(frontier["netuid"])
        or int(row.get("allocation_epoch", -1))
        != int(frontier["allocation_epoch"])
        or int(row.get("settled_through_epoch", -2))
        != int(frontier["settled_through_epoch"])
        or str(row.get("schema_version") or "")
        != str(frontier["schema_version"])
        or row.get("predecessor_frontier_hash")
        != frontier.get("predecessor_frontier_hash")
    ):
        raise AttestedV2StoreError("allocation settlement frontier differs")
    return {
        "frontier": frontier,
        "source": dict(source),
        "row": {
            key: row.get(key)
            for key in (
                "netuid",
                "allocation_epoch",
                "settled_through_epoch",
                "schema_version",
                "frontier_hash",
                "predecessor_frontier_hash",
                "source_receipt_hash",
                "source_state_hash",
                "frontier_doc",
            )
        },
    }


async def _load_allocation_settlement_frontier_source_v2(
    receipt_hash: str,
) -> dict[str, Any]:
    from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        validate_allocation_settlement_frontier_bootstrap_v2,
    )

    normalized = str(receipt_hash or "").lower()
    if not _HASH_RE.fullmatch(normalized):
        raise AttestedV2StoreError("allocation frontier source hash is invalid")
    row = await select_one(
        EXECUTION_RESULT_TABLE,
        filters=(("receipt_hash", normalized),),
    )
    if not isinstance(row, Mapping):
        raise AttestedV2StoreError("allocation frontier source is unavailable")
    pair = (str(row.get("operation") or ""), str(row.get("purpose") or ""))
    if pair == ("research_lab_allocation", "research_lab.allocation.v2"):
        return await load_execution_result_by_receipt_v2(
            normalized,
            expected_operation=pair[0],
            expected_purpose=pair[1],
            require_checkpointed_graph=True,
        )
    if pair != (
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    ):
        raise AttestedV2StoreError(
            "allocation frontier source operation is unsupported"
        )
    source = await load_execution_result_by_receipt_v2(
        normalized,
        expected_operation=pair[0],
        expected_purpose=pair[1],
        require_checkpointed_graph=True,
    )
    bootstrap = validate_allocation_settlement_frontier_bootstrap_v2(
        source.get("result")
    )
    allocation_source = await load_execution_result_by_receipt_v2(
        str(bootstrap["allocation_source_receipt_hash"]),
        expected_operation="research_lab_allocation",
        expected_purpose="research_lab.allocation.v2",
        require_checkpointed_graph=True,
    )
    source["allocation_source"] = allocation_source
    return source


async def load_allocation_settlement_frontier_context_v2(
    *,
    netuid: int,
    before_epoch: int,
) -> Optional[dict[str, Any]]:
    """Load the latest signed frontier before an allocation epoch."""

    normalized_netuid = int(netuid)
    normalized_epoch = int(before_epoch)
    if normalized_netuid <= 0 or normalized_epoch < 0:
        raise AttestedV2StoreError("allocation frontier scope is invalid")
    activation = await select_one(
        ALLOCATION_SETTLEMENT_FRONTIER_ACTIVATION_TABLE,
        filters=(("netuid", normalized_netuid),),
    )
    rows = await select_many(
        ALLOCATION_SETTLEMENT_FRONTIER_TABLE,
        filters=(
            ("netuid", normalized_netuid),
            ("allocation_epoch", "lt", normalized_epoch),
        ),
        order_by=(("allocation_epoch", True),),
        limit=1,
    )
    if not isinstance(activation, Mapping):
        if rows:
            raise AttestedV2StoreError(
                "allocation frontier exists without activation"
            )
        return None
    if (
        activation.get("schema_version")
        != "leadpoet.research_lab_allocation_settlement_frontier_activation.v2"
        or int(activation.get("netuid", -1)) != normalized_netuid
        or int(activation.get("first_allocation_epoch", -1)) < 1
        or int(activation.get("first_allocation_epoch", -1))
        >= normalized_epoch
        or not _HASH_RE.fullmatch(
            str(activation.get("first_frontier_hash") or "")
        )
        or not _HASH_RE.fullmatch(
            str(activation.get("source_receipt_hash") or "")
        )
        or not rows
    ):
        raise AttestedV2StoreError(
            "allocation frontier activation is incomplete"
        )
    first_epoch = int(activation["first_allocation_epoch"])
    first_row = await select_one(
        ALLOCATION_SETTLEMENT_FRONTIER_TABLE,
        filters=(
            ("netuid", normalized_netuid),
            ("allocation_epoch", first_epoch),
        ),
    )
    if not isinstance(first_row, Mapping):
        raise AttestedV2StoreError(
            "allocation frontier activation source is missing"
        )
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        validate_allocation_settlement_frontier_v2,
    )

    first_frontier = validate_allocation_settlement_frontier_v2(
        first_row.get("frontier_doc")
    )
    if (
        first_frontier.get("mode") != "legacy_full_history_bootstrap"
        or int(first_frontier.get("allocation_epoch", -1)) != first_epoch
        or first_frontier.get("predecessor_frontier_hash") is not None
        or first_frontier.get("frontier_hash")
        != activation.get("first_frontier_hash")
        or int(first_row.get("netuid", -1)) != normalized_netuid
        or int(first_row.get("allocation_epoch", -1)) != first_epoch
        or int(first_row.get("settled_through_epoch", -2))
        != int(first_frontier["settled_through_epoch"])
        or first_row.get("schema_version")
        != first_frontier.get("schema_version")
        or first_row.get("frontier_hash")
        != first_frontier.get("frontier_hash")
        or first_row.get("predecessor_frontier_hash") is not None
        or first_row.get("source_receipt_hash")
        != activation.get("source_receipt_hash")
    ):
        raise AttestedV2StoreError(
            "allocation frontier activation source differs"
        )
    row = rows[0]
    first_receipt_hash = str(first_row.get("source_receipt_hash") or "")
    latest_receipt_hash = str(row.get("source_receipt_hash") or "")
    first_source = await _load_allocation_settlement_frontier_source_v2(
        first_receipt_hash
    )
    first_validated = _validate_allocation_settlement_frontier_storage_v2(
        first_row,
        source=first_source,
    )
    if first_validated["frontier"] != first_frontier:
        raise AttestedV2StoreError(
            "allocation frontier activation authority differs"
        )
    source = (
        first_source
        if latest_receipt_hash == first_receipt_hash
        else await _load_allocation_settlement_frontier_source_v2(
            latest_receipt_hash
        )
    )
    validated = _validate_allocation_settlement_frontier_storage_v2(
        row,
        source=source,
    )
    validated["activation"] = dict(activation)
    validated["activation_source"] = dict(first_source)
    return validated


async def persist_allocation_settlement_frontier_v2(
    *,
    frontier: Mapping[str, Any],
    source_receipt_hash: str,
    source_state_hash: str,
) -> dict[str, Any]:
    """Atomically append one signed frontier and exact predecessor edge."""

    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        validate_allocation_settlement_frontier_v2,
    )

    normalized_frontier = validate_allocation_settlement_frontier_v2(frontier)
    normalized_receipt_hash = str(source_receipt_hash or "").lower()
    normalized_source_state_hash = str(source_state_hash or "").lower()
    if (
        not _HASH_RE.fullmatch(normalized_receipt_hash)
        or not _HASH_RE.fullmatch(normalized_source_state_hash)
    ):
        raise AttestedV2StoreError("allocation frontier source hash is invalid")
    source = await _load_allocation_settlement_frontier_source_v2(
        normalized_receipt_hash
    )
    candidate_row = {
        "netuid": int(normalized_frontier["netuid"]),
        "allocation_epoch": int(normalized_frontier["allocation_epoch"]),
        "settled_through_epoch": int(
            normalized_frontier["settled_through_epoch"]
        ),
        "schema_version": str(normalized_frontier["schema_version"]),
        "frontier_hash": str(normalized_frontier["frontier_hash"]),
        "predecessor_frontier_hash": normalized_frontier.get(
            "predecessor_frontier_hash"
        ),
        "source_receipt_hash": normalized_receipt_hash,
        "source_state_hash": normalized_source_state_hash,
        "frontier_doc": normalized_frontier,
    }
    _validate_allocation_settlement_frontier_storage_v2(
        candidate_row,
        source=source,
    )
    source_row = source.get("row")
    if not isinstance(source_row, Mapping):
        raise AttestedV2StoreError("allocation frontier source is incomplete")
    source_operation = str(source_row.get("operation") or "")
    rpc_name = (
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_RPC
        if source_operation == "allocation_settlement_frontier_bootstrap_v2"
        else ALLOCATION_SETTLEMENT_FRONTIER_RPC
    )
    rpc_payload = {
        "requested_frontier": normalized_frontier,
        "requested_source_receipt_hash": normalized_receipt_hash,
        "requested_source_state_hash": normalized_source_state_hash,
    }
    result: Any = None
    for attempt in range(_EXACT_INSERT_ATTEMPTS):
        try:
            result = await call_rpc(
                rpc_name,
                rpc_payload,
            )
            break
        except Exception as exc:
            if not _is_transient_store_error(exc):
                raise
            try:
                stored_after_error = await select_one(
                    ALLOCATION_SETTLEMENT_FRONTIER_TABLE,
                    filters=(
                        ("netuid", int(normalized_frontier["netuid"])),
                        (
                            "allocation_epoch",
                            int(normalized_frontier["allocation_epoch"]),
                        ),
                    ),
                )
            except Exception as readback_exc:
                if not _is_transient_store_error(readback_exc):
                    raise
                stored_after_error = None
            if isinstance(stored_after_error, Mapping):
                _assert_stored_row(
                    ALLOCATION_SETTLEMENT_FRONTIER_TABLE,
                    stored_after_error,
                    candidate_row,
                )
                _validate_allocation_settlement_frontier_storage_v2(
                    stored_after_error,
                    source=source,
                )
                result = {
                    "status": "already_persisted",
                    "netuid": int(normalized_frontier["netuid"]),
                    "allocation_epoch": int(
                        normalized_frontier["allocation_epoch"]
                    ),
                    "frontier_hash": str(normalized_frontier["frontier_hash"]),
                    "source_receipt_hash": normalized_receipt_hash,
                    "source_state_hash": normalized_source_state_hash,
                }
                break
            if attempt == _EXACT_INSERT_ATTEMPTS - 1:
                raise
            backoff = _EXACT_INSERT_BACKOFF_SECONDS[
                min(attempt, len(_EXACT_INSERT_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "allocation_settlement_frontier_retry netuid=%s epoch=%s "
                "attempt=%s/%s type=%s error=%s",
                normalized_frontier["netuid"],
                normalized_frontier["allocation_epoch"],
                attempt + 1,
                _EXACT_INSERT_ATTEMPTS,
                type(exc).__name__,
                str(exc)[:160],
            )
            await asyncio.sleep(backoff)
    if (
        not isinstance(result, Mapping)
        or result.get("status") not in {"persisted", "already_persisted"}
        or int(result.get("netuid", -1)) != int(normalized_frontier["netuid"])
        or int(result.get("allocation_epoch", -1))
        != int(normalized_frontier["allocation_epoch"])
        or result.get("frontier_hash") != normalized_frontier["frontier_hash"]
        or result.get("source_receipt_hash") != normalized_receipt_hash
        or result.get("source_state_hash") != normalized_source_state_hash
    ):
        raise AttestedV2StoreError(
            "allocation frontier durable readback differs"
        )
    stored = await select_one(
        ALLOCATION_SETTLEMENT_FRONTIER_TABLE,
        filters=(
            ("netuid", int(normalized_frontier["netuid"])),
            (
                "allocation_epoch",
                int(normalized_frontier["allocation_epoch"]),
            ),
        ),
    )
    if not isinstance(stored, Mapping):
        raise AttestedV2StoreError(
            "allocation frontier durable readback is missing"
        )
    stored_receipt_hash = str(stored.get("source_receipt_hash") or "")
    stored_source = await _load_allocation_settlement_frontier_source_v2(
        stored_receipt_hash
    )
    validated = _validate_allocation_settlement_frontier_storage_v2(
        stored,
        source=stored_source,
    )
    if (
        validated["frontier"] != normalized_frontier
        or str(stored.get("source_state_hash") or "")
        != normalized_source_state_hash
        or result.get("source_receipt_hash") != stored_receipt_hash
    ):
        raise AttestedV2StoreError(
            "allocation frontier durable readback differs"
        )
    return dict(result)


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
    graphs = await load_business_artifact_graphs_v2(
        ((kind, ref, digest),),
        allow_failed_root=allow_failed_root,
    )
    return graphs[(kind, ref, digest)]


async def load_business_artifact_graphs_v2(
    artifacts: Iterable[tuple[str, str, str]],
    *,
    allow_failed_root: bool = False,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Resolve exact immutable artifacts while loading shared ancestry once."""

    requested = sorted(
        {
            (
                str(kind or "").strip(),
                str(ref or "").strip(),
                str(digest or "").lower(),
            )
            for kind, ref, digest in artifacts
        }
    )
    if not requested:
        return {}
    if (
        len(requested) > _MAX_GRAPH_ROWS
        or any(
            not kind or not ref or not _HASH_RE.fullmatch(digest)
            for kind, ref, digest in requested
        )
    ):
        raise AttestedV2StoreError("V2 business artifact lookup is invalid")

    requested_set = set(requested)
    refs_by_kind: dict[str, list[str]] = {}
    for kind, ref, _digest in requested:
        refs_by_kind.setdefault(kind, []).append(ref)

    rows_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
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
                order_by=(
                    ("artifact_kind", False),
                    ("artifact_ref", False),
                    ("artifact_hash", False),
                ),
                max_rows=_MAX_GRAPH_ROWS,
                allow_partial=False,
            )
            for row in rows:
                key = (
                    str(row.get("artifact_kind") or ""),
                    str(row.get("artifact_ref") or ""),
                    str(row.get("artifact_hash") or "").lower(),
                )
                if key not in requested_set:
                    continue
                if key in rows_by_key:
                    raise AttestedV2StoreError(
                        "V2 business artifact lineage is missing or ambiguous"
                    )
                receipt_hash = str(row.get("receipt_hash") or "").lower()
                if (
                    row.get("artifact_kind") != key[0]
                    or row.get("artifact_ref") != key[1]
                    or row.get("artifact_hash") != key[2]
                    or row.get("receipt_hash") != receipt_hash
                    or not _HASH_RE.fullmatch(receipt_hash)
                ):
                    raise AttestedV2StoreError(
                        "V2 business artifact row conflicts"
                    )
                rows_by_key[key] = row

    if set(rows_by_key) != requested_set:
        raise AttestedV2StoreError(
            "V2 business artifact lineage is missing or ambiguous"
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

    resolved: dict[tuple[str, str, str], dict[str, Any]] = {}
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
                order_by=(
                    ("artifact_kind", False),
                    ("artifact_ref", False),
                    ("artifact_hash", False),
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


async def persist_chain_realized_settlement_v1(
    *,
    package: Mapping[str, Any],
    receipt_hash: str,
) -> dict[str, Any]:
    """Atomically persist one complete coordinator-attested settlement."""

    from gateway.research_lab.champion_settlement_v2 import (
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1,
        CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1,
        CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2,
        validate_chain_realized_epoch_settlements_v1,
        validate_chain_realized_obligation_credits_v1,
    )

    if not isinstance(package, Mapping) or set(package) != {
        "settlement_doc",
        "settlement_hash",
        "credits",
    }:
        raise AttestedV2StoreError(
            "chain-realized settlement package fields are invalid"
        )
    settlement_doc = package.get("settlement_doc")
    credits = package.get("credits")
    settlement_hash = str(package.get("settlement_hash") or "").lower()
    normalized_receipt_hash = str(receipt_hash or "").lower()
    if (
        not isinstance(settlement_doc, Mapping)
        or not isinstance(credits, list)
        or settlement_hash != sha256_json(dict(settlement_doc))
        or not _HASH_RE.fullmatch(settlement_hash)
        or not _HASH_RE.fullmatch(normalized_receipt_hash)
    ):
        raise AttestedV2StoreError(
            "chain-realized settlement package is invalid"
        )
    try:
        netuid = int(settlement_doc["netuid"])
        epoch_id = int(settlement_doc["epoch_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AttestedV2StoreError(
            "chain-realized settlement scope is invalid"
        ) from exc

    loaded_graphs = await load_receipt_graphs_v2({normalized_receipt_hash})
    graph = loaded_graphs.get(normalized_receipt_hash)
    if not isinstance(graph, Mapping):
        raise AttestedV2StoreError(
            "chain-realized settlement receipt graph is not durable"
        )
    graph_by_root = {normalized_receipt_hash: dict(graph)}
    settlement_row = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "schema_version": str(settlement_doc.get("schema_version") or ""),
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": normalized_receipt_hash,
        "settlement_doc": dict(settlement_doc),
    }
    credit_rows: list[dict[str, Any]] = []
    for raw_credit in credits:
        if not isinstance(raw_credit, Mapping) or set(raw_credit) != {
            "credit_hash",
            "credit_doc",
        }:
            raise AttestedV2StoreError(
                "chain-realized credit package fields are invalid"
            )
        document = raw_credit.get("credit_doc")
        credit_hash = str(raw_credit.get("credit_hash") or "").lower()
        if (
            not isinstance(document, Mapping)
            or credit_hash != sha256_json(dict(document))
            or not _HASH_RE.fullmatch(credit_hash)
        ):
            raise AttestedV2StoreError(
                "chain-realized credit package is invalid"
            )
        credit_rows.append(
            {
                "netuid": int(document["netuid"]),
                "epoch_id": int(document["epoch_id"]),
                "settlement_hash": settlement_hash,
                "schema_version": str(document["schema_version"]),
                "obligation_kind": str(document["obligation_kind"]),
                "obligation_source_id": str(
                    document["obligation_source_id"]
                ),
                "miner_hotkey": str(document["miner_hotkey"]),
                "miner_uid": int(document["miner_uid"]),
                "observed_chain_alpha_percent": str(
                    document["observed_chain_alpha_percent"]
                ),
                "lab_attributed_alpha_percent": str(
                    document["lab_attributed_alpha_percent"]
                ),
                "scheduled_alpha_percent": str(
                    document["scheduled_alpha_percent"]
                ),
                "credited_alpha_percent": str(
                    document["credited_alpha_percent"]
                ),
                "champion_credit_policy": str(
                    document.get("champion_credit_policy")
                    or CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1
                ),
                "credit_hash": credit_hash,
                "credit_receipt_hash": normalized_receipt_hash,
                "credit_doc": dict(document),
            }
        )
    credit_rows.sort(key=lambda row: str(row["credit_hash"]))

    settlements = validate_chain_realized_epoch_settlements_v1(
        [settlement_row],
        receipt_graphs=graph_by_root,
    )
    validate_chain_realized_obligation_credits_v1(
        credit_rows,
        settlement_rows=settlements,
        receipt_graphs=graph_by_root,
    )
    if (
        settlement_row["schema_version"]
        not in {
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        }
        or (
            settlement_row["schema_version"]
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2
            and credit_rows
        )
        or any(
            row["schema_version"]
            not in {
                CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1,
                CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2,
            }
            for row in credit_rows
        )
        or (
            settlement_row["schema_version"]
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            and any(
                row["schema_version"]
                != CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
                for row in credit_rows
            )
        )
        or (
            settlement_row["schema_version"]
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1
            and any(
                row["schema_version"]
                != CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1
                for row in credit_rows
            )
        )
    ):
        raise AttestedV2StoreError(
            "chain-realized settlement schema differs"
        )

    result: Any = None
    persistence_rpc = {
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1: (
            CHAIN_REALIZED_SETTLEMENT_RPC
        ),
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2: (
            CHAIN_REALIZED_UNATTRIBUTED_SETTLEMENT_RPC
        ),
        CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3: (
            CHAIN_REALIZED_LIFETIME_SETTLEMENT_RPC
        ),
    }[settlement_row["schema_version"]]
    for attempt in range(_EXACT_INSERT_ATTEMPTS):
        try:
            result = await call_rpc(
                persistence_rpc,
                {
                    "requested_settlement": settlement_row,
                    "requested_credits": credit_rows,
                },
            )
            break
        except Exception as exc:
            if (
                not _is_transient_store_error(exc)
                or attempt == _EXACT_INSERT_ATTEMPTS - 1
            ):
                raise
            backoff = _EXACT_INSERT_BACKOFF_SECONDS[
                min(attempt, len(_EXACT_INSERT_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "chain_realized_settlement_retry epoch=%s attempt=%s/%s "
                "type=%s error=%s",
                epoch_id,
                attempt + 1,
                _EXACT_INSERT_ATTEMPTS,
                type(exc).__name__,
                str(exc)[:160],
            )
            await asyncio.sleep(backoff)
    if not isinstance(result, Mapping):
        raise AttestedV2StoreError(
            "chain-realized settlement RPC response is invalid"
        )
    expected_result = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_persistence.v1"
        ),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": normalized_receipt_hash,
        "credit_count": len(credit_rows),
        "credit_hashes": [
            str(row["credit_hash"]) for row in credit_rows
        ],
    }
    if dict(result) != expected_result:
        raise AttestedV2StoreError(
            "chain-realized settlement RPC response differs"
        )

    stored_settlement = await select_one(
        CHAIN_REALIZED_SETTLEMENT_TABLE,
        filters=(("netuid", netuid), ("epoch_id", epoch_id)),
    )
    if not isinstance(stored_settlement, Mapping):
        raise AttestedV2StoreError(
            "chain-realized settlement durable readback is missing"
        )
    stored_credits = await select_all(
        CHAIN_REALIZED_CREDIT_TABLE,
        filters=(("netuid", netuid), ("epoch_id", epoch_id)),
        order_by=(("credit_hash", False),),
        max_rows=max(1, len(credit_rows) + 1),
        allow_partial=False,
    )
    durable_settlements = validate_chain_realized_epoch_settlements_v1(
        [stored_settlement],
        receipt_graphs=graph_by_root,
    )
    durable_allocations = validate_chain_realized_obligation_credits_v1(
        stored_credits,
        settlement_rows=durable_settlements,
        receipt_graphs=graph_by_root,
    )
    expected_credit_hashes = sorted(
        str(row["credit_hash"]) for row in credit_rows
    )
    stored_credit_hashes = sorted(
        str(row.get("credit_hash") or "") for row in stored_credits
    )
    durable_credit_hashes = (
        sorted(
            str(item)
            for item in (
                durable_allocations[0].get("chain_realized_credit_hashes")
                or ()
            )
        )
        if len(durable_allocations) == 1
        else []
    )
    if (
        len(durable_settlements) != 1
        or len(durable_allocations) != 1
        or int(durable_allocations[0].get("netuid", -1)) != netuid
        or int(durable_allocations[0].get("epoch", -1)) != epoch_id
        or durable_allocations[0].get("chain_realized_settlement_hash")
        != settlement_hash
        or durable_allocations[0].get(
            "chain_realized_settlement_receipt_hash"
        )
        != normalized_receipt_hash
        or stored_credit_hashes != expected_credit_hashes
        or durable_credit_hashes != expected_credit_hashes
    ):
        raise AttestedV2StoreError(
            "chain-realized settlement durable readback differs"
        )
    return {
        **expected_result,
        "durable_readback_hash": sha256_json(
            {
                "settlement": dict(stored_settlement),
                "credits": [dict(row) for row in stored_credits],
            }
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
        if not require_finalization:
            return None
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
