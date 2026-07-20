"""Append-only persistence for receipt-backed stateful subnet epochs.

Every row is derived from a canonical finalized snapshot and is inserted with
exact readback semantics.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import re
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
)
from gateway.research_lab.attested_v2_store import (
    load_receipt_graph_v2,
    persist_receipt_graph_v2,
)
from gateway.research_lab.store import insert_row, select_one
from gateway.tee.coordinator_epoch_cutover_v2 import (
    CUTOVER_AUTHORITY_SCHEMA_VERSION,
    CUTOVER_PURPOSE,
    FINALIZATION_PURPOSE,
    SNAPSHOT_PURPOSE,
)
from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    WEIGHT_ROLE,
    merkle_root,
    sha256_bytes,
    sha256_json,
    validate_boot_identity,
    validate_receipt_graph,
)


CANDIDATE_TABLE = "research_lab_stateful_subnet_epoch_candidates_v1"
CUTOVER_TABLE = "research_lab_stateful_subnet_epoch_cutovers_v1"
BOUNDARY_TABLE = "research_lab_stateful_subnet_epoch_boundaries_v1"
SNAPSHOT_TABLE = "research_lab_stateful_subnet_epoch_snapshots_v1"

BOUNDARY_SCHEMA_VERSION = "leadpoet.subnet_epoch_boundary.v1"
CAPTURE_SCHEMA_VERSION = "leadpoet.subnet_epoch_boundary_capture.v1"
EVIDENCE_SCHEMA_VERSION = "leadpoet.validator_subnet_epoch_evidence.v1"
CANDIDATE_SUBMISSION_SCHEMA_VERSION = (
    "leadpoet.subnet_epoch_boundary_candidate_submission.v1"
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HOTKEY_SIGNATURE_RE = re.compile(r"^0x[0-9a-f]{128}$")
_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "epoch_scheme",
        "network_genesis_hash",
        "netuid",
        "head_kind",
        "block_hash",
        "current_block",
        "last_epoch_block",
        "pending_epoch_at",
        "subnet_epoch_index",
        "tempo",
        "blocks_since_last_step",
        "observed_at",
        "epoch_id",
        "epoch_ref",
        "epoch_block",
        "next_epoch_block",
        "blocks_remaining",
        "settlement_epoch_id",
        "cutover_mapping_hash",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "mapping_hash",
        "first_epoch_ref",
        "first_snapshot_hash",
        "first_snapshot_receipt_hash",
        "last_legacy_bundle_hash",
        "last_legacy_weight_finalization_event_hash",
        "last_legacy_finalization_receipt_hash",
        "manifest",
    }
)
_CAPTURE_FIELDS = frozenset(
    {
        "schema_version",
        "epoch_authority",
        "epoch_boundary",
        "epoch_authority_receipt_hash",
        "epoch_boundary_receipt_hash",
        "receipt_graph",
        "boot_identity",
        "source_artifacts",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "validator_hotkey",
        "bundle_hash",
        "cutover_mapping_hash",
        "epoch_authority",
        "epoch_authority_hash",
        "epoch_authority_receipt_hash",
        "epoch_boundary",
        "epoch_boundary_hash",
        "epoch_boundary_receipt_hash",
        "receipt_graph",
    }
)
_CANDIDATE_ROW_FIELDS = frozenset(
    {
        "snapshot_hash",
        "schema_version",
        "mapping_hash",
        "epoch_scheme",
        "network_genesis_hash",
        "netuid",
        "head_kind",
        "block_hash",
        "current_block",
        "last_epoch_block",
        "pending_epoch_at",
        "subnet_epoch_index",
        "epoch_ref",
        "proposed_settlement_epoch_id",
        "tempo",
        "blocks_since_last_step",
        "next_epoch_block",
        "blocks_remaining",
        "chain_state_receipt_hash",
        "validator_hotkey",
        "candidate_payload_hash",
        "validator_hotkey_signature",
        "candidate_authorization_hash",
        "snapshot_doc",
        "observed_at",
        "created_at",
    }
)


class StatefulEpochAuthorityStoreError(RuntimeError):
    """Receipt-backed epoch state is malformed, conflicting, or not durable."""


def _is_duplicate_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "duplicate" in message or "unique" in message or "23505" in message


def _row_value_equal(field: str, observed: Any, expected: Any) -> bool:
    if observed == expected:
        return True
    if field not in {"observed_at", "first_observed_at"}:
        return False
    try:
        observed_time = datetime.fromisoformat(
            str(observed).replace("Z", "+00:00")
        )
        expected_time = datetime.fromisoformat(
            str(expected).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    if observed_time.tzinfo is None or expected_time.tzinfo is None:
        return False
    return observed_time.astimezone(timezone.utc) == expected_time.astimezone(
        timezone.utc
    )


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise StatefulEpochAuthorityStoreError("%s is invalid" % field)
    return normalized


def _cutover(value: Any) -> SubnetEpochCutover:
    if not isinstance(value, Mapping):
        raise StatefulEpochAuthorityStoreError("epoch cutover is invalid")
    try:
        cutover = SubnetEpochCutover.from_mapping(value)
    except (TypeError, SubnetEpochError) as exc:
        raise StatefulEpochAuthorityStoreError("epoch cutover is invalid") from exc
    if dict(value) != cutover.to_dict():
        raise StatefulEpochAuthorityStoreError("epoch cutover is not canonical")
    return cutover


def validate_snapshot_document_v1(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
    require_boundary: bool,
) -> Tuple[SubnetEpochSnapshot, Dict[str, Any]]:
    """Recompute every derived snapshot field and the settlement mapping."""

    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
        raise StatefulEpochAuthorityStoreError(
            "subnet epoch snapshot fields are invalid"
        )
    supplied = dict(value)
    try:
        snapshot = SubnetEpochSnapshot.from_mapping(supplied)
        expected = snapshot.to_dict(cutover=cutover)
    except (KeyError, TypeError, SubnetEpochError) as exc:
        raise StatefulEpochAuthorityStoreError(
            "subnet epoch snapshot is invalid"
        ) from exc
    if supplied != expected:
        raise StatefulEpochAuthorityStoreError(
            "subnet epoch snapshot derivations differ"
        )
    if (
        snapshot.head_kind != "finalized"
        or snapshot.network_genesis_hash != cutover.network_genesis_hash
        or snapshot.netuid != cutover.netuid
    ):
        raise StatefulEpochAuthorityStoreError(
            "subnet epoch snapshot lineage differs"
        )
    if require_boundary and (
        snapshot.current_block != snapshot.last_epoch_block
        or snapshot.epoch_block != 0
    ):
        raise StatefulEpochAuthorityStoreError(
            "subnet epoch boundary is not an exact finalized transition"
        )
    return snapshot, expected


def _snapshot_receipt(
    graph: Mapping[str, Any],
    *,
    snapshot_doc: Mapping[str, Any],
    declared_receipt_hash: Any,
) -> Tuple[Mapping[str, Any], str]:
    validate_receipt_graph(graph)
    snapshot_hash = sha256_json(dict(snapshot_doc))
    receipt_hash = _hash(
        declared_receipt_hash,
        "declared subnet epoch snapshot receipt hash",
    )
    matches = [
        receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
        and receipt.get("receipt_hash") == receipt_hash
        and receipt.get("role") == WEIGHT_ROLE
        and receipt.get("purpose") == SNAPSHOT_PURPOSE
        and receipt.get("status") == "succeeded"
        and int(receipt.get("epoch_id", -1))
        == int(snapshot_doc["settlement_epoch_id"])
        and receipt.get("output_root") == snapshot_hash
    ]
    if len(matches) != 1:
        raise StatefulEpochAuthorityStoreError(
            "declared snapshot receipt does not bind the boundary document"
        )
    return matches[0], snapshot_hash


def validate_epoch_evidence_envelope_v1(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
) -> Dict[str, Any]:
    """Validate one exact validator shadow-capture or normal evidence object."""

    if not isinstance(value, Mapping):
        raise StatefulEpochAuthorityStoreError(
            "validator subnet epoch evidence is invalid"
        )
    schema = value.get("schema_version")
    observed_fields = frozenset(value)
    if schema == CAPTURE_SCHEMA_VERSION:
        if observed_fields != _CAPTURE_FIELDS:
            raise StatefulEpochAuthorityStoreError(
                "subnet epoch capture fields are invalid"
            )
    elif schema == EVIDENCE_SCHEMA_VERSION:
        if observed_fields != _EVIDENCE_FIELDS:
            raise StatefulEpochAuthorityStoreError(
                "validator subnet epoch evidence fields are invalid"
            )
    else:
        raise StatefulEpochAuthorityStoreError(
            "validator subnet epoch evidence schema is invalid"
        )

    current_value = value.get("epoch_authority")
    boundary_value = value.get("epoch_boundary")
    graph = value.get("receipt_graph")
    if (
        not isinstance(current_value, Mapping)
        or not isinstance(boundary_value, Mapping)
        or not isinstance(graph, Mapping)
    ):
        raise StatefulEpochAuthorityStoreError(
            "validator subnet epoch evidence documents are invalid"
        )
    validate_receipt_graph(graph)
    current, current_doc = validate_snapshot_document_v1(
        current_value,
        cutover=cutover,
        require_boundary=False,
    )
    boundary, boundary_doc = validate_snapshot_document_v1(
        boundary_value,
        cutover=cutover,
        require_boundary=True,
    )
    if (
        current.subnet_epoch_index != boundary.subnet_epoch_index
        or current.epoch_ref != boundary.epoch_ref
        or current.last_epoch_block != boundary.current_block
        or current.current_block < boundary.current_block
        or current.settlement_epoch_id(cutover)
        != boundary.settlement_epoch_id(cutover)
    ):
        raise StatefulEpochAuthorityStoreError(
            "current subnet epoch snapshot does not bind its boundary"
        )
    current_receipt, current_hash = _snapshot_receipt(
        graph,
        snapshot_doc=current_doc,
        declared_receipt_hash=value.get("epoch_authority_receipt_hash"),
    )
    boundary_receipt, boundary_hash = _snapshot_receipt(
        graph,
        snapshot_doc=boundary_doc,
        declared_receipt_hash=value.get("epoch_boundary_receipt_hash"),
    )

    if schema == CAPTURE_SCHEMA_VERSION:
        boot = value.get("boot_identity")
        artifacts = value.get("source_artifacts")
        if not isinstance(boot, Mapping) or not isinstance(artifacts, list):
            raise StatefulEpochAuthorityStoreError(
                "subnet epoch capture evidence is invalid"
            )
        try:
            validate_boot_identity(boot)
        except Exception as exc:
            raise StatefulEpochAuthorityStoreError(
                "subnet epoch capture boot identity is invalid"
            ) from exc
        graph_boots = {
            str(item.get("boot_identity_hash") or ""): item
            for item in graph.get("boot_identities") or ()
            if isinstance(item, Mapping)
        }
        artifacts_by_hash = {}
        for item in artifacts:
            if not isinstance(item, Mapping) or set(item) != {
                "artifact_hash",
                "kind",
                "body_b64",
            }:
                raise StatefulEpochAuthorityStoreError(
                    "subnet epoch capture source artifact fields are invalid"
                )
            try:
                body = base64.b64decode(str(item["body_b64"]), validate=True)
            except Exception as exc:
                raise StatefulEpochAuthorityStoreError(
                    "subnet epoch capture source artifact is invalid"
                ) from exc
            artifact_hash = str(item.get("artifact_hash") or "")
            if (
                sha256_bytes(body) != artifact_hash
                or not str(item.get("kind") or "").strip()
                or artifact_hash in artifacts_by_hash
            ):
                raise StatefulEpochAuthorityStoreError(
                    "subnet epoch capture source artifact differs"
                )
            artifacts_by_hash[artifact_hash] = dict(item)
        attempts = list(graph.get("transport_attempts") or ())
        required_artifacts = set()
        for attempt in attempts:
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("job_id") != current_receipt.get("job_id")
                or attempt.get("purpose") != SNAPSHOT_PURPOSE
            ):
                raise StatefulEpochAuthorityStoreError(
                    "subnet epoch capture transport scope differs"
                )
            required_artifacts.add(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                required_artifacts.add(str(attempt["response_artifact_hash"]))
        expected_artifact_root = merkle_root(
            [current_hash, *required_artifacts],
            domain="leadpoet-artifact-v2",
        )
        if (
            boot.get("role") != WEIGHT_ROLE
            or graph_boots.get(str(boot.get("boot_identity_hash") or ""))
            != dict(boot)
            or current_doc != boundary_doc
            or current_receipt["receipt_hash"]
            != boundary_receipt["receipt_hash"]
            or graph.get("root_receipt_hash")
            != boundary_receipt["receipt_hash"]
            or len(graph.get("receipts") or ()) != 1
            or boundary_receipt.get("parent_receipt_hashes") != []
            or graph.get("host_operations") != []
            or set(artifacts_by_hash) != required_artifacts
            or boundary_receipt.get("artifact_root")
            != expected_artifact_root
        ):
            raise StatefulEpochAuthorityStoreError(
                "subnet epoch shadow capture is not one exact boundary receipt"
            )
    else:
        if (
            value.get("cutover_mapping_hash") != cutover.mapping_hash
            or _hash(value.get("epoch_authority_hash"), "epoch authority hash")
            != current_hash
            or _hash(value.get("epoch_boundary_hash"), "epoch boundary hash")
            != boundary_hash
        ):
            raise StatefulEpochAuthorityStoreError(
                "validator subnet epoch evidence commitments differ"
            )
        _hash(value.get("bundle_hash"), "validator bundle hash")
        if not _SS58_RE.fullmatch(
            str(value.get("validator_hotkey") or "").strip()
        ):
            raise StatefulEpochAuthorityStoreError(
                "validator subnet epoch evidence hotkey is invalid"
            )
    return {
        "schema_version": schema,
        "current": current,
        "current_doc": current_doc,
        "current_hash": current_hash,
        "current_receipt": current_receipt,
        "boundary": boundary,
        "boundary_doc": boundary_doc,
        "boundary_hash": boundary_hash,
        "boundary_receipt": boundary_receipt,
        "receipt_graph": graph,
    }


def build_pre_cutover_candidate_row_v1(
    envelope: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
    validator_hotkey: str,
    candidate_payload_hash: str,
    validator_hotkey_signature: str,
    candidate_authorization_hash: str,
) -> Dict[str, Any]:
    """Build the exact append-only candidate row defined by migration 100."""

    mapping = _cutover(cutover)
    evidence = validate_epoch_evidence_envelope_v1(
        envelope,
        cutover=mapping,
    )
    if evidence["schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise StatefulEpochAuthorityStoreError(
            "pre-cutover candidate requires shadow capture evidence"
        )
    snapshot = evidence["boundary"]
    snapshot_doc = evidence["boundary_doc"]
    if (
        snapshot.current_block != mapping.cutover_block
        or snapshot.block_hash != mapping.cutover_block_hash
        or snapshot.subnet_epoch_index != mapping.first_subnet_epoch_index
        or snapshot.settlement_epoch_id(mapping)
        != mapping.first_settlement_epoch_id
    ):
        raise StatefulEpochAuthorityStoreError(
            "candidate snapshot differs from proposed cutover"
        )
    hotkey = str(validator_hotkey or "").strip()
    signature = str(validator_hotkey_signature or "").strip()
    if not _SS58_RE.fullmatch(hotkey):
        raise StatefulEpochAuthorityStoreError(
            "candidate validator hotkey is invalid"
        )
    if not _HOTKEY_SIGNATURE_RE.fullmatch(signature):
        raise StatefulEpochAuthorityStoreError(
            "candidate validator signature is not canonical"
        )
    expected_payload_hash = sha256_json(
        {
            "schema_version": CANDIDATE_SUBMISSION_SCHEMA_VERSION,
            "cutover_manifest": mapping.to_dict(),
            "capture": dict(envelope),
        }
    )
    if _hash(candidate_payload_hash, "candidate payload hash") != expected_payload_hash:
        raise StatefulEpochAuthorityStoreError(
            "candidate payload hash differs"
        )
    expected_authorization_hash = sha256_json(
        {
            "validator_hotkey": hotkey,
            "candidate_payload_hash": expected_payload_hash,
            "validator_hotkey_signature": signature,
        }
    )
    if (
        _hash(candidate_authorization_hash, "candidate authorization hash")
        != expected_authorization_hash
    ):
        raise StatefulEpochAuthorityStoreError(
            "candidate authorization hash differs"
        )
    receipt = evidence["boundary_receipt"]
    snapshot_hash = evidence["boundary_hash"]
    return {
        "snapshot_hash": snapshot_hash,
        "schema_version": snapshot.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "epoch_scheme": snapshot.epoch_scheme,
        "network_genesis_hash": mapping.network_genesis_hash,
        "netuid": mapping.netuid,
        "head_kind": snapshot.head_kind,
        "block_hash": snapshot.block_hash,
        "current_block": snapshot.current_block,
        "last_epoch_block": snapshot.last_epoch_block,
        "pending_epoch_at": snapshot.pending_epoch_at,
        "subnet_epoch_index": snapshot.subnet_epoch_index,
        "epoch_ref": snapshot.epoch_ref,
        "proposed_settlement_epoch_id": snapshot.settlement_epoch_id(mapping),
        "tempo": snapshot.tempo,
        "blocks_since_last_step": snapshot.blocks_since_last_step,
        "next_epoch_block": snapshot.next_epoch_block,
        "blocks_remaining": snapshot.blocks_remaining,
        "chain_state_receipt_hash": receipt["receipt_hash"],
        "validator_hotkey": hotkey,
        "candidate_payload_hash": expected_payload_hash,
        "validator_hotkey_signature": signature,
        "candidate_authorization_hash": expected_authorization_hash,
        "snapshot_doc": snapshot_doc,
        "observed_at": snapshot.observed_at,
    }


def validate_stored_pre_cutover_candidate_row_v1(
    value: Any,
    *,
    cutover: Mapping[str, Any],
    receipt_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one staged candidate using only its durable SQL evidence.

    The route has already verified the wallet signature and persisted the hash
    of the full capture (including source artifact bodies).  Those bodies are
    intentionally absent from the SQL row, so this readback validator checks
    the stored authorization commitment, the complete receipt graph, and every
    reconstructable manifest/snapshot column without pretending to recreate
    the signed capture payload.
    """

    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_ROW_FIELDS:
        raise StatefulEpochAuthorityStoreError(
            "stored stateful epoch candidate fields are invalid"
        )
    row = dict(value)
    mapping = _cutover(cutover)
    snapshot, snapshot_doc = validate_snapshot_document_v1(
        row.get("snapshot_doc"),
        cutover=mapping,
        require_boundary=True,
    )
    receipt, snapshot_hash = _snapshot_receipt(
        receipt_graph,
        snapshot_doc=snapshot_doc,
        declared_receipt_hash=row.get("chain_state_receipt_hash"),
    )
    if (
        snapshot.current_block != mapping.cutover_block
        or snapshot.block_hash != mapping.cutover_block_hash
        or snapshot.subnet_epoch_index != mapping.first_subnet_epoch_index
        or snapshot.settlement_epoch_id(mapping)
        != mapping.first_settlement_epoch_id
    ):
        raise StatefulEpochAuthorityStoreError(
            "stored candidate snapshot differs from cutover"
        )
    hotkey = str(row.get("validator_hotkey") or "").strip()
    signature = str(row.get("validator_hotkey_signature") or "").strip()
    payload_hash = _hash(
        row.get("candidate_payload_hash"),
        "stored candidate payload hash",
    )
    if not _SS58_RE.fullmatch(hotkey) or not _HOTKEY_SIGNATURE_RE.fullmatch(
        signature
    ):
        raise StatefulEpochAuthorityStoreError(
            "stored candidate authorization is invalid"
        )
    authorization_hash = sha256_json(
        {
            "validator_hotkey": hotkey,
            "candidate_payload_hash": payload_hash,
            "validator_hotkey_signature": signature,
        }
    )
    expected = {
        "snapshot_hash": snapshot_hash,
        "schema_version": snapshot.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "epoch_scheme": snapshot.epoch_scheme,
        "network_genesis_hash": mapping.network_genesis_hash,
        "netuid": mapping.netuid,
        "head_kind": snapshot.head_kind,
        "block_hash": snapshot.block_hash,
        "current_block": snapshot.current_block,
        "last_epoch_block": snapshot.last_epoch_block,
        "pending_epoch_at": snapshot.pending_epoch_at,
        "subnet_epoch_index": snapshot.subnet_epoch_index,
        "epoch_ref": snapshot.epoch_ref,
        "proposed_settlement_epoch_id": snapshot.settlement_epoch_id(mapping),
        "tempo": snapshot.tempo,
        "blocks_since_last_step": snapshot.blocks_since_last_step,
        "next_epoch_block": snapshot.next_epoch_block,
        "blocks_remaining": snapshot.blocks_remaining,
        "chain_state_receipt_hash": receipt["receipt_hash"],
        "validator_hotkey": hotkey,
        "candidate_payload_hash": payload_hash,
        "validator_hotkey_signature": signature,
        "candidate_authorization_hash": authorization_hash,
        "snapshot_doc": snapshot_doc,
        "observed_at": snapshot.observed_at,
    }
    for field, expected_value in expected.items():
        if not _row_value_equal(field, row.get(field), expected_value):
            raise StatefulEpochAuthorityStoreError(
                "stored candidate conflicts at %s" % field
            )
    if not str(row.get("created_at") or "").strip():
        raise StatefulEpochAuthorityStoreError(
            "stored candidate creation time is invalid"
        )
    return row


def build_boundary_row_v1(
    envelope: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
) -> Dict[str, Any]:
    mapping = _cutover(cutover)
    evidence = validate_epoch_evidence_envelope_v1(
        envelope,
        cutover=mapping,
    )
    if evidence["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise StatefulEpochAuthorityStoreError(
            "post-cutover boundary requires normal validator evidence"
        )
    snapshot = evidence["boundary"]
    snapshot_doc = evidence["boundary_doc"]
    if snapshot.subnet_epoch_index <= mapping.first_subnet_epoch_index:
        raise StatefulEpochAuthorityStoreError(
            "post-cutover boundary does not follow the cutover epoch"
        )
    receipt = evidence["boundary_receipt"]
    snapshot_hash = evidence["boundary_hash"]
    return {
        "boundary_hash": snapshot_hash,
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "mapping_hash": mapping.mapping_hash,
        "epoch_scheme": snapshot.epoch_scheme,
        "network_genesis_hash": snapshot.network_genesis_hash,
        "netuid": snapshot.netuid,
        "subnet_epoch_index": snapshot.subnet_epoch_index,
        "epoch_ref": snapshot.epoch_ref,
        "settlement_epoch_id": snapshot.settlement_epoch_id(mapping),
        "boundary_block": snapshot.current_block,
        "boundary_block_hash": snapshot.block_hash,
        "tempo": snapshot.tempo,
        "pending_epoch_at": snapshot.pending_epoch_at,
        "blocks_since_last_step": snapshot.blocks_since_last_step,
        "next_epoch_block": snapshot.next_epoch_block,
        "chain_state_receipt_hash": receipt["receipt_hash"],
        "boundary_doc": {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "mapping_hash": mapping.mapping_hash,
            "snapshot": snapshot_doc,
        },
        "observed_at": snapshot.observed_at,
    }


def build_snapshot_row_v1(
    evidence_value: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one current finalized snapshot row from normal weight evidence."""

    mapping = _cutover(cutover)
    evidence = validate_epoch_evidence_envelope_v1(
        evidence_value,
        cutover=mapping,
    )
    if evidence["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise StatefulEpochAuthorityStoreError(
            "current snapshot requires normal validator evidence"
        )
    snapshot = evidence["current"]
    snapshot_doc = evidence["current_doc"]
    receipt = evidence["current_receipt"]
    return {
        "snapshot_hash": evidence["current_hash"],
        "schema_version": snapshot.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "epoch_scheme": snapshot.epoch_scheme,
        "network_genesis_hash": snapshot.network_genesis_hash,
        "netuid": snapshot.netuid,
        "head_kind": snapshot.head_kind,
        "block_hash": snapshot.block_hash,
        "current_block": snapshot.current_block,
        "last_epoch_block": snapshot.last_epoch_block,
        "pending_epoch_at": snapshot.pending_epoch_at,
        "subnet_epoch_index": snapshot.subnet_epoch_index,
        "epoch_ref": snapshot.epoch_ref,
        "settlement_epoch_id": snapshot.settlement_epoch_id(mapping),
        "tempo": snapshot.tempo,
        "blocks_since_last_step": snapshot.blocks_since_last_step,
        "epoch_block": snapshot.epoch_block,
        "next_epoch_block": snapshot.next_epoch_block,
        "blocks_remaining": snapshot.blocks_remaining,
        "chain_state_receipt_hash": receipt["receipt_hash"],
        "snapshot_doc": snapshot_doc,
        "observed_at": snapshot.observed_at,
    }


def build_cutover_row_v1(
    *,
    authority_doc: Mapping[str, Any],
    first_snapshot_doc: Mapping[str, Any],
    receipt_graph: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the migration-100 cutover row from its coordinator receipt graph."""

    if not isinstance(authority_doc, Mapping) or set(authority_doc) != _AUTHORITY_FIELDS:
        raise StatefulEpochAuthorityStoreError("cutover authority fields are invalid")
    if authority_doc.get("schema_version") != CUTOVER_AUTHORITY_SCHEMA_VERSION:
        raise StatefulEpochAuthorityStoreError("cutover authority schema is invalid")
    mapping = _cutover(authority_doc.get("manifest"))
    snapshot, normalized_snapshot = validate_snapshot_document_v1(
        first_snapshot_doc,
        cutover=mapping,
        require_boundary=True,
    )
    snapshot_hash = sha256_json(normalized_snapshot)
    for field, expected in (
        ("mapping_hash", mapping.mapping_hash),
        ("first_epoch_ref", snapshot.epoch_ref),
        ("first_snapshot_hash", snapshot_hash),
    ):
        if authority_doc.get(field) != expected:
            raise StatefulEpochAuthorityStoreError(
                "cutover authority differs at %s" % field
            )
    validate_receipt_graph(receipt_graph)
    root_hash = str(receipt_graph.get("root_receipt_hash") or "")
    by_hash = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in receipt_graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    root = by_hash.get(root_hash)
    snapshot_receipt_hash = _hash(
        authority_doc.get("first_snapshot_receipt_hash"),
        "first snapshot receipt hash",
    )
    finalization_receipt_hash = _hash(
        authority_doc.get("last_legacy_finalization_receipt_hash"),
        "last legacy finalization receipt hash",
    )
    snapshot_receipt = by_hash.get(snapshot_receipt_hash)
    finalization_receipt = by_hash.get(finalization_receipt_hash)
    authority_hash = sha256_json(dict(authority_doc))
    if (
        not isinstance(root, Mapping)
        or root.get("role") != COORDINATOR_ROLE
        or root.get("purpose") != CUTOVER_PURPOSE
        or int(root.get("epoch_id", -1)) != mapping.first_settlement_epoch_id
        or root.get("output_root") != authority_hash
        or root.get("parent_receipt_hashes")
        != sorted([snapshot_receipt_hash, finalization_receipt_hash])
    ):
        raise StatefulEpochAuthorityStoreError(
            "cutover coordinator receipt is invalid"
        )
    if (
        not isinstance(snapshot_receipt, Mapping)
        or snapshot_receipt.get("role") != WEIGHT_ROLE
        or snapshot_receipt.get("purpose") != SNAPSHOT_PURPOSE
        or snapshot_receipt.get("output_root") != snapshot_hash
        or int(snapshot_receipt.get("epoch_id", -1))
        != mapping.first_settlement_epoch_id
    ):
        raise StatefulEpochAuthorityStoreError("cutover snapshot receipt is invalid")
    if (
        not isinstance(finalization_receipt, Mapping)
        or finalization_receipt.get("role") != WEIGHT_ROLE
        or finalization_receipt.get("purpose") != FINALIZATION_PURPOSE
        or int(finalization_receipt.get("epoch_id", -1))
        != mapping.last_legacy_epoch_id
    ):
        raise StatefulEpochAuthorityStoreError(
            "cutover finalization receipt is invalid"
        )
    return {
        "cutover_authority_hash": authority_hash,
        "schema_version": CUTOVER_AUTHORITY_SCHEMA_VERSION,
        "mapping_hash": mapping.mapping_hash,
        "manifest_schema_version": mapping.schema_version,
        "epoch_scheme": mapping.epoch_scheme,
        "previous_epoch_scheme": "legacy_global_360_v1",
        "network_genesis_hash": mapping.network_genesis_hash,
        "netuid": mapping.netuid,
        "cutover_block": mapping.cutover_block,
        "cutover_block_hash": mapping.cutover_block_hash,
        "first_subnet_epoch_index": mapping.first_subnet_epoch_index,
        "first_epoch_ref": snapshot.epoch_ref,
        "first_settlement_epoch_id": mapping.first_settlement_epoch_id,
        "last_legacy_epoch_id": mapping.last_legacy_epoch_id,
        "first_tempo": snapshot.tempo,
        "first_pending_epoch_at": snapshot.pending_epoch_at,
        "first_blocks_since_last_step": snapshot.blocks_since_last_step,
        "first_next_epoch_block": snapshot.next_epoch_block,
        "first_observed_at": snapshot.observed_at,
        "first_snapshot_hash": snapshot_hash,
        "first_snapshot_receipt_hash": snapshot_receipt_hash,
        "last_legacy_bundle_hash": _hash(
            authority_doc.get("last_legacy_bundle_hash"),
            "last legacy bundle hash",
        ),
        "last_legacy_weight_finalization_event_hash": _hash(
            authority_doc.get("last_legacy_weight_finalization_event_hash"),
            "last legacy finalization event hash",
        ),
        "last_legacy_finalization_receipt_hash": finalization_receipt_hash,
        "cutover_receipt_hash": root_hash,
        "manifest_doc": mapping.to_dict(),
        "first_snapshot_doc": normalized_snapshot,
        "authority_doc": dict(authority_doc),
    }


async def _assert_graph_durable(
    graph: Mapping[str, Any],
    *,
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]],
) -> str:
    expected_hash = sha256_json(dict(graph))
    result = await persist_graph(graph)
    if (
        not isinstance(result, Mapping)
        or result.get("root_receipt_hash") != graph.get("root_receipt_hash")
        or result.get("graph_hash") != expected_hash
    ):
        raise StatefulEpochAuthorityStoreError(
            "receipt graph persistence acknowledgment differs"
        )
    reloaded = await load_graph(str(graph["root_receipt_hash"]))
    if (
        not isinstance(reloaded, Mapping)
        or reloaded.get("root_receipt_hash") != graph.get("root_receipt_hash")
        or sha256_json(dict(reloaded)) != expected_hash
    ):
        raise StatefulEpochAuthorityStoreError("receipt graph readback differs")
    return expected_hash


async def _insert_exact(
    table: str,
    row: Mapping[str, Any],
    *,
    key_field: str,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]],
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    expected = dict(row)
    try:
        stored = await insert(table, expected)
    except Exception as exc:
        if not _is_duplicate_error(exc):
            raise
        stored = await select(
            table,
            filters=((key_field, expected[key_field]),),
        )
        if not isinstance(stored, Mapping):
            raise StatefulEpochAuthorityStoreError(
                "%s duplicate could not be reloaded" % table
            ) from exc
    if not isinstance(stored, Mapping):
        raise StatefulEpochAuthorityStoreError("%s insert returned no row" % table)
    for field, expected_value in expected.items():
        if not _row_value_equal(field, stored.get(field), expected_value):
            raise StatefulEpochAuthorityStoreError(
                "%s stored row conflicts at %s" % (table, field)
            )
    durable = await select(
        table,
        filters=((key_field, expected[key_field]),),
    )
    if not isinstance(durable, Mapping):
        raise StatefulEpochAuthorityStoreError(
            "%s durable readback returned no row" % table
        )
    for field, expected_value in expected.items():
        if not _row_value_equal(field, durable.get(field), expected_value):
            raise StatefulEpochAuthorityStoreError(
                "%s durable readback conflicts at %s" % (table, field)
            )
    return dict(durable)


async def persist_pre_cutover_candidate_v1(
    envelope: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
    validator_hotkey: str,
    candidate_payload_hash: str,
    validator_hotkey_signature: str,
    candidate_authorization_hash: str,
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]] = insert_row,
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = select_one,
) -> Dict[str, Any]:
    row = build_pre_cutover_candidate_row_v1(
        envelope,
        cutover=cutover,
        validator_hotkey=validator_hotkey,
        candidate_payload_hash=candidate_payload_hash,
        validator_hotkey_signature=validator_hotkey_signature,
        candidate_authorization_hash=candidate_authorization_hash,
    )
    await _assert_graph_durable(
        envelope["receipt_graph"],
        persist_graph=persist_graph,
        load_graph=load_graph,
    )
    return await _insert_exact(
        CANDIDATE_TABLE,
        row,
        key_field="snapshot_hash",
        insert=insert,
        select=select,
    )


async def persist_boundary_v1(
    envelope: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]] = insert_row,
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = select_one,
) -> Dict[str, Any]:
    row = build_boundary_row_v1(envelope, cutover=cutover)
    await _assert_graph_durable(
        envelope["receipt_graph"],
        persist_graph=persist_graph,
        load_graph=load_graph,
    )
    return await _insert_exact(
        BOUNDARY_TABLE,
        row,
        key_field="boundary_hash",
        insert=insert,
        select=select,
    )


async def persist_snapshot_v1(
    evidence: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]] = insert_row,
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = select_one,
) -> Dict[str, Any]:
    row = build_snapshot_row_v1(evidence, cutover=cutover)
    await _assert_graph_durable(
        evidence["receipt_graph"],
        persist_graph=persist_graph,
        load_graph=load_graph,
    )
    return await _insert_exact(
        SNAPSHOT_TABLE,
        row,
        key_field="snapshot_hash",
        insert=insert,
        select=select,
    )


async def persist_post_cutover_evidence_v1(
    evidence: Mapping[str, Any],
    *,
    cutover: Mapping[str, Any],
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]] = insert_row,
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = select_one,
) -> Dict[str, Any]:
    """Persist one normal current snapshot and its stable epoch boundary."""

    mapping = _cutover(cutover)
    snapshot_row = build_snapshot_row_v1(evidence, cutover=mapping.to_dict())
    boundary_index = int(snapshot_row["subnet_epoch_index"])
    boundary_row = None
    if boundary_index == mapping.first_subnet_epoch_index:
        normalized = validate_epoch_evidence_envelope_v1(
            evidence,
            cutover=mapping,
        )
        boundary = normalized["boundary"]
        if (
            boundary.current_block != mapping.cutover_block
            or boundary.block_hash != mapping.cutover_block_hash
            or boundary.epoch_ref != snapshot_row["epoch_ref"]
        ):
            raise StatefulEpochAuthorityStoreError(
                "first normal epoch evidence differs from cutover boundary"
            )
    else:
        boundary_row = build_boundary_row_v1(
            evidence,
            cutover=mapping.to_dict(),
        )

    graph_hash = await _assert_graph_durable(
        evidence["receipt_graph"],
        persist_graph=persist_graph,
        load_graph=load_graph,
    )
    durable_boundary = None
    if boundary_row is not None:
        durable_boundary = await _insert_exact(
            BOUNDARY_TABLE,
            boundary_row,
            key_field="boundary_hash",
            insert=insert,
            select=select,
        )
    durable_snapshot = await _insert_exact(
        SNAPSHOT_TABLE,
        snapshot_row,
        key_field="snapshot_hash",
        insert=insert,
        select=select,
    )
    durable_readback_hash = sha256_json(
        {
            "boundary": durable_boundary,
            "snapshot": durable_snapshot,
            "receipt_graph_hash": graph_hash,
        }
    )
    return {
        "boundary": durable_boundary,
        "snapshot": durable_snapshot,
        "receipt_graph_hash": graph_hash,
        "durable_readback_hash": durable_readback_hash,
    }


async def persist_cutover_v1(
    *,
    authority_doc: Mapping[str, Any],
    first_snapshot_doc: Mapping[str, Any],
    receipt_graph: Mapping[str, Any],
    persist_graph: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]] = persist_receipt_graph_v2,
    load_graph: Callable[[str], Awaitable[Mapping[str, Any]]] = load_receipt_graph_v2,
    insert: Callable[[str, Dict[str, Any]], Awaitable[Mapping[str, Any]]] = insert_row,
    select: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = select_one,
) -> Dict[str, Any]:
    row = build_cutover_row_v1(
        authority_doc=authority_doc,
        first_snapshot_doc=first_snapshot_doc,
        receipt_graph=receipt_graph,
    )
    await _assert_graph_durable(
        receipt_graph,
        persist_graph=persist_graph,
        load_graph=load_graph,
    )
    return await _insert_exact(
        CUTOVER_TABLE,
        row,
        key_field="cutover_authority_hash",
        insert=insert,
        select=select,
    )
