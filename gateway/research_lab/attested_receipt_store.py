"""Append-only persistence for enclave-signed scoring and weight sidecars."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from typing import Any, Iterable, Mapping

from gateway.research_lab.store import insert_row, select_many, select_one
from leadpoet_canonical.attested_receipts import (
    validate_signed_receipt,
    verify_receipt_lineage,
)


_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
logger = logging.getLogger(__name__)


class AttestedReceiptStoreError(ValueError):
    """Raised when a receipt sidecar is invalid or conflicts with stored data."""


def _attestation_hash(receipt: Mapping[str, Any]) -> str:
    try:
        document = base64.b64decode(
            str(receipt.get("attestation_document_b64") or ""),
            validate=True,
        )
    except Exception as exc:
        raise AttestedReceiptStoreError("receipt attestation is not valid base64") from exc
    if not document:
        raise AttestedReceiptStoreError("receipt attestation is empty")
    return "sha256:" + hashlib.sha256(document).hexdigest()


def receipt_storage_row(*, receipt: Mapping[str, Any], pcr0: str) -> dict[str, Any]:
    """Build the exact SQL sidecar row without performing I/O."""

    validate_signed_receipt(receipt)
    normalized_pcr0 = str(pcr0 or "").strip().lower()
    if not _PCR0_RE.fullmatch(normalized_pcr0) or normalized_pcr0 == "0" * 96:
        raise AttestedReceiptStoreError("receipt PCR0 is invalid")
    document_hash = _attestation_hash(receipt)
    return {
        "receipt_hash": receipt["receipt_hash"],
        "schema_version": receipt["schema_version"],
        "role": receipt["role"],
        "purpose": receipt["purpose"],
        "job_id": receipt["job_id"],
        "epoch_id": int(receipt["epoch_id"]),
        "commit_sha": receipt["commit_sha"],
        "pcr0": normalized_pcr0,
        "build_manifest_hash": receipt["build_manifest_hash"],
        "config_hash": receipt["config_hash"],
        "input_root": receipt["input_root"],
        "output_root": receipt["output_root"],
        "parent_receipt_hashes": list(receipt["parent_receipt_hashes"]),
        "evidence_roots": dict(receipt["evidence_roots"]),
        "receipt_status": receipt["status"],
        "receipt_doc": dict(receipt),
        "attestation_document_ref": "inline:%s" % document_hash,
        "attestation_document_hash": document_hash,
    }


def _is_duplicate_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "duplicate" in text or "unique" in text or "23505" in text


async def persist_attested_receipt(
    *,
    receipt: Mapping[str, Any],
    pcr0: str,
    artifact_links: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Insert one receipt and optional artifact links idempotently."""

    row = receipt_storage_row(receipt=receipt, pcr0=pcr0)
    try:
        stored = await insert_row("research_lab_attested_execution_receipts", row)
    except Exception as exc:
        if not _is_duplicate_error(exc):
            raise
        stored = await select_one(
            "research_lab_attested_execution_receipts",
            filters=(("receipt_hash", row["receipt_hash"]),),
        )
        if not isinstance(stored, Mapping):
            raise AttestedReceiptStoreError("duplicate receipt could not be reloaded") from exc
        for field, expected in row.items():
            if stored.get(field) != expected:
                raise AttestedReceiptStoreError("stored receipt conflicts at %s" % field) from exc

    for link in artifact_links:
        artifact_hash = str(link.get("artifact_hash") or "").strip().lower()
        if not _HASH_RE.fullmatch(artifact_hash):
            raise AttestedReceiptStoreError("artifact_hash is invalid")
        link_row = {
            "receipt_hash": row["receipt_hash"],
            "artifact_kind": str(link.get("artifact_kind") or "").strip(),
            "artifact_ref": str(link.get("artifact_ref") or "").strip(),
            "artifact_hash": artifact_hash,
        }
        if not link_row["artifact_kind"] or not link_row["artifact_ref"]:
            raise AttestedReceiptStoreError("artifact link kind and ref are required")
        try:
            await insert_row("research_lab_attested_artifact_links", link_row)
        except Exception as exc:
            if not _is_duplicate_error(exc):
                raise
    return dict(stored)


def _require_hash(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise AttestedReceiptStoreError("%s is invalid" % field_name)
    return normalized


async def load_attested_receipt(receipt_hash: str) -> dict[str, Any]:
    """Load and cryptographically validate one persisted receipt."""

    normalized_hash = _require_hash(receipt_hash, "receipt_hash")
    row = await select_one(
        "research_lab_attested_execution_receipts",
        filters=(("receipt_hash", normalized_hash),),
    )
    if not isinstance(row, Mapping):
        raise AttestedReceiptStoreError("attested receipt was not found")
    receipt = row.get("receipt_doc")
    if not isinstance(receipt, Mapping):
        raise AttestedReceiptStoreError("stored receipt_doc is missing")
    validate_signed_receipt(receipt)
    if receipt.get("receipt_hash") != normalized_hash:
        raise AttestedReceiptStoreError("stored receipt hash conflicts with receipt_doc")
    return dict(receipt)


async def load_receipt_for_artifact(
    *,
    artifact_kind: str,
    artifact_ref: str,
    artifact_hash: str | None = None,
) -> dict[str, Any] | None:
    """Resolve the newest valid receipt linked to one immutable artifact."""

    normalized_kind = str(artifact_kind or "").strip()
    normalized_ref = str(artifact_ref or "").strip()
    if not normalized_kind or not normalized_ref:
        raise AttestedReceiptStoreError("artifact kind and ref are required")
    filters: list[tuple[Any, ...]] = [
        ("artifact_kind", normalized_kind),
        ("artifact_ref", normalized_ref),
    ]
    if artifact_hash is not None:
        filters.append(("artifact_hash", _require_hash(artifact_hash, "artifact_hash")))
    links = await select_many(
        "research_lab_attested_artifact_links",
        filters=tuple(filters),
        order_by=(("created_at", True),),
        limit=20,
    )
    for link in links:
        try:
            return await load_attested_receipt(str(link.get("receipt_hash") or ""))
        except AttestedReceiptStoreError as exc:
            logger.warning(
                "research_lab_attested_artifact_link_invalid "
                "artifact_kind=%s artifact_ref=%s receipt_hash=%s error=%s",
                normalized_kind,
                normalized_ref,
                str(link.get("receipt_hash") or "")[:80],
                str(exc)[:240],
            )
            continue
    return None


async def load_attested_receipt_lineage(
    receipt: Mapping[str, Any],
    *,
    max_receipts: int = 256,
) -> list[dict[str, Any]]:
    """Load every transitive parent and validate the complete receipt graph."""

    validate_signed_receipt(receipt)
    if max_receipts < 1:
        raise AttestedReceiptStoreError("max_receipts must be positive")
    root_hash = str(receipt["receipt_hash"])
    pending = list(receipt.get("parent_receipt_hashes") or [])
    parents: dict[str, dict[str, Any]] = {}
    while pending:
        parent_hash = _require_hash(pending.pop(), "parent receipt hash")
        if parent_hash == root_hash:
            raise AttestedReceiptStoreError("receipt lineage references its root")
        if parent_hash in parents:
            continue
        if len(parents) >= max_receipts:
            raise AttestedReceiptStoreError("receipt lineage exceeds maximum size")
        parent = await load_attested_receipt(parent_hash)
        parents[parent_hash] = parent
        pending.extend(parent.get("parent_receipt_hashes") or [])
    ordered = verify_receipt_lineage(receipt, parents)
    return [dict(parents[item]) for item in ordered if item != root_hash]


async def persist_attested_weight_bundle(
    *,
    bundle: Mapping[str, Any],
    validator_pcr0: str,
    verification_mode: str,
) -> dict[str, Any]:
    """Persist one already-verified v2 bundle without touching v1 weights."""

    from leadpoet_canonical.attested_receipts import sha256_json
    from leadpoet_canonical.weight_bundle_v2 import validate_weight_bundle_v2

    normalized_mode = str(verification_mode or "").strip().lower()
    if normalized_mode not in {"shadow", "required"}:
        raise AttestedReceiptStoreError("weight bundle verification mode is invalid")
    normalized_pcr0 = str(validator_pcr0 or "").strip().lower()
    verified = validate_weight_bundle_v2(bundle, require_allocation_ancestry=False)
    receipt = bundle.get("weight_receipt")
    if not isinstance(receipt, Mapping):
        raise AttestedReceiptStoreError("weight bundle receipt is missing")
    weights_hash = str(verified.get("weights_hash") or "").lower()
    if not _RAW_HASH_RE.fullmatch(weights_hash):
        raise AttestedReceiptStoreError("weight bundle weights_hash is invalid")
    artifact_ref = "weight_bundle_v2:%s:%s:%s" % (
        int(verified["netuid"]),
        int(verified["epoch_id"]),
        str(bundle.get("validator_hotkey") or ""),
    )
    await persist_attested_receipt(
        receipt=receipt,
        pcr0=normalized_pcr0,
        artifact_links=[
            {
                "artifact_kind": "weight_bundle_v2",
                "artifact_ref": artifact_ref,
                "artifact_hash": "sha256:" + weights_hash,
            }
        ],
    )
    row = {
        "weight_receipt_hash": str(receipt["receipt_hash"]),
        "netuid": int(verified["netuid"]),
        "epoch_id": int(verified["epoch_id"]),
        "block": int(verified["block"]),
        "validator_hotkey": str(bundle.get("validator_hotkey") or ""),
        "weights_hash": weights_hash,
        "validator_commit_sha": str(receipt.get("commit_sha") or ""),
        "validator_pcr0": normalized_pcr0,
        "verification_mode": normalized_mode,
        "bundle_hash": sha256_json(dict(bundle)),
        "bundle_doc": dict(bundle),
    }
    try:
        return dict(await insert_row("research_lab_attested_weight_bundles", row))
    except Exception as exc:
        if not _is_duplicate_error(exc):
            raise
        stored = await select_one(
            "research_lab_attested_weight_bundles",
            filters=(("weight_receipt_hash", row["weight_receipt_hash"]),),
        )
        if not isinstance(stored, Mapping):
            raise AttestedReceiptStoreError("duplicate v2 weight bundle could not be reloaded") from exc
        for field, expected in row.items():
            if stored.get(field) != expected:
                raise AttestedReceiptStoreError(
                    "stored v2 weight bundle conflicts at %s" % field
                ) from exc
        return dict(stored)


async def load_attested_weight_bundle(*, netuid: int, epoch_id: int) -> dict[str, Any] | None:
    """Load and canonically validate a persisted shadow v2 weight bundle."""

    from leadpoet_canonical.attested_receipts import sha256_json
    from leadpoet_canonical.weight_bundle_v2 import validate_weight_bundle_v2

    rows = await select_many(
        "research_lab_attested_weight_bundles",
        filters=(("netuid", int(netuid)), ("epoch_id", int(epoch_id))),
        order_by=(("created_at", True),),
        limit=2,
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise AttestedReceiptStoreError("multiple v2 weight bundles exist for one epoch")
    row = rows[0]
    bundle = row.get("bundle_doc") if isinstance(row, Mapping) else None
    if not isinstance(bundle, Mapping):
        raise AttestedReceiptStoreError("stored v2 weight bundle document is missing")
    validate_weight_bundle_v2(bundle, require_allocation_ancestry=False)
    if row.get("bundle_hash") != sha256_json(dict(bundle)):
        raise AttestedReceiptStoreError("stored v2 weight bundle hash diverged")
    if row.get("weight_receipt_hash") != bundle.get("weight_receipt", {}).get("receipt_hash"):
        raise AttestedReceiptStoreError("stored v2 weight receipt hash diverged")
    return dict(bundle)
