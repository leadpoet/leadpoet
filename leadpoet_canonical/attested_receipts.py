"""Canonical scoring and weight receipts signed by existing enclave keys.

The module is intentionally compatible with Python 3.7 and depends only on the
standard library for receipt construction. Cryptography is imported lazily by
the verifier, matching the existing canonical event implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RECEIPT_SCHEMA_VERSION = "leadpoet.attested_receipt.v1"
SCORING_ROLE = "gateway_scoring"
WEIGHT_ROLE = "validator_weights"

SCORING_PURPOSES = frozenset(
    {
        "research_lab.candidate_score.v1",
        "research_lab.baseline_score.v1",
        "research_lab.benchmark.v1",
        "research_lab.rebenchmark.v1",
        "research_lab.promotion_metric.v1",
        "research_lab.promotion_decision.v1",
        "research_lab.allocation.v1",
    }
)
WEIGHT_PURPOSE = "validator.weights.computed.v2"
ROLE_PURPOSES = {
    SCORING_ROLE: SCORING_PURPOSES,
    WEIGHT_ROLE: frozenset({WEIGHT_PURPOSE}),
}
RECEIPT_STATUSES = frozenset({"succeeded", "failed"})

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PUBKEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ReceiptError(ValueError):
    """Raised when an attested receipt fails canonical validation."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReceiptError("receipt value is not canonical JSON: %s" % exc) from exc


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _hash(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s must be sha256:<64 lowercase hex>" % field_name)
    return normalized


def _identifier(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    _require(bool(_IDENTIFIER_RE.fullmatch(normalized)), "%s is not a valid identifier" % field_name)
    return normalized


def artifact_commitment(name: str, content: bytes) -> Dict[str, Any]:
    return {
        "name": _identifier(name, "artifact name"),
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def artifact_merkle_root(artifacts: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for artifact in artifacts:
        name = _identifier(artifact.get("name"), "artifact name")
        digest = _hash(artifact.get("sha256"), "artifact sha256")
        size = artifact.get("size_bytes")
        _require(isinstance(size, int) and size >= 0, "artifact size_bytes must be non-negative")
        normalized.append({"name": name, "sha256": digest, "size_bytes": size})
    normalized.sort(key=lambda item: (item["name"], item["sha256"], item["size_bytes"]))
    if not normalized:
        return sha256_bytes(b"\x02leadpoet-empty-artifact-tree-v1")
    nodes = [
        hashlib.sha256(b"\x00" + canonical_json(item).encode("utf-8")).digest()
        for item in normalized
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return "sha256:" + nodes[0].hex()


def build_receipt_body(
    *,
    role: str,
    purpose: str,
    job_id: str,
    epoch_id: int,
    commit_sha: str,
    build_manifest_hash: str,
    config_hash: str,
    input_root: str,
    output_root: str,
    evidence_roots: Mapping[str, str],
    parent_receipt_hashes: Iterable[str],
    status: str,
    issued_at: str,
    failure_code: Optional[str] = None,
) -> Dict[str, Any]:
    role = str(role or "")
    purpose = str(purpose or "")
    _require(role in ROLE_PURPOSES, "unsupported receipt role")
    _require(purpose in ROLE_PURPOSES[role], "purpose is not valid for receipt role")
    _require(isinstance(epoch_id, int) and epoch_id >= 0, "epoch_id must be non-negative")
    commit_sha = str(commit_sha or "").strip().lower()
    _require(bool(_COMMIT_RE.fullmatch(commit_sha)), "commit_sha must be a full Git object id")
    _require(status in RECEIPT_STATUSES, "unsupported receipt status")
    if status == "failed":
        failure_code = _identifier(failure_code, "failure_code")
    else:
        _require(failure_code in (None, ""), "successful receipt cannot include failure_code")
        failure_code = None
    issued_at = str(issued_at or "")
    _require(bool(_TIMESTAMP_RE.fullmatch(issued_at)), "issued_at must be RFC3339 UTC without fractional seconds")
    parents = sorted({_hash(item, "parent_receipt_hash") for item in parent_receipt_hashes})
    evidence = {
        _identifier(name, "evidence root name"): _hash(value, "evidence root")
        for name, value in sorted(evidence_roots.items())
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "role": role,
        "purpose": purpose,
        "job_id": _identifier(job_id, "job_id"),
        "epoch_id": epoch_id,
        "commit_sha": commit_sha,
        "build_manifest_hash": _hash(build_manifest_hash, "build_manifest_hash"),
        "config_hash": _hash(config_hash, "config_hash"),
        "input_root": _hash(input_root, "input_root"),
        "output_root": _hash(output_root, "output_root"),
        "evidence_roots": evidence,
        "parent_receipt_hashes": parents,
        "status": status,
        "failure_code": failure_code,
        "issued_at": issued_at,
    }


def receipt_hash(body: Mapping[str, Any]) -> str:
    validate_receipt_body(body)
    return sha256_json(dict(body))


def create_signed_receipt(
    *,
    body: Mapping[str, Any],
    enclave_pubkey: str,
    attestation_document_b64: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    body = dict(body)
    digest_hash = receipt_hash(body)
    digest = bytes.fromhex(digest_hash.split(":", 1)[1])
    signature_value = sign_digest(digest)
    signature = signature_value.hex() if isinstance(signature_value, bytes) else str(signature_value)
    enclave_pubkey = str(enclave_pubkey or "").lower()
    _require(bool(_PUBKEY_RE.fullmatch(enclave_pubkey)), "enclave_pubkey must be 32-byte lowercase hex")
    _require(bool(_SIGNATURE_RE.fullmatch(signature)), "enclave_signature must be 64-byte lowercase hex")
    _require(bool(attestation_document_b64), "attestation_document_b64 is required")
    receipt = {
        **body,
        "receipt_hash": digest_hash,
        "enclave_pubkey": enclave_pubkey,
        "enclave_signature": signature,
        "attestation_document_b64": str(attestation_document_b64),
    }
    validate_signed_receipt(receipt, verify_signature=False)
    return receipt


def validate_receipt_body(body: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "role",
        "purpose",
        "job_id",
        "epoch_id",
        "commit_sha",
        "build_manifest_hash",
        "config_hash",
        "input_root",
        "output_root",
        "evidence_roots",
        "parent_receipt_hashes",
        "status",
        "failure_code",
        "issued_at",
    }
    _require(set(body) == required, "receipt body fields do not match the canonical schema")
    rebuilt = build_receipt_body(
        role=body["role"],
        purpose=body["purpose"],
        job_id=body["job_id"],
        epoch_id=body["epoch_id"],
        commit_sha=body["commit_sha"],
        build_manifest_hash=body["build_manifest_hash"],
        config_hash=body["config_hash"],
        input_root=body["input_root"],
        output_root=body["output_root"],
        evidence_roots=body["evidence_roots"],
        parent_receipt_hashes=body["parent_receipt_hashes"],
        status=body["status"],
        failure_code=body["failure_code"],
        issued_at=body["issued_at"],
    )
    _require(dict(body) == rebuilt, "receipt body is not canonically normalized")


def validate_signed_receipt(receipt: Mapping[str, Any], *, verify_signature: bool = True) -> None:
    signed_fields = {"receipt_hash", "enclave_pubkey", "enclave_signature", "attestation_document_b64"}
    body = {key: value for key, value in receipt.items() if key not in signed_fields}
    _require(set(receipt) == set(body) | signed_fields, "signed receipt has unknown or missing fields")
    expected_hash = receipt_hash(body)
    _require(receipt.get("receipt_hash") == expected_hash, "receipt_hash does not match canonical body")
    pubkey = str(receipt.get("enclave_pubkey") or "").lower()
    signature = str(receipt.get("enclave_signature") or "").lower()
    _require(bool(_PUBKEY_RE.fullmatch(pubkey)), "invalid enclave_pubkey")
    _require(bool(_SIGNATURE_RE.fullmatch(signature)), "invalid enclave_signature")
    _require(bool(receipt.get("attestation_document_b64")), "attestation_document_b64 is required")
    if not verify_signature:
        return
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        verifier = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey))
        verifier.verify(bytes.fromhex(signature), bytes.fromhex(expected_hash.split(":", 1)[1]))
    except Exception as exc:
        raise ReceiptError("invalid enclave receipt signature") from exc


def verify_receipt_lineage(
    receipt: Mapping[str, Any],
    parent_receipts: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    validate_signed_receipt(receipt)
    normalized: Dict[str, Mapping[str, Any]] = {}
    for key, parent in parent_receipts.items():
        validate_signed_receipt(parent)
        _require(key == parent.get("receipt_hash"), "parent map key does not match receipt_hash")
        normalized[key] = parent
    visiting = set()
    visited = set()
    ordered: List[str] = []

    def visit(current: Mapping[str, Any]) -> None:
        current_hash = str(current["receipt_hash"])
        if current_hash in visited:
            return
        _require(current_hash not in visiting, "receipt lineage contains a cycle")
        visiting.add(current_hash)
        for parent_hash in current["parent_receipt_hashes"]:
            _require(parent_hash in normalized, "missing parent receipt %s" % parent_hash)
            parent = normalized[parent_hash]
            _require(int(parent["epoch_id"]) <= int(current["epoch_id"]), "parent epoch is newer than child")
            visit(parent)
        visiting.remove(current_hash)
        visited.add(current_hash)
        ordered.append(current_hash)

    visit(receipt)
    return tuple(ordered)
