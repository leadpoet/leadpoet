"""Detached, recursive certificates for bounded V2 receipt ancestry.

Raw receipts and their content-addressed sidecars remain immutable audit
evidence. Operational ancestry uses a bounded local delta plus predecessor
authorities; a full ``leadpoet.attested_receipt_graph.v2`` is accepted only as
the finite bootstrap for an uncheckpointed legacy root. A measured enclave
validates that bootstrap or the exact local delta, then directly signs a
domain-separated certificate claim. Once a root is checkpointed, durable
storage rejects any attempt to expand that root as a full graph again.

The equivalence is *attestation-recursive*, not a SNARK.  A compact verifier
checks the checkpoint enclave's exact Nitro/release boot, signature, bounded
receipt delta, parent authority map, and every committed root. It relies on
that measured code having validated omitted attempt/host-operation documents;
those immutable documents remain available by their committed hashes without
being copied into every later operational graph.

This module is I/O-free and Python 3.7 compatible.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical.attested_v2 import (
    RECEIPT_GRAPH_SCHEMA_VERSION,
    canonical_json,
    host_operation_root,
    merkle_root,
    sha256_bytes,
    transport_root,
    validate_boot_identity,
    validate_host_operation_record,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    validate_transport_attempt,
)


ANCESTRY_PROJECTION_SCHEMA_VERSION = "leadpoet.attested_ancestry_projection.v2"
ANCESTRY_DELTA_SCHEMA_VERSION = "leadpoet.attested_ancestry_delta.v2"
ANCESTRY_CERTIFICATE_CLAIM_SCHEMA_VERSION = (
    "leadpoet.attested_ancestry_certificate_claim.v2"
)
ANCESTRY_CERTIFICATE_SCHEMA_VERSION = (
    "leadpoet.attested_ancestry_certificate.v2"
)
ANCESTRY_COMPACT_PROOF_SCHEMA_VERSION = (
    "leadpoet.attested_ancestry_compact_proof.v2"
)
ANCESTRY_POLICY_SCHEMA_VERSION = "leadpoet.attested_ancestry_policy.v2"
ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION = (
    "leadpoet.attested_ancestry_parent_authority.v2"
)
ANCESTRY_FULL_GRAPH_PARENT_SCHEMA_VERSION = (
    "leadpoet.attested_ancestry_full_graph_parent.v2"
)

MAX_FULL_GRAPH_RECEIPTS = 10_000
MAX_FULL_GRAPH_BOOTS = 10_000
MAX_FULL_GRAPH_ATTEMPTS = 250_000
MAX_FULL_GRAPH_HOST_OPERATIONS = 250_000
MAX_FULL_GRAPH_EDGES = 100_000
MAX_DELTA_RECEIPTS = 256
MAX_DELTA_BOOTS = 64
MAX_DELTA_ATTEMPTS = 100_000
MAX_DELTA_HOST_OPERATIONS = 100_000
MAX_DELTA_EDGES = 4_096
MAX_PARENT_AUTHORITIES = 128
MAX_ALLOWED_FAILED_RECEIPTS = 128
MAX_REQUIRED_PURPOSES = 128
MAX_NETUID = 2 ** 31 - 1

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHAIN_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_PROJECTION_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "projection_kind",
        "source_schema_version",
        "root_receipt_hash",
        "root_epoch_id",
        "root_role",
        "root_purpose",
        "ancestry_purposes",
        "root_boot_identity_hash",
        "evidence_commitment",
        "receipt_hashes",
        "receipt_hashes_root",
        "receipt_count",
        "parent_edges_root",
        "parent_edge_count",
        "external_parent_receipt_hashes",
        "boot_identity_hashes",
        "boot_identities_root",
        "boot_identity_count",
        "transport_roots_root",
        "transport_root_count",
        "host_operation_roots_root",
        "host_operation_root_count",
        "release_pcr_commitments_root",
        "release_pcr_commitment_count",
        "transport_attempt_count",
        "host_operation_count",
    }
)
_PROJECTION_FIELDS = frozenset(set(_PROJECTION_BODY_FIELDS) | {"projection_hash"})
_DELTA_FIELDS = frozenset(
    {
        "schema_version",
        "root_receipt_hash",
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    }
)
_POLICY_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "allowed_failed_receipt_hashes",
        "required_purposes",
    }
)
_POLICY_FIELDS = frozenset(set(_POLICY_BODY_FIELDS) | {"policy_hash"})
_PARENT_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "authority_kind",
        "parent_receipt_hash",
        "parent_epoch_id",
        "parent_role",
        "parent_purpose",
        "authority_hash",
        "authority_policy_hash",
        "authority_sequence",
        "authority_purposes",
    }
)
_FULL_GRAPH_PARENT_FIELDS = frozenset({"schema_version", "graph", "policy"})
_CLAIM_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "lineage_id",
        "certificate_sequence",
        "output_root_receipt_hash",
        "local_delta_projection",
        "parent_authorities",
        "parent_authorities_root",
        "policy",
        "issuer_boot_identity_hash",
        "issued_at",
    }
)
_CLAIM_FIELDS = frozenset(set(_CLAIM_BODY_FIELDS) | {"claim_hash"})
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "claim",
        "issuer_boot_identity",
        "enclave_pubkey",
        "enclave_signature",
        "certificate_hash",
    }
)
_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "certificate",
        "disclosed_receipts",
        "disclosed_boot_identities",
        "proof_hash",
    }
)


class AncestryCheckpointV2Error(ValueError):
    """A detached ancestry projection, certificate, or proof is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AncestryCheckpointV2Error(message)


def _exact(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "%s must be an object" % label)
    _require(set(value) == set(fields), "%s fields do not match schema" % label)
    return value


def _hash(value: Any, field: str) -> str:
    raw = str(value or "")
    normalized = raw.strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    _require(raw == normalized, "%s is not canonical" % field)
    return normalized


def _identifier(value: Any, field: str) -> str:
    raw = str(value or "")
    normalized = raw.strip()
    _require(bool(_IDENTIFIER_RE.fullmatch(normalized)), "%s is invalid" % field)
    _require(raw == normalized, "%s is not canonical" % field)
    return normalized


def _chain_hash(value: Any, field: str) -> str:
    raw = str(value or "")
    _require(bool(_CHAIN_HASH_RE.fullmatch(raw)), "%s is invalid" % field)
    return raw


def _timestamp(value: Any, field: str) -> str:
    normalized = str(value or "")
    _require(bool(_TIMESTAMP_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _count(value: Any, field: str, maximum: int, minimum: int = 0) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum,
        "%s is outside the canonical bound" % field,
    )
    return int(value)


def _domain_hash(domain: str, value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return sha256_bytes(
        b"\x00" + _identifier(domain, "hash domain").encode("ascii") + b"\x00" + encoded
    )


def _signature(value: Any) -> str:
    raw = value.hex() if isinstance(value, bytes) else str(value or "")
    normalized = raw.lower()
    _require(bool(_SIGNATURE_RE.fullmatch(normalized)), "certificate signature is invalid")
    _require(raw == normalized, "certificate signature is not canonical")
    return normalized


def derive_ancestry_lineage_id_v2(
    *, cutover_mapping_hash: str, network_genesis_hash: str, netuid: int
) -> str:
    """Derive the stable lineage namespace from immutable cutover identity.

    Release hashes deliberately do not participate: a forward release or a
    verified rollback remains in the same anti-fork lineage.
    """

    normalized_netuid = _count(netuid, "lineage netuid", MAX_NETUID)
    return _domain_hash(
        "leadpoet-ancestry-lineage-id-v2",
        {
            "cutover_mapping_hash": _hash(
                cutover_mapping_hash, "cutover mapping hash"
            ),
            "network_genesis_hash": _chain_hash(
                network_genesis_hash, "network genesis hash"
            ),
            "netuid": normalized_netuid,
        },
    )


def validate_ancestry_lineage_id_v2(
    value: str,
    *,
    cutover_mapping_hash: str,
    network_genesis_hash: str,
    netuid: int
) -> str:
    normalized = _hash(value, "ancestry lineage id")
    _require(
        normalized
        == derive_ancestry_lineage_id_v2(
            cutover_mapping_hash=cutover_mapping_hash,
            network_genesis_hash=network_genesis_hash,
            netuid=netuid,
        ),
        "ancestry lineage id differs from immutable cutover identity",
    )
    return normalized


def _verify_ed25519(pubkey: str, signature: str, digest_hash: str) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(
            bytes.fromhex(signature),
            bytes.fromhex(digest_hash.split(":", 1)[1]),
        )
    except Exception as exc:
        raise AncestryCheckpointV2Error("certificate signature is invalid") from exc


def _sorted_hashes(values: Any, field: str, maximum: int) -> List[str]:
    _require(isinstance(values, (list, tuple, set, frozenset)), "%s must be an array" % field)
    normalized = [_hash(item, field) for item in values]
    _require(len(normalized) <= maximum, "%s exceeds canonical bound" % field)
    _require(len(normalized) == len(set(normalized)), "%s is duplicated" % field)
    return sorted(normalized)


def _sorted_identifiers(values: Any, field: str, maximum: int) -> List[str]:
    _require(isinstance(values, (list, tuple, set, frozenset)), "%s must be an array" % field)
    normalized = [_identifier(item, field) for item in values]
    _require(len(normalized) <= maximum, "%s exceeds canonical bound" % field)
    _require(len(normalized) == len(set(normalized)), "%s is duplicated" % field)
    return sorted(normalized)


def _host_key(record: Mapping[str, Any]) -> str:
    request = record.get("request") if isinstance(record, Mapping) else None
    _require(isinstance(request, Mapping), "host operation request is missing")
    return _hash(request.get("request_hash"), "host operation request hash")


def _normalized_evidence(value: Mapping[str, Any], schema_version: str) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "root_receipt_hash": _hash(value.get("root_receipt_hash"), "root receipt hash"),
        "boot_identities": sorted(
            (dict(item) for item in value["boot_identities"]),
            key=lambda item: str(item["boot_identity_hash"]),
        ),
        "receipts": sorted(
            (dict(item) for item in value["receipts"]),
            key=lambda item: str(item["receipt_hash"]),
        ),
        "transport_attempts": sorted(
            (dict(item) for item in value["transport_attempts"]),
            key=lambda item: str(item["attempt_hash"]),
        ),
        "host_operations": sorted(
            (dict(item) for item in value["host_operations"]),
            key=_host_key,
        ),
    }


def _release_leaf(identity: Mapping[str, Any]) -> str:
    return _domain_hash(
        "leadpoet-ancestry-release-pcr-leaf-v2",
        {
            "boot_identity_hash": identity["boot_identity_hash"],
            "role": identity["role"],
            "physical_role": identity["physical_role"],
            "commit_sha": identity["commit_sha"],
            "pcr0": identity["pcr0"],
            "build_manifest_hash": identity["build_manifest_hash"],
            "dependency_lock_hash": identity["dependency_lock_hash"],
            "config_hash": identity["config_hash"],
        },
    )


def _projection_from_validated_evidence(
    normalized: Mapping[str, Any],
    *,
    projection_kind: str,
    external_parent_receipt_hashes: Sequence[str],
    ancestry_purposes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    receipts = list(normalized["receipts"])
    boots = list(normalized["boot_identities"])
    root_hash = str(normalized["root_receipt_hash"])
    receipt_by_hash = {str(item["receipt_hash"]): item for item in receipts}
    root = receipt_by_hash.get(root_hash)
    _require(root is not None, "projection root receipt is missing")
    receipt_hashes = [str(item["receipt_hash"]) for item in receipts]
    boot_hashes = [str(item["boot_identity_hash"]) for item in boots]
    edge_leaves = []  # type: List[str]
    transport_leaves = []  # type: List[str]
    host_leaves = []  # type: List[str]
    for receipt in receipts:
        receipt_hash = str(receipt["receipt_hash"])
        for parent_hash in receipt["parent_receipt_hashes"]:
            edge_leaves.append(
                _domain_hash(
                    "leadpoet-ancestry-parent-edge-leaf-v2",
                    {
                        "child_receipt_hash": receipt_hash,
                        "parent_receipt_hash": str(parent_hash),
                    },
                )
            )
        transport_leaves.append(
            _domain_hash(
                "leadpoet-ancestry-transport-root-leaf-v2",
                {
                    "receipt_hash": receipt_hash,
                    "boot_identity_hash": receipt["boot_identity_hash"],
                    "transport_root": receipt["transport_root"],
                },
            )
        )
        host_leaves.append(
            _domain_hash(
                "leadpoet-ancestry-host-operation-root-leaf-v2",
                {
                    "receipt_hash": receipt_hash,
                    "boot_identity_hash": receipt["boot_identity_hash"],
                    "host_operation_root": receipt["host_operation_root"],
                },
            )
        )
    _require(len(edge_leaves) == len(set(edge_leaves)), "projection parent edge is duplicated")
    release_leaves = [_release_leaf(identity) for identity in boots]
    purpose_set = {
        str(item["purpose"]) for item in receipts
    }
    if ancestry_purposes is not None:
        purpose_set.update(str(item) for item in ancestry_purposes)
    canonical_purposes = _sorted_identifiers(
        list(purpose_set), "ancestry purpose", MAX_REQUIRED_PURPOSES
    )
    body = {
        "schema_version": ANCESTRY_PROJECTION_SCHEMA_VERSION,
        "projection_kind": projection_kind,
        "source_schema_version": str(normalized["schema_version"]),
        "root_receipt_hash": root_hash,
        "root_epoch_id": int(root["epoch_id"]),
        "root_role": str(root["role"]),
        "root_purpose": str(root["purpose"]),
        "ancestry_purposes": canonical_purposes,
        "root_boot_identity_hash": str(root["boot_identity_hash"]),
        "evidence_commitment": _domain_hash(
            "leadpoet-ancestry-%s-evidence-v2" % projection_kind.replace("_", "-"),
            dict(normalized),
        ),
        "receipt_hashes": receipt_hashes,
        "receipt_hashes_root": merkle_root(receipt_hashes, domain="leadpoet-ancestry-receipts-v2"),
        "receipt_count": len(receipt_hashes),
        "parent_edges_root": merkle_root(edge_leaves, domain="leadpoet-ancestry-parent-edges-v2"),
        "parent_edge_count": len(edge_leaves),
        "external_parent_receipt_hashes": list(external_parent_receipt_hashes),
        "boot_identity_hashes": boot_hashes,
        "boot_identities_root": merkle_root(boot_hashes, domain="leadpoet-ancestry-boots-v2"),
        "boot_identity_count": len(boot_hashes),
        "transport_roots_root": merkle_root(
            transport_leaves, domain="leadpoet-ancestry-transport-roots-v2"
        ),
        "transport_root_count": len(transport_leaves),
        "host_operation_roots_root": merkle_root(
            host_leaves, domain="leadpoet-ancestry-host-operation-roots-v2"
        ),
        "host_operation_root_count": len(host_leaves),
        "release_pcr_commitments_root": merkle_root(
            release_leaves, domain="leadpoet-ancestry-release-pcr-commitments-v2"
        ),
        "release_pcr_commitment_count": len(release_leaves),
        "transport_attempt_count": len(normalized["transport_attempts"]),
        "host_operation_count": len(normalized["host_operations"]),
    }
    return {**body, "projection_hash": _domain_hash("leadpoet-ancestry-projection-v2", body)}


def validate_ancestry_projection_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    projection = _exact(value, _PROJECTION_FIELDS, "ancestry projection")
    _require(
        projection.get("schema_version") == ANCESTRY_PROJECTION_SCHEMA_VERSION,
        "ancestry projection schema is invalid",
    )
    kind = str(projection.get("projection_kind") or "")
    _require(kind in {"full_graph", "local_delta"}, "ancestry projection kind is invalid")
    _require(
        projection.get("source_schema_version")
        == (RECEIPT_GRAPH_SCHEMA_VERSION if kind == "full_graph" else ANCESTRY_DELTA_SCHEMA_VERSION),
        "ancestry projection source schema differs",
    )
    for field in (
        "root_receipt_hash",
        "root_boot_identity_hash",
        "evidence_commitment",
        "receipt_hashes_root",
        "parent_edges_root",
        "boot_identities_root",
        "transport_roots_root",
        "host_operation_roots_root",
        "release_pcr_commitments_root",
    ):
        _hash(projection.get(field), field)
    _identifier(projection.get("root_role"), "projection root role")
    _identifier(projection.get("root_purpose"), "projection root purpose")
    purposes = _sorted_identifiers(
        projection.get("ancestry_purposes"),
        "projection ancestry purpose",
        MAX_REQUIRED_PURPOSES,
    )
    _require(
        list(projection["ancestry_purposes"]) == purposes,
        "projection ancestry purposes are not canonical",
    )
    _require(
        projection["root_purpose"] in purposes,
        "projection root purpose is absent from ancestry",
    )
    _count(projection.get("root_epoch_id"), "projection root epoch", 2 ** 63 - 1)
    receipt_max = MAX_FULL_GRAPH_RECEIPTS if kind == "full_graph" else MAX_DELTA_RECEIPTS
    boot_max = MAX_FULL_GRAPH_BOOTS if kind == "full_graph" else MAX_DELTA_BOOTS
    edge_max = MAX_FULL_GRAPH_EDGES if kind == "full_graph" else MAX_DELTA_EDGES
    attempt_max = MAX_FULL_GRAPH_ATTEMPTS if kind == "full_graph" else MAX_DELTA_ATTEMPTS
    host_max = MAX_FULL_GRAPH_HOST_OPERATIONS if kind == "full_graph" else MAX_DELTA_HOST_OPERATIONS
    _require(isinstance(projection.get("receipt_hashes"), list), "projection receipt hashes must be an array")
    _require(isinstance(projection.get("boot_identity_hashes"), list), "projection boot hashes must be an array")
    _require(isinstance(projection.get("external_parent_receipt_hashes"), list), "projection external parents must be an array")
    receipt_hashes = _sorted_hashes(projection.get("receipt_hashes"), "projection receipt hash", receipt_max)
    boot_hashes = _sorted_hashes(projection.get("boot_identity_hashes"), "projection boot hash", boot_max)
    external = _sorted_hashes(
        projection.get("external_parent_receipt_hashes"),
        "external parent receipt hash",
        MAX_PARENT_AUTHORITIES,
    )
    _require(list(projection["receipt_hashes"]) == receipt_hashes, "projection receipt hashes are not canonical")
    _require(list(projection["boot_identity_hashes"]) == boot_hashes, "projection boot hashes are not canonical")
    _require(list(projection["external_parent_receipt_hashes"]) == external, "projection external parents are not canonical")
    receipt_count = _count(projection.get("receipt_count"), "projection receipt count", receipt_max, 1)
    boot_count = _count(projection.get("boot_identity_count"), "projection boot count", boot_max, 1)
    _require(receipt_count == len(receipt_hashes), "projection receipt count differs")
    _require(boot_count == len(boot_hashes), "projection boot count differs")
    _require(projection["root_receipt_hash"] in receipt_hashes, "projection root is absent")
    _require(projection["root_boot_identity_hash"] in boot_hashes, "projection root boot is absent")
    _require(
        projection["receipt_hashes_root"] == merkle_root(receipt_hashes, domain="leadpoet-ancestry-receipts-v2"),
        "projection receipt root differs",
    )
    _require(
        projection["boot_identities_root"] == merkle_root(boot_hashes, domain="leadpoet-ancestry-boots-v2"),
        "projection boot root differs",
    )
    edge_count = _count(projection.get("parent_edge_count"), "projection parent edge count", edge_max)
    _count(projection.get("transport_attempt_count"), "projection attempt count", attempt_max)
    _count(projection.get("host_operation_count"), "projection host operation count", host_max)
    _require(
        _count(projection.get("transport_root_count"), "projection transport root count", receipt_max) == receipt_count,
        "projection transport root count differs",
    )
    _require(
        _count(projection.get("host_operation_root_count"), "projection host root count", receipt_max) == receipt_count,
        "projection host root count differs",
    )
    _require(
        _count(projection.get("release_pcr_commitment_count"), "projection release count", boot_max) == boot_count,
        "projection release count differs",
    )
    _require(
        edge_count >= max(0, receipt_count - 1) + len(external),
        "projection ancestry is too sparse",
    )
    body = {field: projection[field] for field in _PROJECTION_BODY_FIELDS}
    _require(
        projection.get("projection_hash") == _domain_hash("leadpoet-ancestry-projection-v2", body),
        "projection hash differs",
    )
    return dict(projection)


def project_receipt_graph_v2(
    graph: Mapping[str, Any],
    *,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_failed_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
) -> Dict[str, Any]:
    """Validate and project an unchanged complete durable V2 graph."""

    _require(callable(boot_attestation_verifier), "full projection requires a boot verifier")
    _validate_evidence_shape(
        graph,
        schema_version=RECEIPT_GRAPH_SCHEMA_VERSION,
        receipt_max=MAX_FULL_GRAPH_RECEIPTS,
        boot_max=MAX_FULL_GRAPH_BOOTS,
        attempt_max=MAX_FULL_GRAPH_ATTEMPTS,
        host_max=MAX_FULL_GRAPH_HOST_OPERATIONS,
        edge_max=MAX_FULL_GRAPH_EDGES,
    )
    allowed = _sorted_hashes(list(allowed_failed_receipt_hashes), "allowed failed receipt hash", MAX_ALLOWED_FAILED_RECEIPTS)
    purposes = _sorted_identifiers(list(required_purposes), "required purpose", MAX_REQUIRED_PURPOSES)
    try:
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=allowed,
            required_purposes=purposes,
            boot_attestation_verifier=boot_attestation_verifier,
            require_boot_attestation_verification=True,
        )
    except Exception as exc:
        raise AncestryCheckpointV2Error("full receipt graph failed attested validation") from exc
    normalized = _normalized_evidence(graph, RECEIPT_GRAPH_SCHEMA_VERSION)
    return _projection_from_validated_evidence(
        normalized,
        projection_kind="full_graph",
        external_parent_receipt_hashes=(),
    )


def _validate_evidence_shape(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    receipt_max: int,
    boot_max: int,
    attempt_max: int,
    host_max: int,
    edge_max: int,
) -> None:
    expected_fields = _DELTA_FIELDS
    _exact(value, expected_fields, "ancestry evidence")
    _require(value.get("schema_version") == schema_version, "ancestry evidence schema is invalid")
    for field, maximum in (
        ("receipts", receipt_max),
        ("boot_identities", boot_max),
        ("transport_attempts", attempt_max),
        ("host_operations", host_max),
    ):
        rows = value.get(field)
        _require(isinstance(rows, list), "%s must be an array" % field)
        _require(len(rows) <= maximum, "%s exceeds canonical bound" % field)
    _require(bool(value["receipts"]), "ancestry evidence has no receipts")
    edges = sum(
        len(item.get("parent_receipt_hashes") or ())
        for item in value["receipts"]
        if isinstance(item, Mapping)
    )
    _require(edges <= edge_max, "ancestry evidence parent edges exceed canonical bound")


def build_ancestry_policy_v2(
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
) -> Dict[str, Any]:
    body = {
        "schema_version": ANCESTRY_POLICY_SCHEMA_VERSION,
        "allowed_failed_receipt_hashes": _sorted_hashes(
            list(allowed_failed_receipt_hashes),
            "allowed failed receipt hash",
            MAX_ALLOWED_FAILED_RECEIPTS,
        ),
        "required_purposes": _sorted_identifiers(
            list(required_purposes), "required purpose", MAX_REQUIRED_PURPOSES
        ),
    }
    return {**body, "policy_hash": _domain_hash("leadpoet-ancestry-policy-v2", body)}


def validate_ancestry_policy_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    policy = _exact(value, _POLICY_FIELDS, "ancestry policy")
    _require(policy.get("schema_version") == ANCESTRY_POLICY_SCHEMA_VERSION, "ancestry policy schema is invalid")
    rebuilt = build_ancestry_policy_v2(
        allowed_failed_receipt_hashes=policy.get("allowed_failed_receipt_hashes"),
        required_purposes=policy.get("required_purposes"),
    )
    _require(dict(policy) == rebuilt, "ancestry policy is not canonical")
    return rebuilt


def build_full_graph_parent_v2(
    graph: Mapping[str, Any],
    *,
    allowed_failed_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
) -> Dict[str, Any]:
    """Package one legacy graph with its exact, graph-local validation policy."""

    _require(isinstance(graph, Mapping), "full graph parent graph must be an object")
    return {
        "schema_version": ANCESTRY_FULL_GRAPH_PARENT_SCHEMA_VERSION,
        "graph": dict(graph),
        "policy": build_ancestry_policy_v2(
            allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
            required_purposes=required_purposes,
        ),
    }


def _validate_full_graph_parent_shape(value: Mapping[str, Any]) -> Dict[str, Any]:
    parent = _exact(value, _FULL_GRAPH_PARENT_FIELDS, "full graph parent")
    _require(
        parent.get("schema_version") == ANCESTRY_FULL_GRAPH_PARENT_SCHEMA_VERSION,
        "full graph parent schema is invalid",
    )
    _require(isinstance(parent.get("graph"), Mapping), "full graph parent graph is invalid")
    return {
        "schema_version": ANCESTRY_FULL_GRAPH_PARENT_SCHEMA_VERSION,
        "graph": dict(parent["graph"]),
        "policy": validate_ancestry_policy_v2(parent.get("policy")),
    }


def _validated_full_graph_parent(
    value: Mapping[str, Any],
    *,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parent = _validate_full_graph_parent_shape(value)
    policy = parent["policy"]
    projection = project_receipt_graph_v2(
        parent["graph"],
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_failed_receipt_hashes=policy["allowed_failed_receipt_hashes"],
        required_purposes=policy["required_purposes"],
    )
    return parent, projection


def validate_full_graph_parent_v2(
    value: Mapping[str, Any],
    *,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any]
) -> Dict[str, Any]:
    """Strictly validate a legacy graph together with its bound policy."""

    parent, _ = _validated_full_graph_parent(
        value, boot_attestation_verifier=boot_attestation_verifier
    )
    return parent


def _parent_descriptor_from_projection(
    projection: Mapping[str, Any], policy: Mapping[str, Any]
) -> Dict[str, Any]:
    value = validate_ancestry_projection_v2(projection)
    policy_doc = validate_ancestry_policy_v2(policy)
    _require(value["projection_kind"] == "full_graph", "full parent authority is not a full projection")
    return {
        "schema_version": ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "full_projection",
        "parent_receipt_hash": value["root_receipt_hash"],
        "parent_epoch_id": value["root_epoch_id"],
        "parent_role": value["root_role"],
        "parent_purpose": value["root_purpose"],
        "authority_hash": _domain_hash(
            "leadpoet-ancestry-full-parent-authority-v2",
            {
                "projection_hash": value["projection_hash"],
                "policy_hash": policy_doc["policy_hash"],
            },
        ),
        "authority_policy_hash": policy_doc["policy_hash"],
        "authority_sequence": None,
        "authority_purposes": list(value["ancestry_purposes"]),
    }


def _parent_descriptor_from_certificate(certificate: Mapping[str, Any]) -> Dict[str, Any]:
    claim = certificate["claim"]
    projection = claim["local_delta_projection"]
    return {
        "schema_version": ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "certificate",
        "parent_receipt_hash": claim["output_root_receipt_hash"],
        "parent_epoch_id": projection["root_epoch_id"],
        "parent_role": projection["root_role"],
        "parent_purpose": projection["root_purpose"],
        "authority_hash": certificate["certificate_hash"],
        "authority_policy_hash": claim["policy"]["policy_hash"],
        "authority_sequence": claim["certificate_sequence"],
        "authority_purposes": list(projection["ancestry_purposes"]),
    }


def _parent_descriptor_from_certificate_disclosure(
    proof: Mapping[str, Any], receipt_hash: str
) -> Dict[str, Any]:
    certificate = proof["certificate"]
    claim = certificate["claim"]
    disclosed = [
        item
        for item in proof["disclosed_receipts"]
        if item.get("receipt_hash") == receipt_hash
    ]
    _require(
        len(disclosed) == 1,
        "certificate disclosure authority receipt is ambiguous",
    )
    return {
        "schema_version": ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "certificate_disclosure",
        "parent_receipt_hash": receipt_hash,
        "parent_epoch_id": int(disclosed[0]["epoch_id"]),
        "parent_role": str(disclosed[0]["role"]),
        "parent_purpose": str(disclosed[0]["purpose"]),
        "authority_hash": _domain_hash(
            "leadpoet-ancestry-certificate-disclosure-authority-v2",
            {
                "certificate_hash": certificate["certificate_hash"],
                "proof_hash": proof["proof_hash"],
                "disclosed_receipt_hash": receipt_hash,
            },
        ),
        "authority_policy_hash": claim["policy"]["policy_hash"],
        "authority_sequence": claim["certificate_sequence"],
        "authority_purposes": list(
            claim["local_delta_projection"]["ancestry_purposes"]
        ),
    }


def build_full_graph_parent_authority_v2(
    full_graph_parent: Mapping[str, Any],
    *,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any]
) -> Dict[str, Any]:
    """Recompute the exact descriptor for one policy-bound legacy graph."""

    parent, projection = _validated_full_graph_parent(
        full_graph_parent,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    return _parent_descriptor_from_projection(projection, parent["policy"])


def build_certificate_parent_authority_v2(
    certificate: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    """Recompute the exact descriptor for one verified predecessor cert."""

    normalized = validate_ancestry_certificate_v2(
        certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    return _parent_descriptor_from_certificate(normalized)


def build_certificate_disclosure_parent_authority_v2(
    proof: Mapping[str, Any],
    receipt_hash: str,
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    """Recompute one exact parent descriptor from a verified disclosure."""

    normalized = validate_compact_ancestry_proof_v2(
        proof,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        required_receipt_hashes=(receipt_hash,),
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    return _parent_descriptor_from_certificate_disclosure(
        normalized,
        _hash(receipt_hash, "certificate disclosure receipt hash"),
    )


def _validate_parent_authority(value: Mapping[str, Any]) -> Dict[str, Any]:
    item = _exact(value, _PARENT_AUTHORITY_FIELDS, "parent authority")
    _require(item.get("schema_version") == ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION, "parent authority schema is invalid")
    kind = str(item.get("authority_kind") or "")
    _require(
        kind in {"full_projection", "certificate", "certificate_disclosure"},
        "parent authority kind is invalid",
    )
    _hash(item.get("parent_receipt_hash"), "parent authority receipt hash")
    _hash(item.get("authority_hash"), "parent authority hash")
    _hash(item.get("authority_policy_hash"), "parent authority policy hash")
    _identifier(item.get("parent_role"), "parent authority role")
    _identifier(item.get("parent_purpose"), "parent authority root purpose")
    purposes = _sorted_identifiers(
        item.get("authority_purposes"),
        "parent authority purpose",
        MAX_REQUIRED_PURPOSES,
    )
    _require(
        list(item["authority_purposes"]) == purposes,
        "parent authority purposes are not canonical",
    )
    _require(
        item["parent_purpose"] in purposes,
        "parent authority root purpose is absent from ancestry",
    )
    _count(item.get("parent_epoch_id"), "parent authority epoch", 2 ** 63 - 1)
    if kind in {"certificate", "certificate_disclosure"}:
        _count(item.get("authority_sequence"), "parent authority sequence", 2 ** 63 - 1)
    else:
        _require(item.get("authority_sequence") is None, "full projection has a certificate sequence")
    return dict(item)


def _validate_delta_and_project(
    delta: Mapping[str, Any],
    *,
    parent_authorities: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    _require(callable(boot_attestation_verifier), "local delta requires a boot verifier")
    _validate_evidence_shape(
        delta,
        schema_version=ANCESTRY_DELTA_SCHEMA_VERSION,
        receipt_max=MAX_DELTA_RECEIPTS,
        boot_max=MAX_DELTA_BOOTS,
        attempt_max=MAX_DELTA_ATTEMPTS,
        host_max=MAX_DELTA_HOST_OPERATIONS,
        edge_max=MAX_DELTA_EDGES,
    )
    normalized = _normalized_evidence(delta, ANCESTRY_DELTA_SCHEMA_VERSION)
    policy_doc = validate_ancestry_policy_v2(policy)
    authorities = [_validate_parent_authority(item) for item in parent_authorities]
    authority_by_root = {str(item["parent_receipt_hash"]): item for item in authorities}
    _require(len(authority_by_root) == len(authorities), "parent authority root is duplicated")
    _require(len(authorities) <= MAX_PARENT_AUTHORITIES, "parent authority count exceeds bound")

    boots = {}  # type: Dict[str, Mapping[str, Any]]
    for identity in normalized["boot_identities"]:
        try:
            validate_boot_identity(identity)
            boot_attestation_verifier(identity)
        except Exception as exc:
            raise AncestryCheckpointV2Error("local delta boot is not approved") from exc
        key = str(identity["boot_identity_hash"])
        _require(key not in boots, "local delta boot is duplicated")
        boots[key] = identity
    receipts = {}  # type: Dict[str, Mapping[str, Any]]
    scopes = set()
    allowed_failed = set(policy_doc["allowed_failed_receipt_hashes"])
    observed_failed = set()
    used_boots = set()
    for receipt in normalized["receipts"]:
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise AncestryCheckpointV2Error("local delta receipt signature is invalid") from exc
        receipt_hash = str(receipt["receipt_hash"])
        _require(receipt_hash not in receipts, "local delta receipt is duplicated")
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        _require(scope not in scopes, "local delta receipt scope is duplicated")
        scopes.add(scope)
        boot = boots.get(str(receipt["boot_identity_hash"]))
        _require(boot is not None, "local delta receipt boot is missing")
        used_boots.add(str(receipt["boot_identity_hash"]))
        for receipt_field, boot_field in (
            ("role", "role"),
            ("commit_sha", "commit_sha"),
            ("pcr0", "pcr0"),
            ("build_manifest_hash", "build_manifest_hash"),
            ("dependency_lock_hash", "dependency_lock_hash"),
            ("config_hash", "config_hash"),
            ("enclave_pubkey", "signing_pubkey"),
        ):
            _require(receipt[receipt_field] == boot[boot_field], "local receipt differs from boot at %s" % receipt_field)
        if receipt["status"] != "succeeded":
            _require(receipt_hash in allowed_failed, "local delta contains unauthorized failed receipt")
            observed_failed.add(receipt_hash)
        receipts[receipt_hash] = receipt
    _require(
        observed_failed == allowed_failed,
        "local delta failed receipt policy is not exact",
    )
    _require(used_boots == set(boots), "local delta contains unused boot identity")
    observed_purposes = {str(item["purpose"]) for item in receipts.values()}
    _require(
        set(policy_doc["required_purposes"]).issubset(observed_purposes),
        "local delta is missing required purpose",
    )

    attempts_by_scope = {}  # type: Dict[Tuple[str, str], List[Mapping[str, Any]]]
    attempt_hashes = set()
    for attempt in normalized["transport_attempts"]:
        try:
            validate_transport_attempt(attempt)
        except Exception as exc:
            raise AncestryCheckpointV2Error("local delta transport attempt is invalid") from exc
        key = str(attempt["attempt_hash"])
        _require(key not in attempt_hashes, "local delta attempt is duplicated")
        attempt_hashes.add(key)
        attempts_by_scope.setdefault((str(attempt["job_id"]), str(attempt["purpose"])), []).append(attempt)
    for receipt in receipts.values():
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        _require(transport_root(attempts_by_scope.pop(scope, [])) == receipt["transport_root"], "local receipt transport root differs")
    _require(not attempts_by_scope, "local delta contains unclaimed attempts")

    hosts_by_scope = {}  # type: Dict[Tuple[str, str], List[Mapping[str, Any]]]
    host_keys = set()
    for record in normalized["host_operations"]:
        try:
            validate_host_operation_record(record)
        except Exception as exc:
            raise AncestryCheckpointV2Error("local delta host operation is invalid") from exc
        key = _host_key(record)
        _require(key not in host_keys, "local delta host operation is duplicated")
        host_keys.add(key)
        request = record["request"]
        hosts_by_scope.setdefault((str(request["job_id"]), str(request["purpose"])), []).append(record)
    for receipt in receipts.values():
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        records = hosts_by_scope.pop(scope, [])
        for record in records:
            request = record["request"]
            _require(request["enclave_pubkey"] == receipt["enclave_pubkey"], "host operation key differs from receipt")
            _require(request["boot_identity_hash"] == receipt["boot_identity_hash"], "host operation boot differs from receipt")
        _require(host_operation_root(records) == receipt["host_operation_root"], "local receipt host root differs")
    _require(not hosts_by_scope, "local delta contains unclaimed host operations")

    root_hash = str(normalized["root_receipt_hash"])
    _require(root_hash in receipts, "local delta root receipt is missing")
    visiting = set()
    visited = set()
    used_external = set()

    def visit(receipt_hash: str) -> None:
        if receipt_hash in visited:
            return
        _require(receipt_hash not in visiting, "local delta contains a cycle")
        visiting.add(receipt_hash)
        receipt = receipts[receipt_hash]
        for parent_hash in receipt["parent_receipt_hashes"]:
            if parent_hash in receipts:
                _require(int(receipts[parent_hash]["epoch_id"]) <= int(receipt["epoch_id"]), "local parent epoch is newer than child")
                visit(parent_hash)
            else:
                authority = authority_by_root.get(str(parent_hash))
                _require(authority is not None, "local delta parent authority is missing")
                _require(int(authority["parent_epoch_id"]) <= int(receipt["epoch_id"]), "external parent epoch is newer than child")
                used_external.add(str(parent_hash))
        visiting.remove(receipt_hash)
        visited.add(receipt_hash)

    visit(root_hash)
    _require(visited == set(receipts), "local delta contains disconnected receipts")
    _require(used_external == set(authority_by_root), "local delta contains unused or omitted parent authority")
    ancestry_purposes = set(observed_purposes)
    for authority in authorities:
        ancestry_purposes.update(authority["authority_purposes"])
    return _projection_from_validated_evidence(
        normalized,
        projection_kind="local_delta",
        external_parent_receipt_hashes=sorted(used_external),
        ancestry_purposes=ancestry_purposes,
    )


def _claim_hash(body: Mapping[str, Any]) -> str:
    return _domain_hash("leadpoet-ancestry-certificate-claim-v2", dict(body))


def _certificate_hash(claim_hash: str, pubkey: str, signature: str) -> str:
    return _domain_hash(
        "leadpoet-ancestry-detached-certificate-v2",
        {"claim_hash": claim_hash, "enclave_pubkey": pubkey, "enclave_signature": signature},
    )


def issue_ancestry_certificate_v2(
    *,
    local_delta: Mapping[str, Any],
    lineage_id: str,
    certificate_sequence: int,
    issuer_boot_identity: Mapping[str, Any],
    issued_at: str,
    sign_digest: Callable[[bytes], Any],
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    parent_certificates: Sequence[Mapping[str, Any]] = (),
    parent_proof_disclosures: Sequence[
        Tuple[Mapping[str, Any], str]
    ] = (),
    parent_full_graphs: Sequence[Mapping[str, Any]] = (),
    allowed_failed_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
) -> Dict[str, Any]:
    """Validate exact predecessor authorities and issue a detached certificate."""

    _require(callable(sign_digest), "certificate signer is unavailable")
    lineage = _hash(lineage_id, "lineage id")
    sequence = _count(certificate_sequence, "certificate sequence", 2 ** 63 - 1)
    roles = set(_sorted_identifiers(list(allowed_issuer_roles), "allowed issuer role", 16))
    try:
        validate_boot_identity(issuer_boot_identity)
        boot_attestation_verifier(issuer_boot_identity)
    except Exception as exc:
        raise AncestryCheckpointV2Error("certificate issuer boot is not approved") from exc
    _require(str(issuer_boot_identity["role"]) in roles, "certificate issuer role is unauthorized")
    policy = build_ancestry_policy_v2(
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
        required_purposes=required_purposes,
    )
    _require(
        len(parent_certificates)
        + len(parent_proof_disclosures)
        + len(parent_full_graphs)
        <= MAX_PARENT_AUTHORITIES,
        "parent authority count exceeds bound",
    )
    descriptors = []  # type: List[Dict[str, Any]]
    for parent_value in parent_full_graphs:
        full_parent, projection = _validated_full_graph_parent(
            parent_value,
            boot_attestation_verifier=boot_attestation_verifier,
        )
        parent_policy = full_parent["policy"]
        descriptors.append(
            _parent_descriptor_from_projection(projection, parent_policy)
        )
    parent_sequences = []  # type: List[int]
    for certificate in parent_certificates:
        normalized = validate_ancestry_certificate_v2(
            certificate,
            expected_lineage_id=lineage,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles=roles,
        )
        descriptors.append(_parent_descriptor_from_certificate(normalized))
        parent_sequences.append(int(normalized["claim"]["certificate_sequence"]))
    for raw_proof, raw_receipt_hash in parent_proof_disclosures:
        normalized_proof = validate_compact_ancestry_proof_v2(
            raw_proof,
            expected_lineage_id=lineage,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles=roles,
            required_receipt_hashes=(raw_receipt_hash,),
        )
        receipt_hash = _hash(
            raw_receipt_hash, "certificate disclosure receipt hash"
        )
        descriptors.append(
            _parent_descriptor_from_certificate_disclosure(
                normalized_proof, receipt_hash
            )
        )
        parent_sequences.append(
            int(
                normalized_proof["certificate"]["claim"][
                    "certificate_sequence"
                ]
            )
        )
    _require(len(descriptors) <= MAX_PARENT_AUTHORITIES, "parent authority count exceeds bound")
    descriptors.sort(key=lambda item: str(item["parent_receipt_hash"]))
    _require(len({item["parent_receipt_hash"] for item in descriptors}) == len(descriptors), "parent authority root is duplicated")
    expected_sequence = max(parent_sequences) + 1 if parent_sequences else 0
    _require(sequence == expected_sequence, "certificate sequence does not extend predecessors")
    projection = _validate_delta_and_project(
        local_delta,
        parent_authorities=descriptors,
        policy=policy,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    descriptor_hashes = [
        _domain_hash("leadpoet-ancestry-parent-authority-leaf-v2", item)
        for item in descriptors
    ]
    body = {
        "schema_version": ANCESTRY_CERTIFICATE_CLAIM_SCHEMA_VERSION,
        "lineage_id": lineage,
        "certificate_sequence": sequence,
        "output_root_receipt_hash": projection["root_receipt_hash"],
        "local_delta_projection": projection,
        "parent_authorities": descriptors,
        "parent_authorities_root": merkle_root(
            descriptor_hashes, domain="leadpoet-ancestry-parent-authorities-v2"
        ),
        "policy": policy,
        "issuer_boot_identity_hash": issuer_boot_identity["boot_identity_hash"],
        "issued_at": _timestamp(issued_at, "certificate issued_at"),
    }
    claim = {**body, "claim_hash": _claim_hash(body)}
    raw_signature = sign_digest(bytes.fromhex(claim["claim_hash"].split(":", 1)[1]))
    signature = _signature(raw_signature)
    pubkey = str(issuer_boot_identity["signing_pubkey"])
    certificate = {
        "schema_version": ANCESTRY_CERTIFICATE_SCHEMA_VERSION,
        "claim": claim,
        "issuer_boot_identity": dict(issuer_boot_identity),
        "enclave_pubkey": pubkey,
        "enclave_signature": signature,
        "certificate_hash": _certificate_hash(claim["claim_hash"], pubkey, signature),
    }
    return validate_ancestry_certificate_v2(
        certificate,
        expected_lineage_id=lineage,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=roles,
    )


def validate_ancestry_certificate_v2(
    value: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    certificate = _exact(value, _CERTIFICATE_FIELDS, "ancestry certificate")
    _require(certificate.get("schema_version") == ANCESTRY_CERTIFICATE_SCHEMA_VERSION, "ancestry certificate schema is invalid")
    _require(callable(boot_attestation_verifier), "certificate boot verifier is unavailable")
    roles = set(_sorted_identifiers(list(allowed_issuer_roles), "allowed issuer role", 16))
    claim = _exact(certificate.get("claim"), _CLAIM_FIELDS, "ancestry certificate claim")
    _require(claim.get("schema_version") == ANCESTRY_CERTIFICATE_CLAIM_SCHEMA_VERSION, "certificate claim schema is invalid")
    lineage = _hash(expected_lineage_id, "expected lineage id")
    _require(claim.get("lineage_id") == lineage, "certificate lineage differs")
    sequence = _count(claim.get("certificate_sequence"), "certificate sequence", 2 ** 63 - 1)
    _require(sequence >= _count(minimum_certificate_sequence, "minimum certificate sequence", 2 ** 63 - 1), "certificate sequence rewinds")
    _timestamp(claim.get("issued_at"), "certificate issued_at")
    projection = validate_ancestry_projection_v2(claim.get("local_delta_projection"))
    _require(projection["projection_kind"] == "local_delta", "certificate projection is not local delta")
    _require(claim.get("output_root_receipt_hash") == projection["root_receipt_hash"], "certificate output root differs")
    policy = validate_ancestry_policy_v2(claim.get("policy"))
    parent_rows = claim.get("parent_authorities")
    _require(isinstance(parent_rows, list), "certificate parent authorities must be an array")
    _require(len(parent_rows) <= MAX_PARENT_AUTHORITIES, "certificate parent authorities exceed bound")
    parents = [_validate_parent_authority(item) for item in parent_rows]
    _require(parents == sorted(parents, key=lambda item: str(item["parent_receipt_hash"])), "certificate parent authorities are not sorted")
    _require(len({item["parent_receipt_hash"] for item in parents}) == len(parents), "certificate parent authority is duplicated")
    _require(
        [item["parent_receipt_hash"] for item in parents] == projection["external_parent_receipt_hashes"],
        "certificate parent authorities differ from local delta boundary",
    )
    descriptor_hashes = [_domain_hash("leadpoet-ancestry-parent-authority-leaf-v2", item) for item in parents]
    _require(
        claim.get("parent_authorities_root") == merkle_root(descriptor_hashes, domain="leadpoet-ancestry-parent-authorities-v2"),
        "certificate parent authority root differs",
    )
    parent_sequences = [
        int(item["authority_sequence"])
        for item in parents
        if item["authority_kind"]
        in {"certificate", "certificate_disclosure"}
    ]
    expected_sequence = max(parent_sequences) + 1 if parent_sequences else 0
    _require(sequence == expected_sequence, "certificate sequence does not extend parent authorities")
    _require(all(int(item["parent_epoch_id"]) <= int(projection["root_epoch_id"]) for item in parents), "certificate output epoch rewinds parent")
    if trusted_parent_authorities is not None:
        trusted = {
            _hash(root, "trusted parent root"): _hash(authority, "trusted parent authority hash")
            for root, authority in dict(trusted_parent_authorities).items()
        }
        observed = {str(item["parent_receipt_hash"]): str(item["authority_hash"]) for item in parents}
        _require(observed == trusted, "certificate parent authorities differ from trusted state")
    body = {field: claim[field] for field in _CLAIM_BODY_FIELDS}
    expected_claim_hash = _claim_hash(body)
    _require(claim.get("claim_hash") == expected_claim_hash, "certificate claim hash differs")
    boot = certificate.get("issuer_boot_identity")
    try:
        validate_boot_identity(boot)
        boot_attestation_verifier(boot)
    except Exception as exc:
        raise AncestryCheckpointV2Error("certificate issuer boot is not approved") from exc
    _require(str(boot["role"]) in roles, "certificate issuer role is unauthorized")
    _require(claim.get("issuer_boot_identity_hash") == boot["boot_identity_hash"], "certificate issuer boot differs")
    pubkey = str(certificate.get("enclave_pubkey") or "").lower()
    _require(pubkey == boot["signing_pubkey"], "certificate signing key differs from boot")
    signature = _signature(certificate.get("enclave_signature"))
    _verify_ed25519(pubkey, signature, expected_claim_hash)
    expected_certificate_hash = _certificate_hash(expected_claim_hash, pubkey, signature)
    _require(certificate.get("certificate_hash") == expected_certificate_hash, "certificate hash differs")
    known = {
        _hash(root, "known certificate root"): _hash(cert_hash, "known certificate hash")
        for root, cert_hash in dict(known_certificate_hashes_by_root or {}).items()
    }
    known_hash = known.get(str(claim["output_root_receipt_hash"]))
    _require(known_hash is None or known_hash == expected_certificate_hash, "certificate conflicts with known fork")
    return {
        "schema_version": ANCESTRY_CERTIFICATE_SCHEMA_VERSION,
        "claim": {**dict(claim), "local_delta_projection": projection, "policy": policy, "parent_authorities": parents},
        "issuer_boot_identity": dict(boot),
        "enclave_pubkey": pubkey,
        "enclave_signature": signature,
        "certificate_hash": expected_certificate_hash,
    }


def validate_ancestry_parent_authority_v2(
    value: Mapping[str, Any]
) -> Dict[str, Any]:
    """Public strict validator for a canonical frontier authority descriptor."""

    return _validate_parent_authority(value)


def validate_ancestry_delta_v2(
    local_delta: Mapping[str, Any],
    *,
    parent_authorities: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Validate bounded local bodies and return their canonical projection."""

    return _validate_delta_and_project(
        local_delta,
        parent_authorities=parent_authorities,
        policy=policy,
        boot_attestation_verifier=boot_attestation_verifier,
    )


def _validated_certificate_and_local_delta(
    local_delta: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_certificate = validate_ancestry_certificate_v2(
        certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    claim = normalized_certificate["claim"]
    projection = _validate_delta_and_project(
        local_delta,
        parent_authorities=claim["parent_authorities"],
        policy=claim["policy"],
        boot_attestation_verifier=boot_attestation_verifier,
    )
    _require(
        projection == claim["local_delta_projection"],
        "local delta differs from certificate evidence commitment",
    )
    return normalized_certificate, projection


def validate_local_delta_against_certificate_v2(
    local_delta: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    """Prove persisted local attempts/operations are the exact certified bodies."""

    _, projection = _validated_certificate_and_local_delta(
        local_delta,
        certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    return projection


def _extract_local_delta_from_full_graph(
    graph: Mapping[str, Any], certificate: Mapping[str, Any]
) -> Dict[str, Any]:
    projection = certificate["claim"]["local_delta_projection"]
    receipt_hashes = set(projection["receipt_hashes"])
    boot_hashes = set(projection["boot_identity_hashes"])
    receipts = [dict(item) for item in graph["receipts"] if str(item.get("receipt_hash")) in receipt_hashes]
    boots = [dict(item) for item in graph["boot_identities"] if str(item.get("boot_identity_hash")) in boot_hashes]
    scopes = {(str(item["job_id"]), str(item["purpose"])) for item in receipts}
    attempts = [dict(item) for item in graph["transport_attempts"] if (str(item.get("job_id")), str(item.get("purpose"))) in scopes]
    hosts = [
        dict(item)
        for item in graph["host_operations"]
        if isinstance(item, Mapping)
        and isinstance(item.get("request"), Mapping)
        and (str(item["request"].get("job_id")), str(item["request"].get("purpose"))) in scopes
    ]
    _require({str(item["receipt_hash"]) for item in receipts} == receipt_hashes, "full graph is missing certified local receipt")
    _require({str(item["boot_identity_hash"]) for item in boots} == boot_hashes, "full graph is missing certified local boot")
    return {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": projection["root_receipt_hash"],
        "boot_identities": boots,
        "receipts": receipts,
        "transport_attempts": attempts,
        "host_operations": hosts,
    }


def _proof_hash(body: Mapping[str, Any]) -> str:
    return _domain_hash("leadpoet-ancestry-compact-proof-v2", dict(body))


def build_compact_ancestry_proof_from_delta_v2(
    local_delta: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    """Build the transport proof from bounded local evidence only.

    Attempts and host-operation bodies are validated against their signed
    receipt roots and the certificate evidence commitment, then deliberately
    omitted from the transport object.  They remain durable and independently
    expandable at the host.
    """

    normalized_certificate, _ = _validated_certificate_and_local_delta(
        local_delta,
        certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    normalized_delta = _normalized_evidence(
        local_delta, ANCESTRY_DELTA_SCHEMA_VERSION
    )
    body = {
        "schema_version": ANCESTRY_COMPACT_PROOF_SCHEMA_VERSION,
        "certificate": normalized_certificate,
        "disclosed_receipts": normalized_delta["receipts"],
        "disclosed_boot_identities": normalized_delta["boot_identities"],
    }
    return {**body, "proof_hash": _proof_hash(body)}


def build_compact_ancestry_proof_v2(
    full_graph: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    allowed_failed_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build a compact proof while retaining the supplied full graph unchanged."""

    project_receipt_graph_v2(
        full_graph,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
        required_purposes=required_purposes,
    )
    normalized_certificate = validate_ancestry_certificate_v2(
        certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
    )
    delta = _extract_local_delta_from_full_graph(full_graph, normalized_certificate)
    return build_compact_ancestry_proof_from_delta_v2(
        delta,
        normalized_certificate,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
    )


def _projection_from_disclosures(
    *,
    projection: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    boots: Sequence[Mapping[str, Any]],
    parent_authorities: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    certificate_sequence: int,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> None:
    policy_doc = validate_ancestry_policy_v2(policy)
    current_sequence = _count(
        certificate_sequence, "certificate sequence", 2 ** 63 - 1
    )
    receipt_docs = [dict(item) for item in receipts]
    boot_docs = [dict(item) for item in boots]
    _require(receipt_docs == sorted(receipt_docs, key=lambda item: str(item.get("receipt_hash") or "")), "proof receipts are reordered")
    _require(boot_docs == sorted(boot_docs, key=lambda item: str(item.get("boot_identity_hash") or "")), "proof boots are reordered")
    boot_by_hash = {}  # type: Dict[str, Mapping[str, Any]]
    for boot in boot_docs:
        try:
            validate_boot_identity(boot)
            boot_attestation_verifier(boot)
        except Exception as exc:
            raise AncestryCheckpointV2Error("proof boot is not approved") from exc
        key = str(boot["boot_identity_hash"])
        _require(key not in boot_by_hash, "proof boot is duplicated")
        boot_by_hash[key] = boot
    receipt_by_hash = {}  # type: Dict[str, Mapping[str, Any]]
    observed_failed = set()
    used_boots = set()
    for receipt in receipt_docs:
        try:
            validate_signed_execution_receipt(receipt)
        except Exception as exc:
            raise AncestryCheckpointV2Error("proof receipt signature is invalid") from exc
        key = str(receipt["receipt_hash"])
        _require(key not in receipt_by_hash, "proof receipt is duplicated")
        boot = boot_by_hash.get(str(receipt["boot_identity_hash"]))
        _require(boot is not None, "proof receipt boot is missing")
        used_boots.add(str(receipt["boot_identity_hash"]))
        for receipt_field, boot_field in (
            ("role", "role"), ("commit_sha", "commit_sha"), ("pcr0", "pcr0"),
            ("build_manifest_hash", "build_manifest_hash"),
            ("dependency_lock_hash", "dependency_lock_hash"),
            ("config_hash", "config_hash"), ("enclave_pubkey", "signing_pubkey"),
        ):
            _require(receipt[receipt_field] == boot[boot_field], "proof receipt differs from boot")
        if receipt["status"] != "succeeded":
            observed_failed.add(key)
        receipt_by_hash[key] = receipt
    _require(
        observed_failed == set(policy_doc["allowed_failed_receipt_hashes"]),
        "proof failed receipt policy is not exact",
    )
    _require(used_boots == set(boot_by_hash), "proof contains unused boot identity")
    observed_purposes = {str(item["purpose"]) for item in receipt_by_hash.values()}
    _require(
        set(policy_doc["required_purposes"]).issubset(observed_purposes),
        "proof is missing signed-policy purpose",
    )
    _require(sorted(receipt_by_hash) == projection["receipt_hashes"], "proof receipt disclosure is incomplete")
    _require(sorted(boot_by_hash) == projection["boot_identity_hashes"], "proof boot disclosure is incomplete")
    root = receipt_by_hash.get(str(projection["root_receipt_hash"]))
    _require(root is not None, "proof output root receipt is missing")
    _require(
        int(root["epoch_id"]) == projection["root_epoch_id"]
        and root["role"] == projection["root_role"]
        and root["purpose"] == projection["root_purpose"]
        and root["boot_identity_hash"] == projection["root_boot_identity_hash"],
        "proof output root metadata differs",
    )
    authorities = {str(item["parent_receipt_hash"]): item for item in parent_authorities}
    for authority in authorities.values():
        if authority["authority_kind"] in {
            "certificate",
            "certificate_disclosure",
        }:
            _require(
                int(authority["authority_sequence"]) < current_sequence,
                "proof parent certificate sequence does not precede child",
            )
    edges = []
    transport_leaves = []
    host_leaves = []
    visited = set()
    visiting = set()
    used_external = set()

    def visit(receipt_hash: str) -> None:
        if receipt_hash in visited:
            return
        _require(receipt_hash not in visiting, "proof receipt delta contains a cycle")
        visiting.add(receipt_hash)
        receipt = receipt_by_hash[receipt_hash]
        for parent_hash in receipt["parent_receipt_hashes"]:
            if parent_hash in receipt_by_hash:
                _require(
                    int(receipt_by_hash[parent_hash]["epoch_id"])
                    <= int(receipt["epoch_id"]),
                    "proof local parent epoch is newer than child",
                )
                visit(parent_hash)
            else:
                _require(parent_hash in authorities, "proof parent authority is missing")
                _require(
                    int(authorities[parent_hash]["parent_epoch_id"])
                    <= int(receipt["epoch_id"]),
                    "proof external parent epoch is newer than child",
                )
                used_external.add(str(parent_hash))
            edges.append(_domain_hash("leadpoet-ancestry-parent-edge-leaf-v2", {"child_receipt_hash": receipt_hash, "parent_receipt_hash": str(parent_hash)}))
        visiting.remove(receipt_hash)
        visited.add(receipt_hash)

    visit(str(projection["root_receipt_hash"]))
    _require(visited == set(receipt_by_hash), "proof contains disconnected receipt disclosure")
    _require(used_external == set(authorities), "proof parent authorities contain extras")
    for receipt in receipt_docs:
        transport_leaves.append(_domain_hash("leadpoet-ancestry-transport-root-leaf-v2", {"receipt_hash": receipt["receipt_hash"], "boot_identity_hash": receipt["boot_identity_hash"], "transport_root": receipt["transport_root"]}))
        host_leaves.append(_domain_hash("leadpoet-ancestry-host-operation-root-leaf-v2", {"receipt_hash": receipt["receipt_hash"], "boot_identity_hash": receipt["boot_identity_hash"], "host_operation_root": receipt["host_operation_root"]}))
    release_leaves = [_release_leaf(item) for item in boot_docs]
    checks = {
        "receipt_hashes_root": merkle_root(sorted(receipt_by_hash), domain="leadpoet-ancestry-receipts-v2"),
        "parent_edges_root": merkle_root(edges, domain="leadpoet-ancestry-parent-edges-v2"),
        "boot_identities_root": merkle_root(sorted(boot_by_hash), domain="leadpoet-ancestry-boots-v2"),
        "transport_roots_root": merkle_root(transport_leaves, domain="leadpoet-ancestry-transport-roots-v2"),
        "host_operation_roots_root": merkle_root(host_leaves, domain="leadpoet-ancestry-host-operation-roots-v2"),
        "release_pcr_commitments_root": merkle_root(release_leaves, domain="leadpoet-ancestry-release-pcr-commitments-v2"),
    }
    for field, observed in checks.items():
        _require(projection[field] == observed, "proof %s differs" % field)
    _require(projection["parent_edge_count"] == len(edges), "proof parent edge count differs")


def validate_compact_ancestry_proof_v2(
    value: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
    allowed_issuer_roles: Iterable[str],
    required_receipt_hashes: Iterable[str] = (),
    required_purposes: Iterable[str] = (),
    trusted_parent_authorities: Optional[Mapping[str, str]] = None,
    known_certificate_hashes_by_root: Optional[Mapping[str, str]] = None,
    minimum_certificate_sequence: int = 0,
) -> Dict[str, Any]:
    """Validate a compact proof and its caller-owned anti-fork state."""

    proof = _exact(value, _PROOF_FIELDS, "compact ancestry proof")
    _require(proof.get("schema_version") == ANCESTRY_COMPACT_PROOF_SCHEMA_VERSION, "compact proof schema is invalid")
    certificate = validate_ancestry_certificate_v2(
        proof.get("certificate"),
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles=allowed_issuer_roles,
        trusted_parent_authorities=trusted_parent_authorities,
        known_certificate_hashes_by_root=known_certificate_hashes_by_root,
        minimum_certificate_sequence=minimum_certificate_sequence,
    )
    projection = certificate["claim"]["local_delta_projection"]
    receipts = proof.get("disclosed_receipts")
    boots = proof.get("disclosed_boot_identities")
    _require(isinstance(receipts, list), "compact proof receipts must be an array")
    _require(isinstance(boots, list), "compact proof boots must be an array")
    _require(len(receipts) <= MAX_DELTA_RECEIPTS and len(boots) <= MAX_DELTA_BOOTS, "compact proof disclosure exceeds bound")
    _projection_from_disclosures(
        projection=projection,
        receipts=receipts,
        boots=boots,
        parent_authorities=certificate["claim"]["parent_authorities"],
        policy=certificate["claim"]["policy"],
        certificate_sequence=certificate["claim"]["certificate_sequence"],
        boot_attestation_verifier=boot_attestation_verifier,
    )
    receipt_hashes = {str(item["receipt_hash"]) for item in receipts}
    required_hashes = set(_sorted_hashes(list(required_receipt_hashes), "required receipt hash", MAX_DELTA_RECEIPTS))
    _require(required_hashes.issubset(receipt_hashes), "compact proof is missing required receipt")
    observed_purposes = set(projection["ancestry_purposes"])
    required_purpose_set = set(_sorted_identifiers(list(required_purposes), "required purpose", MAX_REQUIRED_PURPOSES))
    _require(required_purpose_set.issubset(observed_purposes), "compact proof is missing required purpose")
    body = {field: proof[field] for field in _PROOF_FIELDS if field != "proof_hash"}
    _require(proof.get("proof_hash") == _proof_hash(body), "compact proof hash differs")
    return {
        "schema_version": ANCESTRY_COMPACT_PROOF_SCHEMA_VERSION,
        "certificate": certificate,
        "disclosed_receipts": [dict(item) for item in receipts],
        "disclosed_boot_identities": [dict(item) for item in boots],
        "proof_hash": str(proof["proof_hash"]),
    }


__all__ = [
    "ANCESTRY_CERTIFICATE_CLAIM_SCHEMA_VERSION",
    "ANCESTRY_CERTIFICATE_SCHEMA_VERSION",
    "ANCESTRY_COMPACT_PROOF_SCHEMA_VERSION",
    "ANCESTRY_DELTA_SCHEMA_VERSION",
    "ANCESTRY_FULL_GRAPH_PARENT_SCHEMA_VERSION",
    "ANCESTRY_PARENT_AUTHORITY_SCHEMA_VERSION",
    "ANCESTRY_POLICY_SCHEMA_VERSION",
    "ANCESTRY_PROJECTION_SCHEMA_VERSION",
    "AncestryCheckpointV2Error",
    "MAX_ALLOWED_FAILED_RECEIPTS",
    "MAX_DELTA_ATTEMPTS",
    "MAX_DELTA_BOOTS",
    "MAX_DELTA_EDGES",
    "MAX_DELTA_HOST_OPERATIONS",
    "MAX_DELTA_RECEIPTS",
    "MAX_FULL_GRAPH_ATTEMPTS",
    "MAX_FULL_GRAPH_BOOTS",
    "MAX_FULL_GRAPH_EDGES",
    "MAX_FULL_GRAPH_HOST_OPERATIONS",
    "MAX_FULL_GRAPH_RECEIPTS",
    "MAX_NETUID",
    "MAX_PARENT_AUTHORITIES",
    "MAX_REQUIRED_PURPOSES",
    "build_ancestry_policy_v2",
    "build_certificate_disclosure_parent_authority_v2",
    "build_certificate_parent_authority_v2",
    "build_compact_ancestry_proof_from_delta_v2",
    "build_compact_ancestry_proof_v2",
    "build_full_graph_parent_v2",
    "build_full_graph_parent_authority_v2",
    "derive_ancestry_lineage_id_v2",
    "issue_ancestry_certificate_v2",
    "project_receipt_graph_v2",
    "validate_ancestry_certificate_v2",
    "validate_ancestry_delta_v2",
    "validate_ancestry_lineage_id_v2",
    "validate_ancestry_parent_authority_v2",
    "validate_ancestry_policy_v2",
    "validate_ancestry_projection_v2",
    "validate_compact_ancestry_proof_v2",
    "validate_full_graph_parent_v2",
    "validate_local_delta_against_certificate_v2",
]
