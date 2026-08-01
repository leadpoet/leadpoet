"""Canonical bounded transport for authoritative V2 weight evidence.

The compact submission is a first-class public authority.  Its identity is
derived from the complete validator-signed compact envelope, so publication,
finalization, and independent auditor verification never need to reconstruct
historical receipt bodies.  Legacy expansion helpers remain available only for
pre-activation/N-1 compatibility and must not be used by activated paths.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import (
    build_checkpointed_receipt_graph,
    build_receipt_graph,
    merkle_root,
    sha256_json,
    validate_transport_attempt,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_certificate_disclosure_parent_authority_v2,
    build_compact_ancestry_proof_v2,
    validate_local_delta_against_certificate_v2,
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    build_published_weight_bundle_v2,
    build_weight_finalization_submission_v2,
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)


COMPACT_WEIGHT_SUBMISSION_SCHEMA_VERSION = (
    "leadpoet.compact_weight_submission.v2"
)
VALIDATOR_WEIGHT_RECEIPT_DELTA_SCHEMA_VERSION = (
    "leadpoet.validator_weight_receipt_delta.v2"
)
VALIDATOR_WEIGHT_FINALIZATION_DELTA_SCHEMA_VERSION = (
    "leadpoet.validator_weight_finalization_delta.v2"
)
WEIGHT_ANCESTRY_COMMITMENT_SCHEMA_VERSION = (
    "leadpoet.validator_weight_ancestry_commitment.v2"
)
COMPACT_WEIGHT_BUNDLE_COMMITMENT_SCHEMA_VERSION = (
    "leadpoet.compact_weight_bundle_commitment.v2"
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_COMPACT_BODY_FIELDS = {
    "schema_version",
    "validator_hotkey",
    "binding_message",
    "validator_hotkey_signature",
    "weight_snapshot",
    "weight_result",
    "weights_signature",
    "ancestry_commitment",
    "upstream_ancestry_proofs",
    "upstream_transport_attempts",
    "validator_receipt_delta",
    "binding_receipt",
    "validator_ancestry_proof",
    "epoch_authority",
    "epoch_boundary",
}
_COMPACT_FIELDS = _COMPACT_BODY_FIELDS | {"compact_submission_hash"}
_DELTA_FIELDS = {
    "schema_version",
    "root_receipt_hash",
    "boot_identities",
    "receipts",
    "transport_attempts",
    "host_operations",
}
_COMPACT_FINALIZATION_BODY_FIELDS = {
    "schema_version",
    "validator_hotkey",
    "weight_submission_event_hash",
    "finalization",
    "ancestry_commitment",
    "validator_receipt_delta",
    "validator_ancestry_proof",
}
_COMPACT_FINALIZATION_FIELDS = _COMPACT_FINALIZATION_BODY_FIELDS | {
    "compact_finalization_hash"
}
_EPOCH_SNAPSHOT_FIELDS = {
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


class CompactWeightAuthorityV2Error(ValueError):
    """A compact handoff is malformed, forked, or not exactly expandable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompactWeightAuthorityV2Error(message)


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _terminal_receipt_hashes(receipts: Iterable[Mapping[str, Any]]) -> list[str]:
    receipt_hashes = {str(item["receipt_hash"]) for item in receipts}
    parent_hashes = {
        str(parent)
        for receipt in receipts
        for parent in receipt.get("parent_receipt_hashes") or ()
    }
    return sorted(receipt_hashes - parent_hashes)


def compact_gateway_frontier_receipt_hashes_v2(
    proofs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return the terminal roots after recursively joining category proofs."""

    roots = set()
    referenced_roots = set()
    for category in sorted(proofs):
        proof = proofs[category]
        certificate = proof.get("certificate")
        claim = certificate.get("claim") if isinstance(certificate, Mapping) else None
        _require(isinstance(claim, Mapping), "compact proof claim is missing")
        roots.add(_hash(claim.get("output_root_receipt_hash"), "proof root"))
        parents = claim.get("parent_authorities")
        _require(isinstance(parents, list), "compact proof parents are invalid")
        for parent in parents:
            _require(isinstance(parent, Mapping), "compact proof parent is invalid")
            referenced_roots.add(
                _hash(parent.get("parent_receipt_hash"), "proof parent root")
            )
    frontier = sorted(roots - referenced_roots)
    _require(bool(frontier), "compact proof frontier is empty")
    return frontier


def build_weight_ancestry_commitment_v2(
    *,
    lineage_id: str,
    input_receipt_hashes: Mapping[str, Any],
    upstream_ancestry_proofs: Mapping[str, Mapping[str, Any]],
    upstream_transport_attempts: Sequence[Mapping[str, Any]],
) -> str:
    """Commit every input, certificate, proof, frontier, and direct attempt."""

    _require(
        set(input_receipt_hashes) == set(WEIGHT_INPUT_PURPOSES),
        "weight input receipt categories are incomplete",
    )
    _require(
        set(upstream_ancestry_proofs) == set(GATEWAY_WEIGHT_INPUT_CATEGORIES),
        "gateway compact proof categories are incomplete",
    )
    attempt_hashes = []
    seen_attempts = set()
    for attempt in upstream_transport_attempts:
        try:
            validate_transport_attempt(attempt)
        except Exception as exc:
            raise CompactWeightAuthorityV2Error(
                "gateway compact transport attempt is invalid"
            ) from exc
        attempt_hash = str(attempt["attempt_hash"])
        _require(
            attempt_hash not in seen_attempts,
            "gateway compact transport attempt is duplicated",
        )
        seen_attempts.add(attempt_hash)
        attempt_hashes.append(attempt_hash)
    proof_hashes = {}
    certificate_hashes = {}
    for category in sorted(upstream_ancestry_proofs):
        proof = upstream_ancestry_proofs[category]
        certificate = proof.get("certificate")
        _require(
            isinstance(certificate, Mapping),
            "gateway compact certificate is missing",
        )
        proof_hashes[category] = _hash(
            proof.get("proof_hash"), "compact proof hash"
        )
        certificate_hashes[category] = _hash(
            certificate.get("certificate_hash"),
            "compact certificate hash",
        )
    return sha256_json(
        {
            "schema_version": WEIGHT_ANCESTRY_COMMITMENT_SCHEMA_VERSION,
            "lineage_id": _hash(lineage_id, "ancestry lineage id"),
            "input_receipt_hashes": {
                category: _hash(
                    input_receipt_hashes[category],
                    "%s input receipt hash" % category,
                )
                for category in sorted(input_receipt_hashes)
            },
            "proof_hashes": proof_hashes,
            "certificate_hashes": certificate_hashes,
            "gateway_frontier_receipt_hashes": (
                compact_gateway_frontier_receipt_hashes_v2(
                    upstream_ancestry_proofs
                )
            ),
            "transport_attempt_hashes": sorted(attempt_hashes),
        }
    )


def validate_compact_weight_submission_shape_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the bounded envelope without trusting an external boot."""

    _require(isinstance(value, Mapping), "compact weight submission is invalid")
    _require(
        set(value) == _COMPACT_FIELDS,
        "compact weight submission fields are invalid",
    )
    _require(
        value.get("schema_version") == COMPACT_WEIGHT_SUBMISSION_SCHEMA_VERSION,
        "compact weight submission schema is invalid",
    )
    body = {field: value[field] for field in _COMPACT_BODY_FIELDS}
    _require(
        value.get("compact_submission_hash") == sha256_json(body),
        "compact weight submission hash differs",
    )
    hotkey = str(value.get("validator_hotkey") or "")
    _require(
        1 <= len(hotkey) <= 128 and not any(item.isspace() for item in hotkey),
        "compact validator hotkey is invalid",
    )
    _require(
        1 <= len(str(value.get("binding_message") or "")) <= 4096,
        "compact binding message is invalid",
    )
    for field in ("validator_hotkey_signature", "weights_signature"):
        _require(
            bool(_SIGNATURE_RE.fullmatch(str(value.get(field) or "").lower())),
            "compact %s is invalid" % field,
        )
    _hash(value.get("ancestry_commitment"), "ancestry commitment")
    _require(
        isinstance(value.get("weight_snapshot"), Mapping)
        and isinstance(value.get("weight_result"), Mapping),
        "compact weight documents are invalid",
    )
    proofs = value.get("upstream_ancestry_proofs")
    _require(
        isinstance(proofs, Mapping)
        and set(proofs) == set(GATEWAY_WEIGHT_INPUT_CATEGORIES),
        "compact proof categories are incomplete",
    )
    attempts = value.get("upstream_transport_attempts")
    _require(isinstance(attempts, list), "compact transport evidence is invalid")
    delta = value.get("validator_receipt_delta")
    _require(
        isinstance(delta, Mapping) and set(delta) == _DELTA_FIELDS,
        "validator receipt delta fields are invalid",
    )
    _require(
        delta.get("schema_version") == VALIDATOR_WEIGHT_RECEIPT_DELTA_SCHEMA_VERSION,
        "validator receipt delta schema is invalid",
    )
    for field in (
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    ):
        _require(isinstance(delta.get(field), list), "validator delta %s is invalid" % field)
    binding_receipt = value.get("binding_receipt")
    _require(isinstance(binding_receipt, Mapping), "binding receipt is missing")
    _require(
        binding_receipt.get("parent_receipt_hashes")
        == [_hash(delta.get("root_receipt_hash"), "validator delta root")],
        "binding receipt does not extend the validator delta",
    )
    _require(
        isinstance(value.get("validator_ancestry_proof"), Mapping),
        "validator ancestry proof is missing",
    )
    for field in ("epoch_authority", "epoch_boundary"):
        _require(
            value.get(field) is None or isinstance(value.get(field), Mapping),
            "compact %s is invalid" % field,
        )
    _require(
        (value.get("epoch_authority") is None)
        == (value.get("epoch_boundary") is None),
        "compact epoch evidence is incomplete",
    )
    return {field: value[field] for field in _COMPACT_FIELDS}


def compact_weight_bundle_hash_v2(value: Mapping[str, Any]) -> str:
    """Return the durable identity of one complete compact weight authority."""

    normalized = validate_compact_weight_submission_shape_v2(value)
    result = normalized["weight_result"]
    _require(isinstance(result, Mapping), "compact weight result is invalid")
    netuid = result.get("netuid")
    epoch_id = result.get("epoch_id")
    _require(
        isinstance(netuid, int)
        and not isinstance(netuid, bool)
        and netuid > 0,
        "compact weight netuid is invalid",
    )
    _require(
        isinstance(epoch_id, int)
        and not isinstance(epoch_id, bool)
        and epoch_id >= 0,
        "compact weight epoch is invalid",
    )
    weights_hash = str(result.get("weights_hash") or "").lower()
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", weights_hash)),
        "compact weights hash is invalid",
    )
    return sha256_json(
        {
            "schema_version": COMPACT_WEIGHT_BUNDLE_COMMITMENT_SCHEMA_VERSION,
            "compact_submission_hash": normalized["compact_submission_hash"],
            "validator_hotkey": normalized["validator_hotkey"],
            "netuid": netuid,
            "epoch_id": epoch_id,
            "weights_hash": weights_hash,
            "weight_receipt_hash": _hash(
                normalized["binding_receipt"].get("receipt_hash"),
                "compact weight receipt hash",
            ),
        }
    )


def build_checkpointed_weight_graph_from_compact_v2(
    value: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Materialize only the signed validator delta and its detached proof."""

    normalized = validate_compact_weight_ancestry_v2(
        value,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    delta = normalized["validator_receipt_delta"]
    binding = normalized["binding_receipt"]
    return build_checkpointed_receipt_graph(
        root_receipt_hash=str(binding["receipt_hash"]),
        boot_identities=delta["boot_identities"],
        receipts=[
            *[dict(item) for item in delta["receipts"]],
            dict(binding),
        ],
        transport_attempts=delta["transport_attempts"],
        host_operations=delta["host_operations"],
        ancestry_lineage_id=expected_lineage_id,
        ancestry_proof=normalized["validator_ancestry_proof"],
        boot_attestation_verifier=boot_attestation_verifier,
        require_boot_attestation_verification=True,
    )


def validate_compact_weight_ancestry_v2(
    value: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Verify detached checkpoint signatures and the exact direct disclosures."""

    normalized = validate_compact_weight_submission_shape_v2(value)
    snapshot = normalized["weight_snapshot"]
    input_hashes = snapshot.get("input_receipt_hashes")
    _require(
        isinstance(input_hashes, Mapping)
        and set(input_hashes) == set(WEIGHT_INPUT_PURPOSES),
        "compact snapshot inputs are incomplete",
    )
    proofs = {}
    direct_scopes = set()
    for category in sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES):
        expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
        direct_hash = _hash(input_hashes[category], "%s input receipt" % category)
        try:
            proof = validate_compact_ancestry_proof_v2(
                normalized["upstream_ancestry_proofs"][category],
                expected_lineage_id=expected_lineage_id,
                boot_attestation_verifier=boot_attestation_verifier,
                allowed_issuer_roles={"gateway_coordinator"},
                required_receipt_hashes={direct_hash},
                required_purposes={expected_purpose},
            )
        except Exception as exc:
            raise CompactWeightAuthorityV2Error(
                "compact ancestry failed for %s" % category
            ) from exc
        direct = [
            item
            for item in proof["disclosed_receipts"]
            if item.get("receipt_hash") == direct_hash
        ]
        _require(len(direct) == 1, "compact direct receipt is ambiguous")
        receipt = direct[0]
        _require(
            receipt.get("role") == expected_role
            and receipt.get("purpose") == expected_purpose
            and int(receipt.get("epoch_id", -1))
            == int(snapshot.get("epoch_id", -2)),
            "compact direct receipt differs for %s" % category,
        )
        direct_scopes.add((str(receipt["job_id"]), str(receipt["purpose"])))
        proofs[category] = proof
    attempts = []
    seen_attempts = set()
    for attempt in normalized["upstream_transport_attempts"]:
        try:
            validate_transport_attempt(attempt)
        except Exception as exc:
            raise CompactWeightAuthorityV2Error(
                "compact transport attempt is invalid"
            ) from exc
        _require(
            (str(attempt["job_id"]), str(attempt["purpose"])) in direct_scopes,
            "compact transport attempt is outside the direct input scopes",
        )
        attempt_hash = str(attempt["attempt_hash"])
        _require(attempt_hash not in seen_attempts, "compact transport attempt is duplicated")
        seen_attempts.add(attempt_hash)
        attempts.append(dict(attempt))
    expected_commitment = build_weight_ancestry_commitment_v2(
        lineage_id=expected_lineage_id,
        input_receipt_hashes=input_hashes,
        upstream_ancestry_proofs=proofs,
        upstream_transport_attempts=attempts,
    )
    _require(
        normalized["ancestry_commitment"] == expected_commitment,
        "compact ancestry commitment differs",
    )
    try:
        validator_proof = validate_compact_ancestry_proof_v2(
            normalized["validator_ancestry_proof"],
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles={"validator_weights"},
            required_receipt_hashes={
                str(normalized["binding_receipt"]["receipt_hash"])
            },
            required_purposes={"validator.hotkey_signature.v2"},
        )
    except Exception as exc:
        raise CompactWeightAuthorityV2Error(
            "validator ancestry proof failed closed"
        ) from exc
    _require(
        validator_proof["certificate"]["claim"][
            "output_root_receipt_hash"
        ]
        == normalized["binding_receipt"]["receipt_hash"],
        "validator ancestry proof root differs from binding receipt",
    )
    disclosed_binding = [
        item
        for item in validator_proof["disclosed_receipts"]
        if item.get("receipt_hash")
        == normalized["binding_receipt"].get("receipt_hash")
    ]
    _require(
        disclosed_binding == [dict(normalized["binding_receipt"])],
        "validator ancestry proof binding receipt differs",
    )
    # The detached proof authenticates its certificate, but the public compact
    # submission also carries the validator's local bodies separately so the
    # gateway can reconstruct the unchanged full graph.  Bind those exact
    # bodies to the signed certificate here: otherwise a service-role sidecar
    # mutation could retain a valid disclosed binding while dropping, adding,
    # or swapping another validator receipt/attempt.
    validator_delta = normalized["validator_receipt_delta"]
    certified_local_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": str(normalized["binding_receipt"]["receipt_hash"]),
        "boot_identities": [
            dict(item) for item in validator_delta["boot_identities"]
        ],
        "receipts": [
            *[dict(item) for item in validator_delta["receipts"]],
            dict(normalized["binding_receipt"]),
        ],
        "transport_attempts": [
            dict(item) for item in validator_delta["transport_attempts"]
        ],
        "host_operations": [
            dict(item) for item in validator_delta["host_operations"]
        ],
    }
    try:
        validate_local_delta_against_certificate_v2(
            certified_local_delta,
            validator_proof["certificate"],
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles={"validator_weights"},
        )
    except Exception as exc:
        raise CompactWeightAuthorityV2Error(
            "validator receipt delta differs from its ancestry certificate"
        ) from exc
    return {
        **normalized,
        "upstream_ancestry_proofs": proofs,
        "upstream_transport_attempts": attempts,
        "validator_ancestry_proof": validator_proof,
    }


def _merge_unique(
    items: Iterable[Mapping[str, Any]],
    *,
    key_field: Optional[str] = None,
    key: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    scope: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> list[Dict[str, Any]]:
    by_key = {}  # type: Dict[str, Dict[str, Any]]
    by_scope = {}
    _require(
        (key_field is None) != (key is None),
        "joined evidence key selector is invalid",
    )
    for item in items:
        normalized = dict(item)
        item_key = str(
            normalized.get(str(key_field))
            if key_field is not None
            else key(normalized)
        )
        _require(bool(item_key), "joined evidence key is missing")
        prior = by_key.get(item_key)
        _require(prior is None or prior == normalized, "joined evidence hash conflicts")
        by_key[item_key] = normalized
        if scope is not None:
            scope_key = scope(normalized)
            scoped_prior = by_scope.get(scope_key)
            _require(
                scoped_prior is None or scoped_prior == item_key,
                "joined evidence scope conflicts",
            )
            by_scope[scope_key] = item_key
    return [by_key[key] for key in sorted(by_key)]


def reconstruct_published_weight_bundle_from_compact_v2(
    value: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    full_graphs_by_root: Mapping[str, Mapping[str, Any]],
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Expand a bounded handoff into the byte-stable legacy public bundle."""

    normalized = validate_compact_weight_ancestry_v2(
        value,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    proofs = normalized["upstream_ancestry_proofs"]
    expected_roots = {
        str(proof["certificate"]["claim"]["output_root_receipt_hash"])
        for proof in proofs.values()
    }
    _require(
        set(full_graphs_by_root) == expected_roots,
        "durable full graph roots differ from compact checkpoints",
    )
    for category in sorted(proofs):
        proof = proofs[category]
        root = str(proof["certificate"]["claim"]["output_root_receipt_hash"])
        try:
            rebuilt = build_compact_ancestry_proof_v2(
                full_graphs_by_root[root],
                proof["certificate"],
                expected_lineage_id=expected_lineage_id,
                boot_attestation_verifier=boot_attestation_verifier,
                allowed_issuer_roles={"gateway_coordinator"},
            )
        except Exception as exc:
            raise CompactWeightAuthorityV2Error(
                "durable full graph does not satisfy %s checkpoint" % category
            ) from exc
        _require(rebuilt == proof, "durable full graph checkpoint differs")

    upstream_graphs = [full_graphs_by_root[root] for root in sorted(expected_roots)]
    direct_scopes = {
        (str(receipt["job_id"]), str(receipt["purpose"]))
        for proof in proofs.values()
        for receipt in proof["disclosed_receipts"]
        if str(receipt["receipt_hash"])
        in set(normalized["weight_snapshot"]["input_receipt_hashes"].values())
    }
    durable_direct_attempts = _merge_unique(
        (
            attempt
            for graph in upstream_graphs
            for attempt in graph.get("transport_attempts") or ()
            if (str(attempt["job_id"]), str(attempt["purpose"])) in direct_scopes
        ),
        key_field="attempt_hash",
    )
    _require(
        durable_direct_attempts
        == sorted(
            [dict(item) for item in normalized["upstream_transport_attempts"]],
            key=lambda item: str(item["attempt_hash"]),
        ),
        "durable direct transport evidence differs from compact handoff",
    )

    delta = normalized["validator_receipt_delta"]
    binding_receipt = dict(normalized["binding_receipt"])
    boots = _merge_unique(
        (
            identity
            for graph in upstream_graphs
            for identity in graph.get("boot_identities") or ()
        ),
        key_field="boot_identity_hash",
    )
    boots = _merge_unique(
        [*boots, *[dict(item) for item in delta["boot_identities"]]],
        key_field="boot_identity_hash",
    )
    receipts = _merge_unique(
        (
            receipt
            for graph in upstream_graphs
            for receipt in graph.get("receipts") or ()
        ),
        key_field="receipt_hash",
        scope=lambda item: (str(item["job_id"]), str(item["purpose"])),
    )
    receipts = _merge_unique(
        [
            *receipts,
            *[dict(item) for item in delta["receipts"]],
            binding_receipt,
        ],
        key_field="receipt_hash",
        scope=lambda item: (str(item["job_id"]), str(item["purpose"])),
    )
    attempts = _merge_unique(
        [
            *[
                dict(attempt)
                for graph in upstream_graphs
                for attempt in graph.get("transport_attempts") or ()
            ],
            *[dict(item) for item in delta["transport_attempts"]],
        ],
        key_field="attempt_hash",
    )
    hosts = _merge_unique(
        [
            *[
                dict(record)
                for graph in upstream_graphs
                for record in graph.get("host_operations") or ()
            ],
            *[dict(item) for item in delta["host_operations"]],
        ],
        key=lambda item: item["request"]["request_hash"],
        scope=lambda item: (
            str(item["request"]["job_id"]),
            str(item["request"]["purpose"]),
        ),
    )
    graph = build_receipt_graph(
        root_receipt_hash=str(binding_receipt["receipt_hash"]),
        boot_identities=boots,
        receipts=receipts,
        transport_attempts=attempts,
        host_operations=hosts,
    )
    try:
        rebuilt_validator_proof = build_compact_ancestry_proof_v2(
            graph,
            normalized["validator_ancestry_proof"]["certificate"],
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles={"validator_weights"},
        )
    except Exception as exc:
        raise CompactWeightAuthorityV2Error(
            "durable validator graph does not satisfy its checkpoint"
        ) from exc
    _require(
        rebuilt_validator_proof == normalized["validator_ancestry_proof"],
        "durable validator graph checkpoint differs",
    )
    snapshot_receipts = [
        receipt
        for receipt in delta["receipts"]
        if receipt.get("purpose") == "validator.weight_snapshot.v2"
    ]
    _require(len(snapshot_receipts) == 1, "validator snapshot receipt is ambiguous")
    expected_artifact_root = merkle_root(
        [
            str(normalized["weight_snapshot"]["calculation_snapshot_hash"]),
            str(normalized["ancestry_commitment"]),
        ],
        domain="leadpoet-validator-weight-artifact-v2",
    )
    _require(
        snapshot_receipts[0].get("artifact_root") == expected_artifact_root,
        "validator snapshot does not bind compact ancestry",
    )
    _require(
        snapshot_receipts[0].get("parent_receipt_hashes")
        == sorted(
            set(normalized["weight_snapshot"]["input_receipt_hashes"].values())
            | set(
                compact_gateway_frontier_receipt_hashes_v2(proofs)
            )
            | set(
                _terminal_receipt_hashes(
                    [
                        item
                        for item in delta["receipts"]
                        if item.get("purpose")
                        not in {
                            "validator.weight_snapshot.v2",
                            "validator.weights.computed.v2",
                        }
                    ]
                )
            )
        ),
        "validator snapshot frontier differs from compact authority",
    )
    bundle = build_published_weight_bundle_v2(
        validator_hotkey=str(normalized["validator_hotkey"]),
        binding_message=str(normalized["binding_message"]),
        validator_hotkey_signature=str(
            normalized["validator_hotkey_signature"]
        ),
        weight_snapshot=normalized["weight_snapshot"],
        weight_result=normalized["weight_result"],
        weights_signature=str(normalized["weights_signature"]),
        receipt_graph=graph,
    )
    validate_published_weight_bundle_v2(
        bundle,
        boot_attestation_verifier=boot_attestation_verifier,
        require_boot_attestation_verification=True,
    )
    return bundle


def build_stateful_epoch_evidence_from_compact_v2(
    value: Mapping[str, Any],
    *,
    bundle_hash: Optional[str] = None,
    published_bundle: Optional[Mapping[str, Any]] = None,
    expected_lineage_id: Optional[str] = None,
    boot_attestation_verifier: Optional[
        Callable[[Mapping[str, Any]], Any]
    ] = None,
) -> Optional[Dict[str, Any]]:
    """Build epoch evidence directly from the bounded validator delta.

    ``published_bundle`` is accepted only for N-1 callers.  Activated callers
    pass ``bundle_hash`` and never materialize a historical receipt graph.
    """

    normalized = validate_compact_weight_submission_shape_v2(value)
    current = normalized.get("epoch_authority")
    boundary = normalized.get("epoch_boundary")
    if current is None and boundary is None:
        return None
    _require(
        isinstance(current, Mapping) and isinstance(boundary, Mapping),
        "stateful epoch evidence is incomplete",
    )
    current_doc = dict(current)
    boundary_doc = dict(boundary)
    _require(
        set(current_doc) == _EPOCH_SNAPSHOT_FIELDS
        and set(boundary_doc) == _EPOCH_SNAPSHOT_FIELDS,
        "stateful epoch evidence fields are invalid",
    )
    for document in (current_doc, boundary_doc):
        _require(
            document.get("schema_version")
            == "leadpoet.subnet_epoch_snapshot.v1"
            and document.get("epoch_scheme")
            == "bittensor.subnet_epoch_index.v1"
            and document.get("head_kind") == "finalized"
            and int(document.get("epoch_id", -1))
            == int(document.get("subnet_epoch_index", -2))
            and int(document.get("settlement_epoch_id", -1)) >= 0,
            "stateful epoch identity is invalid",
        )
    identity_fields = (
        "epoch_scheme",
        "network_genesis_hash",
        "netuid",
        "subnet_epoch_index",
        "epoch_ref",
        "settlement_epoch_id",
        "cutover_mapping_hash",
    )
    _require(
        not any(
            current_doc[field] != boundary_doc[field]
            for field in identity_fields
        ),
        "stateful current and boundary identities differ",
    )
    _require(
        int(boundary_doc["current_block"])
        == int(boundary_doc["last_epoch_block"])
        and int(boundary_doc["epoch_block"]) == 0
        and int(current_doc["current_block"])
        >= int(boundary_doc["current_block"])
        and int(current_doc["epoch_block"])
        == int(current_doc["current_block"])
        - int(current_doc["last_epoch_block"]),
        "stateful epoch arithmetic is invalid",
    )
    computed = normalized["weight_result"]
    _require(isinstance(computed, Mapping), "compact weight result is invalid")
    _require(
        int(computed["netuid"]) == int(current_doc["netuid"])
        and int(computed["epoch_id"])
        == int(current_doc["settlement_epoch_id"])
        and int(computed["block"]) == int(current_doc["current_block"]),
        "stateful epoch evidence differs from the weight bundle",
    )
    if published_bundle is not None:
        verified = validate_published_weight_bundle_v2(published_bundle)
        resolved_bundle_hash = str(verified["bundle_hash"])
        evidence_graph = dict(published_bundle["receipt_graph"])
    else:
        resolved_bundle_hash = _hash(
            bundle_hash or compact_weight_bundle_hash_v2(normalized),
            "compact bundle hash",
        )
        _require(
            resolved_bundle_hash == compact_weight_bundle_hash_v2(normalized),
            "compact bundle hash differs",
        )
        _require(
            isinstance(expected_lineage_id, str)
            and callable(boot_attestation_verifier),
            "compact epoch evidence requires verified ancestry",
        )
        evidence_graph = build_checkpointed_weight_graph_from_compact_v2(
            normalized,
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
        )
    current_hash = sha256_json(current_doc)
    boundary_hash = sha256_json(boundary_doc)
    if published_bundle is not None:
        receipt_values = published_bundle["receipt_graph"]["receipts"]
    else:
        receipt_values = normalized["validator_receipt_delta"]["receipts"]
    candidates = [
        receipt
        for receipt in receipt_values
        if receipt.get("role") == "validator_weights"
        and receipt.get("purpose") == "validator.subnet_epoch_snapshot.v2"
        and receipt.get("output_root") in {current_hash, boundary_hash}
        and int(receipt.get("epoch_id", -1)) == int(computed["epoch_id"])
    ]
    if current_hash == boundary_hash:
        _require(
            len(candidates) == 2,
            "stateful cutover receipts are missing or ambiguous",
        )
        boundary_receipt, current_receipt = sorted(
            candidates,
            key=lambda item: len(item["parent_receipt_hashes"]),
        )
    else:
        current_receipts = [
            item for item in candidates if item.get("output_root") == current_hash
        ]
        boundary_receipts = [
            item for item in candidates if item.get("output_root") == boundary_hash
        ]
        _require(
            len(current_receipts) == 1 and len(boundary_receipts) == 1,
            "stateful epoch receipts are missing or ambiguous",
        )
        current_receipt = current_receipts[0]
        boundary_receipt = boundary_receipts[0]
    _require(
        current_receipt.get("parent_receipt_hashes")
        == [boundary_receipt["receipt_hash"]],
        "stateful current receipt does not bind its boundary",
    )
    return {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": str(normalized["validator_hotkey"]),
        "bundle_hash": resolved_bundle_hash,
        "cutover_mapping_hash": str(current_doc["cutover_mapping_hash"]),
        "epoch_authority": current_doc,
        "epoch_authority_hash": current_hash,
        "epoch_authority_receipt_hash": str(current_receipt["receipt_hash"]),
        "epoch_boundary": boundary_doc,
        "epoch_boundary_hash": boundary_hash,
        "epoch_boundary_receipt_hash": str(boundary_receipt["receipt_hash"]),
        "receipt_graph": evidence_graph,
    }


def build_compact_weight_finalization_submission_v2(
    *,
    validator_hotkey: str,
    weight_submission_event_hash: str,
    finalization: Mapping[str, Any],
    ancestry_commitment: str,
    validator_receipt_delta: Mapping[str, Any],
    validator_ancestry_proof: Mapping[str, Any],
) -> Dict[str, Any]:
    body = {
        "schema_version": "leadpoet.compact_weight_finalization_submission.v2",
        "validator_hotkey": str(validator_hotkey),
        "weight_submission_event_hash": _hash(
            weight_submission_event_hash,
            "weight submission event hash",
        ),
        "finalization": dict(finalization),
        "ancestry_commitment": _hash(
            ancestry_commitment,
            "ancestry commitment",
        ),
        "validator_receipt_delta": dict(validator_receipt_delta),
        "validator_ancestry_proof": dict(validator_ancestry_proof),
    }
    value = {**body, "compact_finalization_hash": sha256_json(body)}
    validate_compact_weight_finalization_shape_v2(value)
    return value


def validate_compact_weight_finalization_shape_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == _COMPACT_FINALIZATION_FIELDS,
        "compact weight finalization fields are invalid",
    )
    _require(
        value.get("schema_version")
        == "leadpoet.compact_weight_finalization_submission.v2",
        "compact weight finalization schema is invalid",
    )
    body = {
        field: value[field] for field in _COMPACT_FINALIZATION_BODY_FIELDS
    }
    _require(
        value.get("compact_finalization_hash") == sha256_json(body),
        "compact weight finalization hash differs",
    )
    _hash(value.get("weight_submission_event_hash"), "weight submission event hash")
    _hash(value.get("ancestry_commitment"), "ancestry commitment")
    _require(
        isinstance(value.get("finalization"), Mapping),
        "compact finalization document is invalid",
    )
    delta = value.get("validator_receipt_delta")
    _require(
        isinstance(delta, Mapping)
        and set(delta) == _DELTA_FIELDS
        and delta.get("schema_version")
        == VALIDATOR_WEIGHT_FINALIZATION_DELTA_SCHEMA_VERSION,
        "compact finalization receipt delta is invalid",
    )
    _require(
        isinstance(value.get("validator_ancestry_proof"), Mapping),
        "compact finalization ancestry proof is missing",
    )
    return {field: value[field] for field in _COMPACT_FINALIZATION_FIELDS}


def validate_compact_weight_finalization_ancestry_v2(
    value: Mapping[str, Any],
    *,
    compact_weight_submission: Mapping[str, Any],
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Verify the finalization delta extends the published computed receipt."""

    normalized = validate_compact_weight_finalization_shape_v2(value)
    published = validate_compact_weight_ancestry_v2(
        compact_weight_submission,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    computed_root = str(
        published["validator_receipt_delta"]["root_receipt_hash"]
    )
    expected_parent = build_certificate_disclosure_parent_authority_v2(
        published["validator_ancestry_proof"],
        computed_root,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
        allowed_issuer_roles={"validator_weights"},
    )
    try:
        proof = validate_compact_ancestry_proof_v2(
            normalized["validator_ancestry_proof"],
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles={"validator_weights"},
            required_receipt_hashes={
                str(normalized["validator_receipt_delta"]["root_receipt_hash"])
            },
            required_purposes={
                "validator.set_weights_extrinsic.v2",
                "validator.weights.finalized.v2",
            },
        )
    except Exception as exc:
        raise CompactWeightAuthorityV2Error(
            "validator finalization ancestry proof failed closed"
        ) from exc
    parents = proof["certificate"]["claim"]["parent_authorities"]
    _require(
        parents == [expected_parent],
        "validator finalization ancestry parent differs",
    )
    delta = normalized["validator_receipt_delta"]
    local_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": delta["root_receipt_hash"],
        "boot_identities": [dict(item) for item in delta["boot_identities"]],
        "receipts": [dict(item) for item in delta["receipts"]],
        "transport_attempts": [
            dict(item) for item in delta["transport_attempts"]
        ],
        "host_operations": [dict(item) for item in delta["host_operations"]],
    }
    try:
        validate_local_delta_against_certificate_v2(
            local_delta,
            proof["certificate"],
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles={"validator_weights"},
        )
    except Exception as exc:
        raise CompactWeightAuthorityV2Error(
            "validator finalization delta differs from its ancestry certificate"
        ) from exc
    return {
        **normalized,
        "validator_ancestry_proof": proof,
        "compact_weight_submission": published,
    }


def build_checkpointed_weight_finalization_graph_v2(
    value: Mapping[str, Any],
    *,
    compact_weight_submission: Mapping[str, Any],
    expected_lineage_id: str,
    boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Materialize one bounded, signed finalization graph."""

    normalized = validate_compact_weight_finalization_ancestry_v2(
        value,
        compact_weight_submission=compact_weight_submission,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=boot_attestation_verifier,
    )
    delta = normalized["validator_receipt_delta"]
    return build_checkpointed_receipt_graph(
        root_receipt_hash=str(delta["root_receipt_hash"]),
        boot_identities=delta["boot_identities"],
        receipts=delta["receipts"],
        transport_attempts=delta["transport_attempts"],
        host_operations=delta["host_operations"],
        ancestry_lineage_id=expected_lineage_id,
        ancestry_proof=normalized["validator_ancestry_proof"],
        boot_attestation_verifier=boot_attestation_verifier,
        require_boot_attestation_verification=True,
    )


def reconstruct_weight_finalization_from_compact_v2(
    value: Mapping[str, Any],
    *,
    published_bundle: Mapping[str, Any],
    chain_signing_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Expand a finalization delta against its unchanged durable bundle."""

    normalized = validate_compact_weight_finalization_shape_v2(value)
    bundle_verified = validate_published_weight_bundle_v2(published_bundle)
    _require(
        normalized["validator_hotkey"] == bundle_verified["validator_hotkey"],
        "compact finalization validator differs",
    )
    delta = normalized["validator_receipt_delta"]
    graph = published_bundle["receipt_graph"]
    binding_root = str(graph["root_receipt_hash"])
    base_receipts = [
        dict(item)
        for item in graph["receipts"]
        if str(item["receipt_hash"]) != binding_root
    ]
    receipts = _merge_unique(
        [*base_receipts, *[dict(item) for item in delta["receipts"]]],
        key_field="receipt_hash",
        scope=lambda item: (str(item["job_id"]), str(item["purpose"])),
    )
    boots = _merge_unique(
        [
            *[dict(item) for item in graph["boot_identities"]],
            *[dict(item) for item in delta["boot_identities"]],
        ],
        key_field="boot_identity_hash",
    )
    attempts = _merge_unique(
        [
            *[dict(item) for item in graph["transport_attempts"]],
            *[dict(item) for item in delta["transport_attempts"]],
        ],
        key_field="attempt_hash",
    )
    hosts = _merge_unique(
        [
            *[dict(item) for item in graph["host_operations"]],
            *[dict(item) for item in delta["host_operations"]],
        ],
        key=lambda item: item["request"]["request_hash"],
        scope=lambda item: (
            str(item["request"]["job_id"]),
            str(item["request"]["purpose"]),
        ),
    )
    expanded_graph = build_receipt_graph(
        root_receipt_hash=str(delta["root_receipt_hash"]),
        boot_identities=boots,
        receipts=receipts,
        transport_attempts=attempts,
        host_operations=hosts,
    )
    submission = build_weight_finalization_submission_v2(
        validator_hotkey=str(normalized["validator_hotkey"]),
        weight_submission_event_hash=str(
            normalized["weight_submission_event_hash"]
        ),
        finalization=normalized["finalization"],
        receipt_graph=expanded_graph,
        chain_signing_profile=chain_signing_profile,
    )
    verified = validate_weight_finalization_submission_v2(
        submission,
        chain_signing_profile=chain_signing_profile,
    )
    _require(
        verified["netuid"] == bundle_verified["netuid"]
        and verified["epoch_id"] == bundle_verified["epoch_id"]
        and verified["weights_hash"] == bundle_verified["weights_hash"]
        and verified["weight_receipt_hash"]
        == bundle_verified["weight_receipt_hash"],
        "compact finalization differs from the durable bundle",
    )
    snapshot_receipts = [
        item
        for item in graph["receipts"]
        if item.get("receipt_hash") == bundle_verified["snapshot_receipt_hash"]
    ]
    _require(len(snapshot_receipts) == 1, "bundle snapshot receipt is unavailable")
    expected_artifact_root = merkle_root(
        [
            str(published_bundle["weight_snapshot"]["calculation_snapshot_hash"]),
            str(normalized["ancestry_commitment"]),
        ],
        domain="leadpoet-validator-weight-artifact-v2",
    )
    _require(
        snapshot_receipts[0].get("artifact_root") == expected_artifact_root,
        "compact finalization ancestry commitment differs",
    )
    return submission


__all__ = [
    "COMPACT_WEIGHT_BUNDLE_COMMITMENT_SCHEMA_VERSION",
    "COMPACT_WEIGHT_SUBMISSION_SCHEMA_VERSION",
    "CompactWeightAuthorityV2Error",
    "VALIDATOR_WEIGHT_FINALIZATION_DELTA_SCHEMA_VERSION",
    "VALIDATOR_WEIGHT_RECEIPT_DELTA_SCHEMA_VERSION",
    "WEIGHT_ANCESTRY_COMMITMENT_SCHEMA_VERSION",
    "build_weight_ancestry_commitment_v2",
    "build_checkpointed_weight_finalization_graph_v2",
    "build_checkpointed_weight_graph_from_compact_v2",
    "compact_weight_bundle_hash_v2",
    "build_stateful_epoch_evidence_from_compact_v2",
    "build_compact_weight_finalization_submission_v2",
    "compact_gateway_frontier_receipt_hashes_v2",
    "reconstruct_published_weight_bundle_from_compact_v2",
    "reconstruct_weight_finalization_from_compact_v2",
    "validate_compact_weight_ancestry_v2",
    "validate_compact_weight_submission_shape_v2",
    "validate_compact_weight_finalization_shape_v2",
    "validate_compact_weight_finalization_ancestry_v2",
]
