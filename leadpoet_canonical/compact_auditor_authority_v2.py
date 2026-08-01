"""Independent bounded verification of published V2 weight authority.

The compact authority is sufficient on its own: validator, gateway, and
auditor verify the same signed local deltas and recursively authenticated
checkpoints without downloading or reconstructing historical receipt bodies.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.ancestry_checkpoint_v2 import (
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    merkle_root,
    sha256_json,
    validate_signed_execution_receipt,
    validate_transport_attempt,
    verify_boot_identity_nitro,
)
from leadpoet_canonical.auditor_v2 import _identity_for_boot
from leadpoet_canonical.binding import parse_binding_message, verify_binding_message
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PORT,
)
from leadpoet_canonical.compact_weight_authority_v2 import (
    compact_weight_bundle_hash_v2,
    validate_compact_weight_ancestry_v2,
    validate_compact_weight_finalization_ancestry_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
    validate_weight_extrinsic_authorization_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    VALIDATOR_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    validate_weight_input_source_evidence_v2,
    validate_weight_snapshot_v2,
    weight_input_value_documents_v2,
)


COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION = (
    "leadpoet.compact_published_weight_authority.v2"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_FIELDS = {
    "schema_version",
    "authority_stage",
    "lineage_id",
    "bundle_hash",
    "compact_submission",
    "publication",
    "finalization",
    "authority_hash",
}
_PUBLICATION_FIELDS = {
    "weight_submission_event_hash",
    "publication_receipt_hash",
    "publication_doc",
    "ancestry_proof",
}
_FINALIZATION_FIELDS = {
    "weight_finalization_event_hash",
    "compact_submission",
}


class CompactAuditorAuthorityV2Error(ValueError):
    """A compact public authority is incomplete, forked, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompactAuditorAuthorityV2Error(message)


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def build_compact_published_weight_authority_v2(
    *,
    authority_stage: str,
    lineage_id: str,
    bundle_hash: str,
    compact_submission: Mapping[str, Any],
    publication: Mapping[str, Any],
    finalization: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build one deterministic, independently verifiable public authority."""

    expected_bundle_hash = compact_weight_bundle_hash_v2(compact_submission)
    _require(
        _hash(bundle_hash, "bundle hash") == expected_bundle_hash,
        "compact public bundle hash differs",
    )

    body = {
        "schema_version": COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION,
        "authority_stage": str(authority_stage),
        "lineage_id": _hash(lineage_id, "lineage id"),
        "bundle_hash": _hash(bundle_hash, "bundle hash"),
        "compact_submission": dict(compact_submission),
        "publication": dict(publication),
        "finalization": dict(finalization) if finalization is not None else None,
    }
    value = {**body, "authority_hash": sha256_json(body)}
    validate_compact_published_weight_authority_shape_v2(value)
    return value


def validate_compact_published_weight_authority_shape_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == _FIELDS, "compact public authority fields are invalid")
    _require(value.get("schema_version") == COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION, "compact public authority schema is invalid")
    stage = str(value.get("authority_stage") or "")
    _require(stage in {"published", "finalized"}, "compact public authority stage is invalid")
    _hash(value.get("lineage_id"), "lineage id")
    _hash(value.get("bundle_hash"), "bundle hash")
    _require(isinstance(value.get("compact_submission"), Mapping), "compact weight submission is missing")
    publication = value.get("publication")
    _require(isinstance(publication, Mapping) and set(publication) == _PUBLICATION_FIELDS, "compact publication fields are invalid")
    _require(isinstance(publication.get("publication_doc"), Mapping), "compact publication document is invalid")
    _require(isinstance(publication.get("ancestry_proof"), Mapping), "compact publication ancestry proof is invalid")
    finalization = value.get("finalization")
    if stage == "published":
        _require(finalization is None, "published compact authority carries finalization")
    else:
        _require(isinstance(finalization, Mapping) and set(finalization) == _FINALIZATION_FIELDS, "compact finalization fields are invalid")
    body = {field: value[field] for field in _FIELDS if field != "authority_hash"}
    _require(value.get("authority_hash") == sha256_json(body), "compact public authority hash differs")
    return {field: value[field] for field in _FIELDS}


def _verify_boot_for_environment(
    boot: Mapping[str, Any],
    *,
    identity_cache: Optional[Mapping[str, Any]],
    boot_verifier: Callable[..., Any],
) -> Any:
    if identity_cache is None:
        return boot_verifier(boot)
    identity = _identity_for_boot(identity_cache, boot)
    return boot_verifier(boot, expected_pcr0=identity["pcr0"])


def _validate_receipt_delta(
    delta: Mapping[str, Any],
    *,
    expected_schema: str,
    identity_cache: Optional[Mapping[str, Any]],
    boot_verifier: Callable[..., Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Mapping[str, Any]], list[Dict[str, Any]]]:
    fields = {"schema_version", "root_receipt_hash", "boot_identities", "receipts", "transport_attempts", "host_operations"}
    _require(isinstance(delta, Mapping) and set(delta) == fields and delta.get("schema_version") == expected_schema, "validator receipt delta is invalid")
    _require(delta.get("host_operations") == [], "validator receipt delta has host operations")
    boots = {}
    for raw_boot in delta.get("boot_identities") or ():
        _require(isinstance(raw_boot, Mapping), "validator delta boot is invalid")
        boot = dict(raw_boot)
        _verify_boot_for_environment(
            boot,
            identity_cache=identity_cache,
            boot_verifier=boot_verifier,
        )
        boot_hash = str(boot.get("boot_identity_hash") or "")
        _require(boot_hash not in boots, "validator delta boot is duplicated")
        boots[boot_hash] = boot
    _require(bool(boots), "validator delta boot is missing")
    receipts = {}
    scopes = set()
    for raw_receipt in delta.get("receipts") or ():
        _require(isinstance(raw_receipt, Mapping), "validator delta receipt is invalid")
        validate_signed_execution_receipt(raw_receipt)
        receipt = dict(raw_receipt)
        receipt_hash = str(receipt["receipt_hash"])
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        _require(receipt_hash not in receipts and scope not in scopes, "validator delta receipt is duplicated")
        scopes.add(scope)
        boot = boots.get(str(receipt.get("boot_identity_hash") or ""))
        _require(isinstance(boot, Mapping), "validator delta receipt boot is missing")
        for receipt_field, boot_field in (
            ("role", "role"), ("commit_sha", "commit_sha"), ("pcr0", "pcr0"),
            ("build_manifest_hash", "build_manifest_hash"),
            ("dependency_lock_hash", "dependency_lock_hash"),
            ("config_hash", "config_hash"), ("enclave_pubkey", "signing_pubkey"),
        ):
            _require(receipt.get(receipt_field) == boot.get(boot_field), "validator delta receipt differs from boot")
        _require(receipt.get("host_operation_root") == EMPTY_HOST_OPERATION_ROOT, "validator delta receipt host root differs")
        receipts[receipt_hash] = receipt
    attempts = []
    attempts_by_scope: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    seen_attempts = set()
    for raw_attempt in delta.get("transport_attempts") or ():
        validate_transport_attempt(raw_attempt)
        attempt = dict(raw_attempt)
        attempt_hash = str(attempt["attempt_hash"])
        _require(attempt_hash not in seen_attempts, "validator delta attempt is duplicated")
        seen_attempts.add(attempt_hash)
        attempts.append(attempt)
        attempts_by_scope.setdefault((str(attempt["job_id"]), str(attempt["purpose"])), []).append(attempt)
    for receipt in receipts.values():
        scoped = attempts_by_scope.pop((str(receipt["job_id"]), str(receipt["purpose"])), [])
        root = merkle_root([str(item["attempt_hash"]) for item in scoped], domain="leadpoet-transport-v2") if scoped else EMPTY_TRANSPORT_ROOT
        _require(receipt.get("transport_root") == root, "validator delta transport root differs")
    _require(not attempts_by_scope, "validator delta has unclaimed transport")
    return receipts, boots, attempts


def verify_compact_weight_submission_v2(
    compact: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    expected_chain: str,
    identity_cache: Optional[Mapping[str, Any]],
    boot_verifier: Callable[..., Any],
) -> Dict[str, Any]:
    def verify_boot(boot: Mapping[str, Any]) -> Any:
        return _verify_boot_for_environment(
            boot,
            identity_cache=identity_cache,
            boot_verifier=boot_verifier,
        )

    normalized = validate_compact_weight_ancestry_v2(
        compact,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=verify_boot,
    )
    computed = validate_weight_snapshot_v2(normalized["weight_snapshot"])
    _require(dict(normalized["weight_result"]) == computed, "compact weight result is not canonical")
    snapshot = normalized["weight_snapshot"]
    receipts, boots, attempts = _validate_receipt_delta(
        normalized["validator_receipt_delta"],
        expected_schema="leadpoet.validator_weight_receipt_delta.v2",
        identity_cache=identity_cache,
        boot_verifier=boot_verifier,
    )
    inputs = dict(snapshot["input_receipt_hashes"])
    documents = weight_input_value_documents_v2(
        calculation_snapshot=snapshot["calculation_snapshot"],
        finalized_chain_state_root=snapshot["finalized_chain_state_root"],
        gateway_authority_event_hash=snapshot["gateway_authority_event_hash"],
    )
    all_attempts = [*normalized["upstream_transport_attempts"], *attempts]
    for category, receipt_hash in inputs.items():
        receipt = receipts.get(str(receipt_hash))
        if receipt is None:
            proof = normalized["upstream_ancestry_proofs"].get(category)
            disclosed = [item for item in proof.get("disclosed_receipts", []) if item.get("receipt_hash") == receipt_hash] if isinstance(proof, Mapping) else []
            _require(len(disclosed) == 1, "%s compact input receipt is missing" % category)
            receipt = dict(disclosed[0])
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        _require(receipt.get("role") == role and receipt.get("purpose") == purpose, "%s compact input scope differs" % category)
        _require(int(receipt.get("epoch_id", -1)) == int(computed["epoch_id"]), "%s compact input epoch differs" % category)
        _require(receipt.get("output_root") == sha256_json(documents[category]), "%s compact input value differs" % category)
        validate_weight_input_source_evidence_v2(category=category, receipt=receipt, document=documents[category], transport_attempts=all_attempts)
    root_hash = str(normalized["validator_receipt_delta"]["root_receipt_hash"])
    computed_receipt = receipts.get(root_hash)
    _require(isinstance(computed_receipt, Mapping) and computed_receipt.get("purpose") == "validator.weights.computed.v2" and computed_receipt.get("output_root") == sha256_json(computed), "compact computed receipt differs")
    snapshot_parents = list(computed_receipt.get("parent_receipt_hashes") or [])
    _require(len(snapshot_parents) == 1, "compact computed receipt parent differs")
    snapshot_receipt = receipts.get(str(snapshot_parents[0]))
    _require(isinstance(snapshot_receipt, Mapping) and snapshot_receipt.get("purpose") == "validator.weight_snapshot.v2" and snapshot_receipt.get("input_root") == snapshot["source_input_root"] and snapshot_receipt.get("output_root") == snapshot["snapshot_hash"], "compact snapshot receipt differs")
    expected_artifact_root = merkle_root([snapshot["calculation_snapshot_hash"], normalized["ancestry_commitment"]], domain="leadpoet-validator-weight-artifact-v2")
    _require(snapshot_receipt.get("artifact_root") == expected_artifact_root, "compact snapshot ancestry commitment differs")
    binding = dict(normalized["binding_receipt"])
    validate_signed_execution_receipt(binding)
    _require(binding.get("parent_receipt_hashes") == [root_hash] and binding.get("purpose") == "validator.hotkey_signature.v2", "compact binding receipt differs")
    parsed, binding_fields, _error = parse_binding_message(str(normalized["binding_message"]))
    _require(parsed and isinstance(binding_fields, Mapping), "compact binding message is invalid")
    boot = boots.get(str(computed_receipt.get("boot_identity_hash") or ""))
    _require(isinstance(boot, Mapping), "compact computing boot is missing")
    _require(verify_binding_message(str(normalized["binding_message"]), str(normalized["validator_hotkey_signature"]), str(normalized["validator_hotkey"]), expected_netuid=int(computed["netuid"]), expected_chain=str(expected_chain), expected_enclave_pubkey=str(computed_receipt["enclave_pubkey"]), expected_code_hash=str(boot["build_manifest_hash"])), "compact validator hotkey signature is invalid")
    application_request = build_application_signature_request_v2(message=str(normalized["binding_message"]).encode("utf-8"), validator_hotkey=str(normalized["validator_hotkey"]), boot_identity_hash=str(boot["boot_identity_hash"]))
    expected_binding_output = {"schema_version": "leadpoet.application_signature_result.v2", "request_hash": application_request["request_hash"], "purpose": "validator.gateway_binding.v2", "validator_hotkey": str(normalized["validator_hotkey"]), "signature": str(normalized["validator_hotkey_signature"])}
    _require(binding.get("input_root") == application_request["request_hash"] and binding.get("output_root") == sha256_json(expected_binding_output), "compact binding receipt output differs")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(computed_receipt["enclave_pubkey"]))).verify(bytes.fromhex(str(normalized["weights_signature"])), bytes.fromhex(str(computed["weights_hash"])))
    except Exception as exc:
        raise CompactAuditorAuthorityV2Error("compact weight signature is invalid") from exc
    return {
        "validator_hotkey": str(normalized["validator_hotkey"]),
        "netuid": int(computed["netuid"]), "epoch_id": int(computed["epoch_id"]),
        "block": int(computed["block"]), "uids": list(computed["sparse_uids"]),
        "weights_u16": list(computed["sparse_weights_u16"]),
        "weights_hash": str(computed["weights_hash"]),
        "root_receipt_hash": str(binding["receipt_hash"]),
        "weight_receipt_hash": root_hash,
        "snapshot_hash": str(snapshot["snapshot_hash"]),
        "validator_enclave_pubkey": str(computed_receipt["enclave_pubkey"]),
        "validator_boot_identity_hash": str(boot["boot_identity_hash"]),
        "compact_submission_hash": str(normalized["compact_submission_hash"]),
        "bundle_hash": compact_weight_bundle_hash_v2(normalized),
    }


def _verify_compact_finalization(
    section: Mapping[str, Any],
    *,
    verified: Mapping[str, Any],
    publication_event_hash: str,
    compact_weight_submission: Mapping[str, Any],
    expected_lineage_id: str,
    identity_cache: Optional[Mapping[str, Any]],
    chain_signing_profile: Optional[Mapping[str, Any]],
    boot_verifier: Callable[..., Any],
) -> Dict[str, Any]:
    def verify_boot(boot: Mapping[str, Any]) -> Any:
        return _verify_boot_for_environment(
            boot,
            identity_cache=identity_cache,
            boot_verifier=boot_verifier,
        )

    compact = validate_compact_weight_finalization_ancestry_v2(
        section["compact_submission"],
        compact_weight_submission=compact_weight_submission,
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=verify_boot,
    )
    _require(compact["validator_hotkey"] == verified["validator_hotkey"] and compact["weight_submission_event_hash"] == publication_event_hash, "compact finalization identity differs")
    _require(compact["ancestry_commitment"] == verified["ancestry_commitment"], "compact finalization ancestry differs")
    receipts, _boots, attempts = _validate_receipt_delta(compact["validator_receipt_delta"], expected_schema="leadpoet.validator_weight_finalization_delta.v2", identity_cache=identity_cache, boot_verifier=boot_verifier)
    finalization = dict(compact["finalization"])
    authorization = finalization.get("extrinsic_authorization")
    _require(isinstance(authorization, Mapping), "compact extrinsic authorization is invalid")
    _require(
        authorization.get("authorization_hash")
        == sha256_json(
            {
                key: authorization[key]
                for key in authorization
                if key != "authorization_hash"
            }
        ),
        "compact extrinsic authorization hash differs",
    )
    authorization_verified = (
        validate_weight_extrinsic_authorization_v2(
            authorization, profile=chain_signing_profile
        )
        if chain_signing_profile is not None
        else dict(authorization)
    )
    for field, expected in (("validator_hotkey", verified["validator_hotkey"]), ("netuid", verified["netuid"]), ("epoch_id", verified["epoch_id"]), ("weights_hash", verified["weights_hash"]), ("weight_receipt_hash", verified["weight_receipt_hash"]), ("weight_submission_event_hash", publication_event_hash)):
        _require(finalization.get(field) == expected and authorization_verified.get(field) == expected, "compact finalization differs at %s" % field)
    root_hash = str(compact["validator_receipt_delta"]["root_receipt_hash"])
    root = receipts.get(root_hash)
    extrinsic_hash = str(finalization.get("extrinsic_receipt_hash") or "")
    extrinsic = receipts.get(extrinsic_hash)
    _require(isinstance(root, Mapping) and root.get("purpose") == "validator.weights.finalized.v2" and root.get("output_root") == sha256_json(finalization), "compact finalization receipt differs")
    expected_extrinsic_output = {"schema_version": "leadpoet.weight_extrinsic_signature.v2", "authorization_hash": finalization.get("extrinsic_authorization_hash"), "validator_hotkey": verified["validator_hotkey"], "signature": finalization.get("extrinsic_signature"), "extrinsic_hash": finalization.get("extrinsic_hash")}
    _require(isinstance(extrinsic, Mapping) and extrinsic.get("purpose") == "validator.set_weights_extrinsic.v2" and extrinsic.get("parent_receipt_hashes") == [verified["weight_receipt_hash"]] and extrinsic.get("input_root") == finalization.get("extrinsic_authorization_hash") and extrinsic.get("output_root") == sha256_json(expected_extrinsic_output), "compact extrinsic receipt differs")
    _require(extrinsic_hash in list(root.get("parent_receipt_hashes") or []), "compact finalization omits included extrinsic")
    scoped = [item for item in attempts if item.get("job_id") == root.get("job_id") and item.get("purpose") == "validator.weights.finalized.v2"]
    _require(bool(scoped) and all((item.get("provider_id"), item.get("destination_host")) in {("bittensor_chain", CHAIN_ENDPOINT_HOST), ("bittensor_archive", CHAIN_ARCHIVE_ENDPOINT_HOST)} and item.get("destination_port") == CHAIN_ENDPOINT_PORT and item.get("terminal_status") == "authenticated_response" for item in scoped), "compact finalization chain evidence is invalid")
    _require(any(item.get("provider_id") == "bittensor_chain" and item.get("destination_host") == CHAIN_ENDPOINT_HOST for item in scoped), "compact finalization live-chain evidence is missing")
    event_hash = sha256_json({"weight_submission_event_hash": publication_event_hash, "bundle_hash": verified["bundle_hash"], "finalization_receipt_hash": root_hash, "extrinsic_authorization_hash": finalization["extrinsic_authorization_hash"], "extrinsic_hash": finalization["extrinsic_hash"], "finalized_block": finalization["finalized_block"], "finalized_block_hash": finalization["finalized_block_hash"], "state_transition_hash": finalization["state_transition_hash"]})
    _require(section.get("weight_finalization_event_hash") == event_hash, "compact finalization event hash differs")
    return {"weight_finalization_event_hash": event_hash, "extrinsic_hash": finalization["extrinsic_hash"], "finalized_block": int(finalization["finalized_block"]), "finalized_block_hash": finalization["finalized_block_hash"], "state_transition_hash": finalization["state_transition_hash"], "finalization_receipt_hash": root_hash}


def verify_compact_published_weight_authority_v2(
    authority: Mapping[str, Any],
    *,
    identity_cache: Optional[Mapping[str, Any]],
    chain_signing_profile: Optional[Mapping[str, Any]],
    expected_lineage_id: str,
    expected_chain: str,
    boot_verifier: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Independently verify one bounded published/finalized authority."""

    normalized = validate_compact_published_weight_authority_shape_v2(authority)
    _require(normalized["lineage_id"] == expected_lineage_id, "compact public lineage differs")
    verifier = boot_verifier or functools.partial(verify_boot_identity_nitro, certificate_validity_at_attestation_time=True)
    verified = verify_compact_weight_submission_v2(normalized["compact_submission"], expected_lineage_id=expected_lineage_id, expected_chain=expected_chain, identity_cache=identity_cache, boot_verifier=verifier)
    _require(
        normalized["bundle_hash"] == verified["bundle_hash"],
        "compact public bundle hash differs",
    )
    verified["ancestry_commitment"] = str(normalized["compact_submission"]["ancestry_commitment"])
    publication = normalized["publication"]
    pub_doc = dict(publication["publication_doc"])
    pub_root = _hash(publication["publication_receipt_hash"], "publication receipt hash")
    proof = validate_compact_ancestry_proof_v2(
        publication["ancestry_proof"],
        expected_lineage_id=expected_lineage_id,
        boot_attestation_verifier=lambda boot: _verify_boot_for_environment(
            boot,
            identity_cache=identity_cache,
            boot_verifier=verifier,
        ),
        allowed_issuer_roles={"gateway_coordinator"},
        required_receipt_hashes={pub_root},
        required_purposes={"gateway.weights.publication.v2"},
    )
    disclosed = [item for item in proof["disclosed_receipts"] if item.get("receipt_hash") == pub_root]
    _require(len(disclosed) == 1, "compact publication receipt is ambiguous")
    receipt = disclosed[0]
    expected_doc = {"schema_version": "leadpoet.weight_publication.v2", "bundle_hash": normalized["bundle_hash"], "root_receipt_hash": verified["root_receipt_hash"], "durable_readback_hash": pub_doc.get("durable_readback_hash"), "transparency_event_hash": pub_doc.get("transparency_event_hash")}
    _require(pub_doc == expected_doc and receipt.get("role") == "gateway_coordinator" and receipt.get("purpose") == "gateway.weights.publication.v2" and int(receipt.get("epoch_id", -1)) == verified["epoch_id"] and receipt.get("parent_receipt_hashes") == [verified["root_receipt_hash"]] and receipt.get("output_root") == sha256_json(expected_doc), "compact publication differs from reconstructed authority")
    event_hash = sha256_json({"bundle_hash": normalized["bundle_hash"], "publication_receipt_hash": pub_root, "transparency_event_hash": expected_doc["transparency_event_hash"], "durable_readback_hash": expected_doc["durable_readback_hash"]})
    _require(publication.get("weight_submission_event_hash") == event_hash, "compact publication event hash differs")
    verified.update({"bundle_hash": normalized["bundle_hash"], "weight_submission_event_hash": event_hash, "authority_stage": normalized["authority_stage"], "authority_hash": normalized["authority_hash"]})
    if normalized["authority_stage"] == "finalized":
        verified.update(
            _verify_compact_finalization(
                normalized["finalization"],
                verified=verified,
                publication_event_hash=event_hash,
                compact_weight_submission=normalized["compact_submission"],
                expected_lineage_id=expected_lineage_id,
                identity_cache=identity_cache,
                chain_signing_profile=chain_signing_profile,
                boot_verifier=verifier,
            )
        )
    return verified


__all__ = [
    "COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION",
    "CompactAuditorAuthorityV2Error",
    "build_compact_published_weight_authority_v2",
    "verify_compact_weight_submission_v2",
    "validate_compact_published_weight_authority_shape_v2",
    "verify_compact_published_weight_authority_v2",
]
