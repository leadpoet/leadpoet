"""Offline composition of the signed routing authorities.

The scoring enclave cannot resolve S3 pointers or call KMS.  The host side
therefore sends a small, already fetched bundle containing the immutable
documents, their signatures, and public verification material.  This module
does not trust the key material in that bundle by itself: callers must pass a
key pin from measured runtime configuration.  A missing pin, a missing
document, or a failed local signature check is a hard failure.

The bundle is an input seam only.  It does not fetch a URI, interpret a KMS
boolean, or expose a generic endpoint.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from research_lab.canonical import sha256_json
from research_lab.eval import PrivateModelArtifactManifest
from gateway.research_lab.routing_experiment_artifacts import (
    SignedRoutingArtifactAuthority,
    VerifiedRoutingArtifactLineage,
)
from gateway.research_lab.routing_provider_bindings import (
    SignedRoutingBindingCatalogLoader,
    SignedRoutingUnitDatasetLoader,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)


ROUTING_AUTHORITY_BUNDLE_SCHEMA = "leadpoet.routing_authority_bundle.v1"
MAX_ROUTING_AUTHORITY_BUNDLE_BYTES = 4 * 1024 * 1024
_DOCUMENT_NAMES = (
    "artifact_pointer",
    "artifact_manifest",
    "artifact_lineage",
    "binding_catalog",
    "unit_dataset",
)
_KEY_NAMES = (
    "artifact",
    "lineage",
    "binding_catalog",
    "unit_dataset",
)
_DOCUMENT_KEY_ROLES = {
    "artifact_pointer": "artifact",
    "artifact_manifest": "artifact",
    "artifact_lineage": "lineage",
    "binding_catalog": "binding_catalog",
    "unit_dataset": "unit_dataset",
}
_KEY_CURVE = ec.SECP256R1
_PRIVATE_ARTIFACT_FIELDS = frozenset(
    {
        "model_artifact_hash",
        "git_commit_sha",
        "image_digest",
        "config_hash",
        "component_registry_version",
        "scoring_adapter_version",
        "manifest_uri",
        "manifest_hash",
        "signature_ref",
        "build_id",
        "compatibility_contract",
        "consumer_parity_fixtures",
    }
)


class RoutingAuthorityBundleError(ValueError):
    """The host-supplied authority bundle is not safe to use."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingAuthorityBundleError(f"routing authority {label} is not an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RoutingAuthorityBundleError(f"routing authority {label} fields are invalid")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoutingAuthorityBundleError(f"routing authority {label} is missing")
    return value.strip()


def _canonical_bundle_size(value: Mapping[str, Any]) -> int:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RoutingAuthorityBundleError("routing authority bundle is not JSON") from exc
    if len(raw) > MAX_ROUTING_AUTHORITY_BUNDLE_BYTES:
        raise RoutingAuthorityBundleError("routing authority bundle is oversized")
    return len(raw)


def _load_p256_public_key(value: Any, label: str) -> tuple[bytes, ec.EllipticCurvePublicKey]:
    pem = _nonempty_text(value, label).encode("utf-8")
    try:
        key = serialization.load_pem_public_key(pem)
        der = key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except Exception as exc:  # noqa: BLE001 - key material is an authority boundary
        raise RoutingAuthorityBundleError(f"routing authority {label} key is invalid") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, _KEY_CURVE):
        raise RoutingAuthorityBundleError(f"routing authority {label} key is not P-256")
    return der, key


def _signature_bytes(value: Any, label: str) -> bytes:
    encoded = _nonempty_text(value, label)
    try:
        result = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RoutingAuthorityBundleError(f"routing authority {label} signature is invalid") from exc
    if not result or len(result) > 512:
        raise RoutingAuthorityBundleError(f"routing authority {label} signature size is invalid")
    return result


@dataclass(frozen=True)
class VerifiedRoutingAuthorityBundle:
    """The three typed authorities reconstructed entirely in enclave memory."""

    artifact_lineage: VerifiedRoutingArtifactLineage
    binding_catalog: VerifiedRoutingBindingCatalog
    unit_dataset: VerifiedRoutingUnitDataset
    bundle_hash: str


def load_verified_routing_authority_bundle(
    bundle: Mapping[str, Any],
    *,
    pinned_public_keys: Mapping[str, str],
) -> VerifiedRoutingAuthorityBundle:
    """Verify and reconstruct a routing authority bundle without network I/O.

    ``pinned_public_keys`` must originate from measured runtime configuration.
    The bundle's copy of a key is checked byte-for-byte against that pin; a
    key or KMS result supplied only by the host is never sufficient.
    """

    bundle_value = _mapping(bundle, "bundle")
    _canonical_bundle_size(bundle_value)
    _exact_keys(
        bundle_value,
        {"schema_version", "pointer_uri", "key_ids", "verification_keys", "documents", "signatures"},
        "bundle",
    )
    if bundle_value.get("schema_version") != ROUTING_AUTHORITY_BUNDLE_SCHEMA:
        raise RoutingAuthorityBundleError("routing authority bundle schema is invalid")
    pointer_uri = _nonempty_text(bundle_value["pointer_uri"], "pointer_uri")
    if not pointer_uri.startswith("s3://") or not pointer_uri.endswith("/branches/leadpoet-lab/current.json"):
        raise RoutingAuthorityBundleError("routing authority pointer URI is invalid")

    key_ids_value = _mapping(bundle_value["key_ids"], "key_ids")
    _exact_keys(key_ids_value, set(_KEY_NAMES), "key_ids")
    key_ids = {name: _nonempty_text(key_ids_value[name], f"key_ids.{name}") for name in _KEY_NAMES}

    pinned_value = _mapping(pinned_public_keys, "pinned_public_keys")
    if set(pinned_value) != set(_KEY_NAMES) and set(pinned_value) != set(key_ids.values()):
        raise RoutingAuthorityBundleError("routing authority public-key pins are incomplete")
    # Accept either role names or key IDs as the measured pin map, but never
    # mix the forms.  This makes rotation explicit and prevents ambiguous pins.
    if set(pinned_value) == set(_KEY_NAMES):
        pinned_by_role = {role: pinned_value[role] for role in _KEY_NAMES}
    else:
        pinned_by_role = {role: pinned_value[key_ids[role]] for role in _KEY_NAMES}

    keys_value = _mapping(bundle_value["verification_keys"], "verification_keys")
    if set(keys_value) != set(key_ids.values()):
        raise RoutingAuthorityBundleError("routing authority verification keys are incomplete")
    verified_keys: dict[str, ec.EllipticCurvePublicKey] = {}
    for role in _KEY_NAMES:
        key_id = key_ids[role]
        bundle_der, bundle_key = _load_p256_public_key(keys_value[key_id], f"verification_keys.{key_id}")
        pinned_der, _ = _load_p256_public_key(pinned_by_role[role], f"pinned_public_keys.{role}")
        if bundle_der != pinned_der:
            raise RoutingAuthorityBundleError(f"routing authority {role} key does not match pin")
        verified_keys[key_id] = bundle_key

    documents_value = _mapping(bundle_value["documents"], "documents")
    signatures_value = _mapping(bundle_value["signatures"], "signatures")
    if set(documents_value) != set(_DOCUMENT_NAMES) or set(signatures_value) != set(_DOCUMENT_NAMES):
        raise RoutingAuthorityBundleError("routing authority documents or signatures are incomplete")
    documents = {name: dict(_mapping(documents_value[name], f"documents.{name}")) for name in _DOCUMENT_NAMES}
    for name in ("artifact_pointer", "artifact_manifest"):
        if not set(documents[name]).issubset(_PRIVATE_ARTIFACT_FIELDS):
            raise RoutingAuthorityBundleError(f"routing authority {name} has unknown fields")
    if documents["artifact_pointer"] != documents["artifact_manifest"]:
        raise RoutingAuthorityBundleError("routing authority artifact pointer and manifest differ")
    signatures: dict[str, tuple[str, bytes]] = {}
    for name in _DOCUMENT_NAMES:
        signature = _mapping(signatures_value[name], f"signatures.{name}")
        _exact_keys(signature, {"key_id", "signature"}, f"signatures.{name}")
        signature_key_id = _nonempty_text(signature["key_id"], f"signatures.{name}.key_id")
        if signature_key_id != key_ids[_DOCUMENT_KEY_ROLES[name]]:
            raise RoutingAuthorityBundleError(
                f"routing authority {name} signature key has the wrong role"
            )
        signatures[name] = (signature_key_id, _signature_bytes(signature["signature"], f"signatures.{name}"))

    # Validate all signatures before any authority parser is allowed to resolve
    # a document.  The same artifact signature is expected for the pointer and
    # immutable manifest, but both entries remain explicit for completeness.
    for name in _DOCUMENT_NAMES:
        document = documents[name]
        manifest_hash = document.get("manifest_hash")
        if not isinstance(manifest_hash, str) or sha256_json({k: v for k, v in document.items() if k != "manifest_hash"}) != manifest_hash:
            raise RoutingAuthorityBundleError(f"routing authority {name} hash is invalid")
        key_id, signature_bytes = signatures[name]
        signature_ref = document.get("signature_ref")
        if not isinstance(signature_ref, str) or not signature_ref.strip():
            raise RoutingAuthorityBundleError(f"routing authority {name} signature reference is missing")
        try:
            verified_keys[key_id].verify(
                signature_bytes,
                manifest_hash.encode("utf-8"),
                ec.ECDSA(hashes.SHA256()),
            )
        except InvalidSignature as exc:
            raise RoutingAuthorityBundleError(f"routing authority {name} signature was rejected") from exc
        except Exception as exc:  # noqa: BLE001
            raise RoutingAuthorityBundleError(f"routing authority {name} signature verification failed") from exc

    uri_by_document = {
        "artifact_pointer": pointer_uri,
        "artifact_manifest": documents["artifact_manifest"].get("manifest_uri"),
        "artifact_lineage": documents["artifact_lineage"].get("manifest_uri"),
        "binding_catalog": documents["binding_catalog"].get("manifest_uri"),
        "unit_dataset": documents["unit_dataset"].get("manifest_uri"),
    }
    if not all(isinstance(value, str) and value for value in uri_by_document.values()):
        raise RoutingAuthorityBundleError("routing authority document URI is missing")
    if len(set(uri_by_document.values())) != len(uri_by_document):
        raise RoutingAuthorityBundleError("routing authority document URI is duplicated")
    by_uri = {
        str(uri_by_document["artifact_pointer"]): documents["artifact_pointer"],
        str(uri_by_document["artifact_manifest"]): documents["artifact_manifest"],
        str(uri_by_document["artifact_lineage"]): documents["artifact_lineage"],
        str(uri_by_document["binding_catalog"]): documents["binding_catalog"],
        str(uri_by_document["unit_dataset"]): documents["unit_dataset"],
    }
    # The existing authorities enforce their own immutable URI and schema
    # contracts.  The in-memory loader makes it impossible for them to fetch a
    # URI, even if a document contains an s3:// reference.
    def load_document(uri: str) -> Mapping[str, Any]:
        try:
            return by_uri[uri]
        except KeyError as exc:
            raise RoutingAuthorityBundleError("routing authority URI is not in bundle") from exc

    signatures_by_hash: dict[tuple[str, str], bytes] = {}
    for name in _DOCUMENT_NAMES:
        document = documents[name]
        key_id, signature_bytes = signatures[name]
        signatures_by_hash[(str(document["manifest_hash"]), key_id)] = signature_bytes

    def verify_document(document: Mapping[str, Any], key_id: str) -> Mapping[str, Any]:
        document_value = (
            document.to_dict()
            if isinstance(document, PrivateModelArtifactManifest)
            else document
        )
        digest = str(document_value.get("manifest_hash") or "")
        signature_bytes = signatures_by_hash.get((digest, key_id))
        if signature_bytes is None or key_id not in verified_keys:
            raise RoutingAuthorityBundleError("routing authority signature is not bundled")
        try:
            verified_keys[key_id].verify(signature_bytes, digest.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
        except Exception as exc:  # noqa: BLE001
            raise RoutingAuthorityBundleError("routing authority signature was rejected") from exc
        return {
            "verified": True,
            "manifest_hash": digest,
            "signature_ref": document_value.get("signature_ref"),
            "key_id": key_id,
            "signing_algorithm": "ECDSA_SHA_256",
            "consumer_contract_binding_mode": "semantic_v1_required",
        }

    try:
        artifact = SignedRoutingArtifactAuthority(
            pointer_uri=str(uri_by_document["artifact_pointer"]),
            lineage_manifest_uri=str(uri_by_document["artifact_lineage"]),
            key_id=key_ids["artifact"],
            lineage_key_id=key_ids["lineage"],
            loader=load_document,
            verifier=verify_document,
            lineage_verifier=verify_document,
        ).resolve()
        catalog = SignedRoutingBindingCatalogLoader(
            manifest_uri=str(uri_by_document["binding_catalog"]),
            key_id=key_ids["binding_catalog"],
            loader=load_document,
            verifier=verify_document,
        ).load_reviewed_bindings()
        dataset = SignedRoutingUnitDatasetLoader(
            manifest_uri=str(uri_by_document["unit_dataset"]),
            key_id=key_ids["unit_dataset"],
            loader=load_document,
            verifier=verify_document,
        ).load_reviewed_dataset()
    except Exception as exc:  # noqa: BLE001 - preserve one fail-closed error type
        if isinstance(exc, RoutingAuthorityBundleError):
            raise
        raise RoutingAuthorityBundleError("routing authority bundle failed typed validation") from exc

    return VerifiedRoutingAuthorityBundle(
        artifact_lineage=artifact,
        binding_catalog=catalog,
        unit_dataset=dataset,
        bundle_hash=sha256_json(bundle_value),
    )


__all__ = [
    "MAX_ROUTING_AUTHORITY_BUNDLE_BYTES",
    "ROUTING_AUTHORITY_BUNDLE_SCHEMA",
    "RoutingAuthorityBundleError",
    "VerifiedRoutingAuthorityBundle",
    "load_verified_routing_authority_bundle",
]
