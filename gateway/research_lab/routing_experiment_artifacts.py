"""Signed artifact and label authorities for routing experiments.

The host process does not trust a caller supplied manifest, model directory,
or label map.  It resolves the configured ``leadpoet-lab/current.json``
pointer, verifies its KMS signature, re-reads the immutable manifest named by
that pointer, and requires both documents to describe one exact lineage.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Callable, Mapping

from research_lab.canonical import sha256_json
from research_lab.eval import (
    DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
    PrivateModelArtifactManifest,
    load_private_artifact_manifest,
    validate_private_model_artifact_manifest,
    verify_private_artifact_manifest_signature,
)
from research_lab.routing_experiments import (
    SourcingModelArtifactIdentity,
    validate_sourcing_model_artifact_identity,
)


ROUTING_ARTIFACT_POINTER_ENV = "RESEARCH_LAB_ROUTING_MODEL_POINTER_URI"
ROUTING_ARTIFACT_LINEAGE_MANIFEST_ENV = "RESEARCH_LAB_ROUTING_LINEAGE_MANIFEST_URI"
ROUTING_ARTIFACT_LINEAGE_KEY_ENV = "RESEARCH_LAB_ROUTING_LINEAGE_KMS_KEY_ID"
ROUTING_GOLD_LABEL_MANIFEST_ENV = "RESEARCH_LAB_ROUTING_GOLD_LABEL_MANIFEST_URI"
ROUTING_ARTIFACT_REPOSITORY = "leadpoet/Sourcing_model"
ROUTING_ARTIFACT_BRANCH = "leadpoet-lab"
ROUTING_GOLD_LABEL_SCHEMA_VERSION = "leadpoet.routing_gold_labels.v1"
ROUTING_ARTIFACT_LINEAGE_SCHEMA_VERSION = "leadpoet.routing_artifact_lineage_manifest.v1"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_UNIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")


class RoutingArtifactAuthorityError(RuntimeError):
    """A signed routing artifact or label document is not exact."""


def _require_hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise RoutingArtifactAuthorityError(f"routing {name} is not a sha256 digest")
    return text


def _pointer_uri(value: Any) -> str:
    uri = str(value or "").strip()
    if (
        not uri.startswith("s3://")
        or not uri.endswith("/branches/leadpoet-lab/current.json")
    ):
        raise RoutingArtifactAuthorityError(
            "routing artifact pointer must be the leadpoet-lab current.json"
        )
    return uri


def _immutable_manifest_uri(value: Any, *, pointer_uri: str) -> str:
    uri = str(value or "").strip()
    if (
        not uri.startswith("s3://")
        or uri == pointer_uri
        or uri.endswith("/current.json")
        or "/branches/" in uri
    ):
        raise RoutingArtifactAuthorityError(
            "routing artifact manifest URI is not immutable"
        )
    return uri


@dataclass(frozen=True)
class VerifiedRoutingArtifactLineage:
    """Complete signed release identity used by a measured routing run."""

    repository: str
    branch: str
    commit_sha: str
    pointer_uri: str
    pointer_document_hash: str
    immutable_manifest_uri: str
    routing_lineage_manifest_uri: str
    routing_lineage_manifest_hash: str
    manifest_hash: str
    signature_ref: str
    signature_key_id: str
    signature_algorithm: str
    model_artifact_hash: str
    image_digest: str
    config_hash: str
    build_id: str
    component_registry_version: str
    scoring_adapter_version: str
    routing_contract_hash: str
    routing_catalog_hash: str
    routing_policy_hash: str
    feature_schema_hash: str
    verifier_contract_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_hash(self) -> str:
        return sha256_json(
            {
                "schema_version": "leadpoet.routing_artifact_lineage.v2",
                **self.to_dict(),
            }
        )

    def sourcing_model_identity(self) -> SourcingModelArtifactIdentity:
        return SourcingModelArtifactIdentity(
            repository=self.repository,
            branch=self.branch,
            commit_sha=self.commit_sha,
            artifact_uri=self.pointer_uri,
            model_artifact_hash=self.model_artifact_hash,
            manifest_hash=self.manifest_hash,
            routing_contract_hash=self.routing_contract_hash,
            routing_catalog_hash=self.routing_catalog_hash,
            routing_policy_hash=self.routing_policy_hash,
            feature_schema_hash=self.feature_schema_hash,
            verifier_contract_hash=self.verifier_contract_hash,
        )


class SignedRoutingArtifactAuthority:
    """Resolve and verify one branch pointer and its immutable manifest."""

    def __init__(
        self,
        *,
        pointer_uri: str | None = None,
        lineage_manifest_uri: str | None = None,
        loader: Callable[[str], Mapping[str, Any]] = load_private_artifact_manifest,
        verifier: Callable[..., Mapping[str, Any]] = verify_private_artifact_manifest_signature,
        key_id: str = DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
        lineage_verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
        lineage_key_id: str | None = None,
    ) -> None:
        self.pointer_uri = _pointer_uri(
            pointer_uri or os.getenv(ROUTING_ARTIFACT_POINTER_ENV, "")
        )
        self._loader = loader
        self._verifier = verifier
        self._key_id = str(key_id or "").strip()
        if not self._key_id:
            raise RoutingArtifactAuthorityError("routing artifact signing key is missing")
        self.lineage_manifest_uri = _immutable_manifest_uri(
            lineage_manifest_uri
            or os.getenv(ROUTING_ARTIFACT_LINEAGE_MANIFEST_ENV, ""),
            pointer_uri=self.pointer_uri,
        )
        self._lineage_verifier = lineage_verifier
        self._lineage_key_id = str(
            lineage_key_id
            or os.getenv(ROUTING_ARTIFACT_LINEAGE_KEY_ENV, "")
        ).strip()
        if not self._lineage_key_id:
            raise RoutingArtifactAuthorityError(
                "routing artifact lineage signing key is missing"
            )
        self._resolved: VerifiedRoutingArtifactLineage | None = None
        self._verified_pointer_document: dict[str, Any] | None = None

    def _load_verified(self, uri: str) -> tuple[PrivateModelArtifactManifest, dict[str, Any]]:
        try:
            document = dict(self._loader(uri))
            manifest = PrivateModelArtifactManifest.from_mapping(document)
            errors = validate_private_model_artifact_manifest(manifest)
            if errors:
                raise RoutingArtifactAuthorityError(
                    "routing artifact manifest failed validation:" + ";".join(errors)
                )
            verification = dict(self._verifier(manifest, key_id=self._key_id))
        except RoutingArtifactAuthorityError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve authority boundary
            raise RoutingArtifactAuthorityError(
                "routing artifact signature verification failed"
            ) from exc
        if (
            verification.get("verified") is not True
            or verification.get("manifest_hash") != manifest.manifest_hash
            or verification.get("signature_ref") != manifest.signature_ref
            or verification.get("signing_algorithm") != "ECDSA_SHA_256"
            or not str(verification.get("key_id") or "").strip()
            or verification.get("consumer_contract_binding_mode")
            != "semantic_v1_required"
        ):
            raise RoutingArtifactAuthorityError(
                "routing artifact signature binding is incomplete"
            )
        return manifest, verification

    def resolve(self) -> VerifiedRoutingArtifactLineage:
        if self._resolved is not None:
            return self._resolved
        pointer, pointer_verification = self._load_verified(self.pointer_uri)
        immutable_uri = _immutable_manifest_uri(
            pointer.manifest_uri,
            pointer_uri=self.pointer_uri,
        )
        immutable, immutable_verification = self._load_verified(immutable_uri)
        if pointer.to_dict() != immutable.to_dict():
            raise RoutingArtifactAuthorityError(
                "routing pointer and immutable manifest differ"
            )
        # The existing private model manifest predates routing policy hashes.
        # Do not infer or trust them from a local artifact.  A second,
        # purpose-specific KMS-signed immutable document binds those hashes to
        # the already-signed current pointer and private artifact identity.
        lineage_document = dict(self._loader(self.lineage_manifest_uri))
        lineage_payload = dict(lineage_document)
        lineage_manifest_hash = _require_hash(
            lineage_payload.pop("manifest_hash", ""),
            "lineage manifest hash",
        )
        if sha256_json(lineage_payload) != lineage_manifest_hash:
            raise RoutingArtifactAuthorityError(
                "routing artifact lineage manifest hash differs"
            )
        lineage_verifier = self._lineage_verifier or verify_routing_json_kms_signature
        try:
            lineage_verification = dict(
                lineage_verifier(lineage_document, self._lineage_key_id)
            )
        except Exception as exc:  # noqa: BLE001
            raise RoutingArtifactAuthorityError(
                "routing artifact lineage signature verification failed"
            ) from exc
        if (
            lineage_verification.get("verified") is not True
            or lineage_verification.get("manifest_hash") != lineage_manifest_hash
            or lineage_verification.get("signature_ref")
            != lineage_document.get("signature_ref")
            or lineage_verification.get("key_id") != self._lineage_key_id
            or lineage_verification.get("signing_algorithm") != "ECDSA_SHA_256"
        ):
            raise RoutingArtifactAuthorityError(
                "routing artifact lineage signature binding is incomplete"
            )
        expected_lineage_fields = {
            "schema_version",
            "manifest_uri",
            "repository",
            "branch",
            "pointer_uri",
            "pointer_document_hash",
            "private_manifest_hash",
            "model_artifact_hash",
            "commit_sha",
            "image_digest",
            "build_id",
            "routing_contract_hash",
            "routing_catalog_hash",
            "routing_policy_hash",
            "feature_schema_hash",
            "verifier_contract_hash",
            "signature_ref",
            "manifest_hash",
        }
        if set(lineage_document) != expected_lineage_fields:
            raise RoutingArtifactAuthorityError(
                "routing signed lineage fields are invalid"
            )
        if (
            lineage_document["schema_version"]
            != ROUTING_ARTIFACT_LINEAGE_SCHEMA_VERSION
            or lineage_document["manifest_uri"] != self.lineage_manifest_uri
            or lineage_document["repository"] != ROUTING_ARTIFACT_REPOSITORY
            or lineage_document["branch"] != ROUTING_ARTIFACT_BRANCH
            or lineage_document["pointer_uri"] != self.pointer_uri
            or lineage_document["pointer_document_hash"]
            != sha256_json(pointer.to_dict())
            or lineage_document["private_manifest_hash"] != pointer.manifest_hash
            or lineage_document["model_artifact_hash"] != pointer.model_artifact_hash
            or lineage_document["commit_sha"] != pointer.git_commit_sha
            or lineage_document["image_digest"] != pointer.image_digest
            or lineage_document["build_id"] != pointer.build_id
            or not _GIT_SHA_RE.fullmatch(pointer.git_commit_sha)
        ):
            raise RoutingArtifactAuthorityError("routing signed lineage is invalid")
        route_hashes = {
            key: _require_hash(lineage_document[key], key)
            for key in expected_lineage_fields
            if key.endswith("_hash")
            and key
            not in {
                "manifest_hash",
                "pointer_document_hash",
                "private_manifest_hash",
                "model_artifact_hash",
            }
        }
        lineage = VerifiedRoutingArtifactLineage(
            repository=ROUTING_ARTIFACT_REPOSITORY,
            branch=ROUTING_ARTIFACT_BRANCH,
            commit_sha=pointer.git_commit_sha,
            pointer_uri=self.pointer_uri,
            pointer_document_hash=sha256_json(pointer.to_dict()),
            immutable_manifest_uri=immutable_uri,
            routing_lineage_manifest_uri=self.lineage_manifest_uri,
            routing_lineage_manifest_hash=lineage_manifest_hash,
            manifest_hash=pointer.manifest_hash,
            signature_ref=pointer.signature_ref,
            signature_key_id=str(immutable_verification["key_id"]),
            signature_algorithm="ECDSA_SHA_256",
            model_artifact_hash=pointer.model_artifact_hash,
            image_digest=pointer.image_digest,
            config_hash=pointer.config_hash,
            build_id=pointer.build_id,
            component_registry_version=pointer.component_registry_version,
            scoring_adapter_version=pointer.scoring_adapter_version,
            **route_hashes,
        )
        errors = validate_sourcing_model_artifact_identity(
            lineage.sourcing_model_identity()
        )
        if errors:
            raise RoutingArtifactAuthorityError(
                "routing signed model identity is invalid:" + ";".join(errors)
            )
        if pointer_verification != immutable_verification:
            raise RoutingArtifactAuthorityError(
                "routing pointer signature identity changed"
            )
        self._verified_pointer_document = pointer.to_dict()
        self._resolved = lineage
        return lineage

    def verify(
        self,
        *,
        artifact: SourcingModelArtifactIdentity,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Implementation of ``RoutingExperimentArtifactAuthority``.

        The supplied manifest is only a comparison value.  Trust always comes
        from the two documents loaded by this authority.
        """

        lineage = self.resolve()
        if artifact.to_dict() != lineage.sourcing_model_identity().to_dict():
            raise RoutingArtifactAuthorityError(
                "routing spec artifact differs from signed current pointer"
            )
        if (
            self._verified_pointer_document is None
            or dict(manifest) != self._verified_pointer_document
        ):
            raise RoutingArtifactAuthorityError(
                "routing spec manifest differs from signed current pointer"
            )
        return {
            "verified": True,
            "model_artifact_hash": lineage.model_artifact_hash,
            "manifest_hash": lineage.manifest_hash,
            "commit_sha": lineage.commit_sha,
            "pointer_document_hash": lineage.pointer_document_hash,
            "artifact_lineage_hash": lineage.identity_hash(),
            "image_digest": lineage.image_digest,
            "build_id": lineage.build_id,
            "signature_ref": lineage.signature_ref,
            "key_id": lineage.signature_key_id,
            "signing_algorithm": lineage.signature_algorithm,
            "consumer_contract_binding_mode": "semantic_v1_required",
        }


@dataclass(frozen=True)
class VerifiedRoutingGoldLabels:
    manifest_uri: str
    manifest_hash: str
    signature_ref: str
    signing_key_id: str
    label_set_hash: str
    labels: Mapping[str, bool]
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "labels": dict(sorted(self.labels.items())),
        }


class SignedRoutingGoldLabelLoader:
    """Load a purpose-specific signed label document by immutable URI."""

    def __init__(
        self,
        *,
        manifest_uri: str | None = None,
        loader: Callable[[str], Mapping[str, Any]] = load_private_artifact_manifest,
        verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
        key_id: str,
    ) -> None:
        self.manifest_uri = str(
            manifest_uri or os.getenv(ROUTING_GOLD_LABEL_MANIFEST_ENV, "")
        ).strip()
        if (
            not self.manifest_uri.startswith("s3://")
            or self.manifest_uri.endswith("/current.json")
            or "/branches/" in self.manifest_uri
        ):
            raise RoutingArtifactAuthorityError(
                "routing gold-label manifest URI must be immutable"
            )
        self._loader = loader
        self._verifier = verifier
        self._key_id = str(key_id or "").strip()
        if not self._key_id:
            raise RoutingArtifactAuthorityError("routing label signing key is missing")

    def load(
        self,
        *,
        expected_label_set_hash: str,
        expected_unit_refs: tuple[str, ...],
    ) -> VerifiedRoutingGoldLabels:
        document = dict(self._loader(self.manifest_uri))
        if set(document) != {
            "schema_version",
            "labels",
            "label_set_hash",
            "provenance_hash",
            "manifest_uri",
            "signature_ref",
            "manifest_hash",
        }:
            raise RoutingArtifactAuthorityError(
                "routing gold-label manifest fields are invalid"
            )
        payload = dict(document)
        manifest_hash = str(payload.pop("manifest_hash") or "")
        if manifest_hash != sha256_json(payload):
            raise RoutingArtifactAuthorityError(
                "routing gold-label manifest hash differs"
            )
        verification = dict(self._verifier(document, self._key_id))
        if (
            verification.get("verified") is not True
            or verification.get("manifest_hash") != manifest_hash
            or verification.get("signature_ref") != document["signature_ref"]
            or verification.get("key_id") != self._key_id
            or verification.get("signing_algorithm") != "ECDSA_SHA_256"
        ):
            raise RoutingArtifactAuthorityError(
                "routing gold-label signature binding is incomplete"
            )
        labels = document.get("labels")
        if (
            document.get("schema_version") != ROUTING_GOLD_LABEL_SCHEMA_VERSION
            or document.get("manifest_uri") != self.manifest_uri
            or not isinstance(labels, Mapping)
            or any(
                not _UNIT_RE.fullmatch(str(key)) or type(value) is not bool
                for key, value in labels.items()
            )
        ):
            raise RoutingArtifactAuthorityError("routing gold labels are invalid")
        normalized = {str(key): value for key, value in sorted(labels.items())}
        expected_units = tuple(sorted(set(expected_unit_refs)))
        if tuple(normalized) != expected_units:
            raise RoutingArtifactAuthorityError(
                "routing gold-label units differ from the experiment"
            )
        label_hash = sha256_json({"labels": list(normalized.items())})
        if (
            document.get("label_set_hash") != label_hash
            or label_hash != _require_hash(expected_label_set_hash, "label set hash")
        ):
            raise RoutingArtifactAuthorityError("routing gold-label hash differs")
        provenance_hash = _require_hash(
            document.get("provenance_hash"), "label provenance hash"
        )
        return VerifiedRoutingGoldLabels(
            manifest_uri=self.manifest_uri,
            manifest_hash=manifest_hash,
            signature_ref=str(document["signature_ref"]),
            signing_key_id=self._key_id,
            label_set_hash=label_hash,
            labels=normalized,
            provenance_hash=provenance_hash,
        )


def verify_routing_json_kms_signature(
    document: Mapping[str, Any],
    key_id: str,
    *,
    s3_client: Any | None = None,
    kms_client: Any | None = None,
) -> Mapping[str, Any]:
    """Verify a purpose-specific JSON manifest without reusing model policy."""

    manifest_hash = _require_hash(document.get("manifest_hash"), "manifest hash")
    signature_ref = str(document.get("signature_ref") or "").strip()
    if not signature_ref.startswith("s3://"):
        raise RoutingArtifactAuthorityError("routing JSON signature ref is invalid")
    if s3_client is None or kms_client is None:
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - deployment dependency
            raise RoutingArtifactAuthorityError("boto3 is unavailable") from exc
        s3_client = s3_client or boto3.client("s3")
        kms_client = kms_client or boto3.client("kms")
    ref = signature_ref[5:]
    bucket, separator, key = ref.partition("/")
    if not bucket or not separator or not key:
        raise RoutingArtifactAuthorityError("routing JSON signature ref is invalid")
    try:
        raw = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        signature = base64.b64decode(raw, validate=True)
        response = kms_client.verify(
            KeyId=key_id,
            Message=manifest_hash.encode("utf-8"),
            MessageType="RAW",
            Signature=signature,
            SigningAlgorithm="ECDSA_SHA_256",
        )
    except Exception as exc:  # noqa: BLE001
        raise RoutingArtifactAuthorityError(
            "routing JSON KMS signature verification failed"
        ) from exc
    if response.get("SignatureValid") is not True:
        raise RoutingArtifactAuthorityError("routing JSON KMS signature was rejected")
    return {
        "verified": True,
        "manifest_hash": manifest_hash,
        "signature_ref": signature_ref,
        "key_id": str(response.get("KeyId") or key_id),
        "signing_algorithm": "ECDSA_SHA_256",
    }


__all__ = [
    "ROUTING_ARTIFACT_POINTER_ENV",
    "ROUTING_ARTIFACT_LINEAGE_MANIFEST_ENV",
    "ROUTING_ARTIFACT_LINEAGE_KEY_ENV",
    "ROUTING_ARTIFACT_LINEAGE_SCHEMA_VERSION",
    "ROUTING_GOLD_LABEL_MANIFEST_ENV",
    "ROUTING_GOLD_LABEL_SCHEMA_VERSION",
    "RoutingArtifactAuthorityError",
    "VerifiedRoutingArtifactLineage",
    "SignedRoutingArtifactAuthority",
    "VerifiedRoutingGoldLabels",
    "SignedRoutingGoldLabelLoader",
    "verify_routing_json_kms_signature",
]
