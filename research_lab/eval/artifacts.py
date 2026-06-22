"""Private model artifact contracts for Research Lab evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from research_lab.canonical import sha256_json


FORBIDDEN_SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)


@dataclass(frozen=True)
class PrivateModelArtifactManifest:
    model_artifact_hash: str
    git_commit_sha: str
    image_digest: str
    config_hash: str
    component_registry_version: str
    scoring_adapter_version: str
    manifest_uri: str
    manifest_hash: str
    signature_ref: str
    build_id: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PrivateModelArtifactManifest":
        return cls(
            model_artifact_hash=str(data["model_artifact_hash"]),
            git_commit_sha=str(data["git_commit_sha"]),
            image_digest=str(data["image_digest"]),
            config_hash=str(data["config_hash"]),
            component_registry_version=str(data["component_registry_version"]),
            scoring_adapter_version=str(data["scoring_adapter_version"]),
            manifest_uri=str(data["manifest_uri"]),
            manifest_hash=str(data["manifest_hash"]),
            signature_ref=str(data["signature_ref"]),
            build_id=str(data.get("build_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("manifest_hash", None)
        return payload


def validate_private_model_artifact_manifest(
    manifest: PrivateModelArtifactManifest | Mapping[str, Any],
) -> list[str]:
    if not isinstance(manifest, PrivateModelArtifactManifest):
        manifest = PrivateModelArtifactManifest.from_mapping(manifest)

    errors: list[str] = []
    data = manifest.to_dict()
    if _contains_secret_material(data):
        errors.append("artifact_manifest_contains_raw_secret_material")
    for field in ("model_artifact_hash", "config_hash"):
        if not str(data[field]).startswith("sha256:"):
            errors.append(f"{field}_must_be_sha256")
    if not str(manifest.manifest_hash).startswith("sha256:"):
        errors.append("manifest_hash_must_be_sha256")
    if not str(manifest.manifest_uri).startswith("s3://"):
        errors.append("manifest_uri_must_be_s3")
    if not manifest.signature_ref:
        errors.append("signature_ref_required")
    image_ref = str(manifest.image_digest)
    if "@sha256:" not in image_ref:
        errors.append("image_digest_must_be_immutable_digest")
    if ":latest" in image_ref or image_ref.endswith(":latest"):
        errors.append("image_digest_must_not_use_latest")
    if ".dkr.ecr." not in image_ref:
        errors.append("image_digest_must_reference_private_ecr")
    if len(manifest.git_commit_sha) < 7:
        errors.append("git_commit_sha_too_short")

    expected_hash = sha256_json(manifest.hash_payload())
    if manifest.manifest_hash != expected_hash:
        errors.append("manifest_hash_mismatch")
    return errors


def verify_private_model_artifact_manifest(
    manifest: PrivateModelArtifactManifest | Mapping[str, Any],
) -> dict[str, Any]:
    normalized = manifest if isinstance(manifest, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(manifest)
    errors = validate_private_model_artifact_manifest(normalized)
    return {
        "passed": not errors,
        "errors": errors,
        "model_artifact_hash": normalized.model_artifact_hash,
        "manifest_hash": normalized.manifest_hash,
        "image_digest": normalized.image_digest,
        "signature_ref": normalized.signature_ref,
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_secret_material(key) or _contains_secret_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in FORBIDDEN_SECRET_MARKERS)
    return False
