#!/usr/bin/env python3
"""Admit the current signed Sourcing artifact through the production seam."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

from gateway.research_lab.official_baseline_model_runner import (
    LEGACY_MODEL_RUNNER_FAMILY,
    select_official_baseline_release,
)
from gateway.research_lab.official_baseline_release_authorities import (
    preflight_official_baseline_artifact_protocol,
)
from research_lab.canonical import sha256_json
from research_lab.eval import (
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    validate_private_model_artifact_manifest,
    verify_private_artifact_manifest_signature,
)
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    PrivateModelRuntimeError,
    validate_sourcing_adapter_metadata,
)


AWS_ACCOUNT_ID = "493765492819"
AWS_REGION = "us-east-1"
ARTIFACT_BUCKET = "leadpoet-private-model-artifacts-493765492819"
ARTIFACT_PREFIX = "research-lab/sourcing-model"
POINTER_URI = (
    "s3://leadpoet-private-model-artifacts-493765492819/"
    "research-lab/sourcing-model/branches/leadpoet-lab/current.json"
)
SIGNING_KEY_ID = "alias/leadpoet-research-lab-artifact-signing"
ECR_REPOSITORY = "leadpoet/sourcing-model"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_IMAGE_RE = re.compile(
    rf"{AWS_ACCOUNT_ID}\.dkr\.ecr\.{AWS_REGION}\.amazonaws\.com/"
    rf"{re.escape(ECR_REPOSITORY)}@sha256:[0-9a-f]{{64}}"
)
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")


class SignedArtifactAdmissionError(RuntimeError):
    """The signed current artifact cannot safely enter Lead CI."""


def _preflight_legacy_signed_artifact(
    *,
    artifact: PrivateModelArtifactManifest,
    selection: Any,
    spec: DockerPrivateModelSpec,
) -> Mapping[str, Any]:
    """Run the public-safe container-metadata seam for a legacy manifest."""

    expected_selection = select_official_baseline_release(artifact)
    if (
        selection != expected_selection
        or getattr(selection, "is_exact", True)
        or getattr(selection, "runner_family", None)
        != LEGACY_MODEL_RUNNER_FAMILY
        or spec.image_digest != artifact.image_digest
        or spec.env_passthrough
        or spec.extra_env
        or not spec.network_disabled
    ):
        raise SignedArtifactAdmissionError(
            "signed legacy artifact preflight context differs"
        )
    return DockerPrivateModelRunner(spec).metadata()


def _validate_legacy_container_metadata(
    *,
    artifact: PrivateModelArtifactManifest,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validated = validate_sourcing_adapter_metadata(
            metadata,
            expected_scoring_adapter_version=artifact.scoring_adapter_version,
        )
    except PrivateModelRuntimeError as exc:
        raise SignedArtifactAdmissionError(
            "signed legacy artifact container metadata differs"
        ) from exc
    config = {
        "component_registry_version": validated.get(
            "component_registry_version"
        ),
        "scoring_adapter_version": validated.get("scoring_adapter_version"),
        "adapter_version": validated.get("adapter_version"),
    }
    if (
        config["component_registry_version"]
        != artifact.component_registry_version
        or config["scoring_adapter_version"]
        != artifact.scoring_adapter_version
        or sha256_json(config) != artifact.config_hash
    ):
        raise SignedArtifactAdmissionError(
            "signed legacy artifact container configuration differs"
        )
    return validated


def _legacy_receipt_fields(
    *,
    artifact: PrivateModelArtifactManifest,
    selection: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validate_legacy_container_metadata(
        artifact=artifact,
        metadata=metadata,
    )
    fields = {
        "legacy_release_sha256": selection.selection_document.get(
            "legacy_release_sha256"
        ),
        "container_metadata_sha256": sha256_json(validated),
        "preflight_mode": "network_none_container_metadata",
    }
    if (
        _HASH_RE.fullmatch(str(fields["legacy_release_sha256"] or ""))
        is None
        or _HASH_RE.fullmatch(str(fields["container_metadata_sha256"] or ""))
        is None
    ):
        raise SignedArtifactAdmissionError(
            "signed legacy artifact metadata receipt differs"
        )
    return fields


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def _closed_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_closed_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignedArtifactAdmissionError(
            f"{label} is not one closed UTF-8 JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise SignedArtifactAdmissionError(f"{label} is not a JSON object")
    return value


def _parse_exact_s3_uri(uri: str) -> tuple[str, str]:
    prefix = "s3://"
    if not uri.startswith(prefix) or uri.count("//") != 1:
        raise SignedArtifactAdmissionError("artifact S3 URI is invalid")
    bucket, separator, key = uri[len(prefix) :].partition("/")
    if not separator or not bucket or not key or "?" in key or "#" in key:
        raise SignedArtifactAdmissionError("artifact S3 URI is invalid")
    return bucket, key


def _read_s3_bytes(s3_client: Any, uri: str) -> bytes:
    bucket, key = _parse_exact_s3_uri(uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    raw = response["Body"].read()
    if not isinstance(raw, bytes):
        raise SignedArtifactAdmissionError("artifact S3 object is not bytes")
    return raw


def _validate_artifact_locations(
    artifact: PrivateModelArtifactManifest,
) -> str:
    commit = str(artifact.git_commit_sha or "")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise SignedArtifactAdmissionError(
            "artifact commit is not one full lowercase Git SHA"
        )
    expected_manifest_uri = (
        f"s3://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{commit}.json"
    )
    expected_signature_ref = (
        f"s3://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX}/{commit}.sig.b64"
    )
    if artifact.manifest_uri != expected_manifest_uri:
        raise SignedArtifactAdmissionError(
            "artifact manifest URI is outside the immutable release path"
        )
    if artifact.signature_ref != expected_signature_ref:
        raise SignedArtifactAdmissionError(
            "artifact signature URI is outside the immutable release path"
        )
    if _IMAGE_RE.fullmatch(str(artifact.image_digest or "")) is None:
        raise SignedArtifactAdmissionError(
            "artifact image is outside the exact production ECR repository"
        )
    return commit


def admit_current_signed_artifact(
    *,
    s3_client: Any | None = None,
    signature_verifier: Callable[..., Mapping[str, Any]] = (
        verify_private_artifact_manifest_signature
    ),
    protocol_preflight: Callable[..., None] = (
        preflight_official_baseline_artifact_protocol
    ),
    legacy_protocol_preflight: Callable[..., Mapping[str, Any]] = (
        _preflight_legacy_signed_artifact
    ),
) -> dict[str, Any]:
    """Verify current/archive/signature/image identity and run no-spend preflight."""

    if s3_client is None:
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - CI dependency preflight
            raise SignedArtifactAdmissionError("boto3 is required") from exc
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    pointer_raw = _read_s3_bytes(s3_client, POINTER_URI)
    pointer_document = _strict_json_object(
        pointer_raw,
        label="signed artifact pointer",
    )
    artifact = PrivateModelArtifactManifest.from_mapping(pointer_document)
    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise SignedArtifactAdmissionError(
            "signed artifact manifest is invalid: " + ",".join(sorted(errors))
        )
    commit = _validate_artifact_locations(artifact)

    archive_raw = _read_s3_bytes(s3_client, artifact.manifest_uri)
    if archive_raw != pointer_raw:
        raise SignedArtifactAdmissionError(
            "signed current pointer differs byte-for-byte from its archive"
        )
    archive_document = _strict_json_object(
        archive_raw,
        label="immutable signed artifact archive",
    )
    if archive_document != pointer_document:
        raise SignedArtifactAdmissionError(
            "signed current pointer differs semantically from its archive"
        )

    verification = signature_verifier(
        artifact,
        key_id=SIGNING_KEY_ID,
        s3_client=s3_client,
    )
    if verification.get("verified") is not True:
        raise SignedArtifactAdmissionError(
            "signed artifact KMS verification did not pass"
        )
    selection = select_official_baseline_release(artifact)
    spec = DockerPrivateModelSpec(
        image_digest=artifact.image_digest,
        timeout_seconds=300,
        env_passthrough=(),
        extra_env={},
        pull_before_run=True,
        network_disabled=True,
    )
    if selection.is_exact:
        protocol_preflight(
            artifact=artifact,
            selection=selection,
            spec=spec,
        )
        selection_fields = {
            "protocol_generation_sha256": selection.selection_document[
                "protocol_generation_sha256"
            ],
            "preflight_mode": "exact_model_runner_protocol",
        }
    else:
        legacy_metadata = legacy_protocol_preflight(
            artifact=artifact,
            selection=selection,
            spec=spec,
        )
        if not isinstance(legacy_metadata, Mapping):
            raise SignedArtifactAdmissionError(
                "signed legacy artifact container metadata is unavailable"
            )
        selection_fields = _legacy_receipt_fields(
            artifact=artifact,
            selection=selection,
            metadata=legacy_metadata,
        )
    final_pointer_raw = _read_s3_bytes(s3_client, POINTER_URI)
    if final_pointer_raw != pointer_raw:
        raise SignedArtifactAdmissionError(
            "signed current pointer changed during artifact admission"
        )
    return {
        "schema_version": "leadpoet.signed_sourcing_artifact_admission.v2",
        "status": "passed",
        "git_commit_sha": commit,
        "manifest_hash": artifact.manifest_hash,
        "model_artifact_hash": artifact.model_artifact_hash,
        "image_digest": artifact.image_digest,
        "config_hash": artifact.config_hash,
        "runner_family": selection.runner_family,
        "selection_sha256": selection.selection_sha256,
        **selection_fields,
        "network": "none",
        "provider_credentials_forwarded": False,
        "pointer_stable_through_admission": True,
    }


def main() -> int:
    receipt = admit_current_signed_artifact()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
