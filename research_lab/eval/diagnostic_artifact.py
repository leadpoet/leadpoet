"""Signed private-artifact admission for read-only scoring diagnostics."""

from __future__ import annotations

import os

from research_lab.sourcing_model_contract_check import (
    QUALIFICATION_SUPPORTED_SCORING_ADAPTER_VERSIONS,
)

from .artifacts import (
    PrivateModelArtifactManifest,
    validate_private_model_artifact_manifest,
)
from .private_runtime import (
    DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
    PrivateModelRuntimeError,
    load_private_artifact_manifest,
    verify_private_artifact_manifest_signature,
)


PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID_ENV = (
    "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID"
)


def load_verified_diagnostic_private_model_artifact(
    manifest_uri: str,
    *,
    expected_image_digest: str,
) -> PrivateModelArtifactManifest:
    """Return a signed artifact bound to the exact diagnostic image.

    Diagnostic entrypoints must not infer rollback semantics from an omitted
    scorer version or from the image name. The version comes only from the
    validated and signature-verified artifact manifest.
    """

    resolved_uri = str(manifest_uri or "").strip()
    resolved_image = str(expected_image_digest or "").strip()
    if not resolved_uri:
        raise PrivateModelRuntimeError(
            "private model artifact manifest URI is required for diagnostics"
        )
    if not resolved_image or "@sha256:" not in resolved_image:
        raise PrivateModelRuntimeError(
            "diagnostic private model image must be an immutable digest"
        )
    try:
        artifact = PrivateModelArtifactManifest.from_mapping(
            load_private_artifact_manifest(resolved_uri)
        )
    except PrivateModelRuntimeError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PrivateModelRuntimeError(
            "diagnostic private model artifact manifest is invalid"
        ) from exc

    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise PrivateModelRuntimeError(
            "diagnostic private model artifact manifest failed validation: "
            + "; ".join(errors)
        )
    signing_key_id = (
        os.getenv(
            PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID_ENV,
            DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
        ).strip()
        or DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID
    )
    try:
        verify_private_artifact_manifest_signature(
            artifact,
            key_id=signing_key_id,
        )
    except Exception as exc:
        raise PrivateModelRuntimeError(
            "diagnostic private model artifact signature verification failed"
        ) from exc
    if artifact.image_digest != resolved_image:
        raise PrivateModelRuntimeError(
            "diagnostic image differs from the signed private model artifact"
        )
    if (
        artifact.scoring_adapter_version
        not in QUALIFICATION_SUPPORTED_SCORING_ADAPTER_VERSIONS
    ):
        raise PrivateModelRuntimeError(
            "diagnostic private model scoring adapter version is unsupported"
        )
    return artifact
