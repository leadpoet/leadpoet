"""Select a lease-pinned runtime without changing its frozen grading identity."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from lab_arena import contracts, images

SCHEMA_VERSION = "leadpoet.lab_arena.reviewer_runtime.v1"
AUDIT_FIELDS = (
    "grading_image_digest", "grading_image_reference",
    "runtime_image_digest", "runtime_image_reference",
)


class ReviewerRuntimeError(ValueError):
    """The persisted runtime does not belong to this exact grading lease."""


def lease_binding(
    run: Mapping[str, Any], configuration: Mapping[str, Any],
) -> Dict[str, Any] | None:
    document = run.get("reviewer_runtime_doc")
    if document is None:
        return None
    if (not isinstance(document, Mapping)
        or set(document) != {"schema_version", "run_id", "lease_generation", *AUDIT_FIELDS}
        or document.get("schema_version") != SCHEMA_VERSION
        or run.get("kind") != "score"
        or document.get("run_id") != run.get("run_id")
        or type(document.get("lease_generation")) is not int
        or document["lease_generation"] < 1
        or document["lease_generation"] != run.get("lease_generation")
        or document.get("grading_image_digest") != configuration.get("scorer_image_digest")
        or document.get("grading_image_reference") != configuration.get("scorer_image_reference")):
        raise ReviewerRuntimeError("reviewer runtime lease binding differs")
    try:
        for prefix in ("grading", "runtime"):
            digest = document[prefix + "_image_digest"]
            reference = document[prefix + "_image_reference"]
            contracts.require_sha256(digest, prefix + " image")
            if images.parse_reference(reference).digest != digest:
                raise ReviewerRuntimeError("reviewer runtime image reference differs")
    except (contracts.ArenaContractError, images.ImageError, TypeError, ValueError) as exc:
        raise ReviewerRuntimeError("reviewer runtime image is invalid") from exc
    return dict(document)


def selected_image(
    run: Mapping[str, Any], configuration: Mapping[str, Any],
) -> Dict[str, str]:
    binding = lease_binding(run, configuration)
    return {
        "image_digest": str(binding["runtime_image_digest"] if binding else configuration["scorer_image_digest"]),
        "image_reference": str(binding["runtime_image_reference"] if binding else configuration["scorer_image_reference"]),
    }


def audit_metadata(lease: Mapping[str, Any]) -> Dict[str, str]:
    document = lease.get("reviewer_runtime_doc")
    if document is None:
        return {}
    if not isinstance(document, Mapping) or any(not isinstance(document.get(key), str) for key in AUDIT_FIELDS):
        raise ReviewerRuntimeError("reviewer runtime audit is invalid")
    if (document["runtime_image_digest"] != lease.get("image_digest")
        or document["runtime_image_reference"] != lease.get("image_reference")):
        raise ReviewerRuntimeError("reviewer runtime audit differs from selected image")
    return {key: document[key] for key in AUDIT_FIELDS}


def validate_completion_audit(
    run: Mapping[str, Any], configuration: Mapping[str, Any], result: Mapping[str, Any],
) -> None:
    binding = lease_binding(run, configuration)
    if binding is not None:
        summary = result.get("resource_summary") or {}
        # Existing external validators can run the composed image through the
        # unchanged image lease protocol. Audit metadata is not attestation.
        if (any(key in summary for key in AUDIT_FIELDS)
            and any(summary.get(key) != binding[key] for key in AUDIT_FIELDS)):
            raise ReviewerRuntimeError("completion runtime audit differs from lease")
