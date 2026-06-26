"""Candidate patch manifest validation for private Research Lab evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.engine_v1 import ENGINE_V1_ENABLED_PATCH_TYPES


FORBIDDEN_PATCH_TYPES = {"CODE_EDIT", "SOURCE_ADD"}
SECRET_MARKERS = ("sk-or-", "raw_secret", "openrouter_api_key", "raw_openrouter_key")
RUNTIME_PROMPT_INSTRUCTION_MAX_CHARS = 220
RUNTIME_COMPATIBLE_STRATEGY_OPTIONS: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True)
class CandidatePatchManifest:
    patch_type: str
    target_component_id: str
    parent_artifact_hash: str
    patch_payload_hash: str
    redacted_summary: str
    validation_result: str
    candidate_artifact_hash: str
    patch_doc: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CandidatePatchManifest":
        return cls(
            patch_type=str(data["patch_type"]),
            target_component_id=str(data["target_component_id"]),
            parent_artifact_hash=str(data["parent_artifact_hash"]),
            patch_payload_hash=str(data["patch_payload_hash"]),
            redacted_summary=str(data["redacted_summary"]),
            validation_result=str(data.get("validation_result", "pending")),
            candidate_artifact_hash=str(data["candidate_artifact_hash"]),
            patch_doc=dict(data.get("patch_doc") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def manifest_hash(self) -> str:
        return sha256_json(self.to_dict())


def validate_candidate_patch_manifest(
    manifest: CandidatePatchManifest | Mapping[str, Any],
    *,
    allowed_component_ids: Sequence[str] = (),
    expected_parent_artifact_hash: str = "",
) -> list[str]:
    if not isinstance(manifest, CandidatePatchManifest):
        manifest = CandidatePatchManifest.from_mapping(manifest)
    errors: list[str] = []
    data = manifest.to_dict()
    if _contains_secret_material(data):
        errors.append("candidate_patch_contains_raw_secret_material")
    if manifest.patch_type in FORBIDDEN_PATCH_TYPES:
        errors.append(f"patch_type_deferred:{manifest.patch_type}")
    if manifest.patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
        errors.append(f"patch_type_not_enabled:{manifest.patch_type}")
    if allowed_component_ids and manifest.target_component_id not in set(allowed_component_ids):
        errors.append("target_component_not_in_registry")
    if expected_parent_artifact_hash and manifest.parent_artifact_hash != expected_parent_artifact_hash:
        errors.append("parent_artifact_hash_mismatch")
    for field in ("parent_artifact_hash", "patch_payload_hash", "candidate_artifact_hash"):
        if not str(getattr(manifest, field)).startswith("sha256:"):
            errors.append(f"{field}_must_be_sha256")
    if manifest.candidate_artifact_hash == manifest.parent_artifact_hash:
        errors.append("candidate_artifact_hash_must_differ_from_parent")
    if manifest.validation_result not in {"passed", "failed", "pending"}:
        errors.append("unknown_validation_result")
    if manifest.patch_type == "STRATEGY_SWAP":
        strategy = str(
            (manifest.patch_doc or {}).get("strategy_name")
            or (manifest.patch_doc or {}).get("strategy_option")
            or ""
        )
        compatible_options = RUNTIME_COMPATIBLE_STRATEGY_OPTIONS.get(manifest.target_component_id)
        if compatible_options is not None and strategy not in set(compatible_options):
            errors.append(
                f"runtime_incompatible_strategy:{manifest.target_component_id}:{strategy or 'missing'}"
            )
    return errors


def normalize_runtime_patch_doc(
    patch_type: str,
    target_component_id: str,
    patch_doc: Mapping[str, Any] | None,
    *,
    redacted_summary: str = "",
) -> dict[str, Any]:
    """Normalize Engine-v1 patch docs to the private adapter runtime contract.

    The Sourcing_model adapter executes candidates as a parent image plus a
    typed patch overlay. Older drafts used generic Engine-v1 shapes such as
    ``param_name/new_value`` and ``new_template``; the adapter expects concrete
    runtime shapes such as ``params.max_leads`` and ``append_instruction``.
    """
    payload = dict(patch_doc or {})
    patch_type = str(patch_type or "")
    target = str(target_component_id or "")

    if patch_type == "PARAM_EDIT" and target == "output_budget":
        params = payload.get("params")
        if isinstance(params, Mapping):
            return {"params": dict(params)}
        param_name = str(payload.get("param_name") or "")
        if param_name:
            return {"params": {param_name: payload.get("new_value")}}
        return payload

    if patch_type == "PROMPT_EDIT" and target == "discovery_query_builder":
        query_prefix = _runtime_instruction(payload.get("query_prefix"))
        append_instruction = _runtime_instruction(payload.get("append_instruction"))
        normalized: dict[str, Any] = {}
        if query_prefix:
            normalized["query_prefix"] = query_prefix
        if append_instruction:
            normalized["append_instruction"] = append_instruction
        if normalized:
            return normalized

        legacy_summary = _runtime_instruction(payload.get("redacted_summary") or redacted_summary)
        if legacy_summary:
            return {"append_instruction": legacy_summary}

        legacy_template = _runtime_instruction(_strip_template_placeholders(payload.get("new_template")))
        if legacy_template:
            return {"append_instruction": legacy_template}
        return payload

    if patch_type == "STRATEGY_SWAP" and target == "source_router":
        strategy = str(payload.get("strategy_option") or payload.get("strategy_name") or "")
        if strategy:
            return {"strategy_option": strategy}
        return payload

    return payload


def runtime_compatible_candidate_patch_manifest(
    manifest: CandidatePatchManifest | Mapping[str, Any],
) -> CandidatePatchManifest:
    """Return a manifest view whose patch_doc can be executed by the adapter."""
    if not isinstance(manifest, CandidatePatchManifest):
        manifest = CandidatePatchManifest.from_mapping(manifest)
    normalized_doc = normalize_runtime_patch_doc(
        manifest.patch_type,
        manifest.target_component_id,
        manifest.patch_doc or {},
        redacted_summary=manifest.redacted_summary,
    )
    if normalized_doc == (manifest.patch_doc or {}):
        return manifest
    return CandidatePatchManifest(
        patch_type=manifest.patch_type,
        target_component_id=manifest.target_component_id,
        parent_artifact_hash=manifest.parent_artifact_hash,
        patch_payload_hash=manifest.patch_payload_hash,
        redacted_summary=manifest.redacted_summary,
        validation_result=manifest.validation_result,
        candidate_artifact_hash=manifest.candidate_artifact_hash,
        patch_doc=normalized_doc,
    )


def runtime_compatible_strategy_options(
    component_id: str,
    strategy_options: Sequence[str],
) -> tuple[str, ...]:
    compatible_options = RUNTIME_COMPATIBLE_STRATEGY_OPTIONS.get(str(component_id))
    if compatible_options is None:
        return tuple(str(item) for item in strategy_options)
    allowed = set(compatible_options)
    return tuple(str(item) for item in strategy_options if str(item) in allowed)


def _runtime_instruction(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text[:RUNTIME_PROMPT_INSTRUCTION_MAX_CHARS].rstrip()


def _strip_template_placeholders(value: Any) -> str:
    text = str(value or "")
    while "{" in text and "}" in text and text.index("{") < text.index("}"):
        start = text.index("{")
        end = text.index("}", start)
        text = text[:start] + " " + text[end + 1 :]
    return text


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_secret_material(key) or _contains_secret_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False
