"""Default hosted Research Lab auto-research prompt and candidate parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.engine_v1 import (
    ENGINE_V1_ENABLED_PATCH_TYPES,
    METRIC_NAMES,
    ComponentRegistry,
    HypothesisRecord,
    PatchRecord,
    validate_component_registry,
    validate_hypothesis,
    validate_patch,
)
from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.patches import CandidatePatchManifest, validate_candidate_patch_manifest


FORBIDDEN_TERMS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
    "judge_prompt",
    "hidden_benchmark",
    "private_repo",
)


@dataclass(frozen=True)
class AutoResearchCandidateDraft:
    failure_mode: str
    mechanism: str
    expected_improvement: str
    risk: str
    patch_type: str
    target_component_id: str
    patch_doc: dict[str, Any]
    redacted_summary: str
    predicted_delta: float = 1.0
    falsifier: str = "proxy_score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": {
                "failure_mode": self.failure_mode,
                "mechanism": self.mechanism,
                "expected_improvement": self.expected_improvement,
                "risk": self.risk,
                "predicted_delta": self.predicted_delta,
                "falsifier": self.falsifier,
            },
            "patch": {
                "patch_type": self.patch_type,
                "target_component_id": self.target_component_id,
                "patch_doc": self.patch_doc,
                "redacted_summary": self.redacted_summary,
            },
        }


def build_default_auto_research_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    budget_context: Mapping[str, Any] | None = None,
    max_candidates: int,
) -> list[dict[str, str]]:
    """Build the default server-side prompt for candidate generation."""
    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "budget_context": _redacted_mapping(budget_context or {}),
        "max_candidates": int(max_candidates),
        "enabled_patch_types": list(ENGINE_V1_ENABLED_PATCH_TYPES),
        "disallowed_patch_types": ["CODE_EDIT", "SOURCE_ADD"],
    }
    system = (
        "You are the Leadpoet Research Lab auto-research engine. Your job is to propose small, "
        "typed improvements to the private sourcing model so it improves general high-intent "
        "lead discovery across the sealed Research Lab evaluation distribution. Optimize for "
        "generalizable ICP fit, fresh observable buying intent, evidence quality, low "
        "hallucination rate, and repeatable lead quality. You must not request or reveal private "
        "code, raw provider keys, hidden benchmark plaintext, private customer data, or judge prompts."
    )
    user = (
        "Generate candidate model-improvement patches from the JSON context below.\n"
        "Return strict JSON only, no markdown.\n\n"
        "Rules:\n"
        "- Return at most max_candidates candidates.\n"
        "- Respect budget_context: use fewer, more precise candidates when the compute budget is small.\n"
        "- For top_up payment_kind, focus on pushing the prior promising direction over starting broad new exploration.\n"
        "- Treat ticket.brief_public_summary as an optional miner research direction, not as a client-specific ICP.\n"
        "- Do not overfit to one supplied market segment; propose changes that should generalize across sealed benchmark items.\n"
        "- Allowed patch types: PROMPT_EDIT, PARAM_EDIT, STRATEGY_SWAP.\n"
        "- Never use CODE_EDIT or SOURCE_ADD.\n"
        "- target_component_id must be one of the component registry entry names.\n"
        "- patch_doc must match the component's patch contract.\n"
        "- For PROMPT_EDIT, include template_name and new_template with all required placeholders.\n"
        "- For PARAM_EDIT, include param_name and numeric new_value within registered bounds.\n"
        "- For STRATEGY_SWAP, include strategy_name from registered options.\n"
        "- Focus on improving lead fit, intent freshness, intent specificity, and evidence quality.\n"
        "- Prefer narrower, testable changes over broad vague changes.\n\n"
        "Expected output shape:\n"
        "{\"candidates\":[{\"hypothesis\":{\"failure_mode\":\"...\",\"mechanism\":\"...\","
        "\"expected_improvement\":\"...\",\"risk\":\"...\",\"predicted_delta\":1.0,"
        "\"falsifier\":\"proxy_score\"},\"patch\":{\"patch_type\":\"PROMPT_EDIT\","
        "\"target_component_id\":\"...\",\"patch_doc\":{},\"redacted_summary\":\"...\"}}]}\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_auto_research_response(raw_text: str, *, max_candidates: int = 3) -> list[AutoResearchCandidateDraft]:
    """Parse and validate strict JSON candidate output from the research LLM."""
    decoded = json.loads(_extract_json_object(raw_text))
    if not isinstance(decoded, Mapping):
        raise ValueError("auto-research response must be a JSON object")
    if _contains_forbidden_material(decoded):
        raise ValueError("auto-research response contains forbidden private or secret material")
    candidates = decoded.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("auto-research response requires a non-empty candidates array")

    drafts: list[AutoResearchCandidateDraft] = []
    for item in candidates[: max(1, int(max_candidates))]:
        if not isinstance(item, Mapping):
            raise ValueError("candidate must be an object")
        hypothesis = item.get("hypothesis")
        patch = item.get("patch")
        if not isinstance(hypothesis, Mapping) or not isinstance(patch, Mapping):
            raise ValueError("candidate requires hypothesis and patch objects")
        patch_type = str(patch.get("patch_type") or "")
        if patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
            raise ValueError(f"candidate patch type is not enabled: {patch_type}")
        patch_doc = patch.get("patch_doc")
        if not isinstance(patch_doc, Mapping):
            raise ValueError("candidate patch_doc must be an object")
        falsifier = str(hypothesis.get("falsifier") or "proxy_score")
        if falsifier not in METRIC_NAMES:
            falsifier = "proxy_score"
        drafts.append(
            AutoResearchCandidateDraft(
                failure_mode=str(hypothesis.get("failure_mode") or "")[:600],
                mechanism=str(hypothesis.get("mechanism") or "")[:900],
                expected_improvement=str(hypothesis.get("expected_improvement") or "")[:900],
                risk=str(hypothesis.get("risk") or "")[:600],
                predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
                falsifier=falsifier,
                patch_type=patch_type,
                target_component_id=str(patch.get("target_component_id") or ""),
                patch_doc=dict(patch_doc),
                redacted_summary=str(patch.get("redacted_summary") or "")[:900],
            )
        )
    return drafts


def coerce_component_registry(metadata: Mapping[str, Any]) -> ComponentRegistry:
    """Extract and validate an Engine v1 component registry from adapter metadata."""
    registry_payload = metadata.get("component_registry", metadata)
    if not isinstance(registry_payload, Mapping):
        raise ValueError("adapter metadata did not include component_registry")
    if "manifest_version" not in registry_payload:
        registry_payload = _coerce_compact_component_registry(metadata, registry_payload)
    registry = ComponentRegistry.from_mapping(registry_payload)
    errors = validate_component_registry(registry)
    if errors:
        raise ValueError("component registry failed validation: " + "; ".join(errors))
    return registry


def _coerce_compact_component_registry(
    metadata: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert private-runtime compact metadata into the Engine v1 registry shape.

    Current private artifacts expose component metadata as
    ``component_registry: {component_name: {...}}`` plus top-level version
    fields.  Engine v1 persists an expanded registry so validator work can be
    audited without needing to inspect the private image again.
    """
    manifest_version = str(
        metadata.get("component_registry_version")
        or metadata.get("manifest_version")
        or "sourcing-model-components:v1"
    )
    champion_base = str(
        metadata.get("adapter_version")
        or metadata.get("champion_base")
        or "sourcing-model-research-lab-adapter:v1"
    )
    eval_version = str(
        metadata.get("scoring_adapter_version")
        or metadata.get("eval_version")
        or "research-lab-private-evaluator:v1"
    )

    entries: list[dict[str, Any]] = []
    for name, raw_entry in sorted(registry_payload.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"component registry entry {name!r} must be an object")
        entry = dict(raw_entry)
        allowed_patch_types = [
            str(item)
            for item in entry.get("allowed_patch_types", [])
            if str(item) in ENGINE_V1_ENABLED_PATCH_TYPES
        ]
        if not allowed_patch_types:
            continue
        entries.append(
            {
                "name": str(name),
                "purpose": str(entry.get("purpose") or f"Private model component {name}"),
                "input_contract": str(entry.get("input_contract") or "Private sourcing model runtime input"),
                "output_contract": str(entry.get("output_contract") or "Private sourcing model runtime output"),
                "ablation_leverage": float(entry.get("ablation_leverage") or 1.0),
                "allowed_patch_types": allowed_patch_types,
                "token_budget": int(entry.get("token_budget") or entry.get("max_instruction_chars") or 800),
                "cost_budget_cents": int(entry.get("cost_budget_cents") or 10),
                "prompt_required_placeholders": [
                    str(item) for item in entry.get("prompt_required_placeholders", [])
                ],
                "param_bounds": dict(entry.get("param_bounds") or {}),
                "strategy_options": [
                    str(item) for item in entry.get("strategy_options", [])
                ],
                "source_evidence_refs": [
                    str(item)
                    for item in (
                        entry.get("source_evidence_refs")
                        or [f"component_registry:{manifest_version}:{name}"]
                    )
                ],
                "current_patch_seq": int(entry.get("current_patch_seq") or 0),
            }
        )

    if not entries:
        raise ValueError("component registry did not include any Engine v1-enabled components")
    return {
        "manifest_version": manifest_version,
        "champion_base": champion_base,
        "eval_version": eval_version,
        "entries": entries,
        "source_receipt_refs": [
            str(item)
            for item in (
                metadata.get("source_receipt_refs")
                or [f"component_registry:{manifest_version}:runtime_metadata"]
            )
        ],
        "meta_allocator_priors_enabled": bool(metadata.get("meta_allocator_priors_enabled", False)),
    }


def build_validated_candidate_manifest(
    *,
    draft: AutoResearchCandidateDraft,
    artifact_manifest: PrivateModelArtifactManifest | Mapping[str, Any],
    component_registry: ComponentRegistry | Mapping[str, Any],
    run_id: str,
    sequence: int,
    miner_brief_ref: str,
) -> tuple[CandidatePatchManifest, HypothesisRecord, PatchRecord]:
    """Convert one parsed candidate into validated Engine v1 records."""
    artifact = (
        artifact_manifest
        if isinstance(artifact_manifest, PrivateModelArtifactManifest)
        else PrivateModelArtifactManifest.from_mapping(artifact_manifest)
    )
    registry = (
        component_registry
        if isinstance(component_registry, ComponentRegistry)
        else ComponentRegistry.from_mapping(component_registry)
    )
    seed = {
        "run_id": str(run_id),
        "sequence": int(sequence),
        "draft": draft.to_dict(),
        "parent_artifact_hash": artifact.model_artifact_hash,
    }
    hypothesis_id = "hypothesis:" + sha256_json({"hypothesis": seed}).split(":", 1)[1][:32]
    patch_id = "patch:" + sha256_json({"patch": seed}).split(":", 1)[1][:32]
    context_refs = (
        str(miner_brief_ref),
        str(artifact.manifest_uri),
        "component_registry:" + registry.manifest_version,
    )
    hypothesis = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        component=draft.target_component_id,
        failure_mode=draft.failure_mode,
        mechanism=draft.mechanism,
        patch_type=draft.patch_type,
        predicted_delta=draft.predicted_delta,
        falsifier=draft.falsifier,
        context_pack_refs=context_refs,
        miner_brief_ref=str(miner_brief_ref),
        generated_by="engine",
    )
    patch = PatchRecord(
        patch_id=patch_id,
        hypothesis_id=hypothesis_id,
        component=draft.target_component_id,
        patch_type=draft.patch_type,
        payload=dict(draft.patch_doc),
    )
    errors = validate_hypothesis(hypothesis, registry) + validate_patch(patch, registry)
    if errors:
        raise ValueError("candidate failed Engine v1 validation: " + "; ".join(errors))

    patch_payload = {
        "patch_type": draft.patch_type,
        "target_component_id": draft.target_component_id,
        "patch_doc": dict(draft.patch_doc),
        "hypothesis": hypothesis.to_dict(),
    }
    patch_payload_hash = sha256_json(patch_payload)
    candidate_artifact_hash = sha256_json(
        {
            "parent_artifact_hash": artifact.model_artifact_hash,
            "patch_payload_hash": patch_payload_hash,
            "patch_id": patch_id,
        }
    )
    manifest = CandidatePatchManifest(
        patch_type=draft.patch_type,
        target_component_id=draft.target_component_id,
        parent_artifact_hash=artifact.model_artifact_hash,
        patch_payload_hash=patch_payload_hash,
        redacted_summary=draft.redacted_summary,
        validation_result="passed",
        candidate_artifact_hash=candidate_artifact_hash,
        patch_doc=dict(draft.patch_doc),
    )
    manifest_errors = validate_candidate_patch_manifest(
        manifest,
        allowed_component_ids=tuple(registry.by_name()),
        expected_parent_artifact_hash=artifact.model_artifact_hash,
    )
    if manifest_errors:
        raise ValueError("candidate patch manifest failed validation: " + "; ".join(manifest_errors))
    return manifest, hypothesis, patch


def _extract_json_object(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("auto-research response did not contain a JSON object")
    return text[start : end + 1]


def _redacted_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value)
    if _contains_forbidden_material(data):
        raise ValueError("auto-research prompt context contains forbidden private or secret material")
    return data


def _contains_forbidden_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden_material(key) or _contains_forbidden_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(term in lowered for term in FORBIDDEN_TERMS)
    return False
