"""LLM final judge for SOURCE_ADD Leg 2 rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Mapping, Sequence

from gateway.research_lab.models import reject_secret_material
from gateway.research_lab.store import canonical_hash

PRIMARY_SOURCE_ADD_JUDGE_MODEL = "openai/gpt-5.6-sol"
FALLBACK_SOURCE_ADD_JUDGE_MODEL = "openai/gpt-5.5"


@dataclass(frozen=True)
class SourceAddJudgeVerdict:
    verdict: str
    confidence: float
    source_used: bool
    adapter_id: str = ""
    registry_provider_id: str = ""
    evidence_summary: str = ""
    reason_codes: tuple[str, ...] = ()
    model_id: str = ""
    provider_usage: dict[str, Any] = field(default_factory=dict)
    raw_doc: dict[str, Any] = field(default_factory=dict)
    raw_doc_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "helped" and self.source_used

    def trigger_evidence(self) -> dict[str, Any]:
        return {
            "llm_judge_passed": self.passed,
            "llm_verdict": self.verdict,
            "llm_confidence": float(self.confidence),
            "source_used": bool(self.source_used),
            "adapter_id": self.adapter_id,
            "registry_provider_id": self.registry_provider_id,
            "evidence_summary": self.evidence_summary[:1000],
            "reason_codes": list(self.reason_codes)[:20],
            "judge_model": self.model_id,
            "judge_doc_hash": self.raw_doc_hash or canonical_hash(self.raw_doc or {}),
            "provider_usage": _safe_usage(self.provider_usage),
        }


def openrouter_key_for_source_add_judge() -> str:
    return str(
        os.getenv("RESEARCH_LAB_SOURCE_ADD_JUDGE_OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()


async def judge_source_add_implementation(
    *,
    api_key: str,
    candidate: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    provisioned_sources: Sequence[Mapping[str, Any]],
    timeout_seconds: int = 180,
) -> SourceAddJudgeVerdict:
    """Return a parsed LLM judgment, trying primary then fallback model."""

    if not api_key:
        raise RuntimeError("OpenRouter API key is required for SOURCE_ADD LLM judge")
    prompt_doc = _judge_prompt_doc(
        candidate=candidate,
        score_bundle=score_bundle,
        provisioned_sources=provisioned_sources,
    )
    reject_secret_material(prompt_doc)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final reward judge for SOURCE_ADD Leg 2. "
                "The candidate already won by the required score threshold. "
                "Decide only whether a known SOURCE_ADD API materially helped the winning change. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(prompt_doc, sort_keys=True, separators=(",", ":"), default=str),
        },
    ]
    last_error: Exception | None = None
    for model_id in (PRIMARY_SOURCE_ADD_JUDGE_MODEL, FALLBACK_SOURCE_ADD_JUDGE_MODEL):
        try:
            from research_lab.openrouter_telemetry import call_openrouter_chat_async

            result = await call_openrouter_chat_async(
                api_key=api_key,
                model_id=model_id,
                messages=messages,
                channel="research_lab_source_add",
                purpose="source_add_leg2_llm_judge",
                stage="source_add_leg2_llm_judge",
                max_tokens=900,
                temperature=0.0,
                timeout_seconds=timeout_seconds,
                response_format={"type": "json_object"},
                include_reasoning=True,
                reasoning_effort="high",
            )
            return _parse_verdict(result.content, model_id=model_id, provider_usage=result.provider_usage)
        except Exception as exc:  # noqa: BLE001 - fallback model handles primary failure
            last_error = exc
            continue
    raise RuntimeError(f"SOURCE_ADD LLM judge failed: {type(last_error).__name__ if last_error else 'unknown'}")


def _judge_prompt_doc(
    *,
    candidate: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    provisioned_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task": (
            "Return verdict helped only if the winning code/model change actually used one of "
            "the listed SOURCE_ADD APIs and that use plausibly contributed to the score win. "
            "Return not_helped if no listed source was used. Return uncertain if evidence is incomplete."
        ),
        "required_json_schema": {
            "verdict": "helped | not_helped | uncertain",
            "confidence": "number from 0 to 1",
            "source_used": "boolean",
            "adapter_id": "matching adapter_id when known",
            "registry_provider_id": "matching registry_provider_id when known",
            "evidence_summary": "short explanation",
            "reason_codes": ["short_machine_readable_codes"],
        },
        "candidate": _bounded_mapping(candidate, 9000),
        "score_bundle": _bounded_mapping(score_bundle, 9000),
        "known_source_add_sources": [_source_summary(row) for row in provisioned_sources][:100],
    }


def _source_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    doc = row.get("provision_doc") if isinstance(row.get("provision_doc"), Mapping) else {}
    catalog_doc = row.get("catalog_doc") if isinstance(row.get("catalog_doc"), Mapping) else {}
    entry = doc.get("provider_registry_entry") if isinstance(doc, Mapping) and isinstance(doc.get("provider_registry_entry"), Mapping) else {}
    safe_entry = {
        key: value
        for key, value in dict(entry).items()
        if str(key) not in {"credential_ref"} and "credential" not in str(key).lower()
    }
    return {
        "adapter_id": str(row.get("adapter_id") or ""),
        "registry_provider_id": str(row.get("registry_provider_id") or ""),
        "source_name": str(row.get("source_name") or catalog_doc.get("source_name") or "")[:200],
        "declared_base_domains": row.get("declared_base_domains") or [],
        "provider_registry_entry": safe_entry,
        "probe_endpoints": doc.get("probe_endpoints") if isinstance(doc, Mapping) else [],
    }


def _parse_verdict(content: str, *, model_id: str, provider_usage: Mapping[str, Any]) -> SourceAddJudgeVerdict:
    try:
        decoded = json.loads(str(content or "").strip())
    except json.JSONDecodeError as exc:
        raise ValueError("SOURCE_ADD LLM judge returned malformed JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("SOURCE_ADD LLM judge returned non-object JSON")
    verdict = str(decoded.get("verdict") or "").strip().lower()
    if verdict not in {"helped", "not_helped", "uncertain"}:
        raise ValueError("SOURCE_ADD LLM judge returned invalid verdict")
    source_used = decoded.get("source_used")
    if not isinstance(source_used, bool):
        raise ValueError("SOURCE_ADD LLM judge returned non-boolean source_used")
    try:
        confidence = max(0.0, min(1.0, float(decoded.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    reasons = decoded.get("reason_codes") if isinstance(decoded.get("reason_codes"), list) else []
    return SourceAddJudgeVerdict(
        verdict=verdict,
        confidence=confidence,
        source_used=source_used,
        adapter_id=str(decoded.get("adapter_id") or "")[:200],
        registry_provider_id=str(decoded.get("registry_provider_id") or "")[:200],
        evidence_summary=str(decoded.get("evidence_summary") or "")[:1200],
        reason_codes=tuple(str(item)[:120] for item in reasons),
        model_id=model_id,
        provider_usage=dict(provider_usage or {}),
        raw_doc=dict(decoded),
    )


def _bounded_mapping(value: Mapping[str, Any], max_chars: int) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= max_chars:
        return dict(value)
    return {"truncated": True, "sha256": canonical_hash(value), "excerpt": encoded[:max_chars]}


def _safe_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in dict(value or {}).items()
        if key
        in {
            "provider",
            "channel",
            "purpose",
            "model",
            "outcome",
            "response_id",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
            "reasoning_capture",
        }
    }
