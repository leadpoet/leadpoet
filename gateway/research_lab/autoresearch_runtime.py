"""Shared runtime contracts for V2 Git-tree autoresearch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OpenRouterCallResult:
    content: str
    provider_usage: dict[str, Any] = field(default_factory=dict)
    cost_microusd: int = 0


@dataclass(frozen=True)
class AutoResearchRuntimeSettings:
    min_seconds: int
    max_seconds: int
    min_iterations: int
    max_iterations: int
    draft_timeout_seconds: int
    reflection_timeout_seconds: int
    estimated_iteration_cost_usd: float
    max_candidates: int

    def normalized(self) -> "AutoResearchRuntimeSettings":
        min_seconds = max(0, int(self.min_seconds))
        max_seconds = max(1, int(self.max_seconds))
        if max_seconds < min_seconds:
            max_seconds = min_seconds
        min_iterations = max(1, int(self.min_iterations))
        max_iterations = max(min_iterations, int(self.max_iterations))
        return AutoResearchRuntimeSettings(
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            draft_timeout_seconds=max(10, int(self.draft_timeout_seconds)),
            reflection_timeout_seconds=max(10, int(self.reflection_timeout_seconds)),
            estimated_iteration_cost_usd=max(
                0.01, float(self.estimated_iteration_cost_usd)
            ),
            max_candidates=max(1, int(self.max_candidates)),
        )


@dataclass(frozen=True)
class AutoResearchLoopEvent:
    event_type: str
    loop_status: str
    elapsed_seconds: float
    node_id: str | None = None
    candidate_artifact_hash: str | None = None
    candidate_patch_hash: str | None = None
    provider_usage: list[dict[str, Any]] = field(default_factory=list)
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    event_doc: dict[str, Any] = field(default_factory=dict)


def runtime_settings_doc(settings: AutoResearchRuntimeSettings) -> dict[str, Any]:
    return {
        "min_seconds": settings.min_seconds,
        "max_seconds": settings.max_seconds,
        "min_iterations": settings.min_iterations,
        "max_iterations": settings.max_iterations,
        "draft_timeout_seconds": settings.draft_timeout_seconds,
        "reflection_timeout_seconds": settings.reflection_timeout_seconds,
        "estimated_iteration_cost_usd": settings.estimated_iteration_cost_usd,
        "max_candidates": settings.max_candidates,
    }


def running_cost_ledger(
    openrouter_call_count: int,
    estimated_cost_usd: float,
    actual_cost_microusd: int,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "stage": stage,
        "total_usd": round(int(actual_cost_microusd) / 1_000_000, 6),
        "actual_openrouter_cost_usd": round(
            int(actual_cost_microusd) / 1_000_000, 6
        ),
        "actual_openrouter_cost_microusd": int(actual_cost_microusd),
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
        "openrouter_call_count": int(openrouter_call_count),
        "official_scoring": False,
    }


def safe_budget_doc(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "research_model_tier",
        "requested_compute_budget_usd",
        "max_compute_budget_usd",
        "payment_kind",
        "budget_policy_version",
        "additional_compute_budget_usd",
        "continue_from_run_id",
        "continuation_context",
        "topup_reason",
        "tree_policy",
    }
    return {key: value[key] for key in allowed if key in value}


def budget_limit_microusd(budget_context: Mapping[str, Any]) -> int:
    try:
        budget_usd = float(
            budget_context.get("requested_compute_budget_usd") or 0.0
        )
    except (TypeError, ValueError):
        budget_usd = 0.0
    return max(0, int(round(budget_usd * 1_000_000)))


def estimated_call_microusd(estimated_cost_usd: float) -> int:
    try:
        estimate = float(estimated_cost_usd)
    except (TypeError, ValueError):
        estimate = 0.0
    return max(1, int(round(max(0.0, estimate) * 1_000_000)))


def would_exceed_budget(
    actual_cost_microusd: int,
    estimated_next_call_microusd: int,
    budget_limit_microusd: int,
) -> bool:
    if budget_limit_microusd <= 0:
        return True
    return (
        max(0, actual_cost_microusd) + max(0, estimated_next_call_microusd)
        > budget_limit_microusd
    )


def coerce_call_result(value: str | OpenRouterCallResult) -> OpenRouterCallResult:
    if isinstance(value, OpenRouterCallResult):
        return value
    return OpenRouterCallResult(
        content=str(value or ""), provider_usage={}, cost_microusd=0
    )
