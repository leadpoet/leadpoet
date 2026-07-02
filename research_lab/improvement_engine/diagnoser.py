"""Gated root-cause diagnosis helpers."""

from __future__ import annotations

from typing import Sequence

from .models import EngineTraceEvent


def draft_root_cause(events: Sequence[EngineTraceEvent]) -> dict[str, object]:
    category = events[0].failure_category if events else "unknown"
    components = sorted({event.component for event in events if event.component})
    reasons = sorted({event.normalized_failure_reason for event in events if event.normalized_failure_reason})[:5]
    return {
        "root_cause_summary": f"Recurring {category} cluster detected in canonical Research Lab records.",
        "evidence": reasons,
        "likely_components": components[:5],
        "fix_strategy": "Review linked canonical traces and add a targeted regression before proposing runtime changes.",
        "regression_evaluator_strategy": f"Create a shadow evaluator that flags future {category} events.",
        "risk_level": "medium",
        "confidence": min(0.95, 0.45 + 0.05 * len(events)),
        "needs_human_review": True,
        "llm_diagnosis_enabled": False,
    }
