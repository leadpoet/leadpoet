"""Safe, review-only fix proposal scaffolding."""

from __future__ import annotations

from .config import ImprovementEngineConfig
from .models import EngineIssue


def draft_fix_spec(issue: EngineIssue, config: ImprovementEngineConfig | None = None) -> dict[str, object]:
    # Belt and suspenders (plan §8.3 / §20.4): auto-apply stays hardcoded off
    # AND the config contract must agree — a config claiming auto-apply is a
    # deployment error, not a capability.
    if config is not None and config.auto_apply_patches:
        raise ValueError(
            "RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_APPLY_PATCHES=true is not supported: "
            "engine fixes are review-only and are never auto-applied"
        )
    return {
        "candidate_kind": "research_direction",
        "summary": f"Investigate and reduce recurring {issue.category}.",
        "files_to_change": [],
        "patch": "",
        "expected_effect": "Reduce recurrence of this issue cluster without changing reward-critical logic.",
        "risk": "medium",
        "requires_human_review": True,
        "auto_apply_allowed": False,
    }


def sanitized_miner_opportunity(issue: EngineIssue) -> dict[str, object]:
    return {
        "opportunity_id": issue.issue_key,
        "title": issue.title,
        "category": issue.category,
        "priority": issue.priority,
        "public_problem_summary": issue.root_cause_doc.get("root_cause_summary", issue.title),
        "affected_component_hint": ", ".join(issue.root_cause_doc.get("likely_components", []) or ["unknown"]),
        "safe_failure_examples": list(issue.root_cause_doc.get("evidence", []) or [])[:3],
        "suggested_research_directions": [
            str(issue.suggested_fix_doc.get("summary") or "Investigate the recurring failure and propose a targeted improvement.")
        ],
        "acceptance_signal": "Candidate score improves over current benchmark without increasing hard failures.",
        "engine_confidence": issue.confidence,
        "created_from_trace_count": issue.occurrence_count,
    }
