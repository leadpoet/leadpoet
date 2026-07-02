"""Draft evaluator specs from deterministic issue clusters."""

from __future__ import annotations

from .models import EngineIssue


def draft_evaluator_spec(issue: EngineIssue) -> dict[str, object]:
    return {
        "name": f"{issue.category}_shadow_v1",
        "evaluator_type": "deterministic",
        "applies_to": "canonical_research_lab_records",
        "status": "draft",
        "output_scores": [issue.category],
        "privacy": {
            "allows_raw_customer_data": False,
            "allows_sealed_icp": False,
        },
        "requires_human_review": True,
    }
