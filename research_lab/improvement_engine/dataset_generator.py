"""Draft redacted dataset refs for Engine issues."""

from __future__ import annotations

from .models import EngineIssue


def draft_dataset_spec(issue: EngineIssue) -> dict[str, object]:
    return {
        "dataset_item_type": "private_ref_regression",
        "status": "draft",
        "input_refs": {
            "run_ids": list(issue.linked_run_ids),
            "score_bundle_hashes": list(issue.linked_score_bundle_hashes),
        },
        "expected_output": {
            "assertions": [f"Future runs should not reproduce {issue.category} for the same fingerprint."],
        },
        "privacy": {
            "refs_only": True,
            "raw_sealed_data_included": False,
        },
    }
