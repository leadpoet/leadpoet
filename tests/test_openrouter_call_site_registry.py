"""Acceptance check: every production OpenRouter call site is deliberately
classified under the current broker-backed provider policy.

A new file that starts talking to OpenRouter without a classification fails
this test — provider policy decisions must be deliberate, never silent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# file (repo-relative) → classification
#   uncaptured_by_decision   — provider calls are bounded and costed by the
#                              active broker and do not produce training data.
CALL_SITE_REGISTRY = {
    # -- uncaptured by dated owner decision ---------------------------------
    "gateway/tasks/icp_generator.py": "uncaptured_by_decision",
    # The closed-lab training-trace sink was retired on 2026-09-04. Arena
    # calls are costed and bounded by the host broker; these shared scorers do
    # not write a second S3 trace or receipt stream.
    "qualification/scoring/intent_signal_gate.py": "uncaptured_by_decision",
    "qualification/scoring/intent_verification_three_stage.py": "uncaptured_by_decision",
    "qualification/scoring/role_batch_check.py": "uncaptured_by_decision",
    "qualification/scoring/verification_helpers.py": "uncaptured_by_decision",
    "qualification/scoring/lead_scorer.py": "uncaptured_by_decision",
    "qualification/scoring/company_evidence_investigator.py": "uncaptured_by_decision",
}


def _files_mentioning_openrouter() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    roots = {
        "gateway",
        "qualification",
        "leadpoet_verifier",
    }
    return {
        path
        for line in result.stdout.splitlines()
        if (path := line.strip())
        and path.split("/", 1)[0] in roots
        and (REPO_ROOT / path).is_file()
        and "openrouter.ai/api/v1"
        in (REPO_ROOT / path).read_text(encoding="utf-8")
    }


def test_every_openrouter_call_site_is_classified():
    found = _files_mentioning_openrouter()
    unclassified = sorted(found - set(CALL_SITE_REGISTRY))
    assert not unclassified, (
        "New OpenRouter call sites without a capture classification "
        f"(add them to CALL_SITE_REGISTRY with a deliberate decision): {unclassified}"
    )


def test_registry_entries_still_exist():
    stale = [
        rel_path
        for rel_path in CALL_SITE_REGISTRY
        if not (REPO_ROOT / rel_path).exists()
    ]
    assert not stale, f"registry entries for deleted files: {stale}"
