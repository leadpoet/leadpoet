"""Bounded multi-file patches: the fallback pass allows a capped multi-file
patch when the contract crosses modules; the validator enforces the cap."""

from __future__ import annotations

import pytest

from research_lab.code_editing import (
    CodeEditDraft,
    build_code_edit_fallback_messages,
    validate_code_edit_draft,
)
from gateway.research_lab.code_loop_engine import _fallback_max_target_files


def _diff(paths: list[str]) -> str:
    chunks = []
    for path in paths:
        chunks.append(
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
    return "".join(chunks)


def _draft(paths: list[str]) -> CodeEditDraft:
    return CodeEditDraft(
        failure_mode="low recall",
        mechanism="retry then alternate surface",
        expected_improvement="more companies sourced",
        risk="low",
        lane="source_routing",
        target_files=tuple(paths),
        unified_diff=_diff(paths),
        redacted_summary="tighten retries",
        test_plan="unit tests",
        rollback_plan="revert",
    )


TWO_FILES = ["gateway/alpha.py", "gateway/beta.py"]
FOUR_FILES = ["gateway/a.py", "gateway/b.py", "gateway/c.py", "gateway/d.py"]


# ---------------------------------------------------------------------------
# validator cap
# ---------------------------------------------------------------------------


def test_multi_file_draft_within_cap_passes():
    assert validate_code_edit_draft(_draft(TWO_FILES), max_target_files=3) == []


def test_multi_file_draft_over_cap_rejected():
    with pytest.raises(ValueError) as excinfo:
        validate_code_edit_draft(_draft(FOUR_FILES), max_target_files=3)
    assert "code_edit_too_many_target_files:4>3" in str(excinfo.value)


def test_cap_zero_means_unlimited():
    assert validate_code_edit_draft(_draft(FOUR_FILES)) == []
    assert validate_code_edit_draft(_draft(FOUR_FILES), max_target_files=0) == []


def test_single_file_cap_restores_strict_behavior():
    with pytest.raises(ValueError):
        validate_code_edit_draft(_draft(TWO_FILES), max_target_files=1)


# ---------------------------------------------------------------------------
# fallback prompt
# ---------------------------------------------------------------------------


def _fallback_messages(max_target_files: int) -> str:
    messages = build_code_edit_fallback_messages(
        ticket={"ticket_id": "t", "brief_public_summary": "improve recall"},
        artifact_manifest={"model_artifact_hash": "sha256:a"},
        component_registry={},
        benchmark_public_summary={},
        budget_context={},
        fallback_reason="draft refused",
        max_candidates=1,
        max_target_files=max_target_files,
    )
    return messages[-1]["content"]


def test_fallback_prompt_single_file_default():
    content = _fallback_messages(1)
    assert "target exactly one file" in content
    assert "crosses module boundaries" not in content


def test_fallback_prompt_bounded_multi_file():
    content = _fallback_messages(3)
    assert "at most 3 files" in content
    assert "crosses module boundaries" in content
    assert "target exactly one file" not in content


def test_fallback_context_carries_cap():
    messages = build_code_edit_fallback_messages(
        ticket={"ticket_id": "t", "brief_public_summary": "improve recall"},
        artifact_manifest={"model_artifact_hash": "sha256:a"},
        component_registry={},
        benchmark_public_summary={},
        budget_context={},
        fallback_reason="draft refused",
        max_target_files=3,
    )
    joined = "".join(m["content"] for m in messages)
    assert '"max_target_files":3' in joined


# ---------------------------------------------------------------------------
# engine accessor
# ---------------------------------------------------------------------------


def test_fallback_cap_accessor_prefers_config(monkeypatch):
    class Cfg:
        code_edit_fallback_max_target_files = 5

    monkeypatch.setenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES", "9")
    assert _fallback_max_target_files(Cfg()) == 5


def test_fallback_cap_accessor_env_fallback(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES", "2")
    assert _fallback_max_target_files(object()) == 2
    monkeypatch.delenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES")
    assert _fallback_max_target_files(object()) == 3  # default


def test_fallback_cap_accessor_floors_at_one():
    class Cfg:
        code_edit_fallback_max_target_files = 0

    assert _fallback_max_target_files(Cfg()) == 1


def test_config_default_and_env(monkeypatch):
    from gateway.research_lab.config import ResearchLabGatewayConfig

    monkeypatch.delenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES", raising=False)
    assert ResearchLabGatewayConfig.from_env().code_edit_fallback_max_target_files == 3
    monkeypatch.setenv("RESEARCH_LAB_CODE_EDIT_FALLBACK_MAX_TARGET_FILES", "1")
    assert ResearchLabGatewayConfig.from_env().code_edit_fallback_max_target_files == 1
