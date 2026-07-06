"""Tests for the code_editing / code_build fixes from fableanalysis.md.

Covers: bug #18 (forbidden-term scan on added lines only), bug #19 (value-level
secret redaction), bug #21 prompt/parser side (verdict synonyms), bug #22
(novelty semantic key matches the worker's stored shape), bug #29(a) (real
head sha recorded), bug #30 (infra-vs-candidate build failure classification).
"""

from __future__ import annotations

import json
import pytest

from gateway.research_lab import code_build
from research_lab import code_editing
from research_lab.code_editing import CodeEditDraft


def _draft(**overrides):
    payload = dict(
        failure_mode="weak recall",
        mechanism="widen fan-out",
        expected_improvement="+2 companies",
        risk="slower",
        lane="provider",
        target_files=("sourcing_model.py",),
        unified_diff="--- a/sourcing_model.py\n+++ b/sourcing_model.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        redacted_summary="widen provider fan-out for recall",
        test_plan="smoke",
        rollback_plan="revert",
    )
    payload.update(overrides)
    return CodeEditDraft(**payload)


# --- bug #18: forbidden terms scanned on added lines only ---


def _diff_with(context_line: str, added_line: str) -> str:
    return (
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1,3 +1,4 @@\n"
        f" {context_line}\n"
        "-old = 1\n"
        f"+{added_line}\n"
        " tail = 2\n"
    )


def test_forbidden_term_in_context_line_passes():
    diff = _diff_with("value = load(judge_prompt)", "new = 2")
    assert code_editing._contains_forbidden_material_diff_aware(diff) is False


def test_forbidden_term_in_added_line_rejects():
    diff = _diff_with("clean = 1", "leak = read('judge_prompt')")
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


def test_forbidden_term_in_removed_line_passes():
    diff = (
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-token = fetch('service_role')\n"
        "+token = fetch_public()\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is False


def test_diff_added_material_keeps_headers_and_added_lines_only():
    # File headers stay (paths are model-chosen — a smuggling vector); context
    # and removed lines are verbatim parent source and are excluded.
    diff = _diff_with("context_material = 1", "also_clean = 1")
    material = code_editing._diff_added_line_material(diff)
    assert "+++ b/gateway/module.py" in material
    assert "also_clean = 1" in material
    assert "context_material" not in material
    assert "old = 1" not in material


def test_forbidden_term_in_model_chosen_path_rejects():
    diff = (
        "--- a/gateway/judge_prompt.py\n"
        "+++ b/gateway/judge_prompt.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


# --- bug #19: value-level redaction preserves structure ---


def test_redact_secret_values_preserves_line_count_and_masks_literal():
    source = (
        "import os\n"
        'OPENROUTER_API_KEY = "sk-or-v1-9a8b7c6d5e4f"\n'
        "def fetch():\n"
        "    return 1\n"
    )
    redacted = code_build._redact_secret_values(source)
    assert len(redacted.splitlines()) == len(source.splitlines())
    assert "sk-or-v1-9a8b7c6d5e4f" not in redacted
    assert "def fetch():" in redacted


def test_redact_source_excerpt_value_mode_keeps_keyword_lines(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_REDACT_VALUES_ONLY", raising=False)
    source = "api_key = os.environ.get('EXA_API_KEY')\nplain = 1\n"
    redacted = code_build._redact_source_excerpt(source)
    # The keyword-mentioning line survives (env lookup, no literal secret) —
    # the legacy mode blanked it and produced un-appliable hunks.
    assert "plain = 1" in redacted
    assert len(redacted.splitlines()) == 2


# --- bug #21 (parser side): verdict synonyms ---


@pytest.mark.parametrize("verdict", ["pass", "PASSED", "Approved", "yes", "aligned", "OK"])
def test_pass_verdict_synonyms(verdict):
    raw = f'{{"verdict": "{verdict}", "reason": "looks good", "confidence": 0.8}}'
    parsed = code_editing.parse_plan_alignment_judge_response(raw)
    assert parsed.verdict == "pass"


@pytest.mark.parametrize("verdict", ["fail", "rejected", "misaligned", "unclear-gibberish"])
def test_unrecognized_verdicts_stay_fail(verdict):
    raw = f'{{"verdict": "{verdict}", "reason": "nope"}}'
    parsed = code_editing.parse_plan_alignment_judge_response(raw)
    assert parsed.verdict == "fail"


def test_boolean_passes_field_accepted():
    parsed = code_editing.parse_plan_alignment_judge_response(
        '{"plan_alignment": {"passes": true, "reason": "matches plan"}}'
    )
    assert parsed.verdict == "pass"


# --- bug #22: novelty semantic key matches the worker's stored shape ---


def test_semantic_summary_key_matches_worker_storage_format():
    summary = "Widen the provider query FAN-OUT   to boost recall!" + " pad" * 200
    draft = _draft(redacted_summary=summary)
    # worker.py stores semantic_edit_summary as the raw summary truncated to
    # 500 chars; both sides must normalize identically or the guard is dead.
    worker_stored = summary[:500]
    assert code_editing._semantic_summary_key(draft) == code_editing._normalize_semantic_summary(
        worker_stored
    )


def test_semantic_summary_key_falls_back_when_summary_empty():
    draft = _draft(redacted_summary="", expected_improvement="boost recall by 2")
    assert code_editing._semantic_summary_key(draft) == code_editing._normalize_semantic_summary(
        "boost recall by 2"
    )


def test_normalize_semantic_summary_is_rewording_stable():
    a = code_editing._normalize_semantic_summary("Widen   provider fan-out.")
    b = code_editing._normalize_semantic_summary("widen provider FAN-OUT")
    assert a == b


# --- bug #30: infra failures classified and retried, not charged to candidate ---


@pytest.mark.parametrize(
    "text",
    [
        "no basic auth credentials",
        "authorization token has expired",
        "toomanyrequests: pull rate limit",
        "dial tcp 10.0.0.1:443: i/o timeout",
        "connection reset by peer",
    ],
)
def test_infra_failure_markers_detected(text):
    assert code_build._is_infra_failure_text(text) is True


def test_candidate_build_failure_not_infra():
    assert code_build._is_infra_failure_text("SyntaxError: invalid syntax in sourcing_model.py") is False
    assert code_build._is_infra_failure_text("assert adapter_version", "test failed") is False


def test_infra_retry_flag_default_on(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", raising=False)
    assert code_build._infra_retry_enabled() is True
    monkeypatch.setenv("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", "false")
    assert code_build._infra_retry_enabled() is False


# --- planner lane regression: do not box all miner focus into provider fallback ---


def test_loop_direction_planner_prompt_allows_source_routing_and_query_construction():
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={
            "ticket_id": "ticket-source-routing",
            "brief_public_summary": "Route to an alternate discovery surface after primary search returns completed-empty.",
        },
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={"zero_company_icps": 8},
        runtime_source_index={
            "editable_files": [
                "sourcing_model/discovery.py",
                "sourcing_model/clients.py",
                "sourcing_model/core.py",
            ]
        },
        budget_context={"requested_compute_budget_usd": 5.0},
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])

    assert "source_routing" in context["allowed_lanes"]
    assert "query_construction" in context["allowed_lanes"]
    assert "Alternate discovery surface/provider routing" in content
    assert '"required_lane":"query_construction"' in content
    assert '"allowed_lanes":["provider_fallback"]' not in content


def test_code_edit_prompt_names_source_routing_lane():
    messages = code_editing.build_code_edit_auto_research_messages(
        ticket={"ticket_id": "ticket-source-routing", "brief_public_summary": "try an alternate discovery surface"},
        artifact_manifest={"git_commit_sha": "a" * 40},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_context={"editable_files": ["sourcing_model/discovery.py"]},
        source_inspection_context={"read_files": ["sourcing_model/discovery.py"]},
        budget_context={},
        loop_direction_plan={
            "required_lane": "source_routing",
            "selected_path_id": "alternate_discovery_surface",
        },
        max_candidates=1,
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])

    assert "source_routing" in context["allowed_lanes"]
    assert "source routing" in content


# --- bug #29(a): real head sha recorded instead of throwaway git-init sha ---


def _manifest(git_sha="1234567890abcdef1234567890abcdef12345678"):
    from research_lab.eval import PrivateModelArtifactManifest

    return PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "a" * 64,
        git_commit_sha=git_sha,
        image_digest="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        component_registry_version="1.0",
        scoring_adapter_version="1.0",
        manifest_uri="s3://bucket/manifest.json",
        manifest_hash="sha256:" + "e" * 64,
        signature_ref="kms://sig",
    )


def test_recorded_sha_prefers_env_then_parent(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", "feedbeef" * 5)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="0" * 40, parent_artifact=_manifest()
    )
    assert (sha, source) == ("feedbeef" * 5, "env")

    monkeypatch.delenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", raising=False)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="0" * 40, parent_artifact=_manifest()
    )
    assert source == "parent_manifest"
    assert sha == "1234567890abcdef1234567890abcdef12345678"


def test_recorded_sha_legacy_flag_restores_workspace(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BUILD_RECORD_REAL_HEAD_SHA", "false")
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="9" * 40, parent_artifact=_manifest()
    )
    assert (sha, source) == ("9" * 40, "build_workspace")


def test_recorded_sha_falls_back_to_workspace_when_nothing_valid(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", raising=False)
    sha, source = code_build._resolve_recorded_commit_sha(
        workspace_sha="9" * 40, parent_artifact=_manifest(git_sha="not-a-sha")
    )
    assert (sha, source) == ("9" * 40, "build_workspace")
