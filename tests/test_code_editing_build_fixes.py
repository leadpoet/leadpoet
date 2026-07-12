"""Tests for the code_editing / code_build fixes from fableanalysis.md.

Covers: bug #18 (forbidden-term scan on added lines only), bug #19 (value-level
secret redaction), bug #21 prompt/parser side (verdict synonyms), bug #22
(novelty semantic key matches the worker's stored shape), bug #29(a) (real
head sha recorded), bug #30 (infra-vs-candidate build failure classification).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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


def test_forbidden_policy_prose_in_added_comment_or_string_passes():
    comment_diff = _diff_with("clean = 1", "# do not use hidden ICP data")
    string_diff = _diff_with("clean = 1", 'policy = "do not use hidden ICP data"')
    policy_fields_diff = _diff_with(
        "clean = 1",
        'forbidden_fields = ["service_role", "hidden_icp"]',
    )
    assert code_editing._contains_forbidden_material_diff_aware(comment_diff) is False
    assert code_editing._contains_forbidden_material_diff_aware(string_diff) is False
    assert code_editing._contains_forbidden_material_diff_aware(policy_fields_diff) is False


def test_multiline_sensitive_environment_access_rejects():
    diff = (
        "diff --git a/gateway/module.py b/gateway/module.py\n"
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1 +1,4 @@\n"
        " clean = 1\n"
        "+value = os.environ[\n"
        "+    \"SUPABASE_SERVICE_ROLE_KEY\"\n"
        "+]\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


def test_secret_shaped_value_in_added_line_rejects():
    diff = _diff_with("clean = 1", 'token = "sk-or-v1-' + "x" * 24 + '"')
    assert code_editing._contains_forbidden_material_diff_aware(diff) is True


@pytest.mark.parametrize(
    "value",
    [
        "AKIA" + "A" * 16,
        "Bearer " + "token-value-" * 3,
        "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 12,
        "-----BEGIN PRIVATE KEY-----",
        "https://user:password@example.test/path",
    ],
)
def test_secret_shaped_values_reject(value):
    assert code_editing._contains_forbidden_material({"reason": value}) is True


def test_forbidden_term_in_removed_line_passes():
    diff = (
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-token = fetch('service_role')\n"
        "+token = fetch_public()\n"
    )
    assert code_editing._contains_forbidden_material_diff_aware(diff) is False


def test_forbidden_api_and_path_remain_blocked():
    api_draft = _draft(
        target_files=("gateway/module.py",),
        unified_diff=(
            "diff --git a/gateway/module.py b/gateway/module.py\n"
            "--- a/gateway/module.py\n"
            "+++ b/gateway/module.py\n"
            "@@ -1 +1,2 @@\n"
            " value = 1\n"
            "+subprocess.run(['unsafe'])\n"
        ),
    )
    with pytest.raises(ValueError, match="code_edit_disallowed_diff_pattern"):
        code_editing.validate_code_edit_draft(api_draft)

    path_draft = _draft(
        target_files=("gateway/.env",),
        unified_diff=(
            "diff --git a/gateway/.env b/gateway/.env\n"
            "--- a/gateway/.env\n"
            "+++ b/gateway/.env\n"
            "@@ -1 +1 @@\n"
            "-SAFE=1\n"
            "+SAFE=2\n"
        ),
    )
    with pytest.raises(ValueError, match="disallowed_repo_path"):
        code_editing.validate_code_edit_draft(path_draft)


def test_new_and_unread_source_paths_remain_blocked():
    builder = object.__new__(code_build.CodeEditCandidateBuilder)
    source_context = SimpleNamespace(editable_files=("sourcing_model/existing.py",))
    new_file_draft = _draft(
        target_files=("sourcing_model/new_file.py",),
        unified_diff=(
            "diff --git a/sourcing_model/new_file.py b/sourcing_model/new_file.py\n"
            "--- a/sourcing_model/new_file.py\n"
            "+++ b/sourcing_model/new_file.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
    )
    assert builder.validate_draft_against_source_context(
        new_file_draft,
        source_context,
    ) == ["code_edit_path_not_in_extracted_source:sourcing_model/new_file.py"]

    existing_draft = _draft(
        target_files=("sourcing_model/existing.py",),
        unified_diff=(
            "diff --git a/sourcing_model/existing.py b/sourcing_model/existing.py\n"
            "--- a/sourcing_model/existing.py\n"
            "+++ b/sourcing_model/existing.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
    )
    assert builder.validate_draft_against_source_context(
        existing_draft,
        source_context,
        read_paths=(),
        require_read=True,
    ) == ["code_edit_unread_source_file:sourcing_model/existing.py"]


def test_git_apply_accepts_exact_replacement_hunk_without_trailing_context(tmp_path):
    source_dir = tmp_path / "sourcing_model"
    source_dir.mkdir()
    source_path = source_dir / "discovery.py"
    source_path.write_text(
        "def build_query_variants():\n"
        "    variants = []\n"
        "    variants.append('strict')\n"
        "    return variants\n"
        "\n"
        "def next_function():\n"
        "    return True\n",
        encoding="utf-8",
    )
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(
        "diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py\n"
        "--- a/sourcing_model/discovery.py\n"
        "+++ b/sourcing_model/discovery.py\n"
        "@@ -1,4 +1,3 @@\n"
        " def build_query_variants():\n"
        "-    variants = []\n"
        "-    variants.append('strict')\n"
        "-    return variants\n"
        "+    return ['strict', 'companion']\n",
        encoding="utf-8",
    )

    code_build._run_git_apply(
        diff_path,
        cwd=tmp_path,
        timeout_seconds=10,
        check=True,
    )
    code_build._run_git_apply(
        diff_path,
        cwd=tmp_path,
        timeout_seconds=10,
        check=False,
    )

    assert "return ['strict', 'companion']" in source_path.read_text(encoding="utf-8")


def test_git_apply_context_fallback_rejects_addition_only_hunks():
    addition_only = (
        "diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py\n"
        "--- a/sourcing_model/discovery.py\n"
        "+++ b/sourcing_model/discovery.py\n"
        "@@ -1,0 +1,1 @@\n"
        "+unsafe_by_line_number = True\n"
    )

    assert code_build._can_retry_git_apply_without_edge_context(addition_only) is False


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


def _v1_1_plan(**overrides):
    path = {
        "path_id": "query-recall",
        "lane": "query_construction",
        "mechanism": "add one bounded query variant",
        "target_behavior": ["recover sparse searches"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": ["query_construction"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["do not weaken ICP gates"],
        "success_criteria": ["runtime checks pass"],
        "novelty_requirements": ["different from prior attempts"],
        "anti_overfit_checks": ["preserve multiple outputs"],
        "validation_mode": "runtime_checks",
        "validation_paths": [],
    }
    payload = {
        "schema_version": "1.1",
        "miner_focus_interpretation": "improve sparse-query recall",
        "loop_goal": "recover qualified companies",
        "required_lane": path["lane"],
        "required_mechanism": path["mechanism"],
        "target_behavior": path["target_behavior"],
        "must_inspect": path["must_inspect"],
        "allowed_lanes": path["allowed_lanes"],
        "disallowed_lanes": path["disallowed_lanes"],
        "must_not_try": path["must_not_try"],
        "success_criteria": path["success_criteria"],
        "novelty_requirements": path["novelty_requirements"],
        "anti_overfit_checks": path["anti_overfit_checks"],
        "ranked_paths": [path],
        "selected_path_id": path["path_id"],
        "validation_mode": "runtime_checks",
        "validation_paths": [],
    }
    payload.update(overrides)
    return payload


def test_loop_direction_v1_0_checkpoint_remains_compatible():
    plan = code_editing.loop_direction_plan_from_mapping(
        {
            "schema_version": "1.0",
            "required_lane": "query_construction",
            "required_mechanism": "bounded query variant",
            "ranked_paths": [{"path_id": "legacy-path"}],
            "selected_path_id": "legacy-path",
        }
    )
    assert plan.validation_mode == "runtime_checks"
    assert plan.validation_paths == ()
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_round_trip_and_contract_validation():
    plan = code_editing.parse_loop_direction_plan_response(json.dumps(_v1_1_plan()))
    first_doc = plan.to_dict()
    reparsed = code_editing.loop_direction_plan_from_mapping(first_doc)
    assert reparsed == plan
    assert reparsed.to_dict()["plan_hash"] == first_doc["plan_hash"]
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_selected_path_overrides_duplicate_cover_fields():
    payload = _v1_1_plan(
        required_lane="output_ranking",
        required_mechanism="different top-level mechanism",
        target_behavior=["different top-level behavior"],
        must_inspect=["sourcing_model/other.py"],
        allowed_lanes=["output_ranking"],
        disallowed_lanes=["query_construction"],
        must_not_try=["different top-level safety rule"],
        success_criteria=["different top-level success rule"],
        novelty_requirements=["different top-level novelty rule"],
        anti_overfit_checks=["different top-level overfit rule"],
        validation_mode="existing_test_files",
        validation_paths=["tests/nonexistent.py"],
    )
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    selected = payload["ranked_paths"][0]
    assert plan.required_lane == selected["lane"]
    assert plan.required_mechanism == selected["mechanism"]
    for field in (
        "target_behavior",
        "must_inspect",
        "allowed_lanes",
        "disallowed_lanes",
        "must_not_try",
        "success_criteria",
        "novelty_requirements",
        "anti_overfit_checks",
        "validation_paths",
    ):
        assert getattr(plan, field) == tuple(selected[field])
    assert plan.validation_mode == selected["validation_mode"]
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_rejects_inconsistent_selected_path():
    payload = _v1_1_plan()
    payload["ranked_paths"][0]["allowed_lanes"] = ["output_ranking"]
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert any(
        error.startswith("ranked_path_lane_not_allowed:")
        for error in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_loop_direction_v1_1_requires_explicit_path_validation_strategy():
    payload = _v1_1_plan()
    payload["ranked_paths"][0].pop("validation_paths")
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert any(
        error.startswith("ranked_path_missing_validation_paths:")
        for error in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_loop_direction_v1_1_allows_explicit_empty_disallowed_lanes():
    payload = _v1_1_plan(disallowed_lanes=[])
    payload["ranked_paths"][0]["disallowed_lanes"] = []
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert code_editing.loop_direction_plan_contract_errors(plan) == []


def test_loop_direction_v1_1_rejects_more_than_three_ranked_paths():
    payload = _v1_1_plan()
    base_path = payload["ranked_paths"][0]
    payload["ranked_paths"] = [
        {**base_path, "path_id": f"path-{index}"}
        for index in range(4)
    ]
    payload["selected_path_id"] = "path-0"
    plan = code_editing.loop_direction_plan_from_mapping(payload)
    assert (
        "loop_direction_plan_v1_1_allows_at_most_three_ranked_paths"
        in code_editing.loop_direction_plan_contract_errors(plan)
    )


def test_existing_test_validation_requires_paths():
    payload = _v1_1_plan()
    payload["ranked_paths"][0]["validation_mode"] = "existing_test_files"
    payload["ranked_paths"][0]["validation_paths"] = []
    with pytest.raises(ValueError, match="requires validation_paths"):
        code_editing.loop_direction_plan_from_mapping(payload)


@pytest.mark.parametrize(
    ("reason", "expected_class"),
    [
        (
            "No existing test file appears in runtime_source_context.editable_files and new files are forbidden.",
            "binding_plan_unimplementable",
        ),
        (
            "No existing test file is listed in editable_files for the required coverage.",
            "binding_plan_unimplementable",
        ),
        ("The provider probe refuted this hypothesis.", "provider_probe_refuted_hypothesis"),
    ],
)
def test_legacy_no_viable_refusal_gets_structured_failure_class(reason, expected_class):
    refusal = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps({"no_viable_patch": True, "reason": reason})
    )
    assert refusal is not None
    assert refusal.failure_class == expected_class


def test_structured_no_viable_refusal_round_trip_and_secret_rejection():
    refusal = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps(
            {
                "no_viable_patch": True,
                "failure_class": "binding_plan_unimplementable",
                "reason": "required symbol is absent",
                "missing_references": ["discover_companies"],
            }
        )
    )
    assert refusal is not None
    assert refusal.missing_references == ("discover_companies",)
    sanitized = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps(
            {
                "no_viable_patch": True,
                "failure_class": "no_safe_patch",
                "reason": "no safe patch\n\twithin the current scope",
                "missing_references": [],
            }
        )
    )
    assert sanitized is not None
    assert sanitized.reason == "no safe patch within the current scope"
    policy = code_editing.parse_code_edit_no_viable_patch_response(
        json.dumps({"no_viable_patch": True, "reason": "service_role must not be accessed"})
    )
    assert policy is not None
    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_code_edit_no_viable_patch_response(
            json.dumps(
                {
                    "no_viable_patch": True,
                    "reason": "credential unavailable",
                    "service_role_key": "synthetic-secret-value-123456",
                }
            )
        )


def test_sensitive_material_claim_rejects_but_policy_prose_passes():
    verdict = code_editing.parse_plan_alignment_judge_response(
        json.dumps(
            {
                "verdict": "pass",
                "reason": "Do not use hidden ICP data when evaluating this patch.",
            }
        )
    )
    assert verdict.verdict == "pass"

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "The patch returns hidden_icp data without redaction.",
                }
            )
        )

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "credential field present",
                    "provider_api_key": "redacted",
                }
            )
        )

    with pytest.raises(ValueError, match="forbidden"):
        code_editing.parse_plan_alignment_judge_response(
            json.dumps(
                {
                    "verdict": "fail",
                    "reason": "The patch mentions hidden_icp material.",
                }
            )
        )


def test_planner_prompt_exposes_safe_validation_capabilities_without_command_text():
    constraints = {
        "new_files_allowed": False,
        "editable_test_path_count": 0,
        "editable_test_paths": [],
        "allowed_validation_modes": ["runtime_checks"],
        "runtime_checks": {"private_test_command_configured": True},
    }
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={"ticket_id": "ticket-validation"},
        artifact_manifest={},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={"editable_files": ["sourcing_model/discovery.py"]},
        budget_context={},
        candidate_edit_constraints=constraints,
    )
    content = messages[-1]["content"]
    context = json.loads(content.split("Context JSON:\n", 1)[1])
    assert context["candidate_edit_constraints"] == constraints
    assert "RESEARCH_LAB_PRIVATE_TEST_CMD" not in content
    assert "do not require adding tests" in content


def test_planner_prompt_example_is_internally_consistent_and_source_bound():
    messages = code_editing.build_loop_direction_planner_messages(
        ticket={"ticket_id": "ticket-example"},
        artifact_manifest={},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={
            "files": [
                {
                    "path": "sourcing_model/discovery.py",
                    "symbols": [{"qualified_name": "Router.discover_companies"}],
                }
            ]
        },
        budget_context={},
        candidate_edit_constraints={},
    )
    content = messages[-1]["content"]
    example_text = content.split(
        "Required output shape (the selected path and duplicate top-level fields match exactly):\n",
        1,
    )[1].split("\n\nContext JSON:\n", 1)[0]
    example = json.loads(example_text)
    selected = example["ranked_paths"][0]

    assert example["required_lane"] == selected["lane"]
    assert example["required_mechanism"] == selected["mechanism"]
    assert example["must_inspect"] == selected["must_inspect"]
    assert example["must_inspect"] == [
        "sourcing_model/discovery.py::Router.discover_companies"
    ]
    assert code_editing.loop_direction_plan_contract_errors(
        code_editing.loop_direction_plan_from_mapping(example)
    ) == []


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
