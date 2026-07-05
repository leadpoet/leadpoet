#!/usr/bin/env python3
"""Verify Research Lab code-edit image candidate contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.canonical import sha256_json  # noqa: E402
from research_lab.code_editing import (  # noqa: E402
    CodeEditDraft,
    build_code_edit_auto_research_messages,
    build_loop_direction_planner_messages,
    build_plan_alignment_judge_messages,
    code_edit_no_viable_patch_reason,
    code_edit_candidate_manifest,
    code_edit_plan_alignment_errors,
    loop_direction_plan_from_mapping,
    normalize_unified_diff_text,
    parse_code_edit_repair_response,
    parse_code_edit_response,
    parse_code_edit_source_inspection_response,
    parse_loop_direction_plan_response,
    parse_plan_alignment_judge_response,
)
from research_lab.eval import PrivateModelArtifactManifest, SealedBenchmarkSet  # noqa: E402
from research_lab.eval.evaluator import evaluate_private_model_pair  # noqa: E402
from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest  # noqa: E402
from gateway.research_lab import scoring_worker as scoring_worker_module  # noqa: E402
from gateway.research_lab.code_build import CodeEditPatchApplyError  # noqa: E402
from gateway.research_lab.scoring_worker import (  # noqa: E402
    ResearchLabGatewayScoringWorker,
    StaleParentDuringScoring,
    _load_candidate_source_diff,
    _status_age_seconds,
)


def _manifest(name: str) -> PrivateModelArtifactManifest:
    payload = {
        "model_artifact_hash": sha256_json({"model": name}),
        "git_commit_sha": "abcdef1234567890",
        "image_digest": f"493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/{name}@sha256:"
        + ("a" if name == "parent" else "b") * 64,
        "config_hash": sha256_json({"config": name}),
        "component_registry_version": "components:v1",
        "scoring_adapter_version": "adapter:v1",
        "manifest_uri": f"s3://leadpoet-private-model-artifacts-493765492819/research-lab/test/{name}.json",
        "signature_ref": f"kms://test/{name}",
        "build_id": f"build-{name}",
    }
    payload["manifest_hash"] = sha256_json(payload)
    return PrivateModelArtifactManifest.from_mapping(payload)


def _valid_response() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "lane": "query_construction",
                    "expected_metric_effect": {
                        "sealed_icp_generalization": "Improves intent-query grounding across future sealed ICPs.",
                        "company_count": "Preserves multiple scoreable company outputs.",
                        "provider_error_rate": "No expected provider error-rate increase.",
                        "precision_recall_tradeoff": "Higher precision with bounded recall impact.",
                    },
                    "hypothesis": {
                        "failure_mode": "Queries are too broad.",
                        "mechanism": "Add ICP-specific signal terms.",
                        "expected_improvement": "More precise intent evidence.",
                        "risk": "May reduce recall.",
                        "predicted_delta": 1.0,
                    },
                    "code_edit": {
                        "target_files": ["sourcing_model/query_builder.py"],
                        "unified_diff": (
                            "diff --git a/sourcing_model/query_builder.py b/sourcing_model/query_builder.py\n"
                            "--- a/sourcing_model/query_builder.py\n"
                            "+++ b/sourcing_model/query_builder.py\n"
                            "@@ -1,2 +1,2 @@\n"
                            "-QUERY_SUFFIX = \"\"\n"
                            "+QUERY_SUFFIX = \" buying intent evidence\"\n"
                        ),
                        "redacted_summary": "Tighten query construction around intent evidence.",
                        "test_plan": "Run adapter metadata and public ICP smoke tests.",
                        "rollback_plan": "Revert the query suffix change.",
                    },
                }
            ]
        }
    )


def test_code_edit_parser_accepts_safe_diff() -> CodeEditDraft:
    drafts = parse_code_edit_response(_valid_response(), max_candidates=1)
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.target_files == ("sourcing_model/query_builder.py",)
    assert "buying intent evidence" in draft.unified_diff
    assert draft.expected_metric_effect["sealed_icp_generalization"].startswith("Improves intent-query")
    assert "multiple" in draft.expected_metric_effect["company_count"]
    return draft


def _provider_fallback_plan() -> dict[str, object]:
    return loop_direction_plan_from_mapping(
        {
            "schema_version": "1.0",
            "miner_focus_interpretation": "Reduce zero-company and HTTP provider failures.",
            "loop_goal": "Improve provider fallback without weakening ICP filters.",
            "required_lane": "provider_fallback",
            "required_mechanism": "Add bounded retry/backoff or secondary-provider fallback for zero-result/provider-error cases.",
            "generalization_claim": "This improves transient provider handling across future sealed ICPs.",
            "target_behavior": ["Retry transient provider failures.", "Fallback when primary provider returns zero usable companies."],
            "must_inspect": ["provider error handling", "zero-result routing"],
            "allowed_lanes": ["provider_fallback"],
            "disallowed_lanes": ["query_construction", "output_ranking"],
            "must_not_try": ["Do not remove LinkedIn employee-count clauses from Exa search queries."],
            "success_criteria": ["Diff touches fallback, retry, provider error handling, or zero-result routing code."],
            "novelty_requirements": ["Do not repeat exact source diff hashes."],
            "anti_overfit_checks": ["Preserve multiple scoreable company outputs.", "Do not tune to one public ICP."],
            "novelty_contrast": "Uses provider runtime classification, not query clause removal.",
            "ranked_paths": [
                {
                    "path_id": "provider_retry_backoff",
                    "lane": "provider_fallback",
                    "mechanism": "Classify retryable provider errors and add bounded retry/backoff.",
                }
            ],
            "selected_path_id": "provider_retry_backoff",
        }
    ).to_dict()


def test_loop_direction_plan_parser_and_prompt_contract() -> None:
    raw = json.dumps({"message": {"content": json.dumps(_provider_fallback_plan())}})
    plan = parse_loop_direction_plan_response(raw)
    assert plan.required_lane == "provider_fallback"
    assert plan.selected_path_id == "provider_retry_backoff"
    assert "future sealed ICPs" in plan.generalization_claim
    assert "scoreable" in " ".join(plan.anti_overfit_checks)
    assert "provider runtime" in plan.novelty_contrast
    assert plan.to_dict()["plan_hash"].startswith("sha256:")

    messages = build_loop_direction_planner_messages(
        ticket={"brief_public_summary": "provider fallback: zero companies and HTTP 4xx failures"},
        artifact_manifest={"manifest_hash": "sha256:test"},
        component_registry={},
        benchmark_public_summary={"public_icp_count": 10},
        runtime_source_index={"editable_files": ["sourcing_model/providers.py"]},
        budget_context={"requested_compute_budget_usd": 5.0},
        prior_attempts=[{"lane": "query_construction", "semantic_edit_summary": "Removed employee count clause."}],
    )
    content = "\n".join(message["content"] for message in messages)
    assert "ticket.brief_public_summary" in content
    assert "prior_attempts" in content
    assert "future sealed ICPs" in content
    assert "noisy validation signal" in content
    assert "one safest company" in content
    assert "Provider fallback paths must classify" in content


def test_code_edit_prompt_contract_discourages_public_icp_overfitting() -> None:
    messages = build_code_edit_auto_research_messages(
        ticket={"brief_public_summary": "Improve provider fallback for zero-company and HTTP 4xx failures."},
        artifact_manifest={"manifest_hash": "sha256:test"},
        component_registry={},
        benchmark_public_summary={"public_icp_count": 10},
        runtime_source_context={"editable_files": ["sourcing_model/query_builder.py"]},
        source_inspection_context={
            "read_files": ["sourcing_model/query_builder.py"],
            "results": [
                {
                    "path": "sourcing_model/query_builder.py",
                    "content": 'QUERY_SUFFIX = ""\n',
                }
            ],
        },
        budget_context={"requested_compute_budget_usd": 5.0},
        loop_direction_plan=_provider_fallback_plan(),
        max_candidates=1,
    )
    content = "\n".join(message["content"] for message in messages)
    assert "future sealed ICPs" in content
    assert "public benchmark quirk" in content
    assert "one safest company" in content
    assert "expected_metric_effect" in content
    assert "sealed_icp_generalization" in content
    assert "Provider fallback changes must classify" in content


def test_plan_alignment_judge_parser_and_prompt_contract() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    judge_messages = build_plan_alignment_judge_messages(
        loop_direction_plan=_provider_fallback_plan(),
        draft=draft,
        prior_attempts=[],
    )
    content = "\n".join(message["content"] for message in judge_messages)
    assert "LoopDirectionPlan" in content
    assert "unified_diff_hash" in content

    verdict = parse_plan_alignment_judge_response(
        json.dumps({"response": {"verdict": "pass", "reason": "aligned", "novel": True, "confidence": 0.9}})
    )
    assert verdict.verdict == "pass"
    assert verdict.novel is True
    try:
        parse_plan_alignment_judge_response(
            json.dumps({"verdict": "fail", "reason": "mentions hidden_icp material"})
        )
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("judge parser accepted forbidden event material")


def test_provider_fallback_plan_rejects_employee_count_query_only_diff() -> None:
    payload = {
        "candidates": [
            {
                "lane": "provider_fallback",
                "plan_path_id": "provider_retry_backoff",
                "plan_alignment": {"implements_required_mechanism": True},
                "hypothesis": {
                    "failure_mode": "Primary provider returns zero companies.",
                    "mechanism": "Claimed provider fallback.",
                    "expected_improvement": "Better recall.",
                    "risk": "May overfit.",
                    "predicted_delta": 1.0,
                },
                "code_edit": {
                    "target_files": ["sourcing_model/query_builder.py"],
                    "unified_diff": (
                        "diff --git a/sourcing_model/query_builder.py b/sourcing_model/query_builder.py\n"
                        "--- a/sourcing_model/query_builder.py\n"
                        "+++ b/sourcing_model/query_builder.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        "-QUERY = \"site:linkedin.com/company employee count 51-200\"\n"
                        "+QUERY = \"site:linkedin.com/company\"\n"
                    ),
                    "redacted_summary": "Remove LinkedIn employee count clause from the query.",
                    "test_plan": "Run public ICP smoke.",
                    "rollback_plan": "Revert query change.",
                },
            }
        ]
    }
    draft = parse_code_edit_response(json.dumps(payload), max_candidates=1)[0]
    errors = code_edit_plan_alignment_errors(
        draft,
        loop_direction_plan=_provider_fallback_plan(),
        prior_attempts=[],
        strict=True,
    )
    assert "provider_fallback_plan_but_employee_count_query_edit" in errors


def test_code_edit_parser_carries_plan_fields_and_detects_no_viable_patch() -> None:
    payload = json.loads(_valid_response())
    payload["candidates"][0]["plan_path_id"] = "query_intent_terms"
    payload["candidates"][0]["plan_alignment"] = {"implements_required_mechanism": True}
    draft = parse_code_edit_response(json.dumps(payload), max_candidates=1)[0]
    assert draft.plan_path_id == "query_intent_terms"
    assert draft.plan_alignment["implements_required_mechanism"] is True
    assert code_edit_no_viable_patch_reason(json.dumps({"no_viable_patch": True, "reason": "no safe source path"})) == "no safe source path"

    payload["candidates"][0].pop("expected_metric_effect", None)
    payload["candidates"][0]["hypothesis"]["expected_metric_effect"] = {
        "company_count": "Maintains broad enough candidate output for scoring."
    }
    draft = parse_code_edit_response(json.dumps(payload), max_candidates=1)[0]
    assert draft.expected_metric_effect["company_count"].startswith("Maintains")

    payload = json.loads(_valid_response())
    payload["candidates"][0].pop("expected_metric_effect", None)
    payload["candidates"][0]["code_edit"]["expectedMetricEffect"] = {
        "provider_error_rate": "Does not increase provider failures."
    }
    draft = parse_code_edit_response(json.dumps(payload), max_candidates=1)[0]
    assert draft.expected_metric_effect["provider_error_rate"].startswith("Does not")


def test_code_edit_parser_normalizes_markdown_wrapped_diff() -> None:
    payload = json.loads(_valid_response())
    code_edit = payload["candidates"][0]["code_edit"]
    code_edit["unified_diff"] = "Here is the patch:\n```diff\n" + code_edit["unified_diff"] + "```\n"
    wrapped = json.dumps(payload)
    drafts = parse_code_edit_response(wrapped, max_candidates=1)
    assert drafts[0].unified_diff.startswith("diff --git ")
    assert not drafts[0].unified_diff.startswith("Here is the patch")
    assert normalize_unified_diff_text("```diff\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n```").startswith("diff --git ")


def test_code_edit_parser_accepts_common_llm_wrapper_shapes() -> None:
    payload = json.loads(_valid_response())
    candidate = payload["candidates"][0]
    code_edit = candidate["code_edit"]
    diff = code_edit["unified_diff"]

    array_root = json.dumps(
        [
            {
                "lane": "query_construction",
                "hypothesis": candidate["hypothesis"],
                "edit": {
                    "target_file": "sourcing_model/query_builder.py",
                    "diff": diff,
                    "summary": "Array-root output with edit/diff aliases.",
                    "tests": "Run py_compile.",
                    "rollback": "Revert patch.",
                },
            }
        ]
    )
    array_drafts = parse_code_edit_response(array_root, max_candidates=1)
    assert array_drafts[0].target_files == ("sourcing_model/query_builder.py",)
    assert array_drafts[0].unified_diff.startswith("diff --git ")

    nested_content = json.dumps(
        {
            "message": {
                "content": json.dumps(
                    {
                        "candidate": {
                            "category": "provider_fallback",
                            "problem": "Fallback query logic is too narrow.",
                            "solution": "Use broader fallback evidence terms.",
                            "codeEdit": {
                                "files": [{"path": "sourcing_model/query_builder.py"}],
                                "gitDiff": diff,
                                "description": "Nested content string output.",
                                "testPlan": "Run public ICP smoke.",
                                "rollbackPlan": "Revert patch.",
                            },
                        }
                    }
                )
            }
        }
    )
    nested_drafts = parse_code_edit_response(nested_content, max_candidates=1)
    assert nested_drafts[0].lane == "provider_fallback"
    assert nested_drafts[0].target_files == ("sourcing_model/query_builder.py",)

    root_code_edit = json.dumps(
        {
            "codeEdit": {
                "targetFiles": ["sourcing_model/query_builder.py"],
                "unifiedDiff": diff,
                "summary": "Root codeEdit output.",
            }
        }
    )
    root_drafts = parse_code_edit_response(root_code_edit, max_candidates=1)
    assert root_drafts[0].target_files == ("sourcing_model/query_builder.py",)

    file_changes = json.dumps(
        {
            "final": {
                "changes": [
                    {
                        "path": "sourcing_model/query_builder.py",
                        "gitDiff": diff,
                        "summary": "File-change array output.",
                    }
                ]
            }
        }
    )
    file_change_drafts = parse_code_edit_response(file_changes, max_candidates=1)
    assert file_change_drafts[0].target_files == ("sourcing_model/query_builder.py",)
    assert file_change_drafts[0].unified_diff.startswith("diff --git ")

    nested_direct_diff = json.dumps({"message": {"content": "```diff\n" + diff + "```\n"}})
    nested_direct_drafts = parse_code_edit_response(nested_direct_diff, max_candidates=1)
    assert nested_direct_drafts[0].target_files == ("sourcing_model/query_builder.py",)

    direct_diff = "The JSON encoder failed, but here is the exact patch:\n```diff\n" + diff + "```\n"
    diff_drafts = parse_code_edit_response(direct_diff, max_candidates=1)
    assert diff_drafts[0].target_files == ("sourcing_model/query_builder.py",)
    assert diff_drafts[0].redacted_summary.startswith("Direct git diff")


def test_source_inspection_parser_accepts_common_llm_shapes() -> None:
    canonical = parse_code_edit_source_inspection_response(
        json.dumps({"requests": [{"operation": "read_file", "path": "sourcing_model/query_builder.py", "rationale": "inspect query builder"}]}),
        max_requests=4,
    )
    assert canonical[0].operation == "read_file"
    assert canonical[0].path == "sourcing_model/query_builder.py"

    array_root = parse_code_edit_source_inspection_response(
        json.dumps([{"action": "read", "file": "sourcing_model/query_builder.py", "reason": "inspect query builder"}]),
        max_requests=4,
    )
    assert array_root[0].operation == "read_file"
    assert array_root[0].path == "sourcing_model/query_builder.py"

    nested = parse_code_edit_source_inspection_response(
        json.dumps(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "sourceRequests": [
                                {"type": "search", "query": "build_query", "why": "locate query construction"}
                            ]
                        }
                    )
                }
            }
        ),
        max_requests=4,
    )
    assert nested[0].operation == "search"
    assert nested[0].query == "build_query"

    text_fallback = parse_code_edit_source_inspection_response(
        "Please read_file: sourcing_model/query_builder.py before drafting.",
        max_requests=4,
    )
    assert text_fallback[0].operation == "read_file"
    assert text_fallback[0].path == "sourcing_model/query_builder.py"


def test_code_edit_repair_parser_accepts_direct_code_edit() -> None:
    original = test_code_edit_parser_accepts_safe_diff()
    repaired_raw = json.dumps(
        {
            "code_edit": {
                "target_files": ["sourcing_model/query_builder.py"],
                "unified_diff": original.unified_diff.replace("buying intent evidence", "verified intent evidence"),
                "redacted_summary": "Repair hunk context only.",
                "test_plan": "Run py_compile.",
                "rollback_plan": "Revert the patch.",
            }
        }
    )
    repaired = parse_code_edit_repair_response(repaired_raw, original_draft=original)
    assert len(repaired) == 1
    assert repaired[0].failure_mode == original.failure_mode
    assert repaired[0].target_files == original.target_files
    assert repaired[0].expected_metric_effect == original.expected_metric_effect
    assert "verified intent evidence" in repaired[0].unified_diff


def test_code_edit_parser_rejects_apply_patch_format() -> None:
    payload = json.loads(_valid_response())
    payload["candidates"][0]["code_edit"]["unified_diff"] = (
        "*** Begin Patch\n"
        "*** Update File: sourcing_model/query_builder.py\n"
        "@@\n"
        "-QUERY_SUFFIX = \"\"\n"
        "+QUERY_SUFFIX = \" buying intent evidence\"\n"
        "*** End Patch\n"
    )
    try:
        parse_code_edit_response(json.dumps(payload), max_candidates=1)
    except ValueError as exc:
        assert "code_edit_uses_apply_patch_format" in str(exc) or "code_edit_requires_git_unified_diff" in str(exc)
    else:
        raise AssertionError("apply_patch format should be rejected before git apply")


def test_code_edit_parser_rejects_dependency_edit() -> None:
    raw = _valid_response().replace("sourcing_model/query_builder.py", "requirements.txt")
    try:
        parse_code_edit_response(raw, max_candidates=1)
    except ValueError as exc:
        assert "disallowed_repo_path" in str(exc) or "path_not_in_code_edit_allowlist" in str(exc)
    else:
        raise AssertionError("requirements.txt edit should be rejected")


def test_candidate_artifact_contract_requires_image_build() -> None:
    parent = _manifest("parent")
    candidate = _manifest("candidate")
    compat_manifest = {
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_artifact_hash": candidate.model_artifact_hash,
    }
    request = ResearchLabCandidateArtifactCreateRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        ticket_id="22222222-2222-4222-8222-222222222222",
        miner_hotkey="5EFakeMinerHotkey111111111111111111111111111",
        island="generalist",
        private_model_manifest=parent.to_dict(),
        candidate_patch_manifest=compat_manifest,
        candidate_model_manifest=candidate.to_dict(),
        candidate_source_diff_hash=sha256_json({"diff": "safe"}),
    )
    reparsed = ResearchLabCandidateArtifactCreateRequest.model_validate(request.model_dump(mode="json"))
    assert reparsed.candidate_kind == "image_build"

    try:
        ResearchLabCandidateArtifactCreateRequest(
            run_id="11111111-1111-4111-8111-111111111111",
            ticket_id="22222222-2222-4222-8222-222222222222",
            miner_hotkey="5EFakeMinerHotkey111111111111111111111111111",
            island="generalist",
            candidate_kind="patch",
            private_model_manifest=parent.to_dict(),
            candidate_patch_manifest=compat_manifest,
        )
    except ValueError:
        return
    raise AssertionError("patch candidate creation should be rejected")


def test_private_source_diff_artifact_loader() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    with tempfile.TemporaryDirectory(prefix="research-lab-diff-loader-") as tmp:
        path = Path(tmp) / "source_diff.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "artifact_type": "research_lab_code_edit_source_diff",
                    "source_diff_hash": source_diff_hash,
                    "unified_diff": draft.unified_diff,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        candidate = {
            "candidate_source_diff_hash": source_diff_hash,
            "candidate_build_doc": {"source_diff_artifact_uri": str(path)},
        }
        assert _load_candidate_source_diff(candidate) == draft.unified_diff

        candidate["candidate_source_diff_hash"] = sha256_json({"unified_diff": "different"})
        try:
            _load_candidate_source_diff(candidate)
        except Exception as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("source diff artifact hash mismatch should fail")


def test_postgrest_variable_fraction_timestamp_parses() -> None:
    parsed = _status_age_seconds("2026-06-30T00:22:44.47389+00:00")
    assert parsed is not None


async def test_stale_parent_rebase_queues_current_parent_candidate() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    old_parent = _manifest("parent")
    active_parent = _manifest("active-parent")
    candidate_manifest = _manifest("candidate")
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    with tempfile.TemporaryDirectory(prefix="research-lab-stale-rebase-") as tmp:
        diff_path = Path(tmp) / "source_diff.json"
        diff_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "artifact_type": "research_lab_code_edit_source_diff",
                    "source_diff_hash": source_diff_hash,
                    "unified_diff": draft.unified_diff,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        candidate = {
            "candidate_id": "candidate:" + "1" * 64,
            "run_id": "11111111-1111-4111-8111-111111111111",
            "ticket_id": "22222222-2222-4222-8222-222222222222",
            "receipt_id": None,
            "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
            "island": "generalist",
            "candidate_kind": "image_build",
            "parent_artifact_hash": old_parent.model_artifact_hash,
            "candidate_source_diff_hash": source_diff_hash,
            "candidate_build_doc": {"source_diff_artifact_uri": str(diff_path)},
            "candidate_patch_manifest": code_edit_candidate_manifest(
                draft=draft,
                parent_artifact_hash=old_parent.model_artifact_hash,
                candidate_artifact_hash=candidate_manifest.model_artifact_hash,
                candidate_model_manifest_hash=candidate_manifest.manifest_hash,
                source_diff_hash=source_diff_hash,
                build_doc_hash=sha256_json({"build": "old"}),
            ),
            "hypothesis_doc": {},
            "redacted_public_summary": "Rebase miner diff onto current parent.",
        }

        captured: dict[str, list[dict[str, object]]] = {
            "builds": [],
            "candidate_requests": [],
            "promotion_events": [],
            "evaluation_events": [],
            "dispatch_events": [],
        }

        class FakeBuilder:
            def __init__(self, config):
                self.config = config

            def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context=None):
                captured["builds"].append(
                    {
                        "parent": parent_artifact.model_artifact_hash,
                        "run_id": run_id,
                        "candidate_index": candidate_index,
                        "source_context": source_context,
                    }
                )
                build_doc = {
                    "schema_version": "1.1",
                    "source_diff_hash": source_diff_hash,
                    "candidate_model_manifest_hash": candidate_manifest.manifest_hash,
                }
                return SimpleNamespace(
                    build_doc=build_doc,
                    code_edit_manifest=code_edit_candidate_manifest(
                        draft=draft,
                        parent_artifact_hash=parent_artifact.model_artifact_hash,
                        candidate_artifact_hash=candidate_manifest.model_artifact_hash,
                        candidate_model_manifest_hash=candidate_manifest.manifest_hash,
                        source_diff_hash=source_diff_hash,
                        build_doc_hash=sha256_json(build_doc),
                    ),
                    candidate_model_manifest=candidate_manifest,
                    source_diff_hash=source_diff_hash,
                )

        async def fake_load_active_private_model(_config, *, register_bootstrap=False):
            return SimpleNamespace(artifact=active_parent, version_row=None)

        async def fake_create_candidate_artifact(request):
            captured["candidate_requests"].append(request.model_dump(mode="json"))
            return {"candidate_id": "candidate:" + candidate_manifest.model_artifact_hash.split(":", 1)[1]}, {}

        async def fake_select_many(*_args, **_kwargs):
            return []

        async def fake_promotion_event(**kwargs):
            captured["promotion_events"].append(kwargs)
            return kwargs

        async def fake_eval_event(**kwargs):
            captured["evaluation_events"].append(kwargs)
            return kwargs

        async def fake_dispatch_event(**kwargs):
            captured["dispatch_events"].append(kwargs)
            return kwargs

        original_builder = scoring_worker_module.CodeEditCandidateBuilder
        original_load_active = scoring_worker_module.load_active_private_model
        original_create_candidate = scoring_worker_module.create_candidate_artifact
        original_select_many = scoring_worker_module.select_many
        original_promotion_event = scoring_worker_module.create_candidate_promotion_event
        original_eval_event = scoring_worker_module.create_candidate_evaluation_event
        original_dispatch_event = scoring_worker_module.create_scoring_dispatch_event
        try:
            scoring_worker_module.CodeEditCandidateBuilder = FakeBuilder
            scoring_worker_module.load_active_private_model = fake_load_active_private_model
            scoring_worker_module.create_candidate_artifact = fake_create_candidate_artifact
            scoring_worker_module.select_many = fake_select_many
            scoring_worker_module.create_candidate_promotion_event = fake_promotion_event
            scoring_worker_module.create_candidate_evaluation_event = fake_eval_event
            scoring_worker_module.create_scoring_dispatch_event = fake_dispatch_event

            worker = ResearchLabGatewayScoringWorker(ResearchLabGatewayConfig(), worker_ref="test-worker")
            result = await worker._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=7,
                elapsed_seconds=lambda: 0.2,
            )
        finally:
            scoring_worker_module.CodeEditCandidateBuilder = original_builder
            scoring_worker_module.load_active_private_model = original_load_active
            scoring_worker_module.create_candidate_artifact = original_create_candidate
            scoring_worker_module.select_many = original_select_many
            scoring_worker_module.create_candidate_promotion_event = original_promotion_event
            scoring_worker_module.create_candidate_evaluation_event = original_eval_event
            scoring_worker_module.create_scoring_dispatch_event = original_dispatch_event

        assert result["status"] == "stale_parent_rebased_to_current"
        assert captured["builds"][0]["parent"] == active_parent.model_artifact_hash
        assert captured["candidate_requests"][0]["private_model_manifest"]["model_artifact_hash"] == active_parent.model_artifact_hash
        assert captured["candidate_requests"][0]["candidate_patch_manifest"]["parent_artifact_hash"] == active_parent.model_artifact_hash
        assert captured["promotion_events"][0]["event_type"] == "rebase_queued"
        assert captured["evaluation_events"][0]["candidate_status"] == "rejected"
        assert captured["dispatch_events"][0]["dispatch_status"] == "rejected"
        assert captured["dispatch_events"][0]["event_doc"]["reason"] == "stale_parent_rebased_to_current"


async def test_stale_parent_rebase_routes_apply_failure_to_repair() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    old_parent = _manifest("parent")
    active_parent = _manifest("active-parent")
    candidate_manifest = _manifest("candidate")
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    with tempfile.TemporaryDirectory(prefix="research-lab-stale-repair-") as tmp:
        diff_path = Path(tmp) / "source_diff.json"
        diff_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "artifact_type": "research_lab_code_edit_source_diff",
                    "source_diff_hash": source_diff_hash,
                    "unified_diff": draft.unified_diff,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        candidate = {
            "candidate_id": "candidate:" + "4" * 64,
            "run_id": "11111111-1111-4111-8111-111111111111",
            "ticket_id": "22222222-2222-4222-8222-222222222222",
            "receipt_id": None,
            "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
            "island": "generalist",
            "candidate_kind": "image_build",
            "parent_artifact_hash": old_parent.model_artifact_hash,
            "candidate_source_diff_hash": source_diff_hash,
            "candidate_build_doc": {"source_diff_artifact_uri": str(diff_path)},
            "candidate_patch_manifest": code_edit_candidate_manifest(
                draft=draft,
                parent_artifact_hash=old_parent.model_artifact_hash,
                candidate_artifact_hash=candidate_manifest.model_artifact_hash,
                candidate_model_manifest_hash=candidate_manifest.manifest_hash,
                source_diff_hash=source_diff_hash,
                build_doc_hash=sha256_json({"build": "old"}),
            ),
            "hypothesis_doc": {},
            "redacted_public_summary": "Repair stale miner diff onto current parent.",
        }
        captured: dict[str, list[dict[str, object]]] = {
            "builds": [],
            "candidate_requests": [],
            "promotion_events": [],
            "evaluation_events": [],
            "dispatch_events": [],
            "repairs": [],
        }

        def build_result(parent_artifact):
            build_doc = {
                "schema_version": "1.1",
                "source_diff_hash": source_diff_hash,
                "candidate_model_manifest_hash": candidate_manifest.manifest_hash,
            }
            return SimpleNamespace(
                build_doc=build_doc,
                code_edit_manifest=code_edit_candidate_manifest(
                    draft=draft,
                    parent_artifact_hash=parent_artifact.model_artifact_hash,
                    candidate_artifact_hash=candidate_manifest.model_artifact_hash,
                    candidate_model_manifest_hash=candidate_manifest.manifest_hash,
                    source_diff_hash=source_diff_hash,
                    build_doc_hash=sha256_json(build_doc),
                ),
                candidate_model_manifest=candidate_manifest,
                source_diff_hash=source_diff_hash,
            )

        class FailingBuilder:
            def __init__(self, config):
                self.config = config

            def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context=None):
                captured["builds"].append({"parent": parent_artifact.model_artifact_hash})
                raise CodeEditPatchApplyError("patch does not apply")

        async def fake_load_active_private_model(_config, *, register_bootstrap=False):
            return SimpleNamespace(artifact=active_parent, version_row=None)

        async def fake_create_candidate_artifact(request):
            captured["candidate_requests"].append(request.model_dump(mode="json"))
            return {"candidate_id": "candidate:" + candidate_manifest.model_artifact_hash.split(":", 1)[1]}, {}

        async def fake_select_many(*_args, **_kwargs):
            return []

        async def fake_promotion_event(**kwargs):
            captured["promotion_events"].append(kwargs)
            return kwargs

        async def fake_eval_event(**kwargs):
            captured["evaluation_events"].append(kwargs)
            return kwargs

        async def fake_dispatch_event(**kwargs):
            captured["dispatch_events"].append(kwargs)
            return kwargs

        async def fake_repair(self, candidate, *, active_artifact, original_error, run_id):
            captured["repairs"].append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "active_parent": active_artifact.model_artifact_hash,
                    "error": str(original_error),
                    "run_id": run_id,
                }
            )
            return draft, build_result(active_artifact)

        original_builder = scoring_worker_module.CodeEditCandidateBuilder
        original_load_active = scoring_worker_module.load_active_private_model
        original_create_candidate = scoring_worker_module.create_candidate_artifact
        original_select_many = scoring_worker_module.select_many
        original_promotion_event = scoring_worker_module.create_candidate_promotion_event
        original_eval_event = scoring_worker_module.create_candidate_evaluation_event
        original_dispatch_event = scoring_worker_module.create_scoring_dispatch_event
        original_repair = ResearchLabGatewayScoringWorker._repair_and_build_stale_candidate
        try:
            scoring_worker_module.CodeEditCandidateBuilder = FailingBuilder
            scoring_worker_module.load_active_private_model = fake_load_active_private_model
            scoring_worker_module.create_candidate_artifact = fake_create_candidate_artifact
            scoring_worker_module.select_many = fake_select_many
            scoring_worker_module.create_candidate_promotion_event = fake_promotion_event
            scoring_worker_module.create_candidate_evaluation_event = fake_eval_event
            scoring_worker_module.create_scoring_dispatch_event = fake_dispatch_event
            ResearchLabGatewayScoringWorker._repair_and_build_stale_candidate = fake_repair

            worker = ResearchLabGatewayScoringWorker(ResearchLabGatewayConfig(), worker_ref="test-worker")
            result = await worker._maybe_rebase_stale_candidate_before_scoring(
                candidate,
                evaluation_epoch=7,
                elapsed_seconds=lambda: 0.2,
            )
        finally:
            scoring_worker_module.CodeEditCandidateBuilder = original_builder
            scoring_worker_module.load_active_private_model = original_load_active
            scoring_worker_module.create_candidate_artifact = original_create_candidate
            scoring_worker_module.select_many = original_select_many
            scoring_worker_module.create_candidate_promotion_event = original_promotion_event
            scoring_worker_module.create_candidate_evaluation_event = original_eval_event
            scoring_worker_module.create_scoring_dispatch_event = original_dispatch_event
            ResearchLabGatewayScoringWorker._repair_and_build_stale_candidate = original_repair

        assert result["status"] == "stale_parent_rebased_to_current"
        assert result["repair_used"] is True
        assert captured["repairs"][0]["active_parent"] == active_parent.model_artifact_hash
        assert captured["promotion_events"][0]["event_doc"]["repair_used"] is True
        assert captured["candidate_requests"][0]["candidate_patch_manifest"]["parent_artifact_hash"] == active_parent.model_artifact_hash


async def test_evaluator_parent_freshness_abort_stops_before_bundle() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    parent = _manifest("parent")
    candidate = _manifest("candidate")
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    compat_manifest = code_edit_candidate_manifest(
        draft=draft,
        parent_artifact_hash=parent.model_artifact_hash,
        candidate_artifact_hash=candidate.model_artifact_hash,
        candidate_model_manifest_hash=candidate.manifest_hash,
        source_diff_hash=source_diff_hash,
        build_doc_hash=sha256_json({"build": "test"}),
    )
    benchmark = SealedBenchmarkSet(
        benchmark_id="benchmark:test",
        icp_set_hash=sha256_json({"icp": "set"}),
        split_ref="split:test",
        item_refs=("icp:test:1", "icp:test:2"),
        scoring_version="qualification-company-scorer:v1",
        hidden_plaintext_available=True,
    )
    calls = {"base": 0, "candidate": 0, "checks": []}

    async def parent_runner(_icp, _context):
        calls["base"] += 1
        return [{"company_name": "BaseCo", "company_website": "https://base.example"}]

    async def candidate_runner(_icp, _context):
        calls["candidate"] += 1
        return [{"company_name": "BetterCo", "company_website": "https://better.example"}]

    def scorer(companies, _icp, is_reference_model):
        return [50.0 if is_reference_model else 75.0 for _ in companies]

    class ParentChanged(RuntimeError):
        pass

    async def parent_check(progress):
        calls["checks"].append(dict(progress))
        if progress.get("phase") == "after_icp" and int(progress.get("completed_icp_count") or 0) == 1:
            raise ParentChanged("parent changed")

    try:
        await evaluate_private_model_pair(
            artifact_manifest=parent,
            candidate_artifact_manifest=candidate,
            benchmark=benchmark,
            patch_manifest=compat_manifest,
            benchmark_items=[
                {"icp_ref": "icp:test:1", "icp_hash": sha256_json({"i": 1}), "icp": {"industry": "Software"}},
                {"icp_ref": "icp:test:2", "icp_hash": sha256_json({"i": 2}), "icp": {"industry": "Software"}},
            ],
            base_runner=parent_runner,
            candidate_runner=candidate_runner,
            company_scorer=scorer,
            run_context={
                "run_id": "run-test",
                "ticket_id": "ticket-test",
                "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
                "island": "generalist",
                "evaluation_epoch": 1,
                "evaluator_version": "test",
                "evidence_bundle_refs": [],
                "execution_trace_ref": "trace:test",
                "cost_ledger_ref": "cost:test",
                "candidate_source_diff_hash": source_diff_hash,
                "candidate_build_ref": sha256_json({"build": "test"}),
                "signature_ref": "pending",
            },
            policy={"min_delta": 1.0, "min_successful_icps": 1},
            parent_freshness_check=parent_check,
        )
    except ParentChanged:
        pass
    else:
        raise AssertionError("parent freshness callback should abort scoring")
    assert calls["base"] == 1
    assert calls["candidate"] == 1
    assert calls["checks"][-1]["phase"] == "after_icp"


async def test_mid_scoring_stale_parent_requeues_without_score_bundle() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    old_parent = _manifest("parent")
    active_parent = _manifest("active-parent")
    candidate_manifest = _manifest("candidate")
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    candidate = {
        "candidate_id": "candidate:" + "7" * 64,
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "receipt_id": None,
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "island": "generalist",
        "candidate_kind": "image_build",
        "parent_artifact_hash": old_parent.model_artifact_hash,
        "private_model_manifest_doc": old_parent.to_dict(),
        "candidate_model_manifest_doc": candidate_manifest.to_dict(),
        "candidate_source_diff_hash": source_diff_hash,
        "candidate_build_doc": {"source_diff_hash": source_diff_hash},
        "candidate_patch_manifest": code_edit_candidate_manifest(
            draft=draft,
            parent_artifact_hash=old_parent.model_artifact_hash,
            candidate_artifact_hash=candidate_manifest.model_artifact_hash,
            candidate_model_manifest_hash=candidate_manifest.manifest_hash,
            source_diff_hash=source_diff_hash,
            build_doc_hash=sha256_json({"build": "old"}),
        ),
        "hypothesis_doc": {},
        "redacted_public_summary": "Stale during scoring.",
    }
    captured: dict[str, list[dict[str, object]]] = {
        "dispatch_events": [],
        "evaluation_events": [],
        "rebase_calls": [],
        "public_activity": [],
        "score_bundles": [],
    }

    async def fake_resolve_epoch(self):
        return 9

    async def fake_load_active_private_model(_config, *, register_bootstrap=False):
        return SimpleNamespace(artifact=old_parent, version_row=None)

    async def fake_fetch_window(**_kwargs):
        return SimpleNamespace(
            benchmark_id="benchmark:test",
            window_hash=sha256_json({"window": "test"}),
            split_ref="split:test",
            item_refs=("icp:test:1",),
            benchmark_items=[{"icp_ref": "icp:test:1", "icp_hash": sha256_json({"i": 1}), "icp": {"industry": "Software"}}],
        )

    async def fake_daily_window_and_gate(self, *, artifact):
        return await fake_fetch_window(), {}

    async def fake_create_rolling(_window):
        return {}

    async def fake_eval_event(**kwargs):
        captured["evaluation_events"].append(kwargs)
        return kwargs

    async def fake_dispatch_event(**kwargs):
        captured["dispatch_events"].append(kwargs)
        return kwargs

    async def fake_evaluate_pair(**_kwargs):
        raise StaleParentDuringScoring(
            active_artifact=active_parent,
            candidate_parent=old_parent.model_artifact_hash,
            progress={
                "phase": "after_icp",
                "completed_icp_count": 16,
                "last_icp_index": 15,
                "icp_ref": "icp:test:16",
            },
        )

    async def fake_score_bundle(_request):
        captured["score_bundles"].append({"called": True})
        raise AssertionError("partial stale scoring must not create a score bundle")

    class FakeDockerPrivateModelRunner:
        def __init__(self, _spec):
            self.spec = _spec

        async def __call__(self, _icp, _context):
            return []

    async def fake_rebase(self, candidate_arg, *, active_artifact, candidate_parent, evaluation_epoch, elapsed_seconds, stage, stale_progress=None):
        captured["rebase_calls"].append(
            {
                "candidate_id": candidate_arg["candidate_id"],
                "active_parent": active_artifact.model_artifact_hash,
                "candidate_parent": candidate_parent,
                "evaluation_epoch": evaluation_epoch,
                "stage": stage,
                "stale_progress": dict(stale_progress or {}),
            }
        )
        return {"status": "stale_parent_rebased_to_current", "derived_candidate_id": "candidate:" + "8" * 64}

    async def fake_finalize(self, _candidate):
        return None

    async def fake_public_activity(ticket_id, *, source_ref, reason, config):
        captured["public_activity"].append({"ticket_id": ticket_id, "source_ref": source_ref, "reason": reason})
        return None

    async def fake_write_audit(self, _epoch):
        return None

    async def fake_holdout_gate(self, *, artifact, window_hash):
        return {}

    original_resolve_epoch = ResearchLabGatewayScoringWorker._resolve_evaluation_epoch
    original_load_active = scoring_worker_module.load_active_private_model
    original_fetch_window = scoring_worker_module.fetch_rolling_icp_window
    original_create_rolling = scoring_worker_module.create_rolling_icp_window
    original_eval_event = scoring_worker_module.create_candidate_evaluation_event
    original_dispatch_event = scoring_worker_module.create_scoring_dispatch_event
    original_evaluate_pair = scoring_worker_module.evaluate_private_model_pair
    original_score_bundle = scoring_worker_module.create_score_bundle
    original_docker_runner = scoring_worker_module.DockerPrivateModelRunner
    original_rebase = ResearchLabGatewayScoringWorker._queue_stale_parent_rebase
    original_finalize = ResearchLabGatewayScoringWorker._maybe_finalize_candidate_receipt
    original_public_activity = scoring_worker_module.safe_project_public_loop_activity
    original_write_audit = ResearchLabGatewayScoringWorker._write_audit_bundle
    original_holdout_gate = ResearchLabGatewayScoringWorker._candidate_private_holdout_gate
    original_daily_window_and_gate = ResearchLabGatewayScoringWorker._daily_candidate_scoring_window_and_gate
    try:
        ResearchLabGatewayScoringWorker._resolve_evaluation_epoch = fake_resolve_epoch
        scoring_worker_module.load_active_private_model = fake_load_active_private_model
        scoring_worker_module.fetch_rolling_icp_window = fake_fetch_window
        scoring_worker_module.create_rolling_icp_window = fake_create_rolling
        scoring_worker_module.create_candidate_evaluation_event = fake_eval_event
        scoring_worker_module.create_scoring_dispatch_event = fake_dispatch_event
        scoring_worker_module.evaluate_private_model_pair = fake_evaluate_pair
        scoring_worker_module.create_score_bundle = fake_score_bundle
        scoring_worker_module.DockerPrivateModelRunner = FakeDockerPrivateModelRunner
        ResearchLabGatewayScoringWorker._queue_stale_parent_rebase = fake_rebase
        ResearchLabGatewayScoringWorker._maybe_finalize_candidate_receipt = fake_finalize
        scoring_worker_module.safe_project_public_loop_activity = fake_public_activity
        ResearchLabGatewayScoringWorker._write_audit_bundle = fake_write_audit
        ResearchLabGatewayScoringWorker._candidate_private_holdout_gate = fake_holdout_gate
        ResearchLabGatewayScoringWorker._daily_candidate_scoring_window_and_gate = fake_daily_window_and_gate

        worker = ResearchLabGatewayScoringWorker(ResearchLabGatewayConfig(), worker_ref="test-worker")
        await worker._score_candidate(candidate)
    finally:
        ResearchLabGatewayScoringWorker._resolve_evaluation_epoch = original_resolve_epoch
        scoring_worker_module.load_active_private_model = original_load_active
        scoring_worker_module.fetch_rolling_icp_window = original_fetch_window
        scoring_worker_module.create_rolling_icp_window = original_create_rolling
        scoring_worker_module.create_candidate_evaluation_event = original_eval_event
        scoring_worker_module.create_scoring_dispatch_event = original_dispatch_event
        scoring_worker_module.evaluate_private_model_pair = original_evaluate_pair
        scoring_worker_module.create_score_bundle = original_score_bundle
        scoring_worker_module.DockerPrivateModelRunner = original_docker_runner
        ResearchLabGatewayScoringWorker._queue_stale_parent_rebase = original_rebase
        ResearchLabGatewayScoringWorker._maybe_finalize_candidate_receipt = original_finalize
        scoring_worker_module.safe_project_public_loop_activity = original_public_activity
        ResearchLabGatewayScoringWorker._write_audit_bundle = original_write_audit
        ResearchLabGatewayScoringWorker._candidate_private_holdout_gate = original_holdout_gate
        ResearchLabGatewayScoringWorker._daily_candidate_scoring_window_and_gate = original_daily_window_and_gate

    assert not captured["score_bundles"]
    assert captured["rebase_calls"][0]["stage"] == "during_scoring_parent_changed"
    assert captured["rebase_calls"][0]["active_parent"] == active_parent.model_artifact_hash
    assert captured["rebase_calls"][0]["stale_progress"]["completed_icp_count"] == 16
    assert captured["public_activity"][0]["reason"] == "stale_parent_rebased_to_current"


async def test_stale_parent_rebase_depth_failure_preserves_reimbursement() -> None:
    old_parent = _manifest("parent")
    active_parent = _manifest("active-parent")
    candidate = {
        "candidate_id": "candidate:" + "9" * 64,
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "receipt_id": None,
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "island": "generalist",
        "candidate_kind": "image_build",
        "parent_artifact_hash": old_parent.model_artifact_hash,
        "candidate_build_doc": {
            "stale_parent_rebase": {
                "depth": 3,
            }
        },
        "candidate_patch_manifest": {},
        "hypothesis_doc": {},
        "redacted_public_summary": "Depth cap candidate.",
    }
    captured: dict[str, list[dict[str, object]]] = {
        "promotion_events": [],
        "evaluation_events": [],
        "dispatch_events": [],
    }

    async def fake_promotion_event(**kwargs):
        captured["promotion_events"].append(kwargs)
        return kwargs

    async def fake_eval_event(**kwargs):
        captured["evaluation_events"].append(kwargs)
        return kwargs

    async def fake_dispatch_event(**kwargs):
        captured["dispatch_events"].append(kwargs)
        return kwargs

    original_promotion_event = scoring_worker_module.create_candidate_promotion_event
    original_eval_event = scoring_worker_module.create_candidate_evaluation_event
    original_dispatch_event = scoring_worker_module.create_scoring_dispatch_event
    try:
        scoring_worker_module.create_candidate_promotion_event = fake_promotion_event
        scoring_worker_module.create_candidate_evaluation_event = fake_eval_event
        scoring_worker_module.create_scoring_dispatch_event = fake_dispatch_event

        worker = ResearchLabGatewayScoringWorker(
            ResearchLabGatewayConfig(stale_parent_rebase_max_depth=3),
            worker_ref="test-worker",
        )
        result = await worker._queue_stale_parent_rebase(
            candidate,
            active_artifact=active_parent,
            candidate_parent=old_parent.model_artifact_hash,
            evaluation_epoch=7,
            elapsed_seconds=1.5,
            stage="during_scoring_parent_changed",
            stale_progress={"phase": "after_icp", "completed_icp_count": 16},
        )
    finally:
        scoring_worker_module.create_candidate_promotion_event = original_promotion_event
        scoring_worker_module.create_candidate_evaluation_event = original_eval_event
        scoring_worker_module.create_scoring_dispatch_event = original_dispatch_event

    assert result["status"] == "stale_parent_rebase_failed"
    event_doc = captured["evaluation_events"][0]["event_doc"]
    assert event_doc["failure_class"] == "stale_parent_rebase_depth_exceeded"
    assert event_doc["reimbursement_preserved"] is True
    assert event_doc["reimbursement_source"] == "hosted_loop_completion"
    assert event_doc["stale_progress"]["completed_icp_count"] == 16


async def test_scoring_worker_terminal_decision_events() -> None:
    candidate = {
        "candidate_id": "candidate:" + "a" * 64,
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "candidate_kind": "image_build",
        "parent_artifact_hash": "sha256:" + "1" * 64,
    }
    score_bundle_row = {"score_bundle_id": "score_bundle:" + "b" * 64}
    score_bundle = {
        "parent_artifact_hash": candidate["parent_artifact_hash"],
        "icp_set_hash": "sha256:" + "c" * 64,
        "aggregates": {"mean_delta": 2.0, "delta_lcb": 1.0},
        "scoring_health": {
            "schema_version": "1.0",
            "health_status": "degraded",
            "reference_runtime_success_rate": 0.2,
            "candidate_runtime_success_rate": 0.9,
            "reference_zero_company_rate": 0.8,
            "candidate_zero_company_rate": 0.0,
            "provider_error_rate": 0.4,
            "timeout_rate": 0.0,
        },
    }
    captured: list[dict[str, object]] = []

    async def fake_promotion_event(**kwargs):
        captured.append(kwargs)
        return kwargs

    original_promotion_event = scoring_worker_module.create_candidate_promotion_event
    try:
        scoring_worker_module.create_candidate_promotion_event = fake_promotion_event
        worker = ResearchLabGatewayScoringWorker(
            ResearchLabGatewayConfig(scoring_health_gate_enabled=True),
            worker_ref="test-worker",
        )
        gate_result = worker._scoring_health_gate_result(score_bundle)
        assert gate_result["decision"] == "quarantined"
        assert gate_result["would_quarantine"] is True

        public_result = await worker._record_public_holdout_rejected(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            gate_result={
                "gate_type": "public_score_before_private_holdout",
                "decision": "rejected_before_private_holdout",
                "baseline_benchmark_bundle_id": "baseline:test",
                "baseline_public_score": 80.0,
                "candidate_public_score": 60.0,
                "paired_base_public_score": 80.0,
                "public_icp_count": 2,
                "private_holdout_icp_count": 18,
                "private_holdout_evaluated": False,
            },
        )
        health_result = await worker._record_scoring_health_quarantined(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            scoring_health_gate=gate_result,
        )
    finally:
        scoring_worker_module.create_candidate_promotion_event = original_promotion_event

    assert public_result["status"] == "rejected_public_holdout_gate"
    assert health_result["status"] == "scoring_health_quarantined"
    assert [event["event_type"] for event in captured] == [
        "promotion_checked",
        "public_holdout_rejected",
        "promotion_checked",
        "scoring_health_quarantined",
    ]
    assert captured[1]["promotion_status"] == "rejected"
    assert captured[3]["event_doc"]["scoring_health_gate"]["decision"] == "quarantined"


async def test_image_build_score_bundle_contract(draft: CodeEditDraft) -> None:
    parent = _manifest("parent")
    candidate = _manifest("candidate")
    source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    build_doc = {
        "schema_version": "1.0",
        "candidate_kind": "image_build",
        "source_diff_hash": source_diff_hash,
        "candidate_model_manifest_hash": candidate.manifest_hash,
    }
    compat_manifest = code_edit_candidate_manifest(
        draft=draft,
        parent_artifact_hash=parent.model_artifact_hash,
        candidate_artifact_hash=candidate.model_artifact_hash,
        candidate_model_manifest_hash=candidate.manifest_hash,
        source_diff_hash=source_diff_hash,
        build_doc_hash=sha256_json(build_doc),
    )
    assert "buying intent evidence" not in json.dumps(compat_manifest)

    benchmark = SealedBenchmarkSet(
        benchmark_id="benchmark:test",
        icp_set_hash=sha256_json({"icp": "set"}),
        split_ref="split:test",
        item_refs=("icp:test:1",),
        scoring_version="qualification-company-scorer:v1",
        hidden_plaintext_available=True,
    )

    async def parent_runner(_icp, _context):
        return [{"company_name": "BaseCo", "company_website": "https://base.example"}]

    async def candidate_runner(_icp, context):
        assert "patch" not in context
        return [{"company_name": "BetterCo", "company_website": "https://better.example"}]

    def scorer(companies, _icp, is_reference_model):
        return [50.0 if is_reference_model else 75.0 for _ in companies]

    bundle = await evaluate_private_model_pair(
        artifact_manifest=parent,
        candidate_artifact_manifest=candidate,
        benchmark=benchmark,
        patch_manifest=compat_manifest,
        benchmark_items=[{"icp_ref": "icp:test:1", "icp_hash": sha256_json({"i": 1}), "icp": {"industry": "Software"}}],
        base_runner=parent_runner,
        candidate_runner=candidate_runner,
        company_scorer=scorer,
        run_context={
            "run_id": "run-test",
            "ticket_id": "ticket-test",
            "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
            "island": "generalist",
            "evaluation_epoch": 1,
            "evaluator_version": "test",
            "evidence_bundle_refs": [],
            "execution_trace_ref": "trace:test",
            "cost_ledger_ref": "cost:test",
            "candidate_source_diff_hash": source_diff_hash,
            "candidate_build_ref": sha256_json(build_doc),
            "signature_ref": "pending",
        },
        policy={"min_delta": 1.0, "min_successful_icps": 1},
    )
    assert bundle["parent_artifact_hash"] == parent.model_artifact_hash
    assert bundle["candidate_artifact_hash"] == candidate.model_artifact_hash
    assert bundle["candidate_model_manifest_hash"] == candidate.manifest_hash
    assert bundle["candidate_source_diff_hash"] == source_diff_hash
    assert bundle["aggregates"]["mean_delta"] > 0


def main() -> None:
    draft = test_code_edit_parser_accepts_safe_diff()
    test_loop_direction_plan_parser_and_prompt_contract()
    test_code_edit_prompt_contract_discourages_public_icp_overfitting()
    test_plan_alignment_judge_parser_and_prompt_contract()
    test_provider_fallback_plan_rejects_employee_count_query_only_diff()
    test_code_edit_parser_carries_plan_fields_and_detects_no_viable_patch()
    test_code_edit_parser_normalizes_markdown_wrapped_diff()
    test_code_edit_parser_accepts_common_llm_wrapper_shapes()
    test_source_inspection_parser_accepts_common_llm_shapes()
    test_code_edit_repair_parser_accepts_direct_code_edit()
    test_code_edit_parser_rejects_apply_patch_format()
    test_code_edit_parser_rejects_dependency_edit()
    test_candidate_artifact_contract_requires_image_build()
    test_private_source_diff_artifact_loader()
    test_postgrest_variable_fraction_timestamp_parses()
    asyncio.run(test_stale_parent_rebase_queues_current_parent_candidate())
    asyncio.run(test_stale_parent_rebase_routes_apply_failure_to_repair())
    asyncio.run(test_evaluator_parent_freshness_abort_stops_before_bundle())
    asyncio.run(test_mid_scoring_stale_parent_requeues_without_score_bundle())
    asyncio.run(test_stale_parent_rebase_depth_failure_preserves_reimbursement())
    asyncio.run(test_scoring_worker_terminal_decision_events())
    asyncio.run(test_image_build_score_bundle_contract(draft))
    print("research_lab_code_edit_pipeline_verifier: ok")


if __name__ == "__main__":
    main()
