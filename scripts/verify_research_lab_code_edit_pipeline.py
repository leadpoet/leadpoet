#!/usr/bin/env python3
"""Verify Research Lab code-edit image candidate contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.canonical import sha256_json  # noqa: E402
from research_lab.code_editing import (  # noqa: E402
    CodeEditDraft,
    code_edit_candidate_manifest,
    normalize_unified_diff_text,
    parse_code_edit_repair_response,
    parse_code_edit_response,
)
from research_lab.eval import PrivateModelArtifactManifest, SealedBenchmarkSet  # noqa: E402
from research_lab.eval.evaluator import evaluate_private_model_pair  # noqa: E402
from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest  # noqa: E402


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
    return draft


def test_code_edit_parser_normalizes_markdown_wrapped_diff() -> None:
    payload = json.loads(_valid_response())
    code_edit = payload["candidates"][0]["code_edit"]
    code_edit["unified_diff"] = "Here is the patch:\n```diff\n" + code_edit["unified_diff"] + "```\n"
    wrapped = json.dumps(payload)
    drafts = parse_code_edit_response(wrapped, max_candidates=1)
    assert drafts[0].unified_diff.startswith("diff --git ")
    assert not drafts[0].unified_diff.startswith("Here is the patch")
    assert normalize_unified_diff_text("```diff\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n```").startswith("diff --git ")


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
    test_code_edit_parser_normalizes_markdown_wrapped_diff()
    test_code_edit_repair_parser_accepts_direct_code_edit()
    test_code_edit_parser_rejects_apply_patch_format()
    test_code_edit_parser_rejects_dependency_edit()
    test_candidate_artifact_contract_requires_image_build()
    asyncio.run(test_image_build_score_bundle_contract(draft))
    print("research_lab_code_edit_pipeline_verifier: ok")


if __name__ == "__main__":
    main()
