from pathlib import Path
import re
import subprocess

import pytest

from gateway.tee import protected_workflows as protected_workflows_module
from gateway.tee.protected_workflows import (
    PROTECTED_SYMBOLS,
    ProtectedWorkflowError,
    build_manifest,
    load_manifest,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "gateway" / "tee" / "protected_workflows.json"


def test_committed_protected_workflow_manifest_matches_source():
    manifest = load_manifest(MANIFEST_PATH)
    verify_manifest(ROOT, manifest)
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["baseline_commit"])
    protected_source = manifest["protected_source_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", protected_source)
    subprocess.run(
        ["git", "cat-file", "-e", protected_source + "^{commit}"],
        cwd=ROOT,
        check=True,
    )
    assert len(manifest["entries"]) == sum(len(items) for items in PROTECTED_SYMBOLS.values())


def test_private_artifact_signature_boundary_is_protected():
    assert {
        "_REVIEWED_CONSUMER_MANIFEST_PAIRS",
        "_verify_consumer_contract_manifest",
        "verify_private_artifact_manifest_signature",
        "_verify_private_artifact_manifest_signature_cached",
        "_verify_private_artifact_manifest_signature_uncached",
    } <= set(PROTECTED_SYMBOLS["research_lab/eval/private_runtime.py"])
    assert {
        "DEFAULT_PRIVATE_TEST_CMD",
        "DEFAULT_PRIVATE_BUILD_CMD",
    } <= set(PROTECTED_SYMBOLS["gateway/research_lab/config.py"])
    assert {
        "_verify_built_candidate_artifact",
        "CodeEditCandidateBuilder._build_under_deadline",
    } <= set(PROTECTED_SYMBOLS["gateway/research_lab/code_build.py"])
    assert {
        "ResearchLabPromotionController._promote_built_image_candidate",
        "_load_valid_artifact",
    } <= set(PROTECTED_SYMBOLS["gateway/research_lab/promotion.py"])


def test_scoring_receipt_failure_policy_is_protected():
    assert "_local_failed_receipt_hashes" in PROTECTED_SYMBOLS[
        "gateway/research_lab/attested_scoring_v2.py"
    ]


def test_git_tree_source_authority_boundaries_are_protected():
    assert {
        "_GIT_STRUCTURAL_DIFF_PREFIXES",
        "git_diff_structural_metadata",
    } <= set(PROTECTED_SYMBOLS["research_lab/code_editing.py"])
    assert "validate_private_code_edit_diff_artifact" in PROTECTED_SYMBOLS[
        "gateway/research_lab/code_build.py"
    ]
    assert "_push_candidate_source_diff_to_repo" in PROTECTED_SYMBOLS[
        "gateway/research_lab/promotion.py"
    ]
    assert "_load_candidate_source_diff" in PROTECTED_SYMBOLS[
        "gateway/research_lab/scoring_worker.py"
    ]
    assert {
        "_HostGitTreeRepository",
        "_HostCandidateBuilder",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/autoresearch_executor_v2.py"])


def test_model_sandbox_launcher_boundaries_are_protected():
    assert {
        "MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES",
        "_pid_is_direct_cgroup_member",
        "_normalized_cgroup_relative_path",
        "_current_cgroup_path",
        "_current_cgroup_v1_paths",
        "_prepare_model_sandbox_cgroup_v1",
        "_runsc_run_command",
        "prepare_model_sandbox_cgroup_v2",
        "model_sandbox_job_cgroup_path",
        "_oci_config",
        "RunscModelSandboxV2",
    } <= set(PROTECTED_SYMBOLS["gateway/tee/model_sandbox_v2.py"])


def test_protected_manifest_detects_logic_change(tmp_path: Path):
    committed = load_manifest(MANIFEST_PATH)
    manifest = build_manifest(
        ROOT,
        baseline_commit=committed["baseline_commit"],
        protected_source_commit=committed["protected_source_commit"],
    )
    copied_root = tmp_path / "repo"
    for relative_path in PROTECTED_SYMBOLS:
        source = ROOT / relative_path
        destination = copied_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    target = copied_root / "research_lab" / "eval" / "promotion_metric.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "return PromotionGateDecision(",
            "assert threshold_points >= 0\n    return PromotionGateDecision(",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtectedWorkflowError, match="promotion_metric.py"):
        verify_manifest(copied_root, manifest)


def test_protected_manifest_detects_policy_constant_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    copied_root = tmp_path / "repo"
    policy_path = copied_root / "policy.py"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text('POLICY_VERSION = "v1"\n', encoding="utf-8")
    monkeypatch.setattr(
        protected_workflows_module,
        "PROTECTED_SYMBOLS",
        {"policy.py": ("POLICY_VERSION",)},
    )
    manifest = build_manifest(
        copied_root,
        baseline_commit="1" * 40,
        protected_source_commit="2" * 40,
    )

    policy_path.write_text('POLICY_VERSION = "v2"\n', encoding="utf-8")

    with pytest.raises(ProtectedWorkflowError, match="policy.py:POLICY_VERSION"):
        verify_manifest(copied_root, manifest)
