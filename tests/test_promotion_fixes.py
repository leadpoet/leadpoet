"""Tests for the Research Lab promotion fail-closed fixes and the score-only
merge path (bugs 2/3/4/24, N3; score-only design decision 2026-07-02).

Score-only promotion (commit 3aaee73c): a candidate promotes purely on its
stored final score vs the stored daily baseline aggregate. Provider/runtime
health, quarantine bookkeeping, and provider-exclusion audit fields are
recorded for observability but never veto the merge.

Covers, with fake store rows (no live Supabase):
  * bug #2  — lineage fail-closed: read error vs genuinely-empty lineage vs
    flag-gated bootstrap registration; manifest hash mismatch raises.
  * bug #3  — reconcile re-activates the newest superseded version.
  * N3      — unavailable basis is an explicit rejection, not 0.0-below-threshold.
  * score-only merge path — health/quarantine/baseline-doc state cannot hold
    or block the decision; provider exclusions never adjust the basis.
  * champion reward windows start at the live epoch at creation time, never
    the bundle's scoring epoch (the 2026-07-02 backdating incident).
  * bug #24 — pending champion reward reconciler happy/retry paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import gateway.research_lab.promotion as promotion
from gateway.research_lab import attested_v2_store, model_authority_v2, v2_authority
from gateway.research_lab.config import DEFAULT_PRIVATE_REPO_BRANCH
from gateway.research_lab.tee_protocol import ResearchLabTeeProtocolError
import gateway.research_lab.store as store_module
from gateway.research_lab.promotion import (
    ActiveManifestHashMismatchError,
    ActivePrivateModel,
    NoActivePrivateModelVersionError,
    PrivateModelLineageUnavailableError,
    ResearchLabPromotionController,
    load_active_private_model,
    promotion_improvement_metric,
    reconcile_active_private_model_lineage,
    reconcile_failed_private_source_pushes,
    reconcile_pending_champion_rewards,
    reconcile_source_add_leg2_reward_activations,
    sync_active_model_to_repo_head,
)
from research_lab.canonical import sha256_json
from research_lab.eval.promotion_metric import (
    PAIRED_LCB_PROMOTION_METRIC_VERSION,
)
from gateway.research_lab.source_add_llm_judge import SourceAddJudgeVerdict
from research_lab.sourcing_model_contract_check import (
    SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
    SEMANTIC_COMPATIBILITY_CONSUMER_API_V1,
    SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1,
    reviewed_consumer_snapshots,
    semantic_compatibility_policy_identity_v1,
)
from tests.private_model_artifact_fixtures import DEFAULT_CONSUMER_CONTRACT_ID


_TEST_CONSUMER_SNAPSHOT = reviewed_consumer_snapshots()[
    DEFAULT_CONSUMER_CONTRACT_ID
]
_REAL_PREFLIGHT_PRIVATE_MODEL_ACTIVATION = (
    promotion._preflight_private_model_activation
)
_REAL_ACTIVATE_PRIVATE_MODEL_GENERATION = (
    promotion._activate_private_model_generation
)


def _git(cmd, *, cwd=None):
    return subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _private_source_push_fixture():
    unified_diff = (
        "diff --git a/sourcing_model.py b/sourcing_model.py\n"
        "--- a/sourcing_model.py\n"
        "+++ b/sourcing_model.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    artifact_payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_source_diff",
        "run_id": "run-private-source-push",
        "candidate_index": 0,
        "parent_artifact_hash": "sha256:" + "1" * 64,
        "parent_manifest_hash": "sha256:" + "2" * 64,
        "source_diff_hash": sha256_json({"unified_diff": unified_diff}),
        "target_files": ["sourcing_model.py"],
        "unified_diff": unified_diff,
        "draft_hash": "sha256:" + "3" * 64,
    }
    artifact = {
        **artifact_payload,
        "artifact_hash": sha256_json(artifact_payload),
    }
    candidate_manifest_payload = {
        "model_artifact_hash": "sha256:" + "4" * 64,
        "git_commit_sha": "5" * 40,
        "image_digest": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/candidate@sha256:"
            + "6" * 64
        ),
        "config_hash": "sha256:" + "7" * 64,
        "component_registry_version": "1",
        "scoring_adapter_version": "1",
        "manifest_uri": "s3://fixture/candidate.json",
        "signature_ref": "kms:fixture",
        "build_id": "fixture-build",
    }
    candidate_manifest = {
        **candidate_manifest_payload,
        "manifest_hash": sha256_json(candidate_manifest_payload),
    }
    build_payload = {
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "parent_manifest_hash": artifact["parent_manifest_hash"],
        "candidate_model_artifact_hash": candidate_manifest[
            "model_artifact_hash"
        ],
        "candidate_model_manifest_hash": candidate_manifest["manifest_hash"],
        "source_diff_hash": artifact["source_diff_hash"],
        "source_diff_artifact_uri": "s3://fixture/candidates/run-private-source-push/0/source_diff.json",
        "source_diff_artifact_hash": artifact["artifact_hash"],
        "changed_files": ["sourcing_model.py"],
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
        "loop_direction_plan_hash": "sha256:" + "8" * 64,
        "selected_path_id": "fixture-path",
        "plan_alignment": {"passes": True},
        "conditional_validation_policy": {"mode": "on"},
        "loop_node_id": "tree-node:" + "9" * 64,
    }
    patch_payload = {
        "candidate_kind": "image_build",
        "patch_type": "IMAGE_BUILD",
        "target_component_id": "private_model_source_tree",
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "candidate_artifact_hash": candidate_manifest["model_artifact_hash"],
        "candidate_model_manifest_hash": candidate_manifest["manifest_hash"],
        "patch_payload_hash": artifact["source_diff_hash"],
        "candidate_source_diff_hash": artifact["source_diff_hash"],
        "candidate_build_doc_hash": build_doc["build_doc_hash"],
        "redacted_summary": "fixture",
        "validation_result": "passed",
        "patch_doc": {"target_files": list(artifact["target_files"])},
    }
    patch_manifest = {
        **patch_payload,
        "manifest_hash": sha256_json(patch_payload),
    }
    return artifact, build_doc, patch_manifest, candidate_manifest


def test_private_source_push_verifies_artifact_before_git_and_pushes_exact_diff(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    _git(["init", "-q", "-b", "main"], cwd=source)
    _git(["config", "user.name", "Fixture"], cwd=source)
    _git(["config", "user.email", "fixture@example.test"], cwd=source)
    (source / "sourcing_model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "sourcing_model.py"], cwd=source)
    _git(["commit", "-q", "-m", "root"], cwd=source)
    active_sha = _git(["rev-parse", "HEAD"], cwd=source)
    remote = tmp_path / "remote.git"
    _git(["clone", "-q", "--bare", str(source), str(remote)])

    artifact, build_doc, patch_manifest, candidate_manifest = (
        _private_source_push_fixture()
    )
    original_run_command = promotion._run_command

    def run_command(cmd, **kwargs):
        if cmd[:3] == ["aws", "s3", "cp"]:
            return json.dumps(artifact, sort_keys=True)
        return original_run_command(cmd, **kwargs)

    monkeypatch.setattr(promotion, "_run_command", run_command)
    result = promotion._push_candidate_source_diff_to_repo(
        repo_url=str(remote),
        branch_name="main",
        active_git_commit_sha=active_sha,
        candidate_id="candidate:fixture",
        score_bundle_id="bundle:fixture",
        candidate_build_doc=build_doc,
        candidate_patch_manifest=patch_manifest,
        candidate_model_manifest_doc=candidate_manifest,
        expected_candidate_patch_hash=sha256_json(patch_manifest),
        expected_source_diff_hash=artifact["source_diff_hash"],
        expected_parent_artifact_hash=artifact["parent_artifact_hash"],
        expected_run_id=artifact["run_id"],
    )

    assert result["status"] == "pushed"
    assert result["target_files"] == ["sourcing_model.py"]
    assert _git(
        ["--git-dir", str(remote), "show", "main:sourcing_model.py"]
    ) == "VALUE = 2"


def test_private_source_push_normalizes_legacy_depth_two_cumulative_targets(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    (source / "gateway").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], cwd=source)
    _git(["config", "user.name", "Fixture"], cwd=source)
    _git(["config", "user.email", "fixture@example.test"], cwd=source)
    (source / "sourcing_model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "gateway" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "sourcing_model.py", "gateway/module.py"], cwd=source)
    _git(["commit", "-q", "-m", "root"], cwd=source)
    active_sha = _git(["rev-parse", "HEAD"], cwd=source)
    remote = tmp_path / "remote.git"
    _git(["clone", "-q", "--bare", str(source), str(remote)])

    base_artifact, _base_build, _base_patch, candidate_manifest = (
        _private_source_push_fixture()
    )
    first_patch = (
        "diff --git a/sourcing_model.py b/sourcing_model.py\n"
        "--- a/sourcing_model.py\n"
        "+++ b/sourcing_model.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    incremental_patch = (
        "diff --git a/gateway/module.py b/gateway/module.py\n"
        "--- a/gateway/module.py\n"
        "+++ b/gateway/module.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    cumulative_patch = first_patch + incremental_patch
    source_diff_hash = sha256_json({"unified_diff": cumulative_patch})
    incremental_hash = sha256_json({"unified_diff": incremental_patch})
    artifact_payload = {
        **{
            key: value
            for key, value in base_artifact.items()
            if key != "artifact_hash"
        },
        "candidate_index": 2,
        "source_diff_hash": source_diff_hash,
        "target_files": ["gateway/module.py"],
        "unified_diff": cumulative_patch,
    }
    artifact = {
        **artifact_payload,
        "artifact_hash": sha256_json(artifact_payload),
    }
    composition = {
        "schema_version": "research_lab.git_tree_composition.v1",
        "incremental_source_diff_hash": incremental_hash,
        "cumulative_source_diff_hash": source_diff_hash,
        "cumulative_changed_files": ["gateway/module.py"],
        "child_source_tree_hash": candidate_manifest["model_artifact_hash"],
    }
    build_payload = {
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "parent_manifest_hash": artifact["parent_manifest_hash"],
        "candidate_model_artifact_hash": candidate_manifest["model_artifact_hash"],
        "candidate_model_manifest_hash": candidate_manifest["manifest_hash"],
        "source_diff_hash": source_diff_hash,
        "source_diff_artifact_uri": "s3://fixture/legacy-depth-two/source_diff.json",
        "source_diff_artifact_hash": artifact["artifact_hash"],
        "changed_files": ["gateway/module.py", "sourcing_model.py"],
        "git_tree": {
            "schema_version": "research_lab.git_tree_lineage.v1",
            "depth": 2,
            "root_artifact_hash": artifact["parent_artifact_hash"],
            "incremental_source_diff_hash": incremental_hash,
            "cumulative_source_diff_hash": source_diff_hash,
            "composition": composition,
        },
    }
    build_doc = {
        **build_payload,
        "build_doc_hash": sha256_json(build_payload),
    }
    patch_payload = {
        "candidate_kind": "image_build",
        "patch_type": "IMAGE_BUILD",
        "target_component_id": "private_model_source_tree",
        "parent_artifact_hash": artifact["parent_artifact_hash"],
        "candidate_artifact_hash": candidate_manifest["model_artifact_hash"],
        "candidate_model_manifest_hash": candidate_manifest["manifest_hash"],
        "patch_payload_hash": source_diff_hash,
        "candidate_source_diff_hash": source_diff_hash,
        "candidate_build_doc_hash": build_doc["build_doc_hash"],
        "redacted_summary": "legacy depth-two fixture",
        "validation_result": "passed",
        "patch_doc": {"target_files": ["gateway/module.py"]},
    }
    patch_manifest = {
        **patch_payload,
        "manifest_hash": sha256_json(patch_payload),
    }
    original_run_command = promotion._run_command

    def run_command(cmd, **kwargs):
        if cmd[:3] == ["aws", "s3", "cp"]:
            return json.dumps(artifact, sort_keys=True)
        return original_run_command(cmd, **kwargs)

    monkeypatch.setattr(promotion, "_run_command", run_command)
    result = promotion._push_candidate_source_diff_to_repo(
        repo_url=str(remote),
        branch_name="main",
        active_git_commit_sha=active_sha,
        candidate_id="candidate:legacy-depth-two",
        score_bundle_id="bundle:legacy-depth-two",
        candidate_build_doc=build_doc,
        candidate_patch_manifest=patch_manifest,
        candidate_model_manifest_doc=candidate_manifest,
        expected_candidate_patch_hash=sha256_json(patch_manifest),
        expected_source_diff_hash=source_diff_hash,
        expected_parent_artifact_hash=artifact["parent_artifact_hash"],
        expected_run_id=artifact["run_id"],
    )

    assert result["status"] == "pushed"
    assert result["target_files"] == ["gateway/module.py", "sourcing_model.py"]
    assert _git(
        ["--git-dir", str(remote), "show", "main:sourcing_model.py"]
    ) == "VALUE = 2"
    assert _git(
        ["--git-dir", str(remote), "show", "main:gateway/module.py"]
    ) == "VALUE = 2"


def test_private_source_push_rejects_tampered_s3_body_before_git(
    monkeypatch,
):
    artifact, build_doc, patch_manifest, candidate_manifest = (
        _private_source_push_fixture()
    )
    artifact["unified_diff"] = artifact["unified_diff"].replace(
        "+VALUE = 2", "+VALUE = 'TAMPERED'"
    )
    calls = []

    def run_command(cmd, **_kwargs):
        calls.append(tuple(cmd))
        if cmd[:3] == ["aws", "s3", "cp"]:
            return json.dumps(artifact, sort_keys=True)
        raise AssertionError("Git must not run for an uncommitted artifact")

    monkeypatch.setattr(promotion, "_run_command", run_command)
    with pytest.raises(RuntimeError, match="artifact commitment differs"):
        promotion._push_candidate_source_diff_to_repo(
            repo_url="unused",
            branch_name="main",
            active_git_commit_sha="",
            candidate_id="candidate:fixture",
            score_bundle_id="bundle:fixture",
            candidate_build_doc=build_doc,
            candidate_patch_manifest=patch_manifest,
            candidate_model_manifest_doc=candidate_manifest,
            expected_candidate_patch_hash=sha256_json(patch_manifest),
            expected_source_diff_hash=artifact["source_diff_hash"],
            expected_parent_artifact_hash=artifact["parent_artifact_hash"],
            expected_run_id=artifact["run_id"],
        )
    assert calls == [
        ("aws", "s3", "cp", build_doc["source_diff_artifact_uri"], "-")
    ]


def test_private_source_push_rejects_hash_consistent_mode_change_before_git(
    monkeypatch,
):
    artifact, build_doc, patch_manifest, candidate_manifest = (
        _private_source_push_fixture()
    )
    structural_diff = artifact["unified_diff"].replace(
        "--- a/sourcing_model.py\n",
        "old mode 100644\nnew mode 100755\n--- a/sourcing_model.py\n",
    )
    source_diff_hash = sha256_json({"unified_diff": structural_diff})
    artifact_payload = {
        **{key: value for key, value in artifact.items() if key != "artifact_hash"},
        "unified_diff": structural_diff,
        "source_diff_hash": source_diff_hash,
    }
    artifact = {
        **artifact_payload,
        "artifact_hash": sha256_json(artifact_payload),
    }
    unhashed_annotations = {
        "conditional_validation_policy",
        "loop_dev_score",
        "loop_dev_score_version",
        "loop_direction_plan_hash",
        "loop_node_id",
        "plan_alignment",
        "selected_path_id",
        "stale_parent_rebase",
    }
    build_doc = {
        **build_doc,
        "source_diff_hash": source_diff_hash,
        "source_diff_artifact_hash": artifact["artifact_hash"],
    }
    immutable_build = {
        key: value
        for key, value in build_doc.items()
        if key != "build_doc_hash" and key not in unhashed_annotations
    }
    build_doc["build_doc_hash"] = sha256_json(immutable_build)
    patch_payload = {
        **{
            key: value
            for key, value in patch_manifest.items()
            if key != "manifest_hash"
        },
        "patch_payload_hash": source_diff_hash,
        "candidate_source_diff_hash": source_diff_hash,
        "candidate_build_doc_hash": build_doc["build_doc_hash"],
    }
    patch_manifest = {
        **patch_payload,
        "manifest_hash": sha256_json(patch_payload),
    }
    calls = []

    def run_command(cmd, **_kwargs):
        calls.append(tuple(cmd))
        if cmd[:3] == ["aws", "s3", "cp"]:
            return json.dumps(artifact, sort_keys=True)
        raise AssertionError("Git must not run for a structural patch")

    monkeypatch.setattr(promotion, "_run_command", run_command)
    with pytest.raises(RuntimeError, match="not a content-only Git patch"):
        promotion._push_candidate_source_diff_to_repo(
            repo_url="unused",
            branch_name="main",
            active_git_commit_sha="",
            candidate_id="candidate:structural-patch",
            score_bundle_id="bundle:structural-patch",
            candidate_build_doc=build_doc,
            candidate_patch_manifest=patch_manifest,
            candidate_model_manifest_doc=candidate_manifest,
            expected_candidate_patch_hash=sha256_json(patch_manifest),
            expected_source_diff_hash=source_diff_hash,
            expected_parent_artifact_hash=artifact["parent_artifact_hash"],
            expected_run_id=artifact["run_id"],
        )
    assert calls == [
        ("aws", "s3", "cp", build_doc["source_diff_artifact_uri"], "-")
    ]


@dataclass
class FakeArtifact:
    model_artifact_hash: str = "sha256:" + "a" * 64
    manifest_hash: str = "sha256:" + "b" * 64
    manifest_uri: str = "s3://bucket/manifest.json"
    git_commit_sha: str = "c" * 40
    image_digest: str = "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:" + "a" * 64
    component_registry_version: str = "1.0"
    scoring_adapter_version: str = "1.0"
    compatibility_contract_override: Mapping[str, str] | None = None
    consumer_parity_fixtures_override: Mapping[str, str] | None = None

    @property
    def compatibility_contract(self) -> dict[str, str]:
        if self.compatibility_contract_override is not None:
            return dict(self.compatibility_contract_override)
        contract = _TEST_CONSUMER_SNAPSHOT["contract"]
        return {
            "contract_id": str(contract["contract_id"]),
            "path": str(contract["canonical_path"]),
            "sha256": str(_TEST_CONSUMER_SNAPSHOT["contract_sha256"]),
        }

    @property
    def consumer_parity_fixtures(self) -> dict[str, str]:
        if self.consumer_parity_fixtures_override is not None:
            return dict(self.consumer_parity_fixtures_override)
        contract = _TEST_CONSUMER_SNAPSHOT["contract"]
        return {
            "path": str(contract["parity_fixture_path"]),
            "sha256": str(_TEST_CONSUMER_SNAPSHOT["parity_sha256"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_artifact_hash": self.model_artifact_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_uri": self.manifest_uri,
            "git_commit_sha": self.git_commit_sha,
            "image_digest": self.image_digest,
            "config_hash": "sha256:" + "d" * 64,
            "component_registry_version": self.component_registry_version,
            "scoring_adapter_version": self.scoring_adapter_version,
            "signature_ref": "sig",
            "build_id": "",
            "compatibility_contract": self.compatibility_contract,
            "consumer_parity_fixtures": self.consumer_parity_fixtures,
        }


def _valid_fake_artifact(**overrides: Any) -> FakeArtifact:
    artifact = FakeArtifact(**overrides)
    payload = artifact.to_dict()
    payload.pop("manifest_hash", None)
    artifact.manifest_hash = sha256_json(payload)
    return artifact


def _valid_fake_artifact_for_mode(
    admission_mode: str,
    **overrides: Any,
) -> FakeArtifact:
    if admission_mode == "semantic_v1":
        policy, _policy_hash = semantic_compatibility_policy_identity_v1()
        overrides = {
            "compatibility_contract_override": {
                "contract_id": "leadpoet-sourcing-wrapper-contract-future-test",
                "path": str(policy["canonical_contract_path"]),
                "sha256": "sha256:" + "6" * 64,
            },
            "consumer_parity_fixtures_override": {
                "path": str(policy["canonical_parity_path"]),
                "sha256": "sha256:" + "7" * 64,
            },
            **overrides,
        }
    return _valid_fake_artifact(**overrides)


def _compatibility_receipt(
    artifact: FakeArtifact,
    *,
    admission_mode: str = "legacy_exact",
    binding_suffix: str = "",
) -> dict[str, Any]:
    _policy, policy_hash = semantic_compatibility_policy_identity_v1()
    source_body = {
        "schema_version": SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1,
        "consumer_api_version": SEMANTIC_COMPATIBILITY_CONSUMER_API_V1,
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "admission_mode": admission_mode,
        "policy_hash": policy_hash,
        "source_tree_hash": artifact.model_artifact_hash,
        "manifest_hash": artifact.manifest_hash,
        "image_digest": artifact.image_digest,
        "contract_id": artifact.compatibility_contract["contract_id"],
        "contract_schema_major": 1,
        "contract_hash": artifact.compatibility_contract["sha256"],
        "parity_hash": artifact.consumer_parity_fixtures["sha256"],
        "bindings": {"adapter_version": "test-adapter" + binding_suffix},
    }
    source_receipt = {
        **source_body,
        "receipt_hash": sha256_json(source_body),
    }
    measured_body = {
        "schema_version": (
            model_authority_v2.MEASURED_COMPATIBILITY_ADMISSION_SCHEMA_V1
        ),
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "admission_mode": admission_mode,
        "consumer_api_version": SEMANTIC_COMPATIBILITY_CONSUMER_API_V1,
        "compatibility_policy_hash": policy_hash,
        "compatibility_admission_hash": source_receipt["receipt_hash"],
        "source_tree_hash": artifact.model_artifact_hash,
        "manifest_hash": artifact.manifest_hash,
        "image_digest": artifact.image_digest,
        "module_name": "research_lab_adapter",
        "callable_name": "adapter_metadata",
        "consumer_runtime_probe_hash": sha256_json(
            {"probe": artifact.manifest_hash, "suffix": binding_suffix}
        ),
        "adapter_metadata_hash": sha256_json(
            {"metadata": artifact.manifest_hash, "suffix": binding_suffix}
        ),
        "execution_receipt_hash": sha256_json(
            {"execution": artifact.manifest_hash, "suffix": binding_suffix}
        ),
    }
    measured = {
        **measured_body,
        "receipt_hash": sha256_json(measured_body),
    }
    combined_body = {
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "host_compatibility_receipt_hash": source_receipt["receipt_hash"],
        "measured_runtime_receipt_hash": measured["receipt_hash"],
        "measured_runtime_probe_hash": measured[
            "consumer_runtime_probe_hash"
        ],
    }
    return {
        **source_receipt,
        "measured_runtime_admission": measured,
        "measured_runtime_decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "measured_runtime_probe_hash": measured[
            "consumer_runtime_probe_hash"
        ],
        "combined_receipt_hash": sha256_json(combined_body),
    }


@pytest.mark.parametrize("admission_mode", ["legacy_exact", "semantic_v1"])
async def test_private_artifact_compatibility_always_uses_source_admission(
    monkeypatch: pytest.MonkeyPatch,
    admission_mode: str,
) -> None:
    artifact = _valid_fake_artifact_for_mode(admission_mode)
    calls: list[tuple[FakeArtifact, int]] = []

    async def _admit(
        admitted_artifact: FakeArtifact,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        calls.append((admitted_artifact, timeout_seconds))
        return _compatibility_receipt(
            admitted_artifact,
            admission_mode=admission_mode,
        )

    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _admit,
    )

    receipt = await promotion._preflight_private_artifact_compatibility(artifact)

    assert receipt["admission_mode"] == admission_mode
    assert calls == [
        (
            artifact,
            promotion.PRIVATE_MODEL_COMPATIBILITY_PREFLIGHT_TIMEOUT_SECONDS,
        )
    ]


async def test_forward_rollback_and_old_restoration_all_reread_pointer_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_legacy = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="8" * 40,
    )
    semantic_forward = _valid_fake_artifact_for_mode(
        "semantic_v1",
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        image_digest="private.invalid/model@sha256:" + "9" * 64,
    )
    legacy_rollback = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "a" * 64,
        git_commit_sha="a" * 40,
        image_digest="private.invalid/model@sha256:" + "a" * 64,
    )
    mode_by_artifact_hash = {
        old_legacy.model_artifact_hash: "legacy_exact",
        semantic_forward.model_artifact_hash: "semantic_v1",
        legacy_rollback.model_artifact_hash: "legacy_exact",
    }
    pointer = [old_legacy]
    branch = [old_legacy.git_commit_sha]
    calls: list[str] = []

    async def _admit(
        admitted_artifact: FakeArtifact,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        del timeout_seconds
        calls.append("source")
        return _compatibility_receipt(
            admitted_artifact,
            admission_mode=mode_by_artifact_hash[
                admitted_artifact.model_artifact_hash
            ],
        )

    def _load_pointer(_uri: str) -> FakeArtifact:
        calls.append("pointer")
        return pointer[0]

    def _resolve_branch(**_kwargs: Any) -> str:
        calls.append("branch")
        return branch[0]

    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _admit,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", _load_pointer)
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        _resolve_branch,
    )

    transitions = [semantic_forward, legacy_rollback, old_legacy]
    observed_modes: list[str] = []
    for artifact in transitions:
        pointer[0] = artifact
        branch[0] = artifact.git_commit_sha
        admission = await _REAL_PREFLIGHT_PRIVATE_MODEL_ACTIVATION(
            _controller_config(),
            artifact,
            pointer_uri="s3://private/current.json",
            mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD,
            expected_branch_sha=artifact.git_commit_sha,
        )
        assert admission.artifact is artifact
        observed_modes.append(
            str(admission.compatibility_receipt["admission_mode"])
        )

    assert observed_modes == ["semantic_v1", "legacy_exact", "legacy_exact"]
    assert calls == ["source", "pointer", "branch"] * 3


@pytest.mark.parametrize(
    ("mode", "sha_length"),
    [
        (promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD, 7),
        (promotion.PRIVATE_MODEL_ACTIVATION_MODE_IMMUTABLE_CANDIDATE, 8),
        (promotion.PRIVATE_MODEL_ACTIVATION_MODE_RECONCILE_SUPERSEDED, 39),
    ],
)
async def test_activation_rejects_collision_prefix_commit_before_slow_work(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    sha_length: int,
) -> None:
    artifact = _valid_fake_artifact(git_commit_sha="a" * sha_length)

    async def _unexpected_admission(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("abbreviated commit must fail before source admission")

    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _unexpected_admission,
    )
    monkeypatch.setattr(
        promotion,
        "_load_valid_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("abbreviated commit must fail before pointer reread")
        ),
    )

    with pytest.raises(RuntimeError, match="exact 40-character commit"):
        await _REAL_PREFLIGHT_PRIVATE_MODEL_ACTIVATION(
            _controller_config(),
            artifact,
            pointer_uri=artifact.manifest_uri,
            mode=mode,
        )


@pytest.mark.parametrize(
    ("repo_url", "resolved_branch", "expected_fence"),
    [
        (
            "git@example.invalid/private.git",
            "e" * 40,
            "remote_branch_stable",
        ),
        ("", None, "immutable_manifest_only"),
    ],
)
async def test_immutable_candidate_uses_signed_manifest_and_optional_branch_fence(
    monkeypatch: pytest.MonkeyPatch,
    repo_url: str,
    resolved_branch: str | None,
    expected_fence: str,
) -> None:
    artifact = _valid_fake_artifact(
        git_commit_sha="d" * 40,
        manifest_uri="s3://bucket/immutable-candidate.json",
    )
    calls: list[str] = []

    async def _admit(
        admitted_artifact: FakeArtifact,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        del timeout_seconds
        assert admitted_artifact is artifact
        calls.append("source")
        return _compatibility_receipt(admitted_artifact)

    def _load(uri: str) -> FakeArtifact:
        assert uri == artifact.manifest_uri
        calls.append("manifest")
        return artifact

    def _branch(**_kwargs: Any) -> str:
        if resolved_branch is None:
            raise AssertionError("repo-less immutable candidate must not resolve a branch")
        calls.append("branch")
        return resolved_branch

    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _admit,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", _load)
    monkeypatch.setattr(promotion, "_resolve_private_repo_head_sha", _branch)

    admitted = await _REAL_PREFLIGHT_PRIVATE_MODEL_ACTIVATION(
        _controller_config(private_repo_url=repo_url),
        artifact,
        pointer_uri=artifact.manifest_uri,
        mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_IMMUTABLE_CANDIDATE,
    )

    assert admitted.artifact is artifact
    assert admitted.branch_fence_mode == expected_fence
    assert admitted.branch_sha == (resolved_branch or artifact.git_commit_sha)
    assert calls == (
        ["source", "manifest", "branch"]
        if resolved_branch is not None
        else ["source", "manifest"]
    )


@dataclass
class FakeStore:
    """Table-name-dispatched fakes for the store functions promotion.py uses."""

    select_many_results: dict[str, Any] = field(default_factory=dict)
    select_one_results: dict[str, Any] = field(default_factory=dict)
    select_many_calls: list[tuple[str, tuple]] = field(default_factory=list)
    select_many_shapes: list[tuple[str, tuple, int | None]] = field(
        default_factory=list
    )
    select_all_calls: list[tuple[str, tuple]] = field(default_factory=list)
    select_all_order_by_calls: list[tuple[str, tuple]] = field(default_factory=list)
    version_writes: list[dict[str, Any]] = field(default_factory=list)
    version_event_writes: list[dict[str, Any]] = field(default_factory=list)
    promotion_event_writes: list[dict[str, Any]] = field(default_factory=list)
    candidate_evaluation_event_writes: list[dict[str, Any]] = field(default_factory=list)
    scoring_dispatch_event_writes: list[dict[str, Any]] = field(default_factory=list)
    reward_obligation_writes: list[dict[str, Any]] = field(default_factory=list)
    private_benchmark_writes: list[dict[str, Any]] = field(default_factory=list)
    public_report_writes: list[dict[str, Any]] = field(default_factory=list)
    generic_insert_writes: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    lineage_generation_value: int = 0

    def _lineage_rows(self) -> list[dict[str, Any]]:
        unfiltered_key = (
            "research_lab_private_model_version_current:unfiltered"
        )
        if unfiltered_key not in self.select_many_results:
            self.select_many_results[unfiltered_key] = list(
                self.select_many_results.get(
                    "research_lab_private_model_version_current:active",
                    [],
                )
            )
        rows = self.select_many_results[unfiltered_key]
        if isinstance(rows, Exception) or callable(rows):
            return []
        return rows

    async def select_many(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        filters = tuple(kwargs.get("filters") or ())
        self.select_many_calls.append((table, filters))
        self.select_many_shapes.append((table, filters, kwargs.get("limit")))
        if table == "research_lab_private_model_version_current":
            configured = self.select_many_results.get(
                self._select_many_key(table, kwargs)
            )
            if isinstance(configured, Exception):
                raise configured
            if callable(configured):
                return list(configured(kwargs))
            rows = list(self._lineage_rows())
            filters = tuple(kwargs.get("filters") or ())
            for field, *rest in filters:
                if field in {
                    "current_version_status",
                    "model_artifact_hash",
                    "private_model_version_id",
                } and rest:
                    rows = [
                        row
                        for row in rows
                        if row.get(field) == rest[0]
                    ]
            return rows[: int(kwargs.get("limit") or len(rows) or 1)]
        if table == "research_lab_private_model_version_events":
            rows = list(self.version_event_writes)
            for spec in tuple(kwargs.get("filters") or ()):
                if len(spec) == 2:
                    rows = [
                        row
                        for row in rows
                        if row.get(spec[0]) == spec[1]
                    ]
            return rows[: int(kwargs.get("limit") or len(rows) or 1)]
        result = self.select_many_results.get(self._select_many_key(table, kwargs))
        if result is None:
            result = self.select_many_results.get(table, [])
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return list(result)

    async def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.select_all_calls.append((table, tuple(kwargs.get("filters") or ())))
        self.select_all_order_by_calls.append(
            (table, tuple(kwargs.get("order_by") or ()))
        )
        if table == "research_lab_private_model_version_current":
            return list(self._lineage_rows())
        result = self.select_many_results.get(self._select_many_key(table, kwargs))
        if result is None:
            result = self.select_many_results.get(table, [])
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return list(result)

    def _select_many_key(self, table: str, kwargs: Mapping[str, Any]) -> str:
        filters = tuple(kwargs.get("filters") or ())
        for spec in filters:
            if len(spec) == 2 and spec[0] in ("current_version_status", "event_type"):
                return f"{table}:{spec[1]}"
        if not filters:
            return f"{table}:unfiltered"
        return table

    async def select_one(self, table: str, **kwargs: Any) -> dict[str, Any] | None:
        result = self.select_one_results.get(table)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            result = result(kwargs)
        return result

    async def create_private_model_version(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.version_writes.append(kwargs)
        return (
            {"private_model_version_id": "private_model_version:sha256:" + "e" * 64, **kwargs},
            {"event_type": kwargs.get("version_status")},
        )

    async def create_private_model_version_event(self, **kwargs: Any) -> dict[str, Any]:
        self.version_event_writes.append(kwargs)
        return {"event_id": f"evt-{len(self.version_event_writes)}", **kwargs}

    async def ensure_private_model_version_row_exact(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], bool]:
        artifact = dict(kwargs["artifact_manifest"])
        rows = self._lineage_rows()
        for row in rows:
            if row.get("model_artifact_hash") == artifact["model_artifact_hash"]:
                return dict(row), False
        self.version_writes.append(dict(kwargs))
        row = {
            "private_model_version_id": (
                "private_model_version:" + artifact["model_artifact_hash"]
            ),
            "model_artifact_hash": artifact["model_artifact_hash"],
            "private_model_manifest_hash": artifact["manifest_hash"],
            "private_model_manifest_uri": kwargs["manifest_uri"],
            "git_commit_sha": artifact["git_commit_sha"],
            "source_candidate_id": kwargs.get("source_candidate_id"),
            "source_score_bundle_id": kwargs.get("source_score_bundle_id"),
            "source_benchmark_bundle_id": kwargs.get(
                "source_benchmark_bundle_id"
            ),
            "current_version_status": None,
            "current_status_at": None,
            "current_event_seq": None,
            "current_event_hash": None,
        }
        rows.append(row)
        return dict(row), True

    async def create_private_model_version_event_cas(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        version_id = str(kwargs["private_model_version_id"])
        rows = self._lineage_rows()
        matching = [
            row
            for row in rows
            if str(row.get("private_model_version_id") or "") == version_id
        ]
        if len(matching) != 1:
            raise RuntimeError("fake private model CAS target missing")
        row = matching[0]
        expected_seq = kwargs.get("expected_current_event_seq")
        if expected_seq is None:
            matches = row.get("current_event_seq") is None
            next_seq = 0
        else:
            matches = (
                row.get("current_event_seq") == expected_seq
                and row.get("current_event_hash")
                == kwargs.get("expected_current_event_hash")
                and row.get("current_version_status")
                == kwargs.get("expected_current_version_status")
            )
            next_seq = int(expected_seq) + 1
        if not matches:
            raise RuntimeError("fake private model CAS conflict")
        payload = {
            "private_model_version_id": version_id,
            "seq": next_seq,
            "event_type": kwargs["event_type"],
            "version_status": kwargs["version_status"],
            "reason": kwargs.get("reason"),
            "event_doc": dict(kwargs.get("event_doc") or {}),
        }
        event = {
            "event_id": f"evt-{len(self.version_event_writes) + 1}",
            **payload,
            "anchored_hash": promotion.canonical_hash(payload),
        }
        self.version_event_writes.append(event)
        self.lineage_generation_value += 1
        row.update(
            {
                "current_event_seq": next_seq,
                "current_event_hash": event["anchored_hash"],
                "current_version_status": kwargs["version_status"],
                "current_status_at": (
                    f"2026-08-17T00:00:{self.lineage_generation_value:02d}+00:00"
                ),
            }
        )
        return dict(event)

    async def private_model_lineage_generation(self) -> int:
        return self.lineage_generation_value

    async def create_candidate_promotion_event(self, **kwargs: Any) -> dict[str, Any]:
        self.promotion_event_writes.append(kwargs)
        return {"promotion_event_id": f"pe-{len(self.promotion_event_writes)}", **kwargs}

    async def create_candidate_evaluation_event(self, **kwargs: Any) -> dict[str, Any]:
        self.candidate_evaluation_event_writes.append(kwargs)
        return {"event_id": f"ce-{len(self.candidate_evaluation_event_writes)}", **kwargs}

    async def create_scoring_dispatch_event(self, **kwargs: Any) -> dict[str, Any]:
        self.scoring_dispatch_event_writes.append(kwargs)
        return {"dispatch_event_id": f"sd-{len(self.scoring_dispatch_event_writes)}", **kwargs}

    async def create_champion_reward_obligation(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reward_obligation_writes.append(kwargs)
        return {"champion_reward_id": "cr-1"}, {"event_type": "active"}

    async def create_private_model_benchmark_bundle(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.private_benchmark_writes.append(kwargs)
        return (
            {
                "benchmark_bundle_id": "private_benchmark:" + "8" * 64,
                "current_benchmark_status": "completed",
                **kwargs,
            },
            {"event_type": "completed"},
        )

    async def create_public_benchmark_report(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.public_report_writes.append(kwargs)
        return (
            {
                "report_id": "public_benchmark:sha256:" + "9" * 64,
                "current_report_status": "published",
                **kwargs,
            },
            {"event_type": "published"},
        )

    async def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        self.generic_insert_writes.append((table, row))
        return dict(row)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(promotion, "select_many", fake.select_many)
    monkeypatch.setattr(promotion, "select_all", fake.select_all)
    monkeypatch.setattr(promotion, "select_one", fake.select_one)
    monkeypatch.setattr(
        promotion,
        "ensure_private_model_version_row_exact",
        fake.ensure_private_model_version_row_exact,
    )
    monkeypatch.setattr(
        promotion,
        "create_private_model_version_event_cas",
        fake.create_private_model_version_event_cas,
    )
    monkeypatch.setattr(
        promotion,
        "private_model_lineage_generation",
        fake.private_model_lineage_generation,
    )
    monkeypatch.setattr(promotion, "create_candidate_promotion_event", fake.create_candidate_promotion_event)
    monkeypatch.setattr(promotion, "create_candidate_evaluation_event", fake.create_candidate_evaluation_event)
    monkeypatch.setattr(promotion, "create_scoring_dispatch_event", fake.create_scoring_dispatch_event)
    monkeypatch.setattr(promotion, "create_champion_reward_obligation", fake.create_champion_reward_obligation)
    monkeypatch.setattr(promotion, "create_private_model_benchmark_bundle", fake.create_private_model_benchmark_bundle)
    monkeypatch.setattr(promotion, "create_public_benchmark_report", fake.create_public_benchmark_report)
    monkeypatch.setattr(store_module, "insert_row", fake.insert_row)
    monkeypatch.setattr(promotion, "insert_row", fake.insert_row)
    monkeypatch.setattr(promotion, "sign_digest_with_kms", lambda **kwargs: "kms-signature:test")

    async def _admit_artifact(
        artifact: FakeArtifact,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        assert timeout_seconds == promotion.PRIVATE_MODEL_COMPATIBILITY_PREFLIGHT_TIMEOUT_SECONDS
        return _compatibility_receipt(artifact)

    async def _admit_activation(
        config: Any,
        artifact: FakeArtifact,
        *,
        pointer_uri: str,
        mode: str,
        expected_branch_sha: str = "",
    ) -> Any:
        receipt = _compatibility_receipt(artifact)
        branch_sha = expected_branch_sha or artifact.git_commit_sha
        branch_fence_mode = (
            "immutable_manifest_only"
            if mode
            == promotion.PRIVATE_MODEL_ACTIVATION_MODE_IMMUTABLE_CANDIDATE
            and not str(getattr(config, "private_repo_url", "") or "")
            else "remote_branch_stable"
        )
        identity = promotion._private_artifact_compatibility_identity(artifact)
        receipt_hash = sha256_json(receipt)
        generation_hash = sha256_json(
            {
                "schema_version": (
                    "leadpoet.private-model-activation-generation.v1"
                ),
                "activation_mode": mode,
                "artifact_identity": list(identity),
                "compatibility_receipt_hash": receipt_hash,
                "pointer_ref_hash": sha256_json(
                    {"private_model_pointer_uri": pointer_uri}
                ),
                "branch_sha": branch_sha,
                "branch_fence_mode": branch_fence_mode,
            }
        )
        return promotion._PrivateModelActivationAdmission(
            mode=mode,
            artifact=artifact,
            artifact_identity=identity,
            compatibility_receipt=receipt,
            compatibility_receipt_hash=receipt_hash,
            pointer_uri=pointer_uri,
            branch_sha=branch_sha,
            branch_fence_mode=branch_fence_mode,
            generation_hash=generation_hash,
        )

    async def _fence_activation_references(
        config: Any,
        admitted: Any,
    ) -> None:
        del config, admitted

    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _admit_artifact,
    )
    monkeypatch.setattr(
        promotion,
        "_preflight_private_model_activation",
        _admit_activation,
    )
    monkeypatch.setattr(
        promotion,
        "_revalidate_private_model_activation_references",
        _fence_activation_references,
    )
    monkeypatch.delenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, raising=False)
    monkeypatch.delenv(promotion.AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV, raising=False)
    return fake


@pytest.fixture
def bootstrap_artifact(monkeypatch: pytest.MonkeyPatch) -> FakeArtifact:
    artifact = FakeArtifact()
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: artifact)
    return artifact


def _config() -> Any:
    return SimpleNamespace(
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        private_repo_url="git@example.invalid/private.git",
        private_repo_branch=DEFAULT_PRIVATE_REPO_BRANCH,
    )


def _active_row(artifact: FakeArtifact) -> dict[str, Any]:
    return {
        "private_model_version_id": "private_model_version:sha256:" + "f" * 64,
        "private_model_manifest_uri": artifact.manifest_uri,
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "git_commit_sha": artifact.git_commit_sha,
        "current_version_status": "active",
        "current_status_at": "2026-07-01T00:00:00+00:00",
        "current_event_seq": 0,
        "current_event_hash": "sha256:" + "1" * 64,
    }


def _assert_db_doc_safe(doc: Mapping[str, Any]) -> None:
    encoded = json.dumps(doc, sort_keys=True, default=str)
    assert not promotion._DB_DOC_FORBIDDEN_RE.search(encoded)


# ---------------------------------------------------------------------------
# Bug #2 — lineage fail-closed
# ---------------------------------------------------------------------------


async def test_lineage_read_error_raises_retryable_and_never_bootstraps(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = RuntimeError("supabase blip")
    with pytest.raises(PrivateModelLineageUnavailableError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_lineage_empty_without_flag_returns_unregistered_bootstrap(store, bootstrap_artifact):
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.artifact is bootstrap_artifact
    assert result.version_row is None
    assert store.version_writes == []


async def test_lineage_empty_with_flag_registers_bootstrap(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.version_row is not None
    assert len(store.version_writes) == 1
    assert store.version_writes[0]["manifest_uri"] == bootstrap_artifact.manifest_uri
    assert store.version_writes[0]["manifest_uri"] != _config().private_model_manifest_uri
    assert len(store.version_event_writes) == 1
    assert store.version_event_writes[0]["version_status"] == "active"
    assert (
        store.version_event_writes[0]["reason"]
        == "bootstrap_private_model_manifest_uri"
    )
    assert (
        store.version_event_writes[0]["event_doc"][
            "activation_protocol_version"
        ]
        == promotion.PRIVATE_MODEL_ACTIVATION_PROTOCOL_V1
    )


async def test_bootstrap_retry_after_row_insert_crash_activates_same_row(
    store: FakeStore,
    bootstrap_artifact: FakeArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    original_cas = store.create_private_model_version_event_cas
    crash_once = True

    async def _crash_after_row_insert(**kwargs: Any) -> dict[str, Any]:
        nonlocal crash_once
        if crash_once:
            crash_once = False
            raise RuntimeError("injected crash after immutable version row insert")
        return await original_cas(**kwargs)

    monkeypatch.setattr(
        promotion,
        "create_private_model_version_event_cas",
        _crash_after_row_insert,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        await load_active_private_model(_config(), register_bootstrap=True)

    assert len(store.version_writes) == 1
    assert store.version_event_writes == []
    orphan_id = store._lineage_rows()[0]["private_model_version_id"]

    recovered = await load_active_private_model(
        _config(), register_bootstrap=True
    )

    assert recovered.artifact is bootstrap_artifact
    assert recovered.version_row["private_model_version_id"] == orphan_id
    assert len(store.version_writes) == 1
    assert len(store.version_event_writes) == 1
    assert store.version_event_writes[0]["seq"] == 0
    assert store.version_event_writes[0]["version_status"] == "active"
    event_doc = store.version_event_writes[0]["event_doc"]
    assert event_doc["source"] == "bootstrap_private_model_manifest_uri"
    assert event_doc["activation_protocol_version"] == (
        promotion.PRIVATE_MODEL_ACTIVATION_PROTOCOL_V1
    )
    assert event_doc["expected_global_lineage_generation"] == 0


async def test_bootstrap_orphan_with_nonzero_generation_never_writes(
    store: FakeStore,
    bootstrap_artifact: FakeArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [
        {
            "private_model_version_id": "private_model_version:orphan",
            "private_model_manifest_uri": bootstrap_artifact.manifest_uri,
            "model_artifact_hash": bootstrap_artifact.model_artifact_hash,
            "private_model_manifest_hash": bootstrap_artifact.manifest_hash,
            "git_commit_sha": bootstrap_artifact.git_commit_sha,
            "current_version_status": None,
            "current_status_at": None,
            "current_event_seq": None,
            "current_event_hash": None,
        }
    ]
    store.lineage_generation_value = 1

    with pytest.raises(NoActivePrivateModelVersionError):
        await load_active_private_model(_config(), register_bootstrap=True)

    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_lineage_empty_register_bootstrap_false_never_writes(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await load_active_private_model(_config(), register_bootstrap=False)
    assert result.version_row is None
    assert store.version_writes == []


async def test_zero_active_but_nonempty_lineage_raises_instead_of_bootstrap(store, bootstrap_artifact, monkeypatch):
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = [
        {"private_model_version_id": "v1", "current_version_status": "superseded"}
    ]
    with pytest.raises(NoActivePrivateModelVersionError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []


async def test_manifest_hash_mismatch_raises_explicit_operator_error(store, monkeypatch):
    row_artifact = FakeArtifact()
    row = _active_row(row_artifact)
    # The manifest URI now yields different hashes than the lineage row recorded.
    loaded = FakeArtifact(
        model_artifact_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
        manifest_uri=row_artifact.manifest_uri,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: loaded)
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    with pytest.raises(ActiveManifestHashMismatchError) as excinfo:
        await load_active_private_model(_config(), register_bootstrap=True)
    assert "reregister-active-manifest" in str(excinfo.value)
    assert excinfo.value.detail["row_model_artifact_hash"] == row_artifact.model_artifact_hash
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_active_manifest_load_failure_raises_retryable(store, monkeypatch):
    row = _active_row(FakeArtifact())

    def _boom(uri: str) -> FakeArtifact:
        raise RuntimeError("s3 timeout")

    monkeypatch.setattr(promotion, "_load_valid_artifact", _boom)
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    with pytest.raises(PrivateModelLineageUnavailableError):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []


async def test_matching_active_row_returned(store, monkeypatch):
    artifact = FakeArtifact()
    row = _active_row(artifact)
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda uri: artifact)
    store.select_many_results["research_lab_private_model_version_current:active"] = [row]
    result = await load_active_private_model(_config(), register_bootstrap=True)
    assert result.version_row == row
    assert result.artifact is artifact
    assert store.version_writes == []


# ---------------------------------------------------------------------------
# Bug #3 — reconcile re-activates the newest superseded version
# ---------------------------------------------------------------------------


async def test_reconcile_noop_when_active_present(store):
    store.select_many_results["research_lab_private_model_version_current:active"] = [
        {"private_model_version_id": "v-active", "current_version_status": "active"}
    ]
    result = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=False)
    assert result["status"] == "active_version_present"
    assert store.version_event_writes == []


async def test_reconcile_picks_newest_superseded(store, monkeypatch):
    artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        manifest_uri="s3://bucket/superseded-v-new.json",
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda _uri: artifact)
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = [
        # Ordered newest-first by current_status_at, as the query requests.
        {"private_model_version_id": "v-tomb", "current_version_status": "tombstoned", "current_status_at": "2026-07-03"},
        {
            "private_model_version_id": "v-new",
            "current_version_status": "superseded",
            "current_status_at": "2026-07-02",
            "current_event_seq": 1,
            "current_event_hash": "sha256:" + "2" * 64,
            "model_artifact_hash": artifact.model_artifact_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "private_model_manifest_uri": artifact.manifest_uri,
            "git_commit_sha": artifact.git_commit_sha,
        },
        {"private_model_version_id": "v-old", "current_version_status": "superseded", "current_status_at": "2026-07-01"},
    ]
    dry = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=True)
    assert dry["status"] == "would_reactivate_newest_superseded"
    assert dry["planned"]["private_model_version_id"] == "v-new"
    assert store.version_event_writes == []

    applied = await reconcile_active_private_model_lineage(
        _config(), actor_ref="test", dry_run=False
    )
    assert applied["status"] == "reactivated_newest_superseded"
    assert len(store.version_event_writes) == 1
    write = store.version_event_writes[0]
    assert write["private_model_version_id"] == "v-new"
    assert write["event_type"] == "active"
    assert write["version_status"] == "active"


async def test_reconcile_lineage_empty(store):
    store.select_many_results["research_lab_private_model_version_current:active"] = []
    store.select_many_results["research_lab_private_model_version_current:unfiltered"] = []
    result = await reconcile_active_private_model_lineage(actor_ref="test", dry_run=False)
    assert result["status"] == "lineage_empty"
    assert store.version_event_writes == []


async def test_bootstrap_final_reference_drift_has_zero_lineage_writes(
    store: FakeStore,
    bootstrap_artifact: FakeArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del bootstrap_artifact
    monkeypatch.setenv(promotion.ALLOW_BOOTSTRAP_REGISTER_ENV, "true")
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = []
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = []

    async def _drift(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("branch changed during bootstrap admission")

    monkeypatch.setattr(
        promotion,
        "_revalidate_private_model_activation",
        _drift,
    )
    with pytest.raises(RuntimeError, match="branch changed"):
        await load_active_private_model(_config(), register_bootstrap=True)
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_reconcile_final_reference_drift_has_zero_lineage_writes(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        manifest_uri="s3://bucket/reconcile-target.json",
    )
    target = {
        **_active_row(artifact),
        "private_model_version_id": "version:reconcile",
        "current_version_status": "superseded",
    }
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda _uri: artifact)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = []
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [target]

    async def _drift(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("pointer changed during reconcile admission")

    monkeypatch.setattr(
        promotion,
        "_revalidate_private_model_activation",
        _drift,
    )
    with pytest.raises(RuntimeError, match="pointer changed"):
        await reconcile_active_private_model_lineage(
            _config(), actor_ref="test", dry_run=False
        )
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_reregister_final_reference_drift_has_zero_lineage_writes(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_artifact = FakeArtifact()
    new_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        manifest_uri=old_artifact.manifest_uri,
    )
    old_row = _active_row(old_artifact)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [old_row]
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [old_row]
    monkeypatch.setattr(
        promotion,
        "_load_valid_artifact",
        lambda _uri: new_artifact,
    )

    async def _drift(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("branch changed during reregister admission")

    monkeypatch.setattr(
        promotion,
        "_revalidate_private_model_activation",
        _drift,
    )
    with pytest.raises(RuntimeError, match="branch changed"):
        await promotion.reregister_active_manifest(
            _config(), actor_ref="test", dry_run=False
        )
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_repo_sync_final_reference_drift_has_zero_lineage_writes(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_artifact = FakeArtifact()
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
    )
    old_row = _active_row(old_artifact)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [old_row]
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [old_row]
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: current_artifact.git_commit_sha,
    )

    async def _manifest(*_args: Any, **_kwargs: Any) -> tuple[FakeArtifact, dict[str, Any]]:
        return current_artifact, {"status": "manifest_ready"}

    async def _drift(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("pointer changed during repo sync admission")

    monkeypatch.setattr(
        promotion,
        "_load_repo_head_current_manifest",
        _manifest,
    )
    monkeypatch.setattr(
        promotion,
        "_revalidate_private_model_activation",
        _drift,
    )
    with pytest.raises(RuntimeError, match="pointer changed"):
        await sync_active_model_to_repo_head(
            _controller_config(), actor_ref="test", dry_run=False
        )
    assert store.version_writes == []
    assert store.version_event_writes == []


async def test_activation_uses_bounded_queries_and_resumes_after_later_event(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = FakeArtifact()
    target = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        manifest_uri="s3://bucket/activation-target.json",
    )
    parent_row = _active_row(parent)
    history = [parent_row]
    history.extend(
        {
            "private_model_version_id": f"private_model_version:history:{index}",
            "model_artifact_hash": "sha256:" + f"{index:064x}"[-64:],
            "current_version_status": "superseded",
            "current_event_seq": 1,
            "current_event_hash": "sha256:" + "a" * 64,
        }
        for index in range(10_000)
    )
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = history
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [parent_row]
    admitted = await promotion._preflight_private_model_activation(
        _config(),
        target,
        pointer_uri=target.manifest_uri,
        mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD,
        expected_branch_sha=target.git_commit_sha,
    )
    original_cas = store.create_private_model_version_event_cas
    injected = False

    async def _cas_with_later_unrelated_event(**kwargs: Any) -> dict[str, Any]:
        nonlocal injected
        event = await original_cas(**kwargs)
        if kwargs["version_status"] == "active" and not injected:
            injected = True
            # Represents an unrelated committed lineage append immediately
            # after activation; the active target itself remains unchanged.
            store.lineage_generation_value += 1
        return event

    monkeypatch.setattr(
        promotion,
        "create_private_model_version_event_cas",
        _cas_with_later_unrelated_event,
    )
    event_doc = {
        "source_candidate_id": "candidate:exact",
        "source_score_bundle_id": "score:exact",
        "source_benchmark_bundle_id": "benchmark:exact",
    }

    first = await _REAL_ACTIVATE_PRIVATE_MODEL_GENERATION(
        _config(),
        admitted,
        expected_active_row=parent_row,
        source_candidate_id="candidate:exact",
        source_score_bundle_id="score:exact",
        source_benchmark_bundle_id="benchmark:exact",
        activation_reason=(
            "research_lab_image_build_candidate_repo_head_manifest_promoted"
        ),
        activation_event_doc=event_doc,
    )
    current_target = next(
        row
        for row in store._lineage_rows()
        if row.get("model_artifact_hash") == target.model_artifact_hash
    )
    writes_after_first = len(store.version_event_writes)
    resumed = await _REAL_ACTIVATE_PRIVATE_MODEL_GENERATION(
        _config(),
        admitted,
        expected_active_row=dict(current_target),
        source_candidate_id="candidate:exact",
        source_score_bundle_id="score:exact",
        source_benchmark_bundle_id="benchmark:exact",
        activation_reason=(
            "research_lab_image_build_candidate_repo_head_manifest_promoted"
        ),
        activation_event_doc=event_doc,
    )

    assert first.lineage_generation_after == 3
    assert first.superseded_event["event_doc"][
        "activation_protocol_version"
    ] == promotion.PRIVATE_MODEL_ACTIVATION_PROTOCOL_V1
    assert first.superseded_event["event_doc"][
        "expected_global_lineage_generation"
    ] == 0
    assert resumed.activation_event == first.activation_event
    assert len(store.version_event_writes) == writes_after_first == 2
    lineage_shapes = [
        shape
        for shape in store.select_many_shapes
        if shape[0] == "research_lab_private_model_version_current"
    ]
    assert lineage_shapes
    assert all(limit == 2 for _table, _filters, limit in lineage_shapes)
    assert store.select_all_calls == []


async def test_promotion_rejects_existing_target_with_other_provenance(
    store: FakeStore,
) -> None:
    parent = FakeArtifact()
    target = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="9" * 40,
        manifest_uri="s3://bucket/existing-target.json",
    )
    parent_row = _active_row(parent)
    target_row = {
        **_active_row(target),
        "current_version_status": "superseded",
        "source_candidate_id": "candidate:other",
        "source_score_bundle_id": "score:other",
        "source_benchmark_bundle_id": "benchmark:other",
    }
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [parent_row, target_row]
    admitted = await promotion._preflight_private_model_activation(
        _config(),
        target,
        pointer_uri=target.manifest_uri,
        mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD,
        expected_branch_sha=target.git_commit_sha,
    )

    with pytest.raises(RuntimeError, match="different promotion provenance"):
        await _REAL_ACTIVATE_PRIVATE_MODEL_GENERATION(
            _config(),
            admitted,
            expected_active_row=parent_row,
            source_candidate_id="candidate:requested",
            source_score_bundle_id="score:requested",
            source_benchmark_bundle_id="benchmark:requested",
            activation_reason="promotion",
        )

    assert store.version_writes == []
    assert store.version_event_writes == []


# ---------------------------------------------------------------------------
# N3 — unavailable basis is an explicit rejection
# ---------------------------------------------------------------------------


def _approved_gate(**overrides: Any) -> dict[str, Any]:
    gate = {
        "decision": "private_holdout_approved",
        "private_holdout_evaluated": True,
        "baseline_aggregate_score": 10.0,
        "candidate_total_score": 12.5,
        "candidate_delta_vs_daily_baseline": 2.5,
        "baseline_benchmark_bundle_id": "bb-1",
    }
    gate.update(overrides)
    return gate


def test_metric_approved_gate_has_no_rejection() -> None:
    metric = promotion_improvement_metric({"private_holdout_gate": _approved_gate(), "aggregates": {}})
    assert metric.rejection_status is None
    assert metric.daily_baseline_available is True
    assert metric.improvement_points == pytest.approx(2.5)


def _paired_improvement_gate(**overrides: Any) -> dict[str, Any]:
    gate = {
        "decision": "eligible_for_probation",
        "eligible_for_probation": True,
        "blockers": [],
        "reference_evaluation_mode": "stored_daily_baseline",
        "advisory_basis": (
            "recomputed_candidate_vs_stored_daily_baseline_per_icp"
        ),
        "mean_delta": 2.5,
        "se_delta": 0.5,
        "delta_lcb": 1.5,
        "compared_icp_count": 20,
    }
    gate.update(overrides)
    return gate


def test_new_metric_uses_paired_lcb_instead_of_aggregate_mean() -> None:
    holdout_gate = _approved_gate(
        promotion_metric_version=PAIRED_LCB_PROMOTION_METRIC_VERSION,
    )
    bundle = {
        "private_holdout_gate": holdout_gate,
        "improvement_gate": _paired_improvement_gate(
            mean_delta=2.5,
            se_delta=0.9,
            delta_lcb=0.7,
        ),
        "aggregates": {},
    }

    metric = promotion_improvement_metric(bundle)
    decision = promotion.promotion_gate_decision(
        bundle,
        candidate_kind="image_build",
        candidate_parent="sha256:parent",
        active_parent="sha256:parent",
        threshold_points=1.0,
        auto_promotion_enabled=True,
    )

    assert metric.improvement_points == pytest.approx(0.7)
    assert metric.paired_mean_delta == pytest.approx(2.5)
    assert metric.paired_delta_lcb == pytest.approx(0.7)
    assert metric.basis == "stored_daily_baseline_paired_delta_lcb"
    assert decision.status == "rejected_below_threshold"


def test_new_metric_rejects_when_paired_confidence_is_missing() -> None:
    bundle = {
        "private_holdout_gate": _approved_gate(
            promotion_metric_version=PAIRED_LCB_PROMOTION_METRIC_VERSION,
        ),
        "improvement_gate": {
            "decision": "not_applicable",
            "eligible_for_probation": False,
            "blockers": ["superseded_metric_not_applicable"],
            "reference_evaluation_mode": "stored_daily_baseline",
            "advisory_basis": "superseded_metric_not_applicable",
        },
        "aggregates": {},
    }

    metric = promotion_improvement_metric(bundle)

    assert metric.improvement_points == 0.0
    assert metric.rejection_status == "rejected_paired_lcb_unavailable"
    assert metric.daily_baseline_available is False


def test_new_metric_passes_when_paired_lcb_clears_threshold() -> None:
    bundle = {
        "private_holdout_gate": _approved_gate(
            promotion_metric_version=PAIRED_LCB_PROMOTION_METRIC_VERSION,
        ),
        "improvement_gate": _paired_improvement_gate(delta_lcb=1.25),
        "aggregates": {},
    }

    decision = promotion.promotion_gate_decision(
        bundle,
        candidate_kind="image_build",
        candidate_parent="sha256:parent",
        active_parent="sha256:parent",
        threshold_points=1.0,
        auto_promotion_enabled=True,
    )

    assert decision.status == "promotion_passed"
    assert decision.improvement_points == pytest.approx(1.25)


def test_metric_missing_basis_is_explicit_rejection_not_zero_pass() -> None:
    gate = _approved_gate(
        baseline_aggregate_score=None,
        candidate_total_score=None,
        candidate_delta_vs_daily_baseline=None,
    )
    metric = promotion_improvement_metric({"private_holdout_gate": gate, "aggregates": {}})
    assert metric.rejection_status == "rejected_basis_unavailable"
    assert metric.daily_baseline_available is False
    assert metric.improvement_points == 0.0
    # A future improvement_threshold_points=0 must never promote this bundle:
    # the rejection is carried explicitly, not implied by 0.0 < threshold.
    assert metric.event_doc()["rejection_status"] == "rejected_basis_unavailable"


def test_metric_unapproved_holdout_is_explicit_rejection() -> None:
    gate = _approved_gate(decision="rejected_before_private_holdout")
    metric = promotion_improvement_metric({"private_holdout_gate": gate, "aggregates": {}})
    assert metric.rejection_status == "rejected_basis_unavailable"
    assert "rejected_before_private_holdout" in metric.basis


def test_metric_legacy_bundle_keeps_paired_mean_delta_path() -> None:
    metric = promotion_improvement_metric({"aggregates": {"mean_delta": 1.75}})
    assert metric.rejection_status is None
    assert metric.basis == "legacy_paired_mean_delta_no_holdout_gate"
    assert metric.improvement_points == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Score-only basis — provider exclusions are audit metadata, never arithmetic
# ---------------------------------------------------------------------------


def _baseline_doc(scores: Mapping[str, float], **extra: Any) -> dict[str, Any]:
    return {
        "per_icp_summaries": [
            {"icp_ref": ref, "score": value} for ref, value in scores.items()
        ],
        "aggregate_score": sum(scores.values()) / len(scores),
        **extra,
    }


def test_metric_exclusions_never_adjust_the_basis() -> None:
    """Score-only: provider-excluded ICPs are carried through as audit fields
    but the delta stays exactly the gate's stored candidate-vs-baseline delta —
    no per-ICP baseline re-aggregation."""
    doc = _baseline_doc({"icp:a": 10.0, "icp:b": 0.0, "icp:c": 20.0})
    gate = _approved_gate(
        baseline_aggregate_score=10.0,
        candidate_total_score=12.0,
        candidate_delta_vs_daily_baseline=2.0,
    )
    bundle = {
        "private_holdout_gate": gate,
        "aggregates": {"provider_excluded_icp_ids": ["icp:b"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.baseline_basis_adjusted is False
    assert metric.baseline_aggregate_score == pytest.approx(10.0)
    assert metric.unadjusted_baseline_aggregate_score is None
    assert metric.improvement_points == pytest.approx(2.0)
    assert metric.provider_excluded_icp_ids == ("icp:b",)
    assert metric.basis == "stored_daily_baseline_total_delta"


def test_metric_exclusions_without_baseline_doc_still_compute_stored_delta() -> None:
    """The stored gate delta is self-sufficient: no baseline per-ICP doc is
    needed (or consulted), so its absence cannot reject the candidate."""
    bundle = {
        "private_holdout_gate": _approved_gate(),
        "aggregates": {"provider_excluded_icp_ids": ["icp:b"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=None)
    assert metric.rejection_status is None
    assert metric.improvement_points == pytest.approx(2.5)
    assert metric.basis == "stored_daily_baseline_total_delta"


def test_metric_unknown_exclusion_ids_do_not_reject() -> None:
    doc = _baseline_doc({"icp:a": 10.0, "icp:c": 20.0})
    bundle = {
        "private_holdout_gate": _approved_gate(),
        "aggregates": {"provider_excluded_icp_ids": ["icp:zzz"]},
    }
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.provider_excluded_icp_ids == ("icp:zzz",)


def test_metric_tolerates_absent_exclusion_list() -> None:
    doc = _baseline_doc({"icp:a": 10.0, "icp:c": 20.0})
    bundle = {"private_holdout_gate": _approved_gate(), "aggregates": {}}
    metric = promotion_improvement_metric(bundle, baseline_score_summary_doc=doc)
    assert metric.rejection_status is None
    assert metric.baseline_basis_adjusted is False
    assert metric.improvement_points == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Score-only merge path — health/quarantine state cannot hold or block (N3
# basis rejection is the only non-score gate left besides the threshold)
# ---------------------------------------------------------------------------


def _controller_config(**overrides: Any) -> Any:
    values = {
        "auto_promotion_enabled": True,
        "auto_commit_enabled": False,
        "improvement_threshold_points": 1.0,
        "private_model_manifest_uri": "s3://bucket/bootstrap-manifest.json",
        "private_repo_url": "git@example.invalid/private.git",
        "private_repo_branch": DEFAULT_PRIVATE_REPO_BRANCH,
        "score_bundle_kms_key_id": "arn:aws:kms:us-east-1:123456789012:alias/test",
        "score_bundle_signature_uri_prefix": "s3://bucket/signatures",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _candidate(artifact: FakeArtifact) -> dict[str, Any]:
    return {
        "candidate_id": "cand-1",
        "parent_artifact_hash": artifact.model_artifact_hash,
        "candidate_kind": "image_build",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }


def _score_bundle(gate: Mapping[str, Any], aggregates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "private_holdout_gate": dict(gate),
        "aggregates": dict(aggregates or {}),
        "icp_set_hash": "sha256:" + "3" * 64,
    }


async def test_repo_head_sync_registers_current_json_with_db_safe_doc(store, monkeypatch):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "9" * 64
        ),
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    previous_row = {
        **_active_row(previous_artifact),
        "git_commit_sha": previous_artifact.git_commit_sha,
    }
    store.select_many_results["research_lab_private_model_version_current:active"] = [previous_row]

    resolved_branch: dict[str, str] = {}

    def _fake_resolve_head(*, repo_url: str, branch_name: str) -> str:
        resolved_branch["branch_name"] = branch_name
        return current_artifact.git_commit_sha

    monkeypatch.setattr(promotion, "_resolve_private_repo_head_sha", _fake_resolve_head)

    async def _fake_current_manifest(config: Any, **kwargs: Any) -> tuple[FakeArtifact, dict[str, Any]]:
        return current_artifact, {
            "status": "manifest_ready",
            "current_json_git_sha": current_artifact.git_commit_sha,
            "current_json_manifest_hash": current_artifact.manifest_hash,
            "current_json_model_artifact_hash": current_artifact.model_artifact_hash,
            "current_json_image_digest": current_artifact.image_digest,
        }

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _fake_current_manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@github.com:tasnimuldatascience/Sourcing_model.git",
            private_repo_branch="",
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert result["status"] == "synced_active_model_to_repo_head"
    assert resolved_branch["branch_name"] == DEFAULT_PRIVATE_REPO_BRANCH
    assert len(store.version_writes) == 1
    version_doc = store.version_writes[0]["redacted_version_doc"]
    _assert_db_doc_safe(version_doc)
    assert version_doc["repo_branch"] == DEFAULT_PRIVATE_REPO_BRANCH
    assert version_doc["current_json_manifest_uri"] == current_artifact.manifest_uri
    assert version_doc["image_ref_hash"].startswith("sha256:")
    assert "image_digest" not in json.dumps(version_doc, sort_keys=True, default=str)


@pytest.mark.parametrize(
    ("commit_status", "has_commit_sha", "event_doc"),
    [
        ("pushed", True, {}),
        ("started", False, {"source_push_attempt": 1}),
    ],
)
async def test_repo_head_sync_defers_candidate_owned_publication_until_activation_completes(
    store, monkeypatch, commit_status, has_commit_sha, event_doc
):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [
        {
            **_active_row(previous_artifact),
            "git_commit_sha": previous_artifact.git_commit_sha,
        }
    ]
    store.select_many_results["research_lab_private_repo_commit_events"] = [
        {
            "commit_event_id": "commit-event-pending",
            "commit_status": commit_status,
            "git_commit_sha": (
                current_artifact.git_commit_sha if has_commit_sha else None
            ),
            "candidate_id": "candidate:pending",
            "score_bundle_id": "score-bundle:pending",
            "event_doc": event_doc,
            "created_at": "2026-08-05T00:00:00+00:00",
        }
    ]
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: current_artifact.git_commit_sha,
    )

    async def _manifest(*_args: Any, **_kwargs: Any):
        return current_artifact, {"status": "manifest_ready"}

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@example.invalid/private.git",
            private_repo_branch="",
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert result["ok"] is False
    assert result["status"] == "candidate_source_publication_pending"
    assert result["candidate_id"] == "candidate:pending"
    assert result["score_bundle_id"] == "score-bundle:pending"
    assert result["repo_main_sha"] == current_artifact.git_commit_sha
    assert store.version_event_writes == []
    assert store.version_writes == []


async def test_repo_head_sync_rereads_candidate_ownership_immediately_before_activation(
    store, monkeypatch
):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [
        {
            **_active_row(previous_artifact),
            "git_commit_sha": previous_artifact.git_commit_sha,
        }
    ]
    ownership_reads = 0

    def _publication_interleaving(_kwargs):
        nonlocal ownership_reads
        ownership_reads += 1
        if ownership_reads == 1:
            return []
        return [
            {
                "commit_event_id": "commit-event-started-during-preflight",
                "commit_status": "started",
                "git_commit_sha": None,
                "branch_name": DEFAULT_PRIVATE_REPO_BRANCH,
                "candidate_id": "candidate:interleaved",
                "score_bundle_id": "score-bundle:interleaved",
                "event_doc": {"source_push_attempt": 1},
                "created_at": "2026-08-17T00:00:00+00:00",
            }
        ]

    store.select_many_results[
        "research_lab_private_repo_commit_events"
    ] = _publication_interleaving
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: current_artifact.git_commit_sha,
    )

    async def _manifest(*_args: Any, **_kwargs: Any):
        return current_artifact, {"status": "manifest_ready"}

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@example.invalid/private.git",
            private_repo_branch=DEFAULT_PRIVATE_REPO_BRANCH,
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert ownership_reads == 2
    assert result["ok"] is False
    assert result["status"] == "candidate_source_publication_pending"
    assert result["candidate_id"] == "candidate:interleaved"
    assert result["score_bundle_id"] == "score-bundle:interleaved"
    assert result["source_commit_event_id"] == (
        "commit-event-started-during-preflight"
    )
    assert store.version_event_writes == []
    assert store.version_writes == []


async def test_repo_head_sync_allows_completed_candidate_owned_commit(store, monkeypatch):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [
        {
            **_active_row(previous_artifact),
            "git_commit_sha": previous_artifact.git_commit_sha,
        }
    ]
    store.select_many_results["research_lab_private_repo_commit_events"] = [
        {
            "commit_event_id": "commit-event-complete",
            "commit_status": "pushed",
            "git_commit_sha": current_artifact.git_commit_sha,
            "candidate_id": "candidate:complete",
            "score_bundle_id": "score-bundle:complete",
            "created_at": "2026-08-05T00:00:00+00:00",
        }
    ]
    store.select_many_results[
        "research_lab_candidate_promotion_events:active_version_created"
    ] = [
        {
            "promotion_event_id": "promotion-complete",
            "candidate_id": "candidate:complete",
            "source_score_bundle_id": "score-bundle:complete",
            "event_type": "active_version_created",
        }
    ]
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: current_artifact.git_commit_sha,
    )

    async def _manifest(*_args: Any, **_kwargs: Any):
        return current_artifact, {"status": "manifest_ready"}

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@example.invalid/private.git",
            private_repo_branch="",
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert result["ok"] is True
    assert result["status"] == "synced_active_model_to_repo_head"
    assert len(store.version_writes) == 1


async def test_repo_head_sync_fails_closed_when_candidate_ownership_is_unavailable(
    store, monkeypatch
):
    current_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "9" * 64,
        git_commit_sha="e" * 40,
    )
    previous_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "8" * 64,
        git_commit_sha="d" * 40,
    )
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [
        {
            **_active_row(previous_artifact),
            "git_commit_sha": previous_artifact.git_commit_sha,
        }
    ]
    store.select_many_results["research_lab_private_repo_commit_events"] = RuntimeError(
        "temporary ownership read failure"
    )
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: current_artifact.git_commit_sha,
    )

    async def _manifest(*_args: Any, **_kwargs: Any):
        return current_artifact, {"status": "manifest_ready"}

    monkeypatch.setattr(promotion, "_load_repo_head_current_manifest", _manifest)

    result = await sync_active_model_to_repo_head(
        _controller_config(
            private_repo_url="git@example.invalid/private.git",
            private_repo_branch="",
        ),
        actor_ref="test",
        dry_run=False,
    )

    assert result["ok"] is False
    assert result["status"] == "candidate_source_ownership_unavailable"
    assert "temporary ownership read failure" in result["error"]
    assert store.version_event_writes == []
    assert store.version_writes == []


async def test_candidate_source_ownership_scan_crosses_prior_page_ceiling(store):
    repo_sha = "e" * 40
    completed_count = 125
    store.select_many_results["research_lab_private_repo_commit_events"] = [
        {
            "commit_event_id": f"commit-event-{index:03d}",
            "commit_status": "pushed",
            "git_commit_sha": repo_sha,
            "candidate_id": f"candidate:{index:03d}",
            "score_bundle_id": f"score-bundle:{index:03d}",
            "created_at": f"2026-08-05T00:00:00.{index:06d}+00:00",
        }
        for index in range(completed_count + 1)
    ]
    completed_rows = [
        {
            "promotion_event_id": f"promotion-event-{index:03d}",
            "candidate_id": f"candidate:{index:03d}",
            "source_score_bundle_id": f"score-bundle:{index:03d}",
            "event_type": "active_version_created",
            "created_at": f"2026-08-05T00:01:00.{index:06d}+00:00",
        }
        for index in range(completed_count)
    ]

    def completed_rows_for_chunk(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        candidate_ids = next(
            set(spec[2])
            for spec in kwargs.get("filters") or ()
            if len(spec) == 3 and spec[:2] == ("candidate_id", "in")
        )
        return [
            row for row in completed_rows if row["candidate_id"] in candidate_ids
        ]

    store.select_many_results[
        "research_lab_candidate_promotion_events:active_version_created"
    ] = completed_rows_for_chunk

    pending = await promotion._pending_candidate_source_publication_for_repo_head(
        repo_sha
    )

    assert pending == {
        "candidate_id": "candidate:125",
        "score_bundle_id": "score-bundle:125",
        "commit_event_id": "commit-event-125",
        "git_commit_sha": repo_sha,
    }
    paginated_orders = [
        order_by
        for table, order_by in store.select_all_order_by_calls
        if table
        in {
            "research_lab_private_repo_commit_events",
            "research_lab_candidate_promotion_events",
        }
    ]
    assert paginated_orders == [
        (("created_at", True), ("commit_event_id", True)),
        (("created_at", True), ("promotion_event_id", True)),
        (("created_at", True), ("promotion_event_id", True)),
    ]
    completion_filters = [
        filters
        for table, filters in store.select_all_calls
        if table == "research_lab_candidate_promotion_events"
    ]
    assert completion_filters == [
        (
            ("event_type", "active_version_created"),
            (
                "candidate_id",
                "in",
                [f"candidate:{index:03d}" for index in range(100)],
            ),
        ),
        (
            ("event_type", "active_version_created"),
            (
                "candidate_id",
                "in",
                [f"candidate:{index:03d}" for index in range(100, completed_count + 1)],
            ),
        ),
    ]


async def test_private_source_upgrade_defaults_to_leadpoet_lab(store, monkeypatch):
    commit_events: list[dict[str, Any]] = []
    pushed: dict[str, Any] = {}

    async def _fake_commit_event(**kwargs: Any) -> dict[str, Any]:
        commit_events.append(kwargs)
        return kwargs

    def _fake_push(**kwargs: Any) -> dict[str, Any]:
        pushed.update(kwargs)
        return {
            "status": "pushed",
            "git_commit_sha": "f" * 40,
            "target_files": ["sourcing_model/core.py"],
            "source_diff_hash": "sha256:" + "1" * 64,
        }

    monkeypatch.setattr(
        promotion,
        "create_private_repo_commit_event",
        _fake_commit_event,
    )
    monkeypatch.setattr(promotion, "_push_candidate_source_diff_to_repo", _fake_push)

    artifact = _valid_fake_artifact()
    controller = ResearchLabPromotionController(
        _controller_config(
            auto_commit_enabled=True,
            private_repo_url="https://github.com/leadpoet/Sourcing_model.git",
            private_repo_branch="",
        ),
        worker_ref="test-worker",
    )
    result = await controller._maybe_push_private_repo_candidate(
        candidate={
            "candidate_id": "cand-branch",
            "candidate_source_diff_hash": "sha256:" + "2" * 64,
            "candidate_build_doc": {},
            "candidate_model_manifest_doc": {},
        },
        score_bundle_row={"score_bundle_id": "bundle-branch"},
        score_bundle={},
        active=ActivePrivateModel(artifact=artifact),
        new_artifact=artifact,
        active_parent=artifact.model_artifact_hash,
        candidate_parent=artifact.model_artifact_hash,
        rolling_window_hash="sha256:" + "3" * 64,
        improvement_points=2.0,
        threshold=1.0,
    )

    assert result["status"] == "pushed"
    assert pushed["branch_name"] == DEFAULT_PRIVATE_REPO_BRANCH
    assert commit_events
    assert {
        event["branch_name"] for event in commit_events
    } == {DEFAULT_PRIVATE_REPO_BRANCH}


def _bridge_baseline_row(window_hash: str, baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "benchmark_bundle_id": baseline_bundle_id,
        "benchmark_date": "2026-07-02",
        "rolling_window_hash": window_hash,
        "evaluation_epoch": 23697,
        "benchmark_attempt": 0,
        "benchmark_quality": "passed",
        "aggregate_score": 16.353333,
        "current_benchmark_status": "completed",
        "score_summary_doc": {
            "schema_version": "1.0",
            "aggregate_score": 16.353333,
            "per_icp_summaries": [
                {
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "score": 10.0,
                    "company_count": 1,
                    "industry": "Software",
                    "sub_industry": "Sales Software",
                    "country": "United States",
                    "company_size_bucket": "51-200",
                    "intent_category_bucket": "vendor_replacement",
                    "diagnostics": {"failure_categories": []},
                },
                {
                    "icp_ref": "icp:b",
                    "icp_hash": "sha256:" + "b" * 64,
                    "score": 22.706666,
                    "company_count": 1,
                    "industry": "Healthcare",
                    "sub_industry": "Clinics",
                    "country": "United States",
                    "company_size_bucket": "201-500",
                    "intent_category_bucket": "growth",
                    "diagnostics": {"failure_categories": []},
                },
            ],
            "visibility_split": {
                "schema_version": "1.0",
                "split_policy": "test_split",
                "rolling_window_hash": window_hash,
                "public_count": 1,
                "private_count": 1,
                "public_strength_counts": {"weak": 1},
                "private_strength_counts": {"strong": 1},
                "items": [
                    {
                        "item_rank": 1,
                        "icp_ref": "icp:a",
                        "icp_hash": "sha256:" + "a" * 64,
                        "set_id": 1,
                        "day_index": 1,
                        "day_rank": 1,
                        "score": 10.0,
                        "visibility": "public",
                        "strength_label": "weak",
                    },
                    {
                        "item_rank": 2,
                        "icp_ref": "icp:b",
                        "icp_hash": "sha256:" + "b" * 64,
                        "set_id": 1,
                        "day_index": 1,
                        "day_rank": 2,
                        "score": 22.706666,
                        "visibility": "private",
                        "strength_label": "strong",
                    },
                ],
            },
        },
    }


def _bridge_public_report_row(baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "report_id": "public_benchmark:sha256:" + "6" * 64,
        "benchmark_bundle_id": baseline_bundle_id,
        "current_report_status": "published",
        "report_doc": {
            "public_icps": [
                {
                    "item_rank": 1,
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "set_id": 1,
                    "day_index": 1,
                    "day_rank": 1,
                    "score": 10.0,
                    "company_count": 1,
                    "strength_label": "weak",
                    "icp": {"industry": "Software"},
                    "diagnostics": {"failure_categories": []},
                }
            ],
        },
    }


def _bridge_score_bundle(candidate_artifact: FakeArtifact, window_hash: str, baseline_bundle_id: str) -> dict[str, Any]:
    return {
        "candidate_artifact_hash": candidate_artifact.model_artifact_hash,
        "parent_artifact_hash": "sha256:" + "a" * 64,
        "private_model_manifest_hash": "sha256:" + "b" * 64,
        "icp_set_hash": window_hash,
        "evaluation_epoch": 23697,
        "score_bundle_hash": "sha256:" + "5" * 64,
        "aggregates": {
            "per_icp_results": [
                {
                    "icp_ref": "icp:a",
                    "icp_hash": "sha256:" + "a" * 64,
                    "candidate_company_scores": [30.0, 20.0],
                    "failure_reason": "",
                },
                {
                    "icp_ref": "icp:b",
                    "icp_hash": "sha256:" + "b" * 64,
                    "candidate_company_scores": [35.945454, 28.945454],
                    "failure_reason": "",
                },
            ],
        },
        "private_holdout_gate": _approved_gate(
            baseline_benchmark_bundle_id=baseline_bundle_id,
            baseline_aggregate_score=16.353333,
            candidate_total_score=28.472727,
            candidate_delta_vs_daily_baseline=12.119394,
            reference_evaluation_mode="stored_daily_baseline",
        ),
    }


@pytest.fixture
def controller_env(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    artifact = FakeArtifact()

    async def _fake_load_active(config: Any, *, register_bootstrap: bool = False) -> ActivePrivateModel:
        return ActivePrivateModel(artifact=artifact, version_row=_active_row(artifact))

    async def _no_pending_activation(
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(promotion, "load_active_private_model", _fake_load_active)
    monkeypatch.setattr(
        promotion,
        "_active_version_has_pending_candidate_activation",
        _no_pending_activation,
    )

    async def _fake_compare_metric(
        *,
        epoch_id: int,
        score_bundle: Mapping[str, Any],
        expected_improvement_points: float,
        expected_event_doc: Mapping[str, Any],
        parent_receipt_hashes: list[str] | None = None,
    ) -> dict[str, Any]:
        metric = promotion_improvement_metric(score_bundle)
        assert expected_improvement_points == float(metric.improvement_points)
        assert dict(expected_event_doc) == metric.event_doc()
        assert parent_receipt_hashes in (None, [])
        return {
            "result": {
                "improvement_points": expected_improvement_points,
                "event_doc": dict(expected_event_doc),
            },
            "receipt_graph": {"root_receipt_hash": "sha256:" + "9" * 64},
            "epoch_id": int(epoch_id),
        }

    async def _fake_compare_gate(
        *,
        epoch_id: int,
        score_bundle: Mapping[str, Any],
        decision_payload: Mapping[str, Any],
        expected_decision: Mapping[str, Any],
        metric_outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = promotion.promotion_gate_decision(
            score_bundle,
            **dict(decision_payload),
        )
        assert dict(expected_decision) == expected.to_dict()
        assert int(metric_outcome["epoch_id"]) == int(epoch_id)
        return {"result": dict(expected_decision)}

    monkeypatch.setattr(promotion, "compare_promotion_metric", _fake_compare_metric)
    monkeypatch.setattr(
        promotion,
        "compare_promotion_gate_decision",
        _fake_compare_gate,
    )
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = []
    controller = ResearchLabPromotionController(_controller_config(), worker_ref="test-worker")
    return {"artifact": artifact, "controller": controller, "store": store}


async def test_pending_exact_activation_resumes_before_stale_parent_gate(
    controller_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact: FakeArtifact = controller_env["artifact"]
    controller: ResearchLabPromotionController = controller_env["controller"]
    pending_event = {
        "event_type": "active",
        "version_status": "active",
        "event_doc": {
            "activation_mode": (
                promotion.PRIVATE_MODEL_ACTIVATION_MODE_IMMUTABLE_CANDIDATE
            )
        },
    }
    calls: list[dict[str, Any]] = []

    async def _pending(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return pending_event

    async def _resume(self: Any, **kwargs: Any) -> dict[str, Any]:
        del self
        calls.append(dict(kwargs))
        assert kwargs["resume_activation_event"] is pending_event
        return {"status": "merged", "private_model_version_id": "version:exact"}

    monkeypatch.setattr(
        promotion,
        "_active_version_has_pending_candidate_activation",
        _pending,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_promote_built_image_candidate",
        _resume,
    )
    candidate = {
        **_candidate(artifact),
        "parent_artifact_hash": "sha256:" + "0" * 64,
    }

    result = await controller.process_scored_candidate(
        candidate=candidate,
        score_bundle_row={"score_bundle_id": "sb-exact"},
        score_bundle=_score_bundle(_approved_gate()),
    )

    assert result == {
        "status": "merged",
        "private_model_version_id": "version:exact",
        "activation_resume": True,
    }
    assert len(calls) == 1
    assert not any(
        event.get("event_type") == "stale_parent_detected"
        for event in controller_env["store"].promotion_event_writes
    )


async def test_pending_activation_with_invalid_event_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _invalid_event(_row: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("private model active target event is unavailable")

    monkeypatch.setattr(
        promotion,
        "_load_current_private_model_activation_event",
        _invalid_event,
    )

    with pytest.raises(
        promotion.PromotionPausedError,
        match="pending activation evidence is unavailable",
    ):
        await promotion._active_version_has_pending_candidate_activation(
            SimpleNamespace(version_row={"private_model_version_id": "version:bad"}),
            candidate_id="candidate:bad",
            score_bundle_id="score-bundle:bad",
        )


async def test_baseline_health_gate_failed_does_not_hold(controller_env, monkeypatch):
    """Score-only: an unhealthy daily baseline (gate_passed=False) is audit
    metadata — the merge proceeds on the stored score alone."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc(
            {"icp:a": 10.0},
            baseline_health={"unresolved_provider_errors": 7, "gate_passed": False},
        ),
    }

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(_approved_gate()),
    )
    assert result["status"] == "merged"
    held = [
        event
        for event in store.promotion_event_writes
        if str((event.get("event_doc") or {}).get("reason") or "").startswith("held_")
    ]
    assert held == []


async def test_baseline_health_absent_is_tolerated(controller_env):
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),  # legacy: no baseline_health
    }
    # Below-threshold delta proves the flow passed the health gate.
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_baseline_doc_not_consulted_on_merge_path(controller_env):
    """Score-only: the merge path never fetches the baseline per-ICP doc, so a
    store error on that table cannot hold the decision. The below-threshold
    rejection proves the flow ran to its normal conclusion."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = RuntimeError("postgrest reset")
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_quarantine_events_do_not_block_score_only_merge(controller_env, monkeypatch):
    """Score-only: historical scoring_health_quarantined bookkeeping does not
    veto the merge — the stored score decides."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),
    }
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = [
        {
            "promotion_event_id": "pe-q1",
            "event_type": "scoring_health_quarantined",
            "promotion_status": "rejected",
            "source_score_bundle_id": "sb-1",
        }
    ]

    async def _fake_promote(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(ResearchLabPromotionController, "_promote_built_image_candidate", _fake_promote)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(_approved_gate()),
    )
    assert result["status"] == "merged"
    blocked = [e for e in store.promotion_event_writes if e["event_type"] == "scoring_health_quarantined"]
    assert blocked == []


async def test_bypass_gates_param_cannot_waive_the_threshold(controller_env):
    """bypass_gates is accepted for replay-command compatibility, but with the
    health/quarantine gates retired the score threshold is the decision — and
    it can never be bypassed."""
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc(
            {"icp:a": 10.0},
            baseline_health={"unresolved_provider_errors": 7, "gate_passed": False},
        ),
    }
    store.select_many_results["research_lab_candidate_promotion_events:scoring_health_quarantined"] = [
        {"promotion_event_id": "pe-q1", "source_score_bundle_id": "sb-1"}
    ]
    gate = _approved_gate(candidate_total_score=10.2, candidate_delta_vs_daily_baseline=0.2)
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
        bypass_gates=frozenset({"scoring_health_quarantine", "baseline_health"}),
    )
    assert result["status"] == "rejected_below_threshold"


async def test_basis_unavailable_rejected_on_merge_path(controller_env):
    store: FakeStore = controller_env["store"]
    artifact: FakeArtifact = controller_env["artifact"]
    store.select_one_results["research_lab_private_model_benchmark_current"] = {
        "benchmark_bundle_id": "bb-1",
        "score_summary_doc": _baseline_doc({"icp:a": 10.0}),
    }
    gate = _approved_gate(
        baseline_aggregate_score=None,
        candidate_total_score=None,
        candidate_delta_vs_daily_baseline=None,
    )
    result = await controller_env["controller"].process_scored_candidate(
        candidate=_candidate(artifact),
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle=_score_bundle(gate),
    )
    assert result["status"] == "rejected_basis_unavailable"
    rejected = [
        e
        for e in store.promotion_event_writes
        if (e.get("event_doc") or {}).get("reason") == "rejected_basis_unavailable"
    ]
    assert len(rejected) == 1
    assert rejected[0]["event_type"] == "below_threshold"
    assert rejected[0]["promotion_status"] == "rejected"


# ---------------------------------------------------------------------------
# Promoted candidate benchmark bridge
# ---------------------------------------------------------------------------


async def test_promoted_candidate_writes_derived_benchmark_and_links_active_version(store, monkeypatch):
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    activation_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "e" * 64,
        git_commit_sha="e" * 40,
        manifest_uri="s3://bucket/research-lab/sourcing-model/current.json",
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "e" * 64
        ),
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    def _public_report_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("benchmark_bundle_id") == baseline_bundle_id:
            return [_bridge_public_report_row(baseline_bundle_id)]
        return []

    store.select_many_results["research_lab_public_benchmark_report_current"] = _public_report_rows

    async def _fake_push(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "private_source_pushed", "git_commit_sha": "e" * 40}

    async def _fake_wait(
        config: Any,
        *,
        expected_git_sha: str,
        timeout_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> tuple[FakeArtifact, dict[str, Any]]:
        assert expected_git_sha == "e" * 40
        return activation_artifact, {
            "status": "manifest_ready",
            "expected_git_sha": expected_git_sha,
            "current_json_git_sha": activation_artifact.git_commit_sha,
            "current_json_manifest_hash": activation_artifact.manifest_hash,
            "current_json_model_artifact_hash": activation_artifact.model_artifact_hash,
            "current_json_image_digest": activation_artifact.image_digest,
            "manifest_uri": activation_artifact.manifest_uri,
        }

    async def _fake_reward(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"champion_reward_status": "created", "champion_reward_id": "cr-1"}

    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_push_private_repo_candidate", _fake_push)
    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_create_champion_reward", _fake_reward)
    monkeypatch.setattr(promotion, "wait_for_current_manifest_git_sha", _fake_wait)
    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )

    controller = ResearchLabPromotionController(_controller_config(auto_commit_enabled=True), worker_ref="test-worker")
    parent_row = _active_row(parent)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [parent_row]
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [parent_row]
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        "candidate_source_diff_hash": "sha256:" + "2" * 64,
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }
    score_bundle = _bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id)
    result = await controller._promote_built_image_candidate(
        candidate=candidate,
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=score_bundle,
        active=ActivePrivateModel(artifact=parent, version_row=parent_row),
        active_parent=parent.model_artifact_hash,
        candidate_parent=parent.model_artifact_hash,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert result["status"] == "merged"
    assert len(store.private_benchmark_writes) == 1
    benchmark_write = store.private_benchmark_writes[0]
    assert benchmark_write["private_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert benchmark_write["private_model_manifest_hash"] == activation_artifact.manifest_hash
    assert benchmark_write["aggregate_score"] == pytest.approx(28.472727)
    assert benchmark_write["benchmark_quality"] == "passed"
    summary_doc = benchmark_write["score_summary_doc"]
    assert summary_doc["source"] == "promoted_candidate_score_bundle"
    assert summary_doc["derived_from_candidate_score"] is True
    assert summary_doc["source_score_bundle_id"] == "score_bundle:" + "7" * 64
    assert summary_doc["source_candidate_artifact_hash"] == candidate_artifact.model_artifact_hash
    assert summary_doc["activation_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert summary_doc["activation_manifest_hash"] == activation_artifact.manifest_hash
    assert summary_doc["activation_git_commit_sha"] == "e" * 40
    assert summary_doc["activation_artifact_differs_from_scored_candidate"] is True
    assert len(store.public_report_writes) == 1
    report_doc = store.public_report_writes[0]["report_doc"]
    assert report_doc["aggregate_score"] == pytest.approx(28.472727)
    assert report_doc["source"] == "promoted_candidate_score_bundle"
    assert report_doc["source_candidate_artifact_hash"] == candidate_artifact.model_artifact_hash
    assert report_doc["activation_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert report_doc["activation_artifact_differs_from_scored_candidate"] is True
    assert report_doc["public_icps"][0]["score"] == pytest.approx(25.0)
    assert len(store.version_writes) == 1
    assert store.version_writes[0]["source_benchmark_bundle_id"] == "private_benchmark:" + "8" * 64
    assert store.version_writes[0]["manifest_uri"] == activation_artifact.manifest_uri
    assert store.version_writes[0]["artifact_manifest"]["git_commit_sha"] == "e" * 40
    assert store.version_writes[0]["artifact_manifest"]["model_artifact_hash"] == (
        activation_artifact.model_artifact_hash
    )
    lineage_active = [
        event
        for event in store.version_event_writes
        if event["version_status"] == "active"
    ]
    assert lineage_active[0]["event_doc"]["activation_mode"] == (
        promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD
    )
    version_doc = store.version_writes[0]["redacted_version_doc"]
    _assert_db_doc_safe(version_doc)
    assert version_doc["image_ref_hash"].startswith("sha256:")
    assert version_doc["manifest_wait_status"]["current_json_image_ref_hash"].startswith("sha256:")
    active_events = [event for event in store.promotion_event_writes if event["event_type"] == "active_version_created"]
    _assert_db_doc_safe(active_events[0]["event_doc"])
    assert active_events[0]["event_doc"]["derived_benchmark_bundle_id"] == "private_benchmark:" + "8" * 64
    assert active_events[0]["event_doc"]["scored_candidate_model_artifact_hash"] == (
        candidate_artifact.model_artifact_hash
    )
    assert active_events[0]["event_doc"]["new_model_artifact_hash"] == activation_artifact.model_artifact_hash
    assert active_events[0]["event_doc"]["new_image_ref_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("auto_commit_enabled", "repo_url", "expected_fence"),
    [
        (False, "git@example.invalid/private.git", "remote_branch_stable"),
        (True, "", "immutable_manifest_only"),
    ],
)
async def test_no_push_promotion_uses_explicit_immutable_candidate_mode(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
    auto_commit_enabled: bool,
    repo_url: str,
    expected_fence: str,
) -> None:
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
        manifest_uri="s3://bucket/immutable-candidate.json",
    )
    parent_row = _active_row(parent)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [parent_row]
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [parent_row]

    async def _bridge(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "created",
            "benchmark_bundle_id": "private_benchmark:" + "8" * 64,
        }

    async def _reward(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"champion_reward_status": "created"}

    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_create_promoted_candidate_benchmark_bridge",
        _bridge,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_maybe_create_champion_reward",
        _reward,
    )
    controller = ResearchLabPromotionController(
        _controller_config(
            auto_commit_enabled=auto_commit_enabled,
            private_repo_url=repo_url,
        ),
        worker_ref="test-worker",
    )

    result = await controller._promote_built_image_candidate(
        candidate={
            "candidate_id": "candidate:" + "1" * 64,
            "parent_artifact_hash": parent.model_artifact_hash,
            "candidate_kind": "image_build",
            "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        },
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={
            "candidate_artifact_hash": candidate_artifact.model_artifact_hash
        },
        active=ActivePrivateModel(artifact=parent, version_row=parent_row),
        active_parent=parent.model_artifact_hash,
        candidate_parent=parent.model_artifact_hash,
        rolling_window_hash="sha256:" + "3" * 64,
        improvement_points=2.0,
        threshold=1.0,
    )

    assert result["status"] == "merged"
    active_lineage_events = [
        event
        for event in store.version_event_writes
        if event["version_status"] == "active"
    ]
    assert len(active_lineage_events) == 1
    activation_doc = active_lineage_events[0]["event_doc"]
    assert activation_doc["activation_mode"] == (
        promotion.PRIVATE_MODEL_ACTIVATION_MODE_IMMUTABLE_CANDIDATE
    )
    assert activation_doc["activation_branch_fence_mode"] == expected_fence


async def test_promoted_candidate_source_push_pending_leaves_previous_active_model_active(store, monkeypatch):
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    async def _fake_push(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "private_source_pushed", "git_commit_sha": "e" * 40}

    async def _fake_wait(
        config: Any,
        *,
        expected_git_sha: str,
        timeout_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> tuple[None, dict[str, Any]]:
        assert expected_git_sha == "e" * 40
        return None, {
            "status": "source_pushed_manifest_pending",
            "expected_git_sha": expected_git_sha,
            "current_json_git_sha": "d" * 40,
        }

    async def _fake_reward(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("champion reward must not be created while source manifest is pending")

    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_push_private_repo_candidate", _fake_push)
    monkeypatch.setattr(ResearchLabPromotionController, "_maybe_create_champion_reward", _fake_reward)
    monkeypatch.setattr(promotion, "wait_for_current_manifest_git_sha", _fake_wait)
    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )

    controller = ResearchLabPromotionController(_controller_config(auto_commit_enabled=True), worker_ref="test-worker")
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        "candidate_source_diff_hash": "sha256:" + "2" * 64,
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }
    result = await controller._promote_built_image_candidate(
        candidate=candidate,
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id),
        active=ActivePrivateModel(artifact=parent, version_row=_active_row(parent)),
        active_parent=parent.model_artifact_hash,
        candidate_parent=parent.model_artifact_hash,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert result["status"] == "source_pushed_manifest_pending"
    assert store.private_benchmark_writes == []
    assert store.public_report_writes == []
    assert store.version_writes == []
    pending_events = [
        event
        for event in store.promotion_event_writes
        if (event.get("event_doc") or {}).get("reason") == "source_pushed_manifest_pending"
    ]
    assert len(pending_events) == 1
    assert pending_events[0]["event_doc"]["action"] == (
        "leave_previous_active_model_active_until_current_json_matches_pushed_commit"
    )


@pytest.mark.parametrize(
    ("admission_mode", "drift"),
    [
        pytest.param("semantic_v1", "pointer", id="semantic-forward-pointer-drift"),
        pytest.param("legacy_exact", "branch", id="legacy-rollback-branch-drift"),
        pytest.param("legacy_exact", "receipt", id="old-restoration-receipt-drift"),
    ],
)
async def test_promoted_candidate_revalidates_after_bridge_before_lineage_write(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
    admission_mode: str,
    drift: str,
) -> None:
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact_for_mode(
        admission_mode,
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    activation_artifact = _valid_fake_artifact_for_mode(
        admission_mode,
        model_artifact_hash="sha256:" + "e" * 64,
        git_commit_sha="e" * 40,
        manifest_uri="s3://bucket/research-lab/sourcing-model/current.json",
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "e" * 64
        ),
    )
    changed_pointer = _valid_fake_artifact_for_mode(
        admission_mode,
        model_artifact_hash="sha256:" + "f" * 64,
        git_commit_sha=activation_artifact.git_commit_sha,
        manifest_uri=activation_artifact.manifest_uri,
        image_digest=(
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/test@sha256:"
            + "f" * 64
        ),
    )
    pointer = [activation_artifact]
    branch = [activation_artifact.git_commit_sha]
    receipt_generation = [0]
    timeline: list[str] = []

    async def _push_source(self: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "private_source_pushed",
            "git_commit_sha": activation_artifact.git_commit_sha,
        }

    async def _wait_for_pointer(
        _config: Any,
        *,
        expected_git_sha: str,
        timeout_seconds: int | None = None,
        poll_seconds: int | None = None,
    ) -> tuple[FakeArtifact, dict[str, Any]]:
        del timeout_seconds, poll_seconds
        assert expected_git_sha == activation_artifact.git_commit_sha
        return activation_artifact, {"status": "manifest_ready"}

    async def _admit_source(
        admitted_artifact: FakeArtifact,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        del timeout_seconds
        receipt_generation[0] += 1
        timeline.append(f"admit:{receipt_generation[0]}")
        suffix = (
            "-changed"
            if drift == "receipt" and receipt_generation[0] > 2
            else ""
        )
        return _compatibility_receipt(
            admitted_artifact,
            admission_mode=admission_mode,
            binding_suffix=suffix,
        )

    def _load_pointer(_uri: str) -> FakeArtifact:
        timeline.append("pointer")
        return pointer[0]

    def _resolve_branch(**_kwargs: Any) -> str:
        timeline.append("branch")
        return branch[0]

    async def _bridge(self: Any, **_kwargs: Any) -> dict[str, Any]:
        timeline.append("bridge")
        if drift == "pointer":
            pointer[0] = changed_pointer
        elif drift == "branch":
            branch[0] = "f" * 40
        return {
            "status": "created",
            "benchmark_bundle_id": "private_benchmark:" + "8" * 64,
        }

    async def _unexpected_reward(self: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("reward must not follow failed final compatibility preflight")

    monkeypatch.setattr(
        promotion,
        "_preflight_private_model_activation",
        _REAL_PREFLIGHT_PRIVATE_MODEL_ACTIVATION,
    )
    monkeypatch.setattr(
        model_authority_v2,
        "preflight_private_model_compatibility_v2",
        _admit_source,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", _load_pointer)
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        _resolve_branch,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_maybe_push_private_repo_candidate",
        _push_source,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_create_promoted_candidate_benchmark_bridge",
        _bridge,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_maybe_create_champion_reward",
        _unexpected_reward,
    )
    monkeypatch.setattr(
        promotion,
        "wait_for_current_manifest_git_sha",
        _wait_for_pointer,
    )
    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )

    controller = ResearchLabPromotionController(
        _controller_config(auto_commit_enabled=True),
        worker_ref="test-worker",
    )
    parent_row = _active_row(parent)
    store.select_many_results[
        "research_lab_private_model_version_current:active"
    ] = [parent_row]
    store.select_many_results[
        "research_lab_private_model_version_current:unfiltered"
    ] = [parent_row]
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
        "candidate_source_diff_hash": "sha256:" + "2" * 64,
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
    }

    with pytest.raises(RuntimeError):
        await controller._promote_built_image_candidate(
            candidate=candidate,
            score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
            score_bundle={
                "candidate_artifact_hash": candidate_artifact.model_artifact_hash
            },
            active=ActivePrivateModel(
                artifact=parent,
                version_row=parent_row,
            ),
            active_parent=parent.model_artifact_hash,
            candidate_parent=parent.model_artifact_hash,
            rolling_window_hash="sha256:" + "3" * 64,
            improvement_points=2.0,
            threshold=1.0,
        )

    assert timeline[:5] == [
        "admit:1",
        "admit:2",
        "pointer",
        "branch",
        "bridge",
    ]
    assert timeline.count("admit:3") == 1
    assert store.version_event_writes == []
    assert store.version_writes == []


async def test_promoted_candidate_rejects_unverified_artifact_before_side_effects(
    store,
    monkeypatch,
):
    from research_lab.eval import PrivateModelRuntimeError

    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )

    async def _unexpected_push(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("private source push must not precede artifact verification")

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_maybe_push_private_repo_candidate",
        _unexpected_push,
    )
    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PrivateModelRuntimeError("consumer contract/parity mismatch")
        ),
    )

    controller = ResearchLabPromotionController(
        _controller_config(auto_commit_enabled=False),
        worker_ref="test-worker",
    )
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_kind": "image_build",
        "candidate_model_manifest_doc": candidate_artifact.to_dict(),
    }
    with pytest.raises(
        PrivateModelRuntimeError,
        match="consumer contract/parity mismatch",
    ):
        await controller._promote_built_image_candidate(
            candidate=candidate,
            score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
            score_bundle={
                "candidate_artifact_hash": candidate_artifact.model_artifact_hash
            },
            active=ActivePrivateModel(
                artifact=parent,
                version_row=_active_row(parent),
            ),
            active_parent=parent.model_artifact_hash,
            candidate_parent=parent.model_artifact_hash,
            rolling_window_hash="sha256:" + "3" * 64,
            improvement_points=2.0,
            threshold=1.0,
        )

    for writes in (
        store.version_writes,
        store.version_event_writes,
        store.promotion_event_writes,
        store.candidate_evaluation_event_writes,
        store.scoring_dispatch_event_writes,
        store.reward_obligation_writes,
        store.private_benchmark_writes,
        store.public_report_writes,
        store.generic_insert_writes,
    ):
        assert writes == []


async def test_incompatible_signed_candidate_is_rejected_before_source_push(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = FakeArtifact()
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )

    async def _reject_compatibility(_artifact: FakeArtifact) -> dict[str, Any]:
        raise RuntimeError("measured compatibility probe rejected")

    async def _unexpected_push(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("incompatible candidate must not mutate source")

    monkeypatch.setattr(
        promotion,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        promotion,
        "_preflight_private_artifact_compatibility",
        _reject_compatibility,
    )
    monkeypatch.setattr(
        ResearchLabPromotionController,
        "_maybe_push_private_repo_candidate",
        _unexpected_push,
    )
    controller = ResearchLabPromotionController(
        _controller_config(auto_commit_enabled=True),
        worker_ref="test-worker",
    )

    with pytest.raises(RuntimeError, match="compatibility probe rejected"):
        await controller._promote_built_image_candidate(
            candidate={
                "candidate_id": "candidate:" + "1" * 64,
                "parent_artifact_hash": parent.model_artifact_hash,
                "candidate_kind": "image_build",
                "candidate_model_manifest_doc": candidate_artifact.to_dict(),
            },
            score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
            score_bundle={
                "candidate_artifact_hash": candidate_artifact.model_artifact_hash
            },
            active=ActivePrivateModel(
                artifact=parent,
                version_row=_active_row(parent),
            ),
            active_parent=parent.model_artifact_hash,
            candidate_parent=parent.model_artifact_hash,
            rolling_window_hash="sha256:" + "3" * 64,
            improvement_points=2.0,
            threshold=1.0,
        )

    assert store.version_writes == []
    assert store.version_event_writes == []
    assert store.promotion_event_writes == []
    assert store.private_benchmark_writes == []
    assert store.public_report_writes == []
    assert store.reward_obligation_writes == []


async def test_promoted_candidate_bridge_reuses_existing_rows_without_duplicate_writes(store):
    candidate_artifact = _valid_fake_artifact(
        model_artifact_hash="sha256:" + "c" * 64,
        git_commit_sha="d" * 40,
    )
    window_hash = "sha256:" + "3" * 64
    baseline_bundle_id = "private_benchmark:" + "6" * 64
    existing_benchmark_id = "private_benchmark:" + "8" * 64
    existing_report_id = "public_benchmark:sha256:" + "9" * 64
    store.select_one_results["research_lab_private_model_benchmark_current"] = _bridge_baseline_row(
        window_hash,
        baseline_bundle_id,
    )

    def _private_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("private_model_manifest_hash") == candidate_artifact.manifest_hash:
            return [
                {
                    "benchmark_bundle_id": existing_benchmark_id,
                    "current_benchmark_status": "completed",
                    "aggregate_score": 28.472727,
                }
            ]
        return []

    def _public_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = {item[0]: item[1] for item in kwargs.get("filters") or () if len(item) == 2}
        if filters.get("private_model_manifest_hash") == candidate_artifact.manifest_hash:
            return [
                {
                    "report_id": existing_report_id,
                    "benchmark_bundle_id": existing_benchmark_id,
                    "current_report_status": "published",
                    "aggregate_score": 28.472727,
                }
            ]
        if filters.get("benchmark_bundle_id") == baseline_bundle_id:
            return [_bridge_public_report_row(baseline_bundle_id)]
        return []

    store.select_many_results["research_lab_private_model_benchmark_current"] = _private_rows
    store.select_many_results["research_lab_public_benchmark_report_current"] = _public_rows
    controller = ResearchLabPromotionController(_controller_config(), worker_ref="test-worker")
    bridge = await controller._create_promoted_candidate_benchmark_bridge(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_bridge_score_bundle(candidate_artifact, window_hash, baseline_bundle_id),
        new_artifact=candidate_artifact,
        rolling_window_hash=window_hash,
        improvement_points=12.119394,
        threshold=1.0,
    )
    assert bridge["status"] == "already_exists"
    assert bridge["benchmark_bundle_id"] == existing_benchmark_id
    assert bridge["public_report_id"] == existing_report_id
    assert store.private_benchmark_writes == []
    assert store.public_report_writes == []


# ---------------------------------------------------------------------------
# Champion reward start_epoch — windows start at creation time (2026-07-02
# backdating incident: a reward scored at epoch N but merged at N+15 paid
# ~2.5h of a ~24h window)
# ---------------------------------------------------------------------------


def _capture_obligation(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def _build(obligation_input: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
        captured.append(dict(obligation_input))
        return {"status": "active", "champion_reward_id": "cr-1", **obligation_input}

    monkeypatch.setattr(promotion, "build_champion_reward_obligation", _build)

    async def _load_promotion_graph(**_kwargs: Any) -> dict[str, Any]:
        return {"root_receipt_hash": "sha256:" + "a" * 64}

    async def _authorize_reward(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["decision_kind"] == "champion"
        assert kwargs["expected_result"]["reward"]["status"] == "active"
        assert "promotion_decision" in kwargs["decision_payload"]
        assert kwargs["artifact_kind"] == "champion_reward_decision"
        assert kwargs["parent_graphs"][0]["root_receipt_hash"] == "sha256:" + "a" * 64
        return {"status": "matched"}

    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        _load_promotion_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "authorize_reward_decision_v2",
        _authorize_reward,
    )
    return captured


def _epoch_config() -> Any:
    return SimpleNamespace(
        auto_promotion_enabled=True,
        auto_commit_enabled=False,
        improvement_threshold_points=1.0,
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        reimbursement_policy_doc=lambda enabled: {"policy_id": "policy-1"},
        lab_reward_epochs=20,
        evaluation_epoch=0,  # no operator override: the live epoch is resolved
    )


async def test_reward_start_epoch_uses_live_epoch_not_bundle_epoch(store, monkeypatch):
    captured = _capture_obligation(monkeypatch)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 150, None, "test"

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")
    result = await controller._maybe_create_champion_reward(
        candidate={
            "candidate_id": "cand-1",
            "miner_hotkey": "hk-1",
            "ticket_id": "ticket-1",
            "run_id": "run-1",
            "island": "generalist",
        },
        score_bundle_row={"score_bundle_id": "sb-1"},
        score_bundle={"evaluation_epoch": 100, "aggregates": {"per_icp_results": []}},
        improvement_points=2.5,
        threshold=1.0,
    )
    assert result["champion_reward_status"] == "created"
    assert len(captured) == 1
    # Scoring provenance is kept, but the window starts NOW: live 150 -> 151,
    # never bundle-epoch 100 -> 101 (which would expire 50 epochs pre-paid).
    assert captured[0]["evaluation_epoch"] == 100
    assert captured[0]["start_epoch"] == 151


async def test_legacy_champion_reward_is_rejected_before_persistence(
    store,
    monkeypatch,
):
    captured = _capture_obligation(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")

    async def _resolve(_hotkey: str) -> int | None:
        return 5

    async def _live_epoch(_configured: Any = None) -> tuple[int, int | None, str]:
        return 150, None, "test"

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")
    with pytest.raises(ResearchLabTeeProtocolError, match="V1 authority is retired"):
        await controller._maybe_create_champion_reward(
            candidate={
                "candidate_id": "cand-legacy",
                "miner_hotkey": "hk-legacy",
                "ticket_id": "ticket-legacy",
                "run_id": "run-legacy",
                "island": "generalist",
            },
            score_bundle_row={"score_bundle_id": "sb-legacy"},
            score_bundle={"evaluation_epoch": 100, "aggregates": {"per_icp_results": []}},
            improvement_points=2.5,
            threshold=1.0,
        )
    # The existing pure obligation calculation may run before protocol
    # validation, but no authoritative write is allowed.
    assert len(captured) == 1
    assert store.reward_obligation_writes == []


async def test_reward_start_epoch_fails_closed_when_chain_unreachable(
    store,
    monkeypatch,
):
    captured = _capture_obligation(monkeypatch)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    async def _broken_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        raise RuntimeError("subtensor unreachable")

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _broken_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")
    with pytest.raises(RuntimeError, match="subtensor unreachable"):
        await controller._maybe_create_champion_reward(
            candidate={
                "candidate_id": "cand-1",
                "miner_hotkey": "hk-1",
                "ticket_id": "ticket-1",
                "run_id": "run-1",
            },
            score_bundle_row={"score_bundle_id": "sb-1"},
            score_bundle={
                "evaluation_epoch": 100,
                "aggregates": {"per_icp_results": []},
            },
            improvement_points=2.5,
            threshold=1.0,
        )
    assert captured == []


async def test_stateful_champion_reward_fails_closed_when_epoch_authority_is_unavailable(
    store,
    monkeypatch,
):
    captured = _capture_obligation(monkeypatch)

    async def _resolve(_hotkey: str) -> int | None:
        return 5

    async def _broken_epoch(_configured: Any = None) -> tuple[int, int | None, str]:
        raise RuntimeError("subtensor unreachable")

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _broken_epoch)
    controller = ResearchLabPromotionController(_epoch_config(), worker_ref="test-worker")

    with pytest.raises(RuntimeError, match="subtensor unreachable"):
        await controller._maybe_create_champion_reward(
            candidate={
                "candidate_id": "cand-stateful",
                "miner_hotkey": "hk-stateful",
                "ticket_id": "ticket-stateful",
                "run_id": "run-stateful",
            },
            score_bundle_row={"score_bundle_id": "sb-stateful"},
            score_bundle={
                "evaluation_epoch": 100,
                "aggregates": {"per_icp_results": []},
            },
            improvement_points=2.5,
            threshold=1.0,
        )

    assert captured == []
    assert store.reward_obligation_writes == []


def _source_add_reward_config() -> Any:
    return SimpleNamespace(
        source_add_rewards_enabled=True,
        source_add_leg2_alpha_percent=5.0,
        lab_reward_epochs=20,
        evaluation_epoch=0,
    )


def _source_add_attribution_bundle() -> dict[str, Any]:
    return {
        "evaluation_epoch": 200,
        "aggregates": {"per_icp_results": []},
    }


def _install_source_add_v2_judge(
    monkeypatch,
    judge,
    *,
    receipt_overrides: Mapping[str, Any] | None = None,
    graph_overrides: Mapping[str, Any] | None = None,
    include_execution_authority: bool = True,
):
    async def _measured_judge(**kwargs: Any):
        verdict = await judge(**kwargs)
        result = {"verdict": verdict.verdict}
        output_root = sha256_json(result)
        receipt = {
            "receipt_hash": "sha256:" + "a" * 64,
            "output_root": output_root,
            "role": "gateway_scoring",
            "purpose": "research_lab.source_add_judge.v2",
            "status": "succeeded",
            **dict(receipt_overrides or {}),
        }
        execution_receipt_graph = {
            "root_receipt_hash": "sha256:" + "a" * 64,
            "receipts": [dict(receipt)],
            **dict(graph_overrides or {}),
        }
        outcome = {
            "receipt": {
                "receipt_hash": "sha256:" + "c" * 64,
                "output_root": output_root,
            },
            "receipt_graph": {
                "root_receipt_hash": "sha256:" + "c" * 64,
                "receipts": [],
            },
            "result": result,
        }
        if include_execution_authority:
            outcome.update(
                {
                    "execution_receipt": receipt,
                    "execution_receipt_graph": execution_receipt_graph,
                }
            )
        return verdict, outcome

    async def _persist_link(**_kwargs: Any):
        return {"business_artifact_link_count": 1}

    async def _authorize_reward(**kwargs: Any):
        assert kwargs["decision_kind"] == "source_add_leg2"
        assert kwargs["artifact_kind"] == "source_add_reward_decision"
        assert kwargs["expected_result"]["reward"]["leg"] == 2
        assert kwargs["parent_graphs"][0]["root_receipt_hash"] == "sha256:" + "a" * 64
        assert kwargs["parent_graphs"][0]["receipts"][0]["purpose"] == (
            "research_lab.source_add_judge.v2"
        )
        return {"status": "matched"}

    monkeypatch.setattr(
        v2_authority,
        "judge_source_add_implementation_v2",
        _measured_judge,
    )
    monkeypatch.setattr(
        v2_authority,
        "persist_source_add_judge_reward_link_v2",
        _persist_link,
    )
    monkeypatch.setattr(
        v2_authority,
        "authorize_reward_decision_v2",
        _authorize_reward,
    )


async def test_source_add_leg2_created_when_llm_judge_says_helped(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
            "accepted_at": "2026-07-06T00:00:00Z",
            "catalog_doc": {"market_open_at": "2026-07-20T00:00:00Z"},
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = []
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 250, None, "test"

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.91,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            evidence_summary="The winning change used the new API for sourcing evidence.",
            reason_codes=("matched_api_usage",),
            model_id="openai/gpt-5.6-sol",
        )

    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={
            "candidate_id": "candidate:" + "1" * 64,
            "miner_hotkey": "hk-implementer",
            "ticket_id": "ticket-1",
            "run_id": "run-1",
        },
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_source_add_attribution_bundle(),
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={"champion_reward_status": "created", "champion_reward_id": "cr-1"},
    )

    assert result["source_add_reward_status"] == "created"
    obligation_rows = [
        row for table, row in store.generic_insert_writes if table == "research_lab_source_add_reward_obligations"
    ]
    assert len(obligation_rows) == 1
    obligation = obligation_rows[0]
    assert obligation["adapter_id"] == "adapter:test-api-source"
    assert obligation["miner_hotkey"] == "hk-source-owner"
    assert obligation["leg"] == 2
    assert obligation["reward_kind"] == "source_implementation"
    assert obligation["alpha_percent"] == pytest.approx(5.0)
    assert obligation["start_epoch"] == 251
    assert obligation["trigger_evidence_doc"]["llm_judge_passed"] is True
    assert obligation["trigger_evidence_doc"]["llm_verdict"] == "helped"
    events = [event for event in store.promotion_event_writes if event["event_type"] == "promotion_checked"]
    source_event = next(
        event
        for event in events
        if (event["event_doc"] or {}).get("reason")
        == "source_add_leg2_reward_created"
    )
    assert source_event["event_doc"]["judge_receipt_hash"] == "sha256:" + "a" * 64
    assert source_event["event_doc"]["judge_output_root"] == sha256_json(
        {"verdict": "helped"}
    )


async def test_stateful_source_add_leg2_fails_closed_when_epoch_authority_is_unavailable(
    store,
    monkeypatch,
):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = []
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _broken_epoch(_configured: Any = None) -> tuple[int, int | None, str]:
        raise RuntimeError("subtensor unreachable")

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.91,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _broken_epoch)
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(
        _source_add_reward_config(),
        worker_ref="test-worker",
    )

    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_source_add_attribution_bundle(),
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        },
    )

    assert result["source_add_reward_status"] == "failed"
    assert result["error_class"] == "RuntimeError"
    assert not any(
        table == "research_lab_source_add_reward_obligations"
        for table, _row in store.generic_insert_writes
    )


async def test_source_add_leg2_blocks_when_llm_judge_says_not_helped(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="not_helped",
            confidence=0.82,
            source_used=False,
            evidence_summary="No evidence the new API affected the winning change.",
            reason_codes=("no_source_use",),
            model_id="openai/gpt-5.6-sol",
        )

    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={
            "candidate_id": "candidate:" + "1" * 64,
            "miner_hotkey": "hk-implementer",
            "ticket_id": "ticket-1",
            "run_id": "run-1",
            "candidate_model_manifest_doc": {"source_add_implementation_attribution": {"adapter_id": "fake"}},
        },
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={"champion_reward_status": "created", "champion_reward_id": "cr-1"},
    )
    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["llm_judge_not_helped"]
    assert store.generic_insert_writes == []
    source_event = next(
        event
        for event in store.promotion_event_writes
        if (event["event_doc"] or {}).get("reason")
        == "source_add_leg2_reward_blocked"
    )
    assert source_event["event_doc"]["judge_receipt_hash"] == "sha256:" + "a" * 64
    assert source_event["event_doc"]["judge_output_root"] == sha256_json(
        {"verdict": "not_helped"}
    )


async def test_source_add_leg2_replaces_unbound_legacy_event(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": "legacy-event",
            "event_doc": {
                "reason": "source_add_leg2_reward_blocked",
                "adapter_id": "",
            },
            "created_at": "2026-08-13T00:00:00Z",
        }
    ]

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="not_helped",
            confidence=0.82,
            source_used=False,
            model_id="openai/gpt-5.6-sol",
        )

    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(
        _source_add_reward_config(), worker_ref="test-worker"
    )
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        },
    )

    assert result["source_add_reward_status"] == "blocked"
    assert len(store.promotion_event_writes) == 1
    event_doc = store.promotion_event_writes[0]["event_doc"]
    assert event_doc["judge_receipt_hash"] == "sha256:" + "a" * 64
    assert event_doc["judge_output_root"] == sha256_json({"verdict": "not_helped"})


@pytest.mark.parametrize(
    ("receipt_overrides", "graph_overrides", "include_execution_authority"),
    [
        ({"receipt_hash": "invalid"}, None, True),
        (None, {"root_receipt_hash": "sha256:" + "c" * 64}, True),
        ({"output_root": "sha256:" + "b" * 64}, None, True),
        (None, None, False),
    ],
)
async def test_source_add_leg2_rejects_malformed_judge_authority(
    store,
    monkeypatch,
    receipt_overrides,
    graph_overrides,
    include_execution_authority,
):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = []
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.91,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="openai/gpt-5.6-sol",
        )

    _install_source_add_v2_judge(
        monkeypatch,
        _judge,
        receipt_overrides=receipt_overrides,
        graph_overrides=graph_overrides,
        include_execution_authority=include_execution_authority,
    )
    controller = ResearchLabPromotionController(
        _source_add_reward_config(), worker_ref="test-worker"
    )
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle=_source_add_attribution_bundle(),
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        },
    )

    assert result["source_add_reward_status"] == "failed"
    assert result["error_class"] == "RuntimeError"
    assert not any(
        table == "research_lab_source_add_reward_obligations"
        for table, _row in store.generic_insert_writes
    )


@pytest.mark.parametrize("verdict", ["not_helped", "uncertain"])
async def test_source_add_leg2_non_helped_and_uncertain_never_create_reward(
    store,
    monkeypatch,
    verdict,
):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict=verdict,
            confidence=0.5,
            source_used=(verdict == "uncertain"),
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={"champion_reward_status": "created", "champion_reward_id": "cr-1"},
    )
    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["llm_judge_not_helped"]
    assert store.generic_insert_writes == []


async def test_source_add_leg2_helped_verdict_must_match_provisioned_source(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:unknown",
            registry_provider_id="unknown_source",
            model_id="test/judge",
        )

    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={"champion_reward_status": "created", "champion_reward_id": "cr-1"},
    )
    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["llm_judge_source_not_matched"]
    assert store.generic_insert_writes == []


async def test_source_add_leg2_duplicate_is_idempotently_blocked(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = [
        {
            "reward_ref": "source_add_reward:" + "3" * 16,
            "adapter_id": "adapter:test-api-source",
            "leg": 2,
            "current_reward_status": "active",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 250, None, "test"

    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={"champion_reward_status": "already_created", "champion_reward_id": "cr-1"},
    )
    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["leg2_already_created"]
    assert store.generic_insert_writes == []


async def test_source_add_leg2_retry_activates_orphaned_obligation(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    reward_ref = "source_add_reward:" + "3" * 16
    store.select_many_results["research_lab_source_add_reward_current"] = [
        {
            "reward_ref": reward_ref,
            "adapter_id": "adapter:test-api-source",
            "leg": 2,
            "current_reward_status": None,
            "trigger_evidence_doc": {
                "llm_judge_passed": True,
                "llm_verdict": "helped",
                "source_used": True,
            },
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 250, None, "test"

    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(
        _source_add_reward_config(), worker_ref="test-worker"
    )
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={
            "champion_reward_status": "already_created",
            "champion_reward_id": "cr-1",
        },
    )

    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["leg2_already_created"]
    event_rows = [
        row
        for table, row in store.generic_insert_writes
        if table == "research_lab_source_add_reward_events"
    ]
    assert event_rows == [
        {
            "reward_ref": reward_ref,
            "seq": 0,
            "reward_status": "active",
            "reason": "leg2_llm_judge_helped",
        }
    ]


@pytest.mark.asyncio
async def test_source_add_leg2_orphan_is_repaired_by_periodic_reconciler(store):
    reward_ref = "source_add_reward:" + "3" * 16
    store.select_many_results["research_lab_source_add_reward_current"] = [
        {
            "reward_ref": reward_ref,
            "adapter_id": "adapter:test-api-source",
            "leg": 2,
            "current_reward_status": None,
            "trigger_evidence_doc": {
                "llm_judge_passed": True,
                "llm_verdict": "helped",
                "source_used": True,
            },
            "created_at": "2026-08-14T00:00:00+00:00",
        }
    ]

    preview = await reconcile_source_add_leg2_reward_activations(dry_run=True)
    assert preview["ok"] is True
    assert preview["planned_count"] == 1
    assert store.generic_insert_writes == []

    result = await reconcile_source_add_leg2_reward_activations(dry_run=False)
    assert result["ok"] is True
    assert result["repaired_count"] == 1
    assert store.generic_insert_writes == [
        (
            "research_lab_source_add_reward_events",
            {
                "reward_ref": reward_ref,
                "seq": 0,
                "reward_status": "active",
                "reason": "leg2_llm_judge_helped",
            },
        )
    ]


async def test_source_add_leg2_duplicate_obligation_repairs_activation_in_same_attempt(
    store, monkeypatch
):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = []
    store.select_many_results["research_lab_candidate_promotion_events"] = []
    reward_ref = "source_add_reward:" + "3" * 16
    original_insert = store.insert_row

    async def _racing_insert(table: str, row: dict[str, Any]) -> dict[str, Any]:
        if table == "research_lab_source_add_reward_obligations":
            store.generic_insert_writes.append((table, row))
            store.select_many_results[
                "research_lab_source_add_reward_current"
            ] = [
                {
                    "reward_ref": reward_ref,
                    "adapter_id": "adapter:test-api-source",
                    "leg": 2,
                    "current_reward_status": None,
                    "trigger_evidence_doc": {
                        "llm_judge_passed": True,
                        "llm_verdict": "helped",
                        "source_used": True,
                    },
                }
            ]
            raise RuntimeError("duplicate key value violates unique constraint")
        return await original_insert(table, row)

    monkeypatch.setattr(promotion, "insert_row", _racing_insert)
    monkeypatch.setattr(store_module, "insert_row", _racing_insert)

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 250, None, "test"

    monkeypatch.setattr(
        promotion, "resolve_research_lab_evaluation_epoch", _live_epoch
    )
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(
        _source_add_reward_config(), worker_ref="test-worker"
    )
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status={
            "champion_reward_status": "already_created",
            "champion_reward_id": "cr-1",
        },
    )

    assert result["source_add_reward_status"] == "blocked"
    assert result["results"][0]["blockers"] == ["leg2_already_created"]
    event_rows = [
        row
        for table, row in store.generic_insert_writes
        if table == "research_lab_source_add_reward_events"
    ]
    assert event_rows == [
        {
            "reward_ref": reward_ref,
            "seq": 0,
            "reward_status": "active",
            "reason": "leg2_llm_judge_helped",
        }
    ]


async def test_source_add_leg2_persistence_failure_is_non_blocking(store, monkeypatch):
    store.select_many_results["research_lab_source_add_provisioning_current"] = [
        {
            "provision_ref": "source_add_provision:" + "2" * 16,
            "catalog_id": "source_catalog:" + "1" * 16,
            "adapter_id": "adapter:test-api-source",
            "miner_hotkey": "hk-source-owner",
            "registry_provider_id": "test_api_source",
            "provision_status": "provisioned_autoresearch_eligible",
        }
    ]
    store.select_many_results["research_lab_source_add_reward_current"] = []
    store.select_many_results["research_lab_candidate_promotion_events"] = []

    async def _judge(**_kwargs: Any) -> SourceAddJudgeVerdict:
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test-api-source",
            registry_provider_id="test_api_source",
            model_id="test/judge",
        )

    async def _live_epoch(configured: Any = None) -> tuple[int, int | None, str]:
        return 250, None, "test"

    async def _failed_insert(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("database temporarily unavailable")

    monkeypatch.setattr(promotion, "resolve_research_lab_evaluation_epoch", _live_epoch)
    monkeypatch.setattr(promotion, "insert_row", _failed_insert)
    _install_source_add_v2_judge(monkeypatch, _judge)
    controller = ResearchLabPromotionController(_source_add_reward_config(), worker_ref="test-worker")
    champion_status = {"champion_reward_status": "created", "champion_reward_id": "cr-1"}
    result = await controller._maybe_create_source_add_implementation_rewards(
        candidate={"candidate_id": "candidate:" + "1" * 64},
        score_bundle_row={"score_bundle_id": "score_bundle:" + "7" * 64},
        score_bundle={"evaluation_epoch": 200, "aggregates": {}},
        improvement_points=2.0,
        threshold=1.0,
        champion_reward_status=champion_status,
    )
    assert result["source_add_reward_status"] == "failed"
    assert result["error_class"] == "RuntimeError"
    assert champion_status == {"champion_reward_status": "created", "champion_reward_id": "cr-1"}


# ---------------------------------------------------------------------------
# Failed private-source push reconciler
# ---------------------------------------------------------------------------


def _failed_private_source_push_rows(store: FakeStore, *, created_at: str = "2026-07-01T00:00:00+00:00") -> None:
    store.select_many_results["research_lab_candidate_promotion_events:promotion_failed"] = [
        {
            "promotion_event_id": "pe-failed",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "promotion_failed",
            "promotion_status": "failed",
            "event_doc": {"reason": "private_source_push_failed", "source_push_attempt": 1},
            "created_at": created_at,
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events"] = []
    store.select_one_results["research_lab_candidate_evaluation_current"] = {
        "candidate_id": "cand-1",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
        "island": "generalist",
        "current_score_bundle_id": "sb-1",
    }
    store.select_one_results["research_evaluation_score_bundle_current"] = {
        "score_bundle_id": "sb-1",
        "score_bundle_doc": {
            "candidate_artifact_hash": "sha256:" + "c" * 64,
            "parent_artifact_hash": "sha256:" + "a" * 64,
            "icp_set_hash": "sha256:" + "3" * 64,
            "evaluation_epoch": 23770,
            "score_bundle_hash": "sha256:" + "5" * 64,
            "aggregates": {},
        },
    }


def _pending_private_source_manifest_rows(
    store: FakeStore,
    *,
    created_at: str = "2026-07-01T00:00:00+00:00",
) -> None:
    _failed_private_source_push_rows(store, created_at=created_at)
    store.select_many_results["research_lab_candidate_promotion_events:promotion_failed"] = []
    store.select_many_results["research_lab_candidate_promotion_events:promotion_checked"] = [
        {
            "promotion_event_id": "pe-manifest-pending",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "promotion_checked",
            "promotion_status": "checked",
            "event_doc": {
                "reason": "source_pushed_manifest_pending",
                "candidate_status_preserved": "scored",
            },
            "created_at": created_at,
        }
    ]


async def test_private_source_push_reconciler_dry_run_plans_retry(store):
    _failed_private_source_push_rows(store)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=True,
        retry_after_seconds=0,
    )

    assert result["ok"] is True
    assert result["found_failed"] == 1
    assert result["attempted_recoveries"] == 1
    assert result["results"][0]["status"] == "would_retry_private_source_push"
    assert any(
        ("event_doc->>reason", "private_source_push_failed") in filters
        for table, filters in store.select_all_calls
        if table == "research_lab_candidate_promotion_events"
    )
    assert store.promotion_event_writes == []
    assert store.version_writes == []
    assert store.reward_obligation_writes == []


async def test_private_source_push_reconciler_candidate_filter_queries_candidate_directly(store):
    _failed_private_source_push_rows(store)

    await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        candidate_ids=["cand-1"],
        dry_run=True,
        retry_after_seconds=0,
    )

    assert (
        "research_lab_candidate_promotion_events",
        (
            ("event_type", "promotion_failed"),
            ("candidate_id", "cand-1"),
            ("event_doc->>reason", "private_source_push_failed"),
        ),
    ) in store.select_many_calls


async def test_private_source_push_reconciler_applies_retry_through_promotion_path(store, monkeypatch):
    _failed_private_source_push_rows(store)
    calls: list[dict[str, Any]] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "merged",
            "private_model_version_id": "pmv-1",
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        }

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["retried"] == 1
    assert result["finalized"] == 1
    assert result["results"][0]["status"] == "merged"
    assert calls[0]["candidate"]["candidate_id"] == "cand-1"
    assert calls[0]["score_bundle_row"]["score_bundle_id"] == "sb-1"


async def test_private_source_manifest_pending_reconciles_through_promotion_path(
    store,
    monkeypatch,
):
    _pending_private_source_manifest_rows(store)
    calls: list[dict[str, Any]] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "merged",
            "private_model_version_id": "pmv-1",
            "champion_reward_status": "created",
            "champion_reward_id": "cr-1",
        }

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "process_scored_candidate",
        _fake_process,
    )

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["retried"] == 1
    assert result["finalized"] == 1
    assert result["results"][0]["status"] == "merged"
    assert result["results"][0]["recovery_event_type"] == "promotion_checked"
    assert result["results"][0]["recovery_reason"] == "source_pushed_manifest_pending"
    assert calls[0]["candidate"]["candidate_id"] == "cand-1"
    assert calls[0]["score_bundle_row"]["score_bundle_id"] == "sb-1"


async def test_private_source_reconciler_ignores_unrelated_checked_events(
    store,
    monkeypatch,
):
    _pending_private_source_manifest_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events:promotion_checked"][0][
        "event_doc"
    ]["reason"] = "threshold_checked"

    async def _unexpected_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("unrelated promotion_checked event must not be replayed")

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "process_scored_candidate",
        _unexpected_process,
    )

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["found_failed"] == 0
    assert result["results"] == []
    assert not any(
        ("event_type", "champion_reward_created") in filters
        for table, filters in store.select_all_calls
        if table == "research_lab_candidate_promotion_events"
    )


async def test_private_source_manifest_reason_filter_prevents_event_window_starvation(
    store,
    monkeypatch,
):
    _pending_private_source_manifest_rows(store)
    pending_row = dict(
        store.select_many_results[
            "research_lab_candidate_promotion_events:promotion_checked"
        ][0]
    )
    unrelated_rows = [
        {
            **pending_row,
            "promotion_event_id": f"pe-unrelated-{index}",
            "candidate_id": f"cand-unrelated-{index}",
            "source_score_bundle_id": f"sb-unrelated-{index}",
            "event_doc": {"reason": "threshold_checked"},
        }
        for index in range(500)
    ]

    def checked_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = tuple(kwargs.get("filters") or ())
        if (
            "event_doc->>reason",
            "source_pushed_manifest_pending",
        ) in filters:
            return [pending_row]
        return unrelated_rows

    store.select_many_results[
        "research_lab_candidate_promotion_events:promotion_checked"
    ] = checked_rows
    calls: list[str] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["candidate"]["candidate_id"]))
        return {"status": "merged"}

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "process_scored_candidate",
        _fake_process,
    )

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["finalized"] == 1
    assert calls == ["cand-1"]


async def test_private_source_manifest_reconciler_json_filter_fallback(
    store,
    monkeypatch,
):
    _pending_private_source_manifest_rows(store)
    pending_rows = list(
        store.select_many_results[
            "research_lab_candidate_promotion_events:promotion_checked"
        ]
    )

    def checked_rows(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        if any(
            field == "event_doc->>reason"
            for field, _value in tuple(kwargs.get("filters") or ())
        ):
            raise RuntimeError("json path filter unavailable")
        return pending_rows

    store.select_many_results[
        "research_lab_candidate_promotion_events:promotion_checked"
    ] = checked_rows
    calls: list[str] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["candidate"]["candidate_id"]))
        return {"status": "merged"}

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "process_scored_candidate",
        _fake_process,
    )

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["finalized"] == 1
    assert calls == ["cand-1"]
    checked_queries = [
        filters
        for table, filters in (*store.select_many_calls, *store.select_all_calls)
        if table == "research_lab_candidate_promotion_events"
        and ("event_type", "promotion_checked") in filters
    ]
    assert any(("event_doc->>reason", "source_pushed_manifest_pending") in filters for filters in checked_queries)
    assert any(("event_doc->>reason", "source_pushed_manifest_pending") not in filters for filters in checked_queries)


async def test_private_source_manifest_reconcile_crosses_terminal_event_pages(
    store,
    monkeypatch,
):
    _pending_private_source_manifest_rows(store)
    target = dict(
        store.select_many_results[
            "research_lab_candidate_promotion_events:promotion_checked"
        ][0]
    )
    resolved = [
        {
            **target,
            "promotion_event_id": f"pe-resolved-{index}",
            "candidate_id": f"cand-resolved-{index}",
            "source_score_bundle_id": f"sb-resolved-{index}",
        }
        for index in range(1_001)
    ]
    store.select_many_results[
        "research_lab_candidate_promotion_events:promotion_checked"
    ] = [*resolved, target]
    store.select_many_results[
        "research_lab_candidate_promotion_events:champion_reward_created"
    ] = [
        {
            "candidate_id": row["candidate_id"],
            "source_score_bundle_id": row["source_score_bundle_id"],
            "event_type": "champion_reward_created",
        }
        for row in resolved
    ]

    candidate_history_calls: list[str] = []
    def candidate_events(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
        filters = tuple(kwargs.get("filters") or ())
        candidate_filter = next(
            (value for field, value in filters if field == "candidate_id"),
            "",
        )
        candidate_history_calls.append(str(candidate_filter))
        return []

    store.select_many_results[
        "research_lab_candidate_promotion_events"
    ] = candidate_events
    calls: list[str] = []

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["candidate"]["candidate_id"]))
        return {"status": "merged"}

    monkeypatch.setattr(
        ResearchLabPromotionController,
        "process_scored_candidate",
        _fake_process,
    )

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
        limit=1,
    )

    assert result["attempted_recoveries"] == 1
    assert result["finalized"] == 1
    assert calls == ["cand-1"]
    assert candidate_history_calls == ["cand-1"]
    assert len(result["results"]) == 1_002
    assert all(
        row["status"] == "already_rewarded"
        for row in result["results"][:-1]
    )
    assert result["results"][-1]["status"] == "merged"


async def test_private_source_reconcile_pages_use_unique_event_order(store):
    _pending_private_source_manifest_rows(store)

    await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=True,
        retry_after_seconds=0,
    )

    paginated_orders = [
        order_by
        for table, order_by in store.select_all_order_by_calls
        if table == "research_lab_candidate_promotion_events"
    ]
    assert paginated_orders
    assert all(
        order_by
        == (("created_at", True), ("promotion_event_id", True))
        for order_by in paginated_orders
    )


async def test_private_source_push_reconciler_marks_fresh_stale_parent_for_rebase(store, monkeypatch):
    _failed_private_source_push_rows(store)

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "stale_parent_needs_rescore"}

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["retried"] == 1
    entry = result["results"][0]
    assert entry["status"] == "stale_parent_needs_rescore"
    assert entry["stale_parent_rebase_eligible"] is True
    assert entry["stale_parent_recovery_event_status"] == "marked"
    assert store.candidate_evaluation_event_writes == [
        {
            "candidate_id": "cand-1",
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "event_type": "rejected",
            "candidate_status": "rejected",
            "evaluator_ref": "test-reconciler",
            "reason": "stale_parent_needs_rescore",
            "score_bundle_id": "sb-1",
            "event_doc": store.candidate_evaluation_event_writes[0]["event_doc"],
        }
    ]
    assert store.candidate_evaluation_event_writes[0]["event_doc"]["reason"] == (
        "private_source_push_failed_retry_stale_parent"
    )
    assert store.scoring_dispatch_event_writes[0]["dispatch_status"] == "rejected"
    assert store.scoring_dispatch_event_writes[0]["event_doc"]["dispatch_context"] == (
        "private_source_push_reconcile"
    )


async def test_private_source_push_reconciler_marks_existing_stale_parent_event_for_rebase(store, monkeypatch):
    _failed_private_source_push_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": "pe-stale",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "stale_parent_detected",
            "promotion_status": "rebase_required",
            "event_doc": {"reason": "stale_parent_needs_rescore"},
            "created_at": "2026-07-01T00:02:00+00:00",
        }
    ]

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("existing stale-parent promotion event should not be retried")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    entry = result["results"][0]
    assert entry["status"] == "stale_parent_needs_rescore"
    assert entry["latest_promotion_event_id"] == "pe-stale"
    assert entry["stale_parent_rebase_eligible"] is True
    assert store.candidate_evaluation_event_writes[0]["reason"] == "stale_parent_needs_rescore"
    assert store.candidate_evaluation_event_writes[0]["event_doc"]["latest_promotion_event_id"] == "pe-stale"


async def test_private_source_push_reconciler_respects_event_backoff(store, monkeypatch):
    _failed_private_source_push_rows(store, created_at=datetime.now(timezone.utc).isoformat())

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("retry should be held by backoff")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=300,
    )

    assert result["results"][0]["status"] == "retry_backoff"
    assert result["retried"] == 0


async def test_private_source_push_reconciler_skips_already_rewarded(store, monkeypatch):
    _failed_private_source_push_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": "pe-created",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "event_type": "champion_reward_created",
            "promotion_status": "reward_created",
            "event_doc": {},
            "created_at": "2026-07-01T00:01:00+00:00",
        }
    ]

    async def _fake_process(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("already rewarded candidates should not be retried")

    monkeypatch.setattr(ResearchLabPromotionController, "process_scored_candidate", _fake_process)

    result = await reconcile_failed_private_source_pushes(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
        retry_after_seconds=0,
    )

    assert result["results"][0]["status"] == "already_rewarded"
    assert result["retried"] == 0


# ---------------------------------------------------------------------------
# Bug #24 — champion reward reconciler
# ---------------------------------------------------------------------------


def _reward_config() -> Any:
    return SimpleNamespace(
        auto_promotion_enabled=True,
        auto_commit_enabled=False,
        improvement_threshold_points=1.0,
        private_model_manifest_uri="s3://bucket/bootstrap-manifest.json",
        reimbursement_policy_doc=lambda enabled: {"policy_id": "policy-1"},
        lab_reward_epochs=3,
        evaluation_epoch=7,
    )


def _pending_reward_rows(store: FakeStore) -> None:
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_pending_uid"] = [
        {
            "promotion_event_id": "pe-pending",
            "candidate_id": "cand-1",
            "source_score_bundle_id": "sb-1",
            "improvement_points": 2.5,
            "threshold_points": 1.0,
            "created_at": "2026-07-01T00:00:00+00:00",
        }
    ]
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_created"] = []
    store.select_one_results["research_lab_candidate_evaluation_current"] = {
        "candidate_id": "cand-1",
        "miner_hotkey": "hk-1",
        "ticket_id": "ticket-1",
        "run_id": "run-1",
        "island": "generalist",
        "current_score_bundle_id": "sb-1",
    }
    store.select_one_results["research_evaluation_score_bundle_current"] = {
        "score_bundle_id": "sb-1",
        "score_bundle_doc": {
            "evaluation_epoch": 7,
            "score_bundle_hash": "sha256:" + "4" * 64,
            "aggregates": {"per_icp_results": []},
            "private_holdout_gate": _approved_gate(
                candidate_delta_vs_daily_baseline=2.5,
            ),
        },
    }


async def test_reward_reconciler_happy_path_creates_reward(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    async def _live_epoch(_configured: Any = None) -> tuple[int, int | None, str]:
        return 150, 23_928, "official_subnet_epoch"

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    monkeypatch.setattr(
        promotion,
        "resolve_research_lab_evaluation_epoch",
        _live_epoch,
    )
    monkeypatch.setattr(
        promotion,
        "build_champion_reward_obligation",
        lambda obligation_input, policy: {
            "status": "active",
            "champion_reward_id": "cr-1",
            "candidate_id": obligation_input["candidate_id"],
            "score_bundle_id": obligation_input["score_bundle_id"],
            "run_id": obligation_input["run_id"],
            "miner_hotkey": obligation_input["miner_hotkey"],
            "uid": obligation_input["uid"],
            "island": obligation_input["island"],
            "evaluation_epoch": obligation_input["evaluation_epoch"],
            "start_epoch": obligation_input["start_epoch"],
            "epoch_count": 3,
            "improvement_points": obligation_input["improvement_points"],
            "threshold_points": obligation_input["threshold_points"],
            "desired_alpha_percent": 1.0,
            "input_hash": "sha256:" + "5" * 64,
            "anchored_hash": "sha256:" + "6" * 64,
        },
    )

    promotion_graph = {
        "root_receipt_hash": "sha256:" + "7" * 64,
        "receipts": [],
    }

    async def _load_promotion_graph(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "artifact_kind": "promotion_decision",
            "artifact_ref": "score_bundle:" + "4" * 64,
            "artifact_hash": "sha256:" + "4" * 64,
        }
        return promotion_graph

    async def _authorize_reward(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["decision_kind"] == "champion"
        assert (
            kwargs["decision_payload"]["promotion_decision"]["status"]
            == "promotion_passed"
        )
        assert kwargs["expected_result"]["reward"]["champion_reward_id"] == "cr-1"
        assert kwargs["parent_graphs"] == (promotion_graph,)
        return {"result": dict(kwargs["expected_result"])}

    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        _load_promotion_graph,
    )
    monkeypatch.setattr(
        v2_authority,
        "authorize_reward_decision_v2",
        _authorize_reward,
    )
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["ok"] is True
    assert result["found_pending"] == 1
    entry = result["results"][0]
    assert entry["status"] == "created"
    assert entry["champion_reward_id"] == "cr-1"
    assert entry["resolved_uid"] == 5
    assert len(store.reward_obligation_writes) == 1
    created_events = [
        e for e in store.promotion_event_writes if e["event_type"] == "champion_reward_created"
    ]
    assert len(created_events) == 1


async def test_reward_reconciler_uid_still_unresolved_retries_later_without_event_spam(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return None

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["results"][0]["status"] == "uid_still_unresolved"
    assert store.promotion_event_writes == []
    assert store.reward_obligation_writes == []


async def test_reward_reconciler_dry_run_plans_without_writes(store, monkeypatch):
    _pending_reward_rows(store)

    async def _resolve(hotkey: str) -> int | None:
        return 5

    monkeypatch.setattr(promotion, "_resolve_miner_uid", _resolve)
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=True,
    )
    assert result["results"][0]["status"] == "would_create_champion_reward"
    assert store.promotion_event_writes == []
    assert store.reward_obligation_writes == []


async def test_reward_reconciler_skips_already_created(store, monkeypatch):
    _pending_reward_rows(store)
    store.select_many_results["research_lab_candidate_promotion_events:champion_reward_created"] = [
        {"promotion_event_id": "pe-created", "event_type": "champion_reward_created"}
    ]
    result = await reconcile_pending_champion_rewards(
        _reward_config(),
        worker_ref="test-reconciler",
        dry_run=False,
    )
    assert result["results"][0]["status"] == "already_created"
    assert store.promotion_event_writes == []
