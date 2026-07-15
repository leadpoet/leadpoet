from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import shutil
import subprocess

import pytest

from gateway.research_lab.git_tree_models import (
    TreePolicy,
    derive_child_slot,
    derive_tree_id,
)
from gateway.research_lab.git_tree_repository import (
    GitTreeRepository,
    GitTreeRepositoryError,
)
from research_lab.code_editing import CodeEditDraft
from research_lab.eval.private_runtime import compute_private_source_tree_hash
from leadpoet_canonical.attested_v2 import sha256_json


ROOT_ARTIFACT_HASH = "sha256:" + "a" * 64


def _concurrent_initialize_and_reserve(
    workspace: str,
    source: str,
    tree_id: str,
    policy_hash: str,
    operation_id: str,
    request_hash: str,
) -> tuple[str, bool]:
    repository = GitTreeRepository(workspace=Path(workspace), tree_id=tree_id)
    root_commit = repository.initialize(
        source_root=Path(source),
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy_hash,
    )
    reservation = repository.reserve_operation(
        operation_id=operation_id,
        operation_kind="provider",
        request_hash=request_hash,
        reservation_doc={"logical_request": "shared"},
    )
    return root_commit, bool(reservation["created"])


def _concurrent_restore(
    workspace: str,
    tree_id: str,
    recovery: dict,
    bundle_path: str,
) -> str:
    repository = GitTreeRepository(workspace=Path(workspace), tree_id=tree_id)
    return repository.restore_recovery_state(
        recovery_state=recovery,
        bundle_path=Path(bundle_path),
    )


def _draft(*, before: str, after: str) -> CodeEditDraft:
    return CodeEditDraft(
        failure_mode="fixture",
        mechanism="change fixture value",
        expected_improvement="fixture improves",
        risk="low",
        lane="query_construction",
        target_files=("gateway/runtime.py",),
        unified_diff=(
            "diff --git a/gateway/runtime.py b/gateway/runtime.py\n"
            "--- a/gateway/runtime.py\n"
            "+++ b/gateway/runtime.py\n"
            "@@ -1 +1 @@\n"
            f"-{before}\n"
            f"+{after}\n"
        ),
        redacted_summary="change fixture value",
        test_plan="compile",
        rollback_plan="revert",
    )


def _rename_draft() -> CodeEditDraft:
    return CodeEditDraft(
        failure_mode="fixture",
        mechanism="rename fixture module",
        expected_improvement="fixture improves",
        risk="low",
        lane="query_construction",
        target_files=("gateway/runtime.py", "gateway/renamed.py"),
        unified_diff=(
            "diff --git a/gateway/runtime.py b/gateway/renamed.py\n"
            "similarity index 100%\n"
            "rename from gateway/runtime.py\n"
            "rename to gateway/renamed.py\n"
        ),
        redacted_summary="rename fixture module",
        test_plan="compile",
        rollback_plan="revert",
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "gateway").mkdir(parents=True)
    (source / "gateway/runtime.py").write_text("VALUE = 0\n", encoding="utf-8")
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n", encoding="utf-8"
    )
    return source


def test_git_tree_commits_direct_parents_and_keeps_siblings_isolated(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-git", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    root_commit = repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    assert len(root_commit) == 64
    root_source_hash = compute_private_source_tree_hash(source)

    first_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    first = repository.commit_child(
        slot=first_slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 1"),
        expected_parent_source_tree_hash=root_source_hash,
    )
    repository.verify_node(commit=first)

    child_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id=first.node_id,
        root_branch_id=first.root_branch_id,
        depth=2,
        slot_index=0,
    )
    child = repository.commit_child(
        slot=child_slot,
        draft=_draft(before="VALUE = 1", after="VALUE = 2"),
        expected_parent_source_tree_hash=first.source_tree_hash,
    )
    repository.verify_node(commit=child)
    assert child.parent_git_commit == first.git_commit

    sibling_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=1,
    )
    sibling = repository.commit_child(
        slot=sibling_slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 3"),
        expected_parent_source_tree_hash=root_source_hash,
    )
    assert sibling.parent_git_commit == root_commit

    materialized_child = tmp_path / "materialized-child"
    repository.materialize_node(node_id=child.node_id, destination=materialized_child)
    try:
        assert (materialized_child / "gateway/runtime.py").read_text() == "VALUE = 2\n"
    finally:
        repository.release_materialized_node(materialized_child)
    materialized_sibling = tmp_path / "materialized-sibling"
    repository.materialize_node(node_id=sibling.node_id, destination=materialized_sibling)
    try:
        assert (materialized_sibling / "gateway/runtime.py").read_text() == "VALUE = 3\n"
    finally:
        repository.release_materialized_node(materialized_sibling)


def test_cumulative_patch_reconstructs_child_from_root(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-cumulative", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    root_hash = compute_private_source_tree_hash(source)
    parent_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    parent = repository.commit_child(
        slot=parent_slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 1"),
        expected_parent_source_tree_hash=root_hash,
    )
    child_slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id=parent.node_id,
        root_branch_id=parent.root_branch_id,
        depth=2,
        slot_index=0,
    )
    child = repository.commit_child(
        slot=child_slot,
        draft=_draft(before="VALUE = 1", after="VALUE = 2"),
        expected_parent_source_tree_hash=parent.source_tree_hash,
    )

    reconstructed = tmp_path / "reconstructed"
    subprocess.run(["cp", "-R", str(source), str(reconstructed)], check=True)
    subprocess.run(["git", "init", "--quiet", str(reconstructed)], check=True)
    patch_path = tmp_path / "cumulative.diff"
    patch_path.write_text(child.cumulative_patch, encoding="utf-8")
    subprocess.run(
        ["git", "apply", "--check", "--recount", str(patch_path)],
        cwd=reconstructed,
        check=True,
    )
    subprocess.run(
        ["git", "apply", "--recount", str(patch_path)],
        cwd=reconstructed,
        check=True,
    )
    assert compute_private_source_tree_hash(reconstructed) == child.source_tree_hash


def test_tree_repository_is_idempotent_but_rejects_root_or_slot_reuse(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-idempotent", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    first_root = repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    assert repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    ) == first_root
    with pytest.raises(GitTreeRepositoryError, match="root_artifact_hash"):
        repository.initialize(
            source_root=source,
            root_artifact_hash="sha256:" + "f" * 64,
            policy_hash=policy.policy_hash,
        )

    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    first_child = repository.commit_child(
        slot=slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 1"),
        expected_parent_source_tree_hash=compute_private_source_tree_hash(source),
    )
    reconciled = repository.commit_child(
        slot=slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 1"),
        expected_parent_source_tree_hash=compute_private_source_tree_hash(source),
    )
    assert reconciled == first_child
    repository.verify_node_identity(
        node_id=slot.node_id,
        git_commit=first_child.git_commit,
        parent_node_id="root",
    )
    with pytest.raises(GitTreeRepositoryError, match="differs from planned draft"):
        repository.commit_child(
            slot=slot,
            draft=_draft(before="VALUE = 0", after="VALUE = 2"),
            expected_parent_source_tree_hash=compute_private_source_tree_hash(source),
        )


def test_tree_repository_reopen_binds_all_root_authority(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-root-authority",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    bindings = {
        "run_id": "run-root-authority",
        "root_manifest_hash": "sha256:" + "b" * 64,
        "root_image_digest": "sha256:" + "c" * 64,
        "evaluator_commitment_hash": "sha256:" + "d" * 64,
        "tree_doc": {"schema_version": "tree-test.v1", "epoch_id": 1},
    }
    root_commit = repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
        **bindings,
    )
    assert repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
        **bindings,
    ) == root_commit

    mutations = (
        ("run_id", "different-run", "run_id"),
        ("root_manifest_hash", "sha256:" + "e" * 64, "root_manifest_hash"),
        ("root_image_digest", "sha256:" + "f" * 64, "root_image_digest"),
        (
            "evaluator_commitment_hash",
            "sha256:" + "1" * 64,
            "evaluator_commitment_hash",
        ),
        ("tree_doc", {"schema_version": "tree-test.v1", "epoch_id": 2}, "tree_doc_hash"),
    )
    for argument, value, expected_field in mutations:
        changed = {**bindings, argument: value}
        with pytest.raises(GitTreeRepositoryError, match=expected_field):
            repository.initialize(
                source_root=source,
                root_artifact_hash=ROOT_ARTIFACT_HASH,
                policy_hash=policy.policy_hash,
                **changed,
            )


def test_tree_repository_changed_files_come_from_committed_diff(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-rename", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )

    child = repository.commit_child(
        slot=slot,
        draft=_rename_draft(),
        expected_parent_source_tree_hash=compute_private_source_tree_hash(source),
    )

    assert child.changed_files == ("gateway/renamed.py",)


def test_ten_processes_create_one_tree_and_one_operation_reservation(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-concurrent",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    workspace = tmp_path / "tree"
    operation_id = "sha256:" + "4" * 64
    request_hash = "sha256:" + "5" * 64
    args = (
        str(workspace),
        str(source),
        tree_id,
        policy.policy_hash,
        operation_id,
        request_hash,
    )

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_concurrent_initialize_and_reserve, *zip(*([args] * 10))))

    root_commits = {root_commit for root_commit, _ in results}
    assert len(root_commits) == 1
    assert sum(created for _, created in results) == 1
    repository = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    assert repository.state_status() == "complete"
    assert repository.inspect_operation(operation_id=operation_id)["operation"] == {
        "tree_id": tree_id,
        "node_id": "",
        "operation_kind": "provider",
        "request_hash": request_hash,
        "operation_id": operation_id,
        "reservation_doc": {"logical_request": "shared"},
        "status": "reserved",
    }


def test_git_bundle_is_content_addressed_and_verifiable(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-bundle", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    bundle = repository.create_bundle(tmp_path / "tree.bundle")
    assert bundle["bundle_hash"].startswith("sha256:")
    assert bundle["bundle_size_bytes"] > 0
    subprocess.run(
        ["git", "bundle", "verify", bundle["bundle_path"]],
        cwd=repository.repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def test_operation_inspection_is_read_only_and_returns_terminal_state(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-operation", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    operation_id = "sha256:" + "4" * 64
    request_hash = "sha256:" + "5" * 64
    repository.plan_slot(
        slot=slot,
        request_hash=request_hash,
        operation_id=operation_id,
        node_doc={"node_id": slot.node_id},
    )
    reserved = repository.inspect_operation(operation_id=operation_id)
    assert reserved["operation"]["status"] == "reserved"
    with pytest.raises(
        GitTreeRepositoryError, match="reserved operation"
    ):
        repository.operation_settlement_commitment()
    repository.settle_operation(
        operation_id=operation_id,
        operation_status="indeterminate",
        request_hash=request_hash,
        result_hash="sha256:" + "6" * 64,
        settled_cost_microusd=0,
        provider_call_count=0,
        settlement_doc={"operation_kind": "generation"},
    )
    terminal = repository.inspect_operation(operation_id=operation_id)
    assert terminal["operation"]["status"] == "indeterminate"
    assert terminal == repository.inspect_operation(operation_id=operation_id)
    commitment = repository.operation_settlement_commitment()
    assert commitment["operation_count"] == 1
    assert commitment["settled_cost_microusd"] == 0
    assert commitment["provider_call_count"] == 0
    assert commitment["operation_settlement_hash"].startswith("sha256:")


def test_no_finalist_failure_is_idempotent_and_blocks_later_selection(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-failed", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    failure = {
        "schema_version": "research_lab.git_tree_failed.v1",
        "tree_id": tree_id,
        "stop_reason": "no_eligible_tree_finalist",
        "selected_node_id": "",
        "paid_finalist_count": 0,
    }
    from research_lab.canonical import sha256_json

    failure_hash = sha256_json(failure)
    assert repository.fail_tree(
        failure_hash=failure_hash, failure_doc=failure
    )["created"] is True
    assert repository.fail_tree(
        failure_hash=failure_hash, failure_doc=failure
    )["created"] is False
    selection = {"selected_node_id": "tree-node:" + "1" * 64}
    with pytest.raises(GitTreeRepositoryError, match="failed tree"):
        repository.select_final(
            selection_hash=sha256_json(selection), selection_doc=selection
        )


def test_recovery_bundle_restores_git_control_and_terminal_operations(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-recovery", root_artifact_hash=ROOT_ARTIFACT_HASH, policy=policy
    )
    workspace = tmp_path / "tree"
    repository = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    root_commit = repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    child = repository.commit_child(
        slot=slot,
        draft=_draft(before="VALUE = 0", after="VALUE = 7"),
        expected_parent_source_tree_hash=compute_private_source_tree_hash(source),
    )
    operation_id = "sha256:" + "4" * 64
    request_hash = "sha256:" + "5" * 64
    repository.plan_slot(
        slot=slot,
        request_hash=request_hash,
        operation_id=operation_id,
        node_doc={"node_id": slot.node_id},
    )
    repository.settle_operation(
        operation_id=operation_id,
        operation_status="succeeded",
        request_hash=request_hash,
        result_hash="sha256:" + "6" * 64,
        settled_cost_microusd=123,
        provider_call_count=1,
        settlement_doc={"operation_kind": "generation", "node_id": slot.node_id},
    )
    checkpoint_doc = {"tree_id": tree_id, "node_ids": [slot.node_id]}
    checkpoint_hash = sha256_json(checkpoint_doc)
    repository.commit_checkpoint(
        checkpoint_hash=checkpoint_hash,
        checkpoint_doc=checkpoint_doc,
    )
    bundle = repository.create_bundle(tmp_path / "tree.bundle")
    recovery = repository.export_recovery_state(
        checkpoint_hash=checkpoint_hash,
        bundle_uri="s3://private/tree.bundle",
        bundle_hash=bundle["bundle_hash"],
        bundle_size_bytes=bundle["bundle_size_bytes"],
    )

    shutil.rmtree(workspace)
    restored = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    assert restored.state_status() == "missing"
    assert restored.restore_recovery_state(
        recovery_state=recovery,
        bundle_path=Path(bundle["bundle_path"]),
    ) == root_commit
    assert restored.state_status() == "complete"
    assert restored.inspect_operation(operation_id=operation_id)["operation"][
        "status"
    ] == "succeeded"
    restored.verify_node_identity(
        node_id=slot.node_id,
        git_commit=child.git_commit,
        parent_node_id="root",
    )


def test_ten_processes_restore_one_identical_tree(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-concurrent-restore",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    source_repository = GitTreeRepository(
        workspace=tmp_path / "source-tree",
        tree_id=tree_id,
    )
    root_commit = source_repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    checkpoint_doc = {"tree_id": tree_id, "node_ids": []}
    checkpoint_hash = sha256_json(checkpoint_doc)
    source_repository.commit_checkpoint(
        checkpoint_hash=checkpoint_hash,
        checkpoint_doc=checkpoint_doc,
    )
    bundle = source_repository.create_bundle(tmp_path / "tree.bundle")
    recovery = source_repository.export_recovery_state(
        checkpoint_hash=checkpoint_hash,
        bundle_uri="s3://private/tree.bundle",
        bundle_hash=bundle["bundle_hash"],
        bundle_size_bytes=bundle["bundle_size_bytes"],
    )
    workspace = tmp_path / "restored"
    args = (str(workspace), tree_id, recovery, str(bundle["bundle_path"]))

    with ProcessPoolExecutor(max_workers=10) as executor:
        roots = list(executor.map(_concurrent_restore, *zip(*([args] * 10))))

    assert roots == [root_commit] * 10
    restored = GitTreeRepository(workspace=workspace, tree_id=tree_id)
    assert restored.state_status() == "complete"


def test_recovery_rejects_tampered_bundle_and_partial_workspace(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-recovery-tamper",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    source_workspace = tmp_path / "source-tree"
    repository = GitTreeRepository(workspace=source_workspace, tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    checkpoint_doc = {"tree_id": tree_id, "node_ids": []}
    checkpoint_hash = sha256_json(checkpoint_doc)
    repository.commit_checkpoint(
        checkpoint_hash=checkpoint_hash,
        checkpoint_doc=checkpoint_doc,
    )
    bundle = repository.create_bundle(tmp_path / "tree.bundle")
    recovery = repository.export_recovery_state(
        checkpoint_hash=checkpoint_hash,
        bundle_uri="s3://private/tree.bundle",
        bundle_hash=bundle["bundle_hash"],
        bundle_size_bytes=bundle["bundle_size_bytes"],
    )
    tampered = tmp_path / "tampered.bundle"
    tampered.write_bytes(Path(bundle["bundle_path"]).read_bytes() + b"tamper")
    restored = GitTreeRepository(workspace=tmp_path / "restored", tree_id=tree_id)
    with pytest.raises(GitTreeRepositoryError, match="bundle commitment"):
        restored.restore_recovery_state(
            recovery_state=recovery,
            bundle_path=tampered,
        )

    partial = GitTreeRepository(workspace=tmp_path / "partial", tree_id=tree_id)
    partial.workspace.mkdir(parents=True)
    partial.metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(GitTreeRepositoryError, match="workspace is partial"):
        partial.state_status()


def test_reconcile_operations_only_fills_missing_authoritative_state(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-operation-reconcile",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    operation = {
        "logical_operation_id": "sha256:" + "1" * 64,
        "tree_id": tree_id,
        "node_id": "",
        "operation_kind": "provider",
        "operation_status": "reserved",
        "request_hash": "sha256:" + "2" * 64,
        "result_hash": "",
        "settled_cost_microusd": 0,
        "provider_call_count": 0,
        "settlement_doc": {"provider": "approved"},
    }
    assert repository.reconcile_operations([operation]) == 1
    assert repository.reconcile_operations([operation]) == 0
    assert repository.inspect_operation(
        operation_id=operation["logical_operation_id"]
    )["operation"]["reservation_doc"] == {"provider": "approved"}
    with pytest.raises(GitTreeRepositoryError, match="differs from persistence"):
        repository.reconcile_operations(
            [{**operation, "request_hash": "sha256:" + "3" * 64}]
        )


def test_reconcile_operations_advances_reserved_state_after_database_commit(
    tmp_path,
):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-operation-advance",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    operation_id = "sha256:" + "1" * 64
    request_hash = "sha256:" + "2" * 64
    repository.reserve_operation(
        operation_id=operation_id,
        operation_kind="provider",
        request_hash=request_hash,
        reservation_doc={"logical_request": "fixture"},
    )
    persisted = {
        "logical_operation_id": operation_id,
        "tree_id": tree_id,
        "node_id": "",
        "operation_kind": "provider",
        "operation_status": "succeeded",
        "request_hash": request_hash,
        "result_hash": "sha256:" + "3" * 64,
        "settled_cost_microusd": 71,
        "provider_call_count": 1,
        "settlement_doc": {"provider": "approved"},
    }

    assert repository.reconcile_operations([persisted]) == 1
    terminal = repository.inspect_operation(operation_id=operation_id)["operation"]
    assert terminal["status"] == "succeeded"
    assert terminal["result_hash"] == persisted["result_hash"]
    assert terminal["settled_cost_microusd"] == 71
    assert terminal["reservation_doc"] == {"logical_request": "fixture"}
    assert repository.reconcile_operations([persisted]) == 0


def test_reconcile_operations_rejects_conflicting_terminal_evidence(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-operation-reconcile-conflict",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    operation_id = "sha256:" + "1" * 64
    request_hash = "sha256:" + "2" * 64
    repository.reserve_operation(
        operation_id=operation_id,
        operation_kind="provider",
        request_hash=request_hash,
    )
    repository.settle_operation(
        operation_id=operation_id,
        operation_status="succeeded",
        request_hash=request_hash,
        result_hash="sha256:" + "3" * 64,
        settled_cost_microusd=71,
        provider_call_count=1,
        settlement_doc={"provider": "approved"},
    )
    with pytest.raises(GitTreeRepositoryError, match="differs from persistence"):
        repository.reconcile_operations(
            [
                {
                    "logical_operation_id": operation_id,
                    "tree_id": tree_id,
                    "node_id": "",
                    "operation_kind": "provider",
                    "operation_status": "succeeded",
                    "request_hash": request_hash,
                    "result_hash": "sha256:" + "4" * 64,
                    "settled_cost_microusd": 71,
                    "provider_call_count": 1,
                    "settlement_doc": {"provider": "approved"},
                }
            ]
        )


def test_duplicate_terminal_settlement_requires_identical_evidence(tmp_path):
    source = _source(tmp_path)
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-operation-idempotency",
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy=policy,
    )
    repository = GitTreeRepository(workspace=tmp_path / "tree", tree_id=tree_id)
    repository.initialize(
        source_root=source,
        root_artifact_hash=ROOT_ARTIFACT_HASH,
        policy_hash=policy.policy_hash,
    )
    operation_id = "sha256:" + "1" * 64
    request_hash = "sha256:" + "2" * 64
    settlement = {
        "operation_id": operation_id,
        "operation_status": "succeeded",
        "request_hash": request_hash,
        "result_hash": "sha256:" + "3" * 64,
        "settled_cost_microusd": 71,
        "provider_call_count": 1,
        "settlement_doc": {"provider": "approved"},
    }
    repository.reserve_operation(
        operation_id=operation_id,
        operation_kind="provider",
        request_hash=request_hash,
    )

    assert repository.settle_operation(**settlement)["created"] is True
    assert repository.settle_operation(**settlement)["created"] is False
    with pytest.raises(
        GitTreeRepositoryError, match="terminal evidence differs"
    ):
        repository.settle_operation(
            **{**settlement, "settled_cost_microusd": 72}
        )
