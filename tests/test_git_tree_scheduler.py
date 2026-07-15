from __future__ import annotations

from dataclasses import replace

import pytest

from gateway.research_lab.git_tree_models import (
    TreeEvaluation,
    TreeNode,
    TreePolicy,
    derive_child_slot,
    derive_tree_id,
)
from gateway.research_lab.git_tree_scheduler import (
    GitTreeScheduler,
    GitTreeSchedulerError,
    sanitized_branch_context,
)


ROOT_HASH = "sha256:" + "a" * 64
SNAPSHOT_HASH = "sha256:" + "b" * 64
DEV_SET_HASH = "sha256:" + "c" * 64
RECEIPT_ROOT = "sha256:" + "d" * 64
CONTEXT_HASH = "sha256:" + "e" * 64
OBJECTIVE_HASH = "sha256:" + "f" * 64
GIT_COMMIT = "1" * 64
SOURCE_TREE_HASH = "sha256:" + "2" * 64
INCREMENTAL_PATCH_HASH = "sha256:" + "3" * 64
CUMULATIVE_PATCH_HASH = "sha256:" + "4" * 64
CANDIDATE_ARTIFACT_HASH = "sha256:" + "5" * 64
LINEAGE_HASH = "sha256:" + "6" * 64


def _evaluation(score: float, *, delta: float | None = None) -> TreeEvaluation:
    return TreeEvaluation(
        score=score,
        eligible=True,
        reason="eligible",
        execution_coverage=1.0,
        snapshot_miss_count=0,
        true_miss_count=0,
        failure_count=0,
        zero_output_count=0,
        snapshot_hash=SNAPSHOT_HASH,
        dev_set_hash=DEV_SET_HASH,
        policy="strict",
        score_version="dev-v1",
        receipt_root=RECEIPT_ROOT,
        context_hash=CONTEXT_HASH,
        parent_delta=delta,
        feedback={"examples": [{"number": 1, "band": "weak"}]},
    )


def _complete(slot, score: float) -> TreeNode:
    return TreeNode(
        **slot.to_dict(),
        status="eligible",
        branch_objective_path_id=f"path-{slot.root_branch_id[-8:]}",
        branch_objective_hash=OBJECTIVE_HASH,
        generation_attempt_count=1,
        git_commit=GIT_COMMIT,
        source_tree_hash=SOURCE_TREE_HASH,
        incremental_patch_hash=INCREMENTAL_PATCH_HASH,
        cumulative_patch_hash=CUMULATIVE_PATCH_HASH,
        candidate_artifact_hash=CANDIDATE_ARTIFACT_HASH,
        lineage_hash=LINEAGE_HASH,
        evaluation=_evaluation(score),
    )


def _scheduler() -> GitTreeScheduler:
    policy = TreePolicy(mode="active")
    return GitTreeScheduler(
        tree_id=derive_tree_id(
            run_id="run-tree", root_artifact_hash=ROOT_HASH, policy=policy
        ),
        policy=policy,
    )


def test_scheduler_predeclares_one_durable_round_and_resumes_it():
    scheduler = _scheduler()
    slot = scheduler.plan_next()
    assert slot is not None
    assert len(scheduler.planned_slots) == 2
    assert scheduler.plan_next() == slot
    restored = GitTreeScheduler.restore(
        tree_id=scheduler.tree_id,
        policy=scheduler.policy,
        nodes=(),
        planned_slots=scheduler.planned_slots,
    )
    assert restored.plan_next() == slot
    assert restored.planned_slots == scheduler.planned_slots


def test_scheduler_preserves_two_root_branches_then_expands_each():
    scheduler = _scheduler()
    first_slot = scheduler.plan_next()
    assert first_slot is not None
    first = _complete(first_slot, 90.0)
    scheduler.record_node(first)
    second_slot = scheduler.plan_next()
    assert second_slot is not None
    second = _complete(second_slot, 10.0)
    scheduler.record_node(second)

    child_a_slot = scheduler.plan_next()
    assert child_a_slot is not None
    assert len(scheduler.planned_slots) == 4
    assert {slot.parent_node_id for slot in scheduler.planned_slots} == {
        first.node_id,
        second.node_id,
    }


def test_scheduler_retains_eligible_ancestor_as_finalist():
    scheduler = _scheduler()
    root_slot = scheduler.plan_next()
    assert root_slot is not None
    parent = _complete(root_slot, 80.0)
    scheduler.record_node(parent)
    sibling_slot = scheduler.plan_next()
    assert sibling_slot is not None
    scheduler.record_node(_complete(sibling_slot, 70.0))
    child_slot = scheduler.plan_next()
    assert child_slot is not None
    scheduler.record_node(_complete(child_slot, 60.0))
    assert scheduler.select_finalist() == parent


def test_scheduler_rejects_orphan_and_topology_mutation():
    scheduler = _scheduler()
    slot = scheduler.plan_next()
    assert slot is not None
    node = _complete(slot, 80.0)
    scheduler.record_node(node)
    with pytest.raises(GitTreeSchedulerError, match="topology changed"):
        scheduler.replace_node(replace(node, depth=2))


def test_branch_context_contains_parent_feedback_but_no_sibling_feedback():
    scheduler = _scheduler()
    slot = scheduler.plan_next()
    assert slot is not None
    parent = _complete(slot, 80.0)
    scheduler.record_node(parent)
    sibling_slot = scheduler.plan_next()
    assert sibling_slot is not None
    sibling = _complete(sibling_slot, 95.0)
    scheduler.record_node(sibling)
    child = scheduler.plan_next()
    assert child is not None
    selected_parent = scheduler.parent(child)
    context = sanitized_branch_context(
        slot=child,
        parent=selected_parent,
        ancestors=scheduler.nodes,
    )
    assert context["parent_node_id"] == selected_parent.node_id
    assert context["parent_feedback"] == selected_parent.evaluation.feedback
    other_root = parent if selected_parent.node_id == sibling.node_id else sibling
    assert other_root.node_id not in context["ancestor_node_ids"]
    assert "sibling_hypothesis_hashes" not in context
    assert "score" not in context["parent_feedback"]


def test_branch_context_contains_only_direct_ancestry_within_one_root_branch():
    scheduler = _scheduler()
    root_slot = scheduler.plan_next()
    assert root_slot is not None
    root = _complete(root_slot, 80.0)
    first_child_slot = derive_child_slot(
        tree_id=scheduler.tree_id,
        parent_node_id=root.node_id,
        root_branch_id=root.root_branch_id,
        depth=2,
        slot_index=0,
    )
    sibling_child_slot = derive_child_slot(
        tree_id=scheduler.tree_id,
        parent_node_id=root.node_id,
        root_branch_id=root.root_branch_id,
        depth=2,
        slot_index=1,
    )
    first_child = _complete(first_child_slot, 85.0)
    sibling_child = _complete(sibling_child_slot, 95.0)
    grandchild_slot = derive_child_slot(
        tree_id=scheduler.tree_id,
        parent_node_id=first_child.node_id,
        root_branch_id=root.root_branch_id,
        depth=3,
        slot_index=0,
    )

    context = sanitized_branch_context(
        slot=grandchild_slot,
        parent=first_child,
        ancestors=(root, first_child, sibling_child),
    )

    assert context["ancestor_node_ids"] == [root.node_id, first_child.node_id]
    assert sibling_child.node_id not in context["ancestor_node_ids"]
    assert context["parent_feedback"] == first_child.evaluation.feedback
