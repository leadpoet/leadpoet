from __future__ import annotations

from dataclasses import replace

import pytest

from gateway.research_lab.config import DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG
from gateway.research_lab.git_tree_models import (
    GitTreeContractError,
    TreeCheckpoint,
    TreeEvaluation,
    TreeNode,
    TreePolicy,
    TreeResult,
    derive_child_slot,
    derive_frontier_commitment_hash,
    derive_tree_id,
    next_child_slot,
    select_finalist,
    summarize_tree_evaluations,
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


def _evaluation(score: float, *, parent_delta: float = 0.0) -> TreeEvaluation:
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
        parent_delta=parent_delta,
        feedback={"summary": "bounded"},
    )


def _node(slot, score: float) -> TreeNode:
    return TreeNode(
        tree_id=slot.tree_id,
        node_id=slot.node_id,
        parent_node_id=slot.parent_node_id,
        root_branch_id=slot.root_branch_id,
        depth=slot.depth,
        slot_index=slot.slot_index,
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


def test_tree_policy_defaults_and_invalid_cross_field_config():
    policy = TreePolicy(mode="active")
    assert policy.branch_factor == 2
    assert policy.beam_width == 2
    assert policy.max_depth == 2
    assert policy.max_nodes == 6
    assert policy.live_max_icps_per_node == 5
    for field_name, expected in (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.to_policy_kwargs().items()
    ):
        if field_name == "mode":
            continue
        assert getattr(policy, field_name) == expected
    assert policy.effective_billable_cap(3_000_000) == 3_000_000
    assert policy.effective_billable_cap(0) == 0
    assert policy.required_final_context_seconds(300) == 420
    with pytest.raises(GitTreeContractError, match="diversity_floor"):
        TreePolicy(mode="active", beam_width=1, diversity_floor=2)
    with pytest.raises(GitTreeContractError, match="shortlist_size"):
        TreePolicy(mode="active", shortlist_size=1, diversity_floor=2)
    with pytest.raises(GitTreeContractError, match="reserve"):
        TreePolicy(
            mode="active",
            deadline_seconds=300,
            finalization_reserve_seconds=300,
        )
    with pytest.raises(GitTreeContractError, match="live_max_icps_per_node"):
        TreePolicy(mode="active", live_max_icps_per_node=9)
    with pytest.raises(GitTreeContractError, match="build_concurrency"):
        TreePolicy(mode="active", build_concurrency=2)
    with pytest.raises(GitTreeContractError, match="final evaluation timeout"):
        TreePolicy(
            mode="active",
            deadline_seconds=300,
            finalization_reserve_seconds=30,
        ).required_final_context_seconds(270)


def test_tree_policy_rejects_deprecated_or_invalid_values_instead_of_coercing():
    with pytest.raises(
        GitTreeContractError,
        match="deprecated flat/sequential autoresearch environment",
    ):
        TreePolicy.from_env(
            {
                "RESEARCH_LAB_TREE_MODE": "active",
                "RESEARCH_LAB_INNER_LOOP_MODE": "rank",
            }
        )
    with pytest.raises(GitTreeContractError, match="tree mode"):
        TreePolicy.from_env({"RESEARCH_LAB_TREE_MODE": "shadow"})
    with pytest.raises(GitTreeContractError, match="must be an integer"):
        TreePolicy.from_env({"RESEARCH_LAB_TREE_BRANCH_FACTOR": "two"})
    configured = TreePolicy.from_env(
        {
            "RESEARCH_LAB_TREE_MODE": "active",
            "RESEARCH_LAB_TREE_LIVE_MAX_ICPS_PER_NODE": "7",
        }
    )
    assert configured.live_max_icps_per_node == 7


def test_tree_identity_and_child_slots_are_deterministic_and_parent_bound():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-1", root_artifact_hash=ROOT_HASH, policy=policy
    )
    first = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    assert first == derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    sibling = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=1,
    )
    child = derive_child_slot(
        tree_id=tree_id,
        parent_node_id=first.node_id,
        root_branch_id=first.root_branch_id,
        depth=2,
        slot_index=0,
    )
    assert len({first.node_id, sibling.node_id, child.node_id}) == 3
    assert child.parent_node_id == first.node_id
    assert child.root_branch_id == first.node_id


def test_durable_frontier_identity_distinguishes_same_frontier_checkpoints():
    tree_id = derive_tree_id(
        run_id="run-frontier",
        root_artifact_hash=ROOT_HASH,
        policy=TreePolicy(mode="active"),
    )
    scheduler_frontier = "sha256:" + "1" * 64
    first = derive_frontier_commitment_hash(
        tree_id=tree_id,
        scheduler_frontier_hash=scheduler_frontier,
        checkpoint_hash="sha256:" + "2" * 64,
    )
    second = derive_frontier_commitment_hash(
        tree_id=tree_id,
        scheduler_frontier_hash=scheduler_frontier,
        checkpoint_hash="sha256:" + "3" * 64,
    )
    assert first != second


def test_default_frontier_builds_two_roots_then_preserves_branch_diversity():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-1", root_artifact_hash=ROOT_HASH, policy=policy
    )
    root_1_slot = next_child_slot(tree_id=tree_id, policy=policy, nodes=[])
    assert root_1_slot is not None
    root_1 = _node(root_1_slot, 90.0)
    root_2_slot = next_child_slot(tree_id=tree_id, policy=policy, nodes=[root_1])
    assert root_2_slot is not None
    root_2 = _node(root_2_slot, 20.0)

    child_1_slot = next_child_slot(
        tree_id=tree_id, policy=policy, nodes=[root_1, root_2]
    )
    assert child_1_slot is not None
    assert child_1_slot.parent_node_id == root_1.node_id
    child_1 = _node(child_1_slot, 95.0)

    child_2_slot = next_child_slot(
        tree_id=tree_id,
        policy=policy,
        nodes=[root_1, root_2, child_1],
    )
    assert child_2_slot is not None
    assert child_2_slot.parent_node_id == root_2.node_id


def test_frontier_stops_at_node_cap_and_never_reuses_a_planned_slot():
    policy = TreePolicy(mode="active", max_nodes=2, shortlist_size=2)
    tree_id = derive_tree_id(
        run_id="run-cap", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slot = next_child_slot(tree_id=tree_id, policy=policy, nodes=[])
    assert slot is not None
    assert (
        next_child_slot(
            tree_id=tree_id,
            policy=policy,
            nodes=[],
            planned_node_ids=[slot.node_id],
        )
        is None
    )
    first = _node(slot, 1.0)
    second_slot = next_child_slot(tree_id=tree_id, policy=policy, nodes=[first])
    assert second_slot is not None
    second = _node(second_slot, 2.0)
    assert next_child_slot(tree_id=tree_id, policy=policy, nodes=[first, second]) is None


def test_final_selection_is_score_first_and_deterministic_for_ties():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-rank", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slots = [
        derive_child_slot(
            tree_id=tree_id,
            parent_node_id="root",
            root_branch_id="",
            depth=1,
            slot_index=index,
        )
        for index in range(2)
    ]
    low = _node(slots[0], 10.0)
    high = _node(slots[1], 11.0)
    assert select_finalist([low, high]) == high

    tied_high_cost = replace(
        low,
        settled_cost_microusd=100,
        evaluation=_evaluation(11.0),
    )
    assert select_finalist([tied_high_cost, high]) == high


def test_final_selection_rejects_mixed_evaluation_contexts():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-context", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slots = [
        derive_child_slot(
            tree_id=tree_id,
            parent_node_id="root",
            root_branch_id="",
            depth=1,
            slot_index=index,
        )
        for index in range(2)
    ]
    first = _node(slots[0], 10.0)
    second = replace(
        _node(slots[1], 11.0),
        evaluation=replace(_evaluation(11.0), context_hash="sha256:" + "f" * 64),
    )
    with pytest.raises(GitTreeContractError, match="evaluation context"):
        select_finalist([first, second])


def test_tree_contracts_round_trip_strictly():
    policy = TreePolicy(mode="active")
    assert TreePolicy.from_mapping(policy.to_dict()) == policy
    evaluation = _evaluation(12.0, parent_delta=2.0)
    assert TreeEvaluation.from_mapping(evaluation.to_dict()) == evaluation


def test_tree_result_round_trip_binds_one_finalist_to_the_checkpoint():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-result", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    finalist = _node(slot, 12.0)
    checkpoint = TreeCheckpoint(
        tree_id=tree_id,
        root_artifact_hash=ROOT_HASH,
        policy=policy,
        nodes=(finalist,),
        frontier_hash="sha256:" + "7" * 64,
        operation_settlement_hash="sha256:" + "8" * 64,
        selected_node_id=finalist.node_id,
        stop_reason="tree_final_selection_committed",
    )
    result = TreeResult(
        tree_id=tree_id,
        status="completed",
        stop_reason="tree_frontier_exhausted",
        selected_node_id=finalist.node_id,
        nodes=(finalist,),
        checkpoint=checkpoint,
    )

    assert TreeCheckpoint.from_mapping(checkpoint.to_dict()) == checkpoint
    assert TreeResult.from_mapping(result.to_dict()) == result
    with pytest.raises(GitTreeContractError, match="nodes differ"):
        replace(result, nodes=())
    with pytest.raises(GitTreeContractError, match="finalist differs"):
        replace(result, selected_node_id="")
    with pytest.raises(GitTreeContractError, match="cannot expose a finalist"):
        replace(result, status="failed")


def test_incomplete_evaluation_cannot_be_eligible_or_selected():
    with pytest.raises(GitTreeContractError, match="full coverage"):
        replace(_evaluation(1.0), execution_coverage=0.875)
    ineligible_eval = TreeEvaluation(
        score=None,
        eligible=False,
        reason="snapshot_miss",
        execution_coverage=0.875,
        snapshot_miss_count=1,
        true_miss_count=1,
        failure_count=0,
        zero_output_count=0,
        snapshot_hash=SNAPSHOT_HASH,
        dev_set_hash=DEV_SET_HASH,
        policy="strict",
        score_version="dev-v1",
        receipt_root=RECEIPT_ROOT,
    )
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-ineligible", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    node = TreeNode(
        tree_id=tree_id,
        node_id=slot.node_id,
        parent_node_id="root",
        root_branch_id=slot.root_branch_id,
        depth=1,
        slot_index=0,
        status="ineligible",
        branch_objective_path_id="path-ineligible",
        branch_objective_hash=OBJECTIVE_HASH,
        generation_attempt_count=1,
        git_commit=GIT_COMMIT,
        source_tree_hash=SOURCE_TREE_HASH,
        incremental_patch_hash=INCREMENTAL_PATCH_HASH,
        cumulative_patch_hash=CUMULATIVE_PATCH_HASH,
        candidate_artifact_hash=CANDIDATE_ARTIFACT_HASH,
        lineage_hash=LINEAGE_HASH,
        evaluation=ineligible_eval,
    )
    assert select_finalist([node]) is None


def test_tree_node_terminal_status_must_match_evaluation_evidence():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-status-evidence", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id="root",
        root_branch_id="",
        depth=1,
        slot_index=0,
    )
    eligible = _node(slot, 10.0)

    with pytest.raises(
        GitTreeContractError, match="ineligible node requires an ineligible"
    ):
        replace(eligible, status="ineligible")
    with pytest.raises(
        GitTreeContractError, match="failed node cannot carry evaluation"
    ):
        replace(eligible, status="failed")
    with pytest.raises(
        GitTreeContractError, match="ineligible node requires an ineligible"
    ):
        replace(eligible, status="ineligible", evaluation=None)


def test_evaluation_summary_counts_built_nodes_without_false_silent_misses():
    policy = TreePolicy(mode="active")
    tree_id = derive_tree_id(
        run_id="run-summary", root_artifact_hash=ROOT_HASH, policy=policy
    )
    slots = [
        derive_child_slot(
            tree_id=tree_id,
            parent_node_id="root",
            root_branch_id="",
            depth=1,
            slot_index=index,
        )
        for index in range(2)
    ]
    eligible = _node(slots[0], 10.0)
    ineligible = replace(
        _node(slots[1], 9.0),
        status="ineligible",
        evaluation=TreeEvaluation(
            score=None,
            eligible=False,
            reason="snapshot_miss",
            execution_coverage=0.875,
            snapshot_miss_count=1,
            true_miss_count=1,
            failure_count=0,
            zero_output_count=0,
            snapshot_hash=SNAPSHOT_HASH,
            dev_set_hash=DEV_SET_HASH,
            policy="strict",
            score_version="dev-v1",
            receipt_root=RECEIPT_ROOT,
            unclassified_error=False,
        ),
    )
    summary = summarize_tree_evaluations((eligible, ineligible))

    assert summary["node_count"] == 2
    assert summary["built_node_count"] == 2
    assert summary["evaluated_node_count"] == 2
    assert summary["eligible_node_count"] == 1
    assert summary["missing_evaluation_count"] == 0
    assert summary["snapshot_miss_count"] == 1
    assert summary["ineligible_reason_counts"] == {"snapshot_miss": 1}
