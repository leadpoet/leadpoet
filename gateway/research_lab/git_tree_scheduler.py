"""Deterministic, restart-safe beam scheduling for Git-tree autoresearch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from gateway.research_lab.git_tree_models import (
    GitTreeContractError,
    TreeChildSlot,
    TreeNode,
    TreePolicy,
    derive_child_slot,
    select_finalist,
    tree_rank_key,
)
from leadpoet_canonical.attested_v2 import sha256_json


class GitTreeSchedulerError(GitTreeContractError):
    """The persisted tree cannot advance without changing its commitments."""


@dataclass
class GitTreeScheduler:
    tree_id: str
    policy: TreePolicy
    _nodes: dict[str, TreeNode] = field(default_factory=dict, repr=False)
    _planned: dict[str, TreeChildSlot] = field(default_factory=dict, repr=False)

    @classmethod
    def restore(
        cls,
        *,
        tree_id: str,
        policy: TreePolicy,
        nodes: Iterable[TreeNode],
        planned_slots: Iterable[TreeChildSlot] = (),
    ) -> "GitTreeScheduler":
        scheduler = cls(tree_id=tree_id, policy=policy)
        for node in nodes:
            scheduler.record_node(node)
        for slot in planned_slots:
            scheduler.record_planned_slot(slot)
        scheduler.validate_topology()
        return scheduler

    @property
    def nodes(self) -> tuple[TreeNode, ...]:
        return tuple(
            sorted(
                self._nodes.values(),
                key=lambda item: (item.depth, item.root_branch_id, item.node_id),
            )
        )

    @property
    def planned_slots(self) -> tuple[TreeChildSlot, ...]:
        return tuple(
            sorted(
                self._planned.values(),
                key=lambda item: (item.depth, item.root_branch_id, item.node_id),
            )
        )

    def record_planned_slot(self, slot: TreeChildSlot) -> None:
        if slot.tree_id != self.tree_id:
            raise GitTreeSchedulerError("planned slot belongs to another tree")
        existing_node = self._nodes.get(slot.node_id)
        if existing_node is not None:
            expected = _slot_from_node(existing_node)
            if expected != slot:
                raise GitTreeSchedulerError("planned slot conflicts with an existing node")
            return
        existing = self._planned.get(slot.node_id)
        if existing is not None and existing != slot:
            raise GitTreeSchedulerError("planned slot identity changed")
        self._validate_parent(slot)
        self._planned[slot.node_id] = slot

    def plan_next(self) -> TreeChildSlot | None:
        if not self._planned:
            self.plan_round()
        return self.next_planned()

    def next_planned(self) -> TreeChildSlot | None:
        return (
            min(
                self._planned.values(),
                key=lambda item: (
                    item.depth,
                    item.root_branch_id,
                    item.slot_index,
                    item.node_id,
                ),
            )
            if self._planned
            else None
        )

    def plan_round(self) -> tuple[TreeChildSlot, ...]:
        """Predeclare one complete deterministic depth/beam expansion round."""

        if self._planned:
            return self.planned_slots
        remaining = self.policy.max_nodes - len(self._nodes)
        if self.policy.mode != "active" or remaining <= 0:
            return ()
        roots = sorted(
            (
                node
                for node in self._nodes.values()
                if node.parent_node_id == "root"
            ),
            key=lambda node: (node.slot_index, node.node_id),
        )
        if len(roots) < self.policy.branch_factor:
            for slot_index in range(len(roots), self.policy.branch_factor):
                if remaining <= 0:
                    break
                slot = derive_child_slot(
                    tree_id=self.tree_id,
                    parent_node_id="root",
                    root_branch_id="",
                    depth=1,
                    slot_index=slot_index,
                )
                self.record_planned_slot(slot)
                remaining -= 1
            return self.planned_slots

        parents = self.frontier()
        existing_children: dict[str, set[int]] = {}
        for node in self._nodes.values():
            existing_children.setdefault(node.parent_node_id, set()).add(
                node.slot_index
            )
        for parent in parents:
            for slot_index in range(self.policy.branch_factor):
                if remaining <= 0:
                    break
                if slot_index in existing_children.get(parent.node_id, set()):
                    continue
                slot = derive_child_slot(
                    tree_id=self.tree_id,
                    parent_node_id=parent.node_id,
                    root_branch_id=parent.root_branch_id,
                    depth=parent.depth + 1,
                    slot_index=slot_index,
                )
                self.record_planned_slot(slot)
                remaining -= 1
            if remaining <= 0:
                break
        return self.planned_slots

    def record_node(self, node: TreeNode) -> None:
        if node.tree_id != self.tree_id:
            raise GitTreeSchedulerError("node belongs to another tree")
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise GitTreeSchedulerError("tree node changed after it was recorded")
        planned = self._planned.get(node.node_id)
        if planned is not None and planned != _slot_from_node(node):
            raise GitTreeSchedulerError("tree node differs from its planned slot")
        self._validate_parent(_slot_from_node(node))
        self._nodes[node.node_id] = node
        self._planned.pop(node.node_id, None)

    def replace_node(self, node: TreeNode) -> None:
        """Advance one node while preserving immutable identity and topology."""

        previous = self._nodes.get(node.node_id)
        if previous is None:
            raise GitTreeSchedulerError("cannot advance an unknown tree node")
        immutable = (
            "tree_id",
            "node_id",
            "parent_node_id",
            "root_branch_id",
            "depth",
            "slot_index",
            "branch_objective_path_id",
            "branch_objective_hash",
            "generation_attempt_count",
        )
        if any(getattr(previous, name) != getattr(node, name) for name in immutable):
            raise GitTreeSchedulerError("tree node topology changed")
        self._nodes[node.node_id] = node

    def parent(self, slot_or_node: TreeChildSlot | TreeNode) -> TreeNode | None:
        parent_id = slot_or_node.parent_node_id
        if parent_id == "root":
            return None
        parent = self._nodes.get(parent_id)
        if parent is None:
            raise GitTreeSchedulerError("tree node parent is missing")
        return parent

    def frontier(self) -> tuple[TreeNode, ...]:
        """Return expandable parents with branch diversity and deterministic ties."""

        children_by_parent: dict[str, int] = {}
        for node in self._nodes.values():
            children_by_parent[node.parent_node_id] = (
                children_by_parent.get(node.parent_node_id, 0) + 1
            )
        expandable = [
            node
            for node in self._nodes.values()
            if node.eligible
            and node.depth < self.policy.max_depth
            and children_by_parent.get(node.node_id, 0) < self.policy.branch_factor
        ]
        branch_best: dict[str, TreeNode] = {}
        for node in expandable:
            prior = branch_best.get(node.root_branch_id)
            if prior is None or tree_rank_key(node) < tree_rank_key(prior):
                branch_best[node.root_branch_id] = node
        diverse = sorted(branch_best.values(), key=tree_rank_key)
        chosen = diverse[: min(len(diverse), self.policy.diversity_floor)]
        chosen_ids = {node.node_id for node in chosen}
        for node in sorted(expandable, key=tree_rank_key):
            if len(chosen) >= self.policy.beam_width:
                break
            if node.node_id not in chosen_ids:
                chosen.append(node)
                chosen_ids.add(node.node_id)
        return tuple(chosen)

    def shortlist(self) -> tuple[TreeNode, ...]:
        eligible = [node for node in self._nodes.values() if node.eligible]
        branch_best: dict[str, TreeNode] = {}
        for node in eligible:
            prior = branch_best.get(node.root_branch_id)
            if prior is None or tree_rank_key(node) < tree_rank_key(prior):
                branch_best[node.root_branch_id] = node
        shortlisted = sorted(branch_best.values(), key=tree_rank_key)
        selected_ids = {node.node_id for node in shortlisted}
        for node in sorted(eligible, key=tree_rank_key):
            if len(shortlisted) >= self.policy.shortlist_size:
                break
            if node.node_id not in selected_ids:
                shortlisted.append(node)
                selected_ids.add(node.node_id)
        return tuple(shortlisted[: self.policy.shortlist_size])

    def select_finalist(self) -> TreeNode | None:
        return select_finalist(self.shortlist())

    @property
    def frontier_hash(self) -> str:
        return sha256_json(
            {
                "schema_version": "research_lab.git_tree_frontier.v1",
                "tree_id": self.tree_id,
                "frontier": [node.node_id for node in self.frontier()],
                "planned_slots": [slot.to_dict() for slot in self.planned_slots],
            }
        )

    def validate_topology(self) -> None:
        if len(self._nodes) + len(self._planned) > self.policy.max_nodes:
            raise GitTreeSchedulerError("tree exceeds its node cap")
        for node in self._nodes.values():
            self._validate_parent(_slot_from_node(node))
        for slot in self._planned.values():
            self._validate_parent(slot)
        for node in self._nodes.values():
            seen = {node.node_id}
            parent_id = node.parent_node_id
            while parent_id != "root":
                if parent_id in seen:
                    raise GitTreeSchedulerError("tree contains an ancestry cycle")
                seen.add(parent_id)
                parent = self._nodes.get(parent_id)
                if parent is None:
                    raise GitTreeSchedulerError("tree contains an orphan node")
                parent_id = parent.parent_node_id

    def _validate_parent(self, slot: TreeChildSlot) -> None:
        if slot.parent_node_id == "root":
            if slot.depth != 1 or slot.root_branch_id != slot.node_id:
                raise GitTreeSchedulerError("root child topology is invalid")
            return
        parent = self._nodes.get(slot.parent_node_id)
        if parent is None:
            raise GitTreeSchedulerError("tree child parent is not committed")
        if not parent.eligible:
            raise GitTreeSchedulerError("tree child parent is not eligible")
        if slot.depth != parent.depth + 1:
            raise GitTreeSchedulerError("tree child depth is invalid")
        if slot.root_branch_id != parent.root_branch_id:
            raise GitTreeSchedulerError("tree child branch identity is invalid")


def _slot_from_node(node: TreeNode) -> TreeChildSlot:
    return TreeChildSlot(
        tree_id=node.tree_id,
        node_id=node.node_id,
        parent_node_id=node.parent_node_id,
        root_branch_id=node.root_branch_id,
        depth=node.depth,
        slot_index=node.slot_index,
    )


def sanitized_branch_context(
    *,
    slot: TreeChildSlot,
    parent: TreeNode | None,
    ancestors: Sequence[TreeNode],
) -> dict[str, Any]:
    """Build bounded parent-only feedback for one generation request."""

    nodes_by_id = {node.node_id: node for node in ancestors}
    lineage: list[TreeNode] = []
    cursor = parent
    seen: set[str] = set()
    while cursor is not None:
        if cursor.node_id in seen:
            raise GitTreeSchedulerError("tree branch context contains an ancestry cycle")
        seen.add(cursor.node_id)
        lineage.append(cursor)
        if cursor.parent_node_id == "root":
            break
        cursor = nodes_by_id.get(cursor.parent_node_id)
        if cursor is None:
            raise GitTreeSchedulerError("tree branch context is missing an ancestor")
    lineage.reverse()
    feedback: Mapping[str, Any] = {}
    if parent is not None and parent.evaluation is not None:
        feedback = dict(parent.evaluation.feedback)
    return {
        "schema_version": "research_lab.git_tree_branch_context.v1",
        "tree_id": slot.tree_id,
        "node_id": slot.node_id,
        "parent_node_id": slot.parent_node_id,
        "root_branch_id": slot.root_branch_id,
        "depth": slot.depth,
        "child_slot": slot.slot_index,
        "instruction": (
            "Create one incremental child from the exact parent source. Use only this "
            "branch's ancestry and redacted parent feedback; do not recreate or merge a sibling."
        ),
        "parent_feedback": feedback,
        "ancestor_node_ids": [node.node_id for node in lineage],
        "ancestor_feedback_hashes": [
            sha256_json(dict(node.evaluation.feedback))
            for node in lineage
            if node.evaluation is not None and node.evaluation.feedback
        ],
    }
