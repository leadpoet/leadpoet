"""Measured contracts and deterministic scheduling for Git-tree autoresearch."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

from gateway.research_lab.config import (
    DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG,
    DEPRECATED_RESEARCH_LAB_GIT_TREE_ENV_NAMES,
    MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
    RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD,
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)
from leadpoet_canonical.attested_v2 import sha256_json


TREE_POLICY_SCHEMA_VERSION = "research_lab.git_tree_policy.v1"
TREE_NODE_SCHEMA_VERSION = "research_lab.git_tree_node.v1"
TREE_EVALUATION_SCHEMA_VERSION = "research_lab.git_tree_evaluation.v1"
TREE_CHECKPOINT_SCHEMA_VERSION = "research_lab.git_tree_checkpoint.v1"
TREE_RESULT_SCHEMA_VERSION = "research_lab.git_tree_result.v1"

TREE_MODE_ENV = RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["mode"]
TREE_MODES = frozenset({"off", "active"})
DEPRECATED_TREE_ENV_NAMES = DEPRECATED_RESEARCH_LAB_GIT_TREE_ENV_NAMES
TREE_NODE_TERMINAL_STATUSES = frozenset(
    {"eligible", "ineligible", "failed", "cancelled", "indeterminate"}
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NODE_ID_RE = re.compile(r"^tree-node:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{64}$")


class GitTreeContractError(ValueError):
    """A tree policy, node, checkpoint, or result is not authoritative."""


def _bounded_int(value: Any, *, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise GitTreeContractError(f"{field_name} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise GitTreeContractError(f"{field_name} must be an integer") from exc
    if normalized < minimum or normalized > maximum:
        raise GitTreeContractError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return normalized


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise GitTreeContractError(f"{field_name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise GitTreeContractError(f"{field_name} must be finite") from exc
    if not math.isfinite(normalized):
        raise GitTreeContractError(f"{field_name} must be finite")
    return normalized


def _hash(value: Any, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise GitTreeContractError(f"{field_name} must be a sha256 commitment")
    return normalized


def _node_id(value: Any, *, field_name: str, allow_root: bool = False) -> str:
    normalized = str(value or "").strip().lower()
    if allow_root and normalized == "root":
        return normalized
    if not _NODE_ID_RE.fullmatch(normalized):
        raise GitTreeContractError(f"{field_name} is invalid")
    return normalized


@dataclass(frozen=True)
class TreePolicy:
    mode: str = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.mode
    branch_factor: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.branch_factor
    beam_width: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.beam_width
    max_depth: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.max_depth
    max_nodes: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.max_nodes
    generation_attempts: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.generation_attempts
    )
    build_concurrency: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.build_concurrency
    )
    evaluation_concurrency: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.evaluation_concurrency
    )
    shortlist_size: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.shortlist_size
    diversity_floor: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.diversity_floor
    deadline_seconds: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.deadline_seconds
    finalization_reserve_seconds: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.finalization_reserve_seconds
    )
    billable_cap_microusd: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.billable_cap_microusd
    )
    live_max_icps_per_node: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    )
    live_max_provider_calls: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_provider_calls
    )
    live_cap_microusd: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_cap_microusd
    )
    live_timeout_seconds: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_timeout_seconds
    )
    evidence_retention_days: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.evidence_retention_days
    )

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        if mode not in TREE_MODES:
            raise GitTreeContractError("tree mode must be off or active")
        object.__setattr__(self, "mode", mode)
        bounds = {
            "branch_factor": (1, 8),
            "beam_width": (1, 16),
            "max_depth": (1, 12),
            "max_nodes": (1, 128),
            "generation_attempts": (1, 5),
            # Builds remain intentionally serialized because Docker/ECR build
            # operations share host storage and one durable operation ledger.
            "build_concurrency": (1, 1),
            "evaluation_concurrency": (1, 16),
            "shortlist_size": (1, 16),
            "diversity_floor": (1, 16),
            "deadline_seconds": (300, 86_400),
            "finalization_reserve_seconds": (30, 3_600),
            "billable_cap_microusd": (0, 1_000_000_000),
            "live_max_icps_per_node": (
                1,
                MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
            ),
            "live_max_provider_calls": (1, 32),
            "live_cap_microusd": (1, 500_000),
            "live_timeout_seconds": (30, 3_600),
            "evidence_retention_days": (1, 365),
        }
        for name, (minimum, maximum) in bounds.items():
            object.__setattr__(
                self,
                name,
                _bounded_int(
                    getattr(self, name),
                    field_name=name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.beam_width > self.max_nodes:
            raise GitTreeContractError("beam_width cannot exceed max_nodes")
        if self.branch_factor > self.max_nodes:
            raise GitTreeContractError("branch_factor cannot exceed max_nodes")
        if self.shortlist_size > self.max_nodes:
            raise GitTreeContractError("shortlist_size cannot exceed max_nodes")
        if self.diversity_floor > self.beam_width:
            raise GitTreeContractError("diversity_floor cannot exceed beam_width")
        if self.diversity_floor > self.branch_factor:
            raise GitTreeContractError("diversity_floor cannot exceed branch_factor")
        if self.diversity_floor > self.shortlist_size:
            raise GitTreeContractError("diversity_floor cannot exceed shortlist_size")
        if self.finalization_reserve_seconds >= self.deadline_seconds:
            raise GitTreeContractError(
                "finalization reserve must be shorter than the tree deadline"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TreePolicy":
        try:
            configured = ResearchLabGitTreeConfig.from_env(environ)
        except ResearchLabGitTreeConfigError as exc:
            raise GitTreeContractError(str(exc)) from exc
        return cls(**configured.to_policy_kwargs())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TREE_POLICY_SCHEMA_VERSION,
            "mode": self.mode,
            "branch_factor": self.branch_factor,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "generation_attempts": self.generation_attempts,
            "build_concurrency": self.build_concurrency,
            "evaluation_concurrency": self.evaluation_concurrency,
            "shortlist_size": self.shortlist_size,
            "diversity_floor": self.diversity_floor,
            "deadline_seconds": self.deadline_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "billable_cap_microusd": self.billable_cap_microusd,
            "live_max_icps_per_node": self.live_max_icps_per_node,
            "live_max_provider_calls": self.live_max_provider_calls,
            "live_cap_microusd": self.live_cap_microusd,
            "live_timeout_seconds": self.live_timeout_seconds,
            "evidence_retention_days": self.evidence_retention_days,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreePolicy":
        document = dict(value)
        schema_version = document.pop("schema_version", TREE_POLICY_SCHEMA_VERSION)
        if schema_version != TREE_POLICY_SCHEMA_VERSION:
            raise GitTreeContractError("tree policy schema version is invalid")
        expected = {
            "mode",
            "branch_factor",
            "beam_width",
            "max_depth",
            "max_nodes",
            "generation_attempts",
            "build_concurrency",
            "evaluation_concurrency",
            "shortlist_size",
            "diversity_floor",
            "deadline_seconds",
            "finalization_reserve_seconds",
            "billable_cap_microusd",
            "live_max_icps_per_node",
            "live_max_provider_calls",
            "live_cap_microusd",
            "live_timeout_seconds",
            "evidence_retention_days",
        }
        if set(document) != expected:
            raise GitTreeContractError("tree policy fields are invalid")
        return cls(**document)

    @property
    def policy_hash(self) -> str:
        return sha256_json(self.to_dict())

    def effective_billable_cap(self, funded_budget_microusd: int) -> int:
        funded = max(0, int(funded_budget_microusd))
        if not funded:
            return 0
        if not self.billable_cap_microusd:
            return funded
        return min(funded, self.billable_cap_microusd)

    def required_final_context_seconds(
        self, evaluation_timeout_seconds: int
    ) -> int:
        """Time tree work must leave for final comparison and handoff."""

        timeout = _bounded_int(
            evaluation_timeout_seconds,
            field_name="evaluation_timeout_seconds",
            minimum=1,
            maximum=86_400,
        )
        required = self.finalization_reserve_seconds + timeout
        if required >= self.deadline_seconds:
            raise GitTreeContractError(
                "tree deadline must exceed the final evaluation timeout plus "
                "the finalization reserve"
            )
        return required


@dataclass(frozen=True)
class TreeEvaluation:
    score: float | None
    eligible: bool
    reason: str
    execution_coverage: float
    snapshot_miss_count: int
    true_miss_count: int
    failure_count: int
    zero_output_count: int
    snapshot_hash: str
    dev_set_hash: str
    policy: str
    score_version: str
    receipt_root: str
    context_hash: str = ""
    parent_delta: float | None = None
    settled_cost_microusd: int = 0
    provider_call_count: int = 0
    evaluation_mode: str = "replay"
    unclassified_error: bool = False
    feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None:
            object.__setattr__(
                self,
                "score",
                _finite_float(self.score, field_name="evaluation score"),
            )
        if self.parent_delta is not None:
            object.__setattr__(
                self,
                "parent_delta",
                _finite_float(self.parent_delta, field_name="parent_delta"),
            )
        coverage = _finite_float(
            self.execution_coverage, field_name="execution_coverage"
        )
        if coverage < 0.0 or coverage > 1.0:
            raise GitTreeContractError("execution_coverage must be between 0 and 1")
        object.__setattr__(self, "execution_coverage", coverage)
        for name in (
            "snapshot_miss_count",
            "true_miss_count",
            "failure_count",
            "zero_output_count",
            "settled_cost_microusd",
            "provider_call_count",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_int(
                    getattr(self, name),
                    field_name=name,
                    minimum=0,
                    maximum=1_000_000_000,
                ),
            )
        if not isinstance(self.unclassified_error, bool):
            raise GitTreeContractError("unclassified_error must be a boolean")
        if self.eligible and self.score is None:
            raise GitTreeContractError("eligible evaluation requires a score")
        if self.eligible and coverage != 1.0:
            raise GitTreeContractError("eligible evaluation requires full coverage")
        if self.eligible and (self.true_miss_count or self.failure_count):
            raise GitTreeContractError(
                "eligible evaluation cannot contain misses or failures"
            )
        if str(self.policy or "") not in {"strict", "hybrid"}:
            raise GitTreeContractError("evaluation policy must be strict or hybrid")
        if str(self.evaluation_mode or "") not in {"replay", "hybrid"}:
            raise GitTreeContractError("evaluation mode must be replay or hybrid")
        if self.snapshot_hash:
            _hash(self.snapshot_hash, field_name="snapshot_hash")
        if self.dev_set_hash:
            _hash(self.dev_set_hash, field_name="dev_set_hash")
        if self.receipt_root:
            _hash(self.receipt_root, field_name="receipt_root")
        if self.context_hash:
            _hash(self.context_hash, field_name="context_hash")
        if self.eligible and not self.context_hash:
            raise GitTreeContractError(
                "eligible evaluation requires a common context commitment"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TREE_EVALUATION_SCHEMA_VERSION,
            "score": self.score,
            "eligible": self.eligible,
            "reason": str(self.reason or "")[:240],
            "execution_coverage": self.execution_coverage,
            "snapshot_miss_count": self.snapshot_miss_count,
            "true_miss_count": self.true_miss_count,
            "failure_count": self.failure_count,
            "zero_output_count": self.zero_output_count,
            "snapshot_hash": self.snapshot_hash,
            "dev_set_hash": self.dev_set_hash,
            "policy": self.policy,
            "score_version": self.score_version,
            "receipt_root": self.receipt_root,
            "context_hash": self.context_hash,
            "parent_delta": self.parent_delta,
            "settled_cost_microusd": self.settled_cost_microusd,
            "provider_call_count": self.provider_call_count,
            "evaluation_mode": self.evaluation_mode,
            "unclassified_error": self.unclassified_error,
            "feedback": dict(self.feedback),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreeEvaluation":
        document = dict(value)
        if document.pop("schema_version", "") != TREE_EVALUATION_SCHEMA_VERSION:
            raise GitTreeContractError("tree evaluation schema version is invalid")
        return cls(**document)


@dataclass(frozen=True)
class TreeNode:
    tree_id: str
    node_id: str
    parent_node_id: str
    root_branch_id: str
    depth: int
    slot_index: int
    status: str
    branch_objective_path_id: str = ""
    branch_objective_hash: str = ""
    generation_attempt_count: int = 0
    git_commit: str = ""
    source_tree_hash: str = ""
    incremental_patch_hash: str = ""
    cumulative_patch_hash: str = ""
    candidate_artifact_hash: str = ""
    lineage_hash: str = ""
    complexity: int = 0
    settled_cost_microusd: int = 0
    evaluation: TreeEvaluation | None = None

    def __post_init__(self) -> None:
        _hash(self.tree_id, field_name="tree_id")
        _node_id(self.node_id, field_name="node_id")
        _node_id(self.parent_node_id, field_name="parent_node_id", allow_root=True)
        _node_id(self.root_branch_id, field_name="root_branch_id")
        object.__setattr__(
            self,
            "depth",
            _bounded_int(self.depth, field_name="depth", minimum=1, maximum=128),
        )
        object.__setattr__(
            self,
            "slot_index",
            _bounded_int(
                self.slot_index, field_name="slot_index", minimum=0, maximum=128
            ),
        )
        status = str(self.status or "").strip().lower()
        if status not in {
            "planned",
            "generating",
            "generated",
            "building",
            "evaluating",
            *TREE_NODE_TERMINAL_STATUSES,
        }:
            raise GitTreeContractError("tree node status is invalid")
        object.__setattr__(self, "status", status)
        objective_path_id = str(self.branch_objective_path_id or "").strip()
        if len(objective_path_id) > 160:
            raise GitTreeContractError("branch_objective_path_id is too long")
        object.__setattr__(self, "branch_objective_path_id", objective_path_id)
        objective_hash = str(self.branch_objective_hash or "")
        if objective_hash:
            _hash(objective_hash, field_name="branch_objective_hash")
        object.__setattr__(self, "branch_objective_hash", objective_hash)
        object.__setattr__(
            self,
            "generation_attempt_count",
            _bounded_int(
                self.generation_attempt_count,
                field_name="generation_attempt_count",
                minimum=0,
                maximum=5,
            ),
        )
        if status in {"generated", "building", "evaluating", "eligible", "ineligible"}:
            if not objective_path_id or not objective_hash:
                raise GitTreeContractError(
                    "built tree node requires a committed branch objective"
                )
            if self.generation_attempt_count < 1:
                raise GitTreeContractError(
                    "built tree node requires at least one generation attempt"
                )
        git_commit = str(self.git_commit or "").strip().lower()
        if git_commit and not _GIT_COMMIT_RE.fullmatch(git_commit):
            raise GitTreeContractError("git_commit must be a SHA-256 Git commit")
        object.__setattr__(self, "git_commit", git_commit)
        committed_hash_fields = (
            "source_tree_hash",
            "incremental_patch_hash",
            "cumulative_patch_hash",
            "candidate_artifact_hash",
            "lineage_hash",
        )
        for name in committed_hash_fields:
            value = str(getattr(self, name) or "")
            if value:
                object.__setattr__(self, name, _hash(value, field_name=name))
        if status in {"evaluating", "eligible", "ineligible"}:
            missing = [
                name
                for name in committed_hash_fields
                if not str(getattr(self, name) or "")
            ]
            if not git_commit or missing:
                raise GitTreeContractError(
                    "evaluated tree node requires committed Git lineage: "
                    + ", ".join((["git_commit"] if not git_commit else []) + missing)
                )
        object.__setattr__(
            self,
            "complexity",
            _bounded_int(
                self.complexity, field_name="complexity", minimum=0, maximum=1_000_000
            ),
        )
        object.__setattr__(
            self,
            "settled_cost_microusd",
            _bounded_int(
                self.settled_cost_microusd,
                field_name="settled_cost_microusd",
                minimum=0,
                maximum=1_000_000_000,
            ),
        )
        if self.status == "eligible" and (
            self.evaluation is None or not self.evaluation.eligible
        ):
            raise GitTreeContractError("eligible node requires an eligible evaluation")
        if self.status == "ineligible" and (
            self.evaluation is None or self.evaluation.eligible
        ):
            raise GitTreeContractError(
                "ineligible node requires an ineligible evaluation"
            )
        if self.status in {"failed", "cancelled", "indeterminate"} and (
            self.evaluation is not None
        ):
            raise GitTreeContractError(
                f"{self.status} node cannot carry evaluation evidence"
            )

    @property
    def eligible(self) -> bool:
        return bool(self.evaluation is not None and self.evaluation.eligible)

    @property
    def score(self) -> float | None:
        return self.evaluation.score if self.evaluation is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TREE_NODE_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "root_branch_id": self.root_branch_id,
            "depth": self.depth,
            "slot_index": self.slot_index,
            "status": self.status,
            "branch_objective_path_id": self.branch_objective_path_id,
            "branch_objective_hash": self.branch_objective_hash,
            "generation_attempt_count": self.generation_attempt_count,
            "git_commit": self.git_commit,
            "source_tree_hash": self.source_tree_hash,
            "incremental_patch_hash": self.incremental_patch_hash,
            "cumulative_patch_hash": self.cumulative_patch_hash,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "lineage_hash": self.lineage_hash,
            "complexity": self.complexity,
            "settled_cost_microusd": self.settled_cost_microusd,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreeNode":
        document = dict(value)
        if document.pop("schema_version", "") != TREE_NODE_SCHEMA_VERSION:
            raise GitTreeContractError("tree node schema version is invalid")
        evaluation = document.get("evaluation")
        document["evaluation"] = (
            TreeEvaluation.from_mapping(evaluation)
            if isinstance(evaluation, Mapping)
            else None
        )
        return cls(**document)


def summarize_tree_evaluations(nodes: Sequence[TreeNode]) -> dict[str, Any]:
    """Return bounded telemetry from the final committed scheduler state."""

    node_status_counts: dict[str, int] = {}
    evaluation_mode_counts: dict[str, int] = {}
    ineligible_reason_counts: dict[str, int] = {}
    built_nodes: list[TreeNode] = []
    evaluated_nodes: list[TreeNode] = []
    for node in nodes:
        node_status_counts[node.status] = node_status_counts.get(node.status, 0) + 1
        if node.status in {"evaluating", "eligible", "ineligible"}:
            built_nodes.append(node)
        evaluation = node.evaluation
        if evaluation is None:
            continue
        evaluated_nodes.append(node)
        evaluation_mode_counts[evaluation.evaluation_mode] = (
            evaluation_mode_counts.get(evaluation.evaluation_mode, 0) + 1
        )
        if not evaluation.eligible:
            reason = str(evaluation.reason or "unspecified")[:160]
            ineligible_reason_counts[reason] = (
                ineligible_reason_counts.get(reason, 0) + 1
            )
    return {
        "schema_version": "research_lab.git_tree_evaluation_summary.v1",
        "node_count": len(nodes),
        "built_node_count": len(built_nodes),
        "evaluated_node_count": len(evaluated_nodes),
        "eligible_node_count": sum(1 for node in evaluated_nodes if node.eligible),
        "missing_evaluation_count": max(
            0, len(built_nodes) - len(evaluated_nodes)
        ),
        "unclassified_error_count": sum(
            1
            for node in evaluated_nodes
            if node.evaluation is not None and node.evaluation.unclassified_error
        ),
        "snapshot_miss_count": sum(
            node.evaluation.snapshot_miss_count
            for node in evaluated_nodes
            if node.evaluation is not None
        ),
        "true_miss_count": sum(
            node.evaluation.true_miss_count
            for node in evaluated_nodes
            if node.evaluation is not None
        ),
        "failure_count": sum(
            node.evaluation.failure_count
            for node in evaluated_nodes
            if node.evaluation is not None
        ),
        "zero_output_count": sum(
            node.evaluation.zero_output_count
            for node in evaluated_nodes
            if node.evaluation is not None
        ),
        "evaluation_mode_counts": dict(sorted(evaluation_mode_counts.items())),
        "ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "node_status_counts": dict(sorted(node_status_counts.items())),
    }


@dataclass(frozen=True)
class TreeChildSlot:
    tree_id: str
    node_id: str
    parent_node_id: str
    root_branch_id: str
    depth: int
    slot_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "root_branch_id": self.root_branch_id,
            "depth": self.depth,
            "slot_index": self.slot_index,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreeChildSlot":
        expected = {
            "tree_id",
            "node_id",
            "parent_node_id",
            "root_branch_id",
            "depth",
            "slot_index",
        }
        document = dict(value)
        if set(document) != expected:
            raise GitTreeContractError("tree child slot fields are invalid")
        slot = cls(**document)
        expected_slot = derive_child_slot(
            tree_id=slot.tree_id,
            parent_node_id=slot.parent_node_id,
            root_branch_id=slot.root_branch_id,
            depth=slot.depth,
            slot_index=slot.slot_index,
        )
        if slot != expected_slot:
            raise GitTreeContractError("tree child slot identity differs")
        return slot


@dataclass(frozen=True)
class TreeCheckpoint:
    tree_id: str
    root_artifact_hash: str
    policy: TreePolicy
    nodes: tuple[TreeNode, ...]
    frontier_hash: str
    operation_settlement_hash: str
    planned_slots: tuple[TreeChildSlot, ...] = ()
    selected_node_id: str = ""
    stop_reason: str = ""

    def __post_init__(self) -> None:
        _hash(self.tree_id, field_name="tree_id")
        _hash(self.root_artifact_hash, field_name="root_artifact_hash")
        if not isinstance(self.policy, TreePolicy) or self.policy.mode != "active":
            raise GitTreeContractError("tree checkpoint requires an active policy")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "planned_slots", tuple(self.planned_slots))
        _hash(self.frontier_hash, field_name="frontier_hash")
        _hash(
            self.operation_settlement_hash,
            field_name="operation_settlement_hash",
        )
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GitTreeContractError("checkpoint contains duplicate nodes")
        if any(node.tree_id != self.tree_id for node in self.nodes):
            raise GitTreeContractError("checkpoint node belongs to another tree")
        planned_ids = [slot.node_id for slot in self.planned_slots]
        if len(planned_ids) != len(set(planned_ids)):
            raise GitTreeContractError("checkpoint contains duplicate planned slots")
        if set(node_ids) & set(planned_ids):
            raise GitTreeContractError("checkpoint node is both planned and committed")
        if any(slot.tree_id != self.tree_id for slot in self.planned_slots):
            raise GitTreeContractError("checkpoint planned slot belongs to another tree")
        if self.selected_node_id:
            _node_id(self.selected_node_id, field_name="selected_node_id")
            selected = [
                node for node in self.nodes if node.node_id == self.selected_node_id
            ]
            if len(selected) != 1 or not selected[0].eligible:
                raise GitTreeContractError(
                    "selected checkpoint node is missing or ineligible"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TREE_CHECKPOINT_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "root_artifact_hash": self.root_artifact_hash,
            "policy": self.policy.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "frontier_hash": self.frontier_hash,
            "operation_settlement_hash": self.operation_settlement_hash,
            "planned_slots": [slot.to_dict() for slot in self.planned_slots],
            "selected_node_id": self.selected_node_id,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreeCheckpoint":
        document = dict(value)
        if document.pop("schema_version", "") != TREE_CHECKPOINT_SCHEMA_VERSION:
            raise GitTreeContractError("tree checkpoint schema version is invalid")
        policy = document.get("policy")
        nodes = document.get("nodes")
        planned_slots = document.get("planned_slots", [])
        if (
            not isinstance(policy, Mapping)
            or not isinstance(nodes, Sequence)
            or not isinstance(planned_slots, Sequence)
        ):
            raise GitTreeContractError("tree checkpoint payload is invalid")
        document["policy"] = TreePolicy.from_mapping(policy)
        document["nodes"] = tuple(
            TreeNode.from_mapping(node)
            for node in nodes
            if isinstance(node, Mapping)
        )
        if len(document["nodes"]) != len(nodes):
            raise GitTreeContractError("tree checkpoint node payload is invalid")
        document["planned_slots"] = tuple(
            TreeChildSlot.from_mapping(slot)
            for slot in planned_slots
            if isinstance(slot, Mapping)
        )
        if len(document["planned_slots"]) != len(planned_slots):
            raise GitTreeContractError("tree checkpoint planned-slot payload is invalid")
        return cls(**document)


@dataclass(frozen=True)
class TreeResult:
    tree_id: str
    status: str
    stop_reason: str
    selected_node_id: str
    nodes: tuple[TreeNode, ...]
    checkpoint: TreeCheckpoint

    def __post_init__(self) -> None:
        _hash(self.tree_id, field_name="tree_id")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        status = str(self.status or "").strip().lower()
        if status not in {"completed", "paused", "failed"}:
            raise GitTreeContractError("tree result status is invalid")
        object.__setattr__(self, "status", status)
        if not isinstance(self.checkpoint, TreeCheckpoint):
            raise GitTreeContractError("tree result checkpoint is invalid")
        if self.checkpoint.tree_id != self.tree_id:
            raise GitTreeContractError("tree result checkpoint belongs to another tree")
        if self.checkpoint.nodes != self.nodes:
            raise GitTreeContractError("tree result nodes differ from its checkpoint")
        if self.checkpoint.selected_node_id != self.selected_node_id:
            raise GitTreeContractError(
                "tree result finalist differs from its checkpoint"
            )
        selected = [node for node in self.nodes if node.node_id == self.selected_node_id]
        if self.selected_node_id:
            _node_id(self.selected_node_id, field_name="selected_node_id")
            if len(selected) != 1 or not selected[0].eligible:
                raise GitTreeContractError("selected tree node is missing or ineligible")
            if status != "completed":
                raise GitTreeContractError(
                    "paused or failed tree cannot expose a finalist"
                )
        elif status == "completed":
            raise GitTreeContractError("completed tree requires exactly one finalist")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TREE_RESULT_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "selected_node_id": self.selected_node_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TreeResult":
        document = dict(value)
        if document.pop("schema_version", "") != TREE_RESULT_SCHEMA_VERSION:
            raise GitTreeContractError("tree result schema version is invalid")
        nodes = document.get("nodes")
        checkpoint = document.get("checkpoint")
        if not isinstance(nodes, Sequence) or not isinstance(checkpoint, Mapping):
            raise GitTreeContractError("tree result payload is invalid")
        document["nodes"] = tuple(
            TreeNode.from_mapping(node)
            for node in nodes
            if isinstance(node, Mapping)
        )
        if len(document["nodes"]) != len(nodes):
            raise GitTreeContractError("tree result node payload is invalid")
        document["checkpoint"] = TreeCheckpoint.from_mapping(checkpoint)
        return cls(**document)


def derive_tree_id(*, run_id: str, root_artifact_hash: str, policy: TreePolicy) -> str:
    return sha256_json(
        {
            "schema_version": "research_lab.git_tree_identity.v1",
            "run_id": str(run_id),
            "root_artifact_hash": _hash(
                root_artifact_hash, field_name="root_artifact_hash"
            ),
            "policy_hash": policy.policy_hash,
        }
    )


def derive_frontier_commitment_hash(
    *, tree_id: str, scheduler_frontier_hash: str, checkpoint_hash: str
) -> str:
    return sha256_json(
        {
            "schema_version": "research_lab.git_tree_frontier_checkpoint.v1",
            "tree_id": _hash(tree_id, field_name="tree_id"),
            "scheduler_frontier_hash": _hash(
                scheduler_frontier_hash,
                field_name="scheduler_frontier_hash",
            ),
            "checkpoint_hash": _hash(
                checkpoint_hash,
                field_name="checkpoint_hash",
            ),
        }
    )


def derive_child_slot(
    *,
    tree_id: str,
    parent_node_id: str,
    root_branch_id: str,
    depth: int,
    slot_index: int,
) -> TreeChildSlot:
    tree_id = _hash(tree_id, field_name="tree_id")
    parent = _node_id(parent_node_id, field_name="parent_node_id", allow_root=True)
    root_branch = (
        ""
        if parent == "root"
        else _node_id(root_branch_id, field_name="root_branch_id")
    )
    depth = _bounded_int(depth, field_name="depth", minimum=1, maximum=128)
    slot_index = _bounded_int(
        slot_index, field_name="slot_index", minimum=0, maximum=128
    )
    node_hash = sha256_json(
        {
            "tree_id": tree_id,
            "parent_node_id": parent,
            "depth": depth,
            "slot_index": slot_index,
        }
    ).split(":", 1)[1]
    node_id = f"tree-node:{node_hash}"
    return TreeChildSlot(
        tree_id=tree_id,
        node_id=node_id,
        parent_node_id=parent,
        root_branch_id=node_id if parent == "root" else root_branch,
        depth=depth,
        slot_index=slot_index,
    )


def generation_operation_id(slot: TreeChildSlot) -> str:
    """Return the stable logical identity for one child-generation operation."""

    return sha256_json(
        {
            "schema_version": "research_lab.git_tree_operation_identity.v1",
            "tree_id": slot.tree_id,
            "node_id": slot.node_id,
            "operation_kind": "generation",
        }
    )


def build_operation_id(slot: TreeChildSlot) -> str:
    """Return the stable logical identity for one child image build."""

    return sha256_json(
        {
            "schema_version": "research_lab.git_tree_operation_identity.v1",
            "tree_id": slot.tree_id,
            "node_id": slot.node_id,
            "operation_kind": "build",
        }
    )


def cohort_evaluation_operation_id(
    *,
    tree_id: str,
    node_ids: Sequence[str],
    stage: str,
) -> str:
    """Return one stable identity for a complete round or final shortlist."""

    normalized_tree_id = _hash(tree_id, field_name="tree_id")
    normalized_node_ids = tuple(
        sorted(_node_id(node_id, field_name="node_id") for node_id in node_ids)
    )
    if not normalized_node_ids or len(normalized_node_ids) != len(
        set(normalized_node_ids)
    ):
        raise GitTreeContractError(
            "evaluation operation requires distinct tree nodes"
        )
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage not in {"round", "final_shortlist"}:
        raise GitTreeContractError("evaluation operation stage is invalid")
    return sha256_json(
        {
            "schema_version": "research_lab.git_tree_operation_identity.v1",
            "tree_id": normalized_tree_id,
            "node_ids": list(normalized_node_ids),
            "operation_kind": "evaluation",
            "stage": normalized_stage,
        }
    )


def tree_rank_key(node: TreeNode) -> tuple[Any, ...]:
    if not node.eligible or node.score is None:
        return (1, 0.0, 0.0, 0.0, node.complexity, node.settled_cost_microusd, node.node_id)
    evaluation = node.evaluation
    assert evaluation is not None
    parent_delta = float(evaluation.parent_delta or 0.0)
    return (
        0,
        -float(node.score),
        -float(evaluation.execution_coverage),
        -parent_delta,
        node.complexity,
        node.settled_cost_microusd,
        node.node_id,
    )


def select_finalist(nodes: Sequence[TreeNode]) -> TreeNode | None:
    eligible = [node for node in nodes if node.eligible]
    context_hashes = {
        node.evaluation.context_hash for node in eligible if node.evaluation
    }
    if len(context_hashes) > 1:
        raise GitTreeContractError(
            "finalists do not share one evaluation context commitment"
        )
    return min(eligible, key=tree_rank_key) if eligible else None


def next_child_slot(
    *,
    tree_id: str,
    policy: TreePolicy,
    nodes: Sequence[TreeNode],
    planned_node_ids: Sequence[str] = (),
) -> TreeChildSlot | None:
    """Return one deterministic child slot while preserving root diversity."""

    if policy.mode != "active" or len(nodes) + len(planned_node_ids) >= policy.max_nodes:
        return None
    existing_ids = {node.node_id for node in nodes} | set(planned_node_ids)
    roots = sorted(
        (node for node in nodes if node.parent_node_id == "root"),
        key=lambda node: (node.slot_index, node.node_id),
    )
    if len(roots) < policy.branch_factor:
        slot_index = len(roots)
        slot = derive_child_slot(
            tree_id=tree_id,
            parent_node_id="root",
            root_branch_id="",
            depth=1,
            slot_index=slot_index,
        )
        return None if slot.node_id in existing_ids else slot

    eligible = [
        node
        for node in nodes
        if node.eligible and node.depth < policy.max_depth
    ]
    if not eligible:
        return None
    children_by_parent: dict[str, list[TreeNode]] = {}
    for node in nodes:
        children_by_parent.setdefault(node.parent_node_id, []).append(node)
    expandable = [
        node
        for node in eligible
        if len(children_by_parent.get(node.node_id, ())) < policy.branch_factor
    ]
    if not expandable:
        return None

    branches: dict[str, list[TreeNode]] = {}
    for node in expandable:
        branches.setdefault(node.root_branch_id, []).append(node)
    branch_best = sorted(
        (min(items, key=tree_rank_key) for items in branches.values()),
        key=tree_rank_key,
    )
    admitted_count = min(
        len(branch_best),
        max(policy.diversity_floor, policy.beam_width),
    )
    admitted_branches = {node.root_branch_id for node in branch_best[:admitted_count]}
    candidates = [
        node for node in expandable if node.root_branch_id in admitted_branches
    ]
    branch_child_counts: dict[str, int] = {}
    for node in nodes:
        if node.parent_node_id != "root":
            branch_child_counts[node.root_branch_id] = (
                branch_child_counts.get(node.root_branch_id, 0) + 1
            )
    parent = min(
        candidates,
        key=lambda node: (
            branch_child_counts.get(node.root_branch_id, 0),
            tree_rank_key(node),
        ),
    )
    slot_index = len(children_by_parent.get(parent.node_id, ()))
    slot = derive_child_slot(
        tree_id=tree_id,
        parent_node_id=parent.node_id,
        root_branch_id=parent.root_branch_id,
        depth=parent.depth + 1,
        slot_index=slot_index,
    )
    return None if slot.node_id in existing_ids else slot
