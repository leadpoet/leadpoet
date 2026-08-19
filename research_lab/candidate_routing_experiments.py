"""Replay-first company-routing experiments.

This module is deliberately separate from the official Research Lab candidate
evaluation and promotion rails.  It stores the immutable inputs and outputs of
candidate-acquisition routing experiments and evaluates normalized, per-step
outcomes from a frozen provider snapshot.  Route compilation remains owned by
``Sourcing_model``; this module only binds the resulting plan/profile hashes to
an experiment and evaluates its observed steps.

The first vertical slice is replay-only.  A future live/shadow worker may use
the contracts here, but it must continue to execute through the existing
model-authority and SOURCE_ADD provider boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.eval.provider_costs import contains_secret_material
from research_lab.eval.snapshot_store import MODE_REPLAY, ProviderSnapshotStore


ROUTING_EXPERIMENT_SCHEMA_VERSION = "leadpoet.candidate_routing_experiment.v1"
ROUTING_ARM_SCHEMA_VERSION = "leadpoet.candidate_routing_arm.v1"
ROUTING_RUN_SCHEMA_VERSION = "leadpoet.candidate_routing_run.v1"
ROUTING_ATTEMPT_SCHEMA_VERSION = "leadpoet.candidate_routing_attempt.v1"
ROUTING_METRIC_SCHEMA_VERSION = "leadpoet.candidate_routing_metric.v1"
ROUTING_DECISION_SCHEMA_VERSION = "leadpoet.candidate_routing_decision.v1"
ROUTING_EVALUATOR_VERSION = "leadpoet.candidate_routing_replay_evaluator.v1"

ROUTING_MODE_REPLAY = "replay"
ROUTING_RUN_STATUSES = frozenset({"completed", "failed", "replay_miss"})
ROUTING_ATTEMPT_DISPOSITIONS = frozenset(
    {"considered", "selected", "attempted", "succeeded", "missed", "failed", "deferred", "skipped"}
)
ROUTING_ATTEMPT_OUTCOMES = frozenset(
    {"success", "miss", "failed", "retryable_failure", "replay_miss", "skipped"}
)
ROUTING_PROMOTION_STATES = frozenset(
    {"rejected", "replay_only", "eligible_for_shadow", "eligible_for_canary"}
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODEL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_TOOL_ID_RE = re.compile(r"^candidate\.[A-Za-z0-9_.:-]{1,160}$")


class RoutingExperimentError(ValueError):
    """Raised when a routing experiment contract is invalid."""


def _text(value: Any, field_name: str, *, max_length: int = 200) -> str:
    value = str(value or "").strip()
    if not value or len(value) > max_length:
        raise RoutingExperimentError(f"{field_name} must be a non-empty bounded string")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name)
    if not _ID_RE.fullmatch(value):
        raise RoutingExperimentError(f"{field_name} contains unsupported characters")
    return value


def _hash(value: Any, field_name: str) -> str:
    value = _text(value, field_name, max_length=80)
    if not _SHA256_RE.fullmatch(value):
        raise RoutingExperimentError(f"{field_name} must be a sha256 digest")
    return value


def _model_hash(value: Any, field_name: str) -> str:
    value = _text(value, field_name, max_length=64)
    if not _MODEL_SHA256_RE.fullmatch(value):
        raise RoutingExperimentError(
            f"{field_name} must be a model-owned lowercase SHA-256"
        )
    return value


def _tool_id(value: Any) -> str:
    value = _text(value, "tool_id", max_length=180)
    if not _TOOL_ID_RE.fullmatch(value):
        raise RoutingExperimentError("tool_id must be a candidate.* ID")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise RoutingExperimentError(f"{field_name} must be a non-negative integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise RoutingExperimentError(f"{field_name} must be a non-negative integer") from exc
    if value < 0:
        raise RoutingExperimentError(f"{field_name} must be a non-negative integer")
    return value


def _bounded_doc(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RoutingExperimentError(f"{field_name} must be an object")
    doc = dict(value)
    if contains_secret_material(doc) or _contains_secret_key(doc):
        raise RoutingExperimentError(f"{field_name} contains secret-like material")
    try:
        # Also proves that values can participate in the canonical hash.
        sha256_json(doc)
    except (TypeError, ValueError) as exc:
        raise RoutingExperimentError(f"{field_name} is not canonical JSON") from exc
    return doc


def _contains_secret_key(value: Any) -> bool:
    """Reject secret-shaped object keys even when the value is a placeholder."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).strip().lower().replace("-", "_")
            if any(marker in lowered for marker in ("api_key", "password", "raw_secret", "authorization", "service_role")):
                return True
            if _contains_secret_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256_json(dict(payload))


def _sorted_docs(items: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    return [dict(item) for item in sorted(items, key=lambda item: str(item.get(key) or ""))]


@dataclass(frozen=True)
class CandidateRoutingExperiment:
    """Immutable replay experiment specification.

    ``model_commit``, ``routing_contract_hash``, ``profile_registry_hash``,
    ``provider_catalog_hash``, ``dev_set_hash`` and ``snapshot_manifest_hash``
    pin every input that can change an experiment result.  The experiment is
    not an official candidate-evaluation ticket and cannot be promoted by the
    existing scoring worker.
    """

    experiment_id: str
    name: str
    model_commit: str
    model_artifact_hash: str
    routing_contract_hash: str
    profile_registry_hash: str
    provider_catalog_hash: str
    dev_set_hash: str
    snapshot_manifest_hash: str
    target_qualified_count: int
    max_provider_calls: int
    max_cost_microusd: int
    max_duration_ms: int
    evaluator_version: str = ROUTING_EVALUATOR_VERSION
    mode: str = ROUTING_MODE_REPLAY
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTING_EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "name", _text(self.name, "name", max_length=300))
        model_commit = _text(self.model_commit, "model_commit", max_length=40)
        if not _MODEL_COMMIT_RE.fullmatch(model_commit):
            raise RoutingExperimentError("model_commit must be a full lowercase commit")
        object.__setattr__(self, "model_commit", model_commit)
        for field_name in (
            "model_artifact_hash",
            "dev_set_hash",
            "snapshot_manifest_hash",
        ):
            object.__setattr__(self, field_name, _hash(getattr(self, field_name), field_name))
        for field_name in (
            "routing_contract_hash",
            "profile_registry_hash",
            "provider_catalog_hash",
        ):
            object.__setattr__(self, field_name, _model_hash(getattr(self, field_name), field_name))
        if self.mode != ROUTING_MODE_REPLAY:
            raise RoutingExperimentError("routing experiments are replay-only in v1")
        if self.schema_version != ROUTING_EXPERIMENT_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported experiment schema version")
        target_qualified_count = _nonnegative_int(self.target_qualified_count, "target_qualified_count")
        if target_qualified_count < 1:
            raise RoutingExperimentError("target_qualified_count must be at least one")
        object.__setattr__(self, "target_qualified_count", target_qualified_count)
        object.__setattr__(self, "max_provider_calls", _nonnegative_int(self.max_provider_calls, "max_provider_calls"))
        object.__setattr__(self, "max_cost_microusd", _nonnegative_int(self.max_cost_microusd, "max_cost_microusd"))
        object.__setattr__(self, "max_duration_ms", _nonnegative_int(self.max_duration_ms, "max_duration_ms"))
        object.__setattr__(self, "metadata", _bounded_doc(self.metadata, "metadata"))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_commit": self.model_commit,
            "model_artifact_hash": self.model_artifact_hash,
            "routing_contract_hash": self.routing_contract_hash,
            "profile_registry_hash": self.profile_registry_hash,
            "provider_catalog_hash": self.provider_catalog_hash,
            "dev_set_hash": self.dev_set_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "target_qualified_count": self.target_qualified_count,
            "max_provider_calls": self.max_provider_calls,
            "max_cost_microusd": self.max_cost_microusd,
            "max_duration_ms": self.max_duration_ms,
            "evaluator_version": self.evaluator_version,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }

    @property
    def experiment_hash(self) -> str:
        return _payload_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "experiment_hash": self.experiment_hash}


@dataclass(frozen=True)
class CandidateRoutingArm:
    """One control or candidate profile for an experiment."""

    arm_id: str
    experiment_id: str
    experiment_hash: str
    label: str
    profile_id: str
    profile_hash: str
    is_control: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTING_ARM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _identifier(self.arm_id, "arm_id"))
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "experiment_hash", _hash(self.experiment_hash, "experiment_hash"))
        object.__setattr__(self, "label", _text(self.label, "label", max_length=300))
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, "profile_id"))
        object.__setattr__(self, "profile_hash", _model_hash(self.profile_hash, "profile_hash"))
        if self.schema_version != ROUTING_ARM_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported arm schema version")
        object.__setattr__(self, "metadata", _bounded_doc(self.metadata, "metadata"))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "label": self.label,
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "is_control": bool(self.is_control),
            "metadata": dict(self.metadata),
        }

    @property
    def arm_hash(self) -> str:
        return _payload_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "arm_hash": self.arm_hash}


@dataclass(frozen=True)
class CandidateRoutingRun:
    """One deterministic ICP/arm replay run."""

    run_id: str
    experiment_id: str
    experiment_hash: str
    arm_id: str
    icp_ref: str
    icp_hash: str
    snapshot_manifest_hash: str
    route_plan_hash: str
    status: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTING_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("run_id", "experiment_id", "arm_id", "icp_ref"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        for field_name in (
            "experiment_hash",
            "icp_hash",
            "snapshot_manifest_hash",
        ):
            object.__setattr__(self, field_name, _hash(getattr(self, field_name), field_name))
        object.__setattr__(self, "route_plan_hash", _model_hash(self.route_plan_hash, "route_plan_hash"))
        if self.status not in ROUTING_RUN_STATUSES:
            raise RoutingExperimentError("unsupported routing run status")
        if self.schema_version != ROUTING_RUN_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported run schema version")
        object.__setattr__(self, "metadata", _bounded_doc(self.metadata, "metadata"))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "arm_id": self.arm_id,
            "icp_ref": self.icp_ref,
            "icp_hash": self.icp_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "route_plan_hash": self.route_plan_hash,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @property
    def run_hash(self) -> str:
        return _payload_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "run_hash": self.run_hash}


@dataclass(frozen=True)
class CandidateRoutingAttempt:
    """Normalized incremental outcome for one route step.

    Counts are incremental for this step, not cumulative across the route.
    This makes aggregation deterministic and prevents counting the same
    company again when a later tool enriches or verifies it.
    """

    attempt_id: str
    run_id: str
    experiment_id: str
    experiment_hash: str
    arm_id: str
    icp_ref: str
    step_order: int
    attempt_sequence: int
    tool_id: str
    disposition: str
    outcome: str
    route_plan_hash: str
    stop_policy_hash: str
    attempt_receipt_hash: str
    verification_receipt_hash: str = ""
    provider_id: str = ""
    provider_call_count: int = 0
    cost_microusd: int = 0
    latency_ms: int = 0
    raw_count: int = 0
    unique_count: int = 0
    verified_count: int = 0
    qualified_count: int = 0
    published_count: int = 0
    snapshot_hit: bool = True
    result_hash: str = ""
    failure_code: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTING_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("attempt_id", "run_id", "experiment_id", "arm_id", "icp_ref"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "experiment_hash", _hash(self.experiment_hash, "experiment_hash"))
        object.__setattr__(self, "tool_id", _tool_id(self.tool_id))
        for field_name in (
            "route_plan_hash",
            "stop_policy_hash",
            "attempt_receipt_hash",
        ):
            object.__setattr__(self, field_name, _model_hash(getattr(self, field_name), field_name))
        if self.verification_receipt_hash:
            object.__setattr__(
                self,
                "verification_receipt_hash",
                _model_hash(
                    self.verification_receipt_hash,
                    "verification_receipt_hash",
                ),
            )
        if self.disposition not in ROUTING_ATTEMPT_DISPOSITIONS:
            raise RoutingExperimentError("unsupported routing attempt disposition")
        if self.outcome not in ROUTING_ATTEMPT_OUTCOMES:
            raise RoutingExperimentError("unsupported routing attempt outcome")
        if self.schema_version != ROUTING_ATTEMPT_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported attempt schema version")
        for field_name in (
            "step_order",
            "attempt_sequence",
            "provider_call_count",
            "cost_microusd",
            "latency_ms",
            "raw_count",
            "unique_count",
            "verified_count",
            "qualified_count",
            "published_count",
        ):
            object.__setattr__(self, field_name, _nonnegative_int(getattr(self, field_name), field_name))
        if not (self.raw_count >= self.unique_count >= self.verified_count >= self.qualified_count >= self.published_count):
            raise RoutingExperimentError("step counts must satisfy raw >= unique >= verified >= qualified >= published")
        if self.result_hash:
            object.__setattr__(self, "result_hash", _hash(self.result_hash, "result_hash"))
        if self.failure_code:
            object.__setattr__(self, "failure_code", _text(self.failure_code, "failure_code", max_length=120))
        object.__setattr__(self, "provider_id", str(self.provider_id or "").strip()[:120])
        if not isinstance(self.snapshot_hit, bool):
            raise RoutingExperimentError("snapshot_hit must be boolean")
        object.__setattr__(self, "metadata", _bounded_doc(self.metadata, "metadata"))
        if self.outcome == "replay_miss" and self.snapshot_hit:
            raise RoutingExperimentError("replay_miss cannot be marked as a snapshot hit")
        if self.outcome not in {"failed", "retryable_failure", "replay_miss"} and self.failure_code:
            raise RoutingExperimentError("failure_code is only valid for failed outcomes")
        if self.qualified_count and not self.verification_receipt_hash:
            raise RoutingExperimentError(
                "qualified_count requires a verification receipt hash"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "arm_id": self.arm_id,
            "icp_ref": self.icp_ref,
            "step_order": self.step_order,
            "attempt_sequence": self.attempt_sequence,
            "tool_id": self.tool_id,
            "disposition": self.disposition,
            "outcome": self.outcome,
            "route_plan_hash": self.route_plan_hash,
            "stop_policy_hash": self.stop_policy_hash,
            "attempt_receipt_hash": self.attempt_receipt_hash,
            "verification_receipt_hash": self.verification_receipt_hash,
            "provider_id": self.provider_id,
            "provider_call_count": self.provider_call_count,
            "cost_microusd": self.cost_microusd,
            "latency_ms": self.latency_ms,
            "raw_count": self.raw_count,
            "unique_count": self.unique_count,
            "verified_count": self.verified_count,
            "qualified_count": self.qualified_count,
            "published_count": self.published_count,
            "snapshot_hit": bool(self.snapshot_hit),
            "result_hash": self.result_hash,
            "failure_code": self.failure_code,
            "metadata": dict(self.metadata),
        }

    @property
    def attempt_hash(self) -> str:
        return _payload_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "attempt_hash": self.attempt_hash}


@dataclass(frozen=True)
class RoutingEvaluationPolicy:
    """Deterministic replay promotion thresholds."""

    min_runs: int = 5
    max_failed_runs: int = 0
    max_replay_misses: int = 0
    max_cost_regression_ratio: float = 1.10
    schema_version: str = "leadpoet.candidate_routing_evaluation_policy.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_runs", _nonnegative_int(self.min_runs, "min_runs"))
        object.__setattr__(self, "max_failed_runs", _nonnegative_int(self.max_failed_runs, "max_failed_runs"))
        object.__setattr__(self, "max_replay_misses", _nonnegative_int(self.max_replay_misses, "max_replay_misses"))
        try:
            ratio = float(self.max_cost_regression_ratio)
        except (TypeError, ValueError) as exc:
            raise RoutingExperimentError("max_cost_regression_ratio must be positive") from exc
        if ratio <= 0:
            raise RoutingExperimentError("max_cost_regression_ratio must be positive")
        object.__setattr__(self, "max_cost_regression_ratio", ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "min_runs": self.min_runs,
            "max_failed_runs": self.max_failed_runs,
            "max_replay_misses": self.max_replay_misses,
            "max_cost_regression_ratio": self.max_cost_regression_ratio,
        }

    @property
    def policy_hash(self) -> str:
        return _payload_hash(self.to_dict())


@dataclass(frozen=True)
class CandidateRoutingMetric:
    """Aggregated immutable metric for one arm."""

    experiment_id: str
    experiment_hash: str
    arm_id: str
    run_count: int
    completed_run_count: int
    fulfilled_run_count: int
    failed_run_count: int
    replay_miss_run_count: int
    attempt_count: int
    provider_call_count: int
    cost_microusd: int
    latency_ms: int
    raw_count: int
    unique_count: int
    verified_count: int
    qualified_count: int
    published_count: int
    validation_errors: tuple[str, ...] = ()
    schema_version: str = ROUTING_METRIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "experiment_hash", _hash(self.experiment_hash, "experiment_hash"))
        object.__setattr__(self, "arm_id", _identifier(self.arm_id, "arm_id"))
        for field_name in (
            "run_count",
            "completed_run_count",
            "fulfilled_run_count",
            "failed_run_count",
            "replay_miss_run_count",
            "attempt_count",
            "provider_call_count",
            "cost_microusd",
            "latency_ms",
            "raw_count",
            "unique_count",
            "verified_count",
            "qualified_count",
            "published_count",
        ):
            object.__setattr__(self, field_name, _nonnegative_int(getattr(self, field_name), field_name))
        if self.schema_version != ROUTING_METRIC_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported metric schema version")
        if self.fulfilled_run_count > self.completed_run_count:
            raise RoutingExperimentError(
                "fulfilled_run_count cannot exceed completed_run_count"
            )
        if (
            self.completed_run_count
            + self.failed_run_count
            + self.replay_miss_run_count
            != self.run_count
        ):
            raise RoutingExperimentError("routing run counts are inconsistent")
        errors = tuple(sorted({_text(item, "validation_error", max_length=180) for item in self.validation_errors}))
        object.__setattr__(self, "validation_errors", errors)
        if not (self.raw_count >= self.unique_count >= self.verified_count >= self.qualified_count >= self.published_count):
            raise RoutingExperimentError("metric counts must satisfy raw >= unique >= verified >= qualified >= published")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "promotion_scope": "candidate_routing_experiment",
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "arm_id": self.arm_id,
            "run_count": self.run_count,
            "completed_run_count": self.completed_run_count,
            "fulfilled_run_count": self.fulfilled_run_count,
            "failed_run_count": self.failed_run_count,
            "replay_miss_run_count": self.replay_miss_run_count,
            "attempt_count": self.attempt_count,
            "provider_call_count": self.provider_call_count,
            "cost_microusd": self.cost_microusd,
            "latency_ms": self.latency_ms,
            "raw_count": self.raw_count,
            "unique_count": self.unique_count,
            "verified_count": self.verified_count,
            "qualified_count": self.qualified_count,
            "published_count": self.published_count,
            "validation_errors": list(self.validation_errors),
        }

    @property
    def metric_hash(self) -> str:
        return _payload_hash(self.payload())

    @property
    def qualified_per_call(self) -> float:
        return round(self.qualified_count / self.provider_call_count, 9) if self.provider_call_count else 0.0

    @property
    def qualified_per_microusd(self) -> float:
        return round(self.qualified_count / self.cost_microusd, 12) if self.cost_microusd else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "metric_hash": self.metric_hash,
            "qualified_per_call": self.qualified_per_call,
            "qualified_per_microusd": self.qualified_per_microusd,
        }


@dataclass(frozen=True)
class CandidateRoutingPromotionDecision:
    """Routing-only decision; never consumed by official model promotion."""

    experiment_id: str
    experiment_hash: str
    arm_id: str
    metric_hash: str
    control_metric_hash: str
    state: str
    reason_codes: tuple[str, ...] = ()
    policy_hash: str = ""
    schema_version: str = ROUTING_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "experiment_hash", _hash(self.experiment_hash, "experiment_hash"))
        object.__setattr__(self, "arm_id", _identifier(self.arm_id, "arm_id"))
        object.__setattr__(self, "metric_hash", _hash(self.metric_hash, "metric_hash"))
        object.__setattr__(self, "control_metric_hash", _hash(self.control_metric_hash, "control_metric_hash"))
        if self.state not in ROUTING_PROMOTION_STATES:
            raise RoutingExperimentError("unsupported routing promotion state")
        if self.policy_hash:
            object.__setattr__(self, "policy_hash", _hash(self.policy_hash, "policy_hash"))
        if self.schema_version != ROUTING_DECISION_SCHEMA_VERSION:
            raise RoutingExperimentError("unsupported decision schema version")
        object.__setattr__(self, "reason_codes", tuple(sorted({_text(item, "reason_code", max_length=180) for item in self.reason_codes})))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "arm_id": self.arm_id,
            "metric_hash": self.metric_hash,
            "control_metric_hash": self.control_metric_hash,
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "policy_hash": self.policy_hash,
        }

    @property
    def decision_hash(self) -> str:
        return _payload_hash(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "decision_hash": self.decision_hash}


@dataclass(frozen=True)
class CandidateRoutingReplayEvaluation:
    """Complete deterministic output of :func:`evaluate_routing_replay`."""

    experiment_hash: str
    snapshot_manifest_hash: str
    policy_hash: str
    metrics: tuple[CandidateRoutingMetric, ...]
    decisions: tuple[CandidateRoutingPromotionDecision, ...]
    evaluation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.candidate_routing_replay_evaluation.v1",
            "experiment_hash": self.experiment_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "policy_hash": self.policy_hash,
            "metrics": [item.to_dict() for item in self.metrics],
            "decisions": [item.to_dict() for item in self.decisions],
            "evaluation_hash": self.evaluation_hash,
        }


def _metric_from_attempts(
    experiment: CandidateRoutingExperiment,
    arm: CandidateRoutingArm,
    runs: Sequence[CandidateRoutingRun],
    attempts: Sequence[CandidateRoutingAttempt],
) -> CandidateRoutingMetric:
    arm_runs = [run for run in runs if run.arm_id == arm.arm_id]
    arm_run_ids = {run.run_id for run in arm_runs}
    arm_attempts = [attempt for attempt in attempts if attempt.run_id in arm_run_ids]
    qualified_by_run = {
        run_id: sum(
            attempt.qualified_count
            for attempt in arm_attempts
            if attempt.run_id == run_id
        )
        for run_id in arm_run_ids
    }
    errors: list[str] = []
    for attempt in arm_attempts:
        if attempt.experiment_hash != experiment.experiment_hash:
            errors.append("attempt_experiment_hash_mismatch")
        if attempt.arm_id != arm.arm_id:
            errors.append("attempt_arm_mismatch")
    metric = CandidateRoutingMetric(
        experiment_id=experiment.experiment_id,
        experiment_hash=experiment.experiment_hash,
        arm_id=arm.arm_id,
        run_count=len(arm_runs),
        completed_run_count=sum(run.status == "completed" for run in arm_runs),
        fulfilled_run_count=sum(
            run.status == "completed"
            and qualified_by_run.get(run.run_id, 0)
            >= experiment.target_qualified_count
            for run in arm_runs
        ),
        failed_run_count=sum(run.status == "failed" for run in arm_runs),
        replay_miss_run_count=sum(run.status == "replay_miss" for run in arm_runs),
        attempt_count=len(arm_attempts),
        provider_call_count=sum(attempt.provider_call_count for attempt in arm_attempts),
        cost_microusd=sum(attempt.cost_microusd for attempt in arm_attempts),
        latency_ms=sum(attempt.latency_ms for attempt in arm_attempts),
        raw_count=sum(attempt.raw_count for attempt in arm_attempts),
        unique_count=sum(attempt.unique_count for attempt in arm_attempts),
        verified_count=sum(attempt.verified_count for attempt in arm_attempts),
        qualified_count=sum(attempt.qualified_count for attempt in arm_attempts),
        published_count=sum(attempt.published_count for attempt in arm_attempts),
        validation_errors=tuple(errors),
    )
    return metric


def _decision_for_arm(
    experiment: CandidateRoutingExperiment,
    arm: CandidateRoutingArm,
    metric: CandidateRoutingMetric,
    control_metric: CandidateRoutingMetric,
    policy: RoutingEvaluationPolicy,
) -> CandidateRoutingPromotionDecision:
    reasons = list(metric.validation_errors)
    if metric.run_count < policy.min_runs:
        reasons.append("insufficient_replay_sample")
    if metric.failed_run_count > policy.max_failed_runs:
        reasons.append("failed_replay_runs")
    if metric.replay_miss_run_count > policy.max_replay_misses:
        reasons.append("snapshot_replay_miss")
    if metric.provider_call_count > experiment.max_provider_calls:
        reasons.append("provider_call_budget_exceeded")
    if metric.cost_microusd > experiment.max_cost_microusd:
        reasons.append("provider_cost_budget_exceeded")
    if metric.latency_ms > experiment.max_duration_ms * max(metric.run_count, 1):
        reasons.append("duration_budget_exceeded")
    if metric.attempt_count == 0:
        reasons.append("no_replay_attempts")

    hard_failure = any(
        reason
        in {
            "attempt_experiment_hash_mismatch",
            "attempt_arm_mismatch",
            "failed_replay_runs",
            "snapshot_replay_miss",
            "provider_call_budget_exceeded",
            "provider_cost_budget_exceeded",
            "duration_budget_exceeded",
        }
        for reason in reasons
    )
    if hard_failure:
        state = "rejected"
    elif reasons:
        state = "replay_only"
    elif arm.is_control:
        state = "eligible_for_shadow"
        reasons.append("control_arm")
    else:
        control_cost = control_metric.qualified_per_microusd
        candidate_cost = metric.qualified_per_microusd
        non_regressing_yield = metric.qualified_count >= control_metric.qualified_count
        if control_cost:
            non_regressing_yield = candidate_cost >= control_cost
        non_regressing_fulfillment = (
            metric.fulfilled_run_count >= control_metric.fulfilled_run_count
        )
        target_met_for_all_completed_runs = (
            metric.completed_run_count > 0
            and metric.fulfilled_run_count == metric.completed_run_count
        )
        cost_ok = (
            control_metric.cost_microusd == 0
            or metric.cost_microusd <= int(control_metric.cost_microusd * policy.max_cost_regression_ratio)
        )
        strict_improvement = (
            metric.fulfilled_run_count > control_metric.fulfilled_run_count
            or metric.qualified_count > control_metric.qualified_count
            or (
                metric.qualified_count == control_metric.qualified_count
                and metric.cost_microusd < control_metric.cost_microusd
            )
            or (control_metric.cost_microusd > 0 and candidate_cost > control_cost)
        )
        if (
            non_regressing_yield
            and non_regressing_fulfillment
            and target_met_for_all_completed_runs
            and cost_ok
            and strict_improvement
        ):
            state = "eligible_for_canary"
            reasons.extend((
                "replay_quality_gate_passed",
                "non_regressing_verified_qualified_yield",
            ))
        else:
            state = "eligible_for_shadow"
            if not target_met_for_all_completed_runs:
                reasons.append("qualified_target_not_met_for_all_runs")
            reasons.append("replay_quality_not_sufficient_for_canary")
    return CandidateRoutingPromotionDecision(
        experiment_id=experiment.experiment_id,
        experiment_hash=experiment.experiment_hash,
        arm_id=arm.arm_id,
        metric_hash=metric.metric_hash,
        control_metric_hash=control_metric.metric_hash,
        state=state,
        reason_codes=tuple(reasons),
        policy_hash=policy.policy_hash,
    )


def evaluate_routing_replay(
    *,
    experiment: CandidateRoutingExperiment,
    arms: Sequence[CandidateRoutingArm],
    runs: Sequence[CandidateRoutingRun],
    attempts: Sequence[CandidateRoutingAttempt],
    policy: RoutingEvaluationPolicy | None = None,
    snapshot_store: ProviderSnapshotStore | None = None,
) -> CandidateRoutingReplayEvaluation:
    """Evaluate a replay-only routing run from normalized per-step outcomes.

    This function never compiles a route and never contacts a provider.  When a
    ``ProviderSnapshotStore`` is supplied it must be strict replay mode and its
    verified manifest hash must match the experiment.  The caller can persist
    the returned metric and decision documents in the additive routing tables;
    no official score bundle or candidate-promotion row is written here.
    """

    policy = policy or RoutingEvaluationPolicy()
    if experiment.mode != ROUTING_MODE_REPLAY:
        raise RoutingExperimentError("routing replay requires replay mode")
    if snapshot_store is not None:
        if snapshot_store.mode != MODE_REPLAY or snapshot_store.miss_policy != "strict":
            raise RoutingExperimentError("routing replay requires strict replay snapshot store")
        manifest = snapshot_store.load_manifest()
        verification = snapshot_store.verify_manifest(manifest)
        if not verification.get("passed"):
            raise RoutingExperimentError("provider snapshot manifest failed verification")
        if str(verification.get("manifest_hash") or "") != experiment.snapshot_manifest_hash:
            raise RoutingExperimentError("provider snapshot manifest hash differs")

    if not arms:
        raise RoutingExperimentError("at least one routing arm is required")
    if sum(bool(arm.is_control) for arm in arms) != 1:
        raise RoutingExperimentError("exactly one control arm is required")
    arm_by_id: dict[str, CandidateRoutingArm] = {}
    for arm in arms:
        if arm.experiment_id != experiment.experiment_id or arm.experiment_hash != experiment.experiment_hash:
            raise RoutingExperimentError("arm is not bound to experiment")
        if arm.arm_id in arm_by_id:
            raise RoutingExperimentError("duplicate arm_id")
        arm_by_id[arm.arm_id] = arm

    run_by_id: dict[str, CandidateRoutingRun] = {}
    for run in runs:
        if run.experiment_id != experiment.experiment_id or run.experiment_hash != experiment.experiment_hash:
            raise RoutingExperimentError("run is not bound to experiment")
        if run.snapshot_manifest_hash != experiment.snapshot_manifest_hash:
            raise RoutingExperimentError("run snapshot manifest hash differs")
        if run.arm_id not in arm_by_id:
            raise RoutingExperimentError("run references unknown arm")
        if run.run_id in run_by_id:
            raise RoutingExperimentError("duplicate run_id")
        run_by_id[run.run_id] = run

    attempt_ids: set[str] = set()
    seen_step_keys: set[tuple[str, int, int]] = set()
    for attempt in attempts:
        if attempt.experiment_id != experiment.experiment_id or attempt.experiment_hash != experiment.experiment_hash:
            raise RoutingExperimentError("attempt is not bound to experiment")
        run = run_by_id.get(attempt.run_id)
        if run is None:
            raise RoutingExperimentError("attempt references unknown run")
        if attempt.arm_id != run.arm_id or attempt.icp_ref != run.icp_ref:
            raise RoutingExperimentError("attempt does not match its run")
        if attempt.route_plan_hash != run.route_plan_hash:
            raise RoutingExperimentError("attempt route plan does not match its run")
        if attempt.attempt_id in attempt_ids:
            raise RoutingExperimentError("duplicate attempt_id")
        attempt_ids.add(attempt.attempt_id)
        step_key = (attempt.run_id, attempt.step_order, attempt.attempt_sequence)
        if step_key in seen_step_keys:
            raise RoutingExperimentError("duplicate route step attempt")
        seen_step_keys.add(step_key)

    metrics = tuple(
        _metric_from_attempts(experiment, arm, runs, attempts)
        for arm in sorted(arms, key=lambda item: item.arm_id)
    )
    metric_by_arm = {metric.arm_id: metric for metric in metrics}
    control_arm = next(arm for arm in arms if arm.is_control)
    control_metric = metric_by_arm[control_arm.arm_id]
    decisions = tuple(
        _decision_for_arm(experiment, arm, metric_by_arm[arm.arm_id], control_metric, policy)
        for arm in sorted(arms, key=lambda item: item.arm_id)
    )
    evaluation_payload = {
        "schema_version": "leadpoet.candidate_routing_replay_evaluation.v1",
        "experiment_hash": experiment.experiment_hash,
        "snapshot_manifest_hash": experiment.snapshot_manifest_hash,
        "policy_hash": policy.policy_hash,
        "metrics": [metric.to_dict() for metric in metrics],
        "decisions": [decision.to_dict() for decision in decisions],
    }
    return CandidateRoutingReplayEvaluation(
        experiment_hash=experiment.experiment_hash,
        snapshot_manifest_hash=experiment.snapshot_manifest_hash,
        policy_hash=policy.policy_hash,
        metrics=metrics,
        decisions=decisions,
        evaluation_hash=_payload_hash(evaluation_payload),
    )


def candidate_routing_attempt_from_model_receipt(
    *,
    experiment: CandidateRoutingExperiment,
    arm: CandidateRoutingArm,
    run: CandidateRoutingRun,
    attempt_id: str,
    receipt_payload: Mapping[str, Any],
    model_runtime: Any,
    provider_id: str = "",
    snapshot_hit: bool = True,
    published_count: int = 0,
) -> CandidateRoutingAttempt:
    """Validate and project one exact branch-model attempt receipt.

    ``model_runtime`` must come from the verified branch-specific private-model
    runner or its isolated replay environment. The model parser owns the
    receipt schema and digest; Lab only adds replay lineage and its native
    record hash. Lab never imports an unpinned checkout from its host process.
    """

    if (
        arm.experiment_id != experiment.experiment_id
        or arm.experiment_hash != experiment.experiment_hash
        or run.experiment_id != experiment.experiment_id
        or run.experiment_hash != experiment.experiment_hash
        or run.arm_id != arm.arm_id
    ):
        raise RoutingExperimentError(
            "model attempt lineage does not match experiment arm and run"
        )

    try:
        validate_candidate_routing_model_runtime(
            experiment=experiment,
            model_runtime=model_runtime,
        )
        receipt = model_runtime.CandidateStepAttemptReceipt.from_payload(
            receipt_payload
        )
    except RoutingExperimentError:
        raise
    except Exception as exc:
        raise RoutingExperimentError(
            "model candidate attempt receipt is invalid"
        ) from exc
    if receipt.plan_sha256 != run.route_plan_hash:
        raise RoutingExperimentError(
            "model candidate attempt route plan differs from run"
        )
    published = _nonnegative_int(published_count, "published_count")
    if published > receipt.verified_qualified_count:
        raise RoutingExperimentError(
            "published_count cannot exceed verified qualified count"
        )
    outcome = {
        "succeeded": "success",
        "missed": "miss",
        "failed": "failed",
        "deferred": "retryable_failure",
        "skipped": "skipped",
    }.get(receipt.disposition)
    if outcome is None:
        raise RoutingExperimentError("model attempt disposition is unsupported")
    failure_code = (
        receipt.outcome_code
        if outcome in {"failed", "retryable_failure"}
        else ""
    )
    return CandidateRoutingAttempt(
        attempt_id=attempt_id,
        run_id=run.run_id,
        experiment_id=experiment.experiment_id,
        experiment_hash=experiment.experiment_hash,
        arm_id=arm.arm_id,
        icp_ref=run.icp_ref,
        step_order=receipt.step_order,
        attempt_sequence=receipt.attempt,
        tool_id=receipt.tool_id,
        disposition=receipt.disposition,
        outcome=outcome,
        route_plan_hash=receipt.plan_sha256,
        stop_policy_hash=receipt.stop_policy_sha256,
        attempt_receipt_hash=receipt.sha256(),
        verification_receipt_hash=receipt.verification_receipt_sha256,
        provider_id=provider_id or receipt.tool_id,
        provider_call_count=receipt.provider_call_count,
        cost_microusd=round(receipt.estimated_cost_usd * 1_000_000),
        latency_ms=round(receipt.latency_seconds * 1_000),
        raw_count=receipt.raw_candidate_count,
        unique_count=receipt.unique_candidate_count,
        verified_count=receipt.verified_qualified_count,
        qualified_count=receipt.verified_qualified_count,
        published_count=published,
        snapshot_hit=snapshot_hit,
        result_hash=_payload_hash({
            "attempt_receipt_hash": receipt.sha256(),
            "snapshot_hit": snapshot_hit,
        }),
        failure_code=failure_code,
    )


def validate_candidate_routing_model_runtime(
    *,
    experiment: CandidateRoutingExperiment,
    model_runtime: Any,
) -> Mapping[str, Any]:
    """Fail closed unless one branch runtime has the complete router surface."""

    required_callables = (
        "candidate_waterfall_execution_contract_identity",
        "compile_candidate_stop_policy",
        "compile_profiled_candidate_acquisition_route",
        "evaluate_candidate_waterfall",
        "runtime_catalog",
        "runtime_policy",
        "runtime_tool_definitions",
    )
    if any(
        not callable(getattr(model_runtime, name, None))
        for name in required_callables
    ):
        raise RoutingExperimentError(
            "model candidate routing runtime contract is incomplete"
        )
    receipt_type = getattr(model_runtime, "CandidateStepAttemptReceipt", None)
    if not callable(getattr(receipt_type, "from_payload", None)):
        raise RoutingExperimentError(
            "model candidate routing receipt parser is unavailable"
        )
    try:
        identity = (
            model_runtime.candidate_waterfall_execution_contract_identity()
        )
    except Exception as exc:
        raise RoutingExperimentError(
            "model candidate routing identity is unavailable"
        ) from exc
    if not isinstance(identity, Mapping):
        raise RoutingExperimentError(
            "model candidate routing identity is invalid"
        )
    if identity.get("contract_sha256") != experiment.routing_contract_hash:
        raise RoutingExperimentError(
            "model routing contract hash differs from experiment"
        )
    if identity.get("provider_results_can_satisfy_target") is not False:
        raise RoutingExperimentError(
            "model candidate routing stop contract is unsafe"
        )
    return dict(identity)


__all__ = [
    "ROUTING_ATTEMPT_DISPOSITIONS",
    "ROUTING_ATTEMPT_OUTCOMES",
    "ROUTING_EVALUATOR_VERSION",
    "ROUTING_PROMOTION_STATES",
    "ROUTING_RUN_STATUSES",
    "CandidateRoutingArm",
    "CandidateRoutingAttempt",
    "CandidateRoutingExperiment",
    "CandidateRoutingMetric",
    "CandidateRoutingPromotionDecision",
    "CandidateRoutingReplayEvaluation",
    "CandidateRoutingRun",
    "RoutingEvaluationPolicy",
    "RoutingExperimentError",
    "candidate_routing_attempt_from_model_receipt",
    "evaluate_routing_replay",
    "validate_candidate_routing_model_runtime",
]
