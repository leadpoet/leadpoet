"""Phase 1 foundation contracts for the Leadpoet Research Lab.

These helpers are intentionally local-only. They define records and guards that
later Phase 1 slices can consume, but they do not write to production services,
schedule jobs, open miner workflows, or publish live champion artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "loop_foundation_fixtures.json"


class VisibilityPolicy(str, Enum):
    DEFAULT_PRIVATE = "default_private"
    PUBLIC_RECEIPT = "public_receipt"
    SANITIZED_TRACE = "sanitized_trace"
    DELAYED_RELEASE = "delayed_release"
    RETIRED_PUBLIC_ARTIFACT = "retired_public_artifact"


class ArtifactReleaseState(str, Enum):
    PRIVATE_LIVE_CHAMPION = "private_live_champion"
    PUBLIC_RECEIPT = "public_receipt"
    SANITIZED_TRACE = "sanitized_trace"
    DELAYED_RELEASE = "delayed_release"
    RETIRED_PUBLIC_ARTIFACT = "retired_public_artifact"


class BaselineDependencyGate(str, Enum):
    PRODUCTION_SQL_APPLIED = "production_sql_applied"
    PAIRED_DAY_CLOCK_STARTED = "paired_day_clock_started"
    SAME_DAY_CROWNING_CERTIFIED = "same_day_crowning_certified"
    MEASURED_EXTERNAL_COST_CALIBRATED = "measured_external_cost_calibrated"


ARTIFACT_RELEASE_STATES: tuple[str, ...] = tuple(state.value for state in ArtifactReleaseState)
VISIBILITY_POLICIES: tuple[str, ...] = tuple(policy.value for policy in VisibilityPolicy)

PUBLIC_RELEASE_STATES = {
    ArtifactReleaseState.PUBLIC_RECEIPT.value,
    ArtifactReleaseState.SANITIZED_TRACE.value,
    ArtifactReleaseState.DELAYED_RELEASE.value,
    ArtifactReleaseState.RETIRED_PUBLIC_ARTIFACT.value,
}

STATE_TO_POLICY = {
    ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value: VisibilityPolicy.DEFAULT_PRIVATE.value,
    ArtifactReleaseState.PUBLIC_RECEIPT.value: VisibilityPolicy.PUBLIC_RECEIPT.value,
    ArtifactReleaseState.SANITIZED_TRACE.value: VisibilityPolicy.SANITIZED_TRACE.value,
    ArtifactReleaseState.DELAYED_RELEASE.value: VisibilityPolicy.DELAYED_RELEASE.value,
    ArtifactReleaseState.RETIRED_PUBLIC_ARTIFACT.value: VisibilityPolicy.RETIRED_PUBLIC_ARTIFACT.value,
}

PROTECTED_MATERIAL_FLAGS = (
    "contains_live_champion_ip",
    "contains_sealed_eval_details",
    "contains_raw_evidence_snapshot",
    "contains_private_customer_data",
    "contains_judge_prompts",
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ReleasePolicyRecord:
    artifact_ref: str
    artifact_type: str
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    release_after: Optional[str] = None
    release_condition: str = ""
    reason: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReleasePolicyRecord":
        return cls(
            artifact_ref=str(data["artifact_ref"]),
            artifact_type=str(data["artifact_type"]),
            visibility_policy=str(data.get("visibility_policy", cls.visibility_policy)),
            artifact_release_state=str(data.get("artifact_release_state", cls.artifact_release_state)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            release_after=data.get("release_after"),
            release_condition=str(data.get("release_condition", "")),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopWorkflowGuards:
    public_miner_workflows: bool = False
    paid_loops: bool = False
    autopilot: bool = False
    hosted_run_crowning: bool = False
    source_add_payments: bool = False
    scheduler_jobs: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    live_champion_publication: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopWorkflowGuards":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineCarryForwardStatus:
    production_sql_applied: bool = False
    paired_day_clock_started: bool = False
    same_day_crowning_certified: bool = False
    measured_external_cost_calibrated: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineCarryForwardStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def default_release_policy(
    artifact_ref: str,
    artifact_type: str,
    *,
    reason: str = "loop default private release policy",
) -> ReleasePolicyRecord:
    return ReleasePolicyRecord(
        artifact_ref=artifact_ref,
        artifact_type=artifact_type,
        reason=reason,
    )


def default_loop_workflow_guards() -> LoopWorkflowGuards:
    return LoopWorkflowGuards()


def loop_workflow_guards_from_env(env: Optional[Mapping[str, str]] = None) -> LoopWorkflowGuards:
    env = env or {}
    return LoopWorkflowGuards(
        public_miner_workflows=_truthy(env.get("RESEARCH_LAB_LOOP_PUBLIC_WORKFLOWS")),
        paid_loops=_truthy(env.get("RESEARCH_LAB_LOOP_PAID_LOOPS")),
        autopilot=_truthy(env.get("RESEARCH_LAB_LOOP_AUTOPILOT")),
        hosted_run_crowning=_truthy(env.get("RESEARCH_LAB_LOOP_HOSTED_RUN_CROWNING")),
        source_add_payments=_truthy(env.get("RESEARCH_LAB_LOOP_SOURCE_ADD_PAYMENTS")),
        scheduler_jobs=_truthy(env.get("RESEARCH_LAB_LOOP_SCHEDULER_JOBS")),
        production_writes=_truthy(env.get("RESEARCH_LAB_LOOP_PRODUCTION_WRITES")),
        supabase_writes=_truthy(env.get("RESEARCH_LAB_LOOP_SUPABASE_WRITES")),
        live_champion_publication=_truthy(env.get("RESEARCH_LAB_LOOP_LIVE_CHAMPION_PUBLICATION")),
    )


def default_baseline_carry_forward_status() -> BaselineCarryForwardStatus:
    return BaselineCarryForwardStatus()


def validate_release_policy(record: ReleasePolicyRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, ReleasePolicyRecord):
        record = ReleasePolicyRecord.from_mapping(record)

    errors: list[str] = []
    state = record.artifact_release_state
    policy = record.visibility_policy

    if state not in ARTIFACT_RELEASE_STATES:
        errors.append(f"unknown artifact_release_state: {state}")
    if policy not in VISIBILITY_POLICIES:
        errors.append(f"unknown visibility_policy: {policy}")
    if errors:
        return errors

    expected_policy = STATE_TO_POLICY[state]
    if policy != expected_policy:
        errors.append(
            f"visibility_policy {policy!r} does not match artifact_release_state {state!r}; "
            f"expected {expected_policy!r}"
        )

    if state in PUBLIC_RELEASE_STATES:
        protected = [flag for flag in PROTECTED_MATERIAL_FLAGS if bool(getattr(record, flag))]
        if protected:
            errors.append(
                f"public release state {state!r} cannot contain protected material flags: "
                + ", ".join(protected)
            )

    if state == ArtifactReleaseState.PUBLIC_RECEIPT.value and record.artifact_type != "receipt":
        errors.append("public_receipt state requires artifact_type='receipt'")

    if state == ArtifactReleaseState.SANITIZED_TRACE.value and record.artifact_type not in {
        "sanitized_trace",
        "trajectory_summary",
        "lesson_summary",
        "map_projection",
    }:
        errors.append("sanitized_trace state requires a sanitized trace, lesson, or map projection artifact type")

    if state == ArtifactReleaseState.DELAYED_RELEASE.value and not (
        record.release_after or record.release_condition
    ):
        errors.append("delayed_release requires release_after or release_condition")

    if state == ArtifactReleaseState.RETIRED_PUBLIC_ARTIFACT.value and record.artifact_type not in {
        "retired_artifact",
        "retired_model",
        "retired_component",
    }:
        errors.append("retired_public_artifact state requires a retired artifact type")

    return errors


def assert_release_policy(record: ReleasePolicyRecord | Mapping[str, Any]) -> None:
    errors = validate_release_policy(record)
    if errors:
        raise ValueError("; ".join(errors))


def validate_loop_workflow_guards(guards: LoopWorkflowGuards | Mapping[str, Any]) -> list[str]:
    if not isinstance(guards, LoopWorkflowGuards):
        guards = LoopWorkflowGuards.from_mapping(guards)
    enabled = [name for name, value in guards.to_dict().items() if value]
    if not enabled:
        return []
    return ["Phase 1 local build must keep these workflows disabled: " + ", ".join(enabled)]


def assert_loop_workflows_disabled(guards: LoopWorkflowGuards | Mapping[str, Any]) -> None:
    errors = validate_loop_workflow_guards(guards)
    if errors:
        raise ValueError("; ".join(errors))


def baseline_gate_ready(status: BaselineCarryForwardStatus | Mapping[str, Any], gate: BaselineDependencyGate | str) -> bool:
    if not isinstance(status, BaselineCarryForwardStatus):
        status = BaselineCarryForwardStatus.from_mapping(status)
    gate_value = gate.value if isinstance(gate, BaselineDependencyGate) else str(gate)
    if gate_value not in BaselineDependencyGate._value2member_map_:
        raise ValueError(f"unknown Phase 0 dependency gate: {gate_value}")
    return bool(getattr(status, gate_value))


def require_baseline_gate(status: BaselineCarryForwardStatus | Mapping[str, Any], gate: BaselineDependencyGate | str) -> None:
    if not baseline_gate_ready(status, gate):
        gate_value = gate.value if isinstance(gate, BaselineDependencyGate) else str(gate)
        raise ValueError(f"Phase 0 dependency gate is not ready: {gate_value}")


def verify_loop_foundation(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _load_fixture(Path(fixture_path))

    _assert(
        fixture["artifact_release_state_values"] == list(ARTIFACT_RELEASE_STATES),
        "fixture release-state enum matches code",
    )
    _assert(
        fixture["visibility_policy_values"] == list(VISIBILITY_POLICIES),
        "fixture visibility-policy enum matches code",
    )

    valid_release_records = fixture["valid_release_records"]
    invalid_release_records = fixture["invalid_release_records"]

    default_policy = default_release_policy("artifact:default", "champion")
    _assert(
        default_policy.artifact_release_state == ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value,
        "default release state is private_live_champion",
    )
    _assert(not validate_release_policy(default_policy), "default release policy validates")

    for record in valid_release_records:
        _assert(not validate_release_policy(record), f"valid release record passes: {record['artifact_ref']}")

    for record in invalid_release_records:
        _assert(validate_release_policy(record), f"invalid release record fails: {record['artifact_ref']}")

    guards = LoopWorkflowGuards.from_mapping(fixture["workflow_guards"]["disabled"])
    _assert(not validate_loop_workflow_guards(guards), "disabled workflow guard state passes")
    invalid_guards = LoopWorkflowGuards.from_mapping(fixture["workflow_guards"]["invalid"])
    _assert(validate_loop_workflow_guards(invalid_guards), "enabled workflow guard state fails")

    env_guards = loop_workflow_guards_from_env({})
    _assert(env_guards == default_loop_workflow_guards(), "empty env keeps Phase 1 disabled")
    enabled_env = loop_workflow_guards_from_env({"RESEARCH_LAB_LOOP_PAID_LOOPS": "1"})
    _assert(validate_loop_workflow_guards(enabled_env), "truthy env flag is detected as unsafe")

    baseline_status = BaselineCarryForwardStatus.from_mapping(fixture["baseline_carry_forward_status"])
    for gate in BaselineDependencyGate:
        _assert(not baseline_gate_ready(baseline_status, gate), f"{gate.value} defaults to not ready")
        try:
            require_baseline_gate(baseline_status, gate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{gate.value} should fail closed when not ready")

    return {
        "valid_release_records": len(valid_release_records),
        "invalid_release_records": len(invalid_release_records),
        "workflow_guard_fields": len(default_loop_workflow_guards().to_dict()),
        "baseline_dependency_gates": len(list(BaselineDependencyGate)),
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in TRUTHY_ENV_VALUES


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
