"""Phase 3 foundation contracts for the Leadpoet Research Lab.

Phase 3 should build the local data pipelines now, but must not claim judge
calibration, fine-tune success, distillation parity, or Workspace API readiness
until measured lab/production evidence exists. This module encodes that split:
local contracts are allowed; production/data-success claims fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .loop_foundation import TRUTHY_ENV_VALUES
from .market_exit_gate import (
    PROTECTED_EXIT_GATE_KEYS,
    PROTECTED_EXIT_GATE_MARKERS,
    verify_market_exit_gate,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "model_pipeline_foundation_fixtures.json"


class ModelPipelineDataGate(str, Enum):
    MARKET_EXIT_GATE_READY = "market_exit_gate_ready"
    TRAJECTORY_CORPUS_READY = "trajectory_corpus_ready"
    OUTCOME_LABEL_VOLUME_READY = "outcome_label_volume_ready"
    COST_DATA_READY = "cost_data_ready"
    COMPONENT_OUTCOMES_READY = "component_outcomes_ready"
    CROWN_OUTCOMES_READY = "crown_outcomes_ready"
    DISTILLATION_DATA_READY = "distillation_data_ready"
    WORKSPACE_API_SECURITY_READY = "workspace_api_security_ready"


class ModelPipelineReadinessState(str, Enum):
    LOCAL_CONTRACT_READY = "local_contract_ready"
    AWAITING_PRODUCTION_DATA = "awaiting_production_data"
    READY_AFTER_MEASURED_EVIDENCE = "ready_after_measured_evidence"
    BLOCKED = "blocked"


MODEL_PIPELINE_DATA_GATES: tuple[str, ...] = tuple(gate.value for gate in ModelPipelineDataGate)

PROTECTED_MODEL_PIPELINE_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_EXIT_GATE_KEYS)
    | {
        "api_key",
        "customer_email",
        "distillation_training_row",
        "judge_prompt",
        "live_champion_weights",
        "model_weights",
        "private_customer_data",
        "raw_outcome_label",
        "raw_training_example",
        "sealed_eval_details",
        "training_secret",
    }
)

PROTECTED_MODEL_PIPELINE_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_EXIT_GATE_MARKERS)
        | {
            "api key",
            "customer email",
            "judge prompt",
            "live champion",
            "model weights",
            "private customer",
            "raw outcome",
            "raw training",
            "sealed eval",
            "training secret",
        }
    )
)


@dataclass(frozen=True)
class ModelPipelineWorkflowGuards:
    judge_calibration_publication: bool = False
    engine_finetune_training: bool = False
    engine_v_next_promotion: bool = False
    reranker_distillation_training: bool = False
    workspace_api_beta: bool = False
    public_agent_track: bool = False
    model_weight_publication: bool = False
    calibration_success_claims: bool = False
    fine_tune_success_claims: bool = False
    distillation_success_claims: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPipelineWorkflowGuards":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPipelineDataReadinessStatus:
    market_exit_gate_ready: bool = False
    trajectory_corpus_ready: bool = False
    outcome_label_volume_ready: bool = False
    cost_data_ready: bool = False
    component_outcomes_ready: bool = False
    crown_outcomes_ready: bool = False
    distillation_data_ready: bool = False
    workspace_api_security_ready: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPipelineDataReadinessStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPipelineBuildStatus:
    trajectory_corpus_builder: bool = False
    run_log_summarizer: bool = False
    cost_ledger_summarizer: bool = False
    fine_tune_dataset_builder: bool = False
    matched_budget_ab_evaluator: bool = False
    reranker_distillation_dataset_builder: bool = False
    workspace_api_beta_contracts: bool = False
    model_pipeline_exit_gate_verifier: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPipelineBuildStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPipelineReadinessClaimRecord:
    readiness_id: str
    data_readiness: ModelPipelineDataReadinessStatus
    pipeline_build_status: ModelPipelineBuildStatus
    local_code_complete: bool = False
    model_pipeline_operation_claimed_ready: bool = False
    judge_calibration_v1_claimed_valid: bool = False
    engine_v_next_yield_claimed: bool = False
    reranker_distillation_parity_claimed: bool = False
    workspace_api_beta_enablement_claimed: bool = False
    uses_local_fixtures: bool = True
    uses_synthetic_labels: bool = False
    local_only: bool = True
    market_exit_gate_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""
    state: str = ModelPipelineReadinessState.AWAITING_PRODUCTION_DATA.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPipelineReadinessClaimRecord":
        return cls(
            readiness_id=str(data["readiness_id"]),
            data_readiness=ModelPipelineDataReadinessStatus.from_mapping(data.get("data_readiness", {})),
            pipeline_build_status=ModelPipelineBuildStatus.from_mapping(data.get("pipeline_build_status", {})),
            local_code_complete=bool(data.get("local_code_complete", False)),
            model_pipeline_operation_claimed_ready=bool(data.get("model_pipeline_operation_claimed_ready", False)),
            judge_calibration_v1_claimed_valid=bool(data.get("judge_calibration_v1_claimed_valid", False)),
            engine_v_next_yield_claimed=bool(data.get("engine_v_next_yield_claimed", False)),
            reranker_distillation_parity_claimed=bool(data.get("reranker_distillation_parity_claimed", False)),
            workspace_api_beta_enablement_claimed=bool(data.get("workspace_api_beta_enablement_claimed", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            uses_synthetic_labels=bool(data.get("uses_synthetic_labels", False)),
            local_only=bool(data.get("local_only", True)),
            market_exit_gate_ref=str(data.get("market_exit_gate_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            state=str(data.get("state", ModelPipelineReadinessState.AWAITING_PRODUCTION_DATA.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_readiness"] = self.data_readiness.to_dict()
        data["pipeline_build_status"] = self.pipeline_build_status.to_dict()
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def default_model_pipeline_workflow_guards() -> ModelPipelineWorkflowGuards:
    return ModelPipelineWorkflowGuards()


def model_pipeline_workflow_guards_from_env(env: Optional[Mapping[str, str]] = None) -> ModelPipelineWorkflowGuards:
    env = env or {}
    return ModelPipelineWorkflowGuards(
        judge_calibration_publication=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_JUDGE_CALIBRATION_PUBLICATION")),
        engine_finetune_training=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_ENGINE_FINETUNE_TRAINING")),
        engine_v_next_promotion=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_ENGINE_V_NEXT_PROMOTION")),
        reranker_distillation_training=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_RERANKER_DISTILLATION_TRAINING")),
        workspace_api_beta=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_WORKSPACE_API_BETA")),
        public_agent_track=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_PUBLIC_AGENT_TRACK")),
        model_weight_publication=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_MODEL_WEIGHT_PUBLICATION")),
        calibration_success_claims=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_CALIBRATION_SUCCESS_CLAIMS")),
        fine_tune_success_claims=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_FINE_TUNE_SUCCESS_CLAIMS")),
        distillation_success_claims=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_DISTILLATION_SUCCESS_CLAIMS")),
        production_writes=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_PRODUCTION_WRITES")),
        supabase_writes=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_SUPABASE_WRITES")),
        public_workflows=_truthy(env.get("RESEARCH_LAB_MODEL_PIPELINE_PUBLIC_WORKFLOWS")),
    )


def default_model_pipeline_data_readiness_status() -> ModelPipelineDataReadinessStatus:
    return ModelPipelineDataReadinessStatus()


def validate_model_pipeline_workflow_guards(guards: ModelPipelineWorkflowGuards | Mapping[str, Any]) -> list[str]:
    if not isinstance(guards, ModelPipelineWorkflowGuards):
        guards = ModelPipelineWorkflowGuards.from_mapping(guards)
    enabled = [name for name, value in guards.to_dict().items() if value]
    if not enabled:
        return []
    return ["Phase 3 local build must keep these workflows disabled: " + ", ".join(enabled)]


def assert_model_pipeline_workflows_disabled(guards: ModelPipelineWorkflowGuards | Mapping[str, Any]) -> None:
    errors = validate_model_pipeline_workflow_guards(guards)
    if errors:
        raise ValueError("; ".join(errors))


def model_pipeline_data_gate_ready(status: ModelPipelineDataReadinessStatus | Mapping[str, Any], gate: ModelPipelineDataGate | str) -> bool:
    if not isinstance(status, ModelPipelineDataReadinessStatus):
        status = ModelPipelineDataReadinessStatus.from_mapping(status)
    gate_value = gate.value if isinstance(gate, ModelPipelineDataGate) else str(gate)
    if gate_value not in ModelPipelineDataGate._value2member_map_:
        raise ValueError(f"unknown Phase 3 data gate: {gate_value}")
    return bool(getattr(status, gate_value))


def require_model_pipeline_data_gate(status: ModelPipelineDataReadinessStatus | Mapping[str, Any], gate: ModelPipelineDataGate | str) -> None:
    if not model_pipeline_data_gate_ready(status, gate):
        gate_value = gate.value if isinstance(gate, ModelPipelineDataGate) else str(gate)
        raise ValueError(f"Phase 3 data gate is not ready: {gate_value}")


def model_pipeline_missing_data_gates(status: ModelPipelineDataReadinessStatus | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(status, ModelPipelineDataReadinessStatus):
        status = ModelPipelineDataReadinessStatus.from_mapping(status)
    return tuple(gate for gate in MODEL_PIPELINE_DATA_GATES if not model_pipeline_data_gate_ready(status, gate))


def model_pipeline_build_complete(status: ModelPipelineBuildStatus | Mapping[str, Any]) -> bool:
    if not isinstance(status, ModelPipelineBuildStatus):
        status = ModelPipelineBuildStatus.from_mapping(status)
    return all(status.to_dict().values())


def validate_model_pipeline_readiness_claim(
    record: ModelPipelineReadinessClaimRecord | Mapping[str, Any],
    *,
    guards: Optional[ModelPipelineWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_model_pipeline_payload_errors(raw)
    if not isinstance(record, ModelPipelineReadinessClaimRecord):
        record = ModelPipelineReadinessClaimRecord.from_mapping(record)
    try:
        assert_model_pipeline_workflows_disabled(guards or default_model_pipeline_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))

    if record.state not in {state.value for state in ModelPipelineReadinessState}:
        errors.append(f"unknown Phase 3 readiness state: {record.state}")
        return errors
    if not record.readiness_id.startswith("model_pipeline_readiness:"):
        errors.append("readiness_id must be model_pipeline_readiness:-prefixed")
    if record.market_exit_gate_ref and not record.market_exit_gate_ref.startswith("market_exit_gate:"):
        errors.append("market_exit_gate_ref must be market_exit_gate:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "market_exit_gate:",
                "trajectory_corpus:",
                "run_log_summary:",
                "cost_ledger:",
                "outcome_label_set:",
                "component_outcome:",
                "crown_outcome:",
                "judge_calibration:",
                "engine_ab:",
                "fine_tune_dataset:",
                "distillation_dataset:",
                "distillation_eval:",
                "workspace_api_beta:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved Phase 3 prefixes")
            break

    missing = model_pipeline_missing_data_gates(record.data_readiness)
    if record.model_pipeline_operation_claimed_ready:
        if not model_pipeline_build_complete(record.pipeline_build_status):
            errors.append("Phase 3 readiness requires all Phase 3 pipeline build fields true")
        if record.uses_local_fixtures:
            errors.append("Phase 3 readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 3 readiness cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 3 readiness claim is missing data gates: " + ", ".join(missing))
        if not record.evidence_refs:
            errors.append("Phase 3 readiness claim requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Phase 3 readiness claim requires owner_approval_ref")
        if record.state != ModelPipelineReadinessState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("model_pipeline_operation_claimed_ready requires ready_after_measured_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready Phase 3 records must remain local_only")
        if record.state == ModelPipelineReadinessState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("ready_after_measured_evidence state requires model_pipeline_operation_claimed_ready")

    _validate_success_claims(record, errors)
    return errors


def verify_model_pipeline_foundation(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_exit_gate()
    fixture = _load_fixture(Path(fixture_path))

    _assert(
        fixture["model_pipeline_data_gate_values"] == list(MODEL_PIPELINE_DATA_GATES),
        "fixture Phase 3 data gates match code",
    )

    disabled_guards = ModelPipelineWorkflowGuards.from_mapping(fixture["workflow_guards"]["disabled"])
    _assert(not validate_model_pipeline_workflow_guards(disabled_guards), "disabled Phase 3 workflow guards validate")

    invalid_guard_errors = validate_model_pipeline_workflow_guards(fixture["workflow_guards"]["invalid"])
    _assert(invalid_guard_errors, "enabled Phase 3 workflow guards fail closed")
    _assert_expected_error(invalid_guard_errors, fixture["workflow_guards"]["invalid"])

    env_guards = model_pipeline_workflow_guards_from_env(fixture["workflow_guard_env"])
    env_errors = validate_model_pipeline_workflow_guards(env_guards)
    _assert(env_errors, "truthy Phase 3 env flags fail closed")
    _assert_expected_error(env_errors, fixture["workflow_guard_env"])

    data_status = ModelPipelineDataReadinessStatus.from_mapping(fixture["data_readiness_status"])
    for gate in MODEL_PIPELINE_DATA_GATES:
        _assert(not model_pipeline_data_gate_ready(data_status, gate), f"local Phase 3 data gate defaults false: {gate}")
        try:
            require_model_pipeline_data_gate(data_status, gate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing Phase 3 data gate raises: {gate}")

    local_claim = ModelPipelineReadinessClaimRecord.from_mapping(fixture["readiness_claims"]["local_foundation_ready"])
    _assert(not validate_model_pipeline_readiness_claim(local_claim), "local Phase 3 foundation-ready claim validates")
    _assert(local_claim.local_code_complete, "P3.0 fixture marks local code complete")
    _assert(not local_claim.model_pipeline_operation_claimed_ready, "P3.0 fixture does not claim operation ready")
    _assert(not model_pipeline_build_complete(local_claim.pipeline_build_status), "P3.0 does not claim later pipelines built")
    _assert(model_pipeline_missing_data_gates(local_claim.data_readiness), "P3.0 fixture reports missing data gates")
    _assert(
        local_claim.market_exit_gate_ref == market_summary["exit_gate_id"],
        "P3.0 fixture pins the P2.10 exit-gate ref",
    )
    _assert(not market_summary["production_operational_ready"], "P2.10 remains not operationally ready")

    for record in fixture["readiness_claims"]["invalid"]:
        base = fixture["readiness_claims"][str(record.get("base", "local_foundation_ready"))]
        merged = _deep_merge(dict(base), dict(record.get("overrides", {})))
        errors = validate_model_pipeline_readiness_claim(merged)
        _assert(errors, f"invalid Phase 3 readiness claim fails: {record['id']}")
        _assert_expected_error(errors, record)

    unsafe_errors = validate_model_pipeline_readiness_claim(
        local_claim,
        guards=fixture["workflow_guards"]["invalid"],
    )
    _assert(unsafe_errors, "unsafe Phase 3 guards block readiness verifier")
    _assert_expected_error(unsafe_errors, fixture["workflow_guards"]["invalid"])

    return {
        "market_exit_gate_id": market_summary["exit_gate_id"],
        "market_production_operational_ready": market_summary["production_operational_ready"],
        "data_gates": len(MODEL_PIPELINE_DATA_GATES),
        "workflow_guard_fields": len(ModelPipelineWorkflowGuards.__dataclass_fields__),
        "local_code_complete": local_claim.local_code_complete,
        "model_pipeline_operation_claimed_ready": local_claim.model_pipeline_operation_claimed_ready,
        "missing_data_gates": list(model_pipeline_missing_data_gates(local_claim.data_readiness)),
        "readiness_id": local_claim.readiness_id,
    }


def _validate_success_claims(record: ModelPipelineReadinessClaimRecord, errors: list[str]) -> None:
    data = record.data_readiness
    if record.judge_calibration_v1_claimed_valid:
        if record.uses_synthetic_labels:
            errors.append("judge calibration validity cannot be claimed from synthetic labels")
        if not data.outcome_label_volume_ready:
            errors.append("judge calibration validity requires outcome_label_volume_ready")
        if not any(ref.startswith("judge_calibration:") for ref in record.evidence_refs):
            errors.append("judge calibration validity requires judge_calibration evidence")
    if record.engine_v_next_yield_claimed:
        if record.uses_local_fixtures:
            errors.append("engine v-next yield claims cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("engine v-next yield claims cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("engine v-next yield claims require owner_approval_ref")
        if not (data.market_exit_gate_ready and data.trajectory_corpus_ready and data.cost_data_ready):
            errors.append("engine v-next yield claims require Phase 2 exit, trajectory corpus, and cost data")
        if not any(ref.startswith("engine_ab:") for ref in record.evidence_refs):
            errors.append("engine v-next yield claims require engine_ab evidence")
    if record.reranker_distillation_parity_claimed:
        if record.uses_local_fixtures:
            errors.append("reranker parity claims cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("reranker parity claims cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("reranker parity claims require owner_approval_ref")
        if not (data.distillation_data_ready and data.cost_data_ready):
            errors.append("reranker parity claims require distillation data and cost data")
        if not any(ref.startswith("distillation_eval:") for ref in record.evidence_refs):
            errors.append("reranker parity claims require distillation_eval evidence")
    if record.workspace_api_beta_enablement_claimed:
        if not data.workspace_api_security_ready:
            errors.append("Workspace API beta enablement requires workspace_api_security_ready")
        if record.local_only:
            errors.append("Workspace API beta enablement cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("Workspace API beta enablement requires owner_approval_ref")
        if not any(ref.startswith("workspace_api_beta:") for ref in record.evidence_refs):
            errors.append("Workspace API beta enablement requires workspace_api_beta evidence")


def _protected_model_pipeline_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_model_pipeline_material(record))
    if not found:
        return []
    return ["Phase 3 readiness payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_model_pipeline_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_MODEL_PIPELINE_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_model_pipeline_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_model_pipeline_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_MODEL_PIPELINE_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in TRUTHY_ENV_VALUES


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if not expected:
        return
    expected_values = [str(item) for item in expected] if isinstance(expected, list) else [str(expected)]
    for expected_value in expected_values:
        _assert(any(expected_value in error for error in errors), f"expected error {expected_value!r}")


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
