"""Phase 3 exit-gate verifier contracts.

P3.7 summarizes Phase 3 completion without enabling it. The local verifier
distinguishes code completeness from production readiness and remains
fail-closed until measured inputs exist for Engine v-next yield, reranker
distillation parity/cost, judge calibration label volume, and Workspace API
beta security.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .engine_ab_evaluator import (
    ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT,
    EngineABComparisonRecord,
    EngineABComparisonState,
    verify_research_lab_engine_ab_evaluator,
)
from .model_pipeline_foundation import (
    PROTECTED_MODEL_PIPELINE_KEYS,
    PROTECTED_MODEL_PIPELINE_MARKERS,
    ModelPipelineWorkflowGuards,
    assert_model_pipeline_workflows_disabled,
    default_model_pipeline_workflow_guards,
    verify_model_pipeline_foundation,
)
from .reranker_distillation import (
    RERANKER_COST_RATIO_MAX_PCT,
    RERANKER_PARITY_THRESHOLD_PCT,
    RerankerParityComparisonRecord,
    RerankerParityState,
    verify_research_lab_reranker_distillation,
)
from .workspace_api_beta import (
    PENDING_WORKSPACE_BETA_REF,
    WorkspaceAPIBetaEnablementRecord,
    WorkspaceAPIBetaState,
    verify_research_lab_workspace_api_beta,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "model_pipeline_exit_gate_fixtures.json"

MODEL_PIPELINE_EXIT_GATE_CONTRACT_VERSION = "model_pipeline_exit_gate:v1:local_contract"
JUDGE_CALIBRATION_CONTRACT_VERSION = "judge_calibration:v1:local_contract"
JUDGE_CALIBRATION_AUC_THRESHOLD = 0.65

PROTECTED_MODEL_PIPELINE_EXIT_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_MODEL_PIPELINE_KEYS)
    | {
        "calibration_label_payload",
        "customer_outcome",
        "heldout_label_payload",
        "judge_prompt",
        "judge_rubric",
        "live_champion_artifact",
        "model_weights",
        "private_customer_data",
        "raw_engine_output",
        "raw_judge_label",
        "raw_outcome_label",
        "raw_reranker_label",
        "sealed_eval_details",
        "workspace_secret",
    }
)

PROTECTED_MODEL_PIPELINE_EXIT_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_MODEL_PIPELINE_MARKERS)
        | {
            "customer outcome",
            "heldout label",
            "judge prompt",
            "judge rubric",
            "live champion",
            "model weights",
            "private customer",
            "raw engine output",
            "raw judge label",
            "raw outcome",
            "raw reranker label",
            "sealed eval",
            "workspace secret",
        }
    )
)


class ModelPipelineExitGateCheck(str, Enum):
    MODEL_PIPELINE_FOUNDATION_READY = "model_pipeline_foundation_ready"
    ENGINE_VNEXT_YIELD_GATE = "engine_vnext_yield_gate"
    RERANKER_PARITY_COST_GATE = "reranker_parity_cost_gate"
    JUDGE_CALIBRATION_GATE = "judge_calibration_gate"
    WORKSPACE_API_BETA_GATE = "workspace_api_beta_gate"


class ModelPipelineExitGateState(str, Enum):
    LOCAL_NOT_READY = "local_not_ready"
    READY_AFTER_MEASURED_EVIDENCE = "ready_after_measured_evidence"
    BLOCKED = "blocked"


class JudgeCalibrationState(str, Enum):
    LOCAL_CALIBRATION_STUB = "local_calibration_stub"
    PASSED_AFTER_LABEL_VOLUME = "passed_after_label_volume"
    BLOCKED = "blocked"


MODEL_PIPELINE_EXIT_GATE_CHECKS: tuple[str, ...] = tuple(check.value for check in ModelPipelineExitGateCheck)


@dataclass(frozen=True)
class JudgeCalibrationEvidenceRecord:
    calibration_id: str
    label_set_ref: str
    feasibility_memo_ref: str
    methodology_ref: str
    calibration_curve_ref: str
    label_volume: int
    required_label_volume: int
    auc: float
    auc_threshold: float = JUDGE_CALIBRATION_AUC_THRESHOLD
    calibration_claimed_valid: bool = False
    published_label_volume_math: bool = False
    heldout_labels_used: bool = False
    outcome_labels_verified: bool = False
    synthetic_labels_used: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = JudgeCalibrationState.LOCAL_CALIBRATION_STUB.value
    contract_version: str = JUDGE_CALIBRATION_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "JudgeCalibrationEvidenceRecord":
        return cls(
            calibration_id=str(data["calibration_id"]),
            label_set_ref=str(data.get("label_set_ref", "")),
            feasibility_memo_ref=str(data.get("feasibility_memo_ref", "")),
            methodology_ref=str(data.get("methodology_ref", "")),
            calibration_curve_ref=str(data.get("calibration_curve_ref", "")),
            label_volume=int(data.get("label_volume", 0)),
            required_label_volume=int(data.get("required_label_volume", 0)),
            auc=float(data.get("auc", 0.0)),
            auc_threshold=float(data.get("auc_threshold", JUDGE_CALIBRATION_AUC_THRESHOLD)),
            calibration_claimed_valid=bool(data.get("calibration_claimed_valid", False)),
            published_label_volume_math=bool(data.get("published_label_volume_math", False)),
            heldout_labels_used=bool(data.get("heldout_labels_used", False)),
            outcome_labels_verified=bool(data.get("outcome_labels_verified", False)),
            synthetic_labels_used=bool(data.get("synthetic_labels_used", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", JudgeCalibrationState.LOCAL_CALIBRATION_STUB.value)),
            contract_version=str(data.get("contract_version", JUDGE_CALIBRATION_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class ModelPipelineExitGateReadinessRecord:
    exit_gate_id: str
    model_pipeline_readiness_ref: str
    engine_ab_ref: str
    reranker_parity_ref: str
    judge_calibration_ref: str
    workspace_beta_ref: str
    engine_yield_delta_pct: float
    reranker_quality_retention_pct: float
    reranker_cost_ratio_to_teacher_pct: float
    judge_label_volume: int
    judge_required_label_volume: int
    judge_auc: float
    workspace_beta_allowlisted: bool
    workspace_entropy_accounting_live: bool
    workspace_security_review_passed: bool
    evidence_refs: tuple[str, ...]
    owner_approval_ref: str = ""
    engine_yield_threshold_pct: float = ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT
    reranker_quality_threshold_pct: float = RERANKER_PARITY_THRESHOLD_PCT
    reranker_cost_ratio_max_pct: float = RERANKER_COST_RATIO_MAX_PCT
    judge_auc_threshold: float = JUDGE_CALIBRATION_AUC_THRESHOLD
    local_code_complete: bool = False
    production_operational_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    production_writes_enabled: bool = False
    supabase_writes_enabled: bool = False
    public_workflows_enabled: bool = False
    state: str = ModelPipelineExitGateState.LOCAL_NOT_READY.value
    contract_version: str = MODEL_PIPELINE_EXIT_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelPipelineExitGateReadinessRecord":
        return cls(
            exit_gate_id=str(data["exit_gate_id"]),
            model_pipeline_readiness_ref=str(data.get("model_pipeline_readiness_ref", "")),
            engine_ab_ref=str(data.get("engine_ab_ref", "")),
            reranker_parity_ref=str(data.get("reranker_parity_ref", "")),
            judge_calibration_ref=str(data.get("judge_calibration_ref", "")),
            workspace_beta_ref=str(data.get("workspace_beta_ref", "")),
            engine_yield_delta_pct=float(data.get("engine_yield_delta_pct", 0.0)),
            reranker_quality_retention_pct=float(data.get("reranker_quality_retention_pct", 0.0)),
            reranker_cost_ratio_to_teacher_pct=float(data.get("reranker_cost_ratio_to_teacher_pct", 0.0)),
            judge_label_volume=int(data.get("judge_label_volume", 0)),
            judge_required_label_volume=int(data.get("judge_required_label_volume", 0)),
            judge_auc=float(data.get("judge_auc", 0.0)),
            workspace_beta_allowlisted=bool(data.get("workspace_beta_allowlisted", False)),
            workspace_entropy_accounting_live=bool(data.get("workspace_entropy_accounting_live", False)),
            workspace_security_review_passed=bool(data.get("workspace_security_review_passed", False)),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            engine_yield_threshold_pct=float(data.get("engine_yield_threshold_pct", ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT)),
            reranker_quality_threshold_pct=float(data.get("reranker_quality_threshold_pct", RERANKER_PARITY_THRESHOLD_PCT)),
            reranker_cost_ratio_max_pct=float(data.get("reranker_cost_ratio_max_pct", RERANKER_COST_RATIO_MAX_PCT)),
            judge_auc_threshold=float(data.get("judge_auc_threshold", JUDGE_CALIBRATION_AUC_THRESHOLD)),
            local_code_complete=bool(data.get("local_code_complete", False)),
            production_operational_ready=bool(data.get("production_operational_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            production_writes_enabled=bool(data.get("production_writes_enabled", False)),
            supabase_writes_enabled=bool(data.get("supabase_writes_enabled", False)),
            public_workflows_enabled=bool(data.get("public_workflows_enabled", False)),
            state=str(data.get("state", ModelPipelineExitGateState.LOCAL_NOT_READY.value)),
            contract_version=str(data.get("contract_version", MODEL_PIPELINE_EXIT_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def model_pipeline_exit_gate_check_results(
    record: ModelPipelineExitGateReadinessRecord | Mapping[str, Any],
) -> dict[str, bool]:
    if not isinstance(record, ModelPipelineExitGateReadinessRecord):
        record = ModelPipelineExitGateReadinessRecord.from_mapping(record)
    return {
        ModelPipelineExitGateCheck.MODEL_PIPELINE_FOUNDATION_READY.value: record.model_pipeline_readiness_ref.startswith("model_pipeline_readiness:"),
        ModelPipelineExitGateCheck.ENGINE_VNEXT_YIELD_GATE.value: (
            record.engine_ab_ref.startswith("engine_ab:")
            and record.engine_yield_delta_pct >= record.engine_yield_threshold_pct
        ),
        ModelPipelineExitGateCheck.RERANKER_PARITY_COST_GATE.value: (
            record.reranker_parity_ref.startswith("reranker_parity:")
            and record.reranker_quality_retention_pct >= record.reranker_quality_threshold_pct
            and record.reranker_cost_ratio_to_teacher_pct <= record.reranker_cost_ratio_max_pct
        ),
        ModelPipelineExitGateCheck.JUDGE_CALIBRATION_GATE.value: (
            record.judge_calibration_ref.startswith("judge_calibration:")
            and record.judge_label_volume >= record.judge_required_label_volume
            and record.judge_required_label_volume > 0
            and record.judge_auc >= record.judge_auc_threshold
        ),
        ModelPipelineExitGateCheck.WORKSPACE_API_BETA_GATE.value: (
            record.workspace_beta_ref.startswith("workspace_api_beta:")
            and record.workspace_beta_ref != PENDING_WORKSPACE_BETA_REF
            and record.workspace_beta_allowlisted
            and record.workspace_entropy_accounting_live
            and record.workspace_security_review_passed
        ),
    }


def model_pipeline_exit_gate_missing_checks(
    record: ModelPipelineExitGateReadinessRecord | Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        check for check, passed in model_pipeline_exit_gate_check_results(record).items()
        if not passed
    )


def validate_judge_calibration_evidence_record(
    record: JudgeCalibrationEvidenceRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_model_pipeline_exit_payload_errors(raw)
    if not isinstance(record, JudgeCalibrationEvidenceRecord):
        try:
            record = JudgeCalibrationEvidenceRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required judge calibration field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid judge calibration field value: {exc}")
            return errors
    if record.contract_version != JUDGE_CALIBRATION_CONTRACT_VERSION:
        errors.append("contract_version must match judge calibration contract")
    if not record.calibration_id.startswith("judge_calibration:"):
        errors.append("calibration_id must be judge_calibration:-prefixed")
    _validate_prefixed(errors, "label_set_ref", record.label_set_ref, "outcome_label_set:")
    _validate_prefixed(errors, "feasibility_memo_ref", record.feasibility_memo_ref, "baseline_feasibility_memo:")
    _validate_prefixed(errors, "methodology_ref", record.methodology_ref, "judge_calibration_methodology:")
    _validate_prefixed(errors, "calibration_curve_ref", record.calibration_curve_ref, "calibration_curve:")
    if record.label_volume < 0:
        errors.append("label_volume must be non-negative")
    if record.required_label_volume <= 0:
        errors.append("required_label_volume must be positive")
    if not 0.0 <= record.auc <= 1.0:
        errors.append("auc must be between 0 and 1")
    if record.auc_threshold != JUDGE_CALIBRATION_AUC_THRESHOLD:
        errors.append("auc_threshold must match Phase 3 judge calibration threshold")
    if record.state not in {state.value for state in JudgeCalibrationState}:
        errors.append(f"unknown judge calibration state: {record.state}")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith((
            "judge_calibration:",
            "outcome_label_set:",
            "baseline_feasibility_memo:",
            "calibration_curve:",
            "owner_approval:",
        )):
            errors.append("judge calibration evidence_refs use unsupported prefixes")
            break
    if record.calibration_claimed_valid or record.state == JudgeCalibrationState.PASSED_AFTER_LABEL_VOLUME.value:
        if record.uses_local_fixtures:
            errors.append("judge calibration cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("judge calibration cannot be claimed by a local_only record")
        if record.synthetic_labels_used:
            errors.append("judge calibration cannot be claimed from synthetic labels")
        if record.label_volume < record.required_label_volume:
            errors.append("judge calibration requires label volume >= required label volume")
        if record.auc < record.auc_threshold:
            errors.append("judge calibration requires auc >= threshold")
        for field_name in ("published_label_volume_math", "heldout_labels_used", "outcome_labels_verified"):
            if not getattr(record, field_name):
                errors.append(f"judge calibration requires {field_name}")
        if not record.owner_approval_ref:
            errors.append("judge calibration requires owner_approval_ref")
        if record.calibration_id not in record.evidence_refs:
            errors.append("judge calibration evidence_refs must include calibration_id")
    else:
        if not record.local_only:
            errors.append("not-claimed judge calibration records must remain local_only")
    return errors


def validate_model_pipeline_exit_gate_readiness_record(
    record: ModelPipelineExitGateReadinessRecord | Mapping[str, Any],
    *,
    engine_comparison: Optional[EngineABComparisonRecord | Mapping[str, Any]] = None,
    reranker_comparison: Optional[RerankerParityComparisonRecord | Mapping[str, Any]] = None,
    judge_calibration: Optional[JudgeCalibrationEvidenceRecord | Mapping[str, Any]] = None,
    workspace_beta: Optional[WorkspaceAPIBetaEnablementRecord | Mapping[str, Any]] = None,
    guards: Optional[ModelPipelineWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_model_pipeline_exit_payload_errors(raw)
    if not isinstance(record, ModelPipelineExitGateReadinessRecord):
        try:
            record = ModelPipelineExitGateReadinessRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Phase 3 exit-gate field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Phase 3 exit-gate field value: {exc}")
            return errors
    try:
        assert_model_pipeline_workflows_disabled(guards or default_model_pipeline_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != MODEL_PIPELINE_EXIT_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P3.7 exit-gate contract")
    if record.state not in {state.value for state in ModelPipelineExitGateState}:
        errors.append(f"unknown Phase 3 exit-gate state: {record.state}")
    if not record.exit_gate_id.startswith("model_pipeline_exit_gate:"):
        errors.append("exit_gate_id must be model_pipeline_exit_gate:-prefixed")
    _validate_prefixed(errors, "model_pipeline_readiness_ref", record.model_pipeline_readiness_ref, "model_pipeline_readiness:")
    _validate_prefixed(errors, "engine_ab_ref", record.engine_ab_ref, "engine_ab:")
    _validate_prefixed(errors, "reranker_parity_ref", record.reranker_parity_ref, "reranker_parity:")
    _validate_prefixed(errors, "judge_calibration_ref", record.judge_calibration_ref, "judge_calibration:")
    _validate_prefixed(errors, "workspace_beta_ref", record.workspace_beta_ref, "workspace_api_beta:")
    if record.engine_yield_threshold_pct != ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT:
        errors.append("engine_yield_threshold_pct must match Phase 3 exit threshold")
    if record.reranker_quality_threshold_pct != RERANKER_PARITY_THRESHOLD_PCT:
        errors.append("reranker_quality_threshold_pct must match Phase 3 exit threshold")
    if record.reranker_cost_ratio_max_pct != RERANKER_COST_RATIO_MAX_PCT:
        errors.append("reranker_cost_ratio_max_pct must match Phase 3 exit threshold")
    if record.judge_auc_threshold != JUDGE_CALIBRATION_AUC_THRESHOLD:
        errors.append("judge_auc_threshold must match Phase 3 exit threshold")
    if record.judge_label_volume < 0:
        errors.append("judge_label_volume must be non-negative")
    if record.judge_required_label_volume <= 0:
        errors.append("judge_required_label_volume must be positive")
    if not 0.0 <= record.judge_auc <= 1.0:
        errors.append("judge_auc must be between 0 and 1")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith((
            "model_pipeline_readiness:",
            "engine_ab:",
            "reranker_parity:",
            "judge_calibration:",
            "outcome_label_set:",
            "workspace_api_beta:",
            "workspace_api_allowlist:",
            "workspace_entropy:",
            "workspace_security_review:",
            "owner_approval:",
        )):
            errors.append("evidence_refs must use approved Phase 3 exit-gate prefixes")
            break
    if record.production_writes_enabled:
        errors.append("P3.7 exit-gate verifier must not enable production writes")
    if record.supabase_writes_enabled:
        errors.append("P3.7 exit-gate verifier must not enable Supabase writes")
    if record.public_workflows_enabled:
        errors.append("P3.7 exit-gate verifier must not enable public workflows")
    _append_supplied_evidence_consistency_errors(
        errors,
        record,
        engine_comparison=engine_comparison,
        reranker_comparison=reranker_comparison,
        judge_calibration=judge_calibration,
        workspace_beta=workspace_beta,
    )
    missing = model_pipeline_exit_gate_missing_checks(record)
    if record.production_operational_ready:
        if not record.local_code_complete:
            errors.append("production_operational_ready requires local_code_complete")
        if engine_comparison is None:
            errors.append("Phase 3 exit requires supplied engine_comparison")
        if reranker_comparison is None:
            errors.append("Phase 3 exit requires supplied reranker_comparison")
        if judge_calibration is None:
            errors.append("Phase 3 exit requires supplied judge_calibration")
        if workspace_beta is None:
            errors.append("Phase 3 exit requires supplied workspace_beta")
        if record.uses_local_fixtures:
            errors.append("Phase 3 exit cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 3 exit cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 3 exit missing checks: " + ", ".join(missing))
        if not record.owner_approval_ref:
            errors.append("Phase 3 exit requires owner_approval_ref")
        for required_ref in (
            record.model_pipeline_readiness_ref,
            record.engine_ab_ref,
            record.reranker_parity_ref,
            record.judge_calibration_ref,
            record.workspace_beta_ref,
        ):
            if required_ref not in record.evidence_refs:
                errors.append("Phase 3 exit evidence_refs must include all core evidence refs")
                break
        if record.state != ModelPipelineExitGateState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("production_operational_ready requires ready_after_measured_evidence state")
    else:
        if not record.local_code_complete:
            errors.append("Phase 3 exit local verifier should mark local_code_complete")
        if not record.local_only:
            errors.append("not-ready Phase 3 exit records must remain local_only")
        if record.state == ModelPipelineExitGateState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("ready_after_measured_evidence state requires production_operational_ready")
    return errors


def verify_model_pipeline_exit_gate(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    model_pipeline_summary = verify_model_pipeline_foundation()
    engine_summary = verify_research_lab_engine_ab_evaluator()
    reranker_summary = verify_research_lab_reranker_distillation()
    workspace_summary = verify_research_lab_workspace_api_beta()
    fixture = _load_fixture(Path(fixture_path))

    local_judge = JudgeCalibrationEvidenceRecord.from_mapping(fixture["local_judge_calibration"])
    _assert(not validate_judge_calibration_evidence_record(local_judge), "local judge calibration stub validates")
    _assert(not local_judge.calibration_claimed_valid, "local judge calibration does not claim validity")

    local_exit = ModelPipelineExitGateReadinessRecord.from_mapping(fixture["local_exit_gate"])
    _assert(not validate_model_pipeline_exit_gate_readiness_record(local_exit), "local Phase 3 exit-gate record validates")
    _assert(local_exit.local_code_complete, "P3.7 local verifier marks code complete")
    _assert(not local_exit.production_operational_ready, "P3.7 local verifier does not claim production readiness")
    _assert(model_pipeline_exit_gate_missing_checks(local_exit), "P3.7 local verifier reports missing measured checks")
    _assert(local_exit.model_pipeline_readiness_ref == model_pipeline_summary["readiness_id"], "P3.7 pins P3.0 readiness ref")
    _assert(local_exit.engine_ab_ref == engine_summary["comparison_id"], "P3.7 local fixture pins P3.4 comparison ref")
    _assert(local_exit.reranker_parity_ref == reranker_summary["comparison_id"], "P3.7 local fixture pins P3.5 comparison ref")
    _assert(local_exit.workspace_beta_ref == workspace_summary["allowlist_id"].replace("workspace_api_allowlist:", "workspace_api_beta:"), "P3.7 local fixture pins P3.6 beta ref")

    measured_engine = EngineABComparisonRecord.from_mapping(fixture["measured_engine_comparison"])
    measured_reranker = RerankerParityComparisonRecord.from_mapping(fixture["measured_reranker_comparison"])
    measured_judge = JudgeCalibrationEvidenceRecord.from_mapping(fixture["measured_judge_calibration"])
    measured_workspace = WorkspaceAPIBetaEnablementRecord.from_mapping(fixture["measured_workspace_beta"])
    measured_exit = ModelPipelineExitGateReadinessRecord.from_mapping(fixture["measured_exit_gate"])
    _assert(not validate_judge_calibration_evidence_record(measured_judge), "measured judge calibration validates")
    measured_errors = validate_model_pipeline_exit_gate_readiness_record(
        measured_exit,
        engine_comparison=measured_engine,
        reranker_comparison=measured_reranker,
        judge_calibration=measured_judge,
        workspace_beta=measured_workspace,
    )
    _assert(not measured_errors, "fully supplied measured Phase 3 exit gate validates")

    bare_errors = validate_model_pipeline_exit_gate_readiness_record(measured_exit)
    _assert(bare_errors, "bare measured Phase 3 exit-gate claim fails closed")
    _assert_expected_error(bare_errors, fixture["bare_measured_exit_expected_errors"])

    unsafe_guard_errors = validate_model_pipeline_exit_gate_readiness_record(
        local_exit,
        guards=fixture["unsafe_model_pipeline_guards"],
    )
    _assert(unsafe_guard_errors, "unsafe Phase 3 guards block P3.7 verifier")
    _assert_expected_error(unsafe_guard_errors, fixture["unsafe_model_pipeline_guards"])

    for invalid in fixture["invalid_judge_calibrations"]:
        base = fixture[str(invalid.get("base", "local_judge_calibration"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_judge_calibration_evidence_record(record)
        _assert(errors, f"invalid judge calibration fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_exit_gates"]:
        base = fixture[str(invalid.get("base", "local_exit_gate"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        supplied = str(invalid.get("supplied", "local"))
        if supplied == "measured":
            engine = measured_engine
            reranker = measured_reranker
            judge = measured_judge
            workspace = measured_workspace
        else:
            engine = None
            reranker = None
            judge = None
            workspace = None
        errors = validate_model_pipeline_exit_gate_readiness_record(
            record,
            engine_comparison=engine,
            reranker_comparison=reranker,
            judge_calibration=judge,
            workspace_beta=workspace,
        )
        _assert(errors, f"invalid Phase 3 exit gate fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    return {
        "exit_gate_id": local_exit.exit_gate_id,
        "local_code_complete": local_exit.local_code_complete,
        "production_operational_ready": local_exit.production_operational_ready,
        "missing_checks": list(model_pipeline_exit_gate_missing_checks(local_exit)),
        "measured_control_validates": True,
        "engine_yield_threshold_pct": ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT,
        "reranker_quality_threshold_pct": RERANKER_PARITY_THRESHOLD_PCT,
        "reranker_cost_ratio_max_pct": RERANKER_COST_RATIO_MAX_PCT,
        "judge_auc_threshold": JUDGE_CALIBRATION_AUC_THRESHOLD,
        "workspace_beta_ref": local_exit.workspace_beta_ref,
    }


def _append_supplied_evidence_consistency_errors(
    errors: list[str],
    record: ModelPipelineExitGateReadinessRecord,
    *,
    engine_comparison: Optional[EngineABComparisonRecord | Mapping[str, Any]],
    reranker_comparison: Optional[RerankerParityComparisonRecord | Mapping[str, Any]],
    judge_calibration: Optional[JudgeCalibrationEvidenceRecord | Mapping[str, Any]],
    workspace_beta: Optional[WorkspaceAPIBetaEnablementRecord | Mapping[str, Any]],
) -> None:
    if engine_comparison is not None:
        engine = engine_comparison if isinstance(engine_comparison, EngineABComparisonRecord) else EngineABComparisonRecord.from_mapping(engine_comparison)
        if record.engine_ab_ref != engine.comparison_id:
            errors.append("Phase 3 exit engine_ab_ref mismatch")
        if record.engine_yield_delta_pct != engine.yield_delta_pct:
            errors.append("Phase 3 exit engine_yield_delta_pct mismatch")
        if record.production_operational_ready:
            if not engine.measured_data_ready or engine.uses_local_fixtures or engine.local_only:
                errors.append("Phase 3 exit requires measured non-local engine comparison")
            if not engine.passed_20_percent_gate or engine.state != EngineABComparisonState.PASSED_20_PERCENT_GATE.value:
                errors.append("Phase 3 exit requires engine comparison passed_20_percent_gate")
            if not engine.improvement_claimed:
                errors.append("Phase 3 exit requires engine improvement_claimed")
    if reranker_comparison is not None:
        reranker = reranker_comparison if isinstance(reranker_comparison, RerankerParityComparisonRecord) else RerankerParityComparisonRecord.from_mapping(reranker_comparison)
        if record.reranker_parity_ref != reranker.comparison_id:
            errors.append("Phase 3 exit reranker_parity_ref mismatch")
        if record.reranker_quality_retention_pct != reranker.quality_retention_pct:
            errors.append("Phase 3 exit reranker_quality_retention_pct mismatch")
        if record.reranker_cost_ratio_to_teacher_pct != reranker.cost_ratio_to_teacher_pct:
            errors.append("Phase 3 exit reranker_cost_ratio_to_teacher_pct mismatch")
        if record.production_operational_ready:
            if not reranker.measured_data_ready or reranker.uses_local_fixtures or reranker.local_only:
                errors.append("Phase 3 exit requires measured non-local reranker comparison")
            if not reranker.passed_parity_cost_gate or reranker.state != RerankerParityState.PASSED_PARITY_COST_GATE.value:
                errors.append("Phase 3 exit requires reranker passed_parity_cost_gate")
            if not reranker.parity_claimed:
                errors.append("Phase 3 exit requires reranker parity_claimed")
    if judge_calibration is not None:
        judge = judge_calibration if isinstance(judge_calibration, JudgeCalibrationEvidenceRecord) else JudgeCalibrationEvidenceRecord.from_mapping(judge_calibration)
        judge_errors = validate_judge_calibration_evidence_record(judge)
        errors.extend(f"judge_calibration invalid: {error}" for error in judge_errors)
        if record.judge_calibration_ref != judge.calibration_id:
            errors.append("Phase 3 exit judge_calibration_ref mismatch")
        if record.judge_label_volume != judge.label_volume:
            errors.append("Phase 3 exit judge_label_volume mismatch")
        if record.judge_required_label_volume != judge.required_label_volume:
            errors.append("Phase 3 exit judge_required_label_volume mismatch")
        if record.judge_auc != judge.auc:
            errors.append("Phase 3 exit judge_auc mismatch")
        if record.production_operational_ready and not judge.calibration_claimed_valid:
            errors.append("Phase 3 exit requires valid judge calibration")
    if workspace_beta is not None:
        workspace = workspace_beta if isinstance(workspace_beta, WorkspaceAPIBetaEnablementRecord) else WorkspaceAPIBetaEnablementRecord.from_mapping(workspace_beta)
        if record.workspace_beta_ref != workspace.beta_id:
            errors.append("Phase 3 exit workspace_beta_ref mismatch")
        if record.production_operational_ready:
            if workspace.beta_id == PENDING_WORKSPACE_BETA_REF:
                errors.append("Phase 3 exit requires non-pending workspace beta")
            if not workspace.beta_enablement_claimed or not workspace.workspace_api_calls_enabled:
                errors.append("Phase 3 exit requires claimed Workspace API beta enablement")
            if workspace.uses_local_fixtures or workspace.local_only:
                errors.append("Phase 3 exit requires measured non-local Workspace API beta")
            if workspace.state != WorkspaceAPIBetaState.ALLOWLISTED_BETA_ENABLED.value:
                errors.append("Phase 3 exit requires allowlisted Workspace API beta state")


def _validate_prefixed(errors: list[str], field_name: str, value: str, prefix: str) -> None:
    if not value.startswith(prefix):
        errors.append(f"{field_name} must be {prefix}-prefixed")


def _protected_model_pipeline_exit_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_model_pipeline_exit_material(record))
    if not found:
        return []
    return ["Phase 3 exit-gate payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_model_pipeline_exit_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_MODEL_PIPELINE_EXIT_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_model_pipeline_exit_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_model_pipeline_exit_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_MODEL_PIPELINE_EXIT_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


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
