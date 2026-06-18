"""Phase 4.5 Stage-3 end-to-end experiment contracts.

P4.5 defines local record shapes for future Stage-3 end-to-end model
experiments. It does not start SFT/RL jobs, run confidential GPUs, publish
weights, deploy models, write SQL, or claim success from local fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .confidential_gpu_training import (
    FIXTURE_PATH as CONFIDENTIAL_GPU_FIXTURE_PATH,
    ConfidentialGPUDataPolicyRecord,
    ConfidentialGPUTrainingRunRecord,
    PROTECTED_CONFIDENTIAL_GPU_KEYS,
    PROTECTED_CONFIDENTIAL_GPU_MARKERS,
    validate_confidential_gpu_training_run_record,
    verify_research_lab_confidential_gpu_training,
)
from .model_pipeline_exit_gate import (
    PROTECTED_MODEL_PIPELINE_EXIT_KEYS,
    PROTECTED_MODEL_PIPELINE_EXIT_MARKERS,
    verify_model_pipeline_exit_gate,
)
from .scale_foundation import (
    PROTECTED_SCALE_KEYS,
    PROTECTED_SCALE_MARKERS,
    ScaleGate,
    ScaleWorkflowGuards,
    assert_scale_workflows_disabled,
    default_scale_workflow_guards,
    require_scale_gate,
    verify_scale_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage3_experiment_fixtures.json"

STAGE3_EXPERIMENT_PLAN_CONTRACT_VERSION = "stage3_experiment_plan:v1:local_contract"
STAGE3_EXPERIMENT_RUN_CONTRACT_VERSION = "stage3_experiment_run:v1:local_contract"
STAGE3_HELDOUT_EVAL_CONTRACT_VERSION = "stage3_heldout_eval:v1:local_contract"
STAGE3_SUCCESS_CLAIM_CONTRACT_VERSION = "stage3_success_claim:v1:local_contract"

DEFAULT_STAGE3_REQUIRED_HELDOUT_SAMPLE_COUNT = 100
DEFAULT_STAGE3_REQUIRED_SCORE_DELTA_PCT = 5.0

PROTECTED_STAGE3_EXPERIMENT_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SCALE_KEYS)
    | set(PROTECTED_MODEL_PIPELINE_EXIT_KEYS)
    | set(PROTECTED_CONFIDENTIAL_GPU_KEYS)
    | {
        "axis_a_trace_payload",
        "axis_b_trace_payload",
        "customer_outcome_label",
        "dense_reward_label",
        "heldout_answer_key",
        "live_rl_checkpoint",
        "raw_reward_trace",
        "raw_stage3_training_row",
        "sequestered_eval_answer",
        "teacher_reasoning_trace",
    }
)

PROTECTED_STAGE3_EXPERIMENT_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SCALE_MARKERS)
        | set(PROTECTED_MODEL_PIPELINE_EXIT_MARKERS)
        | set(PROTECTED_CONFIDENTIAL_GPU_MARKERS)
        | {
            "axis-a trace payload",
            "axis-b trace payload",
            "customer outcome label",
            "dense reward label",
            "heldout answer key",
            "live rl checkpoint",
            "raw reward trace",
            "raw stage3 training row",
            "sequestered eval answer",
            "teacher reasoning trace",
        }
    )
)


class Stage3ExperimentPlanState(str, Enum):
    LOCAL_PLAN_STUB = "local_plan_stub"
    READY_AFTER_METHOD_APPROVAL = "ready_after_method_approval"
    BLOCKED = "blocked"


class Stage3ExperimentRunState(str, Enum):
    LOCAL_RUN_STUB = "local_run_stub"
    READY_AFTER_CONFIDENTIAL_GPU_RUN = "ready_after_confidential_gpu_run"
    BLOCKED = "blocked"


class Stage3HeldoutEvalState(str, Enum):
    LOCAL_EVAL_STUB = "local_eval_stub"
    PASSED_AFTER_MEASURED_HELDOUT = "passed_after_measured_heldout"
    FAILED_AFTER_MEASURED_HELDOUT = "failed_after_measured_heldout"
    BLOCKED = "blocked"


class Stage3SuccessClaimState(str, Enum):
    LOCAL_NOT_CLAIMED = "local_not_claimed"
    READY_AFTER_OWNER_REVIEW = "ready_after_owner_review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Stage3ExperimentPlanRecord:
    plan_id: str
    model_pipeline_exit_gate_ref: str
    confidential_gpu_training_ref: str
    data_policy_ref: str
    axis_selection_policy_ref: str
    rewrite_pipeline_ref: str
    reward_spec_ref: str
    heldout_eval_plan_ref: str
    outcome_label_set_ref: str
    methodology_ref: str
    target_hybrid_champion_ref: str
    axis_a_or_rewritten_only: bool = False
    leak_cluster_split_enforced: bool = False
    token_budget_curation_checked: bool = False
    teacher_model_policy_enforced: bool = False
    reward_uses_attested_eval_ladder: bool = False
    generation_n_never_grades_n_plus_1: bool = False
    plan_approved: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = Stage3ExperimentPlanState.LOCAL_PLAN_STUB.value
    contract_version: str = STAGE3_EXPERIMENT_PLAN_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage3ExperimentPlanRecord":
        return cls(
            plan_id=str(data["plan_id"]),
            model_pipeline_exit_gate_ref=str(data["model_pipeline_exit_gate_ref"]),
            confidential_gpu_training_ref=str(data["confidential_gpu_training_ref"]),
            data_policy_ref=str(data["data_policy_ref"]),
            axis_selection_policy_ref=str(data["axis_selection_policy_ref"]),
            rewrite_pipeline_ref=str(data["rewrite_pipeline_ref"]),
            reward_spec_ref=str(data["reward_spec_ref"]),
            heldout_eval_plan_ref=str(data["heldout_eval_plan_ref"]),
            outcome_label_set_ref=str(data["outcome_label_set_ref"]),
            methodology_ref=str(data["methodology_ref"]),
            target_hybrid_champion_ref=str(data["target_hybrid_champion_ref"]),
            axis_a_or_rewritten_only=bool(data.get("axis_a_or_rewritten_only", False)),
            leak_cluster_split_enforced=bool(data.get("leak_cluster_split_enforced", False)),
            token_budget_curation_checked=bool(data.get("token_budget_curation_checked", False)),
            teacher_model_policy_enforced=bool(data.get("teacher_model_policy_enforced", False)),
            reward_uses_attested_eval_ladder=bool(data.get("reward_uses_attested_eval_ladder", False)),
            generation_n_never_grades_n_plus_1=bool(data.get("generation_n_never_grades_n_plus_1", False)),
            plan_approved=bool(data.get("plan_approved", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", Stage3ExperimentPlanState.LOCAL_PLAN_STUB.value)),
            contract_version=str(data.get("contract_version", STAGE3_EXPERIMENT_PLAN_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class Stage3ExperimentRunRecord:
    experiment_run_id: str
    plan_ref: str
    confidential_gpu_training_ref: str
    private_weight_artifact_ref: str
    sft_seed_dataset_ref: str
    rl_environment_ref: str
    reward_spec_ref: str
    axis_mix_summary_ref: str
    cost_ledger_hash: str
    run_summary_hash: str
    sft_seed_from_rewritten_axis_a: bool = False
    rlvr_reward_from_attested_eval: bool = False
    dense_per_step_reward_enabled: bool = False
    passk_headroom_checked: bool = False
    confidential_gpu_training_valid: bool = False
    training_started: bool = False
    training_completed: bool = False
    production_experiment_run_valid: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = Stage3ExperimentRunState.LOCAL_RUN_STUB.value
    contract_version: str = STAGE3_EXPERIMENT_RUN_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage3ExperimentRunRecord":
        return cls(
            experiment_run_id=str(data["experiment_run_id"]),
            plan_ref=str(data["plan_ref"]),
            confidential_gpu_training_ref=str(data["confidential_gpu_training_ref"]),
            private_weight_artifact_ref=str(data["private_weight_artifact_ref"]),
            sft_seed_dataset_ref=str(data["sft_seed_dataset_ref"]),
            rl_environment_ref=str(data["rl_environment_ref"]),
            reward_spec_ref=str(data["reward_spec_ref"]),
            axis_mix_summary_ref=str(data["axis_mix_summary_ref"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            run_summary_hash=str(data["run_summary_hash"]),
            sft_seed_from_rewritten_axis_a=bool(data.get("sft_seed_from_rewritten_axis_a", False)),
            rlvr_reward_from_attested_eval=bool(data.get("rlvr_reward_from_attested_eval", False)),
            dense_per_step_reward_enabled=bool(data.get("dense_per_step_reward_enabled", False)),
            passk_headroom_checked=bool(data.get("passk_headroom_checked", False)),
            confidential_gpu_training_valid=bool(data.get("confidential_gpu_training_valid", False)),
            training_started=bool(data.get("training_started", False)),
            training_completed=bool(data.get("training_completed", False)),
            production_experiment_run_valid=bool(data.get("production_experiment_run_valid", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", Stage3ExperimentRunState.LOCAL_RUN_STUB.value)),
            contract_version=str(data.get("contract_version", STAGE3_EXPERIMENT_RUN_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class Stage3HeldoutEvaluationRecord:
    heldout_eval_id: str
    experiment_run_ref: str
    sequestered_set_ref: str
    outcome_label_set_ref: str
    baseline_hybrid_ref: str
    candidate_weight_artifact_ref: str
    evaluation_ladder_ref: str
    cost_ledger_hash: str
    result_bundle_hash: str
    sample_count: int = 0
    required_sample_count: int = DEFAULT_STAGE3_REQUIRED_HELDOUT_SAMPLE_COUNT
    score_delta_pct: float = 0.0
    required_score_delta_pct: float = DEFAULT_STAGE3_REQUIRED_SCORE_DELTA_PCT
    outcome_labels_verified: bool = False
    heldout_set_sequestered: bool = False
    leak_cluster_split_verified: bool = False
    judge_calibration_valid: bool = False
    evaluator_blinded: bool = False
    measured_heldout_pass: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = Stage3HeldoutEvalState.LOCAL_EVAL_STUB.value
    contract_version: str = STAGE3_HELDOUT_EVAL_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage3HeldoutEvaluationRecord":
        return cls(
            heldout_eval_id=str(data["heldout_eval_id"]),
            experiment_run_ref=str(data["experiment_run_ref"]),
            sequestered_set_ref=str(data["sequestered_set_ref"]),
            outcome_label_set_ref=str(data["outcome_label_set_ref"]),
            baseline_hybrid_ref=str(data["baseline_hybrid_ref"]),
            candidate_weight_artifact_ref=str(data["candidate_weight_artifact_ref"]),
            evaluation_ladder_ref=str(data["evaluation_ladder_ref"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            result_bundle_hash=str(data["result_bundle_hash"]),
            sample_count=int(data.get("sample_count", 0)),
            required_sample_count=int(data.get("required_sample_count", DEFAULT_STAGE3_REQUIRED_HELDOUT_SAMPLE_COUNT)),
            score_delta_pct=float(data.get("score_delta_pct", 0.0)),
            required_score_delta_pct=float(data.get("required_score_delta_pct", DEFAULT_STAGE3_REQUIRED_SCORE_DELTA_PCT)),
            outcome_labels_verified=bool(data.get("outcome_labels_verified", False)),
            heldout_set_sequestered=bool(data.get("heldout_set_sequestered", False)),
            leak_cluster_split_verified=bool(data.get("leak_cluster_split_verified", False)),
            judge_calibration_valid=bool(data.get("judge_calibration_valid", False)),
            evaluator_blinded=bool(data.get("evaluator_blinded", False)),
            measured_heldout_pass=bool(data.get("measured_heldout_pass", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", Stage3HeldoutEvalState.LOCAL_EVAL_STUB.value)),
            contract_version=str(data.get("contract_version", STAGE3_HELDOUT_EVAL_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class Stage3SuccessClaimRecord:
    success_claim_id: str
    experiment_run_ref: str
    heldout_eval_ref: str
    model_pipeline_exit_gate_ref: str
    confidential_gpu_training_ref: str
    model_weight_artifact_ref: str
    success_claimed: bool = False
    beats_hybrid_champion: bool = False
    heldout_eval_passed: bool = False
    outcome_evidence_verified: bool = False
    production_deployment_requested: bool = False
    model_weight_publication_enabled: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = Stage3SuccessClaimState.LOCAL_NOT_CLAIMED.value
    contract_version: str = STAGE3_SUCCESS_CLAIM_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage3SuccessClaimRecord":
        return cls(
            success_claim_id=str(data["success_claim_id"]),
            experiment_run_ref=str(data["experiment_run_ref"]),
            heldout_eval_ref=str(data["heldout_eval_ref"]),
            model_pipeline_exit_gate_ref=str(data["model_pipeline_exit_gate_ref"]),
            confidential_gpu_training_ref=str(data["confidential_gpu_training_ref"]),
            model_weight_artifact_ref=str(data["model_weight_artifact_ref"]),
            success_claimed=bool(data.get("success_claimed", False)),
            beats_hybrid_champion=bool(data.get("beats_hybrid_champion", False)),
            heldout_eval_passed=bool(data.get("heldout_eval_passed", False)),
            outcome_evidence_verified=bool(data.get("outcome_evidence_verified", False)),
            production_deployment_requested=bool(data.get("production_deployment_requested", False)),
            model_weight_publication_enabled=bool(data.get("model_weight_publication_enabled", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", Stage3SuccessClaimState.LOCAL_NOT_CLAIMED.value)),
            contract_version=str(data.get("contract_version", STAGE3_SUCCESS_CLAIM_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def validate_stage3_experiment_plan_record(
    record: Stage3ExperimentPlanRecord | Mapping[str, Any],
    *,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_stage3_experiment_payload_errors(raw)
    if not isinstance(record, Stage3ExperimentPlanRecord):
        try:
            record = Stage3ExperimentPlanRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Stage-3 experiment plan field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Stage-3 experiment plan payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in Stage3ExperimentPlanState}:
        errors.append(f"unknown Stage-3 experiment plan state: {record.state}")
        return errors
    if record.contract_version != STAGE3_EXPERIMENT_PLAN_CONTRACT_VERSION:
        errors.append("contract_version must match Stage-3 experiment plan contract")
    if not record.plan_id.startswith("stage3_experiment_plan:"):
        errors.append("plan_id must be stage3_experiment_plan:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("model_pipeline_exit_gate_ref", ("model_pipeline_exit_gate:",)),
            ("confidential_gpu_training_ref", ("confidential_gpu_training:",)),
            ("data_policy_ref", ("data_policy:",)),
            ("axis_selection_policy_ref", ("axis_policy:",)),
            ("rewrite_pipeline_ref", ("rewrite_pipeline:",)),
            ("reward_spec_ref", ("reward_spec:",)),
            ("heldout_eval_plan_ref", ("heldout_eval_plan:",)),
            ("outcome_label_set_ref", ("outcome_label_set:",)),
            ("methodology_ref", ("methodology:",)),
            ("target_hybrid_champion_ref", ("champion:",)),
        ),
    )
    _append_stage3_write_guard_errors(record, "P4.5 Stage-3 experiment plan", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Stage-3 experiment plan",
        (
            "stage3_experiment_plan:",
            "model_pipeline_exit_gate:",
            "confidential_gpu_training:",
            "data_policy:",
            "axis_policy:",
            "rewrite_pipeline:",
            "reward_spec:",
            "heldout_eval_plan:",
            "outcome_label_set:",
            "methodology:",
            "owner_approval:",
        ),
    )
    if record.plan_approved:
        _append_missing_true_flags(
            record,
            errors,
            (
                ("axis_a_or_rewritten_only", "approved Stage-3 plan requires axis-A or rewritten-axis-A inputs only"),
                ("leak_cluster_split_enforced", "approved Stage-3 plan requires leak-cluster split enforcement"),
                ("token_budget_curation_checked", "approved Stage-3 plan requires token-budget curation checks"),
                ("teacher_model_policy_enforced", "approved Stage-3 plan requires teacher-model policy enforcement"),
                ("reward_uses_attested_eval_ladder", "approved Stage-3 plan requires attested eval-ladder reward"),
                (
                    "generation_n_never_grades_n_plus_1",
                    "approved Stage-3 plan requires generation N never grades N+1",
                ),
            ),
        )
        if record.uses_local_fixtures:
            errors.append("Stage-3 experiment plan approval cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Stage-3 experiment plan approval cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("Stage-3 experiment plan approval requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Stage-3 experiment plan approval requires owner_approval_ref")
        if record.state != Stage3ExperimentPlanState.READY_AFTER_METHOD_APPROVAL.value:
            errors.append("plan_approved requires ready_after_method_approval state")
    else:
        if not record.local_only:
            errors.append("not-approved Stage-3 experiment plans must remain local_only")
        if record.state == Stage3ExperimentPlanState.READY_AFTER_METHOD_APPROVAL.value:
            errors.append("ready_after_method_approval state requires plan_approved")
    return errors


def validate_stage3_experiment_run_record(
    record: Stage3ExperimentRunRecord | Mapping[str, Any],
    *,
    plan: Optional[Stage3ExperimentPlanRecord | Mapping[str, Any]] = None,
    confidential_gpu_training: Optional[ConfidentialGPUTrainingRunRecord | Mapping[str, Any]] = None,
    confidential_gpu_data_policy: Optional[ConfidentialGPUDataPolicyRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_stage3_experiment_payload_errors(raw)
    if not isinstance(record, Stage3ExperimentRunRecord):
        try:
            record = Stage3ExperimentRunRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Stage-3 experiment run field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Stage-3 experiment run payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in Stage3ExperimentRunState}:
        errors.append(f"unknown Stage-3 experiment run state: {record.state}")
        return errors
    if record.contract_version != STAGE3_EXPERIMENT_RUN_CONTRACT_VERSION:
        errors.append("contract_version must match Stage-3 experiment run contract")
    if not record.experiment_run_id.startswith("stage3_experiment:"):
        errors.append("experiment_run_id must be stage3_experiment:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("plan_ref", ("stage3_experiment_plan:",)),
            ("confidential_gpu_training_ref", ("confidential_gpu_training:",)),
            ("private_weight_artifact_ref", ("model_weight_artifact:",)),
            ("sft_seed_dataset_ref", ("fine_tune_dataset:", "reranker_distillation_dataset:", "trajectory_corpus:")),
            ("rl_environment_ref", ("rl_environment:",)),
            ("reward_spec_ref", ("reward_spec:",)),
            ("axis_mix_summary_ref", ("axis_mix:",)),
        ),
    )
    for field in ("cost_ledger_hash", "run_summary_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    _append_stage3_write_guard_errors(record, "P4.5 Stage-3 experiment run", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Stage-3 experiment run",
        (
            "stage3_experiment:",
            "stage3_experiment_plan:",
            "confidential_gpu_training:",
            "model_weight_artifact:",
            "fine_tune_dataset:",
            "reranker_distillation_dataset:",
            "trajectory_corpus:",
            "rl_environment:",
            "reward_spec:",
            "axis_mix:",
            "cost_ledger:",
            "owner_approval:",
        ),
    )
    if plan is not None:
        if not isinstance(plan, Stage3ExperimentPlanRecord):
            plan = Stage3ExperimentPlanRecord.from_mapping(plan)
        plan_errors = validate_stage3_experiment_plan_record(plan)
        if plan_errors:
            errors.append("source Stage-3 experiment plan is invalid: " + "; ".join(plan_errors))
        if record.plan_ref != plan.plan_id:
            errors.append("Stage-3 experiment run plan_ref mismatch")
    if confidential_gpu_data_policy is not None and not isinstance(
        confidential_gpu_data_policy,
        ConfidentialGPUDataPolicyRecord,
    ):
        confidential_gpu_data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(confidential_gpu_data_policy)
    if confidential_gpu_training is not None:
        if not isinstance(confidential_gpu_training, ConfidentialGPUTrainingRunRecord):
            confidential_gpu_training = ConfidentialGPUTrainingRunRecord.from_mapping(confidential_gpu_training)
        training_errors = validate_confidential_gpu_training_run_record(
            confidential_gpu_training,
            data_policy=confidential_gpu_data_policy,
        )
        if training_errors:
            errors.append("source confidential GPU training is invalid: " + "; ".join(training_errors))
        if record.confidential_gpu_training_ref != confidential_gpu_training.training_run_id:
            errors.append("Stage-3 experiment run confidential_gpu_training_ref mismatch")
        if record.confidential_gpu_training_valid != confidential_gpu_training.production_training_valid:
            errors.append("Stage-3 experiment run confidential_gpu_training_valid flag mismatch")
    if record.production_experiment_run_valid:
        if plan is None:
            errors.append("production Stage-3 experiment run requires supplied plan")
        if confidential_gpu_training is None:
            errors.append("production Stage-3 experiment run requires supplied confidential_gpu_training")
        if confidential_gpu_data_policy is None:
            errors.append("production Stage-3 experiment run requires supplied confidential_gpu_data_policy")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("sft_seed_from_rewritten_axis_a", "production Stage-3 run requires rewritten-axis-A SFT seed"),
                ("rlvr_reward_from_attested_eval", "production Stage-3 run requires attested-eval RLVR reward"),
                ("dense_per_step_reward_enabled", "production Stage-3 run requires dense per-step reward"),
                ("passk_headroom_checked", "production Stage-3 run requires pass@k headroom check"),
                ("confidential_gpu_training_valid", "production Stage-3 run requires confidential-GPU training validity"),
                ("training_started", "production Stage-3 run requires training_started"),
                ("training_completed", "production Stage-3 run requires training_completed"),
            ),
        )
        if plan is not None and not plan.plan_approved:
            errors.append("production Stage-3 experiment run requires approved plan")
        if record.uses_local_fixtures:
            errors.append("Stage-3 experiment run validity cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Stage-3 experiment run validity cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("Stage-3 experiment run validity requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Stage-3 experiment run validity requires owner_approval_ref")
        if record.state != Stage3ExperimentRunState.READY_AFTER_CONFIDENTIAL_GPU_RUN.value:
            errors.append("production_experiment_run_valid requires ready_after_confidential_gpu_run state")
    else:
        if not record.local_only:
            errors.append("not-valid Stage-3 experiment runs must remain local_only")
        if record.state == Stage3ExperimentRunState.READY_AFTER_CONFIDENTIAL_GPU_RUN.value:
            errors.append("ready_after_confidential_gpu_run state requires production_experiment_run_valid")
        if record.training_started:
            errors.append("local P4.5 records must not start Stage-3 training")
        if record.training_completed:
            errors.append("local P4.5 records must not complete Stage-3 training")
    return errors


def validate_stage3_heldout_evaluation_record(
    record: Stage3HeldoutEvaluationRecord | Mapping[str, Any],
    *,
    experiment_run: Optional[Stage3ExperimentRunRecord | Mapping[str, Any]] = None,
    plan: Optional[Stage3ExperimentPlanRecord | Mapping[str, Any]] = None,
    confidential_gpu_training: Optional[ConfidentialGPUTrainingRunRecord | Mapping[str, Any]] = None,
    confidential_gpu_data_policy: Optional[ConfidentialGPUDataPolicyRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_stage3_experiment_payload_errors(raw)
    if not isinstance(record, Stage3HeldoutEvaluationRecord):
        try:
            record = Stage3HeldoutEvaluationRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Stage-3 heldout evaluation field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Stage-3 heldout evaluation payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in Stage3HeldoutEvalState}:
        errors.append(f"unknown Stage-3 heldout evaluation state: {record.state}")
        return errors
    if record.contract_version != STAGE3_HELDOUT_EVAL_CONTRACT_VERSION:
        errors.append("contract_version must match Stage-3 heldout evaluation contract")
    if not record.heldout_eval_id.startswith("heldout_eval:"):
        errors.append("heldout_eval_id must be heldout_eval:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("experiment_run_ref", ("stage3_experiment:",)),
            ("sequestered_set_ref", ("sequestered_set:",)),
            ("outcome_label_set_ref", ("outcome_label_set:",)),
            ("baseline_hybrid_ref", ("champion:",)),
            ("candidate_weight_artifact_ref", ("model_weight_artifact:",)),
            ("evaluation_ladder_ref", ("evaluation_ladder:",)),
        ),
    )
    for field in ("cost_ledger_hash", "result_bundle_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    if record.required_sample_count < DEFAULT_STAGE3_REQUIRED_HELDOUT_SAMPLE_COUNT:
        errors.append("required_sample_count must be at least the Stage-3 default minimum")
    if record.required_score_delta_pct < DEFAULT_STAGE3_REQUIRED_SCORE_DELTA_PCT:
        errors.append("required_score_delta_pct must be at least the Stage-3 default minimum")
    _append_stage3_write_guard_errors(record, "P4.5 Stage-3 heldout evaluation", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Stage-3 heldout evaluation",
        (
            "heldout_eval:",
            "stage3_experiment:",
            "sequestered_set:",
            "outcome_label_set:",
            "evaluation_ladder:",
            "judge_calibration:",
            "cost_ledger:",
            "owner_approval:",
        ),
    )
    if experiment_run is not None:
        if not isinstance(experiment_run, Stage3ExperimentRunRecord):
            experiment_run = Stage3ExperimentRunRecord.from_mapping(experiment_run)
        run_errors = validate_stage3_experiment_run_record(
            experiment_run,
            plan=plan,
            confidential_gpu_training=confidential_gpu_training,
            confidential_gpu_data_policy=confidential_gpu_data_policy,
        )
        if run_errors:
            errors.append("source Stage-3 experiment run is invalid: " + "; ".join(run_errors))
        if record.experiment_run_ref != experiment_run.experiment_run_id:
            errors.append("Stage-3 heldout evaluation experiment_run_ref mismatch")
    if record.measured_heldout_pass:
        if experiment_run is None:
            errors.append("measured Stage-3 heldout pass requires supplied experiment_run")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("outcome_labels_verified", "measured Stage-3 heldout pass requires verified outcome labels"),
                ("heldout_set_sequestered", "measured Stage-3 heldout pass requires sequestered heldout set"),
                ("leak_cluster_split_verified", "measured Stage-3 heldout pass requires leak-cluster split verification"),
                ("judge_calibration_valid", "measured Stage-3 heldout pass requires valid judge calibration"),
                ("evaluator_blinded", "measured Stage-3 heldout pass requires blinded evaluator"),
            ),
        )
        if experiment_run is not None and not experiment_run.production_experiment_run_valid:
            errors.append("measured Stage-3 heldout pass requires production-valid experiment run")
        if record.sample_count < record.required_sample_count:
            errors.append("measured Stage-3 heldout pass requires sample_count >= required_sample_count")
        if record.score_delta_pct < record.required_score_delta_pct:
            errors.append("measured Stage-3 heldout pass requires score_delta_pct >= required_score_delta_pct")
        if record.uses_local_fixtures:
            errors.append("Stage-3 heldout pass cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Stage-3 heldout pass cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("Stage-3 heldout pass requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Stage-3 heldout pass requires owner_approval_ref")
        if record.state != Stage3HeldoutEvalState.PASSED_AFTER_MEASURED_HELDOUT.value:
            errors.append("measured_heldout_pass requires passed_after_measured_heldout state")
    else:
        if not record.local_only:
            errors.append("not-passed Stage-3 heldout evaluations must remain local_only")
        if record.state == Stage3HeldoutEvalState.PASSED_AFTER_MEASURED_HELDOUT.value:
            errors.append("passed_after_measured_heldout state requires measured_heldout_pass")
    return errors


def validate_stage3_success_claim_record(
    record: Stage3SuccessClaimRecord | Mapping[str, Any],
    *,
    experiment_run: Optional[Stage3ExperimentRunRecord | Mapping[str, Any]] = None,
    heldout_evaluation: Optional[Stage3HeldoutEvaluationRecord | Mapping[str, Any]] = None,
    plan: Optional[Stage3ExperimentPlanRecord | Mapping[str, Any]] = None,
    confidential_gpu_training: Optional[ConfidentialGPUTrainingRunRecord | Mapping[str, Any]] = None,
    confidential_gpu_data_policy: Optional[ConfidentialGPUDataPolicyRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_stage3_experiment_payload_errors(raw)
    if not isinstance(record, Stage3SuccessClaimRecord):
        try:
            record = Stage3SuccessClaimRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Stage-3 success claim field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Stage-3 success claim payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in Stage3SuccessClaimState}:
        errors.append(f"unknown Stage-3 success claim state: {record.state}")
        return errors
    if record.contract_version != STAGE3_SUCCESS_CLAIM_CONTRACT_VERSION:
        errors.append("contract_version must match Stage-3 success claim contract")
    if not record.success_claim_id.startswith("stage3_success:"):
        errors.append("success_claim_id must be stage3_success:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("experiment_run_ref", ("stage3_experiment:",)),
            ("heldout_eval_ref", ("heldout_eval:",)),
            ("model_pipeline_exit_gate_ref", ("model_pipeline_exit_gate:",)),
            ("confidential_gpu_training_ref", ("confidential_gpu_training:",)),
            ("model_weight_artifact_ref", ("model_weight_artifact:",)),
        ),
    )
    _append_stage3_write_guard_errors(record, "P4.5 Stage-3 success claim", errors)
    if record.production_deployment_requested:
        errors.append("P4.5 must not request production deployment")
    if record.model_weight_publication_enabled:
        errors.append("P4.5 must not enable model-weight publication")
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Stage-3 success claim",
        (
            "stage3_success:",
            "stage3_experiment:",
            "heldout_eval:",
            "model_pipeline_exit_gate:",
            "confidential_gpu_training:",
            "model_weight_artifact:",
            "outcome_label_set:",
            "sequestered_set:",
            "owner_approval:",
        ),
    )
    if experiment_run is not None:
        if not isinstance(experiment_run, Stage3ExperimentRunRecord):
            experiment_run = Stage3ExperimentRunRecord.from_mapping(experiment_run)
        run_errors = validate_stage3_experiment_run_record(
            experiment_run,
            plan=plan,
            confidential_gpu_training=confidential_gpu_training,
            confidential_gpu_data_policy=confidential_gpu_data_policy,
        )
        if run_errors:
            errors.append("source Stage-3 experiment run is invalid: " + "; ".join(run_errors))
        if record.experiment_run_ref != experiment_run.experiment_run_id:
            errors.append("Stage-3 success claim experiment_run_ref mismatch")
    if heldout_evaluation is not None:
        if not isinstance(heldout_evaluation, Stage3HeldoutEvaluationRecord):
            heldout_evaluation = Stage3HeldoutEvaluationRecord.from_mapping(heldout_evaluation)
        eval_errors = validate_stage3_heldout_evaluation_record(
            heldout_evaluation,
            experiment_run=experiment_run,
            plan=plan,
            confidential_gpu_training=confidential_gpu_training,
            confidential_gpu_data_policy=confidential_gpu_data_policy,
        )
        if eval_errors:
            errors.append("source Stage-3 heldout evaluation is invalid: " + "; ".join(eval_errors))
        if record.heldout_eval_ref != heldout_evaluation.heldout_eval_id:
            errors.append("Stage-3 success claim heldout_eval_ref mismatch")
    if record.success_claimed:
        if experiment_run is None:
            errors.append("Stage-3 success requires supplied experiment_run")
        if heldout_evaluation is None:
            errors.append("Stage-3 success requires supplied heldout_evaluation")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("beats_hybrid_champion", "Stage-3 success requires beating the hybrid champion"),
                ("heldout_eval_passed", "Stage-3 success requires passed heldout evaluation"),
                ("outcome_evidence_verified", "Stage-3 success requires verified outcome evidence"),
            ),
        )
        if experiment_run is not None and not experiment_run.production_experiment_run_valid:
            errors.append("Stage-3 success requires production-valid experiment run")
        if heldout_evaluation is not None and not heldout_evaluation.measured_heldout_pass:
            errors.append("Stage-3 success requires measured heldout pass")
        if record.uses_local_fixtures:
            errors.append("Stage-3 success cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Stage-3 success cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("Stage-3 success requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Stage-3 success requires owner_approval_ref")
        if record.state != Stage3SuccessClaimState.READY_AFTER_OWNER_REVIEW.value:
            errors.append("success_claimed requires ready_after_owner_review state")
    else:
        if not record.local_only:
            errors.append("not-claimed Stage-3 success records must remain local_only")
        if record.state == Stage3SuccessClaimState.READY_AFTER_OWNER_REVIEW.value:
            errors.append("ready_after_owner_review state requires success_claimed")
    return errors


def verify_research_lab_stage3_experiment(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    scale_summary = verify_scale_foundation()
    model_pipeline_summary = verify_model_pipeline_exit_gate()
    confidential_gpu_summary = verify_research_lab_confidential_gpu_training()
    fixture = _load_fixture(Path(fixture_path))
    confidential_gpu_fixture = _load_fixture(CONFIDENTIAL_GPU_FIXTURE_PATH)
    local_confidential_gpu_policy = ConfidentialGPUDataPolicyRecord.from_mapping(
        confidential_gpu_fixture["local_data_policy"]
    )
    ready_confidential_gpu_policy = ConfidentialGPUDataPolicyRecord.from_mapping(
        confidential_gpu_fixture["ready_data_policy"]
    )
    local_confidential_gpu_training = ConfidentialGPUTrainingRunRecord.from_mapping(
        confidential_gpu_fixture["local_training_run"]
    )
    ready_confidential_gpu_training = ConfidentialGPUTrainingRunRecord.from_mapping(
        confidential_gpu_fixture["ready_training_control"]
    )

    local_plan = Stage3ExperimentPlanRecord.from_mapping(fixture["local_experiment_plan"])
    _assert(not validate_stage3_experiment_plan_record(local_plan), "local Stage-3 experiment plan validates")
    _assert(not local_plan.plan_approved, "local Stage-3 plan does not claim approval")

    ready_plan = Stage3ExperimentPlanRecord.from_mapping(fixture["ready_experiment_plan"])
    _assert(not validate_stage3_experiment_plan_record(ready_plan), "ready-control Stage-3 plan validates")
    _assert(ready_plan.plan_approved, "ready-control Stage-3 plan is approved")

    local_run = Stage3ExperimentRunRecord.from_mapping(fixture["local_experiment_run"])
    _assert(
        not validate_stage3_experiment_run_record(
            local_run,
            plan=local_plan,
            confidential_gpu_training=local_confidential_gpu_training,
            confidential_gpu_data_policy=local_confidential_gpu_policy,
        ),
        "local Stage-3 experiment run validates",
    )
    _assert(not local_run.production_experiment_run_valid, "local Stage-3 run does not claim production validity")

    ready_run = Stage3ExperimentRunRecord.from_mapping(fixture["ready_experiment_run"])
    _assert(
        not validate_stage3_experiment_run_record(
            ready_run,
            plan=ready_plan,
            confidential_gpu_training=ready_confidential_gpu_training,
            confidential_gpu_data_policy=ready_confidential_gpu_policy,
        ),
        "ready-control Stage-3 experiment run validates",
    )
    _assert(ready_run.production_experiment_run_valid, "ready-control Stage-3 run is production-valid")

    local_eval = Stage3HeldoutEvaluationRecord.from_mapping(fixture["local_heldout_eval"])
    _assert(
        not validate_stage3_heldout_evaluation_record(
            local_eval,
            experiment_run=local_run,
            plan=local_plan,
            confidential_gpu_training=local_confidential_gpu_training,
            confidential_gpu_data_policy=local_confidential_gpu_policy,
        ),
        "local Stage-3 heldout evaluation validates",
    )
    _assert(not local_eval.measured_heldout_pass, "local heldout eval does not claim pass")

    ready_eval = Stage3HeldoutEvaluationRecord.from_mapping(fixture["ready_heldout_eval"])
    _assert(
        not validate_stage3_heldout_evaluation_record(
            ready_eval,
            experiment_run=ready_run,
            plan=ready_plan,
            confidential_gpu_training=ready_confidential_gpu_training,
            confidential_gpu_data_policy=ready_confidential_gpu_policy,
        ),
        "ready-control Stage-3 heldout evaluation validates",
    )
    _assert(ready_eval.measured_heldout_pass, "ready-control heldout eval passes")

    local_success = Stage3SuccessClaimRecord.from_mapping(fixture["local_success_claim"])
    _assert(
        not validate_stage3_success_claim_record(
            local_success,
            experiment_run=local_run,
            heldout_evaluation=local_eval,
            plan=local_plan,
            confidential_gpu_training=local_confidential_gpu_training,
            confidential_gpu_data_policy=local_confidential_gpu_policy,
        ),
        "local Stage-3 success claim validates",
    )
    _assert(not local_success.success_claimed, "local success claim stays false")

    ready_success = Stage3SuccessClaimRecord.from_mapping(fixture["ready_success_claim"])
    _assert(
        not validate_stage3_success_claim_record(
            ready_success,
            experiment_run=ready_run,
            heldout_evaluation=ready_eval,
            plan=ready_plan,
            confidential_gpu_training=ready_confidential_gpu_training,
            confidential_gpu_data_policy=ready_confidential_gpu_policy,
        ),
        "ready-control Stage-3 success claim validates",
    )
    _assert(ready_success.success_claimed, "ready-control success claim is true")

    for invalid in fixture["invalid_experiment_plans"]:
        base = fixture[str(invalid.get("base", "local_experiment_plan"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_stage3_experiment_plan_record(merged)
        _assert(errors, f"invalid Stage-3 experiment plan fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_experiment_runs"]:
        base = fixture[str(invalid.get("base", "local_experiment_run"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        if invalid.get("omit_plan"):
            errors = validate_stage3_experiment_run_record(merged)
        else:
            plan = Stage3ExperimentPlanRecord.from_mapping(fixture[str(invalid.get("plan_base", "local_experiment_plan"))])
            training = None
            data_policy = None
            if not invalid.get("omit_confidential_gpu_training"):
                training_base_name = str(invalid.get("confidential_gpu_training_base", "local_training_run"))
                training = ConfidentialGPUTrainingRunRecord.from_mapping(confidential_gpu_fixture[training_base_name])
                if not invalid.get("omit_confidential_gpu_data_policy"):
                    policy_base_name = str(
                        invalid.get(
                            "confidential_gpu_policy_base",
                            "ready_data_policy" if training_base_name == "ready_training_control" else "local_data_policy",
                        )
                    )
                    data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(
                        confidential_gpu_fixture[policy_base_name]
                    )
            elif not invalid.get("omit_confidential_gpu_data_policy"):
                data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(
                    confidential_gpu_fixture[str(invalid.get("confidential_gpu_policy_base", "local_data_policy"))]
                )
            errors = validate_stage3_experiment_run_record(
                merged,
                plan=plan,
                confidential_gpu_training=training,
                confidential_gpu_data_policy=data_policy,
            )
        _assert(errors, f"invalid Stage-3 experiment run fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_run_errors = validate_stage3_experiment_run_record(
        local_run,
        plan=local_plan,
        confidential_gpu_training=local_confidential_gpu_training,
        confidential_gpu_data_policy=local_confidential_gpu_policy,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_run_errors, "unsafe Phase 4 guards block Stage-3 experiment validation")
    _assert_expected_error(unsafe_run_errors, fixture["unsafe_scale_guards"])

    unsafe_eval_errors = validate_stage3_heldout_evaluation_record(
        local_eval,
        experiment_run=local_run,
        plan=local_plan,
        confidential_gpu_training=local_confidential_gpu_training,
        confidential_gpu_data_policy=local_confidential_gpu_policy,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_eval_errors, "unsafe Phase 4 guards block Stage-3 heldout validation")
    _assert_expected_error(unsafe_eval_errors, fixture["unsafe_scale_guards"])

    for invalid in fixture["invalid_heldout_evals"]:
        base = fixture[str(invalid.get("base", "local_heldout_eval"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        if invalid.get("omit_experiment_run"):
            errors = validate_stage3_heldout_evaluation_record(merged)
        else:
            run_base_name = str(invalid.get("run_base", "local_experiment_run"))
            run = Stage3ExperimentRunRecord.from_mapping(fixture[run_base_name])
            plan = Stage3ExperimentPlanRecord.from_mapping(
                fixture[str(invalid.get("plan_base", "ready_experiment_plan" if run_base_name == "ready_experiment_run" else "local_experiment_plan"))]
            )
            training_base_name = str(
                invalid.get(
                    "confidential_gpu_training_base",
                    "ready_training_control" if run_base_name == "ready_experiment_run" else "local_training_run",
                )
            )
            training = ConfidentialGPUTrainingRunRecord.from_mapping(confidential_gpu_fixture[training_base_name])
            policy_base_name = str(
                invalid.get(
                    "confidential_gpu_policy_base",
                    "ready_data_policy" if training_base_name == "ready_training_control" else "local_data_policy",
                )
            )
            data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(confidential_gpu_fixture[policy_base_name])
            errors = validate_stage3_heldout_evaluation_record(
                merged,
                experiment_run=run,
                plan=plan,
                confidential_gpu_training=training,
                confidential_gpu_data_policy=data_policy,
            )
        _assert(errors, f"invalid Stage-3 heldout evaluation fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_success_claims"]:
        base = fixture[str(invalid.get("base", "local_success_claim"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        plan = None
        run = None
        eval_record = None
        if not invalid.get("omit_plan"):
            plan = Stage3ExperimentPlanRecord.from_mapping(
                fixture[str(invalid.get("plan_base", "ready_experiment_plan"))]
            )
        if not invalid.get("omit_experiment_run"):
            run = Stage3ExperimentRunRecord.from_mapping(
                fixture[str(invalid.get("run_base", "ready_experiment_run"))]
            )
        if not invalid.get("omit_heldout_eval"):
            eval_record = Stage3HeldoutEvaluationRecord.from_mapping(
                fixture[str(invalid.get("eval_base", "ready_heldout_eval"))]
            )
        training_base_name = str(invalid.get("confidential_gpu_training_base", "ready_training_control"))
        training = ConfidentialGPUTrainingRunRecord.from_mapping(confidential_gpu_fixture[training_base_name])
        policy_base_name = str(
            invalid.get(
                "confidential_gpu_policy_base",
                "ready_data_policy" if training_base_name == "ready_training_control" else "local_data_policy",
            )
        )
        data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(confidential_gpu_fixture[policy_base_name])
        errors = validate_stage3_success_claim_record(
            merged,
            experiment_run=run,
            heldout_evaluation=eval_record,
            plan=plan,
            confidential_gpu_training=training,
            confidential_gpu_data_policy=data_policy,
        )
        _assert(errors, f"invalid Stage-3 success claim fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"stage3_experiment_evidence_ready": False}, ScaleGate.STAGE3_EXPERIMENT_EVIDENCE_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.5 Stage-3 experiment evidence gate remains required by P4.0")

    return {
        "scale_readiness_id": scale_summary["readiness_id"],
        "model_pipeline_exit_gate_id": model_pipeline_summary["exit_gate_id"],
        "confidential_gpu_training_id": confidential_gpu_summary["training_run_id"],
        "experiment_plan_id": local_plan.plan_id,
        "plan_approved": local_plan.plan_approved,
        "experiment_run_id": local_run.experiment_run_id,
        "production_experiment_run_valid": local_run.production_experiment_run_valid,
        "heldout_eval_id": local_eval.heldout_eval_id,
        "measured_heldout_pass": local_eval.measured_heldout_pass,
        "success_claim_id": local_success.success_claim_id,
        "success_claimed": local_success.success_claimed,
        "ready_control_validates": (
            ready_plan.plan_approved
            and ready_run.production_experiment_run_valid
            and ready_eval.measured_heldout_pass
            and ready_success.success_claimed
        ),
    }


def _append_scale_guard_errors(
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]],
    errors: list[str],
) -> None:
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))


def _append_prefix_errors(record: Any, errors: list[str], checks: Sequence[tuple[str, tuple[str, ...]]]) -> None:
    for field, prefixes in checks:
        if not getattr(record, field).startswith(prefixes):
            if len(prefixes) == 1:
                errors.append(f"{field} must be {prefixes[0]}-prefixed")
            else:
                errors.append(f"{field} must use an approved prefix")


def _append_evidence_prefix_errors(
    evidence_refs: Sequence[str],
    errors: list[str],
    label: str,
    prefixes: tuple[str, ...],
) -> None:
    for evidence_ref in evidence_refs:
        if not evidence_ref.startswith(prefixes):
            errors.append(f"{label} evidence_refs must use approved prefixes")
            break


def _append_missing_true_flags(record: Any, errors: list[str], checks: Sequence[tuple[str, str]]) -> None:
    for field, message in checks:
        if not bool(getattr(record, field)):
            errors.append(message)


def _append_stage3_write_guard_errors(record: Any, label: str, errors: list[str]) -> None:
    if bool(getattr(record, "production_writes", False)):
        errors.append(f"{label} must not enable production writes")
    if bool(getattr(record, "supabase_writes", False)):
        errors.append(f"{label} must not enable Supabase writes")
    if bool(getattr(record, "public_workflows", False)):
        errors.append(f"{label} must not enable public workflows")


def _protected_stage3_experiment_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_stage3_experiment_material(record))
    if not found:
        return []
    return ["P4.5 Stage-3 experiment payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_stage3_experiment_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_STAGE3_EXPERIMENT_KEYS and not key_text.endswith(
                ("_ref", "_refs", "_hash", "_hashes")
            ):
                found.add(key_path)
            found.update(_find_protected_stage3_experiment_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(_find_protected_stage3_experiment_material(nested, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_STAGE3_EXPERIMENT_MARKERS:
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
