"""Phase 3.5 stage-2 reranker distillation contracts.

P3.5 prepares local contracts for distilling the high-volume reranker path. It
does not train a student, publish weights, write registries, request
promotion, claim parity/cost success, or write production state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .trajectory_corpus import (
    MIN_HOLDOUT_TRAJECTORIES,
    MIN_TRAINING_TRAJECTORIES,
    PROTECTED_CORPUS_KEYS,
    PROTECTED_CORPUS_MARKERS,
    TRAJECTORY_CORPUS_CONTRACT_VERSION,
    verify_research_lab_trajectory_corpus,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "reranker_distillation_fixtures.json"

RERANKER_DISTILLATION_DATASET_CONTRACT_VERSION = "reranker_distillation_dataset:v1:local_contract"
RERANKER_DISTILLATION_TRAINING_CONTRACT_VERSION = "reranker_distillation_training:v1:local_contract"
RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION = "reranker_distillation_eval:v1:local_contract"
RERANKER_PARITY_THRESHOLD_PCT = 98.0
RERANKER_COST_RATIO_MAX_PCT = 30.0
PENDING_RERANKER_PARITY_REF = "reranker_parity:pending"

MIN_RERANKER_DISTILLATION_TRAIN_ROWS = MIN_TRAINING_TRAJECTORIES
MIN_RERANKER_DISTILLATION_HOLDOUT_ROWS = MIN_HOLDOUT_TRAJECTORIES

PROTECTED_RERANKER_DISTILLATION_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_CORPUS_KEYS)
    | {
        "answer_key",
        "raw_teacher_trace",
        "raw_teacher_output",
        "raw_student_output",
        "raw_reranker_label",
        "teacher_prompt",
        "teacher_policy_secret",
        "student_training_payload",
        "student_model_weights",
        "sealed_reranker_eval",
        "private_customer_record",
        "training_split_task",
    }
)

PROTECTED_RERANKER_DISTILLATION_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_CORPUS_MARKERS)
        | {
            "answer key",
            "private customer record",
            "raw reranker label",
            "raw student output",
            "raw teacher output",
            "raw teacher trace",
            "sealed reranker eval",
            "student model weights",
            "student training payload",
            "teacher policy secret",
            "teacher prompt",
            "training split task",
        }
    )
)


class RerankerDistillationDatasetState(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    READY_AFTER_CORPUS = "ready_after_corpus"
    BLOCKED = "blocked"


class RerankerDistillationTrainingState(str, Enum):
    LOCAL_TRAINING_STUB = "local_training_stub"
    TRAINED_AWAITING_EVAL = "trained_awaiting_eval"
    READY_AFTER_PARITY_EVAL = "ready_after_parity_eval"
    BLOCKED = "blocked"


class RerankerEvalArmKind(str, Enum):
    TEACHER = "teacher_reranker"
    STUDENT = "student_reranker"


class RerankerEvalDataState(str, Enum):
    LOCAL_FIXTURE = "local_fixture"
    MEASURED_LAB_ONLY = "measured_lab_only"
    MEASURED_PRODUCTION = "measured_production"
    BLOCKED = "blocked"


class RerankerParityState(str, Enum):
    LOCAL_STUB = "local_stub"
    READY_AFTER_MEASURED_EVIDENCE = "ready_after_measured_evidence"
    PASSED_PARITY_COST_GATE = "passed_parity_cost_gate"
    FAILED_PARITY_COST_GATE = "failed_parity_cost_gate"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RerankerDistillationDatasetManifestRecord:
    dataset_id: str
    source_corpus_ref: str
    source_corpus_hash: str
    source_contract_version: str
    source_record_hashes: tuple[str, ...]
    train_count: int
    validation_count: int
    holdout_count: int
    teacher_policy_ref: str
    teacher_trace_refs: tuple[str, ...]
    reranker_label_set_ref: str
    split_policy_ref: str
    dataset_hash: str
    dataset_card_ref: str
    rights_summary_ref: str
    distillation_rights_ref: str
    protected_filter_ref: str
    pii_review_ref: str
    label_quality_report_ref: str
    teacher_policy_compliant: bool = False
    clean_reranker_labels: bool = False
    distillation_rights_verified: bool = False
    protected_data_excluded: bool = False
    pii_review_passed: bool = False
    holdout_locked: bool = False
    local_only: bool = True
    uses_local_fixtures: bool = True
    dataset_ready_claimed: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = RerankerDistillationDatasetState.LOCAL_CONTRACT_STUB.value
    contract_version: str = RERANKER_DISTILLATION_DATASET_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RerankerDistillationDatasetManifestRecord":
        return cls(
            dataset_id=str(data["dataset_id"]),
            source_corpus_ref=str(data["source_corpus_ref"]),
            source_corpus_hash=str(data["source_corpus_hash"]),
            source_contract_version=str(data.get("source_contract_version", "")),
            source_record_hashes=tuple(str(item) for item in data.get("source_record_hashes", [])),
            train_count=int(data.get("train_count", 0)),
            validation_count=int(data.get("validation_count", 0)),
            holdout_count=int(data.get("holdout_count", 0)),
            teacher_policy_ref=str(data.get("teacher_policy_ref", "")),
            teacher_trace_refs=tuple(str(item) for item in data.get("teacher_trace_refs", [])),
            reranker_label_set_ref=str(data.get("reranker_label_set_ref", "")),
            split_policy_ref=str(data.get("split_policy_ref", "")),
            dataset_hash=str(data.get("dataset_hash", "")),
            dataset_card_ref=str(data.get("dataset_card_ref", "")),
            rights_summary_ref=str(data.get("rights_summary_ref", "")),
            distillation_rights_ref=str(data.get("distillation_rights_ref", "")),
            protected_filter_ref=str(data.get("protected_filter_ref", "")),
            pii_review_ref=str(data.get("pii_review_ref", "")),
            label_quality_report_ref=str(data.get("label_quality_report_ref", "")),
            teacher_policy_compliant=bool(data.get("teacher_policy_compliant", False)),
            clean_reranker_labels=bool(data.get("clean_reranker_labels", False)),
            distillation_rights_verified=bool(data.get("distillation_rights_verified", False)),
            protected_data_excluded=bool(data.get("protected_data_excluded", False)),
            pii_review_passed=bool(data.get("pii_review_passed", False)),
            holdout_locked=bool(data.get("holdout_locked", False)),
            local_only=bool(data.get("local_only", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            dataset_ready_claimed=bool(data.get("dataset_ready_claimed", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", RerankerDistillationDatasetState.LOCAL_CONTRACT_STUB.value)),
            contract_version=str(data.get("contract_version", RERANKER_DISTILLATION_DATASET_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("source_record_hashes", "teacher_trace_refs", "evidence_refs"):
            data[key] = list(getattr(self, key))
        return data


@dataclass(frozen=True)
class RerankerDistillationTrainingRunRecord:
    training_run_id: str
    dataset_ref: str
    dataset_hash: str
    teacher_policy_ref: str
    student_model_ref: str
    student_model_hash_ref: str
    training_plan_ref: str
    hyperparam_config_hash: str
    cost_cap_ref: str
    parity_eval_ref: str = PENDING_RERANKER_PARITY_REF
    training_enabled: bool = False
    training_started: bool = False
    training_completed: bool = False
    student_model_weights_published: bool = False
    model_registry_write: bool = False
    production_promotion_requested: bool = False
    parity_claimed: bool = False
    parity_eval_passed: bool = False
    measured_quality_retention_pct: float = 0.0
    cost_ratio_to_teacher_pct: float = 0.0
    local_only: bool = True
    uses_local_fixtures: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = RerankerDistillationTrainingState.LOCAL_TRAINING_STUB.value
    contract_version: str = RERANKER_DISTILLATION_TRAINING_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RerankerDistillationTrainingRunRecord":
        return cls(
            training_run_id=str(data["training_run_id"]),
            dataset_ref=str(data["dataset_ref"]),
            dataset_hash=str(data["dataset_hash"]),
            teacher_policy_ref=str(data.get("teacher_policy_ref", "")),
            student_model_ref=str(data["student_model_ref"]),
            student_model_hash_ref=str(data.get("student_model_hash_ref", "")),
            training_plan_ref=str(data.get("training_plan_ref", "")),
            hyperparam_config_hash=str(data.get("hyperparam_config_hash", "")),
            cost_cap_ref=str(data.get("cost_cap_ref", "")),
            parity_eval_ref=str(data.get("parity_eval_ref", PENDING_RERANKER_PARITY_REF)),
            training_enabled=bool(data.get("training_enabled", False)),
            training_started=bool(data.get("training_started", False)),
            training_completed=bool(data.get("training_completed", False)),
            student_model_weights_published=bool(data.get("student_model_weights_published", False)),
            model_registry_write=bool(data.get("model_registry_write", False)),
            production_promotion_requested=bool(data.get("production_promotion_requested", False)),
            parity_claimed=bool(data.get("parity_claimed", False)),
            parity_eval_passed=bool(data.get("parity_eval_passed", False)),
            measured_quality_retention_pct=float(data.get("measured_quality_retention_pct", 0.0)),
            cost_ratio_to_teacher_pct=float(data.get("cost_ratio_to_teacher_pct", 0.0)),
            local_only=bool(data.get("local_only", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", RerankerDistillationTrainingState.LOCAL_TRAINING_STUB.value)),
            contract_version=str(data.get("contract_version", RERANKER_DISTILLATION_TRAINING_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class RerankerEvalArmRecord:
    eval_id: str
    arm_kind: str
    model_ref: str
    dataset_ref: str
    dataset_hash: str
    holdout_split_ref: str
    eval_task_refs: tuple[str, ...]
    quality_score: float
    cost_cents: int
    latency_ms: int
    run_count: int
    receipt_refs: tuple[str, ...]
    cost_ledger_refs: tuple[str, ...]
    teacher_policy_ref: str
    heldout_eval_only: bool = True
    uses_training_split: bool = False
    measured_data_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    source_data_state: str = RerankerEvalDataState.LOCAL_FIXTURE.value
    contract_version: str = RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RerankerEvalArmRecord":
        return cls(
            eval_id=str(data["eval_id"]),
            arm_kind=str(data["arm_kind"]),
            model_ref=str(data["model_ref"]),
            dataset_ref=str(data["dataset_ref"]),
            dataset_hash=str(data["dataset_hash"]),
            holdout_split_ref=str(data["holdout_split_ref"]),
            eval_task_refs=tuple(str(item) for item in data.get("eval_task_refs", [])),
            quality_score=float(data["quality_score"]),
            cost_cents=int(data["cost_cents"]),
            latency_ms=int(data.get("latency_ms", 0)),
            run_count=int(data["run_count"]),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            cost_ledger_refs=tuple(str(item) for item in data.get("cost_ledger_refs", [])),
            teacher_policy_ref=str(data.get("teacher_policy_ref", "")),
            heldout_eval_only=bool(data.get("heldout_eval_only", True)),
            uses_training_split=bool(data.get("uses_training_split", False)),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            source_data_state=str(data.get("source_data_state", RerankerEvalDataState.LOCAL_FIXTURE.value)),
            contract_version=str(data.get("contract_version", RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("eval_task_refs", "receipt_refs", "cost_ledger_refs"):
            data[key] = list(getattr(self, key))
        return data

    def identity_payload(self) -> dict[str, Any]:
        return {
            "arm_kind": self.arm_kind,
            "model_ref": self.model_ref,
            "dataset_ref": self.dataset_ref,
            "dataset_hash": self.dataset_hash,
            "holdout_split_ref": self.holdout_split_ref,
            "eval_task_refs": list(self.eval_task_refs),
            "quality_score": self.quality_score,
            "cost_cents": self.cost_cents,
            "latency_ms": self.latency_ms,
            "run_count": self.run_count,
            "receipt_refs": list(self.receipt_refs),
            "cost_ledger_refs": list(self.cost_ledger_refs),
            "teacher_policy_ref": self.teacher_policy_ref,
            "heldout_eval_only": self.heldout_eval_only,
            "uses_training_split": self.uses_training_split,
            "source_data_state": self.source_data_state,
        }


@dataclass(frozen=True)
class RerankerParityComparisonRecord:
    comparison_id: str
    teacher_eval_ref: str
    student_eval_ref: str
    training_run_ref: str
    dataset_ref: str
    dataset_hash: str
    holdout_split_ref: str
    eval_task_refs: tuple[str, ...]
    teacher_quality_score: float
    student_quality_score: float
    quality_retention_pct: float
    quality_delta_pct: float
    teacher_cost_cents: int
    student_cost_cents: int
    cost_ratio_to_teacher_pct: float
    cost_savings_pct: float
    quality_parity_passed: bool
    cost_gate_passed: bool
    passed_parity_cost_gate: bool
    methodology_ref: str
    measured_data_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    parity_claimed: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    student_model_registry_write: bool = False
    student_model_weights_published: bool = False
    production_promotion_requested: bool = False
    model_pipeline_exit_claimed: bool = False
    public_success_claim_enabled: bool = False
    production_writes: bool = False
    state: str = RerankerParityState.LOCAL_STUB.value
    contract_version: str = RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RerankerParityComparisonRecord":
        return cls(
            comparison_id=str(data["comparison_id"]),
            teacher_eval_ref=str(data["teacher_eval_ref"]),
            student_eval_ref=str(data["student_eval_ref"]),
            training_run_ref=str(data["training_run_ref"]),
            dataset_ref=str(data["dataset_ref"]),
            dataset_hash=str(data["dataset_hash"]),
            holdout_split_ref=str(data["holdout_split_ref"]),
            eval_task_refs=tuple(str(item) for item in data.get("eval_task_refs", [])),
            teacher_quality_score=float(data["teacher_quality_score"]),
            student_quality_score=float(data["student_quality_score"]),
            quality_retention_pct=float(data["quality_retention_pct"]),
            quality_delta_pct=float(data["quality_delta_pct"]),
            teacher_cost_cents=int(data["teacher_cost_cents"]),
            student_cost_cents=int(data["student_cost_cents"]),
            cost_ratio_to_teacher_pct=float(data["cost_ratio_to_teacher_pct"]),
            cost_savings_pct=float(data["cost_savings_pct"]),
            quality_parity_passed=bool(data.get("quality_parity_passed", False)),
            cost_gate_passed=bool(data.get("cost_gate_passed", False)),
            passed_parity_cost_gate=bool(data.get("passed_parity_cost_gate", False)),
            methodology_ref=str(data["methodology_ref"]),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            parity_claimed=bool(data.get("parity_claimed", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            student_model_registry_write=bool(data.get("student_model_registry_write", False)),
            student_model_weights_published=bool(data.get("student_model_weights_published", False)),
            production_promotion_requested=bool(data.get("production_promotion_requested", False)),
            model_pipeline_exit_claimed=bool(data.get("model_pipeline_exit_claimed", False)),
            public_success_claim_enabled=bool(data.get("public_success_claim_enabled", False)),
            production_writes=bool(data.get("production_writes", False)),
            state=str(data.get("state", RerankerParityState.LOCAL_STUB.value)),
            contract_version=str(data.get("contract_version", RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eval_task_refs"] = list(self.eval_task_refs)
        data["evidence_refs"] = list(self.evidence_refs)
        return data

    def identity_payload(self) -> dict[str, Any]:
        return {
            "teacher_eval_ref": self.teacher_eval_ref,
            "student_eval_ref": self.student_eval_ref,
            "training_run_ref": self.training_run_ref,
            "dataset_ref": self.dataset_ref,
            "dataset_hash": self.dataset_hash,
            "holdout_split_ref": self.holdout_split_ref,
            "eval_task_refs": list(self.eval_task_refs),
            "teacher_quality_score": self.teacher_quality_score,
            "student_quality_score": self.student_quality_score,
            "quality_retention_pct": self.quality_retention_pct,
            "quality_delta_pct": self.quality_delta_pct,
            "teacher_cost_cents": self.teacher_cost_cents,
            "student_cost_cents": self.student_cost_cents,
            "cost_ratio_to_teacher_pct": self.cost_ratio_to_teacher_pct,
            "cost_savings_pct": self.cost_savings_pct,
            "passed_parity_cost_gate": self.passed_parity_cost_gate,
            "methodology_ref": self.methodology_ref,
            "measured_data_ready": self.measured_data_ready,
        }


def reranker_distillation_dataset_hash(
    record: RerankerDistillationDatasetManifestRecord | Mapping[str, Any],
) -> str:
    if not isinstance(record, RerankerDistillationDatasetManifestRecord):
        record = RerankerDistillationDatasetManifestRecord.from_mapping(record)
    payload = record.to_dict()
    payload["dataset_hash"] = "sha256:pending"
    return sha256_json(payload)


def calculate_quality_retention_pct(student_quality: float, teacher_quality: float) -> float:
    if teacher_quality <= 0:
        raise ValueError("teacher_quality must be positive")
    return round(float(student_quality) / float(teacher_quality) * 100.0, 6)


def calculate_quality_delta_pct(student_quality: float, teacher_quality: float) -> float:
    if teacher_quality <= 0:
        raise ValueError("teacher_quality must be positive")
    return round((float(student_quality) - float(teacher_quality)) / float(teacher_quality) * 100.0, 6)


def calculate_cost_ratio_to_teacher_pct(student_cost_cents: int, teacher_cost_cents: int) -> float:
    if teacher_cost_cents <= 0:
        raise ValueError("teacher_cost_cents must be positive")
    return round(int(student_cost_cents) / int(teacher_cost_cents) * 100.0, 6)


def build_reranker_distillation_dataset_manifest(
    *,
    dataset_id: str,
    source_corpus_ref: str,
    source_corpus_hash: str,
    source_record_hashes: Sequence[str],
    train_count: int,
    validation_count: int,
    holdout_count: int,
    teacher_policy_ref: str,
    teacher_trace_refs: Sequence[str],
    reranker_label_set_ref: str,
    split_policy_ref: str,
    dataset_card_ref: str,
    rights_summary_ref: str,
    distillation_rights_ref: str,
    protected_filter_ref: str,
    pii_review_ref: str,
    label_quality_report_ref: str,
    uses_local_fixtures: bool = True,
    local_only: bool = True,
) -> RerankerDistillationDatasetManifestRecord:
    manifest = RerankerDistillationDatasetManifestRecord(
        dataset_id=dataset_id,
        source_corpus_ref=source_corpus_ref,
        source_corpus_hash=source_corpus_hash,
        source_contract_version=TRAJECTORY_CORPUS_CONTRACT_VERSION,
        source_record_hashes=tuple(str(item) for item in source_record_hashes),
        train_count=int(train_count),
        validation_count=int(validation_count),
        holdout_count=int(holdout_count),
        teacher_policy_ref=teacher_policy_ref,
        teacher_trace_refs=tuple(str(item) for item in teacher_trace_refs),
        reranker_label_set_ref=reranker_label_set_ref,
        split_policy_ref=split_policy_ref,
        dataset_hash="sha256:pending",
        dataset_card_ref=dataset_card_ref,
        rights_summary_ref=rights_summary_ref,
        distillation_rights_ref=distillation_rights_ref,
        protected_filter_ref=protected_filter_ref,
        pii_review_ref=pii_review_ref,
        label_quality_report_ref=label_quality_report_ref,
        local_only=local_only,
        uses_local_fixtures=uses_local_fixtures,
        dataset_ready_claimed=False,
        state=RerankerDistillationDatasetState.LOCAL_CONTRACT_STUB.value,
    )
    data = manifest.to_dict()
    data["dataset_hash"] = reranker_distillation_dataset_hash(manifest)
    return RerankerDistillationDatasetManifestRecord.from_mapping(data)


def build_reranker_eval_arm(**kwargs: Any) -> RerankerEvalArmRecord:
    record = RerankerEvalArmRecord.from_mapping({"eval_id": "reranker_eval:pending", **kwargs})
    errors = validate_reranker_eval_arm_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    data = record.to_dict()
    data["eval_id"] = "reranker_eval:" + sha256_json(record.identity_payload()).split(":", 1)[1][:16]
    return RerankerEvalArmRecord.from_mapping(data)


def build_reranker_parity_comparison(
    *,
    teacher_eval: RerankerEvalArmRecord | Mapping[str, Any],
    student_eval: RerankerEvalArmRecord | Mapping[str, Any],
    training_run: RerankerDistillationTrainingRunRecord | Mapping[str, Any],
    methodology_ref: str,
) -> RerankerParityComparisonRecord:
    if not isinstance(teacher_eval, RerankerEvalArmRecord):
        teacher_eval = RerankerEvalArmRecord.from_mapping(teacher_eval)
    if not isinstance(student_eval, RerankerEvalArmRecord):
        student_eval = RerankerEvalArmRecord.from_mapping(student_eval)
    if not isinstance(training_run, RerankerDistillationTrainingRunRecord):
        training_run = RerankerDistillationTrainingRunRecord.from_mapping(training_run)
    errors = (
        validate_reranker_eval_arm_record(teacher_eval)
        + validate_reranker_eval_arm_record(student_eval)
        + validate_reranker_distillation_training_run(training_run)
    )
    if errors:
        raise ValueError("; ".join(errors))
    retention = calculate_quality_retention_pct(student_eval.quality_score, teacher_eval.quality_score)
    quality_delta = calculate_quality_delta_pct(student_eval.quality_score, teacher_eval.quality_score)
    cost_ratio = calculate_cost_ratio_to_teacher_pct(student_eval.cost_cents, teacher_eval.cost_cents)
    cost_savings = round(100.0 - cost_ratio, 6)
    quality_pass = retention >= RERANKER_PARITY_THRESHOLD_PCT
    cost_pass = cost_ratio <= RERANKER_COST_RATIO_MAX_PCT
    draft = RerankerParityComparisonRecord(
        comparison_id=PENDING_RERANKER_PARITY_REF,
        teacher_eval_ref=teacher_eval.eval_id,
        student_eval_ref=student_eval.eval_id,
        training_run_ref=training_run.training_run_id,
        dataset_ref=student_eval.dataset_ref,
        dataset_hash=student_eval.dataset_hash,
        holdout_split_ref=student_eval.holdout_split_ref,
        eval_task_refs=student_eval.eval_task_refs,
        teacher_quality_score=teacher_eval.quality_score,
        student_quality_score=student_eval.quality_score,
        quality_retention_pct=retention,
        quality_delta_pct=quality_delta,
        teacher_cost_cents=teacher_eval.cost_cents,
        student_cost_cents=student_eval.cost_cents,
        cost_ratio_to_teacher_pct=cost_ratio,
        cost_savings_pct=cost_savings,
        quality_parity_passed=quality_pass,
        cost_gate_passed=cost_pass,
        passed_parity_cost_gate=quality_pass and cost_pass,
        methodology_ref=methodology_ref,
        measured_data_ready=teacher_eval.measured_data_ready and student_eval.measured_data_ready,
        uses_local_fixtures=teacher_eval.uses_local_fixtures or student_eval.uses_local_fixtures,
        local_only=teacher_eval.local_only or student_eval.local_only,
        state=RerankerParityState.LOCAL_STUB.value,
    )
    data = draft.to_dict()
    data["comparison_id"] = "reranker_parity:" + sha256_json(draft.identity_payload()).split(":", 1)[1][:16]
    record = RerankerParityComparisonRecord.from_mapping(data)
    errors = validate_reranker_parity_comparison_record(
        record,
        teacher_eval=teacher_eval,
        student_eval=student_eval,
        training_run=training_run,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_reranker_distillation_dataset_manifest(
    record: RerankerDistillationDatasetManifestRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_reranker_payload_errors(raw)
    if not isinstance(record, RerankerDistillationDatasetManifestRecord):
        try:
            record = RerankerDistillationDatasetManifestRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required reranker distillation dataset field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid reranker distillation dataset field value: {exc}")
            return errors
    if record.contract_version != RERANKER_DISTILLATION_DATASET_CONTRACT_VERSION:
        errors.append("contract_version must match P3.5 reranker distillation dataset contract")
    if record.state not in {state.value for state in RerankerDistillationDatasetState}:
        errors.append(f"unknown reranker distillation dataset state: {record.state}")
    if not record.dataset_id.startswith("reranker_distillation_dataset:"):
        errors.append("dataset_id must be reranker_distillation_dataset:-prefixed")
    if not record.source_corpus_ref.startswith("trajectory_corpus:"):
        errors.append("source_corpus_ref must be trajectory_corpus:-prefixed")
    if not record.source_corpus_hash.startswith("sha256:"):
        errors.append("source_corpus_hash must be sha256:-prefixed")
    if record.source_contract_version != TRAJECTORY_CORPUS_CONTRACT_VERSION:
        errors.append("source_contract_version must match P3.1 trajectory corpus contract")
    if record.train_count < 0 or record.validation_count < 0 or record.holdout_count < 0:
        errors.append("distillation dataset split counts must be non-negative")
    if record.holdout_count < 1:
        errors.append("reranker distillation dataset requires a holdout split")
    if not record.source_record_hashes:
        errors.append("reranker distillation dataset requires source_record_hashes")
    _validate_prefixes(errors, record.source_record_hashes, "source_record_hashes", ("sha256:",))
    if not record.teacher_policy_ref.startswith("teacher_policy:"):
        errors.append("teacher_policy_ref must be teacher_policy:-prefixed")
    if not record.teacher_trace_refs:
        errors.append("reranker distillation dataset requires teacher_trace_refs")
    _validate_prefixes(errors, record.teacher_trace_refs, "teacher_trace_refs", ("execution_trace:", "teacher_trace:"))
    for field, prefix in (
        ("reranker_label_set_ref", "reranker_label_set:"),
        ("split_policy_ref", "split_policy:"),
        ("dataset_card_ref", "dataset_card:"),
        ("rights_summary_ref", "rights_summary:"),
        ("distillation_rights_ref", "distillation_rights:"),
        ("protected_filter_ref", "protected_filter:"),
        ("pii_review_ref", "pii_review:"),
        ("label_quality_report_ref", "label_quality_report:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    expected_hash = reranker_distillation_dataset_hash(record)
    if record.dataset_hash != expected_hash:
        errors.append("dataset_hash must match reranker distillation dataset manifest")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "trajectory_corpus:",
                "reranker_distillation_dataset:",
                "teacher_policy:",
                "reranker_label_set:",
                "distillation_rights:",
                "protected_filter:",
                "label_quality_report:",
                "owner_approval:",
            )
        ):
            errors.append("reranker distillation dataset evidence_refs use an unsupported prefix")
            break
    if record.dataset_ready_claimed or record.state == RerankerDistillationDatasetState.READY_AFTER_CORPUS.value:
        if record.uses_local_fixtures:
            errors.append("reranker distillation dataset readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("reranker distillation dataset readiness cannot be claimed by a local_only record")
        if record.train_count < MIN_RERANKER_DISTILLATION_TRAIN_ROWS:
            errors.append("reranker distillation readiness requires minimum train rows")
        if record.holdout_count < MIN_RERANKER_DISTILLATION_HOLDOUT_ROWS:
            errors.append("reranker distillation readiness requires minimum holdout rows")
        if not record.teacher_policy_compliant:
            errors.append("reranker distillation readiness requires teacher_policy_compliant")
        if not record.clean_reranker_labels:
            errors.append("reranker distillation readiness requires clean_reranker_labels")
        if not record.distillation_rights_verified:
            errors.append("reranker distillation readiness requires distillation_rights_verified")
        if not record.protected_data_excluded:
            errors.append("reranker distillation readiness requires protected_data_excluded")
        if not record.pii_review_passed:
            errors.append("reranker distillation readiness requires pii_review_passed")
        if not record.holdout_locked:
            errors.append("reranker distillation readiness requires holdout_locked")
        if not record.owner_approval_ref:
            errors.append("reranker distillation readiness requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("reranker distillation readiness requires evidence_refs")
    else:
        if not record.local_only:
            errors.append("not-ready reranker distillation datasets must remain local_only")
    return errors


def validate_reranker_distillation_training_run(
    record: RerankerDistillationTrainingRunRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_reranker_payload_errors(raw)
    if not isinstance(record, RerankerDistillationTrainingRunRecord):
        try:
            record = RerankerDistillationTrainingRunRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required reranker distillation training field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid reranker distillation training field value: {exc}")
            return errors
    if record.contract_version != RERANKER_DISTILLATION_TRAINING_CONTRACT_VERSION:
        errors.append("contract_version must match P3.5 reranker distillation training contract")
    if record.state not in {state.value for state in RerankerDistillationTrainingState}:
        errors.append(f"unknown reranker distillation training state: {record.state}")
    if not record.training_run_id.startswith("reranker_distillation_run:"):
        errors.append("training_run_id must be reranker_distillation_run:-prefixed")
    if not record.dataset_ref.startswith("reranker_distillation_dataset:"):
        errors.append("dataset_ref must be reranker_distillation_dataset:-prefixed")
    if not record.dataset_hash.startswith("sha256:"):
        errors.append("dataset_hash must be sha256:-prefixed")
    if not record.teacher_policy_ref.startswith("teacher_policy:"):
        errors.append("teacher_policy_ref must be teacher_policy:-prefixed")
    for field, prefix in (
        ("student_model_ref", "student_reranker:"),
        ("training_plan_ref", "training_plan:"),
        ("cost_cap_ref", "cost_cap:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    for field in ("student_model_hash_ref", "hyperparam_config_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    if not record.parity_eval_ref.startswith("reranker_parity:"):
        errors.append("parity_eval_ref must be reranker_parity:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "reranker_distillation_dataset:",
                "training_plan:",
                "reranker_parity:",
                "cost_ledger:",
                "owner_approval:",
            )
        ):
            errors.append("reranker distillation training evidence_refs use an unsupported prefix")
            break
    if record.training_enabled or record.training_started or record.training_completed:
        if record.uses_local_fixtures:
            errors.append("reranker distillation training cannot start from local fixtures")
        if record.local_only:
            errors.append("reranker distillation training cannot start from a local_only record")
        if not record.owner_approval_ref:
            errors.append("reranker distillation training requires owner_approval_ref")
    if record.student_model_weights_published:
        errors.append("P3.5 must not publish student model weights")
    if record.model_registry_write:
        errors.append("P3.5 must not write model registry entries")
    if record.production_promotion_requested:
        errors.append("P3.5 must not request production promotion")
    if record.parity_claimed:
        if record.uses_local_fixtures:
            errors.append("reranker parity cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("reranker parity cannot be claimed by a local_only record")
        if not record.training_completed:
            errors.append("reranker parity requires training_completed")
        if not record.parity_eval_passed:
            errors.append("reranker parity requires parity_eval_passed")
        if record.measured_quality_retention_pct < RERANKER_PARITY_THRESHOLD_PCT:
            errors.append("reranker parity requires measured_quality_retention_pct >= 98")
        if record.cost_ratio_to_teacher_pct > RERANKER_COST_RATIO_MAX_PCT:
            errors.append("reranker parity requires cost_ratio_to_teacher_pct <= 30")
        if record.parity_eval_ref == PENDING_RERANKER_PARITY_REF:
            errors.append("reranker parity requires non-pending parity_eval_ref")
        if not record.owner_approval_ref:
            errors.append("reranker parity requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("reranker parity requires evidence_refs")
        elif record.parity_eval_ref not in record.evidence_refs:
            errors.append("reranker parity evidence_refs must include parity_eval_ref")
    else:
        if record.state == RerankerDistillationTrainingState.READY_AFTER_PARITY_EVAL.value:
            errors.append("ready_after_parity_eval state requires parity_claimed")
    return errors


def validate_reranker_eval_arm_record(record: RerankerEvalArmRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_reranker_payload_errors(raw)
    if not isinstance(record, RerankerEvalArmRecord):
        try:
            record = RerankerEvalArmRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required reranker eval arm field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid reranker eval arm field value: {exc}")
            return errors
    if record.contract_version != RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION:
        errors.append("contract_version must match P3.5 reranker eval contract")
    if record.arm_kind not in {kind.value for kind in RerankerEvalArmKind}:
        errors.append(f"unknown reranker eval arm kind: {record.arm_kind}")
    if not record.eval_id.startswith("reranker_eval:"):
        errors.append("eval_id must be reranker_eval:-prefixed")
    if record.arm_kind == RerankerEvalArmKind.TEACHER.value and not record.model_ref.startswith("teacher_reranker:"):
        errors.append("teacher_reranker arm requires teacher_reranker:-prefixed model_ref")
    if record.arm_kind == RerankerEvalArmKind.STUDENT.value and not record.model_ref.startswith("student_reranker:"):
        errors.append("student_reranker arm requires student_reranker:-prefixed model_ref")
    if not record.dataset_ref.startswith("reranker_distillation_dataset:"):
        errors.append("dataset_ref must be reranker_distillation_dataset:-prefixed")
    if not record.dataset_hash.startswith("sha256:"):
        errors.append("dataset_hash must be sha256:-prefixed")
    if not record.teacher_policy_ref.startswith("teacher_policy:"):
        errors.append("teacher_policy_ref must be teacher_policy:-prefixed")
    if not record.holdout_split_ref.startswith("holdout_split:"):
        errors.append("holdout_split_ref must be holdout_split:-prefixed")
    if not record.eval_task_refs:
        errors.append("reranker eval arm requires eval_task_refs")
    _validate_prefixes(errors, record.eval_task_refs, "eval_task_refs", ("reranker_eval_task:",))
    if not 0.0 <= record.quality_score <= 1.0:
        errors.append("quality_score must be between 0 and 1")
    if record.arm_kind == RerankerEvalArmKind.TEACHER.value and record.quality_score <= 0:
        errors.append("teacher_reranker quality_score must be positive")
    if record.cost_cents <= 0:
        errors.append("cost_cents must be positive")
    if record.latency_ms < 0:
        errors.append("latency_ms must be non-negative")
    if record.run_count <= 0:
        errors.append("run_count must be positive")
    if not record.receipt_refs:
        errors.append("reranker eval arm requires receipt_refs")
    _validate_prefixes(errors, record.receipt_refs, "receipt_refs", ("receipt_v2:",))
    if not record.cost_ledger_refs:
        errors.append("reranker eval arm requires cost_ledger_refs")
    _validate_prefixes(errors, record.cost_ledger_refs, "cost_ledger_refs", ("cost_ledger:",))
    if not record.heldout_eval_only:
        errors.append("reranker eval arm must use heldout_eval_only")
    if record.uses_training_split:
        errors.append("reranker eval arm must not use training split tasks")
    if record.source_data_state not in {state.value for state in RerankerEvalDataState}:
        errors.append(f"unknown reranker eval source_data_state: {record.source_data_state}")
    if record.source_data_state == RerankerEvalDataState.LOCAL_FIXTURE.value:
        if not record.uses_local_fixtures:
            errors.append("local fixture reranker eval arms must be marked uses_local_fixtures")
        if record.measured_data_ready:
            errors.append("local fixture reranker eval arms cannot claim measured_data_ready")
        if not record.local_only:
            errors.append("local fixture reranker eval arms must remain local_only")
    if record.source_data_state in {
        RerankerEvalDataState.MEASURED_LAB_ONLY.value,
        RerankerEvalDataState.MEASURED_PRODUCTION.value,
    }:
        if record.uses_local_fixtures:
            errors.append("measured reranker eval arms must not use local fixtures")
        if not record.measured_data_ready:
            errors.append("measured reranker eval arms must mark measured_data_ready")
        if record.local_only:
            errors.append("measured reranker eval arms cannot be local_only")
    return errors


def validate_reranker_parity_comparison_record(
    record: RerankerParityComparisonRecord | Mapping[str, Any],
    *,
    dataset_manifest: Optional[RerankerDistillationDatasetManifestRecord | Mapping[str, Any]] = None,
    training_run: Optional[RerankerDistillationTrainingRunRecord | Mapping[str, Any]] = None,
    teacher_eval: Optional[RerankerEvalArmRecord | Mapping[str, Any]] = None,
    student_eval: Optional[RerankerEvalArmRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_reranker_payload_errors(raw)
    if not isinstance(record, RerankerParityComparisonRecord):
        try:
            record = RerankerParityComparisonRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required reranker parity comparison field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid reranker parity comparison field value: {exc}")
            return errors
    if record.contract_version != RERANKER_DISTILLATION_EVAL_CONTRACT_VERSION:
        errors.append("contract_version must match P3.5 reranker eval contract")
    if record.state not in {state.value for state in RerankerParityState}:
        errors.append(f"unknown reranker parity state: {record.state}")
    if not record.comparison_id.startswith("reranker_parity:"):
        errors.append("comparison_id must be reranker_parity:-prefixed")
    if record.comparison_id == PENDING_RERANKER_PARITY_REF and record.parity_claimed:
        errors.append("reranker parity claims require non-pending comparison_id")
    if not record.teacher_eval_ref.startswith("reranker_eval:"):
        errors.append("teacher_eval_ref must be reranker_eval:-prefixed")
    if not record.student_eval_ref.startswith("reranker_eval:"):
        errors.append("student_eval_ref must be reranker_eval:-prefixed")
    if not record.training_run_ref.startswith("reranker_distillation_run:"):
        errors.append("training_run_ref must be reranker_distillation_run:-prefixed")
    if not record.dataset_ref.startswith("reranker_distillation_dataset:"):
        errors.append("dataset_ref must be reranker_distillation_dataset:-prefixed")
    if not record.dataset_hash.startswith("sha256:"):
        errors.append("dataset_hash must be sha256:-prefixed")
    if not record.holdout_split_ref.startswith("holdout_split:"):
        errors.append("holdout_split_ref must be holdout_split:-prefixed")
    if not record.eval_task_refs:
        errors.append("reranker parity comparison requires eval_task_refs")
    _validate_prefixes(errors, record.eval_task_refs, "eval_task_refs", ("reranker_eval_task:",))
    if not 0.0 <= record.teacher_quality_score <= 1.0 or not 0.0 <= record.student_quality_score <= 1.0:
        errors.append("teacher and student quality scores must be between 0 and 1")
    if record.teacher_quality_score <= 0:
        errors.append("teacher_quality_score must be positive")
    if record.teacher_cost_cents <= 0 or record.student_cost_cents <= 0:
        errors.append("teacher and student costs must be positive")
    _append_parity_metric_errors(errors, record)
    if not record.methodology_ref.startswith("reranker_distillation_methodology:"):
        errors.append("methodology_ref must be reranker_distillation_methodology:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "reranker_parity:",
                "reranker_eval:",
                "reranker_distillation_run:",
                "reranker_distillation_dataset:",
                "cost_ledger:",
                "receipt_v2:",
                "owner_approval:",
            )
        ):
            errors.append("reranker parity evidence_refs use an unsupported prefix")
            break
    _append_supplied_record_consistency_errors(
        errors,
        record,
        dataset_manifest=dataset_manifest,
        training_run=training_run,
        teacher_eval=teacher_eval,
        student_eval=student_eval,
    )
    if record.state == RerankerParityState.LOCAL_STUB.value and not record.local_only:
        errors.append("local_stub reranker parity comparisons must remain local_only")
    if record.state == RerankerParityState.PASSED_PARITY_COST_GATE.value and not record.passed_parity_cost_gate:
        errors.append("passed_parity_cost_gate state requires gate math to pass")
    if record.state == RerankerParityState.FAILED_PARITY_COST_GATE.value and record.passed_parity_cost_gate:
        errors.append("failed_parity_cost_gate state requires gate math to fail")
    if record.parity_claimed and record.state != RerankerParityState.PASSED_PARITY_COST_GATE.value:
        errors.append("reranker parity claims require passed_parity_cost_gate state")
    if record.parity_claimed or record.state == RerankerParityState.PASSED_PARITY_COST_GATE.value:
        if dataset_manifest is None:
            errors.append("reranker parity claims require supplied dataset_manifest")
        if training_run is None:
            errors.append("reranker parity claims require supplied training_run")
        if teacher_eval is None:
            errors.append("reranker parity claims require supplied teacher_eval")
        if student_eval is None:
            errors.append("reranker parity claims require supplied student_eval")
        if record.uses_local_fixtures:
            errors.append("reranker parity cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("reranker parity cannot be claimed by a local_only comparison")
        if not record.measured_data_ready:
            errors.append("reranker parity requires measured_data_ready")
        if not record.quality_parity_passed:
            errors.append("reranker parity requires quality_parity_passed")
        if not record.cost_gate_passed:
            errors.append("reranker parity requires cost_gate_passed")
        if not record.passed_parity_cost_gate:
            errors.append("reranker parity requires passed_parity_cost_gate")
        if record.quality_retention_pct < RERANKER_PARITY_THRESHOLD_PCT:
            errors.append("reranker parity requires quality_retention_pct >= 98")
        if record.cost_ratio_to_teacher_pct > RERANKER_COST_RATIO_MAX_PCT:
            errors.append("reranker parity requires cost_ratio_to_teacher_pct <= 30")
        if not record.owner_approval_ref:
            errors.append("reranker parity requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("reranker parity requires evidence_refs")
        elif record.comparison_id not in record.evidence_refs:
            errors.append("reranker parity evidence_refs must include comparison_id")
    if record.student_model_registry_write:
        errors.append("P3.5 must not write student model registry entries")
    if record.student_model_weights_published:
        errors.append("P3.5 must not publish student model weights")
    if record.production_promotion_requested:
        errors.append("P3.5 must not request production promotion")
    if record.model_pipeline_exit_claimed:
        errors.append("P3.5 must not claim Phase 3 exit")
    if record.public_success_claim_enabled:
        errors.append("P3.5 must not enable public success claims")
    if record.production_writes:
        errors.append("P3.5 must not write production state")
    return errors


def verify_research_lab_reranker_distillation(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    corpus_summary = verify_research_lab_trajectory_corpus()
    fixture = _load_fixture(Path(fixture_path))

    dataset_mapping = _materialize_dataset_fixture(fixture["local_dataset_manifest"])
    dataset = RerankerDistillationDatasetManifestRecord.from_mapping(dataset_mapping)
    _assert(not validate_reranker_distillation_dataset_manifest(dataset), "local reranker distillation dataset validates")
    _assert(dataset.source_corpus_ref == corpus_summary["builder_manifest_id"], "P3.5 pins P3.1 corpus builder ref")
    _assert(not dataset.dataset_ready_claimed, "local distillation dataset does not claim readiness")

    training_mapping = _materialize_training_fixture(fixture["local_training_run"], dataset)
    training = RerankerDistillationTrainingRunRecord.from_mapping(training_mapping)
    _assert(not validate_reranker_distillation_training_run(training), "local reranker distillation training validates")
    _assert(not training.training_started, "local distillation training does not start")
    _assert(not training.parity_claimed, "local distillation training does not claim parity")

    built = build_reranker_distillation_dataset_manifest(**fixture["builder_args"])
    _assert(not validate_reranker_distillation_dataset_manifest(built), "built distillation dataset validates")
    _assert(built.dataset_hash == reranker_distillation_dataset_hash(built), "builder computes dataset hash")

    teacher_eval = build_reranker_eval_arm(**_materialize_eval_fixture(fixture["teacher_eval_input"], dataset))
    student_eval = build_reranker_eval_arm(**_materialize_eval_fixture(fixture["student_eval_input"], dataset))
    _assert(not validate_reranker_eval_arm_record(teacher_eval), "teacher reranker eval validates")
    _assert(not validate_reranker_eval_arm_record(student_eval), "student reranker eval validates")
    _assert(teacher_eval.eval_task_refs == student_eval.eval_task_refs, "teacher/student evals use identical tasks")
    _assert(teacher_eval.dataset_hash == student_eval.dataset_hash == dataset.dataset_hash, "eval arms pin dataset hash")

    comparison = build_reranker_parity_comparison(
        teacher_eval=teacher_eval,
        student_eval=student_eval,
        training_run=training,
        methodology_ref=fixture["comparison_methodology_ref"],
    )
    _assert(
        not validate_reranker_parity_comparison_record(
            comparison,
            dataset_manifest=dataset,
            training_run=training,
            teacher_eval=teacher_eval,
            student_eval=student_eval,
        ),
        "local reranker parity comparison validates",
    )
    _assert(comparison.passed_parity_cost_gate, "local comparison computes parity/cost gate")
    _assert(not comparison.parity_claimed, "local comparison does not claim parity")

    for invalid in fixture["invalid_dataset_manifests"]:
        base_key = str(invalid.get("base", "local_dataset_manifest"))
        base = fixture[base_key]
        if base_key == "local_dataset_manifest":
            base = _materialize_dataset_fixture(base)
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_reranker_distillation_dataset_manifest(record)
        _assert(errors, f"invalid distillation dataset fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_training_runs"]:
        base_key = str(invalid.get("base", "local_training_run"))
        base = fixture[base_key]
        if base_key == "local_training_run":
            base = _materialize_training_fixture(base, dataset)
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_reranker_distillation_training_run(record)
        _assert(errors, f"invalid distillation training fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_eval_arms"]:
        base = teacher_eval.to_dict() if str(invalid.get("base", "teacher_eval_input")) == "teacher_eval_input" else student_eval.to_dict()
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_reranker_eval_arm_record(record)
        _assert(errors, f"invalid reranker eval arm fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_parity_comparisons"]:
        record = _deep_merge(comparison.to_dict(), dict(invalid.get("overrides", {})))
        errors = validate_reranker_parity_comparison_record(
            record,
            dataset_manifest=dataset,
            training_run=training,
            teacher_eval=teacher_eval,
            student_eval=student_eval,
        )
        _assert(errors, f"invalid reranker parity comparison fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    measured_dataset = _deep_merge(dataset.to_dict(), fixture["measured_dataset_overrides"])
    measured_dataset["dataset_hash"] = "sha256:pending"
    measured_dataset["dataset_hash"] = reranker_distillation_dataset_hash(measured_dataset)

    measured_training = _deep_merge(training.to_dict(), fixture["measured_training_overrides"])
    measured_training["dataset_hash"] = measured_dataset["dataset_hash"]

    measured_teacher_eval = _deep_merge(teacher_eval.to_dict(), fixture["measured_eval_overrides"])
    measured_teacher_eval["dataset_hash"] = measured_dataset["dataset_hash"]
    measured_student_eval = _deep_merge(student_eval.to_dict(), fixture["measured_eval_overrides"])
    measured_student_eval["dataset_hash"] = measured_dataset["dataset_hash"]

    measured = _deep_merge(comparison.to_dict(), fixture["measured_pass_overrides"])
    measured["dataset_hash"] = measured_dataset["dataset_hash"]
    measured["evidence_refs"] = [
        comparison.comparison_id,
        teacher_eval.eval_id,
        student_eval.eval_id,
        training.training_run_id,
        "cost_ledger:p3.5:teacher",
        "cost_ledger:p3.5:student",
        "owner_approval:p3.5:parity",
    ]
    measured_errors = validate_reranker_parity_comparison_record(
        measured,
        dataset_manifest=measured_dataset,
        training_run=measured_training,
        teacher_eval=measured_teacher_eval,
        student_eval=measured_student_eval,
    )
    _assert(not measured_errors, "fully measured reranker parity comparison validates")
    bare_measured_errors = validate_reranker_parity_comparison_record(measured)
    _assert(bare_measured_errors, "reranker parity claim requires supplied records")
    _assert_expected_error(bare_measured_errors, fixture["bare_measured_claim_expected_errors"])

    zero_teacher_fixture = fixture["zero_teacher_quality_claim"]
    zero_teacher_eval = _deep_merge(
        dict(measured_teacher_eval),
        dict(zero_teacher_fixture["teacher_eval_overrides"]),
    )
    zero_teacher_claim = _deep_merge(
        dict(measured),
        dict(zero_teacher_fixture["comparison_overrides"]),
    )
    zero_teacher_errors = validate_reranker_parity_comparison_record(
        zero_teacher_claim,
        dataset_manifest=measured_dataset,
        training_run=measured_training,
        teacher_eval=zero_teacher_eval,
        student_eval=measured_student_eval,
    )
    _assert(zero_teacher_errors, "zero-quality teacher parity claim fails")
    _assert_expected_error(zero_teacher_errors, zero_teacher_fixture)

    return {
        "corpus_manifest_id": corpus_summary["builder_manifest_id"],
        "dataset_id": dataset.dataset_id,
        "dataset_ready_claimed": dataset.dataset_ready_claimed,
        "training_run_id": training.training_run_id,
        "training_started": training.training_started,
        "teacher_eval_id": teacher_eval.eval_id,
        "student_eval_id": student_eval.eval_id,
        "comparison_id": comparison.comparison_id,
        "quality_retention_pct": comparison.quality_retention_pct,
        "cost_ratio_to_teacher_pct": comparison.cost_ratio_to_teacher_pct,
        "parity_claimed": comparison.parity_claimed,
    }


def _append_parity_metric_errors(errors: list[str], record: RerankerParityComparisonRecord) -> None:
    try:
        expected_retention = calculate_quality_retention_pct(record.student_quality_score, record.teacher_quality_score)
        expected_delta = calculate_quality_delta_pct(record.student_quality_score, record.teacher_quality_score)
        expected_cost_ratio = calculate_cost_ratio_to_teacher_pct(record.student_cost_cents, record.teacher_cost_cents)
    except ValueError as exc:
        errors.append(str(exc))
        return
    expected_cost_savings = round(100.0 - expected_cost_ratio, 6)
    expected_quality_pass = expected_retention >= RERANKER_PARITY_THRESHOLD_PCT
    expected_cost_pass = expected_cost_ratio <= RERANKER_COST_RATIO_MAX_PCT
    if not _same_float(record.quality_retention_pct, expected_retention):
        errors.append("quality_retention_pct must equal student quality divided by teacher quality")
    if not _same_float(record.quality_delta_pct, expected_delta):
        errors.append("quality_delta_pct must equal student lift over teacher quality")
    if not _same_float(record.cost_ratio_to_teacher_pct, expected_cost_ratio):
        errors.append("cost_ratio_to_teacher_pct must equal student cost divided by teacher cost")
    if not _same_float(record.cost_savings_pct, expected_cost_savings):
        errors.append("cost_savings_pct must equal 100 minus cost ratio")
    if record.quality_parity_passed != expected_quality_pass:
        errors.append("quality_parity_passed must reflect quality_retention_pct >= 98")
    if record.cost_gate_passed != expected_cost_pass:
        errors.append("cost_gate_passed must reflect cost_ratio_to_teacher_pct <= 30")
    if record.passed_parity_cost_gate != (expected_quality_pass and expected_cost_pass):
        errors.append("passed_parity_cost_gate must reflect quality and cost gates")


def _append_supplied_record_consistency_errors(
    errors: list[str],
    record: RerankerParityComparisonRecord,
    *,
    dataset_manifest: Optional[RerankerDistillationDatasetManifestRecord | Mapping[str, Any]],
    training_run: Optional[RerankerDistillationTrainingRunRecord | Mapping[str, Any]],
    teacher_eval: Optional[RerankerEvalArmRecord | Mapping[str, Any]],
    student_eval: Optional[RerankerEvalArmRecord | Mapping[str, Any]],
) -> None:
    if dataset_manifest is not None:
        if not isinstance(dataset_manifest, RerankerDistillationDatasetManifestRecord):
            dataset_manifest = RerankerDistillationDatasetManifestRecord.from_mapping(dataset_manifest)
        errors.extend("dataset_manifest invalid: " + error for error in validate_reranker_distillation_dataset_manifest(dataset_manifest))
        if record.dataset_ref != dataset_manifest.dataset_id:
            errors.append("comparison dataset_ref must match supplied dataset manifest")
        if record.dataset_hash != dataset_manifest.dataset_hash:
            errors.append("comparison dataset_hash must match supplied dataset manifest")
        if record.parity_claimed or record.state == RerankerParityState.PASSED_PARITY_COST_GATE.value:
            if not dataset_manifest.dataset_ready_claimed:
                errors.append("reranker parity claims require ready distillation dataset")
            if dataset_manifest.state != RerankerDistillationDatasetState.READY_AFTER_CORPUS.value:
                errors.append("reranker parity claims require ready_after_corpus dataset state")
    if training_run is not None:
        if not isinstance(training_run, RerankerDistillationTrainingRunRecord):
            training_run = RerankerDistillationTrainingRunRecord.from_mapping(training_run)
        errors.extend("training_run invalid: " + error for error in validate_reranker_distillation_training_run(training_run))
        if record.training_run_ref != training_run.training_run_id:
            errors.append("comparison training_run_ref must match supplied training run")
        if record.dataset_ref != training_run.dataset_ref:
            errors.append("comparison dataset_ref must match supplied training run")
        if record.dataset_hash != training_run.dataset_hash:
            errors.append("comparison dataset_hash must match supplied training run")
        if record.parity_claimed or record.state == RerankerParityState.PASSED_PARITY_COST_GATE.value:
            if not training_run.training_completed:
                errors.append("reranker parity claims require completed distillation training")
            if training_run.state != RerankerDistillationTrainingState.TRAINED_AWAITING_EVAL.value:
                errors.append("reranker parity claims require trained_awaiting_eval training state")
    if teacher_eval is not None:
        if not isinstance(teacher_eval, RerankerEvalArmRecord):
            teacher_eval = RerankerEvalArmRecord.from_mapping(teacher_eval)
        errors.extend("teacher_eval invalid: " + error for error in validate_reranker_eval_arm_record(teacher_eval))
        if teacher_eval.arm_kind != RerankerEvalArmKind.TEACHER.value:
            errors.append("supplied teacher_eval must be teacher_reranker")
        if record.teacher_eval_ref != teacher_eval.eval_id:
            errors.append("comparison teacher_eval_ref must match supplied teacher eval")
        if record.dataset_ref != teacher_eval.dataset_ref:
            errors.append("comparison dataset_ref must match supplied teacher eval")
        if record.dataset_hash != teacher_eval.dataset_hash:
            errors.append("comparison dataset_hash must match supplied teacher eval")
        if record.holdout_split_ref != teacher_eval.holdout_split_ref:
            errors.append("comparison holdout_split_ref must match supplied teacher eval")
        if record.eval_task_refs != teacher_eval.eval_task_refs:
            errors.append("comparison eval_task_refs must match supplied teacher eval")
        if record.teacher_cost_cents != teacher_eval.cost_cents:
            errors.append("comparison teacher_cost_cents must match supplied teacher eval")
        if not _same_float(record.teacher_quality_score, teacher_eval.quality_score):
            errors.append("comparison teacher_quality_score must match supplied teacher eval")
        if record.measured_data_ready and not teacher_eval.measured_data_ready:
            errors.append("measured reranker parity requires measured teacher eval")
    if student_eval is not None:
        if not isinstance(student_eval, RerankerEvalArmRecord):
            student_eval = RerankerEvalArmRecord.from_mapping(student_eval)
        errors.extend("student_eval invalid: " + error for error in validate_reranker_eval_arm_record(student_eval))
        if student_eval.arm_kind != RerankerEvalArmKind.STUDENT.value:
            errors.append("supplied student_eval must be student_reranker")
        if record.student_eval_ref != student_eval.eval_id:
            errors.append("comparison student_eval_ref must match supplied student eval")
        if record.dataset_ref != student_eval.dataset_ref:
            errors.append("comparison dataset_ref must match supplied student eval")
        if record.dataset_hash != student_eval.dataset_hash:
            errors.append("comparison dataset_hash must match supplied student eval")
        if record.holdout_split_ref != student_eval.holdout_split_ref:
            errors.append("comparison holdout_split_ref must match supplied student eval")
        if record.eval_task_refs != student_eval.eval_task_refs:
            errors.append("comparison eval_task_refs must match supplied student eval")
        if record.student_cost_cents != student_eval.cost_cents:
            errors.append("comparison student_cost_cents must match supplied student eval")
        if not _same_float(record.student_quality_score, student_eval.quality_score):
            errors.append("comparison student_quality_score must match supplied student eval")
        if record.measured_data_ready and not student_eval.measured_data_ready:
            errors.append("measured reranker parity requires measured student eval")
    if teacher_eval is not None and student_eval is not None:
        if teacher_eval.eval_task_refs != student_eval.eval_task_refs:
            errors.append("supplied reranker eval arms must use identical eval_task_refs")
        if teacher_eval.holdout_split_ref != student_eval.holdout_split_ref:
            errors.append("supplied reranker eval arms must use identical holdout_split_ref")
        if teacher_eval.dataset_hash != student_eval.dataset_hash:
            errors.append("supplied reranker eval arms must use identical dataset_hash")


def _protected_reranker_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_reranker_material(record))
    if not found:
        return []
    return ["reranker distillation payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_reranker_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_RERANKER_DISTILLATION_KEYS and not key_text.endswith(("_ref", "_refs", "_hash")):
                found.add(key_path)
            found.update(_find_protected_reranker_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_reranker_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_RERANKER_DISTILLATION_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _materialize_dataset_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(record)
    if materialized.get("dataset_hash") == "sha256:pending":
        candidate = RerankerDistillationDatasetManifestRecord.from_mapping(materialized)
        materialized["dataset_hash"] = reranker_distillation_dataset_hash(candidate)
    return materialized


def _materialize_training_fixture(
    record: Mapping[str, Any],
    dataset: RerankerDistillationDatasetManifestRecord,
) -> dict[str, Any]:
    materialized = dict(record)
    if materialized.get("dataset_hash") == "sha256:from-local-dataset":
        materialized["dataset_hash"] = dataset.dataset_hash
    return materialized


def _materialize_eval_fixture(
    record: Mapping[str, Any],
    dataset: RerankerDistillationDatasetManifestRecord,
) -> dict[str, Any]:
    materialized = dict(record)
    if materialized.get("dataset_hash") == "sha256:from-local-dataset":
        materialized["dataset_hash"] = dataset.dataset_hash
    return materialized


def _validate_prefixes(
    errors: list[str],
    values: Sequence[str],
    label: str,
    prefixes: tuple[str, ...],
) -> None:
    for value in values:
        if not value.startswith(prefixes):
            errors.append(f"{label} must use approved prefixes")
            return


def _same_float(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


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
