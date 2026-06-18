"""Phase 3.4 Engine v-next matched-budget A/B evaluator contracts.

P3.4 prepares local contracts for comparing Engine v-next against Engine v1 on
held-out tasks with exact matched budgets. It does not run either engine, train
models, promote candidates, write registries, publish claims, or write
production state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .counterfactual_gate import calculate_yield_points_per_1000_usd
from .fine_tune_dataset import (
    PROTECTED_FINE_TUNE_KEYS,
    PROTECTED_FINE_TUNE_MARKERS,
    EngineFineTuneTrainingRunRecord,
    FineTuneTrainingState,
    validate_engine_fine_tune_training_run,
    verify_research_lab_fine_tune_dataset,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "engine_ab_evaluator_fixtures.json"

ENGINE_AB_EVALUATOR_CONTRACT_VERSION = "engine_ab_evaluator:v1:local_contract"
ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT = 20.0
PENDING_ENGINE_AB_EVAL_REF = "engine_ab:pending"

PROTECTED_ENGINE_AB_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_FINE_TUNE_KEYS)
    | {
        "heldout_answer_key",
        "live_engine_output",
        "private_customer_record",
        "raw_control_output",
        "raw_engine_input",
        "raw_eval_task",
        "raw_model_output",
        "raw_vnext_output",
        "sealed_holdout_set",
        "training_split_task",
    }
)

PROTECTED_ENGINE_AB_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_FINE_TUNE_MARKERS)
        | {
            "heldout answer key",
            "live engine output",
            "private customer record",
            "raw control output",
            "raw engine input",
            "raw eval task",
            "raw model output",
            "raw vnext output",
            "sealed holdout",
            "training split task",
        }
    )
)


class EngineABArmKind(str, Enum):
    ENGINE_V1_CONTROL = "engine_v1_control"
    ENGINE_VNEXT_CANDIDATE = "engine_vnext_candidate"


class EngineABDataState(str, Enum):
    LOCAL_FIXTURE = "local_fixture"
    MEASURED_LAB_ONLY = "measured_lab_only"
    MEASURED_PRODUCTION = "measured_production"
    BLOCKED = "blocked"


class EngineABComparisonState(str, Enum):
    LOCAL_STUB = "local_stub"
    READY_AFTER_MEASURED_EVIDENCE = "ready_after_measured_evidence"
    PASSED_20_PERCENT_GATE = "passed_20_percent_gate"
    FAILED_20_PERCENT_GATE = "failed_20_percent_gate"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EngineABArmResultRecord:
    result_id: str
    arm_kind: str
    engine_ref: str
    fine_tune_dataset_ref: str
    fine_tune_dataset_hash: str
    heldout_split_ref: str
    heldout_task_refs: tuple[str, ...]
    budget_cents: int
    verified_points: float
    run_count: int
    receipt_refs: tuple[str, ...]
    cost_ledger_refs: tuple[str, ...]
    external_cost_cents: int
    heldout_eval_only: bool = True
    uses_training_split: bool = False
    measured_data_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    production_data_claimed: bool = False
    source_data_state: str = EngineABDataState.LOCAL_FIXTURE.value
    contract_version: str = ENGINE_AB_EVALUATOR_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EngineABArmResultRecord":
        return cls(
            result_id=str(data["result_id"]),
            arm_kind=str(data["arm_kind"]),
            engine_ref=str(data["engine_ref"]),
            fine_tune_dataset_ref=str(data["fine_tune_dataset_ref"]),
            fine_tune_dataset_hash=str(data["fine_tune_dataset_hash"]),
            heldout_split_ref=str(data["heldout_split_ref"]),
            heldout_task_refs=tuple(str(item) for item in data.get("heldout_task_refs", [])),
            budget_cents=int(data["budget_cents"]),
            verified_points=float(data["verified_points"]),
            run_count=int(data["run_count"]),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            cost_ledger_refs=tuple(str(item) for item in data.get("cost_ledger_refs", [])),
            external_cost_cents=int(data.get("external_cost_cents", 0)),
            heldout_eval_only=bool(data.get("heldout_eval_only", True)),
            uses_training_split=bool(data.get("uses_training_split", False)),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            production_data_claimed=bool(data.get("production_data_claimed", False)),
            source_data_state=str(data.get("source_data_state", EngineABDataState.LOCAL_FIXTURE.value)),
            contract_version=str(data.get("contract_version", ENGINE_AB_EVALUATOR_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["heldout_task_refs"] = list(self.heldout_task_refs)
        data["receipt_refs"] = list(self.receipt_refs)
        data["cost_ledger_refs"] = list(self.cost_ledger_refs)
        return data

    def identity_payload(self) -> dict[str, Any]:
        return {
            "arm_kind": self.arm_kind,
            "engine_ref": self.engine_ref,
            "fine_tune_dataset_ref": self.fine_tune_dataset_ref,
            "fine_tune_dataset_hash": self.fine_tune_dataset_hash,
            "heldout_split_ref": self.heldout_split_ref,
            "heldout_task_refs": list(self.heldout_task_refs),
            "budget_cents": self.budget_cents,
            "verified_points": self.verified_points,
            "run_count": self.run_count,
            "receipt_refs": list(self.receipt_refs),
            "cost_ledger_refs": list(self.cost_ledger_refs),
            "external_cost_cents": self.external_cost_cents,
            "heldout_eval_only": self.heldout_eval_only,
            "uses_training_split": self.uses_training_split,
            "source_data_state": self.source_data_state,
        }


@dataclass(frozen=True)
class EngineABComparisonRecord:
    comparison_id: str
    control_result_ref: str
    candidate_result_ref: str
    fine_tune_training_ref: str
    fine_tune_dataset_ref: str
    fine_tune_dataset_hash: str
    heldout_split_ref: str
    heldout_task_refs: tuple[str, ...]
    matched_budget_cents: int
    control_budget_cents: int
    candidate_budget_cents: int
    control_verified_points: float
    candidate_verified_points: float
    control_yield_points_per_1000_usd: float
    candidate_yield_points_per_1000_usd: float
    yield_delta_points_per_1000_usd: float
    yield_delta_pct: float
    passed_20_percent_gate: bool
    methodology_ref: str
    measured_data_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    improvement_claimed: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    model_registry_write: bool = False
    model_weights_published: bool = False
    production_promotion_requested: bool = False
    model_pipeline_exit_claimed: bool = False
    public_success_claim_enabled: bool = False
    production_writes: bool = False
    state: str = EngineABComparisonState.LOCAL_STUB.value
    contract_version: str = ENGINE_AB_EVALUATOR_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EngineABComparisonRecord":
        return cls(
            comparison_id=str(data["comparison_id"]),
            control_result_ref=str(data["control_result_ref"]),
            candidate_result_ref=str(data["candidate_result_ref"]),
            fine_tune_training_ref=str(data["fine_tune_training_ref"]),
            fine_tune_dataset_ref=str(data["fine_tune_dataset_ref"]),
            fine_tune_dataset_hash=str(data["fine_tune_dataset_hash"]),
            heldout_split_ref=str(data["heldout_split_ref"]),
            heldout_task_refs=tuple(str(item) for item in data.get("heldout_task_refs", [])),
            matched_budget_cents=int(data["matched_budget_cents"]),
            control_budget_cents=int(data["control_budget_cents"]),
            candidate_budget_cents=int(data["candidate_budget_cents"]),
            control_verified_points=float(data["control_verified_points"]),
            candidate_verified_points=float(data["candidate_verified_points"]),
            control_yield_points_per_1000_usd=float(data["control_yield_points_per_1000_usd"]),
            candidate_yield_points_per_1000_usd=float(data["candidate_yield_points_per_1000_usd"]),
            yield_delta_points_per_1000_usd=float(data["yield_delta_points_per_1000_usd"]),
            yield_delta_pct=float(data["yield_delta_pct"]),
            passed_20_percent_gate=bool(data.get("passed_20_percent_gate", False)),
            methodology_ref=str(data["methodology_ref"]),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            improvement_claimed=bool(data.get("improvement_claimed", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            model_registry_write=bool(data.get("model_registry_write", False)),
            model_weights_published=bool(data.get("model_weights_published", False)),
            production_promotion_requested=bool(data.get("production_promotion_requested", False)),
            model_pipeline_exit_claimed=bool(data.get("model_pipeline_exit_claimed", False)),
            public_success_claim_enabled=bool(data.get("public_success_claim_enabled", False)),
            production_writes=bool(data.get("production_writes", False)),
            state=str(data.get("state", EngineABComparisonState.LOCAL_STUB.value)),
            contract_version=str(data.get("contract_version", ENGINE_AB_EVALUATOR_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["heldout_task_refs"] = list(self.heldout_task_refs)
        data["evidence_refs"] = list(self.evidence_refs)
        return data

    def identity_payload(self) -> dict[str, Any]:
        return {
            "control_result_ref": self.control_result_ref,
            "candidate_result_ref": self.candidate_result_ref,
            "fine_tune_training_ref": self.fine_tune_training_ref,
            "fine_tune_dataset_ref": self.fine_tune_dataset_ref,
            "fine_tune_dataset_hash": self.fine_tune_dataset_hash,
            "heldout_split_ref": self.heldout_split_ref,
            "heldout_task_refs": list(self.heldout_task_refs),
            "matched_budget_cents": self.matched_budget_cents,
            "control_budget_cents": self.control_budget_cents,
            "candidate_budget_cents": self.candidate_budget_cents,
            "control_verified_points": self.control_verified_points,
            "candidate_verified_points": self.candidate_verified_points,
            "control_yield_points_per_1000_usd": self.control_yield_points_per_1000_usd,
            "candidate_yield_points_per_1000_usd": self.candidate_yield_points_per_1000_usd,
            "yield_delta_points_per_1000_usd": self.yield_delta_points_per_1000_usd,
            "yield_delta_pct": self.yield_delta_pct,
            "passed_20_percent_gate": self.passed_20_percent_gate,
            "methodology_ref": self.methodology_ref,
            "measured_data_ready": self.measured_data_ready,
        }


def calculate_yield_delta_pct(candidate_yield: float, control_yield: float) -> float:
    if control_yield <= 0:
        raise ValueError("control_yield must be positive")
    return round(((float(candidate_yield) - float(control_yield)) / float(control_yield)) * 100.0, 6)


def build_engine_ab_arm_result(**kwargs: Any) -> EngineABArmResultRecord:
    record = EngineABArmResultRecord.from_mapping({"result_id": "engine_ab_result:pending", **kwargs})
    errors = validate_engine_ab_arm_result_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    data = record.to_dict()
    data["result_id"] = "engine_ab_result:" + sha256_json(record.identity_payload()).split(":", 1)[1][:16]
    return EngineABArmResultRecord.from_mapping(data)


def build_engine_ab_comparison(
    *,
    control_result: EngineABArmResultRecord | Mapping[str, Any],
    candidate_result: EngineABArmResultRecord | Mapping[str, Any],
    fine_tune_training: EngineFineTuneTrainingRunRecord | Mapping[str, Any],
    methodology_ref: str,
) -> EngineABComparisonRecord:
    if not isinstance(control_result, EngineABArmResultRecord):
        control_result = EngineABArmResultRecord.from_mapping(control_result)
    if not isinstance(candidate_result, EngineABArmResultRecord):
        candidate_result = EngineABArmResultRecord.from_mapping(candidate_result)
    if not isinstance(fine_tune_training, EngineFineTuneTrainingRunRecord):
        fine_tune_training = EngineFineTuneTrainingRunRecord.from_mapping(fine_tune_training)
    errors = (
        validate_engine_ab_arm_result_record(control_result)
        + validate_engine_ab_arm_result_record(candidate_result)
        + validate_engine_fine_tune_training_run(fine_tune_training)
    )
    if errors:
        raise ValueError("; ".join(errors))
    control_yield = calculate_yield_points_per_1000_usd(control_result.verified_points, control_result.budget_cents)
    candidate_yield = calculate_yield_points_per_1000_usd(candidate_result.verified_points, candidate_result.budget_cents)
    delta_points = round(candidate_yield - control_yield, 6)
    delta_pct = calculate_yield_delta_pct(candidate_yield, control_yield)
    draft = EngineABComparisonRecord(
        comparison_id=PENDING_ENGINE_AB_EVAL_REF,
        control_result_ref=control_result.result_id,
        candidate_result_ref=candidate_result.result_id,
        fine_tune_training_ref=fine_tune_training.training_run_id,
        fine_tune_dataset_ref=candidate_result.fine_tune_dataset_ref,
        fine_tune_dataset_hash=candidate_result.fine_tune_dataset_hash,
        heldout_split_ref=candidate_result.heldout_split_ref,
        heldout_task_refs=candidate_result.heldout_task_refs,
        matched_budget_cents=control_result.budget_cents,
        control_budget_cents=control_result.budget_cents,
        candidate_budget_cents=candidate_result.budget_cents,
        control_verified_points=control_result.verified_points,
        candidate_verified_points=candidate_result.verified_points,
        control_yield_points_per_1000_usd=control_yield,
        candidate_yield_points_per_1000_usd=candidate_yield,
        yield_delta_points_per_1000_usd=delta_points,
        yield_delta_pct=delta_pct,
        passed_20_percent_gate=delta_pct >= ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT,
        methodology_ref=methodology_ref,
        measured_data_ready=control_result.measured_data_ready and candidate_result.measured_data_ready,
        uses_local_fixtures=control_result.uses_local_fixtures or candidate_result.uses_local_fixtures,
        local_only=control_result.local_only or candidate_result.local_only,
        state=EngineABComparisonState.LOCAL_STUB.value,
    )
    data = draft.to_dict()
    data["comparison_id"] = "engine_ab:" + sha256_json(draft.identity_payload()).split(":", 1)[1][:16]
    record = EngineABComparisonRecord.from_mapping(data)
    errors = validate_engine_ab_comparison_record(
        record,
        control_result=control_result,
        candidate_result=candidate_result,
        fine_tune_training=fine_tune_training,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_engine_ab_arm_result_record(record: EngineABArmResultRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_engine_ab_payload_errors(raw)
    if not isinstance(record, EngineABArmResultRecord):
        try:
            record = EngineABArmResultRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required engine A/B arm field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid engine A/B arm field value: {exc}")
            return errors
    if record.contract_version != ENGINE_AB_EVALUATOR_CONTRACT_VERSION:
        errors.append("contract_version must match P3.4 engine A/B contract")
    if record.arm_kind not in {kind.value for kind in EngineABArmKind}:
        errors.append(f"unknown engine A/B arm kind: {record.arm_kind}")
    if not record.result_id.startswith("engine_ab_result:"):
        errors.append("result_id must be engine_ab_result:-prefixed")
    if record.arm_kind == EngineABArmKind.ENGINE_V1_CONTROL.value and not record.engine_ref.startswith("engine_v1:"):
        errors.append("engine_v1_control arm requires engine_v1:-prefixed engine_ref")
    if record.arm_kind == EngineABArmKind.ENGINE_VNEXT_CANDIDATE.value and not record.engine_ref.startswith("model_candidate:"):
        errors.append("engine_vnext_candidate arm requires model_candidate:-prefixed engine_ref")
    if not record.fine_tune_dataset_ref.startswith("fine_tune_dataset:"):
        errors.append("fine_tune_dataset_ref must be fine_tune_dataset:-prefixed")
    if not record.fine_tune_dataset_hash.startswith("sha256:"):
        errors.append("fine_tune_dataset_hash must be sha256:-prefixed")
    if not record.heldout_split_ref.startswith("holdout_split:"):
        errors.append("heldout_split_ref must be holdout_split:-prefixed")
    if not record.heldout_task_refs:
        errors.append("engine A/B arm requires heldout_task_refs")
    _validate_prefixes(errors, record.heldout_task_refs, "heldout_task_refs", ("heldout_task:",))
    if record.budget_cents <= 0:
        errors.append("budget_cents must be positive")
    if record.verified_points < 0:
        errors.append("verified_points must be non-negative")
    if record.run_count <= 0:
        errors.append("run_count must be positive")
    if record.external_cost_cents < 0:
        errors.append("external_cost_cents must be non-negative")
    if not record.receipt_refs:
        errors.append("engine A/B arm requires receipt_refs")
    _validate_prefixes(errors, record.receipt_refs, "receipt_refs", ("receipt_v2:",))
    if not record.cost_ledger_refs:
        errors.append("engine A/B arm requires cost_ledger_refs")
    _validate_prefixes(errors, record.cost_ledger_refs, "cost_ledger_refs", ("cost_ledger:",))
    if not record.heldout_eval_only:
        errors.append("engine A/B arm must use heldout_eval_only")
    if record.uses_training_split:
        errors.append("engine A/B arm must not use training split tasks")
    if record.source_data_state not in {state.value for state in EngineABDataState}:
        errors.append(f"unknown engine A/B source_data_state: {record.source_data_state}")
    if record.source_data_state == EngineABDataState.LOCAL_FIXTURE.value:
        if not record.uses_local_fixtures:
            errors.append("local fixture engine A/B arms must be marked uses_local_fixtures")
        if record.measured_data_ready:
            errors.append("local fixture engine A/B arms cannot claim measured_data_ready")
        if not record.local_only:
            errors.append("local fixture engine A/B arms must remain local_only")
    if record.source_data_state in {
        EngineABDataState.MEASURED_LAB_ONLY.value,
        EngineABDataState.MEASURED_PRODUCTION.value,
    }:
        if record.uses_local_fixtures:
            errors.append("measured engine A/B arms must not use local fixtures")
        if not record.measured_data_ready:
            errors.append("measured engine A/B arms must mark measured_data_ready")
        if record.local_only:
            errors.append("measured engine A/B arms cannot be local_only")
    if record.production_data_claimed:
        errors.append("P3.4 local contracts must not claim production A/B data")
    return errors


def validate_engine_ab_comparison_record(
    record: EngineABComparisonRecord | Mapping[str, Any],
    *,
    control_result: Optional[EngineABArmResultRecord | Mapping[str, Any]] = None,
    candidate_result: Optional[EngineABArmResultRecord | Mapping[str, Any]] = None,
    fine_tune_training: Optional[EngineFineTuneTrainingRunRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_engine_ab_payload_errors(raw)
    if not isinstance(record, EngineABComparisonRecord):
        try:
            record = EngineABComparisonRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required engine A/B comparison field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid engine A/B comparison field value: {exc}")
            return errors
    if record.contract_version != ENGINE_AB_EVALUATOR_CONTRACT_VERSION:
        errors.append("contract_version must match P3.4 engine A/B contract")
    if record.state not in {state.value for state in EngineABComparisonState}:
        errors.append(f"unknown engine A/B comparison state: {record.state}")
    if not record.comparison_id.startswith("engine_ab:"):
        errors.append("comparison_id must be engine_ab:-prefixed")
    if record.comparison_id == PENDING_ENGINE_AB_EVAL_REF and record.improvement_claimed:
        errors.append("improvement claims require non-pending engine_ab comparison id")
    if not record.control_result_ref.startswith("engine_ab_result:"):
        errors.append("control_result_ref must be engine_ab_result:-prefixed")
    if not record.candidate_result_ref.startswith("engine_ab_result:"):
        errors.append("candidate_result_ref must be engine_ab_result:-prefixed")
    if not record.fine_tune_training_ref.startswith("engine_finetune_run:"):
        errors.append("fine_tune_training_ref must be engine_finetune_run:-prefixed")
    if not record.fine_tune_dataset_ref.startswith("fine_tune_dataset:"):
        errors.append("fine_tune_dataset_ref must be fine_tune_dataset:-prefixed")
    if not record.fine_tune_dataset_hash.startswith("sha256:"):
        errors.append("fine_tune_dataset_hash must be sha256:-prefixed")
    if not record.heldout_split_ref.startswith("holdout_split:"):
        errors.append("heldout_split_ref must be holdout_split:-prefixed")
    if not record.heldout_task_refs:
        errors.append("engine A/B comparison requires heldout_task_refs")
    _validate_prefixes(errors, record.heldout_task_refs, "heldout_task_refs", ("heldout_task:",))
    if record.matched_budget_cents <= 0:
        errors.append("matched_budget_cents must be positive")
    if record.control_budget_cents != record.candidate_budget_cents:
        errors.append("engine A/B comparison requires exact matched budgets")
    if record.matched_budget_cents not in {record.control_budget_cents, record.candidate_budget_cents}:
        errors.append("matched_budget_cents must equal both arm budgets")
    if record.control_verified_points < 0 or record.candidate_verified_points < 0:
        errors.append("engine A/B verified points must be non-negative")
    if record.control_budget_cents > 0 and record.candidate_budget_cents > 0:
        _append_metric_consistency_errors(errors, record)
    if not record.methodology_ref.startswith("engine_ab_methodology:"):
        errors.append("methodology_ref must be engine_ab_methodology:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "engine_ab:",
                "engine_ab_result:",
                "engine_finetune_run:",
                "fine_tune_dataset:",
                "cost_ledger:",
                "receipt_v2:",
                "owner_approval:",
            )
        ):
            errors.append("engine A/B evidence_refs use an unsupported prefix")
            break
    _append_supplied_record_consistency_errors(
        errors,
        record,
        control_result=control_result,
        candidate_result=candidate_result,
        fine_tune_training=fine_tune_training,
    )
    if record.state == EngineABComparisonState.LOCAL_STUB.value and not record.local_only:
        errors.append("local_stub engine A/B comparisons must remain local_only")
    if record.state == EngineABComparisonState.PASSED_20_PERCENT_GATE.value and not record.passed_20_percent_gate:
        errors.append("passed_20_percent_gate state requires gate math to pass")
    if record.state == EngineABComparisonState.FAILED_20_PERCENT_GATE.value and record.passed_20_percent_gate:
        errors.append("failed_20_percent_gate state requires gate math to fail")
    if record.improvement_claimed and record.state != EngineABComparisonState.PASSED_20_PERCENT_GATE.value:
        errors.append("Engine v-next improvement claims require passed_20_percent_gate state")
    if record.improvement_claimed or record.state == EngineABComparisonState.PASSED_20_PERCENT_GATE.value:
        if control_result is None:
            errors.append("Engine v-next improvement requires supplied control_result")
        if candidate_result is None:
            errors.append("Engine v-next improvement requires supplied candidate_result")
        if fine_tune_training is None:
            errors.append("Engine v-next improvement requires supplied fine_tune_training")
        if record.uses_local_fixtures:
            errors.append("Engine v-next improvement cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Engine v-next improvement cannot be claimed by a local_only comparison")
        if not record.measured_data_ready:
            errors.append("Engine v-next improvement requires measured_data_ready")
        if not record.passed_20_percent_gate:
            errors.append("Engine v-next improvement requires the +20% gate to pass")
        if record.yield_delta_pct < ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT:
            errors.append("Engine v-next improvement requires yield_delta_pct >= 20")
        if not record.owner_approval_ref:
            errors.append("Engine v-next improvement requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("Engine v-next improvement requires evidence_refs")
        elif record.comparison_id not in record.evidence_refs:
            errors.append("Engine v-next improvement evidence_refs must include comparison_id")
    if record.model_registry_write:
        errors.append("P3.4 must not write model registry entries")
    if record.model_weights_published:
        errors.append("P3.4 must not publish model weights")
    if record.production_promotion_requested:
        errors.append("P3.4 must not request production promotion")
    if record.model_pipeline_exit_claimed:
        errors.append("P3.4 must not claim Phase 3 exit")
    if record.public_success_claim_enabled:
        errors.append("P3.4 must not enable public success claims")
    if record.production_writes:
        errors.append("P3.4 must not write production state")
    return errors


def verify_research_lab_engine_ab_evaluator(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fine_tune_summary = verify_research_lab_fine_tune_dataset()
    fixture = _load_fixture(Path(fixture_path))

    control = build_engine_ab_arm_result(**fixture["control_arm_input"])
    candidate = build_engine_ab_arm_result(**fixture["candidate_arm_input"])
    _assert(not validate_engine_ab_arm_result_record(control), "control A/B arm validates")
    _assert(not validate_engine_ab_arm_result_record(candidate), "candidate A/B arm validates")
    _assert(control.arm_kind == EngineABArmKind.ENGINE_V1_CONTROL.value, "control arm is Engine v1")
    _assert(candidate.arm_kind == EngineABArmKind.ENGINE_VNEXT_CANDIDATE.value, "candidate arm is Engine v-next")
    _assert(control.fine_tune_dataset_ref == fine_tune_summary["dataset_id"], "control arm pins P3.3 dataset")
    _assert(candidate.fine_tune_dataset_ref == fine_tune_summary["dataset_id"], "candidate arm pins P3.3 dataset")
    _assert(control.heldout_task_refs == candidate.heldout_task_refs, "arms use identical held-out tasks")
    _assert(control.budget_cents == candidate.budget_cents, "arms use exact matched budget")

    training_stub = EngineFineTuneTrainingRunRecord.from_mapping(fixture["fine_tune_training_stub"])
    _assert(training_stub.training_run_id == fine_tune_summary["training_run_id"], "P3.4 pins P3.3 training run")
    _assert(not validate_engine_fine_tune_training_run(training_stub), "P3.3 training stub validates")

    comparison = build_engine_ab_comparison(
        control_result=control,
        candidate_result=candidate,
        fine_tune_training=training_stub,
        methodology_ref=fixture["comparison_methodology_ref"],
    )
    _assert(
        not validate_engine_ab_comparison_record(
            comparison,
            control_result=control,
            candidate_result=candidate,
            fine_tune_training=training_stub,
        ),
        "local engine A/B comparison validates",
    )
    _assert(comparison.matched_budget_cents == control.budget_cents == candidate.budget_cents, "comparison budget is exact")
    _assert(comparison.heldout_task_refs == control.heldout_task_refs == candidate.heldout_task_refs, "comparison tasks match arms")
    _assert(not comparison.improvement_claimed, "local comparison does not claim improvement")
    _assert(comparison.local_only, "local comparison remains local_only")

    for invalid in fixture["invalid_arms"]:
        base_key = str(invalid.get("base", "control_arm_input"))
        if base_key == "control_arm_input":
            base = control.to_dict()
        elif base_key == "candidate_arm_input":
            base = candidate.to_dict()
        else:
            base = fixture[base_key]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_engine_ab_arm_result_record(record)
        _assert(errors, f"invalid engine A/B arm fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_comparisons"]:
        base = comparison.to_dict()
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_engine_ab_comparison_record(
            record,
            control_result=control,
            candidate_result=candidate,
            fine_tune_training=training_stub,
        )
        _assert(errors, f"invalid engine A/B comparison fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    measured = _deep_merge(comparison.to_dict(), fixture["measured_pass_overrides"])
    measured["evidence_refs"] = [
        comparison.comparison_id,
        control.result_id,
        candidate.result_id,
        training_stub.training_run_id,
        "cost_ledger:p3.4:control",
        "cost_ledger:p3.4:candidate",
        "owner_approval:p3.4:engine-ab",
    ]
    measured_errors = validate_engine_ab_comparison_record(
        measured,
        control_result=_deep_merge(control.to_dict(), fixture["measured_arm_overrides"]),
        candidate_result=_deep_merge(candidate.to_dict(), fixture["measured_arm_overrides"]),
        fine_tune_training=fixture["measured_training_run"],
    )
    _assert(not measured_errors, "fully measured +20% comparison validates")
    bare_measured_errors = validate_engine_ab_comparison_record(measured)
    _assert(bare_measured_errors, "measured improvement claim requires supplied records")
    _assert_expected_error(bare_measured_errors, fixture["bare_measured_claim_expected_errors"])

    return {
        "fine_tune_dataset_id": fine_tune_summary["dataset_id"],
        "fine_tune_training_id": fine_tune_summary["training_run_id"],
        "control_result_id": control.result_id,
        "candidate_result_id": candidate.result_id,
        "comparison_id": comparison.comparison_id,
        "matched_budget_cents": comparison.matched_budget_cents,
        "yield_delta_pct": comparison.yield_delta_pct,
        "improvement_claimed": comparison.improvement_claimed,
        "local_only": comparison.local_only,
    }


def _append_metric_consistency_errors(errors: list[str], record: EngineABComparisonRecord) -> None:
    expected_control_yield = calculate_yield_points_per_1000_usd(
        record.control_verified_points,
        record.control_budget_cents,
    )
    expected_candidate_yield = calculate_yield_points_per_1000_usd(
        record.candidate_verified_points,
        record.candidate_budget_cents,
    )
    expected_delta_points = round(expected_candidate_yield - expected_control_yield, 6)
    try:
        expected_delta_pct = calculate_yield_delta_pct(expected_candidate_yield, expected_control_yield)
    except ValueError:
        errors.append("control yield must be positive for yield_delta_pct")
        return
    if not _same_float(record.control_yield_points_per_1000_usd, expected_control_yield):
        errors.append("control_yield_points_per_1000_usd does not match points and budget")
    if not _same_float(record.candidate_yield_points_per_1000_usd, expected_candidate_yield):
        errors.append("candidate_yield_points_per_1000_usd does not match points and budget")
    if not _same_float(record.yield_delta_points_per_1000_usd, expected_delta_points):
        errors.append("yield_delta_points_per_1000_usd must equal candidate yield minus control yield")
    if not _same_float(record.yield_delta_pct, expected_delta_pct):
        errors.append("yield_delta_pct must equal candidate lift over control yield")
    if record.passed_20_percent_gate != (expected_delta_pct >= ENGINE_AB_IMPROVEMENT_THRESHOLD_PCT):
        errors.append("passed_20_percent_gate must reflect yield_delta_pct >= 20")


def _append_supplied_record_consistency_errors(
    errors: list[str],
    record: EngineABComparisonRecord,
    *,
    control_result: Optional[EngineABArmResultRecord | Mapping[str, Any]],
    candidate_result: Optional[EngineABArmResultRecord | Mapping[str, Any]],
    fine_tune_training: Optional[EngineFineTuneTrainingRunRecord | Mapping[str, Any]],
) -> None:
    if control_result is not None:
        if not isinstance(control_result, EngineABArmResultRecord):
            control_result = EngineABArmResultRecord.from_mapping(control_result)
        errors.extend("control_result invalid: " + error for error in validate_engine_ab_arm_result_record(control_result))
        if control_result.arm_kind != EngineABArmKind.ENGINE_V1_CONTROL.value:
            errors.append("supplied control_result must be engine_v1_control")
        if record.control_result_ref != control_result.result_id:
            errors.append("comparison control_result_ref must match supplied control result")
        if record.control_budget_cents != control_result.budget_cents:
            errors.append("comparison control_budget_cents must match supplied control result")
        if not _same_float(record.control_verified_points, control_result.verified_points):
            errors.append("comparison control_verified_points must match supplied control result")
        if record.fine_tune_dataset_ref != control_result.fine_tune_dataset_ref:
            errors.append("comparison fine_tune_dataset_ref must match supplied control result")
        if record.fine_tune_dataset_hash != control_result.fine_tune_dataset_hash:
            errors.append("comparison fine_tune_dataset_hash must match supplied control result")
        if record.heldout_split_ref != control_result.heldout_split_ref:
            errors.append("comparison heldout_split_ref must match supplied control result")
        if record.heldout_task_refs != control_result.heldout_task_refs:
            errors.append("comparison heldout_task_refs must match supplied control result")
        if record.measured_data_ready and not control_result.measured_data_ready:
            errors.append("measured comparison requires measured control result")
    if candidate_result is not None:
        if not isinstance(candidate_result, EngineABArmResultRecord):
            candidate_result = EngineABArmResultRecord.from_mapping(candidate_result)
        errors.extend("candidate_result invalid: " + error for error in validate_engine_ab_arm_result_record(candidate_result))
        if candidate_result.arm_kind != EngineABArmKind.ENGINE_VNEXT_CANDIDATE.value:
            errors.append("supplied candidate_result must be engine_vnext_candidate")
        if record.candidate_result_ref != candidate_result.result_id:
            errors.append("comparison candidate_result_ref must match supplied candidate result")
        if record.candidate_budget_cents != candidate_result.budget_cents:
            errors.append("comparison candidate_budget_cents must match supplied candidate result")
        if not _same_float(record.candidate_verified_points, candidate_result.verified_points):
            errors.append("comparison candidate_verified_points must match supplied candidate result")
        if record.fine_tune_dataset_ref != candidate_result.fine_tune_dataset_ref:
            errors.append("comparison fine_tune_dataset_ref must match supplied candidate result")
        if record.fine_tune_dataset_hash != candidate_result.fine_tune_dataset_hash:
            errors.append("comparison fine_tune_dataset_hash must match supplied candidate result")
        if record.heldout_split_ref != candidate_result.heldout_split_ref:
            errors.append("comparison heldout_split_ref must match supplied candidate result")
        if record.heldout_task_refs != candidate_result.heldout_task_refs:
            errors.append("comparison heldout_task_refs must match supplied candidate result")
        if record.measured_data_ready and not candidate_result.measured_data_ready:
            errors.append("measured comparison requires measured candidate result")
    if control_result is not None and candidate_result is not None:
        if control_result.budget_cents != candidate_result.budget_cents:
            errors.append("supplied engine A/B arms must use exact matched budgets")
        if control_result.heldout_task_refs != candidate_result.heldout_task_refs:
            errors.append("supplied engine A/B arms must use identical held-out tasks")
        if control_result.heldout_split_ref != candidate_result.heldout_split_ref:
            errors.append("supplied engine A/B arms must use identical heldout_split_ref")
        if control_result.fine_tune_dataset_hash != candidate_result.fine_tune_dataset_hash:
            errors.append("supplied engine A/B arms must use identical fine_tune_dataset_hash")
    if fine_tune_training is not None:
        if not isinstance(fine_tune_training, EngineFineTuneTrainingRunRecord):
            fine_tune_training = EngineFineTuneTrainingRunRecord.from_mapping(fine_tune_training)
        training_errors = validate_engine_fine_tune_training_run(fine_tune_training)
        if training_errors:
            errors.extend("fine-tune training record invalid: " + error for error in training_errors)
        if record.fine_tune_training_ref != fine_tune_training.training_run_id:
            errors.append("comparison fine_tune_training_ref must match supplied fine-tune training run")
        if record.fine_tune_dataset_ref != fine_tune_training.dataset_ref:
            errors.append("comparison fine_tune_dataset_ref must match supplied fine-tune training run")
        if record.fine_tune_dataset_hash != fine_tune_training.dataset_hash:
            errors.append("comparison fine_tune_dataset_hash must match supplied fine-tune training run")
        if record.improvement_claimed or record.state == EngineABComparisonState.PASSED_20_PERCENT_GATE.value:
            if not fine_tune_training.training_completed:
                errors.append("Engine v-next improvement requires completed fine-tune training")
            if fine_tune_training.state != FineTuneTrainingState.TRAINED_AWAITING_EVAL.value:
                errors.append("Engine v-next improvement requires trained_awaiting_eval fine-tune state")


def _protected_engine_ab_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_engine_ab_material(record))
    if not found:
        return []
    return ["engine A/B payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_engine_ab_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_ENGINE_AB_KEYS and not key_text.endswith(("_ref", "_refs", "_hash")):
                found.add(key_path)
            found.update(_find_protected_engine_ab_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_engine_ab_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_ENGINE_AB_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


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
