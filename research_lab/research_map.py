"""Phase 1 static Research Map v0 records.

P1.7 builds the local projection model for the public Research Map: sanitized
failure-board cells, component status, frontier summaries, allocator
predictions, and static map artifacts. It does not publish a live API, mutate
miner workflows, write production data, or expose live champion / sealed eval
material.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .engine_v1 import PatchType
from .loop_game import FailureBoardItem, validate_failure_board_item, verify_research_lab_loop_game
from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    ReleasePolicyRecord,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    validate_release_policy,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "research_map_fixtures.json"

PUBLIC_MAP_RELEASE_STATES = {
    ArtifactReleaseState.PUBLIC_RECEIPT.value,
    ArtifactReleaseState.SANITIZED_TRACE.value,
    ArtifactReleaseState.DELAYED_RELEASE.value,
    ArtifactReleaseState.RETIRED_PUBLIC_ARTIFACT.value,
}

PROTECTED_MAP_KEYS = {
    "raw_content",
    "raw_text",
    "raw_snapshot",
    "raw_customer_data",
    "private_customer_data",
    "customer_email",
    "lead_email",
    "judge_prompt",
    "sealed_judge_prompt",
    "sealed_eval_details",
    "eval_secret",
    "live_champion_prompt",
    "live_champion_code",
    "live_champion_weights",
    "champion_source",
}

PROTECTED_STRING_MARKERS = (
    "sealed judge prompt",
    "sealed eval",
    "raw customer",
    "private customer",
    "live champion prompt",
    "live champion code",
    "live champion weights",
)


class ComponentMapStatus(str, Enum):
    ACTIVE = "active"
    WATCH = "watch"
    DEFERRED = "deferred"
    RETIRED = "retired"


class MapProjectionState(str, Enum):
    STATIC_LOCAL_ONLY = "static_local_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AllocatorPredictionRecord:
    prediction_id: str
    cell_ref: str
    island: str
    target_component: str
    patch_type: str
    predicted_delta: float
    confidence: float
    expected_cost_cents: int
    expected_value_score: float
    provenance_refs: tuple[str, ...]
    model_version_ref: str
    is_prediction: bool = True
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AllocatorPredictionRecord":
        return cls(
            prediction_id=str(data["prediction_id"]),
            cell_ref=str(data["cell_ref"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            patch_type=str(data["patch_type"]),
            predicted_delta=float(data["predicted_delta"]),
            confidence=float(data["confidence"]),
            expected_cost_cents=int(data["expected_cost_cents"]),
            expected_value_score=float(data["expected_value_score"]),
            provenance_refs=tuple(str(item) for item in data.get("provenance_refs", [])),
            model_version_ref=str(data["model_version_ref"]),
            is_prediction=bool(data.get("is_prediction", True)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance_refs"] = list(self.provenance_refs)
        return data


@dataclass(frozen=True)
class FrontierPointRecord:
    point_id: str
    run_ref: str
    receipt_ref: str
    patch_ref: str
    score: float
    delta_vs_parent: float
    cost_cents: int
    status: str
    public_summary: str
    kept: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrontierPointRecord":
        return cls(
            point_id=str(data["point_id"]),
            run_ref=str(data["run_ref"]),
            receipt_ref=str(data["receipt_ref"]),
            patch_ref=str(data["patch_ref"]),
            score=float(data["score"]),
            delta_vs_parent=float(data["delta_vs_parent"]),
            cost_cents=int(data["cost_cents"]),
            status=str(data["status"]),
            public_summary=str(data["public_summary"]),
            kept=bool(data["kept"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierSummaryRecord:
    frontier_id: str
    island: str
    target_component: str
    points: tuple[FrontierPointRecord, ...]
    running_best_score: float
    keep_rate: float
    cost_per_kept_patch_cents: int
    crash_rate: float
    source_ledger_refs: tuple[str, ...]
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrontierSummaryRecord":
        return cls(
            frontier_id=str(data["frontier_id"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            points=tuple(FrontierPointRecord.from_mapping(item) for item in data.get("points", [])),
            running_best_score=float(data["running_best_score"]),
            keep_rate=float(data["keep_rate"]),
            cost_per_kept_patch_cents=int(data["cost_per_kept_patch_cents"]),
            crash_rate=float(data["crash_rate"]),
            source_ledger_refs=tuple(str(item) for item in data.get("source_ledger_refs", [])),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [point.to_dict() for point in self.points]
        data["source_ledger_refs"] = list(self.source_ledger_refs)
        return data


@dataclass(frozen=True)
class ComponentStatusRecord:
    component_ref: str
    island: str
    component_name: str
    status: str
    registry_ref: str
    public_summary: str
    allowed_patch_types: tuple[str, ...]
    current_patch_seq: int
    public_receipt_refs: tuple[str, ...]
    sanitized_trace_refs: tuple[str, ...]
    withheld_private_artifact_count: int = 0
    live_champion_artifact_public: bool = False
    sealed_eval_details_included: bool = False
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentStatusRecord":
        return cls(
            component_ref=str(data["component_ref"]),
            island=str(data["island"]),
            component_name=str(data["component_name"]),
            status=str(data["status"]),
            registry_ref=str(data["registry_ref"]),
            public_summary=str(data["public_summary"]),
            allowed_patch_types=tuple(str(item) for item in data.get("allowed_patch_types", [])),
            current_patch_seq=int(data["current_patch_seq"]),
            public_receipt_refs=tuple(str(item) for item in data.get("public_receipt_refs", [])),
            sanitized_trace_refs=tuple(str(item) for item in data.get("sanitized_trace_refs", [])),
            withheld_private_artifact_count=int(data.get("withheld_private_artifact_count", 0)),
            live_champion_artifact_public=bool(data.get("live_champion_artifact_public", False)),
            sealed_eval_details_included=bool(data.get("sealed_eval_details_included", False)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_patch_types"] = list(self.allowed_patch_types)
        data["public_receipt_refs"] = list(self.public_receipt_refs)
        data["sanitized_trace_refs"] = list(self.sanitized_trace_refs)
        return data


@dataclass(frozen=True)
class ResearchMapCellRecord:
    cell_id: str
    island: str
    target_component: str
    patch_type: str
    failure_board_item_ref: str
    sanitized_failure_summary: str
    sanitized_examples: tuple[str, ...]
    recent_case_count: int
    run_density_7d: int
    achieved_delta_mean: float
    frontier_summary_ref: str
    allocator_prediction_ref: str
    source_receipt_refs: tuple[str, ...]
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMapCellRecord":
        return cls(
            cell_id=str(data["cell_id"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            patch_type=str(data["patch_type"]),
            failure_board_item_ref=str(data["failure_board_item_ref"]),
            sanitized_failure_summary=str(data["sanitized_failure_summary"]),
            sanitized_examples=tuple(str(item) for item in data.get("sanitized_examples", [])),
            recent_case_count=int(data["recent_case_count"]),
            run_density_7d=int(data["run_density_7d"]),
            achieved_delta_mean=float(data["achieved_delta_mean"]),
            frontier_summary_ref=str(data["frontier_summary_ref"]),
            allocator_prediction_ref=str(data["allocator_prediction_ref"]),
            source_receipt_refs=tuple(str(item) for item in data.get("source_receipt_refs", [])),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sanitized_examples"] = list(self.sanitized_examples)
        data["source_receipt_refs"] = list(self.source_receipt_refs)
        return data


@dataclass(frozen=True)
class ResearchMapProjectionRecord:
    projection_id: str
    generated_for_date: str
    map_version: str
    component_statuses: tuple[ComponentStatusRecord, ...]
    cells: tuple[ResearchMapCellRecord, ...]
    frontier_summaries: tuple[FrontierSummaryRecord, ...]
    allocator_predictions: tuple[AllocatorPredictionRecord, ...]
    source_ledger_refs: tuple[str, ...]
    state: str = MapProjectionState.STATIC_LOCAL_ONLY.value
    local_only: bool = True
    static_generation: bool = True
    daily_regeneration_cadence: str = "daily"
    live_api_enabled: bool = False
    production_publish_enabled: bool = False
    public_champion_artifacts_included: bool = False
    unreleased_eval_details_included: bool = False
    raw_customer_data_included: bool = False
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMapProjectionRecord":
        return cls(
            projection_id=str(data["projection_id"]),
            generated_for_date=str(data["generated_for_date"]),
            map_version=str(data["map_version"]),
            component_statuses=tuple(
                ComponentStatusRecord.from_mapping(item) for item in data.get("component_statuses", [])
            ),
            cells=tuple(ResearchMapCellRecord.from_mapping(item) for item in data.get("cells", [])),
            frontier_summaries=tuple(
                FrontierSummaryRecord.from_mapping(item) for item in data.get("frontier_summaries", [])
            ),
            allocator_predictions=tuple(
                AllocatorPredictionRecord.from_mapping(item) for item in data.get("allocator_predictions", [])
            ),
            source_ledger_refs=tuple(str(item) for item in data.get("source_ledger_refs", [])),
            state=str(data.get("state", MapProjectionState.STATIC_LOCAL_ONLY.value)),
            local_only=bool(data.get("local_only", True)),
            static_generation=bool(data.get("static_generation", True)),
            daily_regeneration_cadence=str(data.get("daily_regeneration_cadence", "daily")),
            live_api_enabled=bool(data.get("live_api_enabled", False)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            public_champion_artifacts_included=bool(data.get("public_champion_artifacts_included", False)),
            unreleased_eval_details_included=bool(data.get("unreleased_eval_details_included", False)),
            raw_customer_data_included=bool(data.get("raw_customer_data_included", False)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["component_statuses"] = [record.to_dict() for record in self.component_statuses]
        data["cells"] = [record.to_dict() for record in self.cells]
        data["frontier_summaries"] = [record.to_dict() for record in self.frontier_summaries]
        data["allocator_predictions"] = [record.to_dict() for record in self.allocator_predictions]
        data["source_ledger_refs"] = list(self.source_ledger_refs)
        return data


@dataclass(frozen=True)
class ResearchMapArtifactRecord:
    artifact_ref: str
    projection_id: str
    generated_for_date: str
    artifact_kind: str
    static_uri: str
    artifact_hash: str
    local_only: bool = True
    live_api_enabled: bool = False
    production_publish_enabled: bool = False
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMapArtifactRecord":
        return cls(
            artifact_ref=str(data["artifact_ref"]),
            projection_id=str(data["projection_id"]),
            generated_for_date=str(data["generated_for_date"]),
            artifact_kind=str(data["artifact_kind"]),
            static_uri=str(data["static_uri"]),
            artifact_hash=str(data["artifact_hash"]),
            local_only=bool(data.get("local_only", True)),
            live_api_enabled=bool(data.get("live_api_enabled", False)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_research_map_projection(
    *,
    generated_for_date: str,
    map_version: str,
    component_statuses: Sequence[ComponentStatusRecord],
    cells: Sequence[ResearchMapCellRecord],
    frontier_summaries: Sequence[FrontierSummaryRecord],
    allocator_predictions: Sequence[AllocatorPredictionRecord],
    source_ledger_refs: Sequence[str],
) -> ResearchMapProjectionRecord:
    payload = {
        "generated_for_date": generated_for_date,
        "map_version": map_version,
        "component_refs": [record.component_ref for record in component_statuses],
        "cell_ids": [record.cell_id for record in cells],
        "frontier_ids": [record.frontier_id for record in frontier_summaries],
        "prediction_ids": [record.prediction_id for record in allocator_predictions],
    }
    return ResearchMapProjectionRecord(
        projection_id="research_map_projection:" + sha256_json(payload).split(":", 1)[1][:16],
        generated_for_date=generated_for_date,
        map_version=map_version,
        component_statuses=tuple(component_statuses),
        cells=tuple(cells),
        frontier_summaries=tuple(frontier_summaries),
        allocator_predictions=tuple(allocator_predictions),
        source_ledger_refs=tuple(source_ledger_refs),
    )


def build_research_map_artifact(
    projection: ResearchMapProjectionRecord,
    *,
    static_uri: str,
    artifact_kind: str = "static_json",
) -> ResearchMapArtifactRecord:
    artifact_hash = sha256_json(projection.to_dict())
    payload = {
        "projection_id": projection.projection_id,
        "artifact_hash": artifact_hash,
        "static_uri": static_uri,
    }
    return ResearchMapArtifactRecord(
        artifact_ref="research_map_artifact:" + sha256_json(payload).split(":", 1)[1][:16],
        projection_id=projection.projection_id,
        generated_for_date=projection.generated_for_date,
        artifact_kind=artifact_kind,
        static_uri=static_uri,
        artifact_hash=artifact_hash,
    )


def validate_allocator_prediction_record(record: AllocatorPredictionRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, AllocatorPredictionRecord):
        record = AllocatorPredictionRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, "map_projection", record.prediction_id))
    if record.patch_type not in {patch.value for patch in PatchType}:
        errors.append(f"unknown patch_type: {record.patch_type}")
    if not record.is_prediction:
        errors.append("allocator row must be marked is_prediction=true")
    if not 0 <= record.confidence <= 1:
        errors.append("allocator prediction confidence must be in [0, 1]")
    if record.expected_cost_cents < 0:
        errors.append("expected_cost_cents must be non-negative")
    if not record.provenance_refs:
        errors.append("allocator prediction must include provenance refs")
    if not record.model_version_ref:
        errors.append("allocator prediction must include model_version_ref")
    return errors


def validate_frontier_summary_record(record: FrontierSummaryRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, FrontierSummaryRecord):
        record = FrontierSummaryRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, "map_projection", record.frontier_id))
    if not record.points:
        errors.append("frontier summary must include points")
    if not 0 <= record.keep_rate <= 1:
        errors.append("keep_rate must be in [0, 1]")
    if not 0 <= record.crash_rate <= 1:
        errors.append("crash_rate must be in [0, 1]")
    if record.cost_per_kept_patch_cents < 0:
        errors.append("cost_per_kept_patch_cents must be non-negative")
    if not record.source_ledger_refs:
        errors.append("frontier summary must include source ledger refs")
    if record.points:
        max_score = max(point.score for point in record.points)
        if abs(record.running_best_score - max_score) > 1e-9:
            errors.append("running_best_score must equal the max frontier point score")
    for point in record.points:
        errors.extend(_validate_frontier_point(point))
    return errors


def validate_component_status_record(record: ComponentStatusRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, ComponentStatusRecord):
        record = ComponentStatusRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, "map_projection", record.component_ref))
    if record.status not in {status.value for status in ComponentMapStatus}:
        errors.append(f"unknown component status: {record.status}")
    if not record.allowed_patch_types:
        errors.append("component status must include allowed_patch_types")
    unknown_patch_types = [patch_type for patch_type in record.allowed_patch_types if patch_type not in {p.value for p in PatchType}]
    if unknown_patch_types:
        errors.append("component status has unknown patch types: " + ", ".join(unknown_patch_types))
    if record.current_patch_seq < 0:
        errors.append("current_patch_seq must be non-negative")
    if record.live_champion_artifact_public:
        errors.append("current live champion artifacts must not be public in the research map")
    if record.sealed_eval_details_included:
        errors.append("sealed eval details must not be included in the research map")
    if not (record.public_receipt_refs or record.sanitized_trace_refs or record.withheld_private_artifact_count):
        errors.append("component status must include public refs or a withheld private artifact count")
    return errors


def validate_research_map_cell_record(record: ResearchMapCellRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, ResearchMapCellRecord):
        record = ResearchMapCellRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, "map_projection", record.cell_id))
    if record.patch_type not in {patch.value for patch in PatchType}:
        errors.append(f"unknown patch_type: {record.patch_type}")
    if not record.sanitized_failure_summary:
        errors.append("map cell requires sanitized_failure_summary")
    if not record.sanitized_examples:
        errors.append("map cell requires sanitized examples")
    for example in record.sanitized_examples:
        if len(example) > 240:
            errors.append("sanitized examples must be concise")
    if record.recent_case_count < 0:
        errors.append("recent_case_count must be non-negative")
    if record.run_density_7d < 0:
        errors.append("run_density_7d must be non-negative")
    if not record.source_receipt_refs:
        errors.append("map cell requires source receipt refs")
    return errors


def validate_research_map_projection_record(
    record: ResearchMapProjectionRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, ResearchMapProjectionRecord):
        record = ResearchMapProjectionRecord.from_mapping(record)
    errors = list(raw_errors)
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(_validate_public_policy(record, "map_projection", record.projection_id))
    if record.state not in {state.value for state in MapProjectionState}:
        errors.append(f"unknown map projection state: {record.state}")
    if record.state != MapProjectionState.STATIC_LOCAL_ONLY.value:
        errors.append("P1.7 map projection must remain static_local_only")
    if not record.local_only:
        errors.append("P1.7 map projection must remain local_only")
    if not record.static_generation:
        errors.append("P1.7 map projection must be static_generation")
    if record.daily_regeneration_cadence != "daily":
        errors.append("Research Map v0 cadence must be daily")
    if record.live_api_enabled:
        errors.append("Research Map v0 must not enable a live API")
    if record.production_publish_enabled:
        errors.append("Research Map v0 must not enable production publishing")
    if record.public_champion_artifacts_included:
        errors.append("Research Map must not include current live champion artifacts")
    if record.unreleased_eval_details_included:
        errors.append("Research Map must not include unreleased eval details")
    if record.raw_customer_data_included:
        errors.append("Research Map must not include raw customer data")
    if not record.source_ledger_refs:
        errors.append("map projection must include source ledger refs")
    if not record.component_statuses:
        errors.append("map projection must include component statuses")
    if not record.cells:
        errors.append("map projection must include cells")
    if not record.frontier_summaries:
        errors.append("map projection must include frontier summaries")
    if not record.allocator_predictions:
        errors.append("map projection must include allocator predictions")

    component_names = {component.component_name for component in record.component_statuses}
    frontier_refs = {frontier.frontier_id for frontier in record.frontier_summaries}
    prediction_refs = {prediction.prediction_id for prediction in record.allocator_predictions}
    for component in record.component_statuses:
        errors.extend(validate_component_status_record(component))
    for frontier in record.frontier_summaries:
        errors.extend(validate_frontier_summary_record(frontier))
    for prediction in record.allocator_predictions:
        errors.extend(validate_allocator_prediction_record(prediction))
    for cell in record.cells:
        errors.extend(validate_research_map_cell_record(cell))
        if cell.target_component not in component_names:
            errors.append(f"map cell target_component not in component statuses: {cell.target_component}")
        if cell.frontier_summary_ref not in frontier_refs:
            errors.append(f"map cell frontier_summary_ref missing: {cell.frontier_summary_ref}")
        if cell.allocator_prediction_ref not in prediction_refs:
            errors.append(f"map cell allocator_prediction_ref missing: {cell.allocator_prediction_ref}")
    return errors


def validate_research_map_artifact_record(
    record: ResearchMapArtifactRecord | Mapping[str, Any],
    *,
    projection: Optional[ResearchMapProjectionRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw_errors = _protected_material_errors(record)
    if not isinstance(record, ResearchMapArtifactRecord):
        record = ResearchMapArtifactRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, "map_projection", record.artifact_ref))
    if record.artifact_kind not in {"static_json", "static_markdown"}:
        errors.append("Research Map artifact kind must be static_json or static_markdown")
    if not record.static_uri.startswith("local://"):
        errors.append("Research Map v0 artifact URI must remain local://")
    if not record.artifact_hash.startswith("sha256:"):
        errors.append("Research Map artifact_hash must be sha256-prefixed")
    if not record.local_only:
        errors.append("Research Map artifact must remain local_only")
    if record.live_api_enabled:
        errors.append("Research Map artifact must not enable live API")
    if record.production_publish_enabled:
        errors.append("Research Map artifact must not enable production publishing")
    if projection is not None:
        if not isinstance(projection, ResearchMapProjectionRecord):
            projection = ResearchMapProjectionRecord.from_mapping(projection)
        if record.projection_id != projection.projection_id:
            errors.append("Research Map artifact projection_id mismatch")
        expected_hash = sha256_json(projection.to_dict())
        if record.artifact_hash != expected_hash:
            errors.append("Research Map artifact_hash does not match projection")
    return errors


def verify_research_lab_research_map(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    loop_summary = verify_research_lab_loop_game()
    fixture = _load_fixture(Path(fixture_path))

    for item in fixture["source_failure_board_items"]:
        failure_board_errors = validate_failure_board_item(FailureBoardItem.from_mapping(item))
        _assert(not failure_board_errors, "source failure-board item validates")

    components = tuple(ComponentStatusRecord.from_mapping(item) for item in fixture["component_statuses"])
    frontiers = tuple(FrontierSummaryRecord.from_mapping(item) for item in fixture["frontier_summaries"])
    predictions = tuple(AllocatorPredictionRecord.from_mapping(item) for item in fixture["allocator_predictions"])
    cells = tuple(ResearchMapCellRecord.from_mapping(item) for item in fixture["cells"])
    projection = build_research_map_projection(
        generated_for_date=fixture["generated_for_date"],
        map_version=fixture["map_version"],
        component_statuses=components,
        cells=cells,
        frontier_summaries=frontiers,
        allocator_predictions=predictions,
        source_ledger_refs=fixture["source_ledger_refs"],
    )
    projection_errors = validate_research_map_projection_record(projection)
    _assert(not projection_errors, "research map projection validates")
    artifact = build_research_map_artifact(
        projection,
        static_uri=fixture["static_uri"],
        artifact_kind=fixture.get("artifact_kind", "static_json"),
    )
    artifact_errors = validate_research_map_artifact_record(artifact, projection=projection)
    _assert(not artifact_errors, "research map artifact validates")

    for record in fixture["invalid_cross_reference_cells"]:
        dangling_cell = ResearchMapCellRecord.from_mapping(record)
        dangling_projection = build_research_map_projection(
            generated_for_date=fixture["generated_for_date"],
            map_version=fixture["map_version"],
            component_statuses=components,
            cells=[dangling_cell],
            frontier_summaries=frontiers,
            allocator_predictions=predictions,
            source_ledger_refs=fixture["source_ledger_refs"],
        )
        dangling_errors = validate_research_map_projection_record(dangling_projection)
        _assert(dangling_errors, f"dangling map cell cross-reference fails: {record['cell_id']}")
        _assert_expected_error(dangling_errors, record)

    for record in fixture["invalid_allocator_predictions"]:
        errors = validate_allocator_prediction_record(record)
        _assert(errors, f"invalid allocator prediction fails: {record['prediction_id']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_component_statuses"]:
        errors = validate_component_status_record(record)
        _assert(errors, f"invalid component status fails: {record['component_ref']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_frontier_summaries"]:
        errors = validate_frontier_summary_record(record)
        _assert(errors, f"invalid frontier summary fails: {record['frontier_id']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_cells"]:
        errors = validate_research_map_cell_record(record)
        _assert(errors, f"invalid map cell fails: {record['cell_id']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_projections"]:
        errors = validate_research_map_projection_record(record)
        _assert(errors, f"invalid projection fails: {record['projection_id']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_artifacts"]:
        errors = validate_research_map_artifact_record(record, projection=projection)
        _assert(errors, f"invalid artifact fails: {record['artifact_ref']}")
        _assert_expected_error(errors, record)

    unsafe_errors = validate_research_map_projection_record(
        projection,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe workflow guards block map projection")

    return {
        "loop_failure_board_items": loop_summary["failure_board_items"],
        "projection_id": projection.projection_id,
        "artifact_ref": artifact.artifact_ref,
        "cell_count": len(projection.cells),
        "allocator_predictions": len(projection.allocator_predictions),
    }


def _validate_frontier_point(point: FrontierPointRecord) -> list[str]:
    errors: list[str] = []
    if point.cost_cents < 0:
        errors.append("frontier point cost_cents must be non-negative")
    if point.status not in {"keep", "discard", "crash", "timeout"}:
        errors.append(f"unknown frontier point status: {point.status}")
    if not point.public_summary:
        errors.append("frontier point requires public_summary")
    if point.kept and point.status != "keep":
        errors.append("kept frontier point must have status keep")
    return errors


def _validate_public_policy(record: Any, artifact_type: str, artifact_ref: str) -> list[str]:
    errors: list[str] = []
    if not getattr(record, "protected_material_flags_checked", False):
        errors.append("protected-material flags must be checked before public map release")
    state = getattr(record, "artifact_release_state", "")
    if state not in PUBLIC_MAP_RELEASE_STATES:
        errors.append("research map output must use a public-safe release state")
    policy = ReleasePolicyRecord(
        artifact_ref=artifact_ref,
        artifact_type=artifact_type,
        visibility_policy=getattr(record, "visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value),
        artifact_release_state=state,
        contains_live_champion_ip=bool(getattr(record, "contains_live_champion_ip", False)),
        contains_sealed_eval_details=bool(getattr(record, "contains_sealed_eval_details", False)),
        contains_raw_evidence_snapshot=bool(getattr(record, "contains_raw_evidence_snapshot", False)),
        contains_private_customer_data=bool(getattr(record, "contains_private_customer_data", False)),
        contains_judge_prompts=bool(getattr(record, "contains_judge_prompts", False)),
        release_condition="static local Research Map v0 projection",
        reason="P1.7 sanitized research map projection",
    )
    errors.extend(validate_release_policy(policy))
    return errors


def _protected_material_errors(record: Any) -> list[str]:
    if hasattr(record, "to_dict"):
        payload = record.to_dict()
    else:
        payload = record
    found = sorted(_find_protected_map_material(payload))
    if not found:
        return []
    return ["research map payload contains protected or raw material keys/markers: " + ", ".join(found)]


def _find_protected_map_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_MAP_KEYS:
                found.add(key_path)
            found.update(_find_protected_map_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_map_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_STRING_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if expected:
        _assert(any(str(expected) in error for error in errors), f"expected error {expected!r}")


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
