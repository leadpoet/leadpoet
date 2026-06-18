"""Phase 2 Research Map v1 API-shape contracts.

P2.5 defines local records for the future live novelty-quoting API and public
Research Map v1 projection. It does not start an API server, read production
traffic, query production map data, publish live artifacts, or write to
production systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .component_market import (
    PROTECTED_COMPONENT_KEYS,
    PROTECTED_COMPONENT_MARKERS,
)
from .engine_v1 import PatchType
from .meta_allocator import (
    PROTECTED_META_ALLOCATOR_KEYS,
    PROTECTED_META_ALLOCATOR_MARKERS,
    verify_research_lab_meta_allocator,
)
from .loop_foundation import (
    ArtifactReleaseState,
    ReleasePolicyRecord,
    VisibilityPolicy,
    validate_release_policy,
)
from .market_foundation import (
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    verify_market_foundation,
)
from .research_map import verify_research_lab_research_map


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "research_map_v1_api_fixtures.json"

RESEARCH_MAP_V1_API_VERSION = "research_map:v1:local_api_contract"
RESEARCH_MAP_V1_NOVELTY_ENDPOINT = "/research-map/v1/novelty/quote"

PROTECTED_MAP_V1_KEYS = set(PROTECTED_COMPONENT_KEYS) | set(PROTECTED_META_ALLOCATOR_KEYS) | {
    "api_key",
    "customer_outcome",
    "live_api_token",
    "private_map_index",
    "production_customer_data",
    "raw_brief",
    "raw_customer_data",
    "raw_icp",
    "raw_query",
    "raw_similarity_embedding",
    "sealed_novelty_index",
    "supabase_service_role_key",
}

PROTECTED_MAP_V1_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_COMPONENT_MARKERS)
        | set(PROTECTED_META_ALLOCATOR_MARKERS)
        | {
            "live api token",
            "private map index",
            "production customer",
            "raw brief",
            "raw icp",
            "raw query",
            "raw similarity",
            "sealed novelty",
            "service_role",
            "supabase service",
        }
    )
)


class NoveltyQuoteState(str, Enum):
    LOCAL_STUB = "local_stub"
    READY_AFTER_PRODUCTION_MAP = "ready_after_production_map"
    BLOCKED = "blocked"


class ResearchMapV1ProjectionState(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    READY_AFTER_PRODUCTION_DATA = "ready_after_production_data"
    BLOCKED = "blocked"


class ResearchMapV1ArtifactState(str, Enum):
    LOCAL_CONTRACT_ARTIFACT = "local_contract_artifact"
    READY_AFTER_PRODUCTION_PUBLISH = "ready_after_production_publish"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class NoveltyQuoteRequestRecord:
    request_id: str
    requester_ref: str
    island: str
    target_component: str
    patch_type: str
    sanitized_brief_ref: str
    sanitized_query_terms: tuple[str, ...]
    source_projection_ref: str
    max_neighbors: int = 5
    api_version: str = RESEARCH_MAP_V1_API_VERSION
    endpoint_path: str = RESEARCH_MAP_V1_NOVELTY_ENDPOINT
    uses_local_fixtures: bool = True
    uses_production_map_data: bool = False
    live_api_request: bool = False
    public_api_served: bool = False
    production_traffic: bool = False
    supabase_read_enabled: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NoveltyQuoteRequestRecord":
        return cls(
            request_id=str(data["request_id"]),
            requester_ref=str(data["requester_ref"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            patch_type=str(data["patch_type"]),
            sanitized_brief_ref=str(data["sanitized_brief_ref"]),
            sanitized_query_terms=tuple(str(item) for item in data.get("sanitized_query_terms", [])),
            source_projection_ref=str(data["source_projection_ref"]),
            max_neighbors=int(data.get("max_neighbors", 5)),
            api_version=str(data.get("api_version", RESEARCH_MAP_V1_API_VERSION)),
            endpoint_path=str(data.get("endpoint_path", RESEARCH_MAP_V1_NOVELTY_ENDPOINT)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            uses_production_map_data=bool(data.get("uses_production_map_data", False)),
            live_api_request=bool(data.get("live_api_request", False)),
            public_api_served=bool(data.get("public_api_served", False)),
            production_traffic=bool(data.get("production_traffic", False)),
            supabase_read_enabled=bool(data.get("supabase_read_enabled", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sanitized_query_terms"] = list(self.sanitized_query_terms)
        return data


@dataclass(frozen=True)
class NoveltyQuoteResponseRecord:
    quote_id: str
    request_id: str
    decision: str
    novelty_score: float
    nearest_prior_receipt_refs: tuple[str, ...]
    nearest_cell_refs: tuple[str, ...]
    suggested_cell_refs: tuple[str, ...]
    quote_explanation_sanitized: str
    source_projection_ref: str
    source_prediction_refs: tuple[str, ...]
    state: str = NoveltyQuoteState.LOCAL_STUB.value
    api_version: str = RESEARCH_MAP_V1_API_VERSION
    uses_local_fixtures: bool = True
    uses_production_map_data: bool = False
    live_quote: bool = False
    public_api_response: bool = False
    production_data_ready: bool = False
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NoveltyQuoteResponseRecord":
        return cls(
            quote_id=str(data["quote_id"]),
            request_id=str(data["request_id"]),
            decision=str(data["decision"]),
            novelty_score=float(data["novelty_score"]),
            nearest_prior_receipt_refs=tuple(str(item) for item in data.get("nearest_prior_receipt_refs", [])),
            nearest_cell_refs=tuple(str(item) for item in data.get("nearest_cell_refs", [])),
            suggested_cell_refs=tuple(str(item) for item in data.get("suggested_cell_refs", [])),
            quote_explanation_sanitized=str(data["quote_explanation_sanitized"]),
            source_projection_ref=str(data["source_projection_ref"]),
            source_prediction_refs=tuple(str(item) for item in data.get("source_prediction_refs", [])),
            state=str(data.get("state", NoveltyQuoteState.LOCAL_STUB.value)),
            api_version=str(data.get("api_version", RESEARCH_MAP_V1_API_VERSION)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            uses_production_map_data=bool(data.get("uses_production_map_data", False)),
            live_quote=bool(data.get("live_quote", False)),
            public_api_response=bool(data.get("public_api_response", False)),
            production_data_ready=bool(data.get("production_data_ready", False)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nearest_prior_receipt_refs"] = list(self.nearest_prior_receipt_refs)
        data["nearest_cell_refs"] = list(self.nearest_cell_refs)
        data["suggested_cell_refs"] = list(self.suggested_cell_refs)
        data["source_prediction_refs"] = list(self.source_prediction_refs)
        return data


@dataclass(frozen=True)
class ResearchMapV1ProjectionRecord:
    projection_id: str
    generated_for_date: str
    map_version: str
    source_v0_projection_ref: str
    source_meta_allocator_selection_ref: str
    novelty_index_ref: str
    public_cell_refs: tuple[str, ...]
    allocator_prediction_refs: tuple[str, ...]
    component_registry_refs: tuple[str, ...]
    production_results_ledger_refs: tuple[str, ...]
    quote_endpoint_path: str = RESEARCH_MAP_V1_NOVELTY_ENDPOINT
    state: str = ResearchMapV1ProjectionState.LOCAL_CONTRACT_STUB.value
    uses_local_fixtures: bool = True
    production_map_data_ready: bool = False
    live_api_enabled: bool = False
    public_api_server_started: bool = False
    production_publish_enabled: bool = False
    supabase_read_enabled: bool = False
    supabase_write_enabled: bool = False
    local_only: bool = True
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMapV1ProjectionRecord":
        return cls(
            projection_id=str(data["projection_id"]),
            generated_for_date=str(data["generated_for_date"]),
            map_version=str(data["map_version"]),
            source_v0_projection_ref=str(data["source_v0_projection_ref"]),
            source_meta_allocator_selection_ref=str(data["source_meta_allocator_selection_ref"]),
            novelty_index_ref=str(data["novelty_index_ref"]),
            public_cell_refs=tuple(str(item) for item in data.get("public_cell_refs", [])),
            allocator_prediction_refs=tuple(str(item) for item in data.get("allocator_prediction_refs", [])),
            component_registry_refs=tuple(str(item) for item in data.get("component_registry_refs", [])),
            production_results_ledger_refs=tuple(str(item) for item in data.get("production_results_ledger_refs", [])),
            quote_endpoint_path=str(data.get("quote_endpoint_path", RESEARCH_MAP_V1_NOVELTY_ENDPOINT)),
            state=str(data.get("state", ResearchMapV1ProjectionState.LOCAL_CONTRACT_STUB.value)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            production_map_data_ready=bool(data.get("production_map_data_ready", False)),
            live_api_enabled=bool(data.get("live_api_enabled", False)),
            public_api_server_started=bool(data.get("public_api_server_started", False)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            supabase_read_enabled=bool(data.get("supabase_read_enabled", False)),
            supabase_write_enabled=bool(data.get("supabase_write_enabled", False)),
            local_only=bool(data.get("local_only", True)),
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
        data["public_cell_refs"] = list(self.public_cell_refs)
        data["allocator_prediction_refs"] = list(self.allocator_prediction_refs)
        data["component_registry_refs"] = list(self.component_registry_refs)
        data["production_results_ledger_refs"] = list(self.production_results_ledger_refs)
        return data


@dataclass(frozen=True)
class ResearchMapV1PublicArtifactRecord:
    artifact_ref: str
    projection_id: str
    artifact_kind: str
    local_uri: str
    artifact_hash: str
    api_version: str = RESEARCH_MAP_V1_API_VERSION
    endpoint_path: str = RESEARCH_MAP_V1_NOVELTY_ENDPOINT
    state: str = ResearchMapV1ArtifactState.LOCAL_CONTRACT_ARTIFACT.value
    local_only: bool = True
    public_api_server_started: bool = False
    live_api_enabled: bool = False
    production_publish_enabled: bool = False
    production_traffic_enabled: bool = False
    supabase_read_enabled: bool = False
    supabase_write_enabled: bool = False
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMapV1PublicArtifactRecord":
        return cls(
            artifact_ref=str(data["artifact_ref"]),
            projection_id=str(data["projection_id"]),
            artifact_kind=str(data["artifact_kind"]),
            local_uri=str(data["local_uri"]),
            artifact_hash=str(data["artifact_hash"]),
            api_version=str(data.get("api_version", RESEARCH_MAP_V1_API_VERSION)),
            endpoint_path=str(data.get("endpoint_path", RESEARCH_MAP_V1_NOVELTY_ENDPOINT)),
            state=str(data.get("state", ResearchMapV1ArtifactState.LOCAL_CONTRACT_ARTIFACT.value)),
            local_only=bool(data.get("local_only", True)),
            public_api_server_started=bool(data.get("public_api_server_started", False)),
            live_api_enabled=bool(data.get("live_api_enabled", False)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            production_traffic_enabled=bool(data.get("production_traffic_enabled", False)),
            supabase_read_enabled=bool(data.get("supabase_read_enabled", False)),
            supabase_write_enabled=bool(data.get("supabase_write_enabled", False)),
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


def build_research_map_v1_projection(
    *,
    generated_for_date: str,
    source_v0_projection_ref: str,
    source_meta_allocator_selection_ref: str,
    novelty_index_ref: str,
    public_cell_refs: Sequence[str],
    allocator_prediction_refs: Sequence[str],
    component_registry_refs: Sequence[str],
) -> ResearchMapV1ProjectionRecord:
    payload = {
        "generated_for_date": generated_for_date,
        "source_v0_projection_ref": source_v0_projection_ref,
        "source_meta_allocator_selection_ref": source_meta_allocator_selection_ref,
        "novelty_index_ref": novelty_index_ref,
        "public_cell_refs": list(public_cell_refs),
        "allocator_prediction_refs": list(allocator_prediction_refs),
        "component_registry_refs": list(component_registry_refs),
    }
    return ResearchMapV1ProjectionRecord(
        projection_id="research_map_v1_projection:" + sha256_json(payload).split(":", 1)[1][:16],
        generated_for_date=generated_for_date,
        map_version="v1",
        source_v0_projection_ref=source_v0_projection_ref,
        source_meta_allocator_selection_ref=source_meta_allocator_selection_ref,
        novelty_index_ref=novelty_index_ref,
        public_cell_refs=tuple(str(item) for item in public_cell_refs),
        allocator_prediction_refs=tuple(str(item) for item in allocator_prediction_refs),
        component_registry_refs=tuple(str(item) for item in component_registry_refs),
        production_results_ledger_refs=(),
    )


def build_novelty_quote_response(
    *,
    request: NoveltyQuoteRequestRecord | Mapping[str, Any],
    projection: ResearchMapV1ProjectionRecord | Mapping[str, Any],
    decision: str,
    novelty_score: float,
    nearest_prior_receipt_refs: Sequence[str],
    nearest_cell_refs: Sequence[str],
    suggested_cell_refs: Sequence[str],
    quote_explanation_sanitized: str,
) -> NoveltyQuoteResponseRecord:
    if not isinstance(request, NoveltyQuoteRequestRecord):
        request = NoveltyQuoteRequestRecord.from_mapping(request)
    if not isinstance(projection, ResearchMapV1ProjectionRecord):
        projection = ResearchMapV1ProjectionRecord.from_mapping(projection)
    payload = {
        "request_id": request.request_id,
        "projection_id": projection.projection_id,
        "decision": decision,
        "nearest_prior_receipt_refs": list(nearest_prior_receipt_refs),
        "suggested_cell_refs": list(suggested_cell_refs),
    }
    record = NoveltyQuoteResponseRecord(
        quote_id="novelty_quote:" + sha256_json(payload).split(":", 1)[1][:16],
        request_id=request.request_id,
        decision=decision,
        novelty_score=novelty_score,
        nearest_prior_receipt_refs=tuple(str(item) for item in nearest_prior_receipt_refs),
        nearest_cell_refs=tuple(str(item) for item in nearest_cell_refs),
        suggested_cell_refs=tuple(str(item) for item in suggested_cell_refs),
        quote_explanation_sanitized=quote_explanation_sanitized,
        source_projection_ref=projection.projection_id,
        source_prediction_refs=projection.allocator_prediction_refs,
    )
    errors = validate_novelty_quote_response_record(record, request=request, projection=projection)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def build_research_map_v1_public_artifact(
    projection: ResearchMapV1ProjectionRecord | Mapping[str, Any],
    *,
    local_uri: str,
    artifact_kind: str = "api_contract_json",
) -> ResearchMapV1PublicArtifactRecord:
    if not isinstance(projection, ResearchMapV1ProjectionRecord):
        projection = ResearchMapV1ProjectionRecord.from_mapping(projection)
    artifact_hash = sha256_json(projection.to_dict())
    payload = {
        "projection_id": projection.projection_id,
        "artifact_hash": artifact_hash,
        "local_uri": local_uri,
        "artifact_kind": artifact_kind,
    }
    record = ResearchMapV1PublicArtifactRecord(
        artifact_ref="research_map_v1_artifact:" + sha256_json(payload).split(":", 1)[1][:16],
        projection_id=projection.projection_id,
        artifact_kind=artifact_kind,
        local_uri=local_uri,
        artifact_hash=artifact_hash,
    )
    errors = validate_research_map_v1_public_artifact_record(record, projection=projection)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_novelty_quote_request_record(record: NoveltyQuoteRequestRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_map_v1_payload_errors(raw)
    if not isinstance(record, NoveltyQuoteRequestRecord):
        record = NoveltyQuoteRequestRecord.from_mapping(record)
    errors = list(raw_errors)
    if not record.request_id.startswith("novelty_quote_request:"):
        errors.append("request_id must be novelty_quote_request:-prefixed")
    if not record.requester_ref.startswith(("miner:", "owner:", "lab:")):
        errors.append("requester_ref must identify a miner, owner, or lab requester")
    if not record.island:
        errors.append("novelty quote request requires island")
    if not record.target_component:
        errors.append("novelty quote request requires target_component")
    if record.patch_type not in {patch.value for patch in PatchType}:
        errors.append(f"unknown patch_type: {record.patch_type}")
    if not record.sanitized_brief_ref.startswith("sanitized_brief:"):
        errors.append("sanitized_brief_ref must be sanitized_brief:-prefixed")
    if not record.sanitized_query_terms:
        errors.append("sanitized_query_terms must not be empty")
    if any(len(term) > 120 for term in record.sanitized_query_terms):
        errors.append("sanitized_query_terms must be concise")
    if not record.source_projection_ref.startswith("research_map_v1_projection:"):
        errors.append("source_projection_ref must be research_map_v1_projection:-prefixed")
    if not 1 <= record.max_neighbors <= 20:
        errors.append("max_neighbors must be between 1 and 20")
    if record.api_version != RESEARCH_MAP_V1_API_VERSION:
        errors.append("api_version must match Research Map v1 contract version")
    if record.endpoint_path != RESEARCH_MAP_V1_NOVELTY_ENDPOINT:
        errors.append("endpoint_path must match the v1 novelty quote endpoint")
    if record.uses_local_fixtures and record.uses_production_map_data:
        errors.append("novelty quote request cannot use local fixtures and production map data simultaneously")
    if record.live_api_request:
        errors.append("live_api_request must remain false")
    if record.public_api_served:
        errors.append("public_api_served must remain false")
    if record.production_traffic:
        errors.append("production_traffic must remain false")
    if record.supabase_read_enabled:
        errors.append("supabase_read_enabled must remain false")
    if not record.local_only:
        errors.append("novelty quote request must remain local_only")
    return errors


def validate_novelty_quote_response_record(
    record: NoveltyQuoteResponseRecord | Mapping[str, Any],
    *,
    request: Optional[NoveltyQuoteRequestRecord | Mapping[str, Any]] = None,
    projection: Optional[ResearchMapV1ProjectionRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_map_v1_payload_errors(raw)
    if not isinstance(record, NoveltyQuoteResponseRecord):
        record = NoveltyQuoteResponseRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, record.quote_id, "novelty quote response"))
    if not record.quote_id.startswith("novelty_quote:"):
        errors.append("quote_id must be novelty_quote:-prefixed")
    if not record.request_id.startswith("novelty_quote_request:"):
        errors.append("request_id must be novelty_quote_request:-prefixed")
    if record.decision not in {"pass", "reject", "needs_more_context", "local_stub"}:
        errors.append(f"unknown novelty quote decision: {record.decision}")
    if not 0.0 <= record.novelty_score <= 1.0:
        errors.append("novelty_score must be in [0, 1]")
    if not record.nearest_prior_receipt_refs:
        errors.append("novelty quote response requires nearest_prior_receipt_refs")
    if not record.nearest_cell_refs:
        errors.append("novelty quote response requires nearest_cell_refs")
    if not record.suggested_cell_refs:
        errors.append("novelty quote response requires suggested_cell_refs")
    if not record.quote_explanation_sanitized:
        errors.append("novelty quote response requires quote_explanation_sanitized")
    if not record.source_projection_ref.startswith("research_map_v1_projection:"):
        errors.append("source_projection_ref must be research_map_v1_projection:-prefixed")
    if not record.source_prediction_refs:
        errors.append("novelty quote response requires source_prediction_refs")
    if record.state not in {state.value for state in NoveltyQuoteState}:
        errors.append(f"unknown novelty quote state: {record.state}")
    if record.api_version != RESEARCH_MAP_V1_API_VERSION:
        errors.append("api_version must match Research Map v1 contract version")
    if record.uses_local_fixtures and record.uses_production_map_data:
        errors.append("novelty quote response cannot use local fixtures and production map data simultaneously")
    if record.state == NoveltyQuoteState.READY_AFTER_PRODUCTION_MAP.value:
        if record.uses_local_fixtures or not record.uses_production_map_data or not record.production_data_ready:
            errors.append("ready_after_production_map quote requires real production map data")
    if record.live_quote:
        errors.append("live_quote must remain false")
    if record.public_api_response:
        errors.append("public_api_response must remain false")
    if record.production_data_ready:
        if record.state != NoveltyQuoteState.READY_AFTER_PRODUCTION_MAP.value:
            errors.append("production_data_ready requires ready_after_production_map state")
        if not record.uses_production_map_data:
            errors.append("production_data_ready requires uses_production_map_data")
    if record.production_data_ready and record.uses_local_fixtures:
        errors.append("local quote responses cannot claim production_data_ready")
    if not record.protected_material_flags_checked:
        errors.append("novelty quote response must check protected-material flags")
    protected_flags = _protected_policy_flags(record)
    if protected_flags:
        errors.append("novelty quote response must not contain protected material flags: " + ", ".join(protected_flags))
    if not record.local_only:
        errors.append("novelty quote response must remain local_only")
    if request is not None:
        if not isinstance(request, NoveltyQuoteRequestRecord):
            request = NoveltyQuoteRequestRecord.from_mapping(request)
        request_errors = validate_novelty_quote_request_record(request)
        if request_errors:
            errors.append("source novelty quote request is invalid: " + "; ".join(request_errors))
        if record.request_id != request.request_id:
            errors.append("novelty quote response request_id mismatch")
    if projection is not None:
        if not isinstance(projection, ResearchMapV1ProjectionRecord):
            projection = ResearchMapV1ProjectionRecord.from_mapping(projection)
        projection_errors = validate_research_map_v1_projection_record(projection)
        if projection_errors:
            errors.append("source Research Map v1 projection is invalid: " + "; ".join(projection_errors))
        if record.source_projection_ref != projection.projection_id:
            errors.append("novelty quote response source_projection_ref mismatch")
        missing_cells = [
            cell_ref
            for cell_ref in (*record.nearest_cell_refs, *record.suggested_cell_refs)
            if cell_ref not in projection.public_cell_refs
        ]
        if missing_cells:
            errors.append("novelty quote response references cells outside projection: " + ", ".join(missing_cells))
        missing_predictions = [ref for ref in record.source_prediction_refs if ref not in projection.allocator_prediction_refs]
        if missing_predictions:
            errors.append("novelty quote response references predictions outside projection: " + ", ".join(missing_predictions))
    return errors


def validate_research_map_v1_projection_record(
    record: ResearchMapV1ProjectionRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_map_v1_payload_errors(raw)
    if not isinstance(record, ResearchMapV1ProjectionRecord):
        record = ResearchMapV1ProjectionRecord.from_mapping(record)
    errors = list(raw_errors)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(_validate_public_policy(record, record.projection_id, "Research Map v1 projection"))
    if not record.projection_id.startswith("research_map_v1_projection:"):
        errors.append("projection_id must be research_map_v1_projection:-prefixed")
    if record.map_version != "v1":
        errors.append("map_version must be v1")
    if not record.source_v0_projection_ref.startswith("research_map_projection:"):
        errors.append("source_v0_projection_ref must be research_map_projection:-prefixed")
    if not record.source_meta_allocator_selection_ref.startswith("meta_allocator_selection:"):
        errors.append("source_meta_allocator_selection_ref must be meta_allocator_selection:-prefixed")
    if not record.novelty_index_ref.startswith("novelty_index:"):
        errors.append("novelty_index_ref must be novelty_index:-prefixed")
    if not record.public_cell_refs:
        errors.append("Research Map v1 projection requires public_cell_refs")
    if not record.allocator_prediction_refs:
        errors.append("Research Map v1 projection requires allocator_prediction_refs")
    if not record.component_registry_refs:
        errors.append("Research Map v1 projection requires component_registry_refs")
    if record.quote_endpoint_path != RESEARCH_MAP_V1_NOVELTY_ENDPOINT:
        errors.append("quote_endpoint_path must match the v1 novelty quote endpoint")
    if record.state not in {state.value for state in ResearchMapV1ProjectionState}:
        errors.append(f"unknown Research Map v1 projection state: {record.state}")
    if record.uses_local_fixtures and record.production_map_data_ready:
        errors.append("local Research Map v1 projection cannot claim production_map_data_ready")
    if record.uses_local_fixtures and record.production_results_ledger_refs:
        errors.append("local Research Map v1 projection cannot include production_results_ledger_refs")
    if record.production_map_data_ready:
        if record.state != ResearchMapV1ProjectionState.READY_AFTER_PRODUCTION_DATA.value:
            errors.append("production_map_data_ready requires ready_after_production_data state")
        if not record.production_results_ledger_refs:
            errors.append("production_map_data_ready requires production_results_ledger_refs")
    if record.state == ResearchMapV1ProjectionState.READY_AFTER_PRODUCTION_DATA.value:
        if record.uses_local_fixtures or not record.production_map_data_ready:
            errors.append("ready_after_production_data projection requires real production map data")
        if not record.production_results_ledger_refs:
            errors.append("ready_after_production_data projection requires production_results_ledger_refs")
    bad_ledger_refs = [
        ref for ref in record.production_results_ledger_refs if not ref.startswith("results_ledger:")
    ]
    if bad_ledger_refs:
        errors.append("production_results_ledger_refs must be results_ledger:-prefixed: " + ", ".join(bad_ledger_refs))
    if record.live_api_enabled:
        errors.append("live_api_enabled must remain false")
    if record.public_api_server_started:
        errors.append("public_api_server_started must remain false")
    if record.production_publish_enabled:
        errors.append("production_publish_enabled must remain false")
    if record.supabase_read_enabled:
        errors.append("supabase_read_enabled must remain false")
    if record.supabase_write_enabled:
        errors.append("supabase_write_enabled must remain false")
    if not record.local_only:
        errors.append("Research Map v1 projection must remain local_only")
    if not record.protected_material_flags_checked:
        errors.append("Research Map v1 projection must check protected-material flags")
    protected_flags = _protected_policy_flags(record)
    if protected_flags:
        errors.append("Research Map v1 projection must not contain protected material flags: " + ", ".join(protected_flags))
    return errors


def validate_research_map_v1_public_artifact_record(
    record: ResearchMapV1PublicArtifactRecord | Mapping[str, Any],
    *,
    projection: Optional[ResearchMapV1ProjectionRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_map_v1_payload_errors(raw)
    if not isinstance(record, ResearchMapV1PublicArtifactRecord):
        record = ResearchMapV1PublicArtifactRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(_validate_public_policy(record, record.artifact_ref, "Research Map v1 public artifact"))
    if not record.artifact_ref.startswith("research_map_v1_artifact:"):
        errors.append("artifact_ref must be research_map_v1_artifact:-prefixed")
    if not record.projection_id.startswith("research_map_v1_projection:"):
        errors.append("projection_id must be research_map_v1_projection:-prefixed")
    if record.artifact_kind not in {"api_contract_json", "openapi_stub", "static_json"}:
        errors.append("artifact_kind must be api_contract_json, openapi_stub, or static_json")
    if not record.local_uri.startswith("local://"):
        errors.append("Research Map v1 public artifact URI must remain local://")
    if not record.artifact_hash.startswith("sha256:"):
        errors.append("artifact_hash must be sha256:-prefixed")
    if record.api_version != RESEARCH_MAP_V1_API_VERSION:
        errors.append("api_version must match Research Map v1 contract version")
    if record.endpoint_path != RESEARCH_MAP_V1_NOVELTY_ENDPOINT:
        errors.append("endpoint_path must match the v1 novelty quote endpoint")
    if record.state not in {state.value for state in ResearchMapV1ArtifactState}:
        errors.append(f"unknown Research Map v1 artifact state: {record.state}")
    if record.state == ResearchMapV1ArtifactState.READY_AFTER_PRODUCTION_PUBLISH.value:
        errors.append("ready_after_production_publish artifact state is disabled in P2.5 local contracts")
    if not record.local_only:
        errors.append("Research Map v1 public artifact must remain local_only")
    if record.public_api_server_started:
        errors.append("public_api_server_started must remain false")
    if record.live_api_enabled:
        errors.append("live_api_enabled must remain false")
    if record.production_publish_enabled:
        errors.append("production_publish_enabled must remain false")
    if record.production_traffic_enabled:
        errors.append("production_traffic_enabled must remain false")
    if record.supabase_read_enabled:
        errors.append("supabase_read_enabled must remain false")
    if record.supabase_write_enabled:
        errors.append("supabase_write_enabled must remain false")
    if not record.protected_material_flags_checked:
        errors.append("Research Map v1 public artifact must check protected-material flags")
    protected_flags = _protected_policy_flags(record)
    if protected_flags:
        errors.append("Research Map v1 public artifact must not contain protected material flags: " + ", ".join(protected_flags))
    if projection is not None:
        if not isinstance(projection, ResearchMapV1ProjectionRecord):
            projection = ResearchMapV1ProjectionRecord.from_mapping(projection)
        projection_errors = validate_research_map_v1_projection_record(projection)
        if projection_errors:
            errors.append("source Research Map v1 projection is invalid: " + "; ".join(projection_errors))
        if record.projection_id != projection.projection_id:
            errors.append("Research Map v1 public artifact projection_id mismatch")
        expected_hash = sha256_json(projection.to_dict())
        if record.artifact_hash != expected_hash:
            errors.append("Research Map v1 public artifact_hash does not match projection")
    return errors


def verify_research_lab_research_map_v1_api(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    v0_summary = verify_research_lab_research_map()
    allocator_summary = verify_research_lab_meta_allocator()
    fixture = _load_fixture(Path(fixture_path))
    _assert(
        fixture["projection_request"]["source_v0_projection_ref"] == v0_summary["projection_id"],
        "Research Map v1 fixture source_v0_projection_ref matches live v0 verifier output",
    )
    _assert(
        fixture["projection_request"]["source_meta_allocator_selection_ref"] == allocator_summary["selection_id"],
        "Research Map v1 fixture source_meta_allocator_selection_ref matches live meta-allocator verifier output",
    )

    projection = build_research_map_v1_projection(
        generated_for_date=fixture["projection_request"]["generated_for_date"],
        source_v0_projection_ref=fixture["projection_request"]["source_v0_projection_ref"],
        source_meta_allocator_selection_ref=fixture["projection_request"]["source_meta_allocator_selection_ref"],
        novelty_index_ref=fixture["projection_request"]["novelty_index_ref"],
        public_cell_refs=fixture["projection_request"]["public_cell_refs"],
        allocator_prediction_refs=fixture["projection_request"]["allocator_prediction_refs"],
        component_registry_refs=fixture["projection_request"]["component_registry_refs"],
    )
    _assert(projection.to_dict() == fixture["expected_projection"], "Research Map v1 projection is deterministic")
    _assert(not validate_research_map_v1_projection_record(projection), "Research Map v1 projection validates")
    for record in fixture["invalid_projections"]:
        errors = validate_research_map_v1_projection_record(record)
        _assert(errors, f"invalid Research Map v1 projection fails: {record['projection_id']}")
        _assert_expected_error(errors, record)
    unsafe_projection_errors = validate_research_map_v1_projection_record(
        projection,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_projection_errors, "unsafe Phase 2 guards block Research Map v1 projection")

    request = NoveltyQuoteRequestRecord.from_mapping(fixture["quote_request"])
    _assert(not validate_novelty_quote_request_record(request), "novelty quote request validates")
    for record in fixture["invalid_quote_requests"]:
        errors = validate_novelty_quote_request_record(record)
        _assert(errors, f"invalid novelty quote request fails: {record['request_id']}")
        _assert_expected_error(errors, record)

    response = build_novelty_quote_response(
        request=request,
        projection=projection,
        decision=fixture["quote_response_request"]["decision"],
        novelty_score=fixture["quote_response_request"]["novelty_score"],
        nearest_prior_receipt_refs=fixture["quote_response_request"]["nearest_prior_receipt_refs"],
        nearest_cell_refs=fixture["quote_response_request"]["nearest_cell_refs"],
        suggested_cell_refs=fixture["quote_response_request"]["suggested_cell_refs"],
        quote_explanation_sanitized=fixture["quote_response_request"]["quote_explanation_sanitized"],
    )
    _assert(response.to_dict() == fixture["expected_quote_response"], "novelty quote response is deterministic")
    _assert(
        not validate_novelty_quote_response_record(response, request=request, projection=projection),
        "novelty quote response validates",
    )
    for record in fixture["invalid_quote_responses"]:
        errors = validate_novelty_quote_response_record(record)
        _assert(errors, f"invalid novelty quote response fails: {record['quote_id']}")
        _assert_expected_error(errors, record)

    artifact = build_research_map_v1_public_artifact(
        projection,
        local_uri=fixture["artifact_request"]["local_uri"],
        artifact_kind=fixture["artifact_request"]["artifact_kind"],
    )
    _assert(artifact.to_dict() == fixture["expected_artifact"], "Research Map v1 artifact is deterministic")
    _assert(
        not validate_research_map_v1_public_artifact_record(artifact, projection=projection),
        "Research Map v1 public artifact validates",
    )
    for record in fixture["invalid_artifacts"]:
        errors = validate_research_map_v1_public_artifact_record(record, projection=projection)
        _assert(errors, f"invalid Research Map v1 artifact fails: {record['artifact_ref']}")
        _assert_expected_error(errors, record)

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "v0_projection": v0_summary["projection_id"],
        "meta_allocator_selection": allocator_summary["selection_id"],
        "projection_id": projection.projection_id,
        "quote_id": response.quote_id,
        "artifact_ref": artifact.artifact_ref,
        "public_cells": len(projection.public_cell_refs),
        "endpoint_path": projection.quote_endpoint_path,
    }


def _validate_public_policy(record: Any, artifact_ref: str, reason: str) -> list[str]:
    errors: list[str] = []
    if not getattr(record, "protected_material_flags_checked", False):
        errors.append("protected-material flags must be checked before Research Map v1 public release")
    policy = ReleasePolicyRecord(
        artifact_ref=artifact_ref,
        artifact_type="map_projection",
        visibility_policy=getattr(record, "visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value),
        artifact_release_state=getattr(record, "artifact_release_state", ""),
        contains_live_champion_ip=bool(getattr(record, "contains_live_champion_ip", False)),
        contains_sealed_eval_details=bool(getattr(record, "contains_sealed_eval_details", False)),
        contains_raw_evidence_snapshot=bool(getattr(record, "contains_raw_evidence_snapshot", False)),
        contains_private_customer_data=bool(getattr(record, "contains_private_customer_data", False)),
        contains_judge_prompts=bool(getattr(record, "contains_judge_prompts", False)),
        release_condition="P2.5 local Research Map v1 API contract only",
        reason=reason,
    )
    errors.extend(validate_release_policy(policy))
    return errors


def _protected_policy_flags(record: Any) -> list[str]:
    return [
        flag
        for flag in (
            "contains_live_champion_ip",
            "contains_sealed_eval_details",
            "contains_raw_evidence_snapshot",
            "contains_private_customer_data",
            "contains_judge_prompts",
        )
        if bool(getattr(record, flag, False))
    ]


def _protected_map_v1_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_map_v1_material(record))
    if not found:
        return []
    return ["Research Map v1 payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_map_v1_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_MAP_V1_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_map_v1_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_map_v1_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_MAP_V1_MARKERS:
            if marker in lower:
                found.add(path or marker)
    return found


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if not expected:
        return
    expected_values = [str(item) for item in expected] if isinstance(expected, list) else [str(expected)]
    for expected_value in expected_values:
        _assert(any(expected_value in error for error in errors), f"expected error {expected_value!r}")


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
