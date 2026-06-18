"""Phase 2 meta-allocator v1 local contracts.

P2.4 defines local yield priors, deterministic Thompson-style selection
records, allocator prediction stubs, and pricing quotes. It does not route live
allocation, publish live predictions, mutate balances, or price production
presets.
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
from .research_map import (
    AllocatorPredictionRecord,
    validate_allocator_prediction_record,
    verify_research_lab_research_map,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "meta_allocator_fixtures.json"

META_ALLOCATOR_MODEL_VERSION = "meta_allocator:v1:local"
THOMPSON_STYLE_SELECTION_FORMULA = (
    "sampled_value_per_1000_cents = "
    "(posterior_mean_delta + seeded_uncertainty_draw * posterior_std_delta) "
    "* 1000 / expected_cost_cents"
)

PROTECTED_META_ALLOCATOR_KEYS = set(PROTECTED_COMPONENT_KEYS) | {
    "allocation_secret",
    "customer_outcome",
    "live_allocator_state",
    "live_preset_price",
    "private_preset_formula",
    "raw_outcome_label",
    "raw_results_ledger",
    "sealed_allocator_state",
}

PROTECTED_META_ALLOCATOR_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_COMPONENT_MARKERS)
        | {
            "allocation secret",
            "customer outcome",
            "live allocator state",
            "live preset price",
            "private preset formula",
            "raw outcome label",
            "raw results ledger",
            "sealed allocator state",
        }
    )
)


class CellYieldPriorState(str, Enum):
    LOCAL_FIXTURE = "local_fixture"
    RESULTS_LEDGER_MEASURED = "results_ledger_measured"
    BLOCKED = "blocked"


class MetaAllocatorSelectionState(str, Enum):
    LOCAL_RECOMMENDATION = "local_recommendation"
    READY_AFTER_RESULTS_LEDGER = "ready_after_results_ledger"
    BLOCKED = "blocked"


class AllocatorPredictionPublicationState(str, Enum):
    LOCAL_STUB = "local_stub"
    PUBLISHED_STUB = "published_stub"
    BLOCKED = "blocked"


class AllocationPricingState(str, Enum):
    LOCAL_QUOTE = "local_quote"
    READY_AFTER_OWNER_APPROVAL = "ready_after_owner_approval"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CellYieldPriorRecord:
    prior_id: str
    cell_ref: str
    island: str
    target_component: str
    patch_type: str
    observed_attempts: int
    kept_patches: int
    total_spend_cents: int
    verified_delta_sum: float
    posterior_mean_delta: float
    posterior_std_delta: float
    expected_cost_cents: int
    source_results_ledger_refs: tuple[str, ...]
    data_state: str = CellYieldPriorState.LOCAL_FIXTURE.value
    uses_local_fixtures: bool = True
    results_ledger_input_ready: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CellYieldPriorRecord":
        return cls(
            prior_id=str(data["prior_id"]),
            cell_ref=str(data["cell_ref"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            patch_type=str(data["patch_type"]),
            observed_attempts=int(data["observed_attempts"]),
            kept_patches=int(data["kept_patches"]),
            total_spend_cents=int(data["total_spend_cents"]),
            verified_delta_sum=float(data["verified_delta_sum"]),
            posterior_mean_delta=float(data["posterior_mean_delta"]),
            posterior_std_delta=float(data["posterior_std_delta"]),
            expected_cost_cents=int(data["expected_cost_cents"]),
            source_results_ledger_refs=tuple(str(item) for item in data.get("source_results_ledger_refs", [])),
            data_state=str(data.get("data_state", CellYieldPriorState.LOCAL_FIXTURE.value)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            results_ledger_input_ready=bool(data.get("results_ledger_input_ready", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_results_ledger_refs"] = list(self.source_results_ledger_refs)
        return data


@dataclass(frozen=True)
class MetaAllocatorCellScore:
    cell_ref: str
    prior_id: str
    posterior_mean_delta: float
    posterior_std_delta: float
    expected_cost_cents: int
    seeded_uncertainty_draw: float
    sampled_delta: float
    sampled_value_per_1000_cents: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetaAllocatorCellScore":
        return cls(
            cell_ref=str(data["cell_ref"]),
            prior_id=str(data["prior_id"]),
            posterior_mean_delta=float(data["posterior_mean_delta"]),
            posterior_std_delta=float(data["posterior_std_delta"]),
            expected_cost_cents=int(data["expected_cost_cents"]),
            seeded_uncertainty_draw=float(data["seeded_uncertainty_draw"]),
            sampled_delta=float(data["sampled_delta"]),
            sampled_value_per_1000_cents=float(data["sampled_value_per_1000_cents"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaAllocatorSelectionRecord:
    selection_id: str
    budget_cents: int
    selection_count: int
    seed: int
    formula: str
    prior_refs: tuple[str, ...]
    ranked_cell_refs: tuple[str, ...]
    selected_cell_refs: tuple[str, ...]
    sample_scores: tuple[MetaAllocatorCellScore, ...]
    state: str = MetaAllocatorSelectionState.LOCAL_RECOMMENDATION.value
    uses_local_fixtures: bool = True
    uses_results_ledger_inputs: bool = False
    live_allocation_enabled: bool = False
    allocation_spend_effect_enabled: bool = False
    balance_mutation_enabled: bool = False
    production_allocation_changed: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetaAllocatorSelectionRecord":
        return cls(
            selection_id=str(data["selection_id"]),
            budget_cents=int(data["budget_cents"]),
            selection_count=int(data["selection_count"]),
            seed=int(data["seed"]),
            formula=str(data.get("formula", THOMPSON_STYLE_SELECTION_FORMULA)),
            prior_refs=tuple(str(item) for item in data.get("prior_refs", [])),
            ranked_cell_refs=tuple(str(item) for item in data.get("ranked_cell_refs", [])),
            selected_cell_refs=tuple(str(item) for item in data.get("selected_cell_refs", [])),
            sample_scores=tuple(MetaAllocatorCellScore.from_mapping(item) for item in data.get("sample_scores", [])),
            state=str(data.get("state", MetaAllocatorSelectionState.LOCAL_RECOMMENDATION.value)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            uses_results_ledger_inputs=bool(data.get("uses_results_ledger_inputs", False)),
            live_allocation_enabled=bool(data.get("live_allocation_enabled", False)),
            allocation_spend_effect_enabled=bool(data.get("allocation_spend_effect_enabled", False)),
            balance_mutation_enabled=bool(data.get("balance_mutation_enabled", False)),
            production_allocation_changed=bool(data.get("production_allocation_changed", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prior_refs"] = list(self.prior_refs)
        data["ranked_cell_refs"] = list(self.ranked_cell_refs)
        data["selected_cell_refs"] = list(self.selected_cell_refs)
        data["sample_scores"] = [score.to_dict() for score in self.sample_scores]
        return data


@dataclass(frozen=True)
class AllocatorPredictionStubRecord:
    prediction_id: str
    cell_ref: str
    prior_id: str
    island: str
    target_component: str
    patch_type: str
    predicted_delta: float
    lower_delta: float
    upper_delta: float
    confidence: float
    expected_cost_cents: int
    expected_value_score: float
    provenance_refs: tuple[str, ...]
    model_version_ref: str = META_ALLOCATOR_MODEL_VERSION
    publication_state: str = AllocatorPredictionPublicationState.LOCAL_STUB.value
    uses_local_fixtures: bool = True
    uses_results_ledger_inputs: bool = False
    published_to_live_map: bool = False
    live_allocator_prediction: bool = False
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
    def from_mapping(cls, data: Mapping[str, Any]) -> "AllocatorPredictionStubRecord":
        return cls(
            prediction_id=str(data["prediction_id"]),
            cell_ref=str(data["cell_ref"]),
            prior_id=str(data["prior_id"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            patch_type=str(data["patch_type"]),
            predicted_delta=float(data["predicted_delta"]),
            lower_delta=float(data["lower_delta"]),
            upper_delta=float(data["upper_delta"]),
            confidence=float(data["confidence"]),
            expected_cost_cents=int(data["expected_cost_cents"]),
            expected_value_score=float(data["expected_value_score"]),
            provenance_refs=tuple(str(item) for item in data.get("provenance_refs", [])),
            model_version_ref=str(data.get("model_version_ref", META_ALLOCATOR_MODEL_VERSION)),
            publication_state=str(data.get("publication_state", AllocatorPredictionPublicationState.LOCAL_STUB.value)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            uses_results_ledger_inputs=bool(data.get("uses_results_ledger_inputs", False)),
            published_to_live_map=bool(data.get("published_to_live_map", False)),
            live_allocator_prediction=bool(data.get("live_allocator_prediction", False)),
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
        data["provenance_refs"] = list(self.provenance_refs)
        return data


@dataclass(frozen=True)
class AllocationPricingRecord:
    pricing_id: str
    selection_id: str
    preset_ref: str
    pool_ref: str
    pool_spend_cents: int
    quoted_ticket_count: int
    price_per_ticket_cents: int
    quote_currency: str = "usd_cents"
    quote_source: str = "local_fixture"
    owner_approval_ref: str = ""
    state: str = AllocationPricingState.LOCAL_QUOTE.value
    live_price_enabled: bool = False
    allocation_spend_effect_enabled: bool = False
    balance_mutation_enabled: bool = False
    funds_moved: bool = False
    production_pricing_changed: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AllocationPricingRecord":
        return cls(
            pricing_id=str(data["pricing_id"]),
            selection_id=str(data["selection_id"]),
            preset_ref=str(data["preset_ref"]),
            pool_ref=str(data["pool_ref"]),
            pool_spend_cents=int(data["pool_spend_cents"]),
            quoted_ticket_count=int(data["quoted_ticket_count"]),
            price_per_ticket_cents=int(data["price_per_ticket_cents"]),
            quote_currency=str(data.get("quote_currency", "usd_cents")),
            quote_source=str(data.get("quote_source", "local_fixture")),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            state=str(data.get("state", AllocationPricingState.LOCAL_QUOTE.value)),
            live_price_enabled=bool(data.get("live_price_enabled", False)),
            allocation_spend_effect_enabled=bool(data.get("allocation_spend_effect_enabled", False)),
            balance_mutation_enabled=bool(data.get("balance_mutation_enabled", False)),
            funds_moved=bool(data.get("funds_moved", False)),
            production_pricing_changed=bool(data.get("production_pricing_changed", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seeded_uncertainty_draw(seed: int, cell_ref: str) -> float:
    """Return a deterministic pseudo-random draw in [-1, 1] for local fixtures."""

    digest = sha256_json({"seed": int(seed), "cell_ref": cell_ref}).split(":", 1)[1]
    unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return round((unit * 2.0) - 1.0, 6)


def score_cell_yield_prior(prior: CellYieldPriorRecord | Mapping[str, Any], *, seed: int) -> MetaAllocatorCellScore:
    if not isinstance(prior, CellYieldPriorRecord):
        prior = CellYieldPriorRecord.from_mapping(prior)
    errors = validate_cell_yield_prior_record(prior)
    if errors:
        raise ValueError("; ".join(errors))
    draw = seeded_uncertainty_draw(seed, prior.cell_ref)
    sampled_delta = round(prior.posterior_mean_delta + draw * prior.posterior_std_delta, 6)
    sampled_value = round(sampled_delta * 1000.0 / prior.expected_cost_cents, 6)
    return MetaAllocatorCellScore(
        cell_ref=prior.cell_ref,
        prior_id=prior.prior_id,
        posterior_mean_delta=prior.posterior_mean_delta,
        posterior_std_delta=prior.posterior_std_delta,
        expected_cost_cents=prior.expected_cost_cents,
        seeded_uncertainty_draw=draw,
        sampled_delta=sampled_delta,
        sampled_value_per_1000_cents=sampled_value,
    )


def build_meta_allocator_selection_record(
    *,
    selection_id: str,
    priors: Sequence[CellYieldPriorRecord | Mapping[str, Any]],
    budget_cents: int,
    selection_count: int,
    seed: int,
) -> MetaAllocatorSelectionRecord:
    normalized = _validate_priors_for_builder(priors)
    sample_scores = tuple(score_cell_yield_prior(prior, seed=seed) for prior in normalized)
    ranked_scores = tuple(
        sorted(
            sample_scores,
            key=lambda score: (-score.sampled_value_per_1000_cents, score.cell_ref),
        )
    )
    record = MetaAllocatorSelectionRecord(
        selection_id=selection_id,
        budget_cents=budget_cents,
        selection_count=selection_count,
        seed=seed,
        formula=THOMPSON_STYLE_SELECTION_FORMULA,
        prior_refs=tuple(prior.prior_id for prior in normalized),
        ranked_cell_refs=tuple(score.cell_ref for score in ranked_scores),
        selected_cell_refs=tuple(score.cell_ref for score in ranked_scores[:selection_count]),
        sample_scores=ranked_scores,
    )
    errors = validate_meta_allocator_selection_record(record, priors=normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def build_allocator_prediction_stub(
    *,
    prior: CellYieldPriorRecord | Mapping[str, Any],
    selection: MetaAllocatorSelectionRecord | Mapping[str, Any],
    confidence: float = 0.8,
) -> AllocatorPredictionStubRecord:
    if not isinstance(prior, CellYieldPriorRecord):
        prior = CellYieldPriorRecord.from_mapping(prior)
    prior_errors = validate_cell_yield_prior_record(prior)
    if prior_errors:
        raise ValueError("; ".join(prior_errors))
    if not isinstance(selection, MetaAllocatorSelectionRecord):
        selection = MetaAllocatorSelectionRecord.from_mapping(selection)
    lower_delta = round(prior.posterior_mean_delta - 1.64 * prior.posterior_std_delta, 6)
    upper_delta = round(prior.posterior_mean_delta + 1.64 * prior.posterior_std_delta, 6)
    expected_value = round(prior.posterior_mean_delta * 1000.0 / prior.expected_cost_cents, 6)
    payload = {
        "cell_ref": prior.cell_ref,
        "prior_id": prior.prior_id,
        "selection_id": selection.selection_id,
        "model_version_ref": META_ALLOCATOR_MODEL_VERSION,
    }
    record = AllocatorPredictionStubRecord(
        prediction_id="allocator_prediction_stub:" + sha256_json(payload).split(":", 1)[1][:16],
        cell_ref=prior.cell_ref,
        prior_id=prior.prior_id,
        island=prior.island,
        target_component=prior.target_component,
        patch_type=prior.patch_type,
        predicted_delta=prior.posterior_mean_delta,
        lower_delta=lower_delta,
        upper_delta=upper_delta,
        confidence=confidence,
        expected_cost_cents=prior.expected_cost_cents,
        expected_value_score=expected_value,
        provenance_refs=(prior.prior_id, selection.selection_id, *prior.source_results_ledger_refs),
    )
    errors = validate_allocator_prediction_stub_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def prediction_stub_to_research_map_record(
    record: AllocatorPredictionStubRecord | Mapping[str, Any],
) -> AllocatorPredictionRecord:
    if not isinstance(record, AllocatorPredictionStubRecord):
        record = AllocatorPredictionStubRecord.from_mapping(record)
    return AllocatorPredictionRecord(
        prediction_id=record.prediction_id.replace("allocator_prediction_stub:", "allocator_prediction:"),
        cell_ref=record.cell_ref,
        island=record.island,
        target_component=record.target_component,
        patch_type=record.patch_type,
        predicted_delta=record.predicted_delta,
        confidence=record.confidence,
        expected_cost_cents=record.expected_cost_cents,
        expected_value_score=record.expected_value_score,
        provenance_refs=record.provenance_refs,
        model_version_ref=record.model_version_ref,
        protected_material_flags_checked=record.protected_material_flags_checked,
        contains_live_champion_ip=record.contains_live_champion_ip,
        contains_sealed_eval_details=record.contains_sealed_eval_details,
        contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
        contains_private_customer_data=record.contains_private_customer_data,
        contains_judge_prompts=record.contains_judge_prompts,
        visibility_policy=record.visibility_policy,
        artifact_release_state=record.artifact_release_state,
    )


def build_allocation_pricing_record(
    *,
    selection: MetaAllocatorSelectionRecord | Mapping[str, Any],
    preset_ref: str,
    pool_ref: str,
    quoted_ticket_count: int,
    price_per_ticket_cents: int,
) -> AllocationPricingRecord:
    if not isinstance(selection, MetaAllocatorSelectionRecord):
        selection = MetaAllocatorSelectionRecord.from_mapping(selection)
    record = AllocationPricingRecord(
        pricing_id="allocator_pricing:" + sha256_json(
            {
                "selection_id": selection.selection_id,
                "preset_ref": preset_ref,
                "pool_ref": pool_ref,
                "quoted_ticket_count": quoted_ticket_count,
                "price_per_ticket_cents": price_per_ticket_cents,
            }
        ).split(":", 1)[1][:16],
        selection_id=selection.selection_id,
        preset_ref=preset_ref,
        pool_ref=pool_ref,
        pool_spend_cents=quoted_ticket_count * price_per_ticket_cents,
        quoted_ticket_count=quoted_ticket_count,
        price_per_ticket_cents=price_per_ticket_cents,
    )
    errors = validate_allocation_pricing_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_cell_yield_prior_record(record: CellYieldPriorRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_meta_allocator_payload_errors(raw)
    if not isinstance(record, CellYieldPriorRecord):
        record = CellYieldPriorRecord.from_mapping(record)
    errors = list(raw_errors)
    if not record.prior_id.startswith("cell_yield_prior:"):
        errors.append("cell yield prior requires cell_yield_prior:-prefixed prior_id")
    if not record.cell_ref.startswith("map_cell:"):
        errors.append("cell_ref must be map_cell:-prefixed")
    if not record.island:
        errors.append("cell yield prior requires island")
    if not record.target_component:
        errors.append("cell yield prior requires target_component")
    if record.patch_type not in {patch.value for patch in PatchType}:
        errors.append(f"unknown patch_type: {record.patch_type}")
    if record.observed_attempts < 0:
        errors.append("observed_attempts must be non-negative")
    if record.kept_patches < 0:
        errors.append("kept_patches must be non-negative")
    if record.kept_patches > record.observed_attempts:
        errors.append("kept_patches cannot exceed observed_attempts")
    if record.total_spend_cents < 0:
        errors.append("total_spend_cents must be non-negative")
    if record.posterior_std_delta < 0:
        errors.append("posterior_std_delta must be non-negative")
    if record.expected_cost_cents <= 0:
        errors.append("expected_cost_cents must be positive")
    if not record.source_results_ledger_refs:
        errors.append("cell yield prior must include source_results_ledger_refs")
    if record.data_state not in {state.value for state in CellYieldPriorState}:
        errors.append(f"unknown cell yield prior state: {record.data_state}")
    if record.data_state == CellYieldPriorState.LOCAL_FIXTURE.value:
        if not record.uses_local_fixtures:
            errors.append("local fixture priors must be marked uses_local_fixtures")
        if record.results_ledger_input_ready:
            errors.append("local fixture priors cannot claim results_ledger_input_ready")
    if record.data_state == CellYieldPriorState.RESULTS_LEDGER_MEASURED.value:
        if record.uses_local_fixtures:
            errors.append("results-ledger priors must not use local fixtures")
        if not record.results_ledger_input_ready:
            errors.append("results-ledger priors must mark results_ledger_input_ready")
        if not any(ref.startswith("results_ledger:") for ref in record.source_results_ledger_refs):
            errors.append("results-ledger priors require a results_ledger: source ref")
    if not record.local_only:
        errors.append("cell yield prior records must remain local_only")
    return errors


def validate_meta_allocator_selection_record(
    record: MetaAllocatorSelectionRecord | Mapping[str, Any],
    *,
    priors: Sequence[CellYieldPriorRecord | Mapping[str, Any]] = (),
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_meta_allocator_payload_errors(raw)
    if not isinstance(record, MetaAllocatorSelectionRecord):
        record = MetaAllocatorSelectionRecord.from_mapping(record)
    errors = list(raw_errors)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if not record.selection_id.startswith("meta_allocator_selection:"):
        errors.append("selection_id must be meta_allocator_selection:-prefixed")
    if record.budget_cents <= 0:
        errors.append("budget_cents must be positive")
    if record.selection_count <= 0:
        errors.append("selection_count must be positive")
    if record.formula != THOMPSON_STYLE_SELECTION_FORMULA:
        errors.append("selection formula must match P2.4 Thompson-style formula")
    if not record.prior_refs:
        errors.append("selection must include prior_refs")
    if not record.sample_scores:
        errors.append("selection must include sample_scores")
    if record.selection_count > len(record.sample_scores):
        errors.append("selection_count cannot exceed sample_scores")
    if record.state not in {state.value for state in MetaAllocatorSelectionState}:
        errors.append(f"unknown meta-allocator selection state: {record.state}")
    if record.live_allocation_enabled:
        errors.append("live_allocation_enabled must remain false")
    if record.allocation_spend_effect_enabled:
        errors.append("allocation_spend_effect_enabled must remain false")
    if record.balance_mutation_enabled:
        errors.append("balance_mutation_enabled must remain false")
    if record.production_allocation_changed:
        errors.append("production_allocation_changed must remain false")
    if not record.local_only:
        errors.append("meta-allocator selection must remain local_only")
    if record.uses_results_ledger_inputs and record.uses_local_fixtures:
        errors.append("selection cannot use local fixtures and results-ledger inputs simultaneously")
    if (
        record.state == MetaAllocatorSelectionState.READY_AFTER_RESULTS_LEDGER.value
        and record.uses_local_fixtures
    ):
        errors.append("ready_after_results_ledger selection cannot use local fixtures")
    ranked_from_scores = tuple(
        score.cell_ref
        for score in sorted(record.sample_scores, key=lambda score: (-score.sampled_value_per_1000_cents, score.cell_ref))
    )
    if ranked_from_scores and record.ranked_cell_refs != ranked_from_scores:
        errors.append("ranked_cell_refs must match sampled value ranking")
    if record.ranked_cell_refs and record.selected_cell_refs != record.ranked_cell_refs[: record.selection_count]:
        errors.append("selected_cell_refs must equal top ranked_cell_refs by selection_count")
    score_prior_refs = {score.prior_id for score in record.sample_scores}
    if record.prior_refs and score_prior_refs != set(record.prior_refs):
        errors.append("sample_scores must cover every prior_ref exactly once")
    prior_map = {prior.prior_id: prior for prior in _normalize_priors(priors)}
    if prior_map:
        if set(record.prior_refs) != set(prior_map):
            errors.append("selection prior_refs must match supplied priors")
        for score in record.sample_scores:
            prior = prior_map.get(score.prior_id)
            if prior is None:
                errors.append(f"sample score prior_id not supplied: {score.prior_id}")
                continue
            expected = score_cell_yield_prior(prior, seed=record.seed)
            if score.to_dict() != expected.to_dict():
                errors.append(f"sample score does not match deterministic prior math: {score.cell_ref}")
    if record.uses_results_ledger_inputs:
        if not prior_map:
            errors.append("results-ledger selection requires supplied priors for verification")
        not_ready = [
            prior.prior_id
            for prior in prior_map.values()
            if not prior.results_ledger_input_ready or prior.uses_local_fixtures
        ]
        if not_ready:
            errors.append("results-ledger selection requires real measured priors: " + ", ".join(not_ready))
    if record.state == MetaAllocatorSelectionState.READY_AFTER_RESULTS_LEDGER.value and not record.uses_results_ledger_inputs:
        errors.append("ready_after_results_ledger state requires uses_results_ledger_inputs")
    return errors


def validate_allocator_prediction_stub_record(record: AllocatorPredictionStubRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_meta_allocator_payload_errors(raw)
    if not isinstance(record, AllocatorPredictionStubRecord):
        record = AllocatorPredictionStubRecord.from_mapping(record)
    errors = list(raw_errors)
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.prediction_id,
                artifact_type="map_projection",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                contains_live_champion_ip=record.contains_live_champion_ip,
                contains_sealed_eval_details=record.contains_sealed_eval_details,
                contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
                contains_private_customer_data=record.contains_private_customer_data,
                contains_judge_prompts=record.contains_judge_prompts,
                reason="meta-allocator prediction stubs expose sanitized local estimates only",
            )
        )
    )
    if not record.prediction_id.startswith("allocator_prediction_stub:"):
        errors.append("prediction_id must be allocator_prediction_stub:-prefixed")
    if not record.cell_ref.startswith("map_cell:"):
        errors.append("cell_ref must be map_cell:-prefixed")
    if not record.prior_id.startswith("cell_yield_prior:"):
        errors.append("prior_id must be cell_yield_prior:-prefixed")
    if record.patch_type not in {patch.value for patch in PatchType}:
        errors.append(f"unknown patch_type: {record.patch_type}")
    if record.lower_delta > record.predicted_delta or record.predicted_delta > record.upper_delta:
        errors.append("prediction interval must satisfy lower_delta <= predicted_delta <= upper_delta")
    if not 0 <= record.confidence <= 1:
        errors.append("prediction confidence must be in [0, 1]")
    if record.expected_cost_cents <= 0:
        errors.append("expected_cost_cents must be positive")
    if not record.provenance_refs:
        errors.append("prediction stub must include provenance_refs")
    if record.publication_state not in {state.value for state in AllocatorPredictionPublicationState}:
        errors.append(f"unknown allocator prediction publication state: {record.publication_state}")
    if record.uses_local_fixtures and record.uses_results_ledger_inputs:
        errors.append("prediction stub cannot use local fixtures and results-ledger inputs simultaneously")
    if record.published_to_live_map:
        errors.append("published_to_live_map must remain false")
    if record.live_allocator_prediction:
        errors.append("live_allocator_prediction must remain false")
    if record.uses_local_fixtures and (record.published_to_live_map or record.live_allocator_prediction):
        errors.append("local fixture prediction stubs cannot be published as live allocator predictions")
    if record.publication_state == AllocatorPredictionPublicationState.PUBLISHED_STUB.value:
        if record.uses_local_fixtures or not record.uses_results_ledger_inputs:
            errors.append("published_stub state requires real results-ledger inputs, not local fixtures")
    if not record.protected_material_flags_checked:
        errors.append("prediction stub must check protected-material flags")
    protected_flags = _protected_policy_flags(record)
    if protected_flags:
        errors.append("prediction stub must not contain protected material flags: " + ", ".join(protected_flags))
    if not record.local_only:
        errors.append("prediction stub must remain local_only")
    return errors


def validate_allocation_pricing_record(record: AllocationPricingRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_meta_allocator_payload_errors(raw)
    if not isinstance(record, AllocationPricingRecord):
        record = AllocationPricingRecord.from_mapping(record)
    errors = list(raw_errors)
    if not record.pricing_id.startswith("allocator_pricing:"):
        errors.append("pricing_id must be allocator_pricing:-prefixed")
    if not record.selection_id.startswith("meta_allocator_selection:"):
        errors.append("selection_id must be meta_allocator_selection:-prefixed")
    if not record.preset_ref.startswith("preset:"):
        errors.append("preset_ref must be preset:-prefixed")
    if not record.pool_ref.startswith("allocation_pool:"):
        errors.append("pool_ref must be allocation_pool:-prefixed")
    if record.pool_spend_cents <= 0:
        errors.append("pool_spend_cents must be positive")
    if record.quoted_ticket_count <= 0:
        errors.append("quoted_ticket_count must be positive")
    if record.price_per_ticket_cents <= 0:
        errors.append("price_per_ticket_cents must be positive")
    if record.pool_spend_cents != record.quoted_ticket_count * record.price_per_ticket_cents:
        errors.append("pool_spend_cents must equal quoted_ticket_count * price_per_ticket_cents")
    if record.quote_currency != "usd_cents":
        errors.append("quote_currency must be usd_cents")
    if record.state not in {state.value for state in AllocationPricingState}:
        errors.append(f"unknown allocation pricing state: {record.state}")
    if record.state == AllocationPricingState.READY_AFTER_OWNER_APPROVAL.value and not record.owner_approval_ref:
        errors.append("ready_after_owner_approval pricing requires owner_approval_ref")
    if record.live_price_enabled:
        errors.append("live_price_enabled must remain false")
    if record.allocation_spend_effect_enabled:
        errors.append("allocation_spend_effect_enabled must remain false")
    if record.balance_mutation_enabled:
        errors.append("balance_mutation_enabled must remain false")
    if record.funds_moved:
        errors.append("funds_moved must remain false")
    if record.production_pricing_changed:
        errors.append("production_pricing_changed must remain false")
    if not record.local_only:
        errors.append("allocation pricing records must remain local_only")
    return errors


def verify_research_lab_meta_allocator(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    map_summary = verify_research_lab_research_map()
    fixture = _load_fixture(Path(fixture_path))

    priors = tuple(CellYieldPriorRecord.from_mapping(item) for item in fixture["cell_yield_priors"])
    for prior in priors:
        _assert(not validate_cell_yield_prior_record(prior), f"cell yield prior validates: {prior.prior_id}")
    for record in fixture["invalid_cell_yield_priors"]:
        errors = validate_cell_yield_prior_record(record)
        _assert(errors, f"invalid cell yield prior fails: {record['prior_id']}")
        _assert_expected_error(errors, record)
    invalid_builder_prior = fixture["invalid_builder_prior"]
    try:
        build_meta_allocator_selection_record(
            selection_id="meta_allocator_selection:bad-builder-prior",
            priors=[invalid_builder_prior],
            budget_cents=fixture["selection_request"]["budget_cents"],
            selection_count=1,
            seed=fixture["selection_request"]["seed"],
        )
    except ValueError as exc:
        _assert(
            str(invalid_builder_prior["expected_error_contains"]) in str(exc),
            "selection builder validates priors before scoring",
        )
    else:
        raise AssertionError("selection builder rejects invalid prior before scoring")

    selection = build_meta_allocator_selection_record(
        selection_id=fixture["selection_request"]["selection_id"],
        priors=priors,
        budget_cents=fixture["selection_request"]["budget_cents"],
        selection_count=fixture["selection_request"]["selection_count"],
        seed=fixture["selection_request"]["seed"],
    )
    _assert(selection.ranked_cell_refs == tuple(fixture["expected_ranked_cell_refs"]), "ranked cells are deterministic")
    _assert(selection.selected_cell_refs == tuple(fixture["expected_selected_cell_refs"]), "selected cells are deterministic")
    _assert(
        [score.to_dict() for score in selection.sample_scores] == fixture["expected_sample_scores"],
        "sample scores are deterministic",
    )
    _assert(not validate_meta_allocator_selection_record(selection, priors=priors), "meta-allocator selection validates")
    for record in fixture["invalid_selection_records"]:
        errors = validate_meta_allocator_selection_record(record, priors=priors)
        _assert(errors, f"invalid selection record fails: {record['selection_id']}")
        _assert_expected_error(errors, record)
    for record in fixture["invalid_selection_records_without_priors"]:
        errors = validate_meta_allocator_selection_record(record)
        _assert(errors, f"invalid selection without supplied priors fails: {record['selection_id']}")
        _assert_expected_error(errors, record)
    unsafe_errors = validate_meta_allocator_selection_record(
        selection,
        priors=priors,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 2 guards block meta-allocation")

    first_prior = _prior_by_cell(priors)[selection.selected_cell_refs[0]]
    prediction = build_allocator_prediction_stub(prior=first_prior, selection=selection)
    _assert(prediction.to_dict() == fixture["expected_prediction_stub"], "prediction stub is deterministic")
    _assert(not validate_allocator_prediction_stub_record(prediction), "allocator prediction stub validates")
    map_prediction = prediction_stub_to_research_map_record(prediction)
    _assert(not validate_allocator_prediction_record(map_prediction), "prediction stub projects to research-map prediction")
    for record in fixture["invalid_prediction_stubs"]:
        errors = validate_allocator_prediction_stub_record(record)
        _assert(errors, f"invalid prediction stub fails: {record['prediction_id']}")
        _assert_expected_error(errors, record)
    try:
        build_allocator_prediction_stub(prior=invalid_builder_prior, selection=selection)
    except ValueError as exc:
        _assert(
            str(invalid_builder_prior["expected_error_contains"]) in str(exc),
            "prediction builder validates prior before expected-value math",
        )
    else:
        raise AssertionError("prediction builder rejects invalid prior before expected-value math")

    pricing = build_allocation_pricing_record(
        selection=selection,
        preset_ref=fixture["pricing_request"]["preset_ref"],
        pool_ref=fixture["pricing_request"]["pool_ref"],
        quoted_ticket_count=fixture["pricing_request"]["quoted_ticket_count"],
        price_per_ticket_cents=fixture["pricing_request"]["price_per_ticket_cents"],
    )
    _assert(pricing.to_dict() == fixture["expected_pricing_record"], "pricing quote is deterministic")
    _assert(not validate_allocation_pricing_record(pricing), "allocation pricing validates")
    for record in fixture["invalid_pricing_records"]:
        errors = validate_allocation_pricing_record(record)
        _assert(errors, f"invalid allocation pricing fails: {record['pricing_id']}")
        _assert_expected_error(errors, record)

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "research_map_projection": map_summary["projection_id"],
        "prior_count": len(priors),
        "selection_id": selection.selection_id,
        "selected_cell_refs": list(selection.selected_cell_refs),
        "prediction_id": prediction.prediction_id,
        "pricing_id": pricing.pricing_id,
        "pool_spend_cents": pricing.pool_spend_cents,
    }


def _normalize_priors(priors: Sequence[CellYieldPriorRecord | Mapping[str, Any]]) -> tuple[CellYieldPriorRecord, ...]:
    return tuple(prior if isinstance(prior, CellYieldPriorRecord) else CellYieldPriorRecord.from_mapping(prior) for prior in priors)


def _validate_priors_for_builder(
    priors: Sequence[CellYieldPriorRecord | Mapping[str, Any]],
) -> tuple[CellYieldPriorRecord, ...]:
    normalized = _normalize_priors(priors)
    errors: list[str] = []
    for prior in normalized:
        prior_errors = validate_cell_yield_prior_record(prior)
        if prior_errors:
            errors.append(f"{prior.prior_id}: " + "; ".join(prior_errors))
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def _prior_by_cell(priors: Sequence[CellYieldPriorRecord]) -> dict[str, CellYieldPriorRecord]:
    return {prior.cell_ref: prior for prior in priors}


def _protected_policy_flags(record: AllocatorPredictionStubRecord) -> list[str]:
    return [
        flag
        for flag in (
            "contains_live_champion_ip",
            "contains_sealed_eval_details",
            "contains_raw_evidence_snapshot",
            "contains_private_customer_data",
            "contains_judge_prompts",
        )
        if bool(getattr(record, flag))
    ]


def _protected_meta_allocator_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_meta_allocator_material(record))
    if not found:
        return []
    return ["meta-allocator payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_meta_allocator_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_META_ALLOCATOR_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_meta_allocator_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_meta_allocator_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_META_ALLOCATOR_MARKERS:
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
