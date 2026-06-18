"""Phase 2 island-selection formula contracts.

P2.1 defines the local formula for selecting Phase-2 vertical islands:
delivered-lead volume multiplied by outcome-label density. It is intentionally
inert: it does not create live island champion lines, route production jobs, or
claim the working healthcare/fintech/industrial pick is finalized before
production data exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .market_foundation import (
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    verify_market_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "island_selection_fixtures.json"

DEFAULT_WORKING_ISLANDS: tuple[str, ...] = ("healthcare", "fintech", "industrial")
MARKET_ISLAND_SELECTION_COUNT = 3
ISLAND_SELECTION_FORMULA = "delivered_lead_volume * outcome_label_density"
ISLAND_SELECTION_TIE_BREAKER = (
    "score desc, delivered_lead_volume desc, outcome_label_density desc, island asc"
)


class IslandCandidateDataState(str, Enum):
    WORKING_PLACEHOLDER = "working_placeholder"
    PRODUCTION_MEASURED = "production_measured"
    BLOCKED = "blocked"


class IslandSelectionState(str, Enum):
    WORKING_PLACEHOLDER = "working_placeholder"
    FORMULA_READY_LOCAL_REVIEW = "formula_ready_local_review"
    FINALIZED_AFTER_PRODUCTION_DATA = "finalized_after_production_data"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class IslandCandidateRecord:
    candidate_id: str
    island: str
    display_name: str
    delivered_lead_volume: int
    outcome_label_density: float
    measurement_window: str
    source_refs: tuple[str, ...]
    data_state: str = IslandCandidateDataState.WORKING_PLACEHOLDER.value
    production_data_ready: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "IslandCandidateRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            island=str(data["island"]),
            display_name=str(data.get("display_name", data["island"])),
            delivered_lead_volume=int(data["delivered_lead_volume"]),
            outcome_label_density=float(data["outcome_label_density"]),
            measurement_window=str(data["measurement_window"]),
            source_refs=tuple(str(item) for item in data.get("source_refs", [])),
            data_state=str(data.get("data_state", IslandCandidateDataState.WORKING_PLACEHOLDER.value)),
            production_data_ready=bool(data.get("production_data_ready", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = list(self.source_refs)
        return data

    @property
    def formula_score(self) -> float:
        return island_formula_score(self)


@dataclass(frozen=True)
class IslandSelectionRecord:
    selection_id: str
    candidate_refs: tuple[str, ...]
    selected_islands: tuple[str, ...]
    formula: str = ISLAND_SELECTION_FORMULA
    tie_breaker: str = ISLAND_SELECTION_TIE_BREAKER
    selection_count: int = MARKET_ISLAND_SELECTION_COUNT
    uses_production_data: bool = False
    final_selection: bool = False
    local_only: bool = True
    production_routing_enabled: bool = False
    island_champion_lines_created: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = IslandSelectionState.WORKING_PLACEHOLDER.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "IslandSelectionRecord":
        return cls(
            selection_id=str(data["selection_id"]),
            candidate_refs=tuple(str(item) for item in data.get("candidate_refs", [])),
            selected_islands=tuple(str(item) for item in data.get("selected_islands", [])),
            formula=str(data.get("formula", ISLAND_SELECTION_FORMULA)),
            tie_breaker=str(data.get("tie_breaker", ISLAND_SELECTION_TIE_BREAKER)),
            selection_count=int(data.get("selection_count", MARKET_ISLAND_SELECTION_COUNT)),
            uses_production_data=bool(data.get("uses_production_data", False)),
            final_selection=bool(data.get("final_selection", False)),
            local_only=bool(data.get("local_only", True)),
            production_routing_enabled=bool(data.get("production_routing_enabled", False)),
            island_champion_lines_created=bool(data.get("island_champion_lines_created", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", IslandSelectionState.WORKING_PLACEHOLDER.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_refs"] = list(self.candidate_refs)
        data["selected_islands"] = list(self.selected_islands)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def island_formula_score(candidate: IslandCandidateRecord | Mapping[str, Any]) -> float:
    if not isinstance(candidate, IslandCandidateRecord):
        candidate = IslandCandidateRecord.from_mapping(candidate)
    return round(candidate.delivered_lead_volume * candidate.outcome_label_density, 6)


def rank_island_candidates(
    candidates: Sequence[IslandCandidateRecord | Mapping[str, Any]],
) -> list[IslandCandidateRecord]:
    parsed = [
        candidate if isinstance(candidate, IslandCandidateRecord) else IslandCandidateRecord.from_mapping(candidate)
        for candidate in candidates
    ]
    return sorted(
        parsed,
        key=lambda candidate: (
            -candidate.formula_score,
            -candidate.delivered_lead_volume,
            -candidate.outcome_label_density,
            candidate.island,
        ),
    )


def select_market_islands(
    candidates: Sequence[IslandCandidateRecord | Mapping[str, Any]],
    *,
    count: int = MARKET_ISLAND_SELECTION_COUNT,
) -> tuple[str, ...]:
    return tuple(candidate.island for candidate in rank_island_candidates(candidates)[:count])


def build_island_selection_record(
    *,
    selection_id: str,
    candidates: Sequence[IslandCandidateRecord | Mapping[str, Any]],
    uses_production_data: bool = False,
    final_selection: bool = False,
    local_only: bool = True,
    state: str = IslandSelectionState.WORKING_PLACEHOLDER.value,
    owner_approval_ref: str = "",
    evidence_refs: Sequence[str] = (),
) -> IslandSelectionRecord:
    parsed = [
        candidate if isinstance(candidate, IslandCandidateRecord) else IslandCandidateRecord.from_mapping(candidate)
        for candidate in candidates
    ]
    return IslandSelectionRecord(
        selection_id=selection_id,
        candidate_refs=tuple(candidate.candidate_id for candidate in parsed),
        selected_islands=select_market_islands(parsed),
        uses_production_data=uses_production_data,
        final_selection=final_selection,
        local_only=local_only,
        state=state,
        owner_approval_ref=owner_approval_ref,
        evidence_refs=tuple(str(item) for item in evidence_refs),
    )


def validate_island_candidate(record: IslandCandidateRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, IslandCandidateRecord):
        record = IslandCandidateRecord.from_mapping(record)
    errors: list[str] = []
    if not record.candidate_id:
        errors.append("island candidate requires candidate_id")
    if not record.island:
        errors.append("island candidate requires island")
    if not record.measurement_window:
        errors.append("island candidate requires measurement_window")
    if record.data_state not in {state.value for state in IslandCandidateDataState}:
        errors.append(f"unknown island candidate data_state: {record.data_state}")
    if record.delivered_lead_volume < 0:
        errors.append("delivered_lead_volume must be non-negative")
    if not 0.0 <= record.outcome_label_density <= 1.0:
        errors.append("outcome_label_density must be between 0 and 1")
    if not record.source_refs:
        errors.append("island candidate requires source_refs")
    if record.data_state == IslandCandidateDataState.WORKING_PLACEHOLDER.value:
        if record.production_data_ready:
            errors.append("working-placeholder island candidate must not claim production_data_ready")
        if not record.local_only:
            errors.append("working-placeholder island candidate must remain local_only")
    if record.data_state == IslandCandidateDataState.PRODUCTION_MEASURED.value:
        if not record.production_data_ready:
            errors.append("production-measured island candidate requires production_data_ready")
        if record.local_only:
            errors.append("production-measured island candidate must not be local_only")
    if record.data_state == IslandCandidateDataState.BLOCKED.value and record.production_data_ready:
        errors.append("blocked island candidate must not claim production_data_ready")
    return errors


def validate_island_selection_record(
    record: IslandSelectionRecord | Mapping[str, Any],
    candidates: Sequence[IslandCandidateRecord | Mapping[str, Any]],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    if not isinstance(record, IslandSelectionRecord):
        record = IslandSelectionRecord.from_mapping(record)
    parsed_candidates = [
        candidate if isinstance(candidate, IslandCandidateRecord) else IslandCandidateRecord.from_mapping(candidate)
        for candidate in candidates
    ]
    errors: list[str] = []
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in IslandSelectionState}:
        errors.append(f"unknown island selection state: {record.state}")
        return errors
    if not record.selection_id:
        errors.append("island selection requires selection_id")
    if record.formula != ISLAND_SELECTION_FORMULA:
        errors.append(f"island selection formula must be {ISLAND_SELECTION_FORMULA!r}")
    if record.tie_breaker != ISLAND_SELECTION_TIE_BREAKER:
        errors.append("island selection tie_breaker does not match the deterministic contract")
    if record.selection_count != MARKET_ISLAND_SELECTION_COUNT:
        errors.append("Phase 2 island selection_count must be 3")
    if len(parsed_candidates) < record.selection_count:
        errors.append("island selection requires at least selection_count candidates")
    candidate_errors = []
    for candidate in parsed_candidates:
        candidate_errors.extend(validate_island_candidate(candidate))
    if candidate_errors:
        errors.extend(candidate_errors)
    candidate_id_list = [candidate.candidate_id for candidate in parsed_candidates]
    candidate_ids = set(candidate_id_list)
    if len(candidate_id_list) != len(candidate_ids):
        errors.append("island candidate records must have unique candidate_id values")
    if len(record.candidate_refs) != len(set(record.candidate_refs)):
        errors.append("island selection candidate_refs must not contain duplicates")
    missing_refs = [candidate_ref for candidate_ref in record.candidate_refs if candidate_ref not in candidate_ids]
    if missing_refs:
        errors.append("island selection references unknown candidate_refs: " + ", ".join(missing_refs))
    if set(record.candidate_refs) != candidate_ids or len(record.candidate_refs) != len(candidate_id_list):
        errors.append("island selection candidate_refs must match candidate records exactly")
    expected_islands = select_market_islands(parsed_candidates, count=record.selection_count)
    if record.selected_islands != expected_islands:
        errors.append(
            "selected_islands does not match formula ranking; expected "
            + ", ".join(expected_islands)
        )
    if record.production_routing_enabled:
        errors.append("P2.1 must not enable production island routing")
    if record.island_champion_lines_created:
        errors.append("P2.1 must not create live island champion lines")
    if record.state == IslandSelectionState.WORKING_PLACEHOLDER.value:
        if record.selected_islands != DEFAULT_WORKING_ISLANDS:
            errors.append("working-placeholder selected_islands must be healthcare, fintech, industrial")
        if record.uses_production_data:
            errors.append("working-placeholder island selection must not claim production data")
        if record.final_selection:
            errors.append("working-placeholder island selection must not claim final_selection")
        if not record.local_only:
            errors.append("working-placeholder island selection must remain local_only")
    if record.state == IslandSelectionState.FINALIZED_AFTER_PRODUCTION_DATA.value or record.final_selection:
        if not record.final_selection:
            errors.append("finalized island selection state requires final_selection")
        if not record.uses_production_data:
            errors.append("finalized island selection requires production data")
        if record.local_only:
            errors.append("finalized island selection must not be local_only")
        if not record.owner_approval_ref:
            errors.append("finalized island selection requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("finalized island selection requires evidence_refs")
        not_ready = [
            candidate.island
            for candidate in parsed_candidates
            if candidate.data_state != IslandCandidateDataState.PRODUCTION_MEASURED.value
            or not candidate.production_data_ready
            or candidate.local_only
        ]
        if not_ready:
            errors.append("finalized island selection requires production-measured candidates: " + ", ".join(not_ready))
    if record.state == IslandSelectionState.FORMULA_READY_LOCAL_REVIEW.value:
        if not record.local_only:
            errors.append("formula-ready local review selection must remain local_only")
        if record.final_selection:
            errors.append("formula-ready local review selection must not claim final_selection")
    return errors


def verify_research_lab_island_selection(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    fixture = _load_fixture(Path(fixture_path))

    candidates = [IslandCandidateRecord.from_mapping(item) for item in fixture["candidates"]]
    for candidate in candidates:
        _assert(not validate_island_candidate(candidate), f"island candidate validates: {candidate.candidate_id}")

    for record in fixture["invalid_candidates"]:
        errors = validate_island_candidate(record)
        _assert(errors, f"invalid island candidate fails: {record['candidate_id']}")
        _assert_expected_error(errors, record)

    selection = IslandSelectionRecord.from_mapping(fixture["selection"])
    _assert(
        not validate_island_selection_record(selection, candidates),
        "working island selection validates",
    )
    built = build_island_selection_record(
        selection_id="island_selection:p2.1:built-from-candidates",
        candidates=candidates,
    )
    _assert(built.selected_islands == selection.selected_islands, "builder selects the same working islands")

    for record in fixture["invalid_selections"]:
        errors = validate_island_selection_record(record, candidates)
        _assert(errors, f"invalid island selection fails: {record['selection_id']}")
        _assert_expected_error(errors, record)

    unsafe_errors = validate_island_selection_record(
        selection,
        candidates,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 2 workflow guards block island selection")

    tie_candidates = [IslandCandidateRecord.from_mapping(item) for item in fixture["tie_break_candidates"]]
    tie_expected = tuple(fixture["tie_break_expected"])
    _assert(select_market_islands(tie_candidates) == tie_expected, "island tie-breaker is deterministic")

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "candidate_count": len(candidates),
        "selected_islands": list(selection.selected_islands),
        "tie_break_selected_islands": list(tie_expected),
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
