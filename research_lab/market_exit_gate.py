"""Phase 2 exit-gate verifier contracts.

P2.10 summarizes Phase 2 operational readiness without enabling it. The local
verifier distinguishes code completeness from production readiness and remains
fail-closed until measured inputs exist for active components, crowned
champions, island crowns, anchored ledger audits, measured yield, and the
counterfactual gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .component_market import (
    PROTECTED_COMPONENT_KEYS,
    PROTECTED_COMPONENT_MARKERS,
    verify_research_lab_component_market,
)
from .counterfactual_gate import verify_research_lab_counterfactual_gate
from .island_selection import verify_research_lab_island_selection
from .market_foundation import (
    MARKET_DEPENDENCY_GATES,
    MarketCarryForwardStatus,
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    market_gate_ready,
    verify_market_foundation,
)
from .receipt_ledger_audit import verify_research_lab_receipt_ledger_audit


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "market_exit_gate_fixtures.json"

MARKET_EXIT_GATE_CONTRACT_VERSION = "market_exit_gate:v1:local_contract"
MARKET_ACTIVE_COMPONENT_MIN = 10
MARKET_DISTINCT_ISLAND_CROWN_MINERS_MIN = 2
MARKET_MEASURED_YIELD_THRESHOLD = 1.0

PROTECTED_EXIT_GATE_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_COMPONENT_KEYS)
    | {
        "customer_outcome",
        "live_champion_code",
        "live_champion_prompt",
        "private_customer_data",
        "raw_counterfactual_payload",
        "raw_delivery_data",
        "raw_ledger_entry",
        "raw_outcome_label",
        "sealed_eval_details",
    }
)

PROTECTED_EXIT_GATE_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_COMPONENT_MARKERS)
        | {
            "customer outcome",
            "live champion",
            "private customer",
            "raw counterfactual",
            "raw delivery",
            "raw ledger",
            "raw outcome",
            "sealed eval",
        }
    )
)


class MarketExitGateCheck(str, Enum):
    CARRY_FORWARD_GATES_COMPLETE = "carry_forward_gates_complete"
    ACTIVE_COMPONENT_COUNT = "active_component_count"
    COMPONENT_IN_CROWNED_CHAMPION = "component_in_crowned_champion"
    ISLAND_CROWN_DIVERSITY = "island_crown_diversity"
    LEDGER_AUDITS_ANCHORED = "ledger_audits_anchored"
    MEASURED_YIELD_THRESHOLD = "measured_yield_threshold"
    COUNTERFACTUAL_GATE_PASSED = "counterfactual_gate_passed"


class MarketExitGateState(str, Enum):
    LOCAL_NOT_READY = "local_not_ready"
    READY_AFTER_PRODUCTION_EVIDENCE = "ready_after_production_evidence"
    BLOCKED = "blocked"


MARKET_EXIT_GATE_CHECKS: tuple[str, ...] = tuple(check.value for check in MarketExitGateCheck)


@dataclass(frozen=True)
class MarketExitGateReadinessRecord:
    exit_gate_id: str
    carry_forward_status: MarketCarryForwardStatus
    active_component_count: int
    active_component_refs: tuple[str, ...]
    component_in_crowned_champion: bool
    crowned_champion_component_refs: tuple[str, ...]
    island_crown_miner_refs: tuple[str, ...]
    ledger_audits_anchored: bool
    ledger_audit_refs: tuple[str, ...]
    measured_yield_points_per_1000_usd: float
    measured_yield_ready: bool
    counterfactual_gate_passed: bool
    counterfactual_comparison_ref: str
    evidence_refs: tuple[str, ...]
    owner_approval_ref: str = ""
    min_active_component_count: int = MARKET_ACTIVE_COMPONENT_MIN
    min_distinct_island_crown_miners: int = MARKET_DISTINCT_ISLAND_CROWN_MINERS_MIN
    measured_yield_threshold: float = MARKET_MEASURED_YIELD_THRESHOLD
    production_operational_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    production_writes_enabled: bool = False
    supabase_writes_enabled: bool = False
    public_workflows_enabled: bool = False
    state: str = MarketExitGateState.LOCAL_NOT_READY.value
    contract_version: str = MARKET_EXIT_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketExitGateReadinessRecord":
        return cls(
            exit_gate_id=str(data["exit_gate_id"]),
            carry_forward_status=MarketCarryForwardStatus.from_mapping(data.get("carry_forward_status", {})),
            active_component_count=int(data.get("active_component_count", 0)),
            active_component_refs=tuple(str(item) for item in data.get("active_component_refs", [])),
            component_in_crowned_champion=bool(data.get("component_in_crowned_champion", False)),
            crowned_champion_component_refs=tuple(str(item) for item in data.get("crowned_champion_component_refs", [])),
            island_crown_miner_refs=tuple(str(item) for item in data.get("island_crown_miner_refs", [])),
            ledger_audits_anchored=bool(data.get("ledger_audits_anchored", False)),
            ledger_audit_refs=tuple(str(item) for item in data.get("ledger_audit_refs", [])),
            measured_yield_points_per_1000_usd=float(data.get("measured_yield_points_per_1000_usd", 0.0)),
            measured_yield_ready=bool(data.get("measured_yield_ready", False)),
            counterfactual_gate_passed=bool(data.get("counterfactual_gate_passed", False)),
            counterfactual_comparison_ref=str(data.get("counterfactual_comparison_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            min_active_component_count=int(data.get("min_active_component_count", MARKET_ACTIVE_COMPONENT_MIN)),
            min_distinct_island_crown_miners=int(
                data.get("min_distinct_island_crown_miners", MARKET_DISTINCT_ISLAND_CROWN_MINERS_MIN)
            ),
            measured_yield_threshold=float(data.get("measured_yield_threshold", MARKET_MEASURED_YIELD_THRESHOLD)),
            production_operational_ready=bool(data.get("production_operational_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            production_writes_enabled=bool(data.get("production_writes_enabled", False)),
            supabase_writes_enabled=bool(data.get("supabase_writes_enabled", False)),
            public_workflows_enabled=bool(data.get("public_workflows_enabled", False)),
            state=str(data.get("state", MarketExitGateState.LOCAL_NOT_READY.value)),
            contract_version=str(data.get("contract_version", MARKET_EXIT_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["carry_forward_status"] = self.carry_forward_status.to_dict()
        data["active_component_refs"] = list(self.active_component_refs)
        data["crowned_champion_component_refs"] = list(self.crowned_champion_component_refs)
        data["island_crown_miner_refs"] = list(self.island_crown_miner_refs)
        data["ledger_audit_refs"] = list(self.ledger_audit_refs)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def market_exit_gate_check_results(
    record: MarketExitGateReadinessRecord | Mapping[str, Any],
) -> dict[str, bool]:
    if not isinstance(record, MarketExitGateReadinessRecord):
        record = MarketExitGateReadinessRecord.from_mapping(record)
    return {
        MarketExitGateCheck.CARRY_FORWARD_GATES_COMPLETE.value: all(
            market_gate_ready(record.carry_forward_status, gate)
            for gate in MARKET_DEPENDENCY_GATES
        ),
        MarketExitGateCheck.ACTIVE_COMPONENT_COUNT.value: (
            record.active_component_count >= record.min_active_component_count
            and len(record.active_component_refs) >= record.min_active_component_count
        ),
        MarketExitGateCheck.COMPONENT_IN_CROWNED_CHAMPION.value: (
            record.component_in_crowned_champion
            and bool(record.crowned_champion_component_refs)
        ),
        MarketExitGateCheck.ISLAND_CROWN_DIVERSITY.value: (
            len(set(record.island_crown_miner_refs)) >= record.min_distinct_island_crown_miners
        ),
        MarketExitGateCheck.LEDGER_AUDITS_ANCHORED.value: (
            record.ledger_audits_anchored
            and any(ref.startswith("balance_ledger_audit:") for ref in record.ledger_audit_refs)
            and any(ref.startswith("cost_ledger_audit:") for ref in record.ledger_audit_refs)
        ),
        MarketExitGateCheck.MEASURED_YIELD_THRESHOLD.value: (
            record.measured_yield_ready
            and record.measured_yield_points_per_1000_usd >= record.measured_yield_threshold
        ),
        MarketExitGateCheck.COUNTERFACTUAL_GATE_PASSED.value: (
            record.counterfactual_gate_passed
            and record.counterfactual_comparison_ref.startswith("counterfactual_comparison:")
        ),
    }


def market_exit_gate_missing_checks(
    record: MarketExitGateReadinessRecord | Mapping[str, Any],
) -> tuple[str, ...]:
    results = market_exit_gate_check_results(record)
    return tuple(check for check, passed in results.items() if not passed)


def validate_market_exit_gate_readiness_record(
    record: MarketExitGateReadinessRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_exit_gate_payload_errors(raw)
    if not isinstance(record, MarketExitGateReadinessRecord):
        record = MarketExitGateReadinessRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != MARKET_EXIT_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.10 exit-gate contract")
    if record.state not in {state.value for state in MarketExitGateState}:
        errors.append(f"unknown Phase 2 exit-gate state: {record.state}")
    if not record.exit_gate_id.startswith("market_exit_gate:"):
        errors.append("exit_gate_id must be market_exit_gate:-prefixed")
    if record.min_active_component_count != MARKET_ACTIVE_COMPONENT_MIN:
        errors.append("min_active_component_count must match Phase 2 exit-gate threshold")
    if record.min_distinct_island_crown_miners != MARKET_DISTINCT_ISLAND_CROWN_MINERS_MIN:
        errors.append("min_distinct_island_crown_miners must match Phase 2 exit-gate threshold")
    if record.measured_yield_threshold != MARKET_MEASURED_YIELD_THRESHOLD:
        errors.append("measured_yield_threshold must match Phase 2 exit-gate threshold")
    if record.active_component_count < 0:
        errors.append("active_component_count must be non-negative")
    if record.measured_yield_points_per_1000_usd < 0:
        errors.append("measured_yield_points_per_1000_usd must be non-negative")
    if record.component_in_crowned_champion and not record.crowned_champion_component_refs:
        errors.append("component_in_crowned_champion requires crowned_champion_component_refs")
    for component_ref in record.active_component_refs + record.crowned_champion_component_refs:
        if not component_ref.startswith("component:"):
            errors.append("component refs must be component:-prefixed")
            break
    for miner_ref in record.island_crown_miner_refs:
        if not miner_ref.startswith("miner:"):
            errors.append("island_crown_miner_refs must be miner:-prefixed")
            break
    for audit_ref in record.ledger_audit_refs:
        if not audit_ref.startswith(("balance_ledger_audit:", "cost_ledger_audit:")):
            errors.append("ledger_audit_refs must be balance_ledger_audit: or cost_ledger_audit:-prefixed")
            break
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "market_readiness:",
                "component:",
                "champion:",
                "island_crown:",
                "balance_ledger_audit:",
                "cost_ledger_audit:",
                "counterfactual_comparison:",
                "results_ledger:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved Phase 2 exit-gate prefixes")
            break
    if record.production_writes_enabled:
        errors.append("P2.10 exit-gate verifier must not enable production writes")
    if record.supabase_writes_enabled:
        errors.append("P2.10 exit-gate verifier must not enable Supabase writes")
    if record.public_workflows_enabled:
        errors.append("P2.10 exit-gate verifier must not enable public workflows")

    missing = market_exit_gate_missing_checks(record)
    if record.production_operational_ready:
        if record.uses_local_fixtures:
            errors.append("Phase 2 operational readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 2 operational readiness cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 2 operational readiness missing exit gates: " + ", ".join(missing))
        if not record.evidence_refs:
            errors.append("Phase 2 operational readiness requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Phase 2 operational readiness requires owner_approval_ref")
        if record.state != MarketExitGateState.READY_AFTER_PRODUCTION_EVIDENCE.value:
            errors.append("production_operational_ready requires ready_after_production_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready Phase 2 exit-gate records must remain local_only")
        if record.state == MarketExitGateState.READY_AFTER_PRODUCTION_EVIDENCE.value:
            errors.append("ready_after_production_evidence state requires production_operational_ready")
    return errors


def verify_market_exit_gate(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    component_summary = verify_research_lab_component_market()
    island_summary = verify_research_lab_island_selection()
    receipt_summary = verify_research_lab_receipt_ledger_audit()
    counterfactual_summary = verify_research_lab_counterfactual_gate()
    fixture = _load_fixture(Path(fixture_path))

    local_record = MarketExitGateReadinessRecord.from_mapping(fixture["local_not_ready"])
    _assert(not validate_market_exit_gate_readiness_record(local_record), "local not-ready exit gate validates")
    local_results = market_exit_gate_check_results(local_record)
    local_missing = market_exit_gate_missing_checks(local_record)
    _assert(not local_record.production_operational_ready, "local fixture does not mark production ready")
    _assert(local_missing, "local fixture reports missing exit gates")
    _assert(len(component_summary["component_types"]) > 1, "P2.10 verifier composes with P2.2 component registry")
    _assert(local_record.active_component_count == 0, "local fixture does not treat component types as active components")
    _assert(local_record.ledger_audits_anchored, "local exit record can cite anchored ledger audit stubs")
    _assert(receipt_summary["balance_audit_id"] in local_record.ledger_audit_refs, "exit gate pins P2.8 balance audit")
    _assert(receipt_summary["cost_audit_id"] in local_record.ledger_audit_refs, "exit gate pins P2.8 cost audit")
    _assert(local_record.counterfactual_comparison_ref == counterfactual_summary["comparison_id"], "exit gate pins P2.9 comparison")
    _assert(
        len(local_record.island_crown_miner_refs) < len(island_summary["selected_islands"]),
        "local fixture does not invent island crowns from local island selection",
    )

    for invalid in fixture["invalid_readiness_claims"]:
        base = fixture[str(invalid.get("base", "ready_claim_for_negative_tests"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_market_exit_gate_readiness_record(record)
        _assert(errors, f"invalid Phase 2 exit readiness fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_errors = validate_market_exit_gate_readiness_record(
        local_record,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 2 guards block exit-gate verifier")
    _assert_expected_error(unsafe_errors, fixture["unsafe_workflow_guards"])

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "local_code_complete": True,
        "production_operational_ready": local_record.production_operational_ready,
        "exit_gate_id": local_record.exit_gate_id,
        "missing_exit_gates": list(local_missing),
        "check_results": local_results,
        "component_types_seen": len(component_summary["component_types"]),
        "counterfactual_comparison": counterfactual_summary["comparison_id"],
    }


def _protected_exit_gate_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_exit_gate_material(record))
    if not found:
        return []
    return ["Phase 2 exit-gate payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_exit_gate_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_EXIT_GATE_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_exit_gate_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_exit_gate_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_EXIT_GATE_MARKERS:
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
