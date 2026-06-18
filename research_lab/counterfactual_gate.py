"""Phase 2 counterfactual-gate local contracts.

P2.9 defines the matched-budget counterfactual gate between miner-briefed
yield and allocator-directed baseline-arm yield. It is deliberately inert:
validators can calculate, publish local stubs, and record future consequence
intent, but they do not alter aimed-ticket shares, redesign surfaces, mutate
balances, write production state, or execute the two-quarter consequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .baseline_arm_ops import verify_research_lab_baseline_arm_ops
from .canonical import sha256_json
from .meta_allocator import verify_research_lab_meta_allocator
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
from .receipt_ledger_audit import (
    PROTECTED_LEDGER_AUDIT_KEYS,
    PROTECTED_LEDGER_AUDIT_MARKERS,
    verify_research_lab_receipt_ledger_audit,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "counterfactual_gate_fixtures.json"

COUNTERFACTUAL_GATE_CONTRACT_VERSION = "counterfactual_gate:v1:local_contract"
PENDING_COUNTERFACTUAL_PUBLICATION_REF = "counterfactual_publication:pending"

PROTECTED_COUNTERFACTUAL_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_LEDGER_AUDIT_KEYS)
    | {
        "customer_outcome",
        "live_champion_artifact",
        "live_champion_code",
        "live_champion_prompt",
        "private_customer_data",
        "raw_allocator_trace",
        "raw_brief",
        "raw_customer_data",
        "raw_evidence",
        "raw_miner_prompt",
        "sealed_eval_details",
        "sealed_judge_prompt",
    }
)

PROTECTED_COUNTERFACTUAL_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_LEDGER_AUDIT_MARKERS)
        | {
            "customer outcome",
            "live champion",
            "private customer",
            "raw allocator",
            "raw brief",
            "raw customer",
            "raw evidence",
            "raw miner prompt",
            "sealed eval",
            "sealed judge prompt",
        }
    )
)


class CounterfactualYieldArmKind(str, Enum):
    MINER_BRIEFED = "miner_briefed"
    ALLOCATOR_DIRECTED = "allocator_directed"


class CounterfactualYieldDataState(str, Enum):
    LOCAL_FIXTURE = "local_fixture"
    MEASURED_LAB_ONLY = "measured_lab_only"
    MEASURED_PRODUCTION = "measured_production"
    BLOCKED = "blocked"


class CounterfactualComparisonState(str, Enum):
    LOCAL_STUB = "local_stub"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"


class CounterfactualMethodologyState(str, Enum):
    LOCAL_PUBLICATION_STUB = "local_publication_stub"
    READY_AFTER_MEASURED_DATA = "ready_after_measured_data"
    PUBLISHED_LIVE = "published_live"
    BLOCKED = "blocked"


class CounterfactualFailureTrackerState(str, Enum):
    LOCAL_TRACKER = "local_tracker"
    CONSEQUENCE_DUE = "consequence_due"
    RESET = "reset"
    BLOCKED = "blocked"


class CounterfactualConsequenceState(str, Enum):
    LOCAL_STUB = "local_stub"
    EXECUTED = "executed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CounterfactualYieldRecord:
    yield_id: str
    arm_kind: str
    quarter_ref: str
    budget_cents: int
    verified_points: float
    novelty_weighted_points: float
    run_count: int
    receipt_refs: tuple[str, ...]
    ledger_audit_refs: tuple[str, ...]
    anchor_proposal_ref: str
    allocator_selection_refs: tuple[str, ...] = ()
    brief_refs: tuple[str, ...] = ()
    source_data_state: str = CounterfactualYieldDataState.LOCAL_FIXTURE.value
    uses_local_fixtures: bool = True
    measured_data_ready: bool = False
    production_data_claimed: bool = False
    local_only: bool = True
    contract_version: str = COUNTERFACTUAL_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CounterfactualYieldRecord":
        return cls(
            yield_id=str(data["yield_id"]),
            arm_kind=str(data["arm_kind"]),
            quarter_ref=str(data["quarter_ref"]),
            budget_cents=int(data["budget_cents"]),
            verified_points=float(data["verified_points"]),
            novelty_weighted_points=float(data["novelty_weighted_points"]),
            run_count=int(data["run_count"]),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            ledger_audit_refs=tuple(str(item) for item in data.get("ledger_audit_refs", [])),
            anchor_proposal_ref=str(data["anchor_proposal_ref"]),
            allocator_selection_refs=tuple(str(item) for item in data.get("allocator_selection_refs", [])),
            brief_refs=tuple(str(item) for item in data.get("brief_refs", [])),
            source_data_state=str(data.get("source_data_state", CounterfactualYieldDataState.LOCAL_FIXTURE.value)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            production_data_claimed=bool(data.get("production_data_claimed", False)),
            local_only=bool(data.get("local_only", True)),
            contract_version=str(data.get("contract_version", COUNTERFACTUAL_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["receipt_refs"] = list(self.receipt_refs)
        data["ledger_audit_refs"] = list(self.ledger_audit_refs)
        data["allocator_selection_refs"] = list(self.allocator_selection_refs)
        data["brief_refs"] = list(self.brief_refs)
        return data

    def identity_payload(self) -> dict[str, Any]:
        return {
            "arm_kind": self.arm_kind,
            "quarter_ref": self.quarter_ref,
            "budget_cents": self.budget_cents,
            "verified_points": self.verified_points,
            "novelty_weighted_points": self.novelty_weighted_points,
            "run_count": self.run_count,
            "receipt_refs": list(self.receipt_refs),
            "ledger_audit_refs": list(self.ledger_audit_refs),
            "anchor_proposal_ref": self.anchor_proposal_ref,
            "allocator_selection_refs": list(self.allocator_selection_refs),
            "brief_refs": list(self.brief_refs),
            "source_data_state": self.source_data_state,
        }


@dataclass(frozen=True)
class CounterfactualComparisonRecord:
    comparison_id: str
    quarter_ref: str
    miner_yield_ref: str
    allocator_yield_ref: str
    matched_budget_cents: int
    miner_budget_cents: int
    allocator_budget_cents: int
    miner_verified_points: float
    allocator_verified_points: float
    miner_yield_points_per_1000_usd: float
    allocator_yield_points_per_1000_usd: float
    delta_yield_points_per_1000_usd: float
    passed_gate: bool
    methodology_ref: str
    shared_prior_caveat: str
    measured_data_ready: bool = False
    uses_local_fixtures: bool = True
    state: str = CounterfactualComparisonState.LOCAL_STUB.value
    local_only: bool = True
    production_gate_claimed: bool = False
    public_claim_enabled: bool = False
    consequence_due: bool = False
    publication_ref: str = PENDING_COUNTERFACTUAL_PUBLICATION_REF
    contract_version: str = COUNTERFACTUAL_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CounterfactualComparisonRecord":
        return cls(
            comparison_id=str(data["comparison_id"]),
            quarter_ref=str(data["quarter_ref"]),
            miner_yield_ref=str(data["miner_yield_ref"]),
            allocator_yield_ref=str(data["allocator_yield_ref"]),
            matched_budget_cents=int(data["matched_budget_cents"]),
            miner_budget_cents=int(data["miner_budget_cents"]),
            allocator_budget_cents=int(data["allocator_budget_cents"]),
            miner_verified_points=float(data["miner_verified_points"]),
            allocator_verified_points=float(data["allocator_verified_points"]),
            miner_yield_points_per_1000_usd=float(data["miner_yield_points_per_1000_usd"]),
            allocator_yield_points_per_1000_usd=float(data["allocator_yield_points_per_1000_usd"]),
            delta_yield_points_per_1000_usd=float(data["delta_yield_points_per_1000_usd"]),
            passed_gate=bool(data.get("passed_gate", False)),
            methodology_ref=str(data["methodology_ref"]),
            shared_prior_caveat=str(data["shared_prior_caveat"]),
            measured_data_ready=bool(data.get("measured_data_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            state=str(data.get("state", CounterfactualComparisonState.LOCAL_STUB.value)),
            local_only=bool(data.get("local_only", True)),
            production_gate_claimed=bool(data.get("production_gate_claimed", False)),
            public_claim_enabled=bool(data.get("public_claim_enabled", False)),
            consequence_due=bool(data.get("consequence_due", False)),
            publication_ref=str(data.get("publication_ref", PENDING_COUNTERFACTUAL_PUBLICATION_REF)),
            contract_version=str(data.get("contract_version", COUNTERFACTUAL_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "quarter_ref": self.quarter_ref,
            "miner_yield_ref": self.miner_yield_ref,
            "allocator_yield_ref": self.allocator_yield_ref,
            "matched_budget_cents": self.matched_budget_cents,
            "miner_budget_cents": self.miner_budget_cents,
            "allocator_budget_cents": self.allocator_budget_cents,
            "miner_verified_points": self.miner_verified_points,
            "allocator_verified_points": self.allocator_verified_points,
            "miner_yield_points_per_1000_usd": self.miner_yield_points_per_1000_usd,
            "allocator_yield_points_per_1000_usd": self.allocator_yield_points_per_1000_usd,
            "delta_yield_points_per_1000_usd": self.delta_yield_points_per_1000_usd,
            "passed_gate": self.passed_gate,
            "methodology_ref": self.methodology_ref,
            "shared_prior_caveat": self.shared_prior_caveat,
            "measured_data_ready": self.measured_data_ready,
            "uses_local_fixtures": self.uses_local_fixtures,
            "state": self.state,
        }


@dataclass(frozen=True)
class CounterfactualMethodologyPublicationRecord:
    publication_id: str
    comparison_ref: str
    title: str
    methodology_summary: str
    shared_prior_caveat: str
    matched_budget_caveat: str
    source_refs: tuple[str, ...]
    publication_state: str = CounterfactualMethodologyState.LOCAL_PUBLICATION_STUB.value
    publication_ref: str = PENDING_COUNTERFACTUAL_PUBLICATION_REF
    local_only: bool = True
    live_publication_enabled: bool = False
    production_publish_enabled: bool = False
    claims_counterfactual_gate_satisfied: bool = False
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value
    contract_version: str = COUNTERFACTUAL_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CounterfactualMethodologyPublicationRecord":
        return cls(
            publication_id=str(data["publication_id"]),
            comparison_ref=str(data["comparison_ref"]),
            title=str(data["title"]),
            methodology_summary=str(data["methodology_summary"]),
            shared_prior_caveat=str(data["shared_prior_caveat"]),
            matched_budget_caveat=str(data["matched_budget_caveat"]),
            source_refs=tuple(str(item) for item in data.get("source_refs", [])),
            publication_state=str(data.get("publication_state", CounterfactualMethodologyState.LOCAL_PUBLICATION_STUB.value)),
            publication_ref=str(data.get("publication_ref", PENDING_COUNTERFACTUAL_PUBLICATION_REF)),
            local_only=bool(data.get("local_only", True)),
            live_publication_enabled=bool(data.get("live_publication_enabled", False)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            claims_counterfactual_gate_satisfied=bool(data.get("claims_counterfactual_gate_satisfied", False)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
            contract_version=str(data.get("contract_version", COUNTERFACTUAL_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = list(self.source_refs)
        return data


@dataclass(frozen=True)
class CounterfactualFailureTrackerRecord:
    tracker_id: str
    quarter_refs: tuple[str, ...]
    comparison_refs: tuple[str, ...]
    failure_count: int
    consecutive_failure_count: int
    two_consecutive_failures: bool
    consequence_stub_ref: str
    state: str = CounterfactualFailureTrackerState.LOCAL_TRACKER.value
    local_only: bool = True
    uses_local_fixtures: bool = True
    consequence_execution_enabled: bool = False
    aimed_ticket_share_changed: bool = False
    production_policy_changed: bool = False
    contract_version: str = COUNTERFACTUAL_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CounterfactualFailureTrackerRecord":
        return cls(
            tracker_id=str(data["tracker_id"]),
            quarter_refs=tuple(str(item) for item in data.get("quarter_refs", [])),
            comparison_refs=tuple(str(item) for item in data.get("comparison_refs", [])),
            failure_count=int(data["failure_count"]),
            consecutive_failure_count=int(data["consecutive_failure_count"]),
            two_consecutive_failures=bool(data.get("two_consecutive_failures", False)),
            consequence_stub_ref=str(data.get("consequence_stub_ref", "")),
            state=str(data.get("state", CounterfactualFailureTrackerState.LOCAL_TRACKER.value)),
            local_only=bool(data.get("local_only", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            consequence_execution_enabled=bool(data.get("consequence_execution_enabled", False)),
            aimed_ticket_share_changed=bool(data.get("aimed_ticket_share_changed", False)),
            production_policy_changed=bool(data.get("production_policy_changed", False)),
            contract_version=str(data.get("contract_version", COUNTERFACTUAL_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quarter_refs"] = list(self.quarter_refs)
        data["comparison_refs"] = list(self.comparison_refs)
        return data


@dataclass(frozen=True)
class CounterfactualConsequenceStubRecord:
    consequence_id: str
    tracker_ref: str
    triggering_comparison_refs: tuple[str, ...]
    aimed_ticket_share_shift_target: str
    surface_redesign_ref: str
    precommitted_action_summary: str
    state: str = CounterfactualConsequenceState.LOCAL_STUB.value
    executed: bool = False
    counterfactual_consequences_enabled: bool = False
    aimed_ticket_share_changed: bool = False
    surface_redesign_applied: bool = False
    production_policy_changed: bool = False
    production_writes: bool = False
    public_workflow_enabled: bool = False
    owner_approval_ref: str = ""
    local_only: bool = True
    contract_version: str = COUNTERFACTUAL_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CounterfactualConsequenceStubRecord":
        return cls(
            consequence_id=str(data["consequence_id"]),
            tracker_ref=str(data["tracker_ref"]),
            triggering_comparison_refs=tuple(str(item) for item in data.get("triggering_comparison_refs", [])),
            aimed_ticket_share_shift_target=str(data["aimed_ticket_share_shift_target"]),
            surface_redesign_ref=str(data["surface_redesign_ref"]),
            precommitted_action_summary=str(data["precommitted_action_summary"]),
            state=str(data.get("state", CounterfactualConsequenceState.LOCAL_STUB.value)),
            executed=bool(data.get("executed", False)),
            counterfactual_consequences_enabled=bool(data.get("counterfactual_consequences_enabled", False)),
            aimed_ticket_share_changed=bool(data.get("aimed_ticket_share_changed", False)),
            surface_redesign_applied=bool(data.get("surface_redesign_applied", False)),
            production_policy_changed=bool(data.get("production_policy_changed", False)),
            production_writes=bool(data.get("production_writes", False)),
            public_workflow_enabled=bool(data.get("public_workflow_enabled", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            local_only=bool(data.get("local_only", True)),
            contract_version=str(data.get("contract_version", COUNTERFACTUAL_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["triggering_comparison_refs"] = list(self.triggering_comparison_refs)
        return data


def calculate_yield_points_per_1000_usd(verified_points: float, budget_cents: int) -> float:
    if budget_cents <= 0:
        raise ValueError("budget_cents must be positive")
    return round(float(verified_points) * 100_000 / int(budget_cents), 6)


def build_counterfactual_yield_record(**kwargs: Any) -> CounterfactualYieldRecord:
    record = CounterfactualYieldRecord.from_mapping({"yield_id": "counterfactual_yield:pending", **kwargs})
    data = record.to_dict()
    data["yield_id"] = "counterfactual_yield:" + sha256_json(record.identity_payload()).split(":", 1)[1][:16]
    record = CounterfactualYieldRecord.from_mapping(data)
    errors = validate_counterfactual_yield_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def build_matched_budget_comparison(
    *,
    miner_yield: CounterfactualYieldRecord | Mapping[str, Any],
    allocator_yield: CounterfactualYieldRecord | Mapping[str, Any],
    methodology_ref: str,
    shared_prior_caveat: str,
) -> CounterfactualComparisonRecord:
    if not isinstance(miner_yield, CounterfactualYieldRecord):
        miner_yield = CounterfactualYieldRecord.from_mapping(miner_yield)
    if not isinstance(allocator_yield, CounterfactualYieldRecord):
        allocator_yield = CounterfactualYieldRecord.from_mapping(allocator_yield)
    miner_errors = validate_counterfactual_yield_record(miner_yield)
    allocator_errors = validate_counterfactual_yield_record(allocator_yield)
    if miner_errors or allocator_errors:
        raise ValueError("; ".join(miner_errors + allocator_errors))
    miner_yield_per_1000 = calculate_yield_points_per_1000_usd(miner_yield.verified_points, miner_yield.budget_cents)
    allocator_yield_per_1000 = calculate_yield_points_per_1000_usd(
        allocator_yield.verified_points,
        allocator_yield.budget_cents,
    )
    state = CounterfactualComparisonState.LOCAL_STUB.value
    measured_data_ready = miner_yield.measured_data_ready and allocator_yield.measured_data_ready
    uses_local_fixtures = miner_yield.uses_local_fixtures or allocator_yield.uses_local_fixtures
    draft = CounterfactualComparisonRecord(
        comparison_id="counterfactual_comparison:pending",
        quarter_ref=miner_yield.quarter_ref,
        miner_yield_ref=miner_yield.yield_id,
        allocator_yield_ref=allocator_yield.yield_id,
        matched_budget_cents=miner_yield.budget_cents,
        miner_budget_cents=miner_yield.budget_cents,
        allocator_budget_cents=allocator_yield.budget_cents,
        miner_verified_points=miner_yield.verified_points,
        allocator_verified_points=allocator_yield.verified_points,
        miner_yield_points_per_1000_usd=miner_yield_per_1000,
        allocator_yield_points_per_1000_usd=allocator_yield_per_1000,
        delta_yield_points_per_1000_usd=round(miner_yield_per_1000 - allocator_yield_per_1000, 6),
        passed_gate=miner_yield_per_1000 >= allocator_yield_per_1000,
        methodology_ref=methodology_ref,
        shared_prior_caveat=shared_prior_caveat,
        measured_data_ready=measured_data_ready,
        uses_local_fixtures=uses_local_fixtures,
        state=state,
    )
    data = draft.to_dict()
    data["comparison_id"] = "counterfactual_comparison:" + sha256_json(draft.identity_payload()).split(":", 1)[1][:16]
    record = CounterfactualComparisonRecord.from_mapping(data)
    errors = validate_counterfactual_comparison_record(
        record,
        miner_yield=miner_yield,
        allocator_yield=allocator_yield,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_counterfactual_yield_record(record: CounterfactualYieldRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_counterfactual_payload_errors(raw)
    if not isinstance(record, CounterfactualYieldRecord):
        record = CounterfactualYieldRecord.from_mapping(record)
    if record.contract_version != COUNTERFACTUAL_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.9 counterfactual gate contract")
    if record.arm_kind not in {kind.value for kind in CounterfactualYieldArmKind}:
        errors.append(f"unknown counterfactual yield arm_kind: {record.arm_kind}")
    if not record.yield_id.startswith("counterfactual_yield:"):
        errors.append("yield_id must be counterfactual_yield:-prefixed")
    if not record.quarter_ref.startswith("quarter:"):
        errors.append("quarter_ref must be quarter:-prefixed")
    if record.budget_cents <= 0:
        errors.append("budget_cents must be positive")
    if record.verified_points < 0 or record.novelty_weighted_points < 0:
        errors.append("yield points must be non-negative")
    if record.run_count <= 0:
        errors.append("run_count must be positive")
    if not record.receipt_refs:
        errors.append("counterfactual yield requires receipt_refs")
    for receipt_ref in record.receipt_refs:
        if not receipt_ref.startswith("receipt_v2:"):
            errors.append("receipt_refs must be receipt_v2:-prefixed")
            break
    if not record.ledger_audit_refs:
        errors.append("counterfactual yield requires ledger_audit_refs")
    for audit_ref in record.ledger_audit_refs:
        if not audit_ref.startswith(("balance_ledger_audit:", "cost_ledger_audit:")):
            errors.append("ledger_audit_refs must be balance_ledger_audit: or cost_ledger_audit:-prefixed")
            break
    if not record.anchor_proposal_ref.startswith("anchor_proposal:"):
        errors.append("anchor_proposal_ref must be anchor_proposal:-prefixed")
    if record.arm_kind == CounterfactualYieldArmKind.MINER_BRIEFED.value and not record.brief_refs:
        errors.append("miner_briefed yield requires brief_refs")
    if record.arm_kind == CounterfactualYieldArmKind.ALLOCATOR_DIRECTED.value and not record.allocator_selection_refs:
        errors.append("allocator_directed yield requires allocator_selection_refs")
    for selection_ref in record.allocator_selection_refs:
        if not selection_ref.startswith("meta_allocator_selection:"):
            errors.append("allocator_selection_refs must be meta_allocator_selection:-prefixed")
            break
    if record.source_data_state not in {state.value for state in CounterfactualYieldDataState}:
        errors.append(f"unknown counterfactual yield data state: {record.source_data_state}")
    if record.source_data_state == CounterfactualYieldDataState.LOCAL_FIXTURE.value:
        if not record.uses_local_fixtures:
            errors.append("local fixture yield records must be marked uses_local_fixtures")
        if record.measured_data_ready:
            errors.append("local fixture yield records cannot claim measured_data_ready")
    if record.source_data_state in {
        CounterfactualYieldDataState.MEASURED_LAB_ONLY.value,
        CounterfactualYieldDataState.MEASURED_PRODUCTION.value,
    }:
        if record.uses_local_fixtures:
            errors.append("measured yield records must not use local fixtures")
        if not record.measured_data_ready:
            errors.append("measured yield records must mark measured_data_ready")
    if record.production_data_claimed:
        errors.append("P2.9 yield records must not claim production data in local contracts")
    if not record.local_only:
        errors.append("P2.9 yield records must remain local_only")
    return errors


def validate_counterfactual_comparison_record(
    record: CounterfactualComparisonRecord | Mapping[str, Any],
    *,
    miner_yield: Optional[CounterfactualYieldRecord | Mapping[str, Any]] = None,
    allocator_yield: Optional[CounterfactualYieldRecord | Mapping[str, Any]] = None,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_counterfactual_payload_errors(raw)
    if not isinstance(record, CounterfactualComparisonRecord):
        record = CounterfactualComparisonRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != COUNTERFACTUAL_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.9 counterfactual gate contract")
    if record.state not in {state.value for state in CounterfactualComparisonState}:
        errors.append(f"unknown counterfactual comparison state: {record.state}")
    if not record.comparison_id.startswith("counterfactual_comparison:"):
        errors.append("comparison_id must be counterfactual_comparison:-prefixed")
    if not record.quarter_ref.startswith("quarter:"):
        errors.append("quarter_ref must be quarter:-prefixed")
    if not record.miner_yield_ref.startswith("counterfactual_yield:"):
        errors.append("miner_yield_ref must be counterfactual_yield:-prefixed")
    if not record.allocator_yield_ref.startswith("counterfactual_yield:"):
        errors.append("allocator_yield_ref must be counterfactual_yield:-prefixed")
    if record.matched_budget_cents <= 0:
        errors.append("matched_budget_cents must be positive")
    if record.miner_budget_cents != record.allocator_budget_cents:
        errors.append("counterfactual comparison requires matched budgets")
    if record.matched_budget_cents not in {record.miner_budget_cents, record.allocator_budget_cents}:
        errors.append("matched_budget_cents must match miner and allocator budgets")
    if record.miner_verified_points < 0 or record.allocator_verified_points < 0:
        errors.append("comparison verified points must be non-negative")
    _append_yield_consistency_errors(errors, record, miner_yield=miner_yield, allocator_yield=allocator_yield)
    if record.miner_budget_cents > 0 and record.allocator_budget_cents > 0:
        expected_miner_yield = calculate_yield_points_per_1000_usd(
            record.miner_verified_points,
            record.miner_budget_cents,
        )
        expected_allocator_yield = calculate_yield_points_per_1000_usd(
            record.allocator_verified_points,
            record.allocator_budget_cents,
        )
        expected_delta = round(expected_miner_yield - expected_allocator_yield, 6)
        if not _same_float(record.miner_yield_points_per_1000_usd, expected_miner_yield):
            errors.append("miner_yield_points_per_1000_usd does not match points and budget")
        if not _same_float(record.allocator_yield_points_per_1000_usd, expected_allocator_yield):
            errors.append("allocator_yield_points_per_1000_usd does not match points and budget")
        if not _same_float(record.delta_yield_points_per_1000_usd, expected_delta):
            errors.append("delta_yield_points_per_1000_usd must equal miner yield minus allocator yield")
        if record.passed_gate != (expected_miner_yield >= expected_allocator_yield):
            errors.append("passed_gate must reflect miner yield >= allocator yield")
    if "shared" not in record.shared_prior_caveat.lower():
        errors.append("comparison must retain the shared-prior caveat")
    if not record.methodology_ref.startswith("counterfactual_methodology:"):
        errors.append("methodology_ref must be counterfactual_methodology:-prefixed")
    if record.publication_ref != PENDING_COUNTERFACTUAL_PUBLICATION_REF:
        errors.append("publication_ref must remain pending in P2.9 local contracts")
    if record.state in {
        CounterfactualComparisonState.PASSED.value,
        CounterfactualComparisonState.FAILED.value,
    }:
        if record.uses_local_fixtures:
            errors.append("counterfactual pass/fail claims require measured data, not fixtures")
        if not record.measured_data_ready:
            errors.append("counterfactual pass/fail claims require measured_data_ready")
        if record.local_only:
            errors.append("counterfactual pass/fail claims cannot be local_only")
    if record.state == CounterfactualComparisonState.PASSED.value and not record.passed_gate:
        errors.append("passed comparison state requires passed_gate")
    if record.state == CounterfactualComparisonState.FAILED.value and record.passed_gate:
        errors.append("failed comparison state requires failed gate math")
    if record.production_gate_claimed:
        errors.append("P2.9 must not claim production counterfactual gate status")
    if record.public_claim_enabled:
        errors.append("P2.9 must not enable public gate claims")
    if record.consequence_due:
        errors.append("P2.9 comparison records must not execute or trigger consequences directly")
    if not record.local_only and record.state == CounterfactualComparisonState.LOCAL_STUB.value:
        errors.append("local_stub comparisons must remain local_only")
    return errors


def validate_counterfactual_methodology_publication_record(
    record: CounterfactualMethodologyPublicationRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_counterfactual_payload_errors(raw)
    if not isinstance(record, CounterfactualMethodologyPublicationRecord):
        record = CounterfactualMethodologyPublicationRecord.from_mapping(record)
    if record.contract_version != COUNTERFACTUAL_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.9 counterfactual gate contract")
    if record.publication_state not in {state.value for state in CounterfactualMethodologyState}:
        errors.append(f"unknown counterfactual methodology state: {record.publication_state}")
    for field in ("publication_id", "comparison_ref", "title", "methodology_summary", "shared_prior_caveat", "matched_budget_caveat"):
        if not getattr(record, field):
            errors.append(f"counterfactual methodology publication requires {field}")
    if not record.publication_id.startswith("counterfactual_methodology:"):
        errors.append("publication_id must be counterfactual_methodology:-prefixed")
    if not record.comparison_ref.startswith("counterfactual_comparison:"):
        errors.append("comparison_ref must be counterfactual_comparison:-prefixed")
    if not record.source_refs:
        errors.append("counterfactual methodology publication requires source_refs")
    if "shared" not in record.shared_prior_caveat.lower():
        errors.append("methodology publication must retain the shared-prior caveat")
    if "matched" not in record.matched_budget_caveat.lower():
        errors.append("methodology publication must retain the matched-budget caveat")
    if record.publication_ref != PENDING_COUNTERFACTUAL_PUBLICATION_REF:
        errors.append("publication_ref must remain pending in P2.9 local contracts")
    if record.publication_state == CounterfactualMethodologyState.PUBLISHED_LIVE.value:
        errors.append("published_live methodology state is disabled in P2.9 local contracts")
    if record.publication_state == CounterfactualMethodologyState.READY_AFTER_MEASURED_DATA.value:
        errors.append("ready_after_measured_data methodology state is disabled until production/lab measurements exist")
    if not record.local_only:
        errors.append("P2.9 methodology publications must remain local_only")
    if record.live_publication_enabled:
        errors.append("P2.9 must not enable live methodology publication")
    if record.production_publish_enabled:
        errors.append("P2.9 must not publish methodology to production")
    if record.claims_counterfactual_gate_satisfied:
        errors.append("methodology publication must not claim the counterfactual gate is satisfied")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.publication_id,
                artifact_type="map_projection",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                contains_live_champion_ip=record.contains_live_champion_ip,
                contains_sealed_eval_details=record.contains_sealed_eval_details,
                contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
                contains_private_customer_data=record.contains_private_customer_data,
                contains_judge_prompts=record.contains_judge_prompts,
                reason="P2.9 methodology publications expose sanitized local counterfactual methodology only",
            )
        )
    )
    return errors


def validate_counterfactual_failure_tracker_record(
    record: CounterfactualFailureTrackerRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_counterfactual_payload_errors(raw)
    if not isinstance(record, CounterfactualFailureTrackerRecord):
        record = CounterfactualFailureTrackerRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != COUNTERFACTUAL_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.9 counterfactual gate contract")
    if record.state not in {state.value for state in CounterfactualFailureTrackerState}:
        errors.append(f"unknown counterfactual failure tracker state: {record.state}")
    if not record.tracker_id.startswith("counterfactual_failure_tracker:"):
        errors.append("tracker_id must be counterfactual_failure_tracker:-prefixed")
    if not record.quarter_refs:
        errors.append("failure tracker requires quarter_refs")
    for quarter_ref in record.quarter_refs:
        if not quarter_ref.startswith("quarter:"):
            errors.append("failure tracker quarter_refs must be quarter:-prefixed")
            break
    if not record.comparison_refs:
        errors.append("failure tracker requires comparison_refs")
    for comparison_ref in record.comparison_refs:
        if not comparison_ref.startswith("counterfactual_comparison:"):
            errors.append("failure tracker comparison_refs must be counterfactual_comparison:-prefixed")
            break
    if record.failure_count < 0 or record.consecutive_failure_count < 0:
        errors.append("failure counts must be non-negative")
    if record.consecutive_failure_count > record.failure_count:
        errors.append("consecutive_failure_count cannot exceed failure_count")
    if record.two_consecutive_failures != (record.consecutive_failure_count >= 2):
        errors.append("two_consecutive_failures must reflect consecutive_failure_count >= 2")
    if record.two_consecutive_failures and not record.consequence_stub_ref:
        errors.append("two consecutive failures require a consequence_stub_ref")
    if record.consequence_stub_ref and not record.consequence_stub_ref.startswith("counterfactual_consequence_stub:"):
        errors.append("consequence_stub_ref must be counterfactual_consequence_stub:-prefixed")
    if record.state == CounterfactualFailureTrackerState.CONSEQUENCE_DUE.value and not record.two_consecutive_failures:
        errors.append("consequence_due tracker state requires two consecutive failures")
    if record.state == CounterfactualFailureTrackerState.RESET.value and record.consecutive_failure_count:
        errors.append("reset tracker state requires zero consecutive failures")
    if not record.local_only:
        errors.append("P2.9 failure trackers must remain local_only")
    if record.consequence_execution_enabled:
        errors.append("P2.9 failure tracker must not enable consequence execution")
    if record.aimed_ticket_share_changed:
        errors.append("P2.9 failure tracker must not alter aimed-ticket shares")
    if record.production_policy_changed:
        errors.append("P2.9 failure tracker must not alter production policy")
    return errors


def validate_counterfactual_consequence_stub_record(
    record: CounterfactualConsequenceStubRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_counterfactual_payload_errors(raw)
    if not isinstance(record, CounterfactualConsequenceStubRecord):
        record = CounterfactualConsequenceStubRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != COUNTERFACTUAL_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match P2.9 counterfactual gate contract")
    if record.state not in {state.value for state in CounterfactualConsequenceState}:
        errors.append(f"unknown counterfactual consequence state: {record.state}")
    if not record.consequence_id.startswith("counterfactual_consequence_stub:"):
        errors.append("consequence_id must be counterfactual_consequence_stub:-prefixed")
    if not record.tracker_ref.startswith("counterfactual_failure_tracker:"):
        errors.append("tracker_ref must be counterfactual_failure_tracker:-prefixed")
    if not record.triggering_comparison_refs:
        errors.append("consequence stub requires triggering_comparison_refs")
    for comparison_ref in record.triggering_comparison_refs:
        if not comparison_ref.startswith("counterfactual_comparison:"):
            errors.append("triggering_comparison_refs must be counterfactual_comparison:-prefixed")
            break
    if record.aimed_ticket_share_shift_target != "red_team_autopilot":
        errors.append("consequence stub target must be red_team_autopilot")
    if not record.surface_redesign_ref.startswith("surface_redesign_stub:"):
        errors.append("surface_redesign_ref must be surface_redesign_stub:-prefixed")
    if not record.precommitted_action_summary:
        errors.append("consequence stub requires precommitted_action_summary")
    if record.state == CounterfactualConsequenceState.EXECUTED.value:
        errors.append("executed consequence state is disabled in P2.9 local contracts")
    if record.executed:
        errors.append("P2.9 consequence stub must not execute consequences")
    if record.counterfactual_consequences_enabled:
        errors.append("P2.9 consequence stub must not enable counterfactual consequences")
    if record.aimed_ticket_share_changed:
        errors.append("P2.9 consequence stub must not alter aimed-ticket shares")
    if record.surface_redesign_applied:
        errors.append("P2.9 consequence stub must not apply surface redesign")
    if record.production_policy_changed:
        errors.append("P2.9 consequence stub must not alter production policy")
    if record.production_writes:
        errors.append("P2.9 consequence stub must not write production state")
    if record.public_workflow_enabled:
        errors.append("P2.9 consequence stub must not enable public workflows")
    if record.owner_approval_ref:
        errors.append("P2.9 consequence stub must not claim owner approval")
    if not record.local_only:
        errors.append("P2.9 consequence stubs must remain local_only")
    return errors


def verify_research_lab_counterfactual_gate(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    allocator_summary = verify_research_lab_meta_allocator()
    baseline_summary = verify_research_lab_baseline_arm_ops()
    receipt_summary = verify_research_lab_receipt_ledger_audit()
    fixture = _load_fixture(Path(fixture_path))

    miner_yield = build_counterfactual_yield_record(**fixture["miner_yield_input"])
    allocator_yield = build_counterfactual_yield_record(**fixture["allocator_yield_input"])
    _assert(miner_yield.to_dict() == fixture["expected_miner_yield"], "miner yield record is deterministic")
    _assert(allocator_yield.to_dict() == fixture["expected_allocator_yield"], "allocator yield record is deterministic")
    _assert(miner_yield.anchor_proposal_ref == receipt_summary["anchor_proposal"], "miner yield pins P2.8 anchor")
    _assert(allocator_yield.anchor_proposal_ref == receipt_summary["anchor_proposal"], "allocator yield pins P2.8 anchor")
    _assert(receipt_summary["receipt_ref"] in miner_yield.receipt_refs, "miner yield pins P2.8 receipt")
    _assert(receipt_summary["receipt_ref"] in allocator_yield.receipt_refs, "allocator yield pins P2.8 receipt")
    _assert(receipt_summary["balance_audit_id"] in miner_yield.ledger_audit_refs, "miner yield pins balance audit")
    _assert(receipt_summary["cost_audit_id"] in allocator_yield.ledger_audit_refs, "allocator yield pins cost audit")
    _assert(allocator_summary["selection_id"] in allocator_yield.allocator_selection_refs, "allocator yield pins P2.4 selection")
    _assert(not validate_counterfactual_yield_record(miner_yield), "miner yield validates")
    _assert(not validate_counterfactual_yield_record(allocator_yield), "allocator yield validates")
    for invalid in fixture["invalid_yields"]:
        record = _fixture_record(fixture, invalid, invalid["base"])
        errors = validate_counterfactual_yield_record(record)
        _assert(errors, f"invalid counterfactual yield fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    comparison = build_matched_budget_comparison(
        miner_yield=miner_yield,
        allocator_yield=allocator_yield,
        methodology_ref=fixture["comparison_methodology_ref"],
        shared_prior_caveat=fixture["shared_prior_caveat"],
    )
    _assert(comparison.to_dict() == fixture["expected_comparison"], "matched-budget comparison is deterministic")
    _assert(not validate_counterfactual_comparison_record(comparison, miner_yield=miner_yield, allocator_yield=allocator_yield), "comparison validates")
    for invalid in fixture["invalid_comparisons"]:
        record = _fixture_record(fixture, invalid, "expected_comparison")
        errors = validate_counterfactual_comparison_record(record, miner_yield=miner_yield, allocator_yield=allocator_yield)
        _assert(errors, f"invalid counterfactual comparison fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    unsafe_errors = validate_counterfactual_comparison_record(
        comparison,
        miner_yield=miner_yield,
        allocator_yield=allocator_yield,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 2 guards block counterfactual comparison")
    _assert_expected_error(unsafe_errors, fixture["unsafe_workflow_guards"])

    methodology = CounterfactualMethodologyPublicationRecord.from_mapping(fixture["methodology_publication"])
    _assert(methodology.comparison_ref == comparison.comparison_id, "methodology pins comparison")
    _assert(baseline_summary["policy_id"] in methodology.source_refs, "methodology pins P1.9 baseline policy")
    _assert(allocator_summary["selection_id"] in methodology.source_refs, "methodology pins P2.4 selection")
    _assert(not validate_counterfactual_methodology_publication_record(methodology), "methodology publication validates")
    for invalid in fixture["invalid_methodology_publications"]:
        record = _fixture_record(fixture, invalid, "methodology_publication")
        errors = validate_counterfactual_methodology_publication_record(record)
        _assert(errors, f"invalid methodology publication fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    tracker = CounterfactualFailureTrackerRecord.from_mapping(fixture["failure_tracker"])
    _assert(comparison.comparison_id in tracker.comparison_refs, "failure tracker pins comparison")
    _assert(not validate_counterfactual_failure_tracker_record(tracker), "failure tracker validates")
    for invalid in fixture["invalid_failure_trackers"]:
        record = _fixture_record(fixture, invalid, "failure_tracker")
        errors = validate_counterfactual_failure_tracker_record(record)
        _assert(errors, f"invalid failure tracker fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    consequence = CounterfactualConsequenceStubRecord.from_mapping(fixture["consequence_stub"])
    _assert(consequence.tracker_ref == tracker.tracker_id, "consequence stub pins tracker")
    _assert(not validate_counterfactual_consequence_stub_record(consequence), "consequence stub validates")
    for invalid in fixture["invalid_consequence_stubs"]:
        record = _fixture_record(fixture, invalid, "consequence_stub")
        errors = validate_counterfactual_consequence_stub_record(record)
        _assert(errors, f"invalid consequence stub fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "allocator_selection": allocator_summary["selection_id"],
        "baseline_policy": baseline_summary["policy_id"],
        "miner_yield": miner_yield.yield_id,
        "allocator_yield": allocator_yield.yield_id,
        "comparison_id": comparison.comparison_id,
        "comparison_state": comparison.state,
        "matched_budget_cents": comparison.matched_budget_cents,
        "consequence_state": consequence.state,
    }


def _append_yield_consistency_errors(
    errors: list[str],
    record: CounterfactualComparisonRecord,
    *,
    miner_yield: Optional[CounterfactualYieldRecord | Mapping[str, Any]],
    allocator_yield: Optional[CounterfactualYieldRecord | Mapping[str, Any]],
) -> None:
    if miner_yield is not None:
        if not isinstance(miner_yield, CounterfactualYieldRecord):
            miner_yield = CounterfactualYieldRecord.from_mapping(miner_yield)
        if miner_yield.arm_kind != CounterfactualYieldArmKind.MINER_BRIEFED.value:
            errors.append("supplied miner_yield must be miner_briefed")
        if record.miner_yield_ref != miner_yield.yield_id:
            errors.append("comparison miner_yield_ref must match supplied miner yield")
        if record.quarter_ref != miner_yield.quarter_ref:
            errors.append("comparison quarter_ref must match supplied miner yield")
        if record.miner_budget_cents != miner_yield.budget_cents:
            errors.append("comparison miner_budget_cents must match supplied miner yield")
        if not _same_float(record.miner_verified_points, miner_yield.verified_points):
            errors.append("comparison miner_verified_points must match supplied miner yield")
        if record.measured_data_ready and not miner_yield.measured_data_ready:
            errors.append("comparison measured_data_ready requires measured miner yield")
    if allocator_yield is not None:
        if not isinstance(allocator_yield, CounterfactualYieldRecord):
            allocator_yield = CounterfactualYieldRecord.from_mapping(allocator_yield)
        if allocator_yield.arm_kind != CounterfactualYieldArmKind.ALLOCATOR_DIRECTED.value:
            errors.append("supplied allocator_yield must be allocator_directed")
        if record.allocator_yield_ref != allocator_yield.yield_id:
            errors.append("comparison allocator_yield_ref must match supplied allocator yield")
        if record.quarter_ref != allocator_yield.quarter_ref:
            errors.append("comparison quarter_ref must match supplied allocator yield")
        if record.allocator_budget_cents != allocator_yield.budget_cents:
            errors.append("comparison allocator_budget_cents must match supplied allocator yield")
        if not _same_float(record.allocator_verified_points, allocator_yield.verified_points):
            errors.append("comparison allocator_verified_points must match supplied allocator yield")
        if record.measured_data_ready and not allocator_yield.measured_data_ready:
            errors.append("comparison measured_data_ready requires measured allocator yield")


def _same_float(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_record(fixture: Mapping[str, Any], invalid: Mapping[str, Any], base_key: str) -> dict[str, Any]:
    record = dict(fixture[base_key])
    for key, value in invalid.get("patch", {}).items():
        record[key] = value
    return record


def _protected_counterfactual_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_counterfactual_material(record))
    if not found:
        return []
    return ["counterfactual payload contains protected or raw material keys/markers: " + ", ".join(found)]


def _find_protected_counterfactual_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_COUNTERFACTUAL_KEYS:
                found.add(key_path)
            found.update(_find_protected_counterfactual_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_counterfactual_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_COUNTERFACTUAL_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


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
