"""Phase 4.7 research-slice governance contracts.

P4.7 defines local record shapes for a future manual research-slice raise. It
does not change emissions, move treasury funds, mutate balances, write SQL, or
schedule an automatic increase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .counterfactual_gate import (
    CounterfactualComparisonRecord,
    CounterfactualComparisonState,
    PROTECTED_COUNTERFACTUAL_KEYS,
    PROTECTED_COUNTERFACTUAL_MARKERS,
    validate_counterfactual_comparison_record,
    verify_research_lab_counterfactual_gate,
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


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "research_slice_governance_fixtures.json"

RESEARCH_SLICE_PROPOSAL_CONTRACT_VERSION = "research_slice_proposal:v1:local_contract"
RESEARCH_SLICE_DECISION_CONTRACT_VERSION = "research_slice_decision:v1:local_contract"

DEFAULT_RESEARCH_SLICE_BPS = 200
MAX_RESEARCH_SLICE_BPS = 10000
MIN_MARKET_YIELD_DELTA_POINTS_PER_1000_USD = 0.0

PROTECTED_RESEARCH_SLICE_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SCALE_KEYS)
    | set(PROTECTED_COUNTERFACTUAL_KEYS)
    | {
        "beneficiary_private_key",
        "emission_admin_key",
        "raw_budget_mutation",
        "raw_miner_payout",
        "research_slice_wallet_key",
        "treasury_private_key",
        "wallet_seed",
    }
)

PROTECTED_RESEARCH_SLICE_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SCALE_MARKERS)
        | set(PROTECTED_COUNTERFACTUAL_MARKERS)
        | {
            "beneficiary private key",
            "emission admin key",
            "raw budget mutation",
            "raw miner payout",
            "research slice wallet",
            "treasury private key",
            "wallet seed",
        }
    )
)


class ResearchSliceProposalState(str, Enum):
    LOCAL_PROPOSAL_STUB = "local_proposal_stub"
    READY_AFTER_MEASURED_EVIDENCE = "ready_after_measured_evidence"
    BLOCKED = "blocked"


class ResearchSliceDecisionState(str, Enum):
    LOCAL_DECISION_STUB = "local_decision_stub"
    APPROVED_FOR_FUTURE_MANUAL_ROLLOUT = "approved_for_future_manual_rollout"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ResearchSliceGovernanceProposalRecord:
    proposal_id: str
    quarter_ref: str
    model_pipeline_exit_gate_ref: str
    market_yield_ref: str
    counterfactual_comparison_ref: str
    robustness_evidence_ref: str
    abuse_monitoring_ref: str
    fulfillment_health_ref: str
    outcome_label_supply_ref: str
    ceiling_utilization_ref: str
    alpha_price_window_ref: str
    current_research_slice_bps: int = DEFAULT_RESEARCH_SLICE_BPS
    proposed_research_slice_bps: int = DEFAULT_RESEARCH_SLICE_BPS
    max_research_slice_bps: int = MAX_RESEARCH_SLICE_BPS
    measured_market_yield_ready: bool = False
    counterfactual_gate_passed: bool = False
    market_yield_delta_points_per_1000_usd: float = 0.0
    robustness_evidence_ready: bool = False
    abuse_monitoring_ready: bool = False
    fulfillment_health_checked: bool = False
    outcome_label_supply_checked: bool = False
    ceiling_utilization_published: bool = False
    manual_governance_required: bool = True
    formula_only_increase: bool = False
    pre_scheduled_increase: bool = False
    proposal_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_review_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    budget_mutation_enabled: bool = False
    treasury_transfer_enabled: bool = False
    emission_schedule_changed: bool = False
    payment_distribution_changed: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = ResearchSliceProposalState.LOCAL_PROPOSAL_STUB.value
    contract_version: str = RESEARCH_SLICE_PROPOSAL_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchSliceGovernanceProposalRecord":
        return cls(
            proposal_id=str(data["proposal_id"]),
            quarter_ref=str(data["quarter_ref"]),
            model_pipeline_exit_gate_ref=str(data["model_pipeline_exit_gate_ref"]),
            market_yield_ref=str(data["market_yield_ref"]),
            counterfactual_comparison_ref=str(data["counterfactual_comparison_ref"]),
            robustness_evidence_ref=str(data["robustness_evidence_ref"]),
            abuse_monitoring_ref=str(data["abuse_monitoring_ref"]),
            fulfillment_health_ref=str(data["fulfillment_health_ref"]),
            outcome_label_supply_ref=str(data["outcome_label_supply_ref"]),
            ceiling_utilization_ref=str(data["ceiling_utilization_ref"]),
            alpha_price_window_ref=str(data["alpha_price_window_ref"]),
            current_research_slice_bps=int(data.get("current_research_slice_bps", DEFAULT_RESEARCH_SLICE_BPS)),
            proposed_research_slice_bps=int(data.get("proposed_research_slice_bps", DEFAULT_RESEARCH_SLICE_BPS)),
            max_research_slice_bps=int(data.get("max_research_slice_bps", MAX_RESEARCH_SLICE_BPS)),
            measured_market_yield_ready=bool(data.get("measured_market_yield_ready", False)),
            counterfactual_gate_passed=bool(data.get("counterfactual_gate_passed", False)),
            market_yield_delta_points_per_1000_usd=float(data.get("market_yield_delta_points_per_1000_usd", 0.0)),
            robustness_evidence_ready=bool(data.get("robustness_evidence_ready", False)),
            abuse_monitoring_ready=bool(data.get("abuse_monitoring_ready", False)),
            fulfillment_health_checked=bool(data.get("fulfillment_health_checked", False)),
            outcome_label_supply_checked=bool(data.get("outcome_label_supply_checked", False)),
            ceiling_utilization_published=bool(data.get("ceiling_utilization_published", False)),
            manual_governance_required=bool(data.get("manual_governance_required", True)),
            formula_only_increase=bool(data.get("formula_only_increase", False)),
            pre_scheduled_increase=bool(data.get("pre_scheduled_increase", False)),
            proposal_ready=bool(data.get("proposal_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_review_ref=str(data.get("owner_review_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            budget_mutation_enabled=bool(data.get("budget_mutation_enabled", False)),
            treasury_transfer_enabled=bool(data.get("treasury_transfer_enabled", False)),
            emission_schedule_changed=bool(data.get("emission_schedule_changed", False)),
            payment_distribution_changed=bool(data.get("payment_distribution_changed", False)),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", ResearchSliceProposalState.LOCAL_PROPOSAL_STUB.value)),
            contract_version=str(data.get("contract_version", RESEARCH_SLICE_PROPOSAL_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class ResearchSliceGovernanceDecisionRecord:
    decision_id: str
    proposal_ref: str
    scale_readiness_ref: str
    model_pipeline_exit_gate_ref: str
    owner_approval_ref: str
    governance_meeting_ref: str
    implementation_plan_ref: str
    current_research_slice_bps: int = DEFAULT_RESEARCH_SLICE_BPS
    approved_research_slice_bps: int = DEFAULT_RESEARCH_SLICE_BPS
    max_research_slice_bps: int = MAX_RESEARCH_SLICE_BPS
    raise_approved: bool = False
    proposal_ready: bool = False
    manual_owner_approved: bool = False
    governance_review_complete: bool = False
    post_approval_monitoring_ready: bool = False
    formula_only_decision: bool = False
    pre_scheduled_decision: bool = False
    automatic_execution_enabled: bool = False
    budget_mutation_enabled: bool = False
    treasury_transfer_enabled: bool = False
    emission_schedule_changed: bool = False
    payment_distribution_changed: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    evidence_refs: tuple[str, ...] = ()
    state: str = ResearchSliceDecisionState.LOCAL_DECISION_STUB.value
    contract_version: str = RESEARCH_SLICE_DECISION_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchSliceGovernanceDecisionRecord":
        return cls(
            decision_id=str(data["decision_id"]),
            proposal_ref=str(data["proposal_ref"]),
            scale_readiness_ref=str(data["scale_readiness_ref"]),
            model_pipeline_exit_gate_ref=str(data["model_pipeline_exit_gate_ref"]),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            governance_meeting_ref=str(data["governance_meeting_ref"]),
            implementation_plan_ref=str(data["implementation_plan_ref"]),
            current_research_slice_bps=int(data.get("current_research_slice_bps", DEFAULT_RESEARCH_SLICE_BPS)),
            approved_research_slice_bps=int(data.get("approved_research_slice_bps", DEFAULT_RESEARCH_SLICE_BPS)),
            max_research_slice_bps=int(data.get("max_research_slice_bps", MAX_RESEARCH_SLICE_BPS)),
            raise_approved=bool(data.get("raise_approved", False)),
            proposal_ready=bool(data.get("proposal_ready", False)),
            manual_owner_approved=bool(data.get("manual_owner_approved", False)),
            governance_review_complete=bool(data.get("governance_review_complete", False)),
            post_approval_monitoring_ready=bool(data.get("post_approval_monitoring_ready", False)),
            formula_only_decision=bool(data.get("formula_only_decision", False)),
            pre_scheduled_decision=bool(data.get("pre_scheduled_decision", False)),
            automatic_execution_enabled=bool(data.get("automatic_execution_enabled", False)),
            budget_mutation_enabled=bool(data.get("budget_mutation_enabled", False)),
            treasury_transfer_enabled=bool(data.get("treasury_transfer_enabled", False)),
            emission_schedule_changed=bool(data.get("emission_schedule_changed", False)),
            payment_distribution_changed=bool(data.get("payment_distribution_changed", False)),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", ResearchSliceDecisionState.LOCAL_DECISION_STUB.value)),
            contract_version=str(data.get("contract_version", RESEARCH_SLICE_DECISION_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def validate_research_slice_governance_proposal_record(
    record: ResearchSliceGovernanceProposalRecord | Mapping[str, Any],
    *,
    counterfactual_comparison: Optional[CounterfactualComparisonRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_research_slice_payload_errors(raw)
    if not isinstance(record, ResearchSliceGovernanceProposalRecord):
        try:
            record = ResearchSliceGovernanceProposalRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required research-slice proposal field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid research-slice proposal payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in ResearchSliceProposalState}:
        errors.append(f"unknown research-slice proposal state: {record.state}")
        return errors
    if record.contract_version != RESEARCH_SLICE_PROPOSAL_CONTRACT_VERSION:
        errors.append("contract_version must match research-slice proposal contract")
    if not record.proposal_id.startswith("research_slice_governance:"):
        errors.append("proposal_id must be research_slice_governance:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("quarter_ref", "quarter:"),
            ("model_pipeline_exit_gate_ref", "model_pipeline_exit_gate:"),
            ("market_yield_ref", "market_yield:"),
            ("counterfactual_comparison_ref", "counterfactual_comparison:"),
            ("robustness_evidence_ref", "robustness_evidence:"),
            ("abuse_monitoring_ref", "abuse_monitoring:"),
            ("fulfillment_health_ref", "fulfillment_health:"),
            ("outcome_label_supply_ref", "outcome_label_supply:"),
            ("ceiling_utilization_ref", "ceiling_utilization:"),
            ("alpha_price_window_ref", "alpha_price:"),
        ),
    )
    _append_research_slice_execution_guard_errors(record, "P4.7 research-slice proposal", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "research-slice proposal",
        (
            "research_slice_governance:",
            "market_yield:",
            "counterfactual_comparison:",
            "counterfactual_gate:",
            "robustness_evidence:",
            "abuse_monitoring:",
            "fulfillment_health:",
            "outcome_label_supply:",
            "ceiling_utilization:",
            "owner_review:",
        ),
    )
    if record.current_research_slice_bps < 0:
        errors.append("current_research_slice_bps must be non-negative")
    if record.max_research_slice_bps > MAX_RESEARCH_SLICE_BPS:
        errors.append("max_research_slice_bps must not exceed 10000")
    if record.proposed_research_slice_bps > record.max_research_slice_bps:
        errors.append("proposed_research_slice_bps must not exceed max_research_slice_bps")
    if record.formula_only_increase:
        errors.append("research-slice proposals must not be formula-only increases")
    if record.pre_scheduled_increase:
        errors.append("research-slice proposals must not be pre-scheduled increases")
    if not record.manual_governance_required:
        errors.append("research-slice proposals must require manual governance")
    if counterfactual_comparison is not None:
        if not isinstance(counterfactual_comparison, CounterfactualComparisonRecord):
            counterfactual_comparison = CounterfactualComparisonRecord.from_mapping(counterfactual_comparison)
        comparison_errors = validate_counterfactual_comparison_record(counterfactual_comparison)
        if comparison_errors:
            errors.append("source counterfactual comparison is invalid: " + "; ".join(comparison_errors))
        if record.counterfactual_comparison_ref != counterfactual_comparison.comparison_id:
            errors.append("research-slice proposal counterfactual_comparison_ref mismatch")
        if record.counterfactual_gate_passed != (
            counterfactual_comparison.passed_gate
            and counterfactual_comparison.state == CounterfactualComparisonState.PASSED.value
            and counterfactual_comparison.measured_data_ready
        ):
            errors.append("research-slice proposal counterfactual_gate_passed flag mismatch")
        if not _same_float(
            record.market_yield_delta_points_per_1000_usd,
            counterfactual_comparison.delta_yield_points_per_1000_usd,
        ):
            errors.append("research-slice proposal market_yield_delta_points_per_1000_usd mismatch")
    if record.proposal_ready:
        if counterfactual_comparison is None:
            errors.append("research-slice proposal readiness requires supplied counterfactual comparison")
        if record.uses_local_fixtures:
            errors.append("research-slice proposal readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("research-slice proposal readiness cannot be claimed by a local_only record")
        if record.proposed_research_slice_bps <= record.current_research_slice_bps:
            errors.append("research-slice proposal readiness requires a positive proposed slice increase")
        if record.market_yield_delta_points_per_1000_usd < MIN_MARKET_YIELD_DELTA_POINTS_PER_1000_USD:
            errors.append("research-slice proposal readiness requires non-negative measured market yield delta")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("measured_market_yield_ready", "research-slice proposal readiness requires measured market yield"),
                ("counterfactual_gate_passed", "research-slice proposal readiness requires passed counterfactual gate"),
                ("robustness_evidence_ready", "research-slice proposal readiness requires robustness evidence"),
                ("abuse_monitoring_ready", "research-slice proposal readiness requires abuse monitoring"),
                ("fulfillment_health_checked", "research-slice proposal readiness requires fulfillment health check"),
                ("outcome_label_supply_checked", "research-slice proposal readiness requires outcome-label supply check"),
                ("ceiling_utilization_published", "research-slice proposal readiness requires published ceiling utilization"),
            ),
        )
        if not record.owner_review_ref:
            errors.append("research-slice proposal readiness requires owner_review_ref")
        if not record.evidence_refs:
            errors.append("research-slice proposal readiness requires evidence_refs")
        if record.state != ResearchSliceProposalState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("proposal_ready requires ready_after_measured_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready research-slice proposals must remain local_only")
        if record.state == ResearchSliceProposalState.READY_AFTER_MEASURED_EVIDENCE.value:
            errors.append("ready_after_measured_evidence state requires proposal_ready")
    return errors


def validate_research_slice_governance_decision_record(
    record: ResearchSliceGovernanceDecisionRecord | Mapping[str, Any],
    *,
    proposal: Optional[ResearchSliceGovernanceProposalRecord | Mapping[str, Any]] = None,
    counterfactual_comparison: Optional[CounterfactualComparisonRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_research_slice_payload_errors(raw)
    if not isinstance(record, ResearchSliceGovernanceDecisionRecord):
        try:
            record = ResearchSliceGovernanceDecisionRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required research-slice decision field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid research-slice decision payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in ResearchSliceDecisionState}:
        errors.append(f"unknown research-slice decision state: {record.state}")
        return errors
    if record.contract_version != RESEARCH_SLICE_DECISION_CONTRACT_VERSION:
        errors.append("contract_version must match research-slice decision contract")
    if not record.decision_id.startswith("research_slice_decision:"):
        errors.append("decision_id must be research_slice_decision:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("proposal_ref", "research_slice_governance:"),
            ("scale_readiness_ref", "scale_readiness:"),
            ("model_pipeline_exit_gate_ref", "model_pipeline_exit_gate:"),
            ("governance_meeting_ref", "governance_meeting:"),
            ("implementation_plan_ref", "research_slice_implementation:"),
        ),
    )
    if record.owner_approval_ref and not record.owner_approval_ref.startswith("owner_approval:"):
        errors.append("owner_approval_ref must be owner_approval:-prefixed")
    _append_research_slice_execution_guard_errors(record, "P4.7 research-slice decision", errors)
    if record.automatic_execution_enabled:
        errors.append("P4.7 research-slice decision must not enable automatic execution")
    if record.formula_only_decision:
        errors.append("research-slice decisions must not be formula-only")
    if record.pre_scheduled_decision:
        errors.append("research-slice decisions must not be pre-scheduled")
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "research-slice decision",
        (
            "research_slice_decision:",
            "research_slice_governance:",
            "market_yield:",
            "counterfactual_comparison:",
            "robustness_evidence:",
            "abuse_monitoring:",
            "fulfillment_health:",
            "outcome_label_supply:",
            "ceiling_utilization:",
            "owner_approval:",
            "governance_meeting:",
        ),
    )
    proposal_record = proposal if isinstance(proposal, ResearchSliceGovernanceProposalRecord) else (
        ResearchSliceGovernanceProposalRecord.from_mapping(proposal) if proposal is not None else None
    )
    if proposal_record is not None:
        proposal_errors = validate_research_slice_governance_proposal_record(
            proposal_record,
            counterfactual_comparison=counterfactual_comparison,
        )
        if proposal_errors:
            errors.append("source research-slice proposal is invalid: " + "; ".join(proposal_errors))
        if record.proposal_ref != proposal_record.proposal_id:
            errors.append("research-slice decision proposal_ref mismatch")
        if record.proposal_ready != proposal_record.proposal_ready:
            errors.append("research-slice decision proposal_ready flag mismatch")
        if record.model_pipeline_exit_gate_ref != proposal_record.model_pipeline_exit_gate_ref:
            errors.append("research-slice decision model_pipeline_exit_gate_ref mismatch")
        if record.current_research_slice_bps != proposal_record.current_research_slice_bps:
            errors.append("research-slice decision current_research_slice_bps mismatch")
        if record.approved_research_slice_bps != proposal_record.proposed_research_slice_bps:
            errors.append("research-slice decision approved_research_slice_bps must match proposal")
        if record.max_research_slice_bps != proposal_record.max_research_slice_bps:
            errors.append("research-slice decision max_research_slice_bps mismatch")
    if record.approved_research_slice_bps > record.max_research_slice_bps:
        errors.append("approved_research_slice_bps must not exceed max_research_slice_bps")
    if record.raise_approved:
        if proposal_record is None:
            errors.append("research-slice raise approval requires supplied proposal")
        if proposal_record is not None and not proposal_record.proposal_ready:
            errors.append("research-slice raise approval requires ready proposal")
        if record.uses_local_fixtures:
            errors.append("research-slice raise approval cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("research-slice raise approval cannot be claimed by a local_only record")
        if record.approved_research_slice_bps <= record.current_research_slice_bps:
            errors.append("research-slice raise approval requires a positive slice increase")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("proposal_ready", "research-slice raise approval requires proposal_ready"),
                ("manual_owner_approved", "research-slice raise approval requires manual owner approval"),
                ("governance_review_complete", "research-slice raise approval requires governance review"),
                ("post_approval_monitoring_ready", "research-slice raise approval requires post-approval monitoring"),
            ),
        )
        if not record.owner_approval_ref:
            errors.append("research-slice raise approval requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("research-slice raise approval requires evidence_refs")
        if record.state != ResearchSliceDecisionState.APPROVED_FOR_FUTURE_MANUAL_ROLLOUT.value:
            errors.append("raise_approved requires approved_for_future_manual_rollout state")
    else:
        if not record.local_only:
            errors.append("not-approved research-slice decisions must remain local_only")
        if record.state == ResearchSliceDecisionState.APPROVED_FOR_FUTURE_MANUAL_ROLLOUT.value:
            errors.append("approved_for_future_manual_rollout state requires raise_approved")
    return errors


def verify_research_lab_research_slice_governance(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    scale_summary = verify_scale_foundation()
    counterfactual_summary = verify_research_lab_counterfactual_gate()
    fixture = _load_fixture(Path(fixture_path))
    local_counterfactual = CounterfactualComparisonRecord.from_mapping(fixture["local_counterfactual_comparison"])
    measured_counterfactual = CounterfactualComparisonRecord.from_mapping(fixture["measured_counterfactual_comparison"])
    _assert(
        local_counterfactual.comparison_id == counterfactual_summary["comparison_id"],
        "local counterfactual fixture pins P2.9 comparison",
    )

    local_proposal = ResearchSliceGovernanceProposalRecord.from_mapping(fixture["local_proposal"])
    _assert(
        not validate_research_slice_governance_proposal_record(
            local_proposal,
            counterfactual_comparison=local_counterfactual,
        ),
        "local research-slice proposal validates",
    )
    _assert(not local_proposal.proposal_ready, "local proposal does not claim readiness")

    ready_proposal = ResearchSliceGovernanceProposalRecord.from_mapping(fixture["ready_proposal"])
    _assert(
        not validate_research_slice_governance_proposal_record(
            ready_proposal,
            counterfactual_comparison=measured_counterfactual,
        ),
        "ready research-slice proposal validates",
    )
    _assert(ready_proposal.proposal_ready, "ready proposal claims readiness")

    local_decision = ResearchSliceGovernanceDecisionRecord.from_mapping(fixture["local_decision"])
    _assert(
        not validate_research_slice_governance_decision_record(
            local_decision,
            proposal=local_proposal,
            counterfactual_comparison=local_counterfactual,
        ),
        "local research-slice decision validates",
    )
    _assert(not local_decision.raise_approved, "local decision does not approve raise")

    ready_decision = ResearchSliceGovernanceDecisionRecord.from_mapping(fixture["ready_decision"])
    _assert(
        not validate_research_slice_governance_decision_record(
            ready_decision,
            proposal=ready_proposal,
            counterfactual_comparison=measured_counterfactual,
        ),
        "ready research-slice decision validates",
    )
    _assert(ready_decision.raise_approved, "ready decision approves future manual rollout")
    _assert(not ready_decision.budget_mutation_enabled, "ready decision does not mutate budgets")
    _assert(not ready_decision.emission_schedule_changed, "ready decision does not change emissions")

    for invalid in fixture["invalid_proposals"]:
        base = fixture[str(invalid.get("base", "local_proposal"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        comparison = None
        if not invalid.get("omit_counterfactual"):
            comparison_name = str(invalid.get("counterfactual_base", "local_counterfactual_comparison"))
            comparison = CounterfactualComparisonRecord.from_mapping(fixture[comparison_name])
        errors = validate_research_slice_governance_proposal_record(
            merged,
            counterfactual_comparison=comparison,
        )
        _assert(errors, f"invalid research-slice proposal fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_errors = validate_research_slice_governance_decision_record(
        local_decision,
        proposal=local_proposal,
        counterfactual_comparison=local_counterfactual,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 4 guards block research-slice governance")
    _assert_expected_error(unsafe_errors, fixture["unsafe_scale_guards"])

    for invalid in fixture["invalid_decisions"]:
        base = fixture[str(invalid.get("base", "local_decision"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        proposal = None
        comparison = None
        if not invalid.get("omit_proposal"):
            proposal_name = str(invalid.get("proposal_base", "local_proposal"))
            proposal = ResearchSliceGovernanceProposalRecord.from_mapping(fixture[proposal_name])
        if not invalid.get("omit_counterfactual"):
            comparison_name = str(invalid.get("counterfactual_base", "local_counterfactual_comparison"))
            comparison = CounterfactualComparisonRecord.from_mapping(fixture[comparison_name])
        errors = validate_research_slice_governance_decision_record(
            merged,
            proposal=proposal,
            counterfactual_comparison=comparison,
        )
        _assert(errors, f"invalid research-slice decision fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"research_slice_governance_ready": False}, ScaleGate.RESEARCH_SLICE_GOVERNANCE_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.7 research-slice governance gate remains required by P4.0")

    return {
        "scale_readiness_id": scale_summary["readiness_id"],
        "counterfactual_comparison_id": counterfactual_summary["comparison_id"],
        "proposal_id": local_proposal.proposal_id,
        "proposal_ready": local_proposal.proposal_ready,
        "decision_id": local_decision.decision_id,
        "raise_approved": local_decision.raise_approved,
        "current_research_slice_bps": local_decision.current_research_slice_bps,
        "approved_research_slice_bps": local_decision.approved_research_slice_bps,
        "budget_mutation_enabled": local_decision.budget_mutation_enabled,
        "emission_schedule_changed": local_decision.emission_schedule_changed,
        "ready_control_validates": ready_proposal.proposal_ready and ready_decision.raise_approved,
    }


def _append_scale_guard_errors(
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]],
    errors: list[str],
) -> None:
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))


def _append_prefix_errors(record: Any, errors: list[str], checks: Sequence[tuple[str, str]]) -> None:
    for field, prefix in checks:
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")


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


def _append_research_slice_execution_guard_errors(record: Any, label: str, errors: list[str]) -> None:
    if bool(getattr(record, "budget_mutation_enabled", False)):
        errors.append(f"{label} must not mutate budgets")
    if bool(getattr(record, "treasury_transfer_enabled", False)):
        errors.append(f"{label} must not transfer treasury funds")
    if bool(getattr(record, "emission_schedule_changed", False)):
        errors.append(f"{label} must not change emission schedules")
    if bool(getattr(record, "payment_distribution_changed", False)):
        errors.append(f"{label} must not change payment distributions")
    if bool(getattr(record, "production_writes", False)):
        errors.append(f"{label} must not enable production writes")
    if bool(getattr(record, "supabase_writes", False)):
        errors.append(f"{label} must not enable Supabase writes")
    if bool(getattr(record, "public_workflows", False)):
        errors.append(f"{label} must not enable public workflows")


def _protected_research_slice_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_research_slice_material(record))
    if not found:
        return []
    return ["P4.7 research-slice governance payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_research_slice_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_RESEARCH_SLICE_KEYS and not key_text.endswith(
                ("_ref", "_refs", "_hash", "_hashes")
            ):
                found.add(key_path)
            found.update(_find_protected_research_slice_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(_find_protected_research_slice_material(nested, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_RESEARCH_SLICE_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _same_float(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance


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
