"""Phase 1 baseline-arm operations records.

P1.9 defines the local operating posture for the market-operation baseline
arm: owner-set absolute budget, allocator-directed house accounting, matched
comparison placeholders, market-monitoring placeholders, and publication
stubs. It is intentionally inert: it does not spend budget, schedule baseline
jobs, read live market stats, write production data, or publish live claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    ReleasePolicyRecord,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    validate_release_policy,
)
from .research_map import verify_research_lab_research_map


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "baseline_arm_ops_fixtures.json"

REFERENCE_BASELINE_MODEL_ID = "reference:qualification_model:v1"
RESEARCH_LAB_ARM_B_MODEL_ID = "research_lab:qualification_model:arm_b:v1"
BASELINE_DAILY_BUDGET_MIN_CENTS = 20_000
BASELINE_DAILY_BUDGET_MAX_CENTS = 50_000
BASELINE_SOURCE_TABLE = "qualification_baselines"

PROTECTED_PUBLICATION_KEYS = {
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

PROTECTED_PUBLICATION_MARKERS = (
    "sealed judge prompt",
    "sealed eval",
    "raw customer",
    "private customer",
    "live champion prompt",
    "live champion code",
    "live champion weights",
)


class BaselineArmOpsState(str, Enum):
    LOCAL_POLICY_ONLY = "local_policy_only"
    AWAITING_LAB_SQL_TESTING = "awaiting_lab_sql_testing"
    BLOCKED = "blocked"


class BaselineComparisonState(str, Enum):
    AWAITING_DUAL_ARM_MEASUREMENTS = "awaiting_dual_arm_measurements"
    READY_FOR_LOCAL_REVIEW = "ready_for_local_review"
    BLOCKED = "blocked"


class BaselineMonitoringState(str, Enum):
    LOCAL_PLACEHOLDER_ONLY = "local_placeholder_only"
    AWAITING_MARKET_ACTIVITY = "awaiting_market_activity"
    BLOCKED = "blocked"


class BaselinePublicationState(str, Enum):
    LOCAL_STUB_ONLY = "local_stub_only"
    READY_AFTER_LAB_TESTING = "ready_after_lab_testing"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BaselineArmOperatingPolicyRecord:
    policy_id: str
    island: str
    house_arm_ref: str
    reference_model_id: str
    allocator_directed_model_id: str
    daily_budget_min_cents: int
    daily_budget_max_cents: int
    budget_currency: str
    island_scope_refs: tuple[str, ...]
    exploration_objective_refs: tuple[str, ...]
    comparison_methodology_ref: str
    owner_set_absolute_budget: bool = True
    market_volume_dependent: bool = False
    matched_budget_required: bool = True
    quarterly_publication_required: bool = True
    lab_sql_required_before_operation: bool = True
    production_sql_applied: bool = False
    lab_testing_complete: bool = False
    spend_enabled: bool = False
    scheduler_enabled: bool = False
    production_job_enabled: bool = False
    payment_enabled: bool = False
    grant_eligible: bool = False
    house_improvement_no_grant: bool = True
    local_only: bool = True
    state: str = BaselineArmOpsState.LOCAL_POLICY_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineArmOperatingPolicyRecord":
        return cls(
            policy_id=str(data["policy_id"]),
            island=str(data["island"]),
            house_arm_ref=str(data["house_arm_ref"]),
            reference_model_id=str(data["reference_model_id"]),
            allocator_directed_model_id=str(data["allocator_directed_model_id"]),
            daily_budget_min_cents=int(data["daily_budget_min_cents"]),
            daily_budget_max_cents=int(data["daily_budget_max_cents"]),
            budget_currency=str(data.get("budget_currency", "USD")),
            island_scope_refs=tuple(str(item) for item in data.get("island_scope_refs", [])),
            exploration_objective_refs=tuple(str(item) for item in data.get("exploration_objective_refs", [])),
            comparison_methodology_ref=str(data["comparison_methodology_ref"]),
            owner_set_absolute_budget=bool(data.get("owner_set_absolute_budget", True)),
            market_volume_dependent=bool(data.get("market_volume_dependent", False)),
            matched_budget_required=bool(data.get("matched_budget_required", True)),
            quarterly_publication_required=bool(data.get("quarterly_publication_required", True)),
            lab_sql_required_before_operation=bool(data.get("lab_sql_required_before_operation", True)),
            production_sql_applied=bool(data.get("production_sql_applied", False)),
            lab_testing_complete=bool(data.get("lab_testing_complete", False)),
            spend_enabled=bool(data.get("spend_enabled", False)),
            scheduler_enabled=bool(data.get("scheduler_enabled", False)),
            production_job_enabled=bool(data.get("production_job_enabled", False)),
            payment_enabled=bool(data.get("payment_enabled", False)),
            grant_eligible=bool(data.get("grant_eligible", False)),
            house_improvement_no_grant=bool(data.get("house_improvement_no_grant", True)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", BaselineArmOpsState.LOCAL_POLICY_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["island_scope_refs"] = list(self.island_scope_refs)
        data["exploration_objective_refs"] = list(self.exploration_objective_refs)
        return data


@dataclass(frozen=True)
class BaselineComparisonProjectionRecord:
    projection_id: str
    policy_ref: str
    comparison_window: str
    source_table: str
    source_dual_arm_baseline_refs: tuple[str, ...]
    reference_model_id: str
    allocator_directed_model_id: str
    matched_budget_cents: int
    reference_budget_cents: int
    allocator_budget_cents: int
    paired_day_count: int
    reference_verified_lcb_pts_per_1000_usd: Optional[float] = None
    allocator_verified_lcb_pts_per_1000_usd: Optional[float] = None
    novelty_weighted_corpus_growth_ref: str = ""
    measured_from_dual_arm_sql_rows: bool = False
    lab_sql_smoke_tested: bool = False
    lab_testing_complete: bool = False
    production_reads_enabled: bool = False
    live_market_stats_used: bool = False
    placeholder_only: bool = True
    publication_ready: bool = False
    local_only: bool = True
    state: str = BaselineComparisonState.AWAITING_DUAL_ARM_MEASUREMENTS.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineComparisonProjectionRecord":
        return cls(
            projection_id=str(data["projection_id"]),
            policy_ref=str(data["policy_ref"]),
            comparison_window=str(data["comparison_window"]),
            source_table=str(data.get("source_table", BASELINE_SOURCE_TABLE)),
            source_dual_arm_baseline_refs=tuple(str(item) for item in data.get("source_dual_arm_baseline_refs", [])),
            reference_model_id=str(data["reference_model_id"]),
            allocator_directed_model_id=str(data["allocator_directed_model_id"]),
            matched_budget_cents=int(data.get("matched_budget_cents", 0)),
            reference_budget_cents=int(data.get("reference_budget_cents", 0)),
            allocator_budget_cents=int(data.get("allocator_budget_cents", 0)),
            paired_day_count=int(data.get("paired_day_count", 0)),
            reference_verified_lcb_pts_per_1000_usd=_optional_float(data.get("reference_verified_lcb_pts_per_1000_usd")),
            allocator_verified_lcb_pts_per_1000_usd=_optional_float(
                data.get("allocator_verified_lcb_pts_per_1000_usd")
            ),
            novelty_weighted_corpus_growth_ref=str(data.get("novelty_weighted_corpus_growth_ref", "")),
            measured_from_dual_arm_sql_rows=bool(data.get("measured_from_dual_arm_sql_rows", False)),
            lab_sql_smoke_tested=bool(data.get("lab_sql_smoke_tested", False)),
            lab_testing_complete=bool(data.get("lab_testing_complete", False)),
            production_reads_enabled=bool(data.get("production_reads_enabled", False)),
            live_market_stats_used=bool(data.get("live_market_stats_used", False)),
            placeholder_only=bool(data.get("placeholder_only", True)),
            publication_ready=bool(data.get("publication_ready", False)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", BaselineComparisonState.AWAITING_DUAL_ARM_MEASUREMENTS.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_dual_arm_baseline_refs"] = list(self.source_dual_arm_baseline_refs)
        return data


@dataclass(frozen=True)
class BaselineMarketMonitoringRecord:
    monitor_id: str
    policy_ref: str
    period_start: str
    period_end: str
    hosted_run_market_ref: str
    source_refs: tuple[str, ...]
    hosted_run_count: int = 0
    miner_directed_spend_cents: int = 0
    allocator_directed_spend_cents: int = 0
    uses_live_market_activity: bool = False
    production_reads_enabled: bool = False
    production_writes_enabled: bool = False
    local_only: bool = True
    placeholder_only: bool = True
    state: str = BaselineMonitoringState.LOCAL_PLACEHOLDER_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineMarketMonitoringRecord":
        return cls(
            monitor_id=str(data["monitor_id"]),
            policy_ref=str(data["policy_ref"]),
            period_start=str(data["period_start"]),
            period_end=str(data["period_end"]),
            hosted_run_market_ref=str(data["hosted_run_market_ref"]),
            source_refs=tuple(str(item) for item in data.get("source_refs", [])),
            hosted_run_count=int(data.get("hosted_run_count", 0)),
            miner_directed_spend_cents=int(data.get("miner_directed_spend_cents", 0)),
            allocator_directed_spend_cents=int(data.get("allocator_directed_spend_cents", 0)),
            uses_live_market_activity=bool(data.get("uses_live_market_activity", False)),
            production_reads_enabled=bool(data.get("production_reads_enabled", False)),
            production_writes_enabled=bool(data.get("production_writes_enabled", False)),
            local_only=bool(data.get("local_only", True)),
            placeholder_only=bool(data.get("placeholder_only", True)),
            state=str(data.get("state", BaselineMonitoringState.LOCAL_PLACEHOLDER_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = list(self.source_refs)
        return data


@dataclass(frozen=True)
class BaselineComparisonPublicationStub:
    publication_id: str
    policy_ref: str
    projection_ref: str
    monitor_ref: str
    title: str
    absolute_budget_statement: str
    methodology_summary: str
    shared_prior_caveat: str
    matched_budget_caveat: str
    publication_state: str = BaselinePublicationState.LOCAL_STUB_ONLY.value
    local_only: bool = True
    production_publish_enabled: bool = False
    claims_live_market_operation: bool = False
    claims_baseline_arm_live: bool = False
    claims_comparison_published: bool = False
    claims_counterfactual_gate_satisfied: bool = False
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineComparisonPublicationStub":
        return cls(
            publication_id=str(data["publication_id"]),
            policy_ref=str(data["policy_ref"]),
            projection_ref=str(data["projection_ref"]),
            monitor_ref=str(data["monitor_ref"]),
            title=str(data["title"]),
            absolute_budget_statement=str(data["absolute_budget_statement"]),
            methodology_summary=str(data["methodology_summary"]),
            shared_prior_caveat=str(data["shared_prior_caveat"]),
            matched_budget_caveat=str(data["matched_budget_caveat"]),
            publication_state=str(data.get("publication_state", BaselinePublicationState.LOCAL_STUB_ONLY.value)),
            local_only=bool(data.get("local_only", True)),
            production_publish_enabled=bool(data.get("production_publish_enabled", False)),
            claims_live_market_operation=bool(data.get("claims_live_market_operation", False)),
            claims_baseline_arm_live=bool(data.get("claims_baseline_arm_live", False)),
            claims_comparison_published=bool(data.get("claims_comparison_published", False)),
            claims_counterfactual_gate_satisfied=bool(data.get("claims_counterfactual_gate_satisfied", False)),
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


def validate_baseline_arm_operating_policy_record(
    record: BaselineArmOperatingPolicyRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    if not isinstance(record, BaselineArmOperatingPolicyRecord):
        record = BaselineArmOperatingPolicyRecord.from_mapping(record)
    errors: list[str] = []
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in BaselineArmOpsState}:
        errors.append(f"unknown baseline arm ops state: {record.state}")
    if record.state != BaselineArmOpsState.LOCAL_POLICY_ONLY.value:
        errors.append("P1.9 policy must remain local_policy_only before lab testing")
    if not record.policy_id:
        errors.append("baseline arm policy requires policy_id")
    if not record.island:
        errors.append("baseline arm policy requires island")
    if not record.house_arm_ref.startswith("baseline_house_arm:"):
        errors.append("house_arm_ref must identify the house baseline arm")
    if record.reference_model_id != REFERENCE_BASELINE_MODEL_ID:
        errors.append(f"reference_model_id must be {REFERENCE_BASELINE_MODEL_ID}")
    if record.allocator_directed_model_id == record.reference_model_id:
        errors.append("allocator_directed_model_id must be distinct from reference_model_id")
    if not record.allocator_directed_model_id.startswith("research_lab:"):
        errors.append("allocator_directed_model_id must be a research_lab baseline arm")
    if record.budget_currency != "USD":
        errors.append("baseline arm budget_currency must be USD")
    if record.daily_budget_min_cents < BASELINE_DAILY_BUDGET_MIN_CENTS:
        errors.append("daily_budget_min_cents must be at least $200/day")
    if record.daily_budget_max_cents > BASELINE_DAILY_BUDGET_MAX_CENTS:
        errors.append("daily_budget_max_cents must be at most $500/day")
    if record.daily_budget_min_cents > record.daily_budget_max_cents:
        errors.append("daily_budget_min_cents must not exceed daily_budget_max_cents")
    if not record.owner_set_absolute_budget:
        errors.append("baseline arm must use an owner-set absolute budget")
    if record.market_volume_dependent:
        errors.append("baseline arm budget must not depend on market volume")
    if not record.island_scope_refs:
        errors.append("baseline arm policy requires island_scope_refs")
    if not record.exploration_objective_refs:
        errors.append("baseline arm policy requires exploration_objective_refs")
    if not record.comparison_methodology_ref:
        errors.append("baseline arm policy requires comparison_methodology_ref")
    if not record.matched_budget_required:
        errors.append("baseline arm comparison must require matched budgets")
    if not record.quarterly_publication_required:
        errors.append("baseline arm comparison must require quarterly publication")
    if not record.lab_sql_required_before_operation:
        errors.append("baseline arm operation must require SQL/lab testing first")
    if record.production_sql_applied:
        errors.append("P1.9 local policy must not claim production SQL is applied")
    if record.lab_testing_complete:
        errors.append("P1.9 local policy must not claim lab testing is complete")
    if record.spend_enabled:
        errors.append("P1.9 must not enable budget spend")
    if record.scheduler_enabled:
        errors.append("P1.9 must not enable baseline job scheduling")
    if record.production_job_enabled:
        errors.append("P1.9 must not enable production baseline jobs")
    if record.payment_enabled:
        errors.append("P1.9 must not enable baseline arm payments")
    if record.grant_eligible:
        errors.append("house baseline arm must not be grant eligible")
    if not record.house_improvement_no_grant:
        errors.append("house improvements must take no grant")
    if not record.local_only:
        errors.append("P1.9 policy must remain local_only")
    return errors


def validate_baseline_comparison_projection_record(
    record: BaselineComparisonProjectionRecord | Mapping[str, Any],
) -> list[str]:
    if not isinstance(record, BaselineComparisonProjectionRecord):
        record = BaselineComparisonProjectionRecord.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in BaselineComparisonState}:
        errors.append(f"unknown baseline comparison state: {record.state}")
    if not record.projection_id:
        errors.append("baseline comparison projection requires projection_id")
    if not record.policy_ref:
        errors.append("baseline comparison projection requires policy_ref")
    if record.source_table != BASELINE_SOURCE_TABLE:
        errors.append(f"baseline comparison source_table must be {BASELINE_SOURCE_TABLE}")
    if not record.source_dual_arm_baseline_refs:
        errors.append("baseline comparison requires dual-arm baseline source refs")
    bad_refs = [ref for ref in record.source_dual_arm_baseline_refs if not ref.startswith(f"{BASELINE_SOURCE_TABLE}:")]
    if bad_refs:
        errors.append("baseline comparison source refs must point at qualification_baselines: " + ", ".join(bad_refs))
    if record.reference_model_id != REFERENCE_BASELINE_MODEL_ID:
        errors.append(f"comparison reference_model_id must be {REFERENCE_BASELINE_MODEL_ID}")
    if record.allocator_directed_model_id == record.reference_model_id:
        errors.append("comparison allocator_directed_model_id must be distinct from reference_model_id")
    if record.matched_budget_cents < 0 or record.reference_budget_cents < 0 or record.allocator_budget_cents < 0:
        errors.append("comparison budgets must be non-negative")
    if record.reference_budget_cents != record.allocator_budget_cents:
        errors.append("comparison must use matched budgets")
    if record.matched_budget_cents not in {record.reference_budget_cents, record.allocator_budget_cents}:
        errors.append("matched_budget_cents must match both arm budgets")
    if record.paired_day_count < 0:
        errors.append("paired_day_count must be non-negative")
    if record.production_reads_enabled:
        errors.append("P1.9 comparison must not enable production reads")
    if record.live_market_stats_used:
        errors.append("P1.9 comparison must not use live market stats")
    if not record.local_only:
        errors.append("P1.9 comparison must remain local_only")
    if record.placeholder_only:
        if record.measured_from_dual_arm_sql_rows:
            errors.append("placeholder comparison must not claim measured dual-arm SQL rows")
        if record.lab_sql_smoke_tested:
            errors.append("placeholder comparison must not claim SQL smoke tests passed")
        if record.lab_testing_complete:
            errors.append("placeholder comparison must not claim lab testing is complete")
        if record.publication_ready:
            errors.append("placeholder comparison must not be publication_ready")
        if record.paired_day_count != 0:
            errors.append("placeholder comparison must not claim paired days")
    else:
        if not record.measured_from_dual_arm_sql_rows:
            errors.append("non-placeholder comparison must use measured dual-arm SQL rows")
        if not record.lab_sql_smoke_tested:
            errors.append("non-placeholder comparison requires lab SQL smoke test")
        if not record.lab_testing_complete:
            errors.append("non-placeholder comparison requires lab testing complete")
        if record.paired_day_count < 20:
            errors.append("non-placeholder comparison requires at least 20 paired days")
    if record.publication_ready and not (
        record.measured_from_dual_arm_sql_rows
        and record.lab_sql_smoke_tested
        and record.lab_testing_complete
        and record.paired_day_count >= 20
    ):
        errors.append("publication_ready requires measured SQL rows, lab tests, and at least 20 paired days")
    return errors


def validate_baseline_market_monitoring_record(
    record: BaselineMarketMonitoringRecord | Mapping[str, Any],
) -> list[str]:
    if not isinstance(record, BaselineMarketMonitoringRecord):
        record = BaselineMarketMonitoringRecord.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in BaselineMonitoringState}:
        errors.append(f"unknown baseline monitoring state: {record.state}")
    if not record.monitor_id:
        errors.append("baseline market monitor requires monitor_id")
    if not record.policy_ref:
        errors.append("baseline market monitor requires policy_ref")
    if not record.hosted_run_market_ref:
        errors.append("baseline market monitor requires hosted_run_market_ref")
    if not record.source_refs:
        errors.append("baseline market monitor requires source_refs")
    if record.hosted_run_count < 0:
        errors.append("hosted_run_count must be non-negative")
    if record.miner_directed_spend_cents < 0 or record.allocator_directed_spend_cents < 0:
        errors.append("market monitor spend values must be non-negative")
    if record.uses_live_market_activity:
        errors.append("P1.9 monitor must not use live market activity")
    if record.production_reads_enabled:
        errors.append("P1.9 monitor must not enable production reads")
    if record.production_writes_enabled:
        errors.append("P1.9 monitor must not enable production writes")
    if not record.local_only:
        errors.append("P1.9 monitor must remain local_only")
    if record.placeholder_only:
        if record.hosted_run_count != 0:
            errors.append("placeholder monitor must not claim hosted-run counts")
        if record.miner_directed_spend_cents != 0 or record.allocator_directed_spend_cents != 0:
            errors.append("placeholder monitor must not claim market spend")
    return errors


def validate_baseline_comparison_publication_stub(
    record: BaselineComparisonPublicationStub | Mapping[str, Any],
) -> list[str]:
    raw_errors = _protected_publication_errors(record)
    if not isinstance(record, BaselineComparisonPublicationStub):
        record = BaselineComparisonPublicationStub.from_mapping(record)
    errors: list[str] = list(raw_errors)
    if record.publication_state not in {state.value for state in BaselinePublicationState}:
        errors.append(f"unknown baseline publication state: {record.publication_state}")
    if record.publication_state != BaselinePublicationState.LOCAL_STUB_ONLY.value:
        errors.append("P1.9 publication must remain a local_stub_only before lab testing")
    for field in (
        "publication_id",
        "policy_ref",
        "projection_ref",
        "monitor_ref",
        "title",
        "absolute_budget_statement",
        "methodology_summary",
        "shared_prior_caveat",
        "matched_budget_caveat",
    ):
        if not getattr(record, field):
            errors.append(f"publication stub requires {field}")
    lowered_budget = record.absolute_budget_statement.lower()
    if "absolute" not in lowered_budget or "200" not in lowered_budget or "500" not in lowered_budget:
        errors.append("publication stub must state the absolute $200-$500/day baseline budget posture")
    if "matched" not in record.matched_budget_caveat.lower():
        errors.append("publication stub must retain the matched-budget caveat")
    if "shared" not in record.shared_prior_caveat.lower():
        errors.append("publication stub must retain the shared-prior caveat")
    if not record.local_only:
        errors.append("P1.9 publication stub must remain local_only")
    if record.production_publish_enabled:
        errors.append("P1.9 must not enable production publication")
    if record.claims_live_market_operation:
        errors.append("publication stub must not claim live market operation")
    if record.claims_baseline_arm_live:
        errors.append("publication stub must not claim the baseline arm is live")
    if record.claims_comparison_published:
        errors.append("publication stub must not claim comparison results are published")
    if record.claims_counterfactual_gate_satisfied:
        errors.append("publication stub must not claim the counterfactual gate is satisfied")
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
                reason="P1.9 baseline comparison publication stubs are sanitized local artifacts",
            )
        )
    )
    return errors


def verify_research_lab_baseline_arm_ops(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    map_summary = verify_research_lab_research_map()
    fixture = _load_fixture(Path(fixture_path))

    policy = BaselineArmOperatingPolicyRecord.from_mapping(fixture["operating_policy"])
    _assert(not validate_baseline_arm_operating_policy_record(policy), "baseline arm operating policy validates")
    for record in fixture["invalid_operating_policies"]:
        errors = validate_baseline_arm_operating_policy_record(record)
        _assert(errors, f"invalid operating policy fails: {record['policy_id']}")
        _assert_expected_error(errors, record)
    unsafe_policy_errors = validate_baseline_arm_operating_policy_record(
        policy,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_policy_errors, "unsafe workflow guards block baseline arm policy")

    comparison = BaselineComparisonProjectionRecord.from_mapping(fixture["comparison_projection"])
    _assert(not validate_baseline_comparison_projection_record(comparison), "baseline comparison projection validates")
    for record in fixture["invalid_comparison_projections"]:
        errors = validate_baseline_comparison_projection_record(record)
        _assert(errors, f"invalid comparison projection fails: {record['projection_id']}")
        _assert_expected_error(errors, record)

    monitor = BaselineMarketMonitoringRecord.from_mapping(fixture["market_monitor"])
    _assert(not validate_baseline_market_monitoring_record(monitor), "baseline market monitor validates")
    for record in fixture["invalid_market_monitors"]:
        errors = validate_baseline_market_monitoring_record(record)
        _assert(errors, f"invalid market monitor fails: {record['monitor_id']}")
        _assert_expected_error(errors, record)

    publication = BaselineComparisonPublicationStub.from_mapping(fixture["publication_stub"])
    _assert(not validate_baseline_comparison_publication_stub(publication), "baseline publication stub validates")
    for record in fixture["invalid_publication_stubs"]:
        errors = validate_baseline_comparison_publication_stub(record)
        _assert(errors, f"invalid publication stub fails: {record['publication_id']}")
        _assert_expected_error(errors, record)

    return {
        "map_cells": map_summary["cell_count"],
        "policy_id": policy.policy_id,
        "daily_budget_min_cents": policy.daily_budget_min_cents,
        "daily_budget_max_cents": policy.daily_budget_max_cents,
        "comparison_state": comparison.state,
        "monitor_state": monitor.state,
        "publication_state": publication.publication_state,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _protected_publication_errors(record: Any) -> list[str]:
    payload = record.to_dict() if hasattr(record, "to_dict") else record
    found = sorted(_find_protected_publication_material(payload))
    if not found:
        return []
    return ["baseline publication payload contains protected or raw material keys/markers: " + ", ".join(found)]


def _find_protected_publication_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_PUBLICATION_KEYS:
                found.add(key_path)
            found.update(_find_protected_publication_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_publication_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_PUBLICATION_MARKERS:
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
