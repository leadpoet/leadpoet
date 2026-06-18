"""Phase 4.6 Agent-track GA gate contracts.

P4.6 defines local record shapes for future agent-track general availability.
It does not create accounts, issue credentials, enable public agent access,
open Workspace API production workflows, write SQL, or change miner access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
from .workspace_api_beta import (
    FIXTURE_PATH as WORKSPACE_API_BETA_FIXTURE_PATH,
    MAX_BETA_COHORT_SIZE,
    PROTECTED_WORKSPACE_API_KEYS,
    PROTECTED_WORKSPACE_API_MARKERS,
    WorkspaceAPIBetaEnablementRecord,
    WorkspaceAPIEntropyLedgerRecord,
    WorkspaceAPIMethodContractRecord,
    WorkspaceAPIMinerAllowlistRecord,
    WorkspaceAPISecurityReviewRecord,
    validate_workspace_api_beta_enablement_record,
    verify_research_lab_workspace_api_beta,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "agent_track_ga_fixtures.json"

AGENT_TRACK_GA_CRITERIA_CONTRACT_VERSION = "agent_track_ga_criteria:v1:local_contract"
AGENT_TRACK_GA_CANDIDATE_CONTRACT_VERSION = "agent_track_ga_candidate:v1:local_contract"
AGENT_TRACK_GA_READINESS_CONTRACT_VERSION = "agent_track_ga_readiness:v1:local_contract"

MIN_AGENT_GA_STAKE_USD = 500
MIN_AGENT_GA_CROWNED_PATCHES = 1
MIN_AGENT_GA_FUNDED_LOOPS = 25
MIN_AGENT_GA_CLEAN_RECEIPTS = 25

PROTECTED_AGENT_TRACK_GA_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SCALE_KEYS)
    | set(PROTECTED_WORKSPACE_API_KEYS)
    | {
        "agent_private_key",
        "api_token",
        "credential_secret",
        "miner_wallet_private_key",
        "raw_agent_prompt",
        "raw_workspace_payload",
        "session_token",
        "slashing_admin_key",
        "workspace_access_token",
    }
)

PROTECTED_AGENT_TRACK_GA_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SCALE_MARKERS)
        | set(PROTECTED_WORKSPACE_API_MARKERS)
        | {
            "agent private key",
            "api token",
            "credential secret",
            "miner wallet private key",
            "raw agent prompt",
            "raw workspace payload",
            "session token",
            "slashing admin key",
            "workspace access token",
        }
    )
)


class AgentTrackGACriteriaState(str, Enum):
    LOCAL_CRITERIA_STUB = "local_criteria_stub"
    READY_AFTER_OWNER_POLICY = "ready_after_owner_policy"
    BLOCKED = "blocked"


class AgentTrackGACandidateState(str, Enum):
    LOCAL_CANDIDATE_STUB = "local_candidate_stub"
    ELIGIBLE_AFTER_TRACK_RECORD = "eligible_after_track_record"
    BLOCKED = "blocked"


class AgentTrackGAReadinessState(str, Enum):
    LOCAL_GA_STUB = "local_ga_stub"
    READY_AFTER_OWNER_REVIEW = "ready_after_owner_review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AgentTrackGACriteriaRecord:
    criteria_id: str
    workspace_beta_ref: str
    security_policy_ref: str
    entropy_policy_ref: str
    abuse_monitoring_policy_ref: str
    legal_terms_ref: str
    slashing_policy_ref: str
    stake_threshold_usd: int = MIN_AGENT_GA_STAKE_USD
    min_crowned_patch_count: int = MIN_AGENT_GA_CROWNED_PATCHES
    min_funded_loop_count: int = MIN_AGENT_GA_FUNDED_LOOPS
    min_clean_receipt_count: int = MIN_AGENT_GA_CLEAN_RECEIPTS
    require_clean_receipts: bool = True
    require_entropy_accounting_live: bool = True
    require_security_review_passed: bool = True
    require_abuse_monitoring: bool = True
    require_legal_terms: bool = True
    require_slashing_terms: bool = True
    criteria_approved: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = AgentTrackGACriteriaState.LOCAL_CRITERIA_STUB.value
    contract_version: str = AGENT_TRACK_GA_CRITERIA_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentTrackGACriteriaRecord":
        return cls(
            criteria_id=str(data["criteria_id"]),
            workspace_beta_ref=str(data["workspace_beta_ref"]),
            security_policy_ref=str(data["security_policy_ref"]),
            entropy_policy_ref=str(data["entropy_policy_ref"]),
            abuse_monitoring_policy_ref=str(data["abuse_monitoring_policy_ref"]),
            legal_terms_ref=str(data["legal_terms_ref"]),
            slashing_policy_ref=str(data["slashing_policy_ref"]),
            stake_threshold_usd=int(data.get("stake_threshold_usd", MIN_AGENT_GA_STAKE_USD)),
            min_crowned_patch_count=int(data.get("min_crowned_patch_count", MIN_AGENT_GA_CROWNED_PATCHES)),
            min_funded_loop_count=int(data.get("min_funded_loop_count", MIN_AGENT_GA_FUNDED_LOOPS)),
            min_clean_receipt_count=int(data.get("min_clean_receipt_count", MIN_AGENT_GA_CLEAN_RECEIPTS)),
            require_clean_receipts=bool(data.get("require_clean_receipts", True)),
            require_entropy_accounting_live=bool(data.get("require_entropy_accounting_live", True)),
            require_security_review_passed=bool(data.get("require_security_review_passed", True)),
            require_abuse_monitoring=bool(data.get("require_abuse_monitoring", True)),
            require_legal_terms=bool(data.get("require_legal_terms", True)),
            require_slashing_terms=bool(data.get("require_slashing_terms", True)),
            criteria_approved=bool(data.get("criteria_approved", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", AgentTrackGACriteriaState.LOCAL_CRITERIA_STUB.value)),
            contract_version=str(data.get("contract_version", AGENT_TRACK_GA_CRITERIA_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class AgentTrackGACandidateRecord:
    eligibility_id: str
    miner_ref: str
    criteria_ref: str
    workspace_beta_ref: str
    stake_ref: str
    track_record_ref: str
    clean_receipt_audit_ref: str
    legal_terms_ref: str
    slashing_terms_ref: str
    entropy_ledger_ref: str
    abuse_monitoring_ref: str
    security_review_ref: str
    stake_usd: int = 0
    crowned_patch_count: int = 0
    funded_loop_count: int = 0
    clean_receipt_count: int = 0
    tos_violation_count: int = 0
    entropy_accounting_live: bool = False
    entropy_budget_enforced: bool = False
    abuse_monitoring_clean: bool = False
    legal_terms_accepted: bool = False
    slashing_terms_accepted: bool = False
    security_review_passed: bool = False
    ga_eligible: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = AgentTrackGACandidateState.LOCAL_CANDIDATE_STUB.value
    contract_version: str = AGENT_TRACK_GA_CANDIDATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentTrackGACandidateRecord":
        return cls(
            eligibility_id=str(data["eligibility_id"]),
            miner_ref=str(data["miner_ref"]),
            criteria_ref=str(data["criteria_ref"]),
            workspace_beta_ref=str(data["workspace_beta_ref"]),
            stake_ref=str(data["stake_ref"]),
            track_record_ref=str(data["track_record_ref"]),
            clean_receipt_audit_ref=str(data["clean_receipt_audit_ref"]),
            legal_terms_ref=str(data["legal_terms_ref"]),
            slashing_terms_ref=str(data["slashing_terms_ref"]),
            entropy_ledger_ref=str(data["entropy_ledger_ref"]),
            abuse_monitoring_ref=str(data["abuse_monitoring_ref"]),
            security_review_ref=str(data["security_review_ref"]),
            stake_usd=int(data.get("stake_usd", 0)),
            crowned_patch_count=int(data.get("crowned_patch_count", 0)),
            funded_loop_count=int(data.get("funded_loop_count", 0)),
            clean_receipt_count=int(data.get("clean_receipt_count", 0)),
            tos_violation_count=int(data.get("tos_violation_count", 0)),
            entropy_accounting_live=bool(data.get("entropy_accounting_live", False)),
            entropy_budget_enforced=bool(data.get("entropy_budget_enforced", False)),
            abuse_monitoring_clean=bool(data.get("abuse_monitoring_clean", False)),
            legal_terms_accepted=bool(data.get("legal_terms_accepted", False)),
            slashing_terms_accepted=bool(data.get("slashing_terms_accepted", False)),
            security_review_passed=bool(data.get("security_review_passed", False)),
            ga_eligible=bool(data.get("ga_eligible", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", AgentTrackGACandidateState.LOCAL_CANDIDATE_STUB.value)),
            contract_version=str(data.get("contract_version", AGENT_TRACK_GA_CANDIDATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class AgentTrackGAReadinessRecord:
    ga_readiness_id: str
    criteria_ref: str
    candidate_ref: str
    workspace_beta_ref: str
    scale_readiness_ref: str
    entropy_ledger_ref: str
    security_review_ref: str
    abuse_monitoring_ref: str
    owner_review_ref: str
    allowed_miner_refs: tuple[str, ...]
    ga_claimed: bool = False
    criteria_approved: bool = False
    candidate_eligible: bool = False
    workspace_beta_valid: bool = False
    entropy_accounting_live: bool = False
    security_review_passed: bool = False
    abuse_monitoring_ready: bool = False
    owner_approved: bool = False
    public_agent_track_enabled: bool = False
    production_workflows_enabled: bool = False
    credential_material_issued: bool = False
    account_creation_enabled: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = AgentTrackGAReadinessState.LOCAL_GA_STUB.value
    contract_version: str = AGENT_TRACK_GA_READINESS_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentTrackGAReadinessRecord":
        return cls(
            ga_readiness_id=str(data["ga_readiness_id"]),
            criteria_ref=str(data["criteria_ref"]),
            candidate_ref=str(data["candidate_ref"]),
            workspace_beta_ref=str(data["workspace_beta_ref"]),
            scale_readiness_ref=str(data["scale_readiness_ref"]),
            entropy_ledger_ref=str(data["entropy_ledger_ref"]),
            security_review_ref=str(data["security_review_ref"]),
            abuse_monitoring_ref=str(data["abuse_monitoring_ref"]),
            owner_review_ref=str(data["owner_review_ref"]),
            allowed_miner_refs=tuple(str(item) for item in data.get("allowed_miner_refs", [])),
            ga_claimed=bool(data.get("ga_claimed", False)),
            criteria_approved=bool(data.get("criteria_approved", False)),
            candidate_eligible=bool(data.get("candidate_eligible", False)),
            workspace_beta_valid=bool(data.get("workspace_beta_valid", False)),
            entropy_accounting_live=bool(data.get("entropy_accounting_live", False)),
            security_review_passed=bool(data.get("security_review_passed", False)),
            abuse_monitoring_ready=bool(data.get("abuse_monitoring_ready", False)),
            owner_approved=bool(data.get("owner_approved", False)),
            public_agent_track_enabled=bool(data.get("public_agent_track_enabled", False)),
            production_workflows_enabled=bool(data.get("production_workflows_enabled", False)),
            credential_material_issued=bool(data.get("credential_material_issued", False)),
            account_creation_enabled=bool(data.get("account_creation_enabled", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", AgentTrackGAReadinessState.LOCAL_GA_STUB.value)),
            contract_version=str(data.get("contract_version", AGENT_TRACK_GA_READINESS_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_miner_refs"] = list(self.allowed_miner_refs)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def validate_agent_track_ga_criteria_record(record: AgentTrackGACriteriaRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_agent_track_ga_payload_errors(raw)
    if not isinstance(record, AgentTrackGACriteriaRecord):
        try:
            record = AgentTrackGACriteriaRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Agent-track GA criteria field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Agent-track GA criteria payload: {exc}")
            return errors
    if record.state not in {state.value for state in AgentTrackGACriteriaState}:
        errors.append(f"unknown Agent-track GA criteria state: {record.state}")
        return errors
    if record.contract_version != AGENT_TRACK_GA_CRITERIA_CONTRACT_VERSION:
        errors.append("contract_version must match Agent-track GA criteria contract")
    if not record.criteria_id.startswith("agent_track_ga_criteria:"):
        errors.append("criteria_id must be agent_track_ga_criteria:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("workspace_beta_ref", ("workspace_api_beta:",)),
            ("security_policy_ref", ("workspace_api_ga_security_policy:",)),
            ("entropy_policy_ref", ("workspace_entropy_policy:",)),
            ("abuse_monitoring_policy_ref", ("abuse_monitoring_policy:",)),
            ("legal_terms_ref", ("legal_terms:",)),
            ("slashing_policy_ref", ("slashing_policy:",)),
        ),
    )
    _append_agent_track_write_guard_errors(record, "P4.6 Agent-track GA criteria", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Agent-track GA criteria",
        (
            "agent_track_ga_criteria:",
            "workspace_api_beta:",
            "workspace_api_ga:",
            "workspace_entropy_policy:",
            "abuse_monitoring_policy:",
            "legal_terms:",
            "slashing_policy:",
            "owner_approval:",
        ),
    )
    if record.stake_threshold_usd < MIN_AGENT_GA_STAKE_USD:
        errors.append("stake_threshold_usd must be at least the Agent-track GA minimum")
    if record.min_crowned_patch_count < MIN_AGENT_GA_CROWNED_PATCHES:
        errors.append("min_crowned_patch_count must be at least the Agent-track GA minimum")
    if record.min_funded_loop_count < MIN_AGENT_GA_FUNDED_LOOPS:
        errors.append("min_funded_loop_count must be at least the Agent-track GA minimum")
    if record.min_clean_receipt_count < MIN_AGENT_GA_CLEAN_RECEIPTS:
        errors.append("min_clean_receipt_count must be at least the Agent-track GA minimum")
    if record.criteria_approved:
        _append_missing_true_flags(
            record,
            errors,
            (
                ("require_clean_receipts", "approved Agent-track GA criteria require clean receipts"),
                ("require_entropy_accounting_live", "approved Agent-track GA criteria require live entropy accounting"),
                ("require_security_review_passed", "approved Agent-track GA criteria require passed security review"),
                ("require_abuse_monitoring", "approved Agent-track GA criteria require abuse monitoring"),
                ("require_legal_terms", "approved Agent-track GA criteria require legal terms"),
                ("require_slashing_terms", "approved Agent-track GA criteria require slashing terms"),
            ),
        )
        if record.uses_local_fixtures:
            errors.append("Agent-track GA criteria approval cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Agent-track GA criteria approval cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("Agent-track GA criteria approval requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("Agent-track GA criteria approval requires evidence_refs")
        if record.state != AgentTrackGACriteriaState.READY_AFTER_OWNER_POLICY.value:
            errors.append("criteria_approved requires ready_after_owner_policy state")
    else:
        if not record.local_only:
            errors.append("not-approved Agent-track GA criteria must remain local_only")
        if record.state == AgentTrackGACriteriaState.READY_AFTER_OWNER_POLICY.value:
            errors.append("ready_after_owner_policy state requires criteria_approved")
    return errors


def validate_agent_track_ga_candidate_record(
    record: AgentTrackGACandidateRecord | Mapping[str, Any],
    *,
    criteria: Optional[AgentTrackGACriteriaRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_agent_track_ga_payload_errors(raw)
    if not isinstance(record, AgentTrackGACandidateRecord):
        try:
            record = AgentTrackGACandidateRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Agent-track GA candidate field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Agent-track GA candidate payload: {exc}")
            return errors
    if record.state not in {state.value for state in AgentTrackGACandidateState}:
        errors.append(f"unknown Agent-track GA candidate state: {record.state}")
        return errors
    if record.contract_version != AGENT_TRACK_GA_CANDIDATE_CONTRACT_VERSION:
        errors.append("contract_version must match Agent-track GA candidate contract")
    if not record.eligibility_id.startswith("agent_track_candidate:"):
        errors.append("eligibility_id must be agent_track_candidate:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("miner_ref", ("miner:",)),
            ("criteria_ref", ("agent_track_ga_criteria:",)),
            ("workspace_beta_ref", ("workspace_api_beta:",)),
            ("stake_ref", ("stake:",)),
            ("track_record_ref", ("miner_track_record:",)),
            ("clean_receipt_audit_ref", ("clean_receipt_audit:",)),
            ("legal_terms_ref", ("legal_terms:",)),
            ("slashing_terms_ref", ("slashing_terms:",)),
            ("entropy_ledger_ref", ("workspace_entropy:",)),
            ("abuse_monitoring_ref", ("abuse_monitoring:",)),
            ("security_review_ref", ("workspace_security_review:",)),
        ),
    )
    _append_agent_track_write_guard_errors(record, "P4.6 Agent-track GA candidate", errors)
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Agent-track GA candidate",
        (
            "agent_track_candidate:",
            "agent_track_ga_criteria:",
            "workspace_api_beta:",
            "workspace_entropy:",
            "workspace_security_review:",
            "stake:",
            "miner_track_record:",
            "clean_receipt_audit:",
            "legal_terms:",
            "slashing_terms:",
            "abuse_monitoring:",
            "owner_approval:",
        ),
    )
    if criteria is not None:
        if not isinstance(criteria, AgentTrackGACriteriaRecord):
            criteria = AgentTrackGACriteriaRecord.from_mapping(criteria)
        criteria_errors = validate_agent_track_ga_criteria_record(criteria)
        if criteria_errors:
            errors.append("source Agent-track GA criteria is invalid: " + "; ".join(criteria_errors))
        if record.criteria_ref != criteria.criteria_id:
            errors.append("Agent-track GA candidate criteria_ref mismatch")
        if record.workspace_beta_ref != criteria.workspace_beta_ref:
            errors.append("Agent-track GA candidate workspace_beta_ref mismatch")
    if record.ga_eligible:
        if criteria is None:
            errors.append("Agent-track GA eligibility requires supplied criteria")
        if criteria is not None and not criteria.criteria_approved:
            errors.append("Agent-track GA eligibility requires approved criteria")
        stake_threshold = criteria.stake_threshold_usd if criteria is not None else MIN_AGENT_GA_STAKE_USD
        min_crowned = criteria.min_crowned_patch_count if criteria is not None else MIN_AGENT_GA_CROWNED_PATCHES
        min_loops = criteria.min_funded_loop_count if criteria is not None else MIN_AGENT_GA_FUNDED_LOOPS
        min_clean = criteria.min_clean_receipt_count if criteria is not None else MIN_AGENT_GA_CLEAN_RECEIPTS
        if record.stake_usd < stake_threshold:
            errors.append("Agent-track GA eligibility requires stake_usd above threshold")
        if not (record.crowned_patch_count >= min_crowned or record.funded_loop_count >= min_loops):
            errors.append("Agent-track GA eligibility requires crowned patch or funded-loop track record")
        if record.clean_receipt_count < min_clean:
            errors.append("Agent-track GA eligibility requires clean_receipt_count above threshold")
        if record.tos_violation_count != 0:
            errors.append("Agent-track GA eligibility requires zero ToS violations")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("entropy_accounting_live", "Agent-track GA eligibility requires live entropy accounting"),
                ("entropy_budget_enforced", "Agent-track GA eligibility requires enforced entropy budget"),
                ("abuse_monitoring_clean", "Agent-track GA eligibility requires clean abuse monitoring"),
                ("legal_terms_accepted", "Agent-track GA eligibility requires accepted legal terms"),
                ("slashing_terms_accepted", "Agent-track GA eligibility requires accepted slashing terms"),
                ("security_review_passed", "Agent-track GA eligibility requires passed security review"),
            ),
        )
        if record.uses_local_fixtures:
            errors.append("Agent-track GA eligibility cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Agent-track GA eligibility cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("Agent-track GA eligibility requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("Agent-track GA eligibility requires evidence_refs")
        if record.state != AgentTrackGACandidateState.ELIGIBLE_AFTER_TRACK_RECORD.value:
            errors.append("ga_eligible requires eligible_after_track_record state")
    else:
        if not record.local_only:
            errors.append("not-eligible Agent-track GA candidates must remain local_only")
        if record.state == AgentTrackGACandidateState.ELIGIBLE_AFTER_TRACK_RECORD.value:
            errors.append("eligible_after_track_record state requires ga_eligible")
    return errors


def validate_agent_track_ga_readiness_record(
    record: AgentTrackGAReadinessRecord | Mapping[str, Any],
    *,
    criteria: Optional[AgentTrackGACriteriaRecord | Mapping[str, Any]] = None,
    candidate: Optional[AgentTrackGACandidateRecord | Mapping[str, Any]] = None,
    workspace_beta: Optional[WorkspaceAPIBetaEnablementRecord | Mapping[str, Any]] = None,
    workspace_allowlist: Optional[WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any]] = None,
    workspace_entropy: Optional[WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any]] = None,
    workspace_security: Optional[WorkspaceAPISecurityReviewRecord | Mapping[str, Any]] = None,
    method_contracts: Optional[Sequence[WorkspaceAPIMethodContractRecord | Mapping[str, Any]]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_agent_track_ga_payload_errors(raw)
    if not isinstance(record, AgentTrackGAReadinessRecord):
        try:
            record = AgentTrackGAReadinessRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Agent-track GA readiness field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Agent-track GA readiness payload: {exc}")
            return errors
    _append_scale_guard_errors(guards, errors)
    if record.state not in {state.value for state in AgentTrackGAReadinessState}:
        errors.append(f"unknown Agent-track GA readiness state: {record.state}")
        return errors
    if record.contract_version != AGENT_TRACK_GA_READINESS_CONTRACT_VERSION:
        errors.append("contract_version must match Agent-track GA readiness contract")
    if not record.ga_readiness_id.startswith("agent_track_ga:"):
        errors.append("ga_readiness_id must be agent_track_ga:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("criteria_ref", ("agent_track_ga_criteria:",)),
            ("candidate_ref", ("agent_track_candidate:",)),
            ("workspace_beta_ref", ("workspace_api_beta:",)),
            ("scale_readiness_ref", ("scale_readiness:",)),
            ("entropy_ledger_ref", ("workspace_entropy:",)),
            ("security_review_ref", ("workspace_security_review:",)),
            ("abuse_monitoring_ref", ("abuse_monitoring:",)),
            ("owner_review_ref", ("owner_approval:",)),
        ),
    )
    if not record.allowed_miner_refs:
        errors.append("Agent-track GA readiness requires allowed_miner_refs")
    for miner_ref in record.allowed_miner_refs:
        if not miner_ref.startswith("miner:"):
            errors.append("allowed_miner_refs values must be miner:-prefixed")
            break
    _append_agent_track_write_guard_errors(record, "P4.6 Agent-track GA readiness", errors)
    if record.public_agent_track_enabled:
        errors.append("P4.6 must not enable public agent track")
    if record.production_workflows_enabled:
        errors.append("P4.6 must not enable production workflows")
    if record.credential_material_issued:
        errors.append("P4.6 must not issue credential material")
    if record.account_creation_enabled:
        errors.append("P4.6 must not create Workspace accounts")
    _append_evidence_prefix_errors(
        record.evidence_refs,
        errors,
        "Agent-track GA readiness",
        (
            "agent_track_ga:",
            "agent_track_ga_criteria:",
            "agent_track_candidate:",
            "workspace_api_ga:",
            "workspace_api_beta:",
            "workspace_api_allowlist:",
            "workspace_entropy:",
            "workspace_security_review:",
            "abuse_monitoring:",
            "owner_approval:",
        ),
    )
    criteria_record = criteria if isinstance(criteria, AgentTrackGACriteriaRecord) else (
        AgentTrackGACriteriaRecord.from_mapping(criteria) if criteria is not None else None
    )
    candidate_record = candidate if isinstance(candidate, AgentTrackGACandidateRecord) else (
        AgentTrackGACandidateRecord.from_mapping(candidate) if candidate is not None else None
    )
    beta_record = workspace_beta if isinstance(workspace_beta, WorkspaceAPIBetaEnablementRecord) else (
        WorkspaceAPIBetaEnablementRecord.from_mapping(workspace_beta) if workspace_beta is not None else None
    )
    if criteria_record is not None:
        criteria_errors = validate_agent_track_ga_criteria_record(criteria_record)
        if criteria_errors:
            errors.append("source Agent-track GA criteria is invalid: " + "; ".join(criteria_errors))
        if record.criteria_ref != criteria_record.criteria_id:
            errors.append("Agent-track GA readiness criteria_ref mismatch")
        if record.criteria_approved != criteria_record.criteria_approved:
            errors.append("Agent-track GA readiness criteria_approved flag mismatch")
    if candidate_record is not None:
        candidate_errors = validate_agent_track_ga_candidate_record(candidate_record, criteria=criteria_record)
        if candidate_errors:
            errors.append("source Agent-track GA candidate is invalid: " + "; ".join(candidate_errors))
        if record.candidate_ref != candidate_record.eligibility_id:
            errors.append("Agent-track GA readiness candidate_ref mismatch")
        if record.candidate_eligible != candidate_record.ga_eligible:
            errors.append("Agent-track GA readiness candidate_eligible flag mismatch")
        if set(record.allowed_miner_refs) - {candidate_record.miner_ref}:
            errors.append("Agent-track GA readiness allowed_miner_refs must match eligible candidate")
    if beta_record is not None:
        beta_errors = validate_workspace_api_beta_enablement_record(
            beta_record,
            allowlist=workspace_allowlist,
            entropy_ledger=workspace_entropy,
            security_review=workspace_security,
            method_contracts=method_contracts,
        )
        if beta_errors:
            errors.append("source Workspace API beta is invalid: " + "; ".join(beta_errors))
        if record.workspace_beta_ref != beta_record.beta_id:
            errors.append("Agent-track GA readiness workspace_beta_ref mismatch")
        beta_valid = beta_record.beta_enablement_claimed and beta_record.workspace_api_calls_enabled
        if record.workspace_beta_valid != beta_valid:
            errors.append("Agent-track GA readiness workspace_beta_valid flag mismatch")
    if workspace_entropy is not None:
        entropy_record = workspace_entropy if isinstance(workspace_entropy, WorkspaceAPIEntropyLedgerRecord) else WorkspaceAPIEntropyLedgerRecord.from_mapping(workspace_entropy)
        if record.entropy_ledger_ref != entropy_record.ledger_id:
            errors.append("Agent-track GA readiness entropy_ledger_ref mismatch")
        if record.entropy_accounting_live != (entropy_record.accounting_live and entropy_record.budget_enforced):
            errors.append("Agent-track GA readiness entropy_accounting_live flag mismatch")
    if workspace_security is not None:
        security_record = workspace_security if isinstance(workspace_security, WorkspaceAPISecurityReviewRecord) else WorkspaceAPISecurityReviewRecord.from_mapping(workspace_security)
        if record.security_review_ref != security_record.review_id:
            errors.append("Agent-track GA readiness security_review_ref mismatch")
        if record.abuse_monitoring_ref != security_record.abuse_monitoring_ref:
            errors.append("Agent-track GA readiness abuse_monitoring_ref mismatch")
        if record.security_review_passed != security_record.security_review_passed:
            errors.append("Agent-track GA readiness security_review_passed flag mismatch")
        if record.abuse_monitoring_ready != security_record.abuse_monitoring_ready:
            errors.append("Agent-track GA readiness abuse_monitoring_ready flag mismatch")
    if record.ga_claimed:
        if criteria_record is None:
            errors.append("Agent-track GA requires supplied criteria")
        if candidate_record is None:
            errors.append("Agent-track GA requires supplied candidate")
        if beta_record is None:
            errors.append("Agent-track GA requires supplied workspace_beta")
        if workspace_entropy is None:
            errors.append("Agent-track GA requires supplied workspace_entropy")
        if workspace_security is None:
            errors.append("Agent-track GA requires supplied workspace_security")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("criteria_approved", "Agent-track GA requires approved criteria"),
                ("candidate_eligible", "Agent-track GA requires eligible candidate"),
                ("workspace_beta_valid", "Agent-track GA requires valid Workspace API beta"),
                ("entropy_accounting_live", "Agent-track GA requires live entropy accounting"),
                ("security_review_passed", "Agent-track GA requires passed security review"),
                ("abuse_monitoring_ready", "Agent-track GA requires abuse monitoring"),
                ("owner_approved", "Agent-track GA requires owner approval"),
            ),
        )
        if record.uses_local_fixtures:
            errors.append("Agent-track GA cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Agent-track GA cannot be claimed by a local_only record")
        if not record.owner_approval_ref:
            errors.append("Agent-track GA requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("Agent-track GA requires evidence_refs")
        if record.state != AgentTrackGAReadinessState.READY_AFTER_OWNER_REVIEW.value:
            errors.append("ga_claimed requires ready_after_owner_review state")
    else:
        if not record.local_only:
            errors.append("not-claimed Agent-track GA readiness records must remain local_only")
        if record.state == AgentTrackGAReadinessState.READY_AFTER_OWNER_REVIEW.value:
            errors.append("ready_after_owner_review state requires ga_claimed")
    return errors


def verify_research_lab_agent_track_ga(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    scale_summary = verify_scale_foundation()
    workspace_summary = verify_research_lab_workspace_api_beta()
    fixture = _load_fixture(Path(fixture_path))
    workspace_fixture = _load_fixture(WORKSPACE_API_BETA_FIXTURE_PATH)
    method_contracts = [
        WorkspaceAPIMethodContractRecord.from_mapping(method)
        for method in workspace_fixture["method_contracts"]
    ]
    local_allowlist = WorkspaceAPIMinerAllowlistRecord.from_mapping(workspace_fixture["local_allowlist"])
    local_entropy = WorkspaceAPIEntropyLedgerRecord.from_mapping(workspace_fixture["local_entropy_ledger"])
    local_security = WorkspaceAPISecurityReviewRecord.from_mapping(workspace_fixture["local_security_review"])
    local_beta = WorkspaceAPIBetaEnablementRecord.from_mapping(workspace_fixture["local_beta_enablement"])
    measured_allowlist = _deep_merge(local_allowlist.to_dict(), workspace_fixture["measured_allowlist_overrides"])
    measured_entropy = _deep_merge(local_entropy.to_dict(), workspace_fixture["measured_entropy_overrides"])
    measured_security = _deep_merge(local_security.to_dict(), workspace_fixture["measured_security_overrides"])
    measured_beta = _deep_merge(local_beta.to_dict(), workspace_fixture["measured_beta_overrides"])

    local_criteria = AgentTrackGACriteriaRecord.from_mapping(fixture["local_ga_criteria"])
    _assert(not validate_agent_track_ga_criteria_record(local_criteria), "local Agent-track GA criteria validates")
    _assert(not local_criteria.criteria_approved, "local criteria does not claim approval")

    ready_criteria = AgentTrackGACriteriaRecord.from_mapping(fixture["ready_ga_criteria"])
    _assert(not validate_agent_track_ga_criteria_record(ready_criteria), "ready Agent-track GA criteria validates")
    _assert(ready_criteria.criteria_approved, "ready criteria claims approval")

    local_candidate = AgentTrackGACandidateRecord.from_mapping(fixture["local_ga_candidate"])
    _assert(
        not validate_agent_track_ga_candidate_record(local_candidate, criteria=local_criteria),
        "local Agent-track GA candidate validates",
    )
    _assert(not local_candidate.ga_eligible, "local candidate does not claim eligibility")

    ready_candidate = AgentTrackGACandidateRecord.from_mapping(fixture["ready_ga_candidate"])
    _assert(
        not validate_agent_track_ga_candidate_record(ready_candidate, criteria=ready_criteria),
        "ready Agent-track GA candidate validates",
    )
    _assert(ready_candidate.ga_eligible, "ready candidate claims eligibility")

    local_readiness = AgentTrackGAReadinessRecord.from_mapping(fixture["local_ga_readiness"])
    _assert(
        not validate_agent_track_ga_readiness_record(
            local_readiness,
            criteria=local_criteria,
            candidate=local_candidate,
            workspace_beta=local_beta,
            workspace_allowlist=local_allowlist,
            workspace_entropy=local_entropy,
            workspace_security=local_security,
            method_contracts=method_contracts,
        ),
        "local Agent-track GA readiness validates",
    )
    _assert(not local_readiness.ga_claimed, "local GA readiness does not claim GA")

    ready_readiness = AgentTrackGAReadinessRecord.from_mapping(fixture["ready_ga_readiness"])
    _assert(
        not validate_agent_track_ga_readiness_record(
            ready_readiness,
            criteria=ready_criteria,
            candidate=ready_candidate,
            workspace_beta=measured_beta,
            workspace_allowlist=measured_allowlist,
            workspace_entropy=measured_entropy,
            workspace_security=measured_security,
            method_contracts=method_contracts,
        ),
        "ready Agent-track GA readiness validates",
    )
    _assert(ready_readiness.ga_claimed, "ready GA readiness claims GA")

    for invalid in fixture["invalid_ga_criteria"]:
        base = fixture[str(invalid.get("base", "local_ga_criteria"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_agent_track_ga_criteria_record(merged)
        _assert(errors, f"invalid Agent-track GA criteria fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_ga_candidates"]:
        base = fixture[str(invalid.get("base", "local_ga_candidate"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        if invalid.get("omit_criteria"):
            errors = validate_agent_track_ga_candidate_record(merged)
        else:
            criteria = AgentTrackGACriteriaRecord.from_mapping(fixture[str(invalid.get("criteria_base", "local_ga_criteria"))])
            errors = validate_agent_track_ga_candidate_record(merged, criteria=criteria)
        _assert(errors, f"invalid Agent-track GA candidate fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_errors = validate_agent_track_ga_readiness_record(
        local_readiness,
        criteria=local_criteria,
        candidate=local_candidate,
        workspace_beta=local_beta,
        workspace_allowlist=local_allowlist,
        workspace_entropy=local_entropy,
        workspace_security=local_security,
        method_contracts=method_contracts,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 4 guards block Agent-track GA readiness")
    _assert_expected_error(unsafe_errors, fixture["unsafe_scale_guards"])

    for invalid in fixture["invalid_ga_readiness"]:
        base = fixture[str(invalid.get("base", "local_ga_readiness"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        criteria = None
        candidate = None
        beta = None
        allowlist = None
        entropy = None
        security = None
        if not invalid.get("omit_criteria"):
            criteria = AgentTrackGACriteriaRecord.from_mapping(fixture[str(invalid.get("criteria_base", "ready_ga_criteria"))])
        if not invalid.get("omit_candidate"):
            candidate = AgentTrackGACandidateRecord.from_mapping(fixture[str(invalid.get("candidate_base", "ready_ga_candidate"))])
        supplied = str(invalid.get("workspace_supplied", "measured"))
        if not invalid.get("omit_workspace_beta"):
            if supplied == "local":
                beta = local_beta
                allowlist = local_allowlist
                entropy = local_entropy
                security = local_security
            else:
                beta = measured_beta
                allowlist = measured_allowlist
                entropy = measured_entropy
                security = measured_security
        errors = validate_agent_track_ga_readiness_record(
            merged,
            criteria=criteria,
            candidate=candidate,
            workspace_beta=beta,
            workspace_allowlist=allowlist,
            workspace_entropy=entropy,
            workspace_security=security,
            method_contracts=method_contracts,
        )
        _assert(errors, f"invalid Agent-track GA readiness fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"workspace_api_ga_evidence_ready": False}, ScaleGate.WORKSPACE_API_GA_EVIDENCE_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.6 Workspace API GA evidence gate remains required by P4.0")

    return {
        "scale_readiness_id": scale_summary["readiness_id"],
        "workspace_beta_id": workspace_summary["allowlist_id"],
        "criteria_id": local_criteria.criteria_id,
        "criteria_approved": local_criteria.criteria_approved,
        "candidate_id": local_candidate.eligibility_id,
        "candidate_eligible": local_candidate.ga_eligible,
        "ga_readiness_id": local_readiness.ga_readiness_id,
        "ga_claimed": local_readiness.ga_claimed,
        "public_agent_track_enabled": local_readiness.public_agent_track_enabled,
        "ready_control_validates": (
            ready_criteria.criteria_approved
            and ready_candidate.ga_eligible
            and ready_readiness.ga_claimed
        ),
    }


def _append_scale_guard_errors(
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]],
    errors: list[str],
) -> None:
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))


def _append_prefix_errors(record: Any, errors: list[str], checks: Sequence[tuple[str, tuple[str, ...]]]) -> None:
    for field, prefixes in checks:
        if not getattr(record, field).startswith(prefixes):
            if len(prefixes) == 1:
                errors.append(f"{field} must be {prefixes[0]}-prefixed")
            else:
                errors.append(f"{field} must use an approved prefix")


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


def _append_agent_track_write_guard_errors(record: Any, label: str, errors: list[str]) -> None:
    if bool(getattr(record, "production_writes", False)):
        errors.append(f"{label} must not enable production writes")
    if bool(getattr(record, "supabase_writes", False)):
        errors.append(f"{label} must not enable Supabase writes")
    if bool(getattr(record, "public_workflows", False)):
        errors.append(f"{label} must not enable public workflows")


def _protected_agent_track_ga_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_agent_track_ga_material(record))
    if not found:
        return []
    return ["P4.6 Agent-track GA payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_agent_track_ga_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_AGENT_TRACK_GA_KEYS and not key_text.endswith(
                ("_ref", "_refs", "_hash", "_hashes")
            ):
                found.add(key_path)
            found.update(_find_protected_agent_track_ga_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(_find_protected_agent_track_ga_material(nested, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_AGENT_TRACK_GA_MARKERS:
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
