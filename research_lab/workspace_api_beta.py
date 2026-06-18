"""Phase 3.6 Workspace API beta contracts.

P3.6 prepares an allowlisted agent-track Workspace API beta. The records here
describe method contracts, miner allowlist entries, entropy accounting,
security review, and beta enablement claims. They do not open an API server,
issue credentials, write production state, or enable public agent-track access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .model_pipeline_foundation import (
    PROTECTED_MODEL_PIPELINE_KEYS,
    PROTECTED_MODEL_PIPELINE_MARKERS,
    ModelPipelineWorkflowGuards,
    assert_model_pipeline_workflows_disabled,
    default_model_pipeline_workflow_guards,
    verify_model_pipeline_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "workspace_api_beta_fixtures.json"

WORKSPACE_API_METHOD_CONTRACT_VERSION = "workspace_api_method:v1:local_contract"
WORKSPACE_API_BETA_CONTRACT_VERSION = "workspace_api_beta:v1:local_contract"
PENDING_WORKSPACE_BETA_REF = "workspace_api_beta:pending"

MAX_METHOD_ENTROPY_BITS = 48
MAX_BETA_COHORT_SIZE = 50


class WorkspaceAPIMethod(str, Enum):
    GET_REPO_MAP = "get_repo_map"
    PROPOSE_PATCH = "propose_patch"
    REQUEST_EVAL = "request_eval"
    GET_FAILURE_ANALYSIS = "get_failure_analysis"
    SEARCH_CORPUS = "search_corpus"


class WorkspaceAPIMethodState(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    READY_AFTER_SECURITY_REVIEW = "ready_after_security_review"
    BLOCKED = "blocked"


class WorkspaceAPIAccessState(str, Enum):
    LOCAL_ALLOWLIST_STUB = "local_allowlist_stub"
    ALLOWLISTED_BETA_READY = "allowlisted_beta_ready"
    BLOCKED = "blocked"


class WorkspaceAPIEntropyState(str, Enum):
    LOCAL_LEDGER_STUB = "local_ledger_stub"
    ACCOUNTING_LIVE = "accounting_live"
    OVER_BUDGET = "over_budget"
    BLOCKED = "blocked"


class WorkspaceAPISecurityReviewState(str, Enum):
    LOCAL_CHECKLIST_STUB = "local_checklist_stub"
    PASSED_AFTER_REVIEW = "passed_after_review"
    BLOCKED = "blocked"


class WorkspaceAPIBetaState(str, Enum):
    LOCAL_BETA_STUB = "local_beta_stub"
    READY_AFTER_SECURITY_REVIEW = "ready_after_security_review"
    ALLOWLISTED_BETA_ENABLED = "allowlisted_beta_enabled"
    BLOCKED = "blocked"


ALLOWED_WORKSPACE_API_METHODS: tuple[str, ...] = tuple(method.value for method in WorkspaceAPIMethod)

PROTECTED_WORKSPACE_API_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_MODEL_PIPELINE_KEYS)
    | {
        "access_token",
        "api_key",
        "bearer_token",
        "credential_material",
        "customer_email",
        "full_repo",
        "jwt",
        "live_champion_artifact",
        "live_champion_weights",
        "model_weights",
        "password",
        "private_customer_data",
        "raw_corpus",
        "raw_customer_payload",
        "raw_repo",
        "repo_secret",
        "sealed_eval_details",
        "session_cookie",
        "ssh_key",
        "workspace_secret",
    }
)

PROTECTED_WORKSPACE_API_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_MODEL_PIPELINE_MARKERS)
        | {
            "access token",
            "api key",
            "bearer token",
            "customer email",
            "full repo",
            "live champion",
            "model weights",
            "private customer",
            "raw corpus",
            "raw repo",
            "repo secret",
            "sealed eval",
            "session cookie",
            "ssh key",
            "workspace secret",
        }
    )
)


@dataclass(frozen=True)
class WorkspaceAPIMethodContractRecord:
    contract_id: str
    method: str
    endpoint_ref: str
    input_schema_ref: str
    output_schema_ref: str
    allowed_request_fields: tuple[str, ...]
    allowed_response_fields: tuple[str, ...]
    max_entropy_bits_per_call: int
    rate_limit_per_hour: int
    sanitized_output_only: bool = True
    no_raw_corpus: bool = True
    no_private_customer_data: bool = True
    no_live_champion_ip: bool = True
    no_model_weights: bool = True
    no_direct_repo_write: bool = True
    no_secret_access: bool = True
    no_unbounded_search: bool = True
    production_workflow_enabled: bool = False
    public_ga_enabled: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    state: str = WorkspaceAPIMethodState.LOCAL_CONTRACT_STUB.value
    contract_version: str = WORKSPACE_API_METHOD_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceAPIMethodContractRecord":
        return cls(
            contract_id=str(data["contract_id"]),
            method=str(data["method"]),
            endpoint_ref=str(data.get("endpoint_ref", "")),
            input_schema_ref=str(data.get("input_schema_ref", "")),
            output_schema_ref=str(data.get("output_schema_ref", "")),
            allowed_request_fields=tuple(str(item) for item in data.get("allowed_request_fields", [])),
            allowed_response_fields=tuple(str(item) for item in data.get("allowed_response_fields", [])),
            max_entropy_bits_per_call=int(data.get("max_entropy_bits_per_call", 0)),
            rate_limit_per_hour=int(data.get("rate_limit_per_hour", 0)),
            sanitized_output_only=bool(data.get("sanitized_output_only", True)),
            no_raw_corpus=bool(data.get("no_raw_corpus", True)),
            no_private_customer_data=bool(data.get("no_private_customer_data", True)),
            no_live_champion_ip=bool(data.get("no_live_champion_ip", True)),
            no_model_weights=bool(data.get("no_model_weights", True)),
            no_direct_repo_write=bool(data.get("no_direct_repo_write", True)),
            no_secret_access=bool(data.get("no_secret_access", True)),
            no_unbounded_search=bool(data.get("no_unbounded_search", True)),
            production_workflow_enabled=bool(data.get("production_workflow_enabled", False)),
            public_ga_enabled=bool(data.get("public_ga_enabled", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", WorkspaceAPIMethodState.LOCAL_CONTRACT_STUB.value)),
            contract_version=str(data.get("contract_version", WORKSPACE_API_METHOD_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_request_fields"] = list(self.allowed_request_fields)
        data["allowed_response_fields"] = list(self.allowed_response_fields)
        return data


@dataclass(frozen=True)
class WorkspaceAPIMinerAllowlistRecord:
    allowlist_id: str
    miner_ref: str
    track: str
    scope_ref: str
    stake_ref: str
    track_record_ref: str
    legal_terms_ref: str
    entropy_budget_ref: str
    abuse_monitoring_ref: str
    workspace_role_ref: str
    access_token_ref: str
    allowed_methods: tuple[str, ...]
    stake_gated: bool = False
    track_record_gated: bool = False
    legal_terms_accepted: bool = False
    entropy_accounting_required: bool = True
    abuse_monitoring_enabled: bool = False
    beta_access_enabled: bool = False
    public_access_enabled: bool = False
    account_created: bool = False
    credential_material_issued: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = WorkspaceAPIAccessState.LOCAL_ALLOWLIST_STUB.value
    contract_version: str = WORKSPACE_API_BETA_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceAPIMinerAllowlistRecord":
        return cls(
            allowlist_id=str(data["allowlist_id"]),
            miner_ref=str(data.get("miner_ref", "")),
            track=str(data.get("track", "")),
            scope_ref=str(data.get("scope_ref", "")),
            stake_ref=str(data.get("stake_ref", "")),
            track_record_ref=str(data.get("track_record_ref", "")),
            legal_terms_ref=str(data.get("legal_terms_ref", "")),
            entropy_budget_ref=str(data.get("entropy_budget_ref", "")),
            abuse_monitoring_ref=str(data.get("abuse_monitoring_ref", "")),
            workspace_role_ref=str(data.get("workspace_role_ref", "")),
            access_token_ref=str(data.get("access_token_ref", "")),
            allowed_methods=tuple(str(item) for item in data.get("allowed_methods", [])),
            stake_gated=bool(data.get("stake_gated", False)),
            track_record_gated=bool(data.get("track_record_gated", False)),
            legal_terms_accepted=bool(data.get("legal_terms_accepted", False)),
            entropy_accounting_required=bool(data.get("entropy_accounting_required", True)),
            abuse_monitoring_enabled=bool(data.get("abuse_monitoring_enabled", False)),
            beta_access_enabled=bool(data.get("beta_access_enabled", False)),
            public_access_enabled=bool(data.get("public_access_enabled", False)),
            account_created=bool(data.get("account_created", False)),
            credential_material_issued=bool(data.get("credential_material_issued", False)),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", WorkspaceAPIAccessState.LOCAL_ALLOWLIST_STUB.value)),
            contract_version=str(data.get("contract_version", WORKSPACE_API_BETA_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_methods"] = list(self.allowed_methods)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class WorkspaceAPIEntropyLedgerRecord:
    ledger_id: str
    miner_ref: str
    budget_window_ref: str
    entropy_budget_bits: int
    entropy_used_bits: int
    entropy_remaining_bits: int
    alert_threshold_bits: int
    per_method_bits: dict[str, int]
    accounting_live: bool = False
    budget_enforced: bool = False
    expansion_allowed: bool = False
    over_budget: bool = False
    public_agent_track_enabled: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = WorkspaceAPIEntropyState.LOCAL_LEDGER_STUB.value
    contract_version: str = WORKSPACE_API_BETA_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceAPIEntropyLedgerRecord":
        return cls(
            ledger_id=str(data["ledger_id"]),
            miner_ref=str(data.get("miner_ref", "")),
            budget_window_ref=str(data.get("budget_window_ref", "")),
            entropy_budget_bits=int(data.get("entropy_budget_bits", 0)),
            entropy_used_bits=int(data.get("entropy_used_bits", 0)),
            entropy_remaining_bits=int(data.get("entropy_remaining_bits", 0)),
            alert_threshold_bits=int(data.get("alert_threshold_bits", 0)),
            per_method_bits={str(key): int(value) for key, value in data.get("per_method_bits", {}).items()},
            accounting_live=bool(data.get("accounting_live", False)),
            budget_enforced=bool(data.get("budget_enforced", False)),
            expansion_allowed=bool(data.get("expansion_allowed", False)),
            over_budget=bool(data.get("over_budget", False)),
            public_agent_track_enabled=bool(data.get("public_agent_track_enabled", False)),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", WorkspaceAPIEntropyState.LOCAL_LEDGER_STUB.value)),
            contract_version=str(data.get("contract_version", WORKSPACE_API_BETA_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        data["per_method_bits"] = dict(self.per_method_bits)
        return data


@dataclass(frozen=True)
class WorkspaceAPISecurityReviewRecord:
    review_id: str
    allowlist_ref: str
    entropy_ledger_ref: str
    scope_ref: str
    abuse_monitoring_ref: str
    rate_limit_policy_ref: str
    endpoint_allowlist_ref: str
    redaction_policy_ref: str
    penetration_test_ref: str
    incident_response_ref: str
    tos_enforcement_ref: str
    security_review_passed: bool = False
    rate_limits_configured: bool = False
    endpoint_allowlist_enforced: bool = False
    abuse_monitoring_ready: bool = False
    redaction_review_passed: bool = False
    entropy_accounting_verified: bool = False
    no_raw_secret_access: bool = True
    no_live_champion_exposure: bool = True
    no_private_customer_data: bool = True
    no_production_workflows: bool = True
    no_public_ga: bool = True
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = WorkspaceAPISecurityReviewState.LOCAL_CHECKLIST_STUB.value
    contract_version: str = WORKSPACE_API_BETA_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceAPISecurityReviewRecord":
        return cls(
            review_id=str(data["review_id"]),
            allowlist_ref=str(data.get("allowlist_ref", "")),
            entropy_ledger_ref=str(data.get("entropy_ledger_ref", "")),
            scope_ref=str(data.get("scope_ref", "")),
            abuse_monitoring_ref=str(data.get("abuse_monitoring_ref", "")),
            rate_limit_policy_ref=str(data.get("rate_limit_policy_ref", "")),
            endpoint_allowlist_ref=str(data.get("endpoint_allowlist_ref", "")),
            redaction_policy_ref=str(data.get("redaction_policy_ref", "")),
            penetration_test_ref=str(data.get("penetration_test_ref", "")),
            incident_response_ref=str(data.get("incident_response_ref", "")),
            tos_enforcement_ref=str(data.get("tos_enforcement_ref", "")),
            security_review_passed=bool(data.get("security_review_passed", False)),
            rate_limits_configured=bool(data.get("rate_limits_configured", False)),
            endpoint_allowlist_enforced=bool(data.get("endpoint_allowlist_enforced", False)),
            abuse_monitoring_ready=bool(data.get("abuse_monitoring_ready", False)),
            redaction_review_passed=bool(data.get("redaction_review_passed", False)),
            entropy_accounting_verified=bool(data.get("entropy_accounting_verified", False)),
            no_raw_secret_access=bool(data.get("no_raw_secret_access", True)),
            no_live_champion_exposure=bool(data.get("no_live_champion_exposure", True)),
            no_private_customer_data=bool(data.get("no_private_customer_data", True)),
            no_production_workflows=bool(data.get("no_production_workflows", True)),
            no_public_ga=bool(data.get("no_public_ga", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", WorkspaceAPISecurityReviewState.LOCAL_CHECKLIST_STUB.value)),
            contract_version=str(data.get("contract_version", WORKSPACE_API_BETA_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class WorkspaceAPIBetaEnablementRecord:
    beta_id: str
    allowlist_ref: str
    entropy_ledger_ref: str
    security_review_ref: str
    model_pipeline_readiness_ref: str
    beta_scope_ref: str
    allowed_miner_refs: tuple[str, ...]
    allowed_methods: tuple[str, ...]
    max_cohort_size: int
    beta_enablement_claimed: bool = False
    workspace_api_calls_enabled: bool = False
    public_ga_enabled: bool = False
    production_workflows_enabled: bool = False
    public_agent_track_enabled: bool = False
    credential_material_issued: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = WorkspaceAPIBetaState.LOCAL_BETA_STUB.value
    contract_version: str = WORKSPACE_API_BETA_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceAPIBetaEnablementRecord":
        return cls(
            beta_id=str(data["beta_id"]),
            allowlist_ref=str(data.get("allowlist_ref", "")),
            entropy_ledger_ref=str(data.get("entropy_ledger_ref", "")),
            security_review_ref=str(data.get("security_review_ref", "")),
            model_pipeline_readiness_ref=str(data.get("model_pipeline_readiness_ref", "")),
            beta_scope_ref=str(data.get("beta_scope_ref", "")),
            allowed_miner_refs=tuple(str(item) for item in data.get("allowed_miner_refs", [])),
            allowed_methods=tuple(str(item) for item in data.get("allowed_methods", [])),
            max_cohort_size=int(data.get("max_cohort_size", 0)),
            beta_enablement_claimed=bool(data.get("beta_enablement_claimed", False)),
            workspace_api_calls_enabled=bool(data.get("workspace_api_calls_enabled", False)),
            public_ga_enabled=bool(data.get("public_ga_enabled", False)),
            production_workflows_enabled=bool(data.get("production_workflows_enabled", False)),
            public_agent_track_enabled=bool(data.get("public_agent_track_enabled", False)),
            credential_material_issued=bool(data.get("credential_material_issued", False)),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", WorkspaceAPIBetaState.LOCAL_BETA_STUB.value)),
            contract_version=str(data.get("contract_version", WORKSPACE_API_BETA_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_miner_refs"] = list(self.allowed_miner_refs)
        data["allowed_methods"] = list(self.allowed_methods)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def workspace_api_method_contract_hash(record: WorkspaceAPIMethodContractRecord | Mapping[str, Any]) -> str:
    data = record.to_dict() if isinstance(record, WorkspaceAPIMethodContractRecord) else dict(record)
    data = dict(data)
    data.pop("contract_hash", None)
    return sha256_json(data)


def build_workspace_api_method_contract(**kwargs: Any) -> WorkspaceAPIMethodContractRecord:
    return WorkspaceAPIMethodContractRecord.from_mapping(kwargs)


def build_workspace_api_beta_enablement(
    *,
    allowlist: WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any],
    entropy_ledger: WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any],
    security_review: WorkspaceAPISecurityReviewRecord | Mapping[str, Any],
    model_pipeline_readiness_ref: str,
    beta_scope_ref: str,
    max_cohort_size: int = 1,
    local_only: bool = True,
    uses_local_fixtures: bool = True,
) -> WorkspaceAPIBetaEnablementRecord:
    if not isinstance(allowlist, WorkspaceAPIMinerAllowlistRecord):
        allowlist = WorkspaceAPIMinerAllowlistRecord.from_mapping(allowlist)
    if not isinstance(entropy_ledger, WorkspaceAPIEntropyLedgerRecord):
        entropy_ledger = WorkspaceAPIEntropyLedgerRecord.from_mapping(entropy_ledger)
    if not isinstance(security_review, WorkspaceAPISecurityReviewRecord):
        security_review = WorkspaceAPISecurityReviewRecord.from_mapping(security_review)
    identity_payload = {
        "allowlist_ref": allowlist.allowlist_id,
        "entropy_ledger_ref": entropy_ledger.ledger_id,
        "security_review_ref": security_review.review_id,
        "model_pipeline_readiness_ref": model_pipeline_readiness_ref,
        "beta_scope_ref": beta_scope_ref,
        "allowed_miner_refs": [allowlist.miner_ref],
        "allowed_methods": list(allowlist.allowed_methods),
        "max_cohort_size": max_cohort_size,
    }
    beta_id = "workspace_api_beta:" + sha256_json(identity_payload).removeprefix("sha256:")[:16]
    return WorkspaceAPIBetaEnablementRecord(
        beta_id=beta_id,
        allowlist_ref=allowlist.allowlist_id,
        entropy_ledger_ref=entropy_ledger.ledger_id,
        security_review_ref=security_review.review_id,
        model_pipeline_readiness_ref=model_pipeline_readiness_ref,
        beta_scope_ref=beta_scope_ref,
        allowed_miner_refs=(allowlist.miner_ref,),
        allowed_methods=allowlist.allowed_methods,
        max_cohort_size=max_cohort_size,
        local_only=local_only,
        uses_local_fixtures=uses_local_fixtures,
    )


def validate_workspace_api_method_contract(
    record: WorkspaceAPIMethodContractRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_workspace_api_payload_errors(raw)
    if not isinstance(record, WorkspaceAPIMethodContractRecord):
        try:
            record = WorkspaceAPIMethodContractRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required workspace API method field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid workspace API method field value: {exc}")
            return errors
    if record.contract_version != WORKSPACE_API_METHOD_CONTRACT_VERSION:
        errors.append("contract_version must match Workspace API method contract")
    if not record.contract_id.startswith("workspace_api_method:"):
        errors.append("contract_id must be workspace_api_method:-prefixed")
    if record.method not in ALLOWED_WORKSPACE_API_METHODS:
        errors.append(f"unknown workspace API method: {record.method}")
    if not record.endpoint_ref.startswith("workspace_api_endpoint:"):
        errors.append("endpoint_ref must be workspace_api_endpoint:-prefixed")
    if not record.input_schema_ref.startswith("workspace_api_schema:"):
        errors.append("input_schema_ref must be workspace_api_schema:-prefixed")
    if not record.output_schema_ref.startswith("workspace_api_schema:"):
        errors.append("output_schema_ref must be workspace_api_schema:-prefixed")
    if not record.allowed_request_fields:
        errors.append("method contract requires allowed_request_fields")
    if not record.allowed_response_fields:
        errors.append("method contract requires allowed_response_fields")
    _append_public_field_errors(errors, record.allowed_request_fields, "allowed_request_fields")
    _append_public_field_errors(errors, record.allowed_response_fields, "allowed_response_fields")
    if record.max_entropy_bits_per_call <= 0 or record.max_entropy_bits_per_call > MAX_METHOD_ENTROPY_BITS:
        errors.append("max_entropy_bits_per_call must be within the Workspace API entropy cap")
    if record.rate_limit_per_hour <= 0:
        errors.append("rate_limit_per_hour must be positive")
    for field_name in (
        "sanitized_output_only",
        "no_raw_corpus",
        "no_private_customer_data",
        "no_live_champion_ip",
        "no_model_weights",
        "no_direct_repo_write",
        "no_secret_access",
        "no_unbounded_search",
    ):
        if not getattr(record, field_name):
            errors.append(f"method contract must keep {field_name}=true")
    if record.production_workflow_enabled:
        errors.append("method contract must not enable production workflows")
    if record.public_ga_enabled:
        errors.append("method contract must not enable public GA")
    if record.state not in {state.value for state in WorkspaceAPIMethodState}:
        errors.append(f"unknown workspace API method state: {record.state}")
    if record.state == WorkspaceAPIMethodState.LOCAL_CONTRACT_STUB.value:
        if not record.local_only:
            errors.append("local method contracts must remain local_only")
        if not record.uses_local_fixtures:
            errors.append("local method contracts must be marked uses_local_fixtures")
    return errors


def validate_workspace_api_miner_allowlist_record(
    record: WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any],
    *,
    method_contracts: Optional[Sequence[WorkspaceAPIMethodContractRecord | Mapping[str, Any]]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_workspace_api_payload_errors(raw)
    if not isinstance(record, WorkspaceAPIMinerAllowlistRecord):
        try:
            record = WorkspaceAPIMinerAllowlistRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required workspace API allowlist field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid workspace API allowlist field value: {exc}")
            return errors
    if record.contract_version != WORKSPACE_API_BETA_CONTRACT_VERSION:
        errors.append("contract_version must match Workspace API beta contract")
    if not record.allowlist_id.startswith("workspace_api_allowlist:"):
        errors.append("allowlist_id must be workspace_api_allowlist:-prefixed")
    if not record.miner_ref.startswith("miner:"):
        errors.append("miner_ref must be miner:-prefixed")
    if record.track != "agent_track":
        errors.append("Workspace API beta is agent_track only")
    _validate_prefixed(errors, "scope_ref", record.scope_ref, "workspace_beta_scope:")
    _validate_prefixed(errors, "stake_ref", record.stake_ref, "stake:")
    _validate_prefixed(errors, "track_record_ref", record.track_record_ref, "miner_track_record:")
    _validate_prefixed(errors, "legal_terms_ref", record.legal_terms_ref, "legal_terms:")
    _validate_prefixed(errors, "entropy_budget_ref", record.entropy_budget_ref, "workspace_entropy:")
    _validate_prefixed(errors, "abuse_monitoring_ref", record.abuse_monitoring_ref, "abuse_monitoring:")
    _validate_prefixed(errors, "workspace_role_ref", record.workspace_role_ref, "workspace_role:")
    _validate_prefixed(errors, "access_token_ref", record.access_token_ref, "workspace_token_ref:")
    if not record.allowed_methods:
        errors.append("allowlist requires allowed_methods")
    unknown = sorted(set(record.allowed_methods) - set(ALLOWED_WORKSPACE_API_METHODS))
    if unknown:
        errors.append("allowlist contains unknown Workspace API methods: " + ", ".join(unknown))
    if method_contracts is not None:
        contracted = _method_set(method_contracts)
        missing = sorted(set(record.allowed_methods) - contracted)
        if missing:
            errors.append("allowlist method not present in method contracts: " + ", ".join(missing))
    if not record.entropy_accounting_required:
        errors.append("allowlist must require entropy accounting")
    if record.public_access_enabled:
        errors.append("allowlist must not enable public access")
    if record.account_created:
        errors.append("P3.6 contract must not create Workspace accounts")
    if record.credential_material_issued:
        errors.append("P3.6 contract must not issue credential material")
    if record.production_writes:
        errors.append("P3.6 allowlist must not write production state")
    if record.supabase_writes:
        errors.append("P3.6 allowlist must not enable Supabase writes")
    if record.state not in {state.value for state in WorkspaceAPIAccessState}:
        errors.append(f"unknown workspace API allowlist state: {record.state}")
    if record.beta_access_enabled or record.state == WorkspaceAPIAccessState.ALLOWLISTED_BETA_READY.value:
        if record.uses_local_fixtures:
            errors.append("Workspace API beta access cannot be enabled from local fixtures")
        if record.local_only:
            errors.append("Workspace API beta access cannot be enabled by a local_only record")
        for field_name in ("stake_gated", "track_record_gated", "legal_terms_accepted", "abuse_monitoring_enabled"):
            if not getattr(record, field_name):
                errors.append(f"Workspace API beta access requires {field_name}")
        if not record.owner_approval_ref:
            errors.append("Workspace API beta access requires owner_approval_ref")
        if record.allowlist_id not in record.evidence_refs:
            errors.append("Workspace API beta access evidence_refs must include allowlist_id")
    else:
        if not record.local_only:
            errors.append("inactive Workspace API allowlist records must remain local_only")
    return errors


def validate_workspace_api_entropy_ledger_record(
    record: WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_workspace_api_payload_errors(raw)
    if not isinstance(record, WorkspaceAPIEntropyLedgerRecord):
        try:
            record = WorkspaceAPIEntropyLedgerRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required workspace API entropy field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid workspace API entropy field value: {exc}")
            return errors
    if record.contract_version != WORKSPACE_API_BETA_CONTRACT_VERSION:
        errors.append("contract_version must match Workspace API beta contract")
    if not record.ledger_id.startswith("workspace_entropy:"):
        errors.append("ledger_id must be workspace_entropy:-prefixed")
    if not record.miner_ref.startswith("miner:"):
        errors.append("miner_ref must be miner:-prefixed")
    _validate_prefixed(errors, "budget_window_ref", record.budget_window_ref, "workspace_entropy_window:")
    if record.entropy_budget_bits <= 0:
        errors.append("entropy_budget_bits must be positive")
    if record.entropy_used_bits < 0:
        errors.append("entropy_used_bits must be non-negative")
    if record.alert_threshold_bits <= 0 or record.alert_threshold_bits > record.entropy_budget_bits:
        errors.append("alert_threshold_bits must be positive and within budget")
    if any(value < 0 for value in record.per_method_bits.values()):
        errors.append("per_method_bits must be non-negative")
    unknown = sorted(set(record.per_method_bits) - set(ALLOWED_WORKSPACE_API_METHODS))
    if unknown:
        errors.append("entropy ledger contains unknown Workspace API methods: " + ", ".join(unknown))
    if sum(record.per_method_bits.values()) != record.entropy_used_bits:
        errors.append("entropy_used_bits must equal sum(per_method_bits)")
    expected_remaining = record.entropy_budget_bits - record.entropy_used_bits
    if record.entropy_remaining_bits != expected_remaining:
        errors.append("entropy_remaining_bits must equal budget minus used bits")
    if record.entropy_used_bits > record.entropy_budget_bits and not record.over_budget:
        errors.append("over-budget entropy usage must mark over_budget")
    if record.entropy_used_bits <= record.entropy_budget_bits and record.over_budget:
        errors.append("over_budget must reflect entropy usage over budget")
    if record.public_agent_track_enabled:
        errors.append("entropy ledger must not enable public agent track")
    if record.production_writes:
        errors.append("P3.6 entropy ledger must not write production state")
    if record.supabase_writes:
        errors.append("P3.6 entropy ledger must not enable Supabase writes")
    if record.state not in {state.value for state in WorkspaceAPIEntropyState}:
        errors.append(f"unknown workspace API entropy state: {record.state}")
    if record.expansion_allowed:
        if record.uses_local_fixtures:
            errors.append("agent-track expansion cannot be allowed from local fixtures")
        if record.local_only:
            errors.append("agent-track expansion cannot be allowed by a local_only ledger")
        if not record.accounting_live:
            errors.append("agent-track expansion requires live entropy accounting")
        if not record.budget_enforced:
            errors.append("agent-track expansion requires enforced entropy budget")
        if record.over_budget:
            errors.append("agent-track expansion requires miner not over budget")
        if not record.owner_approval_ref:
            errors.append("agent-track expansion requires owner_approval_ref")
        if record.ledger_id not in record.evidence_refs:
            errors.append("agent-track expansion evidence_refs must include ledger_id")
    else:
        if record.state == WorkspaceAPIEntropyState.ACCOUNTING_LIVE.value and not record.accounting_live:
            errors.append("accounting_live state requires accounting_live=true")
    return errors


def validate_workspace_api_security_review_record(
    record: WorkspaceAPISecurityReviewRecord | Mapping[str, Any],
    *,
    allowlist: Optional[WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any]] = None,
    entropy_ledger: Optional[WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_workspace_api_payload_errors(raw)
    if not isinstance(record, WorkspaceAPISecurityReviewRecord):
        try:
            record = WorkspaceAPISecurityReviewRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required workspace API security field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid workspace API security field value: {exc}")
            return errors
    if record.contract_version != WORKSPACE_API_BETA_CONTRACT_VERSION:
        errors.append("contract_version must match Workspace API beta contract")
    if not record.review_id.startswith("workspace_security_review:"):
        errors.append("review_id must be workspace_security_review:-prefixed")
    _validate_prefixed(errors, "allowlist_ref", record.allowlist_ref, "workspace_api_allowlist:")
    _validate_prefixed(errors, "entropy_ledger_ref", record.entropy_ledger_ref, "workspace_entropy:")
    _validate_prefixed(errors, "scope_ref", record.scope_ref, "workspace_beta_scope:")
    _validate_prefixed(errors, "abuse_monitoring_ref", record.abuse_monitoring_ref, "abuse_monitoring:")
    _validate_prefixed(errors, "rate_limit_policy_ref", record.rate_limit_policy_ref, "rate_limit_policy:")
    _validate_prefixed(errors, "endpoint_allowlist_ref", record.endpoint_allowlist_ref, "workspace_endpoint_allowlist:")
    _validate_prefixed(errors, "redaction_policy_ref", record.redaction_policy_ref, "redaction_policy:")
    _validate_prefixed(errors, "penetration_test_ref", record.penetration_test_ref, "security_test:")
    _validate_prefixed(errors, "incident_response_ref", record.incident_response_ref, "incident_response:")
    _validate_prefixed(errors, "tos_enforcement_ref", record.tos_enforcement_ref, "tos_enforcement:")
    for field_name in (
        "no_raw_secret_access",
        "no_live_champion_exposure",
        "no_private_customer_data",
        "no_production_workflows",
        "no_public_ga",
    ):
        if not getattr(record, field_name):
            errors.append(f"security review must keep {field_name}=true")
    if record.state not in {state.value for state in WorkspaceAPISecurityReviewState}:
        errors.append(f"unknown workspace API security review state: {record.state}")
    if allowlist is not None:
        allowlist_record = allowlist if isinstance(allowlist, WorkspaceAPIMinerAllowlistRecord) else WorkspaceAPIMinerAllowlistRecord.from_mapping(allowlist)
        if record.allowlist_ref != allowlist_record.allowlist_id:
            errors.append("security review allowlist_ref mismatch")
        if record.scope_ref != allowlist_record.scope_ref:
            errors.append("security review scope_ref mismatch")
        if record.abuse_monitoring_ref != allowlist_record.abuse_monitoring_ref:
            errors.append("security review abuse_monitoring_ref mismatch")
    if entropy_ledger is not None:
        entropy_record = entropy_ledger if isinstance(entropy_ledger, WorkspaceAPIEntropyLedgerRecord) else WorkspaceAPIEntropyLedgerRecord.from_mapping(entropy_ledger)
        if record.entropy_ledger_ref != entropy_record.ledger_id:
            errors.append("security review entropy_ledger_ref mismatch")
    if record.security_review_passed or record.state == WorkspaceAPISecurityReviewState.PASSED_AFTER_REVIEW.value:
        if record.uses_local_fixtures:
            errors.append("Workspace API security review cannot pass from local fixtures")
        if record.local_only:
            errors.append("Workspace API security review cannot pass as local_only")
        for field_name in (
            "rate_limits_configured",
            "endpoint_allowlist_enforced",
            "abuse_monitoring_ready",
            "redaction_review_passed",
            "entropy_accounting_verified",
        ):
            if not getattr(record, field_name):
                errors.append(f"Workspace API security review requires {field_name}")
        if not record.owner_approval_ref:
            errors.append("Workspace API security review requires owner_approval_ref")
        if record.review_id not in record.evidence_refs:
            errors.append("Workspace API security review evidence_refs must include review_id")
    else:
        if not record.local_only:
            errors.append("not-passed Workspace API security reviews must remain local_only")
    return errors


def validate_workspace_api_beta_enablement_record(
    record: WorkspaceAPIBetaEnablementRecord | Mapping[str, Any],
    *,
    allowlist: Optional[WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any]] = None,
    entropy_ledger: Optional[WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any]] = None,
    security_review: Optional[WorkspaceAPISecurityReviewRecord | Mapping[str, Any]] = None,
    method_contracts: Optional[Sequence[WorkspaceAPIMethodContractRecord | Mapping[str, Any]]] = None,
    guards: Optional[ModelPipelineWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_workspace_api_payload_errors(raw)
    if not isinstance(record, WorkspaceAPIBetaEnablementRecord):
        try:
            record = WorkspaceAPIBetaEnablementRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required workspace API beta field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid workspace API beta field value: {exc}")
            return errors
    try:
        assert_model_pipeline_workflows_disabled(guards or default_model_pipeline_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != WORKSPACE_API_BETA_CONTRACT_VERSION:
        errors.append("contract_version must match Workspace API beta contract")
    if not record.beta_id.startswith("workspace_api_beta:"):
        errors.append("beta_id must be workspace_api_beta:-prefixed")
    if record.beta_id == PENDING_WORKSPACE_BETA_REF and (
        record.beta_enablement_claimed or record.state == WorkspaceAPIBetaState.ALLOWLISTED_BETA_ENABLED.value
    ):
        errors.append("Workspace API beta enablement requires non-pending beta_id")
    _validate_prefixed(errors, "allowlist_ref", record.allowlist_ref, "workspace_api_allowlist:")
    _validate_prefixed(errors, "entropy_ledger_ref", record.entropy_ledger_ref, "workspace_entropy:")
    _validate_prefixed(errors, "security_review_ref", record.security_review_ref, "workspace_security_review:")
    _validate_prefixed(errors, "model_pipeline_readiness_ref", record.model_pipeline_readiness_ref, "model_pipeline_readiness:")
    _validate_prefixed(errors, "beta_scope_ref", record.beta_scope_ref, "workspace_beta_scope:")
    if not record.allowed_miner_refs:
        errors.append("Workspace API beta requires allowed_miner_refs")
    _validate_sequence_prefixes(errors, "allowed_miner_refs", record.allowed_miner_refs, "miner:")
    if not record.allowed_methods:
        errors.append("Workspace API beta requires allowed_methods")
    unknown_methods = sorted(set(record.allowed_methods) - set(ALLOWED_WORKSPACE_API_METHODS))
    if unknown_methods:
        errors.append("Workspace API beta contains unknown methods: " + ", ".join(unknown_methods))
    if method_contracts is not None:
        contracted = _method_set(method_contracts)
        missing = sorted(set(record.allowed_methods) - contracted)
        if missing:
            errors.append("Workspace API beta method not present in method contracts: " + ", ".join(missing))
    if record.max_cohort_size <= 0 or record.max_cohort_size > MAX_BETA_COHORT_SIZE:
        errors.append("max_cohort_size must be within allowlisted beta limits")
    if record.public_ga_enabled:
        errors.append("Workspace API beta must not enable public GA")
    if record.production_workflows_enabled:
        errors.append("Workspace API beta must not enable production workflows")
    if record.public_agent_track_enabled:
        errors.append("Workspace API beta must not enable public agent track")
    if record.credential_material_issued:
        errors.append("Workspace API beta must not issue credential material")
    if record.production_writes:
        errors.append("Workspace API beta must not write production state")
    if record.supabase_writes:
        errors.append("Workspace API beta must not enable Supabase writes")
    if record.workspace_api_calls_enabled and not record.beta_enablement_claimed:
        errors.append("workspace_api_calls_enabled requires beta_enablement_claimed")
    if record.state not in {state.value for state in WorkspaceAPIBetaState}:
        errors.append(f"unknown Workspace API beta state: {record.state}")
    _append_supplied_record_consistency_errors(
        errors,
        record,
        allowlist=allowlist,
        entropy_ledger=entropy_ledger,
        security_review=security_review,
    )
    if record.beta_enablement_claimed or record.state == WorkspaceAPIBetaState.ALLOWLISTED_BETA_ENABLED.value:
        if allowlist is None:
            errors.append("Workspace API beta enablement requires supplied allowlist")
        if entropy_ledger is None:
            errors.append("Workspace API beta enablement requires supplied entropy_ledger")
        if security_review is None:
            errors.append("Workspace API beta enablement requires supplied security_review")
        if record.uses_local_fixtures:
            errors.append("Workspace API beta enablement cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Workspace API beta enablement cannot be claimed by a local_only record")
        if not record.workspace_api_calls_enabled:
            errors.append("Workspace API beta enablement requires workspace_api_calls_enabled")
        if not record.owner_approval_ref:
            errors.append("Workspace API beta enablement requires owner_approval_ref")
        if record.beta_id not in record.evidence_refs:
            errors.append("Workspace API beta enablement evidence_refs must include beta_id")
    else:
        if not record.local_only:
            errors.append("inactive Workspace API beta records must remain local_only")
        if record.state == WorkspaceAPIBetaState.ALLOWLISTED_BETA_ENABLED.value:
            errors.append("allowlisted_beta_enabled state requires beta_enablement_claimed")
    return errors


def verify_research_lab_workspace_api_beta(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    model_pipeline_summary = verify_model_pipeline_foundation()
    fixture = _load_fixture(Path(fixture_path))

    method_contracts = [
        WorkspaceAPIMethodContractRecord.from_mapping(method)
        for method in fixture["method_contracts"]
    ]
    method_names = {record.method for record in method_contracts}
    _assert(method_names == set(ALLOWED_WORKSPACE_API_METHODS), "P3.6 fixture covers all Workspace API methods")
    for method in method_contracts:
        _assert(not validate_workspace_api_method_contract(method), f"method contract validates: {method.method}")

    allowlist = WorkspaceAPIMinerAllowlistRecord.from_mapping(fixture["local_allowlist"])
    _assert(
        not validate_workspace_api_miner_allowlist_record(allowlist, method_contracts=method_contracts),
        "local Workspace API allowlist validates",
    )
    _assert(not allowlist.beta_access_enabled, "local allowlist does not enable beta access")

    entropy = WorkspaceAPIEntropyLedgerRecord.from_mapping(fixture["local_entropy_ledger"])
    _assert(not validate_workspace_api_entropy_ledger_record(entropy), "local entropy ledger validates")
    _assert(not entropy.accounting_live, "local entropy ledger does not claim live accounting")

    security = WorkspaceAPISecurityReviewRecord.from_mapping(fixture["local_security_review"])
    _assert(
        not validate_workspace_api_security_review_record(security, allowlist=allowlist, entropy_ledger=entropy),
        "local security review checklist validates",
    )
    _assert(not security.security_review_passed, "local security review does not pass")

    beta = WorkspaceAPIBetaEnablementRecord.from_mapping(fixture["local_beta_enablement"])
    _assert(
        not validate_workspace_api_beta_enablement_record(
            beta,
            allowlist=allowlist,
            entropy_ledger=entropy,
            security_review=security,
            method_contracts=method_contracts,
        ),
        "local Workspace API beta enablement stub validates",
    )
    _assert(not beta.beta_enablement_claimed, "local beta record does not claim enablement")
    _assert(not beta.workspace_api_calls_enabled, "local beta record does not enable API calls")

    built_beta = build_workspace_api_beta_enablement(
        allowlist=allowlist,
        entropy_ledger=entropy,
        security_review=security,
        model_pipeline_readiness_ref=fixture["model_pipeline_readiness_ref"],
        beta_scope_ref=allowlist.scope_ref,
    )
    _assert(not validate_workspace_api_beta_enablement_record(built_beta), "built local beta stub validates")
    _assert(built_beta.allowed_methods == allowlist.allowed_methods, "builder carries allowlisted methods")

    for invalid in fixture["invalid_method_contracts"]:
        base = fixture["method_contracts"][0]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_workspace_api_method_contract(record)
        _assert(errors, f"invalid method contract fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_allowlists"]:
        base = fixture[str(invalid.get("base", "local_allowlist"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_workspace_api_miner_allowlist_record(record, method_contracts=method_contracts)
        _assert(errors, f"invalid Workspace API allowlist fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_entropy_ledgers"]:
        base = fixture[str(invalid.get("base", "local_entropy_ledger"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_workspace_api_entropy_ledger_record(record)
        _assert(errors, f"invalid entropy ledger fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_security_reviews"]:
        base = fixture[str(invalid.get("base", "local_security_review"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_workspace_api_security_review_record(record, allowlist=allowlist, entropy_ledger=entropy)
        _assert(errors, f"invalid security review fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    measured_allowlist = _deep_merge(allowlist.to_dict(), fixture["measured_allowlist_overrides"])
    measured_entropy = _deep_merge(entropy.to_dict(), fixture["measured_entropy_overrides"])
    measured_security = _deep_merge(security.to_dict(), fixture["measured_security_overrides"])
    measured_beta = _deep_merge(beta.to_dict(), fixture["measured_beta_overrides"])
    _assert(
        not validate_workspace_api_miner_allowlist_record(measured_allowlist, method_contracts=method_contracts),
        "measured allowlist validates",
    )
    _assert(not validate_workspace_api_entropy_ledger_record(measured_entropy), "measured entropy ledger validates")
    _assert(
        not validate_workspace_api_security_review_record(
            measured_security,
            allowlist=measured_allowlist,
            entropy_ledger=measured_entropy,
        ),
        "measured security review validates",
    )
    measured_errors = validate_workspace_api_beta_enablement_record(
        measured_beta,
        allowlist=measured_allowlist,
        entropy_ledger=measured_entropy,
        security_review=measured_security,
        method_contracts=method_contracts,
    )
    _assert(not measured_errors, "fully supplied measured Workspace API beta claim validates")
    bare_measured_errors = validate_workspace_api_beta_enablement_record(measured_beta, method_contracts=method_contracts)
    _assert(bare_measured_errors, "bare Workspace API beta enablement claim fails closed")
    _assert_expected_error(bare_measured_errors, fixture["bare_measured_beta_expected_errors"])

    unsafe_guard_errors = validate_workspace_api_beta_enablement_record(
        beta,
        allowlist=allowlist,
        entropy_ledger=entropy,
        security_review=security,
        method_contracts=method_contracts,
        guards=fixture["unsafe_model_pipeline_guards"],
    )
    _assert(unsafe_guard_errors, "unsafe Phase 3 guards block Workspace API beta validation")
    _assert_expected_error(unsafe_guard_errors, fixture["unsafe_model_pipeline_guards"])

    for invalid in fixture["invalid_beta_enablements"]:
        base = fixture[str(invalid.get("base", "local_beta_enablement"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        supplied = invalid.get("supplied", "local")
        if supplied == "measured":
            supplied_allowlist = measured_allowlist
            supplied_entropy = measured_entropy
            supplied_security = measured_security
        else:
            supplied_allowlist = allowlist
            supplied_entropy = entropy
            supplied_security = security
        errors = validate_workspace_api_beta_enablement_record(
            record,
            allowlist=supplied_allowlist,
            entropy_ledger=supplied_entropy,
            security_review=supplied_security,
            method_contracts=method_contracts,
        )
        _assert(errors, f"invalid Workspace API beta enablement fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    return {
        "model_pipeline_readiness_id": model_pipeline_summary["readiness_id"],
        "method_contracts": len(method_contracts),
        "allowed_methods": sorted(method_names),
        "allowlist_id": allowlist.allowlist_id,
        "miner_ref": allowlist.miner_ref,
        "local_beta_enablement_claimed": beta.beta_enablement_claimed,
        "local_workspace_api_calls_enabled": beta.workspace_api_calls_enabled,
        "measured_beta_claim_validates": True,
        "entropy_budget_bits": entropy.entropy_budget_bits,
        "max_cohort_size": beta.max_cohort_size,
    }


def _append_supplied_record_consistency_errors(
    errors: list[str],
    record: WorkspaceAPIBetaEnablementRecord,
    *,
    allowlist: Optional[WorkspaceAPIMinerAllowlistRecord | Mapping[str, Any]],
    entropy_ledger: Optional[WorkspaceAPIEntropyLedgerRecord | Mapping[str, Any]],
    security_review: Optional[WorkspaceAPISecurityReviewRecord | Mapping[str, Any]],
) -> None:
    allowlist_record = allowlist if isinstance(allowlist, WorkspaceAPIMinerAllowlistRecord) else (
        WorkspaceAPIMinerAllowlistRecord.from_mapping(allowlist) if allowlist is not None else None
    )
    entropy_record = entropy_ledger if isinstance(entropy_ledger, WorkspaceAPIEntropyLedgerRecord) else (
        WorkspaceAPIEntropyLedgerRecord.from_mapping(entropy_ledger) if entropy_ledger is not None else None
    )
    security_record = security_review if isinstance(security_review, WorkspaceAPISecurityReviewRecord) else (
        WorkspaceAPISecurityReviewRecord.from_mapping(security_review) if security_review is not None else None
    )
    if allowlist_record is not None:
        allowlist_errors = validate_workspace_api_miner_allowlist_record(allowlist_record)
        errors.extend(f"allowlist invalid: {error}" for error in allowlist_errors)
        if record.allowlist_ref != allowlist_record.allowlist_id:
            errors.append("Workspace API beta allowlist_ref mismatch")
        if record.beta_scope_ref != allowlist_record.scope_ref:
            errors.append("Workspace API beta scope_ref mismatch")
        if set(record.allowed_methods) - set(allowlist_record.allowed_methods):
            errors.append("Workspace API beta allowed_methods exceed allowlist")
        if set(record.allowed_miner_refs) - {allowlist_record.miner_ref}:
            errors.append("Workspace API beta allowed_miner_refs must match allowlisted miner")
        if record.beta_enablement_claimed and not allowlist_record.beta_access_enabled:
            errors.append("Workspace API beta enablement requires allowlist beta_access_enabled")
    if entropy_record is not None:
        entropy_errors = validate_workspace_api_entropy_ledger_record(entropy_record)
        errors.extend(f"entropy_ledger invalid: {error}" for error in entropy_errors)
        if record.entropy_ledger_ref != entropy_record.ledger_id:
            errors.append("Workspace API beta entropy_ledger_ref mismatch")
        if allowlist_record is not None and entropy_record.miner_ref != allowlist_record.miner_ref:
            errors.append("Workspace API beta entropy miner_ref mismatch")
        if record.beta_enablement_claimed and not (entropy_record.accounting_live and entropy_record.budget_enforced):
            errors.append("Workspace API beta enablement requires live enforced entropy accounting")
        if record.beta_enablement_claimed and not entropy_record.expansion_allowed:
            errors.append("Workspace API beta enablement requires entropy expansion_allowed")
    if security_record is not None:
        security_errors = validate_workspace_api_security_review_record(
            security_record,
            allowlist=allowlist_record,
            entropy_ledger=entropy_record,
        )
        errors.extend(f"security_review invalid: {error}" for error in security_errors)
        if record.security_review_ref != security_record.review_id:
            errors.append("Workspace API beta security_review_ref mismatch")
        if record.beta_enablement_claimed and not security_record.security_review_passed:
            errors.append("Workspace API beta enablement requires passed security review")


def _append_public_field_errors(errors: list[str], fields: Sequence[str], label: str) -> None:
    for field in fields:
        lowered = str(field).lower()
        if lowered in PROTECTED_WORKSPACE_API_KEYS or any(marker in lowered for marker in PROTECTED_WORKSPACE_API_MARKERS):
            errors.append(f"{label} must not expose protected field: {field}")
            break


def _method_set(records: Sequence[WorkspaceAPIMethodContractRecord | Mapping[str, Any]]) -> set[str]:
    methods: set[str] = set()
    for record in records:
        method = record.method if isinstance(record, WorkspaceAPIMethodContractRecord) else str(record.get("method", ""))
        methods.add(method)
    return methods


def _validate_prefixed(errors: list[str], field_name: str, value: str, prefix: str) -> None:
    if not value.startswith(prefix):
        errors.append(f"{field_name} must be {prefix}-prefixed")


def _validate_sequence_prefixes(errors: list[str], field_name: str, values: Sequence[str], prefix: str) -> None:
    for value in values:
        if not value.startswith(prefix):
            errors.append(f"{field_name} values must be {prefix}-prefixed")
            break


def _protected_workspace_api_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_workspace_api_material(record))
    if not found:
        return []
    return ["Workspace API beta payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_workspace_api_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_WORKSPACE_API_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_workspace_api_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_workspace_api_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_WORKSPACE_API_MARKERS:
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
