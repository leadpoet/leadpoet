"""Phase 2 full component-market contracts.

P2.2 generalizes the P1.5 SOURCE_ADD launch track into local typed component
market records. The contracts stay inert: they do not open paid workflows,
execute submitted code, accept public components, mutate balances, or write to
production systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
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
from .source_add import verify_research_lab_source_add


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "component_market_fixtures.json"

COMPONENT_OUTPUT_REF_PREFIXES: tuple[str, ...] = (
    "artifact:",
    "component_output:",
    "evidence:",
    "metadata:",
    "receipt:",
    "schema:",
    "snapshot:",
    "trace:",
    "sha256:",
)

PROTECTED_COMPONENT_KEYS = {
    "api_key",
    "credential_value",
    "customer_email",
    "judge_prompt",
    "judge_prompts",
    "live_champion_code",
    "live_champion_prompt",
    "live_champion_weights",
    "model_weights",
    "password",
    "private_customer_data",
    "raw_content",
    "raw_credential",
    "raw_customer_data",
    "raw_evidence",
    "raw_snapshot",
    "raw_text",
    "sealed_eval",
    "sealed_eval_details",
    "secret",
    "token",
    "wallet_address",
}

PROTECTED_COMPONENT_MARKERS: tuple[str, ...] = (
    "api key",
    "judge prompt",
    "live champion code",
    "live champion prompt",
    "model weights",
    "password=",
    "private customer",
    "raw customer",
    "raw evidence",
    "raw snapshot",
    "sealed eval",
    "sealed judge prompt",
    "sk-live",
)


class ComponentMarketType(str, Enum):
    SOURCE_ADD = "source_add"
    EVIDENCE_ADAPTER = "evidence_adapter"
    PROMPT_TEMPLATE = "prompt_template"
    PARAMETER_POLICY = "parameter_policy"
    STRATEGY_MODULE = "strategy_module"
    EVAL_CASE = "eval_case"
    RED_TEAM_EXPLOIT = "red_team_exploit"


class ComponentReviewStatus(str, Enum):
    SUBMITTED_LOCAL = "submitted_local"
    SANDBOX_REVIEW_PLANNED = "sandbox_review_planned"
    MEASURED_NOT_ACCEPTED = "measured_not_accepted"
    ACCEPTABLE_PENDING_OWNER_GATE = "acceptable_pending_owner_gate"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class ComponentBountyState(str, Enum):
    CALCULATED_NOT_PAYABLE = "calculated_not_payable"
    REJECTED_NOT_PAYABLE = "rejected_not_payable"


class ComponentIPState(str, Enum):
    ACKNOWLEDGED_PENDING_ACCEPTANCE = "acknowledged_pending_acceptance"
    REJECTED = "rejected"


COMPONENT_MARKET_TYPES: tuple[str, ...] = tuple(component_type.value for component_type in ComponentMarketType)


@dataclass(frozen=True)
class ComponentTypeRegistryEntry:
    component_type: str
    display_name: str
    submission_schema_ref: str
    review_schema_ref: str
    allowed_patch_types: tuple[str, ...]
    accepts_code_bundle: bool = False
    requires_sandbox_review: bool = True
    allowed_output_ref_prefixes: tuple[str, ...] = COMPONENT_OUTPUT_REF_PREFIXES
    max_trial_cost_cents: int = 0
    enabled_for_paid_market: bool = False
    production_enabled: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentTypeRegistryEntry":
        return cls(
            component_type=str(data["component_type"]),
            display_name=str(data["display_name"]),
            submission_schema_ref=str(data["submission_schema_ref"]),
            review_schema_ref=str(data["review_schema_ref"]),
            allowed_patch_types=tuple(str(item) for item in data.get("allowed_patch_types", [])),
            accepts_code_bundle=bool(data.get("accepts_code_bundle", False)),
            requires_sandbox_review=bool(data.get("requires_sandbox_review", True)),
            allowed_output_ref_prefixes=tuple(
                str(item) for item in data.get("allowed_output_ref_prefixes", COMPONENT_OUTPUT_REF_PREFIXES)
            ),
            max_trial_cost_cents=int(data.get("max_trial_cost_cents", 0)),
            enabled_for_paid_market=bool(data.get("enabled_for_paid_market", False)),
            production_enabled=bool(data.get("production_enabled", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_patch_types"] = list(self.allowed_patch_types)
        data["allowed_output_ref_prefixes"] = list(self.allowed_output_ref_prefixes)
        return data


@dataclass(frozen=True)
class ComponentSubmissionRecord:
    submission_id: str
    component_type: str
    miner_ref: str
    artifact_ref: str
    artifact_hash: str
    component_interface_ref: str
    declared_output_refs: tuple[str, ...]
    fixture_refs: tuple[str, ...]
    registry_entry_ref: str
    code_bundle_hash: str = ""
    credential_policy: str = "no_credentials"
    credential_ref: str = ""
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    arbitrary_code_execution_enabled: bool = False
    live_network_enabled: bool = False
    submitted_for_public_market: bool = False
    production_review_enabled: bool = False
    component_enabled_in_market: bool = False
    live_champion_dependency_ref: str = ""
    sealed_eval_dependency_ref: str = ""
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentSubmissionRecord":
        return cls(
            submission_id=str(data["submission_id"]),
            component_type=str(data["component_type"]),
            miner_ref=str(data["miner_ref"]),
            artifact_ref=str(data["artifact_ref"]),
            artifact_hash=str(data["artifact_hash"]),
            component_interface_ref=str(data["component_interface_ref"]),
            declared_output_refs=tuple(str(item) for item in data.get("declared_output_refs", [])),
            fixture_refs=tuple(str(item) for item in data.get("fixture_refs", [])),
            registry_entry_ref=str(data["registry_entry_ref"]),
            code_bundle_hash=str(data.get("code_bundle_hash", "")),
            credential_policy=str(data.get("credential_policy", "no_credentials")),
            credential_ref=str(data.get("credential_ref", "")),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            arbitrary_code_execution_enabled=bool(data.get("arbitrary_code_execution_enabled", False)),
            live_network_enabled=bool(data.get("live_network_enabled", False)),
            submitted_for_public_market=bool(data.get("submitted_for_public_market", False)),
            production_review_enabled=bool(data.get("production_review_enabled", False)),
            component_enabled_in_market=bool(data.get("component_enabled_in_market", False)),
            live_champion_dependency_ref=str(data.get("live_champion_dependency_ref", "")),
            sealed_eval_dependency_ref=str(data.get("sealed_eval_dependency_ref", "")),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(
                data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)
            ),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["declared_output_refs"] = list(self.declared_output_refs)
        data["fixture_refs"] = list(self.fixture_refs)
        return data


@dataclass(frozen=True)
class ComponentReviewRecord:
    review_id: str
    submission_id: str
    component_type: str
    reviewer_ref: str
    trial_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    measured_trial_yield: float
    sandbox_job_ref: str
    status: str = ComponentReviewStatus.SANDBOX_REVIEW_PLANNED.value
    lab_only: bool = True
    sandboxed_execution_performed: bool = False
    arbitrary_code_executed: bool = False
    live_network_used: bool = False
    component_accepted: bool = False
    paid_market_enabled: bool = False
    human_acceptance_gate_passed: bool = False
    production_writes: bool = False
    review_notes_sanitized: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentReviewRecord":
        return cls(
            review_id=str(data["review_id"]),
            submission_id=str(data["submission_id"]),
            component_type=str(data["component_type"]),
            reviewer_ref=str(data["reviewer_ref"]),
            trial_refs=tuple(str(item) for item in data.get("trial_refs", [])),
            output_refs=tuple(str(item) for item in data.get("output_refs", [])),
            measured_trial_yield=float(data["measured_trial_yield"]),
            sandbox_job_ref=str(data["sandbox_job_ref"]),
            status=str(data.get("status", ComponentReviewStatus.SANDBOX_REVIEW_PLANNED.value)),
            lab_only=bool(data.get("lab_only", True)),
            sandboxed_execution_performed=bool(data.get("sandboxed_execution_performed", False)),
            arbitrary_code_executed=bool(data.get("arbitrary_code_executed", False)),
            live_network_used=bool(data.get("live_network_used", False)),
            component_accepted=bool(data.get("component_accepted", False)),
            paid_market_enabled=bool(data.get("paid_market_enabled", False)),
            human_acceptance_gate_passed=bool(data.get("human_acceptance_gate_passed", False)),
            production_writes=bool(data.get("production_writes", False)),
            review_notes_sanitized=str(data.get("review_notes_sanitized", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trial_refs"] = list(self.trial_refs)
        data["output_refs"] = list(self.output_refs)
        return data


@dataclass(frozen=True)
class ComponentBountyBand:
    band_ref: str
    component_type: str
    min_trial_yield: float
    max_trial_yield: Optional[float]
    bounty_cents: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentBountyBand":
        max_yield = data.get("max_trial_yield")
        return cls(
            band_ref=str(data["band_ref"]),
            component_type=str(data["component_type"]),
            min_trial_yield=float(data["min_trial_yield"]),
            max_trial_yield=None if max_yield is None else float(max_yield),
            bounty_cents=int(data["bounty_cents"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentBountyRecord:
    bounty_ref: str
    component_type: str
    submission_id: str
    review_id: str
    miner_ref: str
    measured_trial_yield: float
    band_ref: str
    bounty_cents: int
    state: str = ComponentBountyState.CALCULATED_NOT_PAYABLE.value
    payment_enabled: bool = False
    settlement_ref: str = ""
    paid_at: str = ""
    balance_mutation_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentBountyRecord":
        return cls(
            bounty_ref=str(data["bounty_ref"]),
            component_type=str(data["component_type"]),
            submission_id=str(data["submission_id"]),
            review_id=str(data["review_id"]),
            miner_ref=str(data["miner_ref"]),
            measured_trial_yield=float(data["measured_trial_yield"]),
            band_ref=str(data["band_ref"]),
            bounty_cents=int(data["bounty_cents"]),
            state=str(data.get("state", ComponentBountyState.CALCULATED_NOT_PAYABLE.value)),
            payment_enabled=bool(data.get("payment_enabled", False)),
            settlement_ref=str(data.get("settlement_ref", "")),
            paid_at=str(data.get("paid_at", "")),
            balance_mutation_enabled=bool(data.get("balance_mutation_enabled", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentIPAssignmentRecord:
    assignment_ref: str
    component_type: str
    submission_id: str
    miner_ref: str
    assigned_to_ref: str
    rights_scope: str
    state: str = ComponentIPState.ACKNOWLEDGED_PENDING_ACCEPTANCE.value
    effective_on_acceptance_only: bool = True
    component_publicly_accepted: bool = False
    signature_ref: str = "signature:pending"
    public_release_authorized: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentIPAssignmentRecord":
        return cls(
            assignment_ref=str(data["assignment_ref"]),
            component_type=str(data["component_type"]),
            submission_id=str(data["submission_id"]),
            miner_ref=str(data["miner_ref"]),
            assigned_to_ref=str(data["assigned_to_ref"]),
            rights_scope=str(data["rights_scope"]),
            state=str(data.get("state", ComponentIPState.ACKNOWLEDGED_PENDING_ACCEPTANCE.value)),
            effective_on_acceptance_only=bool(data.get("effective_on_acceptance_only", True)),
            component_publicly_accepted=bool(data.get("component_publicly_accepted", False)),
            signature_ref=str(data.get("signature_ref", "signature:pending")),
            public_release_authorized=bool(data.get("public_release_authorized", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_component_type_registry_entry(
    record: ComponentTypeRegistryEntry | Mapping[str, Any],
) -> list[str]:
    if not isinstance(record, ComponentTypeRegistryEntry):
        record = ComponentTypeRegistryEntry.from_mapping(record)
    errors: list[str] = []
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if not record.display_name:
        errors.append("component registry entry requires display_name")
    if not record.submission_schema_ref.startswith("schema:"):
        errors.append("submission_schema_ref must be schema:-prefixed")
    if not record.review_schema_ref.startswith("schema:"):
        errors.append("review_schema_ref must be schema:-prefixed")
    if not record.allowed_patch_types:
        errors.append("allowed_patch_types must not be empty")
    if not record.allowed_output_ref_prefixes:
        errors.append("allowed_output_ref_prefixes must not be empty")
    if record.max_trial_cost_cents <= 0:
        errors.append("max_trial_cost_cents must be positive")
    if record.enabled_for_paid_market:
        errors.append("full component market paid workflow must remain disabled")
    if record.production_enabled:
        errors.append("component registry entry must not be production_enabled")
    if not record.local_only:
        errors.append("component registry entry must remain local_only")
    return errors


def validate_component_submission_record(
    record: ComponentSubmissionRecord | Mapping[str, Any],
    registry_entries: Sequence[ComponentTypeRegistryEntry | Mapping[str, Any]] = (),
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_component_payload_errors(raw)
    if not isinstance(record, ComponentSubmissionRecord):
        record = ComponentSubmissionRecord.from_mapping(record)
    registry = _registry_by_type(registry_entries)
    errors: list[str] = list(raw_errors)
    if not record.submission_id:
        errors.append("component submission requires submission_id")
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if registry and record.component_type not in registry:
        errors.append("component submission component_type is not in registry")
    if not record.miner_ref.startswith("miner:"):
        errors.append("miner_ref must be miner:-prefixed")
    if not record.artifact_ref.startswith("artifact:"):
        errors.append("artifact_ref must be artifact:-prefixed")
    if not record.artifact_hash.startswith("sha256:"):
        errors.append("artifact_hash must be sha256:-prefixed")
    if not record.component_interface_ref.startswith("component_interface:"):
        errors.append("component_interface_ref must be component_interface:-prefixed")
    if not record.registry_entry_ref.startswith("component_registry:"):
        errors.append("registry_entry_ref must be component_registry:-prefixed")
    if not record.declared_output_refs:
        errors.append("declared_output_refs must not be empty")
    allowed_prefixes = registry.get(record.component_type, _DefaultRegistry()).allowed_output_ref_prefixes
    bad_refs = [
        output_ref
        for output_ref in record.declared_output_refs
        if not any(output_ref.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if bad_refs:
        errors.append("declared_output_refs contain non-ref/raw values: " + ", ".join(bad_refs))
    if not record.fixture_refs:
        errors.append("fixture_refs must not be empty")
    if record.code_bundle_hash and not record.code_bundle_hash.startswith("sha256:"):
        errors.append("code_bundle_hash must be sha256:-prefixed when present")
    if record.credential_policy not in {"no_credentials", "credential_ref_only"}:
        errors.append("credential_policy must be no_credentials or credential_ref_only")
    if record.credential_policy == "no_credentials" and record.credential_ref:
        errors.append("credential_ref must be empty when credential_policy=no_credentials")
    if record.credential_ref and not record.credential_ref.startswith("credential_ref:"):
        errors.append("credential_ref must be credential_ref:-prefixed")
    if not record.protected_material_flags_checked:
        errors.append("component submission must check protected-material flags")
    protected_flags = _submission_protected_flags(record)
    if protected_flags:
        errors.append("component submission must not contain protected material flags: " + ", ".join(protected_flags))
    if record.live_champion_dependency_ref:
        errors.append("component submission must not depend on live champion material")
    if record.sealed_eval_dependency_ref:
        errors.append("component submission must not depend on sealed eval material")
    if record.arbitrary_code_execution_enabled:
        errors.append("arbitrary_code_execution_enabled must remain false")
    if record.live_network_enabled:
        errors.append("live_network_enabled must remain false")
    if record.submitted_for_public_market:
        errors.append("submitted_for_public_market must remain false while Phase 2 is local-only")
    if record.production_review_enabled:
        errors.append("production_review_enabled must remain false")
    if record.component_enabled_in_market:
        errors.append("component_enabled_in_market must remain false")
    if not record.local_only:
        errors.append("component submission must remain local_only")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.submission_id,
                artifact_type="component_submission",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                contains_live_champion_ip=record.contains_live_champion_ip,
                contains_sealed_eval_details=record.contains_sealed_eval_details,
                contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
                contains_private_customer_data=record.contains_private_customer_data,
                contains_judge_prompts=record.contains_judge_prompts,
                reason="component submissions remain private until reviewed and accepted",
            )
        )
    )
    if record.artifact_release_state != ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value:
        errors.append("component submissions must remain private until acceptance gate")
    return errors


def validate_component_review_record(
    record: ComponentReviewRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_component_payload_errors(raw)
    if not isinstance(record, ComponentReviewRecord):
        record = ComponentReviewRecord.from_mapping(record)
    errors: list[str] = list(raw_errors)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if record.status not in {status.value for status in ComponentReviewStatus}:
        errors.append(f"unknown component review status: {record.status}")
    if not record.review_id:
        errors.append("component review requires review_id")
    if not record.submission_id:
        errors.append("component review requires submission_id")
    if not record.reviewer_ref.startswith("reviewer:"):
        errors.append("reviewer_ref must be reviewer:-prefixed")
    if not record.trial_refs:
        errors.append("trial_refs must not be empty")
    if not record.output_refs:
        errors.append("output_refs must not be empty")
    if record.measured_trial_yield < 0:
        errors.append("measured_trial_yield must be non-negative")
    if not record.sandbox_job_ref.startswith("sandbox_job:"):
        errors.append("sandbox_job_ref must be sandbox_job:-prefixed")
    if not record.lab_only:
        errors.append("component review must remain lab_only")
    if record.sandboxed_execution_performed:
        errors.append("P2.2 review records must not claim sandboxed execution performed")
    if record.arbitrary_code_executed:
        errors.append("arbitrary code must not execute in P2.2 local contracts")
    if record.live_network_used:
        errors.append("component review must not use live network")
    if record.component_accepted:
        errors.append("component_accepted must remain false while Phase 2 is local-only")
    if record.paid_market_enabled:
        errors.append("paid_market_enabled must remain false")
    if record.human_acceptance_gate_passed:
        errors.append("human_acceptance_gate_passed must remain false in local P2.2")
    if record.production_writes:
        errors.append("component review must not perform production_writes")
    return errors


def validate_component_bounty_band(record: ComponentBountyBand | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, ComponentBountyBand):
        record = ComponentBountyBand.from_mapping(record)
    errors: list[str] = []
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if record.min_trial_yield < 0:
        errors.append("min_trial_yield must be non-negative")
    if record.max_trial_yield is not None and record.max_trial_yield <= record.min_trial_yield:
        errors.append("max_trial_yield must exceed min_trial_yield")
    if record.bounty_cents < 0:
        errors.append("bounty_cents must be non-negative")
    return errors


def select_component_bounty_band(
    *,
    component_type: str,
    measured_trial_yield: float,
    bands: Sequence[ComponentBountyBand | Mapping[str, Any]],
) -> ComponentBountyBand:
    normalized = [band if isinstance(band, ComponentBountyBand) else ComponentBountyBand.from_mapping(band) for band in bands]
    matches = [
        band
        for band in normalized
        if band.component_type == component_type
        and measured_trial_yield >= band.min_trial_yield
        and (band.max_trial_yield is None or measured_trial_yield < band.max_trial_yield)
    ]
    if not matches:
        raise ValueError("no component bounty band matches measured trial yield")
    return sorted(matches, key=lambda band: (-band.min_trial_yield, band.band_ref))[0]


def build_component_bounty_record(
    *,
    submission: ComponentSubmissionRecord | Mapping[str, Any],
    review: ComponentReviewRecord | Mapping[str, Any],
    bands: Sequence[ComponentBountyBand | Mapping[str, Any]],
    registry_entries: Sequence[ComponentTypeRegistryEntry | Mapping[str, Any]] = (),
) -> ComponentBountyRecord:
    if not isinstance(submission, ComponentSubmissionRecord):
        submission = ComponentSubmissionRecord.from_mapping(submission)
    if not isinstance(review, ComponentReviewRecord):
        review = ComponentReviewRecord.from_mapping(review)
    errors = validate_component_submission_record(submission, registry_entries)
    errors.extend(validate_component_review_record(review))
    if submission.component_type != review.component_type:
        errors.append("component bounty requires matching submission/review component_type")
    if submission.submission_id != review.submission_id:
        errors.append("component bounty requires matching submission_id")
    if errors:
        raise ValueError("; ".join(errors))
    band = select_component_bounty_band(
        component_type=submission.component_type,
        measured_trial_yield=review.measured_trial_yield,
        bands=bands,
    )
    payload = {
        "component_type": submission.component_type,
        "submission_id": submission.submission_id,
        "review_id": review.review_id,
        "miner_ref": submission.miner_ref,
        "measured_trial_yield": review.measured_trial_yield,
        "band_ref": band.band_ref,
        "bounty_cents": band.bounty_cents,
    }
    record = ComponentBountyRecord(
        bounty_ref="component_bounty:" + sha256_json(payload).split(":", 1)[1][:16],
        **payload,
    )
    bounty_errors = validate_component_bounty_record(record)
    if bounty_errors:
        raise ValueError("; ".join(bounty_errors))
    return record


def validate_component_bounty_record(record: ComponentBountyRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, ComponentBountyRecord):
        record = ComponentBountyRecord.from_mapping(record)
    errors: list[str] = []
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if record.state not in {state.value for state in ComponentBountyState}:
        errors.append(f"unknown component bounty state: {record.state}")
    if record.measured_trial_yield < 0:
        errors.append("measured_trial_yield must be non-negative")
    if record.bounty_cents < 0:
        errors.append("bounty_cents must be non-negative")
    if record.payment_enabled:
        errors.append("component bounty payment_enabled must remain false")
    if record.settlement_ref:
        errors.append("component bounty settlement_ref must remain empty")
    if record.paid_at:
        errors.append("component bounty paid_at must remain empty")
    if record.balance_mutation_enabled:
        errors.append("component bounty must not enable balance mutation")
    return errors


def validate_component_ip_assignment(record: ComponentIPAssignmentRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_component_payload_errors(raw)
    if not isinstance(record, ComponentIPAssignmentRecord):
        record = ComponentIPAssignmentRecord.from_mapping(record)
    errors: list[str] = list(raw_errors)
    if record.component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown component_type: {record.component_type}")
    if record.state not in {state.value for state in ComponentIPState}:
        errors.append(f"unknown component IP state: {record.state}")
    if not record.assignment_ref:
        errors.append("component IP assignment requires assignment_ref")
    if not record.submission_id:
        errors.append("component IP assignment requires submission_id")
    if not record.miner_ref.startswith("miner:"):
        errors.append("miner_ref must be miner:-prefixed")
    if not record.assigned_to_ref:
        errors.append("assigned_to_ref is required")
    if "component_market_component" not in record.rights_scope:
        errors.append("rights_scope must cover component_market_component")
    if not record.effective_on_acceptance_only:
        errors.append("component IP assignment must be effective on acceptance only")
    if record.component_publicly_accepted:
        errors.append("component IP assignment must not mark component_publicly_accepted")
    if record.signature_ref != "signature:pending":
        errors.append("component IP assignment must remain signature:pending")
    if record.public_release_authorized:
        errors.append("component IP assignment must not authorize public release")
    return errors


def verify_research_lab_component_market(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    source_add_summary = verify_research_lab_source_add()
    fixture = _load_fixture(Path(fixture_path))

    registry_entries = [ComponentTypeRegistryEntry.from_mapping(item) for item in fixture["registry_entries"]]
    for entry in registry_entries:
        _assert(not validate_component_type_registry_entry(entry), f"registry entry validates: {entry.component_type}")
    _assert(
        len({entry.component_type for entry in registry_entries}) > 1
        and ComponentMarketType.SOURCE_ADD.value in {entry.component_type for entry in registry_entries},
        "component market registry extends beyond SOURCE_ADD",
    )
    for record in fixture["invalid_registry_entries"]:
        errors = validate_component_type_registry_entry(record)
        _assert(errors, f"invalid registry entry fails: {record['component_type']}")
        _assert_expected_error(errors, record)

    submission = ComponentSubmissionRecord.from_mapping(fixture["submission"])
    _assert(
        not validate_component_submission_record(submission, registry_entries),
        "component submission validates",
    )
    for record in fixture["invalid_submissions"]:
        errors = validate_component_submission_record(record, registry_entries)
        _assert(errors, f"invalid component submission fails: {record['submission_id']}")
        _assert_expected_error(errors, record)

    review = ComponentReviewRecord.from_mapping(fixture["review"])
    _assert(not validate_component_review_record(review), "component review validates")
    for record in fixture["invalid_reviews"]:
        errors = validate_component_review_record(record)
        _assert(errors, f"invalid component review fails: {record['review_id']}")
        _assert_expected_error(errors, record)
    unsafe_guard_errors = validate_component_review_record(review, guards=fixture["unsafe_workflow_guards"])
    _assert(unsafe_guard_errors, "unsafe Phase 2 workflow guards block component review")

    bands = [ComponentBountyBand.from_mapping(item) for item in fixture["bounty_bands"]]
    for band in bands:
        _assert(not validate_component_bounty_band(band), f"component bounty band validates: {band.band_ref}")
    bounty = build_component_bounty_record(
        submission=submission,
        review=review,
        bands=bands,
        registry_entries=registry_entries,
    )
    _assert(bounty.band_ref == fixture["expected_bounty_band_ref"], "component bounty band selected")
    _assert(not validate_component_bounty_record(bounty), "component bounty record validates")
    for record in fixture["invalid_bounty_records"]:
        errors = validate_component_bounty_record(record)
        _assert(errors, f"invalid component bounty fails: {record['bounty_ref']}")
        _assert_expected_error(errors, record)

    ip_assignment = ComponentIPAssignmentRecord.from_mapping(fixture["ip_assignment"])
    _assert(not validate_component_ip_assignment(ip_assignment), "component IP assignment validates")
    for record in fixture["invalid_ip_assignments"]:
        errors = validate_component_ip_assignment(record)
        _assert(errors, f"invalid component IP assignment fails: {record['assignment_ref']}")
        _assert_expected_error(errors, record)

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "source_add_adapter_id": source_add_summary["adapter_id"],
        "component_types": [entry.component_type for entry in registry_entries],
        "submission_id": submission.submission_id,
        "measured_trial_yield": review.measured_trial_yield,
        "bounty_ref": bounty.bounty_ref,
        "bounty_band_ref": bounty.band_ref,
        "bounty_cents": bounty.bounty_cents,
    }


@dataclass(frozen=True)
class _DefaultRegistry:
    allowed_output_ref_prefixes: tuple[str, ...] = COMPONENT_OUTPUT_REF_PREFIXES


def _registry_by_type(
    registry_entries: Sequence[ComponentTypeRegistryEntry | Mapping[str, Any]],
) -> dict[str, ComponentTypeRegistryEntry]:
    return {
        entry.component_type: entry
        for entry in (
            item if isinstance(item, ComponentTypeRegistryEntry) else ComponentTypeRegistryEntry.from_mapping(item)
            for item in registry_entries
        )
    }


def _submission_protected_flags(record: ComponentSubmissionRecord) -> list[str]:
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


def _protected_component_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_component_material(record))
    if not found:
        return []
    return ["component market payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_component_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_COMPONENT_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_component_material(nested, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_protected_component_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_COMPONENT_MARKERS:
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
