"""Phase 1 recruiting/admin records.

P1.8 models white-glove recruiting inputs for the first 20-50 crypto-native
domain experts: template briefs, sanitized candidate records, local cohort
plans, and onboarding checklists. It is intentionally inert: no outreach
automation, email sending, account creation, public workflow access, CRM sync,
or production writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .engine_v1 import PatchType
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


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "recruiting_admin_fixtures.json"

RECRUITING_TARGET_MIN = 20
RECRUITING_TARGET_MAX = 50

SENSITIVE_RECRUITING_KEYS = {
    "email",
    "email_address",
    "phone",
    "phone_number",
    "legal_name",
    "home_address",
    "street_address",
    "raw_resume",
    "resume_text",
    "linkedin_url",
    "twitter_dm",
    "telegram_handle",
    "discord_handle",
    "wallet_address",
    "bank_account",
    "passport_number",
    "ssn",
    "dob",
    "date_of_birth",
    "private_notes",
    "raw_contact",
    "contact_email",
    "contact_phone",
}

SENSITIVE_RECRUITING_MARKERS = (
    "@",
    "passport",
    "ssn",
    "date of birth",
    "home address",
    "private key",
    "telegram:",
    "discord:",
    "email:",
    "phone:",
)


class CandidateStatus(str, Enum):
    SOURCED_LOCAL_ONLY = "sourced_local_only"
    SHORTLISTED_LOCAL_ONLY = "shortlisted_local_only"
    MANUAL_REVIEW_READY = "manual_review_ready"
    ONBOARDING_CHECKLIST_DRAFTED = "onboarding_checklist_drafted"
    BLOCKED = "blocked"


class LegalGateStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING_REVIEW = "pending_review"
    BLOCKED = "blocked"
    APPROVED_RECORDED_ONLY = "approved_recorded_only"


class ContributorTermsStatus(str, Enum):
    NOT_SENT = "not_sent"
    PENDING_MANUAL_REVIEW = "pending_manual_review"
    SIGNED_RECORDED_ONLY = "signed_recorded_only"
    BLOCKED = "blocked"


class OnboardingState(str, Enum):
    DRAFT_LOCAL_ONLY = "draft_local_only"
    WAITING_ON_LEGAL = "waiting_on_legal"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TemplateBriefRecord:
    brief_id: str
    target_domain: str
    island: str
    source_failure_board_ref: str
    source_map_cell_ref: str
    problem_statement: str
    domain_context: str
    desired_evidence_refs: tuple[str, ...]
    suggested_patch_types: tuple[str, ...]
    success_metric_refs: tuple[str, ...]
    example_public_refs: tuple[str, ...]
    operator_prompt: str
    protected_material_flags_checked: bool = True
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TemplateBriefRecord":
        return cls(
            brief_id=str(data["brief_id"]),
            target_domain=str(data["target_domain"]),
            island=str(data["island"]),
            source_failure_board_ref=str(data["source_failure_board_ref"]),
            source_map_cell_ref=str(data["source_map_cell_ref"]),
            problem_statement=str(data["problem_statement"]),
            domain_context=str(data["domain_context"]),
            desired_evidence_refs=tuple(str(item) for item in data.get("desired_evidence_refs", [])),
            suggested_patch_types=tuple(str(item) for item in data.get("suggested_patch_types", [])),
            success_metric_refs=tuple(str(item) for item in data.get("success_metric_refs", [])),
            example_public_refs=tuple(str(item) for item in data.get("example_public_refs", [])),
            operator_prompt=str(data["operator_prompt"]),
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
        data["desired_evidence_refs"] = list(self.desired_evidence_refs)
        data["suggested_patch_types"] = list(self.suggested_patch_types)
        data["success_metric_refs"] = list(self.success_metric_refs)
        data["example_public_refs"] = list(self.example_public_refs)
        return data


@dataclass(frozen=True)
class RecruitingCandidateRecord:
    candidate_ref: str
    source_ref: str
    source_channel: str
    status: str
    expertise_tags: tuple[str, ...]
    target_domain_refs: tuple[str, ...]
    public_profile_ref: str
    sanitized_notes: str
    crypto_native: bool = True
    pseudonymous_ref_only: bool = True
    unnecessary_sensitive_data_present: bool = False
    outreach_automation_enabled: bool = False
    email_sent: bool = False
    dm_sent: bool = False
    crm_sync_enabled: bool = False
    account_created: bool = False
    workflow_access_enabled: bool = False
    protected_material_flags_checked: bool = True
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecruitingCandidateRecord":
        return cls(
            candidate_ref=str(data["candidate_ref"]),
            source_ref=str(data["source_ref"]),
            source_channel=str(data["source_channel"]),
            status=str(data["status"]),
            expertise_tags=tuple(str(item) for item in data.get("expertise_tags", [])),
            target_domain_refs=tuple(str(item) for item in data.get("target_domain_refs", [])),
            public_profile_ref=str(data["public_profile_ref"]),
            sanitized_notes=str(data.get("sanitized_notes", "")),
            crypto_native=bool(data.get("crypto_native", True)),
            pseudonymous_ref_only=bool(data.get("pseudonymous_ref_only", True)),
            unnecessary_sensitive_data_present=bool(data.get("unnecessary_sensitive_data_present", False)),
            outreach_automation_enabled=bool(data.get("outreach_automation_enabled", False)),
            email_sent=bool(data.get("email_sent", False)),
            dm_sent=bool(data.get("dm_sent", False)),
            crm_sync_enabled=bool(data.get("crm_sync_enabled", False)),
            account_created=bool(data.get("account_created", False)),
            workflow_access_enabled=bool(data.get("workflow_access_enabled", False)),
            protected_material_flags_checked=bool(data.get("protected_material_flags_checked", True)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expertise_tags"] = list(self.expertise_tags)
        data["target_domain_refs"] = list(self.target_domain_refs)
        return data


@dataclass(frozen=True)
class RecruitingCohortPlanRecord:
    cohort_id: str
    target_min_experts: int
    target_max_experts: int
    target_profile: str
    candidate_refs: tuple[str, ...]
    template_brief_refs: tuple[str, ...]
    local_only: bool = True
    outreach_automation_enabled: bool = False
    email_sequence_enabled: bool = False
    dm_sequence_enabled: bool = False
    crm_sync_enabled: bool = False
    account_creation_enabled: bool = False
    public_workflow_open: bool = False
    production_writes: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecruitingCohortPlanRecord":
        return cls(
            cohort_id=str(data["cohort_id"]),
            target_min_experts=int(data["target_min_experts"]),
            target_max_experts=int(data["target_max_experts"]),
            target_profile=str(data["target_profile"]),
            candidate_refs=tuple(str(item) for item in data.get("candidate_refs", [])),
            template_brief_refs=tuple(str(item) for item in data.get("template_brief_refs", [])),
            local_only=bool(data.get("local_only", True)),
            outreach_automation_enabled=bool(data.get("outreach_automation_enabled", False)),
            email_sequence_enabled=bool(data.get("email_sequence_enabled", False)),
            dm_sequence_enabled=bool(data.get("dm_sequence_enabled", False)),
            crm_sync_enabled=bool(data.get("crm_sync_enabled", False)),
            account_creation_enabled=bool(data.get("account_creation_enabled", False)),
            public_workflow_open=bool(data.get("public_workflow_open", False)),
            production_writes=bool(data.get("production_writes", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_refs"] = list(self.candidate_refs)
        data["template_brief_refs"] = list(self.template_brief_refs)
        return data


@dataclass(frozen=True)
class OnboardingChecklistRecord:
    checklist_id: str
    candidate_ref: str
    assigned_template_brief_refs: tuple[str, ...]
    contributor_terms_status: str
    ip_assignment_status: str
    trajectory_rights_status: str
    distillation_rights_status: str
    pii_review_status: str
    legal_gate_status: str
    onboarding_state: str = OnboardingState.DRAFT_LOCAL_ONLY.value
    local_only: bool = True
    contributor_enabled: bool = False
    legal_gate_bypassed: bool = False
    account_created: bool = False
    workflow_access_enabled: bool = False
    public_miner_workflow_enabled: bool = False
    outreach_automation_enabled: bool = False
    email_sent: bool = False
    production_writes: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OnboardingChecklistRecord":
        return cls(
            checklist_id=str(data["checklist_id"]),
            candidate_ref=str(data["candidate_ref"]),
            assigned_template_brief_refs=tuple(str(item) for item in data.get("assigned_template_brief_refs", [])),
            contributor_terms_status=str(data["contributor_terms_status"]),
            ip_assignment_status=str(data["ip_assignment_status"]),
            trajectory_rights_status=str(data["trajectory_rights_status"]),
            distillation_rights_status=str(data["distillation_rights_status"]),
            pii_review_status=str(data["pii_review_status"]),
            legal_gate_status=str(data["legal_gate_status"]),
            onboarding_state=str(data.get("onboarding_state", OnboardingState.DRAFT_LOCAL_ONLY.value)),
            local_only=bool(data.get("local_only", True)),
            contributor_enabled=bool(data.get("contributor_enabled", False)),
            legal_gate_bypassed=bool(data.get("legal_gate_bypassed", False)),
            account_created=bool(data.get("account_created", False)),
            workflow_access_enabled=bool(data.get("workflow_access_enabled", False)),
            public_miner_workflow_enabled=bool(data.get("public_miner_workflow_enabled", False)),
            outreach_automation_enabled=bool(data.get("outreach_automation_enabled", False)),
            email_sent=bool(data.get("email_sent", False)),
            production_writes=bool(data.get("production_writes", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assigned_template_brief_refs"] = list(self.assigned_template_brief_refs)
        return data


def validate_template_brief_record(record: TemplateBriefRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _sensitive_payload_errors(record)
    if not isinstance(record, TemplateBriefRecord):
        record = TemplateBriefRecord.from_mapping(record)
    errors = list(raw_errors)
    if not record.target_domain:
        errors.append("template brief requires target_domain")
    if not record.source_failure_board_ref:
        errors.append("template brief requires source_failure_board_ref")
    if not record.source_map_cell_ref:
        errors.append("template brief requires source_map_cell_ref")
    if not record.problem_statement:
        errors.append("template brief requires problem_statement")
    if not record.desired_evidence_refs:
        errors.append("template brief requires desired_evidence_refs")
    if not record.suggested_patch_types:
        errors.append("template brief requires suggested_patch_types")
    unknown_patch_types = [patch_type for patch_type in record.suggested_patch_types if patch_type not in {p.value for p in PatchType}]
    if unknown_patch_types:
        errors.append("template brief has unknown suggested_patch_types: " + ", ".join(unknown_patch_types))
    if not record.success_metric_refs:
        errors.append("template brief requires success_metric_refs")
    if not record.example_public_refs:
        errors.append("template brief requires example_public_refs")
    if not record.protected_material_flags_checked:
        errors.append("template brief must check protected-material flags")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.brief_id,
                artifact_type="map_projection",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                contains_live_champion_ip=record.contains_live_champion_ip,
                contains_sealed_eval_details=record.contains_sealed_eval_details,
                contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
                contains_private_customer_data=record.contains_private_customer_data,
                contains_judge_prompts=record.contains_judge_prompts,
                reason="P1.8 template briefs are sanitized public recruiting/admin artifacts",
            )
        )
    )
    return errors


def validate_recruiting_candidate_record(record: RecruitingCandidateRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _sensitive_payload_errors(record)
    if not isinstance(record, RecruitingCandidateRecord):
        record = RecruitingCandidateRecord.from_mapping(record)
    errors = list(raw_errors)
    if not record.candidate_ref.startswith("candidate:"):
        errors.append("candidate_ref must be a pseudonymous candidate: ref")
    if "@" in record.candidate_ref:
        errors.append("candidate_ref must not contain direct contact data")
    if record.status not in {status.value for status in CandidateStatus}:
        errors.append(f"unknown candidate status: {record.status}")
    if not record.crypto_native:
        errors.append("P1.8 launch recruiting is limited to crypto-native experts")
    if not record.pseudonymous_ref_only:
        errors.append("candidate record must be pseudonymous/ref-only")
    if not record.expertise_tags:
        errors.append("candidate record requires expertise_tags")
    if not record.target_domain_refs:
        errors.append("candidate record requires target_domain_refs")
    if not record.public_profile_ref:
        errors.append("candidate record requires public_profile_ref")
    if len(record.sanitized_notes) > 500:
        errors.append("candidate sanitized_notes must be concise")
    if record.unnecessary_sensitive_data_present:
        errors.append("candidate record must not store unnecessary sensitive data")
    if record.outreach_automation_enabled:
        errors.append("P1.8 must not enable outreach automation")
    if record.email_sent:
        errors.append("P1.8 must not send email")
    if record.dm_sent:
        errors.append("P1.8 must not send DMs")
    if record.crm_sync_enabled:
        errors.append("P1.8 must not sync CRM records")
    if record.account_created:
        errors.append("P1.8 must not create accounts")
    if record.workflow_access_enabled:
        errors.append("P1.8 must not grant workflow access")
    if not record.protected_material_flags_checked:
        errors.append("candidate record must check protected-material flags")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.candidate_ref,
                artifact_type="candidate_admin_record",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                reason="P1.8 candidate records are private local admin records",
            )
        )
    )
    return errors


def validate_recruiting_cohort_plan_record(
    record: RecruitingCohortPlanRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw_errors = _sensitive_payload_errors(record)
    if not isinstance(record, RecruitingCohortPlanRecord):
        record = RecruitingCohortPlanRecord.from_mapping(record)
    errors = list(raw_errors)
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.target_min_experts < RECRUITING_TARGET_MIN:
        errors.append("cohort target_min_experts must be at least 20")
    if record.target_max_experts > RECRUITING_TARGET_MAX:
        errors.append("cohort target_max_experts must be at most 50")
    if record.target_min_experts > record.target_max_experts:
        errors.append("cohort target_min_experts must not exceed target_max_experts")
    if "crypto" not in record.target_profile.lower():
        errors.append("cohort target_profile must require crypto-native expertise")
    if not record.candidate_refs:
        errors.append("cohort plan requires candidate_refs")
    if not record.template_brief_refs:
        errors.append("cohort plan requires template_brief_refs")
    if not record.local_only:
        errors.append("P1.8 cohort plan must remain local_only")
    if record.outreach_automation_enabled:
        errors.append("P1.8 must not enable outreach automation")
    if record.email_sequence_enabled:
        errors.append("P1.8 must not enable email sequences")
    if record.dm_sequence_enabled:
        errors.append("P1.8 must not enable DM sequences")
    if record.crm_sync_enabled:
        errors.append("P1.8 must not enable CRM sync")
    if record.account_creation_enabled:
        errors.append("P1.8 must not enable account creation")
    if record.public_workflow_open:
        errors.append("P1.8 must not open public workflows")
    if record.production_writes:
        errors.append("P1.8 must not write production data")
    return errors


def validate_onboarding_checklist_record(record: OnboardingChecklistRecord | Mapping[str, Any]) -> list[str]:
    raw_errors = _sensitive_payload_errors(record)
    if not isinstance(record, OnboardingChecklistRecord):
        record = OnboardingChecklistRecord.from_mapping(record)
    errors = list(raw_errors)
    if record.onboarding_state not in {state.value for state in OnboardingState}:
        errors.append(f"unknown onboarding_state: {record.onboarding_state}")
    if record.contributor_terms_status not in {status.value for status in ContributorTermsStatus}:
        errors.append(f"unknown contributor_terms_status: {record.contributor_terms_status}")
    if record.legal_gate_status not in {status.value for status in LegalGateStatus}:
        errors.append(f"unknown legal_gate_status: {record.legal_gate_status}")
    for field in (
        "ip_assignment_status",
        "trajectory_rights_status",
        "distillation_rights_status",
        "pii_review_status",
    ):
        value = getattr(record, field)
        if value not in {status.value for status in LegalGateStatus}:
            errors.append(f"unknown {field}: {value}")
    if not record.assigned_template_brief_refs:
        errors.append("onboarding checklist requires assigned_template_brief_refs")
    if not record.local_only:
        errors.append("P1.8 onboarding checklist must remain local_only")
    if record.legal_gate_bypassed:
        errors.append("legal gate status must be tracked, not bypassed")
    if record.contributor_enabled:
        errors.append("P1.8 onboarding checklist must not enable contributor access")
    if record.account_created:
        errors.append("P1.8 onboarding checklist must not create accounts")
    if record.workflow_access_enabled:
        errors.append("P1.8 onboarding checklist must not grant workflow access")
    if record.public_miner_workflow_enabled:
        errors.append("P1.8 onboarding checklist must not open public miner workflows")
    if record.outreach_automation_enabled:
        errors.append("P1.8 onboarding checklist must not enable outreach automation")
    if record.email_sent:
        errors.append("P1.8 onboarding checklist must not send email")
    if record.production_writes:
        errors.append("P1.8 onboarding checklist must not write production data")
    return errors


def verify_research_lab_recruiting_admin(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    map_summary = verify_research_lab_research_map()
    fixture = _load_fixture(Path(fixture_path))

    template_briefs = [TemplateBriefRecord.from_mapping(item) for item in fixture["template_briefs"]]
    for record in template_briefs:
        _assert(not validate_template_brief_record(record), f"template brief validates: {record.brief_id}")
    for record in fixture["invalid_template_briefs"]:
        errors = validate_template_brief_record(record)
        _assert(errors, f"invalid template brief fails: {record['brief_id']}")
        _assert_expected_error(errors, record)

    candidates = [RecruitingCandidateRecord.from_mapping(item) for item in fixture["candidates"]]
    for record in candidates:
        _assert(not validate_recruiting_candidate_record(record), f"candidate validates: {record.candidate_ref}")
    for record in fixture["invalid_candidates"]:
        errors = validate_recruiting_candidate_record(record)
        _assert(errors, f"invalid candidate fails: {record['candidate_ref']}")
        _assert_expected_error(errors, record)

    cohort = RecruitingCohortPlanRecord.from_mapping(fixture["cohort_plan"])
    _assert(not validate_recruiting_cohort_plan_record(cohort), "cohort plan validates")
    for record in fixture["invalid_cohort_plans"]:
        errors = validate_recruiting_cohort_plan_record(record)
        _assert(errors, f"invalid cohort plan fails: {record['cohort_id']}")
        _assert_expected_error(errors, record)
    unsafe_errors = validate_recruiting_cohort_plan_record(cohort, guards=fixture["unsafe_workflow_guards"])
    _assert(unsafe_errors, "unsafe workflow guards block recruiting cohort plan")

    checklists = [OnboardingChecklistRecord.from_mapping(item) for item in fixture["onboarding_checklists"]]
    for record in checklists:
        _assert(not validate_onboarding_checklist_record(record), f"onboarding checklist validates: {record.checklist_id}")
    for record in fixture["invalid_onboarding_checklists"]:
        errors = validate_onboarding_checklist_record(record)
        _assert(errors, f"invalid onboarding checklist fails: {record['checklist_id']}")
        _assert_expected_error(errors, record)

    return {
        "map_cells": map_summary["cell_count"],
        "template_briefs": len(template_briefs),
        "candidates": len(candidates),
        "cohort_target_min": cohort.target_min_experts,
        "cohort_target_max": cohort.target_max_experts,
        "onboarding_checklists": len(checklists),
    }


def _sensitive_payload_errors(record: Any) -> list[str]:
    payload = record.to_dict() if hasattr(record, "to_dict") else record
    found = sorted(_find_sensitive_recruiting_material(payload))
    if not found:
        return []
    return ["recruiting/admin payload contains unnecessary sensitive data keys/markers: " + ", ".join(found)]


def _find_sensitive_recruiting_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in SENSITIVE_RECRUITING_KEYS:
                found.add(key_path)
            found.update(_find_sensitive_recruiting_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_sensitive_recruiting_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in SENSITIVE_RECRUITING_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if expected:
        _assert(any(str(expected) in error for error in errors), f"expected error {expected!r}")


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
