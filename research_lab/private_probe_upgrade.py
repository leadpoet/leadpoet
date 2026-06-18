"""Phase 2 private-probe upgrade-path contracts.

P2.3 defines how Phase 1 private probes can graduate into Phase 2 market-depth
mechanics. The records are local-only policy and handoff stubs: they do not
schedule live probes, open public miner workflows, publish private probe
outputs, bypass component review, or write to production systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .component_market import (
    COMPONENT_MARKET_TYPES,
    PROTECTED_COMPONENT_KEYS,
    PROTECTED_COMPONENT_MARKERS,
    ComponentMarketType,
    validate_component_review_record,
    verify_research_lab_component_market,
)
from .loop_game import (
    PrivateProbeReceipt,
    validate_private_probe_receipt,
    verify_research_lab_loop_game,
)
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


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "private_probe_upgrade_fixtures.json"

PROTECTED_PROBE_KEYS = set(PROTECTED_COMPONENT_KEYS) | {
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
    "raw_probe_result",
    "raw_snapshot",
    "raw_text",
    "sealed_eval",
    "sealed_eval_details",
    "secret",
    "token",
    "wallet_address",
}

PROTECTED_PROBE_MARKERS: tuple[str, ...] = tuple(sorted(set(PROTECTED_COMPONENT_MARKERS) | {
    "api key",
    "judge prompt",
    "live champion code",
    "live champion prompt",
    "model weights",
    "password=",
    "private customer",
    "raw customer",
    "raw evidence",
    "raw probe result",
    "raw snapshot",
    "sealed eval",
    "sealed judge prompt",
    "sk-live",
}))


class ProbeGraduationState(str, Enum):
    LOCAL_CANDIDATE = "local_candidate"
    READY_FOR_COMPONENT_REVIEW = "ready_for_component_review"
    BLOCKED = "blocked"


class ProbeOutcomeReleaseState(str, Enum):
    PRIVATE_ONLY = "private_only"
    SANITIZED_TRACE_STUB = "sanitized_trace_stub"
    PUBLIC_RECEIPT_STUB = "public_receipt_stub"
    BLOCKED = "blocked"


class ProbeHandoffState(str, Enum):
    LOCAL_REVIEW_PENDING = "local_review_pending"
    READY_FOR_COMPONENT_REVIEW = "ready_for_component_review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProbeGraduationCriteriaRecord:
    criteria_id: str
    target_component_type: str
    min_dev_delta_lcb: float
    min_result_refs: int = 1
    requires_owner_opt_in: bool = True
    requires_component_review: bool = True
    requires_sanitized_trace_policy: bool = True
    live_probe_scheduling_enabled: bool = False
    public_miner_workflow_enabled: bool = False
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProbeGraduationCriteriaRecord":
        return cls(
            criteria_id=str(data["criteria_id"]),
            target_component_type=str(data["target_component_type"]),
            min_dev_delta_lcb=float(data["min_dev_delta_lcb"]),
            min_result_refs=int(data.get("min_result_refs", 1)),
            requires_owner_opt_in=bool(data.get("requires_owner_opt_in", True)),
            requires_component_review=bool(data.get("requires_component_review", True)),
            requires_sanitized_trace_policy=bool(data.get("requires_sanitized_trace_policy", True)),
            live_probe_scheduling_enabled=bool(data.get("live_probe_scheduling_enabled", False)),
            public_miner_workflow_enabled=bool(data.get("public_miner_workflow_enabled", False)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeGraduationDecisionRecord:
    decision_id: str
    probe_receipt_ref: str
    probe_result_ref: str
    target_component_type: str
    dev_delta_lcb: float
    result_refs: tuple[str, ...]
    criteria_ref: str
    owner_opt_in_ref: str = ""
    graduation_ready: bool = False
    component_submission_ref: str = ""
    component_review_ref: str = ""
    component_review_required: bool = True
    component_review_bypass_requested: bool = False
    live_probe_scheduling_enabled: bool = False
    public_miner_workflow_enabled: bool = False
    local_only: bool = True
    state: str = ProbeGraduationState.LOCAL_CANDIDATE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProbeGraduationDecisionRecord":
        return cls(
            decision_id=str(data["decision_id"]),
            probe_receipt_ref=str(data["probe_receipt_ref"]),
            probe_result_ref=str(data["probe_result_ref"]),
            target_component_type=str(data["target_component_type"]),
            dev_delta_lcb=float(data["dev_delta_lcb"]),
            result_refs=tuple(str(item) for item in data.get("result_refs", [])),
            criteria_ref=str(data["criteria_ref"]),
            owner_opt_in_ref=str(data.get("owner_opt_in_ref", "")),
            graduation_ready=bool(data.get("graduation_ready", False)),
            component_submission_ref=str(data.get("component_submission_ref", "")),
            component_review_ref=str(data.get("component_review_ref", "")),
            component_review_required=bool(data.get("component_review_required", True)),
            component_review_bypass_requested=bool(data.get("component_review_bypass_requested", False)),
            live_probe_scheduling_enabled=bool(data.get("live_probe_scheduling_enabled", False)),
            public_miner_workflow_enabled=bool(data.get("public_miner_workflow_enabled", False)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", ProbeGraduationState.LOCAL_CANDIDATE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["result_refs"] = list(self.result_refs)
        return data


@dataclass(frozen=True)
class ProbeOutcomeReleasePolicyRecord:
    policy_id: str
    probe_receipt_ref: str
    private_result_ref: str
    sanitized_summary_ref: str
    summary_text: str
    proof_refs: tuple[str, ...]
    public_receipt_ref: str = ""
    sanitized_trace_ref: str = ""
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    raw_probe_result_included: bool = False
    public_publication_enabled: bool = False
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value
    state: str = ProbeOutcomeReleaseState.SANITIZED_TRACE_STUB.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProbeOutcomeReleasePolicyRecord":
        return cls(
            policy_id=str(data["policy_id"]),
            probe_receipt_ref=str(data["probe_receipt_ref"]),
            private_result_ref=str(data["private_result_ref"]),
            sanitized_summary_ref=str(data["sanitized_summary_ref"]),
            summary_text=str(data["summary_text"]),
            proof_refs=tuple(str(item) for item in data.get("proof_refs", [])),
            public_receipt_ref=str(data.get("public_receipt_ref", "")),
            sanitized_trace_ref=str(data.get("sanitized_trace_ref", "")),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            raw_probe_result_included=bool(data.get("raw_probe_result_included", False)),
            public_publication_enabled=bool(data.get("public_publication_enabled", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
            state=str(data.get("state", ProbeOutcomeReleaseState.SANITIZED_TRACE_STUB.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proof_refs"] = list(self.proof_refs)
        return data


@dataclass(frozen=True)
class ProbeToComponentHandoffRecord:
    handoff_id: str
    probe_receipt_ref: str
    graduation_decision_ref: str
    release_policy_ref: str
    target_component_type: str
    component_submission_ref: str
    component_review_ref: str
    private_result_ref: str
    sanitized_trace_ref: str
    component_review_required: bool = True
    component_review_bypass_requested: bool = False
    public_miner_workflow_enabled: bool = False
    live_probe_scheduling_enabled: bool = False
    component_market_workflow_enabled: bool = False
    production_writes: bool = False
    local_only: bool = True
    state: str = ProbeHandoffState.LOCAL_REVIEW_PENDING.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProbeToComponentHandoffRecord":
        return cls(
            handoff_id=str(data["handoff_id"]),
            probe_receipt_ref=str(data["probe_receipt_ref"]),
            graduation_decision_ref=str(data["graduation_decision_ref"]),
            release_policy_ref=str(data["release_policy_ref"]),
            target_component_type=str(data["target_component_type"]),
            component_submission_ref=str(data["component_submission_ref"]),
            component_review_ref=str(data["component_review_ref"]),
            private_result_ref=str(data["private_result_ref"]),
            sanitized_trace_ref=str(data["sanitized_trace_ref"]),
            component_review_required=bool(data.get("component_review_required", True)),
            component_review_bypass_requested=bool(data.get("component_review_bypass_requested", False)),
            public_miner_workflow_enabled=bool(data.get("public_miner_workflow_enabled", False)),
            live_probe_scheduling_enabled=bool(data.get("live_probe_scheduling_enabled", False)),
            component_market_workflow_enabled=bool(data.get("component_market_workflow_enabled", False)),
            production_writes=bool(data.get("production_writes", False)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", ProbeHandoffState.LOCAL_REVIEW_PENDING.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_probe_graduation_criteria(
    record: ProbeGraduationCriteriaRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_probe_payload_errors(raw)
    if not isinstance(record, ProbeGraduationCriteriaRecord):
        record = ProbeGraduationCriteriaRecord.from_mapping(record)
    errors: list[str] = list(raw_errors)
    if not record.criteria_id:
        errors.append("probe graduation criteria requires criteria_id")
    if record.target_component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown target_component_type: {record.target_component_type}")
    if record.target_component_type == ComponentMarketType.SOURCE_ADD.value:
        errors.append("private-probe upgrade should target Phase 2 component types beyond source_add")
    if record.min_dev_delta_lcb < 0:
        errors.append("min_dev_delta_lcb must be non-negative")
    if record.min_result_refs <= 0:
        errors.append("min_result_refs must be positive")
    if not record.requires_owner_opt_in:
        errors.append("probe graduation requires owner opt-in")
    if not record.requires_component_review:
        errors.append("probe graduation requires component-market review")
    if not record.requires_sanitized_trace_policy:
        errors.append("probe graduation requires sanitized trace policy")
    if record.live_probe_scheduling_enabled:
        errors.append("probe graduation criteria must not enable live probe scheduling")
    if record.public_miner_workflow_enabled:
        errors.append("probe graduation criteria must not enable public miner workflows")
    if not record.local_only:
        errors.append("probe graduation criteria must remain local_only")
    return errors


def validate_probe_graduation_decision(
    record: ProbeGraduationDecisionRecord | Mapping[str, Any],
    criteria: ProbeGraduationCriteriaRecord | Mapping[str, Any],
    probe_receipt: PrivateProbeReceipt | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_probe_payload_errors(raw)
    if not isinstance(record, ProbeGraduationDecisionRecord):
        record = ProbeGraduationDecisionRecord.from_mapping(record)
    criteria_errors = validate_probe_graduation_criteria(criteria)
    if not isinstance(criteria, ProbeGraduationCriteriaRecord):
        criteria = ProbeGraduationCriteriaRecord.from_mapping(criteria)
    if not isinstance(probe_receipt, PrivateProbeReceipt):
        probe_receipt = PrivateProbeReceipt.from_mapping(probe_receipt)
    errors: list[str] = list(raw_errors)
    errors.extend(criteria_errors)
    errors.extend(validate_private_probe_receipt(probe_receipt))
    if record.state not in {state.value for state in ProbeGraduationState}:
        errors.append(f"unknown probe graduation state: {record.state}")
    if record.probe_receipt_ref != probe_receipt.receipt_ref:
        errors.append("probe graduation decision must reference the private probe receipt")
    if record.probe_result_ref != probe_receipt.result_ref:
        errors.append("probe graduation decision must reference the private probe result")
    if record.target_component_type != criteria.target_component_type:
        errors.append("probe graduation decision target_component_type must match criteria")
    if record.criteria_ref != criteria.criteria_id:
        errors.append("probe graduation decision criteria_ref must match criteria")
    if record.dev_delta_lcb != probe_receipt.dev_delta_lcb:
        errors.append("probe graduation decision dev_delta_lcb must match private probe receipt")
    if len(record.result_refs) < criteria.min_result_refs:
        errors.append("probe graduation decision requires enough result_refs")
    if record.graduation_ready and record.dev_delta_lcb < criteria.min_dev_delta_lcb:
        errors.append("graduation_ready requires dev_delta_lcb to meet criteria")
    if criteria.requires_owner_opt_in and record.graduation_ready and not record.owner_opt_in_ref:
        errors.append("graduation_ready requires owner_opt_in_ref")
    if criteria.requires_component_review and not record.component_review_required:
        errors.append("probe graduation must require component review")
    if record.component_review_bypass_requested:
        errors.append("probe graduation must not request component review bypass")
    if record.graduation_ready and not record.component_submission_ref:
        errors.append("graduation_ready requires component_submission_ref")
    if record.graduation_ready and not record.component_review_ref:
        errors.append("graduation_ready requires component_review_ref")
    if record.live_probe_scheduling_enabled:
        errors.append("probe graduation must not enable live probe scheduling")
    if record.public_miner_workflow_enabled:
        errors.append("probe graduation must not enable public miner workflows")
    if not record.local_only:
        errors.append("probe graduation decision must remain local_only")
    return errors


def validate_probe_outcome_release_policy(
    record: ProbeOutcomeReleasePolicyRecord | Mapping[str, Any],
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_probe_payload_errors(raw)
    if not isinstance(record, ProbeOutcomeReleasePolicyRecord):
        record = ProbeOutcomeReleasePolicyRecord.from_mapping(record)
    errors: list[str] = list(raw_errors)
    if record.state not in {state.value for state in ProbeOutcomeReleaseState}:
        errors.append(f"unknown probe outcome release state: {record.state}")
    if not record.private_result_ref:
        errors.append("probe outcome policy requires private_result_ref")
    if not record.sanitized_summary_ref:
        errors.append("probe outcome policy requires sanitized_summary_ref")
    if not record.summary_text:
        errors.append("probe outcome policy requires summary_text")
    if not record.proof_refs:
        errors.append("probe outcome policy requires proof_refs")
    if record.raw_probe_result_included:
        errors.append("probe outcome policy must not include raw probe result")
    if record.public_publication_enabled:
        errors.append("probe outcome policy must not enable public publication")
    protected_flags = _release_protected_flags(record)
    if protected_flags:
        errors.append("probe outcome policy must not contain protected material flags: " + ", ".join(protected_flags))
    if record.state == ProbeOutcomeReleaseState.PRIVATE_ONLY.value:
        if record.artifact_release_state != ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value:
            errors.append("private-only probe outcome must remain private_live_champion")
        if record.public_receipt_ref or record.sanitized_trace_ref:
            errors.append("private-only probe outcome must not set public refs")
    if record.state == ProbeOutcomeReleaseState.SANITIZED_TRACE_STUB.value:
        if record.artifact_release_state != ArtifactReleaseState.SANITIZED_TRACE.value:
            errors.append("sanitized trace probe outcome requires sanitized_trace release state")
        if not record.sanitized_trace_ref:
            errors.append("sanitized trace probe outcome requires sanitized_trace_ref")
    if record.state == ProbeOutcomeReleaseState.PUBLIC_RECEIPT_STUB.value:
        if record.artifact_release_state != ArtifactReleaseState.PUBLIC_RECEIPT.value:
            errors.append("public receipt probe outcome requires public_receipt release state")
        if not record.public_receipt_ref:
            errors.append("public receipt probe outcome requires public_receipt_ref")
    artifact_type = "receipt" if record.artifact_release_state == ArtifactReleaseState.PUBLIC_RECEIPT.value else "sanitized_trace"
    if record.artifact_release_state == ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value:
        artifact_type = "private_probe_receipt"
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.policy_id,
                artifact_type=artifact_type,
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                contains_live_champion_ip=record.contains_live_champion_ip,
                contains_sealed_eval_details=record.contains_sealed_eval_details,
                contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
                contains_private_customer_data=record.contains_private_customer_data,
                contains_judge_prompts=record.contains_judge_prompts,
                reason="private probe outcomes can only become private, receipt, or sanitized trace stubs",
            )
        )
    )
    return errors


def validate_probe_to_component_handoff(
    record: ProbeToComponentHandoffRecord | Mapping[str, Any],
    decision: ProbeGraduationDecisionRecord | Mapping[str, Any],
    release_policy: ProbeOutcomeReleasePolicyRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    raw_errors = _protected_probe_payload_errors(raw)
    if not isinstance(record, ProbeToComponentHandoffRecord):
        record = ProbeToComponentHandoffRecord.from_mapping(record)
    if not isinstance(decision, ProbeGraduationDecisionRecord):
        decision = ProbeGraduationDecisionRecord.from_mapping(decision)
    if not isinstance(release_policy, ProbeOutcomeReleasePolicyRecord):
        release_policy = ProbeOutcomeReleasePolicyRecord.from_mapping(release_policy)
    errors: list[str] = list(raw_errors)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ProbeHandoffState}:
        errors.append(f"unknown probe handoff state: {record.state}")
    if record.probe_receipt_ref != decision.probe_receipt_ref:
        errors.append("probe handoff must reference the graduation decision receipt")
    if record.graduation_decision_ref != decision.decision_id:
        errors.append("probe handoff graduation_decision_ref must match decision")
    if record.release_policy_ref != release_policy.policy_id:
        errors.append("probe handoff release_policy_ref must match release policy")
    if record.target_component_type not in COMPONENT_MARKET_TYPES:
        errors.append(f"unknown target_component_type: {record.target_component_type}")
    if record.target_component_type != decision.target_component_type:
        errors.append("probe handoff target_component_type must match decision")
    if record.component_submission_ref != decision.component_submission_ref:
        errors.append("probe handoff component_submission_ref must match decision")
    if record.component_review_ref != decision.component_review_ref:
        errors.append("probe handoff component_review_ref must match decision")
    if record.private_result_ref != release_policy.private_result_ref:
        errors.append("probe handoff private_result_ref must match release policy")
    if record.sanitized_trace_ref != release_policy.sanitized_trace_ref:
        errors.append("probe handoff sanitized_trace_ref must match release policy")
    if not record.component_review_required:
        errors.append("probe handoff must require component review")
    if record.component_review_bypass_requested:
        errors.append("probe handoff must not bypass component review")
    if record.public_miner_workflow_enabled:
        errors.append("probe handoff must not enable public miner workflows")
    if record.live_probe_scheduling_enabled:
        errors.append("probe handoff must not enable live probe scheduling")
    if record.component_market_workflow_enabled:
        errors.append("probe handoff must not enable component-market workflow")
    if record.production_writes:
        errors.append("probe handoff must not perform production_writes")
    if not record.local_only:
        errors.append("probe handoff must remain local_only")
    if record.state == ProbeHandoffState.READY_FOR_COMPONENT_REVIEW.value and not decision.graduation_ready:
        errors.append("ready handoff requires graduation_ready decision")
    return errors


def verify_research_lab_private_probe_upgrade(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    loop_summary = verify_research_lab_loop_game()
    component_summary = verify_research_lab_component_market()
    fixture = _load_fixture(Path(fixture_path))

    probe_receipt = PrivateProbeReceipt.from_mapping(fixture["private_probe_receipt"])
    _assert(not validate_private_probe_receipt(probe_receipt), "private probe receipt remains valid and private")

    criteria = ProbeGraduationCriteriaRecord.from_mapping(fixture["graduation_criteria"])
    _assert(not validate_probe_graduation_criteria(criteria), "probe graduation criteria validates")
    for record in fixture["invalid_graduation_criteria"]:
        errors = validate_probe_graduation_criteria(record)
        _assert(errors, f"invalid graduation criteria fails: {record['criteria_id']}")
        _assert_expected_error(errors, record)

    decision = ProbeGraduationDecisionRecord.from_mapping(fixture["graduation_decision"])
    _assert(
        not validate_probe_graduation_decision(decision, criteria, probe_receipt),
        "probe graduation decision validates",
    )
    for record in fixture["invalid_graduation_decisions"]:
        errors = validate_probe_graduation_decision(record, criteria, probe_receipt)
        _assert(errors, f"invalid graduation decision fails: {record['decision_id']}")
        _assert_expected_error(errors, record)

    release_policy = ProbeOutcomeReleasePolicyRecord.from_mapping(fixture["outcome_release_policy"])
    _assert(not validate_probe_outcome_release_policy(release_policy), "probe outcome release policy validates")
    for fixture_key in ("private_only_outcome_release_policy", "public_receipt_outcome_release_policy"):
        branch_policy = ProbeOutcomeReleasePolicyRecord.from_mapping(fixture[fixture_key])
        _assert(not validate_probe_outcome_release_policy(branch_policy), f"{fixture_key} validates")
    for record in fixture["invalid_outcome_release_policies"]:
        errors = validate_probe_outcome_release_policy(record)
        _assert(errors, f"invalid outcome release policy fails: {record['policy_id']}")
        _assert_expected_error(errors, record)

    handoff = ProbeToComponentHandoffRecord.from_mapping(fixture["handoff"])
    _assert(
        not validate_probe_to_component_handoff(handoff, decision, release_policy),
        "probe-to-component handoff validates",
    )
    for record in fixture["invalid_handoffs"]:
        errors = validate_probe_to_component_handoff(record, decision, release_policy)
        _assert(errors, f"invalid probe handoff fails: {record['handoff_id']}")
        _assert_expected_error(errors, record)
    unsafe_guard_errors = validate_probe_to_component_handoff(
        handoff,
        decision,
        release_policy,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_guard_errors, "unsafe Phase 2 workflow guards block private-probe handoff")

    component_review_errors = validate_component_review_record(fixture["component_review_stub"])
    _assert(component_review_errors, "component review stub remains not accepted/executed")
    _assert(
        any("sandboxed execution" in error or "component_accepted" in error for error in component_review_errors),
        "component review stub cannot claim execution or acceptance",
    )

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "loop_private_probe_receipts": loop_summary["private_probe_receipts"],
        "component_types": component_summary["component_types"],
        "probe_receipt_ref": probe_receipt.receipt_ref,
        "target_component_type": decision.target_component_type,
        "graduation_ready": decision.graduation_ready,
        "handoff_id": handoff.handoff_id,
    }


def _release_protected_flags(record: ProbeOutcomeReleasePolicyRecord) -> list[str]:
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


def _protected_probe_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_probe_material(record))
    if not found:
        return []
    return ["private-probe upgrade payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_probe_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_PROBE_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_probe_material(nested, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_protected_probe_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_PROBE_MARKERS:
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
