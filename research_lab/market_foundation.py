"""Phase 2 foundation contracts for the Leadpoet Research Lab.

Phase 2 adds market depth and sealing, but this module is still local-only.
It defines records and guards that later Phase 2 slices can consume without
starting full component-market workflows, releasing KMS keys, launching
research enclaves, publishing on-chain anchors, moving balances, or executing
counterfactual-gate consequences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .loop_foundation import (
    TRUTHY_ENV_VALUES,
    verify_loop_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "market_foundation_fixtures.json"


class MarketDependencyGate(str, Enum):
    PRODUCTION_SQL_APPLIED = "production_sql_applied"
    DUAL_ARM_PAIRED_DAY_CLOCK_STARTED = "dual_arm_paired_day_clock_started"
    DUAL_ARM_PAIRED_DAYS_ACCUMULATED = "dual_arm_paired_days_accumulated"
    SAME_DAY_CROWNING_CERTIFIED = "same_day_crowning_certified"
    MEASURED_EXTERNAL_COST_CALIBRATED = "measured_external_cost_calibrated"
    LOOP_HOSTED_RUN_CORPUS_READY = "loop_hosted_run_corpus_ready"
    HOSTED_RUN_CROWNING_SURVIVED_PROBATION = "hosted_run_crowning_survived_probation"
    EXTERNAL_VALIDATORS_VERIFIED = "external_validators_verified"
    JOB_UNIT_ECONOMICS_PUBLISHED = "job_unit_economics_published"
    BASELINE_ARM_COMPARISON_PUBLISHED = "baseline_arm_comparison_published"
    OUTCOME_LABEL_ACCUMULATION_ON_CURVE = "outcome_label_accumulation_on_curve"
    GOLD_ICP_INVENTORY_READY = "gold_icp_inventory_ready"


class MarketSealingPostureState(str, Enum):
    LOCAL_STUB_ONLY = "local_stub_only"
    FAILED = "failed"
    PRODUCTION_READY = "production_ready"


MARKET_DEPENDENCY_GATES: tuple[str, ...] = tuple(gate.value for gate in MarketDependencyGate)


@dataclass(frozen=True)
class MarketWorkflowGuards:
    full_component_market: bool = False
    live_meta_allocation: bool = False
    research_map_live_api: bool = False
    kms_key_release: bool = False
    research_enclaves: bool = False
    on_chain_anchor_publication: bool = False
    ledger_audit_publication: bool = False
    counterfactual_consequences: bool = False
    balance_mutations: bool = False
    bounty_payments: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketWorkflowGuards":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class MarketCarryForwardStatus:
    production_sql_applied: bool = False
    dual_arm_paired_day_clock_started: bool = False
    dual_arm_paired_days_accumulated: bool = False
    same_day_crowning_certified: bool = False
    measured_external_cost_calibrated: bool = False
    loop_hosted_run_corpus_ready: bool = False
    hosted_run_crowning_survived_probation: bool = False
    external_validators_verified: bool = False
    job_unit_economics_published: bool = False
    baseline_arm_comparison_published: bool = False
    outcome_label_accumulation_on_curve: bool = False
    gold_icp_inventory_ready: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketCarryForwardStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class MarketReadinessClaimRecord:
    readiness_id: str
    carry_forward_status: MarketCarryForwardStatus
    market_operation_claimed_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketReadinessClaimRecord":
        carry_forward = data.get("carry_forward_status", {})
        return cls(
            readiness_id=str(data["readiness_id"]),
            carry_forward_status=MarketCarryForwardStatus.from_mapping(carry_forward),
            market_operation_claimed_ready=bool(data.get("market_operation_claimed_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness_id": self.readiness_id,
            "carry_forward_status": self.carry_forward_status.to_dict(),
            "market_operation_claimed_ready": self.market_operation_claimed_ready,
            "uses_local_fixtures": self.uses_local_fixtures,
            "local_only": self.local_only,
            "evidence_refs": list(self.evidence_refs),
            "owner_approval_ref": self.owner_approval_ref,
        }


@dataclass(frozen=True)
class MarketSealingPostureRecord:
    posture_id: str
    pcr0_allowlist_ref: str
    enclave_measurement_ref: str
    kms_policy_ref: str
    on_chain_anchor_ref: str
    pcr0_allowlist_checked: bool = False
    pcr0_allowed: bool = False
    signature_chain_checked: bool = False
    signature_chain_valid: bool = False
    egress_policy_checked: bool = False
    egress_policy_passed: bool = False
    equivocation_check_performed: bool = False
    equivocation_conflict: bool = False
    kms_policy_checked: bool = False
    kms_policy_passed: bool = False
    live_verification_performed: bool = False
    kms_key_release_requested: bool = False
    kms_key_released: bool = False
    research_enclave_started: bool = False
    on_chain_anchor_submitted: bool = False
    production_valid: bool = False
    local_stub_only: bool = True
    state: str = MarketSealingPostureState.LOCAL_STUB_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketSealingPostureRecord":
        return cls(
            posture_id=str(data["posture_id"]),
            pcr0_allowlist_ref=str(data.get("pcr0_allowlist_ref", "")),
            enclave_measurement_ref=str(data.get("enclave_measurement_ref", "")),
            kms_policy_ref=str(data.get("kms_policy_ref", "")),
            on_chain_anchor_ref=str(data.get("on_chain_anchor_ref", "")),
            pcr0_allowlist_checked=bool(data.get("pcr0_allowlist_checked", False)),
            pcr0_allowed=bool(data.get("pcr0_allowed", False)),
            signature_chain_checked=bool(data.get("signature_chain_checked", False)),
            signature_chain_valid=bool(data.get("signature_chain_valid", False)),
            egress_policy_checked=bool(data.get("egress_policy_checked", False)),
            egress_policy_passed=bool(data.get("egress_policy_passed", False)),
            equivocation_check_performed=bool(data.get("equivocation_check_performed", False)),
            equivocation_conflict=bool(data.get("equivocation_conflict", False)),
            kms_policy_checked=bool(data.get("kms_policy_checked", False)),
            kms_policy_passed=bool(data.get("kms_policy_passed", False)),
            live_verification_performed=bool(data.get("live_verification_performed", False)),
            kms_key_release_requested=bool(data.get("kms_key_release_requested", False)),
            kms_key_released=bool(data.get("kms_key_released", False)),
            research_enclave_started=bool(data.get("research_enclave_started", False)),
            on_chain_anchor_submitted=bool(data.get("on_chain_anchor_submitted", False)),
            production_valid=bool(data.get("production_valid", False)),
            local_stub_only=bool(data.get("local_stub_only", True)),
            state=str(data.get("state", MarketSealingPostureState.LOCAL_STUB_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_market_workflow_guards() -> MarketWorkflowGuards:
    return MarketWorkflowGuards()


def market_workflow_guards_from_env(env: Optional[Mapping[str, str]] = None) -> MarketWorkflowGuards:
    env = env or {}
    return MarketWorkflowGuards(
        full_component_market=_truthy(env.get("RESEARCH_LAB_MARKET_FULL_COMPONENT_MARKET")),
        live_meta_allocation=_truthy(env.get("RESEARCH_LAB_MARKET_LIVE_META_ALLOCATION")),
        research_map_live_api=_truthy(env.get("RESEARCH_LAB_MARKET_RESEARCH_MAP_LIVE_API")),
        kms_key_release=_truthy(env.get("RESEARCH_LAB_MARKET_KMS_KEY_RELEASE")),
        research_enclaves=_truthy(env.get("RESEARCH_LAB_MARKET_RESEARCH_ENCLAVES")),
        on_chain_anchor_publication=_truthy(env.get("RESEARCH_LAB_MARKET_ON_CHAIN_ANCHORS")),
        ledger_audit_publication=_truthy(env.get("RESEARCH_LAB_MARKET_LEDGER_AUDIT_PUBLICATION")),
        counterfactual_consequences=_truthy(env.get("RESEARCH_LAB_MARKET_COUNTERFACTUAL_CONSEQUENCES")),
        balance_mutations=_truthy(env.get("RESEARCH_LAB_MARKET_BALANCE_MUTATIONS")),
        bounty_payments=_truthy(env.get("RESEARCH_LAB_MARKET_BOUNTY_PAYMENTS")),
        production_writes=_truthy(env.get("RESEARCH_LAB_MARKET_PRODUCTION_WRITES")),
        supabase_writes=_truthy(env.get("RESEARCH_LAB_MARKET_SUPABASE_WRITES")),
        public_workflows=_truthy(env.get("RESEARCH_LAB_MARKET_PUBLIC_WORKFLOWS")),
    )


def default_market_carry_forward_status() -> MarketCarryForwardStatus:
    return MarketCarryForwardStatus()


def validate_market_workflow_guards(guards: MarketWorkflowGuards | Mapping[str, Any]) -> list[str]:
    if not isinstance(guards, MarketWorkflowGuards):
        guards = MarketWorkflowGuards.from_mapping(guards)
    enabled = [name for name, value in guards.to_dict().items() if value]
    if not enabled:
        return []
    return ["Phase 2 local build must keep these workflows disabled: " + ", ".join(enabled)]


def assert_market_workflows_disabled(guards: MarketWorkflowGuards | Mapping[str, Any]) -> None:
    errors = validate_market_workflow_guards(guards)
    if errors:
        raise ValueError("; ".join(errors))


def market_gate_ready(status: MarketCarryForwardStatus | Mapping[str, Any], gate: MarketDependencyGate | str) -> bool:
    if not isinstance(status, MarketCarryForwardStatus):
        status = MarketCarryForwardStatus.from_mapping(status)
    gate_value = gate.value if isinstance(gate, MarketDependencyGate) else str(gate)
    if gate_value not in MarketDependencyGate._value2member_map_:
        raise ValueError(f"unknown Phase 2 dependency gate: {gate_value}")
    return bool(getattr(status, gate_value))


def require_market_gate(status: MarketCarryForwardStatus | Mapping[str, Any], gate: MarketDependencyGate | str) -> None:
    if not market_gate_ready(status, gate):
        gate_value = gate.value if isinstance(gate, MarketDependencyGate) else str(gate)
        raise ValueError(f"Phase 2 dependency gate is not ready: {gate_value}")


def validate_market_readiness_claim(record: MarketReadinessClaimRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, MarketReadinessClaimRecord):
        record = MarketReadinessClaimRecord.from_mapping(record)
    errors: list[str] = []
    if not record.readiness_id:
        errors.append("Phase 2 readiness claim requires readiness_id")
    missing = [
        gate
        for gate in MARKET_DEPENDENCY_GATES
        if not market_gate_ready(record.carry_forward_status, gate)
    ]
    if record.market_operation_claimed_ready:
        if record.uses_local_fixtures:
            errors.append("Phase 2 readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 2 readiness cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 2 readiness claim is missing dependency gates: " + ", ".join(missing))
        if not record.evidence_refs:
            errors.append("Phase 2 readiness claim requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Phase 2 readiness claim requires owner_approval_ref")
    else:
        if not record.local_only:
            errors.append("not-ready Phase 2 readiness records must remain local_only")
    return errors


def validate_market_sealing_posture(record: MarketSealingPostureRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, MarketSealingPostureRecord):
        record = MarketSealingPostureRecord.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in MarketSealingPostureState}:
        errors.append(f"unknown Phase 2 sealing posture state: {record.state}")
        return errors
    for field in ("posture_id", "pcr0_allowlist_ref", "enclave_measurement_ref", "kms_policy_ref", "on_chain_anchor_ref"):
        if not getattr(record, field):
            errors.append(f"Phase 2 sealing posture requires {field}")
    if record.local_stub_only:
        if record.state != MarketSealingPostureState.LOCAL_STUB_ONLY.value:
            errors.append("local_stub_only sealing posture must use state local_stub_only")
        if record.kms_key_release_requested:
            errors.append("local sealing stub must not request KMS key release")
        if record.kms_key_released:
            errors.append("local sealing stub must not release KMS keys")
        if record.research_enclave_started:
            errors.append("local sealing stub must not start research enclaves")
        if record.on_chain_anchor_submitted:
            errors.append("local sealing stub must not submit on-chain anchors")
        if record.production_valid:
            errors.append("local sealing stub must not claim production_valid")

    wants_production = (
        record.state == MarketSealingPostureState.PRODUCTION_READY.value
        or record.production_valid
        or record.kms_key_released
    )
    if wants_production:
        if record.local_stub_only:
            errors.append("production sealing cannot be claimed by a local_stub_only record")
        if not record.live_verification_performed:
            errors.append("production sealing requires live_verification_performed")
        if not record.pcr0_allowlist_checked:
            errors.append("PCR0 allowlist check must not be skipped for production sealing")
        if not record.pcr0_allowed:
            errors.append("PCR0 must be allowed for production sealing")
        if not record.signature_chain_checked:
            errors.append("signature-chain check must not be skipped for production sealing")
        if not record.signature_chain_valid:
            errors.append("signature-chain must be valid for production sealing")
        if not record.egress_policy_checked:
            errors.append("egress-policy check must not be skipped for production sealing")
        if not record.egress_policy_passed:
            errors.append("egress policy must pass for production sealing")
        if not record.equivocation_check_performed:
            errors.append("equivocation check must not be skipped for production sealing")
        if record.equivocation_conflict:
            errors.append("equivocation conflict blocks production sealing")
        if not record.kms_policy_checked:
            errors.append("KMS policy check must not be skipped for production sealing")
        if not record.kms_policy_passed:
            errors.append("KMS policy must pass for production sealing")
        if not record.kms_key_release_requested:
            errors.append("production sealing requires explicit KMS key-release request")
        if not record.kms_key_released:
            errors.append("production sealing requires KMS key release")
        if not record.research_enclave_started:
            errors.append("production sealing requires research enclave startup")
        if not record.on_chain_anchor_submitted:
            errors.append("production sealing requires on-chain anchor submission")
    elif record.state == MarketSealingPostureState.FAILED.value:
        if record.production_valid:
            errors.append("failed sealing posture cannot claim production_valid")
    return errors


def verify_market_foundation(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    loop_summary = verify_loop_foundation()
    fixture = _load_fixture(Path(fixture_path))

    _assert(
        fixture["market_dependency_gate_values"] == list(MARKET_DEPENDENCY_GATES),
        "fixture Phase 2 dependency gates match code",
    )

    disabled_guards = MarketWorkflowGuards.from_mapping(fixture["workflow_guards"]["disabled"])
    _assert(not validate_market_workflow_guards(disabled_guards), "disabled Phase 2 workflow guards validate")

    invalid_guard_errors = validate_market_workflow_guards(fixture["workflow_guards"]["invalid"])
    _assert(invalid_guard_errors, "enabled Phase 2 workflow guards fail closed")
    _assert_expected_error(invalid_guard_errors, fixture["workflow_guards"]["invalid"])

    env_guards = market_workflow_guards_from_env(fixture["workflow_guard_env"])
    env_errors = validate_market_workflow_guards(env_guards)
    _assert(env_errors, "truthy Phase 2 env flags fail closed")
    _assert_expected_error(env_errors, fixture["workflow_guard_env"])

    carry_forward = MarketCarryForwardStatus.from_mapping(fixture["carry_forward_status"])
    for gate in MARKET_DEPENDENCY_GATES:
        _assert(not market_gate_ready(carry_forward, gate), f"local carry-forward gate defaults false: {gate}")
        try:
            require_market_gate(carry_forward, gate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing Phase 2 gate raises: {gate}")

    local_claim = MarketReadinessClaimRecord.from_mapping(fixture["readiness_claims"]["local_not_ready"])
    _assert(not validate_market_readiness_claim(local_claim), "local not-ready Phase 2 claim validates")
    for record in fixture["readiness_claims"]["invalid"]:
        errors = validate_market_readiness_claim(record)
        _assert(errors, f"invalid Phase 2 readiness claim fails: {record['readiness_id']}")
        _assert_expected_error(errors, record)

    local_sealing = MarketSealingPostureRecord.from_mapping(fixture["sealing_postures"]["local_stub"])
    _assert(not validate_market_sealing_posture(local_sealing), "local sealing stub validates")
    for record in fixture["sealing_postures"]["invalid"]:
        errors = validate_market_sealing_posture(record)
        _assert(errors, f"invalid sealing posture fails: {record['posture_id']}")
        _assert_expected_error(errors, record)

    return {
        "loop_release_records": loop_summary["valid_release_records"],
        "dependency_gates": len(MARKET_DEPENDENCY_GATES),
        "workflow_guard_fields": len(MarketWorkflowGuards.__dataclass_fields__),
        "local_readiness_id": local_claim.readiness_id,
        "local_sealing_state": local_sealing.state,
    }


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in TRUTHY_ENV_VALUES


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
