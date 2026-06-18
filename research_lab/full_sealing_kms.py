"""Phase 2 full-sealing and KMS contract records.

P2.6 defines the local record shapes for research-enclave measurements and
KMS-sealed key release. It does not start enclaves, call KMS, verify Nitro/COSE
signature chains, release keys, or mark production sealing as valid. The local
validators are intentionally fail-closed: a skipped security check can explain a
denial, but it cannot support a pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from leadpoet_verifier.attestation import (
    is_pcr0_allowed,
    load_pcr0_allowlist,
    validate_attestation_response_shape,
)

from .canonical import sha256_json
from .market_foundation import (
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    validate_market_sealing_posture,
    verify_market_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "full_sealing_kms_fixtures.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
PCR0_ALLOWLIST_PATH = REPO_ROOT / "pcr0_allowlist.json"

FULL_SEALING_CONTRACT_VERSION = "full_sealing_kms:v1:local_contract"
LOCAL_KMS_PENDING_REF = "kms_release:pending"

PROTECTED_SEALING_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "customer_email",
        "customer_private_data",
        "judge_prompt",
        "key_material",
        "kms_plaintext",
        "live_champion_ip",
        "password",
        "plaintext_key",
        "private_customer_data",
        "private_key",
        "raw_secret",
        "sealed_eval",
        "secret",
        "secret_access_key",
        "session_token",
        "token",
    }
)

PROTECTED_SEALING_MARKERS: tuple[str, ...] = (
    "-----begin private key",
    "akia",
    "aws secret",
    "customer email",
    "judge prompt",
    "kms plaintext",
    "live champion",
    "plaintext key",
    "private customer",
    "secret access key",
    "sealed eval",
    "sk-live",
)


class MeasurementGateStatus(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    BLOCKED = "blocked"
    PRODUCTION_VALID = "production_valid"


class ResearchEnclaveRunState(str, Enum):
    LOCAL_RECORD_ONLY = "local_record_only"
    BLOCKED = "blocked"
    PRODUCTION_ENCLAVE_STARTED = "production_enclave_started"


class KMSReleaseRequestState(str, Enum):
    LOCAL_REQUEST_STUB = "local_request_stub"
    BLOCKED = "blocked"
    PRODUCTION_REQUESTED = "production_requested"


class KMSReleaseDecisionState(str, Enum):
    LOCAL_DENIED_STUB = "local_denied_stub"
    DENIED = "denied"
    RELEASE_APPROVED = "release_approved"


@dataclass(frozen=True)
class EnclaveMeasurementGateRecord:
    gate_id: str
    run_id: str
    role: str
    attestation_ref: str
    attestation_response: dict[str, Any]
    pcr0_allowlist_ref: str
    pcr0: str
    enclave_measurement_ref: str
    egress_policy_ref: str
    kms_policy_ref: str
    shape_checked: bool = True
    shape_passed: bool = False
    pcr0_allowlist_checked: bool = True
    pcr0_allowed: bool = False
    signature_chain_checked: bool = False
    signature_chain_valid: bool = False
    egress_policy_checked: bool = True
    egress_policy_passed: bool = False
    equivocation_check_performed: bool = True
    equivocation_conflict: bool = False
    kms_policy_checked: bool = True
    kms_policy_passed: bool = False
    live_verification_performed: bool = False
    production_valid: bool = False
    status: str = MeasurementGateStatus.LOCAL_CONTRACT_STUB.value
    local_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnclaveMeasurementGateRecord":
        return cls(
            gate_id=str(data["gate_id"]),
            run_id=str(data["run_id"]),
            role=str(data["role"]),
            attestation_ref=str(data["attestation_ref"]),
            attestation_response=dict(data["attestation_response"]),
            pcr0_allowlist_ref=str(data["pcr0_allowlist_ref"]),
            pcr0=str(data.get("pcr0", data.get("attestation_response", {}).get("pcr0", ""))),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            egress_policy_ref=str(data["egress_policy_ref"]),
            kms_policy_ref=str(data["kms_policy_ref"]),
            shape_checked=bool(data.get("shape_checked", True)),
            shape_passed=bool(data.get("shape_passed", False)),
            pcr0_allowlist_checked=bool(data.get("pcr0_allowlist_checked", True)),
            pcr0_allowed=bool(data.get("pcr0_allowed", False)),
            signature_chain_checked=bool(data.get("signature_chain_checked", False)),
            signature_chain_valid=bool(data.get("signature_chain_valid", False)),
            egress_policy_checked=bool(data.get("egress_policy_checked", True)),
            egress_policy_passed=bool(data.get("egress_policy_passed", False)),
            equivocation_check_performed=bool(data.get("equivocation_check_performed", True)),
            equivocation_conflict=bool(data.get("equivocation_conflict", False)),
            kms_policy_checked=bool(data.get("kms_policy_checked", True)),
            kms_policy_passed=bool(data.get("kms_policy_passed", False)),
            live_verification_performed=bool(data.get("live_verification_performed", False)),
            production_valid=bool(data.get("production_valid", False)),
            status=str(data.get("status", MeasurementGateStatus.LOCAL_CONTRACT_STUB.value)),
            local_only=bool(data.get("local_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchEnclaveRunRecord:
    run_id: str
    ticket_id: str
    role: str
    engine_version_ref: str
    enclave_image_ref: str
    enclave_measurement_ref: str
    measurement_gate_ref: str
    kms_release_decision_ref: str
    egress_policy_ref: str
    evidence_bundle_ref: str
    output_hash: str
    cost_ledger_hash: str
    state: str = ResearchEnclaveRunState.LOCAL_RECORD_ONLY.value
    local_only: bool = True
    enclave_started: bool = False
    live_enclave_boot_performed: bool = False
    sealed_key_material_mounted: bool = False
    kms_key_released: bool = False
    production_network_egress_enabled: bool = False
    production_valid: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchEnclaveRunRecord":
        return cls(
            run_id=str(data["run_id"]),
            ticket_id=str(data["ticket_id"]),
            role=str(data["role"]),
            engine_version_ref=str(data["engine_version_ref"]),
            enclave_image_ref=str(data["enclave_image_ref"]),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            measurement_gate_ref=str(data["measurement_gate_ref"]),
            kms_release_decision_ref=str(data["kms_release_decision_ref"]),
            egress_policy_ref=str(data["egress_policy_ref"]),
            evidence_bundle_ref=str(data["evidence_bundle_ref"]),
            output_hash=str(data["output_hash"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            state=str(data.get("state", ResearchEnclaveRunState.LOCAL_RECORD_ONLY.value)),
            local_only=bool(data.get("local_only", True)),
            enclave_started=bool(data.get("enclave_started", False)),
            live_enclave_boot_performed=bool(data.get("live_enclave_boot_performed", False)),
            sealed_key_material_mounted=bool(data.get("sealed_key_material_mounted", False)),
            kms_key_released=bool(data.get("kms_key_released", False)),
            production_network_egress_enabled=bool(data.get("production_network_egress_enabled", False)),
            production_valid=bool(data.get("production_valid", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KMSKeyReleaseRequestRecord:
    request_id: str
    run_id: str
    requester_ref: str
    measurement_gate_ref: str
    pcr0_allowlist_ref: str
    enclave_measurement_ref: str
    kms_policy_ref: str
    sealed_key_ref: str
    purpose: str
    state: str = KMSReleaseRequestState.LOCAL_REQUEST_STUB.value
    contract_version: str = FULL_SEALING_CONTRACT_VERSION
    local_only: bool = True
    live_kms_request_performed: bool = False
    production_release_requested: bool = False
    plaintext_key_material_present: bool = False
    raw_secret_material_present: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KMSKeyReleaseRequestRecord":
        return cls(
            request_id=str(data["request_id"]),
            run_id=str(data["run_id"]),
            requester_ref=str(data["requester_ref"]),
            measurement_gate_ref=str(data["measurement_gate_ref"]),
            pcr0_allowlist_ref=str(data["pcr0_allowlist_ref"]),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            kms_policy_ref=str(data["kms_policy_ref"]),
            sealed_key_ref=str(data["sealed_key_ref"]),
            purpose=str(data["purpose"]),
            state=str(data.get("state", KMSReleaseRequestState.LOCAL_REQUEST_STUB.value)),
            contract_version=str(data.get("contract_version", FULL_SEALING_CONTRACT_VERSION)),
            local_only=bool(data.get("local_only", True)),
            live_kms_request_performed=bool(data.get("live_kms_request_performed", False)),
            production_release_requested=bool(data.get("production_release_requested", False)),
            plaintext_key_material_present=bool(data.get("plaintext_key_material_present", False)),
            raw_secret_material_present=bool(data.get("raw_secret_material_present", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KMSKeyReleaseDecisionRecord:
    decision_id: str
    request_id: str
    run_id: str
    measurement_gate_ref: str
    kms_policy_ref: str
    sealed_key_ref: str
    decision: str
    denial_reasons: tuple[str, ...]
    pcr0_allowlist_checked: bool
    pcr0_allowed: bool
    signature_chain_checked: bool
    signature_chain_valid: bool
    egress_policy_checked: bool
    egress_policy_passed: bool
    equivocation_check_performed: bool
    equivocation_conflict: bool
    kms_policy_checked: bool
    kms_policy_passed: bool
    state: str = KMSReleaseDecisionState.LOCAL_DENIED_STUB.value
    contract_version: str = FULL_SEALING_CONTRACT_VERSION
    local_only: bool = True
    live_kms_call_performed: bool = False
    kms_grant_created: bool = False
    key_release_performed: bool = False
    production_valid: bool = False
    ciphertext_key_ref: str = LOCAL_KMS_PENDING_REF
    plaintext_key_material_present: bool = False
    raw_secret_material_present: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KMSKeyReleaseDecisionRecord":
        return cls(
            decision_id=str(data["decision_id"]),
            request_id=str(data["request_id"]),
            run_id=str(data["run_id"]),
            measurement_gate_ref=str(data["measurement_gate_ref"]),
            kms_policy_ref=str(data["kms_policy_ref"]),
            sealed_key_ref=str(data["sealed_key_ref"]),
            decision=str(data["decision"]),
            denial_reasons=tuple(str(item) for item in data.get("denial_reasons", [])),
            pcr0_allowlist_checked=bool(data["pcr0_allowlist_checked"]),
            pcr0_allowed=bool(data["pcr0_allowed"]),
            signature_chain_checked=bool(data["signature_chain_checked"]),
            signature_chain_valid=bool(data["signature_chain_valid"]),
            egress_policy_checked=bool(data["egress_policy_checked"]),
            egress_policy_passed=bool(data["egress_policy_passed"]),
            equivocation_check_performed=bool(data["equivocation_check_performed"]),
            equivocation_conflict=bool(data["equivocation_conflict"]),
            kms_policy_checked=bool(data["kms_policy_checked"]),
            kms_policy_passed=bool(data["kms_policy_passed"]),
            state=str(data.get("state", KMSReleaseDecisionState.LOCAL_DENIED_STUB.value)),
            contract_version=str(data.get("contract_version", FULL_SEALING_CONTRACT_VERSION)),
            local_only=bool(data.get("local_only", True)),
            live_kms_call_performed=bool(data.get("live_kms_call_performed", False)),
            kms_grant_created=bool(data.get("kms_grant_created", False)),
            key_release_performed=bool(data.get("key_release_performed", False)),
            production_valid=bool(data.get("production_valid", False)),
            ciphertext_key_ref=str(data.get("ciphertext_key_ref", LOCAL_KMS_PENDING_REF)),
            plaintext_key_material_present=bool(data.get("plaintext_key_material_present", False)),
            raw_secret_material_present=bool(data.get("raw_secret_material_present", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["denial_reasons"] = list(self.denial_reasons)
        return data


def build_enclave_measurement_gate(
    *,
    run_id: str,
    role: str,
    attestation_ref: str,
    attestation_response: Mapping[str, Any],
    pcr0_allowlist_ref: str,
    enclave_measurement_ref: str,
    egress_policy_ref: str,
    kms_policy_ref: str,
    allowlist: Mapping[str, Sequence[Mapping[str, Any]]],
    egress_policy_passed: bool,
    kms_policy_passed: bool,
    equivocation_conflict: bool = False,
) -> EnclaveMeasurementGateRecord:
    shape = validate_attestation_response_shape(attestation_response)
    pcr0 = str(attestation_response.get("pcr0", ""))
    pcr0_allowed = is_pcr0_allowed(pcr0, allowlist, role=role)
    payload = {
        "run_id": run_id,
        "role": role,
        "attestation_ref": attestation_ref,
        "pcr0": pcr0,
        "egress_policy_ref": egress_policy_ref,
        "kms_policy_ref": kms_policy_ref,
    }
    local_passed = bool(shape["passed"]) and pcr0_allowed and egress_policy_passed and kms_policy_passed and not equivocation_conflict
    return EnclaveMeasurementGateRecord(
        gate_id="measurement_gate:" + sha256_json(payload).split(":", 1)[1][:16],
        run_id=run_id,
        role=role,
        attestation_ref=attestation_ref,
        attestation_response=dict(attestation_response),
        pcr0_allowlist_ref=pcr0_allowlist_ref,
        pcr0=pcr0,
        enclave_measurement_ref=enclave_measurement_ref,
        egress_policy_ref=egress_policy_ref,
        kms_policy_ref=kms_policy_ref,
        shape_checked=True,
        shape_passed=bool(shape["passed"]),
        pcr0_allowlist_checked=True,
        pcr0_allowed=pcr0_allowed,
        signature_chain_checked=False,
        signature_chain_valid=False,
        egress_policy_checked=True,
        egress_policy_passed=egress_policy_passed,
        equivocation_check_performed=True,
        equivocation_conflict=equivocation_conflict,
        kms_policy_checked=True,
        kms_policy_passed=kms_policy_passed,
        live_verification_performed=False,
        production_valid=False,
        status=MeasurementGateStatus.LOCAL_CONTRACT_STUB.value if local_passed else MeasurementGateStatus.BLOCKED.value,
        local_only=True,
    )


def build_research_enclave_run_record(
    *,
    gate: EnclaveMeasurementGateRecord | Mapping[str, Any],
    ticket_id: str,
    engine_version_ref: str,
    enclave_image_ref: str,
    evidence_bundle_ref: str,
    output_hash: str,
    cost_ledger_hash: str,
    kms_release_decision_ref: str = LOCAL_KMS_PENDING_REF,
) -> ResearchEnclaveRunRecord:
    if not isinstance(gate, EnclaveMeasurementGateRecord):
        gate = EnclaveMeasurementGateRecord.from_mapping(gate)
    payload = {
        "run_id": gate.run_id,
        "ticket_id": ticket_id,
        "gate_id": gate.gate_id,
        "output_hash": output_hash,
    }
    return ResearchEnclaveRunRecord(
        run_id="research_enclave_run:" + sha256_json(payload).split(":", 1)[1][:16],
        ticket_id=ticket_id,
        role=gate.role,
        engine_version_ref=engine_version_ref,
        enclave_image_ref=enclave_image_ref,
        enclave_measurement_ref=gate.enclave_measurement_ref,
        measurement_gate_ref=gate.gate_id,
        kms_release_decision_ref=kms_release_decision_ref,
        egress_policy_ref=gate.egress_policy_ref,
        evidence_bundle_ref=evidence_bundle_ref,
        output_hash=output_hash,
        cost_ledger_hash=cost_ledger_hash,
    )


def build_kms_key_release_request(
    *,
    gate: EnclaveMeasurementGateRecord | Mapping[str, Any],
    requester_ref: str,
    sealed_key_ref: str,
    purpose: str,
) -> KMSKeyReleaseRequestRecord:
    if not isinstance(gate, EnclaveMeasurementGateRecord):
        gate = EnclaveMeasurementGateRecord.from_mapping(gate)
    payload = {
        "run_id": gate.run_id,
        "requester_ref": requester_ref,
        "gate_id": gate.gate_id,
        "sealed_key_ref": sealed_key_ref,
        "purpose": purpose,
    }
    return KMSKeyReleaseRequestRecord(
        request_id="kms_release_request:" + sha256_json(payload).split(":", 1)[1][:16],
        run_id=gate.run_id,
        requester_ref=requester_ref,
        measurement_gate_ref=gate.gate_id,
        pcr0_allowlist_ref=gate.pcr0_allowlist_ref,
        enclave_measurement_ref=gate.enclave_measurement_ref,
        kms_policy_ref=gate.kms_policy_ref,
        sealed_key_ref=sealed_key_ref,
        purpose=purpose,
    )


def build_kms_key_release_decision(
    *,
    request: KMSKeyReleaseRequestRecord | Mapping[str, Any],
    gate: EnclaveMeasurementGateRecord | Mapping[str, Any],
) -> KMSKeyReleaseDecisionRecord:
    if not isinstance(request, KMSKeyReleaseRequestRecord):
        request = KMSKeyReleaseRequestRecord.from_mapping(request)
    if not isinstance(gate, EnclaveMeasurementGateRecord):
        gate = EnclaveMeasurementGateRecord.from_mapping(gate)
    denial_reasons: list[str] = []
    if not gate.signature_chain_checked or not gate.signature_chain_valid:
        denial_reasons.append("signature_chain_not_verified")
    if not gate.live_verification_performed:
        denial_reasons.append("live_kms_disabled")
    if not gate.pcr0_allowlist_checked or not gate.pcr0_allowed:
        denial_reasons.append("pcr0_not_allowed")
    if not gate.egress_policy_checked or not gate.egress_policy_passed:
        denial_reasons.append("egress_policy_not_passed")
    if not gate.equivocation_check_performed or gate.equivocation_conflict:
        denial_reasons.append("equivocation_not_cleared")
    if not gate.kms_policy_checked or not gate.kms_policy_passed:
        denial_reasons.append("kms_policy_not_passed")
    payload = {
        "request_id": request.request_id,
        "gate_id": gate.gate_id,
        "denial_reasons": sorted(denial_reasons),
    }
    return KMSKeyReleaseDecisionRecord(
        decision_id="kms_release_decision:" + sha256_json(payload).split(":", 1)[1][:16],
        request_id=request.request_id,
        run_id=request.run_id,
        measurement_gate_ref=gate.gate_id,
        kms_policy_ref=gate.kms_policy_ref,
        sealed_key_ref=request.sealed_key_ref,
        decision="denied",
        denial_reasons=tuple(sorted(denial_reasons)),
        pcr0_allowlist_checked=gate.pcr0_allowlist_checked,
        pcr0_allowed=gate.pcr0_allowed,
        signature_chain_checked=gate.signature_chain_checked,
        signature_chain_valid=gate.signature_chain_valid,
        egress_policy_checked=gate.egress_policy_checked,
        egress_policy_passed=gate.egress_policy_passed,
        equivocation_check_performed=gate.equivocation_check_performed,
        equivocation_conflict=gate.equivocation_conflict,
        kms_policy_checked=gate.kms_policy_checked,
        kms_policy_passed=gate.kms_policy_passed,
    )


def validate_enclave_measurement_gate_record(
    record: EnclaveMeasurementGateRecord | Mapping[str, Any],
    *,
    allowlist: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_sealing_payload_errors(raw)
    if not isinstance(record, EnclaveMeasurementGateRecord):
        record = EnclaveMeasurementGateRecord.from_mapping(record)
    if record.status not in {status.value for status in MeasurementGateStatus}:
        errors.append(f"unknown measurement gate status: {record.status}")
    for field in (
        "gate_id",
        "run_id",
        "role",
        "attestation_ref",
        "pcr0_allowlist_ref",
        "pcr0",
        "enclave_measurement_ref",
        "egress_policy_ref",
        "kms_policy_ref",
    ):
        if not getattr(record, field):
            errors.append(f"measurement gate requires {field}")
    if not record.gate_id.startswith("measurement_gate:"):
        errors.append("gate_id must be measurement_gate:-prefixed")
    if not record.attestation_ref.startswith("attestation:"):
        errors.append("attestation_ref must be attestation:-prefixed")
    if not record.enclave_measurement_ref.startswith("enclave_measurement:"):
        errors.append("enclave_measurement_ref must be enclave_measurement:-prefixed")
    if not record.kms_policy_ref.startswith("kms_policy:"):
        errors.append("kms_policy_ref must be kms_policy:-prefixed")
    if not record.shape_checked:
        errors.append("attestation shape check must not be skipped")
    shape = validate_attestation_response_shape(record.attestation_response)
    if record.shape_passed != bool(shape["passed"]):
        errors.append("shape_passed does not match attestation response shape")
    if not record.shape_passed:
        errors.append("attestation response shape failed")
    if not record.pcr0_allowlist_checked:
        errors.append("PCR0 allowlist check must not be skipped")
    allowlist = allowlist or load_pcr0_allowlist(str(PCR0_ALLOWLIST_PATH))
    expected_allowed = is_pcr0_allowed(record.pcr0, allowlist, role=record.role)
    if record.pcr0_allowed != expected_allowed:
        errors.append("pcr0_allowed does not match allowlist")
    if not record.pcr0_allowed:
        errors.append("PCR0 is not allowlisted")
    if not record.egress_policy_checked:
        errors.append("egress-policy check must not be skipped")
    if not record.egress_policy_passed:
        errors.append("egress policy must pass before key release eligibility")
    if not record.equivocation_check_performed:
        errors.append("equivocation check must not be skipped")
    if record.equivocation_conflict:
        errors.append("equivocation conflict blocks sealing")
    if not record.kms_policy_checked:
        errors.append("KMS policy check must not be skipped")
    if not record.kms_policy_passed:
        errors.append("KMS policy must pass before key release eligibility")
    if record.signature_chain_checked and not record.live_verification_performed:
        errors.append("local P2.6 records must not claim Nitro/COSE signature-chain verification")
    if record.signature_chain_valid and not record.signature_chain_checked:
        errors.append("signature_chain_valid requires signature_chain_checked")
    if record.live_verification_performed:
        errors.append("live_verification_performed must remain false in P2.6 local contracts")
    if record.production_valid:
        errors.append("P2.6 local measurement gates must not claim production_valid")
    if not record.local_only:
        errors.append("P2.6 measurement gates must remain local_only")
    if record.status == MeasurementGateStatus.PRODUCTION_VALID.value:
        if not record.signature_chain_checked:
            errors.append("signature-chain check must not be skipped for production sealing")
        if not record.signature_chain_valid:
            errors.append("signature-chain must be valid for production sealing")
        if not record.live_verification_performed:
            errors.append("production sealing requires live_verification_performed")
        errors.append("production_valid measurement gate state is disabled in P2.6 local contracts")
    if record.status == MeasurementGateStatus.LOCAL_CONTRACT_STUB.value and record.production_valid:
        errors.append("local measurement gate cannot claim production_valid")
    return errors


def validate_research_enclave_run_record(
    record: ResearchEnclaveRunRecord | Mapping[str, Any],
    *,
    gate: Optional[EnclaveMeasurementGateRecord | Mapping[str, Any]] = None,
    decision: Optional[KMSKeyReleaseDecisionRecord | Mapping[str, Any]] = None,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_sealing_payload_errors(raw)
    if not isinstance(record, ResearchEnclaveRunRecord):
        record = ResearchEnclaveRunRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ResearchEnclaveRunState}:
        errors.append(f"unknown research enclave run state: {record.state}")
    for field in (
        "run_id",
        "ticket_id",
        "role",
        "engine_version_ref",
        "enclave_image_ref",
        "enclave_measurement_ref",
        "measurement_gate_ref",
        "kms_release_decision_ref",
        "egress_policy_ref",
        "evidence_bundle_ref",
        "output_hash",
        "cost_ledger_hash",
    ):
        if not getattr(record, field):
            errors.append(f"research enclave run requires {field}")
    for field in ("output_hash", "cost_ledger_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    if not record.run_id.startswith("research_enclave_run:"):
        errors.append("run_id must be research_enclave_run:-prefixed")
    if not record.measurement_gate_ref.startswith("measurement_gate:"):
        errors.append("measurement_gate_ref must be measurement_gate:-prefixed")
    if not record.enclave_measurement_ref.startswith("enclave_measurement:"):
        errors.append("enclave_measurement_ref must be enclave_measurement:-prefixed")
    if not record.local_only:
        errors.append("P2.6 research enclave run records must remain local_only")
    if record.enclave_started:
        errors.append("P2.6 local contracts must not start research enclaves")
    if record.live_enclave_boot_performed:
        errors.append("live_enclave_boot_performed must remain false")
    if record.sealed_key_material_mounted:
        errors.append("sealed key material must not be mounted in local records")
    if record.kms_key_released:
        errors.append("P2.6 local contracts must not release KMS keys")
    if record.production_network_egress_enabled:
        errors.append("production network egress must remain disabled")
    if record.production_valid:
        errors.append("P2.6 research enclave runs must not claim production_valid")
    if record.state == ResearchEnclaveRunState.PRODUCTION_ENCLAVE_STARTED.value:
        errors.append("production_enclave_started state is disabled in P2.6 local contracts")
    if gate is not None:
        if not isinstance(gate, EnclaveMeasurementGateRecord):
            gate = EnclaveMeasurementGateRecord.from_mapping(gate)
        if record.measurement_gate_ref != gate.gate_id:
            errors.append("research enclave run measurement_gate_ref mismatch")
        if record.enclave_measurement_ref != gate.enclave_measurement_ref:
            errors.append("research enclave run enclave_measurement_ref mismatch")
        if record.egress_policy_ref != gate.egress_policy_ref:
            errors.append("research enclave run egress_policy_ref mismatch")
    if decision is not None:
        if not isinstance(decision, KMSKeyReleaseDecisionRecord):
            decision = KMSKeyReleaseDecisionRecord.from_mapping(decision)
        if record.kms_release_decision_ref != decision.decision_id:
            errors.append("research enclave run kms_release_decision_ref mismatch")
        if record.kms_key_released != decision.key_release_performed:
            errors.append("research enclave run KMS release flag mismatch")
    return errors


def validate_kms_key_release_request_record(
    record: KMSKeyReleaseRequestRecord | Mapping[str, Any],
    *,
    gate: Optional[EnclaveMeasurementGateRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_sealing_payload_errors(raw)
    if not isinstance(record, KMSKeyReleaseRequestRecord):
        record = KMSKeyReleaseRequestRecord.from_mapping(record)
    if record.state not in {state.value for state in KMSReleaseRequestState}:
        errors.append(f"unknown KMS release request state: {record.state}")
    for field in (
        "request_id",
        "run_id",
        "requester_ref",
        "measurement_gate_ref",
        "pcr0_allowlist_ref",
        "enclave_measurement_ref",
        "kms_policy_ref",
        "sealed_key_ref",
        "purpose",
    ):
        if not getattr(record, field):
            errors.append(f"KMS key-release request requires {field}")
    if not record.request_id.startswith("kms_release_request:"):
        errors.append("request_id must be kms_release_request:-prefixed")
    if not record.measurement_gate_ref.startswith("measurement_gate:"):
        errors.append("measurement_gate_ref must be measurement_gate:-prefixed")
    if not record.enclave_measurement_ref.startswith("enclave_measurement:"):
        errors.append("enclave_measurement_ref must be enclave_measurement:-prefixed")
    if not record.kms_policy_ref.startswith("kms_policy:"):
        errors.append("kms_policy_ref must be kms_policy:-prefixed")
    if not record.sealed_key_ref.startswith("kms_key:"):
        errors.append("sealed_key_ref must be kms_key:-prefixed")
    if record.contract_version != FULL_SEALING_CONTRACT_VERSION:
        errors.append("contract_version must match P2.6 full-sealing contract")
    if not record.local_only:
        errors.append("P2.6 KMS release requests must remain local_only")
    if record.live_kms_request_performed:
        errors.append("P2.6 local contracts must not perform live KMS requests")
    if record.production_release_requested:
        errors.append("P2.6 local contracts must not request production KMS release")
    if record.state == KMSReleaseRequestState.PRODUCTION_REQUESTED.value:
        errors.append("production_requested state is disabled in P2.6 local contracts")
    if record.plaintext_key_material_present:
        errors.append("KMS release request must not contain plaintext key material")
    if record.raw_secret_material_present:
        errors.append("KMS release request must not contain raw secret material")
    if gate is not None:
        if not isinstance(gate, EnclaveMeasurementGateRecord):
            gate = EnclaveMeasurementGateRecord.from_mapping(gate)
        if record.run_id != gate.run_id:
            errors.append("KMS release request run_id mismatch")
        if record.measurement_gate_ref != gate.gate_id:
            errors.append("KMS release request measurement_gate_ref mismatch")
        if record.pcr0_allowlist_ref != gate.pcr0_allowlist_ref:
            errors.append("KMS release request pcr0_allowlist_ref mismatch")
        if record.enclave_measurement_ref != gate.enclave_measurement_ref:
            errors.append("KMS release request enclave_measurement_ref mismatch")
        if record.kms_policy_ref != gate.kms_policy_ref:
            errors.append("KMS release request kms_policy_ref mismatch")
    return errors


def validate_kms_key_release_decision_record(
    record: KMSKeyReleaseDecisionRecord | Mapping[str, Any],
    *,
    request: Optional[KMSKeyReleaseRequestRecord | Mapping[str, Any]] = None,
    gate: Optional[EnclaveMeasurementGateRecord | Mapping[str, Any]] = None,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_sealing_payload_errors(raw)
    if not isinstance(record, KMSKeyReleaseDecisionRecord):
        record = KMSKeyReleaseDecisionRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in KMSReleaseDecisionState}:
        errors.append(f"unknown KMS release decision state: {record.state}")
    if record.decision not in {"denied", "release_approved"}:
        errors.append("decision must be denied or release_approved")
    for field in ("decision_id", "request_id", "run_id", "measurement_gate_ref", "kms_policy_ref", "sealed_key_ref"):
        if not getattr(record, field):
            errors.append(f"KMS key-release decision requires {field}")
    if not record.decision_id.startswith("kms_release_decision:"):
        errors.append("decision_id must be kms_release_decision:-prefixed")
    if record.contract_version != FULL_SEALING_CONTRACT_VERSION:
        errors.append("contract_version must match P2.6 full-sealing contract")
    if not record.pcr0_allowlist_checked:
        errors.append("KMS decision cannot skip PCR0 allowlist check")
    if not record.pcr0_allowed:
        errors.append("KMS decision requires allowlisted PCR0")
    if not record.egress_policy_checked:
        errors.append("KMS decision cannot skip egress-policy check")
    if not record.egress_policy_passed:
        errors.append("KMS decision requires passing egress policy")
    if not record.equivocation_check_performed:
        errors.append("KMS decision cannot skip equivocation check")
    if record.equivocation_conflict:
        errors.append("equivocation conflict blocks KMS release")
    if not record.kms_policy_checked:
        errors.append("KMS decision cannot skip KMS policy check")
    if not record.kms_policy_passed:
        errors.append("KMS decision requires passing KMS policy")
    if not record.signature_chain_checked and record.decision == "release_approved":
        errors.append("signature-chain check must not be skipped for KMS release")
    if not record.signature_chain_valid and record.decision == "release_approved":
        errors.append("signature-chain must be valid for KMS release")
    if not record.signature_chain_checked and "signature_chain_not_verified" not in record.denial_reasons:
        errors.append("denied KMS release must record signature_chain_not_verified when signature chain is skipped")
    if not record.live_kms_call_performed and "live_kms_disabled" not in record.denial_reasons:
        errors.append("denied KMS release must record live_kms_disabled when live KMS is disabled")
    if not record.local_only:
        errors.append("P2.6 KMS decisions must remain local_only")
    if record.live_kms_call_performed:
        errors.append("P2.6 local contracts must not perform live KMS calls")
    if record.kms_grant_created:
        errors.append("P2.6 local contracts must not create KMS grants")
    if record.key_release_performed:
        errors.append("P2.6 local contracts must not release KMS keys")
    if record.production_valid:
        errors.append("P2.6 KMS decisions must not claim production_valid")
    if record.decision == "release_approved":
        errors.append("release_approved decision is disabled in P2.6 local contracts")
    if record.state == KMSReleaseDecisionState.RELEASE_APPROVED.value:
        errors.append("release_approved state is disabled in P2.6 local contracts")
    if record.ciphertext_key_ref != LOCAL_KMS_PENDING_REF and not record.key_release_performed:
        errors.append("ciphertext_key_ref must remain pending unless key release is performed")
    if record.plaintext_key_material_present:
        errors.append("KMS decision must not contain plaintext key material")
    if record.raw_secret_material_present:
        errors.append("KMS decision must not contain raw secret material")
    if request is not None:
        if not isinstance(request, KMSKeyReleaseRequestRecord):
            request = KMSKeyReleaseRequestRecord.from_mapping(request)
        request_errors = validate_kms_key_release_request_record(request, gate=gate)
        if request_errors:
            errors.append("source KMS release request is invalid: " + "; ".join(request_errors))
        if record.request_id != request.request_id:
            errors.append("KMS decision request_id mismatch")
        if record.run_id != request.run_id:
            errors.append("KMS decision run_id mismatch")
        if record.measurement_gate_ref != request.measurement_gate_ref:
            errors.append("KMS decision measurement_gate_ref mismatch")
        if record.kms_policy_ref != request.kms_policy_ref:
            errors.append("KMS decision kms_policy_ref mismatch")
        if record.sealed_key_ref != request.sealed_key_ref:
            errors.append("KMS decision sealed_key_ref mismatch")
    if gate is not None:
        if not isinstance(gate, EnclaveMeasurementGateRecord):
            gate = EnclaveMeasurementGateRecord.from_mapping(gate)
        if record.measurement_gate_ref != gate.gate_id:
            errors.append("KMS decision measurement_gate_ref does not match gate")
        check_pairs = (
            ("pcr0_allowlist_checked", record.pcr0_allowlist_checked, gate.pcr0_allowlist_checked),
            ("pcr0_allowed", record.pcr0_allowed, gate.pcr0_allowed),
            ("signature_chain_checked", record.signature_chain_checked, gate.signature_chain_checked),
            ("signature_chain_valid", record.signature_chain_valid, gate.signature_chain_valid),
            ("egress_policy_checked", record.egress_policy_checked, gate.egress_policy_checked),
            ("egress_policy_passed", record.egress_policy_passed, gate.egress_policy_passed),
            ("equivocation_check_performed", record.equivocation_check_performed, gate.equivocation_check_performed),
            ("equivocation_conflict", record.equivocation_conflict, gate.equivocation_conflict),
            ("kms_policy_checked", record.kms_policy_checked, gate.kms_policy_checked),
            ("kms_policy_passed", record.kms_policy_passed, gate.kms_policy_passed),
        )
        for field, decision_value, gate_value in check_pairs:
            if decision_value != gate_value:
                errors.append(f"KMS decision {field} does not match measurement gate")
    return errors


def verify_research_lab_full_sealing_kms(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    allowlist = load_pcr0_allowlist(str(PCR0_ALLOWLIST_PATH))
    fixture = _load_fixture(Path(fixture_path))
    _assert_no_actual_secret_like_values(fixture)

    gate = build_enclave_measurement_gate(
        run_id=fixture["measurement_gate_request"]["run_id"],
        role=fixture["measurement_gate_request"]["role"],
        attestation_ref=fixture["measurement_gate_request"]["attestation_ref"],
        attestation_response=fixture["measurement_gate_request"]["attestation_response"],
        pcr0_allowlist_ref=fixture["measurement_gate_request"]["pcr0_allowlist_ref"],
        enclave_measurement_ref=fixture["measurement_gate_request"]["enclave_measurement_ref"],
        egress_policy_ref=fixture["measurement_gate_request"]["egress_policy_ref"],
        kms_policy_ref=fixture["measurement_gate_request"]["kms_policy_ref"],
        allowlist=allowlist,
        egress_policy_passed=fixture["measurement_gate_request"]["egress_policy_passed"],
        kms_policy_passed=fixture["measurement_gate_request"]["kms_policy_passed"],
    )
    _assert(gate.to_dict() == fixture["expected_measurement_gate"], "measurement gate is deterministic")
    _assert(not validate_enclave_measurement_gate_record(gate, allowlist=allowlist), "measurement gate validates")
    for invalid in fixture["invalid_measurement_gates"]:
        record = _fixture_record(fixture, invalid, "expected_measurement_gate")
        errors = validate_enclave_measurement_gate_record(record, allowlist=allowlist)
        _assert(errors, f"invalid measurement gate fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    standalone_gate = _fixture_record(
        fixture,
        fixture["standalone_default_allowlist_gate"],
        "expected_measurement_gate",
    )
    standalone_errors = validate_enclave_measurement_gate_record(standalone_gate)
    _assert(standalone_errors, "standalone measurement gate uses repo PCR0 allowlist by default")
    _assert_expected_error(standalone_errors, fixture["standalone_default_allowlist_gate"])

    kms_request = build_kms_key_release_request(
        gate=gate,
        requester_ref=fixture["kms_request_input"]["requester_ref"],
        sealed_key_ref=fixture["kms_request_input"]["sealed_key_ref"],
        purpose=fixture["kms_request_input"]["purpose"],
    )
    _assert(kms_request.to_dict() == fixture["expected_kms_request"], "KMS request is deterministic")
    _assert(not validate_kms_key_release_request_record(kms_request, gate=gate), "KMS request validates")
    for invalid in fixture["invalid_kms_requests"]:
        record = _fixture_record(fixture, invalid, "expected_kms_request")
        errors = validate_kms_key_release_request_record(record, gate=gate)
        _assert(errors, f"invalid KMS request fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    kms_decision = build_kms_key_release_decision(request=kms_request, gate=gate)
    _assert(kms_decision.to_dict() == fixture["expected_kms_decision"], "KMS decision is deterministic")
    _assert(
        not validate_kms_key_release_decision_record(kms_decision, request=kms_request, gate=gate),
        "KMS decision validates",
    )
    for invalid in fixture["invalid_kms_decisions"]:
        record = _fixture_record(fixture, invalid, "expected_kms_decision")
        errors = validate_kms_key_release_decision_record(record, request=kms_request, gate=gate)
        _assert(errors, f"invalid KMS decision fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    unsafe_decision_errors = validate_kms_key_release_decision_record(
        kms_decision,
        request=kms_request,
        gate=gate,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_decision_errors, "unsafe Phase 2 guards block KMS release decision")

    enclave_run = build_research_enclave_run_record(
        gate=gate,
        ticket_id=fixture["enclave_run_input"]["ticket_id"],
        engine_version_ref=fixture["enclave_run_input"]["engine_version_ref"],
        enclave_image_ref=fixture["enclave_run_input"]["enclave_image_ref"],
        evidence_bundle_ref=fixture["enclave_run_input"]["evidence_bundle_ref"],
        output_hash=fixture["enclave_run_input"]["output_hash"],
        cost_ledger_hash=fixture["enclave_run_input"]["cost_ledger_hash"],
        kms_release_decision_ref=kms_decision.decision_id,
    )
    _assert(enclave_run.to_dict() == fixture["expected_enclave_run"], "research enclave run is deterministic")
    _assert(
        not validate_research_enclave_run_record(enclave_run, gate=gate, decision=kms_decision),
        "research enclave run validates",
    )
    for invalid in fixture["invalid_enclave_runs"]:
        record = _fixture_record(fixture, invalid, "expected_enclave_run")
        errors = validate_research_enclave_run_record(record, gate=gate, decision=kms_decision)
        _assert(errors, f"invalid enclave run fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    unsafe_run_errors = validate_research_enclave_run_record(
        enclave_run,
        gate=gate,
        decision=kms_decision,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_run_errors, "unsafe Phase 2 guards block research enclave run")

    sealing_errors = validate_market_sealing_posture(fixture["market_sealing_posture_link"])
    _assert(sealing_errors, "P2.6 fixture cannot claim P2.0 production sealing posture")
    _assert_expected_error(sealing_errors, fixture["market_sealing_posture_link"])

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "gate_id": gate.gate_id,
        "gate_status": gate.status,
        "kms_request_id": kms_request.request_id,
        "kms_decision_id": kms_decision.decision_id,
        "kms_decision": kms_decision.decision,
        "enclave_run_id": enclave_run.run_id,
        "denial_reasons": list(kms_decision.denial_reasons),
    }


def _protected_sealing_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_sealing_material(record))
    if not found:
        return []
    return ["P2.6 sealing/KMS payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_sealing_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_SEALING_KEYS and not key_text.endswith(("_ref", "_refs", "_hash")):
                found.add(key_path)
            found.update(_find_protected_sealing_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_sealing_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_SEALING_MARKERS:
            if marker in lower:
                found.add(path or marker)
    return found


def _assert_no_actual_secret_like_values(value: Any) -> None:
    """Reject fixture values that look like real secrets, while allowing labels."""
    secret_markers = ("akia", "sk-live-", "-----begin private key", "aws_secret_access_key=")
    values = _flatten_strings(value)
    for item in values:
        lower = item.lower()
        if any(marker in lower for marker in secret_markers):
            raise AssertionError("P2.6 fixtures must not contain actual secret-like values")


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items: list[str] = []
        for nested in value.values():
            items.extend(_flatten_strings(nested))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for nested in value:
            items.extend(_flatten_strings(nested))
        return items
    if isinstance(value, str):
        return [value]
    return []


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_record(fixture: Mapping[str, Any], invalid: Mapping[str, Any], default_base: str) -> dict[str, Any]:
    base = dict(fixture[str(invalid.get("base", default_base))])
    return _deep_merge(base, dict(invalid.get("overrides", {})))


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
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
