"""Phase 1 validator-flow records.

P1.6 models validator-side verification decisions for attestation, PCR0
allowlists, equivocation, open-verifier reruns, sampled judge checks, sampled
liveness checks, and local consensus summaries. It does not emit validator
weights, call sealed judges, perform live liveness HTTP, or change production
consensus behavior.

This local slice fixes the plan's PCR0-skip and equivocation-skip risks in the
record/verifier layer. It deliberately does not claim Nitro/COSE signature-chain
verification; production attestation crypto stays a later live-validator
responsibility.
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
from leadpoet_verifier.golden_vectors import run_golden_vectors

from .canonical import sha256_json
from .fabric import (
    AttestationAnchorStub,
    validate_attestation_anchor_stub,
    verify_research_lab_fabric,
)
from .loop_foundation import (
    LoopWorkflowGuards,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "validator_flow_fixtures.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
PCR0_ALLOWLIST_PATH = REPO_ROOT / "pcr0_allowlist.json"


class ValidatorCheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class SampleRecordStatus(str, Enum):
    SAMPLED_RECORD_ONLY = "sampled_record_only"
    BLOCKED = "blocked"


class ConsensusState(str, Enum):
    LOCAL_SUMMARY_ONLY = "local_summary_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AttestationVerificationRecord:
    verification_id: str
    validator_ref: str
    run_id: str
    attestation_ref: str
    role: str
    attestation_response: dict[str, Any]
    pcr0_allowlist_ref: str
    pcr0: str
    shape_checked: bool
    shape_passed: bool
    pcr0_allowlist_checked: bool
    pcr0_allowed: bool
    equivocation_check_performed: bool
    equivocation_conflict: bool
    observed_output_hashes: tuple[str, ...]
    conflicting_attestation_refs: tuple[str, ...] = ()
    signature_chain_checked: bool = False
    production_valid: bool = False
    status: str = ValidatorCheckStatus.FAILED.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AttestationVerificationRecord":
        return cls(
            verification_id=str(data["verification_id"]),
            validator_ref=str(data["validator_ref"]),
            run_id=str(data["run_id"]),
            attestation_ref=str(data["attestation_ref"]),
            role=str(data["role"]),
            attestation_response=dict(data["attestation_response"]),
            pcr0_allowlist_ref=str(data["pcr0_allowlist_ref"]),
            pcr0=str(data.get("pcr0", data.get("attestation_response", {}).get("pcr0", ""))),
            shape_checked=bool(data["shape_checked"]),
            shape_passed=bool(data["shape_passed"]),
            pcr0_allowlist_checked=bool(data["pcr0_allowlist_checked"]),
            pcr0_allowed=bool(data["pcr0_allowed"]),
            equivocation_check_performed=bool(data["equivocation_check_performed"]),
            equivocation_conflict=bool(data["equivocation_conflict"]),
            observed_output_hashes=tuple(str(item) for item in data.get("observed_output_hashes", [])),
            conflicting_attestation_refs=tuple(str(item) for item in data.get("conflicting_attestation_refs", [])),
            signature_chain_checked=bool(data.get("signature_chain_checked", False)),
            production_valid=bool(data.get("production_valid", False)),
            status=str(data.get("status", ValidatorCheckStatus.FAILED.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_output_hashes"] = list(self.observed_output_hashes)
        data["conflicting_attestation_refs"] = list(self.conflicting_attestation_refs)
        return data


@dataclass(frozen=True)
class OpenVerifierRerunRecord:
    rerun_id: str
    validator_ref: str
    run_id: str
    fixture_ref: str
    fixture_hash: str
    expected_result_hash: str
    actual_result_hash: str
    byte_identical: bool
    golden_vector_errors: tuple[str, ...] = ()
    open_verifier_hash: str = "sha256:open-verifier-placeholder"
    local_only: bool = True
    rerun_performed_locally: bool = True
    live_network_used: bool = False
    production_consensus_write: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OpenVerifierRerunRecord":
        return cls(
            rerun_id=str(data["rerun_id"]),
            validator_ref=str(data["validator_ref"]),
            run_id=str(data["run_id"]),
            fixture_ref=str(data["fixture_ref"]),
            fixture_hash=str(data["fixture_hash"]),
            expected_result_hash=str(data["expected_result_hash"]),
            actual_result_hash=str(data["actual_result_hash"]),
            byte_identical=bool(data["byte_identical"]),
            golden_vector_errors=tuple(str(item) for item in data.get("golden_vector_errors", [])),
            open_verifier_hash=str(data.get("open_verifier_hash", "sha256:open-verifier-placeholder")),
            local_only=bool(data.get("local_only", True)),
            rerun_performed_locally=bool(data.get("rerun_performed_locally", True)),
            live_network_used=bool(data.get("live_network_used", False)),
            production_consensus_write=bool(data.get("production_consensus_write", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["golden_vector_errors"] = list(self.golden_vector_errors)
        return data


@dataclass(frozen=True)
class JudgeRequerySampleRecord:
    sample_id: str
    validator_ref: str
    run_id: str
    evidence_bundle_ref: str
    original_verdict_hash: str
    judge_version_hash: str
    sample_rate: float
    status: str = SampleRecordStatus.SAMPLED_RECORD_ONLY.value
    sampled: bool = True
    live_requery_performed: bool = False
    sealed_judge_accessed: bool = False
    production_consensus_write: bool = False
    requery_result_hash: str = "hash:pending"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "JudgeRequerySampleRecord":
        return cls(
            sample_id=str(data["sample_id"]),
            validator_ref=str(data["validator_ref"]),
            run_id=str(data["run_id"]),
            evidence_bundle_ref=str(data["evidence_bundle_ref"]),
            original_verdict_hash=str(data["original_verdict_hash"]),
            judge_version_hash=str(data["judge_version_hash"]),
            sample_rate=float(data["sample_rate"]),
            status=str(data.get("status", SampleRecordStatus.SAMPLED_RECORD_ONLY.value)),
            sampled=bool(data.get("sampled", True)),
            live_requery_performed=bool(data.get("live_requery_performed", False)),
            sealed_judge_accessed=bool(data.get("sealed_judge_accessed", False)),
            production_consensus_write=bool(data.get("production_consensus_write", False)),
            requery_result_hash=str(data.get("requery_result_hash", "hash:pending")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LivenessCheckSampleRecord:
    sample_id: str
    validator_ref: str
    run_id: str
    evidence_url: str
    evidence_ref: str
    snapshot_ref: str
    sample_rate: float
    status: str = SampleRecordStatus.SAMPLED_RECORD_ONLY.value
    sampled: bool = True
    live_http_performed: bool = False
    anti_bot_wall_seen: bool = False
    anti_bot_tolerant: bool = True
    fabrication_escalated: bool = False
    fabrication_evidence_ref: str = ""
    production_consensus_write: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LivenessCheckSampleRecord":
        return cls(
            sample_id=str(data["sample_id"]),
            validator_ref=str(data["validator_ref"]),
            run_id=str(data["run_id"]),
            evidence_url=str(data["evidence_url"]),
            evidence_ref=str(data["evidence_ref"]),
            snapshot_ref=str(data["snapshot_ref"]),
            sample_rate=float(data["sample_rate"]),
            status=str(data.get("status", SampleRecordStatus.SAMPLED_RECORD_ONLY.value)),
            sampled=bool(data.get("sampled", True)),
            live_http_performed=bool(data.get("live_http_performed", False)),
            anti_bot_wall_seen=bool(data.get("anti_bot_wall_seen", False)),
            anti_bot_tolerant=bool(data.get("anti_bot_tolerant", True)),
            fabrication_escalated=bool(data.get("fabrication_escalated", False)),
            fabrication_evidence_ref=str(data.get("fabrication_evidence_ref", "")),
            production_consensus_write=bool(data.get("production_consensus_write", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatorConsensusSummaryRecord:
    consensus_id: str
    epoch_ref: str
    validator_refs: tuple[str, ...]
    attestation_verification_refs: tuple[str, ...]
    open_verifier_rerun_refs: tuple[str, ...]
    judge_requery_sample_refs: tuple[str, ...]
    liveness_sample_refs: tuple[str, ...]
    passed: bool
    state: str = ConsensusState.LOCAL_SUMMARY_ONLY.value
    local_only: bool = True
    chain_weights_emitted: bool = False
    live_validator_emission_changed: bool = False
    production_consensus_enabled: bool = False
    supabase_writes: bool = False
    consensus_hash: str = "sha256:pending"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ValidatorConsensusSummaryRecord":
        return cls(
            consensus_id=str(data["consensus_id"]),
            epoch_ref=str(data["epoch_ref"]),
            validator_refs=tuple(str(item) for item in data.get("validator_refs", [])),
            attestation_verification_refs=tuple(str(item) for item in data.get("attestation_verification_refs", [])),
            open_verifier_rerun_refs=tuple(str(item) for item in data.get("open_verifier_rerun_refs", [])),
            judge_requery_sample_refs=tuple(str(item) for item in data.get("judge_requery_sample_refs", [])),
            liveness_sample_refs=tuple(str(item) for item in data.get("liveness_sample_refs", [])),
            passed=bool(data["passed"]),
            state=str(data.get("state", ConsensusState.LOCAL_SUMMARY_ONLY.value)),
            local_only=bool(data.get("local_only", True)),
            chain_weights_emitted=bool(data.get("chain_weights_emitted", False)),
            live_validator_emission_changed=bool(data.get("live_validator_emission_changed", False)),
            production_consensus_enabled=bool(data.get("production_consensus_enabled", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            consensus_hash=str(data.get("consensus_hash", "sha256:pending")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field in (
            "validator_refs",
            "attestation_verification_refs",
            "open_verifier_rerun_refs",
            "judge_requery_sample_refs",
            "liveness_sample_refs",
        ):
            data[field] = list(getattr(self, field))
        return data


def build_attestation_verification_record(
    *,
    validator_ref: str,
    anchor: AttestationAnchorStub | Mapping[str, Any],
    role: str,
    allowlist: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_attestations: Sequence[Mapping[str, Any]],
    pcr0_allowlist_ref: str,
) -> AttestationVerificationRecord:
    if not isinstance(anchor, AttestationAnchorStub):
        anchor = AttestationAnchorStub.from_mapping(anchor)
    shape = validate_attestation_response_shape(anchor.attestation_response)
    pcr0 = str(anchor.attestation_response.get("pcr0", ""))
    pcr0_allowed = is_pcr0_allowed(pcr0, allowlist, role=role)
    observed_for_run = [item for item in observed_attestations if str(item.get("run_id")) == anchor.run_id]
    output_hashes = tuple(str(item.get("output_hash", "")) for item in observed_for_run if item.get("output_hash"))
    unique_hashes = {item for item in output_hashes if item}
    equivocation_conflict = len(unique_hashes) > 1
    conflicting_refs = tuple(
        str(item.get("attestation_ref"))
        for item in observed_for_run
        if equivocation_conflict and str(item.get("attestation_ref")) != anchor.attestation_ref
    )
    status = (
        ValidatorCheckStatus.PASSED.value
        if shape["passed"] and pcr0_allowed and not equivocation_conflict
        else ValidatorCheckStatus.FAILED.value
    )
    payload = {
        "validator_ref": validator_ref,
        "run_id": anchor.run_id,
        "attestation_ref": anchor.attestation_ref,
        "role": role,
        "pcr0": pcr0,
        "output_hashes": sorted(unique_hashes),
    }
    return AttestationVerificationRecord(
        verification_id="attestation_verification:" + sha256_json(payload).split(":", 1)[1][:16],
        validator_ref=validator_ref,
        run_id=anchor.run_id,
        attestation_ref=anchor.attestation_ref,
        role=role,
        attestation_response=anchor.attestation_response,
        pcr0_allowlist_ref=pcr0_allowlist_ref,
        pcr0=pcr0,
        shape_checked=True,
        shape_passed=bool(shape["passed"]),
        pcr0_allowlist_checked=True,
        pcr0_allowed=pcr0_allowed,
        equivocation_check_performed=True,
        equivocation_conflict=equivocation_conflict,
        observed_output_hashes=output_hashes,
        conflicting_attestation_refs=conflicting_refs,
        signature_chain_checked=False,
        production_valid=False,
        status=status,
    )


def validate_attestation_verification_record(record: AttestationVerificationRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, AttestationVerificationRecord):
        record = AttestationVerificationRecord.from_mapping(record)
    errors: list[str] = []
    if record.status not in {status.value for status in ValidatorCheckStatus}:
        errors.append(f"unknown attestation verification status: {record.status}")
    if not record.shape_checked:
        errors.append("attestation shape check must not be skipped")
    shape = validate_attestation_response_shape(record.attestation_response)
    if record.shape_passed != bool(shape["passed"]):
        errors.append("shape_passed does not match attestation response shape")
    if not record.shape_passed:
        errors.append("attestation response shape failed")
    if not record.pcr0_allowlist_checked:
        errors.append("PCR0 allowlist check must not be skipped")
    if not record.pcr0_allowed:
        errors.append("PCR0 is not allowlisted")
    if not record.equivocation_check_performed:
        errors.append("equivocation check must not be skipped")
    if record.equivocation_conflict:
        errors.append("equivocation conflict detected")
    if not record.observed_output_hashes:
        errors.append("observed_output_hashes must not be empty")
    if record.signature_chain_checked:
        errors.append("P1.6 local records must not claim Nitro/COSE signature-chain verification")
    if record.production_valid:
        errors.append("P1.6 records must not mark attestation production-valid")
    expected_status = ValidatorCheckStatus.PASSED.value if not errors else ValidatorCheckStatus.FAILED.value
    if record.status != expected_status:
        errors.append(f"attestation status must be {expected_status}")
    return errors


def build_open_verifier_rerun_record(
    *,
    validator_ref: str,
    run_id: str,
    fixture_ref: str,
    fixture_payload: Mapping[str, Any],
    expected_result_payload: Mapping[str, Any],
    actual_result_payload: Mapping[str, Any],
    golden_vector_errors: Sequence[str] = (),
) -> OpenVerifierRerunRecord:
    fixture_hash = sha256_json(fixture_payload)
    expected_result_hash = sha256_json(expected_result_payload)
    actual_result_hash = sha256_json(actual_result_payload)
    byte_identical = expected_result_hash == actual_result_hash and not golden_vector_errors
    payload = {
        "validator_ref": validator_ref,
        "run_id": run_id,
        "fixture_hash": fixture_hash,
        "actual_result_hash": actual_result_hash,
    }
    return OpenVerifierRerunRecord(
        rerun_id="open_verifier_rerun:" + sha256_json(payload).split(":", 1)[1][:16],
        validator_ref=validator_ref,
        run_id=run_id,
        fixture_ref=fixture_ref,
        fixture_hash=fixture_hash,
        expected_result_hash=expected_result_hash,
        actual_result_hash=actual_result_hash,
        byte_identical=byte_identical,
        golden_vector_errors=tuple(golden_vector_errors),
    )


def validate_open_verifier_rerun_record(record: OpenVerifierRerunRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, OpenVerifierRerunRecord):
        record = OpenVerifierRerunRecord.from_mapping(record)
    errors: list[str] = []
    for field in ("fixture_hash", "expected_result_hash", "actual_result_hash", "open_verifier_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256-prefixed")
    if not record.local_only:
        errors.append("open-verifier rerun record must remain local_only")
    if not record.rerun_performed_locally:
        errors.append("open-verifier rerun must be recorded as locally performed")
    if record.live_network_used:
        errors.append("open-verifier rerun must not use live network")
    if record.production_consensus_write:
        errors.append("open-verifier rerun must not write production consensus")
    if record.golden_vector_errors:
        errors.append("open-verifier golden vectors failed")
    if not record.byte_identical:
        errors.append("open-verifier rerun must be byte-identical")
    if record.expected_result_hash != record.actual_result_hash:
        errors.append("expected and actual result hashes differ")
    return errors


def validate_judge_requery_sample_record(record: JudgeRequerySampleRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, JudgeRequerySampleRecord):
        record = JudgeRequerySampleRecord.from_mapping(record)
    errors: list[str] = []
    if record.status not in {status.value for status in SampleRecordStatus}:
        errors.append(f"unknown judge sample status: {record.status}")
    if not 0 < record.sample_rate <= 1:
        errors.append("sample_rate must be in (0, 1]")
    if not record.sampled:
        errors.append("judge re-query record must be sampled")
    if record.live_requery_performed:
        errors.append("P1.6 judge re-query records must not perform live calls")
    if record.sealed_judge_accessed:
        errors.append("P1.6 judge re-query records must not access sealed judges")
    if record.production_consensus_write:
        errors.append("judge re-query record must not write production consensus")
    if record.requery_result_hash != "hash:pending":
        errors.append("judge re-query result must remain pending in local P1.6")
    return errors


def validate_liveness_check_sample_record(record: LivenessCheckSampleRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, LivenessCheckSampleRecord):
        record = LivenessCheckSampleRecord.from_mapping(record)
    errors: list[str] = []
    if record.status not in {status.value for status in SampleRecordStatus}:
        errors.append(f"unknown liveness sample status: {record.status}")
    if not 0 < record.sample_rate <= 1:
        errors.append("sample_rate must be in (0, 1]")
    if not record.sampled:
        errors.append("liveness record must be sampled")
    if record.live_http_performed:
        errors.append("P1.6 liveness records must not perform live HTTP")
    if record.production_consensus_write:
        errors.append("liveness record must not write production consensus")
    if record.anti_bot_wall_seen and not record.anti_bot_tolerant:
        errors.append("anti-bot wall must be treated as tolerant, not fabrication")
    if record.anti_bot_wall_seen and record.fabrication_escalated and not record.fabrication_evidence_ref:
        errors.append("fabrication escalation requires fabrication_evidence_ref")
    if record.anti_bot_wall_seen and record.fabrication_escalated:
        errors.append("anti-bot wall alone must not escalate as fabrication")
    return errors


def validate_validator_consensus_summary_record(
    record: ValidatorConsensusSummaryRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    if not isinstance(record, ValidatorConsensusSummaryRecord):
        record = ValidatorConsensusSummaryRecord.from_mapping(record)
    errors: list[str] = []
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ConsensusState}:
        errors.append(f"unknown consensus state: {record.state}")
    if not record.validator_refs:
        errors.append("validator_refs must not be empty")
    if not record.attestation_verification_refs:
        errors.append("attestation_verification_refs must not be empty")
    if not record.open_verifier_rerun_refs:
        errors.append("open_verifier_rerun_refs must not be empty")
    if not record.local_only:
        errors.append("consensus summary must remain local_only")
    if record.chain_weights_emitted:
        errors.append("P1.6 must not emit chain weights")
    if record.live_validator_emission_changed:
        errors.append("P1.6 must not change live validator emissions")
    if record.production_consensus_enabled:
        errors.append("P1.6 must not enable production consensus")
    if record.supabase_writes:
        errors.append("P1.6 must not write Supabase")
    if not record.consensus_hash.startswith("sha256:"):
        errors.append("consensus_hash must be sha256-prefixed")
    return errors


def build_validator_consensus_summary_record(
    *,
    epoch_ref: str,
    validator_refs: Sequence[str],
    attestation_records: Sequence[AttestationVerificationRecord],
    rerun_records: Sequence[OpenVerifierRerunRecord],
    judge_samples: Sequence[JudgeRequerySampleRecord],
    liveness_samples: Sequence[LivenessCheckSampleRecord],
) -> ValidatorConsensusSummaryRecord:
    passed = all(record.status == ValidatorCheckStatus.PASSED.value for record in attestation_records) and all(
        record.byte_identical for record in rerun_records
    )
    payload = {
        "epoch_ref": epoch_ref,
        "validator_refs": list(validator_refs),
        "attestation_verification_refs": [record.verification_id for record in attestation_records],
        "open_verifier_rerun_refs": [record.rerun_id for record in rerun_records],
        "judge_requery_sample_refs": [record.sample_id for record in judge_samples],
        "liveness_sample_refs": [record.sample_id for record in liveness_samples],
        "passed": passed,
    }
    return ValidatorConsensusSummaryRecord(
        consensus_id="validator_consensus:" + sha256_json(payload).split(":", 1)[1][:16],
        epoch_ref=epoch_ref,
        validator_refs=tuple(validator_refs),
        attestation_verification_refs=tuple(record.verification_id for record in attestation_records),
        open_verifier_rerun_refs=tuple(record.rerun_id for record in rerun_records),
        judge_requery_sample_refs=tuple(record.sample_id for record in judge_samples),
        liveness_sample_refs=tuple(record.sample_id for record in liveness_samples),
        passed=passed,
        consensus_hash=sha256_json(payload),
    )


def verify_research_lab_validator_flow(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fabric_summary = verify_research_lab_fabric()
    fixture = _load_fixture(Path(fixture_path))
    allowlist = load_pcr0_allowlist(str(PCR0_ALLOWLIST_PATH))

    anchor = AttestationAnchorStub.from_mapping(fixture["attestation_anchor"])
    _assert(not validate_attestation_anchor_stub(anchor), "P1.2 anchor stub validates as a local-only record")
    attestation = build_attestation_verification_record(
        validator_ref=fixture["validator_ref"],
        anchor=anchor,
        role=fixture["role"],
        allowlist=allowlist,
        observed_attestations=fixture["observed_attestations"],
        pcr0_allowlist_ref=fixture["pcr0_allowlist_ref"],
    )
    _assert(not validate_attestation_verification_record(attestation), "attestation verification validates")
    _assert(attestation.pcr0_allowlist_checked, "PCR0 allowlist check is recorded")
    _assert(attestation.equivocation_check_performed, "equivocation check is recorded")

    for record in fixture["invalid_attestation_verifications"]:
        errors = validate_attestation_verification_record(record)
        _assert(errors, f"invalid attestation verification fails: {record['verification_id']}")
        _assert_expected_error(errors, record)

    conflict_anchor = AttestationAnchorStub.from_mapping(fixture["equivocation_anchor"])
    conflict = build_attestation_verification_record(
        validator_ref=fixture["validator_ref"],
        anchor=conflict_anchor,
        role=fixture["role"],
        allowlist=allowlist,
        observed_attestations=fixture["equivocation_observed_attestations"],
        pcr0_allowlist_ref=fixture["pcr0_allowlist_ref"],
    )
    conflict_errors = validate_attestation_verification_record(conflict)
    _assert(any("equivocation conflict" in error for error in conflict_errors), "equivocation conflict fails")

    golden_errors = run_golden_vectors(pcr0_allowlist_path=str(PCR0_ALLOWLIST_PATH))
    rerun = build_open_verifier_rerun_record(
        validator_ref=fixture["validator_ref"],
        run_id=fixture["run_id"],
        fixture_ref=fixture["open_verifier_rerun"]["fixture_ref"],
        fixture_payload=fixture["open_verifier_rerun"]["fixture_payload"],
        expected_result_payload=fixture["open_verifier_rerun"]["expected_result_payload"],
        actual_result_payload=fixture["open_verifier_rerun"]["actual_result_payload"],
        golden_vector_errors=golden_errors,
    )
    _assert(not validate_open_verifier_rerun_record(rerun), "open-verifier rerun validates")
    for record in fixture["invalid_open_verifier_reruns"]:
        errors = validate_open_verifier_rerun_record(record)
        _assert(errors, f"invalid open-verifier rerun fails: {record['rerun_id']}")
        _assert_expected_error(errors, record)

    judge_sample = JudgeRequerySampleRecord.from_mapping(fixture["judge_requery_sample"])
    _assert(not validate_judge_requery_sample_record(judge_sample), "judge sample validates")
    for record in fixture["invalid_judge_requery_samples"]:
        errors = validate_judge_requery_sample_record(record)
        _assert(errors, f"invalid judge sample fails: {record['sample_id']}")
        _assert_expected_error(errors, record)

    liveness_sample = LivenessCheckSampleRecord.from_mapping(fixture["liveness_sample"])
    _assert(not validate_liveness_check_sample_record(liveness_sample), "liveness sample validates")
    for record in fixture["invalid_liveness_samples"]:
        errors = validate_liveness_check_sample_record(record)
        _assert(errors, f"invalid liveness sample fails: {record['sample_id']}")
        _assert_expected_error(errors, record)

    consensus = build_validator_consensus_summary_record(
        epoch_ref=fixture["epoch_ref"],
        validator_refs=fixture["validator_refs"],
        attestation_records=[attestation],
        rerun_records=[rerun],
        judge_samples=[judge_sample],
        liveness_samples=[liveness_sample],
    )
    _assert(not validate_validator_consensus_summary_record(consensus), "consensus summary validates")
    for record in fixture["invalid_consensus_summaries"]:
        errors = validate_validator_consensus_summary_record(record)
        _assert(errors, f"invalid consensus summary fails: {record['consensus_id']}")
        _assert_expected_error(errors, record)
    unsafe_consensus_errors = validate_validator_consensus_summary_record(
        consensus,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_consensus_errors, "unsafe workflow guards block consensus summary")

    return {
        "fabric_invalid_records": fabric_summary["invalid_records"],
        "attestation_status": attestation.status,
        "pcr0_allowed": attestation.pcr0_allowed,
        "open_verifier_byte_identical": rerun.byte_identical,
        "judge_samples": 1,
        "liveness_samples": 1,
        "consensus_id": consensus.consensus_id,
    }


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
