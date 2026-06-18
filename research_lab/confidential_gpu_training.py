"""Phase 4 confidential-GPU training contract records.

P4.4 defines local record shapes for future confidential-GPU training evidence,
data policy approval, and private model-weight artifact handling. It does not
start GPU jobs, call KMS, verify Nitro/COSE chains, release keys, write model
registries, publish weights, or claim production validity from fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .fine_tune_dataset import (
    PROTECTED_FINE_TUNE_KEYS,
    PROTECTED_FINE_TUNE_MARKERS,
    verify_research_lab_fine_tune_dataset,
)
from .full_sealing_kms import (
    PROTECTED_SEALING_KEYS,
    PROTECTED_SEALING_MARKERS,
    verify_research_lab_full_sealing_kms,
)
from .scale_foundation import (
    PROTECTED_SCALE_KEYS,
    PROTECTED_SCALE_MARKERS,
    ScaleGate,
    ScaleWorkflowGuards,
    assert_scale_workflows_disabled,
    default_scale_workflow_guards,
    require_scale_gate,
)
from .phase_scaffold_cleanup import verify_phase_scaffold_cleanup


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "confidential_gpu_training_fixtures.json"

CONFIDENTIAL_GPU_TRAINING_CONTRACT_VERSION = "confidential_gpu_training:v1:local_contract"
CONFIDENTIAL_GPU_DATA_POLICY_VERSION = "confidential_gpu_data_policy:v1:local_contract"
CONFIDENTIAL_GPU_WEIGHT_ARTIFACT_VERSION = "confidential_gpu_weight_artifact:v1:local_contract"

PROTECTED_CONFIDENTIAL_GPU_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SCALE_KEYS)
    | set(PROTECTED_SEALING_KEYS)
    | set(PROTECTED_FINE_TUNE_KEYS)
    | {
        "checkpoint_bytes",
        "confidential_gpu_session_token",
        "customer_training_payload",
        "gpu_driver_secret",
        "live_model_weights",
        "model_registry_token",
        "sealed_dataset_row",
        "training_secret",
        "weight_bytes",
    }
)

PROTECTED_CONFIDENTIAL_GPU_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SCALE_MARKERS)
        | set(PROTECTED_SEALING_MARKERS)
        | set(PROTECTED_FINE_TUNE_MARKERS)
        | {
            "checkpoint bytes",
            "confidential gpu session",
            "customer training payload",
            "gpu driver secret",
            "live model weights",
            "model registry token",
            "sealed dataset row",
            "training secret",
            "weight bytes",
        }
    )
)

CONFIDENTIAL_GPU_POLICY_DECLARATION_FIELDS: frozenset[str] = frozenset(
    {
        "allowed_data_classes",
        "prohibited_material_classes",
    }
)


class ConfidentialGPUDataPolicyState(str, Enum):
    LOCAL_POLICY_STUB = "local_policy_stub"
    READY_AFTER_OWNER_APPROVAL = "ready_after_owner_approval"
    BLOCKED = "blocked"


class ConfidentialGPUTrainingState(str, Enum):
    LOCAL_TRAINING_STUB = "local_training_stub"
    READY_AFTER_CONFIDENTIAL_GPU_EVIDENCE = "ready_after_confidential_gpu_evidence"
    BLOCKED = "blocked"


class ConfidentialGPUWeightArtifactState(str, Enum):
    PRIVATE_HASH_ONLY_STUB = "private_hash_only_stub"
    PRIVATE_ARTIFACT_READY = "private_artifact_ready"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ConfidentialGPUDataPolicyRecord:
    policy_id: str
    source_dataset_ref: str
    rights_review_ref: str
    pii_review_ref: str
    retention_policy_ref: str
    sealed_eval_policy_ref: str
    allowed_data_classes: tuple[str, ...]
    prohibited_material_classes: tuple[str, ...]
    private_training_data_allowed: bool = False
    raw_customer_data_excluded: bool = False
    sealed_eval_excluded: bool = False
    live_champion_ip_excluded: bool = False
    policy_approved: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = ConfidentialGPUDataPolicyState.LOCAL_POLICY_STUB.value
    contract_version: str = CONFIDENTIAL_GPU_DATA_POLICY_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConfidentialGPUDataPolicyRecord":
        return cls(
            policy_id=str(data["policy_id"]),
            source_dataset_ref=str(data["source_dataset_ref"]),
            rights_review_ref=str(data["rights_review_ref"]),
            pii_review_ref=str(data["pii_review_ref"]),
            retention_policy_ref=str(data["retention_policy_ref"]),
            sealed_eval_policy_ref=str(data["sealed_eval_policy_ref"]),
            allowed_data_classes=tuple(str(item) for item in data.get("allowed_data_classes", [])),
            prohibited_material_classes=tuple(str(item) for item in data.get("prohibited_material_classes", [])),
            private_training_data_allowed=bool(data.get("private_training_data_allowed", False)),
            raw_customer_data_excluded=bool(data.get("raw_customer_data_excluded", False)),
            sealed_eval_excluded=bool(data.get("sealed_eval_excluded", False)),
            live_champion_ip_excluded=bool(data.get("live_champion_ip_excluded", False)),
            policy_approved=bool(data.get("policy_approved", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", ConfidentialGPUDataPolicyState.LOCAL_POLICY_STUB.value)),
            contract_version=str(data.get("contract_version", CONFIDENTIAL_GPU_DATA_POLICY_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_data_classes"] = list(self.allowed_data_classes)
        data["prohibited_material_classes"] = list(self.prohibited_material_classes)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class ConfidentialGPUTrainingRunRecord:
    training_run_id: str
    source_training_ref: str
    source_dataset_ref: str
    data_policy_ref: str
    measurement_gate_ref: str
    research_enclave_run_ref: str
    kms_release_decision_ref: str
    gpu_attestation_ref: str
    enclave_measurement_ref: str
    egress_policy_ref: str
    input_dataset_hash: str
    output_model_hash: str
    cost_ledger_hash: str
    data_policy_approved: bool = False
    live_confidential_gpu_stack_verified: bool = False
    attestation_signature_chain_verified: bool = False
    pcr0_allowlist_passed: bool = False
    kms_policy_enforced: bool = False
    kms_key_release_performed: bool = False
    egress_policy_enforced: bool = False
    equivocation_check_passed: bool = False
    training_started: bool = False
    training_completed: bool = False
    production_training_valid: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = ConfidentialGPUTrainingState.LOCAL_TRAINING_STUB.value
    contract_version: str = CONFIDENTIAL_GPU_TRAINING_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConfidentialGPUTrainingRunRecord":
        return cls(
            training_run_id=str(data["training_run_id"]),
            source_training_ref=str(data["source_training_ref"]),
            source_dataset_ref=str(data["source_dataset_ref"]),
            data_policy_ref=str(data["data_policy_ref"]),
            measurement_gate_ref=str(data["measurement_gate_ref"]),
            research_enclave_run_ref=str(data["research_enclave_run_ref"]),
            kms_release_decision_ref=str(data["kms_release_decision_ref"]),
            gpu_attestation_ref=str(data["gpu_attestation_ref"]),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            egress_policy_ref=str(data["egress_policy_ref"]),
            input_dataset_hash=str(data["input_dataset_hash"]),
            output_model_hash=str(data["output_model_hash"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            data_policy_approved=bool(data.get("data_policy_approved", False)),
            live_confidential_gpu_stack_verified=bool(data.get("live_confidential_gpu_stack_verified", False)),
            attestation_signature_chain_verified=bool(data.get("attestation_signature_chain_verified", False)),
            pcr0_allowlist_passed=bool(data.get("pcr0_allowlist_passed", False)),
            kms_policy_enforced=bool(data.get("kms_policy_enforced", False)),
            kms_key_release_performed=bool(data.get("kms_key_release_performed", False)),
            egress_policy_enforced=bool(data.get("egress_policy_enforced", False)),
            equivocation_check_passed=bool(data.get("equivocation_check_passed", False)),
            training_started=bool(data.get("training_started", False)),
            training_completed=bool(data.get("training_completed", False)),
            production_training_valid=bool(data.get("production_training_valid", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", ConfidentialGPUTrainingState.LOCAL_TRAINING_STUB.value)),
            contract_version=str(data.get("contract_version", CONFIDENTIAL_GPU_TRAINING_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


@dataclass(frozen=True)
class ConfidentialGPUWeightArtifactRecord:
    weight_artifact_id: str
    training_run_ref: str
    model_weight_hash: str
    private_artifact_ref: str
    release_policy_ref: str
    weight_card_ref: str
    hash_recorded: bool = False
    protected_material_scan_passed: bool = False
    private_artifact_ready: bool = False
    public_release_enabled: bool = False
    public_download_enabled: bool = False
    model_registry_write: bool = False
    production_promotion_requested: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = ConfidentialGPUWeightArtifactState.PRIVATE_HASH_ONLY_STUB.value
    contract_version: str = CONFIDENTIAL_GPU_WEIGHT_ARTIFACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConfidentialGPUWeightArtifactRecord":
        return cls(
            weight_artifact_id=str(data["weight_artifact_id"]),
            training_run_ref=str(data["training_run_ref"]),
            model_weight_hash=str(data["model_weight_hash"]),
            private_artifact_ref=str(data["private_artifact_ref"]),
            release_policy_ref=str(data["release_policy_ref"]),
            weight_card_ref=str(data["weight_card_ref"]),
            hash_recorded=bool(data.get("hash_recorded", False)),
            protected_material_scan_passed=bool(data.get("protected_material_scan_passed", False)),
            private_artifact_ready=bool(data.get("private_artifact_ready", False)),
            public_release_enabled=bool(data.get("public_release_enabled", False)),
            public_download_enabled=bool(data.get("public_download_enabled", False)),
            model_registry_write=bool(data.get("model_registry_write", False)),
            production_promotion_requested=bool(data.get("production_promotion_requested", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", ConfidentialGPUWeightArtifactState.PRIVATE_HASH_ONLY_STUB.value)),
            contract_version=str(data.get("contract_version", CONFIDENTIAL_GPU_WEIGHT_ARTIFACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def validate_confidential_gpu_data_policy_record(record: ConfidentialGPUDataPolicyRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_confidential_gpu_payload_errors(raw)
    if not isinstance(record, ConfidentialGPUDataPolicyRecord):
        try:
            record = ConfidentialGPUDataPolicyRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required confidential GPU data policy field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid confidential GPU data policy payload: {exc}")
            return errors
    if record.state not in {state.value for state in ConfidentialGPUDataPolicyState}:
        errors.append(f"unknown confidential GPU data policy state: {record.state}")
        return errors
    if record.contract_version != CONFIDENTIAL_GPU_DATA_POLICY_VERSION:
        errors.append("contract_version must match confidential GPU data policy contract")
    if not record.policy_id.startswith("data_policy:"):
        errors.append("policy_id must be data_policy:-prefixed")
    if not record.source_dataset_ref.startswith(("fine_tune_dataset:", "reranker_distillation_dataset:", "trajectory_corpus:")):
        errors.append("source_dataset_ref must reference an approved training dataset")
    for field, prefix in (
        ("rights_review_ref", "rights_review:"),
        ("pii_review_ref", "pii_review:"),
        ("retention_policy_ref", "retention_policy:"),
        ("sealed_eval_policy_ref", "sealed_eval_policy:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    if not record.allowed_data_classes:
        errors.append("confidential GPU data policy requires allowed_data_classes")
    if not record.prohibited_material_classes:
        errors.append("confidential GPU data policy requires prohibited_material_classes")
    _append_confidential_gpu_write_guard_errors(record, "P4.4 confidential GPU data policy", errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "data_policy:",
                "rights_review:",
                "pii_review:",
                "retention_policy:",
                "sealed_eval_policy:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved confidential GPU data policy prefixes")
            break
    if record.policy_approved:
        if not record.private_training_data_allowed:
            errors.append("approved confidential GPU data policy must explicitly allow private training data")
        if not record.raw_customer_data_excluded:
            errors.append("approved confidential GPU data policy must exclude raw customer data")
        if not record.sealed_eval_excluded:
            errors.append("approved confidential GPU data policy must exclude sealed eval details")
        if not record.live_champion_ip_excluded:
            errors.append("approved confidential GPU data policy must exclude live champion IP")
        if record.uses_local_fixtures:
            errors.append("confidential GPU data policy approval cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("confidential GPU data policy approval cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("confidential GPU data policy approval requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("confidential GPU data policy approval requires owner_approval_ref")
        if record.state != ConfidentialGPUDataPolicyState.READY_AFTER_OWNER_APPROVAL.value:
            errors.append("policy_approved requires ready_after_owner_approval state")
    else:
        if not record.local_only:
            errors.append("not-approved confidential GPU data policies must remain local_only")
        if record.state == ConfidentialGPUDataPolicyState.READY_AFTER_OWNER_APPROVAL.value:
            errors.append("ready_after_owner_approval state requires policy_approved")
    return errors


def validate_confidential_gpu_training_run_record(
    record: ConfidentialGPUTrainingRunRecord | Mapping[str, Any],
    *,
    data_policy: Optional[ConfidentialGPUDataPolicyRecord | Mapping[str, Any]] = None,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_confidential_gpu_payload_errors(raw)
    if not isinstance(record, ConfidentialGPUTrainingRunRecord):
        try:
            record = ConfidentialGPUTrainingRunRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required confidential GPU training field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid confidential GPU training payload: {exc}")
            return errors
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ConfidentialGPUTrainingState}:
        errors.append(f"unknown confidential GPU training state: {record.state}")
        return errors
    if record.contract_version != CONFIDENTIAL_GPU_TRAINING_CONTRACT_VERSION:
        errors.append("contract_version must match confidential GPU training contract")
    if not record.training_run_id.startswith("confidential_gpu_training:"):
        errors.append("training_run_id must be confidential_gpu_training:-prefixed")
    if not record.source_training_ref.startswith(("engine_finetune_run:", "reranker_distillation_run:")):
        errors.append("source_training_ref must reference an approved Phase 3 training run")
    if not record.source_dataset_ref.startswith(("fine_tune_dataset:", "reranker_distillation_dataset:", "trajectory_corpus:")):
        errors.append("source_dataset_ref must reference an approved training dataset")
    for field, prefix in (
        ("data_policy_ref", "data_policy:"),
        ("measurement_gate_ref", "measurement_gate:"),
        ("research_enclave_run_ref", "research_enclave_run:"),
        ("kms_release_decision_ref", "kms_release_decision:"),
        ("enclave_measurement_ref", "enclave_measurement:"),
        ("egress_policy_ref", "egress_policy:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    if not record.gpu_attestation_ref.startswith(("gpu_attestation:", "attestation:")):
        errors.append("gpu_attestation_ref must be gpu_attestation: or attestation:-prefixed")
    for field in ("input_dataset_hash", "output_model_hash", "cost_ledger_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    _append_confidential_gpu_write_guard_errors(record, "P4.4 confidential GPU training", errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "confidential_gpu:",
                "gpu_attestation:",
                "attestation:",
                "measurement_gate:",
                "research_enclave_run:",
                "kms_release_decision:",
                "kms_policy:",
                "data_policy:",
                "fine_tune_dataset:",
                "reranker_distillation_dataset:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved confidential GPU training prefixes")
            break
    if data_policy is not None:
        if not isinstance(data_policy, ConfidentialGPUDataPolicyRecord):
            data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(data_policy)
        policy_errors = validate_confidential_gpu_data_policy_record(data_policy)
        if policy_errors:
            errors.append("source confidential GPU data policy is invalid: " + "; ".join(policy_errors))
        if record.data_policy_ref != data_policy.policy_id:
            errors.append("confidential GPU training data_policy_ref mismatch")
        if record.data_policy_approved != data_policy.policy_approved:
            errors.append("confidential GPU training data_policy_approved flag mismatch")
    if record.production_training_valid:
        if data_policy is None:
            errors.append("production confidential GPU training requires supplied data_policy")
        _append_missing_true_flags(
            record,
            errors,
            (
                ("data_policy_approved", "production confidential GPU training requires approved data policy"),
                ("live_confidential_gpu_stack_verified", "production confidential GPU training requires live stack verification"),
                ("attestation_signature_chain_verified", "production confidential GPU training requires signature-chain verification"),
                ("pcr0_allowlist_passed", "production confidential GPU training requires PCR0 allowlist pass"),
                ("kms_policy_enforced", "production confidential GPU training requires enforced KMS policy"),
                ("kms_key_release_performed", "production confidential GPU training requires performed KMS key release"),
                ("egress_policy_enforced", "production confidential GPU training requires enforced egress policy"),
                ("equivocation_check_passed", "production confidential GPU training requires equivocation check pass"),
                ("training_started", "production confidential GPU training requires training_started"),
                ("training_completed", "production confidential GPU training requires training_completed"),
            ),
        )
        if record.uses_local_fixtures:
            errors.append("confidential GPU training validity cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("confidential GPU training validity cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("confidential GPU training validity requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("confidential GPU training validity requires owner_approval_ref")
        if record.state != ConfidentialGPUTrainingState.READY_AFTER_CONFIDENTIAL_GPU_EVIDENCE.value:
            errors.append("production_training_valid requires ready_after_confidential_gpu_evidence state")
    else:
        if not record.local_only:
            errors.append("not-valid confidential GPU training records must remain local_only")
        if record.state == ConfidentialGPUTrainingState.READY_AFTER_CONFIDENTIAL_GPU_EVIDENCE.value:
            errors.append("ready_after_confidential_gpu_evidence state requires production_training_valid")
        if record.training_started:
            errors.append("local P4.4 records must not start confidential GPU training")
        if record.training_completed:
            errors.append("local P4.4 records must not complete confidential GPU training")
        if record.kms_key_release_performed:
            errors.append("local P4.4 records must not perform KMS key release")
        if record.live_confidential_gpu_stack_verified:
            errors.append("local P4.4 records must not claim live confidential GPU verification")
    return errors


def validate_confidential_gpu_weight_artifact_record(
    record: ConfidentialGPUWeightArtifactRecord | Mapping[str, Any],
    *,
    training: Optional[ConfidentialGPUTrainingRunRecord | Mapping[str, Any]] = None,
    data_policy: Optional[ConfidentialGPUDataPolicyRecord | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_confidential_gpu_payload_errors(raw)
    if not isinstance(record, ConfidentialGPUWeightArtifactRecord):
        try:
            record = ConfidentialGPUWeightArtifactRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required confidential GPU weight artifact field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid confidential GPU weight artifact payload: {exc}")
            return errors
    if record.state not in {state.value for state in ConfidentialGPUWeightArtifactState}:
        errors.append(f"unknown confidential GPU weight artifact state: {record.state}")
        return errors
    if record.contract_version != CONFIDENTIAL_GPU_WEIGHT_ARTIFACT_VERSION:
        errors.append("contract_version must match confidential GPU weight artifact contract")
    if not record.weight_artifact_id.startswith("model_weight_artifact:"):
        errors.append("weight_artifact_id must be model_weight_artifact:-prefixed")
    if not record.training_run_ref.startswith("confidential_gpu_training:"):
        errors.append("training_run_ref must be confidential_gpu_training:-prefixed")
    if not record.model_weight_hash.startswith("sha256:"):
        errors.append("model_weight_hash must be sha256:-prefixed")
    if not record.private_artifact_ref.startswith("model_weight_private:"):
        errors.append("private_artifact_ref must be model_weight_private:-prefixed")
    if not record.release_policy_ref.startswith("release_policy:"):
        errors.append("release_policy_ref must be release_policy:-prefixed")
    if not record.weight_card_ref.startswith("weight_card:"):
        errors.append("weight_card_ref must be weight_card:-prefixed")
    _append_confidential_gpu_write_guard_errors(record, "P4.4 confidential GPU weight artifact", errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "model_weight_artifact:",
                "model_weight_private:",
                "model_weight_release:",
                "weight_card:",
                "confidential_gpu_training:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved confidential GPU weight artifact prefixes")
            break
    if record.public_release_enabled:
        errors.append("P4.4 must not enable public model-weight release")
    if record.public_download_enabled:
        errors.append("P4.4 must not enable public model-weight downloads")
    if record.model_registry_write:
        errors.append("P4.4 must not write model registries")
    if record.production_promotion_requested:
        errors.append("P4.4 must not request production promotion")
    if training is not None:
        if not isinstance(training, ConfidentialGPUTrainingRunRecord):
            training = ConfidentialGPUTrainingRunRecord.from_mapping(training)
        training_errors = validate_confidential_gpu_training_run_record(training, data_policy=data_policy)
        if training_errors:
            errors.append("source confidential GPU training run is invalid: " + "; ".join(training_errors))
        if record.training_run_ref != training.training_run_id:
            errors.append("confidential GPU weight artifact training_run_ref mismatch")
        if record.private_artifact_ready and not training.production_training_valid:
            errors.append("private weight artifact readiness requires production-valid confidential GPU training")
    if record.private_artifact_ready:
        if training is None:
            errors.append("private weight artifact readiness requires supplied confidential GPU training")
        if not record.hash_recorded:
            errors.append("private weight artifact readiness requires hash_recorded")
        if not record.protected_material_scan_passed:
            errors.append("private weight artifact readiness requires protected_material_scan_passed")
        if record.uses_local_fixtures:
            errors.append("private weight artifact readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("private weight artifact readiness cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("private weight artifact readiness requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("private weight artifact readiness requires owner_approval_ref")
        if record.state != ConfidentialGPUWeightArtifactState.PRIVATE_ARTIFACT_READY.value:
            errors.append("private_artifact_ready requires private_artifact_ready state")
    else:
        if not record.local_only:
            errors.append("not-ready confidential GPU weight artifacts must remain local_only")
        if record.state == ConfidentialGPUWeightArtifactState.PRIVATE_ARTIFACT_READY.value:
            errors.append("private_artifact_ready state requires private_artifact_ready")
    return errors


def verify_research_lab_confidential_gpu_training(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    scaffold_summary = verify_phase_scaffold_cleanup()
    sealing_summary = verify_research_lab_full_sealing_kms()
    fine_tune_summary = verify_research_lab_fine_tune_dataset()
    fixture = _load_fixture(Path(fixture_path))

    local_policy = ConfidentialGPUDataPolicyRecord.from_mapping(fixture["local_data_policy"])
    _assert(not validate_confidential_gpu_data_policy_record(local_policy), "local confidential GPU data policy validates")
    _assert(not local_policy.policy_approved, "local data policy does not claim approval")

    ready_policy = ConfidentialGPUDataPolicyRecord.from_mapping(fixture["ready_data_policy"])
    _assert(not validate_confidential_gpu_data_policy_record(ready_policy), "ready-control data policy validates")

    local_training = ConfidentialGPUTrainingRunRecord.from_mapping(fixture["local_training_run"])
    _assert(
        not validate_confidential_gpu_training_run_record(local_training, data_policy=local_policy),
        "local confidential GPU training run validates",
    )
    _assert(not local_training.production_training_valid, "local training does not claim production validity")

    ready_training = ConfidentialGPUTrainingRunRecord.from_mapping(fixture["ready_training_control"])
    _assert(
        not validate_confidential_gpu_training_run_record(ready_training, data_policy=ready_policy),
        "ready-control confidential GPU training validates",
    )
    _assert(ready_training.production_training_valid, "ready-control training claims production validity")

    local_weight = ConfidentialGPUWeightArtifactRecord.from_mapping(fixture["local_weight_artifact"])
    _assert(
        not validate_confidential_gpu_weight_artifact_record(
            local_weight,
            training=local_training,
            data_policy=local_policy,
        ),
        "local confidential GPU weight artifact validates",
    )
    _assert(not local_weight.private_artifact_ready, "local weight artifact does not claim readiness")

    ready_weight = ConfidentialGPUWeightArtifactRecord.from_mapping(fixture["ready_weight_artifact_control"])
    _assert(
        not validate_confidential_gpu_weight_artifact_record(
            ready_weight,
            training=ready_training,
            data_policy=ready_policy,
        ),
        "ready-control confidential GPU weight artifact validates",
    )
    _assert(ready_weight.private_artifact_ready, "ready-control private weight artifact is ready")
    _assert(not ready_weight.public_release_enabled, "P4.4 ready-control does not enable public weight release")

    for invalid in fixture["invalid_data_policies"]:
        base = fixture[str(invalid.get("base", "local_data_policy"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_confidential_gpu_data_policy_record(merged)
        _assert(errors, f"invalid confidential GPU data policy fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_training_runs"]:
        base = fixture[str(invalid.get("base", "local_training_run"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        if invalid.get("omit_data_policy"):
            errors = validate_confidential_gpu_training_run_record(merged)
        else:
            policy_base = fixture[str(invalid.get("data_policy_base", "local_data_policy"))]
            policy = ConfidentialGPUDataPolicyRecord.from_mapping(policy_base)
            errors = validate_confidential_gpu_training_run_record(merged, data_policy=policy)
        _assert(errors, f"invalid confidential GPU training run fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_training_errors = validate_confidential_gpu_training_run_record(
        local_training,
        data_policy=local_policy,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_training_errors, "unsafe Phase 4 guards block confidential GPU training validation")
    _assert_expected_error(unsafe_training_errors, fixture["unsafe_scale_guards"])

    for invalid in fixture["invalid_weight_artifacts"]:
        base = fixture[str(invalid.get("base", "local_weight_artifact"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        if invalid.get("omit_training"):
            errors = validate_confidential_gpu_weight_artifact_record(merged)
        else:
            training_base_name = str(invalid.get("training_base", "local_training_run"))
            training_base = fixture[training_base_name]
            training = ConfidentialGPUTrainingRunRecord.from_mapping(training_base)
            data_policy_base_name = str(
                invalid.get(
                    "data_policy_base",
                    "ready_data_policy" if training_base_name == "ready_training_control" else "local_data_policy",
                )
            )
            data_policy = ConfidentialGPUDataPolicyRecord.from_mapping(fixture[data_policy_base_name])
            errors = validate_confidential_gpu_weight_artifact_record(
                merged,
                training=training,
                data_policy=data_policy,
            )
        _assert(errors, f"invalid confidential GPU weight artifact fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"confidential_gpu_evidence_ready": False}, ScaleGate.CONFIDENTIAL_GPU_EVIDENCE_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.4 confidential GPU evidence gate remains required by P4.0")

    return {
        "phase_scaffold_audit_id": scaffold_summary["current_audit_id"],
        "sealing_gate_id": sealing_summary["gate_id"],
        "sealing_kms_decision": sealing_summary["kms_decision"],
        "fine_tune_dataset_id": fine_tune_summary["dataset_id"],
        "data_policy_id": local_policy.policy_id,
        "data_policy_approved": local_policy.policy_approved,
        "training_run_id": local_training.training_run_id,
        "production_training_valid": local_training.production_training_valid,
        "weight_artifact_id": local_weight.weight_artifact_id,
        "private_artifact_ready": local_weight.private_artifact_ready,
        "public_release_enabled": local_weight.public_release_enabled,
        "ready_control_validates": ready_training.production_training_valid and ready_weight.private_artifact_ready,
    }


def _append_missing_true_flags(record: Any, errors: list[str], checks: Sequence[tuple[str, str]]) -> None:
    for field, message in checks:
        if not bool(getattr(record, field)):
            errors.append(message)


def _append_confidential_gpu_write_guard_errors(record: Any, label: str, errors: list[str]) -> None:
    if bool(getattr(record, "production_writes", False)):
        errors.append(f"{label} must not enable production writes")
    if bool(getattr(record, "supabase_writes", False)):
        errors.append(f"{label} must not enable Supabase writes")
    if bool(getattr(record, "public_workflows", False)):
        errors.append(f"{label} must not enable public workflows")


def _protected_confidential_gpu_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_confidential_gpu_material(record))
    if not found:
        return []
    return ["P4.4 confidential GPU payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_confidential_gpu_material(
    value: Any,
    path: str = "",
    *,
    skip_value_markers: bool = False,
) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_CONFIDENTIAL_GPU_KEYS and not key_text.endswith(
                ("_ref", "_refs", "_hash", "_hashes")
            ):
                found.add(key_path)
            found.update(
                _find_protected_confidential_gpu_material(
                    nested,
                    key_path,
                    skip_value_markers=skip_value_markers
                    or key_text in CONFIDENTIAL_GPU_POLICY_DECLARATION_FIELDS,
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(
                _find_protected_confidential_gpu_material(
                    nested,
                    f"{path}[{index}]",
                    skip_value_markers=skip_value_markers,
                )
            )
    elif isinstance(value, str) and not skip_value_markers:
        lowered = value.lower()
        for marker in PROTECTED_CONFIDENTIAL_GPU_MARKERS:
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
