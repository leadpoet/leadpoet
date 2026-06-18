"""Phase 3.1 trajectory corpus readiness contracts.

This module builds local corpus manifests from sanitized trajectory references.
It does not train models, read production tables, or publish datasets. The
validators make the Phase 3 split explicit: corpus builders can be coded now,
while training/distillation/calibration readiness requires measured data,
rights, split hygiene, and protected-data exclusion.
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
    ModelPipelineBuildStatus,
    verify_model_pipeline_foundation,
)
from .schema_validation import validate_schema_record


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "trajectory_corpus_fixtures.json"
SCHEMA_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "schemas" / "fixtures"

TRAJECTORY_CORPUS_CONTRACT_VERSION = "trajectory_corpus:v1:local_contract"
MIN_TRAINING_TRAJECTORIES = 25
MIN_HOLDOUT_TRAJECTORIES = 5

PROTECTED_CORPUS_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_MODEL_PIPELINE_KEYS)
    | {
        "api_secret",
        "brief_raw",
        "customer_name",
        "email",
        "full_prompt",
        "judge_rubric_raw",
        "llm_request",
        "llm_response",
        "normalized_text",
        "page_content",
        "phone",
        "prompt",
        "raw_brief",
        "raw_content",
        "raw_customer_payload",
        "request",
        "response",
        "secret_key",
    }
)

PROTECTED_CORPUS_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_MODEL_PIPELINE_MARKERS)
        | {
            "api secret",
            "customer email",
            "full prompt",
            "judge rubric",
            "llm request",
            "llm response",
            "normalized text",
            "page content",
            "raw brief",
            "raw content",
            "raw customer",
            "secret key",
        }
    )
)


class CorpusSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class CorpusDataState(str, Enum):
    LOCAL_FIXTURE = "local_fixture"
    LAB_MEASURED = "lab_measured"
    PRODUCTION_MEASURED = "production_measured"


class CorpusReadinessState(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    READY_AFTER_MEASURED_DATA = "ready_after_measured_data"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TrajectoryCorpusSourceRecord:
    source_id: str
    trajectory_id: str
    trajectory_hash: str
    trajectory_schema_valid: bool
    event_count: int
    execution_trace_refs: tuple[str, ...]
    evidence_bundle_refs: tuple[str, ...]
    results_ledger_refs: tuple[str, ...]
    receipt_refs: tuple[str, ...]
    cost_ledger_refs: tuple[str, ...]
    release_policy_ref: str
    trajectory_rights_ref: str
    distillation_rights_ref: str
    pii_review_ref: str
    legal_gate_ref: str
    island: str
    component_refs: tuple[str, ...] = ()
    outcome_label_refs: tuple[str, ...] = ()
    split: str = CorpusSplit.TRAIN.value
    data_state: str = CorpusDataState.LOCAL_FIXTURE.value
    measured_data: bool = False
    rights_verified: bool = False
    distillation_rights_verified: bool = False
    pii_review_passed: bool = False
    legal_gate_passed: bool = False
    protected_data_scanned: bool = False
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    eligible_for_training: bool = False
    eligible_for_distillation: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrajectoryCorpusSourceRecord":
        return cls(
            source_id=str(data["source_id"]),
            trajectory_id=str(data["trajectory_id"]),
            trajectory_hash=str(data["trajectory_hash"]),
            trajectory_schema_valid=bool(data.get("trajectory_schema_valid", False)),
            event_count=int(data.get("event_count", 0)),
            execution_trace_refs=tuple(str(item) for item in data.get("execution_trace_refs", [])),
            evidence_bundle_refs=tuple(str(item) for item in data.get("evidence_bundle_refs", [])),
            results_ledger_refs=tuple(str(item) for item in data.get("results_ledger_refs", [])),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            cost_ledger_refs=tuple(str(item) for item in data.get("cost_ledger_refs", [])),
            release_policy_ref=str(data.get("release_policy_ref", "")),
            trajectory_rights_ref=str(data.get("trajectory_rights_ref", "")),
            distillation_rights_ref=str(data.get("distillation_rights_ref", "")),
            pii_review_ref=str(data.get("pii_review_ref", "")),
            legal_gate_ref=str(data.get("legal_gate_ref", "")),
            island=str(data.get("island", "")),
            component_refs=tuple(str(item) for item in data.get("component_refs", [])),
            outcome_label_refs=tuple(str(item) for item in data.get("outcome_label_refs", [])),
            split=str(data.get("split", CorpusSplit.TRAIN.value)),
            data_state=str(data.get("data_state", CorpusDataState.LOCAL_FIXTURE.value)),
            measured_data=bool(data.get("measured_data", False)),
            rights_verified=bool(data.get("rights_verified", False)),
            distillation_rights_verified=bool(data.get("distillation_rights_verified", False)),
            pii_review_passed=bool(data.get("pii_review_passed", False)),
            legal_gate_passed=bool(data.get("legal_gate_passed", False)),
            protected_data_scanned=bool(data.get("protected_data_scanned", False)),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            eligible_for_training=bool(data.get("eligible_for_training", False)),
            eligible_for_distillation=bool(data.get("eligible_for_distillation", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "execution_trace_refs",
            "evidence_bundle_refs",
            "results_ledger_refs",
            "receipt_refs",
            "cost_ledger_refs",
            "component_refs",
            "outcome_label_refs",
        ):
            data[key] = list(getattr(self, key))
        return data


@dataclass(frozen=True)
class CorpusSplitPolicyRecord:
    split_policy_id: str
    train_percent: int
    validation_percent: int
    holdout_percent: int
    deterministic_seed_ref: str
    group_by_fields: tuple[str, ...]
    leakage_guard_enabled: bool = True
    holdout_locked: bool = True
    no_cross_split_trajectory_id: bool = True
    no_cross_split_brief_id: bool = True
    no_cross_split_customer_ref: bool = True
    min_holdout_trajectories: int = MIN_HOLDOUT_TRAJECTORIES
    state: str = CorpusReadinessState.LOCAL_CONTRACT_STUB.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CorpusSplitPolicyRecord":
        return cls(
            split_policy_id=str(data["split_policy_id"]),
            train_percent=int(data["train_percent"]),
            validation_percent=int(data["validation_percent"]),
            holdout_percent=int(data["holdout_percent"]),
            deterministic_seed_ref=str(data.get("deterministic_seed_ref", "")),
            group_by_fields=tuple(str(item) for item in data.get("group_by_fields", [])),
            leakage_guard_enabled=bool(data.get("leakage_guard_enabled", True)),
            holdout_locked=bool(data.get("holdout_locked", True)),
            no_cross_split_trajectory_id=bool(data.get("no_cross_split_trajectory_id", True)),
            no_cross_split_brief_id=bool(data.get("no_cross_split_brief_id", True)),
            no_cross_split_customer_ref=bool(data.get("no_cross_split_customer_ref", True)),
            min_holdout_trajectories=int(data.get("min_holdout_trajectories", MIN_HOLDOUT_TRAJECTORIES)),
            state=str(data.get("state", CorpusReadinessState.LOCAL_CONTRACT_STUB.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["group_by_fields"] = list(self.group_by_fields)
        return data


@dataclass(frozen=True)
class TrajectoryCorpusManifestRecord:
    corpus_id: str
    source_records: tuple[TrajectoryCorpusSourceRecord, ...]
    split_policy: CorpusSplitPolicyRecord
    trajectory_count: int
    train_count: int
    validation_count: int
    holdout_count: int
    source_record_hashes: tuple[str, ...]
    local_code_complete: bool = False
    training_ready_claimed: bool = False
    distillation_ready_claimed: bool = False
    calibration_ready_claimed: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    measured_data_only: bool = False
    rights_complete: bool = False
    protected_data_excluded: bool = False
    eval_leakage_guard_passed: bool = False
    outcome_labels_ready: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = CorpusReadinessState.LOCAL_CONTRACT_STUB.value
    contract_version: str = TRAJECTORY_CORPUS_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrajectoryCorpusManifestRecord":
        return cls(
            corpus_id=str(data["corpus_id"]),
            source_records=tuple(
                TrajectoryCorpusSourceRecord.from_mapping(item)
                for item in data.get("source_records", [])
            ),
            split_policy=CorpusSplitPolicyRecord.from_mapping(data.get("split_policy", {})),
            trajectory_count=int(data.get("trajectory_count", 0)),
            train_count=int(data.get("train_count", 0)),
            validation_count=int(data.get("validation_count", 0)),
            holdout_count=int(data.get("holdout_count", 0)),
            source_record_hashes=tuple(str(item) for item in data.get("source_record_hashes", [])),
            local_code_complete=bool(data.get("local_code_complete", False)),
            training_ready_claimed=bool(data.get("training_ready_claimed", False)),
            distillation_ready_claimed=bool(data.get("distillation_ready_claimed", False)),
            calibration_ready_claimed=bool(data.get("calibration_ready_claimed", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            measured_data_only=bool(data.get("measured_data_only", False)),
            rights_complete=bool(data.get("rights_complete", False)),
            protected_data_excluded=bool(data.get("protected_data_excluded", False)),
            eval_leakage_guard_passed=bool(data.get("eval_leakage_guard_passed", False)),
            outcome_labels_ready=bool(data.get("outcome_labels_ready", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", CorpusReadinessState.LOCAL_CONTRACT_STUB.value)),
            contract_version=str(data.get("contract_version", TRAJECTORY_CORPUS_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_records"] = [record.to_dict() for record in self.source_records]
        data["split_policy"] = self.split_policy.to_dict()
        data["source_record_hashes"] = list(self.source_record_hashes)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def source_record_hash(record: TrajectoryCorpusSourceRecord | Mapping[str, Any]) -> str:
    if not isinstance(record, TrajectoryCorpusSourceRecord):
        record = TrajectoryCorpusSourceRecord.from_mapping(record)
    return sha256_json(record.to_dict())


def build_trajectory_corpus_manifest(
    *,
    corpus_id: str,
    source_records: Sequence[TrajectoryCorpusSourceRecord | Mapping[str, Any]],
    split_policy: CorpusSplitPolicyRecord | Mapping[str, Any],
    uses_local_fixtures: bool = True,
    local_only: bool = True,
    state: str = CorpusReadinessState.LOCAL_CONTRACT_STUB.value,
) -> TrajectoryCorpusManifestRecord:
    records = tuple(
        item if isinstance(item, TrajectoryCorpusSourceRecord) else TrajectoryCorpusSourceRecord.from_mapping(item)
        for item in source_records
    )
    policy = split_policy if isinstance(split_policy, CorpusSplitPolicyRecord) else CorpusSplitPolicyRecord.from_mapping(split_policy)
    counts = {split.value: 0 for split in CorpusSplit}
    for record in records:
        if record.split in counts:
            counts[record.split] += 1
    return TrajectoryCorpusManifestRecord(
        corpus_id=corpus_id,
        source_records=records,
        split_policy=policy,
        trajectory_count=len(records),
        train_count=counts[CorpusSplit.TRAIN.value],
        validation_count=counts[CorpusSplit.VALIDATION.value],
        holdout_count=counts[CorpusSplit.HOLDOUT.value],
        source_record_hashes=tuple(source_record_hash(record) for record in records),
        local_code_complete=True,
        training_ready_claimed=False,
        distillation_ready_claimed=False,
        calibration_ready_claimed=False,
        uses_local_fixtures=uses_local_fixtures,
        local_only=local_only,
        measured_data_only=all(record.measured_data for record in records),
        rights_complete=all(_source_rights_complete(record) for record in records),
        protected_data_excluded=all(_source_protected_material_free(record) for record in records),
        eval_leakage_guard_passed=_split_policy_passes(policy, records),
        outcome_labels_ready=all(record.outcome_label_refs for record in records),
        evidence_refs=tuple(sorted({ref for record in records for ref in record.results_ledger_refs})),
        state=state,
    )


def validate_trajectory_corpus_source_record(record: TrajectoryCorpusSourceRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_corpus_payload_errors(raw)
    if not isinstance(record, TrajectoryCorpusSourceRecord):
        record = TrajectoryCorpusSourceRecord.from_mapping(record)
    if not record.source_id.startswith("trajectory_source:"):
        errors.append("source_id must be trajectory_source:-prefixed")
    if not record.trajectory_id:
        errors.append("trajectory_id is required")
    if not record.trajectory_hash.startswith("sha256:"):
        errors.append("trajectory_hash must be sha256:-prefixed")
    if record.event_count <= 0:
        errors.append("event_count must be positive")
    if record.split not in {split.value for split in CorpusSplit}:
        errors.append(f"unknown corpus split: {record.split}")
    if record.data_state not in {state.value for state in CorpusDataState}:
        errors.append(f"unknown corpus data_state: {record.data_state}")
    if record.data_state == CorpusDataState.LOCAL_FIXTURE.value and record.measured_data:
        errors.append("local_fixture source records must not claim measured_data")
    if record.measured_data and record.data_state == CorpusDataState.LOCAL_FIXTURE.value:
        errors.append("measured_data requires lab_measured or production_measured data_state")
    _validate_ref_prefixes(errors, record.execution_trace_refs, "execution_trace_refs", ("execution_trace:",))
    _validate_ref_prefixes(errors, record.evidence_bundle_refs, "evidence_bundle_refs", ("evidence_bundle:",))
    _validate_ref_prefixes(errors, record.results_ledger_refs, "results_ledger_refs", ("results_ledger:",))
    _validate_ref_prefixes(errors, record.receipt_refs, "receipt_refs", ("receipt:", "receipt_v2:"))
    _validate_ref_prefixes(errors, record.cost_ledger_refs, "cost_ledger_refs", ("cost_ledger:",))
    _validate_ref_prefixes(errors, record.component_refs, "component_refs", ("component:",))
    _validate_ref_prefixes(errors, record.outcome_label_refs, "outcome_label_refs", ("outcome_label:",))
    for field, prefix in (
        ("release_policy_ref", "release_policy:"),
        ("trajectory_rights_ref", "trajectory_rights:"),
        ("distillation_rights_ref", "distillation_rights:"),
        ("pii_review_ref", "pii_review:"),
        ("legal_gate_ref", "legal_gate:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    if record.eligible_for_training:
        if not record.measured_data:
            errors.append("training eligibility requires measured_data")
        if not record.trajectory_schema_valid:
            errors.append("training eligibility requires trajectory_schema_valid")
        if not _source_rights_complete(record):
            errors.append("training eligibility requires rights, PII review, and legal gate")
        if not _source_has_required_refs(record):
            errors.append("training eligibility requires trace, evidence, ledger, receipt, and cost refs")
        if not _source_protected_material_free(record):
            errors.append("training eligibility requires protected-data exclusion")
    if record.eligible_for_distillation:
        if not record.eligible_for_training:
            errors.append("distillation eligibility requires training eligibility")
        if not record.distillation_rights_verified:
            errors.append("distillation eligibility requires distillation_rights_verified")
    return errors


def validate_corpus_split_policy_record(record: CorpusSplitPolicyRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, CorpusSplitPolicyRecord):
        record = CorpusSplitPolicyRecord.from_mapping(record)
    errors: list[str] = []
    if not record.split_policy_id.startswith("split_policy:"):
        errors.append("split_policy_id must be split_policy:-prefixed")
    if record.train_percent + record.validation_percent + record.holdout_percent != 100:
        errors.append("split percentages must sum to 100")
    if record.train_percent <= 0 or record.validation_percent <= 0 or record.holdout_percent <= 0:
        errors.append("all split percentages must be positive")
    if not record.deterministic_seed_ref.startswith("sha256:"):
        errors.append("deterministic_seed_ref must be sha256:-prefixed")
    if record.state not in {state.value for state in CorpusReadinessState}:
        errors.append(f"unknown split policy state: {record.state}")
    for required in ("trajectory_id", "brief_id", "customer_ref"):
        if required not in record.group_by_fields:
            errors.append(f"group_by_fields must include {required}")
    if not record.leakage_guard_enabled:
        errors.append("split policy requires leakage_guard_enabled")
    if not record.holdout_locked:
        errors.append("split policy requires holdout_locked")
    if not record.no_cross_split_trajectory_id:
        errors.append("split policy requires no_cross_split_trajectory_id")
    if not record.no_cross_split_brief_id:
        errors.append("split policy requires no_cross_split_brief_id")
    if not record.no_cross_split_customer_ref:
        errors.append("split policy requires no_cross_split_customer_ref")
    if record.min_holdout_trajectories < MIN_HOLDOUT_TRAJECTORIES:
        errors.append("min_holdout_trajectories must not be below Phase 3 minimum")
    return errors


def validate_trajectory_corpus_manifest(record: TrajectoryCorpusManifestRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_corpus_payload_errors(raw)
    if not isinstance(record, TrajectoryCorpusManifestRecord):
        record = TrajectoryCorpusManifestRecord.from_mapping(record)
    if record.contract_version != TRAJECTORY_CORPUS_CONTRACT_VERSION:
        errors.append("contract_version must match P3.1 trajectory corpus contract")
    if not record.corpus_id.startswith("trajectory_corpus:"):
        errors.append("corpus_id must be trajectory_corpus:-prefixed")
    if record.state not in {state.value for state in CorpusReadinessState}:
        errors.append(f"unknown corpus readiness state: {record.state}")
    errors.extend(validate_corpus_split_policy_record(record.split_policy))
    source_errors = []
    for source in record.source_records:
        source_errors.extend(validate_trajectory_corpus_source_record(source))
    if source_errors:
        errors.append("source records invalid: " + "; ".join(source_errors[:5]))
    if record.trajectory_count != len(record.source_records):
        errors.append("trajectory_count must equal source_records length")
    if record.train_count + record.validation_count + record.holdout_count != record.trajectory_count:
        errors.append("split counts must sum to trajectory_count")
    actual_counts = {split.value: 0 for split in CorpusSplit}
    for source in record.source_records:
        if source.split in actual_counts:
            actual_counts[source.split] += 1
    if record.train_count != actual_counts[CorpusSplit.TRAIN.value]:
        errors.append("train_count must match source split assignments")
    if record.validation_count != actual_counts[CorpusSplit.VALIDATION.value]:
        errors.append("validation_count must match source split assignments")
    if record.holdout_count != actual_counts[CorpusSplit.HOLDOUT.value]:
        errors.append("holdout_count must match source split assignments")
    expected_hashes = tuple(source_record_hash(source) for source in record.source_records)
    if record.source_record_hashes != expected_hashes:
        errors.append("source_record_hashes must match source records")
    recomputed_leakage_guard = _split_policy_passes(record.split_policy, record.source_records)
    if record.eval_leakage_guard_passed != recomputed_leakage_guard:
        errors.append("eval_leakage_guard_passed must match recomputed split leakage guard")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(("results_ledger:", "trajectory_corpus:", "owner_approval:")):
            errors.append("evidence_refs must use approved corpus prefixes")
            break
    if record.local_code_complete and not record.source_records:
        errors.append("local_code_complete requires at least one source record")

    wants_ready = (
        record.training_ready_claimed
        or record.distillation_ready_claimed
        or record.calibration_ready_claimed
        or record.state == CorpusReadinessState.READY_AFTER_MEASURED_DATA.value
    )
    if wants_ready:
        if record.uses_local_fixtures:
            errors.append("corpus readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("corpus readiness cannot be claimed by a local_only record")
        if record.trajectory_count < MIN_TRAINING_TRAJECTORIES:
            errors.append("corpus readiness requires minimum training trajectory count")
        if record.holdout_count < record.split_policy.min_holdout_trajectories:
            errors.append("corpus readiness requires minimum holdout trajectories")
        if not record.measured_data_only:
            errors.append("corpus readiness requires measured_data_only")
        if not record.rights_complete:
            errors.append("corpus readiness requires rights_complete")
        if not record.protected_data_excluded:
            errors.append("corpus readiness requires protected_data_excluded")
        if not record.eval_leakage_guard_passed:
            errors.append("corpus readiness requires eval_leakage_guard_passed")
        if not all(source.eligible_for_training for source in record.source_records):
            errors.append("training readiness requires every source eligible_for_training")
        if not record.owner_approval_ref:
            errors.append("corpus readiness requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("corpus readiness requires evidence_refs")
    else:
        if not record.local_only:
            errors.append("not-ready corpus manifests must remain local_only")
    if record.distillation_ready_claimed:
        if not all(source.eligible_for_distillation for source in record.source_records):
            errors.append("distillation readiness requires every source eligible_for_distillation")
    if record.calibration_ready_claimed and not record.outcome_labels_ready:
        errors.append("calibration readiness requires outcome_labels_ready")
    return errors


def verify_research_lab_trajectory_corpus(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    model_pipeline_summary = verify_model_pipeline_foundation()
    fixture = _load_fixture(Path(fixture_path))

    schema_fixture_checks = fixture["schema_fixture_checks"]
    for item in schema_fixture_checks:
        errors = validate_schema_record(item["schema"], _load_fixture(SCHEMA_FIXTURE_DIR / item["fixture"]))
        _assert(not errors if item["valid"] else bool(errors), f"schema fixture expectation: {item['fixture']}")

    source = TrajectoryCorpusSourceRecord.from_mapping(fixture["valid_source_record"])
    _assert(not validate_trajectory_corpus_source_record(source), "valid local source record validates")

    policy = CorpusSplitPolicyRecord.from_mapping(fixture["valid_split_policy"])
    _assert(not validate_corpus_split_policy_record(policy), "valid split policy validates")

    manifest_mapping = _materialize_manifest_fixture(
        fixture["local_manifest"],
        fixture["source_records_for_builder"],
    )
    manifest = TrajectoryCorpusManifestRecord.from_mapping(manifest_mapping)
    _assert(not validate_trajectory_corpus_manifest(manifest), "valid local manifest validates")
    _assert(manifest.local_code_complete, "P3.1 manifest marks local code complete")
    _assert(not manifest.training_ready_claimed, "local manifest does not claim training readiness")
    _assert(manifest.uses_local_fixtures, "local manifest stays fixture-scoped")
    _assert(model_pipeline_summary["model_pipeline_operation_claimed_ready"] is False, "P3.1 composes with not-ready P3.0")

    for invalid in fixture["invalid_sources"]:
        base = fixture[str(invalid.get("base", "valid_source_record"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_trajectory_corpus_source_record(record)
        _assert(errors, f"invalid source fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_split_policies"]:
        base = fixture[str(invalid.get("base", "valid_split_policy"))]
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_corpus_split_policy_record(record)
        _assert(errors, f"invalid split policy fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_manifests"]:
        base_name = str(invalid.get("base", "local_manifest"))
        base = fixture[base_name]
        if base_name == "local_manifest":
            base = _materialize_manifest_fixture(base, fixture["source_records_for_builder"])
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_trajectory_corpus_manifest(record)
        _assert(errors, f"invalid manifest fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    built = build_trajectory_corpus_manifest(
        corpus_id="trajectory_corpus:p3.1:built-local",
        source_records=fixture["source_records_for_builder"],
        split_policy=policy,
    )
    _assert(built.trajectory_count == len(fixture["source_records_for_builder"]), "builder counts trajectories")
    _assert(len(built.source_record_hashes) == built.trajectory_count, "builder hashes each source")
    _assert(not validate_trajectory_corpus_manifest(built), "built local manifest validates")

    return {
        "model_pipeline_readiness_id": model_pipeline_summary["readiness_id"],
        "source_records": len(manifest.source_records),
        "trajectory_count": manifest.trajectory_count,
        "train_count": manifest.train_count,
        "validation_count": manifest.validation_count,
        "holdout_count": manifest.holdout_count,
        "training_ready_claimed": manifest.training_ready_claimed,
        "distillation_ready_claimed": manifest.distillation_ready_claimed,
        "calibration_ready_claimed": manifest.calibration_ready_claimed,
        "builder_manifest_id": built.corpus_id,
    }


def _source_has_required_refs(record: TrajectoryCorpusSourceRecord) -> bool:
    return all(
        (
            record.execution_trace_refs,
            record.evidence_bundle_refs,
            record.results_ledger_refs,
            record.receipt_refs,
            record.cost_ledger_refs,
        )
    )


def _source_rights_complete(record: TrajectoryCorpusSourceRecord) -> bool:
    return (
        record.rights_verified
        and record.distillation_rights_verified
        and record.pii_review_passed
        and record.legal_gate_passed
    )


def _source_protected_material_free(record: TrajectoryCorpusSourceRecord) -> bool:
    return (
        record.protected_data_scanned
        and not record.contains_live_champion_ip
        and not record.contains_sealed_eval_details
        and not record.contains_raw_evidence_snapshot
        and not record.contains_private_customer_data
        and not record.contains_judge_prompts
    )


def _split_policy_passes(policy: CorpusSplitPolicyRecord, records: Sequence[TrajectoryCorpusSourceRecord]) -> bool:
    if validate_corpus_split_policy_record(policy):
        return False
    seen: dict[str, str] = {}
    for record in records:
        prior = seen.get(record.trajectory_id)
        if prior is not None and prior != record.split:
            return False
        seen[record.trajectory_id] = record.split
    return True


def _validate_ref_prefixes(
    errors: list[str],
    values: Sequence[str],
    label: str,
    prefixes: tuple[str, ...],
) -> None:
    for value in values:
        if not value.startswith(prefixes):
            errors.append(f"{label} must use approved prefixes")
            return


def _protected_corpus_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_corpus_material(record))
    if not found:
        return []
    return ["trajectory corpus payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_corpus_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_CORPUS_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_corpus_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_corpus_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_CORPUS_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _materialize_manifest_fixture(
    manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = dict(manifest)
    if not materialized.get("source_records"):
        materialized["source_records"] = list(source_records)
    if not materialized.get("source_record_hashes"):
        materialized["source_record_hashes"] = [source_record_hash(item) for item in materialized["source_records"]]
    return materialized


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
