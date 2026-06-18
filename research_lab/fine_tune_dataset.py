"""Phase 3.3 Engine fine-tune dataset and training contracts.

P3.3 prepares the dataset/training contract layer for Engine v-next. It does
not train models, publish weights, write registries, or claim improvement. The
validators keep local dataset construction separate from measured corpus
readiness and matched-budget evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .trajectory_corpus import (
    PROTECTED_CORPUS_KEYS,
    PROTECTED_CORPUS_MARKERS,
    CorpusReadinessState,
    MIN_HOLDOUT_TRAJECTORIES,
    MIN_TRAINING_TRAJECTORIES,
    TRAJECTORY_CORPUS_CONTRACT_VERSION,
    verify_research_lab_trajectory_corpus,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "fine_tune_dataset_fixtures.json"

FINE_TUNE_DATASET_CONTRACT_VERSION = "fine_tune_dataset:v1:local_contract"
FINE_TUNE_TRAINING_CONTRACT_VERSION = "engine_finetune_training:v1:local_contract"
MIN_FINE_TUNE_TRAIN_ROWS = MIN_TRAINING_TRAJECTORIES
MIN_FINE_TUNE_HOLDOUT_ROWS = MIN_HOLDOUT_TRAJECTORIES

PROTECTED_FINE_TUNE_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_CORPUS_KEYS)
    | {
        "base_model_weights",
        "checkpoint_bytes",
        "fine_tune_secret",
        "model_checkpoint",
        "model_weight_blob",
        "openai_api_key",
        "raw_dataset_row",
        "raw_eval_case",
        "raw_label",
        "training_payload",
        "weights",
    }
)

PROTECTED_FINE_TUNE_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_CORPUS_MARKERS)
        | {
            "base model weights",
            "checkpoint bytes",
            "fine tune secret",
            "model checkpoint",
            "model weight",
            "openai api key",
            "raw dataset",
            "raw eval",
            "raw label",
            "training payload",
        }
    )
)


class FineTuneDatasetState(str, Enum):
    LOCAL_CONTRACT_STUB = "local_contract_stub"
    READY_AFTER_CORPUS = "ready_after_corpus"
    BLOCKED = "blocked"


class FineTuneTrainingState(str, Enum):
    LOCAL_TRAINING_STUB = "local_training_stub"
    READY_FOR_LAB_TRAINING = "ready_for_lab_training"
    TRAINED_AWAITING_EVAL = "trained_awaiting_eval"
    READY_AFTER_MATCHED_BUDGET_EVAL = "ready_after_matched_budget_eval"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FineTuneDatasetManifestRecord:
    dataset_id: str
    source_corpus_ref: str
    source_corpus_hash: str
    source_contract_version: str
    source_record_hashes: tuple[str, ...]
    train_count: int
    validation_count: int
    holdout_count: int
    component_refs: tuple[str, ...]
    island_refs: tuple[str, ...]
    split_policy_ref: str
    dataset_hash: str
    dataset_card_ref: str
    rights_summary_ref: str
    protected_filter_ref: str
    leakage_report_ref: str
    measured_corpus_ready: bool = False
    rights_complete: bool = False
    protected_data_excluded: bool = False
    eval_leakage_guard_passed: bool = False
    train_eval_separation_passed: bool = False
    holdout_locked: bool = False
    local_only: bool = True
    uses_local_fixtures: bool = True
    dataset_ready_claimed: bool = False
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = FineTuneDatasetState.LOCAL_CONTRACT_STUB.value
    contract_version: str = FINE_TUNE_DATASET_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FineTuneDatasetManifestRecord":
        return cls(
            dataset_id=str(data["dataset_id"]),
            source_corpus_ref=str(data["source_corpus_ref"]),
            source_corpus_hash=str(data["source_corpus_hash"]),
            source_contract_version=str(data.get("source_contract_version", "")),
            source_record_hashes=tuple(str(item) for item in data.get("source_record_hashes", [])),
            train_count=int(data.get("train_count", 0)),
            validation_count=int(data.get("validation_count", 0)),
            holdout_count=int(data.get("holdout_count", 0)),
            component_refs=tuple(str(item) for item in data.get("component_refs", [])),
            island_refs=tuple(str(item) for item in data.get("island_refs", [])),
            split_policy_ref=str(data.get("split_policy_ref", "")),
            dataset_hash=str(data.get("dataset_hash", "")),
            dataset_card_ref=str(data.get("dataset_card_ref", "")),
            rights_summary_ref=str(data.get("rights_summary_ref", "")),
            protected_filter_ref=str(data.get("protected_filter_ref", "")),
            leakage_report_ref=str(data.get("leakage_report_ref", "")),
            measured_corpus_ready=bool(data.get("measured_corpus_ready", False)),
            rights_complete=bool(data.get("rights_complete", False)),
            protected_data_excluded=bool(data.get("protected_data_excluded", False)),
            eval_leakage_guard_passed=bool(data.get("eval_leakage_guard_passed", False)),
            train_eval_separation_passed=bool(data.get("train_eval_separation_passed", False)),
            holdout_locked=bool(data.get("holdout_locked", False)),
            local_only=bool(data.get("local_only", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            dataset_ready_claimed=bool(data.get("dataset_ready_claimed", False)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", FineTuneDatasetState.LOCAL_CONTRACT_STUB.value)),
            contract_version=str(data.get("contract_version", FINE_TUNE_DATASET_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("source_record_hashes", "component_refs", "island_refs", "evidence_refs"):
            data[key] = list(getattr(self, key))
        return data


@dataclass(frozen=True)
class EngineFineTuneTrainingRunRecord:
    training_run_id: str
    dataset_ref: str
    dataset_hash: str
    base_engine_version: str
    candidate_model_ref: str
    candidate_model_hash_ref: str
    hyperparam_config_hash: str
    training_plan_ref: str
    cost_cap_ref: str
    matched_budget_eval_ref: str = "engine_ab:pending"
    training_enabled: bool = False
    training_started: bool = False
    training_completed: bool = False
    model_weights_published: bool = False
    model_registry_write: bool = False
    production_promotion_requested: bool = False
    success_claimed: bool = False
    matched_budget_eval_passed: bool = False
    measured_yield_delta_pct: float = 0.0
    local_only: bool = True
    uses_local_fixtures: bool = True
    owner_approval_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    state: str = FineTuneTrainingState.LOCAL_TRAINING_STUB.value
    contract_version: str = FINE_TUNE_TRAINING_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EngineFineTuneTrainingRunRecord":
        return cls(
            training_run_id=str(data["training_run_id"]),
            dataset_ref=str(data["dataset_ref"]),
            dataset_hash=str(data["dataset_hash"]),
            base_engine_version=str(data["base_engine_version"]),
            candidate_model_ref=str(data["candidate_model_ref"]),
            candidate_model_hash_ref=str(data.get("candidate_model_hash_ref", "")),
            hyperparam_config_hash=str(data.get("hyperparam_config_hash", "")),
            training_plan_ref=str(data.get("training_plan_ref", "")),
            cost_cap_ref=str(data.get("cost_cap_ref", "")),
            matched_budget_eval_ref=str(data.get("matched_budget_eval_ref", "engine_ab:pending")),
            training_enabled=bool(data.get("training_enabled", False)),
            training_started=bool(data.get("training_started", False)),
            training_completed=bool(data.get("training_completed", False)),
            model_weights_published=bool(data.get("model_weights_published", False)),
            model_registry_write=bool(data.get("model_registry_write", False)),
            production_promotion_requested=bool(data.get("production_promotion_requested", False)),
            success_claimed=bool(data.get("success_claimed", False)),
            matched_budget_eval_passed=bool(data.get("matched_budget_eval_passed", False)),
            measured_yield_delta_pct=float(data.get("measured_yield_delta_pct", 0.0)),
            local_only=bool(data.get("local_only", True)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            state=str(data.get("state", FineTuneTrainingState.LOCAL_TRAINING_STUB.value)),
            contract_version=str(data.get("contract_version", FINE_TUNE_TRAINING_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def fine_tune_dataset_hash(record: FineTuneDatasetManifestRecord | Mapping[str, Any]) -> str:
    if not isinstance(record, FineTuneDatasetManifestRecord):
        record = FineTuneDatasetManifestRecord.from_mapping(record)
    payload = record.to_dict()
    payload["dataset_hash"] = "sha256:pending"
    return sha256_json(payload)


def build_fine_tune_dataset_manifest(
    *,
    dataset_id: str,
    source_corpus_ref: str,
    source_corpus_hash: str,
    source_record_hashes: Sequence[str],
    train_count: int,
    validation_count: int,
    holdout_count: int,
    component_refs: Sequence[str],
    island_refs: Sequence[str],
    split_policy_ref: str,
    dataset_card_ref: str,
    rights_summary_ref: str,
    protected_filter_ref: str,
    leakage_report_ref: str,
    uses_local_fixtures: bool = True,
    local_only: bool = True,
) -> FineTuneDatasetManifestRecord:
    manifest = FineTuneDatasetManifestRecord(
        dataset_id=dataset_id,
        source_corpus_ref=source_corpus_ref,
        source_corpus_hash=source_corpus_hash,
        source_contract_version=TRAJECTORY_CORPUS_CONTRACT_VERSION,
        source_record_hashes=tuple(str(item) for item in source_record_hashes),
        train_count=int(train_count),
        validation_count=int(validation_count),
        holdout_count=int(holdout_count),
        component_refs=tuple(str(item) for item in component_refs),
        island_refs=tuple(str(item) for item in island_refs),
        split_policy_ref=split_policy_ref,
        dataset_hash="sha256:pending",
        dataset_card_ref=dataset_card_ref,
        rights_summary_ref=rights_summary_ref,
        protected_filter_ref=protected_filter_ref,
        leakage_report_ref=leakage_report_ref,
        local_only=local_only,
        uses_local_fixtures=uses_local_fixtures,
        dataset_ready_claimed=False,
        state=FineTuneDatasetState.LOCAL_CONTRACT_STUB.value,
    )
    manifest_data = manifest.to_dict()
    manifest_data["dataset_hash"] = fine_tune_dataset_hash(manifest)
    return FineTuneDatasetManifestRecord.from_mapping(manifest_data)


def validate_fine_tune_dataset_manifest(record: FineTuneDatasetManifestRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_fine_tune_payload_errors(raw)
    if not isinstance(record, FineTuneDatasetManifestRecord):
        try:
            record = FineTuneDatasetManifestRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required fine-tune dataset field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid fine-tune dataset field value: {exc}")
            return errors
    if record.contract_version != FINE_TUNE_DATASET_CONTRACT_VERSION:
        errors.append("contract_version must match P3.3 fine-tune dataset contract")
    if record.state not in {state.value for state in FineTuneDatasetState}:
        errors.append(f"unknown fine-tune dataset state: {record.state}")
    if not record.dataset_id.startswith("fine_tune_dataset:"):
        errors.append("dataset_id must be fine_tune_dataset:-prefixed")
    if not record.source_corpus_ref.startswith("trajectory_corpus:"):
        errors.append("source_corpus_ref must be trajectory_corpus:-prefixed")
    if not record.source_corpus_hash.startswith("sha256:"):
        errors.append("source_corpus_hash must be sha256:-prefixed")
    if record.source_contract_version != TRAJECTORY_CORPUS_CONTRACT_VERSION:
        errors.append("source_contract_version must match P3.1 trajectory corpus contract")
    if record.train_count < 0 or record.validation_count < 0 or record.holdout_count < 0:
        errors.append("dataset split counts must be non-negative")
    if record.holdout_count < 1:
        errors.append("fine-tune dataset requires a holdout split")
    if not record.source_record_hashes:
        errors.append("fine-tune dataset requires source_record_hashes")
    for source_hash in record.source_record_hashes:
        if not source_hash.startswith("sha256:"):
            errors.append("source_record_hashes must be sha256:-prefixed")
            break
    _validate_prefixes(errors, record.component_refs, "component_refs", ("component:",))
    _validate_prefixes(errors, record.island_refs, "island_refs", ("island:",))
    for field, prefix in (
        ("split_policy_ref", "split_policy:"),
        ("dataset_card_ref", "dataset_card:"),
        ("rights_summary_ref", "rights_summary:"),
        ("protected_filter_ref", "protected_filter:"),
        ("leakage_report_ref", "leakage_report:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "trajectory_corpus:",
                "fine_tune_dataset:",
                "rights_summary:",
                "protected_filter:",
                "leakage_report:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved fine-tune dataset prefixes")
            break

    expected_hash = fine_tune_dataset_hash(record)
    if record.dataset_hash != expected_hash:
        errors.append("dataset_hash must match fine-tune dataset manifest")

    if record.dataset_ready_claimed or record.state == FineTuneDatasetState.READY_AFTER_CORPUS.value:
        if record.uses_local_fixtures:
            errors.append("fine-tune dataset readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("fine-tune dataset readiness cannot be claimed by a local_only record")
        if record.train_count < MIN_FINE_TUNE_TRAIN_ROWS:
            errors.append("fine-tune dataset readiness requires minimum train rows")
        if record.holdout_count < MIN_FINE_TUNE_HOLDOUT_ROWS:
            errors.append("fine-tune dataset readiness requires minimum holdout rows")
        if not record.measured_corpus_ready:
            errors.append("fine-tune dataset readiness requires measured_corpus_ready")
        if not record.rights_complete:
            errors.append("fine-tune dataset readiness requires rights_complete")
        if not record.protected_data_excluded:
            errors.append("fine-tune dataset readiness requires protected_data_excluded")
        if not record.eval_leakage_guard_passed:
            errors.append("fine-tune dataset readiness requires eval_leakage_guard_passed")
        if not record.train_eval_separation_passed:
            errors.append("fine-tune dataset readiness requires train_eval_separation_passed")
        if not record.holdout_locked:
            errors.append("fine-tune dataset readiness requires holdout_locked")
        if not record.owner_approval_ref:
            errors.append("fine-tune dataset readiness requires owner_approval_ref")
        if not record.evidence_refs:
            errors.append("fine-tune dataset readiness requires evidence_refs")
    else:
        if not record.local_only:
            errors.append("not-ready fine-tune dataset manifests must remain local_only")
    return errors


def validate_engine_fine_tune_training_run(record: EngineFineTuneTrainingRunRecord | Mapping[str, Any]) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_fine_tune_payload_errors(raw)
    if not isinstance(record, EngineFineTuneTrainingRunRecord):
        try:
            record = EngineFineTuneTrainingRunRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required fine-tune training field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid fine-tune training field value: {exc}")
            return errors
    if record.contract_version != FINE_TUNE_TRAINING_CONTRACT_VERSION:
        errors.append("contract_version must match P3.3 training contract")
    if record.state not in {state.value for state in FineTuneTrainingState}:
        errors.append(f"unknown fine-tune training state: {record.state}")
    if not record.training_run_id.startswith("engine_finetune_run:"):
        errors.append("training_run_id must be engine_finetune_run:-prefixed")
    if not record.dataset_ref.startswith("fine_tune_dataset:"):
        errors.append("dataset_ref must be fine_tune_dataset:-prefixed")
    for field in ("dataset_hash", "candidate_model_hash_ref", "hyperparam_config_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    for field, prefix in (
        ("candidate_model_ref", "model_candidate:"),
        ("training_plan_ref", "training_plan:"),
        ("cost_cap_ref", "cost_cap:"),
    ):
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")
    if not record.matched_budget_eval_ref.startswith("engine_ab:"):
        errors.append("matched_budget_eval_ref must be engine_ab:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "fine_tune_dataset:",
                "training_plan:",
                "engine_ab:",
                "cost_ledger:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved fine-tune training prefixes")
            break
    if record.training_enabled or record.training_started or record.training_completed:
        if record.uses_local_fixtures:
            errors.append("fine-tune training cannot start from local fixtures")
        if record.local_only:
            errors.append("fine-tune training cannot start from a local_only record")
        if not record.owner_approval_ref:
            errors.append("fine-tune training requires owner_approval_ref")
    if record.model_weights_published:
        errors.append("P3.3 must not publish model weights")
    if record.model_registry_write:
        errors.append("P3.3 must not write model registry entries")
    if record.production_promotion_requested:
        errors.append("P3.3 must not request production promotion")
    if record.success_claimed:
        if record.uses_local_fixtures:
            errors.append("fine-tune success cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("fine-tune success cannot be claimed by a local_only record")
        if not record.training_completed:
            errors.append("fine-tune success requires training_completed")
        if not record.matched_budget_eval_passed:
            errors.append("fine-tune success requires matched_budget_eval_passed")
        if record.measured_yield_delta_pct < 20.0:
            errors.append("fine-tune success requires measured_yield_delta_pct >= 20")
        if (
            not record.matched_budget_eval_ref.startswith("engine_ab:")
            or record.matched_budget_eval_ref == "engine_ab:pending"
        ):
            errors.append("fine-tune success requires non-pending engine_ab matched-budget evidence")
        if not record.evidence_refs:
            errors.append("fine-tune success requires evidence_refs")
        elif record.matched_budget_eval_ref not in record.evidence_refs:
            errors.append("fine-tune success evidence_refs must include matched_budget_eval_ref")
        if not record.owner_approval_ref:
            errors.append("fine-tune success requires owner_approval_ref")
    else:
        if record.state == FineTuneTrainingState.READY_AFTER_MATCHED_BUDGET_EVAL.value:
            errors.append("ready_after_matched_budget_eval state requires success_claimed")
    return errors


def verify_research_lab_fine_tune_dataset(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    corpus_summary = verify_research_lab_trajectory_corpus()
    fixture = _load_fixture(Path(fixture_path))

    dataset_mapping = _materialize_dataset_fixture(fixture["local_dataset_manifest"])
    dataset = FineTuneDatasetManifestRecord.from_mapping(dataset_mapping)
    _assert(not validate_fine_tune_dataset_manifest(dataset), "local fine-tune dataset manifest validates")
    _assert(not dataset.dataset_ready_claimed, "local dataset does not claim readiness")
    _assert(dataset.uses_local_fixtures, "local dataset is fixture-scoped")
    _assert(dataset.source_corpus_ref == corpus_summary["builder_manifest_id"], "P3.3 pins P3.1 corpus builder ref")

    training_mapping = _materialize_training_fixture(fixture["local_training_run"], dataset)
    training = EngineFineTuneTrainingRunRecord.from_mapping(training_mapping)
    _assert(not validate_engine_fine_tune_training_run(training), "local training run stub validates")
    _assert(not training.training_started, "local training run does not start training")
    _assert(not training.success_claimed, "local training run does not claim success")
    _assert(training.dataset_ref == dataset.dataset_id, "training run pins dataset ref")
    _assert(training.dataset_hash == dataset.dataset_hash, "training run pins dataset hash")

    built = build_fine_tune_dataset_manifest(**fixture["builder_args"])
    _assert(not validate_fine_tune_dataset_manifest(built), "built fine-tune dataset manifest validates")
    _assert(built.dataset_hash == fine_tune_dataset_hash(built), "builder computes dataset hash")

    for invalid in fixture["invalid_dataset_manifests"]:
        base = fixture[str(invalid.get("base", "local_dataset_manifest"))]
        if str(invalid.get("base", "local_dataset_manifest")) == "local_dataset_manifest":
            base = _materialize_dataset_fixture(base)
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_fine_tune_dataset_manifest(record)
        _assert(errors, f"invalid dataset manifest fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    for invalid in fixture["invalid_training_runs"]:
        base = fixture[str(invalid.get("base", "local_training_run"))]
        if str(invalid.get("base", "local_training_run")) == "local_training_run":
            base = _materialize_training_fixture(base, dataset)
        record = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_engine_fine_tune_training_run(record)
        _assert(errors, f"invalid training run fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    return {
        "corpus_manifest_id": corpus_summary["builder_manifest_id"],
        "dataset_id": dataset.dataset_id,
        "dataset_ready_claimed": dataset.dataset_ready_claimed,
        "training_run_id": training.training_run_id,
        "training_started": training.training_started,
        "success_claimed": training.success_claimed,
        "builder_dataset_id": built.dataset_id,
    }


def _validate_prefixes(
    errors: list[str],
    values: Sequence[str],
    label: str,
    prefixes: tuple[str, ...],
) -> None:
    for value in values:
        if not value.startswith(prefixes):
            errors.append(f"{label} must use approved prefixes")
            return


def _protected_fine_tune_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_fine_tune_material(record))
    if not found:
        return []
    return ["fine-tune dataset payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_fine_tune_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_FINE_TUNE_KEYS and not key_text.endswith(("_ref", "_refs", "_hash")):
                found.add(key_path)
            found.update(_find_protected_fine_tune_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_fine_tune_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_FINE_TUNE_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _materialize_dataset_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(record)
    if materialized.get("dataset_hash") == "sha256:pending":
        candidate = FineTuneDatasetManifestRecord.from_mapping(materialized)
        materialized["dataset_hash"] = fine_tune_dataset_hash(candidate)
    return materialized


def _materialize_training_fixture(
    record: Mapping[str, Any],
    dataset: FineTuneDatasetManifestRecord,
) -> dict[str, Any]:
    materialized = dict(record)
    if materialized.get("dataset_hash") == "sha256:from-local-dataset":
        materialized["dataset_hash"] = dataset.dataset_hash
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
