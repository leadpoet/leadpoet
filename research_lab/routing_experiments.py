"""ICP-aware intent routing experiments for the Research Lab.

This module is the Lab control-plane contract for provider admission.  It is
deliberately independent from credentialed provider execution and production
activation:

* the model artifact and routing contract are pinned to ``leadpoet-lab``;
* ICP segments and provider bindings are frozen by hash;
* LLM output is only a bounded proposal and can never promote itself;
* calibration and holdout runs use the same content-addressed receipt store;
* provider results have a small, typed outcome vocabulary;
* promotion creates an immutable receipt and never mutates a live pointer.

The production worker and the Sourcing_model consumer should call the same
model-owned route compiler.  This module only evaluates bounded route
variants in the Lab.  The runner seam accepts already-redacted receipts and
does not know credentials, URLs, prompts, or response bodies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .canonical import sha256_json


ROUTING_EXPERIMENT_CONTRACT_VERSION = "leadpoet.intent_routing_experiment:v1"
ROUTING_EVALUATION_RECEIPT_VERSION = "leadpoet.intent_routing_evaluation:v1"
ROUTING_PROMOTION_RECEIPT_VERSION = "leadpoet.intent_routing_promotion:v1"

MAX_PROVIDER_BINDINGS = 64
MAX_PROFILE_VARIANTS = 24
MAX_TOOLS_PER_VARIANT = 6
MAX_UNITS_PER_SPLIT = 5000
MAX_CREDIT_MICROUNITS_PER_VARIANT = 10_000_000
MAX_LATENCY_MS = 900_000

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_SAFE_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,95}$")
_SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,95}$")
_FORBIDDEN_MARKERS = (
    "api_key",
    "access_token",
    "authorization",
    "bearer ",
    "password",
    "raw_credential",
    "raw_secret",
    "client_secret",
    "credential",
    "request_body",
    "response_body",
    "response_text",
    "scraped_content",
    "secret_value",
    "service_role",
    "sk-or-",
    "openrouter_api_key",
)


class RoutingExperimentError(ValueError):
    """A routing experiment violates its immutable Lab contract."""


class ProviderOutcome(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    SOURCE_MISS = "source_miss"
    ADAPTER_FAILURE = "adapter_failure"


class ReceiptExecutionMode(str, Enum):
    FIXTURE = "fixture"
    REPLAY = "replay"
    MEASURED_LAB = "measured_lab"


class ProfileLifecycle(str, Enum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    PUBLISHED = "published"
    RETIRED = "retired"


class ProposalState(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"


def _ensure_safe_ref(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_REF_RE.fullmatch(text):
        raise RoutingExperimentError(f"{field_name} is invalid")
    return text


def _ensure_hash(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_RE.fullmatch(text):
        raise RoutingExperimentError(f"{field_name} must be a sha256 hash")
    return text


def model_hash_to_lab(value: Any, field_name: str = "model_hash") -> str:
    """Convert one model raw SHA-256 digest to the Lab's tagged encoding.

    The model contract uses a raw 64-character lowercase hex digest.  Lab
    receipts use ``sha256:<hex>`` so that a hash cannot be confused with an
    arbitrary identifier.  This is the only encoding boundary.
    """

    text = str(value or "").strip().lower()
    if not _RAW_SHA256_RE.fullmatch(text):
        raise RoutingExperimentError(f"{field_name} must be a raw SHA-256 hash")
    return f"sha256:{text}"


def lab_hash_to_model(value: Any, field_name: str = "lab_hash") -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise RoutingExperimentError(f"{field_name} must be a tagged SHA-256 hash")
    return text.split(":", 1)[1]


def _ensure_git_sha(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not _GIT_SHA_RE.fullmatch(text):
        raise RoutingExperimentError(f"{field_name} must be a full 40-character git SHA")
    return text


def _ensure_feature_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise RoutingExperimentError(f"{field_name} must be a sequence")
    normalized = tuple(
        dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip())
    )
    if any(not _SAFE_FEATURE_RE.fullmatch(value) for value in normalized):
        raise RoutingExperimentError(f"{field_name} contains an invalid feature")
    if len(normalized) > 64:
        raise RoutingExperimentError(f"{field_name} is too large")
    return normalized


def _ensure_ref_tuple(values: Any, field_name: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise RoutingExperimentError(f"{field_name} must be a sequence")
    normalized = tuple(sorted({_ensure_safe_ref(value, field_name) for value in values}))
    if len(normalized) > maximum:
        raise RoutingExperimentError(f"{field_name} is too large")
    return normalized


def _ensure_no_secret_material(value: Any, *, field_name: str = "document") -> None:
    """Reject secret-like keys and values before any receipt is persisted."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).strip().lower().replace("-", "_")
            if any(marker in lowered_key for marker in _FORBIDDEN_MARKERS):
                raise RoutingExperimentError(f"{field_name} contains forbidden field {key}")
            _ensure_no_secret_material(item, field_name=field_name)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _ensure_no_secret_material(item, field_name=field_name)
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
            raise RoutingExperimentError(f"{field_name} contains secret-like material")


def _bounded_int(value: Any, field_name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RoutingExperimentError(f"{field_name} is out of bounds")
    return value


def _bounded_float(value: Any, field_name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoutingExperimentError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RoutingExperimentError(f"{field_name} is out of bounds")
    return round(number, 8)


@dataclass(frozen=True)
class SourcingModelArtifactIdentity:
    """Exact branch-specific model identity consumed by the Lab."""

    repository: str
    branch: str
    commit_sha: str
    artifact_uri: str
    model_artifact_hash: str
    manifest_hash: str
    routing_contract_hash: str
    routing_catalog_hash: str
    routing_policy_hash: str
    feature_schema_hash: str
    verifier_contract_hash: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourcingModelArtifactIdentity":
        return cls(
            repository=str(data.get("repository") or ""),
            branch=str(data.get("branch") or ""),
            commit_sha=str(data.get("commit_sha") or ""),
            artifact_uri=str(data.get("artifact_uri") or ""),
            model_artifact_hash=str(data.get("model_artifact_hash") or ""),
            manifest_hash=str(data.get("manifest_hash") or ""),
            routing_contract_hash=str(data.get("routing_contract_hash") or ""),
            routing_catalog_hash=str(data.get("routing_catalog_hash") or ""),
            routing_policy_hash=str(data.get("routing_policy_hash") or ""),
            feature_schema_hash=str(data.get("feature_schema_hash") or ""),
            verifier_contract_hash=str(data.get("verifier_contract_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_payload(self) -> dict[str, Any]:
        return self.to_dict()


def validate_sourcing_model_artifact_identity(
    identity: SourcingModelArtifactIdentity | Mapping[str, Any],
) -> list[str]:
    if not isinstance(identity, SourcingModelArtifactIdentity):
        identity = SourcingModelArtifactIdentity.from_mapping(identity)
    errors: list[str] = []
    if identity.repository != "leadpoet/Sourcing_model":
        errors.append("artifact_repository_must_be_leadpoet_sourcing_model")
    if identity.branch != "leadpoet-lab":
        errors.append("artifact_branch_must_be_leadpoet_lab")
    try:
        _ensure_git_sha(identity.commit_sha, "commit_sha")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    if not identity.artifact_uri.startswith("s3://"):
        errors.append("artifact_uri_must_be_s3")
    for field_name in (
        "model_artifact_hash",
        "manifest_hash",
        "routing_contract_hash",
        "routing_catalog_hash",
        "routing_policy_hash",
        "feature_schema_hash",
        "verifier_contract_hash",
    ):
        try:
            _ensure_hash(getattr(identity, field_name), field_name)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    try:
        _ensure_no_secret_material(identity.to_dict(), field_name="artifact_identity")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return errors


@dataclass(frozen=True)
class ProviderBindingIdentity:
    """Credential-free identity of one model-owned provider binding."""

    binding_id: str
    provider_id: str
    tool_id: str
    source_lineage_id: str
    adapter_version: str
    manifest_hash: str
    capability_hash: str
    execution_contract_hash: str
    cost_model_hash: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderBindingIdentity":
        return cls(
            binding_id=str(data.get("binding_id") or ""),
            provider_id=str(data.get("provider_id") or ""),
            tool_id=str(data.get("tool_id") or ""),
            source_lineage_id=str(data.get("source_lineage_id") or ""),
            adapter_version=str(data.get("adapter_version") or ""),
            manifest_hash=str(data.get("manifest_hash") or ""),
            capability_hash=str(data.get("capability_hash") or ""),
            execution_contract_hash=str(data.get("execution_contract_hash") or ""),
            cost_model_hash=str(data.get("cost_model_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_provider_binding_identity(
    binding: ProviderBindingIdentity | Mapping[str, Any],
) -> list[str]:
    if not isinstance(binding, ProviderBindingIdentity):
        binding = ProviderBindingIdentity.from_mapping(binding)
    errors: list[str] = []
    for field_name in ("binding_id", "provider_id", "tool_id", "source_lineage_id", "adapter_version"):
        value = str(getattr(binding, field_name) or "")
        regex = _SAFE_PROVIDER_RE if field_name in {"provider_id", "tool_id"} else _SAFE_REF_RE
        if not regex.fullmatch(value):
            errors.append(f"{field_name}_is_invalid")
    for field_name in ("manifest_hash", "capability_hash", "execution_contract_hash", "cost_model_hash"):
        try:
            _ensure_hash(getattr(binding, field_name), field_name)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    try:
        _ensure_no_secret_material(binding.to_dict(), field_name="provider_binding")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return errors


@dataclass(frozen=True)
class FrozenRoutingInput:
    """A hash-bound ICP segment and split membership, without raw ICP text."""

    segment_ref: str
    signal_type: str
    features: tuple[str, ...]
    feature_set_hash: str
    calibration_unit_refs: tuple[str, ...]
    holdout_unit_refs: tuple[str, ...]
    gold_label_set_hash: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrozenRoutingInput":
        return cls(
            segment_ref=str(data.get("segment_ref") or ""),
            signal_type=str(data.get("signal_type") or ""),
            features=_ensure_feature_tuple(data.get("features", ()), "features"),
            feature_set_hash=str(data.get("feature_set_hash") or ""),
            calibration_unit_refs=_ensure_ref_tuple(
                data.get("calibration_unit_refs", ()), "calibration_unit_refs", maximum=MAX_UNITS_PER_SPLIT
            ),
            holdout_unit_refs=_ensure_ref_tuple(
                data.get("holdout_unit_refs", ()), "holdout_unit_refs", maximum=MAX_UNITS_PER_SPLIT
            ),
            gold_label_set_hash=str(data.get("gold_label_set_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = list(self.features)
        data["calibration_unit_refs"] = list(self.calibration_unit_refs)
        data["holdout_unit_refs"] = list(self.holdout_unit_refs)
        return data

def validate_frozen_routing_input(
    frozen: FrozenRoutingInput | Mapping[str, Any],
) -> list[str]:
    if not isinstance(frozen, FrozenRoutingInput):
        frozen = FrozenRoutingInput.from_mapping(frozen)
    errors: list[str] = []
    try:
        _ensure_safe_ref(frozen.segment_ref, "segment_ref")
        _ensure_safe_ref(frozen.signal_type, "signal_type")
        _ensure_hash(frozen.feature_set_hash, "feature_set_hash")
        _ensure_hash(frozen.gold_label_set_hash, "gold_label_set_hash")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    if not frozen.calibration_unit_refs or not frozen.holdout_unit_refs:
        errors.append("calibration_and_holdout_units_are_required")
    if any(not (item.startswith("icp.") or item.startswith("company.")) for item in frozen.features):
        errors.append("feature_ids_must_use_model_namespace")
    if set(frozen.calibration_unit_refs).intersection(frozen.holdout_unit_refs):
        errors.append("calibration_and_holdout_units_must_be_disjoint")
    # The feature-set digest is model-owned.  The Lab stores and compares it
    # byte-for-byte; it must never add segment or signal fields and recompute a
    # second meaning for the same hash.
    try:
        _ensure_no_secret_material(frozen.to_dict(), field_name="frozen_routing_input")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return errors


class RoutingAdmissionPlanAdapter(Protocol):
    """The narrow seam to the model-owned ``RoutingAdmissionPlan`` schema.

    Implementations live with the pinned Sourcing_model artifact.  The Lab
    does not define or inspect route steps, feature predicates, stages, or
    fallback semantics.  It only asks the model adapter to parse, validate,
    serialize, identify, and enumerate the already-owned plan.
    """

    def parse(self, payload: Mapping[str, Any]) -> Any: ...

    def validate(
        self,
        plan: Any,
        *,
        signal_type: str,
        feature_set_hash: str,
        binding_tool_ids: frozenset[str],
    ) -> Sequence[str]: ...

    def as_payload(self, plan: Any) -> Mapping[str, Any]: ...

    def profile_id(self, plan: Any) -> str: ...

    def profile_hash(self, plan: Any) -> str: ...

    def ordered_tool_ids(self, plan: Any) -> Sequence[str]: ...


@dataclass(frozen=True)
class LabRoutingProfile:
    """A Lab identity wrapper around an exact model ``RoutingProfile`` payload."""

    profile_payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.profile_payload, Mapping):
            raise RoutingExperimentError("profile_payload must be a model payload object")
        return {"profile_payload": dict(self.profile_payload)}


def _model_profile(
    profile: LabRoutingProfile | Mapping[str, Any],
    adapter: RoutingAdmissionPlanAdapter,
) -> Any:
    payload = profile.profile_payload if isinstance(profile, LabRoutingProfile) else profile
    if not isinstance(payload, Mapping):
        raise RoutingExperimentError("model routing profile payload must be an object")
    try:
        return adapter.parse(payload)
    except Exception as exc:  # adapter errors become a deterministic Lab rejection
        raise RoutingExperimentError(f"model_routing_profile_parse_failed:{exc}") from exc


def _profile_id(profile: LabRoutingProfile, adapter: RoutingAdmissionPlanAdapter) -> str:
    value = str(adapter.profile_id(_model_profile(profile, adapter)) or "")
    try:
        return _ensure_safe_ref(value, "profile_id")
    except RoutingExperimentError as exc:
        raise RoutingExperimentError(f"model_routing_profile_id_invalid:{exc}") from exc


def _profile_hash(profile: LabRoutingProfile, adapter: RoutingAdmissionPlanAdapter) -> str:
    return model_hash_to_lab(
        adapter.profile_hash(_model_profile(profile, adapter)),
        "model_profile_hash",
    )


def validate_model_routing_profile(
    profile: LabRoutingProfile | Mapping[str, Any],
    *,
    adapter: RoutingAdmissionPlanAdapter,
    signal_type: str,
    feature_set_hash: str,
    binding_tool_ids: Iterable[str],
) -> list[str]:
    """Validate the exact model payload through the pinned model adapter."""

    errors: list[str] = []
    try:
        plan = _model_profile(profile, adapter)
        errors.extend(
            str(item)
            for item in adapter.validate(
                plan,
                signal_type=signal_type,
                feature_set_hash=feature_set_hash,
                binding_tool_ids=frozenset(binding_tool_ids),
            )
        )
        canonical = adapter.as_payload(plan)
        if not isinstance(canonical, Mapping):
            errors.append("model_routing_profile_payload_is_not_an_object")
        elif dict(canonical) != dict(profile.profile_payload if isinstance(profile, LabRoutingProfile) else profile):
            errors.append("model_routing_profile_payload_not_canonical")
        _ensure_no_secret_material(canonical, field_name="model_routing_profile")
        raw_hash = str(adapter.profile_hash(plan) or "").strip().lower()
        if not _RAW_SHA256_RE.fullmatch(raw_hash):
            errors.append("model_routing_profile_hash_is_not_raw_sha256")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"model_routing_profile_validation_failed:{exc}")
    return sorted(set(errors))


@dataclass(frozen=True)
class RoutingEvaluationGates:
    min_calibration_precision: float = 0.80
    min_holdout_precision: float = 0.80
    min_holdout_recall: float = 0.10
    max_holdout_no_signal_credit_microunits: int = MAX_CREDIT_MICROUNITS_PER_VARIANT
    min_marginal_verified_positives_per_credit: float = 0.0
    intent_release_policy_hash: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RoutingEvaluationGates":
        return cls(
            min_calibration_precision=float(data.get("min_calibration_precision", 0.80)),
            min_holdout_precision=float(data.get("min_holdout_precision", 0.80)),
            min_holdout_recall=float(data.get("min_holdout_recall", 0.10)),
            max_holdout_no_signal_credit_microunits=int(
                data.get("max_holdout_no_signal_credit_microunits", MAX_CREDIT_MICROUNITS_PER_VARIANT)
            ),
            min_marginal_verified_positives_per_credit=float(
                data.get("min_marginal_verified_positives_per_credit", 0.0)
            ),
            intent_release_policy_hash=str(data.get("intent_release_policy_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_routing_evaluation_gates(
    gates: RoutingEvaluationGates | Mapping[str, Any],
) -> list[str]:
    if not isinstance(gates, RoutingEvaluationGates):
        gates = RoutingEvaluationGates.from_mapping(gates)
    errors: list[str] = []
    for field_name in (
        "min_calibration_precision",
        "min_holdout_precision",
        "min_holdout_recall",
        "min_marginal_verified_positives_per_credit",
    ):
        try:
            _bounded_float(getattr(gates, field_name), field_name, minimum=0.0, maximum=1.0)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    try:
        _ensure_hash(gates.intent_release_policy_hash, "intent_release_policy_hash")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    try:
        _bounded_int(
            gates.max_holdout_no_signal_credit_microunits,
            "max_holdout_no_signal_credit_microunits",
            minimum=0,
            maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT,
        )
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return errors


@dataclass(frozen=True)
class ExperimentCreditBudget:
    """Explicit measured-Lab spend budget, separate from model route semantics."""

    total_credit_microunits: int
    provider_credit_ceilings: Mapping[str, int]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentCreditBudget":
        raw = data.get("provider_credit_ceilings") or {}
        return cls(
            total_credit_microunits=int(data.get("total_credit_microunits", 0)),
            provider_credit_ceilings={str(key): int(value) for key, value in raw.items()}
            if isinstance(raw, Mapping)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_credit_microunits": self.total_credit_microunits,
            "provider_credit_ceilings": dict(sorted(self.provider_credit_ceilings.items())),
        }


def validate_experiment_credit_budget(
    budget: ExperimentCreditBudget,
    *,
    binding_ids: Iterable[str],
) -> list[str]:
    errors: list[str] = []
    try:
        _bounded_int(
            budget.total_credit_microunits,
            "total_credit_microunits",
            minimum=0,
            maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT * MAX_PROFILE_VARIANTS,
        )
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    known = set(binding_ids)
    for binding_id, ceiling in budget.provider_credit_ceilings.items():
        if binding_id not in known:
            errors.append(f"credit_ceiling_unknown_binding:{binding_id}")
        try:
            _bounded_int(
                ceiling,
                f"provider_credit_ceiling:{binding_id}",
                minimum=0,
                maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT * MAX_PROFILE_VARIANTS,
            )
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    return sorted(set(errors))


@dataclass(frozen=True)
class RoutingExperimentSpec:
    experiment_id: str
    signal_type: str
    artifact: SourcingModelArtifactIdentity
    frozen_input: FrozenRoutingInput
    provider_bindings: tuple[ProviderBindingIdentity, ...]
    # These are exact ``RoutingProfile.as_payload()`` documents from the
    # model.  A Lab wrapper carries them, but never interprets route steps.
    variants: tuple[LabRoutingProfile, ...]
    baseline_profile_id: str
    gates: RoutingEvaluationGates
    credit_budget: ExperimentCreditBudget = field(
        default_factory=lambda: ExperimentCreditBudget(0, {})
    )
    allow_live_credit_spend: bool = False
    lifecycle: str = ProfileLifecycle.DRAFT.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RoutingExperimentSpec":
        bindings = tuple(
            ProviderBindingIdentity.from_mapping(item)
            for item in (data.get("provider_bindings") or ())
            if isinstance(item, Mapping)
        )
        variants = tuple(
            LabRoutingProfile(
                profile_payload=dict(
                    item.get("profile_payload", item.get("profile", item))
                )
            )
            for item in (data.get("variants") or ())
            if isinstance(item, Mapping)
        )
        return cls(
            experiment_id=str(data.get("experiment_id") or ""),
            signal_type=str(data.get("signal_type") or ""),
            artifact=SourcingModelArtifactIdentity.from_mapping(data.get("artifact") or {}),
            frozen_input=FrozenRoutingInput.from_mapping(data.get("frozen_input") or {}),
            provider_bindings=tuple(sorted(bindings, key=lambda item: item.binding_id)),
            variants=variants,
            baseline_profile_id=str(data.get("baseline_profile_id") or ""),
            gates=RoutingEvaluationGates.from_mapping(data.get("gates") or {}),
            credit_budget=ExperimentCreditBudget.from_mapping(
                data.get("credit_budget") or {}
            ),
            allow_live_credit_spend=bool(data.get("allow_live_credit_spend", False)),
            lifecycle=str(data.get("lifecycle", ProfileLifecycle.DRAFT.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": ROUTING_EXPERIMENT_CONTRACT_VERSION,
            "experiment_id": self.experiment_id,
            "signal_type": self.signal_type,
            "artifact": self.artifact.to_dict(),
            "frozen_input": self.frozen_input.to_dict(),
            "provider_bindings": [item.to_dict() for item in self.provider_bindings],
            "variants": [item.to_dict() for item in self.variants],
            "baseline_profile_id": self.baseline_profile_id,
            "gates": self.gates.to_dict(),
            "credit_budget": self.credit_budget.to_dict(),
            "allow_live_credit_spend": self.allow_live_credit_spend,
            "lifecycle": self.lifecycle,
        }

    def identity_payload(self) -> dict[str, Any]:
        return self.to_dict()

    def experiment_hash(self) -> str:
        return sha256_json(self.identity_payload())


def validate_routing_experiment_spec(
    spec: RoutingExperimentSpec | Mapping[str, Any],
    *,
    adapter: RoutingAdmissionPlanAdapter,
) -> list[str]:
    if not isinstance(spec, RoutingExperimentSpec):
        spec = RoutingExperimentSpec.from_mapping(spec)
    errors: list[str] = []
    try:
        _ensure_safe_ref(spec.experiment_id, "experiment_id")
        _ensure_safe_ref(spec.signal_type, "signal_type")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    errors.extend(validate_sourcing_model_artifact_identity(spec.artifact))
    errors.extend(validate_frozen_routing_input(spec.frozen_input))
    if spec.frozen_input.signal_type != spec.signal_type:
        errors.append("frozen_input_signal_type_mismatch")
    if not 1 <= len(spec.provider_bindings) <= MAX_PROVIDER_BINDINGS:
        errors.append("provider_binding_count_out_of_bounds")
    binding_ids = {item.tool_id for item in spec.provider_bindings}
    seen_binding_ids: set[str] = set()
    seen_tool_ids: set[str] = set()
    seen_source_lineages: dict[str, str] = {}
    for binding in spec.provider_bindings:
        errors.extend(validate_provider_binding_identity(binding))
        if binding.binding_id in seen_binding_ids:
            errors.append("provider_binding_ids_must_be_unique")
        seen_binding_ids.add(binding.binding_id)
        if binding.tool_id in seen_tool_ids:
            errors.append("provider_tool_ids_must_be_unique")
        seen_tool_ids.add(binding.tool_id)
        previous = seen_source_lineages.get(binding.tool_id)
        if previous and previous != binding.source_lineage_id:
            errors.append("tool_id_must_have_one_source_lineage")
        seen_source_lineages[binding.tool_id] = binding.source_lineage_id
    if not 1 <= len(spec.variants) <= MAX_PROFILE_VARIANTS:
        errors.append("profile_variant_count_out_of_bounds")
    profile_ids: set[str] = set()
    for variant in spec.variants:
        try:
            profile_ids.add(_profile_id(variant, adapter))
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    if len(profile_ids) != len(spec.variants):
        errors.append("profile_ids_must_be_unique")
    for variant in spec.variants:
        errors.extend(
            validate_model_routing_profile(
                variant,
                adapter=adapter,
                signal_type=spec.signal_type,
                feature_set_hash=spec.frozen_input.feature_set_hash,
                binding_tool_ids=binding_ids,
            )
        )
    if spec.baseline_profile_id not in profile_ids:
        errors.append("baseline_profile_id_must_reference_variant")
    if spec.lifecycle not in {state.value for state in ProfileLifecycle}:
        errors.append("experiment_lifecycle_is_invalid")
    errors.extend(
        validate_experiment_credit_budget(
            spec.credit_budget,
            binding_ids=(item.binding_id for item in spec.provider_bindings),
        )
    )
    if spec.allow_live_credit_spend and spec.credit_budget.total_credit_microunits <= 0:
        errors.append("measured_lab_requires_total_credit_cap")
    try:
        errors.extend(validate_routing_evaluation_gates(spec.gates))
        if spec.gates.intent_release_policy_hash != spec.artifact.routing_policy_hash:
            errors.append("evaluation_gates_policy_hash_mismatch")
        _ensure_no_secret_material(spec.to_dict(), field_name="routing_experiment")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return sorted(set(errors))


@dataclass(frozen=True)
class LLMRoutingProfileProposal:
    """A bounded proposal.  It contains no authority to promote or publish."""

    proposal_id: str
    experiment_id: str
    proposer_model_ref: str
    feature_set_hash: str
    # Exact model-owned RoutingProfile.as_payload(); no Lab route schema.
    proposed_profile: LabRoutingProfile
    rationale_hash: str
    state: str = ProposalState.PROPOSED.value
    promoted: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LLMRoutingProfileProposal":
        return cls(
            proposal_id=str(data.get("proposal_id") or ""),
            experiment_id=str(data.get("experiment_id") or ""),
            proposer_model_ref=str(data.get("proposer_model_ref") or ""),
            feature_set_hash=str(data.get("feature_set_hash") or ""),
            proposed_profile=LabRoutingProfile(
                profile_payload=dict(
                    data.get("proposed_profile", data.get("proposed_variant")) or {}
                )
            ),
            rationale_hash=str(data.get("rationale_hash") or ""),
            state=str(data.get("state", ProposalState.PROPOSED.value)),
            promoted=bool(data.get("promoted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposed_profile"] = self.proposed_profile.to_dict()
        data.pop("proposed_variant", None)
        return data


def validate_llm_routing_profile_proposal(
    proposal: LLMRoutingProfileProposal | Mapping[str, Any],
    spec: RoutingExperimentSpec,
    *,
    adapter: RoutingAdmissionPlanAdapter,
) -> list[str]:
    if not isinstance(proposal, LLMRoutingProfileProposal):
        proposal = LLMRoutingProfileProposal.from_mapping(proposal)
    errors: list[str] = []
    try:
        _ensure_safe_ref(proposal.proposal_id, "proposal_id")
        _ensure_safe_ref(proposal.experiment_id, "proposal_experiment_id")
        _ensure_safe_ref(proposal.proposer_model_ref, "proposer_model_ref")
        _ensure_hash(proposal.feature_set_hash, "proposal_feature_set_hash")
        _ensure_hash(proposal.rationale_hash, "rationale_hash")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    if proposal.experiment_id != spec.experiment_id:
        errors.append("proposal_experiment_id_mismatch")
    if proposal.feature_set_hash != spec.frozen_input.feature_set_hash:
        errors.append("proposal_feature_set_hash_mismatch")
    if proposal.promoted:
        errors.append("llm_proposal_cannot_be_marked_promoted")
    if proposal.state not in {state.value for state in ProposalState}:
        errors.append("proposal_state_is_invalid")
    errors.extend(
        validate_model_routing_profile(
            proposal.proposed_profile,
            adapter=adapter,
            signal_type=spec.signal_type,
            feature_set_hash=spec.frozen_input.feature_set_hash,
            binding_tool_ids={item.tool_id for item in spec.provider_bindings},
        )
    )
    try:
        _ensure_no_secret_material(proposal.to_dict(), field_name="llm_proposal")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    return sorted(set(errors))


def admit_llm_routing_profile_proposal(
    proposal: LLMRoutingProfileProposal | Mapping[str, Any],
    spec: RoutingExperimentSpec,
    *,
    adapter: RoutingAdmissionPlanAdapter,
) -> LLMRoutingProfileProposal:
    """Validate an LLM proposal, without changing profile lifecycle."""

    if not isinstance(proposal, LLMRoutingProfileProposal):
        proposal = LLMRoutingProfileProposal.from_mapping(proposal)
    errors = validate_llm_routing_profile_proposal(proposal, spec, adapter=adapter)
    if errors:
        return replace(proposal, state=ProposalState.REJECTED.value, promoted=False)
    return replace(proposal, state=ProposalState.VALIDATED.value, promoted=False)


@dataclass(frozen=True)
class ProviderReceipt:
    """One redacted provider result, keyed by request identity."""

    receipt_ref: str
    binding_id: str
    tool_id: str
    binding_version: str
    source_lineage_id: str
    unit_ref: str
    request_fingerprint: str
    outcome: str
    evidence_hash: str
    credit_microunits: int
    latency_ms: int
    execution_mode: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderReceipt":
        return cls(
            receipt_ref=str(data.get("receipt_ref") or ""),
            binding_id=str(data.get("binding_id") or ""),
            tool_id=str(data.get("tool_id") or ""),
            binding_version=str(data.get("binding_version") or ""),
            source_lineage_id=str(data.get("source_lineage_id") or ""),
            unit_ref=str(data.get("unit_ref") or ""),
            request_fingerprint=str(data.get("request_fingerprint") or ""),
            outcome=str(data.get("outcome") or ""),
            evidence_hash=str(data.get("evidence_hash") or ""),
            credit_microunits=int(data.get("credit_microunits", 0)),
            latency_ms=int(data.get("latency_ms", 0)),
            execution_mode=str(data.get("execution_mode") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def provider_receipt_key(
    *,
    tool_id: str,
    binding_version: str,
    request_fingerprint: str,
    # Accepted for call-site readability, deliberately excluded from the key.
    binding_id: str = "",
    source_lineage_id: str = "",
    unit_ref: str = "",
) -> str:
    """Key one paid request by exact tool/binding version and input fingerprint.

    Unit labels and source-lineage labels are evaluation metadata.  They are
    not part of request identity, so an identical paid request can be reused
    by another unit or route variant.
    """

    del binding_id, source_lineage_id, unit_ref
    return sha256_json(
        {
            "contract_version": "leadpoet.provider_receipt_key:v2",
            "tool_id": _ensure_safe_ref(tool_id, "tool_id"),
            "binding_version": _ensure_safe_ref(binding_version, "binding_version"),
            "request_fingerprint": _ensure_hash(request_fingerprint, "request_fingerprint"),
        }
    )


def validate_provider_receipt(receipt: ProviderReceipt | Mapping[str, Any]) -> list[str]:
    if not isinstance(receipt, ProviderReceipt):
        receipt = ProviderReceipt.from_mapping(receipt)
    errors: list[str] = []
    for field_name in (
        "receipt_ref",
        "binding_id",
        "tool_id",
        "binding_version",
        "source_lineage_id",
        "unit_ref",
    ):
        try:
            _ensure_safe_ref(getattr(receipt, field_name), field_name)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    for field_name in ("request_fingerprint", "evidence_hash"):
        try:
            _ensure_hash(getattr(receipt, field_name), field_name)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    if receipt.outcome not in {outcome.value for outcome in ProviderOutcome}:
        errors.append("provider_outcome_is_invalid")
    if receipt.execution_mode not in {mode.value for mode in ReceiptExecutionMode}:
        errors.append("provider_receipt_execution_mode_is_invalid")
    try:
        _bounded_int(receipt.credit_microunits, "credit_microunits", minimum=0, maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT)
        _bounded_int(receipt.latency_ms, "latency_ms", minimum=0, maximum=MAX_LATENCY_MS)
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    try:
        _ensure_no_secret_material(receipt.to_dict(), field_name="provider_receipt")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    identity_payload = receipt.to_dict()
    identity_payload.pop("receipt_ref", None)
    expected_ref = "provider_receipt:" + sha256_json(identity_payload).split(":", 1)[1][:16]
    if receipt.receipt_ref != expected_ref:
        errors.append("provider_receipt_ref_mismatch")
    return sorted(set(errors))


class ProviderReceiptRepository(Protocol):
    """Durable persistence seam for redacted Lab provider receipts."""

    def get(self, key: str) -> ProviderReceipt | None: ...

    def append(self, key: str, receipt: ProviderReceipt) -> ProviderReceipt: ...

    def keys(self) -> Iterable[str]: ...


class InMemoryProviderReceiptRepository:
    """Test repository; production must bind the same interface durably."""

    def __init__(self) -> None:
        self._rows: dict[str, ProviderReceipt] = {}

    def get(self, key: str) -> ProviderReceipt | None:
        return self._rows.get(str(key))

    def append(self, key: str, receipt: ProviderReceipt) -> ProviderReceipt:
        existing = self._rows.get(str(key))
        if existing is not None and existing.to_dict() != receipt.to_dict():
            raise RoutingExperimentError("provider receipt key collision")
        self._rows[str(key)] = receipt
        return receipt

    def keys(self) -> Iterable[str]:
        return tuple(self._rows)


class JsonlProviderReceiptRepository:
    """Append-only JSONL repository for the Lab persistence seam.

    Every line is a complete, redacted receipt record.  Existing lines are
    loaded once and conflicting content-addressed keys fail closed.  The
    repository never truncates or rewrites prior records.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._rows: dict[str, ProviderReceipt] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = str(record.get("key") or "")
                    receipt = ProviderReceipt.from_mapping(record.get("receipt") or {})
                    if not key:
                        raise RoutingExperimentError("provider receipt repository key is empty")
                    validate_errors = validate_provider_receipt(receipt)
                    if validate_errors:
                        raise RoutingExperimentError(
                            "provider receipt repository record invalid: " + "; ".join(validate_errors)
                        )
                    existing = self._rows.get(key)
                    if existing is not None and existing.to_dict() != receipt.to_dict():
                        raise RoutingExperimentError("provider receipt key collision")
                    self._rows[key] = receipt

    def get(self, key: str) -> ProviderReceipt | None:
        return self._rows.get(str(key))

    def append(self, key: str, receipt: ProviderReceipt) -> ProviderReceipt:
        key = str(key)
        existing = self._rows.get(key)
        if existing is not None:
            if existing.to_dict() != receipt.to_dict():
                raise RoutingExperimentError("provider receipt key collision")
            return existing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "receipt": receipt.to_dict()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[key] = receipt
        return receipt

    def keys(self) -> Iterable[str]:
        return tuple(self._rows)


class ProviderReceiptStore:
    """Append-only, content-addressed receipt store over the Lab repository seam."""

    def __init__(self, repository: ProviderReceiptRepository | None = None) -> None:
        self.repository = repository or InMemoryProviderReceiptRepository()
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, key: str) -> ProviderReceipt | None:
        receipt = self.repository.get(str(key))
        if receipt is None:
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return receipt

    def put(self, key: str, receipt: ProviderReceipt | Mapping[str, Any]) -> ProviderReceipt:
        normalized = receipt if isinstance(receipt, ProviderReceipt) else ProviderReceipt.from_mapping(receipt)
        errors = validate_provider_receipt(normalized)
        if errors:
            raise RoutingExperimentError("provider receipt invalid: " + "; ".join(errors))
        return self.repository.append(str(key), normalized)

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(receipt.receipt_ref for key in self.repository.keys() if (receipt := self.repository.get(key))))

    def refs_for_keys(self, keys: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.repository.get(key).receipt_ref
                for key in keys
                if self.repository.get(key) is not None
            )
        )

    def __len__(self) -> int:
        return sum(1 for _ in self.repository.keys())


@dataclass(frozen=True)
class RoutingEvaluationMetrics:
    split: str
    unit_count: int
    predicted_positive_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    verified_positive_count: int
    rejected_count: int
    source_miss_count: int
    adapter_failure_count: int
    total_credit_microunits: int
    no_signal_credit_microunits: int
    unique_rescue_count: int
    unique_rescue_credit_microunits: int
    marginal_verified_positives_per_credit: float
    precision: float
    recall: float
    mean_latency_ms: float
    source_lineage_overlap_count: int
    source_lineage_overlap_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VariantEvaluation:
    variant_id: str
    calibration: RoutingEvaluationMetrics
    holdout: RoutingEvaluationMetrics
    passed_precision_gate: bool
    passed_recall_gate: bool
    passed_cost_gate: bool
    passed_efficiency_gate: bool
    passed: bool
    receipt_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["calibration"] = self.calibration.to_dict()
        data["holdout"] = self.holdout.to_dict()
        data["receipt_refs"] = list(self.receipt_refs)
        return data


@dataclass(frozen=True)
class RoutingEvaluationReceipt:
    receipt_id: str
    experiment_id: str
    experiment_hash: str
    artifact_hash: str
    manifest_hash: str
    feature_set_hash: str
    calibration_unit_count: int
    holdout_unit_count: int
    baseline_profile_id: str
    variants: tuple[VariantEvaluation, ...]
    selected_profile_id: str
    provider_receipt_refs: tuple[str, ...]
    provider_cache_hits: int
    provider_cache_misses: int
    live_credit_spend: bool = False
    immutable: bool = True
    contract_version: str = ROUTING_EVALUATION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "receipt_id": self.receipt_id,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "artifact_hash": self.artifact_hash,
            "manifest_hash": self.manifest_hash,
            "feature_set_hash": self.feature_set_hash,
            "calibration_unit_count": self.calibration_unit_count,
            "holdout_unit_count": self.holdout_unit_count,
            "baseline_profile_id": self.baseline_profile_id,
            "variants": [variant.to_dict() for variant in self.variants],
            "selected_profile_id": self.selected_profile_id,
            "provider_receipt_refs": list(self.provider_receipt_refs),
            "provider_cache_hits": self.provider_cache_hits,
            "provider_cache_misses": self.provider_cache_misses,
            "live_credit_spend": self.live_credit_spend,
            "immutable": self.immutable,
        }


def _validate_gold_labels(frozen: FrozenRoutingInput, gold_labels: Mapping[str, bool]) -> list[str]:
    all_units = set(frozen.calibration_unit_refs).union(frozen.holdout_unit_refs)
    if set(gold_labels) != all_units:
        return ["gold_labels_must_cover_exact_calibration_and_holdout_units"]
    normalized = {str(key): bool(value) for key, value in gold_labels.items()}
    expected_hash = sha256_json({"labels": sorted(normalized.items())})
    if expected_hash != frozen.gold_label_set_hash:
        return ["gold_label_set_hash_mismatch"]
    return []


def _receipt_for_runner_result(
    value: ProviderReceipt | Mapping[str, Any],
    *,
    binding: ProviderBindingIdentity,
    unit_ref: str,
    request_fingerprint: str,
) -> ProviderReceipt:
    if isinstance(value, ProviderReceipt):
        receipt = value
    else:
        receipt = ProviderReceipt.from_mapping(value)
    if receipt.binding_id != binding.binding_id or receipt.tool_id != binding.tool_id:
        raise RoutingExperimentError("runner receipt binding identity mismatch")
    if receipt.source_lineage_id != binding.source_lineage_id:
        raise RoutingExperimentError("runner receipt source lineage mismatch")
    if receipt.binding_version != binding.adapter_version:
        raise RoutingExperimentError("runner receipt binding version mismatch")
    if receipt.unit_ref != unit_ref or receipt.request_fingerprint != request_fingerprint:
        raise RoutingExperimentError("runner receipt request identity mismatch")
    if receipt.execution_mode not in {mode.value for mode in ReceiptExecutionMode}:
        raise RoutingExperimentError("runner returned an unsupported execution mode")
    return receipt


def _run_variant_unit(
    variant: LabRoutingProfile,
    *,
    adapter: RoutingAdmissionPlanAdapter,
    unit_ref: str,
    bindings: Mapping[str, ProviderBindingIdentity],
    receipt_store: ProviderReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
) -> tuple[list[ProviderReceipt], bool]:
    receipts: list[ProviderReceipt] = []
    plan = _model_profile(variant, adapter)
    for tool_id in tuple(adapter.ordered_tool_ids(plan))[:MAX_TOOLS_PER_VARIANT]:
        binding = bindings[tool_id]
        request_fingerprint = sha256_json(
            {
                "experiment_request": "intent-route-v1",
                "tool_id": binding.tool_id,
                "unit_ref": unit_ref,
            }
        )
        key = provider_receipt_key(
            binding_id=binding.binding_id,
            tool_id=binding.tool_id,
            binding_version=binding.adapter_version,
            request_fingerprint=request_fingerprint,
        )
        receipt = receipt_store.get(key)
        if receipt is None:
            receipt = _receipt_for_runner_result(
                runner(binding, unit_ref),
                binding=binding,
                unit_ref=unit_ref,
                request_fingerprint=request_fingerprint,
            )
            identity_payload = receipt.to_dict()
            identity_payload.pop("receipt_ref", None)
            expected_ref = "provider_receipt:" + sha256_json(identity_payload).split(":", 1)[1][:16]
            if receipt.receipt_ref != expected_ref:
                raise RoutingExperimentError("runner receipt_ref is not content addressed")
            receipt_store.put(key, receipt)
        receipts.append(receipt)
        if receipt.outcome == ProviderOutcome.VERIFIED.value:
            return receipts, True
    return receipts, False


def _metrics_for_split(
    *,
    split: str,
    unit_refs: Sequence[str],
    gold_labels: Mapping[str, bool],
    variant: LabRoutingProfile,
    adapter: RoutingAdmissionPlanAdapter,
    bindings: Mapping[str, ProviderBindingIdentity],
    receipt_store: ProviderReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    baseline_positive_units: set[str],
) -> RoutingEvaluationMetrics:
    predicted_positive_count = 0
    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    verified_positive_count = 0
    rejected_count = 0
    source_miss_count = 0
    adapter_failure_count = 0
    total_credit = 0
    no_signal_credit = 0
    unique_rescue_count = 0
    unique_rescue_credit = 0
    total_latency = 0
    total_receipts = 0
    overlap_count = 0
    receipt_refs: list[str] = []

    for unit_ref in unit_refs:
        receipts, predicted = _run_variant_unit(
            variant,
            unit_ref=unit_ref,
            adapter=adapter,
            bindings=bindings,
            receipt_store=receipt_store,
            runner=runner,
        )
        receipt_refs.extend(item.receipt_ref for item in receipts)
        lineages = [item.source_lineage_id for item in receipts]
        overlap_count += len(lineages) - len(set(lineages))
        total_receipts += len(receipts)
        total_credit += sum(item.credit_microunits for item in receipts)
        total_latency += sum(item.latency_ms for item in receipts)
        for receipt in receipts:
            if receipt.outcome == ProviderOutcome.VERIFIED.value:
                verified_positive_count += 1
            elif receipt.outcome == ProviderOutcome.REJECTED.value:
                rejected_count += 1
            elif receipt.outcome == ProviderOutcome.SOURCE_MISS.value:
                source_miss_count += 1
            elif receipt.outcome == ProviderOutcome.ADAPTER_FAILURE.value:
                adapter_failure_count += 1
        expected_positive = bool(gold_labels[unit_ref])
        if predicted:
            predicted_positive_count += 1
            if expected_positive:
                true_positive_count += 1
            else:
                false_positive_count += 1
            if unit_ref not in baseline_positive_units:
                unique_rescue_count += 1
                unique_rescue_credit += sum(item.credit_microunits for item in receipts)
        elif expected_positive:
            false_negative_count += 1
        if not predicted:
            no_signal_credit += sum(item.credit_microunits for item in receipts)

    unit_count = len(unit_refs)
    precision = true_positive_count / predicted_positive_count if predicted_positive_count else 1.0
    recall = true_positive_count / sum(1 for ref in unit_refs if gold_labels[ref]) if any(gold_labels[ref] for ref in unit_refs) else 1.0
    marginal = true_positive_count / total_credit if total_credit else 0.0
    return RoutingEvaluationMetrics(
        split=split,
        unit_count=unit_count,
        predicted_positive_count=predicted_positive_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        verified_positive_count=verified_positive_count,
        rejected_count=rejected_count,
        source_miss_count=source_miss_count,
        adapter_failure_count=adapter_failure_count,
        total_credit_microunits=total_credit,
        no_signal_credit_microunits=no_signal_credit,
        unique_rescue_count=unique_rescue_count,
        unique_rescue_credit_microunits=unique_rescue_credit,
        marginal_verified_positives_per_credit=round(marginal, 12),
        precision=round(precision, 8),
        recall=round(recall, 8),
        mean_latency_ms=round(total_latency / total_receipts, 8) if total_receipts else 0.0,
        source_lineage_overlap_count=overlap_count,
        source_lineage_overlap_rate=round(overlap_count / total_receipts, 8) if total_receipts else 0.0,
    )


def _evaluate_variant(
    spec: RoutingExperimentSpec,
    variant: LabRoutingProfile,
    adapter: RoutingAdmissionPlanAdapter,
    *,
    gold_labels: Mapping[str, bool],
    receipt_store: ProviderReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    baseline_positive_by_split: Mapping[str, set[str]],
) -> VariantEvaluation:
    bindings = {item.tool_id: item for item in spec.provider_bindings}
    calibration = _metrics_for_split(
        split="calibration",
        unit_refs=spec.frozen_input.calibration_unit_refs,
        gold_labels=gold_labels,
        variant=variant,
        adapter=adapter,
        bindings=bindings,
        receipt_store=receipt_store,
        runner=runner,
        baseline_positive_units=baseline_positive_by_split.get("calibration", set()),
    )
    holdout = _metrics_for_split(
        split="holdout",
        unit_refs=spec.frozen_input.holdout_unit_refs,
        gold_labels=gold_labels,
        variant=variant,
        adapter=adapter,
        bindings=bindings,
        receipt_store=receipt_store,
        runner=runner,
        baseline_positive_units=baseline_positive_by_split.get("holdout", set()),
    )
    gates = spec.gates
    passed_precision = calibration.precision >= gates.min_calibration_precision and holdout.precision >= gates.min_holdout_precision
    passed_recall = holdout.recall >= gates.min_holdout_recall
    passed_cost = holdout.no_signal_credit_microunits <= gates.max_holdout_no_signal_credit_microunits
    passed_efficiency = holdout.marginal_verified_positives_per_credit >= gates.min_marginal_verified_positives_per_credit
    variant_receipt_keys = []
    for unit_ref in (
        *spec.frozen_input.calibration_unit_refs,
        *spec.frozen_input.holdout_unit_refs,
    ):
        plan = _model_profile(variant, adapter)
        for tool_id in tuple(adapter.ordered_tool_ids(plan))[:MAX_TOOLS_PER_VARIANT]:
            binding = bindings[tool_id]
            request_fingerprint = sha256_json(
                {
                    "experiment_request": "intent-route-v1",
                    "tool_id": binding.tool_id,
                    "unit_ref": unit_ref,
                }
            )
            variant_receipt_keys.append(
                provider_receipt_key(
                    binding_id=binding.binding_id,
                    tool_id=binding.tool_id,
                    binding_version=binding.adapter_version,
                    request_fingerprint=request_fingerprint,
                )
            )
    return VariantEvaluation(
        variant_id=_profile_id(variant, adapter),
        calibration=calibration,
        holdout=holdout,
        passed_precision_gate=passed_precision,
        passed_recall_gate=passed_recall,
        passed_cost_gate=passed_cost,
        passed_efficiency_gate=passed_efficiency,
        passed=passed_precision and passed_recall and passed_cost and passed_efficiency,
        receipt_refs=receipt_store.refs_for_keys(variant_receipt_keys),
    )


def select_smallest_passing_variant(
    evaluations: Sequence[VariantEvaluation],
) -> str:
    """Select the smallest dependency-closed route on the measured frontier."""

    passing = [item for item in evaluations if item.passed]
    if not passing:
        return ""
    selected = sorted(
        passing,
        key=lambda item: (
            len(item.receipt_refs),
            item.holdout.total_credit_microunits,
            -item.holdout.unique_rescue_count,
            -item.holdout.recall,
            item.variant_id,
        ),
    )[0]
    return selected.variant_id


def evaluate_routing_experiment(
    spec: RoutingExperimentSpec | Mapping[str, Any],
    *,
    gold_labels: Mapping[str, bool],
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    adapter: RoutingAdmissionPlanAdapter,
    receipt_store: ProviderReceiptStore | None = None,
    authoritative_billing_rollup: Callable[[ProviderReceiptStore], Mapping[str, Any]] | None = None,
) -> RoutingEvaluationReceipt:
    """Run all bounded variants against shared units and build an immutable receipt.

    The runner is an injected Lab adapter.  It must return a typed,
    credential-free receipt.  No network call is made by this function.
    """

    if not isinstance(spec, RoutingExperimentSpec):
        spec = RoutingExperimentSpec.from_mapping(spec)
    errors = validate_routing_experiment_spec(spec, adapter=adapter)
    if errors:
        raise RoutingExperimentError("invalid routing experiment: " + "; ".join(errors))
    label_errors = _validate_gold_labels(spec.frozen_input, gold_labels)
    if label_errors:
        raise RoutingExperimentError("invalid gold labels: " + "; ".join(label_errors))
    store = receipt_store or ProviderReceiptStore()
    baseline = next(
        item for item in spec.variants if _profile_id(item, adapter) == spec.baseline_profile_id
    )
    bindings = {item.tool_id: item for item in spec.provider_bindings}
    baseline_positive_by_split: dict[str, set[str]] = {"calibration": set(), "holdout": set()}
    for split, refs in (
        ("calibration", spec.frozen_input.calibration_unit_refs),
        ("holdout", spec.frozen_input.holdout_unit_refs),
    ):
        for unit_ref in refs:
            receipts, predicted = _run_variant_unit(
            baseline,
            unit_ref=unit_ref,
            adapter=adapter,
                bindings=bindings,
                receipt_store=store,
                runner=runner,
            )
            if predicted:
                baseline_positive_by_split[split].add(unit_ref)
    evaluations = tuple(
        _evaluate_variant(
            spec,
            variant,
            adapter=adapter,
            gold_labels=gold_labels,
            receipt_store=store,
            runner=runner,
            baseline_positive_by_split=baseline_positive_by_split,
        )
        for variant in spec.variants
    )
    selected_profile_id = select_smallest_passing_variant(evaluations)
    billing_rollup = authoritative_billing_rollup(store) if authoritative_billing_rollup else None
    stored_receipts = [
        receipt
        for key in store.repository.keys()
        if (receipt := store.repository.get(key)) is not None
    ]
    if spec.allow_live_credit_spend:
        if any(item.execution_mode != ReceiptExecutionMode.MEASURED_LAB.value for item in stored_receipts):
            raise RoutingExperimentError("measured_lab_requires_measured_lab_receipts")
        observed_total = sum(item.credit_microunits for item in stored_receipts)
        if observed_total > spec.credit_budget.total_credit_microunits:
            raise RoutingExperimentError("measured_lab_total_credit_cap_exceeded")
        observed_by_binding: dict[str, int] = {}
        for item in stored_receipts:
            observed_by_binding[item.binding_id] = observed_by_binding.get(item.binding_id, 0) + item.credit_microunits
        for binding_id, ceiling in spec.credit_budget.provider_credit_ceilings.items():
            if observed_by_binding.get(binding_id, 0) > ceiling:
                raise RoutingExperimentError(f"measured_lab_provider_credit_cap_exceeded:{binding_id}")
    elif any(item.execution_mode == ReceiptExecutionMode.MEASURED_LAB.value for item in stored_receipts):
        raise RoutingExperimentError("measured_lab_receipt_requires_explicit_credit_cap")
    if spec.allow_live_credit_spend:
        if authoritative_billing_rollup is None:
            raise RoutingExperimentError("measured_lab_requires_authoritative_billing_rollup")
        if not isinstance(billing_rollup, Mapping):
            raise RoutingExperimentError("authoritative_billing_rollup_must_be_an_object")
        billed_total = int(billing_rollup.get("total_credit_microunits", -1))
        observed_total = sum(item.credit_microunits for item in stored_receipts)
        if billed_total != observed_total:
            raise RoutingExperimentError("authoritative_billing_delta_mismatch")
    draft = RoutingEvaluationReceipt(
        receipt_id="routing_evaluation:pending",
        experiment_id=spec.experiment_id,
        experiment_hash=spec.experiment_hash(),
        artifact_hash=spec.artifact.model_artifact_hash,
        manifest_hash=spec.artifact.manifest_hash,
        feature_set_hash=spec.frozen_input.feature_set_hash,
        calibration_unit_count=len(spec.frozen_input.calibration_unit_refs),
        holdout_unit_count=len(spec.frozen_input.holdout_unit_refs),
        baseline_profile_id=spec.baseline_profile_id,
        variants=evaluations,
        selected_profile_id=selected_profile_id,
        provider_receipt_refs=store.refs(),
        provider_cache_hits=store.cache_hits,
        provider_cache_misses=store.cache_misses,
        live_credit_spend=spec.allow_live_credit_spend,
    )
    payload = draft.to_dict()
    payload["receipt_id"] = "routing_evaluation:" + sha256_json(payload).split(":", 1)[1][:16]
    return RoutingEvaluationReceipt(**{**draft.__dict__, "receipt_id": payload["receipt_id"]})


@dataclass(frozen=True)
class RoutingPromotionReceipt:
    receipt_id: str
    experiment_id: str
    evaluation_receipt_id: str
    evaluation_receipt_hash: str
    selected_profile_id: str
    profile_hash: str
    artifact_hash: str
    manifest_hash: str
    experiment_hash: str
    feature_set_hash: str
    routing_policy_hash: str
    target_branch: str
    lifecycle: str
    production_activation: bool = False
    immutable: bool = True
    contract_version: str = ROUTING_PROMOTION_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def promote_routing_profile_to_lab(
    spec: RoutingExperimentSpec | Mapping[str, Any],
    evaluation: RoutingEvaluationReceipt | Mapping[str, Any],
    *,
    adapter: RoutingAdmissionPlanAdapter,
) -> RoutingPromotionReceipt:
    """Create a Lab-only promotion receipt after precision-first gates pass."""

    if not isinstance(spec, RoutingExperimentSpec):
        spec = RoutingExperimentSpec.from_mapping(spec)
    errors = validate_routing_experiment_spec(spec, adapter=adapter)
    if errors:
        raise RoutingExperimentError("invalid routing experiment: " + "; ".join(errors))
    if not isinstance(evaluation, RoutingEvaluationReceipt):
        raise RoutingExperimentError("evaluation mapping must be parsed by the caller")
    if evaluation.live_credit_spend or not evaluation.immutable:
        raise RoutingExperimentError("evaluation receipt is not immutable Lab evidence")
    if evaluation.experiment_id != spec.experiment_id or evaluation.experiment_hash != spec.experiment_hash():
        raise RoutingExperimentError("evaluation receipt experiment identity mismatch")
    if evaluation.artifact_hash != spec.artifact.model_artifact_hash or evaluation.manifest_hash != spec.artifact.manifest_hash:
        raise RoutingExperimentError("evaluation receipt artifact identity mismatch")
    if not evaluation.selected_profile_id:
        raise RoutingExperimentError("no passing profile is eligible for promotion")
    selected = next((item for item in evaluation.variants if item.variant_id == evaluation.selected_profile_id), None)
    if selected is None or not selected.passed:
        raise RoutingExperimentError("selected profile did not pass all gates")
    variant = next(
        item for item in spec.variants if _profile_id(item, adapter) == selected.variant_id
    )
    draft = RoutingPromotionReceipt(
        receipt_id="routing_promotion:pending",
        experiment_id=spec.experiment_id,
        evaluation_receipt_id=evaluation.receipt_id,
        evaluation_receipt_hash=sha256_json(evaluation.to_dict()),
        selected_profile_id=_profile_id(variant, adapter),
        profile_hash=_profile_hash(variant, adapter),
        artifact_hash=spec.artifact.model_artifact_hash,
        manifest_hash=spec.artifact.manifest_hash,
        experiment_hash=spec.experiment_hash(),
        feature_set_hash=spec.frozen_input.feature_set_hash,
        routing_policy_hash=spec.artifact.routing_policy_hash,
        target_branch="leadpoet-lab",
        lifecycle=ProfileLifecycle.APPROVED.value,
        production_activation=False,
    )
    payload = draft.to_dict()
    receipt_id = "routing_promotion:" + sha256_json(payload).split(":", 1)[1][:16]
    return replace(draft, receipt_id=receipt_id)


def verify_lab_routing_artifact_lineage(
    *,
    spec: RoutingExperimentSpec,
    promotion: RoutingPromotionReceipt,
) -> list[str]:
    """Verify that a promotion receipt is tied to one exact Lab artifact."""

    errors: list[str] = []
    if promotion.target_branch != "leadpoet-lab":
        errors.append("promotion_target_branch_must_be_leadpoet_lab")
    if promotion.production_activation:
        errors.append("promotion_must_not_activate_production")
    if promotion.artifact_hash != spec.artifact.model_artifact_hash:
        errors.append("promotion_artifact_hash_mismatch")
    if promotion.manifest_hash != spec.artifact.manifest_hash:
        errors.append("promotion_manifest_hash_mismatch")
    if promotion.experiment_hash != spec.experiment_hash():
        errors.append("promotion_experiment_hash_mismatch")
    if promotion.feature_set_hash != spec.frozen_input.feature_set_hash:
        errors.append("promotion_feature_set_hash_mismatch")
    if promotion.routing_policy_hash != spec.artifact.routing_policy_hash:
        errors.append("promotion_routing_policy_hash_mismatch")
    if promotion.experiment_id != spec.experiment_id:
        errors.append("promotion_experiment_id_mismatch")
    if promotion.lifecycle not in {ProfileLifecycle.APPROVED.value, ProfileLifecycle.PUBLISHED.value}:
        errors.append("promotion_lifecycle_is_invalid")
    return sorted(set(errors))


__all__ = [
    "MAX_CREDIT_MICROUNITS_PER_VARIANT",
    "MAX_PROFILE_VARIANTS",
    "ProviderBindingIdentity",
    "ProviderOutcome",
    "ProviderReceipt",
    "ProviderReceiptRepository",
    "ProviderReceiptStore",
    "InMemoryProviderReceiptRepository",
    "JsonlProviderReceiptRepository",
    "ProposalState",
    "ProfileLifecycle",
    "ReceiptExecutionMode",
    "ExperimentCreditBudget",
    "RoutingEvaluationGates",
    "RoutingEvaluationMetrics",
    "RoutingEvaluationReceipt",
    "RoutingExperimentError",
    "RoutingExperimentSpec",
    "LabRoutingProfile",
    "RoutingAdmissionPlanAdapter",
    "RoutingPromotionReceipt",
    "FrozenRoutingInput",
    "LLMRoutingProfileProposal",
    "SourcingModelArtifactIdentity",
    "VariantEvaluation",
    "admit_llm_routing_profile_proposal",
    "evaluate_routing_experiment",
    "promote_routing_profile_to_lab",
    "provider_receipt_key",
    "select_smallest_passing_variant",
    "validate_frozen_routing_input",
    "validate_llm_routing_profile_proposal",
    "validate_model_routing_profile",
    "validate_provider_binding_identity",
    "validate_provider_receipt",
    "validate_routing_evaluation_gates",
    "validate_routing_experiment_spec",
    "validate_experiment_credit_budget",
    "validate_sourcing_model_artifact_identity",
    "verify_lab_routing_artifact_lineage",
    "model_hash_to_lab",
    "lab_hash_to_model",
]
