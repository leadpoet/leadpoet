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
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import re
import subprocess
import threading
import time
import types
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .canonical import sha256_json


ROUTING_EXPERIMENT_CONTRACT_VERSION = "leadpoet.intent_routing_experiment:v1"
ROUTING_EVALUATION_RECEIPT_VERSION = "leadpoet.intent_routing_evaluation:v1"
ROUTING_PROMOTION_RECEIPT_VERSION = "leadpoet.intent_routing_promotion:v1"
ROUTING_EXPERIMENT_V2_CONTRACT_VERSION = "leadpoet.intent_routing_experiment:v2"
ROUTING_DECISION_RECEIPT_V2_VERSION = "leadpoet.routing_decision_receipt:v2"

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
_MODEL_EXECUTION_MODES = frozenset({"invoke", "observe", "virtual"})
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

ROUTING_EXPERIMENT_STAGES = frozenset({"candidate_acquisition", "intent_evidence"})


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


@dataclass(frozen=True)
class RoutingPlanStepBudget:
    """Model-owned route-step budget projected for Lab accounting only.

    The Lab does not use this value to choose or order tools.  The adapter
    projects the exact model plan into this summary so the Lab can reserve
    worst-case calls, time, and credits before invoking a provider.
    """

    tool_id: str
    execution_mode: str
    max_calls: int
    timeout_seconds: float
    credit_microunits: int

    def __post_init__(self) -> None:
        _ensure_safe_ref(self.tool_id, "route_step_tool_id")
        if self.execution_mode not in _MODEL_EXECUTION_MODES:
            raise RoutingExperimentError("route_step_execution_mode_is_invalid")
        _bounded_int(self.max_calls, "route_step_max_calls", minimum=0, maximum=MAX_TOOLS_PER_VARIANT)
        _bounded_float(self.timeout_seconds, "route_step_timeout_seconds", minimum=0.0, maximum=MAX_LATENCY_MS / 1000)
        _bounded_int(
            self.credit_microunits,
            "route_step_credit_microunits",
            minimum=0,
            maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT,
        )


@dataclass(frozen=True)
class RoutingCallAuthorization:
    """Redacted per-call authorization for measured provider runners."""

    experiment_id: str
    variant_id: str
    artifact_key: str
    stage: str
    unit_ref: str
    tool_id: str
    attempt: int
    request_fingerprint: str
    remaining_credit_microunits: int
    timeout_ceiling_ms: int
    execution_mode: str
    contract_version: str = "leadpoet.routing_call_authz:v1"

    def __post_init__(self) -> None:
        _ensure_safe_ref(self.experiment_id, "routing_authorization_experiment_id")
        _ensure_safe_ref(self.variant_id, "routing_authorization_variant_id")
        _ensure_hash(self.artifact_key, "routing_authorization_artifact_key")
        _v2_safe_stage(self.stage)
        _ensure_safe_ref(self.unit_ref, "routing_authorization_unit_ref")
        _ensure_safe_ref(self.tool_id, "routing_authorization_tool_id")
        _bounded_int(self.attempt, "routing_authorization_attempt", minimum=0, maximum=MAX_TOOLS_PER_VARIANT)
        _ensure_hash(self.request_fingerprint, "routing_authorization_request_fingerprint")
        _bounded_int(
            self.remaining_credit_microunits,
            "routing_authorization_remaining_credit_microunits",
            minimum=0,
            maximum=MAX_CREDIT_MICROUNITS_PER_VARIANT,
        )
        _bounded_int(
            self.timeout_ceiling_ms,
            "routing_authorization_timeout_ceiling_ms",
            minimum=0,
            maximum=MAX_LATENCY_MS,
        )
        if self.execution_mode not in {mode.value for mode in ReceiptExecutionMode}:
            raise RoutingExperimentError("routing_authorization_execution_mode_is_invalid")
        if self.contract_version != "leadpoet.routing_call_authz:v1":
            raise RoutingExperimentError("routing_authorization_contract_version_is_invalid")
        _ensure_no_secret_material(self.to_dict(), field_name="routing_call_authorization")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    # Exact model-owned RoutingFeatureSet.as_payload().  The Lab never
    # reconstructs this document from segment metadata.
    feature_set_payload: Mapping[str, Any] = field(default_factory=dict)

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
            feature_set_payload=dict(data.get("feature_set_payload") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = list(self.features)
        data["calibration_unit_refs"] = list(self.calibration_unit_refs)
        data["holdout_unit_refs"] = list(self.holdout_unit_refs)
        data["feature_set_payload"] = dict(self.feature_set_payload)
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
    if not isinstance(frozen.feature_set_payload, Mapping):
        errors.append("feature_set_payload_must_be_an_object")
    elif not frozen.feature_set_payload:
        errors.append("feature_set_payload_is_required")
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
    fallback semantics.  It asks the model adapter to parse exact profile,
    feature-set, and admission-plan payloads, compile initial/challenger plans,
    and execute those plans through the model-owned route semantics.
    """

    def parse_profile(self, payload: Mapping[str, Any]) -> Any: ...

    def parse_feature_set(self, payload: Mapping[str, Any]) -> Any: ...

    def validate_feature_set(
        self,
        feature_set: Any,
        *,
        expected_hash: str,
        expected_features: Sequence[str],
    ) -> Sequence[str]: ...

    def validate_artifact_identity(
        self,
        artifact: SourcingModelArtifactIdentity,
    ) -> Sequence[str]: ...

    def validate_profile(
        self,
        profile: Any,
        *,
        signal_type: str,
        feature_set: Any,
        binding_tool_ids: frozenset[str],
        binding_source_lineages: Mapping[str, str],
    ) -> Sequence[str]: ...

    def profile_as_payload(self, profile: Any) -> Mapping[str, Any]: ...

    def profile_id(self, profile: Any) -> str: ...

    def profile_hash(self, profile: Any) -> str: ...

    def compile_initial(
        self,
        profile: Any,
        *,
        signal_type: str,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any: ...

    def compile_challenger(
        self,
        parent_plan: Any,
        *,
        profile: Any,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        attempted_tool_ids: Sequence[str],
        attempted_source_lineages: Sequence[str],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any: ...

    def has_conditional_confirmation(self, plan: Any) -> bool: ...

    def compile_confirmation(
        self,
        parent_plan: Any,
        *,
        profile: Any,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any: ...

    def execute_plan(
        self,
        plan: Any,
        invoke: Callable[[str], Any],
    ) -> tuple[Sequence[Any], bool]: ...

    def plan_as_payload(self, plan: Any) -> Mapping[str, Any]: ...

    def parse_plan(self, payload: Mapping[str, Any]) -> Any: ...

    def plan_hash(self, plan: Any) -> str: ...

    def plan_step_budgets(self, plan: Any) -> Sequence[RoutingPlanStepBudget]: ...

    def intent_release_policy_hash(self) -> str: ...


_MODEL_TREE_EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
}
_MODEL_TREE_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".env", ".pem", ".key")


def _model_tree_path_excluded(relative_path: str) -> bool:
    parts = relative_path.split("/")
    return (
        any(part in _MODEL_TREE_EXCLUDED_PARTS for part in parts)
        or relative_path.endswith(_MODEL_TREE_EXCLUDED_SUFFIXES)
        or relative_path == ".env"
        or relative_path.startswith(".env.")
    )


def _compute_model_source_tree_hash(root: Path) -> str:
    digest_inputs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if _model_tree_path_excluded(relative_path):
            continue
        digest_inputs.append(
            (
                relative_path,
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return sha256_json(digest_inputs)


def _observe_model_artifact_identity(
    root: Path,
    *,
    runtime: Any,
) -> Mapping[str, str]:
    """Observe identity fields without inventing unavailable manifest data."""

    observed: dict[str, str] = {
        "model_artifact_hash": _compute_model_source_tree_hash(root),
    }
    try:
        observed["commit_sha"] = (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        metadata = runtime.runtime_routing_metadata()
    except Exception:
        metadata = {}
    if isinstance(metadata, Mapping):
        for metadata_field, artifact_field in (
            ("catalog_sha256", "routing_catalog_hash"),
            ("policy_sha256", "routing_policy_hash"),
        ):
            value = str(metadata.get(metadata_field) or "").strip().lower()
            if _RAW_SHA256_RE.fullmatch(value):
                observed[artifact_field] = model_hash_to_lab(value, metadata_field)

    # A signed build manifest is external to the source checkout.  If a local
    # manifest is supplied by the artifact builder, bind every identity field
    # that it exposes; otherwise leave those fields absent rather than
    # fabricating a hash with a different model meaning.
    for manifest_path in (
        root / "research_lab_manifest.json",
        root / "artifact_manifest.json",
        root / "manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, Mapping):
            continue
        for manifest_field, artifact_field in (
            ("git_commit_sha", "commit_sha"),
            ("model_artifact_hash", "model_artifact_hash"),
            ("manifest_hash", "manifest_hash"),
            ("routing_contract_hash", "routing_contract_hash"),
            ("routing_catalog_hash", "routing_catalog_hash"),
            ("routing_policy_hash", "routing_policy_hash"),
            ("feature_schema_hash", "feature_schema_hash"),
            ("verifier_contract_hash", "verifier_contract_hash"),
        ):
            value = str(manifest.get(manifest_field) or "").strip().lower()
            if _SHA256_RE.fullmatch(value) or (
                artifact_field == "commit_sha" and _GIT_SHA_RE.fullmatch(value)
            ):
                observed[artifact_field] = value
        break
    return observed


class PinnedSourcingModelRoutingAdapter:
    """Concrete adapter for one exact Sourcing_model checkout/artifact.

    The loader is intentionally explicit and fail-closed.  Production wiring
    should pass the extracted, hash-verified ``leadpoet-lab`` model root; the
    Lab never falls back to a site-local routing implementation.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        profiles: Any,
        features: Any,
        candidate_profiles: Any | None = None,
        observed_artifact_identity: Mapping[str, str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.profiles = profiles
        self.features = features
        self.candidate_profiles = candidate_profiles
        self.catalog = runtime.runtime_catalog({})
        self.policy = runtime.runtime_policy()
        self._observed_artifact_identity = dict(observed_artifact_identity or {})

    @classmethod
    def from_model_root(cls, model_root: str | os.PathLike[str]) -> "PinnedSourcingModelRoutingAdapter":
        root = Path(model_root).resolve()
        if not (root / "sourcing_model" / "routing" / "runtime.py").is_file():
            raise RoutingExperimentError("pinned_sourcing_model_runtime_not_found")
        loaded = sys.modules.get("sourcing_model")
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file:
            try:
                loaded_root = Path(str(loaded_file)).resolve().parents[1]
            except (OSError, IndexError):
                loaded_root = None
            if loaded_root is not None and loaded_root != root:
                raise RoutingExperimentError(
                    "pinned_sourcing_model_module_already_loaded_from_other_root"
                )
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        # Do not replace already-loaded site packages.  If this process has
        # imported a different ``gateway``/``qualification`` dependency tree,
        # the model must be loaded in an isolated worker instead of allowing a
        # mixed tree with silently incompatible helper symbols.
        if loaded is None:
            # Import only the model routing package.  The checkout's top-level
            # ``sourcing_model.__init__`` imports credentialed application
            # helpers that are not needed by the pure routing contract and can
            # collide with the host's packages.  A package shell preserves the
            # exact source root while keeping this adapter side-effect free.
            package = types.ModuleType("sourcing_model")
            package.__file__ = str(root / "sourcing_model" / "__init__.py")
            package.__path__ = [str(root / "sourcing_model")]
            package.__package__ = "sourcing_model"
            sys.modules["sourcing_model"] = package
        importlib.invalidate_caches()
        try:
            runtime = importlib.import_module("sourcing_model.routing.runtime")
            profiles = importlib.import_module("sourcing_model.routing.profiles")
            features = importlib.import_module("sourcing_model.routing.features")
            try:
                candidate_profiles = importlib.import_module("sourcing_model.routing.candidate_profiles")
            except Exception:
                candidate_profiles = None
        except Exception as exc:
            raise RoutingExperimentError(f"pinned_sourcing_model_import_failed:{exc}") from exc
        return cls(
            runtime=runtime,
            profiles=profiles,
            features=features,
            candidate_profiles=candidate_profiles,
            observed_artifact_identity=_observe_model_artifact_identity(
                root,
                runtime=runtime,
            ),
        )

    def parse_profile(self, payload: Mapping[str, Any]) -> Any:
        return self.profiles.RoutingProfile.from_payload(payload)

    def parse_feature_set(self, payload: Mapping[str, Any]) -> Any:
        return self.features.RoutingFeatureSet.from_payload(payload)

    def validate_feature_set(
        self,
        feature_set: Any,
        *,
        expected_hash: str,
        expected_features: Sequence[str],
    ) -> Sequence[str]:
        errors: list[str] = []
        if feature_set.sha256() != expected_hash:
            errors.append("model_feature_set_hash_mismatch")
        if tuple(feature_set.features) != tuple(sorted(set(expected_features))):
            errors.append("model_feature_set_payload_features_mismatch")
        return errors

    def validate_artifact_identity(
        self,
        artifact: SourcingModelArtifactIdentity,
    ) -> Sequence[str]:
        errors: list[str] = []
        for artifact_field, observed_field in (
            ("commit_sha", "commit_sha"),
            ("model_artifact_hash", "model_artifact_hash"),
            ("manifest_hash", "manifest_hash"),
            ("routing_contract_hash", "routing_contract_hash"),
            ("routing_catalog_hash", "routing_catalog_hash"),
            ("routing_policy_hash", "routing_policy_hash"),
            ("feature_schema_hash", "feature_schema_hash"),
            ("verifier_contract_hash", "verifier_contract_hash"),
        ):
            observed = self._observed_artifact_identity.get(observed_field)
            if observed and str(getattr(artifact, artifact_field)) != observed:
                errors.append(f"model_artifact_{artifact_field}_mismatch")
        return errors

    def observed_artifact_identity(self) -> Mapping[str, str]:
        """Return hashes observed from the pinned checkout for test/admission."""

        return dict(self._observed_artifact_identity)

    def validate_profile(
        self,
        profile: Any,
        *,
        signal_type: str,
        feature_set: Any,
        binding_tool_ids: frozenset[str],
        binding_source_lineages: Mapping[str, str],
    ) -> Sequence[str]:
        errors: list[str] = []
        if profile.intent_category != str(signal_type or "").strip().upper():
            errors.append("model_profile_intent_category_mismatch")
        if not profile.matches(intent_category=profile.intent_category, features=feature_set):
            errors.append("model_profile_feature_predicate_mismatch")
        policy_tools = {step.tool_id for step in self.policy.steps_for(self.runtime.STAGE_INTENT_EVIDENCE)}
        for step in profile.steps:
            if step.tool_id not in binding_tool_ids:
                errors.append(f"model_profile_tool_not_bound:{step.tool_id}")
            elif binding_source_lineages.get(step.tool_id) != step.source_lineage_id:
                errors.append(f"model_profile_source_lineage_mismatch:{step.tool_id}")
            if step.tool_id not in policy_tools:
                errors.append(f"model_profile_tool_not_in_policy:{step.tool_id}")
            binding = next((item for item in self.catalog.tools if item.tool_id == step.tool_id), None)
            if binding is None or self.runtime.STAGE_INTENT_EVIDENCE not in binding.stages:
                errors.append(f"model_profile_tool_not_intent_capable:{step.tool_id}")
        return errors

    def profile_as_payload(self, profile: Any) -> Mapping[str, Any]:
        return profile.as_payload()

    def profile_id(self, profile: Any) -> str:
        return profile.profile_id

    def profile_hash(self, profile: Any) -> str:
        return profile.sha256()

    def _registry(self, profile: Any) -> Any:
        return self.profiles.RoutingProfileRegistry(profiles=(profile,))

    def compile_initial(
        self,
        profile: Any,
        *,
        signal_type: str,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any:
        return self.runtime.compile_intent_evidence_admission_route(
            str(signal_type or "").strip().upper(),
            feature_set=feature_set,
            existing_evidence=False,
            available_tools=available_tools,
            remaining_seconds=remaining_seconds,
            remaining_calls=remaining_calls,
            credit_cap=credit_cap,
            catalog=self.runtime.runtime_catalog(available_tools),
            policy=self.runtime.runtime_policy(),
            profile_registry=self._registry(profile),
            profile_id=profile.profile_id,
            expected_profile_sha256=profile.sha256(),
            expected_profile_version=profile.version,
        )

    def compile_challenger(
        self,
        parent_plan: Any,
        *,
        profile: Any,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        attempted_tool_ids: Sequence[str],
        attempted_source_lineages: Sequence[str],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any:
        return self.runtime.compile_challenger_admission_plan(
            parent_plan=parent_plan,
            catalog=self.runtime.runtime_catalog(available_tools),
            policy=self.runtime.runtime_policy(),
            context=self.runtime.RouteContext(
                stage=self.runtime.STAGE_INTENT_EVIDENCE,
                features=tuple(feature_set.features),
                intent_category=parent_plan.intent_category,
                remaining_seconds=remaining_seconds,
                remaining_calls=remaining_calls,
                remaining_results=1,
                credit_cap=credit_cap,
            ),
            feature_set=feature_set,
            registry=self._registry(profile),
            attempted_tool_ids=tuple(attempted_tool_ids),
            attempted_source_lineages=tuple(attempted_source_lineages),
        )

    def has_conditional_confirmation(self, plan: Any) -> bool:
        return bool(plan.conditional_tool_ids)

    def compile_confirmation(
        self,
        parent_plan: Any,
        *,
        profile: Any,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
    ) -> Any:
        return self.profiles.compile_confirmation_admission_plan(
            parent_plan=parent_plan,
            catalog=self.runtime.runtime_catalog(available_tools),
            policy=self.runtime.runtime_policy(),
            context=self.runtime.RouteContext(
                stage=self.runtime.STAGE_INTENT_EVIDENCE,
                features=tuple(feature_set.features),
                intent_category=parent_plan.intent_category,
                remaining_seconds=remaining_seconds,
                remaining_calls=remaining_calls,
                remaining_results=1,
                credit_cap=credit_cap,
            ),
            feature_set=feature_set,
            registry=self._registry(profile),
        )

    def execute_plan(self, plan: Any, invoke: Callable[[str], Any]) -> tuple[Sequence[Any], bool]:
        results: list[Any] = []
        predicted = False
        for step in plan.route.steps:
            if step.execution_mode != self.runtime.EXECUTION_INVOKE:
                continue
            for _ in range(step.max_calls):
                result = invoke(step.tool_id)
                results.append(result)
                outcome = str(getattr(result, "outcome", "") or (result.get("outcome") if isinstance(result, Mapping) else ""))
                if outcome == ProviderOutcome.VERIFIED.value:
                    predicted = True
                    if step.stop_on_success:
                        return results, predicted
        return results, predicted

    def plan_as_payload(self, plan: Any) -> Mapping[str, Any]:
        return plan.as_payload()

    def parse_plan(self, payload: Mapping[str, Any]) -> Any:
        if str(payload.get("schema_version") or "") == "candidate-routing-plan:v1":
            if self.candidate_profiles is None:
                raise RoutingExperimentError("candidate_routing_plan_contract_unavailable")
            return self.candidate_profiles.CandidateRoutingPlan.from_payload(payload)
        return self.profiles.RoutingAdmissionPlan.from_payload(payload)

    def plan_hash(self, plan: Any) -> str:
        return plan.sha256()

    def route_hash(self, plan: Any) -> str:
        route = getattr(plan, "route", plan)
        getter = getattr(route, "sha256", None)
        if not callable(getter):
            raise RoutingExperimentError("model_route_hash_unavailable")
        return getter()

    def plan_step_budgets(self, plan: Any) -> Sequence[RoutingPlanStepBudget]:
        return tuple(
            RoutingPlanStepBudget(
                tool_id=step.tool_id,
                execution_mode=step.execution_mode,
                max_calls=step.max_calls,
                timeout_seconds=step.timeout_seconds,
                credit_microunits=int(math.ceil(step.credit_cap * 1_000_000)),
            )
            for step in plan.route.steps
        )

    # V2 adapter seam.  Intent uses the exact existing RoutingProfile.  The
    # candidate branch is deliberately fail-closed until the model exports
    # its follow-on CandidateRoutingProfile contract and compiler; Lab code
    # does not synthesize a candidate profile or route schema.
    def parse_variant_payload(self, payload: Mapping[str, Any], *, stage: str) -> Any:
        if stage == self.runtime.STAGE_INTENT_EVIDENCE:
            return self.parse_profile(payload)
        profile_type = getattr(self.candidate_profiles, "CandidateRoutingProfile", None) if self.candidate_profiles is not None else None
        if profile_type is None or not callable(getattr(profile_type, "from_payload", None)):
            raise RoutingExperimentError("candidate_routing_profile_contract_unavailable")
        return profile_type.from_payload(payload)

    def validate_variant_payload(
        self,
        payload: Any,
        *,
        stage: str,
        feature_set: Any,
        binding_tool_ids: frozenset[str],
        binding_source_lineages: Mapping[str, str],
        expected_signal_type: str = "",
    ) -> Sequence[str]:
        if stage == self.runtime.STAGE_INTENT_EVIDENCE:
            return self.validate_profile(
                payload,
                signal_type=expected_signal_type,
                feature_set=feature_set,
                binding_tool_ids=binding_tool_ids,
                binding_source_lineages=binding_source_lineages,
            )
        if str(expected_signal_type or "").strip():
            raise RoutingExperimentError("candidate_signal_type_must_be_empty")
        errors: list[str] = []
        if not payload.matches(feature_set):
            errors.append("candidate_profile_feature_predicate_mismatch")
        for step in payload.steps:
            if step.tool_id not in binding_tool_ids:
                errors.append(f"candidate_profile_tool_not_bound:{step.tool_id}")
        return errors

    def routing_change_class(self, payload: Any, *, stage: str) -> str:
        """Use the exact model profile type to classify default versus custom routing."""

        if stage not in {
            self.runtime.STAGE_INTENT_EVIDENCE,
            self.runtime.STAGE_CANDIDATE_ACQUISITION,
        }:
            raise RoutingExperimentError("v2_routing_change_stage_is_invalid")
        return "default" if bool(payload.is_default) else "custom"

    def routing_identity(
        self,
        payload: Any,
        *,
        stage: str,
        exclude_tool_ids: Sequence[str] = (),
    ) -> str:
        """Hash the exact typed model profile, optionally omitting declared tools.

        The Lab does not define a route schema.  It asks the pinned model type
        for its canonical payload and uses the model's own payload hash helper;
        the only normalization allowed here is removing explicitly declared
        SourceAdd tool steps for a tool-only comparison.
        """

        if stage not in {
            self.runtime.STAGE_INTENT_EVIDENCE,
            self.runtime.STAGE_CANDIDATE_ACQUISITION,
        }:
            raise RoutingExperimentError("v2_routing_identity_stage_is_invalid")
        excluded = frozenset(_ensure_safe_ref(item, "v2_routing_identity_excluded_tool") for item in exclude_tool_ids)
        as_payload = getattr(payload, "as_payload", None)
        if not callable(as_payload):
            raise RoutingExperimentError("v2_model_routing_identity_payload_unavailable")
        raw_payload = as_payload()
        if not isinstance(raw_payload, Mapping):
            raise RoutingExperimentError("v2_model_routing_identity_payload_is_not_an_object")
        normalized = dict(raw_payload)
        if excluded:
            typed_steps = getattr(payload, "steps", None)
            if not isinstance(typed_steps, Sequence) or isinstance(typed_steps, (str, bytes)):
                raise RoutingExperimentError("v2_model_routing_identity_steps_are_invalid")
            order_by_group: dict[str, int] = {}
            normalized_steps: list[Mapping[str, Any]] = []
            for step in typed_steps:
                tool_id = str(getattr(step, "tool_id", "") or "")
                if tool_id in excluded:
                    continue
                group = str(getattr(step, "phase", "candidate") or "candidate")
                order = order_by_group.get(group, 0)
                order_by_group[group] = order + 1
                try:
                    normalized_step = replace(step, order=order)
                    step_payload = normalized_step.as_payload()
                except Exception as exc:
                    raise RoutingExperimentError("v2_model_routing_identity_step_normalization_failed") from exc
                if not isinstance(step_payload, Mapping):
                    raise RoutingExperimentError("v2_model_routing_identity_step_payload_is_not_an_object")
                normalized_steps.append(step_payload)
            normalized["steps"] = normalized_steps
        try:
            contracts = importlib.import_module("sourcing_model.routing.contracts")
            hasher = getattr(contracts, "sha256_payload", None)
            if not callable(hasher):
                raise RoutingExperimentError("v2_model_routing_identity_hasher_unavailable")
            return model_hash_to_lab(hasher(normalized), "v2_model_routing_identity")
        except RoutingExperimentError:
            raise
        except Exception as exc:
            raise RoutingExperimentError("v2_model_routing_identity_hash_failed") from exc

    def variant_tool_descriptors(self, payload: Any, *, stage: str) -> Sequence[Mapping[str, Any]]:
        raw_steps = getattr(payload, "steps", ())
        descriptors: list[Mapping[str, Any]] = []
        for step in raw_steps:
            tool_id = str(getattr(step, "tool_id", "") or (step.get("tool_id") if isinstance(step, Mapping) else ""))
            if not tool_id:
                continue
            definition = next((item for item in self.catalog.tools if item.tool_id == tool_id), None)
            if definition is None:
                descriptors.append({"tool_id": tool_id, "stage": stage, "source_add": False})
                continue
            manifest_hash = str(getattr(definition, "manifest_sha256", "") or "")
            if manifest_hash and not manifest_hash.startswith("sha256:"):
                manifest_hash = model_hash_to_lab(manifest_hash, "model_tool_manifest_hash")
            descriptors.append(
                {
                    "tool_id": tool_id,
                    "stage": stage,
                    "source_add": str(getattr(definition, "origin", "") or "") == "source_add",
                    "manifest_hash": manifest_hash,
                }
            )
        return tuple(descriptors)

    def lookup_tool_descriptor(self, tool_id: str, *, stage: str) -> Mapping[str, Any] | None:
        definition = next((item for item in self.catalog.tools if item.tool_id == tool_id), None)
        if definition is None or stage not in tuple(getattr(definition, "stages", ())):
            return None
        manifest_hash = str(getattr(definition, "manifest_sha256", "") or "")
        if manifest_hash and not manifest_hash.startswith("sha256:"):
            manifest_hash = model_hash_to_lab(manifest_hash, "model_tool_manifest_hash")
        return {
            "tool_id": tool_id,
            "stage": stage,
            "source_add": str(getattr(definition, "origin", "") or "") == "source_add",
            "manifest_hash": manifest_hash,
        }

    def validate_provider_binding(self, binding: ProviderBindingIdentity, *, stage: str) -> Sequence[str]:
        del stage
        definition = next((item for item in self.catalog.tools if item.tool_id == binding.tool_id), None)
        if definition is None:
            return (f"model_tool_definition_missing:{binding.tool_id}",)
        expected = str(getattr(definition, "manifest_sha256", "") or "")
        if expected and not expected.startswith("sha256:"):
            expected = model_hash_to_lab(expected, "model_tool_manifest_hash")
        return (f"provider_binding_manifest_mismatch:{binding.tool_id}",) if expected and expected != binding.manifest_hash else ()

    def compile_variant(
        self,
        payload: Any,
        *,
        stage: str,
        feature_set: Any,
        available_tools: Mapping[str, bool],
        remaining_seconds: float,
        remaining_calls: int,
        credit_cap: float,
        expected_signal_type: str = "",
    ) -> Any:
        if stage == self.runtime.STAGE_INTENT_EVIDENCE:
            return self.compile_initial(
                payload,
                signal_type=expected_signal_type,
                feature_set=feature_set,
                available_tools=available_tools,
                remaining_seconds=remaining_seconds,
                remaining_calls=remaining_calls,
                credit_cap=credit_cap,
            )
        if self.candidate_profiles is None:
            raise RoutingExperimentError("candidate_routing_profile_compiler_unavailable")
        registry = self.candidate_profiles.CandidateRoutingProfileRegistry(profiles=(payload,))
        return self.candidate_profiles.compile_candidate_routing_plan(
            catalog=self.runtime.runtime_catalog(available_tools),
            policy=self.runtime.runtime_policy(),
            context=self.runtime.RouteContext(
                stage=self.runtime.STAGE_CANDIDATE_ACQUISITION,
                features=tuple(feature_set.features),
                remaining_seconds=remaining_seconds,
                remaining_calls=remaining_calls,
                remaining_results=1,
                credit_cap=credit_cap,
            ),
            feature_set=feature_set,
            registry=registry,
            profile_id=payload.profile_id,
            expected_profile_sha256=payload.sha256(),
            expected_profile_version=payload.version,
        )

    def plan_decision_projection(self, plan: Any) -> Mapping[str, Any]:
        steps = getattr(getattr(plan, "route", None), "steps", ())
        attempted = tuple(str(getattr(step, "tool_id", "")) for step in steps if str(getattr(step, "execution_mode", "")) == self.runtime.EXECUTION_INVOKE)
        exclusions = getattr(getattr(plan, "route", None), "exclusions", ())
        return {
            "attempted_tool_ids": tuple(item for item in attempted if item),
            "skipped_tool_reasons": {
                str(getattr(item, "tool_id", "")): str(getattr(item, "reason", "route_excluded"))
                for item in exclusions
                if str(getattr(item, "tool_id", ""))
            },
            "outcome_reasons": {},
        }

    def intent_release_policy_hash(self) -> str:
        try:
            release_policy = importlib.import_module("sourcing_model.intent_release_benchmark")
            raw_hash = getattr(release_policy, "INTENT_RELEASE_POLICY_V1_SHA256", "")
            return model_hash_to_lab(raw_hash, "intent_release_policy_hash")
        except Exception as exc:
            raise RoutingExperimentError(
                "intent_release_policy_identity_unavailable"
            ) from exc


def _adapter_intent_release_policy_hash(
    adapter: RoutingAdmissionPlanAdapter,
    *,
    required: bool = False,
) -> str | None:
    getter = getattr(adapter, "intent_release_policy_hash", None)
    if not callable(getter):
        if required:
            raise RoutingExperimentError("intent_release_policy_identity_unavailable")
        return None
    try:
        return _ensure_hash(getter(), "intent_release_policy_hash")
    except RoutingExperimentError:
        raise
    except Exception as exc:
        raise RoutingExperimentError("intent_release_policy_identity_unavailable") from exc


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
        return adapter.parse_profile(payload)
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


def _model_feature_set(
    frozen: FrozenRoutingInput,
    adapter: RoutingAdmissionPlanAdapter,
) -> Any:
    try:
        return adapter.parse_feature_set(frozen.feature_set_payload)
    except Exception as exc:
        raise RoutingExperimentError(f"model_feature_set_parse_failed:{exc}") from exc


def validate_model_routing_profile(
    profile: LabRoutingProfile | Mapping[str, Any],
    *,
    adapter: RoutingAdmissionPlanAdapter,
    signal_type: str,
    feature_set: Any,
    binding_tool_ids: Iterable[str],
    binding_source_lineages: Mapping[str, str],
) -> list[str]:
    """Validate the exact model payload through the pinned model adapter."""

    errors: list[str] = []
    try:
        plan = _model_profile(profile, adapter)
        errors.extend(
            str(item)
            for item in adapter.validate_profile(
                plan,
                signal_type=signal_type,
                feature_set=feature_set,
                binding_tool_ids=frozenset(binding_tool_ids),
                binding_source_lineages=binding_source_lineages,
            )
        )
        canonical = adapter.profile_as_payload(plan)
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
    # This is contract metadata, not an evaluation parameter.  It is omitted
    # from the identity payload so missing mapping fields cannot silently
    # become promotion gates.
    explicit: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RoutingEvaluationGates":
        required = {
            "min_calibration_precision",
            "min_holdout_precision",
            "min_holdout_recall",
            "max_holdout_no_signal_credit_microunits",
            "min_marginal_verified_positives_per_credit",
            "intent_release_policy_hash",
        }
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
            explicit=required.issubset(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_calibration_precision": self.min_calibration_precision,
            "min_holdout_precision": self.min_holdout_precision,
            "min_holdout_recall": self.min_holdout_recall,
            "max_holdout_no_signal_credit_microunits": self.max_holdout_no_signal_credit_microunits,
            "min_marginal_verified_positives_per_credit": self.min_marginal_verified_positives_per_credit,
            "intent_release_policy_hash": self.intent_release_policy_hash,
        }


def validate_routing_evaluation_gates(
    gates: RoutingEvaluationGates | Mapping[str, Any],
) -> list[str]:
    if not isinstance(gates, RoutingEvaluationGates):
        gates = RoutingEvaluationGates.from_mapping(gates)
    errors: list[str] = []
    if not gates.explicit:
        errors.append("evaluation_gates_must_be_explicit")
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
    try:
        errors.extend(
            str(item)
            for item in adapter.validate_artifact_identity(spec.artifact)
        )
    except Exception as exc:
        errors.append(f"model_artifact_identity_validation_failed:{exc}")
    errors.extend(validate_frozen_routing_input(spec.frozen_input))
    feature_set = None
    try:
        feature_set = _model_feature_set(spec.frozen_input, adapter)
        errors.extend(
            str(item)
            for item in adapter.validate_feature_set(
                feature_set,
                expected_hash=lab_hash_to_model(
                    spec.frozen_input.feature_set_hash,
                    "feature_set_hash",
                ),
                expected_features=spec.frozen_input.features,
            )
        )
    except RoutingExperimentError as exc:
        errors.append(str(exc))
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
                feature_set=feature_set,
                binding_tool_ids=binding_ids,
                binding_source_lineages={
                    item.tool_id: item.source_lineage_id
                    for item in spec.provider_bindings
                },
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
    if spec.allow_live_credit_spend:
        missing_ceilings = {
            item.binding_id for item in spec.provider_bindings
            if item.binding_id not in spec.credit_budget.provider_credit_ceilings
        }
        errors.extend(
            f"measured_lab_requires_provider_credit_ceiling:{binding_id}"
            for binding_id in sorted(missing_ceilings)
        )
    try:
        errors.extend(validate_routing_evaluation_gates(spec.gates))
        release_policy_hash = _adapter_intent_release_policy_hash(adapter)
        if release_policy_hash is not None and spec.gates.intent_release_policy_hash != release_policy_hash:
            errors.append("evaluation_gates_release_policy_hash_mismatch")
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
            feature_set=_model_feature_set(spec.frozen_input, adapter),
            binding_tool_ids={item.tool_id for item in spec.provider_bindings},
            binding_source_lineages={
                item.tool_id: item.source_lineage_id
                for item in spec.provider_bindings
            },
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
    # Accepted for call-site readability.  The canonical request fingerprint
    # supplied by the caller remains the source of request identity.
    binding_id: str = "",
    source_lineage_id: str = "",
    unit_ref: str = "",
) -> str:
    """Key one paid request by exact tool/binding version and input fingerprint.

    The current Lab request builder includes ``unit_ref`` in the fingerprint,
    so receipts are intentionally not reused across units.  A future shared
    request builder may prove safe cross-unit reuse; this key function must
    not infer that equivalence by dropping metadata on its own.
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
    """Durable persistence seam for redacted Lab provider receipts.

    The site already has ``EvidenceStore`` and ``ProviderOutcomeStoreV2``, but
    neither is a compatible implementation: ``EvidenceStore`` stores replay
    tapes and exposes only fingerprint lookup/record, while
    ``ProviderOutcomeStoreV2`` stores aggregate checkpoint transitions and does
    not expose typed receipt lookup.  Until a site adapter is reviewed, the
    Lab must use this seam with the append-only JSONL implementation below (or
    another implementation with the same contract); it must not silently
    write these receipts into either production store.
    """

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
    repository never truncates or rewrites prior records.  Its lock protects
    repository instances in one Python process only; it is not a cross-process
    locking guarantee.
    """

    _lock_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self._rows: dict[str, ProviderReceipt] = {}
        with self._lock_guard:
            self._lock = self._locks.setdefault(str(self.path), threading.RLock())
        with self._lock:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
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
                expected_key = provider_receipt_key(
                    tool_id=receipt.tool_id,
                    binding_version=receipt.binding_version,
                    request_fingerprint=receipt.request_fingerprint,
                )
                if key != expected_key:
                    raise RoutingExperimentError("provider receipt repository key mismatch")
                existing = self._rows.get(key)
                if existing is not None and existing.to_dict() != receipt.to_dict():
                    raise RoutingExperimentError("provider receipt key collision")
                self._rows[key] = receipt

    def get(self, key: str) -> ProviderReceipt | None:
        with self._lock:
            return self._rows.get(str(key))

    def append(self, key: str, receipt: ProviderReceipt) -> ProviderReceipt:
        with self._lock:
            key = str(key)
            expected_key = provider_receipt_key(
                tool_id=receipt.tool_id,
                binding_version=receipt.binding_version,
                request_fingerprint=receipt.request_fingerprint,
            )
            if key != expected_key:
                raise RoutingExperimentError("provider receipt repository key mismatch")
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
        with self._lock:
            return tuple(self._rows)


class ProviderReceiptStore:
    """Append-only, content-addressed receipt store over the Lab repository seam."""

    def __init__(self, repository: ProviderReceiptRepository | None = None) -> None:
        self.repository = repository if repository is not None else InMemoryProviderReceiptRepository()
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
        expected_key = provider_receipt_key(
            tool_id=normalized.tool_id,
            binding_version=normalized.binding_version,
            request_fingerprint=normalized.request_fingerprint,
        )
        if str(key) != expected_key:
            raise RoutingExperimentError("provider receipt key mismatch")
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
    billing_rollup_id: str = ""
    billing_rollup_hash: str = ""
    billing_rollup_total_credit_microunits: int = 0
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
            "billing_rollup_id": self.billing_rollup_id,
            "billing_rollup_hash": self.billing_rollup_hash,
            "billing_rollup_total_credit_microunits": self.billing_rollup_total_credit_microunits,
            "live_credit_spend": self.live_credit_spend,
            "immutable": self.immutable,
        }


def validate_routing_evaluation_receipt(
    evaluation: RoutingEvaluationReceipt | Mapping[str, Any],
    spec: RoutingExperimentSpec,
    *,
    adapter: RoutingAdmissionPlanAdapter,
    receipt_store: ProviderReceiptStore | None = None,
) -> list[str]:
    """Recompute the immutable evaluation identity before promotion.

    The evaluation receipt stores references, not provider payloads.  When a
    repository is supplied, each reference is additionally resolved and
    validated against the typed receipt.  Promotion calls this validator with
    no repository because its immutable receipt hash is the persisted
    evidence boundary; the Lab must retain the repository alongside that
    receipt when it needs to audit provider contents.
    """

    if not isinstance(evaluation, RoutingEvaluationReceipt):
        return ["evaluation_receipt_must_be_typed"]
    if not isinstance(spec, RoutingExperimentSpec):
        return ["evaluation_spec_must_be_typed"]
    errors: list[str] = []
    if evaluation.contract_version != ROUTING_EVALUATION_RECEIPT_VERSION:
        errors.append("evaluation_receipt_contract_version_mismatch")
    if not evaluation.immutable:
        errors.append("evaluation_receipt_must_be_immutable")
    receipt_id = str(evaluation.receipt_id or "")
    if not _SAFE_REF_RE.fullmatch(receipt_id) or not re.fullmatch(
        r"routing_evaluation:[0-9a-f]{16}", receipt_id
    ):
        errors.append("evaluation_receipt_id_is_invalid")
    else:
        pending = replace(evaluation, receipt_id="routing_evaluation:pending")
        expected_id = "routing_evaluation:" + sha256_json(pending.to_dict()).split(":", 1)[1][:16]
        if evaluation.receipt_id != expected_id:
            errors.append("evaluation_receipt_id_mismatch")
    for field_name, value in (
        ("experiment_hash", evaluation.experiment_hash),
        ("artifact_hash", evaluation.artifact_hash),
        ("manifest_hash", evaluation.manifest_hash),
        ("feature_set_hash", evaluation.feature_set_hash),
    ):
        try:
            _ensure_hash(value, field_name)
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    if evaluation.experiment_id != spec.experiment_id:
        errors.append("evaluation_receipt_experiment_id_mismatch")
    if evaluation.experiment_hash != spec.experiment_hash():
        errors.append("evaluation_receipt_experiment_hash_mismatch")
    if evaluation.artifact_hash != spec.artifact.model_artifact_hash:
        errors.append("evaluation_receipt_artifact_hash_mismatch")
    if evaluation.manifest_hash != spec.artifact.manifest_hash:
        errors.append("evaluation_receipt_manifest_hash_mismatch")
    if evaluation.feature_set_hash != spec.frozen_input.feature_set_hash:
        errors.append("evaluation_receipt_feature_set_hash_mismatch")
    if evaluation.calibration_unit_count != len(spec.frozen_input.calibration_unit_refs):
        errors.append("evaluation_receipt_calibration_unit_count_mismatch")
    if evaluation.holdout_unit_count != len(spec.frozen_input.holdout_unit_refs):
        errors.append("evaluation_receipt_holdout_unit_count_mismatch")
    if evaluation.baseline_profile_id != spec.baseline_profile_id:
        errors.append("evaluation_receipt_baseline_profile_id_mismatch")
    if evaluation.provider_cache_hits < 0 or evaluation.provider_cache_misses < 0:
        errors.append("evaluation_receipt_cache_counts_are_invalid")
    if evaluation.live_credit_spend:
        if not evaluation.billing_rollup_id:
            errors.append("evaluation_receipt_billing_rollup_id_missing")
        try:
            _ensure_hash(evaluation.billing_rollup_hash, "billing_rollup_hash")
        except RoutingExperimentError as exc:
            errors.append(str(exc))
        if evaluation.billing_rollup_total_credit_microunits < 0:
            errors.append("evaluation_receipt_billing_total_is_invalid")
    elif evaluation.billing_rollup_id or evaluation.billing_rollup_hash or evaluation.billing_rollup_total_credit_microunits:
        errors.append("fixture_evaluation_must_not_contain_billing_rollup")

    expected_variant_ids: set[str] = set()
    try:
        expected_variant_ids = {_profile_id(item, adapter) for item in spec.variants}
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    actual_variant_ids = [item.variant_id for item in evaluation.variants]
    if len(actual_variant_ids) != len(set(actual_variant_ids)):
        errors.append("evaluation_variant_ids_must_be_unique")
    if set(actual_variant_ids) != expected_variant_ids:
        errors.append("evaluation_variant_ids_do_not_match_spec")
    if evaluation.baseline_profile_id not in actual_variant_ids:
        errors.append("evaluation_baseline_variant_is_missing")
    if evaluation.selected_profile_id:
        selected = next(
            (item for item in evaluation.variants if item.variant_id == evaluation.selected_profile_id),
            None,
        )
        if selected is None:
            errors.append("evaluation_selected_profile_is_missing")
        elif not selected.passed:
            errors.append("evaluation_selected_profile_did_not_pass")

    top_refs = tuple(evaluation.provider_receipt_refs)
    if top_refs != tuple(sorted(set(top_refs))):
        errors.append("evaluation_provider_receipt_refs_must_be_sorted_unique")
    all_variant_refs: set[str] = set()
    for item in evaluation.variants:
        try:
            _ensure_safe_ref(item.variant_id, "evaluation_variant_id")
        except RoutingExperimentError as exc:
            errors.append(str(exc))
        if tuple(item.receipt_refs) != tuple(sorted(set(item.receipt_refs))):
            errors.append(f"evaluation_variant_receipt_refs_must_be_sorted_unique:{item.variant_id}")
        if not set(item.receipt_refs).issubset(set(top_refs)):
            errors.append(f"evaluation_variant_receipt_refs_not_in_top_level:{item.variant_id}")
        all_variant_refs.update(item.receipt_refs)
        for ref in item.receipt_refs:
            if not re.fullmatch(r"provider_receipt:[0-9a-f]{16}", ref):
                errors.append("evaluation_provider_receipt_ref_is_invalid")
        for metrics in (item.calibration, item.holdout):
            if metrics.split not in {"calibration", "holdout"} or metrics.unit_count < 0:
                errors.append(f"evaluation_metrics_are_invalid:{item.variant_id}")
            if metrics.total_credit_microunits < 0 or metrics.no_signal_credit_microunits < 0:
                errors.append(f"evaluation_metrics_credit_is_invalid:{item.variant_id}")
    for ref in top_refs:
        if not re.fullmatch(r"provider_receipt:[0-9a-f]{16}", ref):
            errors.append("evaluation_provider_receipt_ref_is_invalid")
    if all_variant_refs != set(top_refs):
        errors.append("evaluation_provider_receipt_refs_do_not_match_variants")
    if receipt_store is not None:
        for key in receipt_store.repository.keys():
            receipt = receipt_store.repository.get(key)
            if receipt is None:
                continue
            if receipt.receipt_ref not in set(top_refs):
                errors.append("evaluation_repository_contains_unreferenced_receipt")
            errors.extend(validate_provider_receipt(receipt))
        repository_refs = {
            receipt.receipt_ref
            for key in receipt_store.repository.keys()
            if (receipt := receipt_store.repository.get(key)) is not None
        }
        if repository_refs != set(top_refs):
            errors.append("evaluation_receipt_refs_do_not_match_repository")
    return sorted(set(errors))


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


class _ExperimentBudgetLedger:
    """Reserve measured-Lab spend before a model plan can invoke a provider."""

    def __init__(
        self,
        spec: RoutingExperimentSpec,
        receipt_store: ProviderReceiptStore,
        bindings: Mapping[str, ProviderBindingIdentity],
    ) -> None:
        self.enabled = bool(spec.allow_live_credit_spend)
        self.spec = spec
        self.receipt_store = receipt_store
        self.bindings = bindings
        self.reserved_total = 0
        self.reserved_by_binding: dict[str, int] = {}

    def _actual_spend(self) -> tuple[int, dict[str, int]]:
        total = 0
        by_binding: dict[str, int] = {}
        for key in self.receipt_store.repository.keys():
            receipt = self.receipt_store.repository.get(key)
            if receipt is None:
                continue
            total += receipt.credit_microunits
            by_binding[receipt.binding_id] = by_binding.get(receipt.binding_id, 0) + receipt.credit_microunits
        return total, by_binding

    def reserve(self, budgets: Sequence[RoutingPlanStepBudget]) -> dict[str, int]:
        if not self.enabled:
            return {}
        requested_by_binding: dict[str, int] = {}
        for budget in budgets:
            if budget.execution_mode != "invoke" or budget.max_calls <= 0:
                continue
            binding = self.bindings.get(budget.tool_id)
            if binding is None:
                raise RoutingExperimentError(f"model plan references unbound tool:{budget.tool_id}")
            requested_by_binding[binding.binding_id] = (
                requested_by_binding.get(binding.binding_id, 0) + budget.credit_microunits
            )
        requested_total = sum(requested_by_binding.values())
        actual_total, actual_by_binding = self._actual_spend()
        if actual_total + self.reserved_total + requested_total > self.spec.credit_budget.total_credit_microunits:
            raise RoutingExperimentError("measured_lab_total_credit_cap_would_be_exceeded")
        for binding_id, requested in requested_by_binding.items():
            ceiling = self.spec.credit_budget.provider_credit_ceilings.get(binding_id)
            if ceiling is None:
                raise RoutingExperimentError(f"measured_lab_provider_credit_ceiling_missing:{binding_id}")
            if actual_by_binding.get(binding_id, 0) + self.reserved_by_binding.get(binding_id, 0) + requested > ceiling:
                raise RoutingExperimentError(
                    f"measured_lab_provider_credit_cap_would_be_exceeded:{binding_id}"
                )
        self.reserved_total += requested_total
        for binding_id, requested in requested_by_binding.items():
            self.reserved_by_binding[binding_id] = self.reserved_by_binding.get(binding_id, 0) + requested
        return requested_by_binding

    def release(self, reservation: Mapping[str, int]) -> None:
        if not self.enabled:
            return
        total = sum(reservation.values())
        self.reserved_total -= total
        if self.reserved_total < 0:
            raise RoutingExperimentError("measured_lab_budget_ledger_underflow")
        for binding_id, amount in reservation.items():
            current = self.reserved_by_binding.get(binding_id, 0) - amount
            if current < 0:
                raise RoutingExperimentError("measured_lab_provider_ledger_underflow")
            if current:
                self.reserved_by_binding[binding_id] = current
            else:
                self.reserved_by_binding.pop(binding_id, None)


class _UnitBudgetLedger:
    """Cumulative per-unit budget with pre-call route-step reservations."""

    def __init__(self, experiment_ledger: _ExperimentBudgetLedger) -> None:
        self.experiment_ledger = experiment_ledger
        self.started_at = time.monotonic()
        self.spent_calls = 0
        self.spent_credit_microunits = 0
        self.spent_latency_ms = 0
        self.reserved_calls = 0
        self.reserved_credit_microunits = 0
        self.reserved_seconds = 0.0
        self.active_calls_by_tool: dict[str, int] = {}
        self.active_credit_by_tool: dict[str, int] = {}

    def remaining_seconds(self) -> float:
        elapsed = max(time.monotonic() - self.started_at, self.spent_latency_ms / 1000)
        return max(0.0, 60.0 - elapsed - self.reserved_seconds)

    def remaining_calls(self) -> int:
        return max(0, MAX_TOOLS_PER_VARIANT - self.spent_calls - self.reserved_calls)

    def remaining_credit(self) -> float:
        remaining = (
            MAX_CREDIT_MICROUNITS_PER_VARIANT
            - self.spent_credit_microunits
            - self.reserved_credit_microunits
        )
        return max(0, remaining) / 1_000_000

    def reserve(
        self,
        plan: Any,
        *,
        adapter: RoutingAdmissionPlanAdapter,
        bindings: Mapping[str, ProviderBindingIdentity],
    ) -> dict[str, Any]:
        try:
            raw_budgets = adapter.plan_step_budgets(plan)
        except AttributeError as exc:
            raise RoutingExperimentError("model_adapter_missing_plan_step_budgets") from exc
        try:
            budgets = tuple(
                item if isinstance(item, RoutingPlanStepBudget) else RoutingPlanStepBudget(**dict(item))
                for item in raw_budgets
            )
        except Exception as exc:
            raise RoutingExperimentError(f"model_plan_step_budget_invalid:{exc}") from exc
        calls = sum(
            item.max_calls for item in budgets if item.execution_mode == "invoke"
        )
        credits = sum(
            item.credit_microunits for item in budgets if item.execution_mode == "invoke"
        )
        seconds = sum(
            item.timeout_seconds for item in budgets if item.execution_mode == "invoke"
        )
        if calls > self.remaining_calls():
            raise RoutingExperimentError("route_step_budget_exceeds_remaining_unit_calls")
        if credits > MAX_CREDIT_MICROUNITS_PER_VARIANT - self.spent_credit_microunits - self.reserved_credit_microunits:
            raise RoutingExperimentError("route_step_budget_exceeds_remaining_unit_credit")
        if seconds > self.remaining_seconds():
            raise RoutingExperimentError("route_step_budget_exceeds_remaining_unit_seconds")
        for item in budgets:
            if item.execution_mode == "invoke" and item.max_calls > 0 and item.tool_id not in bindings:
                raise RoutingExperimentError(f"model plan references unbound tool:{item.tool_id}")
        reservation = self.experiment_ledger.reserve(budgets)
        self.reserved_calls += calls
        self.reserved_credit_microunits += credits
        self.reserved_seconds += seconds
        for item in budgets:
            if item.execution_mode != "invoke" or item.max_calls <= 0:
                continue
            self.active_calls_by_tool[item.tool_id] = self.active_calls_by_tool.get(item.tool_id, 0) + item.max_calls
            self.active_credit_by_tool[item.tool_id] = self.active_credit_by_tool.get(item.tool_id, 0) + item.credit_microunits
        return {
            "calls": calls,
            "credits": credits,
            "seconds": seconds,
            "budgets": budgets,
            "experiment": reservation,
        }

    def before_provider_call(self, tool_id: str) -> None:
        remaining = self.active_calls_by_tool.get(tool_id, 0)
        if remaining <= 0:
            raise RoutingExperimentError(f"provider_call_not_reserved:{tool_id}")
        self.active_calls_by_tool[tool_id] = remaining - 1

    def record_provider_receipt(self, receipt: ProviderReceipt) -> None:
        reserved_credit = self.active_credit_by_tool.get(receipt.tool_id, 0)
        if receipt.credit_microunits > reserved_credit:
            raise RoutingExperimentError(
                f"provider_receipt_credit_exceeds_reserved_step:{receipt.tool_id}"
            )
        self.active_credit_by_tool[receipt.tool_id] = reserved_credit - receipt.credit_microunits
        self.spent_calls += 1
        self.spent_credit_microunits += receipt.credit_microunits
        self.spent_latency_ms += receipt.latency_ms

    def release(self, reservation: Mapping[str, Any]) -> None:
        self.reserved_calls -= int(reservation["calls"])
        self.reserved_credit_microunits -= int(reservation["credits"])
        self.reserved_seconds -= float(reservation["seconds"])
        if min(self.reserved_calls, self.reserved_credit_microunits) < 0 or self.reserved_seconds < -1e-9:
            raise RoutingExperimentError("unit_budget_ledger_underflow")
        for item in reservation["budgets"]:
            if item.execution_mode != "invoke" or item.max_calls <= 0:
                continue
            # The reservation may have been partially consumed by provider
            # calls.  Release the remaining per-tool reservation as a whole;
            # subtracting the original worst-case amount would underflow.
            self.active_calls_by_tool.pop(item.tool_id, None)
            self.active_credit_by_tool.pop(item.tool_id, None)
        self.experiment_ledger.release(reservation["experiment"])


def _run_variant_unit(
    spec: RoutingExperimentSpec,
    variant: LabRoutingProfile,
    *,
    adapter: RoutingAdmissionPlanAdapter,
    unit_ref: str,
    bindings: Mapping[str, ProviderBindingIdentity],
    receipt_store: ProviderReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    experiment_ledger: _ExperimentBudgetLedger,
) -> tuple[list[ProviderReceipt], bool]:
    profile = _model_profile(variant, adapter)
    feature_set = _model_feature_set(spec.frozen_input, adapter)
    available_tools = {item.tool_id: True for item in spec.provider_bindings}
    ledger = _UnitBudgetLedger(experiment_ledger)
    plan = adapter.compile_initial(
        profile,
        signal_type=spec.signal_type,
        feature_set=feature_set,
        available_tools=available_tools,
        remaining_seconds=ledger.remaining_seconds(),
        remaining_calls=ledger.remaining_calls(),
        credit_cap=ledger.remaining_credit(),
    )
    _validate_compiled_admission_plan(
        plan,
        adapter=adapter,
        feature_set_hash=spec.frozen_input.feature_set_hash,
    )
    bindings_by_tool = dict(bindings)
    receipts: list[ProviderReceipt] = []

    def invoke(tool_id: str) -> ProviderReceipt:
        binding = bindings_by_tool.get(tool_id)
        if binding is None:
            raise RoutingExperimentError(f"model plan references unbound tool:{tool_id}")
        request_fingerprint = sha256_json(
            {
                "experiment_request": "intent-route-v1",
                "tool_id": binding.tool_id,
                "unit_ref": unit_ref,
            }
        )
        key = provider_receipt_key(
            tool_id=binding.tool_id,
            binding_version=binding.adapter_version,
            request_fingerprint=request_fingerprint,
        )
        receipt = receipt_store.get(key)
        if receipt is None:
            ledger.before_provider_call(tool_id)
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
            ledger.record_provider_receipt(receipt)
            receipt_store.put(key, receipt)
        else:
            ledger.before_provider_call(tool_id)
        receipts.append(receipt)
        return receipt

    def execute_reserved(current_plan: Any) -> tuple[Sequence[Any], bool]:
        reservation = ledger.reserve(
            current_plan,
            adapter=adapter,
            bindings=bindings_by_tool,
        )
        try:
            return adapter.execute_plan(current_plan, invoke)
        finally:
            ledger.release(reservation)

    _initial_results, predicted = execute_reserved(plan)
    if predicted:
        if adapter.has_conditional_confirmation(plan):
            try:
                confirmation = adapter.compile_confirmation(
                    plan,
                    profile=profile,
                    feature_set=feature_set,
                    available_tools=available_tools,
                    remaining_seconds=ledger.remaining_seconds(),
                    remaining_calls=ledger.remaining_calls(),
                    credit_cap=ledger.remaining_credit(),
                )
            except Exception as exc:
                # A primary hit must not silently skip a model-admitted
                # confirmation wave or fall through to an unrelated
                # challenger.  Keep the failure explicit and fail closed.
                if isinstance(exc, RoutingExperimentError):
                    raise
                raise RoutingExperimentError(
                    f"model confirmation compilation failed:{exc}"
                ) from exc
            _validate_compiled_admission_plan(
                confirmation,
                adapter=adapter,
                feature_set_hash=spec.frozen_input.feature_set_hash,
            )
            # The adapter executes the exact model-owned confirmation route;
            # its typed receipts are already appended by invoke().  The
            # primary verification remains the route prediction, while a
            # confirmation adapter failure is retained as explicit evidence.
            execute_reserved(confirmation)
        return receipts, True
    try:
        challenger = adapter.compile_challenger(
            plan,
            profile=profile,
            feature_set=feature_set,
            available_tools=available_tools,
            attempted_tool_ids=tuple(item.tool_id for item in receipts),
            attempted_source_lineages=tuple(item.source_lineage_id for item in receipts),
            remaining_seconds=ledger.remaining_seconds(),
            remaining_calls=ledger.remaining_calls(),
            credit_cap=ledger.remaining_credit(),
        )
    except Exception as exc:
        if "no distinct dormant challenger remains" in str(exc):
            return receipts, False
        if isinstance(exc, RoutingExperimentError):
            raise
        raise RoutingExperimentError(f"model challenger compilation failed:{exc}") from exc
    _validate_compiled_admission_plan(
        challenger,
        adapter=adapter,
        feature_set_hash=spec.frozen_input.feature_set_hash,
    )
    _challenger_results, challenger_predicted = execute_reserved(challenger)
    return receipts, bool(challenger_predicted)


def _validate_compiled_admission_plan(
    plan: Any,
    *,
    adapter: RoutingAdmissionPlanAdapter,
    feature_set_hash: str,
) -> None:
    payload = adapter.plan_as_payload(plan)
    if not isinstance(payload, Mapping):
        raise RoutingExperimentError("model admission plan payload is not an object")
    try:
        parsed = adapter.parse_plan(payload)
        raw_feature_hash = str(payload.get("feature_set_sha256") or "")
        if raw_feature_hash != lab_hash_to_model(feature_set_hash, "feature_set_hash"):
            raise RoutingExperimentError("model admission plan feature-set hash mismatch")
        if adapter.plan_hash(parsed) != adapter.plan_hash(plan):
            raise RoutingExperimentError("model admission plan hash is not stable")
    except RoutingExperimentError:
        raise
    except Exception as exc:
        raise RoutingExperimentError(f"model admission plan validation failed:{exc}") from exc


def _metrics_for_split(
    *,
    spec: RoutingExperimentSpec,
    split: str,
    unit_refs: Sequence[str],
    gold_labels: Mapping[str, bool],
    variant: LabRoutingProfile,
    adapter: RoutingAdmissionPlanAdapter,
    bindings: Mapping[str, ProviderBindingIdentity],
    receipt_store: ProviderReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    baseline_positive_units: set[str],
    receipt_refs: list[str],
    experiment_ledger: _ExperimentBudgetLedger,
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
    for unit_ref in unit_refs:
        receipts, predicted = _run_variant_unit(
            spec,
            variant,
            unit_ref=unit_ref,
            adapter=adapter,
            bindings=bindings,
            receipt_store=receipt_store,
            runner=runner,
            experiment_ledger=experiment_ledger,
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
    experiment_ledger: _ExperimentBudgetLedger,
) -> VariantEvaluation:
    bindings = {item.tool_id: item for item in spec.provider_bindings}
    receipt_refs: list[str] = []
    calibration = _metrics_for_split(
        spec=spec,
        split="calibration",
        unit_refs=spec.frozen_input.calibration_unit_refs,
        gold_labels=gold_labels,
        variant=variant,
        adapter=adapter,
        bindings=bindings,
        receipt_store=receipt_store,
        runner=runner,
        baseline_positive_units=baseline_positive_by_split.get("calibration", set()),
        receipt_refs=receipt_refs,
        experiment_ledger=experiment_ledger,
    )
    holdout = _metrics_for_split(
        spec=spec,
        split="holdout",
        unit_refs=spec.frozen_input.holdout_unit_refs,
        gold_labels=gold_labels,
        variant=variant,
        adapter=adapter,
        bindings=bindings,
        receipt_store=receipt_store,
        runner=runner,
        baseline_positive_units=baseline_positive_by_split.get("holdout", set()),
        receipt_refs=receipt_refs,
        experiment_ledger=experiment_ledger,
    )
    gates = spec.gates
    passed_precision = calibration.precision >= gates.min_calibration_precision and holdout.precision >= gates.min_holdout_precision
    passed_recall = holdout.recall >= gates.min_holdout_recall
    passed_cost = holdout.no_signal_credit_microunits <= gates.max_holdout_no_signal_credit_microunits
    passed_efficiency = holdout.marginal_verified_positives_per_credit >= gates.min_marginal_verified_positives_per_credit
    return VariantEvaluation(
        variant_id=_profile_id(variant, adapter),
        calibration=calibration,
        holdout=holdout,
        passed_precision_gate=passed_precision,
        passed_recall_gate=passed_recall,
        passed_cost_gate=passed_cost,
        passed_efficiency_gate=passed_efficiency,
        passed=passed_precision and passed_recall and passed_cost and passed_efficiency,
        receipt_refs=tuple(sorted(set(receipt_refs))),
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
    if spec.allow_live_credit_spend:
        if receipt_store is None:
            raise RoutingExperimentError(
                "measured_lab_requires_explicit_durable_receipt_repository"
            )
        if isinstance(receipt_store.repository, InMemoryProviderReceiptRepository):
            raise RoutingExperimentError(
                "measured_lab_requires_durable_receipt_repository"
            )
    # Fixture and replay tests may use the in-memory repository.  Measured Lab
    # runs are rejected above unless the caller binds an explicit durable
    # implementation of the existing receipt seam.
    store = receipt_store if receipt_store is not None else ProviderReceiptStore()
    baseline = next(
        item for item in spec.variants if _profile_id(item, adapter) == spec.baseline_profile_id
    )
    bindings = {item.tool_id: item for item in spec.provider_bindings}
    experiment_ledger = _ExperimentBudgetLedger(spec, store, bindings)
    baseline_positive_by_split: dict[str, set[str]] = {"calibration": set(), "holdout": set()}
    for split, refs in (
        ("calibration", spec.frozen_input.calibration_unit_refs),
        ("holdout", spec.frozen_input.holdout_unit_refs),
    ):
        for unit_ref in refs:
            receipts, predicted = _run_variant_unit(
                spec,
                baseline,
                unit_ref=unit_ref,
                adapter=adapter,
                bindings=bindings,
                receipt_store=store,
                runner=runner,
                experiment_ledger=experiment_ledger,
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
            experiment_ledger=experiment_ledger,
        )
        for variant in spec.variants
    )
    selected_profile_id = select_smallest_passing_variant(evaluations)
    billing_rollup = authoritative_billing_rollup(store) if authoritative_billing_rollup else None
    billing_rollup_id = ""
    billing_rollup_hash = ""
    billing_rollup_total = 0
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
        billing_rollup_id = _ensure_safe_ref(
            billing_rollup.get("rollup_id"), "billing_rollup_id"
        )
        billing_rollup_hash = _ensure_hash(
            billing_rollup.get("rollup_hash"), "billing_rollup_hash"
        )
        billed_total = int(billing_rollup.get("total_credit_microunits", -1))
        billing_rollup_total = billed_total
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
        billing_rollup_id=billing_rollup_id,
        billing_rollup_hash=billing_rollup_hash,
        billing_rollup_total_credit_microunits=billing_rollup_total,
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
    receipt_errors = validate_routing_evaluation_receipt(
        evaluation,
        spec,
        adapter=adapter,
    )
    if receipt_errors:
        raise RoutingExperimentError(
            "invalid evaluation receipt: " + "; ".join(receipt_errors)
        )
    release_policy_hash = _adapter_intent_release_policy_hash(adapter, required=True)
    if spec.gates.intent_release_policy_hash != release_policy_hash:
        raise RoutingExperimentError("evaluation_gates_release_policy_hash_mismatch")
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


# ---------------------------------------------------------------------------
# V2 experiment orchestration
# ---------------------------------------------------------------------------
#
# V1 above intentionally remains frozen.  V2 adds an orchestration envelope
# around the model-owned route contracts.  The Lab stores only opaque model
# payloads and redacted projections; it does not define a candidate route or
# a second tool catalog.  A model adapter is therefore required for every
# operation that could interpret a profile or a plan.


def _v2_safe_stage(value: Any) -> str:
    stage = str(value or "").strip()
    if stage not in ROUTING_EXPERIMENT_STAGES:
        raise RoutingExperimentError("routing_experiment_v2_stage_is_invalid")
    return stage


def _v2_hash(value: Any, field_name: str) -> str:
    try:
        return _ensure_hash(value, field_name)
    except RoutingExperimentError:
        raise


def _v2_model_or_lab_hash(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text):
        return text
    if _RAW_SHA256_RE.fullmatch(text):
        return f"sha256:{text}"
    raise RoutingExperimentError(f"{field_name} must be a SHA-256 hash")


@dataclass(frozen=True)
class RoutingExperimentV2Input:
    """Frozen features and units shared by one stage-specific experiment."""

    stage: str
    feature_set_hash: str
    feature_set_payload: Mapping[str, Any]
    calibration_unit_refs: tuple[str, ...]
    holdout_unit_refs: tuple[str, ...]
    gold_label_set_hash: str
    signal_type: str = ""

    def __post_init__(self) -> None:
        _v2_safe_stage(self.stage)
        _v2_hash(self.feature_set_hash, "v2_feature_set_hash")
        _v2_hash(self.gold_label_set_hash, "v2_gold_label_set_hash")
        if not isinstance(self.feature_set_payload, Mapping):
            raise RoutingExperimentError("v2_feature_set_payload_is_required")
        if len(self.calibration_unit_refs) > MAX_UNITS_PER_SPLIT or len(self.holdout_unit_refs) > MAX_UNITS_PER_SPLIT:
            raise RoutingExperimentError("v2_unit_count_exceeds_limit")
        _ensure_ref_tuple(self.calibration_unit_refs, "v2_calibration_unit_refs", maximum=MAX_UNITS_PER_SPLIT)
        _ensure_ref_tuple(self.holdout_unit_refs, "v2_holdout_unit_refs", maximum=MAX_UNITS_PER_SPLIT)
        if set(self.calibration_unit_refs).intersection(self.holdout_unit_refs):
            raise RoutingExperimentError("v2_calibration_and_holdout_units_must_be_disjoint")
        if self.stage == "intent_evidence" and not str(self.signal_type or "").strip():
            raise RoutingExperimentError("v2_intent_signal_type_is_required")
        if self.stage == "candidate_acquisition" and str(self.signal_type or "").strip():
            raise RoutingExperimentError("v2_candidate_signal_type_must_be_empty")
        _ensure_no_secret_material(self.to_dict(), field_name="v2_routing_input")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "signal_type": self.signal_type,
            "feature_set_hash": self.feature_set_hash,
            "feature_set_payload": dict(self.feature_set_payload),
            "calibration_unit_refs": list(self.calibration_unit_refs),
            "holdout_unit_refs": list(self.holdout_unit_refs),
            "gold_label_set_hash": self.gold_label_set_hash,
        }


@dataclass(frozen=True)
class SourceAddProvenance:
    """Approved model SourceAdd identity; no endpoint or provider payload."""

    request_ref: str
    request_hash: str
    tool_id: str
    stage: str
    manifest_hash: str
    artifact_commit_sha: str
    approved: bool = True
    request_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _ensure_safe_ref(self.request_ref, "source_add_request_ref")
        _v2_hash(self.request_hash, "source_add_request_hash")
        _ensure_safe_ref(self.tool_id, "source_add_tool_id")
        _v2_safe_stage(self.stage)
        _v2_hash(self.manifest_hash, "source_add_manifest_hash")
        _ensure_git_sha(self.artifact_commit_sha, "source_add_artifact_commit_sha")
        if not isinstance(self.approved, bool) or not self.approved:
            raise RoutingExperimentError("source_add_provenance_must_be_approved")
        if self.request_payload is not None:
            if not isinstance(self.request_payload, Mapping):
                raise RoutingExperimentError("source_add_request_payload_is_invalid")
            if sha256_json(dict(self.request_payload)) != self.request_hash:
                raise RoutingExperimentError("source_add_request_hash_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingExperimentV2Variant:
    """One immutable model artifact plus an exact model profile payload."""

    variant_id: str
    stage: str
    artifact: SourcingModelArtifactIdentity
    routing_payload: Mapping[str, Any]
    binding_ids: tuple[str, ...]
    change_kind: str = "route_only"
    source_add_provenance: tuple[SourceAddProvenance, ...] = ()
    # This is the already-signed PrivateModelArtifactManifest document.  It
    # is an authority reference, not a Lab-defined replacement identity.
    artifact_authority_manifest: Mapping[str, Any] | None = None
    new_tool_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_safe_ref(self.variant_id, "v2_variant_id")
        _v2_safe_stage(self.stage)
        if not isinstance(self.artifact, SourcingModelArtifactIdentity):
            raise RoutingExperimentError("v2_variant_artifact_must_be_typed")
        if not isinstance(self.routing_payload, Mapping) or not self.routing_payload:
            raise RoutingExperimentError("v2_variant_routing_payload_is_required")
        if not self.binding_ids or len(self.binding_ids) > MAX_PROVIDER_BINDINGS:
            raise RoutingExperimentError("v2_variant_binding_ids_are_invalid")
        if self.change_kind not in {"route_only", "tool_only", "tool_and_route"}:
            raise RoutingExperimentError("v2_variant_change_kind_is_invalid")
        if self.change_kind == "route_only" and (self.source_add_provenance or self.new_tool_ids):
            raise RoutingExperimentError("v2_route_only_variant_has_source_add_provenance")
        if self.change_kind in {"tool_only", "tool_and_route"} and (not self.source_add_provenance or not self.new_tool_ids):
            raise RoutingExperimentError("v2_tool_variant_requires_source_add_provenance")
        if len(set(self.binding_ids)) != len(self.binding_ids):
            raise RoutingExperimentError("v2_variant_binding_ids_must_be_unique")
        for binding_id in self.binding_ids:
            _ensure_safe_ref(binding_id, "v2_variant_binding_id")
        for tool_id in self.new_tool_ids:
            _ensure_safe_ref(tool_id, "v2_variant_new_tool_id")
        if len(set(self.new_tool_ids)) != len(self.new_tool_ids):
            raise RoutingExperimentError("v2_variant_new_tool_ids_must_be_unique")
        if self.artifact_authority_manifest is not None and not isinstance(self.artifact_authority_manifest, Mapping):
            raise RoutingExperimentError("v2_artifact_authority_manifest_is_invalid")
        _ensure_no_secret_material(self.to_dict(), field_name="v2_variant")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "stage": self.stage,
            "artifact": self.artifact.to_dict(),
            "routing_payload": dict(self.routing_payload),
            "binding_ids": list(self.binding_ids),
            "change_kind": self.change_kind,
            "source_add_provenance": [item.to_dict() for item in self.source_add_provenance],
            "artifact_authority_manifest": (
                dict(self.artifact_authority_manifest)
                if self.artifact_authority_manifest is not None
                else None
            ),
            "new_tool_ids": list(self.new_tool_ids),
        }


@dataclass(frozen=True)
class RoutingExperimentV2Spec:
    experiment_id: str
    input: RoutingExperimentV2Input
    variants: tuple[RoutingExperimentV2Variant, ...]
    baseline_variant_id: str
    provider_bindings: tuple[ProviderBindingIdentity, ...]
    credit_budget: ExperimentCreditBudget
    gates: RoutingEvaluationGates
    availability: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    allow_live_credit_spend: bool = False
    receipt_execution_mode: str = ReceiptExecutionMode.FIXTURE.value
    contract_version: str = ROUTING_EXPERIMENT_V2_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RoutingExperimentV2Spec":
        input_data = data.get("input") or {}
        availability_data = data.get("availability", {})
        input_value = (
            input_data
            if isinstance(input_data, RoutingExperimentV2Input)
            else RoutingExperimentV2Input(
                stage=str(input_data.get("stage") or ""),
                signal_type=str(input_data.get("signal_type") or ""),
                feature_set_hash=str(input_data.get("feature_set_hash") or ""),
                feature_set_payload=dict(input_data.get("feature_set_payload") or {}),
                calibration_unit_refs=tuple(input_data.get("calibration_unit_refs") or ()),
                holdout_unit_refs=tuple(input_data.get("holdout_unit_refs") or ()),
                gold_label_set_hash=str(input_data.get("gold_label_set_hash") or ""),
            )
        )
        variants: list[RoutingExperimentV2Variant] = []
        for item in data.get("variants") or ():
            if isinstance(item, RoutingExperimentV2Variant):
                variants.append(item)
                continue
            variants.append(
                RoutingExperimentV2Variant(
                    variant_id=str(item.get("variant_id") or ""),
                    stage=str(item.get("stage") or ""),
                    artifact=SourcingModelArtifactIdentity.from_mapping(item.get("artifact") or {}),
                    routing_payload=dict(item.get("routing_payload") or {}),
                    binding_ids=tuple(item.get("binding_ids") or ()),
                    change_kind=str(item.get("change_kind") or "route_only"),
                    source_add_provenance=tuple(
                        SourceAddProvenance(**dict(provenance))
                        for provenance in item.get("source_add_provenance") or ()
                    ),
                    artifact_authority_manifest=(
                        dict(item["artifact_authority_manifest"])
                        if isinstance(item.get("artifact_authority_manifest"), Mapping)
                        else None
                    ),
                    new_tool_ids=tuple(item.get("new_tool_ids") or ()),
                )
            )
        return cls(
            experiment_id=str(data.get("experiment_id") or ""),
            input=input_value,
            variants=tuple(variants),
            baseline_variant_id=str(data.get("baseline_variant_id") or ""),
            provider_bindings=tuple(
                item if isinstance(item, ProviderBindingIdentity) else ProviderBindingIdentity.from_mapping(item)
                for item in data.get("provider_bindings") or ()
            ),
            credit_budget=(
                data["credit_budget"]
                if isinstance(data.get("credit_budget"), ExperimentCreditBudget)
                else ExperimentCreditBudget.from_mapping(data.get("credit_budget") or {})
            ),
            gates=(
                data["gates"]
                if isinstance(data.get("gates"), RoutingEvaluationGates)
                else RoutingEvaluationGates.from_mapping(data.get("gates") or {})
            ),
            availability=(
                {
                    str(variant_id): (
                        {str(tool_id): available for tool_id, available in value.items()}
                        if isinstance(value, Mapping)
                        else value
                    )
                    for variant_id, value in availability_data.items()
                }
                if isinstance(availability_data, Mapping)
                else availability_data
            ),
            allow_live_credit_spend=data.get("allow_live_credit_spend", False),
            receipt_execution_mode=str(data.get("receipt_execution_mode") or ReceiptExecutionMode.FIXTURE.value),
            contract_version=str(data.get("contract_version") or ROUTING_EXPERIMENT_V2_CONTRACT_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "experiment_id": self.experiment_id,
            "input": self.input.to_dict(),
            "variants": [item.to_dict() for item in self.variants],
            "baseline_variant_id": self.baseline_variant_id,
            "provider_bindings": [item.to_dict() for item in self.provider_bindings],
            "credit_budget": self.credit_budget.to_dict(),
            "gates": self.gates.to_dict(),
            "availability": (
                {
                    variant_id: (
                        dict(sorted(values.items())) if isinstance(values, Mapping) else values
                    )
                    for variant_id, values in sorted(self.availability.items())
                }
                if isinstance(self.availability, Mapping)
                else self.availability
            ),
            "allow_live_credit_spend": self.allow_live_credit_spend,
            "receipt_execution_mode": self.receipt_execution_mode,
        }

    def experiment_hash(self) -> str:
        return sha256_json(self.to_dict())


class RoutingExperimentV2Adapter(Protocol):
    """Model-owned v2 adapter; candidate profiles stay opaque to the Lab."""

    def parse_feature_set(self, payload: Mapping[str, Any]) -> Any: ...
    def validate_feature_set(self, feature_set: Any, *, expected_hash: str, expected_features: Sequence[str]) -> Sequence[str]: ...
    def validate_artifact_identity(self, artifact: SourcingModelArtifactIdentity) -> Sequence[str]: ...
    def parse_variant_payload(self, payload: Mapping[str, Any], *, stage: str) -> Any: ...
    def validate_variant_payload(self, payload: Any, *, stage: str, feature_set: Any, binding_tool_ids: frozenset[str], binding_source_lineages: Mapping[str, str], expected_signal_type: str = "") -> Sequence[str]: ...
    def routing_change_class(self, payload: Any, *, stage: str) -> str: ...
    def routing_identity(self, payload: Any, *, stage: str, exclude_tool_ids: Sequence[str] = ()) -> str: ...
    def variant_tool_descriptors(self, payload: Any, *, stage: str) -> Sequence[Mapping[str, Any]]: ...
    def lookup_tool_descriptor(self, tool_id: str, *, stage: str) -> Mapping[str, Any] | None: ...
    def validate_provider_binding(self, binding: ProviderBindingIdentity, *, stage: str) -> Sequence[str]: ...
    def compile_variant(self, payload: Any, *, stage: str, feature_set: Any, available_tools: Mapping[str, bool], remaining_seconds: float, remaining_calls: int, credit_cap: float, expected_signal_type: str = "") -> Any: ...
    def execute_plan(self, plan: Any, invoke: Callable[[str], Any]) -> tuple[Sequence[Any], bool]: ...
    def plan_as_payload(self, plan: Any) -> Mapping[str, Any]: ...
    def parse_plan(self, payload: Mapping[str, Any]) -> Any: ...
    def plan_hash(self, plan: Any) -> str: ...
    def route_hash(self, plan: Any) -> str: ...
    def plan_step_budgets(self, plan: Any) -> Sequence[RoutingPlanStepBudget]: ...
    def plan_decision_projection(self, plan: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class IsolatedRoutingAdapter:
    """A model adapter handle owned by one artifact worker process.

    The worker is created by the caller's model-authority runtime.  This
    handle carries the worker identity and the observed model artifact key so
    the Lab cannot accidentally compare two artifacts in one interpreter.
    """

    adapter: RoutingExperimentV2Adapter
    process_id: str
    observed_artifact_key: str

    def __post_init__(self) -> None:
        _ensure_safe_ref(self.process_id, "v2_model_process_id")
        _v2_hash(self.observed_artifact_key, "v2_observed_artifact_key")


class RoutingExperimentV2AdapterFactory:
    """Bind exactly one isolated worker identity to each artifact key."""

    def __init__(self, handles: Mapping[str, IsolatedRoutingAdapter]) -> None:
        self.handles = dict(handles)
        artifact_process: dict[str, str] = {}
        for variant_id, handle in self.handles.items():
            artifact_process.setdefault(handle.observed_artifact_key, handle.process_id)
            if artifact_process[handle.observed_artifact_key] != handle.process_id:
                raise RoutingExperimentError("v2_artifact_workers_must_be_stable")
        if len(set(artifact_process.values())) != len(artifact_process):
            raise RoutingExperimentError("v2_distinct_artifacts_require_distinct_workers")

    @classmethod
    def from_worker_factory(
        cls,
        variants: Sequence[RoutingExperimentV2Variant],
        worker_factory: Callable[[SourcingModelArtifactIdentity], IsolatedRoutingAdapter],
    ) -> "RoutingExperimentV2AdapterFactory":
        """Build handles from the model-authority subprocess launcher.

        ``worker_factory`` must launch a fresh model worker for each distinct
        artifact and return the worker's observed artifact key.  Returning a
        plain in-process adapter is rejected, so the Lab cannot accidentally
        mix model roots while retaining a convenient fake-adapter seam for
        contract tests.
        """

        handles: dict[str, IsolatedRoutingAdapter] = {}
        handles_by_artifact: dict[str, IsolatedRoutingAdapter] = {}
        for variant in variants:
            artifact_key = _v2_variant_artifact_key(variant)
            handle = handles_by_artifact.get(artifact_key)
            if handle is None:
                handle = worker_factory(variant.artifact)
                if not isinstance(handle, IsolatedRoutingAdapter):
                    raise RoutingExperimentError("v2_worker_factory_must_return_isolated_adapter")
                if handle.observed_artifact_key != artifact_key:
                    raise RoutingExperimentError(f"v2_worker_factory_artifact_mismatch:{variant.variant_id}")
                handles_by_artifact[artifact_key] = handle
            handles[variant.variant_id] = handle
        return cls(handles)

    def for_variants(
        self,
        variants: Sequence[RoutingExperimentV2Variant],
    ) -> Mapping[str, RoutingExperimentV2Adapter]:
        output: dict[str, RoutingExperimentV2Adapter] = {}
        for variant in variants:
            handle = self.handles.get(variant.variant_id)
            if handle is None:
                raise RoutingExperimentError(f"v2_isolated_adapter_missing:{variant.variant_id}")
            if handle.observed_artifact_key != _v2_variant_artifact_key(variant):
                raise RoutingExperimentError(f"v2_isolated_adapter_artifact_mismatch:{variant.variant_id}")
            output[variant.variant_id] = handle.adapter
        return output


def _v2_unwrap_adapters(
    adapters: Mapping[str, RoutingExperimentV2Adapter | IsolatedRoutingAdapter],
    *,
    variants: Sequence[RoutingExperimentV2Variant],
    require_isolation: bool,
) -> Mapping[str, RoutingExperimentV2Adapter]:
    if require_isolation and any(not isinstance(item, IsolatedRoutingAdapter) for item in adapters.values()):
        raise RoutingExperimentError("v2_isolated_model_adapters_required")
    output: dict[str, RoutingExperimentV2Adapter] = {}
    artifact_process: dict[str, str] = {}
    for variant in variants:
        handle_or_adapter = adapters.get(variant.variant_id)
        if handle_or_adapter is None:
            continue
        if isinstance(handle_or_adapter, IsolatedRoutingAdapter):
            expected_key = _v2_variant_artifact_key(variant)
            if handle_or_adapter.observed_artifact_key != expected_key:
                raise RoutingExperimentError(f"v2_isolated_adapter_artifact_mismatch:{variant.variant_id}")
            artifact_process.setdefault(expected_key, handle_or_adapter.process_id)
            if artifact_process[expected_key] != handle_or_adapter.process_id:
                raise RoutingExperimentError("v2_artifact_workers_must_be_stable")
            output[variant.variant_id] = handle_or_adapter.adapter
        else:
            output[variant.variant_id] = handle_or_adapter
    if len(set(artifact_process.values())) != len(artifact_process):
        raise RoutingExperimentError("v2_distinct_artifacts_require_distinct_workers")
    return output


def _v2_authority_manifest_dict(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize the signed PrivateModelArtifactManifest without copying it."""

    try:
        from research_lab.eval import PrivateModelArtifactManifest
    except Exception as exc:  # pragma: no cover - package import is always present in Lab
        raise RoutingExperimentError("v2_artifact_authority_contract_unavailable") from exc
    try:
        manifest = value if isinstance(value, PrivateModelArtifactManifest) else PrivateModelArtifactManifest.from_mapping(value)
        return manifest.to_dict()
    except Exception as exc:
        raise RoutingExperimentError("v2_artifact_authority_manifest_is_malformed") from exc


def _validate_v2_artifact_authority(
    variant: RoutingExperimentV2Variant,
) -> list[str]:
    """Check the existing signed manifest and its model-owned route hashes."""

    if variant.artifact_authority_manifest is None:
        return ["v2_artifact_authority_manifest_required"]
    errors: list[str] = []
    try:
        from research_lab.eval import validate_private_model_artifact_manifest

        manifest = _v2_authority_manifest_dict(variant.artifact_authority_manifest)
        errors.extend(validate_private_model_artifact_manifest(manifest))
    except RoutingExperimentError as exc:
        errors.append(str(exc))
        return sorted(set(errors))
    except Exception:
        errors.append("v2_artifact_authority_manifest_is_malformed")
        return sorted(set(errors))
    for artifact_field, manifest_field in (
        ("model_artifact_hash", "model_artifact_hash"),
        ("manifest_hash", "manifest_hash"),
        ("commit_sha", "git_commit_sha"),
    ):
        if str(manifest.get(manifest_field) or "").lower() != str(getattr(variant.artifact, artifact_field)).lower():
            errors.append(f"v2_artifact_authority_{artifact_field}_mismatch")
    extensions = manifest.get("signed_extensions")
    if not isinstance(extensions, Mapping):
        extensions = manifest
    for artifact_field in (
        "routing_contract_hash",
        "routing_catalog_hash",
        "routing_policy_hash",
        "feature_schema_hash",
        "verifier_contract_hash",
    ):
        observed = str(extensions.get(artifact_field) or "")
        if observed and observed != str(getattr(variant.artifact, artifact_field)):
            errors.append(f"v2_artifact_authority_{artifact_field}_mismatch")
        if not observed:
            errors.append(f"v2_artifact_authority_{artifact_field}_missing")
    return sorted(set(errors))


def _v2_normalize_source_manifest(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text):
        return text
    if _RAW_SHA256_RE.fullmatch(text):
        return f"sha256:{text}"
    raise RoutingExperimentError(f"{field_name} must be a source-add manifest digest")


def _validate_v2_source_add_provenance(
    provenance: SourceAddProvenance,
    *,
    descriptor: Mapping[str, Any],
    binding: ProviderBindingIdentity,
) -> list[str]:
    """Apply the existing provider-capabilities v3 request validator rules."""

    errors: list[str] = []
    request = provenance.request_payload
    if not isinstance(request, Mapping):
        return [f"v2_source_add_request_payload_missing:{provenance.tool_id}"]
    if str(request.get("schema_version") or "") != "leadpoet.routerverse_source_incorporation.v3":
        errors.append(f"v2_source_add_request_schema_mismatch:{provenance.tool_id}")
    if str(request.get("registration_symbol") or "") != "sourcing_model/routing/runtime.py::SOURCE_ADD_ROUTING_REGISTRATIONS":
        errors.append(f"v2_source_add_registration_symbol_mismatch:{provenance.tool_id}")
    if str(request.get("registration_type") or "") != "SourceAddRoutingRegistration":
        errors.append(f"v2_source_add_registration_type_mismatch:{provenance.tool_id}")
    if str(request.get("tool_id") or "") != provenance.tool_id:
        errors.append(f"v2_source_add_request_tool_mismatch:{provenance.tool_id}")
    if str(request.get("stage") or "") != provenance.stage:
        errors.append(f"v2_source_add_request_stage_mismatch:{provenance.tool_id}")
    if str(request.get("provider_id") or "") != binding.provider_id:
        errors.append(f"v2_source_add_provider_mismatch:{provenance.tool_id}")
    if str(request.get("runtime_binding_id") or "") != binding.provider_id:
        errors.append(f"v2_source_add_runtime_binding_mismatch:{provenance.tool_id}")
    try:
        from gateway.research_lab.provider_capabilities import _normalize_source_add_v8_registration, _source_add_binding_manifest

        values = {field: request.get(field) for field in (
            "provider_id", "stage", "revision", "manifest_sha256", "execution_mode", "priority",
            "capabilities", "idempotency", "cost_class", "unit_cost", "max_calls", "max_results",
            "timeout_seconds", "intent_categories", "evidence_types", "category_contracts",
            "binding_requirements", "best_for", "avoid_when", "best_for_description", "avoid_when_description",
        )}
        registration = _normalize_source_add_v8_registration(values)
        expected_binding_manifest = _source_add_binding_manifest(registration)
        if dict(request.get("binding_manifest") or {}) != expected_binding_manifest:
            errors.append(f"v2_source_add_binding_manifest_invalid:{provenance.tool_id}")
        expected_registration_manifest = _v2_normalize_source_manifest(registration.get("manifest_sha256"), "source_add_registration_manifest")
        if expected_registration_manifest != _v2_normalize_source_manifest(provenance.manifest_hash, "source_add_provenance_manifest"):
            errors.append(f"v2_source_add_provenance_manifest_mismatch:{provenance.tool_id}")
    except Exception:
        errors.append(f"v2_source_add_registration_invalid:{provenance.tool_id}")
    for field_name, value in (
        ("descriptor_manifest_hash", descriptor.get("manifest_hash")),
        ("provider_binding_manifest_hash", binding.manifest_hash),
    ):
        try:
            if _v2_normalize_source_manifest(value, field_name) != _v2_normalize_source_manifest(provenance.manifest_hash, "source_add_provenance_manifest"):
                errors.append(f"v2_source_add_{field_name}_mismatch:{provenance.tool_id}")
        except RoutingExperimentError as exc:
            errors.append(str(exc))
    if str(request.get("tool_id") or "") != str(descriptor.get("tool_id") or ""):
        errors.append(f"v2_source_add_descriptor_tool_mismatch:{provenance.tool_id}")
    if str(request.get("stage") or "") != str(descriptor.get("stage") or ""):
        errors.append(f"v2_source_add_descriptor_stage_mismatch:{provenance.tool_id}")
    if str(request.get("manifest_sha256") or "") != str(descriptor.get("request_manifest_sha256") or request.get("manifest_sha256") or ""):
        # The model descriptor may expose the registration digest under either
        # name; absence of a second copy is acceptable, disagreement is not.
        if descriptor.get("request_manifest_sha256"):
            errors.append(f"v2_source_add_request_manifest_mismatch:{provenance.tool_id}")
    return sorted(set(errors))


def _v2_variant_artifact_key(variant: RoutingExperimentV2Variant) -> str:
    return sha256_json(
        {
            "model_artifact_hash": variant.artifact.model_artifact_hash,
            "manifest_hash": variant.artifact.manifest_hash,
            "commit_sha": variant.artifact.commit_sha,
        }
    )


def validate_routing_experiment_v2_spec(
    spec: RoutingExperimentV2Spec | Mapping[str, Any],
    *,
    adapters: Mapping[str, RoutingExperimentV2Adapter],
) -> list[str]:
    """Validate V2 admission before any model or provider execution."""

    if not isinstance(spec, RoutingExperimentV2Spec):
        try:
            spec = RoutingExperimentV2Spec.from_mapping(spec)
        except Exception as exc:
            return [f"v2_experiment_spec_is_malformed:{type(exc).__name__}"]
    errors: list[str] = []
    if spec.contract_version != ROUTING_EXPERIMENT_V2_CONTRACT_VERSION:
        errors.append("v2_experiment_contract_version_mismatch")
    try:
        _ensure_safe_ref(spec.experiment_id, "v2_experiment_id")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    if spec.input.stage not in ROUTING_EXPERIMENT_STAGES:
        errors.append("v2_input_stage_is_invalid")
    if spec.receipt_execution_mode not in {mode.value for mode in ReceiptExecutionMode}:
        errors.append("v2_receipt_execution_mode_is_invalid")
    if not isinstance(spec.allow_live_credit_spend, bool):
        errors.append("v2_allow_live_credit_spend_must_be_boolean")
    if not spec.variants or len(spec.variants) > MAX_PROFILE_VARIANTS:
        errors.append("v2_variant_count_is_invalid")
    variant_ids = [item.variant_id for item in spec.variants]
    if len(variant_ids) != len(set(variant_ids)):
        errors.append("v2_variant_ids_must_be_unique")
    if spec.baseline_variant_id not in set(variant_ids):
        errors.append("v2_baseline_variant_is_missing")
    baseline_variant = next(
        (item for item in spec.variants if item.variant_id == spec.baseline_variant_id),
        None,
    )
    baseline_artifact_key = (
        _v2_variant_artifact_key(baseline_variant) if baseline_variant is not None else ""
    )
    if not isinstance(spec.availability, Mapping):
        errors.append("v2_availability_is_invalid")
    else:
        for variant_id, values in spec.availability.items():
            if variant_id not in set(variant_ids):
                errors.append(f"v2_availability_variant_is_unknown:{variant_id}")
            if not isinstance(values, Mapping):
                errors.append(f"v2_availability_variant_is_invalid:{variant_id}")
            else:
                for tool_id, available in values.items():
                    try:
                        _ensure_safe_ref(tool_id, "v2_availability_tool_id")
                    except RoutingExperimentError as exc:
                        errors.append(str(exc))
                    if not isinstance(available, bool):
                        errors.append(f"v2_availability_must_be_boolean:{variant_id}:{tool_id}")
    binding_by_id = {item.binding_id: item for item in spec.provider_bindings}
    if len(binding_by_id) != len(spec.provider_bindings):
        errors.append("v2_provider_binding_ids_must_be_unique")
    for variant_id, values in spec.availability.items() if isinstance(spec.availability, Mapping) else ():
        variant = next((item for item in spec.variants if item.variant_id == variant_id), None)
        variant_tools = {
            binding_by_id[binding_id].tool_id
            for binding_id in (variant.binding_ids if variant is not None else ())
            if binding_id in binding_by_id
        }
        for tool_id in values if isinstance(values, Mapping) else ():
            if tool_id not in variant_tools:
                errors.append(f"v2_availability_tool_is_not_bound:{variant_id}:{tool_id}")
    errors.extend(validate_experiment_credit_budget(spec.credit_budget, binding_ids=binding_by_id))
    for binding in spec.provider_bindings:
        errors.extend(validate_provider_binding_identity(binding))
    errors.extend(validate_routing_evaluation_gates(spec.gates))
    if spec.input.stage == "candidate_acquisition" and spec.gates.intent_release_policy_hash:
        # Candidate routes do not use the intent-release policy identity.  A
        # supplied value is accepted as metadata, but it cannot alter the
        # candidate gate semantics.
        pass
    try:
        _ensure_no_secret_material(spec.to_dict(), field_name="v2_experiment_spec")
    except RoutingExperimentError as exc:
        errors.append(str(exc))
    feature_set_errors: list[str] = []
    normalized_features = spec.input.feature_set_payload.get("features") if isinstance(spec.input.feature_set_payload, Mapping) else None
    if not isinstance(normalized_features, Sequence) or isinstance(normalized_features, (str, bytes)):
        feature_set_errors.append("v2_feature_set_payload_features_are_required")
        normalized_features = ()
    model_payload_by_variant: dict[str, Any] = {}
    routing_identity_by_variant: dict[str, str] = {}
    for variant in spec.variants:
        if variant.stage != spec.input.stage:
            errors.append(f"v2_variant_stage_mismatch:{variant.variant_id}")
        errors.extend(validate_sourcing_model_artifact_identity(variant.artifact))
        errors.extend(_validate_v2_artifact_authority(variant))
        variant_artifact_key = _v2_variant_artifact_key(variant)
        if variant.variant_id == spec.baseline_variant_id:
            if variant.change_kind != "route_only":
                errors.append(f"v2_baseline_change_kind_must_be_route_only:{variant.variant_id}")
            if variant.source_add_provenance or variant.new_tool_ids:
                errors.append(f"v2_baseline_must_not_add_tools:{variant.variant_id}")
        elif variant.change_kind == "route_only":
            if baseline_artifact_key and variant_artifact_key != baseline_artifact_key:
                errors.append(f"v2_route_only_artifact_must_match_baseline:{variant.variant_id}")
            if variant.source_add_provenance or variant.new_tool_ids:
                errors.append(f"v2_route_only_must_not_add_tools:{variant.variant_id}")
        elif variant.change_kind in {"tool_only", "tool_and_route"}:
            if baseline_artifact_key and variant_artifact_key == baseline_artifact_key:
                errors.append(f"v2_tool_variant_artifact_must_differ_from_baseline:{variant.variant_id}")
        adapter = adapters.get(variant.variant_id)
        if adapter is None:
            errors.append(f"v2_variant_adapter_missing:{variant.variant_id}")
            continue
        try:
            errors.extend(adapter.validate_artifact_identity(variant.artifact))
            feature_set = adapter.parse_feature_set(spec.input.feature_set_payload)
            errors.extend(
                adapter.validate_feature_set(
                    feature_set,
                    expected_hash=lab_hash_to_model(spec.input.feature_set_hash, "v2_feature_set_hash"),
                    expected_features=tuple(sorted(set(str(item) for item in normalized_features))),
                )
            )
            binding_items = [binding_by_id[item] for item in variant.binding_ids if item in binding_by_id]
            if len(binding_items) != len(variant.binding_ids):
                errors.append(f"v2_variant_binding_missing:{variant.variant_id}")
            binding_tool_ids = frozenset(item.tool_id for item in binding_items)
            source_lineages = {item.tool_id: item.source_lineage_id for item in binding_items}
            model_payload = adapter.parse_variant_payload(variant.routing_payload, stage=variant.stage)
            model_payload_by_variant[variant.variant_id] = model_payload
            errors.extend(
                adapter.validate_variant_payload(
                    model_payload,
                    stage=variant.stage,
                    feature_set=feature_set,
                    binding_tool_ids=binding_tool_ids,
                    binding_source_lineages=source_lineages,
                    expected_signal_type=spec.input.signal_type,
                )
            )
            try:
                routing_identity_by_variant[variant.variant_id] = adapter.routing_identity(
                    model_payload,
                    stage=variant.stage,
                )
            except Exception as exc:
                raise RoutingExperimentError("v2_model_routing_identity_unavailable") from exc
            descriptors = tuple(adapter.variant_tool_descriptors(model_payload, stage=variant.stage))
            descriptor_by_tool: dict[str, Mapping[str, Any]] = {}
            for descriptor in descriptors:
                tool_id = _ensure_safe_ref(descriptor.get("tool_id"), "v2_model_tool_id")
                if tool_id in descriptor_by_tool:
                    errors.append(f"v2_model_tool_descriptor_duplicate:{tool_id}")
                descriptor_by_tool[tool_id] = descriptor
                if str(descriptor.get("stage") or variant.stage) != variant.stage:
                    errors.append(f"v2_model_tool_stage_mismatch:{tool_id}")
                source_add = descriptor.get("source_add", False)
                if not isinstance(source_add, bool):
                    errors.append(f"v2_model_tool_source_add_must_be_boolean:{tool_id}")
                if source_add is True:
                    provenance = next((item for item in variant.source_add_provenance if item.tool_id == tool_id), None)
                    if provenance is None:
                        errors.append(f"v2_source_add_provenance_missing:{tool_id}")
                    else:
                        if provenance.stage != variant.stage:
                            errors.append(f"v2_source_add_provenance_stage_mismatch:{tool_id}")
                        if provenance.artifact_commit_sha != variant.artifact.commit_sha:
                            errors.append(f"v2_source_add_provenance_artifact_mismatch:{tool_id}")
                        binding = next((item for item in binding_items if item.tool_id == tool_id), None)
                        if binding is None:
                            errors.append(f"v2_source_add_binding_missing:{tool_id}")
                        else:
                            errors.extend(_validate_v2_source_add_provenance(provenance, descriptor=descriptor, binding=binding))
                        descriptor_request_hash = str(descriptor.get("request_hash") or "")
                        if descriptor_request_hash and descriptor_request_hash != provenance.request_hash:
                            errors.append(f"v2_source_add_request_mismatch:{tool_id}")
            for tool_id in variant.new_tool_ids:
                descriptor = descriptor_by_tool.get(tool_id)
                if descriptor is None:
                    descriptor = adapter.lookup_tool_descriptor(tool_id, stage=variant.stage)
                    if descriptor is not None:
                        descriptor_by_tool[tool_id] = descriptor
                if descriptor is None:
                    errors.append(f"v2_new_tool_absent_from_variant_artifact:{tool_id}")
                    continue
                source_add = descriptor.get("source_add", False)
                if not isinstance(source_add, bool):
                    errors.append(f"v2_model_tool_source_add_must_be_boolean:{tool_id}")
                if source_add is not True:
                    errors.append(f"v2_new_tool_is_not_source_add:{tool_id}")
                provenance = next((item for item in variant.source_add_provenance if item.tool_id == tool_id), None)
                binding = next((item for item in binding_items if item.tool_id == tool_id), None)
                if provenance is None:
                    errors.append(f"v2_source_add_provenance_missing:{tool_id}")
                elif binding is None:
                    errors.append(f"v2_source_add_binding_missing:{tool_id}")
                else:
                    errors.extend(_validate_v2_source_add_provenance(provenance, descriptor=descriptor, binding=binding))
            for provenance in variant.source_add_provenance:
                if provenance.tool_id not in descriptor_by_tool and provenance.tool_id not in set(variant.new_tool_ids):
                    errors.append(f"v2_source_add_tool_not_in_variant:{provenance.tool_id}")
            for binding_id in variant.binding_ids:
                binding = binding_by_id.get(binding_id)
                if binding is not None:
                    errors.extend(adapter.validate_provider_binding(binding, stage=variant.stage))
        except RoutingExperimentError as exc:
            errors.append(f"v2_variant_validation_failed:{variant.variant_id}:{exc}")
        except Exception as exc:
            errors.append(f"v2_variant_validation_failed:{variant.variant_id}:{type(exc).__name__}")
    baseline_identity = routing_identity_by_variant.get(spec.baseline_variant_id)
    if baseline_identity:
        for variant in spec.variants:
            if variant.variant_id == spec.baseline_variant_id:
                continue
            candidate_identity = routing_identity_by_variant.get(variant.variant_id)
            if not candidate_identity:
                continue
            try:
                if variant.change_kind == "tool_only":
                    candidate_identity = adapters[variant.variant_id].routing_identity(
                        model_payload_by_variant[variant.variant_id],
                        stage=variant.stage,
                        exclude_tool_ids=variant.new_tool_ids,
                    )
                    if candidate_identity != baseline_identity:
                        errors.append(f"v2_tool_only_routing_identity_mismatch:{variant.variant_id}")
                elif variant.change_kind == "tool_and_route":
                    candidate_identity = adapters[variant.variant_id].routing_identity(
                        model_payload_by_variant[variant.variant_id],
                        stage=variant.stage,
                        exclude_tool_ids=variant.new_tool_ids,
                    )
                    if candidate_identity == baseline_identity:
                        errors.append(f"v2_tool_and_route_routing_identity_must_differ:{variant.variant_id}")
            except Exception as exc:
                errors.append(f"v2_variant_routing_identity_comparison_failed:{variant.variant_id}:{type(exc).__name__}")
    return sorted(set(errors))


@dataclass(frozen=True)
class RoutingDecisionReceiptV2:
    """Redacted per-unit route decision and provider outcome projection."""

    receipt_id: str
    experiment_id: str
    variant_id: str
    artifact_key: str
    stage: str
    unit_ref: str
    plan_hash: str
    route_hash: str
    attempted_tool_ids: tuple[str, ...]
    skipped_tool_reasons: tuple[tuple[str, str], ...]
    outcome_reasons: tuple[tuple[str, str], ...]
    provider_receipt_refs: tuple[str, ...]
    total_credit_microunits: int
    latency_ms: int
    execution_mode: str
    immutable: bool = True
    contract_version: str = ROUTING_DECISION_RECEIPT_V2_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "receipt_id": self.receipt_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "artifact_key": self.artifact_key,
            "stage": self.stage,
            "unit_ref": self.unit_ref,
            "plan_hash": self.plan_hash,
            "route_hash": self.route_hash,
            "attempted_tool_ids": list(self.attempted_tool_ids),
            "skipped_tool_reasons": [list(item) for item in self.skipped_tool_reasons],
            "outcome_reasons": [list(item) for item in self.outcome_reasons],
            "provider_receipt_refs": list(self.provider_receipt_refs),
            "total_credit_microunits": self.total_credit_microunits,
            "latency_ms": self.latency_ms,
            "execution_mode": self.execution_mode,
            "immutable": self.immutable,
        }


class RoutingDecisionReceiptStore:
    """Append-only decision receipt store, separate from provider receipts."""

    def __init__(self) -> None:
        self._rows: dict[str, RoutingDecisionReceiptV2] = {}

    def put(self, receipt: RoutingDecisionReceiptV2) -> RoutingDecisionReceiptV2:
        if receipt.receipt_id in self._rows and self._rows[receipt.receipt_id].to_dict() != receipt.to_dict():
            raise RoutingExperimentError("v2_decision_receipt_id_collision")
        self._rows[receipt.receipt_id] = receipt
        return receipt

    def get(self, receipt_id: str) -> RoutingDecisionReceiptV2 | None:
        return self._rows.get(str(receipt_id))

    def values(self) -> tuple[RoutingDecisionReceiptV2, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))


@dataclass(frozen=True)
class RoutingExperimentV2VariantEvaluation:
    variant_id: str
    artifact_key: str
    stage: str
    calibration: RoutingEvaluationMetrics
    holdout: RoutingEvaluationMetrics
    passed_precision_gate: bool
    passed_recall_gate: bool
    passed_cost_gate: bool
    passed_efficiency_gate: bool
    passed: bool
    decision_receipt_refs: tuple[str, ...]
    provider_receipt_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["calibration"] = self.calibration.to_dict()
        data["holdout"] = self.holdout.to_dict()
        data["decision_receipt_refs"] = list(self.decision_receipt_refs)
        data["provider_receipt_refs"] = list(self.provider_receipt_refs)
        return data


@dataclass(frozen=True)
class RoutingExperimentV2Evaluation:
    receipt_id: str
    experiment_id: str
    experiment_hash: str
    variants: tuple[RoutingExperimentV2VariantEvaluation, ...]
    baseline_variant_id: str
    selected_variant_id: str
    decision_receipt_refs: tuple[str, ...]
    provider_receipt_refs: tuple[str, ...]
    provider_cache_hits: int
    provider_cache_misses: int
    billing_rollup_id: str = ""
    billing_rollup_hash: str = ""
    billing_rollup_total_credit_microunits: int = 0
    live_credit_spend: bool = False
    immutable: bool = True
    contract_version: str = ROUTING_EXPERIMENT_V2_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "receipt_id": self.receipt_id,
            "experiment_id": self.experiment_id,
            "experiment_hash": self.experiment_hash,
            "variants": [item.to_dict() for item in self.variants],
            "baseline_variant_id": self.baseline_variant_id,
            "selected_variant_id": self.selected_variant_id,
            "decision_receipt_refs": list(self.decision_receipt_refs),
            "provider_receipt_refs": list(self.provider_receipt_refs),
            "provider_cache_hits": self.provider_cache_hits,
            "provider_cache_misses": self.provider_cache_misses,
            "billing_rollup_id": self.billing_rollup_id,
            "billing_rollup_hash": self.billing_rollup_hash,
            "billing_rollup_total_credit_microunits": self.billing_rollup_total_credit_microunits,
            "live_credit_spend": self.live_credit_spend,
            "immutable": self.immutable,
        }


def _v2_labels(
    experiment_input: RoutingExperimentV2Input,
    gold_labels: Mapping[str, bool],
) -> None:
    if not isinstance(gold_labels, Mapping):
        raise RoutingExperimentError("v2_gold_labels_must_be_an_object")
    if any(not isinstance(key, str) or not isinstance(value, bool) for key, value in gold_labels.items()):
        raise RoutingExperimentError("v2_gold_labels_must_use_boolean_values")
    units = set(experiment_input.calibration_unit_refs).union(experiment_input.holdout_unit_refs)
    if set(gold_labels) != units:
        raise RoutingExperimentError("v2_gold_labels_must_cover_exact_units")
    normalized = dict(gold_labels)
    if sha256_json({"labels": sorted(normalized.items())}) != experiment_input.gold_label_set_hash:
        raise RoutingExperimentError("v2_gold_label_set_hash_mismatch")


def _v2_projection_pairs(value: Any, field_name: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise RoutingExperimentError(f"v2_{field_name}_must_be_an_object")
    pairs: list[tuple[str, str]] = []
    for key, reason in value.items():
        pairs.append((_ensure_safe_ref(key, f"v2_{field_name}_key"), _ensure_safe_ref(reason, f"v2_{field_name}_reason")))
    return tuple(pairs)


def _v2_failure_receipt(
    *,
    binding: ProviderBindingIdentity,
    unit_ref: str,
    request_fingerprint: str,
    execution_mode: str,
    error: BaseException,
) -> ProviderReceipt:
    """Turn an adapter/timeout failure into redacted, fail-closed evidence."""

    payload = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "unit_ref": unit_ref,
        "request_fingerprint": request_fingerprint,
        "outcome": ProviderOutcome.ADAPTER_FAILURE.value,
        "error_type": type(error).__name__,
    }
    identity = {
        "binding_id": binding.binding_id,
        "tool_id": binding.tool_id,
        "binding_version": binding.adapter_version,
        "source_lineage_id": binding.source_lineage_id,
        "unit_ref": unit_ref,
        "request_fingerprint": request_fingerprint,
        "outcome": ProviderOutcome.ADAPTER_FAILURE.value,
        "evidence_hash": sha256_json(payload),
        "credit_microunits": 0,
        "latency_ms": 0,
        "execution_mode": execution_mode,
    }
    return ProviderReceipt(
        receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16],
        **identity,
    )


def _v2_runner_accepts_authorization(
    runner: Callable[..., ProviderReceipt | Mapping[str, Any]],
) -> bool:
    try:
        parameters = tuple(inspect.signature(runner).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        item for item in parameters
        if item.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters) or len(positional) >= 4


def _v2_call_runner(
    runner: Callable[..., ProviderReceipt | Mapping[str, Any]],
    binding: ProviderBindingIdentity,
    unit_ref: str,
    request_fingerprint: str,
    authorization: RoutingCallAuthorization,
) -> ProviderReceipt | Mapping[str, Any]:
    """Pass typed authorization when supported without breaking v1 seams."""

    try:
        signature = inspect.signature(runner)
        positional = [
            item
            for item in signature.parameters.values()
            if item.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        accepts_varargs = any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in signature.parameters.values())
    except (TypeError, ValueError):
        positional = []
        accepts_varargs = True
    if accepts_varargs or len(positional) >= 4:
        return runner(binding, unit_ref, request_fingerprint, authorization)
    if len(positional) >= 3:
        return runner(binding, unit_ref, request_fingerprint)
    return runner(binding, unit_ref)


def _v2_run_unit(
    spec: RoutingExperimentV2Spec,
    variant: RoutingExperimentV2Variant,
    *,
    adapter: RoutingExperimentV2Adapter,
    feature_set: Any,
    bindings: Mapping[str, ProviderBindingIdentity],
    store: ProviderReceiptStore,
    decision_store: RoutingDecisionReceiptStore,
    runner: Callable[[ProviderBindingIdentity, str], ProviderReceipt | Mapping[str, Any]],
    unit_ref: str,
    spent_total: list[int],
    spent_by_binding: dict[str, int],
    new_spend_total: list[int],
    reserved_total: list[int],
    reserved_by_binding: dict[str, int],
) -> tuple[list[ProviderReceipt], bool, RoutingDecisionReceiptV2]:
    variant_bindings: dict[str, ProviderBindingIdentity] = {}
    for binding_id in variant.binding_ids:
        binding = bindings.get(binding_id)
        if binding is None:
            raise RoutingExperimentError(f"v2_variant_binding_missing:{binding_id}")
        variant_bindings[binding.tool_id] = binding
    availability = spec.availability.get(variant.variant_id, {})
    if not isinstance(availability, Mapping):
        raise RoutingExperimentError(f"v2_availability_variant_is_invalid:{variant.variant_id}")
    available_tools: dict[str, bool] = {}
    for tool_id in variant_bindings:
        value = availability.get(tool_id, True)
        if not isinstance(value, bool):
            raise RoutingExperimentError(f"v2_availability_must_be_boolean:{variant.variant_id}:{tool_id}")
        available_tools[tool_id] = value
    model_payload = adapter.parse_variant_payload(variant.routing_payload, stage=variant.stage)
    plan = adapter.compile_variant(
        model_payload,
        stage=variant.stage,
        feature_set=feature_set,
        available_tools=available_tools,
        remaining_seconds=MAX_LATENCY_MS / 1000,
        remaining_calls=MAX_TOOLS_PER_VARIANT,
        credit_cap=spec.credit_budget.total_credit_microunits / 1_000_000,
        expected_signal_type=spec.input.signal_type,
    )
    payload = adapter.plan_as_payload(plan)
    if not isinstance(payload, Mapping):
        raise RoutingExperimentError("v2_model_plan_payload_is_not_an_object")
    try:
        parsed = adapter.parse_plan(payload)
        if adapter.plan_hash(parsed) != adapter.plan_hash(plan):
            raise RoutingExperimentError("v2_model_plan_hash_is_not_stable")
    except RoutingExperimentError:
        raise
    except Exception as exc:
        raise RoutingExperimentError("v2_model_plan_is_malformed") from exc
    plan_hash = adapter.plan_hash(plan)
    route_hash = adapter.route_hash(plan)
    step_budgets = tuple(adapter.plan_step_budgets(plan) or ())
    budget_by_tool: dict[str, RoutingPlanStepBudget] = {}
    for budget in step_budgets:
        if budget.tool_id not in variant_bindings:
            raise RoutingExperimentError(f"v2_model_plan_references_unbound_tool:{budget.tool_id}")
        if budget.execution_mode == "invoke":
            if budget.tool_id in budget_by_tool:
                raise RoutingExperimentError(f"v2_model_plan_repeats_tool_step:{budget.tool_id}")
            budget_by_tool[budget.tool_id] = budget
    planned_call_cap = sum(item.max_calls for item in budget_by_tool.values())
    planned_seconds_cap = sum(
        item.timeout_seconds for item in budget_by_tool.values() if item.max_calls > 0
    )
    if planned_call_cap > MAX_TOOLS_PER_VARIANT:
        raise RoutingExperimentError("v2_model_plan_total_call_cap_exceeded")
    if planned_seconds_cap > MAX_LATENCY_MS / 1000:
        raise RoutingExperimentError("v2_model_plan_total_time_cap_exceeded")
    planned_credit_by_tool = {
        item.tool_id: item.credit_microunits
        for item in budget_by_tool.values()
    }
    receipts: list[ProviderReceipt] = []
    attempt_by_tool: dict[str, int] = {}
    attempt_order: list[str] = []
    consumed_credit_by_tool: dict[str, int] = {}
    consumed_latency_ms = [0]

    def invoke(tool_id: str) -> ProviderReceipt:
        binding = variant_bindings.get(tool_id)
        if binding is None:
            raise RoutingExperimentError(f"v2_model_plan_references_unbound_tool:{tool_id}")
        budget = budget_by_tool.get(tool_id)
        if budget is None:
            raise RoutingExperimentError(f"v2_model_plan_invoked_unplanned_tool:{tool_id}")
        if available_tools.get(tool_id) is not True:
            raise RoutingExperimentError(f"v2_model_plan_invoked_unavailable_tool:{tool_id}")
        attempt = attempt_by_tool.get(tool_id, 0)
        if attempt >= budget.max_calls:
            raise RoutingExperimentError(f"v2_model_plan_max_calls_exceeded:{tool_id}")
        if sum(attempt_by_tool.values()) >= planned_call_cap:
            raise RoutingExperimentError("v2_model_plan_total_call_cap_exceeded")
        if planned_seconds_cap > 0 and consumed_latency_ms[0] >= int(math.ceil(planned_seconds_cap * 1000)):
            raise RoutingExperimentError("v2_model_plan_total_time_cap_exceeded")
        attempt_by_tool[tool_id] = attempt + 1
        attempt_order.append(tool_id)
        request_fingerprint = sha256_json(
            {
                "contract_version": ROUTING_EXPERIMENT_V2_CONTRACT_VERSION,
                "experiment_id": spec.experiment_id,
                "experiment_hash": spec.experiment_hash(),
                "variant_id": variant.variant_id,
                "artifact_key": _v2_variant_artifact_key(variant),
                "stage": variant.stage,
                "plan_hash": plan_hash,
                "route_hash": route_hash,
                "tool_id": tool_id,
                "unit_ref": unit_ref,
                "attempt": attempt,
            }
        )
        key = provider_receipt_key(
            tool_id=tool_id,
            binding_version=binding.adapter_version,
            request_fingerprint=request_fingerprint,
        )
        receipt = store.get(key)
        if receipt is None:
            try:
                remaining_credit = max(
                    0,
                    min(
                        planned_credit_by_tool[tool_id] - consumed_credit_by_tool.get(tool_id, 0),
                        spec.credit_budget.total_credit_microunits - spent_total[0],
                    ),
                )
                remaining_timeout = max(
                    0,
                    min(
                        int(math.ceil(budget.timeout_seconds * 1000)),
                        int(math.ceil(planned_seconds_cap * 1000)) - consumed_latency_ms[0],
                    ),
                )
                authorization = RoutingCallAuthorization(
                    experiment_id=spec.experiment_id,
                    variant_id=variant.variant_id,
                    artifact_key=_v2_variant_artifact_key(variant),
                    stage=variant.stage,
                    unit_ref=unit_ref,
                    tool_id=tool_id,
                    attempt=attempt,
                    request_fingerprint=request_fingerprint,
                    remaining_credit_microunits=remaining_credit,
                    timeout_ceiling_ms=remaining_timeout,
                    execution_mode=spec.receipt_execution_mode,
                )
                value = _v2_call_runner(
                    runner,
                    binding,
                    unit_ref,
                    request_fingerprint,
                    authorization,
                )
                receipt = _receipt_for_runner_result(
                    value,
                    binding=binding,
                    unit_ref=unit_ref,
                    request_fingerprint=request_fingerprint,
                )
            except Exception as exc:
                receipt = _v2_failure_receipt(
                    binding=binding,
                    unit_ref=unit_ref,
                    request_fingerprint=request_fingerprint,
                    execution_mode=spec.receipt_execution_mode,
                    error=exc,
                )
            receipt_errors = validate_provider_receipt(receipt)
            if receipt_errors:
                raise RoutingExperimentError("provider receipt invalid: " + "; ".join(receipt_errors))
            if receipt.execution_mode != spec.receipt_execution_mode:
                raise RoutingExperimentError("v2_runner_receipt_execution_mode_mismatch")
            if receipt.credit_microunits > planned_credit_by_tool[tool_id]:
                raise RoutingExperimentError(f"v2_provider_receipt_exceeds_reserved_credit:{tool_id}")
            if consumed_credit_by_tool.get(tool_id, 0) + receipt.credit_microunits > planned_credit_by_tool[tool_id]:
                raise RoutingExperimentError(f"v2_provider_receipt_exceeds_reserved_credit:{tool_id}")
            if receipt.latency_ms > int(math.ceil(budget.timeout_seconds * 1000)):
                raise RoutingExperimentError(f"v2_provider_receipt_exceeds_reserved_time:{tool_id}")
            if planned_seconds_cap > 0 and consumed_latency_ms[0] + receipt.latency_ms > int(math.ceil(planned_seconds_cap * 1000)):
                raise RoutingExperimentError("v2_provider_receipt_exceeds_total_time_cap")
            next_total = spent_total[0] + receipt.credit_microunits
            next_provider = spent_by_binding.get(binding.binding_id, 0) + receipt.credit_microunits
            if next_total > spec.credit_budget.total_credit_microunits:
                raise RoutingExperimentError("v2_total_credit_budget_exceeded")
            ceiling = spec.credit_budget.provider_credit_ceilings.get(binding.binding_id)
            if ceiling is not None and next_provider > ceiling:
                raise RoutingExperimentError(f"v2_provider_credit_budget_exceeded:{binding.binding_id}")
            store.put(key, receipt)
            new_spend_total[0] += receipt.credit_microunits
        else:
            if receipt.request_fingerprint != request_fingerprint:
                raise RoutingExperimentError("v2_cached_receipt_request_identity_mismatch")
            if receipt.execution_mode != spec.receipt_execution_mode:
                raise RoutingExperimentError("v2_cached_receipt_execution_mode_mismatch")
            if receipt.credit_microunits > planned_credit_by_tool[tool_id]:
                raise RoutingExperimentError(f"v2_cached_receipt_exceeds_reserved_credit:{tool_id}")
            if consumed_credit_by_tool.get(tool_id, 0) + receipt.credit_microunits > planned_credit_by_tool[tool_id]:
                raise RoutingExperimentError(f"v2_cached_receipt_exceeds_reserved_credit:{tool_id}")
        if receipt.latency_ms > int(math.ceil(budget.timeout_seconds * 1000)):
            raise RoutingExperimentError(f"v2_provider_receipt_exceeds_reserved_time:{tool_id}")
        if planned_seconds_cap > 0 and consumed_latency_ms[0] + receipt.latency_ms > int(math.ceil(planned_seconds_cap * 1000)):
            raise RoutingExperimentError("v2_provider_receipt_exceeds_total_time_cap")
        consumed_credit_by_tool[tool_id] = consumed_credit_by_tool.get(tool_id, 0) + receipt.credit_microunits
        consumed_latency_ms[0] += receipt.latency_ms
        spent_total[0] += receipt.credit_microunits
        spent_by_binding[binding.binding_id] = spent_by_binding.get(binding.binding_id, 0) + receipt.credit_microunits
        receipts.append(receipt)
        return receipt

    reservation: dict[str, int] = {}
    for budget in step_budgets:
        if budget.execution_mode != "invoke" or budget.max_calls <= 0:
            continue
        binding = variant_bindings.get(budget.tool_id)
        if binding is None:
            raise RoutingExperimentError(f"v2_model_plan_references_unbound_tool:{budget.tool_id}")
        reservation[binding.binding_id] = reservation.get(binding.binding_id, 0) + budget.credit_microunits
    requested_total = sum(reservation.values())
    if spent_total[0] + reserved_total[0] + requested_total > spec.credit_budget.total_credit_microunits:
        raise RoutingExperimentError("v2_total_credit_budget_would_be_exceeded")
    for binding_id, requested in reservation.items():
        ceiling = spec.credit_budget.provider_credit_ceilings.get(binding_id)
        if ceiling is not None and spent_by_binding.get(binding_id, 0) + reserved_by_binding.get(binding_id, 0) + requested > ceiling:
            raise RoutingExperimentError(f"v2_provider_credit_budget_would_be_exceeded:{binding_id}")
    reserved_total[0] += requested_total
    for binding_id, requested in reservation.items():
        reserved_by_binding[binding_id] = reserved_by_binding.get(binding_id, 0) + requested
    try:
        _results, predicted = adapter.execute_plan(plan, invoke)
    except RoutingExperimentError:
        raise
    except Exception as exc:
        # A model execution failure is recorded as an explicit route failure,
        # never converted into a positive signal or an unrecorded fallback.
        raise RoutingExperimentError(f"v2_model_plan_execution_failed:{type(exc).__name__}") from exc
    finally:
        reserved_total[0] -= requested_total
        for binding_id, requested in reservation.items():
            remaining = reserved_by_binding.get(binding_id, 0) - requested
            if remaining > 0:
                reserved_by_binding[binding_id] = remaining
            else:
                reserved_by_binding.pop(binding_id, None)
    try:
        projection = adapter.plan_decision_projection(plan)
    except Exception as exc:
        raise RoutingExperimentError("v2_model_plan_decision_projection_failed") from exc
    attempted = tuple(_ensure_safe_ref(item, "v2_attempted_tool_id") for item in attempt_order)
    planned_tools = tuple(item.tool_id for item in step_budgets if item.execution_mode == "invoke")
    skipped_projection = dict(projection.get("skipped_tool_reasons") or {})
    for tool_id in planned_tools:
        if tool_id not in attempt_by_tool:
            skipped_projection.setdefault(tool_id, "runtime_unavailable" if not available_tools.get(tool_id, True) else "model_route_stopped")
    skipped = _v2_projection_pairs(skipped_projection, "skipped_tool_reasons")
    # Provider receipts are the only authority for outcomes.  The model may
    # project skipped-route metadata, but it cannot invent or overwrite a
    # provider result for an attempted or unattempted tool.
    outcomes = tuple((item.tool_id, item.outcome) for item in receipts)
    decision_draft = RoutingDecisionReceiptV2(
        receipt_id="routing_decision:pending",
        experiment_id=spec.experiment_id,
        variant_id=variant.variant_id,
        artifact_key=_v2_variant_artifact_key(variant),
        stage=variant.stage,
        unit_ref=unit_ref,
        plan_hash=_v2_model_or_lab_hash(plan_hash, "v2_plan_hash"),
        route_hash=_v2_hash(route_hash, "v2_route_hash"),
        attempted_tool_ids=attempted,
        skipped_tool_reasons=skipped,
        outcome_reasons=outcomes,
        provider_receipt_refs=tuple(item.receipt_ref for item in receipts),
        total_credit_microunits=sum(item.credit_microunits for item in receipts),
        latency_ms=sum(item.latency_ms for item in receipts),
        execution_mode=spec.receipt_execution_mode,
    )
    decision_payload = decision_draft.to_dict()
    decision_payload["receipt_id"] = "routing_decision:" + sha256_json(decision_payload).split(":", 1)[1][:16]
    decision = replace(decision_draft, receipt_id=decision_payload["receipt_id"])
    decision_store.put(decision)
    return receipts, bool(predicted), decision


def _v2_metrics(
    *,
    split: str,
    unit_refs: Sequence[str],
    gold_labels: Mapping[str, bool],
    predictions: Mapping[str, bool],
    receipts_by_unit: Mapping[str, Sequence[ProviderReceipt]],
    baseline_positive_units: set[str],
) -> RoutingEvaluationMetrics:
    predicted_positive = sum(1 for unit_ref in unit_refs if predictions.get(unit_ref, False))
    true_positive = sum(1 for unit_ref in unit_refs if predictions.get(unit_ref, False) and gold_labels[unit_ref])
    false_positive = sum(1 for unit_ref in unit_refs if predictions.get(unit_ref, False) and not gold_labels[unit_ref])
    false_negative = sum(1 for unit_ref in unit_refs if not predictions.get(unit_ref, False) and gold_labels[unit_ref])
    verified = rejected = misses = failures = total_credit = no_signal_credit = rescues = rescue_credit = latency = receipt_count = overlap = 0
    for unit_ref in unit_refs:
        receipts = tuple(receipts_by_unit.get(unit_ref, ()))
        receipt_count += len(receipts)
        total_credit += sum(item.credit_microunits for item in receipts)
        latency += sum(item.latency_ms for item in receipts)
        lineages = [item.source_lineage_id for item in receipts]
        overlap += len(lineages) - len(set(lineages))
        for receipt in receipts:
            if receipt.outcome == ProviderOutcome.VERIFIED.value:
                verified += 1
            elif receipt.outcome == ProviderOutcome.REJECTED.value:
                rejected += 1
            elif receipt.outcome == ProviderOutcome.SOURCE_MISS.value:
                misses += 1
            elif receipt.outcome == ProviderOutcome.ADAPTER_FAILURE.value:
                failures += 1
        if not predictions.get(unit_ref, False):
            no_signal_credit += sum(item.credit_microunits for item in receipts)
        if predictions.get(unit_ref, False) and unit_ref not in baseline_positive_units:
            rescues += 1
            rescue_credit += sum(item.credit_microunits for item in receipts)
    positive_count = sum(1 for unit_ref in unit_refs if gold_labels[unit_ref])
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    recall = true_positive / positive_count if positive_count else 1.0
    return RoutingEvaluationMetrics(
        split=split,
        unit_count=len(unit_refs),
        predicted_positive_count=predicted_positive,
        true_positive_count=true_positive,
        false_positive_count=false_positive,
        false_negative_count=false_negative,
        verified_positive_count=verified,
        rejected_count=rejected,
        source_miss_count=misses,
        adapter_failure_count=failures,
        total_credit_microunits=total_credit,
        no_signal_credit_microunits=no_signal_credit,
        unique_rescue_count=rescues,
        unique_rescue_credit_microunits=rescue_credit,
        marginal_verified_positives_per_credit=(true_positive / total_credit if total_credit else 0.0),
        precision=round(precision, 8),
        recall=round(recall, 8),
        mean_latency_ms=round(latency / receipt_count, 8) if receipt_count else 0.0,
        source_lineage_overlap_count=overlap,
        source_lineage_overlap_rate=round(overlap / receipt_count, 8) if receipt_count else 0.0,
    )


def evaluate_routing_experiment_v2(
    spec: RoutingExperimentV2Spec | Mapping[str, Any],
    *,
    gold_labels: Mapping[str, bool],
    runner: Callable[..., ProviderReceipt | Mapping[str, Any]],
    adapters: Mapping[str, RoutingExperimentV2Adapter | IsolatedRoutingAdapter],
    receipt_store: ProviderReceiptStore | None = None,
    decision_store: RoutingDecisionReceiptStore | None = None,
    authoritative_billing_rollup: Callable[[ProviderReceiptStore], Mapping[str, Any]] | None = None,
    require_isolation: bool = True,
) -> RoutingExperimentV2Evaluation:
    """Evaluate route/tool combinations without changing a live model pointer."""

    if not isinstance(spec, RoutingExperimentV2Spec):
        spec = RoutingExperimentV2Spec.from_mapping(spec)
    adapters = _v2_unwrap_adapters(adapters, variants=spec.variants, require_isolation=require_isolation)
    errors = validate_routing_experiment_v2_spec(spec, adapters=adapters)
    if errors:
        raise RoutingExperimentError("invalid v2 routing experiment: " + "; ".join(errors))
    _v2_labels(spec.input, gold_labels)
    if spec.allow_live_credit_spend:
        if spec.receipt_execution_mode != ReceiptExecutionMode.MEASURED_LAB.value:
            raise RoutingExperimentError("v2_live_spend_requires_measured_lab_execution_mode")
        if not _v2_runner_accepts_authorization(runner):
            raise RoutingExperimentError("v2_live_spend_requires_authorization_aware_runner")
        if receipt_store is None or isinstance(receipt_store.repository, InMemoryProviderReceiptRepository):
            raise RoutingExperimentError("v2_live_spend_requires_durable_receipt_repository")
        if authoritative_billing_rollup is None:
            raise RoutingExperimentError("v2_live_spend_requires_authoritative_billing_rollup")
    elif spec.receipt_execution_mode == ReceiptExecutionMode.MEASURED_LAB.value:
        raise RoutingExperimentError("v2_measured_lab_receipts_require_explicit_live_spend")
    store = receipt_store if receipt_store is not None else ProviderReceiptStore()
    initial_cache_hits = store.cache_hits
    initial_cache_misses = store.cache_misses
    decisions = decision_store if decision_store is not None else RoutingDecisionReceiptStore()
    bindings = {item.binding_id: item for item in spec.provider_bindings}
    feature_sets: dict[str, Any] = {}
    for variant in spec.variants:
        feature_sets[variant.variant_id] = adapters[variant.variant_id].parse_feature_set(spec.input.feature_set_payload)
    spent_total = [0]
    spent_by_binding: dict[str, int] = {}
    new_spend_total = [0]
    reserved_total = [0]
    reserved_by_binding: dict[str, int] = {}
    per_variant_predictions: dict[str, dict[str, bool]] = {}
    per_variant_receipts: dict[str, dict[str, tuple[ProviderReceipt, ...]]] = {}
    per_variant_decisions: dict[str, list[str]] = {}
    per_variant_provider_refs: dict[str, set[str]] = {}
    baseline_predictions: dict[str, dict[str, bool]] = {"calibration": {}, "holdout": {}}
    ordered_units = tuple(spec.input.calibration_unit_refs) + tuple(spec.input.holdout_unit_refs)
    for variant in spec.variants:
        adapter = adapters[variant.variant_id]
        predictions: dict[str, bool] = {}
        receipts_by_unit: dict[str, tuple[ProviderReceipt, ...]] = {}
        decision_refs: list[str] = []
        provider_refs: set[str] = set()
        for unit_ref in ordered_units:
            receipts, predicted, decision = _v2_run_unit(
                spec,
                variant,
                adapter=adapter,
                feature_set=feature_sets[variant.variant_id],
                bindings=bindings,
                store=store,
                decision_store=decisions,
                runner=runner,
                unit_ref=unit_ref,
                spent_total=spent_total,
                spent_by_binding=spent_by_binding,
                new_spend_total=new_spend_total,
                reserved_total=reserved_total,
                reserved_by_binding=reserved_by_binding,
            )
            predictions[unit_ref] = predicted
            receipts_by_unit[unit_ref] = tuple(receipts)
            decision_refs.append(decision.receipt_id)
            provider_refs.update(item.receipt_ref for item in receipts)
            if variant.variant_id == spec.baseline_variant_id:
                split = "calibration" if unit_ref in spec.input.calibration_unit_refs else "holdout"
                baseline_predictions[split][unit_ref] = predicted
        per_variant_predictions[variant.variant_id] = predictions
        per_variant_receipts[variant.variant_id] = receipts_by_unit
        per_variant_decisions[variant.variant_id] = decision_refs
        per_variant_provider_refs[variant.variant_id] = provider_refs
    evaluations: list[RoutingExperimentV2VariantEvaluation] = []
    for variant in spec.variants:
        predictions = per_variant_predictions[variant.variant_id]
        receipts_by_unit = per_variant_receipts[variant.variant_id]
        calibration = _v2_metrics(
            split="calibration",
            unit_refs=spec.input.calibration_unit_refs,
            gold_labels=gold_labels,
            predictions=predictions,
            receipts_by_unit=receipts_by_unit,
            baseline_positive_units={unit for unit, value in baseline_predictions["calibration"].items() if value},
        )
        holdout = _v2_metrics(
            split="holdout",
            unit_refs=spec.input.holdout_unit_refs,
            gold_labels=gold_labels,
            predictions=predictions,
            receipts_by_unit=receipts_by_unit,
            baseline_positive_units={unit for unit, value in baseline_predictions["holdout"].items() if value},
        )
        passed_precision = calibration.precision >= spec.gates.min_calibration_precision and holdout.precision >= spec.gates.min_holdout_precision
        passed_recall = holdout.recall >= spec.gates.min_holdout_recall
        passed_cost = holdout.no_signal_credit_microunits <= spec.gates.max_holdout_no_signal_credit_microunits
        passed_efficiency = holdout.marginal_verified_positives_per_credit >= spec.gates.min_marginal_verified_positives_per_credit
        evaluations.append(
            RoutingExperimentV2VariantEvaluation(
                variant_id=variant.variant_id,
                artifact_key=_v2_variant_artifact_key(variant),
                stage=variant.stage,
                calibration=calibration,
                holdout=holdout,
                passed_precision_gate=passed_precision,
                passed_recall_gate=passed_recall,
                passed_cost_gate=passed_cost,
                passed_efficiency_gate=passed_efficiency,
                passed=passed_precision and passed_recall and passed_cost and passed_efficiency,
                decision_receipt_refs=tuple(sorted(set(per_variant_decisions[variant.variant_id]))),
                provider_receipt_refs=tuple(sorted(per_variant_provider_refs[variant.variant_id])),
            )
        )
    passing = [item for item in evaluations if item.passed]
    selected = ""
    if passing:
        selected = sorted(
            passing,
            key=lambda item: (
                item.holdout.total_credit_microunits,
                len(item.provider_receipt_refs),
                -item.holdout.unique_rescue_count,
                -item.holdout.recall,
                item.variant_id,
            ),
        )[0].variant_id
    billing_id = billing_hash = ""
    billing_total = 0
    if spec.allow_live_credit_spend:
        # The authoritative callback reports the new billing delta for this
        # evaluation, not the modeled cost of receipts already reused from
        # the exact measured-Lab cache.  A resumed all-cache evaluation must
        # therefore reconcile to zero new spend while retaining receipt costs
        # in its evaluation metrics.
        rollup = authoritative_billing_rollup(store) if authoritative_billing_rollup is not None else None
        if not isinstance(rollup, Mapping):
            raise RoutingExperimentError("v2_authoritative_billing_rollup_must_be_an_object")
        billing_id = _ensure_safe_ref(rollup.get("rollup_id"), "v2_billing_rollup_id")
        billing_hash = _ensure_hash(rollup.get("rollup_hash"), "v2_billing_rollup_hash")
        billing_total = int(rollup.get("total_credit_microunits", -1))
        if billing_total != new_spend_total[0]:
            raise RoutingExperimentError("v2_authoritative_billing_delta_mismatch")
    decision_refs = tuple(sorted({ref for item in evaluations for ref in item.decision_receipt_refs}))
    provider_refs = tuple(sorted({ref for item in evaluations for ref in item.provider_receipt_refs}))
    draft = RoutingExperimentV2Evaluation(
        receipt_id="routing_evaluation_v2:pending",
        experiment_id=spec.experiment_id,
        experiment_hash=spec.experiment_hash(),
        variants=tuple(evaluations),
        baseline_variant_id=spec.baseline_variant_id,
        selected_variant_id=selected,
        decision_receipt_refs=decision_refs,
        provider_receipt_refs=provider_refs,
        provider_cache_hits=store.cache_hits - initial_cache_hits,
        provider_cache_misses=store.cache_misses - initial_cache_misses,
        billing_rollup_id=billing_id,
        billing_rollup_hash=billing_hash,
        billing_rollup_total_credit_microunits=billing_total,
        live_credit_spend=spec.allow_live_credit_spend,
    )
    payload = draft.to_dict()
    return replace(
        draft,
        receipt_id="routing_evaluation_v2:" + sha256_json(payload).split(":", 1)[1][:16],
    )


def promote_routing_experiment_v2_to_lab(
    evaluation: RoutingExperimentV2Evaluation,
) -> str:
    """Return an immutable Lab reference; never promote a live pointer."""

    if not isinstance(evaluation, RoutingExperimentV2Evaluation):
        raise RoutingExperimentError("v2_evaluation_must_be_typed")
    if evaluation.live_credit_spend:
        raise RoutingExperimentError("v2_live_evaluation_cannot_auto_promote")
    if not evaluation.selected_variant_id:
        raise RoutingExperimentError("v2_evaluation_has_no_passing_variant")
    return sha256_json(
        {
            "contract_version": "leadpoet.routing_experiment_v2_lab_reference:v1",
            "evaluation_receipt_id": evaluation.receipt_id,
            "experiment_hash": evaluation.experiment_hash,
            "selected_variant_id": evaluation.selected_variant_id,
        }
    )


__all__ = [
    "MAX_CREDIT_MICROUNITS_PER_VARIANT",
    "MAX_PROFILE_VARIANTS",
    "ROUTING_EXPERIMENT_V2_CONTRACT_VERSION",
    "ROUTING_DECISION_RECEIPT_V2_VERSION",
    "ROUTING_EXPERIMENT_STAGES",
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
    "RoutingExperimentV2Input",
    "RoutingExperimentV2Variant",
    "RoutingExperimentV2Spec",
    "RoutingExperimentV2Adapter",
    "IsolatedRoutingAdapter",
    "RoutingExperimentV2AdapterFactory",
    "SourceAddProvenance",
    "RoutingDecisionReceiptV2",
    "RoutingDecisionReceiptStore",
    "RoutingExperimentV2VariantEvaluation",
    "RoutingExperimentV2Evaluation",
    "LabRoutingProfile",
    "RoutingAdmissionPlanAdapter",
    "RoutingPlanStepBudget",
    "RoutingCallAuthorization",
    "PinnedSourcingModelRoutingAdapter",
    "RoutingPromotionReceipt",
    "FrozenRoutingInput",
    "LLMRoutingProfileProposal",
    "SourcingModelArtifactIdentity",
    "VariantEvaluation",
    "admit_llm_routing_profile_proposal",
    "evaluate_routing_experiment",
    "evaluate_routing_experiment_v2",
    "promote_routing_profile_to_lab",
    "promote_routing_experiment_v2_to_lab",
    "provider_receipt_key",
    "select_smallest_passing_variant",
    "validate_frozen_routing_input",
    "validate_llm_routing_profile_proposal",
    "validate_model_routing_profile",
    "validate_provider_binding_identity",
    "validate_provider_receipt",
    "validate_routing_evaluation_receipt",
    "validate_routing_evaluation_gates",
    "validate_routing_experiment_spec",
    "validate_routing_experiment_v2_spec",
    "validate_experiment_credit_budget",
    "validate_sourcing_model_artifact_identity",
    "verify_lab_routing_artifact_lineage",
    "model_hash_to_lab",
    "lab_hash_to_model",
]
