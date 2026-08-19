"""Candidate-specific sidecars for the shared routing experiment contract.

PR 93 owns routing experiments, variants, budgets, provider receipts, route
decisions, evaluations, and Lab promotion. This module adds only the data
that is unique to a model-owned candidate waterfall: exact attempt receipt
projection, verified-company yield metrics, and an exact runtime preflight.

The adapter accepts only receipts that are already linked to the shared V2
experiment and decision contracts. It never compiles a route, calls a
provider, chooses a fallback, or creates a second promotion decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderOutcome,
    ProviderReceipt,
    ReceiptExecutionMode,
    RoutingDecisionReceiptV2,
    RoutingExperimentError,
    RoutingExperimentV2Evaluation,
    RoutingExperimentV2Spec,
    model_hash_to_lab,
    validate_provider_receipt,
    validate_routing_decision_receipt,
)


CANDIDATE_WATERFALL_RECEIPT_VERSION = (
    "leadpoet.candidate_waterfall_receipt_sidecar:v1"
)
CANDIDATE_WATERFALL_METRIC_VERSION = (
    "leadpoet.candidate_waterfall_metric_sidecar:v1"
)

_LAB_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODEL_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9_.:@/-]{1,300}$")
_CANDIDATE_TOOL_RE = re.compile(r"^candidate\.[A-Za-z0-9_.:-]{1,160}$")
_DECISION_RECEIPT_RE = re.compile(r"^routing_decision:[0-9a-f]{16}$")
_PROVIDER_RECEIPT_RE = re.compile(r"^provider_receipt:[0-9a-f]{16}$")
_EVALUATION_RECEIPT_RE = re.compile(r"^routing_evaluation_v2:[0-9a-f]{16}$")
_WATERFALL_RECEIPT_RE = re.compile(r"^candidate_waterfall:[0-9a-f]{24}$")
_DISPOSITIONS = frozenset({"succeeded", "missed", "failed", "deferred", "skipped"})


def _safe_ref(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_REF_RE.fullmatch(normalized):
        raise RoutingExperimentError(f"candidate_{field_name}_is_invalid")
    return normalized


def _lab_hash(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _LAB_HASH_RE.fullmatch(normalized):
        raise RoutingExperimentError(f"candidate_{field_name}_must_be_a_lab_sha256")
    return normalized


def _model_hash(value: Any, field_name: str, *, optional: bool = False) -> str:
    normalized = str(value or "").strip().lower()
    if optional and not normalized:
        return ""
    if not _MODEL_HASH_RE.fullmatch(normalized):
        raise RoutingExperimentError(f"candidate_{field_name}_must_be_a_model_sha256")
    return normalized


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise RoutingExperimentError(f"candidate_{field_name}_must_be_a_nonnegative_integer")
    return value


def _candidate_variant(spec: RoutingExperimentV2Spec, variant_id: str) -> Any:
    if not isinstance(spec, RoutingExperimentV2Spec):
        raise RoutingExperimentError("candidate_experiment_must_use_routing_experiment_v2_spec")
    if spec.input.stage != "candidate_acquisition":
        raise RoutingExperimentError("candidate_experiment_stage_must_be_candidate_acquisition")
    matches = [item for item in spec.variants if item.variant_id == variant_id]
    if len(matches) != 1:
        raise RoutingExperimentError("candidate_variant_must_exist_exactly_once")
    variant = matches[0]
    if variant.stage != "candidate_acquisition":
        raise RoutingExperimentError("candidate_variant_stage_must_be_candidate_acquisition")
    return variant


def validate_candidate_routing_model_runtime(
    *,
    spec: RoutingExperimentV2Spec,
    variant_id: str,
    model_runtime: Any,
) -> Mapping[str, Any]:
    """Fail closed unless the exact variant Model contract is available."""

    variant = _candidate_variant(spec, variant_id)
    required_callables = (
        "candidate_waterfall_execution_contract_identity",
        "compile_candidate_stop_policy",
        "compile_profiled_candidate_acquisition_route",
        "evaluate_candidate_waterfall",
        "runtime_catalog",
        "runtime_policy",
        "runtime_tool_definitions",
    )
    if any(not callable(getattr(model_runtime, name, None)) for name in required_callables):
        raise RoutingExperimentError("model_candidate_routing_runtime_contract_is_incomplete")
    receipt_type = getattr(model_runtime, "CandidateStepAttemptReceipt", None)
    if not callable(getattr(receipt_type, "from_payload", None)):
        raise RoutingExperimentError("model_candidate_routing_receipt_parser_is_unavailable")
    try:
        identity = model_runtime.candidate_waterfall_execution_contract_identity()
    except Exception as exc:
        raise RoutingExperimentError("model_candidate_routing_identity_is_unavailable") from exc
    if not isinstance(identity, Mapping):
        raise RoutingExperimentError("model_candidate_routing_identity_is_invalid")
    actual_raw_hash = _model_hash(identity.get("contract_sha256"), "routing_contract_hash")
    expected_lab_hash = _lab_hash(
        variant.artifact.routing_contract_hash,
        "artifact_routing_contract_hash",
    )
    if model_hash_to_lab(actual_raw_hash, "candidate_routing_contract_hash") != expected_lab_hash:
        raise RoutingExperimentError("model_candidate_routing_contract_hash_differs_from_variant")
    if identity.get("provider_results_can_satisfy_target") is not False:
        raise RoutingExperimentError("model_candidate_routing_stop_contract_is_unsafe")
    for field_name in (
        "contract_version",
        "attempt_receipt_schema_version",
        "stop_policy_schema_version",
        "progress_schema_version",
        "stop_metric",
        "attempt_sequence",
        "step_sequence",
        "retry_precondition",
    ):
        _safe_ref(identity.get(field_name), f"model_identity_{field_name}")
    for field_name in ("attempt_dispositions", "decisions"):
        values = identity.get(field_name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) for item in values)
        ):
            raise RoutingExperimentError(
                f"model_candidate_routing_identity_{field_name}_is_invalid"
            )
    return dict(identity)


@dataclass(frozen=True)
class CandidateWaterfallReceipt:
    """Redacted candidate counts attached to shared routing receipts."""

    experiment_id: str
    experiment_hash: str
    variant_id: str
    artifact_key: str
    decision_receipt_id: str
    provider_receipt_ref: str
    unit_ref: str
    binding_id: str
    tool_id: str
    execution_mode: str
    provider_outcome: str
    decision_plan_hash: str
    decision_route_hash: str
    model_contract_sha256: str
    model_plan_sha256: str
    stop_policy_sha256: str
    attempt_receipt_sha256: str
    verification_receipt_sha256: str
    step_order: int
    attempt_sequence: int
    disposition: str
    outcome_code: str
    provider_call_count: int
    cost_microusd: int
    latency_ms: int
    raw_count: int
    normalized_count: int
    unique_count: int
    verified_qualified_count: int
    published_count: int = 0
    immutable: bool = True
    contract_version: str = CANDIDATE_WATERFALL_RECEIPT_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "variant_id",
            "decision_receipt_id",
            "unit_ref",
            "binding_id",
            "execution_mode",
            "provider_outcome",
            "outcome_code",
        ):
            _safe_ref(getattr(self, field_name), field_name)
        if not _CANDIDATE_TOOL_RE.fullmatch(str(self.tool_id or "")):
            raise RoutingExperimentError("candidate_tool_id_must_use_candidate_namespace")
        if not _DECISION_RECEIPT_RE.fullmatch(self.decision_receipt_id):
            raise RoutingExperimentError("candidate_decision_receipt_id_is_invalid")
        if self.provider_receipt_ref and not _PROVIDER_RECEIPT_RE.fullmatch(
            self.provider_receipt_ref
        ):
            raise RoutingExperimentError("candidate_provider_receipt_ref_is_invalid")
        if self.execution_mode not in {item.value for item in ReceiptExecutionMode}:
            raise RoutingExperimentError("candidate_execution_mode_is_invalid")
        if self.provider_outcome not in {
            *(item.value for item in ProviderOutcome),
            "skipped",
        }:
            raise RoutingExperimentError("candidate_provider_outcome_is_invalid")
        for field_name in (
            "experiment_hash",
            "artifact_key",
            "decision_plan_hash",
            "decision_route_hash",
        ):
            _lab_hash(getattr(self, field_name), field_name)
        for field_name in (
            "model_contract_sha256",
            "model_plan_sha256",
            "stop_policy_sha256",
            "attempt_receipt_sha256",
        ):
            _model_hash(getattr(self, field_name), field_name)
        _model_hash(
            self.verification_receipt_sha256,
            "verification_receipt_sha256",
            optional=True,
        )
        for field_name in (
            "step_order",
            "attempt_sequence",
            "provider_call_count",
            "cost_microusd",
            "latency_ms",
            "raw_count",
            "normalized_count",
            "unique_count",
            "verified_qualified_count",
            "published_count",
        ):
            _nonnegative_int(getattr(self, field_name), field_name)
        if self.disposition not in _DISPOSITIONS:
            raise RoutingExperimentError("candidate_disposition_is_invalid")
        if self.disposition == "skipped":
            if self.provider_receipt_ref or self.provider_outcome != "skipped":
                raise RoutingExperimentError(
                    "candidate_skipped_attempt_cannot_claim_provider_receipt"
                )
        elif not self.provider_receipt_ref:
            raise RoutingExperimentError(
                "candidate_attempt_requires_provider_receipt"
            )
        if not (
            self.raw_count
            >= self.normalized_count
            >= self.unique_count
            >= self.verified_qualified_count
            >= self.published_count
        ):
            raise RoutingExperimentError("candidate_counts_must_decrease_from_raw_to_published")
        if self.verified_qualified_count and not self.verification_receipt_sha256:
            raise RoutingExperimentError("candidate_verified_count_requires_verification_receipt")
        if self.immutable is not True:
            raise RoutingExperimentError("candidate_waterfall_receipt_must_be_immutable")
        if self.contract_version != CANDIDATE_WATERFALL_RECEIPT_VERSION:
            raise RoutingExperimentError("candidate_waterfall_receipt_version_is_invalid")

    def identity_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_hash(self) -> str:
        return sha256_json(self.identity_payload())

    @property
    def receipt_id(self) -> str:
        return "candidate_waterfall:" + self.receipt_hash.split(":", 1)[1][:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
        }


def candidate_waterfall_receipt_from_model(
    *,
    spec: RoutingExperimentV2Spec,
    variant_id: str,
    decision_receipt: RoutingDecisionReceiptV2,
    provider_receipt: ProviderReceipt | None,
    receipt_payload: Mapping[str, Any],
    model_runtime: Any,
    published_count: int = 0,
) -> CandidateWaterfallReceipt:
    """Parse one exact Model receipt and bind it to PR 93 receipts."""

    variant = _candidate_variant(spec, variant_id)
    identity = validate_candidate_routing_model_runtime(
        spec=spec,
        variant_id=variant_id,
        model_runtime=model_runtime,
    )
    decision_errors = validate_routing_decision_receipt(decision_receipt)
    if decision_errors:
        raise RoutingExperimentError(
            "candidate_decision_receipt_is_invalid:" + ";".join(decision_errors)
        )
    if (
        decision_receipt.experiment_id != spec.experiment_id
        or decision_receipt.variant_id != variant_id
        or decision_receipt.stage != "candidate_acquisition"
    ):
        raise RoutingExperimentError("candidate_decision_receipt_lineage_differs_from_experiment")
    expected_artifact_key = sha256_json(
        {
            "model_artifact_hash": variant.artifact.model_artifact_hash,
            "manifest_hash": variant.artifact.manifest_hash,
            "commit_sha": variant.artifact.commit_sha,
        }
    )
    if decision_receipt.artifact_key != expected_artifact_key:
        raise RoutingExperimentError("candidate_decision_receipt_artifact_differs_from_variant")
    try:
        model_receipt = model_runtime.CandidateStepAttemptReceipt.from_payload(receipt_payload)
    except Exception as exc:
        raise RoutingExperimentError("model_candidate_attempt_receipt_is_invalid") from exc
    if (
        model_hash_to_lab(model_receipt.plan_sha256, "candidate_model_plan_hash")
        != decision_receipt.plan_hash
    ):
        raise RoutingExperimentError("model_candidate_attempt_plan_differs_from_decision")
    variant_bindings = [
        item
        for item in spec.provider_bindings
        if item.binding_id in set(variant.binding_ids)
    ]
    if provider_receipt is None:
        if model_receipt.disposition != "skipped":
            raise RoutingExperimentError(
                "model_candidate_attempt_without_provider_receipt_must_be_skipped"
            )
        skipped_by_tool = dict(decision_receipt.skipped_tool_reasons)
        if model_receipt.tool_id not in skipped_by_tool:
            raise RoutingExperimentError(
                "model_candidate_skipped_tool_is_not_in_decision"
            )
        matching_bindings = [
            item for item in variant_bindings if item.tool_id == model_receipt.tool_id
        ]
        if len(matching_bindings) != 1:
            raise RoutingExperimentError(
                "candidate_skipped_tool_binding_must_exist_exactly_once"
            )
        binding = matching_bindings[0]
        provider_receipt_ref = ""
        execution_mode = decision_receipt.execution_mode
        provider_outcome = "skipped"
    else:
        provider_errors = validate_provider_receipt(provider_receipt)
        if provider_errors:
            raise RoutingExperimentError(
                "candidate_provider_receipt_is_invalid:" + ";".join(provider_errors)
            )
        if model_receipt.disposition == "skipped":
            raise RoutingExperimentError(
                "model_candidate_skipped_attempt_cannot_claim_provider_receipt"
            )
        if provider_receipt.receipt_ref not in decision_receipt.provider_receipt_refs:
            raise RoutingExperimentError("candidate_provider_receipt_is_not_in_decision")
        if provider_receipt.unit_ref != decision_receipt.unit_ref:
            raise RoutingExperimentError(
                "candidate_provider_receipt_unit_differs_from_decision"
            )
        if provider_receipt.execution_mode != decision_receipt.execution_mode:
            raise RoutingExperimentError(
                "candidate_provider_receipt_mode_differs_from_decision"
            )
        if provider_receipt.tool_id not in decision_receipt.attempted_tool_ids:
            raise RoutingExperimentError("candidate_provider_tool_was_not_attempted")
        binding_by_id = {item.binding_id: item for item in variant_bindings}
        binding = binding_by_id.get(provider_receipt.binding_id)
        if binding is None:
            raise RoutingExperimentError("candidate_provider_binding_is_not_in_variant")
        if binding.tool_id != provider_receipt.tool_id:
            raise RoutingExperimentError(
                "candidate_provider_binding_tool_differs_from_receipt"
            )
        if (
            binding.adapter_version != provider_receipt.binding_version
            or binding.source_lineage_id != provider_receipt.source_lineage_id
        ):
            raise RoutingExperimentError(
                "candidate_provider_binding_identity_differs_from_receipt"
            )
        if model_receipt.tool_id != provider_receipt.tool_id:
            raise RoutingExperimentError(
                "model_candidate_attempt_tool_differs_from_provider_receipt"
            )
        provider_receipt_ref = provider_receipt.receipt_ref
        execution_mode = provider_receipt.execution_mode
        provider_outcome = provider_receipt.outcome
    published = _nonnegative_int(published_count, "published_count")
    if published > model_receipt.verified_qualified_count:
        raise RoutingExperimentError("candidate_published_count_exceeds_verified_count")
    return CandidateWaterfallReceipt(
        experiment_id=spec.experiment_id,
        experiment_hash=spec.experiment_hash(),
        variant_id=variant_id,
        artifact_key=decision_receipt.artifact_key,
        decision_receipt_id=decision_receipt.receipt_id,
        provider_receipt_ref=provider_receipt_ref,
        unit_ref=decision_receipt.unit_ref,
        binding_id=binding.binding_id,
        tool_id=model_receipt.tool_id,
        execution_mode=execution_mode,
        provider_outcome=provider_outcome,
        decision_plan_hash=decision_receipt.plan_hash,
        decision_route_hash=decision_receipt.route_hash,
        model_contract_sha256=str(identity["contract_sha256"]),
        model_plan_sha256=model_receipt.plan_sha256,
        stop_policy_sha256=model_receipt.stop_policy_sha256,
        attempt_receipt_sha256=model_receipt.sha256(),
        verification_receipt_sha256=model_receipt.verification_receipt_sha256,
        step_order=model_receipt.step_order,
        attempt_sequence=model_receipt.attempt,
        disposition=model_receipt.disposition,
        outcome_code=model_receipt.outcome_code,
        provider_call_count=model_receipt.provider_call_count,
        cost_microusd=round(model_receipt.estimated_cost_usd * 1_000_000),
        latency_ms=round(model_receipt.latency_seconds * 1_000),
        raw_count=model_receipt.raw_candidate_count,
        normalized_count=model_receipt.normalized_candidate_count,
        unique_count=model_receipt.unique_candidate_count,
        verified_qualified_count=model_receipt.verified_qualified_count,
        published_count=published,
    )


@dataclass(frozen=True)
class CandidateWaterfallMetric:
    """Candidate-only yield metrics; never a promotion decision."""

    evaluation_receipt_id: str
    experiment_id: str
    experiment_hash: str
    variant_id: str
    split: str
    target_verified_qualified_count: int
    unit_count: int
    fulfilled_unit_count: int
    waterfall_attempt_count: int
    provider_call_count: int
    total_cost_microusd: int
    total_latency_ms: int
    raw_count: int
    normalized_count: int
    unique_count: int
    verified_qualified_count: int
    published_count: int
    failed_attempt_count: int
    missed_attempt_count: int
    fulfillment_rate: float
    verification_rate: float
    publication_rate: float
    verified_qualified_per_usd: float
    waterfall_receipt_refs: tuple[str, ...]
    provider_receipt_refs: tuple[str, ...]
    decision_receipt_refs: tuple[str, ...]
    immutable: bool = True
    contract_version: str = CANDIDATE_WATERFALL_METRIC_VERSION

    def __post_init__(self) -> None:
        for field_name in ("evaluation_receipt_id", "experiment_id", "variant_id", "split"):
            _safe_ref(getattr(self, field_name), field_name)
        if not _EVALUATION_RECEIPT_RE.fullmatch(self.evaluation_receipt_id):
            raise RoutingExperimentError("candidate_evaluation_receipt_id_is_invalid")
        _lab_hash(self.experiment_hash, "experiment_hash")
        for field_name in (
            "target_verified_qualified_count",
            "unit_count",
            "fulfilled_unit_count",
            "waterfall_attempt_count",
            "provider_call_count",
            "total_cost_microusd",
            "total_latency_ms",
            "raw_count",
            "normalized_count",
            "unique_count",
            "verified_qualified_count",
            "published_count",
            "failed_attempt_count",
            "missed_attempt_count",
        ):
            _nonnegative_int(getattr(self, field_name), field_name)
        if self.target_verified_qualified_count < 1:
            raise RoutingExperimentError("candidate_metric_target_must_be_positive")
        if self.unit_count < 1:
            raise RoutingExperimentError("candidate_metric_unit_count_must_be_positive")
        if self.split not in {"calibration", "holdout"}:
            raise RoutingExperimentError("candidate_metric_split_is_invalid")
        if self.fulfilled_unit_count > self.unit_count:
            raise RoutingExperimentError("candidate_metric_fulfilled_units_exceed_units")
        if not (
            self.raw_count
            >= self.normalized_count
            >= self.unique_count
            >= self.verified_qualified_count
            >= self.published_count
        ):
            raise RoutingExperimentError(
                "candidate_metric_counts_must_decrease_from_raw_to_published"
            )
        if self.failed_attempt_count + self.missed_attempt_count > self.waterfall_attempt_count:
            raise RoutingExperimentError(
                "candidate_metric_failed_and_missed_attempts_exceed_attempts"
            )
        for field_name, refs in (
            ("waterfall_receipt_refs", self.waterfall_receipt_refs),
            ("provider_receipt_refs", self.provider_receipt_refs),
            ("decision_receipt_refs", self.decision_receipt_refs),
        ):
            if not isinstance(refs, tuple) or len(set(refs)) != len(refs):
                raise RoutingExperimentError(f"candidate_metric_{field_name}_is_invalid")
            for value in refs:
                _safe_ref(value, f"metric_{field_name}")
        if any(
            not _WATERFALL_RECEIPT_RE.fullmatch(value)
            for value in self.waterfall_receipt_refs
        ):
            raise RoutingExperimentError(
                "candidate_metric_waterfall_receipt_ref_is_invalid"
            )
        if any(
            not _PROVIDER_RECEIPT_RE.fullmatch(value)
            for value in self.provider_receipt_refs
        ):
            raise RoutingExperimentError(
                "candidate_metric_provider_receipt_ref_is_invalid"
            )
        if any(
            not _DECISION_RECEIPT_RE.fullmatch(value)
            for value in self.decision_receipt_refs
        ):
            raise RoutingExperimentError(
                "candidate_metric_decision_receipt_ref_is_invalid"
            )
        for field_name in (
            "fulfillment_rate",
            "verification_rate",
            "publication_rate",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise RoutingExperimentError(f"candidate_metric_{field_name}_is_invalid")
        if (
            isinstance(self.verified_qualified_per_usd, bool)
            or not isinstance(self.verified_qualified_per_usd, (int, float))
            or not math.isfinite(self.verified_qualified_per_usd)
            or self.verified_qualified_per_usd < 0
        ):
            raise RoutingExperimentError(
                "candidate_metric_verified_qualified_per_usd_is_invalid"
            )
        if (
            self.immutable is not True
            or self.contract_version != CANDIDATE_WATERFALL_METRIC_VERSION
        ):
            raise RoutingExperimentError("candidate_waterfall_metric_contract_is_invalid")

    def identity_payload(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in (
            "waterfall_receipt_refs",
            "provider_receipt_refs",
            "decision_receipt_refs",
        ):
            data[field_name] = list(getattr(self, field_name))
        return data

    @property
    def metric_hash(self) -> str:
        return sha256_json(self.identity_payload())

    @property
    def metric_id(self) -> str:
        return "candidate_metric:" + self.metric_hash.split(":", 1)[1][:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "metric_id": self.metric_id,
            "metric_hash": self.metric_hash,
        }


def evaluate_candidate_waterfall_metrics(
    *,
    spec: RoutingExperimentV2Spec,
    evaluation: RoutingExperimentV2Evaluation,
    receipts: Sequence[CandidateWaterfallReceipt],
    target_verified_qualified_count: int,
) -> tuple[CandidateWaterfallMetric, ...]:
    """Derive candidate yield sidecars from a shared V2 evaluation."""

    if not isinstance(evaluation, RoutingExperimentV2Evaluation):
        raise RoutingExperimentError("candidate_metrics_require_routing_experiment_v2_evaluation")
    for variant in spec.variants:
        _candidate_variant(spec, variant.variant_id)
    if (
        evaluation.experiment_id != spec.experiment_id
        or evaluation.experiment_hash != spec.experiment_hash()
    ):
        raise RoutingExperimentError("candidate_evaluation_lineage_differs_from_experiment")
    target = _nonnegative_int(target_verified_qualified_count, "target_verified_qualified_count")
    if target < 1:
        raise RoutingExperimentError("candidate_metric_target_must_be_positive")
    evaluation_by_variant = {item.variant_id: item for item in evaluation.variants}
    if set(evaluation_by_variant) != {item.variant_id for item in spec.variants}:
        raise RoutingExperimentError("candidate_evaluation_variants_differ_from_experiment")
    receipt_keys: set[tuple[str, str, int, int]] = set()
    for receipt in receipts:
        if not isinstance(receipt, CandidateWaterfallReceipt):
            raise RoutingExperimentError("candidate_metrics_require_typed_waterfall_receipts")
        variant_evaluation = evaluation_by_variant.get(receipt.variant_id)
        if (
            variant_evaluation is None
            or receipt.experiment_id != spec.experiment_id
            or receipt.experiment_hash != spec.experiment_hash()
            or (
                receipt.provider_receipt_ref
                and receipt.provider_receipt_ref
                not in variant_evaluation.provider_receipt_refs
            )
            or receipt.decision_receipt_id not in variant_evaluation.decision_receipt_refs
        ):
            raise RoutingExperimentError("candidate_waterfall_receipt_is_not_in_evaluation")
        key = (
            receipt.variant_id,
            receipt.unit_ref,
            receipt.step_order,
            receipt.attempt_sequence,
        )
        if key in receipt_keys:
            raise RoutingExperimentError("candidate_waterfall_attempt_is_duplicated")
        receipt_keys.add(key)

    for variant_id, variant_evaluation in evaluation_by_variant.items():
        if len(set(variant_evaluation.provider_receipt_refs)) != len(
            variant_evaluation.provider_receipt_refs
        ) or len(set(variant_evaluation.decision_receipt_refs)) != len(
            variant_evaluation.decision_receipt_refs
        ):
            raise RoutingExperimentError(
                "candidate_evaluation_receipt_references_are_duplicated"
            )
        variant_receipts = tuple(
            item for item in receipts if item.variant_id == variant_id
        )
        sidecar_provider_refs = {
            item.provider_receipt_ref
            for item in variant_receipts
            if item.provider_receipt_ref
        }
        if sidecar_provider_refs != set(variant_evaluation.provider_receipt_refs):
            raise RoutingExperimentError(
                "candidate_provider_sidecar_coverage_differs_from_evaluation"
            )
        sidecar_decision_refs = {
            item.decision_receipt_id for item in variant_receipts
        }
        if sidecar_decision_refs != set(variant_evaluation.decision_receipt_refs):
            raise RoutingExperimentError(
                "candidate_decision_sidecar_coverage_differs_from_evaluation"
            )

    all_units = set(spec.input.calibration_unit_refs) | set(spec.input.holdout_unit_refs)
    if any(item.unit_ref not in all_units for item in receipts):
        raise RoutingExperimentError("candidate_waterfall_receipt_unit_is_not_in_experiment")
    split_units = {
        "calibration": tuple(spec.input.calibration_unit_refs),
        "holdout": tuple(spec.input.holdout_unit_refs),
    }
    results: list[CandidateWaterfallMetric] = []
    for variant in sorted(spec.variants, key=lambda item: item.variant_id):
        for split, units in split_units.items():
            unit_set = set(units)
            selected = sorted(
                (
                    item
                    for item in receipts
                    if item.variant_id == variant.variant_id and item.unit_ref in unit_set
                ),
                key=lambda item: (item.unit_ref, item.step_order, item.attempt_sequence),
            )
            verified_by_unit = {
                unit_ref: sum(
                    item.verified_qualified_count
                    for item in selected
                    if item.unit_ref == unit_ref
                )
                for unit_ref in units
            }
            raw_count = sum(item.raw_count for item in selected)
            verified_count = sum(item.verified_qualified_count for item in selected)
            published_count = sum(item.published_count for item in selected)
            total_cost = sum(item.cost_microusd for item in selected)
            fulfilled_count = sum(value >= target for value in verified_by_unit.values())
            results.append(
                CandidateWaterfallMetric(
                    evaluation_receipt_id=evaluation.receipt_id,
                    experiment_id=spec.experiment_id,
                    experiment_hash=spec.experiment_hash(),
                    variant_id=variant.variant_id,
                    split=split,
                    target_verified_qualified_count=target,
                    unit_count=len(units),
                    fulfilled_unit_count=fulfilled_count,
                    waterfall_attempt_count=len(selected),
                    provider_call_count=sum(item.provider_call_count for item in selected),
                    total_cost_microusd=total_cost,
                    total_latency_ms=sum(item.latency_ms for item in selected),
                    raw_count=raw_count,
                    normalized_count=sum(item.normalized_count for item in selected),
                    unique_count=sum(item.unique_count for item in selected),
                    verified_qualified_count=verified_count,
                    published_count=published_count,
                    failed_attempt_count=sum(item.disposition == "failed" for item in selected),
                    missed_attempt_count=sum(item.disposition == "missed" for item in selected),
                    fulfillment_rate=round(fulfilled_count / len(units), 8),
                    verification_rate=round(verified_count / raw_count, 8) if raw_count else 0.0,
                    publication_rate=(
                        round(published_count / verified_count, 8)
                        if verified_count
                        else 0.0
                    ),
                    verified_qualified_per_usd=(
                        round(verified_count / (total_cost / 1_000_000), 8)
                        if total_cost
                        else 0.0
                    ),
                    waterfall_receipt_refs=tuple(item.receipt_id for item in selected),
                    provider_receipt_refs=tuple(
                        sorted(
                            {
                                item.provider_receipt_ref
                                for item in selected
                                if item.provider_receipt_ref
                            }
                        )
                    ),
                    decision_receipt_refs=tuple(
                        sorted({item.decision_receipt_id for item in selected})
                    ),
                )
            )
    return tuple(results)


__all__ = [
    "CANDIDATE_WATERFALL_METRIC_VERSION",
    "CANDIDATE_WATERFALL_RECEIPT_VERSION",
    "CandidateWaterfallMetric",
    "CandidateWaterfallReceipt",
    "candidate_waterfall_receipt_from_model",
    "evaluate_candidate_waterfall_metrics",
    "validate_candidate_routing_model_runtime",
]
