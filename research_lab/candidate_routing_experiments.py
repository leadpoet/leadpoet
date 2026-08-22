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
    validate_sourcing_model_artifact_identity,
)


CANDIDATE_WATERFALL_RECEIPT_VERSION = (
    "leadpoet.candidate_waterfall_receipt_sidecar:v1"
)
CANDIDATE_WATERFALL_METRIC_VERSION = (
    "leadpoet.candidate_waterfall_metric_sidecar:v1"
)
EXACT_MODEL_RUNNER_RECEIPT_VERSION = "model-runner-receipt:v1"
EXACT_MODEL_CANDIDATE_METRIC_VERSION = (
    "leadpoet.exact_model_candidate_attempt_projection:v1"
)
EXACT_MODEL_CANDIDATE_WATERFALL_VERSION = "candidate-waterfall-receipt:v1"
EXACT_MODEL_CANDIDATE_ATTEMPT_VERSION = (
    "candidate-waterfall-attempt:v1"
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
_PROVIDER_OUTCOME_BY_MODEL_OUTCOME = {
    "succeeded": ProviderOutcome.VERIFIED.value,
    "empty": ProviderOutcome.SOURCE_MISS.value,
    "unavailable": ProviderOutcome.ADAPTER_FAILURE.value,
    "timeout": ProviderOutcome.ADAPTER_FAILURE.value,
    "failed": ProviderOutcome.ADAPTER_FAILURE.value,
}


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


def _validate_candidate_artifact_authority(
    *,
    spec: RoutingExperimentV2Spec,
    variant: Any,
    contract_sha256: str,
    artifact_authority: Any | None,
) -> None:
    """Require one exact signed artifact and its separate candidate contract.

    The shared V2 evaluator owns the full experiment admission path.  This
    smaller adapter still runs at the sidecar boundary, so it repeats the
    artifact checks that are required before a candidate receipt can be
    attached to a shared decision.  The manifest is structural evidence in a
    fixture; replay and measured Lab runs additionally require the injected
    cryptographic authority.
    """

    expected_branch = (
        "main" if variant.variant_id == spec.baseline_variant_id else "leadpoet-lab"
    )
    if variant.artifact.branch != expected_branch:
        raise RoutingExperimentError(
            "candidate_artifact_branch_must_be_" + expected_branch.replace("-", "_")
        )
    artifact_errors = validate_sourcing_model_artifact_identity(variant.artifact)
    if artifact_errors:
        raise RoutingExperimentError(
            "candidate_artifact_identity_is_invalid:" + ";".join(artifact_errors)
        )
    manifest_payload = variant.artifact_authority_manifest
    if not isinstance(manifest_payload, Mapping):
        raise RoutingExperimentError("candidate_signed_artifact_manifest_is_missing")
    try:
        from research_lab.eval import (
            PrivateModelArtifactManifest,
            validate_private_model_artifact_manifest,
        )

        manifest = PrivateModelArtifactManifest.from_mapping(manifest_payload)
        manifest_errors = validate_private_model_artifact_manifest(manifest)
    except Exception as exc:
        raise RoutingExperimentError(
            "candidate_signed_artifact_manifest_is_invalid"
        ) from exc
    if manifest_errors:
        raise RoutingExperimentError(
            "candidate_signed_artifact_manifest_is_invalid:"
            + ";".join(manifest_errors)
        )
    normalized_manifest = manifest.to_dict()
    for artifact_field, manifest_field in (
        ("model_artifact_hash", "model_artifact_hash"),
        ("manifest_hash", "manifest_hash"),
        ("commit_sha", "git_commit_sha"),
    ):
        if str(getattr(variant.artifact, artifact_field)).lower() != str(
            normalized_manifest.get(manifest_field) or ""
        ).lower():
            raise RoutingExperimentError(
                f"candidate_artifact_manifest_{artifact_field}_differs"
            )
    candidate_contract = normalized_manifest.get(
        "candidate_waterfall_contract_sha256"
    )
    expected_contract = _model_hash(
        candidate_contract,
        "candidate_waterfall_contract_sha256",
    )
    if expected_contract != contract_sha256:
        raise RoutingExperimentError(
            "candidate_waterfall_contract_differs_from_signed_artifact"
        )
    if artifact_authority is None:
        if spec.receipt_execution_mode != ReceiptExecutionMode.FIXTURE.value:
            raise RoutingExperimentError(
                "candidate_artifact_signature_authority_is_required"
            )
        return
    verifier = getattr(artifact_authority, "verify", None)
    if not callable(verifier):
        raise RoutingExperimentError("candidate_artifact_signature_authority_is_invalid")
    try:
        outcome = verifier(artifact=variant.artifact, manifest=normalized_manifest)
    except RoutingExperimentError:
        raise
    except Exception as exc:
        raise RoutingExperimentError(
            "candidate_artifact_signature_verification_failed"
        ) from exc
    if not isinstance(outcome, Mapping) or outcome.get("verified") is not True:
        raise RoutingExperimentError("candidate_artifact_signature_verification_rejected")
    for field_name, expected in (
        ("model_artifact_hash", variant.artifact.model_artifact_hash),
        ("manifest_hash", variant.artifact.manifest_hash),
        ("commit_sha", variant.artifact.commit_sha),
    ):
        observed = outcome.get(field_name)
        if field_name == "commit_sha" and observed is None:
            observed = outcome.get("git_commit_sha")
        if str(observed or "").lower() != str(expected).lower():
            raise RoutingExperimentError(
                f"candidate_artifact_signature_{field_name}_differs"
            )


def adapt_exact_model_candidate_receipt(
    terminal_result: Mapping[str, Any],
    *,
    expected_release_identity_sha256: str,
    expected_binding_contracts_sha256: str,
    expected_candidate_waterfall_contract_sha256: str,
    authoritative_provider_receipts: Sequence[ProviderReceipt],
) -> Mapping[str, Any]:
    """Validate and project the canonical candidate waterfall receipt.

    This function does not receive a runtime, provider binding, route
    compiler, endpoint, credential, or execution callback. The model has
    already selected and executed the route. Lab verifies the signed model
    receipt against the independently persisted provider receipts, then
    exposes candidate metrics for the shared routing evaluation.
    """

    if not isinstance(terminal_result, Mapping) or (
        terminal_result.get("status") != "completed"
    ):
        raise RoutingExperimentError("exact_model_terminal_result_is_invalid")
    raw_receipt = terminal_result.get("model_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise RoutingExperimentError("exact_model_candidate_receipt_is_missing")
    receipt = dict(raw_receipt)
    claimed_receipt_hash = _model_hash(
        receipt.pop("receipt_sha256", None),
        "runner_receipt_hash",
    )
    if (
        receipt.get("schema_version") != EXACT_MODEL_RUNNER_RECEIPT_VERSION
        or receipt.get("release_identity_sha256")
        != _model_hash(
            expected_release_identity_sha256,
            "release_identity_sha256",
        )
        or receipt.get("tool_binding_manifest_sha256")
        != _model_hash(
            expected_binding_contracts_sha256,
            "binding_contracts_sha256",
        )
        or sha256_json(receipt).split(":", 1)[1] != claimed_receipt_hash
    ):
        raise RoutingExperimentError("exact_model_candidate_receipt_identity_differs")
    raw_waterfall = receipt.get("candidate_waterfall")
    if not isinstance(raw_waterfall, Mapping):
        raise RoutingExperimentError("exact_model_candidate_waterfall_is_missing")
    waterfall = dict(raw_waterfall)
    claimed_waterfall_hash = _model_hash(
        waterfall.pop("waterfall_sha256", None),
        "candidate_waterfall_hash",
    )
    if (
        waterfall.get("schema_version")
        != EXACT_MODEL_CANDIDATE_WATERFALL_VERSION
        or _model_hash(
            waterfall.get("contract_sha256"),
            "candidate_waterfall_contract_sha256",
        )
        != _model_hash(
            expected_candidate_waterfall_contract_sha256,
            "expected_candidate_waterfall_contract_sha256",
        )
        or sha256_json(waterfall).split(":", 1)[1] != claimed_waterfall_hash
        or waterfall.get("start_request_sha256")
        != receipt.get("start_request_sha256")
    ):
        raise RoutingExperimentError("exact_model_candidate_waterfall_identity_differs")
    result = terminal_result.get("result")
    if not isinstance(result, Mapping) or (
        sha256_json(result).split(":", 1)[1]
        != receipt.get("model_result_sha256")
    ):
        raise RoutingExperimentError("exact_model_candidate_result_identity_differs")
    result_receipt = result.get("receipt")
    if not isinstance(result_receipt, Mapping):
        raise RoutingExperimentError("exact_model_candidate_result_receipt_is_missing")
    orchestration_receipt = dict(result_receipt)
    claimed_orchestration_hash = _model_hash(
        orchestration_receipt.pop("receipt_sha256", None),
        "orchestration_receipt_hash",
    )
    if (
        sha256_json(orchestration_receipt).split(":", 1)[1]
        != claimed_orchestration_hash
        or receipt.get("orchestration_receipt_sha256")
        != claimed_orchestration_hash
    ):
        raise RoutingExperimentError(
            "exact_model_candidate_orchestration_receipt_differs"
        )
    route = orchestration_receipt.get("candidate_plan")
    if not isinstance(route, Mapping):
        raise RoutingExperimentError("exact_model_candidate_route_receipt_is_invalid")
    route_payload = dict(route)
    claimed_route_hash = _model_hash(
        route_payload.pop("candidate_plan_sha256", None),
        "candidate_route_hash",
    )
    compiled_route = route_payload.get("route")
    if not isinstance(compiled_route, Mapping):
        raise RoutingExperimentError("exact_model_candidate_compiled_route_is_invalid")
    compiled_route_payload = dict(compiled_route)
    claimed_compiled_route_hash = _model_hash(
        compiled_route_payload.pop("plan_sha256", None),
        "compiled_route_hash",
    )
    if (
        sha256_json(compiled_route_payload).split(":", 1)[1]
        != claimed_compiled_route_hash
    ):
        raise RoutingExperimentError("exact_model_candidate_compiled_route_differs")
    route_payload["route"] = compiled_route_payload
    if sha256_json(route_payload).split(":", 1)[1] != claimed_route_hash:
        raise RoutingExperimentError("exact_model_candidate_route_receipt_differs")
    raw_attempts = waterfall.get("attempts")
    if not isinstance(raw_attempts, list):
        raise RoutingExperimentError("exact_model_candidate_attempts_are_invalid")
    target_verified_qualified_count = _nonnegative_int(
        waterfall.get("target_verified_qualified_count"),
        "target_verified_qualified_count",
    )
    if target_verified_qualified_count < 1:
        raise RoutingExperimentError("exact_model_candidate_target_is_invalid")
    if not isinstance(authoritative_provider_receipts, Sequence) or isinstance(
        authoritative_provider_receipts, (str, bytes)
    ):
        raise RoutingExperimentError("exact_model_provider_receipts_are_invalid")
    providers_by_ref: dict[str, ProviderReceipt] = {}
    for provider_receipt in authoritative_provider_receipts:
        if not isinstance(provider_receipt, ProviderReceipt):
            raise RoutingExperimentError("exact_model_provider_receipt_is_not_typed")
        provider_errors = validate_provider_receipt(provider_receipt)
        if provider_errors:
            raise RoutingExperimentError(
                "exact_model_provider_receipt_is_invalid:"
                + ";".join(provider_errors)
            )
        if provider_receipt.receipt_ref in providers_by_ref:
            raise RoutingExperimentError("exact_model_provider_receipt_is_duplicated")
        providers_by_ref[provider_receipt.receipt_ref] = provider_receipt
    metrics: list[Mapping[str, Any]] = []
    previous_attempt_sha256 = "0" * 64
    aggregate_counts = {
        "raw_candidate_count": 0,
        "normalized_candidate_count": 0,
        "unique_candidate_count": 0,
        "verified_qualified_candidate_count": 0,
        "provider_call_count": 0,
        "credit_microunits": 0,
    }
    for attempt_index, raw_attempt in enumerate(raw_attempts):
        if not isinstance(raw_attempt, Mapping):
            raise RoutingExperimentError("exact_model_candidate_attempt_is_invalid")
        attempt = dict(raw_attempt)
        claimed_attempt_hash = _model_hash(
            attempt.pop("attempt_sha256", None),
            "candidate_attempt_hash",
        )
        if (
            attempt.get("schema_version")
            != EXACT_MODEL_CANDIDATE_ATTEMPT_VERSION
            or attempt.get("attempt_index") != attempt_index
            or attempt.get("previous_attempt_sha256")
            != previous_attempt_sha256
            or sha256_json(attempt).split(":", 1)[1] != claimed_attempt_hash
        ):
            raise RoutingExperimentError("exact_model_candidate_attempt_identity_differs")
        previous_attempt_sha256 = claimed_attempt_hash
        tool_id = str(attempt.get("tool_id") or "")
        raw_count = _nonnegative_int(
            attempt.get("raw_candidate_count"),
            "raw_candidate_count",
        )
        normalized_count = _nonnegative_int(
            attempt.get("normalized_candidate_count"),
            "normalized_candidate_count",
        )
        unique_count = _nonnegative_int(
            attempt.get("unique_candidate_count"),
            "unique_candidate_count",
        )
        verified_count = _nonnegative_int(
            attempt.get("verified_qualified_candidate_count"),
            "verified_qualified_candidate_count",
        )
        if not raw_count >= normalized_count >= unique_count >= verified_count:
            raise RoutingExperimentError("exact_model_candidate_counts_are_not_monotonic")
        raw_verification_hashes = attempt.get(
            "company_verification_receipt_sha256s"
        )
        if not isinstance(raw_verification_hashes, list) or len(
            raw_verification_hashes
        ) != verified_count:
            raise RoutingExperimentError(
                "exact_model_candidate_verification_receipts_differ"
            )
        verification_hashes = tuple(
            _model_hash(value, "company_verification_receipt_sha256")
            for value in raw_verification_hashes
        )
        provider_call_count = _nonnegative_int(
            attempt.get("provider_call_count"),
            "provider_call_count",
        )
        credit_microunits = _nonnegative_int(
            attempt.get("credit_microunits"),
            "credit_microunits",
        )
        latency_ms = attempt.get("latency_ms")
        stop_policy_sha256 = _model_hash(
            attempt.get("stop_policy_sha256")
            or waterfall.get("stop_policy_sha256"),
            "candidate_stop_policy_sha256",
        )
        step_order = _nonnegative_int(
            attempt.get("step_order", attempt_index),
            "candidate_step_order",
        )
        attempt_sequence = _nonnegative_int(
            attempt.get("attempt_sequence", attempt_index),
            "candidate_attempt_sequence",
        )
        if step_order != attempt_index or attempt_sequence != attempt_index:
            raise RoutingExperimentError(
                "exact_model_candidate_attempt_sequence_is_not_contiguous"
            )
        if (
            not _CANDIDATE_TOOL_RE.fullmatch(tool_id)
            or attempt.get("plan_sha256") != claimed_route_hash
            or isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not math.isfinite(latency_ms)
            or latency_ms < 0
        ):
            raise RoutingExperimentError("exact_model_candidate_attempt_differs")
        provider_receipt_ref = str(attempt.get("provider_receipt_ref") or "")
        model_outcome = str(attempt.get("outcome") or "")
        provider_outcome = "skipped"
        authoritative_provider_call_count = 0
        authoritative_billed_credit_microunits = 0
        authoritative_latency_ms = 0
        if provider_receipt_ref:
            provider_receipt = providers_by_ref.pop(provider_receipt_ref, None)
            if provider_receipt is None:
                raise RoutingExperimentError(
                    "exact_model_candidate_provider_receipt_is_missing"
                )
            authoritative_provider_call_count = getattr(
                provider_receipt, "call_count", None
            )
            if type(authoritative_provider_call_count) is not int or authoritative_provider_call_count < 1:
                raise RoutingExperimentError(
                    "exact_model_provider_receipt_call_count_is_invalid"
                )
            expected_provider_outcome = _PROVIDER_OUTCOME_BY_MODEL_OUTCOME.get(
                model_outcome
            )
            if (
                expected_provider_outcome is None
                or provider_call_count != authoritative_provider_call_count
                or provider_receipt.tool_id != tool_id
                or provider_receipt.outcome != expected_provider_outcome
                or provider_receipt.credit_microunits != credit_microunits
                or provider_receipt.latency_ms != latency_ms
            ):
                raise RoutingExperimentError(
                    "exact_model_candidate_attempt_differs_from_provider_receipt"
                )
            _model_hash(attempt.get("action_sha256"), "candidate_action_sha256")
            _model_hash(
                attempt.get("completion_sha256"),
                "candidate_completion_sha256",
            )
            provider_outcome = provider_receipt.outcome
            authoritative_billed_credit_microunits = provider_receipt.credit_microunits
            authoritative_latency_ms = provider_receipt.latency_ms
        elif (
            model_outcome not in {"not_invoked", "not_attempted"}
            or provider_call_count != 0
            or credit_microunits != 0
            or latency_ms != 0
            or attempt.get("action_sha256") is not None
            or attempt.get("completion_sha256") is not None
        ):
            raise RoutingExperimentError(
                "exact_model_candidate_invoked_attempt_requires_provider_receipt"
            )
        for field_name, value in (
            ("raw_candidate_count", raw_count),
            ("normalized_candidate_count", normalized_count),
            ("unique_candidate_count", unique_count),
            ("verified_qualified_candidate_count", verified_count),
            ("provider_call_count", authoritative_provider_call_count),
            ("credit_microunits", authoritative_billed_credit_microunits),
        ):
            aggregate_counts[field_name] += value
        raw_published_count = attempt.get("published_count")
        if raw_published_count is None:
            raise RoutingExperimentError(
                "exact_model_candidate_publication_attribution_is_missing"
            )
        published_count = _nonnegative_int(
            raw_published_count,
            "published_count",
        )
        if published_count > verified_count:
            raise RoutingExperimentError(
                "exact_model_candidate_published_count_exceeds_verified_count"
            )
        verification_bundle_sha256 = (
            sha256_json(list(verification_hashes)).split(":", 1)[1]
            if verification_hashes
            else ""
        )
        metric = {
            "schema_version": EXACT_MODEL_CANDIDATE_METRIC_VERSION,
            "tool_id": tool_id,
            "outcome": _safe_ref(model_outcome, "attempt_outcome"),
            "reason_code": _safe_ref(
                attempt.get("reason_code"),
                "attempt_reason_code",
            ),
            "provider_receipt_ref": provider_receipt_ref,
            "provider_outcome": provider_outcome,
            "raw_candidate_count": raw_count,
            "normalized_candidate_count": normalized_count,
            "unique_candidate_count": unique_count,
            "verified_qualified_candidate_count": verified_count,
            "company_verification_receipt_sha256s": verification_hashes,
            "verification_receipt_bundle_sha256": verification_bundle_sha256,
            "published_count": published_count,
            "provider_call_count": authoritative_provider_call_count,
            "billed_credit_microunits": authoritative_billed_credit_microunits,
            "latency_ms": float(authoritative_latency_ms),
            "candidate_plan_sha256": claimed_route_hash,
            "attempt_sha256": claimed_attempt_hash,
            "stop_policy_sha256": stop_policy_sha256,
            "step_order": step_order,
            "attempt_sequence": attempt_sequence,
        }
        metrics.append(
            {
                **metric,
                "metric_sha256": sha256_json(metric).split(":", 1)[1],
            }
        )
    if not metrics:
        raise RoutingExperimentError("exact_model_candidate_attempts_are_missing")
    if providers_by_ref:
        raise RoutingExperimentError(
            "exact_model_provider_receipt_coverage_differs_from_waterfall"
        )
    if any(
        waterfall.get(field_name) != value
        for field_name, value in aggregate_counts.items()
        if field_name != "published_count"
    ):
        raise RoutingExperimentError("exact_model_candidate_waterfall_totals_differ")
    waterfall_published_count = waterfall.get("published_count")
    if waterfall_published_count is None:
        raise RoutingExperimentError(
            "exact_model_candidate_publication_attribution_is_missing"
        )
    if type(waterfall_published_count) is not int or waterfall_published_count < 0:
        raise RoutingExperimentError("exact_model_candidate_published_count_is_invalid")
    if waterfall_published_count != sum(
        int(metric.get("published_count") or 0) for metric in metrics
    ):
        raise RoutingExperimentError("exact_model_candidate_published_count_differs")
    if waterfall_published_count > aggregate_counts["verified_qualified_candidate_count"]:
        raise RoutingExperimentError(
            "exact_model_candidate_published_count_exceeds_verified_count"
        )
    return {
        "model_receipt_sha256": claimed_receipt_hash,
        "orchestration_receipt_sha256": claimed_orchestration_hash,
        "candidate_waterfall_sha256": claimed_waterfall_hash,
        "candidate_route": dict(route),
        "candidate_stop_reason": _safe_ref(
            waterfall.get("stop_reason"),
            "stop_reason",
        ),
        "candidate_target_verified_qualified_count": target_verified_qualified_count,
        "candidate_attempt_metrics": tuple(metrics),
    }


def candidate_waterfall_receipts_from_exact_model(
    *,
    spec: RoutingExperimentV2Spec,
    variant_id: str,
    decision_receipt: RoutingDecisionReceiptV2,
    terminal_result: Mapping[str, Any],
    expected_release_identity_sha256: str,
    expected_binding_contracts_sha256: str,
    expected_candidate_waterfall_contract_sha256: str,
    authoritative_provider_receipts: Sequence[ProviderReceipt],
) -> tuple[CandidateWaterfallReceipt, ...]:
    """Project one exact Model terminal into durable candidate sidecars.

    The exact runner owns the serialized waterfall. This helper only binds
    that result to the shared decision and independently persisted provider
    receipts; it never selects a route or invokes a provider.
    """

    if not isinstance(spec, RoutingExperimentV2Spec):
        raise RoutingExperimentError("candidate_exact_model_spec_is_invalid")
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
        raise RoutingExperimentError("candidate_decision_receipt_lineage_differs")
    adapted = adapt_exact_model_candidate_receipt(
        terminal_result,
        expected_release_identity_sha256=expected_release_identity_sha256,
        expected_binding_contracts_sha256=expected_binding_contracts_sha256,
        expected_candidate_waterfall_contract_sha256=(
            expected_candidate_waterfall_contract_sha256
        ),
        authoritative_provider_receipts=authoritative_provider_receipts,
    )
    metrics = adapted.get("candidate_attempt_metrics")
    if not isinstance(metrics, tuple) or not metrics:
        raise RoutingExperimentError("candidate_exact_model_attempt_metrics_are_invalid")
    variants = [item for item in spec.variants if item.variant_id == variant_id]
    if len(variants) != 1:
        raise RoutingExperimentError("candidate_variant_must_exist_exactly_once")
    variant_bindings = [
        item
        for item in spec.provider_bindings
        if item.binding_id in set(variants[0].binding_ids)
    ]
    bindings_by_tool = {item.tool_id: item for item in variant_bindings}
    if len(bindings_by_tool) != len(variant_bindings):
        raise RoutingExperimentError("candidate_variant_bindings_are_duplicated")
    providers_by_ref = {item.receipt_ref: item for item in authoritative_provider_receipts}
    model_contract = _model_hash(
        expected_candidate_waterfall_contract_sha256,
        "candidate_waterfall_contract_sha256",
    )
    previous_attempt_hash = ""
    attempt_hashes: list[str] = []
    receipts: list[CandidateWaterfallReceipt] = []
    disposition_by_outcome = {
        "succeeded": "succeeded",
        "empty": "missed",
        "not_invoked": "skipped",
        "not_attempted": "skipped",
        "failed": "failed",
        "unavailable": "failed",
        "timeout": "failed",
        "deferred": "deferred",
    }
    for metric in metrics:
        if not isinstance(metric, Mapping):
            raise RoutingExperimentError("candidate_exact_model_attempt_metric_is_invalid")
        tool_id = str(metric.get("tool_id") or "")
        binding = bindings_by_tool.get(tool_id)
        if binding is None:
            raise RoutingExperimentError("candidate_attempt_binding_is_not_in_variant")
        provider_ref = str(metric.get("provider_receipt_ref") or "")
        provider = providers_by_ref.get(provider_ref) if provider_ref else None
        if provider_ref and provider is None:
            raise RoutingExperimentError("candidate_attempt_provider_receipt_is_missing")
        if provider is not None:
            if (
                provider.binding_id != binding.binding_id
                or provider.tool_id != tool_id
                or provider.unit_ref != decision_receipt.unit_ref
                or provider.execution_mode != decision_receipt.execution_mode
            ):
                raise RoutingExperimentError("candidate_attempt_provider_identity_differs")
            provider_call_count = getattr(provider, "call_count", 1)
            billed_credit = provider.credit_microunits
            latency_ms = provider.latency_ms
        else:
            provider_call_count = 0
            billed_credit = 0
            latency_ms = 0
        if metric.get("provider_call_count") != provider_call_count:
            raise RoutingExperimentError("candidate_attempt_provider_call_count_differs")
        if metric.get("billed_credit_microunits") != billed_credit:
            raise RoutingExperimentError("candidate_attempt_provider_credit_differs")
        if int(float(metric.get("latency_ms") or 0)) != latency_ms:
            raise RoutingExperimentError("candidate_attempt_provider_latency_differs")
        attempt_hash = _model_hash(metric.get("attempt_sha256"), "attempt_sha256")
        expected_index = len(receipts)
        if (
            metric.get("step_order") != expected_index
            or metric.get("attempt_sequence") != expected_index
        ):
            raise RoutingExperimentError("candidate_attempt_sequence_is_not_contiguous")
        attempt_hashes.append(attempt_hash)
        prior_hash = previous_attempt_hash
        previous_attempt_hash = attempt_hash
        raw_verification_hashes = metric.get("company_verification_receipt_sha256s")
        if not isinstance(raw_verification_hashes, tuple):
            raise RoutingExperimentError("candidate_attempt_verification_receipts_are_invalid")
        verification_hash = str(metric.get("verification_receipt_bundle_sha256") or "")
        if raw_verification_hashes:
            expected_bundle_hash = sha256_json(list(raw_verification_hashes)).split(
                ":", 1
            )[1]
            if verification_hash != expected_bundle_hash:
                raise RoutingExperimentError(
                    "candidate_attempt_verification_receipt_bundle_differs"
                )
        elif verification_hash:
            raise RoutingExperimentError(
                "candidate_attempt_verification_receipt_bundle_is_invalid"
            )
        model_outcome = str(metric.get("outcome") or "")
        provider_outcome = str(metric.get("provider_outcome") or "skipped")
        if provider is None:
            if provider_outcome != "skipped":
                raise RoutingExperimentError(
                    "candidate_attempt_skipped_provider_outcome_differs"
                )
        elif provider_outcome != provider.outcome:
            raise RoutingExperimentError(
                "candidate_attempt_provider_outcome_differs"
            )
        disposition = disposition_by_outcome.get(model_outcome)
        if disposition is None or (disposition == "skipped") != (provider is None):
            raise RoutingExperimentError("candidate_attempt_disposition_differs")
        receipts.append(
            CandidateWaterfallReceipt(
                experiment_id=spec.experiment_id,
                experiment_hash=spec.experiment_hash(),
                variant_id=variant_id,
                artifact_key=decision_receipt.artifact_key,
                decision_receipt_id=decision_receipt.receipt_id,
                provider_receipt_ref=provider_ref,
                unit_ref=decision_receipt.unit_ref,
                binding_id=binding.binding_id,
                tool_id=tool_id,
                execution_mode=(
                    provider.execution_mode
                    if provider is not None
                    else decision_receipt.execution_mode
                ),
                provider_outcome=provider_outcome,
                decision_plan_hash=decision_receipt.plan_hash,
                decision_route_hash=decision_receipt.route_hash,
                model_contract_sha256=model_contract,
                model_plan_sha256=_model_hash(
                    metric.get("candidate_plan_sha256"), "candidate_plan_sha256"
                ),
                stop_policy_sha256=_model_hash(
                    metric.get("stop_policy_sha256"), "stop_policy_sha256"
                ),
                attempt_receipt_sha256=attempt_hash,
                prior_attempt_receipt_sha256=prior_hash,
                attempt_chain_sha256=_attempt_chain_sha256(attempt_hashes),
                verification_receipt_sha256=verification_hash,
                company_verification_receipt_sha256s=raw_verification_hashes,
                target_verified_qualified_count=int(
                    adapted["candidate_target_verified_qualified_count"]
                ),
                step_order=int(metric.get("step_order")),
                attempt_sequence=int(metric.get("attempt_sequence")),
                disposition=disposition,
                outcome_code=str(metric.get("reason_code") or ""),
                provider_call_count=provider_call_count,
                billed_credit_microunits=billed_credit,
                latency_ms=latency_ms,
                raw_count=int(metric.get("raw_candidate_count")),
                normalized_count=int(metric.get("normalized_candidate_count")),
                unique_count=int(metric.get("unique_candidate_count")),
                verified_qualified_count=int(
                    metric.get("verified_qualified_candidate_count")
                ),
                published_count=int(metric.get("published_count") or 0),
            )
        )
    return tuple(receipts)


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
    expected_branch = (
        "main" if variant_id == spec.baseline_variant_id else "leadpoet-lab"
    )
    if variant.artifact.branch != expected_branch:
        raise RoutingExperimentError(
            "candidate_artifact_branch_must_be_" + expected_branch.replace("-", "_")
        )
    baseline_matches = [
        item for item in spec.variants if item.variant_id == spec.baseline_variant_id
    ]
    if len(baseline_matches) != 1:
        raise RoutingExperimentError("candidate_baseline_variant_must_exist_exactly_once")
    baseline = baseline_matches[0]
    if baseline.artifact.branch != "main":
        raise RoutingExperimentError("candidate_artifact_branch_must_be_main")
    if variant_id != spec.baseline_variant_id and variant.artifact.identity_payload() == baseline.artifact.identity_payload():
        raise RoutingExperimentError("candidate_variants_must_use_distinct_artifacts")
    return variant


def validate_candidate_routing_model_runtime(
    *,
    spec: RoutingExperimentV2Spec,
    variant_id: str,
    model_adapter: Any,
    artifact_authority: Any | None = None,
) -> Mapping[str, Any]:
    """Fail closed unless the exact variant Model contract is available."""

    variant = _candidate_variant(spec, variant_id)
    runtime = getattr(model_adapter, "runtime", None)
    required_callables = (
        "candidate_waterfall_execution_contract_identity",
        "evaluate_candidate_waterfall_payloads",
        "runtime_routing_metadata",
    )
    adapter_callables = (
        "parse_plan",
        "plan_hash",
        "route_hash",
        "validate_artifact_identity",
    )
    if runtime is None or any(
        not callable(getattr(runtime, name, None)) for name in required_callables
    ) or any(
        not callable(getattr(model_adapter, name, None))
        for name in adapter_callables
    ):
        raise RoutingExperimentError("model_candidate_routing_runtime_contract_is_incomplete")
    for type_name, error_code in (
        (
            "CandidateStepAttemptReceipt",
            "model_candidate_routing_receipt_parser_is_unavailable",
        ),
        (
            "CandidateStopPolicy",
            "model_candidate_stop_policy_parser_is_unavailable",
        ),
    ):
        model_type = getattr(runtime, type_name, None)
        if not callable(getattr(model_type, "from_payload", None)):
            raise RoutingExperimentError(error_code)
    try:
        artifact_errors = tuple(
            model_adapter.validate_artifact_identity(variant.artifact)
        )
        identity = runtime.candidate_waterfall_execution_contract_identity()
        metadata = runtime.runtime_routing_metadata()
    except Exception as exc:
        raise RoutingExperimentError("model_candidate_routing_identity_is_unavailable") from exc
    if artifact_errors:
        raise RoutingExperimentError(
            "model_candidate_artifact_identity_differs_from_variant:"
            + ";".join(str(item) for item in artifact_errors)
        )
    if not isinstance(identity, Mapping):
        raise RoutingExperimentError("model_candidate_routing_identity_is_invalid")
    if not isinstance(metadata, Mapping):
        raise RoutingExperimentError("model_candidate_routing_metadata_is_invalid")
    nested_identity = metadata.get("candidate_waterfall_execution")
    if not isinstance(nested_identity, Mapping) or dict(nested_identity) != dict(
        identity
    ):
        raise RoutingExperimentError("model_candidate_routing_identity_differs_from_metadata")
    _model_hash(identity.get("contract_sha256"), "routing_contract_hash")
    for metadata_field, artifact_field in (
        ("catalog_sha256", "routing_catalog_hash"),
        ("policy_sha256", "routing_policy_hash"),
    ):
        if model_hash_to_lab(
            _model_hash(metadata.get(metadata_field), metadata_field),
            metadata_field,
        ) != _lab_hash(getattr(variant.artifact, artifact_field), artifact_field):
            raise RoutingExperimentError(
                f"model_candidate_{metadata_field}_differs_from_variant"
            )
    if identity.get("provider_results_can_satisfy_target") is not False:
        raise RoutingExperimentError("model_candidate_routing_stop_contract_is_unsafe")
    contract_sha256 = _model_hash(
        identity.get("contract_sha256"),
        "routing_contract_hash",
    )
    baseline = next(
        item for item in spec.variants if item.variant_id == spec.baseline_variant_id
    )
    variants_to_validate = (variant,) if variant is baseline else (baseline, variant)
    for artifact_variant in variants_to_validate:
        _validate_candidate_artifact_authority(
            spec=spec,
            variant=artifact_variant,
            contract_sha256=contract_sha256,
            artifact_authority=artifact_authority,
        )
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


def _attempt_chain_sha256(attempt_hashes: Sequence[str]) -> str:
    for attempt_hash in attempt_hashes:
        _model_hash(attempt_hash, "attempt_chain_member")
    return sha256_json(list(attempt_hashes)).split(":", 1)[1]


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
    prior_attempt_receipt_sha256: str
    attempt_chain_sha256: str
    verification_receipt_sha256: str
    target_verified_qualified_count: int
    step_order: int
    attempt_sequence: int
    disposition: str
    outcome_code: str
    provider_call_count: int
    billed_credit_microunits: int
    latency_ms: int
    raw_count: int
    normalized_count: int
    unique_count: int
    verified_qualified_count: int
    published_count: int = 0
    company_verification_receipt_sha256s: tuple[str, ...] = ()
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
            "attempt_chain_sha256",
        ):
            _model_hash(getattr(self, field_name), field_name)
        _model_hash(
            self.prior_attempt_receipt_sha256,
            "prior_attempt_receipt_sha256",
            optional=True,
        )
        _model_hash(
            self.verification_receipt_sha256,
            "verification_receipt_sha256",
            optional=True,
        )
        if not isinstance(self.company_verification_receipt_sha256s, tuple):
            raise RoutingExperimentError(
                "candidate_company_verification_receipt_hashes_are_invalid"
            )
        for value in self.company_verification_receipt_sha256s:
            _model_hash(value, "company_verification_receipt_sha256")
        for field_name in (
            "step_order",
            "attempt_sequence",
            "target_verified_qualified_count",
            "provider_call_count",
            "billed_credit_microunits",
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
            if (
                self.provider_receipt_ref
                or self.provider_outcome != "skipped"
                or self.provider_call_count != 0
            ):
                raise RoutingExperimentError(
                    "candidate_skipped_attempt_cannot_claim_provider_receipt"
                )
        elif not self.provider_receipt_ref or self.provider_call_count < 1:
            raise RoutingExperimentError(
                "candidate_attempt_requires_one_unique_provider_receipt"
            )
        if self.target_verified_qualified_count < 1:
            raise RoutingExperimentError("candidate_receipt_target_must_be_positive")
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
        if self.verified_qualified_count and not self.company_verification_receipt_sha256s:
            raise RoutingExperimentError(
                "candidate_verified_count_requires_company_verification_receipts"
            )
        if self.step_order != self.attempt_sequence:
            raise RoutingExperimentError(
                "candidate_attempt_step_and_sequence_must_match"
            )
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
        identity = self.identity_payload()
        identity["company_verification_receipt_sha256s"] = list(
            self.company_verification_receipt_sha256s
        )
        return {
            **identity,
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
        }


def candidate_waterfall_receipt_from_model(
    *,
    spec: RoutingExperimentV2Spec,
    variant_id: str,
    decision_receipt: RoutingDecisionReceiptV2,
    provider_receipt: ProviderReceipt | None,
    plan_payload: Mapping[str, Any],
    stop_policy_payload: Mapping[str, Any],
    receipt_payloads: Sequence[Mapping[str, Any]],
    model_adapter: Any,
    published_count: int = 0,
) -> CandidateWaterfallReceipt:
    """Parse one exact Model receipt prefix and bind it to PR 93 receipts."""

    variant = _candidate_variant(spec, variant_id)
    identity = validate_candidate_routing_model_runtime(
        spec=spec,
        variant_id=variant_id,
        model_adapter=model_adapter,
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
    if not isinstance(receipt_payloads, Sequence) or isinstance(
        receipt_payloads, (str, bytes)
    ) or not receipt_payloads:
        raise RoutingExperimentError("model_candidate_attempt_prefix_is_invalid")
    runtime = model_adapter.runtime
    try:
        plan = model_adapter.parse_plan(plan_payload)
        stop_policy = runtime.CandidateStopPolicy.from_payload(stop_policy_payload)
        evaluated = runtime.evaluate_candidate_waterfall_payloads(
            plan_payload,
            stop_policy_payload,
            list(receipt_payloads),
        )
        model_receipts = tuple(
            runtime.CandidateStepAttemptReceipt.from_payload(item)
            for item in receipt_payloads
        )
    except Exception as exc:
        raise RoutingExperimentError("model_candidate_attempt_prefix_is_invalid") from exc
    if not isinstance(evaluated, Mapping) or set(evaluated) != {"progress", "decision"}:
        raise RoutingExperimentError("model_candidate_attempt_evaluation_is_invalid")
    for expected_index, attempt in enumerate(model_receipts):
        if attempt.step_order != expected_index or attempt.attempt != expected_index:
            raise RoutingExperimentError(
                "model_candidate_attempt_sequence_is_not_contiguous"
            )
    model_receipt = model_receipts[-1]
    model_plan_hash = model_adapter.plan_hash(plan)
    if model_receipt.plan_sha256 != model_plan_hash or model_hash_to_lab(
        model_plan_hash,
        "candidate_model_plan_hash",
    ) != decision_receipt.plan_hash:
        raise RoutingExperimentError("model_candidate_attempt_plan_differs_from_decision")
    if model_hash_to_lab(
        model_adapter.route_hash(plan),
        "candidate_model_route_hash",
    ) != decision_receipt.route_hash:
        raise RoutingExperimentError("model_candidate_route_differs_from_decision")
    if model_receipt.stop_policy_sha256 != stop_policy.sha256():
        raise RoutingExperimentError("model_candidate_attempt_stop_policy_differs")
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
        authoritative_provider_call_count = 0
        authoritative_billed_credit_microunits = 0
        authoritative_latency_ms = 0
    else:
        if model_receipt.disposition == "skipped":
            raise RoutingExperimentError(
                "model_candidate_skipped_attempt_cannot_claim_provider_receipt"
            )
        provider_errors = validate_provider_receipt(provider_receipt)
        if provider_errors:
            raise RoutingExperimentError(
                "candidate_provider_receipt_is_invalid:" + ";".join(provider_errors)
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
        authoritative_provider_call_count = getattr(provider_receipt, "call_count", None)
        if type(authoritative_provider_call_count) is not int or authoritative_provider_call_count < 1:
            raise RoutingExperimentError(
                "candidate_provider_receipt_call_count_is_invalid"
            )
        authoritative_billed_credit_microunits = provider_receipt.credit_microunits
        authoritative_latency_ms = provider_receipt.latency_ms
    if model_receipt.provider_call_count != authoritative_provider_call_count:
        raise RoutingExperimentError(
            "model_candidate_provider_call_count_differs_from_provider_receipt"
        )
    if round(model_receipt.latency_seconds * 1_000) != authoritative_latency_ms:
        raise RoutingExperimentError(
            "model_candidate_latency_differs_from_provider_receipt"
        )
    published = _nonnegative_int(published_count, "published_count")
    if published > model_receipt.verified_qualified_count:
        raise RoutingExperimentError("candidate_published_count_exceeds_verified_count")
    attempt_hashes = tuple(item.sha256() for item in model_receipts)
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
        prior_attempt_receipt_sha256=(
            attempt_hashes[-2] if len(attempt_hashes) > 1 else ""
        ),
        attempt_chain_sha256=_attempt_chain_sha256(attempt_hashes),
        verification_receipt_sha256=model_receipt.verification_receipt_sha256,
        target_verified_qualified_count=stop_policy.target_verified_qualified_count,
        step_order=model_receipt.step_order,
        attempt_sequence=model_receipt.attempt,
        disposition=model_receipt.disposition,
        outcome_code=model_receipt.outcome_code,
        provider_call_count=authoritative_provider_call_count,
        billed_credit_microunits=authoritative_billed_credit_microunits,
        latency_ms=authoritative_latency_ms,
        raw_count=model_receipt.raw_candidate_count,
        normalized_count=model_receipt.normalized_candidate_count,
        unique_count=model_receipt.unique_candidate_count,
        verified_qualified_count=model_receipt.verified_qualified_count,
        published_count=published,
        company_verification_receipt_sha256s=(
            (model_receipt.verification_receipt_sha256,)
            if model_receipt.verification_receipt_sha256
            else ()
        ),
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
    total_billed_credit_microunits: int
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
    verified_qualified_per_credit: float
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
            "total_billed_credit_microunits",
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
            isinstance(self.verified_qualified_per_credit, bool)
            or not isinstance(self.verified_qualified_per_credit, (int, float))
            or not math.isfinite(self.verified_qualified_per_credit)
            or self.verified_qualified_per_credit < 0
        ):
            raise RoutingExperimentError(
                "candidate_metric_verified_qualified_per_credit_is_invalid"
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
    authoritative_provider_receipts: Sequence[ProviderReceipt] = (),
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
    if not isinstance(authoritative_provider_receipts, Sequence) or isinstance(
        authoritative_provider_receipts, (str, bytes)
    ):
        raise RoutingExperimentError("candidate_metrics_provider_receipts_are_invalid")
    providers_by_ref: dict[str, ProviderReceipt] = {}
    for provider_receipt in authoritative_provider_receipts:
        if not isinstance(provider_receipt, ProviderReceipt):
            raise RoutingExperimentError("candidate_metrics_provider_receipt_is_not_typed")
        provider_errors = validate_provider_receipt(provider_receipt)
        if provider_errors:
            raise RoutingExperimentError(
                "candidate_metrics_provider_receipt_is_invalid:"
                + ";".join(provider_errors)
            )
        if provider_receipt.receipt_ref in providers_by_ref:
            raise RoutingExperimentError("candidate_metrics_provider_receipt_is_duplicated")
        providers_by_ref[provider_receipt.receipt_ref] = provider_receipt
    evaluation_by_variant = {item.variant_id: item for item in evaluation.variants}
    if set(evaluation_by_variant) != {item.variant_id for item in spec.variants}:
        raise RoutingExperimentError("candidate_evaluation_variants_differ_from_experiment")
    receipt_keys: set[tuple[str, str, int, int]] = set()
    provider_receipt_refs: set[str] = set()
    candidate_provider_refs: set[str] = {
        provider.receipt_ref
        for provider in providers_by_ref.values()
        if provider.tool_id.startswith("candidate.")
    }
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
        if receipt.target_verified_qualified_count != target:
            raise RoutingExperimentError(
                "candidate_receipt_target_differs_from_metric_target"
            )
        if receipt.provider_receipt_ref:
            if receipt.provider_receipt_ref in provider_receipt_refs:
                raise RoutingExperimentError(
                    "candidate_provider_receipt_sidecar_is_duplicated"
                )
            provider_receipt_refs.add(receipt.provider_receipt_ref)
            provider_receipt = providers_by_ref.get(receipt.provider_receipt_ref)
            if provider_receipt is None:
                raise RoutingExperimentError(
                    "candidate_provider_receipt_authority_is_missing"
                )
            provider_call_count = getattr(provider_receipt, "call_count", None)
            if (
                type(provider_call_count) is not int
                or provider_call_count < 1
                or provider_receipt.unit_ref != receipt.unit_ref
                or provider_receipt.tool_id != receipt.tool_id
                or provider_receipt.binding_id != receipt.binding_id
                or provider_receipt.execution_mode != receipt.execution_mode
                or provider_receipt.outcome != receipt.provider_outcome
                or receipt.provider_call_count != provider_call_count
                or receipt.billed_credit_microunits
                != provider_receipt.credit_microunits
                or receipt.latency_ms != provider_receipt.latency_ms
            ):
                raise RoutingExperimentError(
                    "candidate_provider_receipt_authority_differs"
                )

    receipt_groups: dict[tuple[str, str], list[CandidateWaterfallReceipt]] = {}
    for receipt in receipts:
        receipt_groups.setdefault((receipt.variant_id, receipt.unit_ref), []).append(
            receipt
        )
    for group in receipt_groups.values():
        ordered = sorted(
            group,
            key=lambda item: (item.step_order, item.attempt_sequence),
        )
        prefix_hashes: list[str] = []
        for expected_index, receipt in enumerate(ordered):
            if (
                receipt.step_order != expected_index
                or receipt.attempt_sequence != expected_index
            ):
                raise RoutingExperimentError(
                    "candidate_attempt_sidecar_sequence_is_not_contiguous"
                )
            expected_prior = prefix_hashes[-1] if prefix_hashes else ""
            if receipt.prior_attempt_receipt_sha256 != expected_prior:
                raise RoutingExperimentError(
                    "candidate_attempt_sidecar_prefix_is_not_contiguous"
                )
            prefix_hashes.append(receipt.attempt_receipt_sha256)
            if receipt.attempt_chain_sha256 != _attempt_chain_sha256(prefix_hashes):
                raise RoutingExperimentError(
                    "candidate_attempt_sidecar_chain_hash_differs"
                )

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
        expected_variant_provider_refs = {
            ref
            for ref in variant_evaluation.provider_receipt_refs
            if ref in candidate_provider_refs
        }
        if sidecar_provider_refs != expected_variant_provider_refs:
            raise RoutingExperimentError(
                "candidate_provider_sidecar_coverage_differs_from_evaluation"
            )
        sidecar_decision_refs = {
            item.decision_receipt_id for item in variant_receipts
        }
        expected_variant_decision_refs = set(variant_evaluation.decision_receipt_refs)
        if sidecar_decision_refs != expected_variant_decision_refs:
            raise RoutingExperimentError(
                "candidate_decision_sidecar_coverage_differs_from_evaluation"
            )
    evaluation_provider_refs = {
        provider_ref
        for variant_evaluation in evaluation_by_variant.values()
        for provider_ref in variant_evaluation.provider_receipt_refs
        if provider_ref in candidate_provider_refs
    }
    if set(providers_by_ref) != evaluation_provider_refs:
        raise RoutingExperimentError(
            "candidate_provider_receipt_authority_coverage_differs_from_evaluation"
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
            total_billed_credits = sum(
                item.billed_credit_microunits for item in selected
            )
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
                    total_billed_credit_microunits=total_billed_credits,
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
                    verified_qualified_per_credit=(
                        round(
                            verified_count
                            / (total_billed_credits / 1_000_000),
                            8,
                        )
                        if total_billed_credits
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
    "EXACT_MODEL_CANDIDATE_METRIC_VERSION",
    "EXACT_MODEL_RUNNER_RECEIPT_VERSION",
    "CandidateWaterfallMetric",
    "CandidateWaterfallReceipt",
    "adapt_exact_model_candidate_receipt",
    "candidate_waterfall_receipts_from_exact_model",
    "candidate_waterfall_receipt_from_model",
    "evaluate_candidate_waterfall_metrics",
    "validate_candidate_routing_model_runtime",
]
