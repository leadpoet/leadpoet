"""In-enclave entrypoints for unchanged Research Lab scoring functions.

This module owns no scoring formulas. Each operation delegates to an existing
production function so shadow comparisons exercise the same implementation as
the host-authoritative path.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping


SCORING_EXECUTOR_SCHEMA_VERSION = "leadpoet.gateway_scoring_executor.v1"

OP_QUALIFICATION_COMPANY_SCORES = "qualification_company_scores"
OP_BENCHMARK_ICP_SCORE = "benchmark_icp_score"
OP_BUILD_SCORE_BUNDLE = "build_score_bundle"
OP_BUILD_BASELINE_SCORE_SUMMARY = "build_baseline_score_summary"
OP_PROMOTION_IMPROVEMENT = "promotion_improvement"
OP_PROMOTION_GATE_DECISION = "promotion_gate_decision"
OP_RESEARCH_LAB_ALLOCATION = "research_lab_allocation"

SUPPORTED_OPERATIONS = frozenset(
    {
        OP_QUALIFICATION_COMPANY_SCORES,
        OP_BENCHMARK_ICP_SCORE,
        OP_BUILD_SCORE_BUNDLE,
        OP_BUILD_BASELINE_SCORE_SUMMARY,
        OP_PROMOTION_IMPROVEMENT,
        OP_PROMOTION_GATE_DECISION,
        OP_RESEARCH_LAB_ALLOCATION,
    }
)

# Only values that can change scoring behavior are committed here. Provider
# credentials and infrastructure locations are intentionally excluded; their
# accepted responses are committed separately through evidence roots.
SCORING_CONFIG_ENV_NAMES = (
    "INTENT_GATE_STRICT_JUDGE_ENABLED",
    "INTENT_THREE_STAGE_S1_MODEL",
    "INTENT_THREE_STAGE_S3_MODEL",
    "INTENT_VERIFIER_REVIEW_AS_ACCEPT",
    "QUAL_INTENT_CACHE_TTL_DAYS",
    "QUAL_INTENT_CONFIDENCE_THRESHOLD",
    "QUAL_INTENT_SIGNAL_DECAY_25_PCT_MONTHS",
    "QUAL_INTENT_SIGNAL_DECAY_50_PCT_MONTHS",
    "QUAL_LEADS_PER_ICP",
    "QUAL_MAX_COST_PER_LEAD_USD",
    "QUAL_MAX_TIME_PER_LEAD_SECONDS",
    "RESEARCH_LAB_ATTESTED_SCORING_LIVE_PROVIDER_ENABLED",
    "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS",
    "RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS",
    "RESEARCH_LAB_BASELINE_MAX_UNRESOLVED_ICPS",
    "RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY",
    "RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE",
    "RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES",
    "RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY",
    "RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY",
    "RESEARCH_LAB_EVAL_WORK_CONSERVING",
    "RESEARCH_LAB_GLOBAL_SCORING_POOL_SIZE",
    "RESEARCH_LAB_GLOBAL_SCORING_QUEUE",
    "RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS",
    "RESEARCH_LAB_LLM_INCLUDE_REASONING",
    "RESEARCH_LAB_OPENROUTER_GENERATION_ATTEMPTS",
    "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
    "RESEARCH_LAB_PROVIDER_COST_UNKNOWN_ENDPOINT_POLICY",
    "RESEARCH_LAB_CONDITIONAL_FRESH_ICP_COUNT",
    "RESEARCH_LAB_CONDITIONAL_HOLDOUT_TOTAL_ICPS",
    "RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE",
    "RESEARCH_LAB_PRIVATE_HOLDOUT_TOTAL_ICPS",
    "RESEARCH_LAB_PRIVATE_HOLDOUT_WEAK_TOTAL",
    "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS",
    "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_TOTAL",
    "RESEARCH_LAB_PUBLIC_SPLIT_UNBIASED",
    "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD",
    "RESEARCH_LAB_SCRAPINGDOG_UNKNOWN_ENDPOINT_CREDITS",
    "SCRAPINGDOG_PLAN_COST_USD",
    "SCRAPINGDOG_PLAN_CREDITS",
)

SCORING_SECRET_ENV_NAMES = (
    "EXA_API_KEY",
    "FULFILLMENT_OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "SCRAPINGDOG_API_KEY",
)
SCORING_RUNTIME_ENV_NAMES = tuple(
    sorted(set(SCORING_CONFIG_ENV_NAMES + SCORING_SECRET_ENV_NAMES))
)
MAX_RUNTIME_ENV_VALUE_BYTES = 16 * 1024
MAX_RUNTIME_ENV_TOTAL_BYTES = 128 * 1024


class ScoringExecutorError(ValueError):
    """Raised when a scoring operation or payload is unsupported."""


class ScoringExecutionResult:
    """Internal result plus evidence roots derived inside the enclave."""

    def __init__(self, result: Mapping[str, Any], evidence_roots: Mapping[str, str]) -> None:
        self.result = dict(result)
        self.evidence_roots = dict(evidence_roots)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScoringExecutorError("scoring value is not canonical JSON") from exc


def _manifest_configuration_env_names() -> tuple:
    # The import manifest records every literal env reference reachable from
    # broad shared modules, including unrelated gateway/validator operations.
    # Only this reviewed list can alter an enclave scoring operation.
    return SCORING_RUNTIME_ENV_NAMES


def normalize_runtime_environment(values: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(values, Mapping) or set(values) != set(SCORING_RUNTIME_ENV_NAMES):
        raise ScoringExecutorError("scoring runtime environment fields do not match the schema")
    normalized = {}
    total_bytes = 0
    for name in SCORING_RUNTIME_ENV_NAMES:
        value = values.get(name)
        if value is None:
            normalized[name] = None
            continue
        if not isinstance(value, str) or "\x00" in value:
            raise ScoringExecutorError("scoring runtime environment value is invalid")
        encoded_size = len(value.encode("utf-8"))
        if encoded_size > MAX_RUNTIME_ENV_VALUE_BYTES:
            raise ScoringExecutorError("scoring runtime environment value exceeds limit")
        total_bytes += encoded_size
        normalized[name] = value
    if total_bytes > MAX_RUNTIME_ENV_TOTAL_BYTES:
        raise ScoringExecutorError("scoring runtime environment exceeds total limit")
    return normalized


def runtime_environment_values() -> Dict[str, Any]:
    return {name: os.environ.get(name) for name in SCORING_RUNTIME_ENV_NAMES}


def configuration_snapshot(values: Mapping[str, Any] = None) -> Dict[str, Any]:
    source = (
        normalize_runtime_environment(values)
        if values is not None
        else runtime_environment_values()
    )
    environment = {}
    for name in _manifest_configuration_env_names():
        value = source.get(name)
        if name in SCORING_SECRET_ENV_NAMES:
            environment[name] = {
                "configured": bool(value),
                "value_sha256": (
                    "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
                    if value
                    else None
                ),
            }
        else:
            environment[name] = value
    from gateway.tee.egress_policy import destination_policy_hash

    return {
        "schema_version": SCORING_EXECUTOR_SCHEMA_VERSION,
        "environment": environment,
        "egress_policy_hash": destination_policy_hash(),
    }


def configuration_hash(values: Mapping[str, Any] = None) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(configuration_snapshot(values))).hexdigest()


def purpose_allowed_for_operation(operation: str, purpose: str) -> bool:
    candidate_or_baseline = {
        "research_lab.candidate_score.v1",
        "research_lab.baseline_score.v1",
        "research_lab.benchmark.v1",
        "research_lab.rebenchmark.v1",
    }
    allowed = {
        OP_QUALIFICATION_COMPANY_SCORES: candidate_or_baseline,
        OP_BENCHMARK_ICP_SCORE: candidate_or_baseline,
        OP_BUILD_SCORE_BUNDLE: candidate_or_baseline,
        OP_BUILD_BASELINE_SCORE_SUMMARY: {
            "research_lab.baseline_score.v1",
            "research_lab.benchmark.v1",
            "research_lab.rebenchmark.v1",
        },
        OP_PROMOTION_IMPROVEMENT: {"research_lab.promotion_metric.v1"},
        OP_PROMOTION_GATE_DECISION: {"research_lab.promotion_decision.v1"},
        OP_RESEARCH_LAB_ALLOCATION: {"research_lab.allocation.v1"},
    }
    return purpose in allowed.get(operation, set())


async def execute_scoring_operation(operation: str, payload: Mapping[str, Any]) -> Any:
    """Execute one existing pure/scoring entrypoint without changing its logic."""

    if operation not in SUPPORTED_OPERATIONS:
        raise ScoringExecutorError("unsupported scoring operation")
    if not isinstance(payload, Mapping):
        raise ScoringExecutorError("scoring payload must be an object")

    if operation == OP_QUALIFICATION_COMPANY_SCORES:
        provider_tape = payload.get("provider_tape")
        execution_mode = str(payload.get("provider_execution_mode") or "replay")
        if isinstance(provider_tape, Mapping):
            if execution_mode != "replay":
                raise ScoringExecutorError("provider tape is only valid in replay mode")
            from research_lab.eval.http_tape import replay_provider_http_tape

            with replay_provider_http_tape(provider_tape):
                return await _qualification_company_scores(payload)
        if execution_mode != "live_enclave":
            raise ScoringExecutorError("qualification scoring requires replay or live enclave evidence")
        from research_lab.eval.http_tape import record_provider_http_tape

        with record_provider_http_tape() as recorder:
            result = await _qualification_company_scores(payload)
        tape = recorder.document()
        return ScoringExecutionResult(
            result,
            {"provider_http_tape": str(tape["tape_hash"])},
        )
    if operation == OP_BENCHMARK_ICP_SCORE:
        from research_lab.eval.evaluator import benchmark_icp_score_from_company_scores

        scores = payload.get("scores")
        if not isinstance(scores, list):
            raise ScoringExecutorError("scores must be a list")
        return {"score": benchmark_icp_score_from_company_scores(scores)}
    if operation == OP_BUILD_SCORE_BUNDLE:
        from leadpoet_verifier.research_evaluation import score_bundle_hash
        from research_lab.eval.evaluator import build_score_bundle_from_scored_icps

        required = {
            "artifact_manifest",
            "benchmark",
            "patch_manifest",
            "per_icp_results",
            "run_context",
        }
        if not required.issubset(payload):
            raise ScoringExecutorError("score-bundle payload is incomplete")
        bundle = build_score_bundle_from_scored_icps(
            artifact_manifest=payload["artifact_manifest"],
            benchmark=payload["benchmark"],
            patch_manifest=payload["patch_manifest"],
            candidate_artifact_manifest=payload.get("candidate_artifact_manifest"),
            per_icp_results=payload["per_icp_results"],
            run_context=payload["run_context"],
            policy=payload.get("policy"),
            extra_bundle_fields=payload.get("extra_bundle_fields"),
        )
        bundle_hash = str(bundle.get("score_bundle_hash") or "")
        if bundle_hash != score_bundle_hash(bundle):
            raise ScoringExecutorError("score bundle hash does not match enclave output")
        evidence_roots = {"score_bundle": bundle_hash}
        holdout_gate = bundle.get("private_holdout_gate")
        if isinstance(holdout_gate, Mapping):
            baseline_hash = str(holdout_gate.get("baseline_benchmark_hash") or "")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", baseline_hash):
                evidence_roots["baseline_score_summary"] = baseline_hash
        return ScoringExecutionResult(
            {"score_bundle": bundle},
            evidence_roots,
        )
    if operation == OP_BUILD_BASELINE_SCORE_SUMMARY:
        from research_lab.eval.baseline_summary import build_baseline_score_summary

        required = {
            "artifact_manifest",
            "benchmark_date",
            "benchmark_attempt",
            "rolling_window_hash",
            "evaluation_epoch",
            "benchmark_items",
            "per_icp_summaries",
            "public_icps_per_day",
            "public_weak_per_day",
            "public_total_icps",
            "public_weak_total",
            "retried",
            "recovered",
            "max_unresolved_icps",
            "day_jump_points",
            "elapsed_seconds",
        }
        optional = {"conditional_validation_policy"}
        payload_fields = frozenset(payload)
        if payload_fields not in {frozenset(required), frozenset(required | optional)}:
            raise ScoringExecutorError("baseline-summary payload fields do not match schema")
        summary = build_baseline_score_summary(**dict(payload))
        summary_hash = "sha256:" + hashlib.sha256(
            _canonical_json(summary["score_summary_doc"])
        ).hexdigest()
        return ScoringExecutionResult(
            summary,
            {"baseline_score_summary": summary_hash},
        )
    if operation == OP_PROMOTION_IMPROVEMENT:
        from research_lab.eval.promotion_metric import promotion_improvement_metric

        score_bundle = payload.get("score_bundle")
        if not isinstance(score_bundle, Mapping):
            raise ScoringExecutorError("score_bundle must be an object")
        metric = promotion_improvement_metric(
            score_bundle,
            baseline_score_summary_doc=payload.get("baseline_score_summary_doc"),
        )
        result = {
            "improvement_points": metric.improvement_points,
            "event_doc": metric.event_doc(),
        }
        score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "")
        evidence_roots = (
            {"score_bundle": score_bundle_hash}
            if re.fullmatch(r"sha256:[0-9a-f]{64}", score_bundle_hash)
            else {}
        )
        return ScoringExecutionResult(result, evidence_roots)
    if operation == OP_PROMOTION_GATE_DECISION:
        from research_lab.eval.promotion_metric import promotion_gate_decision

        score_bundle = payload.get("score_bundle")
        if not isinstance(score_bundle, Mapping):
            raise ScoringExecutorError("score_bundle must be an object")
        required = {
            "score_bundle",
            "candidate_kind",
            "candidate_parent",
            "active_parent",
            "threshold_points",
            "auto_promotion_enabled",
        }
        if set(payload) != required:
            raise ScoringExecutorError("promotion-decision payload fields do not match schema")
        if not isinstance(payload.get("auto_promotion_enabled"), bool):
            raise ScoringExecutorError("auto_promotion_enabled must be a boolean")
        try:
            threshold_points = float(payload.get("threshold_points"))
        except (TypeError, ValueError) as exc:
            raise ScoringExecutorError("threshold_points must be numeric") from exc
        decision = promotion_gate_decision(
            score_bundle,
            candidate_kind=str(payload.get("candidate_kind") or ""),
            candidate_parent=str(payload.get("candidate_parent") or ""),
            active_parent=str(payload.get("active_parent") or ""),
            threshold_points=threshold_points,
            auto_promotion_enabled=payload["auto_promotion_enabled"],
        ).to_dict()
        score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "")
        evidence_roots = {
            "promotion_decision_status": "sha256:" + hashlib.sha256(
                _canonical_json({"status": decision["status"]})
            ).hexdigest(),
        }
        if re.fullmatch(r"sha256:[0-9a-f]{64}", score_bundle_hash):
            evidence_roots["score_bundle"] = score_bundle_hash
        return ScoringExecutionResult({"decision": decision}, evidence_roots)

    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = payload.get("policy")
    reimbursements = payload.get("active_reimbursement_obligations")
    champions = payload.get("active_champion_obligations")
    source_add = payload.get("active_source_add_obligations", [])
    if not isinstance(policy, Mapping):
        raise ScoringExecutorError("policy must be an object")
    if not isinstance(reimbursements, list) or not isinstance(champions, list) or not isinstance(source_add, list):
        raise ScoringExecutorError("allocation obligations must be lists")
    allocation = allocate_research_lab_epoch(
        int(payload.get("epoch", -1)),
        policy,
        reimbursements,
        champions,
        active_source_add_obligations=source_add,
    )
    allocation_hash = str(allocation.get("allocation_hash") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", allocation_hash):
        raise ScoringExecutorError("allocation hash is invalid")
    return ScoringExecutionResult(
        {"allocation": allocation},
        {"allocation": allocation_hash},
    )


async def _qualification_company_scores(payload: Mapping[str, Any]) -> Dict[str, Any]:
    from research_lab.eval.evaluator import QualificationStyleCompanyScorer

    companies = payload.get("companies")
    icp = payload.get("icp")
    if not isinstance(companies, list):
        raise ScoringExecutorError("companies must be a list")
    if not isinstance(icp, Mapping):
        raise ScoringExecutorError("icp must be an object")
    is_reference_model = payload.get("is_reference_model")
    if not isinstance(is_reference_model, bool):
        raise ScoringExecutorError("is_reference_model must be a boolean")

    scorer = QualificationStyleCompanyScorer()
    result = scorer.score_with_breakdowns(companies, icp, is_reference_model)
    if inspect.isawaitable(result):
        result = await result
    breakdowns = [dict(item) for item in result]
    return {
        "breakdowns": breakdowns,
        "scores": [float(item.get("final_score", 0.0) or 0.0) for item in breakdowns],
    }
