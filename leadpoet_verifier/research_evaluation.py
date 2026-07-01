"""Deterministic Research Lab evaluation bundle verification.

This module is intentionally private-model agnostic. It verifies the public /
anchored score bundle emitted by the private Research Lab evaluator without
requiring access to the sealed champion code or hidden benchmark text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .aggregation import aggregate_set_score, per_icp_normalized_score


SCORE_BUNDLE_TYPE = "research_lab_evaluation_score_bundle"
SCORE_BUNDLE_SCHEMA_VERSION = "1.0"
DEFAULT_LEADS_PER_ICP_NORMALIZER = 5
DEFAULT_LCB_Z = 1.96

SECRET_KEY_RE = re.compile(r"(api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|credential|token)", re.I)
SECRET_VALUE_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def compute_evaluation_aggregates(
    per_icp_results: Sequence[Mapping[str, Any]],
    *,
    leads_per_icp_normalizer: int = DEFAULT_LEADS_PER_ICP_NORMALIZER,
    lcb_z: float = DEFAULT_LCB_Z,
) -> dict[str, Any]:
    """Recompute paired base/candidate metrics from per-company scores."""
    if leads_per_icp_normalizer <= 0:
        raise ValueError("leads_per_icp_normalizer must be positive")
    if not per_icp_results:
        raise ValueError("evaluation requires at least one ICP result")

    normalized_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    base_scores: list[float] = []
    candidate_scores: list[float] = []
    hard_failures = 0
    successful_icps = 0

    for item in sorted(per_icp_results, key=lambda row: str(row.get("icp_ref") or row.get("icp_hash") or "")):
        base_company_scores = _coerce_scores(item.get("base_company_scores", ()))
        candidate_company_scores = _coerce_scores(item.get("candidate_company_scores", ()))
        base_score = per_icp_normalized_score(base_company_scores, max_leads=leads_per_icp_normalizer)
        candidate_score = per_icp_normalized_score(candidate_company_scores, max_leads=leads_per_icp_normalizer)
        delta = candidate_score - base_score
        hard_failure = bool(item.get("hard_failure", False))
        if hard_failure:
            hard_failures += 1
        else:
            successful_icps += 1
        row = {
            "icp_ref": str(item.get("icp_ref") or item.get("icp_hash") or ""),
            "icp_hash": str(item.get("icp_hash") or ""),
            "status": str(item.get("status") or ("failed" if hard_failure else "completed")),
            "hard_failure": hard_failure,
            "base_company_scores": base_company_scores,
            "candidate_company_scores": candidate_company_scores,
            "base_per_icp_score": round(base_score, 6),
            "candidate_per_icp_score": round(candidate_score, 6),
            "delta_vs_base": round(delta, 6),
            "failure_reason": str(item.get("failure_reason") or ""),
        }
        normalized_rows.append(row)
        base_scores.append(base_score)
        candidate_scores.append(candidate_score)
        deltas.append(delta)

    mean_delta = sum(deltas) / len(deltas)
    sd_delta = _sample_sd(deltas)
    se_delta = sd_delta / math.sqrt(len(deltas)) if deltas else 0.0
    delta_lcb = mean_delta - float(lcb_z) * se_delta
    return {
        "icp_count": len(normalized_rows),
        "successful_icp_count": successful_icps,
        "hard_failure_count": hard_failures,
        "base_score": round(aggregate_set_score(base_scores), 6),
        "candidate_score": round(aggregate_set_score(candidate_scores), 6),
        "mean_delta": round(mean_delta, 6),
        "sd_delta": round(sd_delta, 6),
        "se_delta": round(se_delta, 6),
        "delta_lcb": round(delta_lcb, 6),
        "leads_per_icp_normalizer": int(leads_per_icp_normalizer),
        "lcb_z": round(float(lcb_z), 6),
        "per_icp_results": normalized_rows,
    }


def evaluate_improvement_gate(
    aggregates: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic Research Lab improvement eligibility."""
    policy = policy or {}
    min_delta = float(policy.get("min_delta", 0.0))
    min_successful_icps = int(policy.get("min_successful_icps", 1))
    max_hard_failures = int(policy.get("max_hard_failures", 0))
    min_candidate_score = float(policy.get("min_candidate_score", 0.0))
    max_cost_usd = policy.get("max_cost_usd")
    cost_usd = float(aggregates.get("total_cost_usd", policy.get("observed_cost_usd", 0.0)) or 0.0)

    blockers: list[str] = []
    if float(aggregates["mean_delta"]) < min_delta:
        blockers.append("delta_below_minimum")
    if int(aggregates["successful_icp_count"]) < min_successful_icps:
        blockers.append("insufficient_successful_icps")
    if int(aggregates["hard_failure_count"]) > max_hard_failures:
        blockers.append("too_many_hard_failures")
    if float(aggregates["candidate_score"]) < min_candidate_score:
        blockers.append("candidate_score_below_floor")
    if max_cost_usd is not None and cost_usd > float(max_cost_usd):
        blockers.append("cost_above_policy_cap")

    return {
        "decision": "eligible_for_probation" if not blockers else "not_eligible",
        "eligible_for_probation": not blockers,
        "blockers": blockers,
        "policy": {
            "min_delta": min_delta,
            "min_successful_icps": min_successful_icps,
            "max_hard_failures": max_hard_failures,
            "min_candidate_score": min_candidate_score,
            "max_cost_usd": max_cost_usd,
        },
    }


def build_research_evaluation_score_bundle(
    *,
    run_id: str,
    ticket_id: str,
    miner_hotkey: str,
    island: str,
    evaluation_epoch: int,
    parent_artifact_hash: str,
    candidate_artifact_hash: str,
    private_model_manifest_hash: str,
    candidate_patch_hash: str,
    icp_set_hash: str,
    scoring_version: str,
    evaluator_version: str,
    per_icp_results: Sequence[Mapping[str, Any]],
    evidence_bundle_refs: Sequence[str],
    execution_trace_ref: str,
    cost_ledger_ref: str,
    benchmark_split_ref: str,
    candidate_model_manifest_hash: str | None = None,
    candidate_source_diff_hash: str | None = None,
    candidate_build_ref: str | None = None,
    policy: Mapping[str, Any] | None = None,
    signature_ref: str = "",
) -> dict[str, Any]:
    aggregates = compute_evaluation_aggregates(
        per_icp_results,
        leads_per_icp_normalizer=int((policy or {}).get("leads_per_icp_normalizer", DEFAULT_LEADS_PER_ICP_NORMALIZER)),
        lcb_z=float((policy or {}).get("lcb_z", DEFAULT_LCB_Z)),
    )
    total_cost_usd = float((policy or {}).get("observed_cost_usd", 0.0))
    aggregates = {**aggregates, "total_cost_usd": round(total_cost_usd, 6)}
    gate = evaluate_improvement_gate(aggregates, policy)
    bundle = {
        "schema_version": SCORE_BUNDLE_SCHEMA_VERSION,
        "bundle_type": SCORE_BUNDLE_TYPE,
        "run_id": str(run_id),
        "ticket_id": str(ticket_id),
        "miner_hotkey": str(miner_hotkey),
        "island": str(island),
        "evaluation_epoch": int(evaluation_epoch),
        "parent_artifact_hash": str(parent_artifact_hash),
        "candidate_artifact_hash": str(candidate_artifact_hash),
        "private_model_manifest_hash": str(private_model_manifest_hash),
        "candidate_patch_hash": str(candidate_patch_hash),
        "icp_set_hash": str(icp_set_hash),
        "scoring_version": str(scoring_version),
        "evaluator_version": str(evaluator_version),
        "benchmark_split_ref": str(benchmark_split_ref),
        "evidence_bundle_refs": list(evidence_bundle_refs),
        "execution_trace_ref": str(execution_trace_ref),
        "cost_ledger_ref": str(cost_ledger_ref),
        "aggregates": aggregates,
        "improvement_gate": gate,
        "score_bundle_hash": "",
        "anchored_hash": "",
        "signature_ref": str(signature_ref),
        "reward_path": {
            "eligible_for_probation": gate["eligible_for_probation"],
            "eligible_for_crown": False,
            "eligible_for_improvement_grant": False,
            "reason": "probation_gate_only; crown/grant require later gated workflow",
        },
    }
    if candidate_model_manifest_hash:
        bundle["candidate_model_manifest_hash"] = str(candidate_model_manifest_hash)
    if candidate_source_diff_hash:
        bundle["candidate_source_diff_hash"] = str(candidate_source_diff_hash)
    if candidate_build_ref:
        bundle["candidate_build_ref"] = str(candidate_build_ref)
    score_hash = score_bundle_hash(bundle)
    return {**bundle, "score_bundle_hash": score_hash, "anchored_hash": score_hash}


def score_bundle_hash(bundle: Mapping[str, Any]) -> str:
    return sha256_json(_score_bundle_hash_payload(bundle))


def verify_research_evaluation_score_bundle(
    bundle: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    require_signature_ref: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if _contains_secret_material(bundle):
        errors.append("score_bundle_contains_raw_secret_material")
    if bundle.get("schema_version") != SCORE_BUNDLE_SCHEMA_VERSION:
        errors.append("unsupported_score_bundle_schema_version")
    if bundle.get("bundle_type") != SCORE_BUNDLE_TYPE:
        errors.append("unsupported_score_bundle_type")
    if require_signature_ref and not bundle.get("signature_ref"):
        errors.append("score_bundle_missing_signature_ref")

    for field in (
        "run_id",
        "ticket_id",
        "miner_hotkey",
        "island",
        "parent_artifact_hash",
        "candidate_artifact_hash",
        "private_model_manifest_hash",
        "candidate_patch_hash",
        "icp_set_hash",
        "scoring_version",
        "evaluator_version",
        "aggregates",
        "improvement_gate",
        "score_bundle_hash",
    ):
        if field not in bundle:
            errors.append(f"missing_field:{field}")

    if bundle.get("parent_artifact_hash") == bundle.get("candidate_artifact_hash"):
        errors.append("candidate_artifact_must_differ_from_parent")
    for field in ("parent_artifact_hash", "candidate_artifact_hash", "private_model_manifest_hash", "candidate_patch_hash", "icp_set_hash"):
        value = str(bundle.get(field, ""))
        if not value.startswith("sha256:"):
            errors.append(f"{field}_must_be_sha256")

    claimed_hash = str(bundle.get("score_bundle_hash") or "")
    actual_hash = score_bundle_hash(bundle)
    if claimed_hash != actual_hash:
        errors.append("score_bundle_hash_mismatch")
    if bundle.get("anchored_hash") and bundle.get("anchored_hash") != claimed_hash:
        errors.append("anchored_hash_mismatch")

    aggregates = bundle.get("aggregates") if isinstance(bundle.get("aggregates"), Mapping) else {}
    per_icp_results = aggregates.get("per_icp_results", []) if isinstance(aggregates, Mapping) else []
    try:
        recomputed_aggregates = compute_evaluation_aggregates(
            per_icp_results,
            leads_per_icp_normalizer=int(aggregates.get("leads_per_icp_normalizer", DEFAULT_LEADS_PER_ICP_NORMALIZER)),
            lcb_z=float(aggregates.get("lcb_z", DEFAULT_LCB_Z)),
        )
        total_cost_usd = round(float(aggregates.get("total_cost_usd", 0.0) or 0.0), 6)
        recomputed_aggregates = {**recomputed_aggregates, "total_cost_usd": total_cost_usd}
        if _round_public(aggregates) != _round_public(recomputed_aggregates):
            errors.append("aggregates_do_not_match_recomputed_values")
        recomputed_gate = evaluate_improvement_gate(recomputed_aggregates, policy or bundle.get("improvement_gate", {}).get("policy", {}))
        if _round_public(bundle.get("improvement_gate", {})) != _round_public(recomputed_gate):
            errors.append("improvement_gate_does_not_match_recomputed_values")
    except Exception as exc:
        errors.append(f"aggregate_recompute_failed:{str(exc)[:120]}")

    reward_path = bundle.get("reward_path", {})
    if isinstance(reward_path, Mapping):
        if reward_path.get("eligible_for_improvement_grant") or reward_path.get("eligible_for_crown"):
            errors.append("score_bundle_must_not_directly_create_crown_or_grant")
    else:
        errors.append("reward_path_must_be_object")

    return {
        "passed": not errors,
        "errors": errors,
        "score_bundle_hash": actual_hash,
        "claimed_score_bundle_hash": claimed_hash,
        "run_id": bundle.get("run_id"),
        "ticket_id": bundle.get("ticket_id"),
        "miner_hotkey": bundle.get("miner_hotkey"),
        "island": bundle.get("island"),
        "evaluation_epoch": bundle.get("evaluation_epoch"),
        "eligible_for_probation": bool(bundle.get("improvement_gate", {}).get("eligible_for_probation", False)),
        "on_chain_submission_allowed": False,
    }


def score_bundle_to_weight_input(bundle: Mapping[str, Any]) -> dict[str, Any]:
    verification = verify_research_evaluation_score_bundle(bundle)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    aggregates = bundle["aggregates"]
    return {
        "run_id": str(bundle["run_id"]),
        "ticket_id": str(bundle["ticket_id"]),
        "miner_hotkey": str(bundle["miner_hotkey"]),
        "island": str(bundle["island"]),
        "evaluation_epoch": int(bundle["evaluation_epoch"]),
        "candidate_score": float(aggregates["candidate_score"]),
        "base_score": float(aggregates["base_score"]),
        "mean_delta": float(aggregates["mean_delta"]),
        "delta_lcb": float(aggregates["delta_lcb"]),
        "eligible_for_probation": bool(bundle["improvement_gate"]["eligible_for_probation"]),
        "score_bundle_hash": str(bundle["score_bundle_hash"]),
        "on_chain_submission_allowed": False,
    }


def _score_bundle_hash_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"score_bundle_hash", "anchored_hash", "signature", "signature_ref"}
    return {key: value for key, value in dict(bundle).items() if key not in excluded}


def _coerce_scores(value: Any) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("company score fields must be arrays")
    return [round(max(0.0, min(100.0, float(item))), 6) for item in value]


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                return True
            if _contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_VALUE_MARKERS)
    return False


def _round_public(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_public(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _round_public(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value
