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

# Reference-evaluation modes (version gate for the 52a9edce metric change).
#
# Historical bundles ran a paired base/candidate evaluation: ``base_company_scores``
# are populated per ICP and ``aggregates.mean_delta``/``delta_lcb`` are true paired
# deltas. Post-52a9edce bundles no longer run the paired base: every
# ``base_company_scores`` array is empty and the recorded ``mean_delta``/``delta_lcb``
# silently become candidate-vs-zero (absolute score, ~+12-17), which would make the
# advisory ``improvement_gate`` read always-eligible. Those bundles carry
# ``reference_evaluation_mode: "stored_daily_baseline"`` (in the bundle's
# ``private_holdout_gate`` and/or ``aggregates``); all-empty base scores are the
# tell-tale fallback for bundles created in that window before the field landed.
# Historical bundles are never rewritten -- readers version-gate on this mode.
REFERENCE_EVALUATION_MODE_PAIRED = "paired_base"
REFERENCE_EVALUATION_MODE_DAILY_BASELINE = "stored_daily_baseline"
IMPROVEMENT_GATE_NOT_APPLICABLE = "not_applicable"
SUPERSEDED_METRIC_BLOCKER = "superseded_metric_daily_baseline_reference"

# Non-arithmetic annotation keys that live in ``aggregates`` but are not produced
# by :func:`compute_evaluation_aggregates`; they are carried over verbatim before
# the recorded-vs-recomputed equality check.
_AGGREGATE_ANNOTATION_KEYS = (
    "reference_evaluation_mode",
    "provider_excluded_icp_ids",
    "baseline_per_icp_scores",
)

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


# trajectoryimprovements.md P12/P13: capture pointers and grading rows must
# survive the aggregation rebuild — before this passthrough, per-ICP trace
# refs written by the evaluator were stripped here and never reached the
# bundle the trajectory projector reads (the "orphaned in S3" gap).
PER_ICP_CAPTURE_PASSTHROUGH_KEYS = (
    "incontainer_trace_ref",
    "incontainer_trace_sha256",
    "incontainer_trace_call_count",
    "incontainer_trace_truncated_count",
    "incontainer_trace_dropped",
    "incontainer_trace_dropped_call_count",
    "scorer_trace_ref",
    "scorer_trace_sha256",
    "l0_findings",
    "provider_excluded",
    "sourced_zero_no_error",
    "reference_sourced_zero_no_error",
)


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
        for key in PER_ICP_CAPTURE_PASSTHROUGH_KEYS:
            if key in item:
                row[key] = item[key]
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


def detect_reference_evaluation_mode(bundle: Mapping[str, Any]) -> str:
    """Detect whether a bundle was scored under the paired or daily-baseline metric.

    An explicitly declared ``reference_evaluation_mode`` (top level, ``aggregates``,
    or ``private_holdout_gate``) wins. Any declared mode other than ``paired_base``
    fails closed to ``stored_daily_baseline`` treatment. Without a declared mode,
    all-empty ``base_company_scores`` across every per-ICP row is the tell-tale for
    a post-52a9edce bundle (no paired base run happened): its recorded deltas are
    candidate-vs-zero and must never be read as paired improvements.
    """
    aggregates = bundle.get("aggregates") if isinstance(bundle.get("aggregates"), Mapping) else {}
    gate = bundle.get("private_holdout_gate") if isinstance(bundle.get("private_holdout_gate"), Mapping) else {}
    for holder in (bundle, aggregates, gate):
        declared = str(holder.get("reference_evaluation_mode") or "").strip()
        if declared:
            if declared == REFERENCE_EVALUATION_MODE_PAIRED:
                return REFERENCE_EVALUATION_MODE_PAIRED
            return REFERENCE_EVALUATION_MODE_DAILY_BASELINE
    rows = aggregates.get("per_icp_results") if isinstance(aggregates, Mapping) else None
    if _per_icp_rows_all_empty_base(rows if isinstance(rows, Sequence) else ()):
        return REFERENCE_EVALUATION_MODE_DAILY_BASELINE
    return REFERENCE_EVALUATION_MODE_PAIRED


def _builder_reference_evaluation_mode(
    aggregates: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
) -> str:
    declared = str((policy or {}).get("reference_evaluation_mode") or "").strip()
    if declared:
        if declared == REFERENCE_EVALUATION_MODE_PAIRED:
            return REFERENCE_EVALUATION_MODE_PAIRED
        return REFERENCE_EVALUATION_MODE_DAILY_BASELINE
    rows = aggregates.get("per_icp_results")
    if _per_icp_rows_all_empty_base(rows if isinstance(rows, Sequence) else ()):
        return REFERENCE_EVALUATION_MODE_DAILY_BASELINE
    return REFERENCE_EVALUATION_MODE_PAIRED


def _per_icp_rows_all_empty_base(rows: Sequence[Any]) -> bool:
    mapped = [row for row in rows if isinstance(row, Mapping)]
    if not mapped:
        return False
    for row in mapped:
        base = row.get("base_company_scores")
        if isinstance(base, Sequence) and not isinstance(base, (str, bytes, bytearray)) and len(base) > 0:
            return False
    return True


def _normalize_baseline_per_icp_scores(value: Any) -> dict[str, float] | None:
    """Normalize stored per-ICP daily-baseline scores to ``{icp_ref: score}``.

    Accepts either a mapping of ``icp_ref -> score`` or a list of rows carrying
    ``icp_ref``/``icp_hash`` and ``score`` (the daily benchmark doc's
    ``visibility_split.items`` shape). Returns ``None`` when nothing usable is
    present.
    """
    out: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            try:
                out[str(key)] = round(float(item), 6)
            except (TypeError, ValueError):
                continue
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or item.get("icp_hash") or "").strip()
            if not ref:
                continue
            try:
                out[ref] = round(float(item.get("score")), 6)
            except (TypeError, ValueError):
                continue
    return out or None


def _extract_baseline_per_icp_scores(bundle: Mapping[str, Any]) -> dict[str, float] | None:
    aggregates = bundle.get("aggregates") if isinstance(bundle.get("aggregates"), Mapping) else {}
    gate = bundle.get("private_holdout_gate") if isinstance(bundle.get("private_holdout_gate"), Mapping) else {}
    for holder in (gate, aggregates, bundle):
        normalized = _normalize_baseline_per_icp_scores(holder.get("baseline_per_icp_scores"))
        if normalized:
            return normalized
    return None


def _normalize_ref_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    refs = sorted({str(item).strip() for item in value if str(item).strip()})
    return tuple(refs)


def _extract_provider_excluded_icp_ids(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    """Read ``provider_excluded_icp_ids`` from a bundle, tolerating absence.

    The evaluator records ICPs excluded from candidate totals because of
    unresolved provider flakes. They are skipped on both sides of the advisory
    delta recompute and counted as unsuccessful for ``min_successful_icps``.
    """
    aggregates = bundle.get("aggregates") if isinstance(bundle.get("aggregates"), Mapping) else {}
    gate = bundle.get("private_holdout_gate") if isinstance(bundle.get("private_holdout_gate"), Mapping) else {}
    for holder in (aggregates, gate, bundle):
        value = holder.get("provider_excluded_icp_ids")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return _normalize_ref_list(value)
    return ()


def _row_refs(row: Mapping[str, Any]) -> tuple[str, ...]:
    refs = []
    for key in ("icp_ref", "icp_hash"):
        value = str(row.get(key) or "").strip()
        if value:
            refs.append(value)
    return tuple(refs)


def _not_applicable_improvement_gate(
    policy: Mapping[str, Any] | None,
    *,
    reason: str,
    extra_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    normalized_policy = evaluate_improvement_gate(
        {
            "mean_delta": 0.0,
            "successful_icp_count": 0,
            "hard_failure_count": 0,
            "candidate_score": 0.0,
        },
        policy,
    )["policy"]
    return {
        "decision": IMPROVEMENT_GATE_NOT_APPLICABLE,
        "eligible_for_probation": False,
        "blockers": [SUPERSEDED_METRIC_BLOCKER, *extra_blockers],
        "reason": reason,
        "reference_evaluation_mode": REFERENCE_EVALUATION_MODE_DAILY_BASELINE,
        "advisory_basis": "superseded_metric_not_applicable",
        "policy": normalized_policy,
    }


def evaluate_daily_baseline_improvement_gate(
    aggregates: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    *,
    baseline_per_icp_scores: Mapping[str, float] | None = None,
    provider_excluded_icp_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Advisory improvement gate for daily-baseline (post-paired) bundles.

    With stored per-ICP daily-baseline scores available, the per-ICP delta is
    recomputed as candidate-per-ICP minus stored-baseline-per-ICP (skipping
    provider-excluded ICPs on both sides; those count as unsuccessful for
    ``min_successful_icps``). Without them, the paired-delta gate is explicitly
    ``not_applicable`` and NEVER eligible: the recorded candidate-vs-zero
    ``mean_delta``/``delta_lcb`` must not be consumed as improvement deltas.
    """
    if baseline_per_icp_scores is None:
        baseline_per_icp_scores = _normalize_baseline_per_icp_scores(
            aggregates.get("baseline_per_icp_scores")
        )
    if provider_excluded_icp_ids is None:
        provider_excluded_icp_ids = _normalize_ref_list(
            aggregates.get("provider_excluded_icp_ids")
        )
    excluded = set(_normalize_ref_list(provider_excluded_icp_ids))

    if not baseline_per_icp_scores:
        return _not_applicable_improvement_gate(
            policy,
            reason=(
                "daily_baseline_bundle_without_per_icp_baseline_scores: recorded "
                "mean_delta/delta_lcb are candidate-vs-zero absolute scores, not "
                "paired improvement deltas"
            ),
        )

    rows = [
        row
        for row in (aggregates.get("per_icp_results") or ())
        if isinstance(row, Mapping)
    ]
    if not rows:
        return _not_applicable_improvement_gate(
            policy,
            reason="daily_baseline_bundle_without_per_icp_results",
        )

    normalizer = int(aggregates.get("leads_per_icp_normalizer", DEFAULT_LEADS_PER_ICP_NORMALIZER))
    lcb_z = float(aggregates.get("lcb_z", DEFAULT_LCB_Z))
    deltas: list[float] = []
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    successful = 0
    hard_failures = 0
    excluded_applied: set[str] = set()
    missing_baseline_refs: list[str] = []

    for row in sorted(rows, key=lambda item: str(item.get("icp_ref") or item.get("icp_hash") or "")):
        refs = _row_refs(row)
        if any(ref in excluded for ref in refs):
            excluded_applied.update(ref for ref in refs if ref in excluded)
            continue
        baseline_score = None
        for ref in refs:
            if ref in baseline_per_icp_scores:
                baseline_score = float(baseline_per_icp_scores[ref])
                break
        if baseline_score is None:
            missing_baseline_refs.append(refs[0] if refs else "")
            continue
        try:
            candidate_score = float(row["candidate_per_icp_score"])
        except (KeyError, TypeError, ValueError):
            candidate_score = per_icp_normalized_score(
                _coerce_scores(row.get("candidate_company_scores", ())),
                max_leads=normalizer,
            )
        deltas.append(candidate_score - baseline_score)
        candidate_scores.append(candidate_score)
        baseline_scores.append(baseline_score)
        if bool(row.get("hard_failure", False)):
            hard_failures += 1
        else:
            successful += 1

    if missing_baseline_refs:
        preview = ",".join(sorted(ref for ref in missing_baseline_refs if ref)[:5])
        return _not_applicable_improvement_gate(
            policy,
            reason=f"baseline_per_icp_scores_missing_for:{preview}",
            extra_blockers=("baseline_per_icp_coverage_incomplete",),
        )
    if not deltas:
        return _not_applicable_improvement_gate(
            policy,
            reason="all_per_icp_results_provider_excluded",
        )

    mean_delta = sum(deltas) / len(deltas)
    sd_delta = _sample_sd(deltas)
    se_delta = sd_delta / math.sqrt(len(deltas)) if deltas else 0.0
    delta_lcb = mean_delta - lcb_z * se_delta
    candidate_mean = sum(candidate_scores) / len(candidate_scores)
    baseline_mean = sum(baseline_scores) / len(baseline_scores)

    base_gate = evaluate_improvement_gate(
        {
            "mean_delta": mean_delta,
            "successful_icp_count": successful,
            "hard_failure_count": hard_failures,
            "candidate_score": candidate_mean,
            "total_cost_usd": aggregates.get("total_cost_usd", 0.0),
        },
        policy,
    )
    return {
        **base_gate,
        "reference_evaluation_mode": REFERENCE_EVALUATION_MODE_DAILY_BASELINE,
        "advisory_basis": "recomputed_candidate_vs_stored_daily_baseline_per_icp",
        "reason": "",
        "mean_delta": round(mean_delta, 6),
        "sd_delta": round(sd_delta, 6),
        "se_delta": round(se_delta, 6),
        "delta_lcb": round(delta_lcb, 6),
        "lcb_z": round(lcb_z, 6),
        "baseline_score": round(baseline_mean, 6),
        "candidate_score": round(candidate_mean, 6),
        "compared_icp_count": len(deltas),
        "successful_icp_count": successful,
        "hard_failure_count": hard_failures,
        "provider_excluded_icp_count": len(excluded_applied),
        "provider_excluded_icp_ids": sorted(excluded_applied),
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
    mode = _builder_reference_evaluation_mode(aggregates, policy)
    if mode == REFERENCE_EVALUATION_MODE_DAILY_BASELINE:
        # Version-gate the recorded advisory gate: with no paired base run the
        # naive gate would compare candidate-vs-zero and read always-eligible.
        annotations: dict[str, Any] = {
            "reference_evaluation_mode": REFERENCE_EVALUATION_MODE_DAILY_BASELINE,
        }
        baseline_scores = _normalize_baseline_per_icp_scores(
            (policy or {}).get("baseline_per_icp_scores")
        )
        if baseline_scores:
            annotations["baseline_per_icp_scores"] = baseline_scores
        excluded_refs = _normalize_ref_list((policy or {}).get("provider_excluded_icp_ids"))
        if excluded_refs:
            annotations["provider_excluded_icp_ids"] = list(excluded_refs)
        aggregates = {**aggregates, **annotations}
        gate = evaluate_daily_baseline_improvement_gate(aggregates, policy)
    else:
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
    reference_evaluation_mode = detect_reference_evaluation_mode(bundle)
    advisory_improvement_gate: dict[str, Any] | None = None
    try:
        recomputed_aggregates = compute_evaluation_aggregates(
            per_icp_results,
            leads_per_icp_normalizer=int(aggregates.get("leads_per_icp_normalizer", DEFAULT_LEADS_PER_ICP_NORMALIZER)),
            lcb_z=float(aggregates.get("lcb_z", DEFAULT_LCB_Z)),
        )
        total_cost_usd = round(float(aggregates.get("total_cost_usd", 0.0) or 0.0), 6)
        recomputed_aggregates = {**recomputed_aggregates, "total_cost_usd": total_cost_usd}
        # Annotation keys are recorded by the evaluator but are not arithmetic
        # outputs of compute_evaluation_aggregates; carry them over verbatim so
        # the equality check keeps guarding the recomputable fields only.
        for key in _AGGREGATE_ANNOTATION_KEYS:
            if key in aggregates:
                recomputed_aggregates[key] = aggregates[key]
        if _round_public(aggregates) != _round_public(recomputed_aggregates):
            errors.append("aggregates_do_not_match_recomputed_values")
        gate_policy = policy or bundle.get("improvement_gate", {}).get("policy", {})
        if reference_evaluation_mode == REFERENCE_EVALUATION_MODE_PAIRED:
            recomputed_gate = evaluate_improvement_gate(recomputed_aggregates, gate_policy)
            if _round_public(bundle.get("improvement_gate", {})) != _round_public(recomputed_gate):
                errors.append("improvement_gate_does_not_match_recomputed_values")
        else:
            # Daily-baseline mode (post-52a9edce): the recorded gate's naive
            # candidate-vs-zero recompute is superseded. Recompute the advisory
            # verdict against stored per-ICP daily-baseline scores when the
            # bundle carries them; otherwise the verdict is not_applicable and
            # never eligible. The recorded gate stays covered by the bundle hash.
            advisory_improvement_gate = evaluate_daily_baseline_improvement_gate(
                recomputed_aggregates,
                gate_policy,
                baseline_per_icp_scores=_extract_baseline_per_icp_scores(bundle),
                provider_excluded_icp_ids=_extract_provider_excluded_icp_ids(bundle),
            )
    except Exception as exc:
        errors.append(f"aggregate_recompute_failed:{str(exc)[:120]}")

    reward_path = bundle.get("reward_path", {})
    if isinstance(reward_path, Mapping):
        if reward_path.get("eligible_for_improvement_grant") or reward_path.get("eligible_for_crown"):
            errors.append("score_bundle_must_not_directly_create_crown_or_grant")
    else:
        errors.append("reward_path_must_be_object")

    if reference_evaluation_mode == REFERENCE_EVALUATION_MODE_PAIRED:
        eligible_for_probation = bool(bundle.get("improvement_gate", {}).get("eligible_for_probation", False))
        eligible_for_probation_reason = ""
    elif advisory_improvement_gate is not None:
        eligible_for_probation = bool(advisory_improvement_gate.get("eligible_for_probation", False))
        eligible_for_probation_reason = str(advisory_improvement_gate.get("reason") or "") or str(
            advisory_improvement_gate.get("advisory_basis") or ""
        )
    else:
        eligible_for_probation = False
        eligible_for_probation_reason = "daily_baseline_mode_advisory_gate_unavailable"

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
        "reference_evaluation_mode": reference_evaluation_mode,
        "advisory_improvement_gate": advisory_improvement_gate,
        "eligible_for_probation": eligible_for_probation,
        "eligible_for_probation_reason": eligible_for_probation_reason,
        "on_chain_submission_allowed": False,
    }


def score_bundle_to_weight_input(bundle: Mapping[str, Any]) -> dict[str, Any]:
    verification = verify_research_evaluation_score_bundle(bundle)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    aggregates = bundle["aggregates"]
    reference_evaluation_mode = str(verification["reference_evaluation_mode"])
    result = {
        "run_id": str(bundle["run_id"]),
        "ticket_id": str(bundle["ticket_id"]),
        "miner_hotkey": str(bundle["miner_hotkey"]),
        "island": str(bundle["island"]),
        "evaluation_epoch": int(bundle["evaluation_epoch"]),
        "candidate_score": float(aggregates["candidate_score"]),
        "base_score": float(aggregates["base_score"]),
        "mean_delta": float(aggregates["mean_delta"]),
        "delta_lcb": float(aggregates["delta_lcb"]),
        "reference_evaluation_mode": reference_evaluation_mode,
        "eligible_for_probation": bool(verification["eligible_for_probation"]),
        "score_bundle_hash": str(bundle["score_bundle_hash"]),
        "on_chain_submission_allowed": False,
    }
    if reference_evaluation_mode != REFERENCE_EVALUATION_MODE_PAIRED:
        # Version gate: the recorded mean_delta/delta_lcb of a daily-baseline
        # bundle are candidate-vs-zero absolute scores. Expose them only under
        # recorded_* keys and replace the delta fields with the advisory
        # recompute against the stored per-ICP daily baseline (or zero when the
        # metric is superseded and not recomputable).
        advisory = verification.get("advisory_improvement_gate") or {}
        result["recorded_mean_delta"] = float(aggregates["mean_delta"])
        result["recorded_delta_lcb"] = float(aggregates["delta_lcb"])
        result["recorded_base_score"] = float(aggregates["base_score"])
        if str(advisory.get("advisory_basis") or "").startswith("recomputed"):
            result["mean_delta"] = float(advisory["mean_delta"])
            result["delta_lcb"] = float(advisory["delta_lcb"])
            result["base_score"] = float(advisory["baseline_score"])
            result["delta_metrics_basis"] = "recomputed_vs_stored_daily_baseline"
        else:
            result["mean_delta"] = 0.0
            result["delta_lcb"] = 0.0
            result["delta_metrics_basis"] = "superseded_metric_not_applicable"
    return result


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
