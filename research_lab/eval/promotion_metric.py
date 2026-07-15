"""Pure Research Lab promotion metric shared by host and enclave execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PromotionImprovementMetric:
    improvement_points: float
    basis: str
    daily_baseline_available: bool
    baseline_aggregate_score: float | None = None
    candidate_total_score: float | None = None
    candidate_delta_vs_daily_baseline: float | None = None
    rejection_status: str | None = None
    provider_excluded_icp_ids: tuple[str, ...] = ()
    baseline_basis_adjusted: bool = False
    unadjusted_baseline_aggregate_score: float | None = None

    def event_doc(self) -> dict[str, Any]:
        return {
            "improvement_basis": self.basis,
            "daily_baseline_available": self.daily_baseline_available,
            "baseline_aggregate_score": self.baseline_aggregate_score,
            "candidate_total_score": self.candidate_total_score,
            "candidate_delta_vs_daily_baseline": self.candidate_delta_vs_daily_baseline,
            "rejection_status": self.rejection_status,
            "provider_excluded_icp_ids": list(self.provider_excluded_icp_ids),
            "baseline_basis_adjusted": self.baseline_basis_adjusted,
            "unadjusted_baseline_aggregate_score": self.unadjusted_baseline_aggregate_score,
        }


@dataclass(frozen=True)
class PromotionGateDecision:
    """Pure projection of the existing pre-merge promotion gates."""

    status: str
    improvement_points: float
    threshold_points: float
    candidate_kind: str
    auto_promotion_enabled: bool
    active_parent_matches: bool
    metric_rejection_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "improvement_points": self.improvement_points,
            "threshold_points": self.threshold_points,
            "candidate_kind": self.candidate_kind,
            "auto_promotion_enabled": self.auto_promotion_enabled,
            "active_parent_matches": self.active_parent_matches,
            "metric_rejection_status": self.metric_rejection_status,
        }


def preliminary_promotion_gate_projection(
    holdout_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the frozen public/private result through the legacy 20-ICP gate.

    Conditional validation is an extra stage after the existing promotion
    decision.  This projection deliberately presents the already-computed
    public/private aggregate as the legacy ``private_holdout_approved`` shape
    so the unchanged promotion metric and gate functions remain authoritative
    at that boundary.
    """

    if not isinstance(holdout_gate, Mapping):
        raise ValueError("preliminary promotion gate must be an object")
    if not bool(holdout_gate.get("conditional_validation_required")):
        raise ValueError("preliminary promotion gate requires conditional validation")
    if str(holdout_gate.get("decision") or "") != "conditional_validation_required":
        raise ValueError("preliminary promotion gate has not cleared the 20-ICP score gate")
    if not bool(holdout_gate.get("private_holdout_evaluated")):
        raise ValueError("preliminary promotion gate requires a completed private holdout")

    baseline = _required_finite_score(
        holdout_gate.get("baseline_preliminary_score"),
        "baseline_preliminary_score",
    )
    candidate = _required_finite_score(
        holdout_gate.get("candidate_preliminary_score"),
        "candidate_preliminary_score",
    )
    delta = candidate - baseline
    supplied_delta = _optional_float(holdout_gate.get("candidate_preliminary_delta"))
    if supplied_delta is None or not math.isfinite(supplied_delta):
        raise ValueError("candidate_preliminary_delta must be finite")
    if abs(supplied_delta - delta) > 1e-6:
        raise ValueError("candidate preliminary score commitment differs")

    return {
        **dict(holdout_gate),
        "decision": "private_holdout_approved",
        "baseline_aggregate_score": round(baseline, 6),
        "candidate_total_score": round(candidate, 6),
        "candidate_delta_vs_daily_baseline": round(delta, 6),
        "private_holdout_evaluated": True,
        "conditional_holdout_evaluated": False,
        "conditional_validation_required": False,
        "preliminary_decision": "private_holdout_approved",
        "final_decision": "",
    }


def promotion_improvement_metric(
    score_bundle: Mapping[str, Any],
    *,
    baseline_score_summary_doc: Mapping[str, Any] | None = None,
) -> PromotionImprovementMetric:
    """Return the existing score-only promotion metric without host I/O."""

    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    gate = score_bundle.get("private_holdout_gate")
    if isinstance(gate, Mapping):
        decision = str(gate.get("decision") or "")
        baseline_aggregate = _optional_float(gate.get("baseline_aggregate_score"))
        candidate_total = _optional_float(gate.get("candidate_total_score"))
        excluded_icp_ids = _provider_excluded_icp_ids(aggregates)
        daily_delta = _optional_float(gate.get("candidate_delta_vs_daily_baseline"))
        if daily_delta is None and baseline_aggregate is not None and candidate_total is not None:
            daily_delta = candidate_total - baseline_aggregate
        if bool(gate.get("conditional_validation_required")):
            if (
                decision == "conditional_validation_approved"
                and bool(gate.get("conditional_holdout_evaluated"))
                and daily_delta is not None
            ):
                return PromotionImprovementMetric(
                    improvement_points=float(daily_delta),
                    basis="stored_daily_baseline_conditional_full_bank_delta",
                    daily_baseline_available=True,
                    baseline_aggregate_score=baseline_aggregate,
                    candidate_total_score=candidate_total,
                    candidate_delta_vs_daily_baseline=float(daily_delta),
                    provider_excluded_icp_ids=excluded_icp_ids,
                    baseline_basis_adjusted=False,
                    unadjusted_baseline_aggregate_score=None,
                )
            return PromotionImprovementMetric(
                improvement_points=float(daily_delta or 0.0),
                basis="stored_daily_baseline_conditional_unavailable:%s"
                % (decision or "missing_decision"),
                daily_baseline_available=False,
                baseline_aggregate_score=baseline_aggregate,
                candidate_total_score=candidate_total,
                candidate_delta_vs_daily_baseline=daily_delta,
                rejection_status=(
                    decision
                    if decision in {
                        "rejected_before_private_holdout",
                        "rejected_before_conditional_validation",
                        "rejected_after_conditional_validation",
                    }
                    else "conditional_validation_incomplete"
                ),
                provider_excluded_icp_ids=excluded_icp_ids,
                baseline_basis_adjusted=False,
                unadjusted_baseline_aggregate_score=None,
            )
        if (
            decision == "private_holdout_approved"
            and bool(gate.get("private_holdout_evaluated"))
            and daily_delta is not None
        ):
            return PromotionImprovementMetric(
                improvement_points=float(daily_delta),
                basis="stored_daily_baseline_total_delta",
                daily_baseline_available=True,
                baseline_aggregate_score=baseline_aggregate,
                candidate_total_score=candidate_total,
                candidate_delta_vs_daily_baseline=float(daily_delta),
                provider_excluded_icp_ids=excluded_icp_ids,
                baseline_basis_adjusted=False,
                unadjusted_baseline_aggregate_score=None,
            )
        unavailable_reason = decision or "missing_decision"
        return PromotionImprovementMetric(
            improvement_points=0.0,
            basis="stored_daily_baseline_unavailable:%s" % unavailable_reason,
            daily_baseline_available=False,
            baseline_aggregate_score=baseline_aggregate,
            candidate_total_score=candidate_total,
            candidate_delta_vs_daily_baseline=daily_delta,
            rejection_status="rejected_basis_unavailable",
            provider_excluded_icp_ids=excluded_icp_ids,
            baseline_basis_adjusted=False,
            unadjusted_baseline_aggregate_score=None,
        )

    legacy_delta = _optional_float(aggregates.get("mean_delta")) or 0.0
    return PromotionImprovementMetric(
        improvement_points=float(legacy_delta),
        basis="legacy_paired_mean_delta_no_holdout_gate",
        daily_baseline_available=False,
    )


def promotion_gate_decision(
    score_bundle: Mapping[str, Any],
    *,
    candidate_kind: str,
    candidate_parent: str,
    active_parent: str,
    threshold_points: float,
    auto_promotion_enabled: bool,
) -> PromotionGateDecision:
    """Return the existing score/kind/parent gate outcome without side effects."""

    metric = promotion_improvement_metric(score_bundle)
    improvement_points = float(metric.improvement_points)
    threshold = float(threshold_points)
    normalized_kind = str(candidate_kind or "patch")
    parents_match = str(candidate_parent or "") == str(active_parent or "")

    if not auto_promotion_enabled:
        status = "disabled"
    elif normalized_kind != "image_build":
        status = "rejected_legacy_patch_candidate"
    elif metric.rejection_status:
        status = str(metric.rejection_status)
    elif improvement_points < threshold:
        status = "rejected_below_threshold"
    elif not parents_match:
        status = "stale_parent_needs_rescore"
    else:
        status = "promotion_passed"

    return PromotionGateDecision(
        status=status,
        improvement_points=improvement_points,
        threshold_points=threshold,
        candidate_kind=normalized_kind,
        auto_promotion_enabled=bool(auto_promotion_enabled),
        active_parent_matches=parents_match,
        metric_rejection_status=metric.rejection_status,
    )


def _provider_excluded_icp_ids(aggregates: Mapping[str, Any]) -> tuple[str, ...]:
    value = aggregates.get("provider_excluded_icp_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    seen = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _required_finite_score(value: Any, field: str) -> float:
    parsed = _optional_float(value)
    if parsed is None or not math.isfinite(parsed) or parsed < 0.0 or parsed > 100.0:
        raise ValueError(f"{field} must be finite and within 0-100")
    return parsed
