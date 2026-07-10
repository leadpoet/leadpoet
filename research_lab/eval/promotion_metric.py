"""Pure Research Lab promotion metric shared by host and enclave execution."""

from __future__ import annotations

from dataclasses import dataclass
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
