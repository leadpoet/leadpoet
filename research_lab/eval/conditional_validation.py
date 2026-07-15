"""Pure policy and assignment logic for conditional champion validation.

The gateway host and measured scoring executor import this module.  It must
remain free of database, network, Docker, and environment access so both sides
commit to the same category assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


CONDITIONAL_VALIDATION_MODE_OFF = "off"
CONDITIONAL_VALIDATION_MODE_ENFORCE = "enforce"
CONDITIONAL_VALIDATION_POLICY_VERSION = "research_lab_conditional_validation.v1"
CATEGORY_ASSIGNMENT_VERSION = "research_lab_icp_category_assignment.v1"


@dataclass(frozen=True)
class ConditionalValidationPolicy:
    """One validated source of truth for the 40-ICP policy."""

    mode: str = CONDITIONAL_VALIDATION_MODE_OFF
    public_total_icps: int = 10
    public_weak_total: int = 7
    private_total_icps: int = 10
    private_weak_total: int = 3
    conditional_total_icps: int = 20
    fresh_icp_count: int = 20
    threshold_points: float = 1.0
    policy_version: str = CONDITIONAL_VALIDATION_POLICY_VERSION

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode or "").strip().lower()
        object.__setattr__(self, "mode", normalized_mode)
        if normalized_mode not in {
            CONDITIONAL_VALIDATION_MODE_OFF,
            CONDITIONAL_VALIDATION_MODE_ENFORCE,
        }:
            raise ValueError("RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE must be off or enforce")
        for field_name in (
            "public_total_icps",
            "private_total_icps",
            "conditional_total_icps",
            "fresh_icp_count",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for weak_name, total_name in (
            ("public_weak_total", "public_total_icps"),
            ("private_weak_total", "private_total_icps"),
        ):
            weak = int(getattr(self, weak_name))
            total = int(getattr(self, total_name))
            if weak < 0 or weak > total:
                raise ValueError(f"{weak_name} must be between zero and {total_name}")
        if self.fresh_icp_count > self.total_icps:
            raise ValueError("fresh_icp_count cannot exceed the configured ICP bank size")
        if self.retained_icp_count <= 0:
            raise ValueError("conditional validation requires at least one retained ICP")
        if self.low_tail_count != self.high_tail_count:
            raise ValueError(
                "public/private weak totals must leave equal lower and upper tails"
            )
        if (
            not math.isfinite(float(self.threshold_points))
            or float(self.threshold_points) < 0.0
            or float(self.threshold_points) > 100.0
        ):
            raise ValueError("threshold_points must be between zero and 100")

    @property
    def enabled(self) -> bool:
        return self.mode == CONDITIONAL_VALIDATION_MODE_ENFORCE

    @property
    def total_icps(self) -> int:
        return (
            int(self.public_total_icps)
            + int(self.private_total_icps)
            + int(self.conditional_total_icps)
        )

    @property
    def retained_icp_count(self) -> int:
        return self.total_icps - int(self.fresh_icp_count)

    @property
    def low_tail_count(self) -> int:
        return int(self.public_weak_total) + int(self.private_weak_total)

    @property
    def public_strong_total(self) -> int:
        return int(self.public_total_icps) - int(self.public_weak_total)

    @property
    def private_strong_total(self) -> int:
        return int(self.private_total_icps) - int(self.private_weak_total)

    @property
    def high_tail_count(self) -> int:
        return self.public_strong_total + self.private_strong_total

    def to_dict(self) -> dict[str, Any]:
        doc = {
            "schema_version": self.policy_version,
            "mode": self.mode,
            "public_total_icps": int(self.public_total_icps),
            "public_weak_total": int(self.public_weak_total),
            "public_strong_total": self.public_strong_total,
            "private_total_icps": int(self.private_total_icps),
            "private_weak_total": int(self.private_weak_total),
            "private_strong_total": self.private_strong_total,
            "conditional_total_icps": int(self.conditional_total_icps),
            "total_icps": self.total_icps,
            "fresh_icp_count": int(self.fresh_icp_count),
            "retained_icp_count": self.retained_icp_count,
            "threshold_points": round(float(self.threshold_points), 6),
            "selection_policy": "centered_conditional_hash_rotated_tails:v1",
        }
        return {**doc, "policy_hash": sha256_json(doc)}


def build_conditional_category_assignment(
    *,
    rolling_window_hash: str,
    benchmark_items: Sequence[Mapping[str, Any]],
    per_icp_summaries: Sequence[Mapping[str, Any]],
    policy: ConditionalValidationPolicy | Mapping[str, Any],
    baseline_serving_model_version_hash: str,
) -> dict[str, Any]:
    """Assign the exact middle to conditional and rotate tail ownership.

    Baseline score chooses only which rows are weak, centered, or strong.
    Within each tail, a score-independent hash rotation chooses which rows are
    public; the remainder is private.  This preserves the existing 7/3 and 3/7
    weak/strong defaults without exposing the absolute tail extremes.
    """

    resolved = _coerce_policy(policy)
    if not resolved.enabled:
        raise ValueError("conditional category assignment requires enforce mode")
    if len(benchmark_items) != len(per_icp_summaries):
        raise ValueError("benchmark_items and per_icp_summaries must have the same length")
    if len(benchmark_items) != resolved.total_icps:
        raise ValueError(
            "conditional category assignment expected "
            f"{resolved.total_icps} ICPs, found {len(benchmark_items)}"
        )
    window_hash = str(rolling_window_hash or "").strip()
    if not window_hash.startswith("sha256:") or len(window_hash) != 71:
        raise ValueError("rolling window hash must be sha256")
    serving_hash = str(baseline_serving_model_version_hash or "").strip()
    if not serving_hash.startswith("sha256:") or len(serving_hash) != 71:
        raise ValueError("baseline serving-model version hash must be sha256")

    rows: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    seen_hashes: set[str] = set()
    seen_signatures: set[str] = set()
    for item_rank, (item, summary) in enumerate(
        zip(benchmark_items, per_icp_summaries),
        start=1,
    ):
        item_ref = str(item.get("icp_ref") or "").strip()
        summary_ref = str(summary.get("icp_ref") or item_ref).strip()
        if item_ref and summary_ref and item_ref != summary_ref:
            raise ValueError(
                f"benchmark item/ref mismatch at rank {item_rank}: {item_ref} != {summary_ref}"
            )
        icp_ref = item_ref or summary_ref
        icp_hash = str(item.get("icp_hash") or summary.get("icp_hash") or "").strip()
        intent_signature = _normalized_intent_signature(item)
        if not icp_ref or not icp_hash or not intent_signature:
            raise ValueError(f"benchmark item at rank {item_rank} lacks immutable identity")
        if icp_ref in seen_refs:
            raise ValueError(f"duplicate ICP ref in conditional bank: {icp_ref}")
        if icp_hash in seen_hashes:
            raise ValueError(f"duplicate ICP hash in conditional bank: {icp_hash}")
        if intent_signature in seen_signatures:
            raise ValueError(
                "duplicate normalized intent signature in conditional bank: "
                f"{intent_signature}"
            )
        seen_refs.add(icp_ref)
        seen_hashes.add(icp_hash)
        seen_signatures.add(intent_signature)
        try:
            score = float(summary.get("score") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"benchmark score is invalid for {icp_ref}") from exc
        if not math.isfinite(score) or score < 0.0 or score > 100.0:
            raise ValueError(f"benchmark score is outside 0-100 for {icp_ref}")
        rows.append(
            {
                "item_rank": item_rank,
                "icp_ref": icp_ref,
                "icp_hash": icp_hash,
                "intent_signal_signature": intent_signature,
                "set_id": _safe_int(item.get("set_id")),
                "day_index": _safe_int(item.get("day_index")),
                "day_rank": _safe_int(item.get("day_rank"), default=item_rank),
                "score": round(score, 6),
                "cohort": str(item.get("cohort") or ""),
            }
        )

    ranked = sorted(rows, key=lambda row: (float(row["score"]), _score_tiebreaker(rolling_window_hash, row)))
    low_rows = ranked[: resolved.low_tail_count]
    conditional_rows = ranked[
        resolved.low_tail_count : resolved.low_tail_count + resolved.conditional_total_icps
    ]
    high_rows = ranked[resolved.low_tail_count + resolved.conditional_total_icps :]
    if len(low_rows) != resolved.low_tail_count:
        raise ValueError("conditional assignment could not form the configured lower tail")
    if len(conditional_rows) != resolved.conditional_total_icps:
        raise ValueError("conditional assignment could not form the configured centered slice")
    if len(high_rows) != resolved.high_tail_count:
        raise ValueError("conditional assignment could not form the configured upper tail")

    public_low_refs = {
        row["icp_ref"]
        for row in sorted(
            low_rows,
            key=lambda row: _rotation_key(rolling_window_hash, "weak", row),
        )[: resolved.public_weak_total]
    }
    public_high_refs = {
        row["icp_ref"]
        for row in sorted(
            high_rows,
            key=lambda row: _rotation_key(rolling_window_hash, "strong", row),
        )[: resolved.public_strong_total]
    }
    conditional_refs = {row["icp_ref"] for row in conditional_rows}
    weak_refs = {row["icp_ref"] for row in low_rows}
    strong_refs = {row["icp_ref"] for row in high_rows}

    assigned: list[dict[str, Any]] = []
    for row in rows:
        ref = row["icp_ref"]
        if ref in conditional_refs:
            category = "conditional"
            strength = "center"
        elif ref in public_low_refs or ref in public_high_refs:
            category = "public"
            strength = "weak" if ref in weak_refs else "strong"
        else:
            category = "private"
            strength = "weak" if ref in weak_refs else "strong"
        if ref not in weak_refs and ref not in strong_refs and ref not in conditional_refs:
            raise ValueError(f"unassigned ICP ref: {ref}")
        assigned.append({**row, "category": category, "strength_label": strength})

    category_counts = {
        name: sum(1 for row in assigned if row["category"] == name)
        for name in ("public", "private", "conditional")
    }
    expected_counts = {
        "public": resolved.public_total_icps,
        "private": resolved.private_total_icps,
        "conditional": resolved.conditional_total_icps,
    }
    if category_counts != expected_counts:
        raise ValueError(
            f"conditional category counts diverged: {category_counts} != {expected_counts}"
        )
    category_scores = {
        name: round(
            _average([row["score"] for row in assigned if row["category"] == name]),
            6,
        )
        for name in ("public", "private", "conditional")
    }
    policy_doc = resolved.to_dict()
    assignment_doc = {
        "schema_version": CATEGORY_ASSIGNMENT_VERSION,
        "rolling_window_hash": window_hash,
        "baseline_serving_model_version_hash": serving_hash,
        "policy_hash": str(policy_doc["policy_hash"]),
        "selection_policy": str(policy_doc["selection_policy"]),
        "category_counts": category_counts,
        "category_scores": category_scores,
        "aggregate_score": round(_average([row["score"] for row in assigned]), 6),
        "items": assigned,
    }
    return {
        **assignment_doc,
        "assignment_hash": sha256_json(assignment_doc),
        "policy": policy_doc,
    }


def category_assignment_as_visibility_split(
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a V1.1 assignment into the existing baseline split envelope."""

    items = assignment.get("items") if isinstance(assignment.get("items"), list) else []
    public_items = [item for item in items if item.get("category") == "public"]
    private_items = [item for item in items if item.get("category") == "private"]
    conditional_items = [item for item in items if item.get("category") == "conditional"]
    return {
        "schema_version": "1.1",
        "split_policy": str(assignment.get("selection_policy") or ""),
        "rolling_window_hash": str(assignment.get("rolling_window_hash") or ""),
        "assignment_hash": str(assignment.get("assignment_hash") or ""),
        "policy_hash": str(assignment.get("policy_hash") or ""),
        "public_count": len(public_items),
        "private_count": len(private_items),
        "conditional_count": len(conditional_items),
        "public_strength_counts": _strength_counts(public_items),
        "private_strength_counts": _strength_counts(private_items),
        "items": [
            {
                "icp_ref": str(item.get("icp_ref") or ""),
                "icp_hash": str(item.get("icp_hash") or ""),
                "set_id": _safe_int(item.get("set_id")),
                "day_index": _safe_int(item.get("day_index")),
                "day_rank": _safe_int(item.get("day_rank")),
                "score": round(float(item.get("score") or 0.0), 6),
                "visibility": str(item.get("category") or ""),
                "strength_label": str(item.get("strength_label") or ""),
            }
            for item in items
        ],
    }


def _coerce_policy(
    value: ConditionalValidationPolicy | Mapping[str, Any],
) -> ConditionalValidationPolicy:
    if isinstance(value, ConditionalValidationPolicy):
        return value
    return ConditionalValidationPolicy(
        mode=str(value.get("mode") or CONDITIONAL_VALIDATION_MODE_OFF),
        public_total_icps=_safe_int(value.get("public_total_icps"), default=10),
        public_weak_total=_safe_int(value.get("public_weak_total"), default=7),
        private_total_icps=_safe_int(value.get("private_total_icps"), default=10),
        private_weak_total=_safe_int(value.get("private_weak_total"), default=3),
        conditional_total_icps=_safe_int(value.get("conditional_total_icps"), default=20),
        fresh_icp_count=_safe_int(value.get("fresh_icp_count"), default=20),
        threshold_points=float(value.get("threshold_points", 1.0)),
        policy_version=str(value.get("schema_version") or CONDITIONAL_VALIDATION_POLICY_VERSION),
    )


def normalized_icp_intent_signature(icp: Mapping[str, Any]) -> str:
    """Hash the normalized market and intent context used for bank uniqueness."""

    signals = icp.get("intent_signals") or icp.get("intent_signal") or []
    if isinstance(signals, str):
        signals = [signals]
    normalized_signals = sorted(
        {
            " ".join(str(signal).strip().lower().split())
            for signal in signals
            if str(signal).strip()
        }
    )
    normalized = {
        "industry": " ".join(str(icp.get("industry") or "").strip().lower().split()),
        "sub_industry": " ".join(
            str(icp.get("sub_industry") or "").strip().lower().split()
        ),
        "product_service": " ".join(
            str(icp.get("product_service") or "").strip().lower().split()
        ),
        "intent_signals": normalized_signals,
    }
    return sha256_json({"normalized_icp_intent": normalized})


def _normalized_intent_signature(item: Mapping[str, Any]) -> str:
    explicit = str(item.get("intent_signal_signature") or "").strip().lower()
    if explicit:
        return " ".join(explicit.split())
    icp = item.get("icp") if isinstance(item.get("icp"), Mapping) else {}
    return normalized_icp_intent_signature(icp)


def _score_tiebreaker(rolling_window_hash: str, row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "purpose": "conditional_score_tiebreaker",
            "rolling_window_hash": str(rolling_window_hash),
            "icp_ref": str(row.get("icp_ref") or ""),
            "icp_hash": str(row.get("icp_hash") or ""),
        }
    )


def _rotation_key(
    rolling_window_hash: str,
    strength_label: str,
    row: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "purpose": "conditional_tail_category_rotation",
            "rolling_window_hash": str(rolling_window_hash),
            "strength_label": str(strength_label),
            "icp_ref": str(row.get("icp_ref") or ""),
            "icp_hash": str(row.get("icp_hash") or ""),
        }
    )


def _average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _strength_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    labels = sorted({str(item.get("strength_label") or "") for item in items})
    return {
        label: sum(1 for item in items if str(item.get("strength_label") or "") == label)
        for label in labels
        if label
    }


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
