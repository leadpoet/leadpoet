"""Open score aggregation arithmetic for Research Lab verifier runs."""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple


MAX_COMPANY_ICP_FIT_SCORE = 40
MAX_COMPANY_INTENT_SIGNAL_SCORE = 60
MAX_COMPANY_TOTAL_SCORE = MAX_COMPANY_ICP_FIT_SCORE + MAX_COMPANY_INTENT_SIGNAL_SCORE

SOURCE_TYPE_MULTIPLIERS = {
    "linkedin": 1.0,
    "job_board": 1.0,
    "github": 1.0,
    "news": 0.9,
    "company_website": 0.85,
    "social_media": 0.8,
    "review_site": 0.75,
    "wikipedia": 0.6,
    "other": 0.3,
}

SOURCES_DATE_NOT_REQUIRED = frozenset(
    {"github", "company_website", "wikipedia", "review_site"}
)

DEFAULT_INTENT_SIGNAL_DECAY_50_PCT_MONTHS = 2
DEFAULT_INTENT_SIGNAL_DECAY_25_PCT_MONTHS = 12
NO_DATE_DECAY_MULTIPLIER = 0.5


def calculate_age_months(signal_date: date, *, today: Optional[date] = None) -> float:
    clock = today or date.today()
    return (clock - signal_date).days / 30.0


def calculate_time_decay_multiplier(
    age_months: float,
    *,
    decay_50_pct_months: float = DEFAULT_INTENT_SIGNAL_DECAY_50_PCT_MONTHS,
    decay_25_pct_months: float = DEFAULT_INTENT_SIGNAL_DECAY_25_PCT_MONTHS,
) -> float:
    """Return the time-decay multiplier for published verifier tier params."""
    if decay_50_pct_months > decay_25_pct_months:
        raise ValueError("decay_50_pct_months must be <= decay_25_pct_months")
    if age_months <= decay_50_pct_months:
        return 1.0
    if age_months <= decay_25_pct_months:
        return 0.5
    return 0.25


def apply_signal_time_decay(
    raw_score: float,
    signal_date: Optional[str],
    date_status: str,
    source: str,
    content_found_date: Optional[str] = None,
    *,
    today: Optional[date] = None,
    decay_50_pct_months: float = DEFAULT_INTENT_SIGNAL_DECAY_50_PCT_MONTHS,
    decay_25_pct_months: float = DEFAULT_INTENT_SIGNAL_DECAY_25_PCT_MONTHS,
) -> Tuple[float, float]:
    """Apply deterministic date/source time-decay arithmetic.

    ``date_status`` here is the scoring status vocabulary used by production
    aggregation: ``date_omitted`` means the model submitted no date but a
    verifier/scrape found one; ``no_date`` means no usable date is available.
    It is separate from L0's ``check_date_precision`` vocabulary
    (``verified``, ``approximate``, ``year_only``, ``no_match``), which checks
    whether a submitted date appears in the evidence snapshot.
    """
    source_lower = (source or "").lower().strip()

    if date_status == "date_omitted" and content_found_date:
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return float(raw_score), 1.0
        parsed_found = _parse_iso_date(content_found_date)
        if parsed_found is not None:
            age_months = calculate_age_months(parsed_found, today=today)
            decay = calculate_time_decay_multiplier(
                age_months,
                decay_50_pct_months=decay_50_pct_months,
                decay_25_pct_months=decay_25_pct_months,
            )
            return float(raw_score) * decay, decay
        return float(raw_score), 1.0

    if date_status == "no_date":
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return float(raw_score), 1.0
        return float(raw_score) * NO_DATE_DECAY_MULTIPLIER, NO_DATE_DECAY_MULTIPLIER

    parsed = _parse_iso_date(signal_date)
    if parsed is None:
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return float(raw_score), 1.0
        return 0.0, 0.0

    age_months = calculate_age_months(parsed, today=today)
    decay = calculate_time_decay_multiplier(
        age_months,
        decay_50_pct_months=decay_50_pct_months,
        decay_25_pct_months=decay_25_pct_months,
    )
    return float(raw_score) * decay, decay


def source_adjusted_intent_score(raw_score: float, source: str) -> float:
    multiplier = SOURCE_TYPE_MULTIPLIERS.get((source or "").lower().strip(), 0.5)
    return max(0.0, min(float(raw_score), float(MAX_COMPANY_INTENT_SIGNAL_SCORE))) * multiplier


def company_final_score(
    icp_fit: float,
    intent_signal_final: float,
    *,
    run_cost_usd: float = 0.0,
    cost_penalty_threshold: float = 0.0,
    variability_penalty_points: float = 0.0,
    is_reference_model: bool = False,
) -> Dict[str, float]:
    """Compute the final company score and cost-variability penalty."""
    cost_penalty = 0.0
    if not is_reference_model and run_cost_usd > cost_penalty_threshold:
        cost_penalty = float(variability_penalty_points)
    raw = float(icp_fit) + float(intent_signal_final)
    final = min(float(MAX_COMPANY_TOTAL_SCORE), max(0.0, raw - cost_penalty))
    return {
        "icp_fit": float(icp_fit),
        "intent_signal_final": float(intent_signal_final),
        "cost_penalty": cost_penalty,
        "final_score": final,
    }


def per_icp_normalized_score(
    lead_scores: Iterable[float],
    *,
    max_leads: int = 5,
) -> float:
    """Normalize one ICP's company scores by the fixed maximum lead count."""
    if max_leads <= 0:
        raise ValueError("max_leads must be positive")
    total = sum(max(0.0, min(float(score), float(MAX_COMPANY_TOTAL_SCORE))) for score in lead_scores)
    return total / float(max_leads)


def aggregate_set_score(per_icp_scores: Iterable[float]) -> float:
    values = [float(score) for score in per_icp_scores]
    if not values:
        return 0.0
    return sum(values) / len(values)


def u16_weights_from_scores(
    scores_by_uid: Dict[int, float],
    *,
    total_weight: int = 65535,
) -> Dict[int, int]:
    """Deterministically quantize non-negative scores into a u16 vector.

    Ties in fractional remainders are broken by higher score, then lower uid.
    This pins down the rounding order for validators.
    """
    if total_weight <= 0 or total_weight > 65535:
        raise ValueError("total_weight must be in the u16 range")
    if not scores_by_uid:
        return {}

    cleaned = {int(uid): max(0.0, float(score)) for uid, score in scores_by_uid.items()}
    score_sum = sum(cleaned.values())
    if score_sum <= 0:
        return {uid: 0 for uid in sorted(cleaned)}

    exact = {
        uid: (score / score_sum) * total_weight
        for uid, score in cleaned.items()
    }
    weights = {uid: int(value) for uid, value in exact.items()}
    remainder = total_weight - sum(weights.values())
    if remainder > 0:
        order: List[int] = sorted(
            cleaned,
            key=lambda uid: (-(exact[uid] - weights[uid]), -cleaned[uid], uid),
        )
        for uid in order[:remainder]:
            weights[uid] += 1
    return {uid: weights[uid] for uid in sorted(weights)}


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None
