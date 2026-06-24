"""Sanitized Research Lab benchmark reports for miner-facing guidance."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from gateway.research_lab.bundles import contains_secret_material, sha256_json


ZERO_LEAD_SCORE_THRESHOLD = 1e-9


def build_public_benchmark_report(
    *,
    benchmark_date: str,
    rolling_window_hash: str,
    aggregate_score: float,
    per_icp_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a sanitized report from private daily benchmark summaries.

    The output intentionally excludes exact ICP text, exact intent signals,
    URLs, company names, private model outputs, private image refs, and private
    repo details.
    """

    bucket_rows: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    score_bands: Counter[str] = Counter()
    zero_lead_count = 0
    low_intent_count = 0
    low_icp_fit_count = 0

    for index, summary in enumerate(per_icp_summaries, start=1):
        diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), Mapping) else {}
        failures = Counter(str(key) for key in diagnostics.get("failure_categories", []) if key)
        error_counts.update(failures)
        score = float(summary.get("score") or 0.0)
        band = _score_band(score)
        score_bands[band] += 1
        company_count = int(summary.get("company_count") or 0)
        if company_count <= 0 or score <= ZERO_LEAD_SCORE_THRESHOLD:
            zero_lead_count += 1
        if float(diagnostics.get("avg_intent_signal_final") or 0.0) < 15.0:
            low_intent_count += 1
        if float(diagnostics.get("avg_icp_fit") or 0.0) < 15.0:
            low_icp_fit_count += 1
        bucket_rows.append(
            {
                "item_rank": index,
                "industry_bucket": _bucket_text(summary.get("industry")),
                "sub_industry_bucket": _bucket_text(summary.get("sub_industry")),
                "geography_bucket": _bucket_text(summary.get("geography_bucket") or summary.get("country")),
                "company_size_bucket": _bucket_text(summary.get("company_size_bucket")),
                "intent_category_bucket": _bucket_text(summary.get("intent_category_bucket")),
                "score_band": band,
                "company_count_band": _count_band(company_count),
                "failure_categories": sorted(failures),
            }
        )

    report = {
        "schema_version": "1.0",
        "report_type": "research_lab_public_daily_benchmark",
        "benchmark_date": str(benchmark_date),
        "rolling_window_hash": str(rolling_window_hash),
        "aggregate_score_band": _score_band(float(aggregate_score)),
        "aggregate_score": round(float(aggregate_score), 6),
        "item_count": len(bucket_rows),
        "zero_lead_icp_count": zero_lead_count,
        "low_intent_fit_icp_count": low_intent_count,
        "low_icp_fit_count": low_icp_fit_count,
        "score_band_counts": dict(sorted(score_bands.items())),
        "failure_category_counts": dict(sorted(error_counts.items())),
        "icp_buckets": bucket_rows,
        "redaction_policy": {
            "exact_icp_text": "withheld",
            "exact_signal_text": "withheld",
            "urls": "withheld",
            "company_names": "withheld",
            "private_model_outputs": "withheld",
            "private_artifacts": "withheld",
        },
    }
    if contains_secret_material(report):
        raise ValueError("public benchmark report contains forbidden private or secret material")
    report["report_public_hash"] = sha256_json(report)
    return report


def sanitize_benchmark_item_summary(
    *,
    item: Mapping[str, Any],
    score: float,
    company_count: int,
    score_breakdowns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = []
    icp_fit_values: list[float] = []
    intent_values: list[float] = []
    for breakdown in score_breakdowns:
        reason = str(breakdown.get("failure_reason") or "")
        if reason:
            failures.append(_failure_category(reason))
        icp_fit_values.append(float(breakdown.get("icp_fit") or 0.0))
        intent_values.append(float(breakdown.get("intent_signal_final") or 0.0))
    icp = item.get("icp") if isinstance(item.get("icp"), Mapping) else {}
    return {
        "icp_ref": str(item.get("icp_ref") or ""),
        "icp_hash": str(item.get("icp_hash") or ""),
        "score": round(float(score), 6),
        "company_count": int(company_count),
        "industry": _bucket_text(icp.get("industry")),
        "sub_industry": _bucket_text(icp.get("sub_industry")),
        "country": _bucket_text(icp.get("country") or icp.get("target_geography")),
        "company_size_bucket": _company_size_bucket(icp),
        "intent_category_bucket": _intent_category_bucket(icp),
        "diagnostics": {
            "failure_categories": sorted(failures),
            "avg_icp_fit": _average(icp_fit_values),
            "avg_intent_signal_final": _average(intent_values),
        },
    }


def _failure_category(reason: str) -> str:
    lowered = reason.lower()
    if "fabricated" in lowered or "generic" in lowered or "hallucinat" in lowered or "hardcoded" in lowered:
        return "hallucinated_or_generic_intent"
    if "future" in lowered or "date" in lowered or "stale" in lowered or "freshness" in lowered:
        return "invalid_or_stale_intent_date"
    if "company verification" in lowered or "company" in lowered and "not found" in lowered:
        return "company_verification_failed"
    if "url" in lowered or "fetch" in lowered or "scrape" in lowered or "content" in lowered:
        return "source_fetch_or_url_failed"
    if "llm" in lowered:
        return "llm_scoring_error"
    if "parse" in lowered or "json" in lowered:
        return "parser_error"
    if "pre-check" in lowered or "mismatch" in lowered:
        return "icp_or_geo_mismatch"
    return "other_scoring_failure"


def _score_band(score: float) -> str:
    if score >= 80:
        return "80_plus"
    if score >= 60:
        return "60_79"
    if score >= 40:
        return "40_59"
    if score > 0:
        return "1_39"
    return "zero"


def _count_band(count: int) -> str:
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 3:
        return "two_to_three"
    return "four_plus"


def _bucket_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return "unspecified"
    return text[:80]


def _company_size_bucket(icp: Mapping[str, Any]) -> str:
    raw = str(icp.get("employee_count") or icp.get("company_size") or "").lower()
    if any(marker in raw for marker in ("1-10", "small", "smb", "startup")):
        return "small"
    if any(marker in raw for marker in ("11-50", "51-200", "mid")):
        return "mid_market"
    if any(marker in raw for marker in ("201", "500", "1000", "enterprise")):
        return "enterprise"
    return "unspecified"


def _intent_category_bucket(icp: Mapping[str, Any]) -> str:
    signals = icp.get("intent_signals") or []
    joined = " ".join(str(item).lower() for item in signals if str(item).strip())
    if any(term in joined for term in ("hiring", "job", "recruit")):
        return "hiring"
    if any(term in joined for term in ("funding", "raised", "series", "investment")):
        return "funding"
    if any(term in joined for term in ("launch", "expansion", "opening", "new market")):
        return "expansion"
    if any(term in joined for term in ("compliance", "regulation", "audit", "security")):
        return "compliance_or_security"
    if any(term in joined for term in ("partnership", "integration", "migration", "implementation")):
        return "technology_change"
    return "general_buying_intent"


def _average(values: Sequence[float]) -> float:
    return round(float(sum(values) / len(values)), 6) if values else 0.0
