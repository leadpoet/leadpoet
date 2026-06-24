"""Sanitized Research Lab benchmark reports for miner-facing guidance."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from gateway.research_lab.bundles import contains_secret_material, sha256_json


ZERO_LEAD_SCORE_THRESHOLD = 1e-9
DEFAULT_PUBLIC_ICPS_PER_DAY = 3
DEFAULT_PUBLIC_WEAK_PER_DAY = 2
SPLIT_POLICY = "global_score_rank_public_weak_majority:v1"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
PUBLIC_ICP_FORBIDDEN_KEY_MARKERS = (
    "api_key",
    "raw_secret",
    "raw_openrouter",
    "service_role",
    "private_repo",
    "judge_prompt",
    "hidden_icp",
    "icp_plaintext",
    "image_digest",
    "private_model_manifest",
    "candidate_patch_manifest",
    "proxy_url",
    "credential",
    "secret",
)


def build_public_benchmark_report(
    *,
    benchmark_date: str,
    rolling_window_hash: str,
    aggregate_score: float,
    per_icp_summaries: Sequence[Mapping[str, Any]],
    benchmark_items: Sequence[Mapping[str, Any]] = (),
    public_icps_per_day: int = DEFAULT_PUBLIC_ICPS_PER_DAY,
    public_weak_per_day: int = DEFAULT_PUBLIC_WEAK_PER_DAY,
) -> dict[str, Any]:
    """Build a sanitized report from private daily benchmark summaries.

    The output intentionally reveals full miner-facing ICP details for the
    configured public split only. The private holdout ICP bodies, URLs, company
    names, private model outputs, private image refs, and private repo details
    stay withheld.
    """

    split = (
        _build_scored_visibility_rows(
            rolling_window_hash=rolling_window_hash,
            benchmark_items=benchmark_items,
            per_icp_summaries=per_icp_summaries,
            public_icps_per_day=public_icps_per_day,
            public_weak_per_day=public_weak_per_day,
        )
        if benchmark_items
        else []
    )
    split_by_ref = {str(row["icp_ref"]): row for row in split}
    bucket_rows: list[dict[str, Any]] = []
    public_icps: list[dict[str, Any]] = []
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
        icp_ref = str(summary.get("icp_ref") or "")
        split_row = split_by_ref.get(icp_ref)
        bucket_rows.append(
            {
                "item_rank": index,
                "icp_ref": icp_ref,
                "visibility": str(split_row.get("visibility")) if split_row else "summary_only",
                "strength_label": str(split_row.get("strength_label")) if split_row else "unknown",
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
        if split_row and split_row.get("visibility") == "public":
            public_icps.append(_public_icp_entry(split_row))

    split_summary = _visibility_split_summary(split) if split else {
        "split_policy": "summary_only",
        "rolling_window_hash": str(rolling_window_hash),
        "public_count": 0,
        "private_count": 0,
    }

    report = {
        "schema_version": "1.1",
        "report_type": "research_lab_public_daily_benchmark",
        "benchmark_date": str(benchmark_date),
        "rolling_window_hash": str(rolling_window_hash),
        "aggregate_score_band": _score_band(float(aggregate_score)),
        "aggregate_score": round(float(aggregate_score), 6),
        "item_count": len(bucket_rows),
        "public_icp_count": len(public_icps),
        "private_holdout_icp_count": int(split_summary.get("private_count") or 0),
        "zero_lead_icp_count": zero_lead_count,
        "low_intent_fit_icp_count": low_intent_count,
        "low_icp_fit_count": low_icp_fit_count,
        "score_band_counts": dict(sorted(score_bands.items())),
        "failure_category_counts": dict(sorted(error_counts.items())),
        "visibility_split": split_summary,
        "public_icps": public_icps,
        "icp_buckets": bucket_rows,
        "redaction_policy": {
            "exact_icp_text": "public_for_public_split_only",
            "exact_signal_text": "public_for_public_split_only",
            "private_holdout_icp_text": "withheld",
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


def build_benchmark_visibility_split(
    *,
    rolling_window_hash: str,
    benchmark_items: Sequence[Mapping[str, Any]],
    per_icp_summaries: Sequence[Mapping[str, Any]],
    public_icps_per_day: int = DEFAULT_PUBLIC_ICPS_PER_DAY,
    public_weak_per_day: int = DEFAULT_PUBLIC_WEAK_PER_DAY,
) -> dict[str, Any]:
    rows = _build_scored_visibility_rows(
        rolling_window_hash=rolling_window_hash,
        benchmark_items=benchmark_items,
        per_icp_summaries=per_icp_summaries,
        public_icps_per_day=public_icps_per_day,
        public_weak_per_day=public_weak_per_day,
    )
    summary = _visibility_split_summary(rows)
    summary["items"] = [
        {
            "icp_ref": str(row["icp_ref"]),
            "icp_hash": str(row["icp_hash"]),
            "set_id": int(row["set_id"]),
            "day_index": int(row["day_index"]),
            "day_rank": int(row["day_rank"]),
            "score": round(float(row["score"]), 6),
            "visibility": str(row["visibility"]),
            "strength_label": str(row["strength_label"]),
        }
        for row in rows
    ]
    return summary


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


def _build_scored_visibility_rows(
    *,
    rolling_window_hash: str,
    benchmark_items: Sequence[Mapping[str, Any]],
    per_icp_summaries: Sequence[Mapping[str, Any]],
    public_icps_per_day: int,
    public_weak_per_day: int,
) -> list[dict[str, Any]]:
    if len(benchmark_items) != len(per_icp_summaries):
        raise ValueError("benchmark_items and per_icp_summaries must have the same length")
    if public_icps_per_day <= 0:
        raise ValueError("public_icps_per_day must be positive")
    if public_weak_per_day < 0 or public_weak_per_day > public_icps_per_day:
        raise ValueError("public_weak_per_day must be between 0 and public_icps_per_day")

    rows: list[dict[str, Any]] = []
    for index, (item, summary) in enumerate(zip(benchmark_items, per_icp_summaries), start=1):
        item_ref = str(item.get("icp_ref") or "")
        summary_ref = str(summary.get("icp_ref") or item_ref)
        if item_ref and summary_ref and item_ref != summary_ref:
            raise ValueError(f"benchmark item/ref mismatch at rank {index}: {item_ref} != {summary_ref}")
        icp_ref = item_ref or summary_ref
        if not icp_ref:
            raise ValueError(f"benchmark item at rank {index} is missing icp_ref")
        rows.append(
            {
                "item_rank": index,
                "item": dict(item),
                "summary": dict(summary),
                "icp_ref": icp_ref,
                "icp_hash": str(item.get("icp_hash") or summary.get("icp_hash") or ""),
                "set_id": _set_id_from_item(item),
                "day_index": int(item.get("day_index") or 0),
                "day_rank": int(item.get("day_rank") or index),
                "score": float(summary.get("score") or 0.0),
                "rolling_window_hash": str(rolling_window_hash),
            }
        )

    selected_day_count = len(
        {
            (int(row["day_index"]), int(row["set_id"]))
            for row in rows
        }
    )
    public_count = public_icps_per_day * selected_day_count
    public_weak_count = public_weak_per_day * selected_day_count
    public_strong_count = public_count - public_weak_count
    if public_count <= 0 or public_count >= len(rows):
        raise ValueError("public split must expose at least one ICP and leave private holdout ICPs")

    ranked = sorted(rows, key=lambda row: (float(row["score"]), _split_tiebreaker(row)))
    weak_count = len(ranked) // 2
    weak_pool = ranked[:weak_count]
    strong_pool = ranked[weak_count:]
    if public_weak_count > len(weak_pool):
        raise ValueError("public split requested more weak ICPs than the global weak pool contains")
    if public_strong_count > len(strong_pool):
        raise ValueError("public split requested more strong ICPs than the global strong pool contains")

    public_weak_rows = weak_pool[:public_weak_count]
    public_strong_rows = list(reversed(strong_pool))[:public_strong_count]
    public_refs = {
        str(row["icp_ref"])
        for row in [*public_weak_rows, *public_strong_rows]
    }
    weak_refs = {str(row["icp_ref"]) for row in weak_pool}
    strong_refs = {str(row["icp_ref"]) for row in strong_pool}

    for row in rows:
        ref = str(row["icp_ref"])
        row["visibility"] = "public" if ref in public_refs else "private"
        row["strength_label"] = "weak" if ref in weak_refs else "strong"
    return sorted(rows, key=lambda row: int(row["item_rank"]))


def _visibility_split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    public_rows = [row for row in rows if row.get("visibility") == "public"]
    private_rows = [row for row in rows if row.get("visibility") == "private"]
    public_strength = Counter(str(row.get("strength_label")) for row in public_rows)
    private_strength = Counter(str(row.get("strength_label")) for row in private_rows)
    return {
        "schema_version": "1.0",
        "split_policy": SPLIT_POLICY,
        "rolling_window_hash": str(rows[0].get("rolling_window_hash") if rows else ""),
        "public_count": len(public_rows),
        "private_count": len(private_rows),
        "public_strength_counts": dict(sorted(public_strength.items())),
        "private_strength_counts": dict(sorted(private_strength.items())),
    }


def _public_icp_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    item = row.get("item") if isinstance(row.get("item"), Mapping) else {}
    summary = row.get("summary") if isinstance(row.get("summary"), Mapping) else {}
    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), Mapping) else {}
    return {
        "item_rank": int(row.get("item_rank") or 0),
        "icp_ref": str(row.get("icp_ref") or ""),
        "icp_hash": str(row.get("icp_hash") or ""),
        "set_id": int(row.get("set_id") or 0),
        "day_index": int(row.get("day_index") or 0),
        "day_rank": int(row.get("day_rank") or 0),
        "score": round(float(row.get("score") or 0.0), 6),
        "company_count": int(summary.get("company_count") or 0),
        "strength_label": str(row.get("strength_label") or "unknown"),
        "icp": _public_icp_doc(item.get("icp") if isinstance(item.get("icp"), Mapping) else {}),
        "diagnostics": {
            "failure_categories": list(diagnostics.get("failure_categories") or []),
            "avg_icp_fit": float(diagnostics.get("avg_icp_fit") or 0.0),
            "avg_intent_signal_final": float(diagnostics.get("avg_intent_signal_final") or 0.0),
        },
    }


def _public_icp_doc(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_icp_doc(item)
            for key, item in value.items()
            if _public_icp_key_allowed(str(key))
        }
    if isinstance(value, list):
        return [_public_icp_doc(item) for item in value]
    if isinstance(value, str):
        return URL_RE.sub("[withheld_url]", value)
    return value


def _public_icp_key_allowed(key: str) -> bool:
    lowered = key.lower()
    return not any(marker in lowered for marker in PUBLIC_ICP_FORBIDDEN_KEY_MARKERS)


def _set_id_from_item(item: Mapping[str, Any]) -> int:
    try:
        return int(item.get("set_id") or 0)
    except (TypeError, ValueError):
        pass
    parts = str(item.get("icp_ref") or "").split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def _split_tiebreaker(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "rolling_window_hash": row.get("rolling_window_hash"),
            "icp_ref": row.get("icp_ref"),
            "icp_hash": row.get("icp_hash"),
            "score": round(float(row.get("score") or 0.0), 6),
        }
    )


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
