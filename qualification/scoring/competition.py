"""Shared company scorer for the public baseline and miner bundles."""

from __future__ import annotations

from importlib import import_module
import logging
import os
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from leadpoet_verifier.aggregation import per_icp_normalized_score
from qualification.competition_models import CompetitionCompany
from qualification.employee_buckets import (
    normalize_employee_count_bucket,
    normalize_observed_employee_count_bucket,
)


SCORING_ADAPTER_VERSION = "qualification-company-scorer:v1"
DEFAULT_COMPANY_GOAL = 5
MAX_COMPANY_GOAL = 5
FP_PENALTY_POINTS = 10.0
logger = logging.getLogger(__name__)

_PENALIZABLE_FAILURE_MARKERS = (
    "exclusion list",
    "required_attribute",
    "missing employee_count",
    "missing company_stage",
    "country mismatch",
    "missing country",
    "duplicate company",
    "data quality issue",
    "missing industry",
    "company verification failed",
    "intent fabrication detected",
)
_NEVER_PENALIZE_MARKERS = ("error", "timeout", "provider", "429")


class CompetitionScorerInputError(ValueError):
    """A company or ICP does not satisfy the public competition boundary."""


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("intent_signal")
            or value.get("signal")
            or value.get("text")
            or ""
        ).strip()
    return str(value or "").strip()


def _category(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    text = str(
        value.get("intent_category")
        or value.get("category")
        or value.get("evidence_type")
        or ""
    ).strip().upper()
    return text or None


def employee_count_buckets_for_icp(icp: Mapping[str, Any]) -> list[str]:
    """Return the exact employee buckets declared by one ICP."""

    raw = icp.get("employee_count")
    values = (
        list(raw)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))
        else str(raw or "").replace(";", "|").split("|")
    )
    buckets: list[str] = []
    for value in values:
        bucket = normalize_employee_count_bucket(value, default=None)
        if bucket and bucket not in buckets:
            buckets.append(bucket)
    if not buckets:
        raise CompetitionScorerInputError("ICP employee_count has no valid bucket")
    return buckets


def _company_goal(icp: Mapping[str, Any]) -> int:
    try:
        value = int(icp.get("max_companies", DEFAULT_COMPANY_GOAL))
    except (TypeError, ValueError):
        value = DEFAULT_COMPANY_GOAL
    return max(1, min(MAX_COMPANY_GOAL, value))


def _normalized_icp(icp: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(icp, Mapping):
        raise CompetitionScorerInputError("ICP must be an object")
    industry = str(icp.get("industry") or "").strip()
    icp_id = str(icp.get("icp_id") or "").strip()
    if not industry or not icp_id:
        raise CompetitionScorerInputError("ICP is missing icp_id or industry")

    signals: list[str] = []
    evidence_types: list[str | None] = []
    primary_category = str(icp.get("intent_category") or "").strip().upper() or None
    bonus_categories = {
        _text(item): _category(item)
        for item in (icp.get("bonus_intents") or [])
        if isinstance(item, Mapping) and _text(item)
    }
    raw_signals = icp.get("intent_signals") or [icp.get("intent_signal")]
    if isinstance(raw_signals, (str, Mapping)):
        raw_signals = [raw_signals]
    for index, item in enumerate(raw_signals or []):
        signal = _text(item)
        if not signal or signal in signals:
            continue
        signals.append(signal)
        evidence_types.append(
            _category(item)
            or bonus_categories.get(signal)
            or (primary_category if index == 0 else None)
        )
    for item in icp.get("bonus_intents") or []:
        signal = _text(item)
        if signal and signal not in signals:
            signals.append(signal)
            evidence_types.append(_category(item))
    if not signals:
        raise CompetitionScorerInputError("ICP has no intent signal")

    buckets = employee_count_buckets_for_icp(icp)
    stage = icp.get("company_stage") or "Any"
    if isinstance(stage, Sequence) and not isinstance(stage, (str, bytes, bytearray)):
        stage = next((str(value).strip() for value in stage if str(value).strip()), "Any")
    country = str(icp.get("country") or "").strip()
    geography = str(icp.get("geography") or country or "United States").strip()
    required_attribute = str(icp.get("required_attribute") or "").strip()
    product_service = str(
        icp.get("product_service") or required_attribute or industry
    ).strip()
    prompt = str(icp.get("prompt") or "").strip()
    if not prompt:
        prompt = f"Find {industry} companies in {geography} with {signals[0]}"
    excluded = icp.get("excluded_companies") or []
    if not isinstance(excluded, list):
        excluded = []
    return {
        "icp_id": icp_id,
        "prompt": prompt,
        "industry": industry,
        "sub_industry": str(icp.get("sub_industry") or industry).strip(),
        "target_roles": [],
        "target_seniority": "",
        "employee_count": "|".join(buckets),
        "company_stage": str(stage).strip() or "Any",
        "geography": geography,
        "country": country or geography,
        "product_service": product_service,
        "required_attribute": required_attribute,
        "excluded_companies": [str(value) for value in excluded],
        "intent_signals": signals,
        "intent_signal_evidence_types": evidence_types,
        "intent_max_age_days": max(1, int(icp.get("intent_max_age_days") or 365)),
    }


def _normalized_company(company: Mapping[str, Any]) -> dict[str, Any]:
    try:
        row = CompetitionCompany.model_validate(company).model_dump(mode="json")
    except Exception as exc:
        raise CompetitionScorerInputError(
            "company does not satisfy the competition output schema"
        ) from exc
    signals = [
        {
            "source": _evidence_source(
                signal["url"], company_website=row["company_website"]
            ),
            "description": signal["description"],
            "url": signal["url"],
            "date": signal["date"],
            "snippet": signal["snippet"],
            "matched_icp_signal": signal["matched_icp_signal"],
        }
        for signal in row["intent_signals"]
    ]
    return {
        "company_name": row["company_name"],
        "company_website": row["company_website"],
        "company_linkedin": row["company_linkedin"],
        "industry": row["industry"],
        "sub_industry": "",
        "employee_count": row["employee_count"],
        "company_stage": row["company_stage"],
        "country": row["country"],
        "state": row["state"],
        "description": row["fit_summary"][:500],
        "intent_signals": signals,
        "required_attribute": row.get("required_attribute"),
    }


def _evidence_source(url: str, *, company_website: str) -> str:
    """Infer the scorer's source class from the submitted public URL."""

    hostname = (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")
    company_hostname = (
        (urlsplit(str(company_website)).hostname or "").lower().removeprefix("www.")
    )
    path = (urlsplit(str(url)).path or "").lower()
    if hostname == "linkedin.com" or hostname.endswith(".linkedin.com"):
        return "linkedin"
    if hostname == "github.com" or hostname.endswith(".github.com"):
        return "github"
    if any(marker in path for marker in ("/jobs", "/job/", "/careers")):
        return "job_board"
    if company_hostname and (
        hostname == company_hostname or hostname.endswith("." + company_hostname)
    ):
        return "company_website"
    return "news"


def _ensure_provider_environment() -> None:
    key = os.getenv("QUALIFICATION_OPENROUTER_API_KEY") or os.getenv(
        "OPENROUTER_API_KEY"
    )
    if not key:
        return
    os.environ.setdefault("QUALIFICATION_OPENROUTER_API_KEY", key)
    for module_name in ("qualification.scoring.verification_helpers",):
        module = import_module(module_name)
        if not getattr(module, "OPENROUTER_API_KEY", ""):
            setattr(module, "OPENROUTER_API_KEY", key)


class CompetitionCompanyScorer:
    """Use the production company judge for baseline and miner outputs."""

    async def __call__(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        rows = await self.score_with_breakdowns(companies, icp, is_reference_model)
        return [float(row.get("final_score") or 0.0) for row in rows]

    async def score_with_breakdowns(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        models = import_module("gateway.qualification.models")
        scorer_module = import_module("qualification.scoring.lead_scorer")
        _ensure_provider_environment()
        icp_data = _normalized_icp(icp)
        allowed_buckets = employee_count_buckets_for_icp(icp)
        icp_model = getattr(models, "ICPPrompt")(**icp_data)
        company_type = getattr(models, "CompanyOutput")
        score_company = scorer_module.score_company_competition_intent

        seen_companies: set[str] = set()
        breakdowns: list[dict[str, Any]] = []
        for company in list(companies)[: _company_goal(icp)]:
            observed = (company or {}).get("employee_count")
            bucket = normalize_employee_count_bucket(
                observed, default=None
            ) or normalize_observed_employee_count_bucket(observed, default=None)
            if not bucket or bucket not in allowed_buckets:
                continue
            company_model = company_type(**_normalized_company(company))
            result = await score_company(
                company=company_model,
                icp=icp_model,
                run_cost_usd=0.0,
                run_time_seconds=0.0,
                seen_companies=seen_companies,
                is_reference_model=bool(is_reference_model),
            )
            breakdowns.append(
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else dict(result)
            )
        return breakdowns


def scorer_breakdown_has_retryable_infrastructure_failure(
    breakdown: Mapping[str, Any],
) -> bool:
    if not isinstance(breakdown, Mapping):
        return False
    receipts = breakdown.get("verifier_gate_receipts")
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
        for receipt in receipts:
            if (
                isinstance(receipt, Mapping)
                and str(receipt.get("decision") or "") == "unavailable"
                and str(receipt.get("failure_class") or "")
                != "model_contract_incompatible"
            ):
                return True
    details = breakdown.get("intent_signals_detail")
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
        for detail in details:
            verdict = detail.get("judge_verdict") if isinstance(detail, Mapping) else None
            if isinstance(verdict, Mapping) and (
                str(verdict.get("decision") or "") == "rejected_verifier_error"
                or bool(verdict.get("error_class"))
                or str(verdict.get("pipeline_decision") or "") == "unavailable"
            ):
                return True
    reason = str(breakdown.get("failure_reason") or "").strip().lower()
    return bool(reason) and any(
        marker in reason
        for marker in (
            "intent verification unavailable:",
            "llm scoring error:",
            "company verification error:",
            "company verification failed: website unreachable:",
            "company verification failed: website fetch error:",
            "company verification unavailable:",
            "company fit pre-check unavailable:",
            "company web re-verification unavailable:",
            "providerclientv2error",
            "runner external request must use https",
            "provider error",
            "provider timeout",
            "http 429",
            "no_openrouter_key",
        )
    )


def _structured_fit_mismatch(breakdown: Mapping[str, Any]) -> bool:
    receipts = breakdown.get("verifier_gate_receipts")
    return isinstance(receipts, Sequence) and not isinstance(
        receipts, (str, bytes)
    ) and any(
        isinstance(receipt, Mapping)
        and str(receipt.get("gate") or "") == "company_fit"
        and str(receipt.get("decision") or "") == "mismatch"
        for receipt in receipts
    )


def count_penalizable_false_positives(
    breakdowns: Sequence[Mapping[str, Any]], *, icp_has_intent_signals: bool
) -> tuple[int, int]:
    gate_failures = 0
    unverified_primary = 0
    for row in breakdowns:
        if not isinstance(row, Mapping) or scorer_breakdown_has_retryable_infrastructure_failure(row):
            continue
        if _structured_fit_mismatch(row):
            gate_failures += 1
            continue
        reason = str(row.get("failure_reason") or "").strip().lower()
        if reason:
            if not any(marker in reason for marker in _NEVER_PENALIZE_MARKERS) and any(
                marker in reason for marker in _PENALIZABLE_FAILURE_MARKERS
            ):
                gate_failures += 1
            continue
        if not icp_has_intent_signals:
            continue
        details = row.get("intent_signals_detail")
        if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
            continue
        primary_verified = False
        verifier_failed = False
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            verdict = detail.get("judge_verdict")
            if isinstance(verdict, Mapping) and (
                str(verdict.get("decision") or "") == "rejected_verifier_error"
                or bool(verdict.get("error_class"))
            ):
                verifier_failed = True
            try:
                index = int(detail.get("matched_icp_signal", -1))
            except (TypeError, ValueError):
                continue
            if index == 0 and float(detail.get("after_decay") or 0.0) > 0.0:
                primary_verified = True
                break
        if details and not primary_verified and not verifier_failed:
            unverified_primary += 1
    return gate_failures, unverified_primary


def fp_penalty_total_from_breakdowns(
    breakdowns: Sequence[Mapping[str, Any]], icp: Mapping[str, Any]
) -> float:
    gate, primary = count_penalizable_false_positives(
        breakdowns,
        icp_has_intent_signals=bool(
            icp.get("intent_signals") or icp.get("intent_signal")
        ),
    )
    return float(gate + primary) * FP_PENALTY_POINTS


def competition_icp_score_from_company_scores(
    scores: Sequence[float], *, requested_count: int, fp_penalty_total: float = 0.0
) -> float:
    """Use the same capped score arithmetic as the Arena."""

    count = max(1, min(MAX_COMPANY_GOAL, int(requested_count)))
    normalized = float(
        per_icp_normalized_score(
            sorted((float(value or 0.0) for value in scores), reverse=True)[:count],
            max_leads=count,
        )
    )
    return max(0.0, normalized - max(0.0, float(fp_penalty_total)) / count)


def competition_score_from_breakdowns(
    icp: Mapping[str, Any],
    breakdowns: Sequence[Mapping[str, Any]],
    *,
    fp_penalty_points: float = FP_PENALTY_POINTS,
    fp_unverified_primary_penalty_points: float = FP_PENALTY_POINTS,
    score_floor: float = 0.0,
) -> dict[str, Any]:
    """Calculate one ICP score for both the baseline and miner bundles."""

    goal = _company_goal(icp)
    rows = [dict(row) for row in breakdowns]
    gate, primary = count_penalizable_false_positives(
        rows,
        icp_has_intent_signals=bool(
            icp.get("intent_signals") or icp.get("intent_signal")
        ),
    )
    company_scores = [float(row.get("final_score") or 0.0) for row in rows]
    normalized = float(
        per_icp_normalized_score(company_scores[:goal], max_leads=goal)
    )
    penalty = (
        gate * max(0.0, float(fp_penalty_points))
        + primary * max(0.0, float(fp_unverified_primary_penalty_points))
    ) / goal
    return {
        "per_icp_score": max(float(score_floor), normalized - penalty),
        "fp_gate_count": gate,
        "fp_unverified_primary_count": primary,
        "company_goal": goal,
        "company_scores": company_scores,
    }
