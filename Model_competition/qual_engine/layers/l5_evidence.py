"""L5 — Evidence Gathering. Intent-agnostic: LLM plans the search strategy.

For every (company, claim) pair:
  1. Ask the LLM to generate an evidence-gathering plan (queries + time window)
  2. Run Exa neural searches in parallel
  3. (Optional) For hiring-shaped claims, also try ScrapingDog LinkedIn jobs
  4. Filter to Tier-1 URLs (structural)
  5. Filter out aggregators (LLM-cached)
  6. Classify each surviving URL's source_type via LLM
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import EvidenceURL, ICPPrompt, ParsedICP, ResolvedCompany
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.providers.exa import ExaClient
from qual_engine.providers.scrapingdog import ScrapingDogClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer
from qual_engine.validators.url_tier import classify_url_tier
from qual_engine.utils.ai_classifiers import (
    plan_evidence_search,
    classify_source_type,
    is_aggregator,
)

logger = logging.getLogger(__name__)


# URLs production's intent-signal gate rejects at Layer 1 (structural URL check).
# Filtering these BEFORE L6 grounding saves the Exa/contents + Sonnet cost on
# URLs that will be auto-rejected at validation time.
# Source: qualification/scoring/intent_signal_gate.py:_INVALID_URL_PATTERNS
import re as _re
_PROD_INVALID_URL_RE = _re.compile(
    r"/alternatives(?:\b|/|\?|$)"
    r"|/competitors(?:\b|/|\?|$)"
    r"|indeed\.com/hire/job-description/"
    r"|github\.com/[^/]+/[^/]+/labels(?:/|$)"
    r"|github\.com/[^/]+/[^/]+/discussions/\d+(?:/|$)",
    _re.IGNORECASE,
)


def _is_prod_rejected_url(url: str) -> bool:
    """Return True if production's validator will hard-reject this URL pattern."""
    if not url:
        return False
    return bool(_PROD_INVALID_URL_RE.search(url))


async def _try_linkedin_jobs(
    scrapingdog: ScrapingDogClient,
    cost: CostTracker,
    company: ResolvedCompany,
) -> list[EvidenceURL]:
    """Pull company-filtered LinkedIn jobs as a structured-evidence path.
    Used opportunistically — only when the claim involves recruiting/role activity.
    Synthesizes raw_content from SD metadata (LinkedIn job pages aren't scrapable)."""
    out: list[EvidenceURL] = []
    r = await scrapingdog.linkedin_jobs_by_company(
        company.canonical_name, role_keywords="", max_pages=2
    )
    cost.add("scrapingdog", r.get("cost_usd", 0), layer="L5")
    for job in r.get("jobs", []):
        url = job.get("job_link") or ""
        if not url:
            continue
        title_raw = job.get("job_position") or ""
        posted = job.get("job_posting_date", "")
        claimed_date = None
        try:
            claimed_date = datetime.fromisoformat(posted[:10]).date()
        except Exception:
            pass
        synthesized = (
            f"Job posting on LinkedIn for {company.canonical_name}.\n"
            f"Role: {title_raw}\n"
            f"Company: {job.get('company_name','')}\n"
            f"Location: {job.get('job_location','')}\n"
            f"Posted: {posted}\n"
            f"URL: {url}\n\n"
            f"Source: ScrapingDog LinkedIn Jobs API confirmed this posting was filed by "
            f"{job.get('company_name','')}. The role title is exactly: {title_raw}."
        )
        out.append(
            EvidenceURL(
                url=url,
                source_type="job_board",
                claimed_date=claimed_date,
                discovered_via="sd_linkedin_jobs",
                title=title_raw[:200],
                raw_content=synthesized,
            )
        )
    return out


async def _classify_and_filter(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    raw_results: list[dict],
    primary_domain: str,
) -> list[EvidenceURL]:
    """Apply structural Tier-1 filter + LLM aggregator filter + LLM source_type.
    Returns deduped EvidenceURL list."""
    seen = set()
    pre = []
    for r in raw_results:
        url = r.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        if classify_url_tier(url) > 1:
            continue
        pre.append(r)

    if not pre:
        return []

    # Parallel: aggregator + source_type classifications (each LLM-cached)
    titles = [r.get("title", "") for r in pre]
    urls = [r.get("url", "") for r in pre]
    agg_tasks = [is_aggregator(u, t, openrouter, cost) for u, t in zip(urls, titles)]
    stype_tasks = [classify_source_type(u, t, openrouter, cost) for u, t in zip(urls, titles)]
    aggregates, stypes = await asyncio.gather(
        asyncio.gather(*agg_tasks, return_exceptions=True),
        asyncio.gather(*stype_tasks, return_exceptions=True),
    )

    out = []
    for r, is_agg, stype in zip(pre, aggregates, stypes):
        if isinstance(is_agg, Exception):
            is_agg = False
        if isinstance(stype, Exception):
            stype = "other"
        # Allow LinkedIn job-view pages even though LinkedIn is aggregator-ish
        if is_agg and "linkedin.com/jobs/view/" not in r.get("url", ""):
            continue
        out.append(
            EvidenceURL(
                url=r.get("url", ""),
                source_type=stype,
                discovered_via="exa_neural",
                title=(r.get("title", "") or "")[:200],
            )
        )
    return out


async def gather_evidence(
    openrouter: OpenRouterClient,
    exa: ExaClient,
    scrapingdog: ScrapingDogClient,
    cost: CostTracker,
    tracer: Tracer,
    company: ResolvedCompany,
    icp_signal_text: str,
    parsed: ParsedICP,
    *,
    icp=None,                              # ICPPrompt — optional for backward compat
    signal_idx: Optional[int] = None,      # for per-signal intent_class routing
) -> list[EvidenceURL]:
    """Intent-agnostic evidence gathering.

    The LLM plans the search using buyer_prompt + product_service context; downstream
    layers filter and classify. Evidence URLs are mandatory.
    """
    buyer_prompt = (icp.prompt if icp else "") or ""
    product_service = (icp.product_service if icp else "") or ""

    # Compute the intent class for THIS signal (multi-intent ICPs use per-signal
    # classes from parsed.intent_classes; fall back to the dominant intent_class).
    signal_intent = parsed.intent_class
    if signal_idx is not None and 0 <= signal_idx < len(parsed.intent_classes):
        signal_intent = parsed.intent_classes[signal_idx]

    # Step 1: LLM evidence plan — pass primary_domain + intent_class so the plan
    # uses intent-specific verbs and source-type bias.
    plan = await plan_evidence_search(
        company.canonical_name, icp_signal_text, openrouter, cost,
        buyer_prompt=buyer_prompt, product_service=product_service,
        primary_domain=company.primary_domain or "",
        intent_class=signal_intent or "other",
    )
    queries = plan.get("queries", []) or [f"{company.canonical_name} {icp_signal_text}"]
    # Widen window: take max of plan + parsed; widen further if buyer mentions a year
    plan_window = plan.get("time_window_days") or 180
    parsed_window = parsed.time_window_days or 180
    time_window = max(plan_window, parsed_window)
    if buyer_prompt:
        import re as _re
        year_match = _re.search(r"\b(202[3-9])\b", buyer_prompt)
        if year_match:
            time_window = max(time_window, 730)  # ~2 years
    preferred = set(plan.get("preferred_source_types") or [])
    # Force LinkedIn jobs path on for hiring intent regardless of plan output
    # (signal_intent was computed above before the plan call)
    if signal_intent == "hiring":
        preferred.add("job_board")

    # Step 2: Parallel Exa neural searches
    search_tasks = [exa.search_neural(q, num_results=8, days_back=time_window) for q in queries[:3]]
    results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_results = []
    for results in results_per_query:
        if isinstance(results, Exception):
            continue
        cost.add("exa", results.get("cost_usd", 0), layer="L5")
        all_results.extend(results.get("results", []))

    # Step 3: classify + filter
    evidence = await _classify_and_filter(openrouter, cost, all_results, company.primary_domain)

    # Step 4: opportunistic LinkedIn jobs (intent-agnostic — try only if plan mentioned job_board)
    if "job_board" in preferred or "linkedin" in preferred:
        lj = await _try_linkedin_jobs(scrapingdog, cost, company)
        evidence.extend(lj)

    # MANDATORY: drop any evidence with empty url (defensive)
    evidence = [e for e in evidence if e.url and e.url.startswith(("http://", "https://"))]

    tracer.emit(
        "L5", "evidence_gathered",
        company=company.canonical_name,
        n_urls=len(evidence),
        plan_queries=queries[:3],
        preferred_sources=list(preferred),
    )
    return evidence
