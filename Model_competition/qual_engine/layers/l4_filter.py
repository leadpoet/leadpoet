"""L4 — Hard filters (no LLM). Pure deterministic gates."""

from __future__ import annotations

import logging
from typing import Optional

from qual_engine.models import ICPPrompt, ResolvedCompany
from qual_engine.infra.trace import Tracer
from qual_engine.validators.text_match import (
    country_match,
    validate_size_match,
)

logger = logging.getLogger(__name__)


# Stage compatibility is LLM-driven (qual_engine.utils.ai_classifiers.is_stage_compatible)
# — no hardcoded stage taxonomy.


def _industry_direct_overlap(
    icp_industry: str, icp_sub_industry: str, candidate_tags: list[str]
) -> Optional[bool]:
    """Free check. Returns True if obvious overlap, False if no overlap, None if can't tell."""
    if not candidate_tags:
        return None  # unknown — defer
    blob = " ".join(candidate_tags).lower()

    icp_phrases = []
    for t in (icp_industry, icp_sub_industry):
        if t:
            icp_phrases.append(t.lower())
            icp_phrases.extend(t.lower().split("/"))
            icp_phrases.extend(t.lower().split(","))
    icp_phrases = [p.strip().strip(",.;:") for p in icp_phrases if p.strip()]

    for phrase in icp_phrases:
        if phrase and phrase in blob:
            return True
        for token in phrase.split():
            if len(token) >= 5 and token in blob:
                return True
    return False  # no direct overlap, but maybe semantically compatible — caller decides


_INDUSTRY_COMPAT_PROMPT = """Is the CANDIDATE company's industry compatible with the BUYER's target industry for B2B sales targeting?

BUYER'S full ask: "{buyer_prompt}"
BUYER sells: "{product_service}"
BUYER target industry: {icp_industry}
BUYER target sub-industry: {icp_sub_industry}
CANDIDATE industry tags: {candidate_tags}

Rules:
- "Compatible" = the candidate operates in the same broad business domain that the buyer described, so the buyer would consider it worth pursuing.
- The BUYER'S full ask is the strongest signal for the type of company sought — weigh it heavily even when the literal industry-tag string doesn't match the structured target industry field.
- "Not compatible" = the candidate operates in a fundamentally different domain from the buyer's described target.
- When uncertain, lean toward compatible — downstream layers do stricter verification.

Output JSON only:
{{"compatible": true|false, "reason": "<≤12 words>"}}"""


async def _industry_compat_llm(
    openrouter,
    cost,
    icp_industry: str,
    icp_sub_industry: str,
    candidate_tags: list[str],
    *,
    buyer_prompt: str = "",
    product_service: str = "",
) -> bool:
    """Cached LLM compatibility check. Fails open (returns True) on any error."""
    from qual_engine.config import CONFIG
    try:
        prompt = _INDUSTRY_COMPAT_PROMPT.format(
            buyer_prompt=buyer_prompt or "<unspecified>",
            product_service=product_service or "<unspecified>",
            icp_industry=icp_industry or "<unspecified>",
            icp_sub_industry=icp_sub_industry or "<unspecified>",
            candidate_tags="; ".join(candidate_tags) or "<unknown>",
        )
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="L4_industry_compat")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L4")
        parsed = r.get("parsed")
        if isinstance(parsed, dict) and "compatible" in parsed:
            return bool(parsed["compatible"])
    except Exception as e:
        logger.warning("industry compat LLM failed: %s", e)
    return True  # fail-open


async def filter_candidates(
    candidates: list[ResolvedCompany],
    icp: ICPPrompt,
    tracer: Tracer,
    openrouter=None,
    cost=None,
) -> list[ResolvedCompany]:
    """Hard filters. Industry & stage checks are LLM-assisted (cached)."""
    import asyncio as _asyncio
    from qual_engine.utils.ai_classifiers import extract_country

    # Always LLM-normalize via extract_country (cached forever) so that two-letter
    # codes that don't match ISO-3166 (e.g. "UK" → "GB") get resolved correctly.
    icp_country_iso = ""
    if openrouter and icp.country:
        icp_country_iso = await extract_country(icp.country, openrouter, cost)
    if openrouter and not icp_country_iso and icp.geography:
        icp_country_iso = await extract_country(icp.geography, openrouter, cost)
    # Fallback: trust raw if already a plausible ISO-2 (uppercase letters only).
    if not icp_country_iso and icp.country and len(icp.country) == 2 and icp.country.isalpha():
        icp_country_iso = icp.country.upper()

    survivors = []
    drops = {"country": 0, "industry": 0, "stage": 0, "size": 0, "shape": 0}
    drop_details = []

    # First pass: deterministic checks. Collect industry-uncertain candidates for LLM batch.
    industry_uncertain: list[ResolvedCompany] = []

    for c in candidates:
        # Shape check: drop placeholders / personal pages (e.g. "Open To Work Remote")
        if c.canonical_name.lower() in {"open to work remote", "open to work"}:
            drops["shape"] += 1
            continue
        # Size = 1 employee is suspicious (likely a personal LinkedIn page)
        if c.employee_count_band and c.employee_count_band.startswith("1 employee"):
            drops["shape"] += 1
            continue

        # Country: ISO-2 equality (both LLM-extracted upstream). Unknown = pass.
        if icp_country_iso and c.country:
            if not country_match(icp_country_iso, c.country):
                drops["country"] += 1
                drop_details.append(f"{c.canonical_name}: country (icp={icp_country_iso}, cand={c.country})")
                continue

        # Industry — direct check first, defer ambiguous to LLM
        if icp.industry:
            direct = _industry_direct_overlap(icp.industry, icp.sub_industry, c.industry_tags or [])
            if direct is True:
                pass  # clear overlap, accept
            elif direct is False and openrouter is not None:
                # No direct overlap and LLM available → mark for semantic check
                industry_uncertain.append(c)
                continue
            elif direct is False:
                drops["industry"] += 1
                drop_details.append(f"{c.canonical_name}: industry no overlap (no LLM available)")
                continue
            # direct is None → unknown tags, defer to L7 (fail-open)

        # Stage — LLM-driven compat (cached). Skip if either unknown.
        if openrouter and icp.company_stage and c.funding_stage:
            from qual_engine.utils.ai_classifiers import is_stage_compatible
            stage_ok = await is_stage_compatible(icp.company_stage, c.funding_stage, openrouter, cost)
            if not stage_ok:
                drops["stage"] += 1
                drop_details.append(f"{c.canonical_name}: stage incompatible (icp={icp.company_stage}, cand={c.funding_stage})")
                continue

        # Size: handled holistically by L4.5 LLM ICP-fit gate (which sees both
        # the buyer's free-text ask and the structured target). Removing the
        # deterministic 4x-ratio guard avoids dropping candidates when the buyer's
        # natural-language description should allow looser interpretation.

        survivors.append(c)

    # Second pass: LLM semantic compatibility for industry-uncertain candidates
    if industry_uncertain and openrouter is not None:
        compat_results = await _asyncio.gather(*[
            _industry_compat_llm(
                openrouter, cost,
                icp.industry, icp.sub_industry, c.industry_tags or [],
                buyer_prompt=icp.prompt, product_service=icp.product_service,
            )
            for c in industry_uncertain
        ], return_exceptions=True)
        for c, compat in zip(industry_uncertain, compat_results):
            if isinstance(compat, Exception) or compat is True:
                # fail-open on error or compat=True → keep
                survivors.append(c)
            else:
                drops["industry"] += 1
                drop_details.append(f"{c.canonical_name}: industry incompatible (LLM)")

    tracer.emit("L4", "filtered",
                input_count=len(candidates),
                survived=len(survivors),
                drops=drops,
                drop_details=drop_details[:20])
    return survivors
