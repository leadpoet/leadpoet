"""L3 — LinkedIn-anchored Entity Resolver.

For each unique candidate domain, anchors to LinkedIn via ScrapingDog (fallback to
Google site:linkedin.com slug-exists check). If no anchor → drop.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import CandidateCompany, ResolvedCompany
from qual_engine.providers.scrapingdog import ScrapingDogClient
from qual_engine.providers.exa import ExaClient
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer
from qual_engine.utils.ai_classifiers import extract_country, verify_website_anchor
from qual_engine.validators.text_match import (
    extract_linkedin_slug,
    normalize_company_name,
)

logger = logging.getLogger(__name__)


# Country extraction is LLM-driven (see qual_engine.utils.ai_classifiers.extract_country)
# — no hardcoded geography lists.


def _derive_linkedin_slug(sources: list[CandidateCompany], name: str) -> Optional[str]:
    """Extract a LinkedIn slug from discovery URLs ONLY (never guess from name)."""
    for c in sources:
        s = extract_linkedin_slug(c.discovery_url)
        if s:
            return s
    return None


async def _find_linkedin_slug_via_google(
    scrapingdog: ScrapingDogClient,
    cost: CostTracker,
    company_name: str,
    primary_domain: str = "",
) -> Optional[str]:
    """Find a verified LinkedIn slug via Google search. Tries multiple query
    formats so we have several shots at surfacing the LinkedIn page:

      1. site:linkedin.com/company "<name>"   — exact-name site-restricted
      2. "<domain>" linkedin.com/company       — domain-anchored, less ambiguous
      3. "<domain>" linkedin                    — looser domain match
      4. "<name>" linkedin.com/company          — name without site: restriction

    Domain-based queries (2, 3) handle the case where the company's name is
    common/short but the domain is globally unique — the domain is the
    disambiguator the LinkedIn page will be indexed under.
    """
    queries: list[str] = []
    # Domain-based queries first — most reliable when domain is unique.
    # Format with literal `+` between terms (verified working on ScrapingDog Google).
    if primary_domain:
        queries.append(f'"{primary_domain}"+linkedin')
        queries.append(f'"{primary_domain}"+linkedin.com/company')
    # Name-based as backstop
    if company_name:
        queries.append(f'site:linkedin.com/company "{company_name}"')
        queries.append(f'"{company_name}"+linkedin.com/company')

    for q in queries:
        try:
            r = await scrapingdog.google(q, results=5)
            cost.add("scrapingdog", r.get("cost_usd", 0), layer="L3")
            for hit in (r.get("results") or []):
                url = (hit.get("link") or hit.get("url") or "")
                slug = extract_linkedin_slug(url)
                if slug:
                    return slug
        except Exception as e:
            logger.warning("LinkedIn-via-Google %r failed: %s", q, e)
    return None


def _split_industry_tags(raw: str) -> list[str]:
    """SD sometimes returns industry as 'Financial Services, Software & SaaS' — split."""
    if not raw or not isinstance(raw, str):
        return []
    parts: list[str] = []
    for chunk in re.split(r"[,&;/]| and ", raw):
        chunk = chunk.strip()
        if chunk and len(chunk) >= 2:
            parts.append(chunk)
    seen = set()
    deduped = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(p)
    return deduped


_FUNDING_STAGE_PROMPT = """What is the current funding/capital stage of this company?

Company: {name}
Website: {website}
Industry: {industry}
HQ: {hq}

Return a concise label describing the company's current capital/funding state — use whatever terminology accurately describes their reality (a Series letter, a public-market description, bootstrapping status, ownership type, etc.). Be brief and specific. If you cannot determine from public information, return null.

Output JSON only: {{"funding_stage": "<short string or null>"}}"""


async def _extract_funding_stage(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    name: str,
    website: str,
    industry: str,
    hq: str,
) -> Optional[str]:
    if not name:
        return None
    try:
        from qual_engine.config import CONFIG
        prompt = _FUNDING_STAGE_PROMPT.format(
            name=name, website=website or "<unknown>",
            industry=industry or "<unknown>", hq=hq or "<unknown>",
        )
        r = await openrouter.json_call(CONFIG.SONAR_VERIFY_MODEL, prompt, label="L3_funding_stage")
        cost.add("openrouter", r.get("cost_usd", 0), layer="L3")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            v = parsed.get("funding_stage")
            if isinstance(v, str) and v.strip():
                stripped = v.strip()
                # Treat sentinel null-equivalents as unknown without hardcoding a taxonomy
                if stripped.lower() not in ("null", "none", "n/a", "unknown"):
                    return stripped
    except Exception as e:
        logger.warning("funding stage extract failed for %r: %s", name, e)
    return None


async def _try_website_anchor(
    exa: ExaClient,
    openrouter: OpenRouterClient,
    cost: CostTracker,
    domain: str,
    sources: list[CandidateCompany],
    icp_prompt: str,
    icp_industry: str,
    icp_country: str,
) -> Optional[ResolvedCompany]:
    """Fallback path: when LinkedIn can't anchor, try the company's own website.

    Fetch via Exa /contents, ask LLM to verify it's a single operating company
    that matches the ICP, extract facts. Only succeed when LLM says yes to both.
    """
    if not domain or "linkedin.com" in domain:
        return None
    url = f"https://{domain}"
    fetched = await exa.contents(url)
    cost.add("exa", fetched.get("cost_usd", 0), layer="L3")
    content = fetched.get("text", "")
    if not content or len(content) < 200:
        return None

    verdict = await verify_website_anchor(
        domain=domain,
        content=content,
        buyer_prompt=icp_prompt,
        industry=icp_industry,
        country=icp_country,
        openrouter=openrouter,
        cost=cost,
    )
    if not verdict or not verdict.get("is_company") or not verdict.get("matches_icp"):
        return None

    facts = verdict.get("facts") or {}
    canonical = (facts.get("canonical_name") or "").strip()
    if not canonical:
        # Fallback to first non-empty candidate name
        canonical = next((c.name for c in sources if c.name), domain.split(".")[0].title())

    industry_raw = (facts.get("industry") or "").strip()
    industry_tags = _split_industry_tags(industry_raw) if industry_raw else []

    aliases = {canonical}
    aliases.update({c.name for c in sources if c.name})
    aliases.discard("")

    return ResolvedCompany(
        canonical_name=canonical,
        primary_domain=domain,
        aliases=list(aliases),
        country=(facts.get("country") or "").strip().upper()[:2] or None,
        industry_tags=industry_tags,
        employee_count_band=(facts.get("employee_count_band") or None),
        funding_stage=None,
        linkedin_url=None,
        linkedin_slug=None,
        headquarters=None,
        description=(facts.get("description") or "")[:1000],
        discovery_sources=list({c.source for c in sources}),
        anchor_source="website",
    )


async def _resolve_one(
    scrapingdog: ScrapingDogClient,
    exa: ExaClient,
    openrouter: OpenRouterClient,
    cost: CostTracker,
    domain: str,
    sources: list[CandidateCompany],
    icp_prompt: str = "",
    icp_industry: str = "",
    icp_country: str = "",
    icp_geography: str = "",
    icp_company_stage: str = "",
    tracer: Optional[Tracer] = None,
) -> Optional[ResolvedCompany]:
    def _drop(reason: str, name: str = ""):
        if tracer is not None:
            tracer.emit("L3", "drop_reason", domain=domain[:60],
                        name=(name or (sources[0].name if sources else ""))[:50],
                        reason=reason)
        return None
    # The candidates may be under different domains (e.g. linkedin.com) — pick best name
    primary_name = next((c.name for c in sources if c.name and "linkedin.com" not in c.domain), None)
    if not primary_name:
        primary_name = sources[0].name if sources else ""

    # Step 1: derive a slug ONLY from a discovery URL (never guess from name)
    slug = _derive_linkedin_slug(sources, primary_name)

    # Step 1b: if no slug from discovery, try Google with multiple query formats
    # (name-based + domain-based). Domain-based queries disambiguate short/common
    # names that collide with many LinkedIn pages.
    if not slug and (primary_name or (domain and "linkedin.com" not in domain)):
        slug = await _find_linkedin_slug_via_google(
            scrapingdog, cost, primary_name,
            primary_domain=domain if "linkedin.com" not in domain else "",
        )

    if not slug:
        # Still no LinkedIn URL → fall back to website-anchor verification
        wa = await _try_website_anchor(
            exa, openrouter, cost, domain, sources,
            icp_prompt=icp_prompt, icp_industry=icp_industry, icp_country=icp_country,
        )
        if wa is None:
            return _drop("no_linkedin_slug_and_website_anchor_failed", primary_name)
        return wa

    # Step 2: fetch SD LinkedIn data directly
    r = await scrapingdog.linkedin_company(slug)
    cost.add("scrapingdog", r.get("cost_usd", 0), layer="L3")
    sd_data = r.get("data") or {}
    if not sd_data:
        # LinkedIn fetch failed → try website-anchor as last resort
        wa = await _try_website_anchor(
            exa, openrouter, cost, domain, sources,
            icp_prompt=icp_prompt, icp_industry=icp_industry, icp_country=icp_country,
        )
        if wa is None:
            return _drop(f"sd_linkedin_empty_slug={slug}_and_website_anchor_failed", primary_name)
        return wa

    from qual_engine.validators.text_match import (
        validate_name_match,
        normalize_domain as _norm,
        domains_likely_same_company,
    )

    # Compute domain check first — when it passes strongly, we can trust SD's name
    # even if the candidate's name was a domain-stem ("Precisionmedicinegrp" vs SD's
    # "Precision Medicine Group" — the slug came from the candidate's domain so SD's
    # entity IS the right one).
    sd_name = sd_data.get("company_name", "")
    cand_domain = domain if "linkedin.com" not in domain else ""
    sd_website = sd_data.get("website", "") or ""
    domain_ok = False
    if cand_domain and sd_website:
        domain_ok = domains_likely_same_company(cand_domain, sd_website)

    # Sanity 1: name match (token-boundary). When it fails BUT domain check
    # strongly confirms same company, trust SD's name.
    name_ok, name_reason = validate_name_match(primary_name, sd_name)
    if not name_ok and not domain_ok:
        return _drop(
            f"name_and_domain_mismatch: ours={primary_name!r}/{cand_domain} "
            f"sd={sd_name!r}/{_norm(sd_website)}",
            primary_name,
        )

    # Sanity 2: when both name AND a real candidate domain exist, but domains
    # disagree even after stem-fuzz, that's a slug collision → drop.
    if cand_domain and sd_website and not domain_ok and not name_ok:
        return _drop(f"domain_mismatch: ours={cand_domain} sd={_norm(sd_website)}", primary_name)
    # If candidate was LinkedIn-only (no other domain), the slug came directly from
    # the indexed LinkedIn URL — that itself is the sanity signal.

    canonical_name = sd_data.get("company_name", primary_name)
    hq = sd_data.get("headquarters", "")

    # Gate country LLM extraction: only when ICP has a geography/country constraint
    icp_has_geo = bool(icp_country or icp_geography)
    country = await extract_country(hq, openrouter, cost) if (hq and icp_has_geo) else ""

    raw_industry = sd_data.get("industry", "") or ""
    industry_tags = _split_industry_tags(raw_industry)

    size = sd_data.get("company_size", "")
    website = sd_data.get("website", "") or ""

    raw_desc = sd_data.get("about", "") or sd_data.get("description", "")
    if not isinstance(raw_desc, str):
        raw_desc = str(raw_desc) if raw_desc else ""

    # Gate funding_stage Sonar call: only when ICP has a stage constraint
    funding_stage = None
    if icp_company_stage:
        funding_stage = await _extract_funding_stage(
            openrouter, cost, canonical_name, website, raw_industry, hq,
        )

    aliases = {canonical_name}
    aliases.update({c.name for c in sources if c.name})
    aliases.discard("")

    return ResolvedCompany(
        canonical_name=canonical_name,
        primary_domain=domain if "linkedin.com" not in domain else (
            re.sub(r"^https?://(www\.)?", "", website).rstrip("/") if website else domain
        ),
        aliases=list(aliases),
        country=country,
        industry_tags=industry_tags,
        employee_count_band=size,
        funding_stage=funding_stage,
        linkedin_url=f"https://www.linkedin.com/company/{slug}/",
        linkedin_slug=slug,
        headquarters=hq,
        description=raw_desc[:1000],
        discovery_sources=list({c.source for c in sources}),
        anchor_source="linkedin",
    )


async def resolve(
    scrapingdog: ScrapingDogClient,
    exa: ExaClient,
    openrouter: OpenRouterClient,
    cost: CostTracker,
    tracer: Tracer,
    candidates: list[CandidateCompany],
    *,
    max_concurrent: int = 15,
    icp_prompt: str = "",
    icp_industry: str = "",
    icp_country: str = "",
    icp_geography: str = "",
    icp_company_stage: str = "",
) -> list[ResolvedCompany]:
    """Group by domain, enrich each via ScrapingDog LinkedIn. Drop non-resolvable.

    For candidates without a LinkedIn anchor, falls back to a website-anchor
    verification using Exa /contents + an LLM "is this an operating company
    that matches the ICP?" check.
    """
    by_domain: dict[str, list[CandidateCompany]] = {}
    for c in candidates:
        by_domain.setdefault(c.domain, []).append(c)

    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(domain, sources):
        async with sem:
            try:
                return await _resolve_one(
                    scrapingdog, exa, openrouter, cost, domain, sources,
                    icp_prompt=icp_prompt,
                    icp_industry=icp_industry,
                    icp_country=icp_country,
                    icp_geography=icp_geography,
                    icp_company_stage=icp_company_stage,
                    tracer=tracer,
                )
            except Exception as e:
                logger.warning("resolve %s failed: %s", domain, e)
                if tracer is not None:
                    tracer.emit("L3", "drop_reason", domain=domain[:60],
                                name=(sources[0].name if sources else "")[:50],
                                reason=f"exception: {type(e).__name__}: {str(e)[:80]}")
                return None

    results = await asyncio.gather(
        *[bounded(d, srcs) for d, srcs in by_domain.items()]
    )
    resolved = [r for r in results if r is not None]

    tracer.emit("L3", "resolved",
                input_domains=len(by_domain),
                resolved=len(resolved),
                dropped=len(by_domain) - len(resolved),
                by_anchor={
                    "linkedin": sum(1 for r in resolved if r.anchor_source == "linkedin"),
                    "website":  sum(1 for r in resolved if r.anchor_source == "website"),
                })
    return resolved
