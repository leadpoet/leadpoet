"""LLM-based classifiers (cached) replacing hardcoded knowledge lists.

Every function in this module:
  - Uses gpt-4o-mini (cheap, fast)
  - Caches result by deterministic key (temperature=0 → reusable forever)
  - Fails open: returns the safe default on any error
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Is this URL an aggregator / non-company-website?
# ─────────────────────────────────────────────────────────────────────────────

_AGG_PROMPT = """Decide: does this URL point to a single operating company's OWN page, OR to content about/listing/produced-by-an-intermediary?

URL: {url}
Title snippet: {title}

Mark is_aggregator = TRUE when ANY of these structural properties hold:
- The page's purpose is to describe, rank, compare, or round up MULTIPLE companies in one piece of content.
- The page was authored by a publisher whose business model is publishing content (about other companies/topics) rather than operating a B2B product the buyer would purchase.
- The page hosts content produced by many third-party companies (press-release wires, news syndication, content aggregators).
- The domain belongs to an entity whose primary purpose is something other than being a B2B-targetable operating company (research/statistics providers, directories, events, regulators, government, professional associations, talent intermediaries, generic search results).
- The page is a generic listing/search path inside a professional network rather than a specific company's profile.
- The title frames the content as a topic, ranking, comparison, or how-to — not as a specific brand naming itself.

Mark is_aggregator = FALSE when:
- The URL is on the operating company's OWN primary domain and the page describes that company itself (home page, about, product, careers, the company's own newsroom/blog post about itself).
- The URL is that company's own profile page on a professional network (path is structured as a company-profile slug, not a search/listing).

Reason briefly with ≤10 words.

Output JSON only:
{{"is_aggregator": true|false, "reason": "<≤10 words>"}}"""


_EXTRACT_ARTICLE_COMPANY_PROMPT = """This URL is a third-party article (news, press wire, blog, listicle). Identify the SINGLE company that the article is primarily about — if there is one.

URL: {url}
Title: {title}

Decide first whether the article is about ONE specific company (e.g., "Schematic raises $6.5M", "Acme launches X") or about MULTIPLE companies (listicles, market reports, comparisons). If multiple, return null — we don't want to misattribute.

When a single company is the subject, return its commonly-used name and its likely primary domain (homepage). Use the company's OWN domain, NOT the article's host domain. If you cannot identify a likely primary domain, set domain to null.

Output JSON only:
{{"name": "<company name or null>", "domain": "<company-domain.com or null>"}}"""


async def extract_company_from_article(url: str, title: str, openrouter, cost) -> Optional[dict]:
    """Given an article URL+title, extract the company it discusses (if exactly one).

    Returns a dict {name, domain} or None.
    """
    if not (url or title):
        return None
    try:
        from qual_engine.config import CONFIG
        prompt = _EXTRACT_ARTICLE_COMPANY_PROMPT.format(url=url, title=(title or "")[:200])
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="extract_article_company")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L2")
        parsed = r.get("parsed")
        if not isinstance(parsed, dict):
            return None
        name = parsed.get("name")
        domain = parsed.get("domain")
        if not isinstance(name, str) or not name.strip() or name.strip().lower() in ("null", "none", "n/a"):
            return None
        out = {"name": name.strip()[:120]}
        if isinstance(domain, str) and domain.strip() and domain.strip().lower() not in ("null", "none", "n/a"):
            out["domain"] = domain.strip().lower()
        else:
            out["domain"] = None
        return out
    except Exception as e:
        logger.warning("extract_company_from_article failed for %s: %s", url, e)
        return None


async def is_aggregator(url: str, title: str, openrouter, cost) -> bool:
    """Returns True if URL is on an aggregator. Fails open (False) on error."""
    if not url:
        return False
    # Pre-check 1: linkedin /company/ is always non-aggregator
    if "linkedin.com/company/" in url:
        return False
    # Pre-check 2: URLs with no path (or only a short, home/about-style path) are
    # the company's OWN page. Skip the LLM — these are never aggregators.
    try:
        parsed = urlparse(url)
        path = (parsed.path or "").rstrip("/").lower()
        # Empty path, "/", or short paths like /about, /company, /contact, /home, /products.
        if path == "" or len(path) <= 12:
            return False
    except Exception:
        pass
    try:
        prompt = _AGG_PROMPT.format(url=url, title=(title or "")[:200])
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="is_aggregator")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L2")
        parsed = r.get("parsed")
        if isinstance(parsed, dict) and "is_aggregator" in parsed:
            return bool(parsed["is_aggregator"])
    except Exception as e:
        logger.warning("is_aggregator LLM failed for %s: %s", url, e)
    return False  # fail-open: assume it's a company site


# ─────────────────────────────────────────────────────────────────────────────
# Extract ISO-2 country from an HQ string
# ─────────────────────────────────────────────────────────────────────────────

_COUNTRY_PROMPT = """Extract the country (ISO-3166 alpha-2 code) from this LinkedIn-style headquarters string.

HQ string: "{hq}"

The HQ string is typically "City" or "City, State" or "City, State, Country" or "City, Country".
If the country is not explicitly named but the state/region clearly implies a country, use that country.
If unknown or ambiguous, return null.

Output JSON only:
{{"country_iso2": "<2-letter code or null>"}}"""


async def extract_country(hq: str, openrouter, cost) -> str:
    """Returns ISO-2 country code (e.g. 'US') or '' if unknown. Fails open (empty) on error."""
    if not hq:
        return ""
    try:
        prompt = _COUNTRY_PROMPT.format(hq=hq[:200])
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="extract_country")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L3")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            v = parsed.get("country_iso2")
            if isinstance(v, str) and len(v) == 2:
                return v.upper()
    except Exception as e:
        logger.warning("extract_country LLM failed for %r: %s", hq, e)
    return ""  # unknown


# ─────────────────────────────────────────────────────────────────────────────
# Does this content contain a negation of the claim?
# ─────────────────────────────────────────────────────────────────────────────

_NEGATION_PROMPT = """Does the following content actively contradict or negate the CLAIM about the company?

CONTENT (first 4000 chars):
{content}

CLAIM: "{claim}"

A negation is when the content explicitly says the claim is false or no longer true.
Examples of negations: "no longer hiring", "0 open positions", "page not found", "this role has been filled", "not currently accepting applications", "no public record of".

Mere absence of evidence is NOT a negation — only explicit denial/contradiction counts.

Output JSON only:
{{"is_negated": true|false, "evidence": "<short quote if negated, else empty>"}}"""


async def is_negation(content: str, claim: str, openrouter, cost) -> tuple[bool, str]:
    """Returns (negated, evidence_phrase). Fails open (False, '') on error."""
    if not content:
        return False, ""
    try:
        prompt = _NEGATION_PROMPT.format(content=content[:4000], claim=claim[:300])
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="is_negation")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L6")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            negated = bool(parsed.get("is_negated"))
            ev = str(parsed.get("evidence", ""))[:200]
            return negated, ev
    except Exception as e:
        logger.warning("is_negation LLM failed: %s", e)
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Is this description generic marketing fluff?
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_PROMPT = """Is the following description mostly generic marketing fluff with no concrete, verifiable facts?

DESCRIPTION:
"{description}"

Generic fluff = vague phrases like "committed to innovation", "leading provider", "passionate about delivering", "world-class solutions", "trusted partner". These say nothing falsifiable.
Specific content = concrete numbers, dated events, named products, named people, specific roles, specific tech.

Output JSON only:
{{"is_generic": true|false, "reason": "<≤10 words>"}}"""


async def is_generic_marketing(description: str, openrouter, cost) -> bool:
    """Returns True if description is mostly fluff. Fails open (False) on error."""
    if not description or len(description) < 20:
        return False
    try:
        prompt = _GENERIC_PROMPT.format(description=description[:500])
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="is_generic_marketing")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L6")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            return bool(parsed.get("is_generic"))
    except Exception as e:
        logger.warning("is_generic_marketing LLM failed: %s", e)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Are these two funding stages compatible for ICP matching?
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_PROMPT = """Are these two company funding/maturity stages compatible for B2B sales targeting?

ICP target stage: "{icp_stage}"
Candidate stage:  "{candidate_stage}"

"Compatible" = if a buyer asks for one, the other is in the same broad maturity range and worth pursuing. Same stage or adjacent stage on the standard funding-maturity spectrum is compatible. Stages that are far apart on the spectrum (very early vs already-public, for example) are NOT compatible because the go-to-market context differs.

When uncertain or one is unknown, lean toward "yes".

Output JSON only:
{{"compatible": true|false, "reason": "<≤10 words>"}}"""


async def is_stage_compatible(icp_stage: str, candidate_stage: str, openrouter, cost) -> bool:
    """Returns True if stages compatible. Fails open (True) on error or missing data."""
    if not icp_stage or not candidate_stage:
        return True
    try:
        prompt = _STAGE_PROMPT.format(icp_stage=icp_stage, candidate_stage=candidate_stage)
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="is_stage_compatible")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L4")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            return bool(parsed.get("compatible", True))
    except Exception as e:
        logger.warning("is_stage_compatible LLM failed: %s", e)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Classify URL into a source_type label
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_TYPE_PROMPT = """Classify this URL into ONE of these source-type buckets:

company_website | linkedin | job_board | news | social_media | github | wikipedia | other

URL: {url}
Title snippet: {title}

Output JSON only:
{{"source_type": "<one of the buckets>"}}"""


async def classify_source_type(url: str, title: str, openrouter, cost) -> str:
    """Returns source_type bucket. Fails open ('other') on error."""
    if not url:
        return "other"
    try:
        prompt = _SOURCE_TYPE_PROMPT.format(url=url, title=(title or "")[:200])
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="classify_source_type")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L5")
        parsed = r.get("parsed")
        valid = {"company_website", "linkedin", "job_board", "news", "social_media", "github", "wikipedia", "other"}
        if isinstance(parsed, dict):
            v = parsed.get("source_type")
            if isinstance(v, str) and v in valid:
                return v
    except Exception as e:
        logger.warning("classify_source_type LLM failed: %s", e)
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Generate evidence-gathering plan for a (company, claim) pair
# ─────────────────────────────────────────────────────────────────────────────

_EVIDENCE_PLAN_PROMPT = """For verifying this intent claim about a specific company, what kind of source URLs would carry strong evidence?

Buyer's full ask: "{buyer_prompt}"
Buyer sells: "{product_service}"
Target company name: {company}
Target company primary domain: {primary_domain}
Claim to verify: "{claim}"
Intent class: {intent_class}

Output a JSON evidence-gathering plan:

{{
  "queries": [
    "search query 1 (focused on a specific dated event/artifact)",
    "search query 2 (alternative angle)",
    "search query 3"
  ],
  "preferred_source_types": ["<which of: news, company_website, linkedin, job_board, github>"],
  "time_window_days": <integer or null — how recent must evidence be>,
  "needs_dated_event": true|false,
  "rationale": "<≤20 words: what would prove this claim>"
}}

INTENT-CLASS-SPECIFIC GUIDANCE (use the one matching the intent class above):

- funding → target press releases, the company's OWN newsroom, and tier-1 news. Use verbs an authoritative source actually uses: "raises", "raised", "secures", "closes", "announces $", "led by". Always include primary_domain or the company's literal domain in at least one query (site:primary_domain). Time window 90-365 days.

- product_launch → target product-announcement blog posts, press releases, and product launch pages. Use verbs: "announces", "launches", "introduces", "unveils", "now available", "general availability". Include site:primary_domain for at least one query.

- expansion → target press releases / blog posts about geographic or market expansion. Phrases: "expands to", "launches in", "now available in", "enters market", "opens office", "now serves", "new region".

- tech_adoption → target case studies, customer stories, partner announcements, blog mentions. Phrases: "deploys", "switches to", "uses", "powered by", "selected", "now built on".

- partnership → target joint press releases. Phrases: "partners with", "announces partnership with", "alliance with", "collaboration with", "integration with".

- leadership_change → target executive hiring announcements / press releases. Phrases: "appoints", "names new", "joins as", "hires X as", "promoted to", "incoming".

- compliance_event → target regulator filings / compliance announcements. Phrases: "complies with", "received clearance", "approved by", "certified", "regulatory milestone", "FDA approved", "achieves certification".

- hiring → LinkedIn jobs path handles this elsewhere; you can return queries that look for hiring announcements ("hires X engineers", "scaling the team", "expanding the team"), but a specialized jobs path will be tried in parallel.

- other → generic. Translate the claim into the words a source page would actually use. Don't echo the claim verbatim.

UNIVERSAL RULES:
- Each query MUST target a SPECIFIC dated event/artifact URL — NOT a landing page or about page.
- Always include primary_domain in at least one query to disambiguate the target from same-named companies. If the candidate name is short or generic (single token, common word), EVERY query should include primary_domain.
- Translate the buyer's intent phrase into the verbs the source page would use; don't echo the buyer's wording verbatim.
- Use time_window_days = 90 for fresh-news intents (funding, product launch, leadership change), 180 for partnerships / hiring / compliance, 365 for expansion / tech_adoption."""


async def plan_evidence_search(
    company: str, claim: str, openrouter, cost,
    *, buyer_prompt: str = "", product_service: str = "",
    primary_domain: str = "",
    intent_class: str = "other",
) -> dict:
    """Generate intent-agnostic evidence-gathering plan. Fails open with a basic plan."""
    fallback_query = f"{company} {claim}"
    if primary_domain:
        fallback_query = f'"{primary_domain}" {claim}'
    fallback = {
        "queries": [fallback_query],
        "preferred_source_types": ["news", "company_website"],
        "time_window_days": 180,
        "needs_dated_event": False,
        "rationale": "fallback plan",
    }
    if not company or not claim:
        return fallback
    try:
        prompt = _EVIDENCE_PLAN_PROMPT.format(
            buyer_prompt=buyer_prompt or "<unspecified>",
            product_service=product_service or "<unspecified>",
            company=company,
            primary_domain=primary_domain or "<unknown>",
            claim=claim,
            intent_class=intent_class or "other",
        )
        from qual_engine.config import CONFIG
        r = await openrouter.json_call(CONFIG.QUERY_GEN_MODEL, prompt, label="plan_evidence")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L5")
        parsed = r.get("parsed")
        if isinstance(parsed, dict) and parsed.get("queries"):
            return parsed
    except Exception as e:
        logger.warning("plan_evidence_search LLM failed: %s", e)
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# ICP-fit hard gate — does this candidate match the buyer's described target?
# ─────────────────────────────────────────────────────────────────────────────

_ICP_FIT_PROMPT = """The earlier hard filters already verified industry compatibility, country, and stage. Your job is ONLY to detect intermediaries — entities whose business is investing in, advising, or representing OTHER companies rather than operating their own product business.

CANDIDATE:
  Name:            {name}
  Primary domain:  {primary_domain}
  Industry tags:   {industry_tags}
  Country:         {country}
  HQ:              {hq}
  Size:            {employee_count_band}
  Stage:           {funding_stage}
  Description:     {description}

Buyer's ask (context only — DO NOT second-guess product fit, that's handled elsewhere): "{buyer_prompt}"
Buyer's target country: {target_geography}

DEFAULT VERDICT: matches = true.

There are only TWO reasons to return matches = false:

(A) COUNTRY MISMATCH — buyer specified a country and candidate's verified country is a different country.

(B) INTERMEDIARY — the candidate's primary business is investing in, advising, representing, listing, brokering, or funding OTHER companies. The candidate exists to facilitate transactions between other parties rather than operate its own product/service business. Look for signals in description: investment/fund/advisory/brokerage/talent-agency/intermediary language.

For ALL other cases (including: industry tag differs, product differs, sub-industry differs, description doesn't repeat buyer's exact words, size or stage one band off, you're uncertain) — return matches = true. When in doubt, KEEP. Downstream verification will filter real misfits.

Output JSON only:
{{"matches": true|false, "reason": "<≤15 words>"}}"""


async def is_icp_match(
    *, company_name: str, industry_tags: list[str], country: str, hq: str,
    employee_count_band: str, funding_stage: str, description: str,
    buyer_prompt: str, product_service: str, target_industry: str,
    target_sub_industry: str, target_geography: str,
    target_employee_count: str, target_stage: str,
    openrouter, cost,
    primary_domain: str = "",
) -> bool:
    """Cached LLM check. Fails open (True) on error."""
    try:
        from qual_engine.config import CONFIG
        prompt = _ICP_FIT_PROMPT.format(
            buyer_prompt=buyer_prompt or "<unspecified>",
            product_service=product_service or "<unspecified>",
            target_industry=target_industry or "<any>",
            target_sub_industry=target_sub_industry or "<any>",
            target_geography=target_geography or "<any>",
            target_employee_count=target_employee_count or "<any>",
            target_stage=target_stage or "<any>",
            name=company_name,
            primary_domain=primary_domain or "<unknown>",
            industry_tags="; ".join(industry_tags) or "<unknown>",
            country=country or "<unknown>",
            hq=hq or "<unknown>",
            employee_count_band=employee_count_band or "<unknown>",
            funding_stage=funding_stage or "<unknown>",
            description=(description or "")[:400],
        )
        # Use PARSER_MODEL (Gemini Flash) instead of gpt-4o-mini: empirically follows
        # structural "default keep unless X" instructions more literally.
        r = await openrouter.json_call(CONFIG.PARSER_MODEL, prompt, label="icp_fit_gate")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L4_5")
        parsed = r.get("parsed")
        if isinstance(parsed, dict) and "matches" in parsed:
            return bool(parsed["matches"])
    except Exception as e:
        logger.warning("is_icp_match LLM failed for %r: %s", company_name, e)
    return True  # fail-open


# ─────────────────────────────────────────────────────────────────────────────
# Website-anchor verification (kept for future use; not invoked while LinkedIn
# anchor is mandatory)
# ─────────────────────────────────────────────────────────────────────────────

_WEBSITE_ANCHOR_PROMPT = """You are verifying whether a website represents a single operating company that matches a buyer's ICP.

Domain: {domain}
Website content (first {max_chars} chars):
{content}

Buyer's ask: "{buyer_prompt}"
Buyer's target industry: {industry}
Buyer's target country: {country}

Determine:
1. Is this domain owned by a SINGLE OPERATING COMPANY (not a directory, news site, aggregator, listing page, or personal blog)?
2. Does the company plausibly match the buyer's described target?

If yes, extract concise facts from the website content (do not guess what isn't on the page).

Output JSON only:
{{
  "is_company": true|false,
  "matches_icp": true|false,
  "facts": {{
    "canonical_name": "<extracted from content, else null>",
    "industry": "<industry described on the page, else null>",
    "employee_count_band": "<if explicitly mentioned, else null>",
    "country": "<ISO-2 if determinable, else null>",
    "description": "<≤300 chars summary from the page>"
  }},
  "reason": "<≤15 words>"
}}"""


async def verify_website_anchor(
    domain: str,
    content: str,
    buyer_prompt: str,
    industry: str,
    country: str,
    openrouter,
    cost,
) -> Optional[dict]:
    """LLM-verifies a website as an anchor for a candidate company.

    Returns {is_company, matches_icp, facts} or None on error.
    """
    if not domain or not content or len(content) < 200:
        return None
    try:
        from qual_engine.config import CONFIG
        prompt = _WEBSITE_ANCHOR_PROMPT.format(
            domain=domain,
            max_chars=5000,
            content=content[:5000],
            buyer_prompt=buyer_prompt or "<unspecified>",
            industry=industry or "<any>",
            country=country or "<any>",
        )
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="verify_website_anchor")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L3")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning("verify_website_anchor LLM failed for %s: %s", domain, e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Is this "NO" answer a confident negation or a hedged/uncertain one?
# ─────────────────────────────────────────────────────────────────────────────

_HEDGE_PROMPT = """Read this answer to a yes/no fact-check question. Did it issue a CONFIDENT NO (denying the claim with evidence) or a HEDGED NO (unable to verify, uncertain, qualified)?

Answer text:
{text}

A confident NO contradicts the claim with positive evidence (e.g. "No — the company actually does X instead").
A hedged NO is uncertainty (e.g. "I couldn't find", "Unable to verify", "Probably not", "Unclear").

Output JSON only: {{"confident_no": true|false}}"""


async def is_confident_no(answer_text: str, openrouter, cost) -> bool:
    """Returns True if the answer is a confident NO (vs hedged/uncertain). Fails closed (False)."""
    if not answer_text:
        return False
    try:
        from qual_engine.config import CONFIG
        prompt = _HEDGE_PROMPT.format(text=answer_text[:1500])
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="is_confident_no")
        if cost is not None:
            cost.add("openrouter", r.get("cost_usd", 0), layer="L6")
        parsed = r.get("parsed")
        if isinstance(parsed, dict) and "confident_no" in parsed:
            return bool(parsed["confident_no"])
    except Exception as e:
        logger.warning("is_confident_no LLM failed: %s", e)
    return False  # fail-open (hedged) — don't reject on ambiguity


# ─────────────────────────────────────────────────────────────────────────────
# URL specificity tier (structural-only — no domain knowledge)
# ─────────────────────────────────────────────────────────────────────────────

def classify_url_tier_structural(url: str) -> int:
    """Pure structural classification — no path-keyword knowledge.

      1 = specific event (numeric id, year-in-path, or slug-shaped segment)
      2 = moderate (some path)
      3 = shallow path (root + 1 short segment)
      4 = homepage
    """
    if not url:
        return 4
    try:
        parsed = urlparse(url)
    except Exception:
        return 4
    path = parsed.path or "/"
    if path in ("", "/"):
        return 4
    segments = [s for s in path.split("/") if s]
    if not segments:
        return 4

    # Numeric id anywhere (≥4 digits) = specific
    if any(s.isdigit() and len(s) >= 4 for s in segments):
        return 1

    # Year-in-path (2000-2099) = specific (e.g. /2026/04/17/article)
    if any(s.isdigit() and len(s) == 4 and 2000 <= int(s) <= 2099 for s in segments):
        return 1

    # Slug-shaped segment (hyphenated or substantial length) anywhere in path
    # Lowered threshold from 12 to 8 chars + has hyphen — catches more real articles.
    if any(len(s) >= 8 and "-" in s for s in segments):
        return 1

    # Long single segment (≥18 chars, no hyphen) — likely a slug or article ID
    if any(len(s) >= 18 for s in segments):
        return 1

    # Deep nested path (≥3 segments) — likely specific
    if len(segments) >= 3:
        return 1

    if len(segments) == 2:
        # /foo/bar — moderately specific
        return 2

    # Single segment, no slug → category landing
    return 3
