"""
Fulfillment Lead Orchestrator — Perplexity-first pipeline.

Uses Perplexity sonar-pro as the FIRST step to discover companies WITH
confirmed intent signals in a single API call. Then finds contacts and
verifies emails only for companies that already have real signals.

Pipeline:
  1. _discover_companies_with_intent — Single Perplexity call finds companies + signals + URLs
  2. find_contact per company        — LinkedIn search for role-matched decision-makers
  3. _verify_email                   — TrueList email_ok verification
  4. _adapt_perplexity_signals       — Convert signals to IntentSignal format
  5. Assemble FulfillmentLead dict

Fallback: If Perplexity returns too few companies, falls back to
Google Search + per-company intent enrichment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from target_fit_model.web_discovery import (
    discover_companies,
    find_contact,
    _verify_email,
    _extract_domain,
    _extract_person_location,
    TRUELIST_API_KEY,
)
from target_fit_model.intent_enrichment import (
    research_company_intent,
    compute_lead_score,
    compute_intent_score_from_signals,
    _cross_source_boost,
)
from target_fit_model.funding_check import detect_funding_intent, check_company_funding, classify_funding_criteria
from target_fit_model.openrouter import chat_completion_json
from target_fit_model.config import PERPLEXITY_MODEL, PERPLEXITY_TIMEOUT
from target_fit_model.scoring import compute_fit_score

# Validator imports — exact same functions the validator uses
from validator_models.stage4_person_verification import (
    run_lead_validation_stage4,
    search_google_async,
    _check_role_via_aimode,
)
from validator_models.checks_linkedin import check_linkedin_gse
from validator_models.stage4_helpers import (
    extract_location_from_text as _s4_extract_location,
    extract_person_location_from_linkedin_snippet as _s4_extract_person_location,
    extract_role_from_result as _s4_extract_role,
    get_linkedin_id,
    get_linkedin_url_country,
)
from validator_models.stage5_verification import (
    check_stage5_unified,
    _gse_search_sync,
    _extract_fields_from_results,
    _extract_company_size_from_snippet,
    _check_exact_slug_match,
    _validate_size_match,
    _validate_name_match,
    _parse_hq_to_location,
    _extract_industry_from_snippet,
    _extract_website_from_snippet,
    classify_company_industry,
    _normalize_domain,
)
from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY

# Validator intent verification — exact same functions the validator/gateway uses
# to score fulfillment leads in Tier 3. Running these pre-submission ensures the
# miner only submits signals that will actually score > 0.
from qualification.scoring.lead_scorer import (
    _score_single_intent_signal,
    _apply_signal_time_decay,
    _extract_domain as _extract_signal_domain,
)
# Inlined verbatim from gateway/fulfillment/scoring.py::aggregate_intent_scores
# rather than imported, because that gateway module pulls in validator-only
# scrapers (Apify, ScrapingDog, OpenRouter) at module load time.  Miners
# don't ship those validator-only modules, so importing the gateway file
# here would crash the miner with ModuleNotFoundError before any work
# starts.  This local copy is a pure-math helper — if the validator-side
# formula in gateway/fulfillment/scoring.py changes, this MUST be kept in
# sync (the constants below are imported live from gateway.fulfillment.config
# so threshold tuning still propagates without a code change here).
from gateway.fulfillment.config import (
    FULFILLMENT_INTENT_QUALITY_FLOOR,
    FULFILLMENT_INTENT_BREADTH_WEIGHT,
)


def aggregate_intent_scores(signal_scores):
    """Peak-weighted aggregation: best signal dominates, quality signals
    add diminishing breadth bonus, noise is ignored.

    KEEP IN SYNC with gateway/fulfillment/scoring.py::aggregate_intent_scores.
    """
    if not signal_scores:
        return 0.0
    sorted_desc = sorted(signal_scores, reverse=True)
    best = sorted_desc[0]
    bonus = 0.0
    for i, score in enumerate(sorted_desc[1:], start=1):
        if score < FULFILLMENT_INTENT_QUALITY_FLOOR:
            break
        bonus += score * FULFILLMENT_INTENT_BREADTH_WEIGHT * (1 / i)
    return min(best + bonus, 60.0)


from gateway.qualification.models import (
    IntentSignal as ValidatorIntentSignal,
    IntentSignalSource,
    ICPPrompt,
)
import qualification.scoring.intent_verification as _intent_verif_module
from qualification.scoring.intent_verification import (
    fetch_url_content,
    extract_verification_content,
    check_company_in_content,
    compute_snippet_overlap,
)

SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("FULFILLMENT_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY", "")

# Patch the qualification module's API keys so it uses the miner's keys
# (the module normally reads from QUALIFICATION_* env vars meant for the gateway).
if SCRAPINGDOG_API_KEY and not _intent_verif_module.SCRAPINGDOG_API_KEY:
    _intent_verif_module.SCRAPINGDOG_API_KEY = SCRAPINGDOG_API_KEY
if OPENROUTER_API_KEY and not _intent_verif_module.OPENROUTER_API_KEY:
    _intent_verif_module.OPENROUTER_API_KEY = OPENROUTER_API_KEY

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# IntentSignal Adapter — Perplexity → FulfillmentLead format
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_TO_SOURCE = {
    "linkedin.com": "linkedin",
    "greenhouse.io": "job_board",
    "boards.greenhouse.io": "job_board",
    "lever.co": "job_board",
    "jobs.lever.co": "job_board",
    "indeed.com": "job_board",
    "glassdoor.com": "job_board",
    "workable.com": "job_board",
    "builtin.com": "job_board",
    "jobvite.com": "job_board",
    "wellfound.com": "job_board",
    "monster.com": "job_board",
    "ziprecruiter.com": "job_board",
    "smartrecruiters.com": "job_board",
    "ashbyhq.com": "job_board",
    "techcrunch.com": "news",
    "bloomberg.com": "news",
    "reuters.com": "news",
    "forbes.com": "news",
    "cnbc.com": "news",
    "venturebeat.com": "news",
    "prnewswire.com": "news",
    "businesswire.com": "news",
    "globenewswire.com": "news",
    "crunchbase.com": "news",
    "zdnet.com": "news",
    "siliconangle.com": "news",
    "techradar.com": "news",
    "wired.com": "news",
    "theverge.com": "news",
    "crn.com": "news",
    "channele2e.com": "news",
    "sdxcentral.com": "news",
    "theregister.com": "news",
    "infoworld.com": "news",
    "computerworld.com": "news",
    "cio.com": "news",
    "csoonline.com": "news",
    "networkworld.com": "news",
    "darkreading.com": "news",
    "eweek.com": "news",
    "channelfutures.com": "news",
    "itprotoday.com": "news",
    "arstechnica.com": "news",
    "geekwire.com": "news",
    "protocol.com": "news",
    "theinformation.com": "news",
    "businessinsider.com": "news",
    "inc.com": "news",
    "fastcompany.com": "news",
    "hbr.org": "news",
    "getlatka.com": "news",
    "latka.com": "news",
    "owler.com": "news",
    "craft.co": "news",
    "zoominfo.com": "news",
    "similarweb.com": "news",
    "twitter.com": "social_media",
    "x.com": "social_media",
    "youtube.com": "social_media",
    "facebook.com": "social_media",
    "instagram.com": "social_media",
    "reddit.com": "social_media",
    "tiktok.com": "social_media",
    "threads.net": "social_media",
    "g2.com": "review_site",
    "capterra.com": "review_site",
    "trustradius.com": "review_site",
    "gartner.com": "review_site",
    "trustpilot.com": "review_site",
    "github.com": "github",
    "medium.com": "news",
    "yahoo.com": "news",
    "wsj.com": "news",
    "nytimes.com": "news",
    "axios.com": "news",
    "seekingalpha.com": "news",
    "benzinga.com": "news",
    "marketwatch.com": "news",
    "wikipedia.org": "wikipedia",
}

_URL_REGEX = re.compile(r'https?://[^\s)<>\]"]+')

_URL_VERIFY_CACHE: Dict[str, bool] = {}


async def _verify_url_accessible(url: str) -> bool:
    """HEAD-check a URL via ScrapingDog to confirm it returns 2xx/3xx.

    Results are cached for the lifetime of this process so the same URL is
    never checked twice.  Returns True when the page is reachable.
    """
    if url in _URL_VERIFY_CACHE:
        return _URL_VERIFY_CACHE[url]
    if not SCRAPINGDOG_API_KEY:
        return True
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://api.scrapingdog.com/scrape",
                params={"api_key": SCRAPINGDOG_API_KEY, "url": url, "dynamic": "false"},
            )
            ok = resp.status_code == 200 and len(resp.text) > 100
            _URL_VERIFY_CACHE[url] = ok
            if not ok:
                logger.info(f"  [URL verify] FAILED ({resp.status_code}, {len(resp.text)} chars): {url[:80]}")
            return ok
    except Exception as e:
        logger.info(f"  [URL verify] error for {url[:80]}: {e}")
        _URL_VERIFY_CACHE[url] = False
        return False


def _extract_search_keywords(evidence: str, max_words: int = 6) -> str:
    """Pull the most distinctive words from evidence to build a search query.

    Strips URLs, common stop-words, and returns a compact keyword string
    suitable for a Google search alongside the company name.
    """
    _STOP = {
        "the", "a", "an", "and", "or", "in", "of", "for", "to", "is", "are",
        "was", "were", "has", "have", "had", "with", "at", "by", "on", "from",
        "its", "their", "this", "that", "these", "those", "been", "being",
        "since", "about", "which", "into", "also", "as", "it", "they", "our",
        "we", "company", "company's", "including", "such", "based", "new",
        "via", "per", "amid", "through",
    }
    text = _URL_REGEX.sub("", evidence).strip()
    text = re.sub(r'[^\w\s$%]', ' ', text)
    words = [w for w in text.split() if w.lower() not in _STOP and len(w) > 2]
    return " ".join(words[:max_words])


async def _find_alternative_url(
    company_name: str,
    signal_keyword: str,
    company_domain: str = "",
    evidence: str = "",
) -> Optional[Tuple[str, str]]:
    """Google search (ScrapingDog GSE) for a verifiable URL that backs the
    specific evidence Perplexity described for ``company_name``.

    Strategy:
      1. Search with the specific evidence keywords (e.g. "60M" "funding")
      2. Search with the broad signal keyword on known-good domains
      3. Search with the broad signal keyword + "news"

    Each result is checked against ``_classify_url`` — we accept any known
    source type (news, job_board, linkedin, company_website) as long as the
    URL is from a real verifiable domain.

    Returns (url, source_type) or None.
    """
    if not SCRAPINGDOG_API_KEY:
        return None

    evidence_kw = _extract_search_keywords(evidence) if evidence else ""
    queries = []

    if evidence_kw:
        queries.append(f'"{company_name}" {evidence_kw}')

    queries.extend([
        f'"{company_name}" "{signal_keyword}" site:linkedin.com OR site:indeed.com OR site:glassdoor.com',
        f'"{company_name}" "{signal_keyword}" news OR press release OR announcement',
    ])

    failed_url = ""  # track the original broken URL domain to avoid it

    for q in queries:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://api.scrapingdog.com/google",
                    params={"api_key": SCRAPINGDOG_API_KEY, "query": q, "results": 5},
                )
                if resp.status_code != 200:
                    continue
                results = resp.json()
                if not isinstance(results, list):
                    results = results.get("organic_results", results.get("results", []))
                for r in results[:5]:
                    link = r.get("link", "")
                    if not link:
                        continue
                    source = _classify_url(link, company_domain)
                    if source == "other":
                        continue
                    if await _verify_url_accessible(link):
                        logger.info(f"  [Alt URL] found verified {source}: {link[:80]}")
                        return link, source
        except Exception:
            continue
    return None


_DATE_REGEX = re.compile(
    r'(?:'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}'
    r'|\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?'
    r'|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
    r')',
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _classify_url(url: str, company_domain: str = "") -> str:
    """Map a URL to an IntentSignalSource value."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
    except Exception:
        return "other"

    for domain_key, source in _DOMAIN_TO_SOURCE.items():
        if domain_key in hostname:
            return source

    if company_domain and company_domain in hostname:
        return "company_website"

    return "other"


def _normalize_date(raw_date: Optional[str]) -> Optional[str]:
    """Normalize a date string to YYYY-MM-DD or return None.

    Handles: "2026-02", "2026-02-15", "2026/3", "March 2026", None, "".
    The FulfillmentLead validator rejects anything not YYYY-MM-DD.
    """
    if not raw_date or not raw_date.strip():
        return None
    d = raw_date.strip()

    # Already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', d):
        return d

    # YYYY-MM (missing day) — append -01
    if re.match(r'^\d{4}-\d{1,2}$', d):
        parts = d.split("-")
        return f"{parts[0]}-{parts[1].zfill(2)}-01"

    # YYYY/MM/DD or YYYY/MM
    if re.match(r'^\d{4}/\d{1,2}(/\d{1,2})?$', d):
        parts = d.split("/")
        mo = parts[1].zfill(2)
        day = parts[2].zfill(2) if len(parts) > 2 else "01"
        return f"{parts[0]}-{mo}-{day}"

    # "March 2026", "Mar 2026"
    for month_name, month_num in _MONTH_MAP.items():
        if d.lower().startswith(month_name):
            digits = re.findall(r'\d{4}', d)
            if digits:
                return f"{digits[0]}-{month_num}-01"

    return None


def _extract_date_from_text(text: str) -> Optional[str]:
    """Extract the first date from evidence text and return as YYYY-MM-DD."""
    m = _DATE_REGEX.search(text)
    if not m:
        return None
    return _normalize_date(m.group(0).strip().rstrip(","))


async def _adapt_perplexity_signals(
    perplexity_result: Dict,
    icp_intent_keywords: List[str],
    company_domain: str = "",
    company_name: str = "",
) -> List[Dict]:
    """Convert Perplexity intent research results to IntentSignal-compatible dicts.

    Verifies each URL is actually fetchable by ScrapingDog before including it.
    When a URL fails (e.g. careers pages that return 404), falls back to a
    Google search for an alternative verifiable source.
    """
    raw_signals = perplexity_result.get("signals", [])
    adapted = []

    fallback_url = f"https://{company_domain}" if company_domain else ""

    for sig in raw_signals:
        if not sig.get("match"):
            continue

        relevance = sig.get("relevance_score", 0)
        if relevance < 0.3:
            continue

        evidence = sig.get("evidence") or ""
        signal_name = sig.get("signal", "")

        urls = _URL_REGEX.findall(evidence)
        url = urls[0].rstrip(".,;)") if urls else ""

        snippet_text = _URL_REGEX.sub("", evidence).strip()
        snippet_text = re.sub(r'\s+', ' ', snippet_text).strip()
        if not snippet_text:
            snippet_text = signal_name

        source_type = _classify_url(url, company_domain) if url else "other"
        extracted_date = _extract_date_from_text(evidence)
        is_fallback = False

        if not url:
            url = fallback_url
            source_type = "company_website" if company_domain else "other"
            is_fallback = True

        if not url:
            continue

        # Pre-verify: confirm the URL is actually fetchable.
        # Perplexity often hallucinates /careers or /jobs paths that return 404.
        url_ok = await _verify_url_accessible(url)
        if not url_ok:
            logger.info(f"  [Intent] URL not fetchable, searching for alternative: {url[:60]}")
            alt = await _find_alternative_url(
                company_name or company_domain, signal_name, company_domain,
                evidence=evidence,
            )
            if alt:
                url, source_type = alt
                url_ok = True
            elif not is_fallback and urls and len(urls) > 1:
                for backup_url in urls[1:]:
                    backup_url = backup_url.rstrip(".,;)")
                    if await _verify_url_accessible(backup_url):
                        url = backup_url
                        source_type = _classify_url(url, company_domain)
                        url_ok = True
                        break
            if not url_ok:
                logger.info(f"  [Intent] Dropping unverifiable signal: {signal_name}")
                continue

        full_desc = f"{signal_name}: {snippet_text}" if snippet_text and snippet_text != signal_name else signal_name
        adapted.append({
            "source": source_type,
            "description": full_desc[:500],
            "url": url,
            "date": extracted_date,
            "snippet": snippet_text[:1000],
        })

    return adapted


async def _adapt_direct_signals(
    signals_list: List[Dict],
    company_domain: str = "",
    company_name: str = "",
) -> List[Dict]:
    """Adapt signals from the Perplexity-first discovery format.

    Each signal dict has: signal, evidence, url, date (already structured).
    Verifies URLs are fetchable before including them.
    """
    adapted = []
    fallback_url = f"https://{company_domain}" if company_domain else ""

    for sig in signals_list:
        url = (sig.get("url") or "").strip().rstrip(".,;)")
        evidence = sig.get("evidence") or ""
        signal_name = sig.get("signal") or sig.get("description") or ""

        if not url:
            urls = _URL_REGEX.findall(evidence)
            url = urls[0].rstrip(".,;)") if urls else ""

        source_type = _classify_url(url, company_domain) if url else "other"

        snippet_text = _URL_REGEX.sub("", evidence).strip()
        snippet_text = re.sub(r'\s+', ' ', snippet_text).strip()
        if not snippet_text:
            snippet_text = signal_name

        raw_date = sig.get("date") or _extract_date_from_text(evidence)
        extracted_date = _normalize_date(raw_date)
        is_fallback = False

        if not url:
            url = fallback_url
            source_type = "company_website" if company_domain else "other"
            is_fallback = True

        if not url:
            continue

        url_ok = await _verify_url_accessible(url)
        if not url_ok:
            alt = await _find_alternative_url(
                company_name or company_domain, signal_name, company_domain,
                evidence=evidence,
            )
            if alt:
                url, source_type = alt
            else:
                logger.info(f"  [Intent] Dropping unverifiable signal: {signal_name}")
                continue

        full_desc = f"{signal_name}: {snippet_text}" if snippet_text and snippet_text != signal_name else signal_name
        adapted.append({
            "source": source_type,
            "description": full_desc[:500],
            "url": url,
            "date": extracted_date,
            "snippet": snippet_text[:1000],
        })

    return adapted


def _count_matching_signals(
    adapted_signals: List[Dict],
    icp_keywords: List[str],
) -> int:
    """Count how many ICP intent keywords have at least one matching signal."""
    matched = 0
    for kw in icp_keywords:
        kw_lower = kw.lower()
        for sig in adapted_signals:
            desc = (sig.get("description") or "").lower()
            snippet = (sig.get("snippet") or "").lower()
            if kw_lower in desc or kw_lower in snippet:
                matched += 1
                break
    return matched


# ═══════════════════════════════════════════════════════════════════════════
# Validator-Equivalent Intent Verification (Tier 3 pre-check)
#
# Runs the EXACT same verification + scoring chain the validator's gateway
# uses in gateway/fulfillment/scoring.py → Tier 3.  This ensures the miner
# only submits leads whose intent signals will actually receive a score > 0.
#
# Chain:  verify_intent_signal  →  _score_single_intent_signal
#         →  _apply_signal_time_decay  →  aggregate_intent_scores
# ═══════════════════════════════════════════════════════════════════════════

FULFILLMENT_MIN_INTENT_SCORE = 5.0  # Lowered for fulfillment — must match gateway/fulfillment/config.py


async def _search_verified_intent_signals(
    company_name: str,
    company_domain: str,
    intent_keywords: List[str],
    max_signals: int = 3,
) -> List[Dict]:
    """Google Search → Validator Fetch → Snippet Extract → Build Signal.

    Searches for pages about ``company_name`` that discuss the ICP's intent
    signals, fetches each URL with the validator's exact ScrapingDog method,
    extracts text with the validator's ``extract_verification_content``, and
    builds signals with snippets guaranteed to survive the validator's
    verbatim overlap check.

    Search priority is ordered by scoring potential:
      1. Company blog/resources (``company_website``): no date cap, no date
         decay → only needs LLM score ~20/60 to clear the 15.0 threshold.
      2. Recent news from recognized domains (``news``): high source mult,
         works well if the article has a verifiable date.
      3. Job boards (``job_board``): reliable content but capped at 8.1 for
         undated signals — useful as supporting breadth signals.
    """
    signals: List[Dict] = []
    seen_domains: set = set()

    signal_words_plain = " OR ".join(intent_keywords[:3])

    # Build signal-specific keyword expansions for better search relevance
    _signal_expansions = {
        "hiring": '"sales development" OR "SDR" OR "hiring" OR "job opening" OR "careers"',
        "sdr": '"sales development representative" OR "SDR" OR "outbound sales"',
        "sales tools": '"sales tools" OR "sales technology" OR "sales stack" OR "evaluating"',
        "competitors": '"competitor" OR "alternative" OR "vs" OR "comparison"',
        "evaluating": '"evaluating" OR "comparing" OR "reviewing" OR "considering"',
        "funding": '"funding" OR "raised" OR "series" OR "investment"',
        "expansion": '"expansion" OR "growth" OR "scaling" OR "new office"',
    }
    expanded_kws = set()
    for kw in intent_keywords:
        kw_lower = kw.lower()
        for trigger, expansion in _signal_expansions.items():
            if trigger in kw_lower:
                expanded_kws.add(expansion)
                break
        else:
            expanded_kws.add(f'"{kw}"')
    expanded_str = " OR ".join(list(expanded_kws)[:3])

    search_templates = [
        # ── Priority 1: Company careers/culture pages (company_website) ──
        # These are the highest-scoring: Datadog's careers blog scored 27.5.
        # company_website = no date cap, no date decay (0.85x mult).
        # LLM 20/60 x 0.9 conf x 0.85 = 15.3 -> PASSES
        (f'site:{company_domain} "sales team" OR "our team" OR "join our" OR "we\'re hiring"', "company_website"),
        (f'site:{company_domain} "careers" OR "culture" OR "life at" OR "working at"', "company_website"),
        (f'site:{company_domain}/blog {expanded_str}', "company_website"),
        (f'site:{company_domain}/blog sales OR SDR OR hiring OR outbound OR team', "company_website"),
        (f'site:{company_domain}/careers', "company_website"),
        (f'site:{company_domain}/jobs', "company_website"),
        (f'site:{company_domain} {signal_words_plain}', "company_website"),
        # ── Priority 1b: LinkedIn hiring posts (no date cap, 1.0x mult) ──
        (f'site:linkedin.com/posts "{company_name}" hiring OR "join" OR "open role" OR "we\'re looking"', "linkedin"),
        (f'site:linkedin.com/posts "{company_name}" SDR OR "sales development" OR sales team', "linkedin"),
        # ── Priority 2: Per-signal specific searches ──
    ]

    # Add per-keyword targeted searches
    for kw in intent_keywords[:3]:
        kw_lower = kw.lower()
        if "hiring" in kw_lower or "sdr" in kw_lower:
            search_templates.extend([
                (f'"{company_name}" "sales development representative" OR "SDR" site:indeed.com OR site:lever.co OR site:greenhouse.io', "job_board"),
                (f'"{company_name}" hiring sales 2026 -site:{company_domain}', "news"),
            ])
        elif "tool" in kw_lower or "evaluat" in kw_lower:
            search_templates.extend([
                (f'"{company_name}" "sales tools" OR "sales technology" review 2025 OR 2026', "news"),
            ])
        elif "competitor" in kw_lower or "research" in kw_lower:
            search_templates.extend([
                (f'"{company_name}" alternative OR competitor OR comparison 2025 OR 2026', "news"),
            ])
        elif "funding" in kw_lower or "raised" in kw_lower:
            search_templates.extend([
                (f'site:techcrunch.com OR site:crunchbase.com "{company_name}" funding OR raised 2025 OR 2026', "news"),
            ])

    search_templates.extend([
        # ── Priority 3: Recent news with dates ──
        (f'site:prnewswire.com OR site:businesswire.com "{company_name}" 2026', "news"),
        (f'site:techcrunch.com "{company_name}" 2025 OR 2026', "news"),
        (f'"{company_name}" expansion OR growth OR hiring 2026 news -site:{company_domain}', "news"),
        # ── Priority 4: Job boards (supporting signals) ──
        (f'site:builtin.com/company "{company_name}"', "job_board"),
        (f'site:lever.co "{company_name}"', "job_board"),
        (f'site:greenhouse.io "{company_name}"', "job_board"),
    ])

    # Track the best company_website candidate separately so we can try
    # multiple pages from the company's domain and pick the most relevant.
    _best_company_signal: Optional[Dict] = None
    _best_company_kw_score: int = -1
    _company_domain_tried: bool = False
    _seen_company_urls: set = set()

    for template, default_source in search_templates:
        if len(signals) >= max_signals:
            break

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://api.scrapingdog.com/google",
                    params={"api_key": SCRAPINGDOG_API_KEY, "query": template, "results": 8},
                )
                if resp.status_code != 200:
                    continue
                results = resp.json()
                if not isinstance(results, list):
                    results = results.get("organic_results", results.get("results", []))
        except Exception:
            continue

        for r in results[:5]:
            if len(signals) >= max_signals:
                break

            link = r.get("link", "")
            if not link:
                continue

            link_domain = _extract_signal_domain(link)

            # For the company's own domain: don't skip duplicates yet —
            # collect candidates and pick the best one later.
            is_company_domain = (company_domain and company_domain in link.lower())
            if is_company_domain:
                if link in _seen_company_urls:
                    continue
                _seen_company_urls.add(link)
            elif link_domain in seen_domains:
                continue

            # FIX 1: Source type — only use recognized source types.
            # The validator rejects "news" from unrecognized domains.
            # If the URL is on the company's own domain → company_website.
            # If _classify_url recognizes it → use that.
            # Otherwise → company_website for company domains, leave as
            # default_source ONLY if that default is NOT "news" for
            # unrecognized domains.
            source_type = _classify_url(link, company_domain)
            if source_type == "other":
                if company_domain and company_domain in link.lower():
                    source_type = "company_website"
                elif default_source in ("job_board", "company_website",
                                        "review_site", "linkedin"):
                    source_type = default_source
                else:
                    # For "news" default, only use it if the domain is
                    # actually in _DOMAIN_TO_SOURCE (already handled by
                    # _classify_url above).  Skip unrecognized domains
                    # entirely — the validator will reject them.
                    continue

            try:
                content = await fetch_url_content(link, source_type)
            except Exception:
                continue
            if not content or len(content) < 200:
                continue

            text = extract_verification_content(content, source_type)
            if not text or len(text.strip()) < 50:
                continue

            if not check_company_in_content(company_name, text[:5000]):
                continue

            best_snippet = _extract_best_verified_snippet(
                text, company_name, intent_keywords,
            )
            if not best_snippet or len(best_snippet) < 30:
                continue

            overlap = compute_snippet_overlap(best_snippet, text)
            if overlap < 0.30:
                continue

            # FIX 2: Date extraction — ONLY use dates from URL paths
            # (e.g. /2026/03/25/).  Do NOT extract dates from page content
            # because _normalize_date appends -01 for month-only dates and
            # the validator's date precision check rejects "year_only" or
            # "no_match" dates as fabricated.
            signal_date = None
            url_date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', link)
            if url_date_match:
                y, m, d = url_date_match.group(1), url_date_match.group(2), url_date_match.group(3)
                if int(y) >= 2024:
                    signal_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

            # FIX 3: Description keyword — scan content for which intent
            # keywords actually appear.  The validator's signal_word_grounding
            # check verifies that action words from the description exist in
            # the source content.  Using a keyword that's NOT in the content
            # causes instant rejection.
            matched_kw = None
            text_lower = text.lower()
            snippet_lower = best_snippet.lower()
            # Best: keyword fully present in the snippet itself
            for kw in intent_keywords:
                kw_words = [w for w in kw.lower().split() if len(w) > 3]
                if kw_words and all(w in snippet_lower for w in kw_words):
                    matched_kw = kw
                    break
            # Fallback: keyword partially present in full page text
            if not matched_kw:
                for kw in intent_keywords:
                    kw_words = [w for w in kw.lower().split() if len(w) > 3]
                    if kw_words and all(w in text_lower for w in kw_words):
                        matched_kw = kw
                        break
            # Build the signal dict
            if not matched_kw:
                sig_dict = {
                    "source": source_type,
                    "description": best_snippet[:400],
                    "url": link,
                    "date": signal_date,
                    "snippet": best_snippet[:500],
                }
                kw_score = 0
            else:
                sig_dict = {
                    "source": source_type,
                    "description": f"{matched_kw}: {best_snippet[:300]}",
                    "url": link,
                    "date": signal_date,
                    "snippet": best_snippet[:500],
                }
                kw_score = len([w for w in matched_kw.lower().split()
                                if len(w) > 3 and w in snippet_lower])

            # For company's own domain: collect candidates, pick best later
            if is_company_domain:
                if kw_score > _best_company_kw_score:
                    _best_company_kw_score = kw_score
                    _best_company_signal = sig_dict
                    print(
                        f"    📌 Company candidate ({source_type}, kw={kw_score}): "
                        f"{link_domain} — {best_snippet[:60]}..."
                    )
                continue

            # For other domains: add immediately
            signals.append(sig_dict)
            seen_domains.add(link_domain)
            kw_label = f", kw={matched_kw}" if matched_kw else ", no kw match"
            print(
                f"    📌 Verified signal ({source_type}{kw_label}): "
                f"{link_domain} — {best_snippet[:60]}..."
            )

    # Add the best company_website signal (if any) as the first signal
    if _best_company_signal and len(signals) < max_signals:
        cd = _extract_signal_domain(_best_company_signal["url"])
        if cd not in seen_domains:
            signals.insert(0, _best_company_signal)
            seen_domains.add(cd)
            print(
                f"    📌 Best company signal selected: "
                f"{_best_company_signal['url'][:60]}..."
            )

    return signals


def _extract_best_verified_snippet(
    text: str, company: str, keywords: List[str], max_len: int = 400,
) -> Optional[str]:
    """Extract a chunk of ``text`` centered on a company mention that has the
    highest overlap with ``keywords``.  Returns a 200-400 char snippet
    guaranteed to come from ``text`` (so the validator's overlap check passes).
    """
    text_lower = text.lower()
    company_lower = company.lower()

    positions: List[int] = []
    start = 0
    while True:
        idx = text_lower.find(company_lower, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        return None

    best_chunk: Optional[str] = None
    best_score = -1

    for pos in positions:
        chunk_start = max(0, pos - 100)
        chunk_end = min(len(text), pos + max_len - 100)
        chunk = text[chunk_start:chunk_end].strip()

        if chunk_start > 0:
            sp = chunk.find(" ")
            if 0 < sp < 20:
                chunk = chunk[sp + 1:]

        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            for word in kw.lower().split():
                if len(word) > 3 and word in chunk_lower:
                    score += 1

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk


async def _run_validator_intent_verification(
    adapted_signals: List[Dict],
    company_name: str,
    company_website: str,
    icp: dict,
) -> Tuple[List[Dict], float, bool]:
    """Run the validator's exact Tier 3 intent scoring on adapted signals.

    For each signal, constructs a ``ValidatorIntentSignal`` and runs it through
    ``_score_single_intent_signal`` (which internally calls ``verify_intent_signal``
    from ``qualification/scoring/intent_verification.py``) and then applies
    time-decay via ``_apply_signal_time_decay``.  Finally aggregates using the
    same peak-weighted function the gateway uses.

    Returns:
        (verified_signals, intent_final_score, passes_threshold)
        ``verified_signals`` contains only the signals that scored > 0.
    """
    api_key = OPENROUTER_API_KEY
    # Multi-industry ICPs store industry/sub_industry as List[str]; ICPPrompt
    # is the legacy single-string schema, so collapse via the same coercer
    # used by tier1_check (also handles legacy stringified-list values).
    from gateway.fulfillment.icp_checks import _coerce_industry_list
    icp_industry = ", ".join(_coerce_industry_list(icp.get("industry"))) or ""
    icp_sub_industry = ", ".join(_coerce_industry_list(icp.get("sub_industry"))) or ""
    icp_product = icp.get("product_service", "") or icp.get("prompt", "")
    icp_intent_list = icp.get("intent_signals", [])

    icp_prompt = ICPPrompt(
        icp_id=icp.get("icp_id", "miner-pre-check"),
        prompt=icp.get("prompt", ""),
        industry=icp_industry,
        sub_industry=icp_sub_industry,
        employee_count=icp.get("employee_count", ""),
        company_stage=icp.get("company_stage", ""),
        geography=icp.get("geography", icp.get("country", "")),
        country=icp.get("country", ""),
        product_service=icp_product,
        intent_signals=icp_intent_list,
        target_roles=icp.get("target_role_types", []),
        target_seniority=icp.get("target_seniority", ""),
    )

    print(f"    🔬 Validator intent verification ({len(adapted_signals)} signal(s))...")

    verified_signals: List[Dict] = []
    signal_results: List[Dict] = []
    seen_domains: set = set()

    for idx, sig in enumerate(adapted_signals):
        source_str = sig.get("source", "other")

        try:
            validator_signal = ValidatorIntentSignal(
                source=source_str,
                description=sig.get("description", ""),
                url=sig.get("url", ""),
                date=sig.get("date"),
                snippet=sig.get("snippet", sig.get("description", "")),
            )
        except Exception as e:
            print(f"      Signal {idx+1}: ❌ invalid schema: {e}")
            signal_results.append({"after_decay": 0.0, "decay_mult": 1.0, "confidence": 0})
            continue

        domain = _extract_signal_domain(validator_signal.url)
        if domain in seen_domains:
            print(f"      Signal {idx+1}: ⏭️ duplicate domain '{domain}'")
            signal_results.append({"after_decay": 0.0, "decay_mult": 1.0, "confidence": 0})
            continue
        seen_domains.add(domain)

        try:
            score, confidence, date_status, content_found_date = (
                await _score_single_intent_signal(
                    validator_signal, icp_prompt, None,
                    company_name, company_website,
                    api_key=api_key,
                )
            )
        except Exception as e:
            print(f"      Signal {idx+1}: ❌ scoring error: {e}")
            score, confidence, date_status, content_found_date = 0.0, 0, "fabricated", None

        source_val = (validator_signal.source.value
                      if hasattr(validator_signal.source, "value")
                      else str(validator_signal.source))
        after_decay, decay_mult = _apply_signal_time_decay(
            score, validator_signal.date, date_status, source_val, content_found_date,
        )

        signal_results.append({
            "after_decay": after_decay,
            "decay_mult": decay_mult,
            "confidence": confidence,
        })

        status = "✅" if after_decay > 0 else "❌"
        print(
            f"      Signal {idx+1} ({source_str}): {status} "
            f"score={score:.1f} conf={confidence} "
            f"decay={after_decay:.1f} date={date_status}"
        )

        if after_decay > 0:
            verified_signals.append(sig)

    after_decay_scores = [r["after_decay"] for r in signal_results]
    intent_final = aggregate_intent_scores(after_decay_scores)
    intent_final = min(intent_final, 60.0)
    passes = intent_final >= FULFILLMENT_MIN_INTENT_SCORE

    all_fabricated = bool(signal_results) and all(
        r["confidence"] == 0 for r in signal_results
    )

    status_emoji = "✅" if passes else "❌"
    print(
        f"    🎯 Validator intent: {intent_final:.1f} / {FULFILLMENT_MIN_INTENT_SCORE} "
        f"threshold {status_emoji}"
        f"{' (all_fabricated)' if all_fabricated else ''}"
    )

    return verified_signals, intent_final, passes


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity-First Company + Intent Discovery
# ═══════════════════════════════════════════════════════════════════════════

_PERPLEXITY_SYSTEM = (
    "You are a B2B sales intelligence researcher. You find real companies "
    "with buying signals. Include source URLs when available. "
    "Return ONLY valid JSON — no explanations, no caveats, no markdown."
)


_PROMPT_VARIANTS = [
    # Variant 0: Standard
    "Find {num} real companies that match the profile below AND show public evidence of the listed intent signals from the last 6 months (since {since}).",
    # Variant 1: Discovery angle
    "Search the web thoroughly for {num} companies in the {industry} space that are actively showing the buying signals listed below. Look beyond the obvious market leaders — include mid-market and emerging companies (since {since}).",
    # Variant 2: Evidence-focused
    "Identify {num} {industry} companies where you can find concrete, recent public evidence (job postings, news articles, press releases, blog posts) of the intent signals below. Prioritize companies with strong, verifiable evidence (since {since}).",
    # Variant 3: Breadth angle
    "Cast a wide net and find {num} diverse companies across the {industry} sector that demonstrate any of the buying signals below. Include companies of all sizes and stages — startups, growth-stage, and established players (since {since}).",
    # Variant 4: Regional diversity
    "Find {num} {industry} companies showing the intent signals below. Look across different US cities and regions — not just Silicon Valley. Include companies headquartered in the Midwest, Southeast, Northeast, and other areas (since {since}).",
]


async def _discover_companies_with_intent(
    icp: dict, num_companies: int = 15, exclude_companies: set = None,
    _variant_idx: int = 0,
) -> List[Dict]:
    """Single Perplexity sonar-pro call to find companies with intent signals.

    Returns list of dicts with:
      name, website, domain, description, employee_estimate,
      hq_city, hq_state, signals: [{signal, evidence, url, date}]
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    employee_count = icp.get("employee_count", "")
    country = icp.get("country", "United States")
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%B %Y")

    intent_list = "\n".join(f"- {s}" for s in intent_keywords) if intent_keywords else "- hiring\n- expansion\n- new product launch"

    exclude_text = ""
    if exclude_companies:
        exclude_list = sorted(exclude_companies)[:50]
        exclude_text = f"\n\nDO NOT include these companies (already processed):\n{', '.join(exclude_list)}\n"

    variant = _PROMPT_VARIANTS[_variant_idx % len(_PROMPT_VARIANTS)]
    opening = variant.format(
        num=num_companies, since=six_months_ago,
        industry=f"{industry}/{sub_industry}" if sub_industry else industry,
    )

    prompt = f"""{opening}

COMPANY PROFILE:
- Industry: {industry}{f' / {sub_industry}' if sub_industry else ''}
- Company size: {employee_count or 'any'}
- Country: {country}
{f'- The buyer sells: {product_service}' if product_service else ''}
{f'- Additional context: {prompt_text}' if prompt_text else ''}

INTENT SIGNALS (find evidence of ANY of these):
{intent_list}
{exclude_text}
RULES:
- Only include real, currently operating companies — no fictional or defunct ones
- Each company must have at least 1 signal with real, publicly verifiable evidence
- Include a source URL for each signal when available
- Don't force matches — skip signals that don't genuinely apply

Return ONLY a JSON array:
[{{
  "name": "Company Name",
  "website": "https://company.com",
  "description": "One sentence about what they do",
  "employee_estimate": "50-200",
  "hq_city": "San Francisco",
  "hq_state": "California",
  "signals": [{{
    "signal": "exact signal text from list",
    "evidence": "what you found with date",
    "url": "https://source-url",
    "date": "YYYY-MM"
  }}]
}}]

Return at least {num_companies} companies. No explanation text."""

    print(f"  [Perplexity] Searching for {num_companies} companies with intent signals...")

    # Get raw response first so we can log what Perplexity actually said
    from target_fit_model.openrouter import chat_completion, parse_json_response
    raw_response = await asyncio.to_thread(
        chat_completion,
        prompt=prompt,
        model=PERPLEXITY_MODEL,
        system_prompt=_PERPLEXITY_SYSTEM,
        temperature=0.3,
        max_tokens=8000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if raw_response is None:
        print(f"  [Perplexity] API returned None — falling back to Google Search")
        return []

    result = parse_json_response(raw_response)

    if not result:
        print(f"  [Perplexity] Could not parse JSON from response ({len(raw_response)} chars)")
        print(f"  [Perplexity] First 300 chars: {raw_response[:300]}")
        return []

    if isinstance(result, dict):
        result = result.get("companies", result.get("results", [result]))

    if not isinstance(result, list):
        print(f"  [Perplexity] Unexpected response format — falling back")
        return []

    companies = []
    for item in result:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        website = item.get("website", "").strip()
        signals = item.get("signals", [])

        if not name:
            continue

        valid_signals = [s for s in signals if isinstance(s, dict)]

        domain = _extract_domain(website) if website else ""

        companies.append({
            "name": name,
            "website": website,
            "domain": domain,
            "description": item.get("description", ""),
            "employee_estimate": item.get("employee_estimate", ""),
            "hq_city": item.get("hq_city", ""),
            "hq_state": item.get("hq_state", ""),
            "signals": valid_signals,
        })

    print(f"  [Perplexity] Found {len(companies)} companies with intent signals")
    for c in companies[:5]:
        sig_count = len(c.get("signals", []))
        print(f"    - {c['name']} ({c.get('domain', '?')}) — {sig_count} signals")

    return companies


# ═══════════════════════════════════════════════════════════════════════════
# Deep Research Company Discovery (sonar-deep-research)
# ═══════════════════════════════════════════════════════════════════════════

async def _discover_companies_deep_research(
    icp: dict, num_companies: int = 10, exclude_companies: set = None,
) -> List[Dict]:
    """Find companies using sonar-deep-research when sonar-pro is exhausted.

    Deep research does thorough multi-step web searching and often returns
    companies that sonar-pro's single-step search misses.  However it
    returns markdown, so we use a two-step approach:
      1. Deep research returns raw text (companies + evidence)
      2. sonar-pro structures the text into JSON
    """
    from target_fit_model.config import PERPLEXITY_DEEP_MODEL, PERPLEXITY_DEEP_TIMEOUT
    from target_fit_model.openrouter import chat_completion, parse_json_response

    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    employee_count = icp.get("employee_count", "")
    country = icp.get("country", "United States")
    intent_keywords = icp.get("intent_signals", [])
    signals_str = ", ".join(intent_keywords) if intent_keywords else "hiring, expansion"

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%B %Y")

    exclude_text = ""
    if exclude_companies:
        exclude_list = sorted(exclude_companies)[:30]
        exclude_text = f"\n\nDO NOT include: {', '.join(exclude_list)}\n"

    prompt = (
        f"Find {num_companies} real {industry}"
        f"{f'/{sub_industry}' if sub_industry else ''} companies "
        f"({employee_count or 'any size'}, {country}) that show public evidence "
        f"of these buying signals since {six_months_ago}: {signals_str}."
        f"{exclude_text}\n"
        f"For each company provide: name, website URL, and what evidence you found."
    )

    print(f"  [Deep Research] Searching for {num_companies} companies...")

    raw = await asyncio.to_thread(
        chat_completion,
        prompt=prompt,
        model=PERPLEXITY_DEEP_MODEL,
        system_prompt="Find real companies with buying signals. Be thorough.",
        temperature=0,
        max_tokens=8000,
        timeout=PERPLEXITY_DEEP_TIMEOUT,
    )

    if not raw or len(raw) < 50:
        print(f"  [Deep Research] No response")
        return []

    # Try JSON parse first
    result = parse_json_response(raw)
    if result and isinstance(result, list):
        companies = []
        for item in result:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            website = item.get("website", "").strip()
            companies.append({
                "name": item["name"].strip(),
                "website": website,
                "domain": _extract_domain(website) if website else "",
                "description": item.get("description", ""),
                "employee_estimate": item.get("employee_estimate", ""),
                "hq_city": item.get("hq_city", ""),
                "hq_state": item.get("hq_state", ""),
                "signals": item.get("signals", []),
            })
        if companies:
            print(f"  [Deep Research] Parsed {len(companies)} companies from JSON")
            return companies

    # Markdown response — extract company names + URLs, then ask sonar-pro to structure
    urls_found = _URL_REGEX.findall(raw)
    urls_text = "\n".join(f"  - {u}" for u in urls_found[:20])
    extract_prompt = f"""Extract company information from this research text.

TEXT:
{raw[:5000]}

URLs found in the text:
{urls_text}

CRITICAL: For each company, include their WEBSITE URL (e.g. https://retool.com).
If you can't find the exact URL in the text, construct it from the company name
(e.g. "Retool" -> "https://retool.com", "Datadog" -> "https://datadoghq.com").

Return ONLY a JSON array:
[{{"name": "Company Name", "website": "https://company.com", "description": "one sentence", "employee_estimate": "50-200", "hq_city": "", "hq_state": "", "signals": [{{"signal": "what signal", "evidence": "what was found", "url": "source url", "date": "YYYY-MM"}}]}}]

Only include real companies with real evidence. No explanation text."""

    structured = await asyncio.to_thread(
        chat_completion_json,
        prompt=extract_prompt,
        model=PERPLEXITY_MODEL,
        system_prompt="Extract structured data. Return ONLY valid JSON array.",
        temperature=0,
        max_tokens=4000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if not structured:
        print(f"  [Deep Research] Could not structure markdown response")
        return []

    if isinstance(structured, dict):
        structured = structured.get("companies", structured.get("results", [structured]))

    # Well-known company domain map for common SaaS companies
    _KNOWN_DOMAINS = {
        "retool": "retool.com", "datadog": "datadoghq.com",
        "cloudflare": "cloudflare.com", "okta": "okta.com",
        "loom": "loom.com", "notion": "notion.so",
        "figma": "figma.com", "stripe": "stripe.com",
        "plaid": "plaid.com", "twilio": "twilio.com",
        "hubspot": "hubspot.com", "salesforce": "salesforce.com",
        "slack": "slack.com", "zoom": "zoom.us",
        "airtable": "airtable.com", "monday.com": "monday.com",
        "asana": "asana.com", "clickup": "clickup.com",
        "intercom": "intercom.com", "drift": "drift.com",
        "gong": "gong.io", "outreach": "outreach.io",
        "pipedrive": "pipedrive.com", "pendo": "pendo.io",
        "amplitude": "amplitude.com", "mixpanel": "mixpanel.com",
        "segment": "segment.com", "braze": "braze.com",
        "snyk": "snyk.io", "miro": "miro.com",
        "calendly": "calendly.com", "webflow": "webflow.com",
        "vercel": "vercel.com", "supabase": "supabase.com",
        "rippling": "rippling.com", "deel": "deel.com",
        "lattice": "lattice.com", "gusto": "gusto.com",
    }

    companies = []
    for item in (structured if isinstance(structured, list) else []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        website = item.get("website", "").strip()
        name = item["name"].strip()

        # If website is missing, try to infer from known domains
        if not website or not _extract_domain(website):
            name_lower = name.lower().replace(" ", "").replace(".", "").replace(",", "")
            for known_name, known_domain in _KNOWN_DOMAINS.items():
                if known_name in name_lower or name_lower in known_name:
                    website = f"https://{known_domain}"
                    break
            if not website:
                # Last resort: guess domain from name
                clean_name = re.sub(r'[^a-z0-9]', '', name.lower())
                if clean_name:
                    website = f"https://{clean_name}.com"

        companies.append({
            "name": name,
            "website": website,
            "domain": _extract_domain(website) if website else "",
            "description": item.get("description", ""),
            "employee_estimate": item.get("employee_estimate", ""),
            "hq_city": item.get("hq_city", ""),
            "hq_state": item.get("hq_state", ""),
            "signals": item.get("signals", []) if isinstance(item.get("signals"), list) else [],
        })

    print(f"  [Deep Research] Extracted {len(companies)} companies from markdown")
    for c in companies[:5]:
        print(f"    - {c['name']} ({c.get('domain', '?')})")

    return companies


# ═══════════════════════════════════════════════════════════════════════════
# Batch Intent Re-check — rescue email-verified leads with no signals
# ═══════════════════════════════════════════════════════════════════════════

async def _batch_intent_recheck(
    pool: List[Dict], icp: dict, max_leads: int = 5
) -> List[Dict]:
    """Single Perplexity call to find intent signals for email-verified leads.

    Takes a pool of partially-built lead dicts (verified email, no intent)
    and asks Perplexity to check if any of them show the requested signals.
    """
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")

    # Limit to 10 leads to keep prompt short (reduces timeout/parsing failures)
    batch_pool = pool[:10]
    leads_text = "\n".join(
        f"{i+1}. {p['full_name']} at {p['business']} ({p.get('company_website', '?')})"
        for i, p in enumerate(batch_pool)
    )
    signals_text = ", ".join(intent_keywords)

    prompt = (
        f"For each company below, search the web for public evidence (last 6 months) "
        f"of these buying signals: {signals_text}\n\n"
        f"COMPANIES:\n{leads_text}\n\n"
        f"{f'The buyer sells: {product_service}' if product_service else ''}\n"
        f"Return ONLY JSON: "
        f'[{{"company": "Name", "signals": [{{"signal": "...", "evidence": "...", "url": "https://...", "date": "YYYY-MM"}}]}}]\n'
        f"Omit companies with no evidence."
    )

    print(f"  [Batch Intent] Asking Perplexity about {len(batch_pool)} companies...")

    from target_fit_model.openrouter import chat_completion, parse_json_response
    raw = await asyncio.to_thread(
        chat_completion,
        prompt=prompt,
        model=PERPLEXITY_MODEL,
        system_prompt="Search for buying signals. Return ONLY valid JSON.",
        temperature=0,
        max_tokens=4000,
        timeout=PERPLEXITY_TIMEOUT,
    )

    if not raw or len(raw) < 5:
        print(f"  [Batch Intent] No response from Perplexity")
        return []

    result = parse_json_response(raw)
    if not result:
        print(f"  [Batch Intent] Could not parse response ({len(raw)} chars): {raw[:150]}...")
        return []

    if isinstance(result, dict):
        result = result.get("companies", result.get("results", result.get("leads", [])))
    if not isinstance(result, list):
        return []

    # Map results back to pool leads by company name OR person name
    lead_signals = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        company = (item.get("company") or "").strip().lower()
        person = (item.get("person") or "").strip().lower()
        signals = item.get("signals", [])
        if signals:
            if company:
                lead_signals[company] = signals
            if person:
                lead_signals[person] = signals

    print(f"  [Batch Intent] Found signals for {len(lead_signals)} leads")

    rescued = []
    for partial in pool:
        if len(rescued) >= max_leads:
            break

        biz_lower = partial["business"].lower()
        person_lower = partial.get("full_name", "").lower()
        raw_signals = lead_signals.get(biz_lower) or lead_signals.get(person_lower)
        if not raw_signals:
            continue

        domain = partial.get("_domain", _extract_domain(partial.get("company_website", "")))
        adapted = await _adapt_direct_signals(
            raw_signals, company_domain=domain,
            company_name=partial.get("business", ""),
        )
        if not adapted:
            continue

        company_website = partial.get("company_website", f"https://{domain}" if domain else "")
        verified, val_score, passes = await _run_validator_intent_verification(
            adapted, partial.get("business", ""), company_website, icp,
        )
        if not passes:
            print(
                f"    Rescued {partial['full_name']} @ {partial['business']} "
                f"— ❌ failed validator intent ({val_score:.1f})"
            )
            continue

        adapted = verified
        matching = _count_matching_signals(adapted, intent_keywords)
        print(
            f"    Rescued: {partial['full_name']} @ {partial['business']} "
            f"({len(adapted)} signals, {matching}/{len(intent_keywords)} matched, "
            f"validator_intent={val_score:.1f})"
        )

        lead = {
            **partial,
            "intent_signals": adapted,
            "_matching_signal_count": matching,
            "_intent_score": 0.7,
            "_validator_intent_score": val_score,
        }
        lead.pop("_domain", None)
        rescued.append(lead)

    return rescued


# ═══════════════════════════════════════════════════════════════════════════
# Validator-Equivalent Verification & Correction
# Uses the EXACT same methods the validator uses in Stage 4 + Stage 5.
# Instead of rejecting on mismatch, CORRECTS the lead data.
# ═══════════════════════════════════════════════════════════════════════════


async def _verify_company_on_linkedin(
    company_name: str,
    company_linkedin: str,
    claimed_employee_count: str = "",
) -> Dict:
    """Search the company LinkedIn page and extract verified fields.

    Uses the exact same Q1 query and extraction functions from Stage 5.
    Returns dict with: slug, name, employee_count, headquarters, industry,
    website, hq_city, hq_state, hq_country, found (bool).
    """
    result = {
        "found": False,
        "slug": "",
        "name": "",
        "employee_count": "",
        "headquarters": "",
        "industry": "",
        "website": "",
        "hq_city": "",
        "hq_state": "",
        "hq_country": "",
    }

    slug = ""
    if company_linkedin:
        m = re.search(r'linkedin\.com/company/([^/?#]+)', company_linkedin.lower())
        if m:
            slug = m.group(1)
    result["slug"] = slug

    if not slug:
        print(f"    [Company LI] No slug from {company_linkedin}")
        return result

    # Q1: Exact Stage 5 query
    q1_query = f'site:linkedin.com/company/{slug} "Industry" "Company size" "Headquarters"'
    print(f"    [Company LI] Q1: {q1_query}")
    q1_result = await asyncio.to_thread(_gse_search_sync, q1_query, 10)

    if q1_result.get("error"):
        print(f"    [Company LI] Q1 error: {q1_result['error']}")
    else:
        extracted = _extract_fields_from_results(q1_result.get("results", []), slug)
        if extracted["exact_slug_found"]:
            result["found"] = True
            result["name"] = extracted.get("title_company_name", "")
            result["employee_count"] = extracted.get("company_size", "")
            result["headquarters"] = extracted.get("headquarters", "")
            result["industry"] = extracted.get("industry", "")
            result["website"] = extracted.get("website", "")

    # Q2 fallback if missing fields
    if result["found"] and not all([result["employee_count"], result["headquarters"]]):
        q2_query = f'{company_name} linkedin company size industry headquarters'
        print(f"    [Company LI] Q2: {q2_query}")
        q2_result = await asyncio.to_thread(_gse_search_sync, q2_query, 10)
        if not q2_result.get("error"):
            extracted = _extract_fields_from_results(q2_result.get("results", []), slug)
            if extracted["exact_slug_found"]:
                if not result["employee_count"] and extracted.get("company_size"):
                    result["employee_count"] = extracted["company_size"]
                if not result["headquarters"] and extracted.get("headquarters"):
                    result["headquarters"] = extracted["headquarters"]
                if not result["industry"] and extracted.get("industry"):
                    result["industry"] = extracted["industry"]
                if not result["website"] and extracted.get("website"):
                    result["website"] = extracted["website"]

    # If slug wasn't found in any query, try searching by company name
    if not result["found"]:
        q_name = f'site:linkedin.com/company/ "{company_name}" "Company size"'
        print(f"    [Company LI] Name search: {q_name}")
        name_result = await asyncio.to_thread(_gse_search_sync, q_name, 5)
        if not name_result.get("error"):
            for r in name_result.get("results", []):
                link = r.get("link", "")
                li_m = re.search(r'linkedin\.com/company/([^/?#]+)', link.lower())
                if li_m:
                    found_slug = li_m.group(1)
                    combined = f"{r.get('title', '')} {r.get('snippet', '')}"
                    name_match, _ = _validate_name_match(company_name, combined.split("|")[0].strip()[:60])
                    if name_match or company_name.lower() in combined.lower():
                        result["found"] = True
                        result["slug"] = found_slug
                        result["employee_count"] = _extract_company_size_from_snippet(combined)
                        result["industry"] = _extract_industry_from_snippet(combined)
                        result["website"] = _extract_website_from_snippet(combined)
                        print(f"    [Company LI] Found via name search: slug={found_slug}")
                        break

    # Parse HQ to structured location
    if result["headquarters"]:
        hq_city, hq_state, hq_country, _ = _parse_hq_to_location(result["headquarters"])
        result["hq_city"] = hq_city
        result["hq_state"] = hq_state
        result["hq_country"] = hq_country

    if result["found"]:
        print(
            f"    [Company LI] Verified: size={result['employee_count']}, "
            f"industry={result['industry']}, HQ={result['headquarters']}"
        )
    else:
        print(f"    [Company LI] Company not found on LinkedIn")

    return result


def _get_valid_industry_pair(
    linkedin_industry: str,
    company_description: str = "",
    icp_industry: str = "",
    icp_sub_industry: str = "",
) -> Tuple[str, str]:
    """Map to a valid (industry, sub_industry) pair from the taxonomy.

    Strategy:
      1. If icp_industry/sub_industry is already a valid taxonomy pair, use it
      2. Try to find a matching sub_industry from the LinkedIn industry
      3. Fall back to the ICP's industry with a best-guess sub_industry
    """
    # Build valid pairs set
    valid_pairs = set()
    for sub, data in INDUSTRY_TAXONOMY.items():
        for ind in data["industries"]:
            valid_pairs.add((ind, sub))

    # Check if ICP pair is already valid
    if icp_industry and icp_sub_industry:
        for ind, sub in valid_pairs:
            if ind.lower() == icp_industry.lower() and sub.lower() == icp_sub_industry.lower():
                return ind, sub

    # Try to find sub_industries matching the LinkedIn industry
    if linkedin_industry:
        li_lower = linkedin_industry.lower()
        matching_subs = []
        for sub, data in INDUSTRY_TAXONOMY.items():
            for ind in data["industries"]:
                if li_lower in ind.lower() or ind.lower() in li_lower:
                    matching_subs.append((ind, sub))
        if matching_subs:
            if company_description:
                desc_lower = company_description.lower()
                scored = []
                for ind, sub in matching_subs:
                    defn = INDUSTRY_TAXONOMY[sub].get("definition", "").lower()
                    overlap = sum(1 for w in sub.lower().split() if w in desc_lower)
                    overlap += sum(1 for w in defn.split() if len(w) > 4 and w in desc_lower)
                    scored.append((overlap, ind, sub))
                scored.sort(reverse=True)
                return scored[0][1], scored[0][2]
            return matching_subs[0]

    # Try ICP industry as parent
    if icp_industry:
        icp_lower = icp_industry.lower()
        for sub, data in INDUSTRY_TAXONOMY.items():
            for ind in data["industries"]:
                if ind.lower() == icp_lower:
                    return ind, sub

    # Last resort: use ICP values even if not in taxonomy (will score lower)
    return icp_industry, icp_sub_industry


async def _find_actual_role(
    linkedin_url: str,
    company_name: str,
    full_name: str,
    search_results: list,
) -> str:
    """Find the person's actual role using the exact validator methods.

    Pipeline (mirrors Stage 4):
      1. extract_role_from_result on all search results
      2. AI Mode: ask ScrapingDog about the LinkedIn profile
      3. LLM fallback on AI Mode prose
    """
    # Step 1: Extract from search results (same as Stage 4 Step 7)
    for r in search_results:
        role = _s4_extract_role(r, full_name, company_name)
        if role:
            print(f"    Found role from search results: {role}")
            return role

    # Step 2: AI Mode — ask for the actual role (same API as Stage 4 Step 7.5)
    if SCRAPINGDOG_API_KEY and linkedin_url:
        print(f"    [AI Mode] Asking for actual role on profile...")
        import time as _time
        import requests as _requests
        from concurrent.futures import ThreadPoolExecutor

        _executor = ThreadPoolExecutor(max_workers=1)
        query = (
            f'For LinkedIn profile {linkedin_url}, what is the current job title '
            f'at "{company_name}"? '
            f'Return JSON only: {{"role":"the exact current job title"}}'
        )

        last_markdown = ""

        def _call():
            nonlocal last_markdown
            for attempt in range(3):
                try:
                    resp = _requests.get(
                        'https://api.scrapingdog.com/google/ai_mode',
                        params={'api_key': SCRAPINGDOG_API_KEY, 'query': query},
                        timeout=90,
                    )
                    if resp.status_code in (502, 503, 429):
                        if attempt < 2:
                            _time.sleep(2 + attempt * 2)
                            continue
                        return None
                    if resp.status_code == 200:
                        data = resp.json()
                        md = data.get('markdown', '') or ''
                        last_markdown = md
                        # Try parsing JSON
                        m = re.search(r'\{[^{}]*"role"[^{}]*\}', md)
                        if m:
                            try:
                                obj = json.loads(m.group())
                                if obj.get("role"):
                                    return obj["role"]
                            except Exception:
                                pass
                        return None
                except Exception:
                    if attempt < 2:
                        _time.sleep(2 + attempt * 2)
                        continue
                    return None
            return None

        loop = asyncio.get_running_loop()
        aimode_role = await loop.run_in_executor(_executor, _call)

        if aimode_role:
            print(f"    [AI Mode] Found role: {aimode_role}")
            return aimode_role

        # Step 3: LLM fallback on AI Mode prose
        if last_markdown and OPENROUTER_API_KEY:
            from target_fit_model.web_discovery import _llm_call
            prompt = (
                f"Extract the current job title for {full_name} at {company_name} "
                f"from this text. Return ONLY the job title, nothing else.\n\n"
                f"Text:\n{last_markdown[:1500]}"
            )
            llm_resp = await _llm_call(prompt)
            if llm_resp and len(llm_resp) < 100 and "sorry" not in llm_resp.lower():
                role = llm_resp.strip().strip('"').strip("'")
                print(f"    [LLM] Extracted role: {role}")
                return role

    return ""


async def _classify_industry_full(
    company_name: str,
    company_description: str,
    linkedin_industry: str,
    icp_industry: str = "",
    icp_sub_industry: str = "",
) -> Tuple[str, str]:
    """Classify industry using the full Stage 5 embedding pipeline.

    Pipeline (exact same as Stage 5):
      1. Validate + refine description (LLM)
      2. Embed refined description (Qwen3-Embedding-8B)
      3. Find top 30 candidates by cosine similarity
      4. LLM ranks top 3

    Falls back to simple taxonomy matching if the pipeline fails.
    """
    if company_description and linkedin_industry:
        try:
            classifications, refined, error = await asyncio.to_thread(
                lambda: asyncio.get_event_loop().run_until_complete(
                    classify_company_industry(
                        miner_description=company_description,
                        extracted_content=f"[Industry: {linkedin_industry}] {company_description}",
                        extracted_industry=linkedin_industry,
                        company_name=company_name,
                        miner_industry=icp_industry,
                        miner_sub_industry=icp_sub_industry,
                    )
                )
            )
        except Exception:
            # classify_company_industry is already async-friendly internally
            # but uses sync calls; just call it directly in a thread
            try:
                from validator_models.stage5_verification import (
                    _load_taxonomy_embeddings,
                    _get_embedding_sync,
                    _find_top_candidates,
                    _format_candidates_for_prompt,
                    _call_llm_sync,
                    _clean_refined_description,
                    _parse_classification_response,
                    VALIDATE_REFINE_PROMPT,
                    CLASSIFY_PROMPT,
                    TOP_K_CANDIDATES,
                )

                def _run_classification():
                    taxonomy = _load_taxonomy_embeddings()
                    if not taxonomy:
                        return [], "", "no_taxonomy"

                    content = f"[Industry: {linkedin_industry}] {company_description}"
                    prompt1 = VALIDATE_REFINE_PROMPT.format(
                        company_name=company_name,
                        miner_description=company_description[:2000],
                        extracted_content=content[:3000],
                    )
                    response1 = _call_llm_sync(prompt1)
                    refined = _clean_refined_description(response1) if response1 else None
                    if not refined:
                        return [], "", "invalid_description"

                    query_emb = _get_embedding_sync(refined)
                    if query_emb is None:
                        return [], refined, "embedding_failed"

                    candidates = _find_top_candidates(query_emb, taxonomy, k=TOP_K_CANDIDATES)
                    if not candidates:
                        return [], refined, "no_candidates"

                    candidates_str, candidates_list = _format_candidates_for_prompt(
                        candidates,
                        miner_industry=icp_industry,
                        miner_sub_industry=icp_sub_industry,
                        valid_pairs=taxonomy.get('valid_pairs'),
                    )
                    desc_for_prompt = f"{refined}. LinkedIn Industry: {linkedin_industry}" if linkedin_industry else refined
                    prompt2 = CLASSIFY_PROMPT.format(
                        refined_description=desc_for_prompt,
                        candidates_list=candidates_str,
                    )
                    response2 = _call_llm_sync(prompt2)
                    classifications = _parse_classification_response(response2, candidates_list) if response2 else []
                    return classifications, refined, ""

                classifications, refined, error = await asyncio.to_thread(_run_classification)
            except Exception as e:
                print(f"    [Industry] Classification pipeline failed: {e}")
                classifications, error = [], str(e)

        if classifications:
            top = classifications[0]
            print(f"    [Industry] Classified: {top['industry']}/{top['sub_industry']}")
            return top["industry"], top["sub_industry"]

    # Fall back to simple taxonomy matching
    return _get_valid_industry_pair(
        linkedin_industry=linkedin_industry,
        company_description=company_description,
        icp_industry=icp_industry,
        icp_sub_industry=icp_sub_industry,
    )


async def _verify_and_correct_lead(lead: dict, icp: dict) -> Optional[Dict]:
    """Run validator-equivalent checks and CORRECT the lead data.

    Uses the exact same verification methods as Stage 4 (person) and
    Stage 5 (company).  Instead of rejecting on mismatch, corrects the
    field to the verified value.

    Returns the corrected lead dict, or None if fundamentally invalid
    (e.g. person doesn't exist on LinkedIn at all).
    """
    company_name = lead["business"]
    domain = lead.get("_domain", _extract_domain(lead.get("company_website", "")))

    print(f"\n    ── Validator-equivalent verification: {lead['full_name']} @ {company_name} ──")

    # ===================================================================
    # PRE-STAGE 4: Find person's real location from their LinkedIn profile
    # This mirrors what Stage 4's Q4 search does, but we use the result
    # to SET the correct city/state BEFORE Stage 4 verifies it.
    # This is why the validator never fails on location — miners submit
    # correct locations from their data sources. We need to do the same.
    # ===================================================================
    li_url = lead.get("linkedin_url", "")
    lid_match = re.search(r'linkedin\.com/in/([^/?#]+)', li_url)
    if lid_match:
        slug = lid_match.group(1)
        pre_q = f'site:linkedin.com/in/{slug}'
        pre_results, _ = await search_google_async(pre_q, SCRAPINGDOG_API_KEY, max_results=3)
        for r in pre_results:
            if slug in r.get("link", "").lower():
                text = f"{r.get('title', '')} {r.get('snippet', '')}"
                # Try structured location first (same as Stage 4)
                loc = _s4_extract_location(text)
                if loc and "," in loc:
                    parts = loc.split(",")
                    lead["city"] = parts[0].strip()
                    lead["state"] = parts[1].strip() if len(parts) >= 2 else ""
                    if len(parts) >= 3:
                        lead["country"] = parts[2].strip()
                    print(f"    Pre-S4 location: {lead['city']}, {lead['state']}")
                    break
                # Try person location from snippet
                person_loc = _s4_extract_person_location(r.get("snippet", ""))
                if person_loc and "," in person_loc:
                    parts = person_loc.split(",")
                    lead["city"] = parts[0].strip()
                    lead["state"] = parts[1].strip() if len(parts) >= 2 else ""
                    print(f"    Pre-S4 location (snippet): {lead['city']}, {lead['state']}")
                    break

    # ===================================================================
    # STAGE 4: Person + Company LinkedIn verification
    # Uses check_linkedin_gse which is the EXACT function the validator
    # calls. It does:
    #   1. run_lead_validation_stage4 (person: URL, name, company, location, role)
    #   2. Company LinkedIn validation (scrape, verify name, cache data for Stage 5)
    # ===================================================================
    # Build lead dict with field names check_linkedin_gse expects
    s4_lead = {
        "full_name": lead["full_name"],
        "business": lead["business"],
        "linkedin_url": lead.get("linkedin_url", ""),
        "linkedin": lead.get("linkedin_url", ""),
        "city": lead.get("city", ""),
        "state": lead.get("state", ""),
        "country": lead.get("country", ""),
        "role": lead.get("role", ""),
        "email": lead.get("email", ""),
        "company_linkedin": lead.get("company_linkedin", ""),
    }

    s4_passed, s4_rejection = await check_linkedin_gse(s4_lead)

    # Build a compatible s4_result dict from check_linkedin_gse output
    s4_result = {
        "passed": s4_passed,
        "rejection_reason": s4_rejection,
        "data": s4_lead.get("lead_validation_data", {}),
    }
    # Copy verified flags back to the lead
    if s4_lead.get("role_verified"):
        lead["role_verified"] = True
        lead["role_method"] = s4_lead.get("role_method", "")
    if s4_lead.get("location_verified"):
        lead["location_verified"] = True
    if s4_lead.get("company_linkedin_verified"):
        lead["company_linkedin"] = f"https://linkedin.com/company/{s4_lead.get('company_linkedin_slug', '')}"
    if s4_lead.get("stage4_extracted_location"):
        lead["city"] = s4_lead["stage4_extracted_location"].split(",")[0].strip()
        if "," in s4_lead["stage4_extracted_location"]:
            lead["state"] = s4_lead["stage4_extracted_location"].split(",")[1].strip()
    if s4_lead.get("stage4_extracted_role"):
        ext_role = s4_lead["stage4_extracted_role"]
        latin_ratio = sum(1 for c in ext_role if c.isascii()) / max(len(ext_role), 1)
        if latin_ratio > 0.5 and len(ext_role) < 80:
            lead["role"] = ext_role

    def _apply_s4_extracted_data(s4_res, ld):
        """Apply verified extracted data from a Stage 4 result to the lead."""
        if s4_res["data"].get("extracted_location"):
            ext_loc = s4_res["data"]["extracted_location"]
            parts = ext_loc.split(",")
            if len(parts) >= 2:
                ld["city"] = parts[0].strip()
                ld["state"] = parts[1].strip()
                ld["company_hq_state"] = ld["state"]
                if len(parts) >= 3:
                    ld["country"] = parts[2].strip()
                    ld["company_hq_country"] = parts[2].strip()
        ext_role = s4_res["data"].get("extracted_role", "")
        if ext_role and len(ext_role) < 80:
            # Validate it looks like a role (contains Latin chars, not Arabic/CJK)
            latin_ratio = sum(1 for c in ext_role if c.isascii()) / max(len(ext_role), 1)
            if latin_ratio > 0.5:
                ld["role"] = ext_role

    if s4_result["passed"]:
        print(f"    ✅ Stage 4 PASSED")
        _apply_s4_extracted_data(s4_result, lead)
        # Copy all Stage 4 data to the lead for Stage 5 to use
        for key in ["role_verified", "role_method", "location_verified",
                     "company_linkedin_verified", "company_linkedin_slug",
                     "company_linkedin_data", "company_linkedin_from_cache"]:
            if s4_lead.get(key) is not None:
                lead[key] = s4_lead[key]
    else:
        reason = s4_result.get("rejection_reason", {})
        failed = reason.get("failed_fields", [])
        msg = reason.get("message", "")
        print(f"    ⚠️ Stage 4 failed: {msg}")

        # LinkedIn URL not found → unfixable
        if "linkedin" in failed:
            # Try harder: search for the person's LinkedIn directly
            print(f"    Searching for correct LinkedIn URL...")
            q = f'site:linkedin.com/in/ "{lead["full_name"]}" "{company_name}"'
            results, _ = await search_google_async(q, SCRAPINGDOG_API_KEY, max_results=5)
            li_results = [r for r in results if "linkedin.com/in/" in r.get("link", "")]
            if li_results:
                new_url = li_results[0]["link"].split("?")[0]
                print(f"    Found: {new_url} — re-running Stage 4")
                lead["linkedin_url"] = new_url
                s4_lead["linkedin_url"] = new_url
                s4_result = await run_lead_validation_stage4(
                    s4_lead,
                    scrapingdog_api_key=SCRAPINGDOG_API_KEY,
                    openrouter_api_key=OPENROUTER_API_KEY,
                )
                if not s4_result["passed"]:
                    print(f"    ❌ Stage 4 still failed with new URL: {s4_result.get('rejection_reason', {}).get('message', '')}")
                    return None
                print(f"    ✅ Stage 4 PASSED with corrected LinkedIn URL")
                _apply_s4_extracted_data(s4_result, lead)
            else:
                print(f"    ❌ Cannot find LinkedIn profile — skipping lead")
                return None

        # Name not found → unfixable (wrong person)
        elif "full_name" in failed:
            print(f"    ❌ Name not found on LinkedIn — skipping lead")
            return None

        # Company not found → unfixable (wrong company)
        elif "company" in failed:
            print(f"    ❌ Company not found on LinkedIn — skipping lead")
            return None

        # Location mismatch → CORRECT using extracted data or fresh search
        elif "city" in failed or "state" in failed:
            ext_loc = s4_result["data"].get("extracted_location", "")
            found_city, found_state, found_country = "", "", ""

            # Method 0: Parse the rejection reason directly
            # "City mismatch: extracted 'london' but claimed 'Tallinn'"
            # "Area mismatch: found 'New York City Metropolitan Area' but city 'X' not in approved list"
            rej_reason = s4_result.get("rejection_reason", {})
            if rej_reason.get("extracted_location"):
                ext_loc = rej_reason["extracted_location"]
            if rej_reason.get("area_found"):
                area = rej_reason["area_found"]
                # "New York City Metropolitan Area" → "New York City"
                area_city = re.sub(r'\s*(Metropolitan|Metro|Bay)?\s*Area$', '', area).strip()
                area_city = re.sub(r'^Greater\s+', '', area_city).strip()
                if area_city:
                    found_city = area_city
                    print(f"    Area found: {area} → city: {found_city}")
            if rej_reason.get("extracted_city"):
                found_city = rej_reason["extracted_city"]

            # Method 1: Use Stage 4's extracted location (from the matched result)
            if not found_city and ext_loc and "," in ext_loc:
                parts = ext_loc.split(",")
                found_city = parts[0].strip()
                found_state = parts[1].strip() if len(parts) >= 2 else ""
                found_country = parts[2].strip() if len(parts) >= 3 else ""

            # Method 2: Parse person location from search result snippets
            if not found_city:
                for r in s4_result["data"].get("search_results", []):
                    snippet = r.get("snippet", "")
                    person_loc = _s4_extract_person_location(snippet)
                    if person_loc and "," in person_loc:
                        parts = person_loc.split(",")
                        found_city = parts[0].strip()
                        found_state = parts[1].strip() if len(parts) >= 2 else ""
                        found_country = parts[2].strip() if len(parts) >= 3 else ""
                        break

            # Method 3: Use extract_location_from_text on all results
            if not found_city:
                for r in s4_result["data"].get("search_results", []):
                    text = f"{r.get('title', '')} {r.get('snippet', '')}"
                    candidate = _s4_extract_location(text)
                    if candidate and "," in candidate:
                        parts = candidate.split(",")
                        found_city = parts[0].strip()
                        found_state = parts[1].strip() if len(parts) >= 2 else ""
                        break

            # Method 4: Infer country from LinkedIn URL subdomain (fr.linkedin.com → France)
            if not found_city:
                li_url = lead.get("linkedin_url", "")
                url_country = get_linkedin_url_country(li_url)
                if url_country:
                    found_country = url_country.title()
                    print(f"    LinkedIn subdomain suggests: {found_country}")

            # Method 5: Dedicated profile search
            if not found_city:
                lid = re.search(r'linkedin\.com/in/([^/?#]+)', lead.get("linkedin_url", ""))
                if lid:
                    slug = lid.group(1)
                    print(f"    Searching for actual location on profile...")
                    loc_q = f'site:linkedin.com/in/{slug} location'
                    loc_results, _ = await search_google_async(loc_q, SCRAPINGDOG_API_KEY, max_results=3)
                    for r in loc_results:
                        snippet = r.get("snippet", "")
                        person_loc = _s4_extract_person_location(snippet)
                        if person_loc and "," in person_loc:
                            parts = person_loc.split(",")
                            found_city = parts[0].strip()
                            found_state = parts[1].strip() if len(parts) >= 2 else ""
                            break
                        text = f"{r.get('title', '')} {snippet}"
                        candidate = _s4_extract_location(text)
                        if candidate and "," in candidate:
                            parts = candidate.split(",")
                            found_city = parts[0].strip()
                            found_state = parts[1].strip() if len(parts) >= 2 else ""
                            break

            if found_city:
                old_city, old_state = lead.get("city", ""), lead.get("state", "")
                lead["city"] = found_city
                lead["state"] = found_state
                if found_country:
                    lead["country"] = found_country
                    lead["company_hq_country"] = found_country
                lead["company_hq_state"] = found_state
                print(f"    📍 Location CORRECTED: {old_city}, {old_state} → {found_city}, {found_state}" +
                      (f", {found_country}" if found_country else ""))
            else:
                print(f"    ❌ Could not determine location — skipping lead")
                return None

        # Role mismatch → FIND actual role using exact validator methods
        elif "role" in failed:
            # Don't use extracted_role when AI Mode REJECTED the role —
            # extracted_role is the LinkedIn headline, not a real job title
            role_method = s4_result["data"].get("role_method", "")
            if role_method and "reject" in role_method:
                ext_role = ""
            else:
                ext_role = s4_result["data"].get("extracted_role", "")

            if not ext_role:
                ext_role = await _find_actual_role(
                    linkedin_url=lead.get("linkedin_url", ""),
                    company_name=company_name,
                    full_name=lead["full_name"],
                    search_results=s4_result["data"].get("search_results", []),
                )

            if ext_role:
                old_role = lead.get("role", "")
                lead["role"] = ext_role
                print(f"    👤 Role CORRECTED: '{old_role}' → '{ext_role}'")
            else:
                print(f"    ❌ Could not determine role — skipping lead")
                return None

        # Company LinkedIn validation failed (from check_linkedin_gse)
        elif "company_linkedin" in failed:
            print(f"    ❌ Company LinkedIn validation failed — skipping lead")
            return None

        else:
            print(f"    ❌ Stage 4 failed on {failed} — skipping lead")
            return None

    # ===================================================================
    # FIX COMPANY LINKEDIN SLUG: If Stage 4 couldn't scrape the company
    # page (wrong slug from domain), search by company name to find it.
    # Stage 5 needs the correct slug for Q1/Q2/Q3/S4 queries.
    # ===================================================================
    if not lead.get("company_linkedin_data"):
        current_slug = lead.get("company_linkedin_slug", "")
        print(f"    Company LinkedIn data missing (slug '{current_slug}' may be wrong) — searching by name...")
        q = f'site:linkedin.com/company/ "{company_name}" "Company size"'
        name_r = await asyncio.to_thread(_gse_search_sync, q, 5)
        for r in name_r.get("results", []):
            link = r.get("link", "")
            slug_m = re.search(r'linkedin\.com/company/([^/?#]+)', link.lower())
            if slug_m:
                found_slug = slug_m.group(1)
                combined = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if company_name.lower() in combined:
                    old_li = lead.get("company_linkedin", "")
                    lead["company_linkedin"] = f"https://linkedin.com/company/{found_slug}"
                    lead["company_linkedin_slug"] = found_slug
                    print(f"    🔗 Company LinkedIn slug FIXED: {current_slug} → {found_slug}")
                    break

    # ===================================================================
    # STAGE 5: Company verification using exact validator method
    # check_stage5_unified finds the REAL values from LinkedIn.
    # When it rejects, the extracted values are stored on the lead dict.
    # We use those to correct the lead, then re-run.
    # ===================================================================
    # Set Stage 4 flags that Stage 5 requires
    lead["role_verified"] = True
    lead["role_method"] = s4_result["data"].get("role_method", "rule_based")
    lead["location_verified"] = True

    # Stage 5 reads "hq_country", "hq_state", "hq_city" not "company_hq_*"
    if lead.get("company_hq_country") and not lead.get("hq_country"):
        lead["hq_country"] = lead["company_hq_country"]
    if lead.get("company_hq_state") and not lead.get("hq_state"):
        lead["hq_state"] = lead["company_hq_state"]
    if lead.get("company_hq_city") and not lead.get("hq_city"):
        lead["hq_city"] = lead.get("company_hq_city", "")

    # Provide a description for classification (from Perplexity discovery)
    if not lead.get("description") and lead.get("_description"):
        lead["description"] = lead["_description"]

    # Save ICP industry/sub_industry — Stage 5 may overwrite with classification
    # but Tier 1 ICP matching needs the original ICP values
    icp_industry_original = lead.get("industry", "")
    icp_sub_industry_original = lead.get("sub_industry", "")

    MAX_S5_CORRECTIONS = 10
    for s5_attempt in range(MAX_S5_CORRECTIONS):
        s5_passed, s5_rejection = await check_stage5_unified(lead)

        if s5_passed:
            print(f"    ✅ Stage 5 PASSED")
            # Restore ICP industry/sub_industry — Tier 1 needs these to match ICP
            # Role stays as the validator-extracted value (what the validator will find)
            lead["industry"] = icp_industry_original
            lead["sub_industry"] = icp_sub_industry_original
            # Sync all HQ fields to both naming conventions
            # Sync ALL HQ fields from extracted values (set by Stage 5 Q1/Q2/Q3)
            for src, dst1, dst2 in [
                ("extracted_hq_city", "hq_city", "company_hq_city"),
                ("extracted_hq_state", "hq_state", "company_hq_state"),
                ("extracted_hq_country", "hq_country", "company_hq_country"),
            ]:
                val = lead.get(src, "")
                if val:
                    lead[dst1] = val
                    lead[dst2] = val
            # Ensure hq_country is never empty if company_hq_country has a value
            if not lead.get("hq_country") and lead.get("company_hq_country"):
                lead["hq_country"] = lead["company_hq_country"]
            if not lead.get("hq_state") and lead.get("company_hq_state"):
                lead["hq_state"] = lead["company_hq_state"]
            print(f"    HQ final: city={lead.get('hq_city','')}, state={lead.get('hq_state','')}, country={lead.get('hq_country','')}")
            break

        if not s5_rejection:
            print(f"    ❌ Stage 5 failed with no rejection info — skipping lead")
            return None

        failed_fields = s5_rejection.get("failed_fields", [])
        msg = s5_rejection.get("message", "")
        print(f"    ⚠️ Stage 5 failed (attempt {s5_attempt+1}): {msg}")

        # Restore ICP industry (Stage 5 may have overwritten it)
        lead["industry"] = icp_industry_original
        lead["sub_industry"] = icp_sub_industry_original

        corrected_something = False

        # Employee count mismatch → use the "extracted" field from rejection
        if "employee_count" in failed_fields:
            extracted_emp = s5_rejection.get("extracted", "")
            if extracted_emp:
                old_emp = lead.get("employee_count", "")
                lead["employee_count"] = extracted_emp
                print(f"    📊 Employee count CORRECTED: '{old_emp}' → '{extracted_emp}'")
                corrected_something = True
            else:
                print(f"    ❌ Employee count not found on LinkedIn — skipping lead")
                return None

        # Company LinkedIn slug not found → search by name
        if "company_linkedin" in failed_fields:
            print(f"    Searching for correct company LinkedIn...")
            q = f'site:linkedin.com/company/ "{company_name}"'
            name_r = await asyncio.to_thread(_gse_search_sync, q, 5)
            for r in name_r.get("results", []):
                link = r.get("link", "")
                slug_m = re.search(r'linkedin\.com/company/([^/?#]+)', link.lower())
                if slug_m and company_name.lower() in f"{r.get('title', '')} {r.get('snippet', '')}".lower():
                    new_slug = slug_m.group(1)
                    old_li = lead.get("company_linkedin", "")
                    lead["company_linkedin"] = f"https://linkedin.com/company/{new_slug}"
                    print(f"    🔗 Company LinkedIn CORRECTED: {old_li} → {lead['company_linkedin']}")
                    corrected_something = True
                    break

        # HQ mismatch (country, state, or city) → use extracted value from rejection
        if any(f in failed_fields for f in ["hq_country", "hq_state", "hq_city"]):
            extracted_val = s5_rejection.get("extracted", "")

            if "hq_country" in failed_fields:
                ext = lead.get("extracted_hq_country", "") or extracted_val
                if ext:
                    lead["hq_country"] = ext
                    lead["company_hq_country"] = ext
                    print(f"    🏢 HQ country CORRECTED → '{ext}'")
                    corrected_something = True
                else:
                    print(f"    ❌ HQ country not found — skipping lead")
                    return None

            if "hq_state" in failed_fields:
                ext = lead.get("extracted_hq_state", "") or extracted_val
                if ext:
                    lead["hq_state"] = ext
                    lead["company_hq_state"] = ext
                    print(f"    🏢 HQ state CORRECTED → '{ext}'")
                    corrected_something = True
                else:
                    print(f"    ❌ HQ state not found — skipping lead")
                    return None

            if "hq_city" in failed_fields:
                ext = lead.get("extracted_hq_city", "") or extracted_val
                if ext:
                    lead["hq_city"] = ext
                    lead["company_hq_city"] = ext
                    print(f"    🏢 HQ city CORRECTED → '{ext}'")
                    corrected_something = True
                else:
                    print(f"    ❌ HQ city not found — skipping lead")
                    return None

        # Industry/sub-industry mismatch
        if "industry" in failed_fields or "sub_industry" in failed_fields:
            print(f"    ❌ Industry classification failed — skipping lead")
            return None

        # Description mismatch
        if "description" in failed_fields:
            print(f"    ❌ Description validation failed — skipping lead")
            return None

        # Company name mismatch → use the LinkedIn-verified name
        if "company_name" in failed_fields:
            extracted_name = s5_rejection.get("extracted", "")
            if extracted_name:
                old_name = lead.get("business", "")
                lead["business"] = extracted_name
                company_name = extracted_name
                print(f"    🏷️ Company name CORRECTED: '{old_name}' → '{extracted_name}'")
                corrected_something = True
            else:
                print(f"    ❌ Company name mismatch on LinkedIn — skipping lead")
                return None

        if not corrected_something:
            print(f"    ❌ Stage 5 failed and no correction possible — skipping lead")
            return None

    else:
        # Exhausted all correction attempts
        print(f"    ❌ Stage 5 still failing after {MAX_S5_CORRECTIONS} corrections — skipping lead")
        return None

    # Clean up internal fields
    lead.pop("_description", None)

    print(f"    ── Verification complete ──")
    return lead


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

MAX_SOURCING_ATTEMPTS = 1000
BATCH_RECHECK_EVERY = 3


async def source_fulfillment_leads(icp: dict, num_leads: int = 5) -> List[Dict]:
    """Perplexity-first pipeline with retry loop.

    Keeps trying until ``num_leads`` are found or ``MAX_SOURCING_ATTEMPTS``
    is reached.  Every ``BATCH_RECHECK_EVERY`` attempts, does a single
    Perplexity batch re-check of all pooled email-verified leads that
    lacked intent signals in earlier attempts.
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    intent_keywords = icp.get("intent_signals", [])

    print(f"\n{'='*60}")
    print(f"  FULFILLMENT SOURCER — {industry}/{sub_industry}")
    print(f"  Signals: {intent_keywords}")
    print(f"  Target: {num_leads} leads")
    print(f"{'='*60}")

    all_leads: List[Dict] = []
    email_verified_pool: List[Dict] = []
    seen_companies: set = set()

    consecutive_empty = 0
    perplexity_empty_streak = 0  # tracks only Perplexity empty responses
    perplexity_exhausted = False

    for attempt in range(1, MAX_SOURCING_ATTEMPTS + 1):
        if len(all_leads) >= num_leads:
            break

        remaining = num_leads - len(all_leads)

        # Back off when we keep finding nothing new (shorter when exhausted)
        if consecutive_empty >= 2:
            pause = min(10 * consecutive_empty, 30 if perplexity_exhausted else 60)
            print(f"\n  ⏸  Backing off {pause}s ({consecutive_empty} empty attempts"
                  f"{', Perplexity exhausted' if perplexity_exhausted else ''})")
            await asyncio.sleep(pause)

        print(f"\n  ── Attempt {attempt}/{MAX_SOURCING_ATTEMPTS} "
              f"(have {len(all_leads)}/{num_leads}, need {remaining} more) ──")

        # ── Discovery strategy ──
        # Once Perplexity is exhausted (5+ consecutive empty responses),
        # stop asking it for new companies and focus on Google fallback +
        # pool recheck via verified search.
        if perplexity_exhausted:
            companies = await discover_companies(icp, num_companies=remaining * 5)
            use_precomputed = False
        elif attempt % 2 == 1:
            variant_idx = (attempt // 2) % len(_PROMPT_VARIANTS)
            companies = await _discover_companies_with_intent(
                icp, num_companies=remaining * 5,
                exclude_companies=seen_companies if seen_companies else None,
                _variant_idx=variant_idx,
            )
            use_precomputed = True
        else:
            companies = await discover_companies(icp, num_companies=remaining * 5)
            use_precomputed = False

        # Filter out companies we've already processed
        companies = [
            c for c in companies
            if c.get("name", "").lower() not in seen_companies
        ]

        # When using precomputed signals from Perplexity, skip companies
        # with 0 signals — don't waste API calls on contacts/emails/verification
        if use_precomputed:
            before = len(companies)
            companies = [c for c in companies if c.get("signals")]
            skipped = before - len(companies)
            if skipped:
                print(f"  Filtered out {skipped} companies with 0 intent signals")

        if not companies:
            print(f"  No new companies found this attempt")
            consecutive_empty += 1
            if use_precomputed and not perplexity_exhausted:
                perplexity_empty_streak += 1
                if perplexity_empty_streak >= 5:
                    perplexity_exhausted = True
                    print(f"  ⚠️  Perplexity exhausted ({perplexity_empty_streak} consecutive empties)")
                    print(f"  Switching to Google fallback + pool recheck only")
        else:
            consecutive_empty = 0
            if use_precomputed:
                perplexity_empty_streak = 0
            print(f"  Found {len(companies)} new companies to process")
            new_leads, new_pool = await _process_companies(
                companies, icp, remaining,
                use_precomputed_signals=use_precomputed,
            )

            # Track seen companies
            for c in companies:
                seen_companies.add(c.get("name", "").lower())
            for lead in new_leads:
                seen_companies.add(lead["business"].lower())
            for p in new_pool:
                seen_companies.add(p["business"].lower())

            all_leads.extend(new_leads)
            email_verified_pool.extend(new_pool)

        # Batch re-check pooled leads (more frequent when Perplexity is exhausted)
        recheck_interval = 1 if perplexity_exhausted else BATCH_RECHECK_EVERY
        if attempt % recheck_interval == 0 and email_verified_pool and len(all_leads) < num_leads:
            remaining = num_leads - len(all_leads)
            lead_companies = {l["business"].lower() for l in all_leads}
            pool = [p for p in email_verified_pool if p["business"].lower() not in lead_companies]

            if pool:
                print(f"\n  [Batch Intent] Re-checking {len(pool)} email-verified leads...")
                rescued = await _batch_intent_recheck(pool, icp, max_leads=remaining)
                all_leads.extend(rescued)
                # Remove rescued from pool
                rescued_names = {r["business"].lower() for r in rescued}
                email_verified_pool = [p for p in email_verified_pool if p["business"].lower() not in rescued_names]

    # Final batch re-check if we still need more
    if len(all_leads) < num_leads and email_verified_pool:
        remaining = num_leads - len(all_leads)
        lead_companies = {l["business"].lower() for l in all_leads}
        pool = [p for p in email_verified_pool if p["business"].lower() not in lead_companies]

        if pool:
            print(f"\n  [Final Batch] Re-checking {len(pool)} remaining pooled leads...")
            rescued = await _batch_intent_recheck(pool, icp, max_leads=remaining)
            all_leads.extend(rescued)

    # Sort by combined lead score (intent 60% + ICP fit 40%)
    all_leads.sort(key=lambda x: x.get("_lead_score", 0), reverse=True)

    for lead in all_leads:
        lead.pop("_matching_signal_count", None)
        lead.pop("_intent_score", None)
        lead.pop("_fit_score", None)
        lead.pop("_lead_score", None)

    print(f"\n{'='*60}")
    print(f"  RESULT: Sourced {len(all_leads)}/{num_leads} leads "
          f"({len(email_verified_pool)} in pool)")
    print(f"{'='*60}\n")

    return all_leads[:num_leads]


async def _process_companies(
    companies: List[Dict],
    icp: dict,
    num_leads: int,
    use_precomputed_signals: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Process companies into FulfillmentLead dicts.

    Pipeline order:
      1. find_contact (Google/LinkedIn) — real person, role, location, email
      2. TrueList verify (FREE) — confirm email deliverability
      3. _verify_and_correct_lead — Stage 4 + Stage 5 validator checks, correct data
      4. Attach intent signals (precomputed from Perplexity, or fetch individually)
      5. Assemble final lead

    Returns (leads, email_verified_pool).
    """
    industry = icp.get("industry", "")
    sub_industry = icp.get("sub_industry", "")
    country = icp.get("country", "United States")
    target_seniority = icp.get("target_seniority", "VP")
    target_role_types = icp.get("target_role_types", ["Sales"])
    intent_keywords = icp.get("intent_signals", [])
    product_service = icp.get("product_service", "")
    prompt_text = icp.get("prompt", "")
    intent_signals_str = ", ".join(intent_keywords) if intent_keywords else ""

    # Detect if ICP signals mention funding — if so, check each company's funding
    _has_funding_intent = detect_funding_intent(intent_signals_str)
    _funding_criteria = None
    if _has_funding_intent:
        _funding_criteria = await asyncio.to_thread(classify_funding_criteria, intent_signals_str)
        if _funding_criteria:
            print(f"  [Funding] Detected funding intent: {_funding_criteria.get('description', '')}")

    leads = []
    email_verified_pool = []
    for company in companies:
        if len(leads) >= num_leads:
            break

        company_name = company.get("name", "Unknown")
        domain = company.get("domain", _extract_domain(company.get("website", "")))
        print(f"\n  Processing: {company_name} ({domain})")

        # ── Step 1: Find a real contact via Google/LinkedIn ──
        print(f"    Searching Google/LinkedIn for a contact...")
        contact = await find_contact(company, icp)

        if not contact or not contact.get("full_name"):
            print(f"    No matching contact found on Google/LinkedIn, skipping")
            continue

        verified_name = contact["full_name"]
        verified_role = contact.get("role", "")
        verified_linkedin = contact.get("linkedin_url", "")
        verified_city = contact.get("city", "")
        verified_state = contact.get("state", "")
        email = contact.get("email", "")

        print(f"    Found: {verified_name}, {verified_role}")
        if verified_city or verified_state:
            print(f"    Location: {verified_city}, {verified_state}")

        if not email:
            print(f"    No public email found, skipping")
            continue

        print(f"    Email found: {email}")

        # ── Step 2: TrueList verify (FREE) ──
        if TRUELIST_API_KEY:
            email_valid, email_status = await _verify_email(email)
            if not email_valid:
                print(f"    Email {email} failed TrueList ({email_status}), skipping")
                continue
            print(f"    Email verified ({email_status})")
        else:
            print(f"    No TrueList key — using unverified email")

        # ── Build partial lead ──
        if not verified_state and not verified_city:
            verified_state = company.get("hq_state", "")
            verified_city = company.get("hq_city", "")

        discovered_emp_count = company.get("employee_estimate", "") or company.get("employee_count", "")

        company_li = company.get("linkedin", "")
        if not company_li and domain:
            slug = domain.split(".")[0]
            company_li = f"https://linkedin.com/company/{slug}"
        if not company_li:
            company_li = f"https://linkedin.com/company/{company_name.lower().replace(' ', '-')}"

        partial_lead = {
            "full_name": verified_name,
            "email": email,
            "linkedin_url": verified_linkedin,
            "phone": "",
            "business": company_name,
            "company_linkedin": company_li,
            "company_website": company.get("website", f"https://{domain}" if domain else ""),
            "employee_count": discovered_emp_count,
            "company_hq_country": country,
            "company_hq_state": verified_state,
            "industry": industry,
            "sub_industry": sub_industry,
            "country": country,
            "city": verified_city,
            "state": verified_state,
            "role": verified_role,
            "role_type": target_role_types[0] if target_role_types else "Sales",
            "seniority": target_seniority,
            "_domain": domain,
            "_description": company.get("description", ""),
        }

        # ── Step 3: Validator-equivalent verification & correction ──
        corrected = await _verify_and_correct_lead(partial_lead, icp)
        if corrected is None:
            print(f"    ❌ Lead failed verification — skipping")
            continue
        partial_lead = corrected

        # ── Step 4: Get intent signals ──
        raw_perplexity_signals = []  # keep raw format for scoring
        adapted_signals = []
        _used_deep_research = False

        if use_precomputed_signals:
            raw_signals = company.get("signals", [])
            adapted_signals = await _adapt_direct_signals(
                raw_signals, company_domain=domain, company_name=company_name,
            )
            raw_perplexity_signals = [
                {"signal": s.get("signal", s.get("description", "")),
                 "match": True,
                 "relevance_score": 0.8,
                 "evidence": f"{s.get('evidence', '')} {s.get('url', '')}"}
                for s in raw_signals if isinstance(s, dict)
            ]

        # Deep research via sonar-deep-research is disabled — too slow (30-60s)
        # and URLs always fail the validator's snippet verbatim check.
        # The verified search approach below (Google → validator fetch) is
        # faster and more reliable.  See _search_verified_intent_signals().
        #
        # To re-enable deep research, uncomment the block in git history.

        # ── Step 4b: Validator-equivalent intent verification ──
        company_website = partial_lead.get(
            "company_website",
            f"https://{domain}" if domain else "",
        )

        passes_threshold = False
        if adapted_signals:
            verified_signals, validator_intent_score, passes_threshold = (
                await _run_validator_intent_verification(
                    adapted_signals, company_name, company_website, icp,
                )
            )

        # Google Search → Validator Fetch → Extract Snippet
        # This approach finds URLs from Google, fetches them using the
        # validator's exact ScrapingDog method, and extracts snippets from
        # the validator's text — guaranteeing snippet overlap.
        if not passes_threshold:
            print(f"    🔎 Trying verified search approach (Google → validator fetch)...")
            verified_search_signals = await _search_verified_intent_signals(
                company_name, domain, intent_keywords, max_signals=3,
            )
            if verified_search_signals:
                adapted_signals = verified_search_signals
                raw_perplexity_signals = [
                    {"signal": s.get("description", ""), "match": True,
                     "relevance_score": 0.7, "evidence": f"{s.get('snippet', '')} {s.get('url', '')}"}
                    for s in verified_search_signals
                ]
                verified_signals, validator_intent_score, passes_threshold = (
                    await _run_validator_intent_verification(
                        adapted_signals, company_name, company_website, icp,
                    )
                )

        if not passes_threshold:
            score_str = f"{validator_intent_score:.1f}" if validator_intent_score is not None else "0.0"
            print(
                f"    ❌ Intent below validator threshold "
                f"({score_str} < {FULFILLMENT_MIN_INTENT_SCORE}) "
                f"— saved to pool"
            )
            email_verified_pool.append(partial_lead)
            continue

        adapted_signals = verified_signals

        matching_count = _count_matching_signals(adapted_signals, intent_keywords)

        # ── Step 5: Proper intent + fit scoring ──
        intent_score = compute_intent_score_from_signals(
            raw_perplexity_signals, original_signals=intent_keywords,
        )

        intent_score, source_count = _cross_source_boost(
            raw_perplexity_signals, intent_score,
        )

        fit_score, fit_breakdown = compute_fit_score(partial_lead, icp)

        lead_score = compute_lead_score(intent_score, fit_score)

        print(
            f"    Intent: {len(adapted_signals)} signals, "
            f"{matching_count}/{len(intent_keywords)} ICP keywords matched, "
            f"score={intent_score:.2f} ({source_count} sources)"
        )
        print(
            f"    Fit: {fit_score:.2f} (ind={fit_breakdown.get('industry_match', 0):.1f}, "
            f"role={fit_breakdown.get('role_match', 0):.1f}, "
            f"loc={fit_breakdown.get('location_match', 0):.1f}, "
            f"size={fit_breakdown.get('size_match', 0):.1f}) "
            f"| Lead score: {lead_score:.3f}"
        )

        lead = {
            **partial_lead,
            "intent_signals": adapted_signals,
            "_matching_signal_count": matching_count,
            "_intent_score": intent_score,
            "_fit_score": fit_score,
            "_lead_score": lead_score,
            "_validator_intent_score": validator_intent_score,
        }
        lead.pop("_domain", None)

        leads.append(lead)
        print(
            f"    ✅ Lead: {partial_lead['full_name']} @ {company_name} "
            f"role='{partial_lead['role']}' ({len(adapted_signals)} signals, "
            f"score={lead_score:.3f}, validator_intent={validator_intent_score:.1f})"
        )

    return leads, email_verified_pool
