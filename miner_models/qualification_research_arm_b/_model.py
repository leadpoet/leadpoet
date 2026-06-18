"""Generic evidence-first lead qualification model — no per-ICP hardcoding.

Pipeline (identical for every ICP):
  1. DISCOVERY (parallel) — find candidate companies by:
     a) Exa news search filtered to recent press releases (date-bound, primary sources)
     b) Sonar (Perplexity) exclusion-driven 3-pass — each pass excludes companies
        from previous passes, forcing it to surface different events.
     Each candidate is (company, date, headline) extracted via a strict LLM filter
     that drops articles whose headline doesn't match the ICP intent activity.

  2. ANCHOR LOOKUP — for each candidate, ask Sonar for website/LinkedIn/description.
     Descriptions are framed in the ICP's industry language to reduce the production
     scorer's semantic-mapping work.

  3. URL SEARCH — Exa keyword search biased to two paths:
       site:<company-domain> {headline}            (the company's own newsroom)
       {company} {headline} site:businesswire.com OR site:prnewswire.com OR ...

  4. VERIFY — production 3-stage verifier (`verify_three_stage`):
       Stage 1: Sonar independent corroboration of the claim
       Stage 2: ScrapingDog scrape of the URL (with cascade + Wayback fallback)
       Stage 3: Sonar-pro judges the scraped content
     Throttled with a semaphore (8 concurrent) to avoid OpenRouter degradation.

  5. SCORE — production `score_company()` evaluates company + intent signal
     and returns a final score (0-100). The highest-scoring URL per company is kept.

Usage:
    # All 20 ICPs from fresh_icps_20260525.json
    python3 generic_model.py

    # Subset
    python3 generic_model.py icp_20260525_009 icp_20260525_011

Environment (loaded from {repo_root}/.env or system env):
    OPENROUTER_API_KEY (or OPENROUTER_KEY)
    EXA_API_KEY
    SCRAPINGDOG_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

# ─────────────────────────────────────────────────────────────────────────────
# Path + env setup
# ─────────────────────────────────────────────────────────────────────────────
# This module lives under miner_models/qualification_research_arm_b/_model.py inside
# the leadpoet repo. Production scoring modules (qualification.scoring.*,
# gateway.qualification.*) are siblings under the repo root, so we add the
# repo root to sys.path to make imports work whether the script is run as a
# module or invoked from a sandbox harness.

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# Alias OpenRouter key variants so any caller (model or production scorer) sees the same value.
_master_key = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENROUTER_KEY")
    or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
)
if _master_key:
    for _alias in ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "QUALIFICATION_OPENROUTER_API_KEY"):
        os.environ[_alias] = _master_key

sys.path.insert(0, str(REPO_ROOT))

import httpx

from gateway.qualification.models import (
    CompanyOutput as ProdCompanyOutput,
    ICPPrompt as ProdICPPrompt,
)
from qualification.scoring.intent_verification_three_stage import (
    _scrape_sd_hardened,
    verify_three_stage,
)
from qualification.scoring.lead_scorer import score_company


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight stdout logger — `log()` flushes immediately so monitoring tails
# see per-step progress as it happens. Not using `logging` to keep deps minimal
# and output deterministic across runs.
# ─────────────────────────────────────────────────────────────────────────────

def log(*args, **kwargs):
    """Flushed print — for live progress monitoring."""
    print(*args, **kwargs, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Time-window policy.
#   Hiring intents = 180-day window (production Stage 3 enforces active-state).
#   All other intents (funding, acquisition, product launch, expansion, etc.)
#   = 365-day window.
# ─────────────────────────────────────────────────────────────────────────────

HIRING_HINTS = (
    "hiring", "open positions", "expanding the team", "actively recruiting",
)


def time_window_days_for_intent(intent_text: str) -> int:
    """Returns the freshness window in days for a given intent signal."""
    low = (intent_text or "").lower()
    if any(hint in low for hint in HIRING_HINTS):
        return 180
    return 365


def today_iso() -> str:
    """Today's date as YYYY-MM-DD (dynamic — never hardcode)."""
    return date.today().isoformat()


def cutoff_iso(window_days: int) -> str:
    """Earliest acceptable event date for a given freshness window."""
    return (date.today() - timedelta(days=window_days)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Company-name normalization.
#
# Sonar tends to return formal/legal names ("Mastercard Incorporated",
# "JPMorgan Chase & Co.") while press releases use canonical brand names
# ("Mastercard", "JPMorgan Chase"). The production scorer's fabrication check
# matches the submitted name against page text, so a suffix mismatch can flag
# a real lead as fabricated. Strip well-known legal-entity designators.
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_SUFFIX_RE = re.compile(
    r"\s*[,.]?\s+("
    # "& Company" / "and Company" must precede plain "Company" to avoid
    # leaving a dangling "&" or "and".
    r"&\s*Company|and\s+Company|"
    # Long forms
    r"Incorporated|Corporation|Limited|Company|"
    # English abbreviations
    r"Inc\.?|Corp\.?|Ltd\.?|Co\.?|"
    # English LLC / LLP / LP / PLC (with or without periods)
    r"L\.?L\.?C\.?|L\.?L\.?P\.?|L\.?P\.?|P\.?L\.?C\.?|"
    # Public benefit corporation
    r"PBC|P\.B\.C\.|"
    # German / continental Europe
    r"GmbH|AG|A\.G\.|SE|S\.E\.|"
    # Romance languages
    r"S\.?A\.?S\.?|S\.?A\.?|S\.?L\.?|S\.?R\.?L\.?|"
    # Netherlands / Belgium
    r"N\.?V\.?|B\.?V\.?|"
    # Nordics
    r"A\.?B\.?|A\.?S\.?A\.?|A\.?S\.?|O\.?Y\.?|OYJ|"
    # India / Asia
    r"Pvt\.?\s+Ltd\.?|Pte\.?\s+Ltd\.?|Pty\.?\s+Ltd\.?|"
    # Trailing "& Co." / "and Co."
    r"&\s*Co\.?|and\s+Co\.?"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)


def normalize_company_name(name: str) -> str:
    """Iteratively strip trailing legal-entity suffixes.

    'Mastercard Incorporated' → 'Mastercard'
    'Goldman Sachs Group, Inc.' → 'Goldman Sachs Group'  (keeps 'Group' — brand token)
    'Foo, Inc., LLC' → 'Foo'  (handles compounded suffixes)
    """
    if not name:
        return ""
    s = name.strip().strip('"').strip("'").strip()
    for _ in range(5):
        new_s = _LEGAL_SUFFIX_RE.sub("", s).strip().rstrip(",").strip()
        if new_s == s or not new_s:
            break
        s = new_s
    # Defensive: drop dangling connectors / punctuation
    s = re.sub(r"\s*(?:&|and|,|\.|\s)+$", "", s).strip()
    return s or name.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Sonar (Perplexity) primitive.
# ─────────────────────────────────────────────────────────────────────────────

async def call_sonar(
    client: httpx.AsyncClient,
    prompt: str,
    model: str = "perplexity/sonar-pro",
    timeout: float = 90.0,
) -> str:
    """POST a prompt to Sonar via OpenRouter. Returns the raw text response.

    Retries once after a 1-second backoff on transient errors (5xx, 429,
    network/timeout). Returns "" on persistent failure rather than raising —
    downstream code already handles empty/invalid JSON responses gracefully.
    """
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"}

    last_err: Exception | None = None
    for attempt in (0, 1):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            if resp.status_code >= 500 or resp.status_code == 429:
                # Transient — retry once.
                last_err = httpx.HTTPStatusError(
                    f"upstream {resp.status_code}", request=resp.request, response=resp,
                )
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                return ""
            if resp.status_code != 200:
                # Non-retryable client error (401, 403, 400) — give up quietly.
                return ""
            data = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_err = e
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return ""
        except Exception:
            return ""

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError):
            return ""
        # Strip code fences if Sonar wrapped the JSON
        if content.startswith("```"):
            content = content.split("```", 2)[1].lstrip("json").strip()
        return content

    # Defensive — loop guarantees a return, but make the type-checker happy.
    return ""


# Sonar sometimes appends citation markers like  "value"[3],  after JSON
# string-closing quotes — invalid JSON. Strip them before parsing.
_CITATION_RE = re.compile(r'(?<=")\s*\[\d+\](?=\s*[,}\]])')


def _strip_sonar_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def parse_sonar_json(text: str) -> dict:
    """Best-effort JSON parse on a Sonar response. Strips citation markers first."""
    cleaned = _strip_sonar_citations(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Exa primitives.
# ─────────────────────────────────────────────────────────────────────────────

async def exa_keyword_search(
    client: httpx.AsyncClient,
    query: str,
    n: int = 3,
) -> list[str]:
    """Exa keyword search → list of URLs. Used for per-event URL discovery."""
    body = {"query": query, "numResults": n, "type": "keyword"}
    try:
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": os.environ.get("EXA_API_KEY", "")},
            json=body,
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        return [
            r.get("url", "")
            for r in resp.json().get("results", [])
            if r.get("url")
        ]
    except Exception:
        return []


async def exa_news_search(
    client: httpx.AsyncClient,
    query: str,
    days_back: int,
    n: int = 10,
) -> list[dict]:
    """Exa news search → list of {url, title, date}.

    Exa's `startPublishedDate` filter isn't strictly enforced on neural search
    (returns popular older articles), so we over-fetch and client-side filter
    by both `publishedDate` and any obvious year in the URL path.
    """
    cutoff = date.today() - timedelta(days=days_back)
    body = {
        "query": query,
        "numResults": n * 2,
        "type": "neural",
        "category": "news",
        "startPublishedDate": cutoff.isoformat(),
    }
    try:
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": os.environ.get("EXA_API_KEY", "")},
            json=body,
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
    except Exception:
        return []

    out: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if not url:
            continue
        pubdate_str = (r.get("publishedDate") or "")[:10]
        keep = True
        # Filter 1: structured publishedDate older than cutoff
        if pubdate_str:
            try:
                if date.fromisoformat(pubdate_str) < cutoff:
                    keep = False
            except Exception:
                pass
        # Filter 2: year appearing in URL path older than cutoff year
        year_match = re.search(r"/(20\d{2})/", url)
        if year_match and int(year_match.group(1)) < cutoff.year:
            keep = False
        if not keep:
            continue
        out.append({
            "url": url,
            "title": (r.get("title") or "")[:300],
            "date": pubdate_str,
        })
        if len(out) >= n:
            break
    return out


def _domain_from_website_url(website: str) -> str:
    """'https://www.mastercard.com/about' → 'mastercard.com' """
    if not website:
        return ""
    try:
        host = urlparse(website).hostname or website
    except Exception:
        host = website
    return host.lower().lstrip(".").removeprefix("www.")


# Host substring → IntentSignalSource enum value.  Picks the source the
# production scorer will multiply by the highest weight (linkedin/job_board/
# github=1.0 > news=0.9 > company_website=0.85 > other=0.5).
_HOST_SOURCE_HINTS = (
    # Most specific patterns first.
    ("linkedin.com/jobs", "job_board"),
    ("linkedin.com/posts", "linkedin"),
    ("linkedin.com", "linkedin"),
    ("indeed.com", "job_board"),
    ("glassdoor.com", "job_board"),
    ("greenhouse.io", "job_board"),
    ("lever.co", "job_board"),
    ("ashbyhq.com", "job_board"),
    ("workable.com", "job_board"),
    ("github.com", "github"),
    ("g2.com", "review_site"),
    ("trustpilot.com", "review_site"),
    ("capterra.com", "review_site"),
    ("wikipedia.org", "wikipedia"),
    ("twitter.com", "social_media"),
    ("x.com", "social_media"),
    ("facebook.com", "social_media"),
    ("businesswire.com", "news"),
    ("prnewswire.com", "news"),
    ("globenewswire.com", "news"),
    ("techcrunch.com", "news"),
    ("reuters.com", "news"),
    ("bloomberg.com", "news"),
    ("venturebeat.com", "news"),
    ("axios.com", "news"),
    ("wsj.com", "news"),
    ("ft.com", "news"),
    ("forbes.com", "news"),
    ("cnbc.com", "news"),
)


def _classify_intent_signal_source(signal_url: str, company_website: str) -> str:
    """Map the signal URL host to the IntentSignalSource enum value that the
    production scorer applies the highest weight to.  Falls back to
    'company_website' when the host matches the company domain, else 'news'
    for unrecognized 3rd-party hosts (slightly higher weight than 'other').
    """
    if not signal_url:
        return "company_website"
    try:
        parsed = urlparse(signal_url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        path = (parsed.path or "").lower()
    except Exception:
        return "company_website"
    if not host:
        return "company_website"
    # Match against host+path so "linkedin.com/jobs" beats "linkedin.com".
    haystack = host + path
    for hint, src in _HOST_SOURCE_HINTS:
        if hint in haystack:
            return src
    company_host = _domain_from_website_url(company_website)
    if company_host and (host == company_host or host.endswith("." + company_host)):
        return "company_website"
    # Unknown 3rd-party domain — prefer 'news' (0.9 multiplier) over 'other' (0.5).
    return "news"


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY — event-first via Exa news + extraction.
# ─────────────────────────────────────────────────────────────────────────────

async def extract_event_from_article(
    client: httpx.AsyncClient,
    article: dict,
    intent: str,
) -> dict | None:
    """Given an Exa news article {url, title, date}, ask Sonar whether the
    headline describes EXACTLY the intent activity by a single named company.

    Returns {company, date, headline} or None if the article should be skipped
    (listicle, intent mismatch, ambiguous subject, etc.).
    """
    prompt = f"""Read this news article title and decide whether the article describes a SPECIFIC NAMED COMPANY performing EXACTLY the following intent activity:

INTENT activity: "{intent}"
TITLE: "{article.get('title', '')}"
URL:   "{article.get('url', '')}"
DATE:  "{article.get('date', '')}"

STRICT MATCH RULES — output {{"skip":true}} for ANY of:
- Title is a listicle / weekly roundup / "top X startups" / industry report (multi-company)
- Title describes a DIFFERENT activity than the intent (e.g., intent="Just closed a round" but title is about a product launch / acquisition / partnership / hiring — REJECT)
- Title is about a company being acquired, but intent is about acquiring (reverse direction — REJECT)
- Intent is "Acquired another company" but title is a partnership / strategic investment / joint venture
- Intent is "Launched a new product" but title is about funding / partnership / customer adoption
- Intent is "Just closed a round" but title is about product release / acquisition / customer story
- Intent is "Expanded to new markets" but title is about funding / product / partnership only
- Subject company can't be unambiguously identified
- Same company name could refer to several different real companies and the title doesn't disambiguate

If the title clearly describes the named subject company doing EXACTLY the intent activity, output:
{{"company":"<canonical company name — the SUBJECT performing the action>","date":"YYYY-MM-DD","headline":"<one sentence reformulating the event>"}}

Output JSON only, nothing else."""

    raw = await call_sonar(client, prompt, model="perplexity/sonar", timeout=30)
    parsed = parse_sonar_json(raw)
    if not parsed or parsed.get("skip") or not parsed.get("company"):
        return None
    return {
        "company": normalize_company_name(parsed.get("company") or ""),
        "date": parsed.get("date") or article.get("date") or "",
        "headline": parsed.get("headline") or article.get("title") or "",
    }


async def discover_events_via_news(
    client: httpx.AsyncClient,
    icp: dict,
    n_articles: int = 10,
) -> list[dict]:
    """Discovery from real-time news indices (Exa news), independent of Sonar.

    Returns deduped events (by company name) after strict LLM intent-matching.
    """
    window_days = time_window_days_for_intent(icp["intent_signals"][0])
    industry = icp.get("industry", "")
    intent = icp["intent_signals"][0]
    geo = icp.get("geography") or icp.get("country", "")
    stage = icp.get("company_stage", "")
    base_query = f"{industry} {stage} {geo} {intent}".strip()

    # Single source (Exa news) — SD Google reserved for later expansion.
    articles = await exa_news_search(client, base_query, days_back=window_days, n=n_articles)
    log(f"  [discovery] Exa news returned {len(articles)} articles")

    if not articles:
        return []

    # Extract (company, date, headline) per article via strict LLM filter.
    # return_exceptions=True so one bad extraction call doesn't lose all.
    extracts = await asyncio.gather(
        *[extract_event_from_article(client, art, intent) for art in articles],
        return_exceptions=True,
    )
    events = [e for e in extracts if e and not isinstance(e, BaseException)]
    log(f"  [discovery] {len(events)} events after intent-strict filtering")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY — Sonar 3-pass exclusion-driven, complementary to news search.
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic filters for Sonar's meta/refusal responses (e.g., when it can't
# find enough events). Keeps junk events out of the pool.
_BAD_COMPANY_PREFIXES = (
    "no verified", "no real", "no verifiable", "insufficient",
    "none ", "could not", "n/a", "unknown", "—", "-",
)
_BAD_HEADLINE_SUBSTRINGS = (
    "insufficient search", "could not verify", "could be confirmed",
    "no verifiable", "no data", "no qualifying events", "no events that match",
    "no events found", "no results", "no public",
)


def _filter_meta_events(events: list[dict]) -> list[dict]:
    """Drop Sonar's refusal stubs (e.g., {'company':'Unknown', 'headline':'No verifiable events...'})."""
    out = []
    for ev in events:
        name = (ev.get("company") or "").strip()
        headline = (ev.get("headline") or "").lower()
        if not name:
            continue
        if name.lower().startswith(_BAD_COMPANY_PREFIXES):
            continue
        if any(meta in headline for meta in _BAD_HEADLINE_SUBSTRINGS):
            continue
        out.append(ev)
    return out


def _build_sonar_discovery_prompt(
    n: int,
    icp: dict,
    today: str,
    cutoff: str,
    window_days: int,
    exclude_companies: list[str],
) -> str:
    buyer = icp.get("prompt") or icp.get("buyer_prompt") or ""
    intent = icp["intent_signals"][0]
    industry = icp.get("industry", "")
    stage = icp.get("company_stage", "")
    geo = icp.get("geography") or icp.get("country", "")

    exclude_clause = ""
    if exclude_companies:
        exclude_list = ", ".join(f'"{c}"' for c in exclude_companies)
        exclude_clause = (
            f"\nEXCLUDE these companies (already found by previous searches; "
            f"find DIFFERENT ones): {exclude_list}\n"
            f"Do NOT return any of the excluded companies. Find new ones."
        )

    return f"""Find up to {n} real, verifiable, recent (last {window_days} days as of {today}) events for: {buyer}
Intent: "{intent}"
Industry: {industry}
Stage: {stage}
Geography: {geo}{exclude_clause}
For each: exact company name, date (YYYY-MM-DD), one-sentence headline describing the SPECIFIC event that matches the intent.
The event MUST be the exact intent activity (not an adjacent one). For example, if intent is "Acquired another company", the event must be the company BUYING another business — not a partnership, sale, or product launch.
Only events you have HIGH confidence happened. Prefer events with date on or after {cutoff}. Do NOT include URLs.
Output JSON only: {{"events":[{{"company":"...","date":"YYYY-MM-DD","headline":"..."}}]}}"""


async def discover_events_via_sonar(
    client: httpx.AsyncClient,
    icp: dict,
    seen_companies: set[str],
    n: int = 10,
    n_passes: int = 3,
) -> list[dict]:
    """Sonar 3-pass exclusion-driven discovery.

    Each pass tells Sonar to EXCLUDE the companies found so far, forcing it to
    surface different events. Stops early if a pass returns no new companies.
    Falls back to regular `sonar` (less precise) if `sonar-pro` produced < 3
    after all passes.
    """
    window_days = time_window_days_for_intent(icp["intent_signals"][0])
    today = today_iso()
    cutoff = cutoff_iso(window_days)
    new_events: list[dict] = []

    for pass_idx in range(n_passes):
        excluded = [
            ev.get("company") for ev in new_events if ev.get("company")
        ] + list(seen_companies)
        prompt = _build_sonar_discovery_prompt(n, icp, today, cutoff, window_days, excluded)
        raw = await call_sonar(client, prompt, model="perplexity/sonar-pro")
        added_this_pass = 0
        for ev in _filter_meta_events(parse_sonar_json(raw).get("events", [])):
            key = (ev.get("company") or "").lower().strip()
            if key and key not in seen_companies and key not in {
                (e.get("company") or "").lower() for e in new_events
            }:
                new_events.append(ev)
                added_this_pass += 1
        log(f"  [discovery] sonar-pro pass {pass_idx + 1} added {added_this_pass} new")
        if added_this_pass == 0:
            break

    # Fallback: regular `sonar` model if combined Sonar+seen still < 3
    if len(new_events) + len(seen_companies) < 3:
        excluded = [
            ev.get("company") for ev in new_events if ev.get("company")
        ] + list(seen_companies)
        raw = await call_sonar(
            client,
            _build_sonar_discovery_prompt(n, icp, today, cutoff, window_days, excluded),
            model="perplexity/sonar",
        )
        for ev in _filter_meta_events(parse_sonar_json(raw).get("events", [])):
            key = (ev.get("company") or "").lower().strip()
            if key and key not in seen_companies and key not in {
                (e.get("company") or "").lower() for e in new_events
            }:
                new_events.append(ev)
        log(f"  [discovery] sonar fallback added → total {len(new_events)} from Sonar")

    return new_events


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY — merged entry point.
# ─────────────────────────────────────────────────────────────────────────────

async def discover_events(client: httpx.AsyncClient, icp: dict) -> list[dict]:
    """Top-level discovery for Arm B: Sonar first, then Exa news fill.

    This is intentionally distinct from the reference arm's Exa-first routing
    while preserving the same verification and scoring stack.
    """
    events: list[dict] = []
    seen: set[str] = set()

    # Branch A — Sonar exclusion-driven. This Arm B routing starts with the
    # broad real-time search path, then lets Exa fill gaps with news results.
    try:
        sonar_events = await discover_events_via_sonar(
            client, icp, seen_companies=seen, n=10, n_passes=3,
        )
    except Exception as _exc:
        log(f"  [discovery] Sonar-first branch raised {type(_exc).__name__}; continuing")
        sonar_events = []
    for ev in sonar_events:
        key = (ev.get("company") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            events.append(ev)
    log(f"  [discovery] Sonar-first branch yielded {len(events)} events")

    # Branch B — Exa news supplements rather than replaces.
    try:
        exa_events = await discover_events_via_news(client, icp, n_articles=10)
    except Exception as _exc:
        log(f"  [discovery] Exa-fill branch raised {type(_exc).__name__}; continuing")
        exa_events = []
    for ev in exa_events:
        key = (ev.get("company") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            events.append(ev)

    # Normalize names once; all downstream code sees canonical brand.
    for ev in events:
        ev["company"] = normalize_company_name(ev.get("company") or "")
    log(f"  [discovery] total {len(events)} unique candidate events")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# ANCHORS — for each candidate, look up website/LinkedIn/description.
# ─────────────────────────────────────────────────────────────────────────────

async def lookup_company_anchors(
    client: httpx.AsyncClient,
    company: str,
    icp_industry: str = "",
    icp_sub_industry: str = "",
) -> tuple[str, str, str]:
    """Returns (website, linkedin, description).

    Description is framed in the ICP's industry language so the production
    scorer's icp_fit LLM has less semantic work mapping company → ICP industry.
    """
    industry_phrase = (
        f"{icp_industry}/{icp_sub_industry}" if icp_sub_industry else icp_industry
    )
    desc_directive = (
        f"<1-2 sentences explaining how {company} operates within the {industry_phrase} "
        f"industry — name their primary products/services and the customer segment they serve>"
        if industry_phrase
        else "<1 sentence>"
    )
    prompt = (
        f'For "{company}", output JSON only: '
        f'{{"website":"<https URL>","linkedin":"<linkedin URL>",'
        f'"description":"{desc_directive}"}}'
    )
    raw = await call_sonar(client, prompt, model="perplexity/sonar", timeout=30)
    parsed = parse_sonar_json(raw)
    return (
        parsed.get("website", ""),
        parsed.get("linkedin", ""),
        parsed.get("description", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# URL SEARCH — find primary-source URLs for each event.
# ─────────────────────────────────────────────────────────────────────────────

async def search_urls_for_event(
    client: httpx.AsyncClient,
    event: dict,
    company_website: str,
    n_per_query: int = 3,
) -> list[str]:
    """Two parallel Exa queries (merged unique):
        1. site:<company-domain> {headline}  →  company's own newsroom
        2. {company} {headline} site:(BW|PRN|GN)  →  press wires
    """
    company = event.get("company") or ""
    headline = event.get("headline") or ""
    domain = _domain_from_website_url(company_website)

    queries: list[str] = []
    if domain:
        queries.append(f"site:{domain} {headline}")
    queries.append(
        f"{company} {headline} "
        f"site:businesswire.com OR site:prnewswire.com OR site:globenewswire.com"
    )

    results = await asyncio.gather(*[
        exa_keyword_search(client, q, n=n_per_query) for q in queries
    ])
    seen: set[str] = set()
    merged: list[str] = []
    for urls in results:
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY + SCORE — production 3-stage verifier + production scorer.
# ─────────────────────────────────────────────────────────────────────────────

async def verify_and_score_lead(
    client: httpx.AsyncClient,
    event: dict,
    url: str,
    anchors: tuple[str, str, str],
    icp: dict,
) -> dict | None:
    """Run the production verifier on (event, URL). If client_ready, build the
    submission dict and call production `score_company()`. Returns the scored
    result or None.
    """
    website, linkedin, description = anchors
    intent = icp["intent_signals"][0]

    # Pre-scrape to ensure URL is real and contains the company name
    scrape = await _scrape_sd_hardened(url)
    if not (scrape.get("ok") and scrape.get("content")):
        return None
    content = scrape.get("content") or ""
    if (event.get("company") or "").lower() not in content.lower():
        return None

    # Production 3-stage verifier
    try:
        verdict = await verify_three_stage(
            client,
            company_name=event.get("company") or "",
            company_linkedin=linkedin,
            company_website=website,
            source_url=url,
            miner_claim=event.get("headline") or "",
            target_signal_text=intent,
        )
    except Exception:
        return None
    if not verdict.get("client_ready"):
        return None

    # Build submission for production scorer
    fallback_website = (
        f"https://{(event.get('company') or '').lower().replace(' ', '').replace(',', '').replace('.', '')}.com"
    )
    submission = {
        "company_name": event.get("company"),
        "company_website": website or fallback_website,
        "company_linkedin": linkedin,
        "industry": icp.get("industry") or "Other",
        "sub_industry": icp.get("sub_industry") or "",
        "employee_count": icp.get("employee_count") or "1-50",
        "company_stage": icp.get("company_stage") or "",
        "country": icp.get("country") or "United States",
        "state": "",
        "description": (description or f"{event.get('company')} is a company.")[:500],
        "intent_signals": [{
            "source": _classify_intent_signal_source(url, website),
            "description": (event.get("headline") or "")[:350],
            "url": url,
            "date": event.get("date"),
            "snippet": (content or event.get("headline") or "")[:600],
            "matched_icp_signal": 0,
        }],
    }
    try:
        prod_company = ProdCompanyOutput.model_validate(submission)
    except Exception:
        return None

    prod_icp = ProdICPPrompt(
        icp_id=icp["icp_id"],
        prompt=icp.get("prompt") or "",
        industry=icp.get("industry") or "Other",
        sub_industry=icp.get("sub_industry") or "",
        target_roles=[],
        target_seniority="",
        employee_count=icp.get("employee_count") or "1-50",
        company_stage=icp.get("company_stage") or "",
        geography=icp.get("geography") or "United States",
        country=icp.get("country") or "United States",
        product_service=icp.get("product_service") or "",
        intent_signals=icp["intent_signals"],
    )
    try:
        breakdown = await score_company(
            company=prod_company,
            icp=prod_icp,
            # This model is the reference / baseline — `is_reference_model=True`
            # skips the cost/time variability penalty inside score_company so our
            # internal top-5 ranking reflects the same no-penalty scoring path
            # the validator will use when scoring the baseline run externally.
            run_cost_usd=0.0,
            run_time_seconds=0.0,
            seen_companies=set(),
            force_fail_reason=None,
            is_reference_model=True,
        )
    except Exception:
        # score_company exceptions (network blip, LLM error, etc.) must not
        # propagate — one bad scoring call shouldn't kill the whole ICP.
        return None
    return {
        "company": event.get("company"),
        "url": url,
        "date": event.get("date"),
        "headline": event.get("headline"),
        "score": breakdown.final_score,
        "icp_fit": breakdown.icp_fit,
        "intent": breakdown.intent_signal_final,
        "failure_reason": breakdown.failure_reason,
        # CompanyOutput-shaped submission (what the validator scores)
        "submission": submission,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-ICP ORCHESTRATION.
# ─────────────────────────────────────────────────────────────────────────────

# Cap concurrent verifier calls per ICP. The verifier calls OpenRouter multiple
# times per pair (Stage 1 + Stage 3); too many concurrent calls cause degraded
# responses that the verifier interprets as "wrong_entity" rejections.
MAX_CONCURRENT_VERIFIES = 8


async def qualify_icp(client: httpx.AsyncClient, icp: dict) -> list[dict]:
    """Discover candidates → look up anchors → search URLs → verify + score.

    Returns the deduped list of verified leads (sorted desc by score).
    """
    t_start = time.time()
    events = await discover_events(client, icp)
    log(f"  [orchestrate] {len(events)} candidate events in {time.time() - t_start:.1f}s")
    if not events:
        return []

    # Anchor lookup first — we need each company's domain for site-biased
    # URL search in the next step. return_exceptions so a failed lookup
    # falls back to empty anchors rather than killing the ICP.
    anchors_raw = await asyncio.gather(
        *[
            lookup_company_anchors(
                client,
                ev.get("company") or "",
                icp_industry=icp.get("industry", ""),
                icp_sub_industry=icp.get("sub_industry", ""),
            )
            for ev in events
        ],
        return_exceptions=True,
    )
    anchors_per_event = [
        a if not isinstance(a, BaseException) else ("", "", "")
        for a in anchors_raw
    ]
    # Per-event URL search
    urls_raw = await asyncio.gather(
        *[
            search_urls_for_event(client, ev, anch[0], n_per_query=3)
            for ev, anch in zip(events, anchors_per_event)
        ],
        return_exceptions=True,
    )
    urls_per_event = [
        u if not isinstance(u, BaseException) else []
        for u in urls_raw
    ]

    # Flatten to (event, url, anchors) tuples
    pairs: list[tuple[dict, str, tuple[str, str, str]]] = []
    for ev, urls, anch in zip(events, urls_per_event, anchors_per_event):
        for url in urls:
            pairs.append((ev, url, anch))
    log(
        f"  [orchestrate] verifying {len(pairs)} (event, url) pairs "
        f"(throttled to {MAX_CONCURRENT_VERIFIES} concurrent)"
    )

    verify_sem = asyncio.Semaphore(MAX_CONCURRENT_VERIFIES)

    async def _verify_one(ev, url, anch):
        async with verify_sem:
            return await verify_and_score_lead(client, ev, url, anch, icp)

    # return_exceptions=True so one task crash doesn't cancel siblings —
    # the validator must not see us return [] for an ICP just because one
    # of N verifier calls hit an edge case.
    results = await asyncio.gather(
        *[_verify_one(ev, url, anch) for (ev, url, anch) in pairs],
        return_exceptions=True,
    )

    # Dedupe by company name; keep the highest-scoring URL per company.
    by_company: dict[str, dict] = {}
    for r in results:
        if r is None or isinstance(r, BaseException):
            continue
        key = (r["company"] or "").lower()
        if key not in by_company or r["score"] > by_company[key]["score"]:
            by_company[key] = r

    leads = sorted(by_company.values(), key=lambda x: -x["score"])
    log(
        f"  [orchestrate] {len(leads)} verified leads "
        f"(from {len(pairs)} pairs) in {time.time() - t_start:.1f}s"
    )
    return leads


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT.
# ─────────────────────────────────────────────────────────────────────────────

ICP_DATASET_PATH = Path(__file__).parent / "fresh_icps_20260525.json"


async def main():
    icps = json.loads(ICP_DATASET_PATH.read_text())
    by_id = {i["icp_id"]: i for i in icps}
    targets = sys.argv[1:] if len(sys.argv) > 1 else [i["icp_id"] for i in icps]

    async with httpx.AsyncClient() as client:
        all_results: dict[str, list] = {}
        t_total = time.time()

        for icp_id in targets:
            if icp_id not in by_id:
                log(f"\n[SKIP] unknown ICP {icp_id}")
                continue
            icp = by_id[icp_id]
            log(
                f"\n[START] {icp_id}  "
                f"{icp.get('industry')} / {icp['intent_signals'][0]}"
            )
            leads = await qualify_icp(client, icp)
            all_results[icp_id] = leads
            log(f"\n[DONE]  {icp_id}  leads={len(leads)}")
            for j, lead in enumerate(leads, 1):
                log(
                    f"  [{j}] score={lead['score']:5.1f}  "
                    f"{lead['company'][:30]:30}  "
                    f"{lead['date']:10}  "
                    f"{lead['url'][:70]}"
                )

        # ─── SUMMARY ──────────────────────────────────────────────────
        log(f"\n{'=' * 100}")
        log("GENERIC MODEL — SUMMARY")
        log(f"{'=' * 100}")
        total_sum = 0.0
        top_score_sum = 0.0
        for icp_id in targets:
            leads = all_results.get(icp_id, [])
            sum_score = sum(l["score"] for l in leads)
            top_score = max((l["score"] for l in leads), default=0.0)
            log(
                f"  {icp_id}  "
                f"leads={len(leads)}  "
                f"top_score={top_score:5.1f}  "
                f"sum={sum_score:6.1f}"
            )
            total_sum += sum_score
            top_score_sum += top_score
        n = max(len(targets), 1)
        log(f"\n  Total leads-sum across {len(targets)} ICPs : {total_sum:.1f}")
        log(f"  Sum of top scores per ICP             : {top_score_sum:.1f}")
        log(f"  Avg top score per ICP (competition)   : {top_score_sum / n:.1f}")
        log(f"  Wall time                             : {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
