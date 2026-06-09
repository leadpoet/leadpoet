"""Intent verification — 3-stage sonar -> (SD/Exa scrape) -> sonar-pro pipeline.

Production port of Intent_check/pipeline_sonar_exa_contents.py. Identical
prompts, models, JSON schema, guardrails, and decision rule. The ONLY change
versus the standalone .py file is Stage 2: instead of Exa-Contents-only
extraction, we use the current production scraping flow (Scrapingdog primary
with host-aware hardening + Exa fallback per URL).

Activated via INTENT_VERIFIER_THREE_STAGE env flag.

What the pipeline does, in order:

  STAGE 1 — Sonar first-pass (no pre-scraping; sonar uses native web search).
    Call perplexity/sonar with the build_verification_prompt prompt from the
    standalone pipeline.  The model decides supported / partially_supported /
    contradicted / wrong_entity / unable_to_verify with a confidence level.

  Decision after Stage 1 (decision() function from the standalone pipeline):
    - same_entity_check == 'fail'                   -> reject (STOP)
    - signal_status == 'supported' AND high conf    -> approve (STOP)
    - signal_status in {contradicted, wrong_entity} -> reject (STOP)
    - otherwise                                       -> review (escalate)

  STAGE 2 — only when Stage 1 returns 'review'. SD-primary + Exa-fallback per
  supplied URL. Content-driven progressive escalation (NO hardcoded host
  list):
    * Tier 1 (baseline)        — cheap default call
    * Tier 2 (dynamic+wait)    — escalate when body is empty/short/JS-shell
    * Tier 3 (premium+stealth) — escalate when anti-bot markers detected
    * Tier 4 (full combined)   — last resort: dynamic + premium + stealth
    * Wayback Machine snapshot — final fallback when all tiers exhaust
    * Per-tier timeout caps + structural detectors (HTTP status, body length,
      anti-bot markers, JS-shell hydration shape)
    * Exa fallback per URL when SD fails

  Optional pre-LLM company-name check (after Stage 2, before Stage 3):
    company_in_scrape() word-boundary regex catches obvious wrong-entity URLs
    deterministically, saving the cost of a sonar-pro call when the scraped
    text doesn't even mention the company.

  STAGE 3 — Sonar-pro final judge with the SD/Exa content.
    Call perplexity/sonar-pro with the standalone pipeline's
    build_final_judge_prompt — strict rules saying only the exact extracted
    content can support the claim.

  Apply guardrails again (supplied URL must appear in evidence_urls_used).
  Final decision: same decision() function.

  Final mapping to production binary semantics (verify_three_stage's
  client_ready):
    approve                  -> client_ready=True
    reject                   -> client_ready=False
    review                   -> client_ready=False by default; can be flipped
                                to True with INTENT_VERIFIER_REVIEW_AS_ACCEPT.

Public API: verify_three_stage() — mirrors verify_single_call()'s contract so
the caller in lead_scorer.py can swap between them via env flag.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunsplit

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Scraping config — content-driven progressive escalation (NO host lists)
# ─────────────────────────────────────────────────────────────────────
MAX_SCRAPED_CHARS = 60_000
SCRAPE_TIMEOUT = 60

# Anti-bot / login-wall / parked-page text markers. When found in a short
# response body, indicates the scraper hit a challenge page instead of real
# content — escalate to a stronger tier.
ANTI_BOT_MARKERS = [
    "checking your browser", "captcha", "verify you are human",
    "ddos protection", "challenge-platform", "access denied",
    "security check", "just a moment", "verifying you are human",
    "enable javascript", "please enable js",
    "sign in to (linkedin|see|join|view|continue)",
    "page can.?t be found", "403\\s*forbidden", "404\\s*not found",
    # Facebook-specific failure / dead-content markers — surface when Exa
    # renders an FB URL and the post has been deleted, restricted, or the
    # URL was fabricated. Without these, FB's "content not available" page
    # leaks through as if it were a successful scrape.
    "this content isn't available",
    "this content is no longer available",
    "content isn't available right now",
    "log in to facebook",
    "log in or sign up.+facebook",
]
_ANTI_BOT_RE = re.compile("|".join(ANTI_BOT_MARKERS), re.IGNORECASE)

# Hydration / SPA markers — pages that are JS-rendered and need
# dynamic=true to actually fetch the article body.
_HYDRATION_MARKERS = (
    "__NEXT_DATA__", "window.__INITIAL_STATE__", "__APOLLO_STATE__",
    "__NUXT__", "window.__PRELOADED_STATE__",
)
_SPA_ROOT_RE = re.compile(
    r'<div id="(root|__next|app|__nuxt)"[^>]*></div>',
    re.IGNORECASE,
)

# ScrapingDog escalation tiers. Each fired only when the previous tier's
# response was inadequate (HTTP fail, empty body, anti-bot marker, JS shell).
# NO host list — every URL gets the same cascade.
_SD_TIERS = (
    ("baseline",        {}),
    ("dynamic_render",  {"dynamic": "true", "wait": "5000"}),
    ("premium_stealth", {"premium": "true", "stealth_mode": "true"}),
    ("full_combined",   {"dynamic": "true", "wait": "8000",
                         "premium": "true", "stealth_mode": "true"}),
)
# Per-tier timeout (seconds) — cheap tiers should fail fast so we can
# escalate quickly when content is missing.
_SD_TIER_TIMEOUT = {
    "baseline":        25,
    "dynamic_render":  40,
    "premium_stealth": 40,
    "full_combined":   55,
}


JOB_BOARD_HOSTS = (
    "indeed.com", "builtin.com", "builtinnyc.com",
    "lever.co", "wellfound.com", "ziprecruiter.com",
    "greenhouse.io", "glassdoor.com",
    "startup.jobs", "remoterocketship.com", "salesjobs.com",
    "myworkdayjobs.com",
)
JOB_BODY_ANCHORS = (
    "responsibilities", "qualifications", "requirements",
    "about the role", "about the position", "about this role",
    "what you'll do", "what you will do", "what you’ll do",
    "we are looking for", "we're looking for", "we are seeking",
    "we’re looking for",
    "apply now", "apply for this job", "submit application",
    "job description",
    "job_position:", "job_description",
)


def _looks_like_job_body(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(a in low for a in JOB_BODY_ANCHORS)


def _is_job_board_url(url: str) -> bool:
    host = _host(url)
    return any(h in host for h in JOB_BOARD_HOSTS)


# ─────────────────────────────────────────────────────────────────────
# Deterministic helpers
# ─────────────────────────────────────────────────────────────────────
def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _looks_textual(content: str) -> bool:
    if not content:
        return False
    sample = content[:2000]
    printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
    return printable / max(len(sample), 1) > 0.85


def _has_anti_bot_marker(content: str) -> bool:
    if not content:
        return False
    return bool(_ANTI_BOT_RE.search(content[:5000]))


def _looks_like_js_shell(body: str) -> bool:
    """Heuristic: page is a JS framework shell whose content hasn't hydrated.
    True positives → escalate to a dynamic-render tier.

    Signals:
      - very short total length (< 3000 chars)
      - empty SPA root container present (`<div id="root"></div>`)
      - hydration markers present but visible-text density tiny (< 2% of HTML)
    """
    if not body:
        return False
    if len(body) < 3000:
        return True
    if _SPA_ROOT_RE.search(body):
        return True
    if any(m in body for m in _HYDRATION_MARKERS):
        text_only = re.sub(r"<[^>]+>", " ", body)
        text_only = re.sub(r"\s+", " ", text_only).strip()
        if len(text_only) < 500:
            return True
        if len(text_only) / max(len(body), 1) < 0.02:
            return True
    return False


def _evaluate_sd_response(status_code: int, body: str) -> str:
    """Classify a ScrapingDog response. Returns 'ok' or a short failure label.

    Failure labels drive escalation: js_shell / anti_bot_marker / body_too_short
    → retry with a stronger tier. http_404 → likely genuine dead URL, but try
    one render tier in case it's a JS-rendered page.
    """
    if status_code == 404:
        return "http_404"
    if status_code >= 500:
        return f"http_{status_code}"
    if status_code != 200:
        return f"http_{status_code}"
    if not body or len(body) < 500:
        return "body_too_short"
    if _has_anti_bot_marker(body):
        return "anti_bot_marker"
    if _looks_like_js_shell(body):
        return "js_shell"
    if not _looks_textual(body):
        return "non_textual"
    return "ok"


# Social-media post URL routing — ScrapingDog's specialized endpoints return
# clean structured post data instead of the JS shell the generic /scrape
# returns. Without these, FB/LinkedIn/X post URLs leak through as 60KB of
# page chrome (login walls, scripts) that look like a successful scrape.
_X_POST_RE = re.compile(r"(?:x|twitter)\.com/[^/?#]+/status/(\d+)", re.IGNORECASE)
_LINKEDIN_POST_RE = re.compile(r"linkedin\.com/posts/[^?#]*activity[-:](\d+)", re.IGNORECASE)
_FB_POST_RE = re.compile(r"facebook\.com/[^/?#]+/posts/", re.IGNORECASE)


async def _scrape_x_post(url: str) -> Dict[str, Any]:
    """SD specialized X (Twitter) post endpoint. Returns clean post text."""
    m = _X_POST_RE.search(url)
    if not m:
        return {"ok": False, "stage": "x_no_id", "content": "", "error": "tweetId not found in URL"}
    tweet_id = m.group(1)
    api_key = os.environ.get("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {"ok": False, "stage": "no_sd_key", "content": "", "error": "SCRAPINGDOG_API_KEY missing"}
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            r = await cli.get(
                "https://api.scrapingdog.com/x/post",
                params={"api_key": api_key, "tweetId": tweet_id},
            )
        if r.status_code != 200:
            return {"ok": False, "stage": f"x_http_{r.status_code}", "content": "", "error": r.text[:200]}
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        body = data.get("full_tweet") or data.get("tweet") or ""
        if not body:
            return {"ok": False, "stage": "x_empty", "content": "", "error": "no tweet body in response"}
        user = data.get("user") or {}
        # Synthesize a content block that the downstream verifier can parse like normal scraped text.
        synth = (
            f"X Post by {user.get('name', 'unknown')} (@{user.get('screen_name', '')}) "
            f"on {data.get('created_at', '')}\n\n{body}"
        )
        return {"ok": True, "stage": "sd:x_post", "content": synth, "error": ""}
    except Exception as e:
        return {"ok": False, "stage": "x_exception", "content": "", "error": f"{type(e).__name__}: {e}"}


async def _scrape_linkedin_post(url: str) -> Dict[str, Any]:
    """SD specialized LinkedIn post endpoint. Returns clean post text."""
    m = _LINKEDIN_POST_RE.search(url)
    if not m:
        return {"ok": False, "stage": "linkedin_post_no_id", "content": "", "error": "activity id not found in URL"}
    activity_id = m.group(1)
    api_key = os.environ.get("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {"ok": False, "stage": "no_sd_key", "content": "", "error": "SCRAPINGDOG_API_KEY missing"}
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            r = await cli.get(
                "https://api.scrapingdog.com/profile/post",
                params={"api_key": api_key, "id": activity_id},
            )
        if r.status_code != 200:
            return {"ok": False, "stage": f"linkedin_post_http_{r.status_code}", "content": "", "error": r.text[:200]}
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        post = data.get("post_results") or {}
        # The SD endpoint stores the actual post body under several non-obvious
        # locations depending on activity type — check in priority order.
        body = (
            post.get("post_text")
            or post.get("text")
            or post.get("content")
            or (post.get("related_post") or {}).get("text")
            or (post.get("related_post") or {}).get("post_text")
            or ""
        )
        author = (post.get("author") or {}).get("name", "unknown")
        date = post.get("activity_date", "")
        if not body:
            return {"ok": False, "stage": "linkedin_post_empty", "content": "", "error": "no post body in response"}
        synth = f"LinkedIn Post by {author} on {date}\n\n{body}"
        return {"ok": True, "stage": "sd:linkedin_post", "content": synth, "error": ""}
    except Exception as e:
        return {"ok": False, "stage": "linkedin_post_exception", "content": "", "error": f"{type(e).__name__}: {e}"}


def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse((url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return ""
        path = parsed.path.rstrip("/") if parsed.path != "/" else ""
        return urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, "")
        )
    except Exception:
        return url or ""


def _strip_www(host: str) -> str:
    h = (host or "").lower()
    return h[4:] if h.startswith("www.") else h


def _url_on_lead_domain(source_url: str,
                        company_website: str,
                        company_linkedin: str = "") -> bool:
    """True iff source URL is on the lead's own property:
      (a) URL hostname == (or subdomain of) ``company_website`` hostname, OR
      (b) URL is on linkedin.com AND its ``/company/<slug>`` matches the
          ``<slug>`` in ``company_linkedin``.

    Used to suppress wrong_entity flagging when the source URL is
    provably on the lead's own property.
    """
    if not source_url:
        return False
    try:
        src = _strip_www(urlparse(source_url).hostname or "")
    except Exception:
        return False
    if not src:
        return False

    # (a) Same-website match
    if company_website:
        try:
            web = _strip_www(urlparse(company_website).hostname or "")
            if web and (src == web or src.endswith("." + web)):
                return True
        except Exception:
            pass

    # (b) Same-LinkedIn match
    if company_linkedin and ("linkedin.com" in src):
        try:
            m = re.search(r"/company/([^/]+)", company_linkedin, re.I)
            lead_slug = (m.group(1).lower() if m else "")
            if lead_slug:
                # Source URL must reference the same slug
                if re.search(rf"/company/{re.escape(lead_slug)}(?:/|$)",
                             source_url, re.I):
                    return True
        except Exception:
            pass

    return False


def _normalize_company_for_match(name: str) -> str:
    """Strip legal-suffix tokens AND any preceding/following punctuation so
    the residual matches articles that omit the suffix.

    Example: "Emery Sapp & Sons, Inc." → "emery sapp & sons"
    (without this normalization, the trailing comma from "Sons," would stay
    after "Inc." was stripped, and `\bemery sapp & sons,\b` would fail to
    match articles that just say "Emery Sapp & Sons announced…").
    """
    n = name.lower().strip()
    # Strip one or more legal suffixes, each optionally preceded by ", " or
    # plain spaces, optionally followed by a period.  Apply globally so chains
    # like "Tractian Technologies, Inc." reduce both tokens in one pass.
    n = re.sub(
        r"\s*,?\s*\b(inc|llc|ltd|corp|corporation|company|"
        r"co|technologies?|holdings?|group)\b\.?",
        "",
        n,
    )
    # Clean up leftover trailing punctuation/whitespace.
    return n.strip(" ,;:.\t").strip()


def company_in_scrape(company_name: str, scraped_text: str) -> bool:
    """True iff the company name (or its base form with common legal/structural
    suffixes stripped) appears as a whole word in the scraped text
    (case-insensitive).  Word-boundary regex prevents false positives on
    incidental occurrences of short common-word company names."""
    if not company_name or not scraped_text:
        return False
    text_lower = scraped_text.lower()
    target = company_name.lower().strip()
    if re.search(rf"\b{re.escape(target)}\b", text_lower):
        return True
    base = _normalize_company_for_match(company_name)
    if base and base != target:
        return bool(re.search(rf"\b{re.escape(base)}\b", text_lower))
    return False


def _get_openrouter_key() -> str:
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    )


# ─────────────────────────────────────────────────────────────────────
# Scraping — SD primary (host-aware hardened) + Exa fallback
# ─────────────────────────────────────────────────────────────────────
async def _try_wayback(url: str) -> Dict[str, Any]:
    """Final-fallback: Wayback Machine snapshot of the URL.

    Tradeoff: snapshots may be 6-12 months stale, but a stale snapshot is
    far better evidence than nothing when ScrapingDog tiers all fail. Used
    only when every direct fetch tier comes back inadequate.
    """
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as cli:
            avail = await cli.get(
                f"http://archive.org/wayback/available?url={url}",
            )
            data = avail.json()
            snap = (data.get("archived_snapshots") or {}).get("closest") or {}
            if not snap.get("url"):
                return {"ok": False, "stage": "wayback_no_snapshot",
                        "content": "", "error": "no archived snapshot"}
            r = await cli.get(snap["url"])
            if r.status_code != 200:
                return {"ok": False, "stage": "wayback_http_error",
                        "content": "", "error": f"HTTP {r.status_code}"}
            body = r.text or ""
            if len(body) < 500:
                return {"ok": False, "stage": "wayback_too_short",
                        "content": "", "error": f"len={len(body)}"}
            return {"ok": True, "stage": "wayback",
                    "content": body[:MAX_SCRAPED_CHARS], "error": None}
    except Exception as e:
        return {"ok": False, "stage": "wayback_exception",
                "content": "", "error": f"{type(e).__name__}: {str(e)[:80]}"}


async def _scrape_sd_hardened(url: str) -> Dict[str, Any]:
    """Content-driven progressive escalation.

    Starts with the cheapest ScrapingDog call (baseline) and escalates only
    when the response is inadequate (HTTP failure / empty body / anti-bot
    marker / JS shell shape). When all ScrapingDog tiers exhaust, falls back
    to a Wayback Machine snapshot. NO hardcoded host list — every URL goes
    through the same cascade.

    Returns the original {ok, stage, content, error} contract so callers in
    verify_three_stage and the attribute-verification path work unchanged.
    A new 'stage_history' key carries the per-tier verdicts for telemetry.
    """
    api_key = os.environ.get("SCRAPINGDOG_API_KEY") or os.environ.get(
        "QUALIFICATION_SCRAPINGDOG_API_KEY"
    )
    if not api_key:
        return {"ok": False, "stage": "no_sd_key",
                "content": "", "error": "missing key"}

    # Request raw HTML (default) instead of markdown so trafilatura body
    # extraction (applied to the returned `content` below) can drop
    # nav/menu/footer/related-posts at the DOM level. SD's markdown
    # conversion preserves enough boilerplate that the first few thousand
    # chars are often chrome, not article body — feeding Sonar noise.
    # If trafilatura is unavailable or fails to extract, the helper returns
    # the raw HTML unchanged so behavior degrades gracefully.
    base_params = {"api_key": api_key, "url": url}
    history: List[tuple] = []
    last_status: Optional[int] = None
    last_verdict: str = "no_tier_attempted"

    async with httpx.AsyncClient() as cli:
        for tier_name, extra in _SD_TIERS:
            tier_timeout = _SD_TIER_TIMEOUT.get(tier_name, SCRAPE_TIMEOUT)
            params = {**base_params, **extra}
            try:
                r = await cli.get(
                    "https://api.scrapingdog.com/scrape",
                    params=params, timeout=tier_timeout,
                )
                body = r.text or ""
                verdict = _evaluate_sd_response(r.status_code, body)
                history.append((tier_name, verdict))
                last_status = r.status_code
                last_verdict = verdict
                if verdict == "ok":
                    # Extract article body from raw HTML before truncation.
                    # Removes nav/sidebar/footer/related-posts that otherwise
                    # eat the first chars of the prompt input.
                    try:
                        from qualification.scoring.verification_helpers import extract_article_body
                        body = extract_article_body(body)
                    except Exception:
                        pass  # fall through with original content
                    return {"ok": True, "stage": f"sd:{tier_name}",
                            "content": body[:MAX_SCRAPED_CHARS],
                            "error": None, "stage_history": history}
                # If origin returned 404 even after dynamic rendering, the URL
                # is genuinely dead — escalating to premium proxies won't help.
                if verdict == "http_404" and tier_name == "dynamic_render":
                    break
            except Exception as e:
                history.append((tier_name, f"exception:{type(e).__name__}"))
                last_verdict = f"exception:{type(e).__name__}"
                continue

    # All ScrapingDog tiers exhausted. Try Wayback as the final source of
    # content — stale snapshot is better than nothing for evidence verification.
    if last_verdict != "http_404" or last_status != 404:
        wb = await _try_wayback(url)
        history.append(("wayback", wb["stage"]))
        if wb["ok"]:
            return {"ok": True, "stage": "wayback",
                    "content": wb["content"], "error": None,
                    "stage_history": history}

    # Genuine unfetchable. Caller should treat this as "verifier infrastructure
    # could not reach the URL" — NOT as miner fabrication.
    fail_label = (
        "genuine_404" if last_verdict == "http_404"
        else f"all_tiers_exhausted:{last_verdict}"
    )
    return {"ok": False, "stage": fail_label,
            "content": "", "error": last_verdict,
            "stage_history": history}


async def _scrape_exa(url: str) -> Dict[str, Any]:
    """Exa Contents API fallback for URLs Scrapingdog cannot crack."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return {"ok": False, "stage": "no_exa_key",
                "content": "", "error": "missing key"}
    payload = {"ids": [url], "text": {"maxCharacters": MAX_SCRAPED_CHARS},
               "maxAgeHours": 0}
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.post(
                "https://api.exa.ai/contents",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=payload, timeout=SCRAPE_TIMEOUT,
            )
            if r.status_code != 200:
                return {"ok": False, "stage": "exa_http_error",
                        "content": "", "error": f"HTTP {r.status_code}"}
            data = r.json()
    except Exception as e:
        return {"ok": False, "stage": "exa_failed",
                "content": "", "error": f"{type(e).__name__}: {str(e)[:80]}"}

    results = data.get("results") or []
    if not results:
        return {"ok": False, "stage": "exa_no_results",
                "content": "",
                "error": (json.dumps(data.get("statuses") or [])[:120])}
    text = (results[0].get("text") or "")[:MAX_SCRAPED_CHARS]
    if len(text) < 300:
        return {"ok": False, "stage": "exa_thin",
                "content": text, "error": "<300 chars"}
    return {"ok": True, "stage": "exa_scraped", "content": text, "error": None}


# ─────────────────────────────────────────────────────────────────────
# LinkedIn-aware routing
# ─────────────────────────────────────────────────────────────────────
_LINKEDIN_JOB_ID_RE = re.compile(
    r"linkedin\.com/jobs/view/(?:[^/?#]*-)?(\d+)", re.IGNORECASE,
)

_LINKEDIN_JOB_CLOSED_RE = re.compile(
    r"(?i)\b("
    r"no longer accepting applications?"
    r"|no longer accepting"
    r"|applications? (?:are )?closed"
    r"|this job is closed"
    r"|position (?:has been )?filled"
    r"|we are no longer hiring"
    r"|job is no longer available"
    r"|expired"
    r")\b"
)

_LINKEDIN_REL_DATE_RE = re.compile(
    r"(?i)(\d+)\s+(year|month|week|day|hour|minute)s?\s+ago"
)

LINKEDIN_JOB_MAX_AGE_MONTHS = 6

_ACTIVE_HIRING_INTENT_RE = re.compile(
    r"(?i)\b("
    r"hiring|recruiting|recruits"
    r"|open\s+(?:position|role|vacancy|job)s?"
    r"|active\s+job\s+post(?:ing)?s?"
    r"|actively\s+seek|currently\s+seek"
    r")\b"
)


def _is_active_hiring_claim(miner_claim: str, target_signal_text: str) -> bool:
    """True if the miner's claim or the ICP intent signal is about active/
    current hiring. Used to scope the LinkedIn freshness/staleness gates
    so they don't block legitimate non-hiring claims (funding announcements,
    expansion signals, product launches, etc.) that can still be proven by
    closed or older job postings."""
    combined = f"{miner_claim or ''} {target_signal_text or ''}"
    return bool(_ACTIVE_HIRING_INTENT_RE.search(combined))


def _extract_linkedin_job_id(url: str) -> Optional[str]:
    m = _LINKEDIN_JOB_ID_RE.search(url or "")
    return m.group(1) if m else None


def _parse_relative_age_to_months(s: str) -> Optional[float]:
    """Convert 'N <unit> ago' → months (float).  Returns None if unrecognized."""
    if not s:
        return None
    m = _LINKEDIN_REL_DATE_RE.search(s)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "year":
        return n * 12.0
    if unit == "month":
        return float(n)
    if unit == "week":
        return n / 4.345
    if unit in ("day", "hour", "minute"):
        return n / 30.0 if unit == "day" else 0.0
    return None


async def _scrape_linkedin_job(url: str) -> Dict[str, Any]:
    api_key = os.environ.get("SCRAPINGDOG_API_KEY") or os.environ.get(
        "QUALIFICATION_SCRAPINGDOG_API_KEY"
    )
    if not api_key:
        return {"ok": False, "stage": "no_sd_key",
                "content": "", "error": "missing key"}
    job_id = _extract_linkedin_job_id(url)
    if not job_id:
        return {"ok": False, "stage": "linkedin_jobs_no_id",
                "content": "", "error": "could not extract job_id"}
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.get(
                "https://api.scrapingdog.com/linkedinjobs",
                params={"api_key": api_key, "job_id": job_id},
                timeout=SCRAPE_TIMEOUT,
            )
    except Exception as e:
        return {"ok": False, "stage": "linkedin_jobs_failed",
                "content": "", "error": f"{type(e).__name__}: {str(e)[:120]}"}
    if r.status_code != 200:
        return {"ok": False, "stage": "linkedin_jobs_http_error",
                "content": "", "error": f"HTTP {r.status_code}"}
    try:
        data = r.json()
    except Exception as e:
        return {"ok": False, "stage": "linkedin_jobs_parse_error",
                "content": "", "error": f"{type(e).__name__}"}
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict) or not data.get("job_position"):
        return {"ok": False, "stage": "linkedin_jobs_empty",
                "content": "", "error": "no job fields in response"}

    parts: List[str] = []
    jobs_status = data.get("jobs_status")
    if jobs_status:
        parts.append(f"jobs_status: {jobs_status}")
    posted = data.get("job_posting_time")
    if posted:
        parts.append(f"posted: {posted}")
    for key in ("job_position", "company_name", "job_location",
                "Employment_type", "Seniority_level", "Industries",
                "number_of_applicants", "base_pay"):
        v = data.get(key)
        if v:
            parts.append(f"{key}: {v}")
    desc = data.get("job_description") or ""
    if desc:
        parts.append("")
        parts.append(desc)

    text = "\n".join(parts)[:MAX_SCRAPED_CHARS]
    if len(text) < 50:
        return {"ok": False, "stage": "linkedin_jobs_thin",
                "content": text, "error": "<50 chars"}

    is_closed = bool(jobs_status and _LINKEDIN_JOB_CLOSED_RE.search(jobs_status))
    months_ago = _parse_relative_age_to_months(posted or "")
    is_stale = (
        months_ago is not None and months_ago > LINKEDIN_JOB_MAX_AGE_MONTHS
    )

    return {
        "ok": True,
        "stage": "linkedin_jobs_scraped",
        "content": text,
        "error": None,
        "meta": {
            "kind": "linkedin_job",
            "jobs_status": jobs_status,
            "posted_raw": posted,
            "months_ago": months_ago,
            "is_closed": is_closed,
            "is_stale": is_stale,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STAGE1_MODEL = os.environ.get("INTENT_THREE_STAGE_S1_MODEL", "perplexity/sonar")
STAGE3_MODEL = os.environ.get("INTENT_THREE_STAGE_S3_MODEL", "perplexity/sonar-pro")
TIMEOUT_SECONDS = 180
SCRAPE_TIMEOUT = 60

SIGNAL_STATUSES = [
    "supported", "partially_supported", "contradicted",
    "unable_to_verify", "wrong_entity",
]
CONFIDENCE_VALUES = ["high", "medium", "low"]


# ─────────────────────────────────────────────────────────────────────
# JSON schema (identical to standalone pipeline)
# ─────────────────────────────────────────────────────────────────────
def _output_schema() -> Dict[str, Any]:
    signal_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "signal_id", "claim", "verification_mode", "signal_status",
            "source_urls_supplied", "evidence_urls_used",
            "source_accessibility", "same_entity_check",
            "entity_match_reason", "supporting_quotes",
            "contradicting_quotes", "unsupported_parts",
            "source_quality", "risk_notes", "confidence",
            "claim_matches_miner_date",
            "author_type", "author_employer_matches_lead",
            "author_role_matches_spec", "author_satisfies_role_spec",
        ],
        "properties": {
            "signal_id": {"type": "string"},
            "claim": {"type": "string"},
            "verification_mode": {
                "type": "string", "enum": ["source_grounded", "discovery"],
            },
            "signal_status": {"type": "string", "enum": SIGNAL_STATUSES},
            "source_urls_supplied": {
                "type": "array", "items": {"type": "string"},
            },
            "evidence_urls_used": {
                "type": "array", "items": {"type": "string"},
            },
            "source_accessibility": {"type": "string"},
            "same_entity_check": {
                "type": "string", "enum": ["pass", "fail", "unclear"],
            },
            "entity_match_reason": {"type": "string"},
            "supporting_quotes": {
                "type": "array", "items": {"type": "string"},
            },
            "contradicting_quotes": {
                "type": "array", "items": {"type": "string"},
            },
            "unsupported_parts": {
                "type": "array", "items": {"type": "string"},
            },
            "source_quality": {"type": "string"},
            "risk_notes": {
                "type": "array", "items": {"type": "string"},
            },
            "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
            "claim_matches_miner_date": {
                "type": "string",
                "enum": ["consistent", "contradicted", "no_date_in_content"],
            },
            # PART D — AUTHOR-ROLE CHECK fields (apply only on social-post URLs
            # AND when target_icp_signal names a person role; otherwise "n/a")
            "author_type": {
                "type": "string",
                "enum": ["person", "company", "unknown", "n/a"],
            },
            "author_employer_matches_lead": {
                "type": "string",
                "enum": ["yes", "no", "unknown", "n/a"],
            },
            "author_role_matches_spec": {
                "type": "string",
                "enum": ["yes", "no", "unknown", "n/a"],
            },
            "author_satisfies_role_spec": {
                "type": "string",
                "enum": ["yes", "no", "unknown", "n/a"],
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overall_verdict", "overall_confidence", "summary",
            "signal_evaluations", "missing_or_risks",
        ],
        "properties": {
            "overall_verdict": {
                "type": "string",
                "enum": ["qualified", "disqualified", "needs_review"],
            },
            "overall_confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
            "summary": {"type": "string"},
            "signal_evaluations": {"type": "array", "items": signal_schema},
            "missing_or_risks": {
                "type": "array", "items": {"type": "string"},
            },
        },
    }


_SCHEMA = _output_schema()
_SYS_MESSAGE = (
    "You are a conservative B2B lead verification judge. Return JSON only."
)


# ─────────────────────────────────────────────────────────────────────
# Prompts — per-evidence-type builders live in qualification.scoring.prompts
# ─────────────────────────────────────────────────────────────────────
from qualification.scoring.prompts import default as _prompts_default
from qualification.scoring.prompts import social_posting as _prompts_social
from qualification.scoring.prompts import techstack as _prompts_techstack
from qualification.scoring.prompts import podcast as _prompts_podcast
from qualification.scoring.prompts._common import (
    lead_profile as _lead_profile_impl,
    visible_signal as _visible_signal_impl,
)


def _lead_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    return _lead_profile_impl(row)


def _visible_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    return _visible_signal_impl(row)


def _build_verification_prompt(row: Dict[str, Any]) -> str:
    """Stage 1 verification prompt — dispatcher.

    Routes to the per-evidence-type builder in
    ``qualification.scoring.prompts``.  The pre-refactor mega-prompt
    is preserved byte-for-byte:  for SOCIAL_POSTING the dispatcher
    returns the PART D-augmented builder; everything else (HIRING /
    FUNDING / None / unknown) returns the legacy-compat builder which
    also still emits PART D so the LLM input is identical to the
    pre-refactor input.

    Snapshot equality test: ``tests/test_prompt_refactor.py``.
    """
    et = row.get("_evidence_type")
    sig_id = row.get("id", "?")
    # NOTE: print() (not logger.info) — the validator's root logger is at
    # WARNING level which filters INFO records out of docker logs.  Audit
    # routing tracing MUST be visible at the docker stdout level so
    # operators can grep for it; print() satisfies that requirement
    # regardless of the logging-config tree.
    if et == "TECHSTACK":
        print(f"   verify[{sig_id}]: prompt_route=techstack (PART E)",
              flush=True)
        return _prompts_techstack.build_verification_prompt(row)
    if et == "SOCIAL_POSTING":
        print(f"   verify[{sig_id}]: prompt_route=social_posting (PART D)",
              flush=True)
        return _prompts_social.build_verification_prompt(row)
    if et == "PODCAST_APPEARANCE":
        print(f"   verify[{sig_id}]: prompt_route=podcast (PART F)",
              flush=True)
        return _prompts_podcast.build_verification_prompt(row)
    print(f"   verify[{sig_id}]: prompt_route=default evidence_type={et!r}",
          flush=True)
    return _prompts_default.build_verification_prompt(row)


def _build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    """Stage 3 final-judge prompt — dispatcher (mirrors verification).

    Snapshot equality test: ``tests/test_prompt_refactor.py``.
    """
    if row.get("_evidence_type") == "TECHSTACK":
        return _prompts_techstack.build_final_judge_prompt(row, contents, source_name)
    if row.get("_evidence_type") == "SOCIAL_POSTING":
        return _prompts_social.build_final_judge_prompt(row, contents, source_name)
    if row.get("_evidence_type") == "PODCAST_APPEARANCE":
        return _prompts_podcast.build_final_judge_prompt(row, contents, source_name)
    return _prompts_default.build_final_judge_prompt(row, contents, source_name)


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — SD-primary + Exa-fallback per URL
# ─────────────────────────────────────────────────────────────────────
async def _fetch_sd_then_exa(
    urls: List[str], max_chars: int = MAX_SCRAPED_CHARS,
) -> Dict[str, Any]:
    """For each supplied URL: try Scrapingdog (hardened) first; if SD fails,
    fall back to Exa Contents.  Returns the same {"results", "statuses"}
    envelope the standalone pipeline's fetch_exa_contents produced, so
    _build_final_judge_prompt is unchanged.
    """
    results: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    for url in (urls or [])[:3]:
        if not url:
            continue

        if _extract_linkedin_job_id(url):
            lij = await _scrape_linkedin_job(url)
            if lij.get("ok") and lij.get("content"):
                results.append({
                    "url": url, "title": "",
                    "text": lij["content"][:max_chars],
                    "meta": lij.get("meta") or {},
                })
                statuses.append({
                    "url": url, "source": "scrapingdog_linkedinjobs",
                    "stage": lij.get("stage"),
                    "meta": lij.get("meta") or {},
                })
                continue
            statuses.append({
                "url": url, "source": "scrapingdog_linkedinjobs_fallback",
                "linkedinjobs_stage": lij.get("stage"),
                "linkedinjobs_error": lij.get("error"),
            })

        # X (Twitter) post → SD specialized /x/post endpoint. Generic /scrape
        # returns the SPA shell; the specialized endpoint returns clean post text.
        if _X_POST_RE.search(url):
            xp = await _scrape_x_post(url)
            if xp.get("ok") and xp.get("content"):
                results.append({
                    "url": url, "title": "",
                    "text": xp["content"][:max_chars],
                })
                statuses.append({"url": url, "source": "scrapingdog_x_post", "stage": xp.get("stage")})
                continue
            statuses.append({
                "url": url, "source": "scrapingdog_x_post_failed",
                "x_post_stage": xp.get("stage"), "x_post_error": xp.get("error"),
            })

        # LinkedIn post → SD specialized /profile/post endpoint.
        if _LINKEDIN_POST_RE.search(url):
            lp = await _scrape_linkedin_post(url)
            if lp.get("ok") and lp.get("content"):
                results.append({
                    "url": url, "title": "",
                    "text": lp["content"][:max_chars],
                })
                statuses.append({"url": url, "source": "scrapingdog_linkedin_post", "stage": lp.get("stage")})
                continue
            statuses.append({
                "url": url, "source": "scrapingdog_linkedin_post_failed",
                "linkedin_post_stage": lp.get("stage"), "linkedin_post_error": lp.get("error"),
            })

        # Facebook post → skip SD generic (returns the JS shell that fools the
        # scraper into thinking it succeeded). Go straight to Exa, which renders
        # the page and exposes FB's "content not available" error for dead URLs.
        if _FB_POST_RE.search(url):
            exa = await _scrape_exa(url)
            if exa.get("ok") and exa.get("content") and not _has_anti_bot_marker(exa["content"]):
                results.append({
                    "url": url, "title": "",
                    "text": exa["content"][:max_chars],
                })
                statuses.append({"url": url, "source": "exa_fb_route", "stage": exa.get("stage")})
                continue
            statuses.append({
                "url": url, "source": "fb_unscrapable",
                "exa_stage": exa.get("stage"), "exa_error": exa.get("error"),
                "reason": "FB URL: Exa returned anti-bot/error page or empty",
            })
            continue

        sd = await _scrape_sd_hardened(url)
        if sd.get("ok") and sd.get("content"):
            results.append({
                "url": url, "title": "",
                "text": sd["content"][:max_chars],
            })
            statuses.append({
                "url": url, "source": "scrapingdog",
                "stage": sd.get("stage"),
            })
            continue
        exa = await _scrape_exa(url)
        if exa.get("ok") and exa.get("content"):
            results.append({
                "url": url, "title": "",
                "text": exa["content"][:max_chars],
            })
            statuses.append({
                "url": url, "source": "exa_fallback",
                "stage": exa.get("stage"),
                "sd_stage": sd.get("stage"),
            })
        else:
            statuses.append({
                "url": url, "source": "none",
                "sd_stage": sd.get("stage"),
                "sd_error": sd.get("error"),
                "exa_stage": exa.get("stage"),
                "exa_error": exa.get("error"),
            })
    return {"results": results, "statuses": statuses}


# ─────────────────────────────────────────────────────────────────────
# OpenRouter call with 429 retry / fail-soft
# ─────────────────────────────────────────────────────────────────────
async def _call_openrouter(
    client: httpx.AsyncClient, model: str, prompt: str,
) -> Dict[str, Any]:
    or_key = _get_openrouter_key()
    if not or_key:
        return {"_error": "no_openrouter_key"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYS_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verification",
                "strict": True,
                "schema": _SCHEMA,
            },
        },
    }
    for attempt in range(3):
        try:
            r = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                },
                json=body, timeout=TIMEOUT_SECONDS,
            )
            if r.status_code == 429:
                await asyncio.sleep(8 * (attempt + 1))
                continue
            if r.status_code != 200:
                return {
                    "_error": f"http_{r.status_code}",
                    "_body": r.text[:400],
                }
            resp = r.json()
            content = (resp.get("choices") or [{}])[0].get(
                "message", {}
            ).get("content", "")
            try:
                ans = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content)
                ans = (
                    json.loads(m.group(0))
                    if m else {"_raw_unparsed": content[:400]}
                )
            return {
                "answer": ans,
                "citations": resp.get("citations") or [],
                "usage": resp.get("usage") or {},
                "model": model,
            }
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt == 2:
                return {"_error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(3)
    return {"_error": "retries_exhausted"}


# ─────────────────────────────────────────────────────────────────────
# Guardrails + decision (identical to standalone pipeline)
# ─────────────────────────────────────────────────────────────────────
def _apply_guardrails(
    row: Dict[str, Any], verdict: Dict[str, Any],
) -> Dict[str, Any]:
    """Same rule as the standalone pipeline: in source_grounded mode, every
    cited evidence URL must be one of the supplied source URLs.  If any
    cited URL is off-list (or no URLs were cited), downgrade the status to
    unable_to_verify."""
    supplied = {
        _normalize_url(u) for u in (row.get("claimed_source_urls") or [])
    }
    for item in (verdict.get("signal_evaluations") or []):
        item["source_urls_supplied"] = list(
            row.get("claimed_source_urls")
            or item.get("source_urls_supplied")
            or []
        )
        if (
            item.get("verification_mode") == "source_grounded"
            and item.get("signal_status") in {"supported", "partially_supported"}
        ):
            evidence = [
                _normalize_url(u)
                for u in (item.get("evidence_urls_used") or [])
            ]
            bad = [u for u in evidence if u not in supplied]
            if bad or not evidence:
                item["signal_status"] = "unable_to_verify"
                item.setdefault("risk_notes", []).append(
                    "Provider used non-supplied evidence URL."
                )
    return verdict


def _decision(verdict: Dict[str, Any]) -> str:
    item = ((verdict.get("signal_evaluations") or [{}]) or [{}])[0]
    if item.get("same_entity_check") == "fail":
        return "reject"
    if (
        item.get("signal_status") == "supported"
        and item.get("confidence") == "high"
    ):
        return "approve"
    if item.get("signal_status") in {"contradicted", "wrong_entity"}:
        return "reject"
    return "review"


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
async def verify_three_stage(
    client: httpx.AsyncClient,
    *,
    company_name: str,
    company_linkedin: str,
    company_website: str,
    source_url: str,
    miner_claim: str,
    target_signal_text: str,
    contact_linkedin: str = "",
    stage1_model: Optional[str] = None,
    stage3_model: Optional[str] = None,
    miner_signal_date: Optional[str] = None,
    evidence_type: Optional[str] = None,
) -> Dict[str, Any]:
    """3-stage intent verification (sonar -> SD/Exa -> sonar-pro).

    Pipeline:
      1. Stage 1 sonar verdict.  approve / reject -> STOP and return.
      2. On review: fetch supplied URLs via SD (hardened) with Exa fallback.
      3. Pre-LLM company-name-in-scrape check on the fetched content.
         If the company name isn't anywhere in any fetched page, short-
         circuit as wrong_entity (no sonar-pro call).
      4. Stage 3 sonar-pro final verdict on the fetched content.

    Returns:
        client_ready (bool): True iff the FINAL pipeline decision is
            approve, OR review with INTENT_VERIFIER_REVIEW_AS_ACCEPT=on.
        decision (str): one of approve / reject / review (the raw pipeline
            output, before binary mapping).
        rejection_reason (str): empty when client_ready=True; otherwise
            describes which stage/status caused the rejection.
        stage1 (dict): {model, status, conf, decision, citations, usage}
        scrape (dict | None): {results, statuses} when Stage 2 fired;
            None when Stage 1 short-circuited.
        stage3 (dict | None): {model, status, conf, decision, citations,
            usage} when Stage 3 fired; None otherwise.
        company_check (bool | None): result of the pre-LLM company-in-scrape
            short-circuit.  None when not applicable (Stage 2 didn't fetch
            anything textual).
    """
    review_as_accept = os.environ.get(
        "INTENT_VERIFIER_REVIEW_AS_ACCEPT", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    row = {
        "id": "signal-1",
        "company": company_name,
        "website": company_website,
        "company_linkedin": company_linkedin,
        "contact_linkedin": contact_linkedin,
        "claim": miner_claim,
        "signal_date": miner_signal_date,
        "signal_type": "intent",
        "claimed_source_urls": [source_url] if source_url else [],
        "_target_signal_text": target_signal_text,
        # Dispatcher in _build_verification_prompt routes on this — TECHSTACK
        # adds PART E (tech-stack anti-patterns), SOCIAL_POSTING adds PART D
        # (author-role check), other values fall through to the default
        # builder.  None is fine; the dispatcher's default branch handles it.
        "_evidence_type": (evidence_type or "").strip().upper() or None,
    }

    # Structural same-entity override: when the source URL is on the
    # lead's own ``company_website`` host (or subdomain), or on the
    # lead's exact LinkedIn ``/company/<slug>`` path, ``wrong_entity``
    # is logically impossible — the entity IS the lead by hostname/slug
    # match alone.  Used below to downgrade any Stage 1 / Stage 3
    # wrong_entity verdict on those URLs.
    _on_lead_domain = _url_on_lead_domain(
        source_url, company_website, company_linkedin,
    )

    # ── STAGE 1: sonar first-pass ──────────────────────────────────
    s1_prompt = _build_verification_prompt(row)
    s1_envelope = await _call_openrouter(
        client, stage1_model or STAGE1_MODEL, s1_prompt
    )
    if s1_envelope.get("_error"):
        return {
            "client_ready": False,
            "decision": "reject",
            "rejection_reason": f"stage1_llm_error:{s1_envelope['_error']}",
            "stage1": {
                "model": stage1_model or STAGE1_MODEL,
                "status": "llm_error",
                "confidence": None,
                "decision": "reject",
                "same_entity_check": None,
                "usage": {},
                "error": s1_envelope.get("_error"),
            },
            "scrape": None,
            "stage3": None,
            "company_check": None,
        }
    s1_verdict_raw = (s1_envelope.get("answer") or {})
    s1_verdict = _apply_guardrails(row, s1_verdict_raw)
    s1_item = ((s1_verdict.get("signal_evaluations") or [{}]) or [{}])[0]
    s1_decision = _decision(s1_verdict)
    stage1_info = {
        "model": s1_envelope.get("model"),
        "status": s1_item.get("signal_status"),
        "confidence": s1_item.get("confidence"),
        "decision": s1_decision,
        "same_entity_check": s1_item.get("same_entity_check"),
        "author_type": s1_item.get("author_type"),
        "author_employer_matches_lead": s1_item.get("author_employer_matches_lead"),
        "author_role_matches_spec": s1_item.get("author_role_matches_spec"),
        "author_satisfies_role_spec": s1_item.get("author_satisfies_role_spec"),
        "usage": s1_envelope.get("usage") or {},
    }

    if s1_decision == "approve":
        return {
            "client_ready": True,
            "decision": "approve",
            "rejection_reason": "",
            "stage1": stage1_info,
            "scrape": None,
            "stage3": None,
            "company_check": None,
            "verdict": s1_verdict,
        }
    if s1_decision == "reject":
        # Override: when URL is on the lead's own domain AND the
        # rejection is specifically for entity-identity reasons
        # (same_entity_check == "fail"), downgrade to review.  URLs on
        # the lead's own property are structural proof of same-entity,
        # so wrong_entity for ENTITY reasons can't logically apply.
        # We do NOT override `wrong_entity` rejects that are actually
        # for claim-mismatch (same_entity_check != "fail"), nor any
        # other reject status like `contradicted`.
        if (
            _on_lead_domain
            and s1_item.get("signal_status") == "wrong_entity"
            and s1_item.get("same_entity_check") == "fail"
        ):
            stage1_info["status"] = "review"
            stage1_info["decision"] = "review"
            stage1_info["same_entity_check"] = "pass"
            stage1_info["domain_override"] = "url_on_lead_domain"
            # Fall through to Stage 2/3 below
        else:
            return {
                "client_ready": False,
                "decision": "reject",
                "rejection_reason": (
                    f"stage1_{s1_item.get('signal_status') or 'reject'}"
                ),
                "stage1": stage1_info,
                "scrape": None,
                "stage3": None,
                "company_check": None,
                "verdict": s1_verdict,
            }

    # ── STAGE 2: SD-primary + Exa-fallback fetch ───────────────────
    if not row["claimed_source_urls"]:
        return {
            "client_ready": False,
            "decision": "reject",
            "rejection_reason": "no_source_url_for_stage2",
            "stage1": stage1_info,
            "scrape": None,
            "stage3": None,
            "company_check": None,
            "verdict": s1_verdict,
        }

    contents = await _fetch_sd_then_exa(row["claimed_source_urls"])

    # ── PRE-STAGE-3: deterministic company-name-in-scrape check ───
    # Skip the sonar-pro call entirely when the scraped text doesn't even
    # mention the lead's company (or its base form).  Saves token cost and
    # latency on obvious wrong-entity URLs (Marriott PR for Artha Capital,
    # Sordo Madaleno post tagged as Artha Capital, Grupo Integra-T page
    # tagged as Grupo Integra, etc.).
    combined_text = "\n".join(
        (r.get("text") or "") for r in (contents.get("results") or [])
    )
    company_check: Optional[bool] = None
    if combined_text.strip():
        company_check = company_in_scrape(company_name, combined_text)
        if not company_check:
            return {
                "client_ready": False,
                "decision": "reject",
                "rejection_reason": (
                    "wrong_entity_company_not_in_fetched_content"
                ),
                "stage1": stage1_info,
                "scrape": {"statuses": contents.get("statuses") or [],
                           "result_count": len(contents.get("results") or [])},
                "stage3": None,
                "company_check": False,
                "verdict": {
                    "signal_evaluations": [{
                        "signal_status": "wrong_entity",
                        "verification_mode": "source_grounded",
                        "entity_match_reason":
                            "company name absent in fetched content",
                        "confidence": "high",
                    }],
                },
            }

    is_hiring_claim = _is_active_hiring_claim(
        row.get("claim") or "",
        row.get("_target_signal_text") or "",
    )
    for res in (contents.get("results") or []):
        meta = res.get("meta") or {}
        if meta.get("kind") != "linkedin_job":
            continue
        if not is_hiring_claim:
            continue
        if meta.get("is_closed"):
            return {
                "client_ready": False,
                "decision": "reject",
                "rejection_reason": "linkedin_job_closed",
                "stage1": stage1_info,
                "scrape": {"statuses": contents.get("statuses") or [],
                           "result_count": len(contents.get("results") or [])},
                "stage3": None,
                "company_check": company_check,
                "verdict": {
                    "signal_evaluations": [{
                        "signal_status": "contradicted",
                        "verification_mode": "source_grounded",
                        "entity_match_reason": (
                            f"LinkedIn /linkedinjobs API reports posting is "
                            f"closed; jobs_status={meta.get('jobs_status')!r}"
                        ),
                        "confidence": "high",
                    }],
                },
            }
        if meta.get("is_stale"):
            return {
                "client_ready": False,
                "decision": "reject",
                "rejection_reason": "linkedin_job_stale",
                "stage1": stage1_info,
                "scrape": {"statuses": contents.get("statuses") or [],
                           "result_count": len(contents.get("results") or [])},
                "stage3": None,
                "company_check": company_check,
                "verdict": {
                    "signal_evaluations": [{
                        "signal_status": "contradicted",
                        "verification_mode": "source_grounded",
                        "entity_match_reason": (
                            f"LinkedIn posting age {meta.get('months_ago'):.1f}"
                            f" months exceeds {LINKEDIN_JOB_MAX_AGE_MONTHS}"
                            f"-month freshness cap "
                            f"(posted: {meta.get('posted_raw')!r})"
                        ),
                        "confidence": "high",
                    }],
                },
            }

    has_linkedin_structured = any(
        (r.get("meta") or {}).get("kind") == "linkedin_job"
        for r in (contents.get("results") or [])
    )
    is_job_board = any(
        _is_job_board_url(u) for u in (row.get("claimed_source_urls") or [])
    )
    if is_job_board and not has_linkedin_structured:
        combined_for_gate = "\n".join(
            (r.get("text") or "") for r in (contents.get("results") or [])
        )
        if not _looks_like_job_body(combined_for_gate):
            return {
                "client_ready": False,
                "decision": "reject",
                "rejection_reason": "job_body_not_in_fetched_content",
                "stage1": stage1_info,
                "scrape": {"statuses": contents.get("statuses") or [],
                           "result_count": len(contents.get("results") or [])},
                "stage3": None,
                "company_check": company_check,
                "verdict": {
                    "signal_evaluations": [{
                        "signal_status": "unable_to_verify",
                        "verification_mode": "source_grounded",
                        "entity_match_reason": (
                            "scrape returned shell-only content for a "
                            "job-board URL — no job body anchors found"
                        ),
                        "confidence": "high",
                    }],
                },
            }

    # ── STAGE 3: sonar-pro final judge ─────────────────────────────
    s3_prompt = _build_final_judge_prompt(row, contents)
    s3_envelope = await _call_openrouter(
        client, stage3_model or STAGE3_MODEL, s3_prompt
    )
    if s3_envelope.get("_error"):
        return {
            "client_ready": False,
            "decision": "reject",
            "rejection_reason": f"stage3_llm_error:{s3_envelope['_error']}",
            "stage1": stage1_info,
            "scrape": {"statuses": contents.get("statuses") or [],
                       "result_count": len(contents.get("results") or [])},
            "stage3": {
                "model": stage3_model or STAGE3_MODEL,
                "status": "llm_error",
                "confidence": None,
                "decision": "reject",
                "same_entity_check": None,
                "usage": {},
                "error": s3_envelope.get("_error"),
            },
            "company_check": company_check,
        }
    s3_verdict_raw = (s3_envelope.get("answer") or {})
    s3_verdict = _apply_guardrails(row, s3_verdict_raw)
    s3_item = ((s3_verdict.get("signal_evaluations") or [{}]) or [{}])[0]
    s3_decision = _decision(s3_verdict)
    stage3_info = {
        "model": s3_envelope.get("model"),
        "status": s3_item.get("signal_status"),
        "confidence": s3_item.get("confidence"),
        "decision": s3_decision,
        "same_entity_check": s3_item.get("same_entity_check"),
        "claim_matches_miner_date": s3_item.get("claim_matches_miner_date"),
        "author_type": s3_item.get("author_type"),
        "author_employer_matches_lead": s3_item.get("author_employer_matches_lead"),
        "author_role_matches_spec": s3_item.get("author_role_matches_spec"),
        "author_satisfies_role_spec": s3_item.get("author_satisfies_role_spec"),
        "usage": s3_envelope.get("usage") or {},
    }

    # Override: same precise condition as Stage 1.  Only downgrade when
    # ``wrong_entity`` is specifically for entity-identity reasons
    # (``same_entity_check == "fail"``).  A claim-mismatch wrong_entity
    # verdict (which shouldn't happen with the updated prompt, but is
    # defended against here) is left as-is.
    if (
        _on_lead_domain
        and s3_item.get("signal_status") == "wrong_entity"
        and s3_item.get("same_entity_check") == "fail"
    ):
        stage3_info["status"] = "review"
        stage3_info["decision"] = "review"
        stage3_info["same_entity_check"] = "pass"
        stage3_info["domain_override"] = "url_on_lead_domain"
        s3_decision = "review"

    # Binary mapping for production: approve -> accept; reject -> reject;
    # review -> reject by default (set INTENT_VERIFIER_REVIEW_AS_ACCEPT=on
    # to flip review to accept).
    if s3_decision == "approve":
        client_ready = True
        reason = ""
    elif s3_decision == "reject":
        client_ready = False
        reason = f"stage3_{s3_item.get('signal_status') or 'reject'}"
    else:  # review
        client_ready = review_as_accept
        reason = "" if review_as_accept else "stage3_review"

    return {
        "client_ready": client_ready,
        "decision": s3_decision,
        "rejection_reason": reason,
        "stage1": stage1_info,
        "scrape": {"statuses": contents.get("statuses") or [],
                   "result_count": len(contents.get("results") or [])},
        "stage3": stage3_info,
        "company_check": company_check,
        "verdict": s3_verdict,
    }
