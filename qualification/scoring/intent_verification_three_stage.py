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
  supplied URL with the full hardening:
    * JS_HEAVY_HOSTS -> dynamic=true + wait=5000
    * ANTI_BOT_HOSTS -> premium=true + stealth_mode=true
    * 3x retry with exponential backoff
    * anti-bot marker detection + premium escalation
    * length sanity check after retry
    * binary blob detection
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
# Scraping config (host hardening profiles)
# ─────────────────────────────────────────────────────────────────────
MAX_SCRAPED_CHARS = 60_000
SCRAPE_TIMEOUT = 60

JS_HEAVY_HOSTS = (
    "indeed.com", "builtin.com", "linkedin.com",
    "ziprecruiter.com", "lever.co", "greenhouse.io",
)
ANTI_BOT_HOSTS = (
    "forbes.com", "bloomberg.com", "reuters.com", "wsj.com",
    "ft.com", "finance.yahoo.com",
)
ANTI_BOT_MARKERS = [
    "checking your browser", "captcha", "verify you are human",
    "ddos protection", "challenge-platform", "access denied",
    "security check",
]

HOST_SCRAPE_CONFIG = {
    "ziprecruiter.com": {"dynamic": "true", "wait": "15000",
                         "premium": "true", "stealth_mode": "true"},
    "wellfound.com":    {"dynamic": "true", "wait": "15000"},
    "glassdoor.com":    {},
    "builtin.com":      {},
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
    low = (content or "")[:5000].lower()
    return any(m in low for m in ANTI_BOT_MARKERS)


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
    base = re.sub(
        r"\b(inc|llc|ltd|corp|company|technologies?|holdings?|group)\b\.?",
        "", target,
    ).strip()
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
async def _scrape_sd_hardened(url: str) -> Dict[str, Any]:
    """Direct Scrapingdog /scrape call with host-specific hardening (dynamic
    rendering for JS-heavy hosts, premium proxy + stealth for anti-bot hosts)
    and 3x retry with exponential backoff.  After anti-bot retry, re-validates
    length, textuality, and absence of challenge markers."""
    api_key = os.environ.get("SCRAPINGDOG_API_KEY") or os.environ.get(
        "QUALIFICATION_SCRAPINGDOG_API_KEY"
    )
    if not api_key:
        return {"ok": False, "stage": "no_sd_key", "content": "", "error": "missing key"}

    host = _host(url)
    params = {"api_key": api_key, "url": url, "format": "markdown"}

    override = next(
        (cfg for h, cfg in HOST_SCRAPE_CONFIG.items() if h in host),
        None,
    )
    if override is not None:
        params.update(override)
    else:
        if any(h in host for h in JS_HEAVY_HOSTS):
            params["dynamic"] = "true"
            params["wait"] = "5000"
        if any(h in host for h in ANTI_BOT_HOSTS):
            params["premium"] = "true"
            params["stealth_mode"] = "true"

    content = None
    last_err = None
    async with httpx.AsyncClient() as cli:
        for i in range(3):
            try:
                r = await cli.get(
                    "https://api.scrapingdog.com/scrape",
                    params=params, timeout=SCRAPE_TIMEOUT,
                )
                if r.status_code == 200:
                    content = r.text
                    if content and len(content.strip()) > 300:
                        break
                else:
                    last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{type(e).__name__}"
            await asyncio.sleep(2 ** i)

    if not content or len(content) < 300:
        return {"ok": False, "stage": "fetch_failed",
                "content": "", "error": last_err or "empty"}
    if not _looks_textual(content):
        return {"ok": False, "stage": "binary_blob",
                "content": "", "error": "non_textual_content"}
    if _has_anti_bot_marker(content) and override is None:
        params["premium"] = "true"
        params["stealth_mode"] = "true"
        try:
            async with httpx.AsyncClient() as cli:
                r = await cli.get(
                    "https://api.scrapingdog.com/scrape",
                    params=params, timeout=SCRAPE_TIMEOUT,
                )
                content = r.text
        except Exception as e:
            return {"ok": False, "stage": "anti_bot_retry_failed",
                    "content": "", "error": str(e)[:120]}
        if not content or len(content) < 300:
            return {"ok": False, "stage": "anti_bot_retry_thin",
                    "content": "", "error": f"retry returned {len(content or '')} chars"}
        if not _looks_textual(content):
            return {"ok": False, "stage": "anti_bot_retry_binary",
                    "content": "", "error": "non_textual_content_after_retry"}
        if _has_anti_bot_marker(content):
            return {"ok": False, "stage": "anti_bot_blocked",
                    "content": "", "error": "challenge_persisted"}
    return {"ok": True, "stage": "sd_scraped",
            "content": content[:MAX_SCRAPED_CHARS], "error": None}


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
# Prompts (identical to standalone pipeline)
# ─────────────────────────────────────────────────────────────────────
def _lead_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: row.get(k, "")
        for k in ("company", "website", "company_linkedin", "contact_linkedin")
    }


def _visible_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    # Surface BOTH the miner's free-text claim AND the buyer's ICP signal so
    # the LLM can check semantic alignment between them.  Without
    # target_icp_signal in the prompt, the verifier only judged "does the URL
    # support what the miner wrote?" — a miner could submit a true-but-
    # orthogonal claim (e.g. "Artisan offers an AI BDR product") and get
    # approved even though it doesn't satisfy the actual ICP signal ("Company
    # has active job postings for SDR/BDR roles"). Observed 2026-05-18 on
    # multiple winners. Including target_icp_signal lets the prompt rules
    # below enforce: the claim must semantically address the ICP signal AND
    # the URL must support the claim.
    return {
        "id": str(row.get("id") or "signal-1"),
        "signal_type": row.get("signal_type") or "unknown",
        "miner_claim": row.get("claim") or "",
        "target_icp_signal": row.get("_target_signal_text") or "",
        "claimed_source_urls": row.get("claimed_source_urls") or [],
    }


def _build_verification_prompt(row: Dict[str, Any]) -> str:
    signal = _visible_signal(row)
    return f"""Evaluate this B2B sales lead.

Lead profile:
{json.dumps(_lead_profile(row), indent=2)}

Intent signal to verify:
{json.dumps(signal, indent=2)}

Two-part verification — BOTH must hold for `supported`:

  PART A: Does `miner_claim` semantically satisfy `target_icp_signal`?
    The miner asserts their evidence proves the buyer's target_icp_signal.
    First check whether the miner_claim, even if true, would actually mean
    the target_icp_signal is satisfied.
      * "Company X offers product Y" is NOT the same as "Company X hires for
        role Y". Selling an AI BDR is the opposite signal of hiring BDRs.
      * "Company X had a funding round" is NOT the same as "Company X is
        hiring sales roles" unless the URL specifically ties the funding to
        sales-team expansion.
      * "Company X has a careers page" is NOT the same as "Company X has
        open positions for {{role}}".
    If miner_claim does not semantically map to target_icp_signal, return
    wrong_entity (the claim is about a different thing) regardless of how
    well the URL supports the claim.

  PART B: Does the supplied source URL support `miner_claim`?
    - Treat the lead profile as context, not proof.
    - Anchor company identity to company website and company LinkedIn.
    - If source URL(s) are provided, use source_grounded mode.
    - In source_grounded mode, validate only against exact supplied source URL(s).
    - Do not use alternate URLs, same-domain pages, cached pages, search results, or replacement URLs as support.
    - Independent search may only flag contradiction, stale data, or wrong-entity issues.
    - If no source URL is provided, use discovery mode and search credible sources.

Signal status decision:
- supported: PART A holds AND exact evidence directly supports miner_claim AND entity match.
- partially_supported: PART A holds AND exact evidence supports only part of miner_claim.
- contradicted: exact evidence or stronger current evidence clearly contradicts miner_claim.
- unable_to_verify: evidence is missing, inaccessible, ambiguous, stale, or insufficient.
- wrong_entity: PART A fails (miner_claim does not semantically address target_icp_signal),
                OR evidence is about a different company/person.

Return only schema-valid JSON."""


def _build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    blocks: List[str] = []
    for res in (contents.get("results") or []):
        url = res.get("url") or res.get("id") or ""
        text = (res.get("text") or "")[:MAX_SCRAPED_CHARS]
        title = res.get("title") or ""
        blocks.append(f"URL: {url}\nTITLE: {title}\nCONTENT:\n{text}")
    if not blocks:
        blocks = [
            "NO CONTENT. STATUSES:\n"
            + json.dumps(contents.get("statuses") or [], indent=2)
        ]
    today_str = date.today().isoformat()
    return f"""{_build_verification_prompt(row)}

{source_name} exact supplied source extraction:
{chr(10).join(blocks)}

Today's date: {today_str}

Final judge rules:
- Re-apply the PART A check from above BEFORE judging content support: if
  miner_claim does not semantically map to target_icp_signal, return
  wrong_entity regardless of what the extracted content shows. Do not let a
  factually-true but orthogonal claim pass just because the URL supports it.
- Use only the exact source extraction above as supporting evidence.
- Page titles, navigation menus, headers, and breadcrumbs are NOT evidence.
  Only specific factual claims in the page BODY count.
- If the body contains explicit negation about the claim ("0 open positions",
  "no longer open", "no longer accepting applications", "not currently hiring",
  "position has been removed", "page not found", "404",
  "the job you are looking for is no longer"),
  return contradicted regardless of titles, headers, or partial context.
- Job-posting freshness rule: when the claim is about an active/open job
  posting (hiring signal), the source MUST show that posting is still open.
  Treat any of these closed-state phrases as contradicted — even if other
  parts of the page still describe the role:
    "no longer accepting applications", "applications are closed",
    "this job is closed", "position filled", "we are no longer hiring",
    "job is no longer available", "expired".
- Job-posting timeline rule: when the claim is about active/current hiring
  (e.g. "is hiring", "actively recruiting", "open positions for X"), if the
  source body shows a posting age (e.g. "Posted 7 months ago", "Posted on
  YYYY-MM-DD") AND that age is > 6 months relative to "Today's date" above,
  return contradicted (stale_posting). Job-board sidebars and "similar
  jobs" timestamps do NOT count — only the posting age of the ACTUAL job
  being verified.
  For non-hiring claims (e.g. funding announcements, expansion signals,
  product launches, acquisitions, tech-stack inferences from job
  requirements), do NOT penalize on age — older or closed job postings can
  still validate those factual claims.  If no posting age is visible on the
  page, do NOT penalize on staleness; judge content only.
- If exact extracted content directly supports miner_claim AND PART A holds,
  return supported.
- If extracted content supports only part of miner_claim AND PART A holds,
  return partially_supported.
- If extraction failed or content is insufficient, return unable_to_verify.
- If content contradicts miner_claim, return contradicted.
- If content is about another company/person, return wrong_entity.
- evidence_urls_used must contain only exact supplied source URLs whose
  extracted content supports or contradicts the claim."""


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
        "signal_type": "intent",
        "claimed_source_urls": [source_url] if source_url else [],
        "_target_signal_text": target_signal_text,
    }

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
        "usage": s3_envelope.get("usage") or {},
    }

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
