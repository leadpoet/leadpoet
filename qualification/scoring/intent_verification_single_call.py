"""Intent verification — single-LLM-call pipeline.

Alternative to the two-prompt verifier in intent_verification_v2.py.  Where v2
splits verification into two sequential LLM calls (Sonar grounding + Claude
signal-match), this module does the whole job in ONE LLM call after scraping.
Activated via the INTENT_VERIFIER_SINGLE_CALL env flag.

What the pipeline does, in order:

  1. SCRAPE the miner-supplied source URL.
     • Scrapingdog is the primary scraper, with host-aware hardening:
         - JS-heavy hosts (Indeed, Built In, LinkedIn, Greenhouse, etc.) →
           dynamic rendering + 5s wait so the page actually executes its JS.
         - Anti-bot hosts (Forbes, Bloomberg, Reuters, WSJ, Yahoo Finance,
           etc.) → premium proxy + stealth fingerprint.
         - 3× retry with exponential backoff on transient failures.
     • Exa Contents is used ONLY as a fallback when Scrapingdog cannot
       crack the URL (returns anti_bot_blocked, binary_blob, fetch_failed,
       or empty).  Burns Scrapingdog credit by default; Exa fires on the
       small minority of URLs SD cannot reach (e.g., Indeed search pages).

  2. DETERMINISTIC PYTHON CHECKS (no LLM cost):
     • Textuality:   page must be >300 chars and >85% printable.
     • Anti-bot:     scan first 5KB for challenge-page markers; escalate
                      to premium+stealth retry if found.
     • Company name: if the scrape succeeded but the lead's company name
                      does not appear anywhere in the scraped text, return
                      wrong_entity immediately without calling the LLM.
                      This prevents the model from accepting a clearly off-
                      topic page on the strength of web-search corroboration
                      when the supplied URL itself is irrelevant.

  3. ONE LLM CALL via OpenRouter:
     • Default model: perplexity/sonar:online
       (override with INTENT_SINGLE_CALL_MODEL env var)
     • Strict JSON schema enforced via response_format.
     • Scraped page content is embedded in the prompt as PRIMARY evidence.
     • Web search is enabled (the :online plugin) for corroboration — the
       prompt explicitly says "scraped content is primary; web search may
       corroborate or contradict".

  4. GUARDRAIL on the model's response:
     • If the scrape succeeded, the supplied URL MUST appear in the model's
       evidence_urls_used list.  Otherwise the model bypassed our scraped
       content (suspicious) — downgrade to unable_to_verify.
     • Additional web URLs cited alongside the supplied URL are allowed
       (the :online plugin naturally adds search citations).
     • If the scrape failed, any evidence URL is accepted — this is how the
       verifier recovers a verdict for URLs we could not fetch (e.g.,
       Indeed search pages, paywalled news articles).

Public API: verify_single_call() — mirrors verify_v2()'s contract so the
caller in lead_scorer.py can swap between them via env flag.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from qualification.scoring.intent_verification import scrapingdog_generic

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("INTENT_SINGLE_CALL_MODEL", "perplexity/sonar:online")
TIMEOUT_SECONDS = 120
SCRAPE_TIMEOUT = 60
MAX_SCRAPED_CHARS = 60_000

JS_HEAVY_HOSTS = (
    "indeed.com", "builtin.com", "linkedin.com", "glassdoor.com",
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
        from urllib.parse import urlunsplit
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path,
                           parsed.query, ""))
    except Exception:
        return url or ""


def company_in_scrape(company_name: str, scraped_text: str) -> bool:
    """True iff the company name (or its base form, stripping common suffixes)
    appears as a whole word in the scraped text (case-insensitive).

    Uses a word-boundary regex match rather than plain substring containment
    so that short common-word company names do not false-positive on
    incidental occurrences inside ordinary prose. For example, a company
    named "Apple" should not match a page that merely contains the phrase
    "apple of his eye"; only a real mention of the company would set off
    the word-boundary regex.

    Returns True if either the full company name or its base form
    (with trailing legal/structural suffixes such as Inc / LLC / Group
    stripped) appears as a whole-word match anywhere in the scraped text.
    """
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


# ─────────────────────────────────────────────────────────────────────
# Scraping — SD primary + Exa fallback
# ─────────────────────────────────────────────────────────────────────
async def _scrape_sd_hardened(url: str) -> Dict[str, Any]:
    """Direct Scrapingdog /scrape call with host-specific hardening (dynamic
    rendering, premium proxy, stealth fingerprint) and 3× retry.  Returns
    {ok, stage, content, error}.

    Falls back internally to scrapingdog_generic() (the v1 helper) ONLY when
    the hardened config call itself raises an exception, so we still benefit
    from v1's static-then-dynamic logic on simple pages."""
    api_key = os.environ.get("SCRAPINGDOG_API_KEY") or os.environ.get(
        "QUALIFICATION_SCRAPINGDOG_API_KEY"
    )
    if not api_key:
        return {"ok": False, "stage": "no_sd_key", "content": "", "error": "missing key"}

    host = _host(url)
    params = {"api_key": api_key, "url": url, "format": "markdown"}
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
    if _has_anti_bot_marker(content):
        # Escalate to premium proxy + stealth fingerprint and retry once.
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
        # After the retry, re-validate the response — it must be (a) long
        # enough, (b) actually textual, and (c) free of anti-bot markers.
        # Without this check, a retry that returns a short error / stub page
        # (e.g., 210-char "request unsuccessful" body) would slip through as
        # ok=sd_scraped and block the Exa fallback path.
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


async def scrape_url(url: str) -> Dict[str, Any]:
    """SD-primary, Exa-fallback wrapper."""
    if not url:
        return {"ok": False, "stage": "no_url", "content": "", "error": "no url"}
    sd = await _scrape_sd_hardened(url)
    if sd["ok"]:
        return sd
    exa = await _scrape_exa(url)
    if exa["ok"]:
        exa["stage"] = f"sd_{sd['stage']}_then_exa_scraped"
        return exa
    return {"ok": False,
            "stage": f"sd_{sd['stage']}_and_exa_{exa['stage']}",
            "content": "",
            "error": f"sd:{sd.get('error','?')} | exa:{exa.get('error','?')}"}


# ─────────────────────────────────────────────────────────────────────
# Prompt + schema
# ─────────────────────────────────────────────────────────────────────
# The JSON schema enforced on the model's response.  Every field is
# required; the model literally cannot return a malformed object.
_VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "signal_status", "confidence", "evidence_urls_used",
        "supporting_quotes", "contradicting_quotes",
        "entity_match_reason", "rationale",
    ],
    "properties": {
        "signal_status": {
            "type": "string",
            "enum": ["supported", "partially_supported", "contradicted",
                     "unable_to_verify", "wrong_entity"],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence_urls_used": {"type": "array", "items": {"type": "string"}},
        "supporting_quotes": {"type": "array", "items": {"type": "string"}},
        "contradicting_quotes": {"type": "array", "items": {"type": "string"}},
        "entity_match_reason": {"type": "string"},
        "rationale": {"type": "string"},
    },
}

_SYS_MESSAGE = (
    "You are a strict B2B intent-signal verifier. Respond ONLY with a JSON "
    "object matching the json_schema enforced by response_format. No markdown."
)


def _build_prompt(
    *,
    company_name: str,
    company_linkedin: str,
    company_website: str,
    contact_linkedin: str,
    source_url: str,
    miner_claim: str,
    target_signal_text: str,
    scraped_content: str,
    scrape_ok: bool,
    scrape_error: str,
) -> str:
    if scrape_ok:
        scraped_block = scraped_content
    else:
        scraped_block = f"(scrape failed: {scrape_error or 'unknown'})"

    # Render contact LinkedIn only when present — most leads omit it, and we
    # don't want to feed the model an empty 'contact_linkedin:' line that
    # could confuse the entity check.
    contact_line = f"  contact_linkedin: {contact_linkedin}\n" if contact_linkedin else ""

    return f"""Verify whether the intent signal below is credibly supported by the supplied source.

Lead profile:
  company: {company_name}
  website: {company_website}
  company_linkedin: {company_linkedin}
{contact_line}
Intent signal to verify:
  claim:       {miner_claim}
  source_url:  {source_url}

Client's specific intent-signal target this evidence is being mapped to:
  {target_signal_text}

Scraped supplied source content (PRIMARY evidence):
{scraped_block}

Verification rules:
- Scraped supplied content above is your PRIMARY evidence.
- You have web search enabled — use it ONLY to corroborate or contradict the
  claim against the lead's actual identity. Do not substitute a different
  page as evidence when the supplied page exists.
- supported: scraped content directly proves the claim AND identity matches.
- partially_supported: scraped content supports only part of the claim.
- wrong_entity: scraped content is about a different company or person.
- contradicted: scraped content directly contradicts the claim.
- unable_to_verify: scrape failed AND web search cannot independently confirm.
- When scrape failed but credible independent web sources confirm the claim,
  return supported / partially_supported and list those web URLs in
  evidence_urls_used.
- evidence_urls_used MUST include the supplied source_url when the scrape
  succeeded.

Return only schema-valid JSON."""


# ─────────────────────────────────────────────────────────────────────
# LLM call (reuses v2's pattern for retry / 429 backoff)
# ─────────────────────────────────────────────────────────────────────
def _get_openrouter_key() -> str:
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    )


async def _llm_call(
    client: httpx.AsyncClient, model: str, prompt: str,
) -> Dict[str, Any]:
    or_key = _get_openrouter_key()
    if not or_key:
        return {"_error": "no_openrouter_key"}
    for attempt in range(3):
        try:
            r = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {or_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYS_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "intent_v3_verdict",
                            "strict": True,
                            "schema": _VERDICT_SCHEMA,
                        },
                    },
                    "temperature": 0,
                },
                timeout=TIMEOUT_SECONDS,
            )
            if r.status_code == 429:
                await asyncio.sleep(8 * (attempt + 1))
                continue
            if r.status_code != 200:
                return {"_error": f"http_{r.status_code}",
                        "_body": r.text[:400]}
            body = r.json()
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            try:
                ans = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content)
                ans = json.loads(m.group(0)) if m else {"_raw_unparsed": content[:400]}
            return {"answer": ans, "citations": body.get("citations") or [],
                    "usage": body.get("usage") or {}, "model": model}
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt == 2:
                return {"_error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(3)
    return {"_error": "retries_exhausted"}


# ─────────────────────────────────────────────────────────────────────
# Guardrail — post-process the model's verdict
# ─────────────────────────────────────────────────────────────────────
def _apply_evidence_url_guardrail(
    answer: Dict[str, Any], scrape_ok: bool, supplied_url: str,
) -> Dict[str, Any]:
    """Validate the evidence URLs the model cited.

    Rule when the scrape succeeded: the supplied URL must appear in
    evidence_urls_used.  Otherwise the model ignored the page we gave it
    and built a verdict purely from web search — downgrade to unable_to_verify
    because we cannot trust a source-grounded verdict that didn't use the
    source.  Additional web URLs cited alongside the supplied URL are
    accepted; the :online plugin naturally produces them.

    Rule when the scrape failed: accept any evidence the model found via web
    search.  This is the intentional recall-recovery path for URLs we
    couldn't fetch (e.g., Indeed search pages, Forbes paywalls).
    """
    status = answer.get("signal_status")
    if status not in ("supported", "partially_supported"):
        return answer
    evidence = [_normalize_url(u) for u in (answer.get("evidence_urls_used") or [])]
    if not evidence:
        answer["signal_status"] = "unable_to_verify"
        answer["rationale"] = (
            (answer.get("rationale") or "") +
            " [guardrail: no evidence URLs provided]"
        )
        return answer
    if scrape_ok:
        supplied_norm = _normalize_url(supplied_url)
        if supplied_norm and supplied_norm not in evidence:
            answer["signal_status"] = "unable_to_verify"
            answer["rationale"] = (
                (answer.get("rationale") or "") +
                " [guardrail: scrape succeeded but supplied URL not cited]"
            )
    return answer


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
async def verify_single_call(
    client: httpx.AsyncClient,
    *,
    company_name: str,
    company_linkedin: str,
    company_website: str,
    source_url: str,
    miner_claim: str,
    target_signal_text: str,
    contact_linkedin: str = "",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end intent-signal verification in a single LLM call.

    Pipeline: scrape (SD primary, Exa fallback) → company-name-in-scrape
    Python check → one sonar:online LLM call → evidence-URL guardrail.

    Returns:
        client_ready (bool): True iff signal_status in {supported,
            partially_supported}.  Caller (lead_scorer.py) treats this as
            the accept/reject decision.
        rejection_reason (str): empty when client_ready=True; otherwise
            the specific verdict status or stage that caused rejection.
        scrape (dict): outcome of the scrape stage — keys: ok, stage,
            content, error.
        company_check (bool | None): True if the lead's company name was
            found in the scraped text.  None when the scrape failed
            (the check is not applicable in that case).
        verdict (dict): the model's structured verdict, post-guardrail.
        llm_envelope (dict): raw _llm_call envelope (model, citations,
            usage, etc.) for debugging.
    """
    if not source_url:
        return {"client_ready": False, "rejection_reason": "no_source_url",
                "scrape": None, "company_check": None,
                "verdict": None, "llm_envelope": None}

    # 1. Scrape (SD primary + Exa fallback)
    scrape = await scrape_url(source_url)

    # 2. Deterministic company-name-in-scrape check
    company_check: Optional[bool] = None
    if scrape["ok"]:
        company_check = company_in_scrape(company_name, scrape["content"])
        if not company_check:
            return {
                "client_ready": False,
                "rejection_reason": "wrong_entity_company_not_in_scrape",
                "scrape": scrape, "company_check": False,
                "verdict": {"signal_status": "wrong_entity",
                            "rationale": "company name absent in scraped page"},
                "llm_envelope": None,
            }

    # 3. Single LLM call
    prompt = _build_prompt(
        company_name=company_name,
        company_linkedin=company_linkedin,
        company_website=company_website,
        contact_linkedin=contact_linkedin,
        source_url=source_url,
        miner_claim=miner_claim,
        target_signal_text=target_signal_text,
        scraped_content=scrape.get("content", ""),
        scrape_ok=scrape["ok"],
        scrape_error=scrape.get("error", ""),
    )
    envelope = await _llm_call(client, model or DEFAULT_MODEL, prompt)
    if envelope.get("_error"):
        return {
            "client_ready": False,
            "rejection_reason": f"llm_error:{envelope['_error']}",
            "scrape": scrape, "company_check": company_check,
            "verdict": None, "llm_envelope": envelope,
        }

    # 4. Guardrail — validate evidence URLs the model cited
    answer = envelope.get("answer") or {}
    answer = _apply_evidence_url_guardrail(answer, scrape["ok"], source_url)
    status = answer.get("signal_status")

    client_ready = status in ("supported", "partially_supported")
    return {
        "client_ready": client_ready,
        "rejection_reason": "" if client_ready else status or "no_verdict",
        "scrape": scrape, "company_check": company_check,
        "verdict": answer, "llm_envelope": envelope,
    }
