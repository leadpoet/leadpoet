"""Intent-signal validation gate.

Decides whether a miner-submitted intent signal — (URL, claim description,
matched buyer ICP signal) — is genuine, on-ICP, in-window evidence.

The gate is a four-layer pipeline.  Layers 1-3 are deterministic, run on the
caller's machine, and reject the cheap-to-detect failure modes before any
LLM cost is incurred.  Layer 4 is the authoritative judgment for signals
that survive.

    Layer 1 — structural URL check
        Rejects aggregator pages, employer templates, and repo metadata URLs
        whose path makes them incapable of carrying intent evidence.

    Layer 2 — freshness window
        Rejects evidence dated outside the time bound implied by the claim
        text (e.g. "in the last 6 months" allows up to 200 days).

    Layer 3 — anti-bot / login-wall detection
        Rejects short pages whose content is a Cloudflare challenge, login
        wall, or "page not found" disguised as a 200-response body.

    Layer 4 — strict LLM judge (Claude Sonnet 4.5)
        Structured-output judgment with five required gates: company named
        in the page, verbatim supporting quote, recency satisfied, ICP
        fulfilled, and verdict="valid".

Public API:

    run_all_prechecks(signal, page_content)
        Runs Layers 1-3 in order and returns the first rejection reason,
        or None if all pre-checks pass.

    judge_intent_signal(company, icp_signal, description, url,
                        page_content, openrouter_api_key)
        Runs Layer 4 and returns (passes, reason, raw_judge_output).
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx


# ---------------------------------------------------------------------------
# Layer 1 — structural URL check
# ---------------------------------------------------------------------------

# Path fragments that are structurally incapable of carrying intent evidence
# regardless of the claim being made.
_INVALID_URL_PATTERNS = [
    r"/alternatives(?:\b|/|\?|$)",
    r"/competitors(?:\b|/|\?|$)",
    r"indeed\.com/hire/job-description/",              # employer template
    r"github\.com/[^/]+/[^/]+/labels(?:/|$)",          # repo metadata
    r"github\.com/[^/]+/[^/]+/discussions/\d+(?:/|$)",
]
_INVALID_URL_RE = re.compile("|".join(_INVALID_URL_PATTERNS), re.IGNORECASE)


def check_url_structural_validity(url: str) -> Optional[str]:
    """Return a rejection reason if the URL path cannot be valid evidence.

    Returns None for any URL whose path is structurally acceptable; the URL
    may still fail later content-based checks.
    """
    if not url:
        return None
    match = _INVALID_URL_RE.search(url)
    if match:
        return (
            f"URL path '{match.group()}' is not a valid intent-evidence source "
            f"(aggregator / template / repo metadata)"
        )
    return None


# ---------------------------------------------------------------------------
# Layer 3 — anti-bot / login-wall detection
# ---------------------------------------------------------------------------

_ANTIBOT_PATTERNS = [
    r"access denied",
    r"verifying your connection",
    r"verifying.{0,30}browser",
    r"just a moment",
    r"enable javascript",
    r"please enable js",
    r"additional verification required",
    r"verifying you are human",
    r"sign in to (?:linkedin|see|join|view|continue)",
    r"join linkedin to",
    r"create an account to (?:see|join)",
    r"page can.?t be found",
    r"this page (?:doesn.?t|does not) exist",
    r"403\s*[-:|—]?\s*forbidden",
    r"404\s*[-:|—]?\s*(?:not\s*found|page.*not.*found)",
    r"this content isn.?t available",
]
_ANTIBOT_RE = re.compile("|".join(_ANTIBOT_PATTERNS), re.IGNORECASE)

# Pages above this length are assumed to be real content even if they happen
# to mention anti-bot phrases in passing.
_ANTIBOT_MAX_LEN = 4000


def check_antibot_wall(content: str) -> Optional[str]:
    """Return a rejection reason if the fetched page is a bot challenge or
    login wall rather than real content."""
    if not content:
        return None
    head = content[:5000].lower()
    match = _ANTIBOT_RE.search(head)
    if match and len(content) < _ANTIBOT_MAX_LEN:
        return (
            f"Page returned anti-bot / login wall (matched pattern: "
            f"{match.group()[:40]!r}) — cannot verify claim from this content"
        )
    return None


# ---------------------------------------------------------------------------
# Layer 2 — freshness window
# ---------------------------------------------------------------------------

# Maps a time-bound phrase in the claim text to its maximum evidence age in
# days.  Windows are intentionally a few days wider than the literal phrase
# to absorb typical publication and indexing lag.
_FRESHNESS_WINDOWS = {
    "in the last few weeks": 60,
    "in the last 30 days": 45,
    "in the last 60 days": 75,
    "in the last 90 days": 105,
    "in the last 6 months": 200,
    "in the last 12 months": 400,
    "in the past few weeks": 60,
    "in the past 30 days": 45,
    "in the past 60 days": 75,
    "in the past 90 days": 105,
    "in the past 6 months": 200,
    "in the past 12 months": 400,
    "last few weeks": 60,
    "last 30 days": 45,
    "last 60 days": 75,
    "last 90 days": 105,
    "last 6 months": 200,
    "last 12 months": 400,
    "past 30 days": 45,
    "past 60 days": 75,
    "past 90 days": 105,
    "past 6 months": 200,
    "past 12 months": 400,
    "recently": 180,
}


def _claim_max_age_days(claim_text: str) -> Optional[int]:
    """Return the tightest matching freshness window for the claim, or None
    if the claim has no time bound."""
    if not claim_text:
        return None
    lowered = claim_text.lower()
    best: Optional[int] = None
    for phrase, days in _FRESHNESS_WINDOWS.items():
        if phrase in lowered and (best is None or days < best):
            best = days
    return best


def _parse_signal_date(date_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 or YYYY-MM-DD date into an aware UTC datetime."""
    try:
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def check_evidence_freshness(claim_text: str,
                             signal_date: Optional[str],
                             content_found_date: Optional[str] = None
                             ) -> Optional[str]:
    """Return a rejection reason if the evidence is older than the freshness
    window implied by the claim.

    Returns None when the claim has no time bound, the date cannot be
    parsed, or the evidence falls within the window.  Unparseable dates
    fail open here — the LLM judge re-extracts the date downstream.
    """
    if not claim_text:
        return None
    max_age = _claim_max_age_days(claim_text)
    if max_age is None:
        return None

    date_str = signal_date or content_found_date
    if not date_str:
        return None

    parsed = _parse_signal_date(date_str)
    if parsed is None:
        return None

    age_days = (datetime.now(timezone.utc) - parsed).days
    if age_days > max_age:
        return (
            f"Signal date {date_str} is {age_days} days old, but claim's "
            f"freshness window allows max {max_age} days"
        )
    return None


# ---------------------------------------------------------------------------
# Pre-check aggregator
# ---------------------------------------------------------------------------

def run_all_prechecks(signal: Dict[str, Any],
                      page_content: Optional[str] = None) -> Optional[str]:
    """Run Layers 1-3 in order and return the first rejection reason.

    Args:
        signal: Signal record with at minimum a ``url`` key and either a
            ``matched_icp_signal`` or ``description`` carrying the claim text.
            ``date`` and ``content_found_date`` are consulted by the freshness
            check.
        page_content: HTML-stripped page text used by the anti-bot check.
            Optional; if omitted, the anti-bot layer is skipped.

    Returns:
        The first rejection reason as a human-readable string, or None if
        every pre-check passes.
    """
    reason = check_url_structural_validity(signal.get("url") or "")
    if reason:
        return reason

    reason = check_evidence_freshness(
        signal.get("matched_icp_signal") or signal.get("description") or "",
        signal.get("date"),
        signal.get("content_found_date"),
    )
    if reason:
        return reason

    if page_content:
        reason = check_antibot_wall(page_content)
        if reason:
            return reason

    return None


# ---------------------------------------------------------------------------
# Layer 4 — LLM judge
# ---------------------------------------------------------------------------

JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

JUDGE_SYSTEM_PROMPT = """You are a strict intent-signal auditor for a B2B sales platform. A miner submitted an intent signal — (URL, claim description, matched buyer ICP signal) — and you must judge whether the URL content genuinely supports the claim AND the claim genuinely fulfills the buyer's ICP signal.

TODAY'S DATE: {today}

You will receive:
  COMPANY:                the lead's company name
  BUYER_ICP_SIGNAL:       the exact ICP signal the miner claims this evidence satisfies
  MINER_DESCRIPTION:      what the miner says the URL shows
  URL:                    the full URL of the source (use for date extraction)
  URL_CONTENT:            the actual scraped text of the URL (HTML stripped)

Output ONLY a JSON object with these keys:

  company_named_in_page         : true if the company name explicitly appears in URL_CONTENT text. NOT just inferred from URL path.

  quote_supporting_claim        : a VERBATIM substring from URL_CONTENT (max 200 chars) that directly supports the miner's claim. Use "" if no verbatim text in the page supports the claim.

  date_extracted                : an ISO date "YYYY-MM-DD" representing when the event described in the claim occurred. Extract from (in priority order):
                                  1. Date phrase inside quote_supporting_claim
                                  2. Date in URL path (e.g. /2026/05/06/, /2024-04-11/)
                                  3. Byline / "Published on" / "Posted" line near the headline
                                  4. Use "" if no date is found.
                                  IMPORTANT: numerical IDs (e.g. LinkedIn "-activity-7445069...") are NOT dates. Do not extract dates from such IDs.

  quote_indicates_event_recency : true if date_extracted falls within the time window implied by BUYER_ICP_SIGNAL:
                                  - "in the last few weeks" / "in the past few weeks" → within 60 days of TODAY
                                  - "in the last 30 days" → within 45 days
                                  - "in the last 60 days" → within 75 days
                                  - "in the last 90 days" / "60-90 days" → within 105 days
                                  - "in the last 6 months" → within 200 days
                                  - "in the last 12 months" → within 400 days
                                  - "recently" → within 180 days
                                  - No time-bound phrase → set true (recency not required)
                                  - If date_extracted is "" AND claim has time-bound language → set false.

  fulfills_icp_signal           : "yes" if the evidence DIRECTLY satisfies BUYER_ICP_SIGNAL as stated;
                                  "partial" if related but doesn't exactly match the ICP signal's specific intent;
                                  "no" if the evidence has no relationship to BUYER_ICP_SIGNAL OR if BUYER_ICP_SIGNAL is empty/blank.

  verdict                       : "valid" if ALL: company_named_in_page=true, quote non-empty, quote_indicates_event_recency=true, fulfills_icp_signal="yes".
                                  "invalid" if any of the above fails.
                                  "uncertain" ONLY if URL_CONTENT is a login wall / anti-bot page / SPA-empty.

  reason                        : one-sentence justification (max 200 chars)

STRICT RULES (NO EXCEPTIONS):
  - quote_supporting_claim MUST be a literal substring of URL_CONTENT. Do not paraphrase. If you cannot extract a verbatim quote, set "" and verdict="invalid".
  - "Partial" ICP fit is NOT enough — reject.
  - If BUYER_ICP_SIGNAL is empty/blank, set fulfills_icp_signal="no" and verdict="invalid".
  - date_extracted must be a real date from the content. Do not invent.
  - Numerical activity-IDs in LinkedIn URLs are not dates.
  - Anti-bot / login wall pages: verdict="uncertain".

Output strict JSON only — no markdown, no commentary outside JSON."""


def _build_judge_user_prompt(company: str,
                             icp_signal: Optional[str],
                             description: str,
                             url: str,
                             content: str) -> str:
    """Render the user-message half of the judge prompt."""
    return (
        f"COMPANY: {company or ''}\n"
        f"BUYER_ICP_SIGNAL: {icp_signal or '<empty — buyer did not provide a signal>'}\n"
        f"MINER_DESCRIPTION: {description or ''}\n"
        f"URL: {url}\n\n"
        f"URL_CONTENT (first 6500 chars):\n"
        f'"""\n{(content or "")[:6500]}\n"""'
    )


# ---------------------------------------------------------------------------
# Quote-substring normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")

# Folds typographic substitutions that Sonnet sometimes introduces when
# emitting a "verbatim" quote.  NFKC normalization alone does not unify
# curly and straight quotes — they are distinct code points with no
# compatibility decomposition — so an explicit map is required.
_TYPOGRAPHIC_MAP = str.maketrans({
    "‘": "'",   # left single quotation mark
    "’": "'",   # right single quotation mark
    "‚": "'",   # single low-9 quotation mark
    "‛": "'",   # single high-reversed-9 quotation mark
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
    "„": '"',   # double low-9 quotation mark
    "‟": '"',   # double high-reversed-9 quotation mark
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",   # horizontal bar
    "−": "-",   # minus sign
    " ": " ",   # non-breaking space
    " ": " ",   # narrow no-break space
    " ": " ",   # thin space
    "​": "",    # zero-width space
    "…": "...", # horizontal ellipsis
})


def _normalize_for_substring(s: str) -> str:
    """Lowercase, decode HTML entities, apply NFKC, fold typographic chars,
    and collapse whitespace so HTML-stripping quirks don't break the
    quote-substring check."""
    if not s:
        return ""
    decoded = html.unescape(s)
    folded = unicodedata.normalize("NFKC", decoded).translate(_TYPOGRAPHIC_MAP)
    return _WHITESPACE_RE.sub(" ", folded.lower()).strip()


# ---------------------------------------------------------------------------
# Judge response evaluation
# ---------------------------------------------------------------------------

def _evaluate_judge_response(judge_output: Dict[str, Any],
                             page_content: str = "") -> "tuple[bool, str]":
    """Apply the gate rules to the model's structured response.

    Verifies in order: response shape, verdict, company presence, verbatim
    quote, quote substring match against the page, ICP fulfillment, and
    recency.  Returns ``(passes, reason)``; ``reason`` is "ok" on pass.
    """
    if not isinstance(judge_output, dict):
        return False, "judge response not a dict"
    if judge_output.get("_error"):
        return False, f"LLM error: {judge_output['_error']}"

    verdict = (judge_output.get("verdict") or "").lower()
    if verdict == "uncertain":
        return False, f"page anti-bot/unreadable: {(judge_output.get('reason') or '')[:120]}"
    if verdict == "invalid":
        return False, f"judge invalid: {(judge_output.get('reason') or '')[:120]}"

    if not judge_output.get("company_named_in_page"):
        return False, "company not named in page"

    quote = (judge_output.get("quote_supporting_claim") or "").strip()
    if not quote:
        return False, "no verbatim supporting quote"

    if page_content:
        norm_quote = _normalize_for_substring(quote)
        norm_content = _normalize_for_substring(page_content)
        if norm_quote and norm_quote not in norm_content:
            return False, (
                f"judge's quote is NOT a substring of page content "
                f"(LLM may have hallucinated): {quote[:80]!r}"
            )

    fulfills = (judge_output.get("fulfills_icp_signal") or "").lower()
    if fulfills != "yes":
        return False, f"fulfills_icp_signal={(judge_output.get('fulfills_icp_signal') or 'no')!r}"

    if not judge_output.get("quote_indicates_event_recency"):
        return False, "no date establishing claim's recency"

    return True, "ok"


# ---------------------------------------------------------------------------
# OpenRouter transport with retries
# ---------------------------------------------------------------------------

# HTTP status codes that justify a retry: transient infrastructure issues and
# rate limits.  Auth (401/403) and request-format errors (400/404/422) are
# not retried — the same payload will not succeed on a second attempt.
_RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524})

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_TIMEOUT = 60.0


def _strip_markdown_fence(msg: str) -> str:
    """Remove a ``` ... ``` wrapper around a JSON payload if the model
    emitted one despite response_format=json_object."""
    if not msg.startswith("```"):
        return msg
    body = "\n".join(line for line in msg.splitlines() if not line.startswith("```"))
    if body.lstrip().lower().startswith("json"):
        body = body.lstrip()[4:]
    return body


async def _call_openrouter_once(payload: Dict[str, Any],
                                openrouter_api_key: str,
                                client: Optional[httpx.AsyncClient] = None,
                                ) -> "tuple[Optional[Dict[str, Any]], str, bool]":
    """Issue a single OpenRouter chat-completions request asynchronously.

    Returns ``(parsed_judge_dict, error_message, retryable)``.  On success
    the dict is populated and the error message is empty.  On failure the
    dict is None, the error message describes the failure, and the
    ``retryable`` flag indicates whether retrying the same payload is
    worthwhile.
    """
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://leadpoet.ai",
        "X-Title": "Leadpoet Intent Gate",
    }

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=_OPENROUTER_TIMEOUT)

    try:
        try:
            # Explicit per-request timeout so a caller-supplied client without
            # a configured timeout cannot cause this call to hang indefinitely.
            response = await client.post(
                _OPENROUTER_URL, headers=headers, json=payload,
                timeout=_OPENROUTER_TIMEOUT,
            )
        except httpx.TimeoutException:
            return None, f"http timeout (>{_OPENROUTER_TIMEOUT:g}s)", True
        except httpx.ConnectError as e:
            return None, f"connect error: {str(e)[:120]}", True
        except httpx.HTTPError as e:
            return None, f"http error: {str(e)[:120]}", True

        if response.status_code >= 400:
            body_tail = (response.text or "")[:160]
            retryable = response.status_code in _RETRYABLE_HTTP_CODES
            return None, f"http {response.status_code}: {body_tail}", retryable

        try:
            resp = response.json()
        except json.JSONDecodeError:
            return None, f"non-JSON HTTP body: {(response.text or '')[:120]!r}", True
    finally:
        if owns_client:
            await client.aclose()

    # OpenRouter error envelope: {"error": {"code": ..., "message": ...}}
    if isinstance(resp, dict) and resp.get("error") and not resp.get("choices"):
        err = resp["error"]
        if isinstance(err, dict):
            code = err.get("code")
            err_msg = err.get("message") or str(err)
        else:
            code = None
            err_msg = str(err)
        retryable = (code in _RETRYABLE_HTTP_CODES) or (code is None)
        return None, f"openrouter error code={code}: {str(err_msg)[:120]}", retryable

    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        return None, "openrouter returned empty choices array", True

    first = choices[0] or {}
    content = ((first.get("message") or {}).get("content") or "").strip()
    if not content:
        return None, "model returned empty content", True

    content = _strip_markdown_fence(content)

    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        return None, f"model returned non-JSON: {content[:120]!r}", True

    if not isinstance(out, dict):
        return None, f"model returned non-object JSON: {type(out).__name__}", False

    return out, "", False


# ---------------------------------------------------------------------------
# Public judge entry point
# ---------------------------------------------------------------------------

async def judge_intent_signal(company: str,
                              icp_signal: Optional[str],
                              description: str,
                              url: str,
                              page_content: str,
                              openrouter_api_key: str,
                              today: Optional[str] = None,
                              model: str = JUDGE_MODEL,
                              max_attempts: int = 3,
                              client: Optional[httpx.AsyncClient] = None,
                              ) -> "tuple[bool, str, Dict[str, Any]]":
    """Run the strict LLM judge on a single signal.

    Args:
        company: Lead company name shown to the judge.
        icp_signal: The buyer's ICP signal the miner claims to satisfy.
        description: The miner's natural-language description of the evidence.
        url: Source URL.
        page_content: HTML-stripped body of ``url``.  Truncated to 6500
            characters for the prompt; the full string is used for the
            substring check.
        openrouter_api_key: Credential for the OpenRouter chat-completions API.
        today: Override for the date interpolated into the system prompt.
            Defaults to the current UTC date.
        model: OpenRouter model identifier.  Defaults to Sonnet 4.5.
        max_attempts: Total attempts including the initial call.  Defaults to 3.
        client: Optional ``httpx.AsyncClient`` reused across calls.  When
            None, the function creates and closes its own client per call.

    Returns:
        ``(passes, reason, raw_judge_output)``.  ``passes`` is True only if
        every gate in :func:`_evaluate_judge_response` passes.  ``reason``
        is suitable for logging and dashboard surfacing.

    Early-exit cases (no LLM call):
        - Missing API key
        - Empty / blank ``icp_signal`` — the judge would always return invalid
        - ``page_content`` shorter than 100 characters

    Retry policy:
        Up to ``max_attempts`` calls with exponential backoff (0.5s, 1s, ...).
        Retries on transport errors, 5xx/429 responses, empty bodies, and
        model-side JSON parse failures.  Auth and non-429 4xx are not retried.
        On exhausted retries the function fails closed: returns
        ``(False, error_message, {})``.
    """
    if not (openrouter_api_key and openrouter_api_key.strip()):
        return False, "missing OpenRouter API key", {}
    if not (icp_signal and icp_signal.strip()):
        return False, "buyer ICP signal empty (no signal to fulfill)", {}
    if not page_content or len(page_content) < 100:
        return False, "page empty/unreadable", {}

    today_str = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = JUDGE_SYSTEM_PROMPT.replace("{today}", today_str)
    user_prompt = _build_judge_user_prompt(company, icp_signal, description, url, page_content)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 1200,
    }

    judge_output: Optional[Dict[str, Any]] = None
    last_error = "no attempts made"

    # When the caller did not supply a client, allocate one for the full retry
    # loop instead of creating + tearing down a fresh connection per attempt.
    owns_client = client is None
    local_client = client or httpx.AsyncClient(timeout=_OPENROUTER_TIMEOUT)
    try:
        for attempt in range(max_attempts):
            result, err, retryable = await _call_openrouter_once(
                payload, openrouter_api_key, client=local_client,
            )
            if result is not None:
                judge_output = result
                break
            last_error = err
            if not retryable or attempt == max_attempts - 1:
                break
            await asyncio.sleep(0.5 * (2 ** attempt))
    finally:
        if owns_client:
            await local_client.aclose()

    if judge_output is None:
        return False, f"LLM transport error: {last_error}", {}

    passes, reason = _evaluate_judge_response(judge_output, page_content=page_content)
    return passes, reason, judge_output
