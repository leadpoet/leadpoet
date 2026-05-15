"""
Two-prompt intent signal verifier.

Replaces the single-LLM grounding+scoring call in
``_score_single_intent_signal`` with a two-step pipeline whose stages are
independently swappable and individually testable:

  PROMPT 1 — Source-link grounding.  Sends the miner-supplied source URL,
    the lead's LinkedIn + company website, and the miner's claim to
    Perplexity Sonar (default; swappable via env).  Sonar must decide:
      (a) does the source URL describe the SAME company as the LinkedIn +
          company website?  (same_company_check)
      (b) does the source URL literally support the miner's claim?
          (status)
    Co-founder's prompt template is used verbatim (without the signal-
    match addendum) — the addendum lives in Prompt 2.

  PROMPT 2 — Client-signal match.  Takes Prompt 1's ``confirmed_details``
    plus the SINGLE client-listed intent signal that the miner declared
    (via the new ``IntentSignal.matched_icp_signal`` index field) and
    asks Claude Sonnet 4.5 whether the verified claim semantically
    satisfies THAT specific signal.  The miner already declared which
    signal they think they're proving, so Prompt 2 just decides yes/no
    for that one signal — not "match against all signals" (that's the
    old conflated flow).

A signal is "client-ready" iff BOTH return positive results:
  Prompt 1: status in {Accurate, Partially accurate} AND same_company_check == Pass
  Prompt 2: matches == true

Both calls are NATIVE structured output (no Claude wrapper around Sonar
or vice versa).  Each model produces its own JSON object via its native
``response_format`` / ``outputSchema``.

Env vars:
  ``OPENROUTER_API_KEY`` (or ``FULFILLMENT_OPENROUTER_API_KEY``) — required
  ``INTENT_GROUNDING_MODEL`` (default: ``perplexity/sonar``)
  ``INTENT_SIGNAL_MATCH_MODEL`` (default: ``anthropic/claude-sonnet-4.5``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GROUNDING_MODEL = os.environ.get("INTENT_GROUNDING_MODEL", "perplexity/sonar")
DEFAULT_SIGNAL_MATCH_MODEL = os.environ.get("INTENT_SIGNAL_MATCH_MODEL", "anthropic/claude-sonnet-4.5")
TIMEOUT_SECONDS = 120


# ─────────────────────────────────────────────────────────────────────
# Prompt 1: Source-link grounding (co-founder's prompt verbatim)
# ─────────────────────────────────────────────────────────────────────

GROUNDING_PROMPT_TEMPLATE = """Use the LinkedIn profile and company site to identify the exact company, then verify the provided details using the source link.

Company LinkedIn: {company_linkedin}
Company site: {company_website}
Source link: {source_url}
Details to verify: {details_to_verify}

First confirm the source link is about the same company by comparing it to the LinkedIn profile and company site. Check name, domain, product, location, industry, and description.

Important:
The source link should clearly reference the same company or its official domain.
Similar name alone is not enough.
If the source link appears to describe a different company, or the match is unclear, return Unable to verify.
Only validate details that are directly supported by the source link.

Return:
Status: Accurate / Partially accurate / Inaccurate / Unable to verify
Same-company check: Pass / Fail / Unclear
Confirmed details
Discrepancies or unsupported claims
Corrected details, with source links

Do not guess, infer, or validate details for a different company with the same/similar name.

Respond ONLY in JSON matching the schema. No markdown."""


GROUNDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "same_company_check", "confirmed_details", "discrepancies", "corrected_details"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["Accurate", "Partially accurate", "Inaccurate", "Unable to verify"],
            "description": "Is the miner's claim supported by content on the source link?",
        },
        "same_company_check": {
            "type": "string",
            "enum": ["Pass", "Fail", "Unclear"],
            "description": "Is the source link unambiguously about the target company at the LinkedIn URL + company site?",
        },
        "confirmed_details": {
            "type": "string",
            "description": "Which claim elements are verifiable from the source link (quote verbatim where possible).",
        },
        "discrepancies": {
            "type": "string",
            "description": "Claim elements the source link does NOT support, contradicts, or fails to address.",
        },
        "corrected_details": {
            "type": "string",
            "description": "If the source link supports a related but different claim about the target company, state it here. Empty if N/A.",
        },
    },
}


def _build_grounding_prompt(
    company_linkedin: str, company_website: str, source_url: str, details_to_verify: str
) -> str:
    return GROUNDING_PROMPT_TEMPLATE.format(
        company_linkedin=company_linkedin or "(none)",
        company_website=company_website or "(none)",
        source_url=source_url or "(none)",
        details_to_verify=(details_to_verify or "")[:2000],
    )


# ─────────────────────────────────────────────────────────────────────
# Prompt 2: Client-signal match (against the ONE signal the miner declared)
# ─────────────────────────────────────────────────────────────────────

SIGNAL_MATCH_PROMPT_TEMPLATE = """You are checking whether a VERIFIED business fact satisfies a buyer's SPECIFIC intent signal.

VERIFIED CLAIM (already confirmed against the source URL):
{confirmed_details}

DISCREPANCIES (claim elements the source URL did NOT support, included for context only):
{discrepancies}

CLIENT'S TARGET INTENT SIGNAL (this is the specific signal the lead is being matched against):
{target_signal_text}

Does the VERIFIED CLAIM semantically satisfy the TARGET INTENT SIGNAL?

A claim "matches" only if a reasonable salesperson, reading the verified claim, would tell the client "yes, this company exhibits the specific buying behavior you asked for." Mere topical relatedness is NOT enough — the claim must actually demonstrate the buyer's named signal.

Be strict about recency windows when the signal includes one (e.g. "in the last 6 months", "within 60 days"). A claim from 5 years ago does NOT satisfy "in the last 12 months" even if the underlying event type is correct.

Examples:
- Verified: "Company X raised $25M Series B in Jan 2026"
  Target: "Raised funds in the last 3 months"
  → matches: true (specific funding event, recency window OK)

- Verified: "Company X is a SaaS fintech company"
  Target: "Raised funds in the last 3 months"
  → matches: false (industry descriptor, not a funding event)

- Verified: "Company X opened a Tokyo office in Jan 2026"
  Target: "Expanded into new regional or international content distribution"
  → matches: false (real-estate expansion is unrelated to content distribution)

- Verified: "Company X press release dated 2018-03-02 announced expansion"
  Target: "Posted press releases about regional expansion within the last 12 months"
  → matches: false (event type OK but recency window failed)

Respond ONLY in JSON matching the schema. No markdown."""


SIGNAL_MATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["matches", "reasoning"],
    "properties": {
        "matches": {
            "type": "boolean",
            "description": "True iff the verified claim semantically satisfies the target signal.  Recency windows must be honored.",
        },
        "reasoning": {
            "type": "string",
            "description": "One-sentence justification for the matches verdict.",
        },
    },
}


def _build_signal_match_prompt(confirmed_details: str, discrepancies: str, target_signal_text: str) -> str:
    return SIGNAL_MATCH_PROMPT_TEMPLATE.format(
        confirmed_details=(confirmed_details or "(empty)")[:1500],
        discrepancies=(discrepancies or "(none)")[:800],
        target_signal_text=(target_signal_text or "(empty)")[:600],
    )


# ─────────────────────────────────────────────────────────────────────
# OpenRouter shim — works for both Sonar and Claude (both are
# OpenAI-compatible chat completions endpoints behind OpenRouter)
# ─────────────────────────────────────────────────────────────────────

def _get_openrouter_key() -> str:
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    )


async def _openrouter_json_call(
    client: httpx.AsyncClient, model: str, prompt: str, schema: Dict[str, Any], schema_name: str,
) -> Dict[str, Any]:
    """Single OpenAI-compatible chat completion that enforces a JSON
    schema via ``response_format``. Both Sonar and Claude support this
    through OpenRouter."""
    or_key = _get_openrouter_key()
    if not or_key:
        return {"_error": "no_openrouter_key"}
    sys_msg = (
        "You are a strict B2B intent-signal verifier. Respond ONLY with a JSON "
        "object matching the json_schema enforced by response_format. No markdown."
    )
    for attempt in range(3):
        try:
            r = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                    },
                    "temperature": 0,
                },
                timeout=TIMEOUT_SECONDS,
            )
            if r.status_code == 429:
                await asyncio.sleep(8 * (attempt + 1))
                continue
            if r.status_code != 200:
                return {"_error": f"http_{r.status_code}", "_body": r.text[:400]}
            try:
                body = r.json()
            except Exception as je:
                return {"_error": f"json_decode: {je}", "_body": r.text[:400]}
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            try:
                ans = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content)
                ans = json.loads(m.group(0)) if m else {"_raw_unparsed": content[:400]}
            return {
                "answer": ans,
                "citations": body.get("citations") or [],
                "usage": body.get("usage") or {},
                "model": model,
            }
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt == 2:
                return {"_error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(3)
    return {"_error": "retries_exhausted"}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

async def grounding_check(
    client: httpx.AsyncClient,
    *,
    company_linkedin: str,
    company_website: str,
    source_url: str,
    details_to_verify: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Prompt 1 — source-URL grounding.  Returns the raw
    ``_openrouter_json_call`` envelope; caller inspects ``["answer"]``
    for the structured GROUNDING_SCHEMA result."""
    prompt = _build_grounding_prompt(company_linkedin, company_website, source_url, details_to_verify)
    return await _openrouter_json_call(
        client, model or DEFAULT_GROUNDING_MODEL, prompt, GROUNDING_SCHEMA, "grounding_verdict",
    )


async def signal_match_check(
    client: httpx.AsyncClient,
    *,
    confirmed_details: str,
    discrepancies: str,
    target_signal_text: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Prompt 2 — client-signal match against the SINGLE signal the
    miner declared via ``IntentSignal.matched_icp_signal``."""
    prompt = _build_signal_match_prompt(confirmed_details, discrepancies, target_signal_text)
    return await _openrouter_json_call(
        client, model or DEFAULT_SIGNAL_MATCH_MODEL, prompt, SIGNAL_MATCH_SCHEMA, "signal_match_verdict",
    )


async def verify_v2(
    client: httpx.AsyncClient,
    *,
    company_linkedin: str,
    company_website: str,
    source_url: str,
    miner_claim: str,
    target_signal_text: str,
    grounding_model: Optional[str] = None,
    signal_match_model: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end 2-prompt verifier.  Returns a dict with:
        client_ready: bool          — True iff Prompt 1 passes AND Prompt 2 matches
        rejection_reason: str       — Empty when client_ready=True; otherwise the
                                       specific step that failed
        prompt1: full ``grounding_check`` envelope
        prompt2: full ``signal_match_check`` envelope (or None if skipped)

    Short-circuits Prompt 2 when Prompt 1 has already disqualified the
    signal (same_company_check != Pass OR status in {Inaccurate, Unable to verify}).
    """
    p1 = await grounding_check(
        client,
        company_linkedin=company_linkedin, company_website=company_website,
        source_url=source_url, details_to_verify=miner_claim, model=grounding_model,
    )
    if p1.get("_error"):
        return {"client_ready": False, "rejection_reason": f"grounding_error:{p1['_error']}", "prompt1": p1, "prompt2": None}
    a1 = p1.get("answer") or {}
    if a1.get("same_company_check") != "Pass":
        return {"client_ready": False, "rejection_reason": "same_company_check_failed", "prompt1": p1, "prompt2": None}
    if a1.get("status") not in ("Accurate", "Partially accurate"):
        return {"client_ready": False, "rejection_reason": f"grounding_status_{a1.get('status')}", "prompt1": p1, "prompt2": None}

    confirmed = a1.get("confirmed_details") or ""
    discrepancies = a1.get("discrepancies") or ""
    p2 = await signal_match_check(
        client,
        confirmed_details=confirmed, discrepancies=discrepancies,
        target_signal_text=target_signal_text, model=signal_match_model,
    )
    if p2.get("_error"):
        return {"client_ready": False, "rejection_reason": f"signal_match_error:{p2['_error']}", "prompt1": p1, "prompt2": p2}
    a2 = p2.get("answer") or {}
    if a2.get("matches") is not True:
        return {"client_ready": False, "rejection_reason": "signal_did_not_match", "prompt1": p1, "prompt2": p2}

    return {"client_ready": True, "rejection_reason": "", "prompt1": p1, "prompt2": p2}
