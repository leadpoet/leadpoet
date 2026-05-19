"""Intent pre-check — Gemini 2.5 Flash-Lite semantic gate before Tier 2/3.

Tier 1.7 of the fulfillment scoring pipeline.  Runs AFTER Tier 1.5
(deterministic + geo / sub-industry LLM gates) and BEFORE Tier 2 (Sonar +
Gemini company / required-attribute verification) and Tier 3 (three-stage
URL verifier).

For each (miner_signal, mapped_icp_signal) pair, asks Gemini:
  "Does the miner's claim semantically satisfy the target ICP signal,
   treating miner_source + miner_url as evidence for any venue qualifier?"

If ALL of a lead's signals fail the pre-check, the lead is rejected at
Tier 1.7 with ``failure_reason="intent_precheck_no_match"`` — saving the
~$0.05-0.30 of downstream Sonar/Gemini cost on a hopeless lead.  Signals
that individually fail are tagged so Tier 3 can skip the expensive
three-stage verifier on them (lead_scorer reads the parallel verdicts
list passed in via scoring.py).

Failure mode: fail-OPEN.  Any HTTP non-200 (after 429 retries), timeout,
unparseable response, or unexpected exception returns verdict=True for
that signal so the three-stage verifier — which is authoritative on URL
content — still gets a chance.  Failing-closed during a transient rate
limit would reject every lead in flight; that's worse than briefly
letting the cheap gate be a no-op.

Activated via the INTENT_PRECHECK_ENABLED env flag (default off).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Config (env-overridable to match three-stage verifier conventions)
# ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("INTENT_PRECHECK_MODEL", "google/gemini-2.5-flash-lite")
TIMEOUT_SECONDS = int(os.environ.get("INTENT_PRECHECK_TIMEOUT_S", "30"))
NUM_RETRIES = int(os.environ.get("INTENT_PRECHECK_RETRIES", "3"))
CONCURRENCY = int(os.environ.get("INTENT_PRECHECK_CONCURRENCY", "8"))

_SYS_MESSAGE = "You are a strict B2B intent-signal semantic-match judge. Return JSON only."


def _get_openrouter_key() -> str:
    """Mirror of qualification/scoring/intent_verification_three_stage.py.
    Keep the resolution order identical so an operator can swap keys in
    one place and have it apply to both verifiers."""
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    )


# ─────────────────────────────────────────────────────────────────────
# Prompt + response schema
# ─────────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """Judge whether a miner's evidence directly satisfies a target intent signal.

INPUTS
  miner_claim:       "{miner_claim}"
  miner_source:      "{miner_source}"
  miner_url:         "{miner_url}"
  target_icp_signal: "{target_icp_signal}"

POLICY
  Answer "yes" only if ALL rules below are satisfied for THIS pair.
  Answer "no" if ANY rule rejects.

RULES
  1. DIRECT SUBSTANCE MATCH. The claim must directly state the fact the target
     asks for. Inference, implication, or extrapolation does NOT count. Phrases
     like "indicates potential for", "suggests interest in", "implies need for",
     "could mean", "likely uses" are inference — reject.

  2. SPECIFICITY. If the target names a specific value, category, or range, the
     claim must match it specifically. Examples of mismatches: "Series A" vs
     "Series B/C/D"; "$300M raised" alone (no series named) vs "Series B/C/D";
     "import activity" alone vs "import from Asia/Europe/LatAm specifically".

  3. NEGATION CONTRADICTS. A claim asserting the opposite or zero of what the
     target asks for is a reject. Examples: "0 open positions" vs "active job
     postings"; "no longer hiring" vs "actively hiring"; "404 page" / "job no
     longer open" vs "active listings".

  4. TOPIC MATCH. Claim and target must be about the same topic. Selling a
     product is NOT the same topic as hiring for that role. A CEO keynote is
     NOT the same topic as job postings.

  5. NO CROSS-LANGUAGE / CROSS-TITLE ALIASING. A role name in another language
     or a different job title is NOT a direct match unless the target explicitly
     allows the alias.

  6. RECENCY. If the target requires a time window ("within the last 6 months",
     "in the last 30 days"), the claim must mention an event that fits that
     window, or the miner_url/miner_source must clearly support it. Today's
     date is {today}; treat earlier-than-window claims as reject.

  7. VENUE QUALIFIER IS SATISFIED BY source/url, NOT by claim text. If the
     target names specific evidence venues (e.g., "on LinkedIn or job boards",
     "in a press release", "on the careers page", "on G2", "via Twitter"),
     judge venue from miner_source + miner_url — NOT by requiring the claim
     text to repeat the venue language. The miner is not expected to restate
     the venue in narrative form.

     Example (the only one needed):
       target:  "Hiring SDRs on LinkedIn or job boards"
       claim:   "Company is hiring SDRs"
       source:  "job_board"
       url:     "lever.co/acme/..."
       → yes (rule #1 satisfied by claim, rule #7 satisfied by source/url)

  8. WHEN AMBIGUOUS, REJECT. Strict mode: if you cannot map the claim onto
     the target unambiguously under rules 1-7, answer "no".

OUTPUT
  JSON only, schema:
  {{"verdict": "yes" | "no", "reason": "<one sentence naming the rule(s)>"}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "reason":  {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def _build_prompt(
    miner_claim: str,
    miner_source: str,
    miner_url: str,
    target_icp_signal: str,
    today_iso: str,
) -> str:
    return _PROMPT_TEMPLATE.format(
        miner_claim=miner_claim,
        miner_source=miner_source,
        miner_url=miner_url,
        target_icp_signal=target_icp_signal,
        today=today_iso,
    )


# ─────────────────────────────────────────────────────────────────────
# OpenRouter call with 429 + network retry, fail-open on exhaustion
# Matches qualification/scoring/intent_verification_three_stage.py
# ─────────────────────────────────────────────────────────────────────
async def _call_openrouter(
    http_client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
) -> Tuple[Optional[dict], str]:
    """Call Gemini via OpenRouter with retries.

    Returns ``(parsed_json, error_kind)``:
      - on success:                ``({"verdict": ..., "reason": ...}, "")``
      - on transient failure:      ``(None, "<error_kind>")`` — caller fails-open
    """
    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYS_MESSAGE},
            {"role": "user",   "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": False, "schema": _RESPONSE_SCHEMA},
        },
    }
    last_err = "retries_exhausted"
    for attempt in range(NUM_RETRIES):
        try:
            r = await http_client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=TIMEOUT_SECONDS,
            )
            if r.status_code == 429:
                wait_s = 8 * (attempt + 1)
                logger.warning(
                    "Intent pre-check 429 rate-limited  attempt=%d/%d  "
                    "sleeping=%ds  model=%s",
                    attempt + 1, NUM_RETRIES, wait_s, MODEL,
                )
                await asyncio.sleep(wait_s)
                last_err = "rate_limited_429"
                continue
            if r.status_code != 200:
                logger.warning(
                    "Intent pre-check HTTP %s  body=%s  model=%s — fail-open",
                    r.status_code, r.text[:200], MODEL,
                )
                return None, f"http_{r.status_code}"
            resp = r.json()
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content or "")
                if not m:
                    logger.warning(
                        "Intent pre-check unparseable response  content=%r  model=%s — fail-open",
                        (content or "")[:200], MODEL,
                    )
                    return None, "unparseable"
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    logger.warning(
                        "Intent pre-check regex-extracted JSON still invalid  "
                        "content=%r  model=%s — fail-open",
                        (content or "")[:200], MODEL,
                    )
                    return None, "unparseable"
            return parsed, ""
        except httpx.HTTPError as e:
            # Root of all httpx transport/timeout/network/protocol/proxy
            # errors — retry on any of them.  Non-httpx exceptions (e.g.
            # NameError) escape so programming bugs stay visible.
            last_err = f"{type(e).__name__}"
            if attempt == NUM_RETRIES - 1:
                logger.warning(
                    "Intent pre-check %s after %d attempts  model=%s — fail-open",
                    last_err, NUM_RETRIES, MODEL,
                )
                return None, last_err
            await asyncio.sleep(3)
        except ValueError as e:
            # r.json() raises ValueError (json.JSONDecodeError) on a non-JSON
            # body.  Response corruption won't fix itself — fail-open now.
            logger.warning(
                "Intent pre-check JSON decode error: %s  model=%s — fail-open",
                e, MODEL,
            )
            return None, "json_decode_error"
    logger.warning(
        "Intent pre-check %s after %d attempts  model=%s — fail-open",
        last_err, NUM_RETRIES, MODEL,
    )
    return None, last_err


# ─────────────────────────────────────────────────────────────────────
# Public API — batch precheck across one lead's intent signals
# ─────────────────────────────────────────────────────────────────────
async def precheck_lead_signals(
    http_client: httpx.AsyncClient,
    lead_signals: list,                       # List[IntentSignal] — duck-typed
    icp_intent_signal_texts: List[str],
    api_key: Optional[str] = None,
    today_iso: Optional[str] = None,
) -> List[bool]:
    """Run the Gemini pre-check across one lead's intent signals.

    Returns a list of booleans, ORDER-MATCHED to ``lead_signals``:
      True  = signal passes the semantic check (proceed to Tier 2/3)
      False = signal fails (Tier 3 will skip the three-stage verifier
              on this signal; if every signal is False, scoring.py
              rejects the whole lead at Tier 1.7)

    Signals with ``matched_icp_signal`` unset or out-of-range are marked
    False immediately (no LLM call wasted — Tier 3's Gate 0 would have
    rejected them anyway, see lead_scorer._score_single_intent_signal).

    Any OpenRouter error after retries returns True for that signal
    (fail-open) — the three-stage verifier remains authoritative.
    """
    if not lead_signals:
        return []

    key = api_key or _get_openrouter_key()
    if not key:
        logger.warning(
            "Intent pre-check: no OPENROUTER_API_KEY set — fail-open  "
            "(all %d signal(s) marked PASS)",
            len(lead_signals),
        )
        return [True] * len(lead_signals)

    today = today_iso or date.today().isoformat()
    icp_texts = list(icp_intent_signal_texts or [])
    num_icp = len(icp_texts)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(idx: int, signal) -> Tuple[int, bool, str]:
        try:
            matched_idx = getattr(signal, "matched_icp_signal", -1)
            if not isinstance(matched_idx, int) or matched_idx < 0 or matched_idx >= num_icp:
                return idx, False, f"matched_icp_signal_out_of_range(={matched_idx})"

            source_attr = getattr(signal, "source", "")
            source_str = source_attr.value if hasattr(source_attr, "value") else str(source_attr)
            miner_claim = getattr(signal, "description", "") or ""
            miner_url   = getattr(signal, "url", "") or ""
            target_text = icp_texts[matched_idx] or ""

            async with sem:
                parsed, err = await _call_openrouter(
                    http_client, key,
                    _build_prompt(miner_claim, source_str, miner_url, target_text, today),
                )
            if parsed is None:
                return idx, True, f"fail_open({err})"
            verdict = (parsed.get("verdict") or "").strip().lower()
            reason = parsed.get("reason") or ""
            # Only "no" is a reject — every other value (yes, empty, garbage,
            # truncated) falls through to PASS so the three-stage verifier
            # downstream stays authoritative on edge cases.
            return idx, verdict != "no", reason
        except Exception as e:
            # Isolation: any uncaught error in one signal must NOT break the
            # gather for the other signals on this lead.  Fail-open per signal.
            logger.warning(
                "Intent pre-check signal[%d] unexpected %s: %s — fail-open",
                idx, type(e).__name__, e,
            )
            return idx, True, f"fail_open(unexpected:{type(e).__name__})"

    results: List[Tuple[int, bool, str]] = await asyncio.gather(
        *[_one(i, s) for i, s in enumerate(lead_signals)]
    )
    results.sort(key=lambda t: t[0])
    verdicts = [ok for _, ok, _ in results]

    # Mirrors three-stage's "Intent signal three-stage ACCEPT/REJECT" log shape.
    for idx, ok, reason in results:
        logger.info(
            "Intent pre-check signal[%d] %s  reason=%r  model=%s",
            idx, "PASS" if ok else "REJECT", reason[:160], MODEL,
        )
    return verdicts
