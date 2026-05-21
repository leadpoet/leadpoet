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

A separate free deterministic URL pre-filter (no LLM cost) runs INSIDE
this module before the Gemini call, gated by INTENT_URL_PREFILTER_ENABLED.  It
catches the obvious bare-LinkedIn-company-page class of bad URLs on
hiring and funding claims without spending any LLM call.  See
``_check_url_pre_filter`` below.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Config (env-overridable to match three-stage verifier conventions)
# ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _env_int(name: str, default: int) -> int:
    """``int(os.environ.get(name, default))`` crashes when the env var is set
    to an empty string — which is exactly how the container deploy script
    passes optional overrides (``-e VAR="${VAR:-}"``).  Use ``or default``
    after stripping so empty / whitespace / missing all collapse to default."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw or default


MODEL = _env_str("INTENT_PRECHECK_MODEL", "google/gemini-2.5-flash-lite")
TIMEOUT_SECONDS = _env_int("INTENT_PRECHECK_TIMEOUT_S", 30)
NUM_RETRIES = _env_int("INTENT_PRECHECK_RETRIES", 3)
CONCURRENCY = _env_int("INTENT_PRECHECK_CONCURRENCY", 8)

_SYS_MESSAGE = "You are a strict B2B intent-signal semantic-match judge. Return JSON only."


# ─────────────────────────────────────────────────────────────────────
# URL pre-filter (free deterministic check, runs BEFORE the LLM substance
# check inside precheck_lead_signals._one()).  Only fires for HIRING and
# FUNDING claims, where the URL itself can deterministically be classified
# as inadequate evidence.  Other claim types (geography, product launch,
# leadership posts, etc.) skip this check entirely — they go straight to
# the LLM substance check.
#
# Returns 'reject' (skip LLM, fail signal) or 'pass' (defer to LLM).
# NEVER returns 'accept' — substance check still runs on every 'pass'.
#
# Why: bare LinkedIn company pages (e.g. linkedin.com/company/<slug>) do
# not show job listings — they sit on a separate /jobs subpage — so the
# client cannot verify a hiring claim by clicking the URL.  Same for
# funding claims citing a bare company profile.  Catch these without
# paying for an LLM call.
#
# Activated via INTENT_URL_PREFILTER_ENABLED env flag (default off).
# ─────────────────────────────────────────────────────────────────────

URL_PREFILTER_ENABLED = (
    os.environ.get("INTENT_URL_PREFILTER_ENABLED", "false").strip().lower() == "true"
)

_HIRING_RE = re.compile(
    r"\b("
    r"hir(?:e|es|ing|ed)|"
    r"recruit(?:s|ing|er|ers|ed|ment)?|"
    r"job\s+(?:post(?:s|ing|ings)?|listing(?:s)?|opening(?:s)?|vacanc(?:y|ies))|"
    r"open\s+(?:position(?:s)?|role(?:s)?|vacanc(?:y|ies))|"
    r"vacanc(?:y|ies)|"
    r"career\s+opportunit(?:y|ies)|"
    r"actively\s+(?:hiring|recruiting)"
    r")\b",
    re.IGNORECASE,
)

_FUNDING_RE = re.compile(
    r"\b("
    r"series\s+[a-fz]\b|"
    r"seed\s+(?:funding|round)|"
    r"raised\s+\$|"
    r"closed\s+(?:a\s+)?(?:series|seed|growth|pre-?ipo)|"
    r"funding\s+round|"
    r"secured\s+\$|"
    r"announced\s+(?:a\s+)?(?:series|seed|funding)"
    r")\b",
    re.IGNORECASE,
)

# LinkedIn URL patterns.  All match against the lower-cased URL path.
_LINKEDIN_JOBS_PREFIX = "/jobs"                       # any /jobs/* path = listing or search
_LINKEDIN_COMPANY_JOBS_RE = re.compile(r"^/company/[^/]+/jobs")  # /company/<slug>/jobs subpage
_LINKEDIN_BARE_COMPANY_PAGE_RE = re.compile(r"^/company/[^/]+/?$")  # bare /company/<slug> — static profile, no event-specific evidence
_LINKEDIN_PERSONAL_PROFILE_PREFIX = "/in/"            # /in/<slug> personal profile


def _classify_target_type(target_text: str) -> Optional[str]:
    """Classify the TARGET ICP signal topic from target text only.

    Returns 'HIRING' | 'FUNDING' | None.  Used to pick which URL pre-filter
    rules apply.  Deliberately ignores the miner's claim text — the target
    is the source of truth for what topic the buyer is asking about.
    """
    t = target_text or ""
    if _FUNDING_RE.search(t):
        return "FUNDING"
    if _HIRING_RE.search(t):
        return "HIRING"
    return None


def _classify_claim_evidence_type(claim_text: str) -> Optional[str]:
    """Classify the CLAIM (miner-supplied evidence) topic from claim text only.

    Returns 'HIRING' | 'FUNDING' | None.  Used to detect topic-mismatch
    rejects: if the target is FUNDING but the claim text matches HIRING,
    the evidence is for the wrong topic regardless of any URL.
    """
    c = claim_text or ""
    if _FUNDING_RE.search(c):
        return "FUNDING"
    if _HIRING_RE.search(c):
        return "HIRING"
    return None


def _check_intent_url_evidence_quality(
    url: str,
    target_type: Optional[str],
) -> Tuple[str, str]:
    """Free deterministic URL evidence quality check.

    Runs two layers, both deterministic and zero-cost:

    1) UNIVERSAL rules (apply regardless of ``target_type``)
       - empty URL → reject
       - bare LinkedIn company page (``linkedin.com/company/<slug>``)
         → reject.  A bare company page is a static profile — no jobs,
         no posts, no event-specific evidence — so it cannot
         substantively back ANY intent claim, regardless of topic.
         Production audit (2026-05-21): 71% of intent URLs on the Mexican
         construction chain and 42% on the Spanish creatives chain were
         bare LinkedIn slugs masquerading as evidence.

    2) TARGET-TYPE-SPECIFIC LinkedIn rules (only when ``target_type`` is set)
       - HIRING target: require /jobs/* or /company/<slug>/jobs path;
         any other LinkedIn path lacks visible hiring evidence → reject.
       - FUNDING target: reject personal profiles (/in/<slug>) — a
         personal page doesn't substantiate company-level funding events.

    For any URL not caught by either layer, returns ``('pass', reason)`` so
    the LLM substance check downstream remains authoritative.

    Args:
        url: the miner's submitted URL for this signal.
        target_type: 'HIRING' | 'FUNDING' | None.  None disables the
                     target-specific layer; the universal layer always
                     applies.

    Returns:
        ('reject', short_reason)  — URL is inadequate evidence; skip the
                                    LLM substance check and fail this signal.
        ('pass',   short_reason)  — defer to the LLM substance check; this
                                    function never says 'accept'.
    """
    if not url:
        return ("reject", "empty_url")

    parsed = urlparse(url.lower())
    host = parsed.hostname or ""
    path = parsed.path or ""

    # ── Layer 1 (UNIVERSAL): bare LinkedIn company pages have no
    # event-specific content regardless of what the miner is claiming.
    if "linkedin.com" in host and _LINKEDIN_BARE_COMPANY_PAGE_RE.match(path):
        return ("reject", "bare_linkedin_company_page")

    # ── Layer 2 (TARGET-TYPE-SPECIFIC): only fires when target topic
    # is known.  No target_type → skip and let the LLM handle it.
    if target_type == "HIRING":
        if "linkedin.com" in host:
            # Acceptable LinkedIn URLs for hiring evidence:
            #   /jobs/*                       — specific listings, search, collections
            #   /company/<slug>/jobs[/...]    — company-scoped jobs listing index
            if path.startswith(_LINKEDIN_JOBS_PREFIX):
                return ("pass", "linkedin_jobs_path")
            if _LINKEDIN_COMPANY_JOBS_RE.match(path):
                return ("pass", "linkedin_company_jobs_subpage")
            # Everything else on LinkedIn lacks visible job evidence for
            # a hiring claim (bare /company/<slug> already rejected above;
            # /company/<slug>/{about,people,life,...}, /in/<slug>,
            # /posts/<id>, /pulse/<slug>, /feed/*, /showcase/*, etc.).
            return ("reject", "linkedin_no_visible_job_evidence")
        # Non-LinkedIn URLs for hiring claims — defer to LLM.
        return ("pass", "non_linkedin_defer")

    if target_type == "FUNDING":
        if "linkedin.com" in host:
            if path.startswith(_LINKEDIN_PERSONAL_PROFILE_PREFIX):
                return ("reject", "linkedin_personal_profile_for_funding_claim")
            # Other LinkedIn paths (/posts, /jobs subpages, /pulse) might
            # legitimately host a funding article share — defer to LLM.
            return ("pass", "linkedin_other_defer")
        # Non-LinkedIn URLs (TechCrunch, PRNewswire, Crunchbase, company
        # /press/news pages, even bare homepages) — defer to LLM, which
        # can read content to judge whether the round is on the page.
        return ("pass", "non_linkedin_defer")

    # No target_type — universal layer passed, no further rules to apply.
    return ("pass", "universal_passed_no_target_specific_rules")


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

            # ── Free deterministic gates (run BEFORE LLM call) ──
            # 1) TOPIC-MISMATCH: if the TARGET asks for one topic (e.g.
            #    FUNDING) and the miner's CLAIM text matches a different
            #    topic (e.g. HIRING — a job posting attached to a funding
            #    target), the evidence is for the wrong category.  Reject
            #    deterministically without paying for an LLM call.  Either
            #    side being None (no clear topic) skips this check.
            # 2) URL evidence quality: always runs the UNIVERSAL layer
            #    (bare LinkedIn company page → reject regardless of target
            #    topic).  Production audit 2026-05-21 found 71% of intent
            #    URLs on the Mexican construction chain were bare
            #    LinkedIn slugs — substantively zero-evidence.  The
            #    TARGET-TYPE-SPECIFIC layer (HIRING / FUNDING rules)
            #    additionally fires only when target_type is known.
            # Both gates activated via INTENT_URL_PREFILTER_ENABLED.
            if URL_PREFILTER_ENABLED:
                target_type = _classify_target_type(target_text)
                claim_evidence_type = _classify_claim_evidence_type(miner_claim)

                if (
                    target_type
                    and claim_evidence_type
                    and target_type != claim_evidence_type
                ):
                    logger.info(
                        "Intent pre-check signal[%d] REJECT (topic-mismatch)  "
                        "target_type=%s  claim_type=%s  url=%r",
                        idx, target_type, claim_evidence_type, miner_url[:120],
                    )
                    return idx, False, (
                        f"topic_mismatch(target={target_type.lower()},"
                        f"claim={claim_evidence_type.lower()})"
                    )

                # Always invoke the URL quality check — the universal
                # layer fires regardless of target_type, the
                # target-specific layer is a no-op when target_type is None.
                pre_verdict, pre_reason = _check_intent_url_evidence_quality(
                    miner_url, target_type,
                )
                if pre_verdict == "reject":
                    logger.info(
                        "Intent pre-check signal[%d] REJECT (url-evidence-quality)  "
                        "target_type=%s  reason=%s  url=%r",
                        idx, target_type or "none", pre_reason, miner_url[:120],
                    )
                    return idx, False, f"url_evidence_quality_reject({pre_reason})"

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
