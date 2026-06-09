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

A separate free deterministic URL-evidence-quality gate (no LLM cost) runs
INSIDE this module before the Gemini call, gated by
INTENT_URL_PREFILTER_ENABLED.  It catches the obvious bare-LinkedIn-
company-page class of bad URLs on every claim type (not just hiring/
funding), plus target-type-specific LinkedIn rules.  See
``_check_intent_url_evidence_quality`` below.
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

# HIRING-STRONG — language that unambiguously calls for a JOB LISTING URL,
# not a post.  Wins over SOCIAL_POSTING when both match (e.g. "Hiring
# manager posts open roles" should still require /jobs/, not /posts/).
_HIRING_STRONG_RE = re.compile(
    r"\b("
    r"open\s+(?:position(?:s)?|role(?:s)?|vacanc(?:y|ies))|"
    r"job\s+(?:post(?:s|ing|ings)?|listing(?:s)?|opening(?:s)?)|"
    r"actively\s+(?:hiring|recruiting)\s+(?:for|a|an|the|its?)\b|"
    r"hiring\s+(?:for|a|an|the|its?)\s+\w+"
    r")\b",
    re.IGNORECASE,
)

# SOCIAL_POSTING — fires when BOTH a posting verb AND a person/role subject
# AND a social platform are present in the same signal.  All three are
# required, so a signal like "Company is hiring SDRs" doesn't trip this.
_SOCIAL_POSTING_VERB_RE = re.compile(
    r"\b("
    r"post(?:s|ing|ed)?|"
    r"publicly\s+post(?:s|ing|ed)?|"
    r"actively\s+post(?:s|ing|ed)?|"
    r"sharing|shares\b|"
    r"writing\s+on|writes\s+on|"
    r"publish(?:es|ing|ed)|"
    r"announc(?:es|ing|ed)\s+on"
    r")\b",
    re.IGNORECASE,
)
_SOCIAL_POSTING_PERSON_RE = re.compile(
    r"\b("
    # Person/role subjects
    r"founder|co-?founder|"
    r"ceo|cto|cro|cmo|cfo|coo|cso|"
    r"chief\s+\w+\s+officer|"
    r"president|owner|managing\s+(?:director|partner)|"
    r"head\s+of\s+\w+|"
    r"vp\s+(?:of\s+)?\w+|vice\s+president|"
    r"director\s+of\s+\w+|"
    r"executive|leader|manager|"
    # Entity-as-poster subjects (company social presence is also valid
    # SOCIAL_POSTING evidence — the buyer wants a specific dated post
    # by the company's social account, NOT a job listing)
    r"(?:company|company's)\s+linkedin\s+page|"
    r"(?:linkedin|twitter|\bx\b)\s+(?:page|account|profile)|"
    r"company\s+page|"
    r"brand\s+account"
    r")\b",
    re.IGNORECASE,
)
_SOCIAL_POSTING_PLATFORM_RE = re.compile(
    r"\b("
    r"linkedin|twitter|\bx\b|"
    r"social\s+media|social\s+post"
    r")\b",
    re.IGNORECASE,
)


def _matches_social_posting(target_text: str) -> bool:
    """Return True iff the target text describes a PERSON POSTING on a
    SOCIAL PLATFORM.  All three triggers (posting verb + person subject +
    platform) must be present in the same signal."""
    if not target_text:
        return False
    return bool(
        _SOCIAL_POSTING_VERB_RE.search(target_text)
        and _SOCIAL_POSTING_PERSON_RE.search(target_text)
        and _SOCIAL_POSTING_PLATFORM_RE.search(target_text)
    )


# PODCAST_APPEARANCE — fires when a podcast/interview/episode noun AND
# either a person-role or company subject is present.  Distinct from
# SOCIAL_POSTING (which requires a SHARING verb + social-media platform)
# because podcast claims describe EVENT participation, not authorship of a
# post.  Examples that should match:
#   "CEO discussed AML strategy on a podcast"
#   "Founder appeared on YouTube interview"
#   "Company representative was a guest on the [Show Name] podcast"
_PODCAST_APPEARANCE_NOUN_RE = re.compile(
    r"\b("
    r"podcast(?:s)?|"
    r"podcast\s+(?:appearance|interview|episode)|"
    r"video\s+interview|"
    r"on(?:\s+the)?\s+podcast|"
    r"guest(?:\s+on)?\b|"
    r"interview(?:ed|ing)?\s+on|"
    r"appeared\s+on|"
    r"speaking\s+on|"
    r"discussed\s+on(?:\s+the)?\s+(?:podcast|show|episode)|"
    r"episode\s+of|"
    r"youtube\s+interview"
    r")\b",
    re.IGNORECASE,
)

# Subject matcher for PODCAST_APPEARANCE — superset of SOCIAL_POSTING's
# person matcher.  Adds generic podcast-context subjects (representative,
# spokesperson, "company" alone) because podcast signals often phrase
# the subject more loosely than social-posting signals ("Company guest on
# a podcast", "Founder representative interviewed").  The downstream URL
# precheck still requires a specific YouTube video URL, so being lenient
# at the classification stage doesn't open up false-positive evidence —
# it just routes more signals through the podcast verifier where actual
# attribution is checked.
_PODCAST_APPEARANCE_SUBJECT_RE = re.compile(
    r"\b("
    # Reuse all person roles from SOCIAL_POSTING
    r"founder|co-?founder|"
    r"ceo|cto|cro|cmo|cfo|coo|cso|"
    r"chief\s+\w+\s+officer|"
    r"president|owner|managing\s+(?:director|partner)|"
    r"head\s+of\s+\w+|"
    r"vp\s+(?:of\s+)?\w+|vice\s+president|"
    r"director\s+of\s+\w+|"
    r"executive|leader|manager|"
    # Podcast-specific generic subjects
    r"representative|spokesperson|company\s+representative|"
    r"team\s+member|employee|"
    # Entity subjects
    r"company\b|"
    r"brand\b"
    r")\b",
    re.IGNORECASE,
)


def _matches_podcast_appearance(target_text: str) -> bool:
    """Return True iff the target text describes a PODCAST / INTERVIEW
    APPEARANCE by someone (specific role or generic representative) from
    the company.  Requires both:
      - a podcast/interview noun phrase (podcast, episode, guest, interview)
      - a person or company subject (lenient matcher; see notes above)
    The verifier (PR-2) handles attribution at the URL level."""
    if not target_text:
        return False
    if not _PODCAST_APPEARANCE_NOUN_RE.search(target_text):
        return False
    return bool(_PODCAST_APPEARANCE_SUBJECT_RE.search(target_text))


# TECHSTACK — fires when a sales / marketing / data tool noun AND a
# usage verb are both present in the same signal.  Examples of signals
# that match:
#   "Company uses Salesforce as their CRM"
#   "BuiltWith data shows recent adoption of HubSpot"
#   "Job postings require Salesforce administration"
#   "Tech stack includes Marketo and Outreach"
# Examples that should NOT match (lack tool noun OR usage verb):
#   "Company is hiring SDRs"                       — no tool noun
#   "Founder posted on LinkedIn about hiring"      — no tool noun
#   "Raised Series B"                              — no tool noun
_TECHSTACK_TOOL_RE = re.compile(
    r"\b("
    # Sales & marketing tools
    r"salesforce|hubspot|marketo|pardot|"
    r"outreach|salesloft|apollo|gong|"
    r"zoominfo|6sense|demandbase|clay|"
    r"chili\s*piper|linkedin\s+sales\s+navigator|"
    # Generic categories
    r"crm\b|sales\s+engagement|sales\s+intelligence|"
    r"marketing\s+automation|revops\s+(?:tool|stack|tooling)|"
    r"tech\s+stack|techstack|tech-stack|"
    # Data sources for tech stack claims
    r"builtwith|wappalyzer|similartech"
    r")\b",
    re.IGNORECASE,
)
_TECHSTACK_USE_VERB_RE = re.compile(
    r"\b("
    r"uses?\b|using\b|deployed\b|adopted\b|adoption\b|adopting\b|"
    r"implemented\b|implementing\b|"
    r"standardiz(?:e|ed|ing|ation)|"
    r"migrated\s+to|switched\s+to|moved\s+to|"
    r"is\s+on\s+\w+|"
    r"tech\s+stack|techstack|tech-stack|tooling|tool\s*kit|"
    r"requires?\s+experience\s+with|"
    r"requires?\s+\w+\s+certification|"
    r"\w+\s+administration\b|\w+\s+certification\b|"
    r"references?\b|reference\s+\w+\s+as|"
    r"indicates?\s+active\s+use|shows?\s+\w+\s+adoption"
    r")\b",
    re.IGNORECASE,
)


def _matches_techstack(target_text: str) -> bool:
    """Return True iff the target text describes a COMPANY USING A TOOL.
    Requires both:
      - a tool noun (Salesforce, HubSpot, CRM, tech stack, etc.)
      - a usage verb / phrase (uses, deployed, requires experience with, etc.)

    Placed BEFORE SOCIAL_POSTING in _classify_target_type because tech-stack
    signals often phrase as "Job postings on LinkedIn mention Salesforce as
    required" — both LinkedIn (social-platform trigger) and Salesforce (tool
    noun) appear in the same text.  TECHSTACK is the more specific intent
    and should win.

    The downstream URL precheck rejects vendor comparison / integration /
    data-broker URL shapes; PART E (specialty prompt) enforces the harder
    anti-patterns at LLM substance time."""
    if not target_text:
        return False
    return bool(
        _TECHSTACK_TOOL_RE.search(target_text)
        and _TECHSTACK_USE_VERB_RE.search(target_text)
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

# SOCIAL_POSTING-acceptable URL patterns — exact post / activity links only.
# Profile URLs ("/in/<slug>") are NOT acceptable for posting evidence — they
# don't point at a specific dated post.  Same standard as HIRING requiring
# /jobs/* not generic profile pages.
_LINKEDIN_POSTS_PREFIX = "/posts/"                    # /posts/<id>
_LINKEDIN_FEED_ACTIVITY_RE = re.compile(r"^/feed/update/urn:li:activity:\d+")
_LINKEDIN_PULSE_PREFIX = "/pulse/"                    # /pulse/<slug>
_X_TWEET_RE = re.compile(r"^/[^/]+/status/\d+")        # /<user>/status/<id>

# PODCAST_APPEARANCE-acceptable URL patterns — exact YouTube video URLs only.
# Channel pages (/c/<slug>, /@<handle>, /channel/<id>) and playlists
# (/playlist?list=<id>) are not acceptable evidence — they don't point at a
# specific episode.  Same standard as HIRING requiring /jobs/<id> not a
# generic profile page.  Other podcast platforms (Spotify, Apple Podcasts)
# are deferred — the verifier (PR-2) is YouTube-only for now, so accepting
# their URLs here would let leads through that the verifier can't process.
_YOUTUBE_WATCH_RE = re.compile(r"^/watch")                   # /watch?v=<id>
_YOUTUBE_SHORTS_RE = re.compile(r"^/shorts/[A-Za-z0-9_-]{11}")
_YOUTUBE_EMBED_RE = re.compile(r"^/embed/[A-Za-z0-9_-]{11}")

# TECHSTACK URL anti-patterns.  Documented from a production audit of the
# Techstack request (148 unique URLs, 25 hand-examined, ~48% INVALID).  All
# patterns below are zero-cost deterministic rejections; matched at the
# URL-shape gate so miners get a specific failure reason and no API budget
# is spent fetching the URL contents.
#
# Anti-pattern 1: VENDOR_MARKETING — vendor's own comparison pages like
#   `6sense.com/tech/cyber-security/industrialdefender-vs-mitre`.  Path
#   shape is `/tech/.../X-vs-Y` or `/compare/.../X-vs-Y` (case insensitive,
#   path may include intermediate category segments before the X-vs-Y
#   leaf).
_TECHSTACK_VENDOR_COMPARE_RE = re.compile(
    r"/(?:tech|compare)/.*?-vs-[^/]+", re.IGNORECASE,
)
# Anti-pattern 2: PRODUCT_INTEGRATION — integration listings and
# integration directory pages.  Matches:
#   /integrations/ (e.g. acme.com/integrations/salesforce)
#   /integration/  (singular variant)
#   /integrations  at end of path (no trailing slash)
#   -integrations  as a suffix on the LAST path segment (e.g.
#     carahsoft.com/resources/15571-industrial-defender-integrations)
_TECHSTACK_INTEGRATION_RE = re.compile(
    r"(?:^|/)integrations?(?:/|$)|-integrations(?:/|$)", re.IGNORECASE,
)
# Anti-pattern 4: DATA_BROKER — generic company profile pages on data
# brokers don't substantiate specific tech usage.
_ZOOMINFO_PROFILE_RE = re.compile(r"^/c/")
_CRUNCHBASE_ORG_RE = re.compile(r"^/organization/")
_GROWJO_COMPANY_RE = re.compile(r"^/company/[^/]+/?$")
_APOLLO_COMPANIES_RE = re.compile(r"^/companies/")
# Builtin lists BOTH /company/<slug> (profile) AND /job/<id>/<slug> (job
# posting that may legitimately require a tool).  Only reject the bare
# /company/<slug> form — job postings under /jobs/ or /job/ stay routed
# through the LLM substance check.
_BUILTIN_COMPANY_PROFILE_RE = re.compile(r"^/company/[^/]+/?$")

# Generic company-feed subpages: /company/<slug>/{posts,life,people,insights}
# These are aggregated feeds or generic listing pages — none point at a
# specific dated event, so they can't substantively back any intent claim.
# LinkedIn serves SPECIFIC posts at /posts/<id> or
# /feed/update/urn:li:activity:<id> (different path entirely), so excluding
# /company/<slug>/posts cannot accidentally reject a specific-post URL.
#
# DELIBERATELY NOT in this set:
#   /company/<slug>/jobs   — visible current job listings; used by HIRING rule
#   /company/<slug>/about  — static company info (industry, HQ, size,
#                            description, specialties) that CAN substantiate
#                            static-fact intent signals (e.g., "company
#                            description mentions X", "size is 500-1000").
_LINKEDIN_COMPANY_GENERIC_FEED_RE = re.compile(
    r"^/company/[^/]+/(posts|life|people|insights)(?:/|$)"
)


def _classify_target_type(
    target_text: str,
    evidence_type_override: Optional[str] = None,
) -> Optional[str]:
    """Classify the TARGET ICP signal topic from target text.

    Returns 'HIRING' | 'FUNDING' | 'SOCIAL_POSTING' | None.  Used to pick
    which URL pre-filter rules apply.  Deliberately ignores the miner's
    claim text — the target is the source of truth for what topic the
    buyer is asking about.

    When ``evidence_type_override`` is set (operator-confirmed via the
    dashboard parse LLM), that wins over the regex classifier — there's
    no risk of a topic word ("hiring") being misread as the URL-type
    requirement.

    Classifier order (when no override):
      1. FUNDING       — funding-specific phrases (Series, raised $, etc.)
      2. HIRING-STRONG — explicit job-listing language ("open positions",
                         "job postings", "actively hiring for [role]")
      3. SOCIAL_POSTING — all 3 triggers (verb + person subject + platform)
      4. HIRING-WEAK   — fallback for bare "hire/hiring" word as topic
    """
    # Operator-confirmed evidence_type wins — bypass regex classifier.
    if evidence_type_override:
        et = evidence_type_override.strip().upper()
        if et in ("HIRING", "FUNDING", "SOCIAL_POSTING", "PODCAST_APPEARANCE",
                  "TECHSTACK"):
            return et
        # CASE_STUDY, OTHER, or unknown → no URL pre-filter (defer to LLM)
        return None

    t = target_text or ""
    # 1) FUNDING wins first — very specific phrases
    if _FUNDING_RE.search(t):
        return "FUNDING"
    # 2) TECHSTACK — tool noun + usage verb.  Runs BEFORE HIRING-STRONG
    #    and SOCIAL_POSTING because tech-stack signals frequently mention
    #    "Job postings on LinkedIn require Salesforce" — both job-posting
    #    and LinkedIn triggers would fire, but the SPECIFIC intent is the
    #    tech-stack adoption check.  The PART E prompt + tighter URL
    #    rejection rules handle the actual substance verification.
    if _matches_techstack(t):
        return "TECHSTACK"
    # 3) HIRING-STRONG — explicit job-listing language, overrides social
    if _HIRING_STRONG_RE.search(t):
        return "HIRING"
    # 4) PODCAST_APPEARANCE — podcast/interview noun + person-or-company
    #    subject.  Runs before SOCIAL_POSTING because podcast-shaped
    #    signals often also contain "LinkedIn" (e.g. "CEO discussed on
    #    podcast that was shared on LinkedIn") and we want podcast
    #    classification to win for those.
    if _matches_podcast_appearance(t):
        return "PODCAST_APPEARANCE"
    # 5) SOCIAL_POSTING — all three triggers in same signal
    if _matches_social_posting(t):
        return "SOCIAL_POSTING"
    # 6) HIRING-WEAK fallback — bare "hire/hiring" topic word
    if _HIRING_RE.search(t):
        return "HIRING"
    return None


async def llm_classify_evidence_type(text: str) -> Optional[str]:
    """Strict LLM classifier for evidence_type (Gemini Flash via OpenRouter).

    Returns one of the canonical upper-case enum strings:
      HIRING | FUNDING | SOCIAL_POSTING | PODCAST_APPEARANCE |
      TECHSTACK | CASE_STUDY | OTHER
    or None on final failure (after 3 retries).

    Used as a fallback when the deterministic regex classifier in
    ``_classify_target_type`` returns None.  The closed enum + explicit
    prompt prevents misspelling drift.  Caller decides whether None
    means "skip" (best-effort recycle path) or "raise HTTP 400"
    (create_request path).
    """
    import aiohttp, asyncio, os
    or_key = (
        os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    )
    if not or_key:
        return None
    prompt = (
        "Classify the buyer-side intent signal below into EXACTLY ONE of\n"
        "the seven categories.  Reply with ONLY the category string (no\n"
        "JSON, no quotes, no explanation).\n\n"
        "CATEGORIES:\n"
        "  HIRING              — open job postings, active recruitment\n"
        "  FUNDING             — series A/B/C, raised $, closed seed\n"
        "  SOCIAL_POSTING      — posts on LinkedIn / X by a person\n"
        "  PODCAST_APPEARANCE  — guest on podcast / video interview\n"
        "  TECHSTACK           — uses a specific tool / CRM / sales tech\n"
        "  CASE_STUDY          — published case study, customer story\n"
        "  OTHER               — none of the above\n\n"
        f"SIGNAL: \"{text}\"\n\n"
        "CATEGORY:"
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    allowed = {
        "HIRING", "FUNDING", "SOCIAL_POSTING", "PODCAST_APPEARANCE",
        "TECHSTACK", "CASE_STUDY", "OTHER",
    }
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "google/gemini-2.5-flash-lite",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return None
                    body = await resp.json()
                    raw = body["choices"][0]["message"]["content"] or ""
                    norm = (
                        raw.strip().strip("\"'`. ").upper()
                           .replace("-", "_").replace(" ", "_")
                    )
                    if norm in allowed:
                        return norm
                    if attempt < 2:
                        continue
                    return None
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return None
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


def lead_has_unverifiable_linkedin_intent_url(intent_signals) -> Optional[Tuple[str, str]]:
    """Return ``(url, reason)`` for the first unverifiable LinkedIn URL found
    in any of the lead's intent signals, or None if all URLs are OK.

    "Unverifiable" means a LinkedIn URL that does NOT point at a specific
    dated event and therefore cannot substantively back any intent claim:

      1. Bare company profile: ``linkedin.com/company/<slug>``
         A static profile — no posts, no jobs, no news — useless as evidence.

      2. Generic company feed pages:
         ``linkedin.com/company/<slug>/{posts,life,people,insights}``
         These are aggregated feeds or generic listing pages — a claim
         about a specific dated event (e.g., a 2023-11-30 LinkedIn post)
         is not verifiable by clicking a feed URL, since the feed only
         surfaces recent items, may have moved older posts off the
         visible window, or the specific post may have been deleted.
         LinkedIn serves SPECIFIC posts at ``/posts/<id>`` or
         ``/feed/update/urn:li:activity:<id>`` (different path entirely)
         — those remain acceptable.  ``/company/<slug>/about`` is also
         acceptable — it shows static company info (industry, HQ, size,
         description, specialties) that CAN substantiate static-fact
         intent signals.

    Used as a deterministic pre-Tier-1.5 lead-level hard gate: if a miner
    included EVEN ONE such URL among the lead's intent evidence, the
    entire lead is rejected before any LLM call (Tier 1.5, Tier 1.7
    substance, Tier 2, Tier 3).  Stricter than the per-signal check —
    a single unverifiable URL fails the whole lead, closing the
    "pad-with-one-legit-URL" loophole.

    Args:
        intent_signals: iterable of IntentSignal-shaped objects (duck-typed
                        — each must expose a ``url`` attribute).

    Returns:
        ``(offending_url, reason_short_name)`` for the first match, where
        reason is one of:
          * "bare_linkedin_company_page"
          * "linkedin_company_generic_feed_page"
        OR None if no unverifiable URLs are present.
    """
    for sig in (intent_signals or []):
        url = getattr(sig, "url", "") or ""
        if not url:
            continue
        parsed = urlparse(url.lower())
        host = parsed.hostname or ""
        path = parsed.path or ""
        if "linkedin.com" not in host:
            continue
        if _LINKEDIN_BARE_COMPANY_PAGE_RE.match(path):
            return (url, "bare_linkedin_company_page")
        if _LINKEDIN_COMPANY_GENERIC_FEED_RE.match(path):
            return (url, "linkedin_company_generic_feed_page")
    return None


# Backward-compatible alias for the prior, narrower function name.
# Existing imports in gateway/fulfillment/scoring.py reference this name;
# keeping it as a thin wrapper avoids touching the call site.  Returns only
# the URL (drops the reason) so the existing signature is preserved.  New
# callers should prefer ``lead_has_unverifiable_linkedin_intent_url`` to get
# the (url, reason) tuple.
def lead_has_bare_linkedin_intent_url(intent_signals) -> Optional[str]:
    result = lead_has_unverifiable_linkedin_intent_url(intent_signals)
    return result[0] if result else None


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

    # ── Layer 1 (UNIVERSAL): LinkedIn URLs that don't point at a specific
    # verifiable item.  Two shapes are rejected here regardless of claim
    # topic:
    #   - bare /company/<slug>             — static profile, no events
    #   - generic /company/<slug>/{posts,about,life,people,insights}
    #                                       — aggregated feed or static
    #                                       info page, doesn't substantiate
    #                                       any specific dated claim.
    # LinkedIn serves specific posts at /posts/<id> or
    # /feed/update/urn:li:activity:<id> — those keep passing because they
    # ARE verifiable.  /company/<slug>/jobs also keeps passing — it's the
    # acceptable HIRING evidence path.
    if "linkedin.com" in host:
        if _LINKEDIN_BARE_COMPANY_PAGE_RE.match(path):
            return ("reject", "bare_linkedin_company_page")
        if _LINKEDIN_COMPANY_GENERIC_FEED_RE.match(path):
            return ("reject", "linkedin_company_generic_feed_page")

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

    if target_type == "SOCIAL_POSTING":
        # Buyer asked for evidence of a SPECIFIC POST by a person.  Same
        # standard as HIRING requiring /jobs/*: only EXACT POST URLs
        # qualify — profile pages, generic feeds, bare company pages
        # don't substantiate a specific dated post.
        if "linkedin.com" in host:
            #   /posts/<id>                    — specific LinkedIn post
            #   /feed/update/urn:li:activity:* — specific activity post
            #   /pulse/<slug>                  — specific Pulse article
            if path.startswith(_LINKEDIN_POSTS_PREFIX):
                return ("pass", "linkedin_posts_path")
            if _LINKEDIN_FEED_ACTIVITY_RE.match(path):
                return ("pass", "linkedin_feed_activity")
            if path.startswith(_LINKEDIN_PULSE_PREFIX):
                return ("pass", "linkedin_pulse_path")
            # /jobs/* is wrong topic — buyer asked for a post, not a job.
            # /in/<slug> is a profile, not a dated post — reject.
            # /company/<slug>/posts/ is generic feed — universal layer
            # already rejected that, but in case it slipped through:
            return ("reject", "linkedin_not_a_specific_post")
        if "twitter.com" in host or "x.com" in host:
            #   /<user>/status/<id> — specific tweet
            if _X_TWEET_RE.match(path):
                return ("pass", "x_tweet_status_path")
            return ("reject", "x_not_a_specific_tweet")
        # Non-LinkedIn/X URLs — could be a company blog post, press
        # mention, etc.  Defer to LLM to read content.
        return ("pass", "non_social_defer")

    if target_type == "PODCAST_APPEARANCE":
        # Buyer asked for evidence of a podcast / video interview.  The
        # verifier (PR-2) fetches video metadata + transcript from
        # ScrapingDog's YouTube endpoints, so only YouTube watch / shorts /
        # embed URLs can be processed.  Channel pages, playlist URLs, and
        # other podcast platforms (Spotify, Apple Podcasts, SoundCloud) are
        # deferred — the verifier can't currently extract transcripts from
        # them, so accepting them here would let leads through that the
        # downstream verifier rejects with no actionable feedback.
        # Reject deterministically at the URL-shape gate (zero API cost)
        # so miners get the specific failure reason.
        if "youtube.com" in host or "youtu.be" in host:
            if host.endswith("youtu.be"):
                # https://youtu.be/<id> — short share form, any non-root path
                if len(path.strip("/")) >= 11:
                    return ("pass", "youtube_short_url")
                return ("reject", "youtube_short_url_missing_id")
            # youtube.com / m.youtube.com / www.youtube.com
            if _YOUTUBE_WATCH_RE.match(path):
                return ("pass", "youtube_watch_video")
            if _YOUTUBE_SHORTS_RE.match(path):
                return ("pass", "youtube_shorts_video")
            if _YOUTUBE_EMBED_RE.match(path):
                return ("pass", "youtube_embed_video")
            # /channel/<id>, /c/<slug>, /@<handle>, /user/<name> — channel
            # pages, not specific videos.
            # /playlist?list=<id> — playlist of multiple videos, not one.
            # /feed/* — feed pages.
            return ("reject", "youtube_not_a_specific_video")
        # Apple Podcasts / Spotify / etc — not yet supported by verifier.
        # Reject explicitly with a clear reason so miners know YouTube is
        # the only platform currently accepted.
        if ("podcasts.apple.com" in host or
            "open.spotify.com" in host or
            "soundcloud.com" in host or
            "anchor.fm" in host):
            return ("reject", "podcast_platform_not_youtube")
        # Non-podcast URLs (random blog posts, news articles, generic
        # press) — buyer asked specifically for a podcast appearance, so
        # a non-YouTube URL doesn't satisfy that.  Reject explicitly.
        return ("reject", "not_a_youtube_url_for_podcast_claim")

    if target_type == "TECHSTACK":
        # Eight production-audited anti-patterns, all rejected at zero API
        # cost.  Remaining patterns (REPLACED, TANGENTIAL, WRONG_COMPANY,
        # PREVIOUS_EMPLOYER, PRODUCT_INTEGRATION semantic check) require
        # reading the page content and are enforced by PART E of the
        # TECHSTACK verification prompt.

        # 1. VENDOR_MARKETING — vendor's own comparison pages like
        #    `/tech/X-vs-Y` (e.g. 6sense.com/tech/...).
        if _TECHSTACK_VENDOR_COMPARE_RE.search(path):
            return ("reject", "techstack_vendor_comparison_page")

        # 2. PRODUCT_INTEGRATION — vendor's "X integrates with our product"
        #    pages substantiate integration availability, not target's
        #    internal tool usage.  Canonical shapes:
        #       /integrations/ or /integration/ as a path segment
        #       trailing -integrations slug (e.g. carahsoft.com
        #         /resources/<id>-<vendor>-integrations)
        if _TECHSTACK_INTEGRATION_RE.search(path):
            return ("reject", "techstack_integration_listing")

        # 4. DATA_BROKER — generic company profile pages without
        #    tool-specific assertions.
        if "zoominfo.com" in host and _ZOOMINFO_PROFILE_RE.match(path):
            return ("reject", "techstack_zoominfo_profile")
        if "crunchbase.com" in host and _CRUNCHBASE_ORG_RE.match(path):
            return ("reject", "techstack_crunchbase_profile")
        if "growjo.com" in host and _GROWJO_COMPANY_RE.match(path):
            return ("reject", "techstack_growjo_profile")
        if "apollo.io" in host and _APOLLO_COMPANIES_RE.match(path):
            return ("reject", "techstack_apollo_profile")
        # Builtin: only reject /company/<slug> profile pages — job
        # postings under /job/ or /jobs/ may legitimately list tool
        # requirements and stay routed through the LLM check.
        if ("builtin.com" in host
            and _BUILTIN_COMPANY_PROFILE_RE.match(path)
            and not path.lower().startswith(("/job/", "/jobs/"))):
            return ("reject", "techstack_builtin_company_profile")

        # 6. PERSONAL_PROFILE — individual LinkedIn pages.  Personal
        #    profiles don't substantiate company tech choices even if the
        #    person works at the target company.
        if "linkedin.com" in host and path.startswith("/in/"):
            return ("reject", "techstack_linkedin_personal_profile")

        # All remaining anti-patterns (REPLACED, TANGENTIAL, WRONG_COMPANY,
        # PREVIOUS_EMPLOYER, semantic-level VENDOR_MARKETING) require page
        # text — defer to PART E in the LLM verifier.
        return ("pass", "techstack_defer")

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
  lead_company:      "{lead_company}"
  miner_claim:       "{miner_claim}"
  miner_source:      "{miner_source}"
  miner_url:         "{miner_url}"
  target_icp_signal: "{target_icp_signal}"

The target intent signal is about lead_company specifically. Judge the claim
relative to lead_company — not any other entity mentioned in the claim.

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

  8. SUBJECT/OBJECT DIRECTION (for lead_company). When the target describes
     an event happening TO the company (e.g., "secured/raised/received
     investment", "was acquired", "received funding"), lead_company must
     appear in the claim as the RECIPIENT / PASSIVE party.  If the claim
     shows lead_company as the ACTIVE party — performing the action onto
     someone else — REJECT, even if the same investment / acquisition
     keywords appear.

     Examples (lead_company = "Acme" in all):
       target: "Company secured strategic or corporate investment"
       claim:  "Acme acquired a majority stake in Y"           → no  (Acme is acquirer)
       claim:  "Acme invested in Y"                             → no  (Acme is investor)
       claim:  "Acme raised $50M Series B"                      → yes (Acme is recipient)
       claim:  "Acme received strategic investment from Z"       → yes (Acme is recipient)

       target: "Company was acquired in the past 12 months"
       claim:  "Acme acquired Y"                                → no  (Acme is acquirer, not acquired)
       claim:  "Acme was acquired by Z"                         → yes (Acme is target)

       target: "Company partnered with a major enterprise"
       claim:  "Acme signed partnership with Y, a Fortune 500"   → yes
       claim:  "Acme expanded partner network with 5 small firms" → no  (no major enterprise)

  9. WHEN AMBIGUOUS, REJECT. Strict mode: if you cannot map the claim onto
     the target unambiguously under rules 1-8, answer "no".

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
    lead_company: str = "",
) -> str:
    return _PROMPT_TEMPLATE.format(
        lead_company=lead_company or "<unknown>",
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
        "max_tokens": 400,
        "reasoning": {"exclude": True},
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
    lead_company: str = "",
    icp_intent_signal_evidence_types: Optional[List[Optional[str]]] = None,
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
    # Per-target evidence_type override list, matched 1:1 with icp_texts.
    # When the operator-confirmed value is present we use it instead of
    # the regex classifier.  Length-mismatch or None → fall back to regex
    # for every index.
    icp_evidence_types: List[Optional[str]] = list(
        icp_intent_signal_evidence_types or []
    )
    if len(icp_evidence_types) != num_icp:
        icp_evidence_types = [None] * num_icp
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
                target_type = _classify_target_type(
                    target_text,
                    evidence_type_override=icp_evidence_types[matched_idx],
                )
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
                    _build_prompt(miner_claim, source_str, miner_url, target_text, today, lead_company),
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
