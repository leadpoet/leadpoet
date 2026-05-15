"""Fulfillment Tier 2c: Required-attribute verification via Perplexity Sonar.

Runs after person verification (Stage 4) and before intent scoring (Tier 3).
For each attribute in the buyer's ``required_attributes`` (company[] + contact[]):

  - Positive attributes → ``COMPANY_PROMPT`` or ``CONTACT_PROMPT``
  - Negative attributes ("Does not have X", "No Y in place") →
    ``POSITIVE_PROXY_PROMPT`` (searches for evidence of the positive opposite;
    if found, the negative is false → REJECT this attribute)

Aggregation (fail-closed on NO, soft on DEFERRED):
  - Any attribute → ``NO``                       → decision = REJECT
  - All attributes ``YES``                       → decision = ACCEPT
  - Mix of ``YES`` + ``DEFERRED``                → decision = ACCEPT_WITH_DEFERRAL
    (treated as accept for gating; deferral surfaces in result for audit)

The Sonar calls are made in parallel per lead via an asyncio.Semaphore.

Entry point::

    passed, result = await verify_required_attributes(
        lead=validator_dict,
        required_attributes={"company": [...], "contact": [...]},
        apify_person_data=lead.get("_apify_data"),   # set by Stage 4
        openrouter_key=OPENROUTER_KEY,
    )

The ``result`` dict is what gets persisted into
``automated_checks_data["attribute_verification"]`` (JSONB) and surfaced on
``FulfillmentScoreResult.attribute_verification``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""

SONAR_MODEL = "perplexity/sonar"
SONAR_URL = "https://openrouter.ai/api/v1/chat/completions"
SONAR_TIMEOUT_S = 90
MAX_CONCURRENCY = 8

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_PROMPT = """You are verifying whether a specific factual statement is true about a specific company.
Use public web sources only. Respond strictly in the format below.

COMPANY:
  Name: {company_name}
  Website: {company_website}
  LinkedIn: {company_linkedin}

STATEMENT TO VERIFY:
  "{attribute_text}"

PRIORITIZE these source types in order: official company website, SEC filings,
press releases, Crunchbase, reputable business news (TechCrunch, Bloomberg, Reuters,
Forbes, WSJ), LinkedIn company page. De-prioritize: marketing blogs, listicles,
aggregator sites, Wikipedia.

Respond strictly in this exact format:

VERDICT: YES | NO
EVIDENCE: <one or two sentences quoting the supporting source text verbatim>
CITATIONS: <newline-separated URLs you actually used>
REASONING: <one sentence explaining why you reached this verdict>

Decision rules:
- You MUST answer YES or NO. Do not answer UNCERTAIN.
- Answer YES iff you have explicit public evidence the statement is true about THIS specific company.
- Otherwise → NO.

OR-LIST RULE:
- If the statement is a list of options joined by "or" (e.g. "Vertical is X, Y, or Z"),
  answer YES if the company matches ANY ONE item in the list.
- You do NOT need an exact phrase match — list membership is enough.
- Example: statement "Vertical is professional services, hospitality, or technology" →
  YES for a sports-bar chain (hospitality), YES for a telecom (technology),
  YES for a bank (financial services).

INFERENCE RULE (one-step inference is ALLOWED):
- You MAY conclude YES from one logical step on explicit, public, non-strategic facts.
- Examples of allowed inference:
    "Sells products in 100+ countries"             → YES on "is exporter / ships internationally"
    "Operating airline" or "Operates trucking"     → YES on "has frontline / non-desk workforce"
    "Manufactures and distributes physical goods"  → YES on "operates in manufacturing"
    "Lists 3 office addresses on contact page"     → YES on "has 1-10 fixed locations"
    "Family-owned and operated"                    → YES on "independently owned"
- You may NOT infer from:
    · strategic / internal claims (e.g. "operates sales-led GTM motion", "is product-led")
    · absence of evidence (silence ≠ disproof for positive attributes)
    · unrelated tangential facts (e.g. "has a Transportation INSURANCE team" does NOT mean "is a transport business")

ENTITY MATCHING:
- Verify the SPECIFIC company at the website/LinkedIn provided. If the name is common,
  prefer the entity matching the website/LinkedIn over similarly-named entities."""


CONTACT_PROMPT = """You are verifying whether a specific factual statement is true about a specific person.
Use public web sources only. Respond strictly in the format below.

PERSON:
  Name: {contact_name}
  LinkedIn: {contact_linkedin}
{apify_block}

STATEMENT TO VERIFY:
  "{attribute_text}"

PRIORITIZE these source types in order: LinkedIn profile (data above), official
company "about" or "team" pages, press releases naming the person, conference speaker bios,
reputable news articles. De-prioritize: data-broker sites, social-media aggregators.

Respond strictly in this exact format:

VERDICT: YES | NO
EVIDENCE: <one or two sentences quoting the supporting source text verbatim>
CITATIONS: <newline-separated URLs you actually used>
REASONING: <one sentence explaining why you reached this verdict>

Decision rules:
- You MUST answer YES or NO. Do not answer UNCERTAIN.
- Answer YES iff you have explicit public evidence the statement is true about THIS specific person.
- Otherwise → NO.

INFERENCE RULE (one-step inference is ALLOWED on public facts):
- "Director of Sales at Fortune-500"  → YES on "earns >$200K"
- "Graduated college 2005"            → YES on "age 35-58"
- "Title contains 'Director' or 'VP'" → YES on "senior leader"
- "Education started 2010"            → YES on "age 30+" (derived from typical start age)
- You may NOT infer about private financial state, retirement accounts, tax vehicles,
  internal beliefs, or any other non-publicly-observable personal information —
  answer NO for those.

ENTITY MATCHING:
- The LinkedIn-derived data block above is the source of truth for this person.
- If the name is common, prefer the person matching the LinkedIn URL over similarly-named individuals."""


POSITIVE_PROXY_PROMPT = """You are verifying a NEGATIVE statement about a {subject_type}.
Negative statements cannot be verified directly — you cannot prove absence
from public sources. Instead, search aggressively for evidence of the POSITIVE OPPOSITE.

{subject_type_upper}:
{subject_block}

ORIGINAL NEGATIVE STATEMENT (treat as a hypothesis to test):
  "{attribute_text}"

YOUR TASK:
1. Mentally invert the statement to its positive opposite.
2. Search the public web aggressively for ANY evidence of that positive opposite.
3. If you find any credible public evidence the positive opposite is true for this
   specific {subject_type} → original negative is FALSE → verdict NO.
4. If after thorough search no such evidence exists → original negative is PRESUMED
   TRUE → verdict YES.

WHERE TO SEARCH (cast a wide net):
  · For "doesn't have a digital product / paid offering": Linktree, Gumroad,
    Stripe, Patreon, Substack paid tiers, Teachable, Kajabi, Thinkific,
    Mighty Networks, Beehiiv, Buy Me a Coffee, personal website "shop"/"products"
    /"courses"/"buy" pages.
  · For "not using competitor X": case studies on X's website, public
    testimonials, job postings mentioning X, conference talks, GitHub repos.
  · For "no Y in place": any public statement referencing Y.
  · For private financial state (retirement accounts, salary, tax status):
    these are NOT publicly observable — you will almost certainly find nothing,
    but say so explicitly via PRIVATE_INFO_CAVEAT=YES.

PRIORITIZE: official sources, LinkedIn, personal websites, well-known platforms.
DE-PRIORITIZE: data brokers, social-media aggregators.

Respond strictly in this exact format:

VERDICT: YES | NO
EVIDENCE: <if NO: one or two sentences quoting the source proving the positive
           opposite. If YES: "No public evidence of [positive opposite] found after
           searching [3-5 sources you actually checked].">
CITATIONS: <newline-separated URLs you actually consulted, even if dead ends>
REASONING: <one sentence>
SEARCH_BREADTH: <count of distinct sources you actually checked>
PRIVATE_INFO_CAVEAT: <YES if the negative is about non-publicly-observable info
                     (private finances, internal beliefs, tax vehicles); NO otherwise>

Decision rules:
- You MUST answer YES or NO.
- Default to NO if you find the positive opposite anywhere credible.
- Default to YES only after thorough search (≥3 distinct sources).
- If PRIVATE_INFO_CAVEAT is YES the YES verdict is weak — flag it clearly."""


# ─────────────────────────────────────────────────────────────────────────────
# Negative-attribute detection
# ─────────────────────────────────────────────────────────────────────────────

_NEGATIVE_PATTERNS = [
    r"^does not\b", r"^doesn'?t\b",
    r"^is not\b",   r"^isn'?t\b",
    r"^has not\b",  r"^hasn'?t\b",
    r"^have not\b", r"^haven'?t\b",
    r"^no\s+\w+",          # "No tax-free retirement vehicle..."
    r"^lack[s]?\b",
    r"^without\b",
]
_EXCLUSION_PREFIX = re.compile(r"^not\s+(in|a\s|an\s)\b")


def is_negative_attribute(text: str) -> bool:
    """Detect negatives that require proving an absence — Sonar cannot do this
    directly via positive evidence and must use the proxy-search path.

    Exclusion-style negatives like "Not in manufacturing" or "Not McDonald's, KFC"
    are NOT treated as proxy-eligible negatives: they can be verified positively
    by observing the company's actual industry / chain affiliation.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if _EXCLUSION_PREFIX.match(t):
        return False
    return any(re.match(p, t) for p in _NEGATIVE_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Apify-data block builder (injected into CONTACT_PROMPT)
# ─────────────────────────────────────────────────────────────────────────────

def _format_date(d: Optional[Dict[str, Any]]) -> str:
    """Apify dates look like {"text": "2018-01"} or {"text": "Present"}."""
    if not isinstance(d, dict):
        return ""
    return (d.get("text") or "").strip()


def build_apify_contact_block(apify_data: Optional[dict], contact_description: str = "") -> str:
    """Render the Apify-extracted person data into a block injected into the
    CONTACT_PROMPT.  Provides the LinkedIn information Sonar cannot fetch
    directly (LinkedIn blocks scraping)."""
    if not apify_data:
        # Minimal fallback: just whatever miner-side description we have
        desc = (contact_description or "").strip() or "(unknown)"
        return f"  Profile description: {desc}"

    lines: List[str] = []

    # Headline / summary
    headline = (apify_data.get("headline") or "").strip()
    if headline:
        lines.append(f"  Headline: {headline}")

    summary = (apify_data.get("summary") or apify_data.get("about") or "").strip()
    if summary:
        # Keep it bounded so we don't blow up the prompt for chatty profiles
        if len(summary) > 800:
            summary = summary[:800] + "…"
        lines.append(f"  About: {summary}")
    elif contact_description:
        lines.append(f"  Description: {contact_description}")

    # Current position
    current_positions = apify_data.get("currentPosition") or []
    if current_positions:
        cp = current_positions[0]
        title = (cp.get("position") or cp.get("title") or "").strip()
        company = (cp.get("companyName") or "").strip()
        start = _format_date(cp.get("startDate"))
        end = _format_date(cp.get("endDate")) or "Present"
        if title or company:
            line = f"  Current: {title} at {company}".rstrip()
            if start:
                line += f" ({start} – {end})"
            lines.append(line)

    # Experience timeline (limited to ~5 entries to keep prompt compact)
    experience = apify_data.get("experience") or []
    if experience:
        exp_lines: List[str] = []
        for exp in experience[:5]:
            title = (exp.get("position") or "").strip()
            company = (exp.get("companyName") or "").strip()
            start = _format_date(exp.get("startDate"))
            end = _format_date(exp.get("endDate")) or "Present"
            if title or company:
                row = f"    · {title} at {company}".rstrip()
                if start:
                    row += f" ({start} – {end})"
                exp_lines.append(row)
        if exp_lines:
            lines.append("  Experience:")
            lines.extend(exp_lines)

    # Education (start dates are useful for age inference)
    education = apify_data.get("education") or []
    if education:
        edu_lines: List[str] = []
        for edu in education[:3]:
            school = (edu.get("schoolName") or "").strip()
            field = (edu.get("fieldOfStudy") or "").strip()
            start = _format_date(edu.get("startDate"))
            end = _format_date(edu.get("endDate"))
            if school:
                row = f"    · {school}"
                if field:
                    row += f", {field}"
                if start:
                    row += f" ({start}" + (f" – {end})" if end else ")")
                edu_lines.append(row)
        if edu_lines:
            lines.append("  Education:")
            lines.extend(edu_lines)

    # Location
    loc_parsed = (apify_data.get("location") or {}).get("parsed") or {}
    loc_city = (loc_parsed.get("city") or "").strip()
    loc_state = (loc_parsed.get("state") or "").strip()
    loc_country = (loc_parsed.get("country") or "").strip()
    if loc_city or loc_state or loc_country:
        loc = ", ".join(x for x in (loc_city, loc_state, loc_country) if x)
        lines.append(f"  Location: {loc}")

    # Connections (often returned as "500+", treat as a string)
    connections = apify_data.get("connections")
    if connections:
        lines.append(f"  Connections: {connections}")

    return "\n".join(lines) if lines else "  Profile description: (unknown)"


# ─────────────────────────────────────────────────────────────────────────────
# Sonar response parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sonar_response(text: str) -> Dict[str, Any]:
    """Parse one Sonar response into a normalized dict.

    Defensive: anything not literally "YES" maps to "NO".
    Always emits the SEARCH_BREADTH and PRIVATE_INFO_CAVEAT fields used by
    the proxy path so callers don't need to branch.
    """
    out: Dict[str, Any] = {
        "verdict": "NO",
        "evidence": "",
        "citations": [],
        "reasoning": "",
        "search_breadth": 0,
        "private_info_caveat": False,
    }

    m = re.search(r"VERDICT:\s*(YES|NO)\b", text, re.IGNORECASE)
    if m:
        v = m.group(1).upper()
        out["verdict"] = "YES" if v == "YES" else "NO"

    m = re.search(r"EVIDENCE:\s*(.+?)(?=\n[A-Z_]+:|\Z)", text, re.DOTALL)
    if m:
        out["evidence"] = m.group(1).strip()

    m = re.search(r"CITATIONS:\s*(.+?)(?=\n[A-Z_]+:|\Z)", text, re.DOTALL)
    if m:
        out["citations"] = re.findall(r"https?://[^\s\)\]]+", m.group(1))

    m = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z_]+:|\Z)", text, re.DOTALL)
    if m:
        out["reasoning"] = m.group(1).strip()

    m = re.search(r"SEARCH_BREADTH:\s*(\d+)", text)
    if m:
        out["search_breadth"] = int(m.group(1))

    m = re.search(r"PRIVATE_INFO_CAVEAT:\s*(YES|NO|TRUE|FALSE)", text, re.IGNORECASE)
    if m:
        out["private_info_caveat"] = m.group(1).upper() in ("YES", "TRUE")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sonar caller
# ─────────────────────────────────────────────────────────────────────────────

async def _call_sonar(
    session: aiohttp.ClientSession, prompt: str, api_key: str
) -> Dict[str, Any]:
    """Single Sonar request via OpenRouter. Returns the parsed response."""
    started = time.perf_counter()
    try:
        async with session.post(
            SONAR_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": SONAR_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=aiohttp.ClientTimeout(total=SONAR_TIMEOUT_S),
        ) as resp:
            elapsed = time.perf_counter() - started
            if resp.status != 200:
                body = await resp.text()
                return {
                    "verdict": "NO",
                    "evidence": "",
                    "citations": [],
                    "reasoning": f"Sonar HTTP {resp.status}: {body[:120]}",
                    "search_breadth": 0,
                    "private_info_caveat": False,
                    "_latency_s": elapsed,
                    "_error": True,
                }
            payload = await resp.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = _parse_sonar_response(content)
            parsed["_latency_s"] = elapsed
            parsed["_tokens"] = payload.get("usage", {})
            return parsed
    except Exception as e:
        return {
            "verdict": "NO",
            "evidence": "",
            "citations": [],
            "reasoning": f"Sonar exception: {type(e).__name__}: {str(e)[:120]}",
            "search_breadth": 0,
            "private_info_caveat": False,
            "_latency_s": time.perf_counter() - started,
            "_error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-attribute verifiers
# ─────────────────────────────────────────────────────────────────────────────

async def _verify_one_attribute(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    attribute_text: str,
    scope: str,                            # "company" or "contact"
    identity: Dict[str, Any],
    apify_block: str,                      # rendered Apify info, contact only
    api_key: str,
) -> Dict[str, Any]:
    """Verify one attribute. Routes to positive prompt or proxy prompt based on
    whether the attribute is a negative."""
    is_neg = is_negative_attribute(attribute_text)

    if is_neg:
        subject_type = "person" if scope == "contact" else "company"
        subject_type_upper = subject_type.upper()
        if scope == "contact":
            subject_block = (
                f"  Name: {identity.get('contact_name', '(unknown)')}\n"
                f"  LinkedIn: {identity.get('contact_linkedin', '(unknown)')}\n"
                f"{apify_block}"
            )
        else:
            subject_block = (
                f"  Name: {identity.get('company_name', '(unknown)')}\n"
                f"  Website: {identity.get('company_website', '(unknown)')}\n"
                f"  LinkedIn: {identity.get('company_linkedin', '(unknown)')}"
            )
        prompt = POSITIVE_PROXY_PROMPT.format(
            subject_type=subject_type,
            subject_type_upper=subject_type_upper,
            subject_block=subject_block,
            attribute_text=attribute_text,
        )
    elif scope == "contact":
        prompt = CONTACT_PROMPT.format(
            contact_name=identity.get("contact_name", "(unknown)"),
            contact_linkedin=identity.get("contact_linkedin", "(unknown)"),
            apify_block=apify_block,
            attribute_text=attribute_text,
        )
    else:  # company
        prompt = COMPANY_PROMPT.format(
            company_name=identity.get("company_name", "(unknown)"),
            company_website=identity.get("company_website", "(unknown)"),
            company_linkedin=identity.get("company_linkedin", "(unknown)"),
            attribute_text=attribute_text,
        )

    async with sem:
        result = await _call_sonar(session, prompt, api_key)

    return {
        "attribute_text": attribute_text,
        "scope": scope,
        "is_negative": is_neg,
        "verdict": result.get("verdict", "NO"),
        "evidence": result.get("evidence", ""),
        "citations": result.get("citations", []),
        "reasoning": result.get("reasoning", ""),
        "private_info_caveat": result.get("private_info_caveat", False),
        "search_breadth": result.get("search_breadth", 0),
        "latency_s": result.get("_latency_s", 0.0),
        "_proxy_used": is_neg,
        "_error": result.get("_error", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Identity builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_identity(lead: dict) -> Dict[str, Any]:
    """Extract the identity fields the prompts need from the validator_dict.
    Tolerates a few field-name variants (business/company, linkedin/contact_linkedin)
    that exist across the codebase."""
    contact_name = (
        lead.get("full_name")
        or " ".join(
            x for x in (lead.get("first_name", ""), lead.get("last_name", "")) if x
        ).strip()
        or "(unknown)"
    )
    return {
        "company_name": lead.get("business") or lead.get("company") or "(unknown)",
        "company_website": lead.get("website") or lead.get("company_website") or "(unknown)",
        "company_linkedin": lead.get("company_linkedin") or "(unknown)",
        "contact_name": contact_name,
        "contact_linkedin": lead.get("linkedin") or lead.get("contact_linkedin") or "(unknown)",
        "contact_description": lead.get("description") or lead.get("bio") or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def verify_required_attributes(
    lead: dict,
    required_attributes: Optional[Dict[str, List[str]]],
    apify_person_data: Optional[dict] = None,
    openrouter_key: str = "",
    max_concurrency: int = MAX_CONCURRENCY,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify every required attribute on this lead via Sonar.

    Args:
        lead: validator dict (FulfillmentLead.to_validator_dict())
        required_attributes: ``{"company": [...], "contact": [...]}`` from the
            buyer's ICP. ``None`` or empty → no-op (returns passed=True).
        apify_person_data: Apify LinkedIn profile dict from Stage 4, used to
            populate the contact-side prompt. ``None`` → graceful degradation.
        openrouter_key: API key. Falls back to env if empty.
        max_concurrency: how many Sonar calls in flight per lead.

    Returns:
        ``(passed, result_dict)`` where
          - ``passed`` ∈ {True, False}
          - ``result_dict`` always contains: decision, counts, per_attribute,
            model, elapsed_s, timestamp.  On REJECT it also has rejection_reason
            (the sibling-style {stage, check_name, message, failed_fields} dict).
    """
    started = time.perf_counter()
    key = openrouter_key or OPENROUTER_KEY

    # Normalize the attribute list. Empty / missing → no gate to apply.
    ra = required_attributes or {}
    company_attrs: List[str] = [str(a).strip() for a in (ra.get("company") or []) if str(a).strip()]
    contact_attrs: List[str] = [str(a).strip() for a in (ra.get("contact") or []) if str(a).strip()]

    if not company_attrs and not contact_attrs:
        return True, {
            "decision": "ACCEPT",
            "counts": {"yes": 0, "no": 0, "deferred": 0},
            "per_attribute": [],
            "model": SONAR_MODEL,
            "elapsed_s": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "no required_attributes on this ICP — gate skipped",
        }

    if not key:
        # No API key: fail-safe to REJECT so we don't silently admit unverified leads
        return False, {
            "decision": "REJECT",
            "counts": {"yes": 0, "no": 0, "deferred": 0},
            "per_attribute": [],
            "model": SONAR_MODEL,
            "elapsed_s": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rejection_reason": {
                "stage": "Tier 2c: Required Attribute Verification",
                "check_name": "attribute_verification_no_api_key",
                "message": "OPENROUTER_KEY not configured; cannot run Sonar checks",
                "failed_fields": [],
            },
        }

    identity = _build_identity(lead)
    apify_block = build_apify_contact_block(apify_person_data, identity["contact_description"])

    sem = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=SONAR_TIMEOUT_S + 30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _verify_one_attribute(session, sem, a, "company", identity, apify_block, key)
            for a in company_attrs
        ] + [
            _verify_one_attribute(session, sem, a, "contact", identity, apify_block, key)
            for a in contact_attrs
        ]
        per_attribute = await asyncio.gather(*tasks)

    # ─── Aggregate ─────────────────────────────────────────────────────────
    # Verdict-to-status mapping with private-info caveat policy:
    #   YES + caveat=False                 → "yes"      (real verification)
    #   YES + caveat=True (private-info)   → "deferred" (weak — needs alt layer)
    #   NO                                 → "no"
    counts = {"yes": 0, "no": 0, "deferred": 0}
    first_failure: Optional[Dict[str, Any]] = None
    for pa in per_attribute:
        verdict = pa["verdict"]
        if verdict == "YES":
            if pa["is_negative"] and pa.get("private_info_caveat"):
                counts["deferred"] += 1
                pa["_status"] = "deferred"
            else:
                counts["yes"] += 1
                pa["_status"] = "yes"
        else:  # NO
            counts["no"] += 1
            pa["_status"] = "no"
            if first_failure is None:
                first_failure = pa

    if counts["no"] > 0:
        decision = "REJECT"
        passed = False
        rejection_reason = {
            "stage": "Tier 2c: Required Attribute Verification",
            "check_name": "required_attribute_failed",
            "message": (
                f"{counts['no']} required attribute(s) failed verification. "
                f"First: [{first_failure['scope']}] "
                f"\"{first_failure['attribute_text']}\" — "
                f"{first_failure['reasoning'][:160] if first_failure['reasoning'] else 'no evidence'}"
            ),
            "failed_fields": ["required_attributes"],
        }
    elif counts["deferred"] > 0:
        decision = "ACCEPT_WITH_DEFERRAL"
        passed = True
        rejection_reason = None
    else:
        decision = "ACCEPT"
        passed = True
        rejection_reason = None

    elapsed = time.perf_counter() - started
    result: Dict[str, Any] = {
        "decision": decision,
        "counts": counts,
        "per_attribute": per_attribute,
        "model": SONAR_MODEL,
        "elapsed_s": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rejection_reason is not None:
        result["rejection_reason"] = rejection_reason

    return passed, result
