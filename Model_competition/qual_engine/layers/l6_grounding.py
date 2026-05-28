"""L6 — Strict Grounding (10 gates) for a single (company, evidence_url, claim) tuple.

Returns a VerifiedSignal (passed) or None (rejected). Logs gate-level decisions
into the supplied tracer.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import EvidenceURL, ResolvedCompany, VerifiedSignal
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.providers.exa import ExaClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer
from qual_engine.validators.text_match import normalize_domain
from qual_engine.utils.ai_classifiers import is_negation, is_generic_marketing

import httpx as _httpx_for_verify

# Shadow the production scorer's exact 3-stage intent verifier inside L6.
# What passes here is what production will pass — eliminates "Intent
# fabrication detected" and "wrong_entity" rejects at scoring time.
from qualification.scoring.intent_verification_three_stage import (
    verify_three_stage,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Anti-bot / login-wall detector — narrowly tuned. Only catches pages that
# are clearly bot challenges or login walls (very short pages dominated by
# the challenge text). Real articles that happen to contain phrases like
# "sign in to subscribe" in a footer are not rejected.
#
# Tighter than production's check (which has a 4000-char ceiling) to avoid
# false-positives that drop legitimate own-newsroom evidence URLs.
_ANTIBOT_RE = re.compile(
    r"access denied|verifying your connection|verifying.{0,30}browser|just a moment|"
    r"additional verification required|verifying you are human|"
    r"this page (?:doesn.?t|does not) exist|"
    r"403\s*[-:|—]?\s*forbidden|404\s*[-:|—]?\s*(?:not\s*found|page.*not.*found)|"
    r"this content isn.?t available",
    re.IGNORECASE,
)
# Real articles are typically >2000 chars even when truncated. Bot/login walls
# are usually <500 chars of actual content. Set ceiling to 1500 to catch
# challenges without sweeping up truncated-but-legitimate articles.
_ANTIBOT_MAX_LEN = 1500


def _gate_1_url_reachable(content: str) -> tuple[bool, Optional[str]]:
    """Cheap reachability: length-based only.

    Anti-bot / login-wall detection is intentionally NOT enforced here —
    earlier attempts to mirror production's check at this gate caused false
    positives that dropped legitimate own-newsroom evidence (cached Exa
    /contents sometimes returns short markdown that contains "sign in"
    snippets from page chrome). Production's check is the authoritative
    one; ours would only filter redundantly. Negation / parked detection
    is unified into Gate 3 (LLM-driven).
    """
    if not content or len(content) < 100:
        return False, "content too short"
    return True, None


def _gate_2_company_linked(content: str, company: ResolvedCompany, url: str) -> tuple[bool, Optional[str]]:
    # Strong signal A: URL host matches primary domain
    url_host = normalize_domain(url)
    if url_host and company.primary_domain and url_host == company.primary_domain:
        return True, "host match"

    # Strong signal B: LinkedIn slug match
    if "linkedin.com/company/" in url and company.linkedin_slug:
        if company.linkedin_slug.lower() in url.lower():
            return True, "linkedin slug match"

    # Build alias list with canonical_name first; dedupe; case-insensitive — so
    # short canonical names aren't accidentally absorbed by similar aliases.
    aliases = [company.canonical_name]
    aliases.extend(a for a in (company.aliases or []) if a and a != company.canonical_name)
    seen = set()
    unique_aliases = []
    for a in aliases:
        k = a.lower().strip()
        if k and k not in seen:
            seen.add(k)
            unique_aliases.append(a)

    head = content[:6000].lower()
    name_hits = sum(head.count(a.lower()) for a in unique_aliases)
    if name_hits >= CONFIG.NAME_MENTIONS_REQUIRED:
        return True, f"{name_hits} name mentions"

    # Title match fallback
    title_block = content[:300].lower()
    if any(a.lower() in title_block for a in unique_aliases):
        return True, "title match"

    return False, f"insufficient linkage (host={url_host}, hits={name_hits})"


async def _gate_3_negation(
    content: str, claim: str, openrouter, cost
) -> tuple[bool, Optional[str]]:
    """LLM-driven negation/contradiction check. Replaces hardcoded regex list."""
    negated, evidence = await is_negation(content, claim, openrouter, cost)
    if negated:
        return False, f"negation: {evidence!r}"
    return True, None


def _gate_4_anchor_present(company: ResolvedCompany) -> tuple[bool, Optional[str]]:
    """Gate 4: candidate must have been verified at L3 as a real operating company,
    either via LinkedIn slug+name+domain match OR via website-anchor with LLM
    verification (both paths validate the company's identity)."""
    if company.linkedin_slug or company.linkedin_url:
        return True, "linkedin anchor"
    if getattr(company, "anchor_source", None) == "website" and company.primary_domain:
        return True, "website anchor"
    return False, "no anchor"


async def _gate_5_grounding(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    company: ResolvedCompany,
    icp_signal_text: str,
    url: str,
    content: str,
) -> Optional[dict]:
    """Strict, semantic-aware grounding.

    Primary: Claude Sonnet 4.6 (high accuracy, can be conservative on non-funding).
    Tier-2 fallback: when Sonnet says supported=false, retry with gpt-4o-mini
    (less strict). If tier-2 accepts AND has a valid proof_quote, return it
    with confidence capped at 70 so the downstream ranker can weight accordingly.
    Defense-in-depth (Gate 6 anti-hallucination, Gate 7 Sonar cross-check, Gate
    9 negation proxy) catches false positives the tier-2 model might admit.
    """
    prompt = f"""You are a fact-checker for a B2B sales-intelligence pipeline. You verify that a page contains evidence the company actually did the claimed activity. Allow reasonable semantic equivalence; reject only when the content is unrelated to the claim or contradicts it.

URL: {url}
CONTENT (≤6000 chars):
{content[:6000]}

CLAIM: "{company.canonical_name} {icp_signal_text}"

Decide: does the CONTENT support the CLAIM with at least one specific sentence?

ACCEPT as evidence:
- A dated factual record (press release, news article, blog post, company newsroom, public filing, posted job listing, customer case study) that describes the company doing the claimed activity. The event may be a recent COMPLETED occurrence OR a clearly-described ongoing one — both count.
- A specific, more-detailed instance in the content that satisfies a broader category in the claim (e.g., claim says "expanded to new markets"; content says "launched in three new states this quarter" — that counts).
- An adjacent-but-aligned activity that a B2B salesperson would consider the same signal class. Examples:
  * Claim "launched new product"; content describes a specific new product line, feature suite, or major release with a name and date → ACCEPT.
  * Claim "expanded to new markets"; content describes entering new geographies, customer segments, or product lines → ACCEPT.
  * Claim "raised funding"; content describes ANY closed round (Seed/A/B/C/D/PE/debt facility) → ACCEPT (stage check happens later).
  * Claim "leadership change"; content describes a new C-suite hire, board appointment, or executive transition → ACCEPT.
  * Claim "digital transformation commentary"; content includes a quote, interview, or blog post from the company's leadership describing their transformation strategy or modernization initiative → ACCEPT.
- Content from the company's OWN newsroom/blog about the claimed activity is high-trust evidence.

REJECT as evidence:
- Generic marketing copy with no specific activity (broad statements of values or mission alone).
- Pure future intent ("we plan to…", "we are considering…") with no past or ongoing action.
- The page describes a DIFFERENT company than the candidate (mention of the candidate is only in passing — e.g., as an investor, partner, or comparison).
- Content directly contradicts the claim (e.g., "we are NOT pursuing X").

When you accept, the proof_quote MUST be a sentence copy-able VERBATIM from the content that captures the supporting evidence.

Output JSON only (no code fences):
{{
  "supported": true|false,
  "proof_quote": "<exact sentence from CONTENT, ≤300 chars, copied verbatim>",
  "confidence": <0-100>,
  "extracted_date": "<YYYY-MM-DD or null>",
  "rejection_reason": "<short, if not supported>"
}}"""
    r = await openrouter.json_call(
        CONFIG.GROUNDING_MODEL, prompt, label="L6_grounding"
    )
    cost.add("openrouter", r["cost_usd"], layer="L6")
    parsed = r["parsed"]
    if not isinstance(parsed, dict):
        return None

    # Tier-2 fallback: when Sonnet rejected (supported=false), retry with a less
    # strict model. If it accepts AND extracts a proof_quote, return that with
    # confidence capped at 70.
    if not parsed.get("supported"):
        try:
            r2 = await openrouter.json_call(
                CONFIG.TRIAGE_MODEL, prompt, label="L6_grounding_tier2"
            )
            cost.add("openrouter", r2["cost_usd"], layer="L6")
            t2 = r2.get("parsed")
            if isinstance(t2, dict) and t2.get("supported") and (t2.get("proof_quote") or "").strip():
                t2["confidence"] = min(int(t2.get("confidence") or 70), 70)
                t2["_tier2"] = True
                return t2
        except Exception as e:
            logger.warning("L6 tier-2 grounding failed: %s", e)
    return parsed


def _gate_6_anti_hallucination(proof_quote: str, content: str, fuzzy_threshold: float = 0.92) -> bool:
    if not proof_quote:
        return False
    nq = " ".join(proof_quote.split())
    nc = " ".join(content.split())
    if nq in nc:
        return True
    # Fuzzy fallback for minor LLM rewrites (smart-quote substitution, etc.)
    try:
        from difflib import SequenceMatcher
    except Exception:
        return False
    if len(nq) < 20:
        return False
    best = 0.0
    step = max(50, len(nq) // 4)
    for i in range(0, max(1, len(nc) - len(nq)), step):
        window = nc[i : i + len(nq) + 100]
        ratio = SequenceMatcher(None, nq, window).ratio()
        if ratio > best:
            best = ratio
        if best > fuzzy_threshold:
            return True
    return False


async def _gate_7_cross_check(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    company: ResolvedCompany,
    icp_signal_text: str,
) -> tuple[bool, bool]:
    """Returns (passed, corroborated). passed=False means hard contradiction; UNKNOWN passes."""
    prompt = f"""Verify whether this is currently true.

Question: Is it currently true that {company.canonical_name} {icp_signal_text}?

Strict rules:
- Use real-time web search
- Answer YES with sources if you find direct evidence
- Answer NO with sources if you find contradicting evidence
- Answer UNKNOWN if you cannot determine

Output format (no other text):
ANSWER: <YES|NO|UNKNOWN>
EVIDENCE: <one sentence>
SOURCES: <url1>, <url2>"""
    r = await openrouter.chat(CONFIG.SONAR_VERIFY_MODEL, prompt, label="L6_cross_check")
    cost.add("openrouter", r["cost_usd"], layer="L6")
    text = r["text"]

    # Parse: find ANSWER: <YES|NO|UNKNOWN> as the primary verdict
    m = re.search(r"answer\s*:?\s*(yes|no|unknown)\b", text, re.IGNORECASE)
    raw = (m.group(1).upper() if m else "UNKNOWN")

    if raw == "YES":
        return True, True
    if raw == "UNKNOWN":
        return True, False
    # NO: ask an LLM whether this is a CONFIDENT NO or a hedged one.
    # No hardcoded phrase list.
    from qual_engine.utils.ai_classifiers import is_confident_no
    confident = await is_confident_no(text, openrouter, cost)
    if confident:
        return False, False    # confident NO → reject
    return True, False         # hedged NO → pass without corroboration


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return None


def _gate_8_date_sanity(
    extracted: Optional[date],
    content: str,
    is_time_bound: bool,
    time_window_days: Optional[int],
) -> tuple[bool, Optional[date]]:
    if extracted is None:
        return True, None
    if extracted > date.today():
        return False, None
    if (date.today() - extracted).days > 365 * 3:
        return False, None
    # Cover many date format variants, including non-padded day numbers — real
    # prose says "February 9, 2026" not "February 09, 2026".
    import calendar as _cal

    y, m, d = extracted.year, extracted.month, extracted.day
    month_full = _cal.month_name[m]    # "February"
    month_abbr = _cal.month_abbr[m]    # "Feb"

    variants = [
        extracted.isoformat(),                # 2026-02-09
        f"{month_full} {d}, {y}",             # February 9, 2026
        f"{month_full} {d:02d}, {y}",         # February 09, 2026
        f"{month_full} {d} {y}",              # February 9 2026
        f"{month_abbr} {d}, {y}",             # Feb 9, 2026
        f"{month_abbr} {d:02d}, {y}",         # Feb 09, 2026
        f"{month_abbr}. {d}, {y}",            # Feb. 9, 2026
        f"{d} {month_full} {y}",              # 9 February 2026
        f"{d} {month_abbr} {y}",              # 9 Feb 2026
        f"{y}/{m:02d}/{d:02d}",               # 2026/02/09
        f"{y}/{m}/{d}",                       # 2026/2/9
        f"{m}/{d}/{y}",                       # 2/9/2026
        f"{m:02d}/{d:02d}/{y}",               # 02/09/2026
        f"{d}/{m}/{y}",                       # 9/2/2026
        f"{m}-{d}-{y}",                       # 2-9-2026
        f"{month_full} {y}",                  # February 2026 (month + year only)
    ]
    found = any(v and v in content for v in variants)
    if not found:
        return False, None
    if is_time_bound and time_window_days:
        if (date.today() - extracted).days > time_window_days:
            return False, None
    return True, extracted


async def _gate_9_negative_claim(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    company: ResolvedCompany,
    icp_signal_text: str,
) -> bool:
    """For 'does NOT have X' claims, verify positive search FAILS. Returns True if passes."""
    lower = icp_signal_text.lower()
    is_negative = any(
        kw in lower for kw in ["not have", "no longer", "does not", "doesn't", "not using", "without"]
    )
    if not is_negative:
        return True  # not a negative claim → gate passes trivially
    # Construct the positive form
    positive = (
        icp_signal_text.lower()
        .replace("does not", "does")
        .replace("doesn't", "does")
        .replace("not have", "have")
        .replace("no longer", "")
        .replace("not using", "using")
        .replace("without", "with")
    )
    prompt = f"""Question: Is it currently true that {company.canonical_name} {positive}?
Answer ONLY 'YES' or 'NO' or 'UNKNOWN'."""
    r = await openrouter.chat(CONFIG.SONAR_VERIFY_MODEL, prompt, label="L6_neg_claim")
    cost.add("openrouter", r["cost_usd"], layer="L6")
    text = r["text"].lower()
    # If the positive form is YES → the negative claim is contradicted
    if "yes" in text.split("\n")[0]:
        return False
    return True


async def _gate_10_generic(description: str, openrouter, cost) -> bool:
    """LLM-driven generic-fluff check. Replaces hardcoded regex list."""
    generic = await is_generic_marketing(description, openrouter, cost)
    return not generic


_STAGE_SANITY_PROMPT = """The buyer's ICP requires a specific company funding stage. A piece of verified evidence contains text describing a funding event. Decide whether the stage mentioned in the evidence is COMPATIBLE with the ICP target stage.

ICP target stage     : {target_stage}
Evidence proof_quote : "{proof_quote}"

Apply your knowledge of how funding stages map (Seed, Series A/B/C/D+, growth-stage, public, bootstrapped, PE-owned, etc.). The proof_quote may mention a specific round, a total funding amount, or both.

Decide:
- "compatible" : the proof_quote's stage matches or is adjacent enough to the ICP target stage (off by one stage is borderline-acceptable; the salesperson would still consider this company).
- "conflict"   : the proof_quote names a stage that is clearly different from the ICP target (multiple stages away, or completely different category like Public when target is Series A).
- "unclear"   : the proof_quote does not name a stage at all.

When "unclear", do NOT manufacture a conflict — return unclear.

Output JSON only:
{{"verdict": "compatible|conflict|unclear", "stage_in_proof": "<short label or null>", "reason": "<≤12 words>"}}"""


async def _gate_stage_sanity(
    target_stage: str, proof_quote: str, openrouter, cost,
) -> tuple[bool, Optional[str]]:
    """Stage sanity: if ICP has a target stage AND the proof_quote names a
    conflicting stage, reject. Unclear or compatible → pass."""
    if not target_stage or not proof_quote:
        return True, None
    try:
        from qual_engine.config import CONFIG
        prompt = _STAGE_SANITY_PROMPT.format(
            target_stage=target_stage, proof_quote=proof_quote[:400],
        )
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="stage_sanity")
        cost.add("openrouter", r.get("cost_usd", 0), layer="L6")
        parsed = r.get("parsed")
        if isinstance(parsed, dict):
            verdict = parsed.get("verdict")
            if verdict == "conflict":
                return False, f"stage_conflict: proof has {parsed.get('stage_in_proof')!r}, ICP wants {target_stage!r}"
    except Exception:
        pass
    return True, None


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

async def ground(
    openrouter: OpenRouterClient,
    exa: ExaClient,
    cost: CostTracker,
    tracer: Tracer,
    *,
    company: ResolvedCompany,
    evidence: EvidenceURL,
    icp_signal_text: str,
    icp_signal_idx: int,
    is_time_bound: bool = False,
    time_window_days: Optional[int] = None,
    target_stage: str = "",
) -> Optional[VerifiedSignal]:
    """Run gates 1..10. Returns VerifiedSignal or None."""

    url = evidence.url

    # ─── MANDATORY: evidence source must be a valid HTTP(S) URL ───
    if not url or not url.startswith(("http://", "https://")):
        tracer.emit("L6", "gate0_url_present", url=url, ok=False)
        return None

    # Gate 1: reachable + content present
    content = evidence.raw_content
    if not content:
        fetched = await exa.contents(url)
        cost.add("exa", fetched.get("cost_usd", 0), layer="L6")
        content = fetched.get("text", "")
    ok, reason = _gate_1_url_reachable(content or "")
    tracer.emit("L6", "gate1_reachable", url=url, ok=ok, reason=reason)
    if not ok:
        return None

    # Gate 2: company linkage
    ok, reason = _gate_2_company_linked(content, company, url)
    tracer.emit("L6", "gate2_linkage", url=url, ok=ok, reason=reason)
    if not ok:
        return None

    # Gate 3: negation (LLM)
    ok, reason = await _gate_3_negation(content, icp_signal_text, openrouter, cost)
    tracer.emit("L6", "gate3_negation", url=url, ok=ok, reason=reason)
    if not ok:
        return None

    # Gate 4: anchor present (LinkedIn OR verified website)
    ok, reason = _gate_4_anchor_present(company)
    tracer.emit("L6", "gate4_anchor", ok=ok, reason=reason)
    if not ok:
        return None

    # Gate 5: strict grounding
    grounding = await _gate_5_grounding(
        openrouter, cost, company, icp_signal_text, url, content
    )
    sup = bool(grounding and grounding.get("supported"))
    conf = int((grounding or {}).get("confidence", 0))
    tracer.emit(
        "L6", "gate5_grounding", url=url, supported=sup, confidence=conf,
        reason=(grounding or {}).get("rejection_reason"),
    )
    if not sup or conf < CONFIG.GROUNDING_MIN_CONFIDENCE:
        return None

    # Gate 6: anti-hallucination
    proof = (grounding.get("proof_quote") or "").strip()
    # Synthesized SD-LinkedIn-jobs content may have minor LLM rewrites of canned text →
    # loosen the fuzzy threshold for those evidence items.
    is_synthesized = (
        evidence.raw_content is not None
        and evidence.discovered_via == "sd_linkedin_jobs"
    )
    fuzzy_thresh = 0.85 if is_synthesized else 0.92
    quote_ok = _gate_6_anti_hallucination(proof, content, fuzzy_threshold=fuzzy_thresh)
    tracer.emit("L6", "gate6_anti_halluc", ok=quote_ok)
    if not quote_ok:
        return None

    # Gate 6c: shadow the production 3-stage intent verifier. The production
    # scorer runs Sonar → ScrapingDog → Sonar-pro on every signal it scores.
    # Mirror that exact verification here so any candidate that would be
    # rejected at scoring time gets dropped before L7 / L8 even sees it.
    # Skip for synthesized SD-LinkedIn-jobs content (the production verifier
    # has a separate path for those that we don't want to double-handle).
    if evidence.discovered_via != "sd_linkedin_jobs":
        try:
            async with _httpx_for_verify.AsyncClient(timeout=120) as _vc:
                v = await verify_three_stage(
                    _vc,
                    company_name=company.canonical_name,
                    company_linkedin=company.linkedin_url or "",
                    company_website=(
                        f"https://{company.primary_domain}"
                        if company.primary_domain else ""
                    ),
                    source_url=url,
                    miner_claim=proof,
                    target_signal_text=icp_signal_text,
                )
            tracer.emit(
                "L6", "gate6c_prod_verify",
                url=url, decision=v.get("decision"),
                client_ready=v.get("client_ready"),
                reason=v.get("rejection_reason"),
            )
            if not v.get("client_ready"):
                return None
        except Exception as e:
            # Fail-OPEN on infrastructure errors so transient issues don't
            # punish candidates. Production may still catch them.
            tracer.emit(
                "L6", "gate6c_prod_verify",
                url=url, skipped="verifier_exception", error=str(e)[:200],
            )

    # Gate 6b: stage sanity — when ICP has a target stage and the proof_quote names
    # a clearly conflicting stage (e.g., proof says "Seed" but ICP wants "Series B"),
    # reject. Unclear/compatible → pass.
    stage_ok, stage_reason = await _gate_stage_sanity(target_stage, proof, openrouter, cost)
    tracer.emit("L6", "gate6b_stage_sanity", ok=stage_ok, reason=stage_reason)
    if not stage_ok:
        return None

    # Gate 8 (date sanity) before Gate 7 (expensive Sonar) — saves cost on bad dates
    extracted_date = _parse_date(grounding.get("extracted_date"))
    ok, final_date = _gate_8_date_sanity(extracted_date, content, is_time_bound, time_window_days)
    tracer.emit(
        "L6", "gate8_date", ok=ok,
        extracted=str(extracted_date), final=str(final_date),
    )
    if not ok:
        return None

    # Gate 9: negative-claim proxy
    ok = await _gate_9_negative_claim(openrouter, cost, company, icp_signal_text)
    tracer.emit("L6", "gate9_neg_claim", ok=ok)
    if not ok:
        return None

    # Gate 10: generic-fluff check (LLM)
    description = _summarize_for_description(proof, icp_signal_text)
    ok = await _gate_10_generic(description, openrouter, cost)
    tracer.emit("L6", "gate10_generic", ok=ok)
    if not ok:
        return None

    # Gate 7: independent cross-check (expensive, last)
    passed, corroborated = await _gate_7_cross_check(
        openrouter, cost, company, icp_signal_text
    )
    tracer.emit("L6", "gate7_cross_check", passed=passed, corroborated=corroborated)
    if not passed:
        return None

    # ─── MANDATORY: proof_quote must be non-empty + URL must be present ───
    if not proof or not url:
        tracer.emit("L6", "final_guard", reason="empty proof or url", url=url)
        return None

    return VerifiedSignal(
        url=url,
        source_type=evidence.source_type,
        date=final_date,
        snippet=proof[:500],
        description=description[:500],
        matched_icp_signal_idx=icp_signal_idx,
        grounding_confidence=conf,
        sonar_corroborated=corroborated,
        proof_quote=proof[:500],
    )


def _summarize_for_description(proof_quote: str, icp_signal_text: str) -> str:
    """Cheap summarization: prepend intent context, trim to sentence."""
    proof = proof_quote.strip()
    if len(proof) <= 200:
        return proof
    return proof[:200].rsplit(" ", 1)[0] + "…"
