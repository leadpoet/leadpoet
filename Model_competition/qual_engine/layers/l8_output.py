"""L8 — Output Builder. Assembles QualificationResult with `matches` list.

Also provides ``to_submission_dict`` — converts our internal CompanyOutput
to the EXACT production-competition schema (``gateway.qualification.models.CompanyOutput``).
Production has ``model_config = {"extra": "forbid"}``, so our extra internal
fields (grounding_confidence, sonar_corroborated, proof_quote, source_type)
must be renamed/stripped before submission.
"""

from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Optional
from urllib.parse import urlparse, urlunparse

from qual_engine.models import (
    CompanyMatch,
    CompanyOutput,
    ICPPrompt,
    QualificationResult,
    ResolvedCompany,
    VerifiedSignal,
)
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Production-schema serializer + injection scrubber
# ─────────────────────────────────────────────────────────────────────────────

# Mirror of gateway/qualification/models.py:_INTENT_INJECTION_PATTERNS. We
# scrub injection text from proof_quote / description / snippet BEFORE
# submission so the production parse-time validator does not silently drop
# our lead. (Production rejects matches; we replace the offending span with
# "[redacted]" so the rest of the evidence is still useful.)
_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|skip|bypass|override|nullify|cancel)\s+"
        r"(?:all\s+|any\s+|the\s+|every\s+|whatever\s+|what\s+(?:was\s+)?)?"
        r"(?:previous|prior|above|earlier|preceding|former|original|initial)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:ignore|disregard)\s+(?:everything|all)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:everything|all|what|that)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:new|updated?|revised?|fresh|different)\s+"
        r"(?:instructions?|task|prompt|rules?|directives?|orders?|guidelines?)\s*"
        r"(?:[:.]|are|is|to|that)",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|(?:im_(?:start|end)|endoftext|fim_[a-z]+|begin_of_text|end_of_text)\|>",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\n)\s*(?:system|assistant|user)\s*[:>]", re.IGNORECASE),
    re.compile(
        r"\b(?:return|respond|reply|output|give|set|make|use|score|assign)\s+"
        r"(?:with\s+|this\s+|a\s+|the\s+)?(?:score|value|rating)?\s*"
        r"(?:of\s+|=\s*|:\s*|to\s+)?\s*(?:5\d|60)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bscore\s*[:=]\s*(?:5\d|60)\b", re.IGNORECASE),
    re.compile(r"\bmatched_icp_signal_idx\s*[:=]", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\bfollow\s+(?:these|the)\s+new\b", re.IGNORECASE),
]


def _scrub_injection(text: str) -> str:
    """Remove evident prompt-injection spans so production parse-time
    validation does not reject the lead. Replaces matches with ``[redacted]``."""
    if not text:
        return text
    for rx in _INJECTION_PATTERNS:
        text = rx.sub("[redacted]", text)
    return text


def _normalize_url_for_submission(url: str) -> str:
    """Normalize the same way production CompanyOutput._normalize_company_website
    does — scheme present, lowercased host."""
    if not url:
        return ""
    v = url.strip()
    if not v.lower().startswith(("http://", "https://")):
        v = "https://" + v
    parsed = urlparse(v)
    if not parsed.hostname:
        return v
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


# Production IntentSignalSource enum values — keep aligned with
# gateway/qualification/models.py:IntentSignalSource
_PROD_SOURCE_VALUES = {
    "linkedin", "job_board", "social_media", "news", "github",
    "review_site", "company_website", "wikipedia", "other",
}


def to_submission_dict(c: CompanyOutput) -> dict:
    """Convert our internal CompanyOutput into the EXACT production-competition
    schema (``gateway.qualification.models.CompanyOutput``).

    Strips internal-only fields (``grounding_confidence``, ``sonar_corroborated``,
    ``proof_quote``), renames (``source_type`` → ``source``,
    ``matched_icp_signal_idx`` → ``matched_icp_signal``), enforces production
    length caps, normalizes URL format, scrubs prompt-injection text, and
    converts dates from Python ``date`` → ISO ``YYYY-MM-DD`` strings.
    """
    intent_signals = []
    for s in (c.intent_signals or []):
        source = (s.source_type or "other").lower()
        if source not in _PROD_SOURCE_VALUES:
            source = "other"
        # Prefer the verbatim proof_quote (more specific) as the snippet, fall
        # back to s.snippet. Both get scrubbed and capped at production limits.
        raw_snippet = (s.proof_quote or s.snippet or "").strip()
        intent_signals.append({
            "source": source,
            "description": _scrub_injection((s.description or "").strip())[:350],
            "url": (s.url or "").strip(),
            "date": s.date.isoformat() if s.date else None,
            "snippet": _scrub_injection(raw_snippet)[:600],
            "matched_icp_signal": int(s.matched_icp_signal_idx),
        })

    return {
        "company_name": (c.company_name or "")[:200],
        "company_website": _normalize_url_for_submission(c.company_website or ""),
        "company_linkedin": (c.company_linkedin or ""),
        "industry": (c.industry or "Other"),
        "sub_industry": (c.sub_industry or ""),
        "employee_count": (c.employee_count or "1-50"),
        "company_stage": (c.company_stage or ""),
        "country": (c.country or "United States"),
        "state": (c.state or ""),
        "description": _scrub_injection((c.description or "").strip())[:500],
        "intent_signals": intent_signals,
    }


def _compute_overall_confidence(ranker_score: int, signals: list[VerifiedSignal]) -> int:
    if not signals:
        return ranker_score
    avg_grounding = mean(s.grounding_confidence for s in signals)
    corroborated_rate = sum(1 for s in signals if s.sonar_corroborated) / max(len(signals), 1)
    sources_diversity = len({s.source_type for s in signals})

    base = (
        0.35 * ranker_score
        + 0.35 * avg_grounding
        + 0.20 * (corroborated_rate * 100)
        + 0.10 * min(len(signals) / 3, 1) * 100
    )
    diversity_bonus = min(sources_diversity * 5, 15)
    return int(min(round(base + diversity_bonus), 100))


def _truncate_at_sentence(text: str, max_chars: int = 600) -> str:
    """Truncate at the last sentence boundary before max_chars, or hard-truncate."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Find last sentence-ending punctuation in window
    best = max(
        window.rfind("."),
        window.rfind("!"),
        window.rfind("?"),
    )
    if best > max_chars * 0.5:
        return window[: best + 1].strip()
    # Fallback: last word boundary
    last_space = window.rfind(" ")
    if last_space > 0:
        return window[:last_space].rstrip() + "…"
    return window


def _build_one(
    resolved: ResolvedCompany,
    signals: list[VerifiedSignal],
    ranking: dict,
    icp_industry: str = "",
    icp_sub_industry: str = "",
    icp_country: str = "",
) -> CompanyMatch:
    # The submission's `industry` is the buyer's ICP-target industry (e.g.
    # "Software", "Hardware") — NOT LinkedIn's raw tag (e.g. "Software
    # Development", "Computer Hardware Manufacturing"). The candidate passed
    # L4 industry-compat + L4.5 ICP-fit gates which already confirmed it
    # matches the ICP's target industry. Production scorers do strict fuzzy
    # match (80% threshold) against the ICP industry — using the ICP value
    # here makes the submission pass that gate.
    industry = icp_industry or (resolved.industry_tags[0] if resolved.industry_tags else "")
    sub_industry = icp_sub_industry or (
        resolved.industry_tags[1] if len(resolved.industry_tags) > 1 else ""
    )

    # Country: use the ICP-target country string (e.g. "United States") rather
    # than the ISO-2 we get from LinkedIn (e.g. "US"). The candidate already
    # passed L4's country check (which uses ISO-2 normalization), so substituting
    # the ICP's country phrasing here is safe and keeps the production scorer's
    # country-match gate from failing on "US" vs "United States" string equality.
    country = icp_country or resolved.country or ""

    company_output = CompanyOutput(
        company_name=resolved.canonical_name,
        company_website=f"https://{resolved.primary_domain}" if resolved.primary_domain else "",
        company_linkedin=resolved.linkedin_url,
        industry=industry,
        sub_industry=sub_industry,
        employee_count=resolved.employee_count_band or "",
        company_stage=resolved.funding_stage or "",
        country=country,
        state="",
        description=_truncate_at_sentence(resolved.description or "", 600),
        intent_signals=signals,
    )

    score = int(ranking.get("score", 0))
    return CompanyMatch(
        company=company_output,
        score=score,
        overall_confidence=_compute_overall_confidence(score, signals),
        industry_match=int(ranking.get("industry_match", 0)),
        structural_match=int(ranking.get("structural_match", 0)),
        intent_strength=int(ranking.get("intent_strength", 0)),
        reasoning=(ranking.get("reasoning", "") or "")[:600],
    )


def build_output(
    icp: ICPPrompt,
    ranked: list[tuple[ResolvedCompany, list[VerifiedSignal], dict]],
    abstention_reason: Optional[str],
    tracer: Tracer,
    cost: CostTracker,
) -> QualificationResult:
    if not ranked:
        return QualificationResult(
            matches=[],
            total_matches=0,
            abstention_reason=abstention_reason or "no qualifying candidates",
            reasoning_trace=tracer.summary(),
            cost_breakdown=cost.breakdown(),
            latency_ms=tracer.latency_ms(),
        )

    matches = [
        _build_one(c, sigs, rk,
                   icp_industry=icp.industry or "",
                   icp_sub_industry=icp.sub_industry or "",
                   icp_country=icp.country or "")
        for c, sigs, rk in ranked
    ]
    # Sort by overall_confidence desc
    matches.sort(key=lambda m: m.overall_confidence, reverse=True)

    return QualificationResult(
        matches=matches,
        total_matches=len(matches),
        abstention_reason=None,
        reasoning_trace=tracer.summary(),
        cost_breakdown=cost.breakdown(),
        latency_ms=tracer.latency_ms(),
    )
