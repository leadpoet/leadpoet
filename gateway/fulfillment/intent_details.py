"""
Intent Details passage generator (LLM).

Runs on the gateway once per winning fulfillment lead, after consensus.
Synthesizes the miner's verified intent signals into a single
client-ready paragraph for the "Intent Details" UI column.

Model: ``openai/gpt-4o-mini`` via OpenRouter.
Key:   ``FULFILLMENT_OPENROUTER_API_KEY`` (already wired into the gateway env).

Why not Perplexity sonar-pro (previous model)?  sonar-pro is a
web-search-augmented model — when its search returns no hits for the
company name, it *refuses* and emits strings like "I need to clarify..."
or "the search results don't contain information about...".  That's the
correct bias for an open-web Q&A product but wrong for us: we already
have the evidence (miner-submitted intent signals with URLs, snippets,
and dates) and just want faithful synthesis.  gpt-4o-mini is non-search,
deterministic-leaning, cheaper, faster, and much better at staying
strictly inside the provided context.

This module is self-contained: no dependency on miner-side helpers
(``target_fit_model`` / ``openrouter.py``) which aren't deployed on the
gateway.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TIMEOUT_SECONDS = 60

# Label used when a credited signal (after_decay_score > 0) arrives without
# a specific matched_icp_signal — gives the LLM and the customer-facing UI
# a stable header instead of an empty string.
UNTAGGED_SIGNAL_LABEL = "Aligned with overall buyer profile"


# Prompt text closely mirrors the exact rule list the client uses when
# hand-crafting Intent Details in Perplexity.  If you adjust a rule, this
# should be a product decision synced with whatever's in the client-facing
# UI spec, not a silent code change.
_PROMPT_TEMPLATE = """Produce two outputs for the company below: (A) one overall Intent Details paragraph, and (B) one client-ready explanation per intent signal. Both grounded strictly in the provided signals.

Rules (strictly enforced for BOTH outputs):
- No preamble, no apology, no disclaimer, no mention of "search results", "available information", or your own limitations.
- Synthesize ONLY from the intent signals provided below. Do not invent facts, do not speculate beyond what the signals support, and do not reference any source outside the inputs.
- Stay strictly inside the provided evidence. NEVER introduce specific quantities — dollar amounts, percentages, headcounts, round sizes, valuations, role counts, dates, or company stage labels — that are not explicitly stated in the Summary or Evidence fields below. If the snippet says "secured seed funding" with no dollar figure, write "secured seed funding"; do NOT write "raised $10 million" or any other invented number. If the snippet says "expanded their team" with no headcount, do NOT specify a headcount.
- Stay close to the wording the snippet actually uses. You may rephrase for fluency, but every concrete claim (the WHAT, WHEN, HOW MUCH) must trace back to a verbatim phrase in the Summary or Evidence above. If something is not in the evidence, leave it out.
- Each output must be expanded with rich buying-signal context explaining why the observed activity indicates relevance for the ICP — but that context must come from interpreting the signals provided, NOT from adding new facts about the company. Explain why the signal matters, not what additionally exists.
- Use natural prose. No bullets, no labels inside the prose, no markdown, no em dashes, no links, no citations.
- Do not restate the client name or ICP facts (country, employee count, industry, etc.).
- If a claim cannot be supported by the provided signals, omit it — write shorter rather than fabricating detail.
- Output must be client-ready for direct use in the UI.

Inputs:

ICP:
{icp_block}

Intent Signals (the only evidence you may cite; numbered 1..N):
{signals_block}

Output format (return EXACTLY this structure; the === markers are required):

=== OVERALL ===
<single natural paragraph synthesizing all the signals into one buying-intent narrative>

=== SIGNAL 1 ===
<single short paragraph explaining how signal 1 specifically indicates buying intent for the ICP>

=== SIGNAL 2 ===
<single short paragraph for signal 2>

(continue for every numbered signal; do not skip any; do not invent extras)"""


# Patterns that indicate the LLM emitted meta-commentary / refusal instead
# of a real passage.  If the first ~200 characters match any of these, we
# discard the output and return empty — the caller (lifecycle) treats that
# as "skip this lead, don't persist intent_details".
_REFUSAL_RE = re.compile(
    r"\b(i need to clarify|i appreciate (the )?(detailed )?request|"
    r"i apologi[sz]e|i(?:'m| am) (unable|sorry|not able)|"
    r"unfortunately (i|we) (cannot|can't|don't|do not)|"
    r"i (cannot|can't|don't|do not) (generate|produce|find|have)|"
    r"the (provided )?(search results|available information|given "
    r"(data|information|context|signals|search results)) (do not|does not|don't|doesn't) "
    r"contain|no (verifiable |factual |specific |explicit )?"
    r"(information|data|evidence|details) (about|on|regarding|for) "
    r"[a-z0-9][\w\s&.,'()-]*? (is|was|has been|could|can) "
    r"(found|available|provided|located|retrieved))",
    flags=re.IGNORECASE,
)


def _format_intent_signals_for_icp_block(val: Any) -> str:
    """Render ``icp.intent_signals`` for the LLM's ICP block.

    Accepts the legacy ``List[str]`` shape and the structured dict shape
    (JSONB-serialized ``IntentSignalSpec`` rows). Renders optional
    ``[REQUIRED]`` suffix so prose generation knows which gates are hard.
    """
    if not val:
        return ""
    if not isinstance(val, list):
        return str(val)
    rendered: List[str] = []
    for entry in val:
        if isinstance(entry, dict):
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            required = bool(entry.get("required", False))
            suffix = " [REQUIRED]" if required else ""
            rendered.append(f"{text}{suffix}")
        else:
            rendered.append(str(entry))
    return "; ".join(rendered)


def _format_icp_block(icp: Dict[str, Any]) -> str:
    """Render the ICP dict in a compact, LLM-friendly block."""
    if not icp:
        return "(no ICP provided)"

    ordered_fields = [
        ("prompt", "Prompt"),
        ("industry", "Industry"),
        ("sub_industry", "Sub-industry"),
        ("target_role_types", "Target role types"),
        ("target_roles", "Target roles"),
        ("target_seniority", "Target seniority"),
        ("employee_count", "Employee count"),
        ("company_stage", "Company stage"),
        ("geography", "Geography"),
        ("country", "Country"),
        ("product_service", "Product / service"),
        ("intent_signals", "Expected intent signals"),
    ]

    lines = []
    for key, label in ordered_fields:
        val = icp.get(key)
        if val in (None, "", [], {}):
            continue
        # intent_signals is structured (dicts with required flag).
        # Render so the LLM grounds
        # its intent_details prose in the contract. Every other list
        # field stays comma-joined for compactness.
        if key == "intent_signals":
            rendered = _format_intent_signals_for_icp_block(val)
            if not rendered:
                continue
            lines.append(f"- {label}: {rendered}")
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "(empty ICP)"


def _compute_keepers(signals: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """Return ``[(source_index, signal), ...]`` for signals that passed scoring.

    ``source_index`` is the 0-based position in the ORIGINAL
    ``intent_signal_mapping`` array.  Filtered list preserves that position
    so the per-signal breakdown can store a ``source_index`` field pointing
    at the correct raw evidence row even when failed signals are skipped.

    Returns an empty list when no signal has ``after_decay_score > 0`` (or
    ``raw_score > 0`` as a fallback for legacy rows missing the decay
    field).  In that case the caller renders "(no intent signals)" to the
    LLM and the lifecycle persists nothing — by design.  A lead that won
    consensus must have ``intent_final > 0`` and therefore at least one
    passing signal, so this empty return should not fire in production for
    a legitimate winner; the explicit empty path exists so a failed-signal
    edge case never produces per-signal explanations for evidence that did
    not qualify.
    """
    raw = list(signals or [])
    return [
        (i, s) for i, s in enumerate(raw)
        if float(s.get("after_decay_score") or s.get("raw_score") or 0) > 0
    ]


def _format_signals_block(signals: List[Dict[str, Any]]) -> str:
    """Render the verified intent signals as a numbered block for the LLM.

    Each ``signal`` entry should have the shape produced by
    ``gateway/fulfillment/scoring.py`` -> ``intent_signals_detail`` items:
    ``url, description, snippet, date, source, matched_icp_signal``.
    Unscored / zero-score signals are filtered out so the LLM only sees the
    evidence that actually counted.  The 1-based LLM-facing number is the
    position in the filtered list; the parser separately stores a
    ``source_index`` so callers can locate the raw row in the original
    intent_signal_mapping.
    """
    keepers = _compute_keepers(signals)
    if not keepers:
        return "(no intent signals)"

    lines = []
    for i, (_src_idx, s) in enumerate(keepers, 1):
        desc = (s.get("description") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        date = s.get("date") or "n/a"
        matched = (s.get("matched_icp_signal") or "").strip() or UNTAGGED_SIGNAL_LABEL
        source = s.get("source") or "n/a"
        header = f"{i}. Source: {source}"
        header += f"  |  Matches ICP signal: \"{matched}\""
        header += f"  |  Date: {date}"
        lines.append(header)
        if desc:
            lines.append(f"   Summary: {desc}")
        if snippet:
            # Trim snippet so we don't blow the context window on long pages.
            trimmed = snippet if len(snippet) <= 1500 else snippet[:1500] + "..."
            lines.append(f"   Evidence: {trimmed}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _is_refusal(text: str) -> bool:
    """Detect LLM refusal / meta-commentary outputs we should discard.

    Checks only the opening ~200 characters so a passage that legitimately
    mentions a keyword deep in the prose isn't thrown away.  The regex
    covers the families of refusal openings we've actually observed in
    production: "I need to clarify...", "I appreciate the detailed
    request...", "the search results don't contain...", etc.
    """
    if not text:
        return False
    head = text[:250].lower()
    return bool(_REFUSAL_RE.search(head))


def _clean_passage(text: str) -> str:
    """Post-process the LLM output so it actually conforms to the rules.

    Returns empty string if the output is a refusal — the caller treats
    empty as "don't persist", so refusals are dropped silently rather
    than shown to the client as if they were real intent data.
    """
    if not text:
        return ""

    cleaned = text.strip()
    if (cleaned.startswith("\"") and cleaned.endswith("\"")) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # Em/en-dashes -> commas (rules explicitly forbid em dashes).
    cleaned = cleaned.replace("\u2014", ",").replace("\u2013", ",")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(#+\s*|[-*]\s+)", "", cleaned)

    if _is_refusal(cleaned):
        logger.warning(
            f"Intent details LLM returned a refusal / meta-commentary, "
            f"discarding.  First 200 chars: {cleaned[:200]!r}"
        )
        return ""

    return cleaned


def _get_api_key() -> Optional[str]:
    """Pick up the OpenRouter key already provisioned for fulfillment scoring."""
    return (
        os.getenv("FULFILLMENT_OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_KEY")
    )


_OVERALL_HEADER_RE = re.compile(r"^={2,}\s*OVERALL\s*={2,}\s*$", re.IGNORECASE | re.MULTILINE)
_SIGNAL_HEADER_RE = re.compile(r"^={2,}\s*SIGNAL\s+(\d+)\s*(?:[:\-]\s*(.+?))?\s*={2,}\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_combined_output(
    content: str,
    keepers: List[Tuple[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Parse the marker-delimited LLM response into (passage, per_signal).

    Robust to per-section malformation: an unparseable SIGNAL N block simply
    omits that index from per_signal, leaving the rest intact.  The overall
    passage is independently extractable.

    ``keepers`` is ``[(source_index, signal), ...]`` from ``_compute_keepers``.
    Each emitted per_signal entry carries:
      - ``index``: the 1-based number the LLM saw (position in keepers)
      - ``source_index``: 0-based position in the ORIGINAL
        intent_signal_mapping array — frontend zips by this to land on the
        correct raw evidence row when failed signals were filtered out
      - ``icp_signal``: matched ICP label (from the keeper)
      - ``details``: the LLM's client-ready paragraph for this signal
    """
    result: Dict[str, Any] = {"passage": "", "per_signal": []}
    if not content:
        return result

    # Find all section boundaries in order. We treat the content as a
    # series of [marker, body] pairs and slice between markers.
    headers: List[Tuple[int, int, str, Optional[int]]] = []
    for m in _OVERALL_HEADER_RE.finditer(content):
        headers.append((m.start(), m.end(), "overall", None))
    for m in _SIGNAL_HEADER_RE.finditer(content):
        try:
            idx = int(m.group(1))
        except (TypeError, ValueError):
            continue
        headers.append((m.start(), m.end(), "signal", idx))

    if not headers:
        # No markers at all — fall back to treating the whole content as the
        # overall passage. Per-signal is empty.
        result["passage"] = _clean_passage(content)
        return result

    headers.sort(key=lambda h: h[0])

    # Track signal indices we've already captured so a model that emits
    # duplicate "=== SIGNAL N ===" sections doesn't produce duplicate
    # per_signal entries.  First non-empty occurrence wins.
    seen_signal_indices: set = set()

    for i, (_start, body_start, kind, idx) in enumerate(headers):
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(content)
        body = content[body_start:body_end].strip()
        cleaned = _clean_passage(body)
        if not cleaned:
            continue
        if kind == "overall":
            # If the model emits multiple OVERALL blocks (rare), keep the
            # first non-empty one.
            if not result["passage"]:
                result["passage"] = cleaned
        elif kind == "signal" and idx is not None:
            # 1-based index from the prompt; clamp to keepers range and
            # skip duplicates (first non-empty section per index wins).
            if not (1 <= idx <= len(keepers)):
                continue
            if idx in seen_signal_indices:
                continue
            seen_signal_indices.add(idx)
            src_idx, keeper = keepers[idx - 1]
            icp_signal = (keeper.get("matched_icp_signal") or "").strip() or UNTAGGED_SIGNAL_LABEL
            # _idx is the 1-based LLM-emit position; carried only for the
            # canonical sort below, stripped before return so it doesn't
            # bloat the stored JSONB (array order is the public contract).
            result["per_signal"].append(
                {
                    "_idx": idx,
                    "source_index": src_idx,
                    "icp_signal": icp_signal,
                    "details": cleaned,
                }
            )

    # Canonical order — consumers rely on array position, not on any
    # numeric field, so sort here and drop the internal _idx key.
    result["per_signal"].sort(key=lambda e: e["_idx"])
    for entry in result["per_signal"]:
        entry.pop("_idx", None)
    return result


async def generate_intent_details(
    icp: Dict[str, Any],
    intent_signals_detail: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate both the overall Intent Details paragraph and a per-signal breakdown.

    One LLM round-trip produces both outputs from the same ICP + signals
    context.  Returns a dict with two keys:

        ``passage`` -> the overall client-ready paragraph (string; empty
            on any failure).  Persisted to
            ``fulfillment_score_consensus.intent_details``.
        ``per_signal`` -> list of ``{source_index, icp_signal, details}``
            entries in canonical order, one per verified (score > 0)
            signal.  Empty list on failure.  Persisted to
            ``fulfillment_score_consensus.intent_breakdown.per_signal``.

    Caller (lifecycle) treats any empty piece as non-fatal and proceeds
    with reward payout.  The two outputs are independent: a parse failure
    on a single SIGNAL section drops only that index, the overall paragraph
    and the other signals survive.
    """
    empty: Dict[str, Any] = {"passage": "", "per_signal": []}

    api_key = _get_api_key()
    if not api_key:
        logger.warning("generate_intent_details: no OpenRouter API key set")
        return empty

    # Same keepers filter used to format the signals block.  Each entry is
    # ``(source_index, signal)`` so the parser can emit a per_signal record
    # whose ``source_index`` field points at the correct raw row in the
    # original intent_signal_mapping array even when failed signals were
    # filtered out.
    keepers = _compute_keepers(intent_signals_detail)

    icp_block = _format_icp_block(icp or {})
    signals_block = _format_signals_block(intent_signals_detail or [])

    prompt = _PROMPT_TEMPLATE.format(
        icp_block=icp_block,
        signals_block=signals_block,
    )

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                    "X-Title": "LeadPoet Intent Details",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write client-ready buying-intent summaries "
                                "strictly from the evidence provided by the user. "
                                "Absolute rules: respond with ONLY the requested "
                                "marker-delimited sections. Never apologize, never "
                                "say 'I need to clarify', 'I appreciate', "
                                "'unfortunately', 'based on search results', or any "
                                "similar meta-commentary. Never mention your own "
                                "limitations or the source of the information. "
                                "Never invent facts beyond the provided intent "
                                "signals. If the provided signals are thin, write "
                                "shorter sections using only what is there. "
                                "No preamble, no extra labels, no markdown, no em "
                                "dashes, no links."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 3000,
                },
            )
    except Exception as e:
        logger.warning(f"Intent details LLM call failed: {e}")
        return empty

    if resp.status_code != 200:
        logger.warning(
            f"Intent details LLM returned {resp.status_code}: {resp.text[:300]}"
        )
        return empty

    try:
        data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )
    except Exception as e:
        logger.warning(f"Intent details LLM response parse failed: {e}")
        return empty

    return _parse_combined_output(content, keepers)


async def generate_intent_details_passage(
    icp: Dict[str, Any],
    intent_signals_detail: List[Dict[str, Any]],
) -> str:
    """Backward-compatible wrapper returning only the overall passage.

    New callers should prefer ``generate_intent_details`` and persist both
    the passage and the per-signal breakdown.  This shim exists so any
    code still calling the old name keeps working with no behavior change.
    """
    out = await generate_intent_details(icp, intent_signals_detail)
    return out.get("passage", "")
