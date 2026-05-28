"""L1 — Parse ICPPrompt into a structured ParsedICP via Claude Sonnet 4.6."""

from __future__ import annotations

import logging
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import ICPPrompt, ParsedICP
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer

logger = logging.getLogger(__name__)


PROMPT = """You structure B2B sales ICPs into a search plan.

Buyer prompt: "{prompt}"
Industry: {industry} / {sub_industry}
Geography: {geography}
Size band: {employee_count}
Intent signals requested: {intent_signals}

Output JSON only (no code fences):
{{
  "intent_class": "<one of: hiring|funding|product_launch|tech_adoption|expansion|partnership|leadership_change|compliance_event|other — pick the dominant one>",
  "intent_classes": [
    "<intent class for intent_signals[0]>",
    "<intent class for intent_signals[1]>",
    "..."
  ],
  "is_time_bound": true|false,
  "time_window_days": <integer or null>,
  "hard_filters": {{
    "country": "<ISO-2 or null>",
    "industry_keywords": ["...", "..."]
  }},
  "semantic_queries": ["...", "...", "..."],
  "sonar_query_angles": [
    "List <criteria>...",
    "Which companies have recently <event>...",
    "Find <peer of>..."
  ],
  "keyword_queries": [
    "<a SHORT, OR-heavy Boolean query designed to find company HOMEPAGES — use few ANDs (1-2 max), wrap synonyms in (a OR b), avoid stacking every constraint as AND — e.g. (\"mobile payments\" OR \"payments API\") (Series A OR Series B OR Series C) site:linkedin.com/company>",
    "<a SECOND query designed to find FUNDING/NEWS articles — can be more conjunctive, may include site:crunchbase.com or site:techcrunch.com, may include amount keywords like \"raised\" OR \"announces\">"
  ],
  "linkedin_jobs_query": "<role + skills, or null if no hiring signal>",
  "news_keywords": ["...", "..."]
}}

For intent_classes: classify EACH intent_signal independently into one of the same nine classes. The list must be the same length as the input intent_signals."""


_STRICT_PROMPT_SUFFIX = "\n\nReturn ONLY valid JSON. No prose, no markdown, no code fences. Pick the most specific intent_class — avoid 'other' unless truly nothing else fits."


async def _attempt_parse(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    tracer: Tracer,
    prompt: str,
    attempt: int,
    model: Optional[str] = None,
) -> Optional[dict]:
    r = await openrouter.json_call(
        model or CONFIG.PARSER_MODEL, prompt, label=f"L1_parse_icp_attempt{attempt}"
    )
    cost.add("openrouter", r["cost_usd"], layer="L1")
    parsed = r["parsed"]
    if not isinstance(parsed, dict):
        tracer.emit("L1", "parse_failed", attempt=attempt, model=(model or CONFIG.PARSER_MODEL), raw=r["raw"][:200])
        return None
    return parsed


async def parse_icp(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    tracer: Tracer,
    icp: ICPPrompt,
) -> Optional[ParsedICP]:
    intent_signals_str = ", ".join(icp.intent_signals) if icp.intent_signals else "<none>"
    base_prompt = PROMPT.format(
        prompt=icp.prompt or "<none>",
        industry=icp.industry,
        sub_industry=icp.sub_industry or "<unspecified>",
        geography=icp.geography or "<any>",
        employee_count=icp.employee_count or "<any>",
        intent_signals=intent_signals_str,
    )

    # Attempt 1: standard prompt with the configured parser model
    parsed = await _attempt_parse(openrouter, cost, tracer, base_prompt, attempt=1)

    # Attempt 2: retry with stricter prompt (also bypasses cache via different content)
    if parsed is None:
        parsed = await _attempt_parse(
            openrouter, cost, tracer, base_prompt + _STRICT_PROMPT_SUFFIX, attempt=2,
        )

    # Attempt 3: fallback to a different model for JSON robustness — when the
    # configured parser keeps returning non-JSON, gpt-4o-mini is highly reliable
    # at JSON output for structured tasks.
    if parsed is None:
        parsed = await _attempt_parse(
            openrouter, cost, tracer, base_prompt + _STRICT_PROMPT_SUFFIX, attempt=3,
            model=CONFIG.TRIAGE_MODEL,
        )

    if parsed is None:
        return None

    # If intent_class came back as 'other', retry once asking for a specific class —
    # 'other' is the laziest LLM output and downstream layers route worse on it.
    if parsed.get("intent_class") == "other":
        retry = await _attempt_parse(
            openrouter, cost, tracer,
            base_prompt + _STRICT_PROMPT_SUFFIX, attempt=3,
        )
        if isinstance(retry, dict) and retry.get("intent_class") != "other":
            parsed = retry

    # Build per-signal intent_classes; pad/truncate to align with icp.intent_signals
    raw_classes = parsed.get("intent_classes") or []
    valid_classes = {"hiring", "funding", "product_launch", "tech_adoption",
                     "expansion", "partnership", "leadership_change",
                     "compliance_event", "other"}
    aligned: list = []
    for i in range(len(icp.intent_signals)):
        cand = raw_classes[i] if i < len(raw_classes) else None
        if isinstance(cand, str) and cand in valid_classes:
            aligned.append(cand)
        else:
            aligned.append(parsed.get("intent_class") or "other")

    try:
        out = ParsedICP(
            intent_class=parsed.get("intent_class", "other"),
            intent_classes=aligned,
            is_time_bound=bool(parsed.get("is_time_bound")),
            time_window_days=parsed.get("time_window_days"),
            hard_filters=parsed.get("hard_filters") or {},
            semantic_queries=parsed.get("semantic_queries") or [],
            sonar_query_angles=parsed.get("sonar_query_angles") or [],
            keyword_queries=parsed.get("keyword_queries") or [],
            linkedin_jobs_query=parsed.get("linkedin_jobs_query"),
            news_keywords=parsed.get("news_keywords") or [],
        )
    except Exception as e:
        tracer.emit("L1", "validation_failed", error=str(e))
        return None
    tracer.emit(
        "L1", "parsed",
        intent_class=out.intent_class,
        time_bound=out.is_time_bound,
        n_semantic=len(out.semantic_queries),
        n_sonar=len(out.sonar_query_angles),
    )
    return out
