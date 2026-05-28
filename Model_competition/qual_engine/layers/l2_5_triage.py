"""L2.5 — Cheap triage to cut candidates to top N via gpt-4o-mini."""

from __future__ import annotations

import logging
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import ICPPrompt, ResolvedCompany
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer

logger = logging.getLogger(__name__)


PROMPT = """Pre-screen candidates for an ICP. Be FAST and use ONLY the info given (do not guess facts).

ICP:
  Buyer's actual ask:    "{buyer_prompt}"
  Buyer sells:           "{product_service}"
  Industry:              {industry} / {sub_industry}
  Geography:             {geography}
  Employee count:        {employee_count}
  Stage:                 {stage}
  Intent signals wanted: {intent}

CANDIDATES (id, name, industry tags, employee band, country, hq):
{candidates_block}

For each candidate, decide: is this candidate plausibly worth deep-verifying against the ICP?

Lean toward keep=true:
- The candidate could be the kind of company the buyer described, even if its industry tag
  doesn't literally match the structured industry field. Trust the buyer's free-text
  description for the type of company; downstream layers do strict verification.

Use keep=false only when you can clearly identify one of these:
- The candidate is NOT an operating company the buyer would sell to (e.g., an
  intermediary, listing, aggregator, or fund — something that represents OTHER companies
  rather than being a company itself).
- The candidate's domain of business is unrelated to what the buyer described.

Output JSON array ONLY (no prose, no code fences):
[{{"id": 0, "keep": true, "reason": "<≤10 words>"}}, ...]"""


def _format_candidates(candidates: list[ResolvedCompany]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"  {i}: {c.canonical_name[:40]} | "
            f"industry={(c.industry_tags or ['?'])[0][:35]} | "
            f"size={c.employee_count_band or '?':25.25s} | "
            f"country={c.country or '?':12.12s} | "
            f"hq={(c.headquarters or '')[:40]}"
        )
    return "\n".join(lines)


async def triage(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    tracer: Tracer,
    candidates: list[ResolvedCompany],
    icp: ICPPrompt,
    *,
    top_n: int = 25,
    batch_size: int = 20,
) -> list[ResolvedCompany]:
    """Binary triage: LLM says keep=true|false per candidate. No numeric threshold."""
    if not candidates:
        return []

    kept_list: list[ResolvedCompany] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        prompt = PROMPT.format(
            buyer_prompt=icp.prompt or "<unspecified>",
            product_service=icp.product_service or "<unspecified>",
            industry=icp.industry,
            sub_industry=icp.sub_industry or "<any>",
            geography=icp.geography or "<any>",
            employee_count=icp.employee_count or "<any>",
            stage=icp.company_stage or "<any>",
            intent="; ".join(icp.intent_signals) or "<unspecified>",
            candidates_block=_format_candidates(batch),
        )
        r = await openrouter.json_call(CONFIG.TRIAGE_MODEL, prompt, label="L2_5_triage")
        cost.add("openrouter", r["cost_usd"], layer="L2_5")
        parsed = r["parsed"]
        if not isinstance(parsed, list):
            tracer.emit("L2_5", "batch_parse_failed", raw=r["raw"][:200])
            # Fail-open: if LLM can't be parsed, keep the whole batch
            kept_list.extend(batch)
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            i = item.get("id")
            if isinstance(i, int) and 0 <= i < len(batch) and item.get("keep"):
                kept_list.append(batch[i])

    # Cap to top_n (preserving input order)
    if len(kept_list) > top_n:
        kept_list = kept_list[:top_n]

    tracer.emit(
        "L2_5", "triaged",
        input_count=len(candidates),
        kept=len(kept_list),
        dropped=len(candidates) - len(kept_list),
    )
    return kept_list
