"""qualification_model — universal evidence-first qualification miner.

Validator entry point:
    qualify(icp: dict) -> list[CompanyOutput dict]

Returns up to MAX_LEADS_PER_ICP (5) CompanyOutput-shaped dicts per ICP,
sorted by internal `score_company` score descending (best first).

The implementation lives in `_model.py` (an async pipeline). This thin
adapter wraps it in the synchronous `qualify(icp)` signature the
validator harness expects, and maps each verified lead to a clean
CompanyOutput dict (stripping internal metadata).

Pipeline (identical for every ICP — no per-ICP hardcoding):
    1. DISCOVERY   — Exa news (date-bound, primary-source biased) + Sonar
                     exclusion-driven 3-pass
    2. EXTRACT     — Strict LLM intent-match filter on every candidate
    3. ANCHOR LOOKUP — Sonar resolves website / LinkedIn / description
    4. URL SEARCH  — Exa keyword bias (company-domain + press wires)
    5. VERIFY      — Production `verify_three_stage` (Sonar → SD → Sonar-pro)
    6. SCORE       — Production `score_company` for internal ranking
    7. SUBMIT      — Dedupe by company, keep top-5

Environment (auto-loaded from {repo_root}/.env if present):
    OPENROUTER_API_KEY     Sonar / verifier LLM calls
    EXA_API_KEY            Exa news + keyword search
    SCRAPINGDOG_API_KEY    Verifier Stage 2 scraping
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx

from . import _model

MAX_LEADS_PER_ICP = 5


def qualify(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validator entry point.

    Discovers, verifies, and ranks companies for the given ICP, then
    returns up to MAX_LEADS_PER_ICP (5) CompanyOutput-shaped dicts.

    Args:
        icp: ICPPrompt dict (industry, sub_industry, intent_signals,
             company_stage, geography/country, prompt, icp_id, ...).

    Returns:
        List of 0 to MAX_LEADS_PER_ICP CompanyOutput dicts, sorted by
        internal score desc. Empty list = honest abstention (no
        candidates verified for this ICP).

    Defensive: returns ``[]`` rather than raising on malformed ICP input
    (missing required fields). The validator must not crash on us.
    """
    # Validate input shape — return [] (abstain) rather than raise.
    if not isinstance(icp, dict):
        return []
    intent_signals = icp.get("intent_signals") or []
    if not isinstance(intent_signals, list) or not intent_signals:
        return []
    if not (icp.get("industry") or icp.get("icp_id")):
        return []

    try:
        return asyncio.run(_qualify_async(icp))
    except Exception:
        # Anything that escaped the per-call error handling — abstain.
        return []


async def _qualify_async(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        scored_leads = await _model.qualify_icp(client, icp)
    # `scored_leads` is sorted by score desc; take top-5 and return only
    # the CompanyOutput dict for each. Internal score / icp_fit metadata
    # is dropped — the validator computes its own scoring.
    top = scored_leads[:MAX_LEADS_PER_ICP]
    return [lead["submission"] for lead in top if "submission" in lead]
