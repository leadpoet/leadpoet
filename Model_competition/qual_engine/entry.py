"""Public API: qualify(icp) → QualificationResult.

Wires L1 → L2 → L3 → L2.5 → L4 → L5 → L6 → L7 → L8.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from qual_engine.config import CONFIG
from qual_engine.models import (
    EvidenceURL,
    ICPPrompt,
    QualificationResult,
    ResolvedCompany,
    VerifiedSignal,
)
from qual_engine.infra.cache import Cache
from qual_engine.infra.cost_tracker import CostTracker, CostCeilingExceeded
from qual_engine.infra.trace import Tracer
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.providers.exa import ExaClient
from qual_engine.providers.scrapingdog import ScrapingDogClient
from qual_engine.layers.l1_icp_parser import parse_icp
from qual_engine.layers.l2_discovery import wide_discovery
from qual_engine.layers.l2_5_triage import triage
from qual_engine.layers.l3_resolver import resolve
from qual_engine.layers.l4_filter import filter_candidates
from qual_engine.layers.l5_evidence import gather_evidence
from qual_engine.layers.l6_grounding import ground
from qual_engine.layers.l7_ranker import rank
from qual_engine.layers.l8_output import build_output

logger = logging.getLogger(__name__)


async def _ground_one_company(
    openrouter: OpenRouterClient,
    exa: ExaClient,
    scrapingdog: ScrapingDogClient,
    cost: CostTracker,
    tracer: Tracer,
    icp: ICPPrompt,
    parsed,
    company: ResolvedCompany,
) -> list[VerifiedSignal]:
    """For a single company, ground signals against evidence.
    Early-stop policy: once we have a single strong-and-corroborated signal,
    stop trying additional ICP signals for this candidate (saves cost).
    """
    verified: list[VerifiedSignal] = []
    min_conf = CONFIG.GROUNDING_MIN_CONFIDENCE

    def have_strong_corroborated() -> bool:
        return any(
            v.grounding_confidence >= min_conf and v.sonar_corroborated
            for v in verified
        )

    for signal_idx, sig_text in enumerate(icp.intent_signals):
        # Early-stop across signals
        if have_strong_corroborated():
            break

        evidences = await gather_evidence(
            openrouter, exa, scrapingdog, cost, tracer,
            company, sig_text, parsed,
            icp=icp,
            signal_idx=signal_idx,
        )
        attempted = 0
        accepted_for_signal = 0
        for ev in evidences:
            # Per-signal cap: try at most 4 URLs; accept at most 1 (we only need one).
            if attempted >= 4 or accepted_for_signal >= 1:
                break
            attempted += 1
            try:
                v = await ground(
                    openrouter, exa, cost, tracer,
                    company=company,
                    evidence=ev,
                    icp_signal_text=sig_text,
                    icp_signal_idx=signal_idx,
                    is_time_bound=parsed.is_time_bound,
                    time_window_days=parsed.time_window_days,
                    target_stage=icp.company_stage or "",
                )
            except Exception as e:
                logger.warning("ground failed for %s / %s: %s", company.canonical_name, ev.url, e)
                continue
            if v:
                verified.append(v)
                accepted_for_signal += 1
                # If this signal IS strong + corroborated, break out of all loops
                if v.grounding_confidence >= min_conf and v.sonar_corroborated:
                    break
    return verified


async def qualify(
    icp: ICPPrompt,
    *,
    max_to_ground: Optional[int] = None,
) -> QualificationResult:
    """Run the full 9-layer pipeline.

    Always returns a QualificationResult. If anything goes wrong or no candidate
    qualifies, returns one with company=None and abstention_reason set.
    """
    cache = Cache(CONFIG.CACHE_DB_PATH)
    cost = CostTracker(
        hard_ceiling_usd=CONFIG.PER_ICP_HARD_CEILING_USD,
        soft_ceiling_usd=CONFIG.PER_ICP_SOFT_CEILING_USD,
    )
    tracer = Tracer(icp_id=icp.icp_id)
    max_to_ground = max_to_ground or CONFIG.MAX_CANDIDATES_TO_GROUND

    try:
        async with httpx.AsyncClient() as client:
            opr = OpenRouterClient(client, cache)
            exa = ExaClient(client, cache)
            sd = ScrapingDogClient(client, cache)

            # ----- L1: Parse ICP -----
            parsed = await parse_icp(opr, cost, tracer, icp)
            if not parsed:
                return build_output(icp, [], "L1 parse failed", tracer, cost)

            # ----- L2: Wide Discovery -----
            candidates = await wide_discovery(opr, exa, sd, cost, tracer, icp, parsed)
            if not candidates:
                return build_output(icp, [], "L2 no candidates discovered", tracer, cost)

            # ----- L3: LinkedIn-anchored Resolver (with website-anchor fallback) -----
            resolved = await resolve(
                sd, exa, opr, cost, tracer, candidates,
                max_concurrent=10,
                icp_prompt=icp.prompt or "",
                icp_industry=icp.industry or "",
                icp_country=icp.country or "",
                icp_geography=icp.geography or "",
                icp_company_stage=icp.company_stage or "",
            )
            if not resolved:
                return build_output(icp, [], "L3 no LinkedIn-anchored candidates", tracer, cost)

            # ----- L2.5: Cheap Triage -----
            # Skip triage entirely when L3 already returned a small pool —
            # triage exists to save downstream cost on huge candidate sets,
            # not to filter modest ones. Otherwise it drops valid candidates
            # whose LinkedIn industry tag doesn't literally match the ICP.
            if len(resolved) <= 30:
                triaged = resolved
                tracer.emit("L2_5", "skipped",
                            reason=f"pool size {len(resolved)} ≤ 30",
                            kept=len(resolved))
            else:
                triaged = await triage(opr, cost, tracer, resolved, icp, top_n=25)
                if not triaged:
                    return build_output(icp, [], "L2.5 triage rejected all", tracer, cost)

            # ----- L4: Hard Filters (LLM-assisted industry compat) -----
            l4_survivors = await filter_candidates(triaged, icp, tracer, openrouter=opr, cost=cost)
            if not l4_survivors:
                return build_output(icp, [], "L4 hard filters dropped all candidates", tracer, cost)

            # ----- L4.5: ICP-fit hard gate (LLM, no hardcoded knowledge) -----
            from qual_engine.utils.ai_classifiers import is_icp_match
            gate_results = await asyncio.gather(*[
                is_icp_match(
                    company_name=c.canonical_name,
                    primary_domain=c.primary_domain or "",
                    industry_tags=c.industry_tags or [],
                    country=c.country or "",
                    hq=c.headquarters or "",
                    employee_count_band=c.employee_count_band or "",
                    funding_stage=c.funding_stage or "",
                    description=c.description or "",
                    buyer_prompt=icp.prompt or "",
                    product_service=icp.product_service or "",
                    target_industry=icp.industry or "",
                    target_sub_industry=icp.sub_industry or "",
                    target_geography=icp.geography or "",
                    target_employee_count=icp.employee_count or "",
                    target_stage=icp.company_stage or "",
                    openrouter=opr,
                    cost=cost,
                )
                for c in l4_survivors
            ], return_exceptions=True)
            survivors: list = []
            dropped_at_fit = []
            kept_diag: list = []
            for c, ok in zip(l4_survivors, gate_results):
                # diagnostic — log what each candidate looked like to the gate
                kept_diag.append({
                    "name": c.canonical_name,
                    "domain": c.primary_domain,
                    "industry": (c.industry_tags or [None])[0],
                    "anchor": getattr(c, "anchor_source", "?"),
                    "verdict": "keep" if (isinstance(ok, Exception) or ok is True) else "drop",
                })
                if isinstance(ok, Exception) or ok is True:
                    survivors.append(c)
                else:
                    dropped_at_fit.append(c.canonical_name)
            tracer.emit("L4_5", "icp_fit_gate",
                        input_count=len(l4_survivors),
                        survived=len(survivors),
                        dropped=dropped_at_fit[:10],
                        per_candidate=kept_diag[:30])
            if not survivors:
                return build_output(
                    icp, [],
                    f"L4.5 ICP-fit gate rejected all {len(l4_survivors)} candidates ({dropped_at_fit[:5]}…)",
                    tracer, cost,
                )
            # Cap to max_to_ground
            survivors = survivors[:max_to_ground]

            # ----- L5 + L6 per candidate (parallel) -----
            ground_tasks = [
                _ground_one_company(opr, exa, sd, cost, tracer, icp, parsed, c)
                for c in survivors
            ]
            verified_per_company = await asyncio.gather(*ground_tasks, return_exceptions=True)
            candidates_with_signals: list[tuple[ResolvedCompany, list[VerifiedSignal]]] = []
            for c, sigs in zip(survivors, verified_per_company):
                if isinstance(sigs, Exception):
                    logger.warning("grounding pipeline failed for %s: %s", c.canonical_name, sigs)
                    continue
                if sigs:
                    candidates_with_signals.append((c, sigs))

            if not candidates_with_signals:
                return build_output(icp, [], "L6 produced no verified signals across all candidates", tracer, cost)

            # ----- L7: Final Ranker (multi-answer mode) -----
            ranked, abstain_reason = await rank(
                opr, cost, tracer, icp, candidates_with_signals,
            )

            # ----- L8: Output Builder -----
            return build_output(icp, ranked, abstain_reason, tracer, cost)

    except CostCeilingExceeded as e:
        return build_output(icp, [], f"cost ceiling: {e}", tracer, cost)
    except Exception as e:
        logger.exception("qualify pipeline crashed")
        return build_output(
            icp, [], f"pipeline error: {type(e).__name__}: {str(e)[:200]}",
            tracer, cost,
        )


__all__ = ["qualify"]
