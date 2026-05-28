"""L2 — Wide Discovery across Sonar + Exa + ScrapingDog."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from qual_engine.config import CONFIG
from qual_engine.models import CandidateCompany, ICPPrompt, ParsedICP
from qual_engine.providers.openrouter import OpenRouterClient
from qual_engine.providers.exa import ExaClient
from qual_engine.providers.scrapingdog import ScrapingDogClient
from qual_engine.infra.cost_tracker import CostTracker
from qual_engine.infra.trace import Tracer
from qual_engine.validators.text_match import normalize_domain
from qual_engine.utils.ai_classifiers import is_aggregator, extract_company_from_article

logger = logging.getLogger(__name__)


SONAR_LIST_PROMPT = """List up to {n} real operating companies matching: {criteria}

HARD RULES (do not relax — return fewer results rather than violate any):
- Geography is HARD: if the criteria specifies a region/country/state/city, EVERY returned company's headquarters MUST be in that region. If you cannot find {n} that satisfy this, return ONLY the ones that do — even if that means 0, 1, or 2 results.
- Real OPERATING companies ONLY — NOT recruiters, agencies, talent firms, job marketplaces, ETFs/funds, holding companies, news outlets, market-research firms, directories, or generic "client" placeholders.
- The company must be the actual business described — not someone hiring or reporting on its behalf.
- Each entry must include the company's OWN primary domain (the canonical root domain the company itself operates), NEVER a job-board, news, or press-release URL.
- `source_url` must point to a page on the company's OWN domain OR a credible news/funding announcement that names this company directly — NOT a job board.
- Do not pad. Quality over quantity. Returning 2 correct rows is better than 10 with location/intent mismatch.

For each company, also return structured attribute hints (use null if you don't know):
- `hq`: headquarters as "City, State, Country" exactly as widely reported (null if uncertain).
- `funding_stage`: most recent stage (e.g. "Series B", "Seed", "Series C", "Public", null if unknown).
- `employee_size`: approximate band (e.g. "50-200", "200-500", "10-50", null if unknown).

Output JSON array ONLY (no prose, no code fences):
[{{"name": "...", "domain": "...", "why_it_matches": "<≤12 words>", "source_url": "...", "hq": "...", "funding_stage": "...", "employee_size": "..."}}, ...]"""


async def _sonar_list(
    openrouter: OpenRouterClient,
    cost: CostTracker,
    criteria: str,
    n: int = 12,
) -> list[CandidateCompany]:
    prompt = SONAR_LIST_PROMPT.format(criteria=criteria, n=n)
    r = await openrouter.json_call(
        CONFIG.SONAR_MODEL, prompt, label="L2_sonar_list"
    )
    cost.add("openrouter", r["cost_usd"], layer="L2")
    parsed = r["parsed"]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        domain = normalize_domain(item.get("domain") or item.get("source_url") or "")
        if not domain:
            continue

        def _clean(v):
            if not isinstance(v, str): return None
            v = v.strip()
            return v if v and v.lower() not in ("null", "none", "unknown", "n/a") else None

        out.append(
            CandidateCompany(
                name=item.get("name", "")[:120],
                domain=domain,
                source="sonar",
                discovery_url=item.get("source_url", ""),
                discovery_snippet=item.get("why_it_matches", "")[:200],
                hint_hq=_clean(item.get("hq")),
                hint_stage=_clean(item.get("funding_stage")),
                hint_size=_clean(item.get("employee_size")),
            )
        )
    return out


def _build_icp_criteria(icp: ICPPrompt, angle_hint: str = "") -> str:
    parts = []
    if icp.prompt:
        parts.append(f'Buyer\'s exact ask: "{icp.prompt}"')
    if icp.product_service:
        parts.append(f"Buyer sells: {icp.product_service}")
    parts.append(f"Industry: {icp.industry}{f' / {icp.sub_industry}' if icp.sub_industry else ''}")
    if icp.geography:
        parts.append(f"Geography: {icp.geography}")
    if icp.employee_count:
        parts.append(f"Employee count: {icp.employee_count}")
    if icp.company_stage:
        parts.append(f"Stage: {icp.company_stage}")
    if icp.intent_signals:
        parts.append(f"Currently showing intent: {'; '.join(icp.intent_signals)}")
    if angle_hint:
        parts.append(f"Angle: {angle_hint}")
    return ". ".join(p for p in parts if p)


def _build_sonar_criteria(icp: ICPPrompt, angle_hint: str = "") -> str:
    """Sonar list-discovery criteria.

    Geography is marked HARD inline; the prompt-level HARD RULES enforce it.
    Other fields are passed as targeted hints (Sonar will still try to honor them
    but won't drop to 0 results — the HARD RULES say "return fewer rather than relax").
    """
    parts = []
    region = icp.geography or icp.country
    if region:
        parts.append(f"REQUIRED HQ region (do not relax): {region}")
    if icp.prompt:
        parts.append(f'Buyer\'s ask: "{icp.prompt}"')
    if icp.product_service:
        parts.append(f"Buyer sells: {icp.product_service}")
    if icp.industry:
        sub = f" / {icp.sub_industry}" if icp.sub_industry else ""
        parts.append(f"Target industry: {icp.industry}{sub}")
    if icp.employee_count:
        parts.append(f"Target employee band (preferred): {icp.employee_count}")
    if icp.company_stage:
        parts.append(f"Target stage (preferred): {icp.company_stage}")
    if icp.intent_signals:
        parts.append(f"Showing intent: {'; '.join(icp.intent_signals)}")
    if angle_hint:
        parts.append(f"Angle: {angle_hint}")
    return ". ".join(p for p in parts if p)


# Aggregator classification is LLM-driven (see qual_engine.utils.ai_classifiers.is_aggregator)
# — no hardcoded domain or path lists.


def _extract_from_exa_result(result: dict, source: str) -> Optional[dict]:
    """Build a raw candidate dict from an Exa result. Aggregator filtering is
    done downstream via LLM (qual_engine.utils.ai_classifiers.is_aggregator)."""
    url = result.get("url") or ""
    if not url:
        return None
    domain = normalize_domain(url)
    if not domain:
        return None
    # LinkedIn company URLs → keep, extract slug as name
    if "linkedin.com/company/" in url:
        m = re.search(r"linkedin\.com/company/([^/?#]+)", url)
        name = m.group(1).replace("-", " ").title() if m else ""
        return {
            "name": name,
            "domain": domain,
            "source": source,
            "discovery_url": url,
            "discovery_snippet": result.get("title", "")[:200],
        }
    title = result.get("title", "")[:200]
    name = (
        title.split(" - ")[0].split(" | ")[0][:80]
        if " - " in title or " | " in title
        else domain.split(".")[0].title()
    )
    return {
        "name": name,
        "domain": domain,
        "source": source,
        "discovery_url": url,
        "discovery_snippet": title,
    }


async def wide_discovery(
    openrouter: OpenRouterClient,
    exa: ExaClient,
    scrapingdog: ScrapingDogClient,
    cost: CostTracker,
    tracer: Tracer,
    icp: ICPPrompt,
    parsed: ParsedICP,
) -> list[CandidateCompany]:
    """Run parallel discovery across Sonar + Exa + ScrapingDog Google."""

    tasks = []
    task_labels = []

    # Sonar lists — 3 angles. n reduced from 20 → 8: with HARD geography rule,
    # asking for 20 forces relaxation. Keep counts realistic so Sonar returns
    # only true matches.
    angles = parsed.sonar_query_angles[:3] if parsed.sonar_query_angles else [""]
    for i, angle in enumerate(angles):
        criteria = _build_sonar_criteria(icp, angle_hint=angle)
        tasks.append(_sonar_list(openrouter, cost, criteria, n=8))
        task_labels.append(f"sonar:{i}")

    # Exa neural — 3 semantic queries (or generate from ICP)
    semantic_queries = parsed.semantic_queries[:3] if parsed.semantic_queries else [
        f"{icp.sub_industry or icp.industry} companies {icp.country or icp.geography}"
    ]
    for q in semantic_queries:
        tasks.append(exa.search_neural(q, num_results=CONFIG.DISCOVERY_PER_QUERY_RESULTS, days_back=CONFIG.EXA_DEFAULT_DAYS_BACK))
        task_labels.append("exa_neural")

    # Exa keyword — use L1's keyword_queries (they encode all constraints).
    # Strip `site:` operators (Exa keyword doesn't honor them; those go to Google).
    exa_kw_queries = []
    for q in (parsed.keyword_queries or []):
        if "site:" in q:
            continue  # route to SD Google only
        exa_kw_queries.append(q)
    if not exa_kw_queries:
        exa_kw_queries = [f"{icp.sub_industry or icp.industry} companies {icp.country or icp.geography}"]
    for q in exa_kw_queries[:2]:
        tasks.append(exa.search_keyword(q, num_results=CONFIG.DISCOVERY_PER_QUERY_RESULTS))
        task_labels.append("exa_keyword")

    # ScrapingDog Google — fire each L1 keyword_query directly (Google honors
    # all operators including `site:`). Keeps L1's precision intact.
    sd_queries = list(parsed.keyword_queries or [])
    if not sd_queries:
        sd_queries = [
            f'"{icp.sub_industry or icp.industry}" site:linkedin.com/company {icp.country or icp.geography or ""}'.strip()
        ]
    for q in sd_queries[:2]:
        tasks.append(scrapingdog.google(q, results=10))
        task_labels.append("sd_google")

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    raw_candidates: list[CandidateCompany] = []

    for label, results in zip(task_labels, raw):
        if isinstance(results, Exception):
            tracer.emit("L2", "task_error", source=label, error=str(results)[:200])
            continue
        if label.startswith("sonar:"):
            for c in results:  # already CandidateCompany
                raw_candidates.append(c)
        elif label.startswith("exa_"):
            cost.add("exa", results.get("cost_usd", 0), layer="L2")
            src_map = {"exa_neural": "exa_neural", "exa_keyword": "exa_keyword"}
            for r in results.get("results", []):
                d = _extract_from_exa_result(r, src_map.get(label, "exa_neural"))
                if d:
                    raw_candidates.append(CandidateCompany(**d))
        elif label == "sd_google":
            cost.add("scrapingdog", results.get("cost_usd", 0), layer="L2")
            for r in results.get("results", []):
                fake = {"url": r.get("link") or r.get("url"), "title": r.get("title", "")}
                d = _extract_from_exa_result(fake, "sd_google")
                if d:
                    raw_candidates.append(CandidateCompany(**d))

    # Pre-dedupe source breakdown (diagnostic)
    raw_by_source = {}
    for c in raw_candidates:
        raw_by_source[c.source] = raw_by_source.get(c.source, 0) + 1

    # Dedupe by normalized domain — keep first occurrence
    seen = set()
    pre_filter: list[CandidateCompany] = []
    for c in raw_candidates:
        if not c.domain or c.domain in seen:
            continue
        seen.add(c.domain)
        pre_filter.append(c)

    # Post-dedupe source breakdown (diagnostic)
    dedupe_by_source = {}
    for c in pre_filter:
        dedupe_by_source[c.source] = dedupe_by_source.get(c.source, 0) + 1

    # Hint-based geography pre-filter (cheap, no LLM):
    # If the candidate carries a hint_hq (Sonar fills this) AND we have a target
    # region, drop the candidate when the hint clearly doesn't contain the region.
    # Substring match against either the geography string or any of its tokens
    # ≥4 chars — keeps the test conservative so we never wrongly reject.
    target_region = (icp.geography or icp.country or "").strip()
    region_tokens = [t.strip(",.;:").lower() for t in target_region.split()
                     if len(t.strip(",.;:")) >= 4]
    dropped_geo = 0
    if region_tokens:
        keep_after_geo: list[CandidateCompany] = []
        for c in pre_filter:
            if c.hint_hq:
                hq_lower = c.hint_hq.lower()
                if not any(tok in hq_lower for tok in region_tokens):
                    dropped_geo += 1
                    continue
            keep_after_geo.append(c)
        pre_filter = keep_after_geo

    # LLM-based aggregator filtering (parallel, cached per-domain)
    is_agg_results = await asyncio.gather(
        *[
            is_aggregator(c.discovery_url or f"https://{c.domain}", c.discovery_snippet, openrouter, cost)
            for c in pre_filter
        ],
        return_exceptions=True,
    )
    filtered = []
    agg_drops_diag = []                   # diagnostic
    extraction_targets: list[CandidateCompany] = []  # aggregator URLs to extract from
    for c, is_agg in zip(pre_filter, is_agg_results):
        if isinstance(is_agg, Exception):
            filtered.append(c)  # fail-open
        elif not is_agg:
            filtered.append(c)
        else:
            agg_drops_diag.append(f"{c.domain[:30]} ({c.source}) — {(c.name or '')[:35]}")
            extraction_targets.append(c)
    tracer.emit("L2", "agg_drops_sample", drops=agg_drops_diag[:30])

    # Article → company extraction (recovers signal from dropped article URLs)
    extracted_added = 0
    if extraction_targets:
        ext_results = await asyncio.gather(
            *[
                extract_company_from_article(
                    t.discovery_url or f"https://{t.domain}",
                    t.discovery_snippet or t.name,
                    openrouter, cost,
                )
                for t in extraction_targets
            ],
            return_exceptions=True,
        )
        already_have = {c.domain for c in filtered if c.domain}
        for src, ext in zip(extraction_targets, ext_results):
            if isinstance(ext, Exception) or not ext:
                continue
            name = ext.get("name")
            extracted_domain = ext.get("domain")
            domain_norm = normalize_domain(extracted_domain) if extracted_domain else ""
            if not name or not domain_norm:
                continue
            if domain_norm in already_have:
                continue
            already_have.add(domain_norm)
            filtered.append(CandidateCompany(
                name=name,
                domain=domain_norm,
                source="extracted_from_article",
                discovery_url=src.discovery_url or "",
                discovery_snippet=(src.discovery_snippet or "")[:200],
            ))
            extracted_added += 1
    tracer.emit("L2", "article_extraction",
                attempted=len(extraction_targets),
                recovered=extracted_added)

    final_by_source = {}
    for c in filtered:
        final_by_source[c.source] = final_by_source.get(c.source, 0) + 1
    tracer.emit("L2", "discovered",
                raw_count=len(raw_candidates),
                raw_by_source=raw_by_source,
                dedupe_by_source=dedupe_by_source,
                pre_filter_count=len(pre_filter),
                dropped_geo=dropped_geo,
                final_count=len(filtered),
                final_by_source=final_by_source)
    return filtered
