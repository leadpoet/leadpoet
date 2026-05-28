"""Generalized evidence-discovery + grounding test across 3 intent classes.

Tests one principle: L5 must produce SPECIFIC-EVENT URLs (Tier 1), not
category pages. Same pipeline works for hiring/funding/product_launch
by varying the search query, not the architecture.

Cases:
  Hiring         — Ramp, claim 'is hiring senior backend engineers'
  Funding        — Mercury, claim 'raised funding in 2025'
  Product launch — Anthropic, claim 'launched a new Claude model'
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from datetime import date, timedelta

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
EXA_KEY = os.environ.get("EXA_API_KEY")

import httpx


# =============================================================================
# URL Specificity Tier Classifier (deterministic, no LLM)
# =============================================================================

_CATEGORY_PATH_PATTERNS = re.compile(
    r"/(careers?|jobs?|press|news|blog|about|team|investors|company|insights|stories|updates|media)/?$",
    re.IGNORECASE,
)
_SPECIFIC_PATH_PATTERNS = re.compile(
    r"/(careers?|jobs?|press(?:-releases?)?|press_releases?|news|blog|insights|stories|articles|posts|"
    r"announcements?|launches?|product[-_]launch)/[\w\-/]{8,}",
    re.IGNORECASE,
)
_JOB_ID_PATTERNS = re.compile(r"/(jobs|positions?|careers?)/(view/)?\d{4,}|/jobs/view/\d+", re.IGNORECASE)
_DATE_IN_URL = re.compile(r"/20[12]\d[/-][01]\d[/-][0-3]\d", re.IGNORECASE)


def classify_url_tier(url: str) -> int:
    """Return 1=specific event, 2=aggregator, 3=category page, 4=homepage."""
    try:
        parsed = urlparse(url)
    except Exception:
        return 4
    path = parsed.path or "/"

    # Homepage check
    if path in ("", "/"):
        return 4

    # Specific patterns first (most specific wins)
    if _JOB_ID_PATTERNS.search(path):
        return 1
    if _DATE_IN_URL.search(path):
        return 1
    if _SPECIFIC_PATH_PATTERNS.search(path):
        return 1

    # Category landing pages
    if _CATEGORY_PATH_PATTERNS.search(path):
        return 3

    # LinkedIn job IDs
    if "linkedin.com/jobs/view/" in url:
        return 1

    # LinkedIn search/listing pages = aggregator
    if "linkedin.com/jobs/search" in url or "/search?" in path:
        return 2

    # Path with substantial depth and slug-like content = likely specific
    segments = [s for s in path.split("/") if s]
    if len(segments) >= 2 and any(len(s) > 12 and "-" in s for s in segments):
        return 1

    return 2  # default: aggregator


# =============================================================================
# Provider wrappers
# =============================================================================

async def exa_search_specific(client: httpx.AsyncClient, query: str, n: int = 8, days_back: int = 90) -> list[dict]:
    """Exa neural search filtered to recent specific posts."""
    start = (date.today() - timedelta(days=days_back)).isoformat()
    r = await client.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
        json={
            "query": query,
            "numResults": n,
            "useAutoprompt": True,
            "type": "neural",
            "startPublishedDate": start,
        },
        timeout=30.0,
    )
    if r.status_code != 200:
        return []
    return r.json().get("results") or []


async def exa_contents(client: httpx.AsyncClient, url: str) -> str:
    r = await client.post(
        "https://api.exa.ai/contents",
        headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
        json={"ids": [url], "text": True},
        timeout=45.0,
    )
    if r.status_code != 200:
        return ""
    results = r.json().get("results") or []
    return results[0].get("text", "") if results else ""


async def llm_json(client: httpx.AsyncClient, model: str, prompt: str) -> tuple[dict, float, float]:
    """Returns (parsed_json, cost_usd, elapsed_s)."""
    t0 = time.time()
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=60.0,
    )
    elapsed = time.time() - t0
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    cost = (data.get("usage") or {}).get("cost", 0)
    try:
        return json.loads(text), cost, elapsed
    except json.JSONDecodeError:
        return {"_raw": text[:300]}, cost, elapsed


# =============================================================================
# Intent-class-specific query templates (the only per-intent thing)
# =============================================================================

INTENT_QUERY_TEMPLATES = {
    "hiring": "{company} job posting senior {role} 2026",
    "funding": "{company} announces funding round raised investment 2025 2026",
    "product_launch": "{company} announces launches new product release",
    "leadership_change": "{company} appoints names hires CEO CTO VP",
    "expansion": "{company} opens new office expands international",
    "tech_adoption": "{company} uses adopts implemented {tech}",
    "compliance_event": "{company} certified SOC2 ISO compliance audit",
    "partnership": "{company} partners with announces partnership",
}


# =============================================================================
# Generalized verifier (works for any intent class)
# =============================================================================

async def verify_intent(
    client: httpx.AsyncClient,
    company: str,
    intent_class: str,
    claim: str,
    intent_params: dict = None,
) -> dict:
    """
    Generalized 4-stage verification (works across all intent classes):
      1. Discover candidate URLs via Exa neural (intent-specific query)
      2. Filter to Tier-1 (specific event) URLs
      3. Drill into best candidate via Exa /contents
      4. Strict grounding LLM
    """
    params = intent_params or {}
    template = INTENT_QUERY_TEMPLATES.get(intent_class, "{company} {claim}")
    query = template.format(company=company, **{**params, "claim": claim})

    print(f"  [Step 1] Exa search: query={query!r}")
    results = await exa_search_specific(client, query, n=10, days_back=180)
    print(f"     → {len(results)} raw results")

    if not results:
        return {"verdict": "ABSTAIN", "reason": "no search results"}

    # Step 2: classify and filter
    tiered = []
    for r in results:
        url = r.get("url", "")
        tier = classify_url_tier(url)
        tiered.append((tier, url, r.get("title", "")))
    tier1 = [(u, t) for tier, u, t in tiered if tier == 1]
    tier2 = [(u, t) for tier, u, t in tiered if tier == 2]
    print(f"  [Step 2] Specificity: Tier1={len(tier1)} Tier2={len(tier2)} "
          f"Tier3/4={sum(1 for tier,_,_ in tiered if tier >= 3)}")

    if not tier1:
        return {"verdict": "ABSTAIN", "reason": "no Tier-1 specific URLs found"}

    print(f"     Top Tier-1 candidates:")
    for u, t in tier1[:3]:
        print(f"       - {u}  [{t[:60]}]")

    # Step 3: try grounding against top-3 Tier-1 URLs in turn
    for candidate_url, candidate_title in tier1[:3]:
        content = await exa_contents(client, candidate_url)
        if not content or len(content) < 200:
            print(f"  [Step 3] {candidate_url} → empty content, skip")
            continue

        # Company mention check (Gate 2)
        head = content[:5000].lower()
        if head.count(company.lower()) < 1:
            print(f"  [Step 3] {candidate_url} → no company mention, skip")
            continue

        # Gate 5: strict grounding
        prompt = f"""You are a strict fact-checker.

URL: {candidate_url}
CONTENT (≤6000 chars):
{content[:6000]}

CLAIM: "{company} {claim}"

Does the CONTENT contain DIRECT, EXPLICIT evidence for the CLAIM?

Rules:
- Direct = literally states or shows the fact
- Marketing copy / general statements is NOT direct evidence
- For hiring claims: a specific job listing or role announcement counts
- For funding claims: a dated round announcement counts
- For product launches: a dated release announcement counts
- The proof must be copyable VERBATIM from the content

Output JSON only (no code fences):
{{
  "supported": true|false,
  "proof_quote": "<exact sentence from CONTENT, ≤300 chars, verbatim>",
  "confidence": <0-100>,
  "extracted_date": "<YYYY-MM-DD or null>",
  "rejection_reason": "<short, if not supported>"
}}"""
        grounding, cost, elapsed = await llm_json(client, "anthropic/claude-sonnet-4.6", prompt)
        sup = grounding.get("supported")
        conf = grounding.get("confidence", 0)
        proof = grounding.get("proof_quote", "")
        quote_in_content = bool(proof) and (proof in content)
        print(f"  [Step 3] Ground {candidate_url}")
        print(f"     supported={sup}, confidence={conf}, "
              f"date={grounding.get('extracted_date')}, "
              f"anti_halluc={quote_in_content}, "
              f"cost=${cost:.4f} {elapsed:.1f}s")
        if proof:
            print(f"     proof: {proof[:130]!r}")
        if grounding.get("rejection_reason"):
            print(f"     reject_reason: {grounding['rejection_reason'][:130]}")

        if sup and conf >= 80 and quote_in_content:
            return {
                "verdict": "ACCEPT",
                "url": candidate_url,
                "proof_quote": proof,
                "confidence": conf,
                "extracted_date": grounding.get("extracted_date"),
            }

    return {"verdict": "ABSTAIN", "reason": "no Tier-1 URL grounded successfully"}


# =============================================================================
# Runner — 3 intent classes
# =============================================================================

CASES = [
    {"label": "hiring",
     "company": "Ramp",
     "intent_class": "hiring",
     "claim": "is hiring senior backend engineers",
     "params": {"role": "backend engineer"},
     "expected": "ACCEPT"},
    {"label": "funding",
     "company": "Mercury",
     "intent_class": "funding",
     "claim": "raised funding in 2025 or 2026",
     "params": {},
     "expected": "ACCEPT"},
    {"label": "product_launch",
     "company": "Anthropic",
     "intent_class": "product_launch",
     "claim": "launched a new Claude model in 2026",
     "params": {},
     "expected": "ACCEPT"},
]


async def main():
    async with httpx.AsyncClient() as client:
        total_cost = 0.0
        results = []
        for case in CASES:
            print("\n" + "=" * 78)
            print(f" CASE: {case['label']}  ({case['intent_class']})")
            print(f"  Company: {case['company']}    Claim: {case['claim']!r}")
            print(f"  Expected: {case['expected']}")
            print("=" * 78)
            t0 = time.time()
            result = await verify_intent(
                client, case["company"], case["intent_class"], case["claim"], case["params"],
            )
            elapsed = time.time() - t0
            print(f"\n  ▶ Verdict: {result['verdict']}  ({elapsed:.1f}s)")
            if result["verdict"] == "ACCEPT":
                print(f"     URL: {result['url']}")
                print(f"     Date: {result.get('extracted_date')}")
            else:
                print(f"     Reason: {result.get('reason')}")
            agree = result["verdict"] == case["expected"]
            print(f"  Expected={case['expected']}, Got={result['verdict']} → {'✓ AGREE' if agree else '✗ DISAGREE'}")
            results.append((case, result, agree))

        print("\n" + "=" * 78)
        print(" SUMMARY")
        print("=" * 78)
        n_agree = sum(1 for _, _, a in results if a)
        print(f"  {n_agree}/{len(results)} cases agreed with expected verdict")
        for case, result, agree in results:
            mark = "✓" if agree else "✗"
            print(f"   {mark} {case['label']:15s}  expected={case['expected']:8s}  got={result['verdict']}")


if __name__ == "__main__":
    asyncio.run(main())
