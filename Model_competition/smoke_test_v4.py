"""v4 — generalized verifier with LLM-driven search query + semantic-aware grounding.

Two architectural changes vs v3:
  1. Search queries are LLM-generated per (company, intent_class, claim).
     This lets the system surface domain-specific URLs Exa's neural search
     wouldn't find from a static template.
  2. Grounding prompt explicitly allows reasonable semantic equivalence
     (e.g. "Software Engineer, Backend Platform" ≈ "backend engineer")
     while still requiring direct textual evidence — no marketing fluff.

Generalizes across intent classes — no per-intent code changes needed.
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
# URL specificity tier (unchanged)
# =============================================================================

_CATEGORY_PATH = re.compile(
    r"/(careers?|jobs?|press|news|blog|about|team|investors|company|insights|stories|updates|media)/?$",
    re.IGNORECASE,
)
_SPECIFIC_PATH = re.compile(
    r"/(careers?|jobs?|press(?:-releases?)?|press_releases?|news|blog|insights|stories|articles|posts|"
    r"announcements?|launches?|product[-_]launch)/[\w\-/]{8,}",
    re.IGNORECASE,
)
_JOB_ID = re.compile(r"/(jobs|positions?|careers?)/(view/)?\d{4,}|/jobs/view/\d+", re.IGNORECASE)
_DATE_IN_URL = re.compile(r"/20[12]\d[/-][01]\d[/-][0-3]\d", re.IGNORECASE)


def classify_url_tier(url: str) -> int:
    try:
        parsed = urlparse(url)
    except Exception:
        return 4
    path = parsed.path or "/"
    if path in ("", "/"):
        return 4
    if _JOB_ID.search(path):
        return 1
    if _DATE_IN_URL.search(path):
        return 1
    if _SPECIFIC_PATH.search(path):
        return 1
    if "linkedin.com/jobs/view/" in url:
        return 1
    if _CATEGORY_PATH.search(path):
        return 3
    if "linkedin.com/jobs/search" in url or "/search?" in path:
        return 2
    segments = [s for s in path.split("/") if s]
    if len(segments) >= 2 and any(len(s) > 12 and "-" in s for s in segments):
        return 1
    return 2


# =============================================================================
# Providers
# =============================================================================

async def exa_search(client, query: str, n: int = 10, days_back: int = 180) -> list[dict]:
    start = (date.today() - timedelta(days=days_back)).isoformat()
    r = await client.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
        json={"query": query, "numResults": n, "useAutoprompt": True,
              "type": "neural", "startPublishedDate": start},
        timeout=30.0,
    )
    return r.json().get("results", []) if r.status_code == 200 else []


async def exa_contents(client, url: str) -> str:
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


async def llm_json(client, model: str, prompt: str):
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
# Generalized verifier v4
# =============================================================================

async def generate_search_queries(client, company: str, intent_class: str, claim: str) -> tuple[list[str], float]:
    """LLM-generated, intent-aware search queries — generalizes across intents."""
    prompt = f"""Generate 3 web search queries that would find PROOF of this claim about a company.

Company: {company}
Intent class: {intent_class}
Claim: "{claim}"

Rules:
- Each query should target a SPECIFIC event/post URL — not a landing page
- Include the company name and the most evidence-bearing keywords
- For hiring: include role keywords + "job" or "career"
- For funding: include "raised" / "closed" / "Series X" / dollar amounts
- For product_launch: include "announces" / "launches" / "release" / product name
- For tech_adoption: include the specific tech + "uses" / "adopted"
- For leadership_change: include "appointed" / "joins" / "named" + role
- For expansion: include "opens" / "expands" / location name
- Recent date words like "2025" or "2026" when claim is time-bound

Output JSON ONLY:
{{"queries": ["...", "...", "..."]}}"""
    result, cost, _ = await llm_json(client, "openai/gpt-4o-mini", prompt)
    queries = result.get("queries", [])
    return queries[:3], cost


async def ground_semantic(client, company: str, claim: str, url: str, content: str) -> tuple[dict, float, float]:
    """Semantic-aware grounding — allows reasonable equivalents."""
    prompt = f"""You are a fact-checker. You are STRICT but allow reasonable semantic equivalence.

URL: {url}
CONTENT (≤6000 chars):
{content[:6000]}

CLAIM: "{company} {claim}"

Does the CONTENT contain DIRECT evidence supporting the CLAIM, allowing semantic equivalents?

What counts as DIRECT evidence (with semantic equivalence):
- "hiring backend engineer" ← supported by job titles like:
  Software Engineer · Backend Platform, Server Engineer, Infrastructure Engineer,
  Platform Engineer, API Engineer, Distributed Systems Engineer
  (NOT supported by: pure frontend, pure mobile, design, marketing, sales roles)
- "hiring senior" ← supported by: Senior / Sr. / Staff / Principal / Lead in title
  (NOT supported by: Intern, Junior, Entry-level)
- "raised funding" ← supported by: "announces $X round", "closed Series X", "secured $X investment"
  (NOT supported by: "in talks", "exploring", "considering", "rumored")
- "launched a model" ← supported by a specific dated release announcement for AI model/product
  (NOT supported by: feature announcements that BUILD ON existing models)
- "uses [tech]" ← supported by case studies, engineering blog posts, conference talks
  (NOT supported by: vague "modern stack" marketing)

What is NEVER evidence:
- Generic marketing copy ("committed to X")
- Future intent without specifics
- Page sections describing the company in general

The proof must be a sentence you can copy VERBATIM from the content.

Output JSON only (no code fences):
{{
  "supported": true|false,
  "proof_quote": "<exact sentence from CONTENT, ≤300 chars, verbatim>",
  "confidence": <0-100>,
  "semantic_match_notes": "<short — what equivalent was matched, if any>",
  "extracted_date": "<YYYY-MM-DD or null>",
  "rejection_reason": "<short, if not supported>"
}}"""
    result, cost, elapsed = await llm_json(client, "anthropic/claude-sonnet-4.6", prompt)
    return result, cost, elapsed


async def verify_intent(client, company: str, intent_class: str, claim: str) -> dict:
    total_cost = 0.0
    t0 = time.time()

    # Step 1: LLM-generated queries
    queries, cost = await generate_search_queries(client, company, intent_class, claim)
    total_cost += cost
    print(f"  [Step 1] LLM-generated queries (cost ${cost:.4f}):")
    for q in queries:
        print(f"     • {q}")

    # Step 2: Exa search across all queries (parallel)
    search_tasks = [exa_search(client, q, n=8, days_back=180) for q in queries]
    raw_results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
    seen_urls = set()
    all_results = []
    for results in raw_results_per_query:
        if isinstance(results, Exception):
            continue
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
    print(f"  [Step 2] {len(all_results)} unique URLs across {len(queries)} queries")

    if not all_results:
        return {"verdict": "ABSTAIN", "reason": "no search results", "cost": total_cost}

    # Step 3: Tier classification
    tier1 = []
    for r in all_results:
        url = r.get("url", "")
        if classify_url_tier(url) == 1:
            tier1.append((url, r.get("title", "")))
    print(f"  [Step 3] {len(tier1)} Tier-1 URLs after specificity filter")

    if not tier1:
        return {"verdict": "ABSTAIN", "reason": "no Tier-1 specific URLs", "cost": total_cost}

    # Step 4: Ground top 5
    for candidate_url, candidate_title in tier1[:5]:
        content = await exa_contents(client, candidate_url)
        if not content or len(content) < 200:
            continue

        # Company mention check
        head = content[:5000].lower()
        if head.count(company.lower()) < 1:
            print(f"  [Step 4] {candidate_url[:80]} → no company mention, skip")
            continue

        # Negation scan
        if re.search(r"\b(0\s+(open|available)|no\s+(open|active)|no\s+longer|404)\b", content[:10000], re.I):
            print(f"  [Step 4] {candidate_url[:80]} → negation, skip")
            continue

        # Semantic-aware grounding
        g, cost, elapsed = await ground_semantic(client, company, claim, candidate_url, content)
        total_cost += cost
        sup = g.get("supported")
        conf = g.get("confidence", 0)
        proof = g.get("proof_quote", "")
        quote_in_content = bool(proof) and (proof in content)
        print(f"  [Step 4] Ground {candidate_url[:75]}")
        print(f"           supported={sup}, conf={conf}, semantic={g.get('semantic_match_notes','')[:80]!r}")
        if proof:
            print(f"           proof: {proof[:120]!r}")
        if g.get("rejection_reason"):
            print(f"           reject: {g['rejection_reason'][:120]}")
        print(f"           cost=${cost:.4f} {elapsed:.1f}s")

        if sup and conf >= 80 and quote_in_content:
            return {
                "verdict": "ACCEPT",
                "url": candidate_url,
                "proof_quote": proof,
                "confidence": conf,
                "extracted_date": g.get("extracted_date"),
                "semantic_match": g.get("semantic_match_notes"),
                "cost": total_cost,
                "elapsed_s": time.time() - t0,
            }

    return {"verdict": "ABSTAIN", "reason": "no Tier-1 URL grounded successfully", "cost": total_cost,
            "elapsed_s": time.time() - t0}


# =============================================================================
# Cases — 4 intent classes to verify generalization
# =============================================================================

CASES = [
    {"label": "hiring",
     "company": "Ramp",
     "intent_class": "hiring",
     "claim": "is hiring senior backend engineers",
     "expected": "ACCEPT"},
    {"label": "funding",
     "company": "Anthropic",
     "intent_class": "funding",
     "claim": "raised funding in 2025 or 2026",
     "expected": "ACCEPT"},
    {"label": "product_launch",
     "company": "Anthropic",
     "intent_class": "product_launch",
     "claim": "launched Claude Opus 4.7 or Claude Sonnet 4.6 in 2026",
     "expected": "ACCEPT"},
    {"label": "leadership_change",
     "company": "OpenAI",
     "intent_class": "leadership_change",
     "claim": "appointed new executive in 2024 or 2025",
     "expected": "ACCEPT"},
]


async def main():
    async with httpx.AsyncClient() as client:
        results = []
        grand_cost = 0.0
        for case in CASES:
            print("\n" + "=" * 78)
            print(f" CASE: {case['label']}  Company: {case['company']}")
            print(f"  Claim: {case['claim']!r}    Expected: {case['expected']}")
            print("=" * 78)
            result = await verify_intent(client, case["company"], case["intent_class"], case["claim"])
            grand_cost += result.get("cost", 0)
            print(f"\n  ▶ VERDICT: {result['verdict']}  (cost ${result.get('cost',0):.4f}, {result.get('elapsed_s', 0):.1f}s)")
            if result["verdict"] == "ACCEPT":
                print(f"     URL: {result['url']}")
                print(f"     Proof: {result['proof_quote'][:150]!r}")
            else:
                print(f"     Reason: {result.get('reason')}")
            agree = result["verdict"] == case["expected"]
            print(f"  Expected={case['expected']}, Got={result['verdict']} → {'✓ AGREE' if agree else '✗ DISAGREE'}")
            results.append((case, result, agree))

        print("\n" + "=" * 78)
        print(" SUMMARY")
        print("=" * 78)
        n_agree = sum(1 for _, _, a in results if a)
        print(f"  {n_agree}/{len(results)} cases agree    Total cost: ${grand_cost:.4f}")
        for case, result, agree in results:
            mark = "✓" if agree else "✗"
            print(f"   {mark} {case['label']:18s}  expected={case['expected']:8s}  got={result['verdict']}")


if __name__ == "__main__":
    asyncio.run(main())
