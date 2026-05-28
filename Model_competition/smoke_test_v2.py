"""Smoke test v2 — sharper grounding test.

Cases:
  A. Accept case: a real Mercury careers page for the claim 'is hiring backend engineers'
  B. Reject case: the actual Pierce Manufacturing URL from competition data
     (claim: 'undergoing ERP modernization') — known fabrication, our gates should refuse it.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

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


def parse_json(text: str) -> dict:
    """Parse JSON, tolerating ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    r = await client.post(
        "https://api.exa.ai/contents",
        headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
        json={"ids": [url], "text": True},
        timeout=45.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Exa HTTP {r.status_code}: {r.text[:200]}")
    results = r.json().get("results") or []
    if not results:
        return ""
    return results[0].get("text", "")


async def ground(client: httpx.AsyncClient, url: str, claim: str, content: str) -> dict:
    prompt = f"""You are a strict fact-checker.

URL: {url}
CONTENT (≤6000 chars):
{content[:6000]}

CLAIM: "{claim}"

Does the CONTENT contain DIRECT, EXPLICIT evidence for the CLAIM?

Rules:
- Direct = literally states or shows the fact
- Marketing copy / general culture talk is NOT direct evidence
- The proof must be copyable VERBATIM from the content

Output JSON only (no code fences):
{{
  "supported": true|false,
  "proof_quote": "<exact sentence from CONTENT, ≤200 chars, copied verbatim>",
  "confidence": <0-100>,
  "rejection_reason": "<short, if not supported>"
}}"""
    t0 = time.time()
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60.0,
    )
    elapsed = time.time() - t0
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    parsed = parse_json(raw)
    # Anti-hallucination check: proof_quote literally in content
    proof = parsed.get("proof_quote") or ""
    quote_in_content = bool(proof) and proof in content
    return {
        **parsed,
        "anti_hallucination_pass": quote_in_content,
        "elapsed_s": round(elapsed, 2),
        "cost_usd": (data.get("usage") or {}).get("cost", 0),
    }


async def cross_check(client: httpx.AsyncClient, company: str, claim: str) -> dict:
    prompt = f"""Is it currently true that {company} {claim}?

Strict rules:
- Use real-time web search
- Answer YES with sources if you find direct evidence
- Answer NO with sources if you find contradicting evidence
- Answer UNKNOWN if you can't determine

Output format:
ANSWER: <YES|NO|UNKNOWN>
EVIDENCE: <one sentence>
SOURCES: <url1>, <url2>"""
    t0 = time.time()
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": "perplexity/sonar-pro",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60.0,
    )
    elapsed = time.time() - t0
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    m = re.search(r"ANSWER:\s*(YES|NO|UNKNOWN)", raw, re.IGNORECASE)
    answer = m.group(1).upper() if m else "UNKNOWN"
    return {
        "answer": answer,
        "raw": raw[:400],
        "elapsed_s": round(elapsed, 2),
        "cost_usd": (data.get("usage") or {}).get("cost", 0),
    }


async def run_case(client: httpx.AsyncClient, label: str, company: str, claim: str, url: str, expected: str):
    print(f"\n{'='*70}\n CASE {label}: expected={expected}\n  Company: {company}\n  Claim: {claim!r}\n  URL: {url}\n{'='*70}")
    try:
        content = await fetch_url(client, url)
    except Exception as e:
        print(f"  Exa fetch failed: {e}")
        return
    print(f"  Content length: {len(content)} chars")
    if not content or len(content) < 100:
        print("  → Gate 1 (reachability) FAIL — content too short")
        return

    # Gate 2: company linkage (simple check — name appears ≥ 2 times in first 5000 chars)
    head = content[:5000].lower()
    mentions = head.count(company.lower())
    print(f"  Gate 2 (company linkage): {mentions} mentions of '{company}' in first 5k chars → {'PASS' if mentions >= 2 else 'FAIL'}")
    if mentions < 1:
        print("  → Stop (page not about this company)")
        return

    # Gate 3: negation scan
    negation_re = re.compile(
        r"\b(0\s+(open|available)|no\s+(open|active)|no\s+longer|not\s+(accepting|hiring|currently)|page\s+not\s+found|404)\b",
        re.IGNORECASE,
    )
    neg = negation_re.search(content[:10000])
    if neg:
        print(f"  Gate 3 (negation): MATCH {neg.group(0)!r} → REJECT")
        return
    print("  Gate 3 (negation): no patterns → PASS")

    # Gate 5: strict grounding
    g = await ground(client, url, f"{company} {claim}", content)
    print(f"  Gate 5 (grounding): supported={g.get('supported')}, confidence={g.get('confidence')}, "
          f"reason={g.get('rejection_reason')!r}")
    print(f"    Proof quote: {g.get('proof_quote','')[:150]!r}")
    print(f"    Anti-hallucination (quote in content): {g.get('anti_hallucination_pass')}")
    print(f"    Cost: ${g.get('cost_usd',0):.4f}  Latency: {g.get('elapsed_s')}s")

    if not g.get("supported") or g.get("confidence", 0) < 80:
        print(f"  → REJECT at Gate 5")
        verdict = "REJECT"
    elif not g.get("anti_hallucination_pass"):
        print("  → REJECT at Gate 6 (hallucinated quote)")
        verdict = "REJECT"
    else:
        # Gate 7: cross-check
        c = await cross_check(client, company, claim)
        print(f"  Gate 7 (cross-check Sonar): {c['answer']} (cost ${c['cost_usd']:.4f}, {c['elapsed_s']}s)")
        if c["answer"] == "NO":
            verdict = "REJECT"
        else:
            verdict = "ACCEPT"
    print(f"  ▶ FINAL VERDICT: {verdict}  (expected: {expected})  → {'✓ AGREE' if verdict == expected else '✗ DISAGREE'}")


async def main():
    async with httpx.AsyncClient() as client:
        # Case A: REJECT — Mercury homepage for a hiring claim (homepage doesn't mention hiring)
        await run_case(client,
            label="A",
            company="Mercury",
            claim="is hiring senior backend engineers",
            url="https://mercury.com",
            expected="REJECT",
        )
        # Case B: known fabrication from the actual competition leaderboard data
        # The miner submitted this URL as proof that Pierce Manufacturing is
        # "Undergoing digital transformation" for an ERP modernization ICP.
        # The validator's rejected this as 'Intent fabrication'.
        await run_case(client,
            label="B",
            company="Pierce Manufacturing",
            claim="is undergoing ERP system modernization for new supply chain pressures",
            url="https://www.piercemfg.com/pierce/press-release/pierce-manufacturing-launches-phase-i-of-high-flow-production-line-to-further-accelerate-lead-time-reductions-and-enhance-quality",
            expected="REJECT",
        )
        # Case C: ACCEPT — a real LinkedIn/builtin job page for backend role
        await run_case(client,
            label="C",
            company="Ramp",
            claim="is hiring backend engineers",
            url="https://ramp.com/careers",
            expected="ACCEPT",
        )


if __name__ == "__main__":
    asyncio.run(main())
