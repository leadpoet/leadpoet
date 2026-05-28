"""Smoke test for the qualification engine plan.

Tests three pillars against real APIs:
  1. Exa /contents: can we fetch clean content from a company website?
  2. Sonar discovery: can we get a synthesized list of candidates for an ICP?
  3. Sonnet 4.6 grounding: can we strictly verify a real claim against fetched content?

No code from the plan is implemented yet — this just smoke-tests the providers.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Load .env
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
SCRAPINGDOG_KEY = os.environ.get("SCRAPINGDOG_API_KEY")

assert OPENROUTER_KEY and EXA_KEY and SCRAPINGDOG_KEY, "Missing API keys"

import httpx


async def test_exa_contents(client: httpx.AsyncClient) -> dict:
    """Test 1: Fetch clean markdown of a known company website."""
    t0 = time.time()
    r = await client.post(
        "https://api.exa.ai/contents",
        headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
        json={"ids": ["https://mercury.com"], "text": True},
        timeout=30.0,
    )
    elapsed = time.time() - t0
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "elapsed_s": elapsed}
    data = r.json()
    results = data.get("results", [])
    if not results:
        return {"ok": False, "error": "no results", "elapsed_s": elapsed, "raw": data}
    text = results[0].get("text", "")
    return {
        "ok": len(text) > 500,
        "url": results[0].get("url"),
        "title": results[0].get("title"),
        "text_len": len(text),
        "mentions_mercury": "Mercury" in text or "mercury" in text.lower(),
        "elapsed_s": round(elapsed, 2),
        "first_500_chars": text[:500],
    }


async def test_sonar_discovery(client: httpx.AsyncClient) -> dict:
    """Test 2: Use Perplexity Sonar via OpenRouter to find candidate companies for an ICP."""
    t0 = time.time()
    icp_summary = (
        "Series B fintech companies headquartered in New York City, "
        "50-200 employees, currently hiring senior backend engineers in 2026."
    )
    prompt = f"""List 8 real companies matching: {icp_summary}

Strict rules:
- Real companies only — do not invent
- Each must include a source URL
- If you cannot find 8, return what you can

Output ONLY a JSON array (no prose):
[{{"name": "...", "domain": "...", "why_it_matches": "<10 words>", "source_url": "..."}}, ...]"""

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
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}", "elapsed_s": elapsed}
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # Try to find JSON array in response
    import re
    json_match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
    candidates = []
    if json_match:
        try:
            candidates = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"JSON parse: {e}", "raw": content[:500], "elapsed_s": elapsed}
    cost = (data.get("usage") or {}).get("cost", 0)
    return {
        "ok": len(candidates) >= 3,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "cost_usd": cost,
        "elapsed_s": round(elapsed, 2),
        "raw_excerpt": content[:300],
    }


async def test_sonnet_grounding(client: httpx.AsyncClient, content: str) -> dict:
    """Test 3: Run strict grounding on Mercury homepage for a hiring claim."""
    t0 = time.time()
    claim = "Mercury is hiring senior backend engineers"
    prompt = f"""You are a strict fact-checker.

URL: https://mercury.com
CONTENT (excerpt, ≤4000 chars):
{content[:4000]}

CLAIM: "{claim}"

Question: Does the CONTENT contain DIRECT, EXPLICIT evidence for the CLAIM?

Rules:
- Direct = content literally states or shows the fact
- Marketing copy / general statements about hiring culture is NOT direct evidence
- The proof must be copy-able VERBATIM from the content

Output JSON only:
{{
  "supported": true|false,
  "proof_quote": "<exact sentence from CONTENT, ≤200 chars, copied verbatim>",
  "confidence": <0-100>,
  "rejection_reason": "<short, if not supported>"
}}"""

    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=60.0,
    )
    elapsed = time.time() - t0
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}", "elapsed_s": elapsed}
    data = r.json()
    content_out = data["choices"][0]["message"]["content"]
    try:
        result = json.loads(content_out)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON parse: {e}", "raw": content_out[:300], "elapsed_s": elapsed}
    cost = (data.get("usage") or {}).get("cost", 0)
    # Verify anti-hallucination: proof_quote should be in content (substring)
    proof = result.get("proof_quote") or ""
    quote_in_content = proof in content if proof else False
    return {
        "ok": True,
        "supported": result.get("supported"),
        "proof_quote": proof,
        "confidence": result.get("confidence"),
        "rejection_reason": result.get("rejection_reason"),
        "anti_hallucination_pass": quote_in_content,
        "cost_usd": cost,
        "elapsed_s": round(elapsed, 2),
    }


async def main():
    print("=" * 70)
    print("Smoke test: Exa + OpenRouter Sonar + Claude Sonnet 4.6 grounding")
    print("=" * 70)
    async with httpx.AsyncClient() as client:
        print("\n[1/3] Exa /contents on https://mercury.com")
        exa_result = await test_exa_contents(client)
        print(json.dumps({k: v for k, v in exa_result.items() if k != "first_500_chars"}, indent=2))
        if exa_result.get("ok"):
            print(f"  First 300 chars: {exa_result['first_500_chars'][:300]!r}")

        print("\n[2/3] Sonar discovery for 'Series B fintech NYC hiring backend engineers'")
        sonar_result = await test_sonar_discovery(client)
        print(json.dumps({k: v for k, v in sonar_result.items() if k != "candidates"}, indent=2))
        if sonar_result.get("candidates"):
            print(f"  Top candidates: {[c.get('name') for c in sonar_result['candidates'][:5]]}")

        print("\n[3/3] Sonnet 4.6 strict grounding on Mercury homepage for 'is hiring senior backend engineers'")
        if exa_result.get("ok"):
            content = exa_result.get("first_500_chars", "")
            # Re-fetch full content for grounding
            r = await client.post(
                "https://api.exa.ai/contents",
                headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
                json={"ids": ["https://mercury.com"], "text": True},
                timeout=30.0,
            )
            full_content = r.json()["results"][0]["text"]
            grounding_result = await test_sonnet_grounding(client, full_content)
            print(json.dumps(grounding_result, indent=2))
        else:
            print("  SKIPPED — Exa fetch failed")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Exa /contents:      {'PASS' if exa_result.get('ok') else 'FAIL'}")
    print(f"  Sonar discovery:    {'PASS' if sonar_result.get('ok') else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
