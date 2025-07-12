"""
Mini-Intent Model (port of IntentModel-main scoring essentials).

API  ───────────────────────────────────────────────────────────────
    ranked = await rank_leads(leads, industry="Tech & AI", region="US")

Each lead already contains a `conversion_score` (0-1) from the open-source
validator.  We add:

    • fit_score         — firmographic match (industry & region)
    • intent_score      — keyword intent relevance
    • final_score       — blended   conversion ⨉ (1-α) + intent ⨉ α

Weights come from IntentModel-main settings (≈ 0.6 / 0.4 blend).
"""
from __future__ import annotations
import re, math, asyncio
from typing import List, Dict, Optional

# ---------------------------------------------------------------------
# Config (mirrors app/core/config.py defaults, hard-coded for now)
FIT_WEIGHT_INDUSTRY = 0.45
FIT_WEIGHT_REGION   = 0.15
FINAL_SCORE_FIT_W   = 0.6
FINAL_SCORE_INT_W   = 0.4

KEYWORDS = {
    "tech & ai": ["saas", "ai", "ml", "software", "cloud", "data"],
    "finance & fintech": ["fintech", "payments", "bank", "crypto", "lending"],
    "health & wellness": ["health", "clinic", "medical", "fitness", "wellness"],
    "media & education": ["edtech", "education", "course", "content", "media"],
    "energy & industry": ["energy", "solar", "oil", "gas", "renewable"],
}

# ---------------------------------------------------------------------
def _norm(txt: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (txt or "").lower()).strip()

def _industry_match(lead_ind: str, target: Optional[str]) -> float:
    if not target:
        return 0.5
    if not lead_ind:
        return 0.0
    lead   = _norm(lead_ind)
    target = _norm(target)
    if target in lead or lead in target:
        return 1.0
    # partial word overlap
    lw = set(lead.split())
    tw = set(target.split())
    return 0.7 if lw & tw else 0.0

def _region_match(lead_reg: str, target: Optional[str]) -> float:
    if not target:
        return 0.5
    return 1.0 if _norm(lead_reg) == _norm(target) else 0.0

def _intent_score(business: str, website: str, industry: Optional[str]) -> float:
    if not industry:
        return 0.0
    tokens = set((_norm(business) + " " + _norm(website)).split())
    score  = 0.0
    for kw in KEYWORDS.get(industry.lower(), []):
        if kw in tokens:
            score += 0.2     # simple additive weight
    return min(score, 1.0)

# ---------------------------------------------------------------------
import asyncio, os, json, textwrap, requests

OPENROUTER = os.getenv("OPENROUTER_API_KEY")

PROMPT_SYSTEM = (
    "You are a B2B lead-generation assistant.  "
    "Given a buyer description and a candidate lead, reply with JSON ONLY: "
    '{"score": <0-1 float where 1 = perfect fit>}')

async def _score_one(lead: dict, description: str) -> float:
    if not OPENROUTER:
        return 0.5             # fallback default
    prompt_user = (
        f"BUYER:\n{description}\n\n"
        f"LEAD:\n"
        f"Company: {lead.get('Business')}\n"
        f"Sub-industry: {lead.get('sub_industry')}\n"
        f"Website: {lead.get('Website')}"
    )
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER}",
                     "Content-Type": "application/json"},
            json={
                "model": "mistralai/mistral-7b-instruct",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user",   "content": prompt_user}
                ]},
            timeout=15)
        r.raise_for_status()
        j = json.loads(r.json()["choices"][0]["message"]["content"])
        return float(j.get("score", 0.5))
    except Exception:
        return 0.5    # graceful degrade

async def rank_leads(leads: list[dict], description: str) -> list[dict]:
    tasks = [_score_one(ld, description) for ld in leads]
    scores = await asyncio.gather(*tasks)
    for ld, sc in zip(leads, scores):
        ld["conversion_score"] = sc
    return sorted(leads, key=lambda x: x["conversion_score"], reverse=True) 