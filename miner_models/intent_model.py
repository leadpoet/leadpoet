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
import logging
logging.basicConfig(level=logging.WARNING)

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
# Fast heuristic to map a free-form buyer description → high-level industry
def infer_industry(description: str) -> Optional[str]:
    """
    Very lightweight – scans the buyer text for any keyword token belonging to
    one of our five umbrella industries.
    Returns the matching industry string (e.g. 'tech & ai') or None.
    """
    d = _norm(description)
    for bucket, kws in KEYWORDS.items():
        for kw in kws:
            if kw in d:
                return bucket
    return None

# ------------- LLM industry classifier ---------------------------------
INDUSTRIES = [
    "Tech & AI",
    "Finance & Fintech",
    "Health & Wellness",
    "Media & Education",
    "Energy & Industry",
]


def classify_industry(description: str) -> Optional[str]:
    """
    Use OpenRouter to map free-form buyer text → one of the 5 umbrella
    INDUSTRIES.

    Flow:
        1. Try PRIMARY_MODEL.
        2. On failure → try FALLBACK_MODEL.
        3. On failure → keyword heuristic.
    """
    fallback = infer_industry(description)

    if not OPENROUTER:
        return fallback

    prompt_system = (
        "You are an industry classifier. "
        "Return JSON ONLY: {\"industry\":\"<exact one of "
        + " | ".join(INDUSTRIES)
        + ">\"}"
    )
    prompt_user = _norm(description)[:400]

    print("\n🛈  INDUSTRY-LLM  INPUT ↓")
    print(prompt_user)

    def _try_model(model_name: str) -> Optional[str]:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user},
                ],
            },
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        print("🛈  INDUSTRY-LLM  OUTPUT ↓")
        print(raw[:300])
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        ind = json.loads(raw).get("industry", "")
        if ind in INDUSTRIES:
            print(f"✅ INDUSTRY-LLM succeeded → {ind}  (model: {model_name})")
            return ind.lower()
        return None

    # 1️⃣ Primary
    try:
        result = _try_model(PRIMARY_MODEL)
        if result:
            return result
    except Exception as e:
        logging.warning(f"Industry LLM primary model failed: {e}")

    # 2️⃣ Fallback
    try:
        print("⚠️  Primary model failed – trying fallback")
        result = _try_model(FALLBACK_MODEL)
        if result:
            return result
    except Exception as e:
        logging.warning(f"Industry LLM fallback model failed: {e}")

    # 3️⃣ Heuristic
    print("⚠️  INDUSTRY-LLM failed – using heuristic")
    return fallback.lower() if fallback else None

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
    "You are a B2B lead-generation assistant.\n"
    "FIRST LINE → JSON ONLY  {\"score\": <0-1 float>}  (0-0.8 bad ↔ 1 perfect)\n"
    "SECOND LINE → a brief explanation (max 40 words).\n")

PRIMARY_MODEL   = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL  = "mistralai/mistral-7b-instruct"
# For backward-compat parts of the file (industry LLM)  
MODEL_NAME      = PRIMARY_MODEL

def _call(model: str, prompt_user: str):             # ← sync now
    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user",   "content": prompt_user}
            ]},
        timeout=15)

async def _score_one(lead: dict, description: str) -> float:
    """
    Returns a 0-1 intent-fit score for one lead.
    1) Try OpenRouter → JSON {"score": x}
    2) If key missing or request fails → keyword heuristic
    """

    # ---------------- heuristic fallback -----------------
    def _heuristic() -> float:
        desc_tokens = set(_norm(description).split())
        lead_ind    = _norm(lead.get("Industry", ""))
        match = 0.0
        for bucket, kws in KEYWORDS.items():
            if bucket in lead_ind:
                for kw in kws:
                    if kw in desc_tokens:
                        match += 0.2
        return min(match, 1.0) or 0.05      # never zero

    if not OPENROUTER:
        return _heuristic()

    prompt_user = (
        f"BUYER:\n{description}\n\n"
        f"LEAD:\n"
        f"Company: {lead.get('Business')}\n"
        f"Sub-industry: {lead.get('sub_industry')}\n"
        f"Website: {lead.get('Website')}"
    )
    # ─── debug ─────────────────────────────────────────────────────
    print("\n🛈  INTENT-LLM  INPUT ↓")
    print(textwrap.shorten(prompt_user, width=300, placeholder=" …"))

    try:
        r = await asyncio.to_thread(_call, PRIMARY_MODEL, prompt_user)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        print("🛈  INTENT-LLM  OUTPUT ↓")
        print(textwrap.shorten(raw, width=300, placeholder=" …"))
        print(f"✅ OpenRouter call succeeded (model: {PRIMARY_MODEL})")
        # strip back-ticks / code-block fences
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        # grab first {...} span
        start, end = raw.find("{"), raw.rfind("}")
        payload    = raw[start:end + 1] if start != -1 and end != -1 else raw
        try:
            score = float(json.loads(payload).get("score", _heuristic()))
        except Exception:
            # LLM responded with non-JSON despite our prompt
            logging.warning(f"LLM non-JSON response: {raw[:100]}…")
            score = _heuristic()
        return score
    except Exception as e:
        print(f"⚠️  Primary model failed ({e}) – trying fallback")
        try:
            r = await asyncio.to_thread(_call, FALLBACK_MODEL, prompt_user)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()

            # ─── debug output for fallback ───────────────────────────────
            print("🛈  INTENT-LLM  OUTPUT ↓")
            print(textwrap.shorten(raw, width=300, placeholder=" …"))
            print(f"✅ OpenRouter call succeeded (model: {FALLBACK_MODEL})")
            # ------------------------------------------------------------

            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            start, end = raw.find("{"), raw.rfind("}")
            payload = raw[start:end + 1]
            return float(json.loads(payload).get("score", _heuristic()))
        except Exception as e2:
            logging.warning(f"Fallback model failed: {e2}")
            return _heuristic()

async def rank_leads(leads: list[dict], description: str) -> list[dict]:
    """
    Blend existing conversion_score (fit / legitimacy) with fresh
    intent_score from the LLM or heuristic:
        final = 0.6 * conversion + 0.4 * intent
    """
    tasks  = [_score_one(ld, description) for ld in leads]
    intents = await asyncio.gather(*tasks)

    for ld, intent in zip(leads, intents):
        ld["miner_intent_score"] = round(intent, 3)

    return sorted(leads, key=lambda x: x["miner_intent_score"], reverse=True) 