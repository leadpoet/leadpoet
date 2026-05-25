"""
Fulfillment Stage 4 person verification.

Multi-step pipeline that uses cheap checks first, escalating to Apify only
for leads that pass initial filters:

  1. Q1 GSE search (site:linkedin.com/in/{slug}) → name check from title
  2. ScrapingDog person LinkedIn scraper (if Q1 fails) → name + location
  3. Apify (only if name verified) → company name, company URL, role,
     and location (if Q1 path — SD path already checked location)
"""

import asyncio
import json
import os
import re
import unicodedata
from typing import Optional, Tuple

import aiohttp

from gateway.utils.geo_normalize import (
    COUNTRY_ALIASES,
    normalize_country,
    normalize_city,
    normalize_state,
)
from gateway.utils.role_translate import translate_to_english, is_configured

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")

_APIFY_ACTOR = "harvestapi~linkedin-profile-scraper"
_APIFY_ENDPOINT = (
    f"https://api.apify.com/v2/acts/{_APIFY_ACTOR}/run-sync-get-dataset-items"
)
_APIFY_TIMEOUT_S = 120
_SD_GOOGLE_URL = "https://api.scrapingdog.com/google"
_SD_LINKEDIN_URL = "https://api.scrapingdog.com/linkedin"
_MAX_RETRIES = 2

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCATION_MODEL = "google/gemini-2.5-flash-lite"
_TM_RE = re.compile(r'[®™©℠]+')
_LEGAL_SUFFIXES_RE = re.compile(
    r',?\s*\b(?:Inc\.?|LLC|L\.L\.C\.?|Corp\.?|Corporation|Ltd\.?|Limited|'
    r'Co\.?|Company|Group|PLC|P\.L\.C\.?|LLP|L\.L\.P\.?|PLLC|P\.L\.L\.C\.?|'
    r'PC|P\.C\.?|LP|L\.P\.?|NA|N\.A\.?)\s*\.?\s*$', re.IGNORECASE
)


def _normalize_company(name: str) -> str:
    """Normalize company name for comparison.
    Strips trademarks, legal suffixes, normalizes &/and, case-insensitive."""
    s = _TM_RE.sub('', name).strip()
    s = _LEGAL_SUFFIXES_RE.sub('', s).strip().rstrip(',')
    s = re.sub(r'\band\b', '&', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

_ROLE_PROMPT = (
    'LinkedIn shows current role: "{actual_role}" at "{company}".\n'
    'Miner claims role: "{claimed_role}".\n\n'
    'IMPORTANT: Compare ONLY the two title strings. Do NOT use your knowledge '
    'about the company or who actually holds this role.\n\n'
    'Are these the SAME job title? Answer true if:\n'
    '- Only formatting differs (Sr./Senior, VP/Vice President, &/and, |/-, ®, punctuation)\n'
    '- One is a shorter version of the other (e.g. "Director, Marketing | Americas" = "Director, Marketing")\n'
    '- Title with extra qualifier matches base title (e.g. "Manufacturing Engineer - Advanced" = "Manufacturing Engineer")\n'
    '- A combined title contains the claimed role (e.g. "Co-Founder & CEO" = "CEO", "Co-Founder/CTO" = "CTO")\n'
    '- Owner/Founder at the same small company\n'
    'Answer false if:\n'
    '- Different job function (e.g. Engineer vs Sales, Planner vs Student)\n'
    '- Different department (e.g. Marketing vs Operations)\n'
    '- Different seniority (e.g. Associate vs Director)\n'
    'Return JSON only: {{"match": true/false}}'
)

_LOCATION_PROMPT = (
    'Miner claims: city="{m_city}", state="{m_state}", country="{m_country}"\n'
    'LinkedIn shows: "{li_location}"\n\n'
    'Step 1: Is the miner\'s claimed location a real, valid place? '
    'If the city+state+country combination does not exist, return {{"valid": false, "match": false}}.\n'
    'Step 2: Specificity check — the miner MUST be at least as specific as LinkedIn.\n'
    '- If LinkedIn shows a city (e.g. "San Luis, Argentina") but the miner only provided '
    'a country (no city, no state), return {{"valid": true, "match": false}}.\n'
    '- If LinkedIn shows a state (e.g. "California, USA") but the miner only provided '
    'a country, return {{"valid": true, "match": false}}.\n'
    '- Miner is allowed to be MORE specific than LinkedIn, not less.\n'
    'Step 3: If specificity is OK, does it actually match the LinkedIn location?\n'
    '- Match only the fields the miner provided.\n'
    '- For non-US locations, state is not required to match — only city and country.\n'
    '- The miner\'s city must correspond to a real city in that country, not '
    'simply the country name repeated.\n'
    'Return JSON only: {{"valid": true/false, "match": true/false}}'
)


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

_NAME_PREFIX_RE = re.compile(
    r'^\s*(?:Dr\.?|Prof\.?|Rev\.?|Sir|Dame)\s+', re.IGNORECASE
)


def _normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower().strip()
    return s.rstrip('.')


def _normalize_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')


def _check_name_in_title(full_name: str, title: str) -> bool:
    """Previous Stage 4 logic: substring name match in search result title."""
    if not full_name or not title:
        return False
    if full_name in title:
        return True
    if _normalize_accents(full_name) in _normalize_accents(title):
        return True
    return False


def _check_name_fullname(lead_full: str, sd_full: str) -> bool:
    """Name match using ScrapingDog's fullName field (join + comma strip)."""
    if not lead_full or not sd_full:
        return False
    # Strip after comma, strip prefix
    cleaned = sd_full.split(',')[0].strip()
    cleaned = _NAME_PREFIX_RE.sub('', cleaned).strip()
    if _normalize_name(cleaned) == _normalize_name(lead_full):
        return True
    # First word + last word
    c_parts = cleaned.split()
    l_parts = lead_full.split()
    if len(c_parts) >= 2 and len(l_parts) >= 2:
        if (_normalize_name(c_parts[0]) == _normalize_name(l_parts[0])
                and _normalize_name(c_parts[-1]) == _normalize_name(l_parts[-1])):
            return True
    return False


# ---------------------------------------------------------------------------
# Slug / URL helpers
# ---------------------------------------------------------------------------

def _get_linkedin_slug(url: str) -> str:
    """Extract LinkedIn person slug from URL, lowercased."""
    if not url:
        return ""
    m = re.search(r'linkedin\.com/in/([^/?#]+)', str(url), re.IGNORECASE)
    return m.group(1).lower().rstrip('/') if m else ""


def _extract_company_slug(url: str) -> str:
    """Extract the slug after /company/ without lowering or trimming."""
    parts = url.rstrip("/").split("/company/")
    if len(parts) > 1:
        return parts[-1].split("/")[0]
    return ""


# ---------------------------------------------------------------------------
# Rejection helper
# ---------------------------------------------------------------------------

def _rejection(check_name: str, message: str, failed_fields: list) -> dict:
    return {
        "stage": "Stage 4: Fulfillment Person Verification",
        "check_name": check_name,
        "message": message,
        "failed_fields": failed_fields,
    }


# ---------------------------------------------------------------------------
# Step 1: Q1 GSE search
# ---------------------------------------------------------------------------

async def _q1_gse_search(slug: str, sd_key: str) -> list:
    """Search Google for site:linkedin.com/in/{slug}. Returns organic results."""
    query = f"site:linkedin.com/in/{slug}"
    timeout = aiohttp.ClientTimeout(total=15)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(_SD_GOOGLE_URL, params={
                    "api_key": sd_key, "query": query, "results": 5,
                }) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("organic_results", [])
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    else:
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return []
        except Exception:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return []
    return []


# ---------------------------------------------------------------------------
# Step 2: ScrapingDog person LinkedIn scraper
# ---------------------------------------------------------------------------

async def _sd_person_scrape(slug: str, sd_key: str) -> Optional[dict]:
    """Fetch person profile from ScrapingDog LinkedIn scraper."""
    timeout = aiohttp.ClientTimeout(total=30)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(_SD_LINKEDIN_URL, params={
                    "api_key": sd_key, "type": "profile", "linkId": slug,
                }) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            return data[0]
                        if isinstance(data, dict) and data:
                            return data
                        return None
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    else:
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return None
        except Exception:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Step 3: Location LLM check
# ---------------------------------------------------------------------------

def _strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _us_location_match(m_city: str, m_state: str, m_country: str,
                       a_city: str, a_state: str, a_country: str) -> Optional[bool]:
    """Rule-based US location match. Returns True/False if resolved, None for LLM."""
    # Normalize both sides
    mc = normalize_city(m_city, m_country).lower() if m_city else ""
    ms = normalize_state(m_state, m_country).lower() if m_state else ""
    ac = normalize_city(a_city, a_country).lower() if a_city else ""
    a_s = normalize_state(a_state, a_country).lower() if a_state else ""

    # Also try diacritics stripped
    mc_s = _strip_diacritics(mc)
    ac_s = _strip_diacritics(ac)

    city_match = (mc == ac or mc_s == ac_s) if mc and ac else False
    state_match = (ms == a_s) if ms and a_s else False

    # Both city and state provided by miner
    if mc and ms:
        if city_match and state_match:
            return True
        return None  # Let LLM try (could be metro area vs city)

    # Only city provided
    if mc and not ms:
        if city_match:
            return True
        return None

    # Only state provided
    if ms and not mc:
        if state_match:
            return True
        return None

    return None


def _rule_based_location_match(m_city: str, m_state: str, m_country: str,
                                li_location: str) -> Optional[bool]:
    """
    Rule-based location match. Miner may have partial fields.
    Match only what the miner submitted.

    City/state are matched against the NON-country portion of li_location
    (everything except the last comma-separated part) so a miner who passes
    the country name as their city/state cannot pass via substring re-use.
    The country itself is matched against the full string.
    """
    if not li_location:
        return None

    li_lower = li_location.lower()
    li_stripped = _strip_diacritics(li_lower)

    parts = [p.strip() for p in li_location.split(",") if p.strip()]
    if len(parts) > 1:
        non_country_li = ", ".join(parts[:-1]).lower()
    else:
        non_country_li = ""
    non_country_stripped = _strip_diacritics(non_country_li)

    m_country_norm = normalize_country(m_country).lower() if m_country else ""

    if m_country_norm:
        country_in_li = (
            m_country_norm in li_lower
            or m_country_norm in li_stripped
            or m_country.lower() in li_lower
        )
        if not country_in_li:
            for alias, canonical in COUNTRY_ALIASES.items():
                if canonical == m_country_norm and alias in li_lower:
                    country_in_li = True
                    break
        if not country_in_li:
            return None

    if m_state:
        if not non_country_li:
            return None
        m_state_norm = normalize_state(m_state, m_country).lower()
        state_in_li = (
            m_state_norm in non_country_li
            or m_state.lower() in non_country_li
            or _strip_diacritics(m_state.lower()) in non_country_stripped
        )
        if not state_in_li:
            return None

    if m_city:
        if not non_country_li:
            return None
        m_city_norm = normalize_city(m_city, m_country).lower()
        city_in_li = (
            m_city_norm in non_country_li
            or m_city.lower() in non_country_li
            or _strip_diacritics(m_city.lower()) in non_country_stripped
        )
        if not city_in_li:
            return None

    return True


def _location_specificity_gate(m_city: str, m_state: str, m_country: str,
                                li_location: str) -> Optional[bool]:
    """Reject when miner's specificity is below LinkedIn's.

    Returns False (reject) if miner provided fewer location parts than
    LinkedIn shows (i.e. miner could have been more specific but wasn't).
    Returns None to let the regular match logic run.

    Comma-separated parts count as the specificity proxy:
      LI "San Luis, Argentina" = 2 parts (city + country)
      LI "Argentina" = 1 part (country only)
    Miner specificity = number of non-empty fields among city/state/country.
    """
    if not li_location:
        return None

    miner_n = sum(1 for v in (m_city, m_state, m_country) if (v or "").strip())
    li_n = len([p for p in li_location.split(",") if p.strip()])

    if miner_n < li_n:
        return False
    return None


async def _llm_location_check(
    m_city: str, m_state: str, m_country: str,
    li_location: str, openrouter_key: str,
) -> bool:
    """Location check: specificity gate (deterministic), then LLM."""
    gate = _location_specificity_gate(m_city, m_state, m_country, li_location)
    if gate is False:
        print(f"   📍 Location ❌ (miner less specific than LinkedIn)")
        return False

    if not openrouter_key:
        print("   ⚠️ No OpenRouter key for location LLM")
        return False

    prompt = _LOCATION_PROMPT.format(
        m_city=m_city, m_state=m_state, m_country=m_country,
        li_location=li_location,
    )

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _LOCATION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0,
                },
            ) as resp:
                if resp.status != 200:
                    print(f"   ⚠️ Location LLM HTTP {resp.status}")
                    return False
                body = await resp.json()
                content = body["choices"][0]["message"]["content"]
                match = re.search(r"\{[^}]+\}", content)
                if match:
                    result = json.loads(match.group())
                    valid = bool(result.get("valid", True))
                    llm_match = bool(result.get("match", False))
                    if not valid:
                        print(f"   ❌ Location: miner claimed invalid location (LLM)")
                        return False
                    print(f"   📍 Location {'✅' if llm_match else '❌'} (LLM)")
                    return llm_match
                return False
    except Exception as e:
        print(f"   ⚠️ Location LLM error: {e}")
        return False


# ---------------------------------------------------------------------------
# Role LLM fallback
# ---------------------------------------------------------------------------

async def _llm_role_match(
    actual_role: str, claimed_role: str, company: str, openrouter_key: str,
) -> bool:
    """LLM role comparison. Called when exact match fails."""
    a = actual_role.strip().lower()
    c = claimed_role.strip().lower()
    if a == c:
        return True
    # Check if claimed role is contained in a combined title
    # e.g. "Co-Founder & CEO" contains "CEO", "President & COO" contains "COO"
    for sep in (" & ", " and ", "/", " | ", ", "):
        parts = [p.strip() for p in a.split(sep)]
        if c in parts:
            return True
        parts_c = [p.strip() for p in c.split(sep)]
        if a in parts_c:
            return True
    if not openrouter_key:
        print("   ⚠️ No OpenRouter key for role LLM")
        return False

    prompt = _ROLE_PROMPT.format(
        actual_role=actual_role, claimed_role=claimed_role, company=company,
    )

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _LOCATION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0,
                },
            ) as resp:
                if resp.status != 200:
                    print(f"   ⚠️ Role LLM HTTP {resp.status}")
                    return False
                body = await resp.json()
                content = body["choices"][0]["message"]["content"]
                match = re.search(r"\{[^}]+\}", content)
                if match:
                    result = json.loads(match.group())
                    llm_match = bool(result.get("match", False))
                    print(f"   📋 Role {'✅' if llm_match else '❌'} (LLM)")
                    return llm_match
                return False
    except Exception as e:
        print(f"   ⚠️ Role LLM error: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 5: Apify fetch
# ---------------------------------------------------------------------------

async def _fetch_apify_profile(linkedin_url: str, api_token: str) -> Optional[dict]:
    """Fetch LinkedIn profile data from Apify with retry."""
    payload = {
        "profileScraperMode": "Profile details no email ($4 per 1k)",
        "queries": [linkedin_url],
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=_APIFY_TIMEOUT_S)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    _APIFY_ENDPOINT, json=payload, headers=headers,
                ) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            return data[0]
                        return None
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        print(f"   ⚠️ Apify rate limited, retrying ({attempt + 1}/{_MAX_RETRIES})...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        print(f"   ❌ Apify HTTP {resp.status}")
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return None
        except Exception as e:
            print(f"   ❌ Apify fetch error: {e}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Apify position helpers
# ---------------------------------------------------------------------------

def _get_current_position(apify_data: dict, lead_company: str = "") -> Optional[dict]:
    """Return the first current position from Apify data.

    Search order:
    1. currentPosition entries with valid position
    2. currentPosition with empty position → experience fallback for same company
    3. If currentPosition is empty → first current entry from experience
    """
    # Try currentPosition first — trust Apify's filtering
    for pos in apify_data.get("currentPosition", []):
        raw_pos = (pos.get("position") or "").strip()
        if raw_pos and raw_pos != "None":
            return pos
        # Position empty — try experience for same company
        company = pos.get("companyName", "")
        exp_pos = _find_role_in_experience(apify_data, company)
        if exp_pos:
            merged = dict(pos)
            merged["position"] = exp_pos["position"]
            merged["_role_source"] = "experience"
            return merged
        return pos

    # currentPosition empty/no "Present" — search experience
    first_current = _find_first_current_experience(apify_data, lead_company)
    if first_current:
        return first_current

    return None


def _find_first_current_experience(apify_data: dict, lead_company: str = "") -> Optional[dict]:
    """Find the best current experience entry.

    If lead_company is provided, prefer the entry matching that company.
    Otherwise return the first current entry with a valid role.
    """
    current_entries = []
    for exp in apify_data.get("experience", []):
        end = exp.get("endDate") or {}
        if end.get("text") != "Present":
            continue
        exp_role = (exp.get("position") or "").strip()
        if not exp_role or exp_role == "None":
            continue
        entry = {
            "position": exp_role,
            "companyName": (exp.get("companyName") or "").strip(),
            "companyLinkedinUrl": exp.get("companyLinkedinUrl") or exp.get("company_url") or "",
            "location": (exp.get("location") or "").strip(),
            "_role_source": "experience",
        }
        current_entries.append(entry)

    if not current_entries:
        return None

    # Prefer entry matching miner's claimed company
    if lead_company:
        for entry in current_entries:
            if entry["companyName"].lower() == lead_company.strip().lower():
                return entry

    # No company match — return first current entry
    return current_entries[0]


def _best_li_location(apify_data: dict, current_position: Optional[dict],
                       lead_company: str = "") -> str:
    """Build the LinkedIn location string.

    Title/headline location is the source of truth. The current job's
    experience location is used ONLY to fill in fields that are missing
    in the title (e.g. headline has just country, experience adds city).

    Country conflict between title and experience → trust the title.
    If neither source has any location data → returns "".
    """
    loc_field = apify_data.get("location") or {}
    parsed = (loc_field.get("parsed") if isinstance(loc_field, dict) else None) or {}
    h_city = (parsed.get("city") or "").strip()
    h_state = (parsed.get("state") or "").strip()
    h_country = (parsed.get("country") or "").strip()

    # If the title already has city + country, that's the canonical answer.
    if h_city and h_country:
        return ", ".join(p for p in (h_city, h_state, h_country) if p)

    # Otherwise look for an experience location to fill gaps. Prefer the
    # passed-in current_position; fall back to the first Present experience
    # with a location string.
    exp_loc = ""
    if current_position:
        exp_loc = (current_position.get("location") or "").strip()
    if not exp_loc:
        for exp in apify_data.get("experience", []) or []:
            end = exp.get("endDate") or {}
            if end.get("text") != "Present":
                continue
            cand = (exp.get("location") or "").strip()
            if cand:
                exp_loc = cand
                break

    if not exp_loc:
        # No enrichment available — return whatever the title has (may be "")
        return ", ".join(p for p in (h_city, h_state, h_country) if p)

    exp_parts = [p.strip() for p in exp_loc.split(",") if p.strip()]
    if not exp_parts:
        return ", ".join(p for p in (h_city, h_state, h_country) if p)

    if len(exp_parts) == 1:
        e_city, e_state, e_country = "", "", exp_parts[0]
    elif len(exp_parts) == 2:
        e_city, e_state, e_country = exp_parts[0], "", exp_parts[1]
    else:
        e_city = exp_parts[0]
        e_state = exp_parts[-2]
        e_country = exp_parts[-1]

    # Country conflict between title and experience → trust the title.
    if h_country and e_country and h_country.strip().lower() != e_country.strip().lower():
        return ", ".join(p for p in (h_city, h_state, h_country) if p)

    merged_city = h_city or e_city
    merged_state = h_state or e_state
    merged_country = h_country or e_country
    return ", ".join(p for p in (merged_city, merged_state, merged_country) if p)


def _find_role_in_experience(apify_data: dict, company_name: str) -> Optional[dict]:
    """Find a current role in the experience array matching the company."""
    if not company_name:
        return None
    company_clean = company_name.strip().lower()
    for exp in apify_data.get("experience", []):
        end = exp.get("endDate") or {}
        if end.get("text") != "Present":
            continue
        exp_company = (exp.get("companyName") or "").strip()
        exp_role = (exp.get("position") or "").strip()
        if not exp_role or exp_role == "None":
            continue
        if exp_company.lower() == company_clean:
            return exp
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fulfillment_person_verification(
    lead: dict,
    apify_token: str = "",
    openrouter_key: str = "",
    scrapingdog_key: str = "",
) -> Tuple[bool, Optional[dict]]:
    """
    Fulfillment Stage 4 person verification.

    Pipeline:
      1. Q1 GSE → name check from title (cheap)
      2. ScrapingDog person scraper (if Q1 fails) → name + location
      3. Apify → company name, company URL, role,
         and location (if Q1 path — SD path already checked location)

    Returns: (passed, rejection_reason)
    """
    token = apify_token or APIFY_API_TOKEN
    or_key = openrouter_key or OPENROUTER_KEY
    sd_key = scrapingdog_key or SCRAPINGDOG_API_KEY

    linkedin_url = lead.get("linkedin", "")
    full_name = lead.get("full_name", "")
    lead_company = lead.get("company", "") or lead.get("business", "")

    slug = _get_linkedin_slug(linkedin_url)
    if not slug:
        return False, _rejection(
            "fulfillment_person_no_linkedin", "No LinkedIn URL on lead", ["linkedin"],
        )

    if not token:
        return False, _rejection(
            "fulfillment_person_no_token", "APIFY_API_TOKEN not configured", [],
        )

    # ==================================================================
    # Step 1: Q1 GSE search — cheap profile existence + name check
    # ==================================================================
    print(f"   🔍 Step 1: Q1 GSE search for {slug}")
    q1_results = await _q1_gse_search(slug, sd_key) if sd_key else []

    # Find result matching our slug
    url_matched = None
    for r in q1_results:
        if _get_linkedin_slug(r.get("link", "")) == slug:
            url_matched = r
            break

    name_verified = False
    li_location = ""

    if url_matched:
        # Name check from title (previous Stage 4 logic)
        title = url_matched.get("title", "")
        if _check_name_in_title(full_name, title):
            name_verified = True
            print(f"   ✅ Step 1: Name found in Q1 title: '{title[:60]}'")
        else:
            print(f"   ❌ Step 1: Name '{full_name}' not in title: '{title[:60]}'")
            return False, _rejection(
                "fulfillment_person_name_mismatch",
                f"Name '{full_name}' not found in LinkedIn title '{title}'",
                ["full_name"],
            )
    else:
        print(f"   ⚠️ Step 1: Slug not found in Q1 results")

    # ==================================================================
    # Step 2: ScrapingDog person scraper (if Q1 didn't find slug)
    # ==================================================================
    sd_profile = None
    if not name_verified:
        print(f"   🔍 Step 2: ScrapingDog person scraper for {slug}")
        sd_profile = await _sd_person_scrape(slug, sd_key) if sd_key else None

        if not sd_profile:
            print(f"   ❌ Step 2: Profile not found")
            return False, _rejection(
                "fulfillment_person_profile_not_found",
                "Profile not found via Q1 GSE or ScrapingDog scraper",
                ["linkedin"],
            )

        # Name check from ScrapingDog fullName
        sd_fullname = (sd_profile.get("fullName") or "").strip()
        if _check_name_fullname(full_name, sd_fullname):
            name_verified = True
            print(f"   ✅ Step 2: Name match: SD='{sd_fullname}' vs '{full_name}'")
        else:
            print(f"   ❌ Step 2: Name mismatch: SD='{sd_fullname}' vs '{full_name}'")
            return False, _rejection(
                "fulfillment_person_name_mismatch",
                f"Name mismatch: ScrapingDog '{sd_fullname}' vs '{full_name}'",
                ["full_name"],
            )

    # ==================================================================
    # Step 5: Apify — role + company URL + location (always)
    # ==================================================================
    print(f"   🔍 Step 5: Apify fetch for verification")
    apify_data = await _fetch_apify_profile(linkedin_url, token)
    if not apify_data:
        return False, _rejection(
            "fulfillment_person_fetch_failed",
            "Failed to fetch LinkedIn profile from Apify",
            ["linkedin"],
        )

    # Find current position
    pos = _get_current_position(apify_data, lead_company)
    if not pos:
        return False, _rejection(
            "fulfillment_person_no_current_position",
            "No current position found in Apify data",
            ["role"],
        )

    actual_role = (pos.get("position") or "").strip()
    actual_company = (pos.get("companyName") or "").strip()
    actual_company_url = (pos.get("companyLinkedinUrl") or "").strip()

    # --- Company URL slug match (always from Apify) ---
    lead_company_linkedin = lead.get("company_linkedin", "")
    apify_slug = _extract_company_slug(actual_company_url)
    lead_slug = _extract_company_slug(lead_company_linkedin)

    if apify_slug and not apify_slug.isdigit():
        # Real slug — must match miner's slug
        if apify_slug != lead_slug:
            print(f"   ❌ Company URL mismatch: Apify='{apify_slug}' vs Lead='{lead_slug}'")
            return False, _rejection(
                "fulfillment_person_company_url_mismatch",
                f"Company LinkedIn URL mismatch: '{apify_slug}' vs '{lead_slug}'",
                ["company_linkedin"],
            )
        print(f"   ✅ Company URL match: {apify_slug}")
    elif apify_slug and apify_slug.isdigit():
        # Numeric ID — compare against ScrapingDog's linkedin_internal_id
        sd_internal_id = lead.get("_sd_linkedin_internal_id", "")
        if sd_internal_id and apify_slug != sd_internal_id:
            print(f"   ❌ Company ID mismatch: Apify='{apify_slug}' vs SD='{sd_internal_id}'")
            return False, _rejection(
                "fulfillment_person_company_id_mismatch",
                f"Company LinkedIn ID mismatch: Apify '{apify_slug}' vs ScrapingDog '{sd_internal_id}'",
                ["company_linkedin"],
            )
        elif not sd_internal_id:
            print(f"   ❌ Cannot verify numeric company ID '{apify_slug}' — no SD internal ID")
            return False, _rejection(
                "fulfillment_person_company_id_unverifiable",
                f"Apify returned numeric company ID '{apify_slug}' but no ScrapingDog internal ID to verify against",
                ["company_linkedin"],
            )
        print(f"   ✅ Company ID match: {apify_slug}")
    else:
        # No /company/ URL or numeric ID from Apify. This commonly happens for
        # small businesses where the LinkedIn profile lists the company as
        # plain text without linking to a /company/ page — Apify substitutes
        # a "linkedin.com/search/results/all/?keywords=..." URL.  Apify still
        # returns the company NAME in that case, so let the name-match step
        # below decide.  Do not reject yet.
        print(f"   ⚠️ No /company/ URL from Apify — falling through to name match")

    # --- Company name match (always from Apify, normalized) ---
    if _normalize_company(actual_company) != _normalize_company(lead_company):
        print(f"   ❌ Company name mismatch: Apify='{actual_company}' vs Lead='{lead_company}'")
        return False, _rejection(
            "fulfillment_person_company_name_mismatch",
            f"Company name mismatch: '{actual_company}' vs '{lead_company}'",
            ["company"],
        )
    print(f"   ✅ Company name match: {actual_company}")

    # --- Location match (always from Apify, title-primary + experience enrichment) ---
    m_city = lead.get("city", "")
    m_state = lead.get("state", "")
    m_country = lead.get("country", "")

    if m_city or m_state or m_country:
        apify_loc = _best_li_location(apify_data, pos, lead_company)
        if not apify_loc:
            print(f"   ❌ No location on Apify profile")
            return False, _rejection(
                "fulfillment_person_no_location",
                "No location found on LinkedIn profile (headline empty and no current-experience location)",
                ["city", "state", "country"],
            )
        print(f"   🔍 Location check — miner='{m_city}, {m_state}, {m_country}' vs LinkedIn='{apify_loc}'")
        loc_match = await _llm_location_check(m_city, m_state, m_country, apify_loc, or_key)
        if not loc_match:
            return False, _rejection(
                "fulfillment_person_location_mismatch",
                f"Location mismatch: miner '{m_city}, {m_state}, {m_country}' vs LinkedIn '{apify_loc}'",
                ["city", "state", "country"],
            )

    # --- Role match (exact first, LLM fallback) ---
    lead_role = lead.get("role", "")
    if not actual_role:
        print(f"   ❌ Role empty in Apify")
        return False, _rejection(
            "fulfillment_person_role_empty",
            "No role found in Apify current position",
            ["role"],
        )
    role_match = actual_role.lower().strip() == lead_role.lower().strip()
    role_for_llm = actual_role  # Apify role; may be replaced by translation below
    if not role_match and is_configured():
        # Apify returns LinkedIn role in the profile's locale ("Diretora de
        # Tecnologia" for a Brazilian profile).  Ask the gateway to translate
        # to English; on success, retry exact match before paying for the LLM.
        translated = await translate_to_english(actual_role)
        if translated and translated.lower().strip() != actual_role.lower().strip():
            print(f"   🌐 Role translated via DeepL: '{actual_role}' → '{translated}'")
            role_for_llm = translated
            role_match = translated.lower().strip() == lead_role.lower().strip()
    if not role_match:
        # LLM fallback for formatting differences (e.g. "Sr Eng" vs "Senior Engineer")
        print(f"   🔍 Role exact mismatch, trying LLM: '{role_for_llm}' vs '{lead_role}'")
        role_match = await _llm_role_match(role_for_llm, lead_role, actual_company, or_key)
    if not role_match:
        print(f"   ❌ Role mismatch: Apify='{actual_role}' vs Lead='{lead_role}'")
        return False, _rejection(
            "fulfillment_person_role_mismatch",
            f"Role mismatch: '{actual_role}' vs '{lead_role}'",
            ["role"],
        )
    print(f"   ✅ Role match: {actual_role}")

    # ---- Populate lead dict for Stage 5 downstream ----
    lead["role_verified"] = True
    lead["role_method"] = "apify"
    lead["stage4_extracted_role"] = actual_role
    lead["location_verified"] = True
    lead["location_method"] = "sd_scraper" if sd_profile else "apify"
    parsed = apify_data.get("location", {}).get("parsed", {})
    lead["stage4_extracted_location"] = (
        f"{parsed.get('city', '')}, {parsed.get('state', '')}, {parsed.get('country', '')}"
    )
    lead["gse_search_count"] = len(q1_results)
    lead["llm_confidence"] = "apify"
    # Stash the full Apify profile so Tier 2c (attribute verification) can
    # build a contact-side prompt without re-fetching.  Carries headline,
    # summary, experience[], education[], currentPosition[], location, etc.
    lead["_apify_data"] = apify_data

    print(f"   ✅ Stage 4 PASSED: {full_name} @ {actual_company} as {actual_role}")
    return True, None
