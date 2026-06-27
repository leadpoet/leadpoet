"""
Fulfillment Stage 5: Company verification using ScrapingDog LinkedIn company endpoint.

Replaces the GSE-based Stage 5 (check_stage5_unified) with a single ScrapingDog
LinkedIn company API call that returns structured company data.  Checks:
  1. Company name match (case-sensitive, trademark symbols stripped)
  2. Employee count match (LinkedIn range normalization)
  3. HQ location match (rule-based for US, LLM for all)
  4. Website domain match
  5. Description + Industry/sub-industry classification (3-stage embedding pipeline)
"""

import asyncio
import json
import os
import re
import unicodedata
from typing import Optional, Tuple
import aiohttp

from gateway.utils.geo_normalize import (
    normalize_country,
    normalize_city,
    normalize_state,
)

from validator_models.stage5_verification import (
    _validate_name_match,
    _validate_size_match,
    _normalize_domain,
    _load_taxonomy_embeddings,
)

SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

_SD_LINKEDIN_URL = "https://api.scrapingdog.com/linkedin"
_SD_GOOGLE_URL = "https://api.scrapingdog.com/google"
_SD_TIMEOUT_S = 30
_MAX_RETRIES = 2

_APIFY_COMPANY_ACTOR = "harvestapi~linkedin-company"
_APIFY_COMPANY_ENDPOINT = (
    f"https://api.apify.com/v2/acts/{_APIFY_COMPANY_ACTOR}/run-sync-get-dataset-items"
)
_APIFY_TIMEOUT_S = 120

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCATION_MODEL = "google/gemini-2.5-flash-lite"
_VALIDATE_MODEL = "google/gemini-2.5-flash"

_ENRICH_PROMPT = (
    'What does "{company_name}" ({website}) do?\n\n'
    'Known information:\n'
    'LinkedIn Description: "{linkedin_description}"\n'
    'LinkedIn Industry: "{linkedin_industry}"\n\n'
    'Search the web for additional details about their products, services, '
    'and target market. Provide a 2-3 sentence description IN ENGLISH '
    '(translate any source content from Spanish, French, Portuguese, '
    'German, or any other language into clear professional English).'
)

_VALIDATE_MATCH_PROMPT = (
    'Company: "{company_name}"\n'
    'Description: "{enriched_description}"\n'
    'LinkedIn Industry: "{linkedin_industry}"\n'
    'Miner Description: "{miner_description}"\n\n'
    'ICP Target: {icp_target}\n\n'
    '1. Does the miner description match what the company actually does? '
    '(same type of business, ignore exact wording)\n'
    '2. Does the company match any of the target entries above?\n'
    '   a. FIRST check the LinkedIn industry field. If it directly matches '
    'a target industry, industry_match=true.\n'
    '   b. ONLY if the LinkedIn industry does not match, use the Description '
    'above to determine the company\'s ACTUAL core business. Match the target '
    'only if the company\'s PRIMARY revenue-generating activity clearly falls '
    'in the target industry. Be strict: adjacent, tangential, or "serves '
    'customers in" alignment does NOT qualify — the company itself must '
    'operate in that industry as its core business.\n'
    '   c. If neither the LinkedIn industry nor the description clearly '
    'places the company in the target industry, industry_match=false.\n'
    '{sub_industry_instruction}'
    '\nIf industry_match is true, return the matched industry and sub-industry.\n\n'
    'Return JSON only: {{"description_valid": true/false, "industry_match": true/false, '
    '"matched_industry": "", "matched_sub_industry": "", "reason": "one sentence"}}'
)

_SUB_INDUSTRY_PICK_PROMPT = (
    'Company: "{company_name}"\n'
    'Description: "{enriched_description}"\n'
    'Industry: "{matched_industry}"\n\n'
    'Pick the most appropriate sub-industry from this list:\n'
    '{sub_industry_options}\n\n'
    'Return JSON only: {{"sub_industry": "the picked sub-industry"}}'
)

_HQ_LOCATION_PROMPT = (
    'Miner claims company HQ: city="{m_city}", state="{m_state}", country="{m_country}"\n'
    'LinkedIn shows HQ: "{sd_hq}"\n'
    'LinkedIn HQ address: "{sd_hq_address}"\n\n'
    'Step 1: Is the miner\'s claimed location a real, valid place? '
    'If city+state+country combination does not exist, return {{"valid": false, "match": false}}.\n'
    'Step 2: If valid, does it match the LinkedIn HQ location?\n'
    '- If LinkedIn shows a city, the miner must also provide a city.\n'
    '- For non-US companies, state is not required to match — only city and country.\n'
    'Return JSON only: {{"valid": true/false, "match": true/false}}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rejection(check_name: str, message: str, failed_fields: list) -> dict:
    return {
        "stage": "Stage 5: Fulfillment Company Verification",
        "check_name": check_name,
        "message": message,
        "failed_fields": failed_fields,
    }


def _extract_company_slug(url: str) -> str:
    """Extract company slug from LinkedIn URL."""
    if not url:
        return ""
    match = re.search(r'linkedin\.com/company/([^/?#]+)', url.lower())
    return match.group(1) if match else ""




# ---------------------------------------------------------------------------
# ScrapingDog LinkedIn company fetch
# ---------------------------------------------------------------------------

async def _fetch_sd_company(slug: str, api_key: str) -> Optional[dict]:
    """Fetch company data from ScrapingDog LinkedIn endpoint with retry."""
    params = {
        "api_key": api_key,
        "type": "company",
        "linkId": slug,
    }
    timeout = aiohttp.ClientTimeout(total=_SD_TIMEOUT_S)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(_SD_LINKEDIN_URL, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            return data[0]
                        if isinstance(data, dict):
                            return data
                        return None
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        print(f"   ⚠️ ScrapingDog rate limited, retrying ({attempt + 1}/{_MAX_RETRIES})...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        print(f"   ❌ ScrapingDog HTTP {resp.status}")
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return None
        except Exception as e:
            print(f"   ❌ ScrapingDog fetch error: {e}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return None
    return None


async def _gse_company_slug_exists(slug: str, sd_key: str) -> bool:
    """Check if a LinkedIn company slug exists via ScrapingDog Google Search.

    Searches ``site:linkedin.com/company/{slug}`` and returns True if any
    result URL contains the slug.  Used to confirm a company page is real
    before falling back to Apify when the direct SD fetch fails.
    """
    query = f"site:linkedin.com/company/{slug}"
    timeout = aiohttp.ClientTimeout(total=15)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(_SD_GOOGLE_URL, params={
                    "api_key": sd_key, "query": query, "results": 5,
                }) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("organic_results", [])
                        for r in results:
                            link = (r.get("link") or "").lower()
                            if f"/company/{slug}" in link:
                                return True
                        return False
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    else:
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return False
        except Exception:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return False
    return False


async def _fetch_apify_company(slug: str, api_token: str) -> Optional[dict]:
    """Fetch company data from Apify LinkedIn company scraper.

    Returns a dict with fields mapped to match the ScrapingDog response
    format so downstream code doesn't need changes.
    """
    company_url = f"https://www.linkedin.com/company/{slug}"
    payload = {
        "companies": [company_url],
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
                    _APIFY_COMPANY_ENDPOINT, json=payload, headers=headers,
                ) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            raw = data[0]
                            # Map Apify fields to ScrapingDog format
                            return _map_apify_to_sd_format(raw, slug)
                        return None
                    elif resp.status == 429 and attempt < _MAX_RETRIES:
                        print(f"   ⚠️ Apify company rate limited, retrying ({attempt + 1}/{_MAX_RETRIES})...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        print(f"   ❌ Apify company HTTP {resp.status}")
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        return None
        except Exception as e:
            print(f"   ❌ Apify company fetch error: {e}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(1)
                continue
            return None
    return None


def _map_apify_to_sd_format(apify_data: dict, slug: str) -> dict:
    """Map Apify company scraper response to ScrapingDog field names.

    Apify (harvestapi~linkedin-company) fields:
      name, description, tagline, website, employeeCount,
      employeeCountRange, locations, industries, specialities,
      id, universalName, phone

    ScrapingDog fields used downstream:
      company_name, company_size, headquarters, website, about,
      tagline, specialties, industry, locations, linkedin_internal_id,
      universal_name_id
    """
    # Find HQ from locations array
    hq_str = ""
    sd_locations = []
    for loc in apify_data.get("locations", []):
        parsed = loc.get("parsed", {})
        is_hq = loc.get("headquarter", False)
        address_line = parsed.get("text", "")
        if not address_line:
            parts = [loc.get("city", ""), loc.get("geographicArea", ""), loc.get("country", "")]
            address_line = ", ".join(p for p in parts if p)
        sd_locations.append({
            "is_hq": is_hq,
            "office_address_line_2": address_line,
        })
        if is_hq:
            hq_str = address_line

    # If no location marked as HQ, use the first one
    if not hq_str and sd_locations:
        hq_str = sd_locations[0].get("office_address_line_2", "")

    # Employee count: Apify returns int (234) and range dict ({start:201, end:500})
    # SD returns a string like "201-500 employees"
    emp_range = apify_data.get("employeeCountRange", {})
    if emp_range and emp_range.get("start"):
        end = emp_range.get("end")
        if end:
            company_size = f"{emp_range['start']}-{end}"
        else:
            company_size = f"{emp_range['start']}+"
    elif apify_data.get("employeeCount"):
        company_size = str(apify_data["employeeCount"])
    else:
        company_size = ""

    # Industry: Apify returns list of dicts [{id, name, urn, title}]
    # SD returns a single string
    industries = apify_data.get("industries", [])
    industry_str = industries[0].get("name", "") if industries else ""

    # Specialities: Apify returns list of strings
    specialities = apify_data.get("specialities", [])
    specialties_str = ", ".join(specialities) if isinstance(specialities, list) else str(specialities or "")

    return {
        "company_name": apify_data.get("name", ""),
        "company_size": company_size,
        "headquarters": hq_str,
        "website": apify_data.get("website", ""),
        "about": apify_data.get("description", ""),
        "tagline": apify_data.get("tagline", ""),
        "specialties": specialties_str,
        "industry": industry_str,
        "locations": sd_locations,
        "linkedin_internal_id": str(apify_data.get("id", "")),
        "universal_name_id": apify_data.get("universalName", slug),
    }


# ---------------------------------------------------------------------------
# HQ location helpers
# ---------------------------------------------------------------------------

def _strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _parse_city_from_hq(hq: str) -> str:
    """Extract city from headquarters string (e.g. 'San Carlos, CA' → 'San Carlos')."""
    if not hq:
        return ""
    return hq.split(",")[0].strip()


def _parse_state_from_hq(hq: str) -> str:
    """Extract state from headquarters string (e.g. 'San Carlos, CA' → 'CA')."""
    if not hq:
        return ""
    parts = [p.strip() for p in hq.split(",")]
    return parts[1] if len(parts) > 1 else ""


def _parse_country_from_address(address: str) -> str:
    """Extract country code from office_address_line_2 (e.g. 'San Carlos, CA 94070, US' → 'US')."""
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",")]
    if parts:
        tokens = parts[-1].strip().split()
        if tokens:
            candidate = tokens[-1]
            if len(candidate) <= 3:
                return candidate
    return ""


def _us_hq_match(m_city: str, m_state: str, sd_hq: str) -> Optional[bool]:
    """Rule-based US HQ match. Returns True/False if resolved, None for LLM."""
    sd_city = normalize_city(_parse_city_from_hq(sd_hq), "United States").lower()
    sd_state = normalize_state(_parse_state_from_hq(sd_hq), "United States").lower()
    mc = normalize_city(m_city, "United States").lower() if m_city else ""
    ms = normalize_state(m_state, "United States").lower() if m_state else ""

    mc_s = _strip_diacritics(mc)
    sd_city_s = _strip_diacritics(sd_city)

    city_match = (mc == sd_city or mc_s == sd_city_s) if mc and sd_city else False
    state_match = (ms == sd_state) if ms and sd_state else False

    if mc and ms:
        if city_match and state_match:
            return True
        return None  # LLM fallback

    if mc and not ms:
        if city_match:
            return True
        return None

    if ms and not mc:
        if state_match:
            return True
        return None

    return None


# ---------------------------------------------------------------------------
# HQ location comparison — rule-based for US, LLM for all
# ---------------------------------------------------------------------------

async def _llm_hq_match(
    lead: dict,
    sd_data: dict,
    openrouter_key: str,
    location_required: bool = True,
) -> bool:
    """Compare HQ location. US: rule-based first, LLM fallback. Non-US: LLM only.

    LinkedIn page is the source of truth. If LinkedIn has no HQ location and
    the ICP imposes a location requirement, reject (return False) — the
    company's region can't be confirmed.  If the ICP has NO location
    requirement (``location_required=False``) and LinkedIn also exposes no
    HQ, pass: there is nothing the client asked for and nothing to verify.
    When LinkedIn does show an HQ, the miner's claimed location is still
    verified against it regardless of ``location_required`` (anti-fraud).
    """
    sd_hq = sd_data.get("headquarters", "")
    sd_address = ""
    for loc in sd_data.get("locations", []):
        if loc.get("is_hq"):
            sd_address = loc.get("office_address_line_2", "")
            break

    # No HQ location on LinkedIn page.
    if not sd_hq and not sd_address:
        if not location_required:
            print("   ✅ HQ location: not required by ICP and absent from LinkedIn — skipped")
            return True
        print("   ❌ No HQ location on LinkedIn page")
        return False

    m_city = lead.get("hq_city", "")
    m_state = lead.get("hq_state", "")
    m_country = lead.get("hq_country", "")

    # US: rule-based first
    m_country_norm = normalize_country(m_country).lower() if m_country else ""
    sd_country_code = _parse_country_from_address(sd_address)
    sd_country_norm = normalize_country(sd_country_code).lower() if sd_country_code else ""

    if m_country_norm == "united states" or sd_country_norm == "united states":
        rule = _us_hq_match(m_city, m_state, sd_hq)
        if rule is not None:
            print(f"   📍 HQ Location {'✅' if rule else '❌'} (rule-based US)")
            return rule

    if not openrouter_key:
        print("   ⚠️ No OpenRouter key for HQ location LLM")
        return False

    prompt = _HQ_LOCATION_PROMPT.format(
        m_city=m_city, m_state=m_state, m_country=m_country,
        sd_hq=sd_hq, sd_hq_address=sd_address,
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
                    print(f"   ⚠️ HQ Location LLM HTTP {resp.status}")
                    return False
                body = await resp.json()
                content = body["choices"][0]["message"]["content"]
                match = re.search(r"\{[^}]+\}", content)
                if match:
                    result = json.loads(match.group())
                    valid = bool(result.get("valid", True))
                    llm_match = bool(result.get("match", False))
                    if not valid:
                        print(f"   ❌ HQ Location: miner claimed invalid location (LLM)")
                        return False
                    print(f"   📍 HQ Location {'✅' if llm_match else '❌'} (LLM)")
                    return llm_match
                return False
    except Exception as e:
        print(f"   ⚠️ HQ Location LLM error: {e}")
        return False


async def _sonar_enrich(
    company_name: str,
    website: str,
    linkedin_description: str,
    linkedin_industry: str,
    openrouter_key: str,
) -> str:
    """Enrich company description using Perplexity Sonar web search.

    Returns enriched description string. On failure returns empty string.
    """
    safe_name = company_name.replace("{", "{{").replace("}", "}}")
    safe_website = (website or "unknown").replace("{", "{{").replace("}", "}}")
    safe_desc = (linkedin_description or "")[:1500].replace("{", "{{").replace("}", "}}")
    safe_ind = (linkedin_industry or "unknown").replace("{", "{{").replace("}", "}}")

    prompt = _ENRICH_PROMPT.format(
        company_name=safe_name,
        website=safe_website,
        linkedin_description=safe_desc,
        linkedin_industry=safe_ind,
    )

    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    _OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "perplexity/sonar",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        print(f"   ⚠️ Sonar enrich rate limited, retrying ({attempt + 1}/3)...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            print(f"   ⚠️ Sonar enrich HTTP {resp.status}, retrying ({attempt + 1}/3)...")
                            await asyncio.sleep(1)
                            continue
                        print(f"   ❌ Sonar enrich HTTP {resp.status} after 3 attempts")
                        return ""
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    return content.strip()
        except Exception as e:
            if attempt < 2:
                print(f"   ⚠️ Sonar enrich error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(1)
                continue
            print(f"   ❌ Sonar enrich error after 3 attempts: {e}")
            return ""
    return ""


async def _gemini_validate_and_match(
    company_name: str,
    enriched_description: str,
    linkedin_industry: str,
    miner_description: str,
    icp_industry: list,
    icp_sub_industry: list,
    openrouter_key: str,
) -> dict:
    """Validate miner description + match industry using Gemini.

    Returns dict with description_valid, industry_match, matched_industry,
    matched_sub_industry, reason. On failure returns None.
    """
    # Defense-in-depth: tolerate any caller that hands us a stringified list
    # or single string instead of the canonical List[str].
    from gateway.fulfillment.icp_checks import _coerce_industry_list
    icp_industry = _coerce_industry_list(icp_industry)
    icp_sub_industry = _coerce_industry_list(icp_sub_industry)

    safe_name = company_name.replace("{", "{{").replace("}", "}}")
    safe_enriched = enriched_description[:1500].replace("{", "{{").replace("}", "}}")
    safe_ind = (linkedin_industry or "unknown").replace("{", "{{").replace("}", "}}")
    safe_miner = miner_description[:500].replace("{", "{{").replace("}", "}}")

    # Format ICP target and sub-industry instruction
    if icp_sub_industry:
        pairs = [f"{i} / {s}" for i in icp_industry for s in icp_sub_industry]
        icp_target = "\n".join(pairs)
        sub_inst = (
            'If a sub-industry is specified (e.g., "Software / SaaS"), '
            'the company must match BOTH the industry AND the sub-industry.\n'
        )
    else:
        icp_target = "\n".join(icp_industry) if icp_industry else "any"
        sub_inst = (
            'No sub-industry is specified — pick the most appropriate '
            'sub-industry based on the company description.\n'
        )
    safe_target = icp_target.replace("{", "{{").replace("}", "}}")
    safe_sub_inst = sub_inst.replace("{", "{{").replace("}", "}}")

    prompt = _VALIDATE_MATCH_PROMPT.format(
        company_name=safe_name,
        enriched_description=safe_enriched,
        linkedin_industry=safe_ind,
        miner_description=safe_miner,
        icp_target=safe_target,
        sub_industry_instruction=safe_sub_inst,
    )

    for attempt in range(3):
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
                        "model": _VALIDATE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        print(f"   ⚠️ Gemini validate rate limited, retrying ({attempt + 1}/3)...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            print(f"   ⚠️ Gemini validate HTTP {resp.status}, retrying ({attempt + 1}/3)...")
                            await asyncio.sleep(1)
                            continue
                        print(f"   ❌ Gemini validate HTTP {resp.status} after 3 attempts")
                        return None
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    match = re.search(r"\{[^}]+\}", content)
                    if match:
                        return json.loads(match.group())
                    if attempt < 2:
                        print(f"   ⚠️ Gemini validate response not valid JSON, retrying ({attempt + 1}/3)...")
                        continue
                    print(f"   ❌ Gemini validate unparseable response after 3 attempts")
                    return None
        except Exception as e:
            if attempt < 2:
                print(f"   ⚠️ Gemini validate error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(1)
                continue
            print(f"   ❌ Gemini validate error after 3 attempts: {e}")
            return None
    return None


def _get_valid_sub_industries(industry: str) -> list:
    """Get valid sub-industries for a given industry from taxonomy."""
    taxonomy = _load_taxonomy_embeddings()
    if not taxonomy:
        return []
    # taxonomy['industries'] maps sub_industry → [list of industries]
    subs = []
    for sub, ind_list in taxonomy.get('industries', {}).items():
        if industry.lower() in [i.lower() for i in ind_list]:
            subs.append(sub)
    return sorted(subs)


async def _gemini_pick_sub_industry(
    company_name: str,
    enriched_description: str,
    matched_industry: str,
    sub_industry_options: list,
    openrouter_key: str,
) -> str:
    """Ask Gemini to pick the best sub-industry from taxonomy list."""
    safe_name = company_name.replace("{", "{{").replace("}", "}}")
    safe_enriched = enriched_description[:1500].replace("{", "{{").replace("}", "}}")
    safe_ind = matched_industry.replace("{", "{{").replace("}", "}}")
    options_str = "\n".join(f"- {s}" for s in sub_industry_options)

    prompt = _SUB_INDUSTRY_PICK_PROMPT.format(
        company_name=safe_name,
        enriched_description=safe_enriched,
        matched_industry=safe_ind,
        sub_industry_options=options_str,
    )

    for attempt in range(3):
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
                        "model": _VALIDATE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return ""
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    match = re.search(r"\{[^}]+\}", content)
                    if match:
                        result = json.loads(match.group())
                        return str(result.get("sub_industry", ""))
                    if attempt < 2:
                        continue
                    return ""
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            print(f"   ⚠️ Sub-industry pick error: {e}")
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fulfillment_company_verification(
    lead: dict,
    scrapingdog_key: str = "",
    openrouter_key: str = "",
    icp_prompt: str = "",
    icp_product_service: str = "",
    icp_industry: list = None,
    icp_sub_industry: list = None,
    location_required: bool = True,
) -> Tuple[bool, Optional[dict]]:
    """
    Fulfillment-specific Stage 5 company verification using ScrapingDog LinkedIn.

    1. Fetches company data from ScrapingDog LinkedIn endpoint
    2. Verifies: company name, employee count, HQ location, website domain
    3. Runs industry classification (3-stage embedding pipeline)
    4. Populates lead dict with verification results

    Returns: (passed, rejection_reason)
    """
    sd_key = scrapingdog_key or SCRAPINGDOG_API_KEY
    or_key = openrouter_key or OPENROUTER_KEY

    # Defense-in-depth: callers may hand us a stringified list ("['X', 'Y']")
    # for legacy/CSV-mediated paths.  Normalize once at the entry point.
    from gateway.fulfillment.icp_checks import _coerce_industry_list
    icp_industry = _coerce_industry_list(icp_industry)
    icp_sub_industry = _coerce_industry_list(icp_sub_industry)

    company_linkedin = lead.get("company_linkedin", "")
    slug = _extract_company_slug(company_linkedin)
    company = lead.get("company", "") or lead.get("business", "")

    if not slug:
        return False, _rejection(
            "fulfillment_company_no_slug",
            "Could not extract company LinkedIn slug",
            ["company_linkedin"],
        )

    if not sd_key:
        return False, _rejection(
            "fulfillment_company_no_key",
            "SCRAPINGDOG_API_KEY not configured",
            [],
        )

    # ---- Fetch company data ----
    print(f"   🔍 Fulfillment Stage 5: Fetching company '{company}' (slug: {slug})")
    sd_data = await _fetch_sd_company(slug, sd_key)
    if not sd_data:
        # SD LinkedIn fetch failed — verify slug exists via GSE before Apify fallback
        print(f"   ⚠️ ScrapingDog fetch failed, checking if slug exists via GSE...")
        slug_exists = await _gse_company_slug_exists(slug, sd_key)
        if not slug_exists:
            print(f"   ❌ GSE confirms company slug '{slug}' does not exist on LinkedIn")
            return False, _rejection(
                "fulfillment_company_not_found",
                f"Company LinkedIn page '{slug}' not found (SD fetch failed, GSE confirms no match)",
                ["company_linkedin"],
            )
        # Slug exists but SD couldn't fetch — try Apify
        apify_key = APIFY_API_TOKEN
        if not apify_key:
            print(f"   ❌ SD fetch failed, slug exists but no APIFY_API_TOKEN for fallback")
            return False, _rejection(
                "fulfillment_company_fetch_failed",
                f"ScrapingDog fetch failed and no Apify token configured for fallback",
                ["company_linkedin"],
            )
        print(f"   🔄 Slug exists on LinkedIn — falling back to Apify company scraper")
        sd_data = await _fetch_apify_company(slug, apify_key)
        if not sd_data:
            print(f"   ❌ Apify company fetch also failed for slug '{slug}'")
            return False, _rejection(
                "fulfillment_company_fetch_failed",
                f"Both ScrapingDog and Apify failed to fetch company data for '{slug}'",
                ["company_linkedin"],
            )

    # Store company IDs for person verification (Stage 4) to use
    lead["_sd_linkedin_internal_id"] = str(sd_data.get("linkedin_internal_id", ""))
    lead["_sd_universal_name_id"] = sd_data.get("universal_name_id", "")

    # ---- 1. Company name match ----
    sd_name = sd_data.get("company_name", "")
    name_match, name_reason = _validate_name_match(company, sd_name)
    if not name_match:
        print(f"   ❌ Company name mismatch: {name_reason}")
        return False, _rejection(
            "fulfillment_company_name_mismatch",
            name_reason,
            ["company_name"],
        )
    print(f"   ✅ Company name match: {sd_name}")

    # ---- 2. Employee count match ----
    claimed_employee_count = lead.get("employee_count", "")
    sd_size = sd_data.get("company_size", "")
    # Strip " employees" suffix if present
    sd_size_clean = re.sub(r'\s*employees?\s*$', '', sd_size, flags=re.IGNORECASE).strip()

    if not sd_size_clean:
        print(f"   ❌ No employee count from LinkedIn")
        return False, _rejection(
            "fulfillment_company_no_size",
            "LinkedIn company page has no employee count",
            ["employee_count"],
        )
    if not claimed_employee_count:
        print(f"   ❌ Miner did not submit employee count")
        return False, _rejection(
            "fulfillment_company_no_claimed_size",
            "Miner did not submit employee count",
            ["employee_count"],
        )
    size_match, size_reason = _validate_size_match(claimed_employee_count, sd_size_clean)
    if not size_match:
        print(f"   ❌ Employee count mismatch: claimed '{claimed_employee_count}' vs LinkedIn '{sd_size_clean}'")
        return False, _rejection(
            "fulfillment_company_size_mismatch",
            f"Size mismatch: claimed '{claimed_employee_count}' vs LinkedIn '{sd_size_clean}'",
            ["employee_count"],
        )
    print(f"   ✅ Employee count match: {sd_size_clean}")

    # ---- 3. HQ location match ----
    hq_match = await _llm_hq_match(lead, sd_data, or_key, location_required=location_required)
    if not hq_match:
        sd_hq = sd_data.get("headquarters", "")
        m_hq = f"{lead.get('hq_city', '')}, {lead.get('hq_state', '')}, {lead.get('hq_country', '')}"
        print(f"   ❌ HQ location mismatch: LinkedIn='{sd_hq}' vs Miner='{m_hq}'")
        return False, _rejection(
            "fulfillment_company_hq_mismatch",
            f"HQ location mismatch: LinkedIn '{sd_hq}' vs '{m_hq}'",
            ["hq_country", "hq_city"],
        )
    print(f"   ✅ HQ location match")

    # ---- 4. Website domain match ----
    sd_website = sd_data.get("website", "")
    lead_website = lead.get("website", "") or lead.get("company_website", "")
    if sd_website and lead_website:
        sd_domain = _normalize_domain(sd_website)
        lead_domain = _normalize_domain(lead_website)
        if sd_domain and lead_domain:
            if sd_domain != lead_domain:
                print(f"   ❌ Website domain mismatch: LinkedIn='{sd_domain}' vs Miner='{lead_domain}'")
                return False, _rejection(
                    "fulfillment_company_website_mismatch",
                    f"Website domain mismatch: '{sd_domain}' vs '{lead_domain}'",
                    ["website"],
                )
            print(f"   ✅ Website domain match: {sd_domain}")
        else:
            print(f"   ❌ Website domain could not be parsed")
            return False, _rejection(
                "fulfillment_company_website_parse_failed",
                f"Could not parse website domains: LinkedIn='{sd_website}', Miner='{lead_website}'",
                ["website"],
            )
    elif sd_website and not lead_website:
        print(f"   ❌ LinkedIn has website but miner didn't submit one")
        return False, _rejection(
            "fulfillment_company_no_website",
            f"LinkedIn has website '{sd_website}' but miner didn't provide company_website",
            ["website"],
        )
    elif not sd_website:
        print(f"   ❌ No website on LinkedIn company page")
        return False, _rejection(
            "fulfillment_company_no_linkedin_website",
            "LinkedIn company page has no website",
            ["website"],
        )

    # ---- Apify enrichment for sparse SD data ----
    # If SD returned data but is missing description/industry, try Apify
    # to fill the gaps before Sonar+Gemini step. Only fills empty fields,
    # never overrides SD data that already passed structural checks.
    _sd_about = sd_data.get("about", "")
    _sd_industry = sd_data.get("industry", "")
    _has_desc = _sd_about or sd_data.get("tagline") or sd_data.get("specialties")

    if not _has_desc or not _sd_industry:
        _apify_key = APIFY_API_TOKEN
        if _apify_key:
            print(f"   🔄 SD data sparse (desc={bool(_has_desc)}, industry={bool(_sd_industry)}), trying Apify enrichment...")
            _apify_data = await _fetch_apify_company(slug, _apify_key)
            if _apify_data:
                if not _has_desc and _apify_data.get("about"):
                    sd_data["about"] = _apify_data["about"]
                    print(f"   ✅ Apify filled description: {_apify_data['about'][:60]}...")
                if not _sd_industry and _apify_data.get("industry"):
                    sd_data["industry"] = _apify_data["industry"]
                    print(f"   ✅ Apify filled industry: {_apify_data['industry']}")
                if not sd_data.get("tagline") and _apify_data.get("tagline"):
                    sd_data["tagline"] = _apify_data["tagline"]
                if not sd_data.get("specialties") and _apify_data.get("specialties"):
                    sd_data["specialties"] = _apify_data["specialties"]
            else:
                print(f"   ⚠️ Apify enrichment also returned no data — continuing with sparse SD data")
        else:
            print(f"   ⚠️ SD data sparse but no APIFY_API_TOKEN — continuing with sparse SD data")

    # ---- 5. Description validation + Industry match ----
    # Two-step: Sonar enriches company description, Gemini validates
    # miner's description and checks industry match against ICP.
    sd_about = sd_data.get("about", "")
    sd_industry = sd_data.get("industry", "")
    lead_website = lead.get("website", "") or lead.get("company_website", "")
    claimed_description = lead.get("description", "")
    miner_desc = claimed_description or sd_about

    # Build LinkedIn description from available data
    linkedin_desc = sd_about
    if not linkedin_desc:
        parts = []
        if sd_data.get("tagline"):
            parts.append(sd_data["tagline"])
        if sd_data.get("specialties"):
            parts.append(f"Specialties: {sd_data['specialties']}")
        linkedin_desc = ". ".join(parts)

    # Step 1: Sonar enrich
    print(f"   🔍 Sonar enriching: {company}")
    enriched = await _sonar_enrich(
        company_name=company,
        website=lead_website,
        linkedin_description=linkedin_desc,
        linkedin_industry=sd_industry,
        openrouter_key=or_key,
    )
    if not enriched:
        print(f"   ❌ Sonar enrichment failed — no description available")
        return False, _rejection(
            "fulfillment_company_enrich_failed",
            "Could not enrich company description via web search",
            ["description"],
        )
    print(f"   ✅ Sonar enriched: {enriched[:80]}...")

    # Step 2: Gemini validate + industry match
    print(f"   🔍 Gemini validating description + industry match")
    result = await _gemini_validate_and_match(
        company_name=company,
        enriched_description=enriched,
        linkedin_industry=sd_industry,
        miner_description=miner_desc,
        icp_industry=icp_industry or [],
        icp_sub_industry=icp_sub_industry or [],
        openrouter_key=or_key,
    )
    if result is None:
        print(f"   ❌ Gemini validation failed")
        return False, _rejection(
            "fulfillment_company_validation_failed",
            "Company validation failed (LLM error)",
            ["description", "industry"],
        )

    desc_valid = bool(result.get("description_valid", False))
    ind_match = bool(result.get("industry_match", False))
    matched_ind = str(result.get("matched_industry", ""))
    matched_sub = str(result.get("matched_sub_industry", ""))
    reason = str(result.get("reason", ""))

    if not desc_valid:
        print(f"   ❌ Miner description does not match company: {reason}")
        return False, _rejection(
            "fulfillment_company_description_invalid",
            f"Miner description does not match company: {reason}",
            ["description"],
        )

    if not ind_match:
        print(f"   ❌ Industry mismatch: {reason}")
        return False, _rejection(
            "fulfillment_company_industry_classification_mismatch",
            f"Company verified as '{sd_industry}' industry — does not match target industry",
            ["industry", "sub_industry"],
        )

    # Step 3: Pick sub-industry from taxonomy (only if ICP has no sub-industry)
    if matched_ind and not icp_sub_industry:
        valid_subs = _get_valid_sub_industries(matched_ind)
        if valid_subs:
            print(f"   🔍 Picking sub-industry from {len(valid_subs)} taxonomy options for '{matched_ind}'")
            picked = await _gemini_pick_sub_industry(
                company_name=company,
                enriched_description=enriched,
                matched_industry=matched_ind,
                sub_industry_options=valid_subs,
                openrouter_key=or_key,
            )
            if picked:
                matched_sub = picked
                print(f"   ✅ Sub-industry picked: {picked}")

    # Update lead with matched industry
    if matched_ind:
        old_pair = (lead.get("industry", ""), lead.get("sub_industry", ""))
        new_pair = (matched_ind, matched_sub)
        if old_pair != new_pair:
            print(f"   🔄 Industry corrected: {old_pair} → {new_pair}")
        else:
            print(f"   ✅ Industry confirmed: {new_pair}")
        lead["industry"] = matched_ind
        lead["sub_industry"] = matched_sub

    # Store for company table
    lead["_insert_new_company"] = True
    lead["_company_refined_description"] = enriched
    lead["_company_industry_top3"] = {
        "industry_match1": matched_ind,
    }
    lead["_company_sub_industry_top3"] = {
        "sub_industry_match1": matched_sub,
    }
    lead["_company_verified_employee_count"] = claimed_employee_count

    print(f"   ✅ Industry match: {matched_ind}/{matched_sub}")

    # ---- Populate lead dict for downstream ----
    lead["stage5_name_match"] = True
    lead["stage5_size_match"] = True
    lead["stage5_hq_match"] = True
    lead["stage5_industry_match"] = True
    lead["stage5_extracted_name"] = sd_name
    lead["stage5_extracted_size"] = sd_size_clean
    lead["stage5_extracted_hq"] = sd_data.get("headquarters", "")
    lead["stage5_extracted_industry"] = sd_industry

    print(f"   ✅ Fulfillment Stage 5 PASSED: {company}")
    return True, None
