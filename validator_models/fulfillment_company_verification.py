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
    classify_company_industry,
    _scrape_website_content,
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
_ICP_FIT_MODEL = "google/gemini-2.5-flash"

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

_ICP_BUYER_FIT_PROMPT = (
    'The BUYER is selling: "{product_service}"\n\n'
    'The BUYER\'s ideal customer profile:\n'
    '"{icp_prompt}"\n\n'
    'Target Industry: "{target_industry}"\n'
    'Target Sub-Industry: "{target_sub_industry}"\n\n'
    'Company being evaluated:\n'
    'Name: "{company_name}"\n'
    'Description: "{description}"\n'
    'LinkedIn Industry: "{linkedin_industry}"\n\n'
    'Does this company match the BUYER\'s ideal customer profile?\n'
    'Consider:\n'
    '- Is this company a potential CUSTOMER for what the buyer sells?\n'
    '- Or is it a PROVIDER of the same service as the buyer (competitor/peer)?\n'
    '- A competitor that sells the same product/service as the buyer is NOT a match.\n\n'
    'Return JSON only: {{"match": true/false, "reason": "one sentence explanation"}}'
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
) -> bool:
    """Compare HQ location. US: rule-based first, LLM fallback. Non-US: LLM only.

    LinkedIn page is the source of truth. If LinkedIn has no HQ location,
    reject (return False). Otherwise check if miner's claimed location
    matches what LinkedIn shows.
    """
    sd_hq = sd_data.get("headquarters", "")
    sd_address = ""
    for loc in sd_data.get("locations", []):
        if loc.get("is_hq"):
            sd_address = loc.get("office_address_line_2", "")
            break

    # No HQ location on LinkedIn page → invalid
    if not sd_hq and not sd_address:
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


async def _llm_icp_buyer_fit(
    company_name: str,
    extracted_description: str,
    linkedin_industry: str,
    icp_prompt: str,
    icp_product_service: str,
    icp_industry: list,
    icp_sub_industry: list,
    openrouter_key: str,
) -> Tuple[bool, str]:
    """Check if company matches the ICP buyer profile.

    Returns (matches, reason).  On LLM failure returns (False, reason)
    so that unverified leads never slip through.
    """
    # Escape curly braces in free-text fields to prevent .format() errors
    # (client ICP prompt or LinkedIn descriptions may contain { or })
    safe_icp = icp_prompt[:3000].replace("{", "{{").replace("}", "}}")
    safe_ps = (icp_product_service or "(not specified)").replace("{", "{{").replace("}", "}}")
    safe_desc = extracted_description[:1500].replace("{", "{{").replace("}", "}}")
    safe_name = company_name.replace("{", "{{").replace("}", "}}")
    safe_ind = (linkedin_industry or "unknown").replace("{", "{{").replace("}", "}}")
    target_ind_str = ", ".join(icp_industry) if icp_industry else "any"
    target_sub_str = ", ".join(icp_sub_industry) if icp_sub_industry else "any"
    safe_target_ind = target_ind_str.replace("{", "{{").replace("}", "}}")
    safe_target_sub = target_sub_str.replace("{", "{{").replace("}", "}}")

    prompt = _ICP_BUYER_FIT_PROMPT.format(
        product_service=safe_ps,
        icp_prompt=safe_icp,
        target_industry=safe_target_ind,
        target_sub_industry=safe_target_sub,
        company_name=safe_name,
        description=safe_desc,
        linkedin_industry=safe_ind,
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
                        "model": _ICP_FIT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        print(f"   ⚠️ ICP buyer-fit LLM rate limited, retrying ({attempt + 1}/3)...")
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            print(f"   ⚠️ ICP buyer-fit LLM HTTP {resp.status}, retrying ({attempt + 1}/3)...")
                            await asyncio.sleep(1)
                            continue
                        print(f"   ❌ ICP buyer-fit LLM HTTP {resp.status} after 3 attempts")
                        return False, "icp_buyer_fit_check_failed"
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    match = re.search(r"\{[^}]+\}", content)
                    if match:
                        result = json.loads(match.group())
                        is_match = bool(result.get("match", False))
                        reason = str(result.get("reason", ""))
                        return is_match, reason
                    if attempt < 2:
                        print(f"   ⚠️ ICP buyer-fit LLM response not valid JSON, retrying ({attempt + 1}/3)...")
                        continue
                    print(f"   ❌ ICP buyer-fit LLM returned unparseable response after 3 attempts")
                    return False, "icp_buyer_fit_check_failed"
        except Exception as e:
            if attempt < 2:
                print(f"   ⚠️ ICP buyer-fit LLM error: {e}, retrying ({attempt + 1}/3)...")
                await asyncio.sleep(1)
                continue
            print(f"   ❌ ICP buyer-fit LLM error after 3 attempts: {e}")
            return False, "icp_buyer_fit_check_failed"
    return False, "icp_buyer_fit_check_failed"


def _pick_best_industry(
    classifications: list,
    icp_industry: list,
    icp_sub_industry: list,
) -> Tuple[str, str]:
    """Pick the best industry/sub_industry from top 3 classifications.

    Priority:
      1. Pair matching both ICP industry AND sub_industry
      2. Pair matching ICP industry (any sub_industry)
      3. Classification match 1 (highest confidence)

    ``icp_industry`` and ``icp_sub_industry`` are lists of strings.
    """
    if not classifications:
        return "", ""

    icp_inds = {i.strip().lower() for i in (icp_industry or []) if i.strip()}
    icp_subs = {s.strip().lower() for s in (icp_sub_industry or []) if s.strip()}

    # Priority 1: exact match on both industry + sub_industry
    for c in classifications:
        ind = c.get("industry", "")
        sub = c.get("sub_industry", "")
        if ind and sub and ind.lower() in icp_inds and sub.lower() in icp_subs:
            return ind, sub

    # Priority 2: match on industry only
    for c in classifications:
        ind = c.get("industry", "")
        sub = c.get("sub_industry", "")
        if ind and ind.lower() in icp_inds:
            return ind, sub

    # Priority 3: highest confidence (first in list)
    return classifications[0].get("industry", ""), classifications[0].get("sub_industry", "")


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
    hq_match = await _llm_hq_match(lead, sd_data, or_key)
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

    # ---- 5. Description + Industry classification ----
    # Build description content from ScrapingDog data
    sd_about = sd_data.get("about", "")
    extracted_content = sd_about

    if not extracted_content:
        # Fallback: combine tagline + specialties + industry
        parts = []
        if sd_data.get("tagline"):
            parts.append(sd_data["tagline"])
        if sd_data.get("specialties"):
            parts.append(f"Specialties: {sd_data['specialties']}")
        if sd_data.get("industry"):
            parts.append(f"Industry: {sd_data['industry']}")
        extracted_content = ". ".join(parts)

    if not extracted_content:
        # Fallback: scrape company website
        print(f"   ⚠️ No LinkedIn description — falling back to website scrape")
        if lead_website:
            try:
                website_content = await _scrape_website_content(lead_website)
                extracted_content = website_content.get("combined_description", "")
            except Exception as e:
                print(f"   ⚠️ Website scrape failed: {e}")

    if not extracted_content:
        print(f"   ❌ No description available for classification")
        return False, _rejection(
            "fulfillment_company_no_description",
            "No description available (LinkedIn about, tagline, specialties, and website all empty)",
            ["description"],
        )

    # ---- 5a. ICP Buyer-Fit Check ----
    # Before running the expensive classification pipeline, verify that the
    # company actually matches the ICP buyer profile.  Catches competitors
    # (companies that SELL the same service the client sells) and companies
    # that are simply the wrong type for the ICP.
    sd_industry = sd_data.get("industry", "")
    if icp_prompt or icp_product_service:
        buyer_fit, fit_reason = await _llm_icp_buyer_fit(
            company_name=company,
            extracted_description=extracted_content,
            linkedin_industry=sd_industry,
            icp_prompt=icp_prompt,
            icp_product_service=icp_product_service,
            icp_industry=icp_industry,
            icp_sub_industry=icp_sub_industry,
            openrouter_key=or_key,
        )
        if not buyer_fit:
            print(f"   ❌ ICP buyer-fit mismatch: {fit_reason}")
            return False, _rejection(
                "fulfillment_company_icp_mismatch",
                f"Company does not match ICP buyer profile: {fit_reason}",
                ["description", "industry"],
            )
        print(f"   ✅ ICP buyer-fit match: {fit_reason}")

    # ---- 5b. Industry classification ----
    # Run 3-stage classification pipeline
    claimed_description = lead.get("description", "")
    claimed_industry = lead.get("industry", "")
    claimed_sub_industry = lead.get("sub_industry", "")

    # Use claimed_description or extracted_content as miner_description
    miner_desc = claimed_description or extracted_content

    try:
        classifications, refined_description, classify_error = await classify_company_industry(
            miner_description=miner_desc,
            extracted_content=extracted_content,
            extracted_industry=sd_industry,
            company_name=company,
            miner_industry=claimed_industry,
            miner_sub_industry=claimed_sub_industry,
        )

        if classifications and len(classifications) >= 1:
            industry_top3 = {}
            sub_industry_top3 = {}
            for i, c in enumerate(classifications[:3], 1):
                industry_top3[f"industry_match{i}"] = c["industry"]
                sub_industry_top3[f"sub_industry_match{i}"] = c["sub_industry"]
            print(f"   ✅ Classification: {[(c['industry'], c['sub_industry']) for c in classifications[:3]]}")

            # Pick the best-fit industry pair from top 3 that aligns
            # with the ICP's target industry.  If the miner's claim matches,
            # great — otherwise override it with the corrected pair.
            best_ind, best_sub = _pick_best_industry(
                classifications[:3], icp_industry, icp_sub_industry,
            )
            if best_ind and best_sub:
                old_pair = (claimed_industry, claimed_sub_industry)
                new_pair = (best_ind, best_sub)
                if old_pair != new_pair:
                    print(f"   🔄 Industry corrected: {old_pair} → {new_pair}")
                else:
                    print(f"   ✅ Industry confirmed: {new_pair}")
                lead["industry"] = best_ind
                lead["sub_industry"] = best_sub

            # Store for company table
            lead["_insert_new_company"] = True
            lead["_company_refined_description"] = refined_description
            lead["_company_industry_top3"] = industry_top3
            lead["_company_sub_industry_top3"] = sub_industry_top3
            lead["_company_verified_employee_count"] = claimed_employee_count
        else:
            if classify_error == "stage1_invalid_description":
                print(f"   ❌ Miner description does not match LinkedIn content")
                return False, _rejection(
                    "fulfillment_company_description_invalid",
                    "Miner description does not match LinkedIn content (INVALID)",
                    ["description"],
                )
            print(f"   ❌ Classification failed: {classify_error}")
            return False, _rejection(
                "fulfillment_company_classification_failed",
                f"Classification failed: {classify_error}",
                ["industry", "sub_industry"],
            )
    except Exception as e:
        print(f"   ❌ Classification exception: {e}")
        return False, _rejection(
            "fulfillment_company_classification_failed",
            f"Classification failed: {e}",
            ["industry", "sub_industry"],
        )

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
