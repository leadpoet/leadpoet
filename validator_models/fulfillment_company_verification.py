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

_SD_LINKEDIN_URL = "https://api.scrapingdog.com/linkedin"
_SD_TIMEOUT_S = 30
_MAX_RETRIES = 2

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCATION_MODEL = "google/gemini-2.5-flash-lite"

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fulfillment_company_verification(
    lead: dict,
    scrapingdog_key: str = "",
    openrouter_key: str = "",
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
        return False, _rejection(
            "fulfillment_company_fetch_failed",
            f"Failed to fetch company data from ScrapingDog LinkedIn after {_MAX_RETRIES + 1} attempts",
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
            website_content = await _scrape_website_content(lead_website)
            extracted_content = website_content.get("combined_description", "")

    if not extracted_content:
        print(f"   ❌ No description available for classification")
        return False, _rejection(
            "fulfillment_company_no_description",
            "No description available (LinkedIn about, tagline, specialties, and website all empty)",
            ["description"],
        )

    # Run 3-stage classification pipeline
    claimed_description = lead.get("description", "")
    claimed_industry = lead.get("industry", "")
    claimed_sub_industry = lead.get("sub_industry", "")
    sd_industry = sd_data.get("industry", "")

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

            # Reject if miner's claimed pair not in top 3
            if claimed_industry and claimed_sub_industry:
                claimed_pair = (claimed_industry.lower(), claimed_sub_industry.lower())
                top3_pairs = [(c["industry"].lower(), c["sub_industry"].lower()) for c in classifications[:3]]
                if claimed_pair not in top3_pairs:
                    print(f"   ❌ Miner claimed ({claimed_industry}, {claimed_sub_industry}) not in top 3: {top3_pairs}")
                    return False, _rejection(
                        "fulfillment_company_industry_mismatch",
                        f"Claimed industry/sub_industry not in top 3 classifications",
                        ["industry", "sub_industry"],
                    )

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
