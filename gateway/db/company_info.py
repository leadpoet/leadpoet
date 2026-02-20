"""
Company Information Table Operations
=====================================

Handles read/write operations for company_information_table in Supabase.

Table Schema:
- company_name (Non Nullable) - Case-sensitive
- company_linkedin (Non Nullable) - Original URL from miner
- company_slug (Non Nullable, Unique) - Primary lookup key (extracted from URL)
- company_website (Non Nullable)
- company_description (Non Nullable) - Refined version, not miner claimed
- company_hq_country (Non Nullable)
- company_hq_state (Nullable) - UAE companies may not have state
- company_hq_city (Nullable)
- company_industry (Non Nullable) - JSON: {industry_match1, industry_match2, industry_match3}
- company_sub_industry (Non Nullable) - JSON: {sub_industry_match1, sub_industry_match2, sub_industry_match3}
- company_employee_count (Non Nullable)
- company_last_updated (Non Nullable)

Flow:
1. If company_slug exists in table → Gateway validates against stored data
2. If NOT exists → Full validation, then add to table with refined description
3. Employee count changes → Update table with new count + last_updated
"""

import json
import logging
import re
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from gateway.db.client import get_write_client

logger = logging.getLogger(__name__)

TABLE_NAME = "company_information_table"
_TRADEMARK_SYMBOLS_RE = re.compile(r'[®™©℠]+')
_SLUG_RE = re.compile(r'^(?:https?://)?(?:www\.)?(?:[a-z]{1,2}\.)?linkedin\.com/company/([^/?#]+)')


def _extract_slug(company_linkedin: str) -> str:
    """Extract company slug from any LinkedIn company URL format.

    Examples:
        https://www.linkedin.com/company/small-batch-standard → small-batch-standard
        linkedin.com/company/small-batch-standard/about → small-batch-standard
        https://tw.linkedin.com/company/small-batch-standard → small-batch-standard
    """
    if not company_linkedin:
        return ""
    m = _SLUG_RE.match(company_linkedin.strip().lower())
    return m.group(1) if m else ""


def get_company_by_linkedin(company_linkedin: str) -> Optional[Dict[str, Any]]:
    """
    Look up company by slug extracted from company_linkedin URL.

    Args:
        company_linkedin: Company LinkedIn URL (any format)

    Returns:
        Company record dict if found, None if not found
    """
    slug = _extract_slug(company_linkedin)
    if not slug:
        return None

    try:
        client = get_write_client()
        result = client.table(TABLE_NAME) \
            .select("*") \
            .eq("company_slug", slug) \
            .limit(1) \
            .execute()

        if result.data and len(result.data) > 0:
            logger.info(f"✅ Found company in table: {slug}")
            return result.data[0]

        return None

    except Exception as e:
        logger.error(f"❌ Error looking up company: {e}")
        return None


def validate_against_stored(
    stored: Dict[str, Any],
    claimed_name: str,
    claimed_hq_city: str,
    claimed_hq_state: str,
    claimed_hq_country: str,
    claimed_website: str,
    claimed_industry: str,
    claimed_sub_industry: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate miner-claimed data against stored company data.

    Validates:
    1. Company name (CASE-SENSITIVE exact match)
    2. HQ location (city/state/country)
    3. Website domain
    4. Industry/Sub-industry in top 3 matches

    Args:
        stored: Stored company record from database
        claimed_*: Miner-claimed values

    Returns:
        (True, None) if valid
        (False, "rejection_reason") if invalid
    """
    # 1. Company name - CASE SENSITIVE exact match (trademark symbols stripped from both sides)
    stored_name = stored.get("company_name", "")
    claimed_name_clean = ' '.join(_TRADEMARK_SYMBOLS_RE.sub('', claimed_name).split())
    stored_name_clean = ' '.join(_TRADEMARK_SYMBOLS_RE.sub('', stored_name).split())
    if claimed_name_clean != stored_name_clean:
        return False, f"company_name_mismatch: Claimed '{claimed_name}' but stored '{stored_name}' (case-sensitive)"

    # 2. HQ Location validation (EXACT FIELD MATCHING)
    # Miner must submit EXACTLY the same fields as stored - can't add or omit fields
    stored_hq_country = stored.get("company_hq_country", "")
    stored_hq_state = stored.get("company_hq_state", "") or ""
    stored_hq_city = stored.get("company_hq_city", "") or ""

    claimed_hq_city = claimed_hq_city.strip() if claimed_hq_city else ""
    claimed_hq_state = claimed_hq_state.strip() if claimed_hq_state else ""
    claimed_hq_country = claimed_hq_country.strip() if claimed_hq_country else ""

    # Country must always match
    if claimed_hq_country.lower() != stored_hq_country.lower():
        return False, f"hq_country_mismatch: Claimed '{claimed_hq_country}' but stored '{stored_hq_country}'"

    # State: exact field matching
    if stored_hq_state:
        # Stored has state - miner MUST provide state and it must match
        if not claimed_hq_state:
            return False, f"hq_state_missing: Stored has state '{stored_hq_state}' but miner did not provide state"
        if claimed_hq_state.lower() != stored_hq_state.lower():
            return False, f"hq_state_mismatch: Claimed '{claimed_hq_state}' but stored '{stored_hq_state}'"
    else:
        # Stored has NO state - miner MUST NOT provide state
        if claimed_hq_state:
            return False, f"hq_state_extra: Stored has no state but miner provided '{claimed_hq_state}'"

    # City: exact field matching
    if stored_hq_city:
        # Stored has city - miner MUST provide city and it must match
        if not claimed_hq_city:
            return False, f"hq_city_missing: Stored has city '{stored_hq_city}' but miner did not provide city"
        if claimed_hq_city.lower() != stored_hq_city.lower():
            return False, f"hq_city_mismatch: Claimed '{claimed_hq_city}' but stored '{stored_hq_city}'"
    else:
        # Stored has NO city - miner MUST NOT provide city
        if claimed_hq_city:
            return False, f"hq_city_extra: Stored has no city but miner provided '{claimed_hq_city}'"

    # 3. Website domain validation
    stored_website = stored.get("company_website", "")
    claimed_domain = _extract_domain(claimed_website)
    stored_domain = _extract_domain(stored_website)

    if claimed_domain != stored_domain:
        return False, f"website_mismatch: Claimed domain '{claimed_domain}' but stored '{stored_domain}'"

    # 4. Industry + Sub-industry must match as a PAIR (not separately)
    # Pair 1: industry_match1 + sub_industry_match1
    # Pair 2: industry_match2 + sub_industry_match2
    # Pair 3: industry_match3 + sub_industry_match3
    stored_industry = stored.get("company_industry", {})
    if isinstance(stored_industry, str):
        try:
            stored_industry = json.loads(stored_industry)
        except (json.JSONDecodeError, ValueError):
            stored_industry = {}

    stored_sub_industry = stored.get("company_sub_industry", {})
    if isinstance(stored_sub_industry, str):
        try:
            stored_sub_industry = json.loads(stored_sub_industry)
        except (json.JSONDecodeError, ValueError):
            stored_sub_industry = {}

    # Build valid pairs
    valid_pairs = []
    for i in range(1, 4):
        ind = stored_industry.get(f"industry_match{i}", "").lower()
        sub = stored_sub_industry.get(f"sub_industry_match{i}", "").lower()
        if ind and sub:
            valid_pairs.append((ind, sub))

    claimed_pair = (claimed_industry.lower(), claimed_sub_industry.lower())

    if claimed_pair not in valid_pairs:
        return False, f"industry_pair_mismatch: Claimed pair ('{claimed_industry}', '{claimed_sub_industry}') not in stored pairs: {valid_pairs}"

    return True, None


def _extract_domain(url: str) -> str:
    """Extract domain from URL (lowercase, no www prefix)."""
    if not url:
        return ""

    url = url.lower().strip()

    # Remove protocol
    if "://" in url:
        url = url.split("://", 1)[1]

    # Remove path
    if "/" in url:
        url = url.split("/", 1)[0]

    # Remove www prefix
    if url.startswith("www."):
        url = url[4:]

    return url


def insert_company(
    company_linkedin: str,
    company_name: str,
    company_website: str,
    company_description: str,
    company_hq_country: str,
    company_hq_state: Optional[str],
    company_hq_city: Optional[str],
    industry_top3: Dict[str, str],
    sub_industry_top3: Dict[str, str],
    company_employee_count: str
) -> bool:
    """
    Insert new company into table.

    Args:
        company_linkedin: Company LinkedIn URL (primary key)
        company_name: Company name (case-sensitive)
        company_website: Company website
        company_description: REFINED description (not miner claimed)
        company_hq_country: HQ country
        company_hq_state: HQ state (nullable)
        company_hq_city: HQ city (nullable)
        industry_top3: {industry_match1, industry_match2, industry_match3}
        sub_industry_top3: {sub_industry_match1, sub_industry_match2, sub_industry_match3}
        company_employee_count: Employee count range

    Returns:
        True if inserted successfully, False otherwise
    """
    slug = _extract_slug(company_linkedin)
    if not slug:
        logger.error(f"❌ Cannot extract slug from: {company_linkedin}")
        return False

    try:
        client = get_write_client()

        # Strip trademark symbols before storing (clean data in table)
        company_name = ' '.join(_TRADEMARK_SYMBOLS_RE.sub('', company_name).split())

        record = {
            "company_slug": slug,
            "company_linkedin": f"https://www.linkedin.com/company/{slug}",
            "company_name": company_name,
            "company_website": company_website,
            "company_description": company_description,
            "company_hq_country": company_hq_country,
            "company_hq_state": company_hq_state or None,
            "company_hq_city": company_hq_city or None,
            "company_industry": json.dumps(industry_top3),
            "company_sub_industry": json.dumps(sub_industry_top3),
            "company_employee_count": company_employee_count,
            "company_last_updated": datetime.utcnow().isoformat()
        }

        result = client.table(TABLE_NAME).upsert(
            record,
            on_conflict="company_slug"
        ).execute()

        if result.data:
            logger.info(f"✅ Inserted new company: {company_linkedin}")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error inserting company: {e}")
        return False


def update_employee_count(
    company_linkedin: str,
    new_employee_count: str,
    prev_employee_count: str
) -> bool:
    """
    Update company employee count and last_updated timestamp.

    Args:
        company_linkedin: Company LinkedIn URL
        new_employee_count: New employee count range
        prev_employee_count: Previous employee count (for logging)

    Returns:
        True if updated successfully, False otherwise
    """
    slug = _extract_slug(company_linkedin)
    if not slug:
        return False

    try:
        client = get_write_client()

        result = client.table(TABLE_NAME) \
            .update({
                "company_employee_count": new_employee_count,
                "company_last_updated": datetime.utcnow().isoformat()
            }) \
            .eq("company_slug", slug) \
            .execute()

        if result.data:
            logger.info(f"✅ Updated employee count: {slug} ({prev_employee_count} → {new_employee_count})")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error updating employee count: {e}")
        return False


def check_employee_count_changed(stored: Dict[str, Any], claimed_count: str) -> Tuple[bool, str]:
    """
    Check if employee count has changed from stored value.

    Args:
        stored: Stored company record
        claimed_count: Miner-claimed employee count

    Returns:
        (changed: bool, stored_count: str)
    """
    stored_count = stored.get("company_employee_count", "")
    changed = claimed_count != stored_count
    return changed, stored_count


def is_company_data_fresh(stored: Dict[str, Any], days: int = 30) -> Tuple[bool, Optional[str]]:
    """
    Check if company data was updated within the specified number of days.

    Args:
        stored: Stored company record
        days: Number of days to consider data "fresh" (default 30)

    Returns:
        (is_fresh: bool, last_updated: str or None)
    """
    last_updated_str = stored.get("company_last_updated")

    if not last_updated_str:
        return False, None

    try:
        # Parse ISO format timestamp
        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
        now = datetime.utcnow()

        # Make both timezone-naive for comparison
        if last_updated.tzinfo is not None:
            last_updated = last_updated.replace(tzinfo=None)

        days_since_update = (now - last_updated).days
        is_fresh = days_since_update <= days

        return is_fresh, last_updated_str

    except Exception as e:
        logger.error(f"❌ Error parsing last_updated timestamp: {e}")
        return False, last_updated_str
