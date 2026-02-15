"""
Qualification System: Lead Database Verification

Verifies that leads returned by qualification models match the actual data
in the miner_test_leads table. This prevents gaming where models modify
fields (employee_count, role, industry, etc.) to better match the ICP.

Design:
- ONE batch query at the start of scoring (low latency)
- Local dict lookups per lead (O(1) per lead)
- If lead_id is provided: primary key lookup (fastest)
- If lead_id is not provided: business name + role fallback

IMPORTANT: This verification time is NOT counted against the model's
execution time. It runs during the validator's scoring phase.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import httpx

from gateway.qualification.models import LeadOutput

logger = logging.getLogger(__name__)

# Supabase config (validator environment)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qplwoislplkcegvdmbim.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
)

# Fields to verify (model field → DB column)
# These are the fields that MUST match between what the model returns
# and what's actually stored in miner_test_leads.
FIELDS_TO_VERIFY: Dict[str, str] = {
    "business": "business",
    "employee_count": "employee_count",
    "role": "role",
    "role_type": "role_type",
    "industry": "industry",
    "sub_industry": "sub_industry",
    "city": "city",
    "state": "state",
    "country": "country",
}

# URL fields need normalization before comparison
URL_FIELDS_TO_VERIFY: Dict[str, str] = {
    "company_linkedin": "company_linkedin",
    "company_website": "website",
}


def _normalize_for_comparison(value: Optional[str]) -> str:
    """Normalize a string value for comparison (strip, lowercase)."""
    if value is None:
        return ""
    return value.strip().lower()


def _normalize_url(url: Optional[str]) -> str:
    """Normalize a URL for comparison (strip protocol, www, trailing slash)."""
    if not url:
        return ""
    url = url.strip().lower()
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "")
    return url.rstrip("/")


def verify_lead_fields(lead: LeadOutput, db_row: dict) -> Tuple[bool, Optional[str]]:
    """
    Compare a lead's fields against the DB row.
    
    Args:
        lead: Model's output lead
        db_row: Row from miner_test_leads
    
    Returns:
        (passed, failure_reason) — if passed=False, the lead was tampered with
    """
    mismatches = []
    
    # Check standard fields (exact match, case-insensitive)
    for model_field, db_field in FIELDS_TO_VERIFY.items():
        model_value = _normalize_for_comparison(getattr(lead, model_field, None))
        db_value = _normalize_for_comparison(db_row.get(db_field))
        
        if model_value != db_value:
            mismatches.append(
                f"{model_field}: model='{getattr(lead, model_field, '')}' vs db='{db_row.get(db_field, '')}'"
            )
    
    # Check URL fields (normalized)
    for model_field, db_field in URL_FIELDS_TO_VERIFY.items():
        model_value = _normalize_url(getattr(lead, model_field, None))
        db_value = _normalize_url(db_row.get(db_field))
        
        if model_value != db_value:
            mismatches.append(
                f"{model_field}: model='{getattr(lead, model_field, '')}' vs db='{db_row.get(db_field, '')}'"
            )
    
    if mismatches:
        reason = f"DB verification failed — fields tampered: {'; '.join(mismatches[:3])}"
        if len(mismatches) > 3:
            reason += f" (+{len(mismatches) - 3} more)"
        return False, reason
    
    return True, None


async def batch_fetch_db_leads_by_ids(lead_ids: List[int]) -> Dict[int, dict]:
    """
    Fetch leads from miner_test_leads by primary key IDs in one query.
    
    Args:
        lead_ids: List of lead IDs to fetch
    
    Returns:
        Dict mapping lead_id → row data
    """
    if not lead_ids:
        return {}
    
    # Supabase REST API: select by IDs using `id=in.(1,2,3)`
    ids_str = ",".join(str(i) for i in lead_ids)
    url = f"{SUPABASE_URL}/rest/v1/miner_test_leads"
    params = {
        "select": "id,business,website,employee_count,role,role_type,industry,sub_industry,city,state,country,company_linkedin",
        "id": f"in.({ids_str})",
    }
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=10.0)
            resp.raise_for_status()
            rows = resp.json()
            return {row["id"]: row for row in rows}
    except Exception as e:
        logger.error(f"Failed to fetch leads by IDs from DB: {e}")
        return {}


async def batch_fetch_db_leads_by_names(business_names: List[str]) -> Dict[str, List[dict]]:
    """
    Fetch leads from miner_test_leads by business name in one query.
    Fallback when lead_id is not provided.
    
    Args:
        business_names: List of business names to look up
    
    Returns:
        Dict mapping lowercase business name → list of matching rows
        (multiple rows possible for same company with different roles)
    """
    if not business_names:
        return {}
    
    # Supabase REST API: select by business name using `business=in.("name1","name2")`
    # Note: names need to be quoted for the Supabase filter
    names_str = ",".join(f'"{name}"' for name in business_names)
    url = f"{SUPABASE_URL}/rest/v1/miner_test_leads"
    params = {
        "select": "id,business,website,employee_count,role,role_type,industry,sub_industry,city,state,country,company_linkedin",
        "business": f"in.({names_str})",
        "limit": "1000",
    }
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=10.0)
            resp.raise_for_status()
            rows = resp.json()
            
            # Group by lowercase business name
            result: Dict[str, List[dict]] = {}
            for row in rows:
                key = row["business"].strip().lower()
                result.setdefault(key, []).append(row)
            return result
    except Exception as e:
        logger.error(f"Failed to fetch leads by names from DB: {e}")
        return {}


async def verify_leads_batch(leads: List[LeadOutput]) -> Dict[int, str]:
    """
    Verify a batch of leads against the miner_test_leads DB.
    
    Makes ONE query to the DB, then verifies each lead locally.
    
    Args:
        leads: List of leads from the model
    
    Returns:
        Dict mapping lead index → failure reason.
        Leads NOT in the dict passed verification.
    """
    if not leads:
        return {}
    
    failures: Dict[int, str] = {}
    
    # Separate leads with and without lead_id
    leads_with_id = [(i, lead) for i, lead in enumerate(leads) if lead.lead_id is not None]
    leads_without_id = [(i, lead) for i, lead in enumerate(leads) if lead.lead_id is None]
    
    # --- Batch 1: Verify leads WITH lead_id (fast primary key lookup) ---
    if leads_with_id:
        ids = [lead.lead_id for _, lead in leads_with_id]
        db_rows_by_id = await batch_fetch_db_leads_by_ids(ids)
        
        for idx, lead in leads_with_id:
            db_row = db_rows_by_id.get(lead.lead_id)
            if db_row is None:
                failures[idx] = f"DB verification failed — lead_id={lead.lead_id} not found in miner_test_leads"
                continue
            
            passed, reason = verify_lead_fields(lead, db_row)
            if not passed:
                failures[idx] = reason
    
    # --- Batch 2: Verify leads WITHOUT lead_id (business name fallback) ---
    if leads_without_id:
        names = list(set(lead.business for _, lead in leads_without_id))
        db_rows_by_name = await batch_fetch_db_leads_by_names(names)
        
        for idx, lead in leads_without_id:
            name_key = lead.business.strip().lower()
            matching_rows = db_rows_by_name.get(name_key, [])
            
            if not matching_rows:
                failures[idx] = f"DB verification failed — business '{lead.business}' not found in miner_test_leads"
                continue
            
            # Find the best matching row (match on business + role)
            best_match = None
            lead_role_norm = _normalize_for_comparison(lead.role)
            for row in matching_rows:
                if _normalize_for_comparison(row.get("role")) == lead_role_norm:
                    best_match = row
                    break
            
            # If no exact role match, use first row with same business
            # (fields like employee_count, industry etc. are per-company, not per-role)
            if best_match is None:
                best_match = matching_rows[0]
            
            passed, reason = verify_lead_fields(lead, best_match)
            if not passed:
                failures[idx] = reason
    
    if failures:
        logger.warning(f"DB verification: {len(failures)}/{len(leads)} leads failed field verification")
    else:
        logger.info(f"DB verification: all {len(leads)} leads passed field verification")
    
    return failures
