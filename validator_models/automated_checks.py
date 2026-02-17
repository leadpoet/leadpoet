import aiohttp
import asyncio
import dns.resolver
import os
import re
import time
import json
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, List, Optional
from disposable_email_domains import blocklist as DISPOSABLE_DOMAINS
from Leadpoet.utils.utils_lead_extraction import (
    get_email, get_website, get_company, get_first_name, get_last_name,
    get_location, get_industry, get_role, get_linkedin, get_field,
    get_employee_count, get_description
)
from validator_models.stage5_verification import (
    check_stage5_unified, parse_employee_count,
)

# Sub-module imports
from validator_models.checks_utils import (
    EmailVerificationUnavailableError,
    validation_cache, log_validation_metrics,
    get_standardized_company_name, set_standardized_company_name,
    TRUELIST_BATCH_MAX_RETRIES, EMAIL_CACHE_FILE,
    HTTP_PROXY_URL, TRUELIST_API_KEY,
)
from validator_models.checks_icp import (
    calculate_icp_adjustment, is_enterprise_company,
    _matches_icp_definitions, batch_validate_roles_llm,
)
from validator_models.checks_email import (
    check_domain_age, check_mx_record, check_spf_dmarc,
    check_head_request, check_dnsbl,
    submit_truelist_batch, poll_truelist_batch,
    delete_truelist_batch, delete_all_truelist_batches,
    retry_truelist_batch, verify_emails_inline,
    run_centralized_truelist_batch,
    is_disposable_email,
)
from validator_models.checks_linkedin import check_linkedin_gse
from validator_models.checks_repscore import (
    check_wayback_machine, check_sec_edgar,
    check_whois_dnsbl_reputation, check_gdelt_mentions,
    check_companies_house,
    check_source_provenance, check_licensed_resale_proof,
)

# Re-exports for backward compatibility (other files import these from automated_checks)
from validator_models.checks_email import run_centralized_truelist_batch, check_domain_age
from validator_models.checks_utils import EmailVerificationUnavailableError
# get_email is already imported from Leadpoet.utils.utils_lead_extraction

MAX_REP_SCORE = 48  # Wayback (6) + SEC (12) + WHOIS/DNSBL (10) + GDELT (10) + Companies House (10) = 48

# ========================================================================
# Stage 0: Basic Hardcoded Checks
# ========================================================================

async def check_required_fields(lead: dict) -> Tuple[bool, dict]:
    """Check that all required fields are present and non-empty.

    Region validation:
    - country and city are ALWAYS required
    - state is required ONLY for United States leads
    - The validator builds the region string internally from country/state/city
    """
    required_fields = {
        "industry": ["industry", "Industry"],
        "sub_industry": ["sub_industry", "sub-industry", "Sub-industry", "Sub_industry"],
        "role": ["role", "Role"],
        "country": ["country", "Country"],
        "city": ["city", "City"],
    }

    missing_fields = []

    # Check for name (either full_name OR both first + last)
    full_name = lead.get("full_name") or lead.get("Full_name") or lead.get("Full Name")
    first_name = lead.get("first") or lead.get("First") or lead.get("first_name")
    last_name = lead.get("last") or lead.get("Last") or lead.get("last_name")

    has_name = bool(full_name) or (bool(first_name) and bool(last_name))
    if not has_name:
        missing_fields.append("contact_name")

    # Check other required fields
    for field_name, possible_keys in required_fields.items():
        found = False
        for key in possible_keys:
            value = lead.get(key)
            if value and str(value).strip():  # Check for non-empty string
                found = True
                break

        if not found:
            missing_fields.append(field_name)

    # Special check: state is required for US leads
    country = lead.get("country") or lead.get("Country") or ""
    country_lower = country.lower().strip() if country else ""
    us_aliases = ["united states", "usa", "us", "u.s.", "u.s.a.", "america", "united states of america"]

    if country_lower in us_aliases:
        state = lead.get("state") or lead.get("State") or ""
        if not state or not str(state).strip():
            missing_fields.append("state (required for US leads)")

    # Return structured rejection if any fields are missing
    if missing_fields:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_required_fields",
            "message": f"Missing required fields: {', '.join(missing_fields)}",
            "failed_fields": missing_fields
        }

    return True, {}

async def check_email_regex(lead: dict) -> Tuple[bool, dict]:
    """Check email format using RFC-5322 simplified regex with Unicode support (RFC 6531)"""
    try:
        email = get_email(lead)
        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            # Cache result
            cache_key = f"email_regex:no_email"
            validation_cache[cache_key] = (False, rejection_reason)
            await log_validation_metrics(lead, {"passed": False, "reason": rejection_reason["message"]}, "email_regex")
            return False, rejection_reason

        # RFC-5322 simplified regex (original ASCII validation)
        pattern_ascii = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        is_valid_ascii = bool(re.match(pattern_ascii, email))

        # RFC-6531 - Internationalized Email (Unicode support for international characters)
        # Allows emails like: anna.kosińska@cdprojekt.com, müller@siemens.de
        pattern_unicode = r"^[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}$"
        is_valid_unicode = bool(re.match(pattern_unicode, email, re.UNICODE))

        # Accept if EITHER pattern matches (ASCII OR Unicode)
        is_valid = is_valid_ascii or is_valid_unicode

        if not is_valid:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": f"Invalid email format: {email}",
                "failed_fields": ["email"]
            }
            # Cache result
            cache_key = f"email_regex:{email}"
            validation_cache[cache_key] = (False, rejection_reason)
            await log_validation_metrics(lead, {"passed": False, "reason": rejection_reason["message"]}, "email_regex")
            return False, rejection_reason

        # Reject emails with "+" sign (prevents duplicate submission exploit via email aliasing)
        # Example: jwest+alias1@domain.com and jwest+alias2@domain.com are the same email
        if "+" in email.split("@")[0]:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": f"Email contains '+' alias character (not allowed): {email}",
                "failed_fields": ["email"]
            }
            # Cache result
            cache_key = f"email_regex:{email}"
            validation_cache[cache_key] = (False, rejection_reason)
            await log_validation_metrics(lead, {"passed": False, "reason": rejection_reason["message"]}, "email_regex")
            return False, rejection_reason

        # Valid email - cache success result
        cache_key = f"email_regex:{email}"
        validation_cache[cache_key] = (True, {})
        await log_validation_metrics(lead, {"passed": True, "reason": "Valid email format"}, "email_regex")

        return True, {}
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_email_regex",
            "message": f"Email regex check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        await log_validation_metrics(lead, {"passed": False, "reason": str(e)}, "email_regex")
        return False, rejection_reason

async def check_name_email_match(lead: dict) -> Tuple[bool, dict]:
    """
    Check if first name or last name appears in the email address.
    This is a HARD check that prevents costly API calls for leads that will fail anyway.

    Returns:
        (True, {}): If first OR last name found in email
        (False, rejection_reason): If NO name found in email
    """
    try:
        email = get_email(lead)
        first_name = get_first_name(lead)
        last_name = get_last_name(lead)

        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason

        if not first_name or not last_name:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": "Missing first name or last name",
                "failed_fields": ["first_name", "last_name"]
            }
            return False, rejection_reason

        # Extract local part of email (before @)
        local_part = email.split("@")[0].lower() if "@" in email else email.lower()

        # Normalize names for comparison (lowercase, remove special chars)
        first_normalized = re.sub(r'[^a-z0-9]', '', first_name.lower())
        last_normalized = re.sub(r'[^a-z0-9]', '', last_name.lower())
        local_normalized = re.sub(r'[^a-z0-9]', '', local_part)

        # Check if either first OR last name appears in email
        # Pattern matching: full name, first initial + last, last + first initial, etc.
        # Also handles shortened names by checking if email local part is a prefix of the name
        # Examples: "rich@" matches "Richard" (prefix check), "greg@" matches "Gregory" (prefix check)
        # Security: Requires minimum 3 characters and checks that local part matches BEGINNING of name (not substring)

        # Minimum match length to prevent false positives (e.g., "an" in "daniel")
        MIN_NAME_MATCH_LENGTH = 3

        name_match = False

        # Strategy 1: Check if normalized name patterns appear in local part
        # This handles: john@example.com, johndoe@example.com, jdoe@example.com
        patterns = []

        # Full normalized names
        if len(first_normalized) >= MIN_NAME_MATCH_LENGTH:
            patterns.append(first_normalized)  # john
        if len(last_normalized) >= MIN_NAME_MATCH_LENGTH:
            patterns.append(last_normalized)  # doe

        # Full name combinations
        patterns.append(f"{first_normalized}{last_normalized}")  # johndoe

        # Initial + last name combinations
        if len(first_normalized) > 0:
            patterns.append(f"{first_normalized[0]}{last_normalized}")  # jdoe
            patterns.append(f"{last_normalized}{first_normalized[0]}")  # doej

        # Check if any pattern appears in the normalized local part
        patterns = [p for p in patterns if p and len(p) >= MIN_NAME_MATCH_LENGTH]
        name_match = any(pattern in local_normalized for pattern in patterns)

        # Strategy 2: Check if local part matches shortened versions of the name
        # This handles: greg@example.com where first_name is "Gregory"
        # Check if local_part is a prefix of the normalized name (shortened form)
        if not name_match and len(local_normalized) >= MIN_NAME_MATCH_LENGTH:
            # Check if local_part matches beginning of first name (shortened)
            # e.g., "greg" matches "gregory" (local_part is prefix of name)
            if len(first_normalized) >= len(local_normalized):
                if first_normalized.startswith(local_normalized):
                    name_match = True

            # Check if local_part matches beginning of last name (shortened)
            if not name_match and len(last_normalized) >= len(local_normalized):
                if last_normalized.startswith(local_normalized):
                    name_match = True

            # Check if name prefixes appear in local part (reverse direction)
            # e.g., "gregory" prefix "greg" in local_part "greg"
            if not name_match:
                # Check first name prefixes (3-6 characters)
                for length in range(MIN_NAME_MATCH_LENGTH, min(len(first_normalized) + 1, 7)):
                    name_prefix = first_normalized[:length]
                    if name_prefix == local_normalized or name_prefix in local_normalized:
                        name_match = True
                        break

                # Check last name prefixes if still no match
                if not name_match:
                    for length in range(MIN_NAME_MATCH_LENGTH, min(len(last_normalized) + 1, 7)):
                        name_prefix = last_normalized[:length]
                        if name_prefix == local_normalized or name_prefix in local_normalized:
                            name_match = True
                            break

        if not name_match:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": f"Name '{first_name} {last_name}' does not match email pattern '{email}'",
                "failed_fields": ["email", "first_name", "last_name"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - Name not found in email")
            return False, rejection_reason

        print(f"   ✅ Stage 0: {email} @ {get_company(lead)} - Name found in email")
        return True, {}

    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_name_email_match",
            "message": f"Name-email match check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_general_purpose_email(lead: dict) -> Tuple[bool, dict]:
    """
    Check if email is a general-purpose email address (instant fail).

    General-purpose emails are not personal contacts and should be rejected immediately
    to save API costs and maintain lead quality.

    Returns:
        (True, {}): If email is NOT general purpose (personal contact)
        (False, rejection_reason): If email IS general purpose (instant fail)
    """
    try:
        email = get_email(lead)

        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_general_purpose_email",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason

        # Define general-purpose email prefixes (must match calculate-rep-score exactly)
        general_purpose_prefixes = [
            'info@', 'hello@', 'owner@', 'ceo@', 'founder@', 'contact@', 'support@',
            'team@', 'admin@', 'office@', 'mail@', 'connect@', 'help@', 'hi@',
            'welcome@', 'inquiries@', 'general@', 'feedback@', 'ask@', 'outreach@',
            'communications@', 'crew@', 'staff@', 'community@', 'reachus@', 'talk@',
            'service@'
        ]

        email_lower = email.lower()

        # Check if email starts with any general-purpose prefix
        matched_prefix = next((prefix for prefix in general_purpose_prefixes if email_lower.startswith(prefix)), None)

        if matched_prefix:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_general_purpose_email",
                "message": f"Email '{email}' is a general purpose email (starts with {matched_prefix}) - not a personal contact",
                "failed_fields": ["email"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - General purpose email detected: {matched_prefix}")
            return False, rejection_reason

        # Not a general-purpose email - proceed
        print(f"   ✅ Stage 0: {email} @ {get_company(lead)} - Personal email (not general purpose)")
        return True, {}

    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_general_purpose_email",
            "message": f"General purpose email check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_free_email_domain(lead: dict) -> Tuple[bool, dict]:
    """
    Check if email uses a free/personal email domain (instant fail).

    B2B leads should use corporate email domains, not free consumer services.
    This prevents low-quality leads from free email providers.

    Returns:
        (True, {}): If email is corporate domain
        (False, rejection_reason): If email is free domain (gmail, yahoo, etc.)
    """
    try:
        email = get_email(lead)

        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_free_email_domain",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason

        # Extract domain from email
        try:
            domain = email.split("@")[1].lower() if "@" in email else ""
        except IndexError:
            return True, {}  # Invalid format handled by other checks

        # Common free email domains (comprehensive list)
        free_domains = {
            'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.fr',
            'outlook.com', 'hotmail.com', 'live.com', 'msn.com',
            'aol.com', 'mail.com', 'protonmail.com', 'proton.me',
            'icloud.com', 'me.com', 'mac.com',
            'zoho.com', 'yandex.com', 'gmx.com', 'mail.ru'
        }

        if domain in free_domains:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_free_email_domain",
                "message": f"Email uses free consumer domain '{domain}' - B2B leads require corporate email",
                "failed_fields": ["email"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - Free email domain rejected: {domain}")
            return False, rejection_reason

        # Corporate domain - proceed
        return True, {}

    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_free_email_domain",
            "message": f"Free email domain check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_disposable(lead: dict) -> Tuple[bool, dict]:
    """Check if email domain is disposable"""
    email = get_email(lead)
    if not email:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_disposable",
            "message": "No email provided",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

    cache_key = f"disposable:{email}"
    if cache_key in validation_cache:
        return validation_cache[cache_key]

    try:
        is_disposable, reason = await is_disposable_email(email)
        # For validation pipeline: return True if check PASSES (email is NOT disposable)
        # return False if check FAILS (email IS disposable)
        if is_disposable:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_disposable",
                "message": f"Disposable email domain detected: {email}",
                "failed_fields": ["email"]
            }
            validation_cache[cache_key] = (False, rejection_reason)
            return False, rejection_reason
        else:
            validation_cache[cache_key] = (True, {})
            return True, {}
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_disposable",
            "message": f"Disposable check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        validation_cache[cache_key] = (False, rejection_reason)
        return False, rejection_reason

# ========================================================================
# Orchestrators
# ========================================================================

async def run_stage0_2_checks(lead: dict) -> Tuple[bool, dict]:
    """
    Run Stage 0, 1, and 2 checks only (no email verification).

    This function is extracted from run_automated_checks() to support
    batch email verification. It runs all checks BEFORE Stage 3 (email
    verification), which is handled separately by the batch process.

    The actual check functions are IDENTICAL to run_automated_checks() -
    only the orchestration is different.

    Stages included:
    - Pre-checks: Source provenance verification
    - Stage 0: Required fields, email regex, name-email match,
               general purpose email, free email, disposable, HEAD request
    - Stage 1: Domain age, MX record, SPF/DMARC (parallel)
    - Stage 2: DNSBL reputation check

    Args:
        lead: Lead dict with all fields

    Returns:
        Tuple[bool, dict]: (passed, partial_automated_checks_data)
            - If passed: (True, data with stage_0, stage_1, stage_2 populated)
            - If failed: (False, data with rejection_reason)

    Note:
        This function does NOT run Stage 3 (email verification).
        Email verification is handled by the batch process (submit_truelist_batch
        + poll_truelist_batch).
    """
    email = get_email(lead)
    company = get_company(lead)

    # Initialize structured data collection (same structure as run_automated_checks)
    automated_checks_data = {
        "stage_0_hardcoded": {
            "name_in_email": False,
            "is_general_purpose_email": False
        },
        "stage_1_dns": {
            "has_mx": False,
            "has_spf": False,
            "has_dmarc": False,
            "dmarc_policy": None
        },
        "stage_2_domain": {
            "dnsbl_checked": False,
            "dnsbl_blacklisted": False,
            "dnsbl_list": None,
            "domain_age_days": None,
            "domain_registrar": None,
            "domain_nameservers": None,
            "whois_updated_days_ago": None
        },
        "stage_3_email": {
            "email_status": "unknown",
            "email_score": 0,
            "is_disposable": False,
            "is_role_based": False,
            "is_free": False
        },
        "stage_4_linkedin": {
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        },
        "stage_5_verification": {
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None
        },
        "rep_score": {
            "total_score": 0,
            "max_score": MAX_REP_SCORE,
            "breakdown": {
                "wayback_machine": 0,
                "uspto_trademarks": 0,
                "sec_edgar": 0,
                "whois_dnsbl": 0,
                "gdelt": 0,
                "companies_house": 0
            }
        },
        "passed": False,
        "rejection_reason": None
    }

    # ========================================================================
    # Pre-Attestation Check: REMOVED
    # ========================================================================
    # NOTE: Attestation verification removed from validators.
    # Gateway verifies attestations during POST /submit.
    print(f"🔍 Pre-Attestation Check: Skipped (gateway verifies during submission)")

    # ========================================================================
    # Source Provenance Verification: Source Validation (HARD)
    # Validates source_url, source_type, denylist, and licensed resale proof
    # ========================================================================
    print(f"🔍 Source Provenance Verification: Source validation for {email} @ {company}")

    checks_stage0_5 = [
        check_source_provenance,       # Validate source URL, type, denylist
        check_licensed_resale_proof,   # Validate license hash if applicable
    ]

    for check_func in checks_stage0_5:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Source Provenance Verification failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    print("   ✅ Source Provenance Verification passed")

    # ========================================================================
    # Stage 0: Hardcoded Checks (MIXED)
    # - Required Fields, Email Regex, Name-Email Match, General Purpose Email, Disposable, HEAD Request
    # ========================================================================
    print(f"🔍 Stage 0: Hardcoded checks for {email} @ {company}")

    # OPTIMIZATION: Run instant checks first, then overlap HEAD request with Stage 1 DNS checks
    checks_stage0_instant = [
        check_required_fields,      # Required fields validation (HARD)
        check_email_regex,          # RFC-5322 regex validation (HARD)
        check_name_email_match,     # Name in email check (HARD)
        check_general_purpose_email,# General purpose email filter (HARD)
        check_free_email_domain,    # Reject free email domains (HARD)
        check_disposable,           # Filter throwaway email providers (HARD)
    ]

    for check_func in checks_stage0_instant:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 0 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    # Collect Stage 0 data after successful instant checks
    automated_checks_data["stage_0_hardcoded"]["name_in_email"] = True
    automated_checks_data["stage_0_hardcoded"]["is_general_purpose_email"] = False

    print("   ✅ Stage 0 instant checks passed")

    # OPTIMIZATION: Start HEAD request as background task (will check result after Stage 1)
    head_request_task = asyncio.create_task(check_head_request(lead))

    # ========================================================================
    # Stage 1: DNS Layer (MIXED)
    # - Domain Age, MX Record (HARD)
    # - SPF/DMARC (SOFT - always passes, appends data)
    # ========================================================================
    print(f"🔍 Stage 1: DNS layer checks for {email} @ {company}")

    # OPTIMIZATION: Run all Stage 1 DNS checks in parallel
    results = await asyncio.gather(
        check_domain_age(lead),
        check_mx_record(lead),
        check_spf_dmarc(lead),
        return_exceptions=True
    )

    # Check results
    check_names = ["check_domain_age", "check_mx_record", "check_spf_dmarc"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"   ❌ Stage 1 failed: {str(result)}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = {
                "stage": "Stage 1: DNS Layer",
                "check_name": check_names[i],
                "message": f"Check failed: {str(result)}",
                "failed_fields": ["domain"]
            }
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

        passed, rejection_reason = result
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 1 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

    # Collect Stage 1 DNS data after successful checks
    automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", True)
    automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
    automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
    automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"

    print("   ✅ Stage 1 passed")

    # ========================================================================
    # Stage 0 (continued): HEAD Request Check
    # Check result of background HEAD request task
    # ========================================================================
    print(f"🔍 Stage 0: Website HEAD request check for {email} @ {company}")
    passed, rejection_reason = await head_request_task
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 0 (HEAD request) failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 0 (HEAD request) passed")

    # ========================================================================
    # Stage 2: Lightweight Domain Reputation Checks (HARD)
    # - DNSBL (Domain Block List) - Spamhaus DBL lookup
    # ========================================================================
    print(f"🔍 Stage 2: Domain reputation checks for {email} @ {company}")
    passed, rejection_reason = await check_dnsbl(lead)

    # Collect Stage 2 domain data (DNSBL + WHOIS from Stage 1)
    automated_checks_data["stage_2_domain"]["dnsbl_checked"] = lead.get("dnsbl_checked", False)
    automated_checks_data["stage_2_domain"]["dnsbl_blacklisted"] = lead.get("dnsbl_blacklisted", False)
    automated_checks_data["stage_2_domain"]["dnsbl_list"] = lead.get("dnsbl_list")
    automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
    automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
    automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
    automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 2 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 2 passed")

    # ========================================================================
    # STOP HERE - Stage 3 (email verification) is handled by batch process
    # ========================================================================
    # Mark as passed up to Stage 2
    # The batch orchestrator will handle email verification separately
    automated_checks_data["passed"] = True  # Passed Stage 0-2

    print(f"   ✅ Stage 0-2 complete for {email} @ {company}")
    return True, automated_checks_data

async def run_stage4_5_repscore(
    lead: dict,
    email_result: dict,
    stage0_2_data: dict
) -> Tuple[bool, dict]:
    """
    Run Stage 4, Stage 5, and Rep Score checks only.

    This function is extracted from run_automated_checks() to support
    batch email verification. It runs AFTER the lead has passed both:
    1. TrueList batch email verification (email_result)
    2. Stage 0-2 checks (stage0_2_data from run_stage0_2_checks)

    The actual check functions (check_linkedin_gse, check_stage5_unified,
    check_wayback_machine, etc.) are called EXACTLY as in run_automated_checks().

    Args:
        lead: Lead dict with email, company, linkedin, etc.
        email_result: Result from TrueList batch for this email
                     {"status": "email_ok", "passed": True, "rejection_reason": None}
        stage0_2_data: Partial automated_checks_data from run_stage0_2_checks()

    Returns:
        Tuple[bool, dict]: (passed, complete_automated_checks_data)
    """
    email = get_email(lead)
    company = get_company(lead)

    # ========================================================================
    # MERGE: Start with stage0_2_data and extend with Stage 3-5 + Rep Score
    # ========================================================================
    automated_checks_data = stage0_2_data.copy()

    # Ensure Stage 3-5 and rep_score sections exist
    if "stage_3_email" not in automated_checks_data:
        automated_checks_data["stage_3_email"] = {
            "email_status": "unknown",
            "email_score": 0,
            "is_disposable": False,
            "is_role_based": False,
            "is_free": False
        }
    if "stage_4_linkedin" not in automated_checks_data:
        automated_checks_data["stage_4_linkedin"] = {
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        }
    if "stage_5_verification" not in automated_checks_data:
        automated_checks_data["stage_5_verification"] = {
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None
        }
    if "rep_score" not in automated_checks_data:
        automated_checks_data["rep_score"] = {
            "total_score": 0,
            "max_score": MAX_REP_SCORE,
            "breakdown": {
                "wayback_machine": 0,
                "uspto_trademarks": 0,
                "sec_edgar": 0,
                "whois_dnsbl": 0,
                "gdelt": 0,
                "companies_house": 0
            }
        }

    # ========================================================================
    # Stage 3: Populate from Batch Email Result (NO API CALL - already done)
    # ========================================================================
    print(f"🔍 Stage 3: Email verification (from batch) for {email} @ {company}")

    # Map TrueList batch status to internal format for lead["email_verifier_status"]
    # This matches the mapping in run_automated_checks() Stage 3 data collection
    batch_status = email_result.get("status", "unknown")

    if batch_status == "email_ok":
        lead["email_verifier_status"] = "Valid"
        email_status = "valid"
        email_passed = True
    elif batch_status in ["disposable"]:
        lead["email_verifier_status"] = "Disposable"
        email_status = "invalid"
        email_passed = False
    elif batch_status in ["failed_no_mailbox", "failed_syntax_check", "failed_mx_check"]:
        lead["email_verifier_status"] = "Invalid"
        email_status = "invalid"
        email_passed = False
    else:
        # unknown, timeout, error - should have been retried, treat as failure
        lead["email_verifier_status"] = "Unknown"
        email_status = "unknown"
        email_passed = False

    # Populate batch result flags on lead (for downstream compatibility)
    lead["email_verifier_disposable"] = email_result.get("is_disposable", False)
    lead["email_verifier_role_based"] = email_result.get("is_role_based", False)
    lead["email_verifier_free"] = email_result.get("is_free", False)

    # Collect Stage 3 email data
    automated_checks_data["stage_3_email"]["email_status"] = email_status
    automated_checks_data["stage_3_email"]["email_score"] = 10 if email_passed else 0
    automated_checks_data["stage_3_email"]["is_disposable"] = lead.get("email_verifier_disposable", False)
    automated_checks_data["stage_3_email"]["is_role_based"] = lead.get("email_verifier_role_based", False)
    automated_checks_data["stage_3_email"]["is_free"] = lead.get("email_verifier_free", False)

    if not email_passed:
        rejection_reason = email_result.get("rejection_reason") or {
            "stage": "Stage 3: Email Verification (Batch)",
            "check_name": "truelist_batch",
            "message": f"Email verification failed: {batch_status}",
            "failed_fields": ["email"]
        }
        print(f"   ❌ Stage 3 failed: {rejection_reason.get('message', 'Email verification failed')}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 3 passed (batch verified)")

    # ========================================================================
    # Stage 4: LinkedIn/GSE Validation (HARD)
    # EXTRACTED VERBATIM from run_automated_checks()
    # ========================================================================
    print(f"🔍 Stage 4: LinkedIn/GSE validation for {email} @ {company}")

    passed, rejection_reason = await check_linkedin_gse(lead)

    # Collect Stage 4 data even on failure
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 4 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 4 passed")

    # Collect Stage 4 data after successful check
    automated_checks_data["stage_4_linkedin"]["linkedin_verified"] = True
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")

    # ========================================================================
    # Stage 5: Role/Region/Industry Verification (HARD)
    # EXTRACTED VERBATIM from run_automated_checks()
    # - Uses ScrapingDog search + rule-based matching + LLM to verify role, region, industry
    # - Early exit: if role fails → skip region/industry
    # - Early exit: if region fails → skip industry
    # - Anti-gaming: rejects if miner puts multiple states in region
    # ========================================================================
    print(f"🔍 Stage 5: Role/Region/Industry verification for {email} @ {company}")

    passed, rejection_reason = await check_stage5_unified(lead)

    # Collect Stage 5 data (company verification - role/location handled by Stage 4)
    automated_checks_data["stage_5_verification"]["company_name_verified"] = lead.get("stage5_name_match", False)
    automated_checks_data["stage_5_verification"]["company_size_verified"] = lead.get("stage5_size_match", False)
    automated_checks_data["stage_5_verification"]["company_hq_verified"] = lead.get("stage5_hq_match", False)
    automated_checks_data["stage_5_verification"]["industry_verified"] = lead.get("stage5_industry_match", False)
    automated_checks_data["stage_5_verification"]["extracted_name"] = lead.get("stage5_extracted_name")
    automated_checks_data["stage_5_verification"]["extracted_size"] = lead.get("stage5_extracted_size")
    automated_checks_data["stage_5_verification"]["extracted_hq"] = lead.get("stage5_extracted_hq")
    automated_checks_data["stage_5_verification"]["extracted_industry"] = lead.get("stage5_extracted_industry")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 5 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        automated_checks_data["stage_5_verification"]["early_exit"] = rejection_reason.get("early_exit") if rejection_reason else None
        return False, automated_checks_data

    print("   ✅ Stage 5 passed")

    # ========================================================================
    # Company Table Data (for gateway to insert/update after consensus)
    # ========================================================================
    # Stage 5 sets flags on lead for company table operations.
    # Include in evidence_blob so gateway can process after lead is approved.
    # - stage5_skipped: Fresh cached company, no table action needed
    # - _insert_new_company: New company, insert into table
    # - _update_employee_count: Stale cached company, update timestamp + count
    # ========================================================================
    automated_checks_data["stage_5_verification"]["company_table_action"] = None
    automated_checks_data["stage_5_verification"]["stage5_skipped"] = lead.get("stage5_skipped", False)

    if lead.get("stage5_skipped"):
        # Fresh cached company - no table action needed
        automated_checks_data["stage_5_verification"]["company_table_action"] = "none_fresh"
    elif lead.get("_insert_new_company"):
        automated_checks_data["stage_5_verification"]["company_table_action"] = "insert"
        automated_checks_data["stage_5_verification"]["company_refined_description"] = lead.get("_company_refined_description", "")
        automated_checks_data["stage_5_verification"]["company_industry_top3"] = lead.get("_company_industry_top3", {})
        automated_checks_data["stage_5_verification"]["company_sub_industry_top3"] = lead.get("_company_sub_industry_top3", {})
        automated_checks_data["stage_5_verification"]["company_verified_employee_count"] = lead.get("_company_verified_employee_count", "")
    elif lead.get("_update_employee_count"):
        automated_checks_data["stage_5_verification"]["company_table_action"] = "update_employee_count"
        automated_checks_data["stage_5_verification"]["new_employee_count"] = lead.get("_new_employee_count", "")
        automated_checks_data["stage_5_verification"]["prev_employee_count"] = lead.get("_prev_employee_count", "")

    # ========================================================================
    # Rep Score: Soft Reputation Checks (SOFT)
    # EXTRACTED VERBATIM from run_automated_checks()
    # - Wayback Machine (max 6 points), SEC (max 12 points),
    #   WHOIS/DNSBL (max 10 points), GDELT Press/Media (max 10 points),
    #   Companies House (max 10 points)
    # - Always passes, appends scores to lead
    # - Total: 0-48 points
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company} (parallel execution)")

    # OPTIMIZATION: Run all rep score checks in parallel to save time
    # Old: Sequential execution = 6-12s total
    # New: Parallel execution = 3-4s total (time of slowest API)
    results = await asyncio.gather(
        check_wayback_machine(lead),
        check_sec_edgar(lead),
        check_whois_dnsbl_reputation(lead),
        check_gdelt_mentions(lead),
        check_companies_house(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )

    # Unpack results (handle exceptions gracefully)
    wayback_score, wayback_data = results[0] if not isinstance(results[0], Exception) else (0, {"error": str(results[0])})
    sec_score, sec_data = results[1] if not isinstance(results[1], Exception) else (0, {"error": str(results[1])})
    whois_dnsbl_score, whois_dnsbl_data = results[2] if not isinstance(results[2], Exception) else (0, {"error": str(results[2])})
    gdelt_score, gdelt_data = results[3] if not isinstance(results[3], Exception) else (0, {"error": str(results[3])})
    companies_house_score, companies_house_data = results[4] if not isinstance(results[4], Exception) else (0, {"error": str(results[4])})

    total_rep_score = (
        wayback_score + sec_score + whois_dnsbl_score + gdelt_score +
        companies_house_score
    )

    # ========================================================================
    # ENTERPRISE COMPANY REP SCORE MULTIPLIER (10,001+ employees)
    # For enterprise companies, apply a multiplier that caps rep score:
    # - ICP match: target = 10, multiplier = min(0, 10 - rep_score)
    # - No ICP match: target = 5, multiplier = min(0, 5 - rep_score)
    # Final rep = rep_score + multiplier (so it never exceeds target)
    # ========================================================================
    enterprise_multiplier = 0
    is_enterprise = is_enterprise_company(lead)
    if is_enterprise:
        matches_icp = _matches_icp_definitions(lead)
        target_score = 10 if matches_icp else 5
        enterprise_multiplier = min(0, target_score - total_rep_score)
        final_rep_score = total_rep_score + enterprise_multiplier
        print(f"   🏢 ENTERPRISE COMPANY (10,001+): Raw rep={total_rep_score:.1f}, target={target_score}, multiplier={enterprise_multiplier:.1f}, final={final_rep_score:.1f} ({'ICP match' if matches_icp else 'No ICP match'})")
    else:
        final_rep_score = total_rep_score

    # Append to lead data
    lead["rep_score"] = final_rep_score
    lead["rep_score_details"] = {
        "wayback": wayback_data,
        "sec": sec_data,
        "whois_dnsbl": whois_dnsbl_data,
        "gdelt": gdelt_data,
        "companies_house": companies_house_data
    }
    if is_enterprise:
        lead["rep_score_details"]["enterprise_company"] = True
        lead["rep_score_details"]["enterprise_multiplier"] = enterprise_multiplier
        lead["rep_score_details"]["raw_rep_score"] = total_rep_score

    # Append to automated_checks_data
    automated_checks_data["rep_score"] = {
        "total_score": final_rep_score,
        "max_score": MAX_REP_SCORE,
        "breakdown": {
            "wayback_machine": wayback_score,       # 0-6 points
            "sec_edgar": sec_score,                 # 0-12 points
            "whois_dnsbl": whois_dnsbl_score,       # 0-10 points
            "gdelt": gdelt_score,                   # 0-10 points
            "companies_house": companies_house_score      # 0-10 points
        }
    }
    if is_enterprise:
        automated_checks_data["rep_score"]["enterprise_company"] = True
        automated_checks_data["rep_score"]["enterprise_multiplier"] = enterprise_multiplier
        automated_checks_data["rep_score"]["raw_rep_score"] = total_rep_score

    print(f"   📊 Rep Score: {final_rep_score:.1f}/{MAX_REP_SCORE} (Wayback: {wayback_score:.1f}/6, SEC: {sec_score:.1f}/12, WHOIS/DNSBL: {whois_dnsbl_score:.1f}/10, GDELT: {gdelt_score:.1f}/10, Companies House: {companies_house_score:.1f}/10)")

    # ========================================================================
    # ICP Adjustment Calculation (NEW SYSTEM - Absolute Points)
    # Replaces the old multiplier system with absolute point adjustments
    # ========================================================================
    icp_adjustment = calculate_icp_adjustment(lead)
    # Store in is_icp_multiplier field for backwards compatibility
    # Values: -15 to +20 (new format) vs 1.0/1.5/5.0 (old format)
    lead["is_icp_multiplier"] = float(icp_adjustment)
    automated_checks_data["is_icp_multiplier"] = float(icp_adjustment)

    # ========================================================================
    # Company Name Standardization (only on approval)
    # ========================================================================
    # Use the company LinkedIn slug to get/set the standardized company name.
    # This ensures all leads with the same company_linkedin URL have the same
    # standardized company name, regardless of how the miner submitted it.
    # ========================================================================
    company_slug = lead.get("company_linkedin_slug")
    company_linkedin_data = lead.get("company_linkedin_data")

    if company_slug:
        # Check cache first
        standardized_name = get_standardized_company_name(company_slug)

        if standardized_name:
            # Cache hit - use cached standardized name
            print(f"   📦 Company name from cache: '{company_slug}' → '{standardized_name}'")
        else:
            # Cache miss - get from Stage 4 scraped data and save to cache
            if company_linkedin_data and company_linkedin_data.get("company_name_from_linkedin"):
                standardized_name = company_linkedin_data["company_name_from_linkedin"]
                set_standardized_company_name(company_slug, standardized_name)
            else:
                # Fallback to miner's submitted company name if no scraped data
                standardized_name = company
                print(f"   ⚠️ No scraped company name available, using submitted: '{standardized_name}'")

        # Set on lead and automated_checks_data
        lead["company_standardized"] = standardized_name
        automated_checks_data["company_standardized"] = standardized_name
        print(f"   ✅ Company standardized: '{company}' → '{standardized_name}'")
    else:
        # No company_linkedin_slug - use submitted company name
        lead["company_standardized"] = company
        automated_checks_data["company_standardized"] = company
        print(f"   ⚠️ No company LinkedIn slug, using submitted name: '{company}'")

    print(f"🎉 All stages passed for {email} @ {company}")

    # All checks passed - return structured success data
    automated_checks_data["passed"] = True
    automated_checks_data["rejection_reason"] = None

    # NOTE: rep_score is already set on lead object:
    # - Enterprise companies: line 3539 (hardcoded_score)
    # - Non-enterprise: line 3591 (total_rep_score)

    return True, automated_checks_data

def _check_epoch_from_block_file(current_epoch: int, container_id: int = 0) -> bool:
    """Check if epoch has changed by reading the shared block file.

    Returns True if epoch has changed (should abort), False if same epoch.
    """
    try:
        import json
        from pathlib import Path
        block_file = Path("validator_weights") / "current_block.json"
        if not block_file.exists():
            return False  # Can't check, assume same epoch
        with open(block_file, 'r') as f:
            data = json.load(f)
        file_epoch = data.get("epoch", current_epoch)
        if file_epoch > current_epoch:
            print(f"   ⚠️ Container {container_id}: Epoch changed {current_epoch} → {file_epoch} during batch processing!")
            return True
        return False
    except Exception:
        return False  # On error, don't abort


async def run_batch_automated_checks(
    leads: List[dict],
    container_id: int = 0,
    precomputed_email_results: Dict[str, dict] = None,
    leads_file_path: str = None,
    current_epoch: int = None
) -> List[Tuple[bool, dict]]:
    """
    Batch validation with SEQUENTIAL Stage 0-2 and Stage 4-5.
    Stage 0-2 runs IN PARALLEL with coordinator's centralized TrueList batch.

    This REPLACES calling run_automated_checks() individually for each lead.
    Orchestrates the full batch flow without modifying any actual validation checks.

    Flow (when leads_file_path is provided - worker/coordinator polling mode):
    1. Run Stage 0-2 SEQUENTIALLY for all leads
    2. POLL leads_file_path for truelist_results (coordinator updates file when done)
    3. Use polled results for Stage 4-5

    Flow (when precomputed_email_results is provided - already has results):
    1. Run Stage 0-2 SEQUENTIALLY for all leads
    2. Use precomputed email results directly
    3. Run Stage 4-5 SEQUENTIALLY

    Args:
        leads: List of lead dicts (e.g., 110 leads per container)
        container_id: Container ID (0-29) for logging.
        precomputed_email_results: Dict mapping email (lowercase) -> result dict.
                                   If provided, skip polling and use these directly.
        leads_file_path: Path to shared leads file for polling truelist_results.
                         If provided, poll this file after Stage 0-2 until truelist_results is available.

    Returns:
        List of (passed, automated_checks_data) tuples in SAME ORDER as input
        - passed: True (approved), False (rejected), or None (skipped)

    CRITICAL: Results are returned in the SAME ORDER as input leads.
    """
    print(f"📦 Starting batch validation for {len(leads)} leads")
    start_time = time.time()

    n = len(leads)

    # Handle empty batch
    if n == 0:
        print("   ⚠️ Empty batch - nothing to validate")
        return []

    # Initialize results array with None (will be filled in order)
    results = [None] * n  # Index-based for order preservation

    # ========================================================================
    # Step 1: Extract emails and build lookup maps
    # ========================================================================
    emails = []
    email_to_idx = {}  # email (lowercase) -> index in leads list

    for i, lead in enumerate(leads):
        email = get_email(lead)
        if email:
            email_lower = email.lower()  # Normalize to lowercase for matching with CSV results
            emails.append(email_lower)
            email_to_idx[email_lower] = i
        else:
            # No email - immediate rejection
            results[i] = (False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Pre-Batch",
                    "check_name": "email_extraction",
                    "message": "No email found in lead",
                                "failed_fields": ["email"]
                }
            })

    print(f"   📧 Extracted {len(emails)} emails from {n} leads")

    # Check if all leads rejected (no valid emails)
    if not emails:
        print("   ⚠️ No valid emails found - all leads rejected")
        return results

    # ========================================================================
    # Step 1.5: FAST PRE-FILTER - Reject emails without @ sign
    # ========================================================================
    # This is a super low-latency check that:
    # 1. Prevents TrueList batch from failing (it rejects entire batch if ANY email lacks @)
    # 2. Saves Stage 0-2 processing time for obviously invalid emails

    valid_emails = []
    invalid_syntax_count = 0

    for email in emails:
        if '@' not in email:
            # Instant rejection - no @ sign
            idx = email_to_idx[email]
            results[idx] = (False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Pre-Batch",
                    "check_name": "email_syntax_prefilter",
                    "message": "Email missing @ symbol (instant rejection)",
                                "failed_fields": ["email"]
                }
            })
            invalid_syntax_count += 1
        else:
            valid_emails.append(email)

    if invalid_syntax_count > 0:
        print(f"   ⚡ Pre-filter: Rejected {invalid_syntax_count} emails (missing @ sign)")

    print(f"   ✅ {len(valid_emails)} valid emails ready for batch processing")

    # Update email list and check if any remain
    emails = valid_emails

    if not emails:
        print("   ⚠️ No valid emails after pre-filter - all leads rejected")
        return results

    # ========================================================================
    # Step 2: Determine TrueList results source
    # ========================================================================
    # Centralized TrueList is handled EXTERNALLY by coordinator's background task.
    # This function just runs Stage 0-2, then polls file OR uses precomputed results.

    has_precomputed = precomputed_email_results is not None and len(precomputed_email_results) > 0
    needs_polling = leads_file_path is not None and not has_precomputed

    if has_precomputed:
        print(f"   📥 Using precomputed TrueList results ({len(precomputed_email_results)} emails)")
    elif needs_polling:
        print(f"   ⏳ Will poll {leads_file_path} for TrueList results after Stage 0-2")
    else:
        print(f"   ⚠️ No TrueList source - leads will fail email verification")

    # ========================================================================
    # Step 2.5: STAGGER DELAY for Stage 0-2 (prevents WHOIS rate limiting)
    # ========================================================================
    # With centralized TrueList, all containers start Stage 0-2 simultaneously.
    # This causes WHOIS servers to rate-limit us (connection resets).
    # Add container-specific delay so WHOIS requests are staggered across containers.
    STAGE0_2_STAGGER_DELAY_SECONDS = 8  # 8s between containers
    stagger_delay = container_id * STAGE0_2_STAGGER_DELAY_SECONDS

    if stagger_delay > 0:
        print(f"   ⏳ Container {container_id}: Waiting {stagger_delay}s before Stage 0-2 (staggered WHOIS)...")
        await asyncio.sleep(stagger_delay)

    # ========================================================================
    # Step 3: Run Stage 0-2 SEQUENTIALLY (while TrueList batch processes)
    # ========================================================================
    print(f"   🔍 Running Stage 0-2 checks SEQUENTIALLY for {n} leads...")

    stage0_2_results = []  # List of (passed, data) in order, indexed by lead position

    for i, lead in enumerate(leads):
        email = get_email(lead)

        # Skip leads without email (already rejected in Step 1)
        if not email:
            stage0_2_results.append((False, results[i][1] if results[i] else {}))
            continue

        print(f"   Stage 0-2: Lead {i+1}/{n} ({email})")

        try:
            passed, data = await run_stage0_2_checks(lead)
            stage0_2_results.append((passed, data))
        except Exception as e:
            print(f"      ❌ Stage 0-2 error: {e}")
            stage0_2_results.append((False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Stage 0-2",
                    "check_name": "run_stage0_2_checks",
                    "message": f"Stage 0-2 error: {str(e)}",
                    "error": str(e)
                }
            }))

        # 0.8-second delay between Stage 0-2 leads (rate limiting)
        if i < len(leads) - 1:
            await asyncio.sleep(0.8)

        # Epoch boundary check every 10 leads (avoid excessive file reads)
        if current_epoch is not None and (i + 1) % 10 == 0:
            if _check_epoch_from_block_file(current_epoch, container_id):
                print(f"   ❌ ABORTING Stage 0-2 at lead {i+1}/{n} - epoch changed!")
                # Fill remaining leads with rejection
                for j in range(i + 1, n):
                    if results[j] is None:  # Not already rejected
                        stage0_2_results.append((False, {
                            "passed": False,
                            "rejection_reason": {
                                "stage": "Stage 0-2",
                                "check_name": "epoch_boundary_abort",
                                "message": f"Aborted: epoch changed during processing"
                            }
                        }))
                    else:
                        stage0_2_results.append((False, results[j][1] if results[j] else {}))
                break

    stage0_2_passed_count = sum(1 for passed, _ in stage0_2_results if passed)
    print(f"   ✅ Stage 0-2 complete: {stage0_2_passed_count}/{n} passed")

    # ========================================================================
    # Step 4: Get TrueList results (precomputed OR poll file)
    # ========================================================================

    email_results = {}

    if has_precomputed:
        # Use precomputed results directly
        email_results = precomputed_email_results
        print(f"   ✅ Using precomputed email results: {len(email_results)} emails")
    elif needs_polling:
        # POLL the leads file until truelist_results is available (not None)
        print(f"   ⏳ Polling for TrueList results from coordinator...")

        poll_interval = 5  # seconds
        max_poll_time = 1200  # 20 minutes max wait
        poll_start = time.time()
        poll_waited = 0

        while True:
            try:
                import json
                with open(leads_file_path, 'r') as f:
                    file_data = json.load(f)
                    file_truelist = file_data.get("truelist_results")

                    if file_truelist is not None:
                        # Results available (dict, possibly empty if coordinator failed)
                        email_results = file_truelist
                        print(f"   ✅ Received TrueList results from coordinator: {len(email_results)} emails (waited {poll_waited}s)")
                        break
                    else:
                        # Still None = in progress
                        if poll_waited % 30 == 0 and poll_waited > 0:
                            print(f"   ⏳ Still waiting for TrueList... ({poll_waited}s elapsed)")
            except Exception as e:
                print(f"   ⚠️ Error reading leads file: {e}")

            await asyncio.sleep(poll_interval)
            poll_waited += poll_interval

            if poll_waited >= max_poll_time:
                print(f"   ❌ Timeout waiting for TrueList results ({max_poll_time}s)")
                print(f"   ⚠️ Leads will fail email verification")
                email_results = {}
                break

            # Epoch boundary check every 30s during polling
            if current_epoch is not None and poll_waited % 30 == 0:
                if _check_epoch_from_block_file(current_epoch, container_id):
                    print(f"   ❌ Epoch changed while waiting for TrueList - aborting poll")
                    email_results = {}
                    break
    else:
        # No source - all leads fail email verification
        print(f"   ⚠️ No TrueList results available - leads will fail email verification")
        email_results = {}

    # ========================================================================
    # Step 5: Categorize leads
    # ========================================================================
    stage4_5_queue = []  # List of (index, lead, email_result, stage0_2_data)
    needs_retry = []     # List of emails that errored

    for i, lead in enumerate(leads):
        email = get_email(lead)

        # Skip leads without email (already rejected)
        if not email:
            continue

        email_lower = email.lower()  # Use lowercase for lookup (CSV results are lowercase)
        stage0_2_passed, stage0_2_data = stage0_2_results[i]
        email_result = email_results.get(email_lower, None)  # None if not in results

        if not stage0_2_passed:
            # Failed Stage 0-2 → immediate reject
            results[i] = (False, stage0_2_data)
        elif email_result is None:
            # Email NOT IN results at all
            if has_precomputed or needs_polling:
                # Using precomputed/polled results: Coordinator couldn't verify this email → skip
                # (Coordinator has already done retries, so we trust the absence)
                results[i] = (None, {
                    "skipped": True,
                    "reason": "EmailNotInPrecomputedResults",
                    "message": "Coordinator could not verify this email"
                })
            else:
                # No external results: Queue for retry
                # This happens when TrueList's CSV doesn't include all emails
                needs_retry.append(email_lower)
        elif email_result.get("needs_retry"):
            # Email explicitly errored
            if has_precomputed or needs_polling:
                # WORKER MODE: Coordinator marked as needing retry but couldn't resolve → skip
                results[i] = (None, {
                    "skipped": True,
                    "reason": "EmailVerificationIncomplete",
                    "message": "Coordinator could not complete email verification"
                })
            else:
                # COORDINATOR MODE: Queue for retry
                needs_retry.append(email_lower)
        elif email_result.get("passed"):
            # Both passed → queue for Stage 4-5
            stage4_5_queue.append((i, lead, email_result, stage0_2_data))
        else:
            # Email explicitly failed (has status) → reject
            rejection_data = stage0_2_data.copy()
            rejection_data["passed"] = False
            rejection_data["rejection_reason"] = email_result.get("rejection_reason") or {
                "stage": "Stage 3: Email Verification (Batch)",
                "check_name": "truelist_batch",
                "message": f"Email verification failed: {email_result.get('status', 'unknown')}",
                                "failed_fields": ["email"]
            }
            results[i] = (False, rejection_data)

    print(f"   📊 Categorization: {len(stage4_5_queue)} ready for Stage 4-5, {sum(1 for r in results if r and r[0] == False)} rejected, {len(needs_retry)} need retry")

    # ========================================================================
    # Role Batch Validation
    # ========================================================================
    if stage4_5_queue:
        print(f"\n   🔍 Role Batch Validation: {len(stage4_5_queue)} leads...")

        # Extract unique roles from queue
        roles_to_validate = list(set(
            (lead.get("role") or "").strip()
            for _, lead, _, _ in stage4_5_queue
            if (lead.get("role") or "").strip()
        ))

        if roles_to_validate:
            role_validation_results = await batch_validate_roles_llm(roles_to_validate)

            # Filter queue - remove leads with invalid roles
            valid_queue = []
            for item in stage4_5_queue:
                idx, lead, email_result, stage0_2_data = item
                role = (lead.get("role") or "").strip()

                if not role or role_validation_results.get(role, True):
                    valid_queue.append(item)
                else:
                    rejection_data = stage0_2_data.copy()
                    rejection_data["passed"] = False
                    rejection_data["rejection_reason"] = {
                        "stage": "Role Batch Validation",
                        "check_name": "role_batch_validation",
                        "message": f"Invalid role format.",
                        "failed_fields": ["role"]
                    }
                    results[idx] = (False, rejection_data)

            rejected_by_llm = len(stage4_5_queue) - len(valid_queue)
            stage4_5_queue = valid_queue
            print(f"   📊 Role Batch Validation complete: {len(stage4_5_queue)} passed, {rejected_by_llm} rejected")
        else:
            print(f"   ℹ️ No roles to validate (all empty)")

    # ========================================================================
    # Step 6: Start Stage 4-5 SEQUENTIALLY + Handle retries in parallel
    # ========================================================================

    # Start retry batch if needed (runs in background)
    # NOTE: When polling file, retries are handled by coordinator - workers don't retry
    retry_task = None
    inline_task = None  # Inline verification task (runs in background after retries exhaust)
    retry_attempt = 0
    last_retry_batch_id = None  # Track retry batch_id for deletion before next retry

    if needs_retry and not has_precomputed and not needs_polling:
        # COORDINATOR MODE ONLY: Retry failed emails
        # CRITICAL: Delete the original batch BEFORE retrying
        # TrueList detects duplicate email content and rejects re-submissions.
        # Deleting the batch clears their duplicate detection for those emails.
        if original_batch_id:
            await delete_truelist_batch(original_batch_id)

        print(f"   🔄 Starting retry batch #1 for {len(needs_retry)} emails...")
        retry_task = asyncio.create_task(retry_truelist_batch(needs_retry, None))

    # Process Stage 4-5 queue SEQUENTIALLY
    queue_idx = 0
    total_stage4_5 = len(stage4_5_queue)
    epoch_aborted = False

    while queue_idx < len(stage4_5_queue) or retry_task is not None or inline_task is not None:
        # Epoch boundary check every 10 leads in Stage 4-5
        if current_epoch is not None and queue_idx > 0 and queue_idx % 10 == 0 and not epoch_aborted:
            if _check_epoch_from_block_file(current_epoch, container_id):
                print(f"   ❌ ABORTING Stage 4-5 at lead {queue_idx}/{len(stage4_5_queue)} - epoch changed!")
                epoch_aborted = True
                # Mark remaining queued leads as rejected
                for remaining_idx in range(queue_idx, len(stage4_5_queue)):
                    r_idx = stage4_5_queue[remaining_idx][0]
                    if results[r_idx] is None:
                        results[r_idx] = (False, {
                            "passed": False,
                            "rejection_reason": {
                                "stage": "Stage 4-5",
                                "check_name": "epoch_boundary_abort",
                                "message": "Aborted: epoch changed during processing"
                            }
                        })
                # Cancel retry/inline tasks if running
                if retry_task is not None and not retry_task.done():
                    retry_task.cancel()
                    retry_task = None
                if inline_task is not None and not inline_task.done():
                    inline_task.cancel()
                    inline_task = None
                break

        # Process next lead in Stage 4-5 queue (if available)
        if queue_idx < len(stage4_5_queue):
            idx, lead, email_result, stage0_2_data = stage4_5_queue[queue_idx]
            email = get_email(lead)
            print(f"   Stage 4-5: Lead {queue_idx+1}/{len(stage4_5_queue)} ({email})")

            try:
                passed, data = await run_stage4_5_repscore(lead, email_result, stage0_2_data)
                results[idx] = (passed, data)
            except Exception as e:
                print(f"      ❌ Stage 4-5 error: {e}")
                results[idx] = (False, {
                    "passed": False,
                    "rejection_reason": {
                        "stage": "Stage 4-5",
                        "check_name": "run_stage4_5_repscore",
                        "message": f"Stage 4-5 error: {str(e)}",
                        "error": str(e)
                    }
                })

            queue_idx += 1

            # No delay between Stage 4-5 leads - ScrapingDog/OpenRouter can handle it
            # (Stage 0-2 still has 0.8s delay for DNS/HEAD request rate limiting)

        # Check if retry batch completed (non-blocking check)
        if retry_task is not None and retry_task.done():
            try:
                last_retry_batch_id, retry_results = retry_task.result()
            except Exception as e:
                print(f"   ⚠️ Retry batch failed: {e}")
                last_retry_batch_id = None
                retry_results = {email: {"needs_retry": True, "error": str(e)} for email in needs_retry}

            retry_task = None
            still_needs_retry = []

            for email in needs_retry:
                result = retry_results.get(email, {"needs_retry": True})
                idx = email_to_idx[email]
                stage0_2_passed, stage0_2_data = stage0_2_results[idx]

                if result.get("needs_retry"):
                    still_needs_retry.append(email)
                elif result.get("passed"):
                    # Retry succeeded → add to Stage 4-5 queue
                    print(f"   ✅ Retry succeeded for: {email}")
                    stage4_5_queue.append((idx, leads[idx], result, stage0_2_data))
                else:
                    # Retry failed → reject
                    rejection_data = stage0_2_data.copy()
                    rejection_data["passed"] = False
                    rejection_data["rejection_reason"] = result.get("rejection_reason") or {
                        "stage": "Stage 3: Email Verification (Batch Retry)",
                        "check_name": "truelist_batch",
                        "message": f"Email verification failed after retry: {result.get('status', 'unknown')}",
                        "failed_fields": ["email"]
                    }
                    results[idx] = (False, rejection_data)

            needs_retry = still_needs_retry
            retry_attempt += 1

            print(f"   📊 After retry #{retry_attempt}: {len(still_needs_retry)} still pending, {len(stage4_5_queue) - queue_idx} added to queue")

            # Start next retry if needed and haven't exceeded max
            if needs_retry and retry_attempt < TRUELIST_BATCH_MAX_RETRIES:
                print(f"   🔄 Starting retry batch #{retry_attempt+1} for {len(needs_retry)} emails...")
                # Pass the previous retry batch_id so it gets deleted before new submission
                retry_task = asyncio.create_task(retry_truelist_batch(needs_retry, last_retry_batch_id))
            elif needs_retry and retry_attempt >= TRUELIST_BATCH_MAX_RETRIES:
                # ================================================================
                # INLINE FALLBACK: Retries exhausted, start inline in BACKGROUND
                # Stage 4-5 continues processing while inline runs
                # ================================================================
                print(f"   🔍 Starting inline verification for {len(needs_retry)} emails (retries exhausted)...")
                inline_task = asyncio.create_task(verify_emails_inline(needs_retry))

        # Check if inline verification completed (non-blocking check)
        if inline_task is not None and inline_task.done():
            try:
                inline_results = inline_task.result()
            except Exception as e:
                print(f"   ⚠️ Inline verification failed: {e}")
                inline_results = {email: {"needs_retry": True, "error": str(e)} for email in needs_retry}

            inline_task = None

            for email in needs_retry:
                idx = email_to_idx[email]
                stage0_2_passed, stage0_2_data = stage0_2_results[idx]
                result = inline_results.get(email.lower(), {"needs_retry": True})

                if result.get("passed"):
                    # Inline passed → add to Stage 4-5 queue
                    print(f"   ✅ Inline verified: {email}")
                    stage4_5_queue.append((idx, leads[idx], result, stage0_2_data))
                elif result.get("needs_retry"):
                    # Still can't verify → skip
                    print(f"   ⏭️ Cannot verify (batch + inline failed): {email}")
                    results[idx] = (None, {
                        "skipped": True,
                        "reason": "EmailVerificationUnavailable",
                        "message": f"Email verification unavailable after batch + inline"
                    })
                else:
                    # Inline explicitly failed → reject
                    rejection_data = stage0_2_data.copy()
                    rejection_data["passed"] = False
                    rejection_data["rejection_reason"] = result.get("rejection_reason") or {
                        "stage": "Stage 3: Email Verification (Inline)",
                        "check_name": "truelist_inline",
                        "message": f"Email failed inline verification: {result.get('status', 'unknown')}",
                        "failed_fields": ["email"]
                    }
                    results[idx] = (False, rejection_data)

            # Clear needs_retry since we've handled them
            needs_retry = []
            print(f"   📊 After inline: {len(stage4_5_queue) - queue_idx} leads added to Stage 4-5 queue")

        # If queue is empty but tasks pending, wait briefly before checking again
        if queue_idx >= len(stage4_5_queue) and (retry_task is not None or inline_task is not None):
            await asyncio.sleep(1)

    # ========================================================================
    # Step 7: (Moved) Inline verification now happens inside the while loop
    # immediately after retries are exhausted, so leads get added to Stage 4-5 queue
    # ========================================================================

    # ========================================================================
    # Summary
    # ========================================================================

    # Safety: Fill any remaining None slots (e.g. from epoch abort mid-processing)
    for i in range(len(results)):
        if results[i] is None:
            results[i] = (False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Batch",
                    "check_name": "unprocessed",
                    "message": "Lead was not processed (epoch boundary abort or unexpected skip)"
                }
            })

    elapsed = time.time() - start_time
    passed_count = sum(1 for r in results if r and r[0] is True)
    failed_count = sum(1 for r in results if r and r[0] is False)
    skipped_count = sum(1 for r in results if r and r[0] is None)

    print(f"📦 Batch validation complete in {elapsed:.1f}s")
    print(f"   ✅ Passed: {passed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏭️ Skipped: {skipped_count}")

    return results

async def run_automated_checks(lead: dict) -> Tuple[bool, dict]:
    """
    Run all automated checks in stages, returning (passed, structured_data).

    Returns:
        Tuple[bool, dict]: (passed, structured_automated_checks_data)
            - If passed: (True, structured_data with stage_1_dns, stage_2_domain, stage_3_email)
            - If failed: (False, structured_data with rejection_reason and partial check data)

    Structured data format (tasks2.md Phase 1):
    {
        "stage_1_dns": {
            "has_mx": bool,
            "has_spf": bool,
            "has_dmarc": bool,
            "dmarc_policy": str
        },
        "stage_2_domain": {
            "dnsbl_checked": bool,
            "dnsbl_blacklisted": bool,
            "dnsbl_list": str,
            "domain_age_days": int,
            "domain_registrar": str,
            "domain_nameservers": list,
            "whois_updated_days_ago": int
        },
        "stage_3_email": {
            "email_status": str,  # "valid", "catch-all", "invalid", "unknown"
            "email_score": int,
            "is_disposable": bool,
            "is_role_based": bool,
            "is_free": bool
        },
        "passed": bool,
        "rejection_reason": dict or None
    }
    """

    email = get_email(lead)
    company = get_company(lead)

    # Initialize structured data collection
    automated_checks_data = {
        "stage_0_hardcoded": {
            "name_in_email": False,
            "is_general_purpose_email": False
        },
        "stage_1_dns": {
            "has_mx": False,
            "has_spf": False,
            "has_dmarc": False,
            "dmarc_policy": None
        },
        "stage_2_domain": {
            "dnsbl_checked": False,
            "dnsbl_blacklisted": False,
            "dnsbl_list": None,
            "domain_age_days": None,
            "domain_registrar": None,
            "domain_nameservers": None,
            "whois_updated_days_ago": None
        },
        "stage_3_email": {
            "email_status": "unknown",
            "email_score": 0,
            "is_disposable": False,
            "is_role_based": False,
            "is_free": False
        },
        "stage_4_linkedin": {
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        },
        "stage_5_verification": {  # NEW: Role/Region/Industry verification
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None  # "role_failed", "region_failed", or None
        },
        "rep_score": {
            "total_score": 0,
            "max_score": MAX_REP_SCORE,
            "breakdown": {
                "wayback_machine": 0,
                "uspto_trademarks": 0,
                "sec_edgar": 0,
                "whois_dnsbl": 0,
                "gdelt": 0,
                "companies_house": 0
            }
        },
        "passed": False,
        "rejection_reason": None
    }

    # ========================================================================
    # Pre-Attestation Check: REMOVED
    # ========================================================================
    # NOTE: Attestation verification removed from validators.
    # Validators don't have Supabase credentials and shouldn't verify attestations.
    #
    # SECURITY: Gateway verifies attestations during POST /submit:
    # - If lead is in validator queue → gateway already verified attestation
    # - Validators trust gateway's verification (gateway is TEE-protected)
    # - This prevents security bypass where validator skips check due to 401 errors
    #
    # If you need attestation verification, implement it in gateway/api/submit.py
    print(f"🔍 Pre-Attestation Check: Skipped (gateway verifies during submission)")

    # ========================================================================
    # Source Provenance Verification: Source Validation (HARD)
    # Validates source_url, source_type, denylist, and licensed resale proof
    # ========================================================================
    print(f"🔍 Source Provenance Verification: Source validation for {email} @ {company}")

    checks_stage0_5 = [
        check_source_provenance,       # Validate source URL, type, denylist
        check_licensed_resale_proof,   # Validate license hash if applicable
    ]

    for check_func in checks_stage0_5:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Source Provenance Verification failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    print("   ✅ Source Provenance Verification passed")

    # ========================================================================
    # Stage 0: Hardcoded Checks (MIXED)
    # - Required Fields, Email Regex, Name-Email Match, General Purpose Email, Disposable, HEAD Request
    # - Deduplication (handled in validate_lead_list)
    # ========================================================================
    print(f"🔍 Stage 0: Hardcoded checks for {email} @ {company}")

    # OPTIMIZATION: Run instant checks first, then overlap HEAD request with Stage 1 DNS checks
    # Instant checks (run sequentially - they're <0.01s each anyway)
    checks_stage0_instant = [
        check_required_fields,      # Required fields validation (HARD)
        check_email_regex,          # RFC-5322 regex validation (HARD)
        check_name_email_match,     # Name in email check (HARD) - NEW
        check_general_purpose_email,# General purpose email filter (HARD) - NEW
        check_free_email_domain,    # Reject free email domains (HARD) - NEW
        check_disposable,           # Filter throwaway email providers (HARD)
    ]

    for check_func in checks_stage0_instant:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 0 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    # Collect Stage 0 data after successful instant checks
    automated_checks_data["stage_0_hardcoded"]["name_in_email"] = True  # Passed name-email match
    automated_checks_data["stage_0_hardcoded"]["is_general_purpose_email"] = False  # Not general purpose

    print("   ✅ Stage 0 instant checks passed")

    # OPTIMIZATION: Start HEAD request as background task (will check result after Stage 1)
    # This overlaps the 5-10s HEAD request with 1-3s Stage 1 DNS checks
    head_request_task = asyncio.create_task(check_head_request(lead))

    # ========================================================================
    # Stage 1: DNS Layer (MIXED)
    # - Domain Age, MX Record (HARD)
    # - SPF/DMARC (SOFT - always passes, appends data)
    # ========================================================================
    print(f"🔍 Stage 1: DNS layer checks for {email} @ {company}")

    # OPTIMIZATION: Run all Stage 1 DNS checks in parallel to save time
    # Old: Sequential execution = 2-5s total
    # New: Parallel execution = 1-3s (time of slowest check)
    results = await asyncio.gather(
        check_domain_age(lead),
        check_mx_record(lead),
        check_spf_dmarc(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )

    # Check results
    check_names = ["check_domain_age", "check_mx_record", "check_spf_dmarc"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exception
            print(f"   ❌ Stage 1 failed: {str(result)}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = {
                "stage": "Stage 1: DNS Layer",
                "check_name": check_names[i],
                "message": f"Check failed: {str(result)}",
                "failed_fields": ["domain"]
            }
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            # Collect partial Stage 2 data (WHOIS)
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

        passed, rejection_reason = result
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 1 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            # Collect partial Stage 2 data (WHOIS)
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

    # Collect Stage 1 DNS data after successful checks
    automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", True)  # Passed MX check
    automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
    automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
    automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"

    print("   ✅ Stage 1 passed")

    # ========================================================================
    # Stage 0 (continued): HEAD Request Check
    # Check result of background HEAD request task that was started before Stage 1
    # ========================================================================
    print(f"🔍 Stage 0: Website HEAD request check for {email} @ {company}")
    passed, rejection_reason = await head_request_task
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 0 (HEAD request) failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 0 (HEAD request) passed")

    # ========================================================================
    # Stage 2: Lightweight Domain Reputation Checks (HARD)
    # - DNSBL (Domain Block List) - Spamhaus DBL lookup
    # ========================================================================
    print(f"🔍 Stage 2: Domain reputation checks for {email} @ {company}")
    passed, rejection_reason = await check_dnsbl(lead)

    # Collect Stage 2 domain data (DNSBL + WHOIS from Stage 1)
    automated_checks_data["stage_2_domain"]["dnsbl_checked"] = lead.get("dnsbl_checked", False)
    automated_checks_data["stage_2_domain"]["dnsbl_blacklisted"] = lead.get("dnsbl_blacklisted", False)
    automated_checks_data["stage_2_domain"]["dnsbl_list"] = lead.get("dnsbl_list")
    automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
    automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
    automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
    automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 2 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 2 passed")

    # ========================================================================
    # Stage 3: Email Verification (DEPRECATED - use run_batch_automated_checks instead)
    # ========================================================================
    # NOTE: Single-email validation has been removed. Email validation is now
    # handled by TrueList BATCH API in run_batch_automated_checks().
    # This function is kept for backwards compatibility but should not be used.
    print(f"⚠️  Stage 3: DEPRECATED - use run_batch_automated_checks() for email validation")
    print(f"   Skipping single-email validation for {email}")

    # Mark Stage 3 as skipped (not verified)
    automated_checks_data["stage_3_email"]["email_status"] = "skipped"
    automated_checks_data["stage_3_email"]["email_score"] = 0
    automated_checks_data["stage_3_email"]["is_disposable"] = False
    automated_checks_data["stage_3_email"]["is_role_based"] = False
    automated_checks_data["stage_3_email"]["is_free"] = False

    print("   ⏭️  Stage 3 skipped (use batch validation)")

    # ========================================================================
    # Stage 4: LinkedIn/GSE Validation (HARD)
    # ========================================================================
    print(f"🔍 Stage 4: LinkedIn/GSE validation for {email} @ {company}")

    passed, rejection_reason = await check_linkedin_gse(lead)

    # Collect Stage 4 data even on failure
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 4 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 4 passed")

    # Collect Stage 4 data after successful check
    automated_checks_data["stage_4_linkedin"]["linkedin_verified"] = True
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")

    # ========================================================================
    # Stage 5: Role/Region/Industry Verification (HARD)
    # - Uses ScrapingDog search + rule-based matching + LLM to verify role, region, industry
    # - Early exit: if role fails → skip region/industry
    # - Early exit: if region fails → skip industry
    # - Anti-gaming: rejects if miner puts multiple states in region
    # ========================================================================
    print(f"🔍 Stage 5: Role/Region/Industry verification for {email} @ {company}")

    passed, rejection_reason = await check_stage5_unified(lead)

    # Collect Stage 5 data (company verification - role/location handled by Stage 4)
    automated_checks_data["stage_5_verification"]["company_name_verified"] = lead.get("stage5_name_match", False)
    automated_checks_data["stage_5_verification"]["company_size_verified"] = lead.get("stage5_size_match", False)
    automated_checks_data["stage_5_verification"]["company_hq_verified"] = lead.get("stage5_hq_match", False)
    automated_checks_data["stage_5_verification"]["industry_verified"] = lead.get("stage5_industry_match", False)
    automated_checks_data["stage_5_verification"]["extracted_name"] = lead.get("stage5_extracted_name")
    automated_checks_data["stage_5_verification"]["extracted_size"] = lead.get("stage5_extracted_size")
    automated_checks_data["stage_5_verification"]["extracted_hq"] = lead.get("stage5_extracted_hq")
    automated_checks_data["stage_5_verification"]["extracted_industry"] = lead.get("stage5_extracted_industry")

    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 5 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        automated_checks_data["stage_5_verification"]["early_exit"] = rejection_reason.get("early_exit") if rejection_reason else None
        return False, automated_checks_data

    print("   ✅ Stage 5 passed")

    # ========================================================================
    # Company Table Data (for gateway to insert/update after consensus)
    # ========================================================================
    # Stage 5 sets flags on lead for company table operations.
    # Include in evidence_blob so gateway can process after lead is approved.
    # - stage5_skipped: Fresh cached company, no table action needed
    # - _insert_new_company: New company, insert into table
    # - _update_employee_count: Stale cached company, update timestamp + count
    # ========================================================================
    automated_checks_data["stage_5_verification"]["company_table_action"] = None
    automated_checks_data["stage_5_verification"]["stage5_skipped"] = lead.get("stage5_skipped", False)

    if lead.get("stage5_skipped"):
        # Fresh cached company - no table action needed
        automated_checks_data["stage_5_verification"]["company_table_action"] = "none_fresh"
    elif lead.get("_insert_new_company"):
        automated_checks_data["stage_5_verification"]["company_table_action"] = "insert"
        automated_checks_data["stage_5_verification"]["company_refined_description"] = lead.get("_company_refined_description", "")
        automated_checks_data["stage_5_verification"]["company_industry_top3"] = lead.get("_company_industry_top3", {})
        automated_checks_data["stage_5_verification"]["company_sub_industry_top3"] = lead.get("_company_sub_industry_top3", {})
        automated_checks_data["stage_5_verification"]["company_verified_employee_count"] = lead.get("_company_verified_employee_count", "")
    elif lead.get("_update_employee_count"):
        automated_checks_data["stage_5_verification"]["company_table_action"] = "update_employee_count"
        automated_checks_data["stage_5_verification"]["new_employee_count"] = lead.get("_new_employee_count", "")
        automated_checks_data["stage_5_verification"]["prev_employee_count"] = lead.get("_prev_employee_count", "")

    # ========================================================================
    # Rep Score: Soft Reputation Checks (SOFT)
    # - Wayback Machine (max 6 points), SEC (max 12 points),
    #   WHOIS/DNSBL (max 10 points), GDELT Press/Media (max 10 points),
    #   Companies House (max 10 points)
    # - Always passes, appends scores to lead
    # - Total: 0-48 points
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company} (parallel execution)")

    # OPTIMIZATION: Run all rep score checks in parallel to save time
    # Old: Sequential execution = 6-12s total
    # New: Parallel execution = 3-4s total (time of slowest API)
    results = await asyncio.gather(
        check_wayback_machine(lead),
        check_sec_edgar(lead),
        check_whois_dnsbl_reputation(lead),
        check_gdelt_mentions(lead),
        check_companies_house(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )

    # Unpack results (handle exceptions gracefully)
    wayback_score, wayback_data = results[0] if not isinstance(results[0], Exception) else (0, {"error": str(results[0])})
    sec_score, sec_data = results[1] if not isinstance(results[1], Exception) else (0, {"error": str(results[1])})
    whois_dnsbl_score, whois_dnsbl_data = results[2] if not isinstance(results[2], Exception) else (0, {"error": str(results[2])})
    gdelt_score, gdelt_data = results[3] if not isinstance(results[3], Exception) else (0, {"error": str(results[3])})
    companies_house_score, companies_house_data = results[4] if not isinstance(results[4], Exception) else (0, {"error": str(results[4])})

    total_rep_score = (
        wayback_score + sec_score + whois_dnsbl_score + gdelt_score +
        companies_house_score
    )

    # ========================================================================
    # ENTERPRISE COMPANY REP SCORE MULTIPLIER (10,001+ employees)
    # For enterprise companies, apply a multiplier that caps rep score:
    # - ICP match: target = 10, multiplier = min(0, 10 - rep_score)
    # - No ICP match: target = 5, multiplier = min(0, 5 - rep_score)
    # Final rep = rep_score + multiplier (so it never exceeds target)
    # ========================================================================
    enterprise_multiplier = 0
    is_enterprise = is_enterprise_company(lead)
    if is_enterprise:
        matches_icp = _matches_icp_definitions(lead)
        target_score = 10 if matches_icp else 5
        enterprise_multiplier = min(0, target_score - total_rep_score)
        final_rep_score = total_rep_score + enterprise_multiplier
        print(f"   🏢 ENTERPRISE COMPANY (10,001+): Raw rep={total_rep_score:.1f}, target={target_score}, multiplier={enterprise_multiplier:.1f}, final={final_rep_score:.1f} ({'ICP match' if matches_icp else 'No ICP match'})")
    else:
        final_rep_score = total_rep_score

    # Append to lead data
    lead["rep_score"] = final_rep_score
    lead["rep_score_details"] = {
        "wayback": wayback_data,
        "sec": sec_data,
        "whois_dnsbl": whois_dnsbl_data,
        "gdelt": gdelt_data,
        "companies_house": companies_house_data
    }
    if is_enterprise:
        lead["rep_score_details"]["enterprise_company"] = True
        lead["rep_score_details"]["enterprise_multiplier"] = enterprise_multiplier
        lead["rep_score_details"]["raw_rep_score"] = total_rep_score

    # Append to automated_checks_data
    automated_checks_data["rep_score"] = {
        "total_score": final_rep_score,
        "max_score": MAX_REP_SCORE,
        "breakdown": {
            "wayback_machine": wayback_score,       # 0-6 points
            "sec_edgar": sec_score,                 # 0-12 points
            "whois_dnsbl": whois_dnsbl_score,       # 0-10 points
            "gdelt": gdelt_score,                   # 0-10 points
            "companies_house": companies_house_score      # 0-10 points
        }
    }
    if is_enterprise:
        automated_checks_data["rep_score"]["enterprise_company"] = True
        automated_checks_data["rep_score"]["enterprise_multiplier"] = enterprise_multiplier
        automated_checks_data["rep_score"]["raw_rep_score"] = total_rep_score

    print(f"   📊 Rep Score: {final_rep_score:.1f}/{MAX_REP_SCORE} (Wayback: {wayback_score:.1f}/6, SEC: {sec_score:.1f}/12, WHOIS/DNSBL: {whois_dnsbl_score:.1f}/10, GDELT: {gdelt_score:.1f}/10, Companies House: {companies_house_score:.1f}/10)")

    # ========================================================================
    # ICP Adjustment Calculation (NEW SYSTEM - Absolute Points)
    # Replaces the old multiplier system with absolute point adjustments
    # ========================================================================
    icp_adjustment = calculate_icp_adjustment(lead)
    # Store in is_icp_multiplier field for backwards compatibility
    # Values: -15 to +20 (new format) vs 1.0/1.5/5.0 (old format)
    lead["is_icp_multiplier"] = float(icp_adjustment)
    automated_checks_data["is_icp_multiplier"] = float(icp_adjustment)

    # ========================================================================
    # Company Name Standardization (only on approval)
    # ========================================================================
    # Use the company LinkedIn slug to get/set the standardized company name.
    # This ensures all leads with the same company_linkedin URL have the same
    # standardized company name, regardless of how the miner submitted it.
    # ========================================================================
    company_slug = lead.get("company_linkedin_slug")
    company_linkedin_data = lead.get("company_linkedin_data")

    if company_slug:
        # Check cache first
        standardized_name = get_standardized_company_name(company_slug)

        if standardized_name:
            # Cache hit - use cached standardized name
            print(f"   📦 Company name from cache: '{company_slug}' → '{standardized_name}'")
        else:
            # Cache miss - get from Stage 4 scraped data and save to cache
            if company_linkedin_data and company_linkedin_data.get("company_name_from_linkedin"):
                standardized_name = company_linkedin_data["company_name_from_linkedin"]
                set_standardized_company_name(company_slug, standardized_name)
            else:
                # Fallback to miner's submitted company name if no scraped data
                standardized_name = company
                print(f"   ⚠️ No scraped company name available, using submitted: '{standardized_name}'")

        # Set on lead and automated_checks_data
        lead["company_standardized"] = standardized_name
        automated_checks_data["company_standardized"] = standardized_name
        print(f"   ✅ Company standardized: '{company}' → '{standardized_name}'")
    else:
        # No company_linkedin_slug - use submitted company name
        lead["company_standardized"] = company
        automated_checks_data["company_standardized"] = company
        print(f"   ⚠️ No company LinkedIn slug, using submitted name: '{company}'")

    print(f"🎉 All stages passed for {email} @ {company}")

    # All checks passed - return structured success data
    automated_checks_data["passed"] = True
    automated_checks_data["rejection_reason"] = None

    # NOTE: rep_score is already set on lead object:
    # - Enterprise companies: line 11450 (hardcoded_score)
    # - Non-enterprise: line 11501 (total_rep_score)

    return True, automated_checks_data

# ========================================================================
# Legacy functions - maintained for backward compatibility
# ========================================================================

async def check_duplicates(leads: list) -> Tuple[bool, dict]:
    """Check for duplicate emails and return which leads are duplicates (not first occurrence)"""
    email_first_occurrence = {}  # Track first occurrence of each email
    duplicate_leads = {}  # Track which lead indices are duplicates

    for i, lead in enumerate(leads):
        email = get_email(lead)

        if email in email_first_occurrence:
            # This is a duplicate - mark this lead index as duplicate
            duplicate_leads[i] = email
        else:
            # First occurrence - record the lead index
            email_first_occurrence[email] = i

    return len(duplicate_leads) > 0, duplicate_leads

async def validate_lead_list(leads: list) -> list:
    """Main validation function - maintains backward compatibility"""

    # Check for duplicates
    has_duplicates, duplicate_leads = await check_duplicates(leads)
    if has_duplicates:
        duplicate_emails = set(duplicate_leads.values())
        print(f"Duplicate emails detected: {duplicate_emails}")
        print(f"Duplicate lead indices: {list(duplicate_leads.keys())}")

        # Process all leads, but mark duplicates as invalid
        report = []
        for i, lead in enumerate(leads):
            email = get_email(lead)
            website = get_website(lead)
            domain = urlparse(website).netloc if website else ""

            if i in duplicate_leads:
                # Mark duplicate lead as invalid
                report.append({
                    "lead_index": i,
                    "email": email,
                    "company_domain": domain,
                    "status": "Invalid",
                    "reason": "Duplicate email"
                })
            else:
                # Process non-duplicate leads through automated checks
                passed, automated_checks_data = await run_automated_checks(lead)
                status = "Valid" if passed else "Invalid"
                # Extract rejection_reason for backwards compatibility
                reason = automated_checks_data.get("rejection_reason", {}) if not passed else {}
                report.append({
                    "lead_index": i,
                    "email": email,
                    "company_domain": domain,
                    "status": status,
                    "reason": reason,
                    "automated_checks": automated_checks_data  # NEW: Include full structured data
                })

        return report

    # Process each lead through the new validation pipeline
    report = []
    for i, lead in enumerate(leads):
        email = get_email(lead)
        website = get_website(lead)
        domain = urlparse(website).netloc if website else ""

        # Run new automated checks
        passed, automated_checks_data = await run_automated_checks(lead)

        status = "Valid" if passed else "Invalid"
        # Extract rejection_reason for backwards compatibility
        reason = automated_checks_data.get("rejection_reason", {}) if not passed else {}
        report.append({
            "lead_index": i,
            "email": email,
            "company_domain": domain,
            "status": status,
            "reason": reason,
            "automated_checks": automated_checks_data  # NEW: Include full structured data
        })

    return report
