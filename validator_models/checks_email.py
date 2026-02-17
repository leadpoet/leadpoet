import aiohttp
import asyncio
import csv
import dns.resolver
import json
import os
import pickle
import re
import requests
import ssl
import time
import uuid
import whois
from datetime import datetime
from io import StringIO
from typing import Dict, Any, Tuple, List, Optional
from urllib.parse import urlparse
from disposable_email_domains import blocklist as DISPOSABLE_DOMAINS
from Leadpoet.utils.utils_lead_extraction import get_email, get_website
from validator_models.checks_utils import (
    EmailVerificationUnavailableError,
    HTTP_PROXY_URL,
    PROXY_CONFIG,
    TRUELIST_API_KEY,
    TRUELIST_BATCH_POLL_INTERVAL,
    TRUELIST_BATCH_TIMEOUT,
    TRUELIST_BATCH_STRATEGY,
    EMAIL_CACHE_FILE,
    CACHE_TTLS,
    API_SEMAPHORE,
    validation_cache,
    extract_root_domain,
    log_validation_metrics,
)


async def check_domain_age(lead: dict) -> Tuple[bool, dict]:
    """
    Check domain age using WHOIS lookup.
    Appends WHOIS data to lead object for reputation scoring.
    """
    website = get_website(lead)
    if not website:
        # Append default WHOIS data
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"domain_age:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois"]):
        cached_result = validation_cache[cache_key]
        # Restore cached WHOIS data to lead
        cached_data = validation_cache.get(f"{cache_key}_data")
        if cached_data:
            lead["whois_checked"] = cached_data.get("checked", True)
            lead["domain_age_days"] = cached_data.get("age_days")
            lead["domain_creation_date"] = cached_data.get("creation_date")
            lead["domain_registrar"] = cached_data.get("registrar")
            lead["domain_nameservers"] = cached_data.get("nameservers")
            lead["whois_updated_date"] = cached_data.get("updated_date")
            lead["whois_updated_days_ago"] = cached_data.get("whois_updated_days_ago")
        return cached_result

    try:
        # Implement actual WHOIS lookup
        def get_domain_age_sync(domain_name):
            try:
                w = whois.whois(domain_name)

                # Extract registrar, nameservers, and updated_date for reputation scoring
                registrar = getattr(w, 'registrar', None)
                nameservers = getattr(w, 'name_servers', None)
                if isinstance(nameservers, list):
                    nameservers = nameservers[:3]  # Limit to first 3 nameservers

                # Extract updated_date for WHOIS stability check
                updated_date = getattr(w, 'updated_date', None)
                if updated_date:
                    if isinstance(updated_date, list):
                        updated_date = updated_date[0]
                    # Make timezone-naive if timezone-aware
                    if hasattr(updated_date, 'tzinfo') and updated_date.tzinfo is not None:
                        updated_date = updated_date.replace(tzinfo=None)

                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        creation_date = w.creation_date[0]
                    else:
                        creation_date = w.creation_date

                    # Make creation_date timezone-naive if it's timezone-aware
                    if creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(tzinfo=None)

                    age_days = (datetime.now() - creation_date).days
                    min_age_days = 7  # 7 days minimum

                    # Calculate whois_updated_days_ago
                    whois_updated_days_ago = None
                    if updated_date:
                        whois_updated_days_ago = (datetime.now() - updated_date).days

                    # Return WHOIS data along with result
                    whois_data = {
                        "age_days": age_days,
                        "creation_date": creation_date.isoformat(),
                        "registrar": registrar,
                        "nameservers": nameservers,
                        "updated_date": updated_date.isoformat() if updated_date else None,
                        "whois_updated_days_ago": whois_updated_days_ago,
                        "checked": True
                    }

                    if age_days >= min_age_days:
                        return (True, {}, whois_data)
                    else:
                        return (False, {
                            "stage": "Stage 1: DNS Layer",
                            "check_name": "check_domain_age",
                            "message": f"Domain too new: {age_days} days (minimum: {min_age_days})",
                            "failed_fields": ["website"]
                        }, whois_data)
                else:
                    # Calculate whois_updated_days_ago even if creation_date is missing
                    whois_updated_days_ago = None
                    if updated_date:
                        whois_updated_days_ago = (datetime.now() - updated_date).days

                    whois_data = {
                        "age_days": None,
                        "creation_date": None,
                        "registrar": registrar,
                        "nameservers": nameservers,
                        "updated_date": updated_date.isoformat() if updated_date else None,
                        "whois_updated_days_ago": whois_updated_days_ago,
                        "checked": True
                    }
                    return False, {
                        "stage": "Stage 1: DNS Layer",
                        "check_name": "check_domain_age",
                        "message": "Could not determine domain creation date",
                        "failed_fields": ["website"]
                    }, whois_data
            except Exception as e:
                whois_data = {
                    "age_days": None,
                    "creation_date": None,
                    "registrar": None,
                    "nameservers": None,
                    "updated_date": None,
                    "whois_updated_days_ago": None,
                    "checked": False,
                    "error": str(e)
                }
                return False, {
                    "stage": "Stage 1: DNS Layer",
                    "check_name": "check_domain_age",
                    "message": f"WHOIS lookup failed: {str(e)}",
                    "failed_fields": ["website"]
                }, whois_data

        # Run WHOIS lookup in executor to avoid blocking
        loop = asyncio.get_event_loop()
        passed, rejection_reason, whois_data = await loop.run_in_executor(None, get_domain_age_sync, domain)

        # Append WHOIS data to lead
        lead["whois_checked"] = whois_data.get("checked", True)
        lead["domain_age_days"] = whois_data.get("age_days")
        lead["domain_creation_date"] = whois_data.get("creation_date")
        lead["domain_registrar"] = whois_data.get("registrar")
        lead["domain_nameservers"] = whois_data.get("nameservers")
        lead["whois_updated_date"] = whois_data.get("updated_date")
        lead["whois_updated_days_ago"] = whois_data.get("whois_updated_days_ago")
        if "error" in whois_data:
            lead["whois_error"] = whois_data["error"]

        # Cache both result and data
        result = (passed, rejection_reason)
        validation_cache[cache_key] = result
        validation_cache[f"{cache_key}_data"] = whois_data

        return result

    except Exception as e:
        # Append error state
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        lead["whois_error"] = str(e)

        result = (False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": f"Domain age check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_mx_record(lead: dict) -> Tuple[bool, dict]:
    """Check if domain has MX records"""
    website = get_website(lead)
    if not website:
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"mx_record:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]

    try:
        passed, msg = await check_domain_existence(domain)
        if passed:
            result = (True, {})
        else:
            result = (False, {
                "stage": "Stage 1: DNS Layer",
                "check_name": "check_mx_record",
                "message": msg,
                "failed_fields": ["website"]
            })
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": f"MX record check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_spf_dmarc(lead: dict) -> Tuple[bool, dict]:
    """
    Check SPF and DMARC DNS records (SOFT check - always passes, appends data to lead)

    This is a SOFT check that:
    - Checks DNS TXT record for v=spf1
    - Checks DNS TXT record at _dmarc.{domain} for v=DMARC1
    - Checks DMARC policy for p=quarantine or p=reject
    - Appends results to lead but NEVER rejects

    Args:
        lead: Dict containing email/website

    Returns:
        (True, dict): Always passes with empty dict (SOFT check)
    """
    def fail_lead(lead):
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        return lead

    email = get_email(lead)
    if not email:
        # No email to check - append default values
        lead = fail_lead(lead)
        return True, {}

    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead = fail_lead(lead)
            return True, {}
    except (IndexError, AttributeError):
        lead = fail_lead(lead)
        return True, {}

    cache_key = f"spf_dmarc:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        cached_data = validation_cache[cache_key]
        # Apply cached values to lead
        lead["has_spf"] = cached_data.get("has_spf", False)
        lead["has_dmarc"] = cached_data.get("has_dmarc", False)
        lead["dmarc_policy_strict"] = cached_data.get("dmarc_policy_strict", False)
        return True, {}

    try:
        # Initialize results
        has_spf = False
        has_dmarc = False
        dmarc_policy_strict = False

        # Run DNS lookups in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def check_spf_sync(domain_name):
            """Check if domain has SPF record"""
            try:
                txt_records = dns.resolver.resolve(domain_name, "TXT")
                for record in txt_records:
                    txt_string = "".join([s.decode() if isinstance(s, bytes) else s for s in record.strings])
                    if "v=spf1" in txt_string.lower():
                        return True
                return False
            except Exception:
                return False

        def check_dmarc_sync(domain_name):
            """Check if domain has DMARC record and return policy strictness"""
            try:
                dmarc_domain = f"_dmarc.{domain_name}"
                txt_records = dns.resolver.resolve(dmarc_domain, "TXT")
                for record in txt_records:
                    txt_string = "".join([s.decode() if isinstance(s, bytes) else s for s in record.strings])
                    txt_lower = txt_string.lower()

                    if "v=dmarc1" in txt_lower:
                        # Check if policy is strict (quarantine or reject)
                        is_strict = "p=quarantine" in txt_lower or "p=reject" in txt_lower
                        return True, is_strict
                return False, False
            except Exception:
                return False, False

        # Execute DNS checks
        has_spf = await loop.run_in_executor(None, check_spf_sync, domain)
        has_dmarc, dmarc_policy_strict = await loop.run_in_executor(None, check_dmarc_sync, domain)

        # Append results to lead (SOFT check data)
        lead["has_spf"] = has_spf
        lead["has_dmarc"] = has_dmarc
        lead["dmarc_policy_strict"] = dmarc_policy_strict

        # Create informational message
        spf_status = "✓" if has_spf else "✗"
        dmarc_status = "✓" if has_dmarc else "✗"
        policy_status = "✓ (strict)" if dmarc_policy_strict else ("✓ (permissive)" if has_dmarc else "✗")

        message = f"SPF: {spf_status}, DMARC: {dmarc_status}, Policy: {policy_status}"

        # Cache the results
        cache_data = {
            "has_spf": has_spf,
            "has_dmarc": has_dmarc,
            "dmarc_policy_strict": dmarc_policy_strict,
            "message": message
        }
        validation_cache[cache_key] = cache_data

        print(f"📧 SPF/DMARC Check (SOFT): {domain} - {message}")

        # ALWAYS return True (SOFT check never fails)
        return True, {}

    except Exception as e:
        # On any error, append False values and pass
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False

        message = f"SPF/DMARC check error (SOFT - passed): {str(e)}"
        print(f"⚠️ {message}")

        # Cache the error result
        cache_data = {
            "has_spf": False,
            "has_dmarc": False,
            "dmarc_policy_strict": False,
            "message": message
        }
        validation_cache[cache_key] = cache_data

        # ALWAYS return True (SOFT check never fails)
        return True, {}

async def check_head_request(lead: dict) -> Tuple[bool, dict]:
    """Wrapper around existing verify_company function"""
    website = get_website(lead)
    if not website:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"head_request:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]

    try:
        passed, msg = await verify_company(domain)
        if passed:
            result = (True, {})
        else:
            result = (False, {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_head_request",
                "message": f"Website not accessible: {msg}",
                "failed_fields": ["website"]
            })
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": f"HEAD request check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_dnsbl(lead: dict) -> Tuple[bool, dict]:
    """
    Check if lead's email domain is listed in Spamhaus DBL.
    Appends DNSBL data to lead object for reputation scoring.

    Args:
        lead: Dict containing email field

    Returns:
        (bool, dict): (is_valid, rejection_reason_dict)
    """
    email = get_email(lead)
    if not email:
        # Append default DNSBL data
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return False, {
            "stage": "Stage 2: Domain Reputation",
            "check_name": "check_dnsbl",
            "message": "No email provided",
            "failed_fields": ["email"]
        }

    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead["dnsbl_checked"] = False
            lead["dnsbl_blacklisted"] = False
            lead["dnsbl_list"] = None
            return True, {}  # Invalid format handled by other checks
    except (IndexError, AttributeError):
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return True, {}  # Invalid format handled by other checks

    # Use root domain extraction helper
    root_domain = extract_root_domain(domain)
    if not root_domain:
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return True, {}  # Could not extract - handled by other checks

    cache_key = f"dnsbl_{root_domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        cached_result = validation_cache[cache_key]
        # Restore cached DNSBL data to lead
        cached_data = validation_cache.get(f"{cache_key}_data")
        if cached_data:
            lead["dnsbl_checked"] = cached_data.get("checked", True)
            lead["dnsbl_blacklisted"] = cached_data.get("blacklisted", False)
            lead["dnsbl_list"] = cached_data.get("list", "cloudflare_dbl")
            lead["dnsbl_domain"] = cached_data.get("domain", root_domain)
        return cached_result

    try:
        async with API_SEMAPHORE:
            # Perform Cloudflare DNSBL lookup (more reliable than Spamhaus for free tier)
            # Cloudflare has no rate limits and fewer false positives
            query = f"{root_domain}.dbl.cloudflare.com"

            # Run DNS lookup in executor to avoid blocking
            loop = asyncio.get_event_loop()
            def dns_lookup():
                try:
                    print(f"   🔍 DNSBL Query: {query}")
                    answers = dns.resolver.resolve(query, "A")
                    # If we get A records, domain IS blacklisted
                    a_records = [str(rdata) for rdata in answers]

                    # Check for actual blacklist codes (127.0.0.x where x < 128)
                    for record in a_records:
                        if record.startswith("127.0.0."):
                            print(f"   ⚠️  DNSBL returned A records: {a_records} → BLACKLISTED")
                            return True

                    # Any other response is not a confirmed blacklist
                    print(f"   ✅ DNSBL returned A records: {a_records} → CLEAN (not a blacklist code)")
                    return False

                except dns.resolver.NXDOMAIN:
                    # NXDOMAIN = not in blacklist (expected for clean domains)
                    print(f"   ✅ DNSBL returned NXDOMAIN → CLEAN")
                    return False  # No record = domain is clean
                except dns.resolver.NoAnswer:
                    # No answer = not in blacklist
                    print(f"   ✅ DNSBL returned NoAnswer → CLEAN")
                    return False
                except dns.resolver.Timeout:
                    # Timeout = treat as clean (don't block on infrastructure issues)
                    print(f"   ⚠️  DNSBL query timeout for {query} → treating as CLEAN")
                    return False
                except Exception as e:
                    # On any DNS error, default to valid (don't block on infrastructure issues)
                    print(f"   ⚠️  DNS lookup error for {query}: {type(e).__name__}: {e} → treating as CLEAN")
                    return False

            is_blacklisted = await loop.run_in_executor(None, dns_lookup)

            # Append DNSBL data to lead
            lead["dnsbl_checked"] = True
            lead["dnsbl_blacklisted"] = is_blacklisted
            lead["dnsbl_list"] = "cloudflare_dbl"
            lead["dnsbl_domain"] = root_domain

            # Cache the data separately for restoration
            dnsbl_data = {
                "checked": True,
                "blacklisted": is_blacklisted,
                "list": "cloudflare_dbl",
                "domain": root_domain
            }
            validation_cache[f"{cache_key}_data"] = dnsbl_data

            if is_blacklisted:
                result = (False, {
                    "stage": "Stage 2: Domain Reputation",
                    "check_name": "check_dnsbl",
                    "message": f"Domain {root_domain} blacklisted in Cloudflare DBL",
                    "failed_fields": ["email"]
                })
                print(f"❌ DNSBL: Domain {root_domain} found in Cloudflare blacklist")
            else:
                result = (True, {})
                print(f"✅ DNSBL: Domain {root_domain} clean")

            validation_cache[cache_key] = result
            return result

    except Exception as e:
        # On any unexpected error, append error state
        lead["dnsbl_checked"] = True
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = "spamhaus_dbl"
        lead["dnsbl_domain"] = root_domain
        lead["dnsbl_error"] = str(e)

        result = (True, {})  # Don't block on infrastructure issues
        validation_cache[cache_key] = result
        print(f"⚠️ DNSBL check error for {root_domain}: {e}")
        return result


# ============================================================================
# TrueList Batch Email Validation Functions
# ============================================================================
# These functions support batch email verification for improved throughput.
# See tasks9.md for the full migration plan.
# API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation
# ============================================================================

async def submit_truelist_batch(emails: List[str]) -> str:
    """
    Submit a list of emails to TrueList batch API.

    This function submits emails for batch verification. The batch is processed
    asynchronously by TrueList and must be polled for completion using
    poll_truelist_batch().

    API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation

    Args:
        emails: List of email addresses to validate (max 5000 per batch)

    Returns:
        batch_id: UUID of the created batch for polling

    Raises:
        EmailVerificationUnavailableError: If batch submission fails
        ValueError: If no emails provided or API key not configured

    Example:
        batch_id = await submit_truelist_batch(["user1@example.com", "user2@example.com"])
        # Then poll with: results = await poll_truelist_batch(batch_id)
    """
    if not emails:
        raise ValueError("No emails provided for batch validation")

    if not TRUELIST_API_KEY:
        raise EmailVerificationUnavailableError("TRUELIST_API_KEY not configured")

    # Log batch submission
    print(f"\n📧 TrueList Batch: Submitting {len(emails)} emails for validation...")

    try:
        async with aiohttp.ClientSession() as session:
            # TrueList batch API endpoint
            url = "https://api.truelist.io/api/v1/batches"

            # IMPORTANT: TrueList batch API requires multipart/form-data, NOT JSON body
            # The 'data' parameter is a JSON string sent as a form field
            headers = {
                "Authorization": f"Bearer {TRUELIST_API_KEY}",
                # Note: Do NOT set Content-Type header - aiohttp sets it automatically for FormData
            }

            # CRITICAL: TrueList batch API rejects the ENTIRE batch if ANY email
            # doesn't have an @ sign. Pre-filter to avoid this.
            # Emails without @ will be handled separately with immediate rejection.
            valid_emails = [email for email in emails if '@' in email]
            invalid_emails = [email for email in emails if '@' not in email]

            if invalid_emails:
                print(f"   ⚠️  Filtered {len(invalid_emails)} invalid emails (no @ sign)")

            if not valid_emails:
                print(f"   ❌ No valid emails to submit (all filtered)")
                return None  # Return None to indicate no batch was created

            # IMPORTANT: TrueList file upload is currently broken (returns 500)
            # Using JSON data format instead which works correctly
            # JSON format: {"data": [["email1"], ["email2"]], "validation_strategy": "accurate"}

            # Convert emails to JSON array format: [["email1"], ["email2"], ...]
            email_data = [[email] for email in valid_emails]

            # Generate unique batch name to avoid "Duplicate file upload" error
            unique_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

            json_payload = {
                "data": email_data,
                "validation_strategy": TRUELIST_BATCH_STRATEGY,  # "accurate" or "fast"
                "name": unique_name  # Unique name prevents duplicate detection
            }

            print(f"   📤 POST {url} (JSON format)")
            print(f"   📋 Batch name: {unique_name}")
            print(f"   📋 Strategy: {TRUELIST_BATCH_STRATEGY}")
            print(f"   📊 Email count: {len(valid_emails)}")

            async with session.post(
                url,
                headers=headers,
                json=json_payload,  # Use JSON format (file upload returns 500)
                timeout=60,  # 60s timeout for batch submission
                proxy=HTTP_PROXY_URL
            ) as response:

                # Handle error responses
                if response.status == 401:
                    raise EmailVerificationUnavailableError("TrueList API: Invalid or expired API key")
                elif response.status == 402:
                    raise EmailVerificationUnavailableError("TrueList API: Insufficient credits")
                elif response.status == 429:
                    raise EmailVerificationUnavailableError("TrueList API: Rate limited")
                elif response.status >= 500:
                    raise EmailVerificationUnavailableError(f"TrueList API server error: HTTP {response.status}")
                elif response.status != 200:
                    error_text = await response.text()
                    raise EmailVerificationUnavailableError(f"TrueList API error: HTTP {response.status} - {error_text[:200]}")

                # Parse successful response
                data = await response.json()

                batch_id = data.get("id")
                batch_state = data.get("batch_state", "unknown")
                email_count = data.get("email_count", 0)

                if not batch_id:
                    raise EmailVerificationUnavailableError("TrueList API: No batch_id in response")

                print(f"   ✅ Batch created successfully!")
                print(f"   🆔 Batch ID: {batch_id}")
                print(f"   📊 State: {batch_state}")
                print(f"   📧 Emails queued: {email_count}")

                return batch_id

    except aiohttp.ClientError as e:
        raise EmailVerificationUnavailableError(f"TrueList batch submission network error: {str(e)}")
    except asyncio.TimeoutError:
        raise EmailVerificationUnavailableError("TrueList batch submission timed out (60s)")
    except EmailVerificationUnavailableError:
        raise
    except Exception as e:
        raise EmailVerificationUnavailableError(f"TrueList batch submission error: {str(e)}")


async def poll_truelist_batch(batch_id: str) -> Dict[str, dict]:
    """
    Poll TrueList batch until completion or timeout.

    This function polls the batch status every TRUELIST_BATCH_POLL_INTERVAL seconds
    until the batch is complete or TRUELIST_BATCH_TIMEOUT is reached. When complete,
    it downloads and parses the annotated CSV to get per-email results.

    API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation

    Args:
        batch_id: UUID of the batch to poll (from submit_truelist_batch)

    Returns:
        Dict mapping email -> result dict:
        {
            "email@domain.com": {
                "status": "email_ok",      # TrueList email_sub_state
                "passed": True,            # True if email_ok
                "needs_retry": False,      # True if unknown/timeout/error
                "rejection_reason": None   # Rejection reason if failed
            },
            ...
        }

    Raises:
        EmailVerificationUnavailableError: If polling times out or batch fails

    Example:
        batch_id = await submit_truelist_batch(emails)
        results = await poll_truelist_batch(batch_id)
        for email, result in results.items():
            if result["passed"]:
                print(f"{email} is valid")
    """
    import time
    import csv
    from io import StringIO

    if not batch_id:
        raise ValueError("No batch_id provided for polling")

    if not TRUELIST_API_KEY:
        raise EmailVerificationUnavailableError("TRUELIST_API_KEY not configured")

    url = f"https://api.truelist.io/api/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}

    start_time = time.time()
    poll_count = 0

    print(f"\n⏳ TrueList Batch: Polling for completion...")
    print(f"   🆔 Batch ID: {batch_id}")
    print(f"   ⏱️  Poll interval: {TRUELIST_BATCH_POLL_INTERVAL}s")
    print(f"   ⏰ Timeout: {TRUELIST_BATCH_TIMEOUT // 60} minutes")

    while True:
        elapsed = time.time() - start_time

        # Check timeout
        if elapsed >= TRUELIST_BATCH_TIMEOUT:
            raise EmailVerificationUnavailableError(
                f"TrueList batch polling timed out after {TRUELIST_BATCH_TIMEOUT // 60} minutes"
            )

        poll_count += 1

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=30,
                    proxy=HTTP_PROXY_URL
                ) as response:

                    if response.status == 404:
                        raise EmailVerificationUnavailableError(f"TrueList batch not found: {batch_id}")
                    elif response.status == 401:
                        raise EmailVerificationUnavailableError("TrueList API: Invalid or expired API key")
                    elif response.status >= 500:
                        # Server error - retry polling
                        print(f"   ⚠️  Poll #{poll_count}: Server error (HTTP {response.status}), retrying...")
                        await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
                        continue
                    elif response.status != 200:
                        error_text = await response.text()
                        raise EmailVerificationUnavailableError(
                            f"TrueList API error: HTTP {response.status} - {error_text[:200]}"
                        )

                    # Success (HTTP 200) - parse JSON response
                    data = await response.json()

                    batch_state = data.get("batch_state", "unknown")
                    email_count = data.get("email_count", 0)
                    processed_count = data.get("processed_count", 0)
                    ok_count = data.get("ok_count", 0)
                    unknown_count = data.get("unknown_count", 0)

                    # Progress update every 5 polls or when state changes
                    if poll_count % 5 == 1 or batch_state == "completed":
                        progress_pct = (processed_count / email_count * 100) if email_count > 0 else 0
                        print(f"   📊 Poll #{poll_count} ({elapsed:.0f}s): {batch_state} - {processed_count}/{email_count} ({progress_pct:.0f}%)")

                    # Check if batch is complete
                    # CRITICAL: TrueList may say "completed" before all emails are processed!
                    # We must check BOTH state AND processed_count
                    if batch_state == "completed" and processed_count >= email_count:
                        print(f"   ✅ Batch fully completed!")
                        print(f"   📧 Total: {email_count}, OK: {ok_count}, Unknown: {unknown_count}")
                    elif batch_state == "completed" and processed_count < email_count:
                        # TrueList says completed but not all processed - keep polling!
                        print(f"   ⚠️ Batch says 'completed' but only {processed_count}/{email_count} processed - continuing to poll...")
                        await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
                        continue

                    if batch_state == "completed":

                        # CRITICAL: Wait for CSV generation to finish
                        # TrueList's "completed" state doesn't mean CSV is ready
                        # CSV generation happens asynchronously after processing
                        CSV_GENERATION_DELAY = 15  # seconds
                        print(f"   ⏳ Waiting {CSV_GENERATION_DELAY}s for CSV generation...")
                        await asyncio.sleep(CSV_GENERATION_DELAY)

                        # Re-fetch batch data to get fresh CSV URLs
                        print(f"   🔄 Re-fetching batch data for fresh CSV URLs...")
                        async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as refresh_response:
                            if refresh_response.status == 200:
                                data = await refresh_response.json()

                        # ============================================================
                        # FALLBACK: CSV downloads (the /emails endpoint returns 404)
                        # ============================================================

                        # Get the annotated CSV URL - try multiple possible fields
                        annotated_csv_url = (
                            data.get("annotated_csv_url") or
                            data.get("results_url") or
                            data.get("download_url") or
                            data.get("csv_url")
                        )

                        if not annotated_csv_url:
                            # CSV URL is null - TrueList may still be generating it
                            # Wait and retry polling - TrueList CSV generation is ASYNC
                            # and can take 30-60+ seconds after batch shows "completed"
                            CSV_URL_RETRY_DELAY = 10  # seconds
                            CSV_URL_MAX_RETRIES = 6   # Total: 60 seconds of waiting

                            for csv_retry in range(CSV_URL_MAX_RETRIES):
                                print(f"   ⚠️  No CSV URL in response, waiting {CSV_URL_RETRY_DELAY}s and retrying ({csv_retry + 1}/{CSV_URL_MAX_RETRIES})...")
                                await asyncio.sleep(CSV_URL_RETRY_DELAY)

                                # Re-poll the batch
                                async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as retry_response:
                                    if retry_response.status == 200:
                                        retry_data = await retry_response.json()
                                        annotated_csv_url = retry_data.get("annotated_csv_url")
                                        if annotated_csv_url:
                                            print(f"   ✅ CSV URL now available after retry!")
                                            data = retry_data  # Update data for later use
                                            break

                            if not annotated_csv_url:
                                # ================================================================
                                # FALLBACK: Combine multiple CSV files when annotated_csv_url is null
                                # TrueList provides separate CSVs for different email categories:
                                # - highest_reach_csv_url: email_ok + accept_all emails
                                # - only_invalid_csv_url: failed emails (failed_mx, failed_no_mailbox, etc)
                                # By combining these, we can reconstruct all email results!
                                # ================================================================
                                print(f"   ⚠️  annotated_csv_url is null - trying to combine alternative CSVs...")

                                combined_results = {}

                                # Try highest_reach_csv_url (contains ok + accept_all)
                                highest_reach_url = data.get("highest_reach_csv_url")
                                if highest_reach_url:
                                    print(f"   📥 Downloading highest_reach CSV...")
                                    try:
                                        reach_results = await _download_and_parse_batch_csv(highest_reach_url, headers)
                                        if reach_results:
                                            print(f"   ✅ Got {len(reach_results)} emails from highest_reach CSV")
                                            combined_results.update(reach_results)
                                    except Exception as e:
                                        print(f"   ⚠️  highest_reach CSV failed: {str(e)[:50]}")

                                # Try only_invalid_csv_url (contains failed emails)
                                invalid_url = data.get("only_invalid_csv_url")
                                if invalid_url:
                                    print(f"   📥 Downloading only_invalid CSV...")
                                    try:
                                        invalid_results = await _download_and_parse_batch_csv(invalid_url, headers)
                                        if invalid_results:
                                            print(f"   ✅ Got {len(invalid_results)} emails from only_invalid CSV")
                                            combined_results.update(invalid_results)
                                    except Exception as e:
                                        print(f"   ⚠️  only_invalid CSV failed: {str(e)[:50]}")

                                # Try safest_bet_csv_url as additional source (email_ok only)
                                safest_url = data.get("safest_bet_csv_url")
                                if safest_url and len(combined_results) < email_count:
                                    print(f"   📥 Downloading safest_bet CSV...")
                                    try:
                                        safest_results = await _download_and_parse_batch_csv(safest_url, headers)
                                        if safest_results:
                                            # Only add emails we don't already have
                                            new_count = 0
                                            for email, result in safest_results.items():
                                                if email not in combined_results:
                                                    combined_results[email] = result
                                                    new_count += 1
                                            if new_count > 0:
                                                print(f"   ✅ Got {new_count} additional emails from safest_bet CSV")
                                    except Exception as e:
                                        print(f"   ⚠️  safest_bet CSV failed: {str(e)[:50]}")

                                if combined_results:
                                    print(f"   🎉 Combined {len(combined_results)} total email results from alternative CSVs!")
                                    return combined_results

                                # If alternative CSVs also failed, try constructed URLs
                                constructed_url = f"https://api.truelist.io/api/v1/batches/{batch_id}/download"
                                print(f"   ⚠️  Alternative CSVs failed, trying constructed URL: {constructed_url}")

                                try:
                                    results = await _download_and_parse_batch_csv(constructed_url, headers)
                                    if results:
                                        print(f"   ✅ Constructed URL worked! Parsed {len(results)} email results")
                                        return results
                                except Exception as download_err:
                                    print(f"   ⚠️  Constructed URL failed: {str(download_err)[:100]}")

                                # Final fallback: Use batch stats (won't work for individual emails)
                                print(f"   ❌ Could not download CSV results after all fallbacks. Full response:")
                                print(f"   {json.dumps(data, default=str)[:500]}")
                                return _parse_batch_status_from_response(data, batch_id)

                        # Download and parse the CSV
                        print(f"   📥 Downloading results from: {annotated_csv_url[:80]}...")
                        results = await _download_and_parse_batch_csv(annotated_csv_url, headers)

                        print(f"   ✅ Parsed {len(results)} email results")

                        # CRITICAL FIX: If CSV has fewer results than expected, use fallback CSVs
                        if len(results) < email_count:
                            print(f"   ⚠️  CSV only had {len(results)}/{email_count} emails - using fallback CSVs...")

                            combined_results = {}

                            # Try highest_reach_csv_url (contains ok + accept_all)
                            highest_reach_url = data.get("highest_reach_csv_url")
                            if highest_reach_url:
                                print(f"   📥 Fallback: Downloading highest_reach CSV...")
                                try:
                                    reach_results = await _download_and_parse_batch_csv(highest_reach_url, headers)
                                    if reach_results:
                                        print(f"   ✅ Got {len(reach_results)} emails from highest_reach CSV")
                                        combined_results.update(reach_results)
                                except Exception as e:
                                    print(f"   ⚠️  highest_reach CSV failed: {str(e)[:50]}")

                            # Try only_invalid_csv_url (contains failed emails)
                            invalid_url = data.get("only_invalid_csv_url")
                            if invalid_url:
                                print(f"   📥 Fallback: Downloading only_invalid CSV...")
                                try:
                                    invalid_results = await _download_and_parse_batch_csv(invalid_url, headers)
                                    if invalid_results:
                                        print(f"   ✅ Got {len(invalid_results)} emails from only_invalid CSV")
                                        combined_results.update(invalid_results)
                                except Exception as e:
                                    print(f"   ⚠️  only_invalid CSV failed: {str(e)[:50]}")

                            # Try safest_bet_csv_url as additional source
                            safest_url = data.get("safest_bet_csv_url")
                            if safest_url and len(combined_results) < email_count:
                                print(f"   📥 Fallback: Downloading safest_bet CSV...")
                                try:
                                    safest_results = await _download_and_parse_batch_csv(safest_url, headers)
                                    if safest_results:
                                        print(f"   ✅ Got {len(safest_results)} emails from safest_bet CSV")
                                        for email, result in safest_results.items():
                                            if email not in combined_results:
                                                combined_results[email] = result
                                except Exception as e:
                                    print(f"   ⚠️  safest_bet CSV failed: {str(e)[:50]}")

                            if combined_results:
                                print(f"   ✅ Combined fallback CSVs: {len(combined_results)} total emails")
                                return combined_results
                            else:
                                print(f"   ❌ All fallback CSVs failed or empty")
                                # Return empty results - will trigger retry logic

                        return results

                    elif batch_state == "failed":
                        raise EmailVerificationUnavailableError(
                            f"TrueList batch failed: {data.get('error', 'Unknown error')}"
                        )

                    # Still processing - wait and poll again
                    await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)

        except EmailVerificationUnavailableError:
            raise
        except aiohttp.ClientError as e:
            # Network error - retry polling
            print(f"   ⚠️  Poll #{poll_count}: Network error ({str(e)[:50]}), retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
        except asyncio.TimeoutError:
            # Timeout on single request - retry polling
            print(f"   ⚠️  Poll #{poll_count}: Request timeout, retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
        except Exception as e:
            print(f"   ⚠️  Poll #{poll_count}: Unexpected error ({str(e)[:50]}), retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)


async def _download_and_parse_batch_csv(csv_url: str, headers: dict) -> Dict[str, dict]:
    """
    Download and parse TrueList annotated CSV results.

    IMPORTANT: CSV downloads are done WITHOUT proxy because TrueList's
    S3 signed URLs may not work correctly through proxy servers.
    The API calls (submit, poll) still use proxy for rate limit protection.

    Args:
        csv_url: URL to the annotated CSV file
        headers: Auth headers for the request

    Returns:
        Dict mapping email -> result dict
    """
    import csv
    from io import StringIO

    try:
        async with aiohttp.ClientSession() as session:
            # NOTE: NO PROXY for CSV downloads - S3 signed URLs don't work through proxies
            async with session.get(
                csv_url,
                headers=headers,
                timeout=60
                # proxy removed - CSVs must be downloaded directly
            ) as response:

                if response.status != 200:
                    raise EmailVerificationUnavailableError(
                        f"Failed to download batch CSV: HTTP {response.status}"
                    )

                csv_content = await response.text()

                return parse_truelist_batch_csv(csv_content)

    except aiohttp.ClientError as e:
        raise EmailVerificationUnavailableError(f"Failed to download batch CSV: {str(e)}")
    except asyncio.TimeoutError:
        raise EmailVerificationUnavailableError("Batch CSV download timed out")


async def _fetch_batch_email_results(batch_id: str, headers: dict, email_count: int) -> Dict[str, dict]:
    """
    Fetch email results using TrueList's /emails endpoint with pagination.

    This is the CORRECT way to retrieve individual email results per the API docs:
    GET /api/v1/batches/{batch_uuid}/emails

    Args:
        batch_id: UUID of the completed batch
        headers: Auth headers with Bearer token
        email_count: Expected number of emails (for progress reporting)

    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry
    """
    # Define which statuses pass, fail, or need retry
    PASS_STATUSES = {"email_ok"}
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error"}

    results = {}
    page = 1
    per_page = 100  # Maximum allowed per the docs

    print(f"   📥 Fetching email results via /emails endpoint (paginated)...")

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                url = f"https://api.truelist.io/api/v1/batches/{batch_id}/emails?page={page}&per_page={per_page}"

                async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"   ⚠️ /emails endpoint returned HTTP {response.status}: {error_text[:100]}")
                        break

                    data = await response.json()
                    email_addresses = data.get("email_addresses", [])

                    if not email_addresses:
                        # No more results
                        break

                    # Process each email result
                    for email_data in email_addresses:
                        # The email object structure from the API
                        email = email_data.get("email_address", email_data.get("email", "")).lower()
                        if not email:
                            continue

                        email_state = email_data.get("email_state", "unknown")
                        email_sub_state = email_data.get("email_sub_state", email_state)

                        # Determine pass/fail/retry
                        if email_sub_state in PASS_STATUSES:
                            results[email] = {
                                "status": email_sub_state,
                                "passed": True,
                                "needs_retry": False,
                                "rejection_reason": None
                            }
                        elif email_sub_state in RETRY_STATUSES:
                            results[email] = {
                                "status": email_sub_state,
                                "passed": False,
                                "needs_retry": True,
                                "rejection_reason": None
                            }
                        else:
                            # Failed status
                            results[email] = {
                                "status": email_sub_state,
                                "passed": False,
                                "needs_retry": False,
                                "rejection_reason": {
                                    "stage": "Stage 3",
                                    "check_name": "truelist_email_verification",
                                    "message": f"Email verification failed: {email_sub_state}",
                                    "truelist_status": email_sub_state
                                }
                            }

                    print(f"   📄 Page {page}: Got {len(email_addresses)} emails (total so far: {len(results)})")

                    # Check if we got all expected emails
                    if len(results) >= email_count:
                        break

                    # Check if this was the last page (fewer than per_page results)
                    if len(email_addresses) < per_page:
                        break

                    page += 1

                    # Small delay between pages to avoid rate limiting
                    await asyncio.sleep(0.5)

        print(f"   ✅ Fetched {len(results)}/{email_count} email results via API")
        return results

    except Exception as e:
        print(f"   ⚠️ Error fetching email results: {str(e)[:100]}")
        return results


def parse_truelist_batch_csv(csv_content: str) -> Dict[str, dict]:
    """
    Parse TrueList annotated CSV into email -> result mapping.

    Maps TrueList statuses to our internal format:
    - email_ok -> passed=True
    - accept_all, disposable, failed_* -> passed=False
    - unknown, timeout, error -> needs_retry=True

    Args:
        csv_content: Raw CSV content from TrueList

    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry, rejection_reason
    """
    import csv
    from io import StringIO

    results = {}

    # Define which statuses pass, fail, or need retry
    PASS_STATUSES = {"email_ok"}
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error"}
    # All other statuses are considered failures

    try:
        reader = csv.DictReader(StringIO(csv_content))

        # Debug: Print first few lines and column names
        rows = list(reader)
        if rows:
            print(f"   📋 CSV columns: {list(rows[0].keys())}")
            print(f"   📋 First row: {dict(list(rows[0].items())[:5])}")  # First 5 fields
        else:
            print(f"   ⚠️ CSV is empty! Content preview: {csv_content[:200]}")

        for row in rows:
            # TrueList CSV has columns: Try multiple column name formats
            # API may use different column names: "Email Address", "email", "Email", etc.
            email = (row.get("Email Address") or row.get("email") or
                     row.get("Email") or row.get("email_address") or "").strip().lower()

            if not email:
                continue

            # Get the detailed status - try multiple column name formats
            # TrueList uses "Email Sub-State" or "Email State"
            status = (row.get("Email Sub-State") or row.get("email_sub_state") or
                      row.get("Email State") or row.get("email_state") or
                      row.get("sub_state") or row.get("state") or "unknown")
            status = status.lower() if status else "unknown"

            # Determine pass/fail/retry
            if status in PASS_STATUSES:
                results[email] = {
                    "status": status,
                    "passed": True,
                    "needs_retry": False,
                    "rejection_reason": None
                }
            elif status in RETRY_STATUSES:
                results[email] = {
                    "status": status,
                    "passed": False,
                    "needs_retry": True,
                    "rejection_reason": None  # Don't reject - will retry
                }
            else:
                # Failed status - build rejection reason
                rejection_reason = _build_email_rejection_reason(status)
                results[email] = {
                    "status": status,
                    "passed": False,
                    "needs_retry": False,
                    "rejection_reason": rejection_reason
                }

        return results

    except Exception as e:
        raise EmailVerificationUnavailableError(f"Failed to parse batch CSV: {str(e)}")


def _parse_batch_status_from_response(data: dict, batch_id: str) -> Dict[str, dict]:
    """
    Fallback: Parse batch results from API response when CSV URL is not available.

    This is used when annotated_csv_url is missing from the response.
    Returns aggregate counts but may not have per-email details.

    Args:
        data: Batch API response data
        batch_id: Batch ID for logging

    Returns:
        Dict with limited results (may need alternative approach)
    """
    print(f"   ⚠️  Using fallback batch parsing (no CSV URL)")

    # This is a fallback - in practice, TrueList should always provide the CSV URL
    # Log a warning and return empty results to trigger retry logic
    email_count = data.get("email_count", 0)
    ok_count = data.get("ok_count", 0)
    unknown_count = data.get("unknown_count", 0)

    print(f"   📊 Batch stats: {email_count} total, {ok_count} ok, {unknown_count} unknown")
    print(f"   ⚠️  Cannot map to individual emails without CSV - returning empty results")

    # Return empty dict - the orchestrator should handle this case
    return {}


def _build_email_rejection_reason(status: str) -> dict:
    """
    Build a rejection reason dict for a failed email status.

    Maps TrueList statuses to user-friendly rejection messages.

    Args:
        status: TrueList email_sub_state value

    Returns:
        Rejection reason dict compatible with our validation format
    """
    # Map TrueList statuses to rejection messages
    # Note: TrueList uses both "disposable" and "is_disposable" for different cases
    status_messages = {
        "accept_all": "Email is catch-all/accept-all (instant rejection)",
        "disposable": "Email is from a disposable provider",
        "is_disposable": "Email is from a disposable provider",
        "failed_no_mailbox": "Mailbox does not exist",
        "failed_syntax_check": "Invalid email syntax",
        "failed_mx_check": "Domain has no MX records (cannot receive email)",
        "role": "Email is a role-based address (e.g., info@, support@)",
        "invalid": "Email is invalid",
        "spam_trap": "Email is a known spam trap",
        "complainer": "Email owner is a known complainer",
        "ok_for_all": "Email domain accepts all emails (catch-all)",
    }

    message = status_messages.get(status, f"Email status '{status}' (only 'email_ok' accepted)")

    return {
        "stage": "Stage 3: TrueList Batch",
        "check_name": "truelist_batch_validation",
        "message": message,
        "failed_fields": ["email"],
        "truelist_status": status
    }


async def submit_and_poll_truelist(emails: List[str]) -> Tuple[str, Dict[str, dict]]:
    """
    Submit batch and poll for results (combined for background task).

    This is a helper function that combines submit_truelist_batch() and
    poll_truelist_batch() for use with asyncio.create_task().

    Args:
        emails: List of email addresses to validate

    Returns:
        Tuple of (batch_id, results_dict) where results_dict maps email -> result
        batch_id is returned so caller can delete the batch before retrying
    """
    batch_id = await submit_truelist_batch(emails)
    results = await poll_truelist_batch(batch_id)
    return batch_id, results


async def verify_emails_inline(emails: List[str]) -> Dict[str, dict]:
    """
    Verify emails using TrueList's INLINE verification API (not batch).

    This is a FALLBACK for emails that TrueList's batch API silently drops.
    Some enterprise domains (spglobal.com, jacobs.com, etc.) work with inline
    verification but not batch verification.

    Rate limit: 10 requests/second per TrueList docs.
    Each request can verify up to 3 emails.

    Args:
        emails: List of email addresses to verify

    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry
    """
    if not TRUELIST_API_KEY:
        print("   ⚠️ TRUELIST_API_KEY not configured for inline verification")
        return {email: {"needs_retry": True, "error": "No API key"} for email in emails}

    results = {}
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}

    # TrueList inline API accepts up to 3 emails per request (space-separated)
    BATCH_SIZE = 3
    PASS_STATUSES = {"email_ok"}  # Only email_ok passes - accept_all is rejected
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error", "failed_greylisted"}

    print(f"   🔍 Inline verification for {len(emails)} emails (TrueList batch fallback)...")
    import time as _time
    _start = _time.time()

    try:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(emails), BATCH_SIZE):
                batch = emails[i:i+BATCH_SIZE]
                email_param = " ".join(batch)

                url = f"https://api.truelist.io/api/v1/verify_inline?email={email_param}"

                try:
                    async with session.post(url, headers=headers, timeout=35, proxy=HTTP_PROXY_URL) as response:
                        if response.status == 429:
                            print(f"   ⚠️ Rate limited, waiting 2s...")
                            await asyncio.sleep(2)
                            continue

                        if response.status != 200:
                            error_text = await response.text()
                            print(f"   ⚠️ Inline verify failed ({response.status}): {error_text[:50]}")
                            for email in batch:
                                results[email.lower()] = {"needs_retry": True, "error": f"HTTP {response.status}"}
                            continue

                        data = await response.json()
                        email_results = data.get("emails", [])

                        # DEBUG: Log first response to see actual structure
                        if i == 0 and email_results:
                            print(f"   📋 Inline API first response: {email_results[0]}")

                        for email_data in email_results:
                            # TrueList inline uses "address" not "email_address"
                            email = email_data.get("address", email_data.get("email_address", email_data.get("email", ""))).lower()
                            if not email:
                                continue

                            email_state = email_data.get("email_state", "unknown")
                            email_sub_state = email_data.get("email_sub_state", email_state)

                            # DEBUG: Log non-email_ok statuses
                            if email_sub_state != "email_ok":
                                print(f"   📋 Inline status: {email} -> {email_state}/{email_sub_state}")

                            if email_sub_state in PASS_STATUSES:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": True,
                                    "needs_retry": False,
                                    "rejection_reason": None
                                }
                            elif email_sub_state in RETRY_STATUSES:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": False,
                                    "needs_retry": True,
                                    "rejection_reason": None
                                }
                            else:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": False,
                                    "needs_retry": False,
                                    "rejection_reason": {
                                        "stage": "Stage 3",
                                        "check_name": "truelist_inline_verification",
                                        "message": f"Email verification failed: {email_sub_state}",
                                        "truelist_status": email_sub_state
                                    }
                                }

                except asyncio.TimeoutError:
                    print(f"   ⚠️ Inline verify timeout for: {batch}")
                    for email in batch:
                        results[email.lower()] = {"needs_retry": True, "error": "timeout"}
                except Exception as e:
                    print(f"   ⚠️ Inline verify error: {e}")
                    for email in batch:
                        results[email.lower()] = {"needs_retry": True, "error": str(e)}

                # Rate limit: 10 req/sec = 100ms between requests
                await asyncio.sleep(0.15)

        passed = sum(1 for r in results.values() if r.get("passed"))
        elapsed = _time.time() - _start
        print(f"   ✅ Inline verification: {passed}/{len(emails)} passed ({elapsed:.1f}s)")
        return results

    except Exception as e:
        print(f"   ⚠️ Inline verification error: {e}")
        return {email.lower(): {"needs_retry": True, "error": str(e)} for email in emails}


async def retry_truelist_batch(emails: List[str], prev_batch_id: str = None) -> Tuple[str, Dict[str, dict]]:
    """
    Submit a retry batch and poll for results.

    IMPORTANT: If prev_batch_id is provided, deletes it first to clear
    TrueList's duplicate detection before submitting the retry.

    Args:
        emails: List of email addresses to retry
        prev_batch_id: Optional batch_id from previous retry to delete first

    Returns:
        Tuple of (batch_id, results_dict)
        On error, returns (None, {email: needs_retry=True})
    """
    try:
        # Delete previous retry batch if provided (clears duplicate detection)
        if prev_batch_id:
            await delete_truelist_batch(prev_batch_id)

        batch_id = await submit_truelist_batch(emails)
        results = await poll_truelist_batch(batch_id)
        return batch_id, results
    except Exception as e:
        print(f"   ⚠️ Retry batch error: {e}")
        # On error, mark all as needing retry
        return None, {email: {"needs_retry": True, "error": str(e)} for email in emails}


async def delete_truelist_batch(batch_id: str) -> bool:
    """
    Delete a TrueList batch.

    IMPORTANT: This must be called before retrying emails that were in the batch.
    TrueList detects duplicate email content and rejects re-submissions.
    Deleting the batch clears TrueList's duplicate detection for those emails.

    Args:
        batch_id: The batch ID to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    if not batch_id:
        return False

    url = f"https://api.truelist.io/api/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                if response.status == 204:
                    print(f"   🗑️ Deleted batch {batch_id[:8]}... (clearing duplicate detection)")
                    return True
                else:
                    print(f"   ⚠️ Failed to delete batch {batch_id[:8]}... (status {response.status})")
                    return False
    except Exception as e:
        print(f"   ⚠️ Error deleting batch: {str(e)[:50]}")
        return False


async def delete_all_truelist_batches() -> int:
    """
    Delete ALL TrueList batches to clear duplicate detection.

    CRITICAL: TrueList detects duplicate emails across ALL batches ever submitted.
    Even if a batch is "completed", TrueList remembers the emails and may return
    incomplete CSV results for subsequent batches containing the same emails.

    This function queries all batches and deletes them one by one.
    Should be called before submitting a new batch in each epoch.

    Returns:
        Number of batches deleted
    """
    url = "https://api.truelist.io/api/v1/batches"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    deleted_count = 0

    try:
        async with aiohttp.ClientSession() as session:
            # Get list of all batches
            async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                if response.status != 200:
                    print(f"   ⚠️ Failed to list batches (status {response.status})")
                    return 0

                data = await response.json()
                batches = data.get("batches", [])

                if not batches:
                    print(f"   ✅ No old batches to delete")
                    return 0

                print(f"   🗑️ Deleting {len(batches)} old TrueList batches to clear duplicate detection...")

                # Delete each batch
                for batch in batches:
                    batch_id = batch.get("id")
                    if batch_id:
                        delete_url = f"{url}/{batch_id}"
                        try:
                            async with session.delete(delete_url, headers=headers, timeout=10, proxy=HTTP_PROXY_URL) as del_response:
                                if del_response.status == 204:
                                    deleted_count += 1
                        except Exception:
                            pass  # Silently skip failed deletes

                print(f"   ✅ Deleted {deleted_count}/{len(batches)} batches")
                return deleted_count

    except Exception as e:
        print(f"   ⚠️ Error deleting batches: {str(e)[:50]}")
        return deleted_count


async def run_centralized_truelist_batch(leads: List[dict]) -> Dict[str, dict]:
    """
    COORDINATOR ONLY: Run TrueList batch on ALL leads at once.

    This function extracts all emails from leads, submits them to TrueList,
    handles retries (up to 3 times), and falls back to inline verification.

    The coordinator calls this BEFORE distributing leads to workers.
    Workers then receive the precomputed results with their leads.

    Flow:
    1. Extract all valid emails from leads
    2. Delete old TrueList batches (clean slate)
    3. Submit batch with all emails
    4. Poll for completion
    5. Retry any emails with errors (up to 3 times total)
    6. Fall back to inline verification for remaining errors
    7. Return complete results dict

    Args:
        leads: List of ALL lead dicts from gateway (e.g., 2700 leads)

    Returns:
        Dict mapping email (lowercase) -> result dict with:
        - passed: bool
        - status: str (email_ok, failed_*, etc.)
        - needs_retry: bool (if unresolved)
        - rejection_reason: dict (if failed)

    NOTE: This function is ONLY called by the coordinator.
    Workers should use precomputed_email_results parameter of run_batch_automated_checks.
    """
    print(f"\n{'='*60}")
    print(f"📧 COORDINATOR: Centralized TrueList batch for {len(leads)} leads")
    print(f"{'='*60}")

    start_time = time.time()

    # ========================================================================
    # Step 1: Extract all valid emails
    # ========================================================================
    emails = []
    email_to_lead_idx = {}  # Track which lead each email came from (for debugging)

    for i, lead in enumerate(leads):
        # Handle both formats: {"lead_blob": {...}} wrapper OR flat lead dict
        lead_blob = lead.get("lead_blob", lead) if isinstance(lead, dict) else lead
        email = get_email(lead_blob)
        if email and '@' in email:
            email_lower = email.lower()
            emails.append(email_lower)
            email_to_lead_idx[email_lower] = i

    print(f"   📧 Extracted {len(emails)} valid emails from {len(leads)} leads")

    if not emails:
        print(f"   ⚠️ No valid emails found - returning empty results")
        return {}

    # ========================================================================
    # Step 2: Clean up old TrueList batches
    # ========================================================================
    print(f"   🧹 Cleaning up old TrueList batches...")
    await delete_all_truelist_batches()

    # ========================================================================
    # Step 3: Submit batch and poll (with retries)
    # ========================================================================
    email_results = {}
    batch_id = None

    # Try batch up to 3 times total
    for batch_attempt in range(3):
        try:
            print(f"   🚀 Submitting TrueList batch (attempt {batch_attempt + 1}/3) for {len(emails)} emails...")

            batch_id = await submit_truelist_batch(emails)
            results = await poll_truelist_batch(batch_id)

            # Merge results
            email_results.update(results)

            # Check for emails that need retry
            needs_retry = []
            for email in emails:
                result = email_results.get(email)
                if result is None or result.get("needs_retry"):
                    needs_retry.append(email)

            print(f"   ✅ Batch {batch_attempt + 1} complete: {len(results)} results, {len(needs_retry)} need retry")

            if not needs_retry:
                # All emails resolved
                break

            if batch_attempt < 2:
                # More retries available
                print(f"   🔄 Retrying {len(needs_retry)} emails in 10s...")

                # Delete batch before retry (clears duplicate detection)
                if batch_id:
                    await delete_truelist_batch(batch_id)
                    batch_id = None

                await asyncio.sleep(10)
                emails = needs_retry  # Only retry failed emails

        except Exception as e:
            print(f"   ❌ TrueList batch attempt {batch_attempt + 1} failed: {e}")

            # Delete batch before retry
            if batch_id:
                try:
                    await delete_truelist_batch(batch_id)
                except:
                    pass
                batch_id = None

            if batch_attempt < 2:
                await asyncio.sleep(10 * (batch_attempt + 1))

    # ========================================================================
    # Step 4: Inline fallback for remaining errors
    # ========================================================================
    needs_inline = []
    for email in emails:
        result = email_results.get(email)
        if result is None or result.get("needs_retry"):
            needs_inline.append(email)

    if needs_inline:
        print(f"   🔄 Falling back to inline verification for {len(needs_inline)} emails...")

        try:
            inline_results = await verify_emails_inline(needs_inline)
            email_results.update(inline_results)
            print(f"   ✅ Inline verification complete: {len(inline_results)} results")
        except Exception as e:
            print(f"   ❌ Inline verification failed: {e}")
            # Mark remaining as unresolved
            for email in needs_inline:
                if email not in email_results or email_results[email].get("needs_retry"):
                    email_results[email] = {
                        "needs_retry": True,
                        "error": f"All verification methods failed: {str(e)}"
                    }

    # ========================================================================
    # Step 5: Summary
    # ========================================================================
    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)

    passed = sum(1 for r in email_results.values() if r.get("passed"))
    failed = sum(1 for r in email_results.values() if not r.get("passed") and not r.get("needs_retry"))
    unresolved = sum(1 for r in email_results.values() if r.get("needs_retry"))

    print(f"\n{'='*60}")
    print(f"📊 CENTRALIZED TRUELIST COMPLETE")
    print(f"{'='*60}")
    print(f"   📦 Total leads from gateway: {len(leads)}")
    print(f"   📧 Total emails processed: {len(email_results)}")
    print(f"   ✅ Passed (email_ok): {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Unresolved: {unresolved}")
    print(f"   ⏱️  TIME: {elapsed_mins}m {elapsed_secs}s ({elapsed:.1f} seconds total)")
    print(f"{'='*60}\n")

    return email_results


# ============================================================================
# Legacy functions - DO NOT TOUCH (maintained for backward compatibility)
# ============================================================================

async def load_email_cache():
    if os.path.exists(EMAIL_CACHE_FILE):
        try:
            with open(EMAIL_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

async def save_email_cache(cache):
    try:
        with open(EMAIL_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

# EMAIL_CACHE = asyncio.run(load_email_cache())  # Disabled to avoid event loop issues
EMAIL_CACHE = {}

async def is_disposable_email(email: str) -> Tuple[bool, str]:
    domain = email.split("@")[1].lower() if "@" in email else ""
    # Return True if email IS disposable, False if NOT disposable
    is_disposable = domain in DISPOSABLE_DOMAINS
    return is_disposable, "Disposable domain" if is_disposable else "Not disposable"

async def check_domain_existence(domain: str) -> Tuple[bool, str]:
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(domain, "MX"))
        return True, "Domain has MX records"
    except Exception as e:
        return False, f"Domain check failed: {str(e)}"

async def verify_company(company_domain: str) -> Tuple[bool, str]:
    """
    Verify company website is accessible.

    Strategy: Try HEAD first (lightweight), fall back to GET if HEAD fails.
    Many enterprise sites (Intuit, 3M, etc.) block HEAD requests but work with GET.
    Uses browser User-Agent to avoid anti-bot blocking.
    Uses custom SSL context with broader cipher support for enterprise sites (Hartford, etc.)
    """
    import ssl

    if not company_domain:
        return False, "No domain provided"
    if not company_domain.startswith(("http://", "https://")):
        company_domain = f"https://{company_domain}"

    # Status codes that indicate website exists (pass immediately)
    # 429 = Too Many Requests (rate limiting/bot protection) - proves site exists, just blocking automated requests
    PASS_STATUS_CODES = {200, 301, 302, 307, 308, 401, 403, 405, 429, 500, 502, 503}

    # Browser User-Agent to avoid anti-bot blocking (3M, etc.)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Create custom SSL context with broader cipher support
    # Some enterprise sites (Hartford, etc.) have strict SSL configs that reject default ciphers
    ssl_context = ssl.create_default_context()
    # Allow older TLS versions for compatibility with enterprise sites
    ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')
    # Add additional options for maximum compatibility
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    # Create connector with custom SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    try:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            # Try HEAD request first (lightweight)
            head_status = None
            head_error = None
            try:
                async with session.head(company_domain, timeout=10, allow_redirects=True) as response:
                    head_status = response.status
                    if head_status in PASS_STATUS_CODES:
                        return True, f"Website accessible (HEAD: {head_status})"
            except aiohttp.ClientError as e:
                head_error = str(e) or "connection_error"
                # Handle large enterprise headers - pass immediately
                if "Header value is too long" in head_error or "Got more than" in head_error:
                    return True, "Website accessible (large enterprise headers detected)"
            except asyncio.TimeoutError:
                head_error = "timeout"
            except Exception as e:
                head_error = str(e) or type(e).__name__

            # HEAD failed or returned non-pass status - try GET as fallback
            # Many enterprise sites (Intuit, 3M) block HEAD but allow GET
            try:
                async with session.get(company_domain, timeout=10, allow_redirects=True) as response:
                    get_status = response.status
                    if get_status in PASS_STATUS_CODES:
                        return True, f"Website accessible (GET fallback: {get_status})"
                    else:
                        # Both HEAD and GET returned non-pass status
                        return False, f"Website not accessible (HEAD: {head_status}, GET: {get_status})"
            except aiohttp.ClientError as e:
                get_error = str(e) or "connection_error"
                # Handle large enterprise headers on GET too
                if "Header value is too long" in get_error or "Got more than" in get_error:
                    return True, "Website accessible (large enterprise headers detected)"
                # Both HEAD and GET failed
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: {get_error})"
            except asyncio.TimeoutError:
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: timeout)"
            except Exception as e:
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: {str(e) or type(e).__name__})"
    except Exception as e:
        return False, f"Website inaccessible: {str(e)}"
