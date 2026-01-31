"""
Stage 4: Person Verification
=============================
Verifies the PERSON exists and matches the claimed data.

Checks performed:
- LinkedIn URL matching (Q4 + Q1 fallback)
- Name verification
- Company verification
- Location verification (structured city, area mapping, Q3 fallback)
- Role verification (rule-based + LLM fallback)

Usage in automated_checks.py:
    from validator_models.stage4_person_verification import run_lead_validation_stage4

    # In check_linkedin_gse:
    result = await run_lead_validation_stage4(lead)
    if not result['passed']:
        return False, result['rejection_reason']
"""

import os
import re
import asyncio
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from .stage4_helpers import (
    check_name_in_result,
    check_company_in_result,
    extract_location_from_text,
    extract_role_from_result,
    check_locations_match,
    check_q3_location_fallback,
    validate_role_rule_based,
    validate_role_with_llm,
    get_linkedin_id,
    is_valid_state,
    is_area_in_mappings,
    is_city_in_area_approved,
    is_ambiguous_city,
    is_english_word_city,
    _verify_state_or_country_for_strict_validation,
    should_reject_city_match,
    GEO_LOOKUP,
    CITY_EQUIVALENTS,
)

# API Keys from environment
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")

# Thread pool for running sync functions
_executor = ThreadPoolExecutor(max_workers=4)


async def search_google_async(query: str, api_key: str, max_results: int = 10) -> Tuple[list, Optional[str]]:
    """Async wrapper for ScrapingDog Google search."""
    import requests

    def _search():
        try:
            resp = requests.get('https://api.scrapingdog.com/google', params={
                'api_key': api_key,
                'query': query,
                'results': max_results
            }, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return [{
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', ''),
                    'link': r.get('link', ''),
                    'missing': r.get('missing', [])
                } for r in data.get('organic_results', [])], None
            else:
                return [], f"HTTP {resp.status_code}"
        except Exception as e:
            return [], str(e)[:100]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _search)


async def run_lead_validation_stage4(
    lead: Dict[str, Any],
    scrapingdog_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run lead validation for Stage 4 integration.

    Performs:
    1. Q4 search: "{name}" "{company}" linkedin location
    2. Q1 fallback: site:linkedin.com/in/{slug} (if URL not found)
    3. URL match verification
    4. Name check
    5. Company check
    6. Location check with Q3 fallback

    Args:
        lead: Lead data dict with keys:
            - full_name / Full_name / Full Name
            - company / business / Company
            - linkedin / Linkedin
            - city / City
            - state / State
            - country / Country
            - role / Role / job_title
            - email / Email (optional)
        scrapingdog_api_key: ScrapingDog API key (uses env var if not provided)
        openrouter_api_key: OpenRouter API key (uses env var if not provided)

    Returns:
        {
            'passed': bool,
            'rejection_reason': dict or None,
            'data': {
                'url_matched': bool,
                'name_matched': bool,
                'company_matched': bool,
                'location_passed': bool,
                'location_method': str,
                'extracted_location': str,
                'q3_called': bool,
                'q3_result': str,
                'query_used': str,  # Q4, Q4+Q1, Q4+Q3, Q4+Q1+Q3
                'search_results': list
            }
        }
    """
    api_key = scrapingdog_api_key or SCRAPINGDOG_API_KEY

    # Normalize field names
    full_name = (lead.get("full_name") or lead.get("Full_name") or
                 lead.get("Full Name") or lead.get("name") or "").strip()
    company = (lead.get("company") or lead.get("business") or
               lead.get("Company") or lead.get("Business") or "").strip()
    linkedin_url = (lead.get("linkedin") or lead.get("Linkedin") or
                    lead.get("linkedin_url") or "").strip()
    city = (lead.get("city") or lead.get("City") or "").strip()
    state = (lead.get("state") or lead.get("State") or "").strip()
    country = (lead.get("country") or lead.get("Country") or "").strip()
    email = (lead.get("email") or lead.get("Email") or "").strip()

    result = {
        'passed': False,
        'rejection_reason': None,
        'data': {
            'role_format_valid': False,
            'url_matched': False,
            'name_matched': False,
            'company_matched': False,
            'location_passed': False,
            'location_method': None,
            'extracted_location': None,
            'extracted_role': None,  # For Stage 5 to use
            'role_verified': False,  # For Stage 5 to check
            'role_method': None,  # For Stage 5 to check
            'q3_called': False,
            'q3_result': None,
            'query_used': None,
            'search_results': []
        }
    }

    # Validate inputs
    if not full_name:
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": "Missing full_name",
            "failed_fields": ["full_name"]
        }
        return result

    if not company:
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": "Missing company",
            "failed_fields": ["company"]
        }
        return result

    if not linkedin_url:
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": "Missing linkedin URL",
            "failed_fields": ["linkedin"]
        }
        return result

    expected_lid = get_linkedin_id(linkedin_url)
    if not expected_lid:
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": "Invalid LinkedIn URL format",
            "failed_fields": ["linkedin"]
        }
        return result

    # Get role for format check
    role = (lead.get("role") or lead.get("Role") or
            lead.get("job_title") or lead.get("Job_title") or "").strip()

    # STEP 0: Role Format Check - NOW DONE AT GATEWAY (submit.py)
    # Gateway's check_role_sanity() handles all format checks including:
    # - Length, spam patterns, typos, placeholders
    # - Person's name in role, company name in role
    # - Marketing taglines, geographic endings
    # Leads with invalid role format never reach Stage 4
    result['data']['role_format_valid'] = True  # Passed gateway check

    # STEP 1: Q4 Search
    print(f"   🔍 Q4: Searching \"{full_name}\" \"{company}\" linkedin location")
    q4_query = f'"{full_name}" "{company}" linkedin location'
    q4_results, q4_error = await search_google_async(q4_query, api_key)

    queries_used = ['Q4']
    all_results = q4_results.copy()

    # Find URL-matched result
    url_matched_result = None
    for r in q4_results:
        if get_linkedin_id(r.get('link', '')) == expected_lid:
            url_matched_result = r
            break

    # STEP 2: Q1 Fallback if URL not found
    if not url_matched_result:
        print(f"   🔍 Q1: Fallback site:linkedin.com/in/{expected_lid}")
        q1_query = f'site:linkedin.com/in/{expected_lid}'
        q1_results, q1_error = await search_google_async(q1_query, api_key)
        queries_used.append('Q1')
        all_results.extend(q1_results)

        for r in q1_results:
            if get_linkedin_id(r.get('link', '')) == expected_lid:
                url_matched_result = r
                break

    result['data']['search_results'] = all_results

    # STEP 3: URL Check
    if not url_matched_result:
        result['data']['query_used'] = '+'.join(queries_used)
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": f"LinkedIn URL {linkedin_url} not found in search results",
            "failed_fields": ["linkedin"]
        }
        return result

    result['data']['url_matched'] = True
    print(f"   ✅ URL matched: {linkedin_url}")

    # Extract role from matched result (for Stage 5 to use)
    extracted_role = extract_role_from_result(url_matched_result, full_name, company)
    if extracted_role:
        result['data']['extracted_role'] = extracted_role
        print(f"   📝 Extracted role: {extracted_role}")

    # STEP 4: Name Check
    if not check_name_in_result(full_name, url_matched_result, linkedin_url):
        result['data']['query_used'] = '+'.join(queries_used)
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": f"Name '{full_name}' not found in LinkedIn result",
            "failed_fields": ["full_name"]
        }
        return result

    result['data']['name_matched'] = True
    print(f"   ✅ Name matched: {full_name}")

    # STEP 5: Company Check
    if not check_company_in_result(company, url_matched_result, email):
        result['data']['query_used'] = '+'.join(queries_used)
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": f"Company '{company}' not found in LinkedIn result",
            "failed_fields": ["company"]
        }
        return result

    result['data']['company_matched'] = True
    print(f"   ✅ Company matched: {company}")

    # STEP 6: Location Check
    full_text = f"{url_matched_result.get('title', '')} {url_matched_result.get('snippet', '')}"
    extracted_loc = extract_location_from_text(full_text)

    # Validate extracted state for structured locations
    structured_loc_valid = False
    if extracted_loc and ',' in extracted_loc:
        parts = extracted_loc.split(',')
        if len(parts) >= 2:
            state_part = parts[1].strip()

            # Case 1: Valid US state - this is a structured US location
            if state_part and is_valid_state(state_part):
                structured_loc_valid = True

            # Case 2: Check if it's an international location (use GEO_LOOKUP)
            elif state_part:
                # Check parts[2] for country if exists, or parts[1] if 2-part location
                valid_countries = [c.lower() for c in GEO_LOOKUP.get('countries', [])]
                country_part = parts[2].strip().lower() if len(parts) >= 3 else parts[1].strip().lower()

                if country_part in valid_countries and country_part != 'united states':
                    structured_loc_valid = True
                else:
                    # Check for metro area patterns which are still valid
                    is_metro = any(x.lower() in extracted_loc.lower() for x in ['Area', 'Metropolitan', 'Greater'])
                    if not is_metro:
                        extracted_loc = ''  # Invalid location format
            # Case 3: Empty state_part - not a valid structured location
            # structured_loc_valid remains False

    result['data']['extracted_location'] = extracted_loc

    location_passed = False
    location_method = None

    # 6a. Structured city check (validates BOTH city AND state)
    if structured_loc_valid and extracted_loc and city:
        parts = extracted_loc.split(',')
        ext_city = parts[0].strip().lower()
        ext_state = parts[1].strip().lower() if len(parts) >= 2 else ''
        claimed_city = city.lower().strip()
        claimed_state = state.lower().strip() if state else ''

        # Normalize city names using equivalents (e.g., Bangalore = Bengaluru)
        ext_city_norm = CITY_EQUIVALENTS.get(ext_city, ext_city)
        claimed_city_norm = CITY_EQUIVALENTS.get(claimed_city, claimed_city)
        city_match = (claimed_city_norm in ext_city_norm or ext_city_norm in claimed_city_norm or
                      claimed_city in ext_city or ext_city in claimed_city)

        # Normalize state abbreviations for comparison
        state_abbr = GEO_LOOKUP.get('state_abbr', {})
        if ext_state in state_abbr:
            ext_state_full = state_abbr[ext_state].lower()
        else:
            ext_state_full = ext_state
        if claimed_state in state_abbr:
            claimed_state_full = state_abbr[claimed_state].lower()
        else:
            claimed_state_full = claimed_state

        state_match = (
            not claimed_state or  # No claimed state - skip state check
            claimed_state_full in ext_state_full or
            ext_state_full in claimed_state_full or
            claimed_state in ext_state or
            ext_state in claimed_state
        )

        if city_match and state_match:
            location_passed = True
            location_method = 'structured_city_match'
        elif not city_match:
            # DIRECT FAIL - city mismatch, no Q3
            result['data']['query_used'] = '+'.join(queries_used)
            result['data']['location_method'] = 'city_mismatch'
            result['rejection_reason'] = {
                "stage": "Stage 4: Lead Validation",
                "check_name": "lead_validation_stage4",
                "message": f"City mismatch: extracted '{ext_city}' but claimed '{city}'",
                "failed_fields": ["city"],
                "extracted_location": extracted_loc,
                "claimed_city": city
            }
            return result
        elif not state_match:
            # DIRECT FAIL - state mismatch, no Q3
            result['data']['query_used'] = '+'.join(queries_used)
            result['data']['location_method'] = 'state_mismatch'
            result['rejection_reason'] = {
                "stage": "Stage 4: Lead Validation",
                "check_name": "lead_validation_stage4",
                "message": f"State mismatch: extracted '{ext_state}' but claimed '{state}'",
                "failed_fields": ["state"],
                "extracted_location": extracted_loc,
                "claimed_state": state
            }
            return result

    # 6b. Other location checks
    if not location_passed and city:
        gt_location = f"{city}, {state}, {country}".strip(', ')
        city_lower = city.lower().strip()

        # Flexible location match
        if not structured_loc_valid and extracted_loc:
            loc_match, loc_method = check_locations_match(extracted_loc, gt_location, full_text)
            if loc_match:
                # Ambiguous/English word cities need state verification even on
                # non-structured match (e.g. "Richmond" alone could be VA, CA, KY...)
                needs_strict = is_ambiguous_city(city_lower) or is_english_word_city(city_lower)
                if needs_strict:
                    has_state = _verify_state_or_country_for_strict_validation(
                        city_lower, state, country, full_text, linkedin_url)
                    if has_state:
                        location_passed = True
                        location_method = loc_method
                else:
                    location_passed = True
                    location_method = loc_method

        # City fallback
        if not location_passed:
            if re.search(r'\b' + re.escape(city_lower) + r'\b', full_text.lower()):
                # Get result URL for domain check
                result_url = url_matched_result.get('link', '') if url_matched_result else linkedin_url
                # Check for institution context, ambiguous cities, and URL domain
                if should_reject_city_match(city_lower, state, country, full_text, full_name, linkedin_url=result_url, role=role, company=company):
                    pass  # Skip - institution context, ambiguous city, or contradicting location
                else:
                    location_passed = True
                    location_method = 'city_fallback'

        # Area check
        if not location_passed:
            area_match = re.search(r'(Greater\s+[\w\s]+|[\w\s]+\s+Metropolitan|[\w\s]+\s+Bay|[\w\s]+\s+Metro)\s*Area', full_text, re.IGNORECASE)
            if area_match:
                area_found = area_match.group(0).strip()
                if city_lower not in area_found.lower():
                    if is_city_in_area_approved(city, area_found, state, country):
                        location_passed = True
                        location_method = 'area_approved'
                        result['data']['extracted_location'] = area_found
                    elif is_area_in_mappings(area_found):
                        # DIRECT FAIL - area mismatch, no Q3
                        result['data']['query_used'] = '+'.join(queries_used)
                        result['data']['location_method'] = 'area_mismatch'
                        result['rejection_reason'] = {
                            "stage": "Stage 4: Lead Validation",
                            "check_name": "lead_validation_stage4",
                            "message": f"Area mismatch: found '{area_found}' but city '{city}' not in approved list",
                            "failed_fields": ["city"],
                            "area_found": area_found,
                            "claimed_city": city
                        }
                        return result

        # Non-LinkedIn fallback (structured location only, no city-only fallback)
        # Note: City-only fallback from non-LinkedIn sources was too loose
        # (matched company HQ locations instead of person locations)
        if not location_passed:
            for r in all_results[:5]:
                if get_linkedin_id(r.get('link', '')):
                    continue
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}"
                r_loc = extract_location_from_text(r_text)
                if r_loc:
                    loc_match, loc_method = check_locations_match(r_loc, gt_location, r_text)
                    if loc_match:
                        location_passed = True
                        location_method = f'non_linkedin_{loc_method}'
                        result['data']['extracted_location'] = r_loc
                        break

    # 6c. Q3 Location Fallback
    if not location_passed and city and linkedin_url:
        print(f"   🔍 Q3: Fallback \"{full_name}\" \"{company}\" \"{city}\" \"{linkedin_url}\"")
        result['data']['q3_called'] = True
        queries_used.append('Q3')

        # Run Q3 search (pass state/country for ambiguous city verification)
        q3_result = check_q3_location_fallback(full_name, company, city, linkedin_url, api_key, state, country)

        if q3_result.get('passed'):
            location_passed = True
            location_method = 'q3_fallback'
            result['data']['q3_result'] = 'pass'
            print("   ✅ Q3 passed: all search terms found")
        else:
            result['data']['q3_result'] = 'fail'
            print(f"   ❌ Q3 failed: {q3_result.get('error', 'no match')}")

    result['data']['query_used'] = '+'.join(queries_used)

    # Final location check
    if not location_passed:
        result['data']['location_method'] = 'not_found'
        result['rejection_reason'] = {
            "stage": "Stage 4: Lead Validation",
            "check_name": "lead_validation_stage4",
            "message": f"Location not verified for city '{city}'",
            "failed_fields": ["city"],
            "extracted_location": result['data']['extracted_location'],
            "claimed_city": city,
            "q3_called": result['data']['q3_called'],
            "q3_result": result['data']['q3_result']
        }
        return result

    result['data']['location_passed'] = True
    result['data']['location_method'] = location_method

    print(f"   ✅ Location verified: {location_method}")

    # =========================================================================
    # STEP 7: Role Rule-Based Verification
    # =========================================================================
    if role:
        print(f"   🔍 Role verification: checking if '{role}' matches profile")

        role_passed, role_method = validate_role_rule_based(
            role, all_results, linkedin_url, full_name
        )

        if role_passed:
            result['data']['role_verified'] = True
            result['data']['role_method'] = role_method
            print(f"   ✅ Role verified (rule-based): {role_method}")
        else:
            # =========================================================================
            # STEP 8: Role LLM Fallback
            # =========================================================================
            print("   🤖 Role rule-based failed, trying LLM verification")
            result['data']['llm_used'] = True

            # Prepare LLM input
            exact_url_text = None
            if url_matched_result:
                exact_url_text = f"Title: {url_matched_result.get('title', '')}\nSnippet: {url_matched_result.get('snippet', '')}"

            other_results_text = []
            for r in all_results[:10]:
                if 'linkedin.com' not in r.get('link', '').lower():
                    other_results_text.append(f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}")

            # Get OpenRouter API key
            openrouter_key = openrouter_api_key or OPENROUTER_KEY

            if openrouter_key:
                llm_result = validate_role_with_llm(
                    full_name, company, role,
                    exact_url_text, other_results_text[:5],
                    openrouter_key
                )

                if llm_result.get('success') and llm_result.get('role_pass'):
                    result['data']['role_verified'] = True
                    result['data']['role_method'] = 'llm'
                    result['data']['llm_result'] = 'pass'
                    print("   ✅ Role verified (LLM)")
                elif llm_result.get('success'):
                    result['data']['role_verified'] = False
                    result['data']['role_method'] = 'llm'
                    result['data']['llm_result'] = 'fail'
                    result['rejection_reason'] = {
                        "stage": "Stage 4: Lead Validation",
                        "check_name": "lead_validation_stage4",
                        "message": f"Role '{role}' not verified by LLM",
                        "failed_fields": ["role"],
                        "claimed_role": role
                    }
                    print("   ❌ Role rejected (LLM)")
                    return result
                else:
                    # LLM error - fail safe
                    result['data']['role_verified'] = False
                    result['data']['role_method'] = 'llm_error'
                    result['data']['llm_result'] = llm_result.get('error', 'unknown')
                    result['rejection_reason'] = {
                        "stage": "Stage 4: Lead Validation",
                        "check_name": "lead_validation_stage4",
                        "message": f"Role verification LLM error: {llm_result.get('error', 'unknown')}",
                        "failed_fields": ["role"],
                        "claimed_role": role,
                        "llm_error": llm_result.get('error', 'unknown')
                    }
                    print(f"   ❌ Role LLM error: {llm_result.get('error', 'unknown')}")
                    return result
            else:
                # No API key - fail
                result['data']['role_verified'] = False
                result['data']['role_method'] = 'no_api_key'
                result['rejection_reason'] = {
                    "stage": "Stage 4: Lead Validation",
                    "check_name": "lead_validation_stage4",
                    "message": "Role verification failed: no OpenRouter API key for LLM fallback",
                    "failed_fields": ["role"],
                    "claimed_role": role
                }
                print("   ❌ Role verification failed: no OpenRouter API key")
                return result
    else:
        # No role provided - skip verification
        result['data']['role_verified'] = False
        result['data']['role_method'] = 'no_role_provided'
        print("   ⚠️ No role provided, skipping role verification")

    result['passed'] = True

    print(f"   ✅ Lead validation passed (queries: {result['data']['query_used']})")

    return result


async def run_location_validation_only(
    lead: Dict[str, Any],
    search_results: list,
    url_matched_result: dict,
    scrapingdog_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run only the location validation portion.

    Use this when you already have search results from Stage 4's existing flow
    and just want to add the new location validation logic.

    Args:
        lead: Lead data dict
        search_results: Existing search results from Stage 4
        url_matched_result: The URL-matched result from Stage 4
        scrapingdog_api_key: API key for Q3 fallback

    Returns:
        {
            'passed': bool,
            'method': str,
            'extracted_location': str,
            'q3_called': bool,
            'q3_result': str,
            'rejection_reason': dict or None
        }
    """
    api_key = scrapingdog_api_key or SCRAPINGDOG_API_KEY

    # Normalize field names
    full_name = (lead.get("full_name") or lead.get("Full_name") or
                 lead.get("Full Name") or lead.get("name") or "").strip()
    company = (lead.get("company") or lead.get("business") or
               lead.get("Company") or lead.get("Business") or "").strip()
    linkedin_url = (lead.get("linkedin") or lead.get("Linkedin") or
                    lead.get("linkedin_url") or "").strip()
    city = (lead.get("city") or lead.get("City") or "").strip()
    state = (lead.get("state") or lead.get("State") or "").strip()
    country = (lead.get("country") or lead.get("Country") or "").strip()
    role = (lead.get("role") or lead.get("Role") or
            lead.get("job_title") or lead.get("Job_title") or "").strip()

    result = {
        'passed': False,
        'method': None,
        'extracted_location': None,
        'q3_called': False,
        'q3_result': None,
        'rejection_reason': None
    }

    if not url_matched_result:
        result['rejection_reason'] = {"message": "No URL-matched result provided"}
        return result

    # Extract location from result
    full_text = f"{url_matched_result.get('title', '')} {url_matched_result.get('snippet', '')}"
    extracted_loc = extract_location_from_text(full_text)

    # Validate extracted state for structured locations
    structured_loc_valid = False
    if extracted_loc and ',' in extracted_loc:
        parts = extracted_loc.split(',')
        if len(parts) >= 2:
            state_part = parts[1].strip()

            # Case 1: Valid US state - this is a structured US location
            if state_part and is_valid_state(state_part):
                structured_loc_valid = True

            # Case 2: Check if it's an international location (use GEO_LOOKUP)
            elif state_part:
                # Check parts[2] for country if exists, or parts[1] if 2-part location
                valid_countries = [c.lower() for c in GEO_LOOKUP.get('countries', [])]
                country_part = parts[2].strip().lower() if len(parts) >= 3 else parts[1].strip().lower()

                if country_part in valid_countries and country_part != 'united states':
                    structured_loc_valid = True
                else:
                    # Check for metro area patterns which are still valid
                    is_metro = any(x.lower() in extracted_loc.lower() for x in ['Area', 'Metropolitan', 'Greater'])
                    if not is_metro:
                        extracted_loc = ''  # Invalid location format
            # Case 3: Empty state_part - not a valid structured location
            # structured_loc_valid remains False

    result['extracted_location'] = extracted_loc

    location_passed = False
    location_method = None

    # Structured city check (validates BOTH city AND state)
    if structured_loc_valid and extracted_loc and city:
        parts = extracted_loc.split(',')
        ext_city = parts[0].strip().lower()
        ext_state = parts[1].strip().lower() if len(parts) >= 2 else ''
        claimed_city = city.lower().strip()
        claimed_state = state.lower().strip() if state else ''

        # Normalize city names using equivalents (e.g., Bangalore = Bengaluru)
        ext_city_norm = CITY_EQUIVALENTS.get(ext_city, ext_city)
        claimed_city_norm = CITY_EQUIVALENTS.get(claimed_city, claimed_city)
        city_match = (claimed_city_norm in ext_city_norm or ext_city_norm in claimed_city_norm or
                      claimed_city in ext_city or ext_city in claimed_city)

        # Normalize state abbreviations for comparison
        state_abbr = GEO_LOOKUP.get('state_abbr', {})
        if ext_state in state_abbr:
            ext_state_full = state_abbr[ext_state].lower()
        else:
            ext_state_full = ext_state
        if claimed_state in state_abbr:
            claimed_state_full = state_abbr[claimed_state].lower()
        else:
            claimed_state_full = claimed_state

        state_match = (
            not claimed_state or  # No claimed state - skip state check
            claimed_state_full in ext_state_full or
            ext_state_full in claimed_state_full or
            claimed_state in ext_state or
            ext_state in claimed_state
        )

        if city_match and state_match:
            location_passed = True
            location_method = 'structured_city_match'
        elif not city_match:
            result['method'] = 'city_mismatch'
            result['rejection_reason'] = {
                "message": f"City mismatch: extracted '{ext_city}' but claimed '{city}'",
                "extracted_city": ext_city,
                "claimed_city": city
            }
            return result
        elif not state_match:
            result['method'] = 'state_mismatch'
            result['rejection_reason'] = {
                "message": f"State mismatch: extracted '{ext_state}' but claimed '{state}'",
                "extracted_state": ext_state,
                "claimed_state": state
            }
            return result

    # Other checks...
    if not location_passed and city:
        gt_location = f"{city}, {state}, {country}".strip(', ')
        city_lower = city.lower().strip()

        if not structured_loc_valid and extracted_loc:
            loc_match, loc_method = check_locations_match(extracted_loc, gt_location, full_text)
            if loc_match:
                # Ambiguous/English word cities need state verification even on
                # non-structured match (e.g. "Richmond" alone could be VA, CA, KY...)
                needs_strict = is_ambiguous_city(city_lower) or is_english_word_city(city_lower)
                if needs_strict:
                    has_state = _verify_state_or_country_for_strict_validation(
                        city_lower, state, country, full_text, linkedin_url)
                    if has_state:
                        location_passed = True
                        location_method = loc_method
                else:
                    location_passed = True
                    location_method = loc_method

        if not location_passed and re.search(r'\b' + re.escape(city_lower) + r'\b', full_text.lower()):
            # Get result URL for domain check
            result_url = url_matched_result.get('link', '') if url_matched_result else linkedin_url
            # Check for institution context, ambiguous cities, and URL domain
            if should_reject_city_match(city_lower, state, country, full_text, full_name, linkedin_url=result_url, role=role, company=company):
                pass  # Skip - institution context, ambiguous city, or contradicting location
            else:
                location_passed = True
                location_method = 'city_fallback'

        # Area check
        if not location_passed:
            area_match = re.search(r'(Greater\s+[\w\s]+|[\w\s]+\s+Metropolitan|[\w\s]+\s+Bay|[\w\s]+\s+Metro)\s*Area', full_text, re.IGNORECASE)
            if area_match:
                area_found = area_match.group(0).strip()
                if city_lower not in area_found.lower():
                    if is_city_in_area_approved(city, area_found, state, country):
                        location_passed = True
                        location_method = 'area_approved'
                        result['extracted_location'] = area_found
                    elif is_area_in_mappings(area_found):
                        result['method'] = 'area_mismatch'
                        result['rejection_reason'] = {
                            "message": f"Area mismatch: '{area_found}' but city '{city}' not approved",
                            "area_found": area_found,
                            "claimed_city": city
                        }
                        return result

        # Non-LinkedIn fallback (structured location only, no city-only fallback)
        # Note: Require name AND company in result text to avoid matching
        # a different person with the same name from aggregator sites
        if not location_passed:
            name_lower = full_name.lower()
            company_lower = company.lower() if company else ''
            for r in search_results[:5]:
                if get_linkedin_id(r.get('link', '')):
                    continue
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}"
                r_text_lower = r_text.lower()
                # Must contain person's name
                if name_lower not in r_text_lower:
                    continue
                # Must contain company if available
                if company_lower and company_lower not in r_text_lower:
                    continue
                r_loc = extract_location_from_text(r_text)
                if r_loc:
                    loc_match, loc_method = check_locations_match(r_loc, gt_location, r_text)
                    if loc_match:
                        location_passed = True
                        location_method = f'non_linkedin_{loc_method}'
                        result['extracted_location'] = r_loc
                        break

    # Q3 Fallback (pass state/country for ambiguous city verification)
    if not location_passed and city and linkedin_url and api_key:
        result['q3_called'] = True
        q3_result = check_q3_location_fallback(full_name, company, city, linkedin_url, api_key, state, country, role)

        if q3_result.get('passed'):
            location_passed = True
            location_method = 'q3_fallback'
            result['q3_result'] = 'pass'
        else:
            result['q3_result'] = 'fail'

    if location_passed:
        result['passed'] = True
        result['method'] = location_method
    else:
        result['method'] = 'not_found'
        result['rejection_reason'] = {
            "message": f"Location not verified for city '{city}'",
            "claimed_city": city,
            "q3_called": result['q3_called'],
            "q3_result": result['q3_result']
        }

    return result
