import asyncio
import re
import requests
import time
from typing import Dict, Any, Tuple, List, Optional
from Leadpoet.utils.utils_lead_extraction import get_company, get_linkedin
from validator_models.checks_utils import (
    SCRAPINGDOG_API_KEY,
    PROXY_CONFIG,
    normalize_accents,
    get_company_linkedin_from_cache,
    set_company_linkedin_cache,
    get_standardized_company_name,
    set_standardized_company_name,
)
from validator_models.stage4_person_verification import run_lead_validation_stage4
from validator_models.stage5_verification import (
    validate_company_linkedin_url,
    scrape_company_linkedin_gse,
    verify_company_linkedin_data,
)


async def search_linkedin_gse(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> Tuple[List[dict], bool]:
    """
    Search LinkedIn using ScrapingDog Google Search API.

    Uses 2-step approach:
    1. Q4: "{name}" "{company}" linkedin location (primary)
    2. Q1: site:linkedin.com/in/{profile_slug} (fallback if URL not found in Q4)

    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: LinkedIn URL provided by miner (required)
        max_results: Max search results to return

    Returns:
        Tuple of (List of search results with title, link, snippet, url_match_exact: bool)
    """
    if not linkedin_url:
        print(f"   ⚠️ No LinkedIn URL provided")
        return [], False

    if not SCRAPINGDOG_API_KEY:
        raise Exception("SCRAPINGDOG_API_KEY not set")

    # Extract profile slug from LinkedIn URL
    profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0] if "/in/" in linkedin_url else None

    # Track if URL matched exactly (strong identity proof)
    url_match_exact = False

    # Build search queries: Q4 primary, Q1 fallback
    # Q4: Primary query - name + company + linkedin location
    q4_query = f'"{full_name}" "{company}" linkedin location'

    # Q1: Fallback query - direct profile search (only used if URL not found in Q4)
    q1_query = f'site:linkedin.com/in/{profile_slug}' if profile_slug else None

    print(f"   🔍 Q4 Query: {q4_query[:80]}...")

    def _search_linkedin_sync(query: str) -> List[dict]:
        """Synchronous ScrapingDog search helper for Stage 4 with one retry"""
        url = "https://api.scrapingdog.com/google"
        params = {
            "api_key": SCRAPINGDOG_API_KEY,
            "query": query,
            "results": max_results
        }

        for attempt in range(2):  # Max 2 attempts (1 original + 1 retry)
            try:
                response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG)

                if response.status_code == 200:
                    data = response.json()
                    items = []

                    # Convert ScrapingDog format to standard format
                    for item in data.get("organic_results", []):
                        items.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })

                    return items

                # Retry once on 5xx server errors or rate limits (429)
                if attempt == 0 and (response.status_code >= 500 or response.status_code == 429):
                    print(f"         ⚠️ GSE API error: HTTP {response.status_code}, retrying in 2s...")
                    time.sleep(2)
                    continue

                # Non-retryable error or retry exhausted
                print(f"         ⚠️ GSE API error: HTTP {response.status_code}: {response.text[:100]}")
                return []

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == 0:
                    print(f"         ⚠️ Network error: {str(e)[:50]}, retrying in 2s...")
                    time.sleep(2)
                    continue
                print(f"         ⚠️ Network error after retry: {str(e)[:50]}")
                return []
            except Exception as e:
                print(f"         ⚠️ Request error: {str(e)}")
                return []

        return []

    def _check_url_match(items: List[dict], profile_slug: str) -> Tuple[List[dict], List[str], bool]:
        """Check if target profile URL is in results."""
        linkedin_results = []
        found_profile_urls = []

        for item in items:
            link = item.get("link", "")
            if "linkedin.com/in/" in link:
                result_slug = link.split("/in/")[-1].strip("/").split("?")[0]
                found_profile_urls.append(result_slug)
                linkedin_results.append(item)
            elif "linkedin.com" in link:
                # Include other LinkedIn pages (company, posts) for context
                linkedin_results.append(item)

        url_match = False
        url_match_type = None
        if profile_slug and found_profile_urls:
            profile_slug_norm = profile_slug.lower().replace("-", "").replace("_", "")
            # Check exact match
            exact_match = any(
                profile_slug_norm == rs.lower().replace("-", "").replace("_", "")
                for rs in found_profile_urls
            )
            # Check partial match
            partial_match = any(
                profile_slug_norm in rs.lower().replace("-", "").replace("_", "") or
                rs.lower().replace("-", "").replace("_", "") in profile_slug_norm
                for rs in found_profile_urls
            )
            if exact_match:
                url_match = True
                url_match_type = "exact"
            elif partial_match:
                url_match = True
                url_match_type = "partial"

        return linkedin_results, found_profile_urls, url_match, url_match_type

    try:
        # Step 1: Run Q4 query (primary)
        print(f"      🔄 Running Q4 query...")
        q4_items = await asyncio.to_thread(_search_linkedin_sync, q4_query)

        url_found_in_q4 = False
        linkedin_results = []
        found_profile_urls = []

        if q4_items:
            print(f"         ✅ Q4 found {len(q4_items)} result(s)")
            linkedin_results, found_profile_urls, url_found_in_q4, url_match_type = _check_url_match(q4_items, profile_slug)

            if url_found_in_q4:
                if url_match_type == "exact":
                    print(f"         ✅ URL MATCH (Q4): Profile '{profile_slug}' confirmed (exact)")
                    url_match_exact = True
                else:
                    print(f"         ✅ URL MATCH (Q4): Profile '{profile_slug}' confirmed (partial)")
                    url_match_exact = "partial"
            else:
                print(f"         ⚠️ URL not found in Q4 results, expected: {profile_slug}")
        else:
            print(f"         ⚠️ Q4 returned no results")

        # Step 2: Run Q1 fallback if URL not found in Q4
        if not url_found_in_q4 and q1_query:
            print(f"      🔄 Running Q1 fallback: {q1_query}")
            q1_items = await asyncio.to_thread(_search_linkedin_sync, q1_query)

            if q1_items:
                print(f"         ✅ Q1 found {len(q1_items)} result(s)")
                q1_linkedin_results, q1_found_urls, q1_url_found, q1_match_type = _check_url_match(q1_items, profile_slug)

                if q1_url_found:
                    # Merge Q1 results with Q4 results (Q1 results first since they have the URL)
                    linkedin_results = q1_linkedin_results + linkedin_results
                    found_profile_urls = q1_found_urls + found_profile_urls
                    url_found_in_q4 = True  # Now we found the URL (in Q1)

                    if q1_match_type == "exact":
                        print(f"         ✅ URL MATCH (Q1): Profile '{profile_slug}' confirmed (exact)")
                        url_match_exact = True
                    else:
                        print(f"         ✅ URL MATCH (Q1): Profile '{profile_slug}' confirmed (partial)")
                        url_match_exact = "partial"
                else:
                    print(f"         ⚠️ URL not found in Q1 results either")
            else:
                print(f"         ⚠️ Q1 returned no results")

        # If no URL match after Q4 + Q1, fail
        if not url_found_in_q4:
            print(f"   ❌ URL not found in Q4 or Q1 results")
            return [], False

        # Continue with filtering if we have results
        if not linkedin_results:
            print(f"   ❌ No LinkedIn URLs in results")
            return [], False

        print(f"         ✅ Found {len(linkedin_results)} LinkedIn result(s)")

        # FILTER 1: Clean up concatenated titles and separate profile headlines from posts
        # ScrapingDog often concatenates multiple result titles together
        profile_headlines = []
        posts = []

        for item in linkedin_results:
            title = item.get("title", "")

            # ScrapingDog concatenates titles - extract only the FIRST profile
            # Pattern: "Name - Title | LinkedIn Name2 - Title2"
            if " | LinkedIn " in title:
                # Take only the first profile (before the concatenation)
                title = title.split(" | LinkedIn ")[0] + " | LinkedIn"
                item = dict(item)  # Copy to avoid modifying original
                item["title"] = title

            # Skip non-profile results (posts, intro requests, etc.)
            if " on LinkedIn:" in title or " on LinkedIn :" in title:
                posts.append(item)
                continue
            if title.lower().startswith("seeking intro"):
                posts.append(item)
                continue
            # Skip directory pages (but not profiles that just have this text concatenated)
            is_directory_title = ("profiles | LinkedIn" in title or
                                 "profiles - LinkedIn" in title or
                                 "profiles on LinkedIn" in title)

            if is_directory_title:
                link = item.get("link", "")
                is_directory_link = "/pub/dir/" in link or "/directory/" in link
                starts_with_profiles = re.match(r'^\d+\+?\s+"?[^"]*"?\s+profiles', title.lower())

                if is_directory_link or starts_with_profiles:
                    continue  # Skip directory pages

            profile_headlines.append(item)

        # FILTER 2: Only keep results for TARGET PERSON (filter out other people)
        name_parts = full_name.lower().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        # Normalize accents for matching (José -> Jose, François -> Francois)
        first_name_normalized = normalize_accents(first_name)
        last_name_normalized = normalize_accents(last_name)

        target_person_results = []
        other_person_results = []

        for item in profile_headlines:
            title_lower = item.get("title", "").lower()
            link = item.get("link", "")

            # Normalize the title too for accent-insensitive matching
            title_normalized = normalize_accents(title_lower)

            # PRIORITY: If this result's URL matches our target profile slug, it's THE profile!
            if profile_slug and "linkedin.com/in/" in link:
                result_slug = link.split("/in/")[-1].strip("/").split("?")[0]
                if profile_slug.lower() == result_slug.lower():
                    target_person_results.append(item)
                    continue  # Skip name check - URL match is definitive

            # Check if target person's name is in the title (accent-insensitive)
            if first_name_normalized in title_normalized and last_name_normalized in title_normalized:
                target_person_results.append(item)
            else:
                other_person_results.append(item)

        # Return target person's profile headlines
        if target_person_results:
            print(f"      📊 GSE Profile Headlines for {full_name}:")
            for i, item in enumerate(target_person_results[:3], 1):
                print(f"         {i}. {item.get('title', '')[:70]}")
            if other_person_results:
                print(f"      📊 Other profiles filtered out: {len(other_person_results)}")
            if posts:
                print(f"      📊 Posts filtered out: {len(posts)}")
            return target_person_results[:max_results], url_match_exact
        elif profile_headlines:
            # No exact name match but have profile headlines - return them anyway since URL matched
            print(f"      ⚠️ Name not found in results but URL matched, returning results")
            return profile_headlines[:max_results], url_match_exact
        elif posts:
            # Only posts found (no profile headlines) - return posts
            print(f"      📊 ScrapingDog Posts only (no profile headlines found):")
            for i, item in enumerate(posts[:3], 1):
                print(f"         {i}. {item.get('title', '')[:70]}")
            return posts[:max_results], url_match_exact
        else:
            print(f"   ❌ No usable results after filtering")
            return [], False

    except Exception as e:
        print(f"   ⚠️ GSE API error: {str(e)}")
        return [], False


async def verify_linkedin_rule_based(full_name: str, company: str, linkedin_url: str, search_results: List[dict], url_match_exact: bool = False) -> Tuple[bool, str]:
    """
    Rule-based verification of LinkedIn profile from search results.

    NOTE: This function no longer uses LLM - it's purely rule-based for speed and consistency.
    Name kept for backward compatibility with callers.

    Verification logic:
    1. URL match (exact or partial) = strong identity proof
    2. Name match = first/last name found in title or URL slug
    3. Company match = company name found in title/snippet

    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: Provided LinkedIn URL
        search_results: Google search results
        url_match_exact: If True, URL slug matched exactly (strong identity proof)

    Returns:
        (is_verified, reasoning)
    """
    try:
        if not search_results:
            return False, "No LinkedIn search results found"

        # ========================================================================
        # STEP 1: Find URL-matched result (authoritative source)
        # ========================================================================
        profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower() if linkedin_url and "/in/" in linkedin_url else None
        target_result = search_results[0]  # Default to first

        if profile_slug:
            for result in search_results:
                result_url = result.get("link", "").lower()
                if f"/in/{profile_slug}" in result_url or (
                    profile_slug.replace("-", "").replace("_", "") in
                    result_url.replace("-", "").replace("_", "")
                ):
                    target_result = result
                    print(f"   🎯 Using URL-matched result: {result_url[:60]}")
                    break

        # ========================================================================
        # STEP 2: Rule-based NAME match
        # ========================================================================
        name_parts = full_name.lower().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else first_name

        # Normalize accents
        first_name_norm = normalize_accents(first_name)
        last_name_norm = normalize_accents(last_name)

        title = target_result.get("title", "").lower()
        title_norm = normalize_accents(title)

        # Method 1: Name in title
        name_in_title = (first_name_norm in title_norm and last_name_norm in title_norm)

        # Method 2: Name in URL slug (e.g., "john-smith" matches "John Smith")
        name_in_url = False
        if profile_slug:
            slug_clean = profile_slug.replace("-", " ").replace("_", " ").replace("%20", " ")
            slug_norm = normalize_accents(slug_clean)
            name_in_url = (first_name_norm in slug_norm and last_name_norm in slug_norm)

        name_match = name_in_title or name_in_url

        # If URL matched exactly, trust it as identity proof
        if url_match_exact and not name_match:
            print(f"   ✅ URL EXACT MATCH: Trusting URL as identity proof even without name in title")
            name_match = True

        print(f"   🔍 NAME CHECK: first='{first_name}' last='{last_name}' → in_title={name_in_title}, in_url={name_in_url}")

        # ========================================================================
        # STEP 3: Rule-based COMPANY match
        # ========================================================================
        company_lower = company.lower().strip()
        company_lower = company_lower.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
        company_lower = re.sub(r"\s*'\s*", "'", company_lower)
        company_lower = re.sub(r"\s*-\s*", "-", company_lower)

        # Remove legal suffixes
        LEGAL_SUFFIXES = [
            " corporation", " corp.", " corp", " incorporated", " inc.", " inc",
            " llc", " l.l.c.", " ltd.", " ltd", " limited", " plc", " p.l.c.",
            " co.", " co", " company", " gmbh", " ag", " sa", " nv", " bv",
            " holdings", " holding", " group", " international", " intl"
        ]
        company_normalized = company_lower
        for suffix in LEGAL_SUFFIXES:
            if company_normalized.endswith(suffix):
                company_normalized = company_normalized[:-len(suffix)].strip()
                break

        company_words = company_normalized.split()

        # Normalize title for company check
        first_title = title.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
        first_title = re.sub(r"\s*'\s*", "'", first_title)
        first_title = re.sub(r"\s*-\s*", "-", first_title)
        if "| linkedin" in first_title:
            first_title = first_title.split("| linkedin")[0].strip()

        first_snippet = target_result.get("snippet", "").lower()
        first_snippet = re.sub(r"\s*-\s*", "-", first_snippet)

        # Method 1: Exact company name in title
        company_in_title = company_normalized in first_title

        # Method 2: All significant words in title
        if not company_in_title and len(company_words) > 1:
            significant_words = [w for w in company_words if len(w) > 2]
            company_in_title = all(word in first_title for word in significant_words)

        # Method 3: Extract company from title pattern "at Company" or "@ Company"
        if not company_in_title:
            title_company_match = re.search(r'(?:at|@)\s+([^|\-]+?)(?:\s*[\|\-]|$)', first_title, re.IGNORECASE)
            if title_company_match:
                extracted_company = title_company_match.group(1).strip().lower()
                extracted_company = re.sub(r'\s+(linkedin|profile|page).*$', '', extracted_company)
                extracted_company = re.sub(r'\s*\([^)]*\).*$', '', extracted_company)
                extracted_company = re.sub(r'\s*\|.*$', '', extracted_company)
                extracted_company = extracted_company.strip()

                # Normalize extracted company
                extracted_normalized = extracted_company
                for suffix in LEGAL_SUFFIXES:
                    if extracted_normalized.endswith(suffix):
                        extracted_normalized = extracted_normalized[:-len(suffix)].strip()
                        break

                # Bidirectional containment check
                if len(extracted_normalized) >= 4 and len(company_normalized) >= 4:
                    if (extracted_normalized in company_normalized or
                        company_normalized in extracted_normalized):
                        if ' ' in extracted_normalized or ' ' in company_normalized:
                            company_in_title = True
                            print(f"   ✅ RULE-BASED MATCH: '{extracted_normalized}' ≈ '{company_normalized}'")
                        else:
                            longer = max(extracted_normalized, company_normalized, key=len)
                            shorter = min(extracted_normalized, company_normalized, key=len)
                            if len(shorter) / len(longer) >= 0.6:
                                company_in_title = True

        # Method 4: Company in snippet
        company_in_snippet = False
        if not company_in_title:
            if company_normalized in first_snippet:
                company_in_snippet = True
            elif len(company_words) > 1:
                significant_words = [w for w in company_words if len(w) > 2]
                if all(word in first_snippet for word in significant_words):
                    company_in_snippet = True

        company_match = company_in_title or company_in_snippet

        # If URL matched but company not found (truncation), defer to Stage 5
        if url_match_exact and name_match and not company_match:
            print(f"   ⚠️ URL + Name match but company not found → deferring to Stage 5")
            company_match = True  # Pass Stage 4, Stage 5 will verify

        match_location = "title" if company_in_title else ("snippet" if company_in_snippet else "NOT FOUND")
        print(f"   🔍 COMPANY CHECK: '{company}' in {match_location} = {company_match}")

        # ========================================================================
        # STEP 4: Make decision
        # ========================================================================
        profile_valid = True  # URL was found in search results

        if name_match and company_match and profile_valid:
            reasoning = f"Rule-based PASS: URL={'exact' if url_match_exact else 'partial'}, name_match={name_match}, company_match={company_match}"
            print(f"   ✅ RULE-BASED VERIFICATION: PASSED")
            print(f"      {reasoning}")
            return True, reasoning
        else:
            failures = []
            if not name_match:
                failures.append("name mismatch")
            if not company_match:
                failures.append("company mismatch")

            failure_str = ", ".join(failures)
            reasoning = f"Rule-based FAIL: {failure_str}"
            print(f"   ❌ RULE-BASED VERIFICATION: FAILED")
            print(f"      {reasoning}")
            return False, reasoning

    except Exception as e:
        return False, f"Verification error: {str(e)}"


async def check_linkedin_gse(lead: dict) -> Tuple[bool, dict]:
    """
    Stage 4: LinkedIn/GSE validation (HARD check).

    Verifies lead using the new lead_validation module:
    1. Role format validation (before any API calls)
    2. Q4 search + Q1 fallback
    3. URL match, Name check, Company check
    4. Location validation with Q3 fallback
    5. Company LinkedIn validation (existing)

    This is a HARD check - instant rejection if fails.

    Args:
        lead: Lead data with full_name, company, linkedin

    Returns:
        (passed, rejection_reason)
    """
    try:
        full_name = lead.get("full_name") or lead.get("Full_name") or lead.get("Full Name")
        company = get_company(lead)
        linkedin_url = get_linkedin(lead)

        # ========================================================================
        # NEW: Run lead validation (role format, URL, name, company, location)
        # ========================================================================
        print(f"   🔍 Stage 4: Running lead validation for {full_name} at {company}")

        validation_result = await run_lead_validation_stage4(lead)

        if not validation_result['passed']:
            # Store validation data on lead for debugging
            lead["lead_validation_data"] = validation_result.get('data', {})
            return False, validation_result['rejection_reason']

        # Store validation data on lead
        validation_data = validation_result.get('data', {})
        lead["gse_search_count"] = len(validation_data.get('search_results', []))
        lead["query_used"] = validation_data.get('query_used', '')
        lead["q3_called"] = validation_data.get('q3_called', False)
        lead["q3_result"] = validation_data.get('q3_result')

        # Store role verification results for Stage 5
        lead["role_verified"] = validation_data.get('role_verified', False)
        lead["role_method"] = validation_data.get('role_method', '')
        if validation_data.get('extracted_role'):
            lead["stage4_extracted_role"] = validation_data['extracted_role']
            print(f"   📝 Stage 4: Role verified: '{validation_data['extracted_role']}' (method: {validation_data.get('role_method', '')})")

        # Store location verification results for Stage 5
        lead["location_verified"] = validation_data.get('location_passed', False)
        lead["location_method"] = validation_data.get('location_method', '')
        lead["extracted_location"] = validation_data.get('extracted_location', '')
        if validation_data.get('extracted_location'):
            lead["stage4_extracted_location"] = validation_data['extracted_location']
            print(f"   📍 Stage 4: Location verified: '{validation_data['extracted_location']}' (method: {validation_data.get('location_method', '')})")

        print(f"   ✅ Stage 4: Lead validation passed (queries: {validation_data.get('query_used', 'N/A')})")

        # ========================================================================
        # STAGE 4: COMPANY LINKEDIN VALIDATION
        # ========================================================================
        # Validates company_linkedin URL, verifies company name matches, and caches
        # company data (industry, description, employee_count) for Stage 5.
        # FAIL HERE = Stage 5 never runs = saves all Stage 5 API costs
        # ========================================================================

        company_linkedin = lead.get("company_linkedin", "") or ""

        if company_linkedin:
            print(f"   🏢 Stage 4: Validating company LinkedIn URL...")

            # Step 1: Validate URL format (must be /company/, not /in/)
            url_valid, url_reason, company_slug = validate_company_linkedin_url(company_linkedin)

            if not url_valid:
                print(f"   ❌ Stage 4: Company LinkedIn URL INVALID: {url_reason}")
                return False, {
                    "stage": "Stage 4: Company LinkedIn Validation",
                    "check_name": "check_linkedin_gse",
                    "message": f"Company LinkedIn URL is invalid: {url_reason}",
                    "failed_fields": ["company_linkedin"],
                    "provided_url": company_linkedin,
                    "expected_format": "https://linkedin.com/company/{company-name}"
                }

            print(f"   ✅ Stage 4: Company LinkedIn URL format valid: /company/{company_slug}")

            # Step 2: Check global cache first
            cached_data = get_company_linkedin_from_cache(company_slug)

            if cached_data:
                print(f"   📦 Stage 4: Using CACHED company LinkedIn data for '{company_slug}'")

                # Verify company name still matches (cache might have different company)
                cached_company_name = cached_data.get("company_name_from_linkedin", "")
                if cached_company_name:
                    # Check if cached company matches current company claim
                    cached_lower = cached_company_name.lower().strip()
                    claimed_lower = company.lower().strip()

                    if cached_lower != claimed_lower and cached_lower not in claimed_lower and claimed_lower not in cached_lower:
                        print(f"   ❌ Stage 4: Cached company name '{cached_company_name}' doesn't match claimed '{company}'")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn page shows '{cached_company_name}' but miner claimed '{company}'",
                            "failed_fields": ["company_linkedin", "company"],
                            "linkedin_company": cached_company_name,
                            "claimed_company": company
                        }

                # Store cached data on lead for Stage 5
                lead["company_linkedin_verified"] = True
                lead["company_linkedin_slug"] = company_slug
                lead["company_linkedin_data"] = cached_data
                lead["company_linkedin_from_cache"] = True

                # Log what data we have cached
                if cached_data.get("employee_count"):
                    print(f"   📊 Cached employee count: {cached_data['employee_count']}")
                if cached_data.get("industry"):
                    print(f"   🏭 Cached industry: {cached_data['industry']}")
                if cached_data.get("description"):
                    print(f"   📝 Cached description: {cached_data['description'][:80]}...")
            else:
                # Step 3: Not cached - scrape company LinkedIn page via GSE
                print(f"   🔍 Stage 4: Scraping company LinkedIn page for '{company_slug}'...")
                scraped_data = await scrape_company_linkedin_gse(company_slug, company)

                if scraped_data.get("success"):
                    # Step 4: Verify company name matches
                    if not scraped_data.get("company_name_match"):
                        linkedin_company = scraped_data.get("company_name_from_linkedin", "Unknown")
                        print(f"   ❌ Stage 4: Company name MISMATCH: LinkedIn shows '{linkedin_company}' but miner claimed '{company}'")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn page shows '{linkedin_company}' but miner claimed '{company}'",
                            "failed_fields": ["company_linkedin", "company"],
                            "linkedin_company": linkedin_company,
                            "claimed_company": company
                        }

                    print(f"   ✅ Stage 4: Company name verified: '{company}'")

                    # Step 5: Cache the data globally (only on success)
                    set_company_linkedin_cache(company_slug, scraped_data)
                    print(f"   💾 Stage 4: Cached company LinkedIn data for future leads")

                    # Store on lead for Stage 5
                    lead["company_linkedin_verified"] = True
                    lead["company_linkedin_slug"] = company_slug
                    lead["company_linkedin_data"] = scraped_data
                    lead["company_linkedin_from_cache"] = False

                    # Log what data we scraped
                    if scraped_data.get("employee_count"):
                        print(f"   📊 Scraped employee count: {scraped_data['employee_count']}")
                    if scraped_data.get("industry"):
                        print(f"   🏭 Scraped industry: {scraped_data['industry']}")
                    if scraped_data.get("description"):
                        print(f"   📝 Scraped description: {scraped_data['description'][:80]}...")
                else:
                    # Scraping failed - check if it's a URL mismatch (reject) or just scraping error (warn)
                    error_msg = scraped_data.get('error', 'Unknown')

                    # If URL doesn't match, this is a CRITICAL error - reject immediately
                    if "does not match expected slug" in error_msg or "URL mismatch" in error_msg:
                        print(f"   ❌ Stage 4: Company LinkedIn URL mismatch: {error_msg}")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn URL is incorrect or ambiguous: {error_msg}",
                            "failed_fields": ["company_linkedin"],
                            "provided_url": company_linkedin,
                            "hint": "The URL provided returns a different company page. Ensure you're using the exact LinkedIn company slug."
                        }

                    # Other scraping errors (network, API, etc.) - warn but don't fail
                    # Stage 5 will use fallback GSE searches
                    print(f"   ⚠️ Stage 4: Could not scrape company LinkedIn: {error_msg}")
                    print(f"   ⚠️ Stage 4: Stage 5 will use fallback GSE searches for industry/employee data")
                    lead["company_linkedin_verified"] = True  # URL was valid format
                    lead["company_linkedin_slug"] = company_slug
                    lead["company_linkedin_data"] = None  # No data - Stage 5 will fallback
                    lead["company_linkedin_from_cache"] = False
        else:
            # No company_linkedin provided - Stage 5 will use fallback GSE searches
            print(f"   ⚠️ Stage 4: No company_linkedin URL provided")
            lead["company_linkedin_verified"] = False
            lead["company_linkedin_data"] = None

        print(f"   ✅ Stage 4: LinkedIn verified for {full_name} at {company}")
        return True, {}

    except Exception as e:
        return False, {
            "stage": "Stage 4: LinkedIn/GSE Validation",
            "check_name": "check_linkedin_gse",
            "message": f"LinkedIn/GSE check failed: {str(e)}",
            "failed_fields": ["linkedin"]
        }
