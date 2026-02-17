import aiohttp
import asyncio
import base64
import re
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from Leadpoet.utils.utils_lead_extraction import get_email, get_website, get_company
from validator_models.checks_utils import (
    HTTP_PROXY_URL,
    PROXY_CONFIG,
    COMPANIES_HOUSE_API_KEY,
    extract_root_domain,
)


async def check_wayback_machine(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check domain history in Wayback Machine.

    Returns score (0-6) based on:
    - Number of snapshots
    - Age of domain in archive
    - Consistency of snapshots

    This is a SOFT check - always passes, appends score.

    Args:
        lead: Lead data with website

    Returns:
        (score, metadata)
    """
    try:
        website = get_website(lead)
        if not website:
            return 0, {"checked": False, "reason": "No website provided"}

        domain = extract_root_domain(website)
        if not domain:
            return 0, {"checked": False, "reason": "Invalid website format"}

        # Query Wayback Machine CDX API (with 3 retries for timeout)
        url = f"https://web.archive.org/cdx/search/cdx"
        params = {
            "url": domain,
            "output": "json",
            "limit": 1000,
            "fl": "timestamp"
        }

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=15, proxy=HTTP_PROXY_URL) as response:
                        if response.status != 200:
                            return 0, {"checked": False, "reason": f"Wayback API error: {response.status}"}

                        data = await response.json()

                        if len(data) <= 1:  # First row is header
                            return 0, {"checked": True, "snapshots": 0, "reason": "No archive history"}

                        snapshots = len(data) - 1  # Exclude header

                        # Parse timestamps to calculate age
                        timestamps = [row[0] for row in data[1:]]  # Skip header
                        oldest = timestamps[0] if timestamps else None
                        newest = timestamps[-1] if timestamps else None

                        # Calculate age in years
                        if oldest:
                            oldest_year = int(oldest[:4])
                            current_year = datetime.now().year
                            age_years = current_year - oldest_year
                        else:
                            age_years = 0

                        # Scoring logic (UPDATED: max 6 points for Wayback):
                        if snapshots < 10:
                            score = min(1.2, snapshots * 0.12)
                        elif snapshots < 50:
                            score = 1.8 + (snapshots - 10) * 0.03
                        elif snapshots < 200:
                            score = 3.6 + (snapshots - 50) * 0.008
                        else:
                            score = 5.4 + min(0.6, (snapshots - 200) * 0.0006)

                        # Age bonus
                        if age_years >= 5:
                            score = min(6, score + 0.6)

                        return score, {
                            "checked": True,
                            "snapshots": snapshots,
                            "age_years": age_years,
                            "oldest_snapshot": oldest,
                            "newest_snapshot": newest,
                            "score": score
                        }
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                return 0, {"checked": False, "reason": "Wayback API timeout (3 attempts)"}
            except Exception as e:
                return 0, {"checked": False, "reason": f"Wayback check error: {str(e)}"}

        # Fallback if loop completes without returning
        return 0, {"checked": False, "reason": "Wayback check failed unexpectedly"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"Wayback check error: {str(e)}"}

async def check_sec_edgar(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check SEC EDGAR for company filings.

    Returns score (0-12) based on:
    - Number of filings
    - Recent filing activity
    - Types of filings (10-K, 10-Q, 8-K)

    This is a SOFT check - always passes, appends score.
    Uses official SEC.gov API (free, no API key needed - just User-Agent)

    Args:
        lead: Lead data with company

    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}

        print(f"   🔍 SEC: Searching for company: '{company}'")

        # SEC.gov requires User-Agent header with contact info (no API key needed)
        headers = {
            "User-Agent": "LeadPoet/1.0 (hello@leadpoet.com)"
        }

        # Try multiple company name variations for better matching
        # SEC often uses abbreviated forms (e.g., "Microsoft Corp" not "Microsoft Corporation")
        company_variations = [
            company,  # Original name
            company.replace(" Company, Inc.", "").replace(" Corporation", " Corp").replace(", Inc.", ""),  # Abbreviated
            company.split()[0] if len(company.split()) > 1 else company,  # First word only (e.g., "Microsoft")
        ]

        # Remove duplicates while preserving order (e.g., if abbreviated = original)
        company_variations = list(dict.fromkeys(company_variations))

        print(f"      🔍 Trying {len(company_variations)} name variations: {company_variations}")

        # Use SEC.gov company search endpoint to find CIK
        # This searches the submissions index for company name matches
        search_url = "https://www.sec.gov/cgi-bin/browse-edgar"

        # Try each variation until we find results
        async with aiohttp.ClientSession() as session:
            for idx, company_variation in enumerate(company_variations):
                print(f"      🔄 Attempt {idx+1}/{len(company_variations)}: Searching for '{company_variation}'")

                # Request actual filings, not just company landing page
                # type=&dateb=&owner=include&start=0
                params = {
                    "company": company_variation,
                    "action": "getcompany",
                    "type": "",  # All filing types
                    "dateb": "",  # All dates
                    "owner": "include",  # Include company filings
                    "start": "0",  # Start from first filing
                    "count": "100"  # Get up to 100 recent filings
                }

                async with session.get(search_url, headers=headers, params=params, timeout=7, proxy=HTTP_PROXY_URL) as response:
                    if response.status != 200:
                        print(f"      ❌ SEC API returned HTTP {response.status}")
                        continue  # Try next variation

                    # Parse HTML response (SEC doesn't return JSON for this endpoint)
                    html = await response.text()
                    print(f"      📄 SEC response length: {len(html)} bytes")

                    # Check if company was found (HTML contains "No matching" if not found)
                    if "No matching" in html or "No results" in html:
                        print(f"      ❌ SEC: 'No matching' found for '{company_variation}'")
                        continue  # Try next variation

                    # Found a result! Count filing indicators in HTML
                    print(f"      ✅ SEC: Found match for '{company_variation}'")
                    filing_types = ["10-K", "10-Q", "8-K", "S-1", "10-K/A", "10-Q/A", "4", "3", "SC 13", "DEF 14A"]
                    total_filings = 0
                    for filing_type in filing_types:
                        # Look for the filing type in HTML context (e.g., ">10-K<" or " 10-K ")
                        count = html.count(f">{filing_type}<") + html.count(f" {filing_type} ")
                        if count > 0:
                            print(f"      📊 Found {count}x {filing_type}")
                        total_filings += count

                    print(f"      📊 Total filings detected: {total_filings}")

                    if total_filings == 0:
                        # The HTML might be a landing page with a link to the actual filings
                        # Try to extract CIK from the HTML and query directly
                        cik_match = re.search(r'CIK=(\d{10})', html)
                        if cik_match:
                            cik = cik_match.group(1)
                            print(f"      🔍 Found CIK: {cik}, fetching actual filings...")

                            # Query the filings page directly using CIK
                            cik_params = {
                                "action": "getcompany",
                                "CIK": cik,
                                "type": "",
                                "dateb": "",
                                "owner": "include",
                                "count": "100"
                            }

                            async with session.get(search_url, headers=headers, params=cik_params, timeout=7, proxy=HTTP_PROXY_URL) as cik_response:
                                if cik_response.status == 200:
                                    cik_html = await cik_response.text()
                                    print(f"      📄 CIK response length: {len(cik_html)} bytes")

                                    # Count filings again (use HTML-aware matching)
                                    total_filings = 0
                                    for filing_type in filing_types:
                                        count = cik_html.count(f">{filing_type}<") + cik_html.count(f" {filing_type} ")
                                        if count > 0:
                                            print(f"      📊 Found {count}x {filing_type}")
                                        total_filings += count

                                    # DEBUG: Check if HTML contains filing table markers
                                    has_filing_table = "filingTable" in cik_html or "Filing" in cik_html
                                    print(f"      🔍 DEBUG: Has 'filingTable' or 'Filing': {has_filing_table}")

                                    # If we have a valid CIK and filing indicators but can't parse exact counts,
                                    # give partial credit (company IS SEC-registered with filings)
                                    if total_filings == 0 and has_filing_table:
                                        print(f"      ⚠️  CIK {cik} has filings but HTML parsing failed")
                                        print(f"      ✅ SEC: Giving partial credit (3.6/12) for SEC-registered company")
                                        return 3.6, {
                                            "checked": True,
                                            "filings": "unknown (parsing failed)",
                                            "score": 3.6,
                                            "cik": cik,
                                            "company_name_used": company_variation,
                                            "reason": f"Company registered with SEC (CIK {cik}) but exact filing count unavailable"
                                        }

                                    if total_filings > 0:
                                        # Success! Calculate score
                                        print(f"      📊 Total filings detected: {total_filings}")

                                        if total_filings <= 5:
                                            score = min(3.6, total_filings * 0.72)
                                        elif total_filings <= 20:
                                            score = 7.2
                                        elif total_filings <= 50:
                                            score = 9.6
                                        else:
                                            score = 12

                                        print(f"      ✅ SEC: {score}/12 pts for CIK {cik}")
                                        return score, {
                                            "checked": True,
                                            "filings": total_filings,
                                            "score": score,
                                            "cik": cik,
                                            "company_name_used": company_variation,
                                            "reason": f"Found {total_filings} SEC filing indicators for CIK {cik}"
                                        }

                        print(f"      ⚠️  Match found but no filing types detected (showing first 500 chars):")
                        print(f"         {html[:500]}")
                        continue  # Try next variation

                    # Scoring logic (UPDATED: max 12 points for SEC):
                    # - 1-5 filings: 3.6 points
                    # - 6-20 filings: 7.2 points
                    # - 21-50 filings: 9.6 points
                    # - 50+ filings: 12 points

                    if total_filings <= 5:
                        score = min(3.6, total_filings * 0.72)
                    elif total_filings <= 20:
                        score = 7.2
                    elif total_filings <= 50:
                        score = 9.6
                    else:
                        score = 12

                    print(f"      ✅ SEC: {score}/12 pts for '{company_variation}'")
                    return score, {
                        "checked": True,
                        "filings": total_filings,
                        "score": score,
                        "company_name_used": company_variation,
                        "reason": f"Found {total_filings} SEC filing indicators for {company_variation}"
                    }

            # All variations failed
            print(f"      ❌ SEC: No results found for any name variation")
            return 0, {
                "checked": True,
                "filings": 0,
                "variations_tried": company_variations,
                "reason": f"No SEC filings found for {company} (tried {len(company_variations)} variations)"
            }

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "SEC API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"SEC check error: {str(e)}"}


async def check_gdelt_mentions(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check GDELT for press mentions and trusted domain coverage.

    Returns score (0-10) based on:
    - Press wire mentions (PRNewswire, BusinessWire, GlobeNewswire, ENPresswire)
    - Trusted domain mentions (.edu, .gov, high-authority sites)

    This is a SOFT check - always passes, appends score.
    Uses GDELT 2.0 DOC API (free, no API key needed)

    Scoring breakdown:
    - 0-5 points: Press wire mentions (verified company PR)
    - 0-5 points: Trusted domain mentions (.edu, .gov, DA>60)

    Args:
        lead: Lead data with company

    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}

        print(f"   🔍 GDELT: Searching for company: '{company}'")

        # GDELT 2.0 DOC API endpoint
        # Uses free public API - no key required
        gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"

        # Query for company mentions in last 3 months
        # Format: "company name" sourcelang:eng
        # NOTE: GDELT requires minimum 5 characters in query, so append "company" for short names
        search_term = company
        if len(company) <= 4:
            search_term = f"{company} company"
            print(f"      ℹ️  Short name detected, searching: '{search_term}'")
        query = f'"{search_term}" sourcelang:eng'

        async with aiohttp.ClientSession() as session:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": 250,  # Get up to 250 recent articles
                "format": "json",
                "sort": "datedesc"
            }

            async with session.get(gdelt_url, params=params, timeout=15, proxy=HTTP_PROXY_URL) as response:
                if response.status != 200:
                    print(f"      ❌ GDELT API returned HTTP {response.status}")
                    return 0, {
                        "checked": False,
                        "reason": f"GDELT API error: HTTP {response.status}"
                    }

                # GDELT sometimes returns HTML instead of JSON for short/uncommon company names
                # Check Content-Type before parsing to avoid json decode errors
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    # GDELT returned HTML page - treat as no coverage (not an error)
                    print(f"      ⚠️  GDELT returned HTML instead of JSON (no articles for '{company}')")
                    return 0, {
                        "checked": True,
                        "press_mentions": 0,
                        "trusted_mentions": 0,
                        "reason": f"No GDELT coverage found for {company}"
                    }

                data = await response.json()
                articles = data.get("articles", [])
                print(f"      📰 GDELT found {len(articles)} articles")

                if not articles:
                    print(f"      ❌ No GDELT articles found for '{company}'")
                    return 0, {
                        "checked": True,
                        "press_mentions": 0,
                        "trusted_mentions": 0,
                        "reason": f"No GDELT coverage found for {company}"
                    }

                # Parse articles for press wires and trusted domains
                press_wire_domains = {
                    "prnewswire.com",
                    "businesswire.com",
                    "globenewswire.com",
                    "enpresswire.com",
                    "prweb.com",
                    "marketwired.com"
                }

                trusted_tlds = {".edu", ".gov", ".mil"}

                # High-authority domains (Fortune 500, major news outlets, financial news)
                high_authority_domains = {
                    # Major news outlets
                    "forbes.com", "fortune.com", "bloomberg.com", "wsj.com",
                    "nytimes.com", "reuters.com", "ft.com", "economist.com",
                    "theguardian.com", "washingtonpost.com", "bbc.com", "cnbc.com",
                    # Tech news
                    "techcrunch.com", "wired.com", "theverge.com", "cnet.com",
                    "arstechnica.com", "zdnet.com", "venturebeat.com",
                    # Financial news
                    "finance.yahoo.com", "yahoo.com", "marketwatch.com", "fool.com",
                    "seekingalpha.com", "investing.com", "benzinga.com", "zacks.com",
                    "morningstar.com", "barrons.com", "investopedia.com",
                    # International business news
                    "thehindubusinessline.com", "business-standard.com", "economictimes.indiatimes.com",
                    "scmp.com", "japantimes.co.jp", "straitstimes.com"
                }

                press_mentions = []
                trusted_mentions = []
                seen_domains = set()  # Track unique domains (no spam)
                all_domains_found = []  # DEBUG: Track all domains for logging

                for article in articles:
                    url = article.get("url", "")
                    domain = article.get("domain", "")
                    title = article.get("title", "")

                    # DEBUG: Track all domains
                    if domain:
                        all_domains_found.append(domain)

                    # Skip if we've seen this domain (cap at 3 mentions per domain)
                    if domain in seen_domains:
                        domain_count = sum(1 for m in trusted_mentions if m["domain"] == domain)
                        if domain_count >= 3:
                            continue

                    seen_domains.add(domain)

                    # Check if company name appears in title (stronger signal)
                    company_in_title = company.lower() in title.lower()

                    # Check for press wire mentions
                    is_press_wire = any(wire in domain for wire in press_wire_domains)
                    if is_press_wire:
                        press_mentions.append({
                            "domain": domain,
                            "url": url[:100],
                            "title": title[:100],
                            "company_in_title": company_in_title
                        })

                    # Check for trusted domain mentions
                    is_trusted_tld = any(domain.endswith(tld) for tld in trusted_tlds)
                    is_high_authority = any(auth in domain for auth in high_authority_domains)

                    if is_trusted_tld or is_high_authority:
                        trusted_mentions.append({
                            "domain": domain,
                            "url": url[:100],
                            "title": title[:100],
                            "company_in_title": company_in_title,
                            "type": "tld" if is_trusted_tld else "high_authority"
                        })

                # DEBUG: Print domain analysis
                unique_domains = set(all_domains_found)
                print(f"      🌐 Unique domains in articles: {len(unique_domains)}")
                print(f"      📰 Press wire matches: {len(press_mentions)}")
                print(f"      🏛️  Trusted domain matches: {len(trusted_mentions)}")

                # Show sample of domains if we didn't find any matches
                if len(press_mentions) == 0 and len(trusted_mentions) == 0 and len(unique_domains) > 0:
                    sample_domains = list(unique_domains)[:10]
                    print(f"      🔍 Sample domains (showing first 10):")
                    for d in sample_domains:
                        print(f"         - {d}")

                # Calculate score
                # Press wire mentions: 0-5 points
                # - 1+ mention: 2 points
                # - 3+ mentions: 3 points
                # - 5+ mentions: 4 points
                # - 10+ mentions: 5 points
                press_score = 0
                if len(press_mentions) >= 10:
                    press_score = 5.0
                elif len(press_mentions) >= 5:
                    press_score = 4.0
                elif len(press_mentions) >= 3:
                    press_score = 3.0
                elif len(press_mentions) >= 1:
                    press_score = 2.0

                # Trusted domain mentions: 0-5 points
                # - 1+ mention: 2 points
                # - 3+ mentions: 3 points
                # - 5+ mentions: 4 points
                # - 10+ mentions: 5 points
                trusted_score = 0
                if len(trusted_mentions) >= 10:
                    trusted_score = 5.0
                elif len(trusted_mentions) >= 5:
                    trusted_score = 4.0
                elif len(trusted_mentions) >= 3:
                    trusted_score = 3.0
                elif len(trusted_mentions) >= 1:
                    trusted_score = 2.0

                total_score = press_score + trusted_score

                print(f"      ✅ GDELT: {total_score}/10 pts (Press: {press_score}/5, Trusted: {trusted_score}/5)")
                print(f"         Press wires: {len(press_mentions)}, Trusted domains: {len(trusted_mentions)}")

                return total_score, {
                    "checked": True,
                    "score": total_score,
                    "press_score": press_score,
                    "trusted_score": trusted_score,
                    "press_mentions_count": len(press_mentions),
                    "trusted_mentions_count": len(trusted_mentions),
                    "press_mentions": press_mentions[:5],  # Sample of top 5
                    "trusted_mentions": trusted_mentions[:5],  # Sample of top 5
                    "reason": f"GDELT coverage: {len(press_mentions)} press mentions, {len(trusted_mentions)} trusted domain mentions"
                }

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "GDELT API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"GDELT check error: {str(e)}"}


async def check_companies_house(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check UK Companies House registry.

    Returns score (0-10) based on company found in UK Companies House.
    This is a SOFT check - always passes, appends score.
    Uses UK Companies House API (free, requires API key registration).

    API Key: Register at https://developer.company-information.service.gov.uk/
    If API key not configured, returns 0 points and continues.

    Args:
        lead: Lead data with company

    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}

        if not COMPANIES_HOUSE_API_KEY or COMPANIES_HOUSE_API_KEY == "":
            print(f"   ❌ Companies House: API key not configured - skipping check (0 points)")
            return 0, {
                "checked": True,
                "score": 0,
                "reason": "Companies House API key not configured (register at https://developer.company-information.service.gov.uk/)"
            }

        print(f"   🔍 Companies House: Searching for '{company}'")

        import base64
        auth_b64 = base64.b64encode(f"{COMPANIES_HOUSE_API_KEY}:".encode()).decode()
        search_url = "https://api.company-information.service.gov.uk/search/companies"

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Basic {auth_b64}"}

            async with session.get(
                search_url,
                headers=headers,
                params={"q": company, "items_per_page": 5},
                timeout=10,
                proxy=HTTP_PROXY_URL
            ) as response:
                if response.status != 200:
                    return 0, {"checked": False, "reason": f"Companies House API error: HTTP {response.status}"}

                data = await response.json()
                items = data.get("items", [])

                if not items:
                    print(f"      ❌ Companies House: No results found")
                    return 0, {"checked": True, "score": 0, "reason": "Company not found in UK Companies House"}

                company_upper = company.upper()
                for item in items[:5]:
                    ch_name = item.get("title", "").upper()
                    status = item.get("company_status", "").lower()

                    if company_upper == ch_name:
                        score = 10.0 if status == "active" else 8.0
                    elif company_upper in ch_name or ch_name in company_upper:
                        score = 8.0 if status == "active" else 6.0
                    else:
                        continue

                    print(f"      ✅ Companies House: Found - {item.get('title')} ({status})")
                    return score, {
                        "checked": True,
                        "score": score,
                        "matched_company": item.get("title"),
                        "company_status": status
                    }

                return 0, {"checked": True, "score": 0, "reason": "No close name match"}

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "Companies House API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"Companies House check error: {str(e)}"}


async def check_whois_dnsbl_reputation(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: WHOIS + DNSBL reputation check using cached validator data.

    Returns score (0-10) based on:
    - WHOIS Stability: 0-3 points (whois_updated_days_ago)
    - Registrant Consistency: 0-3 points (corporate signals)
    - Hosting Provider: 0-3 points (nameservers)
    - DNSBL: 0-1 points (not blacklisted)

    This is a SOFT check - always passes, appends score.
    Uses FREE data already collected in Stage 1 (WHOIS) and Stage 2 (DNSBL).

    Mirrors TypeScript calculate-rep-score/checks/operational.ts checks.

    Args:
        lead: Lead data with WHOIS and DNSBL fields

    Returns:
        (score, metadata)
    """
    try:
        score = 0
        details = {
            "whois_stability": 0,
            "registrant_consistency": 0,
            "hosting_provider": 0,
            "dnsbl": 0
        }

        # ============================================================
        # 1. WHOIS Stability (0-3 points)
        # ============================================================
        # TypeScript: checkWhoisStabilityDays() - 4 points
        # Python: 3 points (scaled down for 10-point total)
        #
        # Checks if WHOIS record was updated recently (instability signal)
        # Recent updates indicate potential domain instability, ownership changes,
        # or drop-catch scenarios
        # ============================================================

        whois_updated_days = lead.get("whois_updated_days_ago")
        if isinstance(whois_updated_days, (int, float)) and whois_updated_days >= 0:
            # Scoring:
            # >= 180 days (6 months): 3.0 points (very stable)
            # >= 90 days (3 months): 2.0 points (stable)
            # >= 30 days (1 month): 1.0 points (acceptable)
            # < 30 days: 0 points (unstable)
            if whois_updated_days >= 180:
                details["whois_stability"] = 3.0
            elif whois_updated_days >= 90:
                details["whois_stability"] = 2.0
            elif whois_updated_days >= 30:
                details["whois_stability"] = 1.0
            else:
                details["whois_stability"] = 0

            score += details["whois_stability"]
            details["whois_updated_days_ago"] = whois_updated_days
        else:
            # Fallback: Use domain age if WHOIS update date not available
            domain_age = lead.get("domain_age_days")
            if isinstance(domain_age, (int, float)) and domain_age > 30:
                # Old domain, assume stable (weak signal)
                details["whois_stability"] = 1.0
                score += 1.0
                details["whois_updated_days_ago"] = "unavailable (used domain_age fallback)"

        # ============================================================
        # 2. Registrant Consistency (0-3 points)
        # ============================================================
        # TypeScript: checkRegistrantConsistency() - 3 points
        # Python: 3 points
        #
        # Counts corporate signals:
        # - Corporate registrar name (Inc, LLC, Corp, etc.)
        # - Reputable hosting providers in nameservers
        # - Established domain (> 1 year old)
        # ============================================================

        corporate_signals = []

        # Check registrar for corporate keywords
        registrar = lead.get("domain_registrar", "")
        if registrar:
            corporate_keywords = ["inc", "corp", "llc", "ltd", "company", "corporation",
                                 "enterprises", "group", "holdings"]
            registrar_lower = registrar.lower()
            if any(keyword in registrar_lower for keyword in corporate_keywords):
                corporate_signals.append("corporate_registrant")

        # Check for reputable hosting providers in nameservers
        nameservers = lead.get("domain_nameservers", [])
        if isinstance(nameservers, list) and len(nameservers) > 0:
            reputable_providers = ["aws", "google", "cloudflare", "azure", "amazon"]
            for ns in nameservers:
                ns_lower = str(ns).lower()
                if any(provider in ns_lower for provider in reputable_providers):
                    corporate_signals.append("reputable_hosting")
                    break

        # Check domain age (> 1 year = established)
        domain_age = lead.get("domain_age_days", 0)
        if domain_age > 365:
            corporate_signals.append("established_domain")

        # Score based on signals count
        # 3+ signals: 3 points
        # 2 signals: 2 points
        # 1 signal: 1 point
        # 0 signals: 0 points
        if len(corporate_signals) >= 3:
            details["registrant_consistency"] = 3.0
        elif len(corporate_signals) == 2:
            details["registrant_consistency"] = 2.0
        elif len(corporate_signals) == 1:
            details["registrant_consistency"] = 1.0
        else:
            details["registrant_consistency"] = 0

        score += details["registrant_consistency"]
        details["corporate_signals"] = corporate_signals

        # ============================================================
        # 3. Hosting Provider Reputation (0-3 points)
        # ============================================================
        # TypeScript: checkHostingProviderReputation() - 3 points
        # Python: 3 points
        #
        # Checks if domain is hosted on reputable infrastructure:
        # AWS, Google Cloud, Cloudflare, Azure, Amazon
        # ============================================================

        if isinstance(nameservers, list) and len(nameservers) > 0:
            reputable_providers = ["aws", "google", "cloudflare", "azure", "amazon"]
            found_provider = None

            for ns in nameservers:
                ns_lower = str(ns).lower()
                for provider in reputable_providers:
                    if provider in ns_lower:
                        found_provider = provider
                        break
                if found_provider:
                    break

            if found_provider:
                details["hosting_provider"] = 3.0
                details["hosting_provider_name"] = found_provider
                score += 3.0

        # ============================================================
        # 4. DNSBL Reputation (0-1 points)
        # ============================================================
        # TypeScript: checkDnsblReputation() - 1 point
        # Python: 1 point
        #
        # Checks if domain is NOT blacklisted in Spamhaus DBL
        # Uses FREE data already collected in Stage 2
        # ============================================================

        dnsbl_checked = lead.get("dnsbl_checked")
        dnsbl_blacklisted = lead.get("dnsbl_blacklisted")

        if dnsbl_checked:
            if not dnsbl_blacklisted:
                details["dnsbl"] = 1.0
                score += 1.0
                details["dnsbl_status"] = "clean"
            else:
                details["dnsbl"] = 0
                details["dnsbl_status"] = "blacklisted"
                details["dnsbl_list"] = lead.get("dnsbl_list", "unknown")

        # ============================================================
        # Return final score and details
        # ============================================================

        return score, {
            "checked": True,
            "score": score,
            "max_score": 10,
            "details": details,
            "reason": f"WHOIS/DNSBL reputation: {score:.1f}/10 (Stability: {details['whois_stability']}, Consistency: {details['registrant_consistency']}, Hosting: {details['hosting_provider']}, DNSBL: {details['dnsbl']})"
        }

    except Exception as e:
        return 0, {
            "checked": False,
            "reason": f"WHOIS/DNSBL check error: {str(e)}"
        }


async def check_terms_attestation(lead: dict) -> Tuple[bool, dict]:
    """
    Verify miner's attestation metadata against Supabase database (SOURCE OF TRUTH).

    Security Checks:
    1. Query contributor_attestations table for wallet's attestation record
    2. Reject if no valid attestation exists (prevents local file manipulation)
    3. Verify lead metadata matches Supabase attestation record
    4. Validate terms version and boolean attestations

    This is Stage -1 (runs BEFORE all other checks) to ensure regulatory compliance.
    """
    from Leadpoet.utils.contributor_terms import TERMS_VERSION_HASH
    from Leadpoet.utils.cloud_db import get_supabase_client

    # Check required attestation fields in lead
    required_fields = ["wallet_ss58", "terms_version_hash", "lawful_collection",
                      "no_restricted_sources", "license_granted"]

    missing = [f for f in required_fields if f not in lead]
    if missing:
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Missing attestation fields: {', '.join(missing)}",
            "failed_fields": missing
        }

    wallet_ss58 = lead.get("wallet_ss58")
    lead_terms_hash = lead.get("terms_version_hash")

    # SECURITY CHECK 1: Query Supabase for authoritative attestation record
    try:
        supabase = get_supabase_client()
        if not supabase:
            # If Supabase not available, log warning but don't fail validation
            # This prevents breaking validators during network issues
            print(f"   ⚠️  Supabase client not available - skipping attestation verification")
            return True, {}

        result = supabase.table("contributor_attestations")\
            .select("*")\
            .eq("wallet_ss58", wallet_ss58)\
            .eq("terms_version_hash", TERMS_VERSION_HASH)\
            .eq("accepted", True)\
            .execute()

        # SECURITY CHECK 2: Reject if no valid attestation in database
        if not result.data or len(result.data) == 0:
            return False, {
                "stage": "Stage -1: Terms Attestation",
                "check_name": "check_terms_attestation",
                "message": f"No valid attestation found in database for wallet {wallet_ss58[:10]}...",
                "failed_fields": ["wallet_ss58"]
            }

        # Attestation exists in Supabase - miner has legitimately accepted terms
        supabase_attestation = result.data[0]

    except Exception as e:
        # Log error but don't fail validation - prevents breaking validators
        print(f"   ⚠️  Failed to verify attestation in database: {str(e)}")
        return True, {}

    # SECURITY CHECK 3: Verify lead metadata matches Supabase record
    if lead_terms_hash != supabase_attestation.get("terms_version_hash"):
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Lead attestation hash mismatch (lead: {lead_terms_hash[:8]}, db: {supabase_attestation.get('terms_version_hash', '')[:8]})",
            "failed_fields": ["terms_version_hash"]
        }

    # Check: Verify terms version is current
    if lead_terms_hash != TERMS_VERSION_HASH:
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Outdated terms version (lead: {lead_terms_hash[:8]}, current: {TERMS_VERSION_HASH[:8]})",
            "failed_fields": ["terms_version_hash"]
        }

    # Check: Verify boolean attestations in lead
    if not all([lead.get("lawful_collection"),
                lead.get("no_restricted_sources"),
                lead.get("license_granted")]):
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": "Incomplete attestations",
            "failed_fields": ["lawful_collection", "no_restricted_sources", "license_granted"]
        }

    return True, {}


async def check_source_provenance(lead: dict) -> Tuple[bool, dict]:
    """
    Verify source provenance metadata.

    Validates:
    - source_url is present and valid
    - source_type is in allowed list
    - Domain not in restricted sources denylist
    - Domain age >= 7 days (reuses existing check)

    This ensures miners are providing valid source information and not using
    prohibited data brokers without proper authorization.
    """
    from Leadpoet.utils.source_provenance import (
        validate_source_url,
        is_restricted_source,
        extract_domain_from_url
    )

    # Check required fields
    source_url = lead.get("source_url")
    source_type = lead.get("source_type")

    if not source_url:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": "Missing source_url",
            "failed_fields": ["source_url"]
        }

    if not source_type:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": "Missing source_type",
            "failed_fields": ["source_type"]
        }

    # Validate source_type against allowed list
    valid_types = ["public_registry", "company_site", "first_party_form",
                   "licensed_resale", "proprietary_database"]
    if source_type not in valid_types:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": f"Invalid source_type: {source_type}",
            "failed_fields": ["source_type"]
        }

    # Validate source URL (checks denylist, domain age, reachability)
    # SECURITY: Pass source_type to prevent spoofing proprietary_database
    try:
        is_valid, reason = await validate_source_url(source_url, source_type)
        if not is_valid:
            return False, {
                "stage": "Stage 0.5: Source Provenance",
                "check_name": "check_source_provenance",
                "message": f"Source URL validation failed: {reason}",
                "failed_fields": ["source_url"]
            }
    except Exception as e:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": f"Error validating source URL: {str(e)}",
            "failed_fields": ["source_url"]
        }

    # Additional check: Extract domain and verify not restricted
    # (This is redundant with validate_source_url but provides explicit feedback)
    domain = extract_domain_from_url(source_url)
    if domain and is_restricted_source(domain):
        # Only fail if NOT a licensed resale (those are handled in next check)
        if source_type != "licensed_resale":
            return False, {
                "stage": "Stage 0.5: Source Provenance",
                "check_name": "check_source_provenance",
                "message": f"Source domain {domain} is in restricted denylist",
                "failed_fields": ["source_url"]
            }

    return True, {}


async def check_licensed_resale_proof(lead: dict) -> Tuple[bool, dict]:
    """
    Validate license document proof for licensed resale submissions.

    If source_type = "licensed_resale", validates that:
    - license_doc_hash is present
    - license_doc_hash is valid SHA-256 format

    This allows miners to use restricted data brokers (ZoomInfo, Apollo, etc.)
    IF they have a valid resale agreement and provide cryptographic proof.
    """
    from Leadpoet.utils.source_provenance import validate_licensed_resale

    source_type = lead.get("source_type")

    # Only validate if this is a licensed resale submission
    if source_type != "licensed_resale":
        return True, {}

    # Validate license proof
    is_valid, reason = validate_licensed_resale(lead)

    if not is_valid:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_licensed_resale_proof",
            "message": reason,
            "failed_fields": ["license_doc_hash"]
        }

    # Log for audit trail
    license_hash = lead.get("license_doc_hash", "")
    print(f"   📄 Licensed resale detected: hash={license_hash[:16]}...")

    return True, {}
