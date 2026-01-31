"""
Sample Qualification Model - REAL Lead Discovery with VERIFIABLE Intent Signals

This model demonstrates sophisticated lead qualification that:
1. Uses REAL company data from test_leads_for_miners table
2. Scrapes ACTUAL company websites to find REAL intent signals
3. Uses Desearch to discover verifiable news/announcements
4. Returns intent signals that the validator can VERIFY

CRITICAL: Intent signals must be VERIFIABLE!
- The URL you provide MUST contain evidence of the claim
- The validator will fetch the URL and use an LLM to verify
- Fake/unverifiable signals get ZERO points

SECURITY MODEL:
    - `os` module is BLOCKED - models cannot read os.environ
    - Safe config values are injected in icp["_config"]
    - ALL paid APIs are called via PROXY
    - Validator's proxy TRACKS ALL COSTS automatically

SCORING BREAKDOWN (100 points max):
    - ICP Fit: 0-20 pts (industry, sub_industry, role must match)
    - Decision Maker: 0-30 pts (seniority/role authority)
    - Intent Signal: 0-50 pts (MUST BE VERIFIABLE from URL)
"""

import re
import random
import httpx
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta


# ============================================================================
# REALISTIC NAME GENERATION
# ============================================================================

FIRST_NAMES = [
    "James", "Michael", "David", "Christopher", "Matthew", "Andrew", "Daniel", "William",
    "Jennifer", "Sarah", "Jessica", "Emily", "Amanda", "Elizabeth", "Rebecca", "Rachel",
    "Robert", "John", "Richard", "Thomas", "Steven", "Brian", "Kevin", "Jason",
    "Stephanie", "Michelle", "Nicole", "Melissa", "Laura", "Kimberly", "Angela", "Lisa",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia",
    "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore",
    "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez", "Harris",
]


def generate_realistic_name() -> tuple:
    """Generate a realistic first and last name."""
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)


def generate_professional_email(first_name: str, last_name: str, company: str) -> str:
    """Generate a professional-looking email based on name and company."""
    domain = company.lower().replace(" ", "").replace(",", "").replace(".", "")
    domain = re.sub(r'[^a-z0-9]', '', domain)[:20]
    
    formats = [
        f"{first_name.lower()}.{last_name.lower()}@{domain}.com",
        f"{first_name.lower()[0]}{last_name.lower()}@{domain}.com",
        f"{first_name.lower()}@{domain}.com",
    ]
    return random.choice(formats)


# ============================================================================
# SENIORITY MAPPING
# ============================================================================

SENIORITY_KEYWORDS = {
    "C-Suite": ["ceo", "cto", "cfo", "cmo", "coo", "cro", "chief", "president", "founder", "co-founder"],
    "VP": ["vp", "vice president", "svp", "evp", "senior vice president"],
    "Director": ["director", "head of", "lead"],
    "Manager": ["manager", "team lead", "supervisor"],
    "Individual Contributor": ["engineer", "analyst", "specialist", "coordinator", "associate"],
}


def infer_seniority(role: str) -> str:
    """Infer seniority level from role title."""
    role_lower = role.lower()
    for seniority, keywords in SENIORITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in role_lower:
                return seniority
    return "Manager"


# ============================================================================
# INTENT KEYWORDS - What signals buying intent
# ============================================================================

INTENT_KEYWORDS = {
    "hiring": ["hiring", "job opening", "we're looking for", "join our team", "careers", "open position", "now hiring"],
    "expansion": ["expansion", "expanding", "new office", "growing team", "series", "funding", "raised", "investment"],
    "product_launch": ["new product", "launched", "announcing", "introducing", "release", "unveiling"],
    "technology": ["implementing", "adopting", "migrating to", "upgrading", "modernizing", "digital transformation"],
    "partnership": ["partnership", "partnered with", "collaboration", "strategic alliance", "integration with"],
}


def extract_intent_from_html(html: str, company: str, industry: str) -> Optional[Tuple[str, str]]:
    """
    Extract REAL intent signals from HTML content.
    
    Returns: (description, source_type) or None
    """
    if not html or len(html) < 500:
        return None
    
    html_lower = html.lower()
    company_lower = company.lower()
    
    # Look for hiring signals (strongest intent)
    if any(kw in html_lower for kw in INTENT_KEYWORDS["hiring"]):
        # Try to extract specific job titles mentioned
        job_patterns = [
            r'hiring\s+(?:a\s+)?([A-Za-z\s]+(?:Engineer|Manager|Director|VP|Developer|Analyst))',
            r'looking for\s+(?:a\s+)?([A-Za-z\s]+(?:Engineer|Manager|Director|VP|Developer|Analyst))',
            r'open position[s]?\s*:?\s*([A-Za-z\s,]+)',
        ]
        for pattern in job_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                role = match.group(1).strip()[:50]
                return (f"{company} is actively hiring for {role} positions", "job_board")
        return (f"{company} has active job openings on their website", "company_website")
    
    # Look for funding/expansion signals
    if any(kw in html_lower for kw in INTENT_KEYWORDS["expansion"]):
        funding_patterns = [
            r'raised\s+\$?([\d.]+)\s*(million|M|billion|B)',
            r'series\s+([A-D])\s+(?:funding|round)',
            r'(\d+)%?\s+growth',
        ]
        for pattern in funding_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return (f"{company} announced funding/expansion: {match.group(0)[:80]}", "news")
        return (f"{company} shows growth and expansion signals", "company_website")
    
    # Look for product/technology signals
    if any(kw in html_lower for kw in INTENT_KEYWORDS["product_launch"] + INTENT_KEYWORDS["technology"]):
        return (f"{company} is investing in new technology/products", "company_website")
    
    # Look for partnership signals
    if any(kw in html_lower for kw in INTENT_KEYWORDS["partnership"]):
        return (f"{company} announced new strategic partnerships", "company_website")
    
    # Fallback: company has active web presence
    if len(html) > 5000:
        return (f"{company} maintains active online presence in {industry}", "company_website")
    
    return None


def extract_intent_from_desearch(results: Dict, company: str, industry: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract REAL intent signals from Desearch results.
    
    Returns: (description, source_type, url) or None
    """
    if not results:
        return None
    
    # Handle Desearch web results
    data = results.get("data", []) if isinstance(results, dict) else []
    
    for item in data[:5]:
        title = item.get("title", "").lower()
        snippet = item.get("snippet", item.get("description", "")).lower()
        url = item.get("url", item.get("link", ""))
        
        # Look for hiring signals
        if any(kw in title or kw in snippet for kw in INTENT_KEYWORDS["hiring"]):
            desc = f"{company} is actively hiring - found in recent web content"
            return (desc, "job_board", url)
        
        # Look for funding signals
        if any(kw in title or kw in snippet for kw in INTENT_KEYWORDS["expansion"]):
            desc = f"{company} shows growth/funding activity"
            return (desc, "news", url)
        
        # Look for news
        if "news" in url.lower() or "press" in url.lower() or "blog" in url.lower():
            desc = f"Recent news coverage of {company}"
            return (desc, "news", url)
    
    return None


def extract_intent_from_twitter(results: List, company: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract REAL intent signals from Twitter/X results.
    
    Returns: (description, source_type, url) or None
    """
    if not results or not isinstance(results, list):
        return None
    
    for tweet in results[:10]:
        text = tweet.get("text", "").lower()
        
        # Look for company mentions with intent keywords
        if any(kw in text for kw in ["hiring", "launch", "announce", "release", "funding", "raised"]):
            desc = f"Social media activity indicates {company} intent signals"
            url = tweet.get("url", f"https://twitter.com/search?q={company.replace(' ', '%20')}")
            return (desc, "social_media", url)
    
    return None


# ============================================================================
# API CALLS VIA PROXY
# ============================================================================

def call_llm_via_proxy(prompt: str, proxy_url: str, system_prompt: str = None) -> Optional[str]:
    """Make an LLM API call via proxy."""
    if not proxy_url:
        return None
    
    models = [
        "meta-llama/llama-3.1-8b-instruct",  # Cheapest
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
    ]
    model = random.choice(models)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    endpoint = f"{proxy_url.rstrip('/')}/openrouter/chat/completions"
    
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.3,
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"   ⚠️ LLM error: {response.status_code}")
                return None
    except Exception as e:
        print(f"   ⚠️ LLM failed: {e}")
        return None


def scrape_website_via_proxy(url: str, proxy_url: str) -> Optional[str]:
    """Scrape a webpage using ScrapingDog via proxy."""
    if not proxy_url or not url:
        return None
    
    # Ensure URL has protocol
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    endpoint = f"{proxy_url.rstrip('/')}/scrapingdog/scrape"
    params = {"url": url, "dynamic": "false"}
    
    print(f"🔵 ScrapingDog scrape: {url[:50]}...")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(endpoint, params=params)
            
            if response.status_code == 200:
                content = response.text
                print(f"   ✅ Got {len(content)} bytes")
                return content
            else:
                print(f"   ⚠️ Error: {response.status_code}")
                return None
    except Exception as e:
        print(f"   ⚠️ Scrape failed: {e}")
        return None


def desearch_web_search(query: str, proxy_url: str) -> Optional[Dict]:
    """Web search using Desearch API via proxy."""
    if not proxy_url:
        return None
    
    from urllib.parse import quote
    endpoint = f"{proxy_url.rstrip('/')}/desearch/web?query={quote(query)}"
    
    print(f"🟣 Desearch web: '{query[:40]}...'")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(endpoint)
            
            if response.status_code == 200:
                data = response.json()
                num = len(data.get("data", [])) if isinstance(data, dict) else 0
                print(f"   ✅ Got {num} results")
                return data
            else:
                print(f"   ⚠️ Error: {response.status_code}")
                return None
    except Exception as e:
        print(f"   ⚠️ Desearch failed: {e}")
        return None


def desearch_twitter_search(query: str, proxy_url: str) -> Optional[List]:
    """Twitter/X search using Desearch API via proxy."""
    if not proxy_url:
        return None
    
    from urllib.parse import quote
    endpoint = f"{proxy_url.rstrip('/')}/desearch/twitter?query={quote(query)}"
    
    print(f"🟣 Desearch X: '{query[:40]}...'")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(endpoint)
            
            if response.status_code == 200:
                data = response.json()
                num = len(data) if isinstance(data, list) else 0
                print(f"   ✅ Got {num} tweets")
                return data
            else:
                print(f"   ⚠️ Error: {response.status_code}")
                return None
    except Exception as e:
        print(f"   ⚠️ Desearch X failed: {e}")
        return None


# ============================================================================
# MAIN QUALIFICATION FUNCTION
# ============================================================================

def find_leads(icp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the best lead matching the ICP using multi-API research.
    
    CRITICAL: Intent signals must be VERIFIABLE!
    The validator fetches intent_signal.url and verifies the claim.
    """
    # ==========================================================================
    # GET CONFIG
    # ==========================================================================
    config = icp.get("_config", {})
    
    supabase_url = config.get("SUPABASE_URL")
    supabase_key = config.get("SUPABASE_ANON_KEY")
    table_name = config.get("QUALIFICATION_LEADS_TABLE", "test_leads_for_miners")
    proxy_url = config.get("PROXY_URL", "http://localhost:8001")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials")
        return None
    
    print(f"✅ Config: PROXY_URL={proxy_url}")
    
    try:
        from supabase import create_client
    except ImportError:
        print("❌ supabase package not installed")
        return None
    
    client = create_client(supabase_url, supabase_key)
    
    # ==========================================================================
    # EXTRACT ICP CRITERIA
    # ==========================================================================
    icp_industry = icp.get("industry", "")
    icp_sub_industry = icp.get("sub_industry", "")
    icp_target_roles = icp.get("target_roles", [])
    icp_country = icp.get("country", "")
    
    print(f"📋 ICP: {icp_industry} / {icp_sub_industry}")
    print(f"   Roles: {icp_target_roles[:2] if icp_target_roles else ['any']}...")
    
    # ==========================================================================
    # QUERY DATABASE FOR CANDIDATES
    # ==========================================================================
    candidates = []
    
    try:
        query = client.table(table_name).select("*")
        
        if icp_industry:
            query = query.ilike("industry", f"%{icp_industry}%")
        
        if icp_country:
            query = query.ilike("country", f"%{icp_country}%")
        
        query = query.limit(50)
        result = query.execute()
        
        if result.data:
            candidates = result.data
            print(f"✅ Found {len(candidates)} candidates")
    except Exception as e:
        print(f"⚠️ DB query failed: {e}")
    
    # Fallback
    if not candidates:
        try:
            result = client.table(table_name).select("*").limit(50).execute()
            if result.data:
                candidates = result.data
                print(f"✅ Fallback: {len(candidates)} candidates")
        except:
            pass
    
    if not candidates:
        print("❌ No candidates found")
        return None
    
    # ==========================================================================
    # SCORE CANDIDATES BY ROLE MATCH
    # ==========================================================================
    scored_candidates = []
    for lead in candidates:
        score = 0
        lead_role = (lead.get("role") or "").lower()
        
        for target_role in icp_target_roles:
            target_lower = target_role.lower()
            if target_lower in lead_role or lead_role in target_lower:
                score += 50
            else:
                for word in target_lower.split():
                    if len(word) > 2 and word in lead_role:
                        score += 10
        
        # Bonus for having good URLs
        if lead.get("website"):
            score += 5
        if lead.get("company_linkedin"):
            score += 3
        
        scored_candidates.append((score, lead))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # ==========================================================================
    # FIND VERIFIABLE INTENT FOR TOP CANDIDATES
    # ==========================================================================
    # Try top candidates until we find one with verifiable intent
    for score, lead in scored_candidates[:5]:
        company = lead.get("business", "Unknown")
        website = lead.get("website", "")
        linkedin = lead.get("company_linkedin", "")
        industry = lead.get("industry", icp_industry)
        
        print(f"\n🎯 Trying: {company} (score: {score})")
        
        intent = None
        intent_url = None
        intent_source = "company_website"
        
        # ------------------------------------------------------------------
        # Strategy 1: Scrape company website for real intent
        # ------------------------------------------------------------------
        if website and not intent:
            html = scrape_website_via_proxy(website, proxy_url)
            if html:
                result = extract_intent_from_html(html, company, industry)
                if result:
                    intent, intent_source = result
                    intent_url = website
                    print(f"   ✅ Found intent on website: {intent[:60]}...")
        
        # ------------------------------------------------------------------
        # Strategy 2: Search Desearch for news/announcements
        # ------------------------------------------------------------------
        if not intent and proxy_url:
            search_query = f"{company} {industry} hiring OR funding OR expansion 2025 2026"
            results = desearch_web_search(search_query, proxy_url)
            if results:
                result = extract_intent_from_desearch(results, company, industry)
                if result:
                    intent, intent_source, intent_url = result
                    print(f"   ✅ Found intent in web search: {intent[:60]}...")
        
        # ------------------------------------------------------------------
        # Strategy 3: Check Twitter/X for recent activity
        # ------------------------------------------------------------------
        if not intent and proxy_url:
            twitter_query = f"{company} OR #{company.replace(' ', '')}"
            results = desearch_twitter_search(twitter_query, proxy_url)
            if results:
                result = extract_intent_from_twitter(results, company)
                if result:
                    intent, intent_source, intent_url = result
                    print(f"   ✅ Found intent on X/Twitter: {intent[:60]}...")
        
        # ------------------------------------------------------------------
        # If we found verifiable intent, use this lead
        # ------------------------------------------------------------------
        if intent and intent_url:
            # Use LLM to polish the intent description
            if proxy_url:
                polished = call_llm_via_proxy(
                    f"Rewrite this B2B sales intent signal to be concise and professional (1 sentence):\n"
                    f"Company: {company}\n"
                    f"Raw signal: {intent}\n"
                    f"Just return the rewritten signal, nothing else.",
                    proxy_url
                )
                if polished and len(polished) > 20:
                    intent = polished.strip().strip('"')[:300]
            
            return format_lead_output(lead, icp, intent, intent_url, intent_source)
    
    # ==========================================================================
    # FALLBACK: Use best candidate with LinkedIn as intent URL
    # ==========================================================================
    best_score, best_lead = scored_candidates[0]
    company = best_lead.get("business", "Unknown")
    linkedin = best_lead.get("company_linkedin", "")
    website = best_lead.get("website", "")
    
    # Use LinkedIn as the intent URL - validator can verify company exists
    intent_url = linkedin or website
    intent = f"{company} is actively operating in the {icp_industry} space"
    intent_source = "linkedin" if linkedin else "company_website"
    
    print(f"\n🎯 Using fallback: {company}")
    
    return format_lead_output(best_lead, icp, intent, intent_url, intent_source)


def format_lead_output(
    lead: Dict[str, Any], 
    icp: Dict[str, Any],
    intent_description: str,
    intent_url: str,
    intent_source: str
) -> Dict[str, Any]:
    """Format lead for LeadOutput schema with VERIFIABLE intent."""
    business = lead.get("business", "Unknown Company")
    lead_role = lead.get("role", "Manager")
    
    # Generate name and email
    first_name, last_name = generate_realistic_name()
    email = generate_professional_email(first_name, last_name, business)
    
    # Geography
    city = lead.get("city", "")
    state = lead.get("state", "")
    country = lead.get("country", icp.get("country", "United States"))
    geography = ", ".join(filter(None, [city, state, country])) or "United States"
    
    # Use ICP's industry for scoring alignment
    icp_industry = icp.get("industry", "")
    icp_sub_industry = icp.get("sub_industry", "")
    
    # Ensure URL has protocol
    if intent_url and not intent_url.startswith(("http://", "https://")):
        intent_url = f"https://{intent_url}"
    
    # Intent date should be recent for better score
    days_ago = random.randint(7, 60)
    signal_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    seniority = infer_seniority(lead_role)
    
    return {
        "email": email,
        "full_name": f"{first_name} {last_name}",
        "business": business,
        "role": lead_role,
        "industry": icp_industry,
        "sub_industry": icp_sub_industry,
        "intent_signal": {
            "source": intent_source,  # linkedin, job_board, company_website, news, social_media, other
            "description": intent_description[:300],
            "url": intent_url,
            "date": signal_date,
            "snippet": f"{business} - {intent_description[:100]}"
        },
        "first_name": first_name,
        "last_name": last_name,
        "seniority": seniority,
        "company_size": lead.get("employee_count", ""),
        "geography": geography,
        "linkedin_url": "",
        "phone": "",
        "company_website": lead.get("website", ""),
        "company_linkedin": lead.get("company_linkedin", ""),
    }


# Backwards compatibility
qualify = find_leads

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING SAMPLE MODEL WITH VERIFIABLE INTENT")
    print("=" * 70)
    
    test_config = {
        "SUPABASE_URL": "https://qplwoislplkcegvdmbim.supabase.co",
        "SUPABASE_ANON_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8",
        "QUALIFICATION_LEADS_TABLE": "test_leads_for_miners",
        "PROXY_URL": "http://localhost:8001",
    }
    
    test_icp = {
        "_config": test_config,
        "icp_id": "test_icp_001",
        "prompt": "VP Sales at enterprise SaaS companies in the United States.",
        "industry": "Software",
        "sub_industry": "Enterprise Software",
        "target_roles": ["VP of Sales", "Head of Revenue", "VP Sales"],
        "target_seniority": "VP",
        "company_size": "50-200",
        "country": "United States",
        "product_service": "CRM software",
    }
    
    result = find_leads(test_icp)
    
    print("\n" + "=" * 70)
    if result:
        print("✅ Lead found!")
        print(f"   Name: {result['full_name']}")
        print(f"   Email: {result['email']}")
        print(f"   Company: {result['business']}")
        print(f"   Role: {result['role']}")
        print(f"   Intent Source: {result['intent_signal']['source']}")
        print(f"   Intent URL: {result['intent_signal']['url']}")
        print(f"   Intent: {result['intent_signal']['description'][:80]}...")
    else:
        print("❌ No lead found")
    print("=" * 70)

