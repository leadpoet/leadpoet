import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, Tuple, List, Optional
from validator_models.checks_utils import OPENROUTER_KEY
from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY
from validator_models.stage5_verification import parse_employee_count
from Leadpoet.utils.utils_lead_extraction import get_employee_count

# ========================================================================
# ICP (Ideal Customer Profile) Definitions
# ========================================================================
# These are the target customer profiles we want to incentivize miners to find.
# Leads matching these criteria receive a 1.5x multiplier on their rep_score during emissions.
# Format: Each ICP is defined by Sub-Industry + Role Details
# ========================================================================

# ========================================================================
# ICP (IDEAL CUSTOMER PROFILE) DEFINITIONS
# ========================================================================
# These definitions specify high-value lead profiles for ICP multiplier scoring.
# IMPORTANT: sub_industries must use EXACT names from INDUSTRY_TAXONOMY keys
# (case-sensitive, e.g., "FinTech" not "fintech", "E-Commerce" not "ecommerce")
# ========================================================================

ICP_DEFINITIONS = [
    # ==================================================================
    # 1. AI/ML/NLP + Robotics/Aerospace/Defence — Decision Makers (+100)
    # ==================================================================
    {
        "sub_industries": [
            "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
            "Predictive Analytics", "Robotics", "Autonomous Vehicles",
            "Aerospace", "Defense and Space", "Drones",
        ],
        "role_details": [
            # C-Suite / Founders
            "ceo", "chief executive officer", "founder", "co-founder", "president",
            "cto", "chief technology officer", "coo", "chief operating officer",
            "cfo", "chief financial officer", "cmo", "chief marketing officer",
            "chief ai officer", "chief data officer", "chief product officer",
            # VP / Director
            "vp of engineering", "vp engineering", "vp of ai", "vp ai",
            "vp of machine learning", "vp machine learning", "vp of product",
            "vp of technology", "vp technology", "vp of research", "vp research",
            "head of engineering", "head of ai", "head of machine learning",
            "head of product", "head of research", "head of data",
            "engineering director", "director of engineering", "director of ai",
            "director of machine learning", "director of product", "director of research",
            # Engineering
            "software engineer", "swe", "senior software engineer", "sr swe",
            "staff software engineer", "principal software engineer", "lead software engineer",
            "software developer", "senior software developer",
        ],
        "bonus": 100,
    },

    # ==================================================================
    # 2. Cyber Security / IT Mgmt — Midwest US, 10-50 employees (+100)
    # ==================================================================
    {
        "sub_industries": ["Cyber Security", "IT Management"],
        "role_details": [
            "owner", "co-owner", "business owner",
            "founder", "co-founder",
            "ceo", "chief executive officer", "president",
        ],
        "regions": [
            "illinois", "indiana", "michigan", "ohio", "wisconsin",
            "iowa", "kansas", "minnesota", "missouri", "nebraska",
            "north dakota", "south dakota",
        ],
        "employee_ranges": ["10-50"],
        "bonus": 100,
    },

    # ==================================================================
    # 3. Cyber Security — Decision Makers, all US (+50)
    # ==================================================================
    {
        "sub_industries": ["Cyber Security", "Network Security", "IT Management",
                          "Information Services"],
        "role_details": [
            # C-Suite / Founders
            "ceo", "chief executive officer", "founder", "co-founder", "president",
            "cto", "chief technology officer", "coo", "chief operating officer",
            "ciso", "chief information security officer", "cio", "chief information officer",
            # VP / Director
            "vp of engineering", "vp engineering", "vp of security", "vp security",
            "vp of it", "vp it", "vp of technology", "vp technology",
            "head of security", "head of it", "head of engineering",
            "director of security", "director of it", "director of engineering",
            "it director", "security director",
            # Owners
            "owner", "co-owner", "business owner",
        ],
        "regions": [
            "united states", "usa", "us", "america",
            "california", "new york", "texas", "florida", "illinois", "pennsylvania",
            "ohio", "georgia", "north carolina", "michigan", "new jersey", "virginia",
            "washington", "arizona", "massachusetts", "tennessee", "indiana", "missouri",
            "maryland", "wisconsin", "colorado", "minnesota", "south carolina", "alabama",
            "louisiana", "kentucky", "oregon", "oklahoma", "connecticut", "utah", "iowa",
            "nevada", "arkansas", "mississippi", "kansas", "new mexico", "nebraska",
            "idaho", "west virginia", "hawaii", "maine", "montana", "rhode island",
            "delaware", "south dakota", "north dakota", "alaska", "vermont", "wyoming",
        ],
    },

    # ==================================================================
    # 4. UAE/Dubai Investors (+100)
    # ==================================================================
    {
        "sub_industries": [
            "Angel Investment", "Asset Management", "Hedge Funds",
            "Impact Investing", "Incubators", "Real Estate Investment",
            "Venture Capital", "Web3 Investor", "Web3 Fund", "Wealth Management",
        ],
        "role_details": [
            "partner", "general partner", "gp", "managing partner", "managing director",
            "principal", "venture partner", "investment partner", "limited partner",
            "cio", "chief investment officer", "director of investments", "vp of investments",
            "vp investments", "head of investments", "investment director",
            "portfolio manager", "fund manager", "investment manager", "asset manager",
            "founder", "co-founder", "ceo", "chief executive officer", "president",
            "investor", "venture capitalist", "vc", "investment analyst", "research analyst",
            "associate", "senior associate", "investment associate",
            "vice president", "vp", "director", "head of",
            "wealth manager", "private banker", "relationship manager",
            "family office manager", "head of family office", "family office director",
        ],
        "regions": ["united arab emirates", "uae", "dubai", "emirati"],
        "bonus": 100,
    },

    # ==================================================================
    # 5. Small/Local Businesses — Owners, US only (+50)
    # ==================================================================
    {
        "sub_industries": [
            "Local Business", "Local", "Retail", "Restaurants", "Food and Beverage",
            "Professional Services", "Home Services", "Real Estate", "Construction",
            "Automotive", "Health Care", "Fitness", "Beauty", "Consulting",
        ],
        "role_details": [
            "owner", "co-owner", "business owner", "sole proprietor", "sole operator",
            "franchise owner", "franchisee", "store owner", "shop owner",
            "founder", "co-founder", "ceo", "chief executive officer",
            "president", "managing director", "principal", "partner",
            "proprietor", "operator", "entrepreneur",
        ],
        "regions": [
            "united states", "usa", "us", "america",
            "california", "new york", "texas", "florida", "illinois", "pennsylvania",
            "ohio", "georgia", "north carolina", "michigan", "new jersey", "virginia",
            "washington", "arizona", "massachusetts", "tennessee", "indiana", "missouri",
            "maryland", "wisconsin", "colorado", "minnesota", "south carolina", "alabama",
            "louisiana", "kentucky", "oregon", "oklahoma", "connecticut", "utah", "iowa",
            "nevada", "arkansas", "mississippi", "kansas", "new mexico", "nebraska",
            "idaho", "west virginia", "hawaii", "maine", "montana", "rhode island",
            "delaware", "south dakota", "north dakota", "alaska", "vermont", "wyoming",
        ],
    },

    # ==================================================================
    # 6. Blockchain/Crypto/Web3 — Investors & Leaders (+100)
    # ==================================================================
    {
        "sub_industries": [
            "Blockchain", "Cryptocurrency", "Bitcoin", "Ethereum",
            "Web3 Investor", "Web3 Fund",
        ],
        "role_details": [
            "partner", "general partner", "gp", "managing partner", "managing director",
            "principal", "venture partner", "investment partner", "limited partner",
            "cio", "chief investment officer", "director of investments", "vp of investments",
            "vp investments", "head of investments", "investment director",
            "portfolio manager", "fund manager", "investment manager", "asset manager",
            "founder", "co-founder", "ceo", "chief executive officer",
            "investor", "venture capitalist", "vc", "investment analyst", "research analyst",
            "associate", "senior associate", "investment associate",
            "vice president", "vp", "director", "head of",
            "token fund manager", "crypto fund manager", "defi lead", "web3 investor",
        ],
        "bonus": 100,
    },

    # ==================================================================
    # 7. Biotech/Pharma — Decision Makers (+50)
    # ==================================================================
    {
        "sub_industries": [
            "Biotechnology", "Biopharma", "Pharmaceutical",
            "Genetics", "Life Science", "Bioinformatics",
            "Clinical Trials",
        ],
        "role_details": [
            # C-Suite / Founders
            "ceo", "chief executive officer", "founder", "co-founder", "president",
            "cto", "chief technology officer", "coo", "chief operating officer",
            "cfo", "chief financial officer", "cso", "chief scientific officer",
            "cmo", "chief medical officer", "chief commercial officer",
            # VP / Director
            "vp of business development", "vp business development", "head of business development",
            "director of business development", "business development director",
            "vp of partnerships", "vp partnerships", "head of partnerships",
            "director of partnerships", "vp of corporate development",
            "director of corporate development",
            "vp of operations", "vp operations", "director of operations",
            "vp of research", "vp research", "director of research", "head of research",
            "vp of commercial", "director of commercial", "head of commercial",
            "vp of regulatory", "director of regulatory", "head of regulatory",
            "vp of clinical", "director of clinical", "head of clinical",
            # BD & Strategy
            "bd lead", "business development lead", "business development manager",
            "managing director", "managing partner",
        ],
    },

    # ==================================================================
    # 8. Wealth Mgmt / VC / Hedge Funds (+100)
    # ==================================================================
    {
        "sub_industries": [
            "Asset Management", "Venture Capital", "Hedge Funds",
            "Financial Services", "Impact Investing",
        ],
        "role_details": [
            # Leadership
            "ceo", "chief executive officer", "president", "managing director", "managing partner",
            "principal", "partner", "founder", "co-founder",
            # Investment
            "cio", "chief investment officer", "director of investments", "vp of investments",
            "vp investments", "head of investments", "investment director", "investment manager",
            "portfolio manager", "head of portfolio management", "director of portfolio management",
            "senior portfolio manager", "lead portfolio manager",
            # Private Markets
            "head of private equity", "director of private equity", "vp private equity",
            "vp of private equity", "head of venture capital", "director of venture capital",
            "vp of venture capital", "vp venture capital", "head of vc", "director of vc",
            "head of real estate", "director of real estate", "vp real estate", "vp of real estate",
            "head of alternatives", "director of alternatives", "vp of alternatives",
            "head of direct investments", "director of direct investments",
            # Operations & Finance
            "coo", "chief operating officer", "director of operations", "vp of operations",
            "vp operations", "head of operations",
            "cfo", "chief financial officer", "director of finance", "vp of finance",
            "vp finance", "head of finance",
            # Wealth & Asset Management
            "family office manager", "wealth manager", "director of wealth management",
            "head of family office", "family office director", "head of wealth management",
            "asset manager", "head of asset management", "director of asset management",
        ],
        "bonus": 100,
    },

    # ==================================================================
    # 9. FinTech/Banking/Payments — Decision Makers (+50)
    # ==================================================================
    {
        "sub_industries": [
            "FinTech", "Banking", "Payments", "Financial Services",
            "Credit Cards", "Mobile Payments", "Transaction Processing",
        ],
        "role_details": [
            # C-Suite / Founders
            "ceo", "chief executive officer", "founder", "co-founder", "president",
            "cto", "chief technology officer", "coo", "chief operating officer",
            "cfo", "chief financial officer",
            # Risk & Compliance Leadership
            "cro", "chief risk officer", "vp of risk", "vp risk", "head of risk",
            "director of risk", "risk director",
            "cco", "chief compliance officer", "vp of compliance", "vp compliance",
            "head of compliance", "director of compliance", "compliance director",
            # VP / Director
            "vp of engineering", "vp engineering", "head of engineering",
            "director of engineering", "vp of product", "head of product",
            "director of product", "vp of technology", "head of technology",
            "managing director", "managing partner",
            # Compliance Operations
            "compliance officer", "senior compliance officer", "compliance manager",
            "bsa officer", "aml officer", "kyc manager", "director of aml",
            "vp of bsa", "head of bsa", "anti-money laundering officer",
            "financial crimes manager", "director of financial crimes",
            # Risk Operations
            "risk officer", "senior risk officer", "risk manager",
            "enterprise risk manager", "operational risk manager",
        ],
    },

    # ==================================================================
    # 10. Robotics / Aerospace / Defence — Technical & Leadership (+50)
    # ==================================================================
    {
        "sub_industries": [
            "Robotics", "Autonomous Vehicles", "Aerospace",
            "Defense and Space", "Drones", "3D Printing",
        ],
        "role_details": [
            # C-Suite / Founders
            "ceo", "chief executive officer", "founder", "co-founder", "president",
            "cto", "chief technology officer", "coo", "chief operating officer",
            # VP / Director
            "vp of engineering", "vp engineering", "head of engineering",
            "engineering director", "director of engineering",
            "vp of technology", "vp technology", "head of technology",
            "vp of product", "head of product", "director of product",
            "vp of research", "vp research", "director of research", "head of research",
            # Engineering
            "software engineer", "senior software engineer", "staff software engineer",
            "principal software engineer", "lead software engineer",
            "mechanical engineer", "systems engineer", "robotics engineer",
            "aerospace engineer", "controls engineer",
        ],
    },
]

# ========================================================================
# INDUSTRY TAXONOMY VALIDATION HELPERS
# ========================================================================

def get_all_valid_industries() -> set:
    """
    Get all valid industry names from industry taxonomy.
    These are the unique industry_groups across all sub-industries.

    Returns:
        Set of valid industry names (case-preserved)
    """
    industries = set()
    for sub_industry, data in INDUSTRY_TAXONOMY.items():
        for group in data.get("industries", []):
            industries.add(group)
    return industries


def get_all_valid_sub_industries() -> set:
    """
    Get all valid sub-industry names from industry taxonomy.
    These are the keys of the INDUSTRY_TAXONOMY dictionary.

    Returns:
        Set of valid sub-industry names (case-preserved)
    """
    return set(INDUSTRY_TAXONOMY.keys())


# NOTE: Industry taxonomy validation functions removed - validation now done at gateway (submit.py)


# ========================================================================
# SUB-INDUSTRY VERIFICATION HELPERS (Using Industry Taxonomy) - LEGACY
# ========================================================================

def fuzzy_match_sub_industry(claimed_sub_industry: str) -> Tuple[Optional[str], Optional[Dict], float]:
    """
    LEGACY: Fuzzy match the miner's claimed sub_industry against the industry taxonomy.
    NOTE: Industry taxonomy validation is now done at gateway (submit.py).

    Returns:
        (matched_key, taxonomy_entry, confidence) where:
        - matched_key: The exact key in INDUSTRY_TAXONOMY (or None if no match)
        - taxonomy_entry: Dict with 'industry_groups' and 'definition' (or None)
        - confidence: 0.0 to 1.0
    """
    if not claimed_sub_industry:
        return None, None, 0.0

    claimed_lower = claimed_sub_industry.strip().lower()

    # Try exact match first (case-insensitive)
    for key in INDUSTRY_TAXONOMY:
        if key.lower() == claimed_lower:
            return key, INDUSTRY_TAXONOMY[key], 1.0

    # Try contains match (if claimed is substring of taxonomy entry or vice versa)
    best_match = None
    best_confidence = 0.0

    for key in INDUSTRY_TAXONOMY:
        key_lower = key.lower()

        # Check if one contains the other
        if claimed_lower in key_lower or key_lower in claimed_lower:
            # Calculate similarity based on length ratio
            longer = max(len(claimed_lower), len(key_lower))
            shorter = min(len(claimed_lower), len(key_lower))
            confidence = shorter / longer

            if confidence > best_confidence:
                best_match = key
                best_confidence = confidence

        # Check for word overlap
        claimed_words = set(claimed_lower.replace('-', ' ').replace('/', ' ').split())
        key_words = set(key_lower.replace('-', ' ').replace('/', ' ').split())

        if claimed_words and key_words:
            overlap = len(claimed_words & key_words)
            total = len(claimed_words | key_words)
            word_confidence = overlap / total if total > 0 else 0

            if word_confidence > best_confidence:
                best_match = key
                best_confidence = word_confidence

    if best_match and best_confidence >= 0.5:
        return best_match, INDUSTRY_TAXONOMY[best_match], best_confidence

    return None, None, 0.0


def validate_industry_sub_industry_pairing(claimed_industry: str, matched_sub_industry: str) -> Tuple[bool, str]:
    """
    Validate that the miner's claimed industry is a valid industry_group for the sub_industry.

    Returns:
        (is_valid, reason)
    """
    if not matched_sub_industry or matched_sub_industry not in INDUSTRY_TAXONOMY:
        return False, f"Sub-industry '{matched_sub_industry}' not found in industry taxonomy"

    valid_groups = INDUSTRY_TAXONOMY[matched_sub_industry].get("industries", [])

    if not valid_groups:
        # Some entries have empty industry_groups (like "Association", "Commercial")
        return True, f"Sub-industry '{matched_sub_industry}' has no specific industry group requirements"

    # Normalize claimed industry for comparison
    claimed_lower = claimed_industry.strip().lower()

    for group in valid_groups:
        if group.lower() == claimed_lower:
            return True, f"Industry '{claimed_industry}' is valid for sub-industry '{matched_sub_industry}'"
        # Allow partial matches (e.g., "Technology" matches "Information Technology")
        if claimed_lower in group.lower() or group.lower() in claimed_lower:
            return True, f"Industry '{claimed_industry}' loosely matches valid group '{group}' for sub-industry '{matched_sub_industry}'"

    return False, f"Industry '{claimed_industry}' is NOT valid for sub-industry '{matched_sub_industry}'. Valid groups: {valid_groups}"


async def verify_sub_industry_with_llm(
    company: str,
    claimed_sub_industry: str,
    matched_sub_industry: str,
    definition: str,
    industry_search_results: List[Dict],
    openrouter_key: str
) -> Tuple[bool, str, float]:
    """
    Use LLM to verify that the company actually matches the claimed sub_industry,
    using the industry definition as the ground truth.

    Args:
        company: Company name
        claimed_sub_industry: What the miner claimed
        matched_sub_industry: The matched industry taxonomy key
        definition: sub-industry definition of the sub_industry
        industry_search_results: GSE search results about the company's industry
        openrouter_key: API key for OpenRouter

    Returns:
        (is_match, reasoning, confidence)
    """
    if not industry_search_results:
        return False, "No industry search results to verify against", 0.0

    # Build context from search results
    search_context = ""
    for i, result in enumerate(industry_search_results[:5], 1):
        title = result.get("title", "")
        snippet = result.get("snippet", result.get("body", ""))
        search_context += f"{i}. {title}\n   {snippet[:200]}\n"

    prompt = f"""You are verifying if a company matches a specific sub-industry classification.

COMPANY: {company}

CLAIMED SUB-INDUSTRY: {claimed_sub_industry}
MATCHED SUB-INDUSTRY CATEGORY: {matched_sub_industry}

SUB-INDUSTRY DEFINITION FOR THIS SUB-INDUSTRY:
"{definition}"

SEARCH RESULTS ABOUT THE COMPANY:
{search_context}

TASK: Based on the search results, does this company match the sub-industry definition above?

RULES:
1. The company's actual business must match the industry definition
2. Be STRICT - the company must genuinely fit the sub-industry category
3. If search results don't clearly show what the company does, return false
4. If the company operates in a DIFFERENT industry than claimed, return false

RESPOND WITH JSON ONLY:
{{
    "sub_industry_match": true/false,
    "extracted_business_type": "what the company actually does based on search results",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0
                },
                timeout=20
            ) as response:
                if response.status != 200:
                    return False, f"LLM API error: HTTP {response.status}", 0.0

                data = await response.json()
                llm_response = data["choices"][0]["message"]["content"].strip()

                # Parse JSON response
                if llm_response.startswith("```"):
                    lines = llm_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    llm_response = "\n".join(lines).strip()

                result = json.loads(llm_response)

                is_match = result.get("sub_industry_match", False)
                reasoning = result.get("reasoning", "No reasoning provided")
                confidence = float(result.get("confidence", 0.0))
                extracted_type = result.get("extracted_business_type", "Unknown")

                full_reasoning = f"{reasoning} (Detected: {extracted_type})"

                return is_match, full_reasoning, confidence

    except json.JSONDecodeError as e:
        return False, f"Failed to parse LLM response: {str(e)}", 0.0
    except Exception as e:
        return False, f"LLM verification failed: {str(e)}", 0.0


# ========================================================================
# ROLE LLM VALIDATION
# ========================================================================

ROLE_LLM_PROMPT = """You are a job title validator. Analyze each role and determine if it's a VALID professional job title.

VALID roles:
- Real job titles (CEO, Marketing Manager, Software Engineer, etc.)
- Abbreviations (VP, SVP, CFO, CMO, etc.)
- Seniority levels (Senior, Junior, Lead, Head of, etc.)
- Academic titles (Professor, Dean, Researcher, etc.)
- Government/Military titles (Colonel, Senator, Ambassador, etc.)

INVALID roles:
- Random words or gibberish (A Cargo, Hello World, Test User)
- Personal descriptions (Coffee Lover, Tech Enthusiast)
- Hobbies (Golfer, Traveler, Photographer as hobby)
- Marketing slogans or taglines
- Single letters or meaningless strings
- Product/company names used as roles
- Non-job descriptions (Looking for work, Open to opportunities)
- Degrees alone (MBA, PhD - unless part of title like "PhD Student")

For each role, respond with ONLY "VALID" or "INVALID".

Roles to validate:
{roles}

Respond in this exact format (one per line, matching order):
1. VALID or INVALID
2. VALID or INVALID
...

Be strict - when in doubt, mark as INVALID."""

ROLE_LLM_BATCH_SIZE = 20

async def batch_validate_roles_llm(roles: List[str]) -> Dict[str, bool]:
    """
    Validate roles using Gemini LLM via OpenRouter.
    Returns dict mapping role -> is_valid (True/False).
    """
    if not roles:
        return {}

    if not OPENROUTER_KEY:
        print("   ⚠️ OPENROUTER_KEY not set, skipping LLM role validation")
        return {role: True for role in roles}

    results = {}

    for batch_start in range(0, len(roles), ROLE_LLM_BATCH_SIZE):
        batch = roles[batch_start:batch_start + ROLE_LLM_BATCH_SIZE]
        batch_num = (batch_start // ROLE_LLM_BATCH_SIZE) + 1
        total_batches = (len(roles) + ROLE_LLM_BATCH_SIZE - 1) // ROLE_LLM_BATCH_SIZE

        print(f"   🤖 Role LLM batch {batch_num}/{total_batches}: Validating {len(batch)} roles...")

        roles_text = "\n".join([f"{i+1}. {role}" for i, role in enumerate(batch)])
        prompt = ROLE_LLM_PROMPT.format(roles=roles_text)

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "google/gemini-2.5-flash-lite",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0
                    }
                ) as response:
                    if response.status != 200:
                        print(f"   ⚠️ Role LLM API error: HTTP {response.status}, passing batch")
                        for role in batch:
                            results[role] = True
                        continue

                    data = await response.json()
                    llm_response = data["choices"][0]["message"]["content"].strip()
                    lines = llm_response.strip().split('\n')

                    for i, role in enumerate(batch):
                        if i < len(lines):
                            line = lines[i].strip().upper()
                            is_valid = "VALID" in line and "INVALID" not in line
                            results[role] = is_valid
                            if not is_valid:
                                print(f"      ❌ LLM rejected: '{role}'")
                        else:
                            results[role] = True

        except asyncio.TimeoutError:
            print(f"   ⚠️ Role LLM timeout, passing batch")
            for role in batch:
                results[role] = True
        except Exception as e:
            print(f"   ⚠️ Role LLM error: {e}, passing batch")
            for role in batch:
                results[role] = True

    valid_count = sum(1 for v in results.values() if v)
    invalid_count = sum(1 for v in results.values() if not v)
    print(f"   📊 Role LLM results: {valid_count} valid, {invalid_count} invalid")

    return results


# ========================================================================
# ICP MULTIPLIER AND ADJUSTMENT FUNCTIONS
# ========================================================================

def determine_icp_multiplier(lead: dict) -> float:
    """
    LEGACY FUNCTION - Kept for backwards compatibility.

    Determine if a lead matches our ICP (Ideal Customer Profile) criteria.

    Uses the ICP_DEFINITIONS table (defined at top of file) to check if a lead matches
    any target customer profile based on:
    - Sub-Industry (e.g., "Gas Stations", "AI Startups")
    - Role Type (e.g., "Operations", "Technology", "Leadership")
    - Role Details (specific titles like "CEO", "CTO", "VP of Operations")
    - Region (optional - e.g., "Africa" for streaming/broadcast ICP)

    Returns:
        Custom multiplier if defined in ICP (e.g., 5.0 for Africa)
        1.5 if lead matches ICP criteria (default)
        1.0 if lead is standard (non-ICP)

    NOTE: This function is deprecated. Use calculate_icp_adjustment() instead.
    """
    # Extract lead fields (case-insensitive)
    sub_industry = lead.get("sub_industry", "").strip().lower()
    role = lead.get("role", "").strip().lower()
    region = lead.get("region", "").strip().lower()

    # Helper function to check if any keyword matches in text
    def matches_any(text: str, keywords: list) -> bool:
        """Check if any keyword from the list is found in the text (case-insensitive)"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    # Iterate through all ICP definitions
    for icp in ICP_DEFINITIONS:
        # Step 1: Check if sub_industry matches
        if not matches_any(sub_industry, icp["sub_industries"]):
            continue  # No match, try next ICP

        # Step 2: Check region if specified in ICP definition
        # If "regions" is defined, lead must be from one of those regions
        if "regions" in icp:
            if not matches_any(region, icp["regions"]):
                continue  # Region doesn't match, try next ICP

        # Step 3: Check if role contains role_details (specific titles)
        # Role details are the most specific check (e.g., "CEO", "CTO", "VP of Operations")
        if matches_any(role, icp["role_details"]):
            # Return custom multiplier if defined, otherwise default 1.5x
            return icp.get("multiplier", 1.5)

    # No ICP match found
    return 1.0


def _matches_icp_definitions(lead: dict) -> bool:
    """
    Check if a lead matches any ICP definition (without returning multiplier value).

    Returns:
        True if lead matches any ICP definition
        False otherwise
    """
    return _get_icp_bonus(lead) > 0


def _get_icp_bonus(lead: dict) -> int:
    """
    Get the ICP bonus points for a lead.

    Returns:
        - 0 if no ICP match
        - 50 (default) if ICP match but no custom "bonus" field
        - Custom "bonus" value if specified in matching ICP definition

    Some ICPs have higher bonuses for rare, high-value profiles:
        - Blockchain/Crypto Investors: +100
        - UAE/Dubai Investors: +100
        - Cyber Security/IT Management (Midwest US): +100
    """
    # Null-safe extraction - handles None values gracefully
    sub_industry = (lead.get("sub_industry") or "").strip().lower()
    role = (lead.get("role") or "").strip().lower()

    # SECURITY FIX: Use VALIDATED location fields, not miner-submitted 'region'
    # 'country', 'state', and 'city' are verified against LinkedIn data in Stage 4
    # 'region' is unvalidated and was being gamed by miners
    country = (lead.get("country") or "").strip().lower()
    state = (lead.get("state") or "").strip().lower()
    city = (lead.get("city") or "").strip().lower()

    def matches_any(text: str, keywords: list) -> bool:
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def matches_regional_filter(icp_regions: list) -> bool:
        """
        Check if lead's VALIDATED location matches ICP regional filter.

        Uses country (primary), state (secondary), and city (tertiary) - all validated in Stage 4.
        Does NOT use miner-submitted 'region' field which is unvalidated.
        """
        # Check country first (e.g., "united arab emirates", "united states")
        if matches_any(country, icp_regions):
            return True

        # Check state for US-specific ICPs (e.g., "california", "new york")
        # Only applies when country is US and ICP has US state names
        if country in ["united states", "usa", "us", "america"]:
            if matches_any(state, icp_regions):
                return True

        # Check city for city-specific ICPs (e.g., "dubai" for UAE)
        # This is needed since UAE leads are only accepted from Dubai
        if matches_any(city, icp_regions):
                return True

        return False

    # Track highest bonus (in case lead matches multiple ICPs)
    highest_bonus = 0

    # Get employee count for ICPs that filter by company size
    employee_count = (lead.get("employee_count") or "").strip()
    lead_emp_range = parse_employee_count(employee_count) if employee_count else None

    for icp in ICP_DEFINITIONS:
        # Check sub_industry: "*" means all industries (skip check)
        icp_sub_industries = icp.get("sub_industries", [])
        if icp_sub_industries != ["*"]:
            if not matches_any(sub_industry, icp_sub_industries):
                continue

        # Check regions if specified
        if "regions" in icp:
            # SECURITY: Use validated country/state, NOT unvalidated region
            if not matches_regional_filter(icp["regions"]):
                continue

        # Check employee_ranges if specified (e.g., ["201-500", "501-1000"])
        if "employee_ranges" in icp:
            if not lead_emp_range:
                continue  # No employee count in lead, skip this ICP

            lead_min, lead_max = lead_emp_range
            range_matched = False

            for emp_range in icp["employee_ranges"]:
                icp_range = parse_employee_count(emp_range)
                if icp_range:
                    icp_min, icp_max = icp_range
                    # Check if lead's range overlaps with ICP's range
                    # Lead range [lead_min, lead_max] overlaps [icp_min, icp_max]
                    # if lead_min <= icp_max AND lead_max >= icp_min
                    if lead_min <= icp_max and lead_max >= icp_min:
                        range_matched = True
                        break

            if not range_matched:
                continue

        # Check role match
        if matches_any(role, icp["role_details"]):
            # Get bonus: use custom "bonus" field, or default to 50
            icp_bonus = icp.get("bonus", 50)
            highest_bonus = max(highest_bonus, icp_bonus)

    return highest_bonus


def is_enterprise_company(lead: dict) -> bool:
    """
    Check if a lead is from an enterprise company (10,001+ employees).

    Enterprise companies get a rep score multiplier that caps their final score:
    - ICP match: target = 10, multiplier = min(0, 10 - raw_rep_score)
    - No ICP match: target = 5, multiplier = min(0, 5 - raw_rep_score)
    - Final rep = raw_rep_score + multiplier

    This means:
    - If raw rep score is <= target, no change (multiplier = 0)
    - If raw rep score > target, it gets capped at target

    Returns:
        True if employee_count indicates 10,001+ employees
        False otherwise
    """
    employee_count = lead.get("employee_count", "")
    if not employee_count:
        return False

    # Parse the employee count
    parsed = parse_employee_count(str(employee_count))
    if not parsed:
        return False

    emp_min, emp_max = parsed

    # Enterprise = 10,001+ employees (minimum 10001)
    return emp_min >= 10001


def calculate_icp_adjustment(lead: dict) -> int:
    """
    Calculate ICP adjustment points (NEW SYSTEM - replaces multiplier).

    This function calculates an absolute point adjustment based on:
    1. ICP Definition Match: +50 points (default), or custom bonus for high-value ICPs:
       - Africa Broadcasting/Media: +100 points
       - Blockchain/Crypto/Web3 Investors: +100 points
    2. Small Company in Major Hub Bonus: +50 points
       - ≤10 employees AND in major hub (NYC, SF, LA, Austin, Chicago, etc.)
    3. Small Company Bonus:
       - ≤50 employees: +20 points
    4. Large Company Penalty:
       - >1,000 employees: -10 points
       - >5,000 or >10,000 employees: -15 points

    DYNAMIC CAP: Cap is +50 for normal leads, or ICP bonus for high-value ICPs
    PENALTIES STACK: Penalties are applied AFTER capping the bonus

    Args:
        lead: Lead dictionary with employee_count, city, and ICP-relevant fields

    Returns:
        Integer adjustment (bonus capped, then penalties applied)

    Examples:
        - ICP match only = +50
        - High-value ICP (Africa/Crypto) = +100
        - ICP + ≤50 employees = +70 → capped to +50
        - High-value ICP + >1k employees = +100 - 10 = +90
        - Small hub (≤10 + NYC) = +50
        - Non-ICP + ≤50 employees = +20
        - Non-ICP + >5k employees = -15
    """
    bonus = 0
    penalty = 0
    breakdown = {"icp_match": 0, "major_hub_bonus": 0, "employee_bonus": 0, "employee_penalty": 0}

    # ========================================================================
    # MAJOR HUBS BY COUNTRY (city + country must BOTH match)
    # ========================================================================
    # Uses CANONICAL city names from geo_lookup_fast.json (post-gateway normalization)
    # Gateway normalizes: "NYC" -> "New York City", "SF" -> "San Francisco", etc.
    # So we only need the canonical names here - no aliases needed!
    #
    # Country names MUST match gateway/api/submit.py VALID_COUNTRIES (lowercase)
    MAJOR_HUBS_BY_COUNTRY = {
        # ----------------------------------------------------------------
        # NORTH AMERICA (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "united states": {
            # NYC area (manhattan/brooklyn are separate cities in JSON)
            "new york city", "manhattan", "brooklyn",
            # West Coast
            "san francisco", "los angeles", "san diego", "san jose", "seattle", "portland",
            # Texas
            "austin", "dallas", "houston",
            # Other major hubs
            "chicago", "boston", "denver", "miami", "washington", "atlanta", "phoenix",
        },
        "canada": {
            "toronto", "vancouver", "montréal",  # Note: "montréal" is canonical (not "montreal")
        },
        # ----------------------------------------------------------------
        # EUROPE (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "united kingdom": {
            "london", "manchester", "edinburgh", "cambridge", "oxford",
        },
        "germany": {
            "berlin", "münchen", "frankfurt am main", "hamburg",  # "münchen" is canonical
        },
        "france": {
            "paris",
        },
        "netherlands": {
            "amsterdam", "rotterdam",
        },
        "switzerland": {
            "zürich", "genève",  # Canonical names with accents
        },
        "ireland": {
            "dublin",
        },
        "sweden": {
            "stockholm",
        },
        "spain": {
            "barcelona", "madrid",
        },
        # ----------------------------------------------------------------
        # ASIA-PACIFIC (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "hong kong": {
            "hong kong",
        },
        "singapore": {
            "singapore",
        },
        "japan": {
            "tokyo", "osaka",
        },
        "south korea": {
            "seoul",
        },
        "china": {
            "shanghai", "beijing", "shenzhen",
        },
        "india": {
            "bengaluru", "mumbai", "new delhi", "hyderabad", "pune",  # "bengaluru" is canonical
        },
        "australia": {
            "sydney", "melbourne",
        },
        "new zealand": {
            "auckland",
        },
        # ----------------------------------------------------------------
        # MIDDLE EAST (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "israel": {
            "tel aviv",
        },
        "united arab emirates": {
            "dubai", "abu dhabi",
        },
        # ----------------------------------------------------------------
        # SOUTH AMERICA (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "brazil": {
            "são paulo",  # Canonical name with accent
        },
    }

    # Get city and country for major hub check
    city = lead.get("city", "").strip().lower()
    country = lead.get("country", "").strip().lower()

    # Check if BOTH country AND city match a major hub
    # Simple exact matching - gateway already normalized cities to canonical form
    is_major_hub = False
    matched_hub = None

    if country in MAJOR_HUBS_BY_COUNTRY:
        hub_cities = MAJOR_HUBS_BY_COUNTRY[country]  # This is now a set
        if city in hub_cities:
            is_major_hub = True
            matched_hub = f"{city} ({country})"

    # ========================================================================
    # STEP 1: ICP Definition Match (+50 default, or custom bonus)
    # High-value ICPs (Africa Broadcasting, Blockchain/Crypto Investors) get +100
    # ========================================================================
    icp_bonus = _get_icp_bonus(lead)
    if icp_bonus > 0:
        bonus += icp_bonus
        breakdown["icp_match"] = icp_bonus
        if icp_bonus > 50:
            print(f"   🎯 HIGH-VALUE ICP MATCH: +{icp_bonus} points")
        else:
            print(f"   🎯 ICP MATCH: +{icp_bonus} points")

    # ========================================================================
    # STEP 2: Employee Count Bonuses and Penalties
    # ========================================================================
    employee_count_str = get_employee_count(lead) or ""

    if employee_count_str:
        parsed = parse_employee_count(employee_count_str)

        if parsed:
            emp_min, emp_max = parsed

            # Small company in major hub bonus (+50 points)
            if emp_max <= 10 and is_major_hub:
                bonus += 50
                breakdown["major_hub_bonus"] = 50
                print(f"   🌆 SMALL COMPANY IN MAJOR HUB (≤10 + {matched_hub}): +50 points")
            # Small company bonus (+20 points for ≤50 employees)
            elif emp_max <= 50:
                bonus += 20
                breakdown["employee_bonus"] = 20
                print(f"   🏢 SMALL COMPANY (≤50): +20 points")

            # Large company penalty (stacks with capped bonus)
            # Note: Uses emp_min to determine the MINIMUM company size
            # Note: 10,001+ (enterprise) companies already have hardcoded rep scores, no ICP penalty
            if 5000 < emp_min < 10001:
                # 5,001-10,000 employees only (enterprise 10,001+ handled separately)
                penalty = 15
                breakdown["employee_penalty"] = -15
                print(f"   🏭 LARGE COMPANY (5k-10k): -15 points")
            elif 1000 < emp_min <= 5000:
                # 1,001-5,000 employees (NOT 10,001+ which is enterprise)
                penalty = 10
                breakdown["employee_penalty"] = -10
                print(f"   🏭 LARGE COMPANY (1k-5k): -10 points")
            # Note: 10,001+ employees (enterprise) get NO ICP penalty - they have hardcoded rep scores
    else:
        print(f"   📋 No employee count available - no size adjustment")

    # ========================================================================
    # STEP 3: Cap bonus, then apply penalties
    # Cap is dynamic: +50 for normal ICPs, or ICP bonus value for high-value ICPs
    # ========================================================================
    # Determine cap: use ICP bonus if > 50 (high-value ICP), otherwise 50
    bonus_cap = max(50, icp_bonus) if icp_bonus > 0 else 50

    if bonus > bonus_cap:
        print(f"   ⚠️  Bonus {bonus} exceeds cap, capping at +{bonus_cap}")
        bonus = bonus_cap

    # Penalties stack with capped bonus
    adjustment = bonus - penalty

    print(f"   📊 FINAL ICP ADJUSTMENT: {adjustment:+d} points")
    print(f"      Bonus (capped at {bonus_cap}): {min(bonus, bonus_cap):+d} = ICP:{breakdown['icp_match']:+d} + Hub:{breakdown['major_hub_bonus']:+d} + Size:{breakdown['employee_bonus']:+d}")
    print(f"      Penalty: {-penalty:+d}")

    return adjustment
