"""
ICP Set Generation Task

Generates new ICP (Ideal Customer Profile) sets for benchmark evaluation.
Runs daily at 12:00 AM UTC (midnight UTC).

CRITICAL DESIGN:
1. ICPs are generated RANDOMLY but held CONSTANT until next reset
2. ICPs are stored PRIVATELY in qualification_private_icp_sets
3. Miners NEVER see the ICPs until evaluation time
4. ICP hash is logged to transparency_log for verifiability

GENERATION PROCESS:
1. Generate 100 ICPs distributed across industries
2. Use LLM to create realistic, varied prompts
3. Compute ICP set hash
4. Store in database
5. Log to transparency_log
6. Activate the new set
"""

import os
import json
import hashlib
import random
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

import pytz

logger = logging.getLogger(__name__)

# =============================================================================
# OpenRouter Configuration for LLM-Based ICP Generation
# =============================================================================
# We use OpenRouter with o3-mini to generate varied, human-like ICP prompts
# This prevents miners from overfitting to hardcoded templates

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "openai/o3-mini"  # High context window, good reasoning
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# =============================================================================
# Configuration
# =============================================================================

# Industry distribution for 100 ICPs
# CRITICAL: These are from gateway/utils/industry_taxonomy.py (source of truth)
# Model queries use .lower() for case-insensitive matching with the leads database
INDUSTRY_DISTRIBUTION = {
    "Software": 15,                    # From taxonomy
    "Information Technology": 15,       # From taxonomy  
    "Health Care": 10,                  # From taxonomy
    "Biotechnology": 10,                # From taxonomy
    "Financial Services": 15,           # From taxonomy
    "Manufacturing": 10,                # From taxonomy
    "Commerce and Shopping": 10,        # From taxonomy (lowercase 'and')
    "Professional Services": 10,        # From taxonomy
    "Data and Analytics": 5,            # From taxonomy (lowercase 'and')
}

# Sub-industries per industry
# CRITICAL: These MUST match values in gateway/utils/industry_taxonomy.py
# The taxonomy has 723 sub-industries - we select relevant ones per industry
SUB_INDUSTRIES = {
    "Software": [
        # From taxonomy: companies that develop software products
        "SaaS", "Enterprise Software", "Developer Tools", "Developer Platform",
        "Developer APIs", "Cloud Computing", "Machine Learning", "Artificial Intelligence",
        "CRM", "Marketing Automation", "Productivity Tools", "Collaboration"
    ],
    "Information Technology": [
        # From taxonomy: IT services and infrastructure
        "IT Infrastructure", "IT Management", "Cloud Computing", "Cloud Management",
        "Cyber Security", "Network Security", "Information Services", "Technical Support",
        "CivicTech", "GovTech", "Business Information Systems"
    ],
    "Health Care": [
        # From taxonomy: healthcare services and technology
        "Health Diagnostics", "Medical Device", "Electronic Health Record (EHR)",
        "mHealth", "Wellness", "Pharmaceutical", "Clinical Trials", "Hospital",
        "Nursing and Residential Care", "Home Health Care", "Therapeutics"
    ],
    "Biotechnology": [
        # From taxonomy: biotech and life sciences
        "Biopharma", "Bioinformatics", "Life Science", "Genetics", "Neuroscience",
        "Clinical Trials", "Pharmaceutical", "Biometrics"
    ],
    "Financial Services": [
        # From taxonomy: financial services companies
        "Banking", "Insurance", "Asset Management", "Payments", "Wealth Management",
        "FinTech", "InsurTech", "Credit", "Commercial Lending", "Consumer Lending",
        "Trading Platform", "Venture Capital", "Hedge Funds"
    ],
    "Manufacturing": [
        # From taxonomy: manufacturing companies
        "Industrial Manufacturing", "Industrial Engineering", "Machinery Manufacturing",
        "Aerospace", "Automotive", "Electronics", "Semiconductor", "3D Printing",
        "Plastics and Rubber Manufacturing", "Paper Manufacturing", "Textiles"
    ],
    "Commerce and Shopping": [
        # From taxonomy: commerce and retail
        "E-Commerce", "E-Commerce Platforms", "Retail", "Retail Technology",
        "Marketplace", "Wholesale", "Point of Sale", "Personalization",
        "Price Comparison", "Social Shopping", "Local Shopping"
    ],
    "Professional Services": [
        # From taxonomy: professional services
        "Consulting", "Management Consulting", "Legal", "Legal Tech", "Accounting",
        "Recruiting", "Staffing Agency", "Compliance", "Risk Management",
        "Business Development", "Quality Assurance"
    ],
    "Data and Analytics": [
        # From taxonomy: data and analytics
        "Business Intelligence", "Analytics", "Big Data", "Data Integration",
        "Data Mining", "Data Visualization", "Predictive Analytics",
        "Consumer Research", "Market Research"
    ]
}

# Target roles by industry
TARGET_ROLES = {
    "Software": [
        "VP of Engineering", "CTO", "VP of Product", "Head of DevOps",
        "VP of Sales", "Chief Revenue Officer", "VP of Marketing",
        "Engineering Director", "Product Director", "Head of Data"
    ],
    "Information Technology": [
        "CTO", "VP of Engineering", "IT Director", "Head of Infrastructure",
        "VP Technology", "Chief Digital Officer", "Head of Cloud",
        "VP IT Operations", "Director of Engineering"
    ],
    "Health Care": [
        "Chief Medical Officer", "VP Clinical Operations", "CTO",
        "VP of R&D", "VP Regulatory Affairs", "VP Medical Affairs",
        "Head of Patient Services", "Chief Nursing Officer"
    ],
    "Biotechnology": [
        "Chief Scientific Officer", "VP of R&D", "Chief Medical Officer",
        "Head of Clinical Trials", "VP Regulatory Affairs",
        "Director of Research", "VP Drug Discovery"
    ],
    "Financial Services": [
        "Chief Risk Officer", "VP Compliance", "CTO", "CFO",
        "Head of Trading", "VP Operations", "Chief Data Officer",
        "Head of Digital Transformation", "VP Strategy"
    ],
    "Manufacturing": [
        "VP of Operations", "Plant Manager", "Supply Chain Director",
        "VP of Engineering", "Quality Director", "COO",
        "Head of Production", "VP Procurement"
    ],
    "Commerce and Shopping": [
        "VP of E-commerce", "Chief Digital Officer", "VP Marketing",
        "Head of Merchandising", "VP Supply Chain", "Chief Customer Officer",
        "VP Store Operations", "Head of Growth"
    ],
    "Professional Services": [
        "Managing Partner", "COO", "VP of Client Services",
        "Head of Business Development", "Chief Strategy Officer",
        "VP Operations", "Practice Lead", "VP Talent"
    ],
    "Data and Analytics": [
        "Chief Data Officer", "VP of Data Science", "Head of Analytics",
        "VP Data Engineering", "Director of BI", "Head of Data Platform"
    ]
}

# Company sizes
COMPANY_SIZES = [
    "10-50", "50-200", "200-500", "500-1000", "1000-5000", "5000+"
]

# Company stages
COMPANY_STAGES = [
    "Seed", "Series A", "Series B", "Series C+", "Private Equity", "Public"
]

# Geographies - ALL 51 US states/territories + 5 international metros
# US states from gateway/utils/geo_lookup_fast.json (source of truth)
# International metros: Major global business hubs
GEOGRAPHIES = [
    # === ALL 51 US STATES/TERRITORIES ===
    "United States, Alabama",
    "United States, Alaska",
    "United States, Arizona",
    "United States, Arkansas",
    "United States, California",
    "United States, Colorado",
    "United States, Connecticut",
    "United States, Delaware",
    "United States, District of Columbia",
    "United States, Florida",
    "United States, Georgia",
    "United States, Hawaii",
    "United States, Idaho",
    "United States, Illinois",
    "United States, Indiana",
    "United States, Iowa",
    "United States, Kansas",
    "United States, Kentucky",
    "United States, Louisiana",
    "United States, Maine",
    "United States, Maryland",
    "United States, Massachusetts",
    "United States, Michigan",
    "United States, Minnesota",
    "United States, Mississippi",
    "United States, Missouri",
    "United States, Montana",
    "United States, Nebraska",
    "United States, Nevada",
    "United States, New Hampshire",
    "United States, New Jersey",
    "United States, New Mexico",
    "United States, New York",
    "United States, North Carolina",
    "United States, North Dakota",
    "United States, Ohio",
    "United States, Oklahoma",
    "United States, Oregon",
    "United States, Pennsylvania",
    "United States, Rhode Island",
    "United States, South Carolina",
    "United States, South Dakota",
    "United States, Tennessee",
    "United States, Texas",
    "United States, Utah",
    "United States, Vermont",
    "United States, Virginia",
    "United States, Washington",
    "United States, West Virginia",
    "United States, Wisconsin",
    "United States, Wyoming",
    # === 5 INTERNATIONAL METROPOLITAN AREAS ===
    "United Kingdom, London",
    "Hong Kong",
    "Singapore",
    "Germany, Berlin",
    "Canada, Toronto",
]

# Products/Services by industry (what the miner's model should help sell)
PRODUCTS_BY_INDUSTRY = {
    "Software": [
        "CRM software", "DevOps platform", "Cloud security solution",
        "Data analytics tool", "AI/ML platform", "Marketing automation",
        "HR management system", "Project management software",
        "Customer success platform", "API management tool"
    ],
    "Information Technology": [
        "Cloud migration services", "IT infrastructure", "Managed services",
        "Security solutions", "Network optimization", "System integration",
        "IT support platform", "DevOps automation"
    ],
    "Health Care": [
        "Electronic health records", "Telemedicine platform", "Patient engagement app",
        "Clinical decision support", "Medical imaging AI", "Compliance software",
        "Revenue cycle management", "Care coordination platform"
    ],
    "Biotechnology": [
        "Clinical trial management system", "Lab information system",
        "Drug discovery platform", "Genomics analysis tool",
        "Regulatory submission software", "Research data management"
    ],
    "Financial Services": [
        "Risk management platform", "Regulatory compliance tool",
        "Trading platform", "Fraud detection system", "Payment processing",
        "Wealth management software", "Credit scoring AI", "KYC solution"
    ],
    "Manufacturing": [
        "ERP system", "Supply chain management", "Quality management software",
        "Industrial IoT platform", "Predictive maintenance", "Inventory optimization",
        "Production planning tool", "Supplier management system"
    ],
    "Commerce and Shopping": [
        "E-commerce platform", "Inventory management", "Customer data platform",
        "Personalization engine", "Shipping optimization", "POS system",
        "Loyalty program software", "Returns management"
    ],
    "Professional Services": [
        "Practice management software", "Time tracking tool", "CRM for services",
        "Knowledge management", "Resource planning", "Proposal automation",
        "Client portal", "Billing and invoicing"
    ],
    "Data and Analytics": [
        "Business intelligence platform", "Data visualization tool",
        "Analytics dashboard", "Data pipeline tool", "Predictive analytics",
        "Customer analytics platform", "Data quality software"
    ]
}

# Intent signals / additional context
INTENT_SIGNALS = [
    "Recently raised funding and expanding team",
    "Posted job openings for relevant roles",
    "Company mentioned in industry news for growth",
    "Executive spoke at industry conference",
    "Company blog discusses digital transformation",
    "Hiring for specific technology skills",
    "Announced new market expansion",
    "Recent leadership change",
    "Published case study on operational challenges",
    "Active on LinkedIn discussing industry trends"
]


# =============================================================================
# OpenRouter LLM-Based ICP Generation
# =============================================================================
# This generates VARIED, HUMAN-LIKE ICP prompts using an LLM
# to prevent miners from overfitting to template patterns

async def generate_icps_with_openrouter(
    set_id: int,
    total_icps: int = 100
) -> tuple:
    """
    Generate 100 ICP prompts using OpenRouter LLM (o3-mini).
    
    This creates varied, human-like prompts that read as if typed by
    real sales/marketing professionals looking for leads.
    
    Args:
        set_id: Set identifier (YYYYMMDD format) for ICP naming
        total_icps: Number of ICPs to generate (default 100)
    
    Returns:
        Tuple of (icps_list, industry_distribution, icp_set_hash)
        or None if LLM generation fails (falls back to template-based)
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set, falling back to template-based generation")
        return None
    
    # Load industry taxonomy to ensure we use valid sub-industries
    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY
    
    # Get all valid industries and sub-industries
    all_industries = list(INDUSTRY_DISTRIBUTION.keys())
    
    # Build comprehensive industry->sub_industry mapping from taxonomy
    taxonomy_sub_industries = {}
    for sub_ind, data in INDUSTRY_TAXONOMY.items():
        for ind in data.get("industries", []):
            if ind not in taxonomy_sub_industries:
                taxonomy_sub_industries[ind] = []
            taxonomy_sub_industries[ind].append(sub_ind)
    
    # Build the COMPREHENSIVE prompt for the LLM
    # This prompt must be EXTREMELY detailed to get human-like, varied outputs
    system_prompt = """You are generating search queries that real B2B salespeople would type into a lead-finding tool.

CRITICAL: These must sound like REAL HUMANS typing, NOT templates. Imagine a salesperson quickly typing what they need.

ABSOLUTE RULES:
1. NEVER start with "Who should we target" or "Who should we find for you" - these are robotic
2. NEVER use the same sentence pattern more than 5 times across all 100 prompts
3. Each prompt should feel like a different person typed it

MANDATORY FIRST-PERSON VOICE (distribute evenly, ~20 each category):

CATEGORY A - Direct requests (first person):
- "I need VP of Sales contacts at SaaS companies, Series B, hiring SDRs right now"
- "I'm looking for CFOs at biotech startups that just closed funding rounds"
- "I want to reach out to CIOs at healthcare systems in Texas"

CATEGORY B - Casual/conversational:
- "yo can you pull some CTOs at fintech startups in NYC"
- "hey need some sales leaders at ecommerce companies, west coast"
- "gonna need a list of marketing VPs at B2B software companies"

CATEGORY C - Shorthand/telegraphic (no full sentences):
- "VP eng series B saas california hiring"
- "CFO biotech boston recently funded"
- "CTO devtools startups 50-200 employees"

CATEGORY D - Question format (genuine questions):
- "who handles procurement at manufacturing companies in Ohio?"
- "what's the best title to target for selling HR software?"
- "anyone know who buys cybersecurity at mid-market companies?"

CATEGORY E - Descriptive/specific requests:
- "Looking for VP of Engineering at cloud computing companies with 200-500 employees in the US, ideally they've posted about hiring challenges on LinkedIn"
- "Searching for Chief Medical Officers at digital health startups that raised Series A in the last 6 months"

REAL HUMAN EXAMPLES TO EMULATE:
- "I need to find heads of IT at hospitals in California. They should be actively evaluating new EHR systems."
- "can you get me a list of VPs of Sales at SaaS companies? series B or later, 100-500 employees, US based"
- "looking for fintech CFOs, specifically at payments companies, ideally ones that just raised"
- "VP Supply Chain at manufacturing - midwest region, companies doing digital transformation"
- "CISO or VP Security contacts at companies 1000+ employees. financial services preferred"
- "startup CTOs in the AI/ML space, seed to series A, SF bay area"
- "i need procurement heads at large retailers, the ones who buy supply chain software"
- "sales ops leaders at b2b saas - director level or above, companies using salesforce"

THINGS THAT MAKE PROMPTS FEEL FAKE (AVOID):
- "Who should we target for our X?" - TOO ROBOTIC
- "Who should we find for you?" - TOO ROBOTIC  
- "Find decision makers in X sector" - TOO TEMPLATED
- "Ideal buyer: X at Y companies" - TOO STRUCTURED
- Starting every prompt the same way
- Always using the exact same structure

VARIATION TECHNIQUES:
- Some prompts are 1 line, others are 2-3 sentences
- Some mention specific products, others don't
- Some have typos: "eng" "cos" "ppl" "b2b" "saas"
- Some are very specific, others are vague
- Mix formal ("I am seeking") with casual ("yo need")

Your output must be valid JSON."""

    # Build the detailed user prompt with distribution requirements
    user_prompt = f"""Generate exactly {total_icps} ICP prompts distributed across these industries:

INDUSTRY DISTRIBUTION (must match exactly):
"""
    for industry, count in INDUSTRY_DISTRIBUTION.items():
        sub_inds = taxonomy_sub_industries.get(industry, SUB_INDUSTRIES.get(industry, ["General"]))[:15]
        user_prompt += f"""
{industry}: {count} prompts
  Valid sub-industries to use: {', '.join(sub_inds)}
  Example roles: {', '.join(TARGET_ROLES.get(industry, ['Manager'])[:5])}
"""

    user_prompt += f"""

CRITICAL REMINDER - BANNED PHRASES (DO NOT USE):
- "Who should we target" - BANNED
- "Who should we find for you" - BANNED
- "Find decision makers in" - BANNED (use at most 3 times)
- "Ideal buyer:" - BANNED
- Any robotic template language - BANNED

REQUIRED STARTER DISTRIBUTION (enforce strictly):
- 20 prompts starting with "I need..." or "I'm looking for..." or "I want..."
- 20 prompts starting casually: "yo", "hey", "gonna need", "can you get me"
- 20 prompts in shorthand/telegraphic style (no full sentences): "VP eng saas series B california"
- 20 prompts as questions: "who handles...", "what's the best...", "anyone know..."
- 20 prompts starting with "Looking for..." or "Searching for..." (but varied structure after)

EXAMPLE PROMPTS THAT SOUND HUMAN:

For Software:
- "I need VP of Engineering contacts at SaaS startups, Series A-B, west coast. Bonus if they're hiring."
- "yo can you pull CTOs at devtools companies? 50-200 employees, recently raised"
- "VP eng series B saas SF hiring engineers"

For Financial Services:
- "I'm looking for VP of Compliance and VP of Strategy at Credit Companies. Series B, 200-500 employees in the US. Ideally they've raised funding recently."
- "CFOs at fintech startups that just closed rounds - NYC or boston"
- "who handles risk management purchasing at banks? VP level?"

For Healthcare:
- "I need to find Chief Medical Officers at digital health companies. They should be actively looking at new platforms."
- "healthcare IT buyers - CIO or VP level, hospital systems, evaluating EHR"
- "CMO contacts telemedicine startups series A"

For Manufacturing:
- "heads of supply chain at manufacturing plants in ohio and michigan. companies doing digital transformation"
- "I want VP of Operations contacts at industrial companies, 1000+ employees, midwest"
- "plant managers automotive suppliers hiring"

GEOGRAPHIES TO USE:
US States: Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, DC, Florida, Georgia, Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri, Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming

International (5-10 prompts only): London UK, Hong Kong, Singapore, Berlin Germany, Toronto Canada

COMPANY SIZES: 10-50, 50-200, 200-500, 500-1000, 1000-5000, 5000+
COMPANY STAGES: Seed, Series A, Series B, Series C+, Private Equity, Public

PRODUCTS/SERVICES TO MENTION NATURALLY:
- Software: CRM, DevOps, Cloud security, AI/ML platforms, HR software
- IT: Cloud services, Managed IT, Cybersecurity
- Healthcare: EHR systems, Telemedicine, Patient engagement
- Biotech: Lab software, Clinical trial management
- Financial: Risk management, Compliance, Trading platforms
- Manufacturing: ERP, Supply chain, Industrial IoT
- Commerce: E-commerce platforms, Inventory management
- Professional Services: Practice management, Billing software
- Data: BI tools, Analytics platforms

INTENT SIGNALS TO WEAVE IN:
- Recently raised funding / just closed a round
- Hiring for specific roles
- Spoke at conference / published content
- Recent leadership change
- Expanding to new markets
- Digital transformation / evaluating new solutions
- Posted on LinkedIn about challenges

OUTPUT FORMAT - Return a JSON object with "icps" array containing exactly 100 objects:
{{
  "icps": [
    {{
      "icp_id": "icp_{set_id}_001",
      "prompt": "I need VP of Engineering at SaaS startups, Series B, west coast. They should be hiring engineers.",
      "industry": "Software",
      "sub_industry": "SaaS",
      "target_roles": ["VP of Engineering"],
      "employee_count": "50-200",
      "company_stage": "Series B",
      "geography": "United States, California",
      "country": "United States",
      "product_service": "DevOps platform",
      "intent_signals": ["Hiring for specific roles"]
    }},
    ...
  ]
}}

FINAL CHECK - Before outputting, verify:
1. NO prompt starts with "Who should we target" or "Who should we find"
2. NO prompt uses "Find decision makers in X sector" more than 3 times
3. Prompts use 5 different starting styles distributed evenly
4. Each prompt sounds like a different human typed it"""

    try:
        logger.info(f"Calling OpenRouter {OPENROUTER_MODEL} to generate {total_icps} ICPs...")
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                    "X-Title": "LeadPoet ICP Generator"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.9,  # Higher temperature for more variety
                    "max_tokens": 32000,  # Need space for 100 ICPs
                    "response_format": {"type": "json_object"}
                }
            )
        
        if response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not content:
            logger.error("OpenRouter returned empty content")
            return None
        
        # Parse the JSON response
        try:
            # Handle if the model wrapped in a key
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Check for common wrapper keys
                icps = parsed.get("icps") or parsed.get("icp_prompts") or parsed.get("prompts") or parsed.get("data")
                if icps is None and len(parsed) == 1:
                    icps = list(parsed.values())[0]
                if icps is None:
                    icps = list(parsed.values())[0] if parsed else []
            else:
                icps = parsed
            
            if not isinstance(icps, list):
                logger.error(f"Expected list of ICPs, got {type(icps)}")
                return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenRouter JSON response: {e}")
            logger.error(f"Content preview: {content[:500]}...")
            return None
        
        logger.info(f"OpenRouter returned {len(icps)} ICPs")
        
        # Validate and normalize each ICP
        validated_icps = []
        actual_distribution = {ind: 0 for ind in INDUSTRY_DISTRIBUTION.keys()}
        
        for i, icp in enumerate(icps):
            if not isinstance(icp, dict):
                logger.warning(f"ICP {i} is not a dict, skipping")
                continue
            
            # Ensure required fields exist
            icp_id = icp.get("icp_id", f"icp_{set_id}_{i+1:03d}")
            prompt = icp.get("prompt", "")
            industry = icp.get("industry", "")
            
            if not prompt or not industry:
                logger.warning(f"ICP {icp_id} missing prompt or industry, skipping")
                continue
            
            # Normalize industry name (case-insensitive match)
            industry_normalized = None
            for valid_ind in INDUSTRY_DISTRIBUTION.keys():
                if industry.lower() == valid_ind.lower():
                    industry_normalized = valid_ind
                    break
            
            if not industry_normalized:
                logger.warning(f"ICP {icp_id} has invalid industry '{industry}', assigning to Software")
                industry_normalized = "Software"
            
            # Count distribution
            actual_distribution[industry_normalized] += 1
            
            # Build normalized ICP
            target_roles = icp.get("target_roles", [])
            if isinstance(target_roles, str):
                target_roles = [target_roles]
            if not target_roles:
                target_roles = ["Manager"]
            
            intent_signals = icp.get("intent_signals", [])
            if isinstance(intent_signals, str):
                intent_signals = [intent_signals]
            
            geography = icp.get("geography", "United States, California")
            country = icp.get("country", "United States")
            if not country:
                # Extract country from geography
                if "United States" in geography:
                    country = "United States"
                elif "United Kingdom" in geography or "London" in geography:
                    country = "United Kingdom"
                elif "Hong Kong" in geography:
                    country = "Hong Kong"
                elif "Singapore" in geography:
                    country = "Singapore"
                elif "Germany" in geography or "Berlin" in geography:
                    country = "Germany"
                elif "Canada" in geography or "Toronto" in geography:
                    country = "Canada"
                else:
                    country = "United States"
            
            validated_icp = {
                "icp_id": icp_id,
                "prompt": prompt,
                "industry": industry_normalized,
                "sub_industry": icp.get("sub_industry", SUB_INDUSTRIES.get(industry_normalized, ["General"])[0]),
                "target_roles": target_roles,
                "target_seniority": _get_seniority_from_role(target_roles[0]) if target_roles else "Director",
                "employee_count": icp.get("employee_count", icp.get("company_size", "50-200")),
                "company_stage": icp.get("company_stage", "Series A"),
                "geography": geography,
                "country": country,
                "product_service": icp.get("product_service", "Software solution"),
                "intent_signals": intent_signals,
                "buyer_description": prompt  # Legacy field
            }
            
            validated_icps.append(validated_icp)
        
        if len(validated_icps) < 90:
            logger.error(f"Only {len(validated_icps)} valid ICPs (expected ~100), falling back to template")
            return None
        
        # If we got fewer than 100, that's OK - the distribution is approximate
        logger.info(f"Validated {len(validated_icps)} ICPs with distribution: {actual_distribution}")
        
        # Compute hash
        icp_set_hash = compute_icp_set_hash(validated_icps)
        
        return validated_icps, actual_distribution, icp_set_hash
        
    except httpx.TimeoutException:
        logger.error("OpenRouter request timed out (180s)")
        return None
    except Exception as e:
        logger.error(f"OpenRouter ICP generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# Template-Based ICP Generation (Fallback)
# =============================================================================

def generate_single_icp(
    icp_id: str,
    industry: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a single ICP with a NATURAL LANGUAGE PROMPT.
    
    The prompt is what real customers would type to describe who they're looking for.
    Models must INTERPRET this prompt to find matching leads.
    
    Args:
        icp_id: Unique identifier for this ICP
        industry: Industry category
        seed: Optional random seed for reproducibility
    
    Returns:
        Dict containing the ICP definition with a natural language prompt
    """
    if seed is not None:
        random.seed(seed)
    
    sub_industries = SUB_INDUSTRIES.get(industry, ["General"])
    roles = TARGET_ROLES.get(industry, ["Manager"])
    products = PRODUCTS_BY_INDUSTRY.get(industry, ["Software solution"])
    
    # Randomly select parameters
    sub_industry = random.choice(sub_industries)
    target_roles = random.sample(roles, min(random.randint(1, 3), len(roles)))  # 1-3 roles
    company_size = random.choice(COMPANY_SIZES)
    company_stage = random.choice(COMPANY_STAGES)
    geography = random.choice(GEOGRAPHIES)
    product = random.choice(products)
    intent_signals = random.sample(INTENT_SIGNALS, random.randint(1, 2))  # 1-2 signals
    
    # =================================================================
    # BUILD NATURAL LANGUAGE PROMPT
    # =================================================================
    # This is what real customers would type - the model must interpret it!
    # Example: "VP Sales and Heads of Revenue at Series A-C SaaS companies 
    #           in the US. Showing signals: researching outbound tools, 
    #           hiring SDRs, or evaluating competitors."
    # =================================================================
    
    # Format roles naturally
    if len(target_roles) == 1:
        roles_text = target_roles[0]
    elif len(target_roles) == 2:
        roles_text = f"{target_roles[0]} and {target_roles[1]}"
    else:
        roles_text = f"{', '.join(target_roles[:-1])}, and {target_roles[-1]}"
    
    # Format company stage naturally
    if company_stage in ["Seed", "Series A", "Series B"]:
        stage_text = f"early-stage ({company_stage})"
    elif company_stage in ["Series C+", "Private Equity"]:
        stage_text = f"growth-stage ({company_stage})"
    else:  # Public
        stage_text = "enterprise/public"
    
    # Format geography naturally
    country = geography.split(",")[0].strip()
    
    # Format company size naturally
    size_text = f"with {company_size} employees"
    
    # Format intent signals naturally
    signals_text = " or ".join([s.lower() for s in intent_signals])
    
    # Build the prompt - varied templates for diversity
    prompt_templates = [
        # Template 1: Role-focused
        f"{roles_text} at {stage_text} {sub_industry} companies {size_text} in {country}. "
        f"Showing signals: {signals_text}. Looking to sell {product}.",
        
        # Template 2: Industry-focused
        f"Find decision makers in {sub_industry} ({industry} sector). Target: {roles_text} "
        f"at {stage_text} companies {size_text} based in {country}. "
        f"Intent indicators: {signals_text}.",
        
        # Template 3: Product-focused
        f"Who should we target for our {product}? Ideal buyer: {roles_text} "
        f"at {sub_industry} companies ({company_stage}, {company_size} employees) in {country}. "
        f"Look for companies that are {signals_text}.",
        
        # Template 4: Concise
        f"{roles_text} at {company_stage} {sub_industry} companies in {country} "
        f"({company_size} employees). Signals: {signals_text}.",
        
        # Template 5: Question format (like UI screenshot)
        f"Who should we find for you? {roles_text} at {stage_text} {sub_industry} "
        f"companies in {country}. Showing signals: {signals_text}."
    ]
    
    # Pick a random template
    prompt = random.choice(prompt_templates)
    
    return {
        "icp_id": icp_id,
        # PRIMARY FIELD - models should interpret this prompt
        "prompt": prompt,
        # Structured fields for reference/validation (but models should use prompt)
        "industry": industry,
        "sub_industry": sub_industry,
        "target_roles": target_roles,  # Now a list
        "target_seniority": _get_seniority_from_role(target_roles[0]),
        "employee_count": company_size,  # Using company_size variable for employee_count field
        "company_stage": company_stage,
        "geography": geography,
        "country": country,  # Extracted for convenience
        "product_service": product,
        "intent_signals": intent_signals,  # Now a list
        # Legacy field for backward compatibility
        "buyer_description": prompt,
    }


def _get_seniority_from_role(role: str) -> str:
    """Map role to seniority level."""
    role_lower = role.lower()
    if any(x in role_lower for x in ["chief", "cto", "cfo", "coo", "cmo", "cro", "cso"]):
        return "C-Suite"
    elif any(x in role_lower for x in ["vp", "vice president"]):
        return "VP"
    elif "director" in role_lower or "head" in role_lower:
        return "Director"
    elif "manager" in role_lower or "lead" in role_lower:
        return "Manager"
    else:
        return "Director"  # Default


def generate_icp_set(
    set_id: int,
    total_icps: int = 100,
    base_seed: Optional[int] = None
) -> tuple:
    """
    Generate a complete ICP set with industry distribution.
    
    Args:
        set_id: Set identifier (YYYYMMDD format)
        total_icps: Number of ICPs to generate (default 100)
        base_seed: Base seed for reproducibility
    
    Returns:
        Tuple of (icps_list, industry_distribution, icp_set_hash)
    """
    icps = []
    icp_counter = 1
    actual_distribution = {}
    
    # Generate ICPs according to distribution
    for industry, count in INDUSTRY_DISTRIBUTION.items():
        actual_distribution[industry] = count
        
        for i in range(count):
            icp_id = f"icp_{set_id}_{icp_counter:03d}"
            
            # Use deterministic seed if provided
            seed = None
            if base_seed is not None:
                seed = base_seed + icp_counter
            
            icp = generate_single_icp(icp_id, industry, seed)
            icps.append(icp)
            icp_counter += 1
    
    # Shuffle to mix industries (but deterministically if seeded)
    if base_seed is not None:
        random.seed(base_seed)
    random.shuffle(icps)
    
    # Compute hash
    icp_set_hash = compute_icp_set_hash(icps)
    
    logger.info(f"Generated {len(icps)} ICPs for set {set_id}, hash={icp_set_hash[:16]}...")
    
    return icps, actual_distribution, icp_set_hash


def compute_icp_set_hash(icps: List[Dict[str, Any]]) -> str:
    """
    Compute SHA256 hash of ICP set for verifiability.
    
    This hash is logged to transparency_log so external auditors
    can verify the exact ICPs that were used.
    """
    # Sort by icp_id for deterministic ordering
    sorted_icps = sorted(icps, key=lambda x: x.get("icp_id", ""))
    
    # Canonicalize JSON
    canonical_json = json.dumps(sorted_icps, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(canonical_json.encode()).hexdigest()


# =============================================================================
# Database Operations
# =============================================================================

async def store_icp_set(
    set_id: int,
    icps: List[Dict[str, Any]],
    icp_set_hash: str,
    industry_distribution: Dict[str, int],
    active_from: datetime,
    active_until: datetime,
    generation_seed: Optional[str] = None
) -> bool:
    """
    Store a new ICP set in the database.
    
    Args:
        set_id: Set identifier
        icps: List of ICP dictionaries
        icp_set_hash: SHA256 hash of ICPs
        industry_distribution: Count per industry
        active_from: When this set becomes active
        active_until: When this set expires
        generation_seed: Optional seed used for generation
    
    Returns:
        True if stored successfully
    """
    try:
        # MUST use write_client (service_role) because this table has
        # RLS that only allows service_role access
        from gateway.db.client import get_write_client
        
        write_client = get_write_client()
        
        data = {
            "set_id": set_id,
            "icps": icps,
            "icp_set_hash": icp_set_hash,
            "industry_distribution": industry_distribution,
            "active_from": active_from.isoformat(),
            "active_until": active_until.isoformat(),
            "generation_seed": generation_seed,
            "is_active": False  # Not active until explicitly activated
        }
        
        # Upsert (in case regenerating same set_id)
        result = write_client.table("qualification_private_icp_sets") \
            .upsert(data, on_conflict="set_id") \
            .execute()
        
        logger.info(f"Stored ICP set {set_id} with {len(icps)} ICPs")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store ICP set {set_id}: {e}")
        return False


async def activate_icp_set(set_id: int) -> bool:
    """
    Activate an ICP set (deactivates all others).
    
    Args:
        set_id: Set identifier to activate
    
    Returns:
        True if activated successfully
    """
    try:
        # MUST use write_client (service_role) because this table has
        # RLS that only allows service_role access
        from gateway.db.client import get_write_client
        
        write_client = get_write_client()
        
        # Deactivate all sets
        write_client.table("qualification_private_icp_sets") \
            .update({"is_active": False}) \
            .neq("set_id", 0) \
            .execute()
        
        # Activate the target set
        write_client.table("qualification_private_icp_sets") \
            .update({"is_active": True}) \
            .eq("set_id", set_id) \
            .execute()
        
        logger.info(f"Activated ICP set {set_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to activate ICP set {set_id}: {e}")
        return False


async def get_active_icp_set() -> Optional[Dict[str, Any]]:
    """
    Get the currently active ICP set.
    
    Returns:
        Dict with set_id, icps, icp_set_hash or None
    """
    try:
        # MUST use write_client (service_role) because this table has
        # RLS that only allows service_role access
        from gateway.db.client import get_write_client
        
        write_client = get_write_client()
        
        result = write_client.table("qualification_private_icp_sets") \
            .select("set_id, icps, icp_set_hash, active_from, active_until") \
            .eq("is_active", True) \
            .limit(1) \
            .execute()
        
        if result.data:
            return result.data[0]
        
        logger.warning("No active ICP set found")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get active ICP set: {e}")
        return None


# =============================================================================
# Daily Reset Task
# =============================================================================

def get_next_reset_time() -> datetime:
    """
    Get the next ICP reset time (12:00 AM UTC).
    
    Returns:
        datetime: Next reset time in UTC
    """
    now_utc = datetime.now(timezone.utc)
    
    # Next 12:00 AM UTC (midnight UTC)
    next_midnight_utc = datetime(
        now_utc.year, now_utc.month, now_utc.day, 0, 0, 0,
        tzinfo=timezone.utc
    )
    
    # If already past midnight today, go to tomorrow
    if now_utc >= next_midnight_utc:
        next_midnight_utc += timedelta(days=1)
    
    return next_midnight_utc


def get_set_id_for_date(dt: datetime) -> int:
    """
    Get the set_id for a given date (based on UTC).
    
    Args:
        dt: datetime object
    
    Returns:
        int: Set ID in YYYYMMDD format (UTC date)
    """
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.strftime("%Y%m%d"))


async def generate_and_activate_icp_set(
    for_date: Optional[datetime] = None
) -> Optional[int]:
    """
    Generate and activate a new ICP set using OpenRouter LLM.
    
    Uses OpenRouter LLM (o3-mini) for varied, human-like prompts.
    NO FALLBACK - if OpenRouter fails, returns None and the system will
    automatically retry on the next gateway restart or rotation check.
    
    Args:
        for_date: Optional date to generate for (defaults to today UTC)
    
    Returns:
        set_id if successful, None otherwise
    """
    if for_date is None:
        for_date = datetime.now(timezone.utc)
    
    # Compute set_id (based on UTC date)
    set_id = get_set_id_for_date(for_date)
    
    # Compute active window (12 AM UTC to 12 AM UTC next day)
    date_utc = for_date.astimezone(timezone.utc)
    active_from = datetime(
        date_utc.year, date_utc.month, date_utc.day, 0, 0, 0,
        tzinfo=timezone.utc
    )
    active_until = active_from + timedelta(days=1)
    
    # Generate seed from set_id for reproducibility
    # (allows regenerating same set if needed)
    generation_seed = str(set_id)
    
    logger.info(f"Generating ICP set {set_id} for {active_from} to {active_until}")
    
    # =================================================================
    # OPENROUTER LLM - REQUIRED (no fallback)
    # =================================================================
    # This prevents miners from overfitting to template patterns
    # If OpenRouter fails, the system will automatically retry on next check
    
    if not OPENROUTER_API_KEY:
        logger.error("❌ OPENROUTER_API_KEY not set! Cannot generate ICPs.")
        logger.error("   Set OPENROUTER_API_KEY environment variable and restart gateway.")
        return None
    
    logger.info("Generating ICPs with OpenRouter LLM...")
    try:
        result = await generate_icps_with_openrouter(set_id, total_icps=100)
        if not result:
            logger.error("❌ OpenRouter returned None - will retry on next check/restart")
            return None
        
        icps, distribution, icp_hash = result
        logger.info(f"✅ OpenRouter generated {len(icps)} ICPs successfully")
        
    except Exception as e:
        logger.error(f"❌ OpenRouter ICP generation failed: {e}")
        logger.error("   Will retry automatically on next gateway restart or rotation check")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    logger.info(f"Generated {len(icps)} ICPs using OpenRouter LLM, hash={icp_hash[:16]}...")
    
    # Store in database
    stored = await store_icp_set(
        set_id=set_id,
        icps=icps,
        icp_set_hash=icp_hash,
        industry_distribution=distribution,
        active_from=active_from,
        active_until=active_until,
        generation_seed=generation_seed
    )
    
    if not stored:
        return None
    
    # Activate the new set
    activated = await activate_icp_set(set_id)
    
    if not activated:
        return None
    
    # Log to transparency log (ONLY on production, not testnet)
    BITTENSOR_NETWORK = os.environ.get("BITTENSOR_NETWORK", "finney")
    if BITTENSOR_NETWORK == "test":
        logger.info(f"TESTNET: Skipping ICP_SET_ACTIVATED log to protect transparency_log")
    else:
        try:
            from gateway.utils.logger import log_event
            
            await log_event({
                "event_type": "ICP_SET_ACTIVATED",
                "actor_hotkey": "system",
                "nonce": str(uuid4()),
                "ts": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "set_id": set_id,
                    "icp_count": len(icps),
                    "icp_set_hash": icp_hash,
                    "industry_distribution": distribution,
                    "active_from": active_from.isoformat(),
                    "active_until": active_until.isoformat()
                }
            })
            logger.info(f"Logged ICP_SET_ACTIVATED to transparency log")
        except Exception as e:
            logger.warning(f"Failed to log ICP_SET_ACTIVATED: {e}")
    
    return set_id


async def icp_rotation_task():
    """
    Background task that rotates ICPs daily at 12:00 AM UTC (midnight UTC).
    
    Uses polling approach (checks every minute) instead of long sleeps.
    This is more robust against event loop issues and silent failures.
    """
    logger.info("Starting ICP rotation task (polling every 60s)")
    
    # Track the last date we generated for (to avoid duplicate generation)
    last_generated_date: Optional[str] = None
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            current_date = now.strftime("%Y-%m-%d")
            
            # Check if we're within the first 5 minutes after midnight UTC
            # AND we haven't already generated for today
            is_rotation_window = now.hour == 0 and now.minute < 5
            
            if is_rotation_window and last_generated_date != current_date:
                logger.info(f"ICP rotation: Midnight UTC detected ({now}), starting generation...")
                
                # Generate and activate new ICP set
                set_id = await generate_and_activate_icp_set()
                
                if set_id:
                    logger.info(f"ICP rotation: Successfully activated set {set_id}")
                    last_generated_date = current_date
                else:
                    logger.error("ICP rotation: Failed to generate/activate set, will retry next minute")
            
            # Log status every hour (at minute 0)
            if now.minute == 0:
                next_reset = get_next_reset_time()
                hours_until = (next_reset - now).total_seconds() / 3600
                logger.info(f"ICP rotation: Next reset at {next_reset} ({hours_until:.1f} hours)")
            
            # Sleep for 60 seconds then check again
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.info("ICP rotation task cancelled")
            break
        except Exception as e:
            logger.error(f"ICP rotation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Sleep briefly and retry
            await asyncio.sleep(60)


# =============================================================================
# Initialization
# =============================================================================

async def ensure_icp_set_exists():
    """
    Ensure there's an active AND VALID (not expired) ICP set on startup.
    
    If no active set exists OR the active set is expired, generates one for today.
    
    This fixes a bug where the gateway would keep using expired sets
    if it was restarted after midnight but before the rotation task ran.
    """
    active_set = await get_active_icp_set()
    
    if active_set:
        # Check if the set is still valid (not expired)
        active_until_str = active_set.get('active_until')
        if active_until_str:
            try:
                # Parse the active_until timestamp
                if isinstance(active_until_str, str):
                    active_until_str = active_until_str.replace('Z', '+00:00')
                    active_until = datetime.fromisoformat(active_until_str)
                else:
                    active_until = active_until_str
                
                now = datetime.now(timezone.utc)
                
                if now < active_until:
                    logger.info(f"Active ICP set found: {active_set['set_id']} (valid until {active_until})")
                    return active_set['set_id']
                else:
                    logger.warning(
                        f"Active ICP set {active_set['set_id']} EXPIRED at {active_until} "
                        f"(current time: {now}). Generating new set..."
                    )
            except Exception as e:
                logger.error(f"Error parsing active_until: {e}, regenerating set...")
        else:
            logger.warning("Active set has no active_until, regenerating...")
    else:
        logger.info("No active ICP set found, generating one for today...")
    
    return await generate_and_activate_icp_set()
