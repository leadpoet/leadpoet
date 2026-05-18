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
1. Generate 25 ICPs — one per industry across 25 distinct industries
2. Use LLM to create realistic, varied prompts
3. Compute ICP set hash
4. Store in database
5. Log to transparency_log
6. Activate the new set

COMPANY-MODE ONLY (May 2026+):
The qualification model competition is single-path company-mode. Models
return a CompanyOutput keyed off industry / sub-industry / size / geography
/ stage / intent_signals. ICP prompts no longer carry target_roles or
target_seniority — anything role-shaped is legacy from the contact-mode
era and is intentionally absent from generated sets.
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

# Industry distribution for 25 ICPs — exactly one ICP per industry.
# CRITICAL: Every entry MUST exist in gateway/utils/industry_taxonomy.py
# (source of truth). Model queries use .lower() for case-insensitive
# matching with the leads database.
#
# The list is intentionally wide and B2B-friendly: it spans tech, regulated
# verticals, physical/industrial, consumer, and capital markets so a single
# day's benchmark exercises model generalization rather than rewarding
# overfitting to two or three industries.
INDUSTRY_DISTRIBUTION = {
    "Software": 1,
    "Information Technology": 1,
    "Artificial Intelligence": 1,
    "Hardware": 1,
    "Data and Analytics": 1,
    "Privacy and Security": 1,
    "Health Care": 1,
    "Biotechnology": 1,
    "Financial Services": 1,
    "Lending and Investments": 1,
    "Payments": 1,
    "Manufacturing": 1,
    "Commerce and Shopping": 1,
    "Professional Services": 1,
    "Advertising": 1,
    "Sales and Marketing": 1,
    "Real Estate": 1,
    "Energy": 1,
    "Education": 1,
    "Transportation": 1,
    "Media and Entertainment": 1,
    "Food and Beverage": 1,
    "Sustainability": 1,
    "Gaming": 1,
    "Travel and Tourism": 1,
}
# Sanity check at import time so a regression in the list is loud, not silent.
assert len(INDUSTRY_DISTRIBUTION) == 25 and sum(INDUSTRY_DISTRIBUTION.values()) == 25, (
    "INDUSTRY_DISTRIBUTION must be exactly 25 distinct industries with 1 ICP each"
)

# Sub-industries per industry
# CRITICAL: These MUST match values in gateway/utils/industry_taxonomy.py
# The taxonomy has 723 sub-industries - we select relevant ones per industry.
# Each of the 25 industries in INDUSTRY_DISTRIBUTION must appear here.
SUB_INDUSTRIES = {
    "Software": [
        "SaaS", "Enterprise Software", "Developer Tools", "Developer Platform",
        "Developer APIs", "Cloud Computing", "Machine Learning",
        "CRM", "Marketing Automation", "Productivity Tools", "Collaboration",
    ],
    "Information Technology": [
        "IT Infrastructure", "IT Management", "Cloud Computing", "Cloud Management",
        "Cyber Security", "Network Security", "Information Services",
        "Technical Support", "Business Information Systems",
    ],
    "Artificial Intelligence": [
        "Machine Learning", "Natural Language Processing", "Predictive Analytics",
        "Artificial Intelligence",
    ],
    "Hardware": [
        "Semiconductor", "Electronics", "Consumer Electronics", "Robotics",
        "Industrial Engineering", "3D Printing", "IoT", "Wearables",
    ],
    "Data and Analytics": [
        "Business Intelligence", "Analytics", "Big Data", "Data Integration",
        "Data Mining", "Data Visualization", "Predictive Analytics",
        "Consumer Research", "Market Research",
    ],
    "Privacy and Security": [
        "Cyber Security", "Network Security", "Identity Management",
        "Fraud Detection", "Cloud Security", "DevSecOps",
    ],
    "Health Care": [
        "Health Diagnostics", "Medical Device", "Electronic Health Record (EHR)",
        "mHealth", "Wellness", "Pharmaceutical", "Clinical Trials", "Hospital",
        "Nursing and Residential Care", "Home Health Care", "Therapeutics",
    ],
    "Biotechnology": [
        "Biopharma", "Bioinformatics", "Life Science", "Genetics", "Neuroscience",
        "Clinical Trials", "Pharmaceutical", "Biometrics",
    ],
    "Financial Services": [
        "Banking", "Insurance", "Asset Management", "Wealth Management",
        "FinTech", "InsurTech", "Credit", "Commercial Lending", "Consumer Lending",
        "Trading Platform", "Hedge Funds",
    ],
    "Lending and Investments": [
        "Venture Capital", "Private Equity", "Hedge Funds", "Asset Management",
        "Commercial Lending", "Consumer Lending", "Credit",
    ],
    "Payments": [
        "Payments", "Mobile Payments", "Billing", "Subscription Service",
        "Point of Sale", "FinTech",
    ],
    "Manufacturing": [
        "Industrial Manufacturing", "Industrial Engineering", "Machinery Manufacturing",
        "Aerospace", "Automotive", "Electronics", "Semiconductor", "3D Printing",
        "Plastics and Rubber Manufacturing", "Paper Manufacturing", "Textiles",
    ],
    "Commerce and Shopping": [
        "E-Commerce", "E-Commerce Platforms", "Retail", "Retail Technology",
        "Marketplace", "Wholesale", "Point of Sale", "Personalization",
        "Price Comparison", "Social Shopping", "Local Shopping",
    ],
    "Professional Services": [
        "Consulting", "Management Consulting", "Legal", "Legal Tech", "Accounting",
        "Recruiting", "Staffing Agency", "Compliance", "Risk Management",
        "Business Development", "Quality Assurance",
    ],
    "Advertising": [
        "Advertising", "Ad Network", "Ad Exchange", "Ad Retargeting",
        "Ad Server", "Affiliate Marketing",
    ],
    "Sales and Marketing": [
        "Marketing Automation", "CRM", "Lead Generation", "Email Marketing",
        "SEO", "Content Marketing", "Sales Automation",
    ],
    "Real Estate": [
        "PropTech", "Commercial Real Estate", "Residential", "Real Estate Investment",
        "Property Management", "Construction",
    ],
    "Energy": [
        "Renewable Energy", "Solar", "Wind Energy", "Energy Storage",
        "Energy Management", "Oil and Gas", "Electric Utilities",
    ],
    "Education": [
        "EdTech", "E-Learning", "Higher Education", "K-12 Education",
        "Corporate Training", "Tutoring",
    ],
    "Transportation": [
        "Logistics", "Last Mile Transportation", "Freight Service", "Fleet Management",
        "Public Transportation", "Ride Sharing", "Shipping",
    ],
    "Media and Entertainment": [
        "Video Streaming", "Music Streaming", "Podcasting", "Film",
        "Digital Media", "Broadcasting",
    ],
    "Food and Beverage": [
        "Restaurants", "Food Delivery", "Food Processing", "Beverages",
        "Coffee", "Brewing",
    ],
    "Sustainability": [
        "CleanTech", "Recycling", "Sustainability", "Waste Management",
        "Environmental Engineering", "Green Building",
    ],
    "Gaming": [
        "Video Games", "Mobile Games", "Console Games", "PC Games",
        "Game Development", "Esports",
    ],
    "Travel and Tourism": [
        "Travel", "Travel Accommodations", "Hospitality", "Tour Operator",
        "Travel Agency", "Vacation Rental",
    ],
}

# Company sizes
COMPANY_SIZES = [
    "10-50", "50-200", "200-500", "500-1000", "1000-5000", "5000+"
]

# Company stages
COMPANY_STAGES = [
    "Seed", "Series A", "Series B", "Series C+", "Private Equity", "Public"
]

# Geographies - ALL 51 US states/territories + Dubai & Abu Dhabi (UAE)
# US states from gateway/utils/geo_lookup_fast.json (source of truth)
GEOGRAPHIES = [
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
    # === UAE (investor-focused ICPs only) ===
    "United Arab Emirates, Dubai",
    "United Arab Emirates, Abu Dhabi",
]

# Products/Services by industry (what the miner's model should help sell)
# One entry per industry in INDUSTRY_DISTRIBUTION.
PRODUCTS_BY_INDUSTRY = {
    "Software": [
        "CRM software", "DevOps platform", "Cloud security solution",
        "Data analytics tool", "AI/ML platform", "Marketing automation",
        "HR management system", "Project management software",
        "Customer success platform", "API management tool",
    ],
    "Information Technology": [
        "Cloud migration services", "IT infrastructure", "Managed services",
        "Security solutions", "Network optimization", "System integration",
        "IT support platform", "DevOps automation",
    ],
    "Artificial Intelligence": [
        "ML training platform", "LLM observability suite", "Computer vision API",
        "AI agent orchestration", "Vector database",
    ],
    "Hardware": [
        "Industrial IoT platform", "Edge compute appliance", "Hardware testing software",
        "PCB design tooling", "Robotics SDK",
    ],
    "Data and Analytics": [
        "Business intelligence platform", "Data visualization tool",
        "Analytics dashboard", "Data pipeline tool", "Predictive analytics",
        "Customer analytics platform", "Data quality software",
    ],
    "Privacy and Security": [
        "SIEM platform", "Endpoint detection and response", "Identity governance suite",
        "Cloud security posture management", "Fraud detection system",
    ],
    "Health Care": [
        "Electronic health records", "Telemedicine platform", "Patient engagement app",
        "Clinical decision support", "Medical imaging AI", "Compliance software",
        "Revenue cycle management", "Care coordination platform",
    ],
    "Biotechnology": [
        "Clinical trial management system", "Lab information system",
        "Drug discovery platform", "Genomics analysis tool",
        "Regulatory submission software", "Research data management",
    ],
    "Financial Services": [
        "Risk management platform", "Regulatory compliance tool",
        "Trading platform", "Fraud detection system", "Payment processing",
        "Wealth management software", "Credit scoring AI", "KYC solution",
    ],
    "Lending and Investments": [
        "Portfolio management platform", "Underwriting automation",
        "Investor reporting software", "Deal flow CRM", "Loan origination system",
    ],
    "Payments": [
        "Payments orchestration platform", "Subscription billing", "Embedded payments API",
        "PoS terminals", "Cross-border payments rails",
    ],
    "Manufacturing": [
        "ERP system", "Supply chain management", "Quality management software",
        "Industrial IoT platform", "Predictive maintenance", "Inventory optimization",
        "Production planning tool", "Supplier management system",
    ],
    "Commerce and Shopping": [
        "E-commerce platform", "Inventory management", "Customer data platform",
        "Personalization engine", "Shipping optimization", "POS system",
        "Loyalty program software", "Returns management",
    ],
    "Professional Services": [
        "Practice management software", "Time tracking tool", "CRM for services",
        "Knowledge management", "Resource planning", "Proposal automation",
        "Client portal", "Billing and invoicing",
    ],
    "Advertising": [
        "Programmatic ad platform", "Attribution analytics", "Creative testing suite",
        "Influencer marketing platform", "Ad fraud detection",
    ],
    "Sales and Marketing": [
        "Outbound sales platform", "Marketing automation", "Lead enrichment API",
        "Conversation intelligence software", "ABM platform",
    ],
    "Real Estate": [
        "Property management software", "Real estate CRM", "Construction management platform",
        "Tenant experience app", "PropTech marketplace",
    ],
    "Energy": [
        "Grid management software", "Energy storage controls", "Carbon accounting platform",
        "Renewables monitoring suite", "Smart meter analytics",
    ],
    "Education": [
        "LMS platform", "Student information system", "Online course platform",
        "Tutoring marketplace", "Workforce learning software",
    ],
    "Transportation": [
        "Fleet management software", "Last-mile logistics platform",
        "Freight visibility platform", "Route optimization software", "Telematics suite",
    ],
    "Media and Entertainment": [
        "OTT streaming platform", "Audience analytics", "Content monetization tools",
        "Live production software", "Rights management platform",
    ],
    "Food and Beverage": [
        "Restaurant POS suite", "Food supply chain platform", "Online ordering platform",
        "Inventory management for restaurants", "Beverage distribution software",
    ],
    "Sustainability": [
        "ESG reporting platform", "Waste tracking software", "Carbon accounting platform",
        "Energy efficiency analytics", "Sustainable sourcing platform",
    ],
    "Gaming": [
        "Game analytics platform", "Live-ops tooling", "User acquisition platform",
        "Anti-cheat service", "Game engine middleware",
    ],
    "Travel and Tourism": [
        "Booking management platform", "Revenue management software",
        "Travel CRM", "Property management system for hotels", "Trip planning software",
    ],
}

# Intent signals / additional context
INTENT_SIGNALS = [
    "Recently raised funding and expanding team",
    "Hiring for senior engineering or sales roles",
    "Company mentioned in industry news for growth",
    "Executive spoke at industry conference",
    "Company blog discusses digital transformation",
    "Evaluating new vendors or platforms",
    "Announced new market expansion",
    "Recent leadership change",
    "Launched or announced a new product line",
    "Posted on LinkedIn about upcoming initiatives"
]


# =============================================================================
# OpenRouter LLM-Based ICP Generation
# =============================================================================
# This generates VARIED, HUMAN-LIKE ICP prompts using an LLM
# to prevent miners from overfitting to template patterns

async def generate_icps_with_openrouter(
    set_id: int,
    total_icps: int = 25
) -> tuple:
    """
    Generate ICP prompts using OpenRouter LLM (o3-mini).

    This creates varied, human-like prompts that read as if typed by
    real sales/marketing professionals looking for companies to sell into.

    COMPANY-MODE ONLY: Each ICP describes a company profile (industry,
    size, geography, stage, intent signals, product context). Prompts
    MUST NOT specify job titles, seniority, or any contact-level role.
    The model competition no longer scores contact-level data — anything
    role-shaped is legacy and intentionally absent.

    Args:
        set_id: Set identifier (YYYYMMDD format) for ICP naming
        total_icps: Number of ICPs to generate (default 25, one per industry)

    Returns:
        Tuple of (icps_list, industry_distribution, icp_set_hash)
        or None if LLM generation fails (falls back to template-based)
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set, falling back to template-based generation")
        return None

    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY

    all_industries = list(INDUSTRY_DISTRIBUTION.keys())

    # Build comprehensive industry->sub_industry mapping from taxonomy
    taxonomy_sub_industries = {}
    for sub_ind, data in INDUSTRY_TAXONOMY.items():
        for ind in data.get("industries", []):
            if ind not in taxonomy_sub_industries:
                taxonomy_sub_industries[ind] = []
            taxonomy_sub_industries[ind].append(sub_ind)

    # Build the COMPREHENSIVE prompt for the LLM
    # This prompt must be EXTREMELY detailed to get human-like, varied outputs.
    system_prompt = """You are generating search queries that real B2B salespeople would type into a tool that surfaces TARGET COMPANIES (accounts) — not individual contacts.

CRITICAL: These must sound like REAL HUMANS typing, NOT templates. Imagine a salesperson quickly typing the kind of company they want to reach.

ABSOLUTE RULES:
1. The prompt describes a COMPANY profile only. NEVER mention job titles, seniority levels, decision-maker names, or any contact-level role (no "VP of X", "CTO", "Director", "Head of Y", "C-Suite", "buyers", etc.). Talk about the COMPANY: industry, size, stage, location, what they make/sell, what signals they're showing.
2. NEVER start with "Who should we target" or "Who should we find for you" — these are robotic.
3. NEVER use the same sentence pattern more than 2-3 times across the set.
4. Each prompt should feel like a different person typed it.

MANDATORY VOICE MIX (spread evenly across the set):

CATEGORY A — Direct first-person requests:
- "I need a list of Series B fintech companies in NYC that are scaling card issuing this year."
- "I'm looking for 50-200 person AI startups on the west coast that recently raised a round."

CATEGORY B — Casual / conversational:
- "yo can you pull mid-market e-commerce companies in texas that just launched a new SKU"
- "hey gonna need biotech startups in boston working on gene therapy, ideally post-series A"

CATEGORY C — Shorthand / telegraphic:
- "series B saas companies SF 50-200 hiring fast"
- "fintech startups NYC just raised, payments focus"

CATEGORY D — Question format:
- "what are the fastest-growing manufacturing companies in ohio doing digital transformation?"
- "anyone tracking renewable energy startups expanding into europe?"

CATEGORY E — Descriptive / detailed:
- "Looking for cloud security companies, 200-500 employees, US-based, ideally ones that have posted publicly about evaluating new vendors in the last 90 days."
- "Searching for digital health companies that closed Series A in the last 6 months and are actively running clinical pilots."

VARIATION TECHNIQUES:
- Some prompts are 1 line, others 2-3 sentences
- Some mention specific products/categories, others stay broader
- Some use industry shorthand: "saas", "fintech", "biotech"
- Some are very specific, others are vague
- Mix formal ("I am seeking") with casual ("yo need")

THINGS THAT MAKE PROMPTS FAKE OR WRONG (AVOID):
- "Who should we target" / "Who should we find for you" — TOO ROBOTIC
- "Find decision makers in X sector" — BANNED (talks about people, not companies)
- "Ideal buyer: X at Y companies" — BANNED (talks about people)
- ANY job title, seniority, or contact-level descriptor — BANNED
- Starting every prompt the same way

Your output must be valid JSON."""

    # Build the detailed user prompt with distribution requirements
    user_prompt = f"""Generate exactly {total_icps} ICP prompts — ONE PER INDUSTRY across these industries:

INDUSTRY DISTRIBUTION (must match exactly, exactly one ICP per industry):
"""
    for industry, count in INDUSTRY_DISTRIBUTION.items():
        sub_inds = taxonomy_sub_industries.get(industry, SUB_INDUSTRIES.get(industry, ["General"]))[:15]
        products = PRODUCTS_BY_INDUSTRY.get(industry, ["Software solution"])[:5]
        user_prompt += f"""
{industry}: {count} prompt
  Valid sub-industries to use: {', '.join(sub_inds)}
  Example products the buyer might be selling INTO this industry: {', '.join(products)}
"""

    user_prompt += f"""

CRITICAL REMINDER - BANNED CONTENT (DO NOT INCLUDE):
- ANY job title, seniority, role, or contact-level descriptor (no "VP of X", "CTO", "Director", "Head of Y", "decision-makers", "buyers", "C-suite", "executives", "leaders", etc.).
- "Who should we target" — BANNED
- "Who should we find for you" — BANNED
- "Find decision makers in" — BANNED (refers to people)
- "Ideal buyer:" — BANNED
- Any robotic template language — BANNED

REQUIRED STARTER DISTRIBUTION (enforce strictly across the {total_icps} prompts):
- ~5 starting with "I need..." / "I'm looking for..." / "I want..."
- ~5 casual: "yo", "hey", "gonna need", "can you pull"
- ~5 shorthand/telegraphic
- ~5 questions: "what are...", "anyone know...", "which ..."
- ~5 starting with "Looking for..." / "Searching for..."

EXAMPLE COMPANY-LEVEL PROMPTS THAT SOUND HUMAN:

For Software:
- "I need Series A-B SaaS companies on the west coast that are visibly scaling their devops footprint right now."
- "yo can you pull devtools companies 50-200 people, recently raised? we sell developer infra"

For Financial Services:
- "Looking for fintech companies between Series A and Series C in the US that just closed a round in payments or lending."
- "what mid-market banks are publicly modernizing their core systems in 2026?"

For Healthcare:
- "digital health companies that started clinical pilots in the last 90 days. 50-500 employees, US."
- "I want hospital systems or health systems in california that are actively evaluating new EHR tooling."

For Manufacturing:
- "industrial manufacturers in the midwest doing a digital transformation push, ideally 1000+ employees."
- "anyone tracking automotive suppliers in ohio/michigan that are publicly announcing factory expansions?"

GEOGRAPHIES TO USE:
US States (most prompts): Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, DC, Florida, Georgia, Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri, Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming

UAE (optional, AT MOST 1 prompt — and only if the industry naturally fits, e.g. Financial Services or Lending and Investments): Dubai or Abu Dhabi. If used, write it as a company-level investor or capital-markets ICP (e.g. "looking for sovereign wealth funds or family offices in Abu Dhabi deploying into US growth equity"). Do NOT name people.

IMPORTANT: Do NOT include any international geographies besides Dubai/Abu Dhabi. The remaining prompts MUST use US states.

COMPANY SIZES: 10-50, 50-200, 200-500, 500-1000, 1000-5000, 5000+
COMPANY STAGES: Seed, Series A, Series B, Series C+, Private Equity, Public

PRODUCTS/SERVICES (what the SELLER is offering — mention naturally where useful):
- Software: CRM, DevOps, Cloud security, AI/ML platforms
- IT: Cloud services, Managed IT, Cybersecurity
- Healthcare: EHR systems, Telemedicine, Patient engagement
- Biotech: Lab software, Clinical trial management
- Financial: Risk management, Compliance, Trading platforms
- Manufacturing: ERP, Supply chain, Industrial IoT
- Commerce: E-commerce platforms, Inventory management
- Professional Services: Practice management, Billing software
- Data: BI tools, Analytics platforms
(See the per-industry product list above for more.)

INTENT SIGNALS TO WEAVE IN (COMPANY-level — events the COMPANY did, not anything about people):
- Recently raised funding / just closed a round
- Expanding to new markets or new geographies
- Launched or announced a new product/feature
- Public commentary about digital transformation / evaluating new vendors
- Recent factory / facility / store opening
- New regulatory or compliance pressure
- Acquired another company / acquired by another company

OUTPUT FORMAT — Return a JSON object with "icps" array containing exactly {total_icps} objects, one per industry. Do NOT include target_roles, target_seniority, role_types, or any role/seniority field; the schema is company-only.
{{
  "icps": [
    {{
      "icp_id": "icp_{set_id}_001",
      "prompt": "I need Series B SaaS startups on the west coast that are actively scaling their devops footprint.",
      "industry": "Software",
      "sub_industry": "SaaS",
      "employee_count": "50-200",
      "company_stage": "Series B",
      "geography": "United States, California",
      "country": "United States",
      "product_service": "DevOps platform",
      "intent_signals": ["Recently raised funding"]
    }},
    ...
  ]
}}

FINAL CHECK — Before outputting, verify:
1. NO prompt contains a job title, seniority level, "decision-makers", "buyers", "executives", or any contact-level descriptor.
2. NO prompt starts with "Who should we target" or "Who should we find".
3. There are exactly {total_icps} prompts, one per industry listed above.
4. Each prompt sounds like a different human typed it.
5. Prompts use US states (and at most 1 UAE entry under Financial Services / Lending and Investments).
6. No icp object has a target_roles, target_seniority, or role_types field."""

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
                    "max_tokens": 16000,  # Generous headroom for 25 ICPs
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

            intent_signals = icp.get("intent_signals", [])
            if isinstance(intent_signals, str):
                intent_signals = [intent_signals]
            if not intent_signals:
                intent_signals = random.sample(INTENT_SIGNALS, random.randint(1, 2))
                logger.warning(f"ICP {icp_id} had empty intent_signals from LLM, assigned fallback: {intent_signals}")

            geography = icp.get("geography", "United States, California")

            # Allow US and UAE only; override anything else
            if geography and "United Arab Emirates" in geography:
                country = "United Arab Emirates"
            elif geography and "United States" in geography:
                country = "United States"
            else:
                logger.warning(f"ICP {icp_id} has non-US/UAE geography '{geography}', overriding to California")
                geography = "United States, California"
                country = "United States"

            # COMPANY-MODE ONLY: do NOT carry forward target_roles or
            # target_seniority — they are legacy contact-mode fields. We
            # populate them as empty defaults to satisfy any older miner
            # code that still reads them via dict.get(), but no real role
            # data flows through.
            validated_icp = {
                "icp_id": icp_id,
                "prompt": prompt,
                "industry": industry_normalized,
                "sub_industry": icp.get("sub_industry", SUB_INDUSTRIES.get(industry_normalized, ["General"])[0]),
                "target_roles": [],
                "target_seniority": "",
                "employee_count": icp.get("employee_count", "50-200"),
                "company_stage": icp.get("company_stage", "Series A"),
                "geography": geography,
                "country": country,
                "product_service": icp.get("product_service", "Software solution"),
                "intent_signals": intent_signals,
                "buyer_description": prompt  # Legacy alias of prompt
            }

            validated_icps.append(validated_icp)

        # Set is exactly 25; allow some slack but anything significantly
        # short of expected count is a bad generation and we fall back.
        min_acceptable = max(int(total_icps * 0.9), total_icps - 2)
        if len(validated_icps) < min_acceptable:
            logger.error(f"Only {len(validated_icps)} valid ICPs (expected ~{total_icps}), falling back to template")
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
    Generate a single COMPANY-LEVEL ICP with a natural language prompt.

    The prompt is what real B2B salespeople would type to describe the
    KIND OF COMPANY (account) they want to reach — no contact-level
    descriptors. Models must interpret this prompt to surface matching
    companies and verify at least one intent signal.

    Args:
        icp_id: Unique identifier for this ICP
        industry: Industry category
        seed: Optional random seed for reproducibility

    Returns:
        Dict containing the ICP definition with a natural language prompt.
        ``target_roles`` and ``target_seniority`` are kept as empty defaults
        for backward compatibility with older miner code that may dict.get
        them, but they are not populated with real data.
    """
    if seed is not None:
        random.seed(seed)

    sub_industries = SUB_INDUSTRIES.get(industry, ["General"])
    products = PRODUCTS_BY_INDUSTRY.get(industry, ["Software solution"])

    sub_industry = random.choice(sub_industries)
    employee_count_range = random.choice(COMPANY_SIZES)
    company_stage = random.choice(COMPANY_STAGES)
    geography = random.choice(GEOGRAPHIES)
    product = random.choice(products)
    intent_signals = random.sample(INTENT_SIGNALS, random.randint(1, 2))

    # Stage phrasing
    if company_stage in ["Seed", "Series A", "Series B"]:
        stage_text = f"early-stage ({company_stage})"
    elif company_stage in ["Series C+", "Private Equity"]:
        stage_text = f"growth-stage ({company_stage})"
    else:  # Public
        stage_text = "enterprise/public"

    country = geography.split(",")[0].strip()
    size_text = f"with {employee_count_range} employees"
    signals_text = " or ".join([s.lower() for s in intent_signals])

    # COMPANY-only prompt templates — never mention people, titles, or
    # seniority. The miner returns a company, not a contact.
    prompt_templates = [
        f"I need {stage_text} {sub_industry} companies {size_text} in {country}. "
        f"Showing signals: {signals_text}. We sell {product}.",

        f"Looking for {sub_industry} ({industry}) companies "
        f"at {stage_text}, {size_text}, based in {country}. "
        f"Intent indicators: {signals_text}.",

        f"Searching for {sub_industry} companies "
        f"({company_stage}, {employee_count_range} employees) in {country} "
        f"that are {signals_text}. Selling {product}.",

        f"{company_stage} {sub_industry} companies in {country} "
        f"({employee_count_range} employees). Signals: {signals_text}.",

        f"What {stage_text} {sub_industry} companies in {country} "
        f"are {signals_text}? We're selling {product}.",
    ]

    prompt = random.choice(prompt_templates)

    return {
        "icp_id": icp_id,
        "prompt": prompt,
        "industry": industry,
        "sub_industry": sub_industry,
        # Legacy fields — kept as empty defaults so older miner code that
        # dict.gets them doesn't crash. The competition is company-only.
        "target_roles": [],
        "target_seniority": "",
        "employee_count": employee_count_range,
        "company_stage": company_stage,
        "geography": geography,
        "country": country,
        "product_service": product,
        "intent_signals": intent_signals,
        "buyer_description": prompt,  # Legacy alias of prompt
    }


def generate_icp_set(
    set_id: int,
    total_icps: int = 25,
    base_seed: Optional[int] = None
) -> tuple:
    """
    Generate a complete ICP set — one ICP per industry across the 25 distinct
    industries in ``INDUSTRY_DISTRIBUTION``.

    Args:
        set_id: Set identifier (YYYYMMDD format)
        total_icps: Number of ICPs to generate (default 25, one per industry).
            Currently informational — the actual count comes from
            ``INDUSTRY_DISTRIBUTION`` which is the source of truth for the
            industry list.
        base_seed: Base seed for reproducibility

    Returns:
        Tuple of (icps_list, industry_distribution, icp_set_hash)
    """
    icps = []
    icp_counter = 1
    actual_distribution: Dict[str, int] = {}

    for industry, count in INDUSTRY_DISTRIBUTION.items():
        actual_distribution[industry] = count

        for _ in range(count):
            icp_id = f"icp_{set_id}_{icp_counter:03d}"

            seed = None
            if base_seed is not None:
                seed = base_seed + icp_counter

            icp = generate_single_icp(icp_id, industry, seed)
            icps.append(icp)
            icp_counter += 1

    if base_seed is not None:
        random.seed(base_seed)
    random.shuffle(icps)

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

    Uses OpenRouter LLM (o3-mini) for varied, human-like company-level prompts.
    NO FALLBACK - if OpenRouter fails, returns None and the system will
    automatically retry on the next gateway restart or rotation check.

    Generates exactly ``len(INDUSTRY_DISTRIBUTION)`` ICPs (currently 25, one
    per industry).

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
    
    target_count = len(INDUSTRY_DISTRIBUTION)
    logger.info(f"Generating ICPs with OpenRouter LLM (target {target_count} ICPs, one per industry)...")
    try:
        result = await generate_icps_with_openrouter(set_id, total_icps=target_count)
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
    
    Polls every 60s and checks whether today's ICP set (by YYYYMMDD set_id)
    is active. If not, generates and activates a new one. This is robust
    against restarts, missed midnight windows, and transient failures.
    """
    logger.info("Starting ICP rotation task (polling every 60s)")
    
    # In-memory guard: skip DB queries once today's set is confirmed active
    last_generated_date: Optional[str] = None
    
    while True:
        try:
            now = datetime.now(timezone.utc)
            current_date = now.strftime("%Y-%m-%d")
            today_set_id = get_set_id_for_date(now)
            
            # Fast path: already confirmed today's set is active
            if last_generated_date == current_date:
                if now.minute == 0:
                    next_reset = get_next_reset_time()
                    hours_until = (next_reset - now).total_seconds() / 3600
                    logger.info(f"ICP rotation: Set {today_set_id} active. Next reset at {next_reset} ({hours_until:.1f}h)")
                await asyncio.sleep(60)
                continue
            
            # Check DB: is today's set already active?
            active_set = await get_active_icp_set()
            if active_set and active_set.get('set_id') == today_set_id:
                logger.info(f"ICP rotation: Today's set {today_set_id} already active")
                last_generated_date = current_date
                await asyncio.sleep(60)
                continue
            
            # Today's set is missing or a stale set is active — generate
            stale_id = active_set.get('set_id') if active_set else None
            logger.info(f"ICP rotation: Need set {today_set_id} (current active: {stale_id}), generating...")
            
            set_id = await generate_and_activate_icp_set()
            
            if set_id:
                logger.info(f"ICP rotation: Successfully activated set {set_id}")
                last_generated_date = current_date
            else:
                logger.error("ICP rotation: Failed to generate/activate set, will retry in 5 minutes")
                await asyncio.sleep(300)
                continue
            
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.info("ICP rotation task cancelled")
            break
        except Exception as e:
            logger.error(f"ICP rotation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
