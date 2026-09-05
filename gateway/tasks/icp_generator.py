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
1. Generate 20 ICPs — one per industry across 20 distinct industries
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

from __future__ import annotations

import os
import json
import re
import time
import hashlib
import random
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

from research_lab.employee_buckets import (
    DEFAULT_EMPLOYEE_BUCKET_RADIUS,
    GENERATED_EMPLOYEE_BUCKETS,
    LINKEDIN_EMPLOYEE_BUCKETS,
    normalize_employee_count_bucket,
    normalize_employee_count_buckets,
)

import pytz

logger = logging.getLogger(__name__)


def _rebenchmark_now() -> datetime:
    """Use the attested parity date only inside a complete parity run."""

    from leadpoet_canonical.production_parity_boundary_v2 import (
        configured_rebenchmark_now_v2,
    )

    return configured_rebenchmark_now_v2(
        now=datetime.now(timezone.utc)
    )

# =============================================================================
# OpenRouter Configuration for LLM-Based ICP Generation
# =============================================================================
# We use OpenRouter with o3-mini to generate varied, human-like ICP prompts
# This prevents miners from overfitting to hardcoded templates

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "perplexity/sonar-pro"  # Real-time web search → realism-grounded ICPs
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# =============================================================================
# Configuration
# =============================================================================

# Industry distribution for 20 ICPs — exactly one ICP per industry.
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
}
# Sanity check at import time so a regression in the list is loud, not silent.
assert len(INDUSTRY_DISTRIBUTION) == 20 and sum(INDUSTRY_DISTRIBUTION.values()) == 20, (
    "INDUSTRY_DISTRIBUTION must be exactly 20 distinct industries with 1 ICP each"
)

# Sub-industries per industry
# CRITICAL: These MUST match values in gateway/utils/industry_taxonomy.py
# The taxonomy has 723 sub-industries - we select relevant ones per industry.
# Each of the 20 industries in INDUSTRY_DISTRIBUTION must appear here.
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
}

# Avoid tiny buckets in generated ICPs unless explicitly introduced later.
COMPANY_SIZES = list(GENERATED_EMPLOYEE_BUCKETS[1:])

# Company stages
COMPANY_STAGES = [
    "Seed", "Series A", "Series B", "Series C+", "Private Equity", "Public"
]

# Stage -> employee-size coherence: the single authority for which exact
# LinkedIn bands each funding/ownership stage can plausibly carry. Stage and
# size were previously drawn independently, which produced impossible ICPs
# (a Seed company with 5,001-10,000 employees) that zero out at the size gate
# no matter how good sourcing is. A stage-pinned ICP now carries this FULL
# band list so the stage never pincers the ICP into one arbitrary band.
STAGE_EMPLOYEE_BUCKETS: Dict[str, tuple] = {
    "Seed": ("2-10", "11-50"),
    "Series A": ("11-50", "51-200"),
    "Series B": ("51-200", "201-500"),
    "Series C+": ("201-500", "501-1,000", "1,001-5,000"),
    "Private Equity": ("201-500", "501-1,000", "1,001-5,000", "5,001-10,000"),
    "Public": ("1,001-5,000", "5,001-10,000", "10,001+"),
}

# Geographies - all US states/territories, deliberate multi-state regions
# (broader supply; see note below), + Dubai & Abu Dhabi (UAE)
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
    # === Multi-region / country-wide values (PREFER these for broader supply) ===
    # State-level ICPs frequently hit empty intersections in practice (e.g. a
    # specific industry + stage + state combination often has 0-2 real companies).
    # Regional and country-wide values let the model find real targets while
    # still respecting the buyer's geographic intent.
    "United States",
    "United States, West Coast",
    "United States, Northeast",
    "United States, Midwest",
    "United States, South",
    "United States, Southwest",
]

# International geographies for the 25% international quota —
# English-speaking business markets only, so sourcing/verification content
# (news, job posts, company sites) stays reliably parseable. Country-first
# format only ("Country" or "Country, Region"): the deployed scorer gate
# reads the country from the first comma segment, so bare region names
# ("Europe") would zero every company until the dynamic geography gate is
# deployed. Broad country-level values keep real-candidate supply high.
INTERNATIONAL_GEOGRAPHIES = [
    "United Kingdom",
    "United Kingdom, London",
    "Ireland",
    "Canada",
    "Canada, Toronto",
    "Australia",
    "Australia, Sydney",
    "New Zealand",
    "Singapore",
    "United Arab Emirates, Dubai",
    "United Arab Emirates, Abu Dhabi",
]

# Share of each generated set that must use international geographies.
INTERNATIONAL_ICP_SHARE = 0.25


def international_icp_target(total_icps: int) -> int:
    """How many ICPs of a set must be international (25%, at least 1)."""
    return max(1, round(total_icps * INTERNATIONAL_ICP_SHARE))


def count_international_icps(icps: List[Dict[str, Any]]) -> int:
    """ICPs whose country is set and is not the United States."""
    count = 0
    for icp in icps:
        if not isinstance(icp, dict):
            continue
        country = str(icp.get("country") or "").strip().lower()
        if country and country not in ("united states", "usa", "us"):
            count += 1
    return count

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
}

# Intent signals — ordered by verifiability (most-evidence-grounded first).
# Generators sample-weighted toward the top: funding, product-launch, expansion,
# leadership-change, and acquisition all produce dated press releases that
# L5/L6 can ground with a verbatim proof_quote. Vague editorial signals
# ("digital transformation commentary", "evaluating vendors") rarely yield
# a single dated event and routinely fail verification — they are kept here
# but the LLM prompt steers away from them.
INTENT_SIGNALS = [
    # ---- Highly verifiable (dated events with press coverage) ----
    "Recently raised funding",
    "Just closed a round",
    "Launched or announced a new product",
    "Expanded to new markets",
    "Acquired another company",
    "Recent leadership change",
    "Achieved regulatory clearance or certification",
    "Announced a strategic partnership",
    # ---- Moderately verifiable (specific but harder to date) ----
    "Hiring for senior engineering or sales roles",
    "Recent factory / facility / store opening",
]

INTENT_SIGNAL_CATEGORY_MAP = {
    "Recently raised funding": "FUNDING",
    "Just closed a round": "FUNDING",
    "Launched or announced a new product": "PRODUCT_LAUNCH",
    "Expanded to new markets": "MARKET_EXPANSION",
    "Acquired another company": "ACQUISITION",
    "Recent leadership change": "LEADERSHIP_CHANGE",
    "Achieved regulatory clearance or certification": "REGULATORY_CLEARANCE",
    "Announced a strategic partnership": "PARTNERSHIP",
    "Hiring for senior engineering or sales roles": "HIRING",
    "Recent factory / facility / store opening": "FACILITY_OPENING",
}

INTENT_CATEGORY_MAX_AGE_DAYS = {
    "HIRING": 90,
    "TECHSTACK": 180,
    "SOCIAL_POSTING": 180,
    "FUNDING": 365,
    "ACQUISITION": 365,
    "PARTNERSHIP": 365,
    "PRODUCT_LAUNCH": 365,
    "LEADERSHIP_CHANGE": 365,
    "MARKET_EXPANSION": 365,
    "REGULATORY_CLEARANCE": 365,
    "FACILITY_OPENING": 365,
    "SALES_GROWTH": 365,
}


def intent_category_for_signal(signal: Any) -> str:
    text = " ".join(str(signal or "").strip().split())
    if text in INTENT_SIGNAL_CATEGORY_MAP:
        return INTENT_SIGNAL_CATEGORY_MAP[text]
    lowered = text.lower()
    # Explicit event semantics must win over broad channel and technology
    # nouns.  For example, "launched a platform capability" is a product
    # launch, not a TECHSTACK signal merely because it contains "platform".
    if any(word in lowered for word in ("hiring", "job", "role", "career", "recruit")):
        return "HIRING"
    if any(word in lowered for word in ("funding", "raised", "round", "financing")):
        return "FUNDING"
    if any(word in lowered for word in ("acquired", "acquisition", "merger", "bought")):
        return "ACQUISITION"
    if any(word in lowered for word in ("partner", "partnership", "integration")):
        return "PARTNERSHIP"
    if any(word in lowered for word in ("executive", "ceo", "cfo", "cto", "appointed", "joined", "leadership")):
        return "LEADERSHIP_CHANGE"
    if any(
        phrase in lowered
        for phrase in (
            "expanded into",
            "expanded to",
            "expanding into",
            "expanding to",
            "market expansion",
            "new market",
            "new geography",
            "new region",
            "new country",
        )
    ):
        return "MARKET_EXPANSION"
    if any(word in lowered for word in ("regulatory", "clearance", "certification", "approved", "compliance")):
        return "REGULATORY_CLEARANCE"
    if any(word in lowered for word in ("factory", "facility", "store", "warehouse", "office opening")):
        return "FACILITY_OPENING"
    if any(
        word in lowered
        for word in (
            "launch",
            "launched",
            "released",
            "new product",
            "new feature",
            "platform capability",
            "platform module",
        )
    ):
        return "PRODUCT_LAUNCH"
    if any(word in lowered for word in ("linkedin", "posted", "social", "tweet", "x.com")):
        return "SOCIAL_POSTING"
    if any(word in lowered for word in ("tech stack", "installed", "uses ", "using ", "software", "tool", "vendor", "platform")):
        return "TECHSTACK"
    return "SALES_GROWTH"


def intent_max_age_days_for_category(category: str) -> int:
    return INTENT_CATEGORY_MAX_AGE_DAYS.get(str(category or "").strip().upper(), 365)


def _normalize_intent_signals(value: Any) -> list[str]:
    if isinstance(value, str):
        signals = [value]
    elif isinstance(value, list):
        signals = [str(item) for item in value]
    else:
        signals = []
    cleaned = [" ".join(signal.strip().split()) for signal in signals if signal.strip()]
    return cleaned[:5]


def _required_attribute_for_icp(icp: Dict[str, Any], *, industry: str, sub_industry: str) -> str:
    existing = " ".join(str(icp.get("required_attribute") or "").strip().split())
    if existing:
        return existing
    product = " ".join(str(icp.get("product_service") or "").strip().split())
    if product:
        return f"The company offers or provides {product}"
    if sub_industry:
        return f"The company operates in {sub_industry}"
    return f"The company operates in {industry}" if industry else "The company matches the target customer profile"


def _configured_employee_bucket_radius() -> int:
    try:
        return max(0, int(os.getenv("RESEARCH_LAB_ICP_EMPLOYEE_BUCKET_RADIUS", str(DEFAULT_EMPLOYEE_BUCKET_RADIUS))))
    except ValueError:
        logger.warning(
            "invalid RESEARCH_LAB_ICP_EMPLOYEE_BUCKET_RADIUS=%r; using default radius %s",
            os.getenv("RESEARCH_LAB_ICP_EMPLOYEE_BUCKET_RADIUS"),
            DEFAULT_EMPLOYEE_BUCKET_RADIUS,
        )
        return DEFAULT_EMPLOYEE_BUCKET_RADIUS


def _configured_employee_all_buckets() -> bool:
    return os.getenv("RESEARCH_LAB_ICP_EMPLOYEE_ALL_BUCKETS", "").strip().lower() in {"1", "true", "yes", "on"}


def _employee_count_display(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "").strip()


def canonicalize_generated_icp(
    icp: Dict[str, Any],
    *,
    industry: str,
    sub_industry: str,
    employee_bucket_radius: int | None = None,
    all_employee_buckets: bool | None = None,
) -> Dict[str, Any]:
    """Apply the shared competition ICP contract before storage."""

    normalized = dict(icp)
    raw_employee_count = normalized.get("employee_count")
    employee_count = normalize_employee_count_bucket(raw_employee_count)
    allowed_employee_counts = normalize_employee_count_buckets(
        normalized.get("employee_count_buckets")
        or normalized.get("employee_counts")
        or raw_employee_count,
        primary_bucket=employee_count,
        radius=_configured_employee_bucket_radius() if employee_bucket_radius is None else employee_bucket_radius,
        all_buckets=_configured_employee_all_buckets() if all_employee_buckets is None else all_employee_buckets,
    )
    prompt = str(normalized.get("prompt") or "")
    if raw_employee_count and isinstance(raw_employee_count, str):
        prompt = prompt.replace(str(raw_employee_count), _employee_count_display(allowed_employee_counts))
    elif raw_employee_count and employee_count:
        prompt = prompt.replace(employee_count, _employee_count_display(allowed_employee_counts))

    intent_signals = _normalize_intent_signals(normalized.get("intent_signals"))
    explicit_required = " ".join(str(normalized.get("intent_signal") or "").strip().split())
    if explicit_required and explicit_required not in intent_signals:
        intent_signals.insert(0, explicit_required)
    if not intent_signals:
        intent_signals = random.sample(INTENT_SIGNALS, random.randint(1, 2))

    intent_signal = intent_signals[0]
    intent_category = str(normalized.get("intent_category") or "").strip().upper()
    if not intent_category:
        intent_category = intent_category_for_signal(intent_signal)
    intent_max_age_days = normalized.get("intent_max_age_days")
    try:
        intent_max_age_days = int(intent_max_age_days)
    except (TypeError, ValueError):
        intent_max_age_days = intent_max_age_days_for_category(intent_category)
    if intent_max_age_days <= 0:
        intent_max_age_days = intent_max_age_days_for_category(intent_category)

    bonus_intents = []
    for signal in intent_signals[1:5]:
        category = intent_category_for_signal(signal)
        bonus_intents.append(
            {
                "intent_signal": signal,
                "intent_category": category,
                "intent_max_age_days": intent_max_age_days_for_category(category),
            }
        )

    normalized.update(
        {
            "prompt": prompt,
            "employee_count": allowed_employee_counts,
            "intent_signals": intent_signals,
            "intent_signal": intent_signal,
            "intent_category": intent_category,
            "intent_max_age_days": intent_max_age_days,
            "bonus_intents": bonus_intents,
            "required_attribute": _required_attribute_for_icp(
                normalized,
                industry=industry,
                sub_industry=sub_industry,
            ),
        }
    )
    normalized.pop("employee_count_buckets", None)
    normalized.pop("employee_counts", None)
    return normalized


# =============================================================================
# OpenRouter LLM-Based ICP Generation
# =============================================================================
# This generates VARIED, HUMAN-LIKE ICP prompts using an LLM
# to prevent miners from overfitting to template patterns

async def generate_icps_with_openrouter(
    set_id: int,
    total_icps: int = 20
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
        total_icps: Number of ICPs to generate (default 20, one per industry)

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

    # Sonar-realism prompt — single-shot generation with web-grounded verification.
    # Every ICP must include `verified_example_company` (the real company Sonar
    # found while verifying supply), proving the combination is non-empty.
    system_prompt = """You are generating B2B sales-targeting ICPs (Ideal Customer Profiles) for a benchmark. You have real-time web access — USE IT.

YOUR JOB
Generate exactly 20 ICPs, one per industry from the distribution list. Each ICP must describe a real, currently-existing target market that a salesperson could actually go prospect.

THE ONE RULE THAT MATTERS MOST — REALISM
Before outputting any ICP, mentally verify: "Can I name at least ONE real, currently-operating company that satisfies ALL the criteria of this ICP — with verifiable recent activity matching the intent signal?"

If you cannot name a specific real company that fits, the ICP is INVALID. Broaden one of the constraints (geography, stage, employee band, sub-industry) until you CAN name at least one real company. Do not output an unrealistic combination under any circumstances.

To enforce this, every ICP MUST include a `verified_example_company` field naming the real company you found while verifying. This is not optional. If you cannot fill this field with a real company, you must rewrite the ICP with broader constraints until you can.

DO NOT generate ICPs where:
- The industry × stage × geography intersection has zero real companies you can name
- The intent signal doesn't fit the industry shape (Consulting firms don't "launch products"; Industrial Manufacturers don't have SaaS-style product launches; Real Estate firms do deals, not product launches)
- The stage doesn't match the intent (Seed companies don't acquire other companies; Series A startups don't make big-name strategic partnerships)
- The geography is so narrow that no real candidates exist
- The product/service is so specific that the buyer's universe collapses (use broad categories, not single named tools)

PROMPT VOICE
Each ICP's `prompt` field should sound like a different real salesperson typed it. Mix tones across the 20 prompts:
- Direct first-person ("I need", "I'm looking for")
- Casual ("yo can you pull", "hey, gonna need")
- Shorthand / telegraphic
- Question format ("what AI companies in...", "anyone tracking...")
- Descriptive / detailed ("Searching for...")

Never use job titles, seniority levels, "decision-makers", "executives", or any contact-level descriptor. Company-only.

CONSTRAINT LISTS (use ONLY these values)

ALLOWED INDUSTRIES (exactly one ICP per industry, in this order):
Software, Information Technology, Artificial Intelligence, Hardware, Data and Analytics, Privacy and Security, Health Care, Biotechnology, Financial Services, Lending and Investments, Payments, Manufacturing, Commerce and Shopping, Professional Services, Advertising, Sales and Marketing, Real Estate, Energy, Education, Transportation

INTENT SIGNAL — WRITE IT LIKE A REAL SALES-INTELLIGENCE BRIEF (not a generic label):
Pick 1-2 intents per ICP. Do NOT output a bare category like "Launched a new product".
Write each intent as a specific, descriptive sentence in the style a real buyer would
brief a sourcing team — name the kind of behavior and the kind of evidence that proves
it. Match the intent to the industry.

The intent MUST be based on ONE of these underlying event types (this keeps it verifiable):
funding round, new product / major capability launch, expansion into a new market,
acquisition, leadership change, regulatory clearance or certification, strategic
partnership, hiring for specific roles, or a new facility / office / store opening.

Fulfillment-style examples (write yours the same way — specific wording, still broad enough
that MANY real companies match):
- "Launched a new product or major platform capability in the last 12 months, per a press
   release, product page, or changelog"
- "Announced a Series A or later funding round in the last 12 months, per a press release,
   Crunchbase, or news coverage"
- "Actively hiring for platform, integration, or revenue-operations roles, per current job
   postings or careers page"
- "Expanded into a new country or region in the last year, per a press release or company
   announcement"

CRITICAL — SPECIFIC WORDING, NOT A NARROW REQUIREMENT: the intent must be worded
specifically but stay broad enough that many real companies verifiably match. Do NOT
narrow it to a single tool, a single named event, or a niche so tight that fewer than a
handful of companies qualify. Specificity belongs in the phrasing, not in shrinking the
candidate pool.

ALLOWED COMPANY STAGES: Seed, Series A, Series B, Series C+, Private Equity, Public

STAGE DISTRIBUTION — SPREAD EVENLY ACROSS STAGES:
The benchmark needs to test miner performance at ALL stages, not just late-stage companies. Skewing toward Series C+/Public (because those have the most PR coverage) makes the benchmark too easy and fails to test the harder verification cases.

Target distribution across the 20 ICPs (approximate, ±2 per bucket is fine):
- Seed: 2-3 ICPs
- Series A: 4-5 ICPs
- Series B: 4-5 ICPs
- Series C+: 3-4 ICPs
- Private Equity: 1-2 ICPs
- Public: 2-3 ICPs

Do NOT cluster on later stages just because they're easier to verify. The realism rule still applies (every ICP must have a real `verified_example_company`), but Seed and Series A startups exist with verifiable funding announcements — find them.

ALLOWED EMPLOYEE BANDS — USE THESE EXACT LINKEDIN BUCKETS ONLY:
11-50, 51-200, 201-500, 501-1,000, 1,001-5,000, 5,001-10,000, 10,001+
- `employee_count` must be a JSON array of the allowed LinkedIn buckets.
- Prefer 3-5 contiguous buckets around the most realistic target size.
- Do not use fake broad ranges like "51-5000".

ALLOWED GEOGRAPHIES — STRONGLY PREFER BROAD VALUES:
EXACTLY {international_target} of the {total_icps} ICPs MUST use an international geography from this list (pick industries where that market genuinely thrives, e.g. FinTech in London, Mining tech in Australia, Payments in Singapore):
- "United Kingdom" / "United Kingdom, London"
- "Ireland"
- "Canada" / "Canada, Toronto"
- "Australia" / "Australia, Sydney"
- "New Zealand"
- "Singapore"
- "United Arab Emirates, Dubai" / "United Arab Emirates, Abu Dhabi"
These are English-speaking business markets only; never use any other country and never use a bare region name like "Europe" or "APAC". For international ICPs, `country` must be the country portion of the geography (e.g. "United Kingdom", "Canada").

The remaining {domestic_count} ICPs are United States:
- "United States" (whole country — use this for ~50% of the US ICPs)
- "United States, West Coast" / "Northeast" / "Midwest" / "South" / "Southwest" (~40%)
- "United States, <State>" only when the industry has a known concentration there (~10%)

INDUSTRY × INTENT PAIRING (Sonar should naturally honor these):
- Service industries (Consulting, Legal, Accounting, Recruiting) → leadership change, hiring, expansion, acquisition, partnership — NOT product launch
- Real Estate / Commercial Real Estate / PropTech → acquisition, expansion, leadership change, funding, facility opening — NOT product launch
- Industrial Manufacturing → acquisition, expansion, partnership, facility opening, hiring — NOT SaaS-style product launches
- Tech / SaaS / AI / Hardware / Biotech / Payments / FinTech / Cyber / Health → all intents work
- Banking / large traditional finance → leadership change, acquisition, regulatory clearance, partnership

STAGE × INTENT PAIRING:
- "Acquired another company" → REQUIRES Series B or later
- "Announced a strategic partnership" → REQUIRES Series B or later
- "Recent factory / facility / store opening" → REQUIRES Series A or later
- All other intent × stage combinations are valid

OUTPUT — JSON ONLY, NO PROSE, NO MARKDOWN

{{
  "icps": [
    {{
      "icp_id": "icp_{set_id}_001",
      "prompt": "<one-sentence salesperson-voice description>",
      "industry": "<from industry list, in order>",
      "sub_industry": "<natural sub-industry>",
      "geography": "<from allowed geographies, prefer broad>",
      "country": "<the country portion of the geography, e.g. United States, United Kingdom, Canada>",
      "employee_count": ["<3-5 contiguous allowed LinkedIn buckets>"],
      "company_stage": "<from allowed stages>",
      "product_service": "<a specific, descriptive value proposition — what the company actually sells and the job it does for its buyer, e.g. 'A subscription B2B platform that revenue and operations teams use to manage pipeline, deals, and customer workflows'. NOT a bare category like 'B2B software'. Do NOT narrow to a single named tool.>",
      "required_attribute": "<a specific, descriptive attribute the company must have, written like the product_service — e.g. 'Sells a subscription software platform used by revenue or operations teams to manage pipeline, deals, or customer workflows'. Specific WORDING, but broad enough that many real companies match. NOT the bare template 'offers or provides X'.>",
      "intent_signal": "<a specific, descriptive intent sentence in fulfillment style — see the INTENT SIGNAL section above>",
      "intent_category": "<FUNDING | ACQUISITION | PARTNERSHIP | PRODUCT_LAUNCH | LEADERSHIP_CHANGE | MARKET_EXPANSION | REGULATORY_CLEARANCE | FACILITY_OPENING | HIRING>",
      "intent_max_age_days": 365,
      "intent_signals": ["<required intent first, specific fulfillment-style sentence>", "<optional bonus intent second, specific fulfillment-style sentence>"],
      "bonus_intents": [
        {{"intent_signal": "<optional bonus intent>", "intent_category": "<matching category>", "intent_max_age_days": 365}}
      ],
      "verified_example_company": "<MANDATORY: the real company name you found while verifying this ICP>"
    }}
  ]
}}

FINAL CHECK before output (for every ICP):
1. Is `verified_example_company` a real, currently-operating company? If not, REWRITE with broader constraints.
2. Does the named example company actually match ALL the ICP's stated criteria — including the specific `required_attribute` and `intent_signal`? If not, REWRITE.
3. SUPPLY CHECK: can you name at least THREE more real, currently-operating companies (four total) that verifiably match this ICP's specific `required_attribute` and `intent_signal`? If you cannot, the attribute or intent is too narrow — keep the specific WORDING but broaden the requirement (or broaden geography/stage/employee band) until many real companies match. Specific phrasing, broad candidate pool.
4. Are the `product_service` and `required_attribute` specific, descriptive value propositions (like a real sales brief) rather than bare categories or the "offers or provides X" template?
5. Is each `intent_signal` a specific fulfillment-style sentence naming the behavior and the kind of evidence, rather than a bare label?
6. Is the geography broad enough that real candidates exist?
7. Are there exactly 20 ICPs, one per industry in the listed order?
8. Are EXACTLY {international_target} ICPs international (non-US, from the allowed international list) with `country` matching their geography?
9. No job titles, no seniority, no contact-level descriptors in the prompts?"""

    international_target = international_icp_target(total_icps)
    system_prompt = (
        system_prompt
        .replace("{international_target}", str(international_target))
        .replace("{total_icps}", str(total_icps))
        .replace("{domestic_count}", str(total_icps - international_target))
    )

    user_prompt = f"""Generate 20 ICPs for set_id={set_id}. Follow every instruction in the system message exactly. Output JSON only, no commentary."""

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
                    # Sonar drifts from JSON format at higher temperatures;
                    # 0.7 keeps it grounded while still varying voice across the 20.
                    "temperature": 0.7,
                    "max_tokens": 16000,
                    # Perplexity Sonar does NOT accept `response_format: json_object`;
                    # the prompt explicitly demands JSON-only output instead.
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
        
        # Strip Sonar-style markdown fences and surrounding prose if present.
        # Sonar often returns ```json ... ``` blocks despite the prompt.
        stripped = content.strip()
        if stripped.startswith("```"):
            # Drop the opening fence (optionally "```json")
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
            # Drop the closing fence
            if "```" in stripped:
                stripped = stripped.rsplit("```", 1)[0]
            stripped = stripped.strip()
        # Last-resort: extract the first {...} JSON object substring
        if not stripped.startswith("{"):
            obrace = stripped.find("{")
            cbrace = stripped.rfind("}")
            if obrace >= 0 and cbrace > obrace:
                stripped = stripped[obrace : cbrace + 1]
        content = stripped

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

            intent_signals = _normalize_intent_signals(icp.get("intent_signals"))
            if not intent_signals and icp.get("intent_signal"):
                intent_signals = _normalize_intent_signals(icp.get("intent_signal"))
            if not intent_signals:
                intent_signals = random.sample(INTENT_SIGNALS, random.randint(1, 2))
                logger.warning(f"ICP {icp_id} had empty intent_signals from LLM, assigned fallback: {intent_signals}")

            # Default to whole-US (broadest supply) when the LLM omits geography.
            # State-level defaults (e.g. "United States, California") narrow the
            # intersection unnecessarily and cause downstream empty markets.
            geography = icp.get("geography", "United States")

            # Allow the US plus the English-speaking international markets
            # (25% quota); override anything else to whole-US. Geography is
            # country-first, so the country is its first comma segment.
            international_match = next(
                (
                    candidate
                    for candidate in INTERNATIONAL_GEOGRAPHIES
                    if geography
                    and geography.split(",")[0].strip().lower()
                    == candidate.split(",")[0].strip().lower()
                ),
                None,
            )
            if geography and "United States" in geography:
                country = "United States"
            elif international_match:
                country = international_match.split(",")[0].strip()
            else:
                logger.warning(
                    f"ICP {icp_id} has unsupported geography {geography!r}, "
                    f"overriding to 'United States' (whole-country, broad supply)"
                )
                geography = "United States"
                country = "United States"

            # COMPANY-MODE ONLY: do NOT carry forward target_roles or
            # target_seniority — they are legacy contact-mode fields. We
            # populate them as empty defaults to satisfy any older miner
            # code that still reads them via dict.get(), but no real role
            # data flows through.
            # Capture the verified example company (Sonar's supply receipt) — the
            # whole point of using Sonar is that this field is non-empty, proving
            # the ICP has real-world supply. If empty, log a warning but keep
            # the ICP (fail-open — Sonar sometimes omits the field even when
            # it found one).
            verified_example = (icp.get("verified_example_company") or "").strip()
            if not verified_example:
                logger.warning(
                    f"ICP {icp_id} has empty verified_example_company — Sonar "
                    f"may not have grounded the supply check for this combo "
                    f"(industry={industry_normalized})"
                )

            sub_industry = icp.get("sub_industry", SUB_INDUSTRIES.get(industry_normalized, ["General"])[0])
            validated_icp = {
                "icp_id": icp_id,
                "prompt": prompt,
                "industry": industry_normalized,
                "sub_industry": sub_industry,
                "target_roles": [],
                "target_seniority": "",
                "employee_count": icp.get("employee_count", "51-200"),
                "company_stage": icp.get("company_stage", "Series A"),
                "geography": geography,
                "country": country,
                "product_service": icp.get("product_service", "Software solution"),
                "intent_signals": intent_signals,
                "intent_signal": icp.get("intent_signal", intent_signals[0]),
                "intent_category": icp.get("intent_category", ""),
                "intent_max_age_days": icp.get("intent_max_age_days"),
                "bonus_intents": icp.get("bonus_intents", []),
                "required_attribute": icp.get("required_attribute", ""),
                "buyer_description": prompt,                  # Legacy alias of prompt
                "verified_example_company": verified_example, # Sonar's supply receipt
            }
            validated_icp = canonicalize_generated_icp(
                validated_icp,
                industry=industry_normalized,
                sub_industry=str(sub_industry or ""),
            )

            validated_icps.append(validated_icp)

        # Set is exactly the configured industry count; allow some slack but anything significantly
        # short of expected count is a bad generation and we fall back.
        min_acceptable = max(int(total_icps * 0.9), total_icps - 2)
        if len(validated_icps) < min_acceptable:
            logger.error(f"Only {len(validated_icps)} valid ICPs (expected ~{total_icps}), falling back to template")
            return None
        
        # If the distribution is slightly imperfect, that's OK - the LLM output is approximate
        logger.info(f"Validated {len(validated_icps)} ICPs with distribution: {actual_distribution}")

        # International quota (25% of the set, English-speaking markets). The
        # generation prompt demands the exact count; a short set still ships
        # (template fallback would be worse) but never silently — the tag
        # below is the alert hook for drift.
        international_count = count_international_icps(validated_icps)
        if international_count < international_target:
            logger.warning(
                "icp_international_quota_missed generated=%d target=%d total=%d",
                international_count,
                international_target,
                len(validated_icps),
            )
        else:
            logger.info(
                f"International ICP quota met: {international_count}/{international_target}"
            )

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
    # Stage first, then size FROM the stage's coherent band list — never an
    # independent draw. The ICP carries the stage's full band list so a pinned
    # stage can't pincer the ICP into one arbitrary band.
    company_stage = random.choice(COMPANY_STAGES)
    stage_buckets = list(STAGE_EMPLOYEE_BUCKETS[company_stage])
    employee_count_range = random.choice(stage_buckets)
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

    icp = {
        "icp_id": icp_id,
        "prompt": prompt,
        "industry": industry,
        "sub_industry": sub_industry,
        # Legacy fields — kept as empty defaults so older miner code that
        # dict.gets them doesn't crash. The competition is company-only.
        "target_roles": [],
        "target_seniority": "",
        "employee_count": employee_count_range,
        # Full coherent band list for the pinned stage; canonicalization takes
        # an explicit multi-band list verbatim (no radius expansion beyond it).
        "employee_count_buckets": stage_buckets,
        "company_stage": company_stage,
        "geography": geography,
        "country": country,
        "product_service": product,
        "intent_signals": intent_signals,
        "buyer_description": prompt,  # Legacy alias of prompt
    }
    return canonicalize_generated_icp(icp, industry=industry, sub_industry=sub_industry)


COMPANY_GOAL_MIN = 1
COMPANY_GOAL_MAX = 25
COMPANY_GOAL_AVERAGE = 5

# Lab exclusion lists: each generated ICP names real companies the model must
# NOT return, so the benchmark exercises exclusion honoring end to end (the
# scorer zeroes any returned excluded company). Kept small by default so an
# ICP with thin supply is not starved into a zero by its own exclusion.
_EXCLUSIONS_ENABLED_ENV = "RESEARCH_LAB_ICP_EXCLUSIONS_ENABLED"
_EXCLUSIONS_COUNT_ENV = "RESEARCH_LAB_ICP_EXCLUSIONS_COUNT"
_EXCLUSIONS_DEFAULT_COUNT = 1
_EXCLUSIONS_MAX_COUNT = 3
_EXCLUSION_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")


def _exclusions_enabled() -> bool:
    return os.getenv(_EXCLUSIONS_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _exclusions_count() -> int:
    try:
        n = int(os.getenv(_EXCLUSIONS_COUNT_ENV, str(_EXCLUSIONS_DEFAULT_COUNT)))
    except ValueError:
        n = _EXCLUSIONS_DEFAULT_COUNT
    return max(1, min(n, _EXCLUSIONS_MAX_COUNT))


def generate_exclusions_for_icp(icp: Dict[str, Any], count: Optional[int] = None,
                                timeout_s: float = 45.0) -> List[str]:
    """Real companies matching this ICP's hard profile, as flat exclusion
    entries (domain preferred, name fallback). Fail-open: any provider or
    parse problem returns [] — an ICP without exclusions is always valid."""
    if not OPENROUTER_API_KEY:
        logger.warning("icp_exclusions_skipped reason=no_openrouter_key")
        return []
    want = count if count is not None else _exclusions_count()
    bands = icp.get("employee_count")
    bands_text = ", ".join(bands) if isinstance(bands, (list, tuple)) else str(bands or "")
    plural = "y" if want == 1 else "ies"
    stage = str(icp.get("company_stage") or "").strip()
    prompt = (
        f"Name {want} real, currently-operating compan{plural} that match "
        f"ALL of this profile:\n"
        f"- Industry: {icp.get('industry', '')} ({icp.get('sub_industry', '')})\n"
        f"- Headquarters: {icp.get('geography', icp.get('country', ''))}\n"
        f"- Employee count in one of these LinkedIn bands: {bands_text}\n"
        + (f"- Funding/ownership stage: {stage}\n" if stage else "")
        + "Only name companies you can verify exist on the web right now. "
          'Return STRICT JSON only: [{"name": "<company name>", "domain": "<bare homepage domain like example.com>"}]'
    )
    try:
        response = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0},
            timeout=timeout_s,
        )
        if response.status_code != 200:
            logger.warning("icp_exclusions_failed status=%s", response.status_code)
            return []
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*\]", content, re.S)
        rows = json.loads(match.group(0)) if match else []
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("icp_exclusions_failed error=%s", str(exc)[:120])
        return []
    entries: List[str] = []
    for row in rows[:want] if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain") or "").strip().lower()
        domain = re.sub(r"^https?://", "", domain).split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        name = " ".join(str(row.get("name") or "").split())[:120]
        if domain and _EXCLUSION_DOMAIN_RE.match(domain):
            entries.append(domain)
        elif name:
            entries.append(name)
    return list(dict.fromkeys(entries))[:want]


def attach_generated_exclusions(icps: List[Dict[str, Any]]) -> None:
    """Populate excluded_companies on each generated ICP (fail-open per ICP)."""
    if not _exclusions_enabled():
        logger.info("icp_exclusions_disabled via %s", _EXCLUSIONS_ENABLED_ENV)
        for icp in icps:
            icp.setdefault("excluded_companies", [])
        return
    for icp in icps:
        icp["excluded_companies"] = generate_exclusions_for_icp(icp)


def _widen_employee_count_to_stage(icp: Dict[str, Any]) -> None:
    """Union the ICP's employee_count with its stage's full coherent band list
    (CEO: use ALL corresponding ranges when a stage is defined). Union — never
    replace — so a realism-grounded band (and its verified_example_company) is
    always preserved; stages outside the dict are left untouched."""
    stage = str(icp.get("company_stage") or "").strip()
    bands = STAGE_EMPLOYEE_BUCKETS.get(stage)
    if not bands:
        return
    existing = icp.get("employee_count")
    if isinstance(existing, (list, tuple)):
        existing_list = [str(b).strip() for b in existing if str(b).strip()]
    elif existing:
        existing_list = [str(existing).strip()]
    else:
        existing_list = []
    order = {b: i for i, b in enumerate(LINKEDIN_EMPLOYEE_BUCKETS)}
    merged = list(dict.fromkeys([*existing_list, *bands]))
    icp["employee_count"] = sorted(merged, key=lambda b: order.get(b, 999))


def apply_generated_icp_contract(icps: List[Dict[str, Any]]) -> None:
    """Apply the ICP-contract features that must ride on EVERY generated set,
    regardless of which generator produced it. The production OpenRouter path
    builds ICP dicts directly and previously skipped these, so exclusions,
    the company goal, and stage-size coherence never reached live ICPs.

    Exclusions are attached separately by the async caller because they do
    provider I/O and must not block the event loop.
    """
    for icp in icps:
        if not isinstance(icp, dict):
            continue
        icp["max_companies"] = COMPANY_GOAL_AVERAGE
        _widen_employee_count_to_stage(icp)


def allocate_company_goals(
    count: int,
    *,
    total: Optional[int] = None,
    goal_min: int = COMPANY_GOAL_MIN,
    goal_max: int = COMPANY_GOAL_MAX,
) -> List[int]:
    """Per-ICP company goals in [goal_min, goal_max] summing exactly to
    ``total`` (default: COMPANY_GOAL_AVERAGE per ICP, i.e. 100 for a 20-ICP
    set). Draws from the caller's current ``random`` stream so seeded sets
    stay reproducible.

    Everyone starts at the floor, then the remainder is dealt one unit at a
    time to a random ICP that still has headroom — an unbiased composition
    that cannot violate either bound or the total.
    """
    if count <= 0:
        return []
    if total is None:
        total = COMPANY_GOAL_AVERAGE * count
    total = max(goal_min * count, min(total, goal_max * count))
    goals = [goal_min] * count
    remaining = total - goal_min * count
    open_indices = list(range(count))
    while remaining > 0 and open_indices:
        pick = random.choice(open_indices)
        goals[pick] += 1
        remaining -= 1
        if goals[pick] >= goal_max:
            open_indices.remove(pick)
    return goals


def generate_icp_set(
    set_id: int,
    total_icps: int = 20,
    base_seed: Optional[int] = None
) -> tuple:
    """
    Generate a complete ICP set — one ICP per industry across the 20 distinct
    industries in ``INDUSTRY_DISTRIBUTION``.

    Args:
        set_id: Set identifier (YYYYMMDD format)
        total_icps: Number of ICPs to generate (default 20, one per industry).
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

    # Per-ICP company goal: uniform 5 for every ICP for now, so benchmark
    # volume and per-company cost stay exactly as the flat design costs
    # (5 x 20 = 100 companies per set). allocate_company_goals() is the
    # ready-made varied allocation (1-25 summing to the same total) for when
    # operators decide to switch.
    for icp in icps:
        icp["max_companies"] = COMPANY_GOAL_AVERAGE

    # Lab exclusion lists (real matching companies the model must avoid);
    # attached before hashing so the set hash covers them.
    attach_generated_exclusions(icps)

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
    now_utc = _rebenchmark_now()
    
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

    Generates exactly ``len(INDUSTRY_DISTRIBUTION)`` ICPs (currently 20, one
    per industry).

    Args:
        for_date: Optional date to generate for (defaults to today UTC)

    Returns:
        set_id if successful, None otherwise
    """
    if for_date is None:
        for_date = _rebenchmark_now()
    
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

        # Set-level ICP contract: uniform company goal + stage-size coherence
        # (sync, no I/O), then web-verified lab exclusions (provider I/O, run
        # off the event loop). Applied here so they ride the production
        # OpenRouter path, not just the key-absent template fallback. The set
        # hash is recomputed afterward so the stored ICPs, the stored hash, and
        # the transparency-log hash all cover the final ICPs.
        apply_generated_icp_contract(icps)
        try:
            await asyncio.to_thread(attach_generated_exclusions, icps)
        except Exception as exc:  # noqa: BLE001 - exclusions are fail-open
            logger.warning("icp_exclusions_attach_failed error=%s", str(exc)[:160])
            for icp in icps:
                if isinstance(icp, dict):
                    icp.setdefault("excluded_companies", [])
        icp_hash = compute_icp_set_hash(icps)
        logger.info(
            "icp_contract_applied max_companies=%s exclusions=%d/%d stage_widened=%d hash=%s",
            COMPANY_GOAL_AVERAGE,
            sum(1 for i in icps if isinstance(i, dict) and i.get("excluded_companies")),
            len(icps),
            sum(1 for i in icps if isinstance(i, dict)
                and str(i.get("company_stage") or "") in STAGE_EMPLOYEE_BUCKETS),
            icp_hash[:16],
        )

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
                "ts": _rebenchmark_now().isoformat(),
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
            now = _rebenchmark_now()
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
                
                now = _rebenchmark_now()
                
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
