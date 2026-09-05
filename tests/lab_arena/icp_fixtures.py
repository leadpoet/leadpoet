"""Small daily-ICP fixtures for Arena tests."""

from __future__ import annotations

from typing import Any, Dict


INDUSTRIES = (
    "Software",
    "Information Technology",
    "Artificial Intelligence",
    "Hardware",
    "Data and Analytics",
    "Privacy and Security",
    "Health Care",
    "Biotechnology",
    "Financial Services",
    "Lending and Investments",
    "Payments",
    "Manufacturing",
    "Commerce and Shopping",
    "Professional Services",
    "Advertising",
    "Sales and Marketing",
    "Real Estate",
    "Energy",
    "Education",
    "Transportation",
)


def raw_icp(industry: str, ordinal: int) -> Dict[str, Any]:
    signal = (
        "Announced a Series A or later funding round in the last 12 months, "
        "per a company announcement"
    )
    return {
        "icp_id": "icp_20260902_%03d" % (INDUSTRIES.index(industry) + 1),
        "prompt": "Find %s companies showing recent momentum" % industry.lower(),
        "industry": industry,
        "sub_industry": "%s specialists" % industry,
        "geography": "United States",
        "country": "United States",
        "employee_count": ["51-200", "201-500"],
        "company_stage": "Series A",
        "product_service": "A business platform used by operating teams",
        "required_attribute": "Sells a business product used by operating teams",
        "intent_signal": signal,
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
        "intent_signals": [signal],
        "bonus_intents": [],
        "verified_example_company": "%s Example Co %d" % (industry, ordinal),
    }


def daily_icps() -> list[Dict[str, Any]]:
    return [raw_icp(industry, index + 1) for index, industry in enumerate(INDUSTRIES)]
