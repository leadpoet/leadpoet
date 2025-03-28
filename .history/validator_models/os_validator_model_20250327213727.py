import json
import requests
import re
import asyncio
import aiohttp
from typing import Dict, Set, List

CLEARBIT_API_KEY = "YOUR_CLEARBIT_API_KEY"
DISPOSABLE_DOMAINS_URL = "https://raw.githubusercontent.com/Pranavmr100/Leadpoet/refs/heads/main/sampleleads.json"

async def load_disposable_domains():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(DISPOSABLE_DOMAINS_URL, timeout=10) as response:
                return set(await response.json())
    except Exception as e:
        bt.logging.error(f"Error loading disposable domains: {e}")
        return set()

async def get_company_industry(domain: str) -> str:
    url = f"https://company.clearbit.com/v1/domains/{domain}"
    headers = {"Authorization": f"Bearer {CLEARBIT_API_KEY}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("category", {}).get("sector", "Unknown")
    except Exception as e:
        bt.logging.error(f"Error fetching industry for {domain}: {e}")
        return "Unknown"

async def is_disposable_email(email: str, disposable_domains: Set[str]) -> bool:
    domain = email.split("@")[-1]
    return domain in disposable_domains

async def validate_lead(lead: Dict, industry: str, seen_emails: Set[str], disposable_domains: Set[str]) -> Dict:
    validation = {"status": "High Quality", "reasons": []}
    email = lead.get("Owner(s) Email", "")
    domain = urlparse(lead.get("Website", "")).netloc if lead.get("Website") else ""

    if not email or not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        validation["status"] = "Low Quality"
        validation["reasons"].append("Invalid email format")
    elif await is_disposable_email(email, disposable_domains):
        validation["status"] = "Low Quality"
        validation["reasons"].append("Disposable email domain")
    elif email in seen_emails:
        validation["status"] = "Low Quality"
        validation["reasons"].append("Duplicate email")
    else:
        seen_emails.add(email)

    company_industry = await get_company_industry(domain)
    if industry and company_industry.lower() != industry.lower():
        validation["status"] = "Low Quality"
        validation["reasons"].append(f"Industry mismatch: expected {industry}, got {company_industry}")

    if not validation["reasons"]:
        validation["reasons"].append("All checks passed")
    return validation

async def validate_lead_list(leads: List[Dict], industry: str) -> Dict:
    disposable_domains = await load_disposable_domains()
    seen_emails = set()
    validation_report = []
    high_quality_count = 0

    for i, lead in enumerate(leads):
        validation = await validate_lead(lead, industry, seen_emails, disposable_domains)
        validation_report.append({
            "lead_index": i,
            "email": lead.get("Owner(s) Email", "N/A"),
            "company_domain": lead.get("Website", "N/A"),
            "status": validation["status"],
            "reasons": validation["reasons"]
        })
        if validation["status"] == "High Quality":
            high_quality_count += 1

    total_leads = len(leads)
    score = (high_quality_count / total_leads) * 100 if total_leads > 0 else 0
    return {
        "validation_report": validation_report,
        "score": score,
        "estimated_good_leads": int((score / 100) * total_leads)
    }