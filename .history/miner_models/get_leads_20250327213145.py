import requests
import os
import random
from urllib.parse import urlparse
import asyncio
import bittensor as bt

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "YOUR_HUNTER_API_KEY")
CLEARBIT_API_KEY = os.getenv("CLEARBIT_API_KEY", "YOUR_CLEARBIT_API_KEY")
COMPANY_LIST_URL = "https://raw.githubusercontent.com/Pranavmr100/Leadpoet/refs/heads/main/sampleleads.json"

industry_keywords = {
    "Tech & AI": ["saas", "software", "cloud", "tech"],
    "Finance & Fintech": ["fintech", "finance", "bank"],
    "Health & Wellness": ["health", "medical", "wellness"],
    "Media & Education": ["media", "entertainment", "education"],
    "Energy & Industry": ["energy", "manufacturing", "logistics"]
}

VALID_INDUSTRIES = list(industry_keywords.keys())

async def fetch_industry_from_api(domain):
    url = f"https://company.clearbit.com/v1/domains/{domain}"
    headers = {"Authorization": f"Bearer {CLEARBIT_API_KEY}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                sector = data.get("category", {}).get("sector", "").lower()
                if "software" in sector or "technology" in sector:
                    return "Tech & AI"
                elif "finance" in sector:
                    return "Finance & Fintech"
                elif "health" in sector or "education" in sector:
                    return "Health & Wellness"
                elif "media" in sector or "marketing" in sector:
                    return "Media & Education"
                elif "energy" in sector or "manufacturing" in sector:
                    return "Energy & Industry"
                return None
    except Exception:
        return None

async def assign_industry(name, website):
    domain = urlparse(website).netloc if website else ""
    name_lower = name.lower() if name else ""
    website_lower = website.lower() if website else ""
    if domain:
        api_industry = await fetch_industry_from_api(domain)
        if api_industry in VALID_INDUSTRIES:
            return api_industry
    industry_scores = {industry: 0 for industry in industry_keywords}
    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if (name_lower and keyword in name_lower) or (website_lower and keyword in website_lower):
                industry_scores[industry] += 1
    max_score = max(industry_scores.values())
    if max_score > 0:
        top_industries = [ind for ind, score in industry_scores.items() if score == max_score]
        return top_industries[0]
    return "Tech & AI"

async def get_emails_hunter(domain):
    url = f"https://api.hunter.io/v2/domain-search?domain={domain}&api_key={HUNTER_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                emails = data.get("data", {}).get("emails", [])
                return [email["value"] for email in emails]
    except Exception as e:
        bt.logging.error(f"Error fetching emails for {domain}: {e}")
        return []

async def get_leads(num_leads: int, industry: str = None, region: str = None) -> list:
    """Generate leads for LeadPoet miners."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(COMPANY_LIST_URL, timeout=30) as response:
                businesses = await response.json()
                random.shuffle(businesses)
    except Exception as e:
        bt.logging.error(f"Error fetching business list: {e}")
        return []

    leads = []
    for business in businesses:
        if len(leads) >= num_leads:
            break
        name = business.get("Business", "")
        website = business.get("Website", "")
        assigned_industry = await assign_industry(name, website)
        if industry and assigned_industry.lower() != industry.lower():
            continue
        domain = urlparse(website).netloc if website else ""
        if domain:
            hunter_emails = await get_emails_hunter(domain)
            json_emails = business.get("Owner(s) Email", "").split("/") if business.get("Owner(s) Email") else []
            all_emails = list(set(json_emails + hunter_emails))
            lead = {
                "Business": name,
                "Owner Full name": business.get("Owner Full name", ""),
                "First": business.get("First", ""),
                "Last": business.get("Last", ""),
                "Owner(s) Email": all_emails[0] if all_emails else "",
                "LinkedIn": business.get("LinkedIn", ""),
                "Website": website,
                "Industry": assigned_industry,
                "Region": region or "Unknown"
            }
            leads.append(lead)
    return leads