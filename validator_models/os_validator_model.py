import re
import sys
import os
import random
import asyncio
import aiohttp
import bittensor as bt
from typing import List, Dict, Optional
from urllib.parse import urlparse
from datetime import datetime, timedelta

MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "YOUR_MAILGUN_API_KEY")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "YOUR_MAILGUN_DOMAIN")
EMAIL_ENGAGEMENT_WINDOW_DAYS = 30

async def validate_lead_list(leads: List[Dict], industry: Optional[str] = None) -> Dict:
    bt.logging.debug(f"validate_lead_list raw input: {leads}")
    bt.logging.debug(f"Validating {len(leads)} leads")
    from validator_models.automated_checks import validate_lead_list as auto_check_leads
    
    # First map all leads to consistent field names
    mapped_leads = []
    for lead in leads:
        email = (
            lead.get("email") or
            lead.get("owner_email") or
            lead.get("Owner(s) Email") or
            lead.get("Email") or
            ""
        )
        mapped_lead = {
            "email": email,
            "website": lead.get("Website", lead.get("website", "")),
            "industry": lead.get("Industry", lead.get("industry", "")),
            "business": lead.get("Business", lead.get("business", "")),
            "owner_full_name": lead.get("Owner Full name", lead.get("owner_full_name", "")),
            "first": lead.get("First", lead.get("first", "")),
            "last": lead.get("Last", lead.get("last", "")),
            "linkedin": lead.get("LinkedIn", lead.get("linkedin", "")),
            "region": lead.get("Region", lead.get("region", "")),
            "conversion_score": lead.get("conversion_score", None)
        }
        mapped_leads.append(mapped_lead)
    
    # Run automated checks on the mapped leads
    report = await auto_check_leads(mapped_leads)
    
    valid_count = sum(1 for entry in report if entry["status"] == "Valid")
    if valid_count / len(mapped_leads) < 0.9:
        bt.logging.warning("Lead list failed automated checks")
        return {"validation_report": report, "score": 0, "O_v": 0.0}

    validation_report = []
    valid_count = 0
    disposable_domains = await load_disposable_domains()

    for idx, lead in enumerate(mapped_leads):
        email = lead["email"]  # Use the mapped email field
        website = lead["website"]
        lead_industry = lead["industry"]
        
        result = {
            "lead_index": idx,
            "email": email,  # This is now using the mapped field
            "status": "Invalid",
            "reasons": []
        }

        # First check for required fields
        if not email:
            result["status"] = "Rejected"
            result["reasons"].append("Missing required field: email")
            validation_report.append(result)
            continue

        if not website:
            result["status"] = "Rejected"
            result["reasons"].append("Missing required field: website")
            validation_report.append(result)
            continue

        # Then do the rest of validation
        if not is_valid_email(email):
            result["status"] = "Rejected"
            result["reasons"].append("Invalid email format")
            validation_report.append(result)
            continue

        domain = email.split("@")[1].lower()
        if domain in disposable_domains:
            result["status"] = "Rejected"
            result["reasons"].append("Disposable email domain")
            validation_report.append(result)
            continue

        try:
            domain_valid = await validate_domain(domain)
            if not domain_valid:
                result["status"] = "Rejected"
                result["reasons"].append("Invalid email domain")
                validation_report.append(result)
                continue
        except Exception as e:
            result["status"] = "Rejected"
            result["reasons"].append(f"Domain validation error: {str(e)}")
            validation_report.append(result)
            continue

        try:
            parsed_url = urlparse(website)
            if not parsed_url.scheme or not parsed_url.netloc:
                result["status"] = "Rejected"
                result["reasons"].append("Invalid website URL")
                validation_report.append(result)
                continue

            website_valid = await validate_website(website)
            if not website_valid:
                result["status"] = "Rejected"
                result["reasons"].append("Website unreachable")
                validation_report.append(result)
                continue
        except Exception as e:
            result["status"] = "Rejected"
            result["reasons"].append(f"Website validation error: {str(e)}")
            validation_report.append(result)
            continue

        # Only check industry if explicitly specified
        if industry and lead_industry:
            if industry.lower() != lead_industry.lower():
                result["status"] = "Rejected"
                result["reasons"].append(f"Industry mismatch: expected {industry}, got {lead_industry}")
                validation_report.append(result)
                continue

        if email and is_valid_email(email):
            try:
                engagement_valid, engagement_reason = await check_email_engagement(email)
                if not engagement_valid:
                    result["status"] = "Medium Quality"
                    result["reasons"].append(engagement_reason)
                    valid_count += 0.75
                else:
                    result["status"] = "High Quality"
                    valid_count += 1.0
            except Exception as e:
                result["status"] = "Medium Quality"
                result["reasons"].append(f"Email engagement check failed: {str(e)}")
            valid_count += 0.75
        else:
            result["status"] = "Rejected"
            result["reasons"].append("Invalid email format")
            validation_report.append(result)
            continue

        # --- NEW: simple conversion-score heuristic ------------------------
        conv = 0.5
        if lead["linkedin"]:                       conv += 0.2
        if lead["website"]:                        conv += 0.2
        if industry and lead_industry:             conv += 0.1 if industry.lower()==lead_industry.lower() else 0
        if lead.get("region"):                     conv += 0.1
        lead["conversion_score"] = round(min(conv, 1.0), 3)
        # -------------------------------------------------------------------
        result["status"] = "High Quality"
        valid_count += lead["conversion_score"]

        validation_report.append(result)

    score = (valid_count / len(mapped_leads)) * 100 if mapped_leads else 0
    O_v = score / 100.0
    bt.logging.debug(f"Validation complete: score={score}, O_v={O_v}")
    bt.logging.debug(f"Validation report: {validation_report}")
    return {
        "validation_report": validation_report,
        "score": score,
        "O_v": O_v,
        "scored_leads": mapped_leads
    }

async def check_email_engagement(email: str) -> tuple[bool, str]:
    bt.logging.trace(f"Checking email engagement for: {email}")
    if (bt.config().mock or 
        not MAILGUN_API_KEY or "YOUR_" in MAILGUN_API_KEY or 
        not MAILGUN_DOMAIN or "YOUR_" in MAILGUN_DOMAIN):
        bt.logging.debug(f"Mock mode: Simulating email engagement check for {email}")
        if random.random() < 0.9:  # Increased to 90% for robustness
            days_ago = random.randint(1, EMAIL_ENGAGEMENT_WINDOW_DAYS)
            timestamp = (datetime.utcnow() - timedelta(days=days_ago)).isoformat() + "Z"
            return True, f"Mock: Email opened {days_ago} days ago at {timestamp}"
        return False, "Mock: No recent email engagement detected"

    url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/events"
    params = {
        "recipient": email,
        "event": "opened",
        "begin": (datetime.utcnow() - timedelta(days=EMAIL_ENGAGEMENT_WINDOW_DAYS)).isoformat(),
        "limit": 1,
        "ascending": "desc"
    }
    auth = ("api", MAILGUN_API_KEY)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=auth, params=params, timeout=5) as response:
                if response.status != 200:
                    bt.logging.warning(f"Mailgun API error for {email}: HTTP {response.status}")
                    return False, f"Mailgun API error: HTTP {response.status}"
                data = await response.json()
                events = data.get("items", [])
                if not events:
                    return False, "No recent email opens detected"
                event = events[0]
                timestamp = event.get("timestamp")
                if timestamp:
                    event_time = datetime.fromtimestamp(timestamp)
                    days_ago = (datetime.utcnow() - event_time).days
                    if days_ago <= EMAIL_ENGAGEMENT_WINDOW_DAYS:
                        return True, f"Email opened {days_ago} days ago at {event_time.isoformat()}"
                    return False, f"Last email open was {days_ago} days ago, outside {EMAIL_ENGAGEMENT_WINDOW_DAYS}-day window"
                return False, "No valid timestamp in event data"
    except Exception as e:
        bt.logging.error(f"Error checking email engagement for {email}: {str(e)}")
        return False, f"Email engagement check failed: {str(e)}"

async def load_disposable_domains() -> set:
    bt.logging.debug("Loading disposable domains")
    domains = set()
    if bt.config().mock:
        bt.logging.trace("Mock mode: Skipping disposable domains fetch")
        return domains
    try:
        base_path = sys.path[0]
        file_path = os.path.join(base_path, "validator_models", "disposable_domains.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                domains.update(line.strip().lower() for line in f if line.strip())
            bt.logging.debug(f"Loaded {len(domains)} domains from local file")
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://raw.githubusercontent.com/disposable-email-domains/disposable-email-domains/master/disposable_email_blocklist.conf") as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        domains.update(line.strip().lower() for line in text.splitlines() if line.strip())
                        bt.logging.debug(f"Loaded {len(domains)} domains from remote source")
                    else:
                        bt.logging.warning(f"Failed to fetch disposable domains: HTTP {resp.status}")
    except Exception as e:
        bt.logging.error(f"Error loading disposable domains: {e}")
    return domains

def is_valid_email(email: str) -> bool:
    if not email or email.lower() == "no email":
        return False
    email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_regex.match(email))

async def validate_domain(domain: str) -> bool:
    bt.logging.trace(f"Validating domain: {domain}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{domain}", timeout=5) as resp:
                bt.logging.trace(f"Domain {domain} status: {resp.status}")
                return resp.status in [200, 403]
    except Exception as e:
        bt.logging.trace(f"Domain {domain} unreachable: {e}")
        return False

async def validate_website(url: str) -> bool:
    bt.logging.trace(f"Validating website: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                bt.logging.trace(f"Website {url} status: {resp.status}")
                return resp.status in [200, 403]
    except Exception as e:
        bt.logging.trace(f"Website {url} unreachable: {e}")
        return False

def extract_email(lead):
    for key in ["Owner(s) Email", "owner_email", "email", "Email"]:
        for actual_key in lead.keys():
            if actual_key.strip().lower() == key.strip().lower():
                value = lead[actual_key]
                if value and value.strip():
                    return value.strip()
    return ""