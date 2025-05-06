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
    bt.logging.debug(f"Validating {len(leads)} leads")
    from validator_models.automated_checks import validate_lead_list as auto_check_leads
    
    report = await auto_check_leads(leads)
    valid_count = sum(1 for entry in report if entry["status"] == "Valid")
    if valid_count / len(leads) < 0.9:
        bt.logging.warning("Lead list failed automated checks")
        return {"validation_report": report, "score": 0, "O_v": 0.0}

    validation_report = []
    valid_count = 0
    disposable_domains = await load_disposable_domains()

    for idx, lead in enumerate(leads):
        email = lead.get("Owner(s) Email", "")
        website = lead.get("Website", "")
        lead_industry = lead.get("Industry", "")
        result = {"lead_index": idx, "email": email, "status": "Invalid", "reasons": []}

        if not is_valid_email(email):
            result["reasons"].append("Invalid email format")
        else:
            domain = email.split("@")[1].lower()
            if domain in disposable_domains:
                result["reasons"].append("Disposable email domain")
            else:
                try:
                    domain_valid = await validate_domain(domain)
                    if not domain_valid:
                        result["reasons"].append("Invalid email domain")
                except Exception as e:
                    result["reasons"].append(f"Domain validation error: {str(e)}")

        if website:
            try:
                parsed_url = urlparse(website)
                if not parsed_url.scheme or not parsed_url.netloc:
                    result["reasons"].append("Invalid website URL")
                else:
                    website_valid = await validate_website(website)
                    if not website_valid:
                        result["reasons"].append("Website unreachable")
            except Exception as e:
                result["reasons"].append(f"Website validation error: {str(e)}")
        else:
            result["reasons"].append("Missing website")

        # Only check industry if explicitly specified
        if industry and lead_industry:
            if industry.lower() != lead_industry.lower():
                result["reasons"].append(f"Industry mismatch: expected {industry}, got {lead_industry}")

        if email and is_valid_email(email):
            try:
                engagement_valid, engagement_reason = await check_email_engagement(email)
                if not engagement_valid:
                    result["reasons"].append(engagement_reason)
            except Exception as e:
                result["reasons"].append(f"Email engagement check failed: {str(e)}")

        if not result["reasons"]:
            result["status"] = "High Quality"
            valid_count += 1
        elif len(result["reasons"]) == 1 and "Mock: No recent email engagement detected" in result["reasons"]:
            result["status"] = "Medium Quality"
            valid_count += 0.75
        else:
            result["status"] = "Low Quality"
            valid_count += 0.25

        validation_report.append(result)

    score = (valid_count / len(leads)) * 100 if leads else 0
    O_v = score / 100.0
    bt.logging.debug(f"Validation complete: score={score}, O_v={O_v}")
    bt.logging.debug(f"Validation report: {validation_report}")
    return {
        "validation_report": validation_report,
        "score": score,
        "O_v": O_v
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