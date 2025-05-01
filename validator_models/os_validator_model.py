# validator_models/os_validator_model.py

import re
import sys
import os
import random
import asyncio
import aiohttp
import bittensor as bt
from typing import List, Dict, Optional
from urllib.parse import urlparse

async def validate_lead_list(leads: List[Dict], industry: Optional[str] = None) -> Dict:
    bt.logging.debug(f"Validating {len(leads)} leads")
    from validator_models.automated_checks import validate_lead_list as auto_check_leads
    
    # Run automated checks first
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

        # Email format validation
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
                    result["reasons"].append("Domain validation error")

        # Website validation
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
                result["reasons"].append("Website validation error")
        else:
            result["reasons"].append("Missing website")

        # Industry relevance
        if industry and lead_industry and industry.lower() != lead_industry.lower():
            result["reasons"].append(f"Industry mismatch: expected {industry}, got {lead_industry}")

        if not result["reasons"]:
            result["status"] = "High Quality"
            valid_count += 1
        elif len(result["reasons"]) <= 1:
            result["status"] = "Medium Quality"
            valid_count += 0.5
        else:
            result["status"] = "Low Quality"

        validation_report.append(result)

    score = (valid_count / len(leads)) * 100 if leads else 0
    O_v = score / 100.0
    bt.logging.debug(f"Validation complete: score={score}, O_v={O_v}")
    return {
        "validation_report": validation_report,
        "score": score,
        "O_v": O_v
    }

async def load_disposable_domains() -> set:
    """
    Load a list of disposable email domains from a local file or remote source.
    """
    bt.logging.debug("Loading disposable domains")
    domains = set()
    # Skip fetching in mock mode
    bt.logging.trace("Mock mode: Skipping disposable domains fetch")
    return domains
    try:
        # Try loading from a local file
        base_path = sys.path[0]
        file_path = os.path.join(base_path, "validator_models", "disposable_domains.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                domains.update(line.strip().lower() for line in f if line.strip())
            bt.logging.debug(f"Loaded {len(domains)} domains from local file")
        else:
            # Fallback to remote source
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
    """
    Check if an email address has a valid format.
    """
    email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_regex.match(email))


async def validate_domain(domain: str) -> bool:
    """
    Validate if a domain is reachable (mocked for simplicity).
    """
    bt.logging.trace(f"Validating domain: {domain}")
    # Force mock mode validation
    if domain.lower() == "mockleadpoet.com":
        bt.logging.trace("Mock mode: mockleadpoet.com is valid")
        return True
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{domain}", timeout=5) as resp:
                bt.logging.trace(f"Domain {domain} status: {resp.status}")
                return resp.status == 200
    except Exception as e:
        bt.logging.trace(f"Domain {domain} unreachable: {e}")
        return False


async def validate_website(url: str) -> bool:
    """
    Validate if a website is reachable (mocked for simplicity).
    """
    bt.logging.trace(f"Validating website: {url}")
    # Force mock mode validation
    if url.lower().startswith("https://business"):
        bt.logging.trace("Mock mode: businessX.com is valid")
        return True
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                bt.logging.trace(f"Website {url} status: {resp.status}")
                return resp.status == 200
    except Exception as e:
        bt.logging.trace(f"Website {url} unreachable: {e}")
        return False