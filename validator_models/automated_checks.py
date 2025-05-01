# validator_models/automated_checks.py

import aiohttp
import asyncio
import dns.resolver
import pickle
import os
from urllib.parse import urlparse
import bittensor as bt
from pygod.detector import DOMINANT
import numpy as np

HUNTER_API_KEY = "YOUR_HUNTER_API_KEY"
DISPOSABLE_DOMAINS = {"mailinator.com", "temp-mail.org", "guerrillamail.com"}
EMAIL_CACHE_FILE = "email_verification_cache.pkl"

async def load_email_cache():
    if os.path.exists(EMAIL_CACHE_FILE):
        try:
            with open(EMAIL_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

async def save_email_cache(cache):
    try:
        with open(EMAIL_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

EMAIL_CACHE = asyncio.run(load_email_cache())

async def is_disposable_email(email: str) -> tuple[bool, str]:
    domain = email.split("@")[1].lower() if "@" in email else ""
    return domain in DISPOSABLE_DOMAINS, "Disposable domain" if domain in DISPOSABLE_DOMAINS else "Not disposable"

async def check_domain_existence(domain: str) -> tuple[bool, str]:
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(domain, "MX"))
        return True, "Domain has MX records"
    except Exception as e:
        return False, f"Domain check failed: {str(e)}"

async def verify_email(email: str) -> tuple[bool, str]:
    if email in EMAIL_CACHE:
        return EMAIL_CACHE[email]
    is_disposable, reason = await is_disposable_email(email)
    if is_disposable:
        EMAIL_CACHE[email] = (False, reason)
        await save_email_cache(EMAIL_CACHE)
        return False, reason
    domain = email.split("@")[1] if "@" in email else ""
    domain_exists, domain_reason = await check_domain_existence(domain)
    if not domain_exists:
        EMAIL_CACHE[email] = (False, domain_reason)
        await save_email_cache(EMAIL_CACHE)
        return False, domain_reason
    if "YOUR_HUNTER_API_KEY" in HUNTER_API_KEY:
        return True, "Mock pass"
    url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={HUNTER_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                data = await response.json()
                status = data.get("data", {}).get("status", "unknown")
                is_valid = status in ["valid", "accept_all"]
                reason = data.get("data", {}).get("result", "unknown")
                EMAIL_CACHE[email] = (is_valid, reason)
                await save_email_cache(EMAIL_CACHE)
                return is_valid, reason
    except Exception as e:
        EMAIL_CACHE[email] = (False, str(e))
        await save_email_cache(EMAIL_CACHE)
        return False, str(e)

async def verify_company(company_domain: str) -> tuple[bool, str]:
    if not company_domain:
        return False, "No domain provided"
    if not company_domain.startswith(("http://", "https://")):
        company_domain = f"https://{company_domain}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(company_domain, timeout=10, allow_redirects=True) as response:
                return response.status == 200, "Website accessible"
    except Exception as e:
        return False, f"Website inaccessible: {str(e)}"

async def check_duplicates(leads: list) -> bool:
    emails = [lead.get("Owner(s) Email", "") for lead in leads]
    return len(emails) != len(set(emails))

async def validate_lead_list(leads: list) -> list:
    if "YOUR_HUNTER_API_KEY" in HUNTER_API_KEY:
        bt.logging.info("Mock mode: Assuming all leads pass automated checks")
        return [{"lead_index": i, "email": lead.get("Owner(s) Email", ""), "company_domain": urlparse(lead.get("Website", "")).netloc, "status": "Valid", "reason": "Mock pass"} for i, lead in enumerate(leads)]
    
    if await check_duplicates(leads):
        bt.logging.warning("Duplicate emails detected")
        return [{"lead_index": i, "email": lead.get("Owner(s) Email", ""), "company_domain": urlparse(lead.get("Website", "")).netloc, "status": "Invalid", "reason": "Duplicate email"} for i, lead in enumerate(leads)]
    
    report = []
    for i, lead in enumerate(leads):
        email = lead.get("Owner(s) Email", "")
        domain = urlparse(lead.get("Website", "")).netloc if lead.get("Website") else ""
        email_valid, email_reason = await verify_email(email) if email else (False, "No email")
        company_valid, company_reason = await verify_company(domain)
        status = "Valid" if email_valid and company_valid else "Invalid"
        reason = f"Email: {email_reason}, Company: {company_reason}"
        report.append({
            "lead_index": i,
            "email": email,
            "company_domain": domain,
            "status": status,
            "reason": reason
        })
    return report

async def collusion_check(validators: list, responses: list) -> dict:
    """Simulate PyGOD/DBScan collusion detection."""
    validator_scores = []
    for v in validators:
        for r in responses:
            validation = await v.validate_leads(r.leads)
            validator_scores.append({"hotkey": v.wallet.hotkey.ss58_address, "O_v": validation["O_v"]})
    
    # Mock PyGOD analysis
    data = np.array([[s["O_v"]] for s in validator_scores])
    detector = DOMINANT()
    detector.fit(data)
    V_c = detector.decision_score_.max()
    
    collusion_flags = {}
    for v in validators:
        collusion_flags[v.wallet.hotkey.ss58_address] = 0 if V_c > 0.7 else 1
    return collusion_flags