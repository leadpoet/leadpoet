import json
import os
import bittensor as bt
import threading
from Leadpoet.utils.cloud_db import get_cloud_leads
from Leadpoet.base.utils import safe_json_load

DATA_DIR = "data"
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
CURATED_LEADS_FILE = os.path.join(DATA_DIR, "curated_leads.json")
_leads_lock = threading.Lock()
_curated_lock = threading.Lock()

def initialize_pool():
    """Initialize the leads pool file if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(CURATED_LEADS_FILE):
        with open(CURATED_LEADS_FILE, "w") as f:
            json.dump([], f)

def add_to_pool(prospects):
    """Add valid prospects to leads.json, ensuring no duplicates by email."""
    with _leads_lock:
        if not os.path.exists(LEADS_FILE):
            leads = []
        else:
            leads = safe_json_load(LEADS_FILE)
        existing_emails = {lead.get("email", "").lower() for lead in leads}

        sanitised = []
        for p in prospects:
            p = dict(p)
            p.pop("curated_by", None)
            p.pop("conversion_score", None)
            sanitised.append(p)

        new_prospects = [p for p in sanitised
                         if p.get("email", "").lower() not in existing_emails]
        leads.extend(new_prospects)
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=2)


def get_leads_from_pool(num_leads, industry=None, region=None, wallet=None):
    """Return up-to-date leads from Firestore with local JSON as fallback."""
    if wallet is not None:
        try:
            leads = get_cloud_leads(wallet, limit=max(1000, num_leads))
        except Exception as e:
            bt.logging.error(f"Cloud read failed, falling back to JSON: {e}")
            leads = []
    else:
        leads = []

    if not leads:
        with _leads_lock:
            if not os.path.exists(LEADS_FILE):
                return []

            leads = safe_json_load(LEADS_FILE)

    filtered_leads = leads
    if industry:
        filtered_leads = [lead for lead in filtered_leads
                          if lead.get("industry", "").lower() == industry.lower()]
    if region:
        filtered_leads = [lead for lead in filtered_leads
                          if lead.get("region", "").lower() == region.lower()]

    required_fields = ["email", "website", "business"]
    filtered_leads = [lead for lead in filtered_leads
                      if all(lead.get(f) for f in required_fields)]

    import random
    if len(filtered_leads) <= num_leads:
        return filtered_leads
    return random.sample(filtered_leads, num_leads)

def save_curated_leads(curated_leads):
    """Save curated leads to curated_leads.json."""
    with _curated_lock:
        existing_curated = safe_json_load(CURATED_LEADS_FILE)
        
        existing_curated.extend(curated_leads)
        
        with open(CURATED_LEADS_FILE, "w") as f:
            json.dump(existing_curated, f, indent=2)

def check_duplicates(email: str) -> bool:
    """Check if email exists in leads.json."""
    with _leads_lock:
        if not os.path.exists(LEADS_FILE):
            return False
        with open(LEADS_FILE, "r") as f:
            try:
                leads = json.load(f)
            except Exception:
                return False
        return any(lead.get("email", "").lower() == email.lower() for lead in leads)
