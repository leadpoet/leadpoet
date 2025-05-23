import json
import os
import threading
import bt

DATA_DIR = "data"
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
_leads_lock = threading.Lock()

def initialize_pool():
    """Initialize the leads pool file if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "w") as f:
            json.dump([], f)

def add_to_pool(prospects):
    """Add valid prospects to leads.json, ensuring no duplicates by owner_email."""
    with _leads_lock:
        if not os.path.exists(LEADS_FILE):
            leads = []
        else:
            with open(LEADS_FILE, "r") as f:
                try:
                    leads = json.load(f)
                except Exception:
                    leads = []
        existing_emails = {lead.get("owner_email", "").lower() for lead in leads}
        new_prospects = [p for p in prospects if p.get("owner_email", "").lower() not in existing_emails]
        leads.extend(new_prospects)
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=2)

def check_duplicates(email: str) -> bool:
    """Check if owner_email exists in leads.json."""
    with _leads_lock:
        if not os.path.exists(LEADS_FILE):
            return False
        with open(LEADS_FILE, "r") as f:
            try:
                leads = json.load(f)
            except Exception:
                return False
        return any(lead.get("owner_email", "").lower() == email.lower() for lead in leads)

def get_leads_from_pool(num_leads, industry=None, region=None):
    """Get leads from the pool, filtered by industry and region if specified."""
    with _leads_lock:
        if not os.path.exists(LEADS_FILE):
            return []
            
        try:
            with open(LEADS_FILE, "r") as f:
                leads = json.load(f)
                
            # Filter leads by industry and region if specified
            filtered_leads = leads
            if industry:
                filtered_leads = [l for l in filtered_leads if l.get("industry", "").lower() == industry.lower()]
            if region:
                filtered_leads = [l for l in filtered_leads if l.get("region", "").lower() == region.lower()]
                
            # Sort by conversion score (highest first) and ensure all required fields
            filtered_leads = [l for l in filtered_leads if all(l.get(field) for field in 
                ["owner_email", "website", "business", "owner_full_name"])]
            filtered_leads.sort(key=lambda x: x.get("conversion_score", 0), reverse=True)
            
            # Return top N leads
            return filtered_leads[:num_leads]
        except Exception as e:
            bt.logging.error(f"Error reading leads from pool: {e}")
            return []