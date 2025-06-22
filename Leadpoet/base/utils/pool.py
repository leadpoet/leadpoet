import json
import os
import threading
import bittensor as bt
from Leadpoet.validator import reward as _reward
from collections import defaultdict

DATA_DIR = "data"
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
CURATED_LEADS_FILE = os.path.join(DATA_DIR, "curated_leads.json")
_leads_lock = threading.Lock()
_curation_lock = threading.Lock()

def initialize_pool():
    """Initialize the leads pool file if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(CURATED_LEADS_FILE):
        with open(CURATED_LEADS_FILE, "w") as f:
            json.dump([], f)

def _save_curated_leads(curated_leads):
    """Save curated leads to the curated_leads.json file."""
    with open(CURATED_LEADS_FILE, "w") as f:
        json.dump(curated_leads, f, indent=2)

def record_delivery_rewards(delivered_leads, curator_hotkey):
    """
    Record reward events when leads are delivered to buyers.
    Now handles proportional splitting PER QUERY (no historical tracking).
    """
    # For each API query, we start fresh with no historical tracking
    # Track curators only within this current delivery batch
    current_query_curators = defaultdict(dict)
    
    # First pass: collect all curators for this query
    for lead in delivered_leads:
        email = lead.get('owner_email', lead.get('email', '')).lower()
        current_query_curators[email][curator_hotkey] = lead.get('conversion_score', 1.0)
    
    # Second pass: calculate proportional rewards
    for lead in delivered_leads:
        email = lead.get('owner_email', lead.get('email', '')).lower()
        lead["curated_by"] = curator_hotkey
        
        if not lead.get("source") or not lead.get("curated_by"):
            bt.logging.warning(f"Lead missing source or curated_by: {email}")
            continue
        
        # Check if multiple miners have curated this lead in THIS QUERY
        if email in current_query_curators and len(current_query_curators[email]) > 1:
            # Multiple curators in this query - calculate proportional score
            total_score = sum(current_query_curators[email].values())
            original_score = lead.get('conversion_score', 1.0)
            proportional_score = (original_score / total_score) * original_score
            lead["conversion_score"] = proportional_score
            
            bt.logging.info(f"Proportional reward event: {email} | "
                          f"Source: {lead['source']} | Curator: {curator_hotkey} | "
                          f"Score: {proportional_score:.3f} (original: {original_score:.3f}) | "
                          f"Total curators in this query: {len(current_query_curators[email])}")
        else:
            # Single curator in this query - record normally
            bt.logging.info(f"Single curator reward event: {email} | "
                          f"Source: {lead['source']} | Curator: {curator_hotkey}")
        
        try:
            _reward.record_event(lead)
        except Exception as e:
            bt.logging.error(f"Error recording reward event: {e}")
    
    # Save to curated_leads.json for storage (not used for mining)
    try:
        with open(CURATED_LEADS_FILE, "r") as f:
            existing_curated = json.load(f)
    except Exception:
        existing_curated = []
    
    # Add the delivered leads to curated_leads.json
    for lead in delivered_leads:
        existing_curated.append(lead)
    
    _save_curated_leads(existing_curated)
    
    # Print current reward distribution after recording
    _reward.print_current_rewards()

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

    # Note: Rewards are now recorded when leads are delivered to buyers, not when added to pool

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