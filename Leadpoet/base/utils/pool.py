import json
import os
import threading
import bittensor as bt
from collections import defaultdict
from Leadpoet.validator import reward as _reward
import time

DATA_DIR = "data"
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
CURATED_LEADS_FILE = os.path.join(DATA_DIR, "curated_leads.json")
_leads_lock = threading.Lock()
_curated_lock = threading.Lock()
EMISSION_INTERVAL = 12      # seconds
state_lock = threading.Lock()
scores_file  = os.path.join(DATA_DIR, "scores.json")   # persistent Σ's

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

        # ‼️  NEVER store the curator field inside leads.json
        sanitised = []
        for p in prospects:
            p = dict(p)                       # shallow-copy
            p.pop("curated_by", None)         # strip curation info
            sanitised.append(p)

        new_prospects = [p for p in sanitised
                         if p.get("owner_email", "").lower() not in existing_emails]
        leads.extend(new_prospects)
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=2)

    # Note: Rewards are now recorded when leads are delivered to buyers, not when added to pool

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

def save_curated_leads(curated_leads):
    """Save curated leads to curated_leads.json."""
    with _curated_lock:
        try:
            with open(CURATED_LEADS_FILE, "r") as f:
                existing_curated = json.load(f)
        except Exception:
            existing_curated = []
        
        # Add new curated leads
        existing_curated.extend(curated_leads)
        
        with open(CURATED_LEADS_FILE, "w") as f:
            json.dump(existing_curated, f, indent=2)

def calculate_per_query_rewards(all_delivered_leads):
    """
    Calculate rewards for the current API query only (no historical tracking).
    Implements proper proportional splitting when multiple miners curate the same lead.
    """
    # Group leads by email to find duplicates within this query
    lead_groups = defaultdict(list)
    for lead in all_delivered_leads:
        email = lead.get('owner_email', lead.get('email', '')).lower()
        lead_groups[email].append(lead)
    
    # Track all unique hotkeys for sourcing and curating
    sourcing_scores = defaultdict(float)
    curating_scores = defaultdict(float)
    
    for email, leads in lead_groups.items():
        if len(leads) == 1:
            # Single curator - simple case
            lead = leads[0]
            source_hotkey = lead.get("source")
            curator_hotkey = lead.get("curated_by")
            score = lead.get('conversion_score', 1.0)
            
            # 40% for sourcing, 60% for curating
            sourcing_scores[source_hotkey] += score * 0.4
            curating_scores[curator_hotkey] += score * 0.6
            
            bt.logging.info(f"Single curator: {email} | Source: {source_hotkey} | Curator: {curator_hotkey} | Score: {score}")
            
        else:
            # Multiple curators for same lead - PROPORTIONAL SPLITTING
            source_hotkey = leads[0].get("source")  # All leads have same source
            total_curator_score = sum(lead.get('conversion_score', 1.0) for lead in leads)
            
            # 40% sourcing reward goes to source miner (divided equally if multiple instances)
            sourcing_reward_per_lead = (leads[0].get('conversion_score', 1.0) * 0.4) / len(leads)
            sourcing_scores[source_hotkey] += sourcing_reward_per_lead * len(leads)
            
            # 60% curating reward split proportionally among curators
            for lead in leads:
                curator_hotkey = lead.get("curated_by")
                original_score = lead.get('conversion_score', 1.0)
                
                # Proportional splitting: (curator_score / total_score) * total_curating_reward
                proportional_share = original_score / total_curator_score
                curating_reward = (leads[0].get('conversion_score', 1.0) * 0.6) * proportional_share
                
                curating_scores[curator_hotkey] += curating_reward
                
                bt.logging.info(f"Proportional curator: {email} | Source: {source_hotkey} | "
                              f"Curator: {curator_hotkey} | Share: {proportional_share:.3f} | "
                              f"Reward: {curating_reward:.3f}")
    
    # Calculate combined weights (W = S + K) - no need for 0.4*S + 0.6*K since we already applied the split
    combined_weights = {}
    all_hotkeys = set(sourcing_scores.keys()) | set(curating_scores.keys())
    
    for hotkey in all_hotkeys:
        sourcing_weight = sourcing_scores.get(hotkey, 0)
        curating_weight = curating_scores.get(hotkey, 0)
        combined_weights[hotkey] = sourcing_weight + curating_weight
    
    # Calculate emissions (200 Alpha total)
    total_weight = sum(combined_weights.values())
    emissions = {}
    for hotkey, weight in combined_weights.items():
        if total_weight > 0:
            emissions[hotkey] = 200.0 * (weight / total_weight)
        else:
            emissions[hotkey] = 0.0
    
    return {
        "S": dict(sourcing_scores),
        "K": dict(curating_scores), 
        "W": combined_weights,
        "E": emissions
    }

def record_delivery_rewards(delivered):
    rewards = calculate_per_query_rewards(delivered)

    with state_lock:
        scores = _load_scores()
        for hk, w in rewards["W"].items():
            scores[hk] = scores.get(hk, 0) + w     # accumulate
        _save_scores(scores)

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

def _load_scores():
    if not os.path.exists(scores_file):
        return {}
    with open(scores_file) as f:
        return json.load(f)

def _save_scores(scores):
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)

def _emission_loop():
    """Runs in a daemon thread; prints one table per block."""
    while True:
        time.sleep(EMISSION_INTERVAL)
        with state_lock:
            scores = _load_scores()
        if not scores:            # nothing yet
            continue
        total_W = sum(scores.values())
        
        # Only print in validator and miner processes, not API
        import sys
        if 'validator' in sys.argv or 'miner' in sys.argv:
            print("\n========== BLOCK EMISSION ==========")
            for hotkey, W in scores.items():
                share = (W / total_W) if total_W else 0
                print(f"{hotkey:20s}  W={W:7.3f}   share={share:6.1%}")
            print("====================================\n")

# fire the daemon
threading.Thread(target=_emission_loop, daemon=True).start()