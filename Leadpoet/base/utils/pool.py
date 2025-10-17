import json
import os
import threading
import bittensor as bt
from collections import defaultdict
from Leadpoet.validator import reward as _reward
import time
import sys
from Leadpoet.validator.reward import record_event as _rec
from Leadpoet.utils.cloud_db import get_cloud_leads
from Leadpoet.base.utils import safe_json_load
from Leadpoet.utils.utils_lead_extraction import get_email

DATA_DIR = "data"
LEADS_FILE = os.path.join(DATA_DIR, "leads.json")
CURATED_LEADS_FILE = os.path.join(DATA_DIR, "curated_leads.json")
_leads_lock = threading.Lock()
_curated_lock = threading.Lock()
EMISSION_INTERVAL = 12
state_lock = threading.Lock()
scores_file  = os.path.join(DATA_DIR, "scores.json")

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

def calculate_per_query_rewards(all_delivered_leads):
    """
    Calculate rewards for the current API query only (no historical tracking).
    Implements proper proportional splitting when multiple miners curate the same lead.
    """
    bt.logging.info("=== REWARD CALCULATION DEBUG ===")
    bt.logging.info(f"Total delivered leads: {len(all_delivered_leads)}")
    
    # Group leads by email to find duplicates within this query
    lead_groups = defaultdict(list)
    for lead in all_delivered_leads:
        email = get_email(lead).lower()
        lead_groups[email].append(lead)
    
    bt.logging.info(f"Unique leads (by email): {len(lead_groups)}")
    for email, leads in lead_groups.items():
        bt.logging.info(f"  {email}: {len(leads)} instances")
        for i, lead in enumerate(leads):
            bt.logging.info(f"    Instance {i+1}: source={lead.get('source')}, curator={lead.get('curated_by')}, score={lead.get('conversion_score', 1.0)}")
    
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
            
            bt.logging.info(f"Single curator: {email} | Source: {source_hotkey} (+{score * 0.4:.3f}) | Curator: {curator_hotkey} (+{score * 0.6:.3f})")
            
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
                
                bt.logging.info(f"Proportional curator: {email} | Source: {source_hotkey} (+{sourcing_reward_per_lead:.3f}) | "
                              f"Curator: {curator_hotkey} (+{curating_reward:.3f}, share={proportional_share:.3f})")
    
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
    
    bt.logging.info("Final scores:")
    bt.logging.info(f"  Sourcing: {dict(sourcing_scores)}")
    bt.logging.info(f"  Curating: {dict(curating_scores)}")
    bt.logging.info(f"  Combined: {combined_weights}")
    bt.logging.info(f"  Emissions: {emissions}")
    bt.logging.info("=== END REWARD CALCULATION ===")
    
    return {
        "S": dict(sourcing_scores),
        "K": dict(curating_scores), 
        "W": combined_weights,
        "E": emissions
    }

def _print_emission_table(scores: dict):
    """
    Pretty-print the current block-emission table and flush stdout so it
    appears immediately.  Suppressed only for the API client.
    """
    if not scores:
        return
    if 'api' in sys.argv[0]:      #  don't spam the CLI client
        return
    total_W = sum(scores.values())
    print("\n========== BLOCK EMISSION ==========")
    for hotkey, W in scores.items():
        share = (W / total_W) if total_W else 0
        print(f"{hotkey:20s}  W={W:7.3f}   share={share:6.1%}")
    print("====================================\n")
    sys.stdout.flush()

def record_delivery_rewards(delivered):
    rewards = calculate_per_query_rewards(delivered)

    with state_lock:
        scores = _load_scores()
        for hk, w in rewards["W"].items():
            scores[hk] = scores.get(hk, 0) + w     # accumulate
        _save_scores(scores)

    # ALSO persist each delivered lead into reward_events.json so that
    # the 14 / 30 / 90-day weighting in
    #     Leadpoet.validator.reward.calculate_miner_emissions(...)
    # has the correct, up-to-date history.
    # This keeps the fast 'block-emission' table and the periodic weight
    # calculator perfectly in sync.
    for lead in delivered:
        try:
            _rec(lead)          # source, curated_by, conversion_score
        except Exception as e:
            bt.logging.warning(f"Failed to record reward event: {e}")

    # Print immediately in the validator process; miners will pick it up
    # on their next tick but we also flush now for fast feedback.
    _print_emission_table(scores)

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
        _print_emission_table(scores)

# fire the daemon
threading.Thread(target=_emission_loop, daemon=True).start()

def _read_json(path):
     try:
         with open(path, "r") as f:
             try:
                 return json.load(f)
             except json.JSONDecodeError:
                 return []
     except FileNotFoundError:
         return []