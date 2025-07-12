import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import threading

DATA_DIR = "data"
WEIGHTS_LOG = os.path.join(DATA_DIR, "miner_weights.json")
CURATION_HISTORY = os.path.join(DATA_DIR, "curation_history.json")

def ensure_weight_files():
    """Ensure weight tracking files exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in [WEIGHTS_LOG, CURATION_HISTORY]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump([], f)

def log_curation_event(leads: List[Dict], final_leads: List[Dict]):
    """Log curation event for weight calculation."""
    ensure_weight_files()
    
    # Extract source and curator information from final leads
    curation_data = []
    for lead in final_leads:
        source = lead.get('source', 'Unknown')
        curator = lead.get('curated_by', 'Unknown')
        conversion_score = lead.get('conversion_score', 0.0)
        
        curation_data.append({
            'source': source,
            'curator': curator,
            'conversion_score': conversion_score,
            'timestamp': datetime.now().isoformat()
        })
    
    # Append to curation history
    with open(CURATION_HISTORY, "r+") as f:
        try:
            history = json.load(f)
        except Exception:
            history = []
        history.extend(curation_data)
        f.seek(0)
        json.dump(history, f, indent=2)

def calculate_miner_weights() -> Dict[str, float]:
    """Calculate miner weights based on the incentive mechanism."""
    ensure_weight_files()
    
    try:
        with open(CURATION_HISTORY, "r") as f:
            history = json.load(f)
    except Exception:
        return {}
    
    if not history:
        return {}
    
    now = datetime.now()
    periods = {
        "14_days": now - timedelta(days=14),
        "30_days": now - timedelta(days=30),
        "90_days": now - timedelta(days=90)
    }
    
    # Group by miner hotkey
    miner_data = {}
    
    for event in history:
        timestamp = datetime.fromisoformat(event['timestamp'])
        source = event['source']
        curator = event['curator']
        score = event['conversion_score']
        
        # Initialize miner data if not exists
        if source not in miner_data:
            miner_data[source] = {
                'sourcing': {'14_days': [], '30_days': [], '90_days': []},
                'curating': {'14_days': [], '30_days': [], '90_days': []}
            }
        if curator not in miner_data:
            miner_data[curator] = {
                'sourcing': {'14_days': [], '30_days': [], '90_days': []},
                'curating': {'14_days': [], '30_days': [], '90_days': []}
            }
        
        # Categorize by period
        for period_name, period_start in periods.items():
            if timestamp >= period_start:
                # Add to sourcing data (source miner)
                miner_data[source]['sourcing'][period_name].append(score)
                # Add to curating data (curator miner)
                miner_data[curator]['curating'][period_name].append(score)
    
    # Calculate weights for each miner
    miner_weights = {}
    
    # Calculate sourcing scores Pₘ(t) = (Qₘ(t) * Rₘ(t)) / (∑ₘ (Qₘ(t) * Rₘ(t)))
    sourcing_scores_period = {}
    for period in ['14_days', '30_days', '90_days']:
        period_scores = {}
        total_period_score = 0
        
        for hotkey, data in miner_data.items():
            scores = data['sourcing'][period]
            if scores:
                Q_m = len(scores)  # Number of curated sourced prospects
                R_m = sum(scores) / len(scores)  # Average curated sourced prospect score
                period_scores[hotkey] = Q_m * R_m
                total_period_score += Q_m * R_m
        
        # Normalize: Pₘ(t) = (Qₘ(t) * Rₘ(t)) / (∑ₘ (Qₘ(t) * Rₘ(t)))
        if total_period_score > 0:
            for hotkey, score in period_scores.items():
                if hotkey not in sourcing_scores_period:
                    sourcing_scores_period[hotkey] = {}
                sourcing_scores_period[hotkey][period] = score / total_period_score
        else:
            for hotkey in miner_data.keys():
                if hotkey not in sourcing_scores_period:
                    sourcing_scores_period[hotkey] = {}
                sourcing_scores_period[hotkey][period] = 0
    
    # Calculate curating scores Hₘ(t) = (Iₘ(t) * Jₘ(t)) / (∑ₘ (Iₘ(t) * Jₘ(t)))
    curating_scores_period = {}
    for period in ['14_days', '30_days', '90_days']:
        period_scores = {}
        total_period_score = 0
        
        for hotkey, data in miner_data.items():
            scores = data['curating'][period]
            if scores:
                I_m = len(scores)  # Number of curated prospects
                J_m = sum(scores) / len(scores)  # Average curated prospect score
                period_scores[hotkey] = I_m * J_m
                total_period_score += I_m * J_m
        
        # Normalize: Hₘ(t) = (Iₘ(t) * Jₘ(t)) / (∑ₘ (Iₘ(t) * Jₘ(t)))
        if total_period_score > 0:
            for hotkey, score in period_scores.items():
                if hotkey not in curating_scores_period:
                    curating_scores_period[hotkey] = {}
                curating_scores_period[hotkey][period] = score / total_period_score
        else:
            for hotkey in miner_data.keys():
                if hotkey not in curating_scores_period:
                    curating_scores_period[hotkey] = {}
                curating_scores_period[hotkey][period] = 0
    
    # Calculate weighted sourcing score (Sₘ) and weighted curating score (Kₘ)
    for hotkey in miner_data.keys():
        # Weighted Sourcing Score: Sₘ = 0.55 * Pₘ(14d) + 0.25 * Pₘ(30d) + 0.20 * Pₘ(90d)
        S_m = (0.55 * sourcing_scores_period.get(hotkey, {}).get('14_days', 0) + 
               0.25 * sourcing_scores_period.get(hotkey, {}).get('30_days', 0) + 
               0.20 * sourcing_scores_period.get(hotkey, {}).get('90_days', 0))
        
        # Weighted Curating Score: Kₘ = 0.55 * Hₘ(14d) + 0.25 * Hₘ(30d) + 0.20 * Hₘ(90d)
        K_m = (0.55 * curating_scores_period.get(hotkey, {}).get('14_days', 0) + 
               0.25 * curating_scores_period.get(hotkey, {}).get('30_days', 0) + 
               0.20 * curating_scores_period.get(hotkey, {}).get('90_days', 0))
        
        # Calculate final miner weight (Wₘ)
        W_m = 0.40 * S_m + 0.60 * K_m
        miner_weights[hotkey] = W_m
    
    return miner_weights

def display_miner_weights():
    """Display current miner weights in the terminal."""
    weights = calculate_miner_weights()
    
    if not weights:
        print("========== BLOCK EMISSION ==========")
        print("No miner activity recorded yet.")
        print("====================================")
        return weights
    
    # Sort by weight (descending)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total weight for percentage
    total_weight = sum(weights.values())
    
    print("========== BLOCK EMISSION ==========")
    for hotkey, weight in sorted_weights:
        share = (weight / total_weight * 100) if total_weight > 0 else 0
        print(f"{hotkey:<20} W={weight:>6.3f}   share={share:>5.1f}%")
    print("====================================")
    
    return weights

def save_weights_to_file(weights: Dict[str, float]):
    """Save current weights to file for persistence."""
    ensure_weight_files()
    
    weight_entry = {
        'timestamp': datetime.now().isoformat(),
        'weights': weights
    }
    
    with open(WEIGHTS_LOG, "r+") as f:
        try:
            weight_history = json.load(f)
        except Exception:
            weight_history = []
        
        weight_history.append(weight_entry)
        
        # Keep only last 100 entries
        if len(weight_history) > 100:
            weight_history = weight_history[-100:]
        
        f.seek(0)
        json.dump(weight_history, f, indent=2)

def get_miner_weight(hotkey: str) -> float:
    """Get the current weight for a specific miner."""
    weights = calculate_miner_weights()
    return weights.get(hotkey, 0.0) 