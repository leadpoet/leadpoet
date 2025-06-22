# Leadpoet/validator/reward.py

import numpy as np
from typing import List, Dict
from validator_models.automated_checks import validate_lead_list as auto_check_leads
import json, os, threading
from datetime import datetime, timedelta
from collections import defaultdict

async def get_rewards(self, responses: List[List[dict]]) -> np.ndarray:
    rewards = []
    for leads in responses:
        if not leads or len(leads) == 0:
            rewards.append(0.0)
            continue
        validation = await self.validate_leads(leads)
        rewards.append(validation["O_v"])
    return np.array(rewards)

async def post_approval_check(self, leads: List[dict]) -> bool:
    report = await auto_check_leads(leads)
    valid_count = sum(1 for entry in report if entry["status"] == "Valid")
    return valid_count / len(leads) >= 0.9 if leads else False

def calculate_emissions(self, total_emissions: float, validators: list) -> dict:
    Rv_total = sum(v.reputation for v in validators if v.reputation > 15)
    emissions = {}
    for v in validators:
        if v.reputation > 15:
            V_v = total_emissions * (v.reputation / Rv_total) if Rv_total > 0 else 0
            emissions[v.wallet.hotkey.ss58_address] = V_v
        else:
            emissions[v.wallet.hotkey.ss58_address] = 0
    return emissions


DATA_DIR = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "reward_events.json")
_events_lock = threading.Lock()


def _init_event_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "w") as f:
            json.dump([], f)


def print_current_rewards():
    """
    Print current reward distribution for debugging and verification.
    """
    print("\n" + "="*60)
    print("CURRENT REWARD DISTRIBUTION")
    print("="*60)
    
    try:
        rewards = calculate_miner_emissions(100.0)  # 100 Alpha total emission
        
        print(f"\nSourcing Scores (S):")
        for hotkey, score in rewards["S"].items():
            print(f"  {hotkey}: {score:.4f}")
            
        print(f"\nCurating Scores (K):")
        for hotkey, score in rewards["K"].items():
            print(f"  {hotkey}: {score:.4f}")
            
        print(f"\nCombined Weights (W = 0.4*S + 0.6*K):")
        for hotkey, weight in rewards["W"].items():
            print(f"  {hotkey}: {weight:.4f}")
            
        print(f"\nEmission Distribution (E):")
        total_emission = sum(rewards["E"].values())
        for hotkey, emission in rewards["E"].items():
            percentage = (emission / total_emission * 100) if total_emission > 0 else 0
            print(f"  {hotkey}: {emission:.2f} Alpha ({percentage:.1f}%)")
            
        print(f"\nTotal Emission: {total_emission:.2f} Alpha")
        print("="*60)
        
    except Exception as e:
        print(f"Error calculating rewards: {e}")

def record_event(prospect: Dict):
    """
    Persist an event once a prospect reaches the *final curated list*.

    prospect must contain:
        source         – miner hotkey that *sourced* the prospect
        curated_by     – miner hotkey that *curated* the prospect
        conversion_score
    """
    if not {"source", "curated_by", "conversion_score"}.issubset(prospect):
        # Ignore early–stage prospects
        return
    _init_event_file()
    with _events_lock:
        try:
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception:
            events = []
        events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "source": prospect["source"],
                "curated_by": prospect["curated_by"],
                "score": prospect.get("conversion_score", 0.0),
            }
        )
        with open(EVENTS_FILE, "w") as f:
            json.dump(events, f, indent=2)
        
        # Print reward event for debugging
        print(f"\n🎯 REWARD EVENT RECORDED:")
        print(f"   Source: {prospect['source']}")
        print(f"   Curator: {prospect['curated_by']}")
        print(f"   Score: {prospect.get('conversion_score', 0.0):.3f}")
        print(f"   Email: {prospect.get('owner_email', 'unknown')}")


# ----------  internal helpers ------------------------------------------------
def _aggregate(period_days: int):
    cut = datetime.utcnow() - timedelta(days=period_days)
    src, cur = defaultdict(float), defaultdict(float)
    _init_event_file()
    with _events_lock, open(EVENTS_FILE) as f:
        try:
            events = json.load(f)
        except Exception:
            events = []
    for ev in events:
        try:
            ts = datetime.fromisoformat(ev["timestamp"])
        except Exception:
            continue
        if ts < cut:
            continue
        src[ev["source"]] += ev["score"]
        cur[ev["curated_by"]] += ev["score"]
    return src, cur


# ----------  public: calculate miner emissions ------------------------------
def calculate_miner_emissions(total_emission: float = 100.0) -> Dict:
    """
    Returns dict with per-miner sourcing (S), curating (K),
    combined weight (W) and emission share (E).
    
    The 40/60 split is implemented here:
    - 40% of rewards go to sourcing miners
    - 60% of rewards go to curating miners
    """
    weights = [(14, 0.55), (30, 0.25), (90, 0.20)]
    S, K = defaultdict(float), defaultdict(float)

    for days, w in weights:
        src_tot, cur_tot = _aggregate(days)

        tot_src_score = sum(src_tot.values())
        if tot_src_score:
            for m, v in src_tot.items():
                S[m] += w * (v / tot_src_score)

        tot_cur_score = sum(cur_tot.values())
        if tot_cur_score:
            for m, v in cur_tot.items():
                K[m] += w * (v / tot_cur_score)

    # Calculate combined weight with 40/60 split
    W = {m: 0.40 * S.get(m, 0) + 0.60 * K.get(m, 0) for m in set(S) | set(K)}
    total_W = sum(W.values())
    emissions = {m: (total_emission * w / total_W) if total_W else 0 for m, w in W.items()}

    return {"S": dict(S), "K": dict(K), "W": W, "E": emissions}


if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Print current miner emissions")
    parser.add_argument("--emission", type=float, default=100.0, help="Total τ emission to distribute")
    args = parser.parse_args()

    pprint.pp(calculate_miner_emissions(args.emission))