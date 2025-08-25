# Leadpoet/validator/reward.py

import json, os, threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
import numpy as np
from validator_models.automated_checks import validate_lead_list as auto_check_leads

# Global state for tracking current legitimacy miner
_current_K_miner = None
_K_miner_lock = threading.Lock()

DATA_DIR = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "reward_events.json")
_events_lock = threading.Lock()


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


def _init_event_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "w") as f:
            json.dump([], f)


def set_current_K_miner(miner_hotkey: str):
    """
    Set the current miner who successfully passed legitimacy audit.
    This miner will receive Kₘ = 1, all others receive Kₘ = 0.
    """
    global _current_K_miner
    with _K_miner_lock:
        old_miner = _current_K_miner
        _current_K_miner = miner_hotkey
        print(f"\n🔑 K MINER UPDATED: {old_miner} → {miner_hotkey}")
        print(f"   Miner {miner_hotkey} now has Kₘ = 1, all others have Kₘ = 0")


def get_current_K_miner() -> str:
    """Get the current miner with Kₘ = 1."""
    with _K_miner_lock:
        return _current_K_miner


def print_current_rewards():
    """
    Print current reward distribution for debugging and verification.
    """
    print("\n" + "="*60)
    print("CURRENT REWARD DISTRIBUTION (V2)")
    print("="*60)

    try:
        rewards = calculate_weights(100.0)  # 100 Alpha total emission

        print(f"\nLegitimacy Scores (K):")
        for hotkey, score in rewards["K"].items():
            print(f"  {hotkey}: {score:.4f}")

        print(f"\nSourcing Scores (S):")
        for hotkey, score in rewards["S"].items():
            print(f"  {hotkey}: {score:.4f}")

        print(f"\nCurating Scores (C):")
        for hotkey, score in rewards["C"].items():
            print(f"  {hotkey}: {score:.4f}")

        print(f"\nCombined Weights (W = 0.10×K + 0.45×S + 0.45×C):")
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

        # FIXED: Ensure source and curated_by are correctly recorded
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": prospect["source"],  # The miner who sourced the lead
            "curated_by": prospect["curated_by"],  # The miner who curated the lead
            "score": prospect.get("conversion_score", 0.0),
        }
        events.append(event)

        with open(EVENTS_FILE, "w") as f:
            json.dump(events, f, indent=2)

        # Print reward event for debugging
        print(f"\n🎯 REWARD EVENT RECORDED:")
        print(f"   Source: {prospect['source']}")
        print(f"   Curator: {prospect['curated_by']}")
        print(f"   Score: {prospect.get('conversion_score', 0.0):.3f}")
        print(f"   Email: {prospect.get('owner_email', 'unknown')}")


# ----------  V2 internal helpers ------------------------------------------------
def _get_latest_curated_events() -> List[Dict]:
    """
    Get the latest curated events from the events file.
    Returns empty list if no events exist.
    """
    _init_event_file()
    with _events_lock:
        try:
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception:
            events = []

    if not events:
        return []

    # Return all events (latest curated list)
    # In a real implementation, you might want to filter by a specific curation round
    return events


def _calculate_K_S_C_weights() -> tuple:
    """
    Calculate Kₘ, Sₘ, and Cₘ weights based on latest curated list only.

    Returns:
        K_weights: Dict mapping miner hotkey to Kₘ (0 or 1)
        S_weights: Dict mapping miner hotkey to Sₘ (normalized sourcing scores)
        C_weights: Dict mapping miner hotkey to Cₘ (normalized curating scores)
    """
    events = _get_latest_curated_events()

    if not events:
        print("⚠️  No curated events found for weight calculation")
        return {}, {}, {}

    # Initialize weight dictionaries
    K_weights = {}
    S_counts = defaultdict(int)  # Tₘ = count of miner's sourced prospects
    C_counts = defaultdict(int)  # Dₘ = count of miner's curated prospects

    # Count sourced and curated prospects for each miner
    for event in events:
        source = event["source"]
        curator = event["curated_by"]

        S_counts[source] += 1
        C_counts[curator] += 1

    # Calculate Kₘ (legitimacy): 1 for current miner, 0 for others
    current_K_miner = get_current_K_miner()
    all_miners = set(S_counts.keys()) | set(C_counts.keys())

    for miner in all_miners:
        K_weights[miner] = 1.0 if miner == current_K_miner else 0.0

    # Calculate Sₘ (sourcing): Tₘ / ΣTₘ
    total_sourced = sum(S_counts.values())
    S_weights = {}
    if total_sourced > 0:
        for miner, count in S_counts.items():
            S_weights[miner] = count / total_sourced
    else:
        S_weights = {miner: 0.0 for miner in all_miners}

    # Calculate Cₘ (curating): Dₘ / ΣDₘ
    total_curated = sum(C_counts.values())
    C_weights = {}
    if total_curated > 0:
        for miner, count in C_counts.items():
            C_weights[miner] = count / total_curated
    else:
        C_weights = {miner: 0.0 for miner in all_miners}

    # Ensure all miners have entries in all weight dictionaries
    for miner in all_miners:
        if miner not in S_weights:
            S_weights[miner] = 0.0
        if miner not in C_weights:
            C_weights[miner] = 0.0

    return K_weights, S_weights, C_weights


# ----------  public: calculate weights (V2) --------------------------------
def calculate_weights(total_emission: float = 100.0) -> Dict:
    """
    V2 incentive mechanism: Calculate miner weights based on latest curated list only.

    Formula: Wₘ = 0.10 × Kₘ + 0.45 × Sₘ + 0.45 × Cₘ

    Where:
    - Kₘ = 1 for last miner with valid lead, 0 for others (10% of rewards)
    - Sₘ = Tₘ / ΣTₘ where Tₘ = count of miner's sourced prospects (45% of rewards)
    - Cₘ = Dₘ / ΣDₘ where Dₘ = count of miner's curated prospects (45% of rewards)

    Returns:
        Dict with structure: {"K": {...}, "S": {...}, "C": {...}, "W": {...}, "E": {...}}
    """
    # Validate that we have a latest curated list
    events = _get_latest_curated_events()
    if not events:
        print("⚠️  No curated events found - cannot calculate weights")
        return {"K": {}, "S": {}, "C": {}, "W": {}, "E": {}}

    # Calculate Kₘ, Sₘ, and Cₘ weights
    K_weights, S_weights, C_weights = _calculate_K_S_C_weights()

    # Calculate final weight: Wₘ = 0.10 × Kₘ + 0.45 × Sₘ + 0.45 × Cₘ
    W_weights = {}
    all_miners = set(K_weights.keys()) | set(S_weights.keys()) | set(C_weights.keys())

    for miner in all_miners:
        K_m = K_weights.get(miner, 0.0)
        S_m = S_weights.get(miner, 0.0)
        C_m = C_weights.get(miner, 0.0)

        W_m = 0.10 * K_m + 0.45 * S_m + 0.45 * C_m
        W_weights[miner] = W_m

    # Calculate emissions based on weights
    total_weight = sum(W_weights.values())
    emissions = {}

    if total_weight > 0:
        for miner, weight in W_weights.items():
            emissions[miner] = total_emission * (weight / total_weight)
    else:
        emissions = {miner: 0.0 for miner in all_miners}

    # Validate the allocation adds up to 100%
    total_allocated = sum(W_weights.values())
    if total_allocated > 0:
        print(f"✅ Total weight allocation: {total_allocated:.4f} (should be 1.0)")
        print(f"   K allocation: {sum(K_weights.values()):.4f}")
        print(f"   S allocation: {sum(S_weights.values()):.4f}")
        print(f"   C allocation: {sum(C_weights.values()):.4f}")

    return {
        "K": K_weights,
        "S": S_weights,
        "C": C_weights,
        "W": W_weights,
        "E": emissions
    }


if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Print current miner weights (V2)")
    parser.add_argument("--emission", type=float, default=100.0, help="Total τ emission to distribute")
    args = parser.parse_args()

    pprint.pp(calculate_weights(args.emission))
