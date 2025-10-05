# Leadpoet/validator/reward.py

import json, os, threading
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List
import numpy as np
from validator_models.automated_checks import validate_lead_list as auto_check_leads

# ===== STEP 5: EPOCH TIMING AND INTEGRATION =====

# Epoch configuration
EPOCH_DURATION_MINUTES = 72  # 72 minutes = 1.2 hours
EPOCH_DURATION_BLOCKS = 360  # 360 blocks
BITTENSOR_BLOCK_TIME_SECONDS = 12  # Average block time

# Epoch state tracking
_current_epoch = None
_epoch_start_block = None
_epoch_lock = threading.Lock()

# ===== STEP 5: EPOCH DETECTION FUNCTIONS =====

def _get_current_block() -> int:
    """
    Get current block number from the validator's subtensor connection.
    Falls back to estimated block if subtensor unavailable.
    """
    try:
        # Try to get from validator's existing subtensor connection
        import bittensor as bt
        
        # Use same network as validator
        subtensor = bt.subtensor(network="test")  # Use "finney" for mainnet
        current_block = subtensor.get_current_block()
        return current_block
        
    except Exception as e:
        print(f"⚠️  Cannot get current block from subtensor: {e}")
        
        # Fallback: use a reasonable fixed block for testing
        # In production, you'd want better fallback logic
        import time
        estimated_block = int(time.time() / 12)  # Rough estimation
        print(f"   Using estimated block: {estimated_block}")
        return estimated_block

def _calculate_epoch_number(block_number: int) -> int:
    """
    Calculate which epoch a given block number belongs to.
    
    Args:
        block_number: Current block number
        
    Returns:
        Epoch number (incremental counter)
    """
    return block_number // EPOCH_DURATION_BLOCKS

def _get_epoch_boundaries(epoch_number: int) -> tuple:
    """
    Get start and end blocks for a given epoch.
    
    Args:
        epoch_number: Epoch to get boundaries for
        
    Returns:
        Tuple of (start_block, end_block)
    """
    start_block = epoch_number * EPOCH_DURATION_BLOCKS
    end_block = start_block + EPOCH_DURATION_BLOCKS - 1
    return start_block, end_block

import threading
import time

# ===== BACKGROUND EPOCH MONITORING =====

_epoch_monitor_thread = None
_epoch_monitor_running = False
_epoch_monitor_lock = threading.Lock()

def _background_epoch_monitor():
    """
    Background thread that continuously monitors for epoch transitions
    and automatically clears tracking data when epochs end.
    """
    global _epoch_monitor_running
    
    print("🕐 Background epoch monitor started")
    
    while _epoch_monitor_running:
        try:
            current_block = _get_current_block()
            epoch_ended = _is_epoch_ended(current_block)
            
            if epoch_ended:
                print(f"\n🔄 AUTOMATIC EPOCH TRANSITION DETECTED (Background Monitor)")
                print(f"   Block {current_block} triggered epoch completion")
                print(f"   Clearing tracking data automatically...")
                
                try:
                    clear_epoch_tracking()
                    print(f"✅ Epoch tracking auto-cleared by background monitor")
                except Exception as e:
                    print(f"❌ Error auto-clearing epoch tracking: {e}")
            
            # Check every 30 seconds (about 2-3 blocks)
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠️  Error in background epoch monitor: {e}")
            time.sleep(60)  # Wait longer on error
    
    print("🕐 Background epoch monitor stopped")

def start_epoch_monitor():
    """
    Start the background epoch monitoring thread.
    Should be called when the validator starts up.
    """
    global _epoch_monitor_thread, _epoch_monitor_running
    
    with _epoch_monitor_lock:
        if _epoch_monitor_running:
            print("⚠️  Epoch monitor already running")
            return
        
        _epoch_monitor_running = True
        _epoch_monitor_thread = threading.Thread(
            target=_background_epoch_monitor,
            daemon=True,
            name="EpochMonitor"
        )
        _epoch_monitor_thread.start()
        # Remove the print here - it's redundant with the one in _background_epoch_monitor

def stop_epoch_monitor():
    """
    Stop the background epoch monitoring thread.
    Should be called when the validator shuts down.
    """
    global _epoch_monitor_running
    
    with _epoch_monitor_lock:
        if not _epoch_monitor_running:
            return
        
        _epoch_monitor_running = False
        print("🛑 Stopping background epoch monitor...")
        
        if _epoch_monitor_thread and _epoch_monitor_thread.is_alive():
            _epoch_monitor_thread.join(timeout=5)
        
        print("✅ Background epoch monitor stopped")

# ===== MODIFIED EPOCH DETECTION =====

def _is_epoch_ended(current_block: int) -> bool:
    """
    Check if the current 72-minute epoch has ended.
    
    Args:
        current_block: Current block number
        
    Returns:
        True if epoch has ended and we should calculate weights
    """
    global _current_epoch, _epoch_start_block
    
    with _epoch_lock:
        current_epoch = _calculate_epoch_number(current_block)
        
        # Initialize on first run
        if _current_epoch is None:
            _current_epoch = current_epoch
            _epoch_start_block = current_epoch * EPOCH_DURATION_BLOCKS
            
            # Calculate current position in epoch for cleaner display
            blocks_in_epoch = current_block - _epoch_start_block
            
            print(f"🕐 EPOCH TRACKING INITIALIZED:")
            print(f"   Current block: {current_block}")
            print(f"   Current epoch: {current_epoch}")
            print(f"   Epoch block: {blocks_in_epoch + 1}/360")
            return False
        
        # Check if we've moved to a new epoch
        if current_epoch > _current_epoch:
            print(f"🕐 EPOCH TRANSITION DETECTED:")
            print(f"   Previous epoch: {_current_epoch}")
            print(f"   New epoch: {current_epoch}")
            print(f"   Current block: {current_block}")
            
            # Update epoch tracking
            old_epoch = _current_epoch
            _current_epoch = current_epoch
            _epoch_start_block = current_epoch * EPOCH_DURATION_BLOCKS
            
            print(f"   Epoch {old_epoch} has ended - transition detected!")
            return True
        
        return False

def _get_epoch_status() -> Dict:
    """
    Get detailed information about current epoch status.
    Used for debugging and monitoring.
    
    Returns:
        Dict with epoch timing information
    """
    try:
        current_block = _get_current_block()
        current_epoch = _calculate_epoch_number(current_block)
        start_block, end_block = _get_epoch_boundaries(current_epoch)
        
        blocks_elapsed = current_block - start_block
        blocks_remaining = end_block - current_block
        progress_percentage = (blocks_elapsed / EPOCH_DURATION_BLOCKS) * 100
        
        estimated_time_remaining = blocks_remaining * BITTENSOR_BLOCK_TIME_SECONDS / 60  # minutes
        
        return {
            "current_block": current_block,
            "current_epoch": current_epoch,
            "epoch_start_block": start_block,
            "epoch_end_block": end_block,
            "blocks_elapsed": blocks_elapsed,
            "blocks_remaining": blocks_remaining,
            "progress_percentage": progress_percentage,
            "estimated_minutes_remaining": estimated_time_remaining,
            "epoch_duration_blocks": EPOCH_DURATION_BLOCKS,
            "epoch_duration_minutes": EPOCH_DURATION_MINUTES
        }
        
    except Exception as e:
        return {"error": str(e)}

def print_epoch_status():
    """
    Print current epoch status for debugging and monitoring.
    """
    status = _get_epoch_status()
    
    if "error" in status:
        print(f"❌ Error getting epoch status: {status['error']}")
        return
    
    print(f"\n⏰ EPOCH STATUS:")
    print(f"   Current epoch: {status['current_epoch']}")
    print(f"   Current block: {status['current_block']}")
    print(f"   Progress: {status['blocks_elapsed']}/{EPOCH_DURATION_BLOCKS} blocks ({status['progress_percentage']:.1f}%)")
    print(f"   Remaining: ~{status['estimated_minutes_remaining']:.1f} minutes")
    print(f"   Epoch range: {status['epoch_start_block']} → {status['epoch_end_block']}")

# ===== STEP 5: INTEGRATION WITH EXISTING WEIGHT CALCULATION =====

def calculate_weights(total_emission: float = 100.0) -> Dict:
    """
    ===== STEP 5: EPOCH-AWARE WEIGHT CALCULATION =====
    V3 incentive mechanism with 72-minute epoch timing integration.

    This function maintains the EXACT same interface for validator.py compatibility,
    but internally adds epoch timing logic to determine when to actually calculate weights.

    Args:
        total_emission: Total emission to distribute (default: 100.0)
        
    Returns:
        Dict with structure: {"K": {...}, "S": {...}, "C": {...}, "W": {...}, "E": {...}}
        (Same format as before for validator.py compatibility)
    """
    try:
        # ===== EPOCH TIMING CHECK =====
        current_block = _get_current_block()
        epoch_ended = _is_epoch_ended(current_block)
        
        if not epoch_ended:
            # Epoch still in progress - return accumulated data without clearing
            print(f"⏰ Epoch in progress (block {current_block}) - accumulating data")
            
            # Return current state without triggering full calculation
            # This allows validator.py to continue working while we accumulate epoch data
            try:
                epoch_data = get_epoch_tracking_data()
                
                # Create simplified weights based on current accumulation
                # This ensures validator.py always gets valid data structure
                curators = epoch_data.get('curators', [])
                sourcers = epoch_data.get('sourcers_of_curated', [])
                
                if curators or sourcers:
                    print(f"📊 Current epoch accumulation: {len(curators)} curations, {len(sourcers)} sourcings")
                    
                    # Create basic weights for current progress (normalized)
                    all_hotkeys = set(curators + sourcers)
                    basic_weights = {hotkey: 1.0/len(all_hotkeys) if all_hotkeys else 0.0 for hotkey in all_hotkeys}
                    basic_emissions = {hotkey: total_emission * weight for hotkey, weight in basic_weights.items()}
                    
                    return {
                        "S": {hotkey: 0.5 for hotkey in all_hotkeys},  # Even split for S
                        "C": {hotkey: 0.5 for hotkey in all_hotkeys},  # Even split for C  
                        "W": basic_weights,
                        "E": basic_emissions
                    }
                else:
                    # No data yet - return empty but valid structure
                    return {"K": {}, "S": {}, "C": {}, "W": {}, "E": {}}
                    
            except Exception as e:
                print(f"⚠️  Error getting interim epoch data: {e}")
                return {"K": {}, "S": {}, "C": {}, "W": {}, "E": {}}
        
        # ===== EPOCH HAS ENDED - FULL WEIGHT CALCULATION =====
        print(f"\n🎯 EPOCH ENDED - CALCULATING FINAL 72-MINUTE WEIGHTS")
        print(f"   Block {current_block} triggered epoch completion")
        
        # Use the full Step 4 calculation logic
        K_weights, S_weights, C_weights = _calculate_K_S_C_weights()

        if not K_weights and not S_weights and not C_weights:
            print("⚠️  No epoch data found - cannot calculate weights")
            return {"K": {}, "S": {}, "C": {}, "W": {}, "E": {}}

        # Calculate final weight: Wₘ = 0.10 × Kₘ + 0.45 × Sₘ + 0.45 × Cₘ
        W_weights = {}
        all_miners = set(K_weights.keys()) | set(S_weights.keys()) | set(C_weights.keys())

        for miner in all_miners:
            K_m = K_weights.get(miner, 0.0)  # 10% all-sourcers
            S_m = S_weights.get(miner, 0.0)  # 45% sourcers-of-curated
            C_m = C_weights.get(miner, 0.0)  # 45% curators

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

        # Validate the allocation
        total_allocated = sum(W_weights.values())
        if total_allocated > 0:
            print(f"✅ FINAL EPOCH WEIGHT ALLOCATION:")
            print(f"   Total weight allocation: {total_allocated:.4f}")
            print(f"   All-sourcers (10%): {sum(K_weights.values()):.4f}")
            print(f"   Sourcers-of-curated (45%): {sum(S_weights.values()):.4f}")
            print(f"   Curators (45%): {sum(C_weights.values()):.4f}")

        # ===== EPOCH CLEANUP =====
        try:
            clear_epoch_tracking()
            print(f"✅ Epoch tracking cleared - ready for next 72-minute period")
        except Exception as e:
            print(f"⚠️  Error clearing epoch tracking: {e}")

        return {
            "K": K_weights,    # 10% all-sourcers
            "S": S_weights,    # 45% sourcers-of-curated
            "C": C_weights,    # 45% curators
            "W": W_weights,    # Final combined weights
            "E": emissions     # Emission distribution
        }
        
    except Exception as e:
        print(f"❌ Error in epoch-aware weight calculation: {e}")
        return {"K": {}, "S": {}, "C": {}, "W": {}, "E": {}}

# ===== STEP 5: VALIDATOR.PY INTEGRATION HELPERS =====

def is_epoch_calculation_ready() -> bool:
    """
    Check if we're ready for a full epoch calculation.
    This can be called by validator.py to determine timing.
    
    Returns:
        True if epoch has ended and full calculation should occur
    """
    try:
        current_block = _get_current_block()
        return _is_epoch_ended(current_block)
    except Exception as e:
        print(f"⚠️  Error checking epoch readiness: {e}")
        return False

def force_epoch_calculation():
    """
    Force an epoch calculation regardless of timing.
    Useful for testing or manual triggers.
    """
    global _current_epoch
    print(f"🔧 FORCING EPOCH CALCULATION")
    
    with _epoch_lock:
        if _current_epoch is not None:
            _current_epoch += 1  # Trigger epoch transition
            
    return calculate_weights(100.0)

# ===== STEP 3 FIX: USE CLOUD RUN API LIKE VALIDATOR ALREADY DOES =====

# Remove the Firestore client imports and use existing cloud API pattern
import os
import requests

# Use same API URL as validator already uses
API_URL = os.getenv("LEAD_API", "https://leadpoet-api-511161415764.us-central1.run.app")

def get_all_sourced_leads_last_72_minutes() -> Dict[str, int]:
    """
    Query Cloud Run API /leads endpoint to get all miner hotkeys who sourced leads 
    in the last 72 minutes with their lead counts for proportional distribution.
    
    Returns:
        Dict[str, int]: Mapping of miner hotkey -> number of leads sourced
                        (for proportional 10% weight distribution)
    """
    try:
        # Calculate 72 minutes ago timestamp
        seventy_two_minutes_ago = datetime.utcnow() - timedelta(minutes=72)
        
        print(f"🔍 Querying Cloud Run API for leads validated after: {seventy_two_minutes_ago.isoformat()}Z")
        
        # Query the Cloud Run API
        params = {"limit": 1000}  # Get recent leads
        
        r = requests.get(f"{API_URL}/leads", params=params, timeout=30)
        r.raise_for_status()
        leads_data = r.json()
        
        # Count leads per sourcer hotkey (for proportional distribution)
        sourcer_counts = defaultdict(int)
        total_leads_found = 0
        
        for lead in leads_data:
            # Handle timestamp parsing
            validated_at = lead.get('validated_at')  
            
            if validated_at:
                try:
                    # Handle different timestamp formats
                    if isinstance(validated_at, (int, float)):
                        lead_time = datetime.utcfromtimestamp(validated_at)
                    elif isinstance(validated_at, str):
                        if validated_at.endswith('Z'):
                            lead_time = datetime.fromisoformat(validated_at[:-1])
                        else:
                            lead_time = datetime.fromisoformat(validated_at.split('+')[0])
                    else:
                        continue
                    
                    # Check if within last 72 minutes
                    if lead_time >= seventy_two_minutes_ago:
                        total_leads_found += 1
                        source_hotkey = lead.get('source')
                        if source_hotkey:
                            sourcer_counts[source_hotkey] += 1
                        
                except Exception as e:
                    print(f"⚠️  Error parsing timestamp {validated_at}: {e}")
                    continue
        
        print(f"📊 CLOUD API SOURCING QUERY RESULTS:")
        print(f"   Total leads found (last 72min): {total_leads_found}")
        print(f"   Total sourcers found: {len(sourcer_counts)}")
        
        if sourcer_counts:
            print(f"   Sourcer breakdown:")
            for hotkey, count in list(sourcer_counts.items())[:5]:  # Show first 5
                print(f"     {hotkey}: {count} leads sourced")
            if len(sourcer_counts) > 5:
                print(f"     ... and {len(sourcer_counts) - 5} more sourcers")
        
        return dict(sourcer_counts)
        
    except Exception as e:
        print(f"❌ Error querying Cloud Run API for sourced leads: {e}")
        return {}

def get_firestore_connection_status() -> Dict:
    """
    Check Firestore connection status and return diagnostic information.
    Used for debugging and health checks.
    
    Returns:
        Dict with connection status, error messages, and configuration info
    """
    status = {
        "firestore_available": False, # This function is no longer used for Firestore
        "client_initialized": False,
        "connection_healthy": False,
        "collection_accessible": False,
        "error_message": None
    }
    
    try:
        # Test client initialization
        # This function is no longer used for Firestore
        status["error_message"] = "Firestore client is no longer initialized"
        return status
        
    except Exception as e:
        status["error_message"] = str(e)
    
    return status

def test_firestore_leads_query():
    """
    Test function to verify Firestore integration is working correctly.
    Can be called manually for debugging purposes.
    """
    print(f"\n🧪 TESTING FIRESTORE INTEGRATION:")
    print(f"=" * 50)
    
    # Test connection status
    status = get_firestore_connection_status()
    print(f"Connection Status: {status}")
    
    # Test the main query function
    print(f"\nTesting get_all_sourced_leads_last_72_minutes():")
    sourcers = get_all_sourced_leads_last_72_minutes()
    print(f"Result: {len(sourcers)} unique sourcers found")
    
    if sourcers:
        print(f"Sample sourcers: {sourcers[:3]}...")  # Show first 3
    
    print(f"=" * 50)

# ===== END STEP 3 IMPLEMENTATION =====

# Global state for tracking current legitimacy miner
_current_K_miner = None
_K_miner_lock = threading.Lock()

DATA_DIR = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "reward_events.json")
_events_lock = threading.Lock()

# ===== STEP 1: LOCAL FILE TRACKING SYSTEM =====
VALIDATOR_WEIGHTS_DIR = "validator_weights"
VALIDATOR_WEIGHTS_FILE = os.path.join(VALIDATOR_WEIGHTS_DIR, "validator_weights")
_validator_weights_lock = threading.Lock()

def _init_validator_weights_system():
    """
    Initialize the validator weights tracking system:
    1. Create validator_weights folder if it doesn't exist
    2. Create validator_weights/validator_weights file with empty structure
    """
    try:
        # Create validator_weights directory
        os.makedirs(VALIDATOR_WEIGHTS_DIR, exist_ok=True)
        
        # Initialize the file with empty structure if it doesn't exist
        if not os.path.exists(VALIDATOR_WEIGHTS_FILE):
            _reset_validator_weights_file()
        
    except Exception as e:
        print(f"❌ Error initializing validator weights system: {e}")
        raise

def _reset_validator_weights_file():
    """
    Reset/clear the validator weights file with empty structure.
    Called at the beginning of each new 72-minute epoch.
    """
    empty_structure = {
        "curators": [],
        "sourcers_of_curated": []
    }
    
    with _validator_weights_lock:
        try:
            with open(VALIDATOR_WEIGHTS_FILE, "w") as f:
                json.dump(empty_structure, f, indent=2)
            print(f"🔄 Validator weights file reset for new epoch")
        except Exception as e:
            print(f"❌ Error resetting validator weights file: {e}")
            raise

def _read_validator_weights_file() -> Dict:
    """
    Safely read the current validator weights file.
    Returns empty structure if file doesn't exist or is corrupted.
    """
    with _validator_weights_lock:
        try:
            if not os.path.exists(VALIDATOR_WEIGHTS_FILE):
                print(f"⚠️  Validator weights file doesn't exist, initializing...")
                _init_validator_weights_system()
                return {"curators": [], "sourcers_of_curated": []}
            
            with open(VALIDATOR_WEIGHTS_FILE, "r") as f:
                data = json.load(f)
                
            # Validate structure
            if not isinstance(data, dict) or "curators" not in data or "sourcers_of_curated" not in data:
                print(f"⚠️  Invalid validator weights file structure, resetting...")
                _reset_validator_weights_file()
                return {"curators": [], "sourcers_of_curated": []}
                
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in validator weights file: {e}")
            _reset_validator_weights_file()
            return {"curators": [], "sourcers_of_curated": []}
        except Exception as e:
            print(f"❌ Error reading validator weights file: {e}")
            raise

def _write_validator_weights_file(data: Dict):
    """
    Safely write data to the validator weights file.
    """
    with _validator_weights_lock:
        try:
            with open(VALIDATOR_WEIGHTS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error writing validator weights file: {e}")
            raise

def record_curated_lead_event(curator_hotkey: str, sourcer_hotkey: str):
    """
    Record that a lead was curated and sent to a buyer.
    This updates the local tracking file for the current 72-minute epoch.
    
    Args:
        curator_hotkey: Hotkey of the miner who curated the lead
        sourcer_hotkey: Hotkey of the miner who originally sourced the lead
    """
    try:
        # Ensure system is initialized
        _init_validator_weights_system()
        
        # Read current data
        data = _read_validator_weights_file()
        
        # Append new hotkeys
        data["curators"].append(curator_hotkey)
        data["sourcers_of_curated"].append(sourcer_hotkey)
        
        # Write updated data
        _write_validator_weights_file(data)
        
    except Exception as e:
        print(f"❌ Error recording curated lead event: {e}")
        raise

def get_epoch_tracking_data() -> Dict:
    """
    Get the current epoch tracking data.
    Used for debugging and weight calculation at end of epoch.
    
    Returns:
        Dict with 'curators' and 'sourcers_of_curated' arrays
    """
    try:
        _init_validator_weights_system()
        return _read_validator_weights_file()
    except Exception as e:
        print(f"❌ Error getting epoch tracking data: {e}")
        return {"curators": [], "sourcers_of_curated": []}

def clear_epoch_tracking():
    """
    Clear the epoch tracking file.
    Called at the end of each 72-minute epoch after weights are calculated.
    """
    try:
        _reset_validator_weights_file()
        print(f"✅ Epoch tracking cleared - ready for new 72-minute period")
    except Exception as e:
        print(f"❌ Error clearing epoch tracking: {e}")
        raise

def print_epoch_status():
    """
    Print current epoch tracking status for debugging.
    """
    try:
        data = get_epoch_tracking_data()
        
        print(f"\n📊 CURRENT 72-MINUTE EPOCH STATUS:")
        print(f"   Curators: {len(data['curators'])} events")
        print(f"   Sourcers of curated: {len(data['sourcers_of_curated'])} events")
        
        if data['curators']:
            curator_counts = defaultdict(int)
            for hotkey in data['curators']:
                curator_counts[hotkey] += 1
            print(f"   Curator breakdown:")
            for hotkey, count in curator_counts.items():
                print(f"     {hotkey}: {count} curations")
        
        if data['sourcers_of_curated']:
            sourcer_counts = defaultdict(int)
            for hotkey in data['sourcers_of_curated']:
                sourcer_counts[hotkey] += 1
            print(f"   Sourcer breakdown:")
            for hotkey, count in sourcer_counts.items():
                print(f"     {hotkey}: {count} sourced leads curated")
        
    except Exception as e:
        print(f"❌ Error printing epoch status: {e}")

# ===== END STEP 1 IMPLEMENTATION =====


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
        intent_score   – intent matching score (was conversion_score)
    """
    # ===== FIX: Change required fields check =====
    if not {"source", "curated_by", "intent_score"}.issubset(prospect):
        # Ignore early–stage prospects
        print(f"⚠️  Prospect missing required fields. Has: {list(prospect.keys())}")
        return

    # ===== STEP 2: EPOCH TRACKING INTEGRATION =====
    # Record curator and sourcer hotkeys for the current 72-minute epoch
    try:
        curator_hotkey = prospect["curated_by"]
        sourcer_hotkey = prospect["source"]
        
        # Update the local validator_weights file with these hotkeys
        record_curated_lead_event(curator_hotkey, sourcer_hotkey)
        
    except Exception as e:
        print(f"⚠️  Error updating epoch tracking (continuing with normal processing): {e}")
        # Continue with existing logic even if epoch tracking fails
    # ===== END STEP 2 INTEGRATION =====

    _init_event_file()
    with _events_lock:
        try:
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception:
            events = []

        # FIXED: Use intent_score instead of conversion_score
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": prospect["source"],  # The miner who sourced the lead
            "curated_by": prospect["curated_by"],  # The miner who curated the lead
            "score": prospect.get("intent_score", 0.0),  # Use intent_score
        }
        events.append(event)

        with open(EVENTS_FILE, "w") as f:
            json.dump(events, f, indent=2)

        # Print reward event for debugging
        print(f"\n🎯 REWARD EVENT RECORDED:")
        print(f"   Source: {prospect['source']}")
        print(f"   Curator: {prospect['curated_by']}")
        print(f"   Score: {prospect.get('intent_score', 0.0):.3f}")
        print(f"   Email: {prospect.get('email', 'unknown')}")


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
    ===== STEP 4: NEW EPOCH-BASED WEIGHT CALCULATION =====
    Calculate weights based on 72-minute epoch data instead of latest curated list.

    New System:
    - Kₘ = 0 (legitimacy removed from new system, replaced with 10% all-sourcers)
    - Sₘ = 45% to miners who sourced leads that got curated and sent to buyer (from local file)
    - Cₘ = 45% to miners who curated leads sent to buyers (from local file)
    - Additional: 10% to all miners who sourced any lead in last 72 minutes (from Firestore)

    Returns:
        K_weights: Dict mapping miner hotkey to "all-sourcer" weights (10% category)
        S_weights: Dict mapping miner hotkey to sourcing weights (45% category)
        C_weights: Dict mapping miner hotkey to curating weights (45% category)
    """
    try:
        print(f"\n🎯 CALCULATING EPOCH-BASED WEIGHTS (72-minute system)")
        print(f"   Formula: 45% Curators + 45% Sourcers-of-Curated + 10% All-Sourcers")
        
        # ===== 45% CURATOR WEIGHTS (from local epoch tracking) =====
        epoch_data = get_epoch_tracking_data()
        curators_list = epoch_data.get('curators', [])
        sourcers_of_curated_list = epoch_data.get('sourcers_of_curated', [])
        
        print(f"📁 Local epoch data:")
        print(f"   Curators this epoch: {len(curators_list)}")
        print(f"   Sourcers of curated: {len(sourcers_of_curated_list)}")
        
        # Count occurrences for proportional distribution
        curator_counts = defaultdict(int)
        for hotkey in curators_list:
            curator_counts[hotkey] += 1
        
        sourcer_counts = defaultdict(int)  
        for hotkey in sourcers_of_curated_list:
            sourcer_counts[hotkey] += 1
        
        # ===== 10% ALL-SOURCER WEIGHTS (from Firestore last 72 minutes) =====
        all_sourcers_list = get_all_sourced_leads_last_72_minutes()
        print(f"🔥 Firestore all-sourcers (last72min): {len(all_sourcers_list)}")
        
        # Get all unique miners across all categories
        all_miners = set(curator_counts.keys()) | set(sourcer_counts.keys()) | set(all_sourcers_list.keys())
        
        if not all_miners:
            print("⚠️  No miners found in any category - returning empty weights")
            return {}, {}, {}
        
        # ===== CALCULATE C_WEIGHTS (45% - Curators) =====
        C_weights = {}
        total_curations = sum(curator_counts.values())
        if total_curations > 0:
            for miner in all_miners:
                count = curator_counts.get(miner, 0)
                C_weights[miner] = count / total_curations
        else:
            C_weights = {miner: 0.0 for miner in all_miners}
        
        # ===== CALCULATE S_WEIGHTS (45% - Sourcers of Curated) =====
        S_weights = {}
        total_sourcings = sum(sourcer_counts.values())
        if total_sourcings > 0:
            for miner in all_miners:
                count = sourcer_counts.get(miner, 0)
                S_weights[miner] = count / total_sourcings
        else:
            S_weights = {miner: 0.0 for miner in all_miners}
        
        # ===== CALCULATE K_WEIGHTS (10% - All Sourcers, repurposing K slot) =====
        # NOTE: Repurposing K_weights slot for the "10% all sourcers" category
        K_weights = {}
        total_all_sourcers = sum(all_sourcers_list.values())
        if total_all_sourcers > 0:
            for miner in all_miners:
                # Equal distribution among all sourcers (each gets 1/N share)
                K_weights[miner] = all_sourcers_list.get(miner, 0) / total_all_sourcers if miner in all_sourcers_list else 0.0
        else:
            K_weights = {miner: 0.0 for miner in all_miners}
        
        # ===== VALIDATION AND LOGGING =====
        print(f"📊 EPOCH WEIGHT CALCULATION RESULTS:")
        print(f"   Curators (45%): {len([m for m in all_miners if C_weights[m] > 0])} miners")
        print(f"   Sourcers-of-curated (45%): {len([m for m in all_miners if S_weights[m] > 0])} miners") 
        print(f"   All-sourcers (10%): {len([m for m in all_miners if K_weights[m] > 0])} miners")
        print(f"   Total unique miners: {len(all_miners)}")
        
        # Verify normalization
        c_sum = sum(C_weights.values())
        s_sum = sum(S_weights.values()) 
        k_sum = sum(K_weights.values())
        print(f"   Weight sums: C={c_sum:.4f}, S={s_sum:.4f}, K={k_sum:.4f}")
        
        return K_weights, S_weights, C_weights
        
    except Exception as e:
        print(f"❌ Error in epoch weight calculation: {e}")
        print(f"   Falling back to empty weights")
        return {}, {}, {}


# ----------  public: calculate weights (V2) --------------------------------
# REMOVED: Duplicate calculate_weights function that was incomplete

if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Print current miner weights (V2)")
    parser.add_argument("--emission", type=float, default=100.0, help="Total τ emission to distribute")
    args = parser.parse_args()

    pprint.pp(calculate_weights(args.emission))
