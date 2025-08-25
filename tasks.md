# Validator Weight Publishing - Implementation Tasks

## Overview
Implement automatic weight publishing for validators based on scoring logic in `score.py`. This integrates miner weight calculation and on-chain publishing into the validator workflow.

## Task Breakdown

### Task 1: Refactor score.py for Programmatic Access
**File:** `Leadpoet/validator/score.py`

**Current State:** CLI-only script that outputs to terminal
**Required:** Expose callable function for validator integration

#### Subtasks:
- [ ] Extract main calculation logic into `calculate_weights()` function
- [ ] Return structured data instead of printing to terminal
- [ ] Maintain existing logging functionality for transparency
- [ ] Handle Firestore timestamp type mismatches (current warnings)

#### Expected Return Format:
```python
def calculate_weights():
    """Calculate miner weights based on curated vs sourced performance"""
    # ... existing calculation logic ...
    return {
        'weights_dict': {
            '5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth': 1.0000,
            # ... other miners
        },
        'stats': {
            'total_miners': 1,
            'total_share': 1.000000,
            'min_share': 1.000000,
            'max_share': 1.000000,
            'mean_share': 1.000000,
            'zero_share_miners': 0
        }
    }
```

#### Reference Score.py Output to Maintain:
```
INFO:__main__:Starting miner share calculation
INFO:__main__:Configuration loaded for project: neat-coast-467216-a6
INFO:__main__:Will calculate shares for last 30 days
INFO:__main__:Weights: Curated=0.6, Sourced=0.4
INFO:__main__:Final shares calculated for 1 miners
INFO:__main__:Top 5 miners by share:
INFO:__main__:  5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth: 1.0000 (curated: 1, sourced: 1)
```

---

### Task 2: Create Weight Publishing Function
**File:** `validator.py`

**Implementation:** Add bittensor-native weight publishing

#### Subtasks:
- [ ] Import required bittensor modules and score calculation function
- [ ] Create `publish_weights()` async function
- [ ] Handle wallet/hotkey configuration for validator
- [ ] Configure subtensor connection (netuid=401, network='test')
- [ ] Transform score.py output to bittensor format
- [ ] Add proper error handling and logging

#### Required Implementation:
```python
import asyncio
import bittensor as bt
from Leadpoet.validator.score import calculate_weights

async def publish_weights():
    """Publish updated miner weights to bittensor chain"""
    try:
        # Initialize bittensor components
        wallet = bt.wallet(name="validator", hotkey="default")
        subtensor = await bt.subtensor('test')  # adjust network param
        
        # Get weights from score.py
        weight_data = calculate_weights()
        weights_dict = weight_data['weights_dict']
        
        # Convert to bittensor format
        uids = list(weights_dict.keys())
        weights = list(weights_dict.values())
        
        # Publish to chain
        await subtensor.set_weights(
            wallet=wallet,
            netuid=401,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False
        )
        
        # Terminal logging (match score.py style)
        print("✅ Published new weights:", dict(zip(uids, weights)))
        print(f"✓ Found {len(uids)} miners")
        print(f"✓ Total share: {sum(weights):.6f}")
        
        await subtensor.close()
        
    except Exception as e:
        print(f"⚠️ Error publishing weights: {e}")
        raise
```

---

### Task 3: Integrate Into Curated Lead Flow
**File:** `validator.py`

**Trigger Point:** After curated lead is processed and saved

#### Subtasks:
- [ ] Identify curated lead completion point in validator workflow
- [ ] Add weight publishing call after DB save + API response
- [ ] Ensure async execution doesn't block response to buyer
- [ ] Add flow control to prevent sourced lead processing until weights published

#### Integration Pattern:
```python
async def process_curated_lead(lead_data):
    """Process curated lead and trigger weight update"""
    try:
        # 1. Store curated lead in DB
        await store_lead_in_db(lead_data)
        
        # 2. Return response to buyer (API caller)
        response = await return_to_buyer(lead_data)
        
        # 3. THEN publish weights
        print("📊 Triggering weight update after curated lead...")
        await publish_weights()
        
        # 4. Only now resume sourced lead processing
        print("✅ Weights updated, resuming sourced lead review")
        
        return response
        
    except Exception as e:
        print(f"❌ Error in curated lead flow: {e}")
        raise
```

---

### Task 4: Add Scheduled Weight Update Loop
**File:** `validator.py`

**Purpose:** Periodic weight publishing independent of curated leads

#### Subtasks:
- [ ] Create `weights_update_loop()` background task
- [ ] Add configurable update interval (default: 1 hour)
- [ ] Implement retry logic with backoff on failures
- [ ] Run alongside main validator loop

#### Required Implementation:
```python
async def weights_update_loop():
    """Background task to periodically publish weights"""
    while True:
        try:
            print("🔄 Scheduled weight update starting...")
            await publish_weights()
            print("✅ Scheduled weight update completed")
            
            # Wait before next update (configurable)
            await asyncio.sleep(3600)  # 1 hour
            
        except Exception as e:
            print(f"⚠️ Error in scheduled weight update: {e}")
            print("🔄 Retrying in 10 minutes...")
            await asyncio.sleep(600)  # 10 minutes retry
```

#### Integration with Main Loop:
```python
async def main_validator_loop():
    """Main validator with background weight updates"""
    # Start background weight update task
    weight_task = asyncio.create_task(weights_update_loop())
    
    # Main validator logic
    while True:
        # ... existing validator logic ...
        pass
```

---

### Task 5: Terminal Logging & Transparency
**Location:** All weight publishing functions

#### Requirements:
- [ ] Match score.py logging format and style
- [ ] Print weight updates after successful publishing
- [ ] Include miner statistics (similar to score.py output)
- [ ] Add clear success/failure indicators

#### Expected Terminal Output:
```
📊 Triggering weight update after curated lead...
INFO:__main__:Starting miner share calculation
INFO:__main__:Final shares calculated for 1 miners
✅ Published new weights:
 1. 5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth 1.0000 (100.00%)
✓ Found 1 miners
✓ Total share: 1.000000
✅ Weights updated, resuming sourced lead review
```

---

### Task 6: Error Handling & Recovery
**Files:** `validator.py`, `score.py`

#### Subtasks:
- [ ] Handle Firestore connection failures
- [ ] Handle bittensor subtensor connection issues
- [ ] Implement weight publishing retry logic
- [ ] Graceful degradation if weight publishing fails
- [ ] Prevent validator from blocking on weight failures

#### Error Scenarios to Handle:
```python
# Firestore timestamp warnings (from score.py output)
WARNING:__main__:Unexpected created_at type: <class 'float'>

# Bittensor connection failures
# Weight publishing failures
# Wallet/hotkey access issues
```

---

## Configuration Requirements

### Bittensor Network Settings
- **Network:** `test` (configurable)
- **Netuid:** `401`
- **Wallet:** `name="validator", hotkey="default"`

### Firebase/Firestore
- **Project:** `neat-coast-467216-a6`
- **Collectis