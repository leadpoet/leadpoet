# BRD: Validator Weight Publishing Integration

## Overview

Validators need to automatically update miner weights on-chain based on scoring logic defined in `score.py`. Currently, `score.py` (located in `Leadpoet/validator/score.py`) calculates and outputs miner shares using curated vs sourced lead performance. However, this calculation is not yet integrated into the validator loop.

This document defines the required changes so that:

- Every time a curated lead is processed and saved, the validator fetches new weights from `score.py`.
- The validator publishes the updated weights on-chain.
- Only after publishing weights, the validator resumes reviewing sourced leads.
- Published weights should be printed in the validator terminal for transparency.
- In addition to curated-lead triggers, a scheduled update loop must run to periodically publish weights.

## Current State

- `score.py` successfully calculates miner weights and normalizes shares.
- Example run shows shares being calculated correctly and normalized to 1.0.
- `validator.py` currently does not call `score.py` or publish updated weights.

## Reference: Actual score.py Output

Below is the real-world output from `score.py` execution, which serves as a baseline for expected functionality and logging:

```
(venv) (base) pranavramesh@Pranavs-MacBook-Air-2 score.py % python score.py
Firestore MinerUID Share Calculation
==================================================
✓ Firebase Admin SDK available
✓ Configuration loaded for project: neat-coast-467216-a6
✓ Will calculate shares for last 30 days
✓ Weights: Curated=0.6, Sourced=0.4
Starting calculation...
------------------------------
INFO:__main__:Starting miner share calculation
INFO:__main__:Configuration loaded for project: neat-coast-467216-a6
INFO:__main__:Will calculate shares for last 30 days
INFO:__main__:Weights: Curated=0.6, Sourced=0.4
INFO:__main__:Date range: 2025-07-25 06:01:59.470690+00:00 to 2025-08-24 06:01:59.470690+00:00
INFO:__main__:Firebase initialized successfully for project: neat-coast-467216-a6
INFO:__main__:Querying Firestore for curation results...
INFO:__main__:Querying curation_results from 2025-07-25 06:01:59.470690+00:00 to 2025-08-24 06:01:59.470690+00:00
WARNING:__main__:Unexpected created_at type: <class 'float'>
WARNING:__main__:Unexpected created_at type: <class 'float'>
WARNING:__main__:Unexpected created_at type: <class 'float'>
WARNING:__main__:Unexpected created_at type: <class 'float'>
INFO:__main__:Retrieved 1 documents (total: 1)
INFO:__main__:Total documents retrieved: 1
INFO:__main__:Retrieved 1 curation results
INFO:__main__:Aggregating miner data...
INFO:__main__:Found 1 unique miners
INFO:__main__:Calculating weighted scores...
INFO:__main__:Total curated: 1
INFO:__main__:Total sourced: 1
INFO:__main__:Total weighted score: 1.000000
INFO:__main__:Normalizing shares...
INFO:__main__:Calculation validation passed: 1 miners, total share: 1.000000
INFO:__main__:Calculation completed in 0.97 seconds
INFO:__main__:Final shares calculated for 1 miners
INFO:__main__:Top 5 miners by share:
INFO:__main__:  5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth: 1.0000 (curated: 1, sourced: 1)
✓ Calculation completed successfully!
✓ Found 1 miners
✓ Total share: 1.000000
Top 10 miners by share:
------------------------------
 1. 5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth 1.0000 (100.00%)
Statistics:
- Min share: 1.000000
- Max share: 1.000000
- Mean share: 1.000000
- Zero share miners: 0
Recommended Firestore indexes:
- Collection: curation_results
- Fields: timestamp ASC, curated_by ASC
- Fields: timestamp ASC, source ASC
- Collection: miner_shares_results
- Fields: calculation_timestamp DESC
```

## Requirements

### Functional Requirements

#### Trigger Point (Curated Lead):

After each curated lead is processed:
1. Lead stored in DB.
2. Lead returned to buyer (API caller).
3. **THEN** weights must be recalculated and published.

#### Scheduled Update Loop:

- Add a background task that periodically publishes weights (similar to `weights_update_loop` in score-vision).
- If weights fail to publish multiple times, retry with backoff.

#### Weight Calculation:

- Call `score.py` (or its underlying function) to calculate miner weights.
- Extract normalized miner weights (share percentages).

#### Weight Publishing:

- Validator must publish new weights on-chain via `bittensor.subtensor.set_weights`.
- Publishing must include:
  - Correct wallet (validator/hotkey).
  - Correct subnet UID (netuid).
  - Correct miner UIDs and weights.
- Must be asynchronous and not block other validator tasks.

#### Terminal Logging:

- After weights are published, print updated weights in validator terminal (similar to `score.py` output).

#### Execution Flow Enforcement:

- Validator should not process additional sourced leads until weights are published.

### Non-Functional Requirements

- Publishing should be idempotent (if nothing changes, chain state remains stable).
- Should handle Firestore timestamp type mismatches (warnings observed in logs).
- Must recover gracefully from publishing failures (retry logic).

## Implementation Plan

### 1. Refactor score.py:

Expose a callable function (e.g., `calculate_weights()`) instead of CLI-only output.

### 2. Update validator.py:

- Import and call `calculate_weights()` after curated lead flow completes.
- Transform weight results into format required by `subtensor.set_weights`.

```python
import asyncio
import bittensor as bt
from Leadpoet.validator.score import calculate_weights

async def publish_weights():
    wallet = bt.wallet(name="validator", hotkey="default")
    subtensor = await bt.subtensor('test')  # adjust network param
    weights_dict = calculate_weights()
    
    uids = list(weights_dict.keys())
    weights = list(weights_dict.values())
    
    await subtensor.set_weights(
        wallet=wallet,
        netuid=401,
        uids=uids,
        weights=weights,
        wait_for_inclusion=False
    )
    print("✅ Published new weights:", dict(zip(uids, weights)))
    await subtensor.close()
```

### 3. Integrate Into Curated Lead Flow:

After DB insert + response return, call:

```python
asyncio.run(publish_weights())
```

Ensure validator awaits completion before processing new sourced leads.

### 4. Add Scheduled Update Loop:

Run alongside validator main loop:

```python
async def weights_update_loop():
    while True:
        try:
            await publish_weights()
            await asyncio.sleep(3600)  # update every hour (configurable)
        except Exception as e:
            print("⚠️ Error publishing weights:", e)
            await asyncio.sleep(600)  # retry in 10 minutes
```

## Example Workflow

1. Buyer submits lead → Validator curates it.
2. Validator stores curated lead → Returns response to buyer.
3. `score.py` is called → Calculates new miner shares.
4. Validator publishes weights to chain via `subtensor.set_weights`.
5. Terminal prints:
   ```
   ✅ Published new weights:
   { minerUID1: 0.6, minerUID2: 0.4 }
   ```
6. Validator resumes reviewing sourced leads.
7. Separately, the scheduled update loop publishes weights periodically.