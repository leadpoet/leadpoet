# Distributed Validator Consensus System - Implementation Tasks

## Overview
Transform the current single-validator API system into a distributed consensus system where:
1. All validators + miners receive API requests simultaneously
2. Validators independently rank miner leads
3. Final leads are selected using weighted consensus across all validators
4. Consensus formula: `S_lead = (S_1 * V_1) + (S_2 * V_2) + ... + (S_N * V_N)`

---

## Task 1: Implement API Request Broadcasting System
**Goal:** Enable simultaneous API request delivery to all validators and miners on the subnet.

### Subtasks:
1. **Create broadcast queue in Firestore**
   - Collection: `api_requests`
   - Document fields:
     - `request_id` (UUID)
     - `num_leads` (int)
     - `business_desc` (string)
     - `client_id` (string - for tracking)
     - `created_at` (timestamp)
     - `status` ("pending" | "processing" | "completed")
     - `timeout` (timestamp - created_at + 120 seconds)

2. **Modify API endpoint to broadcast instead of direct dendrite**
   - File: `neurons/validator.py` → `handle_api_request()` function
   - Instead of calling `self.dendrite()`, write request to `api_requests` collection
   - Return `request_id` to track the request
   - Set up polling/webhook for completion

3. **Add broadcast listener to validators**
   - File: `neurons/validator.py` → new function `process_broadcast_requests_continuous()`
   - Poll `api_requests` collection for new requests (similar to `process_curation_requests_continuous()`)
   - When new request detected:
     - Set flag to pause sourced lead queue pulling
     - Process request through existing `forward()` logic
     - Write results to new collection (see Task 2)

4. **Add broadcast listener to miners**
   - File: `neurons/miner.py` → add similar polling logic
   - When broadcast detected, stop sourcing and start curation
   - Use existing curation logic (no changes needed)
   - Send curated leads to validators via existing mechanism

---

## Task 2: Implement Validator Consensus Ranking System
**Goal:** Collect and aggregate rankings from all active validators using weighted consensus.

### Subtasks:
1. **Create validator rankings collection in Firestore**
   - Collection: `validator_rankings`
   - Document ID: `{request_id}_{validator_hotkey}`
   - Fields:
     - `request_id` (string - links to api_requests)
     - `validator_hotkey` (string)
     - `validator_uid` (int)
     - `validator_trust` (float - from metagraph)
     - `ranked_leads` (array of objects):
       ```json
       [{
         "lead": {...},  // full lead object
         "score": 0.85,  // validator's score for this lead
         "rank": 1       // validator's rank (1 = best)
       }]
       ```
     - `submitted_at` (timestamp)
     - `num_leads_ranked` (int)

2. **Modify validator forward() to write rankings**
   - File: `neurons/validator.py` → `forward()` function
   - After ranking leads (line ~542 where `top_leads` is created)
   - Write validator's rankings to `validator_rankings` collection
   - Include validator_trust from `self.metagraph.validator_trust[self.uid]`

3. **Create consensus calculation function**
   - File: `Leadpoet/validator/consensus.py` (NEW FILE)
   - Function: `async calculate_consensus_ranking(request_id: str, timeout_sec: int = 90) -> List[Dict]`
   - Logic:
     ```python
     1. Wait for all validators to submit (or timeout)
     2. Fetch all validator_rankings for request_id
     3. Build lead_id → scores mapping
     4. For each unique lead across all validators:
        S_lead = Σ(S_v * V_v) for all validators who ranked it
     5. Sort all leads by S_lead descending
     6. Return top N leads
     ```

4. **Handle timeout and missing validators**
   - If validator doesn't submit within 90 seconds, exclude from consensus
   - Require minimum 1 validator response (can be adjusted)
   - Log which validators participated vs. timed out

---

## Task 3: Integrate Consensus into API Response Flow
**Goal:** Modify the API to wait for consensus and return aggregated results.

### Subtasks:
1. **Update API endpoint to use consensus**
   - File: `neurons/validator.py` → `handle_api_request()` function
   - After broadcasting request to Firestore:
     ```python
     1. Write to api_requests collection
     2. Wait for consensus (call calculate_consensus_ranking())
     3. Receive aggregated top_leads from consensus
     4. Add c_validator_hotkey = "CONSENSUS" to each lead
     5. Return to client
     ```

2. **Add monitoring for broadcast API requests**
   - One designated validator monitors `api_requests` for completion
   - When all validators submit OR timeout reached:
     - Call `calculate_consensus_ranking()`
     - Write final results to `api_requests` document
     - Update status to "completed"

3. **Update client polling mechanism**
   - Client makes API call → receives `request_id`
   - Client polls `/api/leads/status/{request_id}` endpoint
   - Endpoint returns:
     - `status`: "pending" | "processing" | "completed"
     - `leads`: [...] (when completed)
     - `timeout_at`: timestamp

4. **Add fallback for single validator**
   - If only 1 validator active on subnet:
     - Skip consensus calculation
     - Use that validator's ranking directly
     - Avoid unnecessary delays

---

## Task 4: Add Validator Trust Tracking & Monitoring
**Goal:** Track and use validator trust values for consensus weighting.

### Subtasks:
1. **Add trust tracking to Validator class**
   - File: `neurons/validator.py` → `Validator.__init__()`
   - Add: `self.validator_trust = self.metagraph.validator_trust[self.uid].item()`
   - Refresh periodically in `sync()` method

2. **Create validator status monitoring**
   - File: `Leadpoet/validator/consensus.py`
   - Function: `get_active_validators() -> List[Dict]`
   - Returns list of validators with:
     - `uid`, `hotkey`, `trust`, `is_serving`, `last_seen`

3. **Add consensus metrics logging**
   - Log to Firestore after each consensus:
     - Number of validators participated
     - Trust distribution
     - Average response time
     - Top leads selected and their scores

4. **Add validator sync before consensus**
   - Before calculating consensus, refresh metagraph
   - Get latest validator_trust values
   - Handle validators that went offline during request

---

## Implementation Order:
1. Start with **Task 1** (Broadcasting) - Foundation for distributed system
2. Then **Task 2** (Rankings Collection) - Data layer for consensus
3. Then **Task 3** (API Integration) - Wire everything together
4. Finally **Task 4** (Trust Tracking) - Optimize consensus weighting

---

## Testing Checklist:
- [ ] Single validator + single miner (baseline)
- [ ] Two validators + single miner (consensus of 2)
- [ ] Single validator + two miners (multiple lead sources)
- [ ] Two validators + two miners (full distributed)
- [ ] Validator timeout scenario (one validator slow/offline)
- [ ] Client timeout handling (API times out gracefully)
- [ ] Trust weighting validation (high-trust validator has more influence)

---

## Files to Modify:
1. `neurons/validator.py` - Add broadcasting, consensus integration
2. `neurons/miner.py` - Add broadcast listener
3. `Leadpoet/utils/cloud_db.py` - Add broadcast queue functions
4. `Leadpoet/validator/consensus.py` - NEW FILE for consensus logic
5. `Leadpoet/validator/reward.py` - May need updates for distributed weights

---

## Notes:
- Existing miner curation logic stays unchanged ✅
- Existing validator LLM ranking stays unchanged ✅
- Only changing: distribution mechanism + consensus aggregation
- Backward compatible: Falls back to single-validator if only one active
