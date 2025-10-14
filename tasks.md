# Consensus-Based Lead Validation System Implementation

## Overview
Transform the current single-validator lead processing into a consensus-based system where each lead is evaluated by the first 3 validators who pull it from the prospect queue. Leads are only promoted to the main database when at least 2 out of 3 validators agree they are valid.

## Process Flow
1. **Miners** source leads and add to `prospect_queue`
2. **First 3 validators** who pull each lead get a copy (first-come-first-served, NOT randomly assigned)
3. Lead remains in `prospect_queue` until 3 validators have pulled it
4. Each validator independently validates and scores the lead
5. **Consensus Decision**:
   - If 2/3 say **VALID** → Lead goes to BOTH `validation_tracking` table AND main `leads` DB
   - If 2/3 say **INVALID** → Lead goes ONLY to `validation_tracking` table (NOT to main `leads` DB)
6. **Reward Eligibility** (checked at end of epoch):
   - Validators must be "in consensus" for ≥10% of leads they validated
   - "In consensus" means their vote aligned with the majority decision
   - Only validators meeting this threshold receive miner weight allocations

## Task 1: Database Schema Design & Implementation

### 1.1 Create Validation Tracking Table (72-minute rolling window)
Create a new `validation_tracking` table to store each validator's assessment. This table stores ALL validation attempts (both accepted and rejected leads) and is cleared after each epoch:
```sql
CREATE TABLE validation_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID NOT NULL,
    prospect_id UUID NOT NULL REFERENCES prospect_queue(id),
    validator_hotkey TEXT NOT NULL,
    score NUMERIC(3,2) NOT NULL,
    is_valid BOOLEAN NOT NULL,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    epoch_number INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure a validator can only validate a lead once
    UNIQUE(lead_id, validator_hotkey)
);

-- Index for efficient queries
CREATE INDEX idx_validation_tracking_lead_id ON validation_tracking(lead_id);
CREATE INDEX idx_validation_tracking_validator ON validation_tracking(validator_hotkey);
CREATE INDEX idx_validation_tracking_epoch ON validation_tracking(epoch_number);
CREATE INDEX idx_validation_tracking_timestamp ON validation_tracking(validation_timestamp);

-- Function to clear old epoch data (runs every 72 minutes)
CREATE OR REPLACE FUNCTION clear_old_validation_tracking()
RETURNS void AS $$
BEGIN
    DELETE FROM validation_tracking 
    WHERE epoch_number < (
        SELECT MAX(epoch_number) - 1 
        FROM validation_tracking
    );
END;
$$ LANGUAGE plpgsql;
```

### 1.2 Modify Prospect Queue Table
Update `prospect_queue` to track which validators have pulled each lead (first-come-first-served):
```sql
ALTER TABLE prospect_queue 
ADD COLUMN validators_pulled TEXT[] DEFAULT '{}',  -- Track which validators have pulled this lead
ADD COLUMN pull_count INTEGER DEFAULT 0,            -- Count of validators who have pulled (max 3)
ADD COLUMN consensus_status TEXT DEFAULT 'pending', -- 'pending', 'accepted', 'rejected'
ADD COLUMN consensus_timestamp TIMESTAMP WITH TIME ZONE;

-- Index for efficient queries
CREATE INDEX idx_prospect_queue_pull_count ON prospect_queue(pull_count);
CREATE INDEX idx_prospect_queue_consensus_status ON prospect_queue(consensus_status);
```

### 1.3 Create Consensus Results View
Create a materialized view for efficient consensus queries:
```sql
CREATE MATERIALIZED VIEW consensus_results AS
SELECT 
    vt.lead_id,
    vt.prospect_id,
    COUNT(*) as total_validations,
    SUM(CASE WHEN vt.is_valid THEN 1 ELSE 0 END) as valid_count,
    SUM(CASE WHEN NOT vt.is_valid THEN 1 ELSE 0 END) as invalid_count,
    ARRAY_AGG(vt.validator_hotkey) as validators,
    AVG(vt.score) as avg_score,
    MAX(vt.validation_timestamp) as last_validated
FROM validation_tracking vt
GROUP BY vt.lead_id, vt.prospect_id;

-- Refresh trigger
CREATE OR REPLACE FUNCTION refresh_consensus_results()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY consensus_results;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_consensus_after_validation
AFTER INSERT OR UPDATE ON validation_tracking
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_consensus_results();
```

## Task 2: Implement First-Come-First-Served Prospect Fetching

### 2.1 Updated Fetch Function for Validators
Modify the fetch function to implement first-come-first-served logic:
```python
# In Leadpoet/utils/cloud_db.py

def fetch_prospects_from_cloud(wallet: bt.wallet, limit: int = 100) -> List[Dict]:
    """
    Fetch prospects that this validator hasn't pulled yet.
    Prospects remain available until 3 validators have pulled them.
    First-come-first-served: first 3 validators to pull get the lead.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return []
        
        validator_hotkey = wallet.hotkey.ss58_address
        
        # Get prospects where:
        # 1. This validator hasn't pulled it yet
        # 2. Less than 3 validators have pulled it
        # 3. Still pending consensus
        result = supabase.table("prospect_queue") \
            .select("*") \
            .eq("status", "pending") \
            .lt("pull_count", 3) \
            .not_.contains("validators_pulled", [validator_hotkey]) \
            .eq("consensus_status", "pending") \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
        
        if not result.data:
            return []
        
        # Mark that this validator has pulled these prospects
        prospects = []
        for prospect in result.data:
            # Update the prospect to add this validator to the pulled list
            updated_validators = prospect.get('validators_pulled', []) + [validator_hotkey]
            new_pull_count = prospect.get('pull_count', 0) + 1
            
            update_result = supabase.table("prospect_queue") \
                .update({
                    "validators_pulled": updated_validators,
                    "pull_count": new_pull_count
                }) \
                .eq("id", prospect['id']) \
                .execute()
            
            if update_result.data:
                prospects.append(prospect['prospect'])  # Return the actual prospect data
        
        bt.logging.info(f"✅ Pulled {len(prospects)} prospects from queue (first-come-first-served)")
        return prospects
        
    except Exception as e:
        bt.logging.error(f"Failed to fetch prospects: {e}")
        return []
```

### 2.2 Atomic Pull Operation
Ensure the pull operation is atomic to prevent race conditions:
```sql
-- Function to atomically pull prospects for a validator
CREATE OR REPLACE FUNCTION pull_prospects_for_validator(
    p_validator_hotkey TEXT,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE(prospect_id UUID, prospect JSONB) AS $$
DECLARE
    v_prospect RECORD;
BEGIN
    -- Use FOR UPDATE SKIP LOCKED to prevent race conditions
    FOR v_prospect IN
        SELECT id, prospect
        FROM prospect_queue
        WHERE status = 'pending'
        AND pull_count < 3
        AND NOT (validators_pulled @> ARRAY[p_validator_hotkey])
        AND consensus_status = 'pending'
        ORDER BY created_at
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    LOOP
        -- Update the prospect with this validator
        UPDATE prospect_queue
        SET validators_pulled = array_append(validators_pulled, p_validator_hotkey),
            pull_count = pull_count + 1
        WHERE id = v_prospect.id;
        
        -- Return the prospect
        prospect_id := v_prospect.id;
        prospect := v_prospect.prospect;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

## Task 3: Build Consensus Determination Logic

### 3.1 Validator Submission Handler
Update validator to submit assessments to tracking table:
```python
# In neurons/validator.py - modify save_lead_to_cloud function

def submit_validation_assessment(
    wallet: bt.wallet, 
    prospect_id: str,
    lead_id: str,
    lead_data: Dict,
    score: float,
    is_valid: bool
) -> bool:
    """
    Submit validator's assessment to the validation tracking system.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Get current epoch
        from Leadpoet.validator.reward import _calculate_epoch_number, _get_current_block
        current_block = _get_current_block(bt.subtensor(network=NETWORK))
        epoch_number = _calculate_epoch_number(current_block)
        
        # Submit validation assessment
        validation_data = {
            "lead_id": lead_id,
            "prospect_id": prospect_id,
            "validator_hotkey": wallet.hotkey.ss58_address,
            "score": score,
            "is_valid": is_valid,
            "epoch_number": epoch_number
        }
        
        result = supabase.table("validation_tracking").insert([validation_data]).execute()
        
        if result.data:
            bt.logging.info(f"✅ Submitted validation for lead {lead_id[:8]}...")
            
            # Check if consensus is reached
            check_and_process_consensus(prospect_id, lead_id, lead_data)
            
            return True
            
    except Exception as e:
        bt.logging.error(f"Failed to submit validation assessment: {e}")
        return False

def check_and_process_consensus(prospect_id: str, lead_id: str, lead_data: Dict) -> bool:
    """
    Check if consensus has been reached for a lead and process accordingly.
    IMPORTANT: All validations go to validation_tracking table.
    Only ACCEPTED leads (2/3 valid) also go to the main leads table.
    """
    try:
        supabase = get_supabase_client()
        
        # Get all validations for this lead
        validations = supabase.table("validation_tracking") \
            .select("*") \
            .eq("lead_id", lead_id) \
            .execute()
        
        if len(validations.data) >= 3:  # All 3 validators have submitted
            valid_count = sum(1 for v in validations.data if v['is_valid'])
            invalid_count = len(validations.data) - valid_count
            
            # Consensus reached
            if valid_count >= 2:  # ACCEPTED - 2 or more validators said VALID
                # Insert into MAIN leads table (only accepted leads go here)
                lead_data['consensus_score'] = sum(v['score'] for v in validations.data) / len(validations.data)
                lead_data['validator_count'] = len(validations.data)
                lead_data['consensus_validators'] = [v['validator_hotkey'] for v in validations.data]
                lead_data['consensus_status'] = 'accepted'
                
                supabase.table("leads").insert([lead_data]).execute()
                
                # Update prospect_queue status
                supabase.table("prospect_queue") \
                    .update({
                        "consensus_status": "accepted",
                        "consensus_timestamp": datetime.now(timezone.utc).isoformat(),
                        "pull_count": len(validations.data)
                    }) \
                    .eq("id", prospect_id) \
                    .execute()
                
                bt.logging.info(f"✅ Lead {lead_id[:8]}... ACCEPTED by consensus ({valid_count}/3) - Added to main DB")
                
            else:  # REJECTED - 2 or more validators said INVALID
                # DO NOT insert into main leads table
                # Only update prospect_queue status
                supabase.table("prospect_queue") \
                    .update({
                        "consensus_status": "rejected",
                        "consensus_timestamp": datetime.now(timezone.utc).isoformat(),
                        "pull_count": len(validations.data)
                    }) \
                    .eq("id", prospect_id) \
                    .execute()
                
                bt.logging.info(f"❌ Lead {lead_id[:8]}... REJECTED by consensus ({invalid_count}/3) - NOT added to main DB")
            
            return True
            
    except Exception as e:
        bt.logging.error(f"Failed to check consensus: {e}")
        return False
```

### 3.2 Modify Validator Processing Loop
Update the validator's main processing loop:
```python
# In neurons/validator.py - modify process_sourced_leads_continuous

async def process_sourced_leads_continuous(self):
    """Process leads with consensus-based validation."""
    
    while not self.should_exit:
        try:
            # Fetch prospects assigned to this validator
            prospects_batch = fetch_assigned_prospects(self.wallet, limit=50)
            
            if not prospects_batch:
                await asyncio.sleep(5)
                continue
            
            bt.logging.info(f"📥 Processing {len(prospects_batch)} assigned prospects")
            
            for prospect_row in prospects_batch:
                prospect = prospect_row['prospect']
                prospect_id = prospect_row['id']
                
                # Generate unique lead_id
                lead_id = str(uuid.uuid4())
                
                # Perform validation (existing logic)
                is_valid, score = self.validate_lead(prospect)  # Your existing validation logic
                
                # Submit assessment to tracking system
                submit_validation_assessment(
                    wallet=self.wallet,
                    prospect_id=prospect_id,
                    lead_id=lead_id,
                    lead_data=prospect,
                    score=score,
                    is_valid=is_valid
                )
            
            await asyncio.sleep(10)
            
        except Exception as e:
            bt.logging.error(f"Error in consensus validation loop: {e}")
            await asyncio.sleep(5)
```

## Task 4: Update Reward Calculation Logic

### 4.1 Consensus Participation Tracking
Modify the eligibility check to use consensus participation:
```python
# In Leadpoet/validator/reward.py

def check_validator_consensus_eligibility(
    validator_hotkey: str, 
    epoch_start_time: str,
    service_role_key: str = None
) -> Dict:
    """
    Check if validator participated in enough consensus decisions.
    Requirement: Validator must be "in consensus" for ≥10% of leads they validated.
    "In consensus" = validator's vote aligned with the majority decision (2/3).
    """
    try:
        supabase_client = get_supabase_service_client(service_role_key)
        
        # Get all validations by this validator in the epoch
        validator_validations = supabase_client.table("validation_tracking") \
            .select("*") \
            .eq("validator_hotkey", validator_hotkey) \
            .gte("validation_timestamp", epoch_start_time) \
            .execute()
        
        if not validator_validations.data:
            return {
                "eligible": False,
                "reason": "No validations in epoch",
                "stats": {"total": 0, "consensus": 0, "percentage": 0}
            }
        
        # Get consensus results for all leads this validator validated
        lead_ids = [v['lead_id'] for v in validator_validations.data]
        consensus_results = supabase_client.table("consensus_results") \
            .select("*") \
            .in_("lead_id", lead_ids) \
            .execute()
        
        # Count how many times validator was in consensus
        consensus_count = 0
        for result in consensus_results.data:
            if result['total_validations'] >= 3:  # Consensus reached
                # Check if validator's vote aligned with consensus
                validator_vote = next(
                    (v for v in validator_validations.data 
                     if v['lead_id'] == result['lead_id']), 
                    None
                )
                
                if validator_vote:
                    consensus_accepted = result['valid_count'] >= 2
                    validator_said_valid = validator_vote['is_valid']
                    
                    # Validator is in consensus if their vote matches the outcome
                    if consensus_accepted == validator_said_valid:
                        consensus_count += 1
        
        # Calculate percentage
        total_consensus_decisions = len([r for r in consensus_results.data 
                                        if r['total_validations'] >= 3])
        
        if total_consensus_decisions == 0:
            percentage = 0
        else:
            percentage = (consensus_count / total_consensus_decisions) * 100
        
        # Eligibility threshold: 10% of consensus decisions
        eligible = percentage >= 10.0
        
        return {
            "eligible": eligible,
            "validator_hotkey": validator_hotkey,
            "consensus_participation": consensus_count,
            "total_consensus_decisions": total_consensus_decisions,
            "percentage": round(percentage, 2),
            "reason": f"Participated in {consensus_count}/{total_consensus_decisions} consensus decisions ({percentage:.1f}%)"
        }
        
    except Exception as e:
        bt.logging.error(f"Error checking consensus eligibility: {e}")
        return {
            "eligible": False,
            "reason": f"Error: {str(e)}",
            "stats": {}
        }
```

### 4.2 Update Weight Calculation
Modify weight calculation to only count consensus-accepted leads:
```python
def get_miner_sourcing_weights_from_consensus(
    epoch_start_time: str,
    total_emission: float = 100.0,
    service_role_key: str = None
) -> Dict:
    """
    Calculate miner weights based on consensus-accepted leads only.
    """
    try:
        supabase_client = get_supabase_service_client(service_role_key)
        
        # Get all consensus-accepted leads from this epoch
        accepted_leads = supabase_client.table("leads") \
            .select("miner_hotkey") \
            .gte("validated_at", epoch_start_time) \
            .execute()
        
        if not accepted_leads.data:
            return {
                "weights": {},
                "total_leads": 0,
                "message": "No consensus-accepted leads in epoch"
            }
        
        # Count leads per miner
        miner_counts = {}
        for lead in accepted_leads.data:
            miner = lead.get('miner_hotkey')
            if miner:
                miner_counts[miner] = miner_counts.get(miner, 0) + 1
        
        # Calculate proportional weights
        total_leads = sum(miner_counts.values())
        weights = {}
        
        for miner, count in miner_counts.items():
            weight = (count / total_leads) * total_emission
            weights[miner] = round(weight, 4)
        
        return {
            "weights": weights,
            "total_leads": total_leads,
            "unique_miners": len(miner_counts),
            "message": f"Calculated weights for {len(miner_counts)} miners based on {total_leads} consensus-accepted leads"
        }
        
    except Exception as e:
        bt.logging.error(f"Error calculating consensus-based weights: {e}")
        return {
            "weights": {},
            "error": str(e)
        }
```

## Implementation Order

1. **Phase 1: Database Setup**
   - Create new tables and views
   - Add columns to existing tables
   - Set up triggers and functions
   - Test with sample data

2. **Phase 2: Validator Assignment**
   - Implement validator assignment logic
   - Update prospect fetching to use assignments
   - Test round-robin/random distribution

3. **Phase 3: Consensus Processing**
   - Implement validation tracking
   - Add consensus checking logic
   - Update lead promotion logic
   - Test with 3 validators

4. **Phase 4: Reward System**
   - Update eligibility checks
   - Modify weight calculations
   - Test reward distribution
   - Verify consensus participation metrics

## Testing Strategy

1. **Unit Tests**
   - Test validator assignment algorithm
   - Test consensus determination logic
   - Test eligibility calculations

2. **Integration Tests**
   - Test full flow from prospect creation to consensus
   - Test edge cases (< 3 validators, split decisions)
   - Test reward calculation with various scenarios

3. **Load Testing**
   - Test with high volume of prospects
   - Test with many validators
   - Monitor database performance

## Rollback Plan

If issues arise, the system can be rolled back by:
1. Removing the new tables and columns
2. Reverting code changes
3. Restoring original prospect fetching logic
4. The existing leads table remains unchanged, preserving data integrity

## Monitoring & Metrics

Track these key metrics:
- Average time to consensus per lead
- Validator agreement rates
- Distribution of work across validators
- Consensus participation rates
- False positive/negative rates in consensus decisions
