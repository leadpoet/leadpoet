-- TEE System Database Schema Migration
-- =====================================
-- 
-- This migration creates the tables required for the TEE-based weight submission
-- and auditing system as defined in business_files/tasks8.md.
--
-- Tables created:
-- 1. published_weight_bundles - Primary validator weight submissions
-- 2. validator_attestations - Validator TEE attestations (one per epoch per validator)
-- 3. transparency_log - Gateway-signed events (hash-chained audit trail)
-- 4. epoch_audit_logs - Audit results per epoch
--
-- Run this in Supabase SQL Editor or via psql.
--
-- IMPORTANT: This uses CANONICAL field names as defined in the spec.
-- DO NOT use aliases (e.g., use 'enclave_pubkey' NOT 'gateway_pubkey').

-- ============================================================================
-- TABLE 1: published_weight_bundles
-- ============================================================================
-- 
-- Stores weight bundles submitted by validators. One row per epoch per validator.
-- Weights are stored as arrays (sparse representation - only non-zero weights).
--
-- SECURITY: Auditors must NOT trust this table alone. They must:
-- 1. Verify weight_submission_event_hash in transparency_log
-- 2. Recompute weights_hash from uids[] and weights_u16[]
-- 3. Verify validator_signature over the recomputed hash

CREATE TABLE IF NOT EXISTS published_weight_bundles (
    id BIGSERIAL PRIMARY KEY,
    
    -- Epoch identification
    epoch_id INTEGER NOT NULL,
    block INTEGER NOT NULL,           -- Block when weights were computed
    netuid INTEGER NOT NULL,
    
    -- Weights as arrays (sparse representation - only non-zero weights)
    uids INTEGER[] NOT NULL,          -- UIDs with non-zero weights, sorted ascending
    weights_u16 INTEGER[] NOT NULL,   -- Corresponding weights [1-65535], same length as uids
    
    -- Array integrity constraints (prevent silent corruption)
    -- Note: PostgreSQL doesn't support CHECK on array element values
    -- Validation for: uids strictly increasing, weights_u16 in [1,65535] must be at API level
    CONSTRAINT uids_weights_same_length CHECK (cardinality(uids) = cardinality(weights_u16)),
    CONSTRAINT uids_not_empty CHECK (cardinality(uids) > 0),
    
    -- Cryptographic verification data
    weights_hash TEXT NOT NULL,                   -- bundle_weights_hash(netuid, epoch_id, block, weights)
    validator_hotkey TEXT NOT NULL,               -- SS58 address of the submitting validator
    validator_enclave_pubkey TEXT NOT NULL,       -- Ed25519 public key of validator TEE (hex)
    validator_signature TEXT NOT NULL,            -- Ed25519 signature over weights_hash (hex)
    validator_attestation_b64 TEXT NOT NULL,      -- Base64-encoded AWS Nitro attestation document
    validator_code_hash TEXT NOT NULL,            -- SHA256 of validator code (informational, trust comes from PCR0)
    
    -- Chain snapshot for equivocation detection
    -- Captured at weight submission window (~block 345 in epoch)
    -- Enables auditors to verify WITHOUT querying live chain
    chain_snapshot_block INTEGER,                 -- Block number when snapshot was captured
    chain_snapshot_compare_hash TEXT,             -- compare_weights_hash of on-chain weights at snapshot
    
    -- Link to signed gateway event (for auditor verification)
    -- Auditors MUST fetch this event and verify its signature
    weight_submission_event_hash TEXT,            -- event_hash from signed WEIGHT_SUBMISSION log entry
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Uniqueness: One submission per (netuid, epoch, validator)
    -- Prevents both duplicate submissions and cross-subnet collisions
    UNIQUE(netuid, epoch_id, validator_hotkey)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_pwb_epoch ON published_weight_bundles(epoch_id);
CREATE INDEX IF NOT EXISTS idx_pwb_netuid_epoch ON published_weight_bundles(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_pwb_validator ON published_weight_bundles(validator_hotkey);
CREATE INDEX IF NOT EXISTS idx_pwb_event_hash ON published_weight_bundles(weight_submission_event_hash);

-- RLS: Public reads, service_role writes
ALTER TABLE published_weight_bundles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anon can read weight bundles" ON published_weight_bundles 
    FOR SELECT TO anon USING (true);
CREATE POLICY "Service role can insert weight bundles" ON published_weight_bundles 
    FOR INSERT TO service_role WITH CHECK (true);
CREATE POLICY "Service role can update weight bundles" ON published_weight_bundles 
    FOR UPDATE TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role can delete weight bundles" ON published_weight_bundles 
    FOR DELETE TO service_role USING (true);


-- ============================================================================
-- TABLE 2: validator_attestations
-- ============================================================================
-- 
-- Stores validator TEE attestations. One per epoch per validator.
-- Validators MUST regenerate attestations each epoch (epoch_id is in user_data).
-- Gateways/auditors MUST reject stale attestations.

CREATE TABLE IF NOT EXISTS validator_attestations (
    id BIGSERIAL PRIMARY KEY,
    
    -- Identification
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    validator_hotkey TEXT NOT NULL,
    
    -- Attestation data (CANONICAL field names)
    validator_enclave_pubkey TEXT NOT NULL,       -- Ed25519 public key from validator TEE (hex)
    validator_attestation_b64 TEXT NOT NULL,      -- Base64-encoded AWS Nitro attestation (CBOR COSE_Sign1)
    validator_code_hash TEXT NOT NULL,            -- PCR0 measurement (root of trust)
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Uniqueness: One attestation per (netuid, epoch, validator)
    UNIQUE(netuid, epoch_id, validator_hotkey)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_va_epoch ON validator_attestations(epoch_id);
CREATE INDEX IF NOT EXISTS idx_va_netuid_epoch ON validator_attestations(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_va_validator ON validator_attestations(validator_hotkey);

-- RLS: Public reads, service_role writes
ALTER TABLE validator_attestations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anon can read attestations" ON validator_attestations 
    FOR SELECT TO anon USING (true);
CREATE POLICY "Service role can insert attestations" ON validator_attestations 
    FOR INSERT TO service_role WITH CHECK (true);
CREATE POLICY "Service role can update attestations" ON validator_attestations 
    FOR UPDATE TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role can delete attestations" ON validator_attestations 
    FOR DELETE TO service_role USING (true);


-- ============================================================================
-- TABLE 3: transparency_log
-- ============================================================================
-- 
-- Gateway-signed events forming a hash-chained audit trail.
-- This is the PRIMARY audit trail - every significant event is logged here.
--
-- Log Entry Structure (stored in payload JSONB):
-- {
--     "signed_event": {
--         "event_type": "...",
--         "timestamp": "2024-01-01T00:00:00Z",
--         "boot_id": "uuid",
--         "monotonic_seq": 12345,
--         "prev_event_hash": "abc123...",
--         "payload": { ... }
--     },
--     "event_hash": "sha256(canonical_json(signed_event))",
--     "enclave_pubkey": "hex",
--     "enclave_signature": "hex"
-- }
--
-- SECURITY: Auditors verify by:
-- 1. Recomputing event_hash from signed_event
-- 2. Verifying enclave_signature using attested enclave_pubkey
-- 3. Following prev_event_hash to reconstruct hash-chain

CREATE TABLE IF NOT EXISTS transparency_log (
    id BIGSERIAL PRIMARY KEY,
    
    -- Event identification
    event_type TEXT NOT NULL,                     -- WEIGHT_SUBMISSION, ENCLAVE_RESTART, etc.
    
    -- The full log_entry from sign_event() stored as JSONB
    -- Structure: {signed_event, event_hash, enclave_pubkey, enclave_signature}
    payload JSONB NOT NULL,
    
    -- Denormalized for efficient queries (extracted from payload)
    event_hash TEXT NOT NULL UNIQUE,              -- SHA256 of signed_event (primary lookup key)
    enclave_pubkey TEXT NOT NULL,                 -- Gateway pubkey that signed this event
    
    -- Hash-chain reconstruction fields (denormalized from payload)
    -- Auditors reconstruct chain by following prev_event_hash, NOT by created_at
    boot_id TEXT,                                 -- Enclave boot session ID
    monotonic_seq BIGINT,                         -- Monotonically increasing within boot session
    prev_event_hash TEXT,                         -- Previous event hash (chain link)
    
    -- Epoch-based query fields (extracted from payload for relevant events)
    netuid INTEGER,                               -- Subnet ID (for filtering by subnet)
    epoch_id INTEGER,                             -- Epoch ID (for filtering by epoch)
    
    -- Actor identification (for filtering by hotkey)
    actor_hotkey TEXT,                            -- Validator/miner hotkey if applicable
    
    -- Legacy/compatibility fields (may be deprecated)
    email_hash TEXT,                              -- For lead-related events
    linkedin_combo_hash TEXT,                     -- For lead duplicate detection
    build_id TEXT,                                -- Gateway build identifier
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Arweave checkpointing
    arweave_tx_id TEXT                            -- Transaction ID once checkpointed to Arweave
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_tl_event_type ON transparency_log(event_type);
CREATE INDEX IF NOT EXISTS idx_tl_created_at ON transparency_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tl_pubkey ON transparency_log(enclave_pubkey);
CREATE INDEX IF NOT EXISTS idx_tl_actor ON transparency_log(actor_hotkey) WHERE actor_hotkey IS NOT NULL;

-- Indexes for hash-chain reconstruction (auditors)
CREATE INDEX IF NOT EXISTS idx_tl_boot_seq ON transparency_log(boot_id, monotonic_seq);
CREATE INDEX IF NOT EXISTS idx_tl_prev_hash ON transparency_log(prev_event_hash) WHERE prev_event_hash IS NOT NULL;

-- Indexes for epoch-based queries (auditors)
CREATE INDEX IF NOT EXISTS idx_tl_epoch ON transparency_log(netuid, epoch_id) WHERE epoch_id IS NOT NULL;

-- RLS: Public reads (transparency), service_role writes
ALTER TABLE transparency_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anon can read transparency log" ON transparency_log 
    FOR SELECT TO anon USING (true);
CREATE POLICY "Service role can insert transparency log" ON transparency_log 
    FOR INSERT TO service_role WITH CHECK (true);
CREATE POLICY "Service role can update transparency log" ON transparency_log 
    FOR UPDATE TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role can delete transparency log" ON transparency_log 
    FOR DELETE TO service_role USING (true);


-- ============================================================================
-- TABLE 4: epoch_audit_logs
-- ============================================================================
-- 
-- Stores audit results per epoch. Generated by gateway's epoch audit job.
-- Status indicates whether the epoch passed verification or had issues.

CREATE TABLE IF NOT EXISTS epoch_audit_logs (
    id BIGSERIAL PRIMARY KEY,
    
    -- Epoch identification
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    
    -- Audit result
    status TEXT NOT NULL,                         -- VERIFIED, EQUIVOCATION_DETECTED, AUDITOR_MISMATCH, NO_TEE_BUNDLE
    
    -- Detailed audit data (JSONB for flexibility)
    audit_data JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Uniqueness: One audit per (netuid, epoch)
    UNIQUE(netuid, epoch_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_eal_netuid_epoch ON epoch_audit_logs(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_eal_status ON epoch_audit_logs(status);

-- RLS: Public reads, service_role writes
ALTER TABLE epoch_audit_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anon can read audit logs" ON epoch_audit_logs 
    FOR SELECT TO anon USING (true);
CREATE POLICY "Service role can insert audit logs" ON epoch_audit_logs 
    FOR INSERT TO service_role WITH CHECK (true);
CREATE POLICY "Service role can update audit logs" ON epoch_audit_logs 
    FOR UPDATE TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role can delete audit logs" ON epoch_audit_logs 
    FOR DELETE TO service_role USING (true);


-- ============================================================================
-- GRANTS
-- ============================================================================
-- 
-- Ensure roles have appropriate permissions

-- Grant select to anon (for public reads)
GRANT SELECT ON published_weight_bundles TO anon;
GRANT SELECT ON validator_attestations TO anon;
GRANT SELECT ON transparency_log TO anon;
GRANT SELECT ON epoch_audit_logs TO anon;

-- Grant all to service_role (for writes)
GRANT ALL ON published_weight_bundles TO service_role;
GRANT ALL ON validator_attestations TO service_role;
GRANT ALL ON transparency_log TO service_role;
GRANT ALL ON epoch_audit_logs TO service_role;

-- Grant sequence usage for inserts
GRANT USAGE, SELECT ON SEQUENCE published_weight_bundles_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE validator_attestations_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE transparency_log_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE epoch_audit_logs_id_seq TO service_role;


-- ============================================================================
-- COMMENTS
-- ============================================================================
-- 
-- Documentation for the schema

COMMENT ON TABLE published_weight_bundles IS 
    'Weight bundles submitted by validators. One row per (netuid, epoch, validator). Auditors must verify weight_submission_event_hash in transparency_log.';

COMMENT ON TABLE validator_attestations IS 
    'Validator TEE attestations. One per (netuid, epoch, validator). Validators regenerate each epoch; gateways reject stale attestations.';

COMMENT ON TABLE transparency_log IS 
    'Gateway-signed events forming a hash-chained audit trail. Primary source of truth for auditors. Verify by following prev_event_hash links.';

COMMENT ON TABLE epoch_audit_logs IS 
    'Audit results per epoch. Status indicates verification outcome: VERIFIED, EQUIVOCATION_DETECTED, AUDITOR_MISMATCH, or NO_TEE_BUNDLE.';

COMMENT ON COLUMN published_weight_bundles.weight_submission_event_hash IS 
    'Link to signed WEIGHT_SUBMISSION event in transparency_log. Auditors MUST verify this events signature.';

COMMENT ON COLUMN published_weight_bundles.chain_snapshot_compare_hash IS 
    'compare_weights_hash of on-chain weights at submission time. Used for equivocation detection without querying live chain.';

COMMENT ON COLUMN transparency_log.prev_event_hash IS 
    'Hash-chain link to previous event. Auditors reconstruct chain by following these links, NOT by created_at ordering.';

COMMENT ON COLUMN transparency_log.monotonic_seq IS 
    'Monotonically increasing sequence within each boot_id session. Allows detection of missing/reordered events within a boot.';

