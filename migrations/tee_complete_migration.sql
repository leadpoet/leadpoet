-- TEE System Complete Migration
-- ===============================
-- 
-- This script safely:
-- 1. Creates NEW tables (published_weight_bundles, validator_attestations, epoch_audit_logs)
-- 2. ADDS columns to EXISTING transparency_log table (won't break existing data)
--
-- SAFE TO RUN: 
-- - Uses IF NOT EXISTS for tables
-- - Uses DO blocks to check before adding columns
-- - Existing data is NOT modified
-- - Existing inserts will continue to work (new columns are nullable)
--
-- Run this in Supabase SQL Editor.

-- ============================================================================
-- TABLE 1: published_weight_bundles (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS published_weight_bundles (
    id BIGSERIAL PRIMARY KEY,
    epoch_id INTEGER NOT NULL,
    block INTEGER NOT NULL,
    netuid INTEGER NOT NULL,
    uids INTEGER[] NOT NULL,
    weights_u16 INTEGER[] NOT NULL,
    CONSTRAINT uids_weights_same_length CHECK (cardinality(uids) = cardinality(weights_u16)),
    CONSTRAINT uids_not_empty CHECK (cardinality(uids) > 0),
    weights_hash TEXT NOT NULL,
    validator_hotkey TEXT NOT NULL,
    validator_enclave_pubkey TEXT NOT NULL,
    validator_signature TEXT NOT NULL,
    validator_attestation_b64 TEXT NOT NULL,
    validator_code_hash TEXT NOT NULL,
    chain_snapshot_block INTEGER,
    chain_snapshot_compare_hash TEXT,
    weight_submission_event_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(netuid, epoch_id, validator_hotkey)
);

CREATE INDEX IF NOT EXISTS idx_pwb_epoch ON published_weight_bundles(epoch_id);
CREATE INDEX IF NOT EXISTS idx_pwb_netuid_epoch ON published_weight_bundles(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_pwb_validator ON published_weight_bundles(validator_hotkey);
CREATE INDEX IF NOT EXISTS idx_pwb_event_hash ON published_weight_bundles(weight_submission_event_hash);


-- ============================================================================
-- TABLE 2: validator_attestations (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS validator_attestations (
    id BIGSERIAL PRIMARY KEY,
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    validator_hotkey TEXT NOT NULL,
    validator_enclave_pubkey TEXT NOT NULL,
    validator_attestation_b64 TEXT NOT NULL,
    validator_code_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(netuid, epoch_id, validator_hotkey)
);

CREATE INDEX IF NOT EXISTS idx_va_epoch ON validator_attestations(epoch_id);
CREATE INDEX IF NOT EXISTS idx_va_netuid_epoch ON validator_attestations(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_va_validator ON validator_attestations(validator_hotkey);


-- ============================================================================
-- TABLE 3: epoch_audit_logs (NEW)
-- ============================================================================

CREATE TABLE IF NOT EXISTS epoch_audit_logs (
    id BIGSERIAL PRIMARY KEY,
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    audit_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(netuid, epoch_id)
);

CREATE INDEX IF NOT EXISTS idx_eal_netuid_epoch ON epoch_audit_logs(netuid, epoch_id);
CREATE INDEX IF NOT EXISTS idx_eal_status ON epoch_audit_logs(status);


-- ============================================================================
-- TABLE 4: transparency_log (ADD COLUMNS TO EXISTING)
-- ============================================================================
-- These columns are OPTIONAL for existing events.
-- New events can populate them, old events will have NULL.

-- event_hash - unique identifier for signed events
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'event_hash'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN event_hash TEXT;
        RAISE NOTICE 'Added column: event_hash';
    END IF;
END $$;

-- enclave_pubkey - gateway pubkey that signed this event
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'enclave_pubkey'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN enclave_pubkey TEXT;
        RAISE NOTICE 'Added column: enclave_pubkey';
    END IF;
END $$;

-- boot_id - enclave boot session identifier
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'boot_id'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN boot_id TEXT;
        RAISE NOTICE 'Added column: boot_id';
    END IF;
END $$;

-- monotonic_seq - sequence within boot session
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'monotonic_seq'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN monotonic_seq BIGINT;
        RAISE NOTICE 'Added column: monotonic_seq';
    END IF;
END $$;

-- prev_event_hash - hash-chain link
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'prev_event_hash'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN prev_event_hash TEXT;
        RAISE NOTICE 'Added column: prev_event_hash';
    END IF;
END $$;

-- netuid - subnet ID (for filtering)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'netuid'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN netuid INTEGER;
        RAISE NOTICE 'Added column: netuid';
    END IF;
END $$;

-- epoch_id - epoch ID (for filtering)  
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'epoch_id'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN epoch_id INTEGER;
        RAISE NOTICE 'Added column: epoch_id';
    END IF;
END $$;

-- arweave_tx_id - for checkpointing
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'transparency_log' AND column_name = 'arweave_tx_id'
    ) THEN
        ALTER TABLE transparency_log ADD COLUMN arweave_tx_id TEXT;
        RAISE NOTICE 'Added column: arweave_tx_id';
    END IF;
END $$;


-- ============================================================================
-- INDEXES FOR TRANSPARENCY_LOG (safe - IF NOT EXISTS)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_tl_event_hash ON transparency_log(event_hash) WHERE event_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_pubkey ON transparency_log(enclave_pubkey) WHERE enclave_pubkey IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_boot_seq ON transparency_log(boot_id, monotonic_seq) WHERE boot_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_prev_hash ON transparency_log(prev_event_hash) WHERE prev_event_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_epoch ON transparency_log(netuid, epoch_id) WHERE epoch_id IS NOT NULL;


-- ============================================================================
-- UNIQUE CONSTRAINT ON event_hash (only if no duplicates exist)
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'transparency_log_event_hash_key'
    ) THEN
        ALTER TABLE transparency_log ADD CONSTRAINT transparency_log_event_hash_key UNIQUE (event_hash);
        RAISE NOTICE 'Added UNIQUE constraint on event_hash';
    END IF;
EXCEPTION
    WHEN unique_violation THEN
        RAISE WARNING 'Cannot add UNIQUE constraint on event_hash - duplicate values exist';
    WHEN OTHERS THEN
        RAISE WARNING 'Could not add UNIQUE constraint: %', SQLERRM;
END $$;


-- ============================================================================
-- RLS POLICIES FOR NEW TABLES
-- ============================================================================

-- published_weight_bundles
ALTER TABLE published_weight_bundles ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY "Anon can read weight bundles" ON published_weight_bundles FOR SELECT TO anon USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can insert weight bundles" ON published_weight_bundles FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can update weight bundles" ON published_weight_bundles FOR UPDATE TO service_role USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can delete weight bundles" ON published_weight_bundles FOR DELETE TO service_role USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- validator_attestations
ALTER TABLE validator_attestations ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY "Anon can read attestations" ON validator_attestations FOR SELECT TO anon USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can insert attestations" ON validator_attestations FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can update attestations" ON validator_attestations FOR UPDATE TO service_role USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can delete attestations" ON validator_attestations FOR DELETE TO service_role USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- epoch_audit_logs
ALTER TABLE epoch_audit_logs ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY "Anon can read audit logs" ON epoch_audit_logs FOR SELECT TO anon USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can insert audit logs" ON epoch_audit_logs FOR INSERT TO service_role WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can update audit logs" ON epoch_audit_logs FOR UPDATE TO service_role USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "Service role can delete audit logs" ON epoch_audit_logs FOR DELETE TO service_role USING (true);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT SELECT ON published_weight_bundles TO anon;
GRANT SELECT ON validator_attestations TO anon;
GRANT SELECT ON epoch_audit_logs TO anon;

GRANT ALL ON published_weight_bundles TO service_role;
GRANT ALL ON validator_attestations TO service_role;
GRANT ALL ON epoch_audit_logs TO service_role;

GRANT USAGE, SELECT ON SEQUENCE published_weight_bundles_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE validator_attestations_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE epoch_audit_logs_id_seq TO service_role;


-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
    tl_cols TEXT[];
BEGIN
    -- Check transparency_log columns
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'transparency_log'
    AND column_name IN ('event_hash', 'enclave_pubkey', 'boot_id', 'monotonic_seq', 'prev_event_hash', 'netuid', 'epoch_id');
    
    RAISE NOTICE '✅ transparency_log has %/7 TEE columns', col_count;
    
    -- Check new tables exist
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'published_weight_bundles') THEN
        RAISE NOTICE '✅ published_weight_bundles table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'validator_attestations') THEN
        RAISE NOTICE '✅ validator_attestations table exists';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'epoch_audit_logs') THEN
        RAISE NOTICE '✅ epoch_audit_logs table exists';
    END IF;
END $$;

-- Summary
SELECT 'Migration complete! Run this query to verify:' AS message;
SELECT table_name, column_name 
FROM information_schema.columns 
WHERE table_name IN ('published_weight_bundles', 'validator_attestations', 'epoch_audit_logs', 'transparency_log')
AND column_name IN ('event_hash', 'enclave_pubkey', 'boot_id', 'monotonic_seq', 'prev_event_hash', 'weight_submission_event_hash')
ORDER BY table_name, column_name;

