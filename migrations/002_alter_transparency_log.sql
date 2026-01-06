-- Alter existing transparency_log table for TEE signing
-- =====================================================
-- 
-- This migration ADDS new columns to the EXISTING transparency_log table.
-- Run this AFTER 001_tee_tables.sql if transparency_log already exists.
--
-- New columns added:
-- - event_hash (for signed event lookup)
-- - enclave_pubkey (gateway pubkey that signed)
-- - boot_id, monotonic_seq, prev_event_hash (hash-chain fields)
-- - netuid, epoch_id (for epoch-based queries)
--
-- SAFE TO RUN: Uses IF NOT EXISTS for columns where supported,
-- and wraps in DO blocks for idempotency.

-- ============================================================================
-- ADD NEW COLUMNS (if they don't exist)
-- ============================================================================

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
-- ADD UNIQUE CONSTRAINT ON event_hash (if not exists)
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'transparency_log_event_hash_key'
    ) THEN
        -- Only add constraint if there are no duplicate event_hash values
        -- (existing rows may have NULL which is fine)
        ALTER TABLE transparency_log ADD CONSTRAINT transparency_log_event_hash_key UNIQUE (event_hash);
        RAISE NOTICE 'Added UNIQUE constraint on event_hash';
    END IF;
EXCEPTION
    WHEN unique_violation THEN
        RAISE WARNING 'Cannot add UNIQUE constraint on event_hash - duplicate values exist';
END $$;


-- ============================================================================
-- ADD INDEXES (if not exist)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_tl_event_hash ON transparency_log(event_hash) WHERE event_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_pubkey ON transparency_log(enclave_pubkey) WHERE enclave_pubkey IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_boot_seq ON transparency_log(boot_id, monotonic_seq) WHERE boot_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_prev_hash ON transparency_log(prev_event_hash) WHERE prev_event_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tl_epoch ON transparency_log(netuid, epoch_id) WHERE epoch_id IS NOT NULL;


-- ============================================================================
-- VERIFY
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'transparency_log'
    AND column_name IN ('event_hash', 'enclave_pubkey', 'boot_id', 'monotonic_seq', 'prev_event_hash', 'netuid', 'epoch_id');
    
    RAISE NOTICE 'transparency_log now has % of 7 new TEE columns', col_count;
END $$;

