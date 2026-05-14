-- 19-transparency-log-epoch-init-index.sql
--
-- Adds a partial expression index that fixes the recurring 8-second
-- statement_timeout (Postgres error 57014) on the EPOCH_INITIALIZATION
-- lookup performed by /epoch/{n}/leads in gateway/api/epoch.py:266-278.
--
-- The query is:
--   SELECT payload
--   FROM   transparency_log
--   WHERE  event_type   = 'EPOCH_INITIALIZATION'
--     AND  payload->>'epoch_id' = $1
--   LIMIT  1;
--
-- Without an index, this requires scanning every transparency_log row,
-- parsing the JSONB payload, and extracting `epoch_id`.  On the live
-- table that takes longer than the 8s statement_timeout, so every
-- request to /epoch/leads falls through to the slower fallback that
-- queries leads_private directly — which in turn returns 0 leads when
-- the queue is empty and traps validators in an infinite retry loop.
--
-- The index below is a *partial expression* index: it only stores rows
-- whose event_type is 'EPOCH_INITIALIZATION', and the indexed value is
-- the JSONB-extracted epoch_id.  This makes the exact lookup the
-- gateway runs a single index probe (O(log n)) and keeps the index
-- small (~one row per epoch, not one row per transparency event).
--
-- CONCURRENTLY is required because transparency_log is large and
-- write-heavy.  CREATE INDEX without CONCURRENTLY would take an
-- ACCESS EXCLUSIVE lock for the duration of the build, blocking every
-- INSERT into the log.  CONCURRENTLY trades a longer build time for a
-- non-blocking ShareUpdateExclusive lock.
--
-- IMPORTANT: CONCURRENTLY cannot run inside a transaction.  Execute
-- this statement standalone (not in a transaction block).  In Supabase
-- SQL Editor, run each statement individually rather than using the
-- "Run" multi-statement mode.

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_transparency_log_epoch_init_epoch_id
ON public.transparency_log ((payload->>'epoch_id'))
WHERE event_type = 'EPOCH_INITIALIZATION';

-- After the index finishes building, refresh planner stats so it gets
-- picked up immediately by the next query plan.
ANALYZE public.transparency_log;

-- Verify the index exists and is valid (not in a FAILED state from a
-- prior aborted CONCURRENTLY build).  An entry with indisvalid = false
-- indicates a broken index that must be dropped and rebuilt.
-- Run this after the CREATE INDEX completes:
--
--   SELECT indexrelid::regclass AS index,
--          indisvalid,
--          pg_size_pretty(pg_relation_size(indexrelid)) AS size
--   FROM   pg_index
--   WHERE  indrelid = 'public.transparency_log'::regclass
--     AND  indexrelid::regclass::text = 'idx_transparency_log_epoch_init_epoch_id';
