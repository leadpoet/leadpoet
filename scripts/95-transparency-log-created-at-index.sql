-- Restore gateway bootability: index transparency_log.created_at.
--
-- The gateway's enclave event-signer init reads the durable transparency-log
-- tip with:
--     SELECT event_hash, created_at
--     FROM transparency_log
--     ORDER BY created_at DESC
--     LIMIT 100;
-- transparency_log is an append-only audit table that only grows, and there
-- was no index on created_at (only on ts). The ORDER BY forced a full-table
-- sort that eventually exceeded the statement timeout (error 57014), so the
-- gateway refused to start ("cannot read the durable transparency-log tip").
--
-- CONCURRENTLY so the build never locks writes on the hot audit table. Cannot
-- run inside a transaction block — run this statement on its own (psql with a
-- generous statement_timeout, e.g. PGOPTIONS='-c statement_timeout=600000').

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transparency_log_created_at
    ON public.transparency_log USING btree (created_at DESC);

-- Verify (should show an Index Scan, execution in single-digit ms):
--   EXPLAIN ANALYZE
--   SELECT event_hash, created_at FROM transparency_log
--   ORDER BY created_at DESC LIMIT 100;
