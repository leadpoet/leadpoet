-- Index fulfillment_requests.successor_request_id for the recursive successor
-- chain walk (gateway/fulfillment/lifecycle.py _walk_chain_predecessors), which
-- issues up to 1000 sequential .eq("successor_request_id", ...) lookups per
-- consensus pass. Postgres does NOT auto-index a foreign key's referencing
-- column, so without this index those walks are sequential scans that saturate
-- the connection pool (the documented 5-35s REST-stall / dropped-reward outage).
--
-- This index was applied to production manually (sql/add_chain_successor_index.sql,
-- 2026-06-23) but was never captured as a numbered migration, so a rebuild that
-- runs only the ordered scripts/ set would omit it and reintroduce the wedge.
-- This migration makes it part of the ordered set.
--
-- Idempotent + additive. CONCURRENTLY must run OUTSIDE a transaction block; if
-- your migration runner wraps statements in a transaction, apply this one
-- separately (e.g. psql -f) rather than inside the transactional batch.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fr_successor
  ON fulfillment_requests (successor_request_id);

ANALYZE fulfillment_requests;
