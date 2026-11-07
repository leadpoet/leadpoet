-- Run as one session-affine autocommit statement, outside a transaction.
-- Keep the existing runtime-start metadata query and every event unchanged.
-- Match its bound event kind and run IDs in custom and retained generic plans.
-- Repeated per-run event kinds can deduplicate; do not include wide payloads.
-- IF NOT EXISTS does not repair an invalid or differently defined existing index.
-- Before accepting/replaying this migration, verify pg_index.indisvalid and
-- indisready, the exact pg_get_indexdef, and owner lab_arena_owner. A failed
-- concurrent build must be repaired explicitly through the approved SQL channel.
CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_trajectory_kind_run_idx
ON public.lab_arena_trajectory_events (event_kind, run_id);
