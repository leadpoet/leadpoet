-- Run as one session-affine autocommit statement, outside a transaction.
-- Covers the unchanged migration557 uncertainty CTE with scalar metadata only. Migration553 indexes
-- positive settlement native request IDs; it cannot serve this uncertainty scan.
-- IF NOT EXISTS does not repair an invalid or differently defined existing index.
-- Before accepting/replaying this migration, verify pg_index.indisvalid and
-- indisready, the exact pg_get_indexdef, and owner lab_arena_owner. A failed
-- concurrent build must be repaired explicitly through the approved SQL channel.
CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_ledger_deepline_uncertain_round_entry_idx
ON public.lab_arena_ledger (round_id, entry_id)
INCLUDE (run_id, call_identity)
WHERE provider='deepline' AND entry_kind='uncertain' AND call_identity IS NOT NULL;
