-- Run as one session-affine autocommit statement, outside a transaction.
-- Cover the unchanged 573/578 closed-cost selectors with scalar run metadata only.
-- OpenRouter admits more failed terminal causes than Deepline; preserve every
-- accepted/failed score run here and leave each selector's exact proof intact.
-- IF NOT EXISTS does not repair an invalid or differently defined existing index.
-- Before accepting/replaying this migration, verify pg_index.indisvalid and
-- indisready, the exact pg_get_indexdef, and owner lab_arena_owner. A failed
-- concurrent build must be repaired explicitly through the approved SQL channel.
CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_runs_closed_score_covering_idx
ON public.lab_arena_runs (round_id, run_id)
INCLUDE (status, terminal_cause, submission_id, miner_hotkey, stage)
WHERE kind='score' AND status IN ('accepted', 'failed');
