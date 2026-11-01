-- Run as one standalone statement outside a transaction through the configured
-- direct-session migration bridge. A later migration verifies validity/shape.
-- Match the existing OpenRouter uncertainty index without blocking ledger writes.
CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_ledger_deepline_uncertain_run_entry_idx
ON public.lab_arena_ledger USING btree (run_id, entry_id)
WHERE provider = 'deepline' AND entry_kind = 'uncertain';
