-- Run as one standalone statement outside a transaction.
-- Closed billing keeps every existing identity, payload and latest-head check.
-- Limit each run lookup to OpenRouter uncertainty rows instead of all its calls.
CREATE INDEX CONCURRENTLY lab_arena_ledger_openrouter_uncertain_run_entry_idx
ON public.lab_arena_ledger USING btree (run_id, entry_id)
WHERE provider = 'openrouter' AND entry_kind = 'uncertain';
