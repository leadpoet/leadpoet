-- Run as one standalone statement outside a transaction, only after migration
-- 533 and the replacement index have passed exact production readback.
-- The complete round-call index replaces this narrower access path.
DROP INDEX CONCURRENTLY public.lab_arena_ledger_round_open_call_idx;
