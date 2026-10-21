-- Run as one standalone statement outside a transaction. Verify valid/ready and
-- exact definition before testing or installing the accompanying query change.
-- All round-local kinds must be available before pruning closed call heads.
CREATE INDEX CONCURRENTLY lab_arena_ledger_round_call_head_idx
ON public.lab_arena_ledger USING btree (round_id)
WHERE call_identity IS NOT NULL;
