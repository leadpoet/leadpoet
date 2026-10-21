-- Run as one standalone statement outside a transaction.
-- Deadline cancellation must exhaust all round-local open-call candidates when
-- none remain. Preserve the global later-head check and every existing row.
CREATE INDEX CONCURRENTLY lab_arena_ledger_round_open_call_idx
ON public.lab_arena_ledger USING btree (round_id)
WHERE call_identity IS NOT NULL AND entry_kind IN ('reservation', 'dispatch');
