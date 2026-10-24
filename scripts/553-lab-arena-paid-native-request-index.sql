-- Run as one session-affine autocommit statement through the reviewed direct
-- PostgreSQL bridge. Migration554 verifies its exact identity before using it.
CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_ledger_paid_native_request_idx
ON public.lab_arena_ledger (round_id, submission_id, provider,
  ((terminal_response #>> '{provider_cost,request_id}')))
WHERE entry_kind='settlement' AND amount_microusd>0
  AND length(terminal_response #>> '{provider_cost,request_id}') BETWEEN 1 AND 256;
