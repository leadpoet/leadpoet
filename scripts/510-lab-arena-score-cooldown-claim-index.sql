-- Bound the existing correlated score-runner cooldown lookup by exact scope.
-- Build outside a transaction and fail on a name collision. The release must
-- verify the index is valid/ready and has this exact definition before use.
CREATE INDEX CONCURRENTLY lab_arena_runs_score_cooldown_claim_idx
ON public.lab_arena_runs
  (round_id, runner_hotkey, stage, stage_generation, lease_expires_at)
WHERE kind = 'score'
  AND status = 'failed'
  AND terminal_cause = 'lease_expired'
  AND result_doc IS NULL
  AND output_ref IS NULL;
