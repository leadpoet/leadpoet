-- Remove the former twenty-challenger default for unstarted daily rounds.
-- One accepted model per hotkey, replacement rules, credentials, worker limits,
-- and frozen historical configurations remain enforced by the existing schema.
-- A capacity rejection on an expanded open round only returns to uploading;
-- the miner must finalize the saved source and credentials through signed intake.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Serialize with admission, cutoff, and participant/run creation. Restore the
-- write-once guard in the same transaction; a failed migration rolls back all.
LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;
ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;
-- The normal rejected-row guard is restored before commit. ALTER TABLE takes
-- the table lock, so no concurrent submission write can see it disabled.
ALTER TABLE public.lab_arena_submissions
  DISABLE TRIGGER lab_arena_submissions_frozen;

WITH expanded AS (
  UPDATE public.lab_arena_rounds AS round
  SET configuration_doc = pg_catalog.jsonb_set(
    round.configuration_doc, '{max_challengers}', '256'::JSONB, FALSE
  )
  WHERE round.round_id ~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}$'
    AND round.status = 'open'
    AND round.arena_network_name = 'finney'
    AND round.arena_netuid = 71
    AND round.configuration_doc ->> 'mode' = 'live'
    AND round.configuration_doc ->> 'max_challengers' = '20'
    AND COALESCE(
      (round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
        > pg_catalog.clock_timestamp(),
      FALSE
    )
    AND round.benchmark_ref IS NULL
    AND (round.participants IS NULL OR round.participants = '[]'::JSONB)
    AND NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs AS run
      WHERE run.round_id = round.round_id
    )
  RETURNING round.round_id
), latest AS (
  SELECT DISTINCT ON (submission.round_id, submission.miner_hotkey)
    submission.submission_id, submission.status, submission.rejection_rule,
    submission.replaced_by_submission_id, submission.is_king
  FROM public.lab_arena_submissions AS submission
  JOIN expanded USING (round_id)
  ORDER BY submission.round_id, submission.miner_hotkey,
    submission.created_at DESC, submission.submission_id DESC
)
UPDATE public.lab_arena_submissions AS submission
SET status = 'uploading', rejection_rule = NULL
FROM latest
WHERE submission.submission_id = latest.submission_id
  AND latest.status = 'rejected'
  AND latest.rejection_rule = 'capacity.round_full'
  AND latest.replaced_by_submission_id IS NULL
  AND latest.is_king IS FALSE
  AND NOT EXISTS (
    SELECT 1 FROM public.lab_arena_submissions AS active
    WHERE active.round_id = submission.round_id
      AND active.miner_hotkey = submission.miner_hotkey
      AND active.status = 'uploading'
  );

ALTER TABLE public.lab_arena_submissions
  ENABLE TRIGGER lab_arena_submissions_frozen;
ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;
COMMIT;
