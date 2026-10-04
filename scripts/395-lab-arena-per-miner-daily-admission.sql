-- Remove the former twenty-challenger default for unstarted daily rounds.
-- One accepted model per hotkey, replacement rules, credentials, worker limits,
-- and frozen historical configurations remain enforced by the existing schema.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Serialize with admission, cutoff, and participant/run creation. Restore the
-- write-once guard in the same transaction; a failed migration rolls back all.
LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;
ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;

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
  );

ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;
COMMIT;
