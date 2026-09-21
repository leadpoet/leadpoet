-- Adopt the explicitly requested cutoff disclosure for the current and next
-- daily rounds. Change no bank, source, run, score, budget, or reward state.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;

DO $cutoff_disclosure$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count INTEGER := 0;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round write-once trigger is not enabled';
  END IF;

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  FOR v_round IN
    SELECT * FROM public.lab_arena_rounds
    WHERE round_id IN ('arena-2026-09-21', 'arena-2026-09-22')
    ORDER BY round_id FOR UPDATE
  LOOP
    v_count := v_count + 1;
    IF v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
       OR v_round.configuration_doc ->> 'round_id' IS DISTINCT FROM v_round.round_id
       OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
            IS DISTINCT FROM substring(v_round.round_id FROM 7) || 'T00:00:00Z'
       OR v_round.configuration_doc ->> 'benchmark_disclosure_policy'
            NOT IN ('after_scoring_day2_v1', 'cutoff_public_v1')
       OR v_round.configuration_doc ->> 'benchmark_disclosure_policy' IS NULL THEN
      RAISE EXCEPTION 'Daily disclosure transition has unexpected configuration';
    END IF;

    -- Replays preserve already-applied rows, including later publication.
    IF v_round.configuration_doc ->> 'benchmark_disclosure_policy' = 'cutoff_public_v1' THEN
      CONTINUE;
    END IF;
    IF v_round.status IN ('published', 'cancelled')
       OR v_round.publication_doc IS NOT NULL OR v_round.published_at IS NOT NULL
       OR v_round.reward_basis_doc IS NOT NULL OR v_round.reward_activated_at IS NOT NULL THEN
      RAISE EXCEPTION 'Daily disclosure transition refuses a completed round';
    END IF;

    v_expected := pg_catalog.jsonb_set(
      v_round.configuration_doc, '{benchmark_disclosure_policy}',
      '"cutoff_public_v1"'::JSONB, FALSE
    );
    UPDATE public.lab_arena_rounds
    SET configuration_doc = v_expected, updated_at = pg_catalog.clock_timestamp()
    WHERE round_id = v_round.round_id;

    SELECT * INTO v_after FROM public.lab_arena_rounds WHERE round_id = v_round.round_id;
    IF v_after.configuration_doc IS DISTINCT FROM v_expected
       OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
            IS DISTINCT FROM
          (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at') THEN
      RAISE EXCEPTION 'Daily disclosure transition changed protected state';
    END IF;
  END LOOP;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 2 THEN
    RAISE EXCEPTION 'Daily disclosure transition requires both expected rounds';
  END IF;
END;
$cutoff_disclosure$;

COMMIT;
