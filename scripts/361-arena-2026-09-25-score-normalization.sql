-- Enable reachable-score normalization only for the exact, unstarted
-- September 25 Arena round. Historical rounds and all other frozen settings
-- stay unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $sep25_score_normalization$
DECLARE
  v_before public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count INTEGER;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round immutability trigger is not enabled';
  END IF;

  SELECT * INTO v_before FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-25' FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'September 25 round is missing' USING ERRCODE = 'P0002';
  END IF;

  IF v_before.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_before.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-25'
     OR v_before.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_before.configuration_doc ->> 'network_name' IS DISTINCT FROM 'finney'
     OR v_before.configuration_doc ->> 'netuid' IS DISTINCT FROM '71'
     OR v_before.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_before.configuration_doc ->> 'intent_details_policy'
       IS DISTINCT FROM 'intent_details_v1'
     OR v_before.configuration_doc ? 'contact_policy'
     OR v_before.configuration_doc #>> '{scorer_policy,scoring_adapter_version}'
       IS DISTINCT FROM 'qualification_integrity_v2'
     OR v_before.configuration_doc #>> '{scorer_policy,intent_details_policy}'
       IS DISTINCT FROM 'intent_details_v1'
     OR pg_catalog.jsonb_typeof(
          v_before.configuration_doc #> '{scorer_policy,env_bindings}'
        ) IS DISTINCT FROM 'object'
     OR v_before.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-25T00:00:00Z' THEN
    RAISE EXCEPTION 'September 25 configuration identity differs';
  END IF;

  -- A replay after successful conversion is a strict no-op, even after the
  -- round later starts. A different value is never overwritten.
  IF v_before.configuration_doc
       #>> '{scorer_policy,env_bindings,ARENA_SCORE_NORMALIZATION}'
       = 'available_intent_cap_v1' THEN
    RETURN;
  END IF;
  IF (v_before.configuration_doc #> '{scorer_policy,env_bindings}')
       ? 'ARENA_SCORE_NORMALIZATION' THEN
    RAISE EXCEPTION 'September 25 score normalization binding differs';
  END IF;

  IF v_before.status IS DISTINCT FROM 'open'
     OR v_before.stage_generation IS DISTINCT FROM 0
     OR v_before.participants IS NOT NULL
     OR v_before.benchmark_ref IS NOT NULL
     OR v_before.stage1_scoring_plan_doc IS NOT NULL
     OR v_before.stage2_scoring_plan_doc IS NOT NULL
     OR v_before.stage3_scoring_plan_doc IS NOT NULL
     OR v_before.finalists IS NOT NULL
     OR v_before.publication_doc IS NOT NULL
     OR v_before.published_at IS NOT NULL
     OR v_before.cancel_reason IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-25'
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = 'arena-2026-09-25'
         AND (status = 'frozen' OR frozen_at IS NOT NULL)
     ) THEN
    RAISE EXCEPTION 'September 25 requires an open, unfrozen, unstarted round';
  END IF;

  v_expected := pg_catalog.jsonb_set(
    v_before.configuration_doc,
    '{scorer_policy,env_bindings,ARENA_SCORE_NORMALIZATION}',
    '"available_intent_cap_v1"'::JSONB,
    TRUE
  );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-25';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  SELECT * INTO v_after FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-25';
  IF v_count <> 1
     OR v_after.configuration_doc IS DISTINCT FROM v_expected
     OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
        IS DISTINCT FROM
        (pg_catalog.to_jsonb(v_before) - 'configuration_doc' - 'updated_at')
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'Score normalization changed protected round state';
  END IF;
END;
$sep25_score_normalization$;
COMMIT;
