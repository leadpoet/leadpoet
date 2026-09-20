-- Apply the 30 percent future champion pool to the still-open Sep21 round.
-- The round predates the source default change, so its immutable configuration
-- still carries 25 percent. This one-time transition is valid only before the
-- benchmark, participant set, execution plan, scores, or reward basis exists.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $sep21_champion_pool_30$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_submissions_before JSONB;
  v_submissions_after JSONB;
  v_expected_configuration JSONB;
  v_count BIGINT;
  v_old_constants CONSTANT JSONB :=
    '{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25}'::JSONB;
  v_new_constants CONSTANT JSONB :=
    '{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":30}'::JSONB;
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal
      AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round write-once trigger is not enabled'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21'
  FOR UPDATE;
  IF v_round.round_id IS NULL THEN
    RAISE EXCEPTION 'Sep21 open round is missing' USING ERRCODE = 'P0002';
  END IF;

  IF v_round.status IS DISTINCT FROM 'open'
     OR v_round.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-21'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.participants IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.king_hotkey IS NOT NULL
     OR v_round.king_start_epoch IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.promotion_doc IS NOT NULL
     OR v_round.baseline_promoted_at IS NOT NULL THEN
    RAISE EXCEPTION 'Sep21 round is not open and unstarted'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-21'
     ) THEN
    RAISE EXCEPTION 'Sep21 execution or scoring plan already exists'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = 'arena-2026-09-21'
         AND (status = 'frozen' OR frozen_at IS NOT NULL OR is_king)
     ) THEN
    RAISE EXCEPTION 'Sep21 baseline or submission source is already frozen'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.configuration_doc -> 'reward_constants'
       IS DISTINCT FROM v_old_constants
     AND v_round.configuration_doc -> 'reward_constants'
       IS DISTINCT FROM v_new_constants THEN
    RAISE EXCEPTION 'Sep21 reward constants differ from the bounded transition'
      USING ERRCODE = '55000';
  END IF;

  -- A replay is harmless only while every pre-execution guard still holds.
  IF v_round.configuration_doc -> 'reward_constants' = v_new_constants THEN
    RETURN;
  END IF;

  SELECT COALESCE(
           pg_catalog.jsonb_agg(
             pg_catalog.to_jsonb(submission) ORDER BY submission_id
           ),
           '[]'::JSONB
         )
  INTO v_submissions_before
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = 'arena-2026-09-21';

  v_expected_configuration := pg_catalog.jsonb_set(
    v_round.configuration_doc,
    '{reward_constants,pool_percent}',
    '30'::JSONB,
    FALSE
  );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;

  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected_configuration,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-21';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep21 champion pool update count differs'
      USING ERRCODE = '55000';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal
      AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round write-once trigger was not restored'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21';
  SELECT COALESCE(
           pg_catalog.jsonb_agg(
             pg_catalog.to_jsonb(submission) ORDER BY submission_id
           ),
           '[]'::JSONB
         )
  INTO v_submissions_after
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = 'arena-2026-09-21';

  IF v_after.configuration_doc IS DISTINCT FROM v_expected_configuration
     OR v_after.configuration_doc -> 'reward_constants'
          IS DISTINCT FROM v_new_constants
     OR (
       (
         pg_catalog.to_jsonb(v_after)
           #- ARRAY['configuration_doc','reward_constants','pool_percent']::TEXT[]
       ) - 'updated_at'::TEXT
     ) IS DISTINCT FROM (
       (
         pg_catalog.to_jsonb(v_round)
           #- ARRAY['configuration_doc','reward_constants','pool_percent']::TEXT[]
       ) - 'updated_at'::TEXT
     )
     OR v_submissions_after IS DISTINCT FROM v_submissions_before THEN
    RAISE EXCEPTION 'Sep21 champion pool transition changed protected state'
      USING ERRCODE = '55000';
  END IF;
END;
$sep21_champion_pool_30$;

COMMIT;
