-- Raise every scoring provider quota for the existing, unstarted Sep22 round.
-- The round configuration remains write-once outside this exact transition.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $sep22_scoring_quota_prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 351';
  END IF;
END;
$sep22_scoring_quota_prerequisites$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $sep22_scoring_provider_quota$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count BIGINT;
  v_old_quotas CONSTANT JSONB :=
    '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB;
  v_new_quotas CONSTANT JSONB :=
    '{"deepline":2000,"openrouter":2000,"scrapingdog":2000}'::JSONB;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep22 round is missing' USING ERRCODE = 'P0002';
  END IF;

  IF v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-22'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-22T00:00:00Z'
     OR v_round.configuration_doc -> 'scoring_call_quotas'
       NOT IN (v_old_quotas, v_new_quotas)
     OR v_round.configuration_doc -> 'scoring_call_quotas' IS NULL
     OR (v_round.configuration_doc ->> 'scoring_cap_microusd')::BIGINT
       IS DISTINCT FROM 50000000
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 900 THEN
    RAISE EXCEPTION 'Sep22 scoring quota profile differs from migration 351'
      USING ERRCODE = '55000';
  END IF;

  -- A completed transition stays replay-safe after normal round progress.
  IF v_round.configuration_doc -> 'scoring_call_quotas' = v_new_quotas THEN
    RETURN;
  END IF;

  IF v_round.status IS DISTINCT FROM 'open'
     OR v_round.stage_generation <> 0
     OR v_round.participants IS NOT NULL
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-22'
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = 'arena-2026-09-22'
         AND (status = 'frozen' OR frozen_at IS NOT NULL)
     ) THEN
    RAISE EXCEPTION 'Sep22 scoring quota repair requires an open, unfrozen, unstarted round'
      USING ERRCODE = '55000';
  END IF;

  v_expected := pg_catalog.jsonb_set(
    v_round.configuration_doc,
    '{scoring_call_quotas}',
    v_new_quotas,
    FALSE
  );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-22';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep22 scoring quota update count differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22';
  IF v_after.configuration_doc IS DISTINCT FROM v_expected
     OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at')
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'Sep22 scoring quota repair changed protected state'
      USING ERRCODE = '55000';
  END IF;
END;
$sep22_scoring_provider_quota$;

COMMIT;
