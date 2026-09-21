-- Raise only the active Sep21 round's frozen scoring provider quotas.
-- Existing scoring leases, judgments, receipts, billing, and source state stay
-- immutable. Provider admission reads the new limits from this configuration.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $sep21_scoring_quota_prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 352';
  END IF;
END;
$sep21_scoring_quota_prerequisites$;

-- This follows migration 345's lock order. Provider admission must finish its
-- current round read before this short configuration-only transaction runs.
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;

DO $sep21_scoring_provider_quota$
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
  WHERE round_id = 'arena-2026-09-21'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep21 round is missing' USING ERRCODE = 'P0002';
  END IF;

  IF v_round.status NOT IN ('stage1_scoring', 'stage2_scoring')
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL THEN
    RAISE EXCEPTION 'Sep21 scoring quota repair requires an active unpublished scoring round'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-21'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-21T00:00:00Z'
     OR v_round.configuration_doc -> 'scoring_call_quotas'
       NOT IN (v_old_quotas, v_new_quotas)
     OR v_round.configuration_doc -> 'scoring_call_quotas' IS NULL
     OR (v_round.configuration_doc ->> 'scoring_cap_microusd')::BIGINT
       IS DISTINCT FROM 50000000
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 900 THEN
    RAISE EXCEPTION 'Sep21 scoring quota profile differs from migration 352'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.configuration_doc -> 'scoring_call_quotas' = v_new_quotas THEN
    RETURN;
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
  WHERE round_id = 'arena-2026-09-21';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep21 scoring quota update count differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21';
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
    RAISE EXCEPTION 'Sep21 scoring quota repair changed protected state'
      USING ERRCODE = '55000';
  END IF;
END;
$sep21_scoring_provider_quota$;

COMMIT;
