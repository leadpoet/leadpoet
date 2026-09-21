-- Raise only the active Sep21 round's frozen OpenRouter execution quota.
-- Existing execution, billing, source, benchmark, score, and reward state is
-- retained.  The current per-ICP four-dollar cost cap remains unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $sep21_quota_prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 345';
  END IF;
END;
$sep21_quota_prerequisites$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;

DO $sep21_openrouter_quota$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count BIGINT;
  v_old_quotas CONSTANT JSONB :=
    '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB;
  v_new_quotas CONSTANT JSONB :=
    '{"deepline":200,"openrouter":2000,"scrapingdog":200}'::JSONB;
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
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep21 round is missing' USING ERRCODE = 'P0002';
  END IF;

  IF v_round.status IN ('published', 'cancelled')
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL THEN
    RAISE EXCEPTION 'Sep21 quota repair requires an active unpublished round'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-21'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-21T00:00:00Z'
     OR v_round.configuration_doc -> 'call_quotas'
       NOT IN (v_old_quotas, v_new_quotas)
     OR v_round.configuration_doc -> 'call_quotas' IS NULL
     OR v_round.configuration_doc -> 'scoring_call_quotas'
       IS DISTINCT FROM
       '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 2700
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
       IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT
       IS DISTINCT FROM 80000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
       IS DISTINCT FROM 800000
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
       IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true'
     OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
       IS DISTINCT FROM 10
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER
       IS DISTINCT FROM 10
     OR (v_round.configuration_doc ->> 'max_attempts_per_assignment')::INTEGER
       IS DISTINCT FROM 2 THEN
    RAISE EXCEPTION 'Sep21 quota profile differs from the bounded repair'
      USING ERRCODE = '55000';
  END IF;

  -- Replays validate the same active profile and perform no write.
  IF v_round.configuration_doc -> 'call_quotas' = v_new_quotas THEN
    RETURN;
  END IF;

  v_expected := pg_catalog.jsonb_set(
    v_round.configuration_doc,
    '{call_quotas,openrouter}',
    '2000'::JSONB,
    FALSE
  );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;

  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-21';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep21 quota repair update count differs'
      USING ERRCODE = '55000';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21';
  IF v_after.configuration_doc IS DISTINCT FROM v_expected
     OR v_after.configuration_doc -> 'call_quotas'
       IS DISTINCT FROM v_new_quotas
     OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at') THEN
    RAISE EXCEPTION 'Sep21 quota repair changed protected round state'
      USING ERRCODE = '55000';
  END IF;

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
END;
$sep21_openrouter_quota$;

COMMIT;
