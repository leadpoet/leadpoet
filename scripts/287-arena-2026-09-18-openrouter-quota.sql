-- Raise only the still-open Sep18 round's frozen OpenRouter execution quota.
-- This one-time repair is valid only before the submission cutoff and before
-- baseline admission, participant freeze, benchmark selection, or execution.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $requires_sep18_open_quota$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 287';
  END IF;
END;
$requires_sep18_open_quota$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $verify_locked_round_trigger$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal
      AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena rounds write-once trigger is not enabled'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_locked_round_trigger$;

ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;

DO $sep18_open_quota$
DECLARE
  v_round public.lab_arena_rounds;
  v_before JSONB;
  v_after JSONB;
  v_cutoff CONSTANT TIMESTAMPTZ := '2026-09-18T00:00:00Z';
  v_old_quotas CONSTANT JSONB :=
    '{"deepline":30,"openrouter":60,"scrapingdog":30}'::JSONB;
  v_new_quotas CONSTANT JSONB :=
    '{"deepline":30,"openrouter":200,"scrapingdog":30}'::JSONB;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18'
  FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep18 open round missing' USING ERRCODE = 'P0002';
  END IF;

  IF pg_catalog.clock_timestamp() >= v_cutoff THEN
    RAISE EXCEPTION 'sep18 quota repair must be applied before submission cutoff'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.status IS DISTINCT FROM 'open'
     OR v_round.status_generation IS DISTINCT FROM 0
     OR v_round.stage_generation IS DISTINCT FROM 0
     OR NOT v_round.rewards_enabled
     OR v_round.participants IS NOT NULL
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.evaluation_date IS NOT NULL
     OR v_round.icp_set_date IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-18'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'network_name' IS DISTINCT FROM 'finney'
     OR (v_round.configuration_doc ->> 'netuid')::INTEGER IS DISTINCT FROM 71
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
          IS DISTINCT FROM '2026-09-18T00:00:00Z'
     OR (
       v_round.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_old_quotas
       AND v_round.configuration_doc -> 'call_quotas'
             IS DISTINCT FROM v_new_quotas
     )
     OR v_round.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM
          '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
          IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 2700
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 900
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          IS DISTINCT FROM 20
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
          IS DISTINCT FROM 'true'
     OR (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT
          IS DISTINCT FROM 80000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_v1'
     OR v_round.configuration_doc ->> 'baseline_source_url'
          IS DISTINCT FROM
          'https://github.com/leadpoet/champion_model/archive/refs/heads/lab.tar.gz'
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = v_round.round_id
         AND (
           is_king
           OR submission_id = 'baseline-2026-09-18'
           OR miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey'
         )
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = v_round.round_id
         AND status = 'frozen'
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
     ) THEN
    RAISE EXCEPTION 'sep18 open quota state differs from the sealed profile'
      USING ERRCODE = '55000';
  END IF;

  v_before := v_round.configuration_doc;
  IF v_before -> 'call_quotas' = v_old_quotas THEN
    UPDATE public.lab_arena_rounds
    SET configuration_doc = pg_catalog.jsonb_set(
      configuration_doc,
      '{call_quotas,openrouter}',
      '200'::JSONB,
      FALSE
    )
    WHERE round_id = 'arena-2026-09-18';

    IF NOT FOUND THEN
      RAISE EXCEPTION 'sep18 quota update lost its target' USING ERRCODE = '55000';
    END IF;
  END IF;

  SELECT configuration_doc INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18';

  IF (v_after - 'call_quotas') IS DISTINCT FROM (v_before - 'call_quotas')
     OR ((v_after -> 'call_quotas') - 'openrouter') IS DISTINCT FROM
        ((v_before -> 'call_quotas') - 'openrouter')
     OR v_after -> 'call_quotas' IS DISTINCT FROM v_new_quotas THEN
    RAISE EXCEPTION 'sep18 quota repair changed an unapproved field'
      USING ERRCODE = '55000';
  END IF;
END;
$sep18_open_quota$;

ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;

DO $verify_sep18_open_quota$
BEGIN
  IF NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-18'
         AND configuration_doc -> 'call_quotas' =
             '{"deepline":30,"openrouter":200,"scrapingdog":30}'::JSONB
     )
     OR NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal
         AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'sep18 quota repair verification failed'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_sep18_open_quota$;

COMMIT;
