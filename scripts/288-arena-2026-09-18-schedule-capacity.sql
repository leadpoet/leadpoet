-- Extend only the still-open Sep18 round's evaluation schedule to fit the
-- measured ten-slot workload of four reviewed miners plus the future baseline.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_sep18_schedule$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 288';
  END IF;
END;
$requires_sep18_schedule$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $verify_locked_round_trigger$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena rounds write-once trigger is not enabled'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_locked_round_trigger$;

ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;

DO $sep18_schedule$
DECLARE
  v_round public.lab_arena_rounds;
  v_before JSONB;
  v_after JSONB;
  v_submission_count BIGINT;
  v_reviewed_count BIGINT;
  v_ledger_count BIGINT;
  v_ledger_submissions BIGINT;
  v_execution_seconds BIGINT;
  v_scoring_seconds BIGINT;
  v_cutoff CONSTANT TIMESTAMPTZ := '2026-09-18T00:00:00Z';
  v_old_schedule CONSTANT JSONB :=
    $old${"benchmark_deadline":"2026-09-18T00:30:00Z","final_scoring_close":"2026-09-18T20:30:02Z","publication_deadline":"2026-09-18T20:30:03Z","stage_1_close":"2026-09-18T04:30:01Z","stage_1_scoring_close":"2026-09-18T11:00:01Z","stage_1_start":"2026-09-18T00:30:01Z","stage_2_close":"2026-09-18T14:00:02Z","stage_2_start":"2026-09-18T11:00:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$old$::JSONB;
  v_new_schedule CONSTANT JSONB :=
    $new${"benchmark_deadline":"2026-09-18T00:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T16:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T00:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$new$::JSONB;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep18 open round missing' USING ERRCODE = 'P0002';
  END IF;
  IF pg_catalog.clock_timestamp() >= v_cutoff THEN
    RAISE EXCEPTION 'sep18 schedule repair must be applied before submission cutoff'
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
     OR (v_round.configuration_doc -> 'schedule'
           IS DISTINCT FROM v_old_schedule
         AND v_round.configuration_doc -> 'schedule'
           IS DISTINCT FROM v_new_schedule)
     OR v_round.configuration_doc -> 'call_quotas' IS DISTINCT FROM
          '{"deepline":30,"openrouter":200,"scrapingdog":30}'::JSONB
     OR v_round.configuration_doc -> 'scoring_call_quotas' IS DISTINCT FROM
          '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
          IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 2700
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 900
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          IS DISTINCT FROM 20
     OR (v_round.configuration_doc ->> 'max_attempts_per_assignment')::INTEGER
          IS DISTINCT FROM 2
     OR (v_round.configuration_doc ->> 'max_challengers')::INTEGER
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
           OR status = 'frozen'
         )
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs WHERE round_id = v_round.round_id
     ) THEN
    RAISE EXCEPTION 'sep18 open schedule state differs from the sealed profile'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.count(*),
         pg_catalog.count(*) FILTER (
           WHERE status = 'accepted' AND NOT is_king
             AND code_review_status = 'passed'
             AND code_review_doc IS NOT NULL
             AND code_review_claim IS NOT NULL
             AND source_ref IS NOT NULL
         )
  INTO v_submission_count, v_reviewed_count
  FROM public.lab_arena_submissions
  WHERE round_id = v_round.round_id;
  IF v_submission_count IS DISTINCT FROM 4
     OR v_reviewed_count IS DISTINCT FROM 4 THEN
    RAISE EXCEPTION 'sep18 reviewed admission snapshot differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.count(*), pg_catalog.count(DISTINCT submission_id)
  INTO v_ledger_count, v_ledger_submissions
  FROM public.lab_arena_ledger WHERE round_id = v_round.round_id;
  IF v_ledger_count IS DISTINCT FROM 12
     OR v_ledger_submissions IS DISTINCT FROM 4
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_ledger AS ledger
       LEFT JOIN public.lab_arena_submissions AS submission
         ON submission.submission_id = ledger.submission_id
        AND submission.round_id = ledger.round_id
       WHERE ledger.round_id = v_round.round_id
         AND (ledger.call_identity IS NULL
              OR ledger.run_id IS NOT NULL
              OR ledger.stage IS NOT NULL
              OR ledger.provider IS DISTINCT FROM 'openrouter'
              OR ledger.operation_id IS DISTINCT FROM 'openrouter.code_review'
              OR submission.status IS DISTINCT FROM 'accepted'
              OR submission.code_review_status IS DISTINCT FROM 'passed'
              OR submission.is_king IS DISTINCT FROM FALSE)
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS ledger
       WHERE ledger.round_id = v_round.round_id
       GROUP BY ledger.submission_id, ledger.call_identity
       HAVING pg_catalog.count(*) <> 3
          OR pg_catalog.count(*) FILTER (WHERE entry_kind = 'reservation') <> 1
          OR pg_catalog.count(*) FILTER (WHERE entry_kind = 'dispatch') <> 1
          OR pg_catalog.count(*) FILTER (
               WHERE entry_kind = 'settlement'
                 AND entry_doc ->> 'review_status' = 'passed'
             ) <> 1
     ) THEN
    RAISE EXCEPTION 'sep18 review ledger snapshot differs'
      USING ERRCODE = '55000';
  END IF;

  v_execution_seconds := 2 * pg_catalog.ceil(20.0 * 5 / 10)::BIGINT * 2760;
  v_scoring_seconds := 2 * pg_catalog.ceil(10.0 * 5 / 10)::BIGINT * 960;
  IF v_execution_seconds IS DISTINCT FROM 55200
     OR v_scoring_seconds IS DISTINCT FROM 9600
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'stage_1_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_1_start')::TIMESTAMPTZ
        )) IS DISTINCT FROM 57600::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'stage_1_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_1_start')::TIMESTAMPTZ
        )) < v_execution_seconds::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_1_close')::TIMESTAMPTZ
        )) IS DISTINCT FROM 10800::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_1_close')::TIMESTAMPTZ
        )) < v_scoring_seconds::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'stage_2_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_2_start')::TIMESTAMPTZ
        )) IS DISTINCT FROM 1::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_2_close')::TIMESTAMPTZ
        )) IS DISTINCT FROM 10800::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
          - (v_new_schedule ->> 'stage_2_close')::TIMESTAMPTZ
        )) < v_scoring_seconds::NUMERIC
     OR EXTRACT(EPOCH FROM (
          (v_new_schedule ->> 'publication_deadline')::TIMESTAMPTZ
          - (v_new_schedule ->> 'submission_cutoff')::TIMESTAMPTZ
        )) >= 86400 THEN
    RAISE EXCEPTION 'sep18 schedule capacity proof differs'
      USING ERRCODE = '55000';
  END IF;

  v_before := v_round.configuration_doc;
  IF v_before -> 'schedule' = v_old_schedule THEN
    UPDATE public.lab_arena_rounds
    SET configuration_doc = pg_catalog.jsonb_set(
      configuration_doc, '{schedule}', v_new_schedule, FALSE
    )
    WHERE round_id = 'arena-2026-09-18';
    IF NOT FOUND THEN
      RAISE EXCEPTION 'sep18 schedule update lost its target'
        USING ERRCODE = '55000';
    END IF;
  END IF;

  SELECT configuration_doc INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18';
  IF (v_after - 'schedule') IS DISTINCT FROM (v_before - 'schedule')
     OR v_after -> 'schedule' IS DISTINCT FROM v_new_schedule
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = 'arena-2026-09-18') IS DISTINCT FROM 4::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-18') IS DISTINCT FROM 12::BIGINT THEN
    RAISE EXCEPTION 'sep18 schedule repair changed an unapproved field'
      USING ERRCODE = '55000';
  END IF;
END;
$sep18_schedule$;

ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;

DO $verify_sep18_schedule$
BEGIN
  IF NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-18'
         AND configuration_doc #>> '{schedule,stage_1_close}' =
             '2026-09-18T16:30:01Z'
         AND configuration_doc #>> '{schedule,publication_deadline}' =
             '2026-09-18T22:30:04Z'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'sep18 schedule repair verification failed'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_sep18_schedule$;

COMMIT;
