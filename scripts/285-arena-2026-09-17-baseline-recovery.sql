-- Minimal same-round recovery for the cancelled recovery284 execution.
-- Installing this migration is inert. The private RPC performs one atomic move.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $requires_sep17_recovery285$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery284_archive_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL THEN
    RAISE EXCEPTION 'apply current Arena schema before recovery285';
  END IF;
END;
$requires_sep17_recovery285$;

-- The migration caller is a non-superuser member of lab_arena_owner. The new
-- function owner needs CREATE on its schema while ownership is transferred.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep17_baseline_recovery285_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_bank_sha256 TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep17_baseline_recovery285$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-17';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-17';
  v_archive_round_id CONSTANT TEXT := 'arena-2026-09-17-rerun284archive';
  v_archive_submission_id CONSTANT TEXT :=
    'baseline-2026-09-17-rerun284archive';
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery285.tar.gz';
  v_terminal_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery284.tar.gz';
  v_bank_sha256 CONSTANT TEXT :=
    '7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871';
  v_terminal_configuration CONSTANT JSONB :=
    $terminal_configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-17","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-17T14:00:00Z","final_scoring_close":"2026-09-18T00:20:00Z","publication_deadline":"2026-09-18T00:50:00Z","stage_1_close":"2026-09-17T20:00:00Z","stage_1_scoring_close":"2026-09-17T22:00:00Z","stage_1_start":"2026-09-17T14:00:01Z","stage_2_close":"2026-09-17T22:20:00Z","stage_2_start":"2026-09-17T22:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$terminal_configuration$::JSONB;
  v_terminal_participants CONSTANT JSONB :=
    $terminal_participants$[{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-17/sources/baseline-2026-09-17-recovery284.tar.gz","source_size_bytes":582191,"submission_id":"baseline-2026-09-17"}]$terminal_participants$::JSONB;
  v_forward_schedule CONSTANT JSONB :=
    $forward_schedule${"benchmark_deadline":"2026-09-17T18:30:00Z","final_scoring_close":"2026-09-18T02:20:00Z","publication_deadline":"2026-09-18T02:50:00Z","stage_1_close":"2026-09-17T22:00:00Z","stage_1_scoring_close":"2026-09-18T00:00:00Z","stage_1_start":"2026-09-17T18:30:01Z","stage_2_close":"2026-09-18T00:20:00Z","stage_2_start":"2026-09-18T00:00:01Z","submission_cutoff":"2026-09-17T00:00:00Z","submission_open":"2026-09-16T00:00:00Z"}$forward_schedule$::JSONB;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_archive_configuration JSONB;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_expected_round_after JSONB;
  v_expected_baseline_after JSONB;
  v_protected_before JSONB;
  v_protected_after JSONB;
  v_runs_before JSONB;
  v_runs_after JSONB;
  v_ledger_before JSONB;
  v_ledger_after JSONB;
  v_execute_cost JSONB;
  v_score_cost JSONB;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
  v_count BIGINT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  IF p_source_size_bytes IS NULL
     OR p_source_sha256 IS NULL
     OR p_source_commit IS NULL
     OR p_source_size_bytes NOT BETWEEN 1 AND 10485760
     OR p_source_sha256 !~ '^[0-9a-f]{64}$'
     OR p_source_commit !~ '^[0-9a-f]{40}$'
     OR p_bank_sha256 IS DISTINCT FROM v_bank_sha256
     OR p_forward_schedule IS DISTINCT FROM v_forward_schedule THEN
    RAISE EXCEPTION 'Sep17 recovery285 source, bank, or schedule differs'
      USING ERRCODE = '22023';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-17-baseline-recovery285', 0)
  );
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE round_id = v_round_id AND submission_id = v_baseline_id FOR UPDATE;

  -- A response-loss retry is inert. These three existing rows are the marker.
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds
             WHERE round_id = v_archive_round_id)
     OR v_baseline.source_ref = v_source_ref
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND assignment_id LIKE '%:rerun285'
     ) THEN
    v_new_configuration := pg_catalog.jsonb_set(
      v_terminal_configuration, '{schedule}', v_forward_schedule, FALSE
    );
    SELECT pg_catalog.jsonb_agg(
      CASE WHEN item ->> 'submission_id' = v_baseline_id THEN
        item || pg_catalog.jsonb_build_object(
          'source_ref', v_source_ref,
          'source_size_bytes', p_source_size_bytes
        ) ELSE item END ORDER BY ordinal
    ) INTO v_new_participants
    FROM pg_catalog.jsonb_array_elements(v_terminal_participants)
      WITH ORDINALITY AS entries(item, ordinal);
    IF public.lab_arena_sep17_recovery284_archive_valid_v1() IS NOT TRUE
       OR public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()
            IS NOT TRUE
       OR v_round.configuration_doc IS DISTINCT FROM v_new_configuration
       OR v_round.participants IS DISTINCT FROM v_new_participants
       OR v_round.status NOT IN (
         'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
         'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
         'stage2_judged', 'scored', 'published'
       )
       OR v_round.status_generation < 12 OR v_round.stage_generation < 11
       OR v_baseline.source_ref IS DISTINCT FROM v_source_ref
       OR v_baseline.source_size_bytes IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_ref' IS DISTINCT FROM v_source_ref
       OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
            IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_sha256'
            IS DISTINCT FROM p_source_sha256
       OR v_baseline.submission_doc ->> 'source_commit'
            IS DISTINCT FROM p_source_commit
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id = v_archive_round_id AND status = 'cancelled'
           AND rewards_enabled IS FALSE
           AND cancel_reason = 'authorized_sep17_recovery284_baseline_archive'
       )
       OR NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions
         WHERE round_id = v_archive_round_id
           AND submission_id = v_archive_submission_id
           AND source_ref = v_terminal_source_ref
           AND source_size_bytes = 582191
           AND submission_doc ->> 'source_sha256' =
             'b5ca95d7c9c25650c5ecb04b05bde859cffa3fe9c6c7a6602e6a006d5e3dcd99'
           AND submission_doc ->> 'source_commit' =
             '91e8b95637c4cbc919f3150d6643985624dbb34a'
       )
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun285') <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND (
             assignment_id <> v_round_id || ':' || v_baseline_id || ':' ||
               stage::TEXT || ':' || icp_position::TEXT || ':rerun285'
             OR icp_position NOT BETWEEN 0 AND 19
             OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           )
       )
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = v_archive_round_id
             AND submission_id = v_archive_submission_id) <> 23
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_archive_round_id
             AND submission_id = v_archive_submission_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun284') <> 20
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
           WHERE round_id = v_archive_round_id
             AND submission_id = v_archive_submission_id) <>
            5404 THEN
      RAISE EXCEPTION 'Sep17 recovery285 replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_id', v_round_id,
      'baseline_execute_assignments', 20,
      'execute_namespace', 'rerun285',
      'openrouter_calls_per_icp', 200,
      'source_size_bytes', p_source_size_bytes,
      'source_sha256', p_source_sha256,
      'source_commit', p_source_commit
    );
  END IF;

  IF (p_forward_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ <=
       pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'Sep17 recovery285 admission window has closed'
      USING ERRCODE = '22023';
  END IF;

  SELECT public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'execute', NULL
  ) INTO v_execute_cost;
  SELECT public.lab_arena__successful_call_cost_state(
    v_baseline_id, 'score', NULL
  ) INTO v_score_cost;
  IF public.lab_arena_sep17_recovery284_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()
          IS NOT TRUE
     OR v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.cancel_reason IS DISTINCT FROM 'operator'
     OR v_round.status_generation IS DISTINCT FROM 11
     OR v_round.stage_generation IS DISTINCT FROM 10
     OR v_round.evaluation_date IS DISTINCT FROM '2026-09-17'
     OR v_round.icp_set_date IS DISTINCT FROM DATE '2026-09-16'
     OR v_round.benchmark_ref IS DISTINCT FROM
          'arena/arena-2026-09-17/benchmark.json'
     OR v_round.configuration_doc IS DISTINCT FROM v_terminal_configuration
     OR v_round.participants IS DISTINCT FROM v_terminal_participants
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_baseline.status IS DISTINCT FROM 'frozen'
     OR v_baseline.is_king IS NOT TRUE
     OR v_baseline.source_ref IS DISTINCT FROM v_terminal_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM 582191
     OR v_baseline.submission_doc ->> 'source_sha256' IS DISTINCT FROM
          'b5ca95d7c9c25650c5ecb04b05bde859cffa3fe9c6c7a6602e6a006d5e3dcd99'
     OR v_baseline.submission_doc ->> 'source_commit' IS DISTINCT FROM
          '91e8b95637c4cbc919f3150d6643985624dbb34a'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <> 23
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND assignment_id LIKE '%:rerun284') <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND status = 'accepted' AND output_ref IS NOT NULL) <> 8
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND (kind <> 'execute' OR attempt NOT IN (1, 2)
           OR assignment_id <> v_round_id || ':' || v_baseline_id || ':' ||
              stage::TEXT || ':' || icp_position::TEXT || ':rerun284'
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           OR status IN ('pending', 'leased', 'submitted'))
     )
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id AND submission_id <> v_baseline_id)
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND submission_id <> v_baseline_id) <> 8
     OR COALESCE((v_execute_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_execute_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((v_score_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds
                WHERE round_id = v_archive_round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE round_id = v_archive_round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_archive_round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
                WHERE round_id = v_archive_round_id) THEN
    RAISE EXCEPTION 'Sep17 recovery284 terminal state differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id), '[]'::JSONB)
      FROM public.lab_arena_rounds AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND round_id <> v_round_id),
    'submissions', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id, submission_id), '[]'::JSONB)
      FROM public.lab_arena_submissions AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'runs', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id, run_id), '[]'::JSONB)
      FROM public.lab_arena_runs AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'ledger', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY entry_id), '[]'::JSONB)
      FROM public.lab_arena_ledger AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT (round_id = v_round_id AND submission_id = v_baseline_id)),
    'authority278', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery278_authority AS row_value
      WHERE round_id = v_round_id),
    'audit278', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery278_audit AS row_value
      WHERE round_id = v_round_id),
    'authority282', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery282_authority AS row_value
      WHERE round_id = v_round_id),
    'audit282', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery282_audit AS row_value
      WHERE round_id = v_round_id),
    'authority283', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery283_authority AS row_value
      WHERE round_id = v_round_id),
    'audit283', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery283_audit AS row_value
      WHERE round_id = v_round_id),
    'authority284', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery284_authority AS row_value
      WHERE round_id = v_round_id),
    'audit284', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery284_audit AS row_value
      WHERE round_id = v_round_id)
  ) INTO v_protected_before;
  SELECT COALESCE(pg_catalog.jsonb_agg(
    pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
    ORDER BY run_id), '[]'::JSONB)
  INTO v_runs_before FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT COALESCE(pg_catalog.jsonb_agg(
    pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
    ORDER BY entry_id), '[]'::JSONB)
  INTO v_ledger_before FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;

  v_archive_configuration := v_terminal_configuration ||
    pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id, 'rewards_enabled', FALSE
    );
  v_new_configuration := pg_catalog.jsonb_set(
    v_terminal_configuration, '{schedule}', v_forward_schedule, FALSE
  );
  IF (v_new_configuration - 'schedule') IS DISTINCT FROM
       (v_terminal_configuration - 'schedule')
     OR v_new_configuration #>> '{call_quotas,openrouter}' <> '200'
     OR v_new_configuration #>> '{call_quotas,deepline}' <> '30'
     OR v_new_configuration #>> '{call_quotas,scrapingdog}' <> '30' THEN
    RAISE EXCEPTION 'Sep17 recovery285 changed frozen configuration';
  END IF;
  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = v_baseline_id THEN
      item || pg_catalog.jsonb_build_object(
        'source_ref', v_source_ref,
        'source_size_bytes', p_source_size_bytes
      ) ELSE item END ORDER BY ordinal
  ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_terminal_participants)
    WITH ORDINALITY AS entries(item, ordinal);

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  INSERT INTO public.lab_arena_rounds (
    round_id, status, status_generation, stage_generation, configuration_doc,
    rewards_enabled, participants, benchmark_ref, evaluation_date,
    stage1_scoring_plan_doc, stage2_scoring_plan_doc, finalists,
    publication_doc, king_outcome, king_hotkey, king_start_epoch,
    effective_reward_epoch, reward_basis_hash, reward_basis_doc,
    signing_key_doc, reward_activated_at, cancel_reason, published_at,
    created_at, updated_at, promotion_required, promotion_doc,
    baseline_promoted_at, icp_set_date, confirmation_bank_ref,
    confirmation_bank_hash, confirmation_cohort, stage3_scoring_plan_doc,
    champion_funding_frozen, champion_submission_id, champion_hotkey,
    champion_fallback_providers
  ) SELECT
    row_value.round_id, row_value.status, row_value.status_generation,
    row_value.stage_generation, row_value.configuration_doc,
    row_value.rewards_enabled, row_value.participants, row_value.benchmark_ref,
    row_value.evaluation_date, row_value.stage1_scoring_plan_doc,
    row_value.stage2_scoring_plan_doc, row_value.finalists,
    row_value.publication_doc, row_value.king_outcome, row_value.king_hotkey,
    row_value.king_start_epoch, row_value.effective_reward_epoch,
    row_value.reward_basis_hash, row_value.reward_basis_doc,
    row_value.signing_key_doc, row_value.reward_activated_at,
    row_value.cancel_reason, row_value.published_at, row_value.created_at,
    row_value.updated_at, row_value.promotion_required,
    row_value.promotion_doc, row_value.baseline_promoted_at,
    row_value.icp_set_date, row_value.confirmation_bank_ref,
    row_value.confirmation_bank_hash, row_value.confirmation_cohort,
    row_value.stage3_scoring_plan_doc, row_value.champion_funding_frozen,
    row_value.champion_submission_id, row_value.champion_hotkey,
    row_value.champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(v_round) || pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id,
      'status', 'cancelled',
      'configuration_doc', v_archive_configuration,
      'rewards_enabled', FALSE,
      'cancel_reason', 'authorized_sep17_recovery284_baseline_archive'
    )
  ) AS row_value;
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(v_baseline) || pg_catalog.jsonb_build_object(
      'submission_id', v_archive_submission_id,
      'round_id', v_archive_round_id
    )
  )).*;
  UPDATE public.lab_arena_runs
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 23 THEN
    RAISE EXCEPTION 'Sep17 recovery285 run archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 5404 THEN
    RAISE EXCEPTION 'Sep17 recovery285 ledger archive count differs';
  END IF;
  UPDATE public.lab_arena_submissions
  SET source_ref = v_source_ref,
      source_size_bytes = p_source_size_bytes,
      submission_doc = COALESCE(submission_doc, '{}'::JSONB) ||
        pg_catalog.jsonb_build_object(
          'source_ref', v_source_ref,
          'source_size_bytes', p_source_size_bytes,
          'source_sha256', p_source_sha256,
          'source_commit', p_source_commit
        )
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep17 recovery285 baseline update differs';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1', status_generation = 12, stage_generation = 11,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL,
      stage2_scoring_plan_doc = NULL,
      stage3_scoring_plan_doc = NULL,
      finalists = NULL,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun285';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'execute', 'pending', 11
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id), '[]'::JSONB)
      FROM public.lab_arena_rounds AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND round_id <> v_round_id AND round_id <> v_archive_round_id),
    'submissions', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id, submission_id), '[]'::JSONB)
      FROM public.lab_arena_submissions AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'runs', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY round_id, run_id), '[]'::JSONB)
      FROM public.lab_arena_runs AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'ledger', (SELECT COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.to_jsonb(row_value) ORDER BY entry_id), '[]'::JSONB)
      FROM public.lab_arena_ledger AS row_value
      WHERE (round_id LIKE 'arena-2026-09-17%'
         OR round_id LIKE 'arena-2026-09-18%')
        AND NOT ((round_id = v_round_id AND submission_id = v_baseline_id)
        OR (round_id = v_archive_round_id
          AND submission_id = v_archive_submission_id))),
    'authority278', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery278_authority AS row_value
      WHERE round_id = v_round_id),
    'audit278', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery278_audit AS row_value
      WHERE round_id = v_round_id),
    'authority282', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery282_authority AS row_value
      WHERE round_id = v_round_id),
    'audit282', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery282_audit AS row_value
      WHERE round_id = v_round_id),
    'authority283', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery283_authority AS row_value
      WHERE round_id = v_round_id),
    'audit283', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery283_audit AS row_value
      WHERE round_id = v_round_id),
    'authority284', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery284_authority AS row_value
      WHERE round_id = v_round_id),
    'audit284', (SELECT pg_catalog.to_jsonb(row_value)
      FROM public.lab_arena_sep17_baseline_recovery284_audit AS row_value
      WHERE round_id = v_round_id)
  ) INTO v_protected_after;
  SELECT COALESCE(pg_catalog.jsonb_agg(
    pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
    ORDER BY run_id), '[]'::JSONB)
  INTO v_runs_after FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_archive_round_id
    AND submission_id = v_archive_submission_id;
  SELECT COALESCE(pg_catalog.jsonb_agg(
    pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
    ORDER BY entry_id), '[]'::JSONB)
  INTO v_ledger_after FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_archive_round_id
    AND submission_id = v_archive_submission_id;
  v_expected_round_after := pg_catalog.to_jsonb(v_round) ||
    pg_catalog.jsonb_build_object(
      'status', 'stage1', 'status_generation', 12, 'stage_generation', 11,
      'configuration_doc', v_new_configuration,
      'participants', v_new_participants,
      'stage1_scoring_plan_doc', NULL,
      'stage2_scoring_plan_doc', NULL,
      'stage3_scoring_plan_doc', NULL,
      'finalists', NULL, 'cancel_reason', NULL
    );
  v_expected_baseline_after := pg_catalog.to_jsonb(v_baseline) ||
    pg_catalog.jsonb_build_object(
      'source_ref', v_source_ref,
      'source_size_bytes', p_source_size_bytes,
      'submission_doc', COALESCE(v_baseline.submission_doc, '{}'::JSONB) ||
        pg_catalog.jsonb_build_object(
          'source_ref', v_source_ref,
          'source_size_bytes', p_source_size_bytes,
          'source_sha256', p_source_sha256,
          'source_commit', p_source_commit
        )
    );
  IF v_protected_after IS DISTINCT FROM v_protected_before
     OR v_runs_after IS DISTINCT FROM v_runs_before
     OR v_ledger_after IS DISTINCT FROM v_ledger_before
     OR (SELECT pg_catalog.to_jsonb(row_value) - 'updated_at'
         FROM public.lab_arena_rounds AS row_value
         WHERE round_id = v_round_id) IS DISTINCT FROM
          (v_expected_round_after - 'updated_at')
     OR (SELECT pg_catalog.to_jsonb(row_value)
         FROM public.lab_arena_submissions AS row_value
         WHERE round_id = v_round_id AND submission_id = v_baseline_id)
          IS DISTINCT FROM v_expected_baseline_after
     OR (SELECT pg_catalog.to_jsonb(row_value) - ARRAY[
           'round_id', 'status', 'configuration_doc', 'rewards_enabled',
           'cancel_reason'
         ]::TEXT[] FROM public.lab_arena_rounds AS row_value
         WHERE round_id = v_archive_round_id) IS DISTINCT FROM
          (pg_catalog.to_jsonb(v_round) - ARRAY[
           'round_id', 'status', 'configuration_doc', 'rewards_enabled',
           'cancel_reason'
         ]::TEXT[])
     OR (SELECT pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id'
         FROM public.lab_arena_submissions AS row_value
         WHERE round_id = v_archive_round_id
           AND submission_id = v_archive_submission_id) IS DISTINCT FROM
          (pg_catalog.to_jsonb(v_baseline) - 'round_id' - 'submission_id')
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'pending'
           AND output_ref IS NULL AND attempt = 1
           AND stage_generation = 11) <> 20
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND assignment_id LIKE '%:rerun285') <> 20
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND (assignment_id <> v_round_id || ':' || v_baseline_id || ':' ||
             stage::TEXT || ':' || icp_position::TEXT || ':rerun285'
           OR run_id <> assignment_id || ':1'
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END)
     )
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
                WHERE round_id = v_round_id AND submission_id = v_baseline_id)
     OR public.lab_arena_sep17_recovery284_archive_valid_v1() IS NOT TRUE
     OR public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()
          IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep17 recovery285 atomic preservation differs'
      USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_runs', 23,
    'archived_ledger_entries', 5404,
    'preserved_nonparticipant_submissions', 8,
    'execute_namespace', 'rerun285',
    'openrouter_calls_per_icp', 200,
    'source_size_bytes', p_source_size_bytes,
    'source_sha256', p_source_sha256,
    'source_commit', p_source_commit
  );
END;
$prepare_sep17_baseline_recovery285$;

ALTER FUNCTION public.lab_arena_prepare_sep17_baseline_recovery285_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep17_baseline_recovery285_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep17_baseline_recovery285_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;

COMMIT;
