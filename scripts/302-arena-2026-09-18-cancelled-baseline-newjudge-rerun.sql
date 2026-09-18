-- One-time cancelled-baseline rerun for arena-2026-09-18.
-- The final statement invokes the private prepare RPC. It performs one atomic
-- archive/reopen only after every sealed terminal check succeeds.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $requires_sep18_cancelled_rerun302$
DECLARE
  v_cost_schema JSONB;
  v_publication_definition TEXT;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_per_icp_cost_schema_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_icp_cost_state(text,text,integer)'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL THEN
    RAISE EXCEPTION 'apply current Arena schema before rerun302';
  END IF;
  SELECT public.lab_arena_per_icp_cost_schema_v1() INTO v_cost_schema;
  IF v_cost_schema IS DISTINCT FROM
       '{"schema_version":"leadpoet.lab_arena.per_icp_cost_schema.v1",'
       '"version":289,"policy":"successful_calls_per_icp_v1"}'::JSONB THEN
    RAISE EXCEPTION 'Sep18 rerun302 per-ICP cost schema differs';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)'::pg_catalog.regprocedure
  ) INTO v_publication_definition;
  IF pg_catalog.strpos(
       v_publication_definition, 'lab_arena_per_icp_null_final_score_v1'
     ) = 0
     OR pg_catalog.strpos(v_publication_definition, 'v_accepted_count') = 0
     OR pg_catalog.strpos(v_publication_definition, 'v_is_baseline') = 0 THEN
    RAISE EXCEPTION 'apply migration 292 before rerun302';
  END IF;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure
     ), 'sha256'), 'hex') IS DISTINCT FROM
       'b83c135347d6ea80fb3e21ebbc5bbf992afc87a5e04ed6404a4831707ef3d03d'
     OR (SELECT rolname FROM pg_catalog.pg_proc proc
         JOIN pg_catalog.pg_roles role ON role.oid = proc.proowner
         WHERE proc.oid = 'public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure)
          IS DISTINCT FROM 'lab_arena_owner'
     OR (SELECT prosecdef FROM pg_catalog.pg_proc
         WHERE oid = 'public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure)
          IS DISTINCT FROM TRUE
     OR (SELECT proacl::TEXT FROM pg_catalog.pg_proc
         WHERE oid = 'public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure)
          IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  THEN
    RAISE EXCEPTION 'apply exact migration 301 before rerun302';
  END IF;
END;
$requires_sep18_cancelled_rerun302$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Keep the normal driver and add one exact fresh all-participant score namespace.
DO $pin_rerun302_scoring_definition$
BEGIN
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure
     ), 'sha256'), 'hex') NOT IN (
       '2b5adfe57fd9094bb0b8498cb57d3395ace7e1f911ca88c8187f46ee4a332191', '3950c211176bc3097934414b5e913b01c0087e12110cb7d132189d18292819cd'
     ) THEN
    RAISE EXCEPTION 'current integrity scoring definition differs';
  END IF;
END;
$pin_rerun302_scoring_definition$;

CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2(p_round_id text, p_stage smallint, p_work_items jsonb)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_item JSONB;
  v_scored public.lab_arena_runs;
  v_cache public.lab_arena_judgment_cache;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_reused INTEGER := 0;
  v_status TEXT;
  v_rerun295_retained_validated BOOLEAN := FALSE;
  v_rerun297_retained_validated BOOLEAN := FALSE;
  v_rerun298_retained_validated BOOLEAN := FALSE;
  v_rerun299_retained_validated BOOLEAN := FALSE;
  v_rerun300_retained_validated BOOLEAN := FALSE;
BEGIN
  IF p_stage IS NULL OR p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_work_items) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scoring_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1' THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required' USING ERRCODE = '22023';
  END IF;
  v_expected := 'stage' || p_stage::TEXT || '_closed';
  v_next := 'stage' || p_stage::TEXT || '_scoring';
  IF v_round.status <> v_expected THEN
    IF EXISTS (
      SELECT 1 FROM public.lab_arena_runs
      WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score'
        AND judgment_cache_key IS NOT NULL
    ) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  v_generation := v_round.stage_generation + 1;
  IF p_round_id = 'arena-2026-09-18'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun300archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T19:15:00Z","final_scoring_close":"2026-09-19T09:00:03Z","publication_deadline":"2026-09-19T09:00:04Z","stage_1_close":"2026-09-18T22:30:01Z","stage_1_scoring_close":"2026-09-19T03:45:01Z","stage_1_start":"2026-09-18T19:15:01Z","stage_2_close":"2026-09-19T03:45:03Z","stage_2_start":"2026-09-19T03:45:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
       OR public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE THEN
      RAISE EXCEPTION 'lab_arena_rerun302_frozen_state_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;
  FOR v_item IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_work_items)
  LOOP
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'scored_run_id', '') = ''
       OR COALESCE(v_item ->> 'submission_id', '') !~ '^[A-Za-z0-9._:-]{1,64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'icp_position') IS DISTINCT FROM 'number'
       OR (v_item ->> 'icp_position')::INTEGER NOT BETWEEN (CASE p_stage WHEN 1 THEN 0 ELSE 10 END) AND (CASE p_stage WHEN 1 THEN 9 ELSE 19 END)
       OR pg_catalog.char_length(COALESCE(v_item ->> 'output_ref', '')) NOT BETWEEN 1 AND 1024
       OR COALESCE(v_item ->> 'judgment_cache_key', '') !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'judgment_input_hash', '') !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'judgment_scope_doc') IS DISTINCT FROM 'object'
       OR v_item #>> '{judgment_scope_doc,cache_key}' IS DISTINCT FROM
          v_item ->> 'judgment_cache_key'
       OR v_item #>> '{judgment_scope_doc,scoring_input_hash}' IS DISTINCT FROM
          v_item ->> 'judgment_input_hash'
       OR v_item #>> '{judgment_scope_doc,round_id}' IS DISTINCT FROM p_round_id
       OR v_item #>> '{judgment_scope_doc,network_name}' IS DISTINCT FROM
          v_round.arena_network_name
       OR (v_item #>> '{judgment_scope_doc,netuid}')::INTEGER IS DISTINCT FROM
          v_round.arena_netuid
       OR v_item #>> '{judgment_scope_doc,integrity_policy}' IS DISTINCT FROM
          'arena_integrity_v1'
       OR v_item #>> '{judgment_scope_doc,evaluation_date}' IS DISTINCT FROM
          v_round.evaluation_date
       OR v_item #>> '{judgment_scope_doc,scorer_image_digest}' IS DISTINCT FROM
          v_round.configuration_doc ->> 'scorer_image_digest'
       OR v_item #>> '{judgment_scope_doc,scorer_image_reference}' IS DISTINCT FROM
          v_round.configuration_doc ->> 'scorer_image_reference'
       OR (
         v_item ? 'reuse_cache_key'
         AND v_item ->> 'reuse_cache_key' IS DISTINCT FROM
             v_item ->> 'judgment_cache_key'
       )
       OR (
         NOT (v_item ? 'reuse_cache_key')
         AND (
           pg_catalog.jsonb_typeof(v_item -> 'judgment_group_leader')
             IS DISTINCT FROM 'boolean'
           OR pg_catalog.jsonb_typeof(v_item -> 'judgment_group_miner_hotkeys')
             IS DISTINCT FROM 'array'
         )
       ) THEN
      RAISE EXCEPTION 'lab_arena_scoring_item_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_scored FROM public.lab_arena_runs
    WHERE run_id = v_item ->> 'scored_run_id'
      AND round_id = p_round_id
      AND stage = p_stage
      AND submission_id = v_item ->> 'submission_id'
      AND icp_position = (v_item ->> 'icp_position')::INTEGER
      AND output_ref = v_item ->> 'output_ref'
      AND kind = 'execute' AND status = 'accepted';
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_scored_run_invalid' USING ERRCODE = '22023';
    END IF;
    v_cache := NULL;
    IF v_item ? 'reuse_cache_key' THEN
      SELECT * INTO v_cache FROM public.lab_arena_judgment_cache
      WHERE cache_key = v_item ->> 'reuse_cache_key';
      IF NOT FOUND OR v_cache.scoring_input_hash IS DISTINCT FROM
          v_item ->> 'judgment_input_hash' THEN
        RAISE EXCEPTION 'lab_arena_judgment_cache_invalid' USING ERRCODE = '22023';
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs source
        WHERE source.run_id = v_cache.source_score_run_id
          AND source.kind = 'score' AND source.status = 'accepted'
          AND source.runner_hotkey = v_cache.source_runner_hotkey
          AND source.scored_run_id = v_cache.source_scored_run_id
      ) THEN
        RAISE EXCEPTION 'lab_arena_judgment_cache_source_invalid'
          USING ERRCODE = '22023';
      END IF;
    END IF;
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score' ||
      CASE WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun300archive')
           THEN ':rerun302'
           WHEN p_round_id = 'arena-2026-09-17'
                  AND v_scored.submission_id = 'baseline-2026-09-17'
                  AND EXISTS (
                    SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-17-rerun285archive'
                  )
           THEN ':rerun286'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND v_scored.submission_id = 'baseline-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun299archive')
           THEN ':rerun300'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND v_scored.submission_id = 'baseline-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun298archive')
           THEN ':rerun299'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND v_scored.submission_id = 'baseline-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun297archive')
           THEN ':rerun298'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND v_scored.submission_id = 'baseline-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun295archive')
           THEN ':rerun297'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND v_scored.submission_id = 'baseline-2026-09-18'
                  AND EXISTS (
                    SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun291archive'
                  )
           THEN ':rerun295' ELSE '' END;

    IF NOT (p_round_id = 'arena-2026-09-18'
       AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                   WHERE round_id = 'arena-2026-09-18-rerun300archive')) THEN
    IF p_round_id = 'arena-2026-09-18'
       AND v_scored.submission_id <> 'baseline-2026-09-18'
       AND (EXISTS (
         SELECT 1 FROM public.lab_arena_submissions baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.source_ref =
             'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz'
       ) OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.kind = 'execute'
           AND baseline.assignment_id LIKE '%:rerun300'
       )) THEN
      -- Exact configuration canonical SHA256:
      -- 451d158c6c2e88a607553b06e1f9114c53351d30c20bf26b3dd340e334be0b72.
      -- JSONB equality avoids confusing canonical JSON with jsonb::text hashing.
      -- Hash the two immutable seals once per invocation under the round lock.
      IF NOT v_rerun300_retained_validated THEN
      IF p_stage NOT IN (1, 2)
         OR v_round.configuration_doc IS DISTINCT FROM
           $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T17:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T18:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T17:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
         OR public.lab_arena_sep18_rerun299_archive_valid300_v1() IS NOT TRUE
         OR public.lab_arena_sep18_published_rerun300_nonbaseline_valid_v1() IS NOT TRUE
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_submissions baseline
           WHERE baseline.round_id = p_round_id
             AND baseline.submission_id = 'baseline-2026-09-18'
             AND baseline.source_ref =
               'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz'
             AND baseline.source_size_bytes = 660992
             AND baseline.submission_doc ->> 'source_sha256' =
               '6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677'
             AND baseline.submission_doc ->> 'source_commit' =
               'f8e54592a8ad339c2798935bb909244fd586a8db'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun300_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_rerun300_retained_validated := TRUE;
      END IF;
      IF v_scored.assignment_id IS DISTINCT FROM
           p_round_id || ':' || v_scored.submission_id || ':' ||
             p_stage::TEXT || ':' || v_scored.icp_position::TEXT
         OR v_scored.run_id IS DISTINCT FROM v_scored.assignment_id || ':1'
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs retained
           WHERE retained.run_id = v_assignment || ':1'
             AND retained.assignment_id = v_assignment
             AND retained.round_id = p_round_id
             AND retained.submission_id = v_scored.submission_id
             AND retained.miner_hotkey = v_scored.miner_hotkey
             AND retained.stage = p_stage
             AND retained.icp_position = v_scored.icp_position
             AND retained.attempt = 1
             AND retained.kind = 'score' AND retained.status = 'accepted'
             AND retained.terminal_cause = 'accepted'
             AND retained.scored_run_id = v_scored.run_id
             -- Score output is the sealed judge output, not the execution output.
             AND pg_catalog.char_length(retained.output_ref) BETWEEN 1 AND 1024
             AND retained.judgment_cache_key = v_item ->> 'judgment_cache_key'
             AND retained.judgment_input_hash = v_item ->> 'judgment_input_hash'
             AND retained.judgment_scope_doc = v_item -> 'judgment_scope_doc'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun300_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_created := v_created + 1;
      v_reused := v_reused + 1;
      CONTINUE;
    END IF;
    IF p_round_id = 'arena-2026-09-18'
       AND v_scored.submission_id <> 'baseline-2026-09-18'
       AND (EXISTS (
         SELECT 1 FROM public.lab_arena_submissions baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.source_ref =
             'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun299.tar.gz'
       ) OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.kind = 'execute'
           AND baseline.assignment_id LIKE '%:rerun299'
       )) THEN
      -- Exact configuration canonical SHA256:
      -- 434029d004766779d2c8967d064edfec429c9080f4398b2a7cfef2a3d7299a7b.
      -- JSONB equality avoids confusing canonical JSON with jsonb::text hashing.
      -- Hash the two immutable seals once per invocation under the round lock.
      IF NOT v_rerun299_retained_validated THEN
      IF p_stage NOT IN (1, 2)
         OR v_round.configuration_doc IS DISTINCT FROM
           $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T15:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T18:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T15:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
         OR public.lab_arena_sep18_rerun298_archive_valid299_v1() IS NOT TRUE
         OR public.lab_arena_sep18_published_rerun299_nonbaseline_valid_v1() IS NOT TRUE
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_submissions baseline
           WHERE baseline.round_id = p_round_id
             AND baseline.submission_id = 'baseline-2026-09-18'
             AND baseline.source_ref =
               'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun299.tar.gz'
             AND baseline.source_size_bytes = 656257
             AND baseline.submission_doc ->> 'source_sha256' =
               '8bb0939feb7ce14048ad44f8879dcec45155394433f214471ac8644442308869'
             AND baseline.submission_doc ->> 'source_commit' =
               '9e0bf3b014675f9c1d5c7c5e4433e6688a3f8637'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun299_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_rerun299_retained_validated := TRUE;
      END IF;
      IF v_scored.assignment_id IS DISTINCT FROM
           p_round_id || ':' || v_scored.submission_id || ':' ||
             p_stage::TEXT || ':' || v_scored.icp_position::TEXT
         OR v_scored.run_id IS DISTINCT FROM v_scored.assignment_id || ':1'
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs retained
           WHERE retained.run_id = v_assignment || ':1'
             AND retained.assignment_id = v_assignment
             AND retained.round_id = p_round_id
             AND retained.submission_id = v_scored.submission_id
             AND retained.miner_hotkey = v_scored.miner_hotkey
             AND retained.stage = p_stage
             AND retained.icp_position = v_scored.icp_position
             AND retained.attempt = 1
             AND retained.kind = 'score' AND retained.status = 'accepted'
             AND retained.terminal_cause = 'accepted'
             AND retained.scored_run_id = v_scored.run_id
             -- Score output is the sealed judge output, not the execution output.
             AND pg_catalog.char_length(retained.output_ref) BETWEEN 1 AND 1024
             AND retained.judgment_cache_key = v_item ->> 'judgment_cache_key'
             AND retained.judgment_input_hash = v_item ->> 'judgment_input_hash'
             AND retained.judgment_scope_doc = v_item -> 'judgment_scope_doc'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun299_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_created := v_created + 1;
      v_reused := v_reused + 1;
      CONTINUE;
    END IF;
    IF p_round_id = 'arena-2026-09-18'
       AND v_scored.submission_id <> 'baseline-2026-09-18'
       AND (EXISTS (
         SELECT 1 FROM public.lab_arena_submissions baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.source_ref =
             'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun298.tar.gz'
       ) OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.kind = 'execute'
           AND baseline.assignment_id LIKE '%:rerun298'
       )) THEN
      -- Exact configuration canonical SHA256:
      -- 1a2f89106c9a6434411fba9492f6cfec9113bfe027299670e177c786b861f1df.
      -- JSONB equality avoids confusing canonical JSON with jsonb::text hashing.
      -- Hash the two immutable seals once per invocation under the round lock.
      IF NOT v_rerun298_retained_validated THEN
      IF p_stage NOT IN (1, 2)
         OR v_round.configuration_doc IS DISTINCT FROM
           $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T13:00:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T16:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T13:00:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
         OR public.lab_arena_sep18_rerun297_archive_valid298_v1() IS NOT TRUE
         OR public.lab_arena_sep18_published_rerun298_nonbaseline_valid_v1() IS NOT TRUE
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_submissions baseline
           WHERE baseline.round_id = p_round_id
             AND baseline.submission_id = 'baseline-2026-09-18'
             AND baseline.source_ref =
               'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun298.tar.gz'
             AND baseline.source_size_bytes = 650579
             AND baseline.submission_doc ->> 'source_sha256' =
               '61b197f882b949412510c2fe56a30ab62463a73af83c7818f5f3be2b27cd569d'
             AND baseline.submission_doc ->> 'source_commit' =
               '9c47691176fef9be8a9e0b5c92b11b0180bb2e2a'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun298_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_rerun298_retained_validated := TRUE;
      END IF;
      IF v_scored.assignment_id IS DISTINCT FROM
           p_round_id || ':' || v_scored.submission_id || ':' ||
             p_stage::TEXT || ':' || v_scored.icp_position::TEXT
         OR v_scored.run_id IS DISTINCT FROM v_scored.assignment_id || ':1'
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs retained
           WHERE retained.run_id = v_assignment || ':1'
             AND retained.assignment_id = v_assignment
             AND retained.round_id = p_round_id
             AND retained.submission_id = v_scored.submission_id
             AND retained.miner_hotkey = v_scored.miner_hotkey
             AND retained.stage = p_stage
             AND retained.icp_position = v_scored.icp_position
             AND retained.attempt = 1
             AND retained.kind = 'score' AND retained.status = 'accepted'
             AND retained.terminal_cause = 'accepted'
             AND retained.scored_run_id = v_scored.run_id
             -- Score output is the sealed judge output, not the execution output.
             AND pg_catalog.char_length(retained.output_ref) BETWEEN 1 AND 1024
             AND retained.judgment_cache_key = v_item ->> 'judgment_cache_key'
             AND retained.judgment_input_hash = v_item ->> 'judgment_input_hash'
             AND retained.judgment_scope_doc = v_item -> 'judgment_scope_doc'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun298_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_created := v_created + 1;
      v_reused := v_reused + 1;
      CONTINUE;
    END IF;
    IF p_round_id = 'arena-2026-09-18'
       AND v_scored.submission_id <> 'baseline-2026-09-18'
       AND (EXISTS (
         SELECT 1 FROM public.lab_arena_submissions baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.source_ref =
             'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun297.tar.gz'
       ) OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.kind = 'execute'
           AND baseline.assignment_id LIKE '%:rerun297'
       )) THEN
      -- Exact configuration canonical SHA256:
      -- 6d35fa0be248dcd3406ace5bb2cdcbfb0c9f6e7a363a3beb6de64763e025da3f.
      -- JSONB equality avoids confusing canonical JSON with jsonb::text hashing.
      -- Hash the two immutable seals once per invocation under the round lock.
      IF NOT v_rerun297_retained_validated THEN
      IF p_stage NOT IN (1, 2)
         OR v_round.configuration_doc IS DISTINCT FROM
           $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T11:00:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T16:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T11:00:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
         OR public.lab_arena_sep18_rerun295_archive_valid297_v1() IS NOT TRUE
         OR public.lab_arena_sep18_published_rerun297_nonbaseline_valid_v1() IS NOT TRUE
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_submissions baseline
           WHERE baseline.round_id = p_round_id
             AND baseline.submission_id = 'baseline-2026-09-18'
             AND baseline.source_ref =
               'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun297.tar.gz'
             AND baseline.source_size_bytes = 646135
             AND baseline.submission_doc ->> 'source_sha256' =
               '581804245cfc14c9b01b1fdee38cfa932005932e39b7070aaaa74e306325df84'
             AND baseline.submission_doc ->> 'source_commit' =
               'fc7e64ca5fe92381990a8034feb55650c4cab361'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun297_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_rerun297_retained_validated := TRUE;
      END IF;
      IF v_scored.assignment_id IS DISTINCT FROM
           p_round_id || ':' || v_scored.submission_id || ':' ||
             p_stage::TEXT || ':' || v_scored.icp_position::TEXT
         OR v_scored.run_id IS DISTINCT FROM v_scored.assignment_id || ':1'
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs retained
           WHERE retained.run_id = v_assignment || ':1'
             AND retained.assignment_id = v_assignment
             AND retained.round_id = p_round_id
             AND retained.submission_id = v_scored.submission_id
             AND retained.miner_hotkey = v_scored.miner_hotkey
             AND retained.stage = p_stage
             AND retained.icp_position = v_scored.icp_position
             AND retained.attempt = 1
             AND retained.kind = 'score' AND retained.status = 'accepted'
             AND retained.terminal_cause = 'accepted'
             AND retained.scored_run_id = v_scored.run_id
             -- Score output is the sealed judge output, not the execution output.
             AND pg_catalog.char_length(retained.output_ref) BETWEEN 1 AND 1024
             AND retained.judgment_cache_key = v_item ->> 'judgment_cache_key'
             AND retained.judgment_input_hash = v_item ->> 'judgment_input_hash'
             AND retained.judgment_scope_doc = v_item -> 'judgment_scope_doc'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun297_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_created := v_created + 1;
      v_reused := v_reused + 1;
      CONTINUE;
    END IF;
    IF p_round_id = 'arena-2026-09-18'
       AND v_scored.submission_id <> 'baseline-2026-09-18'
       AND (EXISTS (
         SELECT 1 FROM public.lab_arena_submissions baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.source_ref =
             'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun295.tar.gz'
       ) OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs baseline
         WHERE baseline.round_id = p_round_id
           AND baseline.submission_id = 'baseline-2026-09-18'
           AND baseline.kind = 'execute'
           AND baseline.assignment_id LIKE '%:rerun295'
       )) THEN
      -- Exact configuration canonical SHA256:
      -- a583f20ca7573721258e27b5f80fd9128511a01a4ab50203f234ea8c69db3fb9.
      -- JSONB equality avoids confusing canonical JSON with jsonb::text hashing.
      -- Hash the two immutable seals once per invocation under the round lock.
      IF NOT v_rerun295_retained_validated THEN
      IF p_stage NOT IN (1, 2)
         OR v_round.configuration_doc IS DISTINCT FROM
           $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T09:00:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T16:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T09:00:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
         OR public.lab_arena_sep18_rerun291_archive_valid295_v1() IS NOT TRUE
         OR public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1() IS NOT TRUE
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_submissions baseline
           WHERE baseline.round_id = p_round_id
             AND baseline.submission_id = 'baseline-2026-09-18'
             AND baseline.source_ref =
               'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun295.tar.gz'
             AND baseline.source_size_bytes = 641923
             AND baseline.submission_doc ->> 'source_sha256' =
               '1ba24d1abac3849b8900238cb3f8f2ced3aab41fbe8f30064c0dcbeefc9c3cd3'
             AND baseline.submission_doc ->> 'source_commit' =
               'e21d29698edb60e6b5635f5f27e5a0058709210f'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun295_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_rerun295_retained_validated := TRUE;
      END IF;
      IF v_scored.assignment_id IS DISTINCT FROM
           p_round_id || ':' || v_scored.submission_id || ':' ||
             p_stage::TEXT || ':' || v_scored.icp_position::TEXT
         OR v_scored.run_id IS DISTINCT FROM v_scored.assignment_id || ':1'
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs retained
           WHERE retained.run_id = v_assignment || ':1'
             AND retained.assignment_id = v_assignment
             AND retained.round_id = p_round_id
             AND retained.submission_id = v_scored.submission_id
             AND retained.miner_hotkey = v_scored.miner_hotkey
             AND retained.stage = p_stage
             AND retained.icp_position = v_scored.icp_position
             AND retained.attempt = 1
             AND retained.kind = 'score' AND retained.status = 'accepted'
             AND retained.terminal_cause = 'accepted'
             AND retained.scored_run_id = v_scored.run_id
             -- Score output is the sealed judge output, not the execution output.
             AND pg_catalog.char_length(retained.output_ref) BETWEEN 1 AND 1024
             AND retained.judgment_cache_key = v_item ->> 'judgment_cache_key'
             AND retained.judgment_input_hash = v_item ->> 'judgment_input_hash'
             AND retained.judgment_scope_doc = v_item -> 'judgment_scope_doc'
         ) THEN
        RAISE EXCEPTION 'lab_arena_rerun295_retained_score_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_created := v_created + 1;
      v_reused := v_reused + 1;
      CONTINUE;
    END IF;
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      result_doc, terminal_cause, output_ref, judgment_cache_key,
      judgment_input_hash, judgment_scope_doc, judgment_group_leader,
      judgment_group_miner_hotkeys, judgment_cache_source_run_id
    ) VALUES (
      v_assignment || ':1', v_assignment, p_round_id, v_scored.submission_id,
      v_scored.miner_hotkey, p_stage, v_scored.icp_position, 1, v_status,
      v_generation, 'score', v_scored.run_id,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE
        pg_catalog.jsonb_build_object(
          'schema_version', 'leadpoet.lab_arena.cached_run_result.v1',
          'terminal_status', 'accepted',
          'cache_key', v_cache.cache_key,
          'source_score_run_id', v_cache.source_score_run_id
        ) END,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE 'accepted' END,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE
        v_cache.evidence_doc ->> 'source_output_ref' END,
      v_item ->> 'judgment_cache_key', v_item ->> 'judgment_input_hash',
      v_item -> 'judgment_scope_doc',
      CASE WHEN v_cache.cache_key IS NULL THEN
        (v_item ->> 'judgment_group_leader')::BOOLEAN ELSE FALSE END,
      CASE WHEN v_cache.cache_key IS NULL THEN ARRAY(
        SELECT pg_catalog.jsonb_array_elements_text(
          v_item -> 'judgment_group_miner_hotkeys'
        )
      ) ELSE ARRAY[]::TEXT[] END,
      v_cache.source_score_run_id
    );
    v_created := v_created + 1;
    IF v_cache.cache_key IS NOT NULL THEN
      v_reused := v_reused + 1;
    END IF;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_next,
    'stage_generation', v_generation, 'assignments', v_created,
    'reused', v_reused
  );
END;
$function$
;

DROP TRIGGER IF EXISTS lab_arena_sep18_published_rerun300_score_namespace_guard
  ON public.lab_arena_runs;
DROP TRIGGER IF EXISTS lab_arena_sep18_published_rerun300_publication_guard
  ON public.lab_arena_rounds;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun302_score_namespace_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep18_cancelled_rerun302_score_namespace_guard$
BEGIN
  IF NEW.round_id = 'arena-2026-09-18'
     AND NEW.kind = 'score'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-18-rerun300archive'
     )
     AND NEW.assignment_id IS DISTINCT FROM
           NEW.round_id || ':' || NEW.submission_id || ':' ||
           NEW.stage::TEXT || ':' || NEW.icp_position::TEXT ||
           ':score:rerun302' THEN
    RAISE EXCEPTION 'Sep18 rerun302 scoring requires exact all-participant namespace'
      USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$sep18_cancelled_rerun302_score_namespace_guard$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun302_score_namespace_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun302_score_namespace_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun302_score_namespace_guard
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep18_cancelled_rerun302_score_namespace_guard
  BEFORE INSERT ON public.lab_arena_runs
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep18_cancelled_rerun302_score_namespace_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep18_rerun300_cancelled_archive_valid302$
DECLARE
  v_terminal_round CONSTANT JSONB :=
    $terminal_round${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-18/benchmark.json","cancel_reason":"execution_incomplete:stage1:10","champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T17:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T18:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T17:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-17T00:02:22.766685+00:00","effective_reward_epoch":25245,"evaluation_date":"2026-09-18","finalists":null,"icp_set_date":"2026-09-17","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":false,"miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","source_ref":"arena/arena-2026-09-18/sources/sub-c4eb33e444937ca4fcad1ced15dd62d1.tar.gz","source_size_bytes":174296,"submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"is_king":false,"miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","source_ref":"arena/arena-2026-09-18/sources/sub-c35b779803958d95b427e850a441a19d.tar.gz","source_size_bytes":174296,"submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"is_king":false,"miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","source_ref":"arena/arena-2026-09-18/sources/sub-e0fc3e14a60953684473c1969f4a9b8b.tar.gz","source_size_bytes":174296,"submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"is_king":false,"miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","source_ref":"arena/arena-2026-09-18/sources/sub-e079970e02a303e6c3b10f5155b3c33b.tar.gz","source_size_bytes":152187,"submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"submission_id":"baseline-2026-09-18"}],"promotion_doc":null,"promotion_required":true,"publication_doc":null,"published_at":null,"reward_activated_at":"2026-09-18T05:37:45.691659+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25245,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-18T05:37:36Z","reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEYCIQDJwTZJ8yamTivZSsioBaRoPZ4qAGZPvgOUewMP10thJQIhAMp8mFD4mC6IAx5rHcYkgd+CwiskdVFnIjJJJOlizpAP"}},"reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","rewards_enabled":true,"round_id":"arena-2026-09-18","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":null,"stage2_scoring_plan_doc":null,"stage3_scoring_plan_doc":null,"stage_generation":46,"status":"cancelled","status_generation":62,"updated_at":"2026-09-18T18:30:37.348596+00:00"}$terminal_round$::JSONB;
  v_terminal_baseline CONSTANT JSONB :=
    $terminal_baseline${"accepted_at":"2026-09-18T00:00:06.644863+00:00","code_review_attempts":0,"code_review_claim":null,"code_review_doc":null,"code_review_expires_at":null,"code_review_started_at":null,"code_review_status":"pending","consent":{"public_rerun":true},"created_at":"2026-09-18T00:00:06.512576+00:00","frozen_at":"2026-09-18T00:00:06.889287+00:00","is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","owner_block_hash":null,"owner_block_number":null,"owner_coldkey":null,"rejection_rule":null,"replaced_by_submission_id":null,"replaces_submission_id":null,"round_id":"arena-2026-09-18","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"status":"frozen","submission_doc":{"consent":{"public_rerun":true},"is_king":true,"source_commit":"f8e54592a8ad339c2798935bb909244fd586a8db","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_sha256":"6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677","source_size_bytes":660992},"submission_id":"baseline-2026-09-18","updated_at":"2026-09-18T00:00:06.889532+00:00"}$terminal_baseline$::JSONB;
  v_archived_derived JSONB;
  v_archived_derived_hash TEXT;
  v_archive_configuration JSONB;
  v_expected_archive JSONB;
  v_baseline_runs_hash TEXT;
  v_baseline_ledger_hash TEXT;
  v_score_runs_hash TEXT;
  v_score_ledger_hash TEXT;
BEGIN
  SELECT configuration_doc -> 'archived_execution_judgments'
  INTO v_archived_derived
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18-rerun300archive';
  v_archived_derived_hash := 'sha256:' || pg_catalog.encode(
    extensions.digest(COALESCE(v_archived_derived, 'null'::JSONB)::TEXT, 'sha256'),
    'hex'
  );
  v_archive_configuration := (v_terminal_round -> 'configuration_doc') ||
    pg_catalog.jsonb_build_object(
      'round_id', 'arena-2026-09-18-rerun300archive',
      'mode', 'shadow', 'rewards_enabled', FALSE,
      'archived_reward_basis_hash', v_terminal_round -> 'reward_basis_hash',
      'archived_effective_reward_epoch', v_terminal_round -> 'effective_reward_epoch',
      'archived_reward_activated_at', v_terminal_round -> 'reward_activated_at',
      'archived_execution_judgments', v_archived_derived
    );
  v_expected_archive := v_terminal_round || pg_catalog.jsonb_build_object(
    'round_id', 'arena-2026-09-18-rerun300archive',
    'status', 'cancelled', 'configuration_doc', v_archive_configuration,
    'rewards_enabled', FALSE, 'reward_basis_hash', NULL,
    'effective_reward_epoch', NULL, 'reward_activated_at', NULL,
    'cancel_reason', 'authorized_sep18_rerun300_cancelled_archive'
  );
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_baseline_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = 'arena-2026-09-18-rerun300archive'
    AND submission_id = 'baseline-2026-09-18-rerun300archive';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_baseline_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-18-rerun300archive'
    AND submission_id = 'baseline-2026-09-18-rerun300archive';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_score_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = 'arena-2026-09-18-rerun300archive'
    AND kind = 'score';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_score_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-18-rerun300archive'
    AND submission_id <> 'baseline-2026-09-18-rerun300archive';
  RETURN (SELECT pg_catalog.to_jsonb(row_value)
          FROM public.lab_arena_rounds AS row_value
          WHERE round_id = 'arena-2026-09-18-rerun300archive')
           IS NOT DISTINCT FROM v_expected_archive
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = 'arena-2026-09-18-rerun300archive') = 5
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_submissions archived
      WHERE archived.submission_id = 'baseline-2026-09-18-rerun300archive'
        AND archived.round_id = 'arena-2026-09-18-rerun300archive'
        AND (pg_catalog.to_jsonb(archived) - 'round_id' - 'submission_id') =
            (v_terminal_baseline - 'round_id' - 'submission_id')
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.lab_arena_submissions archived
      WHERE archived.round_id = 'arena-2026-09-18-rerun300archive'
        AND archived.submission_id <> 'baseline-2026-09-18-rerun300archive'
        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_submissions active
          WHERE active.round_id = 'arena-2026-09-18'
            AND active.submission_id = pg_catalog.left(
              archived.submission_id,
              pg_catalog.length(archived.submission_id) - pg_catalog.length(':r300archive')
            )
            AND (pg_catalog.to_jsonb(active) - 'round_id' - 'submission_id') =
                (pg_catalog.to_jsonb(archived) - 'round_id' - 'submission_id')
        )
    )
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-18-rerun300archive'
           AND submission_id = 'baseline-2026-09-18-rerun300archive') =
          24
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-18-rerun300archive'
           AND submission_id = 'baseline-2026-09-18-rerun300archive') =
          6693
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-18-rerun300archive'
           AND kind = 'score') = 80
    AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-18-rerun300archive'
           AND submission_id <> 'baseline-2026-09-18-rerun300archive') =
          318
    AND v_baseline_runs_hash = 'sha256:1f34d7b3e5dc4def1e9cbf8aed4a6abd816bea38b727bb0b546fa5b5e0ac3ac7'
    AND v_baseline_ledger_hash = 'sha256:bc4d54de589f8418478f0542c87a2f4fee351303e0d15e1e4704c7f51aa4d1f4'
    AND v_score_runs_hash = 'sha256:b77e27e2647b6826a4e9aa0489e4e68dd5657920d4b1c4e0479b2ced823250f5'
    AND v_score_ledger_hash = 'sha256:e1b67994c341386e8e47a4b6962a604960c5571d914a2cba01014410bc67297b'
    AND pg_catalog.jsonb_typeof(v_archived_derived) = 'array'
    AND pg_catalog.jsonb_array_length(v_archived_derived) =
          80
    AND v_archived_derived_hash = 'sha256:5e136618c577ccded99c10f0cb691242cdb3b397959a98b6a06b5a43e188cb72'
    AND public.lab_arena_sep18_rerun299_archive_valid300_v1() IS TRUE;
END;
$sep18_rerun300_cancelled_archive_valid302$;
ALTER FUNCTION public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $sep18_cancelled_rerun302_active_frozen_valid$
DECLARE
  v_submissions_hash TEXT;
  v_execute_hash TEXT;
  v_execute_ledger_hash TEXT;
  v_orphan_ledger_hash TEXT;
BEGIN
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), '' ORDER BY submission_id), ''),
    'sha256'), 'hex') INTO v_submissions_hash
  FROM public.lab_arena_submissions AS row_value
  WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'per_icp_score' - 'qualification_doc' - 'updated_at')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''),
    'sha256'), 'hex') INTO v_execute_hash
  FROM public.lab_arena_runs AS row_value
  WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18'
    AND kind = 'execute';
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), '' ORDER BY entry_id), ''),
    'sha256'), 'hex') INTO v_execute_ledger_hash
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18'
    AND EXISTS (SELECT 1 FROM public.lab_arena_runs active_execute
      WHERE active_execute.run_id = row_value.run_id
        AND active_execute.round_id = 'arena-2026-09-18'
        AND active_execute.kind = 'execute');
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(COALESCE(
    pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'), 'hex'), '' ORDER BY entry_id), ''),
    'sha256'), 'hex') INTO v_orphan_ledger_hash
  FROM public.lab_arena_ledger AS row_value
  WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18'
    AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs any_run
      WHERE any_run.run_id = row_value.run_id);
  RETURN (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
          WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18') =
           4
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
          WHERE round_id = 'arena-2026-09-18' AND submission_id <> 'baseline-2026-09-18'
            AND kind = 'execute') = 80
     AND pg_catalog.jsonb_array_length((SELECT configuration_doc ->
          'archived_execution_judgments' FROM public.lab_arena_rounds
          WHERE round_id = 'arena-2026-09-18-rerun300archive')) =
          80
     AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs
          WHERE round_id = 'arena-2026-09-18' AND kind = 'score'
            AND assignment_id NOT LIKE '%:score:rerun302')
     AND v_submissions_hash = 'sha256:40318c429446ccfc8e2ad48a1bec6ce640e526f453062bd9e4302c0ca5d8a221'
     AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger row_value
          WHERE round_id = 'arena-2026-09-18'
            AND submission_id <> 'baseline-2026-09-18'
            AND EXISTS (SELECT 1 FROM public.lab_arena_runs active_execute
              WHERE active_execute.run_id = row_value.run_id
                AND active_execute.round_id = 'arena-2026-09-18'
                AND active_execute.kind = 'execute')) =
          2568
     AND v_execute_hash = 'sha256:dd4f2fd3003fa310a626896eafc962d11556aa554bff5cdc3a61870beb090704'
     AND v_execute_ledger_hash = 'sha256:eb141277f1ad8fc2bab99c8a2b9fa7aa3c627217ed9c22ea773c53b4a1eac0ad'
     AND v_orphan_ledger_hash = 'sha256:24edb97d881c1446ef87eb3ae4b7e93d660af72ddb40da4459a215a3535b7718';
END;
$sep18_cancelled_rerun302_active_frozen_valid$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun302_v1(
  p_source_size_bytes BIGINT,
  p_source_sha256 TEXT,
  p_source_commit TEXT,
  p_bank_sha256 TEXT,
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $prepare_sep18_cancelled_rerun302$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-18';
  v_baseline_id CONSTANT TEXT := 'baseline-2026-09-18';
  v_archive_round_id CONSTANT TEXT := 'arena-2026-09-18-rerun300archive';
  v_archive_submission_id CONSTANT TEXT :=
    'baseline-2026-09-18-rerun300archive';
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz';
  v_terminal_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz';
  v_bank_sha256 CONSTANT TEXT :=
    '6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91';
  v_terminal_round CONSTANT JSONB :=
    $terminal_round${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-18/benchmark.json","cancel_reason":"execution_incomplete:stage1:10","champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T17:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T18:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T17:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-17T00:02:22.766685+00:00","effective_reward_epoch":25245,"evaluation_date":"2026-09-18","finalists":null,"icp_set_date":"2026-09-17","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":false,"miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","source_ref":"arena/arena-2026-09-18/sources/sub-c4eb33e444937ca4fcad1ced15dd62d1.tar.gz","source_size_bytes":174296,"submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"is_king":false,"miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","source_ref":"arena/arena-2026-09-18/sources/sub-c35b779803958d95b427e850a441a19d.tar.gz","source_size_bytes":174296,"submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"is_king":false,"miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","source_ref":"arena/arena-2026-09-18/sources/sub-e0fc3e14a60953684473c1969f4a9b8b.tar.gz","source_size_bytes":174296,"submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"is_king":false,"miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","source_ref":"arena/arena-2026-09-18/sources/sub-e079970e02a303e6c3b10f5155b3c33b.tar.gz","source_size_bytes":152187,"submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"submission_id":"baseline-2026-09-18"}],"promotion_doc":null,"promotion_required":true,"publication_doc":null,"published_at":null,"reward_activated_at":"2026-09-18T05:37:45.691659+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25245,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-18T05:37:36Z","reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEYCIQDJwTZJ8yamTivZSsioBaRoPZ4qAGZPvgOUewMP10thJQIhAMp8mFD4mC6IAx5rHcYkgd+CwiskdVFnIjJJJOlizpAP"}},"reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","rewards_enabled":true,"round_id":"arena-2026-09-18","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":null,"stage2_scoring_plan_doc":null,"stage3_scoring_plan_doc":null,"stage_generation":46,"status":"cancelled","status_generation":62,"updated_at":"2026-09-18T18:30:37.348596+00:00"}$terminal_round$::JSONB;
  v_terminal_baseline CONSTANT JSONB :=
    $terminal_baseline${"accepted_at":"2026-09-18T00:00:06.644863+00:00","code_review_attempts":0,"code_review_claim":null,"code_review_doc":null,"code_review_expires_at":null,"code_review_started_at":null,"code_review_status":"pending","consent":{"public_rerun":true},"created_at":"2026-09-18T00:00:06.512576+00:00","frozen_at":"2026-09-18T00:00:06.889287+00:00","is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","owner_block_hash":null,"owner_block_number":null,"owner_coldkey":null,"rejection_rule":null,"replaced_by_submission_id":null,"replaces_submission_id":null,"round_id":"arena-2026-09-18","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"status":"frozen","submission_doc":{"consent":{"public_rerun":true},"is_king":true,"source_commit":"f8e54592a8ad339c2798935bb909244fd586a8db","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_sha256":"6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677","source_size_bytes":660992},"submission_id":"baseline-2026-09-18","updated_at":"2026-09-18T00:00:06.889532+00:00"}$terminal_baseline$::JSONB;
  v_forward_schedule CONSTANT JSONB :=
    $forward_schedule${"benchmark_deadline":"2026-09-18T19:15:00Z","final_scoring_close":"2026-09-19T09:00:03Z","publication_deadline":"2026-09-19T09:00:04Z","stage_1_close":"2026-09-18T22:30:01Z","stage_1_scoring_close":"2026-09-19T03:45:01Z","stage_1_start":"2026-09-18T19:15:01Z","stage_2_close":"2026-09-19T03:45:03Z","stage_2_start":"2026-09-19T03:45:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$forward_schedule$::JSONB;
  v_terminal_derived JSONB;
  v_terminal_derived_hash TEXT;
  v_round public.lab_arena_rounds%ROWTYPE;
  v_baseline public.lab_arena_submissions%ROWTYPE;
  v_new_configuration JSONB;
  v_new_participants JSONB;
  v_archive_configuration JSONB;
  v_expected_round JSONB;
  v_expected_baseline JSONB;
  v_protected_before JSONB;
  v_protected_after JSONB;
  v_unsettled_cost_count BIGINT;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
  v_count BIGINT;
BEGIN
  PERFORM pg_catalog.set_config('lock_timeout', '5s', TRUE);
  PERFORM pg_catalog.set_config('statement_timeout', '120s', TRUE);
  IF p_source_size_bytes IS DISTINCT FROM 660992
     OR p_source_sha256 IS DISTINCT FROM '6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677'
     OR p_source_commit IS DISTINCT FROM 'f8e54592a8ad339c2798935bb909244fd586a8db'
     OR p_bank_sha256 IS DISTINCT FROM v_bank_sha256
     OR p_forward_schedule IS DISTINCT FROM v_forward_schedule THEN
    RAISE EXCEPTION 'Sep18 rerun302 source, bank, or schedule differs'
      USING ERRCODE = '22023';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-18-cancelled-rerun302', 0)
  );
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
  SELECT * INTO STRICT v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  SELECT * INTO STRICT v_baseline FROM public.lab_arena_submissions
  WHERE round_id = v_round_id AND submission_id = v_baseline_id FOR UPDATE;

  v_new_configuration := pg_catalog.jsonb_set(
    pg_catalog.jsonb_set(
      pg_catalog.jsonb_set(
        v_terminal_round -> 'configuration_doc',
        '{schedule}', v_forward_schedule, FALSE
      ),
      '{scorer_image_digest}',
      pg_catalog.to_jsonb('sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f'::TEXT), FALSE
    ),
    '{scorer_image_reference}',
    pg_catalog.to_jsonb('493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f'::TEXT), FALSE
  );
  v_new_participants := v_terminal_round -> 'participants';

  -- Response-loss replay is read-only and verifies both namespaces.
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds
             WHERE round_id = v_archive_round_id)
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND submission_id = v_baseline_id
         AND assignment_id LIKE '%:rerun302'
     ) THEN
    IF public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE
       OR v_round.configuration_doc IS DISTINCT FROM v_new_configuration
       OR v_round.participants IS DISTINCT FROM v_new_participants
       OR v_round.status NOT IN (
         'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
         'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
         'stage2_judged', 'scored', 'published'
       )
       OR v_baseline.source_ref IS DISTINCT FROM v_source_ref
       OR v_baseline.source_size_bytes IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_ref' IS DISTINCT FROM v_source_ref
       OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
            IS DISTINCT FROM p_source_size_bytes
       OR v_baseline.submission_doc ->> 'source_sha256'
            IS DISTINCT FROM p_source_sha256
       OR v_baseline.submission_doc ->> 'source_commit'
            IS DISTINCT FROM p_source_commit
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id AND submission_id = v_baseline_id
             AND kind = 'execute' AND assignment_id LIKE '%:rerun302') <> 20
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND kind = 'score'
           AND assignment_id NOT LIKE '%:score:rerun302') THEN
      RAISE EXCEPTION 'Sep18 rerun302 replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'round_id', v_round_id,
      'baseline_execute_assignments', 20,
      'execute_namespace', 'rerun302',
      'score_namespace', 'score:rerun302',
      'source_size_bytes', p_source_size_bytes,
      'source_sha256', p_source_sha256,
      'source_commit', p_source_commit
    );
  END IF;

  IF (p_forward_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ <=
       pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'Sep18 rerun302 admission window has closed'
      USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.count(*) INTO v_unsettled_cost_count
  FROM public.lab_arena_submissions AS submission
  CROSS JOIN (VALUES ('execute'::TEXT), ('score'::TEXT)) AS kinds(kind)
  CROSS JOIN LATERAL (
    SELECT public.lab_arena__successful_call_cost_state(
      submission.submission_id, kinds.kind, NULL
    ) AS state
  ) AS costs
  WHERE submission.round_id = v_round_id
    AND (
      COALESCE((costs.state ->> 'inflight_calls')::BIGINT, -1) <> 0
      OR COALESCE((costs.state ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
    );
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex')
  INTO v_runs_hash FROM public.lab_arena_runs AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT 'sha256:' || pg_catalog.encode(extensions.digest(
    COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      (pg_catalog.to_jsonb(row_value) - 'round_id' - 'submission_id')::TEXT,
      'sha256'), 'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex')
  INTO v_ledger_hash FROM public.lab_arena_ledger AS row_value
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
    'run_id', run_id, 'per_icp_score', per_icp_score,
    'qualification_doc', qualification_doc
  ) ORDER BY run_id)
  INTO v_terminal_derived
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id AND submission_id <> v_baseline_id
    AND kind = 'execute';
  v_terminal_derived_hash := 'sha256:' || pg_catalog.encode(
    extensions.digest(COALESCE(v_terminal_derived, 'null'::JSONB)::TEXT, 'sha256'),
    'hex'
  );
  IF pg_catalog.to_jsonb(v_round) IS DISTINCT FROM v_terminal_round
     OR pg_catalog.to_jsonb(v_baseline) IS DISTINCT FROM v_terminal_baseline
     OR v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-09-18'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-17'
     OR v_round.benchmark_ref IS DISTINCT FROM
          'arena/arena-2026-09-18/benchmark.json'
     OR v_round.reward_basis_hash IS NULL
     OR v_round.reward_basis_doc IS NULL
     OR v_round.signing_key_doc IS NULL
     OR v_round.effective_reward_epoch IS NULL
     OR v_round.reward_activated_at IS NULL
     OR v_round.king_outcome IS DISTINCT FROM 'no_king'
     OR v_round.king_hotkey IS NOT NULL
     OR v_round.king_start_epoch IS DISTINCT FROM 0
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_baseline.source_ref IS DISTINCT FROM v_terminal_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM 660992
     OR v_baseline.submission_doc ->> 'source_ref'
          IS DISTINCT FROM v_terminal_source_ref
     OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
          IS DISTINCT FROM 660992
     OR (v_baseline.submission_doc ? 'source_sha256'
         AND v_baseline.submission_doc ->> 'source_sha256'
             IS DISTINCT FROM '6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677')
     OR (v_baseline.submission_doc ? 'source_commit'
         AND v_baseline.submission_doc ->> 'source_commit'
             IS DISTINCT FROM 'f8e54592a8ad339c2798935bb909244fd586a8db')
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <>
        24
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = v_round_id AND submission_id = v_baseline_id) <>
        6693
     OR v_runs_hash IS DISTINCT FROM 'sha256:1f34d7b3e5dc4def1e9cbf8aed4a6abd816bea38b727bb0b546fa5b5e0ac3ac7'
     OR v_ledger_hash IS DISTINCT FROM 'sha256:bc4d54de589f8418478f0542c87a2f4fee351303e0d15e1e4704c7f51aa4d1f4'
     OR pg_catalog.jsonb_array_length(v_terminal_derived) IS DISTINCT FROM
          80
     OR v_terminal_derived_hash IS DISTINCT FROM
          'sha256:5e136618c577ccded99c10f0cb691242cdb3b397959a98b6a06b5a43e188cb72'
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id
                  AND status IN ('pending', 'leased', 'submitted'))
     OR v_unsettled_cost_count <> 0
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds
                WHERE round_id = v_archive_round_id)
     OR public.lab_arena_sep18_published_rerun300_nonbaseline_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep18 cancelled terminal differs'
      USING ERRCODE = '55000';
  END IF;

  -- Preserve unrelated state and the miner execution/cost preimage. The exact
  -- baseline history and prior score history move to the shadow archive.
  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY round_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_rounds AS row_value
      WHERE round_id <> v_round_id),
    'submissions', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY round_id, submission_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_submissions AS row_value
      WHERE round_id <> v_round_id),
    'runs', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_runs AS row_value
      WHERE round_id <> v_round_id),
    -- Bound billing preservation to rounds that the transition can affect or
    -- race with. Older ledger rows cannot match its exact mutation predicate.
    'ledger', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_ledger AS row_value
      WHERE round_id IN (v_round_id, 'arena-2026-09-19',
                         'arena-2026-09-17-rerun285archive')
        AND (
          round_id <> v_round_id
          OR EXISTS (
            SELECT 1 FROM public.lab_arena_runs execute_run
            WHERE execute_run.run_id = row_value.run_id
              AND execute_run.round_id = v_round_id
              AND execute_run.submission_id <> v_baseline_id
              AND execute_run.kind = 'execute'
          )
        )),
    'weights', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY network, netuid, epoch), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_accepted_weight_states AS row_value)
  ) INTO v_protected_before;

  v_archive_configuration := (v_terminal_round -> 'configuration_doc') ||
    pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id, 'mode', 'shadow', 'rewards_enabled', FALSE,
      'archived_reward_basis_hash', v_terminal_round -> 'reward_basis_hash',
      'archived_effective_reward_epoch', v_terminal_round -> 'effective_reward_epoch',
      'archived_reward_activated_at', v_terminal_round -> 'reward_activated_at',
      'archived_execution_judgments', v_terminal_derived
    );
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
  INSERT INTO public.lab_arena_rounds (round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers)
  SELECT round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    v_terminal_round || pg_catalog.jsonb_build_object(
      'round_id', v_archive_round_id, 'status', 'cancelled',
      'configuration_doc', v_archive_configuration, 'rewards_enabled', FALSE,
      'reward_basis_hash', NULL, 'effective_reward_epoch', NULL,
      'reward_activated_at', NULL,
      'cancel_reason', 'authorized_sep18_rerun300_cancelled_archive'
    )
  );
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    v_terminal_baseline || pg_catalog.jsonb_build_object(
      'submission_id', v_archive_submission_id, 'round_id', v_archive_round_id
    )
  )).*;
  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(source_row) || pg_catalog.jsonb_build_object(
      'submission_id', source_row.submission_id || ':r300archive',
      'round_id', v_archive_round_id
    )
  )).*
  FROM public.lab_arena_submissions source_row
  WHERE source_row.round_id = v_round_id
    AND source_row.submission_id <> v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 4 THEN
    RAISE EXCEPTION 'Sep18 rerun302 miner submission archive count differs';
  END IF;
  UPDATE public.lab_arena_runs
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 24 THEN
    RAISE EXCEPTION 'Sep18 rerun302 run archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger
  SET round_id = v_archive_round_id,
      submission_id = v_archive_submission_id
  WHERE round_id = v_round_id AND submission_id = v_baseline_id;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 6693 THEN
    RAISE EXCEPTION 'Sep18 rerun302 ledger archive count differs';
  END IF;
  UPDATE public.lab_arena_runs
  SET round_id = v_archive_round_id,
      submission_id = submission_id || ':r300archive'
  WHERE round_id = v_round_id
    AND submission_id <> v_baseline_id
    AND kind = 'score';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 80 THEN
    RAISE EXCEPTION 'Sep18 rerun302 prior score archive count differs';
  END IF;
  UPDATE public.lab_arena_ledger ledger
  SET round_id = v_archive_round_id,
      submission_id = ledger.submission_id || ':r300archive'
  WHERE ledger.round_id = v_round_id
    AND ledger.submission_id <> v_baseline_id
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_runs archived_score
      WHERE archived_score.run_id = ledger.run_id
        AND archived_score.round_id = v_archive_round_id
        AND archived_score.kind = 'score'
    );
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 318 THEN
    RAISE EXCEPTION 'Sep18 rerun302 prior score ledger archive count differs';
  END IF;
  UPDATE public.lab_arena_runs
  SET per_icp_score = NULL,
      qualification_doc = NULL
  WHERE round_id = v_round_id
    AND submission_id <> v_baseline_id
    AND kind = 'execute';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 80 THEN
    RAISE EXCEPTION 'Sep18 rerun302 miner judgment reset count differs';
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
    RAISE EXCEPTION 'Sep18 rerun302 baseline update differs';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = status_generation + 1,
      stage_generation = stage_generation + 1,
      configuration_doc = v_new_configuration,
      participants = v_new_participants,
      stage1_scoring_plan_doc = NULL,
      stage2_scoring_plan_doc = NULL,
      stage3_scoring_plan_doc = NULL,
      finalists = NULL,
      publication_doc = NULL,
      published_at = NULL,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round_id;
  FOR v_position IN 0 .. 19 LOOP
    v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
    v_assignment := v_round_id || ':' || v_baseline_id || ':' ||
      v_stage::TEXT || ':' || v_position::TEXT || ':rerun302';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      v_assignment || ':1', v_assignment, v_round_id, v_baseline_id,
      v_baseline.miner_hotkey, v_stage, v_position, 1, 'execute', 'pending',
      v_round.stage_generation + 1
    );
  END LOOP;
  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY round_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_rounds AS row_value
      WHERE round_id <> v_round_id AND round_id <> v_archive_round_id),
    'submissions', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY round_id, submission_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_submissions AS row_value
      WHERE round_id <> v_round_id AND round_id <> v_archive_round_id),
    'runs', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY run_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_runs AS row_value
      WHERE round_id <> v_round_id AND round_id <> v_archive_round_id),
    'ledger', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY entry_id), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_ledger AS row_value
      WHERE round_id IN (v_round_id, 'arena-2026-09-19',
                         'arena-2026-09-17-rerun285archive')
        AND round_id <> v_archive_round_id
        AND (
          round_id <> v_round_id
          OR EXISTS (
            SELECT 1 FROM public.lab_arena_runs execute_run
            WHERE execute_run.run_id = row_value.run_id
              AND execute_run.round_id = v_round_id
              AND execute_run.submission_id <> v_baseline_id
              AND execute_run.kind = 'execute'
          )
        )),
    'weights', (SELECT pg_catalog.jsonb_build_object(
      'count', pg_catalog.count(*), 'sha256', 'sha256:' || pg_catalog.encode(
        extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(
          extensions.digest(pg_catalog.to_jsonb(row_value)::TEXT, 'sha256'),
          'hex'), '' ORDER BY network, netuid, epoch), ''), 'sha256'), 'hex'))
      FROM public.lab_arena_accepted_weight_states AS row_value)
  ) INTO v_protected_after;
  v_expected_round := v_terminal_round || pg_catalog.jsonb_build_object(
    'status', 'stage1',
    'status_generation', (v_terminal_round ->> 'status_generation')::BIGINT + 1,
    'stage_generation', (v_terminal_round ->> 'stage_generation')::BIGINT + 1,
    'configuration_doc', v_new_configuration,
    'participants', v_new_participants,
    'stage1_scoring_plan_doc', NULL,
    'stage2_scoring_plan_doc', NULL,
    'stage3_scoring_plan_doc', NULL,
    'finalists', NULL, 'publication_doc', NULL,
    'published_at', NULL, 'cancel_reason', NULL
  );
  v_expected_baseline := v_terminal_baseline || pg_catalog.jsonb_build_object(
    'source_ref', v_source_ref,
    'source_size_bytes', p_source_size_bytes,
    'submission_doc', COALESCE(v_terminal_baseline -> 'submission_doc', '{}'::JSONB) ||
      pg_catalog.jsonb_build_object(
        'source_ref', v_source_ref,
        'source_size_bytes', p_source_size_bytes,
        'source_sha256', p_source_sha256,
        'source_commit', p_source_commit
      )
  );
  IF v_protected_after IS DISTINCT FROM v_protected_before
     OR public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
     OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE
     OR (SELECT pg_catalog.to_jsonb(row_value) - 'updated_at'
         FROM public.lab_arena_rounds AS row_value
         WHERE round_id = v_round_id) IS DISTINCT FROM
        (v_expected_round - 'updated_at')
     OR (SELECT pg_catalog.to_jsonb(row_value)
         FROM public.lab_arena_submissions AS row_value
         WHERE round_id = v_round_id AND submission_id = v_baseline_id)
        IS DISTINCT FROM v_expected_baseline
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND submission_id = v_baseline_id
           AND kind = 'execute' AND status = 'pending'
           AND output_ref IS NULL AND attempt = 1
           AND assignment_id LIKE '%:rerun302') <> 20
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
                WHERE round_id = v_round_id AND submission_id = v_baseline_id)
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.generate_series(0, 19) AS positions(icp_position)
       CROSS JOIN LATERAL (
         SELECT public.lab_arena__successful_icp_cost_state(
           v_round_id, v_baseline_id, positions.icp_position
         ) AS state
       ) AS costs
       WHERE COALESCE((costs.state ->> 'settled_microusd')::BIGINT, -1) <> 0
          OR COALESCE((costs.state ->> 'reserved_or_uncertain_microusd')::BIGINT, -1) <> 0
          OR COALESCE((costs.state ->> 'inflight_calls')::BIGINT, -1) <> 0
          OR COALESCE((costs.state ->> 'success_unresolved_microusd')::BIGINT, -1) <> 0
     )
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id AND kind = 'score')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round_id AND kind = 'execute'
                  AND submission_id <> v_baseline_id
                  AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL))
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger ledger
                WHERE ledger.round_id = v_round_id
                  AND EXISTS (SELECT 1 FROM public.lab_arena_runs score_run
                    WHERE score_run.run_id = ledger.run_id
                      AND score_run.kind = 'score')) THEN
    RAISE EXCEPTION 'Sep18 rerun302 atomic preservation differs'
      USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared', 'round_id', v_round_id,
    'baseline_execute_assignments', 20,
    'archived_runs', 24,
    'archived_ledger_entries', 6693,
    'execute_namespace', 'rerun302',
    'score_namespace', 'score:rerun302',
    'openrouter_calls_per_icp', 200,
    'source_size_bytes', p_source_size_bytes,
    'source_sha256', p_source_sha256,
    'source_commit', p_source_commit
  );
END;
$prepare_sep18_cancelled_rerun302$;
ALTER FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun302_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun302_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun302_v1(
  BIGINT, TEXT, TEXT, TEXT, JSONB
) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun302_publication_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $sep18_cancelled_rerun302_publication_guard$
DECLARE
  v_terminal_round CONSTANT JSONB :=
    $terminal_round${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-18/benchmark.json","cancel_reason":"execution_incomplete:stage1:10","champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T17:30:00Z","final_scoring_close":"2026-09-18T22:30:03Z","publication_deadline":"2026-09-18T22:30:04Z","stage_1_close":"2026-09-18T18:30:01Z","stage_1_scoring_close":"2026-09-18T19:30:01Z","stage_1_start":"2026-09-18T17:30:01Z","stage_2_close":"2026-09-18T19:30:03Z","stage_2_start":"2026-09-18T19:30:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-17T00:02:22.766685+00:00","effective_reward_epoch":25245,"evaluation_date":"2026-09-18","finalists":null,"icp_set_date":"2026-09-17","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":false,"miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","source_ref":"arena/arena-2026-09-18/sources/sub-c4eb33e444937ca4fcad1ced15dd62d1.tar.gz","source_size_bytes":174296,"submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"is_king":false,"miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","source_ref":"arena/arena-2026-09-18/sources/sub-c35b779803958d95b427e850a441a19d.tar.gz","source_size_bytes":174296,"submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"is_king":false,"miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","source_ref":"arena/arena-2026-09-18/sources/sub-e0fc3e14a60953684473c1969f4a9b8b.tar.gz","source_size_bytes":174296,"submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"is_king":false,"miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","source_ref":"arena/arena-2026-09-18/sources/sub-e079970e02a303e6c3b10f5155b3c33b.tar.gz","source_size_bytes":152187,"submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"submission_id":"baseline-2026-09-18"}],"promotion_doc":null,"promotion_required":true,"publication_doc":null,"published_at":null,"reward_activated_at":"2026-09-18T05:37:45.691659+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25245,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-18T05:37:36Z","reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEYCIQDJwTZJ8yamTivZSsioBaRoPZ4qAGZPvgOUewMP10thJQIhAMp8mFD4mC6IAx5rHcYkgd+CwiskdVFnIjJJJOlizpAP"}},"reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","rewards_enabled":true,"round_id":"arena-2026-09-18","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":null,"stage2_scoring_plan_doc":null,"stage3_scoring_plan_doc":null,"stage_generation":46,"status":"cancelled","status_generation":62,"updated_at":"2026-09-18T18:30:37.348596+00:00"}$terminal_round$::JSONB;
  v_unsettled_cost_count BIGINT;
BEGIN
  IF NEW.round_id = 'arena-2026-09-18'
     AND OLD.status = 'scored' AND NEW.status = 'published'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun300archive') THEN
    SELECT pg_catalog.count(*) INTO v_unsettled_cost_count
    FROM public.lab_arena_submissions AS submission
    CROSS JOIN (VALUES ('execute'::TEXT), ('score'::TEXT)) AS kinds(kind)
    CROSS JOIN LATERAL (
      SELECT public.lab_arena__successful_call_cost_state(
        submission.submission_id, kinds.kind, NULL
      ) AS state
    ) AS costs
    WHERE submission.round_id = NEW.round_id
      AND (
        COALESCE((costs.state ->> 'inflight_calls')::BIGINT, -1) <> 0
        OR COALESCE((costs.state ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
      );
    IF NEW.configuration_doc ->> 'scorer_image_digest' IS DISTINCT FROM
         'sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f'
       OR NEW.configuration_doc ->> 'scorer_image_reference' IS DISTINCT FROM
         '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f'
       OR (NEW.configuration_doc - 'schedule' - 'scorer_image_digest' -
             'scorer_image_reference')
            IS DISTINCT FROM
            ((v_terminal_round -> 'configuration_doc') - 'schedule' -
             'scorer_image_digest' - 'scorer_image_reference')
       OR public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-18'
             AND kind = 'execute' AND assignment_id LIKE '%:rerun302') <> 20
       OR (SELECT pg_catalog.count(DISTINCT icp_position)
           FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id
             AND submission_id = 'baseline-2026-09-18'
             AND kind = 'execute' AND assignment_id LIKE '%:rerun302'
             AND per_icp_score IS NOT NULL) <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
           WHERE round_id = NEW.round_id AND kind = 'score'
           AND assignment_id NOT LIKE '%:score:rerun302'
       )
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs AS executed
         WHERE executed.round_id = NEW.round_id
           AND executed.kind = 'execute'
           AND executed.status = 'accepted'
           AND NOT EXISTS (
             SELECT 1 FROM public.lab_arena_runs AS judged
             WHERE judged.round_id = executed.round_id
               AND judged.submission_id = executed.submission_id
               AND judged.kind = 'score' AND judged.status = 'accepted'
               AND judged.scored_run_id = executed.run_id
               AND judged.assignment_id LIKE '%:score:rerun302'
           )
       )
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                  WHERE round_id = NEW.round_id
                    AND status IN ('pending', 'leased', 'submitted'))
       OR v_unsettled_cost_count <> 0
       THEN
      RAISE EXCEPTION 'Sep18 rerun302 publication conflicts with sealed source, cost, or completion state'
        USING ERRCODE = '55000';
    END IF;
  END IF;
  RETURN NEW;
END;
$sep18_cancelled_rerun302_publication_guard$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun302_publication_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun302_publication_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun302_publication_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep18_cancelled_rerun302_publication_guard
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_sep18_cancelled_rerun302_publication_guard_v1();

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
SELECT public.lab_arena_prepare_sep18_cancelled_rerun302_v1(660992,'6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677','f8e54592a8ad339c2798935bb909244fd586a8db','6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91','{"benchmark_deadline":"2026-09-18T19:15:00Z","final_scoring_close":"2026-09-19T09:00:03Z","publication_deadline":"2026-09-19T09:00:04Z","stage_1_close":"2026-09-18T22:30:01Z","stage_1_scoring_close":"2026-09-19T03:45:01Z","stage_1_start":"2026-09-18T19:15:01Z","stage_2_close":"2026-09-19T03:45:03Z","stage_2_start":"2026-09-19T03:45:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}'::jsonb);
COMMIT;
