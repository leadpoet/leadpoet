-- One-time scoring-only recovery for the cancelled Sep18 rerun302.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';
DO $requires303$ BEGIN
 IF public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
    OR pg_catalog.to_regprocedure('public.lab_arena_open_scoring_v2(text,smallint,jsonb)') IS NULL THEN
  RAISE EXCEPTION 'exact rerun302 authority required before rerun303';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure),'sha256'),'hex') IS DISTINCT FROM 'b83c135347d6ea80fb3e21ebbc5bbf992afc87a5e04ed6404a4831707ef3d03d'
    OR (SELECT role.rolname FROM pg_catalog.pg_proc proc JOIN pg_catalog.pg_roles role ON role.oid=proc.proowner WHERE proc.oid='public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure) IS DISTINCT FROM 'lab_arena_owner'
    OR (SELECT prosecdef FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure) IS DISTINCT FROM TRUE
    OR (SELECT proacl::TEXT FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_provider_funding(text,text)'::pg_catalog.regprocedure) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'exact score payer boundary required before rerun303';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),'sha256'),'hex') NOT IN ('3950c211176bc3097934414b5e913b01c0087e12110cb7d132189d18292819cd','a22585895de7ea68950b53b65e7ab13bdfd94f2aac95f40a53c84cc49b99cc5b') THEN
  RAISE EXCEPTION 'rerun303 scoring definition preimage differs';
 END IF;
END $requires303$;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
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
                 WHERE round_id = 'arena-2026-09-18-rerun302archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
       OR public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE THEN
      RAISE EXCEPTION 'lab_arena_rerun303_frozen_state_invalid'
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
                    WHERE round_id = 'arena-2026-09-18-rerun302archive')
           THEN ':rerun303'
           WHEN p_round_id = 'arena-2026-09-18'
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
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun302_score_namespace_guard ON public.lab_arena_runs;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun302_publication_guard ON public.lab_arena_rounds;
CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun303_score_namespace_guard_v1() RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
BEGIN
 IF NEW.round_id='arena-2026-09-18' AND NEW.kind='score' AND EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun302archive')
    AND NEW.assignment_id IS DISTINCT FROM NEW.round_id||':'||NEW.submission_id||':'||NEW.stage::TEXT||':'||NEW.icp_position::TEXT||':score:rerun303' THEN
  RAISE EXCEPTION 'Sep18 rerun303 scoring requires exact namespace' USING ERRCODE='55000';
 END IF; RETURN NEW;
END $f$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun303_score_namespace_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun303_score_namespace_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun303_score_namespace_guard ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep18_cancelled_rerun303_score_namespace_guard BEFORE INSERT ON public.lab_arena_runs FOR EACH ROW EXECUTE FUNCTION public.lab_arena_sep18_cancelled_rerun303_score_namespace_guard_v1();
CREATE OR REPLACE FUNCTION public.lab_arena_sep18_rerun302_score_archive_valid303_v1() RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public,extensions AS $f$
DECLARE terminal CONSTANT JSONB=$j${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-18/benchmark.json","cancel_reason":"scoring_incomplete","champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T19:15:00Z","final_scoring_close":"2026-09-19T09:00:03Z","publication_deadline":"2026-09-19T09:00:04Z","stage_1_close":"2026-09-18T22:30:01Z","stage_1_scoring_close":"2026-09-19T03:45:01Z","stage_1_start":"2026-09-18T19:15:01Z","stage_2_close":"2026-09-19T03:45:03Z","stage_2_start":"2026-09-19T03:45:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-17T00:02:22.766685+00:00","effective_reward_epoch":25245,"evaluation_date":"2026-09-18","finalists":null,"icp_set_date":"2026-09-17","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":false,"miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","source_ref":"arena/arena-2026-09-18/sources/sub-c4eb33e444937ca4fcad1ced15dd62d1.tar.gz","source_size_bytes":174296,"submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"is_king":false,"miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","source_ref":"arena/arena-2026-09-18/sources/sub-c35b779803958d95b427e850a441a19d.tar.gz","source_size_bytes":174296,"submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"is_king":false,"miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","source_ref":"arena/arena-2026-09-18/sources/sub-e0fc3e14a60953684473c1969f4a9b8b.tar.gz","source_size_bytes":174296,"submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"is_king":false,"miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","source_ref":"arena/arena-2026-09-18/sources/sub-e079970e02a303e6c3b10f5155b3c33b.tar.gz","source_size_bytes":152187,"submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"submission_id":"baseline-2026-09-18"}],"promotion_doc":null,"promotion_required":true,"publication_doc":null,"published_at":null,"reward_activated_at":"2026-09-18T05:37:45.691659+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25245,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-18T05:37:36Z","reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEYCIQDJwTZJ8yamTivZSsioBaRoPZ4qAGZPvgOUewMP10thJQIhAMp8mFD4mC6IAx5rHcYkgd+CwiskdVFnIjJJJOlizpAP"}},"reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","rewards_enabled":true,"round_id":"arena-2026-09-18","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":{"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":1,"work_items":[{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:0:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:0:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:1:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:1:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:2:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:2:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:3:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:3:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:4:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:4:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:5:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:5:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:6:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:6:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:7:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:7:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:8:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:8:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:9:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:9:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:0:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:1:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:2:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:3:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:4:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:5:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:6:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:7:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:8:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:9:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:0:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:1:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:2:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:3:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:4:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:5:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:6:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:7:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:8:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:9:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:0:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:1:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:2:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:3:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:4:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:5:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:6:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:7:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:8:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:9:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:0:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:1:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:2:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:3:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:4:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:5:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:6:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:7:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:8:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:9:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"}],"zero_rows":[]},"stage2_scoring_plan_doc":null,"stage3_scoring_plan_doc":null,"stage_generation":50,"status":"cancelled","status_generation":66,"updated_at":"2026-09-18T20:39:41.569213+00:00"}$j$::JSONB; derived JSONB; archive_config JSONB; expected JSONB; run_hash TEXT; ledger_hash TEXT;
BEGIN
 SELECT configuration_doc->'archived_execution_judgments' INTO derived FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun302archive';
 archive_config := (terminal->'configuration_doc')||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun302archive','mode','shadow','rewards_enabled',FALSE,'archived_reward_basis_hash',terminal->'reward_basis_hash','archived_effective_reward_epoch',terminal->'effective_reward_epoch','archived_reward_activated_at',terminal->'reward_activated_at','archived_execution_judgments',derived);
 expected := terminal||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun302archive','status','cancelled','configuration_doc',archive_config,'rewards_enabled',FALSE,'reward_basis_hash',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,'cancel_reason','authorized_sep18_rerun302_score_archive');
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(r)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') INTO run_hash FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-18-rerun302archive' AND kind='score';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(l)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') INTO ledger_hash FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18-rerun302archive';
 RETURN (SELECT pg_catalog.to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id='arena-2026-09-18-rerun302archive') IS NOT DISTINCT FROM expected
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18-rerun302archive')=5
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions ar WHERE ar.round_id='arena-2026-09-18-rerun302archive' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions ac WHERE ac.round_id='arena-2026-09-18' AND ar.submission_id=ac.submission_id||':r302archive' AND (pg_catalog.to_jsonb(ar)-'round_id'-'submission_id')=(pg_catalog.to_jsonb(ac)-'round_id'-'submission_id')))
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18-rerun302archive' AND kind='score')=52
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18-rerun302archive' AND kind='score' AND status='accepted' AND terminal_cause='accepted')=47
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18-rerun302archive' AND kind='score' AND status='failed')=5
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger WHERE round_id='arena-2026-09-18-rerun302archive')=818
  AND run_hash='sha256:688ab31e2e73215859b582ce4192a01c3e7b42b763f62a9c1b6691da1073bcc4' AND ledger_hash='sha256:fcd151a18afdda7e8ce98e826b101dc77798ceaedb80c66328f4efba8cf1e2c0'
  AND pg_catalog.jsonb_typeof(derived)='array' AND pg_catalog.jsonb_array_length(derived)=100
  AND NOT EXISTS(SELECT 1 FROM pg_catalog.jsonb_array_elements(derived) AS archived(value) LEFT JOIN public.lab_arena_runs active ON active.round_id='arena-2026-09-18' AND active.kind='execute' AND active.run_id=archived.value->>'run_id' WHERE active.run_id IS NULL OR archived.value IS DISTINCT FROM pg_catalog.jsonb_build_object('run_id',active.run_id,'per_icp_score',NULL,'qualification_doc',NULL))
  AND public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS TRUE;
END $f$;
ALTER FUNCTION public.lab_arena_sep18_rerun302_score_archive_valid303_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_rerun302_score_archive_valid303_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public,extensions AS $f$
DECLARE sh TEXT; eh TEXT; lh TEXT;
BEGIN
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') INTO sh FROM public.lab_arena_submissions r WHERE round_id='arena-2026-09-18';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at')::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') INTO eh FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-18' AND kind='execute';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') INTO lh FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs score_run WHERE score_run.round_id='arena-2026-09-18' AND score_run.kind='score' AND score_run.run_id=l.run_id);
 RETURN (SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18') IS NOT DISTINCT FROM $c${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$c$::JSONB
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18')=5 AND sh='sha256:cb2219f19cae619e3017917f8654d68eed194718b8a93e40e56a4b8a8a6baccc'
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute' AND status='accepted' AND terminal_cause='accepted')=100
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute' AND (output_ref IS NULL OR status<>'accepted'))
  AND eh='sha256:fd4dfadfe47e5b5929dd8ea12661a243881af36d3254ff6492bcba564411835d' AND (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs score_run WHERE score_run.round_id='arena-2026-09-18' AND score_run.kind='score' AND score_run.run_id=l.run_id))=9996 AND lh='sha256:ee03d3cffd2e7744f9fb25b2188110674fa5445f363bc3d28e951bccd7e554d5'
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id='arena-2026-09-18' AND l.run_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-18' AND r.kind IN('execute','score') AND r.run_id=l.run_id))
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND assignment_id NOT LIKE '%:score:rerun303')
  AND public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS TRUE;
END $f$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun303_v1(p_ordered_icp_bank_hash TEXT,p_schedule JSONB) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public,extensions AS $f$
DECLARE terminal CONSTANT JSONB=$j${"arena_netuid":71,"arena_network_name":"finney","baseline_promoted_at":null,"benchmark_ref":"arena/arena-2026-09-18/benchmark.json","cancel_reason":"scoring_incomplete","champion_fallback_providers":[],"champion_funding_frozen":true,"champion_hotkey":null,"champion_submission_id":null,"configuration_doc":{"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T19:15:00Z","final_scoring_close":"2026-09-19T09:00:03Z","publication_deadline":"2026-09-19T09:00:04Z","stage_1_close":"2026-09-18T22:30:01Z","stage_1_scoring_close":"2026-09-19T03:45:01Z","stage_1_start":"2026-09-18T19:15:01Z","stage_2_close":"2026-09-19T03:45:03Z","stage_2_start":"2026-09-19T03:45:02Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10},"confirmation_bank_hash":null,"confirmation_bank_ref":null,"confirmation_cohort":null,"created_at":"2026-09-17T00:02:22.766685+00:00","effective_reward_epoch":25245,"evaluation_date":"2026-09-18","finalists":null,"icp_set_date":"2026-09-17","king_hotkey":null,"king_outcome":"no_king","king_start_epoch":0,"participants":[{"is_king":false,"miner_hotkey":"5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6","source_ref":"arena/arena-2026-09-18/sources/sub-c4eb33e444937ca4fcad1ced15dd62d1.tar.gz","source_size_bytes":174296,"submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"is_king":false,"miner_hotkey":"5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe","source_ref":"arena/arena-2026-09-18/sources/sub-c35b779803958d95b427e850a441a19d.tar.gz","source_size_bytes":174296,"submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"is_king":false,"miner_hotkey":"5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX","source_ref":"arena/arena-2026-09-18/sources/sub-e0fc3e14a60953684473c1969f4a9b8b.tar.gz","source_size_bytes":174296,"submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"is_king":false,"miner_hotkey":"5GQaNW8JJHnNRbhLzUbj8vhXustViJN5SQPrxA3oqoG6gG2u","source_ref":"arena/arena-2026-09-18/sources/sub-e079970e02a303e6c3b10f5155b3c33b.tar.gz","source_size_bytes":152187,"submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"submission_id":"baseline-2026-09-18"}],"promotion_doc":null,"promotion_required":true,"publication_doc":null,"published_at":null,"reward_activated_at":"2026-09-18T05:37:45.691659+00:00","reward_basis_doc":{"champion_reward_factor_ppm":1000000,"effective_reward_epoch":25245,"king_hotkey":"","king_outcome":"no_king","king_start_epoch":0,"published_at":"2026-09-18T05:37:36Z","reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.reward_basis.v1","signature":{"algorithm":"ECDSA_SHA_256","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","signature_b64":"MEYCIQDJwTZJ8yamTivZSsioBaRoPZ4qAGZPvgOUewMP10thJQIhAMp8mFD4mC6IAx5rHcYkgd+CwiskdVFnIjJJJOlizpAP"}},"reward_basis_hash":"sha256:58299055a86002684ce10cb59992a14c627bd0201bc9605017f7257a8b68a42a","rewards_enabled":true,"round_id":"arena-2026-09-18","signing_key_doc":{"algorithm":"ECDSA_SHA_256","key_spec":"ECC_NIST_P256","public_key_der_b64":"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9ASH6X2G6rxbVVZuNzvy12Wj0tMPvJ4HCdZTs2me0xUz/llK9Lp00EGdhWYSLhEEXy/YU0weeMMilC+u4jREzg==","public_key_hash":"sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a","schema_version":"leadpoet.lab_arena.signing_key.v1"},"stage1_scoring_plan_doc":{"round_id":"arena-2026-09-18","schema_version":"leadpoet.lab_arena.scoring_plan.v1","stage":1,"work_items":[{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:0:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:0:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:1:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:1:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:2:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:2:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:3:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:3:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:4:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:4:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:5:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:5:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:6:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:6:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:7:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:7:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:8:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:8:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:baseline-2026-09-18:1:9:rerun302:1.json","scored_run_id":"arena-2026-09-18:baseline-2026-09-18:1:9:rerun302:1","submission_id":"baseline-2026-09-18"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:0:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:1:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:2:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:3:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:4:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:5:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:6:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:7:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:8:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-c35b779803958d95b427e850a441a19d:1:9:1","submission_id":"sub-c35b779803958d95b427e850a441a19d"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:0:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:1:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:2:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:3:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:4:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:5:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:6:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:7:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:8:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-c4eb33e444937ca4fcad1ced15dd62d1:1:9:1","submission_id":"sub-c4eb33e444937ca4fcad1ced15dd62d1"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:0:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:1:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:2:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:3:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:4:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:5:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:6:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:7:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:8:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-e079970e02a303e6c3b10f5155b3c33b:1:9:1","submission_id":"sub-e079970e02a303e6c3b10f5155b3c33b"},{"icp_position":0,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:0:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:0:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":1,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:1:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:1:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":2,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:2:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:2:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":3,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:3:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:3:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":4,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:4:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:4:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":5,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:5:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:5:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":6,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:6:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:6:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":7,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:7:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:7:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":8,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:8:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:8:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"},{"icp_position":9,"output_ref":"arena/arena-2026-09-18/outputs/arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:9:1.json","scored_run_id":"arena-2026-09-18:sub-e0fc3e14a60953684473c1969f4a9b8b:1:9:1","submission_id":"sub-e0fc3e14a60953684473c1969f4a9b8b"}],"zero_rows":[]},"stage2_scoring_plan_doc":null,"stage3_scoring_plan_doc":null,"stage_generation":50,"status":"cancelled","status_generation":66,"updated_at":"2026-09-18T20:39:41.569213+00:00"}$j$::JSONB; baseline CONSTANT JSONB=$j${"accepted_at":"2026-09-18T00:00:06.644863+00:00","code_review_attempts":0,"code_review_claim":null,"code_review_doc":null,"code_review_expires_at":null,"code_review_started_at":null,"code_review_status":"pending","consent":{"public_rerun":true},"created_at":"2026-09-18T00:00:06.512576+00:00","frozen_at":"2026-09-18T00:00:06.889287+00:00","is_king":true,"miner_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","owner_block_hash":null,"owner_block_number":null,"owner_coldkey":null,"rejection_rule":null,"replaced_by_submission_id":null,"replaces_submission_id":null,"round_id":"arena-2026-09-18","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_size_bytes":660992,"status":"frozen","submission_doc":{"consent":{"public_rerun":true},"is_king":true,"source_commit":"f8e54592a8ad339c2798935bb909244fd586a8db","source_ref":"arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun300.tar.gz","source_sha256":"6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677","source_size_bytes":660992},"submission_id":"baseline-2026-09-18","updated_at":"2026-09-18T00:00:06.889532+00:00"}$j$::JSONB; nr public.lab_arena_rounds%ROWTYPE; nb public.lab_arena_submissions%ROWTYPE; derived JSONB; archive_config JSONB; cnt BIGINT; active_count BIGINT; unsettled BIGINT; before_other JSONB; after_other JSONB; pre_sh TEXT; pre_eh TEXT; pre_lh TEXT; pre_rlh TEXT; pre_srh TEXT; pre_slh TEXT;
BEGIN
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun302archive') THEN
  IF public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE OR public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE THEN RAISE EXCEPTION 'existing rerun303 differs'; END IF;
  RETURN pg_catalog.jsonb_build_object('status','existing','round_id','arena-2026-09-18','execute_count',100,'score_namespace','score:rerun303');
 END IF;
 IF p_ordered_icp_bank_hash IS DISTINCT FROM '6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91' OR p_schedule IS DISTINCT FROM $s${"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$s$::JSONB THEN RAISE EXCEPTION 'rerun303 sealed ordered ICP bank or schedule differs'; END IF;
 IF (p_schedule->>'stage_1_scoring_close')::TIMESTAMPTZ<=pg_catalog.clock_timestamp() THEN RAISE EXCEPTION 'rerun303 scoring admission window closed'; END IF;
 SELECT * INTO nr FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18' FOR UPDATE; SELECT * INTO nb FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' FOR UPDATE;
 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind) CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(s.submission_id,k.kind,NULL) state)c WHERE s.round_id='arena-2026-09-18' AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0 OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 IF pg_catalog.to_jsonb(nr) IS DISTINCT FROM terminal OR pg_catalog.to_jsonb(nb) IS DISTINCT FROM baseline OR active_count<>0 OR unsettled<>0 OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE THEN RAISE EXCEPTION 'terminal rerun302 differs' USING ERRCODE='55000'; END IF;
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') INTO pre_sh FROM public.lab_arena_submissions r WHERE round_id='arena-2026-09-18';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at')::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') INTO pre_eh FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-18' AND kind='execute';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') INTO pre_lh FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-18' AND r.kind='execute' AND r.run_id=l.run_id);
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') INTO pre_rlh FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-18' AND r.kind='score' AND r.run_id=l.run_id);
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(r)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') INTO pre_srh FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-18' AND kind='score';
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(l)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') INTO pre_slh FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-18' AND r.kind='score' AND r.run_id=l.run_id);
 IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score')<>52 OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-18' AND s.kind='score' AND s.run_id=l.run_id))<>818 OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-18' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-18' AND s.kind='score' AND s.run_id=l.run_id))<>9996 OR pre_sh<>'sha256:cb2219f19cae619e3017917f8654d68eed194718b8a93e40e56a4b8a8a6baccc' OR pre_eh<>'sha256:fd4dfadfe47e5b5929dd8ea12661a243881af36d3254ff6492bcba564411835d' OR pre_lh<>'sha256:75b15342d5f3db2a4078b8cc7a7e4734382987e8c268f66aea1f5992ce5850b0' OR pre_rlh<>'sha256:ee03d3cffd2e7744f9fb25b2188110674fa5445f363bc3d28e951bccd7e554d5' OR pre_srh<>'sha256:688ab31e2e73215859b582ce4192a01c3e7b42b763f62a9c1b6691da1073bcc4' OR pre_slh<>'sha256:fcd151a18afdda7e8ce98e826b101dc77798ceaedb80c66328f4efba8cf1e2c0' THEN RAISE EXCEPTION 'rerun303 active preimage differs'; END IF;
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object('run_id',run_id,'per_icp_score',per_icp_score,'qualification_doc',qualification_doc) ORDER BY run_id) INTO derived FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute';
 archive_config := nr.configuration_doc||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun302archive','mode','shadow','rewards_enabled',FALSE,'archived_reward_basis_hash',nr.reward_basis_hash,'archived_effective_reward_epoch',nr.effective_reward_epoch,'archived_reward_activated_at',nr.reward_activated_at,'archived_execution_judgments',derived);
 SELECT pg_catalog.jsonb_build_object('rounds',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'submissions',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id,submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'runs',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'ledger',(SELECT pg_catalog.count(*) FROM public.lab_arena_ledger WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'weights',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY network,netuid,epoch),''),'sha256'),'hex')) FROM public.lab_arena_accepted_weight_states r)) INTO before_other;
 ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER; ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER; ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER; ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
 INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers) SELECT round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers FROM pg_catalog.jsonb_populate_record(NULL::public.lab_arena_rounds,terminal||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun302archive','status','cancelled','configuration_doc',archive_config,'rewards_enabled',FALSE,'reward_basis_hash',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,'cancel_reason','authorized_sep18_rerun302_score_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun302archive','submission_id',s.submission_id||':r302archive'))).* FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-18'; GET DIAGNOSTICS cnt=ROW_COUNT; IF cnt<>5 THEN RAISE EXCEPTION 'submission archive count'; END IF;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-18-rerun302archive',submission_id=submission_id||':r302archive' WHERE round_id='arena-2026-09-18' AND kind='score'; GET DIAGNOSTICS cnt=ROW_COUNT; IF cnt<>52 THEN RAISE EXCEPTION 'score archive count'; END IF;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-18-rerun302archive',submission_id=submission_id||':r302archive' WHERE round_id='arena-2026-09-18' AND EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-18-rerun302archive' AND s.kind='score' AND s.run_id=l.run_id); GET DIAGNOSTICS cnt=ROW_COUNT; IF cnt<>818 THEN RAISE EXCEPTION 'score ledger archive count'; END IF;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE round_id='arena-2026-09-18' AND kind='execute'; GET DIAGNOSTICS cnt=ROW_COUNT; IF cnt<>100 THEN RAISE EXCEPTION 'execute judgment reset count'; END IF;
 UPDATE public.lab_arena_rounds SET status='stage1',status_generation=status_generation+1,stage_generation=stage_generation+1,configuration_doc=$c${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$c$::JSONB,stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,finalists=NULL,publication_doc=NULL,published_at=NULL,cancel_reason=NULL,updated_at=pg_catalog.clock_timestamp() WHERE round_id='arena-2026-09-18';
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER; ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER; ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER; ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
 SELECT pg_catalog.jsonb_build_object('rounds',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'submissions',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id,submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'runs',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'ledger',(SELECT pg_catalog.count(*) FROM public.lab_arena_ledger WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun302archive')),'weights',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY network,netuid,epoch),''),'sha256'),'hex')) FROM public.lab_arena_accepted_weight_states r)) INTO after_other;
 IF before_other IS DISTINCT FROM after_other THEN RAISE EXCEPTION 'rerun303 unrelated state changed'; END IF;
 IF public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE THEN RAISE EXCEPTION 'rerun303 score archive postcondition differs'; END IF;
 IF public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE THEN RAISE EXCEPTION 'rerun303 active postcondition differs'; END IF;
 RETURN pg_catalog.jsonb_build_object('status','prepared','round_id','arena-2026-09-18','execute_count',100,'archived_score_count',52,'archived_score_ledger_count',818,'score_namespace','score:rerun303');
END $f$;
ALTER FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun303_v1(TEXT,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun303_v1(TEXT,JSONB) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep18_cancelled_rerun303_v1(TEXT,JSONB) TO lab_arena_service;
CREATE OR REPLACE FUNCTION public.lab_arena_sep18_cancelled_rerun303_publication_guard_v1() RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
BEGIN
 IF NEW.round_id='arena-2026-09-18' AND NEW.status='published' AND EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun302archive') AND (
  public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE OR public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute' AND status='accepted')<>100
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status NOT IN('accepted','failed'))
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun303')<>100
  OR (SELECT pg_catalog.count(DISTINCT scored_run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun303')<>100
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs e WHERE e.round_id='arena-2026-09-18' AND e.kind='execute' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id=e.round_id AND s.kind='score' AND s.status='accepted' AND s.scored_run_id=e.run_id AND s.assignment_id LIKE '%:score:rerun303'))
 ) THEN RAISE EXCEPTION 'Sep18 rerun303 publication requires complete fresh scoring' USING ERRCODE='55000'; END IF; RETURN NEW;
END $f$;
ALTER FUNCTION public.lab_arena_sep18_cancelled_rerun303_publication_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_cancelled_rerun303_publication_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun303_publication_guard ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep18_cancelled_rerun303_publication_guard BEFORE UPDATE OF status ON public.lab_arena_rounds FOR EACH ROW EXECUTE FUNCTION public.lab_arena_sep18_cancelled_rerun303_publication_guard_v1();
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
SELECT public.lab_arena_prepare_sep18_cancelled_rerun303_v1('6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91',$s${"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$s$::JSONB);
NOTIFY pgrst,'reload schema';
COMMIT;
