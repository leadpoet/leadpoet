-- Reuse only sealed Sep18 rerun295 miner judgments; do not rewrite retained evidence.
BEGIN;
DO $reuse_sealed_rerun295_scores$
DECLARE
  v_definition TEXT;
  v_original TEXT;
  v_old_declaration TEXT := E'  v_status TEXT;\nBEGIN';
  v_new_declaration TEXT := E'  v_status TEXT;\n  v_rerun295_retained_validated BOOLEAN := FALSE;\nBEGIN';
  v_old TEXT := $old$
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;
$old$;
  v_new TEXT := $new$
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
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;
$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_sep18_rerun291_archive_valid295_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply sealed migration 295 before retained score repair';
  END IF;
  v_definition := pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure
  );
  IF pg_catalog.strpos(v_definition, v_new) > 0 THEN
    v_original := pg_catalog.replace(
      pg_catalog.replace(v_definition, v_new, v_old),
      v_new_declaration, v_old_declaration
    );
  ELSE
    v_original := v_definition;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_original, 'sha256'), 'hex')
       IS DISTINCT FROM 'cac6ba180ff5b386706885e4318f9d8d3563c53d8afde002ecfb5516cedbed05'
     OR pg_catalog.strpos(v_original, v_old) = 0
     OR pg_catalog.strpos(
       pg_catalog.substr(v_original, pg_catalog.strpos(v_original, v_old)
         + pg_catalog.length(v_old)), v_old) <> 0 THEN
    RAISE EXCEPTION 'current rerun295 scoring definition differs';
  END IF;
  IF v_definition = v_original THEN
    EXECUTE pg_catalog.replace(
      pg_catalog.replace(v_original, v_old, v_new),
      v_old_declaration, v_new_declaration
    );
  END IF;
END;
$reuse_sealed_rerun295_scores$;
COMMIT;
