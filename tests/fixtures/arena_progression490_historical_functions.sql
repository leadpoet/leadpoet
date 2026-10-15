-- Test-only read-only catalog capture, 2026-10-15. No competition rows.
-- These retained historical branches are absent from the ordinary485 fixture.
-- Tests require every captured pg_get_functiondef SHA256 to equal migration490.
CREATE OR REPLACE FUNCTION public.lab_arena_open_stage(p_round_id text, p_stage smallint, p_participants jsonb, p_icp_positions integer[])
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
 SET statement_timeout TO '60s'
AS $function$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_participant JSONB;
  v_submission public.lab_arena_submissions;
  v_index INTEGER;
  v_position INTEGER;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_participant_count INTEGER;
  v_distinct_count INTEGER;
  v_expected_count INTEGER;
  v_all_participants BOOLEAN;
  v_baseline_first BOOLEAN;
  v_stage_1_count INTEGER;
  v_total_count INTEGER;
  v_expected_positions INTEGER[];
BEGIN
  IF p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR p_icp_positions IS NULL
     OR pg_catalog.cardinality(p_icp_positions) < 1 THEN
    RAISE EXCEPTION 'lab_arena_stage_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_expected := CASE p_stage WHEN 1 THEN 'committed' ELSE 'stage1_scored' END;
  v_next := 'stage' || p_stage::TEXT;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> v_expected THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  v_baseline_first := COALESCE(
    v_round.configuration_doc ->> 'execution_sequence_policy'
      = 'baseline_scored_first_v1',
    FALSE
  );
  IF pg_catalog.jsonb_array_length(p_participants) < 1
     AND NOT (v_baseline_first AND p_stage = 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_input_invalid' USING ERRCODE = '22023';
  END IF;
  IF v_baseline_first
     AND COALESCE(
       (v_round.configuration_doc ->> 'parallel_twenty_icp_execution')::BOOLEAN,
       FALSE
     ) THEN
    RAISE EXCEPTION 'lab_arena_execution_sequence_invalid' USING ERRCODE = '22023';
  END IF;
  v_stage_1_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER;
  v_total_count := v_stage_1_count
    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
  IF v_stage_1_count < 1 OR v_total_count NOT BETWEEN 2 AND 100 THEN
    RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT ARRAY(SELECT pg_catalog.generate_series(
    CASE WHEN v_baseline_first OR p_stage = 1 THEN 0 ELSE v_stage_1_count END,
    CASE WHEN v_baseline_first OR p_stage = 2 THEN v_total_count - 1
         ELSE v_stage_1_count - 1 END
  )) INTO v_expected_positions;
  IF p_icp_positions IS DISTINCT FROM v_expected_positions THEN
    RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*), COUNT(DISTINCT participant ->> 'submission_id')
  INTO v_participant_count, v_distinct_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant;
  IF v_participant_count <> v_distinct_count THEN
    RAISE EXCEPTION 'lab_arena_stage_participants_duplicate' USING ERRCODE = '22023';
  END IF;
  v_all_participants := NOT v_baseline_first
    AND (p_stage = 1 OR v_round.icp_set_date IS NOT NULL);
  IF v_baseline_first THEN
    SELECT COUNT(*) INTO v_expected_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE) = (p_stage = 1);
  ELSIF v_all_participants THEN
    v_expected_count := pg_catalog.jsonb_array_length(v_round.participants);
  ELSE
    SELECT pg_catalog.jsonb_array_length(v_round.finalists) + COUNT(*)
    INTO v_expected_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  END IF;
  IF v_participant_count <> v_expected_count THEN
    RAISE EXCEPTION 'lab_arena_stage_participants_invalid' USING ERRCODE = '22023';
  END IF;
  -- Oct10 uniform rejudge452 preserved terminal execution resume
  IF p_round_id = 'arena-2026-10-10' AND p_stage = 2 AND EXISTS (
    SELECT 1 FROM public.lab_arena_rounds
    WHERE round_id = 'arena-2026-10-10-r452archive') THEN
    IF v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM 'sha256:e49d37c4c5181c5dc0dffc081a3894c4e71ab514804949f4e0a2d686e62be1e6'
       OR v_round.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:e49d37c4c5181c5dc0dffc081a3894c4e71ab514804949f4e0a2d686e62be1e6'
       OR (SELECT jsonb_agg(p ORDER BY p->>'submission_id') FROM jsonb_array_elements(p_participants) p)
          IS DISTINCT FROM (SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id','miner_hotkey',p->>'miner_hotkey') ORDER BY p->>'submission_id')
            FROM jsonb_array_elements(v_round.participants) p WHERE NOT (p->>'is_king')::BOOLEAN)
       OR (SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(r)-'qualification_doc'-'per_icp_score'-'updated_at' ORDER BY run_id),'[]'::jsonb)::text,'sha256'),'hex')
           FROM public.lab_arena_runs r WHERE round_id=p_round_id AND kind='execute')
          IS DISTINCT FROM (SELECT configuration_doc#>>'{recovery_terminal_inventory,execution_sha256}'
            FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10-r452archive')
       OR encode(extensions.digest(v_round.participants::text,'sha256'),'hex') IS DISTINCT FROM (SELECT configuration_doc#>>'{recovery_terminal_inventory,participants_sha256}' FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10-r452archive')
       OR (SELECT encode(extensions.digest(jsonb_agg(to_jsonb(s) ORDER BY submission_id)::text,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id=p_round_id) IS DISTINCT FROM (SELECT configuration_doc#>>'{recovery_terminal_inventory,submissions_sha256}' FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10-r452archive')
       OR (SELECT encode(extensions.digest(icps::text,'sha256'),'hex') FROM public.qualification_private_icp_sets WHERE set_id=20261009) IS DISTINCT FROM (SELECT configuration_doc#>>'{recovery_terminal_inventory,bank_sha256}' FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10-r452archive')
       -- Historical judge bodies are sealed by the atomic migration/replay.
       -- They do not authorize the new-image scores opened by this live source fence.
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds a
         WHERE a.round_id='arena-2026-10-10-r452archive' AND a.status='cancelled'
           AND NOT a.rewards_enabled
           AND a.cancel_reason='authorized_oct10_uniform_evidence_rejudge452_recovery_archive'
           AND a.configuration_doc->>'recovery_source_round_id'=p_round_id
           AND a.configuration_doc->>'recovery_reviewed_source_commit'='4ba6795c38d91790139b7fff16275513da840bc4'
           AND a.configuration_doc#>>'{recovery_original_round,configuration_doc,scorer_image_digest}'='sha256:1b3d6f9005362453431ab7710cb0ba0be1288495334e2167a4425d57583db70a'
           AND jsonb_typeof(a.configuration_doc->'recovery_terminal_inventory')='object'
           AND a.configuration_doc#>>'{recovery_terminal_inventory,prior445_archive_sha256}' ~ '^[0-9a-f]{64}$'
           AND a.configuration_doc#>>'{recovery_terminal_inventory,prior448_archive_sha256}' ~ '^[0-9a-f]{64}$'
           AND a.configuration_doc->>'recovery_ledger_sha256' ~ '^[0-9a-f]{64}$'
           AND a.configuration_doc->>'recovery_events_sha256' ~ '^[0-9a-f]{64}$'
           AND a.configuration_doc->>'recovery_score_runs_sha256' ~ '^[0-9a-f]{64}$'
           AND (a.configuration_doc->>'recovery_ledger_max')::bigint=(a.configuration_doc#>>'{recovery_terminal_inventory,ledger_max}')::bigint
           AND (a.configuration_doc->>'recovery_events_max')::bigint=(a.configuration_doc#>>'{recovery_terminal_inventory,events_max}')::bigint)
       OR (SELECT count(DISTINCT assignment_id) FROM public.lab_arena_runs
           WHERE round_id=p_round_id AND stage=2 AND kind='execute') <> 690
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round_id
           AND stage=2 AND kind='execute' AND (status NOT IN ('accepted','failed') OR stage_generation<>9)) THEN
      RAISE EXCEPTION 'Oct10 uniform rejudge452 preserved execution differs' USING ERRCODE='55000';
    END IF;
    -- Advance only the round fence. Do not allocate a new execution generation, insert attempts, or mutate
    -- existing generation9 rows. Only ordinary judging generations advance.
    UPDATE public.lab_arena_rounds SET status='stage2',status_generation=status_generation+1,
      stage_generation=stage_generation+1
      WHERE round_id=p_round_id;
    RETURN jsonb_build_object('status','ok','round_status','stage2',
      'stage_generation',v_round.stage_generation+1,'assignments',690,'resumed',TRUE);
  END IF;
  v_generation := v_round.stage_generation + 1;
  -- Oct01 rerun379 preserved miner execution resume
  IF p_round_id = 'arena-2026-10-01' AND p_stage = 2
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-10-01-r379archive') THEN
    IF v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:0041ce7afd714db500fe3311862f78b9946057e2a44bae3c54aa61939e817761'
       OR EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_participants) p
         WHERE NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
           WHERE s.round_id=p_round_id AND s.submission_id=p->>'submission_id'
             AND s.miner_hotkey=p->>'miner_hotkey'
             AND s.status='frozen' AND NOT s.is_king))
       OR (SELECT COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs
         WHERE round_id=p_round_id AND stage=2 AND kind='execute') <> 130
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
         WHERE r.round_id=p_round_id AND r.stage=2 AND
           (r.kind<>'execute' OR r.status IN ('leased','submitted')
             OR r.icp_position NOT BETWEEN 0 AND 9
             OR r.assignment_id IS DISTINCT FROM p_round_id||':'||r.submission_id||':2:'||r.icp_position::TEXT
             OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
               WHERE s.round_id=p_round_id AND s.submission_id=r.submission_id
                 AND s.miner_hotkey=r.miner_hotkey AND s.status='frozen' AND NOT s.is_king
                 AND EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_participants) p
                   WHERE p->>'submission_id'=r.submission_id
                     AND p->>'miner_hotkey'=r.miner_hotkey)))) THEN
      RAISE EXCEPTION 'Oct01 preserved miner execution differs' USING ERRCODE='55000';
    END IF;
    UPDATE public.lab_arena_runs SET stage_generation=v_generation
      WHERE round_id=p_round_id AND stage=2 AND kind='execute' AND status='pending';
    UPDATE public.lab_arena_rounds SET status='stage2',
      status_generation=status_generation+1,stage_generation=v_generation
      WHERE round_id=p_round_id;
    RETURN pg_catalog.jsonb_build_object('status','ok','round_status','stage2',
      'stage_generation',v_generation,'assignments',130,'resumed',TRUE);
  END IF;
  FOR v_participant IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_participants)
  LOOP
    SELECT * INTO v_submission
    FROM public.lab_arena_submissions
    WHERE submission_id = (v_participant ->> 'submission_id')
      AND round_id = p_round_id
      AND status = 'frozen';
    IF NOT FOUND OR v_submission.miner_hotkey <> (v_participant ->> 'miner_hotkey') THEN
      RAISE EXCEPTION 'lab_arena_participant_not_frozen' USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(v_round.participants) AS original
      WHERE original ->> 'submission_id' = v_submission.submission_id
        AND original ->> 'miner_hotkey' = v_submission.miner_hotkey
        AND (
          (v_baseline_first AND
            COALESCE((original ->> 'is_king')::BOOLEAN, FALSE) = (p_stage = 1))
          OR v_all_participants
          OR (NOT v_baseline_first AND (
            COALESCE((original ->> 'is_king')::BOOLEAN, FALSE)
            OR v_round.finalists ? v_submission.submission_id
          ))
        )
    ) THEN
      RAISE EXCEPTION 'lab_arena_stage_participants_invalid' USING ERRCODE = '22023';
    END IF;
    FOR v_index IN 1 .. pg_catalog.array_length(p_icp_positions, 1) LOOP
      v_position := p_icp_positions[v_index];
      v_assignment := p_round_id || ':' || v_submission.submission_id || ':'
        || p_stage::TEXT || ':' || v_position::TEXT;
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
        icp_position, attempt, status, stage_generation
      ) VALUES (
        v_assignment || ':1', v_assignment, p_round_id,
        v_submission.submission_id, v_submission.miner_hotkey, p_stage,
        v_position, 1, 'pending', v_generation
      );
      v_created := v_created + 1;
    END LOOP;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next,
      status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok',
    'round_status', v_next,
    'stage_generation', v_generation,
    'assignments', v_created
  );
END;
$function$
;
ALTER FUNCTION public.lab_arena_open_stage(text,smallint,jsonb,integer[]) OWNER TO lab_arena_owner;
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
                 WHERE round_id = 'arena-2026-09-18-rerun302archive')
     AND NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-09-18-rerun303archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-18T21:00:00Z","final_scoring_close":"2026-09-19T07:30:04Z","publication_deadline":"2026-09-19T07:30:05Z","stage_1_close":"2026-09-18T21:00:02Z","stage_1_scoring_close":"2026-09-19T02:15:02Z","stage_1_start":"2026-09-18T21:00:01Z","stage_2_close":"2026-09-19T02:15:04Z","stage_2_start":"2026-09-19T02:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3e6a7a8b63de2f32a80b79bf8b788938b49f000c74bb79d17925193a2f9771f2","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
       OR public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE THEN
      RAISE EXCEPTION 'lab_arena_rerun303_frozen_state_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;
  IF p_round_id = 'arena-2026-09-18' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-09-18-rerun303archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $rerun304_configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$rerun304_configuration$::JSONB
       OR public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
       OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions baseline
          WHERE baseline.round_id=p_round_id AND baseline.submission_id='baseline-2026-09-18'
            AND baseline.source_ref='arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz'
            AND baseline.source_size_bytes=673162
            AND baseline.submission_doc->>'source_sha256'='780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0'
            AND baseline.submission_doc->>'source_commit'='e8572c97f5b69b0bd39ebad8edc47f91df2bfe59') THEN
      RAISE EXCEPTION 'lab_arena_rerun304_frozen_state_invalid'
        USING ERRCODE='22023';
    END IF;
  END IF;
  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r325archive')
     AND public.lab_arena_sep19_rerun325_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun325_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_work_items)
  LOOP
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'scored_run_id', '') = ''
       OR COALESCE(v_item ->> 'submission_id', '') !~ '^[A-Za-z0-9._:-]{1,64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'icp_position') IS DISTINCT FROM 'number'
       OR (v_item ->> 'icp_position')::INTEGER NOT BETWEEN
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1' OR p_stage = 1
               THEN 0 ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER END)
         AND
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1' OR p_stage = 2
               THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
                    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1
               ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER - 1 END)
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
                    WHERE round_id = 'arena-2026-09-18-rerun303archive')
           THEN ':rerun304'
           WHEN p_round_id = 'arena-2026-09-18'
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
    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r325archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun325';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r326archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun326';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r328archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun328';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r332archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun332';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r335archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun335';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r337archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun337';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r338archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun338';
    END IF;
    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r339archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun339';
    END IF;
    -- Oct01 rerun376 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r376archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun376';
    END IF;
    -- Oct01 rerun379 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r379archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun379';
    END IF;
    -- Oct01 rerun381 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r381archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun381';
    END IF;
    -- Oct01 rerun383 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r383archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun383';
    END IF;
    -- Oct01 rerun385 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r385archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun385';
    END IF;
    -- Oct01 rerun387 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r387archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun387';
    END IF;
    -- Oct01 rerun389 score namespace
    IF p_round_id = 'arena-2026-10-01' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-10-01-r389archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun389';
    END IF;
    -- Oct10 recovery440 score namespace
    IF p_round_id = 'arena-2026-10-10' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds
      WHERE round_id = 'arena-2026-10-10-r440archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:recovery440';
    END IF;
    -- Oct10 uniform rejudge452 score namespace (replaces rerun448)
    IF p_round_id = 'arena-2026-10-10' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds
      WHERE round_id = 'arena-2026-10-10-r452archive') THEN
      v_assignment := p_round_id || ':' || v_scored.submission_id || ':' || p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun452';
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
ALTER FUNCTION public.lab_arena_open_scoring_v2(text,smallint,jsonb) OWNER TO lab_arena_owner;
CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_sep15_baseline_only_v1(p_round_id text, p_stage smallint, p_work_items jsonb)
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
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun256';
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
ALTER FUNCTION public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb) FROM PUBLIC,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb) TO anon,authenticated,service_role;
CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_sep16_baseline_only_v1(p_round_id text, p_stage smallint, p_work_items jsonb)
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
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score:rerun277';
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
ALTER FUNCTION public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
