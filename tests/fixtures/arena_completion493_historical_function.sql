-- Exact source-only production preimage, read during the authorized2026-10-16 audit.
-- No row data, provider response or credentials.
CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt(p_run_id text, p_lease_token_hash text, p_result jsonb, p_terminal_cause text, p_output_ref text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_runs;
  v_open INTEGER;
  v_status TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_result) IS DISTINCT FROM 'object'
     OR (p_result ->> 'terminal_status') IS DISTINCT FROM p_terminal_cause
     OR p_terminal_cause NOT IN (
       'accepted', 'model_timeout', 'invalid_output', 'budget_exhausted',
       'credential_error', 'model_error', 'provider_error', 'judge_error',
       'judge_timeout'
     )
     OR (p_terminal_cause = 'accepted'
         AND pg_catalog.char_length(COALESCE(p_output_ref, '')) NOT BETWEEN 1 AND 1024)
     OR (p_terminal_cause <> 'accepted' AND COALESCE(p_output_ref, '') <> ''
         AND NOT (
           p_terminal_cause IN ('judge_error', 'judge_timeout', 'credential_error')
           AND pg_catalog.char_length(p_output_ref) BETWEEN 1 AND 1024
           AND EXISTS (
             SELECT 1 FROM public.lab_arena_runs AS partial_source
             WHERE partial_source.run_id = p_run_id
               AND partial_source.kind = 'score'
               AND (
                 pg_catalog.jsonb_typeof(partial_source.company_judgment_refs) = 'array'
                 OR public.lab_arena__legacy_partial_bound(
                   partial_source, p_result -> 'legacy_partial_checkpoint', p_output_ref)
               )
           )
         )) THEN
    RAISE EXCEPTION 'lab_arena_complete_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_existing
  FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF FOUND AND v_existing.status IN ('accepted', 'failed') THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', v_existing.status, 'idempotent', TRUE,
      'run_id', p_run_id, 'attempt', v_existing.attempt
    );
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  IF (v_run.kind = 'execute' AND p_terminal_cause IN ('judge_error', 'judge_timeout'))
     OR (v_run.kind = 'score' AND p_terminal_cause NOT IN (
       'accepted', 'credential_error', 'judge_error', 'judge_timeout'
     )) THEN
    RAISE EXCEPTION 'lab_arena_complete_cause_kind_mismatch' USING ERRCODE = '22023';
  END IF;
  IF v_run.champion_restart_required THEN
    -- A swallowed provider error cannot turn partial work into an accepted ICP.
    p_terminal_cause := 'provider_error';
    p_result := jsonb_set(p_result,'{terminal_status}','"provider_error"'::JSONB);
    p_output_ref := '';
  END IF;
  SELECT COUNT(*) INTO v_open FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch');
  IF v_open > 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'accounting_open', 'open_calls', v_open
    );
  END IF;
  v_status := CASE
    WHEN p_terminal_cause = 'accepted' THEN 'accepted' ELSE 'failed' END;
  UPDATE public.lab_arena_runs
  SET status = v_status,
      result_doc = p_result,
      terminal_cause = p_terminal_cause,
      output_ref = NULLIF(p_output_ref, '')
  WHERE run_id = p_run_id;
  IF v_status = 'failed'
     AND ((v_run.kind='score' AND v_run.attempt<2147483647
       AND p_terminal_cause IN ('judge_error','judge_timeout','credential_error')
       AND public.lab_arena__judge_calls_clear_v1(v_run.assignment_id))
       OR (NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
         AND p_terminal_cause NOT IN ('budget_exhausted', 'credential_error')
         AND (
       (v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required)) OR public.lab_arena__execute_host_retry_ready_v1(v_run))
       OR (
         -- lab_arena_setup_failure_model_retry_v1
         v_run.kind = 'execute'
         AND v_run.attempt = 2
         AND p_terminal_cause IN ('model_error', 'model_timeout', 'invalid_output')
         AND CASE
           WHEN pg_catalog.jsonb_typeof(
                  p_result #> '{resource_summary,wall_seconds}'
                ) = 'number'
           THEN (p_result #>> '{resource_summary,wall_seconds}')::NUMERIC > 0
           ELSE FALSE
         END
         AND EXISTS (
           SELECT 1
           FROM public.lab_arena_runs AS setup_run
           WHERE setup_run.assignment_id = v_run.assignment_id
             AND setup_run.attempt = 1
             AND setup_run.kind = 'execute'
             AND setup_run.status = 'failed'
             AND setup_run.terminal_cause = 'provider_error'
             AND setup_run.output_ref IS NULL
             AND setup_run.result_doc ->> 'terminal_status' = 'provider_error'
             AND setup_run.result_doc -> 'resource_summary' =
               '{"wall_seconds":0,"cpu_seconds":0,"max_rss_bytes":0,"stdout_bytes":0,"stderr_bytes":0,"provider_call_count":0}'::JSONB
             AND setup_run.result_doc -> 'failure_diagnostic' =
               '{"stage":"provider_call","error_class":"provider_unavailable","reason":"provider_error"}'::JSONB
             AND NOT EXISTS (
               SELECT 1 FROM public.lab_arena_ledger AS setup_ledger
               WHERE setup_ledger.run_id = setup_run.run_id
             )
         )
       )
     ))) AND (v_run.kind<>'score' OR
        public.lab_arena__judge_calls_clear_v1(v_run.assignment_id)) THEN
    SELECT * INTO v_round
    FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
    IF (v_run.stage_generation = v_round.stage_generation OR
         public.lab_arena__judge_continuation_v1(v_round,v_run))
       AND (v_run.kind<>'score' OR public.lab_arena__judge_continuation_v1(v_round,v_run))
       AND (
         v_run.kind <> 'execute'
         OR v_round.configuration_doc ->> 'execution_sequence_policy'
              IS DISTINCT FROM 'baseline_scored_first_v1'
         OR pg_catalog.clock_timestamp() <
              (v_round.configuration_doc #>> ARRAY[
                'schedule', 'stage_' || v_run.stage::TEXT || '_close'
              ])::TIMESTAMPTZ
       ) THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
        icp_position, attempt, status, lease_generation, stage_generation,
        kind, scored_run_id, previous_runner_hotkey, judgment_cache_key,
        judgment_input_hash, judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys, company_judgment_refs
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT,
        v_run.assignment_id, v_run.round_id, v_run.submission_id,
        v_run.miner_hotkey, v_run.stage, v_run.icp_position,
        v_run.attempt + 1, 'pending', v_run.lease_generation,
        v_round.stage_generation, v_run.kind, v_run.scored_run_id,
        v_run.runner_hotkey, v_run.judgment_cache_key,
        v_run.judgment_input_hash, v_run.judgment_scope_doc,
        v_run.judgment_group_leader, v_run.judgment_group_miner_hotkeys,
        v_run.company_judgment_refs
      );
      RETURN pg_catalog.jsonb_build_object(
        'status', v_status, 'idempotent', FALSE, 'run_id', p_run_id,
        'attempt', v_run.attempt,
        'confirmation_attempt', v_run.attempt + 1
      );
    END IF;
  END IF;
  -- lab_arena_judgment_group_terminal_handoff_v1: the normal retry branch
  -- returns above. A terminal leader failure now hands the unchanged cache
  -- identity to one follower instead of leaving every follower unclaimable.
  IF v_run.kind = 'score'
     AND v_run.judgment_cache_key IS NOT NULL
     AND COALESCE(v_run.judgment_group_leader, FALSE)
     AND p_terminal_cause IN (
       'credential_error', 'judge_error', 'judge_timeout'
     ) THEN
    -- The completion path already owns the run row. Use a group advisory lock
    -- instead of taking the round row in reverse order from expiry.
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'lab_arena.judgment-group:' || v_run.round_id || ':' ||
        v_run.stage::TEXT || ':' || v_run.stage_generation::TEXT || ':' ||
        v_run.judgment_cache_key,
        0
      )
    );
  END IF;
  IF v_run.kind = 'score'
     AND v_run.judgment_cache_key IS NOT NULL
     AND COALESCE(v_run.judgment_group_leader, FALSE)
     AND p_terminal_cause IN (
       'credential_error', 'judge_error', 'judge_timeout'
     )
     AND EXISTS (
       SELECT 1
       FROM public.lab_arena_rounds AS active_round
       WHERE active_round.round_id = v_run.round_id
         AND active_round.stage_generation = v_run.stage_generation
         AND active_round.status =
             'stage' || v_run.stage::TEXT || '_scoring'
     )
     AND NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS active_leader
       WHERE active_leader.round_id = v_run.round_id
         AND active_leader.stage = v_run.stage
         AND active_leader.stage_generation = v_run.stage_generation
         AND active_leader.kind = 'score'
         AND active_leader.judgment_cache_key = v_run.judgment_cache_key
         AND COALESCE(active_leader.judgment_group_leader, FALSE)
         AND active_leader.status IN ('pending', 'leased', 'submitted')
     )
     AND NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_judgment_cache AS accepted_cache
       WHERE accepted_cache.cache_key = v_run.judgment_cache_key
     ) THEN
    UPDATE public.lab_arena_runs
    SET judgment_group_leader = TRUE
    WHERE run_id = (
      SELECT follower.run_id
      FROM public.lab_arena_runs AS follower
      WHERE follower.round_id = v_run.round_id
        AND follower.stage = v_run.stage
        AND follower.stage_generation = v_run.stage_generation
        AND follower.kind = 'score'
        AND follower.judgment_cache_key = v_run.judgment_cache_key
        AND follower.status = 'pending'
        AND NOT COALESCE(follower.judgment_group_leader, FALSE)
      ORDER BY follower.run_id
      FOR UPDATE
      LIMIT 1
    );
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', v_status, 'idempotent', FALSE,
    'run_id', p_run_id, 'attempt', v_run.attempt
  );
END;
$function$

