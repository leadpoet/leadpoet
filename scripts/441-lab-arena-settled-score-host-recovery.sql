-- Release a signaled, cleaned-up scorer only after every provider call settles.
-- Keep the 412 zero-call path, frozen claim evidence, ledger and retry limits.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $settled_score_host_recovery$
DECLARE
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_preimage CONSTANT TEXT := 'ef61f7bf17884267c7c126529148099e48591eb973e88ecb841eea4498439688';
  v_postimage CONSTANT TEXT := '9735de28e3a59f9143546b3a12d5345fc1627ea8d9b39a80f8863177286ff654';
  v_anchor CONSTANT TEXT := $anchor$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$anchor$;
  v_replacement CONSTANT TEXT := $replacement$  -- lab_arena_settled_score_host_recovery_v1: RuntimeHostError is raised
  -- only after runsc deletion and namespace cleanup succeed. Score socket
  -- requests cannot replay paid calls. The round/run locks also fence any
  -- queued reservation or dispatch against the old lease after recovery.
  -- Settlements can arrive after abandonment and renew the live expiry, so
  -- authenticate against the frozen claim window and exact lease generation.
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND v_early_recovery_allowed AND candidate.kind = 'score'
      AND candidate.stage_generation = v_round.stage_generation
      AND v_round.status = 'stage' || candidate.stage::TEXT || '_scoring'
      AND candidate.lease_expires_at > pg_catalog.clock_timestamp()
    ORDER BY candidate.assignment_id
    FOR UPDATE
  LOOP
    -- Check accounting in a fresh statement after the lease row lock.
    IF v_run.status = 'leased' AND v_run.kind = 'score'
       AND v_run.stage_generation = v_round.stage_generation
       AND v_round.status = 'stage' || v_run.stage::TEXT || '_scoring'
       AND v_run.result_doc IS NULL AND v_run.output_ref IS NULL
       AND v_run.runner_hotkey IS NOT NULL
       AND v_run.lease_token_hash IS NOT NULL
       AND v_run.claim_request_id IS NOT NULL
       AND v_run.claim_response ->> 'status' = 'leased'
       AND v_run.claim_response -> 'lease_generation' =
           pg_catalog.to_jsonb(v_run.lease_generation)
       AND v_run.lease_generation > 0
       AND v_run.lease_expires_at >=
           (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
       AND EXISTS (
         SELECT 1 FROM public.lab_arena_ledger AS cost
         WHERE cost.run_id = v_run.run_id
       )
       AND NOT EXISTS (
         SELECT 1 FROM public.lab_arena_ledger AS unidentified
         WHERE unidentified.run_id = v_run.run_id
           AND (unidentified.call_identity IS NULL
                OR unidentified.call_identity !~ '^sha256:[0-9a-f]{64}$')
       )
       AND NOT EXISTS (
         SELECT 1 FROM (
           SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
           FROM public.lab_arena_ledger AS cost
           WHERE cost.run_id = v_run.run_id
           ORDER BY cost.call_identity, cost.entry_id DESC
         ) AS heads WHERE heads.entry_kind <> 'settlement'
       )
       AND NOT EXISTS (
         SELECT 1 FROM public.lab_arena_trajectory_events AS activity
         WHERE activity.run_id = v_run.run_id
           AND (activity.event_kind IN ('runtime.finished', 'runtime.cleanup_error')
                OR (activity.event_kind = 'runtime.error'
                    AND activity.content ->> 'failure_stage' = 'cleanup'))
       )
       AND EXISTS (
         SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
         WHERE host_error.run_id = v_run.run_id
           AND host_error.runner_hotkey = v_run.runner_hotkey
           AND host_error.round_id = v_run.round_id
           AND host_error.submission_id = v_run.submission_id
           AND host_error.miner_hotkey = v_run.miner_hotkey
           AND host_error.assignment_id = v_run.assignment_id
           AND host_error.stage = v_run.stage
           AND host_error.icp_position = v_run.icp_position
           AND host_error.attempt = v_run.attempt
           AND host_error.run_kind = v_run.kind
           AND host_error.event_kind = 'runtime.error'
           AND host_error.content ->> 'status' = 'abandoned'
           AND host_error.content ->> 'failure_stage' = 'runtime'
           AND host_error.content ->> 'error_class' = 'RuntimeHostError'
           AND host_error.content ->> 'runtime_host_reason' = 'sandbox_launcher_signaled'
           AND pg_catalog.jsonb_typeof(host_error.content -> 'launch_exit_code') = 'number'
           AND host_error.content ->> 'launch_exit_code' ~ '^-[1-9][0-9]*$'
           AND host_error.content -> 'launch_timed_out' = 'false'::JSONB
           AND host_error.created_at BETWEEN
               (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
                 - pg_catalog.make_interval(
                   secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                 ) AND (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
           AND host_error.occurred_at BETWEEN
               pg_catalog.date_trunc('second',
                 (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
                   - pg_catalog.make_interval(
                     secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                   )) AND (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
           AND EXISTS (
             SELECT 1 FROM public.lab_arena_trajectory_events AS started
             WHERE started.run_id = host_error.run_id
               AND started.runner_hotkey = host_error.runner_hotkey
               AND started.round_id = host_error.round_id
               AND started.submission_id = host_error.submission_id
               AND started.miner_hotkey = host_error.miner_hotkey
               AND started.assignment_id = host_error.assignment_id
               AND started.stage = host_error.stage
               AND started.icp_position = host_error.icp_position
               AND started.attempt = host_error.attempt
               AND started.run_kind = host_error.run_kind
               AND started.event_kind = 'runtime.started'
               AND started.content -> 'lease_generation' =
                   pg_catalog.to_jsonb(v_run.lease_generation)
               AND started.created_at BETWEEN
                   (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
                     - pg_catalog.make_interval(
                       secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                     ) AND host_error.created_at
               AND started.occurred_at BETWEEN
                   pg_catalog.date_trunc('second',
                     (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
                       - pg_catalog.make_interval(
                         secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                       )) AND host_error.occurred_at
           )
       ) THEN
      UPDATE public.lab_arena_runs
      SET lease_expires_at = pg_catalog.clock_timestamp(),
          terminal_doc = COALESCE(terminal_doc, '{}'::JSONB) ||
            pg_catalog.jsonb_build_object(
              'original_lease_expires_at', v_run.lease_expires_at,
              'recovery_reason', 'authenticated_settled_score_runtime_host_error'
            )
      WHERE run_id = v_run.run_id;
    END IF;
  END LOOP;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena settled score host recovery security shape differs'
      USING ERRCODE='55000';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash=v_postimage THEN RETURN; END IF;
  IF v_hash<>v_preimage OR
     (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_anchor,''))) / pg_catalog.length(v_anchor) <> 1 THEN
    RAISE EXCEPTION 'Arena settled score host recovery preimage differs'
      USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_anchor,v_replacement);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled score host recovery postimage differs'
      USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    'public.lab_arena_expire_leases(text)'::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena settled score host recovery readback differs'
      USING ERRCODE='55000';
  END IF;
END;
$settled_score_host_recovery$;
COMMIT;
