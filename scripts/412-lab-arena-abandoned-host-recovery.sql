-- Recover only authenticated zero-call host failures through normal expiry.
-- Keep frozen claim evidence, retry limits, scoring budgets and paid work.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $abandoned_host_recovery$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname, p.proacl,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = 'public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] THEN
    RAISE EXCEPTION 'Arena abandoned host recovery security shape differs'
      USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = 'ac1e99fd5bc1b4a458602ce05012d562847bb37d3cf4e69857e45db19077fd00' THEN
    RETURN;
  END IF;
  IF v_hash <> '21634a503b8526bd51dbe102bb89b0bbcbdef582057070ae1d66324b0ebc0532' THEN
    RAISE EXCEPTION 'Arena abandoned host recovery preimage differs'
      USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $oldscan$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
$oldscan$,
    $newscan$  -- lab_arena_abandoned_host_recovery_v1: the round lock excludes
  -- provider reservation/dispatch while each locked lease is checked afresh.
  -- A stopped zero-call host failure must not hold its submission's scoring
  -- budget until the frozen TTL. Reuse ordinary expiry and retry below.
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND candidate.kind IN ('score', 'execute')
      AND candidate.stage_generation = v_round.stage_generation
      AND v_round.status = 'stage' || candidate.stage::TEXT
          || CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END
      AND candidate.lease_expires_at > pg_catalog.clock_timestamp()
    ORDER BY candidate.assignment_id
    FOR UPDATE
  LOOP
    -- This is a separate statement after the row lock. Do not rely on the
    -- candidate scan's MVCC snapshot for absence of billable/provider work.
    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND v_run.stage_generation = v_round.stage_generation
       AND v_round.status = ('stage' || v_run.stage::TEXT
           || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END)
       AND v_run.result_doc IS NULL AND v_run.output_ref IS NULL
       AND v_run.runner_hotkey IS NOT NULL
       AND v_run.lease_token_hash IS NOT NULL
       AND v_run.claim_request_id IS NOT NULL
       AND v_run.claim_response ->> 'status' = 'leased'
       AND v_run.lease_expires_at =
           (v_run.claim_response ->> 'lease_expires_at')::TIMESTAMPTZ
       AND NOT EXISTS (
         SELECT 1 FROM public.lab_arena_ledger AS cost
         WHERE cost.run_id = v_run.run_id
       )
       AND NOT EXISTS (
         SELECT 1 FROM public.lab_arena_trajectory_events AS activity
         WHERE activity.run_id = v_run.run_id
           AND (activity.event_kind LIKE 'provider.%'
                OR activity.event_kind IN ('runtime.finished', 'runtime.cleanup_error')
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
           AND host_error.created_at BETWEEN
               v_run.lease_expires_at - pg_catalog.make_interval(
                 secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
               ) AND v_run.lease_expires_at
           AND host_error.occurred_at BETWEEN
               pg_catalog.date_trunc('second',
                 v_run.lease_expires_at - pg_catalog.make_interval(
                   secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                 )) AND v_run.lease_expires_at
       ) THEN
      UPDATE public.lab_arena_runs
      SET lease_expires_at = pg_catalog.clock_timestamp(),
          terminal_doc = COALESCE(terminal_doc, '{}'::JSONB) ||
            pg_catalog.jsonb_build_object(
              'original_lease_expires_at', v_run.lease_expires_at,
              'recovery_reason', 'authenticated_zero_call_runtime_host_error'
            )
      WHERE run_id = v_run.run_id;
    END IF;
  END LOOP;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
$newscan$);
  v_definition := pg_catalog.replace(v_definition,
    $oldterminal$terminal_doc = pg_catalog.jsonb_build_object('expired_at', pg_catalog.clock_timestamp())$oldterminal$,
    $newterminal$terminal_doc = COALESCE(terminal_doc, '{}'::JSONB) ||
          pg_catalog.jsonb_build_object('expired_at', pg_catalog.clock_timestamp())$newterminal$);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       <> 'ac1e99fd5bc1b4a458602ce05012d562847bb37d3cf4e69857e45db19077fd00' THEN
    RAISE EXCEPTION 'Arena abandoned host recovery postimage differs'
      USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
END;
$abandoned_host_recovery$;
COMMIT;
