-- A gateway-proved, zero-call launcher failure is not a model attempt.
-- Reuse the existing expiry receipt and its exact authenticated host evidence.
-- Keep physical attempts, billing, quota fences, quarantine and accepted work.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $zero_call_host_attempt504$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_execution_attempt_count(public.lab_arena_runs)';
  v_preimage CONSTANT TEXT := 'ef4a51fa5955c0f33b0bbb85e8ac825a4f2af311f93a4eaa9cc2198a8e642668';
  v_postimage CONSTANT TEXT := '7017065b833239221c74bba24788d97dc09558edd7fb86f5d4ae1ffa6b7b546d';
  v_definition TEXT;
  v_identity JSONB;
  v_old CONSTANT TEXT := $old$   AND public.lab_arena__execute_host_interruption_v1(host)))::INTEGER;$old$;
  v_new CONSTANT TEXT := $new$   AND (public.lab_arena__execute_host_interruption_v1(host)
     OR (
       -- lab_arena_zero_call_host_attempt_credit_v1: trust only the existing
       -- gateway recovery receipt, corroborated by its original lease proof.
       -- Generic setup RunnerError can be a submitted source failure and is
       -- deliberately not granted runtime-host credit.
       round.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1'
       AND host.status='failed' AND host.terminal_cause='lease_expired'
       AND host.result_doc IS NULL AND host.output_ref IS NULL
       AND NOT host.champion_restart_required AND host.lease_generation>0
       AND host.runner_hotkey IS NOT NULL AND host.lease_token_hash IS NOT NULL
       AND host.claim_request_id IS NOT NULL
       AND host.claim_response->>'status'='leased'
       AND host.claim_response->'lease_generation'=to_jsonb(host.lease_generation)
       AND host.terminal_doc->>'recovery_reason'='authenticated_zero_call_runtime_host_error'
       AND (host.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ=
         (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
       AND host.lease_expires_at<(host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
       AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger cost WHERE cost.run_id=host.run_id)
       AND NOT EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events activity
         WHERE activity.run_id=host.run_id AND
           (activity.event_kind LIKE 'provider.%'
            OR activity.event_kind IN ('runtime.finished','runtime.cleanup_error')
            OR activity.event_kind='runtime.error' AND activity.content->>'failure_stage'='cleanup'))
       AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events error
         WHERE error.run_id=host.run_id
           AND (error.runner_hotkey,error.round_id,error.submission_id,error.miner_hotkey,
             error.assignment_id,error.stage,error.icp_position,error.attempt,error.run_kind)
             IS NOT DISTINCT FROM
             (host.runner_hotkey,host.round_id,host.submission_id,host.miner_hotkey,
              host.assignment_id,host.stage,host.icp_position,host.attempt,host.kind)
           AND error.event_kind='runtime.error'
           AND error.content->>'status'='abandoned'
           AND error.content->>'failure_stage'='runtime'
           AND error.content->>'error_class'='RuntimeHostError'
           AND error.created_at BETWEEN (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
             AND (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
           AND error.occurred_at BETWEEN date_trunc('second',
             (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
             AND (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
       )
     ))))::INTEGER;$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::REGPROCEDURE;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena zero-call host attempt security shape differs';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (length(v_definition)-length(replace(v_definition,v_old,'')))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena zero-call host attempt preimage differs';
  END IF;
  v_definition:=replace(v_definition,v_old,v_new);
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena zero-call host attempt postimage differs';
  END IF;
  EXECUTE v_definition;
  IF encode(extensions.digest(pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage
    OR (SELECT jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
      WHERE p.oid=v_signature::REGPROCEDURE) IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'Arena zero-call host attempt readback differs';
  END IF;
END;
$zero_call_host_attempt504$;
NOTIFY pgrst,'reload schema';
COMMIT;
