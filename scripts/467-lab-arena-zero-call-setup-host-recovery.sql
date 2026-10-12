-- Recover authenticated, stopped, zero-call setup host failures through ordinary expiry.
-- Preserve restart/hold authority, frozen claims, ledger, cleanup, scores and retry caps.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $zero_call_setup467$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_expire_leases(text)';
  v_preimage CONSTANT TEXT := '11a47cbe7ba950bdc284342a2ede73cd08aba7b0f55e452131d3bf87c9a81c15';
  v_postimage CONSTANT TEXT := 'baf93076fc258ed37f08ba74c872a3ee8b82af73ba05ff1a92a981ce2664b700';
  v_definition TEXT;
  v_identity JSONB;
  v_zero TEXT;
  v_setup_proof TEXT;
  v_start INTEGER;
  v_end INTEGER;
  v_proof_start INTEGER;
  v_proof_end INTEGER;
  v_old_pair CONSTANT TEXT := $old$           AND host_error.content ->> 'failure_stage' = 'runtime'
           AND host_error.content ->> 'error_class' = 'RuntimeHostError'$old$;
  v_setup_pair CONSTANT TEXT := $setup$           AND (host_error.content ->> 'failure_stage' = 'setup'
                AND host_error.content ->> 'error_class' = 'RunnerError'
                AND v_run.lease_generation > 0
                AND v_run.claim_response -> 'lease_generation' = pg_catalog.to_jsonb(v_run.lease_generation)
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
                    AND started.content -> 'lease_generation' = pg_catalog.to_jsonb(v_run.lease_generation)
                    AND started.created_at BETWEEN
                        v_run.lease_expires_at - pg_catalog.make_interval(
                          secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                        ) AND host_error.created_at
                    AND started.occurred_at BETWEEN
                        pg_catalog.date_trunc('second', v_run.lease_expires_at - pg_catalog.make_interval(
                          secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                        )) AND host_error.occurred_at
                ))$setup$;
  v_new_pair CONSTANT TEXT := $new$           -- lab_arena_zero_call_setup_host_recovery_v1: before sandbox entry.
           AND ((host_error.content ->> 'failure_stage' = 'runtime'
                 AND host_error.content ->> 'error_class' = 'RuntimeHostError')
                OR (host_error.content ->> 'failure_stage' = 'setup'
                AND host_error.content ->> 'error_class' = 'RunnerError'
                AND v_run.lease_generation > 0
                AND v_run.claim_response -> 'lease_generation' = pg_catalog.to_jsonb(v_run.lease_generation)
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
                    AND started.content -> 'lease_generation' = pg_catalog.to_jsonb(v_run.lease_generation)
                    AND started.created_at BETWEEN
                        v_run.lease_expires_at - pg_catalog.make_interval(
                          secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                        ) AND host_error.created_at
                    AND started.occurred_at BETWEEN
                        pg_catalog.date_trunc('second', v_run.lease_expires_at - pg_catalog.make_interval(
                          secs => (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER
                        )) AND host_error.occurred_at
                )))$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::text,
    p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::regprocedure;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'Arena setup recovery security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena setup recovery preimage differs' USING ERRCODE='55000';
  END IF;
  -- Edit only412's locked, zero-ledger branch. Later paid recovery, canonical
  -- owner capture, ordinary expiry and attempt creation remain byte-identical.
  v_start:=strpos(v_definition,'  -- lab_arena_abandoned_host_recovery_v1:');
  v_end:=strpos(v_definition,'  -- The existing whole-round expiry must never touch foreign captured work.');
  IF v_start=0 OR v_end<=v_start THEN
    RAISE EXCEPTION 'Arena setup recovery scope differs' USING ERRCODE='55000';
  END IF;
  v_zero:=substr(v_definition,v_start,v_end-v_start);
  IF (length(v_zero)-length(replace(v_zero,v_old_pair,'')))/length(v_old_pair)<>1 THEN
    RAISE EXCEPTION 'Arena setup recovery anchor differs' USING ERRCODE='55000';
  END IF;
  -- Reuse the complete exact host binding in the truthful setup reason. Do
  -- not classify from an unrelated or late event that did not authorize expiry.
  v_proof_start:=strpos(v_zero,'       AND EXISTS (')+length('       AND ');
  v_proof_end:=strpos(substr(v_zero,v_proof_start),') THEN')+v_proof_start-1;
  IF v_proof_start<=length('       AND ') OR v_proof_end<v_proof_start THEN
    RAISE EXCEPTION 'Arena setup recovery proof differs' USING ERRCODE='55000';
  END IF;
  v_setup_proof:=replace(substr(v_zero,v_proof_start,v_proof_end-v_proof_start+1),v_old_pair,v_setup_pair);
  v_zero:=replace(v_zero,v_old_pair,v_new_pair);
  v_zero:=replace(v_zero,
    $reason$'recovery_reason', 'authenticated_zero_call_runtime_host_error'$reason$,
    $reason$'recovery_reason', CASE WHEN $reason$||v_setup_proof||
    $reason$ THEN 'authenticated_zero_call_setup_host_error'
                ELSE 'authenticated_zero_call_runtime_host_error' END$reason$);
  v_definition:=substr(v_definition,1,v_start-1)||v_zero||substr(v_definition,v_end);
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena setup recovery postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF encode(extensions.digest(pg_get_functiondef(v_signature::regprocedure),'sha256'),'hex')<>v_postimage
    OR (SELECT jsonb_build_array(owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig)
      FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
      WHERE p.oid=v_signature::regprocedure) IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'Arena setup recovery readback differs' USING ERRCODE='55000';
  END IF;
END;
$zero_call_setup467$;
NOTIFY pgrst,'reload schema';
COMMIT;
