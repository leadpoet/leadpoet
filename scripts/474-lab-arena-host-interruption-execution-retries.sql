-- Authenticated host interruptions do not consume normal execute retries.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
-- Keep physical attempts and ledger history. Only authenticated host losses are
-- excluded from the two normal execution outcomes, for October 14 onward.
CREATE OR REPLACE FUNCTION public.lab_arena_execution_attempt_count(p_run public.lab_arena_runs)
RETURNS INTEGER LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $host_attempt_count$
 SELECT GREATEST(0, p_run.attempt - (SELECT count(*)::INTEGER
 FROM public.lab_arena_runs host
 JOIN public.lab_arena_rounds round ON round.round_id=host.round_id
 WHERE p_run.kind='execute' AND host.kind='execute'
   AND round.evaluation_date::DATE>=DATE '2026-10-14'
   AND host.assignment_id=p_run.assignment_id AND host.attempt<=p_run.attempt
   AND (host.round_id,host.submission_id,host.miner_hotkey,host.stage,host.icp_position)
     IS NOT DISTINCT FROM
       (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
   AND (host.status='leased' OR host.status='failed'
     AND host.terminal_cause IN ('lease_expired','stage_closed'))
   AND host.result_doc IS NULL AND host.output_ref IS NULL
   AND NOT host.champion_restart_required AND host.lease_generation>0
   AND host.claim_response->>'status'='leased'
   AND host.claim_response->'lease_generation'=to_jsonb(host.lease_generation)
   AND host.runner_hotkey IS NOT NULL AND host.lease_token_hash IS NOT NULL
   AND host.claim_request_id IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events cleanup
     WHERE cleanup.run_id=host.run_id AND
       (cleanup.event_kind IN ('runtime.finished','runtime.cleanup_error') OR
        cleanup.event_kind='runtime.error' AND cleanup.content->>'failure_stage'='cleanup'))
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
       AND error.content->>'runtime_host_reason'='sandbox_launcher_signaled'
       AND jsonb_typeof(error.content->'launch_exit_code')='number'
       AND error.content->>'launch_exit_code' ~ '^-[1-9][0-9]*$'
       AND error.content->'launch_timed_out'='false'::JSONB
       AND error.created_at BETWEEN (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
         -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
         AND (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
       AND error.occurred_at BETWEEN date_trunc('second',
         (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
         -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
         AND (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
       AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events started
         WHERE started.run_id=host.run_id
           AND (started.runner_hotkey,started.round_id,started.submission_id,started.miner_hotkey,
             started.assignment_id,started.stage,started.icp_position,started.attempt,started.run_kind)
             IS NOT DISTINCT FROM
             (host.runner_hotkey,host.round_id,host.submission_id,host.miner_hotkey,
              host.assignment_id,host.stage,host.icp_position,host.attempt,host.kind)
           AND started.event_kind='runtime.started'
           AND started.content->'lease_generation'=to_jsonb(host.lease_generation)
           AND started.created_at BETWEEN (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
             AND error.created_at
           AND started.occurred_at BETWEEN date_trunc('second',
             (host.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             -make_interval(secs=>(round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
             AND error.occurred_at))))::INTEGER;
$host_attempt_count$;
ALTER FUNCTION public.lab_arena_execution_attempt_count(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_execution_attempt_count(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_execution_attempt_count(public.lab_arena_runs) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__execute_assignment_calls_v1(p_assignment_id TEXT,p_provider TEXT)
RETURNS BIGINT LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $assignment_calls$
 SELECT count(*) FROM (SELECT DISTINCT ON(l.call_identity) l.entry_kind
   FROM public.lab_arena_runs r JOIN public.lab_arena_ledger l ON l.run_id=r.run_id
   WHERE r.assignment_id=p_assignment_id AND r.kind='execute' AND l.provider=p_provider
     AND l.call_identity IS NOT NULL ORDER BY l.call_identity,l.entry_id DESC) heads
 WHERE entry_kind IN ('reservation','dispatch','settlement','uncertain');
$assignment_calls$;
ALTER FUNCTION public.lab_arena__execute_assignment_calls_v1(TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__execute_assignment_calls_v1(TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__execute_host_retry_ready_v1(p_input public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $host_retry_ready$
DECLARE r public.lab_arena_runs; p_round public.lab_arena_rounds; state JSONB;
BEGIN
 SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_input.run_id;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 SELECT * INTO p_round FROM public.lab_arena_rounds WHERE round_id=r.round_id;
 IF NOT FOUND OR r.kind<>'execute' OR r.status<>'failed'
   OR r.terminal_cause NOT IN ('provider_error','lease_expired')
   OR r.output_ref IS NOT NULL OR r.champion_restart_required
   OR p_round.evaluation_date::DATE<DATE '2026-10-14'
   OR p_round.status<>'stage'||r.stage::TEXT
   OR r.stage_generation<>p_round.stage_generation
   OR p_round.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
   OR p_round.configuration_doc->>'sourcing_cost_eligibility_policy' IS DISTINCT FROM 'successful_calls_per_icp_v1'
   OR clock_timestamp()>=(p_round.configuration_doc#>>ARRAY['schedule','stage_'||r.stage::TEXT||'_close'])::TIMESTAMPTZ
   OR public.lab_arena_execution_attempt_count(r)>=(p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
   OR public.lab_arena_execution_attempt_count(r)>=r.attempt
   OR EXISTS (SELECT 1 FROM public.lab_arena_runs other WHERE other.assignment_id=r.assignment_id
     AND (other.attempt>r.attempt OR other.status IN ('pending','leased','submitted','accepted')))
   OR EXISTS (SELECT 1 FROM public.lab_arena_restart_claim_control
       WHERE operator_paused OR guard_commitment<>'') THEN RETURN FALSE; END IF;
 -- Attribution does not depend on pricing. Admission still fences in-flight
 -- calls and unresolved successful charges, retaining unknown failed fees.
 state:=public.lab_arena__successful_icp_cost_state(r.round_id,r.submission_id,r.icp_position);
 IF (state->>'inflight_calls')::BIGINT<>0 OR (state->>'success_unresolved_calls')::BIGINT<>0
   OR (state->>'settled_microusd')::BIGINT>=
     (p_round.configuration_doc->>'execution_icp_cap_microusd')::BIGINT
   OR NOT EXISTS (SELECT 1 FROM jsonb_each_text(p_round.configuration_doc->'call_quotas') quota
     WHERE (quota.value::BIGINT=0 AND quota.key='deepline'
       AND p_round.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1')
       OR (quota.value::BIGINT>0 AND public.lab_arena__execute_assignment_calls_v1(r.assignment_id,quota.key)<quota.value::BIGINT*(p_round.configuration_doc->>'max_attempts_per_assignment')::BIGINT
       AND public.lab_arena__submission_stage_consumed(r.submission_id,r.stage,quota.key,'execute')<
         quota.value::BIGINT*(p_round.configuration_doc->>'max_attempts_per_assignment')::BIGINT*(p_round.configuration_doc->>('stage_'||r.stage::TEXT||'_icp_count'))::BIGINT))
   THEN RETURN FALSE; END IF;
 RETURN TRUE;
END;
$host_retry_ready$;
ALTER FUNCTION public.lab_arena__execute_host_retry_ready_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__execute_host_retry_ready_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $patch474_0$
DECLARE d TEXT; security JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,security FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
 WHERE p.oid='lab_arena_close_stage(text,smallint)'::regprocedure;
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
 '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Host retry474 security shape differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='635c881472b7d066ac1e22b0baab79659058710aa74a0310ddd381a6ab2d3aa4' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'185222894f0fae47de2dbf9996ab7acc7f2b9e205ef08cf34e0f8ca95b0a3e8d' THEN
  RAISE EXCEPTION 'Host retry474 preimage differs: lab_arena_close_stage(text,smallint)'; END IF;
 IF (length(d)-length(replace(d,$old0_0$        v_run.attempt >= 2$old0_0$,'')))<>length($old0_0$        v_run.attempt >= 2$old0_0$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old0_0$        v_run.attempt >= 2$old0_0$,$new0_0$        public.lab_arena_execution_attempt_count(v_run) >= 2$new0_0$);
 IF (length(d)-length(replace(d,$old0_1$(pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]$old0_1$,'')))<>length($old0_1$(pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]$old0_1$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old0_1$(pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]$old0_1$,$new0_1$(pg_catalog.array_agg(public.lab_arena_execution_attempt_count(runs) ORDER BY runs.attempt DESC))[1]$new0_1$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'635c881472b7d066ac1e22b0baab79659058710aa74a0310ddd381a6ab2d3aa4' THEN RAISE EXCEPTION 'Host retry474 postimage differs'; END IF;
 EXECUTE d;
 IF encode(extensions.digest(pg_get_functiondef('lab_arena_close_stage(text,smallint)'::regprocedure),'sha256'),'hex')<>'635c881472b7d066ac1e22b0baab79659058710aa74a0310ddd381a6ab2d3aa4' THEN RAISE EXCEPTION 'Host retry474 readback differs'; END IF;
END;
$patch474_0$;

DO $patch474_1$
DECLARE d TEXT; security JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,security FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
 WHERE p.oid='lab_arena_complete_attempt(text,text,jsonb,text,text)'::regprocedure;
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
 '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Host retry474 security shape differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'386d058ca671fb09ec0bca7616e2bbf0dac886f747cb6811eba38df7844fb7fa' THEN
  RAISE EXCEPTION 'Host retry474 preimage differs: lab_arena_complete_attempt(text,text,jsonb,text,text)'; END IF;
 IF (length(d)-length(replace(d,$old1_0$v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required))$old1_0$,'')))<>length($old1_0$v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required))$old1_0$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old1_0$v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required))$old1_0$,$new1_0$(v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required)) OR public.lab_arena__execute_host_retry_ready_v1(v_run))$new1_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b' THEN RAISE EXCEPTION 'Host retry474 postimage differs'; END IF;
 EXECUTE d;
 IF encode(extensions.digest(pg_get_functiondef('lab_arena_complete_attempt(text,text,jsonb,text,text)'::regprocedure),'sha256'),'hex')<>'dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b' THEN RAISE EXCEPTION 'Host retry474 readback differs'; END IF;
END;
$patch474_1$;

DO $patch474_2$
DECLARE d TEXT; security JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,security FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
 WHERE p.oid='lab_arena_expire_leases(text)'::regprocedure;
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
 '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Host retry474 security shape differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='b871e809c70fb12dbacf918e12ca5e75f9d7721f2fdbcb02a3261035cc2d45ff' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dd07e0670de7cac19429132aedf4b73e098f1720d4834979a50cc3b0ef8186dc' THEN
  RAISE EXCEPTION 'Host retry474 preimage differs: lab_arena_expire_leases(text)'; END IF;
 IF (length(d)-length(replace(d,$old2_0$v_run.attempt < LEAST(5, 2 +
        (SELECT count(*) FROM public.lab_arena_runs AS failed_run
         WHERE failed_run.assignment_id = v_run.assignment_id
           AND failed_run.champion_restart_required))$old2_0$,'')))<>length($old2_0$v_run.attempt < LEAST(5, 2 +
        (SELECT count(*) FROM public.lab_arena_runs AS failed_run
         WHERE failed_run.assignment_id = v_run.assignment_id
           AND failed_run.champion_restart_required))$old2_0$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old2_0$v_run.attempt < LEAST(5, 2 +
        (SELECT count(*) FROM public.lab_arena_runs AS failed_run
         WHERE failed_run.assignment_id = v_run.assignment_id
           AND failed_run.champion_restart_required))$old2_0$,$new2_0$(v_run.attempt < LEAST(5, 2 +
        (SELECT count(*) FROM public.lab_arena_runs AS failed_run
         WHERE failed_run.assignment_id = v_run.assignment_id
           AND failed_run.champion_restart_required)) OR public.lab_arena__execute_host_retry_ready_v1(v_run))$new2_0$);
 IF (length(d)-length(replace(d,$old2_1$  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$old2_1$,'')))<>length($old2_1$  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$old2_1$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old2_1$  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$old2_1$,$new2_1$  -- host_interruption_retry474: recover existing unpublished failures only
  -- within their original execution window. Old terminal rows stay unchanged.
  FOR v_run IN SELECT * FROM public.lab_arena_runs candidate
    WHERE candidate.round_id=p_round_id AND candidate.kind='execute'
      AND candidate.status='failed' AND candidate.attempt>=2
      AND candidate.attempt<2147483647
      AND candidate.terminal_cause IN ('provider_error','lease_expired')
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs later
        WHERE later.assignment_id=candidate.assignment_id AND later.attempt>candidate.attempt)
    ORDER BY candidate.assignment_id FOR UPDATE
  LOOP
    IF public.lab_arena__execute_host_retry_ready_v1(v_run) THEN
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, icp_position,
        attempt, status, lease_generation, stage_generation, kind, scored_run_id,
        previous_runner_hotkey, judgment_cache_key, judgment_input_hash,
        judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys, company_judgment_refs
      ) VALUES (
        v_run.assignment_id || ':' || (v_run.attempt + 1)::TEXT, v_run.assignment_id, v_run.round_id,
        v_run.submission_id, v_run.miner_hotkey, v_run.stage, v_run.icp_position,
        v_run.attempt + 1, 'pending', v_run.lease_generation, v_round.stage_generation,
        v_run.kind, v_run.scored_run_id, v_run.runner_hotkey,
        v_run.judgment_cache_key, v_run.judgment_input_hash,
        v_run.judgment_scope_doc, v_run.judgment_group_leader,
        v_run.judgment_group_miner_hotkeys, v_run.company_judgment_refs
      ) ON CONFLICT (run_id) DO NOTHING;
      IF FOUND THEN v_retried:=v_retried+1; END IF;
    END IF;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$new2_1$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b871e809c70fb12dbacf918e12ca5e75f9d7721f2fdbcb02a3261035cc2d45ff' THEN RAISE EXCEPTION 'Host retry474 postimage differs'; END IF;
 EXECUTE d;
 IF encode(extensions.digest(pg_get_functiondef('lab_arena_expire_leases(text)'::regprocedure),'sha256'),'hex')<>'b871e809c70fb12dbacf918e12ca5e75f9d7721f2fdbcb02a3261035cc2d45ff' THEN RAISE EXCEPTION 'Host retry474 readback differs'; END IF;
END;
$patch474_2$;

DO $patch474_3$
DECLARE d TEXT; security JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO d,security FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
 WHERE p.oid='lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure;
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
 '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Host retry474 security shape differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='4da081265076dda69d237f11dafd1d790771e659503d8eedb6f550ff7d3f3211' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f41d91bec7f3182accf9aad86541d8abe6380ed6ff000bac1aeca83c78dc222d' THEN
  RAISE EXCEPTION 'Host retry474 preimage differs: lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'; END IF;
 IF (length(d)-length(replace(d,$old3_0$  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);$old3_0$,'')))<>length($old3_0$  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);$old3_0$) THEN RAISE EXCEPTION 'Host retry474 anchor differs'; END IF;
 d:=replace(d,$old3_0$  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);$old3_0$,$new3_0$  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);
  -- Host replacement never grants more calls than the original two runs.
  IF v_run.kind='execute' AND v_quota>0
    AND public.lab_arena_execution_attempt_count(v_run)<v_run.attempt
    AND public.lab_arena__execute_assignment_calls_v1(v_run.assignment_id,p_provider)>=
      v_quota::BIGINT*(v_round.configuration_doc->>'max_attempts_per_assignment')::BIGINT THEN
    v_reason:='per_icp_quota';
  END IF;$new3_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'4da081265076dda69d237f11dafd1d790771e659503d8eedb6f550ff7d3f3211' THEN RAISE EXCEPTION 'Host retry474 postimage differs'; END IF;
 EXECUTE d;
 IF encode(extensions.digest(pg_get_functiondef('lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure),'sha256'),'hex')<>'4da081265076dda69d237f11dafd1d790771e659503d8eedb6f550ff7d3f3211' THEN RAISE EXCEPTION 'Host retry474 readback differs'; END IF;
END;
$patch474_3$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
