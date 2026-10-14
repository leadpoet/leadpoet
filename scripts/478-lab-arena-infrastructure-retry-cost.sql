-- Charge competition budgets only for work not lost to proved infrastructure.
-- Opt in through a frozen configuration. Raw ledger and actual counters remain.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Reject an unexpected concurrent database definition before any DDL.
DO $exact_release$
DECLARE item RECORD; definition TEXT; identity JSONB;
BEGIN
 FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena_execution_attempt_count(public.lab_arena_runs)','64885c7aa2a8cf00209ff4db558d71ba86c41aff26fb7854238e929c70f40632','ef4a51fa5955c0f33b0bbb85e8ac825a4f2af311f93a4eaa9cc2198a8e642668'),
    ('public.lab_arena__successful_icp_cost_state(text,text,integer)','9998679746cf9a5b18777e5ee8e14ea2e53bb771de9ae3a34a2c5df8918183f7','a227265beb00a1918e26079c1b878ab846dd5849d1928e79bfd4fcf519ba4b97'),
    ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','4da081265076dda69d237f11dafd1d790771e659503d8eedb6f550ff7d3f3211','6c4aa26ea4fe38c070e4096cc91b979463d99583ae33c8faf8ad0c52fd6e304a'),
    ('public.lab_arena__execute_host_retry_ready_v1(public.lab_arena_runs)','aa44c2784f90881d5f78c79dc390a0c7661514a29daf04fff137543c28250053','713c87e768ccb10db8b9270c19d50afbfd63d2d54601b852e4d7a3a46cf6d460'),
    ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)','f256ed3ec6027d620fb9eeebe81040e6cd6f905817c29fe160424923947f2b3e','e2926293aba28e4e8a0b3ef2ed6605790fa2dc0f5ea865a77d33b40d7ed6be5b'),
    ('public.lab_arena_run_quota_snapshot_v2(text,text)','a5758914c60b7012f01b9ecf3b71ebee11e4cbda51f8a0c578ebcb9dcee03837','e8b214e32357dbff32a27ed82443a9f8ae161fe1a07baa4ec3e763b294f10f8a'),
    ('public.lab_arena_rounds_write_once_v1()','0e5347bb11dc6cbfe5949d4b2e43ae2ae56264713d5990d2ce9533352d440cb3','f4213c6f65d83ce631be50df1ca0a2a3ee6ad6bbf7f47c327496b8b716dd0b18'),
    ('public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)','9a913c3b50b61240152d51417d11a8d0e18df88d4ca1372a1d18b9f53e25c1d7','dcc78b9c7615def5db85a9544b8ae8269a10ed2f6fa153293acb68611ee69c09'),
    ('public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)','2843d29c44c01fe7e0a50235607f14f7f6387f6688f3b3c7866a9f6cc0b5bf74','a245d37af1ec82e5b065cfe74b1222d6aef7966b3cd56c06bfc75777d359e8b3')
 ) expected(signature,preimage,postimage) LOOP
   SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
     INTO definition,identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
     WHERE p.oid=item.signature::REGPROCEDURE;
   IF identity->>0 IS DISTINCT FROM 'lab_arena_owner'
     OR identity->>2 IS DISTINCT FROM (CASE WHEN item.signature='public.lab_arena_rounds_write_once_v1()' THEN 'false' ELSE 'true' END)
     OR identity->4 IS DISTINCT FROM (CASE WHEN item.signature='public.lab_arena_rounds_write_once_v1()'
       THEN jsonb_build_array('search_path=pg_catalog') ELSE jsonb_build_array('search_path=pg_catalog, public') END)
     OR encode(extensions.digest(definition,'sha256'),'hex') NOT IN (item.preimage,item.postimage) THEN
     RAISE EXCEPTION 'Infrastructure retry cost exact release preimage differs: %',item.signature;
   END IF;
 END LOOP;
END;
$exact_release$;

CREATE OR REPLACE FUNCTION public.lab_arena__execute_host_interruption_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $host_proof$
 SELECT EXISTS (SELECT 1 FROM public.lab_arena_runs host
 JOIN public.lab_arena_rounds round ON round.round_id=host.round_id
 WHERE host.run_id=p_run.run_id AND host.kind='execute'
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
             AND error.occurred_at)));
$host_proof$;
ALTER FUNCTION public.lab_arena__execute_host_interruption_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__execute_host_interruption_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Reuse the exact existing host attribution; attempt counts remain identical.
DO $host_count$
DECLARE d TEXT; old TEXT := $old$   AND (host.status='leased' OR host.status='failed'
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
$old$;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena_execution_attempt_count(public.lab_arena_runs)'::REGPROCEDURE) INTO d;
 IF strpos(d,'lab_arena__execute_host_interruption_v1')=0 THEN
   IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost host proof preimage differs'; END IF;
   EXECUTE replace(d,old,'   AND public.lab_arena__execute_host_interruption_v1(host)))::INTEGER;'||chr(10));
 END IF;
END;
$host_count$;

CREATE OR REPLACE FUNCTION public.lab_arena__execute_infrastructure_cost_exempt_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $cost_proof$
 SELECT COALESCE(r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1',FALSE)
   AND p_run.kind='execute' AND p_run.status='failed' AND p_run.output_ref IS NULL
   AND NOT p_run.champion_restart_required
   AND (
     public.lab_arena__execute_host_interruption_v1(p_run)
     -- A host launcher that never starts the model has no submitted outcome.
     -- Use the gateway's existing authenticated expiry receipt, not model text.
     OR (p_run.terminal_cause='lease_expired' AND p_run.result_doc IS NULL
       AND p_run.terminal_doc->>'recovery_reason' IN
         ('authenticated_zero_call_runtime_host_error','authenticated_zero_call_setup_host_error')
       AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=p_run.run_id))
     OR (p_run.terminal_cause='provider_error'
       AND p_run.result_doc->>'terminal_status'='provider_error'
       AND jsonb_typeof(p_run.result_doc#>'{resource_summary,provider_unresolved_failure_count}')='number'
       AND (p_run.result_doc#>>'{resource_summary,provider_unresolved_failure_count}')::NUMERIC>0
       AND p_run.runner_hotkey IS NOT NULL AND p_run.lease_token_hash IS NOT NULL
       AND p_run.claim_request_id IS NOT NULL AND p_run.lease_generation>0
       AND p_run.claim_response->>'status'='leased'
       AND p_run.claim_response->'lease_generation'=to_jsonb(p_run.lease_generation)
       AND EXISTS (
         -- The trusted worker's unresolved count uses parameter-aware
         -- operation identities. Require independent broker-ledger proof.
         SELECT 1 FROM (
           SELECT DISTINCT ON (l.call_identity) l.*
           FROM public.lab_arena_ledger l
           WHERE l.run_id=p_run.run_id AND l.call_identity IS NOT NULL
           ORDER BY l.call_identity,l.entry_id DESC
         ) head
         WHERE (head.round_id,head.submission_id,head.miner_hotkey,head.stage)
           IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage)
           AND head.entry_kind IN ('settlement','uncertain')
           AND head.call_identity ~ '^sha256:[0-9a-f]{64}$'
           AND NOT EXISTS (SELECT 1 FROM public.lab_arena_deepline_call_responses response
             WHERE response.run_id=head.run_id AND response.call_identity=head.call_identity
               AND response.terminal_response->'call_succeeded'='true'::JSONB)
           AND head.terminal_response->'account_failure_evidence' IS NULL
           AND head.entry_doc#>'{call,account_failure_evidence}' IS NULL
           AND (
             (head.entry_kind='settlement'
               AND head.terminal_response->'call_succeeded'='false'::JSONB
               AND (head.terminal_response->>'status'='429'
                 OR head.terminal_response->>'status' ~ '^5[0-9][0-9]$'))
             OR (head.entry_kind='settlement'
               AND head.terminal_response->'call_succeeded' IS DISTINCT FROM 'true'::JSONB
               AND (head.entry_doc->>'deepline_delayed_reconciliation'='true'
                 OR head.entry_doc->>'openrouter_delayed_reconciliation'='true')
               AND EXISTS (SELECT 1 FROM public.lab_arena_ledger original
                 WHERE original.entry_id::TEXT=head.entry_doc->>'reconciled_uncertainty_entry_id'
                   AND original.entry_kind='uncertain'
                   AND (original.run_id,original.round_id,original.submission_id,original.miner_hotkey,
                     original.stage,original.call_identity,original.provider,original.operation_id,original.funding_source)
                     IS NOT DISTINCT FROM (head.run_id,head.round_id,head.submission_id,head.miner_hotkey,
                       head.stage,head.call_identity,head.provider,head.operation_id,head.funding_source)
                   AND original.entry_doc->>'reason'='worker_reported'
                   AND original.entry_doc#>'{call,call_succeeded}'='false'::JSONB
                   AND original.entry_doc#>'{call,account_failure_evidence}' IS NULL
                   AND (original.entry_doc#>>'{call,reason}'='transport_failure'
                     OR (original.entry_doc#>>'{call,reason}'='missing_provider_cost'
                       AND (original.entry_doc#>>'{call,provider_status}'='429'
                         OR original.entry_doc#>>'{call,provider_status}' ~ '^5[0-9][0-9]$')))))
             OR (head.entry_kind='uncertain' AND head.entry_doc->>'reason'='worker_reported'
               AND head.entry_doc#>'{call,call_succeeded}'='false'::JSONB
               AND (head.entry_doc#>>'{call,reason}'='transport_failure'
                 OR (head.entry_doc#>>'{call,reason}'='missing_provider_cost'
                   AND (head.entry_doc#>>'{call,provider_status}'='429'
                     OR head.entry_doc#>>'{call,provider_status}' ~ '^5[0-9][0-9]$'))))
           )
           AND EXISTS (SELECT 1 FROM public.lab_arena_ledger dispatched
             WHERE dispatched.call_identity=head.call_identity AND dispatched.entry_kind='dispatch'
               AND (dispatched.run_id,dispatched.round_id,dispatched.submission_id,dispatched.miner_hotkey,
                 dispatched.stage,dispatched.provider,dispatched.operation_id,dispatched.funding_source)
                 IS NOT DISTINCT FROM (head.run_id,head.round_id,head.submission_id,head.miner_hotkey,
                   head.stage,head.provider,head.operation_id,head.funding_source))
       ))
   ) FROM public.lab_arena_rounds r WHERE r.round_id=p_run.round_id;
$cost_proof$;
ALTER FUNCTION public.lab_arena__execute_infrastructure_cost_exempt_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__execute_infrastructure_cost_exempt_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $cost_state$
DECLARE d TEXT; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena__successful_icp_cost_state(text,text,integer)'::REGPROCEDURE) INTO d;
 IF strpos(d,'lab_arena_infrastructure_retry_cost_state_v1')>0 THEN RETURN; END IF;
 old := '    SELECT runs.run_id';
 new := '    SELECT runs.run_id, public.lab_arena__execute_infrastructure_cost_exempt_v1(runs) AS infrastructure_exempt';
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Infrastructure cost state run preimage differs'; END IF;
 d:=replace(d,old,new);
 old := '      ledger.entry_kind, ledger.amount_microusd,';
 new := '      ledger.entry_kind, ledger.amount_microusd, COALESCE(runs.infrastructure_exempt,FALSE) AS infrastructure_exempt,';
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Infrastructure cost state head preimage differs'; END IF;
 d:=replace(d,old,new);
 old := '  ) FROM heads;';
 new := $new$  ) || CASE WHEN EXISTS (SELECT 1 FROM public.lab_arena_rounds r
    WHERE r.round_id=p_round_id AND r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1')
  THEN jsonb_build_object(
    -- lab_arena_infrastructure_retry_cost_state_v1: settled totals are actual.
    -- Only the competition totals exclude failed infrastructure attempts.
    'competition_settled_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND NOT infrastructure_exempt),0)::BIGINT,
    'competition_successful_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND call_succeeded IS TRUE AND NOT infrastructure_exempt),0)::BIGINT,
    'excluded_infrastructure_settled_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND infrastructure_exempt),0)::BIGINT,
    'excluded_infrastructure_successful_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND call_succeeded IS TRUE AND infrastructure_exempt),0)::BIGINT
  ) ELSE '{}'::JSONB END FROM heads;$new$;
 IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
   RAISE EXCEPTION 'Infrastructure cost state aggregate preimage differs'; END IF;
 EXECUTE replace(d,old,new);
END;
$cost_state$;

DO $cost_consumers$
DECLARE d TEXT; signature TEXT; old TEXT; new TEXT;
BEGIN
 FOREACH signature IN ARRAY ARRAY[
   'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
   'public.lab_arena__execute_host_retry_ready_v1(public.lab_arena_runs)',
   'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',
   'public.lab_arena_run_quota_snapshot_v2(text,text)'
 ] LOOP
   SELECT pg_get_functiondef(signature::REGPROCEDURE) INTO d;
   IF strpos(d,'lab_arena_infrastructure_retry_cost_consumer_v1')>0 THEN CONTINUE; END IF;
   IF signature LIKE '%reserve_call%' THEN
     old := '(v_icp_cost ->> ''settled_microusd'')::BIGINT';
     new := 'COALESCE(v_icp_cost ->> ''competition_settled_microusd'',v_icp_cost ->> ''settled_microusd'')::BIGINT';
   ELSIF signature LIKE '%retry_ready%' THEN
     old := '(state->>''settled_microusd'')::BIGINT';
     new := 'COALESCE(state->>''competition_settled_microusd'',state->>''settled_microusd'')::BIGINT';
   ELSIF signature LIKE '%eligibility%' THEN
     old := 'v_spend := (v_state ->> ''successful_microusd'')::BIGINT;';
     new := 'v_spend := COALESCE(v_state ->> ''competition_successful_microusd'',v_state ->> ''successful_microusd'')::BIGINT;';
   ELSE
     old := '  v_execution := v_cost -> ''execution'';';
     new := $quota_new$  v_execution := v_cost -> 'execution';
  IF v_execution ? 'competition_settled_microusd' THEN
    v_execution := v_execution || jsonb_build_object(
      'settled_microusd',v_execution->'competition_settled_microusd',
      'successful_microusd',v_execution->'competition_successful_microusd');
  END IF;$quota_new$;
   END IF;
   IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost consumer preimage differs: %',signature; END IF;
   d:=replace(d,old,new);
   d:=replace(d,'BEGIN','BEGIN'||chr(10)||'  -- lab_arena_infrastructure_retry_cost_consumer_v1');
   EXECUTE d;
 END LOOP;
END;
$cost_consumers$;

-- Enroll an already-open round only in its original atomic benchmark freeze.
-- No existing row is updated by this migration.
DO $freeze_enrollment$
DECLARE d TEXT; signature TEXT; old TEXT; new TEXT;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE) INTO d;
 IF strpos(d,'lab_arena_infrastructure_cost_commit_v1')=0 THEN
   old := '  IF NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc THEN';
   new := $new$  -- lab_arena_infrastructure_cost_commit_v1
  IF NEW.configuration_doc->'infrastructure_retry_cost_policy' IS DISTINCT FROM
     OLD.configuration_doc->'infrastructure_retry_cost_policy' THEN
    IF (OLD.status='open' AND NEW.status='committed'
      AND NOT OLD.configuration_doc ? 'infrastructure_retry_cost_policy'
      AND NEW.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1'
      AND OLD.configuration_doc->>'sourcing_cost_eligibility_policy'='successful_calls_per_icp_v1'
      AND OLD.benchmark_ref IS NULL AND OLD.icp_set_date IS NULL
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=OLD.round_id)) IS NOT TRUE THEN
      RAISE EXCEPTION 'Infrastructure cost policy must freeze before execution' USING ERRCODE='42501';
    END IF;
  END IF;
  IF NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc THEN$new$;
   IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost commit guard preimage differs'; END IF;
   d:=replace(d,old,new);
   old := $anchor$NEW.configuration_doc - 'scorer_image_digest'$anchor$;
   new := $anchor$NEW.configuration_doc - 'infrastructure_retry_cost_policy' - 'scorer_image_digest'$anchor$;
   IF (length(d)-length(replace(d,old,'')))<>2*length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost new configuration guard differs'; END IF;
   d:=replace(d,old,new);
   old := $anchor$OLD.configuration_doc - 'scorer_image_digest'$anchor$;
   new := $anchor$OLD.configuration_doc - 'infrastructure_retry_cost_policy' - 'scorer_image_digest'$anchor$;
   IF (length(d)-length(replace(d,old,'')))<>2*length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost old configuration guard differs'; END IF;
   EXECUTE replace(d,old,new);
 END IF;
 FOREACH signature IN ARRAY ARRAY[
   'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)',
   'public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)'
 ] LOOP
   SELECT pg_get_functiondef(signature::REGPROCEDURE) INTO d;
   IF strpos(d,'lab_arena_infrastructure_cost_commit_v1')>0 THEN CONTINUE; END IF;
   old := '      configuration_doc = ';
   new := $new$      -- lab_arena_infrastructure_cost_commit_v1
      configuration_doc = CASE WHEN
          v_round.configuration_doc->>'sourcing_cost_eligibility_policy'='successful_calls_per_icp_v1'
          AND NOT v_round.configuration_doc ? 'infrastructure_retry_cost_policy'
          AND v_round.benchmark_ref IS NULL AND v_round.icp_set_date IS NULL
          AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round_id)
        THEN jsonb_build_object('infrastructure_retry_cost_policy','trusted_infrastructure_v1')
        ELSE '{}'::JSONB END || $new$;
   IF (length(d)-length(replace(d,old,'')))<>length(old) THEN
     RAISE EXCEPTION 'Infrastructure cost benchmark freeze preimage differs: %',signature; END IF;
   EXECUTE replace(d,old,new);
 END LOOP;
END;
$freeze_enrollment$;

CREATE OR REPLACE FUNCTION public.lab_arena_infrastructure_retry_cost_schema_v1()
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $schema$
BEGIN
 IF strpos(pg_get_functiondef('public.lab_arena__successful_icp_cost_state(text,text,integer)'::REGPROCEDURE),
     'lab_arena_infrastructure_retry_cost_state_v1')=0 THEN
   RAISE EXCEPTION 'Infrastructure cost schema incomplete'; END IF;
 RETURN jsonb_build_object('schema_version','leadpoet.lab_arena.infrastructure_retry_cost_schema.v1',
   'version',478,'policy','trusted_infrastructure_v1');
END;
$schema$;
ALTER FUNCTION public.lab_arena_infrastructure_retry_cost_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_infrastructure_retry_cost_schema_v1()
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_infrastructure_retry_cost_schema_v1() TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
