-- An authenticated missing host interpreter can consume a lease before the miner starts.
-- Exclude at most one such lease per assignment from its normal attempt allowance.
-- The existing recovery path still enforces cutoffs, costs, quotas and accepted-work checks.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $credit711$
DECLARE
 target OID := 'public.lab_arena_execution_attempt_count(public.lab_arena_runs)'::regprocedure;
 definition TEXT; metadata JSONB;
 pre CONSTANT TEXT := '7017065b833239221c74bba24788d97dc09558edd7fb86f5d4ae1ffa6b7b546d';
 post CONSTANT TEXT := 'afcf68832b1eca26491f41289eaa3bd00115fc949552cc84b167d5f3f9ff876a';
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,metadata
 FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::oid)<>'lab_arena_owner'
    OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::jsonb
    OR metadata->>'prosecdef'<>'true' THEN RAISE EXCEPTION 'Attempt count authority differs'; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')=post THEN RETURN; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')<>pre THEN RAISE EXCEPTION 'Attempt count preimage differs'; END IF;
 definition:=replace(definition,$old$     ))))::INTEGER;$old$,$new$     ))) - CASE WHEN p_run.kind='execute' AND EXISTS (
   SELECT 1 FROM public.lab_arena_runs host
   JOIN public.lab_arena_rounds round ON round.round_id=host.round_id
   WHERE host.assignment_id=p_run.assignment_id AND host.attempt<=p_run.attempt
     AND (host.round_id,host.submission_id,host.miner_hotkey,host.stage,host.icp_position)
       IS NOT DISTINCT FROM
         (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
     AND round.evaluation_date::DATE>=DATE '2026-11-21'
     AND round.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1'
     AND host.kind='execute' AND host.status='failed' AND host.terminal_cause='provider_error'
     AND host.output_ref IS NULL AND NOT host.champion_restart_required
     AND host.result_doc->>'terminal_status'='provider_error'
     AND host.result_doc#>'{resource_summary,provider_call_count}'='0'::JSONB
     AND host.result_doc#>'{resource_summary,wall_seconds}'='0'::JSONB
     AND host.lease_token_hash IS NOT NULL AND host.claim_request_hash IS NOT NULL
     AND host.claim_response->>'status'='leased' AND host.claim_response->>'run_id'=host.run_id
     AND host.lease_generation>0 AND host.claim_response->'lease_generation'=to_jsonb(host.lease_generation)
     AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=host.run_id)
     AND NOT EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events e
       WHERE e.run_id=host.run_id AND e.event_kind LIKE 'provider.%')
     AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events e
       WHERE e.run_id=host.run_id
         AND (e.runner_hotkey,e.round_id,e.submission_id,e.miner_hotkey,e.assignment_id,e.stage,e.icp_position,e.attempt,e.run_kind)
           = (host.runner_hotkey,host.round_id,host.submission_id,host.miner_hotkey,host.assignment_id,host.stage,host.icp_position,host.attempt,host.kind)
         AND e.event_kind='runtime.error' AND e.content->>'status'='provider_error'
         AND e.content->>'failure_stage'='setup' AND e.content->>'error_class'='DependencyInstallInfrastructureError'
         AND e.content->>'failure_kind'='installer_error'
         AND e.content->>'diagnostic_detail'='compatible dependency installer unavailable: CPython 3.11 required')
 ) THEN 1 ELSE 0 END)::INTEGER;$new$);
 IF encode(extensions.digest(definition,'sha256'),'hex')<>post THEN RAISE EXCEPTION 'Attempt count postimage differs'; END IF;
 EXECUTE definition;
 IF NOT EXISTS(SELECT 1 FROM pg_proc p WHERE p.oid=target AND to_jsonb(p)-'prosrc'=metadata
    AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')=post)
 THEN RAISE EXCEPTION 'Attempt count readback differs'; END IF;
END;
$credit711$;
COMMIT;
