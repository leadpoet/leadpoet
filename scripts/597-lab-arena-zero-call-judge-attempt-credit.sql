-- Proved zero-call launcher failures do not consume the five judge attempts.
-- Reuse migration504's authenticated receipt, scoped to one scoring generation.
-- Retain physical IDs, backoff, frozen deadline, quarantine, calls and settlement.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
-- Use the existing migration589 ownership-transfer pattern and restore the
-- exact schema ACL before commit. This temporary snapshot contains metadata.
CREATE TEMP TABLE arena597_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
DO $temporary_create597$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena597_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_create597$;

DO $helper597_precondition$
BEGIN
  IF to_regprocedure('public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)') IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
      WHERE p.oid=to_regprocedure('public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)')
        AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')='41f2a3f5fa3a5abf0e6106df52e98c59121a55e99e22daad88673ff8f45d9b84'
        AND pg_get_userbyid(p.proowner)='lab_arena_owner'
        AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
        AND p.prosecdef AND p.provolatile='s'
        AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='sql') THEN
    RAISE EXCEPTION 'zero-call judge helper security or source differs' USING ERRCODE='55000';
  END IF;
END;
$helper597_precondition$;

CREATE OR REPLACE FUNCTION public.lab_arena__score_attempt_count_v1(p_run public.lab_arena_runs)
RETURNS INTEGER LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $score_attempt_count597$
 SELECT GREATEST(0, p_run.attempt - (
   SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
   WHERE generation_root.assignment_id=p_run.assignment_id
     AND generation_root.scoring_generation_floor=p_run.scoring_generation_floor) + 1 - (
   SELECT count(*)::INTEGER FROM public.lab_arena_runs host
   JOIN public.lab_arena_rounds round ON round.round_id=host.round_id
   WHERE p_run.kind='score' AND host.kind='score'
     AND host.assignment_id=p_run.assignment_id AND host.attempt<=p_run.attempt
     AND host.scoring_generation_floor=p_run.scoring_generation_floor
     AND round.evaluation_date::DATE>=DATE '2026-10-14'
     AND (host.round_id,host.submission_id,host.miner_hotkey,host.stage,host.icp_position)
       IS NOT DISTINCT FROM
         (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
     -- Exact migration504 receipt proof. Setup/source errors are not exempt.
     AND round.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1'
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
  ));
$score_attempt_count597$;
ALTER FUNCTION public.lab_arena__score_attempt_count_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $zero_call_judge_attempt597$
DECLARE target RECORD; patch RECORD; function_oid OID;
  definition TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
      WHERE p.oid='public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)'::regprocedure
        AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')='41f2a3f5fa3a5abf0e6106df52e98c59121a55e99e22daad88673ff8f45d9b84'
        AND pg_get_userbyid(p.proowner)='lab_arena_owner'
        AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
        AND p.prosecdef AND p.provolatile='s'
        AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='sql') THEN
    RAISE EXCEPTION 'zero-call judge helper readback differs' USING ERRCODE='55000';
  END IF;
  FOR target IN SELECT * FROM (VALUES
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','fd89e06bf90317d579504869210632457f6fb9bcb79aabca16048e5bc119bc22','df4e15dc9008ff0774e60f56ac7780b85e6783b7c12add61adac94ba8cbf68aa'),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)','1f0f2372abcbaee30424af6ba76258d39e081aa8d197d82707b550ad0ad4abf0','6c5ee308109b55b3b23ee1f1631d894cd72b8582ad1180c898e25f7ae9106ac0'),
    ('public.lab_arena_expire_leases(text)','2e3f75408b96884640580e8c4f4cd8855750a9fd66a0f9acd95f29b113770307','b5887c1e2afa6ac5ff342f1dc4ed1f0fc390b92b7302533a3ba31db964dbfaca')
  ) AS targets(signature,preimage,postimage) LOOP
    function_oid:=to_regprocedure(target.signature);
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
        p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO definition,metadata,identity
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid;
    IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
      RAISE EXCEPTION 'zero-call judge attempt security differs: %',target.signature USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>target.preimage THEN
      RAISE EXCEPTION 'zero-call judge attempt preimage differs: %',target.signature USING ERRCODE='55000';
    END IF;
    updated:=definition;
    FOR patch IN SELECT * FROM (VALUES
      ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',$old597$    AND (runs.kind<>'score' OR runs.attempt - (
      SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
      WHERE generation_root.assignment_id=runs.assignment_id
        AND generation_root.scoring_generation_floor=runs.scoring_generation_floor) < 5)$old597$,$new597$    AND (runs.kind<>'score' OR public.lab_arena__score_attempt_count_v1(runs) <= 5)$new597$),
      ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',$old597$     AND (v_run.kind<>'score' OR v_run.attempt - (
       SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
       WHERE generation_root.assignment_id=v_run.assignment_id
         AND generation_root.scoring_generation_floor=v_run.scoring_generation_floor) < 4)$old597$,$new597$     AND (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5)$new597$),
      ('public.lab_arena_expire_leases(text)',$old597$    IF (v_run.kind<>'score' OR v_run.attempt - (
      SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
      WHERE generation_root.assignment_id=v_run.assignment_id
        AND generation_root.scoring_generation_floor=v_run.scoring_generation_floor) < 4) AND ($old597$,$new597$    IF (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5) AND ($new597$),
      ('public.lab_arena_expire_leases(text)',$old597$      AND candidate.attempt - (
        SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
        WHERE generation_root.assignment_id=candidate.assignment_id
          AND generation_root.scoring_generation_floor=candidate.scoring_generation_floor) < 4$old597$,$new597$      AND public.lab_arena__score_attempt_count_v1(candidate) < 5$new597$)
    ) AS patches(signature,old_text,new_text) WHERE signature=target.signature LOOP
      IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
        RAISE EXCEPTION 'zero-call judge attempt anchor differs: %',target.signature USING ERRCODE='55000';
      END IF;
      updated:=replace(updated,patch.old_text,patch.new_text);
    END LOOP;
    IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'zero-call judge attempt postimage differs: %',target.signature USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    IF encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage
      OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'zero-call judge attempt readback differs: %',target.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$zero_call_judge_attempt597$;
DO $restore_schema_acl597$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena597_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
     (SELECT nspacl FROM pg_temp.arena597_schema_acl) THEN
    RAISE EXCEPTION 'zero-call judge attempt schema ACL differs' USING ERRCODE='42501';
  END IF;
END;
$restore_schema_acl597$;
NOTIFY pgrst,'reload schema';
COMMIT;
