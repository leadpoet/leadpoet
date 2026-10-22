-- Finalize validated terminal executions with missing scores as NOT EVALUATED.
-- No execution, judgment, score, accounting or historical record is changed.
-- SQL checks the authority core and known keys. Authenticated completion and
-- the full service validator retain validation of optional nested diagnostics.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $unmeasured547$
DECLARE target OID:=to_regprocedure('public.lab_arena__participant_incomplete_reason_v1(text,text)');
 definition TEXT;metadata JSONB;updated TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
  AND pg_get_userbyid(p.proowner)='lab_arena_owner'AND p.prosecdef AND p.provolatile='s'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]AND l.lanname='plpgsql';
 IF encode(extensions.digest(definition,'sha256'),'hex')='f0947c0d204d0883d926781cfe03ba22ad2b940ea16cf7618de90a9533cd44c4' THEN RETURN; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')<>'ccb1092649da716d45dd346bce21aadaf4542d3559586805e89ccbe54a0a6eee' THEN
  RAISE EXCEPTION 'Unmeasured execution finalization source/security differs'; END IF;
 updated:=$definition547$CREATE OR REPLACE FUNCTION public.lab_arena__participant_incomplete_reason_v1(p_round_id text, p_submission_id text)
 RETURNS text
 LANGUAGE plpgsql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE r public.lab_arena_rounds; source public.lab_arena_runs; judge public.lab_arena_runs;
 position INTEGER; total INTEGER; reason TEXT; cause TEXT; route BOOLEAN;
BEGIN
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id;
 IF NOT FOUND OR r.status='cancelled'
  OR r.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
  OR jsonb_typeof(r.configuration_doc->'sourcing_cost_eligibility_policy') IS DISTINCT FROM 'string'
  OR r.configuration_doc->>'sourcing_cost_eligibility_policy' NOT IN('successful_calls_per_icp_v1','successful_calls_flat_per_icp_v1')
  OR jsonb_typeof(r.configuration_doc#>'{schedule,publication_deadline}') IS DISTINCT FROM 'string'
  OR now()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p WHERE p->>'submission_id'=p_submission_id
   AND NOT coalesce((p->>'is_king')::BOOLEAN,FALSE)) THEN RETURN NULL; END IF;
 -- The immutable publication is the finalized authority for promotion and
 -- reward gates. Historical publications without this typed outcome get no
 -- new omission authority; late receipts cannot change a finalized outcome.
 IF r.status='published' THEN
  IF jsonb_typeof(r.publication_doc->'final_ranking') IS DISTINCT FROM 'array' THEN RETURN NULL; END IF;
  SELECT CASE WHEN count(*)=1 AND bool_and(
    ranked->'final_score'='null'::JSONB AND ranked->'cost_summary'='null'::JSONB
    AND ranked->'eligible'='false'::JSONB AND ranked->>'eligibility_reason'='execution_incomplete'
    AND ranked->>'evaluation'='not_evaluated'
    AND ranked->>'evaluation_reason' IN('openrouter_zdr_route_unavailable',
      'stage_closed','judge_error','judge_timeout','lease_expired','worker_lost','provider_error',
      'credential_error','result_rejected','judge_missing','judge_pending','execution_missing','execution_score_missing')
    AND ranked->>'evaluation_status'=CASE WHEN ranked->>'evaluation_reason'='openrouter_zdr_route_unavailable'
       THEN 'provider_blocked' ELSE 'evaluation_incomplete' END)
   THEN min(ranked->>'evaluation_reason') END INTO reason
   FROM jsonb_array_elements(r.publication_doc->'final_ranking')ranked
   WHERE ranked->>'submission_id'=p_submission_id;
  RETURN reason;
 END IF;
 total:=(r.configuration_doc->>'stage_1_icp_count')::INTEGER+(r.configuration_doc->>'stage_2_icp_count')::INTEGER;
 IF total NOT BETWEEN 2 AND 100 THEN RETURN NULL; END IF;
 FOR position IN 0..total-1 LOOP
  SELECT * INTO source FROM public.lab_arena_runs a WHERE a.round_id=p_round_id
   AND a.submission_id=p_submission_id AND a.kind='execute' AND a.icp_position=position
   ORDER BY (a.status='accepted') DESC,a.attempt DESC,a.run_id DESC LIMIT 1;
  IF NOT FOUND THEN reason:=CASE WHEN reason='execution_score_missing' THEN 'execution_missing' ELSE coalesce(reason,'execution_missing') END; CONTINUE; END IF;
  IF source.status IN('leased','submitted') THEN RETURN NULL; END IF;
  -- Validate every retained scored ICP, including the complete portion of an
  -- incomplete participant. An omission never masks malformed scored evidence.
  IF source.per_icp_score IS NOT NULL AND NOT coalesce((public.lab_arena__integrity_submission_summary(
    p_round_id,p_submission_id,ARRAY[position])->>'valid')::BOOLEAN,FALSE) THEN RETURN NULL; END IF;
  SELECT * INTO judge FROM public.lab_arena_runs j WHERE j.round_id=p_round_id
   AND j.kind='score' AND j.scoring_generation_floor=r.scoring_generation_floor
   AND (j.scored_run_id=source.run_id OR j.assignment_id=source.assignment_id||':score')
   ORDER BY (j.status='accepted') DESC,j.stage_generation DESC,j.attempt DESC,j.run_id DESC LIMIT 1;
  IF source.status='accepted' THEN
   IF source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL THEN RETURN NULL; END IF;
   IF judge.status='accepted' THEN
    -- The independent finality helper validates current accepted binding.
    -- Its materialized score is required even for a route-blocked empty result.
    IF source.per_icp_score IS NULL OR judge.terminal_cause IS DISTINCT FROM 'accepted'
     OR judge.output_ref IS NULL OR judge.scored_run_id IS DISTINCT FROM source.run_id
     OR (judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position) IS DISTINCT FROM
        (source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
     OR NOT public.lab_arena__qualification_doc_valid(source.qualification_doc)
     OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(r) THEN
       public.lab_arena__company_judgment_refs_v1(r,judge.company_judgment_refs)
       AND judge.judgment_cache_key IS NULL AND judge.judgment_input_hash IS NULL AND judge.judgment_scope_doc IS NULL
      ELSE judge.judgment_scope_doc->>'scorer_image_digest'=r.configuration_doc->>'scorer_image_digest'
       AND judge.judgment_scope_doc->>'scorer_image_reference'=r.configuration_doc->>'scorer_image_reference'
     END) IS NOT TRUE THEN RETURN NULL; END IF;
   ELSIF source.per_icp_score IS NOT NULL THEN RETURN NULL;
   END IF;
   route:=FALSE;
   IF (source.qualification_doc->'companies'='[]'::JSONB AND public.lab_arena__company_judgment_policy_v1(r)
     AND judge.company_judgment_refs='[]'::JSONB) OR judge.status IS DISTINCT FROM 'accepted' THEN
    route:=public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id);
   END IF;
   IF route THEN reason:='openrouter_zdr_route_unavailable';
   ELSIF judge.status='accepted' THEN CONTINUE;
   ELSE
    IF judge.run_id IS NOT NULL AND judge.status NOT IN('pending','failed') THEN RETURN NULL; END IF;
    cause:=coalesce(judge.terminal_cause,CASE WHEN judge.run_id IS NULL THEN 'judge_missing' ELSE 'judge_pending' END);
    IF cause NOT IN('stage_closed','judge_error','judge_timeout','lease_expired','worker_lost','provider_error','credential_error','result_rejected','judge_missing','judge_pending') THEN RETURN NULL; END IF;
    reason:=CASE WHEN reason='execution_score_missing' THEN cause ELSE coalesce(reason,cause) END;
   END IF;
  ELSE
   IF judge.status='accepted' THEN RETURN NULL; END IF;
   cause:=coalesce(source.terminal_cause,'');
   IF source.status IS DISTINCT FROM 'failed' OR cause NOT IN(
     'lease_expired','worker_lost','result_rejected','provider_error','judge_error','judge_timeout','stage_closed') THEN
    -- A label is not a score. Keep ordinary materialized zeros; an exact
    -- failed model result without a score is explicitly not evaluated.
    IF source.per_icp_score IS NOT NULL THEN CONTINUE; END IF;
    IF source.status IS DISTINCT FROM 'failed' OR cause NOT IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')
     OR source.output_ref IS NOT NULL OR source.qualification_doc IS NOT NULL
     OR jsonb_typeof(source.result_doc) IS DISTINCT FROM 'object'
     OR source.result_doc->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.run_result.v1'
     OR source.result_doc->>'terminal_status' IS DISTINCT FROM cause
     OR jsonb_typeof(source.result_doc->'resource_summary') IS DISTINCT FROM 'object'
     OR EXISTS(SELECT 1 FROM jsonb_object_keys(CASE WHEN jsonb_typeof(source.result_doc)='object' THEN source.result_doc ELSE '{}'::JSONB END)k
      WHERE k NOT IN('schema_version','company_judgment_completion_request_hash','company_judgment_consumed_indexes','resource_summary','started_at','finished_at','terminal_status','checkpoint_transition','failure_diagnostic'))
     OR EXISTS(SELECT 1 FROM jsonb_object_keys(CASE WHEN jsonb_typeof(source.result_doc->'resource_summary')='object'
      THEN source.result_doc->'resource_summary' ELSE '{}'::JSONB END)k WHERE k NOT IN('wall_seconds','cpu_seconds','max_rss_bytes','stdout_bytes','stderr_bytes','provider_call_count','provider_refusal_count','provider_local_refusal_count','provider_unresolved_failure_count','worker_observations','validator_source_commit','validator_source_origin','validator_source_dirty','gateway_claim_source_commit','grading_image_digest','grading_image_reference','runtime_image_digest','runtime_image_reference','web_egress'))
     OR jsonb_typeof(source.result_doc->'started_at') IS DISTINCT FROM 'string'
     OR jsonb_typeof(source.result_doc->'finished_at') IS DISTINCT FROM 'string'
     OR coalesce(source.result_doc->>'started_at','') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
     OR coalesce(source.result_doc->>'finished_at','') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
     OR EXISTS(SELECT 1 FROM unnest(ARRAY['wall_seconds','cpu_seconds','max_rss_bytes','stdout_bytes','stderr_bytes','provider_call_count'])k
      WHERE CASE WHEN jsonb_typeof(source.result_doc#>ARRAY['resource_summary',k])='number'
       THEN (source.result_doc#>>ARRAY['resource_summary',k])::NUMERIC<0
        OR(k NOT IN('wall_seconds','cpu_seconds')AND (source.result_doc#>>ARRAY['resource_summary',k]) !~ '^[0-9]+$') ELSE TRUE END) THEN RETURN NULL; END IF;
    BEGIN
     PERFORM (source.result_doc->>'started_at')::TIMESTAMPTZ,(source.result_doc->>'finished_at')::TIMESTAMPTZ,
      (source.result_doc#>>'{resource_summary,wall_seconds}')::DOUBLE PRECISION,
      (source.result_doc#>>'{resource_summary,cpu_seconds}')::DOUBLE PRECISION;
    EXCEPTION WHEN invalid_datetime_format OR datetime_field_overflow OR numeric_value_out_of_range THEN RETURN NULL; END;
    reason:=coalesce(reason,'execution_score_missing'); CONTINUE;
   END IF;
   -- Prior model evidence retains an already materialized zero. It cannot
   -- turn a new unscored provider failure into an inferred zero.
   IF source.per_icp_score IS NOT NULL AND EXISTS(SELECT 1 FROM public.lab_arena_runs earlier
     WHERE earlier.assignment_id=source.assignment_id
      AND earlier.terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')) THEN
    CONTINUE; END IF;
   IF cause IN('provider_error','stage_closed') AND public.lab_arena__execution_privacy_route_unavailable_v1(source.run_id) THEN
    reason:='openrouter_zdr_route_unavailable';
   ELSE reason:=CASE WHEN reason='execution_score_missing' THEN cause ELSE coalesce(reason,cause) END; END IF;
  END IF;
 END LOOP;
 RETURN reason;
END;
$function$
$definition547$;
 EXECUTE updated;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR pg_get_functiondef(target) IS DISTINCT FROM updated THEN
  RAISE EXCEPTION 'Unmeasured execution finalization metadata or source changed'; END IF;
END;
$unmeasured547$;
NOTIFY pgrst,'reload schema';
COMMIT;
