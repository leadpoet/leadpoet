-- Owner-audited transport routing. Keep grading identity, work and leases exact.
-- This migration installs authority only. Activation is a separate owner call.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE arena683_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $pre683$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure(
  'public.lab_arena_set_judge_provider_routing_v1(text,jsonb,jsonb,text,text,boolean)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'47740494c2a9734fc36c11513a2f6d155d85b6bdadbb06e80d0ccc2993b29082'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u') THEN
  RAISE EXCEPTION 'judge routing683 existing function differs';
 END IF;
END;
$pre683$;

CREATE OR REPLACE FUNCTION public.lab_arena_set_judge_provider_routing_v1(
 p_round_id TEXT,p_expected_configuration JSONB,p_routing JSONB,
 p_request_id TEXT,p_actor_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog, public AS $routing683$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds;
 prior public.lab_arena_scoring_supersessions; registry JSONB; models JSONB;
 bindings JSONB; source_routes JSONB; model TEXT; plan JSONB; api TEXT; provider TEXT;
 primary_route JSONB; candidate JSONB; request_hash TEXT; proof JSONB; revision BIGINT;
 required_apis TEXT[]; response JSONB; mode_keys JSONB;
BEGIN
 IF p_round_id IS NULL OR p_round_id !~ '^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(-[a-z0-9]{1,16})?$'
  OR jsonb_typeof(p_expected_configuration) IS DISTINCT FROM 'object'
  OR octet_length(p_expected_configuration::TEXT)>16777216
  OR jsonb_typeof(p_routing) IS DISTINCT FROM 'object'
  OR octet_length(p_routing::TEXT)>16384
  OR p_routing='{}'::JSONB
  OR p_request_id IS NULL OR length(p_request_id) NOT BETWEEN 16 AND 128
  OR p_actor_ref IS NULL OR length(p_actor_ref) NOT BETWEEN 1 AND 200
  OR p_dry_run IS NULL THEN
  RAISE EXCEPTION 'judge provider routing request invalid' USING ERRCODE='22023';
 END IF;
 request_hash:='sha256:'||encode(extensions.digest(jsonb_build_object(
  'round_id',p_round_id,'expected_configuration',p_expected_configuration,
  'judge_provider_routing',p_routing,'actor_ref',p_actor_ref)::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions
  WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.request_hash IS DISTINCT FROM request_hash
   OR prior.recovery_doc->>'operation' IS DISTINCT FROM 'round_operational_update_v1'
   OR prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'judge_provider_routing_v1' THEN
   RAISE EXCEPTION 'judge provider routing request conflict' USING ERRCODE='22023';
  END IF;
  IF r.configuration_doc IS DISTINCT FROM prior.new_round_doc->'configuration_doc' THEN
   RAISE EXCEPTION 'judge provider routing replay superseded' USING ERRCODE='55000';
  END IF;
  RETURN jsonb_build_object('status','existing','request_id',p_request_id,
   'receipt_generation',prior.generation_floor,'judge_provider_routing',p_routing,
   'configuration_sha256','sha256:'||encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex'));
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR r.configuration_doc IS DISTINCT FROM p_expected_configuration
  OR r.status IN ('published','cancelled') OR r.published_at IS NOT NULL
  OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL
  OR r.arena_network_name IS DISTINCT FROM 'finney' OR r.arena_netuid IS DISTINCT FROM 71
  OR r.configuration_doc->>'mode' IS DISTINCT FROM 'live'
  OR ((r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ>clock_timestamp()) IS NOT TRUE
  OR ((r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ>clock_timestamp()) IS NOT TRUE THEN
  RAISE EXCEPTION 'judge provider routing round or expected configuration differs' USING ERRCODE='55000';
 END IF;
 bindings:=r.configuration_doc#>'{scorer_policy,env_bindings}';
 models:=r.configuration_doc#>'{scorer_policy,judge_models}';
 registry:=coalesce(r.configuration_doc->'judge_reasoning_route_registry',
                    r.configuration_doc->'reasoning_route_registry');
 IF bindings->>'ARENA_JUDGE_REASONING_CAPTURE_POLICY' IS DISTINCT FROM 'full_generated_thinking_v1'
  OR jsonb_typeof(models) IS DISTINCT FROM 'object'
  OR jsonb_typeof(registry) IS DISTINCT FROM 'object'
  OR registry->'schema_version' IS DISTINCT FROM '1'::JSONB
  OR jsonb_typeof(registry->'approved_routes') IS DISTINCT FROM 'array' THEN
  RAISE EXCEPTION 'judge provider routing frozen policy invalid' USING ERRCODE='22023';
 END IF;
 source_routes:=(bindings->>'ARENA_JUDGE_PROVIDER_ROUTES')::JSONB;
 IF jsonb_typeof(source_routes) IS DISTINCT FROM 'object' THEN
  RAISE EXCEPTION 'judge provider routing frozen routes invalid' USING ERRCODE='22023';
 END IF;
 FOR model,plan IN SELECT key,value FROM jsonb_each(p_routing) LOOP
  IF model !~ '^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]{0,119}$'
   OR NOT EXISTS(SELECT 1 FROM jsonb_each_text(models) m WHERE m.value=model)
   OR jsonb_typeof(plan) IS DISTINCT FROM 'array' THEN
   RAISE EXCEPTION 'judge provider routing model or plan invalid' USING ERRCODE='22023';
  END IF;
  IF jsonb_array_length(plan) NOT BETWEEN 1 AND 2
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(plan) p WHERE jsonb_typeof(p)<>'string')
   OR (SELECT count(DISTINCT p.value) FROM jsonb_array_elements_text(plan) p)
      <>jsonb_array_length(plan) THEN
   RAISE EXCEPTION 'judge provider routing candidates invalid' USING ERRCODE='22023';
  END IF;
  required_apis:=CASE WHEN EXISTS(SELECT 1 FROM jsonb_each_text(models) m
    WHERE m.value=model AND m.key IN ('company_fit_reverification','intent_three_stage_stage3'))
    THEN ARRAY['openrouter.chat','openrouter.responses'] ELSE ARRAY['openrouter.chat'] END;
  FOREACH api IN ARRAY required_apis LOOP
   SELECT count(*),jsonb_agg(a)->0 INTO revision,primary_route
   FROM jsonb_array_elements(registry->'approved_routes') a
   WHERE a->>'model_id'=model AND a->>'provider'=source_routes->>model AND a->>'api'=api;
   IF revision<>1 OR primary_route->>'status' IS DISTINCT FROM 'approved'
    OR primary_route->>'semantic_class' IS DISTINCT FROM 'full_generated_thinking'
    OR primary_route->>'parser_version' IS DISTINCT FROM 'arena_openrouter_reasoning_v1'
    OR primary_route->>'model_version' IS NULL
    OR primary_route->>'model_version' !~ '^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]{0,119}$'
    OR jsonb_typeof(primary_route->'mode') IS DISTINCT FROM 'object'
    OR jsonb_typeof(primary_route->'full_text_fields') IS DISTINCT FROM 'array'
    OR jsonb_typeof(primary_route->'context_tokens') IS DISTINCT FROM 'number'
    OR jsonb_typeof(primary_route->'max_output_tokens') IS DISTINCT FROM 'number'
    OR primary_route->>'context_tokens' !~ '^[1-9][0-9]*$'
    OR primary_route->>'max_output_tokens' !~ '^[1-9][0-9]*$' THEN
    RAISE EXCEPTION 'judge provider routing primary route invalid' USING ERRCODE='22023';
   END IF;
   SELECT jsonb_agg(k ORDER BY k) INTO mode_keys FROM jsonb_object_keys(primary_route->'mode') k;
   FOR provider IN SELECT value FROM jsonb_array_elements_text(plan) LOOP
    IF provider !~ '^[a-z0-9][a-z0-9._/-]{0,127}$' THEN
     RAISE EXCEPTION 'judge provider routing provider invalid' USING ERRCODE='22023';
    END IF;
    SELECT count(*),jsonb_agg(a)->0 INTO revision,candidate
    FROM jsonb_array_elements(registry->'approved_routes') a
    WHERE a->>'model_id'=model AND a->>'provider'=provider AND a->>'api'=api;
    IF revision<>1 OR candidate->>'status' IS DISTINCT FROM 'approved'
     OR candidate->>'semantic_class' IS DISTINCT FROM primary_route->>'semantic_class'
     OR candidate->>'parser_version' IS DISTINCT FROM primary_route->>'parser_version'
     OR candidate->>'model_version' IS DISTINCT FROM primary_route->>'model_version'
     OR candidate->'full_text_fields' IS DISTINCT FROM primary_route->'full_text_fields'
     OR jsonb_typeof(candidate->'mode') IS DISTINCT FROM 'object'
     OR jsonb_typeof(candidate->'context_tokens') IS DISTINCT FROM 'number'
     OR jsonb_typeof(candidate->'max_output_tokens') IS DISTINCT FROM 'number'
     OR candidate->>'context_tokens' !~ '^[1-9][0-9]*$'
     OR candidate->>'max_output_tokens' !~ '^[1-9][0-9]*$' THEN
     RAISE EXCEPTION 'judge provider routing candidate route differs' USING ERRCODE='22023';
    END IF;
    IF (SELECT jsonb_agg(k ORDER BY k) FROM jsonb_object_keys(candidate->'mode') k)
       IS DISTINCT FROM mode_keys
     OR (candidate->>'context_tokens')::NUMERIC<(primary_route->>'context_tokens')::NUMERIC
     OR (candidate->>'max_output_tokens')::NUMERIC<(primary_route->>'max_output_tokens')::NUMERIC THEN
     RAISE EXCEPTION 'judge provider routing candidate capacity or mode differs' USING ERRCODE='22023';
    END IF;
   END LOOP;
  END LOOP;
 END LOOP;
 n:=r;
 n.configuration_doc:=jsonb_set(r.configuration_doc,'{judge_provider_routing}',p_routing,TRUE);
 IF n.configuration_doc=r.configuration_doc THEN
  RAISE EXCEPTION 'judge provider routing unchanged without matching receipt' USING ERRCODE='22023';
 END IF;
 proof:=jsonb_build_object('operation','round_operational_update_v1',
  'recovery_kind','judge_provider_routing_v1','scoring_generation_floor',r.scoring_generation_floor,
  'old_judge_provider_routing',r.configuration_doc->'judge_provider_routing',
  'judge_provider_routing',p_routing,
  'scorer_policy_sha256','sha256:'||encode(extensions.digest((r.configuration_doc->'scorer_policy')::TEXT,'sha256'),'hex'));
 response:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'applied' END,
  'request_id',p_request_id,'judge_provider_routing',p_routing,
  'configuration_sha256','sha256:'||encode(extensions.digest(n.configuration_doc::TEXT,'sha256'),'hex'));
 IF p_dry_run THEN RETURN response; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision
  FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(p_round_id,revision,p_request_id,request_hash,p_actor_ref,txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}'::JSONB,proof);
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,
  updated_at=clock_timestamp() WHERE round_id=p_round_id;
 SELECT * INTO STRICT n FROM public.lab_arena_rounds WHERE round_id=p_round_id;
 IF n.configuration_doc IS DISTINCT FROM jsonb_set(r.configuration_doc,'{judge_provider_routing}',p_routing,TRUE)
  OR (to_jsonb(n)-ARRAY['configuration_doc','updated_at']) IS DISTINCT FROM
     (to_jsonb(r)-ARRAY['configuration_doc','updated_at']) THEN
  RAISE EXCEPTION 'judge provider routing changed protected state';
 END IF;
 RETURN response||jsonb_build_object('receipt_generation',revision);
END;
$routing683$;
ALTER FUNCTION public.lab_arena_set_judge_provider_routing_v1(TEXT,JSONB,JSONB,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_set_judge_provider_routing_v1(TEXT,JSONB,JSONB,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service,leadpoet_parity_reader;
DO $verify683$
DECLARE p RECORD;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=
  'public.lab_arena_set_judge_provider_routing_v1(text,jsonb,jsonb,text,text,boolean)'::regprocedure;
 IF encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'47740494c2a9734fc36c11513a2f6d155d85b6bdadbb06e80d0ccc2993b29082'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] THEN
  RAISE EXCEPTION 'judge routing683 function readback differs';
 END IF;
 IF NOT(SELECT had_create FROM pg_temp.arena683_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena683_acl) THEN
  RAISE EXCEPTION 'judge routing683 schema ACL differs';
 END IF;
END;
$verify683$;
COMMIT;
