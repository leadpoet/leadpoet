-- Keep a single allocator. A preview runs the normal eligibility/maintenance checks,
-- then returns only the eligible model before creating any lease or charging a slot.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE priced_acl ON COMMIT DROP AS SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $migration$
DECLARE oldfn regprocedure:='public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)'::regprocedure;
 body text; args text; existing text:=$old$  RETURN (to_jsonb(job)-'lease_token_hash')$old$;
 insert_job text:=$old$ INSERT INTO public.collection_jobs(job_id,batch_id,mode,registration_id$old$;
BEGIN
 SELECT p.prosrc,pg_get_function_arguments(p.oid) INTO STRICT body,args FROM pg_proc p WHERE p.oid=oldfn
   AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(body,existing))<>2 OR cardinality(string_to_array(body,insert_job))<>2
   OR position('least(daily_limit,r.max_slots_per_utc_day)' in body)=0
 THEN RAISE EXCEPTION '754 unexpected allocator predecessor'; END IF;
 body:=replace(body,existing,$new$  IF p_expected_model IS NULL THEN RETURN jsonb_build_object('model',job.model); END IF;
  IF p_expected_model NOT IN ('*',job.model) THEN RETURN NULL; END IF;
$new$||existing);
 body:=replace(body,insert_job,$new$ IF p_expected_model IS NULL THEN RETURN jsonb_build_object('model',generator); END IF;
 IF p_expected_model NOT IN ('*',generator) THEN RETURN NULL; END IF;
$new$||insert_job);
 EXECUTE format('CREATE FUNCTION public.collection_priced_claim_v1(%s,p_expected_model text) RETURNS jsonb
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS %L',args,body);
END $migration$;
ALTER FUNCTION public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text) TO lab_arena_service;

-- The historical API supplies externally validated prices and retains its original signature and ACL.
CREATE OR REPLACE FUNCTION public.collection_claim_v1(p_job uuid,p_token_hash text,p_batch text,p_mode text,p_commit text,
 p_scenarios jsonb,p_models jsonb,p_prices jsonb,p_budgets jsonb,p_terms_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 RETURN public.collection_priced_claim_v1(p_job,p_token_hash,p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets,p_terms_hash,'*');
END$$;
DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.priced_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.priced_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
