-- Expose only ordinary, completed pilot jobs with an original owner-side outage.
-- The existing owner review ledger and reservation function authorize the retry.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE prior_review_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE FUNCTION public.collection_interrupted_pilot_judges_v1(p_batch text)
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(jsonb_build_object('job',j.job_id,'request',c.request_id)
                          ORDER BY j.slot,j.job_id,c.request_id),'[]')
 FROM public.collection_jobs j JOIN public.collection_calls c USING(job_id)
 JOIN public.collection_call_receipts r USING(job_id,request_id)
 WHERE j.batch_id=p_batch AND j.network='finney' AND j.netuid=71 AND j.mode='pilot'
   AND j.status='accepted' AND j.price_microusd=0 AND j.artifact_sha256 IS NOT NULL
   AND j.quality->>'training_status'='cannot_judge' AND j.batch_id NOT LIKE 'xp-%'
   AND c.provider='judge' AND c.key_id IS NULL AND c.request_id ~ '^collection-[0-9a-f]{64}-[12]$'
   AND r.http_status IN (408,429,500,502,503,504)
   AND NOT EXISTS (SELECT FROM public.collection_launch_reviews v WHERE v.job_id=j.job_id)
   AND NOT EXISTS (SELECT FROM public.collection_calls g LEFT JOIN public.collection_call_receipts b USING(job_id,request_id)
                   WHERE g.job_id=j.job_id AND g.key_id IS NOT NULL AND b.cost_microusd IS NULL)
   AND EXISTS (SELECT FROM public.collection_calls g JOIN public.collection_call_receipts b USING(job_id,request_id)
               WHERE g.job_id=j.job_id AND g.provider='openrouter' AND g.key_id=j.openrouter_key_id
                 AND b.http_status=200 AND b.cost_microusd>0)
$$;
ALTER FUNCTION public.collection_interrupted_pilot_judges_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_interrupted_pilot_judges_v1(text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_interrupted_pilot_judges_v1(text) TO lab_arena_service;
DO $$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.prior_review_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.prior_review_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
