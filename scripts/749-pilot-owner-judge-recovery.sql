-- Use the existing metered owner-review records for a completed unpaid
-- ordinary job. Its original quality, manifest, receipts and payment are immutable.
BEGIN;
SET LOCAL statement_timeout='60s';
SET LOCAL lock_timeout='5s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE pre749_acl ON COMMIT DROP AS SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $migration$
DECLARE t record; fn regprocedure; definition text; prior jsonb; later jsonb;
BEGIN
 FOR t IN SELECT * FROM (VALUES
  ('public.collection_reserve_launch_judge_v1(uuid,text,uuid,text,text,bigint)',
   $old$j.mode<>'live' OR j.status<>'ready' OR j.artifact_sha256 IS DISTINCT FROM p_source
    OR j.quality->>'training_status' IS DISTINCT FROM 'cannot_judge' OR EXISTS (SELECT FROM public.collection_reward_cutover)
    OR NOT EXISTS (SELECT FROM public.collection_reward_launch)$old$,
   $new$NOT ((j.mode='live' AND j.status='ready' AND NOT EXISTS(SELECT FROM public.collection_reward_cutover)
                AND EXISTS(SELECT FROM public.collection_reward_launch)) OR
             (j.mode='pilot' AND j.status='accepted' AND j.price_microusd=0 AND j.batch_id NOT LIKE 'xp-%'
              AND NOT EXISTS(SELECT FROM public.collection_launch_reviews x WHERE x.job_id=p_job
                                    AND x.artifact_sha256 IS NOT NULL)))
    OR j.artifact_sha256 IS DISTINCT FROM p_source OR j.quality->>'training_status' IS DISTINCT FROM 'cannot_judge'$new$),
  ('public.collection_reserve_launch_judge_v1(uuid,text,uuid,text,text,bigint)',
   $old$IF unknown_count>0 THEN$old$,
   $new$IF j.mode='pilot' AND unknown_count>0 THEN
  -- No miner call is repeated. Retain *all* unresolved owner reservations in
  -- the original total cap before allowing another owner-only judge request.
  IF EXISTS (SELECT FROM public.collection_calls c LEFT JOIN public.collection_call_receipts rc USING(job_id,request_id)
             WHERE c.job_id=p_job AND c.key_id IS NOT NULL AND rc.cost_microusd IS NULL) THEN
   RETURN jsonb_build_object('status','budget_exhausted'); END IF;
  SELECT sum(coalesce(rc.cost_microusd,known.cost_microusd,c.reserved_microusd)) INTO used
  FROM public.collection_calls c LEFT JOIN public.collection_call_receipts rc USING(job_id,request_id)
    LEFT JOIN public.collection_launch_judge_costs known USING(job_id,request_id) WHERE c.job_id=p_job;
  unknown_count:=0;
 END IF;
 IF unknown_count>0 THEN$new$),
  ('public.collection_finish_launch_review_v1(uuid,text,uuid,text,jsonb,jsonb)',
   $old$j.mode<>'live' OR j.status<>'ready'$old$,
   $new$NOT ((j.mode='live' AND j.status='ready') OR
             (j.mode='pilot' AND j.status='accepted' AND j.price_microusd=0 AND j.batch_id NOT LIKE 'xp-%'))$new$),
  ('public.collection_finish_launch_review_v1(uuid,text,uuid,text,jsonb,jsonb)',
   $old$UPDATE public.collection_jobs SET artifact_sha256=p_artifact,quality=p_quality WHERE job_id=p_job;$old$,
   $new$IF j.mode='live' THEN
  UPDATE public.collection_jobs SET artifact_sha256=p_artifact,quality=p_quality WHERE job_id=p_job;
 END IF;$new$)
 ) changes(signature,original,replacement)
 LOOP
  fn=t.signature::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,prior
  FROM pg_proc p WHERE p.oid=fn AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
  IF cardinality(string_to_array(definition,t.original))<>2 THEN RAISE EXCEPTION '749 unexpected review predecessor: %',t.signature; END IF;
  EXECUTE replace(definition,t.original,t.replacement);
  SELECT to_jsonb(p)-'prosrc' INTO later FROM pg_proc p WHERE p.oid=fn;
  IF later IS DISTINCT FROM prior THEN RAISE EXCEPTION '749 changed function metadata'; END IF;
 END LOOP;
 IF NOT(SELECT had_create FROM pg_temp.pre749_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre749_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $migration$;
CREATE OR REPLACE FUNCTION public.collection_batch_jobs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(job) || CASE WHEN review.artifact_sha256 IS NOT NULL
             THEN jsonb_build_object('pilot_review_sha256',review.artifact_sha256) ELSE '{}'::jsonb END
             ORDER BY job.slot,job.attempt),'[]')
 FROM (SELECT j.job_id,j.batch_id,j.mode,j.slot,j.attempt,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,
              j.artifact_sha256,j.status,j.budget_microusd,j.price_microusd
       FROM public.collection_jobs j WHERE j.batch_id=p_batch AND j.network='finney' AND j.netuid=71) job
 LEFT JOIN public.collection_launch_reviews review ON job.mode='pilot' AND job.job_id=review.job_id
    AND review.source_sha256=job.artifact_sha256 AND review.artifact_sha256 IS NOT NULL
$$;
NOTIFY pgrst,'reload schema';
COMMIT;
