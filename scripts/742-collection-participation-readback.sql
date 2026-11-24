-- Aggregate private participation and billing. Never return identities, consent or key material.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE watch_previous_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE FUNCTION public.collection_participation_v1() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH registrations AS (
  SELECT r.mode,CASE r.consent->'body'->>'task_scope'
    WHEN 'synthetic_corrections' THEN 'correction' WHEN 'synthetic_workflows' THEN 'comparison'
    ELSE 'other' END arm,count(*) registrations,
    count(*) FILTER(WHERE r.revoked_at IS NULL AND EXISTS (
      SELECT FROM public.collection_keys k WHERE k.registration_id=r.registration_id
      AND k.provider='openrouter' AND k.revoked_at IS NULL)) active
  FROM public.collection_registrations r GROUP BY 1,2
 ), work AS (
  SELECT j.mode,CASE WHEN j.batch_id LIKE 'xp-%' THEN 'correction' ELSE 'comparison' END arm,
    count(*) assigned,count(*) FILTER(WHERE j.artifact_sha256 IS NOT NULL) artifacts,
    count(*) FILTER(WHERE j.status='failed') failed,count(*) FILTER(WHERE j.status='rejected') rejected,
    count(*) FILTER(WHERE j.status='running' AND j.lease_expires_at<=statement_timestamp()) expired,
    count(*) FILTER(WHERE j.status='accepted') accepted,count(*) FILTER(WHERE j.status='ready') ready
  FROM public.collection_jobs j WHERE j.netuid=71 AND j.network='finney' GROUP BY 1,2
 ), review_candidates AS (
  -- This is an operator handoff. Only the private source bundle can distinguish
  -- genuine judgments and controls and verify the signed human sample.
  SELECT b.batch_id FROM public.collection_batches b JOIN public.collection_jobs j USING(batch_id)
  WHERE b.mode='pilot' AND b.batch_id NOT LIKE 'xp-%' AND j.network='finney' AND j.netuid=71
  GROUP BY b.batch_id,b.scenario_ids,b.models
  HAVING bool_and(j.status IN ('accepted','rejected','failed'))
    AND count(DISTINCT j.slot)=jsonb_array_length(b.scenario_ids)*jsonb_array_length(b.models)
    AND count(DISTINCT j.slot) FILTER (WHERE j.artifact_sha256 IS NOT NULL)>=100
 ), billing AS (
  SELECT j.mode,CASE WHEN j.batch_id LIKE 'xp-%' THEN 'correction' ELSE 'comparison' END arm,
    count(DISTINCT j.job_id) FILTER(WHERE r.http_status=200 AND r.cost_microusd>0) funded_runs,
    coalesce(sum(r.cost_microusd),0) miner_microusd,
    count(*) FILTER(WHERE r.cost_microusd IS NULL AND c.created_at<statement_timestamp()-interval '2 hours') unresolved
  FROM public.collection_jobs j JOIN public.collection_calls c USING(job_id)
  LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
  WHERE j.netuid=71 AND j.network='finney' AND c.key_id IS NOT NULL GROUP BY 1,2
 )
 SELECT jsonb_build_object('enrollments',coalesce((SELECT jsonb_agg(to_jsonb(r) ORDER BY mode,arm) FROM registrations r),'[]'),
  'work',coalesce((SELECT jsonb_agg(to_jsonb(work) ORDER BY mode,arm) FROM work),'[]'),
  'billing',coalesce((SELECT jsonb_agg(to_jsonb(billing) ORDER BY mode,arm) FROM billing),'[]'),
  'completed_review_candidate_packs',(SELECT count(*) FROM review_candidates),
  'correction_quotes',(SELECT count(*) FROM public.collection_correction_tariffs),
  'staged',EXISTS(SELECT FROM public.collection_reward_launch),
  'first_epoch',(SELECT first_epoch FROM public.collection_reward_cutover))
$$;
ALTER FUNCTION public.collection_participation_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_participation_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_participation_v1() TO lab_arena_service;
DO $$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.watch_previous_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.watch_previous_acl)
 THEN RAISE EXCEPTION 'unexpected schema privileges'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
