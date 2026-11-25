-- Find two original billable subjects at a time, without exposing a key or changing a receipt.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE billing_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE FUNCTION public.collection_pending_billing_v1(p_after_job uuid,p_after_request text)
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(candidate) ORDER BY candidate.job,candidate.request),'[]')
 FROM (
  SELECT c.job_id job,c.request_id request,c.provider,j.artifact_sha256 source,r.provider_id generation,
    control.provider_id control_generation,control.cost_microusd control_cost
  FROM public.collection_calls c JOIN public.collection_jobs j USING(job_id)
  JOIN public.collection_call_receipts r USING(job_id,request_id)
  LEFT JOIN public.collection_launch_judge_costs settled USING(job_id,request_id)
  LEFT JOIN LATERAL (
   SELECT pr.provider_id,pr.cost_microusd FROM public.collection_calls pc
    JOIN public.collection_call_receipts pr USING(job_id,request_id)
    WHERE c.provider='judge' AND pc.job_id=c.job_id AND pc.provider='judge' AND pc.key_id IS NULL
     AND pr.http_status=200 AND pr.cost_microusd>0 AND pr.provider_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    ORDER BY pc.created_at,pc.request_id LIMIT 1
  ) control ON true
  WHERE j.network='finney' AND j.netuid=71 AND j.status IN ('accepted','rejected','failed','ready')
    AND r.cost_microusd IS NULL AND r.provider_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    AND (p_after_job IS NULL OR (c.job_id,c.request_id)>(p_after_job,p_after_request))
    AND ((c.provider='openrouter' AND c.key_id=j.openrouter_key_id AND EXISTS (
           SELECT FROM public.collection_keys k JOIN public.collection_registrations en USING(registration_id)
           WHERE k.key_id=c.key_id AND k.provider='openrouter' AND k.revoked_at IS NULL
             AND en.registration_id=j.registration_id AND en.miner_hotkey=j.miner_hotkey AND en.revoked_at IS NULL))
         OR (c.provider='judge' AND c.key_id IS NULL AND control.provider_id IS NOT NULL
             AND j.artifact_sha256 IS NOT NULL AND j.quality->>'training_status'='cannot_judge'
             AND ((j.mode='pilot' AND j.status='accepted' AND j.price_microusd=0) OR
                  (j.mode='live' AND j.status='ready')) AND settled.job_id IS NULL))
  ORDER BY c.job_id,c.request_id LIMIT 2
 ) candidate
$$;
ALTER FUNCTION public.collection_pending_billing_v1(uuid,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_pending_billing_v1(uuid,text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_pending_billing_v1(uuid,text) TO lab_arena_service;

-- Reuse the append-only owner ledger from 729. An already finished, unpaid pilot
-- has the same immutable owner request and response as a staged first live job.
DO $$
DECLARE fn regprocedure:='public.collection_reconcile_launch_judge_v1(uuid,text,text,text,bigint,text)'::regprocedure;
 definition text; prior jsonb; old_guard text:='j.mode<>''live'' OR j.status<>''ready''';
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,prior
 FROM pg_proc p WHERE p.oid=fn AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(definition,old_guard))<>2 THEN RAISE EXCEPTION 'unexpected owner receipt predecessor'; END IF;
 EXECUTE replace(definition, old_guard,
   'NOT ((j.mode=''live'' AND j.status=''ready'') OR (j.mode=''pilot'' AND j.status=''accepted'' AND j.price_microusd=0))');
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=fn) IS DISTINCT FROM prior THEN
   RAISE EXCEPTION 'owner ledger signature/privileges changed'; END IF;
 IF NOT (SELECT had_create FROM pg_temp.billing_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.billing_acl) THEN
   RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
