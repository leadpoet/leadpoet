-- Give each signing miner a receipt total for their own assignments.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';

CREATE OR REPLACE FUNCTION public.collection_statement_v1(p_hotkey text,p_mode text)
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH enrollments AS (
   SELECT r.registration_id,r.created_at,r.revoked_at,r.consent#>>'{body,task_scope}' AS task_scope,
          (r.consent#>>'{body,daily_cap_microusd}')::bigint AS daily_cap_microusd
   FROM public.collection_registrations r WHERE r.miner_hotkey=p_hotkey AND r.mode=p_mode
   ORDER BY r.created_at DESC,r.registration_id LIMIT 50
 ), jobs AS (
   SELECT j.job_id,j.slot,j.model,j.status,j.scenario_id,j.created_at,j.finished_at,
          j.budget_microusd,j.price_microusd,j.quality
   FROM public.collection_jobs j WHERE j.miner_hotkey=p_hotkey AND j.mode=p_mode
   ORDER BY j.created_at DESC,j.job_id LIMIT 100
 ), billed_jobs AS (
   SELECT j.*,b.* FROM jobs j CROSS JOIN LATERAL (
     SELECT coalesce(sum(r.cost_microusd),0) AS confirmed_miner_cost_microusd,
            coalesce(sum(c.reserved_microusd) FILTER (WHERE r.cost_microusd IS NULL),0)
              AS pending_miner_reservations_microusd,
            count(*) FILTER (WHERE r.cost_microusd IS NULL) AS unresolved_miner_calls
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=j.job_id AND c.key_id IS NOT NULL
   ) b
 )
 SELECT jsonb_build_object(
   'enrollments',coalesce((SELECT jsonb_agg(to_jsonb(e) ORDER BY created_at DESC,registration_id) FROM enrollments e),'[]'::jsonb),
   'jobs',coalesce((SELECT jsonb_agg(to_jsonb(j) ORDER BY created_at DESC,job_id) FROM billed_jobs j),'[]'::jsonb)
 );
$$;
COMMIT;
