-- Public paid participation. First enrollment is stable across revocation and re-enrollment.
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE prior_contributor_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE FUNCTION public.collection_public_miners_v1(p_after text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH roster AS MATERIALIZED (
   SELECT miner_hotkey, min(created_at) joined_at
   FROM public.collection_registrations WHERE mode='live' GROUP BY miner_hotkey
 ), page AS MATERIALIZED (
   SELECT * FROM roster WHERE p_after IS NULL OR (joined_at,miner_hotkey) <
       (SELECT joined_at,miner_hotkey FROM roster WHERE miner_hotkey=p_after)
   ORDER BY joined_at DESC,miner_hotkey DESC LIMIT 26
 ), jobs AS MATERIALIZED (
   SELECT j.miner_hotkey,j.status,j.price_microusd,j.accepted_epoch,
     (SELECT max(c.created_at) FROM public.collection_calls c
       JOIN public.collection_call_receipts r USING(job_id,request_id)
       WHERE c.job_id=j.job_id AND r.cost_microusd>0 AND
         ((c.provider='openrouter' AND c.key_id=j.openrouter_key_id) OR
          (c.provider='deepline' AND c.key_id=j.deepline_key_id))) funded_at
   FROM public.collection_jobs j JOIN page USING(miner_hotkey)
   WHERE j.mode='live' AND j.netuid=71 AND j.network='finney'
 ), work AS (
   SELECT miner_hotkey,count(funded_at) funded_runs,max(funded_at) last_funded_at
   FROM jobs GROUP BY miner_hotkey
 ), quotes AS (
   SELECT miner_hotkey,accepted_epoch+1 quote_epoch,sum(price_microusd) fee
   FROM jobs WHERE status='accepted' GROUP BY miner_hotkey,accepted_epoch
 ), earned AS (
   SELECT miner_hotkey,jsonb_object_agg(quote_epoch::text,fee) epochs FROM quotes GROUP BY miner_hotkey
 )
 SELECT coalesce(jsonb_agg(jsonb_build_object(
   'hotkey',p.miner_hotkey,'joined_at',p.joined_at,'last_funded_at',w.last_funded_at,
   'funded_runs',coalesce(w.funded_runs,0),'epochs',coalesce(e.epochs,'{}'::jsonb)
 ) ORDER BY p.joined_at DESC,p.miner_hotkey DESC),'[]'::jsonb)
 FROM page p LEFT JOIN work w USING(miner_hotkey) LEFT JOIN earned e USING(miner_hotkey)
$$;
ALTER FUNCTION public.collection_public_miners_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_public_miners_v1(text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_public_miners_v1(text) TO lab_arena_service;
DO $$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.prior_contributor_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.prior_contributor_acl) THEN
   RAISE EXCEPTION 'unexpected public schema privilege change'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
