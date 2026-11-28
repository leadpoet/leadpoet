-- COMMIT is the cutover. Existing leases, sealed experiment identifiers, provider bills
-- and signed epoch quotes keep their original terms. Only new standing assignments change.
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
SELECT pg_advisory_xact_lock(hashtextextended('collection_pool_registration',0));

CREATE TEMP TABLE standing_rate_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- NULL represents pre-cutover and non-pool jobs. No historical row needs an update.
ALTER TABLE public.collection_jobs ADD COLUMN reward_rate_bps smallint
 CHECK (reward_rate_bps IS NULL OR
        (mode='live' AND batch_id ~ '^pool-[a-f0-9]{64}$' AND reward_rate_bps=15000));

CREATE FUNCTION public.collection_pool_reward_rate_bps_v1() RETURNS integer
 LANGUAGE sql STABLE SET search_path=pg_catalog AS $$ SELECT 15000 $$;
ALTER FUNCTION public.collection_pool_reward_rate_bps_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_pool_reward_rate_bps_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_pool_reward_rate_bps_v1() TO lab_arena_service;

-- Retain the original function bodies and ACLs except for the exact reviewed expressions.
CREATE FUNCTION pg_temp.rate_patch(fn regprocedure,old_part text,new_part text,occurrences integer DEFAULT 1)
 RETURNS void LANGUAGE plpgsql AS $$
DECLARE definition text; prior jsonb;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,prior FROM pg_proc p
 WHERE p.oid=fn AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(definition,old_part))<>occurrences+1 THEN
   RAISE EXCEPTION '772 unexpected predecessor: %',fn;
 END IF;
 EXECUTE replace(definition,old_part,new_part);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=fn) IS DISTINCT FROM prior THEN
   RAISE EXCEPTION '772 changed function authority: %',fn;
 END IF;
END $$;

DO $migration$
DECLARE
 reg regprocedure:='public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)'::regprocedure;
 claim regprocedure:='public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)'::regprocedure;
 guard regprocedure:='public.collection_approved_job_price_v1()'::regprocedure;
 bill regprocedure:='public.collection_pool_target_v1(uuid)'::regprocedure;
 jobs regprocedure:='public.collection_batch_jobs_v1(text)'::regprocedure;
 statement regprocedure:='public.collection_statement_v1(text,text)'::regprocedure;
 previous_terms text:='sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1';
 current_terms text:='sha256:29c76e6f5d945c6e68821a331423395a3445d37c2f9d8943dab79433e8b608bc';
 grants text:=format('r.consent#>>''{body,terms_sha256}'' IN (%L,%L)',previous_terms,current_terms);
BEGIN
 -- Existing registration IDs replay upstream of this check. New grants must sign current terms.
 PERFORM pg_temp.rate_patch(reg,previous_terms,current_terms);
 -- The sealed pack hash still identifies the same permitted work. Either original consent
 -- may fund it; the original scope, owner, revocation, capacity and priority checks still apply.
 PERFORM pg_temp.rate_patch(claim,'r.consent#>>''{body,terms_sha256}''=p_terms_hash AND capacity.available',
                           grants||' AND capacity.available');
 PERFORM pg_temp.rate_patch(guard,'r.consent#>>''{body,terms_sha256}''=b.experiment_terms_hash',grants);
 PERFORM pg_temp.rate_patch(guard,
   $old$THEN RAISE EXCEPTION 'invalid standing-funded job'; END IF;
   RETURN NEW;$old$,
   $new$THEN RAISE EXCEPTION 'invalid standing-funded job'; END IF;
   NEW.reward_rate_bps:=public.collection_pool_reward_rate_bps_v1();
   RETURN NEW;$new$);

 -- Migration 758's UPDATE guard already treats every other assignment field as immutable.
 -- Check the selected job's rate only after the original settled, capped, unique bill is proven.
 PERFORM pg_temp.rate_patch(bill,$old$((bill->>'cost_microusd')::bigint*5+3)/4$old$,
   $new$((bill->>'cost_microusd')::bigint * coalesce(
      (SELECT j.reward_rate_bps FROM public.collection_jobs j WHERE j.job_id=p_job),12500) + 9999)/10000$new$);

 -- Include a rate only when one was recorded. Old RPC documents and export hashes stay byte-compatible.
 PERFORM pg_temp.rate_patch(claim,$old$(to_jsonb(job)-'lease_token_hash')$old$,
   $new$((CASE WHEN job.reward_rate_bps IS NULL THEN to_jsonb(job)-'reward_rate_bps' ELSE to_jsonb(job) END)-'lease_token_hash')$new$,2);
 PERFORM pg_temp.rate_patch(jobs,'j.artifact_sha256,j.status,j.budget_microusd,j.price_microusd',
                                'j.artifact_sha256,j.status,j.budget_microusd,j.price_microusd,j.reward_rate_bps');
 PERFORM pg_temp.rate_patch(jobs,'to_jsonb(job) || CASE WHEN review.',
   $new$(CASE WHEN job.reward_rate_bps IS NULL THEN to_jsonb(job)-'reward_rate_bps' ELSE to_jsonb(job) END) || CASE WHEN review.$new$);
 PERFORM pg_temp.rate_patch(statement,'j.budget_microusd,j.price_microusd,j.quality',
                                     'j.budget_microusd,j.price_microusd,j.quality,j.reward_rate_bps');
 PERFORM pg_temp.rate_patch(statement,'jsonb_agg(to_jsonb(j) ORDER BY created_at DESC,job_id)',
   $new$jsonb_agg(CASE WHEN j.reward_rate_bps IS NULL THEN to_jsonb(j)-'reward_rate_bps' ELSE to_jsonb(j) END ORDER BY created_at DESC,job_id)$new$);
END $migration$;

DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.standing_rate_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.standing_rate_acl) THEN
   RAISE EXCEPTION '772 changed schema privileges';
 END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
