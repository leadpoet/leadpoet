-- An authenticated, model-only HTTP 429 can retry only inside the *other*
-- portion of the original slot. The unresolved original reservation remains.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE pre747_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_throttle_holds (
 job_id uuid NOT NULL,
 request_id text NOT NULL,
 original_request text NOT NULL CHECK(original_request ~ '^[0-9a-f]{64}$'),
 original_response text NOT NULL CHECK(original_response ~ '^[0-9a-f]{64}$'),
 proof_sha256 text NOT NULL CHECK(proof_sha256 ~ '^sha256:[0-9a-f]{64}$'),
 proof_kind text NOT NULL CHECK(proof_kind IN ('verified_typed_error','operator_verified_preacceptance')),
 observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(job_id,request_id),
 FOREIGN KEY(job_id,request_id) REFERENCES public.collection_call_receipts(job_id,request_id)
);
ALTER TABLE public.collection_throttle_holds OWNER TO lab_arena_owner;
ALTER TABLE public.collection_throttle_holds ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_throttle_holds FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.collection_throttle_hold_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'collection throttle evidence is immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF NOT EXISTS (
   SELECT FROM public.collection_jobs j JOIN public.collection_calls c ON c.job_id=j.job_id AND c.key_id=j.openrouter_key_id
   JOIN public.collection_call_receipts r ON r.job_id=c.job_id AND r.request_id=c.request_id
   WHERE j.job_id=NEW.job_id AND j.network='finney' AND j.netuid=71 AND j.mode='pilot' AND j.attempt=1
     AND j.status IN ('running','failed') AND j.deepline_key_id IS NULL AND c.provider='openrouter'
     AND c.request_id=NEW.request_id AND c.request_hash=NEW.original_request AND r.response_hash=NEW.original_response
     AND r.http_status=429 AND r.cost_microusd IS NULL AND c.reserved_microusd>0
     AND NOT EXISTS (SELECT FROM public.collection_jobs later WHERE later.batch_id=j.batch_id AND later.slot=j.slot AND later.attempt>1)
 ) THEN RAISE EXCEPTION 'throttle proof differs from the original eligible pilot request'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER collection_throttle_hold_guard BEFORE INSERT OR UPDATE OR DELETE ON public.collection_throttle_holds
 FOR EACH ROW EXECUTE FUNCTION public.collection_throttle_hold_guard_v1();

CREATE FUNCTION public.collection_record_throttle_hold_v1(p_job uuid,p_lease text,p_request text,
 p_original_request text,p_original_response text,p_proof text) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF NOT EXISTS(SELECT FROM public.collection_jobs j WHERE j.job_id=p_job AND j.lease_token_hash=p_lease
           AND j.status='running' AND j.lease_expires_at>statement_timestamp()) THEN
  RAISE EXCEPTION 'inactive collection throttle lease'; END IF;
 INSERT INTO public.collection_throttle_holds(job_id,request_id,original_request,original_response,proof_sha256,proof_kind)
 VALUES(p_job,p_request,p_original_request,p_original_response,p_proof,'verified_typed_error') ON CONFLICT DO NOTHING;
 IF NOT EXISTS(SELECT FROM public.collection_throttle_holds WHERE job_id=p_job AND request_id=p_request
             AND original_request=p_original_request AND original_response=p_original_response AND proof_sha256=p_proof)
 THEN RAISE EXCEPTION 'conflicting throttle proof'; END IF;
 RETURN jsonb_build_object('status','hold_preserved');
END $$;
ALTER FUNCTION public.collection_throttle_hold_guard_v1() OWNER TO lab_arena_owner;
ALTER FUNCTION public.collection_record_throttle_hold_v1(uuid,text,text,text,text,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_throttle_hold_guard_v1(),public.collection_record_throttle_hold_v1(uuid,text,text,text,text,text)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_record_throttle_hold_v1(uuid,text,text,text,text,text) TO lab_arena_service;

DO $migration$
DECLARE target regprocedure := 'public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)'::regprocedure;
 old_sql text := $old$SELECT coalesce(sum(r.cost_microusd),0) committed, count(*)=count(r.cost_microusd) settled
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=j.job_id$old$;
 new_sql text := $new$SELECT coalesce(sum(coalesce(r.cost_microusd, c.reserved_microusd)),0) committed,
            count(*)=count(CASE WHEN r.cost_microusd IS NOT NULL OR h.job_id IS NOT NULL THEN 1 END) settled
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     LEFT JOIN public.collection_throttle_holds h USING(job_id,request_id)
     WHERE c.job_id=j.job_id$new$;
 definition text; before jsonb; after jsonb;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,before
 FROM pg_proc p WHERE p.oid=target AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(definition,old_sql))<>2 THEN RAISE EXCEPTION '747 unexpected retry budget'; END IF;
 EXECUTE replace(definition,old_sql,new_sql);
 SELECT to_jsonb(p)-'prosrc' INTO after FROM pg_proc p WHERE p.oid=target;
 IF before IS DISTINCT FROM after THEN RAISE EXCEPTION '747 changed collection claim metadata'; END IF;
 IF NOT(SELECT had_create FROM pg_temp.pre747_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre747_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $migration$;
NOTIFY pgrst,'reload schema';
COMMIT;
