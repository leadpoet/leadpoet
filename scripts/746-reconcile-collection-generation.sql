-- Resolve only an originally unknown OpenRouter charge from the original account.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE pre746_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_generation_settlements (
 job_id uuid NOT NULL,
 request_id text NOT NULL,
 generation_id text NOT NULL UNIQUE,
 original_response_hash text NOT NULL CHECK(original_response_hash ~ '^[a-f0-9]{64}$'),
 lookup_hash text NOT NULL CHECK(lookup_hash ~ '^[a-f0-9]{64}$'),
 cost_microusd bigint NOT NULL CHECK(cost_microusd>=0),
 observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(job_id,request_id),
 FOREIGN KEY(job_id,request_id) REFERENCES public.collection_call_receipts(job_id,request_id)
);
ALTER TABLE public.collection_generation_settlements OWNER TO lab_arena_owner;
ALTER TABLE public.collection_generation_settlements ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_generation_settlements FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_generation_settlements_immutable BEFORE UPDATE OR DELETE ON public.collection_generation_settlements
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE FUNCTION public.collection_billing_subject_v1(p_job uuid,p_request text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('generation',r.provider_id,'response_hash',r.response_hash,'cost_microusd',r.cost_microusd,
    'credential',CASE WHEN r.cost_microusd IS NULL AND k.revoked_at IS NULL AND enrollment.revoked_at IS NULL
       THEN jsonb_build_object('key_id',k.key_id,'provider',k.provider,'miner_hotkey',k.miner_hotkey,
                         'submission_id',k.submission_id,'ciphertext_b64',k.ciphertext_b64) END)
 FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id)
 JOIN public.collection_jobs j ON j.job_id=c.job_id AND j.network='finney' AND j.netuid=71
 JOIN public.collection_keys k ON k.key_id=c.key_id AND k.key_id=j.openrouter_key_id
 JOIN public.collection_registrations enrollment ON enrollment.registration_id=k.registration_id
      AND enrollment.registration_id=j.registration_id AND enrollment.miner_hotkey=j.miner_hotkey
 WHERE c.job_id=p_job AND c.request_id=p_request AND c.provider='openrouter' AND k.provider=c.provider
   AND j.status IN ('failed','rejected','accepted','ready') AND r.provider_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
$$;

CREATE FUNCTION public.collection_verified_late_cost_v1() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
 -- All original request/response evidence is write-once. The only permitted
 -- change is carrying the first proved cost into the existing accounting views.
 IF TG_OP='UPDATE' AND OLD.cost_microusd IS NULL AND NEW.cost_microusd IS NOT NULL
    AND (to_jsonb(NEW)-'cost_microusd')=(to_jsonb(OLD)-'cost_microusd')
    AND EXISTS(SELECT FROM public.collection_generation_settlements s WHERE s.job_id=OLD.job_id
       AND s.request_id=OLD.request_id AND s.original_response_hash=OLD.response_hash
       AND s.generation_id=OLD.provider_id AND s.cost_microusd=NEW.cost_microusd) THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'collection receipt is immutable';
END $$;
DROP TRIGGER collection_receipts_immutable ON public.collection_call_receipts;
CREATE TRIGGER collection_receipts_immutable BEFORE UPDATE OR DELETE ON public.collection_call_receipts
 FOR EACH ROW EXECUTE FUNCTION public.collection_verified_late_cost_v1();

CREATE FUNCTION public.collection_settle_generation_v1(p_job uuid,p_request text,p_generation text,
 p_response_hash text,p_lookup_hash text,p_cost bigint) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; r public.collection_call_receipts; key public.collection_keys; saved public.collection_generation_settlements;
BEGIN
 IF p_cost IS NULL OR p_cost<0 OR p_lookup_hash IS NULL OR p_lookup_hash !~ '^[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'invalid generation billing proof'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job AND status IN ('failed','rejected','accepted','ready') FOR UPDATE;
 SELECT * INTO STRICT key FROM public.collection_keys WHERE key_id=j.openrouter_key_id AND registration_id=j.registration_id AND provider='openrouter' FOR UPDATE;
 SELECT rc.* INTO STRICT r FROM public.collection_call_receipts rc JOIN public.collection_calls c USING(job_id,request_id)
 WHERE c.job_id=p_job AND c.request_id=p_request AND c.key_id=key.key_id AND c.provider='openrouter' FOR UPDATE OF rc;
 IF r.provider_id IS NULL OR r.provider_id IS DISTINCT FROM p_generation OR r.response_hash IS DISTINCT FROM p_response_hash
    OR (SELECT count(*) FROM public.collection_call_receipts WHERE provider_id=p_generation)<>1 THEN
  RAISE EXCEPTION 'different or ambiguous original generation'; END IF;
 SELECT * INTO saved FROM public.collection_generation_settlements WHERE job_id=p_job AND request_id=p_request;
 IF FOUND THEN
  IF saved.cost_microusd IS DISTINCT FROM p_cost OR r.cost_microusd IS DISTINCT FROM p_cost THEN
   RAISE EXCEPTION 'conflicting generation cost'; END IF;
  RETURN jsonb_build_object('status','settled','cost_microusd',p_cost);
 END IF;
 IF r.cost_microusd IS NOT NULL THEN RAISE EXCEPTION 'original generation already priced'; END IF;
 INSERT INTO public.collection_generation_settlements(job_id,request_id,generation_id,original_response_hash,lookup_hash,cost_microusd)
 VALUES(p_job,p_request,p_generation,p_response_hash,p_lookup_hash,p_cost);
 UPDATE public.collection_call_receipts SET cost_microusd=p_cost WHERE job_id=p_job AND request_id=p_request;
 RETURN jsonb_build_object('status','settled','cost_microusd',p_cost);
END $$;

DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_billing_subject_v1(uuid,text)'::regprocedure,
   'public.collection_settle_generation_v1(uuid,text,text,text,text,bigint)'::regprocedure,
   'public.collection_verified_late_cost_v1()'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_billing_subject_v1(uuid,text),
   public.collection_settle_generation_v1(uuid,text,text,text,text,bigint) TO lab_arena_service;
 IF NOT(SELECT had_create FROM pg_temp.pre746_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre746_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
