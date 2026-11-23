-- Atomically admit each funded provider request once. An unresolved bill
-- continues consuming its reservation. Only trusted collection workers use RPCs.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE collection715_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_calls (
 job_id uuid NOT NULL REFERENCES public.collection_jobs(job_id),
 request_id text NOT NULL CHECK(length(request_id) BETWEEN 1 AND 160),
 provider text NOT NULL CHECK(provider IN ('openrouter','deepline','judge')),
 key_id uuid REFERENCES public.collection_keys(key_id),
 request_hash text NOT NULL CHECK(request_hash ~ '^[a-f0-9]{64}$'),
 reserved_microusd bigint NOT NULL CHECK(reserved_microusd > 0),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CHECK((provider='judge')=(key_id IS NULL)),
 PRIMARY KEY(job_id,request_id)
);
CREATE INDEX collection_calls_key_time ON public.collection_calls(key_id,created_at);
CREATE TABLE public.collection_call_receipts (
 job_id uuid NOT NULL,
 request_id text NOT NULL,
 response_hash text NOT NULL CHECK(response_hash ~ '^[a-f0-9]{64}$'),
 provider_id text CHECK(length(provider_id) <= 256),
 cost_microusd bigint CHECK(cost_microusd >= 0),
 http_status integer NOT NULL CHECK(http_status=0 OR http_status BETWEEN 100 AND 599),
 PRIMARY KEY(job_id,request_id),
 FOREIGN KEY(job_id,request_id) REFERENCES public.collection_calls
);
ALTER TABLE public.collection_calls OWNER TO lab_arena_owner;
ALTER TABLE public.collection_call_receipts OWNER TO lab_arena_owner;
ALTER TABLE public.collection_calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collection_call_receipts ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_calls,public.collection_call_receipts
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_calls_immutable BEFORE UPDATE OR DELETE ON public.collection_calls
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
CREATE TRIGGER collection_receipts_immutable BEFORE UPDATE OR DELETE ON public.collection_call_receipts
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE FUNCTION public.collection_reserve_call_v1(p_job uuid,p_lease text,p_request text,p_provider text,p_hash text,p_amount bigint)
 RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; k public.collection_keys; job_used numeric; key_used numeric;
BEGIN
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job FOR UPDATE;
 IF j.status<>'running' OR j.lease_token_hash IS DISTINCT FROM p_lease
    OR j.lease_expires_at <= clock_timestamp() THEN RAISE EXCEPTION 'inactive_collection_lease'; END IF;
 IF p_amount IS NULL OR p_amount<1 OR p_provider IS NULL OR p_provider NOT IN ('openrouter','deepline','judge')
    THEN RAISE EXCEPTION 'invalid_collection_reservation'; END IF;
 IF p_provider<>'judge' THEN
 SELECT * INTO STRICT k FROM public.collection_keys
 WHERE key_id=CASE p_provider WHEN 'openrouter' THEN j.openrouter_key_id ELSE j.deepline_key_id END
 AND provider=p_provider AND miner_hotkey=j.miner_hotkey AND revoked_at IS NULL FOR UPDATE;
 END IF;
 IF EXISTS(SELECT 1 FROM public.collection_calls WHERE job_id=p_job AND request_id=p_request) THEN
  RETURN jsonb_build_object('status','already_reserved'); END IF;
 SELECT coalesce(sum(coalesce(r.cost_microusd,c.reserved_microusd)),0) INTO job_used
 FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id) WHERE c.job_id=p_job;
 SELECT coalesce(sum(coalesce(r.cost_microusd,c.reserved_microusd)),0) INTO key_used
 FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
 JOIN public.collection_keys historic ON c.key_id=historic.key_id
 WHERE historic.fingerprint=k.fingerprint AND historic.provider=k.provider AND c.created_at>clock_timestamp()-interval '24 hours';
 IF job_used+p_amount>j.budget_microusd OR (p_provider<>'judge' AND key_used+p_amount>k.daily_cap_microusd) THEN
  RETURN jsonb_build_object('status','budget_exhausted'); END IF;
 INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd)
 VALUES(p_job,p_request,p_provider,k.key_id,p_hash,p_amount);
 RETURN jsonb_build_object('status','reserved');
END $$;

CREATE FUNCTION public.collection_record_call_v1(p_job uuid,p_lease text,p_request text,p_hash text,p_provider_id text,p_cost bigint,p_status integer)
 RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.collection_jobs WHERE job_id=p_job AND lease_token_hash=p_lease)
 THEN RAISE EXCEPTION 'invalid_collection_lease'; END IF;
 INSERT INTO public.collection_call_receipts VALUES(p_job,p_request,p_hash,p_provider_id,p_cost,p_status)
 ON CONFLICT DO NOTHING;
 IF NOT EXISTS(SELECT 1 FROM public.collection_call_receipts WHERE job_id=p_job AND request_id=p_request
 AND response_hash=p_hash AND provider_id IS NOT DISTINCT FROM p_provider_id
 AND cost_microusd IS NOT DISTINCT FROM p_cost AND http_status=p_status)
 THEN RAISE EXCEPTION 'collection_settlement_conflict'; END IF;
 RETURN jsonb_build_object('status','recorded');
END $$;
ALTER FUNCTION public.collection_reserve_call_v1(uuid,text,text,text,text,bigint) OWNER TO lab_arena_owner;
ALTER FUNCTION public.collection_record_call_v1(uuid,text,text,text,text,bigint,integer) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_reserve_call_v1(uuid,text,text,text,text,bigint),
 public.collection_record_call_v1(uuid,text,text,text,text,bigint,integer) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.collection_reserve_call_v1(uuid,text,text,text,text,bigint),
 public.collection_record_call_v1(uuid,text,text,text,text,bigint,integer) TO lab_arena_service;

CREATE FUNCTION public.collection_verify_funding_v1() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $fn$
BEGIN
 IF NEW.status='accepted' AND OLD.status IS DISTINCT FROM NEW.status AND (
   NOT EXISTS(SELECT FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=NEW.job_id AND c.provider='openrouter' AND r.http_status=200 AND r.cost_microusd>0)
   OR EXISTS(SELECT FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=NEW.job_id AND r.cost_microusd IS NULL)
 ) THEN RAISE EXCEPTION 'missing verified collection funding'; END IF;
 RETURN NEW;
END $fn$;
ALTER FUNCTION public.collection_verify_funding_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_verify_funding_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_funding_guard BEFORE UPDATE OF status ON public.collection_jobs
 FOR EACH ROW EXECUTE FUNCTION public.collection_verify_funding_v1();

CREATE FUNCTION public.collection_job_usage_v1(p_job uuid, p_lease text)
 RETURNS jsonb LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE answer jsonb;
BEGIN
 IF NOT EXISTS (SELECT FROM public.collection_jobs WHERE job_id=p_job AND lease_token_hash=p_lease
                AND status='running' AND lease_expires_at>statement_timestamp()) THEN
   RAISE EXCEPTION 'inactive_collection_lease'; END IF;
 WITH per_provider AS (
  SELECT c.provider, count(*) used, count(*) FILTER(WHERE r.request_id IS NULL) inflight,
  sum(coalesce(r.cost_microusd,0)) settled,
  sum(CASE WHEN r.cost_microusd IS NULL THEN c.reserved_microusd ELSE 0 END) held
  FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
  WHERE c.job_id=p_job AND c.provider<>'judge' GROUP BY c.provider)
 SELECT jsonb_build_object('settled_microusd',coalesce(sum(settled),0),'reserved_or_uncertain_microusd',
 coalesce(sum(held),0),'inflight_calls',coalesce(sum(inflight),0),'providers',
 coalesce(jsonb_object_agg(provider,jsonb_build_object('used',used,'inflight',inflight)),'{}'::jsonb))
 INTO answer FROM per_provider;
 RETURN answer;
END $$;
ALTER FUNCTION public.collection_job_usage_v1(uuid,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_job_usage_v1(uuid,text) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.collection_job_usage_v1(uuid,text) TO lab_arena_service;

DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.collection715_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.collection715_acl)
 THEN RAISE EXCEPTION 'collection schema ACL differs'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
