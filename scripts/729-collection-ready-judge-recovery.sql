-- A ready job has finished spending on the miner's account. Only its original
-- owner-funded verification may be retried, within the same job's remaining cap.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE review_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_launch_reviews (
 job_id uuid NOT NULL REFERENCES public.collection_jobs,
 attempt_id uuid NOT NULL,
 source_sha256 text NOT NULL CHECK (source_sha256 ~ '^sha256:[a-f0-9]{64}$'),
 expires_at timestamptz NOT NULL,
 artifact_sha256 text CHECK (artifact_sha256 ~ '^sha256:[a-f0-9]{64}$'),
 quality jsonb,
 requests jsonb,
 PRIMARY KEY(job_id,attempt_id)
);
CREATE UNIQUE INDEX collection_one_launch_review ON public.collection_launch_reviews(job_id)
 WHERE artifact_sha256 IS NOT NULL;
ALTER TABLE public.collection_launch_reviews OWNER TO lab_arena_owner;
ALTER TABLE public.collection_launch_reviews ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_launch_reviews FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Keep older provider receipts immutable. Only an exact, authenticated
-- generation lookup can resolve a previously unknown owner charge.
CREATE TABLE public.collection_launch_judge_costs (
 job_id uuid NOT NULL, request_id text NOT NULL, generation_id text NOT NULL,
 cost_microusd bigint NOT NULL CHECK(cost_microusd>=0),
 receipt_sha256 text NOT NULL CHECK(receipt_sha256 ~ '^[0-9a-f]{64}$'),
 PRIMARY KEY(job_id,request_id), UNIQUE(job_id,generation_id),
 FOREIGN KEY(job_id,request_id) REFERENCES public.collection_call_receipts
);
ALTER TABLE public.collection_launch_judge_costs OWNER TO lab_arena_owner;
ALTER TABLE public.collection_launch_judge_costs ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_launch_judge_costs FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_launch_cost_immutable BEFORE UPDATE OR DELETE ON public.collection_launch_judge_costs
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE FUNCTION public.collection_reconcile_launch_judge_v1(p_job uuid,p_source text,p_request text,
 p_generation text,p_cost bigint,p_hash text) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs;
BEGIN
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job FOR UPDATE;
 IF j.mode<>'live' OR j.status<>'ready' OR j.artifact_sha256 IS DISTINCT FROM p_source
    OR j.quality->>'training_status' IS DISTINCT FROM 'cannot_judge'
    OR p_generation IS NULL OR p_generation !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    OR (SELECT count(*) FROM public.collection_call_receipts WHERE job_id=p_job AND provider_id=p_generation)<>1
    OR NOT EXISTS (SELECT FROM public.collection_call_receipts r JOIN public.collection_calls c USING(job_id,request_id)
                   WHERE c.job_id=p_job AND c.request_id=p_request AND c.provider='judge' AND c.key_id IS NULL
                   AND r.provider_id=p_generation AND r.cost_microusd IS NULL)
 THEN RAISE EXCEPTION 'original owner generation does not match'; END IF;
 INSERT INTO public.collection_launch_judge_costs VALUES(p_job,p_request,p_generation,p_cost,p_hash) ON CONFLICT DO NOTHING;
 IF NOT EXISTS (SELECT FROM public.collection_launch_judge_costs WHERE job_id=p_job AND request_id=p_request
                AND generation_id=p_generation AND cost_microusd=p_cost)
 THEN RAISE EXCEPTION 'conflicting owner generation receipt'; END IF;
 RETURN jsonb_build_object('status','reconciled');
END $$;

CREATE FUNCTION public.collection_reserve_launch_judge_v1(p_job uuid,p_source text,p_attempt uuid,p_request text,
 p_hash text,p_amount bigint) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; r public.collection_launch_reviews; used numeric; unknown_count bigint; recoverable jsonb; call_id text:=p_attempt::text||':'||p_request;
BEGIN
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job FOR UPDATE;
 IF j.mode<>'live' OR j.status<>'ready' OR j.artifact_sha256 IS DISTINCT FROM p_source
    OR j.quality->>'training_status' IS DISTINCT FROM 'cannot_judge' OR EXISTS (SELECT FROM public.collection_reward_cutover)
    OR NOT EXISTS (SELECT FROM public.collection_reward_launch)
    OR p_request IS NULL OR p_request !~ '^collection-[0-9a-f]{64}-[12]$' OR p_attempt IS NULL
    OR p_amount IS NULL OR p_amount<1 THEN RAISE EXCEPTION 'invalid launch judgment request'; END IF;
 IF EXISTS (SELECT FROM public.collection_launch_reviews WHERE job_id=p_job AND attempt_id<>p_attempt
            AND expires_at>clock_timestamp()) THEN RAISE EXCEPTION 'another launch review is in progress'; END IF;
 SELECT * INTO r FROM public.collection_launch_reviews WHERE job_id=p_job AND attempt_id=p_attempt;
 IF FOUND AND (r.source_sha256<>p_source OR r.expires_at<=clock_timestamp() OR r.artifact_sha256 IS NOT NULL)
 THEN RAISE EXCEPTION 'launch review expired or complete'; END IF;
 IF EXISTS (SELECT FROM public.collection_calls WHERE job_id=p_job AND request_id=call_id) THEN
  RETURN jsonb_build_object('status','already_reserved'); END IF;
 IF (SELECT count(*) FROM public.collection_calls WHERE job_id=p_job AND provider='judge'
        AND starts_with(request_id,p_attempt::text||':')) >=4 THEN RAISE EXCEPTION 'launch review call limit'; END IF;
 -- A provider may bill above its estimate. Wait for exact amounts on every
 -- prior call before authorizing a separate recovery dispatch.
 SELECT coalesce(sum(coalesce(receipt.cost_microusd,proven.cost_microusd)),0),
        count(*) FILTER (WHERE coalesce(receipt.cost_microusd,proven.cost_microusd) IS NULL)
 INTO used,unknown_count FROM public.collection_calls c
 LEFT JOIN public.collection_call_receipts receipt USING(job_id,request_id)
 LEFT JOIN public.collection_launch_judge_costs proven USING(job_id,request_id) WHERE c.job_id=p_job;
 IF unknown_count>0 THEN
  SELECT coalesce(jsonb_agg(to_jsonb(item)),'[]') INTO recoverable FROM (
   SELECT c.request_id,receipt.provider_id AS generation_id FROM public.collection_calls c
   JOIN public.collection_call_receipts receipt USING(job_id,request_id)
   LEFT JOIN public.collection_launch_judge_costs proven USING(job_id,request_id)
   WHERE c.job_id=p_job AND c.provider='judge' AND c.key_id IS NULL AND receipt.cost_microusd IS NULL
     AND proven.cost_microusd IS NULL AND receipt.provider_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
   ORDER BY c.created_at,c.request_id LIMIT 16) item;
  RETURN jsonb_build_object('status','unresolved_prior_charge','judge_receipts',recoverable);
 END IF;
 IF used+p_amount>j.budget_microusd THEN RETURN jsonb_build_object('status','budget_exhausted'); END IF;
 INSERT INTO public.collection_launch_reviews(job_id,attempt_id,source_sha256,expires_at)
 VALUES(p_job,p_attempt,p_source,clock_timestamp()+interval '10 minutes') ON CONFLICT DO NOTHING;
 INSERT INTO public.collection_calls(job_id,request_id,provider,key_id,request_hash,reserved_microusd)
 VALUES(p_job,call_id,'judge',NULL,p_hash,p_amount);
 RETURN jsonb_build_object('status','reserved');
END $$;

CREATE FUNCTION public.collection_record_launch_judge_v1(p_job uuid,p_source text,p_attempt uuid,p_request text,p_hash text,
 p_provider_id text,p_cost bigint,p_status integer) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE token_hash text; call_id text:=p_attempt::text||':'||p_request;
BEGIN
 SELECT j.lease_token_hash INTO STRICT token_hash FROM public.collection_launch_reviews r
 JOIN public.collection_jobs j USING(job_id) JOIN public.collection_calls c USING(job_id)
 WHERE r.job_id=p_job AND r.source_sha256=p_source AND r.attempt_id=p_attempt
 AND c.request_id=call_id AND c.provider='judge' AND c.key_id IS NULL;
 -- Settle a dispatched call even after lease expiry or an uncertain response.
 RETURN public.collection_record_call_v1(p_job,token_hash,call_id,p_hash,p_provider_id,p_cost,p_status);
END $$;

CREATE FUNCTION public.collection_finish_launch_review_v1(p_job uuid,p_source text,p_attempt uuid,p_artifact text,
 p_quality jsonb,p_requests jsonb) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; r public.collection_launch_reviews;
BEGIN
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job FOR UPDATE;
 SELECT * INTO STRICT r FROM public.collection_launch_reviews WHERE job_id=p_job AND attempt_id=p_attempt AND source_sha256=p_source;
 IF r.artifact_sha256 IS NOT NULL THEN
  IF (r.artifact_sha256,r.quality,r.requests) IS DISTINCT FROM (p_artifact,p_quality,p_requests)
  THEN RAISE EXCEPTION 'conflicting launch review'; END IF;
  RETURN jsonb_build_object('artifact_sha256',r.artifact_sha256); END IF;
 IF j.mode<>'live' OR j.status<>'ready' OR j.artifact_sha256 IS DISTINCT FROM p_source
    OR j.quality->>'training_status' IS DISTINCT FROM 'cannot_judge' OR r.expires_at<=clock_timestamp()
    OR p_artifact IS NULL OR p_artifact=p_source OR jsonb_typeof(p_quality) IS DISTINCT FROM 'object'
    OR coalesce(p_quality->>'training_status','') NOT IN ('accepted','salvaged','needs_review','rejected')
    OR jsonb_typeof(p_requests) IS DISTINCT FROM 'array' OR jsonb_array_length(p_requests) NOT BETWEEN 1 AND 4
 THEN RAISE EXCEPTION 'invalid launch review result'; END IF;
 IF EXISTS (SELECT FROM jsonb_array_elements_text(p_requests) AS req(id)
    WHERE id IS NULL OR id !~ '^collection-[0-9a-f]{64}-[12]$'
    OR NOT EXISTS (SELECT FROM public.collection_calls c JOIN public.collection_call_receipts rr USING(job_id,request_id)
        WHERE c.job_id=p_job AND c.request_id=p_attempt::text||':'||id AND c.provider='judge' AND c.key_id IS NULL
          AND rr.http_status=200 AND rr.cost_microusd IS NOT NULL AND rr.provider_id IS NOT NULL))
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_requests) req GROUP BY req HAVING count(*)>1)
 THEN RAISE EXCEPTION 'verified owner judgment receipts required'; END IF;
 UPDATE public.collection_launch_reviews SET artifact_sha256=p_artifact,quality=p_quality,requests=p_requests
 WHERE job_id=p_job AND attempt_id=p_attempt;
 UPDATE public.collection_jobs SET artifact_sha256=p_artifact,quality=p_quality WHERE job_id=p_job;
 RETURN jsonb_build_object('artifact_sha256',p_artifact);
END $$;

CREATE OR REPLACE FUNCTION public.collection_reward_job_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE first_epoch bigint;
BEGIN
 IF TG_OP <> 'INSERT' AND OLD.mode='live' AND OLD.status='accepted' THEN
  RAISE EXCEPTION 'accepted collection job is immutable'; END IF;
 IF TG_OP='UPDATE' AND OLD.status='ready' THEN
  IF NEW.status='accepted' AND OLD.quality->>'training_status'='cannot_judge'
  THEN RAISE EXCEPTION 'first live collection job awaits owner verification'; END IF;
  IF NEW.status='ready' AND (to_jsonb(NEW)-'quality'-'artifact_sha256')=(to_jsonb(OLD)-'quality'-'artifact_sha256')
     AND EXISTS (SELECT FROM public.collection_launch_reviews r WHERE r.job_id=OLD.job_id
        AND r.source_sha256=OLD.artifact_sha256 AND r.artifact_sha256=NEW.artifact_sha256 AND r.quality=NEW.quality)
  THEN RETURN NEW; END IF;
 END IF;
 IF TG_OP <> 'INSERT' AND OLD.status='ready' AND (TG_OP='DELETE' OR NEW.status<>'accepted'
    OR (NEW.miner_hotkey,NEW.quality,NEW.artifact_sha256,NEW.price_microusd,NEW.finished_at)
       IS DISTINCT FROM (OLD.miner_hotkey,OLD.quality,OLD.artifact_sha256,OLD.price_microusd,OLD.finished_at)) THEN
  RAISE EXCEPTION 'completed launch evidence is immutable'; END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 IF NEW.mode='live' AND NEW.status IN ('running','accepted') THEN
  PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
  SELECT c.first_epoch INTO first_epoch FROM public.collection_reward_cutover c;
  IF first_epoch IS NULL THEN
   IF NEW.status='running' AND EXISTS (SELECT FROM public.collection_reward_launch) THEN
    -- The claim RPC already serializes allocations by mode. Allow one launch job before activation.
    IF TG_OP='INSERT' AND EXISTS (SELECT FROM public.collection_jobs WHERE mode='live'
       AND (status='ready' OR (status='running' AND lease_expires_at>clock_timestamp()))) THEN
     RAISE EXCEPTION 'first live collection job already in progress'; END IF;
    RETURN NEW;
   END IF;
   RAISE EXCEPTION 'collection payout cutover required';
  END IF;
  IF NEW.status='accepted' THEN
   IF NEW.accepted_epoch < first_epoch-1 THEN RAISE EXCEPTION 'collection job predates cutover'; END IF;
   PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_epoch:'||(NEW.accepted_epoch+1)::text,0));
   IF EXISTS(SELECT FROM public.lab_arena_accepted_weight_states
             WHERE network='finney' AND netuid=71 AND epoch=NEW.accepted_epoch+1) THEN
    RAISE EXCEPTION 'collection job epoch closed: retry with current finalized epoch'; END IF;
  END IF;
 END IF;
 RETURN NEW;
END $$;

DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_reconcile_launch_judge_v1(uuid,text,text,text,bigint,text)'::regprocedure,
  'public.collection_reserve_launch_judge_v1(uuid,text,uuid,text,text,bigint)'::regprocedure,
  'public.collection_record_launch_judge_v1(uuid,text,uuid,text,text,text,bigint,integer)'::regprocedure,
  'public.collection_finish_launch_review_v1(uuid,text,uuid,text,jsonb,jsonb)'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
 END LOOP;
 IF NOT (SELECT had_create FROM pg_temp.review_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.review_acl)
 THEN RAISE EXCEPTION 'collection schema ACL differs'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
