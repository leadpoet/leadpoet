-- Authorize a priced launch separately. Only a transaction that accepts genuine
-- live work may commit the prospective transition away from champion weights.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE launch_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- A prepared terminal record is private, complete, and recoverable without a miner key.
ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_jobs_status_check;
ALTER TABLE public.collection_jobs ADD CHECK(status IN ('running','ready','accepted','rejected','failed')),
 ADD CHECK(status<>'ready' OR (mode='live' AND artifact_sha256 IS NOT NULL));

CREATE TABLE public.collection_reward_launch (LIKE public.collection_reward_cutover INCLUDING ALL);
ALTER TABLE public.collection_reward_launch OWNER TO lab_arena_owner;
ALTER TABLE public.collection_reward_launch ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_reward_launch FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER validate_launch_evidence BEFORE INSERT ON public.collection_reward_launch
 FOR EACH ROW EXECUTE FUNCTION public.collection_approved_pilot_guard_v1();
CREATE TRIGGER validate_launch_epoch BEFORE INSERT OR UPDATE ON public.collection_reward_launch
 FOR EACH ROW EXECUTE FUNCTION public.collection_reward_cutover_guard_v1();

CREATE FUNCTION public.collection_launch_change_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 IF EXISTS (SELECT FROM public.collection_reward_cutover) THEN RAISE EXCEPTION 'collection already active'; END IF;
 IF TG_OP='DELETE' THEN
  IF EXISTS (SELECT FROM public.collection_jobs WHERE mode='live' AND (status='ready' OR
             (status='running' AND lease_expires_at>clock_timestamp()))) THEN RAISE EXCEPTION 'running collection job'; END IF;
  RETURN OLD;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER check_launch_change BEFORE INSERT OR DELETE ON public.collection_reward_launch
 FOR EACH ROW EXECUTE FUNCTION public.collection_launch_change_v1();

CREATE OR REPLACE FUNCTION public.collection_reward_schedule_v1() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce((SELECT jsonb_build_object('first_epoch',first_epoch,'pilot_evidence',pilot_evidence)
                  FROM public.collection_reward_cutover),
                 (SELECT jsonb_build_object('first_epoch',first_epoch,'pilot_evidence',pilot_evidence,'staged',true,
                   'pending',(SELECT jsonb_build_object('job_id',j.job_id,'miner_hotkey',j.miner_hotkey,
                           'owner_coldkey',r.owner_coldkey,'status',j.status,'batch_id',j.batch_id,
                           'model',j.model,'scenario_id',j.scenario_id,'harness_commit',j.harness_commit,
                           'artifact_sha256',j.artifact_sha256)
                     FROM public.collection_jobs j JOIN public.collection_registrations r USING(registration_id)
                     WHERE j.mode='live' AND (j.status='ready' OR (j.status='running' AND j.lease_expires_at>statement_timestamp()))
                     ORDER BY j.created_at LIMIT 1))
                  FROM public.collection_reward_launch),'{}')
$$;

CREATE OR REPLACE FUNCTION public.collection_approved_job_price_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE approval jsonb; family text;
BEGIN
 IF NEW.mode='live' THEN
  PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
  approval:=public.collection_reward_schedule_v1()->'pilot_evidence';
  family:=public.collection_fee_class_v1(NEW.scenario_id);
  IF approval IS NULL OR family IS NULL OR approval->>'harness_commit' IS DISTINCT FROM NEW.harness_commit OR
     (approval->'prices'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.price_microusd OR
     (approval->'budgets'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.budget_microusd THEN
   RAISE EXCEPTION 'unapproved live quote or spend ceiling'; END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION public.collection_reward_job_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE first_epoch bigint;
BEGIN
 IF TG_OP <> 'INSERT' AND OLD.mode='live' AND OLD.status='accepted' THEN
  RAISE EXCEPTION 'accepted collection job is immutable'; END IF;
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

CREATE FUNCTION public.collection_first_paid_job_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF NOT EXISTS (SELECT FROM public.collection_reward_launch s
       WHERE s.pilot_evidence-'cutover_anchor'=NEW.pilot_evidence-'cutover_anchor')
    OR NOT EXISTS (SELECT FROM public.collection_jobs j
       WHERE j.mode='live' AND j.status='accepted' AND j.accepted_epoch=NEW.first_epoch-1
       AND j.job_id::text=NEW.pilot_evidence#>>'{cutover_anchor,job_id}'
       AND j.miner_hotkey=NEW.pilot_evidence#>>'{cutover_anchor,miner_hotkey}') THEN
  RAISE EXCEPTION 'first paid collection job required'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER collection_first_paid_job AFTER INSERT ON public.collection_reward_cutover
 DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION public.collection_first_paid_job_v1();

-- Persist the verified completion before activation so restarts never repeat provider calls.
CREATE FUNCTION public.collection_prepare_launch_v1(p_job uuid,p_token_hash text,p_quality jsonb,p_artifact text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs;
BEGIN
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job AND mode='live' AND lease_token_hash=p_token_hash FOR UPDATE;
 IF j.status IN ('ready','accepted') AND j.quality=p_quality AND j.artifact_sha256=p_artifact THEN
  RETURN jsonb_build_object('job_id',j.job_id,'status',j.status); END IF;
 IF j.status<>'running' OR j.lease_expires_at<=clock_timestamp()
    OR jsonb_typeof(p_quality) IS DISTINCT FROM 'object' OR p_artifact IS NULL
    OR NOT EXISTS (SELECT FROM public.collection_reward_launch)
    OR EXISTS (SELECT FROM public.collection_reward_cutover)
    OR NOT EXISTS (SELECT FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id)
                   WHERE c.job_id=j.job_id AND c.provider='openrouter' AND c.key_id=j.openrouter_key_id
                   AND r.http_status=200 AND r.cost_microusd>0)
    OR EXISTS (SELECT FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
               WHERE c.job_id=j.job_id AND c.key_id IS NOT NULL AND r.cost_microusd IS NULL)
 THEN RAISE EXCEPTION 'settled first paid collection job required'; END IF;
 UPDATE public.collection_jobs SET status='ready',finished_at=clock_timestamp(),quality=p_quality,artifact_sha256=p_artifact
 WHERE job_id=j.job_id;
 RETURN jsonb_build_object('job_id',j.job_id,'status','ready');
END $$;

CREATE FUNCTION public.collection_launch_finish_v1(p_job uuid,p_anchor jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; e bigint:=(p_anchor->>'epoch')::bigint; approval jsonb;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job AND mode='live' FOR UPDATE;
 IF EXISTS (SELECT FROM public.collection_reward_cutover WHERE pilot_evidence#>>'{cutover_anchor,job_id}'=p_job::text)
    AND j.status='accepted' THEN RETURN jsonb_build_object('status',j.status,'job_id',j.job_id); END IF;
 SELECT pilot_evidence INTO STRICT approval FROM public.collection_reward_launch;
 IF j.status<>'ready' OR p_anchor->>'miner_hotkey' IS DISTINCT FROM j.miner_hotkey THEN
  RAISE EXCEPTION 'first paid collection job required'; END IF;
 INSERT INTO public.collection_reward_cutover(network,netuid,first_epoch,pilot_evidence)
 VALUES ('finney',71,e+1,approval||jsonb_build_object('cutover_anchor',p_anchor||jsonb_build_object('job_id',j.job_id)));
 UPDATE public.collection_jobs SET status='accepted',accepted_epoch=e WHERE job_id=j.job_id;
 RETURN jsonb_build_object('status','accepted','job_id',j.job_id);
END $$;

DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_launch_change_v1()'::regprocedure,'public.collection_first_paid_job_v1()'::regprocedure,
                         'public.collection_prepare_launch_v1(uuid,text,jsonb,text)'::regprocedure,'public.collection_launch_finish_v1(uuid,jsonb)'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_prepare_launch_v1(uuid,text,jsonb,text),public.collection_launch_finish_v1(uuid,jsonb) TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.launch_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.launch_acl)
 THEN RAISE EXCEPTION 'collection schema ACL differs'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
