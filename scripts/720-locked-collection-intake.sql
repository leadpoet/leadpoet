-- Locked training collection. Only the scoped service can call these functions.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = pg_catalog;
CREATE TEMP TABLE collection_713_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_registrations (
 registration_id uuid PRIMARY KEY,
 miner_hotkey text NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
 owner_coldkey text NOT NULL CHECK(owner_coldkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
 mode text NOT NULL CHECK(mode IN ('pilot','live')),
 consent jsonb NOT NULL, -- signed envelope containing hashes only
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 revoked_at timestamptz
);
CREATE UNIQUE INDEX collection_active_miner ON public.collection_registrations(miner_hotkey,mode)
 WHERE revoked_at IS NULL;
CREATE TABLE public.collection_keys (
 key_id uuid PRIMARY KEY,
 registration_id uuid NOT NULL REFERENCES public.collection_registrations,
 provider text NOT NULL CHECK(provider IN ('openrouter','deepline')),
 miner_hotkey text NOT NULL,
 submission_id text NOT NULL,
 daily_cap_microusd bigint NOT NULL CHECK(daily_cap_microusd BETWEEN 1 AND 10000000000),
 fingerprint text NOT NULL CHECK(fingerprint ~ '^sha256:[0-9a-f]{64}$'),
 ciphertext_b64 text NOT NULL CHECK(length(ciphertext_b64) BETWEEN 1 AND 16384),
 revoked_at timestamptz,
 UNIQUE(registration_id,provider)
);
CREATE UNIQUE INDEX collection_unique_active_key ON public.collection_keys(provider,fingerprint)
 WHERE revoked_at IS NULL;
CREATE TABLE public.collection_batches (
 batch_id text PRIMARY KEY CHECK(batch_id ~ '^[A-Za-z0-9_.:-]{1,128}$'),
 mode text NOT NULL CHECK(mode IN ('pilot','live')),
 harness_commit text NOT NULL CHECK(harness_commit ~ '^[0-9a-f]{40}$'),
 scenario_ids jsonb NOT NULL CHECK(jsonb_typeof(scenario_ids)='array' AND jsonb_array_length(scenario_ids) BETWEEN 1 AND 4096),
 models jsonb NOT NULL CHECK(jsonb_typeof(models)='array' AND jsonb_array_length(models)=6),
 prices jsonb NOT NULL, budgets jsonb NOT NULL
);
CREATE TABLE public.collection_jobs (
 job_id uuid PRIMARY KEY, batch_id text NOT NULL REFERENCES public.collection_batches,
 mode text NOT NULL CHECK(mode IN ('pilot','live')),
 network text NOT NULL DEFAULT 'finney' CHECK(network='finney'),
 netuid integer NOT NULL DEFAULT 71 CHECK(netuid=71),
 miner_hotkey text NOT NULL, registration_id uuid NOT NULL REFERENCES public.collection_registrations,
 openrouter_key_id uuid NOT NULL REFERENCES public.collection_keys, deepline_key_id uuid NOT NULL REFERENCES public.collection_keys,
 slot bigint NOT NULL CHECK(slot>=0), collection_day date NOT NULL,
 scenario_id text NOT NULL, model text NOT NULL, harness_commit text NOT NULL,
 budget_microusd bigint NOT NULL CHECK(budget_microusd BETWEEN 1 AND 10000000000),
 price_microusd bigint NOT NULL CHECK(price_microusd>=0 AND (price_microusd>0 OR mode='pilot')),
 status text NOT NULL DEFAULT 'running' CHECK(status IN ('running','accepted','rejected','failed')),
 lease_token_hash text NOT NULL CHECK(lease_token_hash ~ '^sha256:[0-9a-f]{64}$'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), lease_expires_at timestamptz NOT NULL,
 finished_at timestamptz, accepted_epoch bigint CHECK(accepted_epoch>=0),
 artifact_sha256 text CHECK(artifact_sha256 ~ '^sha256:[0-9a-f]{64}$'),
 quality jsonb,
 CHECK((status='accepted')=(accepted_epoch IS NOT NULL)),
 CHECK(status<>'accepted' OR artifact_sha256 IS NOT NULL),
 CHECK((status='running')=(finished_at IS NULL)),
 UNIQUE(mode,collection_day,miner_hotkey), UNIQUE(batch_id,slot)
);
CREATE INDEX collection_pending_job_expiry ON public.collection_jobs(mode,lease_expires_at) WHERE status='running';
CREATE INDEX collection_owner_recent ON public.collection_jobs(miner_hotkey,created_at DESC);

CREATE FUNCTION public.collection_register_v1(p_id uuid,p_hotkey text,p_owner text,p_mode text,p_consent jsonb,
 p_cap bigint,p_openrouter jsonb,p_deepline jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE reg public.collection_registrations; k jsonb; prov text;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_registration:' || p_hotkey || ':' || p_mode,0));
 SELECT * INTO reg FROM public.collection_registrations WHERE registration_id=p_id;
 IF FOUND THEN
  IF reg.miner_hotkey<>p_hotkey OR reg.consent<>p_consent THEN RETURN jsonb_build_object('error','registration_conflict'); END IF;
  RETURN jsonb_build_object('status',CASE WHEN reg.revoked_at IS NULL THEN 'enrolled' ELSE 'revoked' END,'registration_id',p_id);
 END IF;
 IF EXISTS(SELECT FROM public.collection_registrations WHERE miner_hotkey=p_hotkey AND mode=p_mode AND revoked_at IS NULL)
 THEN RETURN jsonb_build_object('error','revoke_current_registration_first'); END IF;
 INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent)
 VALUES(p_id,p_hotkey,p_owner,p_mode,p_consent);
 FOREACH prov IN ARRAY ARRAY['openrouter','deepline'] LOOP
  k:=CASE WHEN prov='openrouter' THEN p_openrouter ELSE p_deepline END;
  INSERT INTO public.collection_keys(key_id,registration_id,provider,miner_hotkey,submission_id,daily_cap_microusd,fingerprint,ciphertext_b64)
   VALUES((k->>'key_id')::uuid,p_id,prov,p_hotkey,'collection:'||p_id::text,p_cap,k->>'fingerprint',k->>'ciphertext_b64');
 END LOOP;
 RETURN jsonb_build_object('status','enrolled','registration_id',p_id);
EXCEPTION WHEN unique_violation THEN RETURN jsonb_build_object('error','registration_conflict');
END; $$;

CREATE FUNCTION public.collection_revoke_v1(p_id uuid,p_hotkey text,p_mode text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 PERFORM 1 FROM public.collection_registrations WHERE registration_id=p_id AND miner_hotkey=p_hotkey AND mode=p_mode FOR UPDATE;
 IF NOT FOUND THEN RETURN jsonb_build_object('error','registration_unknown'); END IF;
 UPDATE public.collection_registrations SET revoked_at=clock_timestamp() WHERE registration_id=p_id AND revoked_at IS NULL;
 UPDATE public.collection_keys SET revoked_at=clock_timestamp() WHERE registration_id=p_id AND revoked_at IS NULL;
 RETURN jsonb_build_object('status','revoked','registration_id',p_id);
END; $$;

CREATE FUNCTION public.collection_claim_v1(p_job uuid,p_token_hash text,p_batch text,p_mode text,p_commit text,
 p_scenarios jsonb,p_models jsonb,p_prices jsonb,p_budgets jsonb,p_terms_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE b public.collection_batches; enrollment record; n bigint; generator text; spending bigint; price bigint;
 day_utc date := (clock_timestamp() AT TIME ZONE 'UTC')::date;
 job public.collection_jobs;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:' || p_mode,0));
 INSERT INTO public.collection_batches VALUES(p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets) ON CONFLICT DO NOTHING;
 SELECT * INTO STRICT b FROM public.collection_batches WHERE batch_id=p_batch;
 IF (b.mode,b.harness_commit,b.scenario_ids,b.models,b.prices,b.budgets)
 IS DISTINCT FROM (p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets)
 THEN RETURN jsonb_build_object('error','batch_changed'); END IF;
 SELECT * INTO job FROM public.collection_jobs WHERE job_id=p_job;
 IF FOUND THEN
  IF job.lease_token_hash<>p_token_hash OR job.batch_id<>p_batch OR job.status<>'running' OR job.lease_expires_at<=clock_timestamp()
  THEN RETURN jsonb_build_object('error','claim_conflict'); END IF;
  RETURN to_jsonb(job)-'lease_token_hash';
 END IF;
 UPDATE public.collection_jobs SET status='failed',finished_at=clock_timestamp(),quality='{}'
 WHERE mode=p_mode AND status='running' AND lease_expires_at<=clock_timestamp();
 SELECT coalesce(max(slot)+1,0) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;
 IF n >= jsonb_array_length(p_scenarios)::bigint * jsonb_array_length(p_models) THEN RETURN NULL; END IF;
 generator:=p_models->>(n%6)::integer; spending:=(p_budgets->>generator)::bigint; price:=(p_prices->>generator)::bigint;
 SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id,d.key_id deepline_key_id INTO enrollment
 FROM public.collection_registrations r JOIN public.collection_keys o ON o.registration_id=r.registration_id AND o.provider='openrouter'
 JOIN public.collection_keys d ON d.registration_id=r.registration_id AND d.provider='deepline'
 WHERE r.mode=p_mode AND r.revoked_at IS NULL AND o.revoked_at IS NULL AND d.revoked_at IS NULL
 AND r.consent->'body'->>'terms_sha256'=p_terms_hash
 AND NOT EXISTS(SELECT FROM public.collection_jobs j WHERE j.mode=p_mode AND j.collection_day=day_utc AND j.miner_hotkey=r.miner_hotkey)
 AND NOT EXISTS(SELECT FROM public.collection_jobs j WHERE j.status='running' AND j.miner_hotkey=r.miner_hotkey)
 ORDER BY (SELECT max(j.created_at) FROM public.collection_jobs j WHERE j.miner_hotkey=r.miner_hotkey) ASC NULLS FIRST,
 r.created_at,r.registration_id LIMIT 1 FOR UPDATE OF r;
 IF NOT FOUND THEN RETURN NULL; END IF;
 INSERT INTO public.collection_jobs(job_id,batch_id,mode,registration_id,miner_hotkey,openrouter_key_id,deepline_key_id,
   slot,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,lease_token_hash,lease_expires_at)
 VALUES(p_job,p_batch,p_mode,enrollment.registration_id,enrollment.miner_hotkey,enrollment.openrouter_key_id,enrollment.deepline_key_id,
   n,day_utc,p_scenarios->>(n/6)::integer,generator,p_commit,spending,price,p_token_hash,clock_timestamp()+interval '2 hours')
 RETURNING * INTO job;
 RETURN to_jsonb(job)-'lease_token_hash';
END; $$;

CREATE FUNCTION public.collection_credential_v1(p_job uuid,p_token_hash text,p_provider text)
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('key_id',k.key_id,'provider',k.provider,'miner_hotkey',k.miner_hotkey,
   'submission_id',k.submission_id,'ciphertext_b64',k.ciphertext_b64)
 FROM public.collection_jobs j JOIN public.collection_keys k ON k.key_id=CASE p_provider
 WHEN 'openrouter' THEN j.openrouter_key_id WHEN 'deepline' THEN j.deepline_key_id END
 WHERE j.job_id=p_job AND j.status='running' AND j.lease_token_hash=p_token_hash AND
 j.lease_expires_at>statement_timestamp() AND k.revoked_at IS NULL;
$$;

CREATE FUNCTION public.collection_finish_v1(p_job uuid,p_token_hash text,p_status text,p_quality jsonb,p_artifact text,p_epoch bigint)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs;
BEGIN
 SELECT * INTO j FROM public.collection_jobs WHERE job_id=p_job AND lease_token_hash=p_token_hash FOR UPDATE;
 IF NOT FOUND THEN RETURN jsonb_build_object('error','lease_invalid'); END IF;
 IF j.status<>'running' THEN
  IF (j.status,j.quality,j.artifact_sha256) IS DISTINCT FROM (p_status,p_quality,p_artifact) THEN
   RETURN jsonb_build_object('error','outcome_conflict'); END IF;
  RETURN jsonb_build_object('status',j.status,'job_id',j.job_id);
 END IF;
 IF j.lease_expires_at<=clock_timestamp() THEN RETURN jsonb_build_object('error','lease_expired'); END IF;
 IF p_status NOT IN ('accepted','rejected','failed') OR p_quality IS NULL OR jsonb_typeof(p_quality)<>'object'
 OR (p_status IN ('accepted','rejected') AND p_artifact IS NULL) OR p_epoch<0 THEN
 RETURN jsonb_build_object('error','invalid_outcome'); END IF;
 UPDATE public.collection_jobs SET status=p_status,quality=p_quality,artifact_sha256=p_artifact,
  accepted_epoch=CASE WHEN p_status='accepted' THEN p_epoch END,finished_at=clock_timestamp() WHERE job_id=p_job;
 RETURN jsonb_build_object('status',p_status,'job_id',p_job);
END; $$;

CREATE FUNCTION public.collection_statement_v1(p_hotkey text,p_mode text)
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('enrollments',coalesce((SELECT jsonb_agg(to_jsonb(t)) FROM (
  SELECT registration_id,created_at,revoked_at FROM public.collection_registrations
  WHERE miner_hotkey=p_hotkey AND mode=p_mode ORDER BY created_at DESC LIMIT 50) t),'[]'::jsonb),
  'jobs',coalesce((SELECT jsonb_agg(to_jsonb(t)) FROM (SELECT job_id,slot,model,status,scenario_id,created_at,finished_at,
  budget_microusd,price_microusd,quality FROM public.collection_jobs WHERE miner_hotkey=p_hotkey AND mode=p_mode
  ORDER BY created_at DESC LIMIT 100) t),'[]'::jsonb));
$$;

DO $acl$
DECLARE name text; f regprocedure;
BEGIN
 FOREACH name IN ARRAY ARRAY['collection_registrations','collection_keys','collection_batches','collection_jobs'] LOOP
  EXECUTE format('ALTER TABLE public.%I OWNER TO lab_arena_owner',name);
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',name);
  EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',name);
 END LOOP;
 FOR f IN SELECT p.oid::regprocedure FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 WHERE n.nspname='public' AND p.proname=ANY(ARRAY['collection_register_v1','collection_revoke_v1','collection_claim_v1',
 'collection_credential_v1','collection_finish_v1','collection_statement_v1']) LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
 END LOOP;
 IF NOT(SELECT had_create FROM pg_temp.collection_713_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.collection_713_acl)
 THEN RAISE EXCEPTION 'collection schema ACL changed'; END IF;
END; $acl$;
NOTIFY pgrst,'reload schema';
COMMIT;
