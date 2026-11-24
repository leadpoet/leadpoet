-- Admit a bounded paid correction class after the ordinary collection cutover.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE tariff_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_correction_tariffs (
 batch_id text PRIMARY KEY CHECK(batch_id ~ '^xp-[0-9a-f]{64}$'),
 pilot_id text UNIQUE NOT NULL REFERENCES public.collection_batches(batch_id),
 approval jsonb NOT NULL CHECK(jsonb_typeof(approval)='object'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.collection_correction_tariffs OWNER TO lab_arena_owner;
ALTER TABLE public.collection_correction_tariffs ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_correction_tariffs FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.collection_correction_tariff_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT approval FROM public.collection_correction_tariffs WHERE batch_id=p_batch
$$;
-- Public enrollment reports only amounts and model class, never run artifacts, payer identities or review text.
CREATE FUNCTION public.collection_open_correction_quotes_v1() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(active) ORDER BY active.batch_id),'[]'::jsonb) FROM (
  SELECT t.batch_id,t.approval#>>'{measurement,model}' model,t.approval#>>'{measurement,template}' task_class,
         (t.approval#>>'{measurement,quote_microusd}')::bigint quote_microusd,
         (t.approval#>>'{measurement,cap_microusd}')::bigint cap_microusd
  FROM public.collection_correction_tariffs t WHERE EXISTS(SELECT FROM public.collection_reward_cutover)
     AND jsonb_array_length(t.approval->'cases')>(SELECT count(*) FROM public.collection_jobs j WHERE j.batch_id=t.batch_id)
  ORDER BY t.created_at LIMIT 100
 ) active
$$;
CREATE FUNCTION public.collection_experiment_costs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH measured AS (
  SELECT j.job_id,j.batch_id,j.slot,j.attempt,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,j.artifact_sha256,
    j.status,j.budget_microusd,
    count(*) FILTER (WHERE c.key_id IS NOT NULL AND r.cost_microusd IS NULL) unresolved,
    count(*) FILTER (WHERE c.provider='openrouter' AND r.http_status=200 AND r.cost_microusd>0) generator_calls,
    coalesce(sum(r.cost_microusd) FILTER (WHERE c.provider IN ('openrouter','deepline') AND c.key_id IS NOT NULL),0) miner_cost_microusd
  FROM public.collection_jobs j
  LEFT JOIN public.collection_calls c ON j.job_id=c.job_id
  LEFT JOIN public.collection_call_receipts r ON (c.request_id,c.job_id)=(r.request_id,r.job_id)
  JOIN public.collection_batches b ON b.batch_id=j.batch_id
  WHERE j.batch_id=p_batch AND b.experiment_terms_hash IS NOT NULL AND j.network='finney' AND j.netuid=71
  GROUP BY j.job_id
 ) SELECT coalesce(jsonb_agg(to_jsonb(m) ORDER BY job_id),'[]') FROM measured m
$$;


CREATE FUNCTION public.collection_correction_measurement_v1(p_batch text) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE pilot public.collection_batches; rows jsonb; model text; cap bigint; completed integer; cents numeric;
BEGIN
 SELECT * INTO STRICT pilot FROM public.collection_batches WHERE batch_id=p_batch
   AND mode='pilot' AND experiment_terms_hash IS NOT NULL;
 model:=pilot.models->>0; cap:=(pilot.budgets->>model)::bigint;
 IF EXISTS (SELECT FROM jsonb_array_elements_text(pilot.models) g(id) WHERE g.id<>model)
   OR EXISTS (SELECT FROM jsonb_array_elements_text(pilot.scenario_ids) s(id)
      WHERE split_part(s.id,'-s',1) <> split_part(pilot.scenario_ids->>0,'-s',1))
 THEN RAISE EXCEPTION 'pricing pilot must measure one correction class'; END IF;
 rows:=public.collection_experiment_costs_v1(p_batch);
 IF (SELECT count(DISTINCT (r->>'slot')::int) FROM jsonb_array_elements(rows) r) <> jsonb_array_length(pilot.models)
   OR EXISTS (SELECT FROM jsonb_array_elements(rows) r WHERE (r->>'slot')::int NOT BETWEEN 0 AND jsonb_array_length(pilot.models)-1
      OR r->>'status' NOT IN ('accepted','rejected','failed') OR (r->>'unresolved')::int<>0
      OR r->>'model' IS DISTINCT FROM model OR r->>'scenario_id' IS DISTINCT FROM (pilot.scenario_ids->>((r->>'slot')::int)))
   OR EXISTS (SELECT FROM public.collection_jobs j JOIN public.collection_calls c USING(job_id)
        LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
        WHERE j.batch_id=p_batch AND r.cost_microusd IS NULL)
   OR EXISTS (SELECT FROM jsonb_array_elements(rows) r WHERE r->>'status'='accepted' AND
       (r->>'artifact_sha256' IS NULL OR (r->>'generator_calls')::int<1 OR (r->>'miner_cost_microusd')::numeric<=0))
 THEN RAISE EXCEPTION 'incomplete or unsettled correction pilot'; END IF;
 SELECT count(*),sum((r->>'miner_cost_microusd')::numeric) INTO completed,cents
   FROM jsonb_array_elements(rows) r WHERE r->>'status'='accepted';
 IF completed<1 THEN RAISE EXCEPTION 'no measured correction completions'; END IF;
 RETURN jsonb_build_object('model',model,'template',split_part(pilot.scenario_ids->>0,'-s',1),
   'cap_microusd',cap,'harness_commit',pilot.harness_commit,'completed',completed,'cost_microusd',cents,
   'quote_microusd',ceil(5*cents/(4*completed)),'receipts_sha256',encode(sha256(convert_to(rows::text,'UTF8')),'hex'));
END $$;

CREATE FUNCTION public.collection_correction_tariff_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE d jsonb:=NEW.approval; m jsonb; cutoff public.collection_reward_cutover;
BEGIN
 IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'correction tariff is immutable'; END IF;
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 SELECT * INTO STRICT cutoff FROM public.collection_reward_cutover;
 m:=public.collection_correction_measurement_v1(NEW.pilot_id);
 IF EXISTS(SELECT FROM public.collection_batches WHERE batch_id=NEW.batch_id)
   OR d->>'schema' IS DISTINCT FROM 'collection.compensated-correction.v1'
   OR d->>'live_batch' IS DISTINCT FROM NEW.batch_id OR d->>'pilot_batch' IS DISTINCT FROM NEW.pilot_id
   OR d->'measurement' IS DISTINCT FROM m OR NOT(d ?& ARRAY['minimum_runs','class_sha256','cases','panel_sha256','terms_sha256','evidence_sha256','signature'])
   OR coalesce(d->>'signature','') !~ '^[A-Za-z0-9+/]{86}==$'
   OR coalesce(d->>'class_sha256','') !~ '^[0-9a-f]{64}$' OR coalesce(d->>'evidence_sha256','') !~ '^sha256:[0-9a-f]{64}$'
   OR coalesce(d->>'panel_sha256','') !~ '^[0-9a-f]{64}$' OR d->>'panel_sha256' IS DISTINCT FROM cutoff.pilot_evidence->>'panel_qualification_sha256'
   OR coalesce((d->>'minimum_runs')::int,0) NOT BETWEEN 1 AND (m->>'completed')::int
   OR d->>'terms_sha256' IS DISTINCT FROM 'sha256:88d5b6886024e7aa5960c3d76586e2831d96650c071a348b145d19a7e155ba6c'
   OR jsonb_typeof(d->'cases') IS DISTINCT FROM 'array' OR jsonb_array_length(d->'cases') NOT BETWEEN 1 AND 12
   OR EXISTS(SELECT FROM jsonb_array_elements_text(d->'cases') s(id) WHERE s.id IS NULL
      OR s.id !~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$' OR split_part(s.id,'-s',1)<>m->>'template')
 THEN RAISE EXCEPTION 'unapproved correction tariff'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER correction_tariff_immutable BEFORE INSERT OR UPDATE OR DELETE ON public.collection_correction_tariffs
 FOR EACH ROW EXECUTE FUNCTION public.collection_correction_tariff_guard_v1();

ALTER TABLE public.collection_batches DROP CONSTRAINT collection_batches_check;
ALTER TABLE public.collection_batches ADD CHECK (
 jsonb_typeof(models)='array' AND
 CASE WHEN experiment_terms_hash IS NULL THEN jsonb_array_length(models)=6
 ELSE mode IN ('pilot','live') AND batch_id ~ '^xp-[0-9a-f]{64}$' AND experiment_terms_hash ~ '^sha256:[0-9a-f]{64}$'
      AND jsonb_array_length(models)=jsonb_array_length(scenario_ids)
      AND jsonb_array_length(models) BETWEEN 1 AND 12 END
);

CREATE OR REPLACE FUNCTION public.collection_claim_v1(p_job uuid,p_token_hash text,p_batch text,p_mode text,p_commit text,
 p_scenarios jsonb,p_models jsonb,p_prices jsonb,p_budgets jsonb,p_terms_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE b public.collection_batches; enrollment record; n bigint; generator text; spending bigint; price bigint;
 day_utc date := (clock_timestamp() AT TIME ZONE 'UTC')::date;
 experiment boolean := p_batch ~ '^xp-[0-9a-f]{64}$';
 scenario text; job public.collection_jobs; retry_slot bigint; retry_model text; remaining bigint; attempt_number integer := 1;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF jsonb_typeof(p_scenarios) IS DISTINCT FROM 'array'
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_scenarios) s(id)
               WHERE s.id IS NULL OR s.id !~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$')
 THEN RETURN jsonb_build_object('error','collection_synthetic_scope_required'); END IF;
 IF experiment AND p_mode='live' THEN
   PERFORM 1 FROM public.collection_correction_tariffs t CROSS JOIN public.collection_reward_cutover c
   WHERE t.batch_id=p_batch AND t.approval->>'terms_sha256'=p_terms_hash AND t.approval->'cases'=p_scenarios
     AND t.approval#>>'{measurement,harness_commit}'=p_commit AND jsonb_array_length(p_models)=jsonb_array_length(p_scenarios)
     AND NOT EXISTS (SELECT FROM jsonb_array_elements_text(p_models) v(id)
                     WHERE v.id IS DISTINCT FROM t.approval#>>'{measurement,model}')
     AND p_prices=jsonb_build_object(t.approval#>>'{measurement,model}',t.approval#>'{measurement,quote_microusd}')
     AND p_budgets=jsonb_build_object(t.approval#>>'{measurement,model}',t.approval#>'{measurement,cap_microusd}');
   IF NOT FOUND THEN RETURN jsonb_build_object('error','unapproved_correction_tariff'); END IF;
 ELSIF experiment AND (p_mode IS DISTINCT FROM 'pilot' OR p_terms_hash IS NULL
   OR p_terms_hash !~ '^sha256:[0-9a-f]{64}$'
   OR jsonb_typeof(p_models) IS DISTINCT FROM 'array'
   OR jsonb_array_length(p_scenarios) NOT BETWEEN 1 AND 12
   OR jsonb_array_length(p_scenarios)<>jsonb_array_length(p_models)
   OR EXISTS (SELECT FROM jsonb_array_elements_text(p_models) m(id)
              WHERE m.id IS NULL OR NOT p_prices ? m.id OR p_prices->>m.id IS DISTINCT FROM '0'
                 OR NOT p_budgets ? m.id OR (p_budgets->>m.id)::bigint NOT BETWEEN 1 AND 10000000000))
 THEN RETURN jsonb_build_object('error','invalid_correction_pilot'); END IF;
 INSERT INTO public.collection_batches(batch_id,mode,harness_commit,scenario_ids,models,prices,budgets,experiment_terms_hash)
 VALUES(p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets,
        CASE WHEN experiment THEN p_terms_hash END) ON CONFLICT DO NOTHING;
 SELECT * INTO STRICT b FROM public.collection_batches WHERE batch_id=p_batch;
 IF b.experiment_terms_hash IS DISTINCT FROM (CASE WHEN experiment THEN p_terms_hash END) OR
    (b.mode,b.harness_commit,b.scenario_ids,b.models,b.prices,b.budgets)
 IS DISTINCT FROM (p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets)
 THEN RETURN jsonb_build_object('error','batch_changed'); END IF;
 SELECT * INTO job FROM public.collection_jobs WHERE job_id=p_job;
 IF FOUND THEN
  IF job.lease_token_hash<>p_token_hash OR job.batch_id<>p_batch OR job.status<>'running' OR job.lease_expires_at<=clock_timestamp()
  THEN RETURN jsonb_build_object('error','claim_conflict'); END IF;
  RETURN (to_jsonb(job)-'lease_token_hash') || jsonb_build_object('owner_coldkey',
   (SELECT owner_coldkey FROM public.collection_registrations WHERE registration_id=job.registration_id));
 END IF;
 -- Existing leases above can finish while canonical release or operator holds block new work.
 IF NOT EXISTS (SELECT FROM public.lab_arena_restart_claim_control
                WHERE singleton AND NOT operator_paused AND guard_commitment='') THEN RETURN NULL; END IF;
 UPDATE public.collection_jobs SET status='failed',finished_at=clock_timestamp(),quality='{}'
 WHERE status='running' AND lease_expires_at<=clock_timestamp();
 -- A failed pilot attempt can consume only the original slot's unused budget.
 -- A later bill can exceed its reservation, so wait for every original charge to settle.
 IF p_mode='pilot' THEN
   PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
   IF public.collection_reward_schedule_v1()#>>'{pilot_evidence,batch_id}'=p_batch
      OR EXISTS(SELECT FROM public.collection_correction_tariffs WHERE pilot_id=p_batch) THEN RETURN NULL; END IF;
   SELECT j.slot, j.model, j.budget_microusd-cost.committed INTO retry_slot, retry_model, remaining
   FROM public.collection_jobs j
   CROSS JOIN LATERAL (
     SELECT coalesce(sum(r.cost_microusd),0) committed, count(*)=count(r.cost_microusd) settled
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=j.job_id
   ) cost
   WHERE j.batch_id=p_batch AND j.attempt=1 AND j.status='failed' AND j.artifact_sha256 IS NULL
     AND cost.settled AND j.budget_microusd>cost.committed AND NOT EXISTS (
       SELECT FROM public.collection_jobs retry WHERE retry.batch_id=p_batch AND retry.slot=j.slot AND retry.attempt=2)
   ORDER BY j.slot LIMIT 1;
 END IF;
 IF retry_slot IS NOT NULL THEN
   n:=retry_slot; generator:=retry_model; spending:=remaining; price:=0; attempt_number:=2;
 ELSE
   SELECT coalesce(max(slot)+1,0) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;
   IF experiment THEN
     IF n >= jsonb_array_length(p_scenarios) THEN RETURN NULL; END IF;
     generator:=p_models->>n::integer;
   ELSE
     IF n >= jsonb_array_length(p_scenarios)::bigint * jsonb_array_length(p_models) THEN RETURN NULL; END IF;
     generator:=p_models->>(n%6)::integer;
   END IF;
   spending:=(p_budgets->>generator)::bigint; price:=(p_prices->>generator)::bigint;
 END IF;
 scenario:=p_scenarios->>(CASE WHEN experiment THEN n ELSE n/6 END)::integer;
 SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id INTO enrollment
 FROM public.collection_registrations r JOIN public.collection_keys o ON o.registration_id=r.registration_id AND o.provider='openrouter'
 WHERE r.mode=p_mode AND r.revoked_at IS NULL AND o.revoked_at IS NULL
 AND r.consent->'body'->>'task_scope'=(CASE WHEN experiment THEN 'synthetic_corrections' ELSE 'synthetic_workflows' END)
 AND r.consent->'body'->>'terms_sha256'=p_terms_hash
 AND NOT EXISTS (
   SELECT FROM public.collection_jobs j
   JOIN public.collection_registrations owner ON owner.registration_id=j.registration_id
   JOIN public.collection_keys funding ON funding.key_id=j.openrouter_key_id
   WHERE ((j.mode=p_mode AND j.collection_day=day_utc) OR j.status='running') AND
         (j.miner_hotkey=r.miner_hotkey OR owner.owner_coldkey=r.owner_coldkey OR funding.fingerprint=o.fingerprint)
 )
 ORDER BY (SELECT max(j.created_at) FROM public.collection_jobs j WHERE j.miner_hotkey=r.miner_hotkey) ASC NULLS FIRST,
 r.created_at,r.registration_id LIMIT 1 FOR UPDATE OF r;
 IF NOT FOUND THEN RETURN NULL; END IF;
 INSERT INTO public.collection_jobs(job_id,batch_id,mode,registration_id,miner_hotkey,openrouter_key_id,deepline_key_id,
   slot,attempt,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,lease_token_hash,lease_expires_at)
 VALUES(p_job,p_batch,p_mode,enrollment.registration_id,enrollment.miner_hotkey,enrollment.openrouter_key_id,NULL,
   n,attempt_number,day_utc,scenario,generator,p_commit,spending,price,p_token_hash,clock_timestamp()+interval '2 hours')
 RETURNING * INTO job;
 RETURN (to_jsonb(job)-'lease_token_hash') || jsonb_build_object('owner_coldkey',
   (SELECT owner_coldkey FROM public.collection_registrations WHERE registration_id=job.registration_id));
END; $$;
CREATE OR REPLACE FUNCTION public.collection_approved_job_price_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE approval jsonb; family text;
BEGIN
 IF NEW.mode='live' THEN
  PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
  IF EXISTS(SELECT FROM public.collection_batches b WHERE b.batch_id=NEW.batch_id AND b.experiment_terms_hash IS NOT NULL) THEN
   SELECT t.approval INTO STRICT approval FROM public.collection_correction_tariffs t CROSS JOIN public.collection_reward_cutover c
     JOIN public.collection_registrations r ON r.registration_id=NEW.registration_id
     WHERE t.batch_id=NEW.batch_id AND r.mode='live' AND r.miner_hotkey=NEW.miner_hotkey AND
       r.consent#>>'{body,task_scope}'='synthetic_corrections' AND r.consent#>>'{body,terms_sha256}'=t.approval->>'terms_sha256';
   IF (approval->'cases'->>NEW.slot::int) IS DISTINCT FROM NEW.scenario_id
     OR approval#>>'{measurement,model}' IS DISTINCT FROM NEW.model
     OR approval#>>'{measurement,harness_commit}' IS DISTINCT FROM NEW.harness_commit
     OR (approval#>>'{measurement,quote_microusd}')::bigint IS DISTINCT FROM NEW.price_microusd
     OR (approval#>>'{measurement,cap_microusd}')::bigint IS DISTINCT FROM NEW.budget_microusd
   THEN RAISE EXCEPTION 'live correction differs from approved slot'; END IF;
   RETURN NEW;
  END IF;
  approval:=public.collection_reward_schedule_v1()->'pilot_evidence';
  family:=public.collection_fee_class_v1(NEW.scenario_id);
  IF approval IS NULL OR family IS NULL OR approval->>'harness_commit' IS DISTINCT FROM NEW.harness_commit OR
     (approval->'prices'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.price_microusd OR
     (approval->'budgets'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.budget_microusd THEN
   RAISE EXCEPTION 'unapproved live quote or spend ceiling'; END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION public.collection_register_v1(p_id uuid,p_hotkey text,p_owner text,p_mode text,p_consent jsonb,
 p_cap bigint,p_openrouter jsonb,p_deepline jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE reg public.collection_registrations;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_registration:' || p_hotkey || ':' || p_mode,0));
 SELECT * INTO reg FROM public.collection_registrations WHERE registration_id=p_id;
 IF FOUND THEN
  IF reg.miner_hotkey<>p_hotkey OR reg.consent<>p_consent THEN RETURN jsonb_build_object('error','registration_conflict'); END IF;
  RETURN jsonb_build_object('status',CASE WHEN reg.revoked_at IS NULL THEN 'enrolled' ELSE 'revoked' END,'registration_id',p_id);
 END IF;
 IF EXISTS(SELECT FROM public.collection_registrations WHERE miner_hotkey=p_hotkey AND mode=p_mode AND revoked_at IS NULL)
 THEN RETURN jsonb_build_object('error','revoke_current_registration_first'); END IF;
 IF NOT coalesce(p_consent->'body'->>'task_scope' = 'synthetic_workflows' OR (
        p_mode='pilot' AND p_consent->'body'->>'task_scope'='synthetic_corrections'
        AND p_consent->'body'->>'terms_sha256'='sha256:41588b51710c8218e5dce56ef12386709e4ebd2409ca76fe4427251425cc6119') OR (
        p_mode='live' AND p_consent->'body'->>'task_scope'='synthetic_corrections'
        AND p_consent->'body'->>'terms_sha256'='sha256:88d5b6886024e7aa5960c3d76586e2831d96650c071a348b145d19a7e155ba6c'
        AND jsonb_array_length(public.collection_open_correction_quotes_v1())>0),false)
    OR jsonb_typeof(p_openrouter) IS DISTINCT FROM 'object'
    OR (p_deepline IS NOT NULL AND p_deepline <> 'null'::jsonb)
 THEN RETURN jsonb_build_object('error','collection_consent_invalid'); END IF;
 INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent)
 VALUES(p_id,p_hotkey,p_owner,p_mode,p_consent);
 INSERT INTO public.collection_keys(key_id,registration_id,provider,miner_hotkey,submission_id,daily_cap_microusd,fingerprint,ciphertext_b64)
 VALUES((p_openrouter->>'key_id')::uuid,p_id,'openrouter',p_hotkey,'collection:'||p_id::text,p_cap,
        p_openrouter->>'fingerprint',p_openrouter->>'ciphertext_b64');
 RETURN jsonb_build_object('status','enrolled','registration_id',p_id);
EXCEPTION WHEN unique_violation THEN RETURN jsonb_build_object('error','registration_conflict');
END; $$;


DO $$ DECLARE fn regprocedure; BEGIN
 FOREACH fn IN ARRAY ARRAY['public.collection_open_correction_quotes_v1()'::regprocedure,'public.collection_experiment_costs_v1(text)'::regprocedure,
   'public.collection_correction_measurement_v1(text)'::regprocedure,'public.collection_correction_tariff_v1(text)'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',fn);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',fn);
  EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',fn);
 END LOOP;
 ALTER FUNCTION public.collection_correction_tariff_guard_v1() OWNER TO lab_arena_owner;
 REVOKE ALL ON FUNCTION public.collection_correction_tariff_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
 IF NOT(SELECT had_create FROM pg_temp.tariff_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.tariff_acl)
 THEN RAISE EXCEPTION 'public schema permissions changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
