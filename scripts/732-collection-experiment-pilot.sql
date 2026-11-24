-- A comparison pack uses a Cartesian schedule. An experiment uses an explicit
-- (scenario, model) pair per slot. Both use the same consent, caps and receipts.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;

ALTER TABLE public.collection_batches ADD COLUMN experiment_terms_hash text;
ALTER TABLE public.collection_batches DROP CONSTRAINT collection_batches_models_check;
ALTER TABLE public.collection_batches ADD CHECK (
 jsonb_typeof(models)='array' AND
 CASE WHEN experiment_terms_hash IS NULL THEN jsonb_array_length(models)=6
 ELSE mode='pilot' AND batch_id ~ '^xp-[0-9a-f]{64}$' AND experiment_terms_hash ~ '^sha256:[0-9a-f]{64}$'
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
 IF experiment AND (p_mode IS DISTINCT FROM 'pilot' OR p_terms_hash IS NULL
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
   IF public.collection_reward_schedule_v1()#>>'{pilot_evidence,batch_id}'=p_batch THEN RETURN NULL; END IF;
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

COMMIT;
