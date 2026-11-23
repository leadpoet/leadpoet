-- Enroll one real OpenRouter key for the signed synthetic workflow scope.
-- Existing registrations, key ciphertexts, job receipts, grants and RLS stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;

ALTER TABLE public.collection_jobs ALTER COLUMN deepline_key_id DROP NOT NULL;
ALTER TABLE public.collection_jobs ADD CONSTRAINT collection_synthetic_key_scope CHECK (
  deepline_key_id IS NOT NULL OR scenario_id ~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$'
);

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
 IF p_consent->'body'->>'task_scope' IS DISTINCT FROM 'synthetic_workflows'
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

CREATE OR REPLACE FUNCTION public.collection_claim_v1(p_job uuid,p_token_hash text,p_batch text,p_mode text,p_commit text,
 p_scenarios jsonb,p_models jsonb,p_prices jsonb,p_budgets jsonb,p_terms_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE b public.collection_batches; enrollment record; n bigint; generator text; spending bigint; price bigint;
 day_utc date := (clock_timestamp() AT TIME ZONE 'UTC')::date;
 job public.collection_jobs;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF jsonb_typeof(p_scenarios) IS DISTINCT FROM 'array'
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_scenarios) s(id)
               WHERE s.id IS NULL OR s.id !~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$')
 THEN RETURN jsonb_build_object('error','collection_synthetic_scope_required'); END IF;
 INSERT INTO public.collection_batches VALUES(p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets) ON CONFLICT DO NOTHING;
 SELECT * INTO STRICT b FROM public.collection_batches WHERE batch_id=p_batch;
 IF (b.mode,b.harness_commit,b.scenario_ids,b.models,b.prices,b.budgets)
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
 SELECT coalesce(max(slot)+1,0) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;
 IF n >= jsonb_array_length(p_scenarios)::bigint * jsonb_array_length(p_models) THEN RETURN NULL; END IF;
 generator:=p_models->>(n%6)::integer; spending:=(p_budgets->>generator)::bigint; price:=(p_prices->>generator)::bigint;
 SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id INTO enrollment
 FROM public.collection_registrations r JOIN public.collection_keys o ON o.registration_id=r.registration_id AND o.provider='openrouter'
 WHERE r.mode=p_mode AND r.revoked_at IS NULL AND o.revoked_at IS NULL
 AND r.consent->'body'->>'task_scope'='synthetic_workflows'
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
   slot,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,lease_token_hash,lease_expires_at)
 VALUES(p_job,p_batch,p_mode,enrollment.registration_id,enrollment.miner_hotkey,enrollment.openrouter_key_id,NULL,
   n,day_utc,p_scenarios->>(n/6)::integer,generator,p_commit,spending,price,p_token_hash,clock_timestamp()+interval '2 hours')
 RETURNING * INTO job;
 RETURN (to_jsonb(job)-'lease_token_hash') || jsonb_build_object('owner_coldkey',
   (SELECT owner_coldkey FROM public.collection_registrations WHERE registration_id=job.registration_id));
END; $$;

NOTIFY pgrst,'reload schema';
COMMIT;
