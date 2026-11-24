-- One signed offer for new synthetic work. Original enrollment documents remain immutable.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE pre744_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Only this private reader can expose the original signed document. No key material
-- or judge charges are returned. A later revocation does not rewrite a completed job.
CREATE FUNCTION public.collection_training_consents_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(rows) ORDER BY job_id),'[]'::jsonb) FROM (
   SELECT j.job_id,j.miner_hotkey,j.registration_id,j.mode,r.consent,
          usage.miner_cost_microusd,usage.unresolved,usage.generator_calls
   FROM public.collection_jobs j JOIN public.collection_registrations r
     ON (j.registration_id,j.miner_hotkey,j.mode)=(r.registration_id,r.miner_hotkey,r.mode)
   CROSS JOIN LATERAL (
     SELECT coalesce(sum(rc.cost_microusd),0) miner_cost_microusd,
            count(*) FILTER (WHERE rc.cost_microusd IS NULL) unresolved,
            count(*) FILTER (WHERE c.provider='openrouter' AND rc.http_status=200 AND rc.cost_microusd>0) generator_calls
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts rc USING(job_id,request_id)
     WHERE c.job_id=j.job_id AND c.key_id IS NOT NULL AND c.provider IN ('openrouter','deepline')
   ) usage
   WHERE j.network='finney' AND j.netuid=71 AND j.batch_id=p_batch
 ) rows
$$;
ALTER FUNCTION public.collection_training_consents_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_training_consents_v1(text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.collection_training_consents_v1(text) TO lab_arena_service;

-- The requested slot limit lives in the frozen operator manifest (default 12).
-- Bound storage and RPC arrays at the same absolute ceiling as ordinary collections.
ALTER TABLE public.collection_batches DROP CONSTRAINT collection_batches_check;
ALTER TABLE public.collection_batches ADD CONSTRAINT collection_batches_check CHECK (
 jsonb_typeof(models)='array' AND
 CASE WHEN experiment_terms_hash IS NULL THEN jsonb_array_length(models)=6
 ELSE mode IN ('pilot','live') AND batch_id ~ '^xp-[0-9a-f]{64}$' AND experiment_terms_hash ~ '^sha256:[0-9a-f]{64}$'
      AND jsonb_array_length(models)=jsonb_array_length(scenario_ids)
      AND jsonb_array_length(models) BETWEEN 1 AND 4096 END
);
DO $migration$
DECLARE target record; original text; revised text;
BEGIN
 FOR target IN SELECT * FROM (VALUES
  ('public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)',
   'jsonb_array_length(p_scenarios) NOT BETWEEN 1 AND 12','jsonb_array_length(p_scenarios) NOT BETWEEN 1 AND 4096'),
  ('public.collection_correction_tariff_guard_v1()',
   'jsonb_array_length(d->''cases'') NOT BETWEEN 1 AND 12','jsonb_array_length(d->''cases'') NOT BETWEEN 1 AND 4096'),
  ('public.collection_correction_tariff_guard_v1()',
   'sha256:88d5b6886024e7aa5960c3d76586e2831d96650c071a348b145d19a7e155ba6c',
   'sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b')
 ) AS t(sig,before_text,after_text) LOOP
   SELECT pg_get_functiondef(oid) INTO STRICT original FROM pg_proc WHERE oid=target.sig::regprocedure
    AND prosecdef AND proowner='lab_arena_owner'::regrole;
   IF array_length(string_to_array(original,target.before_text),1)<>2 THEN
     RAISE EXCEPTION '744 unexpected predecessor for %',target.sig;
   END IF;
   revised:=replace(original,target.before_text,target.after_text);
   EXECUTE revised;
 END LOOP;
END $migration$;

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
 IF NOT coalesce((
       (p_consent#>>'{body,terms_sha256}'='sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b'
        AND p_consent->>'round_id'='finney:71:locked-collection:'||p_mode
        AND p_consent->>'scope'='lab_arena.collection.register.v1' AND p_consent->>'hotkey'=p_hotkey
        AND p_consent#>>'{body,daily_cap_microusd}'=p_cap::text)
       OR (p_consent#>>'{body,task_scope}'='synthetic_workflows'
           AND p_consent#>>'{body,terms_sha256}'='sha256:df00d01897792eca6eb2aa88eab4a16dcba253ef3c8998581a124dead92686ca')
       OR (p_consent#>>'{body,task_scope}'='synthetic_corrections' AND p_consent#>>'{body,terms_sha256}'=CASE p_mode
           WHEN 'pilot' THEN 'sha256:41588b51710c8218e5dce56ef12386709e4ebd2409ca76fe4427251425cc6119'
           WHEN 'live' THEN 'sha256:88d5b6886024e7aa5960c3d76586e2831d96650c071a348b145d19a7e155ba6c' END))
     AND p_consent#>>'{body,task_scope}' IN ('synthetic_workflows','synthetic_corrections')
     AND (p_mode='pilot' OR (p_mode='live' AND (p_consent#>>'{body,task_scope}'='synthetic_workflows'
          OR jsonb_array_length(public.collection_open_correction_quotes_v1())>0))),false)
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
DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.pre744_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre744_acl)
 THEN RAISE EXCEPTION 'unexpected schema privileges'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
