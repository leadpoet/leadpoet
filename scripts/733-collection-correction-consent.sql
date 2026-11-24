-- The gateway publishes the reviewed correction terms by digest. A signed
-- correction enrollment has its own scope and cannot authorize another provider.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
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
        AND p_consent->'body'->>'terms_sha256'='sha256:41588b51710c8218e5dce56ef12386709e4ebd2409ca76fe4427251425cc6119'),false)
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

COMMIT;
