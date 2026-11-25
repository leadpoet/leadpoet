-- Count assignments against both the verified enrollment and operator ceiling.
-- A column default preserves all existing grants without interpreting historical JSON.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE quota_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
ALTER TABLE public.collection_registrations ADD COLUMN max_slots_per_utc_day integer NOT NULL DEFAULT 1
 CHECK(max_slots_per_utc_day BETWEEN 1 AND 6);
DO $migration$
DECLARE patch record; fn regprocedure; original text; meta jsonb;
BEGIN
 FOR patch IN SELECT * FROM (VALUES
 ('public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)',
  $old$ INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent)
 VALUES(p_id,p_hotkey,p_owner,p_mode,p_consent);$old$,
  $new$ IF (p_consent->'body') ? 'max_slots_per_utc_day' AND NOT coalesce(
     p_consent#>>'{body,terms_sha256}'='sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b'
     AND jsonb_typeof(p_consent#>'{body,max_slots_per_utc_day}')='number'
     AND p_consent#>>'{body,max_slots_per_utc_day}' ~ '^[1-6]$',false) THEN
    RETURN jsonb_build_object('error','collection_consent_invalid'); END IF;
 INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent,max_slots_per_utc_day)
 VALUES(p_id,p_hotkey,p_owner,p_mode,p_consent,coalesce((p_consent#>>'{body,max_slots_per_utc_day}')::integer,1));$new$),
 ('public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)',
  $old$) >= daily_limit$old$,$new$) >= least(daily_limit,r.max_slots_per_utc_day)$new$),
 ('public.collection_statement_v1(text,text)',
  $old$r.registration_id,r.created_at,r.revoked_at,$old$,
  $new$r.registration_id,r.created_at,r.revoked_at,r.max_slots_per_utc_day,
          least(r.max_slots_per_utc_day,(SELECT daily_slots_per_owner FROM public.collection_dispatch_settings
            WHERE singleton)) effective_max_slots_per_utc_day,$new$),
 ('public.collection_training_consents_v1(text)',
  $old$j.registration_id,j.mode,r.consent,$old$,$new$j.registration_id,j.mode,r.consent,r.max_slots_per_utc_day,$new$)
 ) patches(signature,before_text,after_text) LOOP
  fn:=patch.signature::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT original,meta FROM pg_proc p
   WHERE p.oid=fn AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
  IF cardinality(string_to_array(original,patch.before_text))<>2 THEN
    RAISE EXCEPTION '752 unexpected predecessor: %',fn; END IF;
  EXECUTE replace(original,patch.before_text,patch.after_text);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=fn) IS DISTINCT FROM meta THEN
    RAISE EXCEPTION '752 altered privileges or signature: %',fn; END IF;
 END LOOP;
 IF NOT (SELECT had_create FROM pg_temp.quota_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.quota_acl) THEN
    RAISE EXCEPTION 'schema ACL altered'; END IF;
END $migration$;
NOTIFY pgrst,'reload schema';
COMMIT;
