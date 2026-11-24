-- The P7 training task uses the same local tools, funding contract and payout checks.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = pg_catalog;

CREATE TEMP TABLE capture740_schema ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $migration$
DECLARE
 old_pattern constant text := '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$';
 new_pattern text := replace(old_pattern, 'P[1-6]', 'P[1-7]');
 target record; object_id oid; original text; source_hash text; metadata jsonb; updated jsonb;
BEGIN
 FOR target IN SELECT * FROM (VALUES
  ('public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)', '87a212a85decaff59f08a3d5bf225726787e82bd2a259e44551f0abcd865ceca', 'b1322862871cc3022febfe4aee0c2ff56b1ab71e34f6267aa2073500ff6063e1'),
  ('public.collection_correction_tariff_guard_v1()', 'e79255aa4481b9a9be9cd944084e6ac83e06769e80170d80e06a9661e4d0aeaa', 'c0f4c546ce3d6ab881a90c534aceec5acb48ed62d19f5771c192a6fd01b0184e')
 ) t(signature, previous_sha256, new_sha256) LOOP
  object_id := target.signature::regprocedure;
  SELECT pg_get_functiondef(p.oid),encode(sha256(convert_to(p.prosrc,'UTF8')),'hex'),to_jsonb(p)-'prosrc'
  INTO STRICT original,source_hash,metadata FROM pg_proc p WHERE p.oid=object_id;
  IF source_hash <> target.previous_sha256 OR strpos(original,old_pattern)=0
    OR NOT (metadata->>'prosecdef')::boolean OR (metadata->>'proowner')::oid <> 'lab_arena_owner'::regrole::oid
  THEN RAISE EXCEPTION '740 unexpected predecessor for %',target.signature; END IF;
  EXECUTE replace(original,old_pattern,new_pattern);
  SELECT encode(sha256(convert_to(p.prosrc,'UTF8')),'hex'),to_jsonb(p)-'prosrc' INTO STRICT source_hash,updated
  FROM pg_proc p WHERE p.oid=object_id;
  IF source_hash<>target.new_sha256 OR updated IS DISTINCT FROM metadata
  THEN RAISE EXCEPTION '740 unexpected function or privileges for %',target.signature; END IF;
 END LOOP;

 SELECT pg_get_constraintdef(oid) INTO STRICT original FROM pg_constraint
 WHERE conrelid='public.collection_jobs'::regclass AND conname='collection_synthetic_key_scope' AND contype='c' AND convalidated;
 IF original <> format('CHECK (((deepline_key_id IS NOT NULL) OR (scenario_id ~ %L::text)))',old_pattern)
 THEN RAISE EXCEPTION '740 unexpected synthetic job constraint'; END IF;
 ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_synthetic_key_scope;
 EXECUTE 'ALTER TABLE public.collection_jobs ADD CONSTRAINT collection_synthetic_key_scope ' || replace(original,old_pattern,new_pattern);
 IF NOT (SELECT had_create FROM pg_temp.capture740_schema) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
    (SELECT nspacl FROM pg_temp.capture740_schema)
 THEN RAISE EXCEPTION '740 unexpected schema ACL'; END IF;
END;
$migration$;
NOTIFY pgrst, 'reload schema';
COMMIT;
