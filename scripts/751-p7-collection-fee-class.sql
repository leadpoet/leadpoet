-- P7 is the training-only platform workflow admitted by migration 740.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE pre751_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $migration$
DECLARE
 fn regprocedure := 'public.collection_fee_class_v1(text)'::regprocedure;
 src text; attrs jsonb; new_attrs jsonb; body_hash text;
BEGIN
 SELECT pg_get_functiondef(oid),to_jsonb(p)-'prosrc',encode(sha256(convert_to(prosrc,'UTF8')),'hex')
   INTO STRICT src,attrs,body_hash FROM pg_proc p WHERE oid=fn AND proowner='lab_arena_owner'::regrole;
 IF body_hash<>'3056561fec810d78ff542cf2c930cb38687ad0318afe4980fa2fb90f1c92ab41' THEN RAISE EXCEPTION 'unexpected collection fee class function'; END IF;
 EXECUTE replace(src,'P[1-6]','P[1-7]');
 SELECT to_jsonb(p)-'prosrc',encode(sha256(convert_to(prosrc,'UTF8')),'hex')
   INTO STRICT new_attrs,body_hash FROM pg_proc p WHERE oid=fn;
 IF new_attrs IS DISTINCT FROM attrs OR body_hash<>'d023d60a70230ffa4b92a4efc3b5f4311b704feaacdb68c81fbc465c90d1e2ec' THEN
   RAISE EXCEPTION 'incorrect collection fee class function or attributes';
 END IF;
 IF NOT (SELECT had_create FROM pg_temp.pre751_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre751_acl)
 THEN RAISE EXCEPTION 'unexpected schema privileges'; END IF;
END $migration$;
COMMIT;
