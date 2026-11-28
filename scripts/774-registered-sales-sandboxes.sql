-- Existing paid jobs and signatures are immutable. New sandbox families need a reviewed
-- operator admission; the service role can read but cannot create catalog entries.
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE registry_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_registered_sandboxes (
 family text PRIMARY KEY CHECK (family ~ '^P[1-9][0-9]{0,3}$'),
 kind text NOT NULL CHECK (kind IN ('train','evaluation')),
 registry_sha256 text NOT NULL CHECK (registry_sha256 ~ '^sha256:[0-9a-f]{64}$')
);
ALTER TABLE public.collection_registered_sandboxes OWNER TO lab_arena_owner;
ALTER TABLE public.collection_registered_sandboxes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.collection_registered_sandboxes FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_sandboxes_immutable BEFORE UPDATE OR DELETE ON public.collection_registered_sandboxes
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
INSERT INTO public.collection_registered_sandboxes (family,kind,registry_sha256) VALUES
 ('P7','train','sha256:774bee8f2de1d989a77b2fc860b453311f43c8c3add5d88581a39522852e178c'),
 ('P8','train','sha256:a32e49cdaa71bd3abc761fe4531333ce55aa91068858ab0d36e44f32db03cdcc'),
 ('P9','train','sha256:dbd16488760b192b79e2762b9fa1f967c8b37a8a6d58d5928e0d40fb425a53d0'),
 ('P10','train','sha256:9d68e74d091899fc613453b7a941050421cc7994f34979e55442c8662c338936');

CREATE FUNCTION public.collection_registered_sandbox_v1(p_family text) RETURNS jsonb
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
  SELECT jsonb_build_object('kind',kind,'registry_sha256',registry_sha256)
  FROM public.collection_registered_sandboxes WHERE family=p_family
 $$;
ALTER FUNCTION public.collection_registered_sandbox_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_registered_sandbox_v1(text) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.collection_registered_sandbox_v1(text) TO lab_arena_service;

CREATE FUNCTION public.collection_trainable_scenario_v1(sid text) RETURNS boolean
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
  SELECT CASE WHEN sid ~ '^P[1-9][0-9]{0,3}-s[1-9][0-9]{0,7}$' THEN EXISTS (
   SELECT FROM public.collection_registered_sandboxes s WHERE s.kind='train' AND s.family=split_part(sid,'-s',1)
    AND split_part(sid,'-s',2)::int NOT BETWEEN 101 AND 299
    AND (split_part(sid,'-s',2)::int BETWEEN 10000000 AND 39999999
         OR (s.family IN ('P7','P8','P9','P10') AND split_part(sid,'-s',2)::int<10000000)))
   ELSE false END
 $$;
ALTER FUNCTION public.collection_trainable_scenario_v1(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_trainable_scenario_v1(text) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $migration$
DECLARE body text; attrs jsonb; f regprocedure:='public.collection_governed_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text,bigint,jsonb)'::regprocedure;
 plus text := $old$s.id ~ '^P(7|8|9|10)-s[1-9][0-9]{0,7}$'$old$;
 minus text := $old$s.id !~ '^P(7|8|9|10)-s[1-9][0-9]{0,7}$'$old$;
BEGIN
 SELECT pg_get_functiondef(f),to_jsonb(p)-'prosrc' INTO STRICT body,attrs FROM pg_proc p WHERE p.oid=f
  AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(body,plus))<>2 OR cardinality(string_to_array(body,minus))<>2
 THEN RAISE EXCEPTION '774: unexpected governed allocator predecessor'; END IF;
 EXECUTE replace(replace(body,plus,'public.collection_trainable_scenario_v1(s.id)'),minus,'NOT public.collection_trainable_scenario_v1(s.id)');
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=f) IS DISTINCT FROM attrs
 THEN RAISE EXCEPTION '774: allocator authority changed'; END IF;
 IF NOT (SELECT had_create FROM pg_temp.registry_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.registry_acl)
 THEN RAISE EXCEPTION '774: public schema permissions changed'; END IF;
END $migration$;
NOTIFY pgrst,'reload schema';
COMMIT;
