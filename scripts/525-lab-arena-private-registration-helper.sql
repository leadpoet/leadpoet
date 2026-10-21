-- Only the service-only SECURITY DEFINER wrappers may call this internal helper.
-- Preserve its source, the wrappers, submission history and shared default ACLs.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $private_registration_helper$
DECLARE
  target OID := 'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'::regprocedure;
  definition TEXT;
  prior_acl TEXT;
  metadata JSONB;
  wrappers JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid),p.proacl::TEXT,to_jsonb(p)-'proacl'
    INTO STRICT definition,prior_acl,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.prosecdef AND p.provolatile='v' AND l.lanname='plpgsql'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[];
  IF encode(extensions.digest(definition,'sha256'),'hex')<>
      'ea517cb1876cccae0e92c601d68a34a8ff90ecf4409080f739b196c4974b1a92' THEN
    RAISE EXCEPTION 'private_registration_helper source differs' USING ERRCODE='55000';
  END IF;
  SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) INTO wrappers
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid IN (
      'public.lab_arena_register_submission(text,text,text,jsonb)'::regprocedure,
      'public.lab_arena_register_submission_v2(text,text,text,jsonb,text,bigint,text)'::regprocedure)
    AND pg_get_userbyid(p.proowner)='lab_arena_owner' AND p.prosecdef
    AND p.provolatile='v' AND l.lanname='sql'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex') IN (
      '631cdf9e46c425bf0a5b25e76cc2f7b0961ac265eb451dca341ddb44d29be8a1',
      '2baa48c2c6fb8bb42552ce18ab3636e60c8d006b3e07c280c84755e73ec12d09');
  IF jsonb_array_length(wrappers) IS DISTINCT FROM 2 THEN
    RAISE EXCEPTION 'private_registration_helper wrapper differs' USING ERRCODE='55000';
  END IF;
  IF prior_acl='{lab_arena_owner=X/lab_arena_owner}' THEN RETURN; END IF;
  IF prior_acl IS DISTINCT FROM
      '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}' THEN
    RAISE EXCEPTION 'private_registration_helper ACL differs' USING ERRCODE='55000';
  END IF;
  REVOKE ALL ON FUNCTION public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)
    FROM PUBLIC,anon,authenticated,service_role;
  IF (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=target)
       IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
     OR (SELECT to_jsonb(p)-'proacl' FROM pg_proc p WHERE p.oid=target)
       IS DISTINCT FROM metadata
     OR (SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) FROM pg_proc p
       WHERE p.oid IN (
         'public.lab_arena_register_submission(text,text,text,jsonb)'::regprocedure,
         'public.lab_arena_register_submission_v2(text,text,text,jsonb,text,bigint,text)'::regprocedure))
       IS DISTINCT FROM wrappers THEN
    RAISE EXCEPTION 'private_registration_helper readback differs' USING ERRCODE='55000';
  END IF;
END;
$private_registration_helper$;
COMMIT;
