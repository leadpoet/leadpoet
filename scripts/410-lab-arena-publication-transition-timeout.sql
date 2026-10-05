-- The complete publication guard validates every frozen ranking row in one
-- transition. Give that existing RPC a bounded 60-second database window.
-- This is a function setting; PostgREST hoists it before the RPC statement.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $publication_transition_timeout_410$
DECLARE
  v_oid REGPROCEDURE := pg_catalog.to_regprocedure(
    'public.lab_arena_transition_round(text,text,text,jsonb)'
  );
  v_definition_hash TEXT;
  v_owner TEXT;
  v_acl TEXT;
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
BEGIN
  IF v_oid IS NULL THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_410_function_missing';
  END IF;
  SELECT pg_catalog.md5(pg_catalog.pg_get_functiondef(procedure.oid)),
         owner.rolname, procedure.proacl::TEXT, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
    INTO v_definition_hash, v_owner, v_acl, v_security_definer,
         v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE procedure.oid = v_oid;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v' THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_410_security_shape_changed';
  END IF;
  IF v_definition_hash = '02ed5004801fb47066f06623ae5abc7d'
     AND v_config = ARRAY[
       'search_path=pg_catalog, public', 'statement_timeout=60s'
     ] THEN
    RETURN;
  END IF;
  IF v_definition_hash <> '8cd26a7f0b9737160cce98438013dc91'
     OR v_config IS DISTINCT FROM ARRAY['search_path=pg_catalog, public'] THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_410_preimage_changed';
  END IF;

  ALTER FUNCTION public.lab_arena_transition_round(TEXT,TEXT,TEXT,JSONB)
    SET statement_timeout = '60s';

  SELECT pg_catalog.md5(pg_catalog.pg_get_functiondef(procedure.oid)),
         owner.rolname, procedure.proacl::TEXT, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
    INTO v_definition_hash, v_owner, v_acl, v_security_definer,
         v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE procedure.oid = v_oid;
  IF v_definition_hash IS DISTINCT FROM '02ed5004801fb47066f06623ae5abc7d'
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS DISTINCT FROM ARRAY[
       'search_path=pg_catalog, public', 'statement_timeout=60s'
     ] THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_410_readback_changed';
  END IF;
END;
$publication_transition_timeout_410$;

NOTIFY pgrst, 'reload schema';
COMMIT;
