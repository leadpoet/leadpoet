-- Execution and scoring stage creation insert all assignments atomically,
-- including up to 256 miners across ten ICPs. Give only those two existing
-- RPCs a bounded 60-second database window.
-- This is a function setting; PostgREST hoists it before the RPC statement.
-- ALTER FUNCTION preserves the installed body, including historical recovery
-- branches. Check that body byte-for-byte instead of replacing or pinning it.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $stage_creation_timeout_427$
DECLARE
  v_signature TEXT;
  v_oid REGPROCEDURE;
  v_body TEXT;
  v_original_body TEXT;
  v_owner TEXT;
  v_acl TEXT;
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_open_stage(text,smallint,jsonb,integer[])',
    'public.lab_arena_open_scoring_v3(text,smallint,jsonb)'
  ]
  LOOP
    v_oid := pg_catalog.to_regprocedure(v_signature);
    IF v_oid IS NULL THEN
      RAISE EXCEPTION 'lab_arena_stage_creation_timeout_427_function_missing';
    END IF;
    SELECT procedure.prosrc,
           owner.rolname, procedure.proacl::TEXT, procedure.prosecdef,
           procedure.provolatile, procedure.proconfig
      INTO v_body, v_owner, v_acl, v_security_definer,
           v_volatility, v_config
    FROM pg_catalog.pg_proc AS procedure
    JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
    WHERE procedure.oid = v_oid;
    IF v_owner IS DISTINCT FROM 'lab_arena_owner'
       OR v_acl IS DISTINCT FROM
          '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
       OR v_security_definer IS DISTINCT FROM TRUE
       OR v_volatility IS DISTINCT FROM 'v' THEN
      RAISE EXCEPTION 'lab_arena_stage_creation_timeout_427_security_shape_changed';
    END IF;
    IF v_config = ARRAY[
         'search_path=pg_catalog, public', 'statement_timeout=60s'
       ] THEN
      CONTINUE;
    END IF;
    IF v_config IS DISTINCT FROM ARRAY['search_path=pg_catalog, public'] THEN
      RAISE EXCEPTION 'lab_arena_stage_creation_timeout_427_configuration_changed';
    END IF;

    v_original_body := v_body;
    EXECUTE pg_catalog.format(
      'ALTER FUNCTION %s SET statement_timeout = %L',
      v_signature, '60s'
    );

    SELECT procedure.prosrc,
           owner.rolname, procedure.proacl::TEXT, procedure.prosecdef,
           procedure.provolatile, procedure.proconfig
      INTO v_body, v_owner, v_acl, v_security_definer,
           v_volatility, v_config
    FROM pg_catalog.pg_proc AS procedure
    JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
    WHERE procedure.oid = v_oid;
    IF v_body IS DISTINCT FROM v_original_body
       OR v_owner IS DISTINCT FROM 'lab_arena_owner'
       OR v_acl IS DISTINCT FROM
          '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
       OR v_security_definer IS DISTINCT FROM TRUE
       OR v_volatility IS DISTINCT FROM 'v'
       OR v_config IS DISTINCT FROM ARRAY[
         'search_path=pg_catalog, public', 'statement_timeout=60s'
       ] THEN
      RAISE EXCEPTION 'lab_arena_stage_creation_timeout_427_readback_changed';
    END IF;
  END LOOP;
END;
$stage_creation_timeout_427$;

NOTIFY pgrst, 'reload schema';
COMMIT;
