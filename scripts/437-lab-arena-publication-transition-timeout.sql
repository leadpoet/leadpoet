-- The complete publication guard validates every frozen ranking row in one
-- transition. Oct9's 143-row round exceeded its 60-second database budget:
-- all 92 completed-model guards passed separately, but their aggregate alone
-- was canceled with SQLSTATE 57014 at 60 seconds. Keep every guard unchanged.
-- Give the transition RPC a bounded ten-minute window for the configured
-- 256 challengers plus baseline. This is headroom, not an unlimited capacity
-- guarantee; the dedicated publication transport deadline is 605 seconds.
-- PostgREST hoists this function setting before the RPC statement.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $publication_transition_timeout_437$
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
    RAISE EXCEPTION 'lab_arena_transition_timeout_437_function_missing';
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
    RAISE EXCEPTION 'lab_arena_transition_timeout_437_security_shape_changed';
  END IF;
  IF v_definition_hash = '72ec49013e7796d4e4cba4ee8943884a'
     AND v_config = ARRAY[
       'search_path=pg_catalog, public', 'statement_timeout=600s'
     ] THEN
    RETURN;
  END IF;
  IF v_definition_hash <> 'ee67810d80d4e3d9016e3d6e6eab2a57'
     OR v_config IS DISTINCT FROM ARRAY[
       'search_path=pg_catalog, public', 'statement_timeout=60s'
     ] THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_437_preimage_changed';
  END IF;

  ALTER FUNCTION public.lab_arena_transition_round(TEXT,TEXT,TEXT,JSONB)
    SET statement_timeout = '600s';

  SELECT pg_catalog.md5(pg_catalog.pg_get_functiondef(procedure.oid)),
         owner.rolname, procedure.proacl::TEXT, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
    INTO v_definition_hash, v_owner, v_acl, v_security_definer,
         v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE procedure.oid = v_oid;
  IF v_definition_hash IS DISTINCT FROM '72ec49013e7796d4e4cba4ee8943884a'
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS DISTINCT FROM ARRAY[
       'search_path=pg_catalog, public', 'statement_timeout=600s'
     ] THEN
    RAISE EXCEPTION 'lab_arena_transition_timeout_437_readback_changed';
  END IF;
END;
$publication_transition_timeout_437$;

NOTIFY pgrst, 'reload schema';
COMMIT;
