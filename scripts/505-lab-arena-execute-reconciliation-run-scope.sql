-- Scope V2 billing reads through existing run indexes before head validation.
-- The final run identity, successful-call, latest-head and closed-round guards
-- remain authoritative. No cost, lease, cursor, accepted output or row changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $execute_reconcile_scope_505$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)';
  v_preimage CONSTANT TEXT := '6267d70170c18f3303b6923f2bf5b68deb63eb79e1ee51ec244c3ba1030b43ed';
  v_postimage CONSTANT TEXT := '5b6056afa13230ba797a552b99644583a9345544cb8304422f5e671746e78ba6';
  v_definition TEXT;
  v_updated TEXT;
  v_security JSONB;
  v_security_after JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_security
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_security IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'execute reconciliation scope security shape differs';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM v_preimage THEN
    RAISE EXCEPTION 'execute reconciliation scope preimage differs';
  END IF;
  -- The final join already requires this exact run and round. In the true
  -- execute-only lane its existing kind guard also implies the early filter.
  -- False and null flags retain their original final eligibility behavior.
  v_updated := pg_catalog.replace(v_definition,$before$FROM public.lab_arena_ledger AS early
            WHERE early.round_id = p_round_id$before$,
    $after$FROM public.lab_arena_runs AS scope
            CROSS JOIN LATERAL (
              SELECT scoped.* FROM public.lab_arena_ledger AS scoped
              WHERE scoped.run_id = scope.run_id
              OFFSET 0
            ) AS early
            WHERE scope.round_id = p_round_id
              AND (p_successful_execute_only IS DISTINCT FROM TRUE
                OR scope.kind = 'execute')
              AND early.round_id = p_round_id$after$);
  IF encode(extensions.digest(v_updated,'sha256'),'hex') IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'execute reconciliation scope postimage differs';
  END IF;
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_security_after IS DISTINCT FROM v_security
     OR encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_signature::REGPROCEDURE),'sha256'),'hex') IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'execute reconciliation scope readback differs';
  END IF;
END;
$execute_reconcile_scope_505$;

NOTIFY pgrst,'reload schema';
COMMIT;
