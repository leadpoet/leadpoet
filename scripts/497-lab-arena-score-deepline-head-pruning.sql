-- Exclude superseded score-call uncertainties before V3 sorts and checks
-- provider billing. The unchanged final latest-head guard still validates
-- every returned candidate; this only avoids work on rows it would reject.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $score_deepline_head_pruning$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)';
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
      TRUE, 's', ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'score Deepline head security shape changed';
  END IF;
  IF pg_catalog.md5(v_definition) = 'cef7c5fba9d18fb5126976f136104795' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) IS DISTINCT FROM '7feabbe2544e2e3418a611ed865bacd3' THEN
    RAISE EXCEPTION 'score Deepline head preimage changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition, $before$              AND uncertainty.provider = 'deepline'
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty
$before$, $after$              AND uncertainty.provider = 'deepline'
              AND NOT EXISTS (
                SELECT 1
                FROM public.lab_arena_ledger AS early_later
                WHERE early_later.call_identity = uncertainty.call_identity
                  AND early_later.entry_id > uncertainty.entry_id
              )
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty
$after$);
  IF pg_catalog.md5(v_updated) IS DISTINCT FROM 'cef7c5fba9d18fb5126976f136104795' THEN
    RAISE EXCEPTION 'score Deepline head target changed';
  END IF;
  -- CREATE OR REPLACE retains the function identity, owner, grants and settings.
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE))
       IS DISTINCT FROM 'cef7c5fba9d18fb5126976f136104795'
     OR v_security_after IS DISTINCT FROM v_security THEN
    RAISE EXCEPTION 'score Deepline head readback changed';
  END IF;
END;
$score_deepline_head_pruning$;

NOTIFY pgrst, 'reload schema';
COMMIT;
