-- Prune superseded V2 Deepline heads before full billing eligibility work.
-- Keep the final latest-head guard, ordering, cursor, and successful-call policy.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $successful_deepline_head_pruning$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)';
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
    RAISE EXCEPTION 'successful Deepline head security shape changed';
  END IF;
  IF pg_catalog.md5(v_definition) = 'ea4cd7bc3fd18d3861c6a05addfcb69e' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) IS DISTINCT FROM '82531576cffccfea2937eb215d47c89a' THEN
    RAISE EXCEPTION 'successful Deepline head preimage changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition, $before$FROM public.lab_arena_ledger AS uncertainty
          JOIN public.lab_arena_ledger AS reservation$before$, $after$FROM (
            SELECT early.*
            FROM public.lab_arena_ledger AS early
            WHERE early.round_id = p_round_id
              AND (COALESCE(p_run_id, '') = '' OR early.run_id = p_run_id)
              AND early.entry_kind = 'uncertain'
              AND early.provider = 'deepline'
              AND NOT EXISTS (
                SELECT 1
                FROM public.lab_arena_ledger AS early_later
                WHERE early_later.call_identity = early.call_identity
                  AND early_later.entry_id > early.entry_id
              )
            OFFSET 0
          ) AS uncertainty
          JOIN public.lab_arena_ledger AS reservation$after$);
  IF pg_catalog.md5(v_updated) IS DISTINCT FROM 'ea4cd7bc3fd18d3861c6a05addfcb69e' THEN
    RAISE EXCEPTION 'successful Deepline head target changed';
  END IF;
  -- CREATE OR REPLACE retains function identity, owner, grants and settings.
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE))
       IS DISTINCT FROM 'ea4cd7bc3fd18d3861c6a05addfcb69e'
     OR v_security_after IS DISTINCT FROM v_security THEN
    RAISE EXCEPTION 'successful Deepline head readback changed';
  END IF;
END;
$successful_deepline_head_pruning$;

NOTIFY pgrst, 'reload schema';
COMMIT;
