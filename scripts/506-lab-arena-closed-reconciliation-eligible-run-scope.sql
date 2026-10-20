-- Materialize the existing terminal-score run predicate before candidate joins.
-- This avoids repeated per-uncertainty run lookups under underestimated closed
-- round selectivity. All helper, global-head, scope and cursor guards stay exact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $closed_reconcile_scope_506$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)';
  v_preimage CONSTANT TEXT := '58f74de8066a3c9b0300a66c32820f92a098d584a964574d886c830dbc7f2547';
  v_postimage CONSTANT TEXT := 'a65adc61115dbb1fa7258bffcadc96218cf889534f3de189b82134e2c5effaf0';
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
    RAISE EXCEPTION 'closed reconciliation scope security shape differs';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM v_preimage THEN
    RAISE EXCEPTION 'closed reconciliation scope preimage differs';
  END IF;
  -- The materialized set contains the same unique run IDs admitted by the
  -- original join. No round identity, helper predicate or latest-head rule moves.
  v_updated := pg_catalog.replace(v_definition,$before$  )
  SELECT COALESCE(($before$,
    $after$  ), eligible_runs AS MATERIALIZED (
    SELECT eligible_run.run_id
    FROM public.lab_arena_runs AS eligible_run
    WHERE eligible_run.kind = 'score'
       AND (
         (eligible_run.status = 'failed'
          AND eligible_run.terminal_cause IN (
            'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
            'stage_closed', 'judge_error', 'judge_timeout'
          ))
         OR (eligible_run.status = 'accepted'
             AND eligible_run.terminal_cause = 'accepted')
       )
  )
  SELECT COALESCE(($after$);
  v_updated := pg_catalog.replace(v_updated,$before$JOIN public.lab_arena_runs AS eligible_run
        ON eligible_run.run_id = uncertainty.run_id
       AND eligible_run.kind = 'score'
       AND (
         (eligible_run.status = 'failed'
          AND eligible_run.terminal_cause IN (
            'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
            'stage_closed', 'judge_error', 'judge_timeout'
          ))
         OR (eligible_run.status = 'accepted'
             AND eligible_run.terminal_cause = 'accepted')
       )$before$,
    $after$JOIN eligible_runs AS eligible_run
        ON eligible_run.run_id = uncertainty.run_id$after$);
  IF encode(extensions.digest(v_updated,'sha256'),'hex') IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'closed reconciliation scope postimage differs';
  END IF;
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_security_after
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_security_after IS DISTINCT FROM v_security
     OR encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_signature::REGPROCEDURE),'sha256'),'hex') IS DISTINCT FROM v_postimage THEN
    RAISE EXCEPTION 'closed reconciliation scope readback differs';
  END IF;
END;
$closed_reconcile_scope_506$;

NOTIFY pgrst,'reload schema';
COMMIT;
