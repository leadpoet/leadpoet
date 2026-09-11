-- An unknown provider charge is never cost-eligible, even when its conservative
-- reservation lands exactly on the execution or per-company boundary.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_uncertain_cost_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
      IF v_inflight > 0 OR v_score_inflight > 0 THEN
        v_expected_reason := 'provider_calls_inflight';
      ELSIF v_conservative > v_execution_cap THEN
$old$;
  v_new TEXT := $new$
      IF v_inflight > 0 OR v_score_inflight > 0 THEN
        v_expected_reason := 'provider_calls_inflight';
      ELSIF EXISTS (
        SELECT 1
        FROM (
          SELECT DISTINCT ON (ledger.call_identity)
            ledger.entry_kind
          FROM public.lab_arena_ledger AS ledger
          JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
          WHERE ledger.submission_id = v_submission_id
            AND runs.kind IN ('execute', 'score')
            AND ledger.call_identity IS NOT NULL
          ORDER BY ledger.call_identity, ledger.entry_id DESC
        ) AS uncertain_head
        WHERE uncertain_head.entry_kind = 'uncertain'
      ) THEN
        v_expected_reason := 'provider_cost_uncertain';
      ELSIF v_conservative > v_execution_cap THEN
$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_publication_baseline_guard_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply 206-lab-arena-combined-provider-budget.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_publication_baseline_guard_v1'
    AND procedure.pronargs = 0;
  IF pg_catalog.strpos(v_definition, 'provider_cost_uncertain') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_publication_guard_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$lab_arena_uncertain_cost_guard$;

NOTIFY pgrst, 'reload schema';
COMMIT;
