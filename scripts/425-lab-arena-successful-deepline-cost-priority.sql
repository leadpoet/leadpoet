-- Give successful sourcing receipts a separate bounded reconciliation cursor.
-- Keep the ordinary V1 list and all settlement/accounting functions unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $successful_deepline_candidates$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$            AND (COALESCE(p_run_id, '') = '' OR uncertainty.run_id = p_run_id)$anchor$;
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_deepline_call_responses') IS NULL THEN
    RAISE EXCEPTION 'apply 417-lab-arena-deepline-response-recovery.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'deepline_execution_key') = 0
     OR (pg_catalog.length(v_definition) - pg_catalog.length(
       pg_catalog.replace(v_definition, v_anchor, ''))) / pg_catalog.length(v_anchor) <> 1 THEN
    RAISE EXCEPTION 'Deepline candidate list preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v1(',
    'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v2(');
  v_definition := pg_catalog.replace(v_definition, 'p_limit integer)',
    'p_limit integer, p_successful_execute_only boolean)');
  -- On an uncertain ledger head, cost eligibility recognizes a typed successful
  -- worker result or the existing immutable recovered successful response.
  -- Every original identity, dispatch, latest-head and closed-round guard stays.
  v_definition := pg_catalog.replace(v_definition, v_anchor, v_anchor || $filter$
            AND (p_successful_execute_only IS FALSE OR (
              p_successful_execute_only IS TRUE
              AND runs.kind = 'execute'
              AND (
                uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
                OR EXISTS (
                  SELECT 1 FROM public.lab_arena_deepline_call_responses AS response
                  WHERE response.call_identity = uncertainty.call_identity
                    AND response.run_id = uncertainty.run_id
                    AND response.terminal_response -> 'call_succeeded' = 'true'::JSONB
                )
              )
            ))$filter$);
  EXECUTE v_definition;
END;
$successful_deepline_candidates$;

ALTER FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v2(
  TEXT, TEXT, BIGINT, INTEGER, BOOLEAN
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v2(
  TEXT, TEXT, BIGINT, INTEGER, BOOLEAN
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v2(
  TEXT, TEXT, BIGINT, INTEGER, BOOLEAN
) TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
