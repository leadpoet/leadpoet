-- Extend the existing exact late judge-billing recovery to per-ICP rounds.
-- This changes the audit ledger only, never a published score or research cost.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $per_icp_closed_billing$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_old TEXT := '= ''successful_calls_v1''';
  v_new TEXT := 'IN (''successful_calls_v1'', ''successful_calls_per_icp_v1'')';
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)',
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_signature::pg_catalog.regprocedure)
      INTO v_definition;
    IF pg_catalog.strpos(v_definition, v_new) > 0 THEN
      CONTINUE;
    END IF;
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR (pg_catalog.length(v_definition)
           - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
          / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'per-ICP closed billing policy shape unexpected: %', v_signature;
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END LOOP;
END;
$per_icp_closed_billing$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
