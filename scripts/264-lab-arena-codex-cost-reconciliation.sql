-- Extend the existing exact-generation recovery to Codex Responses calls.
-- No ledger data, ownership, grants, signatures or recovery rules change.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $arena_codex_cost_reconciliation$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
  v_index INTEGER;
  v_old_parts TEXT[];
  v_new_parts TEXT[];
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text)'
  ] LOOP
    IF pg_catalog.to_regprocedure(v_signature) IS NULL THEN
      RAISE EXCEPTION 'apply 225-lab-arena-openrouter-delayed-cost-reconciliation.sql first';
    END IF;
    SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(v_signature))
      INTO v_definition;
    IF v_signature LIKE 'public.lab_arena_list_%' THEN
      v_old_parts := ARRAY[$old$uncertainty.operation_id = 'openrouter.chat'$old$];
      v_new_parts := ARRAY[$new$uncertainty.operation_id IN ('openrouter.chat', 'openrouter.responses')$new$];
    ELSE
      v_old_parts := ARRAY[
        $old$v_head.operation_id IS DISTINCT FROM 'openrouter.chat'$old$,
        $old$v_reservation.operation_id IS DISTINCT FROM 'openrouter.chat'$old$,
        $old$'operation', 'openrouter.chat'$old$
      ];
      v_new_parts := ARRAY[
        $new$COALESCE(v_head.operation_id, '') NOT IN ('openrouter.chat', 'openrouter.responses')$new$,
        $new$v_reservation.operation_id IS DISTINCT FROM v_head.operation_id$new$,
        $new$'operation', v_reservation.operation_id$new$
      ];
    END IF;
    FOR v_index IN 1..pg_catalog.array_length(v_old_parts, 1) LOOP
      v_old := v_old_parts[v_index];
      v_new := v_new_parts[v_index];
      IF pg_catalog.strpos(v_definition, v_old) > 0 THEN
        IF (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
           / pg_catalog.length(v_old) <> 1 THEN
          RAISE EXCEPTION 'unexpected recovery function shape: %', v_signature;
        END IF;
        v_definition := pg_catalog.replace(v_definition, v_old, v_new);
      ELSIF pg_catalog.strpos(v_definition, v_new) = 0 THEN
        RAISE EXCEPTION 'unexpected recovery function shape: %', v_signature;
      END IF;
    END LOOP;
    -- CREATE OR REPLACE retains the existing owner and service-only ACL.
    EXECUTE v_definition;
  END LOOP;
END;
$arena_codex_cost_reconciliation$;

COMMIT;
