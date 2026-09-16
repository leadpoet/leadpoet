-- Align the September 16 rerun publication guard with successful_calls_v1.
-- Failed provider calls may retain an uncertain liability without making the
-- successful-call cost state unresolved. Inflight calls and calls whose
-- success remains unresolved must still block publication.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $sep16_failed_cost_publication$
DECLARE
  v_definition TEXT;
  v_fragment TEXT;
  v_required TEXT;
  v_changed BOOLEAN := FALSE;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_sep16_rerun_publication_guard_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__successful_call_cost_state(text,text,text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migrations 229 and 265 before 267';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(
           pg_catalog.to_regprocedure(
             'public.lab_arena_sep16_rerun_publication_guard_v1()'
           )
         )
    INTO STRICT v_definition;

  FOREACH v_fragment IN ARRAY ARRAY[
    $check$OR (v_execute_cost ->> 'uncertain_calls')::BIGINT <> 0$check$,
    $check$OR (v_score_cost ->> 'uncertain_calls')::BIGINT <> 0$check$
  ] LOOP
    IF pg_catalog.strpos(v_definition, v_fragment) > 0 THEN
      IF (
        pg_catalog.length(v_definition)
        - pg_catalog.length(pg_catalog.replace(v_definition, v_fragment, ''))
      ) / pg_catalog.length(v_fragment) <> 1 THEN
        RAISE EXCEPTION 'unexpected September 16 publication guard shape';
      END IF;
      v_definition := pg_catalog.replace(v_definition, v_fragment, '');
      v_changed := TRUE;
    END IF;
  END LOOP;

  FOREACH v_required IN ARRAY ARRAY[
    $check$OR (v_execute_cost ->> 'inflight_calls')::BIGINT <> 0$check$,
    $check$OR (v_execute_cost ->> 'success_unresolved_calls')::BIGINT <> 0$check$,
    $check$OR (v_score_cost ->> 'inflight_calls')::BIGINT <> 0$check$,
    $check$OR (v_score_cost ->> 'success_unresolved_calls')::BIGINT <> 0$check$
  ] LOOP
    IF pg_catalog.strpos(v_definition, v_required) = 0 THEN
      RAISE EXCEPTION 'September 16 publication cost guard is incomplete';
    END IF;
  END LOOP;

  IF v_changed THEN
    -- CREATE OR REPLACE retains the existing owner, ACL, and trigger binding.
    EXECUTE v_definition;
  END IF;

  SELECT pg_catalog.pg_get_functiondef(
           pg_catalog.to_regprocedure(
             'public.lab_arena_sep16_rerun_publication_guard_v1()'
           )
         )
    INTO STRICT v_definition;
  IF pg_catalog.strpos(
       v_definition,
       $check$OR (v_execute_cost ->> 'uncertain_calls')::BIGINT <> 0$check$
     ) > 0
     OR pg_catalog.strpos(
       v_definition,
       $check$OR (v_score_cost ->> 'uncertain_calls')::BIGINT <> 0$check$
     ) > 0 THEN
    RAISE EXCEPTION 'September 16 failed-call uncertainty guard remains';
  END IF;
END;
$sep16_failed_cost_publication$;

COMMIT;
