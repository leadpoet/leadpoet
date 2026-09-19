-- Admit the existing bounded OpenRouter Responses hosted-search route to the
-- atomic remaining-budget reservation. All other dynamic reservation shapes
-- stay rejected before lease lookup or ledger mutation.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $openrouter_web_search_reservation$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$       OR p_provider <> 'deepline'
       OR p_amount_microusd <> 0 THEN$old$;
  v_new TEXT := $new$       -- lab_arena_openrouter_web_search_reservation
       OR NOT (
         (
           p_provider IS NOT DISTINCT FROM 'deepline'
           AND p_amount_microusd = 0
         )
         OR (
           p_provider IS NOT DISTINCT FROM 'openrouter'
           AND p_operation_id IS NOT DISTINCT FROM 'openrouter.responses'
           AND p_amount_microusd > 0
         )
       ) THEN$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_openrouter_web_search_reservation'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'OpenRouter web-search reservation shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$openrouter_web_search_reservation$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
