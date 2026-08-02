-- Accept the historical allocation-source shape when bootstrapping the first
-- bounded settlement frontier. Older signed allocation source states omitted
-- settlement_frontier; current states encode the same absence as JSON null.

BEGIN;

DO $migration$
DECLARE
    function_signature CONSTANT REGPROCEDURE :=
        'public.persist_research_lab_allocation_frontier_bootstrap_v2(jsonb,text,text)'::REGPROCEDURE;
    function_definition TEXT;
    patched_definition TEXT;
    legacy_guard CONSTANT TEXT := $legacy_guard$
       OR allocation_execution.result_doc->'source_state'->
          'settlement_frontier' IS DISTINCT FROM 'null'::JSONB
$legacy_guard$;
    compatible_guard CONSTANT TEXT := $compatible_guard$
       OR COALESCE(
          allocation_execution.result_doc->'source_state'->
          'settlement_frontier',
          'null'::JSONB
       ) IS DISTINCT FROM 'null'::JSONB
$compatible_guard$;
BEGIN
    SELECT pg_catalog.pg_get_functiondef(function_signature::OID)
      INTO STRICT function_definition;

    IF pg_catalog.position(compatible_guard IN function_definition) > 0 THEN
        RETURN;
    END IF;
    IF pg_catalog.position(legacy_guard IN function_definition) = 0 THEN
        RAISE EXCEPTION
            'allocation_frontier_bootstrap_historical_source_guard_missing'
            USING ERRCODE = '23514';
    END IF;

    patched_definition := pg_catalog.replace(
        function_definition,
        legacy_guard,
        compatible_guard
    );
    IF patched_definition = function_definition
       OR pg_catalog.position(legacy_guard IN patched_definition) > 0
       OR pg_catalog.position(compatible_guard IN patched_definition) = 0 THEN
        RAISE EXCEPTION
            'allocation_frontier_bootstrap_historical_source_patch_invalid'
            USING ERRCODE = '23514';
    END IF;

    EXECUTE patched_definition;
END;
$migration$;

NOTIFY pgrst, 'reload schema';

COMMIT;
