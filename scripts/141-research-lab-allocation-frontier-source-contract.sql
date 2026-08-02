-- Expose the historical allocation-source compatibility required by restart.
-- Migration 140 patches the persistence RPC; this migration makes that exact
-- capability discoverable through the read-only production schema preflight.

BEGIN;

DO $migration$
DECLARE
    function_signature CONSTANT REGPROCEDURE :=
        'public.persist_research_lab_allocation_frontier_bootstrap_v2(jsonb,text,text)'::REGPROCEDURE;
    function_definition TEXT;
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

    IF pg_catalog.strpos(function_definition, compatible_guard) = 0 THEN
        RAISE EXCEPTION
            'allocation_frontier_historical_source_contract_missing'
            USING ERRCODE = '23514';
    END IF;
END;
$migration$;

CREATE OR REPLACE FUNCTION
public.research_lab_allocation_frontier_historical_source_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.allocation_frontier_historical_source_contract.v1',
        'persistence_rpc',
        'persist_research_lab_allocation_frontier_bootstrap_v2',
        'settlement_frontier_compatibility',
        'missing_or_null'
    );
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_allocation_frontier_historical_source_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_allocation_frontier_historical_source_contract_v1()
    TO service_role;

COMMENT ON FUNCTION
    public.research_lab_allocation_frontier_historical_source_contract_v1()
IS 'Read-only capability marker proving the frontier bootstrap accepts historical source states whose settlement_frontier is missing or JSON null.';

NOTIFY pgrst, 'reload schema';

COMMIT;
