-- Repair the private benchmark schema-version constraint left behind by the
-- original unnamed PostgreSQL check. Migration 97 added 1.1 support under a
-- different truncated identifier, so both checks could coexist and the old
-- schema_version = '1.0' check continued rejecting conditional baselines.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS research_lab_private_model_benchmark_bundl_schema_version_check;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS research_lab_private_model_benchmark_bundles_schema_version_che;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS rl_private_benchmark_schema_version_check;

ALTER TABLE public.research_lab_private_model_benchmark_bundles
    ADD CONSTRAINT rl_private_benchmark_schema_version_check
    CHECK (schema_version IN ('1.0', '1.1')) NOT VALID;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    VALIDATE CONSTRAINT rl_private_benchmark_schema_version_check;

CREATE OR REPLACE FUNCTION public.research_lab_private_benchmark_schema_contract_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    target_relation OID :=
        'public.research_lab_private_model_benchmark_bundles'::REGCLASS;
    schema_constraint_count INTEGER;
    legacy_constraint_count INTEGER;
    contract_constraint_valid BOOLEAN;
BEGIN
    SELECT COUNT(*)
      INTO schema_constraint_count
      FROM pg_catalog.pg_constraint AS constraint_row
     WHERE constraint_row.conrelid = target_relation
       AND constraint_row.contype = 'c'
       AND POSITION(
               'schema_version'
               IN pg_catalog.pg_get_expr(
                   constraint_row.conbin,
                   constraint_row.conrelid
               )
           ) > 0;

    SELECT COUNT(*)
      INTO legacy_constraint_count
      FROM pg_catalog.pg_constraint AS constraint_row
     WHERE constraint_row.conrelid = target_relation
       AND constraint_row.conname IN (
           'research_lab_private_model_benchmark_bundl_schema_version_check',
           'research_lab_private_model_benchmark_bundles_schema_version_che'
       );

    SELECT (
               COUNT(*) = 1
               AND COALESCE(BOOL_AND(constraint_row.convalidated), FALSE)
               AND COALESCE(
                   BOOL_AND(
                       POSITION(
                           '''1.0'''
                           IN pg_catalog.pg_get_expr(
                               constraint_row.conbin,
                               constraint_row.conrelid
                           )
                       ) > 0
                       AND POSITION(
                           '''1.1'''
                           IN pg_catalog.pg_get_expr(
                               constraint_row.conbin,
                               constraint_row.conrelid
                           )
                       ) > 0
                   ),
                   FALSE
               )
           )
      INTO contract_constraint_valid
      FROM pg_catalog.pg_constraint AS constraint_row
     WHERE constraint_row.conrelid = target_relation
       AND constraint_row.conname =
           'rl_private_benchmark_schema_version_check';

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.private_benchmark_schema_contract.v1',
        'constraint_name',
        'rl_private_benchmark_schema_version_check',
        'accepted_schema_versions',
        pg_catalog.jsonb_build_array('1.0', '1.1'),
        'schema_constraint_count',
        schema_constraint_count,
        'legacy_constraint_count',
        legacy_constraint_count,
        'constraint_valid',
        contract_constraint_valid
    );
END;
$$;

REVOKE ALL
    ON FUNCTION public.research_lab_private_benchmark_schema_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION public.research_lab_private_benchmark_schema_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.research_lab_private_benchmark_schema_contract_v1() IS
    'Declares the exact validated schema-version contract for private benchmark bundles.';

NOTIFY pgrst, 'reload schema';

COMMIT;
