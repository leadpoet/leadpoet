-- Make the provenance Leg 1 authority readable through the production
-- service-role PostgREST path.  Migration 176 intentionally kept the URL
-- normalization helpers private, but its security-invoker authority view
-- calls both helpers while evaluating a submitted provider origin.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $source_add_provenance_authority_acl_preflight$
BEGIN
    IF pg_catalog.to_regclass(
        'public.research_lab_source_add_provenance_leg1_authority_v1'
    ) IS NULL OR pg_catalog.to_regprocedure(
        'public.research_lab_source_add_provider_origin_host_v1(text)'
    ) IS NULL OR pg_catalog.to_regprocedure(
        'public.research_lab_source_add_provider_origin_hash_v1(text)'
    ) IS NULL THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance authority migration 176 is unavailable';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'anon'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'authenticated'
    ) OR NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_namespace
        WHERE nspname = 'extensions'
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance authority ACL prerequisites are unavailable';
    END IF;
    IF NOT COALESCE((
        SELECT control.paused
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    ), FALSE) THEN
        RAISE EXCEPTION
            'SOURCE_ADD must be paused before provenance authority ACL repair';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
          AND control.restart_guard_commitment <> ''
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD restart guard is active during authority ACL repair';
    END IF;
END;
$source_add_provenance_authority_acl_preflight$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_host_v1(TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provider_origin_hash_v1(TEXT)
    FROM PUBLIC, anon, authenticated;

GRANT USAGE ON SCHEMA public, extensions TO service_role;
GRANT SELECT ON TABLE
    public.research_lab_source_add_provenance_leg1_authority_v1
    TO service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_provider_origin_host_v1(TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_provider_origin_hash_v1(TEXT)
    TO service_role;

DO $source_add_provenance_authority_acl_readback$
BEGIN
    IF NOT pg_catalog.has_schema_privilege(
        'service_role', 'public', 'USAGE'
    ) OR NOT pg_catalog.has_schema_privilege(
        'service_role', 'extensions', 'USAGE'
    ) OR NOT pg_catalog.has_table_privilege(
        'service_role',
        'public.research_lab_source_add_provenance_leg1_authority_v1',
        'SELECT'
    ) OR NOT pg_catalog.has_function_privilege(
        'service_role',
        'public.research_lab_source_add_provider_origin_host_v1(text)',
        'EXECUTE'
    ) OR NOT pg_catalog.has_function_privilege(
        'service_role',
        'public.research_lab_source_add_provider_origin_hash_v1(text)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'anon',
        'public.research_lab_source_add_provider_origin_host_v1(text)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'authenticated',
        'public.research_lab_source_add_provider_origin_host_v1(text)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'anon',
        'public.research_lab_source_add_provider_origin_hash_v1(text)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'authenticated',
        'public.research_lab_source_add_provider_origin_hash_v1(text)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance authority ACL readback differs';
    END IF;
END;
$source_add_provenance_authority_acl_readback$;

NOTIFY pgrst, 'reload schema';

COMMIT;
