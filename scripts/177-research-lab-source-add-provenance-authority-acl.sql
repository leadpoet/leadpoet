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

-- Rebind the exact security-invoker projection in this migration so the
-- restart preflight can identify 177 as the complete relation + ACL repair.
CREATE OR REPLACE VIEW
    public.research_lab_source_add_provenance_leg1_authority_v1
WITH (security_invoker = true) AS
WITH exact_provenance AS (
    SELECT DISTINCT ON (history.submission_id)
        history.submission_id,
        history.adapter_id,
        history.miner_hotkey,
        history.precheck_status,
        receipt.receipt_hash AS provenance_receipt_hash,
        receipt.output_root AS provenance_artifact_hash,
        history.created_at AS provenance_created_at,
        public.research_lab_source_add_provider_origin_hash_v1(
            history.submission_doc #>> '{source_metadata,api_base_url}'
        ) AS provider_origin_hash
    FROM public.research_lab_source_add_submissions history
    JOIN public.research_lab_attested_execution_receipts_v2 receipt
      ON receipt.receipt_hash =
         history.submission_doc->>'provenance_receipt_hash'
    JOIN public.research_lab_attested_business_artifact_links_v2 link
      ON link.receipt_hash = receipt.receipt_hash
     AND link.artifact_kind = 'source_add_provenance'
     AND link.artifact_ref = history.submission_id
     AND link.artifact_hash = receipt.output_root
    WHERE history.precheck_status = 'provenance_precheck_passed'
      AND history.precheck_doc->>'precheck_status' =
          'provenance_precheck_passed'
      AND history.submission_doc->>'provenance_receipt_hash'
          ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.role = 'gateway_coordinator'
      AND receipt.purpose = 'research_lab.source_add_provenance.v2'
      AND receipt.receipt_status = 'succeeded'
      AND receipt.output_root ~ '^sha256:[0-9a-f]{64}$'
      AND receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
      AND NOT EXISTS (
          SELECT 1
          FROM public.research_lab_attested_receipt_edges_v2 edge
          WHERE edge.child_receipt_hash = receipt.receipt_hash
      )
    ORDER BY history.submission_id, history.seq ASC, history.created_at ASC
), ranked AS (
    SELECT
        exact_provenance.*,
        ROW_NUMBER() OVER (
            PARTITION BY exact_provenance.provider_origin_hash
            ORDER BY
                exact_provenance.provenance_created_at ASC,
                exact_provenance.submission_id ASC
        ) AS origin_rank
    FROM exact_provenance
)
SELECT
    ranked.submission_id,
    ranked.adapter_id,
    ranked.miner_hotkey,
    ranked.precheck_status,
    ranked.provenance_receipt_hash,
    ranked.provenance_artifact_hash,
    ranked.provenance_created_at
FROM ranked
JOIN public.research_lab_source_add_provider_origin_current origin
  ON origin.provider_origin_hash = ranked.provider_origin_hash
 AND origin.submission_id = ranked.submission_id
 AND origin.adapter_id = ranked.adapter_id
 AND origin.miner_hotkey = ranked.miner_hotkey
 AND origin.reservation_status = 'reserved'
WHERE ranked.origin_rank = 1;

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
