-- Private Research Lab provider-capability snapshots and service-only access.
--
-- Additive policy migration. Production capability rows are seeded separately
-- by the owner and are intentionally not committed to the public repository.

BEGIN;

ALTER TABLE public.research_lab_provider_registry
    DROP CONSTRAINT IF EXISTS research_lab_provider_registry_registry_hash_key;

CREATE INDEX IF NOT EXISTS idx_research_lab_provider_registry_hash
    ON public.research_lab_provider_registry (registry_hash, created_at DESC);

ALTER TABLE public.research_lab_provider_registry
    DROP CONSTRAINT IF EXISTS research_lab_provider_registry_registry_doc_check;

ALTER TABLE public.research_lab_provider_registry
    ADD CONSTRAINT research_lab_provider_registry_registry_doc_check
    CHECK (
        jsonb_typeof(registry_doc) = 'object'
        AND registry_doc->>'schema_version' = '1.0'
        AND jsonb_typeof(registry_doc->'providers') = 'array'
        AND jsonb_array_length(registry_doc->'providers') = provider_count
        AND registry_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|hidden_prompt|provider_output|request_body|response_body|page_content|raw_content|judge_prompt|private_manifest|private_repo|proxy[_-]?url|://[^/]+:[^/@]+@)'
        AND registry_doc::TEXT !~* '"(password|secret|token|api_key)"\s*:\s*"[^"[:space:]][^"]*"'
    );

CREATE OR REPLACE VIEW public.research_lab_provider_registry_current
WITH (security_invoker = true) AS
SELECT
    registry_snapshot_id,
    schema_version,
    registry_hash,
    provider_count,
    registry_doc,
    created_by,
    created_at
FROM public.research_lab_provider_registry
WHERE jsonb_typeof(registry_doc) = 'object'
  AND registry_doc->>'schema_version' = '1.0'
  AND jsonb_typeof(registry_doc->'providers') = 'array'
  AND jsonb_array_length(registry_doc->'providers') = provider_count
ORDER BY created_at DESC, registry_snapshot_id DESC
LIMIT 1;

REVOKE ALL ON TABLE public.research_lab_provider_registry FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_provider_registry_current FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_provider_registry TO service_role;
GRANT SELECT ON TABLE public.research_lab_provider_registry_current TO service_role;

ALTER TABLE public.research_lab_provider_registry ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_provider_registry_service_select
    ON public.research_lab_provider_registry;
CREATE POLICY research_lab_provider_registry_service_select
    ON public.research_lab_provider_registry
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS research_lab_provider_registry_service_insert
    ON public.research_lab_provider_registry;
CREATE POLICY research_lab_provider_registry_service_insert
    ON public.research_lab_provider_registry
    FOR INSERT TO service_role WITH CHECK (true);

REVOKE ALL ON TABLE public.research_lab_provider_usage_ledger FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_provider_usage_ledger TO service_role;
ALTER TABLE public.research_lab_provider_usage_ledger ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_provider_usage_service_select
    ON public.research_lab_provider_usage_ledger;
CREATE POLICY research_lab_provider_usage_service_select
    ON public.research_lab_provider_usage_ledger
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS research_lab_provider_usage_service_insert
    ON public.research_lab_provider_usage_ledger;
CREATE POLICY research_lab_provider_usage_service_insert
    ON public.research_lab_provider_usage_ledger
    FOR INSERT TO service_role WITH CHECK (true);

REVOKE ALL ON TABLE public.research_lab_source_add_submissions FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_source_add_submissions TO service_role;
ALTER TABLE public.research_lab_source_add_submissions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_source_add_submissions_service_select
    ON public.research_lab_source_add_submissions;
CREATE POLICY research_lab_source_add_submissions_service_select
    ON public.research_lab_source_add_submissions
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS research_lab_source_add_submissions_service_insert
    ON public.research_lab_source_add_submissions;
CREATE POLICY research_lab_source_add_submissions_service_insert
    ON public.research_lab_source_add_submissions
    FOR INSERT TO service_role WITH CHECK (true);

REVOKE ALL ON TABLE public.research_lab_source_catalog FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_source_catalog TO service_role;
ALTER TABLE public.research_lab_source_catalog ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_source_catalog_service_select
    ON public.research_lab_source_catalog;
CREATE POLICY research_lab_source_catalog_service_select
    ON public.research_lab_source_catalog
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS research_lab_source_catalog_service_insert
    ON public.research_lab_source_catalog;
CREATE POLICY research_lab_source_catalog_service_insert
    ON public.research_lab_source_catalog
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON VIEW public.research_lab_provider_registry_current IS
    'Newest structurally valid private Research Lab provider-capability snapshot. Service role only.';
COMMENT ON TABLE public.research_lab_provider_registry IS
    'Append-only private Research Lab provider-capability snapshots. Active production inventory is never projected publicly.';

COMMIT;
