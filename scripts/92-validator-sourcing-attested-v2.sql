-- Authoritative V2 sourcing epoch inputs for final-weight lineage.
--
-- Additive only. The legacy validator_weights_history JSON remains untouched
-- and is not accepted by V2. Each row is a canonical epoch aggregate produced
-- and signed by measured qualification code in a scoring enclave.
-- Apply after scripts 86-91; safe to apply repeatedly.

BEGIN;

CREATE TABLE IF NOT EXISTS public.validator_sourcing_epoch_inputs_v2 (
    epoch_id                 INTEGER     PRIMARY KEY CHECK (epoch_id >= 0),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.sourcing_epoch.v2'),
    epoch_hash               TEXT        NOT NULL UNIQUE CHECK (epoch_hash ~ '^sha256:[0-9a-f]{64}$'),
    decision_root            TEXT        NOT NULL CHECK (decision_root ~ '^sha256:[0-9a-f]{64}$'),
    receipt_hash             TEXT        NOT NULL UNIQUE
                                         REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                         ON DELETE RESTRICT,
    source_doc               JSONB       NOT NULL CHECK (
                                         jsonb_typeof(source_doc) = 'object'
                                         AND source_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                       ),
    receipt_doc              JSONB       NOT NULL CHECK (
                                         jsonb_typeof(receipt_doc) = 'object'
                                         AND receipt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                       ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (source_doc->>'schema_version' = schema_version),
    CHECK ((source_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK (source_doc->>'epoch_hash' = epoch_hash),
    CHECK (source_doc->>'decision_root' = decision_root),
    CHECK (receipt_doc->>'receipt_hash' = receipt_hash),
    CHECK (receipt_doc->>'role' = 'gateway_scoring'),
    CHECK (receipt_doc->>'purpose' = 'qualification.sourcing_epoch.v2'),
    CHECK ((receipt_doc->>'epoch_id')::INTEGER = epoch_id)
);

CREATE INDEX IF NOT EXISTS idx_validator_sourcing_epoch_inputs_v2_created
    ON public.validator_sourcing_epoch_inputs_v2(created_at DESC);

DROP TRIGGER IF EXISTS prevent_validator_sourcing_epoch_inputs_v2_mutation
    ON public.validator_sourcing_epoch_inputs_v2;
CREATE TRIGGER prevent_validator_sourcing_epoch_inputs_v2_mutation
    BEFORE UPDATE OR DELETE ON public.validator_sourcing_epoch_inputs_v2
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL ON TABLE public.validator_sourcing_epoch_inputs_v2 FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.validator_sourcing_epoch_inputs_v2 TO service_role;

ALTER TABLE public.validator_sourcing_epoch_inputs_v2 ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.validator_sourcing_epoch_inputs_v2;
CREATE POLICY service_role_read ON public.validator_sourcing_epoch_inputs_v2
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.validator_sourcing_epoch_inputs_v2;
CREATE POLICY service_role_insert ON public.validator_sourcing_epoch_inputs_v2
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.validator_sourcing_epoch_inputs_v2 IS
    'Append-only measured sourcing epoch aggregates for authoritative V2 final-weight lineage. Legacy host JSON is not a V2 source.';

COMMIT;

-- Production verification after applying migration 86, then this migration:
-- SELECT relrowsecurity
-- FROM pg_class
-- WHERE oid = 'public.validator_sourcing_epoch_inputs_v2'::regclass;
--
-- SELECT grantee, privilege_type
-- FROM information_schema.role_table_grants
-- WHERE table_schema = 'public'
--   AND table_name = 'validator_sourcing_epoch_inputs_v2'
-- ORDER BY grantee, privilege_type;
