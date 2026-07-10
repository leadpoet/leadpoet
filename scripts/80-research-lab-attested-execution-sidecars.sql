-- Research Lab attested scoring and final-weight receipt sidecars.
--
-- Additive only. Existing scoring, benchmark, promotion, reward, allocation,
-- SOURCE_ADD, fulfillment, and published-weight business tables are unchanged.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_attested_receipt_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; insert a new attested receipt', TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_attested_receipt_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_attested_receipt_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_attested_execution_receipts (
    receipt_hash             TEXT        PRIMARY KEY CHECK (receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.attested_receipt.v1'),
    role                     TEXT        NOT NULL CHECK (role IN ('gateway_scoring', 'validator_weights')),
    purpose                  TEXT        NOT NULL,
    job_id                   TEXT        NOT NULL,
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    commit_sha               TEXT        NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    pcr0                     TEXT        NOT NULL CHECK (pcr0 ~ '^[0-9a-f]{96}$'),
    build_manifest_hash      TEXT        NOT NULL CHECK (build_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    config_hash              TEXT        NOT NULL CHECK (config_hash ~ '^sha256:[0-9a-f]{64}$'),
    input_root               TEXT        NOT NULL CHECK (input_root ~ '^sha256:[0-9a-f]{64}$'),
    output_root              TEXT        NOT NULL CHECK (output_root ~ '^sha256:[0-9a-f]{64}$'),
    parent_receipt_hashes    JSONB       NOT NULL DEFAULT '[]'::JSONB CHECK (jsonb_typeof(parent_receipt_hashes) = 'array'),
    evidence_roots           JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (jsonb_typeof(evidence_roots) = 'object'),
    receipt_status           TEXT        NOT NULL CHECK (receipt_status IN ('succeeded', 'failed')),
    receipt_doc              JSONB       NOT NULL CHECK (
                                          jsonb_typeof(receipt_doc) = 'object'
                                          AND receipt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|icp_plaintext|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    attestation_document_ref TEXT        NOT NULL,
    attestation_document_hash TEXT       NOT NULL CHECK (attestation_document_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (role, purpose, job_id, input_root, config_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_artifact_links (
    link_id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts(receipt_hash) ON DELETE RESTRICT,
    artifact_kind            TEXT        NOT NULL,
    artifact_ref             TEXT        NOT NULL,
    artifact_hash            TEXT        NOT NULL CHECK (artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (receipt_hash, artifact_kind, artifact_ref, artifact_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_bundles (
    weight_receipt_hash      TEXT        PRIMARY KEY REFERENCES public.research_lab_attested_execution_receipts(receipt_hash) ON DELETE RESTRICT,
    netuid                   INTEGER     NOT NULL CHECK (netuid >= 0),
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    block                    BIGINT      NOT NULL CHECK (block >= 0),
    validator_hotkey         TEXT        NOT NULL,
    weights_hash             TEXT        NOT NULL CHECK (weights_hash ~ '^[0-9a-f]{64}$'),
    validator_commit_sha     TEXT        NOT NULL CHECK (validator_commit_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    validator_pcr0           TEXT        NOT NULL CHECK (validator_pcr0 ~ '^[0-9a-f]{96}$'),
    verification_mode        TEXT        NOT NULL CHECK (verification_mode IN ('shadow', 'required')),
    bundle_hash              TEXT        NOT NULL CHECK (bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    bundle_doc               JSONB       NOT NULL CHECK (
                                          jsonb_typeof(bundle_doc) = 'object'
                                          AND bundle_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|icp_plaintext|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (netuid, epoch_id, validator_hotkey)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_attested_receipts_epoch
    ON public.research_lab_attested_execution_receipts(epoch_id DESC, role, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_receipts_job
    ON public.research_lab_attested_execution_receipts(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_artifact_links_ref
    ON public.research_lab_attested_artifact_links(artifact_kind, artifact_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_weight_bundles_epoch
    ON public.research_lab_attested_weight_bundles(netuid, epoch_id DESC, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_attested_execution_receipts_mutation
    ON public.research_lab_attested_execution_receipts;
CREATE TRIGGER prevent_research_lab_attested_execution_receipts_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_attested_execution_receipts
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_receipt_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_attested_artifact_links_mutation
    ON public.research_lab_attested_artifact_links;
CREATE TRIGGER prevent_research_lab_attested_artifact_links_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_attested_artifact_links
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_receipt_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_attested_weight_bundles_mutation
    ON public.research_lab_attested_weight_bundles;
CREATE TRIGGER prevent_research_lab_attested_weight_bundles_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_attested_weight_bundles
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_receipt_mutation();

REVOKE ALL ON TABLE public.research_lab_attested_execution_receipts FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_attested_artifact_links FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_attested_weight_bundles FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_attested_execution_receipts TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_attested_artifact_links TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_attested_weight_bundles TO service_role;

ALTER TABLE public.research_lab_attested_execution_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_attested_artifact_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_attested_weight_bundles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_attested_execution_receipts;
CREATE POLICY service_role_read ON public.research_lab_attested_execution_receipts
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_attested_execution_receipts;
CREATE POLICY service_role_insert ON public.research_lab_attested_execution_receipts
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read ON public.research_lab_attested_artifact_links;
CREATE POLICY service_role_read ON public.research_lab_attested_artifact_links
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_attested_artifact_links;
CREATE POLICY service_role_insert ON public.research_lab_attested_artifact_links
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read ON public.research_lab_attested_weight_bundles;
CREATE POLICY service_role_read ON public.research_lab_attested_weight_bundles
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_attested_weight_bundles;
CREATE POLICY service_role_insert ON public.research_lab_attested_weight_bundles
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_attested_execution_receipts IS
    'Append-only enclave-signed Research Lab scoring and validator weight receipts. Sidecar only; no scoring or reward authority.';
COMMENT ON TABLE public.research_lab_attested_artifact_links IS
    'Append-only links from attested receipts to existing immutable scoring, benchmark, allocation, and weight artifacts.';
COMMENT ON TABLE public.research_lab_attested_weight_bundles IS
    'Append-only shadow/required v2 weight bundles for auditor receipt-chain verification. Does not replace published_weight_bundles.';

COMMIT;
