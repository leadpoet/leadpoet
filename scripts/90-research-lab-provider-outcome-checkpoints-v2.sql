-- Encrypted append-only restart checkpoints for measured V2 provider outcomes.
--
-- The coordinator enclave writes an authenticated ciphertext checkpoint after
-- each accepted provider outcome. The gateway parent and PostgREST never see
-- aggregate plaintext, credentials, requests, or provider response bodies.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_provider_outcome_checkpoints_v2 (
    checkpoint_hash          TEXT        PRIMARY KEY
                                           CHECK (checkpoint_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL
                                           CHECK (schema_version = 'leadpoet.provider_outcome_checkpoint_row.v2'),
    utc_day                  DATE        NOT NULL,
    sequence                 BIGINT      NOT NULL CHECK (sequence > 0),
    previous_checkpoint_hash TEXT        NOT NULL DEFAULT ''
                                           CHECK (previous_checkpoint_hash = '' OR previous_checkpoint_hash ~ '^sha256:[0-9a-f]{64}$'),
    state_document_hash      TEXT        NOT NULL
                                           CHECK (state_document_hash ~ '^sha256:[0-9a-f]{64}$'),
    checkpoint_artifact_id   TEXT        NOT NULL
                                           CHECK (checkpoint_artifact_id ~ '^sha256:[0-9a-f]{64}$'),
    encrypted_checkpoint_doc JSONB       NOT NULL
                                           CHECK (jsonb_typeof(encrypted_checkpoint_doc) = 'object'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (utc_day, sequence)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_provider_outcome_checkpoints_latest
    ON public.research_lab_provider_outcome_checkpoints_v2 (utc_day DESC, sequence DESC);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_provider_outcome_checkpoint_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_provider_outcome_checkpoints_v2 is append-only';
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_provider_outcome_checkpoint_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_provider_outcome_checkpoint_mutation()
    TO service_role;

DROP TRIGGER IF EXISTS prevent_research_lab_provider_outcome_checkpoint_mutation
    ON public.research_lab_provider_outcome_checkpoints_v2;
CREATE TRIGGER prevent_research_lab_provider_outcome_checkpoint_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_provider_outcome_checkpoints_v2
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_provider_outcome_checkpoint_mutation();

REVOKE ALL ON TABLE public.research_lab_provider_outcome_checkpoints_v2
    FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_provider_outcome_checkpoints_v2
    TO service_role;

ALTER TABLE public.research_lab_provider_outcome_checkpoints_v2 ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_provider_outcome_checkpoints_v2;
CREATE POLICY service_role_read
    ON public.research_lab_provider_outcome_checkpoints_v2
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_provider_outcome_checkpoints_v2;
CREATE POLICY service_role_insert
    ON public.research_lab_provider_outcome_checkpoints_v2
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_provider_outcome_checkpoints_v2 IS
    'Append-only encrypted coordinator checkpoints preserving measured provider-outcome context across enclave restarts.';

COMMIT;
