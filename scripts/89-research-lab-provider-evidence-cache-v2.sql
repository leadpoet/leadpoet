-- V2 encrypted daily provider-evidence cache.
--
-- Apply after script 86. The shared append-only trigger function is repeated
-- idempotently here so a partially ordered rollout cannot create an unguarded
-- cache table. The measured coordinator is the only plaintext authority:
-- PostgREST stores AES-GCM ciphertext plus content commitments.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; insert a new V2 record', TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_attested_v2_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_attested_v2_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_provider_evidence_cache_v2 (
    schema_version            TEXT        NOT NULL
                                          CHECK (schema_version = 'leadpoet.provider_evidence_cache_row.v2'),
    utc_day                   DATE        NOT NULL,
    request_fingerprint       TEXT        NOT NULL
                                          CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    cache_entry_hash          TEXT        NOT NULL
                                          CHECK (cache_entry_hash ~ '^sha256:[0-9a-f]{64}$'),
    cache_artifact_id         TEXT        NOT NULL
                                          CHECK (cache_artifact_id ~ '^sha256:[0-9a-f]{64}$'),
    source_record_hash        TEXT        NOT NULL
                                          CHECK (source_record_hash ~ '^sha256:[0-9a-f]{64}$'),
    source_boot_identity_hash TEXT        NOT NULL
                                          CHECK (source_boot_identity_hash ~ '^sha256:[0-9a-f]{64}$'),
    response_body_hash        TEXT        NOT NULL
                                          CHECK (response_body_hash ~ '^sha256:[0-9a-f]{64}$'),
    encrypted_cache_doc       JSONB       NOT NULL CHECK (
        jsonb_typeof(encrypted_cache_doc) = 'object'
        AND encrypted_cache_doc->>'schema_version' = 'leadpoet.encrypted_artifact.v2'
        AND encrypted_cache_doc ?& ARRAY[
            'artifact_id',
            'plaintext_hash',
            'ciphertext_hash',
            'nonce_b64',
            'aad_b64',
            'encryption_context_hash',
            'ciphertext_b64',
            'object_lock_mode',
            'retain_until'
        ]
        AND encrypted_cache_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
    ),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (utc_day, request_fingerprint),
    UNIQUE (cache_entry_hash),
    UNIQUE (cache_artifact_id)
);
ALTER TABLE public.research_lab_provider_evidence_cache_v2
    ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_research_lab_provider_evidence_cache_v2_created
    ON public.research_lab_provider_evidence_cache_v2(utc_day DESC, created_at DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgrelid =
              'public.research_lab_provider_evidence_cache_v2'::regclass
          AND tgname =
              'prevent_research_lab_provider_evidence_cache_v2_mutation'
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER prevent_research_lab_provider_evidence_cache_v2_mutation
            BEFORE UPDATE OR DELETE
            ON public.research_lab_provider_evidence_cache_v2
            FOR EACH ROW EXECUTE FUNCTION
                public.prevent_research_lab_attested_v2_mutation();
    END IF;
END;
$$;

REVOKE ALL ON TABLE public.research_lab_provider_evidence_cache_v2
    FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_provider_evidence_cache_v2
    TO service_role;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'research_lab_provider_evidence_cache_v2'
          AND policyname = 'service_role_read'
    ) THEN
        CREATE POLICY service_role_read
            ON public.research_lab_provider_evidence_cache_v2
            FOR SELECT TO service_role USING (true);
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'research_lab_provider_evidence_cache_v2'
          AND policyname = 'service_role_insert'
    ) THEN
        CREATE POLICY service_role_insert
            ON public.research_lab_provider_evidence_cache_v2
            FOR INSERT TO service_role WITH CHECK (true);
    END IF;
END;
$$;

COMMENT ON TABLE public.research_lab_provider_evidence_cache_v2 IS
    'Append-only AES-GCM provider-evidence cache. Plaintext is available only inside approved coordinator enclaves.';

COMMIT;
