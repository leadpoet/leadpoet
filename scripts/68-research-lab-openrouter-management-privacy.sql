-- Research Lab: require OpenRouter management-key privacy proof for miner keys.
--
-- Deployment policy:
--   * Apply after script 31.
--   * Safe to apply repeatedly.
--   * Adds encrypted management-key storage and append-only redacted privacy
--     proof events. Raw OpenRouter keys, hidden prompts, and provider outputs
--     must never be stored in these columns.

BEGIN;

ALTER TABLE public.research_lab_openrouter_key_refs
    ADD COLUMN IF NOT EXISTS encrypted_management_key_ciphertext TEXT,
    ADD COLUMN IF NOT EXISTS management_key_hash TEXT,
    ADD COLUMN IF NOT EXISTS management_kms_key_id TEXT,
    ADD COLUMN IF NOT EXISTS management_encryption_context_hash TEXT,
    ADD COLUMN IF NOT EXISTS openrouter_workspace_hash TEXT,
    ADD COLUMN IF NOT EXISTS privacy_status TEXT NOT NULL DEFAULT 'not_configured',
    ADD COLUMN IF NOT EXISTS privacy_verified_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS privacy_proof_doc JSONB NOT NULL DEFAULT '{}'::JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_openrouter_key_refs_privacy_status_check'
    ) THEN
        ALTER TABLE public.research_lab_openrouter_key_refs
            ADD CONSTRAINT research_lab_openrouter_key_refs_privacy_status_check
            CHECK (privacy_status IN ('not_configured', 'verified'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_openrouter_key_refs_privacy_verified_check'
    ) THEN
        ALTER TABLE public.research_lab_openrouter_key_refs
            ADD CONSTRAINT research_lab_openrouter_key_refs_privacy_verified_check
            CHECK (
                privacy_status <> 'verified'
                OR (
                    encrypted_management_key_ciphertext IS NOT NULL
                    AND management_key_hash IS NOT NULL
                    AND management_kms_key_id IS NOT NULL
                    AND management_encryption_context_hash ~ '^sha256:[0-9a-f]{64}$'
                    AND openrouter_workspace_hash ~ '^[0-9a-f]{64}$'
                    AND privacy_verified_at IS NOT NULL
                    AND jsonb_typeof(privacy_proof_doc) = 'object'
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_openrouter_key_refs_management_secret_check'
    ) THEN
        ALTER TABLE public.research_lab_openrouter_key_refs
            ADD CONSTRAINT research_lab_openrouter_key_refs_management_secret_check
            CHECK (
                COALESCE(encrypted_management_key_ciphertext, '') !~* '(sk-or-|raw[_-]?openrouter)'
                AND COALESCE(management_key_hash, '') !~* '(sk-or-|api[_-]?key|raw[_-]?openrouter|secret)'
                AND COALESCE(management_kms_key_id, '') !~* '(sk-or-|raw[_-]?openrouter)'
                AND privacy_proof_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|raw_openrouter_key|raw_secret|service_role|hidden_prompt|provider_output)'
            );
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS public.research_lab_openrouter_privacy_proof_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0'
                              CHECK (schema_version = '1.0'),
    key_ref       TEXT        NOT NULL CHECK (key_ref ~ '^encrypted_ref:openrouter:[0-9a-f]{32}$'),
    miner_hotkey  TEXT        NOT NULL,
    run_id        UUID,
    stage         TEXT        NOT NULL,
    proof_status  TEXT        NOT NULL CHECK (proof_status IN ('passed', 'failed')),
    proof_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(proof_doc) = 'object'
                              AND proof_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|raw_openrouter_key|raw_secret|service_role|hidden_prompt|provider_output)'
                              ),
    anchored_hash TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_lab_openrouter_privacy_proof_events_key
    ON public.research_lab_openrouter_privacy_proof_events(key_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_openrouter_privacy_proof_events_run
    ON public.research_lab_openrouter_privacy_proof_events(run_id, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_openrouter_privacy_proof_events_mutation
    ON public.research_lab_openrouter_privacy_proof_events;
CREATE TRIGGER prevent_research_lab_openrouter_privacy_proof_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_openrouter_privacy_proof_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_openrouter_privacy_proof_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_openrouter_privacy_proof_events TO service_role;

ALTER TABLE public.research_lab_openrouter_privacy_proof_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_openrouter_privacy_proof_events;
CREATE POLICY service_role_read ON public.research_lab_openrouter_privacy_proof_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_openrouter_privacy_proof_events;
CREATE POLICY service_role_insert ON public.research_lab_openrouter_privacy_proof_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

COMMENT ON COLUMN public.research_lab_openrouter_key_refs.encrypted_management_key_ciphertext IS
    'Base64 KMS ciphertext for the miner OpenRouter management key. Never stores raw key material.';
COMMENT ON TABLE public.research_lab_openrouter_privacy_proof_events IS
    'Append-only redacted OpenRouter workspace privacy proof events written before hidden Research Lab prompts.';

COMMIT;

-- Verification:
-- SELECT column_name
-- FROM information_schema.columns
-- WHERE table_schema = 'public'
--   AND table_name = 'research_lab_openrouter_key_refs'
--   AND column_name IN ('encrypted_management_key_ciphertext', 'privacy_status');
--
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name = 'research_lab_openrouter_privacy_proof_events';
