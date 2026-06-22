-- Research Lab Phase 1: encrypted miner OpenRouter key references.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, 29, and 30.
--   * This script stores encrypted OpenRouter key ciphertext only.
--   * Raw OpenRouter keys must never be inserted into this table or any
--     Research Lab event/receipt/score-bundle JSON document.
--   * No anon/authenticated grants are created.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_openrouter_key_refs (
    key_ref                    TEXT        PRIMARY KEY CHECK (key_ref ~ '^encrypted_ref:openrouter:[0-9a-f]{32}$'),
    schema_version             TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    miner_hotkey               TEXT        NOT NULL,
    key_hash                   TEXT        NOT NULL,
    encrypted_key_ciphertext   TEXT        NOT NULL,
    kms_key_id                 TEXT        NOT NULL,
    encryption_context_hash    TEXT        NOT NULL CHECK (encryption_context_hash ~ '^sha256:[0-9a-f]{64}$'),
    preflight_status           TEXT        NOT NULL CHECK (preflight_status IN ('passed')),
    preflight_doc              JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(preflight_doc) = 'object'
                                            AND preflight_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role)'
                                            ),
    anchored_hash              TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (key_hash !~* '(sk-or-|api[_-]?key|raw[_-]?openrouter|secret)'),
    CHECK (kms_key_id !~* '(sk-or-|raw[_-]?openrouter|secret)'),
    CHECK (encrypted_key_ciphertext !~* '(sk-or-|raw[_-]?openrouter)')
);

CREATE INDEX IF NOT EXISTS idx_research_lab_openrouter_key_refs_miner
    ON public.research_lab_openrouter_key_refs(miner_hotkey, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_openrouter_key_refs_mutation
    ON public.research_lab_openrouter_key_refs;
CREATE TRIGGER prevent_research_lab_openrouter_key_refs_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_openrouter_key_refs
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_openrouter_key_refs FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_openrouter_key_refs TO service_role;

ALTER TABLE public.research_lab_openrouter_key_refs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_openrouter_key_refs;
CREATE POLICY service_role_read ON public.research_lab_openrouter_key_refs
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_openrouter_key_refs;
CREATE POLICY service_role_insert ON public.research_lab_openrouter_key_refs
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE public.research_lab_openrouter_key_refs IS
    'Append-only encrypted miner OpenRouter key references for hosted Research Lab runs. Stores KMS ciphertext only.';
COMMENT ON COLUMN public.research_lab_openrouter_key_refs.encrypted_key_ciphertext IS
    'Base64 KMS ciphertext for the miner OpenRouter key. Never stores raw key material.';

COMMIT;

-- Smoke check after applying this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name = 'research_lab_openrouter_key_refs';
