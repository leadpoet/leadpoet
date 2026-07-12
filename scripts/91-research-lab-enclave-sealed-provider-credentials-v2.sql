-- Permit coordinator-sealed OpenRouter credentials in the existing V2 sidecar.
--
-- Additive authority change only. Existing KMS envelopes remain readable for
-- historical jobs; all new registrations use the enclave-sealed schema.

BEGIN;

ALTER TABLE public.research_lab_provider_credential_envelopes_v2
    DROP CONSTRAINT IF EXISTS research_lab_provider_credential_envelopes_v2_schema_version_check;

ALTER TABLE public.research_lab_provider_credential_envelopes_v2
    ADD CONSTRAINT research_lab_provider_credential_envelopes_v2_schema_version_check
    CHECK (
        schema_version IN (
            'leadpoet.provider_credential_envelope.v2',
            'leadpoet.provider_credential_envelope.enclave.v2'
        )
    );

COMMENT ON TABLE public.research_lab_provider_credential_envelopes_v2 IS
    'Append-only encrypted OpenRouter credential commitments. New V2 credentials are sealed inside the measured coordinator; historical KMS envelopes remain read-only compatible.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_provider_credential_envelopes_v2'::regclass
--   AND conname = 'research_lab_provider_credential_envelopes_v2_schema_version_check';
