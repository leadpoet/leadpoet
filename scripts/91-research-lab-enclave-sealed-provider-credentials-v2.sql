-- Permit coordinator-sealed OpenRouter credentials in the existing V2 sidecar.
--
-- Additive authority change only. Existing KMS envelopes remain readable for
-- historical jobs; all new registrations use the enclave-sealed schema.

BEGIN;

DO $$
BEGIN
    IF to_regclass(
        'public.research_lab_provider_credential_envelopes_v2'
    ) IS NULL THEN
        RAISE EXCEPTION
            'Missing V2 credential table. Apply script 86 before script 91.';
    END IF;
END;
$$;

DO $$
DECLARE
    existing RECORD;
    desired_exists BOOLEAN := FALSE;
BEGIN
    FOR existing IN
        SELECT conname, pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid =
              'public.research_lab_provider_credential_envelopes_v2'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%schema_version%'
    LOOP
        IF existing.definition LIKE
               '%leadpoet.provider_credential_envelope.v2%'
           AND existing.definition LIKE
               '%leadpoet.provider_credential_envelope.enclave.v2%'
        THEN
            desired_exists := TRUE;
        ELSE
            EXECUTE format(
                'ALTER TABLE public.research_lab_provider_credential_envelopes_v2 '
                'DROP CONSTRAINT %I',
                existing.conname
            );
        END IF;
    END LOOP;

    IF NOT desired_exists THEN
        ALTER TABLE public.research_lab_provider_credential_envelopes_v2
            ADD CONSTRAINT rl_provider_credential_schema_v2_check
            CHECK (
                schema_version IN (
                    'leadpoet.provider_credential_envelope.v2',
                    'leadpoet.provider_credential_envelope.enclave.v2'
                )
            );
    END IF;
END;
$$;

COMMENT ON TABLE public.research_lab_provider_credential_envelopes_v2 IS
    'Append-only encrypted OpenRouter credential commitments. New V2 credentials are sealed inside the measured coordinator; historical KMS envelopes remain read-only compatible.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_provider_credential_envelopes_v2'::regclass
--   AND pg_get_constraintdef(oid) LIKE '%schema_version%';
