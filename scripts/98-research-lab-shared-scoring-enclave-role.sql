-- Permit the shared V2 scoring enclave physical role.
--
-- Additive constraint widening only. Historical gateway_scoring_a and
-- gateway_scoring_b boot identities remain valid and immutable. No scoring,
-- benchmark, promotion, reward, allocation, SOURCE_ADD, fulfillment,
-- emission, or validator-weight behavior is changed.
-- Apply after script 86; safe to apply repeatedly.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_attested_boot_identities_v2
    DROP CONSTRAINT IF EXISTS research_lab_attested_boot_identities_v2_physical_role_check;

ALTER TABLE public.research_lab_attested_boot_identities_v2
    ADD CONSTRAINT research_lab_attested_boot_identities_v2_physical_role_check
    CHECK (
        physical_role IN (
            'gateway_coordinator',
            'gateway_scoring',
            'gateway_scoring_a',
            'gateway_scoring_b',
            'gateway_autoresearch',
            'validator_weights'
        )
    );

COMMIT;

-- Verify after applying:
-- SELECT pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conname = 'research_lab_attested_boot_identities_v2_physical_role_check';
