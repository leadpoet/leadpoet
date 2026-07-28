-- Allow measured chain-settlement transport evidence in the V2 authority log.
--
-- Migration 126 added the two chain-realized V1 operations to execution
-- results and receipt authority. Transport attempts retained migration 86's
-- generic ".v2" purpose constraint, so otherwise-valid measured transport
-- evidence for those operations could not be persisted.
--
-- This migration widens only that transport-purpose constraint. It does not
-- change scoring, allocation, settlement credit, weights, or chain behavior.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_attested_transport_attempts_v2_purpose_check;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    ADD CONSTRAINT
        research_lab_attested_transport_attempts_v2_purpose_check
    CHECK (
        purpose ~ '\.v2$'
        OR purpose IN (
            'research_lab.chain_weight_observation.v1',
            'research_lab.chain_realized_epoch_settlement.v1'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    VALIDATE CONSTRAINT
        research_lab_attested_transport_attempts_v2_purpose_check;

CREATE OR REPLACE FUNCTION
    public.research_lab_attested_transport_purpose_contract_v2()
RETURNS JSONB
LANGUAGE sql
STABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_attested_transport_purpose_contract.v2',
        'constraint_name',
        constraint_row.conname,
        'constraint_valid',
        constraint_row.convalidated,
        'constraint_definition',
        pg_catalog.pg_get_constraintdef(constraint_row.oid)
    )
    FROM pg_catalog.pg_constraint constraint_row
    WHERE constraint_row.conrelid =
          'public.research_lab_attested_transport_attempts_v2'
              ::pg_catalog.regclass
      AND constraint_row.conname =
          'research_lab_attested_transport_attempts_v2_purpose_check';
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_attested_transport_purpose_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_attested_transport_purpose_contract_v2()
    TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT public.research_lab_attested_transport_purpose_contract_v2();
