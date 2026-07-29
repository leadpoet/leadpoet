-- Persist measured provider responses served from attested local evidence.
--
-- The canonical V2 transport contract has always distinguished authenticated
-- provider TLS responses from attested local responses. Migration 86 only
-- admitted provider TLS responses and transport failures, so a valid
-- attested_local_response could fail at durable persistence after measured
-- execution completed. This migration aligns the database checks with the
-- canonical contract without relaxing provider TLS requirements.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $drop_terminal_constraints$
DECLARE
    item RECORD;
    dropped_count INTEGER := 0;
BEGIN
    FOR item IN
        SELECT constraint_row.conname
        FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid =
              'public.research_lab_attested_transport_attempts_v2'
                  ::pg_catalog.regclass
          AND constraint_row.contype = 'c'
          AND pg_catalog.pg_get_constraintdef(constraint_row.oid)
              LIKE '%terminal_status%'
          AND pg_catalog.pg_get_constraintdef(constraint_row.oid)
              LIKE '%authenticated_response%'
          AND pg_catalog.pg_get_constraintdef(constraint_row.oid)
              LIKE '%transport_failure%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_attested_transport_attempts_v2 '
            || 'DROP CONSTRAINT %I',
            item.conname
        );
        dropped_count := dropped_count + 1;
    END LOOP;

    IF dropped_count < 2 THEN
        RAISE EXCEPTION
            'attested transport terminal constraints are incomplete: found %',
            dropped_count;
    END IF;
END;
$drop_terminal_constraints$;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    ADD CONSTRAINT research_lab_transport_terminal_status_v2_check
    CHECK (
        terminal_status IN (
            'authenticated_response',
            'attested_local_response',
            'transport_failure'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    ADD CONSTRAINT research_lab_transport_terminal_shape_v2_check
    CHECK (
        (
            terminal_status = 'authenticated_response'
            AND http_status IS NOT NULL
            AND response_hash IS NOT NULL
            AND response_artifact_hash IS NOT NULL
            AND tls_peer_chain_hash IS NOT NULL
            AND failure_code IS NULL
        )
        OR
        (
            terminal_status = 'attested_local_response'
            AND http_status IS NOT NULL
            AND response_hash IS NOT NULL
            AND response_artifact_hash IS NOT NULL
            AND tls_peer_chain_hash IS NULL
            AND failure_code IS NULL
        )
        OR
        (
            terminal_status = 'transport_failure'
            AND http_status IS NULL
            AND response_hash IS NULL
            AND response_artifact_hash IS NULL
            AND failure_code IS NOT NULL
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    VALIDATE CONSTRAINT research_lab_transport_terminal_status_v2_check;

ALTER TABLE public.research_lab_attested_transport_attempts_v2
    VALIDATE CONSTRAINT research_lab_transport_terminal_shape_v2_check;

CREATE OR REPLACE FUNCTION
    public.research_lab_attested_transport_terminal_contract_v2()
RETURNS JSONB
LANGUAGE sql
STABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_attested_transport_terminal_contract.v2',
        'constraints',
        pg_catalog.jsonb_object_agg(
            constraint_row.conname,
            pg_catalog.jsonb_build_object(
                'constraint_valid',
                constraint_row.convalidated,
                'constraint_definition',
                pg_catalog.pg_get_constraintdef(constraint_row.oid)
            )
            ORDER BY constraint_row.conname
        )
    )
    FROM pg_catalog.pg_constraint constraint_row
    WHERE constraint_row.conrelid =
          'public.research_lab_attested_transport_attempts_v2'
              ::pg_catalog.regclass
      AND constraint_row.conname IN (
          'research_lab_transport_terminal_status_v2_check',
          'research_lab_transport_terminal_shape_v2_check'
      );
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_attested_transport_terminal_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_attested_transport_terminal_contract_v2()
    TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT public.research_lab_attested_transport_terminal_contract_v2();
