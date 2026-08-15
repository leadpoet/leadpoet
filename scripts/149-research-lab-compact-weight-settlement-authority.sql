-- Bound compact finalized weight authority for measured chain settlement.
--
-- Post-cutover settlement reads this independently verifiable sidecar instead
-- of expanding the historical full receipt graph. Keep the bound in the
-- database so a future producer change cannot silently recreate an oversized
-- coordinator response.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    existing_definition TEXT;
BEGIN
    SELECT pg_catalog.pg_get_constraintdef(c.oid)
      INTO existing_definition
      FROM pg_catalog.pg_constraint AS c
     WHERE c.conrelid =
           'public.research_lab_compact_weight_authorities_v2'::REGCLASS
       AND c.conname =
           'research_lab_compact_weight_authority_size_v2';

    IF existing_definition IS NULL THEN
        ALTER TABLE public.research_lab_compact_weight_authorities_v2
            ADD CONSTRAINT research_lab_compact_weight_authority_size_v2
            CHECK (
                pg_catalog.octet_length(authority_doc::TEXT) <= 8388608
            ) NOT VALID;
    ELSIF existing_definition NOT LIKE '%octet_length%authority_doc%8388608%' THEN
        RAISE EXCEPTION
            'research_lab_compact_weight_authority_size_v2 differs from the canonical 8 MiB bound';
    END IF;
END;
$$;

ALTER TABLE public.research_lab_compact_weight_authorities_v2
    VALIDATE CONSTRAINT research_lab_compact_weight_authority_size_v2;

CREATE OR REPLACE FUNCTION
public.research_lab_compact_weight_settlement_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_compact_weight_settlement_contract.v1',
        'max_authority_bytes',
        8388608,
        'size_constraint_valid',
        EXISTS (
            SELECT 1
              FROM pg_catalog.pg_constraint AS c
             WHERE c.conrelid =
                   'public.research_lab_compact_weight_authorities_v2'::REGCLASS
               AND c.conname =
                   'research_lab_compact_weight_authority_size_v2'
               AND c.convalidated
               AND pg_catalog.pg_get_constraintdef(c.oid)
                   LIKE '%octet_length%authority_doc%8388608%'
        ),
        'append_only_trigger_enabled',
        EXISTS (
            SELECT 1
              FROM pg_catalog.pg_trigger AS t
             WHERE t.tgrelid =
                   'public.research_lab_compact_weight_authorities_v2'::REGCLASS
               AND t.tgname = 'prevent_research_lab_bounded_v2_mutation'
               AND NOT t.tgisinternal
               AND t.tgenabled IN ('O', 'A')
        ),
        'identity_unique_constraint_enabled',
        EXISTS (
            SELECT 1
              FROM pg_catalog.pg_constraint AS c
             WHERE c.conrelid =
                   'public.research_lab_compact_weight_authorities_v2'::REGCLASS
               AND c.contype = 'u'
               AND pg_catalog.pg_get_constraintdef(c.oid)
                   LIKE '%(netuid, epoch_id, validator_hotkey, authority_stage)%'
        ),
        'row_level_security_enabled',
        COALESCE((
            SELECT cls.relrowsecurity
              FROM pg_catalog.pg_class AS cls
             WHERE cls.oid =
                   'public.research_lab_compact_weight_authorities_v2'::REGCLASS
        ), FALSE),
        'finalized_stage_supported',
        EXISTS (
            SELECT 1
              FROM pg_catalog.pg_constraint AS c
             WHERE c.conrelid =
                   'public.research_lab_compact_weight_authorities_v2'::REGCLASS
               AND c.contype = 'c'
               AND pg_catalog.pg_get_constraintdef(c.oid)
                   LIKE '%authority_stage%finalized%'
        )
    );
$$;

REVOKE ALL ON FUNCTION
public.research_lab_compact_weight_settlement_contract_v1()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
public.research_lab_compact_weight_settlement_contract_v1()
TO service_role;

COMMENT ON FUNCTION
public.research_lab_compact_weight_settlement_contract_v1() IS
    'Fail-closed schema contract for bounded append-only compact weight settlement authority.';

NOTIFY pgrst, 'reload schema';

COMMIT;
