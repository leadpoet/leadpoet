-- Compact operational ancestry checkpoints without duplicating raw sidecars.
--
-- Raw receipts, attempts, host operations, and receipt links remain immutable
-- in their existing authority tables. New v4 checkpoint documents retain the
-- signed compact proof and disclosures while omitting those already-durable
-- bodies. Existing v3 rows remain readable and append-only.

BEGIN;

DO $$
DECLARE
    item RECORD;
    widened_constraint_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
          AND conname = 'research_lab_ancestry_checkpoint_graph_schema_check'
          AND contype = 'c'
    ) INTO widened_constraint_exists;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_trigger
        WHERE tgrelid =
              'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
          AND tgname = 'prevent_research_lab_bounded_v2_mutation'
          AND tgenabled <> 'D'
          AND NOT tgisinternal
    ) THEN
        RAISE EXCEPTION 'ancestry checkpoint history is not append-only';
    END IF;
    IF NOT widened_constraint_exists AND NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
          AND contype = 'c'
          AND convalidated
          AND pg_catalog.pg_get_constraintdef(oid)
              LIKE '%checkpoint_graph_doc%'
          AND pg_catalog.pg_get_constraintdef(oid)
              LIKE '%leadpoet.attested_checkpointed_receipt_graph.v3%'
    ) THEN
        RAISE EXCEPTION 'validated v3 checkpoint history is unavailable';
    END IF;

    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid)
              LIKE '%checkpoint_graph_doc%'
          AND pg_catalog.pg_get_constraintdef(oid)
              LIKE '%leadpoet.attested_checkpointed_receipt_graph.v3%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_attested_ancestry_checkpoints_v2 DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_attested_ancestry_checkpoints_v2
    ADD CONSTRAINT research_lab_ancestry_checkpoint_graph_schema_check
    CHECK (
        checkpoint_graph_doc->>'schema_version' IN (
            'leadpoet.attested_checkpointed_receipt_graph.v3',
            'leadpoet.attested_checkpointed_receipt_graph.v4'
        )
    ) NOT VALID;

CREATE OR REPLACE FUNCTION
public.validate_research_lab_compact_checkpoint_sidecars_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    projection JSONB;
    disclosed_receipts JSONB;
    disclosed_boots JSONB;
    expected_receipt_count BIGINT;
    expected_boot_count BIGINT;
    expected_attempt_count BIGINT;
    expected_host_count BIGINT;
    durable_receipt_count BIGINT;
    durable_boot_count BIGINT;
    durable_attempt_count BIGINT;
    durable_host_count BIGINT;
BEGIN
    IF NEW.checkpoint_graph_doc->>'schema_version'
       <> 'leadpoet.attested_checkpointed_receipt_graph.v4'
    THEN
        RETURN NEW;
    END IF;

    projection := NEW.certificate_doc #> '{claim,local_delta_projection}';
    disclosed_receipts := NEW.proof_doc->'disclosed_receipts';
    disclosed_boots := NEW.proof_doc->'disclosed_boot_identities';
    IF jsonb_typeof(projection) <> 'object'
       OR jsonb_typeof(disclosed_receipts) <> 'array'
       OR jsonb_typeof(disclosed_boots) <> 'array'
       OR NEW.checkpoint_graph_doc->'transport_attempts' <> '[]'::JSONB
       OR NEW.checkpoint_graph_doc->'host_operations' <> '[]'::JSONB
       OR NEW.checkpoint_graph_doc->'receipts' <> disclosed_receipts
       OR NEW.checkpoint_graph_doc->'boot_identities' <> disclosed_boots
    THEN
        RAISE EXCEPTION 'compact checkpoint disclosure contract is invalid'
            USING ERRCODE = '23514';
    END IF;

    expected_receipt_count := (projection->>'receipt_count')::BIGINT;
    expected_boot_count := (projection->>'boot_identity_count')::BIGINT;
    expected_attempt_count := (projection->>'transport_attempt_count')::BIGINT;
    expected_host_count := (projection->>'host_operation_count')::BIGINT;
    IF expected_receipt_count <> jsonb_array_length(disclosed_receipts)
       OR expected_boot_count <> jsonb_array_length(disclosed_boots)
       OR expected_receipt_count < 1
       OR expected_boot_count < 1
       OR expected_attempt_count < 0
       OR expected_host_count < 0
    THEN
        RAISE EXCEPTION 'compact checkpoint projection counts differ'
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO durable_receipt_count
    FROM public.research_lab_attested_execution_receipts_v2 receipt
    WHERE receipt.receipt_hash IN (
        SELECT value->>'receipt_hash'
        FROM pg_catalog.jsonb_array_elements(disclosed_receipts)
    );
    SELECT count(*) INTO durable_boot_count
    FROM public.research_lab_attested_boot_identities_v2 boot
    WHERE boot.boot_identity_hash IN (
        SELECT value->>'boot_identity_hash'
        FROM pg_catalog.jsonb_array_elements(disclosed_boots)
    );
    SELECT count(*) INTO durable_attempt_count
    FROM public.research_lab_attested_receipt_transport_v2 link
    WHERE link.receipt_hash IN (
        SELECT value->>'receipt_hash'
        FROM pg_catalog.jsonb_array_elements(disclosed_receipts)
    );
    SELECT count(*) INTO durable_host_count
    FROM public.research_lab_attested_host_operations_v2 operation
    WHERE operation.receipt_hash IN (
        SELECT value->>'receipt_hash'
        FROM pg_catalog.jsonb_array_elements(disclosed_receipts)
    );

    IF durable_receipt_count <> expected_receipt_count
       OR durable_boot_count <> expected_boot_count
       OR durable_attempt_count <> expected_attempt_count
       OR durable_host_count <> expected_host_count
    THEN
        RAISE EXCEPTION 'compact checkpoint raw sidecars are incomplete'
            USING ERRCODE = '23503';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.validate_research_lab_compact_checkpoint_sidecars_v1()
    FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS validate_research_lab_compact_checkpoint_sidecars_v1
    ON public.research_lab_attested_ancestry_checkpoints_v2;
CREATE TRIGGER validate_research_lab_compact_checkpoint_sidecars_v1
    BEFORE INSERT ON public.research_lab_attested_ancestry_checkpoints_v2
    FOR EACH ROW
    EXECUTE FUNCTION
        public.validate_research_lab_compact_checkpoint_sidecars_v1();

CREATE OR REPLACE FUNCTION
public.research_lab_compact_checkpoint_graph_contract_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.compact_checkpoint_graph_contract.v1',
        'checkpoint_graph_schema_version',
        'leadpoet.attested_checkpointed_receipt_graph.v4',
        'legacy_checkpoint_graph_schema_version',
        'leadpoet.attested_checkpointed_receipt_graph.v3',
        'new_row_constraint_enabled',
        COALESCE((
            SELECT TRUE
            FROM pg_catalog.pg_constraint constraint_row
            WHERE constraint_row.conrelid =
                  'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
              AND constraint_row.conname =
                  'research_lab_ancestry_checkpoint_graph_schema_check'
        ), FALSE),
        'historical_rows_append_only',
        COALESCE((
            SELECT trigger_row.tgenabled <> 'D'
            FROM pg_catalog.pg_trigger trigger_row
            WHERE trigger_row.tgrelid =
                  'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
              AND trigger_row.tgname =
                  'prevent_research_lab_bounded_v2_mutation'
              AND NOT trigger_row.tgisinternal
        ), FALSE),
        'sidecar_trigger_enabled',
        COALESCE((
            SELECT trigger_row.tgenabled <> 'D'
            FROM pg_catalog.pg_trigger trigger_row
            WHERE trigger_row.tgrelid =
                  'public.research_lab_attested_ancestry_checkpoints_v2'::REGCLASS
              AND trigger_row.tgname =
                  'validate_research_lab_compact_checkpoint_sidecars_v1'
              AND NOT trigger_row.tgisinternal
        ), FALSE)
    );
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_compact_checkpoint_graph_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_compact_checkpoint_graph_contract_v1()
    TO service_role;

COMMENT ON FUNCTION
    public.research_lab_compact_checkpoint_graph_contract_v1() IS
    'Read-only contract proving compact v4 ancestry checkpoints and durable raw-sidecar enforcement are installed.';

NOTIFY pgrst, 'reload schema';

COMMIT;
