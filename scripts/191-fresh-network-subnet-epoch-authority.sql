-- Additive, receipt-backed epoch authority for a fresh non-Finney network.
--
-- This does not mutate or reinterpret the terminal Finney/SN71 singleton.
-- A fresh network has no Leadpoet legacy settlement predecessor. Its origin is
-- instead an exact finalized boundary snapshot signed by validator_weights and
-- accepted by the measured gateway coordinator as its only parent.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_schema_v2_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_predecessor_v2_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_authority_v2_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_schema_v3_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_predecessor_v3_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_authority_v3_check,
    DROP CONSTRAINT IF EXISTS research_lab_stateful_epoch_cutover_previous_scheme_v3_check;

DO $constraints$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE
              '%previous_epoch_scheme%legacy_global_360_v1%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 '
            'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$constraints$;

ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_schema_v3_check
        CHECK (
            schema_version IN (
                'leadpoet.subnet_epoch_cutover_authority.v1',
                'leadpoet.subnet_epoch_cutover_authority.v2',
                'leadpoet.subnet_epoch_cutover_authority.v3'
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_previous_scheme_v3_check
        CHECK (
            previous_epoch_scheme IN (
                'legacy_global_360_v1',
                'fresh_network_v1'
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_predecessor_v3_check
        CHECK (
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v1'
                AND previous_epoch_scheme = 'legacy_global_360_v1'
                AND last_legacy_bundle_hash IS NOT NULL
                AND last_legacy_weight_finalization_event_hash IS NOT NULL
                AND last_legacy_finalization_receipt_hash IS NOT NULL
                AND predecessor_kind IS NULL
                AND predecessor_epoch_id IS NULL
                AND predecessor_allocation_hash IS NULL
                AND predecessor_authority_hash IS NULL
                AND predecessor_receipt_hash IS NULL
            ) OR (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v2'
                AND previous_epoch_scheme = 'legacy_global_360_v1'
                AND last_legacy_bundle_hash IS NULL
                AND last_legacy_weight_finalization_event_hash IS NULL
                AND last_legacy_finalization_receipt_hash IS NULL
                AND predecessor_kind = 'legacy_finalized_chain_migration_v2'
                AND predecessor_epoch_id BETWEEN 0 AND last_legacy_epoch_id
                AND predecessor_allocation_hash ~ '^sha256:[0-9a-f]{64}$'
                AND predecessor_authority_hash ~ '^sha256:[0-9a-f]{64}$'
                AND predecessor_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
            ) OR (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v3'
                AND previous_epoch_scheme = 'fresh_network_v1'
                AND last_legacy_bundle_hash IS NULL
                AND last_legacy_weight_finalization_event_hash IS NULL
                AND last_legacy_finalization_receipt_hash IS NULL
                AND predecessor_kind IS NULL
                AND predecessor_epoch_id IS NULL
                AND predecessor_allocation_hash IS NULL
                AND predecessor_authority_hash IS NULL
                AND predecessor_receipt_hash IS NULL
            )
        ),
    ADD CONSTRAINT research_lab_stateful_epoch_cutover_authority_v3_check
        CHECK (
            (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v1'
                AND authority_doc ?& ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'last_legacy_bundle_hash',
                    'last_legacy_weight_finalization_event_hash',
                    'last_legacy_finalization_receipt_hash', 'manifest'
                ]
                AND authority_doc - ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'last_legacy_bundle_hash',
                    'last_legacy_weight_finalization_event_hash',
                    'last_legacy_finalization_receipt_hash', 'manifest'
                ] = '{}'::JSONB
                AND authority_doc->>'last_legacy_bundle_hash' =
                    last_legacy_bundle_hash
                AND authority_doc->>'last_legacy_weight_finalization_event_hash' =
                    last_legacy_weight_finalization_event_hash
                AND authority_doc->>'last_legacy_finalization_receipt_hash' =
                    last_legacy_finalization_receipt_hash
            ) OR (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v2'
                AND authority_doc ?& ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'predecessor_kind', 'predecessor_epoch_id',
                    'predecessor_allocation_hash',
                    'predecessor_authority_hash',
                    'predecessor_receipt_hash', 'manifest'
                ]
                AND authority_doc - ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'predecessor_kind', 'predecessor_epoch_id',
                    'predecessor_allocation_hash',
                    'predecessor_authority_hash',
                    'predecessor_receipt_hash', 'manifest'
                ] = '{}'::JSONB
                AND authority_doc->>'predecessor_kind' = predecessor_kind
                AND (authority_doc->>'predecessor_epoch_id')::INTEGER =
                    predecessor_epoch_id
                AND authority_doc->>'predecessor_allocation_hash' =
                    predecessor_allocation_hash
                AND authority_doc->>'predecessor_authority_hash' =
                    predecessor_authority_hash
                AND authority_doc->>'predecessor_receipt_hash' =
                    predecessor_receipt_hash
            ) OR (
                schema_version = 'leadpoet.subnet_epoch_cutover_authority.v3'
                AND authority_doc ?& ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'origin_kind', 'manifest'
                ]
                AND authority_doc - ARRAY[
                    'schema_version', 'mapping_hash', 'first_epoch_ref',
                    'first_snapshot_hash', 'first_snapshot_receipt_hash',
                    'origin_kind', 'manifest'
                ] = '{}'::JSONB
                AND authority_doc->>'origin_kind' =
                    'fresh_testnet401_network'
            )
        );

CREATE OR REPLACE FUNCTION
public.validate_research_lab_fresh_network_epoch_cutover_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    snapshot_receipt RECORD;
    coordinator_receipt RECORD;
    candidate_row RECORD;
    edge_count BIGINT;
BEGIN
    IF NEW.schema_version IS DISTINCT FROM
       'leadpoet.subnet_epoch_cutover_authority.v3'
       OR NEW.previous_epoch_scheme IS DISTINCT FROM 'fresh_network_v1'
       OR NEW.network_genesis_hash IS DISTINCT FROM
          '0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105'
       OR NEW.network_genesis_hash =
          '0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03'
       OR NEW.netuid IS DISTINCT FROM 401
    THEN
        RAISE EXCEPTION 'fresh-network cutover scope is invalid';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        7102,
        pg_catalog.hashtext(NEW.network_genesis_hash || ':' || NEW.netuid::TEXT)
    );

    SELECT role, purpose, epoch_id, receipt_status, output_root
    INTO snapshot_receipt
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = NEW.first_snapshot_receipt_hash;
    IF NOT FOUND
       OR snapshot_receipt.role IS DISTINCT FROM 'validator_weights'
       OR snapshot_receipt.purpose IS DISTINCT FROM
          'validator.subnet_epoch_snapshot.v2'
       OR snapshot_receipt.epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR snapshot_receipt.receipt_status IS DISTINCT FROM 'succeeded'
       OR snapshot_receipt.output_root IS DISTINCT FROM NEW.first_snapshot_hash
    THEN
        RAISE EXCEPTION 'fresh-network first snapshot receipt is invalid';
    END IF;

    SELECT * INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1
    WHERE snapshot_hash = NEW.first_snapshot_hash;
    IF NOT FOUND
       OR candidate_row.mapping_hash IS DISTINCT FROM NEW.mapping_hash
       OR candidate_row.network_genesis_hash IS DISTINCT FROM
          NEW.network_genesis_hash
       OR candidate_row.netuid IS DISTINCT FROM NEW.netuid
       OR candidate_row.current_block IS DISTINCT FROM NEW.cutover_block
       OR candidate_row.last_epoch_block IS DISTINCT FROM NEW.cutover_block
       OR candidate_row.block_hash IS DISTINCT FROM NEW.cutover_block_hash
       OR candidate_row.subnet_epoch_index IS DISTINCT FROM
          NEW.first_subnet_epoch_index
       OR candidate_row.epoch_ref IS DISTINCT FROM NEW.first_epoch_ref
       OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR candidate_row.chain_state_receipt_hash IS DISTINCT FROM
          NEW.first_snapshot_receipt_hash
       OR candidate_row.snapshot_doc IS DISTINCT FROM NEW.first_snapshot_doc
    THEN
        RAISE EXCEPTION 'fresh-network first snapshot differs';
    END IF;

    SELECT role, purpose, epoch_id, receipt_status, output_root, receipt_doc
    INTO coordinator_receipt
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = NEW.cutover_receipt_hash;
    IF NOT FOUND
       OR coordinator_receipt.role IS DISTINCT FROM 'gateway_coordinator'
       OR coordinator_receipt.purpose IS DISTINCT FROM
          'research_lab.subnet_epoch_cutover.v2'
       OR coordinator_receipt.epoch_id IS DISTINCT FROM
          NEW.first_settlement_epoch_id
       OR coordinator_receipt.receipt_status IS DISTINCT FROM 'succeeded'
       OR coordinator_receipt.output_root IS DISTINCT FROM
          NEW.cutover_authority_hash
       OR coordinator_receipt.receipt_doc->>'receipt_hash' IS DISTINCT FROM
          NEW.cutover_receipt_hash
       OR coordinator_receipt.receipt_doc->'parent_receipt_hashes' IS DISTINCT FROM
          pg_catalog.jsonb_build_array(NEW.first_snapshot_receipt_hash)
    THEN
        RAISE EXCEPTION 'fresh-network coordinator receipt is invalid';
    END IF;

    SELECT pg_catalog.count(*) INTO edge_count
    FROM public.research_lab_attested_receipt_edges_v2
    WHERE child_receipt_hash = NEW.cutover_receipt_hash;
    IF edge_count IS DISTINCT FROM 1
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_receipt_edges_v2
           WHERE child_receipt_hash = NEW.cutover_receipt_hash
             AND parent_receipt_hash = NEW.first_snapshot_receipt_hash
       )
    THEN
        RAISE EXCEPTION 'fresh-network receipt ancestry is invalid';
    END IF;

    RETURN NEW;
END;
$$;

-- Migration 101 attaches the global Finney fence to these relations. Keep the
-- same trigger function for every old row, but let the exact testnet401
-- candidate and v3 cutover reach their receipt validators instead of comparing
-- them with the unrelated terminal Finney singleton.
DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR EACH ROW
    WHEN (
        NEW.network_genesis_hash <>
            '0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105'
        OR NEW.netuid <> 401
    )
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    BEFORE INSERT OR UPDATE
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW
    WHEN (NEW.schema_version <> 'leadpoet.subnet_epoch_cutover_authority.v3')
    EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_legacy_v3
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
DROP TRIGGER IF EXISTS validate_research_lab_fresh_network_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;

CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_legacy_v3
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW
    WHEN (NEW.schema_version <> 'leadpoet.subnet_epoch_cutover_authority.v3')
    EXECUTE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2();

CREATE TRIGGER validate_research_lab_fresh_network_epoch_cutover_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW
    WHEN (NEW.schema_version = 'leadpoet.subnet_epoch_cutover_authority.v3')
    EXECUTE FUNCTION public.validate_research_lab_fresh_network_epoch_cutover_v1();

CREATE OR REPLACE FUNCTION
public.research_lab_fresh_network_epoch_cutover_public_state_v1(
    p_network_genesis_hash TEXT,
    p_netuid INTEGER
)
RETURNS TABLE (
    lifecycle_state TEXT,
    mapping_hash TEXT,
    network_genesis_hash TEXT,
    netuid INTEGER,
    last_legacy_epoch_id INTEGER,
    first_settlement_epoch_id INTEGER
)
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT
        'stateful_active'::TEXT,
        cutover.mapping_hash,
        cutover.network_genesis_hash,
        cutover.netuid,
        cutover.last_legacy_epoch_id,
        cutover.first_settlement_epoch_id
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS cutover
    WHERE cutover.schema_version =
          'leadpoet.subnet_epoch_cutover_authority.v3'
      AND cutover.network_genesis_hash = p_network_genesis_hash
      AND cutover.netuid = p_netuid
$$;

REVOKE ALL ON FUNCTION
    public.validate_research_lab_fresh_network_epoch_cutover_v1()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION
    public.research_lab_fresh_network_epoch_cutover_public_state_v1(TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_fresh_network_epoch_cutover_public_state_v1(TEXT, INTEGER)
    TO anon, authenticated, service_role;

COMMENT ON FUNCTION
    public.research_lab_fresh_network_epoch_cutover_public_state_v1(TEXT, INTEGER)
IS 'Read-only keyed authority for a fresh non-Finney network. The Finney/SN71 singleton remains the only production namespace lifecycle authority.';

NOTIFY pgrst, 'reload schema';

COMMIT;
