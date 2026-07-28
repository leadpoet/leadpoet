-- Fail-closed chain continuity for epochs without finalized V2 bundle proof.
--
-- A V2 unattributed marker proves the exact primary chain vector but grants
-- zero Research Lab credit. Canonical finalized-bundle settlements retain the
-- existing V1 path unchanged.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid =
              'public.research_lab_chain_realized_epoch_settlements_v1'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%schema_version%'
          AND pg_get_constraintdef(oid) LIKE
              '%leadpoet.research_lab_chain_realized_epoch_settlement.v1%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1 '
            || 'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ADD CONSTRAINT research_lab_chain_settlement_schema_check
    CHECK (
        schema_version IN (
            'leadpoet.research_lab_chain_realized_epoch_settlement.v1',
            'leadpoet.research_lab_chain_realized_epoch_settlement.v2'
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    VALIDATE CONSTRAINT research_lab_chain_settlement_schema_check;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    DROP CONSTRAINT IF EXISTS research_lab_chain_unattributed_empty_check;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ADD CONSTRAINT research_lab_chain_unattributed_empty_check
    CHECK (
        schema_version <>
            'leadpoet.research_lab_chain_realized_epoch_settlement.v2'
        OR settlement_doc->'credit_hashes' = '[]'::JSONB
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    VALIDATE CONSTRAINT research_lab_chain_unattributed_empty_check;

CREATE OR REPLACE FUNCTION
public.persist_research_lab_chain_realized_unattributed_v2(
    requested_settlement JSONB,
    requested_credits JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    settlement_netuid INTEGER;
    settlement_epoch INTEGER;
    activation_first_epoch INTEGER;
    requested_settlement_hash TEXT;
    requested_settlement_receipt_hash TEXT;
    requested_settlement_doc JSONB;
    stored_settlement
        public.research_lab_chain_realized_epoch_settlements_v1;
BEGIN
    IF pg_catalog.jsonb_typeof(requested_settlement) <> 'object'
       OR requested_settlement <> requested_settlement - 'created_at'
       OR requested_settlement->>'schema_version' <>
          'leadpoet.research_lab_chain_realized_epoch_settlement.v2'
       OR pg_catalog.jsonb_typeof(requested_credits) <> 'array'
       OR pg_catalog.jsonb_array_length(requested_credits) <> 0 THEN
        RAISE EXCEPTION 'chain_realized_unattributed_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    BEGIN
        settlement_netuid := (requested_settlement->>'netuid')::INTEGER;
        settlement_epoch := (requested_settlement->>'epoch_id')::INTEGER;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'chain_realized_unattributed_scope_invalid'
            USING ERRCODE = '22023';
    END;
    requested_settlement_hash :=
        requested_settlement->>'settlement_hash';
    requested_settlement_receipt_hash :=
        requested_settlement->>'settlement_receipt_hash';
    requested_settlement_doc := requested_settlement->'settlement_doc';
    IF settlement_netuid <= 0
       OR settlement_epoch < 0
       OR requested_settlement_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_settlement_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(requested_settlement_doc) <> 'object'
       OR requested_settlement_doc->>'schema_version' <>
          'leadpoet.research_lab_chain_realized_epoch_settlement.v2'
       OR (requested_settlement_doc->>'netuid')::INTEGER <>
          settlement_netuid
       OR (requested_settlement_doc->>'epoch_id')::INTEGER <>
          settlement_epoch
       OR requested_settlement_doc->'credit_hashes' <> '[]'::JSONB THEN
        RAISE EXCEPTION 'chain_realized_unattributed_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('chain_realized_settlement_v1'),
        settlement_netuid
    );

    SELECT first_epoch_id
    INTO activation_first_epoch
    FROM public.research_lab_chain_realized_settlement_activation_v1
    WHERE netuid = settlement_netuid;
    IF activation_first_epoch IS NULL
       OR settlement_epoch < activation_first_epoch THEN
        RAISE EXCEPTION 'chain_realized_settlement_activation_invalid'
            USING ERRCODE = '55000';
    END IF;
    IF settlement_epoch > activation_first_epoch
       AND NOT EXISTS (
           SELECT 1
           FROM public.research_lab_chain_realized_epoch_settlements_v1
           WHERE netuid = settlement_netuid
             AND epoch_id = settlement_epoch - 1
       ) THEN
        RAISE EXCEPTION 'chain_realized_settlement_predecessor_missing'
            USING ERRCODE = '55000';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = requested_settlement_receipt_hash
          AND role = 'gateway_coordinator'
          AND purpose = 'research_lab.chain_realized_epoch_settlement.v1'
          AND epoch_id = settlement_epoch
          AND output_root = requested_settlement_hash
          AND receipt_status = 'succeeded'
    ) THEN
        RAISE EXCEPTION 'chain_realized_settlement_receipt_invalid'
            USING ERRCODE = '55000';
    END IF;

    INSERT INTO public.research_lab_chain_realized_epoch_settlements_v1 (
        netuid,
        epoch_id,
        schema_version,
        settlement_hash,
        settlement_receipt_hash,
        settlement_doc
    ) VALUES (
        settlement_netuid,
        settlement_epoch,
        requested_settlement->>'schema_version',
        requested_settlement_hash,
        requested_settlement_receipt_hash,
        requested_settlement_doc
    )
    ON CONFLICT DO NOTHING;

    SELECT *
    INTO stored_settlement
    FROM public.research_lab_chain_realized_epoch_settlements_v1
    WHERE netuid = settlement_netuid
      AND epoch_id = settlement_epoch;
    IF stored_settlement.netuid IS NULL
       OR stored_settlement.schema_version IS DISTINCT FROM
          requested_settlement->>'schema_version'
       OR stored_settlement.settlement_hash IS DISTINCT FROM
          requested_settlement_hash
       OR stored_settlement.settlement_receipt_hash IS DISTINCT FROM
          requested_settlement_receipt_hash
       OR stored_settlement.settlement_doc IS DISTINCT FROM
          requested_settlement_doc
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_chain_realized_obligation_credits_v1
           WHERE netuid = settlement_netuid
             AND epoch_id = settlement_epoch
       ) THEN
        RAISE EXCEPTION 'chain_realized_unattributed_conflict'
            USING ERRCODE = '40001';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_chain_realized_settlement_persistence.v1',
        'netuid', settlement_netuid,
        'epoch_id', settlement_epoch,
        'settlement_hash', requested_settlement_hash,
        'settlement_receipt_hash', requested_settlement_receipt_hash,
        'credit_count', 0,
        'credit_hashes', '[]'::JSONB
    );
END;
$$;

REVOKE ALL
    ON FUNCTION
    public.persist_research_lab_chain_realized_unattributed_v2(JSONB, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION
    public.persist_research_lab_chain_realized_unattributed_v2(JSONB, JSONB)
    TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
