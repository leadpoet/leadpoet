-- Prospective chain-realized lifetime credit for accelerated champions.
--
-- Historical V1 credit rows and their persistence RPC remain unchanged.
-- Marked allocations use finalized settlement V3 and obligation credit V2;
-- actual chain-realized champion attribution is retained so deterministic
-- replay can cap applied lifetime credit and expose unavoidable stale excess.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    ADD COLUMN IF NOT EXISTS champion_credit_policy TEXT
    NOT NULL DEFAULT 'scheduled_bonus_v1';

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_chain_realized_epoch_settlements_v1'
                  ::pg_catalog.regclass
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE '%schema_version%'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE
              '%leadpoet.research_lab_chain_realized_epoch_settlement.v1%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1 '
            || 'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;

    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_chain_realized_obligation_credits_v1'
                  ::pg_catalog.regclass
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE
              '%leadpoet.research_lab_chain_realized_obligation_credit.v1%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1 '
            || 'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;

    FOR item IN
        SELECT conname
        FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'public.research_lab_chain_realized_obligation_credits_v1'
                  ::pg_catalog.regclass
          AND contype = 'c'
          AND pg_catalog.pg_get_constraintdef(oid) LIKE
              '%credited_alpha_percent <= scheduled_alpha_percent%'
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1 '
            || 'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    DROP CONSTRAINT IF EXISTS research_lab_chain_settlement_schema_check;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ADD CONSTRAINT research_lab_chain_settlement_schema_check
    CHECK (
        schema_version IN (
            'leadpoet.research_lab_chain_realized_epoch_settlement.v1',
            'leadpoet.research_lab_chain_realized_epoch_settlement.v2',
            'leadpoet.research_lab_chain_realized_epoch_settlement.v3'
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    VALIDATE CONSTRAINT research_lab_chain_settlement_schema_check;

ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    DROP CONSTRAINT IF EXISTS
        research_lab_chain_settlement_champion_policy_check;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    ADD CONSTRAINT research_lab_chain_settlement_champion_policy_check
    CHECK (
        (
            schema_version =
                'leadpoet.research_lab_chain_realized_epoch_settlement.v3'
            AND settlement_doc->>'champion_credit_policy' =
                'accelerated_lifetime_cap_v1'
        )
        OR (
            schema_version <>
                'leadpoet.research_lab_chain_realized_epoch_settlement.v3'
            AND NOT (settlement_doc ? 'champion_credit_policy')
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_epoch_settlements_v1
    VALIDATE CONSTRAINT
        research_lab_chain_settlement_champion_policy_check;

ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    DROP CONSTRAINT IF EXISTS
        research_lab_chain_credit_schema_policy_check;
ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    ADD CONSTRAINT research_lab_chain_credit_schema_policy_check
    CHECK (
        (
            schema_version =
                'leadpoet.research_lab_chain_realized_obligation_credit.v1'
            AND champion_credit_policy = 'scheduled_bonus_v1'
            AND NOT (credit_doc ? 'champion_credit_policy')
        )
        OR (
            schema_version =
                'leadpoet.research_lab_chain_realized_obligation_credit.v2'
            AND champion_credit_policy = 'accelerated_lifetime_cap_v1'
            AND credit_doc->>'champion_credit_policy' =
                champion_credit_policy
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    VALIDATE CONSTRAINT research_lab_chain_credit_schema_policy_check;

ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    DROP CONSTRAINT IF EXISTS
        research_lab_chain_credit_policy_amount_check;
ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    ADD CONSTRAINT research_lab_chain_credit_policy_amount_check
    CHECK (
        (
            schema_version =
                'leadpoet.research_lab_chain_realized_obligation_credit.v2'
            AND obligation_kind IN ('champion', 'queued_champion')
            AND credited_alpha_percent = lab_attributed_alpha_percent
        )
        OR (
            NOT (
                schema_version =
                    'leadpoet.research_lab_chain_realized_obligation_credit.v2'
                AND obligation_kind IN ('champion', 'queued_champion')
            )
            AND (
                scheduled_alpha_percent = 0
                OR credited_alpha_percent <= scheduled_alpha_percent
            )
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_chain_realized_obligation_credits_v1
    VALIDATE CONSTRAINT research_lab_chain_credit_policy_amount_check;

CREATE OR REPLACE FUNCTION
public.persist_research_lab_chain_realized_lifetime_settlement_v2(
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
    requested_settlement_hash TEXT;
    requested_settlement_receipt_hash TEXT;
    requested_settlement_doc JSONB;
    expected_credit_hashes JSONB;
    requested_credit_count INTEGER;
    requested_unique_credit_count INTEGER;
    stored_credit_hashes JSONB;
    activation_first_epoch INTEGER;
    item JSONB;
    stored_settlement
        public.research_lab_chain_realized_epoch_settlements_v1;
BEGIN
    IF pg_catalog.jsonb_typeof(requested_settlement) <> 'object'
       OR pg_catalog.jsonb_typeof(requested_credits) <> 'array'
       OR requested_settlement <> requested_settlement - 'created_at' THEN
        RAISE EXCEPTION 'chain_realized_lifetime_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    BEGIN
        settlement_netuid :=
            (requested_settlement->>'netuid')::INTEGER;
        settlement_epoch :=
            (requested_settlement->>'epoch_id')::INTEGER;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'chain_realized_lifetime_scope_invalid'
            USING ERRCODE = '22023';
    END;
    requested_settlement_hash := requested_settlement->>'settlement_hash';
    requested_settlement_receipt_hash :=
        requested_settlement->>'settlement_receipt_hash';
    requested_settlement_doc := requested_settlement->'settlement_doc';
    expected_credit_hashes := requested_settlement_doc->'credit_hashes';
    IF settlement_netuid <= 0
       OR settlement_epoch < 0
       OR requested_settlement->>'schema_version' <>
          'leadpoet.research_lab_chain_realized_epoch_settlement.v3'
       OR requested_settlement_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_settlement_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(requested_settlement_doc) <> 'object'
       OR requested_settlement_doc->>'schema_version' <>
          'leadpoet.research_lab_chain_realized_epoch_settlement.v3'
       OR requested_settlement_doc->>'champion_credit_policy' <>
          'accelerated_lifetime_cap_v1'
       OR pg_catalog.jsonb_typeof(expected_credit_hashes) <> 'array' THEN
        RAISE EXCEPTION 'chain_realized_lifetime_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    SELECT
        pg_catalog.jsonb_array_length(requested_credits),
        pg_catalog.count(DISTINCT value->>'credit_hash')
    INTO requested_credit_count, requested_unique_credit_count
    FROM pg_catalog.jsonb_array_elements(requested_credits);
    IF requested_credit_count > 1000
       OR requested_credit_count <> requested_unique_credit_count
       OR requested_credit_count <>
          pg_catalog.jsonb_array_length(expected_credit_hashes) THEN
        RAISE EXCEPTION 'chain_realized_lifetime_credit_set_invalid'
            USING ERRCODE = '22023';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(requested_credits) AS credit(value)
        WHERE pg_catalog.jsonb_typeof(value) <> 'object'
           OR (value->>'netuid')::INTEGER IS DISTINCT FROM settlement_netuid
           OR (value->>'epoch_id')::INTEGER IS DISTINCT FROM settlement_epoch
           OR value->>'settlement_hash'
              IS DISTINCT FROM requested_settlement_hash
           OR value->>'credit_receipt_hash'
              IS DISTINCT FROM requested_settlement_receipt_hash
           OR value->>'schema_version' <>
              'leadpoet.research_lab_chain_realized_obligation_credit.v2'
           OR value->>'champion_credit_policy' <>
              'accelerated_lifetime_cap_v1'
           OR value->'credit_doc'->>'champion_credit_policy' <>
              'accelerated_lifetime_cap_v1'
           OR NOT (expected_credit_hashes ? (value->>'credit_hash'))
    ) THEN
        RAISE EXCEPTION 'chain_realized_lifetime_credit_scope_invalid'
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
       OR stored_settlement.settlement_hash
          IS DISTINCT FROM requested_settlement_hash
       OR stored_settlement.settlement_receipt_hash IS DISTINCT FROM
          requested_settlement_receipt_hash
       OR stored_settlement.settlement_doc
          IS DISTINCT FROM requested_settlement_doc THEN
        RAISE EXCEPTION 'chain_realized_settlement_conflict'
            USING ERRCODE = '40001';
    END IF;

    FOR item IN
        SELECT value
        FROM pg_catalog.jsonb_array_elements(requested_credits)
        ORDER BY value->>'credit_hash'
    LOOP
        INSERT INTO
        public.research_lab_chain_realized_obligation_credits_v1 (
            netuid,
            epoch_id,
            settlement_hash,
            schema_version,
            obligation_kind,
            obligation_source_id,
            miner_hotkey,
            miner_uid,
            observed_chain_alpha_percent,
            lab_attributed_alpha_percent,
            scheduled_alpha_percent,
            credited_alpha_percent,
            champion_credit_policy,
            credit_hash,
            credit_receipt_hash,
            credit_doc
        ) VALUES (
            settlement_netuid,
            settlement_epoch,
            requested_settlement_hash,
            item->>'schema_version',
            item->>'obligation_kind',
            item->>'obligation_source_id',
            item->>'miner_hotkey',
            (item->>'miner_uid')::INTEGER,
            (item->>'observed_chain_alpha_percent')::NUMERIC,
            (item->>'lab_attributed_alpha_percent')::NUMERIC,
            (item->>'scheduled_alpha_percent')::NUMERIC,
            (item->>'credited_alpha_percent')::NUMERIC,
            item->>'champion_credit_policy',
            item->>'credit_hash',
            item->>'credit_receipt_hash',
            item->'credit_doc'
        )
        ON CONFLICT DO NOTHING;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(requested_credits) AS credit(value)
        LEFT JOIN
            public.research_lab_chain_realized_obligation_credits_v1 stored
          ON stored.netuid = settlement_netuid
         AND stored.epoch_id = settlement_epoch
         AND stored.obligation_kind = value->>'obligation_kind'
         AND stored.obligation_source_id =
             value->>'obligation_source_id'
        WHERE stored.netuid IS NULL
           OR stored.settlement_hash
              IS DISTINCT FROM requested_settlement_hash
           OR stored.schema_version IS DISTINCT FROM value->>'schema_version'
           OR stored.miner_hotkey IS DISTINCT FROM value->>'miner_hotkey'
           OR stored.miner_uid IS DISTINCT FROM
              (value->>'miner_uid')::INTEGER
           OR stored.observed_chain_alpha_percent IS DISTINCT FROM
              (value->>'observed_chain_alpha_percent')::NUMERIC
           OR stored.lab_attributed_alpha_percent IS DISTINCT FROM
              (value->>'lab_attributed_alpha_percent')::NUMERIC
           OR stored.scheduled_alpha_percent IS DISTINCT FROM
              (value->>'scheduled_alpha_percent')::NUMERIC
           OR stored.credited_alpha_percent IS DISTINCT FROM
              (value->>'credited_alpha_percent')::NUMERIC
           OR stored.champion_credit_policy IS DISTINCT FROM
              value->>'champion_credit_policy'
           OR stored.credit_hash IS DISTINCT FROM value->>'credit_hash'
           OR stored.credit_receipt_hash IS DISTINCT FROM
              value->>'credit_receipt_hash'
           OR stored.credit_doc IS DISTINCT FROM value->'credit_doc'
    ) THEN
        RAISE EXCEPTION 'chain_realized_lifetime_credit_conflict'
            USING ERRCODE = '40001';
    END IF;

    SELECT COALESCE(
        pg_catalog.jsonb_agg(credit_hash ORDER BY credit_hash),
        '[]'::JSONB
    )
    INTO stored_credit_hashes
    FROM public.research_lab_chain_realized_obligation_credits_v1
    WHERE netuid = settlement_netuid
      AND epoch_id = settlement_epoch
      AND settlement_hash = requested_settlement_hash;
    IF stored_credit_hashes IS DISTINCT FROM (
        SELECT COALESCE(
            pg_catalog.jsonb_agg(value ORDER BY value),
            '[]'::JSONB
        )
        FROM pg_catalog.jsonb_array_elements_text(expected_credit_hashes)
    ) THEN
        RAISE EXCEPTION 'chain_realized_lifetime_credit_set_conflict'
            USING ERRCODE = '40001';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_chain_realized_settlement_persistence.v1',
        'netuid',
        settlement_netuid,
        'epoch_id',
        settlement_epoch,
        'settlement_hash',
        requested_settlement_hash,
        'settlement_receipt_hash',
        requested_settlement_receipt_hash,
        'credit_count',
        requested_credit_count,
        'credit_hashes',
        stored_credit_hashes
    );
END;
$$;

REVOKE ALL
    ON FUNCTION
    public.persist_research_lab_chain_realized_lifetime_settlement_v2(
        JSONB,
        JSONB
    )
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE
    ON FUNCTION
    public.persist_research_lab_chain_realized_lifetime_settlement_v2(
        JSONB,
        JSONB
    )
    TO service_role;

CREATE OR REPLACE FUNCTION
public.research_lab_champion_lifetime_credit_contract_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab_champion_lifetime_credit_contract.v1',
        'champion_credit_policy',
        'accelerated_lifetime_cap_v1',
        'credit_policy_column',
        EXISTS (
            SELECT 1
            FROM pg_catalog.pg_attribute
            WHERE attrelid =
                  'public.research_lab_chain_realized_obligation_credits_v1'
                      ::pg_catalog.regclass
              AND attname = 'champion_credit_policy'
              AND NOT attisdropped
        ),
        'constraints',
        (
            SELECT pg_catalog.jsonb_object_agg(
                conname,
                pg_catalog.jsonb_build_object(
                    'validated',
                    convalidated,
                    'definition',
                    pg_catalog.pg_get_constraintdef(oid)
                )
                ORDER BY conname
            )
            FROM pg_catalog.pg_constraint
            WHERE conrelid IN (
                'public.research_lab_chain_realized_epoch_settlements_v1'
                    ::pg_catalog.regclass,
                'public.research_lab_chain_realized_obligation_credits_v1'
                    ::pg_catalog.regclass
            )
              AND conname IN (
                  'research_lab_chain_settlement_schema_check',
                  'research_lab_chain_settlement_champion_policy_check',
                  'research_lab_chain_credit_schema_policy_check',
                  'research_lab_chain_credit_policy_amount_check'
              )
        )
    );
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_champion_lifetime_credit_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_champion_lifetime_credit_contract_v1()
    TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
