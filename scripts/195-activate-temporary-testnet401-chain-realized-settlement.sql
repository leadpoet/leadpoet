-- Activate chain-realized settlement from the first genuine finalized
-- compact testnet401 authority by the expected validator.
--
-- This temporary bridge copies one real compact authority into the immutable
-- migration-126 activation contract. A retry accepts only the same source.

BEGIN;

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.research_lab_chain_realized_settlement_activation_v1
    IN SHARE ROW EXCLUSIVE MODE;

DO $activation$
DECLARE
    expected_validator_hotkey CONSTANT TEXT :=
        '5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz';
    first_epoch BIGINT;
    first_epoch_rows BIGINT;
    source_row RECORD;
    existing_row RECORD;
BEGIN
    SELECT MIN(authority.epoch_id)
    INTO first_epoch
    FROM public.research_lab_compact_weight_authorities_v2 authority
    WHERE authority.netuid = 401
      AND authority.validator_hotkey = expected_validator_hotkey
      AND authority.authority_stage = 'finalized';

    IF first_epoch IS NULL THEN
        RAISE EXCEPTION 'testnet401_first_finalized_compact_authority_unavailable';
    END IF;

    SELECT COUNT(*)
    INTO first_epoch_rows
    FROM public.research_lab_compact_weight_authorities_v2 authority
    WHERE authority.netuid = 401
      AND authority.validator_hotkey = expected_validator_hotkey
      AND authority.authority_stage = 'finalized'
      AND authority.epoch_id = first_epoch;

    IF first_epoch_rows <> 1 THEN
        RAISE EXCEPTION 'testnet401_first_finalized_compact_authority_ambiguous';
    END IF;

    SELECT
        authority.bundle_hash,
        authority.epoch_id,
        (authority.authority_doc #>>
            '{finalization,compact_submission,finalization,finalized_block}'
        )::BIGINT AS finalized_block
    INTO STRICT source_row
    FROM public.research_lab_compact_weight_authorities_v2 authority
    WHERE authority.netuid = 401
      AND authority.validator_hotkey = expected_validator_hotkey
      AND authority.authority_stage = 'finalized'
      AND authority.epoch_id = first_epoch;

    SELECT activation.*
    INTO existing_row
    FROM public.research_lab_chain_realized_settlement_activation_v1 activation
    WHERE activation.netuid = 401;

    IF FOUND THEN
        IF existing_row.schema_version IS DISTINCT FROM
               'leadpoet.research_lab_chain_realized_settlement_activation.v1'
           OR existing_row.first_epoch_id IS DISTINCT FROM source_row.epoch_id
           OR existing_row.source_bundle_hash IS DISTINCT FROM source_row.bundle_hash
           OR existing_row.source_bundle_epoch_id IS DISTINCT FROM source_row.epoch_id
           OR existing_row.source_finalized_block IS DISTINCT FROM
               source_row.finalized_block
        THEN
            RAISE EXCEPTION 'testnet401_settlement_activation_conflicts';
        END IF;
        RETURN;
    END IF;

    INSERT INTO
    public.research_lab_chain_realized_settlement_activation_v1 (
        netuid, schema_version, first_epoch_id, source_bundle_hash,
        source_bundle_epoch_id, source_finalized_block
    )
    SELECT
        authority.netuid,
        'leadpoet.research_lab_chain_realized_settlement_activation.v1',
        authority.epoch_id,
        authority.bundle_hash,
        authority.epoch_id,
        (authority.authority_doc #>>
            '{finalization,compact_submission,finalization,finalized_block}'
        )::BIGINT
    FROM public.research_lab_compact_weight_authorities_v2 authority
    WHERE authority.netuid = 401
      AND authority.validator_hotkey = expected_validator_hotkey
      AND authority.authority_stage = 'finalized'
      AND authority.epoch_id = first_epoch
      AND authority.bundle_hash = source_row.bundle_hash;

    SELECT activation.*
    INTO STRICT existing_row
    FROM public.research_lab_chain_realized_settlement_activation_v1 activation
    WHERE activation.netuid = 401;

    IF existing_row.first_epoch_id IS DISTINCT FROM source_row.epoch_id
       OR existing_row.source_bundle_hash IS DISTINCT FROM source_row.bundle_hash
       OR existing_row.source_bundle_epoch_id IS DISTINCT FROM source_row.epoch_id
       OR existing_row.source_finalized_block IS DISTINCT FROM source_row.finalized_block
    THEN
        RAISE EXCEPTION 'testnet401_settlement_activation_readback_invalid';
    END IF;
END;
$activation$;

COMMIT;
