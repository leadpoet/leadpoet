-- Allow an unactivated cutover fence to bind a newer official boundary.
--
-- The integer settlement plan remains immutable once fenced. A boundary
-- candidate can nevertheless expire while the fence stays closed across
-- operator retries. This RPC clears only the stale, unactivated binding
-- pointers after proving that no cutover row or initialization was staged.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION
public.research_lab_stateful_subnet_epoch_refresh_fence_v1(
    p_network_genesis_hash TEXT,
    p_netuid INTEGER,
    p_last_legacy_epoch_id INTEGER,
    p_first_settlement_epoch_id INTEGER
)
RETURNS TABLE (
    lifecycle_state TEXT,
    network_genesis_hash TEXT,
    netuid INTEGER,
    legacy_high_water BIGINT,
    last_legacy_epoch_id BIGINT,
    first_settlement_epoch_id BIGINT,
    first_settlement_occupied BOOLEAN,
    fenced_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
SET lock_timeout = '5s'
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
BEGIN
    IF p_network_genesis_hash !~ '^0x[0-9a-f]{64}$'
       OR p_netuid <= 0
       OR p_last_legacy_epoch_id < 0
       OR p_first_settlement_epoch_id::BIGINT IS DISTINCT FROM
          p_last_legacy_epoch_id::BIGINT + 1 THEN
        RAISE EXCEPTION 'stateful epoch refresh fence input is invalid';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);

    -- Preserve the original schema/trigger/high-water validation and initial
    -- reservation behavior. The advisory lock is transaction-reentrant.
    PERFORM *
    FROM public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
        p_network_genesis_hash,
        p_netuid,
        p_last_legacy_epoch_id,
        p_first_settlement_epoch_id
    );

    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 AS state
    WHERE state.singleton
    FOR UPDATE;

    IF NOT FOUND
       OR state_row.lifecycle_state IS DISTINCT FROM 'cutover_fenced'
       OR state_row.network_genesis_hash IS DISTINCT FROM p_network_genesis_hash
       OR state_row.netuid IS DISTINCT FROM p_netuid
       OR state_row.last_legacy_epoch_id IS DISTINCT FROM p_last_legacy_epoch_id
       OR state_row.first_settlement_epoch_id IS DISTINCT FROM
          p_first_settlement_epoch_id THEN
        RAISE EXCEPTION 'stateful epoch refresh fence conflicts with durable plan';
    END IF;

    IF state_row.mapping_hash IS NOT NULL THEN
        IF state_row.initialization_nonce IS NOT NULL
           OR state_row.initialization_payload_hash IS NOT NULL
           OR state_row.staged_at IS NOT NULL
           OR state_row.activated_at IS NOT NULL
           OR EXISTS (
               SELECT 1
               FROM public.research_lab_stateful_subnet_epoch_cutovers_v1
           )
           OR EXISTS (
               SELECT 1
               FROM public.transparency_log AS event
               WHERE event.event_type = 'EPOCH_INITIALIZATION'
                 AND pg_catalog.jsonb_typeof(event.payload) = 'object'
                 AND event.payload->>'epoch_id' ~ '^[0-9]+$'
                 AND (event.payload->>'epoch_id')::BIGINT =
                     p_first_settlement_epoch_id::BIGINT
           ) THEN
            RAISE EXCEPTION
                'stateful epoch refresh fence refuses a staged or initialized cutover';
        END IF;

        UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1 AS state
        SET mapping_hash = NULL,
            candidate_snapshot_hash = NULL,
            candidate_receipt_hash = NULL,
            last_legacy_finalization_receipt_hash = NULL,
            cutover_authority_hash = NULL,
            cutover_receipt_hash = NULL,
            updated_at = pg_catalog.clock_timestamp()
        WHERE state.singleton
        RETURNING state.* INTO state_row;
    END IF;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.network_genesis_hash,
        state_row.netuid,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        FALSE,
        state_row.fenced_at;
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_stateful_subnet_epoch_refresh_fence_v1(
        TEXT, INTEGER, INTEGER, INTEGER
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_stateful_subnet_epoch_refresh_fence_v1(
        TEXT, INTEGER, INTEGER, INTEGER
    ) TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
