-- Bounded, signed allocation settlement frontier.
--
-- Historical settlement and receipt rows remain append-only.  This index
-- points to an already durable allocation execution result whose source state
-- contains the cumulative active-obligation checkpoint.  The persistence RPC
-- binds every frontier hash to that execution receipt and enforces one
-- monotonic predecessor chain per subnet.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS
public.research_lab_allocation_settlement_frontiers_v2 (
    netuid                    INTEGER NOT NULL CHECK (netuid > 0),
    allocation_epoch          BIGINT NOT NULL CHECK (allocation_epoch >= 1),
    settled_through_epoch     BIGINT NOT NULL CHECK (
        settled_through_epoch = allocation_epoch - 1
    ),
    schema_version            TEXT NOT NULL CHECK (
        schema_version =
            'leadpoet.research_lab_allocation_settlement_frontier.v2'
    ),
    frontier_hash             TEXT NOT NULL UNIQUE CHECK (
        frontier_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    predecessor_frontier_hash TEXT NULL REFERENCES
        public.research_lab_allocation_settlement_frontiers_v2(frontier_hash)
        ON DELETE RESTRICT,
    source_receipt_hash       TEXT NOT NULL REFERENCES
        public.research_lab_attested_execution_results_v2(receipt_hash)
        ON DELETE RESTRICT,
    source_state_hash         TEXT NOT NULL CHECK (
        source_state_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    frontier_doc              JSONB NOT NULL CHECK (
        jsonb_typeof(frontier_doc) = 'object'
        AND frontier_doc::TEXT !~*
            '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, allocation_epoch),
    CHECK (frontier_doc->>'schema_version' = schema_version),
    CHECK ((frontier_doc->>'netuid')::INTEGER = netuid),
    CHECK ((frontier_doc->>'allocation_epoch')::BIGINT = allocation_epoch),
    CHECK (
        (frontier_doc->>'settled_through_epoch')::BIGINT =
            settled_through_epoch
    ),
    CHECK (frontier_doc->>'frontier_hash' = frontier_hash),
    CHECK (
        frontier_doc->>'predecessor_frontier_hash'
        IS NOT DISTINCT FROM predecessor_frontier_hash
    )
);

CREATE INDEX IF NOT EXISTS
idx_research_lab_allocation_settlement_frontier_latest_v2
    ON public.research_lab_allocation_settlement_frontiers_v2(
        netuid,
        allocation_epoch DESC
    );

CREATE TABLE IF NOT EXISTS
public.research_lab_allocation_settlement_frontier_activation_v2 (
    netuid                    INTEGER PRIMARY KEY CHECK (netuid > 0),
    schema_version            TEXT NOT NULL CHECK (
        schema_version =
            'leadpoet.research_lab_allocation_settlement_frontier_activation.v2'
    ),
    first_allocation_epoch    BIGINT NOT NULL CHECK (
        first_allocation_epoch >= 1
    ),
    first_frontier_hash       TEXT NOT NULL UNIQUE REFERENCES
        public.research_lab_allocation_settlement_frontiers_v2(frontier_hash)
        ON DELETE RESTRICT,
    source_receipt_hash       TEXT NOT NULL REFERENCES
        public.research_lab_attested_execution_results_v2(receipt_hash)
        ON DELETE RESTRICT,
    activated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION
public.persist_research_lab_allocation_settlement_frontier_v2(
    requested_frontier JSONB,
    requested_source_receipt_hash TEXT,
    requested_source_state_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    requested_netuid INTEGER;
    requested_epoch BIGINT;
    requested_settled_through BIGINT;
    requested_schema TEXT;
    requested_mode TEXT;
    requested_hash TEXT;
    requested_predecessor TEXT;
    requested_checkpoint_count INTEGER;
    observed_checkpoint_count INTEGER;
    execution_found BOOLEAN;
    receipt_found BOOLEAN;
    execution_row public.research_lab_attested_execution_results_v2;
    receipt_row public.research_lab_attested_execution_receipts_v2;
    existing_row public.research_lab_allocation_settlement_frontiers_v2;
    previous_row public.research_lab_allocation_settlement_frontiers_v2;
    first_row public.research_lab_allocation_settlement_frontiers_v2;
    activation_row
        public.research_lab_allocation_settlement_frontier_activation_v2;
BEGIN
    IF requested_frontier IS NULL
       OR pg_catalog.jsonb_typeof(requested_frontier) IS DISTINCT FROM 'object'
       OR (
           SELECT pg_catalog.array_agg(key ORDER BY key)
             FROM pg_catalog.jsonb_object_keys(requested_frontier) AS key
       ) <> ARRAY[
           'allocation_epoch',
           'frontier_hash',
           'mode',
           'netuid',
           'predecessor_frontier_hash',
           'reward_checkpoint_count',
           'reward_checkpoint_hashes_root',
           'reward_checkpoints',
           'schema_version',
           'settled_through_epoch'
       ]::TEXT[] THEN
        RAISE EXCEPTION 'allocation_settlement_frontier_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    BEGIN
        requested_netuid := (requested_frontier->>'netuid')::INTEGER;
        requested_epoch :=
            (requested_frontier->>'allocation_epoch')::BIGINT;
        requested_settled_through :=
            (requested_frontier->>'settled_through_epoch')::BIGINT;
        requested_checkpoint_count :=
            (requested_frontier->>'reward_checkpoint_count')::INTEGER;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'allocation_settlement_frontier_scope_invalid'
            USING ERRCODE = '22023';
    END;
    requested_schema := requested_frontier->>'schema_version';
    requested_mode := requested_frontier->>'mode';
    requested_hash := requested_frontier->>'frontier_hash';
    requested_predecessor :=
        requested_frontier->>'predecessor_frontier_hash';

    IF requested_netuid IS NULL
       OR requested_netuid <= 0
       OR requested_epoch IS NULL
       OR requested_epoch < 1
       OR requested_settled_through IS NULL
       OR requested_settled_through <> requested_epoch - 1
       OR requested_schema IS DISTINCT FROM
          'leadpoet.research_lab_allocation_settlement_frontier.v2'
       OR requested_mode IS NULL
       OR requested_mode NOT IN (
          'legacy_full_history_bootstrap',
          'bounded_delta_v1'
       )
       OR requested_hash IS NULL
       OR requested_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_source_receipt_hash IS NULL
       OR requested_source_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_source_state_hash IS NULL
       OR requested_source_state_hash !~ '^sha256:[0-9a-f]{64}$'
       OR requested_checkpoint_count IS NULL
       OR requested_checkpoint_count < 0
       OR requested_checkpoint_count > 512
       OR pg_catalog.jsonb_typeof(
          requested_frontier->'reward_checkpoints'
       ) IS DISTINCT FROM 'array'
       OR requested_frontier->>'reward_checkpoint_hashes_root' IS NULL
       OR requested_frontier->>'reward_checkpoint_hashes_root'
          !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'allocation_settlement_frontier_request_invalid'
            USING ERRCODE = '22023';
    END IF;

    SELECT pg_catalog.count(*)::INTEGER
      INTO observed_checkpoint_count
      FROM pg_catalog.jsonb_array_elements(
          requested_frontier->'reward_checkpoints'
      ) AS checkpoint;
    IF observed_checkpoint_count <> requested_checkpoint_count
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
            WHERE pg_catalog.jsonb_typeof(checkpoint) IS DISTINCT FROM 'object'
               OR checkpoint->>'schema_version' IS DISTINCT FROM
                  'leadpoet.research_lab_reward_settlement_checkpoint.v2'
               OR checkpoint->>'checkpoint_hash' IS NULL
               OR checkpoint->>'checkpoint_hash'
                  !~ '^sha256:[0-9a-f]{64}$'
       )
       OR (
           SELECT pg_catalog.count(DISTINCT checkpoint->>'checkpoint_hash')
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
       ) <> requested_checkpoint_count THEN
        RAISE EXCEPTION 'allocation_settlement_frontier_checkpoint_invalid'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        137,
        requested_netuid
    );

    SELECT * INTO execution_row
      FROM public.research_lab_attested_execution_results_v2
     WHERE receipt_hash = requested_source_receipt_hash;
    execution_found := FOUND;
    SELECT * INTO receipt_row
      FROM public.research_lab_attested_execution_receipts_v2
     WHERE receipt_hash = requested_source_receipt_hash;
    receipt_found := FOUND;
    IF NOT execution_found
       OR NOT receipt_found
       OR execution_row.role <> 'gateway_coordinator'
       OR execution_row.operation <> 'research_lab_allocation'
       OR execution_row.purpose <> 'research_lab.allocation.v2'
       OR execution_row.epoch_id <> requested_epoch
       OR execution_row.result_doc->>'source_state_hash' <>
          requested_source_state_hash
       OR execution_row.result_doc #>>
          '{source_state,settlement_frontier,frontier_hash}' <>
          requested_hash
       OR execution_row.result_doc->'source_state'->'settlement_frontier'
          <> requested_frontier
       OR receipt_row.role <> execution_row.role
       OR receipt_row.purpose <> execution_row.purpose
       OR receipt_row.job_id <> execution_row.job_id
       OR receipt_row.epoch_id <> execution_row.epoch_id
       OR receipt_row.sequence <> execution_row.sequence
       OR receipt_row.input_root <> execution_row.input_root
       OR receipt_row.output_root <> execution_row.output_root
       OR receipt_row.artifact_root <> execution_row.artifact_root
       OR receipt_row.receipt_status <> 'succeeded'
       OR NOT (execution_row.artifact_hashes ? requested_source_state_hash)
       OR NOT (execution_row.artifact_hashes ? requested_hash)
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 requested_frontier->'reward_checkpoints'
             ) AS checkpoint
            WHERE NOT (
                execution_row.artifact_hashes ?
                (checkpoint->>'checkpoint_hash')
            )
       ) THEN
        RAISE EXCEPTION 'allocation_settlement_frontier_source_invalid'
            USING ERRCODE = '23514';
    END IF;

    SELECT * INTO activation_row
      FROM public.research_lab_allocation_settlement_frontier_activation_v2
     WHERE netuid = requested_netuid;
    IF activation_row.netuid IS NOT NULL THEN
        SELECT * INTO first_row
          FROM public.research_lab_allocation_settlement_frontiers_v2
         WHERE frontier_hash = activation_row.first_frontier_hash;
        IF first_row.netuid IS NULL
           OR first_row.netuid <> requested_netuid
           OR first_row.allocation_epoch < 1
           OR first_row.allocation_epoch <>
              activation_row.first_allocation_epoch
           OR first_row.frontier_hash <>
              activation_row.first_frontier_hash
           OR first_row.source_receipt_hash <>
              activation_row.source_receipt_hash
           OR first_row.frontier_doc->>'mode' <>
              'legacy_full_history_bootstrap'
           OR first_row.predecessor_frontier_hash IS NOT NULL THEN
            RAISE EXCEPTION 'allocation_settlement_frontier_activation_invalid'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    SELECT * INTO existing_row
      FROM public.research_lab_allocation_settlement_frontiers_v2
     WHERE netuid = requested_netuid
       AND allocation_epoch = requested_epoch;
    IF FOUND THEN
        IF activation_row.netuid IS NULL THEN
            RAISE EXCEPTION 'allocation_settlement_frontier_activation_invalid'
                USING ERRCODE = '23514';
        ELSIF existing_row.frontier_hash <> requested_hash
           OR existing_row.frontier_doc <> requested_frontier
           OR existing_row.source_receipt_hash <>
              requested_source_receipt_hash
           OR existing_row.source_state_hash <> requested_source_state_hash THEN
            RAISE EXCEPTION 'allocation_settlement_frontier_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'already_persisted',
            'netuid', existing_row.netuid,
            'allocation_epoch', existing_row.allocation_epoch,
            'frontier_hash', existing_row.frontier_hash,
            'source_receipt_hash', existing_row.source_receipt_hash,
            'source_state_hash', existing_row.source_state_hash
        );
    END IF;

    SELECT * INTO previous_row
      FROM public.research_lab_allocation_settlement_frontiers_v2
     WHERE netuid = requested_netuid
     ORDER BY allocation_epoch DESC
     LIMIT 1;

    IF activation_row.netuid IS NULL THEN
        IF previous_row.netuid IS NOT NULL
           OR requested_mode <> 'legacy_full_history_bootstrap'
           OR requested_predecessor IS NOT NULL THEN
            RAISE EXCEPTION 'allocation_settlement_frontier_bootstrap_invalid'
                USING ERRCODE = '23514';
        END IF;
    ELSE
        IF previous_row.netuid IS NULL
           OR requested_mode <> 'bounded_delta_v1'
           OR requested_predecessor IS DISTINCT FROM
              previous_row.frontier_hash
           OR requested_settled_through <=
              previous_row.settled_through_epoch THEN
            RAISE EXCEPTION 'allocation_settlement_frontier_successor_invalid'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    INSERT INTO public.research_lab_allocation_settlement_frontiers_v2 (
        netuid,
        allocation_epoch,
        settled_through_epoch,
        schema_version,
        frontier_hash,
        predecessor_frontier_hash,
        source_receipt_hash,
        source_state_hash,
        frontier_doc
    ) VALUES (
        requested_netuid,
        requested_epoch,
        requested_settled_through,
        requested_schema,
        requested_hash,
        requested_predecessor,
        requested_source_receipt_hash,
        requested_source_state_hash,
        requested_frontier
    );

    IF activation_row.netuid IS NULL THEN
        INSERT INTO
            public.research_lab_allocation_settlement_frontier_activation_v2 (
                netuid,
                schema_version,
                first_allocation_epoch,
                first_frontier_hash,
                source_receipt_hash
            ) VALUES (
                requested_netuid,
                'leadpoet.research_lab_allocation_settlement_frontier_activation.v2',
                requested_epoch,
                requested_hash,
                requested_source_receipt_hash
            );
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'status', 'persisted',
        'netuid', requested_netuid,
        'allocation_epoch', requested_epoch,
        'frontier_hash', requested_hash,
        'source_receipt_hash', requested_source_receipt_hash,
        'source_state_hash', requested_source_state_hash
    );
END;
$$;

REVOKE ALL ON TABLE
    public.research_lab_allocation_settlement_frontiers_v2
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON TABLE
    public.research_lab_allocation_settlement_frontier_activation_v2
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE
    public.research_lab_allocation_settlement_frontiers_v2
    TO service_role;

ALTER TABLE public.research_lab_allocation_settlement_frontiers_v2
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_allocation_settlement_frontier_activation_v2
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS
    research_lab_allocation_settlement_frontiers_service_read_v2
    ON public.research_lab_allocation_settlement_frontiers_v2;
CREATE POLICY research_lab_allocation_settlement_frontiers_service_read_v2
    ON public.research_lab_allocation_settlement_frontiers_v2
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS
    research_lab_allocation_settlement_frontier_activation_service_read_v2
    ON public.research_lab_allocation_settlement_frontier_activation_v2;
CREATE POLICY research_lab_allocation_settlement_frontier_activation_service_read_v2
    ON public.research_lab_allocation_settlement_frontier_activation_v2
    FOR SELECT TO service_role USING (true);

DROP TRIGGER IF EXISTS
    research_lab_allocation_settlement_frontiers_append_only_v2
    ON public.research_lab_allocation_settlement_frontiers_v2;
CREATE TRIGGER research_lab_allocation_settlement_frontiers_append_only_v2
BEFORE UPDATE OR DELETE ON
    public.research_lab_allocation_settlement_frontiers_v2
FOR EACH ROW EXECUTE FUNCTION
    public.prevent_research_lab_attested_v2_mutation();

DROP TRIGGER IF EXISTS
    research_lab_allocation_settlement_frontier_activation_append_only_v2
    ON public.research_lab_allocation_settlement_frontier_activation_v2;
CREATE TRIGGER
    research_lab_allocation_settlement_frontier_activation_append_only_v2
BEFORE UPDATE OR DELETE ON
    public.research_lab_allocation_settlement_frontier_activation_v2
FOR EACH ROW EXECUTE FUNCTION
    public.prevent_research_lab_attested_v2_mutation();
GRANT SELECT ON TABLE
    public.research_lab_allocation_settlement_frontier_activation_v2
    TO service_role;

REVOKE ALL ON FUNCTION
    public.persist_research_lab_allocation_settlement_frontier_v2(
        JSONB,
        TEXT,
        TEXT
    )
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.persist_research_lab_allocation_settlement_frontier_v2(
        JSONB,
        TEXT,
        TEXT
    )
    TO service_role;

COMMIT;
