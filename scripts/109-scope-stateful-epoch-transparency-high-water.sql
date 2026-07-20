-- Repair the pre-boundary transparency high-water scope.
--
-- Historical transparency rows predate network/genesis authority metadata and
-- can contain epoch_id values from other chains. Only rows explicitly labeled
-- with the legacy-global key semantics belong to the namespace being fenced.
-- The installed write trigger remains broad and fail-closed after fencing.

BEGIN;

-- Measure the exact pre-cutover namespace with the same identity scope used by
-- the durable fence. This is service-role-only and read-only; the fence still
-- repeats the measurement while holding relation locks before it writes state.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
RETURNS BIGINT
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
SET statement_timeout = '120s'
AS $$
DECLARE
    epoch_column RECORD;
    column_max BIGINT;
    measured_high_water BIGINT;
BEGIN
    FOR epoch_column IN
        SELECT c.oid AS relation_oid, n.nspname, c.relname,
               a.attname, a.attnum, a.atttypid
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.atttypid IN (20, 21, 23)
          AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid, a.attnum
    LOOP
        EXECUTE pg_catalog.format(
            'SELECT %1$I::BIGINT FROM %2$I.%3$I '
            'WHERE %1$I IS NOT NULL ORDER BY %1$I DESC LIMIT 1',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_max;
        IF column_max IS NOT NULL
           AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
            measured_high_water := column_max;
        END IF;
    END LOOP;

    SELECT (payload->>'epoch_id')::BIGINT
    INTO column_max
    FROM public.transparency_log
    WHERE pg_catalog.jsonb_typeof(payload) = 'object'
      AND payload ? 'epoch_id'
      AND payload->>'epoch_id' ~ '^[0-9]+$'
      AND payload->>'epoch_key_semantics' = 'legacy_global_360'
    ORDER BY (payload->>'epoch_id')::BIGINT DESC
    LIMIT 1;
    IF column_max IS NOT NULL
       AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
        measured_high_water := column_max;
    END IF;

    IF measured_high_water IS NULL THEN
        RAISE EXCEPTION 'stateful epoch legacy high-water is unavailable';
    END IF;
    RETURN measured_high_water;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
    TO service_role;


CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
    p_network_genesis_hash TEXT,
    p_netuid INTEGER,
    p_last_legacy_epoch_id INTEGER,
    p_first_settlement_epoch_id INTEGER
)
RETURNS TABLE (
    lifecycle_state                     TEXT,
    network_genesis_hash                TEXT,
    netuid                              INTEGER,
    legacy_high_water                   BIGINT,
    last_legacy_epoch_id                BIGINT,
    first_settlement_epoch_id           BIGINT,
    first_settlement_occupied           BOOLEAN,
    fenced_at                           TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
SET lock_timeout = '5s'
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    epoch_relation RECORD;
    epoch_column RECORD;
    column_max BIGINT;
    column_occupied BOOLEAN;
    measured_high_water BIGINT := NULL;
    measured_occupied BOOLEAN := FALSE;
BEGIN
    IF p_network_genesis_hash !~ '^0x[0-9a-f]{64}$'
       OR p_netuid <= 0
       OR p_last_legacy_epoch_id < 0
       OR p_first_settlement_epoch_id::BIGINT IS DISTINCT FROM
          p_last_legacy_epoch_id::BIGINT + 1 THEN
        RAISE EXCEPTION 'stateful epoch pre-boundary fence input is invalid';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'stateful epoch cutover singleton state is missing';
    END IF;

    IF state_row.lifecycle_state <> 'legacy_open' THEN
        IF state_row.network_genesis_hash IS DISTINCT FROM p_network_genesis_hash
           OR state_row.netuid IS DISTINCT FROM p_netuid
           OR state_row.last_legacy_epoch_id IS DISTINCT FROM
              p_last_legacy_epoch_id
           OR state_row.first_settlement_epoch_id IS DISTINCT FROM
              p_first_settlement_epoch_id THEN
            RAISE EXCEPTION 'stateful epoch pre-boundary fence conflicts with durable plan';
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
        RETURN;
    END IF;

    -- The payload epoch identity lives inside JSONB and cannot use a normal
    -- column index.  Require the separately installed concurrent expression
    -- index before taking any production locks or attempting the measurement.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_index index_meta
        JOIN pg_catalog.pg_class index_relation
          ON index_relation.oid = index_meta.indexrelid
        JOIN pg_catalog.pg_namespace index_namespace
          ON index_namespace.oid = index_relation.relnamespace
        JOIN pg_catalog.pg_class table_relation
          ON table_relation.oid = index_meta.indrelid
        JOIN pg_catalog.pg_namespace table_namespace
          ON table_namespace.oid = table_relation.relnamespace
        JOIN pg_catalog.pg_attribute payload_column
          ON payload_column.attrelid = table_relation.oid
         AND payload_column.attname = 'payload'
        JOIN pg_catalog.pg_am access_method
          ON access_method.oid = index_relation.relam
        JOIN pg_catalog.pg_opclass operator_class
          ON operator_class.oid = index_meta.indclass[0]
        JOIN pg_catalog.pg_namespace operator_class_namespace
          ON operator_class_namespace.oid = operator_class.opcnamespace
        WHERE index_meta.indrelid =
              'public.transparency_log'::pg_catalog.regclass
          AND table_namespace.nspname = 'public'
          AND table_relation.relname = 'transparency_log'
          AND table_relation.relkind = 'r'
          AND payload_column.attnum > 0
          AND NOT payload_column.attisdropped
          AND payload_column.atttypid =
              'pg_catalog.jsonb'::pg_catalog.regtype
          AND index_namespace.nspname = 'public'
          AND index_relation.relname =
              'idx_transparency_log_payload_epoch_identity_v1'
          AND index_relation.relkind = 'i'
          AND index_relation.relpersistence = table_relation.relpersistence
          AND access_method.amname = 'btree'
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND index_meta.indislive
          AND NOT index_meta.indisunique
          AND NOT index_meta.indisprimary
          AND NOT index_meta.indisexclusion
          AND index_meta.indnatts = 1
          AND index_meta.indnkeyatts = 1
          AND index_meta.indkey[0] = 0
          AND index_meta.indoption[0] = 3
          AND index_meta.indcollation[0] = 0
          AND operator_class.opcdefault
          AND operator_class.opcmethod = index_relation.relam
          AND operator_class.opcintype =
              'pg_catalog.int8'::pg_catalog.regtype
          AND operator_class.opcname = 'int8_ops'
          AND operator_class_namespace.nspname = 'pg_catalog'
          AND index_meta.indexprs IS NOT NULL
          AND index_meta.indpred IS NOT NULL
          AND pg_catalog.pg_get_expr(
                  index_meta.indexprs,
                  index_meta.indrelid,
                  FALSE
              ) = '((payload ->> ''epoch_id''::text))::bigint'
          AND pg_catalog.pg_get_expr(
                  index_meta.indpred,
                  index_meta.indrelid,
                  FALSE
              ) = '((jsonb_typeof(payload) = ''object''::text) AND (payload ? ''epoch_id''::text) AND ((payload ->> ''epoch_id''::text) ~ ''^[0-9]+$''::text))'
    ) THEN
        RAISE EXCEPTION
            'stateful epoch fence prerequisite payload identity index is missing or invalid';
    END IF;

    FOR epoch_relation IN
        SELECT DISTINCT c.oid, n.nspname, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND (
              (
                  a.atttypid IN (20, 21, 23)
                  AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
              )
              OR c.relname = 'transparency_log'
          )
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid
    LOOP
        -- Migration-time trigger installation avoids ACCESS EXCLUSIVE DDL in
        -- the live reservation RPC. A table added after the migration is an
        -- explicit schema-drift failure, never an excuse to reserve an
        -- incompletely guarded namespace.
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_trigger trigger_meta
            JOIN pg_catalog.pg_proc trigger_function
              ON trigger_function.oid = trigger_meta.tgfoid
            JOIN pg_catalog.pg_namespace function_namespace
              ON function_namespace.oid = trigger_function.pronamespace
            WHERE trigger_meta.tgrelid = epoch_relation.oid
              AND NOT trigger_meta.tgisinternal
              AND trigger_meta.tgenabled <> 'D'
              AND function_namespace.nspname = 'public'
              AND trigger_function.proname =
                  'enforce_research_lab_stateful_epoch_fence_v1'
              AND (trigger_meta.tgtype & 1) = 1
              AND (trigger_meta.tgtype & 2) = 2
              AND (trigger_meta.tgtype & 4) = 4
              AND (trigger_meta.tgtype & 16) = 16
        ) THEN
            RAISE EXCEPTION
                'stateful epoch fence trigger is missing for %',
                epoch_relation.relname;
        END IF;
        EXECUTE pg_catalog.format(
            'LOCK TABLE %I.%I IN SHARE MODE',
            epoch_relation.nspname,
            epoch_relation.relname
        );
    END LOOP;

    FOR epoch_column IN
        -- Only physical identity columns participate here; the separate
        -- Explicitly typed legacy-global transparency identities are included
        -- below. Unscoped historical payloads can belong to another network
        -- and must not define this Finney SN71 namespace. In particular,
        -- reward_expires_epoch must not raise the high-water mark because it
        -- is a future schedule/reference, not an allocated epoch identity.
        SELECT c.oid AS relation_oid, n.nspname, c.relname,
               a.attname, a.attnum, a.atttypid
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.atttypid IN (20, 21, 23)
          AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid, a.attnum
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index index_meta
            JOIN pg_catalog.pg_class index_relation
              ON index_relation.oid = index_meta.indexrelid
            JOIN pg_catalog.pg_am access_method
              ON access_method.oid = index_relation.relam
            JOIN pg_catalog.pg_opclass operator_class
              ON operator_class.oid = index_meta.indclass[0]
            WHERE index_meta.indrelid = epoch_column.relation_oid
              AND index_relation.relkind IN ('i', 'I')
              AND access_method.amname = 'btree'
              AND index_meta.indisvalid
              AND index_meta.indisready
              AND index_meta.indislive
              AND index_meta.indpred IS NULL
              AND index_meta.indexprs IS NULL
              AND index_meta.indnkeyatts >= 1
              AND index_meta.indkey[0] = epoch_column.attnum
              AND operator_class.opcdefault
              AND operator_class.opcmethod = index_relation.relam
              AND operator_class.opcintype = epoch_column.atttypid
        ) THEN
            RAISE EXCEPTION
                'stateful epoch fence prerequisite btree index is missing for %.%',
                epoch_column.relname,
                epoch_column.attname;
        END IF;

        EXECUTE pg_catalog.format(
            'SELECT %1$I::BIGINT FROM %2$I.%3$I '
            'WHERE %1$I IS NOT NULL ORDER BY %1$I DESC LIMIT 1',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_max;
        EXECUTE pg_catalog.format(
            'SELECT EXISTS (SELECT 1 FROM %2$I.%3$I WHERE %1$I = $1)',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_occupied
        USING p_first_settlement_epoch_id;
        IF column_max IS NOT NULL
           AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
            measured_high_water := column_max;
        END IF;
        measured_occupied := measured_occupied OR COALESCE(column_occupied, FALSE);
    END LOOP;

    SELECT (payload->>'epoch_id')::BIGINT
    INTO column_max
    FROM public.transparency_log
    WHERE pg_catalog.jsonb_typeof(payload) = 'object'
      AND payload ? 'epoch_id'
      AND payload->>'epoch_id' ~ '^[0-9]+$'
      AND payload->>'epoch_key_semantics' = 'legacy_global_360'
    ORDER BY (payload->>'epoch_id')::BIGINT DESC
    LIMIT 1;
    SELECT EXISTS (
        SELECT 1
        FROM public.transparency_log
        WHERE pg_catalog.jsonb_typeof(payload) = 'object'
          AND payload ? 'epoch_id'
          AND payload->>'epoch_id' ~ '^[0-9]+$'
          AND payload->>'epoch_key_semantics' = 'legacy_global_360'
          AND (payload->>'epoch_id')::BIGINT = p_first_settlement_epoch_id
    )
    INTO column_occupied;
    IF column_max IS NOT NULL
       AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
        measured_high_water := column_max;
    END IF;
    measured_occupied := measured_occupied OR COALESCE(column_occupied, FALSE);

    IF measured_occupied
       OR measured_high_water IS DISTINCT FROM p_last_legacy_epoch_id::BIGINT THEN
        RAISE EXCEPTION
            'stateful epoch pre-boundary fence expected high-water %, observed %, occupied %',
            p_last_legacy_epoch_id, measured_high_water, measured_occupied;
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET lifecycle_state = 'cutover_fenced',
        network_genesis_hash = p_network_genesis_hash,
        netuid = p_netuid,
        last_legacy_epoch_id = p_last_legacy_epoch_id,
        first_settlement_epoch_id = p_first_settlement_epoch_id,
        fenced_at = pg_catalog.clock_timestamp(),
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.network_genesis_hash,
        state_row.netuid,
        measured_high_water,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        measured_occupied,
        state_row.fenced_at;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(TEXT, INTEGER, INTEGER, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(TEXT, INTEGER, INTEGER, INTEGER)
    TO service_role;

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
    TEXT, INTEGER, INTEGER, INTEGER
) SET statement_timeout = '120s';

NOTIFY pgrst, 'reload schema';

COMMIT;
