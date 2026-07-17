-- Non-blocking prerequisite indexes for the stateful subnet epoch cutover.
--
-- Run this file to completion BEFORE applying
-- scripts/101-stateful-subnet-epoch-authority.sql.  Do not wrap this file in
-- BEGIN/COMMIT: PostgreSQL requires each CREATE INDEX CONCURRENTLY to run
-- outside a transaction block.  These indexes let the one-time fence prove the
-- exact legacy high-water and first-ordinal vacancy without scanning large
-- production tables.  The fence fails closed if any discovered physical epoch
-- identity still lacks a valid, ready, non-partial btree whose first key is the
-- identity column.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_epoch_audit_logs_epoch_identity_v1
    ON public.epoch_audit_logs(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_published_weight_bundles_epoch_identity_v1
    ON public.published_weight_bundles(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_attested_weight_bundles_epoch_identity_v1
    ON public.research_lab_attested_weight_bundles(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_attested_weight_bundles_v2_epoch_identity_v1
    ON public.research_lab_attested_weight_bundles_v2(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_champion_rewards_eval_epoch_identity_v1
    ON public.research_lab_champion_reward_obligations(evaluation_epoch DESC);

-- SKIP_VIEW: research_lab_epoch_payouts is a derived view in production.  The
-- physical epoch identities that feed it are indexed independently below and
-- by the catalog catch-all. PostgreSQL cannot create an index on a view.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_legacy_finalized_epoch_identity_v1
    ON public.research_lab_legacy_finalized_allocation_migrations_v2(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_private_benchmark_eval_epoch_identity_v1
    ON public.research_lab_private_model_benchmark_bundles(evaluation_epoch DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rl_scoring_runs_eval_epoch_identity_v1
    ON public.research_lab_scoring_runs(evaluation_epoch DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transparency_log_epoch_identity_v1
    ON public.transparency_log(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_validation_evidence_epoch_identity_v1
    ON public.validation_evidence_private(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_validator_attestations_epoch_identity_v1
    ON public.validator_attestations(epoch_id DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_validator_sourcing_inputs_epoch_identity_v1
    ON public.validator_sourcing_epoch_inputs_v2(epoch_id DESC);

-- Lifecycle identity is stored in JSONB even where transparency_log also has a
-- physical epoch_id column.  The predicate exactly matches the fence queries.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transparency_log_payload_epoch_identity_v1
    ON public.transparency_log (((payload->>'epoch_id')::BIGINT) DESC)
    WHERE pg_catalog.jsonb_typeof(payload) = 'object'
      AND payload ? 'epoch_id'
      AND payload->>'epoch_id' ~ '^[0-9]+$';

-- Fail the prerequisite itself if the live catalog has drifted beyond the
-- enumerated indexes or an interrupted concurrent build left an invalid index.
-- This DO block is a separate short transaction after all concurrent builds;
-- it does not put CREATE INDEX CONCURRENTLY inside a transaction block.
DO $$
DECLARE
    building_identity_indexes TEXT;
    missing_identity_indexes TEXT;
    named_index_contract_errors TEXT;
BEGIN
    -- A browser/proxy timeout does not prove a concurrent build stopped.  Never
    -- tell an operator to drop an index while PostgreSQL still reports a build
    -- on an epoch-bearing table.
    SELECT pg_catalog.string_agg(
               active_build.label,
               ', ' ORDER BY active_build.label
           )
    INTO building_identity_indexes
    FROM (
        SELECT DISTINCT pg_catalog.format(
                   '%I.%I[%s]',
                   relation_namespace.nspname,
                   relation.relname,
                   COALESCE(index_relation.relname, 'catalog-pending')
               ) AS label
        FROM pg_catalog.pg_stat_progress_create_index AS progress
        JOIN pg_catalog.pg_class AS relation
          ON relation.oid = progress.relid
        JOIN pg_catalog.pg_namespace AS relation_namespace
          ON relation_namespace.oid = relation.relnamespace
        JOIN pg_catalog.pg_attribute AS column_meta
          ON column_meta.attrelid = relation.oid
        LEFT JOIN pg_catalog.pg_class AS index_relation
          ON index_relation.oid = progress.index_relid
        WHERE progress.datname = pg_catalog.current_database()
          AND relation_namespace.nspname = 'public'
          AND relation.relkind IN ('r', 'p')
          AND column_meta.attnum > 0
          AND NOT column_meta.attisdropped
          AND column_meta.atttypid IN (20, 21, 23)
          AND column_meta.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
    ) AS active_build;
    IF building_identity_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite indexes are still building: %; wait and rerun the catalog check, and do not drop these indexes',
            building_identity_indexes;
    END IF;

    -- IF NOT EXISTS is name-only in PostgreSQL.  Bind every migration-owned
    -- ordinary index to its exact public table, exact sole integer key, default
    -- btree operator class, and canonical DESC definition.  This rejects a
    -- valid-looking same-name index on the wrong table/key or with a partial,
    -- expression, INCLUDE, alternate access-method, or interrupted definition.
    WITH expected(index_name, table_name, column_name) AS (
        VALUES
            ('idx_epoch_audit_logs_epoch_identity_v1',
             'epoch_audit_logs', 'epoch_id'),
            ('idx_published_weight_bundles_epoch_identity_v1',
             'published_weight_bundles', 'epoch_id'),
            ('idx_rl_attested_weight_bundles_epoch_identity_v1',
             'research_lab_attested_weight_bundles', 'epoch_id'),
            ('idx_rl_attested_weight_bundles_v2_epoch_identity_v1',
             'research_lab_attested_weight_bundles_v2', 'epoch_id'),
            ('idx_rl_champion_rewards_eval_epoch_identity_v1',
             'research_lab_champion_reward_obligations', 'evaluation_epoch'),
            ('idx_rl_legacy_finalized_epoch_identity_v1',
             'research_lab_legacy_finalized_allocation_migrations_v2', 'epoch_id'),
            ('idx_rl_private_benchmark_eval_epoch_identity_v1',
             'research_lab_private_model_benchmark_bundles', 'evaluation_epoch'),
            ('idx_rl_scoring_runs_eval_epoch_identity_v1',
             'research_lab_scoring_runs', 'evaluation_epoch'),
            ('idx_transparency_log_epoch_identity_v1',
             'transparency_log', 'epoch_id'),
            ('idx_validation_evidence_epoch_identity_v1',
             'validation_evidence_private', 'epoch_id'),
            ('idx_validator_attestations_epoch_identity_v1',
             'validator_attestations', 'epoch_id'),
            ('idx_validator_sourcing_inputs_epoch_identity_v1',
             'validator_sourcing_epoch_inputs_v2', 'epoch_id')
    ), plain_contract_failures AS (
        SELECT expected.index_name
        FROM expected
        LEFT JOIN pg_catalog.pg_namespace AS table_namespace
          ON table_namespace.nspname = 'public'
        LEFT JOIN pg_catalog.pg_class AS table_relation
          ON table_relation.relnamespace = table_namespace.oid
         AND table_relation.relname = expected.table_name
        LEFT JOIN pg_catalog.pg_attribute AS column_meta
          ON column_meta.attrelid = table_relation.oid
         AND column_meta.attname = expected.column_name
        LEFT JOIN pg_catalog.pg_namespace AS index_namespace
          ON index_namespace.nspname = 'public'
        LEFT JOIN pg_catalog.pg_class AS index_relation
          ON index_relation.relnamespace = index_namespace.oid
         AND index_relation.relname = expected.index_name
        LEFT JOIN pg_catalog.pg_index AS index_meta
          ON index_meta.indexrelid = index_relation.oid
        LEFT JOIN pg_catalog.pg_am AS access_method
          ON access_method.oid = index_relation.relam
        LEFT JOIN pg_catalog.pg_opclass AS operator_class
          ON operator_class.oid = index_meta.indclass[0]
        WHERE NOT COALESCE(
            table_relation.relkind = 'r'
            AND column_meta.attnum > 0
            AND NOT column_meta.attisdropped
            AND column_meta.atttypid IN (20, 21, 23)
            AND index_relation.relkind = 'i'
            AND index_relation.relpersistence = table_relation.relpersistence
            AND index_meta.indrelid = table_relation.oid
            AND access_method.amname = 'btree'
            AND index_meta.indisvalid
            AND index_meta.indisready
            AND index_meta.indislive
            AND NOT index_meta.indisunique
            AND NOT index_meta.indisprimary
            AND NOT index_meta.indisexclusion
            AND index_meta.indpred IS NULL
            AND index_meta.indexprs IS NULL
            AND index_meta.indnatts = 1
            AND index_meta.indnkeyatts = 1
            AND index_meta.indkey[0] = column_meta.attnum
            AND index_meta.indoption[0] = 3
            AND index_meta.indcollation[0] = 0
            AND operator_class.opcdefault
            AND operator_class.opcmethod = index_relation.relam
            AND operator_class.opcintype = column_meta.atttypid,
            FALSE
        )
    ), payload_contract_failure AS (
        SELECT 'idx_transparency_log_payload_epoch_identity_v1'::TEXT AS index_name
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_namespace AS table_namespace
            JOIN pg_catalog.pg_class AS table_relation
              ON table_relation.relnamespace = table_namespace.oid
             AND table_relation.relname = 'transparency_log'
            JOIN pg_catalog.pg_attribute AS payload_column
              ON payload_column.attrelid = table_relation.oid
             AND payload_column.attname = 'payload'
            JOIN pg_catalog.pg_namespace AS index_namespace
              ON index_namespace.nspname = 'public'
            JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.relnamespace = index_namespace.oid
             AND index_relation.relname =
                 'idx_transparency_log_payload_epoch_identity_v1'
            JOIN pg_catalog.pg_index AS index_meta
              ON index_meta.indexrelid = index_relation.oid
            JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_relation.relam
            JOIN pg_catalog.pg_opclass AS operator_class
              ON operator_class.oid = index_meta.indclass[0]
            JOIN pg_catalog.pg_namespace AS operator_class_namespace
              ON operator_class_namespace.oid = operator_class.opcnamespace
            WHERE table_namespace.nspname = 'public'
              AND table_relation.relkind = 'r'
              AND payload_column.attnum > 0
              AND NOT payload_column.attisdropped
              AND payload_column.atttypid = 'pg_catalog.jsonb'::pg_catalog.regtype
              AND index_relation.relkind = 'i'
              AND index_relation.relpersistence = table_relation.relpersistence
              AND index_meta.indrelid = table_relation.oid
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
        )
    )
    SELECT pg_catalog.string_agg(
               failures.index_name,
               ', ' ORDER BY failures.index_name
           )
    INTO named_index_contract_errors
    FROM (
        SELECT index_name FROM plain_contract_failures
        UNION ALL
        SELECT index_name FROM payload_contract_failure
    ) AS failures;
    IF named_index_contract_errors IS NOT NULL THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite named indexes are missing, invalid, or have the wrong exact definition: %; inspect pg_index, drop only confirmed non-building invalid remnants concurrently, and rerun this file',
            named_index_contract_errors;
    END IF;

    -- Independently discover every physical integer epoch identity.  This
    -- covers pre-existing composite indexes such as the allocation-snapshot
    -- epoch index without requiring an impossible index on a derived view.
    SELECT pg_catalog.string_agg(
               pg_catalog.format('%I.%I', relation.relname, column_meta.attname),
               ', ' ORDER BY relation.relname, column_meta.attname
           )
    INTO missing_identity_indexes
    FROM pg_catalog.pg_class AS relation
    JOIN pg_catalog.pg_namespace AS relation_namespace
      ON relation_namespace.oid = relation.relnamespace
    JOIN pg_catalog.pg_attribute AS column_meta
      ON column_meta.attrelid = relation.oid
    WHERE relation_namespace.nspname = 'public'
      AND relation.relkind IN ('r', 'p')
      AND column_meta.attnum > 0
      AND NOT column_meta.attisdropped
      AND column_meta.atttypid IN (20, 21, 23)
      AND column_meta.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
      AND NOT EXISTS (
          SELECT 1
          FROM pg_catalog.pg_index AS index_meta
          JOIN pg_catalog.pg_class AS index_relation
            ON index_relation.oid = index_meta.indexrelid
          JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = index_relation.relam
          JOIN pg_catalog.pg_opclass AS operator_class
            ON operator_class.oid = index_meta.indclass[0]
          WHERE index_meta.indrelid = relation.oid
            AND index_relation.relkind IN ('i', 'I')
            AND access_method.amname = 'btree'
            AND index_meta.indisvalid
            AND index_meta.indisready
            AND index_meta.indislive
            AND index_meta.indpred IS NULL
            AND index_meta.indexprs IS NULL
            AND index_meta.indnkeyatts >= 1
            AND index_meta.indkey[0] = column_meta.attnum
            AND operator_class.opcdefault
            AND operator_class.opcmethod = index_relation.relam
            AND operator_class.opcintype = column_meta.atttypid
      );
    IF missing_identity_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite lacks identity indexes: %',
            missing_identity_indexes;
    END IF;

END;
$$;

-- The DO block above is the exact contract authority.  This report is
-- deliberately left-joined from the complete registry so a missing expected
-- name remains visible instead of silently disappearing from the result.
WITH expected(index_name, table_name, identity_name) AS (
    VALUES
        ('idx_epoch_audit_logs_epoch_identity_v1',
         'epoch_audit_logs', 'epoch_id'),
        ('idx_published_weight_bundles_epoch_identity_v1',
         'published_weight_bundles', 'epoch_id'),
        ('idx_rl_attested_weight_bundles_epoch_identity_v1',
         'research_lab_attested_weight_bundles', 'epoch_id'),
        ('idx_rl_attested_weight_bundles_v2_epoch_identity_v1',
         'research_lab_attested_weight_bundles_v2', 'epoch_id'),
        ('idx_rl_champion_rewards_eval_epoch_identity_v1',
         'research_lab_champion_reward_obligations', 'evaluation_epoch'),
        ('idx_rl_legacy_finalized_epoch_identity_v1',
         'research_lab_legacy_finalized_allocation_migrations_v2', 'epoch_id'),
        ('idx_rl_private_benchmark_eval_epoch_identity_v1',
         'research_lab_private_model_benchmark_bundles', 'evaluation_epoch'),
        ('idx_rl_scoring_runs_eval_epoch_identity_v1',
         'research_lab_scoring_runs', 'evaluation_epoch'),
        ('idx_transparency_log_epoch_identity_v1',
         'transparency_log', 'epoch_id'),
        ('idx_validation_evidence_epoch_identity_v1',
         'validation_evidence_private', 'epoch_id'),
        ('idx_validator_attestations_epoch_identity_v1',
         'validator_attestations', 'epoch_id'),
        ('idx_validator_sourcing_inputs_epoch_identity_v1',
         'validator_sourcing_epoch_inputs_v2', 'epoch_id'),
        ('idx_transparency_log_payload_epoch_identity_v1',
         'transparency_log', 'payload.epoch_id')
)
SELECT expected.index_name,
       expected.table_name,
       expected.identity_name,
       index_relation.oid IS NOT NULL AS index_exists,
       table_relation.relname AS actual_table_name,
       access_method.amname AS access_method,
       index_meta.indisvalid,
       index_meta.indisready,
       index_meta.indislive,
       index_meta.indnkeyatts,
       index_meta.indnatts,
       pg_catalog.pg_get_indexdef(index_relation.oid) AS index_definition
FROM expected
LEFT JOIN pg_catalog.pg_namespace AS index_namespace
  ON index_namespace.nspname = 'public'
LEFT JOIN pg_catalog.pg_class AS index_relation
  ON index_relation.relnamespace = index_namespace.oid
 AND index_relation.relname = expected.index_name
LEFT JOIN pg_catalog.pg_index AS index_meta
  ON index_meta.indexrelid = index_relation.oid
LEFT JOIN pg_catalog.pg_class AS table_relation
  ON table_relation.oid = index_meta.indrelid
LEFT JOIN pg_catalog.pg_am AS access_method
  ON access_method.oid = index_relation.relam
ORDER BY expected.index_name;
