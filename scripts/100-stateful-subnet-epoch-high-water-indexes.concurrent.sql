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
    missing_identity_indexes TEXT;
    invalid_named_indexes TEXT;
BEGIN
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
          WHERE index_meta.indrelid = relation.oid
            AND access_method.amname = 'btree'
            AND index_meta.indisvalid
            AND index_meta.indisready
            AND index_meta.indpred IS NULL
            AND index_meta.indexprs IS NULL
            AND index_meta.indnkeyatts >= 1
            AND index_meta.indkey[0] = column_meta.attnum
      );
    IF missing_identity_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite lacks identity indexes: %',
            missing_identity_indexes;
    END IF;

    SELECT pg_catalog.string_agg(index_relation.relname, ', ')
    INTO invalid_named_indexes
    FROM pg_catalog.pg_class AS index_relation
    LEFT JOIN pg_catalog.pg_index AS index_meta
      ON index_meta.indexrelid = index_relation.oid
    WHERE index_relation.relname IN (
        'idx_epoch_audit_logs_epoch_identity_v1',
        'idx_published_weight_bundles_epoch_identity_v1',
        'idx_rl_attested_weight_bundles_epoch_identity_v1',
        'idx_rl_attested_weight_bundles_v2_epoch_identity_v1',
        'idx_rl_champion_rewards_eval_epoch_identity_v1',
        'idx_rl_legacy_finalized_epoch_identity_v1',
        'idx_rl_private_benchmark_eval_epoch_identity_v1',
        'idx_rl_scoring_runs_eval_epoch_identity_v1',
        'idx_transparency_log_epoch_identity_v1',
        'idx_validation_evidence_epoch_identity_v1',
        'idx_validator_attestations_epoch_identity_v1',
        'idx_validator_sourcing_inputs_epoch_identity_v1',
        'idx_transparency_log_payload_epoch_identity_v1'
    )
      AND (
          index_meta.indexrelid IS NULL
          OR NOT index_meta.indisvalid
          OR NOT index_meta.indisready
      );
    IF invalid_named_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite has invalid concurrent indexes: %; drop each invalid index concurrently and rerun this file',
            invalid_named_indexes;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_index AS index_meta
        JOIN pg_catalog.pg_class AS index_relation
          ON index_relation.oid = index_meta.indexrelid
        WHERE index_relation.relname =
              'idx_transparency_log_payload_epoch_identity_v1'
          AND index_meta.indrelid =
              'public.transparency_log'::pg_catalog.regclass
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND pg_catalog.pg_get_expr(
                  index_meta.indexprs,
                  index_meta.indrelid
              ) = '((payload ->> ''epoch_id''::text))::bigint'
          AND pg_catalog.strpos(
                  pg_catalog.pg_get_expr(
                      index_meta.indpred,
                      index_meta.indrelid
                  ),
                  '^[0-9]+$'
              ) > 0
    ) THEN
        RAISE EXCEPTION
            'stateful epoch prerequisite payload identity index has the wrong definition';
    END IF;
END;
$$;

-- Every row must be valid/ready before the authority migration or fence is run.
SELECT index_relation.relname,
       index_meta.indisvalid,
       index_meta.indisready
FROM pg_catalog.pg_index AS index_meta
JOIN pg_catalog.pg_class AS index_relation
  ON index_relation.oid = index_meta.indexrelid
WHERE index_relation.relname IN (
    'idx_epoch_audit_logs_epoch_identity_v1',
    'idx_published_weight_bundles_epoch_identity_v1',
    'idx_rl_attested_weight_bundles_epoch_identity_v1',
    'idx_rl_attested_weight_bundles_v2_epoch_identity_v1',
    'idx_rl_champion_rewards_eval_epoch_identity_v1',
    'idx_rl_legacy_finalized_epoch_identity_v1',
    'idx_rl_private_benchmark_eval_epoch_identity_v1',
    'idx_rl_scoring_runs_eval_epoch_identity_v1',
    'idx_transparency_log_epoch_identity_v1',
    'idx_validation_evidence_epoch_identity_v1',
    'idx_validator_attestations_epoch_identity_v1',
    'idx_validator_sourcing_inputs_epoch_identity_v1',
    'idx_transparency_log_payload_epoch_identity_v1'
)
ORDER BY index_relation.relname;
