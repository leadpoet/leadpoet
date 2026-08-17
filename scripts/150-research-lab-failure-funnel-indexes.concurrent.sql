-- Non-blocking prerequisite indexes for failure-funnel reporting.
--
-- Run this file to completion before migration 151. Do not wrap it in a
-- transaction: PostgreSQL requires CREATE INDEX CONCURRENTLY to run in
-- autocommit mode. Interrupted builds fail the verification below and can be
-- retried after the invalid index is repaired by an operator.

SET lock_timeout = '5s';

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_research_eval_score_bundles_ticket_created
    ON public.research_evaluation_score_bundles(ticket_id, created_at DESC)
    WHERE ticket_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_research_lab_company_labels_ticket_candidate
    ON public.research_lab_company_label_examples(
        ticket_id, candidate_id, captured_at DESC
    )
    WHERE ticket_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_research_lab_scoring_runs_ticket_candidate
    ON public.research_lab_scoring_runs(ticket_id, candidate_id, created_at DESC)
    WHERE ticket_id IS NOT NULL;

DO $verify$
DECLARE
    invalid_indexes TEXT;
BEGIN
    SELECT pg_catalog.string_agg(expected.index_name, ', ' ORDER BY expected.index_name)
    INTO invalid_indexes
    FROM (
        VALUES
            (
                'idx_research_eval_score_bundles_ticket_created',
                'research_evaluation_score_bundles'
            ),
            (
                'idx_research_lab_company_labels_ticket_candidate',
                'research_lab_company_label_examples'
            ),
            (
                'idx_research_lab_scoring_runs_ticket_candidate',
                'research_lab_scoring_runs'
            )
    ) AS expected(index_name, table_name)
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class index_relation
        JOIN pg_catalog.pg_namespace index_namespace
          ON index_namespace.oid = index_relation.relnamespace
        JOIN pg_catalog.pg_index index_meta
          ON index_meta.indexrelid = index_relation.oid
        JOIN pg_catalog.pg_class table_relation
          ON table_relation.oid = index_meta.indrelid
        JOIN pg_catalog.pg_namespace table_namespace
          ON table_namespace.oid = table_relation.relnamespace
        WHERE index_namespace.nspname = 'public'
          AND table_namespace.nspname = 'public'
          AND index_relation.relname = expected.index_name
          AND table_relation.relname = expected.table_name
          AND index_relation.relkind = 'i'
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND index_meta.indislive
    );

    IF invalid_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'failure-funnel prerequisite indexes are missing or invalid: %',
            invalid_indexes;
    END IF;
END;
$verify$;

RESET lock_timeout;
