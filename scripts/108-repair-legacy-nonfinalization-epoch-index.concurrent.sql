-- Repair the stateful cutover high-water prerequisite for the legacy
-- nonfinalization table introduced after migration 100. This creates no table
-- or column and is safe to rerun. Do not wrap CREATE INDEX CONCURRENTLY in a
-- transaction block.

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_research_lab_legacy_nonfinalization_epoch_v2
    ON public.research_lab_legacy_allocation_nonfinalizations_v2(epoch_id DESC);

DO $verify$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_namespace AS table_namespace
        JOIN pg_catalog.pg_class AS table_relation
          ON table_relation.relnamespace = table_namespace.oid
         AND table_relation.relname =
             'research_lab_legacy_allocation_nonfinalizations_v2'
        JOIN pg_catalog.pg_attribute AS epoch_column
          ON epoch_column.attrelid = table_relation.oid
         AND epoch_column.attname = 'epoch_id'
        JOIN pg_catalog.pg_namespace AS index_namespace
          ON index_namespace.nspname = 'public'
        JOIN pg_catalog.pg_class AS index_relation
          ON index_relation.relnamespace = index_namespace.oid
         AND index_relation.relname =
             'idx_research_lab_legacy_nonfinalization_epoch_v2'
        JOIN pg_catalog.pg_index AS index_meta
          ON index_meta.indexrelid = index_relation.oid
        JOIN pg_catalog.pg_am AS access_method
          ON access_method.oid = index_relation.relam
        JOIN pg_catalog.pg_opclass AS operator_class
          ON operator_class.oid = index_meta.indclass[0]
        WHERE table_namespace.nspname = 'public'
          AND table_relation.relkind = 'r'
          AND epoch_column.attnum > 0
          AND NOT epoch_column.attisdropped
          AND epoch_column.atttypid IN (20, 21, 23)
          AND index_relation.relkind = 'i'
          AND index_relation.relpersistence = table_relation.relpersistence
          AND index_meta.indrelid = table_relation.oid
          AND access_method.amname = 'btree'
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND index_meta.indislive
          AND index_meta.indpred IS NULL
          AND index_meta.indexprs IS NULL
          AND index_meta.indnkeyatts = 1
          AND index_meta.indnatts = 1
          AND index_meta.indkey[0] = epoch_column.attnum
          AND index_meta.indoption[0] = 3
          AND index_meta.indcollation[0] = 0
          AND operator_class.opcdefault
          AND operator_class.opcmethod = index_relation.relam
          AND operator_class.opcintype = epoch_column.atttypid
    ) THEN
        RAISE EXCEPTION
            'legacy nonfinalization epoch index is missing or invalid';
    END IF;
END;
$verify$;

NOTIFY pgrst, 'reload schema';
