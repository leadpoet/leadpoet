-- Repair fail-closed stateful epoch fence coverage for epoch-bearing tables
-- created after migration 101. Apply after migration 105 and before cutover.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $repair$
DECLARE
    item RECORD;
BEGIN
    IF pg_catalog.to_regprocedure(
        'public.enforce_research_lab_stateful_epoch_fence_v1()'
    ) IS NULL THEN
        RAISE EXCEPTION
            'stateful epoch fence function is missing; apply migration 101 first';
    END IF;

    FOR item IN
        SELECT DISTINCT c.oid, n.nspname, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attribute a
          ON a.attrelid = c.oid
         AND a.attnum > 0
         AND NOT a.attisdropped
        WHERE n.nspname = 'public'
          AND (
              c.relkind = 'p'
              OR (c.relkind = 'r' AND NOT c.relispartition)
          )
          AND c.relname <>
              'research_lab_stateful_subnet_epoch_cutover_state_v1'
          AND (
              (
                  a.atttypid IN (20, 21, 23)
                  AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
              )
              OR c.relname IN (
                  'transparency_log',
                  'research_lab_stateful_subnet_epoch_candidates_v1',
                  'research_lab_stateful_subnet_epoch_cutovers_v1',
                  'research_lab_stateful_subnet_epoch_boundaries_v1',
                  'research_lab_stateful_subnet_epoch_snapshots_v1',
                  'research_lab_attested_weight_finalizations_v2'
              )
          )
        ORDER BY c.oid
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_trigger trigger_meta
            JOIN pg_catalog.pg_proc trigger_function
              ON trigger_function.oid = trigger_meta.tgfoid
            JOIN pg_catalog.pg_namespace function_namespace
              ON function_namespace.oid = trigger_function.pronamespace
            WHERE trigger_meta.tgrelid = item.oid
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
            EXECUTE pg_catalog.format(
                'DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1 '
                'ON %I.%I',
                item.nspname,
                item.relname
            );
            EXECUTE pg_catalog.format(
                'CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1 '
                'BEFORE INSERT OR UPDATE ON %I.%I FOR EACH ROW '
                'EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()',
                item.nspname,
                item.relname
            );
        END IF;
    END LOOP;
END;
$repair$;

COMMENT ON TRIGGER enforce_research_lab_stateful_epoch_fence_v1
    ON public.research_lab_legacy_allocation_nonfinalizations_v2 IS
    'Rejects reserved or stale epoch identities during and after the stateful SN71 cutover.';

NOTIFY pgrst, 'reload schema';

COMMIT;
