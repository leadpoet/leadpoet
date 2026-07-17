-- Preserve historical allocation hashes when the legacy allocation document
-- predates the duplicate embedded netuid field. Scope remains bound by the
-- migration row, signed audit event, validator bundle, chain evidence, and
-- attested settlement receipt. If allocation_doc.netuid exists, it must match.
-- Apply after scripts/99-research-lab-v2-champion-settlement.sql.

BEGIN;

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid =
              'public.research_lab_legacy_finalized_allocation_migrations_v2'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%allocation_doc%'
          AND pg_get_constraintdef(oid) LIKE '%netuid%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_legacy_finalized_allocation_migrations_v2 DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_legacy_finalized_allocation_migrations_v2
    ADD CONSTRAINT research_lab_legacy_allocation_doc_netuid_check
    CHECK (
        CASE
            WHEN NOT (allocation_doc ? 'netuid') THEN TRUE
            WHEN allocation_doc->>'netuid' ~ '^[1-9][0-9]*$'
                THEN (allocation_doc->>'netuid')::NUMERIC = netuid
            ELSE FALSE
        END
    ) NOT VALID;

ALTER TABLE public.research_lab_legacy_finalized_allocation_migrations_v2
    VALIDATE CONSTRAINT research_lab_legacy_allocation_doc_netuid_check;

NOTIFY pgrst, 'reload schema';

COMMIT;
