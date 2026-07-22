-- Scope authoritative V2 receipt idempotency to one settlement epoch.
--
-- A validator hotkey-signature request is stable across epochs, while its
-- signed output and receipt are epoch-bound. The original global uniqueness
-- key therefore rejected the first valid receipt after an earlier epoch used
-- the same request. Preserve strict idempotency within an epoch while allowing
-- distinct, independently verified receipts in later epochs.

BEGIN;

DO $migration$
DECLARE
    epoch_constraint_name TEXT;
    old_constraint RECORD;
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint AS constraint_record
        JOIN pg_class AS table_record
          ON table_record.oid = constraint_record.conrelid
        JOIN pg_namespace AS schema_record
          ON schema_record.oid = table_record.relnamespace
        WHERE schema_record.nspname = 'public'
          AND table_record.relname =
              'research_lab_attested_execution_receipts_v2'
          AND constraint_record.contype = 'u'
          AND ARRAY(
              SELECT attribute.attname::TEXT
              FROM unnest(constraint_record.conkey) WITH ORDINALITY
                   AS constrained_column(attnum, position)
              JOIN pg_attribute AS attribute
                ON attribute.attrelid = table_record.oid
               AND attribute.attnum = constrained_column.attnum
              ORDER BY constrained_column.position
          ) = ARRAY[
              'role',
              'purpose',
              'job_id',
              'epoch_id',
              'input_root',
              'config_hash'
          ]::TEXT[]
    ) THEN
        ALTER TABLE public.research_lab_attested_execution_receipts_v2
            ADD CONSTRAINT research_lab_attested_receipts_v2_epoch_idempotency_key
            UNIQUE (
                role,
                purpose,
                job_id,
                epoch_id,
                input_root,
                config_hash
            );
    END IF;

    FOR old_constraint IN
        SELECT constraint_record.conname
        FROM pg_constraint AS constraint_record
        JOIN pg_class AS table_record
          ON table_record.oid = constraint_record.conrelid
        JOIN pg_namespace AS schema_record
          ON schema_record.oid = table_record.relnamespace
        WHERE schema_record.nspname = 'public'
          AND table_record.relname =
              'research_lab_attested_execution_receipts_v2'
          AND constraint_record.contype = 'u'
          AND ARRAY(
              SELECT attribute.attname::TEXT
              FROM unnest(constraint_record.conkey) WITH ORDINALITY
                   AS constrained_column(attnum, position)
              JOIN pg_attribute AS attribute
                ON attribute.attrelid = table_record.oid
               AND attribute.attnum = constrained_column.attnum
              ORDER BY constrained_column.position
          ) = ARRAY[
              'role',
              'purpose',
              'job_id',
              'input_root',
              'config_hash'
          ]::TEXT[]
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_attested_execution_receipts_v2 '
            'DROP CONSTRAINT %I',
            old_constraint.conname
        );
    END LOOP;

    SELECT constraint_record.conname
      INTO epoch_constraint_name
    FROM pg_constraint AS constraint_record
    JOIN pg_class AS table_record
      ON table_record.oid = constraint_record.conrelid
    JOIN pg_namespace AS schema_record
      ON schema_record.oid = table_record.relnamespace
    WHERE schema_record.nspname = 'public'
      AND table_record.relname =
          'research_lab_attested_execution_receipts_v2'
      AND constraint_record.contype = 'u'
      AND ARRAY(
          SELECT attribute.attname::TEXT
          FROM unnest(constraint_record.conkey) WITH ORDINALITY
               AS constrained_column(attnum, position)
          JOIN pg_attribute AS attribute
            ON attribute.attrelid = table_record.oid
           AND attribute.attnum = constrained_column.attnum
          ORDER BY constrained_column.position
      ) = ARRAY[
          'role',
          'purpose',
          'job_id',
          'epoch_id',
          'input_root',
          'config_hash'
      ]::TEXT[];

    IF epoch_constraint_name IS NULL THEN
        RAISE EXCEPTION
            'epoch-scoped V2 receipt idempotency constraint is missing';
    END IF;

    EXECUTE format(
        'COMMENT ON CONSTRAINT %I ON '
        'public.research_lab_attested_execution_receipts_v2 IS %L',
        epoch_constraint_name,
        'Rejects conflicting logical receipts within one epoch without '
        'blocking independently verified receipts in later epochs.'
    );
END
$migration$;

COMMIT;
