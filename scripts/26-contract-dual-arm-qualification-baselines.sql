-- Research Lab Phase 0: dual-arm baseline contract step.
--
-- Converts qualification_baselines from the P0.1a expand state
--   PRIMARY KEY (set_id) + UNIQUE (set_id, model_id)
-- to the P0.1b final state
--   PRIMARY KEY (set_id, model_id)
-- so two fixed arms can store rows for the same daily ICP set.
--
-- Deployment policy:
--   * Do not apply until all Phase 0 code + SQL files are written/reviewed.
--   * Apply after script 25 and smoke-test before enabling lab workflows.
--   * Keep RESEARCH_LAB_DUAL_ARM_BASELINES unset/false until this migration
--     has been applied and verified.
--   * Set RESEARCH_LAB_DUAL_ARM_CONTRACT_READY=1 only after this migration's
--     smoke tests pass. The runner requires both flags before Arm B appears.

BEGIN;

LOCK TABLE qualification_baselines IN ACCESS EXCLUSIVE MODE;

ALTER TABLE qualification_baselines
    ADD COLUMN IF NOT EXISTS model_id TEXT
    DEFAULT 'reference:qualification_model:v1';

UPDATE qualification_baselines
SET model_id = 'reference:qualification_model:v1'
WHERE model_id IS NULL;

ALTER TABLE qualification_baselines
    ALTER COLUMN model_id SET DEFAULT 'reference:qualification_model:v1',
    ALTER COLUMN model_id SET NOT NULL,
    ALTER COLUMN set_id SET NOT NULL;

DO $$
DECLARE
    set_id_attnum SMALLINT;
    model_id_attnum SMALLINT;
    primary_key_name TEXT;
    primary_key_cols SMALLINT[];
    redundant_unique_name TEXT;
BEGIN
    IF EXISTS (
        SELECT 1
        FROM qualification_baselines
        GROUP BY set_id, model_id
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'qualification_baselines has duplicate (set_id, model_id) rows';
    END IF;

    SELECT attnum INTO set_id_attnum
    FROM pg_attribute
    WHERE attrelid = 'qualification_baselines'::regclass
      AND attname = 'set_id'
      AND NOT attisdropped;

    SELECT attnum INTO model_id_attnum
    FROM pg_attribute
    WHERE attrelid = 'qualification_baselines'::regclass
      AND attname = 'model_id'
      AND NOT attisdropped;

    SELECT conname, conkey INTO primary_key_name, primary_key_cols
    FROM pg_constraint
    WHERE conrelid = 'qualification_baselines'::regclass
      AND contype = 'p'
    LIMIT 1;

    IF primary_key_cols IS DISTINCT FROM ARRAY[set_id_attnum, model_id_attnum]::SMALLINT[] THEN
        IF primary_key_name IS NOT NULL THEN
            EXECUTE format(
                'ALTER TABLE qualification_baselines DROP CONSTRAINT %I',
                primary_key_name
            );
        END IF;

        ALTER TABLE qualification_baselines
            ADD CONSTRAINT qualification_baselines_pkey
            PRIMARY KEY (set_id, model_id);
    END IF;

    FOR redundant_unique_name IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'qualification_baselines'::regclass
          AND contype = 'u'
          AND conkey = ARRAY[set_id_attnum, model_id_attnum]::SMALLINT[]
    LOOP
        EXECUTE format(
            'ALTER TABLE qualification_baselines DROP CONSTRAINT %I',
            redundant_unique_name
        );
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_qualification_baselines_model_id
    ON qualification_baselines(model_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE qualification_baselines TO service_role;

COMMENT ON TABLE qualification_baselines IS
    'Daily fixed-arm baseline scores per ICP set. Stores one row per fixed arm on each daily ICP set.';

COMMENT ON COLUMN qualification_baselines.model_id IS
    'Fixed artifact or model arm evaluated on this ICP set. The reserved reference arm is reference:qualification_model:v1.';

COMMENT ON COLUMN qualification_baselines.baseline_score IS
    'Average of per-ICP normalized scores (per-ICP = sum(score_company over leads) / 5).';

COMMIT;
