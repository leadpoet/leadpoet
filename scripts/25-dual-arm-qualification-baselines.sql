-- Research Lab Phase 0: dual-arm baseline support.
--
-- Expand step for dual-arm baseline support. Adds model_id and a composite
-- uniqueness target while preserving the existing set_id primary key so old
-- and new code can overlap safely during deploy.
--
-- Review before production:
--   1. Confirm there are no duplicate (set_id, model_id) rows.
--   2. Apply before deploying code that uses on_conflict='set_id,model_id'.
--   3. This does NOT yet allow two rows with the same set_id; the contract
--      step that drops the legacy set_id primary key must happen with the
--      Arm B runner wiring.

BEGIN;

ALTER TABLE qualification_baselines
    ADD COLUMN IF NOT EXISTS model_id TEXT NOT NULL
    DEFAULT 'reference:qualification_model:v1';

DO $$
DECLARE
    set_id_attnum SMALLINT;
    model_id_attnum SMALLINT;
BEGIN
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

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.conrelid = 'qualification_baselines'::regclass
          AND c.contype IN ('p', 'u')
          AND c.conkey = ARRAY[set_id_attnum, model_id_attnum]::SMALLINT[]
    ) THEN
        ALTER TABLE qualification_baselines
            ADD CONSTRAINT qualification_baselines_set_id_model_id_key
            UNIQUE (set_id, model_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_qualification_baselines_model_id
    ON qualification_baselines(model_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE qualification_baselines TO service_role;

COMMENT ON TABLE qualification_baselines IS
    'Daily fixed-arm baseline scores per ICP set. Champion selection reads the reserved reference arm; Research Lab Phase 0 reads paired model arms.';

COMMENT ON COLUMN qualification_baselines.model_id IS
    'Fixed artifact or model arm evaluated on this ICP set. The reserved reference arm is reference:qualification_model:v1.';

COMMENT ON COLUMN qualification_baselines.baseline_score IS
    'Average of per-ICP normalized scores (per-ICP = sum(score_company over leads) / 5).';

COMMIT;
