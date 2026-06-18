-- Reference-model baseline persistence.
--
-- The daily reference-model run sets the floor that miners must exceed by
-- CHAMPION_DETHRONING_THRESHOLD_POINTS (10) to become or dethrone the
-- qualification champion. See:
--   * miner_models/qualification_model/           — the reference model itself
--   * qualification/scoring/baseline.py            — load/save accessors
--   * qualification/scoring/champion.py            — threshold-floor consumer
--   * gateway/tasks/icp_generator.py               — daily runner (right after ICP activation)
--
-- Phase-0 final state: keyed on (set_id, model_id), allowing multiple fixed
-- arms to be measured on the same daily ICP set.
--
-- Service role only — RLS-protected like the private ICP table, since the
-- per-ICP scores reveal information about how fixed arms performed against
-- the still-private ICPs of the day.
--
-- Run on production Supabase via SQL editor or psql.

CREATE TABLE IF NOT EXISTS qualification_baselines (
    set_id              BIGINT       NOT NULL,
    model_id            TEXT         NOT NULL DEFAULT 'reference:qualification_model:v1',
    baseline_score      DOUBLE PRECISION NOT NULL,
    per_icp_scores      JSONB        NOT NULL DEFAULT '[]'::JSONB,
    scored_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    icp_set_hash        TEXT,
    run_duration_seconds DOUBLE PRECISION,
    run_status          TEXT         NOT NULL DEFAULT 'completed'
                                     CHECK (run_status IN ('completed', 'failed', 'partial')),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT qualification_baselines_pkey PRIMARY KEY (set_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_qualification_baselines_model_id
    ON qualification_baselines(model_id);

-- RLS: lock down to service_role only (same as qualification_private_icp_sets).
-- Validators reading this table use the service role via their own client.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE qualification_baselines TO service_role;

ALTER TABLE qualification_baselines ENABLE ROW LEVEL SECURITY;

-- Reset any prior policies on this table (idempotent re-run support).
DROP POLICY IF EXISTS service_role_all ON qualification_baselines;

CREATE POLICY service_role_all ON qualification_baselines
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE  qualification_baselines IS
    'Daily fixed-arm baseline scores per ICP set. Phase-0 final state stores '
    'one row per fixed arm on each daily ICP set.';
COMMENT ON COLUMN qualification_baselines.set_id IS
    'Matches qualification_private_icp_sets.set_id (the active set when this run executed).';
COMMENT ON COLUMN qualification_baselines.model_id IS
    'Fixed artifact or model arm evaluated on this ICP set. The reserved reference arm is reference:qualification_model:v1.';
COMMENT ON COLUMN qualification_baselines.baseline_score IS
    'Average of per-ICP normalized scores (per-ICP = sum(score_company over leads) / 5).';
COMMENT ON COLUMN qualification_baselines.per_icp_scores IS
    'JSONB array of per-ICP scores in evaluation order; kept for traceability.';
COMMENT ON COLUMN qualification_baselines.run_status IS
    'completed | failed | partial. Champion selection treats anything other than '
    '"completed" as no-baseline (load_baseline returns None).';
