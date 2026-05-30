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
-- Keyed on set_id (matches qualification_private_icp_sets.set_id), so the
-- baseline is naturally invalidated the moment a new ICP set is activated:
-- champion-selection's `load_baseline(today_set_id)` returns None until the
-- new day's run completes, falling back to legacy champion-only thresholding
-- (no regression).
--
-- Service role only — RLS-protected like the private ICP table, since the
-- per-ICP scores reveal information about how the reference model performed
-- against the still-private ICPs of the day.
--
-- Run on production Supabase via SQL editor or psql.

CREATE TABLE IF NOT EXISTS qualification_baselines (
    set_id              BIGINT       PRIMARY KEY,
    baseline_score      DOUBLE PRECISION NOT NULL,
    per_icp_scores      JSONB        NOT NULL DEFAULT '[]'::JSONB,
    scored_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    model_id            TEXT         NOT NULL DEFAULT 'reference:qualification_model:v1',
    icp_set_hash        TEXT,
    run_duration_seconds DOUBLE PRECISION,
    run_status          TEXT         NOT NULL DEFAULT 'completed'
                                     CHECK (run_status IN ('completed', 'failed', 'partial')),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- RLS: lock down to service_role only (same as qualification_private_icp_sets).
-- Validators reading this table use the service role via their own client.
ALTER TABLE qualification_baselines ENABLE ROW LEVEL SECURITY;

-- Reset any prior policies on this table (idempotent re-run support).
DROP POLICY IF EXISTS service_role_all ON qualification_baselines;

CREATE POLICY service_role_all ON qualification_baselines
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE  qualification_baselines IS
    'Daily reference-model baseline score per ICP set. Champion selection compares '
    'challengers against baseline_score + CHAMPION_DETHRONING_THRESHOLD_POINTS.';
COMMENT ON COLUMN qualification_baselines.set_id IS
    'Matches qualification_private_icp_sets.set_id (the active set when this run executed).';
COMMENT ON COLUMN qualification_baselines.baseline_score IS
    'Sum of per-ICP normalized scores (per-ICP = sum(score_company over leads) / 5).';
COMMENT ON COLUMN qualification_baselines.per_icp_scores IS
    'JSONB array of per-ICP scores in evaluation order; kept for traceability.';
COMMENT ON COLUMN qualification_baselines.run_status IS
    'completed | failed | partial. Champion selection treats anything other than '
    '"completed" as no-baseline (load_baseline returns None).';
