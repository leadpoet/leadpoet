-- Score-backfill calibration rows (inner-loop activation Phase 5).
--
-- One append-only row per scored candidate reconciling what the loop
-- PREDICTED (planner hypothesis predicted_delta), what the cheap dev rung
-- SCORED (ranking-only aggregate_dev_score, when Phase 3 is live), and what
-- the full benchmark REALIZED (score-bundle aggregates + keep/discard
-- outcome). Three consumers:
--   * lesson retrieval hydrates realized deltas onto reflection lessons
--     (score-aware lessons, keyed by node_id),
--   * planner attempt memory shows "this path scored X", not just "tried",
--   * this table IS the future training set for the deferred learned
--     quality predictor (revisit at ~1k rows; do not build the model yet).

CREATE TABLE IF NOT EXISTS public.research_lab_score_calibration (
    calibration_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    candidate_id         TEXT        NOT NULL,
    run_id               TEXT        NOT NULL DEFAULT '',
    node_id              TEXT        NOT NULL DEFAULT '',
    island               TEXT        NOT NULL DEFAULT '',
    lane                 TEXT        NOT NULL DEFAULT '',
    plan_path_id         TEXT        NOT NULL DEFAULT '',
    predicted_delta      DOUBLE PRECISION,
    dev_score            DOUBLE PRECISION,
    dev_score_version    TEXT        NOT NULL DEFAULT '',
    realized_mean_delta  DOUBLE PRECISION,
    realized_delta_lcb   DOUBLE PRECISION,
    outcome              TEXT        NOT NULL DEFAULT '',
    score_bundle_id      TEXT        NOT NULL,
    created_by           TEXT        NOT NULL DEFAULT '',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- One calibration row per scored bundle: idempotent backfill, safe retry.
    UNIQUE (candidate_id, score_bundle_id)
);

-- Lesson/attempt-memory hydration path (newest row per node).
CREATE INDEX IF NOT EXISTS idx_research_lab_score_calibration_node
    ON public.research_lab_score_calibration (node_id, created_at DESC);

-- Predictor-training / drift-monitoring scans by lane over time.
CREATE INDEX IF NOT EXISTS idx_research_lab_score_calibration_lane
    ON public.research_lab_score_calibration (lane, created_at DESC);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_score_calibration_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        'research_lab_score_calibration is append-only; insert a fresh row instead';
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_score_calibration_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_score_calibration_mutation()
    TO service_role;

DROP TRIGGER IF EXISTS prevent_research_lab_score_calibration_mutation
    ON public.research_lab_score_calibration;

CREATE TRIGGER prevent_research_lab_score_calibration_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_score_calibration
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_score_calibration_mutation();

REVOKE ALL ON TABLE public.research_lab_score_calibration FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_score_calibration TO service_role;

ALTER TABLE public.research_lab_score_calibration ENABLE ROW LEVEL SECURITY;

-- Postgres has no CREATE POLICY IF NOT EXISTS; drop-then-create keeps this
-- migration safely re-runnable (a bare re-run otherwise fails 42710).
DROP POLICY IF EXISTS service_role_read ON public.research_lab_score_calibration;
CREATE POLICY service_role_read ON public.research_lab_score_calibration
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_score_calibration;
CREATE POLICY service_role_insert ON public.research_lab_score_calibration
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_score_calibration IS
    'Per-scored-candidate predicted vs dev vs realized deltas + outcome. Feeds score-aware lessons/attempt memory; future training table for the deferred quality predictor.';
