-- Research Lab: hybrid fresh/retained ICP benchmark window metadata.
--
-- Existing rolling-window rows are immutable and remain schema_version 1.0.
-- New hybrid rows use schema_version 1.1 plus nullable audit columns that
-- duplicate the public window_doc metadata for operator/read-only checks.

BEGIN;

ALTER TABLE public.research_lab_rolling_icp_windows
    DROP CONSTRAINT IF EXISTS research_lab_rolling_icp_windows_schema_version_check;

ALTER TABLE public.research_lab_rolling_icp_windows
    ADD CONSTRAINT research_lab_rolling_icp_windows_schema_version_check
    CHECK (schema_version IN ('1.0', '1.1'));

ALTER TABLE public.research_lab_rolling_icp_windows
    ADD COLUMN IF NOT EXISTS window_mode TEXT
        CHECK (window_mode IS NULL OR window_mode IN ('hybrid_fresh_retained', 'legacy_rolling')),
    ADD COLUMN IF NOT EXISTS selection_policy TEXT,
    ADD COLUMN IF NOT EXISTS fresh_set_id INTEGER,
    ADD COLUMN IF NOT EXISTS fresh_icp_count INTEGER
        CHECK (fresh_icp_count IS NULL OR fresh_icp_count >= 0),
    ADD COLUMN IF NOT EXISTS retained_icp_count INTEGER
        CHECK (retained_icp_count IS NULL OR retained_icp_count >= 0),
    ADD COLUMN IF NOT EXISTS min_new_icp_count INTEGER
        CHECK (min_new_icp_count IS NULL OR min_new_icp_count >= 0);

CREATE INDEX IF NOT EXISTS idx_research_lab_rolling_icp_windows_mode_created
    ON public.research_lab_rolling_icp_windows(window_mode, created_at DESC);

COMMENT ON COLUMN public.research_lab_rolling_icp_windows.window_mode IS
    'Hybrid window mode for new Research Lab benchmark windows; NULL for historical rows.';
COMMENT ON COLUMN public.research_lab_rolling_icp_windows.fresh_icp_count IS
    'Number of selected ICP refs from the latest daily private ICP set.';
COMMENT ON COLUMN public.research_lab_rolling_icp_windows.retained_icp_count IS
    'Number of selected ICP refs retained from prior daily private ICP sets.';
COMMENT ON COLUMN public.research_lab_rolling_icp_windows.min_new_icp_count IS
    'Configured minimum fresh/new ICP refs required for the window.';

COMMIT;
