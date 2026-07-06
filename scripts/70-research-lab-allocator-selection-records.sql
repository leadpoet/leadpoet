-- Allocator-priors nightly selection records (inner-loop activation Phase 2).
--
-- One row per (day, ledger-window) deterministic Thompson selection over the
-- results-ledger cell-yield priors. Persisting the nightly selection gives:
--   * one stable daily ranking every worker injects identically (instead of
--     each run recomputing over a possibly-shifted ledger window),
--   * an audit trail of what hint the planner actually saw,
--   * a cheap read path for the loop engine (single-row lookup per run).
--
-- The selection is a prompt ORDERING HINT only: it never funds, merges, or
-- rejects anything (meta_allocator live-allocation flags stay false).
-- Append-only: a fresher window on the same day inserts a new row; readers
-- take the newest row per day.

CREATE TABLE IF NOT EXISTS public.research_lab_allocator_selection_records (
    selection_record_id UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version      TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    selection_id        TEXT        NOT NULL,
    day                 TEXT        NOT NULL CHECK (day ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'),
    seed                BIGINT      NOT NULL,
    window_hash         TEXT        NOT NULL,
    window_row_count    INTEGER     NOT NULL CHECK (window_row_count >= 0),
    cell_count          INTEGER     NOT NULL CHECK (cell_count >= 0),
    selection_doc       JSONB       NOT NULL CHECK (
                                      jsonb_typeof(selection_doc) = 'object'
                                      AND selection_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                      ),
    created_by          TEXT        NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Same day + same ledger window => same deterministic selection; the
    -- refresh job is idempotent against this key.
    UNIQUE (day, window_hash)
);

-- Reader path: newest selection for a given day.
CREATE INDEX IF NOT EXISTS idx_research_lab_allocator_selection_day
    ON public.research_lab_allocator_selection_records (day, created_at DESC);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_allocator_selection_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        'research_lab_allocator_selection_records is append-only; insert a fresh selection instead';
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_allocator_selection_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_allocator_selection_mutation()
    TO service_role;

DROP TRIGGER IF EXISTS prevent_research_lab_allocator_selection_mutation
    ON public.research_lab_allocator_selection_records;

CREATE TRIGGER prevent_research_lab_allocator_selection_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_allocator_selection_records
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_lab_allocator_selection_mutation();

REVOKE ALL ON TABLE public.research_lab_allocator_selection_records FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_allocator_selection_records TO service_role;

ALTER TABLE public.research_lab_allocator_selection_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY service_role_read ON public.research_lab_allocator_selection_records
    FOR SELECT TO service_role USING (true);
CREATE POLICY service_role_insert ON public.research_lab_allocator_selection_records
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_allocator_selection_records IS
    'Nightly deterministic Thompson selections over results-ledger cell-yield priors. Prompt ordering hints only; never funding or promotion inputs.';
