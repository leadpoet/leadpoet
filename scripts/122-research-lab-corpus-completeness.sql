-- Research Lab: reliable rich-capture repair discovery via a completeness marker.
--
-- The earlier backfill delta (research_lab_terminal_runs_missing_traces, script
-- 119) only surfaced runs missing their engine execution_trace. A run whose
-- trace exists but whose provider-usage, evidence, result-ledger, or
-- trajectory-event rows are missing (e.g. a transient provider-ledger insert
-- failure) was never revisited -- a real failure-recovery hole.
--
-- Fix: a single append-only completeness marker. A run is marked complete ONLY
-- after a full per-run inspection (backfill_run_corpus_trace_rows) confirms that
-- EVERY corpus row type is present and no write failed. Discovery returns every
-- projected terminal run WITHOUT that marker, so ANY missing/failed corpus row
-- keeps the run discoverable and repaired -- while fully-complete runs are
-- marked once and never re-inspected, preserving the egress win.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_corpus_complete (
    trajectory_id UUID        PRIMARY KEY,
    run_id        UUID        NOT NULL,
    completed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.research_lab_corpus_complete ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_lab_corpus_complete FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_corpus_complete TO service_role;
DROP POLICY IF EXISTS service_role_all ON public.research_lab_corpus_complete;
CREATE POLICY service_role_all ON public.research_lab_corpus_complete
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Idempotent completeness marker write.
CREATE OR REPLACE FUNCTION public.research_lab_mark_corpus_complete(
    p_trajectory_id UUID,
    p_run_id        UUID
)
RETURNS VOID
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = ''
AS $$
    INSERT INTO public.research_lab_corpus_complete (trajectory_id, run_id)
    VALUES (p_trajectory_id, p_run_id)
    ON CONFLICT (trajectory_id) DO NOTHING;
$$;

-- Discovery: the next p_limit PROJECTED terminal runs whose corpus is not yet
-- confirmed complete (missing OR failed rows of ANY type keep them here).
CREATE OR REPLACE FUNCTION public.research_lab_terminal_runs_needing_corpus(
    p_limit        integer,
    p_newest_first boolean DEFAULT true
)
RETURNS TABLE(run_id uuid)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT q.run_id
  FROM public.research_loop_run_queue_current q
  WHERE q.current_queue_status IN ('completed', 'failed')
    AND EXISTS (
      SELECT 1 FROM public.research_trajectories t
      WHERE t.trajectory_id = public.research_lab_trajectory_id(q.run_id)
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.research_lab_corpus_complete c
      WHERE c.trajectory_id = public.research_lab_trajectory_id(q.run_id)
    )
  ORDER BY
    (CASE WHEN p_newest_first THEN q.current_status_at END) DESC NULLS LAST,
    (CASE WHEN NOT p_newest_first THEN q.current_status_at END) ASC NULLS LAST
  LIMIT GREATEST(1, p_limit);
$$;

REVOKE ALL ON FUNCTION public.research_lab_mark_corpus_complete(UUID, UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_terminal_runs_needing_corpus(integer, boolean) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_mark_corpus_complete(UUID, UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_terminal_runs_needing_corpus(integer, boolean) TO service_role;

COMMIT;
