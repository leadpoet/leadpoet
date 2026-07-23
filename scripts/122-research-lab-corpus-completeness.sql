-- Research Lab: reliable rich-capture repair discovery via a completeness marker
-- carrying a SOURCE WATERMARK.
--
-- The earlier backfill delta (research_lab_terminal_runs_missing_traces, script
-- 119) only surfaced runs missing their engine execution_trace. A run whose
-- trace exists but whose provider-usage, evidence, result-ledger, or
-- trajectory-event rows are missing (e.g. a transient provider-ledger insert
-- failure) was never revisited -- a real failure-recovery hole.
--
-- Fix: an append-only completeness marker stamped with a fingerprint of the
-- run's projection SOURCE data. A run is marked complete ONLY after a full
-- per-run inspection (backfill_run_corpus_trace_rows) confirms EVERY corpus row
-- type is present. Discovery returns every projected terminal run whose marker
-- is absent OR whose stored watermark no longer matches the CURRENT source
-- watermark -- so late loop / promotion / version / score-bundle / evaluation
-- events arriving after the mark change the watermark and make the run
-- discoverable (and re-projectable) again. The watermark is computed BEFORE the
-- inspection and stored at mark time, so an event landing mid-inspection also
-- forces rediscovery (stale watermark stored -> mismatch next pass).

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_corpus_complete (
    trajectory_id    UUID        PRIMARY KEY,
    run_id           UUID        NOT NULL,
    source_watermark TEXT        NOT NULL DEFAULT '',
    completed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE public.research_lab_corpus_complete
    ADD COLUMN IF NOT EXISTS source_watermark TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_corpus_complete ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_lab_corpus_complete FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_corpus_complete TO service_role;
DROP POLICY IF EXISTS service_role_all ON public.research_lab_corpus_complete;
CREATE POLICY service_role_all ON public.research_lab_corpus_complete
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- The source watermark filters evaluation events by run_id and reads MAX(seq)
-- for every candidate run. Keep that lookup indexed as the corpus grows.
CREATE INDEX IF NOT EXISTS research_lab_candidate_evaluation_events_run_seq_idx
    ON public.research_lab_candidate_evaluation_events (run_id, seq DESC);

-- Fingerprint of every projection source for a run, mirroring
-- load_projection_inputs: loop events / candidate artifacts / evaluation events
-- (by run_id), promotion events (by the run's candidate ids), version events
-- (by the promotions' version ids), and score bundles (by run_id). Any late
-- append to ANY of these changes the watermark.
CREATE OR REPLACE FUNCTION public.research_lab_corpus_source_watermark(p_run_id uuid)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH cands AS (
    SELECT ca.candidate_id
    FROM public.research_lab_candidate_artifacts ca
    WHERE ca.run_id = p_run_id
  ),
  promos AS (
    SELECT pe.private_model_version_id, pe.created_at
    FROM public.research_lab_candidate_promotion_events pe
    WHERE pe.candidate_id IN (SELECT candidate_id FROM cands)
  )
  SELECT 'le:' || (SELECT COUNT(*) || ':' || COALESCE(MAX(le.seq), -1)
                   FROM public.research_lab_auto_research_loop_events le
                   WHERE le.run_id = p_run_id)
      || '|ca:' || (SELECT COUNT(*) FROM cands)
      || '|ee:' || (SELECT COUNT(*) || ':' || COALESCE(MAX(ee.seq), -1)
                    FROM public.research_lab_candidate_evaluation_events ee
                    WHERE ee.run_id = p_run_id)
      || '|pe:' || (SELECT COUNT(*) || ':' ||
                    COALESCE(pg_catalog.date_part('epoch', MAX(p.created_at))::text, '')
                    FROM promos p)
      || '|ve:' || (SELECT COUNT(*) || ':' || COALESCE(MAX(ve.seq), -1)
                    FROM public.research_lab_private_model_version_events ve
                    WHERE ve.private_model_version_id IN
                        (SELECT DISTINCT private_model_version_id FROM promos
                         WHERE private_model_version_id IS NOT NULL))
      || '|sb:' || (SELECT COUNT(*) || ':' ||
                    COALESCE(pg_catalog.date_part('epoch', MAX(sb.created_at))::text, '')
                    FROM public.research_evaluation_score_bundles sb
                    WHERE sb.run_id = p_run_id)
$$;

-- Idempotent completeness-marker write. Re-marking updates the watermark (a
-- rediscovered run that is repaired gets its NEW watermark recorded).
CREATE OR REPLACE FUNCTION public.research_lab_mark_corpus_complete(
    p_trajectory_id    UUID,
    p_run_id           UUID,
    p_source_watermark TEXT DEFAULT NULL
)
RETURNS VOID
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = ''
AS $$
    INSERT INTO public.research_lab_corpus_complete
        (trajectory_id, run_id, source_watermark, completed_at)
    VALUES (
        p_trajectory_id,
        p_run_id,
        COALESCE(p_source_watermark,
                 public.research_lab_corpus_source_watermark(p_run_id)),
        pg_catalog.now()
    )
    ON CONFLICT (trajectory_id) DO UPDATE
        SET source_watermark = EXCLUDED.source_watermark,
            completed_at     = EXCLUDED.completed_at;
$$;

-- Discovery: the next p_limit PROJECTED terminal runs whose corpus is not yet
-- confirmed complete FOR THE CURRENT SOURCE DATA. Missing marker, failed rows,
-- or a watermark drifted by late events all keep a run here.
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
        AND c.source_watermark = public.research_lab_corpus_source_watermark(q.run_id)
    )
  ORDER BY
    (CASE WHEN p_newest_first THEN q.current_status_at END) DESC NULLS LAST,
    (CASE WHEN NOT p_newest_first THEN q.current_status_at END) ASC NULLS LAST
  LIMIT GREATEST(1, p_limit);
$$;

REVOKE ALL ON FUNCTION public.research_lab_corpus_source_watermark(UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_mark_corpus_complete(UUID, UUID, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_terminal_runs_needing_corpus(integer, boolean) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_corpus_source_watermark(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_mark_corpus_complete(UUID, UUID, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_terminal_runs_needing_corpus(integer, boolean) TO service_role;

COMMIT;
