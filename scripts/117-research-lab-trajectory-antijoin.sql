-- Research Lab egress reduction: server-side trajectory anti-join.
--
-- project_completed_runs() and backfill_corpus_trace_rows() download up to
-- 5,000 terminal runs and then issue ONE existence SELECT per run against
-- research_trajectories (keyed on the deterministic uuid5 trajectory_id) — the
-- N+1 read that multiplies across every worker and pass.
--
-- research_trajectories has no run_id column (its PK is the client-derived
-- uuid5), so the client computes the candidate trajectory_ids locally (a cheap
-- hash) and this function performs the anti-join in one request, returning only
-- the ids that are NOT yet projected. That collapses the N per-run existence
-- reads into a single call while remaining exactly equivalent (the trajectory_id
-- is deterministic, so id membership is identical to the per-run check).

BEGIN;

CREATE OR REPLACE FUNCTION public.research_lab_missing_trajectory_ids(
    candidate_ids UUID[]
)
RETURNS SETOF UUID
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT c
    FROM pg_catalog.unnest(candidate_ids) AS c
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.research_trajectories t
        WHERE t.trajectory_id = c
    );
$$;

REVOKE ALL ON FUNCTION public.research_lab_missing_trajectory_ids(UUID[])
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_missing_trajectory_ids(UUID[])
    TO service_role;

COMMIT;
