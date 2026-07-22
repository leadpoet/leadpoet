-- Research Lab egress reduction: server-side candidate selection.
--
-- Scoring workers claimed the next candidate by pulling up to 50 full current
-- rows (SELECT *, so every large patch artifact) on every poll, filtering them
-- client-side, then read-write-reading to append the "assigned" event. With N
-- workers polling that is N x 50 full-artifact rows per poll cycle -- the bulk
-- of scoring-side egress.
--
-- This function applies the reason/staleness eligibility filter in the database
-- and returns only the SINGLE next claimable candidate (still the full current
-- row, because the caller scores it immediately and needs the artifact). It
-- does NOT append the claim event: the "assigned" append stays in Python so the
-- anchored_hash canonicalization lives in exactly one place, and the existing
-- guard_research_lab_candidate_claim trigger (script 42) already serializes the
-- append and rejects a double-claim with SQLSTATE 23505. So concurrency safety
-- is unchanged; only the selection read shrinks from ~50 full rows to 1.
--
-- The eligibility filter mirrors scoring_worker._status_is_stale exactly:
-- a candidate parked with reason 'baseline_not_ready' or a retryable-failure
-- reason is only re-claimable once its status is older than the matching retry
-- window (floored at 60s, matching max(60, ...) in Python).

BEGIN;

CREATE OR REPLACE FUNCTION public.claim_next_research_lab_candidate(
    p_baseline_not_ready_retry_seconds INTEGER,
    p_retryable_failure_retry_seconds  INTEGER
)
RETURNS SETOF public.research_lab_candidate_evaluation_current
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT v.*
    FROM public.research_lab_candidate_evaluation_current v
    WHERE v.current_candidate_status = 'queued'
      AND NOT (
            v.current_reason = 'baseline_not_ready'
            AND v.current_status_at > pg_catalog.now()
                - pg_catalog.make_interval(
                    secs => GREATEST(60, COALESCE(p_baseline_not_ready_retry_seconds, 900)))
      )
      AND NOT (
            v.current_reason IN (
                'candidate_scoring_retryable_failure',
                'conditional_validation_retryable_failure')
            AND v.current_status_at > pg_catalog.now()
                - pg_catalog.make_interval(
                    secs => GREATEST(60, COALESCE(p_retryable_failure_retry_seconds, 300)))
      )
    ORDER BY v.current_status_at ASC
    LIMIT 1;
$$;

REVOKE ALL ON FUNCTION public.claim_next_research_lab_candidate(INTEGER, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_next_research_lab_candidate(INTEGER, INTEGER)
    TO service_role;

COMMENT ON FUNCTION public.claim_next_research_lab_candidate(INTEGER, INTEGER) IS
    'Returns the single next eligible queued Research Lab candidate (full current row), applying the baseline/retryable staleness filter server-side. The assigned-event append stays client-side under the guard_research_lab_candidate_claim trigger.';

COMMIT;

-- Smoke check after applying:
--
--   SELECT candidate_id, current_candidate_status, current_reason
--   FROM public.claim_next_research_lab_candidate(900, 300);
