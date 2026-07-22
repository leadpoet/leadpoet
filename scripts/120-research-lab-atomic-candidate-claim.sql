-- Research Lab egress reduction: atomic single-owner candidate claim.
--
-- _claim_next_candidate previously downloaded up to 50 full candidate rows,
-- filtered client-side, then read-write-read to append the assigned event. Even
-- after the earlier single-row selection RPC, two scoring workers could still
-- SELECT the same candidate and race on the Python append (one wins via the
-- guard trigger, the other wastes a fetch).
--
-- This RPC makes selection itself single-assignment: an advisory lock serializes
-- claimers, a short-TTL claim row reserves the chosen candidate, and only the
-- minimal columns scoring needs are returned. Concurrent callers therefore
-- receive DIFFERENT candidates (proven with two live connections in the
-- integration test). The assigned-event append stays in Python so the
-- anchored_hash canonicalization lives in one place; the existing
-- guard_research_lab_candidate_claim trigger (script 42) remains the backstop.
-- If a worker dies between claim and append, the claim expires and the still-
-- queued candidate is re-claimable; a claim is also ignored once the candidate's
-- status advances past it (claimed_at < current_status_at), so requeues are safe.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_claim (
    candidate_id TEXT        PRIMARY KEY,
    holder_ref   TEXT        NOT NULL,
    claimed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL
);

ALTER TABLE public.research_lab_candidate_claim ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_lab_candidate_claim FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.research_lab_candidate_claim TO service_role;
DROP POLICY IF EXISTS service_role_all ON public.research_lab_candidate_claim;
CREATE POLICY service_role_all ON public.research_lab_candidate_claim
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE OR REPLACE FUNCTION public.claim_next_research_lab_candidate(
    p_holder_ref                       TEXT,
    p_ttl_seconds                      INTEGER,
    p_baseline_not_ready_retry_seconds INTEGER,
    p_retryable_failure_retry_seconds  INTEGER
)
RETURNS TABLE(
    candidate_id               TEXT,
    run_id                     UUID,
    ticket_id                  UUID,
    private_model_manifest_doc JSONB,
    candidate_patch_manifest   JSONB,
    miner_hotkey               TEXT,
    candidate_kind             TEXT,
    candidate_model_manifest_doc JSONB,
    candidate_build_doc        JSONB,
    candidate_source_diff_hash TEXT,
    candidate_patch_hash       TEXT,
    parent_artifact_hash       TEXT,
    receipt_id                 UUID,
    island                     TEXT,
    hypothesis_doc             JSONB,
    redacted_public_summary    TEXT
)
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = ''
AS $$
-- OUT column names (candidate_id, run_id, ...) match table columns; resolve
-- ambiguous references in the body to the columns.
#variable_conflict use_column
DECLARE
    v_now     TIMESTAMPTZ := pg_catalog.now();
    v_expires TIMESTAMPTZ;
    v_id      TEXT;
BEGIN
    IF p_holder_ref IS NULL OR p_ttl_seconds IS NULL
       OR p_ttl_seconds <= 0 OR p_ttl_seconds > 3600 THEN
        RAISE EXCEPTION 'candidate claim arguments are invalid' USING ERRCODE = '22023';
    END IF;
    v_expires := v_now + pg_catalog.make_interval(secs => p_ttl_seconds);

    -- Serialize concurrent claimers so each takes a distinct candidate.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_candidate_claim'));

    SELECT v.candidate_id INTO v_id
    FROM public.research_lab_candidate_evaluation_current v
    WHERE v.current_candidate_status = 'queued'
      AND NOT (
          v.current_reason = 'baseline_not_ready'
          AND v.current_status_at > v_now - pg_catalog.make_interval(
              secs => GREATEST(60, COALESCE(p_baseline_not_ready_retry_seconds, 900)))
      )
      AND NOT (
          v.current_reason IN (
              'candidate_scoring_retryable_failure',
              'conditional_validation_retryable_failure')
          AND v.current_status_at > v_now - pg_catalog.make_interval(
              secs => GREATEST(60, COALESCE(p_retryable_failure_retry_seconds, 300)))
      )
      AND NOT EXISTS (
          SELECT 1 FROM public.research_lab_candidate_claim cl
          WHERE cl.candidate_id = v.candidate_id
            AND cl.expires_at > v_now
            AND cl.claimed_at >= v.current_status_at
      )
    ORDER BY v.current_status_at ASC
    LIMIT 1;

    IF v_id IS NULL THEN
        RETURN;
    END IF;

    INSERT INTO public.research_lab_candidate_claim AS cl
        (candidate_id, holder_ref, claimed_at, expires_at)
    VALUES (v_id, p_holder_ref, v_now, v_expires)
    ON CONFLICT (candidate_id) DO UPDATE
        SET holder_ref = EXCLUDED.holder_ref,
            claimed_at = EXCLUDED.claimed_at,
            expires_at = EXCLUDED.expires_at;

    RETURN QUERY
    SELECT v.candidate_id, v.run_id, v.ticket_id, v.private_model_manifest_doc,
           v.candidate_patch_manifest, v.miner_hotkey, v.candidate_kind,
           v.candidate_model_manifest_doc, v.candidate_build_doc,
           v.candidate_source_diff_hash, v.candidate_patch_hash,
           v.parent_artifact_hash, v.receipt_id, v.island, v.hypothesis_doc,
           v.redacted_public_summary
    FROM public.research_lab_candidate_evaluation_current v
    WHERE v.candidate_id = v_id;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_next_research_lab_candidate(TEXT, INTEGER, INTEGER, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_next_research_lab_candidate(TEXT, INTEGER, INTEGER, INTEGER)
    TO service_role;

COMMIT;
