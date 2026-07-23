-- Fail exhausted global ICP queue generations atomically.
--
-- The queue intentionally marks held downstream jobs failed when a scoring
-- gate rejects a candidate. An actual public/private/conditional execution
-- failure is different: it must close the whole generation and must never be
-- interpreted as a completed phase or assembled into a partial score bundle.
-- Apply after migrations 83 and 97.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_scoring_job_candidate
    ADD COLUMN IF NOT EXISTS failure_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
    ADD COLUMN IF NOT EXISTS failure_projection_status TEXT NOT NULL
        DEFAULT 'not_required',
    ADD COLUMN IF NOT EXISTS failure_projection_claimed_by TEXT NOT NULL
        DEFAULT '',
    ADD COLUMN IF NOT EXISTS failure_projection_lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS failure_projection_attempt_count INTEGER NOT NULL
        DEFAULT 0;

ALTER TABLE public.research_lab_scoring_job_candidate
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_candidate_failure_doc_check;
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD CONSTRAINT research_lab_scoring_job_candidate_failure_doc_check CHECK (
        jsonb_typeof(failure_doc) = 'object'
        AND failure_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|intent_signals|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
    ) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_candidate
    VALIDATE CONSTRAINT research_lab_scoring_job_candidate_failure_doc_check;

ALTER TABLE public.research_lab_scoring_job_candidate
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_candidate_failure_projection_check;
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD CONSTRAINT research_lab_scoring_job_candidate_failure_projection_check CHECK (
        failure_projection_status IN (
            'not_required',
            'pending',
            'projecting',
            'projected'
        )
        AND failure_projection_attempt_count >= 0
    ) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_candidate
    VALIDATE CONSTRAINT research_lab_scoring_job_candidate_failure_projection_check;

ALTER TABLE public.research_lab_scoring_job_candidate
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_candidate_assembly_status_check;
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD CONSTRAINT research_lab_scoring_job_candidate_assembly_status_check CHECK (
        assembly_status IN ('pending', 'assembling', 'assembled', 'failed')
    ) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_candidate
    VALIDATE CONSTRAINT research_lab_scoring_job_candidate_assembly_status_check;

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_queue_failure_projection
    ON public.research_lab_scoring_job_candidate(
        failure_projection_status,
        failure_projection_lease_expires_at,
        updated_at
    )
    WHERE assembly_status = 'failed'
      AND failure_projection_status IN ('pending', 'projecting');

CREATE OR REPLACE FUNCTION public.research_lab_fail_scoring_queue_generation(
    target_job_id UUID,
    expected_claimed_by TEXT,
    expected_attempt_count INTEGER,
    target_failure_class TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    observed_job public.research_lab_scoring_job_queue%ROWTYPE;
    locked_job public.research_lab_scoring_job_queue%ROWTYPE;
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    normalized_failure_class TEXT;
    terminal_doc JSONB;
BEGIN
    IF NULLIF(pg_catalog.btrim(expected_claimed_by), '') IS NULL
       OR expected_attempt_count <= 0
    THEN
        RAISE EXCEPTION 'global queue terminal failure requires an exact claim'
            USING ERRCODE = '22023';
    END IF;

    SELECT *
      INTO observed_job
      FROM public.research_lab_scoring_job_queue
     WHERE job_id = target_job_id;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'committed', FALSE,
            'reason', 'not_found'
        );
    END IF;

    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = observed_job.queue_generation_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'global queue candidate generation is missing'
            USING ERRCODE = '23514';
    END IF;
    IF candidate_row.assembly_status = 'failed' THEN
        RETURN pg_catalog.jsonb_build_object(
            'committed', FALSE,
            'reason', 'already_failed'
        );
    END IF;
    IF candidate_row.assembly_status <> 'pending' THEN
        RETURN pg_catalog.jsonb_build_object(
            'committed', FALSE,
            'reason', 'generation_terminal'
        );
    END IF;

    SELECT *
      INTO locked_job
      FROM public.research_lab_scoring_job_queue
     WHERE job_id = target_job_id
     FOR UPDATE;
    IF locked_job.status <> 'claimed'
       OR locked_job.claimed_by <> expected_claimed_by
       OR locked_job.attempt_count <> expected_attempt_count
    THEN
        RETURN pg_catalog.jsonb_build_object(
            'committed', FALSE,
            'reason', 'claim_changed'
        );
    END IF;

    normalized_failure_class := COALESCE(
        NULLIF(pg_catalog.btrim(target_failure_class), ''),
        'candidate_scoring_error'
    );
    terminal_doc := pg_catalog.jsonb_build_object(
        'schema_version', '1.0',
        'terminal', TRUE,
        'failure_class', normalized_failure_class,
        'failed_phase', locked_job.phase,
        'failed_job_id', locked_job.job_id,
        'attempt_count', locked_job.attempt_count,
        'queue_generation_id', locked_job.queue_generation_id
    );

    UPDATE public.research_lab_scoring_job_queue
       SET status = 'failed',
           result_doc = CASE
               WHEN job_id = locked_job.job_id THEN terminal_doc
               ELSE pg_catalog.jsonb_build_object(
                   'schema_version', '1.0',
                   'terminal', TRUE,
                   'generation_cancelled', TRUE,
                   'failure_class', normalized_failure_class,
                   'failed_job_id', locked_job.job_id,
                   'queue_generation_id', locked_job.queue_generation_id
               )
           END,
           claimed_by = '',
           lease_expires_at = NULL,
           updated_at = NOW()
     WHERE queue_generation_id = locked_job.queue_generation_id
       AND status IN ('held', 'queued', 'claimed');

    UPDATE public.research_lab_scoring_job_candidate
       SET assembly_status = 'failed',
           preliminary_gate_status = CASE
               WHEN preliminary_gate_status IN ('pending', 'deciding')
                   THEN 'skipped'
               ELSE preliminary_gate_status
           END,
           preliminary_gate_proof = CASE
               WHEN preliminary_gate_status IN ('pending', 'deciding')
                   THEN '{}'::JSONB
               ELSE preliminary_gate_proof
           END,
           preliminary_gate_claimed_by = '',
           preliminary_gate_lease_expires_at = NULL,
           failure_doc = terminal_doc,
           failure_projection_status = 'pending',
           failure_projection_claimed_by = '',
           failure_projection_lease_expires_at = NULL,
           updated_at = NOW()
     WHERE queue_generation_id = locked_job.queue_generation_id
       AND assembly_status = 'pending'
     RETURNING * INTO candidate_row;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'global queue generation changed inside terminal transition'
            USING ERRCODE = '40001';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'committed', TRUE,
        'generation',
        pg_catalog.to_jsonb(candidate_row) - 'preliminary_gate_proof'
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_claim_scoring_queue_failure_projection(
    target_worker_ref TEXT,
    target_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
BEGIN
    IF NULLIF(pg_catalog.btrim(target_worker_ref), '') IS NULL THEN
        RAISE EXCEPTION 'global queue failure projection worker is required'
            USING ERRCODE = '22023';
    END IF;
    IF target_lease_seconds < 30 OR target_lease_seconds > 3600 THEN
        RAISE EXCEPTION 'global queue failure projection lease must be within 30-3600 seconds'
            USING ERRCODE = '22023';
    END IF;

    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE assembly_status = 'failed'
       AND (
           failure_projection_status = 'pending'
           OR (
               failure_projection_status = 'projecting'
               AND failure_projection_lease_expires_at <= NOW()
           )
       )
     ORDER BY updated_at, queue_generation_id
     FOR UPDATE SKIP LOCKED
     LIMIT 1;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'not_found');
    END IF;

    UPDATE public.research_lab_scoring_job_candidate
       SET failure_projection_status = 'projecting',
           failure_projection_claimed_by = target_worker_ref,
           failure_projection_lease_expires_at = NOW()
               + pg_catalog.make_interval(secs => target_lease_seconds),
           failure_projection_attempt_count =
               failure_projection_attempt_count + 1,
           updated_at = NOW()
     WHERE queue_generation_id = candidate_row.queue_generation_id
     RETURNING * INTO candidate_row;

    RETURN pg_catalog.jsonb_build_object(
        'decision', 'claimed',
        'generation',
        pg_catalog.to_jsonb(candidate_row) - 'preliminary_gate_proof'
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_complete_scoring_queue_failure_projection(
    target_queue_generation_id UUID,
    expected_claimed_by TEXT,
    expected_attempt_count INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
BEGIN
    UPDATE public.research_lab_scoring_job_candidate
       SET failure_projection_status = 'projected',
           failure_projection_claimed_by = '',
           failure_projection_lease_expires_at = NULL,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND assembly_status = 'failed'
       AND failure_projection_status = 'projecting'
       AND failure_projection_claimed_by = expected_claimed_by
       AND failure_projection_attempt_count = expected_attempt_count
     RETURNING * INTO candidate_row;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'committed', FALSE,
            'reason', 'claim_changed'
        );
    END IF;
    RETURN pg_catalog.jsonb_build_object('committed', TRUE);
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_fail_scoring_queue_generation(
    UUID,
    TEXT,
    INTEGER,
    TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_claim_scoring_queue_failure_projection(
    TEXT,
    INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_complete_scoring_queue_failure_projection(
    UUID,
    TEXT,
    INTEGER
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_fail_scoring_queue_generation(
    UUID,
    TEXT,
    INTEGER,
    TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_claim_scoring_queue_failure_projection(
    TEXT,
    INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_complete_scoring_queue_failure_projection(
    UUID,
    TEXT,
    INTEGER
) TO service_role;

COMMENT ON FUNCTION public.research_lab_fail_scoring_queue_generation(
    UUID,
    TEXT,
    INTEGER,
    TEXT
) IS
    'Atomically fails an exhausted claimed ICP job, cancels its outstanding siblings, and prevents partial candidate assembly.';

NOTIFY pgrst, 'reload schema';

COMMIT;
