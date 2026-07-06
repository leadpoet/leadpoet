-- Global (candidate, icp) scoring queue.
--
-- Makes the unit of scoring work a single (candidate, icp) job instead of a
-- whole candidate. Public ICP jobs are enqueued first for every candidate;
-- when a candidate's public set finishes and its public score meets the
-- baseline, its private-holdout ICP jobs are enqueued at the FRONT of the
-- queue (higher priority); a candidate that misses the baseline enqueues no
-- private jobs. Any scoring worker claims the next highest-priority queued
-- job, so the fixed container pool always pulls the front of the queue.
--
-- Unlike the append-only research_lab_*_events tables, a job queue is
-- inherently mutable status (queued -> claimed -> done), so these tables take
-- UPDATE for service_role and are guarded by RLS + secret-shape CHECKs rather
-- than an append-only trigger. Distributed claiming and the exactly-once gate
-- and assembly transitions are done with compare-and-set UPDATEs (update where
-- status = expected) at the application layer.

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_job_queue (
    job_id            UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    candidate_id      TEXT        NOT NULL,
    window_hash       TEXT        NOT NULL,
    icp_ref           TEXT        NOT NULL,
    item_index        INTEGER     NOT NULL CHECK (item_index >= 0),
    phase             TEXT        NOT NULL CHECK (phase IN ('public', 'private')),
    -- 0 = private (front of queue), 1 = public. Lower runs first.
    priority          INTEGER     NOT NULL CHECK (priority IN (0, 1)),
    -- Monotonic FIFO tiebreak within a priority (enqueue order).
    seq               BIGINT      NOT NULL,
    -- held: a private job waiting on its candidate's gate; flipped to queued
    -- when the public score meets the baseline, or failed if it misses.
    status            TEXT        NOT NULL DEFAULT 'queued'
                                    CHECK (status IN ('held', 'queued', 'claimed', 'done', 'failed')),
    claimed_by        TEXT        NOT NULL DEFAULT '',
    lease_expires_at  TIMESTAMPTZ,
    result_doc        JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                     jsonb_typeof(result_doc) = 'object'
                                     AND result_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                     ),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- One job per (candidate, phase, icp): idempotent enqueue, safe to retry.
    UNIQUE (candidate_id, phase, icp_ref)
);

-- Per-candidate coordination: the gate decision and final assembly each happen
-- exactly once, claimed via compare-and-set on these status columns.
CREATE TABLE IF NOT EXISTS public.research_lab_scoring_job_candidate (
    candidate_id      TEXT        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    window_hash       TEXT        NOT NULL,
    public_total      INTEGER     NOT NULL CHECK (public_total >= 0),
    private_total     INTEGER     NOT NULL DEFAULT 0 CHECK (private_total >= 0),
    baseline_public_score  DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- pending -> deciding -> passed | rejected
    gate_status       TEXT        NOT NULL DEFAULT 'pending'
                                    CHECK (gate_status IN ('pending', 'deciding', 'passed', 'rejected')),
    -- pending -> assembling -> assembled
    assembly_status   TEXT        NOT NULL DEFAULT 'pending'
                                    CHECK (assembly_status IN ('pending', 'assembling', 'assembled')),
    enqueued_by       TEXT        NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Claim query: next queued job by (priority, seq). Partial index keeps it hot
-- as most rows transition to done.
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_claim
    ON public.research_lab_scoring_job_queue (priority, seq)
    WHERE status = 'queued';

-- Per-candidate progress lookups (gate readiness, assembly readiness).
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_candidate_phase
    ON public.research_lab_scoring_job_queue (candidate_id, phase, status);

-- Stale-lease recovery scan.
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_lease
    ON public.research_lab_scoring_job_queue (lease_expires_at)
    WHERE status = 'claimed';

REVOKE ALL ON TABLE public.research_lab_scoring_job_queue FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_job_candidate FROM anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_scoring_job_queue TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_scoring_job_candidate TO service_role;

ALTER TABLE public.research_lab_scoring_job_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_scoring_job_candidate ENABLE ROW LEVEL SECURITY;

CREATE POLICY service_role_all ON public.research_lab_scoring_job_queue
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all ON public.research_lab_scoring_job_candidate
    FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMENT ON TABLE public.research_lab_scoring_job_queue IS
    'Global (candidate, icp) scoring jobs. priority 0=private(front), 1=public; claimed via compare-and-set on status.';
COMMENT ON TABLE public.research_lab_scoring_job_candidate IS
    'Per-candidate coordination for the global ICP queue: exactly-once gate decision and final assembly.';
