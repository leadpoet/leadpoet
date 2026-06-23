-- Research Lab Phase 1: hosted auto-research loop execution events.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, 29, 30, 31, 32, and 33.
--   * Gateway hosted workers write iterative auto-research loop events before
--     candidate artifacts are queued for validator scoring.
--   * Validator qualification workers remain the only official scoring path.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, or raw provider secrets may be stored here.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_auto_research_loop_events (
    event_id          UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                    CHECK (schema_version = '1.0'),
    run_id            UUID        NOT NULL,
    ticket_id         UUID        NOT NULL
                                    REFERENCES public.research_loop_tickets(ticket_id)
                                    ON DELETE RESTRICT,
    receipt_id        UUID        REFERENCES public.research_loop_receipts(receipt_id)
                                    ON DELETE RESTRICT,
    seq               INTEGER     NOT NULL CHECK (seq >= 0),
    event_type        TEXT        NOT NULL CHECK (
                                    event_type IN (
                                        'loop_started',
                                        'hypothesis_drafted',
                                        'patch_drafted',
                                        'patch_validation_passed',
                                        'patch_validation_failed',
                                        'dev_check_passed',
                                        'dev_check_failed',
                                        'reflection_recorded',
                                        'candidate_selected',
                                        'loop_completed',
                                        'loop_failed'
                                    )),
    loop_status       TEXT        NOT NULL CHECK (
                                    loop_status IN (
                                        'running',
                                        'completed',
                                        'failed'
                                    )),
    node_id           TEXT,
    worker_ref        TEXT        NOT NULL,
    elapsed_seconds   NUMERIC     NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    candidate_artifact_hash TEXT  CHECK (
                                    candidate_artifact_hash IS NULL
                                    OR candidate_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
                                    ),
    candidate_patch_hash    TEXT  CHECK (
                                    candidate_patch_hash IS NULL
                                    OR candidate_patch_hash ~ '^sha256:[0-9a-f]{64}$'
                                    ),
    provider_usage    JSONB       NOT NULL DEFAULT '[]'::JSONB CHECK (
                                    jsonb_typeof(provider_usage) = 'array'
                                    AND provider_usage::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                    ),
    cost_ledger       JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(cost_ledger) = 'object'
                                    AND cost_ledger::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                    ),
    event_doc         JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(event_doc) = 'object'
                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                    ),
    anchored_hash     TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_auto_loop_events_run_seq_key UNIQUE (run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_auto_loop_events_run
    ON public.research_lab_auto_research_loop_events(run_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_auto_loop_events_ticket
    ON public.research_lab_auto_research_loop_events(ticket_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_auto_loop_events_type
    ON public.research_lab_auto_research_loop_events(event_type, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_auto_loop_events_mutation
    ON public.research_lab_auto_research_loop_events;
CREATE TRIGGER prevent_research_lab_auto_loop_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_auto_research_loop_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_auto_research_loop_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_auto_research_loop_events TO service_role;

ALTER TABLE public.research_lab_auto_research_loop_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_auto_research_loop_events;
CREATE POLICY service_role_read ON public.research_lab_auto_research_loop_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_auto_research_loop_events;
CREATE POLICY service_role_insert ON public.research_lab_auto_research_loop_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_auto_research_loop_current
WITH (security_invoker = true) AS
SELECT
    e.run_id,
    e.ticket_id,
    e.receipt_id,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.loop_status AS current_loop_status,
    e.worker_ref AS current_worker_ref,
    e.elapsed_seconds AS current_elapsed_seconds,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_auto_research_loop_events e
JOIN LATERAL (
    SELECT latest.event_id
    FROM public.research_lab_auto_research_loop_events latest
    WHERE latest.run_id = e.run_id
    ORDER BY latest.seq DESC, latest.created_at DESC
    LIMIT 1
) latest ON latest.event_id = e.event_id;

REVOKE ALL ON TABLE public.research_lab_auto_research_loop_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_auto_research_loop_current TO service_role;

COMMENT ON TABLE public.research_lab_auto_research_loop_events IS
    'Append-only hosted Research Lab auto-research loop events. Candidate artifacts are queued only after loop completion.';
COMMENT ON VIEW public.research_lab_auto_research_loop_current IS
    'Current hosted auto-research loop status projection.';

COMMIT;

-- Smoke checks after applying this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name = 'research_lab_auto_research_loop_events';
--
--   SELECT table_name
--   FROM information_schema.views
--   WHERE table_schema = 'public'
--     AND table_name = 'research_lab_auto_research_loop_current';
