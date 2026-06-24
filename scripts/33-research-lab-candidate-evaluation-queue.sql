-- Research Lab Phase 1: gateway-scored candidate evaluation queue.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, 29, 30, 31, and 32.
--   * Gateway hosted workers create candidate artifacts only.
--   * Gateway qualification workers evaluate these candidates against the
--     active private ICP window and submit official score bundles.
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

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_artifacts (
    candidate_id                 TEXT        PRIMARY KEY
                                                CHECK (candidate_id ~ '^candidate:[0-9a-f]{64}$'),
    schema_version               TEXT        NOT NULL DEFAULT '1.0'
                                                CHECK (schema_version = '1.0'),
    run_id                       UUID        NOT NULL,
    ticket_id                    UUID        NOT NULL
                                                REFERENCES public.research_loop_tickets(ticket_id)
                                                ON DELETE RESTRICT,
    receipt_id                   UUID        REFERENCES public.research_loop_receipts(receipt_id)
                                                ON DELETE RESTRICT,
    miner_hotkey                 TEXT        NOT NULL,
    island                       TEXT        NOT NULL,
    parent_artifact_hash         TEXT        NOT NULL CHECK (parent_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    candidate_artifact_hash      TEXT        NOT NULL UNIQUE CHECK (candidate_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_hash  TEXT        NOT NULL CHECK (private_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_doc   JSONB       NOT NULL CHECK (
                                                jsonb_typeof(private_model_manifest_doc) = 'object'
                                                AND private_model_manifest_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                                ),
    candidate_patch_hash         TEXT        NOT NULL UNIQUE CHECK (candidate_patch_hash ~ '^sha256:[0-9a-f]{64}$'),
    candidate_patch_manifest     JSONB       NOT NULL CHECK (
                                                jsonb_typeof(candidate_patch_manifest) = 'object'
                                                AND candidate_patch_manifest::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                                ),
    hypothesis_doc               JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                jsonb_typeof(hypothesis_doc) = 'object'
                                                AND hypothesis_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                                ),
    redacted_public_summary      TEXT        NOT NULL DEFAULT ''
                                                CHECK (redacted_public_summary !~* '(sk-or-|api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|service_role|private_repo|judge_prompt)'),
    anchored_hash                TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (candidate_artifact_hash <> parent_artifact_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_evaluation_events (
    event_id          UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                    CHECK (schema_version = '1.0'),
    candidate_id      TEXT        NOT NULL
                                    REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                    ON DELETE RESTRICT,
    run_id            UUID        NOT NULL,
    ticket_id         UUID        NOT NULL
                                    REFERENCES public.research_loop_tickets(ticket_id)
                                    ON DELETE RESTRICT,
    seq               INTEGER     NOT NULL CHECK (seq >= 0),
    event_type        TEXT        NOT NULL CHECK (
                                    event_type IN (
                                        'queued',
                                        'assigned',
                                        'evaluating',
                                        'scored',
                                        'failed',
                                        'rejected',
                                        'tombstoned'
                                    )),
    candidate_status  TEXT        NOT NULL CHECK (
                                    candidate_status IN (
                                        'queued',
                                        'assigned',
                                        'evaluating',
                                        'scored',
                                        'failed',
                                        'rejected',
                                        'tombstoned'
                                    )),
    evaluator_ref     TEXT,
    reason            TEXT,
    score_bundle_id   TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                    ON DELETE RESTRICT,
    anchored_hash     TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc         JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(event_doc) = 'object'
                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                    ),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_candidate_eval_events_candidate_seq_key UNIQUE (candidate_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_artifacts_run
    ON public.research_lab_candidate_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_artifacts_ticket
    ON public.research_lab_candidate_artifacts(ticket_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_artifacts_miner
    ON public.research_lab_candidate_artifacts(miner_hotkey, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_eval_events_latest
    ON public.research_lab_candidate_evaluation_events(candidate_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_eval_events_status
    ON public.research_lab_candidate_evaluation_events(candidate_status, created_at);

DROP TRIGGER IF EXISTS prevent_research_lab_candidate_artifacts_mutation
    ON public.research_lab_candidate_artifacts;
CREATE TRIGGER prevent_research_lab_candidate_artifacts_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_candidate_artifacts
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_candidate_eval_events_mutation
    ON public.research_lab_candidate_evaluation_events;
CREATE TRIGGER prevent_research_lab_candidate_eval_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_candidate_evaluation_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_candidate_artifacts FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_candidate_evaluation_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_candidate_artifacts TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_candidate_evaluation_events TO service_role;

ALTER TABLE public.research_lab_candidate_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_evaluation_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_candidate_artifacts;
CREATE POLICY service_role_read ON public.research_lab_candidate_artifacts
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_candidate_artifacts;
CREATE POLICY service_role_insert ON public.research_lab_candidate_artifacts
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_candidate_evaluation_events;
CREATE POLICY service_role_read ON public.research_lab_candidate_evaluation_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_candidate_evaluation_events;
CREATE POLICY service_role_insert ON public.research_lab_candidate_evaluation_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_candidate_evaluation_current
WITH (security_invoker = true) AS
SELECT
    c.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.candidate_status AS current_candidate_status,
    e.evaluator_ref AS current_evaluator_ref,
    e.reason AS current_reason,
    e.score_bundle_id AS current_score_bundle_id,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_candidate_artifacts c
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_candidate_evaluation_events e
    WHERE e.candidate_id = c.candidate_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_candidate_evaluation_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_candidate_evaluation_current TO service_role;

COMMENT ON TABLE public.research_lab_candidate_artifacts IS
    'Append-only Research Lab candidate patch artifacts generated by gateway hosted workers. Gateway qualification workers score these against private Supabase ICP sets.';
COMMENT ON TABLE public.research_lab_candidate_evaluation_events IS
    'Append-only gateway qualification evaluation lifecycle events for Research Lab candidates.';
COMMENT ON VIEW public.research_lab_candidate_evaluation_current IS
    'Current gateway qualification evaluation status projection for Research Lab candidates.';

COMMIT;

-- Smoke checks after applying this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_lab_candidate_artifacts',
--       'research_lab_candidate_evaluation_events'
--     )
--   ORDER BY table_name;
--
--   SELECT table_name
--   FROM information_schema.views
--   WHERE table_schema = 'public'
--     AND table_name = 'research_lab_candidate_evaluation_current';
