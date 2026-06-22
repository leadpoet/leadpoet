-- Research Lab Phase 1: official private-model evaluation score bundles.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, and 29.
--   * This script creates append-only Research Lab evaluation tables only.
--   * It does not activate paid loops, reimbursements, crowning, fulfillment
--     writes, or weight mutation.
--   * No anon/authenticated grants are created.
--   * Score bundles store hashes, refs, score breakdowns, and redacted public
--     verifier inputs only. Never insert private model code, hidden benchmark
--     plaintext, judge prompts, or raw provider secrets.

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

CREATE TABLE IF NOT EXISTS public.research_evaluation_score_bundles (
    score_bundle_id            TEXT        PRIMARY KEY,
    schema_version             TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    run_id                     UUID        NOT NULL,
    ticket_id                  UUID        REFERENCES public.research_loop_tickets(ticket_id)
                                            ON DELETE RESTRICT,
    receipt_id                 UUID        REFERENCES public.research_loop_receipts(receipt_id)
                                            ON DELETE SET NULL,
    miner_hotkey               TEXT        NOT NULL,
    island                     TEXT        NOT NULL,
    evaluation_epoch           INTEGER     NOT NULL CHECK (evaluation_epoch >= 0),
    bundle_status              TEXT        NOT NULL CHECK (
                                            bundle_status IN (
                                                'scored',
                                                'failed',
                                                'rejected',
                                                'tombstoned'
                                            )),
    parent_artifact_hash       TEXT        NOT NULL CHECK (parent_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    candidate_artifact_hash    TEXT        NOT NULL CHECK (candidate_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_hash TEXT       NOT NULL CHECK (private_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    candidate_patch_hash       TEXT        NOT NULL CHECK (candidate_patch_hash ~ '^sha256:[0-9a-f]{64}$'),
    icp_set_hash               TEXT        NOT NULL CHECK (icp_set_hash ~ '^sha256:[0-9a-f]{64}$'),
    scoring_version            TEXT        NOT NULL,
    evaluator_version          TEXT        NOT NULL,
    score_bundle_hash          TEXT        NOT NULL UNIQUE CHECK (score_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash              TEXT        NOT NULL UNIQUE CHECK (anchored_hash = score_bundle_hash),
    signature_ref              TEXT        NOT NULL,
    score_bundle_doc           JSONB       NOT NULL CHECK (
                                            jsonb_typeof(score_bundle_doc) = 'object'
                                            AND score_bundle_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                            ),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (candidate_artifact_hash <> parent_artifact_hash),
    CHECK (signature_ref !~* '(sk-or-|api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|service_role)')
);

CREATE TABLE IF NOT EXISTS public.research_evaluation_score_bundle_events (
    event_id          UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                    CHECK (schema_version = '1.0'),
    score_bundle_id   TEXT        NOT NULL REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                    ON DELETE RESTRICT,
    seq               INTEGER     NOT NULL CHECK (seq >= 0),
    event_type        TEXT        NOT NULL CHECK (
                                    event_type IN (
                                        'scored',
                                        'failed',
                                        'rejected',
                                        'verified',
                                        'tombstoned'
                                    )),
    event_status      TEXT        NOT NULL CHECK (
                                    event_status IN (
                                        'scored',
                                        'failed',
                                        'rejected',
                                        'verified',
                                        'tombstoned'
                                    )),
    reason            TEXT,
    anchored_hash     TEXT        NOT NULL UNIQUE,
    event_doc         JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(event_doc) = 'object'
                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt)'
                                    ),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_evaluation_score_bundle_events_bundle_seq_key UNIQUE (score_bundle_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_eval_score_bundles_epoch
    ON public.research_evaluation_score_bundles(evaluation_epoch, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_eval_score_bundles_run
    ON public.research_evaluation_score_bundles(run_id);
CREATE INDEX IF NOT EXISTS idx_research_eval_score_bundles_miner
    ON public.research_evaluation_score_bundles(miner_hotkey, evaluation_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_research_eval_score_bundles_doc_gin
    ON public.research_evaluation_score_bundles USING GIN (score_bundle_doc jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_research_eval_score_events_latest
    ON public.research_evaluation_score_bundle_events(score_bundle_id, seq DESC);

DROP TRIGGER IF EXISTS prevent_research_eval_score_bundles_mutation
    ON public.research_evaluation_score_bundles;
CREATE TRIGGER prevent_research_eval_score_bundles_mutation
    BEFORE UPDATE OR DELETE ON public.research_evaluation_score_bundles
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_eval_score_bundle_events_mutation
    ON public.research_evaluation_score_bundle_events;
CREATE TRIGGER prevent_research_eval_score_bundle_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_evaluation_score_bundle_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_evaluation_score_bundles FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_evaluation_score_bundle_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_evaluation_score_bundles TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_evaluation_score_bundle_events TO service_role;

ALTER TABLE public.research_evaluation_score_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_evaluation_score_bundle_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_evaluation_score_bundles;
CREATE POLICY service_role_read ON public.research_evaluation_score_bundles
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_evaluation_score_bundles;
CREATE POLICY service_role_insert ON public.research_evaluation_score_bundles
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_evaluation_score_bundle_events;
CREATE POLICY service_role_read ON public.research_evaluation_score_bundle_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_evaluation_score_bundle_events;
CREATE POLICY service_role_insert ON public.research_evaluation_score_bundle_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_evaluation_score_bundle_current
WITH (security_invoker = true) AS
SELECT
    b.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.event_status AS current_event_status,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_evaluation_score_bundles b
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_evaluation_score_bundle_events e
    WHERE e.score_bundle_id = b.score_bundle_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_evaluation_score_bundle_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_evaluation_score_bundle_current TO service_role;

COMMENT ON TABLE public.research_evaluation_score_bundles IS
    'Append-only official Research Lab private-model evaluation score bundles. Validators verify these before using any Research Lab weight input.';
COMMENT ON TABLE public.research_evaluation_score_bundle_events IS
    'Append-only score-bundle lifecycle events. Corrections and tombstones are represented as new events.';
COMMENT ON VIEW public.research_evaluation_score_bundle_current IS
    'Current Research Lab evaluation score-bundle status projection.';
COMMENT ON COLUMN public.research_evaluation_score_bundles.score_bundle_doc IS
    'Verifier-facing score bundle only: hashes, refs, score breakdowns, and redacted public metadata. No raw secrets, private code, hidden benchmark plaintext, or judge prompts.';

COMMIT;

-- Smoke checks after user applies this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_evaluation_score_bundles',
--       'research_evaluation_score_bundle_events'
--     )
--   ORDER BY table_name;
--
--   SELECT table_name
--   FROM information_schema.views
--   WHERE table_schema = 'public'
--     AND table_name = 'research_evaluation_score_bundle_current';
