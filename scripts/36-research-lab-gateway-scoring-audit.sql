-- Research Lab gateway-owned private scoring and validator audit bundles.
--
-- Deployment policy:
--   * Apply after scripts 30, 33, 34, and 35.
--   * Gateway qualification workers score private models and candidates.
--   * Main validators verify redacted audit bundles and score-bundle math only.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, private image refs, candidate patch
--     manifests, or proxy credentials may be stored here.

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

CREATE TABLE IF NOT EXISTS public.research_lab_rolling_icp_windows (
    rolling_window_hash  TEXT        PRIMARY KEY CHECK (rolling_window_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    required_days        INTEGER     NOT NULL CHECK (required_days > 0),
    icps_per_day         INTEGER     NOT NULL CHECK (icps_per_day > 0),
    selected_set_count   INTEGER     NOT NULL CHECK (selected_set_count >= 0),
    selected_icp_count   INTEGER     NOT NULL CHECK (selected_icp_count >= 0),
    window_doc           JSONB       NOT NULL CHECK (
                                        jsonb_typeof(window_doc) = 'object'
                                        AND window_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                      ),
    anchored_hash        TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_dispatch_events (
    dispatch_event_id    UUID        PRIMARY KEY,
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    dispatch_type        TEXT        NOT NULL CHECK (
                                        dispatch_type IN (
                                            'candidate_scoring',
                                            'private_baseline_rebenchmark',
                                            'audit_bundle_build'
                                        )),
    dispatch_status      TEXT        NOT NULL CHECK (
                                        dispatch_status IN (
                                            'assigned',
                                            'evaluating',
                                            'scored',
                                            'failed',
                                            'completed',
                                            'rejected',
                                            'tombstoned'
                                        )),
    candidate_id         TEXT        CHECK (candidate_id IS NULL OR candidate_id ~ '^candidate:[0-9a-f]{64}$'),
    run_id               UUID,
    ticket_id            UUID,
    rolling_window_hash  TEXT        REFERENCES public.research_lab_rolling_icp_windows(rolling_window_hash)
                                        ON DELETE RESTRICT,
    score_bundle_id      TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                        ON DELETE RESTRICT,
    benchmark_bundle_id  TEXT,
    worker_ref           TEXT        NOT NULL,
    proxy_ref_hash       TEXT        CHECK (proxy_ref_hash IS NULL OR proxy_ref_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc            JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                        jsonb_typeof(event_doc) = 'object'
                                        AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                      ),
    anchored_hash        TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_private_model_benchmark_bundles (
    benchmark_bundle_id          TEXT        PRIMARY KEY CHECK (benchmark_bundle_id ~ '^private_benchmark:[0-9a-f]{64}$'),
    schema_version               TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    benchmark_date               DATE        NOT NULL,
    private_model_artifact_hash  TEXT        NOT NULL CHECK (private_model_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_hash  TEXT        NOT NULL CHECK (private_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    rolling_window_hash          TEXT        NOT NULL REFERENCES public.research_lab_rolling_icp_windows(rolling_window_hash)
                                                    ON DELETE RESTRICT,
    evaluation_epoch             INTEGER     NOT NULL DEFAULT 0 CHECK (evaluation_epoch >= 0),
    aggregate_score              DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (aggregate_score >= 0),
    scoring_worker_ref           TEXT        NOT NULL,
    proxy_ref_hash               TEXT        CHECK (proxy_ref_hash IS NULL OR proxy_ref_hash ~ '^sha256:[0-9a-f]{64}$'),
    signature_ref                TEXT        NOT NULL,
    score_summary_doc            JSONB       NOT NULL CHECK (
                                                    jsonb_typeof(score_summary_doc) = 'object'
                                                    AND score_summary_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                  ),
    benchmark_bundle_hash        TEXT        NOT NULL UNIQUE CHECK (benchmark_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash                TEXT        NOT NULL UNIQUE CHECK (anchored_hash = benchmark_bundle_hash),
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_private_model_benchmark_unique_day
        UNIQUE (benchmark_date, private_model_manifest_hash, rolling_window_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_private_model_benchmark_events (
    event_id             UUID        PRIMARY KEY,
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    benchmark_bundle_id  TEXT        NOT NULL REFERENCES public.research_lab_private_model_benchmark_bundles(benchmark_bundle_id)
                                        ON DELETE RESTRICT,
    seq                  INTEGER     NOT NULL CHECK (seq >= 0),
    event_type           TEXT        NOT NULL CHECK (
                                        event_type IN (
                                            'queued',
                                            'assigned',
                                            'evaluating',
                                            'completed',
                                            'failed',
                                            'tombstoned'
                                        )),
    benchmark_status     TEXT        NOT NULL CHECK (
                                        benchmark_status IN (
                                            'queued',
                                            'assigned',
                                            'evaluating',
                                            'completed',
                                            'failed',
                                            'tombstoned'
                                        )),
    event_doc            JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                        jsonb_typeof(event_doc) = 'object'
                                        AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                      ),
    anchored_hash        TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_private_model_benchmark_events_seq_key UNIQUE (benchmark_bundle_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_lab_signed_audit_bundles (
    audit_bundle_id      TEXT        PRIMARY KEY CHECK (audit_bundle_id ~ '^research_lab_audit:[0-9a-f]{64}$'),
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    epoch                INTEGER     NOT NULL CHECK (epoch >= 0),
    audit_bundle_hash    TEXT        NOT NULL UNIQUE CHECK (audit_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    signature_ref        TEXT        NOT NULL,
    bundle_doc           JSONB       NOT NULL CHECK (
                                        jsonb_typeof(bundle_doc) = 'object'
                                        AND bundle_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                      ),
    anchored_hash        TEXT        NOT NULL UNIQUE CHECK (anchored_hash = audit_bundle_hash),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_signed_audit_bundle_events (
    event_id         UUID        PRIMARY KEY,
    schema_version   TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    audit_bundle_id  TEXT        NOT NULL REFERENCES public.research_lab_signed_audit_bundles(audit_bundle_id)
                                    ON DELETE RESTRICT,
    seq              INTEGER     NOT NULL CHECK (seq >= 0),
    event_type       TEXT        NOT NULL CHECK (event_type IN ('created', 'verified', 'failed', 'tombstoned')),
    audit_status     TEXT        NOT NULL CHECK (audit_status IN ('created', 'verified', 'failed', 'tombstoned')),
    event_doc        JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(event_doc) = 'object'
                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                  ),
    anchored_hash    TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_signed_audit_bundle_events_seq_key UNIQUE (audit_bundle_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_dispatch_candidate
    ON public.research_lab_scoring_dispatch_events(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_dispatch_window
    ON public.research_lab_scoring_dispatch_events(rolling_window_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_private_model_benchmark_date
    ON public.research_lab_private_model_benchmark_bundles(benchmark_date DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_signed_audit_epoch
    ON public.research_lab_signed_audit_bundles(epoch, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_rolling_icp_windows_mutation
    ON public.research_lab_rolling_icp_windows;
CREATE TRIGGER prevent_research_lab_rolling_icp_windows_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_rolling_icp_windows
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_scoring_dispatch_events_mutation
    ON public.research_lab_scoring_dispatch_events;
CREATE TRIGGER prevent_research_lab_scoring_dispatch_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_dispatch_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_private_model_benchmark_bundles_mutation
    ON public.research_lab_private_model_benchmark_bundles;
CREATE TRIGGER prevent_research_lab_private_model_benchmark_bundles_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_private_model_benchmark_bundles
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_private_model_benchmark_events_mutation
    ON public.research_lab_private_model_benchmark_events;
CREATE TRIGGER prevent_research_lab_private_model_benchmark_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_private_model_benchmark_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_signed_audit_bundles_mutation
    ON public.research_lab_signed_audit_bundles;
CREATE TRIGGER prevent_research_lab_signed_audit_bundles_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_signed_audit_bundles
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_signed_audit_bundle_events_mutation
    ON public.research_lab_signed_audit_bundle_events;
CREATE TRIGGER prevent_research_lab_signed_audit_bundle_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_signed_audit_bundle_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_rolling_icp_windows FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_dispatch_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_private_model_benchmark_bundles FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_private_model_benchmark_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_signed_audit_bundles FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_signed_audit_bundle_events FROM anon, authenticated;

GRANT SELECT, INSERT ON TABLE public.research_lab_rolling_icp_windows TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_dispatch_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_private_model_benchmark_bundles TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_private_model_benchmark_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_signed_audit_bundles TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_signed_audit_bundle_events TO service_role;

ALTER TABLE public.research_lab_rolling_icp_windows ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_scoring_dispatch_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_private_model_benchmark_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_private_model_benchmark_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_signed_audit_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_signed_audit_bundle_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_rolling_icp_windows;
CREATE POLICY service_role_read ON public.research_lab_rolling_icp_windows
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_rolling_icp_windows;
CREATE POLICY service_role_insert ON public.research_lab_rolling_icp_windows
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_scoring_dispatch_events;
CREATE POLICY service_role_read ON public.research_lab_scoring_dispatch_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_scoring_dispatch_events;
CREATE POLICY service_role_insert ON public.research_lab_scoring_dispatch_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_private_model_benchmark_bundles;
CREATE POLICY service_role_read ON public.research_lab_private_model_benchmark_bundles
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_private_model_benchmark_bundles;
CREATE POLICY service_role_insert ON public.research_lab_private_model_benchmark_bundles
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_private_model_benchmark_events;
CREATE POLICY service_role_read ON public.research_lab_private_model_benchmark_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_private_model_benchmark_events;
CREATE POLICY service_role_insert ON public.research_lab_private_model_benchmark_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_signed_audit_bundles;
CREATE POLICY service_role_read ON public.research_lab_signed_audit_bundles
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_signed_audit_bundles;
CREATE POLICY service_role_insert ON public.research_lab_signed_audit_bundles
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_signed_audit_bundle_events;
CREATE POLICY service_role_read ON public.research_lab_signed_audit_bundle_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_signed_audit_bundle_events;
CREATE POLICY service_role_insert ON public.research_lab_signed_audit_bundle_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_private_model_benchmark_current
WITH (security_invoker = true) AS
SELECT
    b.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.benchmark_status AS current_benchmark_status,
    e.created_at AS current_status_at
FROM public.research_lab_private_model_benchmark_bundles b
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_private_model_benchmark_events e
    WHERE e.benchmark_bundle_id = b.benchmark_bundle_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_lab_signed_audit_bundle_current
WITH (security_invoker = true) AS
SELECT
    b.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.audit_status AS current_audit_status,
    e.created_at AS current_status_at
FROM public.research_lab_signed_audit_bundles b
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_signed_audit_bundle_events e
    WHERE e.audit_bundle_id = b.audit_bundle_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_private_model_benchmark_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_signed_audit_bundle_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_private_model_benchmark_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_signed_audit_bundle_current TO service_role;

COMMENT ON TABLE public.research_lab_rolling_icp_windows IS
    'Public hash/ref description of the gateway-owned Research Lab 10-day rolling ICP scoring window.';
COMMENT ON TABLE public.research_lab_scoring_dispatch_events IS
    'Append-only gateway qualification-worker dispatch records for Research Lab candidate scoring and private baseline rebenchmarks.';
COMMENT ON TABLE public.research_lab_private_model_benchmark_bundles IS
    'Gateway-owned daily private model baseline benchmark bundles over the Research Lab rolling ICP window.';
COMMENT ON TABLE public.research_lab_signed_audit_bundles IS
    'Signed redacted Research Lab audit bundles for main validator verification without private artifacts.';

COMMIT;
