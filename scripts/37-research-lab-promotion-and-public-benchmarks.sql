-- Research Lab Phase 1: gateway promotion, private model lineage, and public benchmark reports.
--
-- Deployment policy:
--   * Apply after scripts 30, 33, 34, 35, and 36.
--   * Gateway qualification workers remain the only runtime allowed to execute
--     private model artifacts.
--   * Promotion events decide whether scored candidates become active private
--     model versions and champion-reward inputs.
--   * Public benchmark reports expose sanitized buckets only.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, exact intent signals, judge prompts, candidate patch
--     manifests, private image refs, private model manifest docs, or proxy
--     credentials may be stored here.

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

CREATE TABLE IF NOT EXISTS public.research_lab_private_model_versions (
    private_model_version_id      TEXT        PRIMARY KEY
                                                CHECK (private_model_version_id ~ '^private_model_version:sha256:[0-9a-f]{64}$'),
    schema_version                TEXT        NOT NULL DEFAULT '1.0'
                                                CHECK (schema_version = '1.0'),
    model_artifact_hash           TEXT        NOT NULL UNIQUE CHECK (model_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_hash   TEXT        NOT NULL CHECK (private_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_uri    TEXT        NOT NULL CHECK (
                                                private_model_manifest_uri ~ '^s3://'
                                                AND private_model_manifest_uri !~* '(sk-or-|api[_-]?key|raw[_-]?secret|service_role|judge_prompt|candidate_patch_manifest|\\.dkr\\.ecr\\.|image_digest|://[^/]+:[^/@]+@)'
                                              ),
    git_commit_sha                TEXT        NOT NULL CHECK (git_commit_sha ~ '^[0-9a-f]{8,64}$'),
    config_hash                   TEXT        NOT NULL CHECK (config_hash ~ '^sha256:[0-9a-f]{64}$'),
    component_registry_version    TEXT        NOT NULL,
    scoring_adapter_version       TEXT        NOT NULL,
    source_candidate_id           TEXT        REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                                ON DELETE RESTRICT,
    source_score_bundle_id        TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                                ON DELETE RESTRICT,
    source_benchmark_bundle_id    TEXT        REFERENCES public.research_lab_private_model_benchmark_bundles(benchmark_bundle_id)
                                                ON DELETE RESTRICT,
    signature_ref                 TEXT        NOT NULL,
    build_id                      TEXT,
    redacted_version_doc          JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                jsonb_typeof(redacted_version_doc) = 'object'
                                                AND redacted_version_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                              ),
    version_hash                  TEXT        NOT NULL UNIQUE CHECK (version_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash                 TEXT        NOT NULL UNIQUE CHECK (anchored_hash = version_hash),
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_private_model_version_events (
    event_id                  UUID        PRIMARY KEY,
    schema_version            TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    private_model_version_id  TEXT        NOT NULL REFERENCES public.research_lab_private_model_versions(private_model_version_id)
                                            ON DELETE RESTRICT,
    seq                       INTEGER     NOT NULL CHECK (seq >= 0),
    event_type                TEXT        NOT NULL CHECK (
                                            event_type IN (
                                                'bootstrap',
                                                'active',
                                                'superseded',
                                                'failed',
                                                'tombstoned'
                                            )),
    version_status            TEXT        NOT NULL CHECK (
                                            version_status IN (
                                                'bootstrap',
                                                'active',
                                                'superseded',
                                                'failed',
                                                'tombstoned'
                                            )),
    reason                    TEXT,
    event_doc                 JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(event_doc) = 'object'
                                            AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                          ),
    anchored_hash             TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_private_model_version_events_seq_key UNIQUE (private_model_version_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_promotion_events (
    promotion_event_id            UUID        PRIMARY KEY,
    schema_version                TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    candidate_id                  TEXT        NOT NULL REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                                ON DELETE RESTRICT,
    derived_candidate_id          TEXT        REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                                ON DELETE RESTRICT,
    source_score_bundle_id        TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                                ON DELETE RESTRICT,
    derived_score_bundle_id       TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                                ON DELETE RESTRICT,
    private_model_version_id      TEXT        REFERENCES public.research_lab_private_model_versions(private_model_version_id)
                                                ON DELETE RESTRICT,
    event_type                    TEXT        NOT NULL CHECK (
                                                event_type IN (
                                                    'promotion_checked',
                                                    'below_threshold',
                                                    'stale_parent_detected',
                                                    'rebase_queued',
                                                    'rebase_scored',
                                                    'promotion_failed',
                                                    'promotion_passed',
                                                    'active_version_created',
                                                    'champion_reward_pending_uid',
                                                    'champion_reward_created',
                                                    'tombstoned'
                                                )),
    promotion_status              TEXT        NOT NULL CHECK (
                                                promotion_status IN (
                                                    'checked',
                                                    'rejected',
                                                    'rebase_required',
                                                    'rebenchmarking',
                                                    'failed',
                                                    'passed',
                                                    'merged',
                                                    'reward_pending_uid',
                                                    'reward_created',
                                                    'tombstoned'
                                                )),
    active_parent_artifact_hash   TEXT        CHECK (
                                                active_parent_artifact_hash IS NULL
                                                OR active_parent_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
                                              ),
    candidate_parent_artifact_hash TEXT       CHECK (
                                                candidate_parent_artifact_hash IS NULL
                                                OR candidate_parent_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
                                              ),
    rolling_window_hash           TEXT        REFERENCES public.research_lab_rolling_icp_windows(rolling_window_hash)
                                                ON DELETE RESTRICT,
    improvement_points            NUMERIC(14, 6) NOT NULL DEFAULT 0,
    threshold_points              NUMERIC(14, 6) NOT NULL DEFAULT 1.0 CHECK (threshold_points >= 0),
    worker_ref                    TEXT,
    event_doc                     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                jsonb_typeof(event_doc) = 'object'
                                                AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                              ),
    anchored_hash                 TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_private_repo_commit_events (
    commit_event_id           UUID        PRIMARY KEY,
    schema_version            TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    candidate_id              TEXT        REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                            ON DELETE RESTRICT,
    score_bundle_id           TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                            ON DELETE RESTRICT,
    private_model_version_id  TEXT        REFERENCES public.research_lab_private_model_versions(private_model_version_id)
                                            ON DELETE RESTRICT,
    commit_status             TEXT        NOT NULL CHECK (
                                            commit_status IN (
                                                'started',
                                                'patch_applied',
                                                'tests_passed',
                                                'build_passed',
                                                'committed',
                                                'pushed',
                                                'failed',
                                                'tombstoned'
                                            )),
    git_commit_sha            TEXT        CHECK (git_commit_sha IS NULL OR git_commit_sha ~ '^[0-9a-f]{8,64}$'),
    branch_name               TEXT        NOT NULL DEFAULT 'main'
                                            CHECK (branch_name !~* '(sk-or-|api[_-]?key|raw[_-]?secret|service_role|://[^/]+:[^/@]+@)'),
    private_repo_ref_hash     TEXT        CHECK (private_repo_ref_hash IS NULL OR private_repo_ref_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc                 JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(event_doc) = 'object'
                                            AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                          ),
    anchored_hash             TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_public_benchmark_reports (
    report_id                     TEXT        PRIMARY KEY CHECK (report_id ~ '^public_benchmark:sha256:[0-9a-f]{64}$'),
    schema_version                TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    benchmark_date                DATE        NOT NULL,
    benchmark_bundle_id           TEXT        NOT NULL REFERENCES public.research_lab_private_model_benchmark_bundles(benchmark_bundle_id)
                                                ON DELETE RESTRICT,
    private_model_artifact_hash   TEXT        NOT NULL CHECK (private_model_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    private_model_manifest_hash   TEXT        NOT NULL CHECK (private_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    rolling_window_hash           TEXT        NOT NULL REFERENCES public.research_lab_rolling_icp_windows(rolling_window_hash)
                                                ON DELETE RESTRICT,
    aggregate_score               DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (aggregate_score >= 0),
    report_doc                    JSONB       NOT NULL CHECK (
                                                jsonb_typeof(report_doc) = 'object'
                                                AND report_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@|https?://)'
                                              ),
    report_hash                   TEXT        NOT NULL UNIQUE CHECK (report_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash                 TEXT        NOT NULL UNIQUE CHECK (anchored_hash = report_hash),
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_public_benchmark_reports_unique_day
        UNIQUE (benchmark_date, private_model_manifest_hash, rolling_window_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_public_benchmark_report_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT      NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    report_id     TEXT       NOT NULL REFERENCES public.research_lab_public_benchmark_reports(report_id)
                                ON DELETE RESTRICT,
    seq           INTEGER    NOT NULL CHECK (seq >= 0),
    event_type    TEXT       NOT NULL CHECK (event_type IN ('created', 'published', 'failed', 'tombstoned')),
    report_status TEXT       NOT NULL CHECK (report_status IN ('created', 'published', 'failed', 'tombstoned')),
    event_doc     JSONB      NOT NULL DEFAULT '{}'::JSONB CHECK (
                                jsonb_typeof(event_doc) = 'object'
                                AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@|https?://)'
                              ),
    anchored_hash TEXT       NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_public_benchmark_report_events_seq_key UNIQUE (report_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_private_model_version_status
    ON public.research_lab_private_model_version_events(version_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_promotion_candidate
    ON public.research_lab_candidate_promotion_events(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_promotion_status
    ON public.research_lab_candidate_promotion_events(promotion_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_private_repo_commit_candidate
    ON public.research_lab_private_repo_commit_events(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_benchmark_date
    ON public.research_lab_public_benchmark_reports(benchmark_date DESC, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_private_model_versions_mutation
    ON public.research_lab_private_model_versions;
CREATE TRIGGER prevent_research_lab_private_model_versions_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_private_model_versions
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_private_model_version_events_mutation
    ON public.research_lab_private_model_version_events;
CREATE TRIGGER prevent_research_lab_private_model_version_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_private_model_version_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_candidate_promotion_events_mutation
    ON public.research_lab_candidate_promotion_events;
CREATE TRIGGER prevent_research_lab_candidate_promotion_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_candidate_promotion_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_private_repo_commit_events_mutation
    ON public.research_lab_private_repo_commit_events;
CREATE TRIGGER prevent_research_lab_private_repo_commit_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_private_repo_commit_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_public_benchmark_reports_mutation
    ON public.research_lab_public_benchmark_reports;
CREATE TRIGGER prevent_research_lab_public_benchmark_reports_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_public_benchmark_reports
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_public_benchmark_report_events_mutation
    ON public.research_lab_public_benchmark_report_events;
CREATE TRIGGER prevent_research_lab_public_benchmark_report_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_public_benchmark_report_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_private_model_versions FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_private_model_version_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_candidate_promotion_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_private_repo_commit_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_public_benchmark_reports FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_public_benchmark_report_events FROM anon, authenticated;

GRANT SELECT, INSERT ON TABLE public.research_lab_private_model_versions TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_private_model_version_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_candidate_promotion_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_private_repo_commit_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_public_benchmark_reports TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_public_benchmark_report_events TO service_role;

ALTER TABLE public.research_lab_private_model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_private_model_version_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_promotion_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_private_repo_commit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_public_benchmark_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_public_benchmark_report_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_private_model_versions;
CREATE POLICY service_role_read ON public.research_lab_private_model_versions
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_private_model_versions;
CREATE POLICY service_role_insert ON public.research_lab_private_model_versions
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_private_model_version_events;
CREATE POLICY service_role_read ON public.research_lab_private_model_version_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_private_model_version_events;
CREATE POLICY service_role_insert ON public.research_lab_private_model_version_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_candidate_promotion_events;
CREATE POLICY service_role_read ON public.research_lab_candidate_promotion_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_candidate_promotion_events;
CREATE POLICY service_role_insert ON public.research_lab_candidate_promotion_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_private_repo_commit_events;
CREATE POLICY service_role_read ON public.research_lab_private_repo_commit_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_private_repo_commit_events;
CREATE POLICY service_role_insert ON public.research_lab_private_repo_commit_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_public_benchmark_reports;
CREATE POLICY service_role_read ON public.research_lab_public_benchmark_reports
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_public_benchmark_reports;
CREATE POLICY service_role_insert ON public.research_lab_public_benchmark_reports
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_public_benchmark_report_events;
CREATE POLICY service_role_read ON public.research_lab_public_benchmark_report_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_public_benchmark_report_events;
CREATE POLICY service_role_insert ON public.research_lab_public_benchmark_report_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_private_model_version_current
WITH (security_invoker = true) AS
SELECT
    v.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.version_status AS current_version_status,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_private_model_versions v
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_private_model_version_events e
    WHERE e.private_model_version_id = v.private_model_version_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_lab_public_benchmark_report_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.report_status AS current_report_status,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_public_benchmark_reports r
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_public_benchmark_report_events e
    WHERE e.report_id = r.report_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_private_model_version_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_public_benchmark_report_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_private_model_version_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_public_benchmark_report_current TO service_role;

COMMENT ON TABLE public.research_lab_private_model_versions IS
    'Append-only private model lineage refs. Private image refs and manifest docs are loaded by gateway from S3, not exposed here.';
COMMENT ON TABLE public.research_lab_candidate_promotion_events IS
    'Append-only gateway promotion, stale-parent rebenchmark, merge, and reward-gating events for Research Lab candidates.';
COMMENT ON TABLE public.research_lab_private_repo_commit_events IS
    'Append-only redacted private repo auto-commit lifecycle events.';
COMMENT ON TABLE public.research_lab_public_benchmark_reports IS
    'Sanitized public benchmark reports for miners; contains buckets and error counts only.';

COMMIT;
