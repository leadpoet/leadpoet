-- Research Lab: append-only redacted provider cost accounting events.
--
-- Deployment policy:
--   * Apply after scripts 31 and 68.
--   * Safe to apply repeatedly.
--   * Stores only redacted request/cost metadata. Raw prompts, provider
--     responses, URLs with sensitive params, and API keys must never be stored.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_provider_cost_events (
    event_id            UUID        PRIMARY KEY,
    schema_version      TEXT        NOT NULL DEFAULT '1.0'
                                    CHECK (schema_version = '1.0'),
    run_scope           TEXT        NOT NULL,
    run_type            TEXT        NOT NULL DEFAULT 'unknown',
    candidate_id        TEXT,
    benchmark_date      DATE,
    rolling_window_hash TEXT,
    icp_ref             TEXT        NOT NULL,
    icp_hash            TEXT,
    runner_role         TEXT        NOT NULL DEFAULT 'unknown',
    provider            TEXT        NOT NULL CHECK (provider IN ('exa', 'or', 'sd', 'unknown')),
    endpoint            TEXT        NOT NULL,
    model               TEXT        NOT NULL DEFAULT '',
    request_fingerprint TEXT        NOT NULL CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    status_code         INTEGER     NOT NULL DEFAULT 0,
    billable            BOOLEAN     NOT NULL DEFAULT FALSE,
    cost_usd            NUMERIC(12,8) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    cost_source         TEXT        NOT NULL DEFAULT 'not_billable',
    credits             INTEGER     NOT NULL DEFAULT 0 CHECK (credits >= 0),
    prompt_tokens       INTEGER     NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens   INTEGER     NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    cap_usd             NUMERIC(12,8) NOT NULL DEFAULT 0 CHECK (cap_usd >= 0),
    spent_before_usd    NUMERIC(12,8) NOT NULL DEFAULT 0 CHECK (spent_before_usd >= 0),
    spent_after_usd     NUMERIC(12,8) NOT NULL DEFAULT 0 CHECK (spent_after_usd >= 0),
    cap_state           TEXT        NOT NULL DEFAULT 'under_cap'
                                    CHECK (cap_state IN (
                                        'under_cap',
                                        'exceeded_after_success',
                                        'blocked_before_call',
                                        'cost_tracking_failed'
                                    )),
    event_doc           JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                    jsonb_typeof(event_doc) = 'object'
                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_prompt|provider_output|request_body|response_body)'
                                    ),
    anchored_hash       TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- CREATE TABLE IF NOT EXISTS does not repair an existing partially-applied
-- table. Keep the runtime insert contract safe across deploys where code using
-- run_scope landed before this final schema shape.
ALTER TABLE public.research_lab_provider_cost_events
    ADD COLUMN IF NOT EXISTS run_scope TEXT NOT NULL DEFAULT 'unscoped';

CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_events_scope
    ON public.research_lab_provider_cost_events(run_scope, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_events_candidate
    ON public.research_lab_provider_cost_events(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_events_benchmark_date
    ON public.research_lab_provider_cost_events(benchmark_date, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_events_icp
    ON public.research_lab_provider_cost_events(icp_ref, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_provider_cost_events_mutation
    ON public.research_lab_provider_cost_events;
CREATE TRIGGER prevent_research_lab_provider_cost_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_provider_cost_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_provider_cost_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_provider_cost_events TO service_role;

ALTER TABLE public.research_lab_provider_cost_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_provider_cost_events;
CREATE POLICY service_role_read ON public.research_lab_provider_cost_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_provider_cost_events;
CREATE POLICY service_role_insert ON public.research_lab_provider_cost_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE public.research_lab_provider_cost_events IS
    'Append-only redacted per-ICP provider cost events for Research Lab private-model scoring.';

COMMIT;

-- Verification:
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name = 'research_lab_provider_cost_events';
