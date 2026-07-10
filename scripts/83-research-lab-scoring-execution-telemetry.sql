-- Unified append-only execution telemetry for Research Lab scoring.
--
-- Apply before enabling RESEARCH_LAB_SCORING_TELEMETRY_V2. This migration is
-- storage-only: all new writer columns are nullable and legacy writers remain
-- valid. It intentionally stores hashes/counts/categories only, never hidden
-- ICP bodies, raw companies, prompts, provider bodies, URLs, or credentials.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_runs (
    scoring_run_id                  UUID        PRIMARY KEY,
    schema_version                  TEXT        NOT NULL DEFAULT '2.0' CHECK (schema_version = '2.0'),
    scoring_id                      TEXT        NOT NULL CHECK (scoring_id ~ '^scoring:sha256:[0-9a-f]{64}$'),
    run_type                        TEXT        NOT NULL CHECK (run_type IN (
                                                    'private_baseline_rebenchmark',
                                                    'candidate_scoring',
                                                    'promotion_confirmation'
                                                )),
    run_attempt                     INTEGER     NOT NULL CHECK (run_attempt >= 0),
    source_run_id                   UUID,
    ticket_id                       UUID,
    candidate_id                    TEXT        CHECK (candidate_id IS NULL OR candidate_id ~ '^candidate:[0-9a-f]{64}$'),
    benchmark_id                    TEXT,
    benchmark_date                  DATE,
    rolling_window_hash             TEXT        CHECK (
                                                    rolling_window_hash IS NULL
                                                    OR rolling_window_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    reference_artifact_hash         TEXT        CHECK (
                                                    reference_artifact_hash IS NULL
                                                    OR reference_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    reference_manifest_hash         TEXT        CHECK (
                                                    reference_manifest_hash IS NULL
                                                    OR reference_manifest_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    candidate_artifact_hash         TEXT        CHECK (
                                                    candidate_artifact_hash IS NULL
                                                    OR candidate_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    candidate_manifest_hash         TEXT        CHECK (
                                                    candidate_manifest_hash IS NULL
                                                    OR candidate_manifest_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    baseline_benchmark_bundle_id    TEXT,
    source_score_bundle_id          TEXT,
    evaluation_epoch                INTEGER     NOT NULL DEFAULT 0 CHECK (evaluation_epoch >= 0),
    expected_icp_count              INTEGER     NOT NULL CHECK (expected_icp_count >= 0),
    scheduler_type                  TEXT        NOT NULL CHECK (scheduler_type IN (
                                                    'serial',
                                                    'fixed_wave',
                                                    'work_conserving',
                                                    'global_icp_queue',
                                                    'confirmation_pair'
                                                )),
    worker_ref                      TEXT        NOT NULL,
    resumed_from_scoring_run_id     UUID,
    anchored_hash                   TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_scoring_runs_attempt_key UNIQUE (scoring_id, run_attempt),
    CONSTRAINT research_lab_scoring_runs_identity_key UNIQUE (scoring_id, scoring_run_id),
    CONSTRAINT research_lab_scoring_runs_resume_fk
        FOREIGN KEY (scoring_id, resumed_from_scoring_run_id)
        REFERENCES public.research_lab_scoring_runs(scoring_id, scoring_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT research_lab_scoring_runs_type_shape CHECK (
        (
            run_type = 'private_baseline_rebenchmark'
            AND benchmark_id IS NOT NULL
            AND benchmark_date IS NOT NULL
            AND rolling_window_hash IS NOT NULL
            AND reference_artifact_hash IS NOT NULL
        )
        OR (
            run_type = 'candidate_scoring'
            AND candidate_id IS NOT NULL
            AND source_run_id IS NOT NULL
        )
        OR (
            run_type = 'promotion_confirmation'
            AND candidate_id IS NOT NULL
            AND source_score_bundle_id IS NOT NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_run_events (
    event_id                UUID        PRIMARY KEY,
    schema_version          TEXT        NOT NULL DEFAULT '2.0' CHECK (schema_version = '2.0'),
    scoring_id              TEXT        NOT NULL,
    scoring_run_id          UUID        NOT NULL,
    event_type              TEXT        NOT NULL CHECK (event_type IN (
                                                'assigned', 'started', 'heartbeat',
                                                'paused', 'resumed', 'completed',
                                                'failed', 'cancelled', 'restarted'
                                            )),
    event_ordinal           BIGINT      NOT NULL DEFAULT 0 CHECK (event_ordinal >= 0),
    occurred_at             TIMESTAMPTZ NOT NULL,
    worker_ref              TEXT        NOT NULL,
    retryable               BOOLEAN,
    failure_category        TEXT        CHECK (
                                                failure_category IS NULL
                                                OR failure_category ~ '^[a-z0-9_:-]{1,120}$'
                                            ),
    failure_fingerprint     TEXT        CHECK (
                                                failure_fingerprint IS NULL
                                                OR failure_fingerprint ~ '^sha256:[0-9a-f]{64}$'
                                            ),
    checkpoint_ref          TEXT        CHECK (
                                                checkpoint_ref IS NULL
                                                OR checkpoint_ref ~ '^scoring_checkpoint:[0-9a-f]{64}$'
                                            ),
    checkpoint_hash         TEXT        CHECK (
                                                checkpoint_hash IS NULL
                                                OR checkpoint_hash ~ '^sha256:[0-9a-f]{64}$'
                                            ),
    telemetry_degraded      BOOLEAN     NOT NULL DEFAULT FALSE,
    benchmark_bundle_id     TEXT,
    score_bundle_id         TEXT,
    promotion_event_id      UUID,
    event_doc               JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                jsonb_typeof(event_doc) = 'object'
                                                AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                            ),
    anchored_hash           TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_scoring_run_events_run_fk
        FOREIGN KEY (scoring_id, scoring_run_id)
        REFERENCES public.research_lab_scoring_runs(scoring_id, scoring_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT research_lab_scoring_run_events_identity_key
        UNIQUE (scoring_run_id, event_type, event_ordinal)
);

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_icp_executions (
    icp_execution_id            UUID        PRIMARY KEY,
    schema_version              TEXT        NOT NULL DEFAULT '2.0' CHECK (schema_version = '2.0'),
    scoring_id                  TEXT        NOT NULL,
    scoring_run_id              UUID        NOT NULL,
    icp_ref                     TEXT        NOT NULL,
    icp_hash                    TEXT        CHECK (icp_hash IS NULL OR icp_hash ~ '^sha256:[0-9a-f]{64}$'),
    icp_ordinal                 INTEGER     NOT NULL CHECK (icp_ordinal >= 0),
    model_role                  TEXT        NOT NULL CHECK (model_role IN ('reference', 'candidate')),
    retry_round                 INTEGER     NOT NULL DEFAULT 0 CHECK (retry_round >= 0),
    attempt_ordinal             INTEGER     NOT NULL DEFAULT 0 CHECK (attempt_ordinal >= 0),
    execution_kind              TEXT        NOT NULL CHECK (execution_kind IN (
                                                    'model_invocation',
                                                    'checkpoint_reuse',
                                                    'gate_skip',
                                                    'latch_skip'
                                                )),
    phase                       TEXT        NOT NULL DEFAULT 'all' CHECK (phase IN ('all', 'public', 'private')),
    worker_ref                  TEXT        NOT NULL,
    source_job_id               UUID,
    reused_from_execution_id    UUID REFERENCES public.research_lab_scoring_icp_executions(icp_execution_id)
                                    ON DELETE RESTRICT,
    anchored_hash               TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_scoring_icp_executions_run_fk
        FOREIGN KEY (scoring_id, scoring_run_id)
        REFERENCES public.research_lab_scoring_runs(scoring_id, scoring_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT research_lab_scoring_icp_executions_attempt_key
        UNIQUE (scoring_run_id, model_role, icp_ref, attempt_ordinal),
    CONSTRAINT research_lab_scoring_icp_executions_identity_key
        UNIQUE (scoring_id, scoring_run_id, icp_execution_id)
);

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_icp_events (
    event_id                    UUID        PRIMARY KEY,
    schema_version              TEXT        NOT NULL DEFAULT '2.0' CHECK (schema_version = '2.0'),
    scoring_id                  TEXT        NOT NULL,
    scoring_run_id              UUID        NOT NULL,
    icp_execution_id            UUID        NOT NULL,
    event_type                  TEXT        NOT NULL CHECK (event_type IN (
                                                    'held', 'queued', 'started', 'heartbeat',
                                                    'sourcing_completed', 'scoring_started',
                                                    'completed', 'failed', 'cancelled', 'skipped'
                                                )),
    event_ordinal               BIGINT      NOT NULL DEFAULT 0 CHECK (event_ordinal >= 0),
    occurred_at                 TIMESTAMPTZ NOT NULL,
    score                       DOUBLE PRECISION,
    sourced_company_count      INTEGER     CHECK (sourced_company_count IS NULL OR sourced_company_count >= 0),
    scored_company_count       INTEGER     CHECK (scored_company_count IS NULL OR scored_company_count >= 0),
    retryable                   BOOLEAN,
    failure_category            TEXT        CHECK (
                                                    failure_category IS NULL
                                                    OR failure_category ~ '^[a-z0-9_:-]{1,120}$'
                                                ),
    failure_fingerprint         TEXT        CHECK (
                                                    failure_fingerprint IS NULL
                                                    OR failure_fingerprint ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    checkpoint_ref              TEXT        CHECK (
                                                    checkpoint_ref IS NULL
                                                    OR checkpoint_ref ~ '^scoring_checkpoint:[0-9a-f]{64}$'
                                                ),
    checkpoint_hash             TEXT        CHECK (
                                                    checkpoint_hash IS NULL
                                                    OR checkpoint_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    result_row_hash             TEXT        CHECK (
                                                    result_row_hash IS NULL
                                                    OR result_row_hash ~ '^sha256:[0-9a-f]{64}$'
                                                ),
    telemetry_degraded          BOOLEAN     NOT NULL DEFAULT FALSE,
    event_doc                   JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                    jsonb_typeof(event_doc) = 'object'
                                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                ),
    anchored_hash               TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_scoring_icp_events_execution_fk
        FOREIGN KEY (icp_execution_id)
        REFERENCES public.research_lab_scoring_icp_executions(icp_execution_id)
        ON DELETE RESTRICT,
    CONSTRAINT research_lab_scoring_icp_events_run_fk
        FOREIGN KEY (scoring_id, scoring_run_id)
        REFERENCES public.research_lab_scoring_runs(scoring_id, scoring_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT research_lab_scoring_icp_events_identity_key
        UNIQUE (icp_execution_id, event_type, event_ordinal)
);

-- Nullable compatibility columns. Existing rows and writers stay valid.
ALTER TABLE public.research_lab_scoring_dispatch_events
    ADD COLUMN IF NOT EXISTS scoring_id TEXT,
    ADD COLUMN IF NOT EXISTS scoring_run_id UUID;

ALTER TABLE public.research_lab_provider_cost_events
    ADD COLUMN IF NOT EXISTS scoring_id TEXT,
    ADD COLUMN IF NOT EXISTS scoring_run_id UUID,
    ADD COLUMN IF NOT EXISTS icp_execution_id UUID;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_dispatch_telemetry_ids_check'
          AND conrelid = 'public.research_lab_scoring_dispatch_events'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_dispatch_events
            ADD CONSTRAINT research_lab_scoring_dispatch_telemetry_ids_check
            CHECK (num_nonnulls(scoring_id, scoring_run_id) IN (0, 2));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_dispatch_run_fk'
          AND conrelid = 'public.research_lab_scoring_dispatch_events'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_dispatch_events
            ADD CONSTRAINT research_lab_scoring_dispatch_run_fk
            FOREIGN KEY (scoring_id, scoring_run_id)
            REFERENCES public.research_lab_scoring_runs(scoring_id, scoring_run_id)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_provider_cost_telemetry_ids_check'
          AND conrelid = 'public.research_lab_provider_cost_events'::regclass
    ) THEN
        ALTER TABLE public.research_lab_provider_cost_events
            ADD CONSTRAINT research_lab_provider_cost_telemetry_ids_check
            CHECK (num_nonnulls(scoring_id, scoring_run_id, icp_execution_id) IN (0, 3));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_provider_cost_execution_fk'
          AND conrelid = 'public.research_lab_provider_cost_events'::regclass
    ) THEN
        ALTER TABLE public.research_lab_provider_cost_events
            ADD CONSTRAINT research_lab_provider_cost_execution_fk
            FOREIGN KEY (icp_execution_id)
            REFERENCES public.research_lab_scoring_icp_executions(icp_execution_id)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_provider_cost_execution_identity_fk'
          AND conrelid = 'public.research_lab_provider_cost_events'::regclass
    ) THEN
        ALTER TABLE public.research_lab_provider_cost_events
            ADD CONSTRAINT research_lab_provider_cost_execution_identity_fk
            FOREIGN KEY (scoring_id, scoring_run_id, icp_execution_id)
            REFERENCES public.research_lab_scoring_icp_executions(
                scoring_id, scoring_run_id, icp_execution_id
            )
            ON DELETE RESTRICT;
    END IF;
END
$$;

-- Global queue generations are distinct from candidate identity. Keep the
-- scoring-run link nullable so disabling telemetry never disables the queue.
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD COLUMN IF NOT EXISTS queue_generation_id UUID DEFAULT gen_random_uuid(),
    ADD COLUMN IF NOT EXISTS scoring_run_id UUID;
ALTER TABLE public.research_lab_scoring_job_queue
    ADD COLUMN IF NOT EXISTS queue_generation_id UUID,
    ADD COLUMN IF NOT EXISTS scoring_run_id UUID,
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.research_lab_scoring_job_candidate
        WHERE queue_generation_id IS NULL
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_scoring_job_queue
        WHERE queue_generation_id IS NULL
    ) THEN
        RAISE EXCEPTION
            'scoring queue must be drained before migration 83 generation scoping';
    END IF;
END
$$;

ALTER TABLE public.research_lab_scoring_job_candidate
    ALTER COLUMN queue_generation_id SET NOT NULL;
ALTER TABLE public.research_lab_scoring_job_queue
    ALTER COLUMN queue_generation_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_queue_attempt_count_check'
          AND conrelid = 'public.research_lab_scoring_job_queue'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_queue
            ADD CONSTRAINT research_lab_scoring_job_queue_attempt_count_check
            CHECK (attempt_count >= 0) NOT VALID;
    END IF;
END
$$;
ALTER TABLE public.research_lab_scoring_job_queue
    VALIDATE CONSTRAINT research_lab_scoring_job_queue_attempt_count_check;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_candidate_pkey'
          AND conrelid = 'public.research_lab_scoring_job_candidate'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_candidate
            DROP CONSTRAINT research_lab_scoring_job_candidate_pkey;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_candidate_generation_pkey'
          AND conrelid = 'public.research_lab_scoring_job_candidate'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_candidate
            ADD CONSTRAINT research_lab_scoring_job_candidate_generation_pkey
            PRIMARY KEY (queue_generation_id);
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_queue_candidate_id_phase_icp_ref_key'
          AND conrelid = 'public.research_lab_scoring_job_queue'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_queue
            DROP CONSTRAINT research_lab_scoring_job_queue_candidate_id_phase_icp_ref_key;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_queue_generation_item_key'
          AND conrelid = 'public.research_lab_scoring_job_queue'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_queue
            ADD CONSTRAINT research_lab_scoring_job_queue_generation_item_key
            UNIQUE (queue_generation_id, phase, icp_ref);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_queue_generation_fk'
          AND conrelid = 'public.research_lab_scoring_job_queue'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_queue
            ADD CONSTRAINT research_lab_scoring_job_queue_generation_fk
            FOREIGN KEY (queue_generation_id)
            REFERENCES public.research_lab_scoring_job_candidate(queue_generation_id)
            ON DELETE RESTRICT;
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS research_lab_scoring_job_candidate_active_generation_key
    ON public.research_lab_scoring_job_candidate(candidate_id, window_hash)
    WHERE assembly_status IN ('pending', 'assembling');

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_candidate_run_fk'
          AND conrelid = 'public.research_lab_scoring_job_candidate'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_candidate
            ADD CONSTRAINT research_lab_scoring_job_candidate_run_fk
            FOREIGN KEY (scoring_run_id)
            REFERENCES public.research_lab_scoring_runs(scoring_run_id)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'research_lab_scoring_job_queue_run_fk'
          AND conrelid = 'public.research_lab_scoring_job_queue'::regclass
    ) THEN
        ALTER TABLE public.research_lab_scoring_job_queue
            ADD CONSTRAINT research_lab_scoring_job_queue_run_fk
            FOREIGN KEY (scoring_run_id)
            REFERENCES public.research_lab_scoring_runs(scoring_run_id)
            ON DELETE RESTRICT;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_type_created
    ON public.research_lab_scoring_runs(run_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_candidate
    ON public.research_lab_scoring_runs(candidate_id, created_at DESC)
    WHERE candidate_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_benchmark
    ON public.research_lab_scoring_runs(benchmark_date DESC, run_attempt DESC)
    WHERE benchmark_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_source
    ON public.research_lab_scoring_runs(source_run_id, created_at DESC)
    WHERE source_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_resumed_from
    ON public.research_lab_scoring_runs(resumed_from_scoring_run_id)
    WHERE resumed_from_scoring_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_run_events_current
    ON public.research_lab_scoring_run_events(scoring_run_id, occurred_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_icp_executions_run
    ON public.research_lab_scoring_icp_executions(scoring_run_id, model_role, icp_ordinal);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_icp_executions_lookup
    ON public.research_lab_scoring_icp_executions(scoring_id, model_role, icp_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_icp_events_current
    ON public.research_lab_scoring_icp_events(icp_execution_id, occurred_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_dispatch_run
    ON public.research_lab_scoring_dispatch_events(scoring_run_id, created_at DESC)
    WHERE scoring_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_scoring_run
    ON public.research_lab_provider_cost_events(scoring_run_id, created_at DESC)
    WHERE scoring_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_execution
    ON public.research_lab_provider_cost_events(icp_execution_id, created_at DESC)
    WHERE icp_execution_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_cost_scoring_icp
    ON public.research_lab_provider_cost_events(scoring_id, icp_ref, created_at DESC)
    WHERE scoring_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_candidate_candidate
    ON public.research_lab_scoring_job_candidate(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_candidate_run
    ON public.research_lab_scoring_job_candidate(scoring_run_id)
    WHERE scoring_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_queue_generation
    ON public.research_lab_scoring_job_queue(queue_generation_id, phase, status, item_index);
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_queue_run
    ON public.research_lab_scoring_job_queue(scoring_run_id)
    WHERE scoring_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS brin_research_lab_scoring_run_events_created
    ON public.research_lab_scoring_run_events USING BRIN(created_at);
CREATE INDEX IF NOT EXISTS brin_research_lab_scoring_icp_events_created
    ON public.research_lab_scoring_icp_events USING BRIN(created_at);

DROP TRIGGER IF EXISTS prevent_research_lab_scoring_runs_mutation
    ON public.research_lab_scoring_runs;
CREATE TRIGGER prevent_research_lab_scoring_runs_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_runs
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();
DROP TRIGGER IF EXISTS prevent_research_lab_scoring_run_events_mutation
    ON public.research_lab_scoring_run_events;
CREATE TRIGGER prevent_research_lab_scoring_run_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_run_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();
DROP TRIGGER IF EXISTS prevent_research_lab_scoring_icp_executions_mutation
    ON public.research_lab_scoring_icp_executions;
CREATE TRIGGER prevent_research_lab_scoring_icp_executions_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_icp_executions
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();
DROP TRIGGER IF EXISTS prevent_research_lab_scoring_icp_events_mutation
    ON public.research_lab_scoring_icp_events;
CREATE TRIGGER prevent_research_lab_scoring_icp_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_icp_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_scoring_runs FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_run_events FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_icp_executions FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_icp_events FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_runs TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_run_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_icp_executions TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_icp_events TO service_role;

ALTER TABLE public.research_lab_scoring_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_scoring_run_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_scoring_icp_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_scoring_icp_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS scoring_telemetry_service_select ON public.research_lab_scoring_runs;
CREATE POLICY scoring_telemetry_service_select ON public.research_lab_scoring_runs
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS scoring_telemetry_service_insert ON public.research_lab_scoring_runs;
CREATE POLICY scoring_telemetry_service_insert ON public.research_lab_scoring_runs
    FOR INSERT TO service_role WITH CHECK (true);
DROP POLICY IF EXISTS scoring_telemetry_service_select ON public.research_lab_scoring_run_events;
CREATE POLICY scoring_telemetry_service_select ON public.research_lab_scoring_run_events
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS scoring_telemetry_service_insert ON public.research_lab_scoring_run_events;
CREATE POLICY scoring_telemetry_service_insert ON public.research_lab_scoring_run_events
    FOR INSERT TO service_role WITH CHECK (true);
DROP POLICY IF EXISTS scoring_telemetry_service_select ON public.research_lab_scoring_icp_executions;
CREATE POLICY scoring_telemetry_service_select ON public.research_lab_scoring_icp_executions
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS scoring_telemetry_service_insert ON public.research_lab_scoring_icp_executions;
CREATE POLICY scoring_telemetry_service_insert ON public.research_lab_scoring_icp_executions
    FOR INSERT TO service_role WITH CHECK (true);
DROP POLICY IF EXISTS scoring_telemetry_service_select ON public.research_lab_scoring_icp_events;
CREATE POLICY scoring_telemetry_service_select ON public.research_lab_scoring_icp_events
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS scoring_telemetry_service_insert ON public.research_lab_scoring_icp_events;
CREATE POLICY scoring_telemetry_service_insert ON public.research_lab_scoring_icp_events
    FOR INSERT TO service_role WITH CHECK (true);

CREATE OR REPLACE VIEW public.research_lab_scoring_run_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    current_event.event_type AS current_run_status,
    current_event.occurred_at AS current_status_at,
    current_event.retryable AS current_retryable,
    current_event.failure_category AS current_failure_category,
    current_event.telemetry_degraded AS current_telemetry_degraded,
    current_event.benchmark_bundle_id,
    current_event.score_bundle_id,
    lifecycle.assigned_at,
    lifecycle.started_at,
    lifecycle.last_heartbeat_at,
    lifecycle.finished_at,
    CASE
        WHEN lifecycle.started_at IS NULL THEN NULL
        WHEN lifecycle.finished_at IS NOT NULL
            THEN GREATEST(0, EXTRACT(EPOCH FROM lifecycle.finished_at - lifecycle.started_at))
        WHEN lifecycle.last_heartbeat_at IS NOT NULL
            THEN GREATEST(0, EXTRACT(EPOCH FROM lifecycle.last_heartbeat_at - lifecycle.started_at))
        ELSE NULL
    END AS observed_runtime_seconds
FROM public.research_lab_scoring_runs r
LEFT JOIN LATERAL (
    SELECT e.*
    FROM public.research_lab_scoring_run_events e
    WHERE e.scoring_run_id = r.scoring_run_id
    ORDER BY e.occurred_at DESC, e.event_id DESC
    LIMIT 1
) current_event ON TRUE
LEFT JOIN LATERAL (
    SELECT
        MIN(e.occurred_at) FILTER (WHERE e.event_type = 'assigned') AS assigned_at,
        MIN(e.occurred_at) FILTER (WHERE e.event_type = 'started') AS started_at,
        MAX(e.occurred_at) FILTER (WHERE e.event_type = 'heartbeat') AS last_heartbeat_at,
        MAX(e.occurred_at) FILTER (WHERE e.event_type IN ('completed', 'failed', 'cancelled', 'restarted')) AS finished_at
    FROM public.research_lab_scoring_run_events e
    WHERE e.scoring_run_id = r.scoring_run_id
) lifecycle ON TRUE;

CREATE OR REPLACE VIEW public.research_lab_scoring_icp_execution_current
WITH (security_invoker = true) AS
SELECT
    x.*,
    current_event.event_type AS current_execution_status,
    current_event.occurred_at AS current_status_at,
    current_event.score,
    current_event.sourced_company_count,
    current_event.scored_company_count,
    current_event.retryable,
    current_event.failure_category,
    current_event.checkpoint_ref,
    current_event.checkpoint_hash,
    current_event.result_row_hash,
    current_event.telemetry_degraded,
    lifecycle.started_at,
    lifecycle.last_heartbeat_at,
    lifecycle.finished_at,
    CASE
        WHEN lifecycle.started_at IS NULL THEN NULL
        WHEN lifecycle.finished_at IS NOT NULL
            THEN GREATEST(0, EXTRACT(EPOCH FROM lifecycle.finished_at - lifecycle.started_at))
        WHEN lifecycle.last_heartbeat_at IS NOT NULL
            THEN GREATEST(0, EXTRACT(EPOCH FROM lifecycle.last_heartbeat_at - lifecycle.started_at))
        ELSE NULL
    END AS observed_runtime_seconds,
    costs.attempt_spend_usd,
    costs.cap_usd
FROM public.research_lab_scoring_icp_executions x
LEFT JOIN LATERAL (
    SELECT e.*
    FROM public.research_lab_scoring_icp_events e
    WHERE e.icp_execution_id = x.icp_execution_id
    ORDER BY e.occurred_at DESC, e.event_id DESC
    LIMIT 1
) current_event ON TRUE
LEFT JOIN LATERAL (
    SELECT
        MIN(e.occurred_at) FILTER (WHERE e.event_type = 'started') AS started_at,
        MAX(e.occurred_at) FILTER (WHERE e.event_type = 'heartbeat') AS last_heartbeat_at,
        MAX(e.occurred_at) FILTER (WHERE e.event_type IN ('completed', 'failed', 'cancelled', 'skipped')) AS finished_at
    FROM public.research_lab_scoring_icp_events e
    WHERE e.icp_execution_id = x.icp_execution_id
) lifecycle ON TRUE
LEFT JOIN LATERAL (
    SELECT
        SUM(c.cost_usd) FILTER (WHERE c.billable) AS attempt_spend_usd,
        MAX(c.cap_usd) AS cap_usd
    FROM public.research_lab_provider_cost_events c
    WHERE c.icp_execution_id = x.icp_execution_id
) costs ON TRUE;

CREATE OR REPLACE VIEW public.research_lab_scoring_dashboard_telemetry_v2
WITH (security_invoker = true) AS
WITH ranked AS (
    SELECT
        x.*,
        r.run_type,
        r.run_attempt,
        r.scheduler_type,
        r.source_run_id,
        r.ticket_id,
        r.candidate_id,
        r.benchmark_id,
        r.benchmark_date,
        r.expected_icp_count,
        COALESCE(r.current_telemetry_degraded, FALSE) AS run_telemetry_degraded,
        CASE
            WHEN x.current_execution_status IN ('completed', 'skipped')
                 AND (x.checkpoint_ref IS NOT NULL OR x.source_job_id IS NOT NULL) THEN 1
            WHEN x.current_execution_status IN ('held', 'queued', 'started', 'heartbeat', 'sourcing_completed', 'scoring_started') THEN 2
            WHEN x.current_execution_status IN ('failed', 'cancelled', 'skipped') THEN 3
            WHEN x.current_execution_status = 'completed' THEN 4
            ELSE 5
        END AS canonical_tier,
        ROW_NUMBER() OVER (
            PARTITION BY x.scoring_id, x.model_role, x.icp_ref
            ORDER BY
                CASE
                    WHEN x.current_execution_status IN ('completed', 'skipped')
                         AND (x.checkpoint_ref IS NOT NULL OR x.source_job_id IS NOT NULL) THEN 1
                    WHEN x.current_execution_status IN ('held', 'queued', 'started', 'heartbeat', 'sourcing_completed', 'scoring_started') THEN 2
                    WHEN x.current_execution_status IN ('failed', 'cancelled', 'skipped') THEN 3
                    WHEN x.current_execution_status = 'completed' THEN 4
                    ELSE 5
                END,
                r.run_attempt DESC,
                x.retry_round DESC,
                x.current_status_at DESC NULLS LAST,
                x.icp_execution_id DESC
        ) AS canonical_rank
    FROM public.research_lab_scoring_icp_execution_current x
    JOIN public.research_lab_scoring_run_current r ON r.scoring_run_id = x.scoring_run_id
), progress AS (
    SELECT
        scoring_id,
        model_role,
        COUNT(*) FILTER (WHERE canonical_rank = 1 AND current_execution_status = 'completed') AS completed_units,
        COUNT(*) FILTER (WHERE canonical_rank = 1 AND current_execution_status = 'skipped') AS skipped_units,
        COUNT(*) FILTER (WHERE canonical_rank = 1 AND current_execution_status = 'failed') AS failed_units,
        COUNT(*) FILTER (
            WHERE canonical_rank = 1
              AND current_execution_status IN ('completed', 'skipped', 'failed', 'cancelled')
        ) AS resolved_units
    FROM ranked
    GROUP BY scoring_id, model_role
), cumulative_costs AS (
    SELECT
        x.scoring_id,
        x.model_role,
        x.icp_ref,
        SUM(c.cost_usd) FILTER (WHERE c.billable) AS cumulative_spend_usd
    FROM public.research_lab_scoring_icp_executions x
    LEFT JOIN public.research_lab_provider_cost_events c
        ON c.icp_execution_id = x.icp_execution_id
    GROUP BY x.scoring_id, x.model_role, x.icp_ref
)
SELECT
    'v2'::TEXT AS telemetry_mode,
    ranked.run_type,
    ranked.scoring_id,
    ranked.scoring_run_id,
    ranked.run_attempt,
    ranked.scheduler_type,
    ranked.source_run_id,
    ranked.ticket_id,
    ranked.candidate_id,
    ranked.benchmark_id,
    ranked.benchmark_date,
    ranked.icp_execution_id,
    ranked.icp_ref,
    ranked.icp_hash,
    ranked.icp_ordinal,
    ranked.phase,
    ranked.model_role,
    ranked.execution_kind,
    ranked.retry_round,
    ranked.current_execution_status AS status,
    ranked.score,
    ranked.sourced_company_count,
    ranked.scored_company_count,
    ranked.attempt_spend_usd,
    cumulative_costs.cumulative_spend_usd,
    ranked.cap_usd,
    ranked.failure_category,
    ranked.retryable,
    ranked.telemetry_degraded OR ranked.run_telemetry_degraded OR ranked.canonical_tier = 4 AS telemetry_degraded,
    ranked.started_at,
    ranked.last_heartbeat_at,
    ranked.finished_at,
    ranked.observed_runtime_seconds,
    ranked.checkpoint_ref,
    ranked.checkpoint_hash,
    ranked.result_row_hash,
    ranked.expected_icp_count AS expected_units,
    progress.completed_units,
    progress.skipped_units,
    progress.failed_units,
    progress.resolved_units,
    CASE
        WHEN ranked.expected_icp_count > 0
            THEN LEAST(1.0, progress.resolved_units::DOUBLE PRECISION / ranked.expected_icp_count)
        ELSE NULL
    END AS progress_ratio
FROM ranked
JOIN progress
    ON progress.scoring_id = ranked.scoring_id
   AND progress.model_role = ranked.model_role
LEFT JOIN cumulative_costs
    ON cumulative_costs.scoring_id = ranked.scoring_id
   AND cumulative_costs.model_role = ranked.model_role
   AND cumulative_costs.icp_ref = ranked.icp_ref
WHERE ranked.canonical_rank = 1;

CREATE OR REPLACE VIEW public.research_lab_scoring_dashboard_telemetry_legacy
WITH (security_invoker = true) AS
WITH baseline_rows AS (
    SELECT
        'legacy'::TEXT AS telemetry_mode,
        'private_baseline_rebenchmark'::TEXT AS run_type,
        'legacy:baseline:' || b.benchmark_bundle_id AS scoring_id,
        NULL::UUID AS scoring_run_id,
        COALESCE(b.benchmark_attempt, 0)::INTEGER AS run_attempt,
        NULL::TEXT AS scheduler_type,
        NULL::UUID AS source_run_id,
        NULL::UUID AS ticket_id,
        NULL::TEXT AS candidate_id,
        'rolling_icp_window:' || b.rolling_window_hash AS benchmark_id,
        b.benchmark_date,
        NULL::UUID AS icp_execution_id,
        item.value->>'icp_ref' AS icp_ref,
        NULLIF(item.value->>'icp_hash', '') AS icp_hash,
        (item.ordinality - 1)::INTEGER AS icp_ordinal,
        'all'::TEXT AS phase,
        'reference'::TEXT AS model_role,
        'model_invocation'::TEXT AS execution_kind,
        0::INTEGER AS retry_round,
        'completed'::TEXT AS status,
        CASE WHEN COALESCE(item.value->>'score', '') ~ '^-?[0-9]+(\.[0-9]+)?$'
            THEN (item.value->>'score')::DOUBLE PRECISION ELSE NULL END AS score,
        CASE WHEN COALESCE(item.value->>'sourced_count', '') ~ '^[0-9]+$'
            THEN (item.value->>'sourced_count')::INTEGER ELSE NULL END AS sourced_company_count,
        CASE WHEN COALESCE(item.value->>'company_count', '') ~ '^[0-9]+$'
            THEN (item.value->>'company_count')::INTEGER ELSE NULL END AS scored_company_count,
        NULL::NUMERIC AS attempt_spend_usd,
        costs.cumulative_spend_usd,
        costs.cap_usd,
        NULL::TEXT AS failure_category,
        NULL::BOOLEAN AS retryable,
        TRUE AS telemetry_degraded,
        NULL::TIMESTAMPTZ AS started_at,
        NULL::TIMESTAMPTZ AS last_heartbeat_at,
        b.created_at AS finished_at,
        NULL::DOUBLE PRECISION AS observed_runtime_seconds,
        NULL::TEXT AS checkpoint_ref,
        NULL::TEXT AS checkpoint_hash,
        NULL::TEXT AS result_row_hash,
        jsonb_array_length(COALESCE(b.score_summary_doc->'per_icp_summaries', '[]'::JSONB))::INTEGER AS expected_units,
        jsonb_array_length(COALESCE(b.score_summary_doc->'per_icp_summaries', '[]'::JSONB))::BIGINT AS completed_units,
        0::BIGINT AS skipped_units,
        0::BIGINT AS failed_units,
        jsonb_array_length(COALESCE(b.score_summary_doc->'per_icp_summaries', '[]'::JSONB))::BIGINT AS resolved_units,
        CASE WHEN jsonb_array_length(COALESCE(b.score_summary_doc->'per_icp_summaries', '[]'::JSONB)) > 0
            THEN 1.0::DOUBLE PRECISION ELSE NULL END AS progress_ratio
    FROM public.research_lab_private_model_benchmark_current b
    CROSS JOIN LATERAL jsonb_array_elements(
        COALESCE(b.score_summary_doc->'per_icp_summaries', '[]'::JSONB)
    ) WITH ORDINALITY AS item(value, ordinality)
    LEFT JOIN LATERAL (
        SELECT
            SUM(c.cost_usd) FILTER (WHERE c.billable) AS cumulative_spend_usd,
            MAX(c.cap_usd) AS cap_usd
        FROM public.research_lab_provider_cost_events c
        WHERE c.scoring_id IS NULL
          AND c.run_type = 'private_baseline_rebenchmark'
          AND c.benchmark_date = b.benchmark_date
          AND c.rolling_window_hash = b.rolling_window_hash
          AND c.icp_ref = item.value->>'icp_ref'
    ) costs ON TRUE
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.research_lab_scoring_runs r
        JOIN public.research_lab_scoring_icp_executions x
          ON x.scoring_run_id = r.scoring_run_id
        WHERE r.run_type = 'private_baseline_rebenchmark'
          AND r.benchmark_date = b.benchmark_date
          AND r.rolling_window_hash = b.rolling_window_hash
          AND r.reference_artifact_hash = b.private_model_artifact_hash
    )
), candidate_rows AS (
    SELECT
        'legacy'::TEXT AS telemetry_mode,
        'candidate_scoring'::TEXT AS run_type,
        'legacy:candidate:' || b.score_bundle_id AS scoring_id,
        NULL::UUID AS scoring_run_id,
        0::INTEGER AS run_attempt,
        NULL::TEXT AS scheduler_type,
        b.run_id AS source_run_id,
        b.ticket_id,
        COALESCE(
            NULLIF(b.score_bundle_doc->>'candidate_id', ''),
            NULLIF(item.value #>> '{evaluation_context,candidate_id}', '')
        ) AS candidate_id,
        COALESCE(
            NULLIF(b.score_bundle_doc->>'benchmark_id', ''),
            NULLIF(item.value #>> '{evaluation_context,benchmark_id}', '')
        ) AS benchmark_id,
        NULL::DATE AS benchmark_date,
        NULL::UUID AS icp_execution_id,
        item.value->>'icp_ref' AS icp_ref,
        NULLIF(item.value->>'icp_hash', '') AS icp_hash,
        (item.ordinality - 1)::INTEGER AS icp_ordinal,
        COALESCE(NULLIF(item.value->>'phase', ''), 'all') AS phase,
        'candidate'::TEXT AS model_role,
        'model_invocation'::TEXT AS execution_kind,
        0::INTEGER AS retry_round,
        'completed'::TEXT AS status,
        scores.score,
        NULL::INTEGER AS sourced_company_count,
        scores.company_count AS scored_company_count,
        NULL::NUMERIC AS attempt_spend_usd,
        costs.cumulative_spend_usd,
        costs.cap_usd,
        NULL::TEXT AS failure_category,
        NULL::BOOLEAN AS retryable,
        TRUE AS telemetry_degraded,
        NULL::TIMESTAMPTZ AS started_at,
        NULL::TIMESTAMPTZ AS last_heartbeat_at,
        b.created_at AS finished_at,
        NULL::DOUBLE PRECISION AS observed_runtime_seconds,
        NULL::TEXT AS checkpoint_ref,
        NULL::TEXT AS checkpoint_hash,
        NULL::TEXT AS result_row_hash,
        jsonb_array_length(COALESCE(b.score_bundle_doc->'per_icp_results', '[]'::JSONB))::INTEGER AS expected_units,
        jsonb_array_length(COALESCE(b.score_bundle_doc->'per_icp_results', '[]'::JSONB))::BIGINT AS completed_units,
        0::BIGINT AS skipped_units,
        0::BIGINT AS failed_units,
        jsonb_array_length(COALESCE(b.score_bundle_doc->'per_icp_results', '[]'::JSONB))::BIGINT AS resolved_units,
        CASE WHEN jsonb_array_length(COALESCE(b.score_bundle_doc->'per_icp_results', '[]'::JSONB)) > 0
            THEN 1.0::DOUBLE PRECISION ELSE NULL END AS progress_ratio
    FROM public.research_evaluation_score_bundle_current b
    CROSS JOIN LATERAL jsonb_array_elements(
        COALESCE(b.score_bundle_doc->'per_icp_results', '[]'::JSONB)
    ) WITH ORDINALITY AS item(value, ordinality)
    LEFT JOIN LATERAL (
        SELECT
            AVG(value::DOUBLE PRECISION) AS score,
            COUNT(*)::INTEGER AS company_count
        FROM jsonb_array_elements_text(
            COALESCE(item.value->'candidate_company_scores', '[]'::JSONB)
        ) score_value(value)
        WHERE value ~ '^-?[0-9]+(\.[0-9]+)?$'
    ) scores ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            SUM(c.cost_usd) FILTER (WHERE c.billable) AS cumulative_spend_usd,
            MAX(c.cap_usd) AS cap_usd
        FROM public.research_lab_provider_cost_events c
        WHERE c.scoring_id IS NULL
          AND c.run_type = 'candidate_scoring'
          AND c.candidate_id = COALESCE(
              NULLIF(b.score_bundle_doc->>'candidate_id', ''),
              NULLIF(item.value #>> '{evaluation_context,candidate_id}', '')
          )
          AND c.rolling_window_hash = b.icp_set_hash
          AND c.icp_ref = item.value->>'icp_ref'
    ) costs ON TRUE
    WHERE b.bundle_status = 'scored'
      AND NOT EXISTS (
        SELECT 1
        FROM public.research_lab_scoring_runs r
        JOIN public.research_lab_scoring_icp_executions x
          ON x.scoring_run_id = r.scoring_run_id
        WHERE r.run_type = 'candidate_scoring'
          AND r.source_run_id = b.run_id
          AND (
              r.candidate_id = COALESCE(
                  NULLIF(b.score_bundle_doc->>'candidate_id', ''),
                  NULLIF(item.value #>> '{evaluation_context,candidate_id}', '')
              )
              OR COALESCE(
                  NULLIF(b.score_bundle_doc->>'candidate_id', ''),
                  NULLIF(item.value #>> '{evaluation_context,candidate_id}', '')
              ) IS NULL
          )
    )
)
SELECT * FROM baseline_rows
UNION ALL
SELECT * FROM candidate_rows;

CREATE OR REPLACE VIEW public.research_lab_scoring_dashboard_telemetry
WITH (security_invoker = true) AS
SELECT * FROM public.research_lab_scoring_dashboard_telemetry_v2
UNION ALL
SELECT * FROM public.research_lab_scoring_dashboard_telemetry_legacy;

CREATE OR REPLACE VIEW public.research_lab_private_benchmark_dashboard_telemetry
WITH (security_invoker = true) AS
SELECT
    t.*,
    t.scoring_run_id AS benchmark_run_id
FROM public.research_lab_scoring_dashboard_telemetry t
WHERE t.run_type = 'private_baseline_rebenchmark';

REVOKE ALL ON TABLE public.research_lab_scoring_run_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_icp_execution_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_dashboard_telemetry_v2 FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_dashboard_telemetry_legacy FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_scoring_dashboard_telemetry FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_private_benchmark_dashboard_telemetry FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_scoring_run_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_scoring_icp_execution_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_scoring_dashboard_telemetry_v2 TO service_role;
GRANT SELECT ON TABLE public.research_lab_scoring_dashboard_telemetry_legacy TO service_role;
GRANT SELECT ON TABLE public.research_lab_scoring_dashboard_telemetry TO service_role;
GRANT SELECT ON TABLE public.research_lab_private_benchmark_dashboard_telemetry TO service_role;

COMMENT ON TABLE public.research_lab_scoring_runs IS
    'Immutable physical scoring-run identities for baseline, candidate, and confirmation scoring telemetry.';
COMMENT ON TABLE public.research_lab_scoring_run_events IS
    'Append-only run lifecycle telemetry; never an authority for scoring, rewards, or weights.';
COMMENT ON TABLE public.research_lab_scoring_icp_executions IS
    'Immutable planned/physical per-ICP attempts, including retries, checkpoint reuse, and explicit skips.';
COMMENT ON TABLE public.research_lab_scoring_icp_events IS
    'Append-only per-ICP lifecycle, counts, redacted failures, and checkpoint/result hashes.';

COMMIT;
