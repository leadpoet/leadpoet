-- Research Lab Phase 0: v5 trajectory, execution-trace, evidence-bundle,
-- and results-ledger storage.
--
-- Deployment policy:
--   * Do not apply until all Phase 0 code + SQL files are written/reviewed.
--   * Apply after scripts 25 and 26 when production SQL rollout begins.
--   * Smoke-test after applying by inserting service-role-only fixture rows
--     into each table in a lab-only / non-production workflow.
--   * If an earlier draft of this file was already applied to a staging DB,
--     drop these five Research Lab tables before re-testing this final file.
--     CREATE TABLE IF NOT EXISTS will not retrofit table-body changes such as
--     FK actions or CHECK constraints onto an earlier-draft table.
--   * Emitters must canonicalize event cost_usd to at most 6 decimal places;
--     the event JSONB/column consistency check compares against NUMERIC(12,6).
--
-- Access model:
--   * Structured Research Lab corpus tables are private, service_role only.
--   * No anon/authenticated grants are created.
--   * RLS is enabled as defense in depth for Supabase exposed-schema safety.
--   * research_trajectory_events is append-only by grant: service_role gets
--     SELECT and INSERT, but not UPDATE or DELETE.
--   * research_trajectory_events is also append-only by trigger so direct SQL
--     paths cannot mutate anchored events accidentally.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_trajectories (
    trajectory_id       UUID        PRIMARY KEY,
    schema_version      TEXT        NOT NULL DEFAULT '1.0'
                                      CHECK (schema_version = '1.0'),
    brief_id            UUID        NOT NULL,
    island              TEXT        NOT NULL,
    funder_hotkey       TEXT,
    brief_sanitized_ref TEXT        NOT NULL,
    novelty_gate        JSONB       NOT NULL
                                      CHECK (jsonb_typeof(novelty_gate) = 'object'),
    engine_version      TEXT        NOT NULL,
    champion_base       TEXT        NOT NULL,
    final               JSONB       CHECK (final IS NULL OR jsonb_typeof(final) = 'object'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_trajectory_events (
    trajectory_id UUID        NOT NULL
                              CONSTRAINT research_trajectory_events_trajectory_id_fkey
                              REFERENCES public.research_trajectories(trajectory_id)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    ts            TIMESTAMPTZ NOT NULL,
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
                                  'PROBE',
                                  'LOOP_FUNDED',
                                  'NODE_DRAFTED',
                                  'NODE_EVALUATED',
                                  'NODE_REFLECTED',
                                  'PLATEAU_STOP',
                                  'L2_PROMOTED',
                                  'LANE_ENTERED',
                                  'PROBATION_SET_SCORED',
                                  'CROWNED',
                                  'REVERTED',
                                  'FINALIZED'
                              )),
    cost_usd      NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    anchored_hash TEXT        NOT NULL,
    event         JSONB       NOT NULL CHECK (
                              jsonb_typeof(event) = 'object'
                              AND event ? 'seq'
                              AND event ? 'ts'
                              AND event ? 'type'
                              AND event ? 'cost_usd'
                              AND event ? 'anchored_hash'
                              AND event->>'type' = event_type
                              AND event->>'anchored_hash' = anchored_hash
                              AND (event->>'seq')::INTEGER = seq
                              AND (event->>'cost_usd')::NUMERIC = cost_usd
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_trajectory_events_pkey PRIMARY KEY (trajectory_id, seq)
);

CREATE TABLE IF NOT EXISTS public.execution_traces (
    run_id            UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                  CHECK (schema_version = '1.0'),
    artifact_hash     TEXT        NOT NULL,
    role              TEXT        NOT NULL CHECK (
                                  role IN (
                                      'champion',
                                      'candidate',
                                      'shadow',
                                      'baseline_arm',
                                      'reference'
                                  )),
    rung              TEXT        NOT NULL CHECK (
                                  rung IN ('L0', 'L1', 'L2', 'L3', 'L4', 'anchor')),
    status            TEXT        NOT NULL CHECK (status IN ('completed', 'crash', 'timeout')),
    lane_id           UUID,
    icp_set_hash      TEXT        NOT NULL,
    eval_version      JSONB       NOT NULL CHECK (jsonb_typeof(eval_version) = 'object'),
    calls             JSONB       NOT NULL DEFAULT '[]'::JSONB
                                  CHECK (jsonb_typeof(calls) = 'array'),
    evidence_bundles  JSONB       NOT NULL DEFAULT '[]'::JSONB
                                  CHECK (jsonb_typeof(evidence_bundles) = 'array'),
    judge_verdicts    JSONB       NOT NULL DEFAULT '[]'::JSONB
                                  CHECK (jsonb_typeof(judge_verdicts) = 'array'),
    outputs_ref       TEXT        NOT NULL,
    score_bundle_ref  TEXT        NOT NULL,
    cost_ledger       JSONB       NOT NULL CHECK (jsonb_typeof(cost_ledger) = 'object'),
    attestation_ref   TEXT,
    trace_doc         JSONB       CHECK (trace_doc IS NULL OR jsonb_typeof(trace_doc) = 'object'),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.evidence_bundles (
    bundle_id            UUID        PRIMARY KEY,
    schema_version       TEXT        NOT NULL DEFAULT '1.0'
                                     CHECK (schema_version = '1.0'),
    run_id               UUID,
    artifact_hash        TEXT        NOT NULL,
    retention_class      TEXT        NOT NULL CHECK (
                                     retention_class IN (
                                         'live_verification',
                                         'regression_anchor',
                                         'general_snapshot'
                                     )),
    verification_state   TEXT        NOT NULL CHECK (
                                     verification_state IN (
                                         'active',
                                         'content_deleted',
                                         'hash_attested'
                                     )),
    bundle_hash          TEXT        NOT NULL UNIQUE,
    merkle_anchor_ref    TEXT,
    deletion_request_ref TEXT,
    snapshots            JSONB       NOT NULL CHECK (
                                     jsonb_typeof(snapshots) = 'array'
                                     AND jsonb_array_length(snapshots) > 0
                                     ),
    bundle_doc           JSONB       CHECK (bundle_doc IS NULL OR jsonb_typeof(bundle_doc) = 'object'),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_results_ledger (
    ledger_row_id   UUID        PRIMARY KEY,
    schema_version  TEXT        NOT NULL DEFAULT '1.0'
                                CHECK (schema_version = '1.0'),
    trajectory_id   UUID        REFERENCES public.research_trajectories(trajectory_id)
                                ON DELETE SET NULL,
    node_id         TEXT        NOT NULL,
    commit          TEXT        NOT NULL,
    island          TEXT        NOT NULL,
    brief_id        UUID,
    targeted_metric TEXT        NOT NULL,
    delta_vs_parent DOUBLE PRECISION,
    cost_usd        NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    status          TEXT        NOT NULL CHECK (status IN ('keep', 'discard', 'crash', 'timeout')),
    description     TEXT        NOT NULL,
    source_event_seq INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.research_trajectory_events
    DROP CONSTRAINT IF EXISTS research_trajectory_events_trajectory_id_fkey;

ALTER TABLE public.research_trajectory_events
    ADD CONSTRAINT research_trajectory_events_trajectory_id_fkey
    FOREIGN KEY (trajectory_id)
    REFERENCES public.research_trajectories(trajectory_id)
    ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS idx_research_trajectories_brief_id
    ON public.research_trajectories(brief_id);

CREATE INDEX IF NOT EXISTS idx_research_trajectories_island_created
    ON public.research_trajectories(island, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_trajectory_events_type_ts
    ON public.research_trajectory_events(event_type, ts DESC);

CREATE INDEX IF NOT EXISTS idx_research_trajectory_events_event_gin
    ON public.research_trajectory_events USING GIN (event jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_execution_traces_role_rung_created
    ON public.execution_traces(role, rung, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_execution_traces_artifact_hash
    ON public.execution_traces(artifact_hash);

CREATE INDEX IF NOT EXISTS idx_execution_traces_calls_gin
    ON public.execution_traces USING GIN (calls jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_run_id
    ON public.evidence_bundles(run_id);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_retention_state
    ON public.evidence_bundles(retention_class, verification_state);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_snapshots_gin
    ON public.evidence_bundles USING GIN (snapshots jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_research_lab_results_ledger_trajectory
    ON public.research_lab_results_ledger(trajectory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_lab_results_ledger_island_status
    ON public.research_lab_results_ledger(island, status, created_at DESC);

CREATE OR REPLACE FUNCTION public.prevent_research_trajectory_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        'research_trajectory_events is append-only; write a correction event instead';
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_trajectory_event_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_trajectory_event_mutation()
    TO service_role;

DROP TRIGGER IF EXISTS prevent_research_trajectory_event_mutation
    ON public.research_trajectory_events;

CREATE TRIGGER prevent_research_trajectory_event_mutation
    BEFORE UPDATE OR DELETE ON public.research_trajectory_events
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_research_trajectory_event_mutation();

REVOKE ALL ON TABLE public.research_trajectories FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_trajectory_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.execution_traces FROM anon, authenticated;
REVOKE ALL ON TABLE public.evidence_bundles FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_results_ledger FROM anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.research_trajectories TO service_role;
REVOKE UPDATE, DELETE ON TABLE public.research_trajectory_events FROM service_role;
GRANT SELECT, INSERT ON TABLE public.research_trajectory_events TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.execution_traces TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.evidence_bundles TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.research_lab_results_ledger TO service_role;

ALTER TABLE public.research_trajectories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_trajectory_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.execution_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evidence_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_results_ledger ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all ON public.research_trajectories;
CREATE POLICY service_role_all ON public.research_trajectories
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_trajectory_events;
CREATE POLICY service_role_read ON public.research_trajectory_events
    FOR SELECT
    USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_insert ON public.research_trajectory_events;
CREATE POLICY service_role_insert ON public.research_trajectory_events
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_all ON public.execution_traces;
CREATE POLICY service_role_all ON public.execution_traces
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_all ON public.evidence_bundles;
CREATE POLICY service_role_all ON public.evidence_bundles
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_all ON public.research_lab_results_ledger;
CREATE POLICY service_role_all ON public.research_lab_results_ledger
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE public.research_trajectories IS
    'Leadpoet Research Lab trajectory envelope, one row per sanitized brief. Events are stored append-only in research_trajectory_events.';

COMMENT ON TABLE public.research_trajectory_events IS
    'Append-only event log for Research Lab trajectories. Every event carries seq, ts, type, cost_usd, and anchored_hash. Corrections must be written as new events.';

COMMENT ON TABLE public.execution_traces IS
    'Notary-proxy and run-fabric execution traces for champion, candidate, shadow, reference, and baseline-arm runs.';

COMMENT ON TABLE public.evidence_bundles IS
    'Canonical verification objects for notarized evidence snapshots. Raw page content lives outside this structured corpus.';

COMMENT ON TABLE public.research_lab_results_ledger IS
    'Flat lab-side projection of trajectory events, one row per expansion, used for keep rate, cost, crash rate, and frontier charts.';

COMMENT ON COLUMN public.research_trajectories.brief_sanitized_ref IS
    'Hash or storage ref for sanitized brief text. Unsanitized miner brief text must not enter the structured corpus.';

COMMENT ON COLUMN public.research_trajectory_events.anchored_hash IS
    'Hash committed to the anchoring pipeline for tamper-evident provenance.';

COMMENT ON COLUMN public.execution_traces.calls IS
    'Array of provider/runtime calls including call_emitter=model|code and teacher_model_flag for distillation curation.';

COMMENT ON COLUMN public.evidence_bundles.verification_state IS
    'active | content_deleted | hash_attested. Supports deletion-with-hash-retention without breaking deterministic L0 behavior.';

COMMIT;

-- Smoke checks to run after production application:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_trajectories',
--       'research_trajectory_events',
--       'execution_traces',
--       'evidence_bundles',
--       'research_lab_results_ledger'
--     )
--   ORDER BY table_name;
--
--   SELECT relname, relrowsecurity
--   FROM pg_class
--   WHERE relname IN (
--       'research_trajectories',
--       'research_trajectory_events',
--       'execution_traces',
--       'evidence_bundles',
--       'research_lab_results_ledger'
--   );
