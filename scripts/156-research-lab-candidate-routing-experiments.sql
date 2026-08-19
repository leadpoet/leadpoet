-- Replay-first company-sourcing routing experiments.
--
-- This is a separate, additive lane.  It does not write official candidate
-- evaluation, score-bundle, reimbursement, allocation, or promotion rows.
-- Route compilation remains model-owned by Sourcing_model.  These tables only
-- persist the immutable experiment inputs, normalized route-step outcomes,
-- derived metrics, and routing-only promotion decisions.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_experiments (
    experiment_id          TEXT PRIMARY KEY CHECK (experiment_id ~ '^[A-Za-z0-9_.:-]{1,200}$'),
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_experiment.v1'
    ),
    lane                   TEXT NOT NULL CHECK (
        lane = 'candidate_routing_experiment'
    ),
    mode                   TEXT NOT NULL CHECK (mode = 'replay'),
    experiment_hash        TEXT NOT NULL UNIQUE CHECK (
        experiment_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    experiment_doc          JSONB NOT NULL CHECK (
        jsonb_typeof(experiment_doc) = 'object'
        AND experiment_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (experiment_doc->>'schema_version' = schema_version),
    CHECK (experiment_doc->>'experiment_hash' = experiment_hash),
    CHECK (experiment_doc->>'mode' = mode),
    CHECK ((experiment_doc->>'target_qualified_count')::INTEGER >= 1),
    CHECK (experiment_doc->>'model_commit' ~ '^[0-9a-f]{40}$'),
    CHECK (experiment_doc->>'routing_contract_hash' ~ '^[0-9a-f]{64}$'),
    CHECK (experiment_doc->>'profile_registry_hash' ~ '^[0-9a-f]{64}$'),
    CHECK (experiment_doc->>'provider_catalog_hash' ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_arms (
    arm_id                 TEXT PRIMARY KEY CHECK (arm_id ~ '^[A-Za-z0-9_.:-]{1,200}$'),
    experiment_id           TEXT NOT NULL REFERENCES
        public.research_lab_candidate_routing_experiments(experiment_id)
        ON DELETE RESTRICT,
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_arm.v1'
    ),
    arm_hash               TEXT NOT NULL UNIQUE CHECK (
        arm_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    arm_doc                 JSONB NOT NULL CHECK (
        jsonb_typeof(arm_doc) = 'object'
        AND arm_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_id, arm_id),
    CHECK (arm_doc->>'schema_version' = schema_version),
    CHECK (arm_doc->>'arm_id' = arm_id),
    CHECK (arm_doc->>'experiment_id' = experiment_id),
    CHECK (arm_doc->>'arm_hash' = arm_hash),
    CHECK (arm_doc->>'profile_hash' ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_runs (
    run_id                 TEXT PRIMARY KEY CHECK (run_id ~ '^[A-Za-z0-9_.:-]{1,200}$'),
    experiment_id           TEXT NOT NULL REFERENCES
        public.research_lab_candidate_routing_experiments(experiment_id)
        ON DELETE RESTRICT,
    arm_id                 TEXT NOT NULL,
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_run.v1'
    ),
    run_hash               TEXT NOT NULL UNIQUE CHECK (
        run_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    icp_ref                TEXT NOT NULL CHECK (icp_ref ~ '^[A-Za-z0-9_.:-]{1,200}$'),
    icp_hash               TEXT NOT NULL CHECK (icp_hash ~ '^sha256:[0-9a-f]{64}$'),
    snapshot_manifest_hash TEXT NOT NULL CHECK (snapshot_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    route_plan_hash        TEXT NOT NULL CHECK (route_plan_hash ~ '^[0-9a-f]{64}$'),
    run_status             TEXT NOT NULL CHECK (
        run_status IN ('completed', 'failed', 'replay_miss')
    ),
    run_doc                JSONB NOT NULL CHECK (
        jsonb_typeof(run_doc) = 'object'
        AND run_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_id, arm_id, run_id),
    FOREIGN KEY (experiment_id, arm_id) REFERENCES
        public.research_lab_candidate_routing_arms(experiment_id, arm_id)
        ON DELETE RESTRICT,
    CHECK (run_doc->>'schema_version' = schema_version),
    CHECK (run_doc->>'run_id' = run_id),
    CHECK (run_doc->>'experiment_id' = experiment_id),
    CHECK (run_doc->>'arm_id' = arm_id),
    CHECK (run_doc->>'run_hash' = run_hash),
    CHECK (run_doc->>'route_plan_hash' = route_plan_hash),
    CHECK (run_doc->>'status' = run_status)
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_attempts (
    attempt_id             TEXT PRIMARY KEY CHECK (attempt_id ~ '^[A-Za-z0-9_.:-]{1,200}$'),
    experiment_id           TEXT NOT NULL REFERENCES
        public.research_lab_candidate_routing_experiments(experiment_id)
        ON DELETE RESTRICT,
    arm_id                 TEXT NOT NULL,
    run_id                 TEXT NOT NULL,
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_attempt.v1'
    ),
    attempt_hash           TEXT NOT NULL UNIQUE CHECK (
        attempt_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    step_order             INTEGER NOT NULL CHECK (step_order >= 0),
    attempt_sequence       INTEGER NOT NULL CHECK (attempt_sequence >= 0),
    tool_id                TEXT NOT NULL CHECK (tool_id ~ '^candidate\.[A-Za-z0-9_.:-]{1,160}$'),
    disposition            TEXT NOT NULL CHECK (
        disposition IN ('considered', 'selected', 'attempted', 'succeeded', 'missed', 'failed', 'deferred', 'skipped')
    ),
    outcome                TEXT NOT NULL CHECK (
        outcome IN ('success', 'miss', 'failed', 'retryable_failure', 'replay_miss', 'skipped')
    ),
    route_plan_hash        TEXT NOT NULL CHECK (route_plan_hash ~ '^[0-9a-f]{64}$'),
    stop_policy_hash       TEXT NOT NULL CHECK (stop_policy_hash ~ '^[0-9a-f]{64}$'),
    attempt_receipt_hash   TEXT NOT NULL CHECK (attempt_receipt_hash ~ '^[0-9a-f]{64}$'),
    verification_receipt_hash TEXT NOT NULL DEFAULT '' CHECK (
        verification_receipt_hash = ''
        OR verification_receipt_hash ~ '^[0-9a-f]{64}$'
    ),
    provider_id            TEXT NOT NULL DEFAULT '',
    provider_call_count    INTEGER NOT NULL DEFAULT 0 CHECK (provider_call_count >= 0),
    cost_microusd          BIGINT NOT NULL DEFAULT 0 CHECK (cost_microusd >= 0),
    latency_ms             BIGINT NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
    raw_count              INTEGER NOT NULL DEFAULT 0 CHECK (raw_count >= 0),
    unique_count           INTEGER NOT NULL DEFAULT 0 CHECK (unique_count >= 0),
    verified_count         INTEGER NOT NULL DEFAULT 0 CHECK (verified_count >= 0),
    qualified_count        INTEGER NOT NULL DEFAULT 0 CHECK (qualified_count >= 0),
    published_count        INTEGER NOT NULL DEFAULT 0 CHECK (published_count >= 0),
    snapshot_hit           BOOLEAN NOT NULL DEFAULT TRUE,
    result_hash            TEXT NOT NULL DEFAULT '' CHECK (
        result_hash = '' OR result_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    attempt_doc            JSONB NOT NULL CHECK (
        jsonb_typeof(attempt_doc) = 'object'
        AND attempt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, step_order, attempt_sequence),
    CHECK (attempt_doc->>'schema_version' = schema_version),
    CHECK (attempt_doc->>'attempt_id' = attempt_id),
    CHECK (attempt_doc->>'experiment_id' = experiment_id),
    CHECK (attempt_doc->>'arm_id' = arm_id),
    CHECK (attempt_doc->>'run_id' = run_id),
    CHECK (attempt_doc->>'attempt_hash' = attempt_hash),
    CHECK (attempt_doc->>'route_plan_hash' = route_plan_hash),
    CHECK (attempt_doc->>'stop_policy_hash' = stop_policy_hash),
    CHECK (attempt_doc->>'attempt_receipt_hash' = attempt_receipt_hash),
    CHECK (attempt_doc->>'verification_receipt_hash' = verification_receipt_hash),
    FOREIGN KEY (experiment_id, arm_id, run_id) REFERENCES
        public.research_lab_candidate_routing_runs(experiment_id, arm_id, run_id)
        ON DELETE RESTRICT,
    CHECK (raw_count >= unique_count),
    CHECK (unique_count >= verified_count),
    CHECK (verified_count >= qualified_count),
    CHECK (qualified_count >= published_count),
    CHECK (qualified_count = 0 OR verification_receipt_hash <> ''),
    CHECK (outcome <> 'replay_miss' OR snapshot_hit IS FALSE)
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_metrics (
    metric_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id           TEXT NOT NULL REFERENCES
        public.research_lab_candidate_routing_experiments(experiment_id)
        ON DELETE RESTRICT,
    arm_id                 TEXT NOT NULL,
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_metric.v1'
    ),
    metric_hash            TEXT NOT NULL UNIQUE CHECK (
        metric_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    metric_doc              JSONB NOT NULL CHECK (
        jsonb_typeof(metric_doc) = 'object'
        AND metric_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (metric_doc->>'schema_version' = schema_version),
    CHECK (metric_doc->>'experiment_id' = experiment_id),
    CHECK (metric_doc->>'arm_id' = arm_id),
    CHECK (metric_doc->>'metric_hash' = metric_hash),
    UNIQUE (experiment_id, arm_id, metric_hash),
    FOREIGN KEY (experiment_id, arm_id) REFERENCES
        public.research_lab_candidate_routing_arms(experiment_id, arm_id)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_routing_decisions (
    decision_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id           TEXT NOT NULL REFERENCES
        public.research_lab_candidate_routing_experiments(experiment_id)
        ON DELETE RESTRICT,
    arm_id                 TEXT NOT NULL,
    metric_hash            TEXT NOT NULL,
    schema_version          TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.candidate_routing_decision.v1'
    ),
    decision_hash          TEXT NOT NULL UNIQUE CHECK (
        decision_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    decision_state         TEXT NOT NULL CHECK (
        decision_state IN ('rejected', 'replay_only', 'eligible_for_shadow', 'eligible_for_canary')
    ),
    decision_doc            JSONB NOT NULL CHECK (
        jsonb_typeof(decision_doc) = 'object'
        AND decision_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (decision_doc->>'schema_version' = schema_version),
    CHECK (decision_doc->>'experiment_id' = experiment_id),
    CHECK (decision_doc->>'arm_id' = arm_id),
    CHECK (decision_doc->>'metric_hash' = metric_hash),
    CHECK (decision_doc->>'decision_hash' = decision_hash),
    -- This is a routing experiment state, never an official model promotion.
    CHECK (decision_doc->>'promotion_scope' = 'candidate_routing_experiment'),
    FOREIGN KEY (experiment_id, arm_id) REFERENCES
        public.research_lab_candidate_routing_arms(experiment_id, arm_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (experiment_id, arm_id, metric_hash) REFERENCES
        public.research_lab_candidate_routing_metrics(experiment_id, arm_id, metric_hash)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_routing_arms_experiment
    ON public.research_lab_candidate_routing_arms(experiment_id, arm_id);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_routing_runs_experiment_arm
    ON public.research_lab_candidate_routing_runs(experiment_id, arm_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_routing_attempts_run_step
    ON public.research_lab_candidate_routing_attempts(run_id, step_order, attempt_sequence);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_routing_metrics_experiment_arm
    ON public.research_lab_candidate_routing_metrics(experiment_id, arm_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_routing_decisions_experiment_arm
    ON public.research_lab_candidate_routing_decisions(experiment_id, arm_id, created_at DESC);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_candidate_routing_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; write a new immutable row', TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_candidate_routing_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_candidate_routing_mutation()
    TO service_role;

DO $routing_triggers$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_candidate_routing_experiments',
        'research_lab_candidate_routing_arms',
        'research_lab_candidate_routing_runs',
        'research_lab_candidate_routing_attempts',
        'research_lab_candidate_routing_metrics',
        'research_lab_candidate_routing_decisions'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_trigger trigger_meta
            JOIN pg_catalog.pg_class relation_meta
              ON relation_meta.oid = trigger_meta.tgrelid
            JOIN pg_catalog.pg_namespace namespace_meta
              ON namespace_meta.oid = relation_meta.relnamespace
            WHERE namespace_meta.nspname = 'public'
              AND relation_meta.relname = relation_name
              AND trigger_meta.tgname = 'trg_' || relation_name || '_no_mutation'
              AND NOT trigger_meta.tgisinternal
        ) THEN
            EXECUTE pg_catalog.format(
                'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I '
                'FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_candidate_routing_mutation()',
                'trg_' || relation_name || '_no_mutation',
                relation_name
            );
        END IF;
    END LOOP;
END;
$routing_triggers$;

REVOKE ALL ON TABLE
    public.research_lab_candidate_routing_experiments,
    public.research_lab_candidate_routing_arms,
    public.research_lab_candidate_routing_runs,
    public.research_lab_candidate_routing_attempts,
    public.research_lab_candidate_routing_metrics,
    public.research_lab_candidate_routing_decisions
FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE
    public.research_lab_candidate_routing_experiments,
    public.research_lab_candidate_routing_arms,
    public.research_lab_candidate_routing_runs,
    public.research_lab_candidate_routing_attempts,
    public.research_lab_candidate_routing_metrics,
    public.research_lab_candidate_routing_decisions
TO service_role;

ALTER TABLE public.research_lab_candidate_routing_experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_experiments FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_arms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_arms FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_attempts FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_metrics FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_routing_decisions FORCE ROW LEVEL SECURITY;

DO $routing_policies$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_candidate_routing_experiments',
        'research_lab_candidate_routing_arms',
        'research_lab_candidate_routing_runs',
        'research_lab_candidate_routing_attempts',
        'research_lab_candidate_routing_metrics',
        'research_lab_candidate_routing_decisions'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_policies
            WHERE schemaname = 'public'
              AND tablename = relation_name
              AND policyname = 'service_role_read'
        ) THEN
            EXECUTE pg_catalog.format(
                'CREATE POLICY service_role_read ON public.%I FOR SELECT TO service_role USING (true)',
                relation_name
            );
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_policies
            WHERE schemaname = 'public'
              AND tablename = relation_name
              AND policyname = 'service_role_insert'
        ) THEN
            EXECUTE pg_catalog.format(
                'CREATE POLICY service_role_insert ON public.%I FOR INSERT TO service_role WITH CHECK (true)',
                relation_name
            );
        END IF;
    END LOOP;
END;
$routing_policies$;

COMMENT ON TABLE public.research_lab_candidate_routing_experiments IS
    'Replay-only candidate-acquisition routing experiments; separate from official candidate evaluation and promotion.';
COMMENT ON TABLE public.research_lab_candidate_routing_arms IS
    'Immutable control and candidate Sourcing_model profile pins for a routing experiment.';
COMMENT ON TABLE public.research_lab_candidate_routing_runs IS
    'Immutable ICP/arm replay runs bound to a verified provider snapshot manifest.';
COMMENT ON TABLE public.research_lab_candidate_routing_attempts IS
    'Append-only normalized per-step candidate routing outcomes; no provider payloads.';
COMMENT ON TABLE public.research_lab_candidate_routing_metrics IS
    'Append-only derived routing metrics; metrics are not official score bundles.';
COMMENT ON TABLE public.research_lab_candidate_routing_decisions IS
    'Routing-only rejected/replay_only/eligible_for_shadow/eligible_for_canary decisions.';

COMMIT;
