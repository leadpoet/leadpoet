-- Candidate-specific PostgreSQL sidecars for the shared routing experiment.
--
-- PR 93 owns experiment, budget, provider-receipt, decision, evaluation, and
-- promotion contracts. These tables persist only exact Model waterfall
-- receipt projections and candidate yield metrics that reference those
-- immutable shared documents. They do not create a parallel lifecycle.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_waterfall_receipts (
    receipt_id                    TEXT PRIMARY KEY CHECK (
        receipt_id ~ '^candidate_waterfall:[0-9a-f]{24}$'
    ),
    receipt_hash                  TEXT NOT NULL UNIQUE CHECK (
        receipt_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    contract_version              TEXT NOT NULL CHECK (
        contract_version = 'leadpoet.candidate_waterfall_receipt_sidecar:v1'
    ),
    experiment_id                 TEXT NOT NULL,
    experiment_hash               TEXT NOT NULL CHECK (
        experiment_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    variant_id                    TEXT NOT NULL,
    artifact_key                  TEXT NOT NULL CHECK (
        artifact_key ~ '^sha256:[0-9a-f]{64}$'
    ),
    decision_receipt_id           TEXT NOT NULL CHECK (
        decision_receipt_id ~ '^routing_decision:[0-9a-f]{16}$'
    ),
    provider_receipt_ref          TEXT NOT NULL DEFAULT '' CHECK (
        provider_receipt_ref = ''
        OR provider_receipt_ref ~ '^provider_receipt:[0-9a-f]{16}$'
    ),
    unit_ref                      TEXT NOT NULL,
    binding_id                    TEXT NOT NULL,
    tool_id                       TEXT NOT NULL CHECK (
        tool_id ~ '^candidate\.[A-Za-z0-9_.:-]{1,160}$'
    ),
    execution_mode                TEXT NOT NULL CHECK (
        execution_mode IN ('fixture', 'replay', 'measured_lab')
    ),
    provider_outcome              TEXT NOT NULL CHECK (
        provider_outcome IN (
            'verified',
            'rejected',
            'source_miss',
            'adapter_failure',
            'skipped'
        )
    ),
    decision_plan_hash            TEXT NOT NULL CHECK (
        decision_plan_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    decision_route_hash           TEXT NOT NULL CHECK (
        decision_route_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    model_contract_sha256         TEXT NOT NULL CHECK (
        model_contract_sha256 ~ '^[0-9a-f]{64}$'
    ),
    model_plan_sha256             TEXT NOT NULL CHECK (
        model_plan_sha256 ~ '^[0-9a-f]{64}$'
    ),
    stop_policy_sha256            TEXT NOT NULL CHECK (
        stop_policy_sha256 ~ '^[0-9a-f]{64}$'
    ),
    attempt_receipt_sha256        TEXT NOT NULL CHECK (
        attempt_receipt_sha256 ~ '^[0-9a-f]{64}$'
    ),
    verification_receipt_sha256   TEXT NOT NULL DEFAULT '' CHECK (
        verification_receipt_sha256 = ''
        OR verification_receipt_sha256 ~ '^[0-9a-f]{64}$'
    ),
    step_order                    INTEGER NOT NULL CHECK (step_order >= 0),
    attempt_sequence              INTEGER NOT NULL CHECK (attempt_sequence >= 0),
    disposition                   TEXT NOT NULL CHECK (
        disposition IN ('succeeded', 'missed', 'failed', 'deferred', 'skipped')
    ),
    outcome_code                  TEXT NOT NULL,
    provider_call_count           INTEGER NOT NULL CHECK (provider_call_count >= 0),
    cost_microusd                 BIGINT NOT NULL CHECK (cost_microusd >= 0),
    latency_ms                    BIGINT NOT NULL CHECK (latency_ms >= 0),
    raw_count                     INTEGER NOT NULL CHECK (raw_count >= 0),
    normalized_count              INTEGER NOT NULL CHECK (normalized_count >= 0),
    unique_count                  INTEGER NOT NULL CHECK (unique_count >= 0),
    verified_qualified_count      INTEGER NOT NULL CHECK (verified_qualified_count >= 0),
    published_count               INTEGER NOT NULL CHECK (published_count >= 0),
    receipt_doc                   JSONB NOT NULL CHECK (
        jsonb_typeof(receipt_doc) = 'object'
        AND receipt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_id, variant_id, unit_ref, step_order, attempt_sequence),
    CHECK (raw_count >= normalized_count),
    CHECK (normalized_count >= unique_count),
    CHECK (unique_count >= verified_qualified_count),
    CHECK (verified_qualified_count >= published_count),
    CHECK (verified_qualified_count = 0 OR verification_receipt_sha256 <> ''),
    CHECK (
        (
            disposition = 'skipped'
            AND provider_receipt_ref = ''
            AND provider_outcome = 'skipped'
        )
        OR (
            disposition <> 'skipped'
            AND provider_receipt_ref <> ''
            AND provider_outcome <> 'skipped'
        )
    ),
    CHECK (receipt_doc = jsonb_build_object(
        'receipt_id', receipt_id,
        'receipt_hash', receipt_hash,
        'contract_version', contract_version,
        'experiment_id', experiment_id,
        'experiment_hash', experiment_hash,
        'variant_id', variant_id,
        'artifact_key', artifact_key,
        'decision_receipt_id', decision_receipt_id,
        'provider_receipt_ref', provider_receipt_ref,
        'unit_ref', unit_ref,
        'binding_id', binding_id,
        'tool_id', tool_id,
        'execution_mode', execution_mode,
        'provider_outcome', provider_outcome,
        'decision_plan_hash', decision_plan_hash,
        'decision_route_hash', decision_route_hash,
        'model_contract_sha256', model_contract_sha256,
        'model_plan_sha256', model_plan_sha256,
        'stop_policy_sha256', stop_policy_sha256,
        'attempt_receipt_sha256', attempt_receipt_sha256,
        'verification_receipt_sha256', verification_receipt_sha256,
        'step_order', step_order,
        'attempt_sequence', attempt_sequence,
        'disposition', disposition,
        'outcome_code', outcome_code,
        'provider_call_count', provider_call_count,
        'cost_microusd', cost_microusd,
        'latency_ms', latency_ms,
        'raw_count', raw_count,
        'normalized_count', normalized_count,
        'unique_count', unique_count,
        'verified_qualified_count', verified_qualified_count,
        'published_count', published_count,
        'immutable', TRUE
    ))
);

CREATE TABLE IF NOT EXISTS public.research_lab_candidate_waterfall_metrics (
    metric_id                       TEXT PRIMARY KEY CHECK (
        metric_id ~ '^candidate_metric:[0-9a-f]{24}$'
    ),
    metric_hash                     TEXT NOT NULL UNIQUE CHECK (
        metric_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    contract_version                TEXT NOT NULL CHECK (
        contract_version = 'leadpoet.candidate_waterfall_metric_sidecar:v1'
    ),
    evaluation_receipt_id           TEXT NOT NULL CHECK (
        evaluation_receipt_id ~ '^routing_evaluation_v2:[A-Za-z0-9_.:-]+$'
    ),
    experiment_id                   TEXT NOT NULL,
    experiment_hash                 TEXT NOT NULL CHECK (
        experiment_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    variant_id                      TEXT NOT NULL,
    split                           TEXT NOT NULL CHECK (
        split IN ('calibration', 'holdout')
    ),
    target_verified_qualified_count INTEGER NOT NULL CHECK (
        target_verified_qualified_count >= 1
    ),
    unit_count                      INTEGER NOT NULL CHECK (unit_count >= 1),
    fulfilled_unit_count            INTEGER NOT NULL CHECK (
        fulfilled_unit_count >= 0 AND fulfilled_unit_count <= unit_count
    ),
    waterfall_attempt_count         INTEGER NOT NULL CHECK (waterfall_attempt_count >= 0),
    provider_call_count             INTEGER NOT NULL CHECK (provider_call_count >= 0),
    total_cost_microusd             BIGINT NOT NULL CHECK (total_cost_microusd >= 0),
    total_latency_ms                BIGINT NOT NULL CHECK (total_latency_ms >= 0),
    raw_count                       INTEGER NOT NULL CHECK (raw_count >= 0),
    normalized_count                INTEGER NOT NULL CHECK (normalized_count >= 0),
    unique_count                    INTEGER NOT NULL CHECK (unique_count >= 0),
    verified_qualified_count        INTEGER NOT NULL CHECK (verified_qualified_count >= 0),
    published_count                 INTEGER NOT NULL CHECK (published_count >= 0),
    failed_attempt_count            INTEGER NOT NULL CHECK (failed_attempt_count >= 0),
    missed_attempt_count            INTEGER NOT NULL CHECK (missed_attempt_count >= 0),
    fulfillment_rate                DOUBLE PRECISION NOT NULL CHECK (
        fulfillment_rate >= 0 AND fulfillment_rate <= 1
    ),
    verification_rate               DOUBLE PRECISION NOT NULL CHECK (
        verification_rate >= 0 AND verification_rate <= 1
    ),
    publication_rate                DOUBLE PRECISION NOT NULL CHECK (
        publication_rate >= 0 AND publication_rate <= 1
    ),
    verified_qualified_per_usd      DOUBLE PRECISION NOT NULL CHECK (
        verified_qualified_per_usd >= 0
    ),
    metric_doc                      JSONB NOT NULL CHECK (
        jsonb_typeof(metric_doc) = 'object'
        AND metric_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|password|authorization:|judge_prompt)'
    ),
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (evaluation_receipt_id, variant_id, split, metric_hash),
    CHECK (raw_count >= normalized_count),
    CHECK (normalized_count >= unique_count),
    CHECK (unique_count >= verified_qualified_count),
    CHECK (verified_qualified_count >= published_count),
    CHECK (jsonb_typeof(metric_doc->'waterfall_receipt_refs') = 'array'),
    CHECK (jsonb_typeof(metric_doc->'provider_receipt_refs') = 'array'),
    CHECK (jsonb_typeof(metric_doc->'decision_receipt_refs') = 'array'),
    CHECK (metric_doc = jsonb_build_object(
        'metric_id', metric_id,
        'metric_hash', metric_hash,
        'contract_version', contract_version,
        'evaluation_receipt_id', evaluation_receipt_id,
        'experiment_id', experiment_id,
        'experiment_hash', experiment_hash,
        'variant_id', variant_id,
        'split', split,
        'target_verified_qualified_count', target_verified_qualified_count,
        'unit_count', unit_count,
        'fulfilled_unit_count', fulfilled_unit_count,
        'waterfall_attempt_count', waterfall_attempt_count,
        'provider_call_count', provider_call_count,
        'total_cost_microusd', total_cost_microusd,
        'total_latency_ms', total_latency_ms,
        'raw_count', raw_count,
        'normalized_count', normalized_count,
        'unique_count', unique_count,
        'verified_qualified_count', verified_qualified_count,
        'published_count', published_count,
        'failed_attempt_count', failed_attempt_count,
        'missed_attempt_count', missed_attempt_count,
        'fulfillment_rate', fulfillment_rate,
        'verification_rate', verification_rate,
        'publication_rate', publication_rate,
        'verified_qualified_per_usd', verified_qualified_per_usd,
        'waterfall_receipt_refs', metric_doc->'waterfall_receipt_refs',
        'provider_receipt_refs', metric_doc->'provider_receipt_refs',
        'decision_receipt_refs', metric_doc->'decision_receipt_refs',
        'immutable', TRUE
    ))
);

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_waterfall_receipts_evaluation
    ON public.research_lab_candidate_waterfall_receipts(
        experiment_id,
        variant_id,
        unit_ref,
        step_order,
        attempt_sequence
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_waterfall_metrics_evaluation
    ON public.research_lab_candidate_waterfall_metrics(
        evaluation_receipt_id,
        variant_id,
        split,
        created_at DESC
    );

CREATE OR REPLACE FUNCTION public.prevent_research_lab_candidate_waterfall_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; write a new immutable row', TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_candidate_waterfall_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_candidate_waterfall_mutation()
    TO service_role;

DO $candidate_waterfall_triggers$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_candidate_waterfall_receipts',
        'research_lab_candidate_waterfall_metrics'
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
                'FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_candidate_waterfall_mutation()',
                'trg_' || relation_name || '_no_mutation',
                relation_name
            );
        END IF;
    END LOOP;
END;
$candidate_waterfall_triggers$;

REVOKE ALL ON TABLE
    public.research_lab_candidate_waterfall_receipts,
    public.research_lab_candidate_waterfall_metrics
FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT ON TABLE
    public.research_lab_candidate_waterfall_receipts,
    public.research_lab_candidate_waterfall_metrics
TO service_role;

ALTER TABLE public.research_lab_candidate_waterfall_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_waterfall_receipts FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_waterfall_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_candidate_waterfall_metrics FORCE ROW LEVEL SECURITY;

DO $candidate_waterfall_policies$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_candidate_waterfall_receipts',
        'research_lab_candidate_waterfall_metrics'
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
$candidate_waterfall_policies$;

COMMENT ON TABLE public.research_lab_candidate_waterfall_receipts IS
    'Append-only Model candidate waterfall receipt sidecars linked to shared routing V2 provider and decision receipts.';
COMMENT ON TABLE public.research_lab_candidate_waterfall_metrics IS
    'Append-only candidate yield metrics linked to shared routing V2 evaluations; never a promotion decision.';

COMMIT;
