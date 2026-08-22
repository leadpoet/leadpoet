-- Candidate-specific PostgreSQL sidecars for the shared routing experiment.
--
-- PR 93 owns experiment, budget, provider-receipt, decision, evaluation, and
-- promotion contracts. These tables persist only exact Model waterfall
-- receipt projections and candidate yield metrics that reference those
-- immutable shared documents. They do not create a parallel lifecycle.

BEGIN;

-- Bind the sidecars to the exact shared experiment lineage.  The parent
-- receipt ids are already primary keys, but these composite keys prevent a
-- caller from combining a receipt id from one experiment with another
-- experiment hash.
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_decision_receipt_experiment_uq
    ON public.research_lab_routing_decision_receipts_v2(
        receipt_id,
        experiment_hash
    );

CREATE UNIQUE INDEX IF NOT EXISTS rl_route_evaluation_receipt_experiment_uq
    ON public.research_lab_routing_evaluation_receipts_v2(
        receipt_id,
        experiment_hash
    );

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
    experiment_hash               TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash)
        CHECK (
        experiment_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    variant_id                    TEXT NOT NULL,
    artifact_key                  TEXT NOT NULL CHECK (
        artifact_key ~ '^sha256:[0-9a-f]{64}$'
    ),
    decision_receipt_id           TEXT NOT NULL,
    FOREIGN KEY (decision_receipt_id, experiment_hash)
        REFERENCES public.research_lab_routing_decision_receipts_v2(
            receipt_id,
            experiment_hash
        )
        DEFERRABLE INITIALLY IMMEDIATE,
    CHECK (
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
    prior_attempt_receipt_sha256  TEXT NOT NULL DEFAULT '' CHECK (
        prior_attempt_receipt_sha256 = ''
        OR prior_attempt_receipt_sha256 ~ '^[0-9a-f]{64}$'
    ),
    attempt_chain_sha256          TEXT NOT NULL CHECK (
        attempt_chain_sha256 ~ '^[0-9a-f]{64}$'
    ),
    verification_receipt_sha256   TEXT NOT NULL DEFAULT '' CHECK (
        verification_receipt_sha256 = ''
        OR verification_receipt_sha256 ~ '^[0-9a-f]{64}$'
    ),
    company_verification_receipt_sha256s JSONB NOT NULL DEFAULT '[]'::JSONB CHECK (
        jsonb_typeof(company_verification_receipt_sha256s) = 'array'
    ),
    step_order                    INTEGER NOT NULL CHECK (step_order >= 0),
    attempt_sequence              INTEGER NOT NULL CHECK (attempt_sequence >= 0),
    target_verified_qualified_count INTEGER NOT NULL CHECK (
        target_verified_qualified_count >= 1
    ),
    disposition                   TEXT NOT NULL CHECK (
        disposition IN ('succeeded', 'missed', 'failed', 'deferred', 'skipped')
    ),
    outcome_code                  TEXT NOT NULL,
    provider_call_count           INTEGER NOT NULL CHECK (provider_call_count >= 0),
    billed_credit_microunits      BIGINT NOT NULL CHECK (billed_credit_microunits >= 0),
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
    UNIQUE (experiment_hash, variant_id, unit_ref, step_order, attempt_sequence),
    CHECK (raw_count >= normalized_count),
    CHECK (normalized_count >= unique_count),
    CHECK (unique_count >= verified_qualified_count),
    CHECK (verified_qualified_count >= published_count),
    CHECK (verified_qualified_count = 0 OR verification_receipt_sha256 <> ''),
    CHECK (jsonb_typeof(receipt_doc->'company_verification_receipt_sha256s') = 'array'),
    CHECK (
        (
            disposition = 'skipped'
            AND provider_receipt_ref = ''
            AND provider_outcome = 'skipped'
            AND provider_call_count = 0
        )
        OR (
            disposition <> 'skipped'
            AND provider_receipt_ref <> ''
            AND provider_outcome <> 'skipped'
            AND provider_call_count >= 1
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
        'prior_attempt_receipt_sha256', prior_attempt_receipt_sha256,
        'attempt_chain_sha256', attempt_chain_sha256,
        'verification_receipt_sha256', verification_receipt_sha256,
        'company_verification_receipt_sha256s', company_verification_receipt_sha256s,
        'step_order', step_order,
        'attempt_sequence', attempt_sequence,
        'target_verified_qualified_count', target_verified_qualified_count,
        'disposition', disposition,
        'outcome_code', outcome_code,
        'provider_call_count', provider_call_count,
        'billed_credit_microunits', billed_credit_microunits,
        'latency_ms', latency_ms,
        'raw_count', raw_count,
        'normalized_count', normalized_count,
        'unique_count', unique_count,
        'verified_qualified_count', verified_qualified_count,
        'published_count', published_count,
        'immutable', TRUE
    )),
    CHECK (
        receipt_hash = public.research_lab_routing_jsonb_hash_v2(
            receipt_doc - ARRAY['receipt_id', 'receipt_hash']
        )
    ),
    CHECK (
        receipt_id = 'candidate_waterfall:' || pg_catalog.substr(receipt_hash, 8, 24)
    )
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
    experiment_id                   TEXT NOT NULL,
    experiment_hash                 TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash)
        CHECK (
        experiment_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    evaluation_receipt_id           TEXT NOT NULL,
    FOREIGN KEY (evaluation_receipt_id, experiment_hash)
        REFERENCES public.research_lab_routing_evaluation_receipts_v2(
            receipt_id,
            experiment_hash
        )
        DEFERRABLE INITIALLY IMMEDIATE,
    CHECK (
        evaluation_receipt_id ~ '^routing_evaluation_v2:[0-9a-f]{16}$'
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
    total_billed_credit_microunits  BIGINT NOT NULL CHECK (total_billed_credit_microunits >= 0),
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
    verified_qualified_per_credit   DOUBLE PRECISION NOT NULL CHECK (
        verified_qualified_per_credit >= 0
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
        'total_billed_credit_microunits', total_billed_credit_microunits,
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
        'verified_qualified_per_credit', verified_qualified_per_credit,
        'waterfall_receipt_refs', metric_doc->'waterfall_receipt_refs',
        'provider_receipt_refs', metric_doc->'provider_receipt_refs',
        'decision_receipt_refs', metric_doc->'decision_receipt_refs',
        'immutable', TRUE
    )),
    CHECK (
        metric_hash = public.research_lab_routing_jsonb_hash_v2(
            metric_doc - ARRAY['metric_id', 'metric_hash']
        )
    ),
    CHECK (
        metric_id = 'candidate_metric:' || pg_catalog.substr(metric_hash, 8, 24)
    )
);

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_waterfall_receipts_evaluation
    ON public.research_lab_candidate_waterfall_receipts(
        experiment_id,
        variant_id,
        unit_ref,
        step_order,
        attempt_sequence
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_research_lab_candidate_waterfall_provider_receipt
    ON public.research_lab_candidate_waterfall_receipts(
        provider_receipt_ref
    )
    WHERE provider_receipt_ref <> '';

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
FROM PUBLIC, anon, authenticated, service_role;

GRANT SELECT ON TABLE
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
    END LOOP;
END;
$candidate_waterfall_policies$;

-- Candidate sidecars are written only through claim-fenced RPCs. The
-- service role can read these append-only tables but cannot insert directly.
CREATE OR REPLACE FUNCTION public.research_lab_candidate_append_waterfall_receipt_v1(
    p_receipt_id TEXT,
    p_receipt_hash TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_receipt_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $candidate_receipt_v1$
DECLARE
    existing public.research_lab_candidate_waterfall_receipts%ROWTYPE;
    prior public.research_lab_candidate_waterfall_receipts%ROWTYPE;
BEGIN
    IF p_receipt_id !~ '^candidate_waterfall:[0-9a-f]{24}$'
       OR p_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_receipt_doc) IS DISTINCT FROM 'object'
       OR p_receipt_doc->>'receipt_id' IS DISTINCT FROM p_receipt_id
       OR p_receipt_doc->>'receipt_hash' IS DISTINCT FROM p_receipt_hash
       OR p_receipt_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_receipt_hash IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(
            p_receipt_doc - ARRAY['receipt_id', 'receipt_hash']
          )
    THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_receipt_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_receipt_doc, 'candidate waterfall receipt v1'
    );
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    IF p_receipt_doc->>'provider_receipt_ref' <> ''
       AND NOT EXISTS (
           SELECT 1
             FROM public.research_lab_routing_provider_attempts_v2 attempt
            WHERE attempt.experiment_hash = p_experiment_hash
              AND attempt.provider_receipt_ref = p_receipt_doc->>'provider_receipt_ref'
              AND attempt.variant_id = p_receipt_doc->>'variant_id'
              AND attempt.unit_ref = p_receipt_doc->>'unit_ref'
              AND attempt.binding_id = p_receipt_doc->>'binding_id'
              AND attempt.tool_id = p_receipt_doc->>'tool_id'
       )
    THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_provider_not_authoritative'
            USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_candidate_waterfall_receipts
     WHERE receipt_id = p_receipt_id;
    IF FOUND THEN
        IF existing.receipt_hash IS DISTINCT FROM p_receipt_hash
           OR existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.receipt_doc IS DISTINCT FROM p_receipt_doc
        THEN
            RAISE EXCEPTION 'research_lab_candidate_waterfall_receipt_replay_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'receipt_id', p_receipt_id, 'receipt_hash', p_receipt_hash,
            'idempotent', TRUE
        );
    END IF;
    IF (p_receipt_doc->>'step_order')::INTEGER
            IS DISTINCT FROM (p_receipt_doc->>'attempt_sequence')::INTEGER
    THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_attempt_sequence_differs'
            USING ERRCODE = '22023';
    END IF;
    SELECT * INTO prior
      FROM public.research_lab_candidate_waterfall_receipts
     WHERE experiment_hash = p_experiment_hash
       AND variant_id = p_receipt_doc->>'variant_id'
       AND unit_ref = p_receipt_doc->>'unit_ref'
     ORDER BY step_order DESC, attempt_sequence DESC
     LIMIT 1;
    IF NOT FOUND THEN
        IF (p_receipt_doc->>'step_order')::INTEGER <> 0
           OR p_receipt_doc->>'prior_attempt_receipt_sha256' <> ''
        THEN
            RAISE EXCEPTION 'research_lab_candidate_waterfall_attempt_sequence_is_not_contiguous'
                USING ERRCODE = '23514';
        END IF;
    ELSIF (p_receipt_doc->>'step_order')::INTEGER <> prior.step_order + 1
       OR p_receipt_doc->>'prior_attempt_receipt_sha256'
            IS DISTINCT FROM prior.attempt_receipt_sha256
    THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_attempt_sequence_is_not_contiguous'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = p_experiment_hash
           AND receipt.target_verified_qualified_count
                IS DISTINCT FROM (p_receipt_doc->>'target_verified_qualified_count')::INTEGER
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_target_differs'
            USING ERRCODE = '23514';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiments_v2 experiment
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
              experiment.spec_doc->'variants'
          ) variant(value)
         WHERE experiment.experiment_hash = p_experiment_hash
           AND variant.value->>'variant_id' = p_receipt_doc->>'variant_id'
           AND (
               EXISTS (
                   SELECT 1
                     FROM pg_catalog.jsonb_array_elements_text(
                         experiment.spec_doc #> '{input,calibration_unit_refs}'
                     ) unit(value)
                    WHERE unit.value = p_receipt_doc->>'unit_ref'
               )
               OR EXISTS (
                   SELECT 1
                     FROM pg_catalog.jsonb_array_elements_text(
                         experiment.spec_doc #> '{input,holdout_unit_refs}'
                     ) unit(value)
                    WHERE unit.value = p_receipt_doc->>'unit_ref'
               )
           )
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_unit_not_authoritative'
            USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_decision_receipts_v2 decision
         WHERE decision.receipt_id = p_receipt_doc->>'decision_receipt_id'
           AND decision.experiment_hash = p_experiment_hash
           AND decision.variant_id = p_receipt_doc->>'variant_id'
           AND decision.unit_ref = p_receipt_doc->>'unit_ref'
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_decision_not_authoritative'
            USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_candidate_waterfall_receipts (
        receipt_id, receipt_hash, contract_version, experiment_id,
        experiment_hash, variant_id, artifact_key, decision_receipt_id,
        provider_receipt_ref, unit_ref, binding_id, tool_id, execution_mode,
        provider_outcome, decision_plan_hash, decision_route_hash,
        model_contract_sha256, model_plan_sha256, stop_policy_sha256,
        attempt_receipt_sha256, prior_attempt_receipt_sha256,
        attempt_chain_sha256, verification_receipt_sha256,
        company_verification_receipt_sha256s, step_order,
        attempt_sequence, target_verified_qualified_count, disposition,
        outcome_code, provider_call_count, billed_credit_microunits,
        latency_ms, raw_count, normalized_count, unique_count,
        verified_qualified_count, published_count, receipt_doc
    ) VALUES (
        p_receipt_id, p_receipt_hash, p_receipt_doc->>'contract_version',
        p_receipt_doc->>'experiment_id', p_experiment_hash,
        p_receipt_doc->>'variant_id', p_receipt_doc->>'artifact_key',
        p_receipt_doc->>'decision_receipt_id', p_receipt_doc->>'provider_receipt_ref',
        p_receipt_doc->>'unit_ref', p_receipt_doc->>'binding_id',
        p_receipt_doc->>'tool_id', p_receipt_doc->>'execution_mode',
        p_receipt_doc->>'provider_outcome', p_receipt_doc->>'decision_plan_hash',
        p_receipt_doc->>'decision_route_hash', p_receipt_doc->>'model_contract_sha256',
        p_receipt_doc->>'model_plan_sha256', p_receipt_doc->>'stop_policy_sha256',
        p_receipt_doc->>'attempt_receipt_sha256', p_receipt_doc->>'prior_attempt_receipt_sha256',
        p_receipt_doc->>'attempt_chain_sha256', p_receipt_doc->>'verification_receipt_sha256',
        p_receipt_doc->'company_verification_receipt_sha256s',
        (p_receipt_doc->>'step_order')::INTEGER, (p_receipt_doc->>'attempt_sequence')::INTEGER,
        (p_receipt_doc->>'target_verified_qualified_count')::INTEGER,
        p_receipt_doc->>'disposition', p_receipt_doc->>'outcome_code',
        (p_receipt_doc->>'provider_call_count')::INTEGER,
        (p_receipt_doc->>'billed_credit_microunits')::BIGINT,
        (p_receipt_doc->>'latency_ms')::BIGINT, (p_receipt_doc->>'raw_count')::INTEGER,
        (p_receipt_doc->>'normalized_count')::INTEGER,
        (p_receipt_doc->>'unique_count')::INTEGER,
        (p_receipt_doc->>'verified_qualified_count')::INTEGER,
        (p_receipt_doc->>'published_count')::INTEGER, p_receipt_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'receipt_id', p_receipt_id, 'receipt_hash', p_receipt_hash,
        'idempotent', FALSE
    );
END;
$candidate_receipt_v1$;

CREATE OR REPLACE FUNCTION public.research_lab_candidate_append_waterfall_metric_v1(
    p_metric_id TEXT,
    p_metric_hash TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_metric_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $candidate_metric_v1$
DECLARE
    existing public.research_lab_candidate_waterfall_metrics%ROWTYPE;
BEGIN
    IF p_metric_id !~ '^candidate_metric:[0-9a-f]{24}$'
       OR p_metric_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_metric_doc) IS DISTINCT FROM 'object'
       OR p_metric_doc->>'metric_id' IS DISTINCT FROM p_metric_id
       OR p_metric_doc->>'metric_hash' IS DISTINCT FROM p_metric_hash
       OR p_metric_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR pg_catalog.jsonb_typeof(p_metric_doc->'waterfall_receipt_refs') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(p_metric_doc->'provider_receipt_refs') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(p_metric_doc->'decision_receipt_refs') IS DISTINCT FROM 'array'
       OR p_metric_hash IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(
            p_metric_doc - ARRAY['metric_id', 'metric_hash']
          )
    THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_metric_doc, 'candidate waterfall metric v1'
    );
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
         WHERE evaluation.receipt_id = p_metric_doc->>'evaluation_receipt_id'
           AND evaluation.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_evaluation_not_authoritative'
            USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = p_experiment_hash
           AND receipt.target_verified_qualified_count
                IS DISTINCT FROM (p_metric_doc->>'target_verified_qualified_count')::INTEGER
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_target_differs'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.jsonb_array_elements_text(
              p_metric_doc->'waterfall_receipt_refs'
          ) refs(value)
         GROUP BY refs.value
        HAVING pg_catalog.count(*) > 1
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.jsonb_array_elements_text(
              p_metric_doc->'provider_receipt_refs'
          ) refs(value)
         GROUP BY refs.value
        HAVING pg_catalog.count(*) > 1
    ) OR EXISTS (
        SELECT 1
          FROM pg_catalog.jsonb_array_elements_text(
              p_metric_doc->'decision_receipt_refs'
          ) refs(value)
         GROUP BY refs.value
        HAVING pg_catalog.count(*) > 1
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_references_duplicated'
            USING ERRCODE = '22023';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.jsonb_array_elements_text(
              p_metric_doc->'waterfall_receipt_refs'
          ) refs(value)
         WHERE NOT EXISTS (
             SELECT 1
               FROM public.research_lab_candidate_waterfall_receipts receipt
              WHERE receipt.receipt_id = refs.value
                AND receipt.experiment_hash = p_experiment_hash
                AND receipt.variant_id = p_metric_doc->>'variant_id'
         )
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_receipt_coverage_missing'
            USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_candidate_waterfall_metrics
     WHERE metric_id = p_metric_id;
    IF FOUND THEN
        IF existing.metric_hash IS DISTINCT FROM p_metric_hash
           OR existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.metric_doc IS DISTINCT FROM p_metric_doc
        THEN
            RAISE EXCEPTION 'research_lab_candidate_waterfall_metric_replay_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'metric_id', p_metric_id, 'metric_hash', p_metric_hash,
            'idempotent', TRUE
        );
    END IF;
    INSERT INTO public.research_lab_candidate_waterfall_metrics (
        metric_id, metric_hash, contract_version, experiment_id,
        experiment_hash, evaluation_receipt_id, variant_id, split,
        target_verified_qualified_count, unit_count, fulfilled_unit_count,
        waterfall_attempt_count, provider_call_count,
        total_billed_credit_microunits, total_latency_ms, raw_count,
        normalized_count, unique_count, verified_qualified_count,
        published_count, failed_attempt_count, missed_attempt_count,
        fulfillment_rate, verification_rate, publication_rate,
        verified_qualified_per_credit, metric_doc
    ) VALUES (
        p_metric_id, p_metric_hash, p_metric_doc->>'contract_version',
        p_metric_doc->>'experiment_id', p_experiment_hash,
        p_metric_doc->>'evaluation_receipt_id', p_metric_doc->>'variant_id',
        p_metric_doc->>'split', (p_metric_doc->>'target_verified_qualified_count')::INTEGER,
        (p_metric_doc->>'unit_count')::INTEGER, (p_metric_doc->>'fulfilled_unit_count')::INTEGER,
        (p_metric_doc->>'waterfall_attempt_count')::INTEGER,
        (p_metric_doc->>'provider_call_count')::INTEGER,
        (p_metric_doc->>'total_billed_credit_microunits')::BIGINT,
        (p_metric_doc->>'total_latency_ms')::BIGINT, (p_metric_doc->>'raw_count')::INTEGER,
        (p_metric_doc->>'normalized_count')::INTEGER,
        (p_metric_doc->>'unique_count')::INTEGER,
        (p_metric_doc->>'verified_qualified_count')::INTEGER,
        (p_metric_doc->>'published_count')::INTEGER,
        (p_metric_doc->>'failed_attempt_count')::INTEGER,
        (p_metric_doc->>'missed_attempt_count')::INTEGER,
        (p_metric_doc->>'fulfillment_rate')::DOUBLE PRECISION,
        (p_metric_doc->>'verification_rate')::DOUBLE PRECISION,
        (p_metric_doc->>'publication_rate')::DOUBLE PRECISION,
        (p_metric_doc->>'verified_qualified_per_credit')::DOUBLE PRECISION,
        p_metric_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'metric_id', p_metric_id, 'metric_hash', p_metric_hash,
        'idempotent', FALSE
    );
END;
$candidate_metric_v1$;

-- Promotion is a separate SQL authority from reconciliation.  Guard the
-- shared promoted event as well, so a caller cannot bypass the sidecar proof
-- by presenting a hand-built reconciliation document.
CREATE OR REPLACE FUNCTION public.research_lab_candidate_promotion_sidecars_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $candidate_promotion_guard_v1$
DECLARE
    experiment_doc JSONB;
    evaluation_doc JSONB;
    metric_row RECORD;
    expected_refs JSONB;
    expected_provider_refs JSONB;
    expected_decision_refs JSONB;
    target_count INTEGER;
BEGIN
    IF NEW.event_type IS DISTINCT FROM 'promoted' THEN
        RETURN NEW;
    END IF;
    SELECT experiment.spec_doc INTO experiment_doc
      FROM public.research_lab_routing_experiments_v2 experiment
     WHERE experiment.experiment_hash = NEW.experiment_hash;
    IF experiment_doc IS NULL
       OR experiment_doc #>> '{input,stage}' IS DISTINCT FROM 'candidate_acquisition'
    THEN
        RETURN NEW;
    END IF;
    SELECT evaluation.evaluation_doc INTO evaluation_doc
      FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
     WHERE evaluation.experiment_hash = NEW.experiment_hash
       AND evaluation.receipt_id = NEW.event_doc->>'evaluation_receipt_id';
    IF evaluation_doc IS NULL THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_evaluation_missing'
            USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
    ) OR NOT EXISTS (
        SELECT 1 FROM public.research_lab_candidate_waterfall_metrics metric
         WHERE metric.experiment_hash = NEW.experiment_hash
           AND metric.evaluation_receipt_id = NEW.event_doc->>'evaluation_receipt_id'
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_sidecars_missing'
            USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_routing_provider_attempts_v2 attempt
         WHERE attempt.experiment_hash = NEW.experiment_hash
           AND attempt.tool_id LIKE 'candidate.%'
        EXCEPT
        SELECT receipt.provider_receipt_ref
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND receipt.provider_receipt_ref <> ''
    ) OR EXISTS (
        SELECT receipt.provider_receipt_ref
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND receipt.provider_receipt_ref <> ''
        EXCEPT
        SELECT attempt.provider_receipt_ref
          FROM public.research_lab_routing_provider_attempts_v2 attempt
         WHERE attempt.experiment_hash = NEW.experiment_hash
           AND attempt.tool_id LIKE 'candidate.%'
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_provider_sidecars_incomplete'
            USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND receipt.step_order <> receipt.attempt_sequence
    ) OR EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
         GROUP BY receipt.variant_id, receipt.unit_ref, receipt.target_verified_qualified_count
        HAVING min(receipt.step_order) <> 0
            OR max(receipt.step_order) <> count(*) - 1
            OR count(DISTINCT receipt.step_order) <> count(* )
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_attempt_sequence_incomplete'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND NOT (
               receipt.unit_ref IN (
                   SELECT value FROM pg_catalog.jsonb_array_elements_text(
                       experiment_doc #> '{input,calibration_unit_refs}'
                   ) units(value)
               )
               OR receipt.unit_ref IN (
                   SELECT value FROM pg_catalog.jsonb_array_elements_text(
                       experiment_doc #> '{input,holdout_unit_refs}'
                   ) units(value)
               )
           )
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_unit_coverage_incomplete'
            USING ERRCODE = '23503';
    END IF;
    SELECT min(receipt.target_verified_qualified_count)
      INTO target_count
      FROM public.research_lab_candidate_waterfall_receipts receipt
     WHERE receipt.experiment_hash = NEW.experiment_hash;
    IF target_count IS NULL OR EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND receipt.target_verified_qualified_count <> target_count
    ) OR EXISTS (
        SELECT 1
          FROM public.research_lab_candidate_waterfall_metrics metric
         WHERE metric.experiment_hash = NEW.experiment_hash
           AND metric.evaluation_receipt_id = NEW.event_doc->>'evaluation_receipt_id'
           AND metric.target_verified_qualified_count <> target_count
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_target_incomplete'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT variant.value->>'variant_id', split_name.split
          FROM pg_catalog.jsonb_array_elements(experiment_doc->'variants') variant(value)
          CROSS JOIN (VALUES ('calibration'), ('holdout')) split_name(split)
        EXCEPT
        SELECT metric.variant_id, metric.split
          FROM public.research_lab_candidate_waterfall_metrics metric
         WHERE metric.experiment_hash = NEW.experiment_hash
           AND metric.evaluation_receipt_id = NEW.event_doc->>'evaluation_receipt_id'
    ) OR EXISTS (
        SELECT metric.variant_id, metric.split
          FROM public.research_lab_candidate_waterfall_metrics metric
         WHERE metric.experiment_hash = NEW.experiment_hash
           AND metric.evaluation_receipt_id = NEW.event_doc->>'evaluation_receipt_id'
        EXCEPT
        SELECT variant.value->>'variant_id', split_name.split
          FROM pg_catalog.jsonb_array_elements(experiment_doc->'variants') variant(value)
          CROSS JOIN (VALUES ('calibration'), ('holdout')) split_name(split)
    ) THEN
        RAISE EXCEPTION 'research_lab_candidate_promotion_metric_coverage_incomplete'
            USING ERRCODE = '23503';
    END IF;
    FOR metric_row IN
        SELECT metric.*
          FROM public.research_lab_candidate_waterfall_metrics metric
         WHERE metric.experiment_hash = NEW.experiment_hash
           AND metric.evaluation_receipt_id = NEW.event_doc->>'evaluation_receipt_id'
    LOOP
        SELECT coalesce(
            pg_catalog.jsonb_agg(to_jsonb(receipt.receipt_id)
                ORDER BY receipt.unit_ref, receipt.step_order, receipt.attempt_sequence),
            '[]'::JSONB
        ) INTO expected_refs
          FROM public.research_lab_candidate_waterfall_receipts receipt
         WHERE receipt.experiment_hash = NEW.experiment_hash
           AND receipt.variant_id = metric_row.variant_id
           AND receipt.unit_ref IN (
               SELECT value
                 FROM pg_catalog.jsonb_array_elements_text(
                     CASE metric_row.split
                         WHEN 'calibration' THEN experiment_doc #> '{input,calibration_unit_refs}'
                         ELSE experiment_doc #> '{input,holdout_unit_refs}'
                     END
                 ) values(value)
           );
        SELECT coalesce(pg_catalog.jsonb_agg(to_jsonb(ref) ORDER BY ref), '[]'::JSONB)
          INTO expected_provider_refs
          FROM (
              SELECT DISTINCT receipt.provider_receipt_ref AS ref
                FROM public.research_lab_candidate_waterfall_receipts receipt
               WHERE receipt.experiment_hash = NEW.experiment_hash
                 AND receipt.variant_id = metric_row.variant_id
                 AND receipt.unit_ref IN (
                     SELECT value FROM pg_catalog.jsonb_array_elements_text(
                         CASE metric_row.split
                             WHEN 'calibration' THEN experiment_doc #> '{input,calibration_unit_refs}'
                             ELSE experiment_doc #> '{input,holdout_unit_refs}'
                         END
                     ) values(value)
                 )
                 AND receipt.provider_receipt_ref <> ''
          ) refs;
        SELECT coalesce(pg_catalog.jsonb_agg(to_jsonb(ref) ORDER BY ref), '[]'::JSONB)
          INTO expected_decision_refs
          FROM (
              SELECT DISTINCT receipt.decision_receipt_id AS ref
                FROM public.research_lab_candidate_waterfall_receipts receipt
               WHERE receipt.experiment_hash = NEW.experiment_hash
                 AND receipt.variant_id = metric_row.variant_id
                 AND receipt.unit_ref IN (
                     SELECT value FROM pg_catalog.jsonb_array_elements_text(
                         CASE metric_row.split
                             WHEN 'calibration' THEN experiment_doc #> '{input,calibration_unit_refs}'
                             ELSE experiment_doc #> '{input,holdout_unit_refs}'
                         END
                     ) values(value)
                 )
          ) refs;
        IF metric_row.metric_doc->'waterfall_receipt_refs' IS DISTINCT FROM expected_refs
           OR metric_row.metric_doc->'provider_receipt_refs' IS DISTINCT FROM expected_provider_refs
           OR metric_row.metric_doc->'decision_receipt_refs' IS DISTINCT FROM expected_decision_refs
        THEN
            RAISE EXCEPTION 'research_lab_candidate_promotion_metric_receipts_incomplete'
                USING ERRCODE = '23503';
        END IF;
    END LOOP;
    RETURN NEW;
END;
$candidate_promotion_guard_v1$;

REVOKE ALL ON FUNCTION public.research_lab_candidate_promotion_sidecars_guard_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_candidate_promotion_sidecars_guard_v1()
    TO service_role;

DO $candidate_promotion_trigger$
BEGIN
    IF pg_catalog.to_regclass(
        'public.research_lab_routing_experiment_events_v2'
    ) IS NOT NULL AND NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_trigger trigger_meta
          JOIN pg_catalog.pg_class relation_meta
            ON relation_meta.oid = trigger_meta.tgrelid
          JOIN pg_catalog.pg_namespace namespace_meta
            ON namespace_meta.oid = relation_meta.relnamespace
         WHERE namespace_meta.nspname = 'public'
           AND relation_meta.relname = 'research_lab_routing_experiment_events_v2'
           AND trigger_meta.tgname = 'trg_research_lab_candidate_promotion_sidecars_guard'
           AND NOT trigger_meta.tgisinternal
    ) THEN
        EXECUTE 'CREATE TRIGGER trg_research_lab_candidate_promotion_sidecars_guard '
            'BEFORE INSERT ON public.research_lab_routing_experiment_events_v2 '
            'FOR EACH ROW EXECUTE FUNCTION '
            'public.research_lab_candidate_promotion_sidecars_guard_v1()';
    END IF;
END;
$candidate_promotion_trigger$;

REVOKE ALL ON FUNCTION public.research_lab_candidate_append_waterfall_receipt_v1(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_candidate_append_waterfall_receipt_v1(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_candidate_append_waterfall_metric_v1(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_candidate_append_waterfall_metric_v1(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) TO service_role;

COMMENT ON TABLE public.research_lab_candidate_waterfall_receipts IS
    'Append-only Model candidate waterfall receipt sidecars linked to shared routing V2 provider and decision receipts.';
COMMENT ON TABLE public.research_lab_candidate_waterfall_metrics IS
    'Append-only candidate yield metrics linked to shared routing V2 evaluations; never a promotion decision.';

COMMIT;
