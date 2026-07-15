-- Conditional 40-ICP champion validation (apply after migrations 30, 37, 69,
-- and 83). Additive only: historical 1.0 score bundles and 20-ICP queue rows
-- remain immutable and readable.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_evaluation_score_bundles
    DROP CONSTRAINT IF EXISTS research_evaluation_score_bundles_schema_version_check;
ALTER TABLE public.research_evaluation_score_bundles
    ADD CONSTRAINT research_evaluation_score_bundles_schema_version_check
    CHECK (schema_version IN ('1.0', '1.1')) NOT VALID;
ALTER TABLE public.research_evaluation_score_bundles
    VALIDATE CONSTRAINT research_evaluation_score_bundles_schema_version_check;

ALTER TABLE public.research_lab_private_model_benchmark_bundles
    DROP CONSTRAINT IF EXISTS research_lab_private_model_benchmark_bundles_schema_version_check;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    ADD CONSTRAINT research_lab_private_model_benchmark_bundles_schema_version_check
    CHECK (schema_version IN ('1.0', '1.1')) NOT VALID;
ALTER TABLE public.research_lab_private_model_benchmark_bundles
    VALIDATE CONSTRAINT research_lab_private_model_benchmark_bundles_schema_version_check;

ALTER TABLE public.research_lab_scoring_job_queue
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_queue_phase_check;
ALTER TABLE public.research_lab_scoring_job_queue
    ADD CONSTRAINT research_lab_scoring_job_queue_phase_check
    CHECK (phase IN ('public', 'private', 'conditional')) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_queue
    VALIDATE CONSTRAINT research_lab_scoring_job_queue_phase_check;

ALTER TABLE public.research_lab_scoring_job_candidate
    ADD COLUMN IF NOT EXISTS conditional_total INTEGER NOT NULL DEFAULT 0
        CHECK (conditional_total >= 0),
    ADD COLUMN IF NOT EXISTS baseline_preliminary_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS threshold_points DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (threshold_points >= 0),
    ADD COLUMN IF NOT EXISTS preliminary_gate_status TEXT NOT NULL DEFAULT 'not_required',
    ADD COLUMN IF NOT EXISTS baseline_benchmark_bundle_id TEXT,
    ADD COLUMN IF NOT EXISTS baseline_benchmark_hash TEXT,
    ADD COLUMN IF NOT EXISTS category_assignment_hash TEXT,
    ADD COLUMN IF NOT EXISTS conditional_policy_hash TEXT,
    ADD COLUMN IF NOT EXISTS candidate_artifact_hash TEXT,
    ADD COLUMN IF NOT EXISTS candidate_parent_artifact_hash TEXT,
    ADD COLUMN IF NOT EXISTS scoring_configuration_hash TEXT,
    ADD COLUMN IF NOT EXISTS preliminary_gate_proof JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(preliminary_gate_proof) = 'object'
        AND preliminary_gate_proof::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|intent_signals|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
    ),
    ADD COLUMN IF NOT EXISTS preliminary_gate_claimed_by TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS preliminary_gate_lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS preliminary_gate_attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (preliminary_gate_attempt_count >= 0);

ALTER TABLE public.research_lab_scoring_job_candidate
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_candidate_conditional_commitment_check;
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD CONSTRAINT research_lab_scoring_job_candidate_conditional_commitment_check CHECK (
        conditional_total = 0
        OR (
            baseline_benchmark_bundle_id IS NOT NULL
            AND baseline_benchmark_hash ~ '^sha256:[0-9a-f]{64}$'
            AND category_assignment_hash ~ '^sha256:[0-9a-f]{64}$'
            AND conditional_policy_hash ~ '^sha256:[0-9a-f]{64}$'
            AND candidate_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
            AND candidate_parent_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
            AND scoring_configuration_hash ~ '^sha256:[0-9a-f]{64}$'
            AND baseline_public_score >= 0
            AND baseline_public_score <= 100
            AND baseline_preliminary_score >= 0
            AND baseline_preliminary_score <= 100
            AND threshold_points >= 0
            AND threshold_points <= 100
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_candidate
    VALIDATE CONSTRAINT research_lab_scoring_job_candidate_conditional_commitment_check;

ALTER TABLE public.research_lab_scoring_job_candidate
    DROP CONSTRAINT IF EXISTS research_lab_scoring_job_candidate_preliminary_gate_status_check;
ALTER TABLE public.research_lab_scoring_job_candidate
    ADD CONSTRAINT research_lab_scoring_job_candidate_preliminary_gate_status_check
    CHECK (
        preliminary_gate_status IN (
            'not_required',
            'pending',
            'deciding',
            'passed',
            'rejected',
            'skipped'
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_scoring_job_candidate
    VALIDATE CONSTRAINT research_lab_scoring_job_candidate_preliminary_gate_status_check;

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_job_candidate_preliminary_gate
    ON public.research_lab_scoring_job_candidate(
        preliminary_gate_status,
        preliminary_gate_lease_expires_at,
        updated_at
    )
    WHERE conditional_total > 0
      AND preliminary_gate_status IN ('pending', 'deciding')
      AND assembly_status = 'pending';

ALTER TABLE public.research_lab_scoring_icp_executions
    DROP CONSTRAINT IF EXISTS research_lab_scoring_icp_executions_phase_check;
ALTER TABLE public.research_lab_scoring_icp_executions
    ADD CONSTRAINT research_lab_scoring_icp_executions_phase_check
    CHECK (phase IN ('all', 'public', 'private', 'conditional')) NOT VALID;
ALTER TABLE public.research_lab_scoring_icp_executions
    VALIDATE CONSTRAINT research_lab_scoring_icp_executions_phase_check;

CREATE TABLE IF NOT EXISTS public.research_lab_conditional_validation_events (
    event_id                     UUID        PRIMARY KEY,
    schema_version               TEXT        NOT NULL DEFAULT '1.1'
                                            CHECK (schema_version = '1.1'),
    candidate_id                 TEXT        NOT NULL
                                            REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                            ON DELETE RESTRICT,
    queue_generation_id          UUID        REFERENCES public.research_lab_scoring_job_candidate(queue_generation_id)
                                            ON DELETE RESTRICT,
    event_type                   TEXT        NOT NULL CHECK (
                                            event_type IN (
                                                'preliminary_gate_passed',
                                                'preliminary_gate_failed',
                                                'conditional_started',
                                                'retryable_failure',
                                                'conditional_completed',
                                                'final_pass',
                                                'final_fail'
                                            )
                                        ),
    assignment_hash              TEXT        NOT NULL
                                            CHECK (assignment_hash ~ '^sha256:[0-9a-f]{64}$'),
    policy_hash                  TEXT        NOT NULL
                                            CHECK (policy_hash ~ '^sha256:[0-9a-f]{64}$'),
    rolling_window_hash          TEXT        NOT NULL
                                            CHECK (rolling_window_hash ~ '^sha256:[0-9a-f]{64}$'),
    baseline_benchmark_bundle_id TEXT        NOT NULL,
    source_score_bundle_id       TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                            ON DELETE RESTRICT,
    source_ref                   TEXT        NOT NULL CHECK (LENGTH(source_ref) BETWEEN 1 AND 500),
    decision_score               DOUBLE PRECISION CHECK (
                                            decision_score IS NULL
                                            OR (
                                                decision_score >= 0
                                                AND decision_score <= 100
                                                AND decision_score <> 'NaN'::DOUBLE PRECISION
                                            )
                                        ),
    threshold_points             DOUBLE PRECISION CHECK (
                                            threshold_points IS NULL
                                            OR (
                                                threshold_points >= 0
                                                AND threshold_points <= 100
                                                AND threshold_points <> 'NaN'::DOUBLE PRECISION
                                            )
                                        ),
    failure_class                TEXT,
    event_doc                    JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(event_doc) = 'object'
                                            AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|intent_signals|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                        ),
    event_hash                   TEXT        NOT NULL UNIQUE
                                            CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS research_lab_conditional_validation_terminal_key
    ON public.research_lab_conditional_validation_events(
        candidate_id,
        event_type,
        assignment_hash
    )
    WHERE event_type <> 'retryable_failure';
CREATE UNIQUE INDEX IF NOT EXISTS research_lab_conditional_validation_source_key
    ON public.research_lab_conditional_validation_events(
        candidate_id,
        event_type,
        assignment_hash,
        source_ref
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_conditional_validation_candidate
    ON public.research_lab_conditional_validation_events(candidate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_conditional_validation_queue
    ON public.research_lab_conditional_validation_events(queue_generation_id, created_at)
    WHERE queue_generation_id IS NOT NULL;

DROP TRIGGER IF EXISTS prevent_research_lab_conditional_validation_events_mutation
    ON public.research_lab_conditional_validation_events;
CREATE TRIGGER prevent_research_lab_conditional_validation_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_conditional_validation_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_conditional_validation_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_conditional_validation_events TO service_role;
ALTER TABLE public.research_lab_conditional_validation_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS service_role_read ON public.research_lab_conditional_validation_events;
CREATE POLICY service_role_read ON public.research_lab_conditional_validation_events
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_conditional_validation_events;
CREATE POLICY service_role_insert ON public.research_lab_conditional_validation_events
    FOR INSERT TO service_role WITH CHECK (true);

CREATE OR REPLACE FUNCTION public.research_lab_conditional_validation_event_hash(
    target_candidate_id TEXT,
    target_event_type TEXT,
    target_assignment_hash TEXT,
    target_source_ref TEXT,
    target_event_doc JSONB
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT 'sha256:' || pg_catalog.encode(
        pg_catalog.sha256(
            pg_catalog.convert_to(
                pg_catalog.jsonb_build_object(
                    'candidate_id', target_candidate_id,
                    'event_type', target_event_type,
                    'assignment_hash', target_assignment_hash,
                    'source_ref', target_source_ref,
                    'event_doc', target_event_doc
                )::TEXT,
                'UTF8'
            )
        ),
        'hex'
    );
$$;

REVOKE ALL ON FUNCTION public.research_lab_conditional_validation_event_hash(TEXT, TEXT, TEXT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_conditional_validation_event_hash(TEXT, TEXT, TEXT, TEXT, JSONB)
    TO service_role;

CREATE OR REPLACE FUNCTION public.research_lab_decide_conditional_public_gate(
    target_queue_generation_id UUID,
    candidate_public_score DOUBLE PRECISION
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    passed BOOLEAN;
    phase_total BIGINT;
    phase_done BIGINT;
BEGIN
    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = target_queue_generation_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('decision', 'not_found');
    END IF;
    IF candidate_row.conditional_total <= 0 THEN
        RAISE EXCEPTION 'conditional public gate requires conditional jobs'
            USING ERRCODE = '23514';
    END IF;
    IF candidate_row.gate_status <> 'pending' THEN
        RETURN jsonb_build_object('decision', 'already_decided');
    END IF;
    IF candidate_public_score < 0
       OR candidate_public_score > 100
       OR candidate_public_score = 'NaN'::DOUBLE PRECISION
    THEN
        RAISE EXCEPTION 'conditional public gate score must be finite and within 0-100'
            USING ERRCODE = '22003';
    END IF;
    SELECT COUNT(*), COUNT(*) FILTER (WHERE status = 'done')
      INTO phase_total, phase_done
      FROM public.research_lab_scoring_job_queue
     WHERE queue_generation_id = target_queue_generation_id
       AND phase = 'public';
    IF phase_total <> candidate_row.public_total
       OR phase_done <> candidate_row.public_total
    THEN
        RETURN jsonb_build_object('decision', 'not_ready');
    END IF;

    passed := candidate_public_score + 0.000000001
        >= candidate_row.baseline_public_score;
    UPDATE public.research_lab_scoring_job_queue
       SET status = CASE WHEN passed THEN 'queued' ELSE 'failed' END,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND phase = 'private'
       AND status = 'held';
    IF NOT passed THEN
        UPDATE public.research_lab_scoring_job_queue
           SET status = 'failed', updated_at = NOW()
         WHERE queue_generation_id = target_queue_generation_id
           AND phase = 'conditional'
           AND status = 'held';
    END IF;
    UPDATE public.research_lab_scoring_job_candidate
       SET gate_status = CASE WHEN passed THEN 'passed' ELSE 'rejected' END,
           preliminary_gate_status = CASE
               WHEN passed THEN preliminary_gate_status
               ELSE 'skipped'
           END,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id;
    RETURN jsonb_build_object(
        'decision', CASE WHEN passed THEN 'passed' ELSE 'rejected' END
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_claim_conditional_preliminary_gate(
    target_queue_generation_id UUID,
    target_worker_ref TEXT,
    target_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    public_total BIGINT;
    public_done BIGINT;
    private_total BIGINT;
    private_done BIGINT;
BEGIN
    IF NULLIF(pg_catalog.btrim(target_worker_ref), '') IS NULL THEN
        RAISE EXCEPTION 'conditional preliminary gate worker is required'
            USING ERRCODE = '22023';
    END IF;
    IF target_lease_seconds < 30 OR target_lease_seconds > 3600 THEN
        RAISE EXCEPTION 'conditional preliminary gate lease must be within 30-3600 seconds'
            USING ERRCODE = '22023';
    END IF;

    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = target_queue_generation_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'not_found');
    END IF;
    IF candidate_row.conditional_total <= 0
       OR candidate_row.gate_status <> 'passed'
       OR candidate_row.assembly_status <> 'pending'
    THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'not_eligible');
    END IF;
    IF candidate_row.preliminary_gate_status IN ('passed', 'rejected', 'skipped') THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'already_decided');
    END IF;
    IF candidate_row.preliminary_gate_status = 'deciding'
       AND candidate_row.preliminary_gate_lease_expires_at > NOW()
    THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'busy');
    END IF;

    SELECT
        COUNT(*) FILTER (WHERE phase = 'public'),
        COUNT(*) FILTER (WHERE phase = 'public' AND status = 'done'),
        COUNT(*) FILTER (WHERE phase = 'private'),
        COUNT(*) FILTER (WHERE phase = 'private' AND status = 'done')
      INTO public_total, public_done, private_total, private_done
      FROM public.research_lab_scoring_job_queue
     WHERE queue_generation_id = target_queue_generation_id
       AND phase IN ('public', 'private');
    IF public_total <> candidate_row.public_total
       OR public_done <> candidate_row.public_total
       OR private_total <> candidate_row.private_total
       OR private_done <> candidate_row.private_total
    THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'not_ready');
    END IF;

    UPDATE public.research_lab_scoring_job_candidate
       SET preliminary_gate_status = 'deciding',
           preliminary_gate_claimed_by = target_worker_ref,
           preliminary_gate_lease_expires_at = NOW()
               + pg_catalog.make_interval(secs => target_lease_seconds),
           preliminary_gate_attempt_count = preliminary_gate_attempt_count + 1,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
     RETURNING * INTO candidate_row;

    RETURN pg_catalog.jsonb_build_object(
        'decision', 'claimed',
        'claim', pg_catalog.to_jsonb(candidate_row) - 'preliminary_gate_proof'
    );
END;
$$;

DROP FUNCTION IF EXISTS public.research_lab_decide_conditional_preliminary_gate(
    UUID,
    DOUBLE PRECISION
);

CREATE OR REPLACE FUNCTION public.research_lab_decide_conditional_preliminary_gate(
    target_queue_generation_id UUID,
    candidate_preliminary_score DOUBLE PRECISION,
    target_preliminary_proof JSONB,
    expected_claimed_by TEXT,
    expected_attempt_count INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    passed BOOLEAN;
    public_total BIGINT;
    public_done BIGINT;
    private_total BIGINT;
    private_done BIGINT;
    lifecycle_event_type TEXT;
    lifecycle_source_ref TEXT;
    lifecycle_event_doc JSONB;
BEGIN
    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = target_queue_generation_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('decision', 'not_found');
    END IF;
    IF candidate_row.gate_status <> 'passed' THEN
        RETURN jsonb_build_object('decision', 'already_decided');
    END IF;
    IF candidate_row.preliminary_gate_status <> 'deciding'
       OR candidate_row.preliminary_gate_claimed_by <> expected_claimed_by
       OR candidate_row.preliminary_gate_attempt_count <> expected_attempt_count
       OR candidate_row.preliminary_gate_lease_expires_at IS NULL
       OR candidate_row.preliminary_gate_lease_expires_at <= NOW()
    THEN
        RETURN jsonb_build_object('decision', 'claim_changed');
    END IF;
    IF candidate_preliminary_score < 0
       OR candidate_preliminary_score > 100
       OR candidate_preliminary_score = 'NaN'::DOUBLE PRECISION
    THEN
        RAISE EXCEPTION 'conditional preliminary score must be finite and within 0-100'
            USING ERRCODE = '22003';
    END IF;
    SELECT
        COUNT(*) FILTER (WHERE phase = 'public'),
        COUNT(*) FILTER (WHERE phase = 'public' AND status = 'done'),
        COUNT(*) FILTER (WHERE phase = 'private'),
        COUNT(*) FILTER (WHERE phase = 'private' AND status = 'done')
      INTO public_total, public_done, private_total, private_done
      FROM public.research_lab_scoring_job_queue
     WHERE queue_generation_id = target_queue_generation_id
       AND phase IN ('public', 'private');
    IF public_total <> candidate_row.public_total
       OR public_done <> candidate_row.public_total
       OR private_total <> candidate_row.private_total
       OR private_done <> candidate_row.private_total
    THEN
        RETURN jsonb_build_object('decision', 'not_ready');
    END IF;

    passed := candidate_preliminary_score
        - candidate_row.baseline_preliminary_score
        + 0.000000001 >= candidate_row.threshold_points;
    IF target_preliminary_proof IS NULL
       OR pg_catalog.jsonb_typeof(target_preliminary_proof) <> 'object'
    THEN
        RAISE EXCEPTION 'conditional preliminary proof must be a JSON object'
            USING ERRCODE = '22023';
    END IF;
    IF passed THEN
        IF target_preliminary_proof->>'schema_version'
               IS DISTINCT FROM 'research_lab_preliminary_promotion_gate.v1'
           OR target_preliminary_proof->>'status'
               IS DISTINCT FROM 'promotion_passed'
           OR target_preliminary_proof->>'candidate_artifact_hash'
               IS DISTINCT FROM candidate_row.candidate_artifact_hash
           OR target_preliminary_proof->>'candidate_parent_artifact_hash'
               IS DISTINCT FROM candidate_row.candidate_parent_artifact_hash
           OR target_preliminary_proof->>'active_parent_artifact_hash'
               IS DISTINCT FROM candidate_row.candidate_parent_artifact_hash
           OR target_preliminary_proof->>'rolling_window_hash'
               IS DISTINCT FROM candidate_row.window_hash
           OR target_preliminary_proof->>'category_assignment_hash'
               IS DISTINCT FROM candidate_row.category_assignment_hash
           OR target_preliminary_proof->>'conditional_validation_policy_hash'
               IS DISTINCT FROM candidate_row.conditional_policy_hash
           OR target_preliminary_proof->>'scoring_configuration_hash'
               IS DISTINCT FROM candidate_row.scoring_configuration_hash
           OR ((target_preliminary_proof->>'threshold_points')::DOUBLE PRECISION)
               IS DISTINCT FROM candidate_row.threshold_points
           OR COALESCE(target_preliminary_proof->>'proof_hash', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(target_preliminary_proof->>'preliminary_score_bundle_hash', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(target_preliminary_proof->>'score_bundle_receipt_hash', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(target_preliminary_proof->>'promotion_metric_receipt_hash', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(target_preliminary_proof->>'promotion_decision_receipt_hash', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(target_preliminary_proof->>'promotion_decision_output_root', '')
               !~ '^sha256:[0-9a-f]{64}$'
           OR target_preliminary_proof->'decision'->>'status'
               IS DISTINCT FROM 'promotion_passed'
           OR target_preliminary_proof->'decision'->>'candidate_kind'
               IS DISTINCT FROM 'image_build'
           OR COALESCE(
               (target_preliminary_proof->'decision'->>'auto_promotion_enabled')::BOOLEAN,
               FALSE
           ) IS NOT TRUE
           OR COALESCE(
               (target_preliminary_proof->'decision'->>'active_parent_matches')::BOOLEAN,
               FALSE
           ) IS NOT TRUE
        THEN
            RAISE EXCEPTION 'conditional preliminary attested proof does not match frozen queue commitments'
                USING ERRCODE = '23514';
        END IF;
    ELSIF target_preliminary_proof <> '{}'::JSONB THEN
        RAISE EXCEPTION 'rejected conditional preliminary gate must not carry an authority proof'
            USING ERRCODE = '23514';
    END IF;
    lifecycle_event_type := CASE
        WHEN passed THEN 'preliminary_gate_passed'
        ELSE 'preliminary_gate_failed'
    END;
    lifecycle_source_ref := 'queue:' || target_queue_generation_id::TEXT
        || ':preliminary_attempt:' || expected_attempt_count::TEXT;
    lifecycle_event_doc := pg_catalog.jsonb_build_object(
        'schema_version', '1.1',
        'candidate_preliminary_score', candidate_preliminary_score,
        'baseline_preliminary_score', candidate_row.baseline_preliminary_score,
        'threshold_points', candidate_row.threshold_points,
        'queue_generation_id', target_queue_generation_id,
        'preliminary_gate_attempt_count', expected_attempt_count,
        'preliminary_gate_proof', target_preliminary_proof
    );
    INSERT INTO public.research_lab_conditional_validation_events (
        event_id,
        candidate_id,
        queue_generation_id,
        event_type,
        assignment_hash,
        policy_hash,
        rolling_window_hash,
        baseline_benchmark_bundle_id,
        source_ref,
        decision_score,
        threshold_points,
        event_doc,
        event_hash
    ) VALUES (
        pg_catalog.gen_random_uuid(),
        candidate_row.candidate_id,
        target_queue_generation_id,
        lifecycle_event_type,
        candidate_row.category_assignment_hash,
        candidate_row.conditional_policy_hash,
        candidate_row.window_hash,
        candidate_row.baseline_benchmark_bundle_id,
        lifecycle_source_ref,
        candidate_preliminary_score,
        candidate_row.threshold_points,
        lifecycle_event_doc,
        public.research_lab_conditional_validation_event_hash(
            candidate_row.candidate_id,
            lifecycle_event_type,
            candidate_row.category_assignment_hash,
            lifecycle_source_ref,
            lifecycle_event_doc
        )
    ) ON CONFLICT DO NOTHING;
    IF passed THEN
        lifecycle_event_doc := lifecycle_event_doc || pg_catalog.jsonb_build_object(
            'preliminary_event', 'persisted_before_conditional_release'
        );
        INSERT INTO public.research_lab_conditional_validation_events (
            event_id,
            candidate_id,
            queue_generation_id,
            event_type,
            assignment_hash,
            policy_hash,
            rolling_window_hash,
            baseline_benchmark_bundle_id,
            source_ref,
            decision_score,
            threshold_points,
            event_doc,
            event_hash
        ) VALUES (
            pg_catalog.gen_random_uuid(),
            candidate_row.candidate_id,
            target_queue_generation_id,
            'conditional_started',
            candidate_row.category_assignment_hash,
            candidate_row.conditional_policy_hash,
            candidate_row.window_hash,
            candidate_row.baseline_benchmark_bundle_id,
            lifecycle_source_ref,
            candidate_preliminary_score,
            candidate_row.threshold_points,
            lifecycle_event_doc,
            public.research_lab_conditional_validation_event_hash(
                candidate_row.candidate_id,
                'conditional_started',
                candidate_row.category_assignment_hash,
                lifecycle_source_ref,
                lifecycle_event_doc
            )
        ) ON CONFLICT DO NOTHING;
    END IF;
    UPDATE public.research_lab_scoring_job_queue
       SET status = CASE WHEN passed THEN 'queued' ELSE 'failed' END,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND phase = 'conditional'
       AND status = 'held';
    UPDATE public.research_lab_scoring_job_candidate
       SET preliminary_gate_status = CASE
               WHEN passed THEN 'passed'
               ELSE 'rejected'
           END,
           preliminary_gate_proof = target_preliminary_proof,
           preliminary_gate_claimed_by = '',
           preliminary_gate_lease_expires_at = NULL,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND preliminary_gate_status = 'deciding'
       AND preliminary_gate_claimed_by = expected_claimed_by
       AND preliminary_gate_attempt_count = expected_attempt_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conditional preliminary claim changed inside serialized transition'
            USING ERRCODE = '40001';
    END IF;
    RETURN jsonb_build_object(
        'decision', CASE WHEN passed THEN 'passed' ELSE 'rejected' END
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_cancel_conditional_generation(
    target_queue_generation_id UUID,
    expected_claimed_by TEXT,
    expected_attempt_count INTEGER,
    target_failure_class TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    normalized_failure_class TEXT;
    lifecycle_source_ref TEXT;
    lifecycle_event_doc JSONB;
BEGIN
    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = target_queue_generation_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('committed', FALSE, 'reason', 'not_found');
    END IF;
    IF candidate_row.preliminary_gate_status <> 'deciding'
       OR candidate_row.preliminary_gate_claimed_by <> expected_claimed_by
       OR candidate_row.preliminary_gate_attempt_count <> expected_attempt_count
    THEN
        RETURN pg_catalog.jsonb_build_object('committed', FALSE, 'reason', 'claim_changed');
    END IF;

    normalized_failure_class := COALESCE(
        NULLIF(pg_catalog.btrim(target_failure_class), ''),
        'stale_parent_needs_rescore'
    );
    lifecycle_source_ref := 'queue:' || target_queue_generation_id::TEXT
        || ':preliminary_attempt:' || expected_attempt_count::TEXT
        || ':cancelled';
    lifecycle_event_doc := pg_catalog.jsonb_build_object(
        'schema_version', '1.1',
        'failure_class', normalized_failure_class,
        'attempt_count', expected_attempt_count,
        'queue_generation_id', target_queue_generation_id,
        'conditional_jobs_released', FALSE,
        'stale_generation_closed', TRUE
    );
    INSERT INTO public.research_lab_conditional_validation_events (
        event_id,
        candidate_id,
        queue_generation_id,
        event_type,
        assignment_hash,
        policy_hash,
        rolling_window_hash,
        baseline_benchmark_bundle_id,
        source_ref,
        threshold_points,
        failure_class,
        event_doc,
        event_hash
    ) VALUES (
        pg_catalog.gen_random_uuid(),
        candidate_row.candidate_id,
        target_queue_generation_id,
        'retryable_failure',
        candidate_row.category_assignment_hash,
        candidate_row.conditional_policy_hash,
        candidate_row.window_hash,
        candidate_row.baseline_benchmark_bundle_id,
        lifecycle_source_ref,
        candidate_row.threshold_points,
        normalized_failure_class,
        lifecycle_event_doc,
        public.research_lab_conditional_validation_event_hash(
            candidate_row.candidate_id,
            'retryable_failure',
            candidate_row.category_assignment_hash,
            lifecycle_source_ref,
            lifecycle_event_doc
        )
    ) ON CONFLICT DO NOTHING;

    UPDATE public.research_lab_scoring_job_queue
       SET status = 'failed',
           result_doc = pg_catalog.jsonb_build_object(
               'retryable', TRUE,
               'failure_class', normalized_failure_class,
               'generation_closed', TRUE
           ),
           claimed_by = '',
           lease_expires_at = NULL,
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND phase = 'conditional'
       AND status IN ('held', 'queued', 'claimed');

    UPDATE public.research_lab_scoring_job_candidate
       SET preliminary_gate_status = 'skipped',
           preliminary_gate_proof = '{}'::JSONB,
           preliminary_gate_claimed_by = '',
           preliminary_gate_lease_expires_at = NULL,
           assembly_status = 'assembled',
           updated_at = NOW()
     WHERE queue_generation_id = target_queue_generation_id
       AND preliminary_gate_status = 'deciding'
       AND preliminary_gate_claimed_by = expected_claimed_by
       AND preliminary_gate_attempt_count = expected_attempt_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conditional cancellation claim changed inside serialized transition'
            USING ERRCODE = '40001';
    END IF;
    RETURN pg_catalog.jsonb_build_object('committed', TRUE);
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_requeue_conditional_scoring_job(
    target_job_id UUID,
    expected_claimed_by TEXT,
    expected_attempt_count INTEGER,
    target_failure_class TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    job_row public.research_lab_scoring_job_queue%ROWTYPE;
    candidate_row public.research_lab_scoring_job_candidate%ROWTYPE;
    normalized_failure_class TEXT;
    lifecycle_source_ref TEXT;
    lifecycle_event_doc JSONB;
BEGIN
    SELECT *
      INTO job_row
      FROM public.research_lab_scoring_job_queue
     WHERE job_id = target_job_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('committed', FALSE, 'reason', 'not_found');
    END IF;
    IF job_row.phase <> 'conditional' THEN
        RAISE EXCEPTION 'conditional retry RPC cannot requeue phase %', job_row.phase
            USING ERRCODE = '23514';
    END IF;
    IF job_row.status <> 'claimed'
       OR job_row.claimed_by <> expected_claimed_by
       OR job_row.attempt_count <> expected_attempt_count
    THEN
        RETURN pg_catalog.jsonb_build_object('committed', FALSE, 'reason', 'claim_changed');
    END IF;

    SELECT *
      INTO candidate_row
      FROM public.research_lab_scoring_job_candidate
     WHERE queue_generation_id = job_row.queue_generation_id
     FOR UPDATE;
    IF NOT FOUND OR candidate_row.conditional_total <= 0 THEN
        RAISE EXCEPTION 'conditional retry candidate commitment is missing'
            USING ERRCODE = '23514';
    END IF;

    normalized_failure_class := COALESCE(
        NULLIF(pg_catalog.btrim(target_failure_class), ''),
        'conditional_validation_retryable_failure'
    );
    lifecycle_source_ref := 'queue:' || job_row.queue_generation_id::TEXT
        || ':job:' || job_row.job_id::TEXT
        || ':attempt:' || job_row.attempt_count::TEXT;
    lifecycle_event_doc := pg_catalog.jsonb_build_object(
        'schema_version', '1.1',
        'failure_class', normalized_failure_class,
        'attempt_count', job_row.attempt_count,
        'queue_generation_id', job_row.queue_generation_id,
        'job_id', job_row.job_id
    );
    INSERT INTO public.research_lab_conditional_validation_events (
        event_id,
        candidate_id,
        queue_generation_id,
        event_type,
        assignment_hash,
        policy_hash,
        rolling_window_hash,
        baseline_benchmark_bundle_id,
        source_ref,
        threshold_points,
        failure_class,
        event_doc,
        event_hash
    ) VALUES (
        pg_catalog.gen_random_uuid(),
        candidate_row.candidate_id,
        job_row.queue_generation_id,
        'retryable_failure',
        candidate_row.category_assignment_hash,
        candidate_row.conditional_policy_hash,
        candidate_row.window_hash,
        candidate_row.baseline_benchmark_bundle_id,
        lifecycle_source_ref,
        candidate_row.threshold_points,
        normalized_failure_class,
        lifecycle_event_doc,
        public.research_lab_conditional_validation_event_hash(
            candidate_row.candidate_id,
            'retryable_failure',
            candidate_row.category_assignment_hash,
            lifecycle_source_ref,
            lifecycle_event_doc
        )
    ) ON CONFLICT DO NOTHING;

    UPDATE public.research_lab_scoring_job_queue
       SET status = 'queued',
           result_doc = pg_catalog.jsonb_build_object(
               'retryable', TRUE,
               'failure_class', normalized_failure_class
           ),
           claimed_by = '',
           lease_expires_at = NULL,
           updated_at = NOW()
     WHERE job_id = target_job_id
       AND status = 'claimed'
       AND claimed_by = expected_claimed_by
       AND attempt_count = expected_attempt_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conditional retry claim changed inside serialized transition'
            USING ERRCODE = '40001';
    END IF;
    RETURN pg_catalog.jsonb_build_object('committed', TRUE);
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_decide_conditional_public_gate(UUID, DOUBLE PRECISION)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_claim_conditional_preliminary_gate(UUID, TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_decide_conditional_preliminary_gate(UUID, DOUBLE PRECISION, JSONB, TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_cancel_conditional_generation(UUID, TEXT, INTEGER, TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_requeue_conditional_scoring_job(UUID, TEXT, INTEGER, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_decide_conditional_public_gate(UUID, DOUBLE PRECISION)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_claim_conditional_preliminary_gate(UUID, TEXT, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_decide_conditional_preliminary_gate(UUID, DOUBLE PRECISION, JSONB, TEXT, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_cancel_conditional_generation(UUID, TEXT, INTEGER, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_requeue_conditional_scoring_job(UUID, TEXT, INTEGER, TEXT)
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_scoring_category_results (
    category_result_id    TEXT        PRIMARY KEY
                                      CHECK (category_result_id ~ '^scoring_category:[0-9a-f]{64}$'),
    schema_version       TEXT        NOT NULL DEFAULT '1.1' CHECK (schema_version = '1.1'),
    source_kind          TEXT        NOT NULL CHECK (source_kind IN ('baseline', 'candidate')),
    source_bundle_ref    TEXT        NOT NULL,
    category             TEXT        NOT NULL CHECK (
                                      category IN (
                                          'public',
                                          'private',
                                          'conditional',
                                          'preliminary',
                                          'overall'
                                      )
                                    ),
    assignment_hash      TEXT        NOT NULL CHECK (assignment_hash ~ '^sha256:[0-9a-f]{64}$'),
    policy_hash          TEXT        NOT NULL CHECK (policy_hash ~ '^sha256:[0-9a-f]{64}$'),
    rolling_window_hash  TEXT        NOT NULL CHECK (rolling_window_hash ~ '^sha256:[0-9a-f]{64}$'),
    scoring_run_id       UUID        REFERENCES public.research_lab_scoring_runs(scoring_run_id)
                                      ON DELETE RESTRICT,
    candidate_id         TEXT        REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                      ON DELETE RESTRICT,
    icp_count            INTEGER     NOT NULL CHECK (icp_count > 0),
    aggregate_score      DOUBLE PRECISION NOT NULL CHECK (
                                      aggregate_score >= 0 AND aggregate_score <= 100
                                    ),
    delta_vs_baseline    DOUBLE PRECISION CHECK (
                                      delta_vs_baseline IS NULL
                                      OR (
                                          delta_vs_baseline >= -100
                                          AND delta_vs_baseline <= 100
                                      )
                                    ),
    result_doc           JSONB       NOT NULL CHECK (
                                      jsonb_typeof(result_doc) = 'object'
                                      AND result_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_icp|icp_plaintext|intent_signals|provider_output|request_body|response_body|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                    ),
    result_hash          TEXT        NOT NULL UNIQUE CHECK (result_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash        TEXT        NOT NULL UNIQUE CHECK (anchored_hash = result_hash),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_scoring_category_results_source_key
        UNIQUE (source_bundle_ref, category, assignment_hash),
    CHECK (
        (source_kind = 'candidate' AND candidate_id IS NOT NULL)
        OR (source_kind = 'baseline' AND candidate_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_category_results_run
    ON public.research_lab_scoring_category_results(scoring_run_id, category)
    WHERE scoring_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_category_results_candidate
    ON public.research_lab_scoring_category_results(candidate_id, created_at DESC)
    WHERE candidate_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_category_results_window
    ON public.research_lab_scoring_category_results(rolling_window_hash, source_kind, category);

DROP TRIGGER IF EXISTS prevent_research_lab_scoring_category_results_mutation
    ON public.research_lab_scoring_category_results;
CREATE TRIGGER prevent_research_lab_scoring_category_results_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_scoring_category_results
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_scoring_category_results FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_scoring_category_results TO service_role;

ALTER TABLE public.research_lab_scoring_category_results ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_scoring_category_results;
CREATE POLICY service_role_read ON public.research_lab_scoring_category_results
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_scoring_category_results;
CREATE POLICY service_role_insert ON public.research_lab_scoring_category_results
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_scoring_category_results IS
    'Service-role-only aggregate public/private/conditional scoring results. Hidden ICP identities remain in immutable private source bundles.';

NOTIFY pgrst, 'reload schema';

COMMIT;
