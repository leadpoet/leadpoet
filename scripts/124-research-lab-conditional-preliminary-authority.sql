-- Make the attested 20-ICP promotion decision authoritative for conditional
-- release. A deterministic rejection is terminal; only infrastructure or
-- evidence failures leave the gate retryable.
--
-- Additive function replacement only. Apply after migration 97.

BEGIN;

SET LOCAL lock_timeout = '5s';

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
    host_score_passed BOOLEAN;
    passed BOOLEAN;
    proof_status TEXT;
    decision_kind TEXT;
    decision_auto_promotion_enabled BOOLEAN;
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
        RETURN pg_catalog.jsonb_build_object('decision', 'not_found');
    END IF;
    IF candidate_row.gate_status <> 'passed' THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'already_decided');
    END IF;
    IF candidate_row.preliminary_gate_status <> 'deciding'
       OR candidate_row.preliminary_gate_claimed_by <> expected_claimed_by
       OR candidate_row.preliminary_gate_attempt_count <> expected_attempt_count
       OR candidate_row.preliminary_gate_lease_expires_at IS NULL
       OR candidate_row.preliminary_gate_lease_expires_at <= NOW()
    THEN
        RETURN pg_catalog.jsonb_build_object('decision', 'claim_changed');
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
        RETURN pg_catalog.jsonb_build_object('decision', 'not_ready');
    END IF;

    host_score_passed := candidate_preliminary_score
        - candidate_row.baseline_preliminary_score
        + 0.000000001 >= candidate_row.threshold_points;
    IF target_preliminary_proof IS NULL
       OR pg_catalog.jsonb_typeof(target_preliminary_proof) <> 'object'
    THEN
        RAISE EXCEPTION 'conditional preliminary proof must be a JSON object'
            USING ERRCODE = '22023';
    END IF;

    IF host_score_passed THEN
        proof_status := COALESCE(target_preliminary_proof->>'status', '');
        decision_kind := COALESCE(
            target_preliminary_proof->'decision'->>'candidate_kind',
            ''
        );
        decision_auto_promotion_enabled := COALESCE(
            (
                target_preliminary_proof->'decision'
                ->>'auto_promotion_enabled'
            )::BOOLEAN,
            FALSE
        );
        IF proof_status NOT IN (
               'promotion_passed',
               'disabled',
               'rejected_legacy_patch_candidate',
               'rejected_basis_unavailable',
               'rejected_paired_lcb_unavailable',
               'rejected_paired_lcb_gate_ineligible',
               'rejected_below_threshold'
           )
           OR target_preliminary_proof->>'schema_version'
               IS DISTINCT FROM 'research_lab_preliminary_promotion_gate.v1'
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
               IS DISTINCT FROM proof_status
           OR (
               proof_status = 'rejected_legacy_patch_candidate'
               AND decision_kind = 'image_build'
           )
           OR (
               proof_status <> 'rejected_legacy_patch_candidate'
               AND decision_kind IS DISTINCT FROM 'image_build'
           )
           OR (
               proof_status = 'disabled'
               AND decision_auto_promotion_enabled IS TRUE
           )
           OR (
               proof_status <> 'disabled'
               AND decision_auto_promotion_enabled IS NOT TRUE
           )
           OR COALESCE(
               (
                   target_preliminary_proof->'decision'
                   ->>'active_parent_matches'
               )::BOOLEAN,
               FALSE
           ) IS NOT TRUE
        THEN
            RAISE EXCEPTION 'conditional preliminary attested proof does not match frozen queue commitments'
                USING ERRCODE = '23514';
        END IF;
        passed := proof_status = 'promotion_passed';
    ELSE
        IF target_preliminary_proof <> '{}'::JSONB THEN
            RAISE EXCEPTION 'host-rejected conditional preliminary gate must not carry an authority proof'
                USING ERRCODE = '23514';
        END IF;
        proof_status := 'rejected_below_host_threshold';
        passed := FALSE;
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
        'authoritative_decision_status', proof_status,
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
        lifecycle_event_doc := lifecycle_event_doc
            || pg_catalog.jsonb_build_object(
                'preliminary_event',
                'persisted_before_conditional_release'
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
    RETURN pg_catalog.jsonb_build_object(
        'decision', CASE WHEN passed THEN 'passed' ELSE 'rejected' END
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_decide_conditional_preliminary_gate(
    UUID,
    DOUBLE PRECISION,
    JSONB,
    TEXT,
    INTEGER
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_decide_conditional_preliminary_gate(
    UUID,
    DOUBLE PRECISION,
    JSONB,
    TEXT,
    INTEGER
) TO service_role;

COMMENT ON FUNCTION public.research_lab_decide_conditional_preliminary_gate(
    UUID,
    DOUBLE PRECISION,
    JSONB,
    TEXT,
    INTEGER
) IS
    'Serializes conditional release against the attested legacy 20-ICP promotion decision; deterministic attested rejection is terminal.';

NOTIFY pgrst, 'reload schema';

COMMIT;
