BEGIN;

-- Bind every redacted continuation marker to the exact registered model
-- artifact. Logical identity is checked under the existing per-experiment
-- advisory lock so a racing artifact cannot create a second marker and cause
-- another paid dispatch.
CREATE OR REPLACE FUNCTION public.research_lab_routing_append_fenced_event_v3(
    p_event_hash TEXT,
    p_experiment_hash TEXT,
    p_event_type TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_fenced_event_v3$
DECLARE
    existing public.research_lab_routing_experiment_events_v2%ROWTYPE;
    existing_logical public.research_lab_routing_experiment_events_v2%ROWTYPE;
BEGIN
    IF p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_type NOT IN (
           'run_started', 'run_completed', 'run_failed', 'promotion_requested',
           'provider_dispatch_started', 'model_transition_completed'
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_event_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing fenced event v3'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_events_v2
     WHERE event_hash = p_event_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.event_type IS DISTINCT FROM p_event_type
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_event_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'event_hash', p_event_hash, 'idempotent', TRUE
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    IF p_event_type = 'provider_dispatch_started' THEN
        IF p_event_doc->>'reservation_id'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'binding_id'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'request_fingerprint'
                !~ '^sha256:[0-9a-f]{64}$'
           OR NOT EXISTS (
               SELECT 1
                 FROM public.research_lab_routing_budget_events_v2 reserve_event
                WHERE reserve_event.experiment_hash = p_experiment_hash
                  AND reserve_event.reservation_id = p_event_doc->>'reservation_id'
                  AND reserve_event.binding_id = p_event_doc->>'binding_id'
                  AND reserve_event.claim_key = p_claim_key
                  AND reserve_event.claim_generation = p_claim_generation
                  AND reserve_event.event_type = 'reserve'
                  AND reserve_event.event_doc = p_event_doc
           )
        THEN
            RAISE EXCEPTION 'research_lab_routing_dispatch_event_v3_reservation_mismatch'
                USING ERRCODE = '23503';
        END IF;
    ELSIF p_event_type = 'model_transition_completed' THEN
        IF (
               SELECT pg_catalog.count(*)
                 FROM pg_catalog.jsonb_object_keys(p_event_doc)
           ) <> 14
           OR NOT (p_event_doc ?& ARRAY[
               'schema_version',
               'event_schema_version',
               'variant_id',
               'unit_ref',
               'artifact_key',
               'idempotency_key',
               'action_sha256',
               'continuation_sha256',
               'completion_sha256',
               'provider_response_sha256',
               'provider_receipt',
               'protected_dispatch_job_id',
               'terminal_receipt_hash',
               'model_completion_contract_hash'
           ]::TEXT[])
           OR p_event_doc->>'schema_version'
                IS DISTINCT FROM 'leadpoet.research_lab.routing_event.v2'
           OR p_event_doc->>'event_schema_version'
                IS DISTINCT FROM 'leadpoet.research_lab.model_transition.v2'
           OR COALESCE(p_event_doc->>'variant_id', '')
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR COALESCE(p_event_doc->>'unit_ref', '')
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR COALESCE(p_event_doc->>'artifact_key', '')
                !~ '^[0-9a-f]{40}:(sha256:)?[0-9a-f]{64}:(sha256:)?[0-9a-f]{64}$'
           OR COALESCE(p_event_doc->>'idempotency_key', '')
                !~ '^[0-9a-f]{64}$'
           OR COALESCE(p_event_doc->>'action_sha256', '')
                !~ '^[0-9a-f]{64}$'
           OR COALESCE(p_event_doc->>'continuation_sha256', '')
                !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(p_event_doc->>'completion_sha256', '')
                !~ '^[0-9a-f]{64}$'
           OR COALESCE(
               p_event_doc->>'provider_response_sha256', ''
           ) !~ '^sha256:[0-9a-f]{64}$'
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND COALESCE(
                   p_event_doc->>'protected_dispatch_job_id', ''
               )
                    !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND COALESCE(
                   p_event_doc->>'terminal_receipt_hash', ''
               )
                    !~ '^sha256:[0-9a-f]{64}$'
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND COALESCE(
                   p_event_doc->>'model_completion_contract_hash', ''
               )
                    !~ '^sha256:[0-9a-f]{64}$'
           )
           OR (
               p_event_doc->'provider_receipt' = 'null'::JSONB
               AND (
                   p_event_doc->'protected_dispatch_job_id' <> 'null'::JSONB
                   OR p_event_doc->'terminal_receipt_hash' <> 'null'::JSONB
                   OR p_event_doc->'model_completion_contract_hash' <> 'null'::JSONB
               )
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND pg_catalog.jsonb_typeof(p_event_doc->'provider_receipt')
                    <> 'object'
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND NOT EXISTS (
                   SELECT 1
                     FROM public.research_lab_routing_provider_attempts_v2 attempt
                    WHERE attempt.experiment_hash = p_experiment_hash
                      AND attempt.variant_id = p_event_doc->>'variant_id'
                      AND attempt.unit_ref = p_event_doc->>'unit_ref'
                      AND attempt.provider_receipt_ref =
                            p_event_doc->'provider_receipt'->>'receipt_ref'
                      AND attempt.terminal_receipt_hash =
                            p_event_doc->>'terminal_receipt_hash'
                      AND attempt.attempt_doc->'terminal_execution_receipt'->>'job_id' =
                            p_event_doc->>'protected_dispatch_job_id'
                      AND attempt.attempt_doc->'terminal_result'->>'model_provider_response_sha256' =
                            p_event_doc->>'provider_response_sha256'
                      AND attempt.attempt_doc->'terminal_result'->>'model_completion_contract_hash' =
                            p_event_doc->>'model_completion_contract_hash'
               )
           )
        THEN
            RAISE EXCEPTION 'research_lab_routing_model_transition_v3_invalid'
                USING ERRCODE = '22023';
        END IF;

        -- Do not include artifact_key in this lookup. A mismatched artifact is
        -- a conflict, not a cache miss that may dispatch the provider again.
        SELECT * INTO existing_logical
          FROM public.research_lab_routing_experiment_events_v2 transition_event
         WHERE transition_event.experiment_hash = p_experiment_hash
           AND transition_event.event_type = 'model_transition_completed'
           AND transition_event.event_doc->>'variant_id' =
                 p_event_doc->>'variant_id'
           AND transition_event.event_doc->>'unit_ref' =
                 p_event_doc->>'unit_ref'
           AND transition_event.event_doc->>'idempotency_key' =
                 p_event_doc->>'idempotency_key'
         ORDER BY transition_event.created_at
         LIMIT 1;
        IF FOUND THEN
            IF existing_logical.event_doc->>'artifact_key'
                    IS DISTINCT FROM p_event_doc->>'artifact_key'
            THEN
                RAISE EXCEPTION
                    'research_lab_routing_model_transition_artifact_conflict'
                    USING ERRCODE = '23505';
            END IF;
            RAISE EXCEPTION
                'research_lab_routing_model_transition_logical_conflict'
                USING ERRCODE = '23505';
        END IF;
    END IF;
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key,
        claim_generation, event_doc
    ) VALUES (
        p_event_hash, p_experiment_hash, p_event_type, p_claim_key,
        p_claim_generation, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'event_hash', p_event_hash, 'idempotent', FALSE
    );
END;
$append_fenced_event_v3$;

CREATE OR REPLACE FUNCTION
public.research_lab_routing_exact_model_transition_contract_v2()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $exact_model_transition_contract_v2$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.research_lab.exact_model_transition_contract.v2',
        'event_type',
        'model_transition_completed',
        'marker_schema_version',
        'leadpoet.research_lab.model_transition.v2',
        'artifact_identity_required',
        TRUE,
        'logical_identity_conflict_guard',
        TRUE,
        'legacy_v1_eligible',
        FALSE
    );
$exact_model_transition_contract_v2$;

REVOKE ALL ON FUNCTION
public.research_lab_routing_exact_model_transition_contract_v2()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
public.research_lab_routing_exact_model_transition_contract_v2()
TO service_role;

COMMENT ON FUNCTION public.research_lab_routing_append_fenced_event_v3(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) IS 'Appends V3 claim-fenced lifecycle events and artifact-bound redacted exact-Model transition markers.';

COMMENT ON FUNCTION
public.research_lab_routing_exact_model_transition_contract_v2() IS
    'Read-only capability proving exact-Model transition markers require artifact identity and reject identityless V1 recovery.';

NOTIFY pgrst, 'reload schema';

COMMIT;
