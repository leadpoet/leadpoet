BEGIN;

ALTER TABLE public.research_lab_routing_experiment_events_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_routing_experiment_events_v2_event_type_check;
ALTER TABLE public.research_lab_routing_experiment_events_v2
    ADD CONSTRAINT research_lab_routing_experiment_events_v2_event_type_check
    CHECK (event_type IN (
        'submitted', 'claimed', 'claim_recovered', 'run_started', 'run_completed',
        'run_failed', 'promotion_requested', 'promoted',
        'provider_dispatch_started', 'model_transition_completed'
    )) NOT VALID;
ALTER TABLE public.research_lab_routing_experiment_events_v2
    VALIDATE CONSTRAINT
        research_lab_routing_experiment_events_v2_event_type_check;

-- Add one redacted, claim-fenced Model transition marker. The provider body
-- and Model continuation are transient and are never stored in PostgreSQL.
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
           ) <> 13
           OR p_event_doc->>'schema_version'
                IS DISTINCT FROM 'leadpoet.research_lab.routing_event.v2'
           OR p_event_doc->>'event_schema_version'
                IS DISTINCT FROM 'leadpoet.research_lab.model_transition.v1'
           OR p_event_doc->>'variant_id'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'unit_ref'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'idempotency_key' !~ '^[0-9a-f]{64}$'
           OR p_event_doc->>'action_sha256' !~ '^[0-9a-f]{64}$'
           OR p_event_doc->>'continuation_sha256' !~ '^sha256:[0-9a-f]{64}$'
           OR p_event_doc->>'completion_sha256' !~ '^[0-9a-f]{64}$'
           OR p_event_doc->>'provider_response_sha256' !~ '^sha256:[0-9a-f]{64}$'
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND p_event_doc->>'protected_dispatch_job_id'
                    !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND p_event_doc->>'terminal_receipt_hash'
                    !~ '^sha256:[0-9a-f]{64}$'
           )
           OR (
               p_event_doc->'provider_receipt' <> 'null'::JSONB
               AND p_event_doc->>'model_completion_contract_hash'
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

-- Retire the twelve legacy mutation and promotion entry points after the V3
-- chain is installed. Keep submit/request, recovery bootstrap, and queue RPCs.
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_experiment_v2(
    TEXT, TEXT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_renew_claim_v2(
    TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_close_claim_v2(
    TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_fenced_event_v2(
    TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_provider_attempt_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT,
    TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, BIGINT, BIGINT, TEXT,
    TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_decision_receipt_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_evaluation_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_reserve_budget_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, BIGINT, INTEGER, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_settle_budget_v2(
    TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_mark_budget_uncertain_v2(
    TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_recover_budget_v2(
    TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_promote_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB
) FROM service_role;

COMMENT ON FUNCTION public.research_lab_routing_append_fenced_event_v3(
    TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB
) IS 'Appends V3 claim-fenced lifecycle events and redacted exact-Model transition markers.';

NOTIFY pgrst, 'reload schema';

COMMIT;
