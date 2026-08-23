\set ON_ERROR_STOP on

-- Run after migrations 157 and 163 against disposable PostgreSQL only.
BEGIN;

DO $official_baseline_happy_path$
DECLARE
    h TEXT := 'sha256:' || repeat('a', 64);
    run_hash TEXT := 'sha256:' || repeat('1', 64);
    unit_value TEXT := 'baseline_icp:' || repeat('2', 64);
    registration JSONB;
    action_auth JSONB;
    terminal JSONB;
    identity JSONB;
    completion JSONB;
    result JSONB;
    frontier_hash TEXT;
    first_attempt_hash TEXT;
BEGIN
    registration := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_run_registration.v1',
        'run_sha256', run_hash,
        'benchmark_date', '2026-08-23',
        'rolling_window_hash', 'sha256:' || repeat('3', 64),
        'model_artifact_hash', 'sha256:' || repeat('4', 64),
        'manifest_hash', 'sha256:' || repeat('5', 64),
        'release_selection_sha256', 'sha256:' || repeat('6', 64),
        'artifact_key_sha256', 'sha256:' || repeat('7', 64),
        'protocol_generation_sha256', 'sha256:' || repeat('8', 64),
        'projection_identity_sha256', 'sha256:' || repeat('9', 64),
        'authority_identity_sha256', 'sha256:' || repeat('a', 64)
    );
    result := public.research_lab_official_baseline_register_run_v1(registration);
    IF result->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_run_registration_result.v1'
       OR (result->>'idempotent')::BOOLEAN IS NOT FALSE
       OR result->>'registration_sha256' IS DISTINCT FROM
            public.research_lab_official_baseline_hash_v1(registration)
    THEN
        RAISE EXCEPTION 'new run registration response invalid: %', result;
    END IF;
    result := public.research_lab_official_baseline_register_run_v1(registration);
    IF (result->>'idempotent')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'run registration replay was not idempotent: %', result;
    END IF;

    frontier_hash := public.research_lab_official_baseline_hash_v1(
        public.research_lab_official_baseline_provider_frontier_doc_v1(
            run_hash, unit_value
        )
    );
    action_auth := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_authorization.v1',
        'attempt_key', 'sha256:' || repeat('b', 64),
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'action_idempotency_sha256', 'sha256:' || repeat('c', 64),
        'action_sha256', 'sha256:' || repeat('d', 64),
        'action_sequence', 0,
        'action_type', 'execute_candidate_tool',
        'tool_id', 'candidate.exa.search',
        'binding_contract_sha256', h,
        'request_fingerprint_sha256', 'sha256:' || repeat('e', 64),
        'request_body_sha256', 'sha256:' || repeat('f', 64),
        'call_cap', 2,
        'credit_cap_microunits', 100,
        'timeout_ms', 30000,
        'protected_job_ref', 'protected_job:first',
        'protected_request_sha256', 'sha256:' || repeat('0', 64),
        'lease_holder_sha256', h,
        'expected_frontier_sha256', frontier_hash
    );
    result := public.research_lab_official_baseline_reserve_action_v1(
        action_auth
    );
    IF result->>'disposition' IS DISTINCT FROM 'reserved_new'
       OR result->>'attempt_key' IS DISTINCT FROM action_auth->>'attempt_key'
       OR result->>'attempt_sha256' IS DISTINCT FROM
            public.research_lab_official_baseline_hash_v1(action_auth)
    THEN
        RAISE EXCEPTION 'new provider reservation invalid: %', result;
    END IF;
    result := public.research_lab_official_baseline_reserve_action_v1(
        action_auth
    );
    IF result->>'disposition' IS DISTINCT FROM 'reserved_existing' THEN
        RAISE EXCEPTION 'provider reservation replay invalid: %', result;
    END IF;

    identity := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_replay_identity.v1',
        'attempt_key', action_auth->>'attempt_key',
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'action_idempotency_sha256',
            action_auth->>'action_idempotency_sha256',
        'action_sha256', action_auth->>'action_sha256',
        'request_fingerprint_sha256',
            action_auth->>'request_fingerprint_sha256'
    );
    result := public.research_lab_official_baseline_load_replay_v1(identity);
    IF result->>'state' IS DISTINCT FROM 'reserved' THEN
        RAISE EXCEPTION 'reserved replay state invalid: %', result;
    END IF;

    terminal := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_terminal_known.v1',
        'attempt_key', action_auth->>'attempt_key',
        'reservation_ref', 'baseline_reservation:' || repeat('b', 64),
        'lease_generation', 1,
        'protected_job_ref', 'protected_job:first',
        'protected_request_sha256', action_auth->>'protected_request_sha256',
        'protected_result_sha256', 'sha256:' || repeat('1', 64),
        'protected_terminal_receipt_ref', 'baseline_terminal:first',
        'protected_terminal_receipt_sha256', 'sha256:' || repeat('2', 64),
        'provider_request_ref', 'provider_request:first',
        'provider_receipt_ref', 'provider_receipt:' || repeat('3', 16),
        'provider_receipt_sha256', 'sha256:' || repeat('4', 64),
        'provider_identity_sha256', 'sha256:' || repeat('5', 64),
        'model_provider_response_sha256', 'sha256:' || repeat('6', 64),
        'outcome', 'succeeded',
        'call_count', 1,
        'cost_microunits', 75,
        'latency_ms', 1200
    );
    result := public.research_lab_official_baseline_record_terminal_known_v1(
        terminal
    );
    IF result->>'state' IS DISTINCT FROM 'terminal_known'
       OR (result->>'idempotent')::BOOLEAN IS NOT FALSE
    THEN
        RAISE EXCEPTION 'provider terminal response invalid: %', result;
    END IF;
    first_attempt_hash := result->>'attempt_sha256';
    result := public.research_lab_official_baseline_record_terminal_known_v1(
        terminal
    );
    IF (result->>'idempotent')::BOOLEAN IS NOT TRUE
       OR result->>'attempt_sha256' IS DISTINCT FROM first_attempt_hash
    THEN
        RAISE EXCEPTION 'provider terminal replay changed identity: %', result;
    END IF;

    frontier_hash := public.research_lab_official_baseline_hash_v1(
        public.research_lab_official_baseline_provider_frontier_doc_v1(
            run_hash, unit_value
        )
    );
    action_auth := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_authorization.v1',
        'attempt_key', 'sha256:' || repeat('7', 64),
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'action_idempotency_sha256', 'sha256:' || repeat('8', 64),
        'action_sha256', 'sha256:' || repeat('9', 64),
        'action_sequence', 1,
        'action_type', 'verify_company',
        'tool_id', 'company.verifier.v2',
        'binding_contract_sha256', h,
        'request_fingerprint_sha256', 'sha256:' || repeat('a', 64),
        'request_body_sha256', 'sha256:' || repeat('b', 64),
        'call_cap', 0,
        'credit_cap_microunits', 0,
        'timeout_ms', 5000,
        'protected_job_ref', 'protected_job:verifier',
        'protected_request_sha256', 'sha256:' || repeat('c', 64),
        'lease_holder_sha256', h,
        'expected_frontier_sha256', frontier_hash
    );
    result := public.research_lab_official_baseline_reserve_action_v1(
        action_auth
    );
    IF result->>'disposition' IS DISTINCT FROM 'reserved_new' THEN
        RAISE EXCEPTION 'verifier reservation invalid: %', result;
    END IF;
    terminal := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_terminal_known.v1',
        'attempt_key', action_auth->>'attempt_key',
        'reservation_ref', 'baseline_reservation:' || repeat('7', 64),
        'lease_generation', 1,
        'protected_job_ref', 'protected_job:verifier',
        'protected_request_sha256', action_auth->>'protected_request_sha256',
        'protected_result_sha256', 'sha256:' || repeat('d', 64),
        'protected_terminal_receipt_ref', 'baseline_terminal:verifier',
        'protected_terminal_receipt_sha256', 'sha256:' || repeat('e', 64),
        'provider_request_ref', NULL,
        'provider_receipt_ref', NULL,
        'provider_receipt_sha256', NULL,
        'provider_identity_sha256', NULL,
        'model_provider_response_sha256', 'sha256:' || repeat('f', 64),
        'outcome', 'failed',
        'call_count', 0,
        'cost_microunits', 0,
        'latency_ms', 2
    );
    PERFORM public.research_lab_official_baseline_record_terminal_known_v1(
        terminal
    );
    identity := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_replay_identity.v1',
        'attempt_key', action_auth->>'attempt_key',
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'action_idempotency_sha256',
            action_auth->>'action_idempotency_sha256',
        'action_sha256', action_auth->>'action_sha256',
        'request_fingerprint_sha256',
            action_auth->>'request_fingerprint_sha256'
    );
    result := public.research_lab_official_baseline_load_replay_v1(identity);
    IF result->>'state' IS DISTINCT FROM 'terminal_known'
       OR result->>'outcome' IS DISTINCT FROM 'failed'
       OR (result->>'call_count')::INTEGER <> 0
       OR (result->>'cost_microunits')::BIGINT <> 0
       OR result->'provider_request_ref' IS DISTINCT FROM 'null'::JSONB
       OR result->'provider_receipt_ref' IS DISTINCT FROM 'null'::JSONB
       OR result->'provider_receipt_sha256' IS DISTINCT FROM 'null'::JSONB
       OR result->'provider_identity_sha256' IS DISTINCT FROM 'null'::JSONB
    THEN
        RAISE EXCEPTION 'failed verifier replay custody invalid: %', result;
    END IF;

    completion := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_unit_completion.v1',
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'protocol_generation_sha256', registration->>'protocol_generation_sha256',
        'raw_input_sha256', 'sha256:' || repeat('1', 64),
        'start_request_sha256', 'sha256:' || repeat('2', 64),
        'terminal_result_sha256', 'sha256:' || repeat('3', 64),
        'model_receipt_sha256', 'sha256:' || repeat('4', 64),
        'projection_sha256', 'sha256:' || repeat('5', 64)
    );
    result := public.research_lab_official_baseline_close_unit_v1(completion);
    IF result->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_unit_closure.v1'
       OR (result->>'idempotent')::BOOLEAN IS NOT FALSE
       OR result->'ordered_attempt_keys' IS DISTINCT FROM
            pg_catalog.jsonb_build_array(
                'sha256:' || repeat('b', 64),
                'sha256:' || repeat('7', 64)
            )
       OR result->>'closure_ref' IS DISTINCT FROM
            'baseline_closure:' || pg_catalog.substr(result->>'closure_sha256', 8)
    THEN
        RAISE EXCEPTION 'unit closure invalid: %', result;
    END IF;
    result := public.research_lab_official_baseline_close_unit_v1(completion);
    IF (result->>'idempotent')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'unit closure replay was not idempotent: %', result;
    END IF;
    result := public.research_lab_official_baseline_load_frontier_v1(
        run_hash, unit_value
    );
    IF (result->>'idempotent')::BOOLEAN IS NOT TRUE
       OR result->>'closure_sha256' IS DISTINCT FROM
            public.research_lab_official_baseline_hash_v1(
                result - ARRAY['closure_ref', 'closure_sha256', 'idempotent']
            )
    THEN
        RAISE EXCEPTION 'unit closure readback invalid: %', result;
    END IF;

    BEGIN
        UPDATE public.research_lab_official_baseline_action_attempts_v1
           SET tool_id = tool_id
         WHERE attempt_key = 'sha256:' || repeat('b', 64);
        RAISE EXCEPTION 'append-only attempt update unexpectedly succeeded';
    EXCEPTION
        WHEN raise_exception THEN
            IF SQLERRM = 'append-only attempt update unexpectedly succeeded' THEN
                RAISE;
            END IF;
    END;
END;
$official_baseline_happy_path$;

DO $official_baseline_uncertain_path$
DECLARE
    run_hash TEXT := 'sha256:' || repeat('1', 64);
    unit_value TEXT := 'baseline_icp:' || repeat('c', 64);
    frontier_hash TEXT;
    action_auth JSONB;
    uncertainty JSONB;
    result JSONB;
BEGIN
    frontier_hash := public.research_lab_official_baseline_hash_v1(
        public.research_lab_official_baseline_provider_frontier_doc_v1(
            run_hash, unit_value
        )
    );
    action_auth := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_authorization.v1',
        'attempt_key', 'sha256:' || repeat('d', 64),
        'run_sha256', run_hash,
        'unit_ref', unit_value,
        'action_idempotency_sha256', 'sha256:' || repeat('e', 64),
        'action_sha256', 'sha256:' || repeat('f', 64),
        'action_sequence', 0,
        'action_type', 'normalize_icp',
        'tool_id', 'model.normalize_icp',
        'binding_contract_sha256', 'sha256:' || repeat('1', 64),
        'request_fingerprint_sha256', 'sha256:' || repeat('2', 64),
        'request_body_sha256', 'sha256:' || repeat('3', 64),
        'call_cap', 1,
        'credit_cap_microunits', 10,
        'timeout_ms', 1000,
        'protected_job_ref', 'protected_job:uncertain',
        'protected_request_sha256', 'sha256:' || repeat('4', 64),
        'lease_holder_sha256', 'sha256:' || repeat('5', 64),
        'expected_frontier_sha256', frontier_hash
    );
    PERFORM public.research_lab_official_baseline_reserve_action_v1(
        action_auth
    );
    uncertainty := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_terminal_uncertain.v1',
        'attempt_key', action_auth->>'attempt_key',
        'reservation_ref', 'baseline_reservation:' || repeat('d', 64),
        'lease_generation', 1,
        'protected_job_ref', action_auth->>'protected_job_ref',
        'protected_request_sha256', action_auth->>'protected_request_sha256',
        'provider_request_ref', 'provider_request:uncertain',
        'uncertainty_sha256', 'sha256:' || repeat('6', 64)
    );
    result := public.research_lab_official_baseline_record_terminal_uncertain_v1(
        uncertainty
    );
    IF result->>'state' IS DISTINCT FROM 'terminal_uncertain' THEN
        RAISE EXCEPTION 'uncertain terminal was not durable: %', result;
    END IF;
    action_auth := action_auth || pg_catalog.jsonb_build_object(
        'attempt_key', 'sha256:' || repeat('6', 64),
        'action_idempotency_sha256', 'sha256:' || repeat('8', 64),
        'action_sha256', 'sha256:' || repeat('9', 64),
        'action_sequence', 1,
        'protected_job_ref', 'protected_job:must-not-dispatch',
        'protected_request_sha256', 'sha256:' || repeat('a', 64),
        'expected_frontier_sha256',
            public.research_lab_official_baseline_hash_v1(
                public.research_lab_official_baseline_provider_frontier_doc_v1(
                    run_hash, unit_value
                )
            )
    );
    BEGIN
        PERFORM public.research_lab_official_baseline_reserve_action_v1(
            action_auth
        );
        RAISE EXCEPTION 'uncertain provider call was redispatched';
    EXCEPTION
        WHEN SQLSTATE '40003' THEN NULL;
    END;
END;
$official_baseline_uncertain_path$;

SET LOCAL ROLE service_role;
DO $official_baseline_service_role_boundary$
DECLARE
    run_hash TEXT := 'sha256:' || repeat('0', 64);
    registration JSONB;
    result JSONB;
BEGIN
    registration := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_run_registration.v1',
        'run_sha256', run_hash,
        'benchmark_date', '2026-08-23',
        'rolling_window_hash', 'sha256:' || repeat('1', 64),
        'model_artifact_hash', 'sha256:' || repeat('2', 64),
        'manifest_hash', 'sha256:' || repeat('3', 64),
        'release_selection_sha256', 'sha256:' || repeat('4', 64),
        'artifact_key_sha256', 'sha256:' || repeat('5', 64),
        'protocol_generation_sha256', 'sha256:' || repeat('6', 64),
        'projection_identity_sha256', 'sha256:' || repeat('7', 64),
        'authority_identity_sha256', 'sha256:' || repeat('8', 64)
    );
    result := public.research_lab_official_baseline_register_run_v1(registration);
    IF result->>'run_sha256' IS DISTINCT FROM run_hash THEN
        RAISE EXCEPTION 'service-role RPC registration failed: %', result;
    END IF;
    BEGIN
        INSERT INTO public.research_lab_official_baseline_runs_v1 (
            run_sha256, registration_sha256, benchmark_date,
            rolling_window_hash, model_artifact_hash, manifest_hash,
            release_selection_sha256, artifact_key_sha256,
            protocol_generation_sha256, projection_identity_sha256,
            authority_identity_sha256, registration_doc
        ) VALUES (
            'sha256:' || repeat('9', 64), 'sha256:' || repeat('a', 64),
            DATE '2026-08-23', 'sha256:' || repeat('b', 64),
            'sha256:' || repeat('c', 64), 'sha256:' || repeat('d', 64),
            'sha256:' || repeat('e', 64), 'sha256:' || repeat('f', 64),
            'sha256:' || repeat('1', 64), 'sha256:' || repeat('2', 64),
            'sha256:' || repeat('3', 64), '{}'::JSONB
        );
        RAISE EXCEPTION 'service role wrote directly to baseline authority';
    EXCEPTION
        WHEN insufficient_privilege THEN NULL;
    END;
END;
$official_baseline_service_role_boundary$;
RESET ROLE;

ROLLBACK;
