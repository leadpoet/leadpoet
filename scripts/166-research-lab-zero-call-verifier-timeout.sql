-- Preserve the model artifact's exact zero-call verifier budget.
--
-- Migration 164 required a positive timeout for every protected action.  The
-- signed model inventory deliberately publishes timeout_seconds=0 for its
-- pure, zero-network company/intent/contact verifiers.  Permit zero only for
-- those verifier actions; paid/provider actions retain their positive bound.

BEGIN;

ALTER TABLE public.research_lab_official_baseline_action_attempts_v1
    DROP CONSTRAINT IF EXISTS
        research_lab_official_baseline_action_attempts_timeout_ms_check;

DO $verifier_timeout_constraint$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_constraint constraint_row
         WHERE constraint_row.conrelid =
                'public.research_lab_official_baseline_action_attempts_v1'::REGCLASS
           AND constraint_row.conname =
                'research_lab_official_baseline_action_timeout_by_type_v1'
    ) THEN
        ALTER TABLE public.research_lab_official_baseline_action_attempts_v1
            ADD CONSTRAINT research_lab_official_baseline_action_timeout_by_type_v1
            CHECK (
                (
                    action_type IN (
                        'verify_company', 'verify_intent', 'verify_contact'
                    )
                    AND timeout_ms = 0
                )
                OR
                (
                    action_type IN (
                        'normalize_icp', 'execute_candidate_tool',
                        'execute_intent_tool', 'execute_contact_tool'
                    )
                    AND timeout_ms BETWEEN 1 AND 900000
                )
            );
    END IF;
END;
$verifier_timeout_constraint$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_reserve_action_v1(
    p_authorization JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $reserve_action$
DECLARE
    authorization_hash TEXT;
    frontier_doc JSONB;
    frontier_hash TEXT;
    expected_sequence INTEGER;
    reservation_ref_value TEXT;
    expires_at_value TIMESTAMPTZ;
    attempt public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    terminal public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_authorization,
        ARRAY[
            'schema_version', 'attempt_key', 'run_sha256', 'unit_ref',
            'action_idempotency_sha256', 'action_sha256', 'action_sequence',
            'action_type', 'tool_id', 'binding_contract_sha256',
            'request_fingerprint_sha256', 'request_body_sha256', 'call_cap',
            'credit_cap_microunits', 'timeout_ms', 'protected_job_ref',
            'protected_request_sha256', 'lease_holder_sha256',
            'expected_frontier_sha256'
        ]
    )
       OR p_authorization->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_action_authorization.v1'
       OR p_authorization->>'attempt_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'run_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'unit_ref' !~ '^baseline_icp:[0-9a-f]{64}$'
       OR p_authorization->>'action_idempotency_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'action_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_authorization->'action_sequence') IS DISTINCT FROM 'number'
       OR (p_authorization->>'action_sequence') !~ '^[0-9]{1,4}$'
       OR (p_authorization->>'action_sequence')::INTEGER NOT BETWEEN 0 AND 9999
       OR p_authorization->>'action_type' NOT IN (
            'normalize_icp', 'execute_candidate_tool', 'verify_company',
            'execute_intent_tool', 'verify_intent',
            'execute_contact_tool', 'verify_contact'
       )
       OR p_authorization->>'tool_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_authorization->>'binding_contract_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'request_fingerprint_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'request_body_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_authorization->'call_cap') IS DISTINCT FROM 'number'
       OR (p_authorization->>'call_cap') !~ '^[0-9]{1,6}$'
       OR (p_authorization->>'call_cap')::INTEGER NOT BETWEEN 0 AND 100000
       OR pg_catalog.jsonb_typeof(p_authorization->'credit_cap_microunits') IS DISTINCT FROM 'number'
       OR (p_authorization->>'credit_cap_microunits') !~ '^[0-9]{1,9}$'
       OR (p_authorization->>'credit_cap_microunits')::BIGINT NOT BETWEEN 0 AND 100000000
       OR pg_catalog.jsonb_typeof(p_authorization->'timeout_ms') IS DISTINCT FROM 'number'
       OR (p_authorization->>'timeout_ms') !~ '^(0|[1-9][0-9]{0,5})$'
       OR (p_authorization->>'timeout_ms')::INTEGER NOT BETWEEN 0 AND 900000
       OR p_authorization->>'protected_job_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_authorization->>'protected_request_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'lease_holder_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'expected_frontier_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR (
            p_authorization->>'action_type' IN (
                'verify_company', 'verify_intent', 'verify_contact'
            )
            AND (
                (p_authorization->>'call_cap')::INTEGER <> 0
                OR (p_authorization->>'credit_cap_microunits')::BIGINT <> 0
                OR (p_authorization->>'timeout_ms')::INTEGER <> 0
            )
       )
       OR (
            p_authorization->>'action_type' IN (
                'normalize_icp', 'execute_candidate_tool',
                'execute_intent_tool', 'execute_contact_tool'
            )
            AND (
                (p_authorization->>'call_cap')::INTEGER < 1
                OR (p_authorization->>'timeout_ms')::INTEGER < 1
            )
       )
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_authorization_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_authorization, 'action_authorization'
    );
    authorization_hash :=
        public.research_lab_official_baseline_hash_v1(p_authorization);

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            (p_authorization->>'run_sha256') || ':' ||
                (p_authorization->>'unit_ref'),
            0
        )
    );
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_runs_v1 run
         WHERE run.run_sha256 = p_authorization->>'run_sha256'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_run_not_registered'
            USING ERRCODE = '23503';
    END IF;

    SELECT * INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 current_attempt
     WHERE current_attempt.attempt_key = p_authorization->>'attempt_key';
    IF FOUND THEN
        IF attempt.authorization_doc IS DISTINCT FROM p_authorization THEN
            RAISE EXCEPTION 'research_lab_official_baseline_action_authorization_conflict'
                USING ERRCODE = '23505';
        END IF;
        SELECT * INTO terminal
          FROM public.research_lab_official_baseline_action_terminals_v1 current_terminal
         WHERE current_terminal.attempt_key = attempt.attempt_key;
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
            'disposition', CASE
                WHEN FOUND THEN terminal.terminal_state
                ELSE 'reserved_existing'
            END,
            'attempt_key', attempt.attempt_key,
            'reservation_ref', attempt.reservation_ref,
            'lease_generation', attempt.lease_generation,
            'lease_expires_at', attempt.lease_expires_at,
            'protected_job_ref', attempt.protected_job_ref,
            'protected_request_sha256', attempt.protected_request_sha256,
            'attempt_sha256', CASE
                WHEN FOUND THEN terminal.terminal_attempt_sha256
                ELSE attempt.authorization_sha256
            END
        );
    END IF;

    IF EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_unit_closures_v1 closure
         WHERE closure.run_sha256 = p_authorization->>'run_sha256'
           AND closure.unit_ref = p_authorization->>'unit_ref'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_already_closed'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
          JOIN public.research_lab_official_baseline_action_terminals_v1 prior_terminal
            ON prior_terminal.attempt_key = prior_attempt.attempt_key
         WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
           AND prior_attempt.unit_ref = p_authorization->>'unit_ref'
           AND prior_terminal.terminal_state = 'terminal_uncertain'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_terminal_uncertain'
            USING ERRCODE = '40003';
    END IF;

    SELECT prior_attempt.* INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
      LEFT JOIN public.research_lab_official_baseline_action_terminals_v1 prior_terminal
        ON prior_terminal.attempt_key = prior_attempt.attempt_key
     WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
       AND prior_attempt.unit_ref = p_authorization->>'unit_ref'
       AND prior_terminal.attempt_key IS NULL
     ORDER BY prior_attempt.action_sequence
     LIMIT 1;
    IF FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
            'disposition', 'inflight',
            'attempt_key', attempt.attempt_key,
            'reservation_ref', attempt.reservation_ref,
            'lease_generation', attempt.lease_generation,
            'lease_expires_at', attempt.lease_expires_at,
            'protected_job_ref', attempt.protected_job_ref,
            'protected_request_sha256', attempt.protected_request_sha256,
            'attempt_sha256', attempt.authorization_sha256
        );
    END IF;

    frontier_doc := public.research_lab_official_baseline_provider_frontier_doc_v1(
        p_authorization->>'run_sha256', p_authorization->>'unit_ref'
    );
    frontier_hash := public.research_lab_official_baseline_hash_v1(frontier_doc);
    IF p_authorization->>'expected_frontier_sha256' IS DISTINCT FROM frontier_hash
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_provider_frontier_conflict'
            USING ERRCODE = '40001';
    END IF;
    SELECT COALESCE(MAX(prior_attempt.action_sequence) + 1, 0)
      INTO expected_sequence
      FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
     WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
       AND prior_attempt.unit_ref = p_authorization->>'unit_ref';
    IF (p_authorization->>'action_sequence')::INTEGER <> expected_sequence THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_sequence_conflict'
            USING ERRCODE = '40001';
    END IF;

    reservation_ref_value := 'baseline_reservation:' ||
        pg_catalog.substr(p_authorization->>'attempt_key', 8);
    expires_at_value := pg_catalog.clock_timestamp() + pg_catalog.make_interval(
        secs => ((p_authorization->>'timeout_ms')::INTEGER + 999) / 1000 + 60
    );
    INSERT INTO public.research_lab_official_baseline_action_attempts_v1 (
        attempt_key, run_sha256, unit_ref, action_idempotency_sha256,
        action_sha256, action_sequence, action_type, tool_id,
        binding_contract_sha256, request_fingerprint_sha256,
        request_body_sha256, call_cap, credit_cap_microunits, timeout_ms,
        protected_job_ref, protected_request_sha256, lease_holder_sha256,
        expected_frontier_sha256, reservation_ref, lease_expires_at,
        authorization_sha256, authorization_doc
    ) VALUES (
        p_authorization->>'attempt_key', p_authorization->>'run_sha256',
        p_authorization->>'unit_ref',
        p_authorization->>'action_idempotency_sha256',
        p_authorization->>'action_sha256',
        (p_authorization->>'action_sequence')::INTEGER,
        p_authorization->>'action_type', p_authorization->>'tool_id',
        p_authorization->>'binding_contract_sha256',
        p_authorization->>'request_fingerprint_sha256',
        p_authorization->>'request_body_sha256',
        (p_authorization->>'call_cap')::INTEGER,
        (p_authorization->>'credit_cap_microunits')::BIGINT,
        (p_authorization->>'timeout_ms')::INTEGER,
        p_authorization->>'protected_job_ref',
        p_authorization->>'protected_request_sha256',
        p_authorization->>'lease_holder_sha256',
        p_authorization->>'expected_frontier_sha256',
        reservation_ref_value, expires_at_value,
        authorization_hash, p_authorization
    ) RETURNING * INTO attempt;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
        'disposition', 'reserved_new',
        'attempt_key', attempt.attempt_key,
        'reservation_ref', attempt.reservation_ref,
        'lease_generation', attempt.lease_generation,
        'lease_expires_at', attempt.lease_expires_at,
        'protected_job_ref', attempt.protected_job_ref,
        'protected_request_sha256', attempt.protected_request_sha256,
        'attempt_sha256', attempt.authorization_sha256
    );
END;
$reserve_action$;

COMMIT;
