\set ON_ERROR_STOP on

-- Run only against a disposable PostgreSQL database after migration 157.
BEGIN;
DO $test$
DECLARE
    exp TEXT := 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    claim TEXT := 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
    token TEXT := 'claim-nonce-routing-test-cccccccccccccccccccc';
    receipt TEXT := 'routing_evaluation_v2:dddddddddddddddd';
    spec JSONB := jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment:v2',
        'experiment_id', 'routing-test',
        'receipt_execution_mode', 'measured_lab',
        'allow_live_credit_spend', TRUE,
        'credit_budget', jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', jsonb_build_object('binding', 20)
        ),
        'provider_bindings', jsonb_build_array(
            jsonb_build_object('binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs', 'action_id', 'intent.source_add.bloomberry_jobs')
        ),
        'variants', jsonb_build_array(
            jsonb_build_object('variant_id', 'candidate', 'binding_ids', jsonb_build_array('binding'))
        )
    );
BEGIN
    PERFORM public.research_lab_routing_submit_experiment_v2(
        exp, 'routing-test', spec, 'measured_lab', TRUE,
        'sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
        jsonb_build_object('event', 'submit')
    );
    PERFORM public.research_lab_routing_claim_experiment_v2(
        exp, claim, token, 'worker', 5, jsonb_build_object('claim', 'one'),
        'sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        jsonb_build_object('event', 'claim')
    );
    PERFORM public.research_lab_routing_renew_claim_v2(
        'sha256:abababababababababababababababababababababababababababababababab',
        exp, claim, 1, token, 5, jsonb_build_object('event', 'heartbeat')
    );
    PERFORM public.research_lab_routing_renew_claim_v2(
        'sha256:abababababababababababababababababababababababababababababababab',
        exp, claim, 1, token, 5, jsonb_build_object('event', 'heartbeat')
    );
    PERFORM public.research_lab_routing_reserve_budget_v2(
        'sha256:1111111111111111111111111111111111111111111111111111111111111111',
        'reservation-one', exp, 'binding', claim, 1, token, 10, 5,
        jsonb_build_object(
            'event', 'reserve', 'reservation_id', 'reservation-one',
            'binding_id', 'binding', 'unit_ref', 'unit-one',
            'variant_id', 'candidate',
            'request_fingerprint', 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1',
            'action_id', 'intent.source_add.bloomberry_jobs'
        )
    );
    -- A duplicate reservation identity cannot consume budget twice.
    BEGIN
        PERFORM public.research_lab_routing_reserve_budget_v2(
            'sha256:2222222222222222222222222222222222222222222222222222222222222222',
            'reservation-one', exp, 'binding', claim, 1, token, 10, 5,
            jsonb_build_object(
                'event', 'duplicate-reserve', 'reservation_id', 'reservation-one',
                'binding_id', 'binding', 'unit_ref', 'unit-one',
                'variant_id', 'candidate',
                'request_fingerprint', 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1',
                'action_id', 'intent.source_add.bloomberry_jobs'
            )
        );
        RAISE EXCEPTION 'duplicate reservation unexpectedly succeeded';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
    -- The old lifecycle RPC is not executable by the service role; only the
    -- claim-fenced RPC may append worker execution events.
    IF has_function_privilege(
        'service_role',
        'public.research_lab_routing_append_event_v2(text,text,text,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'unfenced event RPC remains executable';
    END IF;
    IF NOT has_function_privilege(
        'service_role',
        'public.research_lab_routing_append_fenced_event_v2(text,text,text,text,bigint,text,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'fenced event RPC is unavailable';
    END IF;
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:1212121212121212121212121212121212121212121212121212121212121212',
            exp, 'run_failed', claim, 1,
            'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
            jsonb_build_object('event', 'commitment-replay')
        );
        RAISE EXCEPTION 'stored claim commitment was accepted as bearer';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    PERFORM public.research_lab_routing_append_fenced_event_v2(
        'sha256:3333333333333333333333333333333333333333333333333333333333333333',
        exp, 'run_started', claim, 1, token, jsonb_build_object('event', 'run-started')
    );
    -- Empty evidence cannot create an evaluation that later promotes.
    BEGIN
        PERFORM public.research_lab_routing_append_evaluation_v2(
            receipt, exp,
            'sha256:4444444444444444444444444444444444444444444444444444444444444444',
            'candidate', claim, 1, token,
            jsonb_build_object(
                'receipt_id', receipt,
                'experiment_hash', exp,
                'selected_variant_id', 'candidate',
                'decision_receipt_refs', jsonb_build_array(),
                'provider_receipt_refs', jsonb_build_array()
            )
        );
        RAISE EXCEPTION 'empty evaluation unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    -- A caller-created reconciliation document cannot promote when there is
    -- no exact persisted evaluation and attested receipt chain.
    BEGIN
        PERFORM public.research_lab_routing_promote_v2(
            'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab',
            exp, receipt,
            'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc',
            'candidate',
            jsonb_build_object(
                'reconciled', TRUE,
                'experiment_hash', exp,
                'evaluation_receipt_id', receipt,
                'evaluation_hash', 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc',
                'selected_variant_id', 'candidate',
                'authority_receipt_hash', 'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccd',
                'authority_input_root', 'sha256:ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddde',
                'authority_output_root', 'sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeef'
            ),
            'sha256:fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0',
            jsonb_build_object('event', 'forged-promotion')
        );
        RAISE EXCEPTION 'fabricated promotion unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    BEGIN
        PERFORM public.research_lab_routing_reject_secret_doc_v2(
            jsonb_build_object('client_secret', 'redacted-not-allowed'), 'secret-test'
        );
        RAISE EXCEPTION 'secret document unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    IF has_table_privilege('service_role', 'public.research_lab_routing_budget_events_v2', 'TRUNCATE') THEN
        RAISE EXCEPTION 'service role can truncate routing budget ledger';
    END IF;
    PERFORM public.research_lab_routing_close_claim_v2(
        'sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd',
        exp, claim, 1, token, 'completed', jsonb_build_object('event', 'close')
    );
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:dededededededededededededededededededededededededededededededede',
            exp, 'run_failed', claim, 1, token, jsonb_build_object('event', 'after-close')
        );
        RAISE EXCEPTION 'closed claim unexpectedly wrote an execution event';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
END;
$test$;

DO $attempt_chain$
DECLARE
    exp TEXT := 'sha256:abababababababababababababababababababababababababababababababa1';
    claim TEXT := 'sha256:abababababababababababababababababababababababababababababababa2';
    token TEXT := 'attempt-chain-claim-nonce-aaaaaaaaaaaaaaaaaaaa';
    reservation TEXT := 'attempt-chain-reservation';
    fingerprint TEXT := 'sha256:abababababababababababababababababababababababababababababababa3';
    action TEXT := 'intent.source_add.bloomberry_jobs';
    spec JSONB := jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment:v2',
        'experiment_id', 'routing-attempt-chain',
        'receipt_execution_mode', 'measured_lab',
        'allow_live_credit_spend', TRUE,
        'credit_budget', jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', jsonb_build_object('binding', 20)
        ),
        'provider_bindings', jsonb_build_array(
            jsonb_build_object(
                'binding_id', 'binding',
                'tool_id', 'intent.source_add.bloomberry_jobs',
                'action_id', 'intent.source_add.bloomberry_jobs'
            )
        ),
        'variants', jsonb_build_array(
            jsonb_build_object('variant_id', 'candidate', 'binding_ids', jsonb_build_array('binding'))
        )
    );
    attempt_doc JSONB;
BEGIN
    attempt_doc := jsonb_build_object(
        'binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs',
        'action_id', action, 'variant_id', 'candidate', 'unit_ref', 'unit-chain',
        'reservation_id', reservation, 'request_fingerprint', fingerprint,
        'execution_mode', 'measured_lab'
    );
    PERFORM public.research_lab_routing_submit_experiment_v2(
        exp, 'routing-attempt-chain', spec, 'measured_lab', TRUE,
        'sha256:abababababababababababababababababababababababababababababababa4',
        jsonb_build_object('event', 'submit')
    );
    PERFORM public.research_lab_routing_claim_experiment_v2(
        exp, claim, token, 'worker', 5, jsonb_build_object('claim', 'attempt'),
        'sha256:abababababababababababababababababababababababababababababababa5',
        jsonb_build_object('event', 'claim')
    );
    PERFORM public.research_lab_routing_reserve_budget_v2(
        'sha256:abababababababababababababababababababababababababababababababa6',
        reservation, exp, 'binding', claim, 1, token, 5, 5,
        jsonb_build_object(
            'event', 'reserve', 'reservation_id', reservation,
            'binding_id', 'binding', 'unit_ref', 'unit-chain',
            'variant_id', 'candidate', 'request_fingerprint', fingerprint,
            'action_id', action
        )
    );
    BEGIN
        PERFORM public.research_lab_routing_append_provider_attempt_v2(
            'sha256:abababababababababababababababababababababababababababababababa7',
            exp, 'provider_receipt:abababababababab', 'binding',
            'intent.source_add.bloomberry_jobs', 'candidate', 'unit-chain',
            'wrong-reservation', action, claim, 1, token, fingerprint,
            'verified', 5, 10, 'measured_lab', 'known', 5, attempt_doc
        );
        RAISE EXCEPTION 'attempt without exact reservation unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    PERFORM public.research_lab_routing_append_provider_attempt_v2(
        'sha256:abababababababababababababababababababababababababababababababa7',
        exp, 'provider_receipt:abababababababab', 'binding',
        'intent.source_add.bloomberry_jobs', 'candidate', 'unit-chain',
        reservation, action, claim, 1, token, fingerprint,
        'verified', 5, 10, 'measured_lab', 'known', 5, attempt_doc
    );
    PERFORM public.research_lab_routing_settle_budget_v2(
        'sha256:abababababababababababababababababababababababababababababababa8',
        reservation, 'sha256:abababababababababababababababababababababababababababababababa7',
        claim, 1, token, jsonb_build_object('event', 'settle')
    );
END;
$attempt_chain$;

DO $fencing$
DECLARE
    exp TEXT := 'sha256:5555555555555555555555555555555555555555555555555555555555555555';
    old_claim TEXT := 'sha256:6666666666666666666666666666666666666666666666666666666666666666';
    old_token TEXT := 'old-claim-nonce-777777777777777777777777777777';
    recovery TEXT := 'sha256:8888888888888888888888888888888888888888888888888888888888888888';
    new_claim TEXT := 'sha256:9999999999999999999999999999999999999999999999999999999999999999';
    new_token TEXT := 'new-claim-nonce-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    spec JSONB := jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment:v2',
        'experiment_id', 'routing-fencing-test',
        'receipt_execution_mode', 'fixture',
        'allow_live_credit_spend', FALSE,
        'credit_budget', jsonb_build_object(
            'total_credit_microunits', 10,
            'provider_credit_ceilings', jsonb_build_object('binding', 10)
        ),
        'provider_bindings', jsonb_build_array(
            jsonb_build_object('binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs', 'action_id', 'intent.source_add.bloomberry_jobs')
        ),
        'variants', jsonb_build_array(
            jsonb_build_object('variant_id', 'candidate', 'binding_ids', jsonb_build_array('binding'))
        )
    );
BEGIN
    PERFORM public.research_lab_routing_submit_experiment_v2(
        exp, 'routing-fencing-test', spec, 'fixture', FALSE,
        'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1',
        jsonb_build_object('event', 'submit')
    );
    PERFORM public.research_lab_routing_claim_experiment_v2(
        exp, old_claim, old_token, 'worker', 1, jsonb_build_object('claim', 'old'),
        'sha256:ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc1',
        jsonb_build_object('event', 'claim')
    );
    -- A non-live fixture experiment cannot reserve a provider budget.
    BEGIN
        PERFORM public.research_lab_routing_reserve_budget_v2(
            'sha256:ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd1',
            'fixture-reservation', exp, 'binding', old_claim, 1, old_token, 1, 1,
            jsonb_build_object(
                'event', 'reserve', 'reservation_id', 'fixture-reservation',
                'binding_id', 'binding', 'unit_ref', 'unit-fixture',
                'variant_id', 'candidate',
                'request_fingerprint', 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1',
                'action_id', 'intent.source_add.bloomberry_jobs'
            )
        );
        RAISE EXCEPTION 'fixture budget reservation unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    PERFORM pg_sleep(1.1);
    PERFORM public.research_lab_routing_recover_claim_v2(
        exp, recovery, 'recovery-worker', jsonb_build_object('recovery', 'expired'),
        'sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1',
        jsonb_build_object('event', 'recovered')
    );
    -- Recovery itself is append-only and idempotent, including after a
    -- client loses the first response.
    PERFORM public.research_lab_routing_recover_claim_v2(
        exp, recovery, 'recovery-worker', jsonb_build_object('recovery', 'expired'),
        'sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1',
        jsonb_build_object('event', 'recovered')
    );
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1',
            exp, 'run_failed', old_claim, 1, old_token, jsonb_build_object('event', 'stale')
        );
        RAISE EXCEPTION 'stale claim unexpectedly wrote an event';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
    PERFORM public.research_lab_routing_claim_experiment_v2(
        exp, new_claim, new_token, 'worker', 5, jsonb_build_object('claim', 'new'),
        'sha256:1111111111111111111111111111111111111111111111111111111111111112',
        jsonb_build_object('event', 'new-claim')
    );
    PERFORM public.research_lab_routing_append_fenced_event_v2(
        'sha256:2222222222222222222222222222222222222222222222222222222222222222',
        exp, 'run_started', new_claim, 3, new_token, jsonb_build_object('event', 'fresh')
    );
END;
$fencing$;

DO $recovery$
DECLARE
    exp TEXT := 'sha256:3333333333333333333333333333333333333333333333333333333333333334';
    claim TEXT := 'sha256:4444444444444444444444444444444444444444444444444444444444444445';
    token TEXT := 'recovery-claim-nonce-55555555555555555555555555';
    spec JSONB := jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment:v2',
        'experiment_id', 'routing-reservation-recovery',
        'receipt_execution_mode', 'measured_lab',
        'allow_live_credit_spend', TRUE,
        'credit_budget', jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', jsonb_build_object('binding', 20)
        ),
        'provider_bindings', jsonb_build_array(
            jsonb_build_object('binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs', 'action_id', 'intent.source_add.bloomberry_jobs')
        ),
        'variants', jsonb_build_array(
            jsonb_build_object('variant_id', 'candidate', 'binding_ids', jsonb_build_array('binding'))
        )
    );
BEGIN
    PERFORM public.research_lab_routing_submit_experiment_v2(
        exp, 'routing-reservation-recovery', spec, 'measured_lab', TRUE,
        'sha256:6666666666666666666666666666666666666666666666666666666666666667',
        jsonb_build_object('event', 'submit')
    );
    PERFORM public.research_lab_routing_claim_experiment_v2(
        exp, claim, token, 'worker', 5, jsonb_build_object('claim', 'one'),
        'sha256:7777777777777777777777777777777777777777777777777777777777777778',
        jsonb_build_object('event', 'claim')
    );
    PERFORM public.research_lab_routing_reserve_budget_v2(
        'sha256:8888888888888888888888888888888888888888888888888888888888888889',
        'expired-reservation', exp, 'binding', claim, 1, token, 10, 1,
        jsonb_build_object(
            'event', 'reserve', 'reservation_id', 'expired-reservation',
            'binding_id', 'binding', 'unit_ref', 'unit-recovery',
            'variant_id', 'candidate',
            'request_fingerprint', 'sha256:ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc1',
            'action_id', 'intent.source_add.bloomberry_jobs'
        )
    );
    PERFORM pg_sleep(1.1);
    BEGIN
        PERFORM public.research_lab_routing_reserve_budget_v2(
            'sha256:8888888888888888888888888888888888888888888888888888888888888889',
            'expired-reservation', exp, 'binding', claim, 1, token, 10, 1,
            jsonb_build_object(
                'event', 'reserve', 'reservation_id', 'expired-reservation',
                'binding_id', 'binding', 'unit_ref', 'unit-recovery',
                'variant_id', 'candidate',
                'request_fingerprint', 'sha256:ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc1',
                'action_id', 'intent.source_add.bloomberry_jobs'
            )
        );
        RAISE EXCEPTION 'expired reservation unexpectedly reopened';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
    PERFORM public.research_lab_routing_mark_budget_uncertain_v2(
        'sha256:9999999999999999999999999999999999999999999999999999999999999998',
        'expired-reservation', claim, 1, token, jsonb_build_object('event', 'uncertain')
    );
END;
$recovery$;

-- Both a pre-dispatch crash and a crash after the durable dispatch marker are
-- unknown billing states.  Neither may reopen budget for a retry: the fresh
-- claim converts the expired full reservation to `uncertain` and the total
-- budget remains consumed until an independent broker settlement exists.
DO $unknown_crash_recovery$
DECLARE
    dispatch_started BOOLEAN;
    exp TEXT;
    old_claim TEXT;
    old_token TEXT;
    recovery TEXT;
    fresh_claim TEXT;
    fresh_token TEXT;
    reservation TEXT;
    listing JSONB;
    spec JSONB;
BEGIN
    FOR dispatch_started IN SELECT FALSE UNION ALL SELECT TRUE LOOP
        exp := 'sha256:' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a1' ELSE 'b1' END;
        old_claim := 'sha256:' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a2' ELSE 'b2' END;
        old_token := 'old-crash-claim-nonce-' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a3' ELSE 'b3' END;
        recovery := 'sha256:' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a4' ELSE 'b4' END;
        fresh_claim := 'sha256:' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a5' ELSE 'b5' END;
        fresh_token := 'fresh-crash-claim-nonce-' || repeat('0', 62)
            || CASE WHEN dispatch_started THEN 'a6' ELSE 'b6' END;
        reservation := CASE WHEN dispatch_started THEN 'post-dispatch-reservation' ELSE 'pre-dispatch-reservation' END;
        spec := jsonb_build_object(
            'contract_version', 'leadpoet.intent_routing_experiment:v2',
            'experiment_id', CASE WHEN dispatch_started THEN 'routing-post-dispatch' ELSE 'routing-pre-dispatch' END,
            'receipt_execution_mode', 'measured_lab',
            'allow_live_credit_spend', TRUE,
            'credit_budget', jsonb_build_object(
                'total_credit_microunits', 10,
                'provider_credit_ceilings', jsonb_build_object('binding', 10)
            ),
            'provider_bindings', jsonb_build_array(
                jsonb_build_object('binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs', 'action_id', 'intent.source_add.bloomberry_jobs')
            ),
            'variants', jsonb_build_array(
                jsonb_build_object('variant_id', 'candidate', 'binding_ids', jsonb_build_array('binding'))
            )
        );
        PERFORM public.research_lab_routing_submit_experiment_v2(
            exp, spec->>'experiment_id', spec, 'measured_lab', TRUE,
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'a7' ELSE 'b7' END,
            jsonb_build_object('event', 'submit')
        );
        PERFORM public.research_lab_routing_claim_experiment_v2(
            exp, old_claim, old_token, 'worker', 1, jsonb_build_object('claim', 'old'),
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'a8' ELSE 'b8' END,
            jsonb_build_object('event', 'claim')
        );
        PERFORM public.research_lab_routing_reserve_budget_v2(
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'a9' ELSE 'b9' END,
            reservation, exp, 'binding', old_claim, 1, old_token, 10, 1,
            jsonb_build_object(
                'event', 'reserve', 'reservation_id', reservation,
                'binding_id', 'binding', 'unit_ref', 'unit-crash',
                'variant_id', 'candidate',
                'request_fingerprint', 'sha256:' || repeat('0', 62)
                    || CASE WHEN dispatch_started THEN 'a7' ELSE 'b7' END,
                'action_id', 'intent.source_add.bloomberry_jobs'
            )
        );
        IF dispatch_started THEN
            PERFORM public.research_lab_routing_append_fenced_event_v2(
                'sha256:' || repeat('0', 62) || 'aa', exp, 'provider_dispatch_started',
                old_claim, 1, old_token,
                jsonb_build_object(
                    'reservation_id', reservation,
                    'binding_id', 'binding',
                    'request_fingerprint', 'sha256:' || repeat('0', 62) || 'af'
                )
            );
        END IF;
        PERFORM pg_sleep(1.1);
        PERFORM public.research_lab_routing_recover_claim_v2(
            exp, recovery, 'recovery-worker', jsonb_build_object('recovery', 'expired'),
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'ab' ELSE 'bb' END,
            jsonb_build_object('event', 'recovered')
        );
        PERFORM public.research_lab_routing_claim_experiment_v2(
            exp, fresh_claim, fresh_token, 'worker', 5, jsonb_build_object('claim', 'fresh'),
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'ac' ELSE 'bc' END,
            jsonb_build_object('event', 'fresh')
        );
        SELECT public.research_lab_routing_list_unresolved_budget_reservations_v2(
            exp, fresh_claim, 3, fresh_token
        ) INTO listing;
        IF jsonb_array_length(listing->'reservations') <> 1
           OR (listing->'reservations'->0->>'event_type') IS DISTINCT FROM 'reserve'
           OR (listing->'reservations'->0->>'lease_expired') IS DISTINCT FROM 'true'
           OR (listing->'reservations'->0->>'dispatch_started') IS DISTINCT FROM dispatch_started::TEXT
        THEN
            RAISE EXCEPTION 'crash recovery listing is not exact: %', listing;
        END IF;
        PERFORM public.research_lab_routing_mark_budget_uncertain_v2(
            'sha256:' || repeat('0', 62)
                || CASE WHEN dispatch_started THEN 'ad' ELSE 'bd' END,
            reservation, fresh_claim, 3, fresh_token,
            jsonb_build_object('event', 'unknown-crash', 'billing_state', 'uncertain')
        );
        IF NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_budget_events_v2
            WHERE experiment_hash = exp
              AND reservation_id = reservation
              AND event_type = 'uncertain'
              AND credit_microunits = 10
        ) THEN
            RAISE EXCEPTION 'expired reservation was not retained at full uncertain cost';
        END IF;
        BEGIN
            PERFORM public.research_lab_routing_reserve_budget_v2(
                'sha256:' || repeat('0', 62)
                    || CASE WHEN dispatch_started THEN 'ae' ELSE 'be' END,
                reservation || '-retry', exp, 'binding', fresh_claim, 3, fresh_token, 1, 1,
                jsonb_build_object(
                    'event', 'retry', 'reservation_id', reservation || '-retry',
                    'binding_id', 'binding', 'unit_ref', 'unit-crash',
                    'variant_id', 'candidate',
                    'request_fingerprint', 'sha256:' || repeat('0', 62)
                        || CASE WHEN dispatch_started THEN 'a7' ELSE 'b7' END,
                    'action_id', 'intent.source_add.bloomberry_jobs'
                )
            );
            RAISE EXCEPTION 'retry unexpectedly fit after unknown crash billing';
        EXCEPTION WHEN unique_violation THEN NULL;
        END;
    END LOOP;
END;
$unknown_crash_recovery$;

-- RLS plus grants leave service_role able to call reviewed SECURITY DEFINER
-- RPCs, but unable to inject a row around their immutable checks.
SET LOCAL ROLE service_role;
DO $rls$
BEGIN
    BEGIN
        INSERT INTO public.research_lab_routing_experiments_v2 (
            experiment_hash, experiment_id, spec_doc,
            receipt_execution_mode, allow_live_credit_spend
        ) VALUES (
            'sha256:fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff2',
            'direct-write', jsonb_build_object('contract_version', 'test', 'credit_budget', jsonb_build_object()),
            'fixture', FALSE
        );
        RAISE EXCEPTION 'service role direct write unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
END;
$rls$;
RESET ROLE;
ROLLBACK;
