\set ON_ERROR_STOP on

-- Run after migrations 157/158/159/160 against a disposable PostgreSQL
-- database.  This is deliberately database-native: the failure RPC must be
-- proven independently of the Python repository adapter.

BEGIN;

CREATE OR REPLACE FUNCTION pg_temp.routing_test_failure_doc(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_binding_version TEXT,
    p_source_lineage_id TEXT,
    p_request_fingerprint TEXT,
    p_execution_mode TEXT,
    p_evidence_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $failure_doc$
DECLARE
    provider_receipt JSONB;
    failure_key TEXT;
BEGIN
    provider_receipt := pg_catalog.jsonb_build_object(
        'receipt_ref', '',
        'binding_id', p_binding_id,
        'tool_id', p_tool_id,
        'binding_version', p_binding_version,
        'source_lineage_id', p_source_lineage_id,
        'unit_ref', p_unit_ref,
        'request_fingerprint', p_request_fingerprint,
        'outcome', 'adapter_failure',
        'evidence_hash', p_evidence_hash,
        'credit_microunits', 0,
        'latency_ms', 0,
        'execution_mode', p_execution_mode
    );
    provider_receipt := pg_catalog.jsonb_set(
        provider_receipt,
        '{receipt_ref}',
        pg_catalog.to_jsonb(
            'provider_receipt:' || pg_catalog.substr(
                public.research_lab_routing_jsonb_hash_v2(
                    provider_receipt - 'receipt_ref'
                ), 8, 16
            )
        ),
        TRUE
    );
    failure_key := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'contract_version', 'leadpoet.provider_receipt_key:v2',
            'tool_id', p_tool_id,
            'binding_version', p_binding_version,
            'request_fingerprint', p_request_fingerprint
        )
    );
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_adapter_failure.v3',
        'failure_key', failure_key,
        'experiment_hash', p_experiment_hash,
        'binding_id', p_binding_id,
        'tool_id', p_tool_id,
        'variant_id', p_variant_id,
        'unit_ref', p_unit_ref,
        'claim_key', p_claim_key,
        'claim_generation', p_claim_generation,
        'request_fingerprint', p_request_fingerprint,
        'outcome', 'adapter_failure',
        'credit_microunits', 0,
        'latency_ms', 0,
        'execution_mode', p_execution_mode,
        'pre_dispatch', TRUE,
        'provider_receipt', provider_receipt
    );
END;
$failure_doc$;

DO $adapter_failure_authority$
DECLARE
    exp TEXT := 'sha256:' || repeat('a', 64);
    exp_two TEXT := 'sha256:' || repeat('b', 64);
    request_hash TEXT := 'sha256:' || repeat('c', 64);
    request_hash_two TEXT := 'sha256:' || repeat('d', 64);
    lease_hash TEXT := 'sha256:' || repeat('e', 64);
    lease_hash_two TEXT := 'sha256:' || repeat('f', 64);
    claim_key TEXT := 'sha256:' || repeat('1', 64);
    claim_key_two TEXT := 'sha256:' || repeat('2', 64);
    event_hash TEXT := 'sha256:' || repeat('3', 64);
    event_hash_two TEXT := 'sha256:' || repeat('4', 64);
    binding_id TEXT := 'binding';
    tool_id TEXT := 'intent.source_add.bloomberry_jobs';
    mode TEXT := 'fixture';
    spec JSONB;
    doc JSONB;
    doc_two JSONB;
    receipt JSONB;
    receipt_two JSONB;
    failure_key TEXT;
    failure_key_two TEXT;
    failure_ref TEXT;
    failure_ref_two TEXT;
    result JSONB;
    bad_doc JSONB;
    conflict_doc JSONB;
    decision_id TEXT := 'routing_decision:' || repeat('5', 16);
    evaluation_id TEXT := 'routing_evaluation_v2:' || repeat('6', 16);
    evaluation_doc JSONB;
    update_rejected BOOLEAN;
    delete_rejected BOOLEAN;
BEGIN
    spec := pg_catalog.jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment_v2:v2',
        'experiment_id', 'adapter-failure-native',
        'receipt_execution_mode', mode,
        'allow_live_credit_spend', FALSE,
        'credit_budget', pg_catalog.jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', pg_catalog.jsonb_build_object(binding_id, 20)
        ),
        'provider_bindings', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'binding_id', binding_id,
                'provider_id', 'bloomberry_jobs',
                'tool_id', tool_id,
                'manifest_hash', 'sha256:' || repeat('7', 64)
            )
        ),
        'variants', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'variant_id', 'candidate',
                'binding_ids', pg_catalog.jsonb_build_array(binding_id)
            )
        )
    );

    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash, experiment_id, spec_doc, receipt_execution_mode,
        allow_live_credit_spend
    ) VALUES (exp, 'adapter-failure-native', spec, mode, FALSE);
    INSERT INTO public.research_lab_routing_execution_requests_v2 (
        request_hash, experiment_hash, request_doc
    ) VALUES (
        request_hash, exp,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
            'experiment_hash', exp
        )
    );
    INSERT INTO public.research_lab_routing_execution_request_leases_v2 (
        request_hash, experiment_hash, lease_hash, worker_ref, lease_generation,
        lease_state, lease_expires_at, execution_claim_key,
        execution_claim_generation
    ) VALUES (
        request_hash, exp, lease_hash, 'adapter-failure-test', 1, 'claimed',
        pg_catalog.clock_timestamp() + INTERVAL '1 hour', claim_key, 1
    );
    INSERT INTO public.research_lab_routing_experiment_claims_v3 (
        claim_key, experiment_hash, request_hash, lease_hash, lease_generation,
        claim_generation, worker_ref, claim_state, lease_expires_at, claim_doc
    ) VALUES (
        claim_key, exp, request_hash, lease_hash, 1, 1, 'adapter-failure-test',
        'claimed', pg_catalog.clock_timestamp() + INTERVAL '1 hour',
        pg_catalog.jsonb_build_object(
            'request_hash', request_hash,
            'lease_hash', lease_hash,
            'lease_generation', 1,
            'worker_ref', 'adapter-failure-test'
        )
    );

    doc := pg_temp.routing_test_failure_doc(
        exp, claim_key, 1, 'candidate', 'unit-one', binding_id, tool_id,
        'adapter-v1', 'lineage',
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'experiment_hash', exp, 'unit_ref', 'unit-one', 'attempt', 1
            )
        ), mode,
        'sha256:' || repeat('9', 64)
    );
    failure_key := doc->>'failure_key';
    receipt := doc->'provider_receipt';
    failure_ref := receipt->>'receipt_ref';

    result := public.research_lab_routing_append_adapter_failure_v3(
        failure_key, exp, failure_ref, binding_id, tool_id, 'candidate',
        'unit-one', claim_key, 1, receipt->>'request_fingerprint', 0, mode, doc
    );
    IF result->>'failure_key' IS DISTINCT FROM failure_key
       OR (result->>'idempotent')::BOOLEAN IS NOT FALSE
    THEN
        RAISE EXCEPTION 'exact adapter failure append was not new: %', result;
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_adapter_failures_v2 failure
         WHERE failure.failure_key = doc->>'failure_key'
           AND failure.experiment_hash = exp
           AND failure.claim_key = doc->>'claim_key'
           AND failure.claim_generation = 1
           AND failure.variant_id = 'candidate'
           AND failure.unit_ref = 'unit-one'
           AND failure.outcome = 'adapter_failure'
           AND failure.credit_microunits = 0
           AND failure.failure_doc = doc
    ) THEN
        RAISE EXCEPTION 'adapter failure row did not preserve exact identity';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_provider_attempts_v2 attempt
         WHERE attempt.attempt_key = failure_key
            OR attempt.provider_receipt_ref = failure_ref
    ) THEN
        RAISE EXCEPTION 'adapter failure unexpectedly became provider attempt evidence';
    END IF;

    result := public.research_lab_routing_append_adapter_failure_v3(
        failure_key, exp, failure_ref, binding_id, tool_id, 'candidate',
        'unit-one', claim_key, 1, receipt->>'request_fingerprint', 0, mode, doc
    );
    IF (result->>'idempotent')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'exact adapter failure replay was not idempotent: %', result;
    END IF;

    conflict_doc := pg_catalog.jsonb_set(doc, '{variant_id}', '"other-variant"', TRUE);
    BEGIN
        PERFORM public.research_lab_routing_append_adapter_failure_v3(
            failure_key, exp, failure_ref, binding_id, tool_id, 'other-variant',
            'unit-one', claim_key, 1, receipt->>'request_fingerprint', 0, mode,
            conflict_doc
        );
        RAISE EXCEPTION 'conflicting adapter failure replay unexpectedly succeeded';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    bad_doc := pg_catalog.jsonb_set(doc, '{credit_microunits}', '1', TRUE);
    BEGIN
        PERFORM public.research_lab_routing_append_adapter_failure_v3(
            failure_key, exp, failure_ref, binding_id, tool_id, 'candidate',
            'unit-one', claim_key, 1, receipt->>'request_fingerprint', 0, mode,
            bad_doc
        );
        RAISE EXCEPTION 'non-zero-cost adapter failure unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;

    bad_doc := doc || pg_catalog.jsonb_build_object(
        'terminal_proof', pg_catalog.jsonb_build_object('status', 'succeeded')
    );
    BEGIN
        PERFORM public.research_lab_routing_append_adapter_failure_v3(
            failure_key, exp, failure_ref, binding_id, tool_id, 'candidate',
            'unit-one', claim_key, 1, receipt->>'request_fingerprint', 0, mode,
            bad_doc
        );
        RAISE EXCEPTION 'terminal proof in adapter failure unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;

    -- A dispatch marker is the durable provider boundary. A failure after it
    -- exists must not be downgraded to a zero-cost pre-dispatch receipt.
    doc_two := pg_temp.routing_test_failure_doc(
        exp, claim_key, 1, 'candidate', 'unit-two', binding_id, tool_id,
        'adapter-v1', 'lineage', 'sha256:' || repeat('a', 64), mode,
        'sha256:' || repeat('b', 64)
    );
    failure_key_two := doc_two->>'failure_key';
    receipt_two := doc_two->'provider_receipt';
    failure_ref_two := receipt_two->>'receipt_ref';
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key, claim_generation,
        event_doc
    ) VALUES (
        'sha256:' || repeat('c', 64), exp, 'provider_dispatch_started',
        claim_key, 1,
        pg_catalog.jsonb_build_object(
            'request_fingerprint', receipt_two->>'request_fingerprint',
            'reservation_id', 'dispatch-marker-reservation',
            'binding_id', binding_id
        )
    );
    BEGIN
        PERFORM public.research_lab_routing_append_adapter_failure_v3(
            failure_key_two, exp, failure_ref_two, binding_id, tool_id,
            'candidate', 'unit-two', claim_key, 1,
            receipt_two->>'request_fingerprint', 0, mode, doc_two
        );
        RAISE EXCEPTION 'dispatch-marked adapter failure unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;

    -- A key already used by the provider-attempt ledger cannot be dual-used
    -- by the failure ledger, even when its receipt body is otherwise valid.
    doc_two := pg_temp.routing_test_failure_doc(
        exp, claim_key, 1, 'candidate', 'unit-three', binding_id, tool_id,
        'adapter-v1', 'lineage', 'sha256:' || repeat('d', 64), mode,
        'sha256:' || repeat('e', 64)
    );
    failure_key_two := doc_two->>'failure_key';
    receipt_two := doc_two->'provider_receipt';
    failure_ref_two := receipt_two->>'receipt_ref';
    INSERT INTO public.research_lab_routing_provider_attempts_v2 (
        attempt_key, experiment_hash, provider_receipt_ref, binding_id, tool_id,
        variant_id, unit_ref, reservation_id, action_id,
        binding_catalog_manifest_hash, authorization_hash,
        authorization_proof_hash, request_body_hash, terminal_receipt_hash,
        protected_release_receipt_hash, admission_bundle_hash,
        terminal_provider_record_hash, terminal_billing_projection_hash,
        claim_key, claim_generation, request_fingerprint, outcome,
        credit_microunits, billing_state, authoritative_billed_credit_microunits,
        latency_ms, execution_mode, attempt_doc
    ) VALUES (
        failure_key_two, exp, 'provider_receipt:' || repeat('f', 16), binding_id,
        tool_id, 'candidate', 'unit-three', 'collision-reservation',
        'collision-action', 'sha256:' || repeat('1', 64),
        'sha256:' || repeat('2', 64), 'sha256:' || repeat('3', 64),
        'sha256:' || repeat('4', 64), 'sha256:' || repeat('5', 64),
        'sha256:' || repeat('6', 64), 'sha256:' || repeat('7', 64),
        'sha256:' || repeat('8', 64), 'sha256:' || repeat('9', 64),
        claim_key, 1, receipt_two->>'request_fingerprint', 'source_miss', 0,
        'known', 0, 0, mode, pg_catalog.jsonb_build_object('forged', TRUE)
    );
    BEGIN
        PERFORM public.research_lab_routing_append_adapter_failure_v3(
            failure_key_two, exp, failure_ref_two, binding_id, tool_id,
            'candidate', 'unit-three', claim_key, 1,
            receipt_two->>'request_fingerprint', 0, mode, doc_two
        );
        RAISE EXCEPTION 'cross-ledger failure key collision unexpectedly succeeded';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    -- Provider keys include the experiment-bound request fingerprint. Two
    -- experiments with distinct request fingerprints must not poison each
    -- other even though failure_key is globally unique.
    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash, experiment_id, spec_doc, receipt_execution_mode,
        allow_live_credit_spend
    ) VALUES (
        exp_two, 'adapter-failure-native-two',
        pg_catalog.jsonb_set(spec, '{experiment_id}', '"adapter-failure-native-two"', TRUE),
        mode, FALSE
    );
    INSERT INTO public.research_lab_routing_execution_requests_v2 (
        request_hash, experiment_hash, request_doc
    ) VALUES (
        request_hash_two, exp_two,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
            'experiment_hash', exp_two
        )
    );
    INSERT INTO public.research_lab_routing_execution_request_leases_v2 (
        request_hash, experiment_hash, lease_hash, worker_ref, lease_generation,
        lease_state, lease_expires_at, execution_claim_key,
        execution_claim_generation
    ) VALUES (
        request_hash_two, exp_two, lease_hash_two, 'adapter-failure-test-two', 1,
        'claimed', pg_catalog.clock_timestamp() + INTERVAL '1 hour',
        claim_key_two, 1
    );
    INSERT INTO public.research_lab_routing_experiment_claims_v3 (
        claim_key, experiment_hash, request_hash, lease_hash, lease_generation,
        claim_generation, worker_ref, claim_state, lease_expires_at, claim_doc
    ) VALUES (
        claim_key_two, exp_two, request_hash_two, lease_hash_two, 1, 1,
        'adapter-failure-test-two', 'claimed',
        pg_catalog.clock_timestamp() + INTERVAL '1 hour',
        pg_catalog.jsonb_build_object(
            'request_hash', request_hash_two,
            'lease_hash', lease_hash_two,
            'lease_generation', 1,
            'worker_ref', 'adapter-failure-test-two'
        )
    );
    doc_two := pg_temp.routing_test_failure_doc(
        exp_two, claim_key_two, 1, 'candidate', 'unit-one', binding_id, tool_id,
        'adapter-v1', 'lineage',
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'experiment_hash', exp_two, 'unit_ref', 'unit-one', 'attempt', 1
            )
        ), mode,
        'sha256:' || repeat('b', 64)
    );
    failure_key_two := doc_two->>'failure_key';
    receipt_two := doc_two->'provider_receipt';
    failure_ref_two := receipt_two->>'receipt_ref';
    IF failure_key_two = failure_key THEN
        RAISE EXCEPTION 'experiment-bound request fingerprints produced same failure key';
    END IF;
    PERFORM public.research_lab_routing_append_adapter_failure_v3(
        failure_key_two, exp_two, failure_ref_two, binding_id, tool_id,
        'candidate', 'unit-one', claim_key_two, 1,
        receipt_two->>'request_fingerprint', 0, mode, doc_two
    );
    IF (SELECT count(*) FROM public.research_lab_routing_adapter_failures_v2 AS failure_row
        WHERE failure_row.failure_key IN (doc->>'failure_key', failure_key_two)) <> 2 THEN
        RAISE EXCEPTION 'cross-experiment failure rows were not independently persisted';
    END IF;

    -- Evaluation/promotion provider refs are sourced from provider_attempts_v2
    -- only. A failure ref cannot be used to create an evaluation.
    INSERT INTO public.research_lab_routing_decision_receipts_v2 (
        receipt_id, experiment_hash, variant_id, unit_ref, claim_key,
        claim_generation, plan_hash, route_hash, decision_doc
    ) VALUES (
        decision_id, exp, 'candidate', 'unit-one', claim_key, 1,
        'sha256:' || repeat('1', 64), 'sha256:' || repeat('2', 64),
        pg_catalog.jsonb_build_object('provider_receipt_refs',
            pg_catalog.jsonb_build_array(failure_ref))
    );
    evaluation_doc := pg_catalog.jsonb_build_object(
        'receipt_id', evaluation_id,
        'experiment_hash', exp,
        'selected_variant_id', 'candidate',
        'decision_receipt_refs', pg_catalog.jsonb_build_array(decision_id),
        'provider_receipt_refs', pg_catalog.jsonb_build_array(failure_ref),
        'variants', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'variant_id', 'candidate', 'passed', TRUE,
                'calibration', pg_catalog.jsonb_build_object('adapter_failure_count', 0),
                'holdout', pg_catalog.jsonb_build_object('adapter_failure_count', 0)
            )
        )
    );
    BEGIN
        PERFORM public.research_lab_routing_append_evaluation_v3(
            evaluation_id, exp,
            public.research_lab_routing_jsonb_hash_v2(evaluation_doc),
            'candidate', claim_key, 1, evaluation_doc
        );
        RAISE EXCEPTION 'promotion evaluation accepted adapter failure as provider evidence';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_evaluation_receipts_v2
         WHERE receipt_id = evaluation_id
    ) THEN
        RAISE EXCEPTION 'failed provider evidence unexpectedly created evaluation row';
    END IF;

    IF NOT has_table_privilege(
        'service_role', 'public.research_lab_routing_adapter_failures_v2', 'SELECT'
    ) OR has_table_privilege(
        'service_role', 'public.research_lab_routing_adapter_failures_v2', 'INSERT'
    ) THEN
        RAISE EXCEPTION 'adapter failure table ACL is not read-only for service_role';
    END IF;
    IF NOT has_function_privilege(
        'service_role',
        'public.research_lab_routing_append_adapter_failure_v3(text,text,text,text,text,text,text,text,bigint,text,bigint,text,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'adapter failure append RPC is not executable by service_role';
    END IF;
    IF NOT (
        SELECT relrowsecurity
          FROM pg_catalog.pg_class
         WHERE oid = 'public.research_lab_routing_adapter_failures_v2'::REGCLASS
    ) THEN
        RAISE EXCEPTION 'adapter failure table RLS is disabled';
    END IF;

    update_rejected := FALSE;
    BEGIN
        UPDATE public.research_lab_routing_adapter_failures_v2 AS failure_row
           SET unit_ref = 'mutated'
         WHERE failure_row.failure_key = doc->>'failure_key';
    EXCEPTION WHEN OTHERS THEN
        update_rejected := TRUE;
    END;
    IF NOT update_rejected THEN
        RAISE EXCEPTION 'adapter failure UPDATE was not append-only rejected';
    END IF;
    delete_rejected := FALSE;
    BEGIN
        DELETE FROM public.research_lab_routing_adapter_failures_v2 AS failure_row
         WHERE failure_row.failure_key = doc->>'failure_key';
    EXCEPTION WHEN OTHERS THEN
        delete_rejected := TRUE;
    END;
    IF NOT delete_rejected THEN
        RAISE EXCEPTION 'adapter failure DELETE was not append-only rejected';
    END IF;
END;
$adapter_failure_authority$;

-- Both V3 append paths must acquire both cross-ledger identities before the
-- first existing-row or cross-ledger check. This is the database-native
-- regression guard for the race between the two independent ledgers.
DO $shared_provider_identity_lock_order$
DECLARE
    adapter_source TEXT;
    provider_source TEXT;
BEGIN
    SELECT pg_catalog.pg_get_functiondef(oid)
      INTO adapter_source
      FROM pg_catalog.pg_proc
     WHERE oid = 'public.research_lab_routing_append_adapter_failure_v3(
         text,text,text,text,text,text,text,text,bigint,text,bigint,text,jsonb
     )'::REGPROCEDURE;
    SELECT pg_catalog.pg_get_functiondef(oid)
      INTO provider_source
      FROM pg_catalog.pg_proc
     WHERE oid = 'public.research_lab_routing_append_provider_attempt_v3(
         text,text,text,text,text,text,text,text,text,text,text,text,text,text,
         text,bigint,text,text,bigint,bigint,text,text,bigint,text,text,text,
         text,text,text,text,jsonb
     )'::REGPROCEDURE;
    IF adapter_source IS NULL OR provider_source IS NULL THEN
        RAISE EXCEPTION 'V3 append RPC was not installed';
    END IF;
    IF pg_catalog.strpos(adapter_source, 'hashtextextended(p_failure_key, 0)') = 0
       OR pg_catalog.strpos(adapter_source, 'hashtextextended(p_provider_receipt_ref, 0)') = 0
       OR pg_catalog.strpos(provider_source, 'hashtextextended(p_attempt_key, 0)') = 0
       OR pg_catalog.strpos(provider_source, 'hashtextextended(p_provider_receipt_ref, 0)') = 0
       OR pg_catalog.strpos(adapter_source, 'IF failure_identity_lock <= receipt_identity_lock') = 0
       OR pg_catalog.strpos(provider_source, 'IF attempt_identity_lock <= receipt_identity_lock') = 0
       OR pg_catalog.strpos(adapter_source, 'SELECT * INTO existing')
            < pg_catalog.strpos(adapter_source, 'pg_advisory_xact_lock')
       OR pg_catalog.strpos(provider_source, 'SELECT * INTO existing')
            < pg_catalog.strpos(provider_source, 'pg_advisory_xact_lock')
    THEN
        RAISE EXCEPTION 'cross-ledger V3 identity lock ordering regressed';
    END IF;
END;
$shared_provider_identity_lock_order$;

ROLLBACK;
