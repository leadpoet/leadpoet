\set ON_ERROR_STOP on

-- Run after migrations 157/158/159 against a disposable PostgreSQL database.
-- An expired worker must not be able to close its old lease.  The request
-- must remain reclaimable under a new lease generation.
BEGIN;

DO $queue_expiry$
DECLARE
    v_experiment_hash TEXT := 'sha256:' || repeat('9', 64);
    v_request_hash TEXT := 'sha256:' || repeat('8', 64);
    first_lease_hash TEXT;
    result JSONB;
    request_doc JSONB;
BEGIN
    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash,
        experiment_id,
        spec_doc,
        receipt_execution_mode,
        allow_live_credit_spend
    ) VALUES (
        v_experiment_hash,
        'routing-queue-expiry-test',
        pg_catalog.jsonb_build_object(
            'contract_version', 'leadpoet.intent_routing_experiment:v2',
            'credit_budget', pg_catalog.jsonb_build_object()
        ),
        'fixture',
        FALSE
    );

    PERFORM public.research_lab_routing_request_execution_v2(
        v_request_hash,
        v_experiment_hash,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
            'experiment_hash', v_experiment_hash
        )
    );

    result := public.research_lab_routing_claim_execution_requests_v2(
        'queue-expiry-worker-a', 100, 30
    );
    FOR request_doc IN SELECT jsonb_array_elements(result->'requests') LOOP
        IF request_doc->>'request_hash' = v_request_hash THEN
            first_lease_hash := request_doc->>'lease_hash';
        END IF;
    END LOOP;
    IF first_lease_hash IS NULL THEN
        RAISE EXCEPTION 'queue did not return the first lease';
    END IF;

    UPDATE public.research_lab_routing_execution_request_leases_v2
       SET lease_expires_at = pg_catalog.clock_timestamp() - INTERVAL '1 second'
     WHERE research_lab_routing_execution_request_leases_v2.request_hash = v_request_hash;

    result := public.research_lab_routing_close_execution_request_lease_v2(
        v_request_hash,
        'queue-expiry-worker-a',
        first_lease_hash,
        1,
        'completed'
    );
    IF (result->>'closed')::BOOLEAN IS TRUE
       OR (result->>'stale')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'expired lease was closed instead of fenced: %', result;
    END IF;

    result := public.research_lab_routing_claim_execution_requests_v2(
        'queue-expiry-worker-b', 100, 30
    );
    first_lease_hash := NULL;
    FOR request_doc IN SELECT jsonb_array_elements(result->'requests') LOOP
        IF request_doc->>'request_hash' = v_request_hash THEN
            first_lease_hash := request_doc->>'lease_hash';
            IF (request_doc->>'lease_generation')::BIGINT <> 2 THEN
                RAISE EXCEPTION 'expired lease was not reclaimed with generation 2: %', result;
            END IF;
        END IF;
    END LOOP;
    IF first_lease_hash IS NULL THEN
        RAISE EXCEPTION 'expired lease was not reclaimed with generation 2: %', result;
    END IF;
END;
$queue_expiry$;

DO $queue_claim_binding$
DECLARE
    v_experiment_hash TEXT := 'sha256:' || repeat('1', 64);
    v_request_hash TEXT := 'sha256:' || repeat('2', 64);
    v_lease_hash TEXT;
    v_claim_key TEXT := 'sha256:' || repeat('3', 64);
    v_event_hash TEXT := 'sha256:' || repeat('4', 64);
    v_reserve_event_key TEXT := 'sha256:' || repeat('5', 64);
    v_reservation_id TEXT := 'routing-queue-v3-reservation';
    v_recovery_key TEXT;
    v_recovery_event_hash TEXT;
    v_recovery_doc JSONB;
    v_recovery_event_doc JSONB;
    v_lease_expires_at JSONB;
    v_spec JSONB := pg_catalog.jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment_v2:v2',
        'credit_budget', pg_catalog.jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', pg_catalog.jsonb_build_object('binding', 20)
        ),
        'provider_bindings', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'binding_id', 'binding',
                'provider_id', 'bloomberry_jobs',
                'tool_id', 'intent.source_add.bloomberry_jobs',
                'manifest_hash', 'sha256:' || repeat('6', 64)
            )
        ),
        'variants', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'variant_id', 'candidate',
                'binding_ids', pg_catalog.jsonb_build_array('binding')
            )
        )
    );
    v_envelope JSONB := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_execution_envelope.v2',
        'experiment_hash', v_experiment_hash,
        'binding_catalog_manifest_hash', 'sha256:' || repeat('7', 64),
        'model_binding_observation_receipt_hash', 'sha256:' || repeat('8', 64),
        'bindings', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'binding_id', 'binding',
                'provider_id', 'bloomberry_jobs',
                'tool_id', 'intent.source_add.bloomberry_jobs',
                'binding_manifest_hash', 'sha256:' || repeat('6', 64),
                'action_id', 'bloomberry_search_job_postings',
                'credit_ceiling_microunits', 20
            )
        )
    );
    v_reserve_doc JSONB := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_budget_event.v3',
        'reservation_id', v_reservation_id,
        'binding_id', 'binding',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'unit_ref', 'unit-one',
        'variant_id', 'candidate',
        'request_fingerprint', 'sha256:' || repeat('9', 64),
        'action_id', 'bloomberry_search_job_postings',
        'binding_catalog_manifest_hash', 'sha256:' || repeat('7', 64),
        'call_grant_hash', 'sha256:' || repeat('a', 64),
        'request_body_hash', 'sha256:' || repeat('b', 64)
    );
    v_result JSONB;
BEGIN
    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash, experiment_id, spec_doc,
        receipt_execution_mode, allow_live_credit_spend,
        execution_envelope_hash, execution_envelope_doc
    ) VALUES (
        v_experiment_hash, 'routing-queue-claim-binding',
        v_spec, 'measured_lab', TRUE,
        public.research_lab_routing_jsonb_hash_v2(v_envelope), v_envelope
    );
    PERFORM public.research_lab_routing_request_execution_v2(
        v_request_hash, v_experiment_hash,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
            'experiment_hash', v_experiment_hash
        )
    );
    v_result := public.research_lab_routing_claim_execution_requests_v2(
        'queue-claim-worker', 1, 30
    );
    SELECT request_doc->>'lease_hash' INTO v_lease_hash
      FROM pg_catalog.jsonb_array_elements(v_result->'requests') request_doc
     WHERE request_doc->>'request_hash' = v_request_hash;
    IF v_lease_hash IS NULL THEN
        RAISE EXCEPTION 'queue claim binding did not receive a lease';
    END IF;
    v_result := public.research_lab_routing_claim_execution_v3(
        v_request_hash, v_lease_hash, 1, 'queue-claim-worker', v_claim_key,
        30,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim.v3',
            'request_hash', v_request_hash, 'lease_hash', v_lease_hash,
            'lease_generation', 1, 'worker_ref', 'queue-claim-worker'
        ),
        v_event_hash,
        pg_catalog.jsonb_build_object('event', 'queue-claim-v3')
    );
    IF (v_result->>'claimed')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'queue claim binding did not create an execution claim: %', v_result;
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_execution_request_leases_v2 lease
         WHERE lease.request_hash = v_request_hash
           AND lease.execution_claim_key = v_claim_key
           AND lease.execution_claim_generation = 1
    ) THEN
        RAISE EXCEPTION 'queue lease was not bound exactly once';
    END IF;
    v_result := public.research_lab_routing_claim_execution_v3(
        v_request_hash, v_lease_hash, 1, 'queue-claim-worker', v_claim_key,
        30,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim.v3',
            'request_hash', v_request_hash, 'lease_hash', v_lease_hash,
            'lease_generation', 1, 'worker_ref', 'queue-claim-worker'
        ),
        v_event_hash,
        pg_catalog.jsonb_build_object('event', 'queue-claim-v3')
    );
    IF (v_result->>'idempotent')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'exact queue claim replay was not idempotent: %', v_result;
    END IF;

    v_result := public.research_lab_routing_reserve_budget_v3(
        v_reserve_event_key, v_reservation_id, v_experiment_hash, 'binding',
        v_claim_key, 1, 20, 60, v_reserve_doc
    );
    IF v_result->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.routing_budget_reservation_result.v3'
       OR (v_result->>'reserved')::BOOLEAN IS NOT TRUE
       OR (v_result->>'idempotent')::BOOLEAN IS NOT FALSE
       OR v_result->>'reservation_id' IS DISTINCT FROM v_reservation_id
       OR v_result->>'event_key' IS DISTINCT FROM v_reserve_event_key
       OR v_result->>'experiment_hash' IS DISTINCT FROM v_experiment_hash
       OR v_result->>'binding_id' IS DISTINCT FROM 'binding'
       OR v_result->>'claim_key' IS DISTINCT FROM v_claim_key
       OR (v_result->>'claim_generation')::BIGINT <> 1
       OR (v_result->>'credit_microunits')::BIGINT <> 20
       OR v_result->'lease_expires_at' IS NULL
       OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.jsonb_object_keys(v_result)
          ) <> 11
    THEN
        RAISE EXCEPTION 'new v3 reserve response is incomplete: %', v_result;
    END IF;
    v_lease_expires_at := v_result->'lease_expires_at';
    v_result := public.research_lab_routing_reserve_budget_v3(
        v_reserve_event_key, v_reservation_id, v_experiment_hash, 'binding',
        v_claim_key, 1, 20, 60, v_reserve_doc
    );
    IF (v_result->>'idempotent')::BOOLEAN IS NOT TRUE
       OR v_result->'lease_expires_at' IS DISTINCT FROM v_lease_expires_at
       OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.jsonb_object_keys(v_result)
          ) <> 11
    THEN
        RAISE EXCEPTION 'exact v3 reserve replay changed its proof: %', v_result;
    END IF;

    UPDATE public.research_lab_routing_execution_request_leases_v2
       SET lease_expires_at = pg_catalog.clock_timestamp() - INTERVAL '1 second'
     WHERE request_hash = v_request_hash;
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v3(
            'sha256:' || repeat('c', 64), v_experiment_hash, 'run_started',
            v_claim_key, 1, pg_catalog.jsonb_build_object('event', 'stale-queue')
        );
        RAISE EXCEPTION 'live claim with stale queue lease passed the fence';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;

    v_result := public.research_lab_routing_claim_execution_requests_v2(
        'queue-recovery-worker', 100, 30
    );
    IF EXISTS (
        SELECT 1
          FROM pg_catalog.jsonb_array_elements(v_result->'requests') request_doc
         WHERE request_doc->>'request_hash' = v_request_hash
    ) THEN
        RAISE EXCEPTION 'bound queue request was reclaimed after expiry: %', v_result;
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_execution_request_leases_v2 lease
         WHERE lease.request_hash = v_request_hash
           AND lease.lease_state = 'recovered'
           AND lease.close_reason = 'recovered'
           AND lease.lease_expires_at IS NULL
           AND lease.execution_claim_key = v_claim_key
           AND lease.execution_claim_generation = 1
    ) THEN
        RAISE EXCEPTION 'bound stale queue lease was not terminally recovered';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiment_claims_v3 claim
         WHERE claim.experiment_hash = v_experiment_hash
           AND claim.claim_state = 'recovered'
           AND claim.claim_generation = 2
    ) THEN
        RAISE EXCEPTION 'stale product claim was not terminally recovered';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_budget_events_v2 budget
         WHERE budget.reservation_id = v_reservation_id
           AND budget.event_type = 'uncertain'
           AND budget.credit_microunits = 20
           AND budget.event_doc->>'reason_code'
                = 'claim_recovered_unknown_billing'
    ) THEN
        RAISE EXCEPTION 'open reservation was not retained at full uncertain ceiling';
    END IF;

    v_recovery_key := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_queue_recovery.v3',
            'experiment_hash', v_experiment_hash,
            'request_hash', v_request_hash,
            'lease_hash', v_lease_hash,
            'lease_generation', 1,
            'stale_claim_key', v_claim_key,
            'stale_claim_generation', 1
        )
    );
    v_recovery_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_claim_recovery.v3',
        'worker_ref', 'queue-recovery-worker',
        'stale_claim_key', v_claim_key,
        'stale_claim_generation', 1,
        'request_hash', v_request_hash,
        'lease_hash', v_lease_hash,
        'lease_generation', 1,
        'reason_code', 'queue_lease_expired'
    );
    v_recovery_event_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_event.v2',
        'experiment_hash', v_experiment_hash,
        'recovery_key', v_recovery_key,
        'worker_ref', 'queue-recovery-worker',
        'event_type', 'claim_recovered',
        'reason_code', 'queue_lease_expired'
    );
    v_recovery_event_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'event_type', 'claim_recovered',
            'event_doc', v_recovery_event_doc
        )
    );
    v_result := public.research_lab_routing_recover_claim_v3(
        v_experiment_hash, v_recovery_key, 'queue-recovery-worker',
        v_recovery_doc, v_recovery_event_hash, v_recovery_event_doc
    );
    IF (v_result->>'idempotent')::BOOLEAN IS NOT TRUE
       OR (v_result->>'terminal')::BOOLEAN IS NOT TRUE
       OR v_result->>'billing_state' IS DISTINCT FROM 'uncertain'
    THEN
        RAISE EXCEPTION 'exact terminal recovery replay was not idempotent: %', v_result;
    END IF;
END;
$queue_claim_binding$;

ROLLBACK;
