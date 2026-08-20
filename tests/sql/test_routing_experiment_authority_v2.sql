\set ON_ERROR_STOP on

-- Run only against a disposable PostgreSQL database after migrations 157/158/159.
BEGIN;

CREATE OR REPLACE FUNCTION pg_temp.routing_test_spec(
    p_experiment_id TEXT,
    p_live BOOLEAN
)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $spec$
    SELECT pg_catalog.jsonb_build_object(
        'contract_version', 'leadpoet.intent_routing_experiment_v2:v2',
        'experiment_id', p_experiment_id,
        'input', pg_catalog.jsonb_build_object(
            'unit_input_set_hash', 'sha256:' || repeat('1', 64)
        ),
        'receipt_execution_mode', CASE WHEN p_live THEN 'measured_lab' ELSE 'fixture' END,
        'allow_live_credit_spend', p_live,
        'credit_budget', pg_catalog.jsonb_build_object(
            'total_credit_microunits', 20,
            'provider_credit_ceilings', pg_catalog.jsonb_build_object('binding', 20)
        ),
        'provider_bindings', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'binding_id', 'binding',
                'provider_id', 'bloomberry_jobs',
                'tool_id', 'intent.source_add.bloomberry_jobs',
                'manifest_hash', 'sha256:' || repeat('2', 64)
            )
        ),
        'variants', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'variant_id', 'candidate',
                'binding_ids', pg_catalog.jsonb_build_array('binding')
            )
        )
    )
$spec$;

CREATE OR REPLACE FUNCTION pg_temp.routing_test_envelope(
    p_spec JSONB,
    p_experiment_hash TEXT
)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $envelope$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_execution_envelope.v2',
        'experiment_hash', p_experiment_hash,
        'artifact_lineage_hash', 'sha256:' || repeat('3', 64),
        'pointer_document_hash', 'sha256:' || repeat('4', 64),
        'binding_catalog_manifest_hash', 'sha256:' || repeat('5', 64),
        'binding_catalog_version', 'catalog-test-v1',
        'unit_dataset_manifest_hash', 'sha256:' || repeat('6', 64),
        'unit_set_hash', p_spec #>> '{input,unit_input_set_hash}',
        'gold_label_manifest_hash', 'sha256:' || repeat('7', 64),
        'model_binding_observation_receipt_hash', 'sha256:' || repeat('8', 64),
        'model_binding_observation', pg_catalog.jsonb_build_object(
            'result', pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.routing_model_binding_observation.v2',
                'artifact_lineage_hash', 'sha256:' || repeat('3', 64),
                'request_root', 'sha256:' || repeat('9', 64),
                'requirements', pg_catalog.jsonb_build_array()
            ),
            'receipt', pg_catalog.jsonb_build_object(
                'receipt_hash', 'sha256:' || repeat('8', 64)
            )
        ),
        'bindings', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'binding_id', 'binding',
                'provider_id', 'bloomberry_jobs',
                'tool_id', 'intent.source_add.bloomberry_jobs',
                'binding_manifest_hash', 'sha256:' || repeat('2', 64),
                'action_id', 'bloomberry_search_job_postings',
                'compiler_family', 'deepline_reviewed_action_v1',
                'transport_id', 'deepline',
                'model_binding_requirements_hash', 'sha256:' || repeat('a', 64),
                'output_contract_hash', 'sha256:' || repeat('b', 64),
                'evidence_contract_hash', 'sha256:' || repeat('c', 64),
                'retry_policy_hash', 'sha256:' || repeat('d', 64),
                'credit_ceiling_microunits', 20,
                'timeout_ms', 1000
            )
        )
    )
$envelope$;

DO $identity_and_fencing$
DECLARE
    spec JSONB := pg_temp.routing_test_spec('routing-sql-identity', TRUE);
    v_experiment_hash TEXT;
    envelope JSONB;
    envelope_hash TEXT;
    claim_key TEXT := 'sha256:' || repeat('e', 64);
    claim_capability TEXT := 'sha256:' || repeat('f', 64);
    reserve_doc JSONB;
BEGIN
    v_experiment_hash := public.research_lab_routing_jsonb_hash_v2(spec);
    envelope := pg_temp.routing_test_envelope(spec, v_experiment_hash);
    envelope_hash := public.research_lab_routing_jsonb_hash_v2(envelope);

    BEGIN
        PERFORM public.research_lab_routing_submit_experiment_v2(
            v_experiment_hash, spec->>'experiment_id', spec, 'measured_lab', TRUE,
            'sha256:' || repeat('1', 64), pg_catalog.jsonb_build_object('event', 'bad-envelope'),
            'sha256:' || repeat('0', 64), envelope
        );
        RAISE EXCEPTION 'noncanonical envelope hash unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    PERFORM public.research_lab_routing_submit_experiment_v2(
        v_experiment_hash, spec->>'experiment_id', spec, 'measured_lab', TRUE,
        'sha256:' || repeat('2', 64), pg_catalog.jsonb_build_object('event', 'submit'),
        envelope_hash, envelope
    );
    -- Hash-only request/response metadata is safe to persist.  The scanner
    -- must still reject the corresponding raw material and credential field.
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        pg_catalog.jsonb_build_object(
            'request_body_hash', 'sha256:' || repeat('a', 64),
            'response_body_hash', 'sha256:' || repeat('b', 64),
            'authorization_hash', 'sha256:' || repeat('c', 64),
            'authorization_proof_hash', 'sha256:' || repeat('d', 64),
            'authorization_request_hash', 'sha256:' || repeat('e', 64),
            'authorization_job_id', 'routing-authorization-job',
            'claim_fence_hash', 'sha256:' || repeat('f', 64)
        ),
        'safe-hash-fields'
    );
    FOREACH reserve_doc IN ARRAY ARRAY[
        pg_catalog.jsonb_build_object('request_body', 'raw'),
        pg_catalog.jsonb_build_object('response_body', 'raw'),
        pg_catalog.jsonb_build_object('credentials', 'raw'),
        pg_catalog.jsonb_build_object('authorization', 'raw'),
        pg_catalog.jsonb_build_object('authorization_hash', 'not-a-hash'),
        pg_catalog.jsonb_build_object('claim_capability', claim_capability),
        pg_catalog.jsonb_build_object('claim_nonce', claim_capability),
        pg_catalog.jsonb_build_object('claim_fence_hash', 'not-a-hash')
    ] LOOP
        BEGIN
            PERFORM public.research_lab_routing_reject_secret_doc_v2(
                reserve_doc, 'raw-secret-field'
            );
            RAISE EXCEPTION 'raw secret field unexpectedly succeeded: %', reserve_doc;
        EXCEPTION WHEN invalid_parameter_value THEN NULL;
        END;
    END LOOP;
    -- Exact replay is idempotent; a second envelope cannot be attached to the
    -- same immutable experiment hash.
    PERFORM public.research_lab_routing_submit_experiment_v2(
        v_experiment_hash, spec->>'experiment_id', spec, 'measured_lab', TRUE,
        'sha256:' || repeat('2', 64), pg_catalog.jsonb_build_object('event', 'submit'),
        envelope_hash, envelope
    );
    BEGIN
        PERFORM public.research_lab_routing_submit_experiment_v2(
            'sha256:' || repeat('0', 64), spec->>'experiment_id', spec,
            'measured_lab', TRUE, 'sha256:' || repeat('3', 64),
            pg_catalog.jsonb_build_object('event', 'bad-spec'), envelope_hash, envelope
        );
        RAISE EXCEPTION 'noncanonical spec hash unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;

    PERFORM public.research_lab_routing_claim_experiment_v2(
        v_experiment_hash, claim_key, claim_capability, 'routing-worker', 30,
        pg_catalog.jsonb_build_object('claim', 'one'),
        'sha256:' || repeat('4', 64), pg_catalog.jsonb_build_object('event', 'claim')
    );
    IF public.research_lab_routing_claim_capability_commitment_v2(claim_capability)
       IS DISTINCT FROM 'sha256:' || pg_catalog.encode(
           pg_catalog.sha256(
               pg_catalog.convert_to(
                   'leadpoet.routing.claim-capability-commitment.v2:' || claim_capability,
                   'UTF8'
               )
           ),
           'hex'
       )
    THEN
        RAISE EXCEPTION 'claim capability commitment is not domain-separated';
    END IF;
    BEGIN
        PERFORM public.research_lab_routing_claim_capability_commitment_v2('raw-capability');
        RAISE EXCEPTION 'malformed claim capability unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_claims_v2 claim
        WHERE claim.experiment_hash = v_experiment_hash
          AND claim.claim_capability_commitment = claim_capability
    ) THEN
        RAISE EXCEPTION 'claim capability was persisted instead of its commitment';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_claims_v2 claim
        WHERE claim.experiment_hash = v_experiment_hash
          AND claim.claim_capability_commitment =
              public.research_lab_routing_claim_capability_commitment_v2(claim_capability)
    ) THEN
        RAISE EXCEPTION 'domain-separated claim capability commitment was not persisted';
    END IF;
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:' || repeat('5', 63) || '1', v_experiment_hash, 'run_started',
            claim_key, 1, 'sha256:' || repeat('0', 64),
            pg_catalog.jsonb_build_object('event', 'wrong-capability')
        );
        RAISE EXCEPTION 'wrong claim capability was accepted';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:' || repeat('5', 64), v_experiment_hash, 'run_started',
            claim_key, 1,
            public.research_lab_routing_claim_capability_commitment_v2(claim_capability),
            pg_catalog.jsonb_build_object('event', 'commitment-replay')
        );
        RAISE EXCEPTION 'stored commitment was accepted as a bearer';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;

    reserve_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_budget_event.v2',
        'reservation_id', 'routing-sql-reservation',
        'binding_id', 'binding',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'unit_ref', 'unit-1',
        'variant_id', 'candidate',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'action_id', 'bloomberry_search_job_postings',
        'binding_catalog_manifest_hash', 'sha256:' || repeat('5', 64),
        'call_grant_hash', 'sha256:' || repeat('7', 64),
        'request_body_hash', 'sha256:' || repeat('8', 64)
    );
    BEGIN
        PERFORM public.research_lab_routing_reserve_budget_v2(
            'sha256:' || repeat('9', 64), 'routing-sql-wrong-action',
            v_experiment_hash, 'binding', claim_key, 1, claim_capability, 10, 30,
            pg_catalog.jsonb_set(
                pg_catalog.jsonb_set(
                    reserve_doc,
                    '{reservation_id}',
                    pg_catalog.to_jsonb('routing-sql-wrong-action'::TEXT)
                ),
                '{action_id}', pg_catalog.to_jsonb('podscan_episodes_search'::TEXT)
            )
        );
        RAISE EXCEPTION 'unreviewed action reserved budget';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
    PERFORM public.research_lab_routing_reserve_budget_v2(
        'sha256:' || repeat('a', 64), 'routing-sql-reservation',
        v_experiment_hash, 'binding', claim_key, 1, claim_capability, 10, 30,
        reserve_doc
    );
    PERFORM public.research_lab_routing_append_fenced_event_v2(
        'sha256:' || repeat('b', 64), v_experiment_hash, 'run_started',
        claim_key, 1, claim_capability, pg_catalog.jsonb_build_object('event', 'run-started')
    );
    BEGIN
        PERFORM public.research_lab_routing_append_fenced_event_v2(
            'sha256:' || repeat('b', 64), v_experiment_hash, 'run_started',
            claim_key, 1, claim_capability,
            pg_catalog.jsonb_build_object('event', 'same-hash-different-event')
        );
        RAISE EXCEPTION 'event hash collision unexpectedly succeeded';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
    PERFORM public.research_lab_routing_renew_claim_v2(
        'sha256:' || repeat('6', 64), v_experiment_hash, claim_key, 1,
        claim_capability, 30, pg_catalog.jsonb_build_object('event', 'heartbeat')
    );
    PERFORM public.research_lab_routing_renew_claim_v2(
        'sha256:' || repeat('6', 64), v_experiment_hash, claim_key, 1,
        claim_capability, 30, pg_catalog.jsonb_build_object('event', 'heartbeat')
    );
    PERFORM public.research_lab_routing_close_claim_v2(
        'sha256:' || repeat('c', 64), v_experiment_hash, claim_key, 1,
        claim_capability, 'failed', pg_catalog.jsonb_build_object('event', 'failed')
    );
    IF (public.research_lab_routing_claim_experiment_v2(
        v_experiment_hash, claim_key, claim_capability, 'routing-worker', 30,
        pg_catalog.jsonb_build_object('claim', 'one'),
        'sha256:' || repeat('4', 64), pg_catalog.jsonb_build_object('event', 'claim')
    )->>'idempotent') IS DISTINCT FROM 'true' THEN
        RAISE EXCEPTION 'exact claim replay was not returned idempotently before lifecycle rejection';
    END IF;
    BEGIN
        PERFORM public.research_lab_routing_renew_claim_v2(
            'sha256:' || repeat('7', 64), v_experiment_hash, claim_key, 1,
            claim_capability, 30, pg_catalog.jsonb_build_object('event', 'after-close')
        );
        RAISE EXCEPTION 'closed claim heartbeat unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
    BEGIN
        PERFORM public.research_lab_routing_claim_experiment_v2(
            v_experiment_hash, 'sha256:' || repeat('d', 64),
            'sha256:' || repeat('1', 64), 'routing-worker', 30,
            pg_catalog.jsonb_build_object('claim', 'two'),
            'sha256:' || repeat('e', 64), pg_catalog.jsonb_build_object('event', 'claim-two')
        );
        RAISE EXCEPTION 'terminal experiment was reclaimed';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
END;
$identity_and_fencing$;

DO $terminal_recovery$
DECLARE
    spec JSONB := pg_temp.routing_test_spec('routing-sql-recovery', TRUE);
    v_experiment_hash TEXT;
    envelope JSONB;
    claim_key TEXT := 'sha256:' || repeat('1', 63) || '2';
    claim_capability TEXT := 'sha256:' || repeat('2', 64);
BEGIN
    v_experiment_hash := public.research_lab_routing_jsonb_hash_v2(spec);
    envelope := pg_temp.routing_test_envelope(spec, v_experiment_hash);
    PERFORM public.research_lab_routing_submit_experiment_v2(
        v_experiment_hash, spec->>'experiment_id', spec, 'measured_lab', TRUE,
        'sha256:' || repeat('6', 64), pg_catalog.jsonb_build_object('event', 'submit'),
        public.research_lab_routing_jsonb_hash_v2(envelope), envelope
    );
    PERFORM public.research_lab_routing_claim_experiment_v2(
        v_experiment_hash, claim_key, claim_capability, 'routing-worker', 1,
        pg_catalog.jsonb_build_object('claim', 'expiring'),
        'sha256:' || repeat('7', 64), pg_catalog.jsonb_build_object('event', 'claim')
    );
    -- The claim expires before this reservation lease.  Recovery must still
    -- mark it uncertain because the experiment becomes terminal at claim
    -- recovery, regardless of the reservation's independent lease.
    PERFORM public.research_lab_routing_reserve_budget_v2(
        'sha256:' || repeat('8', 64), 'routing-sql-expired-reservation',
        v_experiment_hash, 'binding', claim_key, 1, claim_capability, 20, 30,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_budget_event.v2',
            'reservation_id', 'routing-sql-expired-reservation',
            'binding_id', 'binding',
            'tool_id', 'intent.source_add.bloomberry_jobs',
            'unit_ref', 'unit-1',
            'variant_id', 'candidate',
            'request_fingerprint', 'sha256:' || repeat('6', 64),
            'action_id', 'bloomberry_search_job_postings',
            'binding_catalog_manifest_hash', 'sha256:' || repeat('5', 64),
            'call_grant_hash', 'sha256:' || repeat('7', 64),
            'request_body_hash', 'sha256:' || repeat('8', 64)
        )
    );
    PERFORM pg_catalog.pg_sleep(1.1);
    PERFORM public.research_lab_routing_recover_claim_v2(
        v_experiment_hash, 'sha256:' || repeat('9', 64), 'recovery-worker',
        pg_catalog.jsonb_build_object('recovery', 'expired'),
        'sha256:' || repeat('9', 64), pg_catalog.jsonb_build_object('event', 'recovered')
    );
    PERFORM public.research_lab_routing_recover_claim_v2(
        v_experiment_hash, 'sha256:' || repeat('9', 64), 'recovery-worker',
        pg_catalog.jsonb_build_object('recovery', 'expired'),
        'sha256:' || repeat('9', 64), pg_catalog.jsonb_build_object('event', 'recovered')
    );
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_routing_budget_events_v2 event
        WHERE event.experiment_hash = v_experiment_hash
          AND event.reservation_id = 'routing-sql-expired-reservation'
          AND event.event_type = 'uncertain'
          AND event.credit_microunits = 20
    ) THEN
        RAISE EXCEPTION 'recovery did not retain the full uncertain ceiling';
    END IF;
    BEGIN
        PERFORM public.research_lab_routing_request_execution_v2(
            'sha256:' || repeat('f', 64), v_experiment_hash,
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
                'experiment_hash', v_experiment_hash
            )
        );
        RAISE EXCEPTION 'recovered experiment accepted a new execution request';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
    BEGIN
        PERFORM public.research_lab_routing_claim_experiment_v2(
            v_experiment_hash, 'sha256:' || repeat('b', 64),
            'sha256:' || repeat('3', 64), 'routing-worker', 30,
            pg_catalog.jsonb_build_object('claim', 'retry'),
            'sha256:' || repeat('d', 64), pg_catalog.jsonb_build_object('event', 'retry')
        );
        RAISE EXCEPTION 'recovered experiment allowed a retry claim';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
END;
$terminal_recovery$;

-- Receipt-chain admission is fail-closed even when the surrounding attempt
-- identity looks plausible.  This negative probe intentionally supplies no
-- authoritative attestation rows; production fixtures add the signed rows
-- before exercising the positive path.
DO $receipt_chain_missing$
DECLARE
    doc JSONB := pg_catalog.jsonb_build_object(
        'call_grant', pg_catalog.jsonb_build_object('job_id', 'routing-job'),
        'call_grant_result', pg_catalog.jsonb_build_object(
            'output_root', 'sha256:' || repeat('1', 64)
        ),
        'call_grant_receipt', pg_catalog.jsonb_build_object(
            'receipt_hash', 'sha256:' || repeat('2', 64),
            'parent_receipt_hashes', pg_catalog.jsonb_build_array()
        ),
        'admission_bundle', pg_catalog.jsonb_build_object(
            'job_id', 'routing-job'
        ),
        'protected_release_receipt', pg_catalog.jsonb_build_object(
            'receipt_hash', 'sha256:' || repeat('3', 64),
            'parent_receipt_hashes', pg_catalog.jsonb_build_array()
        ),
        'terminal_proof', pg_catalog.jsonb_build_object(
            'body', pg_catalog.jsonb_build_object(
                'job_id', 'routing-job',
                'provider_record_hash', 'sha256:' || repeat('8', 64),
                'billing_projection_hash', 'sha256:' || repeat('9', 64),
                'billing_projection', pg_catalog.jsonb_build_object(
                    'outcome', 'verified',
                    'credit_microunits', 1,
                    'latency_ms', 1,
                    'billing_state', 'known'
                )
            ),
            'receipt', pg_catalog.jsonb_build_object(
                'receipt_hash', 'sha256:' || repeat('4', 64),
                'parent_receipt_hashes', pg_catalog.jsonb_build_array(
                    'sha256:' || repeat('3', 64)
                )
            )
        )
    );
BEGIN
    BEGIN
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
            'sha256:' || repeat('5', 64), 'binding',
            'intent.source_add.bloomberry_jobs', 'candidate', 'unit-one',
            'bloomberry_search_job_postings',
            'sha256:' || repeat('6', 64), 'sha256:' || repeat('2', 64),
            'sha256:' || repeat('4', 64), 'sha256:' || repeat('3', 64),
            'sha256:' || repeat('7', 64), 'sha256:' || repeat('8', 64),
            'sha256:' || repeat('9', 64), 'verified', 1, 1, 'known', 1, doc
        );
        RAISE EXCEPTION 'receipt chain without attested rows unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
END;
$receipt_chain_missing$;

-- Build one complete standard ExecutionJobManager receipt graph.  The rows
-- use deterministic redacted documents; signatures are shape-valid only in
-- this SQL probe because cryptographic verification belongs to the protected
-- scorer.  The database still binds every row to its signer, job, roots,
-- parents, and exact receipt document.
DO $receipt_chain_adversarial$
DECLARE
    boot_hash TEXT := 'sha256:' || repeat('a', 64);
    pubkey TEXT := repeat('1', 64);
    auth_hash TEXT := 'sha256:' || repeat('2', 64);
    protected_hash TEXT := 'sha256:' || repeat('3', 64);
    terminal_hash TEXT := 'sha256:' || repeat('4', 64);
    experiment_hash TEXT := 'sha256:' || repeat('5', 64);
    authorization_hash TEXT := 'sha256:' || repeat('6', 64);
    authorization_request_hash TEXT := 'sha256:' || repeat('0', 64);
    authorization_job_id TEXT := 'authorization-job';
    terminal_job_id TEXT := 'terminal-job';
    terminal_request_hash TEXT;
    model_observation_hash TEXT := 'sha256:' || repeat('8', 64);
    admission JSONB;
    admission_hash TEXT;
    protected_input_hash TEXT := 'sha256:' || repeat('7', 64);
    provider_record_hash TEXT := 'sha256:' || repeat('8', 64);
    billing_projection_hash TEXT := 'sha256:' || repeat('9', 64);
    terminal_body JSONB;
    auth_doc JSONB;
    protected_doc JSONB;
    terminal_doc JSONB;
    attempt_doc JSONB;
    base_doc JSONB;
BEGIN
    -- Migration 86 left its original NOT-VALID purpose check alongside the
    -- versioned replacement.  The disposable probe removes only that stale
    -- check inside this transaction so the exact 157/158 contract can be
    -- exercised; ROLLBACK below restores the fixture database unchanged.
    ALTER TABLE public.research_lab_attested_execution_receipts_v2
        DROP CONSTRAINT IF EXISTS research_lab_attested_execution_receipts_v2_check1;
    ALTER TABLE public.research_lab_attested_execution_receipts_v2
        DROP CONSTRAINT IF EXISTS research_lab_attested_execution_receipts_v2_receipt_doc_check;
    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash, experiment_id, spec_doc, receipt_execution_mode,
        allow_live_credit_spend, execution_envelope_hash, execution_envelope_doc
    ) VALUES (
        experiment_hash, 'routing-sql-receipts',
        pg_catalog.jsonb_build_object(
            'contract_version', 'leadpoet.intent_routing_experiment_v2:v2',
            'credit_budget', pg_catalog.jsonb_build_object()
        ),
        'measured_lab', TRUE,
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'model_binding_observation_receipt_hash', model_observation_hash
            )
        ),
        pg_catalog.jsonb_build_object(
            'model_binding_observation_receipt_hash', model_observation_hash
        )
    );
    INSERT INTO public.research_lab_attested_boot_identities_v2 (
        boot_identity_hash, schema_version, role, physical_role, commit_sha,
        pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        signing_pubkey, transport_pubkey, transport_certificate_hash,
        boot_nonce, attestation_user_data_hash, attestation_document_ref,
        attestation_document_hash, identity_doc, issued_at
    ) VALUES (
        boot_hash, 'leadpoet.attested_boot_identity.v2', 'gateway_scoring',
        'gateway_scoring_a', repeat('b', 40), repeat('c', 96),
        'sha256:' || repeat('d', 64), 'sha256:' || repeat('e', 64),
        'sha256:' || repeat('f', 64), pubkey, repeat('2', 64),
        'sha256:' || repeat('1', 64), repeat('3', 32),
        'sha256:' || repeat('4', 64), 'sql-test-attestation',
        'sha256:' || repeat('5', 64), pg_catalog.jsonb_build_object('test', TRUE),
        '2026-08-19T12:00:00Z'
    );
    admission := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_admission.v2',
        'job_id', 'routing-job', 'experiment_id', 'routing-sql-receipts',
        'experiment_hash', experiment_hash,
        'role', 'gateway_scoring',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'envelope_hash', 'sha256:' || repeat('a', 64),
        'artifact_lineage_hash', 'sha256:' || repeat('b', 64),
        'pointer_document_hash', 'sha256:' || repeat('c', 64),
        'immutable_manifest_hash', 'sha256:' || repeat('d', 64),
        'model_artifact_hash', 'sha256:' || repeat('e', 64),
        'gold_label_manifest_hash', 'sha256:' || repeat('f', 64),
        'gold_label_set_hash', 'sha256:' || repeat('1', 64),
        'unit_dataset_manifest_hash', 'sha256:' || repeat('2', 64),
        'unit_set_hash', 'sha256:' || repeat('3', 64),
        'binding_catalog_manifest_hash', 'sha256:' || repeat('4', 64),
        'binding_catalog_version', 'catalog-test-v1',
        'model_binding_observation_hash', 'sha256:' || repeat('7', 64),
        'model_binding_observation_receipt_hash', model_observation_hash,
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(),
        'binding_ids', pg_catalog.jsonb_build_array('binding'),
        'protected_release_hash', 'sha256:' || repeat('9', 64),
        'commit_sha', repeat('b', 40), 'image_digest', 'image@sha256:' || repeat('a', 64),
        'build_id', 'build-test', 'pcr0', repeat('c', 96),
        'build_manifest_hash', 'sha256:' || repeat('d', 64),
        'config_hash', 'sha256:' || repeat('f', 64),
        'boot_identity_hash', boot_hash,
        'protected_receipt_hash', protected_hash
    );
    admission_hash := public.research_lab_routing_jsonb_hash_v2(admission);
    auth_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.attested_execution_receipt.v2',
        'role', 'gateway_scoring', 'purpose', 'research_lab.routing_provider_evidence.v2',
        'job_id', authorization_job_id, 'epoch_id', 1, 'sequence', 1,
        'commit_sha', repeat('b', 40), 'pcr0', repeat('c', 96),
        'build_manifest_hash', 'sha256:' || repeat('d', 64),
        'dependency_lock_hash', 'sha256:' || repeat('e', 64),
        'config_hash', 'sha256:' || repeat('f', 64),
        'boot_identity_hash', boot_hash, 'input_root', authorization_request_hash,
        'output_root', 'sha256:' || repeat('1', 64),
        'transport_root', 'sha256:' || repeat('2', 64),
        'host_operation_root', 'sha256:' || repeat('3', 64),
        'artifact_root', 'sha256:' || repeat('4', 64),
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(protected_hash, model_observation_hash),
        'status', 'succeeded', 'failure_code', NULL,
        'issued_at', '2026-08-19T12:00:00Z',
        'receipt_hash', auth_hash, 'enclave_pubkey', pubkey,
        'enclave_signature', repeat('6', 128)
    );
    protected_doc := jsonb_set(
        auth_doc,
        '{receipt_hash}', pg_catalog.to_jsonb(protected_hash::TEXT)
    ) || pg_catalog.jsonb_build_object(
        'job_id', 'routing-job', 'sequence', 2, 'input_root', protected_input_hash,
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(),
        'output_root', 'sha256:' || repeat('8', 64)
    );
    terminal_body := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.routing_provider_terminal_result.v2',
        'operation', 'routing_provider_terminal_v2',
        'terminal_status', 'authenticated_response',
        'authorization_hash', authorization_hash,
        'authorization_proof_hash', auth_hash,
        'binding', pg_catalog.jsonb_build_object(
            'binding_id', 'binding', 'tool_id', 'intent.source_add.bloomberry_jobs'
        ),
        'unit_ref', 'unit-one',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'provider_record_hash', provider_record_hash,
        'projection', pg_catalog.jsonb_build_object(
            'receipt_ref', 'provider_receipt:' || repeat('1', 16),
            'outcome', 'verified', 'evidence_hash', provider_record_hash,
            'credit_microunits', 1, 'latency_ms', 1, 'billing_state', 'known',
            'binding_id', 'binding', 'provider_id', 'bloomberry_jobs',
            'tool_id', 'intent.source_add.bloomberry_jobs',
            'request_fingerprint', 'sha256:' || repeat('6', 64)
        ),
        'provider_receipt', pg_catalog.jsonb_build_object(
            'receipt_ref', 'provider_receipt:' || repeat('1', 16),
            'outcome', 'verified', 'evidence_hash', provider_record_hash,
            'credit_microunits', 1, 'latency_ms', 1,
            'execution_mode', 'measured_lab', 'binding_id', 'binding',
            'binding_version', 'v1', 'source_lineage_id', 'bloomberry',
            'tool_id', 'intent.source_add.bloomberry_jobs',
            'unit_ref', 'unit-one', 'request_fingerprint', 'sha256:' || repeat('6', 64)
        )
    );
    billing_projection_hash := public.research_lab_routing_jsonb_hash_v2(terminal_body->'projection');
    terminal_request_hash := public.research_lab_routing_jsonb_hash_v2(terminal_body);
    terminal_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.attested_execution_receipt.v2',
        'role', 'gateway_scoring', 'purpose', 'research_lab.routing_provider_evidence.v2',
        'job_id', terminal_job_id, 'epoch_id', 1, 'sequence', 3,
        'commit_sha', repeat('b', 40), 'pcr0', repeat('c', 96),
        'build_manifest_hash', 'sha256:' || repeat('d', 64),
        'dependency_lock_hash', 'sha256:' || repeat('e', 64),
        'config_hash', 'sha256:' || repeat('f', 64),
        'boot_identity_hash', boot_hash,
        'input_root', terminal_request_hash,
        'output_root', public.research_lab_routing_jsonb_hash_v2(terminal_body),
        'transport_root', 'sha256:' || repeat('a', 64),
        'host_operation_root', 'sha256:' || repeat('b', 64),
        'artifact_root', 'sha256:' || repeat('c', 64),
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(auth_hash),
        'status', 'succeeded', 'failure_code', NULL,
        'issued_at', '2026-08-19T12:00:00Z',
        'receipt_hash', terminal_hash, 'enclave_pubkey', pubkey,
        'enclave_signature', repeat('7', 128)
    );
    INSERT INTO public.research_lab_attested_execution_receipts_v2 (
        receipt_hash, schema_version, role, purpose, job_id, epoch_id, sequence,
        commit_sha, pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        boot_identity_hash, input_root, output_root, transport_root,
        host_operation_root, artifact_root, receipt_status, failure_code,
        enclave_pubkey, enclave_signature, receipt_doc, issued_at
    ) VALUES
        (auth_hash, auth_doc->>'schema_version', auth_doc->>'role', auth_doc->>'purpose',
         auth_doc->>'job_id', 1, 1, repeat('b', 40), repeat('c', 96),
         'sha256:' || repeat('d', 64), 'sha256:' || repeat('e', 64),
         'sha256:' || repeat('f', 64), boot_hash, auth_doc->>'input_root',
         auth_doc->>'output_root', auth_doc->>'transport_root',
         auth_doc->>'host_operation_root', auth_doc->>'artifact_root',
         'succeeded', NULL, pubkey, repeat('6', 128), auth_doc,
         '2026-08-19T12:00:00Z'),
        (protected_hash, protected_doc->>'schema_version', protected_doc->>'role',
         protected_doc->>'purpose', protected_doc->>'job_id', 1, 2,
         repeat('b', 40), repeat('c', 96), 'sha256:' || repeat('d', 64),
         'sha256:' || repeat('e', 64), 'sha256:' || repeat('f', 64), boot_hash,
         protected_doc->>'input_root', protected_doc->>'output_root',
         protected_doc->>'transport_root', protected_doc->>'host_operation_root',
         protected_doc->>'artifact_root', 'succeeded', NULL, pubkey, repeat('6', 128),
         protected_doc, '2026-08-19T12:00:00Z'),
        (terminal_hash, terminal_doc->>'schema_version', terminal_doc->>'role',
         terminal_doc->>'purpose', terminal_doc->>'job_id', 1, 3,
         repeat('b', 40), repeat('c', 96), 'sha256:' || repeat('d', 64),
         'sha256:' || repeat('e', 64), 'sha256:' || repeat('f', 64), boot_hash,
         terminal_doc->>'input_root', terminal_doc->>'output_root',
         terminal_doc->>'transport_root', terminal_doc->>'host_operation_root',
         terminal_doc->>'artifact_root',
         'succeeded', NULL, pubkey, repeat('7', 128), terminal_doc,
         '2026-08-19T12:00:00Z');
    base_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_provider_attempt.v3',
        'authorization_request_hash', authorization_request_hash,
        'call_grant', pg_catalog.jsonb_build_object(
            'admission_job_id', 'routing-job',
            'experiment_hash', experiment_hash,
            'admission_bundle_hash', admission_hash
        ),
        'call_grant_result', pg_catalog.jsonb_build_object(
            'authorization_job_id', authorization_job_id,
            'authorization_hash', authorization_hash,
            'experiment_hash', experiment_hash,
            'output_root', 'sha256:' || repeat('1', 64)
        ),
        'call_grant_receipt', auth_doc,
        'admission_bundle', admission,
        'protected_release_receipt', protected_doc,
        'terminal_result', terminal_body,
        'terminal_request_hash', terminal_request_hash,
        'terminal_execution_receipt', terminal_doc,
        'binding_id', 'binding',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'action_id', 'bloomberry_search_job_postings',
        'binding_catalog_manifest_hash', 'sha256:' || repeat('5', 64),
        'request_body_hash', 'sha256:' || repeat('8', 64),
        'variant_id', 'candidate', 'unit_ref', 'unit-one',
        'reservation_id', 'routing-sql-reservation',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'execution_mode', 'measured_lab',
        'provider_receipt', pg_catalog.jsonb_build_object(
            'outcome', 'verified', 'credit_microunits', 1, 'latency_ms', 1
        )
    );
    PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
        experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
        'candidate', 'unit-one', 'bloomberry_search_job_postings', authorization_hash,
        auth_hash, terminal_hash, protected_hash, admission_hash,
        provider_record_hash, billing_projection_hash, 'verified', 1, 1, 'known', 1,
        base_doc
    );
    -- Wrong parent, job, input/output, missing/extra receipt, wrong signer,
    -- and billing mismatch all fail before any durable attempt can pass.
    FOREACH base_doc IN ARRAY ARRAY[
        jsonb_set(base_doc, '{terminal_execution_receipt,parent_receipt_hashes}',
            pg_catalog.jsonb_build_array('sha256:' || repeat('f', 64))),
        jsonb_set(base_doc, '{terminal_execution_receipt,job_id}', pg_catalog.to_jsonb('wrong-job'::TEXT)),
        jsonb_set(base_doc, '{call_grant_receipt,input_root}', pg_catalog.to_jsonb('sha256:' || repeat('f', 64))),
        jsonb_set(base_doc, '{terminal_execution_receipt,output_root}', pg_catalog.to_jsonb('sha256:' || repeat('f', 64))),
        jsonb_set(base_doc, '{call_grant_receipt,parent_receipt_hashes}',
            pg_catalog.jsonb_build_array(protected_hash, model_observation_hash, 'sha256:' || repeat('f', 64)))
    ] LOOP
        BEGIN
            PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
                experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
                'candidate', 'unit-one', 'bloomberry_search_job_postings', authorization_hash,
                auth_hash, terminal_hash, protected_hash, admission_hash,
                provider_record_hash, billing_projection_hash, 'verified', 1, 1, 'known', 1,
                base_doc
            );
            RAISE EXCEPTION 'adversarial receipt graph unexpectedly succeeded';
        EXCEPTION WHEN foreign_key_violation OR invalid_parameter_value THEN NULL;
        END;
    END LOOP;
    BEGIN
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
            experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
            'candidate', 'unit-one', 'bloomberry_search_job_postings', authorization_hash,
            auth_hash, 'sha256:' || repeat('0', 64), protected_hash,
            admission_hash, provider_record_hash, billing_projection_hash,
            'verified', 1, 1, 'known', 1, base_doc
        );
        RAISE EXCEPTION 'missing terminal receipt unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation OR invalid_parameter_value THEN NULL;
    END;
    BEGIN
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
            experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
            'candidate', 'unit-one', 'bloomberry_search_job_postings', authorization_hash,
            auth_hash, terminal_hash, protected_hash, admission_hash,
            provider_record_hash, billing_projection_hash, 'verified', 2, 1, 'known', 2,
            base_doc
        );
        RAISE EXCEPTION 'billing mismatch unexpectedly succeeded';
    EXCEPTION WHEN invalid_parameter_value THEN NULL;
    END;
END;
$receipt_chain_adversarial$;

-- Exercise the complete bearer-free provider-attempt path.  The protected
-- terminal receipt has its own deterministic dispatch job, and it commits the
-- exact durable budget-reservation proof that preceded provider execution.
DO $receipt_chain_v3_full_path$
DECLARE
    spec JSONB := pg_temp.routing_test_spec('routing-sql-receipts-v3', TRUE);
    experiment_hash TEXT;
    envelope JSONB;
    envelope_hash TEXT;
    model_observation_hash TEXT;
    request_doc JSONB;
    request_hash TEXT;
    queue_result JSONB;
    lease_hash TEXT;
    claim_key TEXT := 'sha256:' || repeat('a', 63) || '1';
    claim_fence_hash TEXT;
    boot_hash TEXT := 'sha256:' || repeat('b', 64);
    pubkey TEXT := repeat('2', 64);
    protected_hash TEXT := 'sha256:' || repeat('d', 64);
    auth_hash TEXT := 'sha256:' || repeat('c', 64);
    terminal_hash TEXT := 'sha256:' || repeat('e', 64);
    authorization_request_hash TEXT := 'sha256:' || repeat('1', 64);
    terminal_request_hash TEXT;
    protected_doc JSONB;
    protected_release_hash TEXT;
    admission JSONB;
    admission_hash TEXT;
    call_grant JSONB;
    authorization_hash TEXT;
    grant_result JSONB;
    grant_output_root TEXT;
    auth_doc JSONB;
    reserve_doc JSONB;
    reserve_result JSONB;
    dispatch_result JSONB;
    budget_proof JSONB;
    provider_receipt JSONB;
    projection JSONB;
    provider_record_hash TEXT := 'sha256:' || repeat('f', 64);
    billing_projection_hash TEXT;
    terminal_body JSONB;
    terminal_result_hash TEXT;
    terminal_job_id TEXT;
    terminal_doc JSONB;
    attempt_doc JSONB;
    attempt_key TEXT := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object('fixture', 'provider-attempt-v3')
    );
    append_result JSONB;
    bad_doc JSONB;
    bad_result_hash TEXT;
    decision_doc JSONB;
    decision_receipt_id TEXT := 'routing_decision:' || repeat('1', 16);
    decision_hash TEXT := 'sha256:' || repeat('2', 64);
    evaluation_doc JSONB;
    evaluation_receipt_id TEXT := 'routing_evaluation_v2:' || repeat('3', 16);
    evaluation_hash TEXT;
    decision_root TEXT;
    provider_root TEXT;
    budget_root TEXT;
    authority_input_root TEXT := 'sha256:' || repeat('4', 64);
    authority_output_root TEXT;
    authority_receipt_hash TEXT := 'sha256:' || repeat('5', 63) || 'a';
    reconciliation_doc JSONB;
    reference_hash TEXT;
    promotion_event_hash TEXT := 'sha256:' || repeat('6', 64);
    promotion_event_doc JSONB;
    promotion_result JSONB;
    observation_doc JSONB;
    promotion_experiment_hash TEXT;
BEGIN
    experiment_hash := public.research_lab_routing_jsonb_hash_v2(spec);
    envelope := pg_temp.routing_test_envelope(spec, experiment_hash);
    envelope_hash := public.research_lab_routing_jsonb_hash_v2(envelope);
    model_observation_hash := envelope->>'model_binding_observation_receipt_hash';
    PERFORM public.research_lab_routing_submit_experiment_v2(
        experiment_hash, spec->>'experiment_id', spec, 'measured_lab', TRUE,
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'submit-v3-receipts')
        ),
        pg_catalog.jsonb_build_object('event', 'submit-v3-receipts'),
        envelope_hash, envelope
    );
    request_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
        'experiment_hash', experiment_hash,
        'fixture', 'receipt-chain-v3'
    );
    request_hash := public.research_lab_routing_jsonb_hash_v2(request_doc);
    PERFORM public.research_lab_routing_request_execution_v2(
        request_hash, experiment_hash, request_doc
    );
    queue_result := public.research_lab_routing_claim_execution_requests_v2(
        'routing-receipt-v3-worker', 100, 30
    );
    SELECT request->>'lease_hash' INTO lease_hash
      FROM pg_catalog.jsonb_array_elements(queue_result->'requests') request
     WHERE request->>'request_hash' = request_hash;
    IF lease_hash IS NULL THEN
        RAISE EXCEPTION 'v3 receipt fixture did not receive its queue lease: %', queue_result;
    END IF;
    PERFORM public.research_lab_routing_claim_execution_v3(
        request_hash, lease_hash, 1, 'routing-receipt-v3-worker', claim_key, 30,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim.v3',
            'request_hash', request_hash,
            'lease_hash', lease_hash,
            'lease_generation', 1,
            'worker_ref', 'routing-receipt-v3-worker'
        ),
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'claim-v3-receipts')
        ),
        pg_catalog.jsonb_build_object('event', 'claim-v3-receipts')
    );
    claim_fence_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim_fence.v3',
            'experiment_hash', experiment_hash,
            'claim_key', claim_key,
            'claim_generation', 1
        )
    );

    INSERT INTO public.research_lab_attested_boot_identities_v2 (
        boot_identity_hash, schema_version, role, physical_role, commit_sha,
        pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        signing_pubkey, transport_pubkey, transport_certificate_hash,
        boot_nonce, attestation_user_data_hash, attestation_document_ref,
        attestation_document_hash, identity_doc, issued_at
    ) VALUES (
        boot_hash, 'leadpoet.attested_boot_identity.v2', 'gateway_scoring',
        'gateway_scoring_b', repeat('b', 40), repeat('c', 96),
        'sha256:' || repeat('d', 64), 'sha256:' || repeat('e', 64),
        'sha256:' || repeat('f', 64), pubkey, repeat('3', 64),
        'sha256:' || repeat('2', 64), repeat('4', 32),
        'sha256:' || repeat('5', 64), 'sql-test-attestation-v3',
        'sha256:' || repeat('6', 64), pg_catalog.jsonb_build_object('test', TRUE),
        '2026-08-19T12:01:00Z'
    );
    protected_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.attested_execution_receipt.v2',
        'role', 'gateway_scoring',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'job_id', 'routing-admission-v3',
        'epoch_id', 1,
        'sequence', 1,
        'commit_sha', repeat('b', 40),
        'pcr0', repeat('c', 96),
        'build_manifest_hash', 'sha256:' || repeat('d', 64),
        'dependency_lock_hash', 'sha256:' || repeat('e', 64),
        'config_hash', 'sha256:' || repeat('f', 64),
        'boot_identity_hash', boot_hash,
        'input_root', 'sha256:' || repeat('7', 64),
        'output_root', 'sha256:' || repeat('8', 64),
        'transport_root', 'sha256:' || repeat('9', 64),
        'host_operation_root', 'sha256:' || repeat('a', 64),
        'artifact_root', 'sha256:' || repeat('b', 64),
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(),
        'status', 'succeeded',
        'failure_code', NULL,
        'issued_at', '2026-08-19T12:01:00Z',
        'receipt_hash', protected_hash,
        'enclave_pubkey', pubkey,
        'enclave_signature', repeat('8', 128)
    );
    protected_release_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.routing_protected_release.v2',
            'protected_receipt_hash', protected_hash,
            'protected_commit_sha', protected_doc->>'commit_sha',
            'protected_pcr0', protected_doc->>'pcr0',
            'protected_build_manifest_hash', protected_doc->>'build_manifest_hash',
            'protected_dependency_lock_hash', protected_doc->>'dependency_lock_hash',
            'protected_config_hash', protected_doc->>'config_hash',
            'protected_boot_identity_hash', protected_doc->>'boot_identity_hash',
            'protected_enclave_pubkey', protected_doc->>'enclave_pubkey'
        )
    );
    admission := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_admission.v2',
        'job_id', protected_doc->>'job_id',
        'experiment_id', spec->>'experiment_id',
        'experiment_hash', experiment_hash,
        'role', 'gateway_scoring',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'envelope_hash', envelope_hash,
        'artifact_lineage_hash', envelope->>'artifact_lineage_hash',
        'pointer_document_hash', envelope->>'pointer_document_hash',
        'gold_label_manifest_hash', envelope->>'gold_label_manifest_hash',
        'unit_dataset_manifest_hash', envelope->>'unit_dataset_manifest_hash',
        'unit_set_hash', envelope->>'unit_set_hash',
        'binding_catalog_manifest_hash', envelope->>'binding_catalog_manifest_hash',
        'binding_catalog_version', envelope->>'binding_catalog_version',
        'model_binding_observation_receipt_hash', model_observation_hash,
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(),
        'binding_ids', pg_catalog.jsonb_build_array('binding'),
        'protected_receipt_hash', protected_hash,
        'protected_release_hash', protected_release_hash,
        'protected_commit_sha', protected_doc->>'commit_sha',
        'protected_pcr0', protected_doc->>'pcr0',
        'protected_build_manifest_hash', protected_doc->>'build_manifest_hash',
        'protected_dependency_lock_hash', protected_doc->>'dependency_lock_hash',
        'protected_config_hash', protected_doc->>'config_hash',
        'protected_boot_identity_hash', protected_doc->>'boot_identity_hash',
        'protected_enclave_pubkey', protected_doc->>'enclave_pubkey'
    );
    admission_hash := public.research_lab_routing_jsonb_hash_v2(admission);
    call_grant := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.routing_provider_call_grant.v2',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'experiment_hash', experiment_hash,
        'envelope_hash', envelope_hash,
        'admission_job_id', admission->>'job_id',
        'admission_bundle_hash', admission_hash,
        'protected_release_hash', protected_release_hash,
        'model_binding_observation_receipt_hash', model_observation_hash,
        'binding_catalog_manifest_hash', envelope->>'binding_catalog_manifest_hash',
        'unit_dataset_manifest_hash', envelope->>'unit_dataset_manifest_hash',
        'unit_set_hash', envelope->>'unit_set_hash',
        'binding', pg_catalog.jsonb_build_object(
            'binding_id', 'binding',
            'tool_id', 'intent.source_add.bloomberry_jobs'
        ),
        'variant_id', 'candidate',
        'unit_ref', 'unit-one',
        'action_id', 'bloomberry_search_job_postings',
        'request_body_hash', 'sha256:' || repeat('4', 64),
        'claim_key', claim_key,
        'claim_generation', 1,
        'claim_fence_hash', claim_fence_hash
    );
    authorization_hash := public.research_lab_routing_jsonb_hash_v2(call_grant);
    grant_result := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.routing_provider_call_grant_result.v2',
        'operation', 'attest_routing_provider_call_v2',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'attested', TRUE,
        'authorization_job_id', 'routing-authorization-v3',
        'authorization_hash', authorization_hash,
        'experiment_hash', experiment_hash,
        'admission_job_id', admission->>'job_id',
        'admission_bundle_hash', admission_hash,
        'protected_release_hash', protected_release_hash,
        'model_binding_observation_receipt_hash', model_observation_hash,
        'binding_id', 'binding',
        'variant_id', 'candidate',
        'action_id', 'bloomberry_search_job_postings',
        'request_body_hash', 'sha256:' || repeat('4', 64),
        'claim_generation', 1,
        'claim_fence_hash', claim_fence_hash
    );
    grant_output_root := public.research_lab_routing_jsonb_hash_v2(grant_result);
    grant_result := grant_result || pg_catalog.jsonb_build_object(
        'output_root', grant_output_root
    );
    auth_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.attested_execution_receipt.v2',
        'role', 'gateway_scoring',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'job_id', grant_result->>'authorization_job_id',
        'epoch_id', 1,
        'sequence', 2,
        'commit_sha', protected_doc->>'commit_sha',
        'pcr0', protected_doc->>'pcr0',
        'build_manifest_hash', protected_doc->>'build_manifest_hash',
        'dependency_lock_hash', protected_doc->>'dependency_lock_hash',
        'config_hash', protected_doc->>'config_hash',
        'boot_identity_hash', boot_hash,
        'input_root', authorization_request_hash,
        'output_root', grant_output_root,
        'transport_root', 'sha256:' || repeat('2', 64),
        'host_operation_root', 'sha256:' || repeat('3', 64),
        'artifact_root', 'sha256:' || repeat('4', 64),
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(
            protected_hash, model_observation_hash
        ),
        'status', 'succeeded',
        'failure_code', NULL,
        'issued_at', '2026-08-19T12:01:01Z',
        'receipt_hash', auth_hash,
        'enclave_pubkey', pubkey,
        'enclave_signature', repeat('9', 128)
    );

    reserve_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_budget_event.v3',
        'reservation_id', 'routing-sql-reservation-v3',
        'binding_id', 'binding',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'unit_ref', 'unit-one',
        'variant_id', 'candidate',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'action_id', 'bloomberry_search_job_postings',
        'binding_catalog_manifest_hash', envelope->>'binding_catalog_manifest_hash',
        'call_grant_hash', authorization_hash,
        'request_body_hash', 'sha256:' || repeat('4', 64)
    );
    reserve_result := public.research_lab_routing_reserve_budget_v3(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'reserve-v3-receipts')
        ),
        'routing-sql-reservation-v3', experiment_hash, 'binding', claim_key, 1,
        2, 30, reserve_doc
    );
    dispatch_result := public.research_lab_routing_append_fenced_event_v3(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'dispatch-v3-receipts')
        ),
        experiment_hash, 'provider_dispatch_started', claim_key, 1, reserve_doc
    );
    IF (dispatch_result->>'idempotent')::BOOLEAN IS NOT FALSE THEN
        RAISE EXCEPTION 'v3 provider dispatch marker did not append: %', dispatch_result;
    END IF;
    FOREACH bad_doc IN ARRAY ARRAY[
        pg_catalog.jsonb_set(
            reserve_doc, '{reservation_id}',
            pg_catalog.to_jsonb('routing-sql-wrong-reservation-v3'::TEXT)
        ),
        pg_catalog.jsonb_set(
            reserve_doc, '{binding_id}',
            pg_catalog.to_jsonb('wrong-binding'::TEXT)
        ),
        pg_catalog.jsonb_set(
            reserve_doc, '{request_fingerprint}',
            pg_catalog.to_jsonb(('sha256:' || repeat('0', 64))::TEXT)
        ),
        pg_catalog.jsonb_set(
            reserve_doc, '{unit_ref}',
            pg_catalog.to_jsonb('wrong-unit'::TEXT)
        )
    ] LOOP
        BEGIN
            PERFORM public.research_lab_routing_append_fenced_event_v3(
                public.research_lab_routing_jsonb_hash_v2(
                    pg_catalog.jsonb_build_object(
                        'fixture', 'bad-dispatch-v3-receipts',
                        'event_doc', bad_doc
                    )
                ),
                experiment_hash, 'provider_dispatch_started', claim_key, 1, bad_doc
            );
            RAISE EXCEPTION 'forged v3 provider dispatch marker unexpectedly succeeded';
        EXCEPTION WHEN foreign_key_violation THEN NULL;
        END;
    END LOOP;
    budget_proof := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_budget_reservation_proof.v3',
        'reservation_id', reserve_result->>'reservation_id',
        'event_key', reserve_result->>'event_key',
        'experiment_hash', reserve_result->>'experiment_hash',
        'binding_id', reserve_result->>'binding_id',
        'claim_key', reserve_result->>'claim_key',
        'claim_generation', (reserve_result->>'claim_generation')::BIGINT,
        'credit_microunits', (reserve_result->>'credit_microunits')::BIGINT,
        'lease_expires_at', reserve_result->'lease_expires_at',
        'response_hash', public.research_lab_routing_jsonb_hash_v2(reserve_result),
        'transport_attempt_hash', 'sha256:' || repeat('5', 64)
    );
    provider_receipt := pg_catalog.jsonb_build_object(
        'receipt_ref', 'provider_receipt:' || repeat('1', 16),
        'outcome', 'verified',
        'evidence_hash', provider_record_hash,
        'credit_microunits', 1,
        'latency_ms', 1,
        'execution_mode', 'measured_lab',
        'binding_id', 'binding',
        'binding_version', 'v1',
        'source_lineage_id', 'bloomberry',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'unit_ref', 'unit-one',
        'request_fingerprint', 'sha256:' || repeat('6', 64)
    );
    projection := pg_catalog.jsonb_build_object(
        'receipt_ref', provider_receipt->>'receipt_ref',
        'outcome', 'verified',
        'evidence_hash', provider_record_hash,
        'credit_microunits', 1,
        'latency_ms', 1,
        'billing_state', 'known',
        'binding_id', 'binding',
        'provider_id', 'bloomberry_jobs',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'request_fingerprint', 'sha256:' || repeat('6', 64)
    );
    billing_projection_hash := public.research_lab_routing_jsonb_hash_v2(projection);
    terminal_body := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.routing_provider_terminal_result.v2',
        'operation', 'routing_provider_terminal_v2',
        'terminal_status', 'authenticated_response',
        'authorization_hash', authorization_hash,
        'authorization_proof_hash', auth_hash,
        'binding', pg_catalog.jsonb_build_object(
            'binding_id', 'binding',
            'tool_id', 'intent.source_add.bloomberry_jobs'
        ),
        'unit_ref', 'unit-one',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'provider_record_hash', provider_record_hash,
        'transport_attempt_hash', 'sha256:' || repeat('6', 64),
        'budget_reservation', budget_proof,
        'projection', projection,
        'provider_receipt', provider_receipt
    );
    terminal_result_hash := public.research_lab_routing_jsonb_hash_v2(terminal_body);
    terminal_request_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.routing_provider_dispatch_request.v3',
            'authorization_hash', authorization_hash,
            'authorization_proof_hash', auth_hash,
            'budget_reservation_response_hash', budget_proof->>'response_hash'
        )
    );
    terminal_job_id := 'routing-dispatch:' || pg_catalog.substr(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.routing_provider_dispatch_job.v3',
                'authorization_hash', authorization_hash,
                'authorization_proof_hash', auth_hash,
                'authorization_receipt_hash', auth_hash
            )
        ), 8, 32
    );
    terminal_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.attested_execution_receipt.v2',
        'role', 'gateway_scoring',
        'purpose', 'research_lab.routing_provider_evidence.v2',
        'job_id', terminal_job_id,
        'epoch_id', 1,
        'sequence', 3,
        'commit_sha', protected_doc->>'commit_sha',
        'pcr0', protected_doc->>'pcr0',
        'build_manifest_hash', protected_doc->>'build_manifest_hash',
        'dependency_lock_hash', protected_doc->>'dependency_lock_hash',
        'config_hash', protected_doc->>'config_hash',
        'boot_identity_hash', boot_hash,
        'input_root', terminal_request_hash,
        'output_root', terminal_result_hash,
        'transport_root', 'sha256:' || repeat('a', 64),
        'host_operation_root', 'sha256:' || repeat('b', 64),
        'artifact_root', 'sha256:' || repeat('c', 64),
        'parent_receipt_hashes', pg_catalog.jsonb_build_array(auth_hash),
        'status', 'succeeded',
        'failure_code', NULL,
        'issued_at', '2026-08-19T12:01:02Z',
        'receipt_hash', terminal_hash,
        'enclave_pubkey', pubkey,
        'enclave_signature', repeat('a', 128)
    );
    INSERT INTO public.research_lab_attested_execution_receipts_v2 (
        receipt_hash, schema_version, role, purpose, job_id, epoch_id, sequence,
        commit_sha, pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        boot_identity_hash, input_root, output_root, transport_root,
        host_operation_root, artifact_root, receipt_status, failure_code,
        enclave_pubkey, enclave_signature, receipt_doc, issued_at
    ) VALUES
        (protected_hash, protected_doc->>'schema_version', protected_doc->>'role',
         protected_doc->>'purpose', protected_doc->>'job_id', 1, 1,
         protected_doc->>'commit_sha', protected_doc->>'pcr0',
         protected_doc->>'build_manifest_hash', protected_doc->>'dependency_lock_hash',
         protected_doc->>'config_hash', boot_hash, protected_doc->>'input_root',
         protected_doc->>'output_root', protected_doc->>'transport_root',
         protected_doc->>'host_operation_root', protected_doc->>'artifact_root',
         'succeeded', NULL, pubkey, protected_doc->>'enclave_signature', protected_doc,
         '2026-08-19T12:01:00Z'),
        (auth_hash, auth_doc->>'schema_version', auth_doc->>'role', auth_doc->>'purpose',
         auth_doc->>'job_id', 1, 2, auth_doc->>'commit_sha', auth_doc->>'pcr0',
         auth_doc->>'build_manifest_hash', auth_doc->>'dependency_lock_hash',
         auth_doc->>'config_hash', boot_hash, auth_doc->>'input_root',
         auth_doc->>'output_root', auth_doc->>'transport_root',
         auth_doc->>'host_operation_root', auth_doc->>'artifact_root',
         'succeeded', NULL, pubkey, auth_doc->>'enclave_signature', auth_doc,
         '2026-08-19T12:01:01Z'),
        (terminal_hash, terminal_doc->>'schema_version', terminal_doc->>'role',
         terminal_doc->>'purpose', terminal_doc->>'job_id', 1, 3,
         terminal_doc->>'commit_sha', terminal_doc->>'pcr0',
         terminal_doc->>'build_manifest_hash', terminal_doc->>'dependency_lock_hash',
         terminal_doc->>'config_hash', boot_hash, terminal_doc->>'input_root',
         terminal_doc->>'output_root', terminal_doc->>'transport_root',
         terminal_doc->>'host_operation_root', terminal_doc->>'artifact_root',
         'succeeded', NULL, pubkey, terminal_doc->>'enclave_signature', terminal_doc,
         '2026-08-19T12:01:02Z');

    attempt_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_provider_attempt.v3',
        'authorization_request_hash', authorization_request_hash,
        'call_grant_hash', authorization_hash,
        'call_grant_proof_hash', auth_hash,
        'call_grant', call_grant,
        'call_grant_result', grant_result,
        'call_grant_receipt', auth_doc,
        'admission_bundle', admission,
        'protected_release_receipt', protected_doc,
        'terminal_result', terminal_body,
        'terminal_request_hash', terminal_request_hash,
        'terminal_execution_receipt', terminal_doc,
        'binding_id', 'binding',
        'tool_id', 'intent.source_add.bloomberry_jobs',
        'action_id', 'bloomberry_search_job_postings',
        'binding_catalog_manifest_hash', envelope->>'binding_catalog_manifest_hash',
        'request_body_hash', 'sha256:' || repeat('4', 64),
        'variant_id', 'candidate',
        'unit_ref', 'unit-one',
        'reservation_id', 'routing-sql-reservation-v3',
        'request_fingerprint', 'sha256:' || repeat('6', 64),
        'execution_mode', 'measured_lab',
        'provider_receipt', provider_receipt
    );
    PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
        experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
        'candidate', 'unit-one', 'bloomberry_search_job_postings',
        authorization_hash, authorization_request_hash, auth_hash, terminal_hash,
        protected_hash, admission_hash, terminal_request_hash, terminal_result_hash,
        provider_record_hash, billing_projection_hash, 'verified', 1, 1, 'known', 1,
        attempt_doc
    );
    append_result := public.research_lab_routing_append_provider_attempt_v3(
        attempt_key, experiment_hash, provider_receipt->>'receipt_ref', 'binding',
        'intent.source_add.bloomberry_jobs', 'candidate', 'unit-one',
        'routing-sql-reservation-v3', 'bloomberry_search_job_postings',
        envelope->>'binding_catalog_manifest_hash', authorization_hash,
        authorization_request_hash, auth_hash, 'sha256:' || repeat('4', 64),
        claim_key, 1, 'sha256:' || repeat('6', 64), 'verified', 1, 1,
        'measured_lab', 'known', 1, terminal_hash, protected_hash, admission_hash,
        terminal_request_hash, terminal_result_hash, provider_record_hash,
        billing_projection_hash, attempt_doc
    );
    IF (append_result->>'idempotent')::BOOLEAN IS NOT FALSE THEN
        RAISE EXCEPTION 'v3 provider attempt did not append: %', append_result;
    END IF;
    append_result := public.research_lab_routing_append_provider_attempt_v3(
        attempt_key, experiment_hash, provider_receipt->>'receipt_ref', 'binding',
        'intent.source_add.bloomberry_jobs', 'candidate', 'unit-one',
        'routing-sql-reservation-v3', 'bloomberry_search_job_postings',
        envelope->>'binding_catalog_manifest_hash', authorization_hash,
        authorization_request_hash, auth_hash, 'sha256:' || repeat('4', 64),
        claim_key, 1, 'sha256:' || repeat('6', 64), 'verified', 1, 1,
        'measured_lab', 'known', 1, terminal_hash, protected_hash, admission_hash,
        terminal_request_hash, terminal_result_hash, provider_record_hash,
        billing_projection_hash, attempt_doc
    );
    IF (append_result->>'idempotent')::BOOLEAN IS NOT TRUE THEN
        RAISE EXCEPTION 'exact v3 terminal attempt replay was not idempotent: %', append_result;
    END IF;

    FOREACH bad_doc IN ARRAY ARRAY[
        pg_catalog.jsonb_set(
            attempt_doc, '{terminal_execution_receipt,job_id}',
            pg_catalog.to_jsonb('wrong-dispatch-job'::TEXT)
        ),
        pg_catalog.jsonb_set(
            attempt_doc, '{terminal_execution_receipt,parent_receipt_hashes}',
            pg_catalog.jsonb_build_array(protected_hash)
        ),
        pg_catalog.jsonb_set(
            attempt_doc, '{call_grant_result,authorization_job_id}',
            pg_catalog.to_jsonb((admission->>'job_id')::TEXT)
        ),
        pg_catalog.jsonb_set(
            attempt_doc, '{call_grant,claim_fence_hash}',
            pg_catalog.to_jsonb(('sha256:' || repeat('0', 64))::TEXT)
        )
    ] LOOP
        BEGIN
            PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
                experiment_hash, 'binding', 'intent.source_add.bloomberry_jobs',
                'candidate', 'unit-one', 'bloomberry_search_job_postings',
                authorization_hash, authorization_request_hash, auth_hash, terminal_hash,
                protected_hash, admission_hash, terminal_request_hash, terminal_result_hash,
                provider_record_hash, billing_projection_hash, 'verified', 1, 1,
                'known', 1, bad_doc
            );
            RAISE EXCEPTION 'adversarial v3 receipt graph unexpectedly succeeded';
        EXCEPTION WHEN invalid_parameter_value OR foreign_key_violation THEN NULL;
        END;
    END LOOP;

    bad_doc := pg_catalog.jsonb_set(
        attempt_doc,
        '{terminal_result,budget_reservation,response_hash}',
        pg_catalog.to_jsonb(('sha256:' || repeat('0', 64))::TEXT)
    );
    bad_result_hash := public.research_lab_routing_jsonb_hash_v2(
        bad_doc->'terminal_result'
    );
    BEGIN
        PERFORM public.research_lab_routing_append_provider_attempt_v3(
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object('fixture', 'bad-budget-proof-v3')
            ),
            experiment_hash, provider_receipt->>'receipt_ref', 'binding',
            'intent.source_add.bloomberry_jobs', 'candidate', 'unit-one',
            'routing-sql-reservation-v3', 'bloomberry_search_job_postings',
            envelope->>'binding_catalog_manifest_hash', authorization_hash,
            authorization_request_hash, auth_hash, 'sha256:' || repeat('4', 64),
            claim_key, 1, 'sha256:' || repeat('6', 64), 'verified', 1, 1,
            'measured_lab', 'known', 1, terminal_hash, protected_hash, admission_hash,
            terminal_request_hash, bad_result_hash, provider_record_hash,
            billing_projection_hash, bad_doc
        );
        RAISE EXCEPTION 'forged budget reservation response hash unexpectedly succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;

    -- Complete the durable graph so promotion exercises the reconciliation
    -- helper through its real caller.  The provider attempt above is not
    -- promotable until its reservation is settled and both decision and
    -- evaluation receipts cover the exact durable receipt sets.
    PERFORM public.research_lab_routing_settle_budget_v3(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'settle-v3-receipts')
        ),
        'routing-sql-reservation-v3', attempt_key, claim_key, 1,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_budget_event.v3',
            'attempt_key', attempt_key,
            'billing_state', 'known'
        )
    );

    decision_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_decision.v3',
        'receipt_id', decision_receipt_id,
        'experiment_hash', experiment_hash,
        'variant_id', 'candidate',
        'unit_ref', 'unit-one',
        'plan_hash', 'sha256:' || repeat('7', 64),
        'route_hash', 'sha256:' || repeat('8', 64),
        'provider_receipt_refs', pg_catalog.jsonb_build_array(
            provider_receipt->>'receipt_ref'
        ),
        'passed', TRUE
    );
    PERFORM public.research_lab_routing_append_decision_receipt_v3(
        decision_receipt_id, experiment_hash, 'candidate', 'unit-one',
        decision_doc->>'plan_hash', decision_doc->>'route_hash', claim_key, 1,
        decision_doc
    );

    evaluation_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_evaluation.v3',
        'receipt_id', evaluation_receipt_id,
        'experiment_hash', experiment_hash,
        'selected_variant_id', 'candidate',
        'decision_receipt_refs', pg_catalog.jsonb_build_array(decision_receipt_id),
        'provider_receipt_refs', pg_catalog.jsonb_build_array(
            provider_receipt->>'receipt_ref'
        ),
        'variants', pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'variant_id', 'candidate',
                'passed', TRUE,
                'calibration', pg_catalog.jsonb_build_object(
                    'adapter_failure_count', 0
                ),
                'holdout', pg_catalog.jsonb_build_object(
                    'adapter_failure_count', 0
                )
            )
        )
    );
    evaluation_hash := public.research_lab_routing_jsonb_hash_v2(evaluation_doc);
    PERFORM public.research_lab_routing_append_evaluation_v3(
        evaluation_receipt_id, experiment_hash, evaluation_hash, 'candidate',
        claim_key, 1, evaluation_doc
    );
    promotion_experiment_hash := experiment_hash;

    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', decision.receipt_id,
                    'row', pg_catalog.jsonb_build_object(
                        'receipt_id', decision.receipt_id,
                        'experiment_hash', decision.experiment_hash,
                        'decision_doc', decision.decision_doc
                    )
                ) ORDER BY decision.receipt_id
            ), '[]'::JSONB
        )
    ) INTO decision_root
      FROM public.research_lab_routing_decision_receipts_v2 decision
     WHERE decision.experiment_hash = promotion_experiment_hash;
    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', attempt.attempt_key,
                    'row', pg_catalog.jsonb_build_object(
                        'attempt_key', attempt.attempt_key,
                        'experiment_hash', attempt.experiment_hash,
                        'provider_receipt_ref', attempt.provider_receipt_ref,
                        'binding_id', attempt.binding_id,
                        'tool_id', attempt.tool_id,
                        'variant_id', attempt.variant_id,
                        'unit_ref', attempt.unit_ref,
                        'reservation_id', attempt.reservation_id,
                        'action_id', attempt.action_id,
                        'binding_catalog_manifest_hash', attempt.binding_catalog_manifest_hash,
                        'authorization_hash', attempt.authorization_hash,
                        'authorization_proof_hash', attempt.authorization_proof_hash,
                        'request_body_hash', attempt.request_body_hash,
                        'request_fingerprint', attempt.request_fingerprint,
                        'outcome', attempt.outcome,
                        'credit_microunits', attempt.credit_microunits,
                        'latency_ms', attempt.latency_ms,
                        'execution_mode', attempt.execution_mode,
                        'billing_state', attempt.billing_state,
                        'authoritative_billed_credit_microunits',
                            attempt.authoritative_billed_credit_microunits,
                        'attempt_doc', attempt.attempt_doc
                    )
                ) ORDER BY attempt.attempt_key
            ), '[]'::JSONB
        )
    ) INTO provider_root
      FROM public.research_lab_routing_provider_attempts_v2 attempt
     WHERE attempt.experiment_hash = promotion_experiment_hash;
    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', budget.event_key,
                    'row', pg_catalog.jsonb_build_object(
                        'event_key', budget.event_key,
                        'experiment_hash', budget.experiment_hash,
                        'reservation_id', budget.reservation_id,
                        'binding_id', budget.binding_id,
                        'attempt_key', budget.attempt_key,
                        'event_type', budget.event_type,
                        'credit_microunits', budget.credit_microunits,
                        'event_doc', budget.event_doc
                    )
                ) ORDER BY budget.event_key
            ), '[]'::JSONB
        )
    ) INTO budget_root
      FROM public.research_lab_routing_budget_events_v2 budget
     WHERE budget.experiment_hash = promotion_experiment_hash;

    -- The observation receipt is deliberately the exact compact receipt doc
    -- embedded in the signed execution envelope.  Its signer is bound to the
    -- same measured boot identity as the provider graph.
    observation_doc := pg_catalog.jsonb_build_object('receipt_hash', model_observation_hash);
    INSERT INTO public.research_lab_attested_execution_receipts_v2 (
        receipt_hash, schema_version, role, purpose, job_id, epoch_id, sequence,
        commit_sha, pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        boot_identity_hash, input_root, output_root, transport_root,
        host_operation_root, artifact_root, receipt_status, failure_code,
        enclave_pubkey, enclave_signature, receipt_doc, issued_at
    ) VALUES (
        model_observation_hash, 'leadpoet.attested_execution_receipt.v2',
        'gateway_scoring', 'research_lab.routing_model_binding_observation.v2',
        'routing-observation-v3', 1, 4, protected_doc->>'commit_sha',
        protected_doc->>'pcr0', protected_doc->>'build_manifest_hash',
        protected_doc->>'dependency_lock_hash', protected_doc->>'config_hash',
        boot_hash, envelope->'model_binding_observation'->'result'->>'request_root',
        public.research_lab_routing_jsonb_hash_v2(
            envelope->'model_binding_observation'->'result'
        ), 'sha256:' || repeat('1', 64), 'sha256:' || repeat('2', 64),
        'sha256:' || repeat('3', 64), 'succeeded', NULL, pubkey,
        repeat('b', 128), observation_doc, '2026-08-19T12:01:03Z'
    );

    reconciliation_doc := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_reconciliation.v3',
        'experiment_hash', experiment_hash,
        'evaluation_receipt_id', evaluation_receipt_id,
        'evaluation_hash', evaluation_hash,
        'selected_variant_id', 'candidate',
        'reconciled', TRUE,
        'decision_receipts_root', decision_root,
        'provider_attempts_root', provider_root,
        'budget_events_root', budget_root,
        'artifact_lineage_hash', envelope->>'artifact_lineage_hash',
        'artifact_pointer_document_hash', envelope->>'pointer_document_hash',
        'gold_label_manifest_hash', envelope->>'gold_label_manifest_hash',
        'execution_envelope_hash', envelope_hash,
        'authoritative_billed_credit_microunits', 1,
        'authority_receipt_hash', authority_receipt_hash,
        'authority_input_root', authority_input_root,
        'authority_commit_sha', protected_doc->>'commit_sha',
        'authority_pcr0', protected_doc->>'pcr0',
        'authority_build_manifest_hash', protected_doc->>'build_manifest_hash',
        'authority_boot_identity_hash', boot_hash
    );
    authority_output_root := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_experiment_attestation_result.v2',
            'reconciled', TRUE,
            'experiment_hash', experiment_hash,
            'evaluation_hash', evaluation_hash,
            'evaluation_receipt_id', evaluation_receipt_id,
            'selected_variant_id', 'candidate',
            'decision_receipts_root', decision_root,
            'provider_attempts_root', provider_root,
            'budget_events_root', budget_root,
            'artifact_lineage_hash', envelope->>'artifact_lineage_hash',
            'gold_label_manifest_hash', envelope->>'gold_label_manifest_hash',
            'execution_envelope_hash', envelope_hash,
            'authoritative_billed_credit_microunits', 1,
            'input_root', authority_input_root
        )
    );
    reconciliation_doc := reconciliation_doc || pg_catalog.jsonb_build_object(
        'authority_output_root', authority_output_root
    );
    INSERT INTO public.research_lab_attested_execution_receipts_v2 (
        receipt_hash, schema_version, role, purpose, job_id, epoch_id, sequence,
        commit_sha, pcr0, build_manifest_hash, dependency_lock_hash, config_hash,
        boot_identity_hash, input_root, output_root, transport_root,
        host_operation_root, artifact_root, receipt_status, failure_code,
        enclave_pubkey, enclave_signature, receipt_doc, issued_at
    ) VALUES (
        authority_receipt_hash, 'leadpoet.attested_execution_receipt.v2',
        'gateway_scoring', 'research_lab.routing_experiment.v2',
        'routing-attestation-v3', 1, 5, protected_doc->>'commit_sha',
        protected_doc->>'pcr0', protected_doc->>'build_manifest_hash',
        protected_doc->>'dependency_lock_hash', protected_doc->>'config_hash',
        boot_hash, authority_input_root, authority_output_root,
        'sha256:' || repeat('7', 64), 'sha256:' || repeat('8', 64),
        'sha256:' || repeat('9', 64), 'succeeded', NULL, pubkey,
        repeat('c', 128), pg_catalog.jsonb_build_object(
            'receipt_hash', authority_receipt_hash,
            'input_root', authority_input_root,
            'output_root', authority_output_root
        ), '2026-08-19T12:01:04Z'
    );

    reference_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'contract_version', 'leadpoet.routing_experiment_v2_lab_reference:v2',
            'experiment_hash', experiment_hash,
            'evaluation_hash', evaluation_hash,
            'evaluation_receipt_id', evaluation_receipt_id,
            'selected_variant_id', 'candidate',
            'reconciliation', reconciliation_doc
        )
    );
    promotion_event_doc := pg_catalog.jsonb_build_object(
        'event_type', 'promoted',
        'experiment_hash', experiment_hash,
        'evaluation_receipt_id', evaluation_receipt_id,
        'reference_hash', reference_hash
    );
    promotion_event_hash := public.research_lab_routing_jsonb_hash_v2(
        promotion_event_doc
    );
    promotion_result := public.research_lab_routing_promote_v3(
        reference_hash, experiment_hash, evaluation_receipt_id, evaluation_hash,
        'candidate', reconciliation_doc, promotion_event_hash, promotion_event_doc
    );
    IF (promotion_result->>'idempotent')::BOOLEAN IS NOT FALSE
       OR promotion_result->>'reference_hash' IS DISTINCT FROM reference_hash
    THEN
        RAISE EXCEPTION 'exact v3 promotion did not append: %', promotion_result;
    END IF;

    -- A durable-root substitution must be rejected even when the signed
    -- attestation and evaluation documents remain unchanged.
    bad_doc := pg_catalog.jsonb_set(
        reconciliation_doc, '{provider_attempts_root}',
        pg_catalog.to_jsonb(('sha256:' || repeat('0', 64))::TEXT)
    );
    reference_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'contract_version', 'leadpoet.routing_experiment_v2_lab_reference:v2',
            'experiment_hash', experiment_hash,
            'evaluation_hash', evaluation_hash,
            'evaluation_receipt_id', evaluation_receipt_id,
            'selected_variant_id', 'candidate',
            'reconciliation', bad_doc
        )
    );
    promotion_event_doc := pg_catalog.jsonb_set(
        promotion_event_doc, '{reference_hash}',
        pg_catalog.to_jsonb(reference_hash)
    );
    BEGIN
        PERFORM public.research_lab_routing_promote_v3(
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'contract_version', 'leadpoet.routing_experiment_v2_lab_reference:v2',
                    'experiment_hash', experiment_hash,
                    'evaluation_hash', evaluation_hash,
                    'evaluation_receipt_id', evaluation_receipt_id,
                    'selected_variant_id', 'candidate',
                    'reconciliation', bad_doc
                )
            ),
            experiment_hash, evaluation_receipt_id, evaluation_hash,
            'candidate', bad_doc,
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object('fixture', 'forged-promotion-event')
            ), promotion_event_doc
        );
        RAISE EXCEPTION 'durable root substitution unexpectedly promoted';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
END;
$receipt_chain_v3_full_path$;

-- The append RPC checks an exact replay before claim/reservation lifecycle
-- rejection, but the replay path still invokes the complete receipt validator.
-- Verify that ordering in the installed SQL, which catches regressions that
-- only appear after a worker restart with an expired claim.
DO $stale_replay_order$
DECLARE
    append_source TEXT;
    existing_pos INTEGER;
    claim_pos INTEGER;
BEGIN
    SELECT pg_catalog.pg_get_functiondef(
        'public.research_lab_routing_append_provider_attempt_v3(text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,bigint,text,text,bigint,bigint,text,text,bigint,text,text,text,text,text,text,text,jsonb)'::REGPROCEDURE
    ) INTO append_source;
    existing_pos := pg_catalog.strpos(append_source, 'SELECT * INTO existing');
    claim_pos := pg_catalog.strpos(append_source, 'research_lab_routing_assert_claim_v3');
    IF existing_pos = 0 OR claim_pos = 0 OR existing_pos > claim_pos THEN
        RAISE EXCEPTION 'stale replay is not checked before claim lifecycle';
    END IF;
    IF pg_catalog.strpos(append_source, 'research_lab_routing_assert_provider_receipt_chain_v3') = 0
       OR pg_catalog.strpos(append_source, 'p_authoritative_billed_credit_microunits') = 0
       OR pg_catalog.strpos(append_source, 'p_terminal_result_hash') = 0
       OR pg_catalog.strpos(append_source, 'p_authorization_request_hash') = 0
       OR pg_catalog.strpos(append_source, 'budget_reservation') = 0
    THEN
        RAISE EXCEPTION 'append RPC does not retain strict receipt/billing checks';
    END IF;
END;
$stale_replay_order$;

-- V3 claim fencing has no bearer argument or persisted capability
-- commitment.  The same claim key/generation is the only durable identity;
-- a transaction lock serializes every mutation for the experiment.
DO $claim_v3_contract$
DECLARE
    experiment_hash TEXT := 'sha256:' || repeat('a', 64);
    request_hash TEXT := 'sha256:' || repeat('b', 64);
    lease_hash TEXT;
    claim_key TEXT := 'sha256:' || repeat('d', 64);
    event_hash TEXT := 'sha256:' || repeat('e', 64);
    result JSONB;
    heartbeat_result JSONB;
    claim_recheck JSONB;
BEGIN
    INSERT INTO public.research_lab_routing_experiments_v2 (
        experiment_hash, experiment_id, spec_doc,
        receipt_execution_mode, allow_live_credit_spend
    ) VALUES (
        experiment_hash, 'routing-v3-claim-contract',
        pg_catalog.jsonb_build_object('contract_version', 'test', 'credit_budget', pg_catalog.jsonb_build_object()),
        'fixture', FALSE
    );
    PERFORM public.research_lab_routing_request_execution_v2(
        request_hash, experiment_hash,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_execution_request.v2',
            'experiment_hash', experiment_hash
        )
    );
    result := public.research_lab_routing_claim_execution_requests_v2(
        'routing-v3-worker', 100, 30
    );
    SELECT request_doc->>'lease_hash' INTO lease_hash
      FROM pg_catalog.jsonb_array_elements(result->'requests') request_doc
     WHERE request_doc->>'request_hash' = request_hash;
    IF lease_hash IS NULL THEN
        RAISE EXCEPTION 'v3 contract did not receive its queue lease: %', result;
    END IF;
    result := public.research_lab_routing_claim_execution_v3(
        request_hash, lease_hash, 1, 'routing-v3-worker', claim_key, 1,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim.v3',
            'request_hash', request_hash, 'lease_hash', lease_hash,
            'lease_generation', 1, 'worker_ref', 'routing-v3-worker'
        ),
        event_hash,
        pg_catalog.jsonb_build_object('event', 'claimed-v3')
    );
    IF result->>'claim_key' IS DISTINCT FROM claim_key
       OR (result->>'claim_generation')::BIGINT <> 1
       OR result->>'lease_hash' IS DISTINCT FROM lease_hash
    THEN
        RAISE EXCEPTION 'v3 claim result is not bound to the queue lease: %', result;
    END IF;
    heartbeat_result := public.research_lab_routing_renew_claim_v3(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object('fixture', 'heartbeat-v3-contract')
        ),
        experiment_hash, claim_key, 1, 30,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim_heartbeat.v3',
            'worker_ref', 'routing-v3-worker',
            'index', 1
        )
    );
    IF heartbeat_result->'renewed' IS DISTINCT FROM 'true'::JSONB THEN
        RAISE EXCEPTION 'v3 claim heartbeat did not renew: %', heartbeat_result;
    END IF;
    PERFORM pg_catalog.pg_sleep(1.1);
    claim_recheck := public.research_lab_routing_claim_execution_v3(
        request_hash, lease_hash, 1, 'routing-v3-worker', claim_key, 1,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim.v3',
            'request_hash', request_hash, 'lease_hash', lease_hash,
            'lease_generation', 1, 'worker_ref', 'routing-v3-worker'
        ),
        event_hash,
        pg_catalog.jsonb_build_object('event', 'claimed-v3')
    );
    IF claim_recheck->'claimed' IS DISTINCT FROM 'true'::JSONB
       OR claim_recheck->'recoverable' IS NOT NULL
       OR claim_recheck->>'lease_expires_at'
            IS DISTINCT FROM heartbeat_result->>'lease_expires_at'
    THEN
        RAISE EXCEPTION 'heartbeat-extended v3 claim was treated as stale: %', claim_recheck;
    END IF;
    IF EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'research_lab_routing_experiment_claims_v3'
           AND column_name = 'claim_capability_commitment'
    ) THEN
        RAISE EXCEPTION 'v3 claim table persisted a bearer commitment';
    END IF;
    result := public.research_lab_routing_append_fenced_event_v3(
        'sha256:' || repeat('f', 64), experiment_hash, 'run_started',
        claim_key, 1, pg_catalog.jsonb_build_object('event', 'started-v3')
    );
    IF (result->>'idempotent')::BOOLEAN IS NOT FALSE THEN
        RAISE EXCEPTION 'v3 fenced event was not appended: %', result;
    END IF;
END;
$claim_v3_contract$;

-- Every bearer-free V3 RPC is service-only.  The corresponding bearer V2
-- authority is retained for historical replay by its owner, but service_role
-- must not be able to invoke it after the cutover.
DO $function_acl_cutover$
DECLARE
    target REGPROCEDURE;
BEGIN
    FOREACH target IN ARRAY ARRAY[
        'public.research_lab_routing_assert_claim_v3(text,text,bigint)'::REGPROCEDURE,
        'public.research_lab_routing_claim_experiment_v3(text,text,text,bigint,text,text,integer,jsonb,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_recover_claim_v3(text,text,text,jsonb,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_renew_claim_v3(text,text,text,bigint,integer,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_close_claim_v3(text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_fenced_event_v3(text,text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_assert_provider_receipt_chain_v3(text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,bigint,bigint,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_provider_attempt_v3(text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,bigint,text,text,bigint,bigint,text,text,bigint,text,text,text,text,text,text,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_decision_receipt_v3(text,text,text,text,text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_evaluation_v3(text,text,text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_reserve_budget_v3(text,text,text,text,text,bigint,bigint,integer,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_settle_budget_v3(text,text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_mark_budget_uncertain_v3(text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_recover_budget_v3(text,text,text,bigint,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_list_expired_budget_reservations_v3(text,text,bigint)'::REGPROCEDURE,
        'public.research_lab_routing_list_unresolved_budget_reservations_v3(text,text,bigint)'::REGPROCEDURE,
        'public.research_lab_routing_assert_promotion_receipt_chain_v3(text)'::REGPROCEDURE,
        'public.research_lab_routing_promote_v3(text,text,text,text,text,jsonb,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_claim_execution_v3(text,text,bigint,text,text,integer,jsonb,text,jsonb)'::REGPROCEDURE
    ] LOOP
        IF NOT pg_catalog.has_function_privilege('service_role', target, 'EXECUTE')
           OR pg_catalog.has_function_privilege('anon', target, 'EXECUTE')
           OR pg_catalog.has_function_privilege('authenticated', target, 'EXECUTE')
        THEN
            RAISE EXCEPTION 'v3 function ACL is not service-only: %', target;
        END IF;
    END LOOP;

    IF pg_catalog.has_function_privilege(
           'service_role',
           'public.research_lab_routing_assert_promotion_reconciliation_v3(text,text,text,text,jsonb)'::REGPROCEDURE,
           'EXECUTE'
       )
       OR pg_catalog.has_function_privilege(
           'anon',
           'public.research_lab_routing_assert_promotion_reconciliation_v3(text,text,text,text,jsonb)'::REGPROCEDURE,
           'EXECUTE'
       )
       OR pg_catalog.has_function_privilege(
           'authenticated',
           'public.research_lab_routing_assert_promotion_reconciliation_v3(text,text,text,text,jsonb)'::REGPROCEDURE,
           'EXECUTE'
       )
    THEN
        RAISE EXCEPTION 'promotion reconciliation helper is directly callable';
    END IF;

    FOREACH target IN ARRAY ARRAY[
        'public.research_lab_routing_claim_capability_commitment_v2(text)'::REGPROCEDURE,
        'public.research_lab_routing_assert_claim_v2(text,text,bigint,text)'::REGPROCEDURE,
        'public.research_lab_routing_claim_experiment_v2(text,text,text,text,integer,jsonb,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_renew_claim_v2(text,text,text,bigint,text,integer,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_close_claim_v2(text,text,text,bigint,text,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_fenced_event_v2(text,text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_recover_claim_v2(text,text,text,jsonb,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_provider_attempt_v2(text,text,text,text,text,text,text,text,text,text,text,text,text,text,bigint,text,text,text,bigint,bigint,text,text,bigint,text,text,text,text,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_decision_receipt_v2(text,text,text,text,text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_append_evaluation_v2(text,text,text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_reserve_budget_v2(text,text,text,text,text,bigint,text,bigint,integer,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_settle_budget_v2(text,text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_mark_budget_uncertain_v2(text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_recover_budget_v2(text,text,text,bigint,text,jsonb)'::REGPROCEDURE,
        'public.research_lab_routing_list_expired_budget_reservations_v2(text,text,bigint,text)'::REGPROCEDURE,
        'public.research_lab_routing_list_unresolved_budget_reservations_v2(text,text,bigint,text)'::REGPROCEDURE,
        'public.research_lab_routing_promote_v2(text,text,text,text,text,jsonb,text,jsonb)'::REGPROCEDURE
    ] LOOP
        IF pg_catalog.has_function_privilege('service_role', target, 'EXECUTE')
           OR pg_catalog.has_function_privilege('anon', target, 'EXECUTE')
           OR pg_catalog.has_function_privilege('authenticated', target, 'EXECUTE')
        THEN
            RAISE EXCEPTION 'retired bearer v2 function remains callable: %', target;
        END IF;
    END LOOP;
END;
$function_acl_cutover$;

DO $truncate_privilege$
BEGIN
    IF has_table_privilege(
        'service_role', 'public.research_lab_routing_budget_events_v2', 'TRUNCATE'
    ) THEN
        RAISE EXCEPTION 'service role can truncate the routing ledger';
    END IF;
END;
$truncate_privilege$;

SET LOCAL ROLE service_role;
DO $acl$
BEGIN
    BEGIN
        INSERT INTO public.research_lab_routing_experiments_v2 (
            experiment_hash, experiment_id, spec_doc,
            receipt_execution_mode, allow_live_credit_spend
        ) VALUES (
            'sha256:' || repeat('f', 64), 'direct-write',
            pg_catalog.jsonb_build_object(
                'contract_version', 'test', 'credit_budget', pg_catalog.jsonb_build_object()
            ),
            'fixture', FALSE
        );
        RAISE EXCEPTION 'service role direct write unexpectedly succeeded';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
END;
$acl$;
RESET ROLE;

ROLLBACK;
